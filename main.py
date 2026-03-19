import asyncio
import hashlib
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import feedparser
import httpx
from anthropic import Anthropic
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

article_cache: dict = {}
cache_meta = {"last_updated": 0.0, "is_refreshing": False}
CACHE_TTL = 900

SEARCH_QUERIES = [
    "AI avatars",
    "AI companions",
    "AI girlfriend app",
    "virtual AI companion",
    "AI character companion",
    "digital human avatar",
    "AI companionship",
]


def make_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:10]


def get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def parse_date(raw: str) -> str:
    if not raw:
        return ""
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(raw).isoformat()
    except Exception:
        return raw


async def fetch_google_news(query: str) -> list:
    encoded = query.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            feed = feedparser.parse(resp.text)
            articles = []
            for entry in feed.entries[:8]:
                source = (entry.get("source") or {}).get("title", "News")
                excerpt = re.sub(r"<[^>]+>", "", entry.get("summary", "") or "")[:600]
                # Try to resolve the actual article URL for favicon
                article_url = entry.link
                articles.append({
                    "id": make_id(entry.link),
                    "title": entry.title,
                    "author": entry.get("author", source),
                    "source": source,
                    "url": article_url,
                    "domain": get_domain(article_url),
                    "published": parse_date(entry.get("published", "")),
                    "excerpt": excerpt,
                    "summary": None,
                    "tweet": None,
                    "type": "news",
                    "query": query,
                })
            return articles
    except Exception as e:
        print(f"[Google News] '{query}': {e}")
        return []


async def fetch_hackernews(query: str) -> list:
    encoded = query.replace(" ", "+")
    url = f"https://hn.algolia.com/api/v1/search?query={encoded}&tags=story&hitsPerPage=10"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            data = resp.json()
            articles = []
            for hit in data["hits"][:6]:
                if not hit.get("url"):
                    continue
                article_url = hit["url"]
                articles.append({
                    "id": make_id(article_url),
                    "title": hit["title"],
                    "author": hit.get("author", "Unknown"),
                    "source": "Hacker News",
                    "url": article_url,
                    "domain": get_domain(article_url),
                    "published": hit.get("created_at", ""),
                    "excerpt": "",
                    "summary": None,
                    "tweet": None,
                    "type": "hackernews",
                    "query": query,
                })
            return articles
    except Exception as e:
        print(f"[HackerNews] '{query}': {e}")
        return []


def generate_summaries_and_tweets(articles: list) -> list:
    """Batch generate summaries AND tweet-style posts for new articles."""
    needs = [a for a in articles if a.get("summary") is None]
    if not needs:
        return articles

    items = ""
    for i, a in enumerate(needs):
        items += f"\nARTICLE {i+1}:\nTitle: {a['title']}\nSource: {a['source']}\nExcerpt: {a.get('excerpt') or 'N/A'}\n"

    try:
        msg = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=5000,
            messages=[{
                "role": "user",
                "content": (
                    "For each article, produce TWO things:\n"
                    "1. A 2-sentence summary of what the piece covers and why it matters.\n"
                    "2. A tweet (max 240 chars, punchy, no hashtags, written as a media observer reacting to the story).\n\n"
                    "Format EXACTLY as:\n"
                    "ARTICLE 1:\nSUMMARY: [summary]\nTWEET: [tweet]\n\n"
                    "ARTICLE 2:\nSUMMARY: [summary]\nTWEET: [tweet]\n\netc.\n\n"
                    + items
                )
            }]
        )
        text = msg.content[0].text.strip()

        # Parse each article block
        blocks = re.split(r"ARTICLE\s+\d+:", text)
        for i, block in enumerate(blocks[1:]):  # skip first empty split
            summary_match = re.search(r"SUMMARY:\s*(.+?)(?=TWEET:|$)", block, re.DOTALL)
            tweet_match = re.search(r"TWEET:\s*(.+?)$", block, re.DOTALL)
            if i < len(needs):
                needs[i]["summary"] = summary_match.group(1).strip() if summary_match else needs[i].get("excerpt", "No summary.")
                needs[i]["tweet"] = tweet_match.group(1).strip() if tweet_match else needs[i]["title"]

    except Exception as e:
        print(f"[Claude] {e}")
        for a in needs:
            a["summary"] = a.get("excerpt") or "Summary unavailable."
            a["tweet"] = a["title"][:240]

    return articles


async def refresh_articles():
    cache_meta["is_refreshing"] = True
    try:
        tasks = []
        for query in SEARCH_QUERIES:
            tasks.append(fetch_google_news(query))
            tasks.append(fetch_hackernews(query))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        fresh: list = []
        seen: set = set()
        for result in results:
            if isinstance(result, list):
                for a in result:
                    if a["id"] not in seen:
                        if a["id"] in article_cache:
                            cached = article_cache[a["id"]]
                            a["summary"] = cached.get("summary")
                            a["tweet"] = cached.get("tweet")
                        seen.add(a["id"])
                        fresh.append(a)

        fresh = generate_summaries_and_tweets(fresh)

        for a in fresh:
            article_cache[a["id"]] = a

        cache_meta["last_updated"] = time.time()
    finally:
        cache_meta["is_refreshing"] = False


@app.get("/api/articles")
async def get_articles(background_tasks: BackgroundTasks, force: bool = False):
    age = time.time() - cache_meta["last_updated"]
    stale = age > CACHE_TTL or force

    if stale and not cache_meta["is_refreshing"]:
        if not article_cache:
            await refresh_articles()
        else:
            background_tasks.add_task(refresh_articles)

    articles = sorted(article_cache.values(), key=lambda x: x.get("published", ""), reverse=True)
    return {
        "articles": articles,
        "last_updated": cache_meta["last_updated"],
        "is_refreshing": cache_meta["is_refreshing"],
        "count": len(articles),
    }


@app.get("/api/refresh")
async def force_refresh():
    await refresh_articles()
    return {"status": "ok", "count": len(article_cache)}


@app.get("/", response_class=HTMLResponse)
async def root():
    with open(BASE_DIR / "static" / "index.html") as f:
        return f.read()


app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
