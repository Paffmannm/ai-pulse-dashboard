import asyncio
import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).parent

import feedparser
import httpx
from anthropic import Anthropic
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

# In-memory cache
article_cache: dict = {}
cache_meta = {"last_updated": 0.0, "is_refreshing": False}

CACHE_TTL = 900  # 15 minutes

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


def parse_date(raw: str) -> str:
    """Try to return an ISO date string."""
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
                excerpt = entry.get("summary", "") or ""
                # Strip HTML tags from excerpt
                import re
                excerpt = re.sub(r"<[^>]+>", "", excerpt)[:600]
                articles.append({
                    "id": make_id(entry.link),
                    "title": entry.title,
                    "author": entry.get("author", source),
                    "source": source,
                    "url": entry.link,
                    "published": parse_date(entry.get("published", "")),
                    "excerpt": excerpt,
                    "summary": None,
                    "type": "news",
                    "query": query,
                })
            return articles
    except Exception as e:
        print(f"[Google News] '{query}': {e}")
        return []


async def fetch_reddit(query: str) -> list:
    encoded = query.replace(" ", "+")
    url = f"https://www.reddit.com/search.json?q={encoded}&sort=new&limit=15&type=link&t=week"
    headers = {"User-Agent": "AIPulseDashboard/1.0"}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, headers=headers)
            data = resp.json()
            articles = []
            for post in data["data"]["children"]:
                p = post["data"]
                link = p.get("url", "")
                if not link or "reddit.com" in link:
                    continue
                articles.append({
                    "id": make_id(link),
                    "title": p["title"],
                    "author": f"u/{p['author']}",
                    "source": f"r/{p['subreddit']}",
                    "url": link,
                    "published": datetime.fromtimestamp(p["created_utc"], tz=timezone.utc).isoformat(),
                    "excerpt": (p.get("selftext") or "")[:500],
                    "summary": None,
                    "type": "reddit",
                    "query": query,
                })
            return articles[:6]
    except Exception as e:
        print(f"[Reddit] '{query}': {e}")
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
                articles.append({
                    "id": make_id(hit["url"]),
                    "title": hit["title"],
                    "author": hit.get("author", "Unknown"),
                    "source": "Hacker News",
                    "url": hit["url"],
                    "published": hit.get("created_at", ""),
                    "excerpt": "",
                    "summary": None,
                    "type": "hackernews",
                    "query": query,
                })
            return articles
    except Exception as e:
        print(f"[HackerNews] '{query}': {e}")
        return []


def summarize_articles(articles: list) -> list:
    """Batch summarize articles that don't yet have summaries using Claude Haiku."""
    needs = [a for a in articles if a.get("summary") is None]
    if not needs:
        return articles

    # Build batch prompt
    items = ""
    for i, a in enumerate(needs):
        items += f"\nARTICLE {i+1}:\nTitle: {a['title']}\nSource: {a['source']}\nExcerpt: {a.get('excerpt') or 'N/A'}\n"

    try:
        msg = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=3000,
            messages=[{
                "role": "user",
                "content": (
                    "For each article below, write a 2-sentence summary covering what the piece is about and why it matters. "
                    "Be specific. Format exactly as:\n"
                    "ARTICLE 1: [summary]\nARTICLE 2: [summary]\netc.\n\n"
                    + items
                )
            }]
        )
        text = msg.content[0].text.strip()

        import re
        # Parse responses like "ARTICLE 1: ..."
        summaries = {}
        for match in re.finditer(r"ARTICLE\s+(\d+):\s*(.+?)(?=ARTICLE\s+\d+:|$)", text, re.DOTALL):
            idx = int(match.group(1)) - 1
            summaries[idx] = match.group(2).strip()

        for i, a in enumerate(needs):
            a["summary"] = summaries.get(i) or a.get("excerpt") or "No summary available."

    except Exception as e:
        print(f"[Summarizer] {e}")
        for a in needs:
            a["summary"] = a.get("excerpt") or "Summary unavailable."

    return articles


async def refresh_articles():
    cache_meta["is_refreshing"] = True
    try:
        tasks = []
        for query in SEARCH_QUERIES:
            tasks.append(fetch_google_news(query))
            tasks.append(fetch_reddit(query))
            tasks.append(fetch_hackernews(query))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        fresh: list = []
        seen: set = set()
        for result in results:
            if isinstance(result, list):
                for a in result:
                    if a["id"] not in seen:
                        # Preserve existing summary from cache
                        if a["id"] in article_cache:
                            a["summary"] = article_cache[a["id"]].get("summary")
                        seen.add(a["id"])
                        fresh.append(a)

        fresh = summarize_articles(fresh)

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
            # First load — wait for data
            await refresh_articles()
        else:
            # Refresh in background, return stale data immediately
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
