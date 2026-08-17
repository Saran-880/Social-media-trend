from fastmcp import FastMCP
import os
import json
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchResults, YouTubeSearchTool
from langchain_groq import ChatGroq

load_dotenv()

mcp = FastMCP(name="Social Media Trend Analyzer")

llm = ChatGroq(model="openai/gpt-oss-20b")
youtube = YouTubeSearchTool()
duckduckgo = DuckDuckGoSearchResults(output_format="list", num_results=5)

WIKI_PAGEVIEWS_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/top"
WIKI_HEADERS = {"User-Agent": "SocialMediaTrendAnalyzer/1.0"}


@mcp.tool
def query_agent(query: str) -> str:
    """Extract the main domain (e.g. sports, tech, finance) from the user's query."""
    prompt = f"""Extract the main domain from this user query. Return ONLY the domain (e.g. sports, technology, politics, finance, gaming). No explanation.

Query: {query}"""
    return llm.invoke(prompt).content.strip().lower()


@mcp.tool
def search_bluesky(domain: str, days: int = 7) -> list:
    """Recent Bluesky posts for a domain, filtered to the last `days` days."""
    url = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"
    try:
        response = requests.get(url, params={"q": domain, "limit": 20}, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return [{"error": f"Bluesky search error: {str(e)}"}]

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    results = []
    for post in data.get("posts", []):
        record = post.get("record", {})
        created_at = record.get("createdAt")
        try:
            post_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00")) if created_at else None
        except ValueError:
            post_dt = None
        if post_dt is None or post_dt < cutoff:
            continue
        results.append({
            "text": record.get("text", ""),
            "created_at": created_at,
            "likes": post.get("likeCount", 0),
            "reposts": post.get("repostCount", 0),
            "replies": post.get("replyCount", 0)
        })
    return results


@mcp.tool
def search_hacker_news(domain: str, days: int = 7) -> list:
    """Hacker News stories for a domain from the last `days` days."""
    url = "https://hn.algolia.com/api/v1/search"
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    params = {
        "query": domain,
        "tags": "story",
        "numericFilters": f"created_at_i>={int(cutoff.timestamp())}",
        "hitsPerPage": 20
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return [{"error": f"Hacker News search error: {str(e)}"}]

    return [
        {
            "title": item.get("title"),
            "url": item.get("url"),
            "created_at": item.get("created_at"),
            "points": item.get("points", 0),
            "comments": item.get("num_comments", 0)
        }
        for item in data.get("hits", [])
    ]


@mcp.tool
def search_news(domain: str, days: int = 7) -> list:
    """Recent NewsAPI articles for a domain from the last `days` days."""
    url = "https://newsapi.org/v2/everything"
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    params = {
        "q": domain,
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 20
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return [{"error": f"NewsAPI request error: {str(e)}"}]

    if data.get("status") != "ok":
        return [{"error": data.get("message", "NewsAPI request failed.")}]

    return [
        {
            "title": a.get("title"),
            "description": a.get("description"),
            "source": a.get("source", {}).get("name"),
            "url": a.get("url"),
            "published_at": a.get("publishedAt")
        }
        for a in data.get("articles", [])
    ]


@mcp.tool
def search_wikipedia_trends(domain: str, days: int = 7) -> list:
    """Top-viewed Wikipedia articles over the last `days` days (a domain-agnostic attention signal)."""
    totals = defaultdict(int)
    today = datetime.now(timezone.utc) - timedelta(days=1)

    for i in range(days):
        date = today - timedelta(days=i)
        url = f"{WIKI_PAGEVIEWS_URL}/en.wikipedia.org/all-access/{date.year}/{date.month:02d}/{date.day:02d}"
        try:
            response = requests.get(url, headers=WIKI_HEADERS, timeout=15)
            if response.status_code != 200:
                continue
            articles = response.json()["items"][0]["articles"]
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
            continue

        for article in articles:
            title = article.get("article", "")
            if title in {"Main_Page", "Special:Search", "XXX", "Wikipedia"}:
                continue
            totals[title] += article.get("views", 0)

    results = [{"topic": t.replace("_", " "), "pageviews": v} for t, v in totals.items()]
    results.sort(key=lambda x: x["pageviews"], reverse=True)
    return results[:25]


@mcp.tool
def trend_selection(query: str, domain: str, bluesky_results: list, hacker_news_results: list,
                     news_results: list, wikipedia_results: list) -> str:
    """Pick the single strongest overall trend across all four sources."""
    compact_results = {
        "Bluesky": bluesky_results[:8],
        "Hacker News": hacker_news_results[:8],
        "NewsAPI": news_results[:8],
        "Wikipedia": wikipedia_results[:15]
    }
    prompt = f"""You are a trend analysis agent.

User query: {query}
Domain: {domain}

SOURCE RESULTS:
{json.dumps(compact_results, indent=2, ensure_ascii=False)}

Compare Bluesky (social), Hacker News (tech community), NewsAPI (news coverage), and Wikipedia (public attention).
Wikipedia pageviews are only ONE signal — do not pick a topic just because it has high pageviews.
Look for a topic with strong evidence across multiple independent sources.

Return ONLY the topic name."""
    return llm.invoke(prompt).content.strip()


@mcp.tool
def research_web(selected_trend: str, days: int = 7) -> list:
    """Recent web/news results for the selected trend."""
    query = f"{selected_trend} latest developments news last {days} days"
    try:
        return duckduckgo.invoke(query)
    except Exception as e:
        return [{"error": f"DuckDuckGo error: {str(e)}"}]


@mcp.tool
def research_wikipedia(selected_trend: str) -> list:
    """Top Wikipedia search results for the selected trend."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "search", "srsearch": selected_trend, "format": "json", "srlimit": 3}
    try:
        response = requests.get(url, params=params, headers=WIKI_HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()
        return [
            {
                "title": item.get("title", ""),
                "snippet": item.get("snippet"),
                "page_id": item.get("pageid"),
                "url": "https://en.wikipedia.org/wiki/" + item.get("title", "").replace(" ", "_")
            }
            for item in data.get("query", {}).get("search", [])
        ]
    except Exception as e:
        return [{"error": f"Wikipedia research error: {str(e)}"}]


import ast

@mcp.tool
def research_youtube(selected_trend: str) -> list:
    """Relevant YouTube videos about the selected trend."""
    try:
        raw = youtube.invoke(f"{selected_trend} latest developments,2")
        if isinstance(raw, str):
            try:
                urls = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                urls = [raw]
        else:
            urls = raw
        return [{"url": u} for u in urls]
    except Exception as e:
        return [{"error": f"YouTube error: {str(e)}"}]


@mcp.tool
def final_report(query: str, selected_trend: str, web_results: list,
                  wikipedia_results: list, youtube_results: list) -> str:
    """Generate the final trend intelligence report (trend, why it's trending, developments, evidence, conclusion)."""
    prompt = f"""You are the final trend research analyst.

User query: {query}
Selected trending topic: {selected_trend}

WEB SEARCH RESULTS:
{json.dumps(web_results, indent=2, ensure_ascii=False)}

WIKIPEDIA RESULTS:
{json.dumps(wikipedia_results, indent=2, ensure_ascii=False)}

YOUTUBE RESULTS:
{json.dumps(youtube_results, indent=2, ensure_ascii=False)}

Create a detailed report using exactly this structure:

**Trend** — what the trend is and what's happening around it.
**Why It's Trending** — why it's getting attention right now.
**Recent Developments** — AT LEAST 10 meaningful lines of real events, announcements, decisions, or debates. No short headlines.
**Evidence from Sources** — exactly two items:
1. News Article: title, source, one-line summary, ACTUAL URL from the results.
2. YouTube Video: title if available, ACTUAL URL from the results.
**Conclusion** — 2-3 crisp sentences.

Rules: use ONLY the supplied research, never invent facts/dates/URLs, no Political Implications or Future Outlook sections, no separate Sources list, avoid repetition."""
    return llm.invoke(prompt).content.strip()


if __name__ == "__main__":
    mcp.run()
