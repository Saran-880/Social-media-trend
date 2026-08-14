import os
import json
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import requests
from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchResults,YouTubeSearchTool


load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


mcp = FastMCP("Social Media Trend Analyzer")

WIKIPEDIA_PAGEVIEWS_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/top"
)

WIKIPEDIA_HEADERS = {
    "User-Agent": "SocialMediaTrendAnalyzer/1.0"
}

youtube = YouTubeSearchTool()


duckduckgo = DuckDuckGoSearchResults(
    output_format="list",
    num_results=5
)

@mcp.tool
def query_agent(query: str) -> str:
    """Extract the main domain from the user's query."""

    prompt = f"""
Extract the main domain from the following user query.

User Query:
{query}

Return ONLY the domain.

Examples:
sports
technology
politics
entertainment
business
science
AI
finance
gaming

Do not explain your answer.
"""

    response = llm.invoke(prompt)

    return response.content.strip().lower()


@mcp.tool
def search_bluesky(domain: str, days: int = 7) -> list:
    """
    Search Bluesky posts for a domain.

    days controls the time window considered for recent posts.
    """

    url = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"

    params = {
        "q": domain,
        "limit": 20
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as e:
        return [{
            "error": f"Bluesky search error: {str(e)}"
        }]

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    results = []

    for post in data.get("posts", []):

        record = post.get("record", {})
        created_at = record.get("createdAt")

        if created_at:
            try:
                post_dt = datetime.fromisoformat(
                    created_at.replace("Z", "+00:00")
                )
            except ValueError:
                post_dt = None
        else:
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
    """
    Search Hacker News stories.

    days is converted into the Algolia created_at timestamp filter.
    """

    url = "https://hn.algolia.com/api/v1/search"

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    params = {
        "query": domain,
        "tags": "story",
        "numericFilters": (
            f"created_at_i>={int(cutoff.timestamp())}"
        ),
        "hitsPerPage": 20
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as e:
        return [{
            "error": f"Hacker News search error: {str(e)}"
        }]

    results = []

    for item in data.get("hits", []):

        results.append({
            "title": item.get("title"),
            "url": item.get("url"),
            "created_at": item.get("created_at"),
            "points": item.get("points", 0),
            "comments": item.get("num_comments", 0)
        })

    return results


@mcp.tool
def search_news(domain: str, days: int = 7) -> list:
    """
    Search recent NewsAPI articles.

    days is passed through as the NewsAPI from/to date range.
    """

    url = "https://newsapi.org/v2/everything"

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    params = {
        "q": domain,
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 20,
        "apiKey": NEWS_API_KEY
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as e:
        return [{
            "error": f"NewsAPI request error: {str(e)}"
        }]

    if data.get("status") != "ok":
        return [{
            "error": data.get(
                "message",
                "NewsAPI request failed."
            )
        }]

    articles = []

    for article in data.get("articles", []):

        articles.append({
            "title": article.get("title"),
            "description": article.get("description"),
            "source": article.get("source", {}).get("name"),
            "url": article.get("url"),
            "published_at": article.get("publishedAt")
        })

    return articles


@mcp.tool
def search_wikipedia_trends(
    domain: str,
    days: int = 7
) -> list:
    """
    Get highly viewed Wikipedia articles for the requested
    number of days.

    Wikipedia pageviews are used as one signal for trend
    selection, not as the only decision signal.

    Note: this intentionally returns the overall top-viewed
    articles rather than filtering by `domain` server-side.
    `domain` is accepted for interface consistency with the
    other search_* tools, but relevance filtering against the
    domain is left to trend_selection's LLM comparison step.
    """

    totals = defaultdict(int)

    today = datetime.now(timezone.utc) - timedelta(days=1)

    for i in range(days):

        date = today - timedelta(days=i)

        url = (
            f"{WIKIPEDIA_PAGEVIEWS_URL}/"
            f"en.wikipedia.org/"
            f"all-access/"
            f"{date.year}/"
            f"{date.month:02d}/"
            f"{date.day:02d}"
        )

        try:
            response = requests.get(
                url,
                headers=WIKIPEDIA_HEADERS,
                timeout=15
            )

            if response.status_code != 200:
                continue

            data = response.json()

            articles = data["items"][0]["articles"]

        except (
            requests.RequestException,
            ValueError,
            KeyError,
            IndexError,
            TypeError
        ):
            continue

        for article in articles:

            title = article.get("article", "")

            if title in {
                "Main_Page",
                "Special:Search",
                "XXX",
                "Wikipedia"
            }:
                continue

            totals[title] += article.get("views", 0)

    results = [
        {
            "topic": title.replace("_", " "),
            "weekly_views": views
        }
        for title, views in totals.items()
    ]

    results.sort(
        key=lambda x: x["weekly_views"],
        reverse=True
    )

    return results[:25]


@mcp.tool
def trend_selection(
    query: str,
    domain: str,
    bluesky_results: list,
    hacker_news_results: list,
    news_results: list,
    wikipedia_results: list
) -> str:
    """
    Select one strongest overall trend using all four sources.

    Only a compact subset is placed into the LLM prompt so that
    raw API results do not create an oversized request.
    """

    compact_results = {
        "Bluesky": bluesky_results[:8],
        "Hacker News": hacker_news_results[:8],
        "NewsAPI": news_results[:8],
        "Wikipedia": wikipedia_results[:15]
    }

    prompt = f"""
You are a trend analysis agent.

User query:
{query}

Domain:
{domain}

The following results were collected from four external
sources during the requested recent time window.

SOURCE RESULTS:

{json.dumps(
    compact_results,
    indent=2,
    ensure_ascii=False
)}

Analyze ALL FOUR sources together.

Compare:

- Social media discussions from Bluesky
- Technology/community discussions from Hacker News
- Recent news coverage from NewsAPI
- Public attention from Wikipedia pageviews

Wikipedia pageviews are an important signal, but they are
ONLY one signal.

Do NOT select a topic only because it has high Wikipedia
pageviews.

Look for a topic with strong evidence across multiple
independent sources.

Select ONE topic that represents the strongest overall trend.

Return ONLY the topic name.
"""

    response = llm.invoke(prompt)

    return response.content.strip()


@mcp.tool
def research_web(selected_trend: str, days: int = 7) -> list:
    """
    Find recent web/news results for the selected trend.
    """

    query = (
        f"{selected_trend} latest developments "
        f"news last {days} days"
    )

    try:
        return duckduckgo.invoke(query)

    except Exception as e:
        return [{
            "error": f"DuckDuckGo error: {str(e)}"
        }]


@mcp.tool
def research_wikipedia(selected_trend: str) -> list:
    """
    Research the selected trend using the Wikipedia API.

    This intentionally uses the MediaWiki API directly instead
    of WikipediaQueryRun because the wrapper was producing the
    JSONDecodeError in the notebook.
    """

    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "list": "search",
        "srsearch": selected_trend,
        "format": "json",
        "srlimit": 3
    }

    headers = {
        "User-Agent": "SocialMediaTrendAnalyzer/1.0"
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        results = []

        for item in data.get(
            "query",
            {}
        ).get(
            "search",
            []
        ):

            title = item.get("title", "")

            results.append({
                "title": title,
                "snippet": item.get("snippet"),
                "page_id": item.get("pageid"),
                "url": (
                    "https://en.wikipedia.org/wiki/"
                    + title.replace(" ", "_")
                )
            })

        return results

    except Exception as e:
        return [{
            "error": f"Wikipedia research error: {str(e)}"
        }]



@mcp.tool
def research_youtube(selected_trend: str) -> list:
    """
    Search YouTube for relevant videos about the selected trend.
    """

    query = f"{selected_trend} latest developments,2"

    try:
        results = youtube.invoke(query)

        return results

    except Exception as e:
        return [{
            "error": f"YouTube error: {str(e)}"
        }]



@mcp.tool
def final_report(
    query: str,
    selected_trend: str,
    web_results: list,
    wikipedia_results: list,
    youtube_results: list
) -> str:
    """
    Generate the final 15-20 line trend intelligence report.

    The report contains:
    - Trend
    - Why it's trending
    - At least 10 lines of recent developments
    - One news article with URL
    - One YouTube video with URL
    - Short conclusion
    """

    prompt = f"""
You are the final trend research analyst.

User query:
{query}

Selected trending topic:
{selected_trend}

WEB SEARCH RESULTS:
{json.dumps(
    web_results,
    indent=2,
    ensure_ascii=False
)}

WIKIPEDIA RESULTS:
{json.dumps(
    wikipedia_results,
    indent=2,
    ensure_ascii=False
)}

YOUTUBE RESULTS:
{json.dumps(
    youtube_results,
    indent=2,
    ensure_ascii=False
)}

Create a detailed and informative trend report.

Use exactly this structure:

**Trend**

Explain clearly what the selected trend is and what
is happening around it.

**Why It's Trending**

Explain why this topic is receiving significant attention
right now.

**Recent Developments**

Provide AT LEAST 10 meaningful lines describing recent
developments related to the selected trend.

Do not simply list short headlines.

Explain important events, announcements, decisions,
debates, changes, or other recent activity found in
the research.

Use the available sources to build a coherent picture
of what has happened recently.

**Evidence from Sources**

Provide exactly TWO source items:

1. **News Article**

Select ONE relevant recent article from the web results.

Include:
- Article title
- Publication/source
- A short explanation of what the article reports
- The ACTUAL article URL from the supplied results

2. **YouTube Video**

Select ONE relevant YouTube video from the supplied results.

Include:
- Video title if available
- The ACTUAL YouTube URL from the supplied results

**Conclusion**

Give a short and crisp conclusion in 2–3 sentences.

Rules:

- Use ONLY information contained in the supplied research.
- Do NOT use outside knowledge.
- Do NOT invent facts, dates, statistics, events or claims.
- Do NOT invent URLs.
- Use only URLs that actually appear in the supplied results.
- Do NOT include Political Implications.
- Do NOT include Future Outlook.
- Do NOT include a separate Sources list beyond the
  requested News Article and YouTube Video.
- Recent Developments must contain substantial information
  and be AT LEAST 10 meaningful lines.
- Avoid repeating the same fact multiple times.
- Keep the report informative and readable.
"""

    response = llm.invoke(prompt)

    return response.content.strip()



if __name__ == "__main__":
    mcp.run()