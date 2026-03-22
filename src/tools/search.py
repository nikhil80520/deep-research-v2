import os
import asyncio
from typing import List
from langchain_core.tools import tool
from tavily import AsyncTavilyClient


async def tavily_search_async(queries: List[str], max_results: int = 5) -> list[dict]:
    """Run multiple Tavily searches in parallel."""
    api_key = os.getenv("TAVILY_API_KEY", "")
    client = AsyncTavilyClient(api_key=api_key)
    tasks = [
        client.search(q, max_results=max_results, include_raw_content=True)
        for q in queries
    ]
    return await asyncio.gather(*tasks)


@tool(description="Search the web for current information. Provide a list of specific search queries.")
async def tavily_search(queries: List[str]) -> str:
    """Search the web using Tavily and return summarized results.

    Args:
        queries: List of search query strings (1-3 focused queries).

    Returns:
        Formatted string with search results and source URLs.
    """
    try:
        results = await tavily_search_async(queries, max_results=5)

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for response in results:
            for r in response.get("results", []):
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_results.append(r)

        if not unique_results:
            return "No search results found. Try different queries."

        # Format output
        formatted = "Search results:\n\n"
        for i, r in enumerate(unique_results, 1):
            title = r.get("title", "No title")
            url = r.get("url", "")
            content = r.get("content", "") or r.get("raw_content", "")
            formatted += f"--- SOURCE {i}: {title} ---\n"
            formatted += f"URL: {url}\n"
            formatted += f"CONTENT:\n{content[:2000]}\n"
            formatted += "-" * 60 + "\n\n"

        return formatted

    except Exception as e:
        return f"Search failed: {str(e)}"
