# DuckDuckGo library web search wrapper
"""
Sub-question -> DuckDuckGo search -> Top 5 results (title + snippet + URL)
    -> LLM reads results -> Extracts relevant facts
    -> Stores {question, raw_results, key_facts} in research_data
    -> Stores URLs in sources
"""
import time
from ddgs import DDGS


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Performs a DuckDuckGo web search and returns structured results

    Why DuckDuckGo:
        - Zero API key required
        - No rate-limit billing
        - Returns real-time web results
        - The duckduckgo_search library handles the HTTP requests

    Why we format into dicts:
        - Consistent structure the LLM can reliably read
        - Each result has: title, snippet (body), url
        - Makes downstream parsing deterministic

    Args:
        query: The search query string
        max_results: How many results to fetch (5 is sweet spot -> enough coverage, not too much noise)

    Returns:
        List of dicts: [{"title": ..., "snippet": ..., "url": ...}]
        Returns empty list on any failure (agent handles gracefully)
    """
    try:
        results = []

        # DDGS() is the DuckDuckGo Search context manager
        # text() performs a standard web text search
        with DDGS() as ddgs:
            raw_results = ddgs.text(
                query,
                max_results=max_results,
                # safesearch="moderate" keeps results clean
                safesearch="moderate"
            )

            for r in raw_results:
                # DuckDuckGo returns: title, body (snippet), href (url)
                # We rename 'body' → 'snippet' and 'href' → 'url'
                # for clarity and consistency across the codebase
                results.append({
                    "title":   r.get("title", "No title"),
                    "snippet": r.get("body", "No content available"),
                    "url":     r.get("href", "No URL")
                })

        return results

    except Exception as e:
        # Never let a search failure crash the entire pipeline.
        # Return empty list — Researcher agent handles this case.
        print(f"[SEARCH TOOL] Search failed for query '{query}': {e}")
        return []


def format_results_for_llm(results: list[dict]) -> str:
    """
    Converts the list of search result dicts into a single formatted
    string that the LLM can read and reason about.

    Why format as a string:
        LLMs receive text, not Python objects. We need to render
        the results in a clear, scannable format so the LLM can
        identify which parts are relevant to the sub-question.

    Example output:
        SOURCE 1: "Quantum Computing Reaches New Milestone" (example.com)
        Quantum researchers at IBM have demonstrated...

        SOURCE 2: "The State of Quantum Hardware in 2025" (nature.com)
        A new review paper outlines the current landscape...

    Args:
        results: List of result dicts from web_search()

    Returns:
        A formatted multi-line string ready to inject into a prompt
    """
    if not results:
        return "No search results found."

    formatted = []
    for i, result in enumerate(results, 1):
        formatted.append(
            f'SOURCE {i}: "{result["title"]}" ({result["url"]})\n'
            f'{result["snippet"]}'
        )

    # Join with double newline so each source is visually separated
    return "\n\n".join(formatted)
