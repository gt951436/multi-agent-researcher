import re
import time
import ipaddress
import urllib.parse
from typing import Optional, List, Dict, Any, Literal
import httpx
import lxml.html
from ddgs import DDGS

# User agent for web requests
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"


def is_safe_url(url: str) -> bool:
    """
    Validates that a URL is syntactically sound and does not target
    private IP addresses, loopbacks, or dangerous schemes (SSRF protection).
    """
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False

        hostname = parsed.hostname
        if not hostname:
            return False

        # Block localhost and common local network aliases
        if hostname.lower() in ("localhost", "127.0.0.1", "0.0.0.0", "::1", "local"):
            return False

        # Check if the hostname is a private or reserved IP address
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
        except ValueError:
            # It's a standard domain name, which is safe to query externally
            pass

        return True
    except Exception:
        return False


def fetch_page_content(url: str, timeout: float = 8.0, max_chars: int = 3500) -> Dict[str, Any]:
    """
    Retrieves full article text from a URL using httpx and lxml.html.
    Strips scripts, styles, navigation, headers, footers, and boilerplate.
    Falls back gracefully if the site blocks scrapers or times out.

    Returns:
        dict with keys:
            - status: "success" | "fallback" | "blocked_unsafe"
            - text: Clean readable article content or empty string
            - content_type: "full_text" | "snippet"
            - reason: Optional failure reason
    """
    if not is_safe_url(url):
        return {
            "status": "blocked_unsafe",
            "text": "",
            "content_type": "snippet",
            "reason": "URL rejected by SSRF security filter."
        }

    try:
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        with httpx.Client(timeout=timeout, follow_redirects=True, max_redirects=3) as client:
            response = client.get(url, headers=headers)

            if response.status_code != 200:
                return {
                    "status": "fallback",
                    "text": "",
                    "content_type": "snippet",
                    "reason": f"HTTP {response.status_code}"
                }

            content_type_header = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type_header and "text/plain" not in content_type_header:
                return {
                    "status": "fallback",
                    "text": "",
                    "content_type": "snippet",
                    "reason": f"Unsupported MIME type: {content_type_header}"
                }

            raw_html = response.text
            if not raw_html or len(raw_html.strip()) < 50:
                return {
                    "status": "fallback",
                    "text": "",
                    "content_type": "snippet",
                    "reason": "Empty HTML body"
                }

            # Parse with lxml
            doc = lxml.html.fromstring(raw_html)

            # Strip non-content and noisy elements
            for el in doc.xpath("//script|//style|//noscript|//nav|//header|//footer|//aside|//svg|//iframe"):
                el.drop_tree()

            clean_text = doc.text_content()
            # Normalize whitespace
            clean_text = re.sub(r"\s+", " ", clean_text).strip()

            if len(clean_text) < 150:
                # Content too sparse (e.g. cookie wall / JS-rendered app)
                return {
                    "status": "fallback",
                    "text": "",
                    "content_type": "snippet",
                    "reason": "Extracted text was too sparse (likely JS-rendered or paywalled)."
                }

            # Truncate to maximum characters window to prevent token bloat
            truncated = clean_text[:max_chars]
            return {
                "status": "success",
                "text": truncated,
                "content_type": "full_text",
                "char_count": len(truncated)
            }

    except httpx.TimeoutException:
        return {
            "status": "fallback",
            "text": "",
            "content_type": "snippet",
            "reason": "Request timed out."
        }
    except Exception as e:
        return {
            "status": "fallback",
            "text": "",
            "content_type": "snippet",
            "reason": str(e)
        }


def search_with_status(query: str, max_results: int = 5, max_attempts: int = 2) -> Dict[str, Any]:
    """
    Performs DuckDuckGo search with explicit error classification.
    Distinguishes legitimate zero-result queries from rate limits and network errors.
    Includes bounded retry for transient throttling.
    """
    attempts = 0

    while attempts < max_attempts:
        attempts += 1
        try:
            results = []
            with DDGS() as ddgs:
                raw_results = ddgs.text(
                    query,
                    max_results=max_results,
                    safesearch="moderate"
                )

                for r in raw_results:
                    href = r.get("href", "No URL")
                    domain = urllib.parse.urlparse(href).netloc if href != "No URL" else "Unknown"
                    results.append({
                        "title":   r.get("title", "No title"),
                        "snippet": r.get("body", "No content available"),
                        "url":     href,
                        "domain":  domain
                    })

            if not results:
                return {
                    "query": query,
                    "status": "NO_RESULTS",
                    "results": [],
                    "error_message": None
                }

            return {
                "query": query,
                "status": "SUCCESS",
                "results": results,
                "error_message": None
            }

        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "ratelimit" in err_str:
                if attempts < max_attempts:
                    print(f"[SEARCH TOOL] Throttled on query '{query}'. Retrying after 2s...")
                    time.sleep(2)
                    continue
                return {
                    "query": query,
                    "status": "RATE_LIMITED",
                    "results": [],
                    "error_message": f"Search provider rate limit encountered: {e}"
                }
            elif isinstance(e, TimeoutError) or "timeout" in err_str or "timed out" in err_str or "timeout" in type(e).__name__.lower():
                return {
                    "query": query,
                    "status": "TIMEOUT",
                    "results": [],
                    "error_message": f"Search provider timed out: {e}"
                }
            else:
                return {
                    "query": query,
                    "status": "ERROR",
                    "results": [],
                    "error_message": f"Search failed with error: {e}"
                }

    return {
        "query": query,
        "status": "ERROR",
        "results": [],
        "error_message": "Search retry attempts exhausted."
    }


def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Backward-compatible wrapper returning list of search result dicts.
    """
    outcome = search_with_status(query, max_results=max_results)
    return outcome.get("results", [])


def format_results_for_llm(results: List[Dict[str, Any]]) -> str:
    """
    Renders search evidence into an isolated XML structure with source IDs.
    Applies prompt injection defense: content is explicitly marked as untrusted data.
    """
    if not results:
        return "No search results found."

    blocks = []
    for i, res in enumerate(results, 1):
        sid = res.get("source_id", f"S{i}")
        url = res.get("url", "No URL")
        domain = res.get("domain", "Unknown")
        title = res.get("title", "No Title")
        content_type = res.get("content_type", "snippet")
        content = res.get("full_text") or res.get("snippet", "No content available")

        # Untrusted content delimiter boundary
        block = (
            f'<untrusted_source id="{sid}" url="{url}" domain="{domain}" type="{content_type}">\n'
            f'TITLE: {title}\n'
            f'EVIDENCE CONTENT:\n{content}\n'
            f'</untrusted_source>'
        )
        blocks.append(block)

    return "\n\n".join(blocks)

