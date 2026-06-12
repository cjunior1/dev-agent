"""Web fetch tool for documentation and API lookups."""

import re

import httpx
from langchain_core.tools import tool


def _strip_html(html: str) -> str:
    text = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@tool
async def web_fetch(url: str, max_chars: int = 6000) -> str:
    """Fetch a web page and return its text content (HTML stripped).
    Useful for reading documentation, READMEs, or API references.

    Args:
        url: URL to fetch (must start with http:// or https://).
        max_chars: Maximum characters to return (default 6000).
    """
    if not url.startswith(("http://", "https://")):
        return "ERROR: URL must start with http:// or https://"

    headers = {"User-Agent": "dev-agent/0.1 (documentation lookup)"}
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "html" in content_type:
                text = _strip_html(resp.text)
            else:
                text = resp.text
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n...[content truncated at {max_chars} chars]"
            return text
    except httpx.HTTPStatusError as e:
        return f"ERROR: HTTP {e.response.status_code} for {url}"
    except Exception as e:
        return f"ERROR: {e}"
