"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json  # noqa: F401
import os  # noqa: F401
import random
import re
import threading
import time  # noqa: F401
import xml.etree.ElementTree  # noqa: F401  (arXiv answers with Atom XML)
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import httpx  # noqa: F401
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


_last_arxiv_call = 0.0
_arxiv_lock = threading.Lock()


def _request(method, url, **kwargs):
    try:
        response = httpx.request(method, url, timeout=30, **kwargs)
    except httpx.TransportError as exc:
        raise RetryableError(str(exc)) from exc
    if response.status_code in (429, 500, 502, 503, 504):
        value = response.headers.get("Retry-After")
        try:
            retry_after = float(value) if value else None
        except ValueError:
            try:
                retry_after = max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                retry_after = None
        raise RetryableError(f"HTTP {response.status_code}", retry_after)
    response.raise_for_status()
    return response


def _records(items, *, prefer_ai_summary=False):
    records = []
    for item in items:
        paper = item.get("paper") or {}
        paper_id = paper.get("id")
        if not paper_id:
            continue
        records.append({
            "id": str(paper_id), "url": f"https://huggingface.co/papers/{paper_id}",
            "published": (paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
            "title": " ".join(str(item.get("title") or paper.get("title") or "").split()),
            "summary": " ".join(str((paper.get("ai_summary") if prefer_ai_summary else None) or item.get("ai_summary") or item.get("summary") or paper.get("summary") or "").split())[:600],
            "upvotes": paper.get("upvotes", item.get("upvotes", 0)) or 0,
            "github": paper.get("githubRepo", item.get("githubRepo")),
            "stars": paper.get("githubStars", item.get("githubStars")),
        })
    return records


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    PSEUDO-CODE:
      for attempt in 0 .. attempts-1:
          try: return fn()
          except RetryableError as e:
              if this was the last attempt: raise
              delay = e.retry_after if the server told us, else exponential backoff base * 2**attempt
              cap the delay at `cap` seconds; add random jitter to the exponential case
              sleep(delay)
    Use it to wrap EVERY network call below. Also treat these as retryable: HTTP 429/500/502/503/504,
    httpx.TransportError (timeouts, connection resets). Read the Retry-After header when present.
    """
    for attempt in range(max(1, attempts)):
        try:
            return fn()
        except RetryableError as exc:
            if attempt + 1 >= max(1, attempts):
                raise
            delay = (min(cap, max(0.0, exc.retry_after)) if exc.retry_after is not None else
                     min(cap, base * 2**attempt * random.uniform(0.5, 1.5)))
            time.sleep(delay)


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    global _last_arxiv_call
    try:
        terms = re.findall(r"[A-Za-z0-9-]+", query)
        if not terms:
            return "NO RESULTS"
        def fetch():
            global _last_arxiv_call
            with _arxiv_lock:
                delay = 3 - (time.monotonic() - _last_arxiv_call)
                if delay > 0:
                    time.sleep(delay)
                _last_arxiv_call = time.monotonic()
            return _request("GET", ARXIV_URL, params={
                "search_query": " AND ".join(f"all:{term}" for term in terms),
                "sortBy": "submittedDate", "sortOrder": "descending", "max_results": min(30, max(1, int(max_results))),
            })
        root = xml.etree.ElementTree.fromstring(with_retry(fetch, attempts=7, base=2, cap=60).text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in root.findall("a:entry", ns):
            raw_id = (entry.findtext("a:id", default="", namespaces=ns).rsplit("/", 1)[-1])
            paper_id = re.sub(r"v\d+$", "", raw_id)
            if not paper_id:
                continue
            records.append({"id": paper_id, "url": f"https://arxiv.org/abs/{paper_id}",
                "published": entry.findtext("a:published", default="", namespaces=ns)[:10],
                "title": " ".join(entry.findtext("a:title", default="", namespaces=ns).split()),
                "summary": " ".join(entry.findtext("a:summary", default="", namespaces=ns).split())[:600]})
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    # PSEUDO-CODE:
    #   GET HF_DAILY_URL params: limit (clamp 1..100) and date (only when given)      (with_retry)
    #   response = list of items {"paper": {id, title, summary, upvotes, githubRepo, githubStars, publishedAt}, ...}
    #   map every item to the record shape above (skip items without paper.id); url = https://huggingface.co/papers/<id>
    #   keyword -> keep records whose title+summary contains it (case-insensitive); sort by upvotes descending
    try:
        params = {"limit": min(100, max(1, int(limit)))}
        if date:
            params["date"] = date
        items = with_retry(lambda: _request("GET", HF_DAILY_URL, params=params).json())
        records = _records(items)
        if keyword:
            needle = keyword.casefold()
            records = [r for r in records if needle in (r["title"] + " " + r["summary"]).casefold()]
        records.sort(key=lambda r: r["upvotes"] if isinstance(r["upvotes"], (int, float)) else 0, reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    # PSEUDO-CODE:
    #   GET HF_SEARCH_URL params: q=query, limit (clamp 1..50)                         (with_retry)
    #   same item shape as the daily endpoint; prefer paper["ai_summary"] over paper["summary"] when present
    try:
        items = with_retry(lambda: _request("GET", HF_SEARCH_URL, params={"q": query, "limit": min(50, max(1, int(limit)))}).json())
        records = _records(items, prefer_ai_summary=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        return _exa("web_search_exa", {"query": query, "objective": objective or f"Find reliable sources about {query}", "numResults": min(10, max(1, int(num_results)))})
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    return _exa("web_fetch_exa", {"urls": [url]})


def _exa(name, arguments):
    key = os.getenv("EXA_API_KEY", "")
    def call():
        endpoint = str(httpx.URL(EXA_URL).copy_add_param("exaApiKey", key)) if key else EXA_URL
        response = _request("POST", endpoint, headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}})
        payload = None
        for line in response.text.splitlines():
            if line.startswith("data:"):
                try:
                    payload = json.loads(line[5:].strip())
                except ValueError:
                    continue
                break
        if payload is None:
            try:
                payload = response.json()
            except ValueError as exc:
                raise ValueError("Exa returned no JSON-RPC data") from exc
        if "error" in payload:
            raise ValueError(str(payload["error"]))
        result = payload.get("result") or {}
        meta = json.dumps(result.get("_meta", {})).casefold()
        contents = [str(c.get("text", "")) for c in result.get("content", []) if c.get("type") == "text"]
        message = "\n".join(contents)
        if any(word in meta for word in ("rate", "limit", "too many")) or re.search(r"rate.?limit|too many requests|quota exceeded", message, re.I):
            raise RetryableError("Exa rate limited", 20)
        if result.get("isError"):
            raise ValueError(message or "Exa tool returned an error")
        return message[:12000]
    try:
        value = with_retry(call, attempts=7, base=2, cap=60)
        return value or "NO RESULTS"
    except Exception as exc:
        message = str(exc).replace(key, "[REDACTED]").replace(quote(key, safe=""), "[REDACTED]") if key else str(exc)
        return f"ERROR: {type(exc).__name__}: {message}"


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
