"""
#!/usr/bin/env python3

gitbook_download.py — Download all pages from a GitBook site as markdown.

Usage:
    python gitbook.py <gitbook_url> [--output DIR] [--concurrency N] [--delay SECONDS]

Examples:
    python gitbook.py https://hyperliquid.gitbook.io/hyperliquid-docs
    python gitbook.py https://hyperliquid.gitbook.io/hyperliquid-docs --output ./my-docs
    python gitbook.py https://docs.example.com -o ./output -c 5 -d 0.2

How it works:
    1. Fetches sitemap.xml → discovers sitemap-pages.xml for each section
    2. Parses all page URLs from all sitemaps
    3. Fetches each page with Accept: text/markdown (GitBook serves clean MD natively)
    4. Saves to disk preserving the URL path as folder structure

No Configuration REquired
"""

import argparse
import html
import os
import re
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
USER_AGENT = "Mozilla/5.0 (compatible; GitBookDownloader/1.0)"
DEFAULT_CONCURRENCY = 4
DEFAULT_DELAY = 0.3  # seconds between requests per thread

# ---------------------------------------------------------------------------
# HTTP session with retries
# ---------------------------------------------------------------------------


def make_session() -> requests.Session:
    """Create a requests session with retry/backoff."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# ---------------------------------------------------------------------------
# Sitemap discovery & parsing
# ---------------------------------------------------------------------------


def fetch_sitemap_urls(session: requests.Session, base_url: str) -> list[str]:
    """
    Given a GitBook base URL, discover all page URLs via sitemaps.

    Strategy:
      1. Try {base}/sitemap.xml (may be sitemap index or direct urlset)
      2. If it's a sitemap index, follow each child sitemap
      3. Fallback: try {base}/sitemap-pages.xml directly
    """
    base_url = base_url.rstrip("/")
    all_urls: list[str] = []
    seen_sitemaps: set[str] = set()

    def _parse_urlset(xml_text: str) -> list[str]:
        """Extract <loc> URLs from a <urlset> document."""
        root = ET.fromstring(xml_text)
        urls = []
        for url_elem in root.findall(f"{SITEMAP_NS}url"):
            loc = url_elem.find(f"{SITEMAP_NS}loc")
            if loc is not None and loc.text:
                urls.append(loc.text.strip())
        return urls

    def _parse_sitemap_index(xml_text: str) -> list[str]:
        """Extract child sitemap URLs from a <sitemapindex>."""
        root = ET.fromstring(xml_text)
        urls = []
        for sitemap_elem in root.findall(f"{SITEMAP_NS}sitemap"):
            loc = sitemap_elem.find(f"{SITEMAP_NS}loc")
            if loc is not None and loc.text:
                urls.append(loc.text.strip())
        return urls

    def _fetch_and_parse(sitemap_url: str) -> list[str]:
        """Fetch a sitemap and return page URLs."""
        if sitemap_url in seen_sitemaps:
            return []
        seen_sitemaps.add(sitemap_url)
        try:
            resp = session.get(sitemap_url, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  ⚠ Failed to fetch {sitemap_url}: {e}", file=sys.stderr)
            return []

        xml_text = resp.text
        # Determine if it's a sitemap index or a urlset
        if "<sitemapindex" in xml_text:
            child_urls = _parse_sitemap_index(xml_text)
            pages = []
            for child in child_urls:
                pages.extend(_fetch_and_parse(child))
            return pages
        elif "<urlset" in xml_text:
            return _parse_urlset(xml_text)
        else:
            print(f"  ⚠ Unknown sitemap format: {sitemap_url}", file=sys.stderr)
            return []

    # Try sitemap.xml first
    sitemap_url = f"{base_url}/sitemap.xml"
    print(f"Fetching sitemap: {sitemap_url}")
    try:
        resp = session.get(sitemap_url, timeout=15)
        resp.raise_for_status()
        all_urls = _fetch_and_parse(sitemap_url)
    except requests.RequestException:
        # Fallback: try sitemap-pages.xml directly
        fallback = f"{base_url}/sitemap-pages.xml"
        print(f"  No sitemap.xml, trying: {fallback}")
        try:
            all_urls = _fetch_and_parse(fallback)
        except requests.RequestException as e:
            print(f"  ✗ Could not find any sitemap: {e}", file=sys.stderr)
            return []

    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for url in all_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    return unique_urls


# ---------------------------------------------------------------------------
# URL → local file path mapping
# ---------------------------------------------------------------------------


def url_to_filepath(url: str, base_url: str, output_dir: Path) -> Path:
    """
    Convert a GitBook page URL to a local file path.

    Examples:
        base = https://docs.example.com
        https://docs.example.com              → output_dir/index.md
        https://docs.example.com/guide/intro  → output_dir/guide/intro.md
    """
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.rstrip("/")

    # Remove the base path prefix to get the relative doc path
    base_parsed = urllib.parse.urlparse(base_url)
    base_path = base_parsed.path.rstrip("/")

    if path == base_path or path == "":
        rel_path = "index"
    elif path.startswith(base_path + "/"):
        rel_path = path[len(base_path) + 1 :]
    elif base_path == "" and path.startswith("/"):
        rel_path = path[1:]
    else:
        # Fallback: use the full path minus leading slash
        rel_path = path.lstrip("/")

    return output_dir / f"{rel_path}.md"


# ---------------------------------------------------------------------------
# Page fetching
# ---------------------------------------------------------------------------


def fetch_page_markdown(
    session: requests.Session,
    url: str,
    delay: float,
    _thread_local=None,
) -> Optional[str]:
    """Fetch a single page as markdown from GitBook."""
    time.sleep(delay)  # rate limiting
    headers = {"Accept": "text/markdown"}
    try:
        resp = session.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        content = resp.text.strip()
        if not content:
            return None
        return content
    except requests.RequestException as e:
        print(f"  ✗ Failed: {url} — {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Link rewriting (optional — rewrite internal links to local paths)
# ---------------------------------------------------------------------------


def rewrite_internal_links(markdown: str, base_url: str, output_dir: Path) -> str:
    """
    Rewrite markdown links pointing to other pages in the same GitBook
    to local relative file paths.

    [text](/guide/intro) → [text](guide/intro.md)
    [text](https://docs.example.com/guide/intro) → [text](guide/intro.md)
    """
    base_parsed = urllib.parse.urlparse(base_url)
    base_path = base_parsed.path.rstrip("/")
    base_netloc = base_parsed.netloc

    def _rewrite_link(match):
        text = match.group(1)
        href = match.group(2)

        # Skip external links, anchors, mailto, etc.
        if href.startswith(("#", "mailto:", "tel:")):
            return match.group(0)
        if href.startswith("http") and base_netloc not in href:
            return match.group(0)

        # Resolve to a path
        if href.startswith("http"):
            parsed = urllib.parse.urlparse(href)
            link_path = parsed.path.rstrip("/")
        elif href.startswith("/"):
            link_path = href.rstrip("/")
        else:
            # Relative link — leave as-is
            return match.group(0)

        # Strip base path
        if link_path.startswith(base_path + "/"):
            rel = link_path[len(base_path) + 1 :]
        elif link_path == base_path:
            rel = "index"
        else:
            rel = link_path.lstrip("/")

        return f"[{text}]({rel}.md)"

    # Match [text](url) patterns
    return re.sub(r"\[([^\]]*)\]\(([^)]+)\)", _rewrite_link, markdown)


# ---------------------------------------------------------------------------
# HTML entity / tag cleanup
# ---------------------------------------------------------------------------


def clean_markdown(text: str) -> str:
    """
    Clean up artifacts from GitBook's markdown export:
    - Decode HTML entities (&#x20;, &amp;, etc.)
    - Convert <figure><img ...></figure> to markdown images
    - Strip trailing whitespace per line
    """
    # Decode HTML entities
    text = html.unescape(text)

    # Convert <figure><img src="..." alt="..." ...></figure> → ![](url)
    def _figure_to_md(match):
        tag = match.group(0)
        src_match = re.search(r'src="([^"]+)"', tag)
        alt_match = re.search(r'alt="([^"]*)"', tag)
        if src_match:
            src = html.unescape(src_match.group(1))
            alt = html.unescape(alt_match.group(1)) if alt_match else ""
            return f"![{alt}]({src})"
        return tag

    text = re.sub(
        r"<figure>\s*<img[^>]+/?>\s*(?:<figcaption>[^<]*</figcaption>)?\s*</figure>",
        _figure_to_md,
        text,
        flags=re.IGNORECASE,
    )

    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.splitlines())

    return text


# ---------------------------------------------------------------------------
# Main download orchestrator
# ---------------------------------------------------------------------------


def download_gitbook(
    base_url: str,
    output_dir: Path,
    concurrency: int = DEFAULT_CONCURRENCY,
    delay: float = DEFAULT_DELAY,
    rewrite_links: bool = True,
) -> dict:
    """
    Download all pages from a GitBook site.

    Returns a dict with stats: {total, success, failed, skipped}.
    """
    session = make_session()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Discover all page URLs
    print(f"\n📖 GitBook Downloader")
    print(f"   Source: {base_url}")
    print(f"   Output: {output_dir}")
    print()

    urls = fetch_sitemap_urls(session, base_url)
    print(f"  Found {len(urls)} pages in sitemap(s)")

    if not urls:
        print("  ✗ No pages found. Check the URL.", file=sys.stderr)
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0}

    # Step 2: Fetch all pages in parallel
    stats = {"total": len(urls), "success": 0, "failed": 0, "skipped": 0}
    completed = 0

    def _process_url(url: str) -> tuple[str, Optional[str]]:
        return url, fetch_page_markdown(session, url, delay)

    print(f"  Downloading with concurrency={concurrency}, delay={delay}s...")
    print()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_process_url, url): url for url in urls}
        for future in as_completed(futures):
            url, markdown = future.result()
            completed += 1

            if markdown is None:
                stats["failed"] += 1
                print(f"  [{completed}/{len(urls)}] ✗ {url}")
                continue

            # Optionally rewrite internal links
            if rewrite_links:
                markdown = rewrite_internal_links(markdown, base_url, output_dir)

            # Clean HTML entities and tags
            markdown = clean_markdown(markdown)

            # Map URL → file path and save
            filepath = url_to_filepath(url, base_url, output_dir)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(markdown, encoding="utf-8")

            stats["success"] += 1
            rel = filepath.relative_to(output_dir)
            print(f"  [{completed}/{len(urls)}] ✓ {rel}")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download all pages from a GitBook site as markdown files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
%(prog)s https://hyperliquid.gitbook.io/hyperliquid-docs
%(prog)s https://docs.example.com -o ./my-docs
%(prog)s https://docs.example.com -c 5 -d 0.1 --no-rewrite-links
        """,
    )
    parser.add_argument(
        "url",
        help="GitBook site URL (e.g., https://docs.example.com)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory (default: ./<domain-name>)",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Number of parallel downloads (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "-d",
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Delay between requests per thread in seconds (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--no-rewrite-links",
        action="store_true",
        help="Don't rewrite internal links to local paths",
    )

    args = parser.parse_args()

    # Default output dir: ./<slug>
    if args.output is None:
        parsed = urllib.parse.urlparse(args.url)
        slug = parsed.path.strip("/").split("/")[-1] or parsed.netloc.split(".")[0]
        args.output = f"./{slug}"

    stats = download_gitbook(
        base_url=args.url,
        output_dir=Path(args.output),
        concurrency=args.concurrency,
        delay=args.delay,
        rewrite_links=not args.no_rewrite_links,
    )

    print()
    print(f"Done! {stats['success']}/{stats['total']} pages downloaded.")
    if stats["failed"] > 0:
        print(f"  ({stats['failed']} failed)")
    print(f"  Output: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
