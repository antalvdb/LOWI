#!/usr/bin/env python3
"""Scrape all LOWI scientific integrity advices from lowi.nl/adviezen/

Pagination uses ?lcp_page0=N (1-indexed). Page 1 is the base URL.
Each document is a WordPress HTML page with the full advisory text.
Results are cached under data/lowi/ to allow resuming.
"""

import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://lowi.nl"
INDEX_URL = "https://lowi.nl/adviezen/"
DATA_DIR = Path("data/lowi")
OUTPUT_FILE = Path("data/lowi_corpus.jsonl")
MAX_PAGES = 30  # upper bound; scraper stops when no new links are found

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; academic-corpus-scraper/1.0; "
        "scientific integrity NLP study)"
    )
}

DUTCH_MONTHS = {
    "januari": 1, "februari": 2, "maart": 3, "april": 4,
    "mei": 5, "juni": 6, "juli": 7, "augustus": 8,
    "september": 9, "oktober": 10, "november": 11, "december": 12,
}

DATE_PATTERN = re.compile(
    r"(\d{1,2})\s+"
    r"(januari|februari|maart|april|mei|juni|juli|augustus|"
    r"september|oktober|november|december)\s+"
    r"(20\d{2})",
    re.IGNORECASE,
)

# URLs are at root level: lowi.nl/advies-YYYY-NN/ or lowi.nl/besluit-YYYY-NN/
# Some have a disambiguation suffix: lowi.nl/advies-2025-14-2/
DOC_HREF_RE = re.compile(
    r"https?://lowi\.nl/(advies|besluit)-(\d{4})-(\d+)(?:-\d+)?/?$"
)


def parse_dutch_date(text):
    m = DATE_PATTERN.search(text)
    if m:
        day, month_name, year = m.group(1), m.group(2).lower(), m.group(3)
        month = DUTCH_MONTHS.get(month_name)
        if month:
            return f"{year}-{month:02d}-{int(day):02d}", int(year)
    return None, None


def index_page_url(page_num):
    if page_num == 1:
        return INDEX_URL
    return f"{INDEX_URL}?lcp_page0={page_num}"


def collect_doc_urls(session):
    """Iterate index pages and collect all document hrefs."""
    seen = set()
    urls = []

    for page_num in tqdm(range(1, MAX_PAGES + 1), desc="Index pages"):
        url = index_page_url(page_num)
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            print(f"  Warning: failed to fetch {url}: {e}")
            break

        soup = BeautifulSoup(resp.text, "lxml")
        new_on_page = 0

        for a in soup.find_all("a", href=True):
            href = a["href"].strip().split("#")[0]  # strip fragment
            # Normalise relative → absolute
            if href.startswith("/"):
                href = BASE_URL + href
            elif not href.startswith("http"):
                continue
            href = href.rstrip("/") + "/"
            if DOC_HREF_RE.match(href) and href not in seen:
                seen.add(href)
                urls.append(href)
                new_on_page += 1

        if new_on_page == 0:
            # No new documents: we've passed the last page
            print(f"  No new documents on page {page_num}; stopping.")
            break

        time.sleep(0.6)

    return urls


def extract_doc(soup, url):
    """Extract text and metadata from a single LOWI document page."""
    # Title
    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else ""

    # Year from the case number in the URL is canonical (e.g. advies-2007-01 → 2007).
    # The HTML date can be the website-migration date for old documents, so we trust
    # the case number year over whatever date appears in the HTML.
    url_m = re.search(r"/(advies|besluit)-(\d{4})-", url)
    year = int(url_m.group(2)) if url_m else None

    # Still try to extract an exact date for reference
    date = None
    for time_el in soup.find_all("time"):
        dt = time_el.get("datetime", "") or time_el.get_text()
        d, _ = parse_dutch_date(dt)
        if d:
            date = d
            break

    if not date:
        for text in soup.stripped_strings:
            d, y = parse_dutch_date(text)
            if d and y == year:  # only accept if the parsed year matches case year
                date = d
                break

    if not date and year:
        date = f"{year}-01-01"

    # Main content: try WordPress selectors in order
    content = None
    for selector in [
        ".entry-content",
        ".post-content",
        ".the-content",
        "article",
        "main",
    ]:
        el = soup.select_one(selector)
        if el:
            content = el.get_text("\n", strip=True)
            break

    if not content:
        content = "\n".join(p.get_text(strip=True) for p in soup.find_all("p"))

    m = DOC_HREF_RE.match(url.rstrip("/") + "/")
    doc_type = m.group(1) if m else "advies"
    # Include any disambiguation suffix in the case_id
    raw_slug = url.rstrip("/").split("/")[-1]  # e.g. "advies-2025-14-2"
    case_id = re.sub(r"^(?:advies|besluit)-", "", raw_slug)

    return {
        "id": f"lowi-{case_id}",
        "source": "lowi",
        "doc_type": doc_type,
        "title": title,
        "date": date,
        "year": year,
        "url": url,
        "text": content,
    }


def scrape():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    # --- Collect all document URLs ---
    url_cache = DATA_DIR / "_urls.json"
    if url_cache.exists():
        urls = json.loads(url_cache.read_text(encoding="utf-8"))
        print(f"Loaded {len(urls)} cached document URLs.")
    else:
        print("Collecting document URLs from index pages...")
        urls = collect_doc_urls(session)
        url_cache.write_text(json.dumps(urls, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Found {len(urls)} document URLs.")

    # --- Scrape each document ---
    results = []
    for url in tqdm(urls, desc="Scraping documents"):
        slug = re.sub(r"[^a-z0-9]", "_", url.lower().rstrip("/").split("/")[-1])
        cache_file = DATA_DIR / f"{slug}.json"

        if cache_file.exists():
            doc = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            try:
                resp = session.get(url, headers=HEADERS, timeout=30)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "lxml")
                doc = extract_doc(soup, url)
                cache_file.write_text(
                    json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                time.sleep(0.6)
            except Exception as e:
                print(f"\n  Error scraping {url}: {e}")
                continue

        if doc.get("text") and len(doc["text"]) > 100:
            results.append(doc)

    # --- Write JSONL output ---
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for doc in results:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} LOWI documents to {OUTPUT_FILE}")
    return results


if __name__ == "__main__":
    scrape()
