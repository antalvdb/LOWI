#!/usr/bin/env python3
"""Scrape all UvNL scientific integrity publications from universiteitenvannederland.nl

Each year (2013-2026) has a dedicated page linking to PDF files.
PDFs are downloaded, text is extracted with PyMuPDF, and results are
cached under data/uvnl/ to allow resuming.
"""

import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, unquote

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

try:
    import pymupdf as fitz  # pymupdf >= 1.24
except ImportError:
    import fitz  # older pymupdf installs

BASE_URL = "https://www.universiteitenvannederland.nl"
YEAR_URL_TEMPLATE = (
    BASE_URL
    + "/overzicht-publicaties-klachten-wetenschappelijke-integriteit-{year}"
)
YEARS = range(2013, 2027)
DATA_DIR = Path("data/uvnl")
PDF_DIR = DATA_DIR / "pdfs"
OUTPUT_FILE = Path("data/uvnl_corpus.jsonl")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; academic-corpus-scraper/1.0; "
        "scientific integrity NLP study)"
    )
}


def get_pdf_links(session, year):
    """Return list of {url, title, year} dicts for all PDFs on a year page."""
    url = YEAR_URL_TEMPLATE.format(year=year)
    try:
        resp = session.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
    except Exception as e:
        print(f"  Warning: failed to fetch {url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/files/publications/" in href and href.lower().endswith(".pdf"):
            full_url = urljoin(BASE_URL, href)
            title = a.get_text(" ", strip=True)
            links.append({"url": full_url, "title": title, "year": year})

    return links


def extract_pdf_text(pdf_path):
    """Extract plain text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(str(pdf_path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages).strip()
    except Exception as e:
        print(f"  Warning: could not read PDF {pdf_path.name}: {e}")
        return ""


def safe_stem(url):
    """Derive a filesystem-safe stem from a PDF URL."""
    filename = unquote(url.split("/")[-1])
    stem = Path(filename).stem
    # Keep only printable ASCII for cache key
    return re.sub(r"[^\w\-]", "_", stem)[:120]


def scrape():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    # --- Collect all PDF links across years ---
    all_links = []
    for year in tqdm(YEARS, desc="Year index pages"):
        links = get_pdf_links(session, year)
        all_links.extend(links)
        time.sleep(0.5)

    print(f"Found {len(all_links)} PDFs total.")

    # Deduplicate by URL
    seen_urls = set()
    unique_links = []
    for item in all_links:
        if item["url"] not in seen_urls:
            seen_urls.add(item["url"])
            unique_links.append(item)

    print(f"Unique PDFs: {len(unique_links)}")

    # --- Download and extract ---
    results = []
    for item in tqdm(unique_links, desc="Downloading & extracting PDFs"):
        pdf_url = item["url"]
        filename = unquote(pdf_url.split("/")[-1])
        pdf_path = PDF_DIR / filename
        cache_key = safe_stem(pdf_url)
        cache_file = DATA_DIR / f"{cache_key}.json"

        if cache_file.exists():
            doc = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            # Download
            if not pdf_path.exists():
                try:
                    resp = session.get(pdf_url, headers=HEADERS, timeout=60)
                    resp.raise_for_status()
                    pdf_path.write_bytes(resp.content)
                    time.sleep(0.6)
                except Exception as e:
                    print(f"\n  Error downloading {pdf_url}: {e}")
                    continue

            text = extract_pdf_text(pdf_path)

            # Year from filename prefix (e.g. "2013 Plagiaat - gegrond.pdf")
            year_m = re.match(r"(20\d{2})", filename)
            year = int(year_m.group(1)) if year_m else item["year"]

            doc = {
                "id": f"uvnl-{cache_key}",
                "source": "uvnl",
                "doc_type": "advies",
                "title": item["title"],
                "date": f"{year}-01-01",
                "year": year,
                "url": pdf_url,
                "filename": filename,
                "text": text,
            }
            cache_file.write_text(
                json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        if doc.get("text") and len(doc["text"]) > 50:
            results.append(doc)
        elif not doc.get("text"):
            print(f"\n  Warning: empty text for {doc.get('filename', doc['id'])}")

    # --- Write JSONL output ---
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for doc in results:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} UvNL documents to {OUTPUT_FILE}")
    return results


if __name__ == "__main__":
    scrape()
