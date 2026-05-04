#!/usr/bin/env python3
"""Build a cleaned corpus by applying two filters:

  1. Remove LOWI "besluiten" — procedural inadmissibility decisions that have
     no substantive counterpart in the UvNL corpus.
  2. Remove English-language documents from both corpora — UvNL has published
     a growing share of cases in English; LOWI publishes exclusively in Dutch.

Outputs:
  data/lowi_corpus_filtered.jsonl
  data/uvnl_corpus_filtered.jsonl

Usage:
    python build_filtered_corpus.py [--lang-threshold 0.80]
"""

import argparse
import json
from pathlib import Path

from langdetect import detect_langs, LangDetectException
from langdetect import DetectorFactory

# Fix random seed so langdetect is reproducible
DetectorFactory.seed = 42

LOWI_IN   = Path("data/lowi_corpus.jsonl")
UVNL_IN   = Path("data/uvnl_corpus.jsonl")
LOWI_OUT  = Path("data/lowi_corpus_filtered.jsonl")
UVNL_OUT  = Path("data/uvnl_corpus_filtered.jsonl")


def detect_language(text: str, threshold: float = 0.80) -> str:
    """
    Return ISO 639-1 language code ('nl', 'en', …) if confidence >= threshold,
    else 'unknown'.  Uses first 2000 characters for speed.
    """
    sample = text[:2000]
    try:
        probs = detect_langs(sample)
        best = probs[0]
        if best.prob >= threshold:
            return best.lang
    except LangDetectException:
        pass

    # Fallback: "the" word-frequency (strong English marker absent in Dutch)
    words = sample.lower().split()
    if not words:
        return "unknown"
    the_ratio = words.count("the") / len(words)
    if the_ratio > 0.025:
        return "en"
    return "unknown"


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(docs, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def filter_corpus(docs, source_label: str, threshold: float):
    kept, removed = [], []
    for d in docs:
        # --- Filter 1: LOWI besluiten ---
        if d.get("doc_type") == "besluit":
            removed.append((d["id"], "besluit"))
            continue

        # --- Filter 2: language ---
        lang = detect_language(d.get("text", ""), threshold=threshold)
        d["detected_lang"] = lang          # store for transparency
        if lang not in ("nl", "unknown"):  # keep Dutch and uncertain; drop confirmed non-Dutch
            removed.append((d["id"], f"lang={lang}"))
            continue

        kept.append(d)

    print(f"  {source_label}: {len(docs)} → {len(kept)} kept, {len(removed)} removed")
    for doc_id, reason in removed:
        print(f"    removed  {doc_id}  ({reason})")

    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang-threshold", type=float, default=0.80,
        help="Minimum langdetect confidence to accept a language label (default: 0.80)"
    )
    args = parser.parse_args()

    print("=== Filtering LOWI corpus ===")
    lowi_docs = load_jsonl(LOWI_IN)
    lowi_kept = filter_corpus(lowi_docs, "LOWI", args.lang_threshold)
    write_jsonl(lowi_kept, LOWI_OUT)

    print("\n=== Filtering UvNL corpus ===")
    uvnl_docs = load_jsonl(UVNL_IN)
    uvnl_kept = filter_corpus(uvnl_docs, "UvNL", args.lang_threshold)
    write_jsonl(uvnl_kept, UVNL_OUT)

    print(f"\nFiltered corpora written to:")
    print(f"  {LOWI_OUT}  ({len(lowi_kept)} documents)")
    print(f"  {UVNL_OUT}  ({len(uvnl_kept)} documents)")


if __name__ == "__main__":
    main()
