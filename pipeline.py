#!/usr/bin/env python3
"""
Full pipeline for the LOWI / UvNL corpus linguistics study.

Steps:
  1. scrape_lowi   – scrape ~245 HTML advices from lowi.nl
  2. scrape_uvnl   – download and extract text from ~140+ PDFs from uvnl
  3. run_bertopic  – compute multilingual embeddings and fit BERTopic
  4. gtest_analysis – G-test: overall + per 5-year period, with plots

Usage:
    python pipeline.py                    # full run
    python pipeline.py --skip-scrape      # skip scraping (use cached data)
    python pipeline.py --skip-bertopic    # skip BERTopic (reuse assignments)
    python pipeline.py --min-cluster-size 8
    python pipeline.py --model sentence-transformers/LaBSE
"""

import argparse
import sys
from pathlib import Path


def step(title):
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def check_corpus_files():
    missing = []
    for p in [Path("data/lowi_corpus.jsonl"), Path("data/uvnl_corpus.jsonl")]:
        if not p.exists() or p.stat().st_size == 0:
            missing.append(str(p))
    return missing


def main():
    parser = argparse.ArgumentParser(
        description="LOWI + UvNL corpus linguistics pipeline"
    )
    parser.add_argument(
        "--skip-scrape", action="store_true",
        help="Skip scraping; use existing data/*.jsonl files"
    )
    parser.add_argument(
        "--skip-bertopic", action="store_true",
        help="Skip BERTopic; use existing results/topic_assignments.csv"
    )
    parser.add_argument(
        "--min-cluster-size", type=int, default=5,
        help="HDBSCAN min_cluster_size for BERTopic (default: 5)"
    )
    parser.add_argument(
        "--model",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence-transformer model for embeddings"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    if not args.skip_scrape:
        step("STEP 1 – Scraping LOWI")
        from scrape_lowi import scrape as scrape_lowi
        scrape_lowi()

        step("STEP 2 – Scraping UvNL")
        from scrape_uvnl import scrape as scrape_uvnl
        scrape_uvnl()
    else:
        missing = check_corpus_files()
        if missing:
            print(f"ERROR: --skip-scrape given but these files are missing:\n  " +
                  "\n  ".join(missing), file=sys.stderr)
            sys.exit(1)
        print("Skipping scraping (--skip-scrape). Using cached corpus files.")

    # ------------------------------------------------------------------ #
    if not args.skip_bertopic:
        step("STEP 3 – BERTopic")
        from run_bertopic import main as run_bertopic
        run_bertopic(
            min_cluster_size=args.min_cluster_size,
            model_name=args.model,
        )
    else:
        ta = Path("results/topic_assignments.csv")
        if not ta.exists():
            print(f"ERROR: --skip-bertopic given but {ta} is missing.", file=sys.stderr)
            sys.exit(1)
        print("Skipping BERTopic (--skip-bertopic). Using cached assignments.")

    # ------------------------------------------------------------------ #
    step("STEP 4 – G-test analysis")
    from gtest_analysis import main as gtest_main
    gtest_main()

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("  Results are in:  results/")
    print("    topic_info.csv          – topic labels and sizes")
    print("    topic_assignments.csv   – per-document topic assignment")
    print("    gtest_results.csv       – G-test results (all periods)")
    print("    gtest_overall.png       – bar chart: overall comparison")
    print("    gtest_<period>.png      – bar charts per 5-year period")
    print("    heatmap_source_period.png – topic × source × period heatmap")
    print("    topic_time_series.png   – frequency trajectories")
    print("=" * 60)


if __name__ == "__main__":
    main()
