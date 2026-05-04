#!/usr/bin/env python3
"""Run BERTopic on the combined LOWI + UvNL corpus.

Embeddings are computed with paraphrase-multilingual-MiniLM-L12-v2
(good for Dutch). Results are saved to results/ for the G-test step.

Usage:
    python run_bertopic.py [--min-cluster-size N] [--model NAME]
                           [--lowi-file PATH] [--uvnl-file PATH]
                           [--results-dir DIR]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

# Defaults — overridden by CLI args
LOWI_FILE = Path("data/lowi_corpus.jsonl")
UVNL_FILE = Path("data/uvnl_corpus.jsonl")
RESULTS_DIR = Path("results")

# Common Dutch function words to exclude from topic keywords.
# BERTopic's c-TF-IDF already down-weights corpus-wide common terms,
# but these closed-class words add noise to topic labels.
DUTCH_STOP_WORDS = [
    "de", "het", "een", "en", "van", "in", "is", "op", "aan", "te",
    "dat", "er", "zijn", "voor", "met", "als", "ook", "door", "bij",
    "of", "naar", "om", "die", "niet", "maar", "uit", "ze", "dan",
    "wordt", "worden", "heeft", "hebben", "dit", "hij", "haar", "hun",
    "we", "ik", "je", "u", "zich", "nog", "nu", "al", "geen", "over",
    "werd", "waren", "zo", "kan", "meer", "wel", "wat", "welke", "tot",
    "onder", "tussen", "na", "af", "hier", "daar", "waar", "hoe",
    "wanneer", "waarom", "wie", "welk", "ieder", "elk", "reeds",
    "aldus", "echter", "voorts", "immers", "derhalve", "mitsdien",
    "dient", "dienen", "teneinde", "inzake", "terzake", "betreffende",
    "alsmede", "waarbij", "waarvan", "waarop", "waarna", "waarmee",
    "hetgeen", "zulks", "zodanig", "zodanige", "zulke", "dergelijke",
    "onderhavige", "voornoemde", "genoemde", "voormelde",
    "hierbij", "hierin", "hiervan", "hierop", "hiermee", "hiertoe",
    "daarin", "daarmee", "daartoe", "daarvoor", "daarnaar", "daarbij",
]


def load_corpus(lowi_file=None, uvnl_file=None):
    """Load both corpora, assign 5-year period labels, return a DataFrame."""
    docs = []
    for path in [Path(lowi_file) if lowi_file else LOWI_FILE,
                  Path(uvnl_file) if uvnl_file else UVNL_FILE]:
        if not path.exists():
            print(f"  Warning: {path} not found; skipping.")
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                if doc.get("text") and len(doc["text"].strip()) > 100:
                    docs.append(doc)

    df = pd.DataFrame(docs)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)

    df["period"] = pd.cut(
        df["year"],
        bins=[2004, 2009, 2014, 2019, 2024, 2030],
        labels=["2005-2009", "2010-2014", "2015-2019", "2020-2024", "2025+"],
    ).astype(str)
    df["period"] = df["period"].replace("nan", "unknown")

    return df


def build_topic_model(min_cluster_size, embedding_model):
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=False,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=max(1, min_cluster_size // 2),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=DUTCH_STOP_WORDS,
        min_df=2,
        max_df=0.90,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=10,
        verbose=True,
    )

    return topic_model


def main(
    min_cluster_size=5,
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
    lowi_file=None,
    uvnl_file=None,
    results_dir=None,
):
    out_dir = Path(results_dir) if results_dir else RESULTS_DIR
    model_dir = out_dir / "topic_model"

    print("Loading corpus...")
    df = load_corpus(lowi_file=lowi_file, uvnl_file=uvnl_file)
    n = len(df)
    print(f"Total: {n} documents  |  LOWI: {(df['source']=='lowi').sum()}  |  UvNL: {(df['source']=='uvnl').sum()}")

    texts = df["text"].tolist()

    print(f"\nLoading sentence-transformer '{model_name}'...")
    embedding_model = SentenceTransformer(model_name)

    out_dir.mkdir(parents=True, exist_ok=True)
    emb_cache = out_dir / "embeddings.npy"
    if emb_cache.exists():
        print("Loading cached embeddings...")
        embeddings = np.load(emb_cache)
        if embeddings.shape[0] != n:
            print(f"  Cache size mismatch ({embeddings.shape[0]} vs {n}); recomputing.")
            emb_cache.unlink()
            embeddings = None
    else:
        embeddings = None

    if embeddings is None:
        print("Computing embeddings (this may take a few minutes)...")
        embeddings = embedding_model.encode(
            texts, show_progress_bar=True, batch_size=16, normalize_embeddings=True
        )
        np.save(emb_cache, embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    print(f"\nFitting BERTopic (min_cluster_size={min_cluster_size})...")
    topic_model = build_topic_model(min_cluster_size, embedding_model)
    topics, probs = topic_model.fit_transform(texts, embeddings)

    # probs may be 1-D (per-doc probability) or 2-D (full distribution)
    if isinstance(probs, np.ndarray) and probs.ndim == 2:
        topic_probs = probs.max(axis=1).tolist()
    elif isinstance(probs, np.ndarray):
        topic_probs = probs.tolist()
    else:
        topic_probs = [float(p) for p in probs]

    # --- Save results ---
    model_dir.mkdir(parents=True, exist_ok=True)
    topic_model.save(str(model_dir), serialization="pytorch", save_ctfidf=True)

    df_out = df.drop(columns=["text"], errors="ignore").copy()
    df_out["topic"] = topics
    df_out["topic_prob"] = topic_probs
    df_out.to_csv(out_dir / "topic_assignments.csv", index=False)

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(out_dir / "topic_info.csv", index=False)

    n_topics = len(topic_info[topic_info["Topic"] != -1])
    n_noise = (np.array(topics) == -1).sum()
    print(f"\nFound {n_topics} topics. Noise documents: {n_noise} / {n}")
    print("\nTop topics:")
    print(topic_info.head(20).to_string())

    return topic_model, df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-cluster-size", type=int, default=5,
                        help="HDBSCAN min_cluster_size (default: 5)")
    parser.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2",
                        help="Sentence-transformer model name")
    parser.add_argument("--lowi-file", default=None,
                        help="Path to LOWI corpus JSONL (default: data/lowi_corpus.jsonl)")
    parser.add_argument("--uvnl-file", default=None,
                        help="Path to UvNL corpus JSONL (default: data/uvnl_corpus.jsonl)")
    parser.add_argument("--results-dir", default=None,
                        help="Output directory (default: results/)")
    args = parser.parse_args()
    main(
        min_cluster_size=args.min_cluster_size,
        model_name=args.model,
        lowi_file=args.lowi_file,
        uvnl_file=args.uvnl_file,
        results_dir=args.results_dir,
    )
