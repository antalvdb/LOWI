#!/usr/bin/env python3
"""Interactive 2-D cluster visualization for the filtered corpus.

Loads pre-computed embeddings, reduces to 2-D with UMAP, and writes
an interactive Plotly scatter to results_filtered/clusters_interactive.html.
"""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from umap import UMAP

RESULTS_DIR = Path("results_filtered")
LOWI_FILE   = Path("data/lowi_corpus_filtered.jsonl")
UVNL_FILE   = Path("data/uvnl_corpus_filtered.jsonl")

# Colour-blind-friendly palette (tab20-ish); index -1 (noise) → grey
PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#1f77b4", "#aec7e8", "#ffbb78", "#98df8a", "#d62728",
    "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
    "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22",
    "#dbdb8d", "#17becf", "#9edae5", "#393b79",
]
NOISE_COLOUR = "#cccccc"


def load_texts():
    docs = {}
    for path in [LOWI_FILE, UVNL_FILE]:
        with path.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                docs[d["id"]] = d
    return docs


def topic_label(row):
    rep = row.get("Representation", "")
    if isinstance(rep, str) and rep.startswith("["):
        try:
            words = ast.literal_eval(rep)
            return ", ".join(words[:5])
        except Exception:
            pass
    return str(row.get("Topic", "?"))


def main():
    print("Loading data...")
    embeddings  = np.load(RESULTS_DIR / "embeddings.npy")
    assignments = pd.read_csv(RESULTS_DIR / "topic_assignments.csv")
    topic_info  = pd.read_csv(RESULTS_DIR / "topic_info.csv")
    texts       = load_texts()

    print(f"Embeddings: {embeddings.shape}  |  Docs: {len(assignments)}")

    # Build topic → label map
    label_map = {-1: "Noise"}
    for _, row in topic_info[topic_info["Topic"] != -1].iterrows():
        label_map[int(row["Topic"])] = f"T{int(row['Topic'])}: {topic_label(row.to_dict())}"

    # 2-D UMAP (separate from the 5-D used for clustering)
    print("Reducing to 2-D with UMAP...")
    umap2d = UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        low_memory=False,
    )
    xy = umap2d.fit_transform(embeddings)
    assignments["x"] = xy[:, 0]
    assignments["y"] = xy[:, 1]

    # Enrich with document metadata
    def get_meta(row, field, default=""):
        doc = texts.get(row.get("id", ""), {})
        return doc.get(field, default)

    assignments["title"] = assignments.apply(lambda r: get_meta(r, "title", r.get("id", "")), axis=1)
    assignments["url"]   = assignments.apply(lambda r: get_meta(r, "url", ""), axis=1)
    assignments["topic_label"] = assignments["topic"].map(label_map).fillna("Unknown")

    # Build figure: one trace per topic so the legend is toggleable
    fig = go.Figure()

    topic_ids = sorted(assignments["topic"].unique(), key=lambda t: (t == -1, t))

    for tid in topic_ids:
        sub = assignments[assignments["topic"] == tid]
        colour = NOISE_COLOUR if tid == -1 else PALETTE[tid % len(PALETTE)]
        label  = label_map.get(tid, f"T{tid}")

        # Marker shape: circle = LOWI, diamond = UvNL
        symbols = sub["source"].map({"lowi": "circle", "uvnl": "diamond"}).tolist()

        hover = (
            "<b>%{customdata[0]}</b><br>"
            "Source: %{customdata[1]}  |  Period: %{customdata[2]}<br>"
            "Topic: %{customdata[3]}<extra></extra>"
        )

        fig.add_trace(go.Scatter(
            x=sub["x"],
            y=sub["y"],
            mode="markers",
            name=label,
            marker=dict(
                color=colour,
                symbol=symbols,
                size=7,
                opacity=0.80,
                line=dict(width=0.4, color="white"),
            ),
            customdata=sub[["title", "source", "period", "topic_label"]].values,
            hovertemplate=hover,
        ))

    fig.update_layout(
        title=dict(
            text="BERTopic clusters — filtered corpus (LOWI + UvNL, Dutch only, adviezen only)<br>"
                 "<sup>Circle = LOWI  |  Diamond = UvNL  |  Grey = noise</sup>",
            font_size=14,
        ),
        legend=dict(
            title="Topic",
            font_size=10,
            itemsizing="constant",
            tracegroupgap=2,
        ),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=1200,
        height=800,
        margin=dict(l=20, r=20, t=80, b=20),
    )

    out = RESULTS_DIR / "clusters_interactive.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"\nSaved: {out}")
    print("Open in a browser to explore interactively.")


if __name__ == "__main__":
    main()
