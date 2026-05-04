#!/usr/bin/env python3
"""G-test analysis comparing topic distributions between LOWI and UvNL.

For each topic t we test:
  H0: P(topic=t | LOWI) = P(topic=t | UvNL)

using the log-likelihood ratio (G) test on a 2×2 contingency table:

              in topic t   not in topic t
  LOWI        k_A          N_A - k_A
  UvNL        k_B          N_B - k_B

This is repeated globally and within each 5-year period.
Multiple-testing is corrected with Bonferroni per comparison set.
Results and plots are written to results/.

Usage:
    python gtest_analysis.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

RESULTS_DIR = Path("results")  # default; overridden by CLI --results-dir

PERIODS_ORDER = ["2005-2009", "2010-2014", "2015-2019", "2020-2024", "2025+"]


# ---------------------------------------------------------------------------
# G-test helpers
# ---------------------------------------------------------------------------

def gtest_2x2(k_a, n_a, k_b, n_b):
    """
    G-test for a single 2×2 table.
    Returns (G, p, log2_odds_ratio).
    log2_odds_ratio > 0 → topic overrepresented in corpus A (LOWI).
    """
    table = np.array([[k_a, n_a - k_a],
                      [k_b, n_b - k_b]], dtype=float)
    # Yates-style: add 0.5 to all cells to avoid log(0)
    table += 0.5
    G, p, _, _ = chi2_contingency(table, lambda_="log-likelihood")
    odds_a = (k_a + 0.5) / (n_a - k_a + 0.5)
    odds_b = (k_b + 0.5) / (n_b - k_b + 0.5)
    log2_or = np.log2(odds_a / odds_b)
    return float(G), float(p), float(log2_or)


def topic_keywords(topic_row):
    """Return a short string label for a topic row from topic_info.csv."""
    rep = topic_row.get("Representation", "")
    if isinstance(rep, str) and rep.startswith("["):
        # Stored as Python list repr; parse it
        import ast
        try:
            words = ast.literal_eval(rep)
            return ", ".join(words[:5])
        except Exception:
            pass
    if isinstance(rep, str) and rep:
        return rep[:60]
    return str(topic_row.get("Topic", "?"))


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def run_gtest(df_sub, topic_info, label):
    """
    Run G-test for every non-noise topic in df_sub.

    df_sub must have columns: source ('lowi'/'uvnl'), topic (int).
    Returns a DataFrame of results, or None if one source is missing.
    """
    df_lowi = df_sub[df_sub["source"] == "lowi"]
    df_uvnl = df_sub[df_sub["source"] == "uvnl"]

    n_lowi, n_uvnl = len(df_lowi), len(df_uvnl)
    if n_lowi < 5 or n_uvnl < 5:
        print(f"  [{label}] Skipping: too few docs (LOWI={n_lowi}, UvNL={n_uvnl})")
        return None

    valid_topics = topic_info[topic_info["Topic"] != -1]["Topic"].tolist()
    rows = []
    for tid in valid_topics:
        k_a = (df_lowi["topic"] == tid).sum()
        k_b = (df_uvnl["topic"] == tid).sum()
        G, p, log2_or = gtest_2x2(k_a, n_lowi, k_b, n_uvnl)

        kw = topic_keywords(
            topic_info[topic_info["Topic"] == tid].iloc[0].to_dict()
        )
        rows.append({
            "period": label,
            "topic_id": tid,
            "keywords": kw,
            "count_lowi": int(k_a),
            "count_uvnl": int(k_b),
            "n_lowi": n_lowi,
            "n_uvnl": n_uvnl,
            "freq_lowi": k_a / n_lowi,
            "freq_uvnl": k_b / n_uvnl,
            "G": G,
            "p": p,
            "log2_odds_ratio": log2_or,
            "preferred": "lowi" if log2_or > 0 else "uvnl",
        })

    result = pd.DataFrame(rows)
    n_tests = len(result)
    result["p_bonferroni"] = (result["p"] * n_tests).clip(upper=1.0)
    result["significant"] = result["p_bonferroni"] < 0.05
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_gtest(result_df, title, filename, results_dir=None):
    """Horizontal bar chart of G values; blue = LOWI-preferred, coral = UvNL-preferred."""
    out = Path(results_dir) if results_dir else RESULTS_DIR
    sig = result_df[result_df["significant"]].copy()
    if sig.empty:
        print(f"  [{title}] No significant results; skipping plot.")
        return

    sig = sig.sort_values("G", ascending=True)
    colors = ["steelblue" if r == "lowi" else "coral" for r in sig["preferred"]]

    fig, ax = plt.subplots(figsize=(10, max(3, len(sig) * 0.45)))
    ax.barh(sig["keywords"], sig["G"], color=colors)

    legend = [
        mpatches.Patch(color="steelblue", label="LOWI over-represented"),
        mpatches.Patch(color="coral",     label="UvNL over-represented"),
    ]
    ax.legend(handles=legend, fontsize=9)
    ax.set_xlabel("G statistic (Bonferroni p < 0.05)")
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def plot_heatmap(df, topic_info, results_dir=None):
    """
    Heatmap of normalised topic frequency by (source × period).
    Rows = source_period combinations, columns = topics.
    """
    valid_topics = topic_info[topic_info["Topic"] != -1]["Topic"].tolist()
    df_v = df[df["topic"].isin(valid_topics)].copy()
    df_v["period"] = df_v["period"].astype(str)

    # Build ordered row index
    sources = ["lowi", "uvnl"]
    row_keys = [f"{s}_{p}" for p in PERIODS_ORDER for s in sources]
    df_v["src_period"] = df_v["source"] + "_" + df_v["period"]

    pivot = (
        df_v.groupby(["src_period", "topic"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=[k for k in row_keys if k in df_v["src_period"].unique()],
                 columns=valid_topics,
                 fill_value=0)
    )
    pivot_norm = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    # Topic labels
    kw_map = {
        row["Topic"]: topic_keywords(row.to_dict())
        for _, row in topic_info[topic_info["Topic"] != -1].iterrows()
    }
    pivot_norm.columns = [kw_map.get(c, str(c)) for c in pivot_norm.columns]

    import seaborn as sns
    ncols = len(pivot_norm.columns)
    nrows = len(pivot_norm)
    fig, ax = plt.subplots(figsize=(max(14, ncols * 0.9), max(5, nrows * 0.6)))
    sns.heatmap(
        pivot_norm,
        ax=ax,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.4,
    )
    ax.set_title("Normalised topic frequency by source × period", fontsize=11)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Source × Period")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    out = Path(results_dir) if results_dir else RESULTS_DIR
    plt.savefig(out / "heatmap_source_period.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: heatmap_source_period.png")


def plot_topic_time_series(df, topic_info, results_dir=None):
    """
    For topics that are significantly different overall, show their
    frequency trajectory over 5-year periods per source.
    """
    out = Path(results_dir) if results_dir else RESULTS_DIR
    overall_file = out / "gtest_results.csv"
    if not overall_file.exists():
        return
    gtest = pd.read_csv(overall_file)
    sig_overall = gtest[(gtest["period"] == "overall") & gtest["significant"]]
    if sig_overall.empty:
        return

    valid_topics = sig_overall["topic_id"].tolist()
    kw_map = {
        row["Topic"]: topic_keywords(row.to_dict())
        for _, row in topic_info[topic_info["Topic"] != -1].iterrows()
    }

    df_v = df[df["topic"].isin(valid_topics)].copy()
    df_v["period"] = df_v["period"].astype(str)

    fig, axes = plt.subplots(
        len(valid_topics), 1,
        figsize=(10, max(4, len(valid_topics) * 2.5)),
        squeeze=False,
    )

    for ax, tid in zip(axes[:, 0], valid_topics):
        for src, color, marker in [("lowi", "steelblue", "o"), ("uvnl", "coral", "s")]:
            sub = df_v[(df_v["topic"] == tid) & (df_v["source"] == src)]
            counts = sub.groupby("period").size()
            totals = df[(df["source"] == src)].groupby("period").size()
            freq = (counts / totals).reindex(PERIODS_ORDER, fill_value=0)
            ax.plot(PERIODS_ORDER, freq.values, marker=marker, color=color, label=src)
        ax.set_title(f"Topic {tid}: {kw_map.get(tid, '')}", fontsize=9)
        ax.set_ylabel("Freq.")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(out / "topic_time_series.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: topic_time_series.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(results_dir=None):
    out = Path(results_dir) if results_dir else RESULTS_DIR
    assignments_file = out / "topic_assignments.csv"
    topic_info_file  = out / "topic_info.csv"

    print("Loading topic assignments...")
    df = pd.read_csv(assignments_file)
    topic_info = pd.read_csv(topic_info_file)

    print(f"Total: {len(df)}  |  LOWI: {(df['source']=='lowi').sum()}  |  UvNL: {(df['source']=='uvnl').sum()}")
    print(f"Topics found: {len(topic_info[topic_info['Topic']!=-1])}\n")

    out.mkdir(parents=True, exist_ok=True)
    all_results = []

    # 1. Overall comparison
    print("=== Overall G-test ===")
    overall = run_gtest(df, topic_info, "overall")
    if overall is not None:
        all_results.append(overall)
        sig = overall[overall["significant"]]
        print(f"  Significant topics: {len(sig)} / {len(overall)}")
        if not sig.empty:
            display_cols = ["topic_id", "keywords", "freq_lowi", "freq_uvnl",
                            "G", "p_bonferroni", "preferred"]
            print(sig[display_cols].sort_values("G", ascending=False).to_string(index=False))
        plot_gtest(overall, "Topic comparison: LOWI vs UvNL (overall)",
                   "gtest_overall.png", results_dir=out)

    # 2. Per 5-year period
    print("\n=== Per 5-year period G-tests ===")
    for period in PERIODS_ORDER:
        df_p = df[df["period"] == period]
        print(f"\n--- {period}  (n={len(df_p)}) ---")
        res = run_gtest(df_p, topic_info, period)
        if res is not None:
            all_results.append(res)
            sig = res[res["significant"]]
            print(f"  Significant: {len(sig)} / {len(res)}")
            if not sig.empty:
                display_cols = ["topic_id", "keywords", "freq_lowi", "freq_uvnl",
                                "G", "p_bonferroni", "preferred"]
                print(sig[display_cols].sort_values("G", ascending=False).to_string(index=False))
            safe = period.replace("+", "plus")
            plot_gtest(res, f"Topic comparison: LOWI vs UvNL ({period})",
                       f"gtest_{safe}.png", results_dir=out)

    # 3. Save combined results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(out / "gtest_results.csv", index=False)
        print(f"\nAll G-test results → {out}/gtest_results.csv")

    # 4. Summary plots
    print("\nGenerating summary plots...")
    plot_heatmap(df, topic_info, results_dir=out)
    plot_topic_time_series(df, topic_info, results_dir=out)

    print(f"\nDone. All output in {out}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=None,
                        help="Directory with topic_assignments.csv and topic_info.csv (default: results/)")
    args = parser.parse_args()
    main(results_dir=args.results_dir)
