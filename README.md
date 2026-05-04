# LOWI / UvNL Corpus Linguistics Study

A corpus linguistics pipeline comparing scientific integrity advisory opinions from two Dutch bodies:

- **LOWI** — [Landelijk Orgaan Wetenschappelijke Integriteit](https://lowi.nl/adviezen/) (~232 HTML documents, 2007–2026). LOWI is a national second-instance review body; complainants appeal to it after a university-level ruling.
- **UvNL** — [Universiteiten van Nederland](https://www.universiteitenvannederland.nl/publicaties-klachten-wetenschappelijke-integriteit) (~190 PDF documents, 2013–2026). These are first-instance decisions published by Dutch universities.

The pipeline scrapes both corpora, runs [BERTopic](https://maartengr.github.io/BERTopic/) on the union, and uses a G-test to identify topics that are significantly over- or under-represented in one corpus relative to the other — both overall and within 5-year periods.

---

## Pipeline

```
scrape_lowi.py         ← scrape ~232 LOWI adviezen (HTML)
scrape_uvnl.py         ← download & extract ~190 UvNL PDFs
        ↓
build_filtered_corpus.py   ← optional: remove besluiten + English docs
        ↓
run_bertopic.py        ← multilingual embeddings + BERTopic clustering
        ↓
gtest_analysis.py      ← G-test per topic, overall + per 5-year period
```

Or run everything at once:

```bash
python pipeline.py
```

---

## Setup

```bash
pip install -r requirements.txt
```

Key dependencies: `bertopic`, `sentence-transformers`, `umap-learn`, `hdbscan`, `pymupdf`, `scipy`, `langdetect`.

---

## Usage

### Full pipeline (scrape → topic model → G-test)

```bash
python pipeline.py
```

### Skip scraping if data is already cached

```bash
python pipeline.py --skip-scrape
```

### Filtered corpus (recommended for analysis)

Removes LOWI *besluiten* (inadmissibility decisions) and English-language UvNL documents:

```bash
python build_filtered_corpus.py
python run_bertopic.py \
    --lowi-file data/lowi_corpus_filtered.jsonl \
    --uvnl-file data/uvnl_corpus_filtered.jsonl \
    --results-dir results_filtered
python gtest_analysis.py --results-dir results_filtered
```

### Tuning options

| Flag | Default | Effect |
|------|---------|--------|
| `--min-cluster-size` | 5 | HDBSCAN minimum cluster size; increase for coarser topics |
| `--model` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence-transformer for embeddings |
| `--results-dir` | `results/` | Output directory |
| `--lowi-file` | `data/lowi_corpus.jsonl` | LOWI corpus input |
| `--uvnl-file` | `data/uvnl_corpus.jsonl` | UvNL corpus input |

---

## Output files

| File | Description |
|------|-------------|
| `topic_info.csv` | BERTopic topic labels, sizes, and keyword lists |
| `topic_assignments.csv` | Per-document topic assignment and probability |
| `gtest_results.csv` | G-test results for all topics × periods |
| `gtest_overall.png` | Bar chart: overall LOWI vs UvNL topic differences |
| `gtest_<period>.png` | Bar charts per 5-year period |
| `heatmap_source_period.png` | Normalised topic frequency by source × period |
| `topic_time_series.png` | Frequency trajectories for significant topics |

Results for the full corpus are in `results/`; results for the filtered corpus are in `results_filtered/`.

---

## Key findings

See `firstobservations.txt` for detailed results. In brief:

**Full corpus:** The dominant signal is that ~38% of UvNL documents are in English (absent from LOWI). After controlling for that, both corpora show distinct procedural vocabularies: UvNL first-instance decisions use *beklaagde/klager*, while LOWI appeal reviews use *verzoeker/belanghebbende*.

**Filtered corpus** (besluiten and English docs removed, 354 documents): 8 significantly differentiated topics. UvNL-dominant topics include an *ngwi/beklaagden* cluster that rises to 35% of UvNL documents in 2020–2024, likely reflecting adoption of standardised NGWI report templates after the 2018 Netherlands Code of Conduct revision. LOWI-dominant topics consistently cluster around appeals-procedure vocabulary across all periods.

---

## Caching and reproducibility

Scraped documents are cached per-file under `data/lowi/` and `data/uvnl/` (excluded from the repository). Embeddings are cached in `results*/embeddings.npy` (also excluded). Re-running any step from a warm cache is fast; a full cold run takes roughly 10–15 minutes (dominated by scraping).
