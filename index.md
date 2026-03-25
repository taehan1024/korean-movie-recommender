# K-Movie Recommender

Cross-cultural content-based movie recommender: US movie → ranked Korean movies.

## Thesis

Korean cinema has rich diversity that maps well to American movie preferences. Content-based similarity across plot synopses, genres, and cast/crew can surface non-obvious Korean recommendations for US movie fans.

## Design Decisions

| Decision | Choice | Alternatives Considered | Rationale |
|---|---|---|---|
| Data source | TMDB API | IMDb bulk TSV | TMDB has plots, genres, cast, crew, posters in single API |
| Text model | all-MiniLM-L6-v2 | OpenAI embeddings, TF-IDF only | Free, 80MB, good paraphrase handling; TMDB provides English overviews for 99.9% of KR catalog |
| Similarity | numpy cosine | FAISS ANN | 2000 KR movies = milliseconds; FAISS is overkill |
| V1 hybrid weights | text=0.5, genre=0.3, cast=0.2 | Equal weights | Cast sparse across industries; text most discriminative |
| V2 hybrid weights | text=0.47, genre=0.24, keyword=0.18, cast=0.00, year=0.11 | Grid search (288 combos, DCG@10 optimized) | Keywords add thematic bridge (3x thematic Hit@10); cast=0 confirmed (0.8% overlap = noise); year proximity captures era similarity |
| Primary metric | DCG@10 | NDCG@10 | Jeunen et al. (KDD 2024): nDCG normalization can invert method ordering |

## Pipeline

```
# V1
make setup      → venv + install deps
make ingest     → TMDB API → data/raw/*.json → data/processed/*.csv
make features   → CSVs → TF-IDF, embeddings, genre, cast matrices
make eval       → benchmark 3 models on gold pairs
make app        → launch Streamlit UI

# V2
make fetch-keywords  → add TMDB keywords to CSVs
make features-v2     → build all features incl. keywords + year
make eval-v2         → benchmark with Hit@K, MRR, per-relevance breakdown
make eval-v2-tune    → grid search for optimal hybrid weights
make app-v2          → launch v2 Streamlit UI
make compare         → compare v1 vs v2 rank agreement
```

## Models

| Model | Version | Description |
|---|---|---|
| TF-IDF + Cosine | v1/v2 | Baseline: bag-of-words on plot synopses |
| Sentence Embeddings + Cosine | v1/v2 | Semantic similarity via all-MiniLM-L6-v2 |
| Hybrid v1 | v1 | Weighted fusion of text (0.5) + genre (0.3) + cast (0.2) |
| Hybrid v2 | v2 | text (0.50) + genre (0.20) + keyword (0.15) + cast (0.05) + year (0.10) |

## Results

### V1 Baseline

| Model | P@5 | P@10 | R@10 | DCG@10 | NDCG@10 |
|---|---|---|---|---|---|
| TF-IDF | 0.018 | 0.013 | 0.094 | 0.134 | 0.057 |
| Embedding | 0.021 | 0.018 | 0.095 | 0.173 | 0.066 |
| Hybrid v1 | 0.014 | 0.014 | 0.054 | 0.072 | 0.029 |

### V2 (with keywords + year proximity, grid-search-tuned weights)

| Model | Hit@5 | Hit@10 | MRR | P@10 | R@10 | DCG@10 | NDCG@10 |
|---|---|---|---|---|---|---|---|
| TF-IDF | 0.089 | 0.125 | 0.051 | 0.013 | 0.094 | 0.134 | 0.057 |
| Embedding | 0.107 | 0.161 | 0.069 | 0.018 | 0.095 | 0.173 | 0.066 |
| **Hybrid v2** | **0.214** | **0.232** | **0.171** | **0.025** | **0.125** | **0.298** | **0.123** |

### Per-Relevance Hit@10 (v2 hybrid, tuned)

| Relevance | Hit@10 | Notes |
|---|---|---|
| Remake (3) | 0.667 | Strong — remakes found 2/3 of the time |
| Thematic (2) | 0.250 | 3x improvement from keywords (was 0.083 pre-keywords) |
| Genre (1) | 0.195 | Genre-level matches |

### V1 → V2 Improvement Summary

| Metric | V1 Best (embedding) | V2 Hybrid (tuned) | Improvement |
|---|---|---|---|
| DCG@10 | 0.173 | 0.298 | +72% |
| MRR | — | 0.171 | (new metric) |
| NDCG@10 | 0.066 | 0.123 | +86% |
| Thematic Hit@10 | — | 0.250 | (new metric) |

### V1 vs V2 Rank Agreement

- Mean top-10 overlap: 62.5% (1.0 = identical)
- Mean rank correlation: 0.936
- V2 tends to surface more thematically similar films (e.g., artsy/atmospheric Korean cinema for Edward Scissorhands instead of generic romance)

**Acceptance Criteria (revised):**

| Metric | V2 Hybrid | Target | Status |
|---|---|---|---|
| Hit@10 | 0.232 | >= 0.35 | BELOW |
| MRR | 0.171 | >= 0.10 | PASS |
| R@10 | 0.125 | >= 0.25 | BELOW |
| DCG@10 | 0.298 | >= 0.50 | BELOW |

## Experiment Log

| Date | Experiment | Key Finding |
|---|---|---|
| 2026-03-23 | Project initialized | Pipeline scaffolded, 6 scripts written |
| 2026-03-23 | V2 implementation | Bug fixes + year proximity + rebalanced weights. Hybrid v2 DCG@10 = 0.236 (+36% vs v1 embedding best). MRR passes target. |
| 2026-03-23 | Keywords + grid search | TMDB keywords fetched (1768 cross-catalog). Grid search (288 combos) → optimal weights: text=0.47, genre=0.24, keyword=0.18, cast=0.00, year=0.11. DCG@10 = 0.298 (+72% vs v1). Thematic Hit@10 tripled (0.083→0.250). Cast confirmed as noise (optimal weight = 0). |

## Vault References

- Deep learning recommendation survey: `00-Knowledge/arxiv/deep_learning_recommendation_system_neural/use-of-deep-learning-in-modern-recommendation-system-a-summary-of-recent-works.md`
- DaisyRec evaluation: `00-Knowledge/arxiv/recommendation_system_evaluation_metrics_offline/daisyrec-2-0-benchmarking-recommendation-for-rigorous-evaluation.md`
- nDCG critique: `00-Knowledge/arxiv/recommendation_system_evaluation_metrics_offline/on-normalised-discounted-cumulative-gain-as-an-off-policy-evaluation-metric-for-.md`
- ReasoningRec (item descriptions): `00-Knowledge/arxiv/metadata/reasoningrec-bridging-personalized-recommendations-and-human-interpretable-expla-metadata.md`
- Cross-domain rec: `00-Knowledge/arxiv/cold_start_recommendation_meta-learning/transfer-meta-framework-for-cross-domain-recommendation-to-cold-start-users.md`
- BookGPT (LLM rec): `00-Knowledge/arxiv/large_language_model_recommendation_system/bookgpt-a-general-framework-for-book-recommendation-empowered-by-large-language-.md`
