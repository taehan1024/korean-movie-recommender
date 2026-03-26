# korean-movie-recommender

Cross-cultural content-based movie recommender: US movie in -> ranked Korean movies out.

## Domain

Content-based recommendation using text similarity (TF-IDF, sentence embeddings), genre/keyword Jaccard, and year proximity. NOT collaborative filtering — no user interaction data.

## Pipeline

```bash
make setup          # venv + deps
make ingest         # TMDB API -> data/raw/*.json -> data/processed/*.csv
make fetch-keywords # Add TMDB keywords to catalogs
make features       # CSVs -> feature matrices in data/features/
make eval           # benchmark 3 models on gold pairs
make app            # launch Streamlit UI
make test           # run unit tests
```

## Key Files

| File | Purpose |
|---|---|
| `models.py` | 3 recommendation models (TF-IDF, embedding, hybrid) + explanations |
| `feature_engineering.py` | TF-IDF, sentence embeddings, genre/keyword/cast/year encoding |
| `evaluate.py` | DCG@10, Hit@K, MRR benchmarks + grid search tuning |
| `data_ingestion.py` | TMDB API fetch, concurrent ingestion, CSV export |
| `fetch_keywords.py` | TMDB keyword enrichment for existing catalogs |
| `curate_eval_pairs.py` | Semi-automated gold pair generation |
| `utils.py` | Shared utilities (TokenBucket rate limiter) |
| `app_v2.py` | Streamlit UI (live on Streamlit Cloud) |
| `models_v2.py` | Compatibility shim for live Streamlit app |

## Design Decisions

- **TMDB API only** — plots, genres, keywords, cast, crew, posters in one source
- **Sentence embeddings** — `all-MiniLM-L6-v2` handles translated Korean synopses well
- **Hybrid weights** — text=0.47, genre=0.24, keyword=0.18, year=0.11, cast=0.00 (grid-search tuned)
- **numpy cosine over FAISS** — 1512 KR movies = milliseconds, no need for ANN
- **TF-IDF fit on combined corpus** — shared vocabulary for cross-catalog similarity
- **Cross-catalog keywords only** — 1,768 keywords appearing in both catalogs as thematic bridge

## Data Schema (processed CSVs)

`tmdb_id, title, original_title, year, genres, overview, director, cast_top5, rating, vote_count, runtime, popularity, poster_path, translated, keywords`

## API

- TMDB key in `.env` as `TMDB_API_KEY`






- Rate limit: token bucket (35 req/10s) + exponential backoff on 429
