# korean-movie-recommender

Cross-cultural content-based movie recommender: US movie in → ranked Korean movies out.

## Domain

Content-based recommendation using text similarity (TF-IDF, sentence embeddings), genre overlap, and cast/crew matching. NOT collaborative filtering — no user interaction data.

## Pipeline

```bash
make setup          # venv + deps
make ingest         # TMDB API → data/raw/*.json → data/processed/*.csv
make features       # CSVs → feature matrices in data/features/
make eval           # benchmark 3 models on gold pairs
make app            # launch Streamlit UI
```

## Key Files

| File | Purpose |
|---|---|
| `data_ingestion.py` | TMDB API fetch, translation, CSV export |
| `feature_engineering.py` | TF-IDF, sentence embeddings, genre/cast encoding |
| `models.py` | 3 recommendation models with shared interface |
| `evaluate.py` | P@5, P@10, NDCG@10 benchmarks |
| `curate_eval_pairs.py` | Semi-automated gold pair generation |
| `app.py` | Streamlit UI |

## Design Decisions

- **TMDB API only** — has everything (plots, genres, cast, crew, posters) in one source
- **Sentence embeddings** — `all-MiniLM-L6-v2` handles translated Korean synopses well
- **Hybrid weights** — text=0.5, genre=0.3, cast=0.2 (cast sparse across industries)
- **numpy cosine over FAISS** — 2000 KR movies = milliseconds, no need for ANN
- **TF-IDF fit on combined corpus** — shared vocabulary for cross-catalog similarity

## Data Schema (processed CSVs)

`tmdb_id, title, original_title, year, genres, overview, director, cast_top5, rating, vote_count, runtime, popularity, poster_path, translated`

## API

- TMDB key in `.env` as `TMDB_API_KEY`
- Rate limit: 0.3s between requests + exponential backoff on 429
