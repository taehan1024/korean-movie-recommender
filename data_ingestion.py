"""TMDB data ingestion pipeline with concurrent fetching.

Two-phase fetch: discover movie IDs via paginated search, then fetch
details concurrently (4 threads) with token bucket rate limiting
(35 req/10s, under TMDB's 40 req/10s ceiling). Checkpoints every 100
movies for crash recovery.
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from utils import TokenBucket

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_RAW = SCRIPT_DIR / "data" / "raw"
DATA_PROCESSED = SCRIPT_DIR / "data" / "processed"
DATA_METADATA = SCRIPT_DIR / "data" / "metadata"

TMDB_BASE = "https://api.themoviedb.org/3"
MAX_BACKOFF_SEC = 60
MAX_RETRIES = 3

# Rate limiter config: 35 requests per 10-second window
RATE_LIMIT = 35
RATE_WINDOW_SEC = 10.0

# Concurrency: 4 workers keeps requests in-flight without overwhelming
WORKERS = 4
CHECKPOINT_EVERY = 100

US_MAX_PAGES = 250
KR_MAX_PAGES = 150
US_MIN_VOTES = 50
KR_MIN_VOTES = 10

REQUIRED_FIELDS = [
    "tmdb_id", "title", "original_title", "year", "genres", "overview",
    "director", "cast_top5", "rating", "vote_count", "runtime",
    "popularity", "poster_path", "translated",
]


# Global rate limiter shared across all threads
_bucket = TokenBucket(RATE_LIMIT, RATE_WINDOW_SEC)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def load_api_key() -> str:
    load_dotenv(SCRIPT_DIR / ".env")
    key = os.getenv("TMDB_API_KEY")
    if not key or key == "your_api_key_here":
        raise SystemExit("Set TMDB_API_KEY in .env before running.")
    return key


def _get(url: str, params: dict, api_key: str) -> dict | None:
    """GET with token bucket rate limiting and exponential backoff on 429."""
    params["api_key"] = api_key
    for attempt in range(MAX_RETRIES):
        _bucket.acquire()
        try:
            resp = requests.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            log.info(f"  Request error: {e}")
            return None

        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            wait = min(2 ** (attempt + 1), MAX_BACKOFF_SEC)
            log.info(f"  Rate limited (429), backing off {wait}s...")
            time.sleep(wait)
            continue
        log.info(f"  TMDB error {resp.status_code} for {url}")
        return None
    log.info(f"  Failed after {MAX_RETRIES} retries: {url}")
    return None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_page(language: str, page: int, min_votes: int, api_key: str) -> list[int]:
    """Return list of TMDB IDs from one discover page."""
    params = {
        "with_original_language": language,
        "sort_by": "vote_count.desc",
        "vote_count.gte": min_votes,
        "page": page,
    }
    data = _get(f"{TMDB_BASE}/discover/movie", params, api_key)
    if not data:
        return []
    return [m["id"] for m in data.get("results", [])]


def discover_all_ids(
    language: str, max_pages: int, min_votes: int, api_key: str,
) -> list[int]:
    """Phase 1: Collect all TMDB IDs from discover endpoint.

    Discover pages are sequential (page N+1 depends on page N existing).
    Fast: ~400 requests total, takes ~2 min.
    """
    all_ids = []
    for page in range(1, max_pages + 1):
        ids = discover_page(language, page, min_votes, api_key)
        if not ids:
            log.info(f"  Discover page {page}: no results, stopping")
            break
        all_ids.extend(ids)
        if page % 50 == 0:
            log.info(f"  Discovered {len(all_ids)} IDs through page {page}")
    return all_ids


# ---------------------------------------------------------------------------
# Detail fetch (concurrent)
# ---------------------------------------------------------------------------
def fetch_details(tmdb_id: int, api_key: str, language: str = "en-US") -> dict | None:
    """Fetch movie details + credits in one call."""
    params = {"language": language, "append_to_response": "credits"}
    return _get(f"{TMDB_BASE}/movie/{tmdb_id}", params, api_key)


def extract_record(detail: dict, translated: bool = False) -> dict:
    """Extract flat record from TMDB detail response."""
    genres = "|".join(g["name"] for g in detail.get("genres", []))
    release = detail.get("release_date", "") or ""
    year = int(release[:4]) if len(release) >= 4 else None

    credits = detail.get("credits", {})
    crew = credits.get("crew", [])
    directors = [c["name"] for c in crew if c.get("job") == "Director"]
    director = directors[0] if directors else None

    cast = credits.get("cast", [])
    cast_top5 = "|".join(c["name"] for c in cast[:5])

    return {
        "tmdb_id": detail["id"],
        "title": detail.get("title", ""),
        "original_title": detail.get("original_title", ""),
        "year": year,
        "genres": genres,
        "overview": detail.get("overview", ""),
        "director": director,
        "cast_top5": cast_top5,
        "rating": detail.get("vote_average"),
        "vote_count": detail.get("vote_count"),
        "runtime": detail.get("runtime"),
        "popularity": detail.get("popularity"),
        "poster_path": detail.get("poster_path", ""),
        "translated": translated,
    }


def translate_overview(text: str) -> str:
    """Translate Korean text to English using deep-translator."""
    if not text:
        return ""
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="ko", target="en").translate(text)
    except Exception as e:
        log.info(f"  Translation failed: {e}")
        return ""


def _fetch_one(tmdb_id: int, api_key: str, is_korean: bool) -> dict | None:
    """Fetch and extract one movie record. Used as ThreadPoolExecutor target."""
    detail = fetch_details(tmdb_id, api_key)
    if not detail:
        return None

    overview = detail.get("overview", "")
    translated = False

    if is_korean and not overview:
        kr_detail = fetch_details(tmdb_id, api_key, language="ko")
        if kr_detail:
            kr_overview = kr_detail.get("overview", "")
            if kr_overview:
                overview = translate_overview(kr_overview)
                translated = True
                detail["overview"] = overview

    return extract_record(detail, translated=translated)


def fetch_details_concurrent(
    tmdb_ids: list[int],
    seen_ids: set[int],
    api_key: str,
    is_korean: bool,
    raw_path: Path,
    existing: list[dict],
    workers: int = WORKERS,
) -> list[dict]:
    """Phase 2: Fetch movie details concurrently with rate limiting.

    Uses ThreadPoolExecutor with WORKERS threads, all sharing a single
    TokenBucket rate limiter. Checkpoints to disk every CHECKPOINT_EVERY
    movies.
    """
    # Filter out already-fetched IDs
    new_ids = [tid for tid in tmdb_ids if tid not in seen_ids]
    log.info(f"  {len(new_ids)} new movies to fetch ({len(seen_ids)} already cached)")

    if not new_ids:
        return existing

    movies = list(existing)
    fetched_count = 0

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_id = {
            executor.submit(_fetch_one, tid, api_key, is_korean): tid
            for tid in new_ids
        }

        for future in as_completed(future_to_id):
            tmdb_id = future_to_id[future]
            try:
                record = future.result()
            except Exception as e:
                log.info(f"  Error fetching {tmdb_id}: {e}")
                continue

            if record is None:
                continue

            movies.append(record)
            seen_ids.add(tmdb_id)
            fetched_count += 1

            if fetched_count % CHECKPOINT_EVERY == 0:
                raw_path.write_text(json.dumps(movies, ensure_ascii=False, indent=2))
                log.info(f"  Checkpoint: {fetched_count}/{len(new_ids)} fetched, "
                            f"{len(movies)} total")

    # Final save
    raw_path.write_text(json.dumps(movies, ensure_ascii=False, indent=2))
    log.info(f"  Done: {fetched_count} new movies fetched, {len(movies)} total")
    return movies


# ---------------------------------------------------------------------------
# Catalog fetch (two-phase)
# ---------------------------------------------------------------------------
def fetch_catalog(
    label: str,
    language: str,
    max_pages: int,
    min_votes: int,
    api_key: str,
    workers: int = WORKERS,
) -> list[dict]:
    """Two-phase catalog fetch: discover IDs, then concurrent details."""
    raw_path = DATA_RAW / f"{label}_movies.json"
    existing: list[dict] = []
    seen_ids: set[int] = set()

    if raw_path.exists():
        existing = json.loads(raw_path.read_text())
        seen_ids = {m["tmdb_id"] for m in existing}
        log.info(f"Resuming {label}: {len(existing)} movies already fetched")

    # Phase 1: Discover all IDs (sequential, fast)
    log.info(f"\n--- Phase 1: Discovering {label} movie IDs ---")
    all_ids = discover_all_ids(language, max_pages, min_votes, api_key)
    log.info(f"  Discovered {len(all_ids)} total IDs")

    # Phase 2: Fetch details concurrently
    log.info(f"\n--- Phase 2: Fetching {label} movie details ---")
    is_korean = language == "ko"
    movies = fetch_details_concurrent(
        all_ids, seen_ids, api_key, is_korean, raw_path, existing, workers,
    )

    log.info(f"\n{label} complete: {len(movies)} movies")
    return movies


# ---------------------------------------------------------------------------
# CSV export + quality report
# ---------------------------------------------------------------------------
def build_csv(movies: list[dict], out_path: Path) -> pd.DataFrame:
    """Convert raw records to clean CSV."""
    df = pd.DataFrame(movies, columns=REQUIRED_FIELDS)
    before = len(df)
    df = df[df["overview"].astype(str).str.strip().ne("")]
    after = len(df)
    if before != after:
        log.info(f"  Dropped {before - after} movies with empty overview")

    df = df.drop_duplicates(subset="tmdb_id")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"  Saved {len(df)} movies to {out_path.name}")
    return df


def quality_report(us_df: pd.DataFrame, kr_df: pd.DataFrame) -> None:
    """Print data quality summary."""
    DATA_METADATA.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, df in [("US", us_df), ("KR", kr_df)]:
        rows.append({
            "catalog": label,
            "total": len(df),
            "overview_missing_pct": f"{df['overview'].isna().mean():.1%}",
            "director_missing_pct": f"{df['director'].isna().mean():.1%}",
            "genres_missing_pct": f"{(df['genres'].astype(str).str.strip() == '').mean():.1%}",
            "year_range": f"{df['year'].min():.0f}-{df['year'].max():.0f}",
            "unique_genres": len(set(g for gs in df["genres"].dropna() for g in str(gs).split("|") if g)),
        })
        if "translated" in df.columns:
            rows[-1]["translated_pct"] = f"{df['translated'].sum() / len(df):.1%}"

    report = pd.DataFrame(rows)
    report.to_csv(DATA_METADATA / "data_quality.csv", index=False)
    log.info("--- Data Quality ---")
    log.info(report.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Download movie data from TMDB")
    parser.add_argument("--skip-us", action="store_true", help="Skip US catalog fetch")
    parser.add_argument("--skip-kr", action="store_true", help="Skip KR catalog fetch")
    parser.add_argument("--max-pages-us", type=int, default=US_MAX_PAGES)
    parser.add_argument("--max-pages-kr", type=int, default=KR_MAX_PAGES)
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help="Number of concurrent fetch threads")
    args = parser.parse_args()

    workers = args.workers
    api_key = load_api_key()

    us_movies, kr_movies = [], []

    if not args.skip_us:
        log.info("=== Fetching US movies ===")
        us_movies = fetch_catalog("us", "en", args.max_pages_us, US_MIN_VOTES, api_key, workers)
    elif (DATA_RAW / "us_movies.json").exists():
        us_movies = json.loads((DATA_RAW / "us_movies.json").read_text())

    if not args.skip_kr:
        log.info("=== Fetching KR movies ===")
        kr_movies = fetch_catalog("kr", "ko", args.max_pages_kr, KR_MIN_VOTES, api_key, workers)
    elif (DATA_RAW / "kr_movies.json").exists():
        kr_movies = json.loads((DATA_RAW / "kr_movies.json").read_text())

    us_df = kr_df = pd.DataFrame()
    if us_movies:
        us_df = build_csv(us_movies, DATA_PROCESSED / "us_movies.csv")
    if kr_movies:
        kr_df = build_csv(kr_movies, DATA_PROCESSED / "kr_movies.csv")

    if len(us_df) and len(kr_df):
        quality_report(us_df, kr_df)


if __name__ == "__main__":
    main()
