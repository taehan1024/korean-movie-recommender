"""Fetch TMDB keywords for existing movies and add to processed CSVs.

Lightweight script that adds a 'keywords' column to us_movies.csv and
kr_movies.csv without re-ingesting all movie data. Uses the same token
bucket rate limiter from data_ingestion.py.

Usage:
    python fetch_keywords.py          # fetch for both catalogs
    python fetch_keywords.py --skip-us  # fetch only KR
    python fetch_keywords.py --skip-kr  # fetch only US
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
DATA_PROCESSED = SCRIPT_DIR / "data" / "processed"

TMDB_BASE = "https://api.themoviedb.org/3"
RATE_LIMIT = 35
RATE_WINDOW_SEC = 10.0
WORKERS = 4
CHECKPOINT_EVERY = 200
MAX_RETRIES = 3
MAX_BACKOFF_SEC = 60

_bucket = TokenBucket(RATE_LIMIT, RATE_WINDOW_SEC)


def load_api_key() -> str:
    load_dotenv(SCRIPT_DIR / ".env")
    key = os.getenv("TMDB_API_KEY")
    if not key:
        raise RuntimeError("TMDB_API_KEY not set. Add it to .env")
    return key


def _get(url: str, params: dict, api_key: str) -> dict | None:
    params["api_key"] = api_key
    for attempt in range(MAX_RETRIES):
        _bucket.acquire()
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = min(2 ** attempt * 2, MAX_BACKOFF_SEC)
                time.sleep(wait)
                continue
        except requests.RequestException:
            pass
    return None


def fetch_keywords(tmdb_id: int, api_key: str) -> str:
    """Fetch keywords for one movie, return pipe-delimited string."""
    data = _get(f"{TMDB_BASE}/movie/{tmdb_id}/keywords", {}, api_key)
    if data and "keywords" in data:
        return "|".join(k["name"] for k in data["keywords"])
    return ""


def fetch_keywords_for_catalog(csv_path: Path, api_key: str, label: str) -> None:
    """Add keywords column to a catalog CSV."""
    df = pd.read_csv(csv_path)

    if "keywords" in df.columns:
        existing = df["keywords"].notna() & (df["keywords"] != "")
        log.info(f"  {label}: {existing.sum()}/{len(df)} already have keywords")
        ids_to_fetch = df[~existing]["tmdb_id"].tolist()
    else:
        df["keywords"] = ""
        ids_to_fetch = df["tmdb_id"].tolist()

    if not ids_to_fetch:
        log.info(f"  {label}: all movies already have keywords, skipping")
        return

    log.info(f"  {label}: fetching keywords for {len(ids_to_fetch)} movies...")
    results = {}
    done = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(fetch_keywords, tid, api_key): tid
            for tid in ids_to_fetch
        }
        for future in as_completed(futures):
            tid = futures[future]
            try:
                kw = future.result()
                results[tid] = kw
            except Exception as e:
                log.warning(f"  Error fetching {tid}: {e}")
                results[tid] = ""
            done += 1
            if done % CHECKPOINT_EVERY == 0:
                log.info(f"    {done}/{len(ids_to_fetch)} done")

    # Merge results
    for tid, kw in results.items():
        df.loc[df["tmdb_id"] == tid, "keywords"] = kw

    # Save
    df.to_csv(csv_path, index=False)
    has_kw = df["keywords"].notna() & (df["keywords"] != "")
    log.info(f"  {label}: {has_kw.sum()}/{len(df)} movies now have keywords. Saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch TMDB keywords for existing movies")
    parser.add_argument("--skip-us", action="store_true")
    parser.add_argument("--skip-kr", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    api_key = load_api_key()
    log.info("Fetching TMDB keywords...")

    if not args.skip_us:
        fetch_keywords_for_catalog(DATA_PROCESSED / "us_movies.csv", api_key, "US")
    if not args.skip_kr:
        fetch_keywords_for_catalog(DATA_PROCESSED / "kr_movies.csv", api_key, "KR")

    log.info("Done.")


if __name__ == "__main__":
    main()
