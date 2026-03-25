"""Semi-automated curation of US-KR evaluation movie pairs.

Generates candidate pairs from:
1. Known remakes / adaptations (hardcoded seed list)
2. Genre + theme matching via TMDB search
3. Director crossovers (directors who worked in both industries)

Outputs candidates to data/eval/gold_pairs_candidates.csv for manual review.
After review, save final pairs to data/eval/gold_pairs.csv.
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_PROCESSED = SCRIPT_DIR / "data" / "processed"
DATA_EVAL = SCRIPT_DIR / "data" / "eval"

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_DELAY_SEC = 0.3

# ---------------------------------------------------------------------------
# Seed pairs: known remakes and thematic matches
# Relevance: 3 = remake/direct adaptation, 2 = thematic match, 1 = genre match
# ---------------------------------------------------------------------------
SEED_PAIRS = [
    # Remakes / direct adaptations (relevance 3)
    {"us_title": "Oldboy", "kr_title": "Oldeuboi", "relevance": 3,
     "type": "remake", "genre": "Thriller", "notes": "Spike Lee remake of Park Chan-wook original"},
    {"us_title": "The Lake House", "kr_title": "Siworae", "relevance": 3,
     "type": "remake", "genre": "Romance", "notes": "US remake of Korean Il Mare"},
    {"us_title": "The Uninvited", "kr_title": "Janghwa, Hongryeon", "relevance": 3,
     "type": "remake", "genre": "Horror", "notes": "Remake of A Tale of Two Sisters"},
    {"us_title": "The Last Stand", "kr_title": "The Good, the Bad, the Weird", "relevance": 2,
     "type": "thematic", "genre": "Action", "notes": "Kim Jee-woon directed both"},

    # Thematic matches (relevance 2)
    {"us_title": "Parasite", "kr_title": "Parasite", "relevance": 2,
     "type": "thematic", "genre": "Thriller",
     "notes": "Cross-listed; use as anchor for class-divide thrillers"},
    {"us_title": "The Departed", "kr_title": "New World", "relevance": 2,
     "type": "thematic", "genre": "Crime", "notes": "Undercover cop in crime syndicate"},
    {"us_title": "Inception", "kr_title": "Lucid Dream", "relevance": 2,
     "type": "thematic", "genre": "Sci-Fi", "notes": "Dream manipulation thriller"},
    {"us_title": "Train to Busan", "kr_title": "Train to Busan", "relevance": 2,
     "type": "thematic", "genre": "Horror",
     "notes": "Cross-listed; anchor for zombie/survival horror"},
    {"us_title": "Zodiac", "kr_title": "Memories of Murder", "relevance": 2,
     "type": "thematic", "genre": "Crime", "notes": "Serial killer investigation procedural"},
    {"us_title": "Gone Girl", "kr_title": "Bedevilled", "relevance": 2,
     "type": "thematic", "genre": "Thriller", "notes": "Psychological revenge thriller"},
    {"us_title": "The Shawshank Redemption", "kr_title": "A Bittersweet Life", "relevance": 2,
     "type": "thematic", "genre": "Drama", "notes": "Betrayal and revenge in confined world"},
    {"us_title": "Joker", "kr_title": "Burning", "relevance": 2,
     "type": "thematic", "genre": "Drama", "notes": "Class resentment psychological drama"},
    {"us_title": "Se7en", "kr_title": "I Saw the Devil", "relevance": 2,
     "type": "thematic", "genre": "Thriller", "notes": "Dark cat-and-mouse thriller"},
    {"us_title": "The Silence of the Lambs", "kr_title": "The Chaser", "relevance": 2,
     "type": "thematic", "genre": "Thriller", "notes": "Serial killer pursuit thriller"},
    {"us_title": "Gravity", "kr_title": "Space Sweepers", "relevance": 1,
     "type": "genre", "genre": "Sci-Fi", "notes": "Space survival/adventure"},
    {"us_title": "John Wick", "kr_title": "The Man from Nowhere", "relevance": 2,
     "type": "thematic", "genre": "Action", "notes": "Lone protector revenge action"},
    {"us_title": "Taken", "kr_title": "Ajeossi", "relevance": 2,
     "type": "thematic", "genre": "Action", "notes": "Rescue mission action thriller"},
    {"us_title": "Whiplash", "kr_title": "The King of Jokgu", "relevance": 1,
     "type": "genre", "genre": "Drama", "notes": "Intense mentor-student competition"},
    {"us_title": "The Notebook", "kr_title": "My Sassy Girl", "relevance": 1,
     "type": "genre", "genre": "Romance", "notes": "Iconic romantic drama"},
    {"us_title": "Get Out", "kr_title": "The Wailing", "relevance": 2,
     "type": "thematic", "genre": "Horror", "notes": "Social horror with outsider protagonist"},
]


# ---------------------------------------------------------------------------
# TMDB helpers
# ---------------------------------------------------------------------------
def load_api_key() -> str:
    load_dotenv(SCRIPT_DIR / ".env")
    key = os.getenv("TMDB_API_KEY")
    if not key or key == "your_api_key_here":
        raise SystemExit("Set TMDB_API_KEY in .env")
    return key


def search_tmdb(title: str, language: str, api_key: str) -> dict | None:
    """Search TMDB for a movie by title, return first result."""
    time.sleep(TMDB_DELAY_SEC)
    params = {
        "api_key": api_key,
        "query": title,
        "language": "en-US",
    }
    if language == "ko":
        params["with_original_language"] = "ko"
    resp = requests.get(f"{TMDB_BASE}/search/movie", params=params, timeout=30)
    if resp.status_code != 200:
        return None
    results = resp.json().get("results", [])
    return results[0] if results else None


# ---------------------------------------------------------------------------
# Pair resolution
# ---------------------------------------------------------------------------
def resolve_seed_pairs(api_key: str) -> pd.DataFrame:
    """Resolve seed pairs to TMDB IDs."""
    rows = []
    for pair in SEED_PAIRS:
        us_match = search_tmdb(pair["us_title"], "en", api_key)
        kr_match = search_tmdb(pair["kr_title"], "ko", api_key)

        # Fallback: search KR title without language filter
        if not kr_match:
            kr_match = search_tmdb(pair["kr_title"], "en", api_key)

        us_id = us_match["id"] if us_match else None
        kr_id = kr_match["id"] if kr_match else None

        rows.append({
            "us_tmdb_id": us_id,
            "kr_tmdb_id": kr_id,
            "us_title": pair["us_title"],
            "kr_title": pair["kr_title"],
            "relevance": pair["relevance"],
            "relationship_type": pair["type"],
            "genre": pair["genre"],
            "notes": pair["notes"],
            "resolved": bool(us_id and kr_id),
        })
        print(f"  {pair['us_title']} ↔ {pair['kr_title']}: "
              f"{'OK' if us_id and kr_id else 'MISSING'}")

    return pd.DataFrame(rows)


def find_genre_matches(
    us_df: pd.DataFrame, kr_df: pd.DataFrame, per_genre: int = 3,
) -> pd.DataFrame:
    """Find additional genre-matched pairs from processed data."""
    rows = []
    # Get genre list from US movies
    all_genres = set()
    for gs in us_df["genres"].dropna():
        all_genres.update(str(gs).split("|"))
    all_genres -= {""}

    for genre in sorted(all_genres):
        us_genre = us_df[us_df["genres"].str.contains(genre, na=False)]
        kr_genre = kr_df[kr_df["genres"].str.contains(genre, na=False)]

        if us_genre.empty or kr_genre.empty:
            continue

        # Pick top-rated US movies and top-rated KR movies
        us_top = us_genre.nlargest(per_genre, "rating")
        kr_top = kr_genre.nlargest(per_genre, "rating")

        for _, us_row in us_top.iterrows():
            for _, kr_row in kr_top.iterrows():
                rows.append({
                    "us_tmdb_id": int(us_row["tmdb_id"]),
                    "kr_tmdb_id": int(kr_row["tmdb_id"]),
                    "us_title": us_row["title"],
                    "kr_title": kr_row["title"],
                    "relevance": 1,
                    "relationship_type": "genre",
                    "genre": genre,
                    "notes": f"Top-rated {genre} match",
                    "resolved": True,
                })

    df = pd.DataFrame(rows).drop_duplicates(subset=["us_tmdb_id", "kr_tmdb_id"])
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Curate evaluation movie pairs")
    parser.add_argument("--seeds-only", action="store_true",
                        help="Only resolve seed pairs (no genre expansion)")
    parser.add_argument("--per-genre", type=int, default=3,
                        help="Genre-match pairs to generate per genre")
    args = parser.parse_args()

    api_key = load_api_key()

    print("=== Resolving seed pairs ===")
    seed_df = resolve_seed_pairs(api_key)
    resolved = seed_df[seed_df["resolved"]].drop(columns=["resolved"])
    print(f"  Resolved {len(resolved)}/{len(seed_df)} seed pairs")

    if not args.seeds_only:
        # Load processed data for genre matching
        us_path = DATA_PROCESSED / "us_movies.csv"
        kr_path = DATA_PROCESSED / "kr_movies.csv"
        if us_path.exists() and kr_path.exists():
            print("\n=== Finding genre matches ===")
            us_df = pd.read_csv(us_path)
            kr_df = pd.read_csv(kr_path)
            genre_df = find_genre_matches(us_df, kr_df, per_genre=args.per_genre)
            print(f"  Found {len(genre_df)} genre-matched candidates")

            # Combine, deduplicate
            combined = pd.concat([resolved, genre_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["us_tmdb_id", "kr_tmdb_id"])
        else:
            print("  Processed data not found, skipping genre matching.")
            print("  Run data_ingestion.py first.")
            combined = resolved
    else:
        combined = resolved

    # Save candidates
    DATA_EVAL.mkdir(parents=True, exist_ok=True)
    out_path = DATA_EVAL / "gold_pairs_candidates.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved {len(combined)} candidates to {out_path.name}")
    print("Review and save final pairs to data/eval/gold_pairs.csv")


if __name__ == "__main__":
    main()
