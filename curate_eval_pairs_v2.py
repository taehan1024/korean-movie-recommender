"""V2 gold evaluation set: fixed IDs, expanded seed pairs, embedding-filtered genre matches.

Changes from v1 (curate_eval_pairs.py):
- Hardcoded TMDB IDs (verified against catalog) instead of title search
- Fixed 8 broken seed pairs (wrong KR IDs) and removed 2 invalid pairs
- Added ~30 new thematic/same-director pairs
- Embedding-filtered genre matches replace rating-filtered ones
- Pooling candidate generation for future manual labeling

Usage:
    python curate_eval_pairs_v2.py --seeds          # Seed pairs only
    python curate_eval_pairs_v2.py --embed-genre     # Seed + embedding-filtered genre
    python curate_eval_pairs_v2.py --all             # Everything (default)
    python curate_eval_pairs_v2.py --pool            # Generate pooling candidates
    python curate_eval_pairs_v2.py --stats           # Print gold set stats
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2 import load_all_features, load_dataframes

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_EVAL = SCRIPT_DIR / "data" / "eval"

# ---------------------------------------------------------------------------
# Seed pairs: all TMDB IDs verified against us_movies.csv and kr_movies.csv
# Relevance: 3 = remake, 2 = thematic/same-director, 1 = genre match
# ---------------------------------------------------------------------------
SEED_PAIRS_V2 = [
    # === Remakes (relevance=3) ===
    {"us_id": 87516, "kr_id": 670, "us_title": "Oldboy", "kr_title": "Oldboy",
     "relevance": 3, "type": "remake", "genre": "Thriller",
     "notes": "Spike Lee (2013) remake of Park Chan-wook (2003)"},
    {"us_id": 2044, "kr_id": 12650, "us_title": "The Lake House", "kr_title": "Il Mare",
     "relevance": 3, "type": "remake", "genre": "Romance",
     "notes": "US (2006) remake of Korean Il Mare (2000)"},
    {"us_id": 14254, "kr_id": 4552, "us_title": "The Uninvited", "kr_title": "A Tale of Two Sisters",
     "relevance": 3, "type": "remake", "genre": "Horror",
     "notes": "US (2009) remake of A Tale of Two Sisters (2003)"},

    # === Same-director pairs (relevance=2) ===
    # Park Chan-wook: Stoker (US) ↔ Korean filmography
    {"us_id": 86825, "kr_id": 670, "us_title": "Stoker", "kr_title": "Oldboy",
     "relevance": 2, "type": "same_director", "genre": "Thriller",
     "notes": "Park Chan-wook: psychological thriller, family secrets"},
    {"us_id": 86825, "kr_id": 290098, "us_title": "Stoker", "kr_title": "The Handmaiden",
     "relevance": 2, "type": "same_director", "genre": "Thriller",
     "notes": "Park Chan-wook: gothic psychosexual thriller"},
    {"us_id": 86825, "kr_id": 705996, "us_title": "Stoker", "kr_title": "Decision to Leave",
     "relevance": 2, "type": "same_director", "genre": "Thriller",
     "notes": "Park Chan-wook: obsessive relationship thriller"},
    {"us_id": 86825, "kr_id": 22536, "us_title": "Stoker", "kr_title": "Thirst",
     "relevance": 2, "type": "same_director", "genre": "Horror",
     "notes": "Park Chan-wook: dark desire, transformation"},

    # Kim Jee-woon: The Last Stand (US) ↔ Korean filmography
    {"us_id": 76640, "kr_id": 15067, "us_title": "The Last Stand", "kr_title": "The Good, the Bad, the Weird",
     "relevance": 2, "type": "same_director", "genre": "Action",
     "notes": "Kim Jee-woon directed both"},
    {"us_id": 76640, "kr_id": 49797, "us_title": "The Last Stand", "kr_title": "I Saw the Devil",
     "relevance": 2, "type": "same_director", "genre": "Thriller",
     "notes": "Kim Jee-woon: intense thriller"},
    {"us_id": 76640, "kr_id": 11344, "us_title": "The Last Stand", "kr_title": "A Bittersweet Life",
     "relevance": 2, "type": "same_director", "genre": "Action",
     "notes": "Kim Jee-woon: stylish action thriller"},
    {"us_id": 76640, "kr_id": 363579, "us_title": "The Last Stand", "kr_title": "The Age of Shadows",
     "relevance": 2, "type": "same_director", "genre": "Thriller",
     "notes": "Kim Jee-woon: period action thriller"},

    # Bong Joon-ho: Okja (US) ↔ Korean filmography
    {"us_id": 387426, "kr_id": 1255, "us_title": "Okja", "kr_title": "The Host",
     "relevance": 2, "type": "same_director", "genre": "Sci-Fi",
     "notes": "Bong Joon-ho: creature + social commentary"},
    {"us_id": 387426, "kr_id": 11423, "us_title": "Okja", "kr_title": "Memories of Murder",
     "relevance": 2, "type": "same_director", "genre": "Crime",
     "notes": "Bong Joon-ho: social commentary"},
    {"us_id": 387426, "kr_id": 30018, "us_title": "Okja", "kr_title": "Mother",
     "relevance": 2, "type": "same_director", "genre": "Drama",
     "notes": "Bong Joon-ho: protagonist fights system for loved one"},
    {"us_id": 387426, "kr_id": 496243, "us_title": "Okja", "kr_title": "Parasite",
     "relevance": 2, "type": "same_director", "genre": "Thriller",
     "notes": "Bong Joon-ho: class divide"},

    # Bong Joon-ho: Mickey 17 (US) ↔ Korean filmography
    {"us_id": 696506, "kr_id": 110415, "us_title": "Mickey 17", "kr_title": "Snowpiercer",
     "relevance": 2, "type": "same_director", "genre": "Sci-Fi",
     "notes": "Bong Joon-ho: sci-fi class commentary"},
    {"us_id": 696506, "kr_id": 496243, "us_title": "Mickey 17", "kr_title": "Parasite",
     "relevance": 2, "type": "same_director", "genre": "Thriller",
     "notes": "Bong Joon-ho: class divide"},

    # === Thematic pairs (relevance=2) — different directors, similar themes ===
    {"us_id": 1422, "kr_id": 165213, "us_title": "The Departed", "kr_title": "New World",
     "relevance": 2, "type": "thematic", "genre": "Crime",
     "notes": "Undercover cop in crime syndicate"},
    {"us_id": 27205, "kr_id": 436994, "us_title": "Inception", "kr_title": "Lucid Dream",
     "relevance": 2, "type": "thematic", "genre": "Sci-Fi",
     "notes": "Dream manipulation thriller"},
    {"us_id": 1949, "kr_id": 11423, "us_title": "Zodiac", "kr_title": "Memories of Murder",
     "relevance": 2, "type": "thematic", "genre": "Crime",
     "notes": "Serial killer investigation procedural"},
    {"us_id": 210577, "kr_id": 59421, "us_title": "Gone Girl", "kr_title": "Bedevilled",
     "relevance": 2, "type": "thematic", "genre": "Thriller",
     "notes": "Psychological revenge thriller"},
    {"us_id": 278, "kr_id": 11344, "us_title": "The Shawshank Redemption", "kr_title": "A Bittersweet Life",
     "relevance": 2, "type": "thematic", "genre": "Drama",
     "notes": "Betrayal and revenge in confined world"},
    {"us_id": 475557, "kr_id": 491584, "us_title": "Joker", "kr_title": "Burning",
     "relevance": 2, "type": "thematic", "genre": "Drama",
     "notes": "Class resentment psychological drama"},
    {"us_id": 807, "kr_id": 49797, "us_title": "Se7en", "kr_title": "I Saw the Devil",
     "relevance": 2, "type": "thematic", "genre": "Thriller",
     "notes": "Dark cat-and-mouse thriller"},
    {"us_id": 274, "kr_id": 13855, "us_title": "The Silence of the Lambs", "kr_title": "The Chaser",
     "relevance": 2, "type": "thematic", "genre": "Thriller",
     "notes": "Serial killer pursuit thriller"},
    {"us_id": 245891, "kr_id": 51608, "us_title": "John Wick", "kr_title": "The Man from Nowhere",
     "relevance": 2, "type": "thematic", "genre": "Action",
     "notes": "Lone protector revenge action"},
    {"us_id": 8681, "kr_id": 51608, "us_title": "Taken", "kr_title": "Ajeossi",
     "relevance": 2, "type": "thematic", "genre": "Action",
     "notes": "Rescue mission action thriller"},
    {"us_id": 419430, "kr_id": 293670, "us_title": "Get Out", "kr_title": "The Wailing",
     "relevance": 2, "type": "thematic", "genre": "Horror",
     "notes": "Social horror with outsider protagonist"},
    {"us_id": 146233, "kr_id": 11423, "us_title": "Prisoners", "kr_title": "Memories of Murder",
     "relevance": 2, "type": "thematic", "genre": "Crime",
     "notes": "Dark investigation procedural, moral ambiguity"},
    {"us_id": 242582, "kr_id": 13855, "us_title": "Nightcrawler", "kr_title": "The Chaser",
     "relevance": 2, "type": "thematic", "genre": "Thriller",
     "notes": "Nocturnal urban crime thriller"},
    {"us_id": 64690, "kr_id": 11344, "us_title": "Drive", "kr_title": "A Bittersweet Life",
     "relevance": 2, "type": "thematic", "genre": "Crime",
     "notes": "Stylish loner crime thriller"},
    {"us_id": 493922, "kr_id": 4552, "us_title": "Hereditary", "kr_title": "A Tale of Two Sisters",
     "relevance": 2, "type": "thematic", "genre": "Horror",
     "notes": "Family horror, psychological"},
    {"us_id": 530385, "kr_id": 293670, "us_title": "Midsommar", "kr_title": "The Wailing",
     "relevance": 2, "type": "thematic", "genre": "Horror",
     "notes": "Folk horror, outsider in community"},
    {"us_id": 273481, "kr_id": 57361, "us_title": "Sicario", "kr_title": "The Yellow Sea",
     "relevance": 2, "type": "thematic", "genre": "Crime",
     "notes": "Border crime violence, relentless pursuit"},
    {"us_id": 388, "kr_id": 124157, "us_title": "Inside Man", "kr_title": "The Thieves",
     "relevance": 2, "type": "thematic", "genre": "Crime",
     "notes": "Heist thriller"},
    {"us_id": 161, "kr_id": 124157, "us_title": "Ocean's Eleven", "kr_title": "The Thieves",
     "relevance": 2, "type": "thematic", "genre": "Crime",
     "notes": "Ensemble heist"},
    {"us_id": 6977, "kr_id": 57361, "us_title": "No Country for Old Men", "kr_title": "The Yellow Sea",
     "relevance": 2, "type": "thematic", "genre": "Crime",
     "notes": "Violent crime chase, unstoppable pursuer"},
    {"us_id": 264660, "kr_id": 586047, "us_title": "Ex Machina", "kr_title": "Seobok",
     "relevance": 2, "type": "thematic", "genre": "Sci-Fi",
     "notes": "Sci-fi — artificial being, ethics of creation"},
    {"us_id": 458723, "kr_id": 496243, "us_title": "Us", "kr_title": "Parasite",
     "relevance": 2, "type": "thematic", "genre": "Thriller",
     "notes": "Class divide horror/thriller"},
    {"us_id": 487558, "kr_id": 437068, "us_title": "BlacKkKlansman", "kr_title": "A Taxi Driver",
     "relevance": 2, "type": "thematic", "genre": "Drama",
     "notes": "Political activism period drama"},
    {"us_id": 275, "kr_id": 30018, "us_title": "Fargo", "kr_title": "Mother",
     "relevance": 2, "type": "thematic", "genre": "Crime",
     "notes": "Dark crime, unlikely protagonist investigation"},
    {"us_id": 500, "kr_id": 293413, "us_title": "Reservoir Dogs", "kr_title": "Inside Men",
     "relevance": 2, "type": "thematic", "genre": "Crime",
     "notes": "Crime betrayal ensemble"},
    {"us_id": 329865, "kr_id": 581389, "us_title": "Arrival", "kr_title": "Space Sweepers",
     "relevance": 2, "type": "thematic", "genre": "Sci-Fi",
     "notes": "Sci-fi first contact / space"},
    {"us_id": 11036, "kr_id": 12650, "us_title": "The Notebook", "kr_title": "Il Mare",
     "relevance": 2, "type": "thematic", "genre": "Romance",
     "notes": "Time-crossing romance"},

    # === Genre matches from v1 seed (relevance=1) ===
    {"us_id": 49047, "kr_id": 581389, "us_title": "Gravity", "kr_title": "Space Sweepers",
     "relevance": 1, "type": "genre", "genre": "Sci-Fi",
     "notes": "Space survival/adventure"},
    {"us_id": 244786, "kr_id": 199584, "us_title": "Whiplash", "kr_title": "Secretly, Greatly",
     "relevance": 1, "type": "genre", "genre": "Drama",
     "notes": "Intense pressure drama"},
    {"us_id": 11036, "kr_id": 11178, "us_title": "The Notebook", "kr_title": "My Sassy Girl",
     "relevance": 1, "type": "genre", "genre": "Romance",
     "notes": "Iconic romantic drama"},
]


# ---------------------------------------------------------------------------
# Strategy 1: Seed pairs (hardcoded IDs, no API calls)
# ---------------------------------------------------------------------------
def get_seed_pairs() -> pd.DataFrame:
    """Return seed pairs as DataFrame with verified TMDB IDs."""
    rows = []
    for p in SEED_PAIRS_V2:
        rows.append({
            "us_tmdb_id": p["us_id"],
            "kr_tmdb_id": p["kr_id"],
            "us_title": p["us_title"],
            "kr_title": p["kr_title"],
            "relevance": p["relevance"],
            "relationship_type": p["type"],
            "genre": p["genre"],
            "notes": p["notes"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Strategy 2: Embedding-filtered genre matches
# ---------------------------------------------------------------------------
def find_embedding_genre_matches(
    us_df: pd.DataFrame,
    kr_df: pd.DataFrame,
    features: dict,
    per_genre_us: int = 5,
    top_similar: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Find genre-matched pairs ranked by embedding similarity (not rating).

    For each genre:
    1. Sample per_genre_us US movies (stratified by popularity)
    2. Find KR movies sharing that genre
    3. Rank by embedding cosine similarity
    4. Take top_similar most similar as genre matches (relevance=1)
    """
    rng = np.random.RandomState(seed)
    emb_us = features["emb_us"]
    emb_kr = features["emb_kr"]

    # Build genre sets
    all_genres = set()
    for gs in us_df["genres"].dropna():
        all_genres.update(str(gs).split("|"))
    all_genres -= {""}

    rows = []
    for genre in sorted(all_genres):
        us_mask = us_df["genres"].str.contains(genre, na=False)
        kr_mask = kr_df["genres"].str.contains(genre, na=False)

        us_genre = us_df[us_mask]
        kr_genre = kr_df[kr_mask]

        if len(us_genre) < 3 or len(kr_genre) < 3:
            continue

        # Stratified sample: 2 popular, 2 mid, 1 long-tail
        sorted_by_pop = us_genre.sort_values("popularity", ascending=False)
        n = len(sorted_by_pop)
        tercile = max(1, n // 3)

        popular = sorted_by_pop.iloc[:tercile]
        mid = sorted_by_pop.iloc[tercile:2 * tercile]
        tail = sorted_by_pop.iloc[2 * tercile:]

        samples = []
        for pool, count in [(popular, 2), (mid, 2), (tail, 1)]:
            if len(pool) >= count:
                samples.append(pool.sample(count, random_state=rng))
            elif len(pool) > 0:
                samples.append(pool.sample(min(count, len(pool)), random_state=rng))

        if not samples:
            continue

        us_sample = pd.concat(samples).drop_duplicates(subset=["tmdb_id"])

        # Get KR indices for embedding lookup
        kr_indices = kr_genre.index.tolist()
        kr_embs = emb_kr[kr_indices]

        for _, us_row in us_sample.iterrows():
            us_idx = us_row.name  # DataFrame index = feature matrix index
            us_emb = emb_us[us_idx].reshape(1, -1)

            # Cosine similarity (embeddings are L2-normalized)
            sims = (kr_embs @ us_emb.T).flatten()
            top_kr_local = sims.argsort()[-top_similar:][::-1]

            for local_idx in top_kr_local:
                kr_idx = kr_indices[local_idx]
                kr_row = kr_df.iloc[kr_idx]
                rows.append({
                    "us_tmdb_id": int(us_row["tmdb_id"]),
                    "kr_tmdb_id": int(kr_row["tmdb_id"]),
                    "us_title": us_row["title"],
                    "kr_title": kr_row["title"],
                    "relevance": 1,
                    "relationship_type": "genre",
                    "genre": genre,
                    "notes": f"Embedding-filtered {genre} match (sim={sims[local_idx]:.3f})",
                })

    df = pd.DataFrame(rows).drop_duplicates(subset=["us_tmdb_id", "kr_tmdb_id"])
    return df


# ---------------------------------------------------------------------------
# Strategy 3: Pooling candidates for manual review
# ---------------------------------------------------------------------------
def generate_pool_candidates(
    us_df: pd.DataFrame,
    kr_df: pd.DataFrame,
    features: dict,
    n_queries: int = 100,
    top_k: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate model-ranked candidates for human labeling.

    Selects diverse US movies, runs hybrid model, outputs top-k KR candidates
    with relevance=NaN for manual review.
    """
    from models_v2 import get_recommendations

    rng = np.random.RandomState(seed)

    # Stratified sample of US movies
    sorted_by_pop = us_df.sort_values("popularity", ascending=False)
    n = len(sorted_by_pop)
    tercile = max(1, n // 3)

    popular = sorted_by_pop.iloc[:tercile].sample(min(40, tercile), random_state=rng)
    mid = sorted_by_pop.iloc[tercile:2 * tercile].sample(min(30, tercile), random_state=rng)
    tail = sorted_by_pop.iloc[2 * tercile:].sample(min(30, n - 2 * tercile), random_state=rng)

    us_sample = pd.concat([popular, mid, tail]).drop_duplicates(subset=["tmdb_id"]).head(n_queries)

    rows = []
    for _, us_row in us_sample.iterrows():
        recs = get_recommendations(
            us_row["title"], "hybrid", us_df, kr_df, features, top_k=top_k
        )
        if recs.empty:
            continue

        for rank, (_, kr_row) in enumerate(recs.iterrows(), 1):
            rows.append({
                "us_tmdb_id": int(us_row["tmdb_id"]),
                "kr_tmdb_id": int(kr_row["tmdb_id"]),
                "us_title": us_row["title"],
                "kr_title": kr_row["title"],
                "rank": rank,
                "hybrid_score": round(kr_row["score"], 4),
                "relevance": "",
                "relationship_type": "",
                "genre": str(us_row.get("genres", "")).split("|")[0],
                "notes": "",
                "reviewed": False,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Merge + deduplicate + validate
# ---------------------------------------------------------------------------
def merge_sources(
    seed_df: pd.DataFrame,
    genre_df: pd.DataFrame,
    us_df: pd.DataFrame,
    kr_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge seed and genre pairs, deduplicate, validate IDs."""
    combined = pd.concat([seed_df, genre_df], ignore_index=True)

    # Deduplicate: keep highest relevance when duplicate (us_id, kr_id)
    combined = combined.sort_values("relevance", ascending=False)
    combined = combined.drop_duplicates(subset=["us_tmdb_id", "kr_tmdb_id"], keep="first")

    # Validate all IDs exist in catalogs
    valid_us = combined["us_tmdb_id"].isin(us_df["tmdb_id"])
    valid_kr = combined["kr_tmdb_id"].isin(kr_df["tmdb_id"])
    invalid = combined[~(valid_us & valid_kr)]
    if len(invalid) > 0:
        print(f"  WARNING: {len(invalid)} pairs have invalid TMDB IDs, removing:")
        for _, r in invalid.iterrows():
            print(f"    {r['us_title']} ({r['us_tmdb_id']}) <-> {r['kr_title']} ({r['kr_tmdb_id']})")
        combined = combined[valid_us & valid_kr]

    return combined.sort_values(["relevance", "us_title"], ascending=[False, True]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def print_stats(gold: pd.DataFrame) -> None:
    """Print gold set distribution statistics."""
    print(f"\n{'='*60}")
    print(f"Gold Set Statistics")
    print(f"{'='*60}")
    print(f"Total pairs: {len(gold)}")
    print(f"Unique US queries: {gold['us_tmdb_id'].nunique()}")
    print(f"Unique KR movies: {gold['kr_tmdb_id'].nunique()}")
    print(f"Avg gold matches per US query: {len(gold) / gold['us_tmdb_id'].nunique():.1f}")

    print(f"\nRelevance distribution:")
    for rel in sorted(gold["relevance"].unique(), reverse=True):
        count = (gold["relevance"] == rel).sum()
        label = {3: "remake", 2: "thematic/director", 1: "genre"}.get(rel, f"rel_{rel}")
        print(f"  {label} ({rel}): {count}")

    print(f"\nRelationship types:")
    for rtype, count in gold["relationship_type"].value_counts().items():
        print(f"  {rtype}: {count}")

    print(f"\nGenre coverage:")
    genres = set()
    for g in gold["genre"].dropna():
        genres.add(str(g))
    print(f"  {len(genres)} genres: {', '.join(sorted(genres))}")

    # Top US movies by number of gold matches
    print(f"\nTop US movies by gold matches:")
    top = gold.groupby("us_title").size().sort_values(ascending=False).head(10)
    for title, count in top.items():
        print(f"  {title}: {count} matches")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Curate v2 gold evaluation pairs")
    parser.add_argument("--seeds", action="store_true", help="Seed pairs only")
    parser.add_argument("--embed-genre", action="store_true", help="Seed + embedding-filtered genre")
    parser.add_argument("--pool", action="store_true", help="Generate pooling candidates for review")
    parser.add_argument("--all", action="store_true", help="Everything except pooling (default)")
    parser.add_argument("--stats", action="store_true", help="Print stats for existing gold set")
    parser.add_argument("--per-genre-us", type=int, default=5, help="US movies per genre for embedding match")
    parser.add_argument("--top-similar", type=int, default=3, help="KR matches per US movie per genre")
    args = parser.parse_args()

    # Default to --all if no flags
    if not any([args.seeds, args.embed_genre, args.pool, args.stats, args.all]):
        args.all = True

    # Stats mode: just print and exit
    if args.stats:
        path = DATA_EVAL / "gold_pairs_v2.csv"
        if not path.exists():
            path = DATA_EVAL / "gold_pairs.csv"
        gold = pd.read_csv(path)
        print(f"Reading: {path.name}")
        print_stats(gold)
        return

    us_df, kr_df = load_dataframes()

    # Strategy 1: Seed pairs
    print("=== Strategy 1: Seed Pairs ===")
    seed_df = get_seed_pairs()
    print(f"  {len(seed_df)} seed pairs ({(seed_df['relevance']==3).sum()} remake, "
          f"{(seed_df['relevance']==2).sum()} thematic/director, "
          f"{(seed_df['relevance']==1).sum()} genre)")

    if args.seeds:
        DATA_EVAL.mkdir(parents=True, exist_ok=True)
        out = DATA_EVAL / "gold_pairs_v2.csv"
        merged = merge_sources(seed_df, pd.DataFrame(), us_df, kr_df)
        merged.to_csv(out, index=False)
        print(f"\nSaved {len(merged)} pairs to {out.name}")
        print_stats(merged)
        return

    # Strategy 2: Embedding-filtered genre matches
    if args.embed_genre or args.all:
        print("\n=== Strategy 2: Embedding-Filtered Genre Matches ===")
        features = load_all_features()
        genre_df = find_embedding_genre_matches(
            us_df, kr_df, features,
            per_genre_us=args.per_genre_us,
            top_similar=args.top_similar,
        )
        print(f"  {len(genre_df)} genre pairs across {genre_df['genre'].nunique()} genres")
        print(f"  {genre_df['us_tmdb_id'].nunique()} unique US queries")
    else:
        genre_df = pd.DataFrame()

    # Merge and save
    DATA_EVAL.mkdir(parents=True, exist_ok=True)
    merged = merge_sources(seed_df, genre_df, us_df, kr_df)

    out = DATA_EVAL / "gold_pairs_v2.csv"
    merged.to_csv(out, index=False)
    print(f"\nSaved {len(merged)} pairs to {out.name}")
    print_stats(merged)

    # Strategy 3: Pooling (separate output)
    if args.pool:
        print("\n=== Strategy 3: Pooling Candidates ===")
        if "features" not in dir():
            features = load_all_features()
        pool_df = generate_pool_candidates(us_df, kr_df, features)
        pool_out = DATA_EVAL / "pool_candidates.csv"
        pool_df.to_csv(pool_out, index=False)
        print(f"  {len(pool_df)} candidates for {pool_df['us_tmdb_id'].nunique()} queries")
        print(f"  Saved to {pool_out.name}")
        print(f"  Review and label relevance (0-3), then run --merge to incorporate.")


if __name__ == "__main__":
    main()
