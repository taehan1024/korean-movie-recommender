"""Build feature matrices from processed movie CSVs."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_PROCESSED = SCRIPT_DIR / "data" / "processed"
DATA_FEATURES = SCRIPT_DIR / "data" / "features"

TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
TOP_CAST = 5


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    us = pd.read_csv(DATA_PROCESSED / "us_movies.csv")
    kr = pd.read_csv(DATA_PROCESSED / "kr_movies.csv")
    # Fill missing overviews with empty string for feature extraction
    us["overview"] = us["overview"].fillna("")
    kr["overview"] = kr["overview"].fillna("")
    return us, kr


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------
def build_tfidf(us: pd.DataFrame, kr: pd.DataFrame) -> None:
    """Fit TF-IDF on combined corpus, transform separately."""
    print("Building TF-IDF features...")
    combined = pd.concat([us["overview"], kr["overview"]], ignore_index=True)

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words="english",
        min_df=2,
        max_df=0.95,
    )
    vectorizer.fit(combined)

    us_tfidf = vectorizer.transform(us["overview"])
    kr_tfidf = vectorizer.transform(kr["overview"])

    DATA_FEATURES.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(DATA_FEATURES / "tfidf_us.npz", us_tfidf)
    sparse.save_npz(DATA_FEATURES / "tfidf_kr.npz", kr_tfidf)
    joblib.dump(vectorizer, DATA_FEATURES / "tfidf_vectorizer.joblib")

    print(f"  TF-IDF: vocab={len(vectorizer.vocabulary_)}, "
          f"US={us_tfidf.shape}, KR={kr_tfidf.shape}")


# ---------------------------------------------------------------------------
# Sentence embeddings
# ---------------------------------------------------------------------------
def build_embeddings(us: pd.DataFrame, kr: pd.DataFrame) -> None:
    """Encode overviews with sentence-transformers, L2-normalize."""
    print(f"Building sentence embeddings ({EMBEDDING_MODEL_NAME})...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    us_emb = model.encode(us["overview"].tolist(), batch_size=64, show_progress_bar=True)
    kr_emb = model.encode(kr["overview"].tolist(), batch_size=64, show_progress_bar=True)

    # L2-normalize so dot product = cosine similarity
    us_emb = normalize(us_emb, norm="l2")
    kr_emb = normalize(kr_emb, norm="l2")

    DATA_FEATURES.mkdir(parents=True, exist_ok=True)
    np.save(DATA_FEATURES / "embeddings_us.npy", us_emb)
    np.save(DATA_FEATURES / "embeddings_kr.npy", kr_emb)

    print(f"  Embeddings: US={us_emb.shape}, KR={kr_emb.shape}")


# ---------------------------------------------------------------------------
# Genre multi-hot
# ---------------------------------------------------------------------------
def build_genres(us: pd.DataFrame, kr: pd.DataFrame) -> None:
    """Multi-hot encode genres over shared vocabulary."""
    print("Building genre features...")

    def parse_genres(series: pd.Series) -> list[set[str]]:
        return [set(str(g).split("|")) if pd.notna(g) and str(g).strip() else set()
                for g in series]

    us_genres = parse_genres(us["genres"])
    kr_genres = parse_genres(kr["genres"])

    # Shared genre vocabulary
    all_genres = sorted(set().union(*us_genres, *kr_genres) - {""})

    genre_to_idx = {g: i for i, g in enumerate(all_genres)}

    def encode(genre_sets: list[set[str]]) -> np.ndarray:
        mat = np.zeros((len(genre_sets), len(all_genres)), dtype=np.float32)
        for i, gs in enumerate(genre_sets):
            for g in gs:
                if g in genre_to_idx:
                    mat[i, genre_to_idx[g]] = 1.0
        return mat

    us_mat = encode(us_genres)
    kr_mat = encode(kr_genres)

    DATA_FEATURES.mkdir(parents=True, exist_ok=True)
    np.save(DATA_FEATURES / "genres_us.npy", us_mat)
    np.save(DATA_FEATURES / "genres_kr.npy", kr_mat)
    joblib.dump(all_genres, DATA_FEATURES / "genre_vocab.joblib")

    print(f"  Genres: {len(all_genres)} categories, US={us_mat.shape}, KR={kr_mat.shape}")


# ---------------------------------------------------------------------------
# Cast/crew multi-hot
# ---------------------------------------------------------------------------
def build_cast_crew(us: pd.DataFrame, kr: pd.DataFrame) -> None:
    """Multi-hot encode director + top cast over shared vocabulary."""
    print("Building cast/crew features...")

    def parse_people(df: pd.DataFrame) -> list[set[str]]:
        people = []
        for _, row in df.iterrows():
            names = set()
            if pd.notna(row.get("director")):
                names.add(str(row["director"]).strip())
            if pd.notna(row.get("cast_top5")):
                for name in str(row["cast_top5"]).split("|"):
                    name = name.strip()
                    if name:
                        names.add(name)
            people.append(names)
        return people

    us_people = parse_people(us)
    kr_people = parse_people(kr)

    # Shared vocabulary — only include people who appear in BOTH catalogs
    # (cross-industry overlap) OR appear in 2+ movies within a catalog
    us_all = set().union(*us_people)
    kr_all = set().union(*kr_people)
    cross_industry = us_all & kr_all

    # Count within each catalog
    from collections import Counter
    us_counts = Counter(p for ps in us_people for p in ps)
    kr_counts = Counter(p for ps in kr_people for p in ps)

    # Include: cross-industry people + anyone with 2+ appearances
    vocab = sorted(
        cross_industry
        | {p for p, c in us_counts.items() if c >= 2}
        | {p for p, c in kr_counts.items() if c >= 2}
    )

    person_to_idx = {p: i for i, p in enumerate(vocab)}

    def encode(people_sets: list[set[str]]) -> np.ndarray:
        mat = np.zeros((len(people_sets), len(vocab)), dtype=np.float32)
        for i, ps in enumerate(people_sets):
            for p in ps:
                if p in person_to_idx:
                    mat[i, person_to_idx[p]] = 1.0
        return mat

    us_mat = encode(us_people)
    kr_mat = encode(kr_people)

    DATA_FEATURES.mkdir(parents=True, exist_ok=True)
    np.save(DATA_FEATURES / "cast_us.npy", us_mat)
    np.save(DATA_FEATURES / "cast_kr.npy", kr_mat)
    joblib.dump(vocab, DATA_FEATURES / "cast_vocab.joblib")

    print(f"  Cast/crew: {len(vocab)} people ({len(cross_industry)} cross-industry), "
          f"US={us_mat.shape}, KR={kr_mat.shape}")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def save_manifest(us: pd.DataFrame, kr: pd.DataFrame, groups: list[str]) -> None:
    """Write feature manifest for downstream scripts."""
    manifest = {
        "us_count": len(us),
        "kr_count": len(kr),
        "groups_built": groups,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "tfidf_max_features": TFIDF_MAX_FEATURES,
    }
    (DATA_FEATURES / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
BUILDERS = {
    "tfidf": build_tfidf,
    "embedding": build_embeddings,
    "genre": build_genres,
    "cast": build_cast_crew,
}


def main():
    parser = argparse.ArgumentParser(description="Build feature matrices")
    parser.add_argument(
        "--groups", default="tfidf,embedding,genre,cast",
        help="Comma-separated feature groups to build",
    )
    args = parser.parse_args()

    groups = [g.strip() for g in args.groups.split(",")]
    us, kr = load_processed()
    print(f"Loaded US={len(us)}, KR={len(kr)} movies")

    for group in groups:
        if group not in BUILDERS:
            print(f"Unknown group: {group}")
            continue
        BUILDERS[group](us, kr)

    save_manifest(us, kr, groups)
    print("\nFeature engineering complete.")


if __name__ == "__main__":
    main()
