"""Three recommendation models with a shared interface."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_PROCESSED = SCRIPT_DIR / "data" / "processed"
DATA_FEATURES = SCRIPT_DIR / "data" / "features"

DEFAULT_WEIGHTS = {"text": 0.5, "genre": 0.3, "cast": 0.2}


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------
def load_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    us = pd.read_csv(DATA_PROCESSED / "us_movies.csv")
    kr = pd.read_csv(DATA_PROCESSED / "kr_movies.csv")
    return us, kr


def load_all_features() -> dict:
    """Load all feature matrices into a dict."""
    features = {}

    # TF-IDF (sparse)
    tfidf_us = DATA_FEATURES / "tfidf_us.npz"
    if tfidf_us.exists():
        features["tfidf_us"] = sparse.load_npz(tfidf_us)
        features["tfidf_kr"] = sparse.load_npz(DATA_FEATURES / "tfidf_kr.npz")
        features["tfidf_vectorizer"] = joblib.load(DATA_FEATURES / "tfidf_vectorizer.joblib")

    # Sentence embeddings (dense, L2-normalized)
    emb_us = DATA_FEATURES / "embeddings_us.npy"
    if emb_us.exists():
        features["emb_us"] = np.load(emb_us)
        features["emb_kr"] = np.load(DATA_FEATURES / "embeddings_kr.npy")

    # Genres (dense multi-hot)
    genre_us = DATA_FEATURES / "genres_us.npy"
    if genre_us.exists():
        features["genre_us"] = np.load(genre_us)
        features["genre_kr"] = np.load(DATA_FEATURES / "genres_kr.npy")
        features["genre_vocab"] = joblib.load(DATA_FEATURES / "genre_vocab.joblib")

    # Cast/crew (dense multi-hot)
    cast_us = DATA_FEATURES / "cast_us.npy"
    if cast_us.exists():
        features["cast_us"] = np.load(cast_us)
        features["cast_kr"] = np.load(DATA_FEATURES / "cast_kr.npy")
        features["cast_vocab"] = joblib.load(DATA_FEATURES / "cast_vocab.joblib")

    return features


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
def _tfidf_scores(us_idx: int, features: dict) -> np.ndarray:
    """Cosine similarity between one US movie and all KR movies via TF-IDF."""
    query = features["tfidf_us"][us_idx]
    return cosine_similarity(query, features["tfidf_kr"]).flatten()


def _embedding_scores(us_idx: int, features: dict) -> np.ndarray:
    """Cosine similarity via pre-normalized sentence embeddings (dot product)."""
    query = features["emb_us"][us_idx]
    return (features["emb_kr"] @ query).flatten()


def _genre_scores(us_idx: int, features: dict) -> np.ndarray:
    """Genre overlap score (Jaccard-like via dot product on multi-hot)."""
    query = features["genre_us"][us_idx]
    kr = features["genre_kr"]
    # Dot product = number of shared genres
    dot = kr @ query
    # Normalize by union size to get Jaccard
    query_sum = query.sum()
    kr_sums = kr.sum(axis=1)
    union = query_sum + kr_sums - dot
    # Avoid division by zero
    return np.where(union > 0, dot / union, 0.0)


def _cast_scores(us_idx: int, features: dict) -> np.ndarray:
    """Cast/crew overlap score."""
    query = features["cast_us"][us_idx]
    kr = features["cast_kr"]
    dot = kr @ query
    query_sum = query.sum()
    kr_sums = kr.sum(axis=1)
    union = query_sum + kr_sums - dot
    return np.where(union > 0, dot / union, 0.0)


def _hybrid_scores(
    us_idx: int, features: dict, weights: dict | None = None,
) -> np.ndarray:
    """Weighted combination of text (embedding), genre, and cast scores."""
    w = weights or DEFAULT_WEIGHTS

    # Use embedding for text component (better than TF-IDF for cross-cultural)
    text = _embedding_scores(us_idx, features)
    genre = _genre_scores(us_idx, features)
    cast = _cast_scores(us_idx, features)

    # Min-max normalize each component to [0, 1] before combining
    def minmax(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)

    return w["text"] * minmax(text) + w["genre"] * minmax(genre) + w["cast"] * minmax(cast)


MODEL_SCORERS = {
    "tfidf": _tfidf_scores,
    "embedding": _embedding_scores,
    "hybrid": _hybrid_scores,
}


# ---------------------------------------------------------------------------
# Title resolution
# ---------------------------------------------------------------------------
def resolve_title(query: str, us_df: pd.DataFrame) -> int | None:
    """Find US movie index by exact or fuzzy title match."""
    # Exact match (case-insensitive)
    mask = us_df["title"].str.lower() == query.lower()
    if mask.any():
        return mask.idxmax()

    # Fuzzy: contains match
    mask = us_df["title"].str.lower().str.contains(query.lower(), na=False)
    if mask.any():
        return mask.idxmax()

    return None


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------
def explain_recommendation(
    us_idx: int, kr_idx: int, features: dict,
    us_df: pd.DataFrame, kr_df: pd.DataFrame,
) -> dict:
    """Generate explanation with per-component score decomposition.

    Returns individual component scores (genre_score, cast_score,
    semantic_similarity) so the UI can show which signal drove the match.
    """
    explanation = {}

    # Genre overlap + Jaccard score
    if "genre_us" in features:
        us_genres = set(str(us_df.iloc[us_idx]["genres"]).split("|"))
        kr_genres = set(str(kr_df.iloc[kr_idx]["genres"]).split("|"))
        shared = us_genres & kr_genres - {""}
        if shared:
            explanation["shared_genres"] = sorted(shared)
        union = us_genres | kr_genres - {""}
        explanation["genre_score"] = round(len(shared) / len(union), 3) if union else 0.0

    # Top TF-IDF terms
    if "tfidf_vectorizer" in features:
        vectorizer = features["tfidf_vectorizer"]
        vocab = vectorizer.get_feature_names_out()
        us_vec = features["tfidf_us"][us_idx].toarray().flatten()
        kr_vec = features["tfidf_kr"][kr_idx].toarray().flatten()
        product = us_vec * kr_vec
        top_idx = product.argsort()[-5:][::-1]
        top_terms = [vocab[i] for i in top_idx if product[i] > 0]
        if top_terms:
            explanation["shared_terms"] = top_terms

    # Shared cast/crew + overlap score
    if "cast_us" in features and "cast_vocab" in features:
        cast_vocab = features["cast_vocab"]
        us_cast = features["cast_us"][us_idx]
        kr_cast = features["cast_kr"][kr_idx]
        shared_idx = np.where((us_cast > 0) & (kr_cast > 0))[0]
        shared_people = [cast_vocab[i] for i in shared_idx]
        if shared_people:
            explanation["shared_cast_crew"] = shared_people
        explanation["cast_score"] = round(float(len(shared_idx)) / max(us_cast.sum(), 1), 3)

    # Embedding similarity score
    if "emb_us" in features:
        score = float(features["emb_us"][us_idx] @ features["emb_kr"][kr_idx])
        explanation["semantic_similarity"] = round(score, 3)

    return explanation


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------
def get_recommendations(
    us_title: str,
    model: str,
    us_df: pd.DataFrame,
    kr_df: pd.DataFrame,
    features: dict,
    top_k: int = 10,
    weights: dict | None = None,
    filters: dict | None = None,
) -> pd.DataFrame:
    """High-level recommendation API.

    Args:
        us_title: American movie title (exact or fuzzy).
        model: One of "tfidf", "embedding", "hybrid".
        us_df, kr_df: Processed DataFrames.
        features: Dict from load_all_features().
        top_k: Number of results.
        weights: Hybrid model weights (text/genre/cast).
        filters: Optional dict with genre, year_min, year_max, min_rating.

    Returns:
        DataFrame of top-K Korean movies with scores and metadata.
    """
    us_idx = resolve_title(us_title, us_df)
    if us_idx is None:
        return pd.DataFrame(columns=["title", "error"])

    # Score
    if model == "hybrid":
        scores = _hybrid_scores(us_idx, features, weights)
    else:
        scorer = MODEL_SCORERS[model]
        scores = scorer(us_idx, features)

    # Build result DataFrame
    result = kr_df.copy()
    result["score"] = scores

    # Apply filters
    if filters:
        if "genre" in filters and filters["genre"]:
            genre_set = set(filters["genre"])
            mask = result["genres"].apply(
                lambda g: bool(set(str(g).split("|")) & genre_set)
            )
            result = result[mask]
        if "year_min" in filters:
            result = result[result["year"] >= filters["year_min"]]
        if "year_max" in filters:
            result = result[result["year"] <= filters["year_max"]]
        if "min_rating" in filters:
            result = result[result["rating"] >= filters["min_rating"]]

    # Sort and take top-K
    result = result.nlargest(top_k, "score")

    # Add explanations
    explanations = []
    for kr_idx in result.index:
        exp = explain_recommendation(us_idx, kr_idx, features, us_df, kr_df)
        explanations.append(exp)
    result["explanation"] = explanations

    return result.reset_index(drop=True)
