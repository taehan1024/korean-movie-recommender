"""Content-based recommendation models for cross-cultural movie matching.

Three scoring strategies (TF-IDF, sentence embedding, hybrid) plus
explanation generation. The hybrid model fuses text similarity, genre
Jaccard, keyword Jaccard, and year proximity with grid-search-tuned weights.
"""

import difflib
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

# Grid-search-tuned weights (288 combos, optimized on DCG@10=0.304):
# - Text dominates (ReasoningRec: item descriptions crucial for sparse data)
# - Keywords provide thematic bridge (3x thematic Hit@10 improvement)
# - Cast dropped to 0 (0.8% cross-industry overlap = pure noise)
# - Year proximity captures era similarity
DEFAULT_WEIGHTS = {"text": 0.47, "genre": 0.24, "keyword": 0.18, "cast": 0.00, "year": 0.11}

YEAR_SIGMA = 10  # Gaussian decay parameter for year proximity


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

    # Keywords (dense multi-hot)
    kw_us = DATA_FEATURES / "keywords_us.npy"
    if kw_us.exists():
        features["kw_us"] = np.load(kw_us)
        features["kw_kr"] = np.load(DATA_FEATURES / "keywords_kr.npy")
        features["kw_vocab"] = joblib.load(DATA_FEATURES / "keyword_vocab.joblib")

    # Year arrays
    year_us = DATA_FEATURES / "year_us.npy"
    if year_us.exists():
        features["year_us"] = np.load(year_us)
        features["year_kr"] = np.load(DATA_FEATURES / "year_kr.npy")

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


def _jaccard_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Jaccard similarity between a query vector and each row in matrix."""
    dot = matrix @ query
    union = query.sum() + matrix.sum(axis=1) - dot
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, dot / union, 0.0)


def _genre_scores(us_idx: int, features: dict) -> np.ndarray:
    """Genre overlap score (Jaccard via dot product on multi-hot)."""
    return _jaccard_scores(features["genre_us"][us_idx], features["genre_kr"])


def _cast_scores(us_idx: int, features: dict) -> np.ndarray:
    """Cast/crew overlap score (Jaccard)."""
    return _jaccard_scores(features["cast_us"][us_idx], features["cast_kr"])


def _keyword_scores(us_idx: int, features: dict) -> np.ndarray:
    """Keyword overlap score (Jaccard on multi-hot keyword vectors)."""
    return _jaccard_scores(features["kw_us"][us_idx], features["kw_kr"])


def _year_scores(us_idx: int, features: dict) -> np.ndarray:
    """Year proximity score (Gaussian decay, sigma=YEAR_SIGMA)."""
    us_year = features["year_us"][us_idx]
    kr_years = features["year_kr"]
    # Handle NaN years: give them 0.5 (neutral) score
    valid = ~np.isnan(kr_years) & ~np.isnan(us_year)
    scores = np.full(len(kr_years), 0.5)
    if not np.isnan(us_year):
        scores[valid] = np.exp(-((kr_years[valid] - us_year) ** 2) / (2 * YEAR_SIGMA ** 2))
    return scores


def _minmax(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)


def _hybrid_scores(
    us_idx: int, features: dict, weights: dict | None = None,
) -> np.ndarray:
    """Weighted combination of text, genre, keyword, cast, and year scores."""
    w = weights or DEFAULT_WEIGHTS

    text = _embedding_scores(us_idx, features)
    genre = _genre_scores(us_idx, features)
    combined = (
        w.get("text", 0) * _minmax(text)
        + w.get("genre", 0) * _minmax(genre)
    )

    # Cast (optional — files excluded from repo, weight=0.0 in tuned model)
    if "cast_us" in features:
        cast = _cast_scores(us_idx, features)
        combined += w.get("cast", 0) * _minmax(cast)

    # Optional components (graceful fallback if features not built yet)
    if "kw_us" in features:
        kw = _keyword_scores(us_idx, features)
        combined += w.get("keyword", 0) * _minmax(kw)
    if "year_us" in features:
        yr = _year_scores(us_idx, features)
        combined += w.get("year", 0) * _minmax(yr)

    return combined


MODEL_SCORERS = {
    "tfidf": _tfidf_scores,
    "embedding": _embedding_scores,
    "hybrid": _hybrid_scores,
}


# ---------------------------------------------------------------------------
# Title resolution
# ---------------------------------------------------------------------------
def resolve_title(query: str, us_df: pd.DataFrame) -> int | None:
    """Find US movie index by exact, contains, or fuzzy match."""
    titles = us_df["title"]
    query_lower = query.lower()

    # Exact match (case-insensitive)
    mask = titles.str.lower() == query_lower
    if mask.any():
        return mask.idxmax()

    # Contains match
    mask = titles.str.lower().str.contains(query_lower, na=False, regex=False)
    if mask.any():
        return mask.idxmax()

    # Fuzzy match via difflib
    title_list = titles.tolist()
    matches = difflib.get_close_matches(query, title_list, n=1, cutoff=0.6)
    if matches:
        return titles[titles == matches[0]].index[0]

    return None


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------
def explain_recommendation(
    us_idx: int, kr_idx: int, features: dict,
    us_df: pd.DataFrame, kr_df: pd.DataFrame,
) -> dict:
    """Generate explanation with per-component score decomposition."""
    explanation = {}

    # Genre overlap + Jaccard score
    if "genre_us" in features:
        us_genres = set(str(us_df.iloc[us_idx]["genres"]).split("|"))
        kr_genres = set(str(kr_df.iloc[kr_idx]["genres"]).split("|"))
        shared = (us_genres & kr_genres) - {""}
        if shared:
            explanation["shared_genres"] = sorted(shared)
        union = (us_genres | kr_genres) - {""}
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

    # Shared cast/crew + Jaccard score
    if "cast_us" in features and "cast_vocab" in features:
        cast_vocab = features["cast_vocab"]
        us_cast = features["cast_us"][us_idx]
        kr_cast = features["cast_kr"][kr_idx]
        shared_idx = np.where((us_cast > 0) & (kr_cast > 0))[0]
        shared_people = [cast_vocab[i] for i in shared_idx]
        if shared_people:
            explanation["shared_cast_crew"] = shared_people
        dot = float(len(shared_idx))
        union = float(us_cast.sum() + kr_cast.sum() - dot)
        explanation["cast_score"] = round(dot / union, 3) if union > 0 else 0.0

    # Shared keywords
    if "kw_us" in features and "kw_vocab" in features:
        kw_vocab = features["kw_vocab"]
        us_kw = features["kw_us"][us_idx]
        kr_kw = features["kw_kr"][kr_idx]
        shared_idx = np.where((us_kw > 0) & (kr_kw > 0))[0]
        shared_kws = [kw_vocab[i] for i in shared_idx]
        if shared_kws:
            explanation["shared_keywords"] = shared_kws
        dot = float(len(shared_idx))
        union = float(us_kw.sum() + kr_kw.sum() - dot)
        explanation["keyword_score"] = round(dot / union, 3) if union > 0 else 0.0

    # Embedding similarity score
    if "emb_us" in features:
        score = float(features["emb_us"][us_idx] @ features["emb_kr"][kr_idx])
        explanation["semantic_similarity"] = round(score, 3)

    # Year proximity
    if "year_us" in features:
        us_year = features["year_us"][us_idx]
        kr_year = features["year_kr"][kr_idx]
        if not (np.isnan(us_year) or np.isnan(kr_year)):
            explanation["year_diff"] = int(abs(us_year - kr_year))

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
    """Get top-K Korean movie recommendations for a US movie title."""
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
