"""Benchmark recommendation models on curated gold pairs.

Metrics include DCG@10 (unnormalized) as primary ranking metric per
Jeunen et al. (KDD 2024), Recall@5/10 per DaisyRec survey, and
bootstrap confidence intervals for small eval sets.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models import get_recommendations, load_all_features, load_dataframes

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_EVAL = SCRIPT_DIR / "data" / "eval"
RESULTS_DIR = SCRIPT_DIR / "results"

MODEL_NAMES = ["tfidf", "embedding", "hybrid"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def precision_at_k(recommended_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Binary precision: fraction of top-k that are relevant."""
    top_k = recommended_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / k


def recall_at_k(recommended_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Recall: fraction of relevant items found in top-k."""
    if not relevant_ids:
        return 0.0
    top_k = recommended_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def dcg_at_k(recommended_ids: list[int], relevance_map: dict[int, int], k: int) -> float:
    """Discounted cumulative gain with graded relevance."""
    dcg = 0.0
    for i, rid in enumerate(recommended_ids[:k]):
        rel = relevance_map.get(rid, 0)
        dcg += rel / np.log2(i + 2)  # i+2 because positions are 1-indexed
    return dcg


def ndcg_at_k(recommended_ids: list[int], relevance_map: dict[int, int], k: int) -> float:
    """Normalized DCG at k.

    Note (Jeunen et al. KDD 2024): normalization can invert method ordering.
    Use DCG@10 as the primary ranking metric; report nDCG for comparability only.
    """
    dcg = dcg_at_k(recommended_ids, relevance_map, k)
    # Ideal DCG: sort by relevance descending
    ideal_rels = sorted(relevance_map.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def bootstrap_ci(scores: list[float], n_boot: int = 1000) -> float:
    """Bootstrap standard error for mean estimate."""
    if len(scores) < 2:
        return 0.0
    arr = np.array(scores)
    boot_means = [np.mean(np.random.choice(arr, size=len(arr), replace=True))
                  for _ in range(n_boot)]
    return round(float(np.std(boot_means)), 4)


def load_gold_pairs() -> pd.DataFrame:
    """Load gold pairs from CSV."""
    path = DATA_EVAL / "gold_pairs.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Gold pairs not found at {path}. "
            "Run curate_eval_pairs.py and save final pairs to data/eval/gold_pairs.csv"
        )
    df = pd.read_csv(path)
    required = {"us_tmdb_id", "kr_tmdb_id", "relevance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Gold pairs CSV missing columns: {missing}")
    return df


def evaluate_model(
    model_name: str,
    gold: pd.DataFrame,
    us_df: pd.DataFrame,
    kr_df: pd.DataFrame,
    features: dict,
    weights: dict | None = None,
) -> dict:
    """Evaluate one model on gold pairs. Returns metrics dict."""
    # Group gold pairs by US movie
    us_groups = gold.groupby("us_tmdb_id").agg({
        "kr_tmdb_id": list,
        "relevance": list,
    }).reset_index()

    p5_scores, p10_scores = [], []
    r5_scores, r10_scores = [], []
    dcg_scores, ndcg_scores = [], []
    genre_metrics: dict[str, list[float]] = {}

    for _, row in us_groups.iterrows():
        us_id = row["us_tmdb_id"]
        relevant_kr_ids = set(row["kr_tmdb_id"])
        relevance_map = dict(zip(row["kr_tmdb_id"], row["relevance"]))

        # Find US movie title
        us_match = us_df[us_df["tmdb_id"] == us_id]
        if us_match.empty:
            continue
        us_title = us_match.iloc[0]["title"]

        # Get recommendations
        recs = get_recommendations(
            us_title, model_name, us_df, kr_df, features,
            top_k=10, weights=weights,
        )
        if recs.empty or "error" in recs.columns:
            continue

        rec_ids = recs["tmdb_id"].tolist()

        # Compute metrics
        p5_scores.append(precision_at_k(rec_ids, relevant_kr_ids, 5))
        p10_scores.append(precision_at_k(rec_ids, relevant_kr_ids, 10))
        r5_scores.append(recall_at_k(rec_ids, relevant_kr_ids, 5))
        r10_scores.append(recall_at_k(rec_ids, relevant_kr_ids, 10))
        dcg_scores.append(dcg_at_k(rec_ids, relevance_map, 10))
        ndcg_scores.append(ndcg_at_k(rec_ids, relevance_map, 10))

        # Per-genre tracking
        us_genres = str(us_match.iloc[0].get("genres", "")).split("|")
        for genre in us_genres:
            if genre:
                genre_metrics.setdefault(genre, []).append(p10_scores[-1])

    def safe_mean(lst: list[float]) -> float:
        return round(float(np.mean(lst)), 4) if lst else 0.0

    metrics = {
        "model": model_name,
        "num_queries": len(p5_scores),
        "P@5": safe_mean(p5_scores),
        "P@10": safe_mean(p10_scores),
        "R@5": safe_mean(r5_scores),
        "R@10": safe_mean(r10_scores),
        "DCG@10": safe_mean(dcg_scores),
        "NDCG@10": safe_mean(ndcg_scores),
        "P@10_se": bootstrap_ci(p10_scores),
        "DCG@10_se": bootstrap_ci(dcg_scores),
        "NDCG@10_se": bootstrap_ci(ndcg_scores),
        "per_genre_P@10": {
            g: round(np.mean(scores), 4)
            for g, scores in sorted(genre_metrics.items())
        },
    }

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark recommendation models")
    parser.add_argument("--model", default="all",
                        help="Model to evaluate: tfidf, embedding, hybrid, or all")
    parser.add_argument("--weights", default=None,
                        help="Hybrid weights as comma-separated: text,genre,cast (e.g. 0.4,0.35,0.25)")
    args = parser.parse_args()

    # Parse weights
    weights = None
    if args.weights:
        parts = [float(x) for x in args.weights.split(",")]
        weights = {"text": parts[0], "genre": parts[1], "cast": parts[2]}

    # Load data
    gold = load_gold_pairs()
    us_df, kr_df = load_dataframes()
    features = load_all_features()
    print(f"Loaded {len(gold)} gold pairs, US={len(us_df)}, KR={len(kr_df)}")

    # Determine models to evaluate
    models = MODEL_NAMES if args.model == "all" else [args.model]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    for model_name in models:
        print(f"\n=== Evaluating {model_name} ===")
        metrics = evaluate_model(model_name, gold, us_df, kr_df, features, weights)
        all_metrics.append(metrics)

        # Save individual results
        out_path = RESULTS_DIR / f"metrics_{model_name}.json"
        out_path.write_text(json.dumps(metrics, indent=2))
        print(f"  P@5={metrics['P@5']:.4f}  P@10={metrics['P@10']:.4f} (+/-{metrics['P@10_se']:.4f})")
        print(f"  R@5={metrics['R@5']:.4f}  R@10={metrics['R@10']:.4f}")
        print(f"  DCG@10={metrics['DCG@10']:.4f} (+/-{metrics['DCG@10_se']:.4f})  "
              f"NDCG@10={metrics['NDCG@10']:.4f} (+/-{metrics['NDCG@10_se']:.4f})")
        print(f"  (n={metrics['num_queries']})")

        # Per-genre breakdown
        for genre, score in metrics["per_genre_P@10"].items():
            print(f"    {genre}: P@10={score:.4f}")

    # Comparison table
    if len(all_metrics) > 1:
        comparison = pd.DataFrame([
            {k: v for k, v in m.items() if k != "per_genre_P@10"}
            for m in all_metrics
        ])
        comparison.to_csv(RESULTS_DIR / "benchmark_comparison.csv", index=False)
        print(f"\n=== Benchmark Comparison ===")
        print(comparison.to_string(index=False))

        # Check acceptance criterion
        best_p10 = max(m["P@10"] for m in all_metrics)
        if best_p10 >= 0.40:
            print(f"\nACCEPTED: Best P@10 = {best_p10:.4f} >= 0.40")
        else:
            print(f"\nBELOW TARGET: Best P@10 = {best_p10:.4f} < 0.40")


if __name__ == "__main__":
    main()
