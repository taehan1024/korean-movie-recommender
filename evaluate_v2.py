"""V2 benchmark: adds Hit@K, MRR, per-relevance breakdown, weight tuning.

Changes from v1 (evaluate.py):
- Added hit_at_k() and mrr() metrics (appropriate for sparse gold labels)
- Added per-relevance-level breakdown (remake/thematic/genre)
- Added --tune flag for grid search over hybrid weights
- Revised acceptance criteria (Hit@10, MRR, R@10, DCG@10)
- Imports from models_v2 instead of models
"""

import argparse
import json
from itertools import product as iterproduct
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2 import get_recommendations, load_all_features, load_dataframes

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_EVAL = SCRIPT_DIR / "data" / "eval"
RESULTS_DIR = SCRIPT_DIR / "results"

MODEL_NAMES = ["tfidf", "embedding", "hybrid"]

RELEVANCE_LABELS = {3: "remake", 2: "thematic", 1: "genre"}


# ---------------------------------------------------------------------------
# Metrics (v1 + v2)
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


def hit_at_k(recommended_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Hit@K: 1.0 if any relevant item in top-k, else 0.0.

    More appropriate than P@K when most queries have only 1-3 relevant items.
    """
    top_k = recommended_ids[:k]
    return 1.0 if any(rid in relevant_ids for rid in top_k) else 0.0


def mrr(recommended_ids: list[int], relevant_ids: set[int]) -> float:
    """Mean Reciprocal Rank: 1/(rank of first relevant item), 0 if none found."""
    for i, rid in enumerate(recommended_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(recommended_ids: list[int], relevance_map: dict[int, int], k: int) -> float:
    """Discounted cumulative gain with graded relevance."""
    dcg = 0.0
    for i, rid in enumerate(recommended_ids[:k]):
        rel = relevance_map.get(rid, 0)
        dcg += rel / np.log2(i + 2)
    return dcg


def ndcg_at_k(recommended_ids: list[int], relevance_map: dict[int, int], k: int) -> float:
    """Normalized DCG at k.

    Note (Jeunen et al. KDD 2024): normalization can invert method ordering.
    Use DCG@10 as the primary ranking metric; report nDCG for comparability only.
    """
    dcg = dcg_at_k(recommended_ids, relevance_map, k)
    ideal_rels = sorted(relevance_map.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------
def bootstrap_ci(scores: list[float], n_boot: int = 1000) -> float:
    """Bootstrap standard error for mean estimate."""
    if len(scores) < 2:
        return 0.0
    arr = np.array(scores)
    boot_means = [np.mean(np.random.choice(arr, size=len(arr), replace=True))
                  for _ in range(n_boot)]
    return round(float(np.std(boot_means)), 4)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    model_name: str,
    gold: pd.DataFrame,
    us_df: pd.DataFrame,
    kr_df: pd.DataFrame,
    features: dict,
    weights: dict | None = None,
) -> dict:
    """Evaluate one model on gold pairs. Returns metrics dict."""
    us_groups = gold.groupby("us_tmdb_id").agg({
        "kr_tmdb_id": list,
        "relevance": list,
    }).reset_index()

    p5_scores, p10_scores = [], []
    r5_scores, r10_scores = [], []
    hit5_scores, hit10_scores = [], []
    mrr_scores = []
    dcg_scores, ndcg_scores = [], []
    genre_metrics: dict[str, list[float]] = {}
    relevance_metrics: dict[str, list[float]] = {}  # per-relevance-level

    for _, row in us_groups.iterrows():
        us_id = row["us_tmdb_id"]
        relevant_kr_ids = set(row["kr_tmdb_id"])
        relevance_map = dict(zip(row["kr_tmdb_id"], row["relevance"]))

        us_match = us_df[us_df["tmdb_id"] == us_id]
        if us_match.empty:
            continue
        us_title = us_match.iloc[0]["title"]

        recs = get_recommendations(
            us_title, model_name, us_df, kr_df, features,
            top_k=10, weights=weights,
        )
        if recs.empty or "error" in recs.columns:
            continue

        rec_ids = recs["tmdb_id"].tolist()

        # Compute all metrics
        p5_scores.append(precision_at_k(rec_ids, relevant_kr_ids, 5))
        p10_scores.append(precision_at_k(rec_ids, relevant_kr_ids, 10))
        r5_scores.append(recall_at_k(rec_ids, relevant_kr_ids, 5))
        r10_scores.append(recall_at_k(rec_ids, relevant_kr_ids, 10))
        hit5_scores.append(hit_at_k(rec_ids, relevant_kr_ids, 5))
        hit10_scores.append(hit_at_k(rec_ids, relevant_kr_ids, 10))
        mrr_scores.append(mrr(rec_ids, relevant_kr_ids))
        dcg_scores.append(dcg_at_k(rec_ids, relevance_map, 10))
        ndcg_scores.append(ndcg_at_k(rec_ids, relevance_map, 10))

        # Per-genre tracking
        us_genres = str(us_match.iloc[0].get("genres", "")).split("|")
        for genre in us_genres:
            if genre:
                genre_metrics.setdefault(genre, []).append(hit10_scores[-1])

        # Per-relevance-level tracking
        max_rel = max(row["relevance"])
        rel_label = RELEVANCE_LABELS.get(max_rel, f"rel_{max_rel}")
        relevance_metrics.setdefault(rel_label, []).append(hit10_scores[-1])

    def safe_mean(lst: list[float]) -> float:
        return round(float(np.mean(lst)), 4) if lst else 0.0

    metrics = {
        "model": model_name,
        "num_queries": len(p5_scores),
        # V1 metrics
        "P@5": safe_mean(p5_scores),
        "P@10": safe_mean(p10_scores),
        "R@5": safe_mean(r5_scores),
        "R@10": safe_mean(r10_scores),
        "DCG@10": safe_mean(dcg_scores),
        "NDCG@10": safe_mean(ndcg_scores),
        # V2 metrics
        "Hit@5": safe_mean(hit5_scores),
        "Hit@10": safe_mean(hit10_scores),
        "MRR": safe_mean(mrr_scores),
        # Bootstrap CIs
        "Hit@10_se": bootstrap_ci(hit10_scores),
        "MRR_se": bootstrap_ci(mrr_scores),
        "P@10_se": bootstrap_ci(p10_scores),
        "DCG@10_se": bootstrap_ci(dcg_scores),
        "NDCG@10_se": bootstrap_ci(ndcg_scores),
        # Breakdowns
        "per_genre_Hit@10": {
            g: round(np.mean(scores), 4)
            for g, scores in sorted(genre_metrics.items())
        },
        "per_relevance_Hit@10": {
            r: round(np.mean(scores), 4)
            for r, scores in sorted(relevance_metrics.items())
        },
    }

    return metrics


# ---------------------------------------------------------------------------
# Weight tuning
# ---------------------------------------------------------------------------
def tune_weights(
    gold: pd.DataFrame,
    us_df: pd.DataFrame,
    kr_df: pd.DataFrame,
    features: dict,
) -> dict:
    """Grid search for optimal hybrid weights. Optimizes on DCG@10."""
    print("\n=== Weight Tuning (Grid Search) ===")

    text_vals = [0.4, 0.5, 0.6, 0.7]
    genre_vals = [0.10, 0.20, 0.30]
    keyword_vals = [0.0, 0.10, 0.15, 0.20]
    cast_vals = [0.0, 0.05]
    year_vals = [0.0, 0.05, 0.10]

    best_dcg = -1.0
    best_weights = None
    best_metrics = None
    n_combos = 0

    for t, g, k, c, y in iterproduct(text_vals, genre_vals, keyword_vals, cast_vals, year_vals):
        total = t + g + k + c + y
        if total == 0:
            continue
        # Normalize
        w = {"text": t/total, "genre": g/total, "keyword": k/total, "cast": c/total, "year": y/total}
        metrics = evaluate_model("hybrid", gold, us_df, kr_df, features, weights=w)
        n_combos += 1

        if metrics["DCG@10"] > best_dcg:
            best_dcg = metrics["DCG@10"]
            best_weights = w
            best_metrics = metrics

        if n_combos % 50 == 0:
            print(f"  Tested {n_combos} combos, best DCG@10={best_dcg:.4f}")

    print(f"\n  Total combos tested: {n_combos}")
    print(f"  Best DCG@10: {best_dcg:.4f}")
    print(f"  Best weights: {json.dumps({k: round(v, 3) for k, v in best_weights.items()})}")
    print(f"  Hit@10={best_metrics['Hit@10']:.4f}  MRR={best_metrics['MRR']:.4f}  "
          f"R@10={best_metrics['R@10']:.4f}")

    return {"best_weights": best_weights, "best_metrics": best_metrics, "n_combos": n_combos}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark recommendation models (v2)")
    parser.add_argument("--model", default="all",
                        help="Model to evaluate: tfidf, embedding, hybrid, or all")
    parser.add_argument("--weights", default=None,
                        help="Hybrid weights: text,genre,keyword,cast,year (e.g. 0.5,0.2,0.15,0.05,0.1)")
    parser.add_argument("--tune", action="store_true",
                        help="Run grid search for optimal hybrid weights")
    args = parser.parse_args()

    # Parse weights
    weights = None
    if args.weights:
        parts = [float(x) for x in args.weights.split(",")]
        keys = ["text", "genre", "keyword", "cast", "year"]
        weights = dict(zip(keys[:len(parts)], parts))

    # Load data
    gold = load_gold_pairs()
    us_df, kr_df = load_dataframes()
    features = load_all_features()
    print(f"Loaded {len(gold)} gold pairs, US={len(us_df)}, KR={len(kr_df)}")
    print(f"Features available: {sorted(features.keys())}")

    # Weight tuning mode
    if args.tune:
        result = tune_weights(gold, us_df, kr_df, features)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "tune_results.json").write_text(json.dumps(result, indent=2, default=str))
        return

    # Standard evaluation
    models = MODEL_NAMES if args.model == "all" else [args.model]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    for model_name in models:
        print(f"\n=== Evaluating {model_name} (v2) ===")
        metrics = evaluate_model(model_name, gold, us_df, kr_df, features, weights)
        all_metrics.append(metrics)

        # Save individual results
        out_path = RESULTS_DIR / f"metrics_v2_{model_name}.json"
        out_path.write_text(json.dumps(metrics, indent=2))

        # Print results
        print(f"  Hit@5={metrics['Hit@5']:.4f}  Hit@10={metrics['Hit@10']:.4f} (+/-{metrics['Hit@10_se']:.4f})")
        print(f"  MRR={metrics['MRR']:.4f} (+/-{metrics['MRR_se']:.4f})")
        print(f"  P@5={metrics['P@5']:.4f}  P@10={metrics['P@10']:.4f}")
        print(f"  R@5={metrics['R@5']:.4f}  R@10={metrics['R@10']:.4f}")
        print(f"  DCG@10={metrics['DCG@10']:.4f} (+/-{metrics['DCG@10_se']:.4f})  "
              f"NDCG@10={metrics['NDCG@10']:.4f}")
        print(f"  (n={metrics['num_queries']})")

        # Per-relevance breakdown
        print("  Per-relevance Hit@10:")
        for rel, score in metrics["per_relevance_Hit@10"].items():
            print(f"    {rel}: {score:.4f}")

        # Per-genre breakdown
        print("  Per-genre Hit@10:")
        for genre, score in metrics["per_genre_Hit@10"].items():
            print(f"    {genre}: {score:.4f}")

    # Comparison table
    if len(all_metrics) > 1:
        comparison = pd.DataFrame([
            {k: v for k, v in m.items()
             if k not in ("per_genre_Hit@10", "per_relevance_Hit@10", "per_genre_P@10")}
            for m in all_metrics
        ])
        comparison.to_csv(RESULTS_DIR / "benchmark_v2_comparison.csv", index=False)
        print(f"\n=== V2 Benchmark Comparison ===")
        print(comparison[["model", "Hit@5", "Hit@10", "MRR", "P@10", "R@10", "DCG@10", "NDCG@10"]].to_string(index=False))

        # Check revised acceptance criteria
        best = max(all_metrics, key=lambda m: m["DCG@10"])
        print(f"\nBest model by DCG@10: {best['model']}")
        targets = {"Hit@10": 0.35, "MRR": 0.10, "R@10": 0.25, "DCG@10": 0.50}
        for metric, target in targets.items():
            val = best[metric]
            status = "PASS" if val >= target else "BELOW"
            print(f"  {metric}: {val:.4f} vs target {target:.2f} [{status}]")


if __name__ == "__main__":
    main()
