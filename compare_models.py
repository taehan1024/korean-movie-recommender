"""Compare v1 vs v2 hybrid model rank agreement.

v1: models.py (text=0.5, genre=0.3, cast=0.2)
v2: models_v2.py (text=0.50, genre=0.20, keyword=0.15, cast=0.05, year=0.10)

Uses v2 features (superset of v1) so both scorers can run.
"""

import numpy as np
import pandas as pd
from models import _hybrid_scores as hybrid_v1
from models_v2 import _hybrid_scores as hybrid_v2, load_all_features, load_dataframes

us_df, kr_df = load_dataframes()
features = load_all_features()  # v2 features (superset)

n_us = len(us_df)
overlaps = []
rank_corrs = []

for i in range(n_us):
    scores_v1 = hybrid_v1(i, features)
    scores_v2 = hybrid_v2(i, features)

    top10_v1 = set(scores_v1.argsort()[-10:])
    top10_v2 = set(scores_v2.argsort()[-10:])
    overlaps.append(len(top10_v1 & top10_v2) / 10)

    # Spearman rank correlation on full KR catalog
    ranks_v1 = scores_v1.argsort().argsort()
    ranks_v2 = scores_v2.argsort().argsort()
    corr = np.corrcoef(ranks_v1, ranks_v2)[0, 1]
    rank_corrs.append(corr)

print(f"US movies compared: {n_us}")
print(f"Mean top-10 overlap: {np.mean(overlaps):.3f}  (1.0 = identical)")
print(f"Mean rank correlation: {np.mean(rank_corrs):.3f}")
print(f"Top-10 overlap distribution:")
print(f"  0%:   {sum(1 for o in overlaps if o == 0) / n_us:.1%}")
print(f"  1-30%: {sum(1 for o in overlaps if 0 < o <= 0.3) / n_us:.1%}")
print(f"  31-70%: {sum(1 for o in overlaps if 0.3 < o <= 0.7) / n_us:.1%}")
print(f"  71-99%: {sum(1 for o in overlaps if 0.7 < o < 1.0) / n_us:.1%}")
print(f"  100%: {sum(1 for o in overlaps if o == 1.0) / n_us:.1%}")

# Show 5 movies with lowest overlap (most disagreement)
print("\n--- Most disagreed (lowest overlap) ---")
worst_idx = np.argsort(overlaps)[:5]
for idx in worst_idx:
    top10_v1 = scores_v1_cache = hybrid_v1(idx, features).argsort()[-5:][::-1]
    top10_v2 = hybrid_v2(idx, features).argsort()[-5:][::-1]
    print(f"\n  US: {us_df.iloc[idx]['title']}")
    print(f"    v1 top-5: {[kr_df.iloc[k]['title'] for k in top10_v1]}")
    print(f"    v2 top-5: {[kr_df.iloc[k]['title'] for k in top10_v2]}")
