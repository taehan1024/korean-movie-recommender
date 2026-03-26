"""Tests for evaluation metrics."""

import numpy as np

from evaluate import (
    bootstrap_ci,
    dcg_at_k,
    hit_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK:
    def test_basic(self):
        assert np.isclose(precision_at_k([1, 2, 3], {1, 3}, 3), 2 / 3)

    def test_empty_recommendations(self):
        assert precision_at_k([], {1}, 5) == 0.0

    def test_no_hits(self):
        assert precision_at_k([1, 2, 3], {4, 5}, 3) == 0.0

    def test_all_hits(self):
        assert np.isclose(precision_at_k([1, 2, 3], {1, 2, 3}, 3), 1.0)


class TestRecallAtK:
    def test_basic(self):
        assert np.isclose(recall_at_k([1, 2, 3], {1, 3, 5}, 3), 2 / 3)

    def test_no_relevant(self):
        assert recall_at_k([1, 2], set(), 2) == 0.0

    def test_all_found(self):
        assert np.isclose(recall_at_k([1, 2, 3], {1, 2}, 3), 1.0)


class TestHitAtK:
    def test_hit(self):
        assert hit_at_k([1, 2, 3], {3}, 3) == 1.0

    def test_miss(self):
        assert hit_at_k([1, 2, 3], {4}, 3) == 0.0

    def test_hit_early(self):
        assert hit_at_k([1, 2, 3], {1}, 3) == 1.0


class TestMRR:
    def test_first_position(self):
        assert mrr([1, 2, 3], {1}) == 1.0

    def test_second_position(self):
        assert mrr([1, 2, 3], {2}) == 0.5

    def test_third_position(self):
        assert np.isclose(mrr([1, 2, 3], {3}), 1 / 3)

    def test_no_hit(self):
        assert mrr([1, 2, 3], {4}) == 0.0

    def test_multiple_relevant(self):
        assert mrr([1, 2, 3], {2, 3}) == 0.5


class TestDCGAtK:
    def test_basic(self):
        expected = 3 / np.log2(2) + 2 / np.log2(3) + 1 / np.log2(4)
        result = dcg_at_k([1, 2, 3], {1: 3, 2: 2, 3: 1}, 3)
        assert np.isclose(result, expected)

    def test_no_relevant(self):
        assert dcg_at_k([1, 2, 3], {}, 3) == 0.0

    def test_partial_relevant(self):
        expected = 2 / np.log2(3)
        result = dcg_at_k([1, 2, 3], {2: 2}, 3)
        assert np.isclose(result, expected)


class TestNDCGAtK:
    def test_perfect_ranking(self):
        rel_map = {1: 3, 2: 2, 3: 1}
        result = ndcg_at_k([1, 2, 3], rel_map, 3)
        assert np.isclose(result, 1.0)

    def test_no_relevant(self):
        assert ndcg_at_k([1, 2, 3], {}, 3) == 0.0

    def test_worst_ranking(self):
        rel_map = {1: 1, 2: 2, 3: 3}
        result = ndcg_at_k([1, 2, 3], rel_map, 3)
        assert 0.0 < result < 1.0


class TestBootstrapCI:
    def test_returns_float(self):
        result = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5])
        assert isinstance(result, float)
        assert result >= 0.0

    def test_constant_scores(self):
        result = bootstrap_ci([0.5, 0.5, 0.5, 0.5])
        assert np.isclose(result, 0.0)

    def test_single_score(self):
        assert bootstrap_ci([0.5]) == 0.0
