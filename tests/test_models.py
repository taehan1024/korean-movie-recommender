"""Tests for recommendation models."""

import numpy as np

from models import (
    _jaccard_scores,
    _minmax,
    _year_scores,
    get_recommendations,
    resolve_title,
)


class TestResolveTitle:
    def test_exact_match(self, us_df):
        assert resolve_title("The Dark Knight", us_df) == 0

    def test_case_insensitive(self, us_df):
        assert resolve_title("the dark knight", us_df) == 0

    def test_contains_match(self, us_df):
        assert resolve_title("dark knight", us_df) == 0

    def test_fuzzy_match(self, us_df):
        assert resolve_title("The Dark Knigth", us_df) == 0

    def test_not_found(self, us_df):
        assert resolve_title("NonexistentMovie123456", us_df) is None


class TestJaccardScores:
    def test_identity(self):
        vec = np.array([1.0, 0.0, 1.0, 1.0])
        matrix = vec.reshape(1, -1)
        scores = _jaccard_scores(vec, matrix)
        assert np.isclose(scores[0], 1.0)

    def test_disjoint(self):
        query = np.array([1.0, 0.0, 0.0])
        matrix = np.array([[0.0, 1.0, 0.0]])
        scores = _jaccard_scores(query, matrix)
        assert np.isclose(scores[0], 0.0)

    def test_partial_overlap(self):
        query = np.array([1.0, 1.0, 0.0])
        matrix = np.array([[1.0, 0.0, 1.0]])
        scores = _jaccard_scores(query, matrix)
        assert np.isclose(scores[0], 1.0 / 3.0)

    def test_empty_query(self):
        query = np.array([0.0, 0.0, 0.0])
        matrix = np.array([[1.0, 1.0, 0.0]])
        scores = _jaccard_scores(query, matrix)
        assert np.isclose(scores[0], 0.0)


class TestMinmax:
    def test_range(self):
        arr = np.array([1.0, 5.0, 3.0, 10.0, 2.0])
        result = _minmax(arr)
        assert np.isclose(result.min(), 0.0)
        assert np.isclose(result.max(), 1.0)

    def test_constant_array(self):
        arr = np.array([5.0, 5.0, 5.0])
        result = _minmax(arr)
        assert np.allclose(result, 0.0)

    def test_two_values(self):
        arr = np.array([0.0, 1.0])
        result = _minmax(arr)
        assert np.isclose(result[0], 0.0)
        assert np.isclose(result[1], 1.0)


class TestYearScores:
    def test_same_year(self, synthetic_features):
        synthetic_features["year_us"][0] = 2010
        synthetic_features["year_kr"][0] = 2010
        scores = _year_scores(0, synthetic_features)
        assert np.isclose(scores[0], 1.0)

    def test_distant_years(self, synthetic_features):
        synthetic_features["year_us"][0] = 2020
        synthetic_features["year_kr"][0] = 1970
        scores = _year_scores(0, synthetic_features)
        assert scores[0] < 0.1

    def test_nan_year(self, synthetic_features):
        synthetic_features["year_us"][0] = np.nan
        scores = _year_scores(0, synthetic_features)
        assert np.allclose(scores, 0.5)


class TestGetRecommendations:
    def test_returns_correct_count(self, us_df, kr_df, synthetic_features):
        recs = get_recommendations(
            "Inception", "embedding", us_df, kr_df, synthetic_features, top_k=3,
        )
        assert len(recs) == 3

    def test_bad_title_returns_empty(self, us_df, kr_df, synthetic_features):
        recs = get_recommendations(
            "ZZZZZ_NONEXISTENT", "embedding", us_df, kr_df, synthetic_features,
        )
        assert recs.empty or "error" in recs.columns

    def test_has_score_column(self, us_df, kr_df, synthetic_features):
        recs = get_recommendations(
            "The Matrix", "embedding", us_df, kr_df, synthetic_features, top_k=3,
        )
        assert "score" in recs.columns

    def test_scores_descending(self, us_df, kr_df, synthetic_features):
        recs = get_recommendations(
            "The Matrix", "embedding", us_df, kr_df, synthetic_features, top_k=5,
        )
        scores = recs["score"].tolist()
        assert scores == sorted(scores, reverse=True)
