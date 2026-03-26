"""Tests for feature engineering utilities."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class TestTfidfSharedVocab:
    def test_same_feature_dimensions(self):
        us_overviews = pd.Series(["Batman fights crime in Gotham", "A thief enters dreams"])
        kr_overviews = pd.Series(["A detective investigates murders", "Zombies attack a train"])

        combined = pd.concat([us_overviews, kr_overviews], ignore_index=True)
        vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        vectorizer.fit(combined)

        us_tfidf = vectorizer.transform(us_overviews)
        kr_tfidf = vectorizer.transform(kr_overviews)
        assert us_tfidf.shape[1] == kr_tfidf.shape[1]

    def test_nonzero_features(self):
        overviews = pd.Series(["A thrilling action movie about heroes", "A romantic comedy about love"])
        vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        tfidf = vectorizer.fit_transform(overviews)
        for i in range(tfidf.shape[0]):
            assert tfidf[i].nnz > 0


class TestGenreEncoding:
    def test_parse_genres(self):
        parsed = set("Action|Crime|Drama".split("|"))
        assert parsed == {"Action", "Crime", "Drama"}

    def test_multihot_shape(self):
        all_genres = ["Action", "Comedy", "Drama", "Horror", "Thriller"]
        genre_to_idx = {g: i for i, g in enumerate(all_genres)}

        movie_genres = [{"Action", "Drama"}, {"Comedy"}, {"Horror", "Thriller"}]
        mat = np.zeros((len(movie_genres), len(all_genres)), dtype=np.float32)
        for i, gs in enumerate(movie_genres):
            for g in gs:
                mat[i, genre_to_idx[g]] = 1.0

        assert mat.shape == (3, 5)
        assert mat[0, genre_to_idx["Action"]] == 1.0
        assert mat[0, genre_to_idx["Comedy"]] == 0.0
        assert mat[1].sum() == 1.0
        assert mat[2].sum() == 2.0
