"""Shared test fixtures with synthetic movie data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def us_df():
    """Small synthetic US movie catalog."""
    return pd.DataFrame({
        "tmdb_id": [1, 2, 3, 4, 5],
        "title": [
            "The Dark Knight", "Inception", "Parasite",
            "The Matrix", "Interstellar",
        ],
        "original_title": [
            "The Dark Knight", "Inception", "Parasite",
            "The Matrix", "Interstellar",
        ],
        "year": [2008, 2010, 2019, 1999, 2014],
        "genres": [
            "Action|Crime|Drama", "Action|Sci-Fi|Thriller",
            "Comedy|Drama|Thriller", "Action|Sci-Fi",
            "Adventure|Drama|Sci-Fi",
        ],
        "overview": [
            "Batman fights the Joker in Gotham City",
            "A thief enters dreams to plant ideas",
            "A poor family infiltrates a wealthy household",
            "A hacker discovers reality is a simulation",
            "Astronauts travel through a wormhole to save humanity",
        ],
        "director": [
            "Christopher Nolan", "Christopher Nolan", "Bong Joon-ho",
            "Lana Wachowski", "Christopher Nolan",
        ],
        "cast_top5": [
            "Christian Bale|Heath Ledger", "Leonardo DiCaprio|Tom Hardy",
            "Song Kang-ho|Cho Yeo-jeong", "Keanu Reeves|Laurence Fishburne",
            "Matthew McConaughey|Anne Hathaway",
        ],
        "rating": [9.0, 8.8, 8.5, 8.7, 8.6],
        "vote_count": [25000, 30000, 20000, 22000, 28000],
        "runtime": [152, 148, 132, 136, 169],
        "popularity": [100.0, 95.0, 90.0, 85.0, 92.0],
        "poster_path": ["/p1.jpg", "/p2.jpg", "/p3.jpg", "/p4.jpg", "/p5.jpg"],
        "translated": [False, False, False, False, False],
    })


@pytest.fixture
def kr_df():
    """Small synthetic KR movie catalog."""
    return pd.DataFrame({
        "tmdb_id": [101, 102, 103, 104, 105],
        "title": [
            "Oldboy", "Memories of Murder", "Train to Busan",
            "The Handmaiden", "I Saw the Devil",
        ],
        "original_title": ["올드보이", "살인의 추억", "부산행", "아가씨", "악마를 보았다"],
        "year": [2003, 2003, 2016, 2016, 2010],
        "genres": [
            "Action|Drama|Mystery", "Crime|Drama|Mystery",
            "Action|Horror|Thriller", "Drama|Romance|Thriller",
            "Action|Crime|Horror",
        ],
        "overview": [
            "A man imprisoned for 15 years seeks vengeance",
            "Detectives investigate a series of murders in a rural town",
            "Passengers fight zombies on a train to Busan",
            "A handmaiden plots with a con man against her mistress",
            "A secret agent hunts a serial killer in a deadly game",
        ],
        "director": [
            "Park Chan-wook", "Bong Joon-ho", "Yeon Sang-ho",
            "Park Chan-wook", "Kim Jee-woon",
        ],
        "cast_top5": [
            "Choi Min-sik|Yoo Ji-tae", "Song Kang-ho|Kim Sang-kyung",
            "Gong Yoo|Ma Dong-seok", "Kim Min-hee|Ha Jung-woo",
            "Lee Byung-hun|Choi Min-sik",
        ],
        "rating": [8.4, 8.1, 7.6, 8.1, 7.8],
        "vote_count": [5000, 3000, 8000, 4000, 3500],
        "runtime": [120, 132, 118, 145, 144],
        "popularity": [50.0, 40.0, 70.0, 45.0, 42.0],
        "poster_path": ["/k1.jpg", "/k2.jpg", "/k3.jpg", "/k4.jpg", "/k5.jpg"],
        "translated": [False, False, False, False, False],
    })


@pytest.fixture
def synthetic_features(us_df, kr_df):
    """Synthetic feature matrices matching the fixture DataFrames."""
    rng = np.random.default_rng(42)
    n_us, n_kr = len(us_df), len(kr_df)

    # Sentence embeddings (L2-normalized)
    emb_us = rng.standard_normal((n_us, 384)).astype(np.float32)
    emb_us /= np.linalg.norm(emb_us, axis=1, keepdims=True)
    emb_kr = rng.standard_normal((n_kr, 384)).astype(np.float32)
    emb_kr /= np.linalg.norm(emb_kr, axis=1, keepdims=True)

    # TF-IDF (sparse)
    tfidf_us = sparse.random(n_us, 100, density=0.1, format="csr", dtype=np.float32,
                             random_state=42)
    tfidf_kr = sparse.random(n_kr, 100, density=0.1, format="csr", dtype=np.float32,
                             random_state=43)

    # Genre multi-hot (19 genres)
    genre_us = np.zeros((n_us, 19), dtype=np.float32)
    genre_kr = np.zeros((n_kr, 19), dtype=np.float32)
    for i in range(n_us):
        genre_us[i, :3] = 1.0
    for i in range(n_kr):
        genre_kr[i, :2] = 1.0

    # Keywords (50 cross-catalog)
    kw_us = np.zeros((n_us, 50), dtype=np.float32)
    kw_kr = np.zeros((n_kr, 50), dtype=np.float32)
    kw_us[0, :5] = 1.0
    kw_kr[0, :3] = 1.0

    # Year
    year_us = us_df["year"].values.astype(np.float64)
    year_kr = kr_df["year"].values.astype(np.float64)

    return {
        "emb_us": emb_us, "emb_kr": emb_kr,
        "tfidf_us": tfidf_us, "tfidf_kr": tfidf_kr,
        "genre_us": genre_us, "genre_kr": genre_kr,
        "kw_us": kw_us, "kw_kr": kw_kr,
        "year_us": year_us, "year_kr": year_kr,
    }
