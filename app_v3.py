"""Netflix-style Streamlit UI for browsing Korean movie recommendations."""

from __future__ import annotations

import random

import pandas as pd
import streamlit as st

from models_v2 import DEFAULT_WEIGHTS, get_recommendations, load_all_features, load_dataframes


TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w300"
ROW_COUNT = 5
REC_COUNT = 4
DEFAULT_ROW_TITLES = [
    "Interstellar",
    "The Dark Knight",
    "Inception",
    "The Matrix",
    "Fight Club",
    "The Shawshank Redemption",
]


st.set_page_config(
    page_title="K-Movie Recommender v3",
    page_icon="🎬",
    layout="wide",
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    us_df, kr_df = load_dataframes()
    features = load_all_features()
    return us_df, kr_df, features


def inject_css() -> None:
    """Apply a bright, Netflix-inspired browse layout."""
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, #f0f1f3 0%, #f5f6f8 34%),
                linear-gradient(180deg, #f3f4f6 0%, #eef1f4 100%);
            color: #111111;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1440px;
        }
        h1, h2, h3 {
            color: #111111;
            letter-spacing: -0.02em;
        }
        .hero {
            background: linear-gradient(135deg, #f7f8fa 0%, #eceff3 100%);
            border: 1px solid rgba(17, 17, 17, 0.08);
            border-radius: 28px;
            padding: 1.8rem 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 24px 60px rgba(17, 17, 17, 0.06);
        }
        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            line-height: 1.0;
            margin: 0 0 0.5rem 0;
        }
        .hero-copy {
            font-size: 1rem;
            color: #4a4a4a;
            margin: 0;
            max-width: 880px;
        }
        .browse-caption {
            color: #666666;
            font-size: 0.95rem;
            margin-bottom: 0.6rem;
        }
        .filter-panel {
            background: #f7f8fa;
            border: 1px solid rgba(17, 17, 17, 0.08);
            border-radius: 18px;
            padding: 0.8rem 0.95rem 0.2rem 0.95rem;
            box-shadow: 0 10px 24px rgba(17, 17, 17, 0.05);
            margin-bottom: 0.6rem;
        }
        .filter-title {
            font-size: 0.9rem;
            font-weight: 800;
            color: #111111;
            margin-bottom: 0.1rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .filter-copy {
            color: #666666;
            font-size: 0.82rem;
            margin-bottom: 0.35rem;
        }
        .row-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin: 1.6rem 0 0.8rem 0;
        }
        .movie-rank {
            display: inline-block;
            background: #e50914;
            color: #ffffff;
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            font-size: 0.72rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }
        .movie-name {
            font-weight: 800;
            font-size: 1rem;
            line-height: 1.2;
            margin: 0.5rem 0 0.2rem 0;
            color: #111111;
        }
        .movie-link {
            color: inherit;
            text-decoration: none;
        }
        .movie-link:hover {
            text-decoration: underline;
        }
        .movie-subtitle {
            color: #666666;
            font-size: 0.86rem;
            margin-bottom: 0.5rem;
        }
        .movie-stats {
            color: #242424;
            font-size: 0.85rem;
            line-height: 1.45;
        }
        .movie-overview {
            color: #5a5a5a;
            font-size: 0.82rem;
            line-height: 1.45;
            margin-top: 0.55rem;
        }
        div[data-testid="stHorizontalBlock"] img {
            border-radius: 18px;
        }
        .source-chip {
            display: inline-block;
            background: #111111;
            color: #ffffff;
            border-radius: 999px;
            padding: 0.22rem 0.55rem;
            font-size: 0.72rem;
            font-weight: 700;
            margin-top: 0.45rem;
        }
        div[data-testid="stSidebar"] {
            background: #eef1f4;
            border-left: 1px solid rgba(17, 17, 17, 0.06);
        }
        div[data-testid="stButton"] > button {
            background: #f7f8fa;
            color: #111111;
            border: 1px solid rgba(17, 17, 17, 0.12);
        }
        div[data-testid="stButton"] > button:hover {
            background: #eceff3;
            color: #111111;
            border-color: rgba(17, 17, 17, 0.18);
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="select"] input,
        div[data-baseweb="select"] span,
        div[data-testid="stTextInput"] input {
            background: #f7f8fa;
            color: #111111;
        }
        div[data-testid="stTextInput"] label,
        div[data-testid="stMultiSelect"] label,
        div[data-testid="stSlider"] label {
            font-size: 0.82rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_year(value: object) -> str:
    if pd.notna(value):
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return str(value)
    return "?"


def format_score(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


def abbreviate(text: object, limit: int = 140) -> str:
    if pd.isna(text):
        return ""
    value = str(text).strip()
    if len(value) <= limit:
        return value
    return f"{value[:limit - 3].rstrip()}..."


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return DEFAULT_WEIGHTS.copy()
    return {key: value / total for key, value in weights.items()}


def genre_options(df: pd.DataFrame) -> list[str]:
    values = set()
    for genres in df["genres"].dropna():
        values.update(genre for genre in str(genres).split("|") if genre)
    return sorted(values)


def year_bounds(df: pd.DataFrame) -> tuple[int, int]:
    valid_years = pd.to_numeric(df["year"], errors="coerce").dropna()
    if valid_years.empty:
        return 1950, 2026
    return int(valid_years.min()), int(valid_years.max())


def default_year_range(min_year: int, max_year: int) -> tuple[int, int]:
    start = min(max(2006, min_year), max_year)
    end = min(max_year, 2026)
    if start > end:
        start = min_year
        end = max_year
    return start, end


def default_rating_range(min_rating: float, max_rating: float) -> tuple[float, float]:
    start = min(max(7.5, min_rating), 10.0)
    end = 10.0
    if start > end:
        start = min_rating
        end = 10.0
    return round(start, 1), round(end, 1)


def filter_catalog(
    df: pd.DataFrame,
    selected_genres: list[str],
    selected_years: tuple[int, int],
    selected_ratings: tuple[float, float],
) -> pd.DataFrame:
    filtered = df.copy()
    filtered["year"] = pd.to_numeric(filtered["year"], errors="coerce")
    filtered["rating"] = pd.to_numeric(filtered["rating"], errors="coerce")
    filtered = filtered[
        filtered["year"].between(selected_years[0], selected_years[1], inclusive="both")
    ]
    filtered = filtered[
        filtered["rating"].between(selected_ratings[0], selected_ratings[1], inclusive="both")
    ]
    if selected_genres:
        genre_set = set(selected_genres)
        filtered = filtered[
            filtered["genres"].fillna("").apply(
                lambda value: bool(set(str(value).split("|")) & genre_set)
            )
        ]
    return filtered


def pick_default_titles(us_df: pd.DataFrame, row_count: int) -> list[str]:
    available = set(us_df["title"].dropna().astype(str))
    chosen = [title for title in DEFAULT_ROW_TITLES if title in available]
    if len(chosen) >= row_count:
        return chosen[:row_count]

    fallback = (
        us_df.dropna(subset=["title"])
        .sort_values(["rating", "vote_count"], ascending=[False, False])
        ["title"]
        .astype(str)
        .tolist()
    )
    for title in fallback:
        if title not in chosen:
            chosen.append(title)
        if len(chosen) >= row_count:
            break
    return chosen[:row_count]


def pick_random_titles(us_df: pd.DataFrame, row_count: int, min_rating: float = 8.0) -> list[str]:
    pool = us_df[us_df["rating"] >= min_rating].dropna(subset=["title"]).copy()
    if pool.empty:
        pool = us_df.dropna(subset=["title"]).copy()

    sample_size = min(row_count, len(pool))
    if sample_size == 0:
        return []

    sampled = pool.sample(n=sample_size, random_state=random.randint(0, 10_000_000))
    return sampled["title"].astype(str).tolist()


def rating_bounds(df: pd.DataFrame) -> tuple[float, float]:
    valid_ratings = pd.to_numeric(df["rating"], errors="coerce").dropna()
    if valid_ratings.empty:
        return 0.0, 10.0
    return round(float(valid_ratings.min()), 1), round(float(valid_ratings.max()), 1)


def tmdb_movie_url(movie: pd.Series) -> str | None:
    tmdb_id = movie.get("tmdb_id")
    if pd.isna(tmdb_id):
        return None
    try:
        return f"https://www.themoviedb.org/movie/{int(float(tmdb_id))}"
    except (TypeError, ValueError):
        return None


def render_recommendation_card(movie: pd.Series, rank: int) -> None:
    poster_path = movie.get("poster_path")
    if pd.notna(poster_path) and str(poster_path).strip():
        st.image(f"{TMDB_IMG_BASE}{poster_path}", use_container_width=True)
    else:
        st.markdown(
            '<div style="height:310px;border-radius:18px;background:linear-gradient(180deg,#ededed 0%,#d9d9d9 100%);"></div>',
            unsafe_allow_html=True,
        )

    original_title = str(movie.get("original_title") or "")
    if original_title and original_title != str(movie.get("title", "")):
        subtitle = f"{original_title} · {safe_year(movie.get('year'))}"
    else:
        subtitle = safe_year(movie.get("year"))

    genres = str(movie.get("genres", "")).replace("|", ", ")
    overview = abbreviate(movie.get("overview"), 135)
    score = movie.get("score", 0)
    movie_url = tmdb_movie_url(movie)
    title = str(movie.get("title", "Untitled"))
    title_html = (
        f'<a class="movie-link" href="{movie_url}" target="_blank">{title}</a>'
        if movie_url
        else title
    )
    st.markdown(f'<div class="movie-rank">#{rank} Match · {float(score):.3f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="movie-name">{title_html}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="movie-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="movie-stats"><strong>TMDB:</strong> {format_score(movie.get("rating"))} / 10</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="movie-stats"><strong>Genres:</strong> {genres or "Unknown"}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="movie-overview">{overview or "No overview available."}</div>', unsafe_allow_html=True)


def render_source_card(us_movie: pd.Series) -> None:
    source_title = str(us_movie["title"])
    source_year = safe_year(us_movie.get("year"))
    source_rating = format_score(us_movie.get("rating"))
    source_genres = str(us_movie.get("genres", "")).replace("|", ", ")
    source_director = str(us_movie.get("director", "") or "")
    source_overview = abbreviate(us_movie.get("overview"), 185)
    source_poster = us_movie.get("poster_path")
    source_url = tmdb_movie_url(us_movie)

    if pd.notna(source_poster) and str(source_poster).strip():
        st.image(f"{TMDB_IMG_BASE}{source_poster}", use_container_width=True)
    else:
        st.markdown(
            '<div style="height:250px;border-radius:18px;background:linear-gradient(180deg,#f2f2f2 0%,#dddddd 100%);"></div>',
            unsafe_allow_html=True,
        )
    source_title_html = (
        f'<a class="movie-link" href="{source_url}" target="_blank">{source_title}</a>'
        if source_url
        else source_title
    )
    st.markdown(f'<div class="movie-name">{source_title_html}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="movie-subtitle">{source_year}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="movie-stats"><strong>TMDB:</strong> {source_rating} / 10</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="movie-stats"><strong>Genres:</strong> {source_genres or "Unknown"}</div>', unsafe_allow_html=True)
    if source_director and source_director.lower() != "nan":
        st.markdown(f'<div class="movie-stats"><strong>Director:</strong> {source_director}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="movie-overview">{source_overview or "No overview available."}</div>', unsafe_allow_html=True)


def render_recommendation_row(us_movie: pd.Series, recs: pd.DataFrame) -> None:
    source_title = str(us_movie["title"])
    source_year = safe_year(us_movie.get("year"))
    st.markdown(f'<div class="row-title">{source_title} ({source_year})</div>', unsafe_allow_html=True)

    columns = st.columns(REC_COUNT + 1, gap="medium")
    with columns[0]:
        render_source_card(us_movie)
    for rank, (column, (_, movie)) in enumerate(zip(columns[1:], recs.iterrows()), start=1):
        with column:
            render_recommendation_card(movie, rank)


def main() -> None:
    inject_css()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Korean Movie Recommender v3</div>
            <p class="hero-copy">
                Browse Korean recommendations in rows, starting from strong US movie anchors.
                Each row is sorted left to right by similarity, so the first cards are the closest matches.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        us_df, kr_df, features = load_data()
    except FileNotFoundError as exc:
        st.error(f"Data not found. Run `make ingest && make features-v2` first.\n\n{exc}")
        return

    us_min_year, us_max_year = year_bounds(us_df)
    kr_min_year, kr_max_year = year_bounds(kr_df)
    default_us_year_range = default_year_range(us_min_year, us_max_year)
    default_kr_year_range = default_year_range(kr_min_year, kr_max_year)
    us_min_rating, us_max_rating = rating_bounds(us_df)
    kr_min_rating, kr_max_rating = rating_bounds(kr_df)
    default_us_rating_range = default_rating_range(us_min_rating, us_max_rating)
    default_kr_rating_range = default_rating_range(kr_min_rating, kr_max_rating)
    us_genres = genre_options(us_df)
    kr_genres = genre_options(kr_df)
    all_genres = sorted(set(us_genres) | set(kr_genres))

    with st.sidebar:
        st.header("Browse Settings")
        model = st.selectbox(
            "Recommendation model",
            ["hybrid", "embedding", "tfidf"],
            format_func=lambda value: {
                "hybrid": "Hybrid v2",
                "embedding": "Sentence Embeddings",
                "tfidf": "TF-IDF",
            }[value],
        )

        weights = None
        if model == "hybrid":
            st.subheader("Hybrid Weights")
            custom_weights = {
                "text": st.slider("Text", 0.0, 1.0, float(DEFAULT_WEIGHTS["text"]), 0.05),
                "genre": st.slider("Genre", 0.0, 1.0, float(DEFAULT_WEIGHTS["genre"]), 0.05),
                "keyword": st.slider("Keyword", 0.0, 1.0, float(DEFAULT_WEIGHTS["keyword"]), 0.05),
                "cast": st.slider("Cast", 0.0, 1.0, float(DEFAULT_WEIGHTS["cast"]), 0.05),
                "year": st.slider("Year", 0.0, 1.0, float(DEFAULT_WEIGHTS["year"]), 0.05),
            }
            weights = normalize_weights(custom_weights)
            st.caption(
                "Normalized mix: "
                f"text {weights['text']:.2f}, genre {weights['genre']:.2f}, "
                f"keyword {weights['keyword']:.2f}, cast {weights['cast']:.2f}, year {weights['year']:.2f}"
            )
        selected_genres = st.multiselect("Genres", all_genres)

    controls_col, us_filter_col, kr_filter_col = st.columns([1.05, 1.7, 1.7], gap="medium")
    with controls_col:
        st.markdown(
            '<div class="filter-panel"><div class="filter-title">Browse</div><div class="filter-copy">Pick a US anchor fast.</div></div>',
            unsafe_allow_html=True,
        )
        randomize = st.button("Random US Movies", use_container_width=True)
        us_search_placeholder = st.empty()
    with us_filter_col:
        st.markdown(
            '<div class="filter-panel"><div class="filter-title">US Movies</div><div class="filter-copy">Controls for source movies shown in each row.</div></div>',
            unsafe_allow_html=True,
        )
        us_year_range = st.slider(
            "US movie years",
            min_value=us_min_year,
            max_value=us_max_year,
            value=default_us_year_range,
        )
        us_rating_range = st.slider(
            "US TMDB rating",
            min_value=us_min_rating,
            max_value=10.0,
            value=default_us_rating_range,
            step=0.1,
        )
    with kr_filter_col:
        st.markdown(
            '<div class="filter-panel"><div class="filter-title">Korean Movies</div><div class="filter-copy">Controls for recommendation candidates.</div></div>',
            unsafe_allow_html=True,
        )
        kr_year_range = st.slider(
            "Korean movie years",
            min_value=kr_min_year,
            max_value=kr_max_year,
            value=default_kr_year_range,
        )
        kr_rating_range = st.slider(
            "Korean TMDB rating",
            min_value=kr_min_rating,
            max_value=10.0,
            value=default_kr_rating_range,
            step=0.1,
        )

    filtered_us_df = filter_catalog(us_df, selected_genres, us_year_range, us_rating_range)
    filtered_kr_df = filter_catalog(kr_df, selected_genres, kr_year_range, kr_rating_range)
    filtered_us_titles = filtered_us_df["title"].dropna().astype(str).tolist()
    with controls_col:
        selected_us_movie = us_search_placeholder.selectbox(
            "Search US movie",
            options=filtered_us_titles,
            index=None,
            placeholder="Type to search, e.g. Interstellar",
        )
    filter_signature = (
        tuple(selected_genres),
        us_year_range,
        us_rating_range,
        tuple(selected_genres),
        kr_year_range,
        kr_rating_range,
    )

    if st.session_state.get("selected_us_filter_signature") != filter_signature:
        st.session_state.selected_us_titles = pick_default_titles(filtered_us_df, ROW_COUNT)
        st.session_state.selected_us_filter_signature = filter_signature

    if randomize:
        st.session_state.selected_us_titles = pick_random_titles(filtered_us_df, ROW_COUNT, min_rating=8.0)
        st.session_state.selected_us_filter_signature = filter_signature

    available_titles = set(filtered_us_df["title"].dropna().astype(str))
    current_titles = [title for title in st.session_state.get("selected_us_titles", []) if title in available_titles]

    if selected_us_movie:
        current_titles = [selected_us_movie] + [title for title in current_titles if title != selected_us_movie]

    if len(current_titles) < ROW_COUNT:
        fallback_titles = pick_default_titles(filtered_us_df, ROW_COUNT)
        seen = set(current_titles)
        for title in fallback_titles:
            if title not in seen:
                current_titles.append(title)
                seen.add(title)
            if len(current_titles) >= ROW_COUNT:
                break
        st.session_state.selected_us_titles = current_titles

    if filtered_us_df.empty:
        st.info("No US movies match the current filters. Try widening the US year or genre settings.")
        return

    if filtered_kr_df.empty:
        st.info("No Korean movies match the current filters. Try widening the Korean year or genre settings.")
        return

    for title in current_titles[:ROW_COUNT]:
        us_match = filtered_us_df[filtered_us_df["title"] == title]
        if us_match.empty:
            continue

        us_movie = us_match.iloc[0]
        rec_filters = {
            "year_min": kr_year_range[0],
            "year_max": kr_year_range[1],
            "min_rating": kr_rating_range[0],
        }
        if selected_genres:
            rec_filters["genre"] = selected_genres
        recs = get_recommendations(
            title,
            model,
            us_df,
            kr_df,
            features,
            top_k=REC_COUNT,
            weights=weights,
            filters=rec_filters,
        )
        recs = recs[
            pd.to_numeric(recs["rating"], errors="coerce").between(
                kr_rating_range[0], kr_rating_range[1], inclusive="both"
            )
        ]
        recs = recs.head(REC_COUNT)
        if recs.empty:
            continue
        render_recommendation_row(us_movie, recs)


if __name__ == "__main__":
    main()
