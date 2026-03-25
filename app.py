"""Streamlit app for cross-cultural movie recommendations."""

import streamlit as st
import pandas as pd

from models import (
    get_recommendations,
    load_all_features,
    load_dataframes,
    resolve_title,
    DEFAULT_WEIGHTS,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="K-Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w200"


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    us_df, kr_df = load_dataframes()
    features = load_all_features()
    return us_df, kr_df, features


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def render_movie_card(movie: pd.Series, rank: int) -> None:
    """Render a single recommendation card."""
    col_img, col_info = st.columns([1, 3])

    with col_img:
        poster = movie.get("poster_path", "")
        if poster and str(poster) != "nan":
            st.image(f"{TMDB_IMG_BASE}{poster}", width=150)
        else:
            st.markdown("*No poster*")

    with col_info:
        score = movie.get("score", 0)
        year = int(movie["year"]) if pd.notna(movie.get("year")) else "?"
        rating = movie.get("rating", "?")

        st.markdown(f"### {rank}. {movie['title']} ({year})")
        if movie.get("original_title") and movie["original_title"] != movie["title"]:
            st.caption(movie["original_title"])

        st.markdown(f"**Score:** {score:.3f} &nbsp; **Rating:** {rating} &nbsp; "
                    f"**Genres:** {movie.get('genres', '').replace('|', ', ')}")

        if movie.get("director"):
            st.markdown(f"**Director:** {movie['director']}")

        overview = movie.get("overview", "")
        if overview and str(overview) != "nan":
            st.markdown(f"_{overview[:300]}{'...' if len(str(overview)) > 300 else ''}_")

        # Explanation
        explanation = movie.get("explanation", {})
        if explanation and isinstance(explanation, dict):
            with st.expander("Why this recommendation?"):
                parts = []
                if explanation.get("shared_genres"):
                    parts.append(f"**Shared genres:** {', '.join(explanation['shared_genres'])}")
                if explanation.get("shared_terms"):
                    parts.append(f"**Key terms:** {', '.join(explanation['shared_terms'])}")
                if explanation.get("shared_cast_crew"):
                    parts.append(f"**Shared cast/crew:** {', '.join(explanation['shared_cast_crew'])}")
                if explanation.get("semantic_similarity") is not None:
                    parts.append(f"**Semantic similarity:** {explanation['semantic_similarity']:.3f}")
                st.markdown("  \n".join(parts) if parts else "Based on overall feature similarity.")

    st.divider()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    st.title("K-Movie Recommender")
    st.markdown("Find Korean movies similar to your favorite American films.")

    # Load data
    try:
        us_df, kr_df, features = load_data()
    except FileNotFoundError as e:
        st.error(f"Data not found. Run the pipeline first:\n```\nmake ingest && make features\n```\n\n{e}")
        return

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")

        # Model selector
        model = st.selectbox(
            "Recommendation model",
            ["hybrid", "embedding", "tfidf"],
            format_func=lambda x: {
                "tfidf": "TF-IDF + Cosine (baseline)",
                "embedding": "Sentence Embeddings + Cosine",
                "hybrid": "Hybrid (text + genre + cast)",
            }[x],
        )

        # Hybrid weight sliders
        weights = None
        if model == "hybrid":
            st.subheader("Hybrid Weights")
            w_text = st.slider("Text (synopsis)", 0.0, 1.0, DEFAULT_WEIGHTS["text"], 0.05)
            w_genre = st.slider("Genre", 0.0, 1.0, DEFAULT_WEIGHTS["genre"], 0.05)
            w_cast = st.slider("Cast/Crew", 0.0, 1.0, DEFAULT_WEIGHTS["cast"], 0.05)
            total = w_text + w_genre + w_cast
            if total > 0:
                weights = {
                    "text": w_text / total,
                    "genre": w_genre / total,
                    "cast": w_cast / total,
                }
            st.caption(f"Normalized: text={weights['text']:.2f}, "
                      f"genre={weights['genre']:.2f}, cast={weights['cast']:.2f}")

        st.subheader("Filters")

        # Genre filter
        all_genres = sorted(set(
            g for gs in kr_df["genres"].dropna()
            for g in str(gs).split("|") if g
        ))
        selected_genres = st.multiselect("Genres", all_genres)

        # Year range
        min_year = int(kr_df["year"].min()) if kr_df["year"].notna().any() else 1950
        max_year = int(kr_df["year"].max()) if kr_df["year"].notna().any() else 2026
        year_range = st.slider("Year range", min_year, max_year, (min_year, max_year))

        # Min rating
        min_rating = st.slider("Minimum rating", 0.0, 10.0, 0.0, 0.5)

        # Number of results
        top_k = st.slider("Number of results", 5, 20, 10)

    # --- Search ---
    query = st.text_input(
        "Enter an American movie title",
        placeholder="e.g., The Dark Knight, Inception, Parasite...",
    )

    if query:
        # Check if title resolves
        us_idx = resolve_title(query, us_df)
        if us_idx is not None:
            matched = us_df.iloc[us_idx]
            st.success(f"Matched: **{matched['title']}** ({int(matched['year']) if pd.notna(matched.get('year')) else '?'})")
        else:
            st.warning(f"No match found for '{query}'. Try a different title.")
            # Show suggestions
            suggestions = us_df[
                us_df["title"].str.lower().str.contains(query[:3].lower(), na=False)
            ].head(5)
            if not suggestions.empty:
                st.markdown("**Did you mean:**")
                for _, s in suggestions.iterrows():
                    st.markdown(f"- {s['title']} ({int(s['year']) if pd.notna(s.get('year')) else '?'})")
            return

        # Build filters
        filters = {}
        if selected_genres:
            filters["genre"] = selected_genres
        if year_range != (min_year, max_year):
            filters["year_min"] = year_range[0]
            filters["year_max"] = year_range[1]
        if min_rating > 0:
            filters["min_rating"] = min_rating

        # Get recommendations
        with st.spinner("Finding recommendations..."):
            recs = get_recommendations(
                query, model, us_df, kr_df, features,
                top_k=top_k, weights=weights,
                filters=filters if filters else None,
            )

        if recs.empty:
            st.info("No recommendations found with current filters. Try adjusting them.")
            return

        st.markdown(f"### Top {len(recs)} Korean Movie Recommendations")
        st.caption(f"Model: {model} | Results: {len(recs)}")

        for i, (_, movie) in enumerate(recs.iterrows(), 1):
            render_movie_card(movie, i)


if __name__ == "__main__":
    main()
