"""Streamlit entry point for the Competitor Analysis dashboard."""

import streamlit as st


st.set_page_config(
    page_title="Competitor Analysis",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Competitor Analysis")
st.write(
    "Discover competing apps, compare their market signals, and turn user "
    "feedback into a clearer product strategy."
)

st.divider()

features_column, roadmap_column = st.columns(2, gap="large")

with features_column:
    st.subheader("Key Features")
    st.markdown(
        """
        - Search apps by keyword
        - Browse results in a detailed table
        - Compare a focused shortlist side by side
        - Explore interactive visualizations for ratings, genres, and free vs paid apps
        - Generate a word cloud from app descriptions
        - Analyze sentiment and recurring topics in user reviews
        """
    )

with roadmap_column:
    st.subheader("Roadmap / Improvements")
    st.markdown(
        """
        - Save named competitor watchlists
        - Track rating and review trends over time
        - Add richer filters and market-level comparisons
        - Export shareable competitor reports
        - Expand multilingual sentiment support
        """
    )

st.divider()
st.subheader("How to run")
st.code("uv run streamlit run Home.py", language="bash")

st.info(
    "App data comes from the Google Play Store via the "
    "`google-play-scraper` package."
)
