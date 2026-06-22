"""Interactive visual analysis of the current Google Play result set."""

import math

import pandas as pd
import plotly.express as px
import streamlit as st

import utils


st.set_page_config(page_title="Visualizations", layout="wide")

if (
    "results_df" not in st.session_state
    or st.session_state["results_df"].empty
):
    st.warning("Go to the Search page first and run a search.")
    st.stop()

df = st.session_state["results_df"].copy()

st.title("Visualizations")
st.caption("Explore the market shape of your current Google Play search.")

app_ids = df["appId"].dropna().unique().tolist()
selected_app_ids = st.sidebar.multiselect("Filter by App ID", options=app_ids)
pricing_filter = st.sidebar.selectbox(
    "Free / Paid",
    options=["All", "Free only", "Paid only"],
)

filtered_df = df.copy()
if selected_app_ids:
    filtered_df = filtered_df[filtered_df["appId"].isin(selected_app_ids)]

free_mask = filtered_df["free"].fillna(False).astype(bool)
if pricing_filter == "Free only":
    filtered_df = filtered_df[free_mask]
elif pricing_filter == "Paid only":
    filtered_df = filtered_df[~free_mask]

filtered_df = filtered_df.copy()
st.sidebar.metric("Apps in view", len(filtered_df))

if filtered_df.empty:
    st.warning("No apps match the selected filters.")
    st.stop()


ratings_tab, categories_tab, installs_tab, wordcloud_tab = st.tabs(
    ["Ratings", "Categories", "Size & Installs", "Word Cloud"]
)

with ratings_tab:
    ratings_histogram = px.histogram(
        filtered_df,
        x="score",
        range_x=[0, 5],
        title="Distribution of App Ratings",
        labels={"score": "Rating", "count": "Number of apps"},
        color_discrete_sequence=["#4F46E5"],
    )
    ratings_histogram.update_traces(
        xbins={"start": 0, "end": 5, "size": 0.5}
    )
    ratings_histogram.update_layout(
        xaxis_title="Rating (0–5)",
        yaxis_title="Number of apps",
        bargap=0.08,
    )
    st.plotly_chart(ratings_histogram, use_container_width=True)

    top_rated = (
        filtered_df.dropna(subset=["score"])
        .nlargest(10, "score")
        .sort_values("score")
    )
    top_rated_chart = px.bar(
        top_rated,
        x="score",
        y="title",
        orientation="h",
        range_x=[0, 5],
        title="Top 10 Apps by Rating",
        labels={"score": "Rating", "title": "App"},
        color="score",
        color_continuous_scale="Blues",
    )
    top_rated_chart.update_layout(
        xaxis_title="Rating (0–5)",
        yaxis_title="App title",
        coloraxis_showscale=False,
    )
    st.plotly_chart(top_rated_chart, use_container_width=True)

with categories_tab:
    pricing_counts = (
        filtered_df["free"]
        .fillna(False)
        .astype(bool)
        .map({True: "Free", False: "Paid"})
        .value_counts()
        .rename_axis("pricing")
        .reset_index(name="apps")
    )
    pricing_chart = px.pie(
        pricing_counts,
        names="pricing",
        values="apps",
        title="Free vs Paid Apps",
        labels={"pricing": "Pricing model", "apps": "Number of apps"},
        color="pricing",
        color_discrete_map={"Free": "#4F46E5", "Paid": "#F59E0B"},
    )
    pricing_chart.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(pricing_chart, use_container_width=True)

    genre_counts = (
        filtered_df["genre"]
        .fillna("Unknown")
        .replace("", "Unknown")
        .value_counts()
        .head(10)
        .sort_values()
        .rename_axis("genre")
        .reset_index(name="apps")
    )
    genre_chart = px.bar(
        genre_counts,
        x="apps",
        y="genre",
        orientation="h",
        title="Top 10 App Genres",
        labels={"apps": "Number of apps", "genre": "Genre"},
        color="apps",
        color_continuous_scale="Blues",
    )
    genre_chart.update_layout(
        xaxis_title="Number of apps",
        yaxis_title="Genre",
        coloraxis_showscale=False,
    )
    st.plotly_chart(genre_chart, use_container_width=True)

with installs_tab:
    filtered_df["installs_numeric"] = filtered_df["installs"].map(
        utils.parse_installs
    )
    filtered_df["log10_installs"] = filtered_df["installs_numeric"].map(
        lambda installs: math.log10(installs) if installs > 0 else None
    )

    top_installed = filtered_df.nlargest(10, "installs_numeric").sort_values(
        "installs_numeric"
    )
    installs_chart = px.bar(
        top_installed,
        x="installs_numeric",
        y="title",
        orientation="h",
        title="Top 10 Apps by Installs",
        labels={"installs_numeric": "Installs", "title": "App"},
        color="installs_numeric",
        color_continuous_scale="Blues",
    )
    installs_chart.update_layout(
        xaxis_title="Estimated installs",
        yaxis_title="App title",
        coloraxis_showscale=False,
    )
    st.plotly_chart(installs_chart, use_container_width=True)

    installs_scatter = px.scatter(
        filtered_df,
        x="log10_installs",
        y="score",
        color="free",
        hover_name="title",
        title="Ratings vs Install Reach",
        labels={
            "log10_installs": "log10(Installs)",
            "score": "Rating",
            "free": "Free app",
        },
        color_discrete_map={True: "#4F46E5", False: "#F59E0B"},
    )
    installs_scatter.update_layout(
        xaxis_title="Install reach (log10)",
        yaxis_title="Rating (0–5)",
    )
    st.plotly_chart(installs_scatter, use_container_width=True)

with wordcloud_tab:
    try:
        from wordcloud import WordCloud
    except ImportError:
        st.info(
            "Word Cloud support is not installed. Install it with "
            "`uv pip install wordcloud`."
        )
    else:
        description_text = " ".join(
            filtered_df["description"].dropna().astype(str)
        ).strip()
        if not description_text:
            st.info("No app descriptions are available for the current filters.")
        else:
            try:
                word_cloud = WordCloud(
                    width=800,
                    height=400,
                    background_color="white",
                    max_words=100,
                ).generate(description_text)
                st.image(
                    word_cloud.to_image(),
                    caption="Most common terms in app descriptions",
                    use_container_width=True,
                )
            except ValueError:
                st.info("There is not enough description text to build a word cloud.")
