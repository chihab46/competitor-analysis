"""Compare selected Google Play competitors side by side."""

import pandas as pd
import plotly.express as px
import streamlit as st

import utils


st.set_page_config(page_title="App Comparison", layout="wide")

if (
    "results_df" not in st.session_state
    or st.session_state["results_df"].empty
):
    st.warning("Go to the Search page first and run a search.")
    st.stop()

df = st.session_state["results_df"].drop_duplicates(subset="appId").copy()
app_title_map = df.set_index("appId")["title"].fillna("").to_dict()
app_ids = df["appId"].dropna().tolist()

st.title("App Comparison")
st.caption(
    "Put a focused shortlist side by side, then connect market signals with "
    "any sentiment analysis already completed."
)

selected_app_ids = st.sidebar.multiselect(
    "Select 2–5 apps",
    options=app_ids,
    default=app_ids[: min(3, len(app_ids))],
    max_selections=5,
    format_func=lambda app_id: app_title_map.get(app_id) or app_id,
)
st.sidebar.metric("Apps compared", len(selected_app_ids))

if len(selected_app_ids) < 2:
    st.info("Select at least two apps to build a comparison.")
    st.stop()

comparison_df = (
    df.set_index("appId")
    .loc[selected_app_ids]
    .reset_index()
    .copy()
)
comparison_df["installs_numeric"] = comparison_df["installs"].map(
    utils.parse_installs
)

highest_rated = comparison_df.loc[comparison_df["score"].idxmax()]
widest_reach = comparison_df.loc[comparison_df["installs_numeric"].idxmax()]
free_count = int(comparison_df["free"].fillna(False).astype(bool).sum())
widest_reach_installs = widest_reach["installs"]
if pd.isna(widest_reach_installs) or not str(widest_reach_installs).strip():
    widest_reach_installs = "Unknown"

rating_column, reach_column, pricing_column = st.columns(3)
rating_column.metric(
    "Highest rated",
    highest_rated["title"],
    f"{highest_rated['score']:.2f} stars",
)
reach_column.metric(
    "Largest install base",
    widest_reach["title"],
    str(widest_reach_installs),
)
pricing_column.metric(
    "Pricing mix",
    f"{free_count} free / {len(comparison_df) - free_count} paid",
)

st.subheader("Comparison scorecard")
scorecard_columns = [
    "icon",
    "title",
    "developer",
    "score",
    "installs",
    "free",
    "genre",
    "price",
]
st.dataframe(
    comparison_df[scorecard_columns],
    column_config={
        "icon": st.column_config.ImageColumn("Icon"),
        "title": st.column_config.TextColumn("App"),
        "score": st.column_config.NumberColumn("Rating", format="⭐ %.2f"),
        "free": st.column_config.CheckboxColumn("Free"),
        "price": st.column_config.NumberColumn("Price", format="$%.2f"),
    },
    hide_index=True,
    use_container_width=True,
)

ratings_tab, reach_tab, sentiment_tab = st.tabs(
    ["Ratings", "Market reach", "Sentiment"]
)

with ratings_tab:
    rating_chart = px.bar(
        comparison_df.sort_values("score"),
        x="score",
        y="title",
        orientation="h",
        range_x=[0, 5],
        color="score",
        color_continuous_scale="Blues",
        title="Rating Comparison",
        labels={"score": "Rating", "title": "App"},
    )
    rating_chart.update_layout(
        xaxis_title="Google Play rating (0–5)",
        yaxis_title="App title",
        coloraxis_showscale=False,
    )
    st.plotly_chart(rating_chart, use_container_width=True)

with reach_tab:
    reach_data = comparison_df[comparison_df["installs_numeric"] > 0].copy()
    if reach_data.empty:
        st.info("Install estimates are unavailable for these apps.")
    else:
        reach_chart = px.bar(
            reach_data.sort_values("installs_numeric"),
            x="installs_numeric",
            y="title",
            orientation="h",
            color="free",
            log_x=True,
            title="Estimated Install Reach",
            labels={
                "installs_numeric": "Estimated installs (log scale)",
                "title": "App",
                "free": "Free app",
            },
            color_discrete_map={True: "#1E6FD9", False: "#F59E0B"},
        )
        reach_chart.update_layout(
            xaxis_title="Estimated installs (log scale)",
            yaxis_title="App title",
        )
        st.plotly_chart(reach_chart, use_container_width=True)

with sentiment_tab:
    sentiment_results = st.session_state.get("sentiment_results", [])
    sentiment_df = pd.DataFrame(sentiment_results)

    if sentiment_df.empty or "app_id" not in sentiment_df:
        st.info(
            "Run sentiment analysis for these apps to add review perception "
            "to the comparison."
        )
    else:
        sentiment_df = sentiment_df[
            sentiment_df["app_id"].isin(selected_app_ids)
        ].copy()
        if sentiment_df.empty:
            st.info(
                "No sentiment results match the selected apps. Analyse them "
                "on the Sentiment Analysis page first."
            )
        else:
            sentiment_df["title"] = sentiment_df["app_id"].map(app_title_map)
            sentiment_long = sentiment_df.melt(
                id_vars=["app_id", "title", "total_reviews"],
                value_vars=["positive", "neutral", "negative"],
                var_name="sentiment",
                value_name="percentage",
            )
            sentiment_chart = px.bar(
                sentiment_long,
                x="title",
                y="percentage",
                color="sentiment",
                barmode="group",
                title="Sentiment Comparison",
                labels={
                    "title": "App",
                    "percentage": "Reviews (%)",
                    "sentiment": "Sentiment",
                },
                color_discrete_map={
                    "positive": "#22C55E",
                    "neutral": "#9CA3AF",
                    "negative": "#EF4444",
                },
            )
            sentiment_chart.update_layout(
                xaxis_title="App title",
                yaxis_title="Share of analysed reviews (%)",
                yaxis_range=[0, 100],
            )
            st.plotly_chart(sentiment_chart, use_container_width=True)
