"""Run app-level and review-level sentiment analysis."""

import pandas as pd
import plotly.express as px
import streamlit as st

import utils


st.set_page_config(page_title="Sentiment Analysis", layout="wide")

if (
    "results_df" not in st.session_state
    or st.session_state["results_df"].empty
):
    st.warning("Go to the Search page first and run a search.")
    st.stop()

df = st.session_state["results_df"]

st.title("Sentiment Analysis")
st.markdown(
    "Reviews are fetched from the Google Play Store and analysed with the "
    "**Cardiff RoBERTa sentiment model** from Hugging Face. The overview "
    "compares sentiment shares across apps, while the detail view keeps the "
    "individual review evidence visible."
)

app_options = (
    df[["appId", "title"]]
    .dropna(subset=["appId"])
    .drop_duplicates(subset="appId")
)
app_title_map = app_options.set_index("appId")["title"].fillna("").to_dict()
app_ids = app_options["appId"].tolist()

selected_app_ids = st.sidebar.multiselect(
    "Select apps to analyse",
    options=app_ids,
    default=app_ids[:3],
    format_func=lambda app_id: app_title_map.get(app_id) or app_id,
)
reviews_count = st.sidebar.slider(
    "Reviews per app",
    min_value=20,
    max_value=200,
    value=50,
    step=10,
)
run_analysis = st.sidebar.button(
    "Run Sentiment Analysis",
    type="primary",
    use_container_width=True,
)

if run_analysis:
    sentiment_results = []
    for app_id in selected_app_ids:
        with st.spinner(f"Analysing {app_id}..."):
            result = utils.get_app_sentiment_score(
                app_id,
                reviews_count=reviews_count,
            )
        if result is not None:
            sentiment_results.append(result)

    st.session_state["sentiment_results"] = sentiment_results

if "sentiment_results" in st.session_state:
    sentiment_results = st.session_state["sentiment_results"]

    if not sentiment_results:
        st.warning("No review sentiment data was available for the selected apps.")
    else:
        st.subheader("Sentiment Overview")

        overview_df = pd.DataFrame(sentiment_results)
        overview_df["title"] = overview_df["app_id"].map(app_title_map).fillna(
            overview_df["app_id"]
        )
        overview_long = overview_df.melt(
            id_vars=["app_id", "title", "total_reviews"],
            value_vars=["positive", "neutral", "negative"],
            var_name="sentiment",
            value_name="percentage",
        )

        overview_chart = px.bar(
            overview_long,
            x="title",
            y="percentage",
            color="sentiment",
            barmode="group",
            title="Review Sentiment by App",
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
            category_orders={
                "sentiment": ["positive", "neutral", "negative"]
            },
        )
        overview_chart.update_layout(
            xaxis_title="App title",
            yaxis_title="Share of analysed reviews (%)",
            yaxis_range=[0, 100],
            legend_title="Sentiment",
        )
        st.plotly_chart(overview_chart, use_container_width=True)

        st.subheader("Review-level detail")
        analysed_app_ids = overview_df["app_id"].drop_duplicates().tolist()
        detail_app_id = st.selectbox(
            "Choose an analysed app",
            options=analysed_app_ids,
            format_func=lambda app_id: app_title_map.get(app_id) or app_id,
        )

        with st.spinner(f"Loading review detail for {detail_app_id}..."):
            review_texts = utils.get_app_reviews(
                detail_app_id,
                count=reviews_count,
            )
            review_sentiments = utils.compute_sentiments(review_texts)

        if review_sentiments.empty:
            st.info("No review-level sentiment data is available for this app.")
        else:
            badge_map = {
                "positive": "🟢 positive",
                "neutral": "🟡 neutral",
                "negative": "🔴 negative",
            }
            review_detail = review_sentiments.copy()
            review_detail["label"] = review_detail["label"].map(
                lambda label: badge_map.get(label, label)
            )
            st.dataframe(
                review_detail,
                column_config={
                    "review": st.column_config.TextColumn("Review"),
                    "label": st.column_config.TextColumn("Sentiment"),
                    "score": st.column_config.NumberColumn(
                        "Confidence",
                        format="%.3f",
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )
