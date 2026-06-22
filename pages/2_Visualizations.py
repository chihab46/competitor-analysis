"""Interactive visual analysis of the current Google Play result set."""

from collections import Counter
import math
import re

import pandas as pd
import plotly.express as px
import streamlit as st

import utils


FREE_COLOR = "#1E6FD9"
PAID_COLOR = "#F59E0B"
UNKNOWN_COLOR = "#94A3B8"
PRICING_COLORS = {
    "Free": FREE_COLOR,
    "Paid": PAID_COLOR,
    "Unknown": UNKNOWN_COLOR,
}
INSTALL_STEPS = [
    0,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
    10_000_000_000,
]
CHART_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "competitor-analysis-chart",
        "scale": 2,
    },
}
GENERIC_DESCRIPTION_WORDS = {
    "app",
    "apps",
    "application",
    "feature",
    "features",
    "get",
    "make",
    "new",
    "one",
    "use",
    "used",
    "user",
    "users",
    "using",
    "will",
}


def _format_compact_number(value: int | float) -> str:
    value = float(value)
    for threshold, suffix in (
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    ):
        if abs(value) >= threshold:
            compact = value / threshold
            precision = 0 if compact >= 10 or compact.is_integer() else 1
            return f"{compact:.{precision}f}{suffix}"
    return f"{value:,.0f}"


def _pricing_label(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    return "Free" if bool(value) else "Paid"


def _selected_app_from_event(event: object) -> str | None:
    try:
        points = event.selection.points
    except (AttributeError, KeyError, TypeError):
        return None
    if not points:
        return None
    custom_data = points[0].get("customdata", [])
    return custom_data[0] if custom_data else None


def _top_genre_groups(genres: pd.Series, limit: int = 9) -> pd.Series:
    cleaned = genres.fillna("Unknown").replace("", "Unknown")
    leading_genres = cleaned.value_counts().head(limit).index
    return cleaned.where(cleaned.isin(leading_genres), "Other")


def _description_terms(text: str, stopwords: set[str]) -> tuple[list[str], list[str]]:
    raw_tokens = re.findall(r"[a-z][a-z'-]{2,}", text.lower())
    keywords = [token for token in raw_tokens if token not in stopwords]
    phrases = [
        f"{first} {second}"
        for first, second in zip(raw_tokens, raw_tokens[1:])
        if first not in stopwords and second not in stopwords
    ]
    return keywords, phrases


st.set_page_config(page_title="Visualizations", layout="wide")

if (
    "results_df" not in st.session_state
    or st.session_state["results_df"].empty
):
    st.warning("Go to the Search page first and run a search.")
    st.stop()

df = st.session_state["results_df"].copy()
df["score"] = pd.to_numeric(df["score"], errors="coerce")
df["installs_numeric"] = df["installs"].map(utils.parse_installs)
df["pricing_model"] = df["free"].map(_pricing_label)
df["genre_group"] = _top_genre_groups(df["genre"])

st.title("Visualizations")
st.caption(
    "Define a competitor cohort, scan its market shape, and inspect the apps "
    "behind the signals. Missing ratings and pricing are kept separate from "
    "real zero or paid values."
)

app_options = (
    df[["appId", "title"]]
    .dropna(subset=["appId"])
    .drop_duplicates(subset="appId")
)
app_title_map = app_options.set_index("appId")["title"].fillna("").to_dict()
app_ids = app_options["appId"].tolist()
genre_options = sorted(
    df["genre"].dropna().loc[lambda values: values.ne("")].unique().tolist()
)

st.sidebar.subheader("Cohort filters")
selected_app_ids = st.sidebar.multiselect(
    "Apps",
    options=app_ids,
    format_func=lambda app_id: app_title_map.get(app_id) or app_id,
    help="Titles are displayed here; filtering still uses the stable App ID.",
)
selected_genres = st.sidebar.multiselect("Genres", options=genre_options)
pricing_filter = st.sidebar.selectbox(
    "Pricing",
    options=["All", "Free only", "Paid only", "Unknown only"],
)
minimum_rating = st.sidebar.slider(
    "Minimum rating",
    min_value=0.0,
    max_value=5.0,
    value=0.0,
    step=0.5,
    help="Ratings of 0 are treated as missing or unrated.",
)
install_range = st.sidebar.select_slider(
    "Install range",
    options=INSTALL_STEPS,
    value=(INSTALL_STEPS[0], INSTALL_STEPS[-1]),
    format_func=_format_compact_number,
)

filtered_df = df.copy()
if selected_app_ids:
    filtered_df = filtered_df[filtered_df["appId"].isin(selected_app_ids)]
if selected_genres:
    filtered_df = filtered_df[filtered_df["genre"].isin(selected_genres)]
if pricing_filter != "All":
    pricing_value = {
        "Free only": "Free",
        "Paid only": "Paid",
        "Unknown only": "Unknown",
    }[pricing_filter]
    filtered_df = filtered_df[filtered_df["pricing_model"].eq(pricing_value)]
if minimum_rating > 0:
    filtered_df = filtered_df[filtered_df["score"].ge(minimum_rating)]
filtered_df = filtered_df[
    filtered_df["installs_numeric"].between(*install_range, inclusive="both")
].copy()

st.sidebar.metric("Apps in view", len(filtered_df))
st.sidebar.caption(
    "Charts can be exported with the camera icon in each Plotly toolbar."
)

if filtered_df.empty:
    st.warning("No apps match the selected filters.")
    st.stop()

rated_df = filtered_df[filtered_df["score"].gt(0)].copy()
known_pricing = filtered_df[filtered_df["pricing_model"].ne("Unknown")]
valid_installs = filtered_df[filtered_df["installs_numeric"].gt(0)].copy()

median_rating = rated_df["score"].median() if not rated_df.empty else None
known_genres = filtered_df["genre"].dropna().replace("", pd.NA).dropna()
most_common_genre = (
    known_genres.mode().iloc[0] if not known_genres.empty else "Unknown"
)
median_installs = (
    valid_installs["installs_numeric"].median() if not valid_installs.empty else 0
)
free_percentage = (
    known_pricing["pricing_model"].eq("Free").mean() * 100
    if not known_pricing.empty
    else None
)

apps_metric, rating_metric, genre_metric, installs_metric, pricing_metric = st.columns(5)
apps_metric.metric("Apps in view", f"{len(filtered_df):,}")
rating_metric.metric(
    "Median rating",
    f"{median_rating:.2f}" if median_rating is not None else "N/A",
)
genre_metric.metric("Top genre", most_common_genre)
installs_metric.metric(
    "Median installs",
    _format_compact_number(median_installs) if median_installs else "N/A",
)
pricing_metric.metric(
    "Free apps",
    f"{free_percentage:.1f}%" if free_percentage is not None else "N/A",
)

filter_context = f"{len(filtered_df)} filtered apps"
chart_selected_app_id: str | None = None

ratings_tab, categories_tab, installs_tab, wordcloud_tab = st.tabs(
    ["Ratings", "Categories", "Size & Installs", "Word Cloud"]
)

with ratings_tab:
    if rated_df.empty:
        st.info("No rated apps are available for the current filters.")
    else:
        histogram_column, leaders_column = st.columns(2, gap="large")

        with histogram_column:
            ratings_histogram = px.histogram(
                rated_df,
                x="score",
                range_x=[0, 5],
                title=f"Rating Distribution · {filter_context}",
                labels={"score": "Rating", "count": "Apps"},
                color_discrete_sequence=[FREE_COLOR],
            )
            ratings_histogram.update_traces(
                xbins={"start": 0, "end": 5, "size": 0.5}
            )
            ratings_histogram.add_vline(
                x=median_rating,
                line_dash="dash",
                line_color="#334155",
                annotation_text=f"Median {median_rating:.2f}",
            )
            ratings_histogram.update_layout(
                xaxis_title="Google Play rating (0–5)",
                yaxis_title="Number of rated apps",
                bargap=0.08,
            )
            st.plotly_chart(
                ratings_histogram,
                width="stretch",
                config=CHART_CONFIG,
            )

        with leaders_column:
            top_rated = rated_df.nlargest(10, "score").sort_values("score")
            top_rated_chart = px.bar(
                top_rated,
                x="score",
                y="title",
                orientation="h",
                range_x=[0, 5],
                title="Top Rated Apps",
                labels={"score": "Rating", "title": "App"},
                color="score",
                color_continuous_scale="Blues",
                hover_data={"appId": True, "developer": True},
            )
            top_rated_chart.update_layout(
                xaxis_title="Google Play rating (0–5)",
                yaxis_title="App title",
                coloraxis_showscale=False,
            )
            st.plotly_chart(
                top_rated_chart,
                width="stretch",
                config=CHART_CONFIG,
            )

with categories_tab:
    pricing_column, genre_column = st.columns(2, gap="large")

    with pricing_column:
        pricing_counts = (
            filtered_df["pricing_model"]
            .value_counts()
            .rename_axis("pricing")
            .reset_index(name="apps")
        )
        pricing_chart = px.pie(
            pricing_counts,
            names="pricing",
            values="apps",
            title=f"Pricing Availability · {filter_context}",
            labels={"pricing": "Pricing", "apps": "Apps"},
            color="pricing",
            color_discrete_map=PRICING_COLORS,
        )
        pricing_chart.update_traces(
            textposition="inside",
            textinfo="percent+label",
        )
        st.plotly_chart(pricing_chart, width="stretch", config=CHART_CONFIG)

    with genre_column:
        grouped_genres = _top_genre_groups(filtered_df["genre"])
        genre_counts = (
            grouped_genres.value_counts()
            .sort_values()
            .rename_axis("genre")
            .reset_index(name="apps")
        )
        genre_chart = px.bar(
            genre_counts,
            x="apps",
            y="genre",
            orientation="h",
            title="Genre Mix · smaller categories grouped as Other",
            labels={"apps": "Apps", "genre": "Genre"},
            color="apps",
            color_continuous_scale="Blues",
        )
        genre_chart.update_layout(
            xaxis_title="Number of apps",
            yaxis_title="Genre",
            coloraxis_showscale=False,
        )
        st.plotly_chart(genre_chart, width="stretch", config=CHART_CONFIG)

    if rated_df.empty:
        st.info("Genre quality requires at least one rated app.")
    else:
        genre_quality = rated_df.copy()
        genre_quality["genre_display"] = _top_genre_groups(genre_quality["genre"])
        genre_quality_chart = px.box(
            genre_quality,
            x="genre_display",
            y="score",
            color="genre_display",
            points="all",
            hover_name="title",
            title="Rating Distribution by Genre",
            labels={"genre_display": "Genre", "score": "Rating"},
        )
        genre_quality_chart.update_layout(
            xaxis_title="Genre",
            yaxis_title="Google Play rating (0–5)",
            yaxis_range=[0, 5],
            showlegend=False,
        )
        st.plotly_chart(
            genre_quality_chart,
            width="stretch",
            config=CHART_CONFIG,
        )

with installs_tab:
    if valid_installs.empty:
        st.info("Install estimates are unavailable for the current filters.")
    else:
        leaders_column, opportunity_column = st.columns(2, gap="large")

        with leaders_column:
            top_installed = valid_installs.nlargest(
                10, "installs_numeric"
            ).sort_values("installs_numeric")
            installs_chart = px.bar(
                top_installed,
                x="installs_numeric",
                y="title",
                orientation="h",
                title="Install Leaders",
                labels={"installs_numeric": "Installs", "title": "App"},
                color="pricing_model",
                color_discrete_map=PRICING_COLORS,
                hover_data={"installs": True, "appId": True},
            )
            installs_chart.add_vline(
                x=median_installs,
                line_dash="dash",
                line_color="#334155",
                annotation_text=f"Median {_format_compact_number(median_installs)}",
            )
            installs_chart.update_layout(
                xaxis_title="Estimated installs",
                yaxis_title="App title",
                xaxis_tickformat=".2s",
                legend_title="Pricing",
            )
            st.plotly_chart(
                installs_chart,
                width="stretch",
                config=CHART_CONFIG,
            )

        with opportunity_column:
            opportunity_data = valid_installs[
                valid_installs["score"].gt(0)
            ].copy()
            if opportunity_data.empty:
                st.info("The opportunity matrix also requires app ratings.")
            else:
                opportunity_rating_median = opportunity_data["score"].median()
                opportunity_install_median = opportunity_data[
                    "installs_numeric"
                ].median()
                opportunity_chart = px.scatter(
                    opportunity_data,
                    x="installs_numeric",
                    y="score",
                    color="pricing_model",
                    hover_name="title",
                    custom_data=["appId"],
                    log_x=True,
                    title="Opportunity Matrix · select a point to inspect",
                    labels={
                        "installs_numeric": "Estimated installs",
                        "score": "Rating",
                        "pricing_model": "Pricing",
                    },
                    color_discrete_map=PRICING_COLORS,
                )
                opportunity_chart.add_vline(
                    x=opportunity_install_median,
                    line_dash="dot",
                    line_color="#64748B",
                )
                opportunity_chart.add_hline(
                    y=opportunity_rating_median,
                    line_dash="dot",
                    line_color="#64748B",
                )
                opportunity_chart.add_annotation(
                    x=0.98,
                    y=0.05,
                    xref="paper",
                    yref="paper",
                    text="Disruption opportunity",
                    showarrow=False,
                    font={"color": "#B91C1C"},
                    xanchor="right",
                )
                opportunity_chart.add_annotation(
                    x=0.02,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    text="Hidden gems",
                    showarrow=False,
                    font={"color": "#15803D"},
                    xanchor="left",
                )
                opportunity_chart.update_layout(
                    xaxis_title="Estimated installs (log scale)",
                    yaxis_title="Google Play rating (0–5)",
                    yaxis_range=[0, 5],
                    xaxis_tickformat=".2s",
                    legend_title="Pricing",
                    dragmode="select",
                )
                opportunity_event = st.plotly_chart(
                    opportunity_chart,
                    key="opportunity_matrix",
                    on_select="rerun",
                    selection_mode="points",
                    width="stretch",
                    config=CHART_CONFIG,
                )
                chart_selected_app_id = _selected_app_from_event(
                    opportunity_event
                )

with wordcloud_tab:
    descriptions = filtered_df["description"].dropna().astype(str).tolist()
    description_text = " ".join(descriptions).strip()
    if not description_text:
        st.info("No app descriptions are available for the current filters.")
    else:
        try:
            from wordcloud import STOPWORDS, WordCloud
        except ImportError:
            stopwords = GENERIC_DESCRIPTION_WORDS
            st.info(
                "Word Cloud support is not installed. Install it with "
                "`uv pip install wordcloud`."
            )
            word_cloud = None
        else:
            stopwords = set(STOPWORDS) | GENERIC_DESCRIPTION_WORDS
            try:
                word_cloud = WordCloud(
                    width=800,
                    height=400,
                    background_color="white",
                    max_words=100,
                    stopwords=stopwords,
                    collocations=True,
                    collocation_threshold=5,
                    colormap="Blues",
                ).generate(description_text)
            except ValueError:
                word_cloud = None

        keywords = []
        phrases = []
        for description in descriptions:
            description_keywords, description_phrases = _description_terms(
                description,
                stopwords,
            )
            keywords.extend(description_keywords)
            phrases.extend(description_phrases)
        keyword_counts = Counter(keywords)
        phrase_counts = Counter(phrases)
        combined_terms = keyword_counts.most_common(15) + [
            (phrase, count)
            for phrase, count in phrase_counts.most_common(10)
            if count > 1
        ]
        top_terms = (
            pd.DataFrame(combined_terms, columns=["term", "mentions"])
            .sort_values("mentions", ascending=False)
            .drop_duplicates(subset="term")
            .head(20)
            .sort_values("mentions")
        )

        cloud_column, keywords_column = st.columns(2, gap="large")
        with cloud_column:
            if word_cloud is None:
                st.info("There is not enough text to build a word cloud.")
            else:
                st.image(
                    word_cloud.to_image(),
                    caption="Common words and phrases in app descriptions",
                    width="stretch",
                )

        with keywords_column:
            if top_terms.empty:
                st.info("There are not enough keywords to rank.")
            else:
                keyword_chart = px.bar(
                    top_terms,
                    x="mentions",
                    y="term",
                    orientation="h",
                    title="Top Description Keywords & Phrases",
                    labels={"mentions": "Mentions", "term": "Term"},
                    color="mentions",
                    color_continuous_scale="Blues",
                )
                keyword_chart.update_layout(
                    xaxis_title="Number of mentions",
                    yaxis_title="Keyword or phrase",
                    coloraxis_showscale=False,
                )
                st.plotly_chart(
                    keyword_chart,
                    width="stretch",
                    config=CHART_CONFIG,
                )

st.divider()
st.subheader("App inspector")
inspector_options = filtered_df["appId"].dropna().drop_duplicates().tolist()
inspector_key = "visualization_app_inspector"
if chart_selected_app_id in inspector_options:
    st.session_state[inspector_key] = chart_selected_app_id
elif st.session_state.get(inspector_key) not in inspector_options:
    st.session_state[inspector_key] = inspector_options[0]

inspected_app_id = st.selectbox(
    "Inspect an app",
    options=inspector_options,
    format_func=lambda app_id: app_title_map.get(app_id) or app_id,
    key=inspector_key,
    help="Selecting a point in the opportunity matrix updates this panel.",
)
inspected_app = filtered_df.loc[
    filtered_df["appId"].eq(inspected_app_id)
].iloc[0]
inspected_developer = inspected_app.get("developer")
if pd.isna(inspected_developer) or not str(inspected_developer).strip():
    inspected_developer = "Unknown developer"
inspected_genre = inspected_app.get("genre")
if pd.isna(inspected_genre) or not str(inspected_genre).strip():
    inspected_genre = "Unknown genre"
inspected_installs = inspected_app.get("installs")
if pd.isna(inspected_installs) or not str(inspected_installs).strip():
    inspected_installs = "Unknown"

icon_column, detail_column = st.columns([1, 6], gap="large")
with icon_column:
    if pd.notna(inspected_app.get("icon")) and inspected_app.get("icon"):
        st.image(inspected_app["icon"], width=112)

with detail_column:
    st.markdown(f"### {inspected_app['title']}")
    st.caption(
        f"{inspected_developer} · {inspected_genre} · {inspected_app_id}"
    )
    detail_rating, detail_installs, detail_pricing = st.columns(3)
    detail_rating.metric(
        "Rating",
        f"{inspected_app['score']:.2f}"
        if pd.notna(inspected_app["score"]) and inspected_app["score"] > 0
        else "Unrated",
    )
    detail_installs.metric(
        "Installs",
        str(inspected_installs),
    )
    detail_pricing.metric("Pricing", inspected_app["pricing_model"])
    description = inspected_app.get("description")
    if pd.notna(description) and description:
        st.write(description)

st.download_button(
    "Download filtered data",
    data=filtered_df.drop(
        columns=["installs_numeric", "pricing_model", "genre_group"],
        errors="ignore",
    ).to_csv(index=False),
    file_name="filtered_competitor_apps.csv",
    mime="text/csv",
)
