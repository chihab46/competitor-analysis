"""Search and inspect Google Play competitor results."""

import re

import pandas as pd
import streamlit as st

import utils


st.set_page_config(page_title="Search Results", layout="wide")

st.title("Search Results")
st.caption("Search Google Play and keep the result set available across pages.")

saved_query = st.session_state.get("search_query", "meditation app")
query = st.sidebar.text_input("Search query", value=saved_query)
n_results = st.sidebar.slider(
    "Number of results",
    min_value=10,
    max_value=100,
    value=30,
    step=10,
)
search_clicked = st.sidebar.button("Search", type="primary", use_container_width=True)

# A stored query without stored results can occur when another page initializes
# the shared query first. Restore that state with one fetch, but otherwise reuse
# results_df so normal Streamlit reruns do not repeat the network request.
should_search = search_clicked or (
    "search_query" in st.session_state
    and "results_df" not in st.session_state
)

if should_search:
    normalized_query = query.strip()
    if not normalized_query:
        st.warning("Enter a search query.")
        st.stop()

    with st.spinner("Fetching apps from Google Play..."):
        results_df = utils.search_apps(normalized_query, n_results=n_results)

    st.session_state["results_df"] = results_df
    st.session_state["search_query"] = normalized_query

if "results_df" in st.session_state:
    results_df = st.session_state["results_df"]

    if not isinstance(results_df, pd.DataFrame) or results_df.empty:
        st.warning("No results found.")
        st.stop()

    average_rating = float(results_df["score"].mean())
    free_percentage = float(
        results_df["free"].fillna(False).astype(bool).mean() * 100.0
    )

    total_column, rating_column, free_column = st.columns(3)
    total_column.metric("Apps found", f"{len(results_df):,}")
    rating_column.metric("Average rating", f"{average_rating:.2f}")
    free_column.metric("Free apps", f"{free_percentage:.1f}%")

    st.dataframe(
        results_df,
        column_config={
            "icon": st.column_config.ImageColumn("Icon"),
            "score": st.column_config.NumberColumn(
                "Rating",
                format="⭐ %.2f",
            ),
            "free": st.column_config.CheckboxColumn("Free"),
        },
        hide_index=True,
        use_container_width=True,
    )

    query_slug = re.sub(
        r"[^a-z0-9]+",
        "_",
        st.session_state.get("search_query", "apps").lower(),
    ).strip("_")
    st.download_button(
        "Download CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{query_slug or 'apps'}_results.csv",
        mime="text/csv",
    )
