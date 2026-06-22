"""Shared data-loading and sentiment utilities for the Streamlit app."""

from __future__ import annotations

from functools import lru_cache
import re

import google_play_scraper
import pandas as pd

from scraping.google_play_scraper import expanded_search


APP_COLUMNS = [
    "appId",
    "title",
    "developer",
    "score",
    "ratings",
    "reviews",
    "price",
    "free",
    "genre",
    "icon",
    "description",
    "installs",
]
SENTIMENT_COLUMNS = ["review", "label", "score"]
LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "NEGATIVE": "negative",
    "NEUTRAL": "neutral",
    "POSITIVE": "positive",
}
REVIEW_TOPIC_KEYWORDS = {
    "Pricing & subscriptions": (
        "price",
        "pricing",
        "expensive",
        "cost",
        "subscription",
        "premium",
        "paywall",
        "trial",
        "refund",
        "billing",
        "charged",
    ),
    "Reliability": (
        "crash",
        "crashes",
        "crashing",
        "bug",
        "bugs",
        "broken",
        "error",
        "freeze",
        "freezes",
        "login",
        "sync",
    ),
    "Usability": (
        "interface",
        "navigation",
        "easy",
        "difficult",
        "confusing",
        "design",
        "user friendly",
        "usability",
    ),
    "Content quality": (
        "content",
        "meditation",
        "session",
        "lesson",
        "exercise",
        "course",
        "music",
        "guided",
        "guidance",
        "library",
    ),
    "Ads": ("ad", "ads", "advert", "advertisement"),
    "Customer support": (
        "support",
        "customer service",
        "help desk",
        "response",
    ),
    "Performance": ("slow", "fast", "lag", "laggy", "battery", "speed"),
    "Privacy & data": ("privacy", "data", "permission", "tracking"),
}


def search_apps(
    query: str,
    n_results: int = 30,
    country: str = "us",
    lang: str = "en",
) -> pd.DataFrame:
    """Search Google Play and return normalized app metadata."""
    try:
        raw_apps = expanded_search(
            query,
            n_results=n_results,
            country=country,
            lang=lang,
        )
        apps = pd.DataFrame(raw_apps).reindex(columns=APP_COLUMNS)

        apps["score"] = pd.to_numeric(apps["score"], errors="coerce").fillna(0.0)
        apps["score"] = apps["score"].astype(float)

        for column in ("ratings", "reviews"):
            numeric_values = apps[column].map(
                lambda value: value.replace(",", "")
                if isinstance(value, str)
                else value
            )
            apps[column] = (
                pd.to_numeric(numeric_values, errors="coerce").fillna(0).astype(int)
            )

        return apps
    except Exception as exc:
        print(f"Error searching Google Play apps: {exc}")
        return pd.DataFrame(columns=APP_COLUMNS)


def get_app_reviews(
    app_id: str,
    count: int = 100,
    country: str = "us",
    lang: str = "en",
) -> list[str]:
    """Fetch review text for one Google Play app."""
    try:
        review_items, _ = google_play_scraper.reviews(
            app_id,
            count=count,
            country=country,
            lang=lang,
        )
        return [
            content
            for review in review_items
            if isinstance((content := review.get("content")), str)
        ]
    except Exception as exc:
        print(f"Error fetching reviews for {app_id}: {exc}")
        return []


@lru_cache(maxsize=2)
def _get_sentiment_pipeline(model_name: str):
    """Load each sentiment model once per Python process."""
    from transformers import pipeline

    return pipeline(
        "sentiment-analysis",
        model=model_name,
        truncation=True,
        max_length=512,
    )


def compute_sentiments(
    reviews: list[str],
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
) -> pd.DataFrame:
    """Classify review sentiment in batches of 16."""
    if not reviews:
        return pd.DataFrame(columns=SENTIMENT_COLUMNS)

    sentiment_pipeline = _get_sentiment_pipeline(model_name)
    truncated_reviews = [review[:512] for review in reviews]
    predictions: list[dict[str, object]] = []

    for start in range(0, len(truncated_reviews), 16):
        batch = truncated_reviews[start : start + 16]
        predictions.extend(sentiment_pipeline(batch, batch_size=16))

    rows = []
    for review, prediction in zip(reviews, predictions):
        raw_label = str(prediction.get("label", "")).upper()
        rows.append(
            {
                "review": review,
                "label": LABEL_MAP.get(raw_label, raw_label.lower()),
                "score": float(prediction.get("score", 0.0)),
            }
        )

    return pd.DataFrame(rows, columns=SENTIMENT_COLUMNS)


def classify_review_topics(review: str) -> list[str]:
    """Assign transparent product themes to one review."""
    normalized_review = str(review).lower()
    matched_topics = []

    for topic, keywords in REVIEW_TOPIC_KEYWORDS.items():
        if any(
            re.search(rf"\b{re.escape(keyword)}\b", normalized_review)
            for keyword in keywords
        ):
            matched_topics.append(topic)

    return matched_topics or ["Other"]


def summarize_review_topics(sentiments: pd.DataFrame) -> pd.DataFrame:
    """Summarize topic mentions and sentiment percentages."""
    columns = [
        "topic",
        "mentions",
        "positive",
        "neutral",
        "negative",
        "net_sentiment",
    ]
    if sentiments.empty:
        return pd.DataFrame(columns=columns)

    topic_rows = []
    for row in sentiments.itertuples(index=False):
        for topic in classify_review_topics(row.review):
            topic_rows.append({"topic": topic, "label": row.label})

    topic_data = pd.DataFrame(topic_rows)
    counts = topic_data.groupby(["topic", "label"]).size().unstack(fill_value=0)
    for label in ("positive", "neutral", "negative"):
        if label not in counts:
            counts[label] = 0

    counts["mentions"] = counts[["positive", "neutral", "negative"]].sum(axis=1)
    for label in ("positive", "neutral", "negative"):
        counts[label] = counts[label] / counts["mentions"] * 100.0
    counts["net_sentiment"] = counts["positive"] - counts["negative"]

    return (
        counts.reset_index()[columns]
        .sort_values(["mentions", "topic"], ascending=[False, True])
        .reset_index(drop=True)
    )


def parse_installs(value: object) -> int:
    """Convert Google Play install labels such as 1M+ and 500K+ to ints."""
    if pd.isna(value):
        return 0

    normalized = str(value).strip().upper().replace(",", "").replace("+", "")
    match = re.search(r"(\d+(?:\.\d+)?)\s*([KMB]?)", normalized)
    if not match:
        return 0

    number = float(match.group(1))
    multiplier = {
        "": 1,
        "K": 1_000,
        "M": 1_000_000,
        "B": 1_000_000_000,
    }[match.group(2)]
    return int(number * multiplier)


def get_app_sentiment_score(
    app_id: str,
    reviews_count: int = 100,
) -> dict | None:
    """Return positive, neutral, and negative review percentages for an app."""
    review_texts = get_app_reviews(app_id, count=reviews_count)
    if not review_texts:
        return None

    sentiments = compute_sentiments(review_texts)
    if sentiments.empty:
        return None

    total_reviews = len(sentiments)
    counts = sentiments["label"].value_counts()
    positive = float(counts.get("positive", 0) / total_reviews * 100.0)
    neutral = float(counts.get("neutral", 0) / total_reviews * 100.0)
    negative = float(100.0 - positive - neutral)

    return {
        "app_id": app_id,
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "total_reviews": int(total_reviews),
    }


if __name__ == "__main__":
    df = search_apps("meditation app")
    print(df.head(3))
