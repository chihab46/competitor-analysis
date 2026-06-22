"""Shared data-loading and sentiment utilities for the Streamlit app."""

from __future__ import annotations

import pandas as pd
import google_play_scraper


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


def search_apps(
    query: str,
    n_results: int = 30,
    country: str = "us",
    lang: str = "en",
) -> pd.DataFrame:
    """Search Google Play and return normalized app metadata."""
    try:
        raw_apps = google_play_scraper.search(
            query,
            n_hits=n_results,
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


def compute_sentiments(
    reviews: list[str],
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
) -> pd.DataFrame:
    """Classify review sentiment in batches of 16."""
    if not reviews:
        return pd.DataFrame(columns=SENTIMENT_COLUMNS)

    # Import lazily because importing transformers also initializes its ML stack.
    from transformers import pipeline

    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model_name,
        truncation=True,
        max_length=512,
    )
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
