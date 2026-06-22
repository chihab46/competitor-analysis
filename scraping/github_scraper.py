"""Scrape static GitHub repository search results."""

from __future__ import annotations

import random
import re
import time
from pathlib import Path
from urllib.parse import quote_plus, urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


GITHUB_URL = "https://github.com"
OUTPUT_COLUMNS = [
    "name",
    "full_url",
    "description",
    "stars",
    "language",
    "last_updated_raw",
]
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def scrape_github(query: str, max_pages: int = 3) -> pd.DataFrame:
    """Scrape repository results from GitHub's static search HTML."""
    repositories: list[dict[str, object]] = []
    headers = {"User-Agent": USER_AGENT}

    for page in range(1, max_pages + 1):
        search_url = (
            f"https://github.com/search?q={quote_plus(query)}"
            f"&type=repositories&p={page}"
        )

        try:
            response = requests.get(search_url, headers=headers, timeout=30)
        except requests.RequestException as exc:
            print(f"Warning: GitHub request failed on page {page}: {exc}")
            break

        try:
            assert response.status_code == 200
        except AssertionError:
            print(
                "Warning: GitHub returned status "
                f"{response.status_code} on page {page}; stopping."
            )
            break

        soup = BeautifulSoup(response.text, "html.parser")
        cards = soup.select(
            '[data-testid="results-list"] article, '
            'article[data-testid="results-list-item"], '
            'article[data-testid*="repository"]'
        )

        # Some GitHub variants put the test id on each card's link rather than
        # on the article. Keep only the parent article for each matching link.
        if not cards:
            cards = [
                article
                for link in soup.select('a[data-testid="result-list-item-link"]')
                if (article := link.find_parent("article")) is not None
            ]

        if not cards:
            print(
                "No repository cards found. GitHub may be blocking bots or "
                "the page may require JS rendering."
            )
            break

        for card in cards:
            name_link = card.select_one(
                'a[data-testid="result-list-item-link"], h3 a, h2 a'
            )
            href = name_link.get("href", "") if name_link else ""
            name = name_link.get_text(" ", strip=True) if name_link else ""

            description_element = card.select_one(
                '[data-testid="result-list-item-description"], p'
            )
            description = (
                description_element.get_text(" ", strip=True)
                if description_element
                else ""
            )

            stars_link = card.select_one('a[href$="/stargazers"]')
            stars_text = stars_link.get_text(strip=True) if stars_link else "0"
            stars_digits = re.sub(r"[^0-9]", "", stars_text.replace(",", ""))

            language_element = card.select_one(
                '[data-testid="repository-language"], '
                '[itemprop="programmingLanguage"]'
            )
            language = (
                language_element.get_text(" ", strip=True)
                if language_element
                else ""
            )

            relative_time = card.select_one("relative-time")
            last_updated_raw = (
                relative_time.get("datetime", "") if relative_time else ""
            )

            repositories.append(
                {
                    "name": name,
                    "full_url": urljoin(GITHUB_URL, href) if href else "",
                    "description": description,
                    "stars": int(stars_digits or 0),
                    "language": language,
                    "last_updated_raw": last_updated_raw,
                }
            )

        if page < max_pages:
            time.sleep(random.uniform(1.0, 2.5))

    dataframe = pd.DataFrame(repositories, columns=OUTPUT_COLUMNS)
    dataframe = dataframe.drop_duplicates(subset="full_url").reset_index(drop=True)

    query_slug = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_") or "search"
    data_directory = Path(__file__).resolve().parents[1] / "data"
    data_directory.mkdir(parents=True, exist_ok=True)
    output_path = data_directory / f"github_results_{query_slug}.csv"
    dataframe.to_csv(output_path, index=False)

    return dataframe


if __name__ == "__main__":
    df = scrape_github("mental health ai")
    print(df.shape)
    print(df.head(3))
