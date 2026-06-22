"""Scrape Product Hunt search results with Selenium."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.parse import quote_plus, urljoin

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


PRODUCT_HUNT_URL = "https://www.producthunt.com"
PRODUCT_CARD_SELECTOR = (
    'article[data-test*="post"], '
    '[data-test^="post-item"], '
    '[data-testid^="post-item"]'
)
PRODUCT_LINK_SELECTOR = (
    'a[data-test="post-name"], '
    'a[data-testid="post-name"], '
    'a[href^="/posts/"]'
)
TAGLINE_SELECTOR = (
    '[data-test="post-tagline"], '
    '[data-testid="post-tagline"], '
    'div[class*="tagline"]'
)
VOTE_SELECTOR = (
    '[data-test="vote-button"], '
    '[data-testid="vote-button"], '
    'button[aria-label*="upvote" i]'
)
DESCRIPTION_SELECTOR = (
    '[data-test="product-description"], '
    '[data-testid="product-description"], '
    '[class*="description"]'
)
REVIEW_SELECTOR = (
    '[data-test="review-content"], '
    '[data-testid="review-content"], '
    '[data-test*="review"] [class*="content"]'
)
OUTPUT_COLUMNS = [
    "name",
    "tagline",
    "vote_count",
    "product_url",
    "full_description",
    "reviews",
]


def _parse_vote_count(text: str) -> int:
    normalized = text.strip().replace(",", "").upper()
    match = re.search(r"(\d+(?:\.\d+)?)\s*([KM]?)", normalized)
    if not match:
        raise ValueError(f"No vote count found in {text!r}")

    value = float(match.group(1))
    multiplier = {"": 1, "K": 1_000, "M": 1_000_000}[match.group(2)]
    return int(value * multiplier)


def _load_raw_data(raw_path: Path) -> list[dict[str, object]]:
    if not raw_path.exists():
        return []

    try:
        with raw_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Warning: could not load {raw_path}: {exc}. Starting fresh.")
        return []


def _save_raw_data(raw_path: Path, data: list[dict[str, object]]) -> None:
    with raw_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _enrich_product(driver: webdriver.Chrome, product: dict[str, object]) -> None:
    product_url = product.get("product_url")
    if not product_url:
        product["full_description"] = None
        product["reviews"] = []
        return

    search_window = driver.current_window_handle
    detail_window_opened = False

    try:
        driver.switch_to.new_window("tab")
        detail_window_opened = True
        driver.get(str(product_url))

        try:
            description_element = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, DESCRIPTION_SELECTOR))
            )
            product["full_description"] = description_element.text.strip() or None
        except (TimeoutException, StaleElementReferenceException):
            product["full_description"] = None

        reviews: list[str] = []
        try:
            review_elements = driver.find_elements(By.CSS_SELECTOR, REVIEW_SELECTOR)[:5]
        except WebDriverException:
            review_elements = []

        for review_element in review_elements:
            try:
                review_text = review_element.text.strip()
                if review_text:
                    reviews.append(review_text)
            except StaleElementReferenceException:
                continue
        product["reviews"] = reviews
    except WebDriverException as exc:
        print(f"Warning: could not scrape detail page {product_url}: {exc}")
        product.setdefault("full_description", None)
        product.setdefault("reviews", [])
    finally:
        if detail_window_opened:
            try:
                driver.close()
            except WebDriverException:
                pass
        try:
            driver.switch_to.window(search_window)
        except WebDriverException:
            pass


def scrape_producthunt(query: str, max_pages: int = 3) -> pd.DataFrame:
    """Scrape Product Hunt search pages and their linked product details."""
    data_directory = Path(__file__).resolve().parents[1] / "data"
    data_directory.mkdir(parents=True, exist_ok=True)
    raw_path = data_directory / "producthunt_raw.json"
    results_path = data_directory / "producthunt_results.csv"
    raw_data = _load_raw_data(raw_path)

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")

    driver: webdriver.Chrome | None = None
    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options,
        )
        driver.get(f"https://www.producthunt.com/search?q={quote_plus(query)}")

        for page_number in range(1, max_pages + 1):
            try:
                cards = WebDriverWait(driver, 15).until(
                    EC.presence_of_all_elements_located(
                        (By.CSS_SELECTOR, PRODUCT_CARD_SELECTOR)
                    )
                )
            except TimeoutException:
                print(f"No Product Hunt cards found on page {page_number}.")
                break

            page_products: list[dict[str, object]] = []
            for card in cards:
                product: dict[str, object] = {}

                try:
                    product["name"] = card.find_element(
                        By.CSS_SELECTOR, PRODUCT_LINK_SELECTOR
                    ).text.strip() or None
                except (NoSuchElementException, StaleElementReferenceException):
                    product["name"] = None

                try:
                    product["tagline"] = card.find_element(
                        By.CSS_SELECTOR, TAGLINE_SELECTOR
                    ).text.strip() or None
                except (NoSuchElementException, StaleElementReferenceException):
                    product["tagline"] = None

                try:
                    vote_text = card.find_element(By.CSS_SELECTOR, VOTE_SELECTOR).text
                    product["vote_count"] = _parse_vote_count(vote_text)
                except (
                    NoSuchElementException,
                    StaleElementReferenceException,
                    ValueError,
                ):
                    product["vote_count"] = None

                try:
                    href = card.find_element(
                        By.CSS_SELECTOR, PRODUCT_LINK_SELECTOR
                    ).get_attribute("href")
                    product["product_url"] = (
                        urljoin(PRODUCT_HUNT_URL, href) if href else None
                    )
                except (NoSuchElementException, StaleElementReferenceException):
                    product["product_url"] = None

                page_products.append(product)

            for product in page_products:
                _enrich_product(driver, product)

            raw_data.extend(page_products)
            _save_raw_data(raw_path, raw_data)

            if page_number >= max_pages:
                break

            try:
                next_button = WebDriverWait(driver, 3).until(
                    EC.element_to_be_clickable(
                        (
                            By.XPATH,
                            '//a[@rel="next" or @aria-label="Next page"] | '
                            '//button[@aria-label="Next page" or '
                            'normalize-space()="Next page" or normalize-space()="Next"]',
                        )
                    )
                )
                first_card = cards[0]
                next_button.click()
                try:
                    WebDriverWait(driver, 15).until(EC.staleness_of(first_card))
                except TimeoutException:
                    time.sleep(0.5)
            except (TimeoutException, NoSuchElementException, WebDriverException):
                break
    finally:
        if driver is not None:
            driver.quit()

    clean_data = _load_raw_data(raw_path)
    dataframe = pd.DataFrame(clean_data, columns=OUTPUT_COLUMNS)
    dataframe = dataframe.drop_duplicates(subset="product_url").reset_index(drop=True)
    dataframe.to_csv(results_path, index=False)
    return dataframe


if __name__ == "__main__":
    df = scrape_producthunt("mental health ai", max_pages=2)
    print(df.shape)
    print(df.head(3))
