from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from string import ascii_lowercase
from typing import Iterable

import google_play_scraper


GOOGLE_PLAY_PAGE_SIZE = 30
EXPANSION_BATCH_SIZE = 8
MAX_WORKERS = 4


def _search_once(query: str, country: str, lang: str) -> list[dict]:
    """Run one upstream search without allowing one failed variant to abort."""
    try:
        return google_play_scraper.search(
            query,
            n_hits=GOOGLE_PLAY_PAGE_SIZE,
            country=country,
            lang=lang,
        )
    except Exception as exc:
        print(f"Warning: Google Play search variant {query!r} failed: {exc}")
        return []


def _query_variants(query: str) -> Iterable[str]:
    """Yield deterministic expansions while preserving the exact query first."""
    normalized_query = " ".join(query.split())
    yield normalized_query

    for letter in ascii_lowercase:
        yield f"{normalized_query} {letter}"
    for letter in ascii_lowercase:
        yield f"{letter} {normalized_query}"

    generic_variants = [
        f"best {normalized_query}",
        f"free {normalized_query}",
        f"new {normalized_query}",
        f"top {normalized_query}",
        f"{normalized_query} free",
        f"{normalized_query} pro",
        f"{normalized_query} mobile",
        f"{normalized_query} android",
        f"{normalized_query} daily",
        f"{normalized_query} tool",
    ]
    for variant in generic_variants:
        yield variant

    without_app = " ".join(
        token for token in normalized_query.split() if token.lower() != "app"
    )
    if without_app and without_app != normalized_query:
        yield without_app


def _append_unique(
    destination: list[dict],
    candidates: Iterable[dict],
    seen_app_ids: set[str],
    limit: int,
) -> None:
    for app_data in candidates:
        app_id = app_data.get("appId")
        if not app_id or app_id in seen_app_ids:
            continue
        seen_app_ids.add(app_id)
        destination.append(app_data)
        if len(destination) >= limit:
            return


def expanded_search(
    query: str,
    n_results: int = 30,
    country: str = "us",
    lang: str = "en",
) -> list[dict]:
    """Return up to ``n_results`` unique apps using ranked query expansion.

    Google Play currently returns at most 30 apps and no usable continuation
    token. The exact query is therefore fetched first, followed by controlled
    autocomplete-style variants only when more results are requested.
    """
    if n_results <= 0 or not query.strip():
        return []

    requested = min(n_results, 100)
    variants = iter(_query_variants(query))
    exact_query = next(variants)
    results: list[dict] = []
    seen_app_ids: set[str] = set()

    exact_results = _search_once(exact_query, country=country, lang=lang)
    _append_unique(results, exact_results, seen_app_ids, requested)

    if len(results) >= requested:
        return results[:requested]

    while len(results) < requested:
        batch = []
        for _ in range(EXPANSION_BATCH_SIZE):
            try:
                batch.append(next(variants))
            except StopIteration:
                break

        if not batch:
            break

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            pages = executor.map(
                lambda variant: _search_once(
                    variant,
                    country=country,
                    lang=lang,
                ),
                batch,
            )
            for page in pages:
                _append_unique(results, page, seen_app_ids, requested)
                if len(results) >= requested:
                    break

    if len(results) < requested:
        print(
            f"Warning: requested {requested} apps, but Google Play search "
            f"expansion found {len(results)} unique results."
        )

    return results[:requested]
