"""Point-in-time joins that use availability dates rather than period labels."""

from __future__ import annotations

import pandas as pd


def align_fundamentals(
    prices: pd.DataFrame, fundamentals: pd.DataFrame
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    right_columns = [column for column in fundamentals if column != "symbol"]
    for symbol, price_rows in prices.groupby("symbol", sort=False):
        fundamental_rows = fundamentals.loc[
            fundamentals["symbol"].eq(symbol), right_columns
        ]
        if fundamental_rows.empty:
            chunks.append(price_rows.copy())
            continue
        chunks.append(
            pd.merge_asof(
                price_rows.sort_values("date"),
                fundamental_rows.sort_values("fundamental_available_date"),
                left_on="date",
                right_on="fundamental_available_date",
                direction="backward",
                allow_exact_matches=True,
            )
        )
    panel = pd.concat(chunks, ignore_index=True).sort_values(["symbol", "date"])
    invalid = panel["fundamental_available_date"].notna() & (
        panel["fundamental_available_date"] > panel["date"]
    )
    if invalid.any():
        raise ValueError("Point-in-time fundamental join produced future-dated rows.")
    return panel


def aggregate_available_news(news: pd.DataFrame) -> pd.DataFrame:
    if news.empty:
        return pd.DataFrame(columns=["symbol", "date", "news_count", "news_publishers"])
    return (
        news.groupby(["symbol", "news_available_date"], as_index=False)
        .agg(news_count=("headline", "size"), news_publishers=("publisher", "nunique"))
        .rename(columns={"news_available_date": "date"})
        .sort_values(["symbol", "date"])
    )
