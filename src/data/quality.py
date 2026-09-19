"""Fail-fast quality checks for prepared research and holdout panels."""

from __future__ import annotations

import pandas as pd


REQUIRED_PANEL_COLUMNS = {
    "date",
    "symbol",
    "open",
    "close",
    "low",
    "high",
    "volume",
    "news_count",
    "fundamental_available_date",
}


def validate_panel_split(
    research: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    research_end_year: int,
    holdout_year: int,
) -> dict:
    for name, frame in (("research", research), ("holdout", holdout)):
        missing = REQUIRED_PANEL_COLUMNS - set(frame)
        if missing:
            raise ValueError(f"{name} panel missing columns: {sorted(missing)}")
        duplicates = int(frame.duplicated(["symbol", "date"]).sum())
        if duplicates:
            raise ValueError(
                f"{name} panel has {duplicates} duplicate symbol-date keys."
            )
        future_fundamentals = frame["fundamental_available_date"].notna() & (
            frame["fundamental_available_date"] > frame["date"]
        )
        if future_fundamentals.any():
            raise ValueError(f"{name} panel contains future-dated fundamentals.")

    if research["date"].dt.year.gt(research_end_year).any():
        raise ValueError("Research panel contains post-research years.")
    if not holdout["date"].dt.year.eq(holdout_year).all():
        raise ValueError("Holdout panel is not restricted to the configured year.")
    overlap = research[["symbol", "date"]].merge(
        holdout[["symbol", "date"]], on=["symbol", "date"], how="inner"
    )
    if not overlap.empty:
        raise ValueError("Research and holdout panels overlap.")

    return {
        "research_duplicate_keys": 0,
        "holdout_duplicate_keys": 0,
        "overlap_keys": 0,
        "future_fundamental_rows": 0,
        "research_years": sorted(research["date"].dt.year.unique().tolist()),
        "holdout_years": sorted(holdout["date"].dt.year.unique().tolist()),
    }
