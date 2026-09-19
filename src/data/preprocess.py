"""Build the deterministic panel and physically isolate the 2016 holdout."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.loader import DataCatalog, RawDatasets
from src.data.point_in_time import aggregate_available_news, align_fundamentals
from src.data.quality import validate_panel_split


def build_daily_panel(raw: RawDatasets) -> pd.DataFrame:
    panel = align_fundamentals(raw.prices, raw.fundamentals)
    panel = panel.merge(
        aggregate_available_news(raw.news), on=["symbol", "date"], how="left"
    )
    panel[["news_count", "news_publishers"]] = panel[
        ["news_count", "news_publishers"]
    ].fillna(0.0)
    panel = panel.merge(raw.securities, on="symbol", how="left", validate="many_to_one")
    return panel.sort_values(["symbol", "date"]).reset_index(drop=True)


def apply_universe_filters(panel: pd.DataFrame, universe: dict) -> pd.DataFrame:
    start = pd.Timestamp(universe["start_date"])
    end = pd.Timestamp(universe["end_date"])
    frame = panel.loc[panel["date"].between(start, end)].copy()
    frame = frame.loc[frame["close"] >= float(universe["minimum_price"])]
    frame["history_days"] = frame.groupby("symbol", sort=False).cumcount().add(1)
    frame = frame.loc[frame["history_days"] >= int(universe["minimum_history_days"])]
    if universe.get("require_security_master_match", False):
        frame = frame.loc[frame["security"].notna()]
    return frame.reset_index(drop=True)


def split_research_holdout(
    panel: pd.DataFrame,
    *,
    research_end_year: int,
    holdout_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    research = panel.loc[panel["date"].dt.year <= research_end_year].copy()
    holdout = panel.loc[panel["date"].dt.year.eq(holdout_year)].copy()
    if research.empty or holdout.empty:
        raise ValueError("Research/holdout split produced an empty dataset.")
    if research["date"].max() >= holdout["date"].min():
        raise ValueError("Research and holdout periods overlap.")
    return research, holdout


def prepare_research_datasets(
    config: dict,
    *,
    project_root: str | Path,
    force: bool = False,
) -> dict:
    root = Path(project_root)
    raw_root = root / config["paths"]["raw_data_dir"]
    output_dir = root / config["paths"]["processed_data_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    research_path = output_dir / config["paths"]["research_panel_file"]
    holdout_path = output_dir / config["paths"]["holdout_panel_file"]
    for path in (research_path, holdout_path):
        if path.exists() and not force:
            raise FileExistsError(f"{path} exists; pass --force to rebuild it.")

    catalog = DataCatalog(raw_root, config["data"])
    raw = catalog.load_all(
        start=str(config["universe"]["start_date"]),
        end=str(pd.Timestamp(config["universe"]["end_date"]) + pd.Timedelta(days=1)),
    )
    panel = apply_universe_filters(build_daily_panel(raw), config["universe"])
    research, holdout = split_research_holdout(
        panel,
        research_end_year=int(config["walk_forward"]["research_end_year"]),
        holdout_year=int(config["walk_forward"]["holdout_year"]),
    )
    quality = validate_panel_split(
        research,
        holdout,
        research_end_year=int(config["walk_forward"]["research_end_year"]),
        holdout_year=int(config["walk_forward"]["holdout_year"]),
    )
    research.to_parquet(research_path, index=False)
    holdout.to_parquet(holdout_path, index=False)

    summary = {
        "research": {
            "path": str(research_path),
            "rows": len(research),
            "symbols": int(research["symbol"].nunique()),
            "start": str(research["date"].min().date()),
            "end": str(research["date"].max().date()),
        },
        "holdout": {
            "path": str(holdout_path),
            "rows": len(holdout),
            "symbols": int(holdout["symbol"].nunique()),
            "start": str(holdout["date"].min().date()),
            "end": str(holdout["date"].max().date()),
        },
        "news_rows_after_filtering": len(raw.news),
        "fundamental_reporting_lag_days": config["data"][
            "fundamental_reporting_lag_days"
        ],
        "news_availability_lag_days": config["data"]["news_availability_lag_days"],
        "quality": quality,
    }
    (output_dir / "panel_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
