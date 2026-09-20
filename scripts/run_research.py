#!/usr/bin/env python3
"""Run the adaptive XALPHA-inspired research loop without touching 2016."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.research.controller import AdaptiveResearchController
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without evaluating factors.",
    )
    mode.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Freeze a pruned final library after research.",
    )
    return parser.parse_args()


def load_research_panel(config: dict) -> pd.DataFrame:
    path = (
        PROJECT_ROOT
        / config["paths"]["processed_data_dir"]
        / config["paths"]["research_panel_file"]
    )
    panel = pd.read_parquet(path).sort_values(["symbol", "date"]).reset_index(drop=True)
    if panel["date"].dt.year.ge(int(config["walk_forward"]["holdout_year"])).any():
        raise RuntimeError("Research panel contains final-holdout rows; aborting.")
    return panel


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["paths"]["experiment_dir"] = str(
        PROJECT_ROOT / config["paths"]["experiment_dir"]
    )
    if args.dry_run:
        settings = config.get("adaptive_research", {})
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "controller": "adaptive",
                    "max_rounds": settings.get("max_rounds", 8),
                    "max_candidates": settings.get(
                        "max_candidate_evaluations", config["search"]["max_candidates"]
                    ),
                    "holdout_year": config["walk_forward"]["holdout_year"],
                    "holdout_evaluated": False,
                },
                indent=2,
            )
        )
        return
    controller = AdaptiveResearchController(config, load_research_panel(config))
    controller.run()
    summary = controller.summary()
    if args.freeze:
        summary["freeze_manifest"] = controller.freeze(
            PROJECT_ROOT / config["paths"]["freeze_dir"]
        )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
