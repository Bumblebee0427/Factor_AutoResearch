#!/usr/bin/env python3
"""Validate the research design or execute the configured search loop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.research.loop import ResearchLoop
from src.research.generator import seed_candidates
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Run research folds. Without this flag, perform a safe dry-run only.",
    )
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Evaluate one real candidate against the prepared research panel.",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Persist the promoted library and immutable run manifest after execution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["paths"]["experiment_dir"] = str(
        PROJECT_ROOT / config["paths"]["experiment_dir"]
    )
    if not args.execute and not args.smoke:
        print(json.dumps(ResearchLoop(config).dry_run(), indent=2))
        return

    panel_path = (
        PROJECT_ROOT
        / config["paths"]["processed_data_dir"]
        / config["paths"]["research_panel_file"]
    )
    if not panel_path.exists():
        raise FileNotFoundError(
            "Prepared panel not found. Run scripts/prepare_data.py first."
        )
    panel = (
        pd.read_parquet(panel_path)
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )
    holdout_year = int(config["walk_forward"]["holdout_year"])
    if panel["date"].dt.year.ge(holdout_year).any():
        raise RuntimeError("Research panel contains holdout rows; aborting.")
    loop = ResearchLoop(config, panel)
    if args.smoke:
        spec = next(item for item in seed_candidates() if not item.demonstration_only)
        record, _ = loop.evaluate_candidate(spec)
        print(json.dumps(record.to_dict(), indent=2, default=str))
        return
    results = loop.run()
    summary = {
        "evaluated": len(results),
        "kept": sum(item.decision == "KEEP" for item in results),
        "holdout_evaluated": False,
    }
    if args.freeze:
        freeze_dir = PROJECT_ROOT / config["paths"]["freeze_dir"]
        summary["freeze_manifest"] = loop.freeze(freeze_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
