#!/usr/bin/env python3
"""Run deterministic and/or LLM search arms under one locked research protocol."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.research.comparison import render_comparison_markdown, summarize_search
from src.research.loop import ResearchLoop
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=("deterministic", "llm", "both"), default="both"
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--run-id", type=str)
    return parser.parse_args()


def load_events(directory: Path) -> list[dict]:
    path = directory / "generation_events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def execute_arm(base_config: dict, panel: pd.DataFrame, root: Path, arm: str) -> dict:
    config = copy.deepcopy(base_config)
    experiment_dir = root / arm
    experiment_dir.mkdir(parents=True, exist_ok=False)
    config["paths"]["experiment_dir"] = str(experiment_dir)
    config["llm"]["enabled"] = arm == "llm"
    config["llm"]["required"] = arm == "llm"
    (experiment_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    loop = ResearchLoop(config, panel)
    records = loop.run()
    summary = summarize_search(
        records,
        load_events(experiment_dir),
        arm=arm,
    )
    (experiment_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def completed_summaries(root: Path) -> list[dict]:
    summaries = []
    for arm in ("deterministic", "llm"):
        path = root / arm / "summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    return summaries


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    panel_path = (
        PROJECT_ROOT
        / config["paths"]["processed_data_dir"]
        / config["paths"]["research_panel_file"]
    )
    panel = (
        pd.read_parquet(panel_path)
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )
    holdout_year = int(config["walk_forward"]["holdout_year"])
    if panel["date"].dt.year.ge(holdout_year).any():
        raise RuntimeError("Research panel contains holdout rows; aborting.")
    if args.arm in {"llm", "both"} and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is required for the LLM arm. Export it or place it in "
            "the ignored project .env file."
        )
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = PROJECT_ROOT / "artifacts" / "experiments" / "comparisons" / run_id
    root.mkdir(parents=True, exist_ok=True)
    arms = ("deterministic", "llm") if args.arm == "both" else (args.arm,)
    for arm in arms:
        execute_arm(config, panel, root, arm)
    summaries = completed_summaries(root)
    result = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_period_end": int(config["walk_forward"]["research_end_year"]),
        "holdout_evaluated": False,
        "arms": summaries,
    }
    output = root / "comparison.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "comparison.md").write_text(
        render_comparison_markdown(summaries), encoding="utf-8"
    )
    print(json.dumps({"comparison": str(output), "arms": summaries}, indent=2))


if __name__ == "__main__":
    main()
