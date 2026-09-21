#!/usr/bin/env python3
"""Compare adaptive deterministic and Luna arms under one 2010-2015 protocol."""

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

from src.research.controller import AdaptiveResearchController
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=("deterministic", "llm", "both"), default="both"
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--budget", type=int, default=60)
    return parser.parse_args()


def load_panel(config: dict) -> pd.DataFrame:
    path = (
        PROJECT_ROOT
        / config["paths"]["processed_data_dir"]
        / config["paths"]["research_panel_file"]
    )
    panel = pd.read_parquet(path).sort_values(["symbol", "date"]).reset_index(drop=True)
    holdout_year = int(config["walk_forward"]["holdout_year"])
    if panel["date"].dt.year.ge(holdout_year).any():
        raise RuntimeError("Adaptive comparison cannot load final-holdout rows.")
    return panel


def execute_arm(
    base_config: dict,
    panel: pd.DataFrame,
    root: Path,
    arm: str,
    budget: int,
) -> dict:
    config = copy.deepcopy(base_config)
    directory = root / arm
    directory.mkdir(parents=True, exist_ok=False)
    config["paths"]["experiment_dir"] = str(directory)
    config["adaptive_research"]["max_candidate_evaluations"] = budget
    config["adaptive_research"]["minimum_evidence_before_stop"] = budget
    config["search"]["max_candidates"] = max(
        int(config["search"]["max_candidates"]), budget
    )
    use_llm = arm == "llm"
    config["llm"]["enabled"] = use_llm
    config["llm"]["required"] = use_llm
    (directory / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    controller = AdaptiveResearchController(config, panel, use_llm=use_llm)
    controller.run()
    summary = controller.summary()
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return summary


def render_markdown(summaries: list[dict]) -> str:
    fields = (
        ("Candidates tested", "candidates_tested"),
        ("First parent index", "candidates_to_first_parent"),
        ("First elite index", "candidates_to_first_elite"),
        ("Effective information / 10", "valid_information_per_10"),
        ("Duplicate formula ratio", "duplicate_formula_ratio"),
        ("Invalid evaluated proposal ratio", "invalid_proposal_ratio"),
        ("Mechanisms covered", "mechanism_coverage_count"),
        ("Parent pool size", "parent_pool_size"),
        ("Elite archive size", "elite_archive_size"),
        ("Elite mechanism diversity", "elite_mechanism_diversity"),
        ("Elite fold-sign stability", "elite_walk_forward_sign_stability"),
        ("LLM calls", "llm_calls"),
        ("LLM invalid proposal ratio", "llm_invalid_proposal_ratio"),
    )
    names = [summary["arm"] for summary in summaries]
    lines = [
        "# Adaptive search comparison",
        "",
        "Research folds only; the 2016 final holdout was not loaded.",
        "",
        "| Metric | " + " | ".join(names) + " |",
        "| --- | " + " | ".join("---:" for _ in names) + " |",
    ]
    for label, field in fields:
        values = []
        for summary in summaries:
            value = summary.get(field)
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append("—" if value is None else str(value))
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.budget < 1 or args.budget > int(config["search"]["max_candidates"]):
        raise ValueError("Budget must be positive and within the global search cap.")
    if args.arm in {"llm", "both"} and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for the Luna arm.")
    panel = load_panel(config)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = PROJECT_ROOT / "artifacts" / "experiments" / "adaptive_comparisons" / run_id
    root.mkdir(parents=True, exist_ok=True)
    arms = ("deterministic", "llm") if args.arm == "both" else (args.arm,)
    summaries = [execute_arm(config, panel, root, arm, args.budget) for arm in arms]
    result = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_budget_per_arm": args.budget,
        "research_end_year": config["walk_forward"]["research_end_year"],
        "holdout_evaluated": False,
        "arms": summaries,
    }
    output = root / "comparison.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (root / "comparison.md").write_text(render_markdown(summaries), encoding="utf-8")
    print(json.dumps({"comparison": str(output), "arms": summaries}, indent=2))


if __name__ == "__main__":
    main()
