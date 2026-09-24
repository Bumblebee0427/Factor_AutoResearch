#!/usr/bin/env python3
"""Analyze configured Elite gate failures from an adaptive research artifact."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.selection.diagnostics import diagnose_values
from src.utils.config import load_config


GATE_COLUMNS = (
    ("mean_ic", "failed_mean_ic"),
    ("tstat", "failed_tstat"),
    ("positive_folds", "failed_positive_folds"),
    ("high_cost_sharpe", "failed_high_cost_sharpe"),
    ("turnover", "failed_turnover"),
)


def _artifact_root(path: Path) -> Path:
    if (path / "adaptive" / "research_rounds.jsonl").exists():
        return path
    if (path / "llm" / "adaptive" / "research_rounds.jsonl").exists():
        return path / "llm"
    if (path / "deterministic" / "adaptive" / "research_rounds.jsonl").exists():
        return path / "deterministic"
    raise FileNotFoundError(f"No adaptive research_rounds.jsonl found under {path}")


def _load_outcomes(root: Path) -> list[dict]:
    rounds_path = root / "adaptive" / "research_rounds.jsonl"
    outcomes = []
    for line in rounds_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            round_record = json.loads(line)
            outcomes.extend(round_record.get("outcomes", []))
    return outcomes


def analyze(outcomes: list[dict], config: dict, run_id: str) -> dict:
    adaptive = config["adaptive_research"]
    rows = []
    gate_counts = Counter()
    histogram = Counter()
    mechanism_counts: dict[str, Counter] = {}
    parent_count = 0
    elite_count = 0
    integrity_passed = 0
    for outcome in outcomes:
        metrics = dict(outcome.get("metrics", {}))
        integrity_ok = bool(outcome.get("integrity_passed", True))
        if integrity_ok:
            integrity_passed += 1
        diagnostic = diagnose_values(
            {
                **metrics,
                "integrity_passed": integrity_ok,
                "decision": outcome.get("decision", "RETIRED"),
            },
            adaptive,
            str(outcome.get("factor_id", "")),
        )
        failed = diagnostic["failed_gates"]
        decision = outcome.get("decision", "RETIRED")
        mechanism = outcome.get("mechanism", "UNKNOWN") or "UNKNOWN"
        if decision == "PARENT":
            parent_count += 1
            histogram[len(failed)] += 1
            for gate in failed:
                gate_counts[gate] += 1
        elif decision == "ELITE":
            elite_count += 1
        per_mechanism = mechanism_counts.setdefault(mechanism, Counter())
        if decision == "PARENT":
            per_mechanism["parents"] += 1
            for gate in failed:
                per_mechanism[gate] += 1
        rows.append(
            {
                "factor_id": outcome.get("factor_id"),
                "mechanism": mechanism,
                "decision": decision,
                "integrity_passed": integrity_ok,
                "mean_rank_ic": metrics.get("mean_rank_ic"),
                "newey_west_tstat": metrics.get("newey_west_tstat"),
                "positive_fold_count": metrics.get("positive_fold_count"),
                "high_cost_sharpe": metrics.get("high_cost_sharpe"),
                "turnover": metrics.get("turnover"),
                "failed_gates": failed,
                "distance_to_threshold": diagnostic["distance_to_threshold"],
            }
        )

    by_mechanism = {}
    for mechanism in sorted(set(mechanism_counts) | {
        "PRICE_TREND", "PRICE_REVERSAL", "VOLATILITY", "PRICE_VOLUME",
        "FUNDAMENTAL_VALUE", "FUNDAMENTAL_QUALITY", "NEWS_ATTENTION",
        "CROSS_DOMAIN_REGIME",
    }):
        count = mechanism_counts.get(mechanism, Counter())
        by_mechanism[mechanism] = {
            "parents": count.get("parents", 0),
            **{gate: count.get(gate, 0) for gate, _ in GATE_COLUMNS},
        }
    return {
        "run_id": run_id,
        "sample": "adaptive research outcomes only; 2010-2015 research folds",
        "total_evaluated": len(outcomes),
        "integrity_passed": integrity_passed,
        "parents": parent_count,
        "elites": elite_count,
        "parent_gate_failure_counts": {
            gate: gate_counts.get(gate, 0) for gate, _ in GATE_COLUMNS
        },
        "parent_failure_count_histogram": {
            "exactly_1": histogram.get(1, 0),
            "exactly_2": histogram.get(2, 0),
            "3_or_more": sum(value for key, value in histogram.items() if key >= 3),
            "zero": histogram.get(0, 0),
        },
        "by_mechanism": by_mechanism,
        "elite_thresholds": {
            "mean_ic": adaptive["elite_min_mean_ic"],
            "tstat": adaptive["elite_min_tstat"],
            "positive_folds": adaptive["elite_min_positive_folds"],
            "high_cost_sharpe": adaptive["elite_min_high_cost_sharpe"],
            "turnover_max": adaptive["elite_max_turnover"],
        },
        "thresholds_changed": False,
        "holdout_2016_loaded_or_evaluated": False,
        "rows": rows,
    }


def write_report(result: dict, output_dir: Path) -> None:
    tables = output_dir / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (output_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (output_dir / "outputs" / "elite_bottleneck_summary.json").write_text(
        json.dumps({key: value for key, value in result.items() if key != "rows"},
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    gate_path = tables / "elite_gate_failures.csv"
    with gate_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["factor_id", "mechanism", "decision", "integrity_passed"] + [
            label for _, label in GATE_COLUMNS
        ] + ["failed_gates"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["rows"]:
            failed = set(row["failed_gates"])
            writer.writerow({
                **{key: row.get(key) for key in fields if key in row},
                **{label: int(gate in failed) for gate, label in GATE_COLUMNS},
                "failed_gates": "|".join(row["failed_gates"]),
            })
    mechanism_path = tables / "elite_gate_failures_by_mechanism.csv"
    with mechanism_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["mechanism", "parents"] + [gate for gate, _ in GATE_COLUMNS]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mechanism, values in result["by_mechanism"].items():
            writer.writerow({"mechanism": mechanism, **values})
    distance_path = tables / "elite_gate_distance.csv"
    with distance_path.open("w", newline="", encoding="utf-8") as handle:
        distance_keys = ("mean_ic", "tstat", "positive_folds", "high_cost_sharpe", "turnover")
        fields = ["factor_id", "mechanism", "decision"] + [f"gap_{key}" for key in distance_keys]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow({
                "factor_id": row["factor_id"],
                "mechanism": row["mechanism"],
                "decision": row["decision"],
                **{f"gap_{key}": row["distance_to_threshold"].get(key) for key in distance_keys},
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="comparison run directory, or one arm directory")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    root = _artifact_root(args.run_dir)
    run_config_path = root / "run_config.json"
    config = (
        json.loads(run_config_path.read_text(encoding="utf-8"))
        if run_config_path.exists()
        else load_config(args.config)
    )
    outcomes = _load_outcomes(root)
    result = analyze(outcomes, config, root.name)
    output_dir = args.output_dir or root / "diagnostics"
    write_report(result, output_dir)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
