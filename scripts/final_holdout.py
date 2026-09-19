#!/usr/bin/env python3
"""Evaluate the frozen factor library on 2016 exactly once."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import evaluate_walk_forward
from src.evaluation.validation import WalkForwardFold
from src.factors.builder import FactorBuilder
from src.factors.schema import FactorSpec


def main() -> None:
    freeze_dir = PROJECT_ROOT / "artifacts" / "frozen"
    manifest_path = freeze_dir / "freeze_manifest.json"
    library_path = freeze_dir / "factor_library.json"
    config_path = freeze_dir / "config.json"
    missing = [
        path for path in (manifest_path, library_path, config_path) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Frozen research artifacts are missing: {missing}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    holdout_year = int(manifest["holdout_year"])
    holdout_path = (
        PROJECT_ROOT
        / config["paths"]["processed_data_dir"]
        / config["paths"]["holdout_panel_file"]
    )
    research_path = (
        PROJECT_ROOT
        / config["paths"]["processed_data_dir"]
        / config["paths"]["research_panel_file"]
    )
    holdout = pd.read_parquet(holdout_path)
    if not holdout["date"].dt.year.eq(holdout_year).all():
        raise RuntimeError(
            "Holdout file contains dates outside the frozen holdout year."
        )
    # Research history supplies rolling-feature warm-up only; the evaluator scores 2016.
    history = pd.read_parquet(research_path)
    if history["date"].dt.year.ge(holdout_year).any():
        raise RuntimeError("Research history contains holdout rows.")
    panel = (
        pd.concat([history, holdout], ignore_index=True)
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )

    output_dir = PROJECT_ROOT / config["paths"]["output_dir"] / "holdout"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "holdout_results.json"
    if output_path.exists():
        raise FileExistsError(
            "Holdout results already exist. The final holdout is intentionally one-shot."
        )

    fold = WalkForwardFold(
        name="final_holdout_2016",
        train_start=date(2010, 1, 1),
        train_end=date(2015, 12, 31),
        validation_start=date(2016, 1, 1),
        validation_end=date(2016, 12, 31),
    )
    builder = FactorBuilder()
    specs = [
        FactorSpec.from_dict(item)
        for item in json.loads(library_path.read_text(encoding="utf-8"))
    ]
    results = []
    for spec in specs:
        factor = builder.build(panel, spec)
        metrics = evaluate_walk_forward(panel, factor, [fold], config["evaluation"])
        results.append({"factor": spec.to_dict(), "metrics": metrics.to_dict()})
    output_path.write_text(
        json.dumps(
            {"freeze_manifest": manifest, "factor_results": results},
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
