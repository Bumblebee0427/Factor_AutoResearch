#!/usr/bin/env python3
"""Run the six predeclared research arms with an auditable, resumable manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.brains.micro import infer_mechanism
from src.core.schemas import FactorArtifact
from src.research.comparison import summarize_search
from src.research.controller import AdaptiveResearchController
from src.research.loop import ResearchLoop
from src.selection.archive import ParentPool
from src.selection.gates import classify_tier
from src.utils.config import config_digest, load_config


ARMS = (
    "fixed_deterministic", "adaptive_deterministic", "luna_luna",
    "sol_sol", "sol_luna", "luna_sol",
)
ASSIGNMENTS = {
    "luna_luna": ("luna", "luna"),
    "sol_sol": ("sol", "sol"),
    "sol_luna": ("sol", "luna"),
    "luna_sol": ("luna", "sol"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def protocol_snapshot(config: dict) -> dict:
    adaptive = config["adaptive_research"]
    keys = (
        "parent_min_mean_ic", "parent_min_positive_folds", "parent_max_turnover",
        "elite_min_mean_ic", "elite_min_tstat", "elite_min_positive_folds",
        "elite_min_high_cost_sharpe", "elite_max_turnover", "parent_pool_size",
        "parent_cluster_correlation", "parent_cluster_min_dates",
        "parent_cluster_min_names", "parent_cluster_max_dates",
    )
    return {
        "research_panel_file": config["paths"]["research_panel_file"],
        "random_seed": config["project"]["random_seed"],
        **{key: config[key] for key in ("data", "universe", "walk_forward", "evaluation", "dsl", "gates")},
        "adaptive_selection": {key: adaptive[key] for key in keys},
    }


def load_research_panel(config: dict) -> pd.DataFrame:
    path = PROJECT_ROOT / config["paths"]["processed_data_dir"] / config["paths"]["research_panel_file"]
    if path.name == config["paths"]["holdout_panel_file"]:
        raise RuntimeError("Research panel path points at the final holdout.")
    panel = pd.read_parquet(path).sort_values(["symbol", "date"]).reset_index(drop=True)
    if panel["date"].dt.year.ge(int(config["walk_forward"]["holdout_year"])).any():
        raise RuntimeError("Research panel contains final-holdout rows.")
    return panel


def arm_is_reusable(manifest: dict, arm: str) -> bool:
    return manifest.get("arms", {}).get(arm, {}).get("status") == "complete"


def fixed_common_clusters(loop: ResearchLoop, panel: pd.DataFrame, config: dict) -> dict:
    settings = config["adaptive_research"]
    pool = ParentPool(
        int(settings["parent_pool_size"]),
        correlation_threshold=float(settings["parent_cluster_correlation"]),
        minimum_overlap_dates=int(settings["parent_cluster_min_dates"]),
        minimum_names_per_date=int(settings["parent_cluster_min_names"]),
        maximum_comparison_dates=int(settings["parent_cluster_max_dates"]),
    )
    signals = {}
    admissions = []
    for record in loop.records:
        tier = classify_tier(record, settings).tier
        if tier not in {"PARENT", "ELITE"}:
            continue
        spec = loop.evaluated_specs[record.factor_id]
        signal = loop.builder.build(panel, spec)
        signals[record.factor_id] = signal
        admission = pool.add(FactorArtifact(spec, record, infer_mechanism(spec)), signal, signals, panel["date"])
        admissions.append({"factor_id": record.factor_id, "status": admission.status, "new_cluster": admission.new_cluster})
    return {"stats": pool.cluster_stats(), "admissions": admissions}


def execute_arm(arm: str, base: dict, panel: pd.DataFrame, root: Path, budget: int, models: dict) -> dict:
    config = copy.deepcopy(base)
    directory = root / arm
    directory.mkdir(parents=True, exist_ok=False)
    config["paths"]["experiment_dir"] = str(directory)
    config["search"]["max_candidates"] = budget if arm == "fixed_deterministic" else max(int(config["search"]["max_candidates"]), budget)
    config["adaptive_research"]["max_candidate_evaluations"] = budget
    config["adaptive_research"]["minimum_evidence_before_stop"] = budget
    llm = arm in ASSIGNMENTS
    config["llm"]["enabled"] = llm
    config["llm"]["required"] = llm
    if llm:
        macro, micro = ASSIGNMENTS[arm]
        config["llm"]["macro"] = {"model": models[macro], "reasoning_effort": "high"}
        config["llm"]["micro"] = {"model": models[micro], "reasoning_effort": "high"}
    (directory / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True, default=str) + "\n")

    if arm == "fixed_deterministic":
        loop = ResearchLoop(config, panel)
        records = loop.run()
        specs = directory / "factor_specs.jsonl"
        specs.write_text("".join(json.dumps(spec.to_dict(), sort_keys=True) + "\n" for spec in loop.evaluated_specs.values()))
        clusters = fixed_common_clusters(loop, panel, config)
        (directory / "common_parent_clusters.json").write_text(json.dumps(clusters, indent=2, sort_keys=True) + "\n")
        events_path = directory / "generation_events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line]
        summary = summarize_search(records, events, arm=arm)
    else:
        controller = AdaptiveResearchController(config, panel, use_llm=llm)
        controller.run()
        summary = controller.summary()
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    return {"artifact_dir": str(directory.relative_to(PROJECT_ROOT)), "config_sha256": config_digest(config), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--pricing", type=Path, default=PROJECT_ROOT / "experiment_pricing.yaml")
    parser.add_argument("--luna-model", default=os.environ.get("LUNA_MODEL_ID") or os.environ.get("OPENAI_FACTOR_MODEL"))
    parser.add_argument("--sol-model", default=os.environ.get("SOL_MODEL_ID"))
    args = parser.parse_args()

    config = load_config(args.config)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_id):
        raise ValueError("run-id must be a short filename-safe label.")
    if args.budget != 60:
        raise ValueError("The final main-arm budget is precommitted at 60.")
    selected = tuple(dict.fromkeys(args.arms))
    models = {"luna": args.luna_model or config["llm"]["model"], "sol": args.sol_model}
    if any(arm in ASSIGNMENTS and not all(models[name] for name in ASSIGNMENTS[arm]) for arm in selected):
        raise ValueError("Specify valid --luna-model and --sol-model IDs for the selected LLM arms.")
    if any(arm in ASSIGNMENTS for arm in selected):
        if not os.environ.get(config["llm"].get("api_key_env", "OPENAI_API_KEY")):
            raise RuntimeError("The configured OpenAI API key is unavailable.")
        socket.getaddrinfo("api.openai.com", 443)  # Fail before consuming deterministic coverage budget.
    panel = load_research_panel(config)
    root = PROJECT_ROOT / "artifacts" / "experiments" / "final_experiments" / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    protocol = protocol_snapshot(config)
    pricing = yaml.safe_load(args.pricing.read_text(encoding="utf-8"))["experiment_pricing"]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["git_commit_sha"] != commit or manifest["protocol_sha256"] != digest(protocol) or manifest["candidate_budget"] != args.budget or manifest["pricing_sha256"] != digest(pricing):
            raise RuntimeError("Existing final manifest has a different code or research protocol.")
        for arm in selected:
            if arm in ASSIGNMENTS and manifest["model_ids"] != models:
                raise RuntimeError("Existing final manifest has different LLM model IDs.")
    else:
        manifest = {
            "run_id": args.run_id,
            "started_at_utc": utc_now(),
            "git_commit_sha": commit,
            "base_config_sha256": config_digest(config),
            "protocol_sha256": digest(protocol),
            "protocol": protocol,
            "random_seed": config["project"]["random_seed"],
            "candidate_budget": args.budget,
            "research_end_year": config["walk_forward"]["research_end_year"],
            "holdout_evaluated": False,
            "pricing_as_of": str(pricing["as_of"]),
            "pricing_sha256": digest(pricing),
            "pricing_source_note": pricing["source_note"],
            "pricing_source_urls": pricing["source_urls"],
            "pricing": pricing,
            "model_ids": models,
            "reasoning_effort": "high",
            "arms": {},
        }
    for arm in selected:
        if arm_is_reusable(manifest, arm):
            continue
        if (root / arm).exists():
            raise RuntimeError(f"Partial arm directory exists: {root / arm}; preserve it and use a new run ID.")
        started = utc_now()
        try:
            result = execute_arm(arm, config, panel, root, args.budget, models)
        except Exception as error:
            manifest["arms"][arm] = {"status": "failed", "started_at_utc": started, "failed_at_utc": utc_now(), "error_type": type(error).__name__}
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
            raise
        manifest["arms"][arm] = {"status": "complete", "started_at_utc": started, "completed_at_utc": utc_now(), **result}
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
        print(json.dumps({"arm": arm, "status": "complete", "candidates": result["summary"].get("candidates_tested", result["summary"].get("candidates_evaluated"))}))


if __name__ == "__main__":
    main()
