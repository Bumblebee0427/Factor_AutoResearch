#!/usr/bin/env python3
"""Recover one failed final arm from its last fully committed research round.

The aborted directory is copied intact before removing uncommitted tail records.
Only completed-round outcomes are replayed; the failed model round is requested
again and its extra API usage remains documented in the archived attempt.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_final_experiments import load_research_panel, utc_now
from src.core.schemas import FactorArtifact, ResearchMemory, ResearchOutcome
from src.factors.schema import FactorSpec
from src.research.controller import AdaptiveResearchController
from src.utils.config import config_digest
from src.utils.logging import ExperimentRecord


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def restore_controller(directory: Path, panel) -> tuple[AdaptiveResearchController, int, int]:
    config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    controller = AdaptiveResearchController(config, panel, use_llm=True)
    rounds = read_jsonl(directory / "adaptive" / "research_rounds.jsonl")
    records = read_jsonl(directory / "candidates.jsonl")
    events = read_jsonl(directory / "adaptive" / "llm_events.jsonl")
    if not rounds or [item["round_id"] for item in rounds] != list(range(len(rounds))):
        raise RuntimeError("Completed research rounds are not a contiguous prefix.")
    committed = [outcome for rnd in rounds for outcome in rnd["outcomes"]]
    count = len(committed)
    if count >= len(records) or len(events) <= len(rounds):
        raise RuntimeError("No incomplete candidate/LLM tail was found to recover.")
    if [row["factor_id"] for row in records[:count]] != [row["factor_id"] for row in committed]:
        raise RuntimeError("Candidate record prefix does not match completed rounds.")
    if [item["round_id"] for item in events[:len(rounds)]] != list(range(len(rounds))):
        raise RuntimeError("LLM event prefix does not match completed rounds.")
    checkpoint = directory / "adaptive" / f"checkpoint_round_{len(rounds)-1:03d}.json"
    controller.memory = ResearchMemory.from_dict(json.loads(checkpoint.read_text(encoding="utf-8")))
    controller.total_proposed = sum(int(rnd["proposed"]) for rnd in rounds)
    controller.duplicate_count = sum(int(rnd["duplicates"]) for rnd in rounds)
    controller.llm_events = events[:len(rounds)]
    controller.llm_raw_proposals = sum(int(event["micro"].get("raw_proposal_count", 0)) for event in controller.llm_events)
    controller.llm_rejected_proposals = sum(len(event["micro"].get("rejected", [])) for event in controller.llm_events)
    for raw, saved in zip(records[:count], committed):
        spec = FactorSpec.from_dict(saved["factor_spec"])
        record = ExperimentRecord(**raw)
        tier = saved["decision"]
        mechanism = saved["mechanism"]
        controller.base_loop.records.append(record)
        controller.base_loop.evaluated_specs[spec.factor_id] = spec
        controller.outcomes.append(ResearchOutcome(
            spec, record, tier, mechanism, saved["action"],
            saved.get("elite_gate_diagnostic", {}), saved.get("repair_result", {}),
        ))
        controller.tested_ids.add(spec.factor_id)
        controller.tested_formulas.add(spec.canonical_formula)
        controller.tiers[spec.factor_id] = tier
        if tier in {"PARENT", "ELITE"} and record.integrity_passed:
            signal = controller.base_loop.builder.build(panel, spec)
            controller.signals[spec.factor_id] = signal
            artifact = FactorArtifact(spec, record, mechanism)
            controller.parent_pool.add(artifact, signal, controller.signals, panel["date"])
            if tier == "ELITE":
                controller.elite_archive.add(artifact)
                controller.base_loop.promoted_specs[spec.factor_id] = spec
                controller.base_loop.promoted_signals[spec.factor_id] = signal
    if list(controller.parent_pool.items) != controller.memory.parent_pool_ids:
        raise RuntimeError("Replayed Parent pool differs from saved memory.")
    if controller.parent_pool.cluster_stats() != controller.memory.parent_cluster_stats:
        raise RuntimeError("Replayed Parent clusters differ from saved memory.")
    if list(controller.elite_archive.items) != controller.memory.elite_archive_ids:
        raise RuntimeError("Replayed Elite archive differs from saved memory.")
    if controller.memory.current_budget != controller.max_candidates - count:
        raise RuntimeError("Saved budget differs from committed candidate count.")
    return controller, len(rounds), count


def trim_to_committed(directory: Path, completed_rounds: int, candidate_count: int) -> Path:
    archive = directory.with_name(directory.name + "_aborted_partial")
    if archive.exists():
        raise RuntimeError(f"Recovery archive already exists: {archive}")
    shutil.copytree(directory, archive)
    candidates_path = directory / "candidates.jsonl"
    candidates = candidates_path.read_text(encoding="utf-8").splitlines(keepends=True)
    candidates_path.write_text("".join(candidates[:candidate_count]), encoding="utf-8")
    events_path = directory / "adaptive" / "llm_events.jsonl"
    events = events_path.read_text(encoding="utf-8").splitlines(keepends=True)
    events_path.write_text("".join(events[:completed_rounds]), encoding="utf-8")
    trajectory = directory / "trajectory.csv"
    with trajectory.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.reader(handle))
    if len(csv_rows) - 1 < candidate_count:
        raise RuntimeError("Trajectory is shorter than completed candidates.")
    with trajectory.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(csv_rows[:candidate_count + 1])
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--arm", choices=("luna_luna", "sol_sol", "sol_luna", "luna_sol"), required=True)
    parser.add_argument("--execute", action="store_true", help="Archive the failed tail and continue the arm.")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    directory = root / args.arm
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["arms"].get(args.arm, {}).get("status") != "failed":
        raise RuntimeError("Only an arm marked failed may be recovered.")
    config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    panel = load_research_panel(config)
    controller, completed_rounds, count = restore_controller(directory, panel)
    print(json.dumps({"arm": args.arm, "completed_rounds": completed_rounds, "committed_candidates": count, "budget": controller.max_candidates, "replay_verified": True}))
    if not args.execute:
        return
    archive = trim_to_committed(directory, completed_rounds, count)
    controller.run(start_round=completed_rounds)
    summary = controller.summary()
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    repair_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    previous = manifest["arms"][args.arm]
    manifest["arms"][args.arm] = {
        "status": "complete", "started_at_utc": previous["started_at_utc"],
        "completed_at_utc": utc_now(), "artifact_dir": str(directory.relative_to(PROJECT_ROOT)),
        "config_sha256": config_digest(config), "summary": summary,
        "recovered_from_round": completed_rounds,
        "recovered_committed_candidates": count,
        "aborted_partial_archive": str(archive.relative_to(PROJECT_ROOT)),
        "original_failure": previous,
        "repair_git_commit_sha": repair_commit,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"arm": args.arm, "status": "complete", "candidates": summary["candidates_tested"], "archived_attempt": str(archive)}))


if __name__ == "__main__":
    main()
