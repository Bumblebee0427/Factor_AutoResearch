"""Post-hoc, research-only analysis for the predeclared final arm matrix."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.brains.micro import infer_mechanism
from src.factors.schema import FactorSpec
from src.research.cost_accounting import combine_usage, estimate_cost, per_denominator, role_usage
from src.selection.diagnostics import diagnose_elite_gates
from src.selection.gates import classify_tier
from src.utils.logging import ExperimentRecord


ROLE_ASSIGNMENTS = {
    "luna_luna": ("luna", "luna"),
    "sol_sol": ("sol", "sol"),
    "sol_luna": ("sol", "luna"),
    "luna_sol": ("luna", "sol"),
}
GATES = ("mean_ic", "tstat", "positive_folds", "high_cost_sharpe", "turnover")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finite_values(rows: list[dict], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(row[key])]


def quantile(rows: list[dict], key: str, fraction: float) -> float | None:
    values = finite_values(rows, key)
    return float(np.quantile(values, fraction)) if values else None


def safe_delta(before: Any, after: Any) -> float | None:
    return float(after - before) if before is not None and after is not None else None


def cohort_label(plan: dict) -> str:
    return (
        "INITIAL_COHORT"
        if plan.get("reason") == "Mandatory mechanism-coverage phase is incomplete."
        else "POST_FEEDBACK_COHORT"
    )


def load_arm(name: str, manifest: dict, root: Path) -> dict:
    entry = manifest["arms"][name]
    path = root / name
    config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
    settings = config["adaptive_research"]
    raw_records = read_jsonl(path / "candidates.jsonl")
    rounds = read_jsonl(path / "adaptive" / "research_rounds.jsonl")
    outcomes = {outcome["factor_id"]: outcome for rnd in rounds for outcome in rnd.get("outcomes", [])}
    spec_path = path / "factor_specs.jsonl"
    fixed_specs = {payload["factor_id"]: FactorSpec.from_dict(payload) for payload in read_jsonl(spec_path)}
    rows = []
    for raw in raw_records:
        record = ExperimentRecord(**raw)
        tier = classify_tier(record, settings).tier
        diagnostic = diagnose_elite_gates(record, settings).to_dict()
        outcome = outcomes.get(record.factor_id, {})
        spec = outcome.get("factor_spec") or (fixed_specs[record.factor_id].to_dict() if record.factor_id in fixed_specs else None)
        mechanism = record.mechanism or outcome.get("mechanism") or (infer_mechanism(fixed_specs[record.factor_id]) if record.factor_id in fixed_specs else "UNKNOWN")
        rows.append({
            "arm": name, "factor_id": record.factor_id, "record": raw, "spec": spec,
            "mechanism": mechanism, "tier": tier, "diagnostic": diagnostic,
            "action": outcome.get("action"),
            "cohort": cohort_label(next((rnd["plan"] for rnd in rounds if any(item["factor_id"] == record.factor_id for item in rnd.get("outcomes", []))), {})) if rounds else None,
            "mean_rank_ic": record.mean_rank_ic,
            "newey_west_tstat": record.ic_tstat,
            "positive_fold_count": record.positive_fold_count,
            "high_cost_sharpe": record.high_cost_sharpe,
            "turnover": record.turnover,
            "failed_elite_gate_count": len(diagnostic["failed_gates"]),
            "informative": bool(record.integrity_passed and len(record.fold_metrics) == 3 and all(fold.get("stock_day_observations", 0) and fold.get("mean_ic") is not None for fold in record.fold_metrics)),
        })
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if name == "fixed_deterministic":
        clusters = json.loads((path / "common_parent_clusters.json").read_text(encoding="utf-8"))["stats"]
    else:
        clusters = summary.get("parent_cluster_stats", {})
    return {"name": name, "entry": entry, "config": config, "rows": rows, "rounds": rounds, "outcomes": outcomes, "summary": summary, "clusters": clusters, "path": path}


def quality_stats(rows: list[dict]) -> dict:
    informative = [row for row in rows if row["informative"]]
    parents = [row for row in rows if row["tier"] in {"PARENT", "ELITE"}]
    size = len(rows)
    return {
        "candidate_count": size,
        "valid_informative_candidates": len(informative),
        "median_mean_rank_ic": quantile(informative, "mean_rank_ic", 0.5),
        "p75_mean_rank_ic": quantile(informative, "mean_rank_ic", 0.75),
        "median_newey_west_tstat": quantile(informative, "newey_west_tstat", 0.5),
        "p75_newey_west_tstat": quantile(informative, "newey_west_tstat", 0.75),
        "all_positive_fold_rate": sum(row["positive_fold_count"] == 3 for row in informative) / len(informative) if informative else None,
        "median_high_cost_sharpe": quantile(informative, "high_cost_sharpe", 0.5),
        "p75_high_cost_sharpe": quantile(informative, "high_cost_sharpe", 0.75),
        "median_turnover": quantile(informative, "turnover", 0.5),
        "parent_rate": len(parents) / size if size else None,
        "elite_rate": sum(row["tier"] == "ELITE" for row in rows) / size if size else None,
        "parent_count": len(parents),
        "elite_count": sum(row["tier"] == "ELITE" for row in rows),
        "median_parent_failed_elite_gates": quantile(parents, "failed_elite_gate_count", 0.5),
    }


def paired_repairs(arm: dict) -> list[dict]:
    by_id = {row["factor_id"]: row for row in arm["rows"]}
    pairs = []
    metrics = ("mean_rank_ic", "newey_west_tstat", "high_cost_sharpe", "turnover", "positive_fold_count")
    for round_record in arm["rounds"]:
        for child in round_record.get("outcomes", []):
            if child.get("action") != "IMPROVE" or len(child.get("parent_ids", [])) != 1:
                continue
            parent_id = child["parent_ids"][0]
            parent = by_id.get(parent_id)
            offspring = by_id.get(child["factor_id"])
            if parent is None or offspring is None:
                continue
            repair = child.get("repair_result", {})
            row = {
                "arm": arm["name"], "parent_factor_id": parent_id,
                "child_factor_id": offspring["factor_id"],
                "targeted_failure": repair.get("targeted_failure") or child.get("factor_spec", {}).get("targeted_failure"),
                "target_metric": repair.get("target_metric"),
                "target_improved": repair.get("target_improved"),
                "repair_succeeded": repair.get("repair_succeeded"),
                "collateral_damage": repair.get("collateral_damage"),
            }
            for metric in metrics:
                row[f"parent_{metric}"] = parent[metric]
                row[f"child_{metric}"] = offspring[metric]
                row[f"delta_{metric}"] = safe_delta(parent[metric], offspring[metric])
            pairs.append(row)
    return pairs


def role_metrics(arm: dict, pricing: dict, model_ids: dict) -> tuple[dict, dict, dict]:
    events = [rnd.get("llm", {}) for rnd in arm["rounds"]]
    macro = role_usage(events, "macro")
    micro = role_usage(events, "micro")
    combined = combine_usage(macro, micro)
    labels = ROLE_ASSIGNMENTS.get(arm["name"])
    macro_model = model_ids.get(labels[0]) if labels else None
    micro_model = model_ids.get(labels[1]) if labels else None
    macro_cost = estimate_cost(macro, macro_model, pricing)
    micro_cost = estimate_cost(micro, micro_model, pricing)
    cost = macro_cost + micro_cost if macro_cost is not None and micro_cost is not None else None
    token_row = {"arm": arm["name"], "macro_model_id": macro_model, "micro_model_id": micro_model}
    for role, usage in (("macro", macro), ("micro", micro), ("combined", combined)):
        if role != "combined":
            token_row[f"{role}_llm_calls"] = usage["llm_calls"]
        for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "cache_write_tokens"):
            token_row[f"{role}_{key}"] = usage[key]
    cost_row = {"arm": arm["name"], "macro_estimated_cost_usd": macro_cost, "micro_estimated_cost_usd": micro_cost, "total_estimated_cost_usd": cost}
    return token_row, cost_row, {"macro": macro, "micro": micro, "combined": combined, "cost": cost}


def arm_summary(arm: dict, pairs: list[dict], usage: dict) -> dict:
    rows = arm["rows"]
    basic = quality_stats(rows)
    summary = arm["summary"]
    clusters = sum(item["unique_clusters"] for item in arm["clusters"].values())
    first_parent = next((index for index, row in enumerate(rows, 1) if row["tier"] in {"PARENT", "ELITE"}), None)
    first_elite = next((index for index, row in enumerate(rows, 1) if row["tier"] == "ELITE"), None)
    actions = Counter(rnd.get("action") for rnd in arm["rounds"])
    useful_descendants = sum(row["tier"] in {"PARENT", "ELITE"} and bool(row["record"].get("parent_ids")) for row in rows)
    duplicates = summary.get("duplicate_formula_ratio", summary.get("duplicate_formula_rate"))
    invalid = summary.get("invalid_proposal_ratio", summary.get("invalid_or_noncompliant_rate"))
    total_tokens = usage["combined"]["total_tokens"]
    total_cost = usage["cost"]
    return {
        "arm": arm["name"], **basic,
        "first_parent_index": first_parent, "first_elite_index": first_elite,
        "unique_parent_clusters": clusters,
        "active_parent_representatives": summary.get("parent_pool_size"),
        "unique_parent_clusters_per_10_candidates": per_denominator(clusters, len(rows), 10),
        "valid_information_per_10_candidates": summary.get("valid_information_per_10", summary.get("effective_new_information_per_10_candidates")),
        "unique_formula_rate": per_denominator(len({row["record"]["canonical_formula"] for row in rows}), len(rows)),
        "duplicate_formula_rate": duplicates,
        "invalid_proposal_rate": invalid,
        "raw_llm_proposal_rejection_rate": summary.get("llm_invalid_proposal_ratio"),
        "mechanisms_covered": len({row["mechanism"] for row in rows if row["mechanism"] != "UNKNOWN"}),
        "mechanism_allocation": dict(Counter(row["mechanism"] for row in rows)),
        "useful_descendants": useful_descendants,
        "pivot_count": actions["PIVOT"], "improve_count": actions["IMPROVE"],
        "combine_count": actions["COMBINE"], "stop_count": actions["STOP"],
        "stop_reason": next((rnd.get("plan", {}).get("reason") for rnd in reversed(arm["rounds"]) if rnd.get("action") == "STOP"), "candidate budget exhausted" if len(rows) == 60 else None),
        "repair_attempts": len(pairs),
        "target_metric_improved_count": sum(pair.get("target_improved") is True for pair in pairs),
        "target_metric_improved_rate": per_denominator(sum(pair.get("target_improved") is True for pair in pairs), len(pairs)),
        "successful_repair_count": sum(pair.get("repair_succeeded") is True for pair in pairs),
        "successful_repair_rate": per_denominator(sum(pair.get("repair_succeeded") is True for pair in pairs), len(pairs)),
        "collateral_damage_count": sum(pair.get("collateral_damage") is True for pair in pairs),
        "valid_informative_per_100k_tokens": per_denominator(basic["valid_informative_candidates"], total_tokens, 100_000),
        "parents_per_100k_tokens": per_denominator(basic["parent_count"], total_tokens, 100_000),
        "unique_parent_clusters_per_100k_tokens": per_denominator(clusters, total_tokens, 100_000),
        "successful_repairs_per_100k_tokens": per_denominator(sum(pair.get("repair_succeeded") is True for pair in pairs), total_tokens, 100_000),
        "valid_informative_per_usd": per_denominator(basic["valid_informative_candidates"], total_cost),
        "parents_per_usd": per_denominator(basic["parent_count"], total_cost),
        "unique_parent_clusters_per_usd": per_denominator(clusters, total_cost),
        "successful_repairs_per_usd": per_denominator(sum(pair.get("repair_succeeded") is True for pair in pairs), total_cost),
        "estimated_usd_per_parent": per_denominator(total_cost or 0, basic["parent_count"]) if total_cost is not None else None,
        "estimated_usd_per_unique_parent_cluster": per_denominator(total_cost or 0, clusters) if total_cost is not None else None,
        "holdout_evaluated": False,
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def fields(rows: list[dict], minimum: list[str]) -> list[str]:
    return list(dict.fromkeys([*minimum, *(key for row in rows for key in row)]))


def lineage_case(row: dict) -> dict:
    return {
        "factor_id": row["factor_id"], "arm": row["arm"],
        "hypothesis": row["record"].get("hypothesis"),
        "canonical_formula": row["record"].get("canonical_formula"),
        "parent_ids": row["record"].get("parent_ids", []),
        "action": row["action"],
        "mean_rank_ic": row["mean_rank_ic"],
        "newey_west_tstat": row["newey_west_tstat"],
        "positive_fold_count": row["positive_fold_count"],
        "high_cost_sharpe": row["high_cost_sharpe"],
        "turnover": row["turnover"],
        "failed_elite_gates": row["diagnostic"]["failed_gates"],
        "fold_mean_ics": {
            fold.get("fold", fold.get("fold_name", f"fold_{index}")): fold.get("mean_ic")
            for index, fold in enumerate(row["record"].get("fold_metrics", []), 1)
        },
        "multi_horizon_mean_ic": row["record"].get("multi_horizon_mean_ic", {}),
        "tier": row["tier"],
    }


def choose_cases(arms: dict, pairs: list[dict]) -> dict:
    all_rows = [row for arm in arms.values() for row in arm["rows"]]
    parents = [row for row in all_rows if row["tier"] in {"PARENT", "ELITE"}]
    elites = [row for row in all_rows if row["tier"] == "ELITE"]
    successful_parent = parents[0] if parents else None
    near_elite = next((row for row in parents if row["failed_elite_gate_count"] == 1), None)
    repair_pair = pairs[0] if pairs else None
    child = next((row for row in all_rows if repair_pair and row["arm"] == repair_pair["arm"] and row["factor_id"] == repair_pair["child_factor_id"]), None)
    return {
        "elite": lineage_case(elites[0]) if elites else None,
        "successful_parent": lineage_case(successful_parent) if successful_parent else None,
        "near_elite": lineage_case(near_elite) if near_elite else None,
        "targeted_repair": {"child": lineage_case(child), "paired_metrics": repair_pair} if child else None,
    }


def analyze(manifest: dict, root: Path, pricing: dict) -> dict:
    arms = {name: load_arm(name, manifest, root) for name, entry in manifest["arms"].items() if entry.get("status") == "complete"}
    table_dir = root / "tables"
    summaries, cohorts, pairs, token_rows, cost_rows = [], [], [], [], []
    macro_rows, micro_rows, gate_rows = [], [], []
    mechanism_counter: dict[str, Counter] = {}
    for name, arm in arms.items():
        local_pairs = paired_repairs(arm)
        pairs.extend(local_pairs)
        token, cost, usage = role_metrics(arm, pricing, manifest["model_ids"])
        token_rows.append(token)
        cost_rows.append(cost)
        summary = arm_summary(arm, local_pairs, usage)
        summaries.append(summary)
        if arm["rounds"]:
            for label in ("INITIAL_COHORT", "POST_FEEDBACK_COHORT"):
                cohort = [row for row in arm["rows"] if row["cohort"] == label]
                cohorts.append({"arm": name, "cohort": label, **quality_stats(cohort),
                                "unique_parent_clusters_per_candidate": per_denominator(sum(item.get("new_cluster", False) for rnd in arm["rounds"] if cohort_label(rnd["plan"]) == label for item in rnd.get("parent_admissions", [])), len(cohort))})
        macro_rows.append({"arm": name, "pivot_count": summary["pivot_count"], "improve_count": summary["improve_count"], "combine_count": summary["combine_count"], "stop_count": summary["stop_count"], "mechanism_allocation": summary["mechanism_allocation"], "new_clusters_after_macro_decisions": sum(bool(rnd.get("new_cluster_admitted")) for rnd in arm["rounds"]), "repairable_parents_selected_for_improve": sum(rnd.get("action") == "IMPROVE" and bool(rnd.get("plan", {}).get("target_metric")) for rnd in arm["rounds"]), "saturated_mechanisms_revisited": sum(rnd.get("plan", {}).get("theme") == "saturation_pivot" for rnd in arm["rounds"]), "rejected_macro_plans": sum(len(rnd.get("llm", {}).get("macro", {}).get("rejected", [])) for rnd in arm["rounds"]), "macro_total_tokens": usage["macro"]["total_tokens"], "macro_estimated_cost_usd": cost["macro_estimated_cost_usd"]})
        micro_events = [rnd.get("llm", {}).get("micro", {}) for rnd in arm["rounds"]]
        micro_rows.append({"arm": name, "raw_factor_proposals": sum(int(event.get("raw_proposal_count", 0) or 0) for event in micro_events), "accepted_factor_proposals": sum(int(event.get("llm_candidate_count", 0) or 0) for event in micro_events), "proposal_rejection_rate": summary["raw_llm_proposal_rejection_rate"], "duplicate_formula_rate": summary["duplicate_formula_rate"], "integrity_pass_rate": per_denominator(sum(row["record"].get("integrity_passed", False) for row in arm["rows"]), len(arm["rows"])), "parent_yield": summary["parent_rate"], "elite_yield": summary["elite_rate"], "unique_cluster_yield": summary["unique_parent_clusters_per_10_candidates"], "repair_success_rate": summary["successful_repair_rate"], "micro_total_tokens": usage["micro"]["total_tokens"], "micro_estimated_cost_usd": cost["micro_estimated_cost_usd"]})
        for row in arm["rows"]:
            diagnostic = row["diagnostic"]
            failed = diagnostic["failed_gates"]
            gate_rows.append({"arm": name, "factor_id": row["factor_id"], "mechanism": row["mechanism"], "tier": row["tier"], "integrity_passed": row["record"].get("integrity_passed"), "failed_gates": list(failed), **{f"failed_{gate}": int(gate in failed) for gate in GATES}, **{f"gap_{gate}": diagnostic["distance_to_threshold"].get(gate) for gate in GATES}})
            if row["tier"] in {"PARENT", "ELITE"}:
                counter = mechanism_counter.setdefault(f"{name}|{row['mechanism']}", Counter())
                counter["parents"] += 1
                counter.update(failed)
    by_mechanism = [{"arm": key.split("|")[0], "mechanism": key.split("|")[1], "parents": count["parents"], **{f"failed_{gate}": count[gate] for gate in GATES}} for key, count in sorted(mechanism_counter.items())]
    by_name = {row["arm"]: row for row in summaries}
    overhead = []
    smoke_path = root / "smoke_checks.json"
    if smoke_path.exists():
        for item in json.loads(smoke_path.read_text(encoding="utf-8"))["checks"]:
            usage = {
                "input_tokens": item["input_tokens"],
                "output_tokens": item["output_tokens"],
                "cached_input_tokens": item["cached_input_tokens"],
                "cache_write_tokens": item["cache_write_tokens"],
            }
            overhead.append({"source": "model_smoke_check", "model": item["model"], "calls": 1,
                             "total_tokens": item["total_tokens"], "estimated_cost_usd": estimate_cost(usage, item["model"], pricing),
                             "unmeasured_attempts": 0})
    for name, entry in manifest["arms"].items():
        if not entry.get("aborted_partial_archive"):
            continue
        archive = root / Path(entry["aborted_partial_archive"]).name
        extra_events = read_jsonl(archive / "adaptive" / "llm_events.jsonl")[entry["recovered_from_round"]:]
        assignment = ROLE_ASSIGNMENTS[name]
        for role, model_key in zip(("macro", "micro"), assignment):
            usage = role_usage(extra_events, role)
            model = manifest["model_ids"][model_key]
            overhead.append({"source": f"aborted_{name}_{role}", "model": model,
                             "calls": usage["llm_calls"], "total_tokens": usage["total_tokens"],
                             "estimated_cost_usd": estimate_cost(usage, model, pricing),
                             "unmeasured_attempts": sum(
                                 bool(event.get(role, {}).get("reason"))
                                 and event[role].get("reason") not in {"deterministic_controller", "deterministic_mechanism_coverage"}
                                 and not event[role].get("response_id")
                                 for event in extra_events
                             )})
    architecture = [by_name[name] for name in ("fixed_deterministic", "adaptive_deterministic") if name in by_name]
    agent = [by_name[name] for name in ("adaptive_deterministic", "luna_luna") if name in by_name]
    model = [by_name[name] for name in ROLE_ASSIGNMENTS if name in by_name]
    tables = {
        "run_summary": summaries,
        "seed_vs_post_feedback": cohorts,
        "parent_child_deltas": pairs,
        "architecture_ablation": architecture,
        "adaptive_agent_comparison": agent,
        "model_role_comparison": model,
        "model_token_usage": token_rows,
        "model_cost_comparison": cost_rows,
        "api_overhead": overhead,
        "macro_role_metrics": macro_rows,
        "micro_role_metrics": micro_rows,
        "elite_gate_failures": gate_rows,
        "elite_gate_failures_by_mechanism": by_mechanism,
    }
    for name, rows in tables.items():
        write_csv(table_dir / f"{name}.csv", rows, fields(rows, ["arm"]))
    result = {
        "run_id": manifest["run_id"], "git_commit_sha": manifest["git_commit_sha"],
        "config_sha256": manifest["base_config_sha256"],
        "protocol_sha256": manifest["protocol_sha256"],
        "pricing_as_of": manifest["pricing_as_of"],
        "holdout_evaluated": False,
        "completed_arms": list(arms),
        "incomplete_arms": [name for name in ROLE_ASSIGNMENTS if name not in arms],
        "single_run_comparison_caveat": "Model comparisons are descriptive single-budget research runs, not statistically powered estimates of expected performance.",
        "summaries": summaries, "cohorts": cohorts, "repairs": pairs,
        "elite_gate_counts": {
            name: {
                "parent_count": sum(row["arm"] == name and row["tier"] == "PARENT" for row in gate_rows),
                **{gate: sum(row["arm"] == name and row["tier"] == "PARENT" and row[f"failed_{gate}"] for row in gate_rows) for gate in GATES},
            }
            for name in arms
        },
        "token_usage": token_rows, "costs": cost_rows,
        "api_overhead": overhead,
        "known_total_api_cost_usd": sum(row["total_estimated_cost_usd"] or 0 for row in cost_rows)
        + sum(row["estimated_cost_usd"] or 0 for row in overhead),
        "cases": choose_cases(arms, pairs),
        "table_files": {name: str((table_dir / f"{name}.csv").relative_to(root)) for name in tables},
    }
    (root / "final_experiment_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return result
