#!/usr/bin/env python3
"""Generate the final take-home tables, figures, and report from saved runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_mpl_dir = PROJECT_ROOT / "tmp" / "matplotlib"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(PROJECT_ROOT))

from src.research.final_comparison import GATES, analyze, load_arm


def cell(value: object, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}" if np.isfinite(value) else "—"
    return str(value).replace("|", "\\|")


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> list[str]:
    if not rows:
        return ["No completed arms are available for this comparison."]
    result = ["| " + " | ".join(label for _, label in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    result.extend("| " + " | ".join(cell(row.get(key)) for key, _ in columns) + " |" for row in rows)
    return result


def write_figures(result: dict, root: Path) -> list[str]:
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    names = []
    summaries = {row["arm"]: row for row in result["summaries"]}

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(figures / name, dpi=180, bbox_inches="tight")
        plt.close()
        names.append(str((figures / name).relative_to(root)))

    cohorts = [row for row in result["cohorts"] if row["arm"] == "adaptive_deterministic" and row["candidate_count"]]
    if len(cohorts) == 2:
        plt.figure(figsize=(6, 4))
        x = np.arange(2)
        plt.bar(x - 0.18, [row["median_mean_rank_ic"] or 0 for row in cohorts], 0.36, label="Median RankIC")
        plt.bar(x + 0.18, [row["p75_mean_rank_ic"] or 0 for row in cohorts], 0.36, label="75th percentile")
        plt.xticks(x, ["Initial", "Post-feedback"])
        plt.ylabel("Mean RankIC")
        plt.title("Figure 1 · Feedback cohorts (deterministic adaptive)")
        plt.legend()
        save("figure1_feedback_cohorts.png")

    architecture = [summaries[name] for name in ("fixed_deterministic", "adaptive_deterministic") if name in summaries]
    if len(architecture) == 2:
        plt.figure(figsize=(6, 4))
        plt.bar(["Fixed generation", "Macro/Micro/Cross"], [row["unique_parent_clusters"] for row in architecture], color=["#7389ab", "#2a7498"])
        plt.ylabel("Unique Parent clusters / 60 candidates")
        plt.title("Figure 2 · Architecture comparison")
        save("figure2_architecture.png")

    agent = [summaries[name] for name in ("adaptive_deterministic", "luna_luna") if name in summaries]
    if len(agent) == 2:
        plt.figure(figsize=(6, 4))
        plt.bar(["Deterministic", "Luna High"], [row["unique_parent_clusters"] for row in agent], color=["#7389ab", "#2a7498"])
        plt.ylabel("Unique Parent clusters / 60 candidates")
        plt.title("Figure 3 · Adaptive agent comparison")
        save("figure3_agent_comparison.png")

    model = [row for row in result["summaries"] if row["arm"] in {"luna_luna", "sol_sol", "sol_luna", "luna_sol"}]
    costs = {row["arm"]: row["total_estimated_cost_usd"] for row in result["costs"]}
    if len(model) == 4 and all(costs.get(row["arm"]) is not None for row in model):
        plt.figure(figsize=(6, 4))
        for row in model:
            plt.scatter(costs[row["arm"]], row["unique_parent_clusters"], s=75)
            plt.annotate(row["arm"], (costs[row["arm"]], row["unique_parent_clusters"]), xytext=(5, 4), textcoords="offset points")
        plt.xlabel("Estimated API cost (USD)")
        plt.ylabel("Unique Parent clusters")
        plt.title("Figure 4 · Model role cost and diversity")
        save("figure4_model_roles.png")

    gate_rows = []
    target = "adaptive_deterministic" if "adaptive_deterministic" in summaries else next(iter(summaries), None)
    if target:
        arm = load_arm(target, {"arms": {target: {"status": "complete"}}}, root)
        gate_rows = [row for row in arm["rows"] if row["tier"] in {"PARENT", "ELITE"}]
    if gate_rows:
        counts = [sum(gate in row["diagnostic"]["failed_gates"] for row in gate_rows) for gate in GATES]
        plt.figure(figsize=(7, 4))
        plt.bar(["Mean IC", "NW t-stat", "Positive folds", "High-cost Sharpe", "Turnover"], counts, color="#b96954")
        plt.ylabel("Parent candidates failing gate")
        plt.title(f"Figure 5 · Elite bottlenecks ({target})")
        plt.xticks(rotation=25, ha="right")
        save("figure5_elite_gates.png")
    return names


def render_report(result: dict, manifest: dict, figure_names: list[str], root: Path) -> str:
    by_name = {row["arm"]: row for row in result["summaries"]}
    lines = [
        "# Final factor research experiments", "",
        f"Run: `{result['run_id']}` · Git commit: `{result['git_commit_sha']}` · Config SHA-256: `{result['config_sha256']}`", "",
        f"Protocol SHA-256: `{result['protocol_sha256']}` · Seed: `{manifest['random_seed']}` · Budget: `{manifest['candidate_budget']}` per arm", "",
        "Research: 2010–2015; validation folds: 2013, 2014, 2015. Holdout evaluated: **false**.", "",
        "Model comparisons are descriptive single-budget research runs, not statistically powered estimates of expected performance.", "",
        "## Run status", "",
    ]
    lines += markdown_table([{"arm": name, "status": item["status"], "models": "/".join(ROLE_NAMES.get(name, ("—", "—")))} for name, item in manifest["arms"].items()], [("arm", "Arm"), ("status", "Status"), ("models", "Macro / Micro")])
    lines += ["", "## A · Initial vs post-feedback", ""]
    lines += markdown_table(result["cohorts"], [("arm", "Arm"), ("cohort", "Cohort"), ("candidate_count", "N"), ("median_mean_rank_ic", "Median IC"), ("p75_mean_rank_ic", "P75 IC"), ("median_newey_west_tstat", "Median NW t"), ("all_positive_fold_rate", "All-positive folds"), ("median_high_cost_sharpe", "Median high-cost Sharpe"), ("parent_rate", "Parent rate"), ("elite_rate", "Elite rate")])
    lines += ["", "## B · Architecture", ""]
    lines += markdown_table([by_name[name] for name in ("fixed_deterministic", "adaptive_deterministic") if name in by_name], [("arm", "Arm"), ("candidate_count", "N"), ("median_mean_rank_ic", "Median IC"), ("median_high_cost_sharpe", "Median high-cost Sharpe"), ("parent_count", "Common Parents"), ("elite_count", "Common Elites"), ("unique_parent_clusters", "Unique clusters"), ("valid_information_per_10_candidates", "Valid / 10")])
    lines += ["", "The fixed-generation arm is classified post hoc using the current Parent/Elite gates; its original search decisions are preserved.", "", "## C · Deterministic vs Luna High", ""]
    lines += markdown_table([by_name[name] for name in ("adaptive_deterministic", "luna_luna") if name in by_name], [("arm", "Arm"), ("candidate_count", "N"), ("first_parent_index", "First Parent"), ("first_elite_index", "First Elite"), ("unique_parent_clusters", "Unique clusters"), ("median_mean_rank_ic", "Median IC"), ("median_high_cost_sharpe", "Median high-cost Sharpe"), ("raw_llm_proposal_rejection_rate", "Raw proposal rejection")])
    lines += ["", "## D · Model roles, tokens and cost", ""]
    model_summaries = [by_name[name] for name in ("luna_luna", "sol_sol", "sol_luna", "luna_sol") if name in by_name]
    cost_map = {row["arm"]: row for row in result["costs"]}
    usage_map = {row["arm"]: row for row in result["token_usage"]}
    comparison = [{**row, **cost_map[row["arm"]], **usage_map[row["arm"]]} for row in model_summaries]
    lines += markdown_table(comparison, [("arm", "Arm"), ("macro_model_id", "Macro model"), ("micro_model_id", "Micro model"), ("combined_total_tokens", "Total tokens"), ("total_estimated_cost_usd", "Est. USD"), ("unique_parent_clusters", "Unique clusters"), ("unique_parent_clusters_per_100k_tokens", "Clusters / 100k tokens"), ("median_mean_rank_ic", "Median IC")])
    lines += ["", f"Pricing as of {result['pricing_as_of']} from [Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna) and [Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol) model pages. Costs are estimates from recorded token usage, not invoice amounts.", "", "## Repair evidence", ""]
    lines += markdown_table(result["repairs"], [("arm", "Arm"), ("parent_factor_id", "Parent"), ("child_factor_id", "Child"), ("targeted_failure", "Target"), ("delta_mean_rank_ic", "Δ IC"), ("delta_high_cost_sharpe", "Δ high-cost Sharpe"), ("target_improved", "Target improved"), ("collateral_damage", "Collateral damage"), ("repair_succeeded", "Repair succeeded")])
    lines += ["", "## Representative lineages", ""]
    for label, item in result["cases"].items():
        if item is None:
            lines.extend([f"### {label.replace('_', ' ').title()}", "", "No qualifying case in completed arms.", ""])
            continue
        case = item.get("child", item)
        lines += [f"### {label.replace('_', ' ').title()}", "", f"Factor: `{case['factor_id']}` ({case['arm']}); tier: {case['tier']}; action: {cell(case['action'])}.", "", f"Hypothesis: {cell(case['hypothesis'])}", "", f"Formula: `{case['canonical_formula']}`; parents: {', '.join(case['parent_ids']) or 'none'}.", "", f"IC {cell(case['mean_rank_ic'])}; NW t {cell(case['newey_west_tstat'])}; positive folds {cell(case['positive_fold_count'])}; high-cost Sharpe {cell(case['high_cost_sharpe'])}; turnover {cell(case['turnover'])}; failed gates {', '.join(case['failed_elite_gates']) or 'none'}.", ""]
        if item.get("paired_metrics"):
            pair = item["paired_metrics"]
            lines += [f"Repair target: {cell(pair['targeted_failure'])}; target improved: {cell(pair['target_improved'])}; collateral damage: {cell(pair['collateral_damage'])}; success: {cell(pair['repair_succeeded'])}.", ""]
    lines += ["## Figures", ""]
    lines.extend(f"- `{path}`" for path in figure_names)
    lines += ["", "## Audit trail", "", f"Manifest: `{root / 'manifest.json'}`", f"Tables: `{root / 'tables'}`", f"Machine summary: `{root / 'final_experiment_summary.json'}`", "", "The 2016 holdout remained unopened. An empty Elite archive does not create a final library.", ""]
    return "\n".join(lines)


ROLE_NAMES = {"luna_luna": ("Luna High", "Luna High"), "sol_sol": ("Sol High", "Sol High"), "sol_luna": ("Sol High", "Luna High"), "luna_sol": ("Luna High", "Sol High")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--pricing", type=Path, default=PROJECT_ROOT / "experiment_pricing.yaml")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "docs" / "final_experiment_results.md")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    pricing = manifest.get("pricing") or yaml.safe_load(args.pricing.read_text(encoding="utf-8"))["experiment_pricing"]
    result = analyze(manifest, root, pricing)
    figures = write_figures(result, root)
    args.report.write_text(render_report(result, manifest, figures, root), encoding="utf-8")
    print(json.dumps({"completed_arms": result["completed_arms"], "report": str(args.report), "figures": figures}, indent=2))


if __name__ == "__main__":
    main()
