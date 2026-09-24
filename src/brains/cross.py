"""Cross-cycle reflection and reusable GOOD/BAD research memory."""

from __future__ import annotations

from src.core.schemas import ResearchMemory, ResearchOutcome
from src.research.failure_policy import (
    INTEGRITY_RESPONSES,
    METRIC_FAILURE_POLICIES,
    PRIORITY,
    metric_failure_types,
)


class CrossBrain:
    def reflect(
        self, outcomes: list[ResearchOutcome], research_config: dict | None = None
    ) -> tuple[list[dict], list[dict]]:
        good: list[dict] = []
        bad: list[dict] = []
        for outcome in outcomes:
            record = outcome.record
            evidence = {
                "factor_id": outcome.spec.factor_id,
                "mechanism": outcome.mechanism,
                "mean_rank_ic": record.mean_rank_ic,
                "ic_tstat": record.ic_tstat,
                "positive_fold_count": record.positive_fold_count,
                "high_cost_sharpe": record.high_cost_sharpe,
                "turnover": record.turnover,
                "failed_elite_gates": list(
                    record.elite_gate_diagnostic.get("failed_gates", ())
                ),
            }
            if outcome.tier in {"PARENT", "ELITE"}:
                good.append(
                    {
                        **evidence,
                        "validated_hypothesis": outcome.spec.hypothesis,
                        "reusable_principle": f"Retain the {outcome.mechanism} mechanism and vary one design choice at a time.",
                        "do_not_copy": "The exact formula is evidence, not a template to duplicate.",
                        "next_directions": [
                            "test neighboring horizon",
                            "test incremental information",
                        ],
                    }
                )
            if outcome.tier == "RETIRED":
                codes = list(record.failure_codes)
                policy = next(
                    (INTEGRITY_RESPONSES[code] for code in PRIORITY if code in codes),
                    None,
                )
                bad.append(
                    {
                        **evidence,
                        "failure_types": codes,
                        "failed_assumption": outcome.spec.hypothesis,
                        "avoidance_rule": (
                            policy["instruction"]
                            if policy
                            else f"Do not repeat the same {outcome.mechanism} formula without targeting its measured failure."
                        ),
                        "recommended_action": policy["action"]
                        if policy
                        else "REPAIR_PARENT",
                        "possible_repairs": (
                            []
                            if policy
                            else [
                                "simplify",
                                "change horizon",
                                "change cross-sectional normalization",
                            ]
                        ),
                        "terminal": bool(policy and policy["terminal"]),
                    }
                )
                if policy is None:
                    for failure_type in metric_failure_types(
                        record, record.elite_gate_diagnostic, research_config or {}
                    ):
                        metric_policy = METRIC_FAILURE_POLICIES[failure_type]
                        bad.append({
                            "type": "BAD",
                            **evidence,
                            "failure_type": failure_type,
                            "diagnosis": metric_policy["diagnosis"],
                            "repair_policy": list(metric_policy["repair_policy"]),
                            "terminal": metric_policy["terminal"],
                        })
            elif outcome.tier == "PARENT" and evidence["failed_elite_gates"]:
                failure_types = metric_failure_types(
                    record,
                    record.elite_gate_diagnostic,
                    research_config or {},
                )
                for failure_type in failure_types:
                    policy = METRIC_FAILURE_POLICIES[failure_type]
                    bad.append(
                        {
                            "type": "BAD",
                            **evidence,
                            "failure_type": failure_type,
                            "diagnosis": policy["diagnosis"],
                            "repair_policy": list(policy["repair_policy"]),
                            "terminal": policy["terminal"],
                        }
                    )
        return good, bad

    def update_memory(
        self,
        memory: ResearchMemory,
        outcomes: list[ResearchOutcome],
        round_summary: dict,
        research_config: dict | None = None,
    ) -> ResearchMemory:
        good, bad = self.reflect(outcomes, research_config)
        memory.good_lessons = (memory.good_lessons + good)[-50:]
        memory.bad_lessons = (memory.bad_lessons + bad)[-50:]
        memory.recent_experiments = (
            memory.recent_experiments + [item.to_dict() for item in outcomes]
        )[-100:]
        memory.recent_round_summaries = (
            memory.recent_round_summaries + [round_summary]
        )[-20:]
        memory.current_theme = round_summary.get("mechanism")
        memory.current_budget = int(
            round_summary.get("remaining_budget", memory.current_budget)
        )
        memory.rounds_without_parent = (
            0
            if round_summary.get("new_parent_observed", False)
            else memory.rounds_without_parent + 1
        )
        memory.rounds_without_new_cluster = (
            0
            if round_summary.get("new_cluster_admitted", False)
            else memory.rounds_without_new_cluster + 1
        )
        memory.rounds_without_elite = (
            0
            if round_summary.get("new_elite_admitted", False)
            else memory.rounds_without_elite + 1
        )
        memory.rounds_without_best_quality_improvement = (
            0
            if round_summary.get("best_quality_improved", False)
            else memory.rounds_without_best_quality_improvement + 1
        )
        memory.rounds_without_improvement = (
            memory.rounds_without_best_quality_improvement
        )
        for outcome in outcomes:
            stats = memory.mechanism_stats.setdefault(
                outcome.mechanism, {"tested": 0, "retired": 0, "parent": 0, "elite": 0}
            )
            stats["tested"] += 1
            stats[outcome.tier.lower()] += 1
            if (
                outcome.tier in {"PARENT", "ELITE"}
                and outcome.spec.factor_id not in memory.parent_pool_ids
            ):
                memory.parent_pool_ids.append(outcome.spec.factor_id)
            if (
                outcome.tier == "ELITE"
                and outcome.spec.factor_id not in memory.elite_archive_ids
            ):
                memory.elite_archive_ids.append(outcome.spec.factor_id)
        return memory
