"""Cross-cycle reflection and reusable GOOD/BAD research memory."""

from __future__ import annotations

from src.core.schemas import ResearchMemory, ResearchOutcome
from src.research.failure_policy import INTEGRITY_RESPONSES, PRIORITY


class CrossBrain:
    def reflect(self, outcomes: list[ResearchOutcome]) -> tuple[list[dict], list[dict]]:
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
            else:
                codes = list(record.failure_codes) or ["weak_evidence"]
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
                            else f"Do not repeat the same {outcome.mechanism} formula without targeting {codes[0]}."
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
        return good, bad

    def update_memory(
        self,
        memory: ResearchMemory,
        outcomes: list[ResearchOutcome],
        round_summary: dict,
    ) -> ResearchMemory:
        good, bad = self.reflect(outcomes)
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
        useful_count = sum(item.tier in {"PARENT", "ELITE"} for item in outcomes)
        memory.rounds_without_improvement = (
            0 if useful_count else memory.rounds_without_improvement + 1
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
