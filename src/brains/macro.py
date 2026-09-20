"""Cycle-level research routing inspired by XALPHA's Macro Brain."""

from __future__ import annotations

from src.core.schemas import MECHANISMS, FactorArtifact, ResearchMemory, ResearchPlan


class DeterministicMacroBrain:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.budgets = {
            key.upper(): int(value) for key, value in config.get("budgets", {}).items()
        }

    def _budget(self, action: str, remaining: int) -> int:
        defaults = {"IMPROVE": 4, "COMBINE": 3, "PIVOT": 6, "STOP": 0}
        return min(remaining, self.budgets.get(action, defaults[action]))

    def plan(
        self,
        memory: ResearchMemory,
        parent_pool: dict[str, FactorArtifact],
        elite_archive: dict[str, FactorArtifact],
        remaining_budget: int,
        parent_correlations: dict[tuple[str, str], float] | None = None,
    ) -> ResearchPlan:
        if remaining_budget <= 0:
            return self._stop("Candidate budget exhausted.")
        min_evidence = int(self.config.get("minimum_evidence_before_stop", 10))
        stop_rounds = int(self.config.get("stop_rounds_without_improvement", 2))
        if (
            len(memory.recent_experiments) >= min_evidence
            and memory.rounds_without_improvement >= stop_rounds
        ):
            return self._stop(
                "No elite improvement within the precommitted patience window."
            )

        tested = {
            mechanism: int(memory.mechanism_stats.get(mechanism, {}).get("tested", 0))
            for mechanism in MECHANISMS
        }
        order = {mechanism: index for index, mechanism in enumerate(MECHANISMS)}
        least_tested = min(MECHANISMS, key=lambda item: (tested[item], order[item]))
        if not parent_pool:
            return ResearchPlan(
                "PIVOT",
                least_tested.lower(),
                least_tested,
                "Establish a viable parent in the least-tested economic mechanism.",
                (),
                "No eligible parent exists yet.",
                self._budget("PIVOT", remaining_budget),
            )

        current = memory.current_theme.upper() if memory.current_theme else None
        current_failures = int(
            memory.mechanism_stats.get(current or "", {}).get("retired", 0)
        )
        current_has_parent = any(
            artifact.mechanism == current for artifact in parent_pool.values()
        )
        if memory.rounds_without_improvement >= 2 or (
            current_failures >= int(self.config.get("pivot_failure_threshold", 3))
            and not current_has_parent
        ):
            return ResearchPlan(
                "PIVOT",
                least_tested.lower(),
                least_tested,
                "Move to a less explored mechanism after a stalled research round.",
                (),
                "Recent evidence did not improve the elite archive.",
                self._budget("PIVOT", remaining_budget),
            )

        ranked = sorted(
            parent_pool.values(), key=lambda item: item.quality, reverse=True
        )
        if len(ranked) >= 2:
            corr_limit = float(self.config.get("combine_max_parent_correlation", 0.50))

            def can_combine(left: FactorArtifact, right: FactorArtifact) -> bool:
                if left.mechanism == right.mechanism:
                    return False
                if parent_correlations is None:
                    return True
                key = tuple(sorted((left.spec.factor_id, right.spec.factor_id)))
                return abs(parent_correlations.get(key, 1.0)) < corr_limit

            pair = next(
                (
                    (left, right)
                    for index, left in enumerate(ranked)
                    for right in ranked[index + 1 :]
                    if can_combine(left, right)
                ),
                None,
            )
            if (
                pair
                and memory.recent_round_summaries
                and memory.recent_round_summaries[-1].get("action") != "COMBINE"
            ):
                return ResearchPlan(
                    "COMBINE",
                    "cross_mechanism",
                    "CROSS_DOMAIN_REGIME",
                    "Test whether two independently useful mechanisms have conditional incremental value.",
                    (pair[0].spec.factor_id, pair[1].spec.factor_id),
                    "Two complementary parent mechanisms are available.",
                    self._budget("COMBINE", remaining_budget),
                )

        best = ranked[0]
        return ResearchPlan(
            "IMPROVE",
            best.mechanism.lower(),
            best.mechanism,
            "Repair the strongest parent's main weakness without changing its economic mechanism.",
            (best.spec.factor_id,),
            "The current parent is promising but not yet robust enough.",
            self._budget("IMPROVE", remaining_budget),
        )

    def _stop(self, reason: str) -> ResearchPlan:
        return ResearchPlan(
            "STOP",
            "complete",
            "PRICE_TREND",
            "Stop research and freeze the validated evidence.",
            (),
            reason,
            0,
        )


class LLMMacroBrain:
    """Extension point: an LLM may return ResearchPlan, never executable code."""

    def plan(self, *args: object, **kwargs: object) -> ResearchPlan:
        raise NotImplementedError(
            "LLM Macro routing is intentionally disabled in V1; use DeterministicMacroBrain."
        )
