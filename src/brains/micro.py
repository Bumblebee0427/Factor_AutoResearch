"""Constrained factor evolution inside the approved FactorSpec grammar."""

from __future__ import annotations

from dataclasses import replace

from src.core.schemas import FactorArtifact, ResearchPlan
from src.factors.schema import FactorSpec
from src.research.generator import seed_candidates


def infer_mechanism(spec: FactorSpec) -> str:
    if spec.interaction_feature:
        return "CROSS_DOMAIN_REGIME"
    if spec.base_feature == "return":
        return (
            "PRICE_REVERSAL"
            if spec.direction < 0 or (spec.window or 0) <= 5
            else "PRICE_TREND"
        )
    if spec.base_feature == "volatility":
        return "VOLATILITY"
    if spec.base_feature in {"volume_shock", "distance_to_high"}:
        return "PRICE_VOLUME"
    if spec.base_feature in {"earnings_yield", "asset_growth"}:
        return "FUNDAMENTAL_VALUE"
    if spec.base_feature in {"after_tax_roe", "operating_margin", "profit_margin"}:
        return "FUNDAMENTAL_QUALITY"
    if spec.base_feature == "news_volume":
        return "NEWS_ATTENTION"
    return "CROSS_DOMAIN_REGIME"


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


class DeterministicMicroBrain:
    """Mutation, crossover, and pivot without arbitrary generated code."""

    def generate(
        self,
        plan: ResearchPlan,
        parent_pool: dict[str, FactorArtifact],
        tested_formulas: set[str],
        round_id: int,
    ) -> list[FactorSpec]:
        if plan.action == "STOP" or plan.candidate_budget == 0:
            return []
        if plan.action == "PIVOT":
            proposals = self._pivot(plan, round_id)
        elif plan.action == "COMBINE":
            proposals = self._combine(plan, parent_pool, round_id)
        else:
            proposals = self._improve(plan, parent_pool, round_id)

        accepted: list[FactorSpec] = []
        local_formulas: set[str] = set()
        for proposal in proposals:
            if proposal.canonical_formula in tested_formulas | local_formulas:
                continue
            local_formulas.add(proposal.canonical_formula)
            accepted.append(proposal)
            if len(accepted) >= plan.candidate_budget:
                break
        return accepted

    def _pivot(self, plan: ResearchPlan, round_id: int) -> list[FactorSpec]:
        proposals: list[FactorSpec] = []
        for number, seed in enumerate(seed_candidates()):
            if seed.demonstration_only or infer_mechanism(seed) != plan.mechanism:
                continue
            proposals.append(
                replace(
                    seed,
                    factor_id=f"r{round_id}_pivot_{_slug(plan.mechanism)}_{number}",
                    generation=round_id,
                    proposal_type="xalpha_pivot",
                    mutation_reason=plan.reason,
                    expected_metric_effect="Establish evidence in an under-explored mechanism.",
                    falsification_condition="Retire if predictive sign or fold stability fails.",
                )
            )
        return proposals

    def _improve(
        self,
        plan: ResearchPlan,
        parent_pool: dict[str, FactorArtifact],
        round_id: int,
    ) -> list[FactorSpec]:
        if not plan.parent_ids:
            return self._pivot(replace(plan, action="PIVOT"), round_id)
        parent = parent_pool.get(plan.parent_ids[0])
        if parent is None:
            return []
        spec = parent.spec
        requires_window = spec.base_feature in {
            "return",
            "volatility",
            "volume_shock",
            "distance_to_high",
            "news_volume",
        }
        windows: list[int | None] = (
            list(dict.fromkeys([spec.window, 5, 10, 20, 60]))
            if requires_window
            else [None]
        )
        proposals: list[FactorSpec] = []
        target = (
            spec.targeted_failure or " ".join(parent.record.failure_codes)
        ).lower()
        if "turnover" in target or "cost" in target:
            windows = sorted(
                set(windows + [20, 60]), key=lambda item: item or 0, reverse=True
            )
        if not requires_window:
            ts_operators = [spec.ts_operator]
        elif spec.base_feature == "return":
            ts_operators = list(
                dict.fromkeys([spec.ts_operator, "rolling_mean", "vol_adjust"])
            )
        else:
            ts_operators = list(dict.fromkeys([spec.ts_operator, "rolling_mean"]))
        if "turnover" in target or "cost" in target:
            ts_operators = sorted(ts_operators, key=lambda item: item != "rolling_mean")
        number = 0
        for window in windows:
            for ts_operator in ts_operators:
                for cs_operator in (
                    spec.cs_operator,
                    "rank",
                    "winsorize_zscore",
                    "zscore",
                ):
                    proposals.append(
                        replace(
                            spec,
                            factor_id=f"r{round_id}_improve_{_slug(spec.factor_id)}_{number}",
                            generation=round_id,
                            parent_ids=(spec.factor_id,),
                            window=window,
                            ts_operator=ts_operator,
                            cs_operator=cs_operator,
                            proposal_type="xalpha_refinement",
                            mutation_reason=plan.reason,
                            evidence_factor_ids=(spec.factor_id,),
                            targeted_failure=spec.targeted_failure
                            or (
                                parent.record.failure_codes[0]
                                if parent.record.failure_codes
                                else None
                            ),
                            expected_metric_effect="Improve stability or implementation quality while preserving mechanism.",
                            falsification_condition="Reject if it adds no IC, stability, or cost robustness over its parent.",
                        )
                    )
                    number += 1
        if spec.base_feature == "return":
            direction = -spec.direction
            proposals.append(
                replace(
                    spec,
                    factor_id=f"r{round_id}_improve_{_slug(spec.factor_id)}_direction",
                    generation=round_id,
                    parent_ids=(spec.factor_id,),
                    hypothesis=(
                        "Short-horizon winners may reverse as temporary price pressure dissipates."
                        if direction < 0
                        else "Past winners may continue as information diffuses gradually."
                    ),
                    direction=direction,
                    proposal_type="xalpha_refinement",
                    mutation_reason="Test whether the observed horizon is reversal rather than continuation.",
                    evidence_factor_ids=(spec.factor_id,),
                    targeted_failure="directional_instability",
                )
            )
        return proposals

    def _combine(
        self,
        plan: ResearchPlan,
        parent_pool: dict[str, FactorArtifact],
        round_id: int,
    ) -> list[FactorSpec]:
        parents = [
            parent_pool[parent_id]
            for parent_id in plan.parent_ids
            if parent_id in parent_pool
        ]
        if len(parents) < 2:
            return []
        left, right = parents[:2]
        proposals: list[FactorSpec] = []
        number = 0
        for primary in (left.spec, right.spec):
            secondary = right.spec if primary is left.spec else left.spec
            primary_windows = [primary.window]
            secondary_windows = [secondary.window]
            if primary.window is not None:
                primary_windows = list(dict.fromkeys([primary.window, 5, 20, 60]))
            if secondary.window is not None:
                secondary_windows = list(dict.fromkeys([secondary.window, 5, 20, 60]))
            for primary_window in primary_windows:
                for secondary_window in secondary_windows:
                    for cs_operator in ("winsorize_zscore", "rank"):
                        proposals.append(
                            FactorSpec(
                                factor_id=f"r{round_id}_combine_{_slug(primary.factor_id)}_{_slug(secondary.factor_id)}_{number}",
                                generation=round_id,
                                parent_ids=(left.spec.factor_id, right.spec.factor_id),
                                family="interaction",
                                hypothesis=f"Conditional combination: {primary.hypothesis} Context supplied by {secondary.hypothesis}",
                                base_feature=primary.base_feature,
                                ts_operator=primary.ts_operator,
                                window=primary_window,
                                cs_operator=cs_operator,
                                interaction_feature=secondary.base_feature,
                                interaction_window=secondary_window,
                                direction=primary.direction,
                                mutation_reason=plan.reason,
                                proposal_type="xalpha_crossover",
                                evidence_factor_ids=(
                                    left.spec.factor_id,
                                    right.spec.factor_id,
                                ),
                                targeted_failure="conditional_effect",
                                expected_metric_effect="Add information that is conditional on a distinct mechanism.",
                                falsification_condition="Reject if residual IC is negligible or correlation is excessive.",
                            )
                        )
                        number += 1
        return proposals
