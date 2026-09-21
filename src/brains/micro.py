"""Constrained factor evolution inside the approved FactorSpec grammar."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from src.core.schemas import FactorArtifact, ResearchPlan
from src.factors.schema import FactorSpec
from src.research.generator import seed_candidates
from src.research.llm_generator import LLMFactorGenerator
from src.research.memory import ResearchState, update_state
from src.utils.logging import ExperimentRecord


def infer_mechanism(spec: FactorSpec) -> str:
    if spec.mechanism:
        return spec.mechanism
    if spec.interaction_feature:
        return "CROSS_DOMAIN_REGIME"
    if spec.expression is not None:
        features = spec.required_features
        if "news_volume" in features:
            return "NEWS_ATTENTION"
        if "volatility" in features:
            return "VOLATILITY"
        if features & {"volume_shock", "distance_to_high", "raw_volume"}:
            return "PRICE_VOLUME"
        if features & {"earnings_yield", "asset_growth"}:
            return "FUNDAMENTAL_VALUE"
        if features & {"after_tax_roe", "operating_margin", "profit_margin"}:
            return "FUNDAMENTAL_QUALITY"
        if "return" in features:
            return "PRICE_REVERSAL" if spec.direction < 0 else "PRICE_TREND"
    if spec.base_feature == "return":
        return "PRICE_REVERSAL" if spec.direction < 0 else "PRICE_TREND"
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


def adaptive_seed_candidates() -> list[FactorSpec]:
    """Broad mechanism seeds used before adaptive exploitation begins."""
    seeds = [
        candidate for candidate in seed_candidates() if not candidate.demonstration_only
    ]
    for window in (1, 5, 10):
        for operator in ("rank", "zscore"):
            seeds.append(
                FactorSpec(
                    factor_id=f"price_reversal_{window}d_{operator}",
                    generation=0,
                    parent_ids=(),
                    family="price",
                    hypothesis=(
                        "Very recent relative winners may reverse as temporary price pressure "
                        "and liquidity demand dissipate."
                    ),
                    base_feature="return",
                    window=window,
                    cs_operator=operator,
                    direction=-1,
                )
            )
    seeds.extend(
        [
            FactorSpec(
                factor_id="trend_20d_x_volume_shock_20d",
                generation=0,
                parent_ids=(),
                family="interaction",
                hypothesis="Medium-term continuation may be stronger when confirmed by unusual trading activity.",
                base_feature="return",
                window=20,
                cs_operator="winsorize_zscore",
                interaction_feature="volume_shock",
                interaction_window=20,
            ),
            FactorSpec(
                factor_id="trend_20d_x_news_attention_5d",
                generation=0,
                parent_ids=(),
                family="interaction",
                hypothesis="News attention may condition how quickly medium-term price information is incorporated.",
                base_feature="return",
                window=20,
                cs_operator="winsorize_zscore",
                interaction_feature="news_volume",
                interaction_window=5,
            ),
            FactorSpec(
                factor_id="low_vol_20d_x_roe",
                generation=0,
                parent_ids=(),
                family="interaction",
                hypothesis="Profitable low-volatility firms may combine defensive demand with durable operating quality.",
                base_feature="volatility",
                window=20,
                cs_operator="winsorize_zscore",
                interaction_feature="after_tax_roe",
                interaction_window=None,
                direction=-1,
            ),
            FactorSpec(
                factor_id="earnings_yield_x_roe",
                generation=0,
                parent_ids=(),
                family="interaction",
                hypothesis="Cheap firms with strong profitability may avoid value traps and retain valuation upside.",
                base_feature="earnings_yield",
                cs_operator="winsorize_zscore",
                interaction_feature="after_tax_roe",
                interaction_window=None,
            ),
            FactorSpec(
                factor_id="asset_growth_x_profit_margin",
                generation=0,
                parent_ids=(),
                family="interaction",
                hypothesis="Overinvestment may be less damaging when supported by strong profit margins.",
                base_feature="asset_growth",
                cs_operator="winsorize_zscore",
                interaction_feature="profit_margin",
                interaction_window=None,
                direction=-1,
            ),
            FactorSpec(
                factor_id="reversal_5d_x_volume_shock_5d",
                generation=0,
                parent_ids=(),
                family="interaction",
                hypothesis="Short-term reversal may be stronger after transitory volume and liquidity shocks.",
                base_feature="return",
                window=5,
                cs_operator="winsorize_zscore",
                interaction_feature="volume_shock",
                interaction_window=5,
                direction=-1,
            ),
        ]
    )
    for window in (10, 20, 60):
        for ts_operator in ("rolling_mean", "vol_adjust"):
            for cs_operator in ("rank", "winsorize_zscore"):
                seeds.append(
                    FactorSpec(
                        factor_id=f"expanded_trend_{window}_{ts_operator}_{cs_operator}",
                        generation=0,
                        parent_ids=(),
                        family="price",
                        hypothesis="Persistent medium-horizon returns may reflect gradual information diffusion after noise reduction.",
                        base_feature="return",
                        ts_operator=ts_operator,
                        window=window,
                        cs_operator=cs_operator,
                    )
                )
    for window in (1, 5, 10, 20):
        for ts_operator in (None, "rolling_mean"):
            for cs_operator in ("rank", "winsorize_zscore"):
                seeds.append(
                    FactorSpec(
                        factor_id=f"expanded_reversal_{window}_{ts_operator or 'identity'}_{cs_operator}",
                        generation=0,
                        parent_ids=(),
                        family="price",
                        hypothesis="Recent relative winners may reverse after temporary liquidity pressure dissipates.",
                        base_feature="return",
                        ts_operator=ts_operator,
                        window=window,
                        cs_operator=cs_operator,
                        direction=-1,
                    )
                )
    for window in (5, 10, 20, 60):
        for ts_operator in (None, "rolling_mean"):
            for cs_operator in ("rank", "winsorize_zscore"):
                seeds.append(
                    FactorSpec(
                        factor_id=f"expanded_low_vol_{window}_{ts_operator or 'identity'}_{cs_operator}",
                        generation=0,
                        parent_ids=(),
                        family="price",
                        hypothesis="Low and persistent realized volatility may attract defensive demand and reduce crash exposure.",
                        base_feature="volatility",
                        ts_operator=ts_operator,
                        window=window,
                        cs_operator=cs_operator,
                        direction=-1,
                    )
                )
    for feature, windows in {
        "volume_shock": (5, 10, 20, 60),
        "distance_to_high": (10, 20, 60),
    }.items():
        for window in windows:
            for cs_operator in ("rank", "winsorize_zscore"):
                seeds.append(
                    FactorSpec(
                        factor_id=f"expanded_{feature}_{window}_{cs_operator}",
                        generation=0,
                        parent_ids=(),
                        family="price",
                        hypothesis="Price position and unusual trading activity may reveal persistent demand or attention pressure.",
                        base_feature=feature,
                        window=window,
                        cs_operator=cs_operator,
                    )
                )
    fundamental_hypotheses = {
        "after_tax_roe": (
            "Profitable firms may outperform when durable operating quality is underpriced.",
            1,
        ),
        "operating_margin": (
            "Operating efficiency may identify firms with resilient future cash flows.",
            1,
        ),
        "profit_margin": ("Strong profit margins may reveal durable pricing power.", 1),
        "earnings_yield": (
            "High earnings yield may capture a valuation premium after robust normalization.",
            1,
        ),
        "asset_growth": (
            "Aggressive asset expansion may signal overinvestment and weaker future returns.",
            -1,
        ),
    }
    for feature, (hypothesis, direction) in fundamental_hypotheses.items():
        for cs_operator in ("rank", "zscore", "winsorize", "winsorize_zscore"):
            seeds.append(
                FactorSpec(
                    factor_id=f"expanded_{feature}_{cs_operator}",
                    generation=0,
                    parent_ids=(),
                    family="fundamental",
                    hypothesis=hypothesis,
                    base_feature=feature,
                    cs_operator=cs_operator,
                    direction=direction,
                )
            )
    for window in (1, 5, 10, 20, 60):
        for cs_operator in ("rank", "zscore", "winsorize_zscore"):
            seeds.append(
                FactorSpec(
                    factor_id=f"expanded_news_attention_{window}_{cs_operator}",
                    generation=0,
                    parent_ids=(),
                    family="news",
                    hypothesis="Persistent news attention may proxy for gradual incorporation of public information.",
                    base_feature="news_volume",
                    window=window,
                    cs_operator=cs_operator,
                )
            )
    unique: dict[str, FactorSpec] = {}
    for seed in seeds:
        unique.setdefault(seed.canonical_formula, seed)
    return list(unique.values())


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
        for number, seed in enumerate(adaptive_seed_candidates()):
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
                                direction=primary.direction * secondary.direction,
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


@dataclass(frozen=True)
class MicroProposalResult:
    candidates: tuple[FactorSpec, ...]
    used_llm: bool
    reason: str
    rejected: tuple[str, ...] = ()
    raw_proposal_count: int = 0
    llm_candidate_count: int = 0
    deterministic_fill_count: int = 0
    response_id: str | None = None
    model: str | None = None
    usage: dict = field(default_factory=dict)
    research_summary: str | None = None


class AdaptiveLLMMicroBrain:
    """Use structured LLM proposals, then fill unused budget deterministically."""

    def __init__(
        self,
        llm_config: dict,
        fallback: DeterministicMicroBrain,
        *,
        client=None,
    ) -> None:
        self.generator = LLMFactorGenerator(llm_config, client=client)
        self.fallback = fallback

    def generate(
        self,
        plan: ResearchPlan,
        parent_pool: dict[str, FactorArtifact],
        tested_formulas: set[str],
        tested_ids: set[str],
        records: list[ExperimentRecord],
        tiers: dict[str, str],
        round_id: int,
        total_budget: int,
    ) -> MicroProposalResult:
        state = update_state(
            ResearchState(),
            records,
            total_budget=total_budget,
            tested_count=len(records),
        )
        state.generation = round_id
        state.promoted_factor_ids = [
            factor_id for factor_id, tier in tiers.items() if tier == "ELITE"
        ]
        state.held_factor_ids = [
            factor_id for factor_id, tier in tiers.items() if tier == "PARENT"
        ]
        parent_specs = {
            factor_id: artifact.spec for factor_id, artifact in parent_pool.items()
        }
        parent_decisions = {
            factor_id: "PROMOTE" if tiers.get(factor_id) == "ELITE" else "HOLD"
            for factor_id in parent_pool
        }
        generated = self.generator.propose(
            generation=round_id,
            state=state,
            recent_records=records,
            parent_specs=parent_specs,
            parent_decisions=parent_decisions,
            tested_ids=tested_ids,
            research_plan=plan.to_dict(),
        )
        accepted: list[FactorSpec] = []
        rejected = list(generated.rejected)
        for spec in generated.candidates:
            reason = self._plan_mismatch(spec, plan)
            if reason:
                rejected.append(f"{spec.factor_id}: {reason}")
                continue
            accepted.append(spec)
            if len(accepted) >= plan.candidate_budget:
                break

        deterministic = self.fallback.generate(
            plan, parent_pool, tested_formulas, round_id
        )
        formulas = {spec.canonical_formula for spec in accepted}
        ids = {spec.factor_id for spec in accepted} | tested_ids
        fill: list[FactorSpec] = []
        for spec in deterministic:
            if spec.canonical_formula in formulas or spec.factor_id in ids:
                continue
            fill.append(spec)
            formulas.add(spec.canonical_formula)
            ids.add(spec.factor_id)
            if len(accepted) + len(fill) >= plan.candidate_budget:
                break
        candidates = tuple((accepted + fill)[: plan.candidate_budget])
        reason = generated.reason
        if accepted and fill:
            reason = "structured_llm_proposals_with_deterministic_fill"
        elif not accepted and fill:
            reason = f"{generated.reason}_deterministic_fallback"
        return MicroProposalResult(
            candidates,
            bool(accepted),
            reason,
            tuple(rejected),
            generated.raw_proposal_count,
            len(accepted),
            len(fill),
            generated.response_id,
            generated.model,
            generated.usage,
            generated.research_summary,
        )

    @staticmethod
    def _plan_mismatch(spec: FactorSpec, plan: ResearchPlan) -> str | None:
        if plan.action == "PIVOT":
            if spec.parent_ids:
                return "PIVOT proposal cannot claim a parent"
            if infer_mechanism(spec) != plan.mechanism:
                return "proposal mechanism does not match PIVOT plan"
            return None
        if plan.action == "IMPROVE":
            if spec.parent_ids != plan.parent_ids:
                return "IMPROVE proposal must use the planned parent"
            if infer_mechanism(spec) != plan.mechanism:
                return "IMPROVE proposal changed the planned mechanism"
            return None
        if plan.action == "COMBINE":
            if set(spec.parent_ids) != set(plan.parent_ids):
                return "COMBINE proposal must use both planned parents"
            if not spec.interaction_feature:
                return "COMBINE proposal must be an interaction"
            if infer_mechanism(spec) != "CROSS_DOMAIN_REGIME":
                return "COMBINE proposal must remain cross-domain"
            return None
        return "Micro Brain cannot generate candidates for STOP"
