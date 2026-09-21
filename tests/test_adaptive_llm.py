from __future__ import annotations

from types import SimpleNamespace

from src.brains.macro import (
    LLMMacroBrain,
    LLMResearchPlanProposal,
)
from src.core.schemas import FactorArtifact, ResearchMemory, ResearchPlan
from src.factors.schema import FactorSpec
from src.research.controller import AdaptiveResearchController
from src.utils.logging import ExperimentRecord


def make_artifact() -> FactorArtifact:
    spec = FactorSpec(
        factor_id="trend_parent",
        generation=0,
        parent_ids=(),
        family="price",
        hypothesis="Medium-horizon winners may continue as information diffuses.",
        base_feature="return",
        window=20,
        cs_operator="rank",
    )
    record = ExperimentRecord(
        factor_id=spec.factor_id,
        generation=0,
        parent_ids=(),
        family="price",
        hypothesis=spec.hypothesis,
        canonical_formula=spec.canonical_formula,
        integrity_passed=True,
        integrity_issues=(),
        fold_metrics=(),
        mean_rank_ic=0.01,
        ic_tstat=0.8,
        positive_fold_count=2,
        long_short_sharpe=0.2,
        high_cost_sharpe=-0.1,
        turnover=0.2,
        max_drawdown=-0.1,
        redundancy_corr=0.0,
        closest_factor_id=None,
        residual_ic=None,
        decision="HOLD",
        reasons=("IC t-stat below promotion threshold",),
        failure_codes=("low_statistical_significance",),
    )
    return FactorArtifact(spec, record, "PRICE_TREND")


class FakeResponses:
    def __init__(self, proposal: LLMResearchPlanProposal) -> None:
        self.proposal = proposal
        self.call = None

    def parse(self, **kwargs):
        self.call = kwargs
        return SimpleNamespace(
            output_parsed=self.proposal,
            id="resp_macro_fake",
            usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 50}),
        )


def test_llm_macro_plan_is_structured_and_locally_validated() -> None:
    proposal = LLMResearchPlanProposal(
        action="IMPROVE",
        mechanism="PRICE_TREND",
        theme="trend_stability",
        hypothesis_goal="Repair weak significance without changing the mechanism.",
        parent_ids=["trend_parent"],
        reason="The parent has positive IC but weak Newey-West significance.",
        candidate_budget=4,
    )
    responses = FakeResponses(proposal)
    brain = LLMMacroBrain(
        {
            "enabled": True,
            "model": "gpt-5.6-luna",
            "reasoning_mode": "standard",
            "reasoning_effort": "low",
            "verbosity": "low",
        },
        {"budgets": {"improve": 4}},
        client=SimpleNamespace(responses=responses),
    )
    artifact = make_artifact()
    fallback = ResearchPlan(
        "IMPROVE",
        "price_trend",
        "PRICE_TREND",
        "Repair parent.",
        ("trend_parent",),
        "Fallback evidence.",
        4,
    )
    result = brain.propose(
        ResearchMemory(),
        {"trend_parent": artifact},
        {},
        10,
        {},
        fallback,
    )
    assert result.used_llm
    assert result.plan == ResearchPlan(
        "IMPROVE",
        "trend_stability",
        "PRICE_TREND",
        "Repair weak significance without changing the mechanism.",
        ("trend_parent",),
        "The parent has positive IC but weak Newey-West significance.",
        4,
    )
    assert responses.call["text_format"] is LLMResearchPlanProposal
    assert responses.call["reasoning"] == {"mode": "standard", "effort": "low"}


def test_llm_macro_cannot_stop_before_deterministic_gates_allow_it() -> None:
    proposal = LLMResearchPlanProposal(
        action="STOP",
        mechanism="PRICE_TREND",
        theme="stop",
        hypothesis_goal="Stop.",
        parent_ids=[],
        reason="No further work.",
        candidate_budget=0,
    )
    responses = FakeResponses(proposal)
    brain = LLMMacroBrain(
        {"enabled": True, "model": "gpt-5.6-luna"},
        {},
        client=SimpleNamespace(responses=responses),
    )
    result = brain.propose(
        ResearchMemory(),
        {},
        {},
        10,
        {},
        ResearchPlan(
            "PIVOT",
            "volatility",
            "VOLATILITY",
            "Cover mechanism.",
            (),
            "Coverage incomplete.",
            6,
        ),
    )
    assert not result.used_llm
    assert result.plan is None
    assert "STOP is not yet permitted" in result.rejected[0]


def test_adaptive_summary_handles_an_empty_trajectory() -> None:
    controller = SimpleNamespace(
        outcomes=[],
        elite_archive=SimpleNamespace(items={}),
        parent_pool=SimpleNamespace(items={}),
        total_proposed=0,
        duplicate_count=0,
        macro=SimpleNamespace(
            coverage_status=lambda memory: {"covered": [], "tested": {}}
        ),
        memory=ResearchMemory(),
        llm_events=[],
        use_llm=False,
        llm_raw_proposals=0,
        llm_rejected_proposals=0,
    )

    summary = AdaptiveResearchController.summary(controller)

    assert summary["candidates_tested"] == 0
    assert summary["invalid_proposal_ratio"] == 0.0
    assert not summary["holdout_evaluated"]
