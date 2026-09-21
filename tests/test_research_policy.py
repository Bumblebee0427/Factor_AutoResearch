from __future__ import annotations

import numpy as np
import pandas as pd

from src.brains.cross import CrossBrain
from src.brains.macro import (
    DeterministicMacroBrain,
    LLMMacroBrain,
    LLMResearchPlanProposal,
)
from src.core.schemas import (
    DataContract,
    FactorArtifact,
    ResearchMemory,
    ResearchOutcome,
)
from src.factors.expression import Feature, Op
from src.factors.schema import FactorSpec
from src.quality.static_checks import check_factor_spec
from src.research.failure_policy import (
    dominant_integrity_response,
    valid_group_reproposals,
)
from src.research.failures import classify_failure_reasons
from src.selection.archive import ParentPool
from src.utils.logging import ExperimentRecord


def artifact(
    factor_id: str, ic: float = 0.01, mechanism: str = "PRICE_TREND"
) -> FactorArtifact:
    spec = FactorSpec(
        factor_id=factor_id,
        generation=0,
        parent_ids=(),
        family="price",
        hypothesis="A controlled test of an economic price mechanism.",
        base_feature="return",
        window=20,
        cs_operator="rank",
    )
    record = ExperimentRecord(
        factor_id=factor_id,
        generation=0,
        parent_ids=(),
        family="price",
        hypothesis=spec.hypothesis,
        canonical_formula=spec.canonical_formula,
        integrity_passed=True,
        integrity_issues=(),
        fold_metrics=(),
        mean_rank_ic=ic,
        ic_tstat=1.0,
        positive_fold_count=2,
        long_short_sharpe=0.1,
        high_cost_sharpe=0.1,
        turnover=0.2,
        max_drawdown=-0.1,
        redundancy_corr=0.0,
        closest_factor_id=None,
        residual_ic=None,
        decision="HOLD",
        reasons=(),
    )
    return FactorArtifact(spec, record, mechanism)


def test_integrity_failures_have_distinct_codes_and_next_actions() -> None:
    reasons = (
        "Unknown group: industry",
        "Unsupported transform window for rolling_mean: 7",
        "Future-noise invariance failed at 2015-02-01.",
    )
    assert classify_failure_reasons(reasons, integrity_passed=False) == (
        "unsupported_group",
        "invalid_dsl",
        "lookahead_or_leakage",
    )
    macro = DeterministicMacroBrain({"minimum_mechanisms_before_stop": 0})
    for code, action, same_mechanism in (
        ("invalid_dsl", "REPROPOSE_VALID_DSL", True),
        ("unsupported_group", "REPROPOSE_VALID_GROUP", True),
        ("lookahead_or_leakage", "PIVOT_AWAY", False),
    ):
        summary = {
            "evaluated": 2,
            "integrity_failure_total": 2,
            "integrity_failure_counts": {code: 2},
            "tier_counts": {"PARENT": 0, "ELITE": 0},
        }
        response = dominant_integrity_response(summary)
        assert response is not None and response[1]["action"] == action
        memory = ResearchMemory(
            current_theme="PRICE_TREND", recent_round_summaries=[summary]
        )
        plan = macro.plan(memory, {}, {}, 10)
        assert plan.action == "PIVOT"
        assert (plan.mechanism == "PRICE_TREND") is same_mechanism
        assert plan.parent_ids == ()

        if code == "lookahead_or_leakage":
            coverage_macro = DeterministicMacroBrain(
                {"minimum_mechanisms_before_stop": 8}
            )
            assert coverage_macro.plan(memory, {}, {}, 10).mechanism != "PRICE_TREND"

    invalid = artifact("invalid")
    bad_record = ExperimentRecord(
        **{
            **invalid.record.to_dict(),
            "integrity_passed": False,
            "integrity_issues": ("Unknown group: industry",),
            "failure_codes": ("unsupported_group",),
            "decision": "RETIRE",
        }
    )
    _, bad = CrossBrain().reflect(
        [ResearchOutcome(invalid.spec, bad_record, "RETIRED", "PRICE_TREND", "PIVOT")]
    )
    assert bad[0]["recommended_action"] == "REPROPOSE_VALID_GROUP"
    assert bad[0]["terminal"]


def test_unsupported_group_is_reproposed_with_actual_panel_groups() -> None:
    invalid = FactorSpec(
        factor_id="industry_peer_rank",
        generation=2,
        parent_ids=(),
        family="fundamental",
        hypothesis="Peer-relative profitability may identify durable quality.",
        base_feature="",
        expression=Op(
            "group_rank",
            (Feature("operating_margin"),),
            {"group": "industry"},
        ),
    )
    before = check_factor_spec(invalid, DataContract.antelion(), 4)
    assert "unsupported_group" in classify_failure_reasons(
        before.reasons, integrity_passed=False
    )

    repaired = valid_group_reproposals(
        invalid, round_id=3, allowed_groups=("sector", "subindustry")
    )
    assert len(repaired) == 2
    assert {item.expression.params["group"] for item in repaired} == {
        "sector",
        "subindustry",
    }
    assert all(item.parent_ids == () for item in repaired)
    assert all(item.targeted_failure == "unsupported_group" for item in repaired)
    assert all(
        check_factor_spec(item, DataContract.antelion(), 4).passed for item in repaired
    )


def test_parent_pool_clusters_correlated_signals_and_preserves_historical_breadth() -> (
    None
):
    dates = pd.Series(np.repeat(pd.bdate_range("2015-01-01", periods=25), 6))
    rng = np.random.default_rng(14)
    base = pd.Series(rng.normal(size=len(dates)))
    opposite = -base
    independent = pd.Series(rng.normal(size=len(dates)))
    signals: dict[str, pd.Series] = {}
    pool = ParentPool(capacity=1, correlation_threshold=0.90)

    signals["first"] = base
    assert pool.add(artifact("first", 0.01), base, signals, dates).status == "added"
    signals["weaker"] = opposite
    weaker = pool.add(artifact("weaker", 0.005), opposite, signals, dates)
    assert weaker.status == "suppressed"
    assert weaker.representative_id == "first"
    signals["stronger"] = opposite
    stronger = pool.add(artifact("stronger", 0.02), opposite, signals, dates)
    assert stronger.status == "replaced"
    assert stronger.evicted_ids == ("first",)
    signals["different"] = independent
    pool.add(artifact("different", 0.001), independent, signals, dates)
    stats = pool.cluster_stats()["PRICE_TREND"]
    assert stats["parent_candidates"] == 4
    assert stats["unique_clusters"] == 2
    assert stats["active_representatives"] == 1
    assert list(pool.items) == ["stronger"]
    signals["other_mechanism"] = base
    pool.add(
        artifact("other_mechanism", 0.03, "FUNDAMENTAL_VALUE"),
        base,
        signals,
        dates,
    )
    assert pool.cluster_stats()["FUNDAMENTAL_VALUE"]["unique_clusters"] == 1
    assert pool.cluster_stats()["PRICE_TREND"]["unique_clusters"] == 2


def test_saturated_parent_neighborhood_pivots_and_llm_cannot_override_it() -> None:
    parent = artifact("trend")
    memory = ResearchMemory(
        parent_cluster_stats={
            "PRICE_TREND": {"parent_candidates": 8, "unique_clusters": 2}
        }
    )
    macro = DeterministicMacroBrain(
        {"minimum_mechanisms_before_stop": 0, "saturation_min_parent_candidates": 6}
    )
    plan = macro.plan(memory, {"trend": parent}, {}, 10)
    assert plan.action == "PIVOT"
    assert plan.theme == "saturation_pivot"
    assert "8 qualifying parents" in plan.reason
    assert plan.mechanism != "PRICE_TREND"

    context = LLMMacroBrain({}, {})._context(
        memory, {"trend": parent}, {}, 10, {}, plan
    )
    assert context["parent_cluster_stats"]["PRICE_TREND"] == {
        "parent_candidates": 8,
        "unique_clusters": 2,
        "saturated": True,
        "interpretation": "research neighborhood is becoming saturated",
    }
    assert context["deterministic_plan"]["theme"] == "saturation_pivot"

    proposal = LLMResearchPlanProposal(
        action="IMPROVE",
        mechanism="PRICE_TREND",
        theme="more_trend",
        hypothesis_goal="Search another nearby trend formula.",
        parent_ids=["trend"],
        reason="Try another parameter.",
        candidate_budget=1,
    )
    rejection = LLMMacroBrain({}, {})._validate(
        proposal, {"trend": parent}, 10, {}, plan
    )
    assert rejection is not None and "saturated" in rejection
