from __future__ import annotations

from types import SimpleNamespace
from dataclasses import replace

import pytest

from scripts.analyze_elite_bottlenecks import analyze
from src.brains.cross import CrossBrain
from src.brains.micro import DeterministicMicroBrain
from src.core.schemas import FactorArtifact, ResearchMemory, ResearchOutcome, ResearchPlan
from src.factors.schema import FactorSpec
from src.research.controller import AdaptiveResearchController
from src.research.failure_policy import metric_failure_types
from src.selection.diagnostics import diagnose_values
from src.utils.logging import ExperimentRecord


def record(factor_id: str = "f") -> ExperimentRecord:
    return ExperimentRecord(
        factor_id=factor_id,
        generation=0,
        parent_ids=(),
        family="price",
        hypothesis="testable",
        canonical_formula="rank(return(5))",
        integrity_passed=True,
        integrity_issues=(),
        fold_metrics=(),
        mean_rank_ic=0.004,
        ic_tstat=0.5,
        positive_fold_count=1,
        long_short_sharpe=0.1,
        high_cost_sharpe=-0.2,
        turnover=1.8,
        max_drawdown=-0.2,
        redundancy_corr=0.0,
        closest_factor_id=None,
        residual_ic=None,
        decision="HOLD",
        reasons=(),
    )


def test_diagnostic_reads_config_and_reports_multiple_gates_with_signed_gaps():
    diagnostic = diagnose_values(
        {
            "mean_rank_ic": 0.004,
            "newey_west_tstat": 0.5,
            "positive_fold_count": 1,
            "high_cost_sharpe": -0.2,
            "turnover": 1.8,
        },
        {
            "elite_min_mean_ic": 0.003,
            "elite_min_tstat": 1.0,
            "elite_min_positive_folds": 2,
            "elite_min_high_cost_sharpe": 0.0,
            "elite_max_turnover": 1.5,
        },
    )
    assert diagnostic["failed_gates"] == (
        "tstat", "positive_folds", "high_cost_sharpe", "turnover"
    )
    gaps = diagnostic["distance_to_threshold"]
    assert gaps["mean_ic"] == 0.001
    assert gaps["turnover"] == pytest.approx(-0.3)


def test_failure_policy_distinguishes_turnover_and_temporal_instability():
    value = record()
    failures = metric_failure_types(
        value,
        {"failed_gates": ["turnover", "positive_folds"]},
        {},
    )
    assert "excessive_turnover" in failures
    assert "temporal_instability" in failures


def test_cross_brain_preserves_metrics_and_emits_metric_specific_bad_lessons():
    spec = FactorSpec(
        factor_id="parent", generation=0, parent_ids=(), family="price",
        hypothesis="testable", base_feature="return", window=5, cs_operator="rank",
    )
    measured = replace(
        record("parent"),
        elite_gate_diagnostic={"failed_gates": ("turnover", "positive_folds")},
    )
    _, bad = CrossBrain().reflect(
        [ResearchOutcome(spec, measured, "PARENT", "PRICE_TREND", "IMPROVE")],
        {"redundancy_correlation": 0.9},
    )
    assert {lesson["failure_type"] for lesson in bad} == {
        "excessive_turnover", "temporal_instability"
    }
    assert all(lesson["turnover"] == measured.turnover for lesson in bad)


def test_weak_ic_repair_is_a_single_material_horizon_change():
    spec = FactorSpec(
        factor_id="parent", generation=0, parent_ids=(), family="price",
        hypothesis="testable", base_feature="return", window=5, cs_operator="rank",
    )
    parent = FactorArtifact(spec, record("parent"), "PRICE_TREND")
    plan = ResearchPlan(
        "IMPROVE", "PRICE_TREND", "PRICE_TREND", "Try one horizon shift",
        ("parent",), "Weak IC: allow one meaningful repair", 4,
        "weak_predictive_signal", "mean_rank_ic", "high_cost_sharpe",
    )
    proposals = DeterministicMicroBrain().generate(
        plan, {"parent": parent}, set(), 2
    )
    assert len(proposals) == 1
    assert proposals[0].window != spec.window
    assert proposals[0].targeted_failure == "weak_predictive_signal"


def test_mechanism_aggregation_counts_independent_parent_gate_failures():
    outcomes = [
        {
            "factor_id": "a",
            "mechanism": "PRICE_TREND",
            "decision": "PARENT",
            "integrity_passed": True,
            "metrics": {
                "mean_rank_ic": 0.004,
                "newey_west_tstat": 0.5,
                "positive_fold_count": 1,
                "high_cost_sharpe": -0.2,
                "turnover": 1.8,
            },
        }
    ]
    config = {
        "elite_min_mean_ic": 0.005,
        "elite_min_tstat": 1.0,
        "elite_min_positive_folds": 3,
        "elite_min_high_cost_sharpe": 0.0,
        "elite_max_turnover": 1.5,
    }
    summary = analyze(outcomes, {"adaptive_research": config}, "test")
    trend = summary["by_mechanism"]["PRICE_TREND"]
    assert trend["parents"] == 1
    assert trend["mean_ic"] == 1
    assert trend["turnover"] == 1


def test_progress_counters_only_reset_for_matching_evidence():
    memory = ResearchMemory(
        rounds_without_parent=4,
        rounds_without_new_cluster=4,
        rounds_without_elite=4,
        rounds_without_best_quality_improvement=4,
    )
    brain = CrossBrain()
    brain.update_memory(
        memory,
        [],
        {
            "new_parent_observed": False,
            "new_cluster_admitted": True,
            "new_elite_admitted": False,
            "best_quality_improved": False,
        },
    )
    assert memory.rounds_without_parent == 5
    assert memory.rounds_without_new_cluster == 0
    assert memory.rounds_without_elite == 5
    assert memory.rounds_without_best_quality_improvement == 5


def test_targeted_repair_logs_before_after_and_collateral_damage():
    spec = FactorSpec(
        factor_id="parent", generation=0, parent_ids=(), family="price",
        hypothesis="A price hypothesis", base_feature="return", window=5,
        cs_operator="rank",
    )
    parent_record = record("parent")
    parent_record = ExperimentRecord(
        **{**parent_record.to_dict(), "mean_rank_ic": 0.01, "turnover": 2.0}
    )
    parent = FactorArtifact(spec, parent_record, "PRICE_TREND")
    controller = object.__new__(AdaptiveResearchController)
    controller.settings = {"repair_preserve_tolerance": 0.001}
    controller.parent_pool = SimpleNamespace(items={"parent": parent})
    plan = ResearchPlan(
        "IMPROVE", "PRICE_TREND", "PRICE_TREND", "Reduce turnover", ("parent",),
        "Smooth the factor", 2, "excessive_turnover", "turnover", "mean_rank_ic",
    )
    child_record = ExperimentRecord(
        **{**parent_record.to_dict(), "factor_id": "child", "turnover": 1.0,
           "mean_rank_ic": 0.0095}
    )
    result = controller._repair_result(
        FactorSpec(
            factor_id="child", generation=1, parent_ids=("parent",), family="price",
            hypothesis="repair", base_feature="return", window=20,
            cs_operator="rank",
        ),
        child_record,
        plan,
    )
    assert result["target_before"] == 2.0
    assert result["target_after"] == 1.0
    assert result["repair_succeeded"] is True
    assert result["collateral_damage"] is False
