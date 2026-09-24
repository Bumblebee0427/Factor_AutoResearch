from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.run_final_experiments import arm_is_reusable, load_research_panel
from src.research.cost_accounting import combine_usage, estimate_cost, role_usage
from src.research.final_comparison import cohort_label, paired_repairs
from src.research.model_roles import role_llm_config
from src.selection.gates import classify_tier
from src.utils.logging import ExperimentRecord


def test_role_models_override_shared_and_fall_back(monkeypatch):
    monkeypatch.setenv("OPENAI_FACTOR_MODEL", "shared-env")
    monkeypatch.setenv("OPENAI_MACRO_MODEL", "sol-env")
    shared = {
        "model": "shared-config", "model_env": "OPENAI_FACTOR_MODEL",
        "reasoning_effort": "medium",
        "macro": {"model_env": "OPENAI_MACRO_MODEL", "reasoning_effort": "high"},
        "micro": {"model": "micro-cli", "reasoning_effort": "high"},
    }
    macro = role_llm_config(shared, "macro")
    micro = role_llm_config(shared, "micro")
    assert macro["resolved_model"] == "sol-env"
    assert micro["resolved_model"] == "micro-cli"
    assert macro["reasoning_effort"] == micro["reasoning_effort"] == "high"
    assert role_llm_config({"model": "shared-config", "model_env": "OPENAI_FACTOR_MODEL"}, "micro")["resolved_model"] == "shared-env"


def test_role_tokens_cached_tokens_and_explicit_prices():
    events = [{
        "macro": {"response_id": "m1", "usage": {"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200, "input_tokens_details": {"cached_tokens": 250}}},
        "micro": {"response_id": "u1", "usage": {"input_tokens": 400, "output_tokens": 100, "total_tokens": 500, "input_tokens_details": {"cached_tokens": 0}}},
    }]
    macro = role_usage(events, "macro")
    micro = role_usage(events, "micro")
    combined = combine_usage(macro, micro)
    assert (macro["llm_calls"], micro["llm_calls"]) == (1, 1)
    assert combined["total_tokens"] == 1700
    assert combined["cached_input_tokens"] == 250
    pricing = {"models": {"model": {"input_per_million_usd": 4, "cached_input_per_million_usd": .4, "output_per_million_usd": 20}}}
    assert estimate_cost(macro, "model", pricing) == pytest.approx((750 * 4 + 250 * .4 + 200 * 20) / 1_000_000)
    assert estimate_cost(macro, "missing", pricing) is None
    missing_cache = role_usage([{"macro": {"response_id": "m2", "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}}], "macro")
    assert missing_cache["cached_input_tokens"] is None
    assert estimate_cost(missing_cache, "model", pricing) is None
    write_events = [{"macro": {"response_id": "m3", "usage": {"input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200, "input_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 600}}}}]
    write_usage = role_usage(write_events, "macro")
    assert write_usage["cache_write_tokens"] == 600
    write_pricing = {"models": {"model": {"input_per_million_usd": 2, "cached_input_per_million_usd": .2, "cache_write_per_million_usd": 2.5, "output_per_million_usd": 10}}}
    assert estimate_cost(write_usage, "model", write_pricing) == pytest.approx((300 * 2 + 100 * .2 + 600 * 2.5 + 200 * 10) / 1_000_000)
    assert estimate_cost(missing_cache, "model", write_pricing) is None


def test_gpt6_model_ids_and_standard_price_card():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "config.yaml").read_text())
    pricing = yaml.safe_load((root / "experiment_pricing.yaml").read_text())["experiment_pricing"]
    assert config["llm"]["model"] == "gpt-6-luna"
    assert config["llm"]["macro"]["reasoning_effort"] == "high"
    assert config["llm"]["micro"]["reasoning_effort"] == "high"
    assert pricing["models"]["gpt-6-luna"] == {
        "input_per_million_usd": 0.10,
        "cached_input_per_million_usd": 0.01,
        "cache_write_per_million_usd": 0.125,
        "output_per_million_usd": 0.50,
    }
    assert pricing["models"]["gpt-6-sol"] == {
        "input_per_million_usd": 2.00,
        "cached_input_per_million_usd": 0.20,
        "cache_write_per_million_usd": 2.50,
        "output_per_million_usd": 10.00,
    }


def test_current_gate_reclassifies_legacy_record_for_reporting():
    record = ExperimentRecord(
        factor_id="old", generation=0, parent_ids=(), family="price", hypothesis="test",
        canonical_formula="rank(return(5))", integrity_passed=True, integrity_issues=(), fold_metrics=(),
        mean_rank_ic=.006, ic_tstat=1.5, positive_fold_count=3, long_short_sharpe=.1,
        high_cost_sharpe=.01, turnover=.5, max_drawdown=-.1, redundancy_corr=None,
        closest_factor_id=None, residual_ic=None, decision="RETIRE", reasons=(),
    )
    assert classify_tier(record, {
        "elite_min_mean_ic": .005, "elite_min_tstat": 1.0,
        "elite_min_positive_folds": 3, "elite_min_high_cost_sharpe": 0,
        "elite_max_turnover": 1.5,
    }).tier == "ELITE"


def test_coverage_cohort_and_parent_child_deltas():
    assert cohort_label({"reason": "Mandatory mechanism-coverage phase is incomplete."}) == "INITIAL_COHORT"
    assert cohort_label({"reason": "Repair cost gate"}) == "POST_FEEDBACK_COHORT"
    parent = {"factor_id": "p", "mean_rank_ic": .01, "newey_west_tstat": 2., "high_cost_sharpe": -.1, "turnover": 1.8, "positive_fold_count": 2}
    child = {"factor_id": "c", "mean_rank_ic": .009, "newey_west_tstat": 2.1, "high_cost_sharpe": .1, "turnover": 1.0, "positive_fold_count": 3}
    result = paired_repairs({"name": "a", "rows": [parent, child], "rounds": [{"outcomes": [{"factor_id": "c", "action": "IMPROVE", "parent_ids": ["p"], "repair_result": {"target_metric": "turnover", "target_improved": True, "collateral_damage": False, "repair_succeeded": True}}]}]})
    assert len(result) == 1
    assert result[0]["delta_turnover"] == pytest.approx(-.8)
    assert result[0]["delta_high_cost_sharpe"] == pytest.approx(.2)
    assert result[0]["repair_succeeded"] is True


def test_manifest_reuses_only_completed_arm_and_holdout_path_is_rejected():
    assert arm_is_reusable({"arms": {"x": {"status": "complete"}}}, "x")
    assert not arm_is_reusable({"arms": {"x": {"status": "failed"}}}, "x")
    with pytest.raises(RuntimeError, match="final holdout"):
        load_research_panel({"paths": {"processed_data_dir": "artifacts/processed", "research_panel_file": "holdout_2016.parquet", "holdout_panel_file": "holdout_2016.parquet"}})
