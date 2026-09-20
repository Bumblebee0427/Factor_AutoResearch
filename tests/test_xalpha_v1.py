from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.brains.macro import DeterministicMacroBrain
from src.core.schemas import FactorArtifact, ResearchMemory
from src.factors.primitives import future_return
from src.factors.schema import FactorSpec
from src.memory.store import ResearchMemoryStore
from src.quality.dynamic_leakage import check_dynamic_leakage
from src.quality.static_checks import check_expression, check_factor_spec
from src.research.controller import AdaptiveResearchController
from src.selection.library import build_final_library
from src.utils.logging import ExperimentRecord


def make_spec(factor_id: str = "momentum", **overrides) -> FactorSpec:
    payload = {
        "factor_id": factor_id,
        "generation": 0,
        "parent_ids": (),
        "family": "price",
        "hypothesis": "Medium-horizon winners may continue as information diffuses.",
        "base_feature": "return",
        "window": 20,
        "cs_operator": "rank",
    }
    payload.update(overrides)
    return FactorSpec(**payload)


def make_record(factor_id: str, **overrides) -> ExperimentRecord:
    payload = {
        "factor_id": factor_id,
        "generation": 0,
        "parent_ids": (),
        "family": "price",
        "hypothesis": "A testable mechanism.",
        "canonical_formula": "+1 * rank(return(20))",
        "integrity_passed": True,
        "integrity_issues": (),
        "fold_metrics": (),
        "mean_rank_ic": 0.02,
        "ic_tstat": 2.0,
        "positive_fold_count": 3,
        "long_short_sharpe": 0.5,
        "high_cost_sharpe": 0.2,
        "turnover": 0.3,
        "max_drawdown": -0.1,
        "redundancy_corr": 0.0,
        "closest_factor_id": None,
        "residual_ic": None,
        "decision": "PROMOTE",
        "reasons": ("passed",),
    }
    payload.update(overrides)
    return ExperimentRecord(**payload)


def leakage_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2015-01-01", periods=30)
    rows = []
    for symbol_number, symbol in enumerate(("AAA", "BBB", "CCC", "DDD")):
        for day_number, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "close": 10
                    + symbol_number
                    + day_number * (0.1 + 0.01 * symbol_number),
                }
            )
    return pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)


def test_static_checks_reject_negative_shift() -> None:
    result = check_expression("close.shift(-5)")
    assert not result.passed
    assert "future" in result.reasons[0]


def test_static_checks_reject_centered_rolling() -> None:
    result = check_expression("close.rolling(20, center=True).mean()")
    assert not result.passed
    assert "centered" in result.reasons[0]


def test_data_contract_rejects_future_return() -> None:
    from src.core.schemas import DataContract

    spec = make_spec(
        "leak", base_feature="future_return", window=5, demonstration_only=True
    )
    result = check_factor_spec(spec, DataContract.antelion(), 4)
    assert not result.passed
    assert any("future" in reason.lower() for reason in result.reasons)


def test_truncation_test_catches_leaking_builder() -> None:
    panel = leakage_panel()
    spec = make_spec()
    result = check_dynamic_leakage(
        panel,
        spec,
        build_fn=lambda frame: future_return(frame["close"], frame["symbol"], 3),
        cutoffs=2,
    )
    assert not result.truncation_passed


def test_future_noise_test_catches_leaking_builder() -> None:
    panel = leakage_panel()
    spec = make_spec()
    result = check_dynamic_leakage(
        panel,
        spec,
        build_fn=lambda frame: future_return(frame["close"], frame["symbol"], 3),
        cutoffs=2,
    )
    assert not result.future_noise_passed


def test_adaptive_controller_refuses_holdout_rows() -> None:
    panel = pd.DataFrame(
        {"date": pd.to_datetime(["2016-01-04"]), "symbol": ["AAA"], "close": [10.0]}
    )
    with pytest.raises(RuntimeError, match="holdout"):
        AdaptiveResearchController({"walk_forward": {"holdout_year": 2016}}, panel)


def test_sealed_memory_cannot_be_updated(tmp_path) -> None:
    store = ResearchMemoryStore(tmp_path)
    store.save(ResearchMemory())
    store.seal()
    with pytest.raises(RuntimeError, match="sealed"):
        store.save(ResearchMemory())
    with pytest.raises(RuntimeError, match="sealed"):
        ResearchMemoryStore(tmp_path).append_round({"should": "fail across processes"})


def test_library_prunes_correlation_and_zero_residual_ic() -> None:
    dates = pd.bdate_range("2015-01-01", periods=20)
    symbols = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
    rows = []
    signal_values = []
    for day, date in enumerate(dates):
        for symbol_number, symbol in enumerate(symbols):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "close": 10 + symbol_number + day * (0.01 + symbol_number * 0.002),
                }
            )
            signal_values.append(float(symbol_number))
    panel = pd.DataFrame(rows)
    first = make_spec("first")
    duplicate = make_spec("duplicate", window=60)
    artifacts = [
        FactorArtifact(first, make_record("first", mean_rank_ic=0.03), "PRICE_TREND"),
        FactorArtifact(
            duplicate, make_record("duplicate", mean_rank_ic=0.02), "PRICE_TREND"
        ),
    ]
    signal = pd.Series(signal_values, index=panel.index, dtype=float)
    selected, audit = build_final_library(
        artifacts,
        {"first": signal, "duplicate": signal.copy()},
        panel,
        {
            "final_correlation_threshold": 0.7,
            "min_incremental_ic": 0.001,
            "max_final_factors": 8,
            "prediction_horizon_days": 5,
        },
    )
    assert [item.spec.factor_id for item in selected] == ["first"]
    rejected = next(item for item in audit if item["factor_id"] == "duplicate")
    assert rejected["reason"] == "redundant"
    assert (
        not np.isfinite(rejected["residual_ic"]) or abs(rejected["residual_ic"]) < 1e-12
    )


def test_macro_brain_supports_all_four_actions() -> None:
    macro = DeterministicMacroBrain(
        {"minimum_evidence_before_stop": 10, "stop_rounds_without_improvement": 2}
    )
    empty = ResearchMemory()
    assert macro.plan(empty, {}, {}, 10).action == "PIVOT"

    trend_spec = make_spec("trend")
    trend = FactorArtifact(trend_spec, make_record("trend"), "PRICE_TREND")
    assert macro.plan(ResearchMemory(), {"trend": trend}, {}, 10).action == "IMPROVE"

    value_spec = make_spec(
        "value",
        family="fundamental",
        hypothesis="Cheap firms may mean revert to fair value.",
        base_feature="earnings_yield",
        window=None,
        cs_operator="winsorize_zscore",
    )
    value = FactorArtifact(
        value_spec, make_record("value", family="fundamental"), "FUNDAMENTAL_VALUE"
    )
    combine_memory = ResearchMemory(recent_round_summaries=[{"action": "IMPROVE"}])
    assert (
        macro.plan(combine_memory, {"trend": trend, "value": value}, {}, 10).action
        == "COMBINE"
    )
    high_corr = {tuple(sorted(("trend", "value"))): 0.95}
    assert (
        macro.plan(
            combine_memory,
            {"trend": trend, "value": value},
            {},
            10,
            high_corr,
        ).action
        == "IMPROVE"
    )
    assert macro.plan(empty, {}, {}, 0).action == "STOP"
