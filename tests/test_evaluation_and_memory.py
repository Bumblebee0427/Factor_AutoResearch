from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.evaluator import evaluate_walk_forward
from src.evaluation.ic import newey_west_tstat, summarize_ic
from src.evaluation.validation import WalkForwardFold
from src.research.failures import classify_failure_reasons
from src.research.memory import ResearchState, update_state
from src.utils.logging import ExperimentRecord


def make_record(**overrides) -> ExperimentRecord:
    payload = {
        "factor_id": "factor_a",
        "generation": 0,
        "parent_ids": (),
        "family": "price",
        "hypothesis": "A sufficiently detailed ex-ante economic mechanism.",
        "canonical_formula": "+1 * rank(return(20))",
        "integrity_passed": True,
        "integrity_issues": (),
        "fold_metrics": (),
        "mean_rank_ic": 0.01,
        "ic_tstat": 1.2,
        "positive_fold_count": 2,
        "long_short_sharpe": 0.3,
        "high_cost_sharpe": -0.1,
        "turnover": 0.5,
        "max_drawdown": -0.1,
        "redundancy_corr": 0.0,
        "closest_factor_id": None,
        "residual_ic": None,
        "decision": "HOLD",
        "reasons": ("signal does not survive high-cost scenario",),
        "failure_codes": ("cost_sensitivity",),
    }
    payload.update(overrides)
    return ExperimentRecord(**payload)


def test_newey_west_tstat_penalizes_positive_serial_correlation() -> None:
    values = pd.Series(0.02 + np.repeat([0.05, -0.03], 50))
    naive = values.mean() / (values.std(ddof=1) / np.sqrt(len(values)))
    adjusted = newey_west_tstat(values, max_lags=10)

    assert np.isfinite(adjusted)
    assert abs(adjusted) < abs(naive)


def test_empty_ic_sample_returns_nan_summary() -> None:
    daily, summary = summarize_ic(
        pd.DataFrame(columns=["date", "factor", "future_return"]),
        horizon_days=5,
    )

    assert daily.empty
    assert np.isnan(summary.mean_ic)
    assert np.isnan(summary.ic_tstat)
    assert summary.observations == 0


def test_walk_forward_reports_multi_horizon_ic() -> None:
    dates = pd.bdate_range("2013-01-02", periods=40)
    rows = []
    for symbol_index, symbol in enumerate(("AAA", "BBB", "CCC", "DDD", "EEE")):
        for day_index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "close": 20.0
                    + symbol_index
                    + day_index * (0.01 + symbol_index * 0.003),
                }
            )
    panel = pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)
    factor = panel.groupby("date")["close"].rank(pct=True)
    fold = WalkForwardFold(
        name="validate_2013",
        train_start="2010-01-01",
        train_end="2012-12-31",
        validation_start="2013-01-01",
        validation_end="2013-12-31",
    )
    config = {
        "prediction_horizon_days": 5,
        "ic_horizons_days": [1, 5, 10],
        "newey_west_lags": "auto",
        "portfolio_horizon_days": 1,
        "portfolio_quantile": 0.2,
        "quantile_count": 5,
        "transaction_cost_bps": 10,
        "high_cost_bps": 25,
        "annualization_days": 252,
    }

    result = evaluate_walk_forward(panel, factor, [fold], config)

    assert set(result.folds[0].multi_horizon_ic) == {"1d", "5d", "10d"}
    assert result.folds[0].newey_west_lags >= 4
    assert set(result.multi_horizon_mean_ic) == {"1d", "5d", "10d"}


def test_failure_taxonomy_and_family_state_are_structured() -> None:
    reasons = (
        "signal does not survive high-cost scenario",
        "turnover exceeds implementation threshold",
    )
    codes = classify_failure_reasons(reasons, integrity_passed=True)
    record = make_record(reasons=reasons, failure_codes=codes)

    state = update_state(ResearchState(), [record], total_budget=10, tested_count=1)
    family = state.family_summary_dict()["price"]

    assert codes == ("cost_sensitivity", "excessive_turnover")
    assert state.failure_taxonomy["cost_sensitivity"] == 1
    assert family["decision_counts"]["HOLD"] == 1
    assert family["mean_rank_ic"] == 0.01
    assert family["failure_counts"]["excessive_turnover"] == 1
