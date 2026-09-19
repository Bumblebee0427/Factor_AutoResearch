from __future__ import annotations

import pandas as pd
import pytest

from src.data.preprocess import split_research_holdout
from src.evaluation.integrity import check_integrity
from src.evaluation.validation import HoldoutGuard
from src.factors.schema import FactorSpec
from src.factors.transforms import cross_sectional_zscore
from src.research.generator import seed_candidates


def make_spec(**overrides) -> FactorSpec:
    payload = {
        "factor_id": "test_factor",
        "generation": 0,
        "parent_ids": (),
        "family": "price",
        "hypothesis": "A pre-specified economic mechanism.",
        "base_feature": "return",
        "window": 20,
        "cs_operator": "rank",
    }
    payload.update(overrides)
    return FactorSpec(**payload)


def test_dsl_rejects_arbitrary_primitive() -> None:
    with pytest.raises(ValueError, match="Unsupported DSL primitive"):
        make_spec(base_feature="arbitrary_python", window=None)


def test_generation_zero_is_in_recommended_range() -> None:
    production = [
        candidate for candidate in seed_candidates() if not candidate.demonstration_only
    ]
    assert 20 <= len(production) <= 40


def test_leaky_demo_is_rejected_before_backtest() -> None:
    leaky = next(
        candidate for candidate in seed_candidates() if candidate.demonstration_only
    )
    panel = pd.DataFrame(
        {"date": pd.to_datetime(["2013-01-02"]), "symbol": ["AAA"], "close": [10.0]}
    )
    report = check_integrity(
        leaky,
        panel,
        None,
        max_complexity=4,
        minimum_coverage=0.2,
    )
    assert not report.passed
    assert "future return" in " ".join(report.reasons)


def test_holdout_is_locked_until_freeze() -> None:
    guard = HoldoutGuard(2016)
    with pytest.raises(RuntimeError, match="locked"):
        guard.authorize(2016)
    guard.freeze()
    guard.authorize(2016)


def test_research_and_holdout_are_physically_separable() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2015-12-31", "2016-01-04"]),
            "symbol": ["AAA", "AAA"],
            "close": [10.0, 11.0],
        }
    )
    research, holdout = split_research_holdout(
        panel, research_end_year=2015, holdout_year=2016
    )
    assert research["date"].dt.year.max() == 2015
    assert holdout["date"].dt.year.min() == 2016


def test_cross_sectional_transform_uses_each_date_independently() -> None:
    values = pd.Series([1.0, 3.0, 100.0, 300.0])
    dates = pd.Series(
        pd.to_datetime(["2015-01-02", "2015-01-02", "2015-01-05", "2015-01-05"])
    )
    transformed = cross_sectional_zscore(values, dates)
    assert transformed.iloc[0] == pytest.approx(transformed.iloc[2])
    assert transformed.iloc[1] == pytest.approx(transformed.iloc[3])
