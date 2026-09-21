from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.schemas import DataContract
from src.factors.builder import FactorBuilder
from src.factors.expression import (
    Constant,
    Feature,
    Op,
    expression_hash,
    expr_from_dict,
    validate_expression,
)
from src.factors.operator_registry import OPERATOR_REGISTRY
from src.factors.schema import FactorSpec
from src.quality.static_checks import check_factor_spec


def v2_spec(expression, factor_id: str = "v2") -> FactorSpec:
    return FactorSpec(
        factor_id=factor_id,
        generation=0,
        parent_ids=(),
        family="price",
        hypothesis="A typed expression tests a pre-specified mechanism.",
        base_feature="",
        expression=expression,
    )


def panel() -> pd.DataFrame:
    dates = pd.bdate_range("2015-01-01", periods=6)
    rows = []
    for symbol_index, symbol in enumerate(("AAA", "BBB")):
        for day, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": 10 + symbol_index + day,
                    "high": 11 + symbol_index + day,
                    "low": 9 + symbol_index + day,
                    "close": 10 + symbol_index + day,
                    "volume": 100 + symbol_index * 10 + day,
                    "news_count": day + symbol_index,
                    "after_tax_roe": 0.1 + symbol_index * 0.1,
                    "operating_margin": 0.2 + symbol_index * 0.1,
                    "profit_margin": 0.3 + symbol_index * 0.1,
                    "earnings_per_share": 1.0 + symbol_index,
                    "total_assets": 100.0 + day + symbol_index,
                    "period_ending": pd.Timestamp("2014-12-31"),
                    "gics_sector": "Tech" if symbol_index == 0 else "Finance",
                    "gics_sub_industry": "Software" if symbol_index == 0 else "Banks",
                }
            )
    return pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)


def test_ast_round_trip_canonicalization_and_hash() -> None:
    left = Feature("return", 20)
    right = Feature("volatility", 20)
    expression = Op("cs_rank", (Op("mul", (right, left), {}),), {})

    restored = expr_from_dict(expression.to_dict())

    assert restored.canonical() == "cs_rank(mul(return[20],volatility[20]))"
    assert restored.canonical() == expression.canonical()
    assert expression_hash(restored) == expression_hash(expression)
    assert restored.expression_hash() == expression_hash(expression)
    with pytest.raises(TypeError):
        expression.params["window"] = 5


def test_operator_registry_has_complete_metadata() -> None:
    assert len(OPERATOR_REGISTRY) >= 28
    for name, spec in OPERATOR_REGISTRY.items():
        assert name == spec.name
        assert spec.scope
        assert spec.category
        assert spec.implementation
        assert spec.complexity_cost >= 1
        assert spec.minimum_history_fn({"window": 5}) >= 1


def test_ast_builder_is_ticker_local_and_safe() -> None:
    frame = panel()
    builder = FactorBuilder()
    expression = Op(
        "rolling_mean",
        (Feature("raw_close"),),
        {"window": 3},
    )
    values = builder.evaluate_expr(frame, expression)

    assert values.iloc[2] == pytest.approx(11.0)
    assert values.iloc[8] == pytest.approx(12.0)

    safe = builder.evaluate_expr(
        frame,
        Op("safe_div", (Feature("raw_close"), Constant(0.0)), {}),
    )
    assert safe.isna().all()


def test_ast_cross_section_and_group_scope() -> None:
    frame = panel()
    builder = FactorBuilder()
    ranked = builder.evaluate_expr(
        frame,
        Op("cs_rank", (Feature("raw_close"),), {}),
    )
    assert ranked.groupby(frame["date"]).nunique().max() == 2

    neutralized = builder.evaluate_expr(
        frame,
        Op(
            "group_neutralize",
            (Feature("raw_close"),),
            {"group": "sector", "min_group_size": 1},
        ),
    )
    assert np.allclose(
        neutralized.groupby([frame["date"], frame["gics_sector"]]).mean(), 0.0
    )


def test_legacy_factor_spec_executes_through_ast_path() -> None:
    frame = panel()
    spec = FactorSpec(
        factor_id="legacy",
        generation=0,
        parent_ids=(),
        family="price",
        hypothesis="A legacy flat recipe is migrated into the AST.",
        base_feature="return",
        window=3,
        cs_operator="rank",
    )
    builder = FactorBuilder()
    direct = builder.evaluate_expr(frame, spec.effective_expression).rename("legacy")
    built = builder.build(frame, spec)

    assert np.allclose(direct.fillna(0), built.fillna(0))
    assert spec.effective_expression.canonical() == "cs_rank(return[3])"


def test_expression_static_limits_and_group_whitelist() -> None:
    expression = Op(
        "group_zscore",
        (Op("rolling_mean", (Feature("return", 20),), {"window": 20}),),
        {"group": "industry", "min_group_size": 5},
    )
    result = validate_expression(
        expression,
        allowed_features=set(DataContract.antelion().available_features),
        allowed_groups=set(DataContract.antelion().group_fields),
        max_operator_nodes=1,
    )
    assert not result.passed
    assert any("operator count" in reason for reason in result.reasons)
    assert any("Unknown group" in reason for reason in result.reasons)

    spec_result = check_factor_spec(v2_spec(expression), DataContract.antelion(), 4)
    assert not spec_result.passed
    assert any("Unsupported groups" in reason for reason in spec_result.reasons)
