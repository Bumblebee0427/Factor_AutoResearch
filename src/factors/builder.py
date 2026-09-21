"""Deterministic FactorSpec execution against a sorted stock-date panel."""

from __future__ import annotations

import numpy as np
import pandas as pd

import src.factors.primitives as primitives
import src.factors.transforms as transforms
from src.factors.expression import Constant, Feature, Op, Expr
from src.factors.operator_registry import OPERATOR_REGISTRY
from src.factors.schema import FactorSpec


class FactorBuilder:
    """Maps approved names to fixed functions; never evaluates generated code."""

    def primitive(
        self,
        panel: pd.DataFrame,
        name: str,
        window: int | None,
    ) -> pd.Series:
        close = panel["close"]
        symbols = panel["symbol"]
        if name == "return":
            return primitives.trailing_return(close, symbols, int(window))
        if name == "future_return":
            return primitives.future_return(close, symbols, int(window))
        if name == "volatility":
            return primitives.rolling_volatility(close, symbols, int(window))
        if name == "volume_shock":
            return primitives.volume_shock(panel["volume"], symbols, int(window))
        if name == "distance_to_high":
            return primitives.distance_to_high(close, symbols, int(window))
        if name == "news_volume":
            return primitives.rolling_sum(panel["news_count"], symbols, int(window))
        if name == "raw_open":
            return panel["open"].astype(float)
        if name == "raw_high":
            return panel["high"].astype(float)
        if name == "raw_low":
            return panel["low"].astype(float)
        if name == "raw_close":
            return panel["close"].astype(float)
        if name == "raw_volume":
            return panel["volume"].astype(float)
        if name == "earnings_yield":
            return panel["earnings_per_share"].div(close.replace(0.0, np.nan))
        if name == "asset_growth":
            return primitives.fundamental_change(
                panel["total_assets"], symbols, panel["period_ending"]
            )
        if name in {"after_tax_roe", "operating_margin", "profit_margin"}:
            return panel[name].astype(float)
        raise ValueError(f"No builder registered for primitive: {name}")

    def evaluate_expr(self, panel: pd.DataFrame, expression: Expr) -> pd.Series:
        if isinstance(expression, Feature):
            return self.primitive(panel, expression.name, expression.window).astype(
                float
            )
        if isinstance(expression, Constant):
            return pd.Series(float(expression.value), index=panel.index, dtype=float)
        if not isinstance(expression, Op):
            raise TypeError(f"Unsupported expression node: {type(expression)!r}")
        operator = OPERATOR_REGISTRY.get(expression.name)
        if operator is None:
            raise ValueError(f"Unregistered DSL operator: {expression.name}")
        args = tuple(self.evaluate_expr(panel, child) for child in expression.args)
        if isinstance(operator.arity, int) and len(args) != operator.arity:
            raise ValueError(
                f"Operator {expression.name} expects {operator.arity} args, got {len(args)}."
            )
        values = operator.implementation(panel, args, expression.params)
        return values.astype(float)

    def build(self, panel: pd.DataFrame, spec: FactorSpec) -> pd.Series:
        """Execute both V1 and V2 specs through the authoritative AST path."""
        return self.evaluate_expr(panel, spec.effective_expression).rename(
            spec.factor_id
        )
