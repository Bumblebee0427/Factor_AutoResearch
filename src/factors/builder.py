"""Deterministic FactorSpec execution against a sorted stock-date panel."""

from __future__ import annotations

import numpy as np
import pandas as pd

import src.factors.primitives as primitives
import src.factors.transforms as transforms
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
        if name == "earnings_yield":
            return panel["earnings_per_share"].div(close.replace(0.0, np.nan))
        if name == "asset_growth":
            return primitives.fundamental_change(
                panel["total_assets"], symbols, panel["period_ending"]
            )
        if name in {"after_tax_roe", "operating_margin", "profit_margin"}:
            return panel[name].astype(float)
        raise ValueError(f"No builder registered for primitive: {name}")

    def build(self, panel: pd.DataFrame, spec: FactorSpec) -> pd.Series:
        values = self.primitive(panel, spec.base_feature, spec.window).astype(float)
        if spec.ts_operator == "rolling_mean":
            values = primitives.rolling_mean(values, panel["symbol"], int(spec.window))
        elif spec.ts_operator == "vol_adjust":
            volatility = primitives.rolling_volatility(
                panel["close"], panel["symbol"], int(spec.window)
            ).replace(0.0, np.nan)
            values = values.div(volatility)

        if spec.interaction_feature:
            interaction = self.primitive(
                panel,
                spec.interaction_feature,
                spec.interaction_window,
            )
            values = values.mul(interaction)

        if spec.cs_operator == "rank":
            values = transforms.cross_sectional_rank(values, panel["date"])
        elif spec.cs_operator == "zscore":
            values = transforms.cross_sectional_zscore(values, panel["date"])
        elif spec.cs_operator == "winsorize":
            values = transforms.winsorize(values, panel["date"])
        elif spec.cs_operator == "winsorize_zscore":
            values = transforms.winsorize_zscore(values, panel["date"])
        elif spec.cs_operator == "sign":
            values = np.sign(values)
        return values.mul(spec.direction).rename(spec.factor_id)
