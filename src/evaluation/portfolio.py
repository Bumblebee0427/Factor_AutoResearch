"""Transparent cross-sectional long-short portfolio diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioSummary:
    annualized_return: float
    net_sharpe: float
    high_cost_sharpe: float
    max_drawdown: float
    turnover: float
    return_per_turnover: float
    quantile_monotonicity: float


def long_short_weights(frame: pd.DataFrame, quantile: float) -> pd.Series:
    ranks = frame.groupby("date", sort=False)["factor"].rank(pct=True)
    long_mask = ranks >= 1.0 - quantile
    short_mask = ranks <= quantile
    weights = pd.Series(0.0, index=frame.index)
    long_count = long_mask.groupby(frame["date"]).transform("sum").replace(0, np.nan)
    short_count = short_mask.groupby(frame["date"]).transform("sum").replace(0, np.nan)
    weights.loc[long_mask] = 0.5 / long_count.loc[long_mask]
    weights.loc[short_mask] = -0.5 / short_count.loc[short_mask]
    return weights.fillna(0.0)


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = wealth.div(wealth.cummax()).sub(1.0)
    return float(drawdown.min()) if not drawdown.empty else np.nan


def _sharpe(returns: pd.Series, annualization: int) -> float:
    volatility = returns.std(ddof=1)
    return (
        float(np.sqrt(annualization) * returns.mean() / volatility)
        if volatility and np.isfinite(volatility)
        else np.nan
    )


def evaluate_portfolio(
    sample: pd.DataFrame,
    *,
    quantile: float,
    quantile_count: int,
    cost_bps: float,
    high_cost_bps: float,
    annualization: int,
) -> PortfolioSummary:
    sample = sample.copy()
    sample["weight"] = long_short_weights(sample, quantile)
    sample = sample.sort_values(["symbol", "date"])
    previous = sample.groupby("symbol", sort=False)["weight"].shift().fillna(0.0)
    sample["turnover_contribution"] = (sample["weight"] - previous).abs() * 0.5
    gross = sample.groupby("date").apply(
        lambda day: float((day["weight"] * day["portfolio_return"]).sum()),
        include_groups=False,
    )
    turnover = sample.groupby("date")["turnover_contribution"].sum()
    net = gross.sub(turnover.mul(cost_bps / 10_000.0), fill_value=0.0)
    net_high = gross.sub(turnover.mul(high_cost_bps / 10_000.0), fill_value=0.0)

    sample["quantile"] = sample.groupby("date")["factor"].transform(
        lambda values: pd.qcut(
            values.rank(method="first"), quantile_count, labels=False
        )
        + 1
    )
    quantile_returns = sample.groupby("quantile")["portfolio_return"].mean()
    monotonicity = quantile_returns.corr(
        pd.Series(quantile_returns.index, index=quantile_returns.index),
        method="spearman",
    )
    mean_turnover = float(turnover.mean())
    annualized_return = float(net.mean() * annualization)
    return PortfolioSummary(
        annualized_return=annualized_return,
        net_sharpe=_sharpe(net, annualization),
        high_cost_sharpe=_sharpe(net_high, annualization),
        max_drawdown=max_drawdown(net),
        turnover=mean_turnover,
        return_per_turnover=annualized_return / mean_turnover
        if mean_turnover
        else np.nan,
        quantile_monotonicity=float(monotonicity),
    )
