"""Cross-sectional rank-IC diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ICSummary:
    mean_ic: float
    ic_volatility: float
    ic_ir: float
    ic_tstat: float
    naive_ic_tstat: float
    newey_west_lags: int
    hit_rate: float
    observations: int


def daily_rank_ic(
    frame: pd.DataFrame, *, target_column: str = "future_return"
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float, name="rank_ic")

    def safe_spearman(day: pd.DataFrame) -> float:
        pair = day[["factor", target_column]].dropna()
        if (
            len(pair) < 2
            or pair["factor"].nunique() < 2
            or pair[target_column].nunique() < 2
        ):
            return np.nan
        return float(pair["factor"].corr(pair[target_column], method="spearman"))

    result = frame.groupby("date", sort=True).apply(
        safe_spearman,
        include_groups=False,
    )
    if isinstance(result, pd.DataFrame):
        return pd.Series(dtype=float, name="rank_ic")
    return result.astype(float).rename("rank_ic")


def automatic_newey_west_lags(observations: int, horizon_days: int = 1) -> int:
    """Return a conservative HAC bandwidth, respecting overlapping targets."""

    if observations < 2:
        return 0
    rule_of_thumb = int(np.floor(4 * (observations / 100.0) ** (2.0 / 9.0)))
    return min(max(rule_of_thumb, horizon_days - 1, 0), observations - 1)


def newey_west_tstat(values: pd.Series, max_lags: int) -> float:
    """Newey-West HAC t-statistic for a time-series mean using Bartlett weights."""

    clean = values.dropna().astype(float)
    observations = len(clean)
    if observations < 2:
        return np.nan
    centered = clean.to_numpy() - float(clean.mean())
    max_lags = min(max(int(max_lags), 0), observations - 1)
    long_run_variance = float(np.dot(centered, centered) / observations)
    for lag in range(1, max_lags + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / observations)
        bartlett_weight = 1.0 - lag / (max_lags + 1.0)
        long_run_variance += 2.0 * bartlett_weight * covariance
    if not np.isfinite(long_run_variance) or long_run_variance <= 0:
        return np.nan
    standard_error = np.sqrt(long_run_variance / observations)
    return float(clean.mean() / standard_error) if standard_error else np.nan


def summarize_ic(
    frame: pd.DataFrame,
    *,
    target_column: str = "future_return",
    horizon_days: int = 1,
    newey_west_lags: int | None = None,
) -> tuple[pd.Series, ICSummary]:
    daily = daily_rank_ic(frame, target_column=target_column).dropna()
    volatility = float(daily.std(ddof=1))
    mean = float(daily.mean())
    naive_tstat = (
        mean / (volatility / np.sqrt(len(daily)))
        if volatility and len(daily) > 1
        else np.nan
    )
    lags = (
        automatic_newey_west_lags(len(daily), horizon_days)
        if newey_west_lags is None
        else min(max(int(newey_west_lags), horizon_days - 1), max(len(daily) - 1, 0))
    )
    return daily, ICSummary(
        mean_ic=mean,
        ic_volatility=volatility,
        ic_ir=mean / volatility if volatility else np.nan,
        ic_tstat=newey_west_tstat(daily, lags),
        naive_ic_tstat=float(naive_tstat),
        newey_west_lags=lags,
        hit_rate=float((daily > 0).mean()),
        observations=len(daily),
    )
