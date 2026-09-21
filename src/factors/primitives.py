"""Historical time-series primitives applied within each asset."""

from __future__ import annotations

import pandas as pd
import numpy as np


def trailing_return(close: pd.Series, symbols: pd.Series, window: int) -> pd.Series:
    return close.groupby(symbols, sort=False).pct_change(window, fill_method=None)


def future_return(close: pd.Series, symbols: pd.Series, horizon: int) -> pd.Series:
    return close.groupby(symbols, sort=False).shift(-horizon).div(close).sub(1.0)


def rolling_volatility(close: pd.Series, symbols: pd.Series, window: int) -> pd.Series:
    daily = trailing_return(close, symbols, 1)
    return (
        daily.groupby(symbols, sort=False)
        .rolling(window, min_periods=window)
        .std()
        .reset_index(level=0, drop=True)
    )


def volume_shock(volume: pd.Series, symbols: pd.Series, window: int) -> pd.Series:
    baseline = (
        volume.groupby(symbols, sort=False)
        .rolling(window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return volume.div(baseline).sub(1.0)


def distance_to_high(close: pd.Series, symbols: pd.Series, window: int) -> pd.Series:
    rolling_high = (
        close.groupby(symbols, sort=False)
        .rolling(window, min_periods=window)
        .max()
        .reset_index(level=0, drop=True)
    )
    return close.div(rolling_high).sub(1.0)


def rolling_sum(values: pd.Series, symbols: pd.Series, window: int) -> pd.Series:
    return (
        values.groupby(symbols, sort=False)
        .rolling(window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )


def rolling_mean(values: pd.Series, symbols: pd.Series, window: int) -> pd.Series:
    return (
        values.groupby(symbols, sort=False)
        .rolling(window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )


def rolling_sum_strict(values: pd.Series, symbols: pd.Series, window: int) -> pd.Series:
    return (
        values.groupby(symbols, sort=False)
        .rolling(window, min_periods=window)
        .sum()
        .reset_index(level=0, drop=True)
        .reindex(values.index)
    )


def rolling_std(values: pd.Series, symbols: pd.Series, window: int) -> pd.Series:
    return (
        values.groupby(symbols, sort=False)
        .rolling(window, min_periods=window)
        .std()
        .reset_index(level=0, drop=True)
        .reindex(values.index)
    )


def rolling_min(values: pd.Series, symbols: pd.Series, window: int) -> pd.Series:
    return (
        values.groupby(symbols, sort=False)
        .rolling(window, min_periods=window)
        .min()
        .reset_index(level=0, drop=True)
        .reindex(values.index)
    )


def rolling_max(values: pd.Series, symbols: pd.Series, window: int) -> pd.Series:
    return (
        values.groupby(symbols, sort=False)
        .rolling(window, min_periods=window)
        .max()
        .reset_index(level=0, drop=True)
        .reindex(values.index)
    )


def ts_rank(values: pd.Series, symbols: pd.Series, window: int) -> pd.Series:
    """Rank each observation against its own trailing ticker history in [0, 1]."""
    result = (
        values.groupby(symbols, sort=False)
        .rolling(window, min_periods=window)
        .apply(lambda sample: sample.rank(method="average", pct=True).iloc[-1])
        .reset_index(level=0, drop=True)
    )
    return result.reindex(values.index)


def ts_zscore(values: pd.Series, symbols: pd.Series, window: int) -> pd.Series:
    mean = rolling_mean(values, symbols, window)
    std = rolling_std(values, symbols, window).replace(0.0, np.nan)
    return values.sub(mean).div(std)


def ewma(values: pd.Series, symbols: pd.Series, halflife: float) -> pd.Series:
    result = values.groupby(symbols, sort=False, group_keys=False).apply(
        lambda series: series.ewm(halflife=halflife, adjust=False, min_periods=1).mean()
    )
    return result.reindex(values.index)


def decay_linear(values: pd.Series, symbols: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1.0, float(window) + 1.0)
    weights /= weights.sum()
    result = (
        values.groupby(symbols, sort=False)
        .rolling(window, min_periods=window)
        .apply(lambda sample: float(np.dot(sample.to_numpy(), weights)), raw=False)
        .reset_index(level=0, drop=True)
    )
    return result.reindex(values.index)


def rolling_corr(
    left: pd.Series,
    right: pd.Series,
    symbols: pd.Series,
    window: int,
) -> pd.Series:
    result = pd.Series(np.nan, index=left.index, dtype=float)
    frame = pd.DataFrame({"left": left, "right": right, "symbol": symbols})
    for _, group in frame.groupby("symbol", sort=False):
        result.loc[group.index] = (
            group["left"]
            .rolling(window, min_periods=window)
            .corr(group["right"])
            .to_numpy()
        )
    return result


def rolling_cov(
    left: pd.Series,
    right: pd.Series,
    symbols: pd.Series,
    window: int,
) -> pd.Series:
    result = pd.Series(np.nan, index=left.index, dtype=float)
    frame = pd.DataFrame({"left": left, "right": right, "symbol": symbols})
    for _, group in frame.groupby("symbol", sort=False):
        result.loc[group.index] = (
            group["left"]
            .rolling(window, min_periods=window)
            .cov(group["right"])
            .to_numpy()
        )
    return result


def fundamental_change(
    values: pd.Series,
    symbols: pd.Series,
    periods: pd.Series,
) -> pd.Series:
    """Compute changes on unique reports, then carry the latest known change daily."""
    frame = pd.DataFrame(
        {"symbol": symbols, "period": periods, "value": values}, index=values.index
    )
    reports = (
        frame.dropna(subset=["symbol", "period", "value"])
        # The daily panel repeats each report after it becomes available. The
        # first occurrence is point-in-time; the last occurrence is in the future.
        .drop_duplicates(["symbol", "period"], keep="first").sort_values(
            ["symbol", "period"]
        )
    )
    reports["change"] = reports.groupby("symbol", sort=False)["value"].pct_change(
        fill_method=None
    )
    lookup = reports.set_index(["symbol", "period"])["change"]
    keys = pd.MultiIndex.from_arrays(
        [frame["symbol"].to_numpy(), frame["period"].to_numpy()],
        names=["symbol", "period"],
    )
    return pd.Series(lookup.reindex(keys).to_numpy(), index=values.index)
