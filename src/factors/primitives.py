"""Historical time-series primitives applied within each asset."""

from __future__ import annotations

import pandas as pd


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
        .drop_duplicates(["symbol", "period"], keep="last")
        .sort_values(["symbol", "period"])
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
