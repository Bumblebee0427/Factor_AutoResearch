"""Cross-sectional transforms applied independently on each signal date."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_rank(values: pd.Series, dates: pd.Series) -> pd.Series:
    return values.groupby(dates, sort=False).rank(method="average", pct=True).sub(0.5)


def cross_sectional_zscore(values: pd.Series, dates: pd.Series) -> pd.Series:
    grouped = values.groupby(dates, sort=False)
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    return values.sub(mean).div(std)


def winsorize(values: pd.Series, dates: pd.Series, tail: float = 0.01) -> pd.Series:
    lower = values.groupby(dates, sort=False).transform(lambda x: x.quantile(tail))
    upper = values.groupby(dates, sort=False).transform(
        lambda x: x.quantile(1.0 - tail)
    )
    return values.clip(lower=lower, upper=upper)


def winsorize_zscore(
    values: pd.Series, dates: pd.Series, tail: float = 0.01
) -> pd.Series:
    return cross_sectional_zscore(winsorize(values, dates, tail), dates)


def _group_frame(
    values: pd.Series,
    dates: pd.Series,
    groups: pd.Series,
    minimum_size: int,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"value": values, "date": dates, "group": groups}, index=values.index
    )
    sizes = frame.groupby(["date", "group"], sort=False)["value"].transform(
        lambda series: series.notna().sum()
    )
    frame.loc[sizes < minimum_size, "value"] = np.nan
    return frame


def group_rank(
    values: pd.Series,
    dates: pd.Series,
    groups: pd.Series,
    minimum_size: int = 5,
) -> pd.Series:
    frame = _group_frame(values, dates, groups, minimum_size)
    return (
        frame.groupby(["date", "group"], sort=False)["value"]
        .rank(method="average", pct=True)
        .sub(0.5)
        .reindex(values.index)
    )


def group_zscore(
    values: pd.Series,
    dates: pd.Series,
    groups: pd.Series,
    minimum_size: int = 5,
) -> pd.Series:
    frame = _group_frame(values, dates, groups, minimum_size)
    grouped = frame.groupby(["date", "group"], sort=False)["value"]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    return frame["value"].sub(mean).div(std).reindex(values.index)


def group_neutralize(
    values: pd.Series,
    dates: pd.Series,
    groups: pd.Series,
    minimum_size: int = 5,
) -> pd.Series:
    frame = _group_frame(values, dates, groups, minimum_size)
    grouped = frame.groupby(["date", "group"], sort=False)["value"]
    return frame["value"].sub(grouped.transform("mean")).reindex(values.index)
