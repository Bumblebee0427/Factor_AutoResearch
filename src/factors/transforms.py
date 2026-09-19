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
