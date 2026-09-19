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
    hit_rate: float
    observations: int


def daily_rank_ic(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("date", sort=True).apply(
        lambda day: day["factor"].corr(day["future_return"], method="spearman"),
        include_groups=False,
    )


def summarize_ic(frame: pd.DataFrame) -> tuple[pd.Series, ICSummary]:
    daily = daily_rank_ic(frame).dropna()
    volatility = float(daily.std(ddof=1))
    mean = float(daily.mean())
    tstat = (
        mean / (volatility / np.sqrt(len(daily)))
        if volatility and len(daily) > 1
        else np.nan
    )
    return daily, ICSummary(
        mean_ic=mean,
        ic_volatility=volatility,
        ic_ir=mean / volatility if volatility else np.nan,
        ic_tstat=float(tstat),
        hit_rate=float((daily > 0).mean()),
        observations=len(daily),
    )
