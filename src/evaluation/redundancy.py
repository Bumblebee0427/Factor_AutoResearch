"""Signal novelty and incremental information checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.ic import summarize_ic


def mean_cross_sectional_correlation(
    candidate: pd.Series,
    existing: pd.Series,
    dates: pd.Series,
) -> float:
    frame = pd.DataFrame(
        {"date": dates, "candidate": candidate, "existing": existing}
    ).dropna()

    def safe_spearman(day: pd.DataFrame) -> float:
        if (
            len(day) < 2
            or day["candidate"].nunique() < 2
            or day["existing"].nunique() < 2
        ):
            return np.nan
        return float(day["candidate"].corr(day["existing"], method="spearman"))

    daily = frame.groupby("date").apply(
        safe_spearman,
        include_groups=False,
    )
    return float(daily.mean())


def max_library_correlation(
    candidate: pd.Series,
    library: dict[str, pd.Series],
    dates: pd.Series,
) -> tuple[float, str | None]:
    correlations = {
        factor_id: mean_cross_sectional_correlation(candidate, signal, dates)
        for factor_id, signal in library.items()
    }
    valid = {key: value for key, value in correlations.items() if np.isfinite(value)}
    if not valid:
        return 0.0, None
    closest = max(valid, key=lambda key: abs(valid[key]))
    return float(abs(valid[closest])), closest


def residual_ic(
    candidate: pd.Series,
    parent: pd.Series,
    dates: pd.Series,
    future_returns: pd.Series,
) -> float:
    frame = pd.DataFrame(
        {
            "date": dates,
            "candidate": candidate,
            "parent": parent,
            "future_return": future_returns,
        }
    ).dropna()

    def residualize(day: pd.DataFrame) -> pd.Series:
        x = np.column_stack([np.ones(len(day)), day["parent"].to_numpy()])
        beta, *_ = np.linalg.lstsq(x, day["candidate"].to_numpy(), rcond=None)
        return pd.Series(day["candidate"].to_numpy() - x @ beta, index=day.index)

    frame["factor"] = frame.groupby("date", group_keys=False).apply(
        residualize, include_groups=False
    )
    # An exactly redundant signal leaves only floating-point regression noise.
    # Ranking that noise can manufacture a large, meaningless Spearman IC.
    residual_scale = frame["factor"].abs().max()
    candidate_scale = frame["candidate"].abs().max()
    tolerance = 1e-10 * max(1.0, float(candidate_scale))
    if not np.isfinite(residual_scale) or float(residual_scale) <= tolerance:
        return 0.0
    _, summary = summarize_ic(frame[["date", "factor", "future_return"]])
    return summary.mean_ic
