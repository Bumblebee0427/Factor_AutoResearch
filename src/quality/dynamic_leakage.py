"""Empirical causality tests for factor builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.factors.builder import FactorBuilder
from src.factors.schema import FactorSpec


@dataclass(frozen=True)
class LeakageCheckResult:
    passed: bool
    truncation_passed: bool
    future_noise_passed: bool
    reasons: tuple[str, ...]


def _equivalent(left: pd.Series, right: pd.Series, atol: float = 1e-10) -> bool:
    frame = pd.concat([left.rename("left"), right.rename("right")], axis=1)
    if not frame["left"].isna().equals(frame["right"].isna()):
        return False
    finite = frame.dropna()
    return finite.empty or bool(
        np.allclose(finite["left"], finite["right"], atol=atol, rtol=1e-8)
    )


def check_dynamic_leakage(
    panel: pd.DataFrame,
    spec: FactorSpec,
    *,
    builder: FactorBuilder | None = None,
    build_fn: Callable[[pd.DataFrame], pd.Series] | None = None,
    sample_tickers: int = 4,
    cutoffs: int = 2,
) -> LeakageCheckResult:
    """Signals at or before t must not change when data after t is removed/noised."""
    if build_fn is None:
        factor_builder = builder or FactorBuilder()
        build_fn = lambda frame: factor_builder.build(frame, spec)
    ordered = panel.sort_values(["symbol", "date"]).copy()
    symbols = list(ordered["symbol"].drop_duplicates()[:sample_tickers])
    sample = ordered[ordered["symbol"].isin(symbols)].copy()
    dates = sorted(sample["date"].dropna().unique())
    if len(dates) < 8:
        return LeakageCheckResult(
            False, False, False, ("Insufficient history for dynamic leakage tests.",)
        )
    positions = np.linspace(
        max(2, len(dates) // 3), len(dates) - 3, num=max(1, cutoffs), dtype=int
    )
    full = build_fn(sample)
    truncation_passed = True
    noise_passed = True
    reasons: list[str] = []
    rng = np.random.default_rng(20260708)
    for position in sorted(set(int(item) for item in positions)):
        cutoff = dates[position]
        past_mask = sample["date"] <= cutoff
        truncated_panel = sample.loc[past_mask].copy()
        truncated = build_fn(truncated_panel)
        if not _equivalent(full.loc[truncated_panel.index], truncated):
            truncation_passed = False
            reasons.append(
                f"Truncation invariance failed at {pd.Timestamp(cutoff).date()}."
            )

        perturbed = sample.copy()
        future_mask = perturbed["date"] > cutoff
        protected = {"date", "symbol", "period_ending", "fundamental_available_date"}
        for column in perturbed.columns:
            if column in protected or not pd.api.types.is_numeric_dtype(
                perturbed[column]
            ):
                continue
            perturbed[column] = perturbed[column].astype(float)
            values = perturbed.loc[future_mask, column].astype(float)
            scale = float(values.std()) if values.notna().any() else 1.0
            scale = scale if np.isfinite(scale) and scale > 0 else 1.0
            perturbed.loc[future_mask, column] = values + rng.normal(
                0.0, 25.0 * scale, len(values)
            )
        noisy = build_fn(perturbed)
        if not _equivalent(full.loc[past_mask], noisy.loc[past_mask]):
            noise_passed = False
            reasons.append(
                f"Future-noise invariance failed at {pd.Timestamp(cutoff).date()}."
            )
    return LeakageCheckResult(
        truncation_passed and noise_passed,
        truncation_passed,
        noise_passed,
        tuple(dict.fromkeys(reasons)),
    )
