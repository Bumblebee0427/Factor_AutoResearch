"""Fail-closed point-in-time, target-leakage, and coverage checks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.factors.schema import FactorSpec


@dataclass(frozen=True)
class IntegrityIssue:
    code: str
    message: str


@dataclass
class IntegrityReport:
    issues: list[IntegrityIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.issues

    @property
    def reasons(self) -> list[str]:
        return [issue.message for issue in self.issues]


def check_integrity(
    spec: FactorSpec,
    panel: pd.DataFrame,
    factor: pd.Series | None,
    *,
    max_complexity: int,
    minimum_coverage: float,
) -> IntegrityReport:
    report = IntegrityReport()
    if spec.uses_future_data:
        report.issues.append(
            IntegrityIssue("target_leakage", "Factor references a future return.")
        )
    if spec.complexity > max_complexity:
        report.issues.append(
            IntegrityIssue(
                "complexity_limit",
                f"Complexity {spec.complexity} exceeds cap {max_complexity}.",
            )
        )
    missing = {"date", "symbol", "close"} - set(panel)
    if missing:
        report.issues.append(
            IntegrityIssue(
                "missing_fields", f"Panel is missing fields: {sorted(missing)}"
            )
        )
    if {"fundamental_available_date", "date"} <= set(panel):
        invalid = panel["fundamental_available_date"].notna() & (
            panel["fundamental_available_date"] > panel["date"]
        )
        if invalid.any():
            report.issues.append(
                IntegrityIssue(
                    "fundamental_lookahead",
                    f"{int(invalid.sum())} rows precede fundamental availability.",
                )
            )
    if factor is not None:
        finite = factor.notna() & np.isfinite(factor)
        coverage = float(finite.mean())
        if coverage < minimum_coverage:
            report.issues.append(
                IntegrityIssue(
                    "low_coverage",
                    f"Finite factor coverage {coverage:.1%} is below {minimum_coverage:.1%}.",
                )
            )
        if factor[finite].nunique() <= 1:
            report.issues.append(
                IntegrityIssue("constant_factor", "Factor has no variation.")
            )
    return report
