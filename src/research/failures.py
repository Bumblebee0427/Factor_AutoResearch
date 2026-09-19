"""Stable failure taxonomy shared by selection, memory, and LLM feedback."""

from __future__ import annotations


FAILURE_CODES = (
    "lookahead_or_leakage",
    "insufficient_coverage",
    "integrity_failure",
    "degenerate_signal",
    "unstable_ic",
    "weak_signal",
    "low_statistical_significance",
    "weak_monetization",
    "cost_sensitivity",
    "excessive_turnover",
    "excessive_drawdown",
    "redundancy",
)


def classify_failure_reasons(
    reasons: tuple[str, ...] | list[str], *, integrity_passed: bool
) -> tuple[str, ...]:
    codes: list[str] = []
    for reason in reasons:
        normalized = reason.lower()
        if "future" in normalized or "lookahead" in normalized or "leak" in normalized:
            code = "lookahead_or_leakage"
        elif "coverage" in normalized:
            code = "insufficient_coverage"
        elif "variation" in normalized or "degenerate" in normalized:
            code = "degenerate_signal"
        elif "unstable ic" in normalized:
            code = "unstable_ic"
        elif "weak or negative" in normalized:
            code = "weak_signal"
        elif "t-stat" in normalized or "significance" in normalized:
            code = "low_statistical_significance"
        elif "monetization" in normalized:
            code = "weak_monetization"
        elif "high-cost" in normalized or "cost scenario" in normalized:
            code = "cost_sensitivity"
        elif "turnover" in normalized:
            code = "excessive_turnover"
        elif "drawdown" in normalized:
            code = "excessive_drawdown"
        elif "redundant" in normalized:
            code = "redundancy"
        elif not integrity_passed:
            code = "integrity_failure"
        else:
            continue
        if code not in codes:
            codes.append(code)
    if not integrity_passed and not codes:
        codes.append("integrity_failure")
    return tuple(codes)
