"""Fail-closed checks before any candidate is executed."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.core.schemas import DataContract
from src.factors.schema import FactorSpec


@dataclass(frozen=True)
class StaticCheckResult:
    passed: bool
    reasons: tuple[str, ...]


_FORBIDDEN_PATTERNS = {
    r"shift\s*\(\s*-": "negative shift reads future observations",
    r"center\s*=\s*True": "centered rolling window reads future observations",
    r"future[_ ]?return": "future return is a target, not an input",
    r"\bbfill\s*\(": "backfilling may copy future information backward",
    r"expanding\s*\([^)]*\)\s*\.\s*(?:mean|std).*entire": "full-sample normalization is forbidden",
}


def check_expression(expression: str) -> StaticCheckResult:
    reasons = tuple(
        message
        for pattern, message in _FORBIDDEN_PATTERNS.items()
        if re.search(pattern, expression, flags=re.IGNORECASE)
    )
    return StaticCheckResult(not reasons, reasons)


def check_factor_spec(
    spec: FactorSpec, contract: DataContract, max_complexity: int
) -> StaticCheckResult:
    reasons: list[str] = []
    supported, contract_reasons = contract.supports(spec)
    if not supported:
        reasons.extend(contract_reasons)
    if spec.uses_future_data:
        reasons.append("Factor references future data.")
    if spec.complexity > max_complexity:
        reasons.append(f"Complexity {spec.complexity} exceeds cap {max_complexity}.")
    expression_check = check_expression(spec.canonical_formula)
    reasons.extend(expression_check.reasons)
    return StaticCheckResult(not reasons, tuple(dict.fromkeys(reasons)))
