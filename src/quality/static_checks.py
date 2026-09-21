"""Fail-closed checks before any candidate is executed."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.core.schemas import DataContract
from src.factors.expression import validate_expression
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
    spec: FactorSpec,
    contract: DataContract,
    max_complexity: int,
    dsl_config: dict | None = None,
) -> StaticCheckResult:
    dsl_config = dsl_config or {}
    reasons: list[str] = []
    supported, contract_reasons = contract.supports(spec)
    if not supported:
        reasons.extend(contract_reasons)
    if spec.uses_future_data:
        reasons.append("Factor references future data.")
    if spec.complexity > max_complexity:
        reasons.append(f"Complexity {spec.complexity} exceeds cap {max_complexity}.")
    expression_result = validate_expression(
        spec.effective_expression,
        allowed_features=set(contract.available_features) | {"future_return"},
        allowed_windows=set(
            dsl_config.get("allowed_feature_windows", [1, 5, 10, 20, 60])
        ),
        allowed_groups=set(contract.group_fields),
        max_depth=int(dsl_config.get("max_ast_depth", 5)),
        max_operator_nodes=int(dsl_config.get("max_operator_nodes", 6)),
        max_rolling_nodes=int(dsl_config.get("max_rolling_nodes", 3)),
        max_binary_nodes=int(dsl_config.get("max_binary_nodes", 2)),
        max_group_nodes=int(dsl_config.get("max_group_nodes", 2)),
    )
    reasons.extend(expression_result.reasons)
    expression_check = check_expression(spec.canonical_formula)
    reasons.extend(expression_check.reasons)
    return StaticCheckResult(not reasons, tuple(dict.fromkeys(reasons)))
