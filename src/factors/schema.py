"""Typed JSON-compatible FactorSpec grammar; arbitrary Python is impossible."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.factors.expression import (
    Constant,
    Expr,
    Feature,
    Op,
    expression_hash,
    expr_from_dict,
)


ALLOWED_BASE_FEATURES = {
    "return",
    "volatility",
    "volume_shock",
    "distance_to_high",
    "after_tax_roe",
    "operating_margin",
    "profit_margin",
    "earnings_yield",
    "asset_growth",
    "news_volume",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "raw_volume",
}
DEMONSTRATION_FEATURES = {"future_return"}
ALLOWED_TS_OPERATORS = {None, "identity", "rolling_mean", "vol_adjust"}
ALLOWED_CS_OPERATORS = {
    None,
    "rank",
    "zscore",
    "winsorize",
    "winsorize_zscore",
    "sign",
}


@dataclass(frozen=True)
class FactorSpec:
    factor_id: str
    generation: int
    parent_ids: tuple[str, ...]
    family: str
    hypothesis: str
    base_feature: str
    ts_operator: str | None = None
    window: int | None = None
    cs_operator: str | None = None
    interaction_feature: str | None = None
    interaction_window: int | None = None
    direction: int = 1
    mutation_reason: str | None = None
    demonstration_only: bool = False
    proposal_type: str = "deterministic"
    evidence_factor_ids: tuple[str, ...] = ()
    targeted_failure: str | None = None
    expected_metric_effect: str | None = None
    falsification_condition: str | None = None
    mechanism: str | None = None
    expression: Expr | None = None

    def __post_init__(self) -> None:
        allowed = set(ALLOWED_BASE_FEATURES)
        if self.demonstration_only:
            allowed |= DEMONSTRATION_FEATURES
        referenced = {self.base_feature} if self.base_feature else set()
        if self.interaction_feature:
            referenced.add(self.interaction_feature)
        unknown = referenced - allowed
        if unknown:
            raise ValueError(f"Unsupported DSL primitive(s): {sorted(unknown)}")
        if self.expression is not None:
            if not isinstance(self.expression, (Constant, Feature, Op)):
                raise ValueError("expression must be a Feature or Op AST node.")
        if self.ts_operator not in ALLOWED_TS_OPERATORS:
            raise ValueError(f"Unsupported time-series operator: {self.ts_operator}")
        if self.cs_operator not in ALLOWED_CS_OPERATORS:
            raise ValueError(
                f"Unsupported cross-sectional operator: {self.cs_operator}"
            )
        if self.direction not in {-1, 1}:
            raise ValueError("direction must be either -1 or +1")
        if (
            self.expression is None
            and self.base_feature
            in {
                "return",
                "volatility",
                "volume_shock",
                "distance_to_high",
                "news_volume",
                "future_return",
            }
            and (self.window is None or self.window < 1)
        ):
            raise ValueError(f"{self.base_feature} requires a positive window")
        if not self.hypothesis.strip():
            raise ValueError("Every candidate requires an ex-ante economic hypothesis.")

    @property
    def complexity(self) -> int:
        if self.expression is not None:
            return self.expression.complexity()
        return (
            1
            + int(self.ts_operator not in {None, "identity"})
            + int(self.cs_operator is not None)
            + int(self.interaction_feature is not None)
        )

    @property
    def uses_future_data(self) -> bool:
        if self.expression is not None:
            return bool(self.expression.required_features() & DEMONSTRATION_FEATURES)
        return self.base_feature in DEMONSTRATION_FEATURES or (
            self.interaction_feature in DEMONSTRATION_FEATURES
        )

    @property
    def canonical_formula(self) -> str:
        if self.expression is not None:
            return self.expression.canonical()
        base = (
            f"{self.base_feature}({self.window})" if self.window else self.base_feature
        )
        if self.ts_operator not in {None, "identity"}:
            base = f"{self.ts_operator}({base})"
        if self.interaction_feature:
            interaction = (
                f"{self.interaction_feature}({self.interaction_window})"
                if self.interaction_window
                else self.interaction_feature
            )
            base = f"({base} * {interaction})"
        if self.cs_operator:
            base = f"{self.cs_operator}({base})"
        return f"{self.direction:+d} * {base}"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "factor_id": self.factor_id,
            "generation": self.generation,
            "parent_ids": list(self.parent_ids),
            "family": self.family,
            "hypothesis": self.hypothesis,
            "base_feature": self.base_feature,
            "ts_operator": self.ts_operator,
            "window": self.window,
            "cs_operator": self.cs_operator,
            "interaction_feature": self.interaction_feature,
            "interaction_window": self.interaction_window,
            "direction": self.direction,
            "mutation_reason": self.mutation_reason,
            "demonstration_only": self.demonstration_only,
            "proposal_type": self.proposal_type,
            "evidence_factor_ids": list(self.evidence_factor_ids),
            "targeted_failure": self.targeted_failure,
            "expected_metric_effect": self.expected_metric_effect,
            "falsification_condition": self.falsification_condition,
            "mechanism": self.mechanism,
            "expression": self.expression.to_dict() if self.expression else None,
        }
        payload["complexity"] = self.complexity
        payload["canonical_formula"] = self.canonical_formula
        payload["expression_hash"] = self.expression_hash
        payload["required_features"] = sorted(self.required_features)
        payload["required_history"] = self.required_history
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FactorSpec":
        values = dict(payload)
        for derived in (
            "complexity",
            "canonical_formula",
            "expression_hash",
            "required_features",
            "required_history",
        ):
            values.pop(derived, None)
        values["parent_ids"] = tuple(values.get("parent_ids", ()))
        values["evidence_factor_ids"] = tuple(values.get("evidence_factor_ids", ()))
        if values.get("expression") is not None:
            values["expression"] = expr_from_dict(values["expression"])
        return cls(**values)

    @property
    def effective_expression(self) -> Expr:
        return self.expression or legacy_spec_to_expr(self)

    @property
    def expression_hash(self) -> str:
        return expression_hash(self.effective_expression)

    @property
    def required_features(self) -> frozenset[str]:
        return self.effective_expression.required_features()

    @property
    def required_history(self) -> int:
        return self.effective_expression.required_history()


def legacy_spec_to_expr(spec: FactorSpec) -> Expr:
    """Compile the V1 flat recipe into the sole V2 execution representation."""
    expression: Expr = Feature(spec.base_feature, spec.window)
    if spec.ts_operator == "rolling_mean":
        expression = Op("rolling_mean", (expression,), {"window": spec.window})
    elif spec.ts_operator == "vol_adjust":
        expression = Op(
            "safe_div",
            (expression, Feature("volatility", spec.window)),
            {},
        )
    if spec.interaction_feature:
        expression = Op(
            "mul",
            (expression, Feature(spec.interaction_feature, spec.interaction_window)),
            {},
        )
    cs_aliases = {
        "rank": "cs_rank",
        "zscore": "cs_zscore",
        "winsorize": "winsorize",
        "winsorize_zscore": "winsorize_zscore",
    }
    if spec.cs_operator in cs_aliases:
        expression = Op(cs_aliases[spec.cs_operator], (expression,), {})
    elif spec.cs_operator == "sign":
        expression = Op("sign", (expression,), {})
    if spec.direction < 0:
        expression = Op("neg", (expression,), {})
    return expression
