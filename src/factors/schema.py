"""Typed JSON-compatible FactorSpec grammar; arbitrary Python is impossible."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


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

    def __post_init__(self) -> None:
        allowed = set(ALLOWED_BASE_FEATURES)
        if self.demonstration_only:
            allowed |= DEMONSTRATION_FEATURES
        referenced = {self.base_feature}
        if self.interaction_feature:
            referenced.add(self.interaction_feature)
        unknown = referenced - allowed
        if unknown:
            raise ValueError(f"Unsupported DSL primitive(s): {sorted(unknown)}")
        if self.ts_operator not in ALLOWED_TS_OPERATORS:
            raise ValueError(f"Unsupported time-series operator: {self.ts_operator}")
        if self.cs_operator not in ALLOWED_CS_OPERATORS:
            raise ValueError(
                f"Unsupported cross-sectional operator: {self.cs_operator}"
            )
        if self.direction not in {-1, 1}:
            raise ValueError("direction must be either -1 or +1")
        if self.base_feature in {
            "return",
            "volatility",
            "volume_shock",
            "distance_to_high",
            "news_volume",
            "future_return",
        } and (self.window is None or self.window < 1):
            raise ValueError(f"{self.base_feature} requires a positive window")
        if not self.hypothesis.strip():
            raise ValueError("Every candidate requires an ex-ante economic hypothesis.")

    @property
    def complexity(self) -> int:
        return (
            1
            + int(self.ts_operator not in {None, "identity"})
            + int(self.cs_operator is not None)
            + int(self.interaction_feature is not None)
        )

    @property
    def uses_future_data(self) -> bool:
        return self.base_feature in DEMONSTRATION_FEATURES or (
            self.interaction_feature in DEMONSTRATION_FEATURES
        )

    @property
    def canonical_formula(self) -> str:
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
        payload = asdict(self)
        payload["complexity"] = self.complexity
        payload["canonical_formula"] = self.canonical_formula
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FactorSpec":
        values = dict(payload)
        values.pop("complexity", None)
        values.pop("canonical_formula", None)
        values["parent_ids"] = tuple(values.get("parent_ids", ()))
        return cls(**values)
