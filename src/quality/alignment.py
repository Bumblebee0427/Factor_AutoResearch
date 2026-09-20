"""Rule-based hypothesis, mechanism, and implementation alignment."""

from __future__ import annotations

from dataclasses import dataclass

from src.factors.schema import FactorSpec


@dataclass(frozen=True)
class AlignmentResult:
    passed: bool
    reasons: tuple[str, ...]


def check_alignment(spec: FactorSpec, mechanism: str) -> AlignmentResult:
    hypothesis = spec.hypothesis.lower()
    reasons: list[str] = []
    if (
        "reversal" in hypothesis
        and spec.base_feature == "return"
        and spec.direction > 0
    ):
        reasons.append("Reversal hypothesis requires a negative return direction.")
    if (
        any(word in hypothesis for word in ("momentum", "continuation", "winner"))
        and spec.base_feature == "return"
        and spec.direction < 0
    ):
        reasons.append("Momentum hypothesis conflicts with negative return direction.")
    if "news" in hypothesis and "news_volume" not in {
        spec.base_feature,
        spec.interaction_feature,
    }:
        reasons.append("News-conditioned hypothesis has no news input.")
    if mechanism.startswith("FUNDAMENTAL") and spec.base_feature not in {
        "after_tax_roe",
        "operating_margin",
        "profit_margin",
        "earnings_yield",
        "asset_growth",
    }:
        reasons.append("Fundamental mechanism has no fundamental primary input.")
    if mechanism == "CROSS_DOMAIN_REGIME" and not spec.interaction_feature:
        reasons.append(
            "Cross-domain mechanism requires an explicit interaction feature."
        )
    return AlignmentResult(not reasons, tuple(reasons))
