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
    mentions_reversal = "revers" in hypothesis
    features = spec.required_features
    base_is_return = spec.base_feature == "return" or "return" in features
    reasons: list[str] = []
    if mentions_reversal and base_is_return and spec.direction > 0:
        reasons.append("Reversal hypothesis requires a negative return direction.")
    if (
        not mentions_reversal
        and any(word in hypothesis for word in ("momentum", "continuation", "winner"))
        and base_is_return
        and spec.direction < 0
    ):
        reasons.append("Momentum hypothesis conflicts with negative return direction.")
    if "news" in hypothesis and "news_volume" not in features:
        reasons.append("News-conditioned hypothesis has no news input.")
    fundamental_features = {
        "after_tax_roe",
        "operating_margin",
        "profit_margin",
        "earnings_yield",
        "asset_growth",
    }
    if mechanism.startswith("FUNDAMENTAL") and not features.intersection(
        fundamental_features
    ):
        reasons.append("Fundamental mechanism has no fundamental primary input.")
    if mechanism == "CROSS_DOMAIN_REGIME" and len(features) < 2:
        reasons.append(
            "Cross-domain mechanism requires an explicit interaction feature."
        )
    return AlignmentResult(not reasons, tuple(reasons))
