"""Separate exploratory parents from persistent elite factors."""

from __future__ import annotations

from dataclasses import dataclass

from src.utils.logging import ExperimentRecord


@dataclass(frozen=True)
class TierDecision:
    tier: str
    reasons: tuple[str, ...]


def classify_tier(record: ExperimentRecord, config: dict) -> TierDecision:
    if not record.integrity_passed:
        return TierDecision(
            "RETIRED", tuple(record.integrity_issues) or ("integrity failure",)
        )
    ic = record.mean_rank_ic if record.mean_rank_ic is not None else float("-inf")
    tstat = record.ic_tstat if record.ic_tstat is not None else float("-inf")
    folds = record.positive_fold_count or 0
    turnover = record.turnover if record.turnover is not None else float("inf")
    high_cost = (
        record.high_cost_sharpe
        if record.high_cost_sharpe is not None
        else float("-inf")
    )
    if (
        ic >= float(config.get("elite_min_mean_ic", 0.005))
        and tstat >= float(config.get("elite_min_tstat", 1.0))
        and folds >= int(config.get("elite_min_positive_folds", 3))
        and high_cost >= float(config.get("elite_min_high_cost_sharpe", 0.0))
        and turnover <= float(config.get("elite_max_turnover", 1.5))
    ):
        return TierDecision("ELITE", ("Passed strict archive thresholds.",))
    if (
        ic > float(config.get("parent_min_mean_ic", 0.0))
        and folds >= int(config.get("parent_min_positive_folds", 1))
        and turnover <= float(config.get("parent_max_turnover", 2.0))
    ):
        return TierDecision("PARENT", ("Promising enough for controlled evolution.",))
    return TierDecision(
        "RETIRED", tuple(record.reasons) or ("Insufficient research evidence.",)
    )
