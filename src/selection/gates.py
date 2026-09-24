"""Separate exploratory parents from persistent elite factors."""

from __future__ import annotations

from dataclasses import dataclass

from src.selection.diagnostics import diagnose_elite_gates
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
    diagnostic = diagnose_elite_gates(record, config)
    ic = record.mean_rank_ic if record.mean_rank_ic is not None else float("-inf")
    folds = record.positive_fold_count or 0
    turnover = record.turnover if record.turnover is not None else float("inf")
    if diagnostic.passed:
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
