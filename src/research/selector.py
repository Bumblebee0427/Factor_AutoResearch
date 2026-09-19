"""Pre-committed Promote/Hold/Retire gates."""

from __future__ import annotations

from dataclasses import dataclass

from src.evaluation.evaluator import EvaluationResult
from src.factors.schema import FactorSpec


@dataclass(frozen=True)
class GateDecision:
    decision: str
    reasons: tuple[str, ...]


def decide(
    spec: FactorSpec,
    metrics: EvaluationResult,
    config: dict,
    *,
    redundancy_corr: float,
    residual_ic: float | None,
) -> GateDecision:
    if spec.complexity > int(config["max_complexity"]):
        return GateDecision("RETIRE", ("complexity cap exceeded",))
    if metrics.positive_ic_folds < int(config["min_positive_ic_folds"]):
        return GateDecision("RETIRE", ("unstable IC sign across validation folds",))
    if metrics.mean_ic <= float(config["min_mean_ic"]):
        return GateDecision("RETIRE", ("weak or negative mean rank IC",))
    if metrics.worst_drawdown < float(config["max_fold_drawdown"]):
        return GateDecision("RETIRE", ("catastrophic validation-fold drawdown",))
    if redundancy_corr > float(config["redundancy_correlation"]) and (
        residual_ic is None or residual_ic < float(config["min_incremental_ic"])
    ):
        return GateDecision("RETIRE", ("redundant with promoted library",))

    hold_reasons: list[str] = []
    if metrics.mean_ic_tstat < float(config["min_ic_tstat"]):
        hold_reasons.append("IC t-stat below promotion threshold")
    if metrics.mean_net_sharpe <= float(config["min_net_sharpe"]):
        hold_reasons.append("weak economic monetization after costs")
    if metrics.mean_high_cost_sharpe <= float(config["min_high_cost_sharpe"]):
        hold_reasons.append("signal does not survive high-cost scenario")
    if metrics.mean_turnover > float(config["max_daily_turnover"]):
        hold_reasons.append("turnover exceeds implementation threshold")
    if hold_reasons:
        return GateDecision("HOLD", tuple(hold_reasons))
    return GateDecision(
        "PROMOTE",
        ("passed predictive, economic, stability, complexity, and novelty gates",),
    )
