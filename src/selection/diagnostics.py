"""Transparent, independent diagnostics for the configured Elite gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.utils.logging import ExperimentRecord


@dataclass(frozen=True)
class EliteGateDiagnostic:
    factor_id: str
    mean_ic_pass: bool
    tstat_pass: bool
    positive_folds_pass: bool
    high_cost_sharpe_pass: bool
    turnover_pass: bool
    failed_gates: tuple[str, ...]
    distance_to_threshold: dict[str, float | None]

    @property
    def passed(self) -> bool:
        return not self.failed_gates

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnose_elite_gates(
    record: ExperimentRecord, config: dict
) -> EliteGateDiagnostic:
    """Evaluate every Elite criterion independently using configured thresholds."""
    thresholds = {
        "mean_ic": float(config.get("elite_min_mean_ic", 0.005)),
        "tstat": float(config.get("elite_min_tstat", 1.0)),
        "positive_folds": int(config.get("elite_min_positive_folds", 3)),
        "high_cost_sharpe": float(
            config.get("elite_min_high_cost_sharpe", 0.0)
        ),
        "turnover_max": float(config.get("elite_max_turnover", 1.5)),
    }
    values = {
        "mean_ic": record.mean_rank_ic,
        "tstat": record.ic_tstat,
        "positive_folds": record.positive_fold_count,
        "high_cost_sharpe": record.high_cost_sharpe,
        "turnover": record.turnover,
    }

    def gap(value: float | int | None, threshold: float, *, lower_is_better=False):
        if value is None:
            return None
        return float(threshold - value if lower_is_better else value - threshold)

    distances = {
        "mean_ic": gap(values["mean_ic"], thresholds["mean_ic"]),
        "tstat": gap(values["tstat"], thresholds["tstat"]),
        "positive_folds": gap(
            values["positive_folds"], thresholds["positive_folds"]
        ),
        "high_cost_sharpe": gap(
            values["high_cost_sharpe"], thresholds["high_cost_sharpe"]
        ),
        "turnover": gap(
            values["turnover"], thresholds["turnover_max"], lower_is_better=True
        ),
    }
    pass_flags = {
        "mean_ic": values["mean_ic"] is not None
        and values["mean_ic"] >= thresholds["mean_ic"],
        "tstat": values["tstat"] is not None
        and values["tstat"] >= thresholds["tstat"],
        "positive_folds": values["positive_folds"] is not None
        and values["positive_folds"] >= thresholds["positive_folds"],
        "high_cost_sharpe": values["high_cost_sharpe"] is not None
        and values["high_cost_sharpe"] >= thresholds["high_cost_sharpe"],
        "turnover": values["turnover"] is not None
        and values["turnover"] <= thresholds["turnover_max"],
    }
    return EliteGateDiagnostic(
        factor_id=record.factor_id,
        mean_ic_pass=pass_flags["mean_ic"],
        tstat_pass=pass_flags["tstat"],
        positive_folds_pass=pass_flags["positive_folds"],
        high_cost_sharpe_pass=pass_flags["high_cost_sharpe"],
        turnover_pass=pass_flags["turnover"],
        failed_gates=tuple(name for name, passed in pass_flags.items() if not passed),
        distance_to_threshold=distances,
    )


def diagnose_values(metrics: dict, config: dict, factor_id: str = "") -> dict:
    """Small adapter for machine-readable round outcome dictionaries."""
    record = ExperimentRecord(
        factor_id=factor_id,
        generation=0,
        parent_ids=(),
        family="unknown",
        hypothesis="diagnostic input",
        canonical_formula="",
        integrity_passed=bool(metrics.get("integrity_passed", True)),
        integrity_issues=(),
        fold_metrics=(),
        mean_rank_ic=metrics.get("mean_rank_ic"),
        ic_tstat=metrics.get("newey_west_tstat", metrics.get("ic_tstat")),
        positive_fold_count=metrics.get("positive_fold_count"),
        long_short_sharpe=metrics.get("net_sharpe"),
        high_cost_sharpe=metrics.get("high_cost_sharpe"),
        turnover=metrics.get("turnover"),
        max_drawdown=metrics.get("max_drawdown"),
        redundancy_corr=metrics.get("redundancy_corr"),
        closest_factor_id=None,
        residual_ic=None,
        decision=str(metrics.get("decision", "PARENT")),
        reasons=(),
    )
    return diagnose_elite_gates(record, config).to_dict()
