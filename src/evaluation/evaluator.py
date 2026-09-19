"""One common walk-forward evaluator for every candidate."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.evaluation.ic import summarize_ic
from src.evaluation.portfolio import evaluate_portfolio
from src.evaluation.validation import WalkForwardFold
from src.factors.primitives import future_return


@dataclass(frozen=True)
class FoldMetrics:
    fold: str
    mean_ic: float
    ic_volatility: float
    ic_ir: float
    ic_tstat: float
    ic_hit_rate: float
    annualized_return: float
    net_sharpe: float
    high_cost_sharpe: float
    max_drawdown: float
    turnover: float
    return_per_turnover: float
    quantile_monotonicity: float
    stock_day_observations: int


@dataclass(frozen=True)
class EvaluationResult:
    folds: tuple[FoldMetrics, ...]

    def _mean(self, field: str) -> float:
        return float(np.nanmean([getattr(fold, field) for fold in self.folds]))

    @property
    def mean_ic(self) -> float:
        return self._mean("mean_ic")

    @property
    def mean_ic_tstat(self) -> float:
        return self._mean("ic_tstat")

    @property
    def mean_net_sharpe(self) -> float:
        return self._mean("net_sharpe")

    @property
    def mean_high_cost_sharpe(self) -> float:
        return self._mean("high_cost_sharpe")

    @property
    def mean_turnover(self) -> float:
        return self._mean("turnover")

    @property
    def worst_drawdown(self) -> float:
        return float(np.nanmin([fold.max_drawdown for fold in self.folds]))

    @property
    def positive_ic_folds(self) -> int:
        return sum(fold.mean_ic > 0 for fold in self.folds)

    def to_dict(self) -> dict:
        return {
            "mean_ic": self.mean_ic,
            "mean_ic_tstat": self.mean_ic_tstat,
            "mean_net_sharpe": self.mean_net_sharpe,
            "mean_high_cost_sharpe": self.mean_high_cost_sharpe,
            "mean_turnover": self.mean_turnover,
            "worst_drawdown": self.worst_drawdown,
            "positive_ic_folds": self.positive_ic_folds,
            "folds": [asdict(fold) for fold in self.folds],
        }


def evaluate_fold(
    frame: pd.DataFrame, fold: WalkForwardFold, config: dict
) -> FoldMetrics:
    sample = frame.loc[
        frame["date"].between(
            pd.Timestamp(fold.validation_start), pd.Timestamp(fold.validation_end)
        )
    ].dropna(subset=["factor", "future_return", "portfolio_return"])
    _, ic = summarize_ic(sample)
    portfolio = evaluate_portfolio(
        sample,
        quantile=float(config["portfolio_quantile"]),
        quantile_count=int(config["quantile_count"]),
        cost_bps=float(config["transaction_cost_bps"]),
        high_cost_bps=float(config["high_cost_bps"]),
        annualization=int(config["annualization_days"]),
    )
    return FoldMetrics(
        fold=fold.name,
        mean_ic=ic.mean_ic,
        ic_volatility=ic.ic_volatility,
        ic_ir=ic.ic_ir,
        ic_tstat=ic.ic_tstat,
        ic_hit_rate=ic.hit_rate,
        annualized_return=portfolio.annualized_return,
        net_sharpe=portfolio.net_sharpe,
        high_cost_sharpe=portfolio.high_cost_sharpe,
        max_drawdown=portfolio.max_drawdown,
        turnover=portfolio.turnover,
        return_per_turnover=portfolio.return_per_turnover,
        quantile_monotonicity=portfolio.quantile_monotonicity,
        stock_day_observations=len(sample),
    )


def evaluate_walk_forward(
    panel: pd.DataFrame,
    factor: pd.Series,
    folds: list[WalkForwardFold],
    config: dict,
) -> EvaluationResult:
    frame = panel[["date", "symbol", "close"]].copy()
    frame["factor"] = factor
    frame["future_return"] = future_return(
        frame["close"], frame["symbol"], int(config["prediction_horizon_days"])
    )
    frame["portfolio_return"] = future_return(
        frame["close"], frame["symbol"], int(config["portfolio_horizon_days"])
    )
    return EvaluationResult(tuple(evaluate_fold(frame, fold, config) for fold in folds))
