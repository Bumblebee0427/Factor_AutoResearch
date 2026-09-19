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
    naive_ic_tstat: float
    newey_west_lags: int
    ic_hit_rate: float
    multi_horizon_ic: dict[str, dict[str, float | int]]
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
        values = np.asarray([getattr(fold, field) for fold in self.folds], dtype=float)
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if finite.size else np.nan

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
        values = np.asarray([fold.max_drawdown for fold in self.folds], dtype=float)
        finite = values[np.isfinite(values)]
        return float(finite.min()) if finite.size else np.nan

    @property
    def positive_ic_folds(self) -> int:
        return sum(fold.mean_ic > 0 for fold in self.folds)

    @property
    def multi_horizon_mean_ic(self) -> dict[str, float]:
        horizons = {horizon for fold in self.folds for horizon in fold.multi_horizon_ic}
        means = {}
        for horizon in sorted(horizons, key=lambda value: int(value.rstrip("d"))):
            values = np.asarray(
                [
                    fold.multi_horizon_ic[horizon]["mean_ic"]
                    for fold in self.folds
                    if horizon in fold.multi_horizon_ic
                ],
                dtype=float,
            )
            finite = values[np.isfinite(values)]
            means[horizon] = float(finite.mean()) if finite.size else np.nan
        return means

    @property
    def fold_ic_sign_consistency(self) -> float:
        if not self.folds:
            return np.nan
        values = np.asarray([fold.mean_ic for fold in self.folds], dtype=float)
        finite = values[np.isfinite(values)]
        if not finite.size:
            return np.nan
        signs = np.sign(finite)
        return float(max((signs > 0).mean(), (signs < 0).mean()))

    def to_dict(self) -> dict:
        return {
            "mean_ic": self.mean_ic,
            "mean_ic_tstat": self.mean_ic_tstat,
            "mean_net_sharpe": self.mean_net_sharpe,
            "mean_high_cost_sharpe": self.mean_high_cost_sharpe,
            "mean_turnover": self.mean_turnover,
            "worst_drawdown": self.worst_drawdown,
            "positive_ic_folds": self.positive_ic_folds,
            "multi_horizon_mean_ic": self.multi_horizon_mean_ic,
            "fold_ic_sign_consistency": self.fold_ic_sign_consistency,
            "folds": [asdict(fold) for fold in self.folds],
        }


def evaluate_fold(
    frame: pd.DataFrame, fold: WalkForwardFold, config: dict
) -> FoldMetrics:
    fold_frame = frame.loc[
        frame["date"].between(
            pd.Timestamp(fold.validation_start), pd.Timestamp(fold.validation_end)
        )
    ]
    primary_horizon = int(config["prediction_horizon_days"])
    primary_target = f"future_return_{primary_horizon}d"
    sample = fold_frame.dropna(
        subset=["factor", primary_target, "portfolio_return"]
    ).copy()
    sample["future_return"] = sample[primary_target]
    configured_lags = config.get("newey_west_lags", "auto")
    explicit_lags = None if configured_lags == "auto" else int(configured_lags)
    _, ic = summarize_ic(
        sample,
        horizon_days=primary_horizon,
        newey_west_lags=explicit_lags,
    )
    multi_horizon: dict[str, dict[str, float | int]] = {}
    for horizon in config.get("ic_horizons_days", [primary_horizon]):
        horizon = int(horizon)
        target = f"future_return_{horizon}d"
        horizon_sample = fold_frame.dropna(subset=["factor", target])
        _, horizon_ic = summarize_ic(
            horizon_sample,
            target_column=target,
            horizon_days=horizon,
            newey_west_lags=explicit_lags,
        )
        multi_horizon[f"{horizon}d"] = {
            "mean_ic": horizon_ic.mean_ic,
            "newey_west_tstat": horizon_ic.ic_tstat,
            "naive_tstat": horizon_ic.naive_ic_tstat,
            "hit_rate": horizon_ic.hit_rate,
            "observations": horizon_ic.observations,
            "newey_west_lags": horizon_ic.newey_west_lags,
        }
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
        naive_ic_tstat=ic.naive_ic_tstat,
        newey_west_lags=ic.newey_west_lags,
        ic_hit_rate=ic.hit_rate,
        multi_horizon_ic=multi_horizon,
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
    horizons = {
        int(config["prediction_horizon_days"]),
        *(int(value) for value in config.get("ic_horizons_days", [])),
    }
    for horizon in sorted(horizons):
        frame[f"future_return_{horizon}d"] = future_return(
            frame["close"], frame["symbol"], horizon
        )
    frame["portfolio_return"] = future_return(
        frame["close"], frame["symbol"], int(config["portfolio_horizon_days"])
    )
    return EvaluationResult(tuple(evaluate_fold(frame, fold, config) for fold in folds))
