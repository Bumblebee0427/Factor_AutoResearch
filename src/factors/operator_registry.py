"""Authoritative registry for the safe Factor DSL V2 operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd

import src.factors.primitives as primitives
import src.factors.transforms as transforms


Implementation = Callable[
    [pd.DataFrame, tuple[pd.Series, ...], Mapping[str, object]], pd.Series
]


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    category: str
    arity: int | tuple[int, ...]
    implementation: Implementation
    parameter_schema: Mapping[str, str]
    scope: str
    complexity_cost: int
    minimum_history_fn: Callable[[Mapping[str, object]], int]
    allows_nan: bool
    may_create_inf: bool
    commutative: bool = False
    description: str = ""


def _history(params: Mapping[str, object], key: str = "window") -> int:
    return int(params.get(key, 1))


def _binary(op: Callable[[pd.Series, pd.Series], pd.Series]) -> Implementation:
    def apply(
        panel: pd.DataFrame, args: tuple[pd.Series, ...], params: Mapping[str, object]
    ) -> pd.Series:
        return op(args[0], args[1])

    return apply


def _unary(op: Callable[[pd.Series], pd.Series]) -> Implementation:
    def apply(
        panel: pd.DataFrame, args: tuple[pd.Series, ...], params: Mapping[str, object]
    ) -> pd.Series:
        return op(args[0])

    return apply


def _safe_div(
    panel: pd.DataFrame, args: tuple[pd.Series, ...], params: Mapping[str, object]
) -> pd.Series:
    epsilon = float(params.get("eps", 1.0e-12))
    denominator = args[1].where(args[1].abs() > epsilon)
    return args[0].div(denominator)


def _clip(
    panel: pd.DataFrame, args: tuple[pd.Series, ...], params: Mapping[str, object]
) -> pd.Series:
    return args[0].clip(lower=float(params["lower"]), upper=float(params["upper"]))


def _lag(
    panel: pd.DataFrame, args: tuple[pd.Series, ...], params: Mapping[str, object]
) -> pd.Series:
    return args[0].groupby(panel["symbol"], sort=False).shift(int(params["periods"]))


def _delta(
    panel: pd.DataFrame, args: tuple[pd.Series, ...], params: Mapping[str, object]
) -> pd.Series:
    periods = int(params["periods"])
    return args[0].sub(_lag(panel, args, {"periods": periods}))


def _rolling(name: str) -> Implementation:
    def apply(
        panel: pd.DataFrame, args: tuple[pd.Series, ...], params: Mapping[str, object]
    ) -> pd.Series:
        return getattr(primitives, name)(
            args[0], panel["symbol"], int(params["window"])
        )

    return apply


def _ts_rank(
    panel: pd.DataFrame, args: tuple[pd.Series, ...], params: Mapping[str, object]
) -> pd.Series:
    return primitives.ts_rank(args[0], panel["symbol"], int(params["window"]))


def _ts_zscore(
    panel: pd.DataFrame, args: tuple[pd.Series, ...], params: Mapping[str, object]
) -> pd.Series:
    return primitives.ts_zscore(args[0], panel["symbol"], int(params["window"]))


def _ewma(
    panel: pd.DataFrame, args: tuple[pd.Series, ...], params: Mapping[str, object]
) -> pd.Series:
    return primitives.ewma(args[0], panel["symbol"], float(params["halflife"]))


def _decay_linear(
    panel: pd.DataFrame, args: tuple[pd.Series, ...], params: Mapping[str, object]
) -> pd.Series:
    return primitives.decay_linear(args[0], panel["symbol"], int(params["window"]))


def _rolling_pair(name: str) -> Implementation:
    def apply(
        panel: pd.DataFrame, args: tuple[pd.Series, ...], params: Mapping[str, object]
    ) -> pd.Series:
        return getattr(primitives, name)(
            args[0], args[1], panel["symbol"], int(params["window"])
        )

    return apply


def _group(panel: pd.DataFrame, group: object) -> pd.Series:
    mapping = {
        "sector": "gics_sector",
        "subindustry": "gics_sub_industry",
    }
    column = mapping.get(str(group), str(group))
    if column not in panel:
        raise ValueError(f"Group field is unavailable in panel: {group}")
    return panel[column]


def _group_op(name: str) -> Implementation:
    def apply(
        panel: pd.DataFrame, args: tuple[pd.Series, ...], params: Mapping[str, object]
    ) -> pd.Series:
        groups = _group(panel, params.get("group"))
        minimum = int(params.get("min_group_size", 5))
        return getattr(transforms, name)(args[0], panel["date"], groups, minimum)

    return apply


def _cs(name: str) -> Implementation:
    def apply(
        panel: pd.DataFrame, args: tuple[pd.Series, ...], params: Mapping[str, object]
    ) -> pd.Series:
        return getattr(transforms, name)(args[0], panel["date"])

    return apply


def _spec(
    name: str,
    category: str,
    arity: int,
    implementation: Implementation,
    scope: str,
    cost: int,
    params: Mapping[str, str] | None = None,
    history_key: str = "window",
    commutative: bool = False,
    description: str = "",
) -> OperatorSpec:
    return OperatorSpec(
        name=name,
        category=category,
        arity=arity,
        implementation=implementation,
        parameter_schema=dict(params or {}),
        scope=scope,
        complexity_cost=cost,
        minimum_history_fn=lambda values, key=history_key: _history(values, key),
        allows_nan=True,
        may_create_inf=name not in {"safe_div"},
        commutative=commutative,
        description=description,
    )


OPERATOR_REGISTRY: dict[str, OperatorSpec] = {
    "add": _spec(
        "add",
        "arithmetic",
        2,
        _binary(lambda x, y: x.add(y)),
        "scalar",
        1,
        commutative=True,
    ),
    "sub": _spec("sub", "arithmetic", 2, _binary(lambda x, y: x.sub(y)), "scalar", 1),
    "mul": _spec(
        "mul",
        "arithmetic",
        2,
        _binary(lambda x, y: x.mul(y)),
        "scalar",
        1,
        commutative=True,
    ),
    "safe_div": _spec(
        "safe_div", "arithmetic", 2, _safe_div, "scalar", 2, {"eps": "positive float"}
    ),
    "neg": _spec("neg", "arithmetic", 1, _unary(lambda x: -x), "scalar", 1),
    "abs": _spec("abs", "arithmetic", 1, _unary(lambda x: x.abs()), "scalar", 1),
    "clip": _spec(
        "clip",
        "arithmetic",
        1,
        _clip,
        "scalar",
        1,
        {"lower": "finite float", "upper": "finite float"},
    ),
    "sign": _spec("sign", "arithmetic", 1, _unary(lambda x: np.sign(x)), "scalar", 1),
    "lag": _spec(
        "lag",
        "time_series",
        1,
        _lag,
        "time_series",
        1,
        {"periods": "positive integer"},
        "periods",
    ),
    "delta": _spec(
        "delta",
        "time_series",
        1,
        _delta,
        "time_series",
        1,
        {"periods": "positive integer"},
        "periods",
    ),
    "rolling_sum": _spec(
        "rolling_sum",
        "time_series",
        1,
        _rolling("rolling_sum_strict"),
        "time_series",
        1,
        {"window": "positive integer"},
    ),
    "rolling_mean": _spec(
        "rolling_mean",
        "time_series",
        1,
        _rolling("rolling_mean"),
        "time_series",
        1,
        {"window": "positive integer"},
    ),
    "rolling_std": _spec(
        "rolling_std",
        "time_series",
        1,
        _rolling("rolling_std"),
        "time_series",
        1,
        {"window": "positive integer"},
    ),
    "rolling_min": _spec(
        "rolling_min",
        "time_series",
        1,
        _rolling("rolling_min"),
        "time_series",
        1,
        {"window": "positive integer"},
    ),
    "rolling_max": _spec(
        "rolling_max",
        "time_series",
        1,
        _rolling("rolling_max"),
        "time_series",
        1,
        {"window": "positive integer"},
    ),
    "ts_rank": _spec(
        "ts_rank",
        "time_series",
        1,
        _ts_rank,
        "time_series",
        2,
        {"window": "positive integer"},
    ),
    "ts_zscore": _spec(
        "ts_zscore",
        "time_series",
        1,
        _ts_zscore,
        "time_series",
        2,
        {"window": "positive integer"},
    ),
    "ewma": _spec(
        "ewma",
        "time_series",
        1,
        _ewma,
        "time_series",
        1,
        {"halflife": "positive float"},
        "halflife",
    ),
    "decay_linear": _spec(
        "decay_linear",
        "time_series",
        1,
        _decay_linear,
        "time_series",
        2,
        {"window": "positive integer"},
    ),
    "rolling_corr": _spec(
        "rolling_corr",
        "time_series",
        2,
        _rolling_pair("rolling_corr"),
        "time_series",
        2,
        {"window": "positive integer"},
    ),
    "rolling_cov": _spec(
        "rolling_cov",
        "time_series",
        2,
        _rolling_pair("rolling_cov"),
        "time_series",
        2,
        {"window": "positive integer"},
    ),
    "cs_rank": _spec(
        "cs_rank", "cross_section", 1, _cs("cross_sectional_rank"), "cross_section", 1
    ),
    "cs_zscore": _spec(
        "cs_zscore",
        "cross_section",
        1,
        _cs("cross_sectional_zscore"),
        "cross_section",
        1,
    ),
    "winsorize": _spec(
        "winsorize", "cross_section", 1, _cs("winsorize"), "cross_section", 1
    ),
    "winsorize_zscore": _spec(
        "winsorize_zscore",
        "cross_section",
        1,
        _cs("winsorize_zscore"),
        "cross_section",
        1,
    ),
    "group_rank": _spec(
        "group_rank",
        "group_cross_section",
        1,
        _group_op("group_rank"),
        "group_cross_section",
        2,
        {"group": "allowed group", "min_group_size": "positive integer"},
    ),
    "group_zscore": _spec(
        "group_zscore",
        "group_cross_section",
        1,
        _group_op("group_zscore"),
        "group_cross_section",
        2,
        {"group": "allowed group", "min_group_size": "positive integer"},
    ),
    "group_neutralize": _spec(
        "group_neutralize",
        "group_cross_section",
        1,
        _group_op("group_neutralize"),
        "group_cross_section",
        2,
        {"group": "allowed group", "min_group_size": "positive integer"},
    ),
}


def operator_catalog() -> list[dict[str, object]]:
    return [
        {
            "name": spec.name,
            "category": spec.category,
            "arity": spec.arity,
            "scope": spec.scope,
            "parameters": dict(spec.parameter_schema),
            "complexity_cost": spec.complexity_cost,
            "commutative": spec.commutative,
            "description": spec.description,
        }
        for spec in OPERATOR_REGISTRY.values()
    ]
