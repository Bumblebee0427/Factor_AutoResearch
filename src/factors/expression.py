"""Immutable, JSON-serializable expression AST for Factor DSL V2."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias


def _number(value: float) -> str:
    value = float(value)
    if value == 0:
        return "0"
    if value.is_integer():
        return str(int(value))
    return format(value, ".12g")


class ExprBase:
    """Shared behavior implemented without exposing executable code."""

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def canonical(self) -> str:
        raise NotImplementedError

    def complexity(self) -> int:
        raise NotImplementedError

    def required_history(self) -> int:
        raise NotImplementedError

    def required_features(self) -> frozenset[str]:
        raise NotImplementedError

    def depth(self) -> int:
        raise NotImplementedError

    def operator_counts(self) -> dict[str, int]:
        raise NotImplementedError

    def expression_hash(self) -> str:
        """Stable identity for the canonical expression."""
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "Expr":
        return expr_from_dict(payload)


@dataclass(frozen=True)
class Feature(ExprBase):
    name: str
    window: int | None = None

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Feature name must be a non-empty string.")
        if self.window is not None and (
            isinstance(self.window, bool) or int(self.window) != self.window
        ):
            raise ValueError("Feature window must be an integer or None.")
        if self.window is not None and self.window < 1:
            raise ValueError("Feature window must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "feature", "name": self.name, "window": self.window}

    def canonical(self) -> str:
        return f"{self.name}[{self.window}]" if self.window is not None else self.name

    def complexity(self) -> int:
        return 0

    def required_history(self) -> int:
        return int(self.window or 1)

    def required_features(self) -> frozenset[str]:
        return frozenset({self.name})

    def depth(self) -> int:
        return 0

    def operator_counts(self) -> dict[str, int]:
        return {}


@dataclass(frozen=True)
class Constant(ExprBase):
    value: float

    def __post_init__(self) -> None:
        numeric = float(self.value)
        if not math.isfinite(numeric):
            raise ValueError("Constant must be finite.")
        if numeric < -100 or numeric > 100:
            raise ValueError("Constant must be within [-100, 100].")
        object.__setattr__(self, "value", numeric)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "constant", "value": self.value}

    def canonical(self) -> str:
        return _number(self.value)

    def complexity(self) -> int:
        return 0

    def required_history(self) -> int:
        return 1

    def required_features(self) -> frozenset[str]:
        return frozenset()

    def depth(self) -> int:
        return 0

    def operator_counts(self) -> dict[str, int]:
        return {}


@dataclass(frozen=True)
class Op(ExprBase):
    name: str
    args: tuple["Expr", ...]
    params: Mapping[str, object] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        from src.factors.operator_registry import OPERATOR_REGISTRY

        if self.name not in OPERATOR_REGISTRY:
            raise ValueError(f"Unsupported DSL operator: {self.name}")
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(
            self,
            "params",
            MappingProxyType(dict(self.params or {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "op",
            "name": self.name,
            "args": [argument.to_dict() for argument in self.args],
            "params": dict(self.params),
        }

    def canonical(self) -> str:
        from src.factors.operator_registry import OPERATOR_REGISTRY

        children = [argument.canonical() for argument in self.args]
        if OPERATOR_REGISTRY[self.name].commutative:
            children.sort()
        rendered = ",".join(children)
        if self.params:
            rendered_params = ",".join(
                f"{key}={_number(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value}"
                for key, value in sorted(self.params.items())
            )
            rendered = f"{rendered},{rendered_params}" if rendered else rendered_params
        return f"{self.name}({rendered})"

    def complexity(self) -> int:
        from src.factors.operator_registry import OPERATOR_REGISTRY

        return OPERATOR_REGISTRY[self.name].complexity_cost + sum(
            argument.complexity() for argument in self.args
        )

    def required_history(self) -> int:
        child_history = max(
            (argument.required_history() for argument in self.args), default=1
        )
        if self.name in {"lag", "delta"}:
            return child_history + int(self.params.get("periods", 1))
        if self.name in {
            "rolling_sum",
            "rolling_mean",
            "rolling_std",
            "rolling_min",
            "rolling_max",
            "ts_rank",
            "ts_zscore",
            "decay_linear",
        }:
            return child_history + int(self.params.get("window", 1)) - 1
        if self.name in {"rolling_corr", "rolling_cov"}:
            return child_history + int(self.params.get("window", 1)) - 1
        if self.name == "ewma":
            return child_history + int(math.ceil(float(self.params.get("halflife", 1))))
        return child_history

    def required_features(self) -> frozenset[str]:
        features: set[str] = set()
        for argument in self.args:
            features.update(argument.required_features())
        return frozenset(features)

    def depth(self) -> int:
        return 1 + max((argument.depth() for argument in self.args), default=-1)

    def operator_counts(self) -> dict[str, int]:
        counts = {self.name: 1}
        for argument in self.args:
            for name, count in argument.operator_counts().items():
                counts[name] = counts.get(name, 0) + count
        return counts


Expr: TypeAlias = Feature | Constant | Op


def expr_from_dict(payload: Mapping[str, Any]) -> Expr:
    kind = payload.get("kind")
    if kind == "feature":
        return Feature(str(payload["name"]), payload.get("window"))
    if kind == "constant":
        return Constant(float(payload["value"]))
    if kind == "op":
        return Op(
            str(payload["name"]),
            tuple(expr_from_dict(item) for item in payload.get("args", [])),
            dict(payload.get("params", {})),
        )
    raise ValueError(f"Unknown expression node kind: {kind!r}")


def expression_hash(expr: Expr) -> str:
    return expr.expression_hash()


def expression_to_json(expr: Expr) -> str:
    return json.dumps(expr.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ExpressionValidation:
    passed: bool
    reasons: tuple[str, ...]


def validate_expression(
    expression: Expr,
    *,
    allowed_features: set[str],
    allowed_windows: set[int] | None = None,
    allowed_groups: set[str] | None = None,
    max_depth: int = 5,
    max_operator_nodes: int = 6,
    max_rolling_nodes: int = 3,
    max_binary_nodes: int = 2,
    max_group_nodes: int = 2,
) -> ExpressionValidation:
    """Validate structure and parameters without executing an expression."""
    from src.factors.operator_registry import OPERATOR_REGISTRY

    reasons: list[str] = []
    allowed_windows = allowed_windows or {1, 5, 10, 20, 60}
    allowed_groups = allowed_groups or {"sector", "subindustry"}
    binary = {"add", "sub", "mul", "safe_div"}
    rolling = {
        "rolling_sum",
        "rolling_mean",
        "rolling_std",
        "rolling_min",
        "rolling_max",
        "ts_rank",
        "ts_zscore",
        "decay_linear",
        "rolling_corr",
        "rolling_cov",
    }
    groups = {"group_rank", "group_zscore", "group_neutralize"}

    if expression.depth() > max_depth:
        reasons.append(f"AST depth {expression.depth()} exceeds cap {max_depth}.")
    counts = expression.operator_counts()
    if sum(counts.values()) > max_operator_nodes:
        reasons.append(
            f"AST operator count {sum(counts.values())} exceeds cap {max_operator_nodes}."
        )
    if (
        sum(count for name, count in counts.items() if name in rolling)
        > max_rolling_nodes
    ):
        reasons.append("AST rolling operator count exceeds configured cap.")
    if (
        sum(count for name, count in counts.items() if name in binary)
        > max_binary_nodes
    ):
        reasons.append("AST binary operator count exceeds configured cap.")
    if sum(count for name, count in counts.items() if name in groups) > max_group_nodes:
        reasons.append("AST group operator count exceeds configured cap.")

    def visit(node: Expr) -> None:
        if isinstance(node, Feature):
            if node.name not in allowed_features:
                reasons.append(f"Unknown feature: {node.name}")
            if node.name == "future_return":
                reasons.append("future_return is a target, not an input.")
            windowed = {
                "return",
                "volatility",
                "volume_shock",
                "distance_to_high",
                "news_volume",
            }
            if node.name in windowed:
                if node.window not in allowed_windows:
                    reasons.append(
                        f"Unsupported feature window for {node.name}: {node.window}"
                    )
            elif node.window is not None:
                reasons.append(f"Feature {node.name} cannot have a primitive window.")
            return
        if isinstance(node, Constant):
            return
        operator = OPERATOR_REGISTRY.get(node.name)
        if operator is None:
            reasons.append(f"Unknown operator: {node.name}")
            return
        if isinstance(operator.arity, int) and len(node.args) != operator.arity:
            reasons.append(
                f"Operator {node.name} expects {operator.arity} arguments, got {len(node.args)}."
            )
        expected = set(operator.parameter_schema)
        unknown = set(node.params) - expected
        if unknown:
            reasons.append(f"Unknown parameters for {node.name}: {sorted(unknown)}")
        if node.name in rolling | {"lag", "delta"}:
            parameter = "periods" if node.name in {"lag", "delta"} else "window"
            value = node.params.get(parameter)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                reasons.append(f"{node.name} requires a positive integer {parameter}.")
            elif node.name not in {"lag", "delta"} and value not in allowed_windows:
                reasons.append(f"Unsupported transform window for {node.name}: {value}")
        if node.name == "ewma":
            if float(node.params.get("halflife", 0)) <= 0:
                reasons.append("ewma requires halflife > 0.")
        if node.name == "safe_div" and float(node.params.get("eps", 1.0e-12)) <= 0:
            reasons.append("safe_div eps must be positive.")
        if node.name == "clip":
            lower = node.params.get("lower")
            upper = node.params.get("upper")
            if lower is None or upper is None or not float(lower) < float(upper):
                reasons.append("clip requires finite lower < upper bounds.")
        if node.name in groups:
            group = node.params.get("group")
            if group not in allowed_groups:
                reasons.append(f"Unknown group: {group}")
            minimum = node.params.get("min_group_size", 5)
            if not isinstance(minimum, int) or minimum < 1:
                reasons.append("min_group_size must be a positive integer.")
        for child in node.args:
            visit(child)

    visit(expression)
    return ExpressionValidation(not reasons, tuple(dict.fromkeys(reasons)))
