"""Serializable contracts for the minimal XALPHA-inspired research loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.factors.expression import Op
from src.factors.schema import ALLOWED_BASE_FEATURES, FactorSpec
from src.utils.logging import ExperimentRecord


MECHANISMS = (
    "PRICE_TREND",
    "PRICE_REVERSAL",
    "VOLATILITY",
    "PRICE_VOLUME",
    "FUNDAMENTAL_VALUE",
    "FUNDAMENTAL_QUALITY",
    "NEWS_ATTENTION",
    "CROSS_DOMAIN_REGIME",
)


@dataclass(frozen=True)
class DataContract:
    price_fields: frozenset[str]
    fundamental_fields: frozenset[str]
    news_fields: frozenset[str]
    allowed_transforms: frozenset[str]
    unavailable_fields: frozenset[str]
    group_fields: dict[str, str] = field(default_factory=dict)

    @property
    def available_features(self) -> frozenset[str]:
        return self.price_fields | self.fundamental_fields | self.news_fields

    def supports(self, spec: FactorSpec) -> tuple[bool, tuple[str, ...]]:
        referenced = set(spec.required_features)
        unavailable = referenced & self.unavailable_fields
        unknown = referenced - self.available_features
        reasons = []
        if unavailable:
            reasons.append(f"Unavailable fields requested: {sorted(unavailable)}")
        if unknown:
            reasons.append(f"Fields outside DataContract: {sorted(unknown)}")
        transforms = set(spec.effective_expression.operator_counts())
        transforms |= {spec.ts_operator, spec.cs_operator} - {None, "identity"}
        unsupported = transforms - self.allowed_transforms
        if unsupported:
            reasons.append(f"Unsupported transforms: {sorted(unsupported)}")
        if (
            "group_rank" in transforms
            or "group_zscore" in transforms
            or "group_neutralize" in transforms
        ):
            groups = _expression_groups(spec.effective_expression)
            unknown_groups = groups - set(self.group_fields)
            if unknown_groups:
                reasons.append(f"Unsupported groups: {sorted(unknown_groups)}")
        return not reasons, tuple(reasons)

    @classmethod
    def antelion(cls) -> "DataContract":
        available = set(ALLOWED_BASE_FEATURES)
        price = {
            "return",
            "volatility",
            "volume_shock",
            "distance_to_high",
            "raw_open",
            "raw_high",
            "raw_low",
            "raw_close",
            "raw_volume",
        }
        fundamentals = {
            "after_tax_roe",
            "operating_margin",
            "profit_margin",
            "earnings_yield",
            "asset_growth",
        }
        news = {"news_volume"}
        assert price | fundamentals | news == available
        return cls(
            price_fields=frozenset(price),
            fundamental_fields=frozenset(fundamentals),
            news_fields=frozenset(news),
            allowed_transforms=frozenset(
                {
                    "add",
                    "sub",
                    "mul",
                    "safe_div",
                    "neg",
                    "abs",
                    "clip",
                    "sign",
                    "lag",
                    "delta",
                    "rolling_sum",
                    "rolling_mean",
                    "rolling_std",
                    "rolling_min",
                    "rolling_max",
                    "ts_rank",
                    "ts_zscore",
                    "ewma",
                    "decay_linear",
                    "rolling_corr",
                    "rolling_cov",
                    "cs_rank",
                    "cs_zscore",
                    "vol_adjust",
                    "rank",
                    "zscore",
                    "winsorize",
                    "winsorize_zscore",
                    "sign",
                    "group_rank",
                    "group_zscore",
                    "group_neutralize",
                }
            ),
            unavailable_fields=frozenset(
                {
                    "analyst_revision",
                    "order_book_imbalance",
                    "bid_ask_spread",
                    "market_cap",
                    "future_return",
                }
            ),
            group_fields={
                "sector": "gics_sector",
                "subindustry": "gics_sub_industry",
            },
        )


def _expression_groups(expression) -> set[str]:
    groups: set[str] = set()
    if isinstance(expression, Op):
        if expression.name in {"group_rank", "group_zscore", "group_neutralize"}:
            group = expression.params.get("group")
            if group is not None:
                groups.add(str(group))
        for child in expression.args:
            groups.update(_expression_groups(child))
    return groups


@dataclass(frozen=True)
class ResearchPlan:
    action: str
    theme: str
    mechanism: str
    hypothesis_goal: str
    parent_ids: tuple[str, ...]
    reason: str
    candidate_budget: int

    def __post_init__(self) -> None:
        if self.action not in {"IMPROVE", "COMBINE", "PIVOT", "STOP"}:
            raise ValueError(f"Unsupported research action: {self.action}")
        if self.mechanism not in MECHANISMS:
            raise ValueError(f"Unsupported mechanism: {self.mechanism}")
        if self.candidate_budget < 0:
            raise ValueError("candidate_budget cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchOutcome:
    spec: FactorSpec
    record: ExperimentRecord
    tier: str
    mechanism: str
    action: str

    def __post_init__(self) -> None:
        if self.tier not in {"RETIRED", "PARENT", "ELITE"}:
            raise ValueError(f"Unsupported research tier: {self.tier}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.spec.factor_id,
            "round_id": self.spec.generation,
            "generation": self.spec.generation,
            "action": self.action,
            "mechanism": self.mechanism,
            "hypothesis": self.spec.hypothesis,
            "factor_spec": self.spec.to_dict(),
            "integrity_passed": self.record.integrity_passed,
            "integrity_failures": list(self.record.integrity_issues),
            "metrics": {
                "mean_rank_ic": self.record.mean_rank_ic,
                "newey_west_tstat": self.record.ic_tstat,
                "net_sharpe": self.record.long_short_sharpe,
                "high_cost_sharpe": self.record.high_cost_sharpe,
                "turnover": self.record.turnover,
                "max_drawdown": self.record.max_drawdown,
                "fold_ic_sign_consistency": self.record.fold_ic_sign_consistency,
                "multi_horizon_mean_ic": self.record.multi_horizon_mean_ic,
            },
            "decision": self.tier,
            "decision_reasons": list(self.record.reasons),
            "parent_ids": list(self.spec.parent_ids),
        }


@dataclass
class ResearchMemory:
    good_lessons: list[dict[str, Any]] = field(default_factory=list)
    bad_lessons: list[dict[str, Any]] = field(default_factory=list)
    recent_experiments: list[dict[str, Any]] = field(default_factory=list)
    parent_pool_ids: list[str] = field(default_factory=list)
    elite_archive_ids: list[str] = field(default_factory=list)
    mechanism_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    recent_round_summaries: list[dict[str, Any]] = field(default_factory=list)
    current_theme: str | None = None
    current_budget: int = 0
    rounds_without_improvement: int = 0
    proposal_failure_counts: dict[str, int] = field(default_factory=dict)
    stopped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResearchMemory":
        return cls(**payload)


@dataclass(frozen=True)
class FactorArtifact:
    spec: FactorSpec
    record: ExperimentRecord
    mechanism: str

    @property
    def quality(self) -> float:
        ic = self.record.mean_rank_ic or 0.0
        tstat = self.record.ic_tstat or 0.0
        sharpe = self.record.long_short_sharpe or 0.0
        stability = self.record.fold_ic_sign_consistency or 0.0
        return float(ic + 0.002 * tstat + 0.001 * sharpe + 0.002 * stability)
