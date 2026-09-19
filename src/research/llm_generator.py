"""Restricted LLM proposal adapter for typed, auditable FactorSpec candidates."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from src.factors.schema import FactorSpec
from src.research.memory import ResearchState
from src.utils.logging import ExperimentRecord


class FactorProposal(BaseModel):
    factor_id: str
    parent_ids: list[str] = Field(default_factory=list)
    family: Literal["price", "fundamental", "news", "interaction"]
    hypothesis: str
    base_feature: Literal[
        "return",
        "volatility",
        "volume_shock",
        "distance_to_high",
        "after_tax_roe",
        "operating_margin",
        "profit_margin",
        "earnings_yield",
        "asset_growth",
        "news_volume",
    ]
    ts_operator: Literal["identity", "rolling_mean", "vol_adjust"] | None = None
    window: int | None = None
    cs_operator: Literal[
        "rank", "zscore", "winsorize", "winsorize_zscore", "sign"
    ] | None = None
    interaction_feature: Literal[
        "return",
        "volatility",
        "volume_shock",
        "distance_to_high",
        "after_tax_roe",
        "operating_margin",
        "profit_margin",
        "earnings_yield",
        "asset_growth",
        "news_volume",
    ] | None = None
    interaction_window: int | None = None
    direction: Literal[-1, 1]
    mutation_reason: str

    class Config:
        extra = "forbid"


class FactorProposalBatch(BaseModel):
    research_summary: str
    proposals: list[FactorProposal]

    class Config:
        extra = "forbid"


@dataclass(frozen=True)
class LLMGenerationResult:
    candidates: tuple[FactorSpec, ...]
    used_llm: bool
    reason: str
    rejected: tuple[str, ...] = field(default_factory=tuple)


SYSTEM_CONTRACT = """You are a hypothesis generator, not a backtester.
Return only the structured FactorProposalBatch requested by the schema.
Use only the supplied primitive/operator catalog and promoted parent IDs.
Every proposal must state an ex-ante economic mechanism and why it addresses prior evidence.
Do not reference, infer, or request 2016 holdout results.
Do not invent columns, functions, metrics, or Python code.
Prefer simple interpretable expressions and avoid formulas already tested.
Balance exploitation of promoted parents with exploration of weakly tested families.
If turnover is high, consider longer windows or smoothing. If stability is weak, simplify.
If redundancy is high, change the information family rather than a cosmetic parameter."""


class LLMFactorGenerator:
    """Calls an LLM for proposals, then applies deterministic local validation."""

    def __init__(self, config: dict, *, client=None) -> None:
        self.config = config
        self._client = client

        provider = self.config.get("provider", "openai")
        if provider != "openai":
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", False))

    @property
    def model(self) -> str:
        model_env = self.config.get("model_env", "OPENAI_FACTOR_MODEL")
        return os.environ.get(model_env, self.config.get("model", "gpt-4o-mini"))

    def _api_key(self) -> str | None:
        return os.environ.get(self.config.get("api_key_env", "OPENAI_API_KEY"))

    def _get_client(self):
        if self._client is not None:
            return self._client
        from openai import OpenAI

        return OpenAI(api_key=self._api_key())

    @staticmethod
    def _record_summary(record: ExperimentRecord) -> dict:
        return {
            "factor_id": record.factor_id,
            "family": record.family,
            "formula": record.canonical_formula,
            "decision": record.decision,
            "reasons": list(record.reasons),
            "mean_rank_ic": record.mean_rank_ic,
            "ic_tstat": record.ic_tstat,
            "positive_fold_count": record.positive_fold_count,
            "net_sharpe": record.long_short_sharpe,
            "high_cost_sharpe": record.high_cost_sharpe,
            "turnover": record.turnover,
            "max_drawdown": record.max_drawdown,
            "redundancy_corr": record.redundancy_corr,
            "residual_ic": record.residual_ic,
        }

    def build_context(
        self,
        *,
        generation: int,
        state: ResearchState,
        recent_records: list[ExperimentRecord],
        promoted_specs: dict[str, FactorSpec],
    ) -> dict:
        history_limit = int(self.config.get("max_history_records", 20))
        selected_records = sorted(
            recent_records,
            key=lambda record: (
                {"PROMOTE": 0, "HOLD": 1, "RETIRE": 2}[record.decision],
                -(
                    record.mean_rank_ic
                    if record.mean_rank_ic is not None
                    else float("-inf")
                ),
            ),
        )[:history_limit]
        return {
            "generation_to_propose": generation,
            "research_state": {
                "promoted_factor_ids": state.promoted_factor_ids[-20:],
                "held_factor_ids": state.held_factor_ids[-20:],
                "retired_factor_ids": state.retired_factor_ids[-20:],
                "family_summary": state.family_summary,
                "best_patterns": state.best_patterns[-10:],
                "failed_patterns": state.failed_patterns[-10:],
                "unexplored_families": state.unexplored_families,
                "search_budget_remaining": state.search_budget_remaining,
            },
            "promoted_parent_specs": {
                factor_id: spec.to_dict() for factor_id, spec in promoted_specs.items()
            },
            "recent_experiment_records": [
                self._record_summary(record) for record in selected_records
            ],
            "catalog": {
                "base_features": [
                    "return",
                    "volatility",
                    "volume_shock",
                    "distance_to_high",
                    "after_tax_roe",
                    "operating_margin",
                    "profit_margin",
                    "earnings_yield",
                    "asset_growth",
                    "news_volume",
                ],
                "time_series_operators": ["identity", "rolling_mean", "vol_adjust"],
                "cross_sectional_operators": [
                    "rank",
                    "zscore",
                    "winsorize",
                    "winsorize_zscore",
                    "sign",
                ],
                "allowed_windows": self.config.get(
                    "allowed_windows", [1, 5, 10, 20, 60]
                ),
                "max_complexity": int(self.config.get("max_complexity", 4)),
                "max_proposals": int(
                    self.config.get("max_proposals_per_generation", 6)
                ),
            },
            "constraints": {
                "holdout_year": "withheld; never reference or infer it",
                "raw_data_available_to_llm": False,
                "proposal_mix": "roughly 60-80% exploitation and 20-40% exploration",
                "forbidden": [
                    "future returns",
                    "future windows",
                    "full-sample normalization",
                    "arbitrary Python",
                    "new columns or operators",
                ],
            },
        }

    def _validate_proposals(
        self,
        batch: FactorProposalBatch,
        *,
        generation: int,
        promoted_specs: dict[str, FactorSpec],
        tested_ids: set[str],
        prior_formulas: set[str],
    ) -> tuple[list[FactorSpec], list[str]]:
        candidates: list[FactorSpec] = []
        rejected: list[str] = []
        allowed_windows = set(self.config.get("allowed_windows", [1, 5, 10, 20, 60]))
        max_complexity = int(self.config.get("max_complexity", 4))
        max_total = int(self.config.get("max_proposals_per_generation", 6))
        max_per_parent = int(self.config.get("max_proposals_per_parent", 3))
        parent_counts: dict[str, int] = {}
        seen_ids = set(tested_ids)
        seen_formulas = set(prior_formulas)
        windowed = {
            "return",
            "volatility",
            "volume_shock",
            "distance_to_high",
            "news_volume",
        }

        for proposal in batch.proposals[:max_total]:
            reason_prefix = proposal.factor_id or "unnamed"
            if not re.fullmatch(r"[A-Za-z0-9_]+", proposal.factor_id):
                rejected.append(f"{reason_prefix}: invalid factor_id")
                continue
            if proposal.factor_id in seen_ids:
                rejected.append(f"{reason_prefix}: duplicate factor_id")
                continue
            if len(proposal.hypothesis.strip()) < 20:
                rejected.append(f"{reason_prefix}: hypothesis is too short")
                continue
            if len(proposal.parent_ids) > 2 or any(
                parent not in promoted_specs for parent in proposal.parent_ids
            ):
                rejected.append(f"{reason_prefix}: parent is not in promoted library")
                continue
            if (
                proposal.base_feature in windowed
                and proposal.window not in allowed_windows
            ):
                rejected.append(f"{reason_prefix}: unsupported base window")
                continue
            if proposal.base_feature not in windowed and proposal.window is not None:
                rejected.append(
                    f"{reason_prefix}: fundamental base cannot have a window"
                )
                continue
            if proposal.interaction_feature in windowed:
                if proposal.interaction_window not in allowed_windows:
                    rejected.append(f"{reason_prefix}: unsupported interaction window")
                    continue
            elif (
                proposal.interaction_feature is not None
                and proposal.interaction_window is not None
            ):
                rejected.append(
                    f"{reason_prefix}: fundamental interaction cannot have a window"
                )
                continue
            if (
                proposal.interaction_feature is None
                and proposal.interaction_window is not None
            ):
                rejected.append(f"{reason_prefix}: interaction window without feature")
                continue
            if proposal.parent_ids:
                parent = proposal.parent_ids[0]
                if parent_counts.get(parent, 0) >= max_per_parent:
                    rejected.append(
                        f"{reason_prefix}: per-parent proposal cap exceeded"
                    )
                    continue

            try:
                spec = FactorSpec(
                    factor_id=proposal.factor_id,
                    generation=generation,
                    parent_ids=tuple(proposal.parent_ids),
                    family=proposal.family,
                    hypothesis=proposal.hypothesis,
                    base_feature=proposal.base_feature,
                    ts_operator=proposal.ts_operator,
                    window=proposal.window,
                    cs_operator=proposal.cs_operator,
                    interaction_feature=proposal.interaction_feature,
                    interaction_window=proposal.interaction_window,
                    direction=proposal.direction,
                    mutation_reason=proposal.mutation_reason,
                    demonstration_only=False,
                )
            except (TypeError, ValueError) as error:
                rejected.append(f"{reason_prefix}: FactorSpec rejected: {error}")
                continue
            if spec.complexity > max_complexity:
                rejected.append(f"{reason_prefix}: complexity cap exceeded")
                continue
            if spec.canonical_formula in seen_formulas:
                rejected.append(f"{reason_prefix}: duplicate canonical formula")
                continue
            candidates.append(spec)
            seen_ids.add(spec.factor_id)
            seen_formulas.add(spec.canonical_formula)
            for parent in spec.parent_ids:
                parent_counts[parent] = parent_counts.get(parent, 0) + 1
        return candidates, rejected

    def propose(
        self,
        *,
        generation: int,
        state: ResearchState,
        recent_records: list[ExperimentRecord],
        promoted_specs: dict[str, FactorSpec],
        tested_ids: set[str],
    ) -> LLMGenerationResult:
        if not self.enabled:
            return LLMGenerationResult((), False, "llm_disabled")
        if self._client is None and not self._api_key():
            if self.config.get("required", False):
                raise RuntimeError(
                    "LLM generation is required but OPENAI_API_KEY is absent."
                )
            return LLMGenerationResult((), False, "missing_api_key")

        context = self.build_context(
            generation=generation,
            state=state,
            recent_records=recent_records,
            promoted_specs=promoted_specs,
        )
        try:
            completion = self._get_client().beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_CONTRACT},
                    {
                        "role": "user",
                        "content": json.dumps(context, sort_keys=True, default=str),
                    },
                ],
                response_format=FactorProposalBatch,
            )
            batch = completion.choices[0].message.parsed
        except (
            Exception
        ) as error:  # API failures must not corrupt deterministic research.
            if self.config.get("required", False):
                raise
            return LLMGenerationResult((), False, f"api_error:{type(error).__name__}")
        if batch is None:
            return LLMGenerationResult((), False, "model_refusal_or_empty_output")

        prior_formulas = (
            {record.canonical_formula for record in recent_records}
            | set(state.best_patterns)
            | set(state.failed_patterns)
        )
        candidates, rejected = self._validate_proposals(
            batch,
            generation=generation,
            promoted_specs=promoted_specs,
            tested_ids=tested_ids,
            prior_formulas=prior_formulas,
        )
        if not candidates:
            return LLMGenerationResult(
                (), False, "no_valid_llm_proposals", tuple(rejected)
            )
        return LLMGenerationResult(
            tuple(candidates), True, "structured_llm_proposals", tuple(rejected)
        )
