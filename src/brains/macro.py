"""Cycle-level research routing inspired by XALPHA's Macro Brain."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from src.core.schemas import MECHANISMS, FactorArtifact, ResearchMemory, ResearchPlan


class DeterministicMacroBrain:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.budgets = {
            key.upper(): int(value) for key, value in config.get("budgets", {}).items()
        }

    def _budget(self, action: str, remaining: int) -> int:
        defaults = {"IMPROVE": 4, "COMBINE": 3, "PIVOT": 6, "STOP": 0}
        return min(remaining, self.budgets.get(action, defaults[action]))

    def coverage_status(self, memory: ResearchMemory) -> dict[str, object]:
        minimum = int(self.config.get("min_candidates_per_mechanism", 2))
        required = min(
            len(MECHANISMS),
            int(self.config.get("minimum_mechanisms_before_stop", 6)),
        )
        tested = {
            mechanism: int(memory.mechanism_stats.get(mechanism, {}).get("tested", 0))
            for mechanism in MECHANISMS
        }
        covered = [
            mechanism for mechanism in MECHANISMS if tested[mechanism] >= minimum
        ]
        undercovered = [
            mechanism for mechanism in MECHANISMS if tested[mechanism] < minimum
        ]
        return {
            "minimum_per_mechanism": minimum,
            "required_mechanisms": required,
            "tested": tested,
            "covered": covered,
            "undercovered": undercovered,
            "satisfied": len(covered) >= required,
        }

    def coverage_satisfied(self, memory: ResearchMemory) -> bool:
        return bool(self.coverage_status(memory)["satisfied"])

    def plan(
        self,
        memory: ResearchMemory,
        parent_pool: dict[str, FactorArtifact],
        elite_archive: dict[str, FactorArtifact],
        remaining_budget: int,
        parent_correlations: dict[tuple[str, str], float] | None = None,
    ) -> ResearchPlan:
        if remaining_budget <= 0:
            return self._stop("Candidate budget exhausted.")
        coverage = self.coverage_status(memory)
        if not coverage["satisfied"]:
            tested = coverage["tested"]
            order = {mechanism: index for index, mechanism in enumerate(MECHANISMS)}
            mechanism = min(
                coverage["undercovered"],
                key=lambda item: (tested[item], order[item]),
            )
            return ResearchPlan(
                "PIVOT",
                mechanism.lower(),
                mechanism,
                "Build minimum evidence across distinct economic mechanisms before exploitation or stopping.",
                (),
                "Mandatory mechanism-coverage phase is incomplete.",
                min(
                    remaining_budget,
                    int(coverage["minimum_per_mechanism"]) - int(tested[mechanism]),
                ),
            )
        min_evidence = int(self.config.get("minimum_evidence_before_stop", 10))
        stop_rounds = int(self.config.get("stop_rounds_without_improvement", 2))
        if (
            len(memory.recent_experiments) >= min_evidence
            and memory.rounds_without_improvement >= stop_rounds
        ):
            return self._stop(
                "No elite improvement within the precommitted patience window."
            )

        tested = coverage["tested"]
        order = {mechanism: index for index, mechanism in enumerate(MECHANISMS)}
        least_tested = min(MECHANISMS, key=lambda item: (tested[item], order[item]))
        if not parent_pool:
            return ResearchPlan(
                "PIVOT",
                least_tested.lower(),
                least_tested,
                "Establish a viable parent in the least-tested economic mechanism.",
                (),
                "No eligible parent exists yet.",
                self._budget("PIVOT", remaining_budget),
            )

        current = memory.current_theme.upper() if memory.current_theme else None
        current_failures = int(
            memory.mechanism_stats.get(current or "", {}).get("retired", 0)
        )
        current_has_parent = any(
            artifact.mechanism == current for artifact in parent_pool.values()
        )
        if memory.rounds_without_improvement >= 2 or (
            current_failures >= int(self.config.get("pivot_failure_threshold", 3))
            and not current_has_parent
        ):
            return ResearchPlan(
                "PIVOT",
                least_tested.lower(),
                least_tested,
                "Move to a less explored mechanism after a stalled research round.",
                (),
                "Recent evidence did not improve the elite archive.",
                self._budget("PIVOT", remaining_budget),
            )

        ranked = sorted(
            parent_pool.values(), key=lambda item: item.quality, reverse=True
        )
        if len(ranked) >= 2:
            corr_limit = float(self.config.get("combine_max_parent_correlation", 0.50))

            def can_combine(left: FactorArtifact, right: FactorArtifact) -> bool:
                if left.mechanism == right.mechanism:
                    return False
                if parent_correlations is None:
                    return True
                key = tuple(sorted((left.spec.factor_id, right.spec.factor_id)))
                return abs(parent_correlations.get(key, 1.0)) < corr_limit

            pair = next(
                (
                    (left, right)
                    for index, left in enumerate(ranked)
                    for right in ranked[index + 1 :]
                    if can_combine(left, right)
                ),
                None,
            )
            if (
                pair
                and memory.recent_round_summaries
                and memory.recent_round_summaries[-1].get("action") != "COMBINE"
            ):
                return ResearchPlan(
                    "COMBINE",
                    "cross_mechanism",
                    "CROSS_DOMAIN_REGIME",
                    "Test whether two independently useful mechanisms have conditional incremental value.",
                    (pair[0].spec.factor_id, pair[1].spec.factor_id),
                    "Two complementary parent mechanisms are available.",
                    self._budget("COMBINE", remaining_budget),
                )

        best = ranked[0]
        return ResearchPlan(
            "IMPROVE",
            best.mechanism.lower(),
            best.mechanism,
            "Repair the strongest parent's main weakness without changing its economic mechanism.",
            (best.spec.factor_id,),
            "The current parent is promising but not yet robust enough.",
            self._budget("IMPROVE", remaining_budget),
        )

    def _stop(self, reason: str) -> ResearchPlan:
        return ResearchPlan(
            "STOP",
            "complete",
            "PRICE_TREND",
            "Stop research and freeze the validated evidence.",
            (),
            reason,
            0,
        )


class LLMResearchPlanProposal(BaseModel):
    action: Literal["IMPROVE", "COMBINE", "PIVOT", "STOP"]
    mechanism: Literal[
        "PRICE_TREND",
        "PRICE_REVERSAL",
        "VOLATILITY",
        "PRICE_VOLUME",
        "FUNDAMENTAL_VALUE",
        "FUNDAMENTAL_QUALITY",
        "NEWS_ATTENTION",
        "CROSS_DOMAIN_REGIME",
    ]
    theme: str
    hypothesis_goal: str
    parent_ids: list[str]
    reason: str
    candidate_budget: int

    class Config:
        extra = "forbid"


@dataclass(frozen=True)
class MacroPlanResult:
    plan: ResearchPlan | None
    used_llm: bool
    reason: str
    rejected: tuple[str, ...] = ()
    response_id: str | None = None
    model: str | None = None
    usage: dict = field(default_factory=dict)


MACRO_SYSTEM_CONTRACT = """You are the cycle-level planning policy for a constrained
equity-factor research system. You choose only a research action, mechanism, eligible
parents, and a small candidate budget. You never write factor formulas or executable code.

Use only supplied research evidence. The final holdout and raw observations are unavailable;
never request or infer them. IMPROVE requires exactly one eligible parent. COMBINE requires
exactly two eligible parents from complementary mechanisms whose supplied absolute
correlation is below the limit. PIVOT requires no parent and should target an under-tested
mechanism. STOP requires adequate total evidence, completed mechanism coverage, and stalled
progress, unless the candidate budget is exhausted. Prefer a targeted repair over cosmetic
parameter search. Return only the structured plan."""


class LLMMacroBrain:
    """Structured Luna planner with deterministic fail-closed validation."""

    def __init__(self, llm_config: dict, research_config: dict, *, client=None) -> None:
        self.llm_config = llm_config
        self.research_config = research_config
        self._client = client

    @property
    def model(self) -> str:
        model_env = self.llm_config.get("model_env", "OPENAI_FACTOR_MODEL")
        return os.environ.get(model_env, self.llm_config.get("model", "gpt-5.6-luna"))

    def _api_key(self) -> str | None:
        return os.environ.get(self.llm_config.get("api_key_env", "OPENAI_API_KEY"))

    def _get_client(self):
        if self._client is not None:
            return self._client
        from openai import OpenAI

        return OpenAI(api_key=self._api_key())

    def _context(
        self,
        memory: ResearchMemory,
        parent_pool: dict[str, FactorArtifact],
        elite_archive: dict[str, FactorArtifact],
        remaining_budget: int,
        parent_correlations: dict[tuple[str, str], float],
    ) -> dict:
        parents = {}
        for factor_id, artifact in sorted(parent_pool.items()):
            record = artifact.record
            parents[factor_id] = {
                "mechanism": artifact.mechanism,
                "formula": artifact.spec.canonical_formula,
                "hypothesis": artifact.spec.hypothesis,
                "tier": "ELITE" if factor_id in elite_archive else "PARENT",
                "mean_rank_ic": record.mean_rank_ic,
                "newey_west_tstat": record.ic_tstat,
                "positive_folds": record.positive_fold_count,
                "high_cost_sharpe": record.high_cost_sharpe,
                "turnover": record.turnover,
                "failure_codes": list(record.failure_codes),
            }
        correlations = [
            {"left": pair[0], "right": pair[1], "absolute_correlation": abs(value)}
            for pair, value in sorted(parent_correlations.items())
        ]
        return {
            "remaining_candidate_budget": remaining_budget,
            "mechanism_stats": memory.mechanism_stats,
            "recent_rounds": memory.recent_round_summaries[-6:],
            "good_lessons": memory.good_lessons[-8:],
            "bad_lessons": memory.bad_lessons[-8:],
            "rounds_without_improvement": memory.rounds_without_improvement,
            "eligible_parents": parents,
            "parent_correlations": correlations,
            "constraints": {
                "allowed_actions": ["IMPROVE", "COMBINE", "PIVOT", "STOP"],
                "allowed_mechanisms": list(MECHANISMS),
                "combine_max_absolute_correlation": float(
                    self.research_config.get("combine_max_parent_correlation", 0.50)
                ),
                "action_candidate_caps": {
                    key.upper(): int(value)
                    for key, value in self.research_config.get("budgets", {}).items()
                },
                "raw_data_available": False,
                "holdout_available": False,
            },
        }

    def propose(
        self,
        memory: ResearchMemory,
        parent_pool: dict[str, FactorArtifact],
        elite_archive: dict[str, FactorArtifact],
        remaining_budget: int,
        parent_correlations: dict[tuple[str, str], float],
        deterministic_plan: ResearchPlan,
    ) -> MacroPlanResult:
        if not self.llm_config.get("enabled", False):
            return MacroPlanResult(None, False, "llm_disabled")
        if self._client is None and not self._api_key():
            if self.llm_config.get("required", False):
                raise RuntimeError("LLM Macro planning requires OPENAI_API_KEY.")
            return MacroPlanResult(None, False, "missing_api_key")
        try:
            response = self._get_client().responses.parse(
                model=self.model,
                instructions=MACRO_SYSTEM_CONTRACT,
                input=json.dumps(
                    self._context(
                        memory,
                        parent_pool,
                        elite_archive,
                        remaining_budget,
                        parent_correlations,
                    ),
                    sort_keys=True,
                    default=str,
                ),
                text_format=LLMResearchPlanProposal,
                reasoning={
                    "mode": self.llm_config.get("reasoning_mode", "standard"),
                    "effort": self.llm_config.get("reasoning_effort", "low"),
                },
                text={"verbosity": self.llm_config.get("verbosity", "low")},
                max_output_tokens=min(
                    int(self.llm_config.get("max_output_tokens", 6000)), 2500
                ),
                store=bool(self.llm_config.get("store", False)),
                truncation="disabled",
                prompt_cache_key=f"{self.llm_config.get('prompt_cache_key', 'factor-autoresearch-v1')}-macro",
            )
            proposal = response.output_parsed
        except Exception as error:
            if self.llm_config.get("required", False):
                raise
            return MacroPlanResult(None, False, f"api_error:{type(error).__name__}")
        if proposal is None:
            return MacroPlanResult(None, False, "model_refusal_or_empty_output")

        rejection = self._validate(
            proposal,
            parent_pool,
            remaining_budget,
            parent_correlations,
            deterministic_plan,
        )
        usage_object = getattr(response, "usage", None)
        usage = usage_object.model_dump() if hasattr(usage_object, "model_dump") else {}
        metadata = {
            "response_id": getattr(response, "id", None),
            "model": self.model,
            "usage": usage,
        }
        if rejection:
            return MacroPlanResult(
                None,
                False,
                "invalid_llm_plan",
                (rejection,),
                **metadata,
            )
        plan = ResearchPlan(
            proposal.action,
            proposal.theme,
            proposal.mechanism,
            proposal.hypothesis_goal,
            tuple(proposal.parent_ids),
            proposal.reason,
            proposal.candidate_budget,
        )
        return MacroPlanResult(plan, True, "structured_llm_plan", **metadata)

    def _validate(
        self,
        proposal: LLMResearchPlanProposal,
        parent_pool: dict[str, FactorArtifact],
        remaining_budget: int,
        parent_correlations: dict[tuple[str, str], float],
        deterministic_plan: ResearchPlan,
    ) -> str | None:
        caps = {"IMPROVE": 4, "COMBINE": 3, "PIVOT": 6, "STOP": 0}
        caps.update(
            {
                key.upper(): int(value)
                for key, value in self.research_config.get("budgets", {}).items()
            }
        )
        if proposal.candidate_budget < 0 or proposal.candidate_budget > min(
            remaining_budget, caps[proposal.action]
        ):
            return "candidate budget exceeds the action or remaining-budget cap"
        if proposal.action == "STOP":
            if deterministic_plan.action != "STOP":
                return (
                    "STOP is not yet permitted by deterministic coverage/patience gates"
                )
            if proposal.parent_ids or proposal.candidate_budget != 0:
                return "STOP cannot carry parents or candidate budget"
            return None
        if proposal.candidate_budget < 1:
            return "non-STOP action requires a positive candidate budget"
        if proposal.action == "PIVOT":
            return "PIVOT cannot carry parents" if proposal.parent_ids else None
        if any(parent_id not in parent_pool for parent_id in proposal.parent_ids):
            return "plan names an ineligible parent"
        if proposal.action == "IMPROVE":
            if len(proposal.parent_ids) != 1:
                return "IMPROVE requires exactly one eligible parent"
            parent = parent_pool[proposal.parent_ids[0]]
            if parent.mechanism != proposal.mechanism:
                return "IMPROVE mechanism must match its parent mechanism"
            return None
        if len(proposal.parent_ids) != 2:
            return "COMBINE requires exactly two eligible parents"
        left, right = (parent_pool[parent_id] for parent_id in proposal.parent_ids)
        if left.mechanism == right.mechanism:
            return "COMBINE parents must use complementary mechanisms"
        key = tuple(sorted(proposal.parent_ids))
        corr = abs(parent_correlations.get(key, 1.0))
        if corr >= float(
            self.research_config.get("combine_max_parent_correlation", 0.50)
        ):
            return "COMBINE parent correlation exceeds the configured limit"
        if proposal.mechanism != "CROSS_DOMAIN_REGIME":
            return "COMBINE must target CROSS_DOMAIN_REGIME"
        return None
