"""Budget-driven XALPHA-inspired research controller for the constrained DSL."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.brains.cross import CrossBrain
from src.brains.macro import DeterministicMacroBrain, LLMMacroBrain, MacroPlanResult
from src.brains.micro import (
    AdaptiveLLMMicroBrain,
    DeterministicMicroBrain,
    MicroProposalResult,
    infer_mechanism,
)
from src.core.schemas import (
    MECHANISMS,
    DataContract,
    FactorArtifact,
    ResearchMemory,
    ResearchOutcome,
    ResearchPlan,
)
from src.evaluation.redundancy import mean_cross_sectional_correlation
from src.memory.store import ResearchMemoryStore
from src.quality.alignment import check_alignment
from src.quality.dynamic_leakage import check_dynamic_leakage
from src.quality.static_checks import check_factor_spec
from src.research.failure_policy import (
    INTEGRITY_RESPONSES,
    integrity_response_counts,
    valid_group_reproposals,
)
from src.research.loop import ResearchLoop
from src.selection.archive import EliteArchive, ParentPool
from src.selection.diagnostics import diagnose_elite_gates
from src.selection.gates import classify_tier
from src.selection.library import build_final_library


class AdaptiveResearchController:
    def __init__(
        self,
        config: dict,
        panel: pd.DataFrame,
        *,
        use_llm: bool = False,
        macro_client=None,
        micro_client=None,
    ) -> None:
        holdout_year = int(config["walk_forward"]["holdout_year"])
        if panel["date"].dt.year.ge(holdout_year).any():
            raise RuntimeError("Adaptive research cannot load final-holdout rows.")
        self.config = config
        self.panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
        self.settings = dict(config.get("adaptive_research", {}))
        self.base_loop = ResearchLoop(config, self.panel)
        self.contract = DataContract.antelion()
        self.macro = DeterministicMacroBrain(self.settings)
        self.micro = DeterministicMicroBrain()
        self.use_llm = use_llm
        self.llm_macro = LLMMacroBrain(
            config.get("llm", {}), self.settings, client=macro_client
        )
        self.llm_micro = AdaptiveLLMMicroBrain(
            config.get("llm", {}), self.micro, client=micro_client
        )
        self.cross = CrossBrain()
        self.parent_pool = ParentPool(
            int(self.settings.get("parent_pool_size", 20)),
            correlation_threshold=float(
                self.settings.get("parent_cluster_correlation", 0.90)
            ),
            minimum_overlap_dates=int(
                self.settings.get("parent_cluster_min_dates", 20)
            ),
            minimum_names_per_date=int(
                self.settings.get("parent_cluster_min_names", 5)
            ),
            maximum_comparison_dates=int(
                self.settings.get("parent_cluster_max_dates", 96)
            ),
        )
        self.elite_archive = EliteArchive(
            int(self.settings.get("elite_archive_size", 12))
        )
        self.memory = ResearchMemory(
            current_budget=int(
                self.settings.get(
                    "max_candidate_evaluations", config["search"]["max_candidates"]
                )
            )
        )
        self.memory_store = ResearchMemoryStore(
            Path(config["paths"]["experiment_dir"]) / "adaptive"
        )
        self.signals: dict[str, pd.Series] = {}
        self.outcomes: list[ResearchOutcome] = []
        self.tested_ids: set[str] = set()
        self.tested_formulas: set[str] = set()
        self.total_proposed = 0
        self.duplicate_count = 0
        self.tiers: dict[str, str] = {}
        self.llm_events: list[dict] = []
        self.llm_raw_proposals = 0
        self.llm_rejected_proposals = 0

    @property
    def max_candidates(self) -> int:
        return int(
            self.settings.get(
                "max_candidate_evaluations", self.config["search"]["max_candidates"]
            )
        )

    def run(self) -> list[ResearchOutcome]:
        remaining = self.max_candidates
        max_rounds = int(self.settings.get("max_rounds", 8))
        for round_id in range(max_rounds):
            coverage_complete = self.macro.coverage_satisfied(self.memory)
            parent_correlations: dict[tuple[str, str], float] = {}
            parent_ids = list(self.parent_pool.items)
            for index, left_id in enumerate(parent_ids):
                for right_id in parent_ids[index + 1 :]:
                    if left_id not in self.signals or right_id not in self.signals:
                        continue
                    parent_correlations[
                        tuple(sorted((left_id, right_id)))
                    ] = mean_cross_sectional_correlation(
                        self.signals[left_id],
                        self.signals[right_id],
                        self.panel["date"],
                    )
            deterministic_plan = self.macro.plan(
                self.memory,
                self.parent_pool.items,
                self.elite_archive.items,
                remaining,
                parent_correlations,
            )
            plan = deterministic_plan
            macro_result = MacroPlanResult(None, False, "deterministic_controller")
            if (
                self.use_llm
                and coverage_complete
                and deterministic_plan.action != "STOP"
            ):
                macro_result = self.llm_macro.propose(
                    self.memory,
                    self.parent_pool.items,
                    self.elite_archive.items,
                    remaining,
                    parent_correlations,
                    deterministic_plan,
                )
                if macro_result.plan is not None:
                    plan = macro_result.plan
            if plan.action == "STOP":
                self.memory.stopped = True
                self.memory_store.append_round(
                    {"round_id": round_id, "plan": plan.to_dict(), "outcomes": []}
                )
                break
            micro_result = MicroProposalResult(
                (), False, "deterministic_mechanism_coverage"
            )
            if self.use_llm and coverage_complete:
                micro_result = self.llm_micro.generate(
                    plan,
                    self.parent_pool.items,
                    self.tested_formulas,
                    self.tested_ids,
                    self.base_loop.records,
                    self.tiers,
                    round_id,
                    self.max_candidates,
                )
                proposals = list(micro_result.candidates)
                self.llm_raw_proposals += micro_result.raw_proposal_count
                self.llm_rejected_proposals += len(micro_result.rejected)
            else:
                proposals = self.micro.generate(
                    plan, self.parent_pool.items, self.tested_formulas, round_id
                )
            integrity_reproposals: list = []
            if plan.theme == "integrity_unsupported_group":
                previous_round = self.memory.recent_round_summaries[-1]["round_id"]
                allowed_groups = tuple(
                    group
                    for group, field in self.contract.group_fields.items()
                    if field in self.panel
                )
                for outcome in self.outcomes:
                    if (
                        outcome.spec.generation == previous_round
                        and "unsupported_group" in outcome.record.failure_codes
                    ):
                        integrity_reproposals.extend(
                            valid_group_reproposals(
                                outcome.spec,
                                round_id=round_id,
                                allowed_groups=allowed_groups,
                            )
                        )
                        if len(integrity_reproposals) >= int(
                            self.settings.get("max_group_reproposals_per_round", 2)
                        ):
                            break
                integrity_reproposals = integrity_reproposals[
                    : int(self.settings.get("max_group_reproposals_per_round", 2))
                ]
                merged = integrity_reproposals + proposals
                seen_formulas = set(self.tested_formulas)
                seen_ids = set(self.tested_ids)
                proposals = []
                for candidate in merged:
                    if (
                        candidate.canonical_formula in seen_formulas
                        or candidate.factor_id in seen_ids
                    ):
                        continue
                    proposals.append(candidate)
                    seen_formulas.add(candidate.canonical_formula)
                    seen_ids.add(candidate.factor_id)
                    if len(proposals) >= plan.candidate_budget:
                        break
            rescue_reason = None
            if not proposals and remaining > 0:
                rescue_plan, rescue = self._novelty_rescue(round_id, remaining)
                if rescue:
                    plan = rescue_plan
                    proposals = rescue
                    rescue_reason = (
                        "planned neighborhood exhausted; pivoted to an untested formula"
                    )
                else:
                    self.memory.stopped = True
                    stop_plan = ResearchPlan(
                        "STOP",
                        "saturated",
                        plan.mechanism,
                        "Stop because every constrained deterministic neighborhood is exhausted.",
                        (),
                        "No novel formula remains inside the approved DSL search grid.",
                        0,
                    )
                    self.memory_store.append_round(
                        {
                            "round_id": round_id,
                            "plan": stop_plan.to_dict(),
                            "outcomes": [],
                        }
                    )
                    break
            llm_event = self._llm_event(round_id, macro_result, micro_result)
            llm_event["novelty_rescue"] = rescue_reason
            llm_event["integrity_reproposal_count"] = sum(
                candidate.proposal_type == "integrity_reproposal"
                for candidate in proposals
            )
            self.llm_events.append(llm_event)
            self._append_llm_event(llm_event)
            self.total_proposed += len(proposals)
            round_outcomes: list[ResearchOutcome] = []
            parent_admissions: list[dict] = []
            duplicate_count = 0
            for spec in proposals:
                if remaining <= 0:
                    break
                if (
                    spec.factor_id in self.tested_ids
                    or spec.canonical_formula in self.tested_formulas
                ):
                    duplicate_count += 1
                    self.duplicate_count += 1
                    continue
                self.tested_ids.add(spec.factor_id)
                self.tested_formulas.add(spec.canonical_formula)
                remaining -= 1
                mechanism = infer_mechanism(spec)
                reasons: list[str] = []
                static = check_factor_spec(
                    spec,
                    self.contract,
                    int(self.config["gates"]["max_complexity"]),
                    self.config.get("dsl", {}),
                )
                reasons.extend(static.reasons)
                alignment = check_alignment(spec, mechanism)
                reasons.extend(alignment.reasons)
                if not reasons and bool(
                    self.settings.get("dynamic_leakage_enabled", True)
                ):
                    leakage = check_dynamic_leakage(
                        self.panel,
                        spec,
                        sample_tickers=int(
                            self.settings.get("dynamic_leakage_sample_tickers", 4)
                        ),
                        cutoffs=int(self.settings.get("dynamic_leakage_cutoffs", 2)),
                    )
                    reasons.extend(leakage.reasons)
                if reasons:
                    record = self.base_loop._retired_integrity_record(spec, reasons)
                    signal = None
                    self.memory.proposal_failure_counts["pre_evaluation_rejection"] = (
                        self.memory.proposal_failure_counts.get(
                            "pre_evaluation_rejection", 0
                        )
                        + 1
                    )
                else:
                    record, signal = self.base_loop.evaluate_candidate(spec)
                if not record.integrity_passed:
                    for code in record.failure_codes:
                        if code in INTEGRITY_RESPONSES:
                            self.memory.proposal_failure_counts[code] = (
                                self.memory.proposal_failure_counts.get(code, 0) + 1
                            )
                tier = classify_tier(record, self.settings)
                elite_diagnostic = diagnose_elite_gates(
                    record, self.settings
                ).to_dict()
                record = replace(
                    record,
                    mechanism=mechanism,
                    elite_gate_diagnostic=elite_diagnostic,
                )
                self.base_loop.records.append(record)
                self.base_loop.store.append(record)
                outcome = ResearchOutcome(
                    spec,
                    record,
                    tier.tier,
                    mechanism,
                    plan.action,
                    elite_diagnostic,
                    self._repair_result(spec, record, plan),
                )
                round_outcomes.append(outcome)
                self.outcomes.append(outcome)
                self.tiers[spec.factor_id] = tier.tier
                if signal is not None:
                    self.signals[spec.factor_id] = signal
                if tier.tier in {"PARENT", "ELITE"} and signal is not None:
                    artifact = FactorArtifact(spec, record, mechanism)
                    admission = self.parent_pool.add(
                        artifact, signal, self.signals, self.panel["date"]
                    )
                    parent_admissions.append(
                        {
                            "factor_id": spec.factor_id,
                            "mechanism": mechanism,
                            "status": admission.status,
                            "representative_id": admission.representative_id,
                            "absolute_correlation": admission.absolute_correlation,
                            "evicted_ids": list(admission.evicted_ids),
                            "new_cluster": admission.new_cluster,
                        }
                    )
                    if tier.tier == "ELITE":
                        self.elite_archive.add(artifact)
                        self.base_loop.promoted_specs[spec.factor_id] = spec
                        self.base_loop.promoted_signals[spec.factor_id] = signal
            new_elite_admitted = any(
                item.tier == "ELITE" and item.spec.factor_id in self.elite_archive.items
                for item in round_outcomes
            )
            summary = {
                "round_id": round_id,
                "action": plan.action,
                "mechanism": plan.mechanism,
                "proposed": len(proposals),
                "evaluated": len(round_outcomes),
                "duplicates": duplicate_count,
                "tier_counts": {
                    tier: sum(item.tier == tier for item in round_outcomes)
                    for tier in ("RETIRED", "PARENT", "ELITE")
                },
                "remaining_budget": remaining,
                "coverage": self.macro.coverage_status(self.memory),
                "integrity_failure_counts": integrity_response_counts(round_outcomes),
                "integrity_failure_total": sum(
                    not item.record.integrity_passed
                    and any(
                        code in INTEGRITY_RESPONSES
                        for code in item.record.failure_codes
                    )
                    for item in round_outcomes
                ),
                "parent_cluster_stats": self.parent_pool.cluster_stats(),
                "parent_admissions": parent_admissions,
                "new_parent_observed": any(
                    item.get("status") in {"added", "replaced"}
                    for item in parent_admissions
                ),
                "new_cluster_admitted": any(
                    item.get("new_cluster", False) for item in parent_admissions
                ),
                "new_elite_admitted": new_elite_admitted,
                "best_quality_improved": self._update_best_quality(round_outcomes),
                "llm": llm_event,
            }
            self.cross.update_memory(
                self.memory, round_outcomes, summary, self.settings
            )
            self.memory.parent_pool_ids = list(self.parent_pool.items)
            self.memory.parent_cluster_stats = self.parent_pool.cluster_stats()
            self.memory_store.append_round(
                {
                    "plan": plan.to_dict(),
                    **summary,
                    "outcomes": [item.to_dict() for item in round_outcomes],
                }
            )
            self.memory_store.save(self.memory)
            self.memory_store.checkpoint(round_id, self.memory)
            if remaining <= 0:
                break
        self.base_loop.store.write_family_tree(self.base_loop.records)
        self.construct_library()
        return self.outcomes

    def _update_best_quality(self, outcomes: list[ResearchOutcome]) -> bool:
        """Track a research-visible Pareto improvement across raw gate metrics."""
        tolerance = float(self.settings.get("best_quality_improvement_tolerance", 0.001))
        best = self.memory.best_quality_metrics
        improved = False
        for outcome in outcomes:
            record = outcome.record
            if not record.integrity_passed:
                continue
            candidate = {
                "mean_rank_ic": record.mean_rank_ic,
                "ic_tstat": record.ic_tstat,
                "positive_fold_count": record.positive_fold_count,
                "high_cost_sharpe": record.high_cost_sharpe,
                "turnover_margin": (
                    float(self.settings.get("elite_max_turnover", 1.5))
                    - record.turnover
                    if record.turnover is not None
                    else None
                ),
            }
            candidate = {
                key: float(value)
                for key, value in candidate.items()
                if value is not None and pd.notna(value)
            }
            if not candidate:
                continue
            if not best:
                self.memory.best_quality_metrics = candidate
                self.memory.best_quality_mechanism = outcome.mechanism
                improved = True
                continue
            comparable = set(best) & set(candidate)
            if not comparable:
                continue
            no_material_regression = all(
                candidate[key] >= best[key] - tolerance for key in comparable
            )
            material_gain = any(
                candidate[key] >= best[key] + tolerance for key in comparable
            )
            if no_material_regression and material_gain:
                self.memory.best_quality_metrics = {
                    **best,
                    **candidate,
                }
                self.memory.best_quality_mechanism = outcome.mechanism
                best = self.memory.best_quality_metrics
                improved = True
        return improved

    def _repair_result(self, spec, record, plan: ResearchPlan) -> dict:
        if not plan.targeted_failure or not spec.parent_ids:
            return {}
        parent = self.parent_pool.items.get(spec.parent_ids[0])
        if parent is None:
            return {"targeted_failure": plan.targeted_failure, "parent_available": False}

        names = {
            "mean_rank_ic": "mean_rank_ic",
            "newey_west_tstat": "ic_tstat",
            "positive_fold_count": "positive_fold_count",
            "high_cost_sharpe": "high_cost_sharpe",
            "turnover": "turnover",
        }
        target_name = names.get(plan.target_metric or "")
        preserve_name = names.get(plan.preserve_metric or "")
        before_target = getattr(parent.record, target_name) if target_name else None
        after_target = getattr(record, target_name) if target_name else None
        before_preserve = getattr(parent.record, preserve_name) if preserve_name else None
        after_preserve = getattr(record, preserve_name) if preserve_name else None
        tolerance = float(self.settings.get("repair_preserve_tolerance", 0.001))
        target_improved = (
            before_target is not None
            and after_target is not None
            and (
                after_target < before_target - tolerance
                if plan.target_metric == "turnover"
                else after_target > before_target + tolerance
            )
        )
        collateral_damage = (
            before_preserve is not None
            and after_preserve is not None
            and (
                after_preserve < before_preserve - tolerance
                if plan.preserve_metric != "turnover"
                else after_preserve > before_preserve + tolerance
            )
        )
        return {
            "targeted_failure": plan.targeted_failure,
            "target_metric": plan.target_metric,
            "target_before": before_target,
            "target_after": after_target,
            "preserve_metric": plan.preserve_metric,
            "preserve_before": before_preserve,
            "preserve_after": after_preserve,
            "target_improved": target_improved,
            "collateral_damage": collateral_damage,
            "repair_succeeded": bool(target_improved and not collateral_damage),
        }

    def _novelty_rescue(
        self, round_id: int, remaining: int
    ) -> tuple[ResearchPlan, list]:
        tested = {
            mechanism: int(
                self.memory.mechanism_stats.get(mechanism, {}).get("tested", 0)
            )
            for mechanism in MECHANISMS
        }
        order = {mechanism: index for index, mechanism in enumerate(MECHANISMS)}
        budget = min(
            remaining,
            int(self.settings.get("budgets", {}).get("pivot", 6)),
        )
        for mechanism in sorted(
            MECHANISMS, key=lambda item: (tested[item], order[item])
        ):
            plan = ResearchPlan(
                "PIVOT",
                f"novelty_rescue_{mechanism.lower()}",
                mechanism,
                "Test a still-unseen formula in the least-sampled available mechanism.",
                (),
                "The planned local neighborhood produced no novel candidate.",
                budget,
            )
            proposals = self.micro.generate(
                plan, self.parent_pool.items, self.tested_formulas, round_id
            )
            if proposals:
                return plan, proposals
        return (
            ResearchPlan(
                "STOP",
                "saturated",
                "PRICE_TREND",
                "Stop because the approved formula grid is exhausted.",
                (),
                "No novel constrained candidate remains.",
                0,
            ),
            [],
        )

    def _llm_event(
        self,
        round_id: int,
        macro: MacroPlanResult,
        micro: MicroProposalResult,
    ) -> dict:
        return {
            "round_id": round_id,
            "enabled": self.use_llm,
            "macro": {
                "used_llm": macro.used_llm,
                "reason": macro.reason,
                "rejected": list(macro.rejected),
                "response_id": macro.response_id,
                "model": macro.model,
                "usage": macro.usage,
            },
            "micro": {
                "used_llm": micro.used_llm,
                "reason": micro.reason,
                "raw_proposal_count": micro.raw_proposal_count,
                "llm_candidate_count": micro.llm_candidate_count,
                "deterministic_fill_count": micro.deterministic_fill_count,
                "rejected": list(micro.rejected),
                "response_id": micro.response_id,
                "model": micro.model,
                "usage": micro.usage,
                "research_summary": micro.research_summary,
                "retry_count": micro.retry_count,
                "stage_counts": micro.stage_counts,
            },
        }

    def _append_llm_event(self, payload: dict) -> None:
        path = self.memory_store.directory / "llm_events.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    def construct_library(self) -> tuple[list[FactorArtifact], list[dict]]:
        """Build and save a research-only candidate library without opening holdout."""
        library_config = {
            **self.settings,
            "prediction_horizon_days": self.config["evaluation"][
                "prediction_horizon_days"
            ],
        }
        selected, audit = build_final_library(
            list(self.elite_archive.items.values()),
            self.signals,
            self.panel,
            library_config,
        )
        directory = self.memory_store.directory
        (directory / "factor_library_candidate.json").write_text(
            json.dumps([item.spec.to_dict() for item in selected], indent=2) + "\n",
            encoding="utf-8",
        )
        (directory / "library_selection_audit.json").write_text(
            json.dumps(audit, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return selected, audit

    def freeze(self, directory: str | Path) -> dict:
        freeze_dir = Path(directory)
        selected, audit = self.construct_library()
        self.base_loop.promoted_specs = {
            item.spec.factor_id: item.spec for item in selected
        }
        manifest = self.base_loop.freeze(freeze_dir)
        (freeze_dir / "elite_archive.json").write_text(
            json.dumps(
                [item.spec.to_dict() for item in self.elite_archive.items.values()],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (freeze_dir / "library_selection_audit.json").write_text(
            json.dumps(audit, indent=2, default=str) + "\n", encoding="utf-8"
        )
        (freeze_dir / "research_memory.json").write_text(
            json.dumps(self.memory.to_dict(), indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        self.memory_store.seal()
        return manifest

    def summary(self) -> dict:
        first_elite = next(
            (
                index + 1
                for index, item in enumerate(self.outcomes)
                if item.tier == "ELITE"
            ),
            None,
        )
        valid = sum(item.record.integrity_passed for item in self.outcomes)
        informative = sum(
            item.record.integrity_passed
            and bool(item.record.fold_metrics)
            and all(
                fold.get("stock_day_observations", 0)
                and fold.get("mean_ic") is not None
                for fold in item.record.fold_metrics
            )
            for item in self.outcomes
        )
        mechanisms = {item.mechanism for item in self.elite_archive.items.values()}
        stable = [
            item.record.fold_ic_sign_consistency
            for item in self.elite_archive.items.values()
            if item.record.fold_ic_sign_consistency is not None
        ]
        denominator = max(1, self.total_proposed)
        first_parent = next(
            (
                index + 1
                for index, item in enumerate(self.outcomes)
                if item.tier in {"PARENT", "ELITE"}
            ),
            None,
        )
        coverage = self.macro.coverage_status(self.memory)
        token_usage = sum(
            int(section.get("usage", {}).get("total_tokens", 0) or 0)
            for event in self.llm_events
            for section in (event["macro"], event["micro"])
        )
        llm_calls = sum(
            bool(section.get("response_id"))
            for event in self.llm_events
            for section in (event["macro"], event["micro"])
        )
        return {
            "arm": "adaptive_luna" if self.use_llm else "adaptive_deterministic",
            "candidates_tested": len(self.outcomes),
            "candidates_to_first_parent": first_parent,
            "candidates_to_first_elite": first_elite,
            "valid_information_per_10": 10.0 * informative / len(self.outcomes)
            if self.outcomes
            else 0.0,
            "duplicate_formula_ratio": self.duplicate_count / denominator,
            "invalid_proposal_ratio": 1.0 - valid / len(self.outcomes)
            if self.outcomes
            else 0.0,
            "parent_pool_size": len(self.parent_pool.items),
            "parent_cluster_stats": self.memory.parent_cluster_stats,
            "elite_archive_size": len(self.elite_archive.items),
            "elite_mechanism_diversity": len(mechanisms),
            "elite_walk_forward_sign_stability": sum(stable) / len(stable)
            if stable
            else None,
            "mechanisms_covered": list(coverage["covered"]),
            "mechanism_coverage_count": len(coverage["covered"]),
            "mechanism_test_counts": coverage["tested"],
            "llm_calls": llm_calls,
            "llm_total_tokens": token_usage,
            "llm_raw_factor_proposals": self.llm_raw_proposals,
            "llm_rejected_factor_proposals": self.llm_rejected_proposals,
            "llm_invalid_proposal_ratio": (
                self.llm_rejected_proposals / self.llm_raw_proposals
                if self.llm_raw_proposals
                else 0.0
            ),
            "llm_macro_fallbacks": sum(
                self.use_llm
                and event["macro"]["reason"] != "deterministic_controller"
                and not event["macro"]["used_llm"]
                for event in self.llm_events
            ),
            "holdout_evaluated": False,
        }
