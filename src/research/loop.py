"""Memory-guided autonomous factor research loop."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.evaluation.evaluator import evaluate_walk_forward
from src.evaluation.integrity import check_integrity
from src.evaluation.redundancy import max_library_correlation, residual_ic
from src.evaluation.validation import HoldoutGuard, folds_from_config
from src.factors.builder import FactorBuilder
from src.factors.primitives import future_return
from src.factors.schema import FactorSpec
from src.research.generator import exploration_candidates, seed_candidates
from src.research.llm_generator import LLMFactorGenerator
from src.research.memory import ResearchState, update_state
from src.research.mutator import mutate
from src.research.selector import decide
from src.utils.config import config_digest
from src.utils.logging import ExperimentRecord, ExperimentStore


class ResearchLoop:
    def __init__(self, config: dict, panel: pd.DataFrame | None = None) -> None:
        self.config = config
        self.panel = panel
        self.builder = FactorBuilder()
        self.folds = folds_from_config(config)
        self.holdout = HoldoutGuard(int(config["walk_forward"]["holdout_year"]))
        self.store = ExperimentStore(config["paths"]["experiment_dir"])
        self.state = ResearchState(
            search_budget_remaining=int(config["search"]["max_candidates"])
        )
        llm_config = dict(config.get("llm", {}))
        llm_config.setdefault("max_complexity", int(config["gates"]["max_complexity"]))
        llm_config.setdefault(
            "max_proposals_per_parent",
            int(config["search"]["max_llm_proposals_per_parent"]),
        )
        self.llm_generator = LLMFactorGenerator(llm_config)
        self.records: list[ExperimentRecord] = []
        self.promoted_specs: dict[str, FactorSpec] = {}
        self.promoted_signals: dict[str, pd.Series] = {}

    def dry_run(self) -> dict:
        seeds = seed_candidates()
        return {
            "seed_candidates": len(seeds),
            "production_candidates": sum(not item.demonstration_only for item in seeds),
            "integrity_demonstrations": sum(item.demonstration_only for item in seeds),
            "folds": [fold.name for fold in self.folds],
            "holdout_year": self.holdout.holdout_year,
            "holdout_frozen": self.holdout.frozen,
        }

    def _retired_integrity_record(
        self, spec: FactorSpec, reasons: list[str]
    ) -> ExperimentRecord:
        return ExperimentRecord(
            factor_id=spec.factor_id,
            generation=spec.generation,
            parent_ids=spec.parent_ids,
            family=spec.family,
            hypothesis=spec.hypothesis,
            canonical_formula=spec.canonical_formula,
            integrity_passed=False,
            integrity_issues=tuple(reasons),
            fold_metrics=(),
            mean_rank_ic=None,
            ic_tstat=None,
            positive_fold_count=None,
            long_short_sharpe=None,
            high_cost_sharpe=None,
            turnover=None,
            max_drawdown=None,
            redundancy_corr=None,
            closest_factor_id=None,
            residual_ic=None,
            decision="RETIRE",
            reasons=tuple(reasons),
        )

    def evaluate_candidate(
        self, spec: FactorSpec
    ) -> tuple[ExperimentRecord, pd.Series | None]:
        if self.panel is None:
            raise RuntimeError("A research-only panel is required for execution.")
        static = check_integrity(
            spec,
            self.panel,
            None,
            max_complexity=int(self.config["gates"]["max_complexity"]),
            minimum_coverage=float(self.config["gates"]["minimum_factor_coverage"]),
        )
        if not static.passed:
            return self._retired_integrity_record(spec, static.reasons), None

        signal = self.builder.build(self.panel, spec)
        integrity = check_integrity(
            spec,
            self.panel,
            signal,
            max_complexity=int(self.config["gates"]["max_complexity"]),
            minimum_coverage=float(self.config["gates"]["minimum_factor_coverage"]),
        )
        if not integrity.passed:
            return self._retired_integrity_record(spec, integrity.reasons), None

        metrics = evaluate_walk_forward(
            self.panel, signal, self.folds, self.config["evaluation"]
        )
        corr, closest = max_library_correlation(
            signal, self.promoted_signals, self.panel["date"]
        )
        incremental_ic = None
        if closest is not None and corr > float(
            self.config["gates"]["redundancy_correlation"]
        ):
            target = future_return(
                self.panel["close"],
                self.panel["symbol"],
                int(self.config["evaluation"]["prediction_horizon_days"]),
            )
            incremental_ic = residual_ic(
                signal,
                self.promoted_signals[closest],
                self.panel["date"],
                target,
            )
        gate = decide(
            spec,
            metrics,
            self.config["gates"],
            redundancy_corr=corr,
            residual_ic=incremental_ic,
        )
        payload = metrics.to_dict()
        record = ExperimentRecord(
            factor_id=spec.factor_id,
            generation=spec.generation,
            parent_ids=spec.parent_ids,
            family=spec.family,
            hypothesis=spec.hypothesis,
            canonical_formula=spec.canonical_formula,
            integrity_passed=True,
            integrity_issues=(),
            fold_metrics=tuple(payload["folds"]),
            mean_rank_ic=metrics.mean_ic,
            ic_tstat=metrics.mean_ic_tstat,
            positive_fold_count=metrics.positive_ic_folds,
            long_short_sharpe=metrics.mean_net_sharpe,
            high_cost_sharpe=metrics.mean_high_cost_sharpe,
            turnover=metrics.mean_turnover,
            max_drawdown=metrics.worst_drawdown,
            redundancy_corr=corr,
            closest_factor_id=closest,
            residual_ic=incremental_ic,
            decision=gate.decision,
            reasons=gate.reasons,
        )
        return record, signal

    def run(self) -> list[ExperimentRecord]:
        candidates = seed_candidates()
        max_generations = int(self.config["search"]["generations"])
        max_candidates = int(self.config["search"]["max_candidates"])
        tested_ids: set[str] = set()

        for generation in range(max_generations):
            generation_records: list[ExperimentRecord] = []
            promoted_this_round: list[FactorSpec] = []
            for spec in candidates:
                if spec.factor_id in tested_ids or len(self.records) >= max_candidates:
                    continue
                tested_ids.add(spec.factor_id)
                record, signal = self.evaluate_candidate(spec)
                self.records.append(record)
                generation_records.append(record)
                self.store.append(record)
                if record.decision == "PROMOTE" and signal is not None:
                    self.promoted_specs[spec.factor_id] = spec
                    self.promoted_signals[spec.factor_id] = signal
                    promoted_this_round.append(spec)

            self.state = update_state(
                self.state,
                generation_records,
                total_budget=max_candidates,
                tested_count=len(self.records),
            )
            if (
                generation + 1 >= max_generations
                or self.state.search_budget_remaining == 0
            ):
                break
            next_candidates: list[FactorSpec] = []
            for parent in promoted_this_round:
                next_candidates.extend(
                    mutate(
                        parent,
                        generation + 1,
                        int(self.config["search"]["max_mutations_per_parent"]),
                    )
                )
            llm_result = self.llm_generator.propose(
                generation=generation + 1,
                state=self.state,
                recent_records=self.records,
                promoted_specs=self.promoted_specs,
                tested_ids=tested_ids,
            )
            self.store.append_generation_event(
                {
                    "generation": generation + 1,
                    "used_llm": llm_result.used_llm,
                    "reason": llm_result.reason,
                    "candidate_ids": [
                        candidate.factor_id for candidate in llm_result.candidates
                    ],
                    "rejected": list(llm_result.rejected),
                }
            )
            if llm_result.used_llm:
                next_candidates.extend(llm_result.candidates)
            else:
                next_candidates.extend(exploration_candidates(generation + 1))
            candidates = []
            seen_formulas: set[str] = set()
            for candidate in next_candidates:
                if candidate.canonical_formula in seen_formulas:
                    continue
                seen_formulas.add(candidate.canonical_formula)
                candidates.append(candidate)
            if not candidates:
                break
        self.store.write_family_tree(self.records)
        return self.records

    def freeze(self, directory: str | Path) -> dict:
        freeze_dir = Path(directory)
        freeze_dir.mkdir(parents=True, exist_ok=True)
        library = [spec.to_dict() for spec in self.promoted_specs.values()]
        (freeze_dir / "factor_library.json").write_text(
            json.dumps(library, indent=2) + "\n", encoding="utf-8"
        )
        (freeze_dir / "config.json").write_text(
            json.dumps(self.config, indent=2, default=str) + "\n", encoding="utf-8"
        )
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            commit = "unavailable"
        manifest = {
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config_digest(self.config),
            "git_commit": commit,
            "random_seed": self.config["project"]["random_seed"],
            "factor_count": len(library),
            "holdout_year": self.holdout.holdout_year,
        }
        (freeze_dir / "freeze_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        self.holdout.freeze()
        return manifest
