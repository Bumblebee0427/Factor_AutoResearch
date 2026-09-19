"""Explicit cumulative research state and family-level diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from src.utils.logging import ExperimentRecord


@dataclass
class FamilyResearchSummary:
    tested: int = 0
    integrity_passed: int = 0
    informative: int = 0
    promoted: int = 0
    held: int = 0
    retired: int = 0
    unique_formulas: int = 0
    mean_ic_sum: float = 0.0
    mean_ic_count: int = 0
    best_mean_ic: float | None = None
    newey_west_tstat_sum: float = 0.0
    newey_west_tstat_count: int = 0
    turnover_sum: float = 0.0
    turnover_count: int = 0
    failure_counts: dict[str, int] = field(default_factory=dict)
    _formulas: set[str] = field(default_factory=set, repr=False)

    def update(self, record: ExperimentRecord) -> None:
        self.tested += 1
        self.promoted += int(record.decision == "PROMOTE")
        self.held += int(record.decision == "HOLD")
        self.retired += int(record.decision == "RETIRE")
        if record.canonical_formula not in self._formulas:
            self._formulas.add(record.canonical_formula)
            self.unique_formulas += 1
        if record.integrity_passed:
            self.integrity_passed += 1
            self.informative += 1
        if record.mean_rank_ic is not None and np.isfinite(record.mean_rank_ic):
            self.mean_ic_sum += float(record.mean_rank_ic)
            self.mean_ic_count += 1
            self.best_mean_ic = (
                float(record.mean_rank_ic)
                if self.best_mean_ic is None
                else max(self.best_mean_ic, float(record.mean_rank_ic))
            )
        if record.ic_tstat is not None and np.isfinite(record.ic_tstat):
            self.newey_west_tstat_sum += float(record.ic_tstat)
            self.newey_west_tstat_count += 1
        if record.turnover is not None and np.isfinite(record.turnover):
            self.turnover_sum += float(record.turnover)
            self.turnover_count += 1
        for code in record.failure_codes:
            self.failure_counts[code] = self.failure_counts.get(code, 0) + 1

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.pop("_formulas", None)
        payload["decision_counts"] = {
            "PROMOTE": self.promoted,
            "HOLD": self.held,
            "RETIRE": self.retired,
        }
        payload["mean_rank_ic"] = (
            self.mean_ic_sum / self.mean_ic_count if self.mean_ic_count else None
        )
        payload["mean_newey_west_tstat"] = (
            self.newey_west_tstat_sum / self.newey_west_tstat_count
            if self.newey_west_tstat_count
            else None
        )
        payload["mean_turnover"] = (
            self.turnover_sum / self.turnover_count if self.turnover_count else None
        )
        for internal in (
            "mean_ic_sum",
            "mean_ic_count",
            "newey_west_tstat_sum",
            "newey_west_tstat_count",
            "turnover_sum",
            "turnover_count",
            "promoted",
            "held",
            "retired",
        ):
            payload.pop(internal, None)
        return payload


@dataclass
class ResearchState:
    generation: int = 0
    promoted_factor_ids: list[str] = field(default_factory=list)
    held_factor_ids: list[str] = field(default_factory=list)
    retired_factor_ids: list[str] = field(default_factory=list)
    informative_factor_ids: list[str] = field(default_factory=list)
    family_summary: dict[str, FamilyResearchSummary] = field(default_factory=dict)
    failure_taxonomy: dict[str, int] = field(default_factory=dict)
    best_patterns: list[str] = field(default_factory=list)
    failed_patterns: list[str] = field(default_factory=list)
    unexplored_families: list[str] = field(
        default_factory=lambda: ["price", "fundamental", "news", "interaction"]
    )
    search_budget_remaining: int = 0

    def family_summary_dict(self) -> dict[str, dict]:
        return {
            family: summary.to_dict()
            for family, summary in sorted(self.family_summary.items())
        }


def update_state(
    state: ResearchState,
    records: list[ExperimentRecord],
    *,
    total_budget: int,
    tested_count: int,
) -> ResearchState:
    state.generation += 1
    for record in records:
        summary = state.family_summary.setdefault(
            record.family, FamilyResearchSummary()
        )
        summary.update(record)
        if record.integrity_passed:
            state.informative_factor_ids.append(record.factor_id)
        for code in record.failure_codes:
            state.failure_taxonomy[code] = state.failure_taxonomy.get(code, 0) + 1
        if record.decision == "PROMOTE":
            state.promoted_factor_ids.append(record.factor_id)
            state.best_patterns.append(record.canonical_formula)
        elif record.decision == "HOLD":
            state.held_factor_ids.append(record.factor_id)
        else:
            state.retired_factor_ids.append(record.factor_id)
            state.failed_patterns.append(record.canonical_formula)
        if record.family in state.unexplored_families:
            state.unexplored_families.remove(record.family)
    state.search_budget_remaining = max(total_budget - tested_count, 0)
    return state
