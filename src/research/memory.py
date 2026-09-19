"""Explicit research state derived from structured experiment records."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.utils.logging import ExperimentRecord


@dataclass
class ResearchState:
    generation: int = 0
    promoted_factor_ids: list[str] = field(default_factory=list)
    held_factor_ids: list[str] = field(default_factory=list)
    retired_factor_ids: list[str] = field(default_factory=list)
    family_summary: dict[str, dict[str, int]] = field(default_factory=dict)
    best_patterns: list[str] = field(default_factory=list)
    failed_patterns: list[str] = field(default_factory=list)
    unexplored_families: list[str] = field(
        default_factory=lambda: ["price", "fundamental", "news", "interaction"]
    )
    search_budget_remaining: int = 0


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
            record.family, {"PROMOTE": 0, "HOLD": 0, "RETIRE": 0}
        )
        summary[record.decision] += 1
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
