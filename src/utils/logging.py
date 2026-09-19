"""Append-only experiment records and lineage rendering."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentRecord:
    factor_id: str
    generation: int
    parent_ids: tuple[str, ...]
    family: str
    hypothesis: str
    canonical_formula: str
    integrity_passed: bool
    integrity_issues: tuple[str, ...]
    fold_metrics: tuple[dict[str, Any], ...]
    mean_rank_ic: float | None
    ic_tstat: float | None
    positive_fold_count: int | None
    long_short_sharpe: float | None
    high_cost_sharpe: float | None
    turnover: float | None
    max_drawdown: float | None
    redundancy_corr: float | None
    closest_factor_id: str | None
    residual_ic: float | None
    decision: str
    reasons: tuple[str, ...]
    failure_codes: tuple[str, ...] = ()
    multi_horizon_mean_ic: dict[str, float] = field(default_factory=dict)
    fold_ic_sign_consistency: float | None = None
    naive_ic_tstat: float | None = None
    proposal_type: str = "deterministic"
    evidence_factor_ids: tuple[str, ...] = ()
    targeted_failure: str | None = None
    expected_metric_effect: str | None = None
    falsification_condition: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentStore:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.directory / "candidates.jsonl"
        self.csv_path = self.directory / "trajectory.csv"
        self.generation_events_path = self.directory / "generation_events.jsonl"

    def append(self, record: ExperimentRecord) -> None:
        payload = record.to_dict()
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        row = {
            "generation": record.generation,
            "factor_id": record.factor_id,
            "parents": "|".join(record.parent_ids),
            "family": record.family,
            "proposal_type": record.proposal_type,
            "mean_rank_ic": record.mean_rank_ic,
            "ic_tstat": record.ic_tstat,
            "naive_ic_tstat": record.naive_ic_tstat,
            "net_sharpe": record.long_short_sharpe,
            "high_cost_sharpe": record.high_cost_sharpe,
            "turnover": record.turnover,
            "redundancy_corr": record.redundancy_corr,
            "decision": record.decision,
            "failure_codes": "|".join(record.failure_codes),
            "reasons": "|".join(record.reasons),
        }
        exists = self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def append_generation_event(self, payload: dict[str, Any]) -> None:
        with self.generation_events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    def write_research_state(self, payload: dict[str, Any]) -> Path:
        path = self.directory / "research_state.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return path

    def write_family_tree(self, records: list[ExperimentRecord]) -> Path:
        children: dict[str | None, list[ExperimentRecord]] = {}
        for record in records:
            parent = record.parent_ids[0] if record.parent_ids else None
            children.setdefault(parent, []).append(record)
        lines: list[str] = []

        def visit(node: ExperimentRecord, prefix: str) -> None:
            reason = "; ".join(node.reasons)
            lines.append(
                f"{prefix}{node.factor_id} [{node.decision.lower()}: {reason}]"
            )
            for child in children.get(node.factor_id, []):
                visit(child, prefix + "  |-- ")

        for root in children.get(None, []):
            visit(root, "")
        path = self.directory / "factor_family_tree.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
