"""Comparable search-efficiency metrics for deterministic and LLM research arms."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from src.utils.logging import ExperimentRecord


def _family_entropy(records: list[ExperimentRecord]) -> float:
    promoted = [record.family for record in records if record.decision == "PROMOTE"]
    if not promoted:
        return 0.0
    counts = Counter(promoted)
    probabilities = [count / len(promoted) for count in counts.values()]
    raw = -sum(value * math.log(value) for value in probabilities)
    return float(raw / math.log(4)) if len(counts) > 1 else 0.0


def _has_complete_walk_forward_evidence(record: ExperimentRecord) -> bool:
    if not record.integrity_passed or not record.fold_metrics:
        return False
    required = (
        "mean_ic",
        "ic_tstat",
        "net_sharpe",
        "high_cost_sharpe",
        "turnover",
        "max_drawdown",
    )
    return all(
        int(fold.get("stock_day_observations", 0) or 0) > 0
        and all(
            fold.get(field) is not None and np.isfinite(fold[field])
            for field in required
        )
        for fold in record.fold_metrics
    )


def summarize_search(
    records: list[ExperimentRecord],
    generation_events: list[dict],
    *,
    arm: str,
) -> dict:
    evaluation_events = [
        event for event in generation_events if event.get("event") == "evaluation"
    ]
    proposal_events = [
        event for event in generation_events if event.get("event") == "proposal"
    ]
    rejected_proposals = sum(
        len(event.get("rejected", [])) for event in proposal_events
    )
    evaluated_proposals = sum(
        int(event.get("proposed_count", 0)) for event in evaluation_events
    )
    total_proposals = evaluated_proposals + rejected_proposals
    duplicate_count = sum(
        int(event.get("duplicate_id_count", 0))
        + int(event.get("duplicate_formula_count", 0))
        for event in evaluation_events
    ) + sum(
        "duplicate" in reason.lower()
        for event in proposal_events
        for reason in event.get("rejected", [])
    )
    integrity_failures = sum(not record.integrity_passed for record in records)
    incomplete_evidence = sum(
        record.integrity_passed and not _has_complete_walk_forward_evidence(record)
        for record in records
    )
    invalid_count = integrity_failures + incomplete_evidence + rejected_proposals
    informative = [
        record for record in records if _has_complete_walk_forward_evidence(record)
    ]
    promoted = [record for record in records if record.decision == "PROMOTE"]
    first_promote = next(
        (
            index
            for index, record in enumerate(records, start=1)
            if record.decision == "PROMOTE"
        ),
        None,
    )
    fold_consistency = [
        record.fold_ic_sign_consistency
        for record in informative
        if record.fold_ic_sign_consistency is not None
        and np.isfinite(record.fold_ic_sign_consistency)
    ]
    positive_all_folds = [
        record
        for record in informative
        if record.fold_metrics
        and record.positive_fold_count == len(record.fold_metrics)
    ]
    families = sorted({record.family for record in promoted})
    token_usage = sum(
        int(event.get("usage", {}).get("total_tokens", 0) or 0)
        for event in proposal_events
    )
    return {
        "arm": arm,
        "candidate_proposals": total_proposals,
        "candidates_evaluated": len(records),
        "first_promote_candidate_index": first_promote,
        "informative_candidates": len(informative),
        "effective_new_information_per_10_candidates": (
            10.0 * len(informative) / total_proposals if total_proposals else 0.0
        ),
        "duplicate_formula_count": duplicate_count,
        "duplicate_formula_rate": (
            duplicate_count / total_proposals if total_proposals else 0.0
        ),
        "invalid_or_noncompliant_count": invalid_count,
        "invalid_or_noncompliant_rate": (
            invalid_count / total_proposals if total_proposals else 0.0
        ),
        "promoted_count": len(promoted),
        "promoted_families": families,
        "promoted_family_count": len(families),
        "promoted_family_entropy": _family_entropy(records),
        "walk_forward_stability": {
            "mean_fold_ic_sign_consistency": (
                float(np.mean(fold_consistency)) if fold_consistency else None
            ),
            "positive_ic_in_all_folds_rate": (
                len(positive_all_folds) / len(informative) if informative else None
            ),
        },
        "decision_counts": dict(Counter(record.decision for record in records)),
        "failure_taxonomy": dict(
            Counter(code for record in records for code in record.failure_codes)
        ),
        # A completed API response is an LLM call even when every raw proposal is
        # rejected by deterministic validation and used_llm is therefore false.
        "llm_calls": sum(bool(event.get("response_id")) for event in proposal_events),
        "llm_total_tokens": token_usage,
        "holdout_evaluated": False,
        "metric_definition": (
            "Effective new information means a unique candidate that passed integrity "
            "checks and produced complete walk-forward evaluation evidence."
        ),
    }


def render_comparison_markdown(summaries: list[dict]) -> str:
    """Render the pre-committed search metrics without introducing a mega-score."""

    metrics = (
        ("Candidate proposals", "candidate_proposals"),
        ("Candidates evaluated", "candidates_evaluated"),
        ("First Promote index", "first_promote_candidate_index"),
        ("Effective information / 10", "effective_new_information_per_10_candidates"),
        ("Duplicate formula rate", "duplicate_formula_rate"),
        ("Invalid/noncompliant rate", "invalid_or_noncompliant_rate"),
        ("Promoted count", "promoted_count"),
        ("Promoted family count", "promoted_family_count"),
        ("Promoted family entropy", "promoted_family_entropy"),
    )
    arms = [summary["arm"] for summary in summaries]
    lines = [
        "# Search comparison",
        "",
        "The comparison uses research folds only; the final holdout was not evaluated.",
        "",
        "| Metric | " + " | ".join(arms) + " |",
        "| --- | " + " | ".join("---:" for _ in arms) + " |",
    ]
    for label, key in metrics:
        values = []
        for summary in summaries:
            value = summary.get(key)
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append("—" if value is None else str(value))
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines.extend(["", "## Walk-forward stability", ""])
    for summary in summaries:
        stability = summary["walk_forward_stability"]
        lines.append(
            f"- **{summary['arm']}**: mean fold-sign consistency "
            f"{stability['mean_fold_ic_sign_consistency']}; all-positive-fold rate "
            f"{stability['positive_ic_in_all_folds_rate']}."
        )
    lines.extend(["", "## Failure taxonomy", ""])
    for summary in summaries:
        lines.append(
            f"- **{summary['arm']}**: "
            + json_like_counts(summary.get("failure_taxonomy", {}))
        )
    return "\n".join(lines) + "\n"


def json_like_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
