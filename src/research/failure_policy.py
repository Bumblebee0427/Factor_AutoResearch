"""Precommitted responses to integrity failures, kept separate from metric gates."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from src.factors.expression import Expr, Op
from src.factors.schema import FactorSpec


INTEGRITY_RESPONSES = {
    "lookahead_or_leakage": {
        "action": "PIVOT_AWAY",
        "instruction": (
            "Quarantine the formula and its lineage; explore a different mechanism "
            "with causal, past-only inputs. Never repair by relaxing leakage checks."
        ),
        "terminal": True,
    },
    "unsupported_group": {
        "action": "REPROPOSE_VALID_GROUP",
        "instruction": (
            "Re-propose a parent-free expression in the same mechanism using only "
            "the available sector or subindustry group, or remove the group operator."
        ),
        "terminal": True,
    },
    "invalid_dsl": {
        "action": "REPROPOSE_VALID_DSL",
        "instruction": (
            "Discard the invalid AST and propose a fresh parent-free expression "
            "using the registered operators, allowed features and parameter limits."
        ),
        "terminal": True,
    },
}
PRIORITY = tuple(INTEGRITY_RESPONSES)
GROUP_OPERATORS = frozenset({"group_rank", "group_zscore", "group_neutralize"})

METRIC_FAILURE_POLICIES = {
    "weak_predictive_signal": {
        "diagnosis": "Mean IC is below the configured Elite threshold.",
        "repair_policy": ["one targeted repair if parent-level evidence persists", "otherwise pivot"],
        "terminal": False,
    },
    "low_statistical_significance": {
        "diagnosis": "Mean IC is positive but Newey-West significance is below the Elite threshold.",
        "repair_policy": ["simplify", "smooth with EWMA or decay_linear", "test justified group normalization"],
        "terminal": False,
    },
    "temporal_instability": {
        "diagnosis": "Positive IC does not appear in enough walk-forward folds.",
        "repair_policy": ["simplify", "reduce transient feature dependence", "test a slower same-mechanism variant"],
        "terminal": False,
    },
    "cost_sensitivity": {
        "diagnosis": "Predictive evidence exists but the high-cost Sharpe gate fails.",
        "repair_policy": ["smooth with EWMA or decay_linear", "test a slower signal horizon"],
        "terminal": False,
    },
    "excessive_turnover": {
        "diagnosis": "Turnover exceeds the configured Elite maximum.",
        "repair_policy": ["use rolling_mean or EWMA", "test decay_linear", "test a longer horizon"],
        "terminal": False,
    },
    "redundancy": {
        "diagnosis": "The candidate is too correlated with an existing Parent cluster.",
        "repair_policy": ["pivot to an orthogonal mechanism", "combine only complementary low-correlation parents"],
        "terminal": False,
    },
    "complexity_without_incremental_value": {
        "diagnosis": "Complexity increased without sufficient incremental IC or novelty.",
        "repair_policy": ["simplify expression", "pivot to an independent information source"],
        "terminal": False,
    },
}


def metric_failure_types(record, diagnostic: dict, config: dict) -> list[str]:
    """Translate configured gate evidence into specific, non-integrity failures."""
    failed = set(diagnostic.get("failed_gates", ()))
    result: list[str] = []
    mean_ic = record.mean_rank_ic
    if "mean_ic" in failed:
        result.append("weak_predictive_signal")
    if "tstat" in failed and mean_ic is not None and mean_ic > 0:
        result.append("low_statistical_significance")
    if "positive_folds" in failed and mean_ic is not None and mean_ic > 0:
        result.append("temporal_instability")
    if "high_cost_sharpe" in failed and mean_ic is not None and mean_ic > 0:
        result.append("cost_sensitivity")
    if "turnover" in failed:
        result.append("excessive_turnover")
    if (
        "redundancy" in record.failure_codes
        or (
            record.redundancy_corr is not None
            and abs(record.redundancy_corr)
            >= float(config.get("redundancy_correlation", 0.85))
        )
    ):
        result.append("redundancy")
    if any("complexity" in code for code in record.failure_codes) and (
        "redundancy" in record.failure_codes
        or record.residual_ic is None
        or record.residual_ic < float(config.get("min_incremental_ic", 0.001))
    ):
        result.append("complexity_without_incremental_value")
    return list(dict.fromkeys(result))


def valid_group_reproposals(
    spec: FactorSpec, *, round_id: int, allowed_groups: tuple[str, ...]
) -> list[FactorSpec]:
    """Re-propose an invalid group AST as parent-free, valid-group alternatives."""
    if spec.expression is None:
        return []

    def rewrite(node: Expr, group: str) -> tuple[Expr, bool]:
        if not isinstance(node, Op):
            return node, False
        children = [rewrite(child, group) for child in node.args]
        params = dict(node.params)
        changed = any(was_changed for _, was_changed in children)
        if node.name in GROUP_OPERATORS and params.get("group") not in allowed_groups:
            params["group"] = group
            changed = True
        return Op(node.name, tuple(child for child, _ in children), params), changed

    proposals = []
    for group in allowed_groups:
        expression, changed = rewrite(spec.expression, group)
        if not changed:
            continue
        proposals.append(
            replace(
                spec,
                factor_id=f"r{round_id}_valid_group_{spec.factor_id}_{group}",
                generation=round_id,
                parent_ids=(),
                evidence_factor_ids=(),
                hypothesis=(
                    f"{spec.hypothesis.rstrip()} Test this peer-relative effect "
                    f"within observed {group} groups."
                ),
                expression=expression,
                mutation_reason=(
                    f"Re-propose an unsupported group with the observed {group} field."
                ),
                proposal_type="integrity_reproposal",
                targeted_failure="unsupported_group",
            )
        )
    return proposals


def integrity_response_counts(outcomes: list) -> dict[str, int]:
    counts = Counter(
        code
        for outcome in outcomes
        if not outcome.record.integrity_passed
        for code in outcome.record.failure_codes
        if code in INTEGRITY_RESPONSES
    )
    return {code: counts[code] for code in PRIORITY if counts[code]}


def dominant_integrity_response(round_summary: dict) -> tuple[str, dict] | None:
    """Route only when integrity errors dominate an unsuccessful round."""
    if any(
        round_summary.get("tier_counts", {}).get(tier, 0)
        for tier in ("PARENT", "ELITE")
    ):
        return None
    evaluated = int(round_summary.get("evaluated", 0))
    counts = round_summary.get("integrity_failure_counts", {})
    failure_total = int(
        round_summary.get("integrity_failure_total", sum(counts.values()))
    )
    if not evaluated or failure_total * 2 < evaluated:
        return None
    code = max(
        PRIORITY, key=lambda item: (int(counts.get(item, 0)), -PRIORITY.index(item))
    )
    if not counts.get(code):
        return None
    return code, INTEGRITY_RESPONSES[code]
