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
