from __future__ import annotations

import json
from types import MethodType, SimpleNamespace

import pandas as pd

from src.factors.schema import FactorSpec
from src.research.llm_generator import (
    FactorProposal,
    FactorProposalBatch,
    LLMFactorGenerator,
)
from src.research.loop import ResearchLoop
from src.utils.logging import ExperimentRecord


class FakeResponses:
    def __init__(self, batch: FactorProposalBatch) -> None:
        self.batch = batch
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_parsed=self.batch,
            id="resp_two_generation_fake",
            usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 100}),
        )


def test_two_generation_fake_llm_closed_loop(tmp_path) -> None:
    config = {
        "paths": {"experiment_dir": str(tmp_path / "experiments")},
        "walk_forward": {"holdout_year": 2016, "folds": []},
        "search": {
            "generations": 2,
            "max_candidates": 10,
            "max_mutations_per_parent": 3,
            "max_llm_proposals_per_parent": 3,
        },
        "gates": {"max_complexity": 4},
        "llm": {},
        "project": {"random_seed": 42},
    }
    parent = FactorSpec(
        factor_id="quality_parent",
        generation=0,
        parent_ids=(),
        family="fundamental",
        hypothesis="High profitability may persist because operating quality is underpriced.",
        base_feature="after_tax_roe",
        cs_operator="winsorize_zscore",
    )
    batch = FactorProposalBatch(
        research_summary="Use the promoted quality evidence in a distinct margin recipe.",
        proposals=[
            FactorProposal(
                factor_id="quality_margin_child",
                parent_ids=["quality_parent"],
                family="fundamental",
                hypothesis="Operating margins may confirm durable profitability through a distinct accounting signal.",
                base_feature="operating_margin",
                ts_operator="identity",
                window=None,
                cs_operator="winsorize_zscore",
                interaction_feature=None,
                interaction_window=None,
                direction=1,
                mutation_reason="Test a distinct quality primitive after the parent passed.",
                proposal_type="exploitation",
                evidence_factor_ids=["quality_parent"],
                targeted_failure="none",
                expected_metric_effect="Preserve stable positive IC with lower redundancy.",
                falsification_condition="Retire if mean IC is non-positive or unstable.",
            )
        ],
    )
    responses = FakeResponses(batch)
    generator = LLMFactorGenerator(
        {
            "enabled": True,
            "provider": "openai",
            "model": "gpt-5.6-luna",
            "reasoning_effort": "low",
            "max_proposals_per_generation": 6,
            "max_proposals_per_parent": 3,
            "max_complexity": 4,
            "allowed_windows": [1, 5, 10, 20, 60],
        },
        client=SimpleNamespace(responses=responses),
    )
    panel = pd.DataFrame(
        {"date": pd.to_datetime(["2015-01-02"]), "symbol": ["AAA"], "close": [10.0]}
    )
    loop = ResearchLoop(config, panel, llm_generator=generator)

    def fake_evaluate(self, spec):
        decision = "PROMOTE" if spec.generation == 0 else "HOLD"
        reasons = (
            ("passed predictive, economic, stability, complexity, and novelty gates",)
            if decision == "PROMOTE"
            else ("IC t-stat below promotion threshold",)
        )
        record = ExperimentRecord(
            factor_id=spec.factor_id,
            generation=spec.generation,
            parent_ids=spec.parent_ids,
            family=spec.family,
            hypothesis=spec.hypothesis,
            canonical_formula=spec.canonical_formula,
            integrity_passed=True,
            integrity_issues=(),
            fold_metrics=({"mean_ic": 0.01},),
            mean_rank_ic=0.01,
            ic_tstat=1.2,
            positive_fold_count=1,
            long_short_sharpe=0.2,
            high_cost_sharpe=0.1,
            turnover=0.2,
            max_drawdown=-0.05,
            redundancy_corr=0.0,
            closest_factor_id=None,
            residual_ic=None,
            decision=decision,
            reasons=reasons,
            failure_codes=(
                () if decision == "PROMOTE" else ("low_statistical_significance",)
            ),
            multi_horizon_mean_ic={"1d": 0.005, "5d": 0.01},
            fold_ic_sign_consistency=1.0,
            naive_ic_tstat=1.4,
        )
        return record, pd.Series([1.0], index=self.panel.index)

    loop.evaluate_candidate = MethodType(fake_evaluate, loop)
    records = loop.run(initial_candidates=[parent])

    assert [record.generation for record in records] == [0, 1]
    assert records[1].factor_id == "quality_margin_child"
    assert loop.state.generation == 2
    assert loop.state.family_summary_dict()["fundamental"]["tested"] == 2
    assert responses.calls[0]["reasoning"] == {
        "mode": "standard",
        "effort": "low",
    }
    events = [
        json.loads(line)
        for line in (tmp_path / "experiments" / "generation_events.jsonl")
        .read_text()
        .splitlines()
    ]
    assert any(event.get("used_llm") for event in events)
    assert events[-1]["event"] == "evaluation"
