from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.factors.schema import FactorSpec
from src.research.llm_generator import (
    FactorProposalBatch,
    LLMFactorGenerator,
    SingleFactorProposal,
)
from src.research.memory import ResearchState
from src.utils.logging import ExperimentRecord


def make_record(**overrides) -> ExperimentRecord:
    payload = {
        "factor_id": "price_parent",
        "generation": 0,
        "parent_ids": (),
        "family": "price",
        "hypothesis": "Medium-horizon continuation may persist after cross-sectional ranking.",
        "canonical_formula": "+1 * rank(return(20))",
        "integrity_passed": True,
        "integrity_issues": (),
        "fold_metrics": (),
        "mean_rank_ic": 0.01,
        "ic_tstat": 1.5,
        "positive_fold_count": 2,
        "long_short_sharpe": 0.4,
        "high_cost_sharpe": 0.2,
        "turnover": 0.3,
        "max_drawdown": -0.1,
        "redundancy_corr": 0.0,
        "closest_factor_id": None,
        "residual_ic": None,
        "decision": "PROMOTE",
        "reasons": ("all promotion gates passed",),
    }
    payload.update(overrides)
    return ExperimentRecord(**payload)


def make_parent() -> FactorSpec:
    return FactorSpec(
        factor_id="price_parent",
        generation=0,
        parent_ids=(),
        family="price",
        hypothesis="Medium-horizon continuation may persist after cross-sectional ranking.",
        base_feature="return",
        window=20,
        cs_operator="rank",
    )


class FakeParser:
    def __init__(self, batch: FactorProposalBatch) -> None:
        self.batch = batch
        self.call = None

    def parse(self, **kwargs):
        self.call = kwargs
        return SimpleNamespace(
            output_parsed=self.batch,
            id="resp_fake",
            usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 123}),
        )


def fake_client(batch: FactorProposalBatch):
    parser = FakeParser(batch)
    client = SimpleNamespace(
        responses=parser,
    )
    return client, parser


def generator_config(**overrides) -> dict:
    config = {
        "enabled": True,
        "provider": "openai",
        "model": "test-model",
        "max_history_records": 20,
        "max_proposals_per_generation": 6,
        "max_proposals_per_parent": 3,
        "max_complexity": 4,
        "allowed_windows": [1, 5, 10, 20, 60],
    }
    config.update(overrides)
    return config


def test_missing_api_key_uses_deterministic_fallback(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    generator = LLMFactorGenerator(generator_config())

    result = generator.propose(
        generation=1,
        state=ResearchState(),
        recent_records=[],
        parent_specs={},
        parent_decisions={},
        tested_ids=set(),
    )

    assert not result.used_llm
    assert result.reason == "missing_api_key"


def test_proposal_union_is_nested_and_single_recipe_cannot_carry_interaction_fields() -> (
    None
):
    schema = FactorProposalBatch.schema()
    encoded = json.dumps(schema)

    assert schema["type"] == "object"
    assert "anyOf" not in schema
    assert encoded.count("anyOf") == 1
    assert "allOf" not in encoded

    with pytest.raises(ValidationError):
        FactorProposalBatch.parse_obj(
            {
                "research_summary": "Malformed single recipe.",
                "proposals": [
                    {
                        "factor_id": "bad_single",
                        "parent_ids": [],
                        "hypothesis": "A single feature should not carry an interaction leg.",
                        "direction": 1,
                        "mutation_reason": "Schema boundary test.",
                        "proposal_type": "exploration",
                        "evidence_factor_ids": [],
                        "targeted_failure": "none",
                        "expected_metric_effect": "Unknown.",
                        "falsification_condition": "Retire if unstable.",
                        "recipe_kind": "single",
                        "family": "price",
                        "base_feature": "return",
                        "ts_operator": "identity",
                        "window": 20,
                        "cs_operator": "rank",
                        "interaction_feature": "volatility",
                        "interaction_window": 20,
                    }
                ],
            }
        )


def test_structured_proposal_becomes_valid_factor_spec() -> None:
    batch = FactorProposalBatch(
        research_summary="Turnover is acceptable; test a smoother continuation variant.",
        proposals=[
            SingleFactorProposal(
                factor_id="price_parent_smooth",
                parent_ids=["price_parent"],
                recipe_kind="single",
                family="price",
                hypothesis="Smoothing medium-term returns may retain continuation while reducing noisy turnover.",
                base_feature="return",
                ts_operator="rolling_mean",
                window=20,
                cs_operator="rank",
                direction=1,
                mutation_reason="Reduce turnover while preserving the promoted mechanism.",
                proposal_type="failure_repair",
                evidence_factor_ids=["price_parent"],
                targeted_failure="excessive_turnover",
                expected_metric_effect="Lower turnover with similar positive IC.",
                falsification_condition="Retire if high-cost Sharpe is still non-positive.",
            )
        ],
    )
    client, parser = fake_client(batch)
    generator = LLMFactorGenerator(generator_config(), client=client)

    result = generator.propose(
        generation=1,
        state=ResearchState(held_factor_ids=["price_parent"]),
        recent_records=[make_record(decision="HOLD")],
        parent_specs={"price_parent": make_parent()},
        parent_decisions={"price_parent": "HOLD"},
        tested_ids={"price_parent"},
    )

    assert result.used_llm
    assert result.reason == "structured_llm_proposals"
    assert result.candidates[0].parent_ids == ("price_parent",)
    assert result.candidates[0].generation == 1
    assert parser.call["text_format"] is FactorProposalBatch
    assert parser.call["reasoning"] == {"mode": "standard", "effort": "low"}
    assert parser.call["text"] == {"verbosity": "low"}
    assert "verbosity" not in parser.call
    assert parser.call["store"] is False


def test_unknown_parent_is_rejected() -> None:
    batch = FactorProposalBatch(
        research_summary="Try a continuation variant.",
        proposals=[
            SingleFactorProposal(
                factor_id="orphan_factor",
                parent_ids=["invented_parent"],
                recipe_kind="single",
                family="price",
                hypothesis="A smoother continuation signal could lower turnover without losing persistence.",
                base_feature="return",
                ts_operator="rolling_mean",
                window=20,
                cs_operator="rank",
                direction=1,
                mutation_reason="Smooth an existing mechanism.",
                proposal_type="failure_repair",
                evidence_factor_ids=["invented_parent"],
                targeted_failure="excessive_turnover",
                expected_metric_effect="Lower turnover.",
                falsification_condition="Retire if turnover is not lower.",
            )
        ],
    )
    client, _ = fake_client(batch)
    generator = LLMFactorGenerator(generator_config(), client=client)

    result = generator.propose(
        generation=1,
        state=ResearchState(),
        recent_records=[],
        parent_specs={"price_parent": make_parent()},
        parent_decisions={"price_parent": "PROMOTE"},
        tested_ids=set(),
    )

    assert not result.used_llm
    assert result.reason == "no_valid_llm_proposals"
    assert "parent is not an eligible HOLD/PROMOTE factor" in result.rejected[0]


def test_llm_context_excludes_raw_data_and_holdout_year() -> None:
    generator = LLMFactorGenerator(generator_config(enabled=False))
    context = generator.build_context(
        generation=1,
        state=ResearchState(promoted_factor_ids=["price_parent"]),
        recent_records=[make_record()],
        parent_specs={"price_parent": make_parent()},
        parent_decisions={"price_parent": "PROMOTE"},
    )

    assert context["constraints"]["raw_data_available_to_llm"] is False
    assert (
        context["constraints"]["holdout_year"]
        == "withheld; never reference or infer it"
    )
    assert "panel" not in context
