#!/usr/bin/env python3
"""Make one minimal structured model request without loading market data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.research.llm_generator import LLMFactorGenerator
from src.research.memory import ResearchState
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Override the configured model for this smoke request.")
    args = parser.parse_args()
    config = load_config(PROJECT_ROOT / "config.yaml")
    llm_config = dict(config["llm"])
    if args.model:
        llm_config["resolved_model"] = args.model
    llm_config["required"] = True
    llm_config["max_proposals_per_generation"] = 2
    llm_config["max_complexity"] = int(config["gates"]["max_complexity"])
    llm_config["max_proposals_per_parent"] = int(
        config["search"]["max_llm_proposals_per_parent"]
    )
    result = LLMFactorGenerator(llm_config).propose(
        generation=1,
        state=ResearchState(search_budget_remaining=2),
        recent_records=[],
        parent_specs={},
        parent_decisions={},
        tested_ids=set(),
    )
    print(
        json.dumps(
            {
                "used_llm": result.used_llm,
                "reason": result.reason,
                "model": result.model,
                "response_id": result.response_id,
                "usage": result.usage,
                "candidate_ids": [
                    candidate.factor_id for candidate in result.candidates
                ],
                "rejected": result.rejected,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
