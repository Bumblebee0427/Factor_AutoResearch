"""Configuration loading and fail-fast validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_LEVEL = {
    "project",
    "paths",
    "data",
    "universe",
    "walk_forward",
    "evaluation",
    "search",
    "llm",
    "gates",
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")

    missing = REQUIRED_TOP_LEVEL - config.keys()
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    if len(config["walk_forward"].get("folds", [])) != 3:
        raise ValueError("The specification requires exactly three research folds.")
    if int(config["walk_forward"]["holdout_year"]) != 2016:
        raise ValueError("The untouched final holdout must remain 2016.")
    if int(config["search"]["max_candidates"]) > 300:
        raise ValueError("Search cap must not exceed the recommended 300 candidates.")
    if config["llm"].get("provider", "openai") != "openai":
        raise ValueError("Only the OpenAI LLM provider is supported.")
    if config["llm"].get("endpoint", "responses") != "responses":
        raise ValueError("The LLM generator requires the Responses API.")
    if config["llm"].get("reasoning_effort", "low") not in {
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    }:
        raise ValueError("Unsupported reasoning effort.")
    if int(config["llm"].get("max_proposals_per_generation", 0)) < 1:
        raise ValueError("LLM proposal cap must be positive.")
    horizons = [int(value) for value in config["evaluation"]["ic_horizons_days"]]
    if int(config["evaluation"]["prediction_horizon_days"]) not in horizons:
        raise ValueError("Primary prediction horizon must be included in IC horizons.")
    if any(horizon < 1 for horizon in horizons):
        raise ValueError("IC horizons must be positive.")
    if float(config["data"]["fundamental_reporting_lag_days"]) < 0:
        raise ValueError("Fundamental reporting lag cannot be negative.")
    if float(config["data"]["news_availability_lag_days"]) < 0:
        raise ValueError("News availability lag cannot be negative.")
    adaptive = config.get("adaptive_research", {})
    if int(
        adaptive.get("max_candidate_evaluations", config["search"]["max_candidates"])
    ) > int(config["search"]["max_candidates"]):
        raise ValueError(
            "Adaptive candidate budget cannot exceed the global search cap."
        )
    if int(adaptive.get("max_rounds", 1)) < 1:
        raise ValueError("Adaptive research requires at least one round.")
    return config


def config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
