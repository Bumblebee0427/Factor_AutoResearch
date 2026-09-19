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
    if int(config["llm"].get("max_proposals_per_generation", 0)) < 1:
        raise ValueError("LLM proposal cap must be positive.")
    if float(config["data"]["fundamental_reporting_lag_days"]) < 0:
        raise ValueError("Fundamental reporting lag cannot be negative.")
    if float(config["data"]["news_availability_lag_days"]) < 0:
        raise ValueError("News availability lag cannot be negative.")
    return config


def config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
