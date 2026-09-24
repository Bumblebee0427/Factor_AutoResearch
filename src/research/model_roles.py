"""Resolve role-specific API settings without changing shared LLM defaults."""

from __future__ import annotations

import os


def role_llm_config(shared: dict, role: str) -> dict:
    if role not in {"macro", "micro"}:
        raise ValueError(f"Unknown LLM role: {role}")
    config = {key: value for key, value in shared.items() if key not in {"macro", "micro"}}
    override = dict(shared.get(role, {}))
    config.update(override)
    role_env = override.get("model_env")
    if role_env and os.environ.get(role_env):
        resolved = os.environ[role_env]
    elif override.get("model"):
        resolved = override["model"]
    else:
        shared_env = shared.get("model_env", "OPENAI_FACTOR_MODEL")
        resolved = os.environ.get(shared_env, shared.get("model", "gpt-6-luna"))
    config["resolved_model"] = resolved
    return config
