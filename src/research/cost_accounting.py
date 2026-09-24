"""Experiment-only Responses token and estimated-cost accounting."""

from __future__ import annotations

def role_usage(events: list[dict], role: str) -> dict[str, int | None]:
    sections = [event.get(role, {}) for event in events]
    completed = [item for item in sections if item.get("response_id")]

    def sum_field(field: str) -> int | None:
        if not completed:
            return 0
        values = [item.get("usage", {}).get(field) for item in completed]
        return sum(int(value) for value in values) if all(value is not None for value in values) else None

    cached = []
    cache_writes = []
    for item in completed:
        usage = item.get("usage", {})
        detail = usage.get("input_tokens_details") or {}
        cached.append(detail.get("cached_tokens", usage.get("cached_input_tokens")))
        cache_writes.append(detail.get("cache_write_tokens", usage.get("cache_write_tokens")))
    return {
        "llm_calls": len(completed),
        "input_tokens": sum_field("input_tokens"),
        "output_tokens": sum_field("output_tokens"),
        "total_tokens": sum_field("total_tokens"),
        "cached_input_tokens": (
            0 if not completed else sum(int(value) for value in cached)
            if all(value is not None for value in cached) else None
        ),
        "cache_write_tokens": (
            0 if not completed else sum(int(value) for value in cache_writes)
            if all(value is not None for value in cache_writes) else None
        ),
    }


def estimate_cost(usage: dict, model: str | None, pricing: dict) -> float | None:
    rates = pricing.get("models", {}).get(model or "")
    if rates is None:
        return None
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    cached_tokens = usage.get("cached_input_tokens")
    cache_write_tokens = usage.get("cache_write_tokens")
    if input_tokens is None or output_tokens is None or cached_tokens is None:
        return None
    write_rate = rates.get("cache_write_per_million_usd")
    if write_rate is not None and cache_write_tokens is None:
        return None
    writes = cache_write_tokens or 0
    if cached_tokens + writes > input_tokens:
        raise ValueError("Cached and cache-write tokens exceed total input tokens.")
    keys = (
        "input_per_million_usd",
        "cached_input_per_million_usd",
        "output_per_million_usd",
    )
    if any(rates.get(key) is None for key in keys):
        return None
    return (
        (input_tokens - cached_tokens - writes) * float(rates[keys[0]])
        + cached_tokens * float(rates[keys[1]])
        + writes * float(write_rate if write_rate is not None else rates[keys[0]])
        + output_tokens * float(rates[keys[2]])
    ) / 1_000_000


def combine_usage(macro: dict, micro: dict) -> dict[str, int | None]:
    combined: dict[str, int | None] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "cache_write_tokens"):
        left, right = macro.get(key), micro.get(key)
        combined[key] = left + right if left is not None and right is not None else None
    return combined


def per_denominator(numerator: int | float, denominator: int | float | None, scale: float = 1.0) -> float | None:
    return float(numerator) * scale / denominator if denominator is not None and denominator > 0 else None
