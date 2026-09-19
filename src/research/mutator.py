"""Bounded exploitation mutations around promoted parents."""

from __future__ import annotations

from dataclasses import replace

from src.factors.schema import FactorSpec


PRICE_WINDOWS = (5, 20, 60)


def mutate(parent: FactorSpec, generation: int, limit: int = 3) -> list[FactorSpec]:
    children: list[FactorSpec] = []
    if parent.base_feature in {"return", "volatility", "volume_shock"}:
        for window in PRICE_WINDOWS:
            if window == parent.window:
                continue
            children.append(
                replace(
                    parent,
                    factor_id=f"g{generation}_{parent.factor_id}_window{window}",
                    generation=generation,
                    parent_ids=(parent.factor_id,),
                    window=window,
                    mutation_reason="neighboring lookback around a promoted parent",
                    demonstration_only=False,
                )
            )
    if parent.cs_operator == "rank":
        children.append(
            replace(
                parent,
                factor_id=f"g{generation}_{parent.factor_id}_zscore",
                generation=generation,
                parent_ids=(parent.factor_id,),
                cs_operator="zscore",
                mutation_reason="test transform robustness",
                demonstration_only=False,
            )
        )
    return children[:limit]
