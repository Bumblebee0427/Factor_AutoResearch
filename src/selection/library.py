"""Build a compact, mechanism-diverse final library from elite evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.schemas import FactorArtifact
from src.evaluation.redundancy import max_library_correlation, residual_ic
from src.factors.primitives import future_return


def build_final_library(
    artifacts: list[FactorArtifact],
    signals: dict[str, pd.Series],
    panel: pd.DataFrame,
    config: dict,
) -> tuple[list[FactorArtifact], list[dict]]:
    corr_limit = float(config.get("final_correlation_threshold", 0.70))
    min_incremental = float(config.get("min_incremental_ic", 0.001))
    maximum = int(config.get("max_final_factors", 8))
    horizon = int(config.get("prediction_horizon_days", 5))
    target = future_return(panel["close"], panel["symbol"], horizon)
    # Quality first, then prefer simpler formulas when evidence is close.
    ranked = sorted(
        artifacts, key=lambda item: (-round(item.quality, 4), item.spec.complexity)
    )
    first_by_mechanism: list[FactorArtifact] = []
    seen_mechanisms: set[str] = set()
    for artifact in ranked:
        if artifact.mechanism not in seen_mechanisms:
            first_by_mechanism.append(artifact)
            seen_mechanisms.add(artifact.mechanism)
    ranked = first_by_mechanism + [
        item for item in ranked if item not in first_by_mechanism
    ]
    selected: list[FactorArtifact] = []
    selected_signals: dict[str, pd.Series] = {}
    audit: list[dict] = []
    seen_formulas: set[str] = set()
    for artifact in ranked:
        factor_id = artifact.spec.factor_id
        signal = signals.get(factor_id)
        if signal is None:
            audit.append(
                {"factor_id": factor_id, "selected": False, "reason": "missing signal"}
            )
            continue
        if artifact.spec.canonical_formula in seen_formulas:
            audit.append(
                {
                    "factor_id": factor_id,
                    "selected": False,
                    "reason": "duplicate formula",
                }
            )
            continue
        corr, closest = max_library_correlation(signal, selected_signals, panel["date"])
        incremental = None
        if closest is not None and corr > corr_limit:
            incremental = residual_ic(
                signal, selected_signals[closest], panel["date"], target
            )
            if not np.isfinite(incremental) or incremental < min_incremental:
                audit.append(
                    {
                        "factor_id": factor_id,
                        "selected": False,
                        "reason": "redundant",
                        "correlation": corr,
                        "residual_ic": incremental,
                    }
                )
                continue
        selected.append(artifact)
        selected_signals[factor_id] = signal
        seen_formulas.add(artifact.spec.canonical_formula)
        audit.append(
            {
                "factor_id": factor_id,
                "selected": True,
                "correlation": corr,
                "residual_ic": incremental,
                "mechanism": artifact.mechanism,
            }
        )
        if len(selected) >= maximum:
            break
    return selected, audit
