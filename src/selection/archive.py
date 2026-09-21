"""Bounded exploratory parent pool and persistent elite archive."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.schemas import FactorArtifact
from src.evaluation.redundancy import mean_cross_sectional_correlation


@dataclass(frozen=True)
class ParentAdmission:
    status: str
    representative_id: str | None
    absolute_correlation: float | None = None
    evicted_ids: tuple[str, ...] = ()


class ParentPool:
    def __init__(
        self,
        capacity: int = 20,
        *,
        correlation_threshold: float = 0.90,
        minimum_overlap_dates: int = 20,
        minimum_names_per_date: int = 5,
        maximum_comparison_dates: int = 96,
    ) -> None:
        if not 0 < correlation_threshold <= 1:
            raise ValueError("Parent correlation threshold must be in (0, 1].")
        self.capacity = capacity
        self.correlation_threshold = correlation_threshold
        self.minimum_overlap_dates = minimum_overlap_dates
        self.minimum_names_per_date = minimum_names_per_date
        self.maximum_comparison_dates = maximum_comparison_dates
        self.items: dict[str, FactorArtifact] = {}
        self.qualifying_counts: dict[str, int] = {}
        self._clusters: dict[str, list[FactorArtifact]] = {}

    def add(
        self,
        artifact: FactorArtifact,
        signal: pd.Series,
        signals: dict[str, pd.Series],
        dates: pd.Series,
    ) -> ParentAdmission:
        mechanism = artifact.mechanism
        self.qualifying_counts[mechanism] = self.qualifying_counts.get(mechanism, 0) + 1
        clusters = self._clusters.setdefault(mechanism, [])
        closest: tuple[float, int, FactorArtifact] | None = None
        for index, incumbent in enumerate(clusters):
            existing = signals.get(incumbent.spec.factor_id)
            if existing is None:
                continue
            finite = (
                signal.notna()
                & existing.notna()
                & np.isfinite(signal)
                & np.isfinite(existing)
            )
            comparable_dates = finite.groupby(dates).sum()
            eligible_dates = comparable_dates.index[
                comparable_dates >= self.minimum_names_per_date
            ]
            if len(eligible_dates) < self.minimum_overlap_dates:
                continue
            if len(eligible_dates) > self.maximum_comparison_dates:
                positions = np.linspace(
                    0,
                    len(eligible_dates) - 1,
                    num=self.maximum_comparison_dates,
                    dtype=int,
                )
                eligible_dates = eligible_dates[positions]
            eligible = dates.isin(eligible_dates)
            corr = mean_cross_sectional_correlation(
                signal[eligible], existing[eligible], dates[eligible]
            )
            if np.isfinite(corr) and abs(corr) > self.correlation_threshold:
                if closest is None or abs(corr) > closest[0]:
                    closest = (abs(corr), index, incumbent)
        if closest is not None:
            corr, index, incumbent = closest
            better = artifact.quality > incumbent.quality or (
                artifact.quality == incumbent.quality
                and artifact.spec.complexity < incumbent.spec.complexity
            )
            if not better:
                return ParentAdmission("suppressed", incumbent.spec.factor_id, corr)
            clusters[index] = artifact
            replaced = incumbent.spec.factor_id
        else:
            clusters.append(artifact)
            replaced = None
        previous = set(self.items)
        ranked = sorted(
            (item for members in self._clusters.values() for item in members),
            key=lambda item: item.quality,
            reverse=True,
        )[: self.capacity]
        self.items = {item.spec.factor_id: item for item in ranked}
        evicted = tuple(sorted(previous - set(self.items)))
        return ParentAdmission(
            ("replaced" if replaced else "added")
            if artifact.spec.factor_id in self.items
            else "capacity_suppressed",
            artifact.spec.factor_id
            if artifact.spec.factor_id in self.items
            else replaced,
            closest[0] if closest else None,
            evicted,
        )

    def cluster_stats(self) -> dict[str, dict]:
        mechanisms = set(self.qualifying_counts)
        return {
            mechanism: {
                "parent_candidates": self.qualifying_counts.get(mechanism, 0),
                "unique_clusters": len(self._clusters.get(mechanism, [])),
                "active_representatives": sum(
                    item.mechanism == mechanism for item in self.items.values()
                ),
                "representative_ids": sorted(
                    item.spec.factor_id
                    for item in self.items.values()
                    if item.mechanism == mechanism
                ),
            }
            for mechanism in sorted(mechanisms)
        }


class EliteArchive:
    def __init__(self, capacity: int = 12) -> None:
        self.capacity = capacity
        self.items: dict[str, FactorArtifact] = {}

    def add(self, artifact: FactorArtifact) -> None:
        same_formula = [
            item
            for item in self.items.values()
            if item.spec.canonical_formula == artifact.spec.canonical_formula
        ]
        if same_formula and same_formula[0].quality >= artifact.quality:
            return
        for item in same_formula:
            self.items.pop(item.spec.factor_id, None)
        self.items[artifact.spec.factor_id] = artifact
        ranked = sorted(
            self.items.values(), key=lambda item: item.quality, reverse=True
        )[: self.capacity]
        self.items = {item.spec.factor_id: item for item in ranked}
