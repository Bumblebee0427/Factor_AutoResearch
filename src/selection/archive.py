"""Bounded exploratory parent pool and persistent elite archive."""

from __future__ import annotations

from src.core.schemas import FactorArtifact


class ParentPool:
    def __init__(self, capacity: int = 20) -> None:
        self.capacity = capacity
        self.items: dict[str, FactorArtifact] = {}

    def add(self, artifact: FactorArtifact) -> None:
        self.items[artifact.spec.factor_id] = artifact
        ranked = sorted(
            self.items.values(), key=lambda item: item.quality, reverse=True
        )[: self.capacity]
        self.items = {item.spec.factor_id: item for item in ranked}


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
