"""Walk-forward definitions and the one-time holdout firewall."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class WalkForwardFold:
    name: str
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date


def folds_from_config(config: dict) -> list[WalkForwardFold]:
    return [
        WalkForwardFold(
            name=row["name"],
            train_start=date.fromisoformat(str(row["train_start"])),
            train_end=date.fromisoformat(str(row["train_end"])),
            validation_start=date.fromisoformat(str(row["validation_start"])),
            validation_end=date.fromisoformat(str(row["validation_end"])),
        )
        for row in config["walk_forward"]["folds"]
    ]


class HoldoutGuard:
    def __init__(self, holdout_year: int) -> None:
        self.holdout_year = holdout_year
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        self._frozen = True

    def authorize(self, year: int) -> None:
        if year == self.holdout_year and not self._frozen:
            raise RuntimeError(
                f"Holdout {year} is locked. Freeze factors, thresholds, and code first."
            )
