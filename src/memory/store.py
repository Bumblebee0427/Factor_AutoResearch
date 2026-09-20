"""Append-only research memory with an irreversible in-process seal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core.schemas import ResearchMemory


class ResearchMemoryStore:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rounds_path = self.directory / "research_rounds.jsonl"
        self.memory_path = self.directory / "research_memory.json"
        self.seal_path = self.directory / ".sealed"
        self._sealed = self.seal_path.exists()

    @property
    def sealed(self) -> bool:
        return self._sealed

    def _assert_writable(self) -> None:
        if self._sealed:
            raise RuntimeError(
                "Research memory is sealed; final holdout cannot update it."
            )

    def append_round(self, payload: dict[str, Any]) -> None:
        self._assert_writable()
        with self.rounds_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    def save(self, memory: ResearchMemory) -> Path:
        self._assert_writable()
        self.memory_path.write_text(
            json.dumps(memory.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return self.memory_path

    def checkpoint(self, round_id: int, memory: ResearchMemory) -> Path:
        self._assert_writable()
        path = self.directory / f"checkpoint_round_{round_id:03d}.json"
        path.write_text(
            json.dumps(memory.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return path

    def seal(self) -> None:
        self.seal_path.write_text(
            "Research memory frozen before final holdout.\n", encoding="utf-8"
        )
        self._sealed = True
