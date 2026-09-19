#!/usr/bin/env python3
"""Combine already completed deterministic and LLM run summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.research.comparison import render_comparison_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summaries = [json.loads(path.read_text()) for path in args.summaries]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"arms": summaries, "holdout_evaluated": False}
    (args.output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "comparison.md").write_text(
        render_comparison_markdown(summaries), encoding="utf-8"
    )
    print(args.output_dir / "comparison.md")


if __name__ == "__main__":
    main()
