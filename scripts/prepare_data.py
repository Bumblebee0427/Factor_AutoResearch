#!/usr/bin/env python3
"""Build separate point-in-time research and locked holdout panels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import prepare_research_datasets
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing panel."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import json

    config = load_config(args.config)
    summary = prepare_research_datasets(
        config,
        project_root=PROJECT_ROOT,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
