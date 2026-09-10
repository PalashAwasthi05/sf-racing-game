"""CLI adapter for the generic facade capture planner."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from general_world.facade_capture_selection import run_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Select recent, visible facade captures")
    parser.add_argument("--coverage", required=True, type=Path)
    parser.add_argument("--normalized-input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=160)
    args = parser.parse_args()
    plan = run_paths(args.coverage, args.normalized_input, args.output, limit=args.limit)
    report = plan["planReport"]
    print(
        "selected={selected} frames={covered}/{total} twice={twice}".format(
            selected=report["selectedCount"],
            covered=report["frameCounts"]["covered"],
            total=report["frameCounts"]["total"],
            twice=report["frameCounts"]["coveredTwiceIndependentSequence"],
        )
    )


if __name__ == "__main__":
    main()

