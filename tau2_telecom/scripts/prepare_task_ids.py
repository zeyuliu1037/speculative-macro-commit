#!/usr/bin/env python3
"""Write deterministic TAU2 telecom task-id lists for macro experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tau2_telecom.src.runner import select_tasks


def ids_for_split(name: str) -> list[str]:
    return [task.id for task in select_tasks(name)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--full-rest-limit", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    small = ids_for_split("small")
    train = ids_for_split("train")
    test = ids_for_split("test")
    excluded = set(small) | set(train) | set(test)
    full_rest = [task_id for task_id in ids_for_split("full") if task_id not in excluded]
    lists = {
        "small": small,
        "train": train,
        "small_train": list(dict.fromkeys(small + train)),
        "test": test,
        "full_rest": full_rest,
        "full_rest_100": full_rest[: args.full_rest_limit],
    }
    for name, values in lists.items():
        (args.out_dir / f"{name}.json").write_text(
            json.dumps(values, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.out_dir / f"{name}.txt").write_text(
            "\n".join(values) + "\n",
            encoding="utf-8",
        )
    summary = {name: len(values) for name, values in lists.items()}
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
