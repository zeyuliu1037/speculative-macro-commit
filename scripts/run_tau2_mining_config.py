#!/usr/bin/env python3
"""Expand a checked-in tau2 label-collection or mining configuration."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASES = {
    "collect": (
        REPO_ROOT / "tau2_telecom" / "scripts" / "collect_s1_labels.py",
        "collect_args",
    ),
    "mine": (
        REPO_ROOT / "tau2_telecom" / "scripts" / "mine_macro_library.py",
        "mine_args",
    ),
}


def build_command(config_path: Path, phase: str, extra: list[str]) -> list[str]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    script, key = PHASES[phase]
    configured = payload.get(key)
    if not isinstance(configured, list) or not all(
        isinstance(item, str) for item in configured
    ):
        raise ValueError(f"config must contain a string list named {key!r}")
    return [sys.executable, str(script), *configured, *extra]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("phase", choices=sorted(PHASES))
    parser.add_argument("--dry-run", action="store_true")
    known, extra = parser.parse_known_args()
    command = build_command(known.config, known.phase, extra)
    print(shlex.join(command), flush=True)
    if known.dry_run:
        return 0
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
