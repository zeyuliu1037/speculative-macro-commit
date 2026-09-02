#!/usr/bin/env python3
"""Run the tau2 Telecom CLI from a checked-in paper configuration."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "tau2_telecom" / "scripts" / "run_tau2_telecom.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="JSON file containing an 'args' string list")
    parser.add_argument("--dry-run", action="store_true")
    known, extra = parser.parse_known_args()
    payload = json.loads(Path(known.config).read_text(encoding="utf-8"))
    raw_args = payload.get("args")
    if not isinstance(raw_args, list) or not all(isinstance(item, str) for item in raw_args):
        raise ValueError("config must contain an 'args' list of strings")
    expanded = [item.replace("${REPO_ROOT}", str(REPO_ROOT)) for item in raw_args]
    command = [sys.executable, str(RUNNER), *expanded, *extra]
    print(shlex.join(command), flush=True)
    if known.dry_run:
        return 0
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())

