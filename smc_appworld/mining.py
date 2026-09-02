"""Command-line entry point for mining the AppWorld macro-step library.

This module intentionally wraps :class:`MetaToolMiner` without changing its
camera-ready n-gram algorithm.  Run it from the release root with::

    python -m smc_appworld.mining \
        --trajectory-dir raw_trajectories/appworld_train_maxsteps50 \
        --output runs/mining/appworld_train50.json \
        --min-frequency 5 \
        --min-coverage 0.10

The checked-in config records the paper settings; CLI values may override it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from smc_appworld.runtime.meta_tool import MetaToolLibrary, MetaToolMiner


DEFAULT_MIN_FREQUENCY = 5
DEFAULT_MIN_COVERAGE = 0.10
CONFIG_KEYS = {
    "trajectory_dir",
    "output",
    "min_frequency",
    "min_coverage",
}


@dataclass(frozen=True)
class MiningOptions:
    """Fully resolved AppWorld mining options."""

    trajectory_dir: Path
    output: Path
    min_frequency: int
    min_coverage: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Mine the camera-ready AppWorld frequency/coverage macro-step "
            "library from per-task trajectory JSON files."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Optional JSON config. Command-line values override matching "
            "config fields."
        ),
    )
    parser.add_argument(
        "--trajectory-dir",
        "--trace-dir",
        dest="trajectory_dir",
        type=Path,
        help="Directory containing one AppWorld trajectory JSON per task.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for the mined macro-step library JSON.",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        help=(
            "Minimum number of occurrences across all trajectories "
            f"(default: {DEFAULT_MIN_FREQUENCY})."
        ),
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        help=(
            "Minimum fraction of trajectory files containing a pattern "
            f"(default: {DEFAULT_MIN_COVERAGE})."
        ),
    )
    return parser


def _load_config(parser: argparse.ArgumentParser, path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        parser.error(f"cannot read config {path}: {exc}")
    except json.JSONDecodeError as exc:
        parser.error(f"invalid JSON config {path}: {exc}")

    if not isinstance(payload, dict):
        parser.error(f"config {path} must contain a JSON object")
    unknown = sorted(set(payload) - CONFIG_KEYS)
    if unknown:
        parser.error(f"unknown config field(s): {', '.join(unknown)}")
    return payload


def _coalesce(command_line: Any, config: dict[str, Any], key: str, default: Any = None) -> Any:
    if command_line is not None:
        return command_line
    return config.get(key, default)


def resolve_options(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> MiningOptions:
    config = _load_config(parser, args.config)
    trajectory_value = _coalesce(args.trajectory_dir, config, "trajectory_dir")
    output_value = _coalesce(args.output, config, "output")
    min_frequency = _coalesce(
        args.min_frequency, config, "min_frequency", DEFAULT_MIN_FREQUENCY
    )
    min_coverage = _coalesce(
        args.min_coverage, config, "min_coverage", DEFAULT_MIN_COVERAGE
    )

    if trajectory_value is None:
        parser.error("--trajectory-dir is required (directly or via --config)")
    if output_value is None:
        parser.error("--output is required (directly or via --config)")

    if not isinstance(trajectory_value, (str, Path)):
        parser.error("trajectory_dir must be a path string")
    if not isinstance(output_value, (str, Path)):
        parser.error("output must be a path string")
    trajectory_dir = Path(trajectory_value)
    output = Path(output_value)
    if not trajectory_dir.is_dir():
        parser.error(f"trajectory directory does not exist: {trajectory_dir}")
    if isinstance(min_frequency, bool) or not isinstance(min_frequency, int):
        parser.error("min_frequency must be an integer")
    if min_frequency < 1:
        parser.error("min_frequency must be at least 1")
    if isinstance(min_coverage, bool) or not isinstance(min_coverage, (int, float)):
        parser.error("min_coverage must be a number")
    min_coverage = float(min_coverage)
    if not 0.0 <= min_coverage <= 1.0:
        parser.error("min_coverage must be between 0 and 1 inclusive")
    if output.exists() and output.is_dir():
        parser.error(f"output path is a directory: {output}")

    return MiningOptions(
        trajectory_dir=trajectory_dir,
        output=output,
        min_frequency=min_frequency,
        min_coverage=min_coverage,
    )


def _trajectory_file_count(directory: Path) -> int:
    return sum(
        1
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix == ".json"
        and not path.name.startswith(("selected", "metrics", "run"))
    )


def run(options: MiningOptions) -> int:
    """Mine and write a library, returning its pattern count."""
    trajectory_count = _trajectory_file_count(options.trajectory_dir)
    if trajectory_count == 0:
        raise ValueError(
            f"no eligible trajectory JSON files in {options.trajectory_dir}"
        )

    miner = MetaToolMiner(
        min_frequency=options.min_frequency,
        min_coverage=options.min_coverage,
    )
    meta_tools = miner.mine_from_trajectory_dir(str(options.trajectory_dir))
    options.output.parent.mkdir(parents=True, exist_ok=True)
    MetaToolLibrary(meta_tools=meta_tools).save(str(options.output))

    print(
        f"Mined {len(meta_tools)} patterns from {trajectory_count} trajectories "
        f"(min_frequency={options.min_frequency}, "
        f"min_coverage={options.min_coverage:g})."
    )
    print(f"Wrote {options.output}")
    return len(meta_tools)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    options = resolve_options(parser, parser.parse_args(argv))
    try:
        run(options)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
