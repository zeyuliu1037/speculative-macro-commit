"""Paper-relevant summaries for compact AppWorld result records.

This module intentionally depends only on the Python standard library.  Both
AppWorld runners and the offline comparison command use it so that accuracy,
wall-time, and macro-commit counters have one definition.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


MODES = frozenset({"baseline", "sa", "smc"})


class SummaryError(ValueError):
    """Raised when compact result records cannot be summarized safely."""


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise SummaryError(f"{field} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SummaryError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise SummaryError(f"{field} must be finite, got {value!r}")
    return number


def _nonnegative_counter(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise SummaryError(f"{field} must be an integer, not boolean")
    try:
        counter = int(value)
    except (TypeError, ValueError) as exc:
        raise SummaryError(f"{field} must be an integer, got {value!r}") from exc
    if counter < 0 or counter != _finite_number(value, field):
        raise SummaryError(f"{field} must be a non-negative integer, got {value!r}")
    return counter


def _wall_value(row: Mapping[str, Any], default_mode: str) -> tuple[str, float] | None:
    row_mode = str(row.get("mode", default_mode))
    preferred = "total_wall_time" if row_mode == "baseline" else "wall_time"
    fallback = "wall_time" if preferred == "total_wall_time" else "total_wall_time"
    for field in (preferred, fallback):
        value = row.get(field)
        if value is None:
            continue
        number = _finite_number(value, field)
        if number < 0:
            raise SummaryError(f"{field} must be non-negative, got {value!r}")
        return field, number
    return None


def build_run_summary(
    mode: str,
    selected: int,
    results: Sequence[Mapping[str, Any]],
    *,
    include_results: bool = True,
) -> dict[str, Any]:
    """Summarize one AppWorld arm.

    ``selected`` is the number of tasks chosen before execution. ``completed``
    counts result records without ``error_type``.  Paper accuracy is reported
    with both denominators: selected tasks (errors/missing tasks count as
    failures) and completed tasks.  A task is correct iff ``reward == 1.0``.

    Wall-time means use only completed records that contain an explicit wall
    metric; ``wall_tasks`` makes that denominator visible.  The
    ``mean_wall_s_completed`` field is populated only under complete wall
    coverage, which prevents partial telemetry from looking paper-comparable.
    """

    if mode not in MODES:
        raise SummaryError(f"unknown AppWorld mode: {mode!r}")
    if isinstance(selected, bool) or not isinstance(selected, int) or selected < 0:
        raise SummaryError(f"selected must be a non-negative integer, got {selected!r}")

    rows = list(results)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SummaryError(f"result {index} is not a JSON object")

    # Runner exceptions always write an ``error_type`` key.  Key presence is
    # deliberately decisive so a malformed empty/null error label cannot make
    # a failed task look completed.
    completed_rows = [row for row in rows if "error_type" not in row]
    error_rows = [row for row in rows if "error_type" in row]
    rewards = [
        _finite_number(row.get("reward", 0.0), "reward") for row in completed_rows
    ]
    correct = sum(reward == 1.0 for reward in rewards)

    wall_values: list[float] = []
    wall_fields: set[str] = set()
    for row in completed_rows:
        wall = _wall_value(row, mode)
        if wall is not None:
            field, value = wall
            wall_fields.add(field)
            wall_values.append(value)

    macro_hits = 0
    macro_steps_skipped = 0
    macro_fire_tasks = 0
    for row in completed_rows:
        hits = _nonnegative_counter(row.get("meta_tool_hits", 0), "meta_tool_hits")
        skipped = _nonnegative_counter(
            row.get("meta_tool_steps_skipped", 0), "meta_tool_steps_skipped"
        )
        macro_hits += hits
        macro_steps_skipped += skipped
        macro_fire_tasks += hits > 0

    observed = len(rows)
    completed = len(completed_rows)
    errors = len(error_rows)
    reward_sum = math.fsum(rewards)
    total_wall_s = math.fsum(wall_values) if wall_values else None
    mean_wall_s = total_wall_s / len(wall_values) if wall_values else None
    complete_wall_coverage = completed > 0 and len(wall_values) == completed

    summary: dict[str, Any] = {
        "schema_version": 2,
        "mode": mode,
        "selected": selected,
        # Backwards-compatible alias consumed by the release validator.
        "requested": selected,
        "observed": observed,
        "completed": completed,
        "errors": errors,
        "missing": max(selected - observed, 0),
        "extra_results": max(observed - selected, 0),
        "correct": correct,
        "accuracy_selected": correct / selected if selected else None,
        "accuracy_completed": correct / completed if completed else None,
        "reward_sum": reward_sum,
        # Backwards-compatible name; this is mean reward, not reward==1 accuracy.
        "mean_reward": reward_sum / completed if completed else None,
        "wall_fields": sorted(wall_fields),
        "wall_tasks": len(wall_values),
        "total_wall_s": total_wall_s,
        "mean_wall_s": mean_wall_s,
        "mean_wall_s_completed": mean_wall_s if complete_wall_coverage else None,
        "complete_wall_coverage": complete_wall_coverage,
        "macro_hits": macro_hits,
        "macro_steps_skipped": macro_steps_skipped,
        "macro_fire_tasks": macro_fire_tasks,
    }
    if include_results:
        summary["results"] = rows
    return summary


def run_exit_code(summary: Mapping[str, Any]) -> int:
    """Return a failing process code whenever a task result has ``error_type``."""

    return 1 if _nonnegative_counter(summary.get("errors", 0), "errors") else 0
