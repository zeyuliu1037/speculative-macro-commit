#!/usr/bin/env python3
"""Summarize fresh AppWorld baseline, SA, and SMC run directories.

Accepted directories are either standalone ``smc_appworld.run`` outputs or the
``sa/`` and ``smc/`` arm subdirectories produced by
``scripts/run_appworld_paired.py``.  Output contains metrics only; it never
copies task instructions, model messages, API arguments, or tool responses.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smc_appworld.result_summary import SummaryError, build_run_summary  # noqa: E402


class RunAnalysisError(RuntimeError):
    """Raised when a run directory is not a readable compact AppWorld run."""


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunAnalysisError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RunAnalysisError(f"expected a JSON object: {path}")
    return payload


def _count_value(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RunAnalysisError(f"{label} must be a non-negative integer, got {value!r}")
    return value


def load_run(run_dir: Path, expected_mode: str) -> dict[str, Any]:
    """Load one run directory and return a metrics-only arm report."""

    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise RunAnalysisError(f"run directory not found: {run_dir}")
    config = _read_object(run_dir / "config.json")
    summary_path = run_dir / "summary.json"
    recorded_summary = _read_object(summary_path) if summary_path.is_file() else {}
    results_dir = run_dir / "results"
    if not results_dir.is_dir():
        raise RunAnalysisError(f"results directory not found: {results_dir}")

    rows: list[dict[str, Any]] = []
    result_paths = sorted(results_dir.glob("*.json"))
    for path in result_paths:
        row = _read_object(path)
        rows.append(row)
    if not rows and not recorded_summary:
        raise RunAnalysisError(f"no result files or summary found: {run_dir}")

    issues: list[str] = []
    if not summary_path.is_file():
        issues.append("summary.json is missing")
    modes = {
        str(value)
        for value in [config.get("mode"), recorded_summary.get("mode")]
        + [row.get("mode") for row in rows]
        if value is not None
    }
    if modes != {expected_mode}:
        issues.append(
            f"mode evidence should be only {expected_mode!r}, observed {sorted(modes)!r}"
        )

    count_candidates = {
        "config.task_count": _count_value(config.get("task_count"), "config.task_count"),
        "summary.selected": _count_value(
            recorded_summary.get("selected"), "summary.selected"
        ),
        "summary.requested": _count_value(
            recorded_summary.get("requested"), "summary.requested"
        ),
    }
    available_counts = {
        label: value for label, value in count_candidates.items() if value is not None
    }
    if available_counts:
        selected = next(iter(available_counts.values()))
        if len(set(available_counts.values())) != 1:
            issues.append(f"selected-task counts disagree: {available_counts}")
            selected = next(
                value
                for value in (
                    count_candidates["config.task_count"],
                    count_candidates["summary.selected"],
                    count_candidates["summary.requested"],
                )
                if value is not None
            )
    else:
        selected = len(rows)
        issues.append("selected-task count absent; inferred from result files")

    task_ids = [str(row.get("task_id", "")) for row in rows]
    if any(not task_id for task_id in task_ids):
        issues.append("one or more result records have no task_id")
    if len(task_ids) != len(set(task_ids)):
        issues.append("duplicate task_id values found in result records")
    for index, (path, task_id) in enumerate(zip(result_paths, task_ids)):
        if task_id and path.stem != task_id:
            issues.append(
                f"result filename/task_id mismatch at sorted result index {index}"
            )

    metrics = build_run_summary(
        expected_mode, selected, rows, include_results=False
    )
    for field in (
        "selected",
        "requested",
        "observed",
        "completed",
        "errors",
        "correct",
        "reward_sum",
        "total_wall_s",
        "macro_hits",
        "macro_steps_skipped",
        "macro_fire_tasks",
    ):
        if field not in recorded_summary:
            continue
        recorded = recorded_summary[field]
        recomputed = metrics[field]
        if isinstance(recorded, (int, float)) and isinstance(recomputed, (int, float)):
            same = math.isclose(
                float(recorded), float(recomputed), rel_tol=0.0, abs_tol=1e-9
            )
        else:
            same = recorded == recomputed
        if not same:
            issues.append(
                f"summary.{field} is stale: recorded {recorded!r}, "
                f"recomputed {recomputed!r}"
            )
    inline_results = recorded_summary.get("results")
    if isinstance(inline_results, list):
        inline_ids = [
            str(row.get("task_id", "")) if isinstance(row, dict) else ""
            for row in inline_results
        ]
        if len(inline_ids) != len(task_ids) or set(inline_ids) != set(task_ids):
            issues.append("summary.results task IDs differ from per-task result files")
    if metrics["selected"] == 0:
        issues.append("run selected zero tasks")
    if metrics["errors"]:
        issues.append(f"{metrics['errors']} task result(s) contain error_type")
    if metrics["missing"]:
        issues.append(f"{metrics['missing']} selected task result(s) are missing")
    if metrics["extra_results"]:
        issues.append(f"{metrics['extra_results']} stale/extra result file(s) found")
    if not metrics["complete_wall_coverage"]:
        issues.append(
            f"wall time present for {metrics['wall_tasks']}/{metrics['completed']} completed tasks"
        )

    return {
        **metrics,
        "complete": not issues,
        "integrity_issues": issues,
        "task_ids": task_ids,
    }


def _comparison(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    def delta(field: str) -> float | None:
        left = reference.get(field)
        right = candidate.get(field)
        if left is None or right is None:
            return None
        return float(right) - float(left)

    reference_wall = reference.get("mean_wall_s_completed")
    candidate_wall = candidate.get("mean_wall_s_completed")
    speedup = None
    reduction = None
    if reference_wall is not None and candidate_wall not in (None, 0):
        speedup = float(reference_wall) / float(candidate_wall)
    if reference_wall not in (None, 0) and candidate_wall is not None:
        reduction = 100.0 * (float(reference_wall) - float(candidate_wall)) / float(
            reference_wall
        )
    return {
        "reference": reference["mode"],
        "candidate": candidate["mode"],
        "correct_delta": int(candidate["correct"]) - int(reference["correct"]),
        "accuracy_selected_delta_pp": (
            100.0 * delta("accuracy_selected")
            if delta("accuracy_selected") is not None
            else None
        ),
        "accuracy_completed_delta_pp": (
            100.0 * delta("accuracy_completed")
            if delta("accuracy_completed") is not None
            else None
        ),
        "mean_wall_s_delta": (
            float(candidate_wall) - float(reference_wall)
            if reference_wall is not None and candidate_wall is not None
            else None
        ),
        "wall_speedup_reference_over_candidate": speedup,
        "wall_reduction_pct": reduction,
    }


def analyze_runs(
    baseline_dir: Path, sa_dir: Path, smc_dir: Path
) -> dict[str, Any]:
    arms = {
        "baseline": load_run(baseline_dir, "baseline"),
        "sa": load_run(sa_dir, "sa"),
        "smc": load_run(smc_dir, "smc"),
    }
    cross_arm_issues: list[str] = []
    selected_counts = {arm: report["selected"] for arm, report in arms.items()}
    if len(set(selected_counts.values())) != 1:
        cross_arm_issues.append(f"selected-task counts differ across arms: {selected_counts}")
    task_sets = {arm: set(report["task_ids"]) for arm, report in arms.items()}
    if not (task_sets["baseline"] == task_sets["sa"] == task_sets["smc"]):
        cross_arm_issues.append("observed task-id sets differ across arms")

    # Task identities are used for integrity checks only, not copied to the
    # public metric report.
    for report in arms.values():
        report.pop("task_ids", None)
    comparisons = [
        _comparison(arms["baseline"], arms["sa"]),
        _comparison(arms["baseline"], arms["smc"]),
        _comparison(arms["sa"], arms["smc"]),
    ]
    clean = not cross_arm_issues and all(report["complete"] for report in arms.values())
    return {
        "schema_version": 1,
        "artifact": "fresh AppWorld run comparison",
        "definitions": {
            "selected": "tasks chosen before execution",
            "completed": "result records without error_type",
            "correct": "completed tasks with reward == 1.0",
            "accuracy_selected": "correct / selected",
            "accuracy_completed": "correct / completed",
            "mean_wall_s": "explicit completed-task wall sum / wall_tasks",
            "mean_wall_s_completed": (
                "mean_wall_s only when every completed task has explicit wall time"
            ),
            "macro_hits": "sum over completed tasks",
            "macro_steps_skipped": "sum over completed tasks",
            "macro_fire_tasks": "completed tasks with meta_tool_hits > 0",
        },
        "status": "pass" if clean else "warning",
        "cross_arm_issues": cross_arm_issues,
        "arms": arms,
        "comparisons": comparisons,
    }


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    number = float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def format_table(report: dict[str, Any]) -> str:
    headers = [
        "arm", "selected", "done", "errors", "missing", "correct",
        "acc/selected", "acc/done", "wall total", "wall mean", "wall n",
        "hits", "skips", "fire tasks",
    ]
    rows: list[list[str]] = []
    for arm in ("baseline", "sa", "smc"):
        item = report["arms"][arm]
        rows.append(
            [
                arm,
                str(item["selected"]),
                str(item["completed"]),
                str(item["errors"]),
                str(item["missing"]),
                str(item["correct"]),
                _format_number(item["accuracy_selected"], 4),
                _format_number(item["accuracy_completed"], 4),
                _format_number(item["total_wall_s"]),
                _format_number(item["mean_wall_s"]),
                str(item["wall_tasks"]),
                str(item["macro_hits"]),
                str(item["macro_steps_skipped"]),
                str(item["macro_fire_tasks"]),
            ]
        )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def render(row: list[str]) -> str:
        return "  ".join(value.rjust(width) for value, width in zip(row, widths))

    lines = [render(headers), render(["-" * width for width in widths])]
    lines.extend(render(row) for row in rows)
    lines.append("")
    lines.append(f"status: {report['status']}")
    for issue in report["cross_arm_issues"]:
        lines.append(f"warning: {issue}")
    for arm in ("baseline", "sa", "smc"):
        for issue in report["arms"][arm]["integrity_issues"]:
            lines.append(f"warning [{arm}]: {issue}")
    lines.append("")
    lines.append("comparisons (candidate relative to reference):")
    for item in report["comparisons"]:
        lines.append(
            "  {candidate} vs {reference}: accuracy(selected) {accuracy} pp, "
            "mean-wall {wall} s, speedup {speedup}x".format(
                candidate=item["candidate"],
                reference=item["reference"],
                accuracy=_format_number(item["accuracy_selected_delta_pp"]),
                wall=_format_number(item["mean_wall_s_delta"]),
                speedup=_format_number(
                    item["wall_speedup_reference_over_candidate"], 4
                ),
            )
        )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--sa", type=Path, required=True)
    parser.add_argument("--smc", type=Path, required=True)
    parser.add_argument("--format", choices=("table", "json"), default="table")
    parser.add_argument("--output", type=Path, help="optional metrics-only JSON output")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return non-zero for incomplete, mismatched, or partially timed runs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = analyze_runs(args.baseline, args.sa, args.smc)
    except (RunAnalysisError, SummaryError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    if args.format == "json":
        sys.stdout.write(serialized)
    else:
        print(format_table(report))
    return 2 if args.strict and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
