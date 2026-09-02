#!/usr/bin/env python3
"""Run TAU2 telecom owner-only or speculative-verify experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger
from tau2.data_model.simulation import SimulationRun
from tau2.domains.telecom.environment import get_tasks as get_telecom_tasks
from tau2.evaluator.evaluator import EvaluationType

from tau2_telecom.src.runner import (
    LLMEndpoint,
    run_tau2_telecom_task,
    select_tasks,
    write_result,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="small")
    parser.add_argument("--task-ids-file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=["owner_only", "spec_verify", "macro_skip"],
        default="owner_only",
    )
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--max-errors", type=int, default=3)
    parser.add_argument("--policy-type", choices=["manual", "workflow"], default="manual")
    parser.add_argument("--evaluation-type", default="all", choices=[x.value for x in EvaluationType])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-dir", default=str(ROOT / "runs" / "tau2"))
    parser.add_argument("--owner-model", default="openai/Qwen/Qwen3.5-27B-BF16INT4")
    parser.add_argument("--owner-url", default="http://localhost:8234/v1")
    parser.add_argument("--s1-model", default=None)
    parser.add_argument("--s1-url", default=None)
    parser.add_argument("--peer-model", default=None)
    parser.add_argument("--peer-url", default=None)
    parser.add_argument("--macro-library", default=None)
    parser.add_argument("--macro-min-lcb", type=float, default=0.0)
    parser.add_argument("--macro-max-skip", type=int, default=4)
    parser.add_argument(
        "--macro-min-skip",
        type=int,
        default=2,
        help="Minimum macro tail depth to commit. Default 2 forbids depth=1 fires.",
    )
    parser.add_argument("--macro-softgate", action="store_true")
    parser.add_argument(
        "--macro-runtime-no-fire",
        action="store_true",
        help="Evaluate macro candidates without committing their suffixes.",
    )
    parser.add_argument(
        "--macro-stage-audit",
        action="store_true",
        help="Record decision-neutral macro filter-stage candidates.",
    )
    parser.add_argument(
        "--chain-max-depth",
        type=int,
        default=None,
        help=(
            "Maximum background S1 chain depth. Defaults to 1 for spec_verify "
            "and 5 for macro_skip."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--frontload-splits",
        nargs="*",
        default=None,
        help=(
            "Run these telecom splits first, in order, then run the remaining "
            "tasks from --split in their original order."
        ),
    )
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args()


def result_path(results_dir: Path, index: int, task_id: str) -> Path:
    digest = hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:12]
    intent = task_id.split("]", 1)[0].lstrip("[")
    intent = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in intent)
    return results_dir / f"{index:04d}_{digest}_{intent[:80]}.json"


def frontload_tasks(tasks, split_names: list[str] | None):
    if not split_names:
        return tasks
    task_by_id = {task.id: task for task in tasks}
    ordered = []
    seen: set[str] = set()
    for split_name in split_names:
        for task in get_telecom_tasks(split_name):
            if task.id in task_by_id and task.id not in seen:
                ordered.append(task_by_id[task.id])
                seen.add(task.id)
    ordered.extend(task for task in tasks if task.id not in seen)
    return ordered


def load_task_ids_file(path: str | None) -> list[str] | None:
    if not path:
        return None
    payload = Path(path).read_text(encoding="utf-8").strip()
    if not payload:
        return []
    if payload[0] in "[{":
        data = json.loads(payload)
        if isinstance(data, dict):
            data = data.get("task_ids") or data.get("tasks") or []
        return [str(item) for item in data]
    return [line.strip() for line in payload.splitlines() if line.strip()]


def load_completed_results(results_dir: Path) -> dict[str, SimulationRun]:
    completed: dict[str, SimulationRun] = {}
    if not results_dir.exists():
        return completed
    for path in sorted(results_dir.glob("*.json")):
        try:
            result = SimulationRun.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"warning: failed to load completed result {path}: {exc}", file=sys.stderr)
            continue
        completed.setdefault(result.task_id, result)
    return completed


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    run_id = args.run_id or time.strftime("tau2_%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / run_id
    results_dir = run_dir / "results"
    tasks = select_tasks(args.split, limit=args.limit, offset=args.offset)
    tasks = frontload_tasks(tasks, args.frontload_splits)
    explicit_task_ids = load_task_ids_file(args.task_ids_file)
    if explicit_task_ids is not None:
        task_by_id = {task.id: task for task in select_tasks("full")}
        tasks = [task_by_id[task_id] for task_id in explicit_task_ids]
    owner = LLMEndpoint(
        model=args.owner_model,
        api_base=args.owner_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        enable_thinking=args.thinking,
    )
    s1 = None
    if args.mode in ("spec_verify", "macro_skip"):
        s1 = LLMEndpoint(
            model=args.s1_model or args.owner_model,
            api_base=args.s1_url or args.owner_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            enable_thinking=args.thinking,
        )
    peer = None
    if args.mode in ("spec_verify", "macro_skip") and args.peer_url:
        peer = LLMEndpoint(
            model=args.peer_model or args.owner_model,
            api_base=args.peer_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            enable_thinking=args.thinking,
        )
    config = {
        "split": args.split,
        "task_ids_file": args.task_ids_file,
        "limit": args.limit,
        "offset": args.offset,
        "mode": args.mode,
        "frontload_splits": args.frontload_splits or [],
        "max_steps": args.max_steps,
        "max_errors": args.max_errors,
        "policy_type": args.policy_type,
        "evaluation_type": args.evaluation_type,
        "owner": owner.__dict__,
        "s1": None if s1 is None else s1.__dict__,
        "peer": None if peer is None else peer.__dict__,
        "macro_library": args.macro_library,
        "macro_min_lcb": args.macro_min_lcb,
        "macro_max_skip": args.macro_max_skip,
        "macro_min_skip": args.macro_min_skip,
        "macro_softgate": args.macro_softgate,
        "macro_runtime_no_fire": args.macro_runtime_no_fire,
        "macro_stage_audit": args.macro_stage_audit,
        "chain_max_depth": (
            args.chain_max_depth
            if args.chain_max_depth is not None
            else (5 if args.mode == "macro_skip" else 1)
        ),
        "task_ids": [t.id for t in tasks],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results = []
    completed_by_task = load_completed_results(results_dir) if args.resume else {}
    for index, task in enumerate(tasks):
        out_path = result_path(results_dir, index + args.offset, task.id)
        if args.resume and out_path.exists():
            result = SimulationRun.model_validate_json(
                out_path.read_text(encoding="utf-8")
            )
            results.append(result)
            reward = result.reward_info.reward if result.reward_info else None
            print(
                f"[{index + 1}/{len(tasks)}] {task.id} -> resume "
                f"reward={reward} termination={result.termination_reason}",
                flush=True,
            )
            write_summary(run_dir / "summary_partial.json", config, results)
            continue
        if args.resume and task.id in completed_by_task:
            result = completed_by_task[task.id]
            results.append(result)
            reward = result.reward_info.reward if result.reward_info else None
            print(
                f"[{index + 1}/{len(tasks)}] {task.id} -> resume-by-task "
                f"reward={reward} termination={result.termination_reason}",
                flush=True,
            )
            write_summary(run_dir / "summary_partial.json", config, results)
            continue
        print(f"[{index + 1}/{len(tasks)}] {task.id}", flush=True)
        result = run_tau2_telecom_task(
            task=task,
            owner=owner,
            s1=s1,
            peer=peer,
            mode=args.mode,
            max_steps=args.max_steps,
            max_errors=args.max_errors,
            policy_type=args.policy_type,
            evaluation_type=EvaluationType(args.evaluation_type),
            macro_library=args.macro_library,
            macro_min_lcb=args.macro_min_lcb,
            macro_max_skip=args.macro_max_skip,
            macro_min_skip=args.macro_min_skip,
            macro_softgate=args.macro_softgate,
            chain_max_depth=config["chain_max_depth"],
            macro_runtime_no_fire=args.macro_runtime_no_fire,
            macro_stage_audit=args.macro_stage_audit,
        )
        results.append(result)
        reward = result.reward_info.reward if result.reward_info else None
        print(
            "  ->",
            {
                "reward": reward,
                "termination": str(result.termination_reason),
                "duration": round(float(result.duration or 0.0), 2),
                "owner_calls": (result.info or {}).get("owner_calls"),
                "exact_matches": (result.info or {}).get("exact_matches"),
                "spec_commits": (result.info or {}).get("spec_commits"),
                "macro_hits": (result.info or {}).get("macro_hits"),
                "macro_steps_skipped": (result.info or {}).get("macro_steps_skipped"),
                "macro_depth1_suppressed": (result.info or {}).get(
                    "macro_depth1_suppressed"
                ),
                "macro_runtime_suppressed": (result.info or {}).get(
                    "macro_runtime_suppressed"
                ),
                "macro_steps_would_skip": (result.info or {}).get(
                    "macro_steps_would_skip"
                ),
                "macro_stage_audit_events": len(
                    (result.info or {}).get("macro_stage_audit_events") or []
                ),
            },
            flush=True,
        )
        write_result(out_path, result)
        write_summary(run_dir / "summary_partial.json", config, results)

    write_summary(run_dir / "summary.json", config, results)
    print(f"summary={run_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
