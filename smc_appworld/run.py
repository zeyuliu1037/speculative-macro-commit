#!/usr/bin/env python3
"""Run the camera-ready AppWorld baseline, SA, or SMC configuration.

Compact result files deliberately exclude prompts, messages, API arguments,
and tool responses. Raw trajectories are opt-in because AppWorld tasks can
contain simulated credentials and other benchmark-sensitive values.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smc_appworld.result_summary import build_run_summary, run_exit_code  # noqa: E402


DEFAULT_LIBRARY = REPO_ROOT / "libraries" / "appworld_train50.json"
SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a JSON object: {config_path}")
    payload["_config_dir"] = str(config_path.parent)
    return payload


def resolve_release_path(value: str | None) -> str | None:
    if not value:
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config")
    known, _ = pre_parser.parse_known_args()
    config = load_config(known.config)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=known.config)
    parser.add_argument(
        "--mode",
        choices=["baseline", "sa", "smc"],
        default=config.get("mode", "smc"),
    )
    parser.add_argument("--split", default=config.get("split", "test_normal"))
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--task-ids-file", default=config.get("task_ids_file"))
    parser.add_argument("--limit", type=int, default=config.get("limit"))
    parser.add_argument("--offset", type=int, default=config.get("offset", 0))
    parser.add_argument("--run-id", default=config.get("run_id"))
    parser.add_argument("--out-dir", default=config.get("out_dir", "runs/appworld"))
    parser.add_argument("--appworld-root", default=config.get("appworld_root"))
    parser.add_argument("--actor-model", default=config.get("actor_model", "Qwen/Qwen3.5-27B"))
    parser.add_argument("--speculator-model", default=config.get("speculator_model", "Qwen/Qwen3.5-4B"))
    parser.add_argument("--actor-url", default=config.get("actor_url", "http://localhost:8004/v1"))
    parser.add_argument("--peer-url", default=config.get("peer_url", "http://localhost:8005/v1"))
    parser.add_argument("--speculator-url", default=config.get("speculator_url", "http://localhost:8003/v1"))
    parser.add_argument("--max-steps", type=int, default=config.get("max_steps", 80))
    parser.add_argument("--thinking-budget", type=int, default=config.get("thinking_budget", 4096))
    parser.add_argument("--owner-max-tokens", type=int, default=config.get("owner_max_tokens", 8192))
    parser.add_argument("--chain-max-depth", type=int, default=config.get("chain_max_depth", 4))
    parser.add_argument("--verify-mode", choices=["exact", "exec_api_semantic", "semantic"], default=config.get("verify_mode", "exact"))
    parser.add_argument("--meta-tool-library", default=config.get("meta_tool_library", str(DEFAULT_LIBRARY)))
    parser.add_argument("--save-raw-trajectories", action=argparse.BooleanOptionalAction, default=config.get("save_raw_trajectories", False))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=config.get("resume", True))
    return parser.parse_args()


def load_requested_task_ids(args: argparse.Namespace) -> list[str]:
    if args.task_id:
        task_ids = list(args.task_id)
    elif args.task_ids_file:
        task_ids_path = Path(args.task_ids_file).expanduser()
        if not task_ids_path.is_absolute():
            task_ids_path = REPO_ROOT / task_ids_path
        text = task_ids_path.read_text(encoding="utf-8").strip()
        if text.startswith("[") or text.startswith("{"):
            payload = json.loads(text)
            if isinstance(payload, dict):
                payload = payload.get("task_ids", [])
            task_ids = [str(item) for item in payload]
        else:
            task_ids = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        from appworld import load_task_ids

        task_ids = list(load_task_ids(args.split))
    task_ids = task_ids[args.offset :]
    if args.limit is not None:
        task_ids = task_ids[: args.limit]
    invalid = [task_id for task_id in task_ids if not SAFE_TASK_ID.fullmatch(task_id)]
    if invalid:
        raise ValueError(f"unsafe AppWorld task ID: {invalid[0]!r}")
    return task_ids


def compact_result(mode: str, task_id: str, result: dict[str, Any]) -> dict[str, Any]:
    common = ["reward", "n_steps", "task_completed"]
    baseline = ["total_input_tokens", "total_output_tokens", "avg_step_time", "total_wall_time"]
    speculative = [
        "env_steps", "wall_time", "wall_speedup", "verify_match", "verify_mismatch",
        "ownership_transferred", "peer_dropped_invalid", "peer_dropped_stale",
        "no_chain_count", "no_tool_call_count", "transfer_mut", "meta_tool_hits",
        "meta_tool_steps_skipped", "hard_boundary_mut", "hard_boundary_preexec",
        "ntc_cap_triggered", "verify_cap_triggered", "chain_pre_exec_total",
        "chain_pre_exec_wasted", "chain_fork_exec_total", "chain_fork_exec_wasted",
        "chain_avg_depth", "chain_depth_max", "s1_total_time", "s1_call_count",
        "verify_mode", "chain_max_depth", "env_fork",
    ]
    fields = common + (baseline if mode == "baseline" else speculative)
    compact = {"task_id": task_id, "mode": mode}
    compact.update({key: result[key] for key in fields if key in result})
    return compact


def run_one(args: argparse.Namespace, task_id: str, run_dir: Path) -> dict[str, Any]:
    from appworld import AppWorld
    from smc_appworld.runtime.agent import run_agent
    from smc_appworld.runtime.meta_tool import MetaToolLibrary
    from smc_appworld.runtime.pipeline_agent import run_pipeline_agent
    from smc_appworld.runtime.trajectory_logger import TrajectoryLogger

    env = AppWorld(
        task_id=task_id,
        experiment_name=f"smc_{run_dir.name}_{args.mode}",
        raise_on_failure=False,
        max_interactions=args.max_steps * 3,
    )
    logger = None
    if args.save_raw_trajectories:
        logger = TrajectoryLogger(str(run_dir / "raw_trajectories"), task_id)
    try:
        if args.mode == "baseline":
            result = run_agent(
                env=env,
                task_instruction=env.task.instruction,
                model=args.actor_model,
                api_base=args.actor_url,
                thinking=True,
                thinking_budget=args.thinking_budget,
                max_steps=args.max_steps,
                temperature=0.0,
                logger=logger,
            )
        else:
            library = None
            if args.mode == "smc":
                library_path = resolve_release_path(args.meta_tool_library)
                if not library_path or not Path(library_path).is_file():
                    raise FileNotFoundError(f"SMC meta-tool library not found: {library_path}")
                library = MetaToolLibrary(path=library_path)
            result = run_pipeline_agent(
                env=env,
                task_instruction=env.task.instruction,
                actor_model=args.actor_model,
                speculator_model=args.speculator_model,
                a1_url=args.actor_url,
                a2_url=args.peer_url,
                s1_url=args.speculator_url,
                thinking_budget=args.thinking_budget,
                max_tokens_owner_override=args.owner_max_tokens,
                max_steps=args.max_steps,
                chain_max_depth=args.chain_max_depth,
                verify_mode=args.verify_mode,
                reset_primer=False,
                env_fork=True,
                meta_tool_library=library,
                logger=logger,
            )
        return compact_result(args.mode, task_id, result)
    finally:
        env.close()


def main() -> int:
    args = parse_args()
    from appworld import update_root

    update_root(args.appworld_root)
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir).expanduser()
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    run_dir = out_root / f"{run_id}_{args.mode}"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    task_ids = load_requested_task_ids(args)
    config = vars(args).copy()
    config["task_count"] = len(task_ids)
    config["meta_tool_library"] = resolve_release_path(args.meta_tool_library)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results: list[dict[str, Any]] = []
    summary = build_run_summary(args.mode, len(task_ids), results)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    for index, task_id in enumerate(task_ids, start=1):
        result_path = results_dir / f"{task_id}.json"
        if args.resume and result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            print(f"[{index}/{len(task_ids)}] {task_id}: resume", flush=True)
        else:
            print(f"[{index}/{len(task_ids)}] {task_id}: {args.mode}", flush=True)
            try:
                result = run_one(args, task_id, run_dir)
            except Exception as exc:
                traceback.print_exc()
                result = {"task_id": task_id, "mode": args.mode, "error_type": type(exc).__name__}
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)
        summary = build_run_summary(args.mode, len(task_ids), results)
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    print(f"summary={run_dir / 'summary.json'}", flush=True)
    return run_exit_code(summary)


if __name__ == "__main__":
    sys.exit(main())
