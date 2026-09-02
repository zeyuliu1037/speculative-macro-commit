#!/usr/bin/env python3
"""Run or plan the camera-ready AppWorld SA/SMC paired task order.

The command is dry-run by default. Pass ``--execute`` only after starting one
fresh AppWorld model-server epoch and checking that all endpoints are ready.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from argparse import Namespace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smc_appworld.result_summary import build_run_summary  # noqa: E402
from smc_appworld.run import SAFE_TASK_ID, run_one  # noqa: E402


DEFAULT_TASKS = REPO_ROOT / "configs" / "appworld" / "task_ids_test_normal_168.json"
DEFAULT_SA = REPO_ROOT / "configs" / "appworld" / "sa.json"
DEFAULT_SMC = REPO_ROOT / "configs" / "appworld" / "smc.json"


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_task_ids(path: Path) -> list[str]:
    payload = read_json(path)
    values = payload.get("task_ids") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise ValueError("task manifest must be a JSON list or contain task_ids")
    if len(values) != len(set(values)):
        raise ValueError("task manifest contains duplicate task IDs")
    invalid = [task_id for task_id in values if not SAFE_TASK_ID.fullmatch(task_id)]
    if invalid:
        raise ValueError(f"unsafe AppWorld task ID: {invalid[0]!r}")
    return values


def build_plan(task_ids: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "task_id": task_id,
            "arm_order": ["sa", "smc"] if index % 2 else ["smc", "sa"],
        }
        for index, task_id in enumerate(task_ids, start=1)
    ]


def arm_namespace(config_path: Path, appworld_root: str) -> Namespace:
    config = read_json(config_path)
    mode = config.get("mode")
    if mode not in {"sa", "smc"}:
        raise ValueError(f"paired arm must be sa or smc: {config_path}")
    if config.get("save_raw_trajectories", False):
        raise ValueError("paired release runner refuses raw trajectory output")
    return Namespace(
        mode=mode,
        save_raw_trajectories=False,
        actor_model=config.get("actor_model", "Qwen/Qwen3.5-27B"),
        speculator_model=config.get("speculator_model", "Qwen/Qwen3.5-4B"),
        actor_url=config.get("actor_url", "http://localhost:8004/v1"),
        peer_url=config.get("peer_url", "http://localhost:8005/v1"),
        speculator_url=config.get("speculator_url", "http://localhost:8003/v1"),
        max_steps=int(config.get("max_steps", 80)),
        thinking_budget=int(config.get("thinking_budget", 4096)),
        owner_max_tokens=int(config.get("owner_max_tokens", 8192)),
        chain_max_depth=int(config.get("chain_max_depth", 4)),
        verify_mode=config.get("verify_mode", "exact"),
        meta_tool_library=config.get("meta_tool_library"),
        appworld_root=appworld_root,
    )


def compact_summary(
    results_dir: Path, mode: str, task_ids: list[str] | None = None
) -> dict[str, Any]:
    if task_ids is None:
        rows = [read_json(path) for path in sorted(results_dir.glob("*.json"))]
    else:
        rows = [
            read_json(results_dir / f"{task_id}.json")
            for task_id in task_ids
            if (results_dir / f"{task_id}.json").is_file()
        ]
    return build_run_summary(
        mode,
        len(task_ids) if task_ids is not None else len(rows),
        rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-ids-file", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--sa-config", type=Path, default=DEFAULT_SA)
    parser.add_argument("--smc-config", type=Path, default=DEFAULT_SMC)
    parser.add_argument("--appworld-root")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "appworld_paired")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_ids = load_task_ids(args.task_ids_file)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        task_ids = task_ids[: args.limit]
    plan = build_plan(task_ids)
    sa_config = read_json(args.sa_config)
    smc_config = read_json(args.smc_config)
    identity = {
        "schema_version": 1,
        "protocol": "appworld_paired_sa_smc",
        "task_manifest": str(args.task_ids_file),
        "ordered_task_sha256": canonical_hash(task_ids),
        "task_count": len(task_ids),
        "sa_config_sha256": canonical_hash(sa_config),
        "smc_config_sha256": canonical_hash(smc_config),
        "plan": plan,
    }

    if not args.execute:
        print(json.dumps(identity, indent=2))
        return 0
    if not args.appworld_root:
        raise ValueError("--appworld-root is required with --execute")

    from appworld import update_root

    update_root(args.appworld_root)
    run_id = args.run_id or time.strftime("appworld_pair_%Y%m%d_%H%M%S")
    campaign_dir = args.out_dir / run_id
    campaign_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = campaign_dir / "campaign.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if canonical_hash(existing) != canonical_hash(identity):
            raise ValueError("existing campaign identity differs; choose a new --run-id")
        if not args.resume:
            raise ValueError("campaign already exists; pass --resume or choose a new --run-id")
    else:
        manifest_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")

    namespaces = {
        "sa": arm_namespace(args.sa_config, args.appworld_root),
        "smc": arm_namespace(args.smc_config, args.appworld_root),
    }
    source_configs = {"sa": sa_config, "smc": smc_config}
    try:
        task_manifest_value = str(args.task_ids_file.resolve().relative_to(REPO_ROOT))
    except ValueError:
        task_manifest_value = str(args.task_ids_file.resolve())
    for arm in namespaces:
        arm_dir = campaign_dir / arm
        (arm_dir / "results").mkdir(parents=True, exist_ok=True)
        run_config = dict(source_configs[arm])
        run_config.update(
            {
                "task_count": len(task_ids),
                "task_ids_file": task_manifest_value,
                "appworld_root": args.appworld_root,
                "paired_campaign": str(campaign_dir),
            }
        )
        (arm_dir / "config.json").write_text(
            json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
        )

    for item in plan:
        task_id = item["task_id"]
        print(f"[{item['index']}/{len(plan)}] {task_id}: {' -> '.join(item['arm_order'])}", flush=True)
        for arm in item["arm_order"]:
            arm_dir = campaign_dir / arm
            result_path = arm_dir / "results" / f"{task_id}.json"
            if args.resume and result_path.exists():
                print(f"  {arm}: resume", flush=True)
                continue
            try:
                result = run_one(namespaces[arm], task_id, arm_dir)
            except Exception as exc:
                traceback.print_exc()
                result = {
                    "task_id": task_id,
                    "mode": arm,
                    "error_type": type(exc).__name__,
                }
            result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            (arm_dir / "summary.json").write_text(
                json.dumps(
                    compact_summary(arm_dir / "results", arm, task_ids), indent=2
                )
                + "\n",
                encoding="utf-8",
            )

    failed = 0
    for arm in namespaces:
        summary = compact_summary(campaign_dir / arm / "results", arm, task_ids)
        (campaign_dir / arm / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        failed += int(summary["errors"])
    print(f"campaign={campaign_dir}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
