#!/usr/bin/env python3
"""Summarize and compare TAU2 macro-skip experiment arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_summary(run_dir: Path, arm: str) -> dict[str, Any]:
    path = run_dir / arm / "summary.json"
    if not path.exists():
        path = run_dir / arm / "summary_partial.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"arm": arm, "path": str(path), **payload["summary"]}


def choose_best(rows: list[dict[str, Any]], baseline_arm: str = "baseline") -> dict[str, Any] | None:
    baseline = next((row for row in rows if row["arm"] == baseline_arm), None)
    candidates = [row for row in rows if row["arm"] != baseline_arm]
    if not candidates:
        return None
    if baseline:
        base_reward = float(baseline.get("reward_sum") or 0.0)
        non_regress = [row for row in candidates if float(row.get("reward_sum") or 0.0) >= base_reward]
        if non_regress:
            candidates = non_regress
    candidates.sort(
        key=lambda row: (
            float(row.get("reward_sum") or 0.0),
            -float(row.get("avg_duration_s") or 0.0),
            int(row.get("macro_steps_skipped") or 0),
        ),
        reverse=True,
    )
    return candidates[0]


def paired_delta(rows: list[dict[str, Any]], a: str, b: str) -> dict[str, Any] | None:
    left = next((row for row in rows if row["arm"] == a), None)
    right = next((row for row in rows if row["arm"] == b), None)
    if not left or not right:
        return None
    return {
        "left": a,
        "right": b,
        "n_left": left.get("num_tasks"),
        "n_right": right.get("num_tasks"),
        "reward_delta": round(float(left.get("reward_sum") or 0.0) - float(right.get("reward_sum") or 0.0), 4),
        "avg_duration_delta_s": round(float(left.get("avg_duration_s") or 0.0) - float(right.get("avg_duration_s") or 0.0), 4),
        "avg_duration_delta_pct": round(
            100.0
            * (
                float(left.get("avg_duration_s") or 0.0)
                - float(right.get("avg_duration_s") or 0.0)
            )
            / max(float(right.get("avg_duration_s") or 0.0), 1e-9),
            2,
        ),
        "macro_steps_skipped_delta": int(left.get("macro_steps_skipped") or 0) - int(right.get("macro_steps_skipped") or 0),
        "spec_commits_delta": int(left.get("spec_commits") or 0) - int(right.get("spec_commits") or 0),
        "peer_transfers_delta": int(left.get("peer_transfers") or 0) - int(right.get("peer_transfers") or 0),
    }


def write_report(path: Path, title: str, rows: list[dict[str, Any]], best: dict[str, Any] | None, deltas: list[dict[str, Any]]) -> None:
    lines = [f"# {title}", ""]
    lines.append("| Arm | n | reward | acc | avg_s | terminations | exact | spec | peer submits | peer transfers | macro hits | macro skipped | rejects | d1 suppressed | chain max |")
    lines.append("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {arm} | {num_tasks} | {reward_sum:.3f} | {accuracy:.3f} | {avg_duration_s:.2f} | {terminations} | {exact_matches} | {spec_commits} | {peer_submits} | {peer_transfers} | {macro_hits} | {macro_steps_skipped} | {macro_rejects} | {macro_depth1_suppressed} | {s1_chain_max_depth_observed} |".format(
                **{
                    **row,
                    "reward_sum": float(row.get("reward_sum") or 0.0),
                    "accuracy": float(row.get("accuracy") or 0.0),
                    "avg_duration_s": float(row.get("avg_duration_s") or 0.0),
                    "terminations": json.dumps(row.get("terminations") or {}, sort_keys=True),
                    "exact_matches": int(row.get("exact_matches") or 0),
                    "spec_commits": int(row.get("spec_commits") or 0),
                    "peer_submits": int(row.get("peer_submits") or 0),
                    "peer_transfers": int(row.get("peer_transfers") or 0),
                    "macro_hits": int(row.get("macro_hits") or 0),
                    "macro_steps_skipped": int(row.get("macro_steps_skipped") or 0),
                    "macro_rejects": int(row.get("macro_rejects") or 0),
                    "macro_depth1_suppressed": int(
                        row.get("macro_depth1_suppressed") or 0
                    ),
                    "s1_chain_max_depth_observed": int(
                        row.get("s1_chain_max_depth_observed") or 0
                    ),
                }
            )
        )
    if best:
        lines.extend(["", f"Best test config: `{best['arm']}`"])
    if deltas:
        lines.extend(["", "## Deltas", ""])
        lines.append("| Left - Right | reward | avg_s | avg_% | macro skipped | spec commits | peer transfers |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for delta in deltas:
            lines.append(
                f"| {delta['left']} - {delta['right']} | {delta['reward_delta']} | {delta['avg_duration_delta_s']} | {delta['avg_duration_delta_pct']}% | {delta['macro_steps_skipped_delta']} | {delta['spec_commits_delta']} | {delta['peer_transfers_delta']} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--title", default="TAU2 Macro Experiment Summary")
    parser.add_argument("--baseline-arm", default="baseline")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [load_summary(args.run_dir, arm) for arm in args.arms]
    best = choose_best(rows, args.baseline_arm)
    deltas = []
    if best and any(row["arm"] == args.baseline_arm for row in rows):
        delta = paired_delta(rows, best["arm"], args.baseline_arm)
        if delta:
            deltas.append(delta)
    payload = {"rows": rows, "best": best, "deltas": deltas}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(args.out_md, args.title, rows, best, deltas)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
