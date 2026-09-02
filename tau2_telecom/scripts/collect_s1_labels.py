#!/usr/bin/env python3
"""Collect top-5 S1 accept/reject labels from TAU2 golden trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from litellm import completion
from tau2.agent.llm_agent import LLMSoloAgent
from tau2.data_model.message import AssistantMessage, SystemMessage, ToolCall, UserMessage
from tau2.data_model.simulation import SimulationRun
from tau2.domains.telecom.environment import get_environment
from tau2.utils.llm_utils import to_litellm_messages

from tau2_telecom.src.runner import LLMEndpoint, _initial_state_parts, select_tasks


KICKOFF = "Begin solving the ticket now. Use exactly one tool call batch for the next step."


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(stable_json(obj).encode("utf-8")).hexdigest()[:16]


def action_key(message: AssistantMessage | None) -> str:
    if message is None or not message.tool_calls:
        return "_no_tool_call"
    return "+".join(call.name for call in message.tool_calls)


def exact_equal(a: AssistantMessage | None, b: AssistantMessage | None) -> bool:
    if a is None or b is None or not a.tool_calls or not b.tool_calls:
        return False
    if len(a.tool_calls) != len(b.tool_calls):
        return False
    for ca, cb in zip(a.tool_calls, b.tool_calls, strict=True):
        if ca.name != cb.name:
            return False
        if stable_json(ca.arguments or {}) != stable_json(cb.arguments or {}):
            return False
    return True


def reward_of(run: SimulationRun) -> float:
    return float(run.reward_info.reward if run.reward_info else 0.0)


def load_runs(results_dir: Path, task_ids: set[str]) -> list[SimulationRun]:
    runs: dict[str, SimulationRun] = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            run = SimulationRun.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if run.task_id in task_ids:
            runs.setdefault(run.task_id, run)
    return [runs[task_id] for task_id in sorted(runs)]


def split_task_ids(names: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        for task in select_tasks(name):
            if task.id not in seen:
                seen.add(task.id)
                ordered.append(task.id)
    return ordered


def build_prompt_parts(task, endpoint: LLMEndpoint):
    env = get_environment(solo_mode=True, policy_type="manual")
    initialization_data, initialization_actions, initial_history = _initial_state_parts(task)
    env.set_state(
        initialization_data=initialization_data,
        initialization_actions=initialization_actions,
        message_history=initial_history,
    )
    tools = env.get_tools() + env.get_user_tools()
    prompt_agent = LLMSoloAgent(
        tools=tools,
        domain_policy=env.get_policy(),
        task=task,
        llm=endpoint.model,
        llm_args=endpoint.args(),
    )
    return prompt_agent.system_prompt, prompt_agent.tools


def parse_choice(choice: Any) -> AssistantMessage:
    message = choice.message
    calls = []
    for raw in message.tool_calls or []:
        try:
            args = json.loads(raw.function.arguments)
        except Exception:
            args = {}
        calls.append(ToolCall(id=raw.id, name=raw.function.name, arguments=args))
    return AssistantMessage(
        role="assistant",
        content=message.content,
        tool_calls=calls or None,
        raw_data=message.model_dump() if hasattr(message, "model_dump") else None,
    )


def sample_top5(
    *,
    endpoint: LLMEndpoint,
    system_prompt: str,
    tools: list[Any],
    history: list[Any],
    num_samples: int,
    temperature: float,
    top_p: float,
) -> tuple[list[AssistantMessage], dict[str, Any], float, str]:
    t0 = time.perf_counter()
    try:
        response = completion(
            model=endpoint.model,
            messages=to_litellm_messages(
                [SystemMessage(role="system", content=system_prompt)] + history
            ),
            tools=[tool.openai_schema for tool in tools],
            tool_choice="required",
            n=num_samples,
            temperature=temperature,
            top_p=top_p,
            max_tokens=endpoint.max_tokens,
            api_base=endpoint.api_base,
            api_key=endpoint.api_key,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": endpoint.enable_thinking,
                }
            },
        )
        usage = response.usage.model_dump() if getattr(response, "usage", None) else {}
        return [parse_choice(choice) for choice in response.choices], usage, time.perf_counter() - t0, ""
    except Exception as exc:
        return [], {}, time.perf_counter() - t0, repr(exc)


def assistant_indices(messages: list[Any]) -> list[int]:
    return [
        idx
        for idx, message in enumerate(messages)
        if isinstance(message, AssistantMessage) and message.tool_calls
    ]


def collect(args: argparse.Namespace) -> dict[str, Any]:
    task_ids = split_task_ids(args.mine_splits)
    task_id_set = set(task_ids)
    task_by_id = {task.id: task for task in select_tasks("full") if task.id in task_id_set}
    runs = load_runs(args.results_dir, task_id_set)
    runs = [run for run in runs if reward_of(run) == 1.0]
    runs.sort(key=lambda run: task_ids.index(run.task_id))
    runs = [run for i, run in enumerate(runs) if i % args.num_shards == args.shard_index]
    if args.max_tasks is not None:
        runs = runs[: args.max_tasks]

    endpoint = LLMEndpoint(
        model=args.s1_model,
        api_base=args.s1_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        enable_thinking=args.thinking,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {
        "records": 0,
        "skeleton_accept": 0,
        "skeleton_reject": 0,
        "exact_accept": 0,
        "exact_reject": 0,
        "llm_errors": 0,
    }
    t_start = time.time()
    with args.output.open("w", encoding="utf-8") as f:
        for run in runs:
            task = task_by_id[run.task_id]
            system_prompt, tools = build_prompt_parts(task, endpoint)
            messages = list(run.messages or [])
            action_positions = assistant_indices(messages)
            prefixes = action_positions[:: max(1, args.prefix_stride)]
            if args.max_prefixes_per_task is not None:
                prefixes = prefixes[: args.max_prefixes_per_task]
            print(
                f"task={run.task_id} actions={len(action_positions)} prefixes={len(prefixes)}",
                flush=True,
            )
            for prefix_rank, msg_index in enumerate(prefixes):
                prefix_action_idx = action_positions.index(msg_index)
                history = [UserMessage.text(KICKOFF)] + messages[:msg_index]
                accepted_chain: list[str] = []
                for depth in range(1, args.max_chain_depth + 1):
                    gt_action_idx = prefix_action_idx + depth - 1
                    if gt_action_idx >= len(action_positions):
                        break
                    gt_message = messages[action_positions[gt_action_idx]]
                    preds, usage, wall_s, llm_error = sample_top5(
                        endpoint=endpoint,
                        system_prompt=system_prompt,
                        tools=tools,
                        history=history,
                        num_samples=args.num_samples,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                    if llm_error:
                        counts["llm_errors"] += 1
                    if not preds:
                        preds = [AssistantMessage(role="assistant", content="", tool_calls=None)]
                    any_exact = False
                    any_skeleton = False
                    seen_choices: set[tuple[str, str]] = set()
                    for choice_index, pred in enumerate(preds):
                        pred_key = action_key(pred)
                        gt_key = action_key(gt_message)
                        skeleton_accept = pred_key == gt_key
                        exact_accept = exact_equal(pred, gt_message)
                        any_skeleton = any_skeleton or skeleton_accept
                        any_exact = any_exact or exact_accept
                        choice_key = (
                            pred_key,
                            stable_hash(
                                [call.arguments for call in pred.tool_calls or []]
                            ),
                        )
                        duplicate = choice_key in seen_choices
                        seen_choices.add(choice_key)
                        row = {
                            "task_id": run.task_id,
                            "source_run": str(args.results_dir),
                            "source_reward": reward_of(run),
                            "prefix_rank": prefix_rank,
                            "prefix_action_idx": prefix_action_idx,
                            "depth": depth,
                            "choice_index": choice_index,
                            "num_samples": args.num_samples,
                            "duplicate_in_request": duplicate,
                            "committed_prefix_skeleton": [
                                action_key(messages[pos])
                                for pos in action_positions[:prefix_action_idx]
                            ],
                            "s1_chain_skeleton": accepted_chain + [pred_key],
                            "gold_chain_skeleton": [
                                action_key(messages[pos])
                                for pos in action_positions[
                                    prefix_action_idx : gt_action_idx + 1
                                ]
                            ],
                            "gt_skeleton": gt_key,
                            "pred_skeleton": pred_key,
                            "gt_args_hash": stable_hash(
                                [call.arguments for call in gt_message.tool_calls or []]
                            ),
                            "pred_args_hash": stable_hash(
                                [call.arguments for call in pred.tool_calls or []]
                            ),
                            "label_skeleton": "Accept" if skeleton_accept else "Reject",
                            "label_exact": "Accept" if exact_accept else "Reject",
                            "depth_has_skeleton_accept": any_skeleton,
                            "depth_has_exact_accept": any_exact,
                            "s1_wall_s": round(wall_s, 4),
                            "s1_usage": usage,
                            "llm_error": llm_error,
                        }
                        f.write(json.dumps(row, sort_keys=True) + "\n")
                        counts["records"] += 1
                        counts[
                            "skeleton_accept" if skeleton_accept else "skeleton_reject"
                        ] += 1
                        counts["exact_accept" if exact_accept else "exact_reject"] += 1
                    f.flush()
                    if not any_exact:
                        break
                    accepted_chain.append(action_key(gt_message))
                    next_tool_pos = action_positions[gt_action_idx] + 2
                    history = [UserMessage.text(KICKOFF)] + messages[:next_tool_pos]

    summary = {
        "output": str(args.output),
        "results_dir": str(args.results_dir),
        "mine_splits": args.mine_splits,
        "n_runs": len(runs),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "settings": {
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_chain_depth": args.max_chain_depth,
            "prefix_stride": args.prefix_stride,
            "max_prefixes_per_task": args.max_prefixes_per_task,
        },
        "counts": counts,
        "elapsed_s": round(time.time() - t_start, 3),
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mine-splits", nargs="+", default=["small", "train"])
    parser.add_argument("--s1-url", required=True)
    parser.add_argument("--s1-model", default="openai/Qwen/Qwen3.5-4B")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-chain-depth", type=int, default=4)
    parser.add_argument("--prefix-stride", type=int, default=1)
    parser.add_argument("--max-prefixes-per-task", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    collect(parse_args())
