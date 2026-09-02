"""Minimal TAU2 Telecom runner with owner-only and speculative-verify modes.

This is intentionally separate from tau2-bench.  TAU2 owns the benchmark
environment and evaluator; this package owns speculative-action orchestration.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from tau2.agent.llm_agent import LLMSoloAgent
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.domains.telecom.environment import get_environment, get_tasks
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.utils.llm_utils import generate, get_cost
from tau2.utils.utils import get_now


Mode = Literal["owner_only", "spec_verify", "macro_skip"]


@dataclass
class LLMEndpoint:
    model: str
    api_base: str
    api_key: str = "dummy"
    temperature: float = 0.0
    max_tokens: int = 1024
    enable_thinking: bool = False
    timeout: float | None = None

    def args(self) -> dict[str, Any]:
        args: dict[str, Any] = {
            "api_base": self.api_base,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": self.enable_thinking,
                }
            },
        }
        if self.timeout is not None:
            args["timeout"] = self.timeout
        return args


@dataclass
class StepTrace:
    step: int
    owner_action: list[dict[str, Any]] | None = None
    s1_action: list[dict[str, Any]] | None = None
    peer_action: list[dict[str, Any]] | None = None
    owner_s: float | None = None
    s1_s: float | None = None
    peer_s: float | None = None
    preexec_s: float | None = None
    exact_match: bool = False
    used_spec_result: bool = False
    owner_from_peer: bool = False
    peer_submitted: bool = False
    peer_transferred: bool = False
    peer_cancelled: bool = False
    macro_pattern_id: str | None = None
    macro_skipped: int = 0
    macro_suppressed_depth1: bool = False
    macro_runtime_suppressed: bool = False
    macro_would_skip: int = 0
    owner_error: str | None = None
    s1_error: str | None = None
    peer_error: str | None = None


@dataclass
class RunMetrics:
    mode: Mode
    task_id: str
    max_steps: int
    owner_calls: int = 0
    s1_calls: int = 0
    peer_calls: int = 0
    owner_total_s: float = 0.0
    s1_total_s: float = 0.0
    peer_total_s: float = 0.0
    preexec_total_s: float = 0.0
    exact_matches: int = 0
    spec_commits: int = 0
    peer_submits: int = 0
    peer_transfers: int = 0
    peer_cancels: int = 0
    peer_errors: int = 0
    macro_hits: int = 0
    macro_steps_skipped: int = 0
    macro_rejects: int = 0
    macro_depth1_suppressed: int = 0
    macro_runtime_suppressed: int = 0
    macro_steps_would_skip: int = 0
    macro_patterns: dict[str, int] = field(default_factory=dict)
    macro_stage_audit_events: list[dict[str, Any]] = field(default_factory=list)
    tool_errors: int = 0
    s1_chain_appends: int = 0
    s1_chain_discards: int = 0
    s1_chain_max_depth_observed: int = 0
    step_traces: list[StepTrace] = field(default_factory=list)

    def to_info(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "task_id": self.task_id,
            "max_steps": self.max_steps,
            "owner_calls": self.owner_calls,
            "s1_calls": self.s1_calls,
            "peer_calls": self.peer_calls,
            "owner_total_s": round(self.owner_total_s, 3),
            "s1_total_s": round(self.s1_total_s, 3),
            "peer_total_s": round(self.peer_total_s, 3),
            "preexec_total_s": round(self.preexec_total_s, 3),
            "exact_matches": self.exact_matches,
            "spec_commits": self.spec_commits,
            "peer_submits": self.peer_submits,
            "peer_transfers": self.peer_transfers,
            "peer_cancels": self.peer_cancels,
            "peer_errors": self.peer_errors,
            "macro_hits": self.macro_hits,
            "macro_steps_skipped": self.macro_steps_skipped,
            "macro_rejects": self.macro_rejects,
            "macro_depth1_suppressed": self.macro_depth1_suppressed,
            "macro_runtime_suppressed": self.macro_runtime_suppressed,
            "macro_steps_would_skip": self.macro_steps_would_skip,
            "macro_patterns": self.macro_patterns,
            "macro_stage_audit_events": self.macro_stage_audit_events,
            "tool_errors": self.tool_errors,
            "s1_chain_appends": self.s1_chain_appends,
            "s1_chain_discards": self.s1_chain_discards,
            "s1_chain_max_depth_observed": self.s1_chain_max_depth_observed,
            "step_traces": [
                {
                    "step": t.step,
                    "owner_action": t.owner_action,
                    "s1_action": t.s1_action,
                    "peer_action": t.peer_action,
                    "owner_s": None if t.owner_s is None else round(t.owner_s, 3),
                    "s1_s": None if t.s1_s is None else round(t.s1_s, 3),
                    "peer_s": None if t.peer_s is None else round(t.peer_s, 3),
                    "preexec_s": None if t.preexec_s is None else round(t.preexec_s, 3),
                    "exact_match": t.exact_match,
                    "used_spec_result": t.used_spec_result,
                    "owner_from_peer": t.owner_from_peer,
                    "peer_submitted": t.peer_submitted,
                    "peer_transferred": t.peer_transferred,
                    "peer_cancelled": t.peer_cancelled,
                    "macro_pattern_id": t.macro_pattern_id,
                    "macro_skipped": t.macro_skipped,
                    "macro_suppressed_depth1": t.macro_suppressed_depth1,
                    "macro_runtime_suppressed": t.macro_runtime_suppressed,
                    "macro_would_skip": t.macro_would_skip,
                    "owner_error": t.owner_error,
                    "s1_error": t.s1_error,
                    "peer_error": t.peer_error,
                }
                for t in self.step_traces
            ],
        }


@dataclass
class ChainEntry:
    """One ahead-of-owner S1 prediction plus its forked env result."""

    message: AssistantMessage
    tool_results: list[ToolMessage]
    env_after: Any
    s1_s: float
    preexec_s: float

    @property
    def action_key(self) -> str | None:
        return _action_key(self.message)


def _tool_signature(message: AssistantMessage | None) -> list[dict[str, Any]] | None:
    if message is None or not message.tool_calls:
        return None
    return [
        {
            "name": call.name,
            "arguments": call.arguments,
            "requestor": call.requestor,
        }
        for call in message.tool_calls
    ]


def _action_key(message: AssistantMessage | None) -> str | None:
    signature = _tool_signature(message)
    if not signature:
        return None
    return "+".join(item["name"] for item in signature)


def _same_tool_calls(a: AssistantMessage | None, b: AssistantMessage | None) -> bool:
    return _tool_signature_strict_key(a) == _tool_signature_strict_key(b)


def _strict_value_key(value: Any) -> Any:
    if value is None:
        return ("none", None)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", repr(value))
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, list):
        return ("list", tuple(_strict_value_key(item) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                sorted(
                    (
                        _strict_value_key(key),
                        _strict_value_key(item),
                    )
                    for key, item in value.items()
                )
            ),
        )
    return (type(value).__name__, repr(value))


def _tool_signature_strict_key(message: AssistantMessage | None) -> Any:
    # TAU2 tool responses can depend on JSON numeric type formatting: 2 and
    # 2.0 are equal in Python, but may produce different replay text.
    return _strict_value_key(_tool_signature(message))


def _wrap_tool_results(tool_results: list[ToolMessage]) -> ToolMessage | MultiToolMessage:
    if len(tool_results) > 1:
        return MultiToolMessage(role="tool", tool_messages=tool_results)
    return tool_results[0]


def _flatten_env_message(message: ToolMessage | MultiToolMessage) -> list[ToolMessage]:
    if isinstance(message, MultiToolMessage):
        return list(message.tool_messages)
    return [message]


def _rewrite_tool_result_ids(
    tool_results: Iterable[ToolMessage], owner_calls: Iterable[ToolCall]
) -> list[ToolMessage]:
    rewritten = []
    for result, owner_call in zip(tool_results, owner_calls, strict=True):
        item = deepcopy(result)
        item.id = owner_call.id
        item.requestor = owner_call.requestor
        rewritten.append(item)
    return rewritten


def _generate_action(
    *,
    endpoint: LLMEndpoint,
    system_prompt: str,
    tools: list[Any],
    llm_history: list[Message],
    call_name: str,
    agent: LLMSoloAgent,
) -> tuple[AssistantMessage, float]:
    t0 = time.perf_counter()
    message = generate(
        model=endpoint.model,
        tools=tools,
        messages=[SystemMessage(role="system", content=system_prompt)] + llm_history,
        tool_choice="required",
        call_name=call_name,
        **endpoint.args(),
    )
    dt = time.perf_counter() - t0
    if not isinstance(message, AssistantMessage):
        raise TypeError(f"Expected AssistantMessage, got {type(message)}")
    if message.is_tool_call():
        message = agent._check_if_stop_toolcall(message)
    return message, dt


class S1ChainWorker(threading.Thread):
    """Background S1 chain predictor for TAU2.

    This mirrors the AppWorld pipeline shape: S1 keeps extending a speculative
    chain while the owner/peer lane is busy.  Each entry is generated from the
    committed transcript plus prior chain entries, then pre-executed on a deep
    copy of the current speculative environment.  The main thread can bulk
    commit entries that match a macro tail without synchronously asking S1 for
    each skipped step.
    """

    def __init__(
        self,
        *,
        endpoint: LLMEndpoint,
        system_prompt: str,
        tools: list[Any],
        agent: LLMSoloAgent,
        state: dict[str, Any],
        chain: list[ChainEntry],
        state_lock: threading.Lock,
        max_depth: int,
    ) -> None:
        super().__init__(daemon=True)
        self.endpoint = endpoint
        self.system_prompt = system_prompt
        self.tools = tools
        self.agent = agent
        self.state = state
        self.chain = chain
        self.state_lock = state_lock
        self.max_depth = max(1, max_depth)
        self.stop_flag = False
        self.epoch = 0
        self.s1_call_count = 0
        self.s1_total_s = 0.0
        self.preexec_total_s = 0.0
        self.chain_appends = 0
        self.chain_discards = 0
        self.max_depth_observed = 0

    def publish_state_locked(
        self,
        *,
        env: Any,
        llm_history: list[Message],
        committed_steps: int,
    ) -> None:
        self.state["env"] = env
        self.state["llm_history"] = list(llm_history)
        self.state["committed_steps"] = committed_steps

    def publish_state(
        self,
        *,
        env: Any,
        llm_history: list[Message],
        committed_steps: int,
    ) -> None:
        with self.state_lock:
            self.publish_state_locked(
                env=env,
                llm_history=llm_history,
                committed_steps=committed_steps,
            )

    def signal_rollback_locked(self) -> None:
        self.epoch += 1
        self.chain.clear()

    def signal_rollback(self) -> None:
        with self.state_lock:
            self.signal_rollback_locked()

    def stop(self) -> None:
        self.stop_flag = True

    def run(self) -> None:
        while not self.stop_flag:
            with self.state_lock:
                cur_epoch = self.epoch
                chain_snapshot = list(self.chain)
                base_history = list(self.state.get("llm_history") or [])
                base_env = self.state.get("env")
                committed_steps = int(self.state.get("committed_steps") or 0)

            if (
                base_env is None
                or len(chain_snapshot) >= self.max_depth
                or committed_steps + len(chain_snapshot) >= int(self.state.get("max_steps") or 0)
            ):
                time.sleep(0.01)
                continue

            spec_env = deepcopy(
                chain_snapshot[-1].env_after if chain_snapshot else base_env
            )
            chain_history = list(base_history)
            for entry in chain_snapshot:
                chain_history.append(entry.message)
                chain_history.extend(entry.tool_results)

            try:
                message, s1_dt = _generate_action(
                    endpoint=self.endpoint,
                    system_prompt=self.system_prompt,
                    tools=self.tools,
                    llm_history=chain_history,
                    call_name="tau2_s1_chain_action",
                    agent=self.agent,
                )
            except Exception:
                time.sleep(0.05)
                continue

            self.s1_call_count += 1
            self.s1_total_s += s1_dt
            if self.agent.is_stop(message) or not message.tool_calls:
                time.sleep(0.02)
                continue

            t_pre = time.perf_counter()
            try:
                tool_results = [spec_env.get_response(call) for call in message.tool_calls]
            except Exception:
                time.sleep(0.05)
                continue
            preexec_s = time.perf_counter() - t_pre
            self.preexec_total_s += preexec_s

            entry = ChainEntry(
                message=message,
                tool_results=tool_results,
                env_after=spec_env,
                s1_s=s1_dt,
                preexec_s=preexec_s,
            )
            with self.state_lock:
                # Epoch bumps invalidate the whole speculative suffix.  Front
                # pops are allowed: they mean the main thread committed the
                # prefix this prediction was conditioned on.
                if self.epoch != cur_epoch or len(self.chain) > len(chain_snapshot):
                    self.chain_discards += 1
                    continue
                self.chain.append(entry)
                self.chain_appends += 1
                if len(self.chain) > self.max_depth_observed:
                    self.max_depth_observed = len(self.chain)


def _normalize_pattern_actions(pattern: dict[str, Any]) -> list[str]:
    actions = pattern.get("action_types") or pattern.get("sequence") or []
    normalized: list[str] = []
    for item in actions:
        if isinstance(item, str):
            normalized.append(item)
        elif isinstance(item, dict):
            name = item.get("name") or item.get("tool") or item.get("action")
            if name:
                normalized.append(str(name))
        elif isinstance(item, (list, tuple)):
            names = []
            for sub in item:
                if isinstance(sub, str):
                    names.append(sub)
                elif isinstance(sub, dict):
                    name = sub.get("name") or sub.get("tool") or sub.get("action")
                    if name:
                        names.append(str(name))
            if names:
                normalized.append("+".join(names))
    return normalized


def load_macro_library(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    patterns = payload.get("patterns", payload) if isinstance(payload, dict) else payload
    library: list[dict[str, Any]] = []
    for idx, pattern in enumerate(patterns or []):
        actions = _normalize_pattern_actions(pattern)
        if len(actions) < 2:
            continue
        item = dict(pattern)
        item["_actions"] = actions
        item["_score"] = float(
            item.get("score")
            or item.get("lcb")
            or item.get("success_rate")
            or item.get("posterior_mean")
            or 0.0
        )
        item["pattern_id"] = item.get("pattern_id") or f"macro_{idx:04d}"
        library.append(item)
    library.sort(
        key=lambda p: (
            float(p.get("_score") or 0.0),
            len(p.get("_actions") or []),
            float(p.get("support") or p.get("frequency") or 0.0),
        ),
        reverse=True,
    )
    return library


def select_macro_pattern(
    library: list[dict[str, Any]],
    first_key: str | None,
    *,
    min_lcb: float,
    softgate: bool,
    max_skip: int,
) -> tuple[dict[str, Any], int, list[str]] | None:
    if not first_key:
        return None
    for pattern in library:
        if float(pattern.get("lcb") or pattern.get("success_rate") or 0.0) < min_lcb:
            continue
        actions = list(pattern.get("_actions") or [])
        starts: list[int]
        if softgate:
            starts = [i for i, action in enumerate(actions[:-1]) if action == first_key]
        else:
            starts = [0] if actions and actions[0] == first_key else []
        for start in starts:
            tail = actions[start + 1 : start + 1 + max_skip]
            if tail:
                return pattern, start, tail
    return None


def select_macro_chain_pattern(
    library: list[dict[str, Any]],
    first_key: str | None,
    chain_entries: list[ChainEntry],
    *,
    min_lcb: float,
    softgate: bool,
    max_skip: int,
    min_skip: int,
) -> tuple[dict[str, Any], int, list[ChainEntry]] | None:
    if not first_key or not chain_entries:
        return None
    best_depth1: tuple[dict[str, Any], int, list[ChainEntry]] | None = None
    for pattern in library:
        if float(pattern.get("lcb") or pattern.get("success_rate") or 0.0) < min_lcb:
            continue
        actions = list(pattern.get("_actions") or [])
        starts = (
            [i for i, action in enumerate(actions[:-1]) if action == first_key]
            if softgate
            else ([0] if actions and actions[0] == first_key else [])
        )
        for start in starts:
            tail = actions[start + 1 : start + 1 + max_skip]
            matched: list[ChainEntry] = []
            for expected_key, entry in zip(tail, chain_entries):
                if entry.action_key != expected_key:
                    break
                matched.append(entry)
            if len(matched) >= min_skip:
                return pattern, start, matched
            if len(matched) == 1 and best_depth1 is None:
                best_depth1 = (pattern, start, matched)
    return best_depth1


def _macro_stage_audit_event(
    *,
    stage: str,
    task_id: str,
    trace_index: int,
    step: int,
    owner_action_key: str | None,
    used_spec: bool,
    chain_entries: list[ChainEntry],
    pattern: dict[str, Any],
    start: int,
    predicted_suffix: list[str],
    matched_entries: list[ChainEntry] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Build a compact, decision-neutral filter-stage audit event."""
    matched_entries = matched_entries or []
    suffix = [str(item) for item in predicted_suffix if item is not None]
    return {
        "stage": stage,
        "task_id": task_id,
        "trace_index": trace_index,
        "step": step,
        "owner_action_key": owner_action_key,
        "used_spec": bool(used_spec),
        "s1_chain_depth": len(chain_entries),
        "s1_chain_keys": [str(entry.action_key) for entry in chain_entries],
        "pattern_id": str(pattern.get("pattern_id") or ""),
        "start_index": int(start),
        "predicted_suffix": suffix,
        "matched_action_keys": [str(entry.action_key) for entry in matched_entries],
        "suffix_len": len(suffix),
        "matched_len": len(matched_entries),
        "reason": reason,
    }


def _initial_state_parts(task) -> tuple[Any, Any, list[Message]]:
    if task.initial_state is None:
        return None, None, []
    return (
        task.initial_state.initialization_data,
        task.initial_state.initialization_actions,
        list(task.initial_state.message_history or []),
    )


def run_tau2_telecom_task(
    *,
    task,
    owner: LLMEndpoint,
    s1: LLMEndpoint | None = None,
    peer: LLMEndpoint | None = None,
    mode: Mode = "owner_only",
    max_steps: int = 80,
    max_errors: int = 3,
    policy_type: str = "manual",
    evaluation_type: EvaluationType = EvaluationType.ALL,
    seed: int | None = None,
    macro_library: str | Path | None = None,
    macro_min_lcb: float = 0.0,
    macro_max_skip: int = 4,
    macro_min_skip: int = 2,
    macro_softgate: bool = False,
    chain_max_depth: int = 1,
    macro_runtime_no_fire: bool = False,
    macro_stage_audit: bool = False,
) -> SimulationRun:
    if mode in ("spec_verify", "macro_skip") and s1 is None:
        raise ValueError(f"{mode} mode requires an s1 endpoint")
    macro_patterns = load_macro_library(macro_library) if mode == "macro_skip" else []

    env = get_environment(solo_mode=True, policy_type=policy_type)
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
        llm=owner.model,
        llm_args=owner.args(),
    )
    tools = prompt_agent.tools
    system_prompt = prompt_agent.system_prompt
    kickoff = UserMessage.text(
        "Begin solving the ticket now. Use exactly one tool call batch for the next step."
    )
    llm_history: list[Message] = [kickoff, *deepcopy(initial_history)]
    trajectory: list[Message] = [*deepcopy(initial_history)]
    metrics = RunMetrics(mode=mode, task_id=task.id, max_steps=max_steps)
    termination = TerminationReason.UNEXPECTED_ERROR
    step_count = 0
    start_wall = get_now()
    t_run0 = time.perf_counter()

    def owner_call() -> tuple[AssistantMessage, float]:
        return _generate_action(
            endpoint=owner,
            system_prompt=system_prompt,
            tools=tools,
            llm_history=llm_history,
            call_name="tau2_owner_action",
            agent=prompt_agent,
        )

    def peer_call(peer_history: list[Message]) -> tuple[AssistantMessage, float]:
        assert peer is not None
        return _generate_action(
            endpoint=peer,
            system_prompt=system_prompt,
            tools=tools,
            llm_history=peer_history,
            call_name="tau2_peer_action",
            agent=prompt_agent,
        )

    chain_lock = threading.Lock()
    s1_chain: list[ChainEntry] = []
    chain_state = {
        "env": env,
        "llm_history": list(llm_history),
        "committed_steps": step_count,
        "max_steps": max_steps,
    }
    s1_worker = None
    if mode in ("spec_verify", "macro_skip"):
        assert s1 is not None
        s1_worker = S1ChainWorker(
            endpoint=s1,
            system_prompt=system_prompt,
            tools=tools,
            agent=prompt_agent,
            state=chain_state,
            chain=s1_chain,
            state_lock=chain_lock,
            max_depth=chain_max_depth,
        )
        s1_worker.start()

    def publish_chain_state_locked() -> None:
        if s1_worker is not None:
            s1_worker.publish_state_locked(
                env=env,
                llm_history=llm_history,
                committed_steps=step_count,
            )

    def publish_chain_state() -> None:
        if s1_worker is not None:
            s1_worker.publish_state(
                env=env,
                llm_history=llm_history,
                committed_steps=step_count,
            )

    def reset_chain_state() -> None:
        if s1_worker is None:
            return
        with chain_lock:
            s1_worker.publish_state_locked(
                env=env,
                llm_history=llm_history,
                committed_steps=step_count,
            )
            s1_worker.signal_rollback_locked()

    def chain_head() -> ChainEntry | None:
        with chain_lock:
            return s1_chain[0] if s1_chain else None

    def submit_peer_from_chain() -> Any:
        if peer is None or executor is None:
            return None
        with chain_lock:
            if not s1_chain:
                return None
            entry = s1_chain[0]
            peer_history = list(llm_history) + [entry.message] + list(entry.tool_results)
        return executor.submit(peer_call, peer_history)

    executor = ThreadPoolExecutor(max_workers=2) if mode in ("spec_verify", "macro_skip") else None
    pending_peer_future = None

    try:
        while True:
            trace = StepTrace(step=step_count)
            s1_message: AssistantMessage | None = None
            spec_env = None
            spec_results: list[ToolMessage] | None = None
            next_peer_future = None
            owner_from_peer = False
            chain0: ChainEntry | None = None
            chain0_to_pop: ChainEntry | None = None
            audit_chain_before_commit: list[ChainEntry] = []

            if mode in ("spec_verify", "macro_skip"):
                assert executor is not None
                if pending_peer_future is not None:
                    owner_future = pending_peer_future
                    pending_peer_future = None
                    owner_from_peer = True
                    trace.owner_from_peer = True
                else:
                    owner_future = executor.submit(owner_call)
                while not owner_future.done():
                    if next_peer_future is None:
                        next_peer_future = submit_peer_from_chain()
                        if next_peer_future is not None:
                            trace.peer_submitted = True
                            metrics.peer_submits += 1
                    time.sleep(0.01)
                try:
                    owner_message, owner_dt = owner_future.result()
                except Exception as exc:
                    if owner_from_peer:
                        trace.peer_error = repr(exc)
                        metrics.peer_errors += 1
                    else:
                        trace.owner_error = repr(exc)
                    metrics.step_traces.append(trace)
                    if next_peer_future is not None:
                        next_peer_future.cancel()
                        trace.peer_cancelled = True
                        metrics.peer_cancels += 1
                    raise
                chain_deadline = time.perf_counter() + 0.5
                while chain0 is None and time.perf_counter() < chain_deadline:
                    chain0 = chain_head()
                    if chain0 is None:
                        if next_peer_future is None:
                            next_peer_future = submit_peer_from_chain()
                            if next_peer_future is not None:
                                trace.peer_submitted = True
                                metrics.peer_submits += 1
                        time.sleep(0.01)
                if chain0 is not None:
                    s1_message = chain0.message
                    spec_env = chain0.env_after
                    spec_results = chain0.tool_results
                    trace.s1_s = chain0.s1_s
                    trace.s1_action = _tool_signature(chain0.message)
                    trace.preexec_s = chain0.preexec_s
            else:
                owner_message, owner_dt = owner_call()

            trace.owner_s = owner_dt
            trace.owner_action = _tool_signature(owner_message)
            if owner_from_peer:
                trace.peer_s = owner_dt
                trace.peer_action = trace.owner_action
                metrics.peer_calls += 1
                metrics.peer_total_s += owner_dt
                metrics.peer_transfers += 1
                trace.peer_transferred = True
            else:
                metrics.owner_calls += 1
                metrics.owner_total_s += owner_dt

            trajectory.append(owner_message)
            llm_history.append(owner_message)
            if prompt_agent.is_stop(owner_message):
                termination = TerminationReason.AGENT_STOP
                metrics.step_traces.append(trace)
                break
            if not owner_message.tool_calls:
                termination = TerminationReason.AGENT_ERROR
                metrics.step_traces.append(trace)
                break

            if mode == "macro_skip" and macro_stage_audit:
                with chain_lock:
                    audit_chain_before_commit = list(s1_chain)

            used_spec = False
            if (
                mode in ("spec_verify", "macro_skip")
                and s1_message is not None
                and spec_env is not None
                and spec_results is not None
                and chain0 is not None
                and _same_tool_calls(owner_message, s1_message)
            ):
                metrics.exact_matches += 1
                trace.exact_match = True
                env = spec_env
                tool_results = _rewrite_tool_result_ids(
                    spec_results, owner_message.tool_calls
                )
                used_spec = True
                trace.used_spec_result = True
                metrics.spec_commits += 1
                pending_peer_future = next_peer_future
                chain0_to_pop = chain0
            else:
                if next_peer_future is not None:
                    next_peer_future.cancel()
                    trace.peer_cancelled = True
                    metrics.peer_cancels += 1
                tool_results = [
                    env.get_response(call) for call in owner_message.tool_calls
                ]
                reset_chain_state()

            metrics.tool_errors += sum(1 for result in tool_results if result.error)
            env_message = _wrap_tool_results(tool_results)
            flat_tool_results = _flatten_env_message(env_message)
            trajectory.extend(flat_tool_results)
            llm_history.extend(flat_tool_results)
            step_count += 1
            if chain0_to_pop is not None and s1_worker is not None:
                with chain_lock:
                    if s1_chain and s1_chain[0] is chain0_to_pop:
                        s1_chain.pop(0)
                    publish_chain_state_locked()
            else:
                publish_chain_state()
            if step_count >= max_steps:
                termination = TerminationReason.MAX_STEPS
                metrics.step_traces.append(trace)
                break
            if metrics.tool_errors >= max_errors:
                termination = TerminationReason.TOO_MANY_ERRORS
                metrics.step_traces.append(trace)
                break

            # Passing ENV -> AGENT is another TAU2 half-duplex step.
            step_count += 1
            publish_chain_state()
            metrics.step_traces.append(trace)
            if step_count >= max_steps:
                termination = TerminationReason.MAX_STEPS
                break

            if (
                mode == "macro_skip"
                and macro_stage_audit
                and macro_patterns
                and metrics.tool_errors < max_errors
            ):
                owner_action_key = _action_key(owner_message)
                trace_index = len(metrics.step_traces) - 1
                s1_anchor_action_key = (
                    audit_chain_before_commit[0].action_key
                    if audit_chain_before_commit
                    else None
                )
                s1_suffix_chain = (
                    list(audit_chain_before_commit[1:])
                    if audit_chain_before_commit
                    else []
                )
                verified_suffix_chain = (
                    list(s1_suffix_chain)
                    if s1_anchor_action_key == owner_action_key
                    else []
                )

                library_match = select_macro_pattern(
                    macro_patterns,
                    owner_action_key,
                    min_lcb=macro_min_lcb,
                    softgate=macro_softgate,
                    max_skip=macro_max_skip,
                )
                if library_match is not None:
                    pattern, start, tail = library_match
                    metrics.macro_stage_audit_events.append(
                        _macro_stage_audit_event(
                            stage="library_match_only",
                            task_id=task.id,
                            trace_index=trace_index,
                            step=trace.step,
                            owner_action_key=owner_action_key,
                            used_spec=used_spec,
                            chain_entries=verified_suffix_chain,
                            pattern=pattern,
                            start=start,
                            predicted_suffix=[str(action) for action in tail],
                            reason="owner_action_anchor_match",
                        )
                    )

                materialized = select_macro_chain_pattern(
                    macro_patterns,
                    s1_anchor_action_key,
                    s1_suffix_chain,
                    min_lcb=macro_min_lcb,
                    softgate=macro_softgate,
                    max_skip=macro_max_skip,
                    min_skip=1,
                )
                if materialized is not None and len(materialized[2]) >= 1:
                    pattern, start, matched_entries = materialized
                    matched_keys = [str(entry.action_key) for entry in matched_entries]
                    event = _macro_stage_audit_event(
                        stage="s1_materialized_suffix_depth1",
                        task_id=task.id,
                        trace_index=trace_index,
                        step=trace.step,
                        owner_action_key=owner_action_key,
                        used_spec=used_spec,
                        chain_entries=s1_suffix_chain,
                        pattern=pattern,
                        start=start,
                        predicted_suffix=[str(s1_anchor_action_key), *matched_keys],
                        matched_entries=matched_entries,
                        reason="s1_materialized_macro_before_owner_verification",
                    )
                    event["s1_anchor_action_key"] = s1_anchor_action_key
                    event["compare_from"] = "current_owner"
                    metrics.macro_stage_audit_events.append(event)
                    if used_spec:
                        metrics.macro_stage_audit_events.append(
                            _macro_stage_audit_event(
                                stage="owner_verified_anchor",
                                task_id=task.id,
                                trace_index=trace_index,
                                step=trace.step,
                                owner_action_key=owner_action_key,
                                used_spec=used_spec,
                                chain_entries=verified_suffix_chain,
                                pattern=pattern,
                                start=start,
                                predicted_suffix=matched_keys,
                                matched_entries=matched_entries,
                                reason="owner_anchor_strict_tool_call_match",
                            )
                        )

                if used_spec:
                    guarded = select_macro_chain_pattern(
                        macro_patterns,
                        owner_action_key,
                        verified_suffix_chain,
                        min_lcb=macro_min_lcb,
                        softgate=macro_softgate,
                        max_skip=macro_max_skip,
                        min_skip=macro_min_skip,
                    )
                    if guarded is not None and len(guarded[2]) >= macro_min_skip:
                        pattern, start, matched_entries = guarded
                        metrics.macro_stage_audit_events.append(
                            _macro_stage_audit_event(
                                stage="runtime_depth_guarded_candidate",
                                task_id=task.id,
                                trace_index=trace_index,
                                step=trace.step,
                                owner_action_key=owner_action_key,
                                used_spec=used_spec,
                                chain_entries=verified_suffix_chain,
                                pattern=pattern,
                                start=start,
                                predicted_suffix=[
                                    str(entry.action_key) for entry in matched_entries
                                ],
                                matched_entries=matched_entries,
                                reason="owner_verified_and_depth_ge_macro_min_skip",
                            )
                        )

            if (
                mode == "macro_skip"
                and used_spec
                and macro_patterns
                and metrics.tool_errors < max_errors
            ):
                with chain_lock:
                    macro_chain_snapshot = list(s1_chain)
                selected = select_macro_chain_pattern(
                    macro_patterns,
                    _action_key(owner_message),
                    macro_chain_snapshot,
                    min_lcb=macro_min_lcb,
                    softgate=macro_softgate,
                    max_skip=macro_max_skip,
                    min_skip=macro_min_skip,
                )
                if selected is not None:
                    pattern, _start, matched_entries = selected
                    if len(matched_entries) < macro_min_skip:
                        metrics.macro_depth1_suppressed += 1
                        trace.macro_suppressed_depth1 = True
                        selected = None
                if selected is not None:
                    pattern, _start, matched_entries = selected
                    skipped_this_fire = 0
                    runtime_suppressed = False
                    with chain_lock:
                        current_entries = list(s1_chain[: len(matched_entries)])
                        if current_entries != matched_entries:
                            metrics.macro_rejects += 1
                        elif macro_runtime_no_fire:
                            runtime_suppressed = True
                            skipped_this_fire = len(matched_entries)
                        else:
                            for entry in matched_entries:
                                macro_message = entry.message
                                macro_results = entry.tool_results
                                env = entry.env_after
                                trajectory.append(macro_message)
                                llm_history.append(macro_message)
                                metrics.tool_errors += sum(
                                    1 for result in macro_results if result.error
                                )
                                trajectory.extend(macro_results)
                                llm_history.extend(macro_results)
                                skipped_this_fire += 1
                                step_count += 1
                                if step_count >= max_steps:
                                    termination = TerminationReason.MAX_STEPS
                                    break
                                if metrics.tool_errors >= max_errors:
                                    termination = TerminationReason.TOO_MANY_ERRORS
                                    break
                                step_count += 1
                                if step_count >= max_steps:
                                    termination = TerminationReason.MAX_STEPS
                                    break
                            if skipped_this_fire:
                                del s1_chain[:skipped_this_fire]
                                publish_chain_state_locked()
                    if runtime_suppressed:
                        metrics.macro_runtime_suppressed += 1
                        metrics.macro_steps_would_skip += skipped_this_fire
                        trace.macro_pattern_id = str(pattern.get("pattern_id"))
                        trace.macro_runtime_suppressed = True
                        trace.macro_would_skip = skipped_this_fire
                    elif skipped_this_fire:
                        if pending_peer_future is not None:
                            pending_peer_future.cancel()
                            trace.peer_cancelled = True
                            metrics.peer_cancels += 1
                            pending_peer_future = None
                        pid = str(pattern.get("pattern_id"))
                        metrics.macro_hits += 1
                        metrics.macro_steps_skipped += skipped_this_fire
                        metrics.macro_patterns[pid] = metrics.macro_patterns.get(pid, 0) + 1
                        trace.macro_pattern_id = pid
                        trace.macro_skipped = skipped_this_fire
                    if termination in (
                        TerminationReason.MAX_STEPS,
                        TerminationReason.TOO_MANY_ERRORS,
                    ):
                        break
    except Exception as exc:
        termination = TerminationReason.INFRASTRUCTURE_ERROR
        metrics.step_traces.append(
            StepTrace(step=step_count, owner_error=repr(exc))
        )
    finally:
        if pending_peer_future is not None:
            pending_peer_future.cancel()
        if s1_worker is not None:
            s1_worker.stop()
            s1_worker.join(timeout=1.0)
            metrics.s1_calls = s1_worker.s1_call_count
            metrics.s1_total_s = s1_worker.s1_total_s
            metrics.preexec_total_s = s1_worker.preexec_total_s
            metrics.s1_chain_appends = s1_worker.chain_appends
            metrics.s1_chain_discards = s1_worker.chain_discards
            metrics.s1_chain_max_depth_observed = s1_worker.max_depth_observed
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    duration = time.perf_counter() - t_run0
    messages = deepcopy(trajectory)
    for idx, msg in enumerate(messages):
        msg.turn_idx = idx
    cost = get_cost(messages)
    agent_cost = None if cost is None else cost[0]
    simulation = SimulationRun(
        id=str(uuid.uuid4()),
        task_id=task.id,
        start_time=start_wall,
        end_time=get_now(),
        duration=duration,
        termination_reason=termination,
        agent_cost=agent_cost,
        user_cost=None,
        reward_info=None,
        messages=messages,
        seed=seed,
        mode="half_duplex",
        info=metrics.to_info(),
    )
    simulation.policy = env.get_policy()
    simulation.reward_info = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=evaluation_type,
        solo_mode=True,
        domain=env.get_domain_name(),
    )
    return simulation


def select_tasks(split: str, limit: int | None = None, offset: int = 0):
    tasks = get_tasks(split)
    if limit is not None:
        return tasks[offset : offset + limit]
    return tasks[offset:]


def write_result(path: Path, result: SimulationRun) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def summarize_results(results: list[SimulationRun]) -> dict[str, Any]:
    rewards = [r.reward_info.reward if r.reward_info else 0.0 for r in results]
    return {
        "num_tasks": len(results),
        "reward_sum": sum(rewards),
        "accuracy": (sum(1 for r in rewards if r == 1.0) / len(results))
        if results
        else 0.0,
        "avg_reward": (sum(rewards) / len(results)) if results else 0.0,
        "avg_duration_s": (
            sum(float(r.duration or 0.0) for r in results) / len(results)
        )
        if results
        else 0.0,
        "terminations": {
            reason: sum(1 for r in results if str(r.termination_reason) == reason)
            for reason in sorted({str(r.termination_reason) for r in results})
        },
        "owner_calls": sum((r.info or {}).get("owner_calls", 0) for r in results),
        "s1_calls": sum((r.info or {}).get("s1_calls", 0) for r in results),
        "peer_calls": sum((r.info or {}).get("peer_calls", 0) for r in results),
        "exact_matches": sum((r.info or {}).get("exact_matches", 0) for r in results),
        "spec_commits": sum((r.info or {}).get("spec_commits", 0) for r in results),
        "peer_submits": sum((r.info or {}).get("peer_submits", 0) for r in results),
        "peer_transfers": sum((r.info or {}).get("peer_transfers", 0) for r in results),
        "peer_cancels": sum((r.info or {}).get("peer_cancels", 0) for r in results),
        "peer_errors": sum((r.info or {}).get("peer_errors", 0) for r in results),
        "macro_hits": sum((r.info or {}).get("macro_hits", 0) for r in results),
        "macro_steps_skipped": sum(
            (r.info or {}).get("macro_steps_skipped", 0) for r in results
        ),
        "macro_rejects": sum((r.info or {}).get("macro_rejects", 0) for r in results),
        "macro_depth1_suppressed": sum(
            (r.info or {}).get("macro_depth1_suppressed", 0) for r in results
        ),
        "macro_runtime_suppressed": sum(
            (r.info or {}).get("macro_runtime_suppressed", 0) for r in results
        ),
        "macro_steps_would_skip": sum(
            (r.info or {}).get("macro_steps_would_skip", 0) for r in results
        ),
        "macro_stage_audit_events": sum(
            len((r.info or {}).get("macro_stage_audit_events") or [])
            for r in results
        ),
        "macro_stage_audit_stage_counts": {
            stage: sum(
                1
                for result in results
                for event in ((result.info or {}).get("macro_stage_audit_events") or [])
                if str(event.get("stage")) == stage
            )
            for stage in sorted(
                {
                    str(event.get("stage"))
                    for result in results
                    for event in ((result.info or {}).get("macro_stage_audit_events") or [])
                }
            )
        },
        "tool_errors": sum((r.info or {}).get("tool_errors", 0) for r in results),
        "s1_chain_appends": sum((r.info or {}).get("s1_chain_appends", 0) for r in results),
        "s1_chain_discards": sum(
            (r.info or {}).get("s1_chain_discards", 0) for r in results
        ),
        "s1_chain_max_depth_observed": max(
            ((r.info or {}).get("s1_chain_max_depth_observed", 0) for r in results),
            default=0,
        ),
    }


def write_summary(path: Path, config: dict[str, Any], results: list[SimulationRun]) -> None:
    summary = {"config": config, "summary": summarize_results(results)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
