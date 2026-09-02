"""v6 Actor-Relay Pipeline Agent for AppWorld.

Architecture:
- Two 27B actor lanes (a1, a2) hold dynamic roles: owner and peer
  - owner is the authoritative actor producing ground truth for step t
  - peer is the lookahead actor pre-computing step t+1 on a speculative prefix
  Which GPU holds the owner role at any moment is NOT periodic — it is
  determined solely by the history of verify outcomes so far.
- One 4B speculator (s1) runs as a continuous chain in a background thread,
  pre-executing read-only actions to build speculative contexts (depth ≤ 4).
- The main loop is a verify-driven state machine:
    S0 BOOTSTRAP → S1 WAIT_AUTH_S1 → S2 VERIFY → (S3 SUCCESS_TRANSFER |
                                                   S4 FAILURE_STICKY) → S1 …

Critical fix vs v5: peer_future is launched EAGERLY the moment s1(t) becomes
ready during the owner wait (typically ~0.7s into a ~2s owner call), not
AFTER the owner commits. This gives peer a ~0.7s head start so verify
success can rename peer → new owner with ~0.7s remaining instead of 2s.

Hard invariants:
  Inv 1: every committed step has exactly one authoritative producer
  Inv 2: ownership transfers ONLY on verify success
  Inv 3: verify failure invalidates the ENTIRE speculative suffix from step t
  Inv 4: on verify success → transfer ownership; on verify failure OR
         hard speculation boundary (mutating or pre-exec-failed s1(t)) →
         ownership stays sticky on the current owner_url

Backward compat: function signature and key return-dict fields
(`pipeline_hits`, `tier_counts[tier1/tier3/stall]`) are stable so
run_pipeline.py does not need changes.
"""

import json
import threading
import time
import concurrent.futures
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from openai import OpenAI

from .tools import TOOLS
from .agent import (
    execute_tool as _raw_execute_tool,
    SYSTEM_PROMPT,
    perf_counter,
    DISABLE_THINKING,
    _thinking_body,
    _parse_tool_call as _agent_parse_tool_call,
    _parse_qwen_tool_call,
    _build_canonical_message,
)
from .trajectory_logger import TrajectoryLogger
from .api_classification import is_read_only, is_mutating
from .event_trace import NULL_TRACER, messages_fingerprint


def execute_tool(env, tool_name: str, tool_args: dict) -> str:
    """Wrapper around agent.execute_tool that GUARANTEES the AppWorld
    safety_guard is disabled after the call.

    AppWorld's env.execute() enables a process-wide safety_guard (monkey-
    patches builtins.open). If env.execute hits an exception path that
    bypasses its normal cleanup (e.g., IPython internal state issue under
    threading), the guard is left enabled and subsequent file I/O in the
    main thread (including logger writes) fails with PermissionError.
    This wrapper ensures the guard is always reset before returning.
    """
    try:
        return _raw_execute_tool(env, tool_name, tool_args)
    finally:
        try:
            env.safety_guard.disable()
        except Exception:
            pass


def _fork_exec(env, action_name: str, action_args: dict) -> str:
    """Execute a mutating action speculatively via save → exec → rollback.

    Uses SQLite backup API directly on the cached in-memory connections
    to avoid CachedDBHandler cache-hit bugs (where _load_state silently
    skips the actual data restoration).
    """
    import sqlite3
    from appworld.apps.model_lib import CachedDBHandler

    # 1. Snapshot all in-memory DB connections via SQLite backup
    snapshots = {}
    for db_path, (engine, tracker) in CachedDBHandler.cache.items():
        conn = engine.raw_connection().connection
        backup_conn = sqlite3.connect(":memory:")
        conn.backup(backup_conn)
        snapshots[db_path] = backup_conn

    # 2. Save env metadata that execute() mutates
    saved_num = env.num_interactions
    saved_io_len = len(env.environment_io)

    # 3. Execute the mutating action
    tool_result = execute_tool(env, action_name, action_args)

    # 4. Rollback: restore each DB from its snapshot
    for db_path, backup_conn in snapshots.items():
        cached = CachedDBHandler.cache.get(db_path)
        if cached is None:
            continue
        engine, tracker = cached
        live_conn = engine.raw_connection().connection
        backup_conn.backup(live_conn)
        backup_conn.close()

    # 5. Restore env metadata
    env.num_interactions = saved_num
    del env.environment_io[saved_io_len:]

    return tool_result


# ─── Parsing helpers ────────────────────────────────────────────────────────

def _parse_tool_call(message) -> tuple:
    """Extract (tool_name, tool_args, raw_msg_dict) from response message.

    Delegates native tool_call + Qwen XML fallback to agent.py; the extra
    msg_dict return is what the pipeline main loop needs for history.
    """
    msg_dict = {"role": message.role, "content": message.content}
    name, args = _agent_parse_tool_call(message)
    if name and message.tool_calls:
        tc = message.tool_calls[0]
        msg_dict["tool_calls"] = [{
            "id": tc.id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]
        return name, args, msg_dict
    if message.content:
        name, args = _parse_qwen_tool_call(message.content)
        if name:
            return name, args, msg_dict
    return "", {}, msg_dict


def _verify_action_match(name1, args1, name2, args2) -> bool:
    """Compare two parsed actions for equality (exact match)."""
    if name1 != name2:
        return False
    if name1 == "execute_api":
        return (args1.get("app_name") == args2.get("app_name") and
                args1.get("api_name") == args2.get("api_name") and
                args1.get("parameters", {}) == args2.get("parameters", {}))
    return args1 == args2


def _action_norm(name: str, args: dict) -> str:
    """Normalize action to 'app.api' for is_read_only() and printing."""
    if name == "execute_api":
        return f"{args.get('app_name','?')}.{args.get('api_name','?')}"
    return name



# ─── OpenAI client cache ───────────────────────────────────────────────────

_clients: Dict[str, OpenAI] = {}

def _get_client(url: str) -> OpenAI:
    if url not in _clients:
        _clients[url] = OpenAI(base_url=url, api_key="dummy",
                                timeout=300.0, max_retries=2)
    return _clients[url]


def _completion(messages, model, url, tools, thinking, thinking_budget,
                max_tokens):
    """Call a vLLM model, halving max_tokens on 400-context-length errors.

    vLLM returns HTTP 400 with "context length" in the message when
    `input_tokens + max_tokens > max_model_len`. On pipeline runs the
    committed messages list can grow past the safe window (especially inside
    no_tool_call retry loops or long execute_api traces). On such an error
    we retry with a halved `max_tokens` up to 3 times — the request succeeds
    as long as `input_tokens + max_tokens/k` still fits. A tighter
    max_tokens may truncate thinking, but losing a step's reasoning is
    strictly better than losing the whole task.
    """
    client = _get_client(url)
    extra_body = _thinking_body(thinking, thinking_budget)
    attempts = 0
    cur_max = max_tokens
    while True:
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                temperature=0.0,
                extra_body=extra_body,
                max_tokens=cur_max,
            )
        except Exception as e:
            msg = str(e).lower()
            is_ctx = ("context length" in msg or "maximum context" in msg
                      or "input_tokens" in msg)
            if not is_ctx or attempts >= 3 or cur_max <= 256:
                raise
            attempts += 1
            cur_max = max(256, cur_max // 2)


# ─── Semantic verification (stub for future enable) ────────────────────────

def _s1_yes_no(prompt: str, s1_url: str, speculator_model: str) -> bool:
    """Ask s1 a yes/no question; return True iff the answer starts with YES.

    Shared plumbing for `_semantic_match` and `_exec_api_semantic_match` —
    both judges send a user prompt, read up to 8 tokens with temperature=0
    and thinking disabled, and treat any exception as "no". Any retune of
    the token budget or parsing lives here.
    """
    try:
        client = _get_client(s1_url)
        resp = client.chat.completions.create(
            model=speculator_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=8,
            extra_body=DISABLE_THINKING,
        )
        text = (resp.choices[0].message.content or "").strip().upper()
        return text.startswith("YES")
    except Exception:
        return False


def _semantic_match(name1, args1, name2, args2,
                    s1_url, speculator_model) -> bool:
    """Use s1 forward call to judge if two actions are semantically equivalent.

    Currently only enabled when verify_mode='semantic'. Default verify is exact.
    """
    norm1 = _action_norm(name1, args1)
    norm2 = _action_norm(name2, args2)
    prompt = (
        "Are these two API calls equivalent in their effect on the application "
        "state and the result they return?\n\n"
        f"Call A: {norm1}({json.dumps(args1)})\n"
        f"Call B: {norm2}({json.dumps(args2)})\n\n"
        "Answer with exactly one word: YES or NO."
    )
    return _s1_yes_no(prompt, s1_url, speculator_model)


def _exec_api_semantic_match(name1, args1, name2, args2,
                              s1_url, speculator_model) -> bool:
    """Narrow semantic verify (A2): same app+api only, param judge via s1.

    Conservative scope: both actions must be execute_api with identical
    app_name AND api_name. Only the `parameters` dict is judged. This
    excludes any cross-API equivalence (too risky) and only relaxes the
    exact-parameter requirement when the judge is confident.
    """
    if name1 != "execute_api" or name2 != "execute_api":
        return False
    if args1.get("app_name") != args2.get("app_name"):
        return False
    if args1.get("api_name") != args2.get("api_name"):
        return False
    p1 = args1.get("parameters", {})
    p2 = args2.get("parameters", {})
    if p1 == p2:
        return True
    app = args1.get("app_name")
    api = args1.get("api_name")
    prompt = (
        f"You are verifying two calls to {app}.{api} that have different "
        f"parameter dicts.\n"
        f"Are they semantically equivalent — i.e., they would produce the "
        f"SAME side effects AND return the SAME data?\n"
        f"A trivial difference (e.g., added default, reordered list of "
        f"identical items) counts as equivalent. Any real behavior change "
        f"counts as NOT equivalent.\n\n"
        f"Parameters A: {json.dumps(p1, sort_keys=True)}\n"
        f"Parameters B: {json.dumps(p2, sort_keys=True)}\n\n"
        f"Answer with exactly one word: YES or NO."
    )
    return _s1_yes_no(prompt, s1_url, speculator_model)


# ─── Chain entry & worker thread ───────────────────────────────────────────

@dataclass
class ChainEntry:
    """One s1-predicted step in the chain.

    `is_hard_boundary` is True when the main loop cannot promote a peer
    across this entry — either because it is a writable API (never
    pre-executed) or because its read-only pre-exec raised. In both cases
    there is no spec_env(t) for downstream speculation.
    """
    action_name: str
    action_args: dict
    tool_result: Optional[str]
    is_mutating: bool
    preexec_failed: bool = False
    pre_exec_time: float = 0.0

    @property
    def is_hard_boundary(self) -> bool:
        return self.tool_result is None


class S1ChainWorker(threading.Thread):
    """Background s1 chain predictor.

    Continuously generates new chain entries up to max_depth, each step using
    the previous step's pre_exec result as context. Pauses on demand (e.g.,
    when main loop is committing/rolling back).

    Concurrency model:
      - main thread holds `state_lock` while reading/modifying chain & epoch
      - worker holds `state_lock` only briefly (snapshot/append)
      - epoch counter detects stale work after rollback: worker discards
        results computed under an old epoch
      - paused event lets main thread freeze the worker around state changes

    Note: env.execute calls happen OUTSIDE the lock and may waste interactions
    when discarded by epoch mismatch. This is acceptable for v5; future work
    can use save_state/load_state to fork-and-rollback for free.
    """

    def __init__(self, env, messages_ref, chain, state_lock, env_lock,
                 s1_url, speculator_model, s1_thinking, s1_thinking_budget,
                 max_depth, max_step, tracer=None, env_fork=False):
        super().__init__(daemon=True)
        self.env = env
        self.messages_ref = messages_ref     # main message list (shared)
        self.chain = chain                   # main chain list (shared)
        self.state_lock = state_lock         # main state lock
        self.env_lock = env_lock             # serializes env.execute across threads
        self.s1_url = s1_url
        self.speculator_model = speculator_model
        self.s1_thinking = s1_thinking
        self.s1_thinking_budget = s1_thinking_budget
        self.max_depth = max_depth
        self.max_step = max_step
        self.tracer = tracer if tracer is not None else NULL_TRACER
        self.env_fork = env_fork

        self.committed_step = 0              # protected by state_lock
        self.epoch = 0                        # bumped on rollback
        self.stop_flag = False
        self.paused = threading.Event()
        self.paused.set()                    # initially running
        self.s1_total_time = 0.0
        self.s1_call_count = 0
        self.preexec_total = 0
        self.preexec_wasted = 0              # discarded due to epoch mismatch
        self.fork_exec_total = 0
        self.fork_exec_wasted = 0

        # Speculative fork continuation state: lets the worker build chain
        # entries past mutating boundaries by maintaining a persistent
        # speculative DB branch separate from the real env.
        self._spec_dirty = False
        self._spec_dbs: dict = {}       # post-mutation speculative DB snapshots
        self._clean_dbs: dict = {}      # pre-mutation baseline DB snapshots
        self._spec_num = 0              # speculative num_interactions
        self._clean_num = 0             # baseline num_interactions
        self._clean_io_len = 0          # baseline environment_io length
        self._clean_committed = -1      # committed_step when clean was taken

    def update_committed(self, step: int):
        """Tell worker the new committed step (under state_lock)."""
        self.committed_step = step

    def signal_rollback(self):
        """Bump epoch — any in-flight worker computation becomes stale."""
        self.epoch += 1
        self._clear_spec()

    def _clear_spec(self):
        """Discard speculative fork state.

        Does not explicitly close SQLite connections because this may be
        called from the main thread (signal_rollback) while the connections
        were created in the worker thread. In-memory connections are freed
        by GC when references are dropped.
        """
        self._spec_dbs = {}
        self._clean_dbs = {}
        self._spec_dirty = False
        self._clean_committed = -1

    def _snapshot_clean(self):
        """Snapshot current real env as the clean baseline."""
        import sqlite3
        from appworld.apps.model_lib import CachedDBHandler
        self._clean_dbs = {}
        for db_path, (engine, _) in CachedDBHandler.cache.items():
            conn = engine.raw_connection().connection
            backup = sqlite3.connect(":memory:")
            conn.backup(backup)
            self._clean_dbs[db_path] = backup
        self._clean_num = self.env.num_interactions
        self._clean_io_len = len(self.env.environment_io)

    def _enter_spec(self, action_name, action_args):
        """First mutating entry: snapshot clean, execute, keep dirty state."""
        import sqlite3
        from appworld.apps.model_lib import CachedDBHandler
        self._snapshot_clean()
        saved_num = self.env.num_interactions
        saved_io_len = len(self.env.environment_io)
        tool_result = execute_tool(self.env, action_name, action_args)
        # Snapshot the dirty (post-mutation) state
        self._spec_dbs = {}
        for db_path, (engine, _) in CachedDBHandler.cache.items():
            conn = engine.raw_connection().connection
            backup = sqlite3.connect(":memory:")
            conn.backup(backup)
            self._spec_dbs[db_path] = backup
        self._spec_num = self.env.num_interactions
        # Restore clean state to real DBs
        for db_path, clean_conn in self._clean_dbs.items():
            cached = CachedDBHandler.cache.get(db_path)
            if cached is None:
                continue
            engine, _ = cached
            live_conn = engine.raw_connection().connection
            clean_conn.backup(live_conn)
        self.env.num_interactions = saved_num
        del self.env.environment_io[saved_io_len:]
        self._spec_dirty = True
        self._clean_committed = self.committed_step
        return tool_result

    def _exec_on_spec(self, action_name, action_args):
        """Execute on speculative state, then restore clean to real DBs."""
        import sqlite3
        from appworld.apps.model_lib import CachedDBHandler
        # Restore speculative state to real DBs
        for db_path, spec_conn in self._spec_dbs.items():
            cached = CachedDBHandler.cache.get(db_path)
            if cached is None:
                continue
            engine, _ = cached
            live_conn = engine.raw_connection().connection
            spec_conn.backup(live_conn)
        self.env.num_interactions = self._spec_num
        # Execute
        tool_result = execute_tool(self.env, action_name, action_args)
        # Save updated speculative state
        for db_path, spec_conn in self._spec_dbs.items():
            cached = CachedDBHandler.cache.get(db_path)
            if cached is None:
                continue
            engine, _ = cached
            live_conn = engine.raw_connection().connection
            live_conn.backup(spec_conn)
        self._spec_num = self.env.num_interactions
        # Restore clean state to real DBs
        for db_path, clean_conn in self._clean_dbs.items():
            cached = CachedDBHandler.cache.get(db_path)
            if cached is None:
                continue
            engine, _ = cached
            live_conn = engine.raw_connection().connection
            clean_conn.backup(live_conn)
        self.env.num_interactions = self._clean_num
        del self.env.environment_io[self._clean_io_len:]
        return tool_result

    def pause(self):
        self.paused.clear()

    def resume(self):
        self.paused.set()

    def stop(self):
        self.stop_flag = True
        self.paused.set()
        self._clear_spec()

    def run(self):
        max_tokens_s1 = (self.s1_thinking_budget + 1024) if self.s1_thinking else 1024
        while not self.stop_flag:
            self.paused.wait()
            if self.stop_flag:
                break

            # Snapshot state under lock
            with self.state_lock:
                cur_epoch = self.epoch
                committed_snapshot = self.committed_step
                chain_snapshot = list(self.chain)
                msgs_snapshot = list(self.messages_ref)
                last_entry = chain_snapshot[-1] if chain_snapshot else None
                last_is_boundary = (last_entry is not None
                                    and last_entry.is_hard_boundary)
                last_complete = any(e.action_name == "complete_task"
                                     for e in chain_snapshot)

            if (len(chain_snapshot) >= self.max_depth
                    or committed_snapshot + len(chain_snapshot) >= self.max_step
                    or last_is_boundary or last_complete):
                time.sleep(0.02)
                continue

            # Build context: committed messages + canonical post-step for each chain entry
            ctx = list(msgs_snapshot)
            for i, entry in enumerate(chain_snapshot):
                if entry.is_hard_boundary:
                    break
                a, t = _build_canonical_message(
                    committed_snapshot + 1 + i,
                    entry.action_name, entry.action_args, entry.tool_result)
                ctx.append(a)
                ctx.append(t)

            # 4B forward call (slow ~0.7s)
            self.tracer.emit("s1_worker", "llm_submit",
                             committed_snapshot=committed_snapshot,
                             chain_depth_pre=len(chain_snapshot),
                             prefix_len=len(ctx),
                             prefix_hash=messages_fingerprint(ctx))
            t0 = perf_counter()
            try:
                with self.tracer.span("s1_worker", "llm_call",
                                      chain_depth_pre=len(chain_snapshot)):
                    resp = _completion(
                        ctx, self.speculator_model, self.s1_url, TOOLS,
                        self.s1_thinking,
                        self.s1_thinking_budget if self.s1_thinking else 0,
                        max_tokens_s1)
            except Exception:
                time.sleep(0.05)
                continue
            t1 = perf_counter()
            self.s1_total_time += t1 - t0
            self.s1_call_count += 1

            try:
                action_name, action_args, _ = _parse_tool_call(
                    resp.choices[0].message)
            except Exception:
                continue
            if not action_name:
                time.sleep(0.05)
                continue

            mutating = not is_read_only(_action_norm(action_name, action_args))
            terminal = (action_name == "complete_task")

            tool_result = None
            pre_t = 0.0
            preexec_failed = False
            if not terminal:
                with self.state_lock:
                    if self.epoch != cur_epoch:
                        continue
                tp0 = perf_counter()
                wait_span = self.tracer.span(
                    "env", "env_lock_wait", who="s1_worker", tool=action_name)
                wait_span.__enter__()
                self.env_lock.acquire()
                wait_span.__exit__(None, None, None)
                try:
                    # If committed_step advanced (front-pops), the real env
                    # has been updated by _exec_real. Re-snapshot clean baseline
                    # so we don't revert the main loop's work on restore.
                    if (self._spec_dirty
                            and self.committed_step != self._clean_committed):
                        self._snapshot_clean()
                        self._clean_committed = self.committed_step

                    if self._spec_dirty:
                        with self.tracer.span("env", "env_spec_exec",
                                              who="s1_worker", tool=action_name):
                            try:
                                tool_result = self._exec_on_spec(
                                    action_name, action_args)
                            except Exception:
                                tool_result = None
                                preexec_failed = True
                    elif not mutating:
                        with self.tracer.span("env", "env_preexec",
                                              who="s1_worker", tool=action_name):
                            try:
                                tool_result = execute_tool(
                                    self.env, action_name, action_args)
                            except Exception:
                                tool_result = None
                                preexec_failed = True
                    elif self.env_fork:
                        with self.tracer.span("env", "env_fork_enter",
                                              who="s1_worker", tool=action_name):
                            try:
                                tool_result = self._enter_spec(
                                    action_name, action_args)
                            except Exception:
                                tool_result = None
                                preexec_failed = True
                finally:
                    self.env_lock.release()
                pre_t = perf_counter() - tp0
                if not preexec_failed and tool_result is not None:
                    if mutating:
                        self.fork_exec_total += 1
                    else:
                        self.preexec_total += 1

            # Append entry under lock, double-check epoch
            entry = ChainEntry(
                action_name=action_name,
                action_args=action_args,
                tool_result=tool_result,
                is_mutating=(mutating or terminal),
                preexec_failed=preexec_failed,
                pre_exec_time=pre_t,
            )
            with self.state_lock:
                # Epoch check is sufficient: every invalidation (reset, clear,
                # semantic) bumps epoch. Front-pops (S3 transfer, meta-tool)
                # don't bump epoch and ARE legitimate — the worker's prediction
                # is still valid because popped entries were committed to
                # messages, preserving the effective prefix.
                # Sanity: chain can only shrink (front-pop) or stay same length
                # without epoch bump. Growth means an unexpected appender.
                stale = (self.epoch != cur_epoch
                         or len(self.chain) > len(chain_snapshot))
                if stale:
                    self._clear_spec()
                    if tool_result is not None:
                        if mutating:
                            self.fork_exec_wasted += 1
                        else:
                            self.preexec_wasted += 1
                    self.tracer.emit("s1_worker", "stale_discarded",
                                     committed_snapshot=committed_snapshot,
                                     committed_now=self.committed_step,
                                     had_preexec=(tool_result is not None))
                    continue
                self.chain.append(entry)
                self.tracer.emit("s1_worker", "chain_append",
                                 chain_depth_post=len(self.chain),
                                 action=action_name,
                                 is_mutating=entry.is_mutating,
                                 preexec_failed=preexec_failed)


# ─── Main pipeline function ────────────────────────────────────────────────

def run_pipeline_agent(
    env,
    task_instruction: str,
    actor_model: str = "Qwen/Qwen3.5-27B",
    speculator_model: str = "Qwen/Qwen3.5-4B",
    a1_url: str = "http://localhost:8004/v1",
    a2_url: str = "http://localhost:8005/v1",
    s1_url: str = "http://localhost:8003/v1",
    thinking_budget: int = 4096,
    max_steps: int = 20,
    s1_thinking: bool = False,
    s1_thinking_budget: int = 2048,
    logger: Optional[TrajectoryLogger] = None,
    shell_only: bool = False,
    chain_max_depth: int = 4,            # v6: s1 chain depth
    verify_mode: str = "exact",          # v6: "exact" | "exec_api_semantic" | "semantic"
    reset_primer: bool = False,          # kickstart 4B on reset to seed chain[0]
    primer_gate: str = "uniform",        # "uniform" | "targeted" | "cond1" | "cond2"
    primer_owner_thresh: float = 5.0,    # cond2: mean owner lat > this triggers
    primer_owner_window: int = 5,        # cond2: rolling window size
    verify_mismatch_cap: int = 0,        # bail out after N consecutive verify mismatches (0=off)
    max_tokens_owner_override: Optional[int] = None,  # explicit override of max_tokens
    env_fork: bool = False,              # speculative pre-exec of mutating APIs via save/rollback
    tracer=None,                         # optional EventTracer (NullTracer default = zero overhead)
    meta_tool_library=None,              # MetaToolLibrary for pattern-based bulk-commit
    **_legacy_kwargs,  # absorbs s1_max_depth, canonicalize, confidence_threshold
) -> dict:
    """v6 actor-relay pipeline agent.

    See the module docstring for design and execution details. Set
    shell_only=True for the no-pipeline baseline
    (sequential owner only, no peer, no s1).
    """
    tracer = tracer if tracer is not None else NULL_TRACER

    # Default leaves only +2048 answer budget above the thinking budget;
    # callers hitting long-context tasks should pass `max_tokens_owner_override`
    # directly. `_completion` also halves on any 400 context-length error.
    max_tokens_owner = (max_tokens_owner_override if max_tokens_owner_override
                        else thinking_budget + 2048)

    # AppWorld's env.execute() uses signal-based timeouts which ONLY work in
    # the main Python thread. The s1 chain worker runs in a background thread
    # so its env.execute calls fail with "signal only works in main thread".
    # Disable timeouts so both threads can call env.execute safely.
    try:
        env.timeout_seconds = None
    except Exception:
        pass

    # Only owner_future + peer_future are scheduled on this pool; the s1
    # chain worker is a standalone Thread that does not use the pool.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {task_instruction}"},
    ]

    # ─── Pipeline state (protected by state_lock) ──────────────────────────
    state_lock = threading.Lock()
    # env_lock serializes env.execute() calls across main + worker threads.
    # AppWorld's env.execute enables a PROCESS-WIDE safety_guard that blocks
    # ALL open() calls; if the worker's env.execute races with the main
    # thread's logger file writes, the writes fail with PermissionError.
    env_lock = threading.Lock()
    chain: List[ChainEntry] = []          # s1 chain (depth ≤ chain_max_depth)
    committed_step = 0                    # last committed step number

    # ─── Actor futures & role URLs ─────────────────────────────────────────
    # owner_url is the GPU currently holding the authoritative (owner) role;
    # peer_url is the GPU holding the lookahead (peer) role. The assignment
    # is NOT periodic: ownership transfers on verify success and stays sticky
    # on verify failure or hard speculation boundary, so which GPU holds the
    # owner role at any given moment is fully determined by the history of
    # verify outcomes so far.
    owner_url = a1_url
    peer_url = a2_url
    auth_future: Optional[concurrent.futures.Future] = None
    peer_future: Optional[concurrent.futures.Future] = None

    # ─── Metrics ───────────────────────────────────────────────────────────
    ownership_transferred = 0        # S3 transfer count (= tier3 alias)
    peer_dropped_invalid = 0         # peer discarded on verify fail / hard boundary
    peer_dropped_stale = 0           # peer never launched in time (S3 fallback)
    verify_match = 0
    verify_mismatch = 0
    no_chain_count = 0
    no_tool_call_count = 0
    hard_boundary_mut = 0            # verify matched but s1(t) is mutating AND no fork-exec result
    hard_boundary_preexec = 0        # verify matched but s1(t) pre-exec failed
    transfer_mut = 0                 # verify matched, mutating, fork-exec result used + transfer
    meta_tool_hits = 0               # meta-tool pattern matched → bulk-commit fired
    meta_tool_steps_skipped = 0      # total steps bulk-committed via meta-tool
    semantic_match_count = 0         # exec_api_semantic judge said YES
    semantic_judge_calls = 0         # total s1 judge calls (may be costly)
    primer_launches = 0              # primer fired count
    primer_seeds = 0                 # primer actually populated chain[0]
    primer_gate_passed = 0           # gate allowed the primer
    primer_gate_blocked = 0          # gate suppressed the primer
    primer_gate_by_cond1 = 0         # cond1 (prev ro execute_api) fired
    primer_gate_by_cond2 = 0         # cond2 (long owner pred) fired
    consecutive_mismatch = 0         # run-length counter, resets on verify_match
    max_consecutive_mismatch = 0     # for diagnostics
    consecutive_no_tool_call = 0     # run-length counter, resets on any tool call
    verify_cap_triggered = False     # bailed out due to the cap
    ntc_cap_triggered = False        # bailed out due to consecutive no_tool_call
    cap_trigger_recent_paths: list = []  # last 10 step paths at cap fire
    # Repetition diagnostics (not used as a trigger — just recorded so we can
    # see whether a run got stuck in a repeat-tool loop even under the verify
    # cap).
    current_tool_streak = 0          # consecutive steps with same api_name
    current_args_streak = 0          # consecutive steps with same (api_name, args_json)
    max_tool_streak = 0
    max_args_streak = 0
    prev_tool_name: Optional[str] = None
    prev_args_key: Optional[str] = None
    chain_depth_samples = []
    step_owner_times = []
    step_env_times = []
    step_trace: List[Dict[str, Any]] = []   # per-step diagnostic trace
    # peer timing trace: (step, peer_submit_wall, auth_done_wall, head_start_s)
    peer_submit_wall: Optional[float] = None
    start_time = perf_counter()

    # ─── Spin up s1 chain worker (only if not shell_only) ──────────────────
    s1_worker: Optional[S1ChainWorker] = None
    if not shell_only:
        s1_worker = S1ChainWorker(
            env=env,
            messages_ref=messages,
            chain=chain,
            state_lock=state_lock,
            env_lock=env_lock,
            s1_url=s1_url,
            speculator_model=speculator_model,
            s1_thinking=s1_thinking,
            s1_thinking_budget=s1_thinking_budget,
            max_depth=chain_max_depth,
            max_step=max_steps,
            tracer=tracer,
            env_fork=env_fork,
        )
        s1_worker.start()

    # ─── Helper: launch owner (27B on committed messages) ─────────────────
    def _submit_owner() -> concurrent.futures.Future:
        # Main loop only appends to `messages`, so a shallow snapshot is
        # safe for the executor thread.
        snap = list(messages)
        tracer.emit("owner", "llm_submit",
                    url=owner_url,
                    prefix_len=len(snap),
                    prefix_hash=messages_fingerprint(snap),
                    step_idx=committed_step + 1)
        return executor.submit(
            _completion, snap, actor_model, owner_url,
            TOOLS, True, thinking_budget, max_tokens_owner)

    # ─── Helper: launch peer (27B on spec ctx using s1's chain[0]) ────────
    def _submit_peer_from_s1() -> Optional[concurrent.futures.Future]:
        """Build spec ctx from current chain[0] and submit peer on peer_url.

        Returns None if chain is empty or chain[0] is a hard speculation
        boundary. Chain read + submit happen under state_lock so rollbacks
        can't interleave.
        """
        nonlocal peer_submit_wall
        with state_lock:
            if not chain:
                return None
            entry = chain[0]
            if entry.is_hard_boundary:
                return None
            spec_msgs = list(messages)
            a, t = _build_canonical_message(
                committed_step + 1,
                entry.action_name, entry.action_args, entry.tool_result)
            spec_msgs.append(a)
            spec_msgs.append(t)
        peer_submit_wall = perf_counter()
        tracer.emit("peer", "llm_submit",
                    url=peer_url,
                    prefix_len=len(spec_msgs),
                    prefix_hash=messages_fingerprint(spec_msgs),
                    step_idx=committed_step + 1,
                    s1_action=entry.action_name)
        return executor.submit(
            _completion, spec_msgs, actor_model, peer_url,
            TOOLS, True, thinking_budget, max_tokens_owner)

    # ─── Helper: run the authoritative action against the real env ────────
    def _exec_real(name: str, args: dict) -> tuple:
        """Execute (name, args) under env_lock and return (tool_result, t_env).

        Shared among the three commit branches that must hit the real env:
        S3-SEMANTIC (chain0.tool_result reflects s1's wrong args), hard
        speculation boundary (no spec_env(t) was ever produced), and
        failure-sticky (verify mismatched). The exact-match S3-transfer
        branch reuses chain0.tool_result instead and does NOT call this.
        """
        t0 = perf_counter()
        wait_span = tracer.span("env", "env_lock_wait", who="main", tool=name)
        wait_span.__enter__()
        env_lock.acquire()
        wait_span.__exit__(None, None, None)
        try:
            with tracer.span("env", "env_exec", who="main", tool=name):
                try:
                    r = execute_tool(env, name, args)
                except Exception as e:
                    r = f"Error: {e}"
        finally:
            env_lock.release()
        return r, perf_counter() - t0

    # ─── Helper: primer thread — one-shot s1 rollout to seed chain[0] ────
    # (A1) The worker may be in the middle of a now-stale 4B call when a
    # reset fires, so its fresh rollout is delayed by ~0.7s of wasted work.
    # The primer runs in parallel: it produces chain[0] from the true
    # committed prefix as soon as s1 can return an action + pre-exec result,
    # without waiting for the worker's current iteration to drain.
    primer_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
    if reset_primer and not shell_only:
        primer_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    primer_max_tokens = (s1_thinking_budget + 1024) if s1_thinking else 1024

    # ─── Primer gate ───────────────────────────────────────────────────────
    # Gate semantics:
    #   "uniform"  → fire on every reset
    #   "targeted" → fire only when (cond1 OR cond2) is true
    #   "cond1"    → fire only when the previous committed action was a
    #               read-only execute_api
    #   "cond2"    → fire only when the rolling mean of the last N owner
    #               waits exceeds `primer_owner_thresh`
    def _prev_is_ro_execute_api() -> bool:
        """Walk backwards in messages to find the last assistant tool_call
        and check whether it was a read-only execute_api."""
        with state_lock:
            for msg in reversed(messages):
                if msg.get("role") != "assistant":
                    continue
                tcs = msg.get("tool_calls") or []
                if not tcs:
                    continue
                fn = tcs[0].get("function", {})
                if fn.get("name") != "execute_api":
                    return False
                raw = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except Exception:
                    return False
                app = args.get("app_name", "")
                api = args.get("api_name", "")
                if not app or not api:
                    return False
                return is_read_only(f"{app}.{api}")
        return False

    def _predicted_owner_is_long() -> bool:
        """Rolling mean of the last `primer_owner_window` owner waits.
        Returns True only once we have enough samples AND the mean exceeds
        `primer_owner_thresh`. Cold-start returns False to avoid wasted
        primer launches before we have any signal."""
        if len(step_owner_times) < primer_owner_window:
            return False
        tail = step_owner_times[-primer_owner_window:]
        return (sum(tail) / len(tail)) > primer_owner_thresh

    def _should_fire_primer() -> bool:
        nonlocal primer_gate_passed, primer_gate_blocked
        nonlocal primer_gate_by_cond1, primer_gate_by_cond2
        if primer_gate == "uniform":
            primer_gate_passed += 1
            return True
        c1 = _prev_is_ro_execute_api()
        c2 = _predicted_owner_is_long()
        if primer_gate == "cond1":
            passed = c1
        elif primer_gate == "cond2":
            passed = c2
        else:  # "targeted"
            passed = c1 or c2
        if passed:
            primer_gate_passed += 1
            if c1:
                primer_gate_by_cond1 += 1
            if c2:
                primer_gate_by_cond2 += 1
        else:
            primer_gate_blocked += 1
        return passed

    def _kickstart_primer():
        """Fire a single 4B rollout and append to chain if still relevant.

        Guards: re-checks committed_step + s1_worker.epoch before every
        mutation so a race with the worker or another reset drops the result
        cleanly.
        """
        nonlocal primer_seeds
        with state_lock:
            snap_msgs = list(messages)
            snap_committed = committed_step
            snap_epoch = s1_worker.epoch if s1_worker is not None else 0
            if chain:
                return  # worker already filled it — nothing to do
        try:
            resp = _completion(
                snap_msgs, speculator_model, s1_url, TOOLS,
                s1_thinking,
                s1_thinking_budget if s1_thinking else 0,
                primer_max_tokens)
            name, args, _ = _parse_tool_call(resp.choices[0].message)
        except Exception:
            return
        if not name:
            return
        mutating = is_mutating(_action_norm(name, args))
        terminal = (name == "complete_task")
        tool_result = None
        preexec_failed = False
        if not mutating and not terminal:
            with state_lock:
                if (committed_step != snap_committed
                        or (s1_worker is not None
                            and s1_worker.epoch != snap_epoch)):
                    return
            try:
                with env_lock:
                    tool_result = execute_tool(env, name, args)
            except Exception:
                preexec_failed = True
        entry = ChainEntry(
            action_name=name, action_args=args, tool_result=tool_result,
            is_mutating=(mutating or terminal),
            preexec_failed=preexec_failed,
        )
        with state_lock:
            if (committed_step != snap_committed
                    or (s1_worker is not None
                        and s1_worker.epoch != snap_epoch)):
                return
            if chain:
                return  # worker beat us — drop primer result
            chain.append(entry)
            primer_seeds += 1

    # ─── Helper: reset chain only (keeps peer_future alive) ───────────────
    def _reset_chain_only(reason: str = ""):
        """Clear chain, bump s1 worker epoch, kickstart the primer.

        Used when the entire speculative suffix is dead but peer_future
        must survive — specifically S3-SEMANTIC, where the semantic judge
        certified the peer's prefix even though chain[1..] (conditioned on
        s1's wrong args) must be rebuilt from the true prefix.
        """
        nonlocal primer_launches
        with state_lock:
            chain.clear()
            if s1_worker is not None:
                s1_worker.signal_rollback()
        if primer_pool is not None and _should_fire_primer():
            primer_pool.submit(_kickstart_primer)
            primer_launches += 1
        if reason:
            print(f"  [reset] {reason}")

    # ─── Helper: full spec state reset (Inv 3) ─────────────────────────────
    def _reset_spec_state(reason: str = ""):
        """Cancel in-flight peer + reset_chain_only.

        Called whenever verify fails or a hard speculation boundary forces
        failure-sticky: the entire speculative suffix from step t is dead
        AND the peer_future (if any) is logically dead too.
        """
        nonlocal peer_future
        if peer_future is not None:
            peer_future.cancel()
            peer_future = None
        _reset_chain_only(reason)

    # ─── S0: Bootstrap ─────────────────────────────────────────────────────
    auth_future = _submit_owner()
    task_completed = False

    # ─── Main loop: S1 WAIT → S2 VERIFY → S3 TRANSFER / S4 STICKY ──────────
    while committed_step < max_steps:
        # ── S1 Phase A: poll auth_future, eager peer launch ──
        # While owner is still running, launch peer as soon as s1(t) becomes
        # available. This is the critical v5→v6 fix: peer gets a ~0.7s head
        # start instead of being launched only AFTER the owner commits, so
        # verify-success can rename peer→new owner with ~0.7s remaining on
        # its call instead of ~2s. wait(timeout=0.02) is cheap (status check
        # + one lock in _submit_peer_from_s1), ~100 polls per owner wait.
        t_wait_start = perf_counter()
        phase_a_span = tracer.span("main", "phase_a_wait",
                                   wait_reason="future_not_ready",
                                   step_idx=committed_step + 1)
        phase_a_span.__enter__()
        while True:
            done, _ = concurrent.futures.wait([auth_future], timeout=0.02)
            if peer_future is None:
                peer_future = _submit_peer_from_s1()
            if done:
                break
        phase_a_span.__exit__(None, None, None)

        try:
            auth_resp = auth_future.result()
        except Exception as e:
            print(f"  [error] owner failed: {e}")
            break
        t_auth_done = perf_counter()
        owner_wait = t_auth_done - t_wait_start
        tracer.emit("owner", "llm_end",
                    step_idx=committed_step + 1,
                    owner_wait_s=round(owner_wait, 4))

        # ── S1 Phase B: 0.5s grace window for s1(t) ──
        s1_deadline = t_auth_done + 0.5
        phase_b_span = tracer.span("main", "phase_b_wait",
                                   wait_reason="chain_empty",
                                   step_idx=committed_step + 1)
        phase_b_span.__enter__()
        while not chain and perf_counter() < s1_deadline:
            if peer_future is None:
                peer_future = _submit_peer_from_s1()
            time.sleep(0.01)
        phase_b_span.__exit__(None, None, None)

        step_owner_times.append(owner_wait)

        # Pause s1 chain worker so we can read state atomically
        if s1_worker is not None:
            s1_worker.pause()

        # Parse owner response
        auth_name, auth_args, auth_msg = _parse_tool_call(
            auth_resp.choices[0].message)

        # ── No tool call: error feedback (NOT silent continue) ──
        if not auth_name:
            no_tool_call_count += 1
            consecutive_no_tool_call += 1
            tracer.emit("main", "no_tool_call_detected",
                        step_idx=committed_step + 1)
            content = auth_msg.get("content") or ""
            # Get the canonical "Unknown tool" string the same way baseline
            # does (via execute_tool) so the injected messages are bit-
            # identical between the two arms — divergence here forks the
            # trajectory under greedy decoding.
            if not content.strip():
                with env_lock:
                    ntc_result = execute_tool(env, "_no_tool_call", {})
                ntc_asst, ntc_tool = _build_canonical_message(
                    committed_step + 1, "_no_tool_call", {}, ntc_result)
            with state_lock:
                if content.strip():
                    messages.append({"role": "assistant", "content": content})
                else:
                    messages.append(ntc_asst)
                    messages.append(ntc_tool)
                committed_step += 1
                if s1_worker is not None:
                    s1_worker.update_committed(committed_step)
            _reset_spec_state(reason="no_tool_call")
            tracer.emit("main", "no_tool_call_injected",
                        step_idx=committed_step,
                        tc_id=f"call_{committed_step}")
            print(f"  step {committed_step}: [no_tool_call] error feedback")

            if consecutive_no_tool_call >= 5:
                ntc_cap_triggered = True
                print(f"  [cap] consecutive no_tool_call cap (5) reached — bailing out")
                break

            if committed_step < max_steps:
                auth_future = _submit_owner()
                tracer.emit("main", "no_tool_call_retry_submit",
                            step_idx=committed_step + 1)
                if s1_worker is not None:
                    s1_worker.resume()
            continue

        consecutive_no_tool_call = 0  # got a real tool call

        # ── S2: snapshot chain[0], verify ──
        with state_lock:
            chain_snapshot = list(chain)
            chain_depth_samples.append(len(chain_snapshot))
        chain0 = chain_snapshot[0] if chain_snapshot else None
        cannot_transfer = chain0 is not None and chain0.is_hard_boundary
        tracer.emit("verify", "verify_start",
                    step_idx=committed_step + 1,
                    chain_depth=len(chain_snapshot),
                    chain0_is_hard_boundary=cannot_transfer,
                    auth_tool=auth_name)

        exact_verified = False
        semantic_verified = False
        if chain0 is not None:
            exact_verified = _verify_action_match(
                auth_name, auth_args,
                chain0.action_name, chain0.action_args)
            # A2 narrow: only attempt semantic verify when exact failed,
            # chain0 is not a hard boundary (no peer to transfer anyway),
            # and both actions are execute_api with matching app+api.
            if (not exact_verified
                    and verify_mode == "exec_api_semantic"
                    and not cannot_transfer
                    and auth_name == "execute_api"
                    and chain0.action_name == "execute_api"):
                semantic_judge_calls += 1
                semantic_verified = _exec_api_semantic_match(
                    auth_name, auth_args,
                    chain0.action_name, chain0.action_args,
                    s1_url, speculator_model)
                if semantic_verified:
                    semantic_match_count += 1
            elif not exact_verified and verify_mode == "semantic":
                semantic_verified = _semantic_match(
                    auth_name, auth_args,
                    chain0.action_name, chain0.action_args,
                    s1_url, speculator_model)
                if semantic_verified:
                    semantic_match_count += 1
        verified = exact_verified or semantic_verified

        committed_step += 1
        action_norm = _action_norm(auth_name, auth_args)

        # ── S3 SUCCESS_TRANSFER vs hard-boundary vs S4 FAILURE_STICKY ──
        tracer.emit("verify", "verify_end",
                    step_idx=committed_step,
                    exact=exact_verified, semantic=semantic_verified,
                    cannot_transfer=cannot_transfer)
        if exact_verified and not cannot_transfer:
            # ── S3: SUCCESS_TRANSFER ──
            verify_match += 1
            consecutive_mismatch = 0
            if chain0.is_mutating:
                tool_result, t_env = _exec_real(auth_name, auth_args)
                commit_path = "tier1+transfer_mut"
                transfer_mut += 1
            else:
                tool_result = chain0.tool_result
                t_env = 0.0
                commit_path = "tier1+transfer"
            tracer.emit("verify", "transfer_commit", step_idx=committed_step,
                        mutating=chain0.is_mutating)

            with state_lock:
                if chain and chain[0] is chain0:
                    chain.pop(0)

        elif semantic_verified and not cannot_transfer:
            # ── S3-SEMANTIC: correctness guard — params differed, so DO
            # NOT reuse chain0.tool_result (it reflects s1's args). Run
            # owner's action against the real env for ground truth. Peer
            # (spec_ctx used s1's canonical action) is kept for ownership
            # transfer since the judge certified the post-state as
            # equivalent; but chain[1..] was conditioned on s1's args AND
            # pre-exec result, so rebuild it from the true prefix. We use
            # `_reset_chain_only` (not `_reset_spec_state`) deliberately:
            # calling the full reset would cancel peer_future and block
            # the transfer a few lines below.
            tool_result, t_env = _exec_real(auth_name, auth_args)
            verify_match += 1
            consecutive_mismatch = 0
            commit_path = "tier1+semantic"
            tracer.emit("verify", "semantic_commit", step_idx=committed_step)
            _reset_chain_only("tier1+semantic")

        elif verified and cannot_transfer:
            # Hard speculation boundary: verify matched, but chain[0] was
            # either mutating (never pre-executed) or its pre-exec raised,
            # so there is no spec_env(t) to reuse. Run the owner's action
            # against the real env exactly once — no double-execution risk.
            tool_result, t_env = _exec_real(auth_name, auth_args)
            verify_match += 1
            consecutive_mismatch = 0
            tracer.emit("verify", "hard_boundary_commit",
                        step_idx=committed_step,
                        is_mutating=chain0.is_mutating,
                        preexec_failed=chain0.preexec_failed)
            if chain0.is_mutating:
                hard_boundary_mut += 1
                commit_path = "tier1+exec_mut"
            else:
                hard_boundary_preexec += 1
                commit_path = "tier1+exec_preexec_fail"

        else:
            # ── S4: FAILURE_STICKY (verify mismatch OR no chain[0]) ──
            if chain0 is None:
                no_chain_count += 1
                commit_path = "stall+no_chain"
                # no_chain is an upstream stall, not a semantic divergence —
                # do not bump the consecutive mismatch counter
            else:
                verify_mismatch += 1
                consecutive_mismatch += 1
                if consecutive_mismatch > max_consecutive_mismatch:
                    max_consecutive_mismatch = consecutive_mismatch
                commit_path = "stall+mismatch"
            tracer.emit("verify", "failure_sticky_commit",
                        step_idx=committed_step,
                        reason=commit_path)
            tool_result, t_env = _exec_real(auth_name, auth_args)

        step_env_times.append(t_env)

        # Per-step trace: peer head-start = (auth_done - peer_submit) if
        # peer was launched before auth returned, else 0. Negative would
        # mean peer was submitted AFTER auth returned (Phase B submit).
        head_start = (t_auth_done - peer_submit_wall) if peer_submit_wall else 0.0
        # Update repetition streaks (recorded only, not used as a trigger).
        try:
            args_key = json.dumps(auth_args or {}, sort_keys=True)
        except Exception:
            args_key = str(auth_args)
        tool_same = (auth_name == prev_tool_name)
        args_same = tool_same and (args_key == prev_args_key)
        current_tool_streak = current_tool_streak + 1 if tool_same else 1
        current_args_streak = current_args_streak + 1 if args_same else 1
        prev_tool_name = auth_name
        prev_args_key = args_key
        if current_tool_streak > max_tool_streak:
            max_tool_streak = current_tool_streak
        if current_args_streak > max_args_streak:
            max_args_streak = current_args_streak

        step_trace.append({
            "step": committed_step,
            "path": commit_path,
            "tool": auth_name,
            "args_key": args_key[:120],
            "owner_wait": round(owner_wait, 3),
            "env_time": round(t_env, 3),
            "chain_depth": len(chain_snapshot),
            "peer_head_start": round(head_start, 3),
            "peer_submitted": peer_submit_wall is not None,
            "tool_streak": current_tool_streak,
            "args_streak": current_args_streak,
            "consec_mismatch": consecutive_mismatch,
        })
        peer_submit_wall = None  # reset for next step

        # Commit canonical messages (shared by all three branches)
        canon_asst, canon_tool = _build_canonical_message(
            committed_step, auth_name, auth_args, tool_result)
        with state_lock:
            messages.append(canon_asst)
            messages.append(canon_tool)
            if s1_worker is not None:
                s1_worker.update_committed(committed_step)
        tracer.emit("main", "commit",
                    step_idx=committed_step,
                    commit_path=commit_path,
                    tool=auth_name,
                    t_env_s=round(t_env, 4))

        if logger:
            # env_lock prevents safety_guard (enabled by worker env.execute)
            # from blocking logger._save()'s file open
            with env_lock:
                logger.log_step(
                    step_id=committed_step,
                    instruction=task_instruction[:500],
                    model_output=json.dumps(
                        {"tool": auth_name, "args": auth_args})[:2000],
                    api_name=auth_name, api_params=auth_args,
                    api_result=tool_result[:2000] if isinstance(tool_result, str)
                                else str(tool_result)[:2000],
                    wall_clock_time=owner_wait,
                    input_tokens=0, output_tokens=0,
                    tier=commit_path,
                )

        print(f"  step {committed_step}: [{commit_path}] {action_norm}"
              f" owner={owner_wait:.2f}s env={t_env:.2f}s"
              f" chain_d={len(chain_snapshot)}")

        # Terminal action?
        if auth_name == "complete_task":
            task_completed = True
            break

        if committed_step >= max_steps:
            break

        # Bail out if we've hit `verify_mismatch_cap` consecutive verify
        # mismatches. A runaway mismatch loop inflates the committed
        # messages list toward the context window and the agent is clearly
        # not making progress — cut it off before it crashes.
        if (verify_mismatch_cap > 0
                and consecutive_mismatch >= verify_mismatch_cap):
            verify_cap_triggered = True
            cap_trigger_recent_paths = [
                {"step": s["step"], "path": s["path"], "tool": s["tool"],
                 "args_key": s.get("args_key", "")[:80],
                 "tool_streak": s.get("tool_streak", 0),
                 "args_streak": s.get("args_streak", 0)}
                for s in step_trace[-10:]
            ]
            print(f"  [cap] consecutive verify_mismatch cap "
                  f"({verify_mismatch_cap}) reached — bailing out")
            print(f"  [cap] last paths: "
                  f"{[(s['step'], s['path'], s['tool']) for s in cap_trigger_recent_paths]}")
            break

        # ── Meta-tool matching (exact S3 transfer only) ──────────────────
        meta_tool_fired = False
        if (meta_tool_library is not None
                and exact_verified and not cannot_transfer):
            with state_lock:
                mt_remaining = list(chain)
            if mt_remaining:
                mt_norms = [_action_norm(e.action_name, e.action_args)
                            for e in mt_remaining]
                mt_match = meta_tool_library.match(mt_norms)
                if mt_match is not None:
                    mt_depth = min(len(mt_match.action_types),
                                   len(mt_remaining))
                    mt_actual = 0
                    for i in range(mt_depth):
                        e = mt_remaining[i]
                        if e.is_hard_boundary or e.action_name == "complete_task":
                            break
                        mt_actual = i + 1
                    if mt_actual > 0:
                        meta_tool_fired = True
                        meta_tool_hits += 1
                        for i in range(mt_actual):
                            e = mt_remaining[i]
                            committed_step += 1
                            if e.is_mutating:
                                mt_result, mt_t_env = _exec_real(
                                    e.action_name, e.action_args)
                                transfer_mut += 1
                            else:
                                mt_result = e.tool_result
                                mt_t_env = 0.0
                            mt_norm = mt_norms[i]
                            canon_a, canon_t = _build_canonical_message(
                                committed_step, e.action_name, e.action_args,
                                mt_result)
                            with state_lock:
                                messages.append(canon_a)
                                messages.append(canon_t)
                                if chain and chain[0] is e:
                                    chain.pop(0)
                                if s1_worker is not None:
                                    s1_worker.update_committed(committed_step)
                            step_env_times.append(mt_t_env)
                            try:
                                mt_args_key = json.dumps(
                                    e.action_args or {}, sort_keys=True)
                            except Exception:
                                mt_args_key = str(e.action_args)
                            tool_same = (e.action_name == prev_tool_name)
                            args_same = tool_same and mt_args_key == prev_args_key
                            current_tool_streak = (
                                current_tool_streak + 1 if tool_same else 1)
                            current_args_streak = (
                                current_args_streak + 1 if args_same else 1)
                            prev_tool_name = e.action_name
                            prev_args_key = mt_args_key
                            if current_tool_streak > max_tool_streak:
                                max_tool_streak = current_tool_streak
                            if current_args_streak > max_args_streak:
                                max_args_streak = current_args_streak
                            step_trace.append({
                                "step": committed_step,
                                "path": "meta_tool_skip",
                                "tool": e.action_name,
                                "args_key": mt_args_key[:120],
                                "owner_wait": 0.0,
                                "env_time": round(mt_t_env, 3),
                                "chain_depth": len(mt_remaining) - i - 1,
                                "peer_head_start": 0.0,
                                "peer_submitted": False,
                                "tool_streak": current_tool_streak,
                                "args_streak": current_args_streak,
                                "consec_mismatch": 0,
                            })
                            if logger:
                                with env_lock:
                                    logger.log_step(
                                        step_id=committed_step,
                                        instruction=task_instruction[:500],
                                        model_output=json.dumps({
                                            "tool": e.action_name,
                                            "args": e.action_args})[:2000],
                                        api_name=e.action_name,
                                        api_params=e.action_args,
                                        api_result=(mt_result[:2000]
                                                    if isinstance(mt_result, str)
                                                    else str(mt_result)[:2000]),
                                        wall_clock_time=0.0,
                                        input_tokens=0, output_tokens=0,
                                        tier="meta_tool_skip",
                                    )
                            tracer.emit("main", "commit",
                                        step_idx=committed_step,
                                        commit_path="meta_tool_skip",
                                        tool=e.action_name,
                                        t_env_s=round(mt_t_env, 4))
                            print(f"  step {committed_step}: [meta_tool_skip] "
                                  f"{mt_norm} env={mt_t_env:.2f}s "
                                  f"(pattern={mt_match.pattern_id})")
                            if committed_step >= max_steps:
                                break
                        meta_tool_steps_skipped += i + 1
                        if peer_future is not None:
                            peer_dropped_stale += 1
                            peer_future = None
                        auth_future = _submit_owner()
                        peer_future = _submit_peer_from_s1()

        if meta_tool_fired:
            if task_completed or committed_step >= max_steps:
                break
        elif verified and not cannot_transfer:
            # S3: ownership transfer — rename running peer to new owner.
            if peer_future is not None:
                ownership_transferred += 1
                owner_url, peer_url = peer_url, owner_url
                auth_future = peer_future
                peer_future = _submit_peer_from_s1()
            else:
                # Peer wasn't launched in time (s1 filled chain[0] only
                # during Phase B). Fall back to a fresh sticky owner call.
                peer_dropped_stale += 1
                auth_future = _submit_owner()
                peer_future = _submit_peer_from_s1()
        else:
            # Hard boundary OR failure-sticky: invalidate entire spec
            # suffix (Inv 3) and keep owner role sticky (Inv 4).
            if peer_future is not None:
                peer_dropped_invalid += 1
            _reset_spec_state(
                reason=("hard_boundary" if (verified and cannot_transfer)
                        else commit_path))
            auth_future = _submit_owner()

        if s1_worker is not None:
            s1_worker.resume()

    # ─── Shutdown ──────────────────────────────────────────────────────────
    wall_time = perf_counter() - start_time

    if s1_worker is not None:
        s1_worker.stop()
        # Wait briefly for worker to finish its current iteration so any
        # in-flight env.execute completes (releases safety_guard globally).
        s1_worker.join(timeout=5.0)

    if primer_pool is not None:
        primer_pool.shutdown(wait=True, cancel_futures=True)

    if peer_future is not None:
        peer_future.cancel()

    # ─── Eval ──────────────────────────────────────────────────────────────
    try:
        # Hold env_lock to be safe — worker should already be done but
        # env.evaluate may touch files that safety_guard would block.
        with env_lock:
            eval_result = env.evaluate(suppress_errors=True)
        reward = eval_result.pass_percentage / 100.0
        print(f"\n  Eval: success={eval_result.success} "
              f"passed={eval_result.pass_count}/{eval_result.total_count}")
    except Exception as e:
        print(f"\n  Eval error: {e}")
        reward = 0.0

    if logger:
        with env_lock:
            logger.finalize(task_success=(reward > 0.99), evaluation_score=reward)

    # ─── Metrics aggregation ───────────────────────────────────────────────
    avg_owner = sum(step_owner_times) / max(len(step_owner_times), 1)
    avg_env = sum(step_env_times) / max(len(step_env_times), 1)
    avg_chain_depth = (sum(chain_depth_samples) / max(len(chain_depth_samples), 1)
                       if chain_depth_samples else 0.0)
    max_chain_depth = max(chain_depth_samples) if chain_depth_samples else 0

    s1_total = s1_worker.s1_total_time if s1_worker else 0.0
    s1_calls = s1_worker.s1_call_count if s1_worker else 0
    pre_exec_total = s1_worker.preexec_total if s1_worker else 0
    pre_exec_wasted = s1_worker.preexec_wasted if s1_worker else 0
    fork_exec_total = s1_worker.fork_exec_total if s1_worker else 0
    fork_exec_wasted = s1_worker.fork_exec_wasted if s1_worker else 0

    seq_est = committed_step * (avg_owner + avg_env)
    wall_su = seq_est / max(wall_time, 0.1)

    # Legacy tier_counts for run_pipeline.py backward compat:
    #   tier1 = verify match (any commit path with s1 prediction matching)
    #   tier3 = true ownership transfer count
    #   stall = verify miss + no chain + no tool call
    tier_counts = {
        "tier1": verify_match,
        "tier3": ownership_transferred,
        "stall": verify_mismatch + no_chain_count + no_tool_call_count,
    }

    print(f"  Tiers: {tier_counts}")
    print(f"  Verify: match={verify_match} mismatch={verify_mismatch}")
    print(f"  Peer: transferred={ownership_transferred} "
          f"drop_invalid={peer_dropped_invalid} drop_stale={peer_dropped_stale}")
    print(f"  Chain: avg_depth={avg_chain_depth:.2f} pre_exec={pre_exec_total} "
          f"wasted={pre_exec_wasted}")
    if verify_mode != "exact":
        print(f"  Semantic: judge_calls={semantic_judge_calls} "
              f"matched={semantic_match_count}")
    if reset_primer:
        print(f"  Primer: launches={primer_launches} seeds={primer_seeds} "
              f"gate={primer_gate} passed={primer_gate_passed} "
              f"blocked={primer_gate_blocked} c1={primer_gate_by_cond1} "
              f"c2={primer_gate_by_cond2}")
    if meta_tool_library is not None:
        print(f"  MetaTool: hits={meta_tool_hits} "
              f"steps_skipped={meta_tool_steps_skipped}")
    print(f"  Wall: {wall_time:.0f}s  speedup={wall_su:.2f}x  "
          f"owner_avg={avg_owner:.2f}s  env_avg={avg_env:.3f}s")

    executor.shutdown(wait=False)

    return {
        "reward": reward,
        "n_steps": committed_step,       # total committed (env + control)
        "env_steps": len(step_trace),    # real-action steps only (excludes no_tool_call retries)
        "task_completed": task_completed,
        "tier_counts": tier_counts,
        "pipeline_hits": ownership_transferred,  # legacy alias for tier3
        "wall_speedup": round(wall_su, 3),
        "wall_time": round(wall_time, 1),
        "verify_match": verify_match,
        "verify_mismatch": verify_mismatch,
        "ownership_transferred": ownership_transferred,
        "peer_dropped_invalid": peer_dropped_invalid,
        "peer_dropped_stale": peer_dropped_stale,
        "no_chain_count": no_chain_count,
        "no_tool_call_count": no_tool_call_count,
        "transfer_mut": transfer_mut,
        "meta_tool_hits": meta_tool_hits,
        "meta_tool_steps_skipped": meta_tool_steps_skipped,
        "hard_boundary_mut": hard_boundary_mut,
        "hard_boundary_preexec": hard_boundary_preexec,
        "semantic_match_count": semantic_match_count,
        "semantic_judge_calls": semantic_judge_calls,
        "primer_launches": primer_launches,
        "primer_seeds": primer_seeds,
        "primer_gate": primer_gate,
        "primer_gate_passed": primer_gate_passed,
        "primer_gate_blocked": primer_gate_blocked,
        "primer_gate_by_cond1": primer_gate_by_cond1,
        "primer_gate_by_cond2": primer_gate_by_cond2,
        "primer_owner_thresh": primer_owner_thresh,
        "primer_owner_window": primer_owner_window,
        "verify_mismatch_cap": verify_mismatch_cap,
        "max_consecutive_mismatch": max_consecutive_mismatch,
        "verify_cap_triggered": verify_cap_triggered,
        "ntc_cap_triggered": ntc_cap_triggered,
        "cap_trigger_recent_paths": cap_trigger_recent_paths,
        "max_tool_streak": max_tool_streak,
        "max_args_streak": max_args_streak,
        "chain_pre_exec_total": pre_exec_total,
        "chain_pre_exec_wasted": pre_exec_wasted,
        "chain_fork_exec_total": fork_exec_total,
        "chain_fork_exec_wasted": fork_exec_wasted,
        "env_fork": env_fork,
        "chain_avg_depth": round(avg_chain_depth, 2),
        "chain_depth_max": max_chain_depth,
        "s1_total_time": round(s1_total, 2),
        "s1_call_count": s1_calls,
        "verify_mode": verify_mode,
        "chain_max_depth": chain_max_depth,
        "reset_primer": reset_primer,
        "step_trace": step_trace,
        "messages": messages,
    }
