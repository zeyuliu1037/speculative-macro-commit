"""Event-level tracer for baseline + pipeline agent runs.

Design goals:
  1. Zero-overhead when disabled (NullTracer is a no-op singleton)
  2. Monotonic timestamps via perf_counter (same source as pipeline_agent)
  3. Thread-safe append to a per-sample JSONL file
  4. Logical-lane-aware: caller tags each event with a `component` string
     that maps to a Chrome Trace pid/tid, not an OS thread (thread pools
     recycle workers, which would scramble the Gantt lanes)
  5. Supports B/E pairs (span), instant events, and complete X events
  6. Records prefix fingerprint on llm_submit so we can later verify that
     two submissions really saw the same prompt

Output schema (one JSONL line per event):
    {
      "sample_id": "fd1f8fa_1",
      "mode": "baseline" | "pipeline_a0",
      "ts_s": 12.345678,            # seconds since task start (T=0)
      "ts_us": 12345678.0,          # microseconds (Chrome Trace native)
      "thread": "MainThread",
      "component": "owner" | "peer" | "s1_worker" | "primer" | "main" | "env" | "verify",
      "event_type": "llm_submit",   # see vocabulary below
      "phase": "B" | "E" | "i" | "X",  # Chrome Trace phase tag
      "step_idx": 12,                  # committed step counter (if known)
      "future_id": "owner:1700...",    # unique id of a future (if any)
      "prefix_hash": "abc123",         # short fingerprint of messages
      "prefix_len": 42,                # count of messages in the submitted list
      "wait_reason": "future_not_ready", # why a wait is happening
      "extra": {...}                    # catch-all for path-specific info
    }

Event vocabulary (canonical names):
  component=main:
    trace_begin, trace_end
    phase_a_wait         (main loop polling for auth_future + eager peer launch)
    phase_b_wait         (main loop grace wait for s1(t) after auth returns)
    no_tool_call_detected, no_tool_call_injected, no_tool_call_retry_submit
    commit               (message append block for a committed step)
  component=owner / peer:
    llm_submit           (executor.submit)
    llm_end              (auth_future.result() returned)
    llm_failed
  component=s1_worker:
    iter_start           (worker began a new chain rollout iteration)
    llm_call             (s1 4B completion call, span)
    preexec              (read-only env execute under env_lock, span)
    chain_append         (entry appended under state_lock)
    stale_discarded      (result dropped due to epoch / committed mismatch)
  component=primer:
    llm_call, chain_append, stale_discarded
  component=env:
    env_lock_wait        (time main/peer/worker spent waiting for env_lock)
    env_exec             (time inside env.execute for a real commit)
  component=verify:
    verify               (verify_action_match span)
    transfer_commit      (S3 ownership transfer happened)
    hard_boundary_commit (S3-alt: exec_mut or preexec_fail)
    failure_sticky_commit (S4: verify mismatched or no chain[0])
    semantic_commit      (S3-semantic)
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Any, Dict, Optional

from .agent import perf_counter  # monotonic, freezegun-immune


# ─── Fingerprint helper ────────────────────────────────────────────────────

def messages_fingerprint(messages) -> str:
    """Stable short hash of a message list.

    Used on llm_submit events to verify that owner and peer really see the
    same committed prefix, and that a baseline/pipeline pair at step t
    produced the same prompt.
    """
    try:
        s = json.dumps(messages, sort_keys=True, default=str)
    except Exception:
        s = repr(messages)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# ─── NullTracer: zero-overhead default ─────────────────────────────────────

class _NullSpan:
    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, *exc) -> None:
        pass

    def set(self, **_kwargs) -> None:
        pass


class NullTracer:
    """No-op tracer. All methods are cheap and return immediately."""

    enabled = False

    def emit(self, *_args, **_kwargs) -> None:
        return None

    def span(self, *_args, **_kwargs) -> _NullSpan:
        return _NullSpan()

    def close(self) -> None:
        return None


NULL_TRACER = NullTracer()


# ─── EventTracer: real backend ──────────────────────────────────────────────

class _Span:
    def __init__(self, tracer: "EventTracer", component: str, event_type: str,
                 extra: Dict[str, Any]):
        self._tracer = tracer
        self._component = component
        self._event_type = event_type
        self._extra = dict(extra)
        self._start_us: Optional[float] = None

    def __enter__(self) -> "_Span":
        self._start_us = self._tracer._now_us()
        self._tracer._write(
            component=self._component,
            event_type=self._event_type,
            phase="B",
            ts_us_override=self._start_us,
            **self._extra,
        )
        return self

    def __exit__(self, *exc) -> None:
        self._tracer._write(
            component=self._component,
            event_type=self._event_type,
            phase="E",
            **self._extra,
        )

    def set(self, **kwargs) -> None:
        """Update extra fields on the span (reflected in the E event)."""
        self._extra.update(kwargs)


class EventTracer:
    """Thread-safe append-only event logger for one sample run."""

    enabled = True

    def __init__(self, out_path: str, sample_id: str, mode: str):
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                    exist_ok=True)
        self.out_path = out_path
        self.sample_id = sample_id
        self.mode = mode
        self._origin = perf_counter()
        self._lock = threading.Lock()
        self._fh = open(out_path, "w", buffering=1)
        self.emit("main", "trace_begin", origin_perf_counter=self._origin)

    def _now_us(self) -> float:
        return (perf_counter() - self._origin) * 1e6

    def _write(self, component: str, event_type: str, phase: str = "i",
               ts_us_override: Optional[float] = None,
               **extra) -> None:
        ts_us = ts_us_override if ts_us_override is not None else self._now_us()
        rec = {
            "sample_id": self.sample_id,
            "mode": self.mode,
            "ts_s": ts_us / 1e6,
            "ts_us": ts_us,
            "thread": threading.current_thread().name,
            "component": component,
            "event_type": event_type,
            "phase": phase,
        }
        for k, v in extra.items():
            if v is None:
                continue
            rec[k] = v
        line = json.dumps(rec, default=str)
        with self._lock:
            self._fh.write(line + "\n")

    def emit(self, component: str, event_type: str, phase: str = "i",
             **extra) -> None:
        self._write(component=component, event_type=event_type, phase=phase,
                    **extra)

    def span(self, component: str, event_type: str, **extra) -> _Span:
        return _Span(self, component, event_type, extra)

    def close(self) -> None:
        try:
            self.emit("main", "trace_end")
        finally:
            with self._lock:
                try:
                    self._fh.close()
                except Exception:
                    pass


def make_tracer(out_path: Optional[str], sample_id: str,
                mode: str):
    """Factory: returns EventTracer if out_path is provided, NullTracer
    otherwise.

    This lets pipeline_agent and agent accept an optional tracer without
    adding conditional branches at every call site — just call
    `tracer.emit(...)` unconditionally.
    """
    if not out_path:
        return NULL_TRACER
    return EventTracer(out_path, sample_id=sample_id, mode=mode)
