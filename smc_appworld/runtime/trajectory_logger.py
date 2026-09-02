"""Per-step trajectory logger for AppWorld experiments.

Writes incrementally after each step to survive crashes and records the
per-step fields used by the paper-era AppWorld experiments.
"""

import ctypes
import ctypes.util
import json
import os
from typing import Any, Optional

# AppWorld uses freezegun which freezes all time functions.
_CLOCK_MONOTONIC = 1
class _Timespec(ctypes.Structure):
    _fields_ = [('tv_sec', ctypes.c_long), ('tv_nsec', ctypes.c_long)]
_librt = ctypes.CDLL(ctypes.util.find_library('rt') or 'librt.so.1')
_ts = _Timespec()

def perf_counter():
    _librt.clock_gettime(_CLOCK_MONOTONIC, ctypes.byref(_ts))
    return _ts.tv_sec + _ts.tv_nsec / 1e9


class TrajectoryLogger:
    def __init__(self, output_dir: str, task_id: str):
        self.output_dir = output_dir
        self.task_id = task_id
        self.steps: list[dict] = []
        self.start_time = perf_counter()

        os.makedirs(output_dir, exist_ok=True)
        self.filepath = os.path.join(output_dir, f"{task_id}.json")

    def log_step(
        self,
        step_id: int,
        instruction: str,
        model_output: str,
        api_name: str,
        api_params: dict,
        api_result: str,
        wall_clock_time: float,
        input_tokens: int,
        output_tokens: int,
        schema_error: bool = False,
        execution_error: bool = False,
        thinking_truncated: bool = False,
        tier: str = "",
    ):
        record = {
            "task_id": self.task_id,
            "step_id": step_id,
            "instruction": instruction[:500],
            "model_output": model_output[:2000],
            "api_name": api_name,
            "api_params": api_params,
            "api_result": str(api_result)[:2000],
            "wall_clock_time": round(wall_clock_time, 3),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "schema_error": schema_error,
            "execution_error": execution_error,
            "cumulative_steps": step_id,
            "task_success": None,
            "evaluation_score": None,
            "thinking_truncated": thinking_truncated,
            "tier": tier,
        }
        self.steps.append(record)
        self._save()

    def finalize(self, task_success: bool, evaluation_score: float):
        """Set final success/score on all steps and save."""
        for step in self.steps:
            step["task_success"] = task_success
            step["evaluation_score"] = evaluation_score
        self._save()

    def _save(self):
        with open(self.filepath, "w") as f:
            json.dump(
                {
                    "task_id": self.task_id,
                    "total_steps": len(self.steps),
                    "total_wall_time": round(perf_counter() - self.start_time, 2),
                    "steps": self.steps,
                },
                f,
                indent=2,
            )

    def get_summary(self) -> dict:
        """Return summary stats for this trajectory."""
        n = len(self.steps)
        if n == 0:
            return {"task_id": self.task_id, "n_steps": 0}

        schema_errors = sum(1 for s in self.steps if s["schema_error"])
        exec_errors = sum(1 for s in self.steps if s["execution_error"])
        avg_latency = sum(s["wall_clock_time"] for s in self.steps) / n
        total_input = sum(s["input_tokens"] for s in self.steps)
        total_output = sum(s["output_tokens"] for s in self.steps)

        # Action n-grams
        actions = [s["api_name"] for s in self.steps if s["api_name"]]
        bigrams = []
        for i in range(len(actions) - 1):
            bigrams.append(f"{actions[i]} -> {actions[i+1]}")

        return {
            "task_id": self.task_id,
            "n_steps": n,
            "schema_errors": schema_errors,
            "execution_errors": exec_errors,
            "avg_step_latency": round(avg_latency, 2),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_wall_time": round(perf_counter() - self.start_time, 2),
            "task_success": self.steps[-1].get("task_success"),
            "evaluation_score": self.steps[-1].get("evaluation_score"),
            "action_sequence": actions,
            "action_bigrams": bigrams,
        }
