"""Meta-tool mining and matching for AppWorld.

Mines recurring action subsequences from agent trajectories,
then matches speculator depth predictions against known patterns
at runtime to enable multi-step skipping.

Domain-agnostic: works for any action vocabulary.
"""

import json
import os
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetaTool:
    """A mined meta-tool pattern."""
    pattern_id: str
    action_types: list[str]     # e.g., ["show_account_passwords", "spotify.login"]
    frequency: int              # occurrences across all traces
    coverage: float             # fraction of traces containing this pattern
    success_rate: float         # fraction of occurrences in successful traces
    avg_position: float         # average starting position in trace

    def __repr__(self):
        return f"MetaTool({self.pattern_id}: {' → '.join(self.action_types)} freq={self.frequency} cov={self.coverage:.0%})"


class MetaToolMiner:
    """Mine meta-tool patterns from agent trajectories."""

    def __init__(self, min_frequency: int = 3, min_coverage: float = 0.1):
        """
        Args:
            min_frequency: Minimum absolute count to keep a pattern.
            min_coverage: Minimum fraction of traces containing the pattern.
        """
        self.min_frequency = min_frequency
        self.min_coverage = min_coverage

    def mine(self, traces: list[dict]) -> list[MetaTool]:
        """Mine meta-tools from trajectory data.

        Args:
            traces: List of {"task_id": str, "actions": [str], "success": bool}

        Returns:
            List of MetaTool patterns, sorted by frequency.
        """
        n_traces = len(traces)
        if n_traces == 0:
            return []

        # Extract n-grams (n=2,3,4)
        ngram_counts = Counter()         # pattern_tuple → total count
        ngram_traces = defaultdict(set)  # pattern_tuple → set of task_ids
        ngram_positions = defaultdict(list)  # pattern_tuple → list of positions
        ngram_success = defaultdict(list)    # pattern_tuple → list of success bools

        for trace in traces:
            actions = trace["actions"]
            tid = trace["task_id"]
            success = trace.get("success", False)

            for n in range(2, 5):  # bigrams, trigrams, 4-grams
                for i in range(len(actions) - n + 1):
                    pattern = tuple(actions[i:i + n])
                    ngram_counts[pattern] += 1
                    ngram_traces[pattern].add(tid)
                    ngram_positions[pattern].append(i)
                    ngram_success[pattern].append(success)

        # Filter by frequency and coverage
        meta_tools = []
        for pattern, freq in ngram_counts.most_common():
            coverage = len(ngram_traces[pattern]) / n_traces
            if freq < self.min_frequency or coverage < self.min_coverage:
                continue

            # Check for redundancy: skip if a longer pattern subsumes this one
            # (keep both — the matcher handles prefix matching)

            positions = ngram_positions[pattern]
            successes = ngram_success[pattern]
            success_rate = sum(successes) / len(successes) if successes else 0.0
            avg_pos = sum(positions) / len(positions) if positions else 0.0

            pid = "_".join(a.replace(".", "_") for a in pattern)
            meta_tools.append(MetaTool(
                pattern_id=pid,
                action_types=list(pattern),
                frequency=freq,
                coverage=coverage,
                success_rate=success_rate,
                avg_position=avg_pos,
            ))

        # Sort by frequency (most common first)
        meta_tools.sort(key=lambda m: -m.frequency)
        return meta_tools

    def mine_from_trajectory_dir(self, results_dir: str) -> list[MetaTool]:
        """Mine from saved trajectory JSON files."""
        traces = []
        for fname in sorted(os.listdir(results_dir)):
            if not fname.endswith(".json") or fname.startswith(("selected", "metrics", "run")):
                continue
            with open(os.path.join(results_dir, fname)) as f:
                data = json.load(f)
            steps = data.get("steps", [])
            actions = []
            for s in steps:
                api = s.get("api_name", "")
                if api.startswith("_") or not api:
                    continue
                params = s.get("api_params", {})
                if api == "execute_api":
                    app = params.get("app_name", "?")
                    api_name = params.get("api_name", "?")
                    actions.append(f"{app}.{api_name}")
                else:
                    actions.append(api)

            success = False
            if steps and steps[-1].get("task_success") is not None:
                success = steps[-1]["task_success"]

            traces.append({
                "task_id": data.get("task_id", fname),
                "actions": actions,
                "success": success,
            })
        return self.mine(traces)


class MetaToolLibrary:
    """Runtime meta-tool matching against mined patterns."""

    def __init__(self, meta_tools: list[MetaTool] = None, path: str = None):
        self.meta_tools: list[MetaTool] = []
        if path and os.path.exists(path):
            self.load(path)
        elif meta_tools:
            self.meta_tools = meta_tools

    def match(self, predicted_actions: list[str]) -> Optional[MetaTool]:
        """Check if predicted action sequence matches any meta-tool pattern.

        Args:
            predicted_actions: List of predicted action types (from speculator depth chain).

        Returns:
            Matching MetaTool if found, None otherwise.
            Matches the LONGEST pattern first.
        """
        if not predicted_actions:
            return None

        best_match = None
        best_len = 0

        for mt in self.meta_tools:
            pattern = mt.action_types
            if len(pattern) > len(predicted_actions):
                continue
            # Check prefix match
            if predicted_actions[:len(pattern)] == pattern:
                if len(pattern) > best_len:
                    best_match = mt
                    best_len = len(pattern)

        return best_match

    def save(self, path: str):
        data = []
        for mt in self.meta_tools:
            data.append({
                "pattern_id": mt.pattern_id,
                "action_types": mt.action_types,
                "frequency": mt.frequency,
                "coverage": mt.coverage,
                "success_rate": mt.success_rate,
                "avg_position": mt.avg_position,
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.meta_tools = [MetaTool(**d) for d in data]

    def __len__(self):
        return len(self.meta_tools)

    def __repr__(self):
        return f"MetaToolLibrary({len(self.meta_tools)} patterns)"


def print_meta_tools(meta_tools: list[MetaTool]):
    """Pretty-print mined meta-tools."""
    print(f"{'#':<4} {'Pattern':<60} {'Freq':>5} {'Cov':>6} {'SuccR':>6} {'AvgPos':>7}")
    print("-" * 92)
    for i, mt in enumerate(meta_tools):
        pattern_str = " → ".join(mt.action_types)
        print(f"{i+1:<4} {pattern_str:<60} {mt.frequency:>5} {mt.coverage:>5.0%} "
              f"{mt.success_rate:>5.0%} {mt.avg_position:>7.1f}")
