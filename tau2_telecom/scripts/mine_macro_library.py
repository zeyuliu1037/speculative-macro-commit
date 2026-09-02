#!/usr/bin/env python3
"""Build TAU2 macro-skip libraries from top-5 S1 label JSONL files."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tau2_telecom.src.macro_safety import annotate_pattern_safety


def reliability_lcb(
    alpha: float,
    beta: float,
    quantile: float,
    method: str = "camera_ready_wilson",
) -> float:
    """Return the requested lower confidence bound deterministically.

    The camera-ready library was generated in an environment without SciPy,
    so the historical implementation followed its Wilson fallback. Making
    that choice explicit prevents the same command from silently producing a
    different library merely because SciPy happens to be installed.

    ``beta`` implements the posterior-quantile equation in the paper and
    requires SciPy. It is useful for audits, but it does not reproduce the
    frozen library shipped with this release.
    """
    if method == "beta":
        try:
            from scipy.stats import beta as scipy_beta
        except ImportError as exc:
            raise RuntimeError(
                "--lcb-method beta requires scipy; the frozen paper "
                "artifact uses camera_ready_wilson"
            ) from exc
        return float(scipy_beta.ppf(quantile, alpha, beta))
    if method != "camera_ready_wilson":
        raise ValueError(f"unknown LCB method: {method}")
    n = alpha + beta
    if n <= 0:
        return 0.0
    p = alpha / n
    z = 1.645 if abs(quantile - 0.05) < 1e-9 else 1.282
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(
        p * (1.0 - p) / n + z * z / (4.0 * n * n)
    )
    return max(0.0, center - margin)


def beta_lcb(alpha: float, beta: float, quantile: float) -> float:
    """Backward-compatible name for the camera-ready Wilson computation."""
    return reliability_lcb(alpha, beta, quantile, "camera_ready_wilson")


def safe_id(prefix: str, rank: int, sequence: tuple[str, ...]) -> str:
    hint = "_".join(sequence[:2])
    hint = re.sub(r"[^A-Za-z0-9_]+", "_", hint).strip("_")[:48]
    return f"{prefix}_{rank:03d}_{len(sequence)}x_{hint}"


def load_rows(paths: list[Path], label_field: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get(label_field) in ("Accept", "Reject"):
                    rows.append(row)
    return rows


def collect_counts(
    rows: list[dict[str, Any]], label_field: str, dedupe: bool
) -> tuple[dict[tuple[str, ...], Counter[str]], dict[tuple[str, ...], set[str]]]:
    counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    tasks: dict[tuple[str, ...], set[str]] = defaultdict(set)
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        seq = tuple(str(item) for item in row.get("s1_chain_skeleton") or [])
        if len(seq) < 2:
            continue
        label = row.get(label_field)
        if label not in ("Accept", "Reject"):
            continue
        if dedupe:
            key = (
                row.get("task_id"),
                row.get("prefix_action_idx"),
                row.get("depth"),
                row.get("choice_index"),
                seq,
                label,
            )
            if key in seen:
                continue
            seen.add(key)
        counts[seq][label] += 1
        tasks[seq].add(str(row.get("task_id")))
    return counts, tasks


def score_patterns(
    counts: dict[tuple[str, ...], Counter[str]],
    tasks: dict[tuple[str, ...], set[str]],
    prior_p: float,
    kappa: float,
    quantile: float,
    length_weight: float,
    lcb_method: str = "camera_ready_wilson",
) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    for seq, ctr in counts.items():
        pos = float(ctr.get("Accept", 0))
        neg = float(ctr.get("Reject", 0))
        support = pos + neg
        alpha = kappa * prior_p + pos
        beta = kappa * (1.0 - prior_p) + neg
        post_mean = alpha / (alpha + beta) if (alpha + beta) else 0.0
        lcb = reliability_lcb(alpha, beta, quantile, lcb_method)
        score = lcb * math.log1p(pos) * (len(seq) ** length_weight)
        patterns.append(
            {
                "sequence": list(seq),
                "length": len(seq),
                "support": int(support),
                "n_pos": int(pos),
                "n_neg": int(neg),
                "task_support": len(tasks.get(seq, set())),
                "precision": round(pos / support, 4) if support else 0.0,
                "posterior_mean": round(post_mean, 4),
                "lcb": round(lcb, 4),
                "score": round(score, 4),
            }
        )
    patterns.sort(
        key=lambda p: (p["score"], p["lcb"], p["length"], p["support"]),
        reverse=True,
    )
    return patterns


def write_library(
    *,
    out_path: Path,
    patterns: list[dict[str, Any]],
    prefix: str,
    n_tasks: int,
    min_support: int,
    min_lcb: float,
    min_length: int,
    max_length: int,
    max_patterns: int,
    lcb_method: str,
) -> list[dict[str, Any]]:
    library: list[dict[str, Any]] = []
    for pattern in patterns:
        if len(library) >= max_patterns:
            break
        if pattern["support"] < min_support:
            continue
        if pattern["lcb"] < min_lcb:
            continue
        if pattern["length"] < min_length or pattern["length"] > max_length:
            continue
        sequence = tuple(pattern["sequence"])
        safety = annotate_pattern_safety({}, list(sequence))
        library.append(
            {
                "pattern_id": safe_id(prefix, len(library) + 1, sequence),
                "action_types": list(sequence),
                "macro_family": safety["macro_family"],
                "safety_class": safety["safety_class"],
                "support": pattern["support"],
                "frequency": pattern["support"],
                "coverage": round(pattern["task_support"] / max(n_tasks, 1), 4),
                "success_rate": pattern["lcb"],
                "precision": pattern["precision"],
                "posterior_mean": pattern["posterior_mean"],
                "lcb": pattern["lcb"],
                "score": pattern["score"],
                "task_support": pattern["task_support"],
                "source": "tau2_top5_s1_labels",
            }
        )
    payload = {
        "metadata": {
            "builder": "tau2_telecom/scripts/mine_macro_library.py",
            "prefix": prefix,
            "n_tasks": n_tasks,
            "min_support": min_support,
            "min_lcb": min_lcb,
            "min_length": min_length,
            "max_length": max_length,
            "max_patterns": max_patterns,
            "lcb_method": lcb_method,
        },
        "patterns": library,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return library


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label-field", choices=["label_skeleton", "label_exact"], default="label_skeleton")
    parser.add_argument("--prefix", default="tau2_top5_partial_skeleton")
    parser.add_argument("--prior-p", type=float, default=0.5)
    parser.add_argument("--kappa", type=float, default=2.0)
    parser.add_argument("--quantile", type=float, default=0.05)
    parser.add_argument("--length-weight", type=float, default=1.0)
    parser.add_argument(
        "--lcb-method",
        choices=["camera_ready_wilson", "beta"],
        default="camera_ready_wilson",
        help=(
            "camera_ready_wilson reproduces the frozen paper library; beta "
            "implements the paper equation and requires scipy"
        ),
    )
    parser.add_argument("--min-length", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=4)
    parser.add_argument("--max-patterns", type=int, default=200)
    parser.add_argument("--dedupe", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_rows(args.labels, args.label_field)
    counts, tasks = collect_counts(rows, args.label_field, args.dedupe)
    patterns = score_patterns(
        counts,
        tasks,
        prior_p=args.prior_p,
        kappa=args.kappa,
        quantile=args.quantile,
        length_weight=args.length_weight,
        lcb_method=args.lcb_method,
    )
    configs = [
        ("sup3_lcb050", 3, 0.50),
        ("sup3_lcb040", 3, 0.40),
        ("sup2_lcb035", 2, 0.35),
        ("sup3_lcb000", 3, 0.00),
        ("sup2_lcb000", 2, 0.00),
    ]
    n_tasks = len({str(row.get("task_id")) for row in rows})
    summaries = []
    for suffix, min_support, min_lcb in configs:
        name = f"{args.prefix}_{suffix}"
        out_path = args.out_dir / f"{name}.json"
        library = write_library(
            out_path=out_path,
            patterns=patterns,
            prefix=name,
            n_tasks=n_tasks,
            min_support=min_support,
            min_lcb=min_lcb,
            min_length=args.min_length,
            max_length=args.max_length,
            max_patterns=args.max_patterns,
            lcb_method=args.lcb_method,
        )
        summaries.append(
            {
                "name": name,
                "path": str(out_path),
                "patterns": len(library),
                "min_support": min_support,
                "min_lcb": min_lcb,
            }
        )
    summary = {
        "labels": [str(path) for path in args.labels],
        "label_field": args.label_field,
        "lcb_method": args.lcb_method,
        "n_rows": len(rows),
        "n_sequences": len(patterns),
        "n_tasks": n_tasks,
        "configs": summaries,
        "top_patterns": patterns[:30],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
