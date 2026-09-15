# [Speculative Macro Commit for Faster Tool-Using Agents](https://arxiv.org/abs/2609.03236)

Official implementation of **Speculative Macro Commit (SMC)**. SMC verifies a
speculative action and, when the verified prefix matches a learned macro-step,
commits several following actions together. This repository contains the
AppWorld and tau2 Telecom implementations used in the camera-ready paper.

The release has two independent parts:

- **Library construction:** mine reusable action sequences from training runs.
- **Macro execution:** load a frozen library, match it after verification, and
  skip the corresponding actor decisions.

The two libraries used for the paper are included under `libraries/`.

## Main results

| Benchmark | Tasks | Baseline | Speculative Action | SMC |
| --- | ---: | ---: | ---: | ---: |
| tau2 Telecom | 2,285 | 99.52% / 27.60 s | 99.47% / 25.03 s | **99.52% / 22.47 s** |
| AppWorld `test_normal` | 168 | 41.67% / 355.7 s | 41.67% / 212.1 s | **40.48% / 195.9 s** |

Each cell reports success and mean wall-clock time per task. Runtime depends on
the GPU, serving stack, and cache state; use the checked-in configurations when
comparing against these numbers.

## Repository structure

```text
configs/          paper and mining configurations for both benchmarks
environment/      Conda environments for serving, AppWorld, and tau2
libraries/        the two frozen macro-step libraries and their manifest
scripts/          model-server, experiment, and result-summary entry points
smc_appworld/     AppWorld runtime, runner, and library miner
tau2_telecom/     tau2 runtime, runner, label collector, and library miner
```

The main method implementations are
`smc_appworld/runtime/pipeline_agent.py` and
`tau2_telecom/src/runner.py`. Most experiments only need the JSON configs and
the entry points shown below.

## Installation

The model server and benchmark clients use separate environments:

```bash
conda env create -f environment/serving.yml
conda env create -f environment/appworld.yml
conda env create -f environment/tau2.yml
```

Install AppWorld data using its upstream CLI, then pass the resulting root to
the runner:

```bash
conda activate smc-appworld
appworld install
appworld download data
```

The tau2 environment installs the fixed code revision used by this release.
Its benchmark data can be obtained from the upstream repository:

```bash
git clone https://github.com/sierra-research/tau2-bench.git external/tau2-bench
git -C external/tau2-bench checkout 7483cc60e4957fb2cf834c05a41059a715ee287c
export TAU2_DATA_DIR="$PWD/external/tau2-bench/data"
```

## Launch model servers

Activate `smc-serving` and start each command in a separate terminal. The
default paper setup uses one drafter and two actor endpoints:

```bash
# AppWorld
./scripts/serve_vllm.sh appworld-drafter 0 8003
./scripts/serve_vllm.sh appworld-actor   1 8004
./scripts/serve_vllm.sh appworld-actor   2 8005

# tau2 Telecom: use these profiles instead
./scripts/serve_vllm.sh tau2-drafter 0 8003
./scripts/serve_vllm.sh tau2-actor   1 8004
./scripts/serve_vllm.sh tau2-actor   2 8005
```

The checkpoint names and vLLM flags are kept in `scripts/serve_vllm.sh`.
Before a run, check that `/v1/models` responds on ports 8003, 8004, and 8005.
The paper timing path used a fresh server epoch followed by readiness checks;
it did not require a separate generation warmup.

## AppWorld

With the AppWorld servers ready, a one-task mechanism check is:

```bash
conda activate smc-appworld
python -m smc_appworld.run \
  --config configs/appworld/smc.json \
  --task-ids-file configs/appworld/smoke.json \
  --appworld-root /path/to/appworld-root \
  --run-id smc_smoke --no-resume
```

For the paper evaluation, SA and SMC are alternated task by task to balance the
shared prefix cache. The command is a dry run unless `--execute` is supplied:

```bash
python scripts/run_appworld_paired.py
python scripts/run_appworld_paired.py \
  --execute \
  --appworld-root /path/to/appworld-root \
  --run-id appworld_sa_smc
```

Run the baseline in a separate fresh server epoch:

```bash
python -m smc_appworld.run \
  --config configs/appworld/baseline.json \
  --appworld-root /path/to/appworld-root \
  --run-id appworld_baseline --no-resume
```

Summarize a new baseline/SA/SMC set with:

```bash
python scripts/summarize_appworld_runs.py \
  --baseline runs/appworld/appworld_baseline_baseline \
  --sa runs/appworld_paired/appworld_sa_smc/sa \
  --smc runs/appworld_paired/appworld_sa_smc/smc \
  --strict
```

## tau2 Telecom

The SMC smoke uses a task that fired a macro in the paper environment:

```bash
conda activate smc-tau2
python scripts/run_tau2_config.py configs/tau2/smc.json \
  --task-ids-file configs/tau2/smoke.json \
  --run-id smc_smoke
```

The full configurations are direct entry points. Start a fresh server epoch
before each independent arm:

```bash
python scripts/run_tau2_config.py configs/tau2/baseline.json --run-id baseline
python scripts/run_tau2_config.py configs/tau2/sa.json       --run-id sa
python scripts/run_tau2_config.py configs/tau2/smc.json      --run-id smc
```

The runner writes compact and resumable results under `runs/tau2/`. A new
three-arm run can be summarized with:

```bash
python tau2_telecom/scripts/analyze_macro_runs.py \
  --run-dir runs/tau2 --arms baseline sa smc --baseline-arm baseline \
  --out-json runs/tau2_summary.json --out-md runs/tau2_summary.md
```

## Build macro-step libraries

For reproducing the paper evaluation, use the frozen files rather than
re-mining them:

| Benchmark | Construction | Frozen library |
| --- | --- | --- |
| AppWorld | action n-grams, frequency >= 5, coverage >= 0.10; no LCB | `libraries/appworld_train50.json` (90 patterns) |
| tau2 Telecom | support >= 3, offline LCB >= 0.50, top 200 | `libraries/tau2_sup3_lcb050.json` |

To rebuild an AppWorld library, first collect successful training trajectories
and then run the miner:

```bash
python -m smc_appworld.run \
  --mode baseline --split train --max-steps 50 \
  --actor-url http://localhost:8004/v1 \
  --appworld-root /path/to/appworld-root \
  --run-id appworld_train_maxsteps50 \
  --save-raw-trajectories --no-resume

python -m smc_appworld.mining --config configs/appworld/mining.json
```

For tau2, collect successful owner runs, label drafter continuations, and mine
the library grid:

```bash
python tau2_telecom/scripts/prepare_task_ids.py \
  --out-dir runs/mining/tau2_task_ids

python scripts/run_tau2_config.py configs/tau2/baseline.json \
  --task-ids-file runs/mining/tau2_task_ids/small_train.json \
  --run-id golden_small_train --out-dir runs/mining/tau2_golden

python scripts/run_tau2_mining_config.py configs/tau2/mining.json collect \
  --results-dir runs/mining/tau2_golden/golden_small_train/results \
  --output runs/mining/tau2_labels.jsonl

python scripts/run_tau2_mining_config.py configs/tau2/mining.json mine \
  --labels runs/mining/tau2_labels.jsonl \
  --out-dir runs/mining/tau2_libraries
```

LCB is only part of tau2's **offline library construction**. AppWorld did not
use it. The released tau2 library was filtered with the historical
`camera_ready_wilson` fallback; the paper describes the corresponding Beta
posterior lower quantile. An exact-Beta audit kept the same final 200 action
sequences, but changed their scores and order, so evaluation uses the frozen
file. The runtime threshold remains zero because it does not filter the
already-filtered library a second time.

Freshly sampled trajectories may produce a different library. Re-mining
reproduces the construction procedure; loading the frozen files reproduces the
paper execution path.

## Evaluation notes

- Each runner creates fresh benchmark state for every task.
- Restart the model servers between independent arms; do not reuse a smoke
  epoch for timed evaluation.
- A smoke checks that verification, matching, and macro commit can fire. It is
  successful when its summary has no task error and reports nonzero macro hits
  and skipped steps; it is not evidence for full-benchmark accuracy or latency.
- Raw trajectories and run outputs may contain benchmark-sensitive content;
  `runs/`, `*.jsonl`, and raw trajectories are ignored by Git.

## Citation

Please cite the accompanying paper if you use this code:

```bibtex
@misc{liu2026speculativemacrocommitfaster,
      title={Speculative Macro Commit for Faster Tool-Using Agents}, 
      author={Zeyu Liu and Souvik Kundu and Peter A. Beerel},
      year={2026},
      eprint={2609.03236},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2609.03236}, 
}
```

The same metadata is available in `CITATION.cff` for GitHub's citation menu.

## License and acknowledgments

The authors' code is released under Apache-2.0. [AppWorld](https://github.com/StonyBrookNLP/appworld),
[tau2-bench](https://github.com/sierra-research/tau2-bench), model weights,
and benchmark data retain their upstream licenses and are not redistributed
here.
