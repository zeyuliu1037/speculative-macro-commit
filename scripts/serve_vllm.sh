#!/usr/bin/env bash
# Start exactly one paper model server in the foreground; invoke once per GPU.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <appworld-actor|tau2-actor|appworld-drafter|tau2-drafter> <gpu-id> <port>" >&2
  exit 2
fi

PROFILE="$1"
GPU_ID="$2"
PORT="$3"

case "$PROFILE" in
  appworld-actor)
    MODEL="cyankiwi/Qwen3.5-27B-AWQ-4bit"
    SERVED_MODEL="Qwen/Qwen3.5-27B"
    DEFAULT_GPU_MEMORY_UTILIZATION="0.95"
    PROFILE_ARGS=(--language-model-only)
    ;;
  tau2-actor)
    MODEL="cyankiwi/Qwen3.5-27B-AWQ-BF16-INT4"
    SERVED_MODEL="Qwen/Qwen3.5-27B-BF16INT4"
    DEFAULT_GPU_MEMORY_UTILIZATION="0.90"
    PROFILE_ARGS=()
    ;;
  appworld-drafter)
    MODEL="Qwen/Qwen3.5-4B"
    SERVED_MODEL="Qwen/Qwen3.5-4B"
    DEFAULT_GPU_MEMORY_UTILIZATION="0.95"
    PROFILE_ARGS=()
    ;;
  tau2-drafter)
    MODEL="Qwen/Qwen3.5-4B"
    SERVED_MODEL="Qwen/Qwen3.5-4B"
    DEFAULT_GPU_MEMORY_UTILIZATION="0.90"
    PROFILE_ARGS=()
    ;;
  *)
    echo "Unknown profile: $PROFILE" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU_ID"
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED_MODEL" \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-$DEFAULT_GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-40960}" \
  --swap-space "${VLLM_SWAP_SPACE_GB:-16}" \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS:-32}" \
  --enable-prefix-caching \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  "${PROFILE_ARGS[@]}"
