#!/usr/bin/env bash
#
# Start vLLM serving Qwen3-30B-A3B for the assignment.
#
# Flags are chosen for THIS workload, not defaults: 1.5-3K-token prompts, short
# structured SQL outputs, ~2-3 dependent LLM calls per request, on one H100 80GB.
# One-line rationale per flag lives in REPORT.md Section 1.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
#
# Every knob is env-overridable, so an experiment changes ONE lever and restarts
# without editing this file. See the symptom -> lever guide at the bottom.

set -euo pipefail

# Qwen3-30B-A3B is a Mixture-of-Experts model: ~30.5B total params, ~3.3B active per token.
# Memory is set by the TOTAL (all experts stay resident), so FP8 weights are what make this
# fit on one 80GB card: BF16 weights are ~61GB and leave almost no room for KV cache.
# Compute is set by the small ACTIVE count, which keeps decode cheap.
# The pre-quantized FP8 checkpoint loads with no quantization flag (29.1 GiB measured).
# We expose it under the base model id so the agent's VLLM_MODEL
# ("Qwen/Qwen3-30B-A3B-Instruct-2507") needs no change.
MODEL="${VLLM_MODEL_SERVE:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"
SERVED_NAME="${VLLM_SERVED_NAME:-Qwen/Qwen3-30B-A3B-Instruct-2507}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"                    # prompts 1.5-3K; 262K native reserves KV for nothing
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"                      # KV pool size; KV ran ~5% so this is headroom, not a limit
MAX_NUM_SEQS="${MAX_NUM_SEQS:-48}"                        # running-batch cap; concurrency reported 57x at 8K tokens
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"  # prefill-heavy: one full prompt fits the budget, low TTFT

# CUDA graphs ON (no --enforce-eager): faster decode at steady state, the right default
# for a served endpoint. We pass --enforce-eager ONLY when ENFORCE_EAGER=1, which buys a
# faster boot at a small runtime cost (useful when slot time is tight, not for reported runs).
EAGER_FLAG=""
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
    EAGER_FLAG="--enforce-eager"
fi

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enable-prefix-caching \
    $EAGER_FLAG

# --- Changing ONE lever per iteration, matched to what the dashboard shows ---
#   waiting queue grows but KV has headroom     -> MAX_NUM_SEQS=64          (admit more concurrency)
#   TTFT high under load                        -> MAX_NUM_BATCHED_TOKENS=16384
#   OOM / CUDA-graph capture failure at start   -> GPU_MEM_UTIL=0.88        (back off)
#   model load + graph capture eating the clock -> ENFORCE_EAGER=1          (faster boot, not for reported runs)
# Example: MAX_NUM_SEQS=64 bash scripts/start_vllm.sh
#
# NOTE on this workload: KV cache was NOT the bottleneck (gpu_cache_usage ~5% throughout),
# so FP8 KV cache and similar memory levers do not apply here. The binding constraint at
# 10 RPS is throughput plus the agent's 2-3 sequential LLM calls per /answer, which is an
# agent-side fix (async handler, more uvicorn workers, fewer calls) or more GPU, not a flag.
