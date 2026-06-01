#!/usr/bin/env bash
#
# Start vLLM with YOUR chosen configuration.
#
# The model and hardware are fixed. Everything else is your call.
# Add the flags you decide on below. The line you see here is the
# bare-minimum invocation; it WILL run, but it WILL NOT hit the SLO.

set -euo pipefail

# Pick the model (or a pre-quantized variant) you want to serve.
#   BF16:    Qwen/Qwen3-30B-A3B-Instruct-2507
#   FP8:     Qwen/Qwen3-30B-A3B-Instruct-2507-FP8     (if available)
#   AWQ:     a community AWQ-Int4 of the same model
#   GPTQ:    a community GPTQ-Int4 of the same model
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

# Your flags go here. Examples (NOT recommendations):
#   --max-model-len <int>
#   --max-num-seqs <int>
#   --gpu-memory-utilization <float>
#   --quantization <name>
#   --enable-chunked-prefill
#   --speculative-model <draft-model-repo>
#   --num-speculative-tokens <int>
#   --kv-cache-dtype <fp8|auto>
#   --enable-prefix-caching
#
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
EXTRA_FLAGS=(
    # add yours
)

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    "${EXTRA_FLAGS[@]}"
