#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Magpie Generic vLLM Benchmark Script for MI355X

source "$(dirname "$0")/benchmark_lib.sh"
source "$(dirname "$0")/server_cleanup.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

hf download "$MODEL" 2>/dev/null || true

# MI355X specific: Check MEC firmware version for RCCL memory reclaim
version=$(rocm-smi --showfw 2>/dev/null | grep MEC | head -n 1 | awk '{print $NF}')
if [[ "$version" == "" || $version -lt 177 ]]; then
  export HSA_NO_SCRATCH_RECLAIM=1
fi

# ROCR_VISIBLE_DEVICES already re-indexes visible GPUs to 0..N-1, so HIP
# must use the logical range, not the original physical ids.
if [ -n "$ROCR_VISIBLE_DEVICES" ] && [ -z "$HIP_VISIBLE_DEVICES" ]; then
    n=$(echo "$ROCR_VISIBLE_DEVICES" | awk -F, '{print NF}')
    export HIP_VISIBLE_DEVICES=$(seq -s, 0 $((n-1)))
fi

# vLLM optimizations for MI355X
export VLLM_ROCM_USE_AITER=1

WORKSPACE_DIR=${RESULT_DIR:-/workspace}
SERVER_LOG=${SERVER_LOG:-$WORKSPACE_DIR/server.log}
PORT=${PORT:-8888}

# Build profiler args for vLLM >= 0.15 (env var VLLM_TORCH_PROFILER_DIR is deprecated)
PROFILER_ARGS=()
if [[ "${PROFILE:-}" == "1" ]]; then
  TRACE_DIR="${VLLM_TORCH_PROFILER_DIR:-$WORKSPACE_DIR/torch_trace}"
  mkdir -p "$TRACE_DIR"
  PROFILER_ARGS+=(--profiler-config.profiler torch)
  PROFILER_ARGS+=(--profiler-config.torch_profiler_dir "$TRACE_DIR")
  PROFILER_ARGS+=(--profiler-config.torch_profiler_record_shapes True)
  PROFILER_ARGS+=(--profiler-config.torch_profiler_with_memory True)
  PROFILER_ARGS+=(--profiler-config.torch_profiler_with_flops True)
  PROFILER_ARGS+=(--profiler-config.torch_profiler_use_gzip True)
fi

set -x
# setsid: PG leader so magpie_stop_benchmark_server_stack can kill the whole tree.
setsid vllm serve $MODEL --port $PORT \
  --tensor-parallel-size=$TP \
  --gpu-memory-utilization 0.95 \
  --max-model-len $MAX_MODEL_LEN \
  --trust-remote-code \
  "${PROFILER_ARGS[@]}" \
  $EXTRA_VLLM_ARGS > $SERVER_LOG 2>&1 &

SERVER_PID=$!
trap 'magpie_stop_benchmark_server_stack "$SERVER_PID"' EXIT INT TERM

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts ${NUM_PROMPTS:-$(( $CONC * 10 ))} \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir "$WORKSPACE_DIR/" \
    --server-pid "$SERVER_PID" \
    --trust-remote-code

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi
set +x
