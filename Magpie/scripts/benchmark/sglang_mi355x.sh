#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Magpie Generic SGLang Benchmark Script for MI355X

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

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

hf download "$MODEL" 2>/dev/null || true

# MI355X specific: Check MEC firmware version for RCCL memory reclaim
version=$(rocm-smi --showfw 2>/dev/null | grep MEC | head -n 1 | awk '{print $NF}')
if [[ "$version" == "" || $version -lt 177 ]]; then
  export HSA_NO_SCRATCH_RECLAIM=1
fi

# SGLang optimizations for MI355X
export SGLANG_USE_AITER=1
export SGLANG_AITER_MLA_PERSIST=1

WORKSPACE_DIR=${RESULT_DIR:-/workspace}
SERVER_LOG=${SERVER_LOG:-$WORKSPACE_DIR/server.log}
PORT=${PORT:-8888}

if [[ "${PROFILE:-}" == "1" ]]; then
  TRACE_DIR="${SGLANG_TORCH_PROFILER_DIR:-$WORKSPACE_DIR/torch_trace}"
  mkdir -p "$TRACE_DIR"
  export SGLANG_TORCH_PROFILER_DIR="$TRACE_DIR"
fi

# Build default args, skipping any that EXTRA_SGLANG_ARGS already overrides
DEFAULT_ARGS=""
for flag_val in "--mem-fraction-static=0.8" "--disable-radix-cache"; do
  flag="${flag_val%%=*}"
  if [[ -z "$EXTRA_SGLANG_ARGS" ]] || ! echo "$EXTRA_SGLANG_ARGS" | grep -q -- "$flag"; then
    DEFAULT_ARGS="$DEFAULT_ARGS $flag_val"
  fi
done

set -x
setsid python3 -m sglang.launch_server \
  --model-path=$MODEL \
  --host=0.0.0.0 \
  --port=$PORT \
  --trust-remote-code \
  --tensor-parallel-size=$TP \
  $DEFAULT_ARGS \
  $EXTRA_SGLANG_ARGS > $SERVER_LOG 2>&1 &

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
    --result-dir ${RESULT_DIR:-/workspace/}

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi
set +x




