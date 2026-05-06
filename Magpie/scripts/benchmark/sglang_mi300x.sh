#!/usr/bin/env bash
###############################################################################
# Magpie Generic SGLang Benchmark Script for MI300X
###############################################################################

source "$(dirname "$0")/benchmark_lib.sh"
source "$(dirname "$0")/server_cleanup.sh"

PHASE="${MAGPIE_RUN_PHASE:-all}"
case "$PHASE" in
  all|server|client) ;;
  *) echo "ERROR: Invalid MAGPIE_RUN_PHASE='$PHASE'. Must be all|server|client." >&2; exit 2 ;;
esac

if [[ "$PHASE" == "server" || "$PHASE" == "all" ]]; then
  check_env_vars MODEL TP
fi
if [[ "$PHASE" == "client" || "$PHASE" == "all" ]]; then
  check_env_vars MODEL CONC ISL OSL RANDOM_RANGE_RATIO RESULT_FILENAME
fi

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [[ "$PHASE" != "client" ]]; then
  hf download "$MODEL" 2>/dev/null || true
fi

# MI300X specific: Check MEC firmware version for RCCL memory reclaim
version=$(rocm-smi --showfw 2>/dev/null | grep MEC | head -n 1 | awk '{print $NF}')
if [[ "$version" == "" || $version -lt 177 ]]; then
  export HSA_NO_SCRATCH_RECLAIM=1
fi

# SGLang optimizations for MI300X
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
if [[ "$PHASE" == "server" || "$PHASE" == "all" ]]; then
  setsid python3 -m sglang.launch_server \
    --model-path=$MODEL \
    --host=0.0.0.0 \
    --port=$PORT \
    --trust-remote-code \
    --tensor-parallel-size=$TP \
    $DEFAULT_ARGS \
    $EXTRA_SGLANG_ARGS > $SERVER_LOG 2>&1 &

  SERVER_PID=$!
  if [[ -n "${MAGPIE_SERVER_PID_FILE:-}" ]]; then
    echo "$SERVER_PID" > "$MAGPIE_SERVER_PID_FILE"
  fi
  if [[ "$PHASE" == "all" ]]; then
    trap 'magpie_stop_benchmark_server_stack "$SERVER_PID"' EXIT INT TERM
  else
    trap 'kill -TERM "-$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; exit 0' INT TERM
  fi

  # Wait for server to be ready
  wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

  if [[ "$PHASE" == "server" ]]; then
    wait "$SERVER_PID"
    exit 0
  fi
fi

if [[ "$PHASE" == "client" || "$PHASE" == "all" ]]; then
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
fi

# After throughput, run evaluation only if RUN_EVAL is true
if [[ "$PHASE" != "server" && "${RUN_EVAL}" = "true" ]]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi
set +x




