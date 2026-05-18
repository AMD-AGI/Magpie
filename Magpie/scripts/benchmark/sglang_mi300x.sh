#!/usr/bin/env bash
###############################################################################
# Magpie Generic SGLang Benchmark Script for MI300X
###############################################################################
#
# Phases (via MAGPIE_RUN_PHASE): all | server | client (default all).
#
# Remote server (BENCHMARK_BASE_URL): when set, the client phase points
# benchmark_serving at an external SGLang-compatible HTTP endpoint
# instead of localhost:$PORT, and forces PHASE=client (no local server
# launch, no server-side cleanup, no SERVER_PID monitoring). Use this
# whenever the server is hosted off-pod (different node, different
# cluster, externally managed). Leave the env unset to keep the
# default behaviour of launching a local server.

source "$(dirname "$0")/benchmark_lib.sh"
source "$(dirname "$0")/server_cleanup.sh"
# shellcheck source=magpie_bench_remote_compat.sh
[[ -f "$(dirname "$0")/magpie_bench_remote_compat.sh" ]] && source "$(dirname "$0")/magpie_bench_remote_compat.sh"

PHASE="${MAGPIE_RUN_PHASE:-all}"
case "$PHASE" in
  all|server|client) ;;
  *) echo "ERROR: Invalid MAGPIE_RUN_PHASE='$PHASE'. Must be all|server|client." >&2; exit 2 ;;
esac

# When BENCHMARK_BASE_URL is set, force phase=client so the local server
# launch is skipped even if the operator (or Magpie default) asked for
# `all`. This keeps the contract simple: "set this env => run client only".
if [[ -n "${BENCHMARK_BASE_URL:-}" ]]; then
  if [[ "$PHASE" != "client" ]]; then
    echo "[sglang_mi300x] BENCHMARK_BASE_URL set; forcing PHASE=client (was $PHASE)"
    PHASE=client
  fi
fi

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
  if [[ "$PHASE" == "all" ]]; then
    trap 'magpie_stop_benchmark_server_stack "$SERVER_PID"' EXIT INT TERM
  fi

  wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

  if [[ "$PHASE" == "server" ]]; then
    if [[ -z "${MAGPIE_SERVER_PID_FILE:-}" ]]; then
      echo "ERROR: MAGPIE_SERVER_PID_FILE must be set for MAGPIE_RUN_PHASE=server" >&2
      kill -TERM "-$SERVER_PID" 2>/dev/null || true
      exit 3
    fi
    printf '%s\n' "$SERVER_PID" > "$MAGPIE_SERVER_PID_FILE"
    disown "$SERVER_PID" 2>/dev/null || true
    exit 0
  fi
fi

SERVER_MONITOR_ARGS=()
if [[ -n "${SERVER_PID:-}" ]]; then
  SERVER_MONITOR_ARGS+=(--server-pid "$SERVER_PID")
fi

if [[ "$PHASE" == "client" || "$PHASE" == "all" ]]; then
  if [[ -n "${BENCHMARK_BASE_URL:-}" ]]; then
    # Remote server: call Python benchmark_serving.py directly. Older
    # InferenceX benchmark_lib.sh run_benchmark_serving() rejects --base-url.
    SERVER_MONITOR_ARGS=()
    magpie_run_benchmark_serving_remote_direct || exit $?
  else
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
        "${SERVER_MONITOR_ARGS[@]}" \
        --result-dir ${RESULT_DIR:-/workspace/} || exit $?
  fi
fi

if [[ "$PHASE" != "server" && "${RUN_EVAL}" = "true" ]]; then
    if [[ -n "${BENCHMARK_BASE_URL:-}" ]]; then
        if declare -F magpie_run_eval_remote_direct &>/dev/null; then
            magpie_run_eval_remote_direct || exit $?
        else
            echo "[sglang_mi300x] RUN_EVAL=true with BENCHMARK_BASE_URL but magpie_run_eval_remote_direct shim not available; skipping eval (results gate will see accuracy=None)."
        fi
    else
        run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC || exit $?
        append_lm_eval_summary
    fi
fi
set +x

