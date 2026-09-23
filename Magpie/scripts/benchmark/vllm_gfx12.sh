#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Magpie Generic vLLM Benchmark Script for AMD gfx12 (RDNA4)
#
# Phases (via MAGPIE_RUN_PHASE): all | server | client (default all).
# Server-only writes PID to MAGPIE_SERVER_PID_FILE. It either disowns and exits
# (local reuse) or waits when MAGPIE_KEEP_CONTAINER_ALIVE=1 (Docker reuse).
# Remote server (BENCHMARK_BASE_URL) forces PHASE=client; see vllm_mi300x.sh
# for the full contract.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for dependency in benchmark_lib.sh server_cleanup.sh magpie_bench_remote_compat.sh; do
  if [[ ! -r "$SCRIPT_DIR/$dependency" ]]; then
    echo "ERROR: Required benchmark dependency is missing: $SCRIPT_DIR/$dependency" >&2
    exit 3
  fi
done

source "$SCRIPT_DIR/benchmark_lib.sh"
source "$SCRIPT_DIR/server_cleanup.sh"
# shellcheck source=magpie_bench_remote_compat.sh
source "$SCRIPT_DIR/magpie_bench_remote_compat.sh"

PHASE="${MAGPIE_RUN_PHASE:-all}"
case "$PHASE" in
  all|server|client) ;;
  *) echo "ERROR: Invalid MAGPIE_RUN_PHASE='$PHASE'. Must be all|server|client." >&2; exit 2 ;;
esac

if [[ -n "${BENCHMARK_BASE_URL:-}" ]]; then
  if [[ "$PHASE" != "client" ]]; then
    echo "[vllm_gfx12] BENCHMARK_BASE_URL set; forcing PHASE=client (was $PHASE)"
    PHASE=client
  fi
fi

if [[ "$PHASE" == "server" || "$PHASE" == "all" ]]; then
  check_env_vars MODEL TP
fi
if [[ "$PHASE" == "client" || "$PHASE" == "all" ]]; then
  check_env_vars MODEL CONC ISL OSL RANDOM_RANGE_RATIO RESULT_FILENAME
fi

MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
RUN_EVAL_VALUE=${RUN_EVAL:-false}
RUN_EVAL_VALUE=${RUN_EVAL_VALUE,,}

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [[ "$PHASE" != "client" ]]; then
  hf download "$MODEL" 2>/dev/null || true
fi

# ROCR_VISIBLE_DEVICES already re-indexes visible GPUs to 0..N-1, so HIP
# must use the logical range, not the original physical ids.
if [ -n "$ROCR_VISIBLE_DEVICES" ] && [ -z "$HIP_VISIBLE_DEVICES" ]; then
    n=$(echo "$ROCR_VISIBLE_DEVICES" | awk -F, '{print NF}')
    export HIP_VISIBLE_DEVICES=$(seq -s, 0 $((n-1)))
fi

unset HSA_OVERRIDE_GFX_VERSION
export VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-0}
export FLASH_ATTENTION_TRITON_AMD_ENABLE=${FLASH_ATTENTION_TRITON_AMD_ENABLE:-TRUE}

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
if [[ "$PHASE" == "server" || "$PHASE" == "all" ]]; then
  EXTRA_SERVER_ARGS=()
  if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
    # Unquoted on purpose: split on IFS so block-scalar newlines survive.
    # shellcheck disable=SC2206
    EXTRA_SERVER_ARGS=($EXTRA_VLLM_ARGS)
  fi
  SERVER_CMD=(
    vllm serve "$MODEL"
    --port "$PORT"
    --tensor-parallel-size "$TP"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --trust-remote-code
    "${PROFILER_ARGS[@]}"
    "${EXTRA_SERVER_ARGS[@]}"
  )
  setsid "${SERVER_CMD[@]}" > "$SERVER_LOG" 2>&1 &

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
    if [[ "${MAGPIE_KEEP_CONTAINER_ALIVE:-0}" == "1" ]]; then
      trap 'magpie_stop_benchmark_server_stack "$SERVER_PID"' EXIT INT TERM
      wait "$SERVER_PID"
      exit $?
    fi
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
    SERVER_MONITOR_ARGS=()
    magpie_run_benchmark_serving_remote_direct trust || exit $?
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
        --result-dir "$WORKSPACE_DIR/" \
        "${SERVER_MONITOR_ARGS[@]}" \
        --trust-remote-code || exit $?
  fi
fi

if [[ "$PHASE" != "server" && "$RUN_EVAL_VALUE" == "true" ]]; then
    if [[ -n "${BENCHMARK_BASE_URL:-}" ]]; then
        if declare -F magpie_run_eval_remote_direct &>/dev/null; then
            magpie_run_eval_remote_direct || exit $?
        else
            echo "[vllm_gfx12] RUN_EVAL=true with BENCHMARK_BASE_URL but magpie_run_eval_remote_direct shim not available; skipping eval (results gate will see accuracy=None)."
        fi
    else
        export EVAL_CONCURRENT_REQUESTS="${MAGPIE_EVAL_CONCURRENCY:-${EVAL_CONCURRENT_REQUESTS:-$CONC}}"
        magpie_run_eval_persisted --framework lm-eval --port "$PORT" || exit $?
    fi
fi
set +x
