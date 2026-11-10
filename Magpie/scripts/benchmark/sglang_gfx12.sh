#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Magpie Generic SGLang Benchmark Script for AMD gfx12 (RDNA4)
#
# Phases (via MAGPIE_RUN_PHASE): all | server | client (default all).
# Server-only writes PID to MAGPIE_SERVER_PID_FILE. It either disowns and exits
# (local reuse) or waits when MAGPIE_KEEP_CONTAINER_ALIVE=1 (Docker reuse).
# Remote server (BENCHMARK_BASE_URL) forces PHASE=client; see sglang_mi300x.sh
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
    echo "[sglang_gfx12] BENCHMARK_BASE_URL set; forcing PHASE=client (was $PHASE)"
    PHASE=client
  fi
fi

if [[ "$PHASE" == "server" || "$PHASE" == "all" ]]; then
  check_env_vars MODEL TP
fi
if [[ "$PHASE" == "client" || "$PHASE" == "all" ]]; then
  check_env_vars MODEL CONC ISL OSL RANDOM_RANGE_RATIO RESULT_FILENAME
fi

for numeric in TP CONC ISL OSL PORT; do
  value=${!numeric:-}
  if [[ -n "$value" && ! "$value" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $numeric must be an unsigned integer, got '$value'." >&2
    exit 2
  fi
done

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
# The ROCm SGLang images export SGLANG_USE_AITER=1, but build AITER for
# gfx942/gfx950 only, so a gfx12 server has to opt out rather than inherit it.
export SGLANG_USE_AITER=0
unset SGLANG_AITER_MLA_PERSIST

WORKSPACE_DIR=${RESULT_DIR:-/workspace}
SERVER_LOG=${SERVER_LOG:-$WORKSPACE_DIR/server.log}
PORT=${PORT:-8888}

if [[ "${PROFILE:-}" == "1" ]]; then
  TRACE_DIR="${SGLANG_TORCH_PROFILER_DIR:-$WORKSPACE_DIR/torch_trace}"
  mkdir -p "$TRACE_DIR"
  export SGLANG_TORCH_PROFILER_DIR="$TRACE_DIR"
fi

# Split on spaces/tabs/newlines with pathname expansion off, so *, ? and
# brackets in an arg stay literal.
EXTRA_SGLANG_ARGV=()
EXTRA_SGLANG_RAW="${EXTRA_SGLANG_ARGS:-}"
EXTRA_SGLANG_RAW="${EXTRA_SGLANG_RAW//$'\r'/}"
if [[ -n "$EXTRA_SGLANG_RAW" ]]; then
  read -r -d '' -a EXTRA_SGLANG_ARGV <<< "$EXTRA_SGLANG_RAW" || true
fi

extra_sglang_sets() {
  local flag="$1" arg
  for arg in "${EXTRA_SGLANG_ARGV[@]}"; do
    if [[ "$arg" == "$flag" || "$arg" == "$flag="* ]]; then
      return 0
    fi
  done
  return 1
}

# Each default applies only when EXTRA_SGLANG_ARGS leaves its flag out.
DEFAULT_ARGS=()
extra_sglang_sets --mem-fraction-static || DEFAULT_ARGS+=("--mem-fraction-static=0.9")
# SGLang refuses --disable-radix-cache next to --enable-hierarchical-cache: HiCache is built on the radix cache.
if ! extra_sglang_sets --disable-radix-cache && ! extra_sglang_sets --enable-hierarchical-cache; then
  DEFAULT_ARGS+=("--disable-radix-cache")
fi
extra_sglang_sets --attention-backend || DEFAULT_ARGS+=("--attention-backend=triton")

set -x
if [[ "$PHASE" == "server" || "$PHASE" == "all" ]]; then
  SERVER_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tensor-parallel-size "$TP"
    "${DEFAULT_ARGS[@]}"
    "${EXTRA_SGLANG_ARGV[@]}"
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
    # InferenceX names its OpenAI-compatible completions client "vllm";
    # SGLang serves that protocol.
    magpie_run_benchmark_serving \
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
            echo "[sglang_gfx12] RUN_EVAL=true with BENCHMARK_BASE_URL but magpie_run_eval_remote_direct shim not available; skipping eval (results gate will see accuracy=None)."
        fi
    else
        export EVAL_CONCURRENT_REQUESTS="${MAGPIE_EVAL_CONCURRENCY:-${EVAL_CONCURRENT_REQUESTS:-$CONC}}"
        magpie_run_eval_persisted --framework lm-eval --port "$PORT" || exit $?
    fi
fi
set +x
