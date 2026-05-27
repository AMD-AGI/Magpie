#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Magpie Generic Atom Benchmark Script for MI355X
#
# Phases (via MAGPIE_RUN_PHASE): all | server | client (default all).
# Server-only writes PID to MAGPIE_SERVER_PID_FILE then disowns and exits.
#
# Atom exposes an OpenAI-compatible HTTP server at
# atom.entrypoints.openai_server. Its REST surface is wire-compatible
# with vLLM, so the bench client uses --backend vllm unchanged. See
# atom_mi300x.sh for the full contract.

source "$(dirname "$0")/benchmark_lib.sh"
source "$(dirname "$0")/server_cleanup.sh"
# shellcheck source=magpie_bench_remote_compat.sh
[[ -f "$(dirname "$0")/magpie_bench_remote_compat.sh" ]] && source "$(dirname "$0")/magpie_bench_remote_compat.sh"

PHASE="${MAGPIE_RUN_PHASE:-all}"
case "$PHASE" in
  all|server|client) ;;
  *) echo "ERROR: Invalid MAGPIE_RUN_PHASE='$PHASE'. Must be all|server|client." >&2; exit 2 ;;
esac

if [[ -n "${BENCHMARK_BASE_URL:-}" ]]; then
  if [[ "$PHASE" != "client" ]]; then
    echo "[atom_mi355x] BENCHMARK_BASE_URL set; forcing PHASE=client (was $PHASE)"
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

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [[ "$PHASE" != "client" ]]; then
  hf download "$MODEL" 2>/dev/null || true
fi

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

WORKSPACE_DIR=${RESULT_DIR:-/workspace}
SERVER_LOG=${SERVER_LOG:-$WORKSPACE_DIR/server.log}
PORT=${PORT:-8888}

if [[ "${PROFILE:-}" == "1" ]]; then
  echo "[atom_mi355x] PROFILE=1 received but atom profiler wiring is not yet implemented; ignoring."
fi

set -x
if [[ "$PHASE" == "server" || "$PHASE" == "all" ]]; then
  setsid python3 -m atom.entrypoints.openai_server \
    --model "$MODEL" \
    -tp "$TP" \
    --server-port "$PORT" \
    $EXTRA_ATOM_ARGS > "$SERVER_LOG" 2>&1 &

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

if [[ "$PHASE" != "server" && "${RUN_EVAL}" = "true" ]]; then
    if [[ -n "${BENCHMARK_BASE_URL:-}" ]]; then
        if declare -F magpie_run_eval_remote_direct &>/dev/null; then
            magpie_run_eval_remote_direct || exit $?
        else
            echo "[atom_mi355x] RUN_EVAL=true with BENCHMARK_BASE_URL but magpie_run_eval_remote_direct shim not available; skipping eval (results gate will see accuracy=None)."
        fi
    else
        run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC || exit $?
        append_lm_eval_summary
    fi
fi
set +x
