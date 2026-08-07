#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Magpie Generic vLLM Benchmark Script for MI355X
#
# Phases (via MAGPIE_RUN_PHASE): all | server | client (default all).
# Server-only writes PID to MAGPIE_SERVER_PID_FILE then disowns and exits.
#
# Remote server (BENCHMARK_BASE_URL): when set, the client phase points
# benchmark_serving at an external vLLM-compatible HTTP endpoint
# instead of localhost:$PORT, and forces PHASE=client (no local server
# launch). See vllm_mi300x.sh for the full contract.

source "$(dirname "$0")/benchmark_lib.sh"
source "$(dirname "$0")/lm_eval_runtime.sh" || exit $?
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
    echo "[vllm_mi355x] BENCHMARK_BASE_URL set; forcing PHASE=client (was $PHASE)"
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
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
WORKSPACE_DIR=${RESULT_DIR:-/workspace}
MODEL_REVISION_RECEIPT="$WORKSPACE_DIR/model_revision_receipt.json"

mkdir -p "$WORKSPACE_DIR"

MODEL_REVISION_ARGS=()
if [[ -n "${MODEL_REVISION:-}" ]]; then
  if [[ ! "$MODEL_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: MODEL_REVISION must be an exact lowercase 40-hex commit." >&2
    exit 4
  fi
  MODEL_REVISION_ARGS+=(--revision "$MODEL_REVISION")
fi

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [[ "$PHASE" != "client" && -n "${MODEL_REVISION:-}" ]]; then
  if ! rm -f "$MODEL_REVISION_RECEIPT"; then
    echo "ERROR: could not clear stale model revision receipt." >&2
    exit 4
  fi
  MODEL_SNAPSHOT_PATH="$(
    hf download "$MODEL" --revision "$MODEL_REVISION" --format quiet
  )"
  download_status=$?
  if [[ $download_status -ne 0 ]]; then
    echo "ERROR: hf download failed for MODEL_REVISION=$MODEL_REVISION." >&2
    exit "$download_status"
  fi

  python3 - "$MODEL" "$MODEL_REVISION" "$MODEL_SNAPSHOT_PATH" \
    "$MODEL_REVISION_RECEIPT" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

model, requested, raw_snapshot, raw_receipt = sys.argv[1:]
commit_re = re.compile(r"^[0-9a-f]{40}$")
snapshot = Path(raw_snapshot).resolve(strict=True)
if not snapshot.is_dir():
    raise SystemExit(f"ERROR: hf download did not resolve to a directory: {snapshot}")
resolved = snapshot.name
if not commit_re.fullmatch(resolved):
    raise SystemExit(
        "ERROR: resolved Hugging Face snapshot is not an exact 40-hex commit: "
        f"{resolved!r}"
    )
if resolved != requested:
    raise SystemExit(
        "ERROR: resolved Hugging Face snapshot does not match MODEL_REVISION: "
        f"{resolved} != {requested}"
    )

receipt = Path(raw_receipt)
receipt.parent.mkdir(parents=True, exist_ok=True)
temporary = receipt.with_name(f".{receipt.name}.{os.getpid()}.tmp")
payload = {
    "schema": "magpie.model-revision-receipt/v1",
    "model": model,
    "requested_revision": requested,
    "resolved_revision": resolved,
    "snapshot_path": str(snapshot),
    "verified": True,
}
try:
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, receipt)
except Exception:
    temporary.unlink(missing_ok=True)
    raise
PY
  receipt_status=$?
  if [[ $receipt_status -ne 0 ]]; then
    echo "ERROR: failed to verify or persist model revision receipt." >&2
    exit "$receipt_status"
  fi
elif [[ "$PHASE" != "client" ]]; then
  # Keep legacy unpinned benchmarks working, but never leave a stale receipt
  # that a report consumer could mistake for evidence for the current run.
  if ! rm -f "$MODEL_REVISION_RECEIPT"; then
    echo "ERROR: could not clear stale model revision receipt." >&2
    exit 4
  fi
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

# vLLM optimizations for MI355X
export VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-1}

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
  setsid vllm serve "$MODEL" --port "$PORT" \
    "${MODEL_REVISION_ARGS[@]}" \
    --tensor-parallel-size=$TP \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len $MAX_MODEL_LEN \
    --trust-remote-code \
    "${PROFILER_ARGS[@]}" \
    $EXTRA_VLLM_ARGS > $SERVER_LOG 2>&1 &

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

if [[ "$PHASE" != "server" && "${RUN_EVAL,,}" = "true" ]]; then
    if [[ -n "${BENCHMARK_BASE_URL:-}" ]]; then
        if declare -F magpie_run_eval_remote_direct &>/dev/null; then
            magpie_run_eval_remote_direct || exit $?
        else
            echo "[vllm_mi355x] RUN_EVAL=true with BENCHMARK_BASE_URL but magpie_run_eval_remote_direct shim not available; skipping eval (results gate will see accuracy=None)."
        fi
    else
        magpie_mark_lm_eval_start || exit $?
        if [[ -n "${MAGPIE_EVAL_POLICY_ID:-}" ]]; then
            EVAL_CONCURRENT_REQUESTS="${EVAL_CONCURRENT_REQUESTS:-$CONC}" \
                magpie_run_lm_eval --port "$PORT" || exit $?
        else
            EVAL_CONCURRENT_REQUESTS="${EVAL_CONCURRENT_REQUESTS:-$CONC}" \
                run_eval --framework lm-eval --port "$PORT" || exit $?
        fi
        magpie_preserve_lm_eval_artifacts || exit $?
        append_lm_eval_summary
        magpie_preserve_lm_eval_artifacts || exit $?
    fi
fi
set +x
