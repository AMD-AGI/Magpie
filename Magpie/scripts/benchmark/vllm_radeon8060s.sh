#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Magpie Generic vLLM Benchmark Script for Radeon 8060S / gfx1151
#
# Phases (via MAGPIE_RUN_PHASE): all | server | client (default all).
# Server-only waits when MAGPIE_KEEP_CONTAINER_ALIVE=1 so a detached Docker
# container owns the server lifecycle; local reuse continues to disown it.
#
# Remote server (BENCHMARK_BASE_URL): when set, the client phase points
# benchmark_serving at an external vLLM-compatible HTTP endpoint
# instead of localhost:$PORT, and forces PHASE=client (no local server
# launch). See vllm_mi300x.sh for the full contract.

source "$(dirname "$0")/benchmark_lib.sh"
source "$(dirname "$0")/server_cleanup.sh"
# shellcheck source=magpie_bench_remote_compat.sh
source "$(dirname "$0")/magpie_bench_remote_compat.sh"

PHASE="${MAGPIE_RUN_PHASE:-all}"
case "$PHASE" in
  all|server|client) ;;
  *) echo "ERROR: Invalid MAGPIE_RUN_PHASE='$PHASE'. Must be all|server|client." >&2; exit 2 ;;
esac

if [[ -n "${BENCHMARK_BASE_URL:-}" ]]; then
  if [[ "$PHASE" != "client" ]]; then
    echo "[vllm_radeon8060s] BENCHMARK_BASE_URL set; forcing PHASE=client (was $PHASE)"
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
# 0.7 rather than the MI scripts' 0.95: leaves headroom for profiler buffers
# under PROFILE=1. Throughput-only runs can raise it.
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.7}

# Multimodal: MM_MAX_IMAGES>0 benchmarks text+image via vLLM's random-mm
# dataset; 0 keeps the text-only path. Folded in here rather than shipped as a
# separate vllm_radeon8060s_mm.sh because the caller composes the script name
# as {framework}_{runner_type}.sh and has no way to ask for an _mm variant.
MM_MAX_IMAGES=${MM_MAX_IMAGES:-0}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-512}
IMAGE_WIDTH=${IMAGE_WIDTH:-512}
SEED=${SEED:-0}
for numeric in TP CONC ISL OSL MAX_MODEL_LEN PORT \
               MM_MAX_IMAGES IMAGE_HEIGHT IMAGE_WIDTH SEED; do
  value=${!numeric:-}
  if [[ -n "$value" && ! "$value" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $numeric must be an unsigned integer, got '$value'." >&2
    exit 2
  fi
done
if [[ "$MODEL" == -* ]]; then
  echo "ERROR: MODEL cannot begin with '-'." >&2
  exit 2
fi

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [[ "$PHASE" != "client" ]]; then
  hf download "$MODEL" 2>/dev/null || true
fi

# PYTORCH_ROCM_ARCH is build metadata, not a runtime GPU selector. Published
# ROCm images may list multiple targets; preserve a list containing gfx1151.
if [[ -n "${PYTORCH_ROCM_ARCH:-}" && ";$PYTORCH_ROCM_ARCH;" != *";gfx1151;"* ]]; then
  echo "ERROR: vllm_radeon8060s requires PYTORCH_ROCM_ARCH=gfx1151 or a semicolon-separated list containing gfx1151." >&2
  exit 2
fi
# Retain the default build hint when absent; this does not change kernel dispatch.
if [[ -z "${PYTORCH_ROCM_ARCH:-}" ]]; then
  export PYTORCH_ROCM_ARCH=gfx1151
fi
export HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA:-0}
unset HSA_OVERRIDE_GFX_VERSION

# ROCR_VISIBLE_DEVICES already re-indexes visible GPUs to 0..N-1, so HIP
# must use the logical range, not the original physical ids.
if [ -n "$ROCR_VISIBLE_DEVICES" ] && [ -z "$HIP_VISIBLE_DEVICES" ]; then
    n=$(echo "$ROCR_VISIBLE_DEVICES" | awk -F, '{print NF}')
    export HIP_VISIBLE_DEVICES=$(seq -s, 0 $((n-1)))
fi

# AITER attention is not qualified for the tested gfx1151 geometry.
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_AITER_RMSNORM=0
unset VLLM_USE_AITER

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

read -r -a EXTRA_VLLM_ARGV <<< "${EXTRA_VLLM_ARGS:-}"

set -x
if [[ "$PHASE" == "server" || "$PHASE" == "all" ]]; then
  SERVER_CMD=(
    vllm serve "$MODEL"
    --port "$PORT"
    --tensor-parallel-size "$TP"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --trust-remote-code
  )
  SERVER_CMD+=("${PROFILER_ARGS[@]}")
  SERVER_CMD+=("${EXTRA_VLLM_ARGV[@]}")
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
  if [[ "${MM_MAX_IMAGES}" -gt 0 ]]; then
    # InferenceX's run_benchmark_serving has no multimodal dataset, and neither
    # does the remote-direct shim, so drive `vllm bench serve` directly. One
    # image-size bucket gives every request the same IMAGE_HEIGHT x IMAGE_WIDTH.
    BASE_URL="${BENCHMARK_BASE_URL:-http://localhost:${PORT}}"
    NUM_PROMPTS_VAL=${NUM_PROMPTS:-$(( CONC * 10 ))}
    NUM_WARMUPS_VAL=${NUM_WARMUPS:-$(( CONC < 8 ? CONC : 8 ))}
    vllm bench serve \
        --backend openai-chat \
        --endpoint /v1/chat/completions \
        --base-url "$BASE_URL" \
        --model "$MODEL" \
        --dataset-name random-mm \
        --random-input-len "$ISL" \
        --random-output-len "$OSL" \
        --random-mm-base-items-per-request 1 \
        --random-mm-limit-mm-per-prompt "{\"image\": ${MM_MAX_IMAGES}, \"video\": 0}" \
        --random-mm-bucket-config "{(${IMAGE_HEIGHT}, ${IMAGE_WIDTH}, 1): 1.0}" \
        --num-prompts "$NUM_PROMPTS_VAL" \
        --num-warmups "$NUM_WARMUPS_VAL" \
        --max-concurrency "$CONC" \
        --ignore-eos \
        --seed "$SEED" \
        --save-result \
        --result-dir "$WORKSPACE_DIR/" \
        --result-filename "${RESULT_FILENAME}.json" \
        --trust-remote-code || exit $?
  elif [[ -n "${BENCHMARK_BASE_URL:-}" ]]; then
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
            echo "[vllm_radeon8060s] RUN_EVAL=true with BENCHMARK_BASE_URL but magpie_run_eval_remote_direct shim not available; skipping eval (results gate will see accuracy=None)."
        fi
    else
        export EVAL_CONCURRENT_REQUESTS="${MAGPIE_EVAL_CONCURRENCY:-${EVAL_CONCURRENT_REQUESTS:-$CONC}}"
        magpie_run_eval_persisted --framework lm-eval --port "$PORT" || exit $?
    fi
fi
set +x
