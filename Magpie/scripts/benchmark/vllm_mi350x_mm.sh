#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Magpie Generic vLLM Multimodal Benchmark Script for MI350X
#
# Variant of vllm_mi350x.sh for vision-language (VL) models.  The server
# phase is identical; the client phase uses ``vllm bench serve`` with
# ``--dataset-name random-mm`` to generate synthetic multimodal requests
# (text + one synthetic image per request) instead of InferenceX's
# text-only ``run_benchmark_serving``.
#
# New env vars (all optional with sensible defaults):
#   IMAGE_HEIGHT  — synthetic image height in pixels  (default 512)
#   IMAGE_WIDTH   — synthetic image width in pixels   (default 512)
#   MM_MAX_IMAGES — max images per request            (default 1)
#   SEED          — random seed for reproducibility   (default 5678)
#
# All standard env vars from vllm_mi350x.sh still apply:
#   MODEL, TP, CONC, ISL, OSL, RANDOM_RANGE_RATIO, RESULT_FILENAME,
#   MAX_MODEL_LEN, EXTRA_VLLM_ARGS, MAGPIE_RUN_PHASE, etc.
#
# Phases (via MAGPIE_RUN_PHASE): all | server | client (default all).
# Server-only writes PID to MAGPIE_SERVER_PID_FILE then disowns and exits.

source "$(dirname "$0")/benchmark_lib.sh"
source "$(dirname "$0")/server_cleanup.sh"
# shellcheck source=magpie_bench_remote_compat.sh
[[ -f "$(dirname "$0")/magpie_bench_remote_compat.sh" ]] && source "$(dirname "$0")/magpie_bench_remote_compat.sh"

# Fallback: some server_cleanup.sh variants (e.g. InferenceX's) do not define
# magpie_stop_benchmark_server_stack. Provide a safe inline version so the
# EXIT/INT/TERM trap always fires correctly regardless of which library was sourced.
if ! declare -F magpie_stop_benchmark_server_stack &>/dev/null; then
  magpie_stop_benchmark_server_stack() {
    local pid="${1:-}"
    [[ -z "$pid" ]] && return 0
    # Kill the entire process group (setsid gives the server its own pgid).
    kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    # Give it up to 15 s to exit cleanly before SIGKILL.
    local deadline=$(( $(date +%s) + 15 ))
    while kill -0 "$pid" 2>/dev/null && [[ $(date +%s) -lt $deadline ]]; do
      sleep 1
    done
    kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  }
fi

PHASE="${MAGPIE_RUN_PHASE:-all}"
case "$PHASE" in
  all|server|client) ;;
  *) echo "ERROR: Invalid MAGPIE_RUN_PHASE='$PHASE'. Must be all|server|client." >&2; exit 2 ;;
esac

if [[ -n "${BENCHMARK_BASE_URL:-}" ]]; then
  if [[ "$PHASE" != "client" ]]; then
    echo "[vllm_mi350x_mm] BENCHMARK_BASE_URL set; forcing PHASE=client (was $PHASE)"
    PHASE=client
  fi
fi

if [[ "$PHASE" == "server" || "$PHASE" == "all" ]]; then
  check_env_vars MODEL TP
fi
if [[ "$PHASE" == "client" || "$PHASE" == "all" ]]; then
  check_env_vars MODEL CONC ISL OSL RESULT_FILENAME
fi

MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-512}
IMAGE_WIDTH=${IMAGE_WIDTH:-512}
MM_MAX_IMAGES=${MM_MAX_IMAGES:-1}
SEED=${SEED:-5678}

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [[ "$PHASE" != "client" ]]; then
  hf download "$MODEL" 2>/dev/null || true
fi

# MI350X specific: Check MEC firmware version for RCCL memory reclaim
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

# vLLM optimizations for MI350X
export VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-1}

WORKSPACE_DIR=${RESULT_DIR:-/workspace}
SERVER_LOG=${SERVER_LOG:-$WORKSPACE_DIR/server.log}
PORT=${PORT:-8888}

# Build profiler args for vLLM >= 0.15
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
  setsid vllm serve $MODEL --port $PORT \
    --tensor-parallel-size=$TP \
    --gpu-memory-utilization 0.95 \
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

if [[ "$PHASE" == "client" || "$PHASE" == "all" ]]; then
  BASE_URL="${BENCHMARK_BASE_URL:-http://localhost:${PORT}}"
  NUM_PROMPTS_VAL=${NUM_PROMPTS:-$(( CONC * 10 ))}
  NUM_WARMUPS_VAL=${NUM_WARMUPS:-$(( CONC < 8 ? CONC : 8 ))}

  # Use vllm bench serve with random-mm dataset for multimodal workloads.
  # --random-mm-bucket-config selects a single image size bucket so all
  # requests use the same (IMAGE_HEIGHT × IMAGE_WIDTH) synthetic image.
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

  # No post-processing: vllm bench serve emits the same output JSON schema
  # regardless of input modality (images change only the input, not the
  # metrics). Hyperloom's canonical parser (benchmark_result.py) consumes the
  # raw vllm keys (output_throughput, mean_ttft_ms, mean_tpot_ms, mean_e2el_ms,
  # p99_*, completed, duration) directly — identical to the text-only path.
fi
set +x
