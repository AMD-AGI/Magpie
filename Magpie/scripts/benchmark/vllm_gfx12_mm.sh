#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Magpie vLLM random multimodal benchmark for AMD gfx12.

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
  PHASE=client
fi

if [[ "$PHASE" == "server" || "$PHASE" == "all" ]]; then
  check_env_vars MODEL TP
fi
if [[ "$PHASE" == "client" || "$PHASE" == "all" ]]; then
  check_env_vars MODEL CONC ISL OSL RESULT_FILENAME
fi

MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.0}
REQUEST_RATE=${REQUEST_RATE:-inf}
NUM_PROMPTS=${NUM_PROMPTS:-$(( CONC * 10 ))}
MM_BASE_ITEMS_PER_REQUEST=${MM_BASE_ITEMS_PER_REQUEST:-1}
MM_ITEMS_RANGE_RATIO=${MM_ITEMS_RANGE_RATIO:-0.0}
MM_LIMIT_IMAGES=${MM_LIMIT_IMAGES:-3}
MM_LIMIT_VIDEOS=${MM_LIMIT_VIDEOS:-0}
MM_IMAGE_HEIGHT=${MM_IMAGE_HEIGHT:-256}
MM_IMAGE_WIDTH=${MM_IMAGE_WIDTH:-256}
MM_IMAGE_NUM_FRAMES=${MM_IMAGE_NUM_FRAMES:-1}
SEED=${SEED:-42}
RUN_EVAL_VALUE=${RUN_EVAL:-false}
RUN_EVAL_VALUE=${RUN_EVAL_VALUE,,}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

if [[ "$PHASE" != "client" ]]; then
  hf download "$MODEL" 2>/dev/null || true
fi

if [[ -n "${ROCR_VISIBLE_DEVICES:-}" && -z "${HIP_VISIBLE_DEVICES:-}" ]]; then
  n=$(awk -F, '{print NF}' <<< "$ROCR_VISIBLE_DEVICES")
  export HIP_VISIBLE_DEVICES
  HIP_VISIBLE_DEVICES=$(seq -s, 0 $((n-1)))
fi

unset HSA_OVERRIDE_GFX_VERSION
export VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-0}
export FLASH_ATTENTION_TRITON_AMD_ENABLE=${FLASH_ATTENTION_TRITON_AMD_ENABLE:-TRUE}

WORKSPACE_DIR=${RESULT_DIR:-/workspace}
SERVER_LOG=${SERVER_LOG:-$WORKSPACE_DIR/server.log}
PORT=${PORT:-8888}

write_command() {
  local output_path=$1
  shift
  {
    printf '#!/usr/bin/env bash\n'
    printf '%q ' "$@"
    printf '\n'
  } > "$output_path"
}

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
    read -r -a EXTRA_SERVER_ARGS <<< "$EXTRA_VLLM_ARGS"
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
  write_command "$WORKSPACE_DIR/server_command.sh" "${SERVER_CMD[@]}"
  python3 -c 'import json, platform, torch, vllm; print(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "vllm": vllm.__version__, "hip": torch.version.hip}, indent=2))' \
    > "$WORKSPACE_DIR/runtime_versions.json"

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

if [[ "$PHASE" == "client" || "$PHASE" == "all" ]]; then
  BASE_URL="${BENCHMARK_BASE_URL:-http://localhost:${PORT}}"
  RESULT_FILE="${RESULT_FILENAME%.json}.json"
  NUM_WARMUPS_VAL=${NUM_WARMUPS:-$(( CONC < 8 ? CONC : 8 ))}
  CLIENT_CMD=(
    vllm bench serve
    --model "$MODEL"
    --backend openai-chat
    --endpoint /v1/chat/completions
    --base-url "$BASE_URL"
    --dataset-name random-mm
    --num-prompts "$NUM_PROMPTS"
    --max-concurrency "$CONC"
    --random-input-len "$ISL"
    --random-output-len "$OSL"
    --random-range-ratio "$RANDOM_RANGE_RATIO"
    --random-mm-base-items-per-request "$MM_BASE_ITEMS_PER_REQUEST"
    --random-mm-num-mm-items-range-ratio "$MM_ITEMS_RANGE_RATIO"
    --random-mm-limit-mm-per-prompt "{\"image\": ${MM_LIMIT_IMAGES}, \"video\": ${MM_LIMIT_VIDEOS}}"
    --random-mm-bucket-config "{(${MM_IMAGE_HEIGHT}, ${MM_IMAGE_WIDTH}, ${MM_IMAGE_NUM_FRAMES}): 1.0}"
    --request-rate "$REQUEST_RATE"
    --num-warmups "$NUM_WARMUPS_VAL"
    --percentile-metrics ttft,tpot,itl,e2el
    --ignore-eos
    --seed "$SEED"
  )
  if [[ "${PROFILE:-}" == "1" ]]; then
    CLIENT_CMD+=(--profile)
  fi
  CLIENT_CMD+=(
    --save-result
    --result-dir "$WORKSPACE_DIR/"
    --result-filename "$RESULT_FILE"
    --trust-remote-code
  )
  write_command "$WORKSPACE_DIR/client_command.sh" "${CLIENT_CMD[@]}"
  "${CLIENT_CMD[@]}" || exit $?
fi

if [[ "$PHASE" != "server" && "$RUN_EVAL_VALUE" == "true" ]]; then
  if [[ -n "${BENCHMARK_BASE_URL:-}" ]]; then
    magpie_run_eval_remote_direct || exit $?
  else
    export EVAL_CONCURRENT_REQUESTS="${MAGPIE_EVAL_CONCURRENCY:-${EVAL_CONCURRENT_REQUESTS:-$CONC}}"
    magpie_run_eval_persisted --framework lm-eval --port "$PORT" || exit $?
  fi
fi
set +x
