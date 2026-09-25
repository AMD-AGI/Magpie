#!/usr/bin/env bash
###############################################################################
# Remote benchmark compat (Magpie).
#
# InferenceX benchmarks/benchmark_lib.sh defines run_benchmark_serving() which
# parses a fixed set of flags. Older (and some current) trees reject
# --base-url at the bash layer even though utils/bench_serving/benchmark_serving.py
# accepts --base-url for OpenAI-compatible servers (SGLang/vLLM HTTP).
#
# When BENCHMARK_BASE_URL is set, Magpie *mi*.sh scripts call
# magpie_run_benchmark_serving_remote_direct() instead of passing --base-url into
# run_benchmark_serving().
#
# Working directory must be the InferenceX repository root (Magpie benchmarker
# runs: cd <inferencex> && bash benchmarks/<script>.sh). Override with
# MAGPIE_INFERENCEX_ROOT if needed.
###############################################################################

magpie_run_benchmark_serving_remote_direct() {
  local trust_mode="${1:-}"

  local inferx_root="${MAGPIE_INFERENCEX_ROOT:-$(pwd)}"
  local bench_py="$inferx_root/utils/bench_serving/benchmark_serving.py"
  if [[ ! -f "$bench_py" ]]; then
    echo "[magpie_bench_remote_compat] ERROR: missing $bench_py (pwd=$(pwd)). " \
      "Set MAGPIE_INFERENCEX_ROOT to your InferenceX checkout root." >&2
    return 1
  fi

  local py="${MAGPIE_BENCHMARK_PYTHON:-python3}"
  local result_dir="${RESULT_DIR:-${WORKSPACE_DIR:-/workspace}}"
  local num_prompts="${NUM_PROMPTS:-$(( CONC * 10 ))}"
  local num_warmups="$((2 * CONC))"
  local -a profile_args=()
  if [[ "${PROFILE:-}" == "1" ]]; then
    profile_args+=(--profile)
    num_prompts="$CONC"
  fi

  # Backend + endpoint are configurable so the same remote-bench path serves
  # both OpenAI completions (prompt) and chat (messages) servers — e.g. a SaFE
  # dynamo frontend exposes both /v1/completions and /v1/chat/completions.
  #   MAGPIE_BENCHMARK_BACKEND : vllm|openai (completions, default) | openai-chat (chat)
  #   MAGPIE_BENCHMARK_ENDPOINT: overrides the path; when unset it defaults to
  #                              the one matching the backend.
  # Default backend stays "vllm" so every existing caller is bit-for-bit
  # unchanged.
  local backend="${MAGPIE_BENCHMARK_BACKEND:-vllm}"
  local default_endpoint="/v1/completions"
  if [[ "$backend" == "openai-chat" ]]; then
    default_endpoint="/v1/chat/completions"
  fi
  local endpoint="${MAGPIE_BENCHMARK_ENDPOINT:-$default_endpoint}"

  local -a cmd=(
    "$py" "$bench_py"
    --model "$MODEL"
    --backend "$backend"
    --base-url "${BENCHMARK_BASE_URL}"
    --endpoint "$endpoint"
    --dataset-name random
    --random-input-len "$ISL"
    --random-output-len "$OSL"
    --random-range-ratio "$RANDOM_RANGE_RATIO"
    --num-prompts "$num_prompts"
    --max-concurrency "$CONC"
    --request-rate inf
    --ignore-eos
    "${profile_args[@]}"
    --save-result
    --num-warmups "$num_warmups"
    --percentile-metrics "ttft,tpot,itl,e2el"
    --result-dir "$result_dir"
    --result-filename "${RESULT_FILENAME}.json"
  )

  if [[ "$trust_mode" == "trust" ]]; then
    cmd+=(--trust-remote-code)
  fi

  set -x
  "${cmd[@]}"
  local rc=$?
  set +x

  if [[ "${PROFILE:-}" == "1" ]] && declare -F move_profile_trace_for_relay &>/dev/null; then
    move_profile_trace_for_relay
  fi
  return "$rc"
}

###############################################################################
# magpie_eval_needs_unsafe_code
#
# lm-eval refuses HumanEval (and other tasks with unsafe_code: true) unless
# --confirm_run_unsafe_code is set. Enable that flag when MAGPIE_EVAL_TASKS
# contains humaneval*, or when MAGPIE_EVAL_CONFIRM_UNSAFE_CODE is true/1.
###############################################################################
magpie_eval_needs_unsafe_code() {
  case "${MAGPIE_EVAL_CONFIRM_UNSAFE_CODE:-}" in
    1|true|TRUE|yes|YES) return 0 ;;
  esac
  local tasks="${MAGPIE_EVAL_TASKS:-}"
  [[ ",${tasks}," == *humaneval* ]]
}

###############################################################################
# magpie_eval_tasks_look_builtin
#
# Builtin harness names (gsm8k, mmlu, humaneval_instruct) must not take
# --include_path pointing at InferenceX utils/evals. Those YAMLs shadow the
# builtins and GPQA then fails on utils.process_docs.
###############################################################################
magpie_eval_tasks_look_builtin() {
  local tasks="${MAGPIE_EVAL_TASKS:-}"
  [[ -n "$tasks" && "$tasks" != *".yaml"* && "$tasks" != *"/"* ]]
}

###############################################################################
# magpie_eval_tasks_cli_mode
#
# lm-eval 0.4.8 takes one comma-separated --tasks string. The harness in
# lmsysorg/sglang:v0.5.18-rocm700-mi35x takes nargs="+" and does not split
# commas. MAGPIE_EVAL_TASKS_CLI=words|comma overrides detection.
###############################################################################
magpie_eval_tasks_cli_mode() {
  local override="${MAGPIE_EVAL_TASKS_CLI:-}"
  override="$(printf '%s' "$override" | tr '[:upper:]' '[:lower:]')"
  case "$override" in
    words|argv|multi) printf 'words\n'; return 0 ;;
    comma|string) printf 'comma\n'; return 0 ;;
  esac
  local py="${MAGPIE_EVAL_PYTHON:-python3}"
  local help_text
  help_text="$("$py" -m lm_eval --help 2>&1 || true)"
  if [[ "$help_text" == *"[TASKS ...]"* || "$help_text" == *"[TASK ...]"* ]]; then
    printf 'words\n'
    return 0
  fi
  printf 'comma\n'
}

magpie_eval_skip_stock_include() {
  local include="${1%/}"
  magpie_eval_tasks_look_builtin || return 1
  case "$include" in
    utils/evals|*/utils/evals|utils/evals/*|*/utils/evals/*) return 0 ;;
  esac
  return 1
}

###############################################################################
# magpie_eval_task_needs_stock_include
#
# --include_path is global, but the stock InferenceX YAMLs suit only some
# tasks. gsm8k needs them: the shipped builtin still points at the bare
# `gsm8k` dataset id that newer huggingface_hub rejects. gpqa must not see
# them: InferenceX's own `utils` package then hides the task's process_docs.
# Decide per task so one suite can mix both.
###############################################################################
magpie_eval_task_needs_stock_include() {
  local task="$1"
  local allow="${MAGPIE_EVAL_STOCK_INCLUDE_TASKS:-gsm8k}"
  local item
  local IFS=','
  # shellcheck disable=SC2206
  local -a _allow=(${allow})
  unset IFS
  for item in "${_allow[@]}"; do
    if [[ "$task" == "${item// /}" ]]; then
      return 0
    fi
  done
  return 1
}

###############################################################################
# magpie_eval_apply_code_eval_env
#
# HumanEval's Hugging Face code_eval metric also requires HF_ALLOW_CODE_EVAL=1
# at task-load time, independent of --confirm_run_unsafe_code.
###############################################################################
magpie_eval_apply_code_eval_env() {
  if magpie_eval_needs_unsafe_code; then
    export HF_ALLOW_CODE_EVAL=1
  fi
}

###############################################################################
# magpie_eval_prepare_deps
#
# InferenceX run_lm_eval installs packages with _install_lm_eval_deps and then
# applies _patch_lm_eval before launching the evaluator. Magpie-owned
# invocations skip run_lm_eval, so run those same hooks when they are sourced.
# If they are absent, install lm-eval into MAGPIE_EVAL_PYTHON.
###############################################################################
magpie_eval_prepare_deps() {
  local py="${MAGPIE_EVAL_PYTHON:-python3}"
  if declare -F _install_lm_eval_deps &>/dev/null; then
    _install_lm_eval_deps
  fi
  if declare -F _patch_lm_eval &>/dev/null; then
    _patch_lm_eval
  fi
  if [[ "${MAGPIE_EVAL_SKIP_DEPS:-}" == "1" ]]; then
    return 0
  fi
  if ! "$py" -c "import sys" >/dev/null 2>&1; then
    return 0
  fi
  if "$py" -c "import lm_eval" >/dev/null 2>&1; then
    return 0
  fi
  echo "[magpie_bench_remote_compat] lm-eval missing for $py; installing lm-eval[api]" >&2
  "$py" -m pip install --quiet "lm-eval[api]"
}

###############################################################################
# magpie_eval_context_length / magpie_eval_generation_budget
#
# InferenceX run_lm_eval passes max_length from EVAL_MAX_MODEL_LEN and a
# generation budget via --gen_kwargs max_tokens (context minus 4096, cap
# 16384). Unset, local-completions defaults to 2048 / 256.
###############################################################################
magpie_eval_context_length() {
  local max_length="${EVAL_MAX_MODEL_LEN:-${MAGPIE_EVAL_MAX_LENGTH:-${MAX_MODEL_LEN:-16384}}}"
  printf '%s\n' "$max_length"
}

magpie_eval_generation_budget() {
  local max_length="$1"
  local explicit="${EVAL_MAX_GEN_TOKS:-${MAGPIE_EVAL_MAX_GEN_TOKS:-}}"
  if [[ -n "$explicit" ]]; then
    printf '%s\n' "$explicit"
    return 0
  fi
  if [[ ! "$max_length" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s\n' "256"
    return 0
  fi
  local max_output_tokens
  if [[ "$max_length" -gt 4096 ]]; then
    max_output_tokens=$((max_length - 4096))
  else
    max_output_tokens=$((max_length / 2))
  fi
  if [[ "$max_output_tokens" -gt 16384 ]]; then
    max_output_tokens=16384
  fi
  printf '%s\n' "$max_output_tokens"
}

###############################################################################
# magpie_eval_model_args
#
# local-completions defaults to 2048 context / 256 generated tokens unless the
# task overrides generation. Carry EVAL_MAX_MODEL_LEN and the generation budget
# so long prompts are not truncated.
###############################################################################
magpie_eval_model_args() {
  local conc="$1"
  local base_url="$2"
  local max_length="$3"
  local max_gen_toks="$4"
  local args="model=${MODEL},base_url=${base_url},num_concurrent=${conc},tokenizer_backend=huggingface,trust_remote_code=true,max_length=${max_length},max_gen_toks=${max_gen_toks}"
  if [[ -n "${MAGPIE_EVAL_TOKENIZED_REQUESTS:-}" ]]; then
    args+=",tokenized_requests=${MAGPIE_EVAL_TOKENIZED_REQUESTS}"
  fi
  printf '%s\n' "$args"
}

###############################################################################
# magpie_prepare_eval_include_and_limit
#
# Optional include path and sample limit for Magpie-owned lm-eval invocations.
# Resolve relative directories to an absolute path here: magpie_run_eval_persisted
# later cds into a temp results directory, which would otherwise hide a
# relative MAGPIE_EVAL_INCLUDE_PATH such as custom-tasks.
###############################################################################
magpie_prepare_eval_include_and_limit() {
  if [[ -z "${EVAL_LIMIT:-}" && -n "${MAGPIE_EVAL_LIMIT:-}" ]]; then
    export EVAL_LIMIT="$MAGPIE_EVAL_LIMIT"
  fi
  local include="${MAGPIE_EVAL_INCLUDE_PATH:-${MAGPIE_EVAL_TASK_PATH:-}}"
  local stock=0
  if magpie_eval_skip_stock_include "$include"; then
    stock=1
  fi
  local resolved=""
  if [[ -n "$include" && "$include" != *","* ]]; then
    if [[ -d "$include" ]]; then
      resolved="$(cd "$include" && pwd)"
    elif [[ -f "$include" ]]; then
      resolved="$(cd "$(dirname "$include")" && pwd)"
    fi
  fi
  if [[ -z "$resolved" ]]; then
    return 0
  fi
  # Keep the stock directory available to the tasks that need it rather than
  # discarding it for the whole suite.
  if [[ "$stock" -eq 1 ]]; then
    export MAGPIE_EVAL_STOCK_INCLUDE_PATH="$resolved"
  else
    export EVAL_INCLUDE_PATH="$resolved"
  fi
}

magpie_eval_port_from_args() {
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--port" && -n "${2:-}" && "$2" != --* ]]; then
      printf '%s\n' "$2"
      return 0
    fi
    shift
  done
  printf '%s\n' "${PORT:-8888}"
}

magpie_eval_concurrency_values() {
  local raw="${MAGPIE_EVAL_CONCURRENCY:-${EVAL_CONCURRENT_REQUESTS:-${CONC:-8}}}"
  raw="${raw//,/ }"
  # shellcheck disable=SC2086
  printf '%s\n' $raw
}

###############################################################################
# magpie_run_eval_concurrency_loop
#
# Run magpie_run_lm_eval once per requested concurrency. Keep going after a
# failure so later concs still produce artifacts, but record completed vs
# failed lists instead of marking every requested value as completed.
###############################################################################
magpie_run_eval_concurrency_loop() {
  local out_dir="$1"
  local base_url="$2"
  local conc
  local requested_concs=""
  local completed_concs=""
  local failed_concs=""
  local eval_rc=0
  local rc=0

  while read -r conc; do
    [[ -z "$conc" ]] && continue
    requested_concs+="${requested_concs:+ }${conc}"
    rc=0
    magpie_run_lm_eval "$out_dir" "$conc" "$base_url" || rc=$?
    if [[ "$rc" -eq 0 ]]; then
      completed_concs+="${completed_concs:+ }${conc}"
    else
      eval_rc="$rc"
      failed_concs+="${failed_concs:+ }${conc}"
    fi
  done < <(magpie_eval_concurrency_values)

  export EVAL_BATCHED_CONCS="$requested_concs"
  export EVAL_BATCHED_COMPLETED_CONCS="$completed_concs"
  export EVAL_BATCHED_FAILED_CONCS="$failed_concs"
  return "$eval_rc"
}

###############################################################################
# magpie_publish_conc_result
#
# lm-eval treats --output_path as a directory and writes results_*.json inside
# it. Downstream gates match concurrency from the basename (results_conc8.json),
# so copy the newest result out of the per-concurrency directory.
###############################################################################
magpie_publish_conc_result() {
  local source_dir="$1"
  local dest_dir="$2"
  local conc="$3"
  local dest="${dest_dir%/}/results_conc${conc}.json"
  local newest="" path

  [[ -d "$source_dir" ]] || return 0
  mkdir -p "$dest_dir" || return 1

  while IFS= read -r -d '' path; do
    if [[ "$(basename "$path")" == "accuracy_report.json" ]]; then
      continue
    fi
    if [[ -z "$newest" || "$path" -nt "$newest" ]]; then
      newest="$path"
    fi
  done < <(find "$source_dir" -type f -name '*.json' -print0 2>/dev/null)

  if [[ -z "$newest" ]]; then
    return 0
  fi
  cp -p "$newest" "$dest" || return 1
}

###############################################################################
# magpie_merge_conc_results
#
# A suite that mixes stock-include and plain tasks runs lm-eval twice, so the
# concurrency directory holds one results file per invocation. Downstream
# gates expect a single results_conc<N>.json, so merge the task maps instead
# of letting magpie_publish_conc_result keep only the newest file.
###############################################################################
magpie_merge_conc_results() {
  local source_dir="$1"
  local dest_dir="$2"
  local conc="$3"
  local py="${MAGPIE_ACCURACY_REPORT_PYTHON:-python3}"

  [[ -d "$source_dir" ]] || return 1
  mkdir -p "$dest_dir" || return 1
  "$py" - "$source_dir" "${dest_dir%/}/results_conc${conc}.json" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
dest = Path(sys.argv[2])
per_task_keys = ("results", "n-samples", "configs", "versions", "higher_is_better")

merged = {}
found = False
for path in sorted(source.rglob("*.json"), key=lambda item: item.stat().st_mtime):
    if path == dest or path.name == "accuracy_report.json":
        continue
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
        continue
    found = True
    for key, value in payload.items():
        if key in per_task_keys and isinstance(value, dict):
            merged.setdefault(key, {}).update(value)
        else:
            merged.setdefault(key, value)

if not found:
    sys.exit(1)
dest.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
PY
}

###############################################################################
# magpie_run_lm_eval
#
# Drive lm-eval with the local-completions backend so multiple-choice tasks
# that need loglikelihood (mmlu, hellaswag) work against an OpenAI-compatible
# /v1/completions server. InferenceX's local run_eval uses
# local-chat-completions, which raises NotImplementedError for those requests.
#
# Magpie builds the argv here so --confirm_run_unsafe_code is on the actual
# lm-eval process even when a later InferenceX server-watch wrapper would
# hide a Bash python3() intercept.
###############################################################################
magpie_run_lm_eval_invocation() {
  local conc_dir="$1"
  local model_args="$2"
  local gen_kwargs="$3"
  local batch_size="$4"
  local include="$5"
  local cli_mode="$6"
  shift 6
  local py="${MAGPIE_EVAL_PYTHON:-python3}"
  local -a task_flag
  if [[ "$cli_mode" == "words" ]]; then
    task_flag=(--tasks "$@")
  else
    local joined=""
    local item
    for item in "$@"; do
      [[ -z "$item" ]] && continue
      if [[ -n "$joined" ]]; then
        joined+=","
      fi
      joined+="$item"
    done
    task_flag=(--tasks "$joined")
  fi
  local -a cmd=(
    "$py" -m lm_eval
    --model local-completions
    "${task_flag[@]}"
    --model_args "$model_args"
    --gen_kwargs "$gen_kwargs"
    --batch_size "$batch_size"
    --output_path "$conc_dir"
  )
  local limit="${EVAL_LIMIT:-${MAGPIE_EVAL_LIMIT:-}}"
  if [[ -n "$limit" ]]; then
    cmd+=(--limit "$limit")
  fi
  if [[ -n "$include" ]]; then
    cmd+=(--include_path "$include")
  fi
  if magpie_eval_needs_unsafe_code; then
    cmd+=(--confirm_run_unsafe_code)
  fi

  echo "[magpie_bench_remote_compat] lm_eval cmd: ${cmd[*]}" >&2
  set -x
  "${cmd[@]}"
  local rc=$?
  set +x
  return "$rc"
}

magpie_run_lm_eval() {
  local out_dir="$1"
  local conc="$2"
  local base_url="$3"
  local tasks="${MAGPIE_EVAL_TASKS:-gsm8k}"
  tasks="${tasks// /}"
  local -a task_args=()
  local task_item
  local IFS=','
  # shellcheck disable=SC2206
  local -a _split=(${tasks})
  unset IFS
  for task_item in "${_split[@]}"; do
    [[ -n "$task_item" ]] && task_args+=("$task_item")
  done
  if [[ ${#task_args[@]} -eq 0 ]]; then
    task_args=("gsm8k")
  fi
  local cli_mode
  cli_mode="$(magpie_eval_tasks_cli_mode)"
  local batch_size="${MAGPIE_EVAL_BATCH_SIZE:-auto}"
  local conc_dir="${out_dir%/}/conc${conc}"
  mkdir -p "$conc_dir" || return 1
  local max_length max_gen_toks model_args gen_kwargs
  max_length="$(magpie_eval_context_length)"
  max_gen_toks="$(magpie_eval_generation_budget "$max_length")"
  model_args="$(magpie_eval_model_args "$conc" "$base_url" "$max_length" "$max_gen_toks")"
  gen_kwargs="max_tokens=${max_gen_toks},temperature=0,top_p=1"

  local stock_include="${MAGPIE_EVAL_STOCK_INCLUDE_PATH:-}"
  local -a stock_tasks=() plain_tasks=()
  for task_item in "${task_args[@]}"; do
    if [[ -n "$stock_include" ]] && magpie_eval_task_needs_stock_include "$task_item"; then
      stock_tasks+=("$task_item")
    else
      plain_tasks+=("$task_item")
    fi
  done

  local rc=0
  local group_rc=0
  local invocations=0
  if [[ ${#plain_tasks[@]} -gt 0 ]]; then
    invocations=$((invocations + 1))
    group_rc=0
    magpie_run_lm_eval_invocation "$conc_dir" "$model_args" "$gen_kwargs" \
      "$batch_size" "${EVAL_INCLUDE_PATH:-}" "$cli_mode" "${plain_tasks[@]}" || group_rc=$?
    if [[ "$group_rc" -ne 0 ]]; then
      rc="$group_rc"
    fi
  fi
  if [[ ${#stock_tasks[@]} -gt 0 ]]; then
    invocations=$((invocations + 1))
    group_rc=0
    magpie_run_lm_eval_invocation "$conc_dir" "$model_args" "$gen_kwargs" \
      "$batch_size" "$stock_include" "$cli_mode" "${stock_tasks[@]}" || group_rc=$?
    if [[ "$group_rc" -ne 0 ]]; then
      rc="$group_rc"
    fi
  fi

  if [[ "$invocations" -gt 1 ]]; then
    magpie_merge_conc_results "$conc_dir" "$out_dir" "$conc" || {
      if [[ "$rc" -ne 0 ]]; then
        return "$rc"
      fi
      return 1
    }
    return "$rc"
  fi
  magpie_publish_conc_result "$conc_dir" "$out_dir" "$conc" || {
    if [[ "$rc" -ne 0 ]]; then
      return "$rc"
    fi
    return 1
  }
  return "$rc"
}

###############################################################################
# magpie_run_eval_remote_direct
#
# Remote-server analogue of InferenceX run_eval (which only takes --port and
# always targets localhost). When BENCHMARK_BASE_URL is set, this shim drives
# lm-eval-harness directly at the remote OpenAI-compatible endpoint via
# `local-completions` model, writing results under $RESULT_DIR so the
# downstream `_accuracy_gate.parse_eval_results` finds them via the standard
# lm-eval `results*.json` schema (`{"results": {"<task>": {"exact_match,*"...}}}`).
#
# Inputs (env, must already be set by the calling mi*x.sh):
#   MODEL                 model id passed to lm-eval
#   BENCHMARK_BASE_URL    e.g. http://<head_pod_ip>:8888
#   CONC                  performance concurrency (fallback for accuracy)
#   RESULT_DIR            workspace dir; results land at $RESULT_DIR/lm_eval/
#
# Inputs (env, optional):
#   MAGPIE_EVAL_CONCURRENCY independent accuracy concurrency; falls back to
#                         EVAL_CONCURRENT_REQUESTS, then CONC, then 8
#   MAGPIE_EVAL_TASKS     comma-separated lm-eval task names (default: gsm8k)
#   MAGPIE_EVAL_LIMIT     int; cap samples for smoke runs (default: empty = full)
#   MAGPIE_EVAL_BATCH_SIZE size for lm-eval (default: auto)
#   MAGPIE_EVAL_PYTHON    interpreter (default: python3)
#
# Returns lm-eval's exit code; prints diagnostics on stderr; never overrides
# upstream lm-eval flags so future task adds are pure env tweaks.
###############################################################################
magpie_run_eval_remote_direct() {
  if [[ -z "${BENCHMARK_BASE_URL:-}" ]]; then
    echo "[magpie_bench_remote_compat] ERROR magpie_run_eval_remote_direct called without BENCHMARK_BASE_URL" >&2
    return 1
  fi

  magpie_eval_apply_code_eval_env
  magpie_prepare_eval_include_and_limit
  magpie_eval_prepare_deps || return $?

  local result_dir="${RESULT_DIR:-${WORKSPACE_DIR:-/workspace}}"
  local out_dir="${result_dir%/}/lm_eval"
  mkdir -p "$out_dir" || {
    echo "[magpie_bench_remote_compat] ERROR cannot mkdir $out_dir" >&2
    return 1
  }

  # local-completions hits an OpenAI-compatible /v1/completions endpoint.
  # MAGPIE_EVAL_TOKENIZED_REQUESTS (optional) controls the prompt wire format.
  # Unset => lm_eval's default (token-id-array prompts), which a direct sglang
  # server accepts. A PD-disaggregated sglang_router's /v1/completions only
  # accepts StringOrArray and rejects token-id arrays with HTTP 422; set
  # MAGPIE_EVAL_TOKENIZED_REQUESTS=false there to send string prompts instead.
  local base_url="${BENCHMARK_BASE_URL%/}/v1/completions"
  local eval_rc=0
  magpie_run_eval_concurrency_loop "$out_dir" "$base_url" || eval_rc=$?
  if [[ "$eval_rc" -ne 0 ]]; then
    echo "[magpie_bench_remote_compat] WARN lm_eval exited rc=$eval_rc; accuracy gate will see no results" >&2
  fi
  local report_rc=0
  magpie_write_accuracy_result "$out_dir" "$eval_rc" "$result_dir" || report_rc=$?
  if [[ "$eval_rc" -ne 0 ]]; then
    return "$eval_rc"
  fi
  return "$report_rc"
}

###############################################################################
# magpie_write_accuracy_result
#
# Convert the newest standard lm-eval result into a small, stable Magpie
# artifact. The raw lm-eval JSON remains alongside this file for consumers
# that need the complete result schema.
###############################################################################
magpie_write_accuracy_result() {
  local eval_dir="$1"
  local eval_rc="${2:-0}"
  local summary_dir="${3:-$eval_dir}"
  local py="${MAGPIE_ACCURACY_REPORT_PYTHON:-python3}"

  "$py" - "$eval_dir" "$eval_rc" "$summary_dir" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
eval_rc = int(sys.argv[2])
output_root = Path(sys.argv[3])
output = output_root / "accuracy_report.json"
priority = (
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "exact_match,extract_abcd",
    "acc_norm,none",
    "acc,none",
    "acc_norm",
    "acc",
    "pass@1,create_test",
    "pass@1,none",
    "pass@1",
)

candidates = []
for path in root.rglob("*.json"):
    if path == output:
        continue
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if isinstance(payload, dict) and isinstance(payload.get("results"), dict):
        candidates.append((path.stat().st_mtime, path, payload))

summary = {
    "schema_version": "1.0",
    "provider": "lm-eval",
    "status": "ERROR",
    "task": None,
    "metric": None,
    "score": None,
    "samples": None,
    "source_result": None,
    "tasks": {},
    "error": None,
    "created_at": datetime.now(timezone.utc).isoformat(),
}

if not candidates:
    summary["error"] = (
        f"lm-eval exited with code {eval_rc} and produced no result"
        if eval_rc
        else "lm-eval produced no result"
    )
else:
    _, source, payload = max(candidates, key=lambda item: item[0])
    sample_payload = payload.get("n-samples", {})
    for task, task_result in payload["results"].items():
        if not isinstance(task_result, dict):
            continue
        sample_info = sample_payload.get(task) if isinstance(sample_payload, dict) else None
        if isinstance(sample_info, dict):
            samples = sample_info.get("effective", sample_info.get("original"))
        else:
            samples = sample_info
        metrics = {
            key: value
            for key, value in task_result.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        summary["tasks"][task] = {"metrics": metrics, "samples": samples}

    if summary["tasks"]:
        requested_tasks = [
            item.strip()
            for item in os.environ.get("MAGPIE_EVAL_TASKS", "").split(",")
            if item.strip()
        ]
        task = next(
            (item for item in requested_tasks if item in summary["tasks"]),
            next(iter(summary["tasks"])),
        )
        metrics = summary["tasks"][task]["metrics"]
        metric = next((item for item in priority if item in metrics), None)
        summary.update(
            status="COMPLETED" if eval_rc == 0 else "ERROR",
            task=task,
            metric=metric,
            score=metrics.get(metric) if metric else None,
            samples=summary["tasks"][task]["samples"],
            source_result=None,
            error=None if eval_rc == 0 else f"lm-eval exited with code {eval_rc}",
        )
    else:
        summary["source_result"] = None
        summary["error"] = "lm-eval result contains no task metrics"

    try:
        summary["source_result"] = str(source.relative_to(output_root))
    except ValueError:
        summary["source_result"] = str(source)

output_root.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[magpie_bench_remote_compat] accuracy artifact: {output}", file=sys.stderr)
PY
}

###############################################################################
# magpie_write_batched_eval_meta
#
# Fallback when InferenceX append_lm_eval_summary is not sourced. Record
# requested, completed, and failed concurrencies instead of treating every
# requested value as completed.
###############################################################################
magpie_write_batched_eval_meta() {
  local dest="$1"
  local py="${MAGPIE_ACCURACY_REPORT_PYTHON:-python3}"
  EVAL_BATCHED_CONCS="${EVAL_BATCHED_CONCS:-}" \
  EVAL_BATCHED_COMPLETED_CONCS="${EVAL_BATCHED_COMPLETED_CONCS:-}" \
  EVAL_BATCHED_FAILED_CONCS="${EVAL_BATCHED_FAILED_CONCS:-}" \
  "$py" - "$dest" <<'PY'
import json
import os
import sys
from pathlib import Path

def numbers(raw):
    return [int(part) for part in str(raw or "").split() if part]

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "eval_concs": numbers(os.environ.get("EVAL_BATCHED_CONCS")),
            "completed_eval_concs": numbers(
                os.environ.get("EVAL_BATCHED_COMPLETED_CONCS")
            ),
            "failed_eval_concs": numbers(os.environ.get("EVAL_BATCHED_FAILED_CONCS")),
        }
    )
    + "\n",
    encoding="utf-8",
)
PY
}

###############################################################################
# magpie_run_eval_persisted
#
# Persist lm-eval artifacts under $RESULT_DIR/lm_eval.
#
# When MAGPIE_EVAL_TASKS is set, Magpie drives lm-eval with local-completions
# against the local server port. That path supports loglikelihood multiple-choice
# tasks and puts --confirm_run_unsafe_code on the lm-eval argv (including when
# InferenceX server-watch would otherwise spawn lm-eval as a subprocess).
#
# When MAGPIE_EVAL_TASKS is unset, keep InferenceX run_eval for the default
# single-YAML GSM8K flow.
###############################################################################
magpie_run_eval_persisted() {
  magpie_eval_apply_code_eval_env
  magpie_prepare_eval_include_and_limit

  local result_dir="${RESULT_DIR:-${WORKSPACE_DIR:-/workspace}}"
  local eval_dir="${result_dir%/}/lm_eval"
  local raw_dir="${EVAL_RESULT_DIR:-}"
  local eval_rc=0
  local stage_rc=0

  if [[ -z "$raw_dir" ]]; then
    raw_dir=$(mktemp -d /tmp/eval_out-magpie-XXXXXX) || {
      echo "[magpie_bench_remote_compat] ERROR cannot create eval result directory" >&2
      return 1
    }
  fi
  mkdir -p "$raw_dir" "$eval_dir" || {
    echo "[magpie_bench_remote_compat] ERROR cannot prepare accuracy directories" >&2
    return 1
  }
  raw_dir=$(cd "$raw_dir" && pwd -P) || return 1

  export EVAL_RESULT_DIR="$raw_dir"
  local caller_dir="$PWD"
  cd "$raw_dir" || return 1

  if [[ -n "${MAGPIE_EVAL_TASKS:-}" ]]; then
    magpie_eval_prepare_deps || {
      eval_rc=$?
      cd "$caller_dir" || return 1
      return "$eval_rc"
    }
    local port
    port="$(magpie_eval_port_from_args "$@")"
    local base_url="http://127.0.0.1:${port}/v1/completions"
    magpie_run_eval_concurrency_loop "$raw_dir" "$base_url" || eval_rc=$?
  else
    if ! declare -F run_eval &>/dev/null; then
      echo "[magpie_bench_remote_compat] ERROR run_eval is unavailable" >&2
      cd "$caller_dir" || return 1
      return 1
    fi
    run_eval "$@" || eval_rc=$?
  fi
  cd "$caller_dir" || return 1

  if [[ -n "${EVAL_BATCHED_CONCS:-}" ]]; then
    if declare -F append_lm_eval_summary &>/dev/null; then
      (cd "$raw_dir" && append_lm_eval_summary) || stage_rc=$?
    elif declare -F _write_lm_eval_meta_json &>/dev/null; then
      _write_lm_eval_meta_json \
        "$raw_dir/meta_env.json" "" \
        "${EVAL_CONCURRENT_REQUESTS:-${CONC:-1}}" || stage_rc=$?
    else
      magpie_write_batched_eval_meta "$raw_dir/meta_env.json" || stage_rc=$?
    fi
  else
    if declare -F _write_lm_eval_meta_json &>/dev/null; then
      _write_lm_eval_meta_json \
        "$raw_dir/meta_env.json" "" \
        "${EVAL_CONCURRENT_REQUESTS:-${CONC:-1}}" || stage_rc=$?
    fi
  fi

  local source_file destination
  while IFS= read -r -d '' source_file; do
    destination="$eval_dir/$(basename "$source_file")"
    if ! cp -p "$source_file" "$destination"; then
      echo "[magpie_bench_remote_compat] WARN failed to copy $source_file" >&2
      stage_rc=1
    fi
  done < <(find "$raw_dir" -type f \( -name "*.json" -o -name "*.jsonl" \) -print0 2>/dev/null)

  magpie_write_accuracy_result "$eval_dir" "$eval_rc" "$result_dir" || stage_rc=$?
  if [[ $eval_rc -ne 0 ]]; then
    return "$eval_rc"
  fi
  return "$stage_rc"
}
