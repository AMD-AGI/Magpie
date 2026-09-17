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
# magpie_prepare_eval_include_and_limit
#
# Optional include path and sample limit for Magpie-owned lm-eval invocations.
###############################################################################
magpie_prepare_eval_include_and_limit() {
  if [[ -z "${EVAL_LIMIT:-}" && -n "${MAGPIE_EVAL_LIMIT:-}" ]]; then
    export EVAL_LIMIT="$MAGPIE_EVAL_LIMIT"
  fi
  local include="${MAGPIE_EVAL_INCLUDE_PATH:-${MAGPIE_EVAL_TASK_PATH:-}}"
  if [[ -n "$include" && "$include" != *","* ]]; then
    if [[ -d "$include" ]]; then
      export EVAL_INCLUDE_PATH="$include"
    elif [[ -f "$include" ]]; then
      export EVAL_INCLUDE_PATH="$(cd "$(dirname "$include")" && pwd)"
    fi
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
magpie_run_lm_eval() {
  local out_dir="$1"
  local conc="$2"
  local base_url="$3"
  local py="${MAGPIE_EVAL_PYTHON:-python3}"
  local tasks="${MAGPIE_EVAL_TASKS:-gsm8k}"
  tasks="${tasks// /}"
  local batch_size="${MAGPIE_EVAL_BATCH_SIZE:-auto}"
  local conc_dir="${out_dir%/}/conc${conc}"
  mkdir -p "$conc_dir" || return 1
  local model_args="model=${MODEL},base_url=${base_url},num_concurrent=${conc},tokenizer_backend=huggingface,trust_remote_code=true${MAGPIE_EVAL_TOKENIZED_REQUESTS:+,tokenized_requests=${MAGPIE_EVAL_TOKENIZED_REQUESTS}}"
  local -a cmd=(
    "$py" -m lm_eval
    --model local-completions
    --tasks "$tasks"
    --model_args "$model_args"
    --batch_size "$batch_size"
    --output_path "$conc_dir"
  )
  local limit="${EVAL_LIMIT:-${MAGPIE_EVAL_LIMIT:-}}"
  if [[ -n "$limit" ]]; then
    cmd+=(--limit "$limit")
  fi
  if [[ -n "${EVAL_INCLUDE_PATH:-}" ]]; then
    cmd+=(--include_path "$EVAL_INCLUDE_PATH")
  fi
  if magpie_eval_needs_unsafe_code; then
    cmd+=(--confirm_run_unsafe_code)
  fi

  echo "[magpie_bench_remote_compat] lm_eval cmd: ${cmd[*]}" >&2
  set -x
  "${cmd[@]}"
  local rc=$?
  set +x
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
  local conc
  while read -r conc; do
    [[ -z "$conc" ]] && continue
    magpie_run_lm_eval "$out_dir" "$conc" "$base_url" || eval_rc=$?
  done < <(magpie_eval_concurrency_values)
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
    "acc,none",
    "acc",
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
    local port
    port="$(magpie_eval_port_from_args "$@")"
    local base_url="http://127.0.0.1:${port}/v1/completions"
    local conc
    local requested_concs=""
    while read -r conc; do
      [[ -z "$conc" ]] && continue
      requested_concs+="${requested_concs:+ }${conc}"
      magpie_run_lm_eval "$raw_dir" "$conc" "$base_url" || eval_rc=$?
    done < <(magpie_eval_concurrency_values)
    export EVAL_BATCHED_CONCS="$requested_concs"
    export EVAL_BATCHED_COMPLETED_CONCS="$requested_concs"
    export EVAL_BATCHED_FAILED_CONCS=""
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
      printf '{"eval_concs":[%s]}\n' "${EVAL_BATCHED_CONCS// /,}" \
        > "$raw_dir/meta_env.json" || stage_rc=$?
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
