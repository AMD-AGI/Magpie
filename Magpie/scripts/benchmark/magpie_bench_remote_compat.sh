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
#   CONC                  concurrency cap (passed to local-completions)
#   RESULT_DIR            workspace dir; results land at $RESULT_DIR/lm_eval/
#
# Inputs (env, optional):
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

  _install_lm_eval_deps || return $?

  local py="${MAGPIE_EVAL_PYTHON:-python3}"
  local result_dir="${RESULT_DIR:-${WORKSPACE_DIR:-/workspace}}"
  local out_dir="${result_dir%/}/lm_eval"
  mkdir -p "$out_dir" || {
    echo "[magpie_bench_remote_compat] ERROR cannot mkdir $out_dir" >&2
    return 1
  }

  local tasks="${MAGPIE_EVAL_TASKS:-gsm8k}"
  local include_path="${MAGPIE_EVAL_INCLUDE_PATH:-${MAGPIE_INFERENCEX_ROOT:-$(pwd)}/utils/evals}"
  local batch_size="${MAGPIE_EVAL_BATCH_SIZE:-auto}"
  local conc="${CONC:-8}"

  # local-completions hits an OpenAI-compatible /v1/completions endpoint.
  # base_url ends in /v1/completions; tokenizer_backend=huggingface uses
  # the local hub tokenizer (model path/id) so we don't pay a server-side
  # tokenization roundtrip.
  #
  # MAGPIE_EVAL_TOKENIZED_REQUESTS (optional) controls the prompt wire format.
  # Unset => lm_eval's default (token-id-array prompts), which a direct sglang
  # server accepts. A PD-disaggregated sglang_router's /v1/completions only
  # accepts StringOrArray and rejects token-id arrays with HTTP 422, collapsing
  # the accuracy eval; set MAGPIE_EVAL_TOKENIZED_REQUESTS=false there to send
  # string prompts instead. Absent env => byte-for-byte the previous behaviour.
  local base_url="${BENCHMARK_BASE_URL%/}/v1/completions"
  local model_args="model=${MODEL},base_url=${base_url},num_concurrent=${conc},tokenizer_backend=huggingface,trust_remote_code=true${MAGPIE_EVAL_TOKENIZED_REQUESTS:+,tokenized_requests=${MAGPIE_EVAL_TOKENIZED_REQUESTS}}"
  if [[ ! -d "$include_path" ]]; then
    echo "[magpie_bench_remote_compat] ERROR evaluator include path is unavailable: $include_path" >&2
    return 1
  fi
  local -a cmd=(
    "$py" -m lm_eval
    --model local-completions
    --include_path "$include_path"
    --tasks "$tasks"
    --model_args "$model_args"
    --batch_size "$batch_size"
    --output_path "$out_dir"
  )
  if [[ -n "${MAGPIE_EVAL_LIMIT:-}" ]]; then
    cmd+=(--limit "$MAGPIE_EVAL_LIMIT")
  fi

  echo "[magpie_bench_remote_compat] lm_eval cmd: ${cmd[*]}" >&2
  set -x
  "${cmd[@]}"
  local rc=$?
  set +x
  if [[ $rc -ne 0 ]]; then
    echo "[magpie_bench_remote_compat] WARN lm_eval exited rc=$rc; accuracy gate will see no results" >&2
  fi
  return $rc
}

###############################################################################
# magpie_preserve_lm_eval_artifacts
#
# InferenceX's append_lm_eval_summary may move files out of EVAL_RESULT_DIR into
# its repository working directory. Preserve the raw evaluation evidence under
# the Magpie workspace both before and after that helper runs. The fallback only
# copies known lm-eval artifact names from the working-directory root.
###############################################################################
magpie_mark_lm_eval_start() {
  local result_dir="${RESULT_DIR:-${WORKSPACE_DIR:-/workspace}}"
  local inferx_root="${MAGPIE_INFERENCEX_ROOT:-$(pwd)}"
  local snapshot="${result_dir%/}/.lm_eval_preexisting.tsv"
  mkdir -p "$result_dir" || return 1
  # Snapshot the exact pre-run artifact identities. Timestamp-only filtering is
  # racy on filesystems whose timestamp resolution cannot distinguish the
  # marker from an immediately-created lm-eval result.
  find "$inferx_root" -maxdepth 1 -type f \
    \( -name 'results*.json' -o -name 'samples*.jsonl' \
       -o -name '*lm_eval*summary*.json' \) \
    -printf '%f\t%s\t%T@\n' | LC_ALL=C sort > "$snapshot" || return 1
  touch "${result_dir%/}/.lm_eval_started" || return 1
}

magpie_preserve_lm_eval_artifacts() {
  local result_dir="${RESULT_DIR:-${WORKSPACE_DIR:-/workspace}}"
  local out_dir="${result_dir%/}/lm_eval"
  local marker="${result_dir%/}/.lm_eval_started"
  local snapshot="${result_dir%/}/.lm_eval_preexisting.tsv"
  mkdir -p "$out_dir" || return 1

  if [[ -n "${EVAL_RESULT_DIR:-}" && -d "${EVAL_RESULT_DIR}" ]]; then
    local source_real out_real
    source_real=$(readlink -f "${EVAL_RESULT_DIR}" 2>/dev/null || true)
    out_real=$(readlink -f "$out_dir" 2>/dev/null || true)
    if [[ -n "$source_real" && "$source_real" != "$out_real" ]]; then
      cp -a "${EVAL_RESULT_DIR}"/. "$out_dir"/ || return 1
    fi
  fi

  local artifact signature
  while IFS= read -r -d '' artifact; do
    if [[ -e "$marker" && -f "$snapshot" ]]; then
      signature=$(find "$artifact" -maxdepth 0 -printf '%f\t%s\t%T@\n')
      if grep -Fqx -- "$signature" "$snapshot"; then
        continue
      fi
    fi
    cp -a "$artifact" "$out_dir"/ || return 1
  done < <(
    find "${MAGPIE_INFERENCEX_ROOT:-$(pwd)}" -maxdepth 1 -type f \
      \( -name 'results*.json' -o -name 'samples*.jsonl' \
         -o -name '*lm_eval*summary*.json' \) \
      -print0
  )
}
