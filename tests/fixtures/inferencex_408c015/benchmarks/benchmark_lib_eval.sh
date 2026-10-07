# Unmodified function excerpts from inferencex-e2e/benchmarks/benchmark_lib.sh
# at 408c015be4b22d14c69518643609669405507077 (Apache-2.0; ../LICENSE).

check_env_vars() {
    local missing_vars=()

    for var_name in "$@"; do
        if [[ -z "${!var_name:-}" ]]; then
            missing_vars+=("$var_name")
        fi
    done

    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        echo "Error: The following required environment variables are not set:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        exit 1
    fi
}

run_lm_eval() {
    check_env_vars OPENAI_API_KEY PORT
    local port="${PORT}"
    local tasks_dir="${EVAL_TASKS_DIR:-infx/evals/gsm8k.yaml}"
    local results_dir="${EVAL_RESULT_DIR:-$(mktemp -d /tmp/eval_out-XXXXXX)}"
    local eval_context_len="${EVAL_MAX_MODEL_LEN}"
    local temperature=0
    local top_p=1
    local concurrent_requests="${EVAL_CONCURRENT_REQUESTS:-${CONC}}"
    check_env_vars concurrent_requests
    # SWE-bench adds a repo-local task YAML, hence --include_path. --limit is
    # passed only when EVAL_LIMIT requests a smoke-test slice.
    local eval_limit="${EVAL_LIMIT:-}"
    local include_path="${EVAL_INCLUDE_PATH:-}"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port|--task|--results-dir|--gen-max-tokens|--temperature|--top-p)
                if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
                    echo "ERROR: $1 requires a value" >&2
                    return 2
                fi
                case "$1" in
                    --port)           port="$2" ;;
                    --task)           tasks_dir="$2" ;;
                    --results-dir)    results_dir="$2" ;;
                    --gen-max-tokens) eval_context_len="$2" ;;
                    --temperature)    temperature="$2" ;;
                    --top-p)          top_p="$2" ;;
                esac
                shift 2
                ;;
            *)
                echo "Unknown parameter: $1" >&2
                return 2
                ;;
        esac
    done

    check_env_vars eval_context_len

    # Serving images may use a different WORKDIR.
    local _repo_root="$INFERENCEX_REPO_ROOT"
    if [[ "$tasks_dir" == *.yaml && "$tasks_dir" != /* \
          && ! -f "$tasks_dir" && -f "$_repo_root/$tasks_dir" ]]; then
        echo "run_lm_eval: anchoring relative task '$tasks_dir' to repo root -> $_repo_root/$tasks_dir"
        tasks_dir="$_repo_root/$tasks_dir"
    fi

    export EVAL_TASKS_DIR="$tasks_dir"

    if [ "${INFERENCEX_LM_EVAL_RUNTIME_READY:-false}" != "true" ]; then
        _install_lm_eval_deps
        _patch_lm_eval
        export INFERENCEX_LM_EVAL_RUNTIME_READY=true
    fi

    local openai_server_base="http://0.0.0.0:${port}"
    local openai_chat_base="${openai_server_base}/v1/chat/completions"
    export OPENAI_API_KEY=${OPENAI_API_KEY}
    MODEL_NAME=${MODEL_NAME:-$MODEL} # Prefer MODEL_NAME, else MODEL

    # Leave room for input within the context window and avoid excessive
    # per-request KV cache reservation on TRT.
    local max_output_tokens=$(( eval_context_len > 4096 ? eval_context_len - 4096 : eval_context_len / 2 ))
    if [ "$max_output_tokens" -gt 16384 ]; then
        max_output_tokens=16384
    fi
    echo "Eval budget: eval_context_len=${eval_context_len}, max_output_tokens=${max_output_tokens}"

    # Read by append_lm_eval_summary.
    export EVAL_RESULT_DIR="$results_dir"
    set -x
    run_server_client python3 -m lm_eval --model local-chat-completions --apply_chat_template \
      ${include_path:+--include_path "$include_path"} \
      --tasks "${tasks_dir}" \
      --output_path "${results_dir}" \
      --log_samples \
      --model_args "model=${MODEL_NAME},base_url=${openai_chat_base},api_key=${OPENAI_API_KEY},eos_string=</s>,max_retries=5,num_concurrent=${concurrent_requests},timeout=1800,tokenized_requests=False,max_length=${eval_context_len}" \
      --gen_kwargs "max_tokens=${max_output_tokens},temperature=${temperature},top_p=${top_p}" \
      ${eval_limit:+--limit "$eval_limit"}
    local eval_exit=$?
    set +x
    return $eval_exit
}

run_eval() {
    check_env_vars EVAL_ONLY IS_AGENTIC
    local cli_framework=""
    local forwarded=()
    # Keep runner-selected suite identity scoped to this invocation.
    local EVAL_SUITE="${EVAL_SUITE:-}"
    unset EVAL_COMPLETED_SUITE

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --framework)
                if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
                    echo "ERROR: --framework requires a value" >&2
                    return 2
                fi
                cli_framework="$2"
                shift 2
                ;;
            *)
                forwarded+=("$1")
                shift
                ;;
        esac
    done

    local scenario_default="lm-eval"
    local scenario_is_agentic=0
    if [ "${IS_AGENTIC}" = "1" ] || [ "${SCENARIO_TYPE:-}" = "agentic-coding" ]; then
        scenario_is_agentic=1
    fi

    local framework="${EVAL_FRAMEWORK:-${cli_framework:-$scenario_default}}"
    case "$framework" in
        kimi-vendor)
            [ -n "${EVAL_SUITE:-}" ] || EVAL_SUITE="kimi_tool_call_schema"
            ;;
        minimax-vendor)
            [ -n "${EVAL_SUITE:-}" ] || EVAL_SUITE="minimax_m3_smoke"
            ;;
        bfcl)
            [ -n "${EVAL_SUITE:-}" ] || EVAL_SUITE="bfcl_smoke"
            ;;
    esac

    case "${EVAL_SUITE:-}" in
        "") ;;
        *[!A-Za-z0-9_.-]*)
            echo "ERROR: EVAL_SUITE may contain only letters, digits, '.', '_', and '-'" >&2
            return 2
            ;;
    esac

    if [ -n "${EVAL_SUITE:-}" ] \
        && [ "$framework" != "kimi-vendor" ] \
        && [ "$framework" != "minimax-vendor" ] \
        && [ "$framework" != "bfcl" ]; then
        echo "ERROR: EVAL_SUITE is only supported with kimi-vendor, minimax-vendor, or bfcl" >&2
        return 2
    fi

    if [ "${EVAL_ONLY}" = "true" ]; then
        case "$framework" in
            kimi-vendor|minimax-vendor|bfcl)
                _wait_for_openai_chat_route "${forwarded[@]}" || return $?
                ;;
        esac
    fi

    # Explicit verifier suites use fixed request budgets and do not consume
    # EVAL_MAX_MODEL_LEN, so avoid loading model configuration for those paths.
    if [ "$framework" != "kimi-vendor" ] \
        && [ "$framework" != "minimax-vendor" ] \
        && [ "$framework" != "bfcl" ] \
        && [ -z "${EVAL_MAX_MODEL_LEN:-}" ]; then
        compute_eval_context_length "$MODEL" "${MAX_MODEL_LEN:-0}" > /dev/null
    fi

    unset EVAL_BATCHED_CONCS
    unset EVAL_BATCHED_COMPLETED_CONCS
    unset EVAL_BATCHED_FAILED_CONCS

    local requested_concs="${EVAL_CONCURRENT_REQUESTS:-}"
    local eval_concs=()
    if [ -n "$requested_concs" ]; then
        read -r -a eval_concs <<< "$requested_concs"
    fi

    if [ "${#eval_concs[@]}" -gt 1 ]; then
        if [[ "$framework" != "lm-eval" && "$framework" != "lm_eval" ]]; then
            echo "ERROR: batched eval concurrency is only supported for lm-eval" >&2
            return 1
        fi

        local eval_conc results_dir eval_rc stage_rc
        local completed_concs=()
        local failed_concs=()

        for eval_conc in "${eval_concs[@]}"; do
            if [[ ! "$eval_conc" =~ ^[1-9][0-9]*$ ]]; then
                echo "ERROR: invalid eval concurrency '${eval_conc}'" >&2
                return 1
            fi

            if ! results_dir=$(mktemp -d /tmp/eval_out-conc"${eval_conc}"-XXXXXX); then
                echo "ERROR: failed to create eval output directory for concurrency ${eval_conc}" >&2
                failed_concs+=("$eval_conc")
                continue
            fi

            echo "Running lm-eval at concurrency ${eval_conc} using the existing engine"
            export EVAL_CONCURRENT_REQUESTS="$eval_conc"
            export CONC="$eval_conc"
            eval_rc=0
            stage_rc=0
            run_lm_eval "${forwarded[@]}" --results-dir "$results_dir" \
                || eval_rc=$?
            _stage_lm_eval_artifacts "$results_dir" "$eval_conc" \
                || stage_rc=$?

            if [ "$eval_rc" -eq 0 ] && [ "$stage_rc" -eq 0 ]; then
                completed_concs+=("$eval_conc")
            else
                echo "ERROR: lm-eval failed at concurrency ${eval_conc} (eval_rc=${eval_rc}, stage_rc=${stage_rc})" >&2
                failed_concs+=("$eval_conc")
            fi
        done

        export EVAL_CONCURRENT_REQUESTS="$requested_concs"
        export EVAL_RESULT_DIR=""
        export EVAL_BATCHED_CONCS="${eval_concs[*]}"
        export EVAL_BATCHED_COMPLETED_CONCS="${completed_concs[*]}"
        export EVAL_BATCHED_FAILED_CONCS="${failed_concs[*]}"

        if [ "${#failed_concs[@]}" -gt 0 ]; then
            echo "ERROR: batched eval failed for concurrency: ${failed_concs[*]}" >&2
            echo "Deferring failure until post-upload score validation preserves all artifacts" >&2
        fi
        return 0
    fi

    if [ -n "${EVAL_CONCURRENT_REQUESTS:-}" ]; then
        export CONC="$EVAL_CONCURRENT_REQUESTS"
    fi

    local eval_rc=0
    case "$framework" in
        lm-eval|lm_eval) run_lm_eval "${forwarded[@]}" || eval_rc=$? ;;
        swebench)        run_swebench_eval "${forwarded[@]}" || eval_rc=$? ;;
        kimi-vendor)     run_kimi_vendor_eval "${forwarded[@]}" || eval_rc=$? ;;
        minimax-vendor)  run_minimax_vendor_eval "${forwarded[@]}" || eval_rc=$? ;;
        bfcl)           run_bfcl_eval "${forwarded[@]}" || eval_rc=$? ;;
        *)               echo "Unknown framework '${framework}'"; eval_rc=1 ;;
    esac

    if [ -n "${EVAL_SUITE:-}" ]; then
        export EVAL_COMPLETED_SUITE="$EVAL_SUITE"
    fi

    local stage_rc=0
    # Agentic eval-only recipes have no separate staging step. Provider
    # failures are staged before returning so diagnostic artifacts survive.
    if { [ "${EVAL_ONLY}" = "true" ] && [ "$scenario_is_agentic" = "1" ]; } \
        || { { [ "$framework" = "kimi-vendor" ] \
            || [ "$framework" = "minimax-vendor" ] \
            || [ "$framework" = "bfcl" ]; } \
            && [ "$eval_rc" -ne 0 ]; }; then
        append_lm_eval_summary || stage_rc=$?
    fi
    if [ "$eval_rc" -ne 0 ]; then
        echo "ERROR: run_eval failed with exit code $eval_rc" >&2
        if [ "${EVAL_ONLY}" = "true" ]; then
            echo "Eval-only mode: failing after artifact collection" >&2
        fi
        return "$eval_rc"
    fi
    if [ "$stage_rc" -ne 0 ]; then
        echo "ERROR: eval artifact staging failed with exit code $stage_rc" >&2
        return "$stage_rc"
    fi
    return 0
}
