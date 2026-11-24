#!/usr/bin/env bash

if [[ -f "$(dirname -- "${BASH_SOURCE[0]}")/benchmark_lib.sh" ]]; then
  source "$(dirname -- "${BASH_SOURCE[0]}")/benchmark_lib.sh"
else
  check_env_vars() {
    local name missing=()
    for name in "$@"; do
      [[ -n "${!name}" ]] || missing+=("$name")
    done
    if (( ${#missing[@]} )); then
      echo "ERROR: missing required environment variables: ${missing[*]}" >&2
      exit 1
    fi
  }

  wait_for_server_ready() {
    local port="" pid="" log="" root
    while (( $# )); do
      case "$1" in
        --port) port="$2"; shift 2 ;;
        --server-pid) pid="$2"; shift 2 ;;
        --server-log) log="$2"; shift 2 ;;
        *) echo "ERROR: unsupported readiness option: $1" >&2; return 2 ;;
      esac
    done
    [[ -n "$port" ]] || return 2
    root="$(magpie_inferencex_project_root)" || return $?
    local args=(--url "http://127.0.0.1:$port/health")
    [[ -z "$pid" ]] || args+=(--pid "$pid")
    [[ -z "$log" ]] || args+=(--log "$log")
    PYTHONPATH="$root" python3 -m infx.bench wait "${args[@]}" || exit $?
  }

  run_benchmark_serving() {
    python3 "$(dirname -- "${BASH_SOURCE[0]}")/inferencex_client.py" "$@"
  }

  run_eval() {
    local port root py
    port="$(magpie_eval_port_from_args "$@")" || return $?
    root="$(magpie_inferencex_project_root)" || return $?
    py="$(magpie_infx_client_python "$root")" || return $?
    IS_MULTINODE=false EVAL_ONLY="${EVAL_ONLY:-false}" OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}" \
    PYTHONPATH="$root" "$py" -m infx.bench eval \
      --endpoint "http://127.0.0.1:$port" \
      --concurrency "${EVAL_CONCURRENT_REQUESTS:-$CONC}" \
      --stage-to "${EVAL_RESULT_DIR:?}" --framework lm-eval
  }
fi
