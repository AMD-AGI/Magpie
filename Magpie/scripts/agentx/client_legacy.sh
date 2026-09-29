#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
# Explicit adapter for the pre-srt client library. Never launches a server.
set -eo pipefail
: "${INFMAX_CONTAINER_WORKSPACE:?}" "${AIPERF_SERVER_URL:?}" "${RESULT_DIR:?}"
source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"
for function in resolve_trace_source install_agentic_deps build_replay_cmd run_agentic_replay_and_write_outputs; do
    declare -F "$function" >/dev/null || { echo "Unsupported legacy AgentX client library: $function missing" >&2; exit 2; }
done
resolve_trace_source
install_agentic_deps
mkdir -p "$RESULT_DIR"
build_replay_cmd "$RESULT_DIR"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
