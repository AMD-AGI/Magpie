#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Magpie generic xDiT benchmark entrypoint for MI355X (gfx950).
# Resolved by benchmarker._get_benchmark_script as the Priority-3 generic
# script "xdit_<runner>.sh". Delegates to the shared body.
#
set -euo pipefail

export RUNNER_TYPE="${RUNNER_TYPE:-mi355x}"
# aiter is the BF16-correct attention backend on gfx950.
export XDIT_ATTENTION_BACKEND="${XDIT_ATTENTION_BACKEND:-aiter}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/xdit_bench_common.sh"
