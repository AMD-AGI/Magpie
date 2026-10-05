###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Split EXTRA_VLLM_ARGS on IFS (spaces/tabs/newlines) without pathname expansion.
# Callers must use the resulting EXTRA_SERVER_ARGS array.

magpie_split_extra_vllm_args() {
  EXTRA_SERVER_ARGS=()
  local raw="${EXTRA_VLLM_ARGS:-}"
  raw="${raw//$'\r'/}"
  if [[ -z "$raw" ]]; then
    return 0
  fi
  # Read the whole string (NUL delimiter), then IFS-split. Quoted so *, ?,
  # and brackets stay literal. `read` returns 1 at EOF without a NUL.
  read -r -d '' -a EXTRA_SERVER_ARGS <<< "$raw" || true
}
