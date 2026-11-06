###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# R9700 and R9600D vLLM server policy applied at the gfx12 runner boundary.
# Match the exact product; do not infer it from gfx1201 or the gfx12 runner label.

magpie_apply_r9700_aiter_rmsnorm_default() {
  case "${TARGET_GPU_TYPE:-}" in
    r9700|r9600d) ;;
    *) return 0 ;;
  esac

  local aiter="${VLLM_ROCM_USE_AITER:-0}"
  aiter="${aiter,,}"
  case "$aiter" in
    1|true|yes|on) ;;
    *) return 0 ;;
  esac

  if [[ -z "${VLLM_ROCM_USE_AITER_RMSNORM+x}" ]]; then
    export VLLM_ROCM_USE_AITER_RMSNORM=0
  fi
}
