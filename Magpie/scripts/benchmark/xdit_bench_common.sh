#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Magpie generic benchmark body for xDiT (diffusion, server-less).
#
# Sourced by the runner-specific entrypoints xdit_mi355x.sh / xdit_mi300x.sh.
# Unlike the serving scripts (sglang/vllm), xDiT runs a SINGLE command: there
# is no OpenAI server and no benchmark_serving.py client. It runs the xDiT CLI
# once, parses E2E per-image latency, runs an image-quality gate (LPIPS / SSIM
# / MSE vs a fixed BF16 reference), and writes an InferenceX-shaped result JSON
# that Magpie's ResultParser consumes.
#
# Magpie contract (env in):
#   RESULT_DIR        output dir (Magpie sets to the workspace)
#   RESULT_FILENAME   result basename (Magpie sets to "inferencex_result")
#   RUNNER_TYPE       gpu runner (mi355x / mi300x) — set by the entrypoint
#   MODEL             model path (xdit --model uses its basename)
#   PRECISION         locked bf16
#   TP                Ulysses sequence-parallel degree (= GPU count)
#   ROCR_VISIBLE_DEVICES  GPU pin from Magpie/Hyperloom (mapped to HIP)
#   EXTRA_XDIT_ARGS   extra xdit CLI flags injected by the grid runner
#   PROFILE           "1" enables best-effort torch profiler (see below)
#
# Tunable env (BF16-safe knobs the optimizer may set):
#   XDIT_ATTENTION_BACKEND (aiter) XDIT_USE_TORCH_COMPILE (1)
#   XDIT_HEIGHT (1024) XDIT_WIDTH (1024) XDIT_NUM_STEPS (28)
#   XDIT_NUM_ITERATIONS (25) XDIT_WARMUP_CALLS (5) XDIT_GUIDANCE_SCALE (4.0)
#   XDIT_ULYSSES_DEGREE (defaults to $TP)
#
# Quality gate env:
#   XDIT_QUALITY_REF / XDIT_QUALITY_LPIPS_MAX / XDIT_QUALITY_SSIM_MIN /
#   XDIT_QUALITY_MSE_MAX  (empty ref => gate skipped, passed=true)
#
# LOCKED (precision = BF16; a precision change is a different model):
#   XDIT_USE_FP4_GEMMS / XDIT_USE_FP8_GEMMS forced OFF (overrides ignored).
#
set -euo pipefail

# ── ROCm toolchain: aiter JIT needs hipcc ────────────────────────────
if ! command -v hipcc &>/dev/null && [ -x /opt/rocm/bin/hipcc ]; then
    export PATH="/opt/rocm/bin:${PATH}"
fi

# ── Precision lock: BF16 ─────────────────────────────────────────────
if [ "${XDIT_USE_FP4_GEMMS:-0}" != "0" ]; then
    echo "[xdit][lock] XDIT_USE_FP4_GEMMS=${XDIT_USE_FP4_GEMMS} IGNORED — precision locked to BF16."
fi
if [ "${XDIT_USE_FP8_GEMMS:-0}" != "0" ]; then
    echo "[xdit][lock] XDIT_USE_FP8_GEMMS=${XDIT_USE_FP8_GEMMS} IGNORED — precision locked to BF16."
fi

# ── GPU visibility (rocm index space -> HIP) ─────────────────────────
# ROCR_VISIBLE_DEVICES already re-indexes visible GPUs to 0..N-1, so HIP must
# use the logical range, not the original physical ids. Only derive it when the
# caller has not already pinned HIP_VISIBLE_DEVICES (matches vLLM/Atom scripts).
unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
    n=$(echo "${ROCR_VISIBLE_DEVICES}" | awk -F, '{print NF}')
    export HIP_VISIBLE_DEVICES=$(seq -s, 0 $((n - 1)))
fi
export HSA_NO_SCRATCH_RECLAIM=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

# ── Resolve config ───────────────────────────────────────────────────
RESULT_DIR="${RESULT_DIR:?RESULT_DIR must be set by Magpie}"
RESULT_FILENAME="${RESULT_FILENAME:-inferencex_result}"
OUTPUT_FILE="${RESULT_DIR}/${RESULT_FILENAME}.json"
mkdir -p "${RESULT_DIR}"

MODEL_PATH="${MODEL:?MODEL must be set}"
# xDiT CLI --model accepts a model name (basename) or full path depending on
# the build. Default to basename (arbor parity); set XDIT_MODEL_ARG=path to
# pass the full local path instead.
case "${XDIT_MODEL_ARG:-name}" in
    path) XDIT_MODEL_NAME="${MODEL_PATH}" ;;
    *)    XDIT_MODEL_NAME="${XDIT_MODEL_NAME:-$(basename "${MODEL_PATH}")}" ;;
esac

ATTENTION_BACKEND="${XDIT_ATTENTION_BACKEND:-aiter}"
ULYSSES="${XDIT_ULYSSES_DEGREE:-${TP:-8}}"
SEED="${XDIT_SEED:-42}"
MAX_SEQ_LEN="${XDIT_MAX_SEQ_LEN:-512}"
HEIGHT="${XDIT_HEIGHT:-1024}"
WIDTH="${XDIT_WIDTH:-1024}"
NUM_STEPS="${XDIT_NUM_STEPS:-28}"
NUM_ITERATIONS="${XDIT_NUM_ITERATIONS:-25}"
WARMUP="${XDIT_WARMUP_CALLS:-5}"
GUIDANCE="${XDIT_GUIDANCE_SCALE:-4.0}"

QUALITY_REF="${XDIT_QUALITY_REF:-}"
QUALITY_LPIPS_MAX="${XDIT_QUALITY_LPIPS_MAX:-0.05}"
QUALITY_SSIM_MIN="${XDIT_QUALITY_SSIM_MIN:-0.95}"
QUALITY_MSE_MAX="${XDIT_QUALITY_MSE_MAX:-0.002}"

# torch.compile flag (BF16-safe; the big proven win)
EXTRA_FLAGS=()
if [ "${XDIT_USE_TORCH_COMPILE:-1}" = "1" ]; then
    EXTRA_FLAGS+=(--use_torch_compile)
fi

# Best-effort torch profiler: xDiT uses --profile + --output_directory (not
# --torch_profiler_dir). Traces land as profile_trace_rank_*.json.gz; a
# post-run rename bridges them to the *.trace.json.gz glob Hyperloom expects.
PROFILER_DIR="${VLLM_TORCH_PROFILER_DIR:-${SGLANG_TORCH_PROFILER_DIR:-${RESULT_DIR}/torch_trace}}"
XDIT_PROFILE_ENABLED=0
if [ "${PROFILE:-0}" = "1" ] && [ "${XDIT_SUPPORTS_PROFILER:-0}" = "1" ]; then
    mkdir -p "${PROFILER_DIR}"
    EXTRA_FLAGS+=(--profile --profile_wait 0 --profile_warmup "${XDIT_PROFILE_WARMUP:-2}" --profile_active "${XDIT_PROFILE_ACTIVE:-1}")
    XDIT_PROFILE_ENABLED=1
fi

case "${ATTENTION_BACKEND}" in
    aiter_fp8|aiter_sage|aiter_sage_v2|aiter_sparse_sage|aiter_sparse_sage_v2)
        echo "[xdit][warn] ${ATTENTION_BACKEND} is QUANTIZED attention — known regression on FLUX (small seqlen). Prefer 'aiter'."
        ;;
esac

XDIT_INPUT_IMAGE="${XDIT_INPUT_IMAGE:-/app/data/flux_cat.png}"
XDIT_PROMPT="${XDIT_PROMPT:-Add a cool hat to the cat}"

# ── Preflight checks ─────────────────────────────────────────────────
if ! command -v xdit &>/dev/null; then
    echo "[xdit][fatal] 'xdit' binary not found in PATH. Ensure the xDiT package is installed in the current environment." >&2
    exit 1
fi
if [ ! -f "${XDIT_INPUT_IMAGE}" ]; then
    echo "[xdit][warn] input image not found: ${XDIT_INPUT_IMAGE} — xDiT may fail or use its default." >&2
fi

echo "=== Magpie xDiT bench (BF16, runner=${RUNNER_TYPE:-unknown}) ==="
echo "  model=${MODEL_PATH} attn=${ATTENTION_BACKEND} res=${HEIGHT}x${WIDTH} ulysses=${ULYSSES}"
echo "  steps=${NUM_STEPS} iters=${NUM_ITERATIONS} warmup=${WARMUP} compile=${XDIT_USE_TORCH_COMPILE:-1}"
echo "  quality_ref=${QUALITY_REF:-<none>} thresholds: LPIPS<${QUALITY_LPIPS_MAX} SSIM>${QUALITY_SSIM_MIN} MSE<${QUALITY_MSE_MAX}"
echo "  extra_xdit_args=${EXTRA_XDIT_ARGS:-<none>}"

XDIT_RUN_DIR="$(mktemp -d "${RESULT_DIR}/xdit_run.XXXXXX")"
XDIT_LOG="${XDIT_RUN_DIR}/xdit_stdout.log"

START_NS=$(date +%s%N)

# shellcheck disable=SC2086  # EXTRA_XDIT_ARGS is intentionally word-split.
xdit \
    --model "${XDIT_MODEL_NAME}" \
    --seed "${SEED}" \
    --prompt "${XDIT_PROMPT}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --num_inference_steps "${NUM_STEPS}" \
    --max_sequence_length "${MAX_SEQ_LEN}" \
    --warmup_calls "${WARMUP}" \
    --ulysses_degree "${ULYSSES}" \
    --guidance_scale "${GUIDANCE}" \
    --num_iterations "${NUM_ITERATIONS}" \
    --attention_backend "${ATTENTION_BACKEND}" \
    --input_images "${XDIT_INPUT_IMAGE}" \
    --output_directory "${XDIT_RUN_DIR}" \
    ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
    ${EXTRA_XDIT_ARGS:-} \
    2>&1 | tee "${XDIT_LOG}"

END_NS=$(date +%s%N)
WALL_MS=$(( (END_NS - START_NS) / 1000000 ))

# Bridge xDiT profiler traces to Hyperloom's expected *.trace.json.gz naming.
if [ "${XDIT_PROFILE_ENABLED}" = "1" ]; then
    for src in "${XDIT_RUN_DIR}"/profile_trace_rank_*.json.gz; do
        [ -f "$src" ] || continue
        rank=$(echo "$src" | grep -oP 'rank_\K\d+')
        dst="${PROFILER_DIR}/rank_${rank}.trace.json.gz"
        cp "$src" "$dst"
        echo "[xdit][profile] copied $src -> $dst"
    done
fi

python3 - "${XDIT_LOG}" "${OUTPUT_FILE}" "${WALL_MS}" "${NUM_ITERATIONS}" "${XDIT_RUN_DIR}" \
          "${QUALITY_REF}" "${QUALITY_LPIPS_MAX}" "${QUALITY_SSIM_MIN}" "${QUALITY_MSE_MAX}" \
          "${MODEL_PATH}" "${ATTENTION_BACKEND}" <<'PYEOF'
import glob
import json
import os
import re
import sys

log_path, output_path, wall_ms_str, num_iter_str, run_dir = sys.argv[1:6]
quality_ref_path, lpips_max_str, ssim_min_str, mse_max_str = sys.argv[6:10]
model_path, attention_backend = sys.argv[10:12]

wall_ms = int(wall_ms_str)
num_iterations = int(num_iter_str)
lpips_max = float(lpips_max_str)
ssim_min = float(ssim_min_str)
mse_max = float(mse_max_str)

try:
    log_text = open(log_path, encoding="utf-8", errors="replace").read()
except OSError:
    log_text = ""

# ── Parse E2E per-image latency ──────────────────────────────────────
e2e_latency_s = None
for pattern in [
    r"Average time over \d+ runs:\s*([\d.]+)s",
    r"Iteration \d+ completed in ([\d.]+)s",
    r"100%\|[^|]*\|\s*\d+/\d+\s*\[[\d:]+<[\d:]+,\s*([\d.]+)s/it\]",
    r"([\d.]+)s/it\]",
    r"(?:average|mean|avg)\s+(?:e2e|end.to.end|latency|time)\s*[:\s=]+\s*([\d.]+)\s*(?:s|sec)",
    r"(?:e2e|latency|inference)\s*[:\s=]+\s*([\d.]+)\s*s",
]:
    matches = re.findall(pattern, log_text, re.IGNORECASE)
    if matches:
        e2e_latency_s = float(matches[-1])
        break

if e2e_latency_s is None:
    e2e_latency_s = (wall_ms / 1000.0) / max(num_iterations, 1)
    print(f"[xdit] no parsed timing; wall-clock fallback {e2e_latency_s:.3f}s/image")
else:
    print(f"[xdit] parsed E2E latency {e2e_latency_s:.3f}s/image")

output_throughput = 1.0 / e2e_latency_s if e2e_latency_s > 0 else 0.0

generated = sorted(
    glob.glob(os.path.join(run_dir, "*.png")) + glob.glob(os.path.join(run_dir, "*.jpg"))
)
completed = len(generated) if generated else num_iterations

# ── Image-quality gate (vs fixed BF16 reference) ─────────────────────
quality_gate = None
ref_write_path = os.environ.get("XDIT_QUALITY_REF_WRITE", "").strip()


def _try_load_image_libs():
    """Return (numpy, PIL.Image) or (None, None) if unavailable."""
    try:
        import numpy as _np
        from PIL import Image as _Img
        return _np, _Img
    except Exception as exc:
        print(f"[xdit][warn] image libs unavailable ({exc}); quality gate skipped",
              file=sys.stderr)
        return None, None


if quality_ref_path and os.path.isfile(quality_ref_path) and generated:
    np, Image = _try_load_image_libs()
    if np is None:
        quality_gate = {"passed": True, "skipped": True, "reason": "image_libs_unavailable"}
    else:
        ref = np.array(Image.open(quality_ref_path).convert("RGB")).astype("float32") / 255.0
        gen_path = generated[-1]
        gen_img = Image.open(gen_path).convert("RGB")
        if (gen_img.height, gen_img.width) != (ref.shape[0], ref.shape[1]):
            gen_img = gen_img.resize((ref.shape[1], ref.shape[0]), Image.LANCZOS)
        gen = np.array(gen_img).astype("float32") / 255.0

        mse_val = float(np.mean((ref - gen) ** 2))
        try:
            from skimage.metrics import structural_similarity
            ssim_val = float(structural_similarity(ref, gen, channel_axis=2, data_range=1.0))
        except Exception as exc:
            print(f"[xdit] SSIM failed ({exc}); fallback 1.0")
            ssim_val = 1.0
        lpips_val = 0.0
        try:
            import lpips
            import torch
            loss_fn = lpips.LPIPS(net="alex", verbose=False)
            ref_t = torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0) * 2 - 1
            gen_t = torch.from_numpy(gen).permute(2, 0, 1).unsqueeze(0) * 2 - 1
            with torch.no_grad():
                lpips_val = float(loss_fn(ref_t, gen_t).item())
        except Exception as exc:
            print(f"[xdit] LPIPS failed ({exc}); fallback 0.0")
            lpips_val = 0.0

        passed = (mse_val <= mse_max) and (ssim_val >= ssim_min) and (lpips_val <= lpips_max)
        quality_gate = {
            "passed": bool(passed),
            "lpips": round(lpips_val, 6),
            "ssim": round(ssim_val, 6),
            "mse": round(mse_val, 6),
            "lpips_max": lpips_max,
            "ssim_min": ssim_min,
            "mse_max": mse_max,
            "reference": quality_ref_path,
        }
        print(
            f"[xdit] quality {'PASS' if passed else 'FAIL'} "
            f"lpips={lpips_val:.4f} ssim={ssim_val:.4f} mse={mse_val:.6f}"
        )
elif generated and ref_write_path:
    np, Image = _try_load_image_libs()
    if np is not None:
        try:
            os.makedirs(os.path.dirname(ref_write_path) or ".", exist_ok=True)
            Image.open(generated[-1]).convert("RGB").save(ref_write_path)
            print(f"[xdit] established quality reference -> {ref_write_path}", file=sys.stderr)
        except Exception as exc:
            print(f"[xdit][warn] could not write reference ({exc})", file=sys.stderr)
    quality_gate = {"passed": True, "skipped": True, "reason": "reference_established"}
else:
    print(
        "[xdit][warn] XDIT_QUALITY_REF is empty — image-quality gate SKIPPED. "
        "Throughput tuning has NO quality guard. Set XDIT_QUALITY_REF to a BF16 "
        "reference image or XDIT_QUALITY_REF_WRITE to auto-capture one.",
        file=sys.stderr,
    )
    quality_gate = {"passed": True, "skipped": True, "reason": "no_reference_or_image"}

result = {
    "framework": "xdit",
    "model": model_path,
    "workload_kind": "scriptable",
    "throughput_unit": "img/s",
    "output_throughput": round(output_throughput, 6),
    "request_throughput": round(output_throughput, 6),
    "completed": completed,
    "num_prompts": num_iterations,
    "duration": round(wall_ms / 1000.0, 3),
    "latency_s": round(e2e_latency_s, 4),
    "mean_e2el_ms": round(e2e_latency_s * 1000.0, 3),
    "attention_backend": attention_backend,
    "precision_locked": "bf16",
    "quality_gate": quality_gate,
}

with open(output_path, "w", encoding="utf-8") as fh:
    json.dump(result, fh, indent=2)

print(
    f"[xdit] throughput={output_throughput:.4f} img/s latency={e2e_latency_s:.3f}s "
    f"completed={completed} -> {output_path}"
)
PYEOF

echo "=== Magpie xDiT bench complete ==="
