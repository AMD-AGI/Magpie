###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Source file searcher with ripgrep and Python fallback.
"""

import glob
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Dict

from .models import (
    KernelKind,
    KernelCategory,
    ParsedKernelName,
    SourceMatch,
    TestMatch,
    BaselineRefMatch,
    TritonRefMatch,
)
from .repo_config import RepoConfig, SUBPROJECT_MAPPINGS, GITHUB_URL_TEMPLATES

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Baseline / Triton reference discovery -- convention-driven, NOT hardcoded
# per kernel.
#
# We do not maintain a (kernel_name -> ref_symbol) lookup table. Instead we
# rely on the *naming conventions* that aiter and vllm consistently follow
# inside their test files and triton kernel directories:
#
# Eager reference functions inside aiter/op_tests/test_*.py and
# vllm/tests/kernels/**/test_*.py are uniformly named one of:
#   - `def run_torch(...)`            (aiter, @perftest-wrapped)
#   - `def torch_<op>[_test](...)`    (aiter)
#   - `def <op>_native(...)`          (aiter pure-Python ref)
#   - `def ref_<op>(...)`             (vllm)
#   - `def naive_<op>(...)`           (vllm / triton-kernels)
#   - `def check_<op>_allclose(...)`  (aiter inline-in-test wrapper)
#   - `def test_<op>(...)`            (pytest fn whose body builds the ref
#                                      inline -- baseline_ref_kind=inline_in_test)
#
# Triton kernels are uniformly decorated with `@triton.jit` (sometimes with
# `repr=...`) immediately above a `def <symbol>(...)`.
#
# We resolve baselines as follows:
#   1. Call search_test() to get the test file Magpie already routes us to
#      (driven by the existing per-kind / per-category mapping). This is the
#      *same* test_file the user already sees in the CSV's test_file column.
#   2. ripgrep that file for the EAGER_REF_FN_REGEX patterns above.
#   3. Detect `@perftest` on the matched def -> baseline_ref_kind.
#   4. Prefer canonical names (`run_torch` > `ref_*` > `torch_*` > `*_native`
#      > `check_*_allclose` > `test_*`).
#
# Triton refs are resolved by:
#   1. Mapping the kernel's KernelCategory to a small (5-entry) triton-kernels
#      directory list -- a STRUCTURAL fact, not per-kernel hardcoding.
#   2. ripgrep that directory for `@triton.jit\\s*\\n\\s*def\\s+\\w+`.
#   3. Pick the kernel function whose name has the strongest substring
#      overlap with the profiled kernel name.
# ----------------------------------------------------------------------------

# Names we accept as an eager-reference function (matched against the bare
# function name on a `def NAME(` line). Order matters: earlier = higher
# priority when multiple candidates exist in the same file.
EAGER_REF_NAME_PRIORITY = [
    re.compile(r"^run_torch$"),
    re.compile(r"^ref_\w+$"),
    re.compile(r"^torch_\w+$"),
    re.compile(r"^naive_\w+$"),
    re.compile(r"^\w+_native$"),
    re.compile(r"^check_\w+_allclose$"),  # inline-in-test wrapper
    re.compile(r"^test_\w+$"),            # last resort: reference is inside a test body
]

# Map of {KernelCategory -> ordered list of candidate test files}. Used only
# as a fallback when `search_test()` (which keys on KernelKind) cannot route
# because the parser left kind=UNKNOWN (e.g. images that strip `.kd` suffixes).
# Pure category->file facts, not per-kernel hardcoding -- discovery of the
# actual ref symbol inside each file remains convention-driven (ripgrep for
# `run_torch` / `ref_*` / `torch_*` / `*_native`).
CATEGORY_TO_TEST_FILES = {
    KernelCategory.ATTENTION: [
        "$AITER_DIR/op_tests/test_mha.py",
        "$VLLM_DIR/tests/kernels/attention/test_attention.py",
    ],
    KernelCategory.GEMM: [
        "$AITER_DIR/op_tests/test_gemm_a16w16.py",
        "$AITER_DIR/op_tests/test_gemm_a8w8.py",
    ],
    KernelCategory.MOE_GEMM: [
        "$AITER_DIR/op_tests/test_moe.py",
    ],
    KernelCategory.LAYERNORM: [
        "$AITER_DIR/op_tests/test_rmsnorm2d.py",
        "$AITER_DIR/op_tests/test_layernorm2d.py",
    ],
    KernelCategory.SOFTMAX: [
        "$AITER_DIR/op_tests/test_topk.py",
    ],
    KernelCategory.ROUTER: [
        "$AITER_DIR/op_tests/test_moe_sorting.py",
        "$AITER_DIR/op_tests/test_topk.py",
    ],
    KernelCategory.KV_CACHE: [
        "$VLLM_DIR/tests/kernels/attention/test_cache.py",
    ],
}

# Map of {KernelCategory -> list of triton-kernels root globs (with $VAR
# prefixes)}. Pure category->directory facts, no kernel-name lookups.
# Discovery within these dirs is convention-driven via @triton.jit scanning.
CATEGORY_TO_TRITON_DIRS = {
    KernelCategory.ATTENTION: [
        "$AITER_DIR/aiter/ops/triton/_triton_kernels/attention",
    ],
    KernelCategory.GEMM: [
        "$AITER_DIR/aiter/ops/triton/_triton_kernels/gemm",
    ],
    KernelCategory.MOE_GEMM: [
        # MoE FlatMM-style kernels live under moe/ directly (moe_op_mxfp4*.py,
        # moe_op_gemm_*.py). Keep moe_routing/ out of this category --
        # router/softmax helpers are mapped under KernelCategory.ROUTER below.
        "$AITER_DIR/aiter/ops/triton/_triton_kernels/moe",
    ],
    KernelCategory.LAYERNORM: [
        "$AITER_DIR/aiter/ops/triton/_triton_kernels/normalization",
        "$AITER_DIR/aiter/ops/triton/_triton_kernels/quant",  # fused rms+quant variants
    ],
    KernelCategory.SOFTMAX: [
        "$AITER_DIR/aiter/ops/triton",  # softmax.py + topk.py live at top
        "$AITER_DIR/aiter/ops/triton/_triton_kernels",
    ],
    KernelCategory.ROUTER: [
        # moe_routing/ has topk.py + routing.py + bitmatrix.py; moe_align_block_size.py
        # sits in moe/. These two dirs cover every router/topk/softmax/sort
        # kernel aiter ships. We intentionally DO NOT include the root
        # `_triton_kernels/` here -- letting the search walk the root pulls
        # in attention/chunked_pa_prefill.py and softmax.py as false matches
        # for MoE routing kernels (P0.2 in MAGPIE_ROBUSTNESS_BACKLOG.md).
        "$AITER_DIR/aiter/ops/triton/_triton_kernels/moe/moe_routing",
        "$AITER_DIR/aiter/ops/triton/_triton_kernels/moe",
    ],
    KernelCategory.KV_CACHE: [
        # vllm v1 ships a `triton_reshape_and_cache_flash.py` here.
        # The legacy `vllm/attention/ops/` path is empty in recent
        # checkouts; we keep both for back-compat.
        "$VLLM_DIR/vllm/v1/attention/ops",
        "$VLLM_DIR/vllm/attention/ops",
    ],
    KernelCategory.REDUCE: [
        "$AITER_DIR/aiter/ops/triton/_triton_kernels",
    ],
}


# ----------------------------------------------------------------------------
# Tokenized ranker for @triton.jit reference matching.
#
# Replaces the previous longest-common-substring scorer, which produced these
# real failure modes seen in the GPT-OSS-20B CSV:
#
#   wvSplitK_hf_sml_<__hip_bfloat16,…>  -> _fused_gemm_a8w8_blockscale_split_cat
#                       (LCS matched on the literal substring "split")
#   MoeSortingMultiPhaseKernel_P0/P23   -> _kernel_paged_attention_2d
#                       (LCS matched on "kernel")
#   ck_tile::MoeFlatmmKernel<…>         -> unshuffle_weights
#                       (9-line helper picked over the real GEMM kernel in
#                        the same file, because LCS doesn't know "helper")
#
# The new scorer:
#   1. Tokenizes both sides on non-alnum boundaries AND on lower->upper
#      camel-case transitions (so "MoeSortingKernel" -> [moe, sorting,
#      kernel] and "wvSplitK" -> [wv, split, k]).
#   2. Scores by IDF-weighted token-set intersection over the candidate dir
#      -- "kernel" / "fused" / "k" / "v" appear in nearly every file and get
#      ~0 weight, while "moe", "rmsnorm", "paged", "rope" are highly
#      discriminative.
#   3. Penalizes dtype mismatch: if both sides have a hard dtype tag
#      ({fp8, fp16, bf16, fp32, int8, int4, mxfp4, a8w8, a16w16, a4w4})
#      and they don't intersect, the candidate is rejected outright.
#   4. Penalizes "helper" defs: candidates with <8 function args are
#      demoted whenever another candidate with >=8 args lives in the same
#      file and shares a meaningful token. (8 is the empirically lowest
#      arg count of the "main" @triton.jit kernels in
#      aiter/ops/triton/_triton_kernels/, where helper defs like
#      `unshuffle_weights(w, BLOCK_N, BLOCK_K)` have 3 args.)
#   5. Returns None (no guess) when the best non-rejected score is below
#      the `MIN_REF_SCORE` threshold. The CSV column is then left empty
#      instead of carrying a misleading "best-effort" pointer.
#
# Tested in tools/amd_kernel_finder/tests/test_refs.py against a golden
# corpus that includes every failure mode listed above.
# ----------------------------------------------------------------------------

# Dtype tags are split into two AXES because they can legitimately
# co-exist on a single kernel:
#
#   COMPUTE axis -- the dtype the kernel operates on internally / for
#     inputs (bf16, fp16, fp32, fp4). A kernel takes its inputs in this
#     dtype.
#
#   QUANT axis -- the kernel's output / weight quantization scheme
#     (fp8, int8, int4, blockscale, smoothquant, a8w8/a16w16/...). A
#     bf16-input kernel may have fp8 output quantization.
#
# We only hard-reject a candidate when its COMPUTE dtype is non-empty
# AND the kernel's COMPUTE dtype is non-empty AND they are disjoint --
# OR when both QUANT sets are non-empty AND disjoint. A kernel with
# only compute tags vs a candidate with only quant tags is NOT
# rejected (e.g. `add_rmsnorm_quant_kernel<bf16>` legitimately matches
# `_fused_rms_fp8_per_tensor_static_quant_kernel`: bf16 input, fp8
# quant output).
# Note: we use explicit alphanumeric-boundary lookarounds rather than `\b`
# because Python's `\b` treats `_` as a word character, so `\bfp8\b`
# does NOT match `_fp8_` -- which is exactly how aiter spells the tag in
# `_fused_rms_fp8_per_tensor_static_quant_kernel`.
_NA_BL = r"(?<![A-Za-z0-9])"  # not preceded by alnum
_NA_BR = r"(?![A-Za-z0-9])"   # not followed by alnum

_COMPUTE_DTYPE_REGEXES: List[tuple] = [
    (re.compile(r"DF16b"), "bf16"),
    (re.compile(r"DF16(?!b)"), "fp16"),
    (re.compile(r"DF32"), "fp32"),
    (re.compile(_NA_BL + r"bf16" + _NA_BR + r"|bfloat16|__hip_bfloat16"), "bf16"),
    (re.compile(_NA_BL + r"fp16" + _NA_BR + r"|__half"), "fp16"),
    (re.compile(_NA_BL + r"fp32" + _NA_BR + r"|float32"), "fp32"),
    # fp4 is a compute dtype when it appears as afp4/wfp4 (activation /
    # weight precision); standalone "fp4"/"mxfp4" also treated as compute.
    # CK mangled names spell these as F8xMXF4 / F16xMXF4 / pk_float4_e2m1
    # so we recognize those alternate spellings too.
    (re.compile(r"mxfp4|MXF4|afp4wfp4|afp4|wfp4|pk_float4|"
                + _NA_BL + r"fp4" + _NA_BR), "fp4"),
]

_QUANT_DTYPE_REGEXES: List[tuple] = [
    # `F8x...` in CK mangled names means activations FP8 (e.g.
    # F8xMXF4FlatmmPipeline = fp8 acts + mxfp4 weights).
    (re.compile(_NA_BL + r"fp8" + _NA_BR + r"|Fp8|afp8|wfp8|F8x"), "fp8"),
    (re.compile(_NA_BL + r"int8" + _NA_BR + r"|" + _NA_BL + r"i8" + _NA_BR + r"|w8a8|a8w8"), "int8"),
    (re.compile(_NA_BL + r"int4" + _NA_BR + r"|" + _NA_BL + r"i4" + _NA_BR + r"|w4a4|a4w4|a8w4|a16w4"), "int4"),
    (re.compile(r"blockscale|block_scale"), "blockscale"),
    (re.compile(r"smoothquant"), "smoothquant"),
    # GPTQ / AWQ are int4 weight-only quant schemes; treat as quant tags
    # so a pure-bf16 kernel doesn't get matched to a `*_gptq_awq` variant.
    (re.compile(_NA_BL + r"gptq" + _NA_BR + r"|GPTQ"), "gptq"),
    (re.compile(_NA_BL + r"awq" + _NA_BR + r"|AWQ"), "awq"),
]


def _extract_compute_dtypes(s: str) -> frozenset:
    """Extract normalized COMPUTE dtype tags (bf16/fp16/fp32/fp4) from `s`."""
    if not s:
        return frozenset()
    out = set()
    for rx, tag in _COMPUTE_DTYPE_REGEXES:
        if rx.search(s):
            out.add(tag)
    return frozenset(out)


def _extract_quant_dtypes(s: str) -> frozenset:
    """Extract normalized QUANT dtype/mode tags (fp8/int8/int4/blockscale/...)."""
    if not s:
        return frozenset()
    out = set()
    for rx, tag in _QUANT_DTYPE_REGEXES:
        if rx.search(s):
            out.add(tag)
    return frozenset(out)

# Tokens that are too common across triton kernel filenames to be informative;
# they get near-zero idf weight regardless of corpus.
_STOPWORD_TOKENS = frozenset({
    "kernel", "fused", "op", "ops", "k", "v", "d", "_", "the",
})

# Minimum IDF-weighted intersection score for a triton ref to be emitted.
# Below this, the searcher returns None rather than a misleading guess.
MIN_REF_SCORE = 1.0

# Empirical floor for "this is the main kernel, not a 3-arg helper". The
# `_moe_gemm_int8_smoothquant` / `_gemm_a16_w16_kernel` / etc. all carry
# >=15 args; helpers like `unshuffle_weights(w, BLOCK_N, BLOCK_K)` carry 3.
_MAIN_KERNEL_MIN_ARGS = 8

# Operation-defining tokens per category. Both the candidate AND the
# profiled kernel must contain at least one of these tokens (matched via
# stem-prefix) for the candidate to be eligible. Bypassed on strong
# (>= 2 non-stopword) direct token overlap. This is the rule that keeps
# `_moe_align_block_size_stage4_kernel` (a routing kernel that happens
# to live under moe/) off MoE FlatMM (GEMM) rows -- they share `moe`
# but the routing kernel lacks any GEMM-defining token.
#
# Per-category candidate blocklist: tokens that, if present on the
# candidate, disqualify it for this category. This complements the
# primary-token gate for cases where the canonical aiter triton kernel
# uses a different vocabulary than the CK kernel (e.g. CK MoE FlatMM
# uses "flatmm" but aiter spells its MoE GEMMs "_fused_moe_kernel_*"
# without a "gemm" token).
_CATEGORY_PRIMARY_TOKENS: Dict = {}
_CATEGORY_CAND_BLOCKLIST: Dict = {}
_CATEGORY_CAND_ALLOWLIST: Dict = {}


def _category_tables_init():
    if _CATEGORY_PRIMARY_TOKENS:
        return
    from .models import KernelCategory as _KC
    _CATEGORY_PRIMARY_TOKENS.update({
        _KC.ATTENTION: frozenset({
            "attention", "attn", "mha", "fmha", "decode", "prefill", "paged",
        }),
        _KC.GEMM: frozenset({
            "gemm", "matmul", "mm", "wmma",
        }),
        _KC.MOE_GEMM: frozenset({
            # accept any of: explicit gemm-flavored tokens (CK side)
            # OR the aiter MoE-matmul naming "fused_moe_kernel"
            "gemm", "matmul", "flatmm", "mm",
            "fused",   # aiter spells all MoE matmuls _fused_moe_kernel*
            "moeop", "moe_op",
        }),
        _KC.LAYERNORM: frozenset({
            "rmsnorm", "layernorm", "norm", "rms",
        }),
        _KC.SOFTMAX: frozenset({
            "softmax",
        }),
        _KC.ROUTER: frozenset({
            "align", "topk", "routing", "sort", "gating", "bitmatrix", "block",
        }),
        _KC.KV_CACHE: frozenset({
            "cache", "reshape",
        }),
        _KC.REDUCE: frozenset({
            "reduce", "sum", "argmax", "argmin",
        }),
    })
    _CATEGORY_CAND_BLOCKLIST.update({
        # MoE FlatMM rows are NOT routing kernels; reject sort/align/topk
        # candidates even when they live under moe/.
        _KC.MOE_GEMM: frozenset({"align", "sort", "topk", "gating", "bitmatrix"}),
        # ROUTER: never resolve to a GEMM or attention kernel.
        _KC.ROUTER: frozenset({"gemm", "matmul", "flatmm", "attention", "mha"}),
    })


def _category_primary_tokens(cat) -> frozenset:
    _category_tables_init()
    return _CATEGORY_PRIMARY_TOKENS.get(cat, frozenset())


def _category_blocklist(cat) -> frozenset:
    _category_tables_init()
    return _CATEGORY_CAND_BLOCKLIST.get(cat, frozenset())


# Compound-word stem aliases. Pre-existing token -> additional tokens to
# add. Keeps the tokenizer's "split-on-boundary" rule from missing the
# fact that `rmsnorm` is `rms`+`norm`, `bfloat16` is `bf16`, etc.
_TOKEN_ALIASES: Dict = {
    "rmsnorm": ("rms", "norm"),
    "layernorm": ("layer", "norm"),
    "softmax": ("soft", "max"),
    "bfloat16": ("bf16",),
    "rmsnorm2d": ("rms", "norm", "rmsnorm"),
    "matmul": ("mm",),
}


def _tokenize_symbol(s: str) -> List[str]:
    """Tokenize a symbol or kernel name into lowercased word tokens.

    Splits on:
      - any non-alphanumeric character (handles `_`, `<`, `>`, `,`, `:`, ...)
      - lower->upper camel-case transitions (so "MoeSortingKernel" yields
        ["moe", "sorting", "kernel"], "wvSplitK" yields ["wv", "split", "k"]).

    After splitting we expand common compound-word stems (see
    `_TOKEN_ALIASES`) so that "rmsnorm" also matches "rms" and "norm".

    Empty strings and single-character tokens that are not dtype-significant
    are dropped to reduce noise. Numeric tokens are kept (e.g. "16" in
    "a16w16") because they help discriminate dtype/quant variants.
    """
    if not s:
        return []
    # Insert spaces at lower->upper boundaries (camelCase -> camel Case).
    s2 = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    raw = re.split(r"[^A-Za-z0-9]+", s2)
    out: List[str] = []
    seen = set()
    for t in raw:
        if not t:
            continue
        tl = t.lower()
        if len(tl) <= 1 and tl not in {"k", "v", "q"}:
            # drop single-letter noise except for q/k/v (used in attention)
            continue
        if tl not in seen:
            out.append(tl)
            seen.add(tl)
        for alias in _TOKEN_ALIASES.get(tl, ()):
            if alias not in seen:
                out.append(alias)
                seen.add(alias)
    return out


def _scan_triton_dir(
    dir_path: Path,
) -> List[tuple]:
    """Recursively scan a triton-kernels directory and return all kernels.

    Returns a list of (file_path, sym, lineno, num_args) tuples for every
    `@triton.jit`-decorated def found. `num_args` counts comma-separated
    arguments in the def signature (across multi-line signatures) and is
    used by the ranker to distinguish main kernels from helpers.
    """
    out: List[tuple] = []
    def_re = re.compile(r"^def\s+(\w+)\s*\(")
    for py_file in sorted(dir_path.rglob("*.py")):
        if "__pycache__" in str(py_file) or py_file.name == "__init__.py":
            continue
        try:
            text = py_file.read_text(errors="ignore")
        except OSError:
            continue
        lines = text.splitlines()
        pending = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("@triton.jit"):
                pending = True
                continue
            if pending:
                m = def_re.match(line)
                if m:
                    sym = m.group(1)
                    # Count args by scanning forward until the closing ).
                    # We accumulate the raw text between ( and ), strip
                    # comments, and split on commas.
                    sig_buf: List[str] = []
                    open_idx = line.find("(", m.end() - 1)
                    if open_idx < 0:
                        out.append((py_file, sym, i + 1, 0))
                        pending = False
                        continue
                    sig_buf.append(line[open_idx + 1:])
                    j = i
                    depth = line.count("(", open_idx) - line.count(")", open_idx)
                    while depth > 0 and j + 1 < len(lines):
                        j += 1
                        sig_buf.append(lines[j])
                        depth += lines[j].count("(") - lines[j].count(")")
                    # NB: join with "\n" (not " ") so that the
                    # comment-stripper below only consumes one line
                    # of trailing text, not the whole signature.
                    sig = "\n".join(sig_buf)
                    close = sig.rfind(")")
                    if close >= 0:
                        sig = sig[:close]
                    sig = re.sub(r"#[^\n]*", "", sig)
                    # split top-level commas only (ignore commas inside [] or ())
                    args = _split_top_level_commas(sig)
                    args = [a.strip() for a in args if a.strip()]
                    out.append((py_file, sym, i + 1, len(args)))
                    pending = False
                    continue
                # decorator args may span multiple lines; tolerate
                if not stripped or stripped.endswith(",") or stripped.startswith(")"):
                    continue
                if stripped.startswith("@"):
                    continue
                pending = False
    return out


def _split_top_level_commas(s: str) -> List[str]:
    """Split `s` on top-level commas (ignoring those inside [] / ())."""
    out: List[str] = []
    depth = 0
    start = 0
    for i, c in enumerate(s):
        if c in "([":
            depth += 1
        elif c in ")]":
            if depth > 0:
                depth -= 1
        elif c == "," and depth == 0:
            out.append(s[start:i])
            start = i + 1
    out.append(s[start:])
    return out


def _score_candidate(
    kernel_tokens: List[str],
    kernel_compute: frozenset,
    kernel_quant: frozenset,
    kernel_has_quant_intent: bool,
    sym_tokens: List[str],
    sym_compute: frozenset,
    sym_quant: frozenset,
    idf: Dict[str, float],
    num_args: int,
    has_main_in_file: bool,
    primary_tokens: frozenset = frozenset(),
    blocklist_tokens: frozenset = frozenset(),
) -> Optional[float]:
    """Return a score >= 0, or None if the candidate is hard-rejected.

    Hard reject when:
      * compute dtypes are both non-empty AND disjoint
        (e.g. profiled {bf16} vs candidate {fp4})
      * OR quant dtypes are both non-empty AND disjoint
        (e.g. profiled {int8} vs candidate {fp8})

    Crucially, a compute-only kernel (e.g. bf16) is NOT rejected against
    a quant-only candidate (e.g. fp8 quantization) -- those legitimately
    coexist (a bf16-input kernel can have an fp8 output quant).

    Helper penalty: candidates with <`_MAIN_KERNEL_MIN_ARGS` args are
    halved when another candidate in the same file has >= that many args.

    Dtype-alignment bonus: shared compute or quant tags add a fixed bonus.

    Category primary-token gate: see body. Bypassed on strong (>=2
    non-stopword) direct overlap.
    """
    s_set = set(sym_tokens)
    if not s_set:
        return None

    # Per-category candidate blocklist (cross-category contamination).
    if blocklist_tokens and (s_set & blocklist_tokens):
        return None

    if (kernel_compute and sym_compute
            and kernel_compute.isdisjoint(sym_compute)):
        return None  # compute-dtype mismatch
    if (kernel_quant and sym_quant
            and kernel_quant.isdisjoint(sym_quant)):
        return None  # quant-dtype mismatch

    # Quant-asymmetry rejection: when the profiled kernel has NO quant
    # signal at all (no quant dtype, no "quant" token, no a8w8/a16w4-style
    # marker) but the candidate IS a quantized kernel, they are different
    # operations and the candidate must be rejected. This is the rule
    # that keeps `ck_tile::MoeFlatmmKernel<DF16b,…>` (pure bf16 MoE GEMM)
    # off `_moe_gemm_a16w4` (int4 quant), while still letting
    # `add_rmsnorm_quant_kernel<bf16>` (has "quant" token) match the
    # fp8-quantizing triton ref.
    if sym_quant and not kernel_has_quant_intent:
        return None

    k_set = set(kernel_tokens)
    intersect = k_set & s_set

    strong_overlap = (
        sum(1 for t in intersect if t not in _STOPWORD_TOKENS) >= 2
    )
    if primary_tokens and not strong_overlap:
        def _has_primary(toks):
            return any(
                any(t.startswith(p) or p.startswith(t) for p in primary_tokens)
                for t in toks
            )
        if not _has_primary(s_set):
            return None
        if not _has_primary(k_set):
            return None

    if not intersect:
        return 0.0

    score = sum(idf.get(t, 1.0) for t in intersect)

    if kernel_compute and sym_compute and (kernel_compute & sym_compute):
        score += 1.5
    if kernel_quant and sym_quant and (kernel_quant & sym_quant):
        score += 1.0
    # Quant-intent alignment: when the kernel says "I am a quantization
    # kernel" (via the `quant` token or an explicit quant dtype) and the
    # candidate is a quantizing triton kernel (has a sym_quant tag from
    # its name or filename), push it up. This is what makes
    # `aiter::add_rmsnorm_quant_kernel<bf16>` prefer
    # `_fused_rms_fp8_per_tensor_static_quant_kernel` over
    # `_fused_add_rmsnorm_pad` even when the latter has more raw
    # token overlap.
    if kernel_has_quant_intent and sym_quant:
        score += 3.0

    if num_args < _MAIN_KERNEL_MIN_ARGS and has_main_in_file:
        score *= 0.5

    return score


class KernelSourceSearcher:
    """Search for kernel source files in repositories."""
    
    def __init__(self, repos: List[str], repo_configs: Dict[str, RepoConfig] = None,
                 auto_install_ripgrep: bool = True):
        """
        Initialize searcher with repository paths.
        
        Args:
            repos: List of repository root paths
            repo_configs: Optional custom repo configurations
            auto_install_ripgrep: If True, attempt to install ripgrep if missing
        """
        self.repos = repos
        self.repo_configs = repo_configs or {}
        self._repo_var_map: Dict[str, str] = {}
        
        # Check/install ripgrep
        self._has_ripgrep = self._check_ripgrep()
        if not self._has_ripgrep and auto_install_ripgrep:
            self._has_ripgrep = self._ensure_ripgrep()
        
        if not self._has_ripgrep:
            logger.info("ripgrep not available, using Python fallback for searches")
        
        # Build repo variable map
        self._build_repo_var_map()
    
    def _check_ripgrep(self) -> bool:
        """Check if ripgrep is available."""
        try:
            result = subprocess.run(
                ["rg", "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _ensure_ripgrep(self) -> bool:
        """Attempt to install ripgrep if missing."""
        logger.info("ripgrep not found, attempting to install...")
        
        install_cmds = [
            (["apt-get", "update"], ["apt-get", "install", "-y", "ripgrep"]),
            (None, ["yum", "install", "-y", "ripgrep"]),
            (None, ["dnf", "install", "-y", "ripgrep"]),
            (None, ["brew", "install", "ripgrep"]),
            (None, ["cargo", "install", "ripgrep"]),
        ]
        
        for pre_cmd, install_cmd in install_cmds:
            try:
                if pre_cmd:
                    subprocess.run(pre_cmd, capture_output=True, timeout=60)
                
                result = subprocess.run(
                    install_cmd,
                    capture_output=True,
                    timeout=300,
                )
                
                if result.returncode == 0 and self._check_ripgrep():
                    logger.info(f"ripgrep installed successfully via {install_cmd[0]}")
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
                continue
        
        logger.warning("Could not install ripgrep automatically")
        return False
    
    def _run_python_search(self, pattern: str, search_path: str,
                           file_extensions: List[str] = None,
                           max_results: int = 5) -> List[str]:
        """
        Python-native search fallback using glob + re.
        
        Args:
            pattern: Regex pattern to search for
            search_path: Directory to search in
            file_extensions: List of extensions (e.g., ["py", "cpp"])
            max_results: Maximum results to return
            
        Returns:
            List of matching file paths
        """
        if not Path(search_path).exists():
            return []
        
        results = []
        extensions = file_extensions or ["py", "cpp", "cu", "hip", "hpp"]
        
        try:
            compiled_pattern = re.compile(pattern, re.MULTILINE)
        except re.error:
            pattern = re.escape(pattern)
            compiled_pattern = re.compile(pattern, re.MULTILINE)
        
        for ext in extensions:
            glob_pattern = f"{search_path}/**/*.{ext}"
            for filepath in glob.iglob(glob_pattern, recursive=True):
                if ".git" in filepath or "__pycache__" in filepath:
                    continue
                
                try:
                    with open(filepath, 'r', errors='ignore') as f:
                        content = f.read()
                        if compiled_pattern.search(content):
                            results.append(filepath)
                            if len(results) >= max_results:
                                return results
                except (IOError, OSError):
                    continue
        
        return results
    
    def _search_files(self, pattern: str, search_path: str,
                      file_types: List[str] = None,
                      max_results: int = 5) -> List[str]:
        """
        Search files using ripgrep with Python fallback.
        
        Args:
            pattern: Search pattern
            search_path: Directory to search
            file_types: List of file type filters for ripgrep
            max_results: Maximum results
            
        Returns:
            List of matching file paths
        """
        if self._has_ripgrep:
            results = self._run_ripgrep(pattern, search_path, file_types, max_results)
            if results:
                return results
        
        ext_map = {
            "py": "py",
            "cpp": "cpp",
            "cu": "cu",
            "hip": "hip",
            "hpp": "hpp",
        }
        extensions = [ext_map.get(ft, ft) for ft in (file_types or [])]
        return self._run_python_search(pattern, search_path, extensions, max_results)
    
    def _build_repo_var_map(self):
        """Build mapping from repo variable names to actual paths."""
        for repo_path in self.repos:
            from .repo_config import detect_repo_type, REPO_CONFIGS
            repo_type = detect_repo_type(repo_path)
            if repo_type and repo_type in REPO_CONFIGS:
                config = REPO_CONFIGS[repo_type]
                self._repo_var_map[config.var_name] = repo_path
                
                # Add subproject mappings
                for subvar, (parent_var, subpath) in SUBPROJECT_MAPPINGS.items():
                    if config.var_name == parent_var:
                        self._repo_var_map[subvar] = str(Path(repo_path) / subpath)
    
    def get_repo_paths(self) -> Dict[str, str]:
        """Get mapping of repo variable names to actual paths."""
        return self._repo_var_map.copy()
    
    def search_source(self, parsed: ParsedKernelName) -> Optional[SourceMatch]:
        """
        Search for kernel source file.
        
        Args:
            parsed: Parsed kernel name information
            
        Returns:
            SourceMatch if found, None otherwise
        """
        if parsed.kind == KernelKind.ANNOTATION:
            return None
        
        if parsed.kind == KernelKind.TRITON_JIT:
            return self._search_triton_source(parsed)
        elif parsed.kind == KernelKind.TENSILE_GEMM:
            return self._search_tensile_source(parsed)
        elif parsed.kind == KernelKind.CK_TILE:
            return self._search_ck_source(parsed)
        elif parsed.kind == KernelKind.ATEN_NATIVE:
            return self._search_aten_source(parsed)
        elif parsed.kind == KernelKind.HIP_CPP:
            return self._search_hip_source(parsed)
        elif parsed.kind == KernelKind.INDUCTOR:
            return self._search_inductor_source(parsed)
        elif parsed.kind == KernelKind.AITER:
            return self._search_aiter_source(parsed)
        
        return None
    
    def search_test(self, parsed: ParsedKernelName, source: Optional[SourceMatch] = None) -> Optional[TestMatch]:
        """
        Search for test files and generate test command.
        
        Args:
            parsed: Parsed kernel name information
            source: Optional source match for context
            
        Returns:
            TestMatch if found, None otherwise
        """
        if parsed.kind == KernelKind.ANNOTATION:
            return None
        
        if parsed.kind == KernelKind.TRITON_JIT:
            return self._search_triton_test(parsed, source)
        elif parsed.kind == KernelKind.TENSILE_GEMM:
            return self._search_tensile_test(parsed)
        elif parsed.kind == KernelKind.CK_TILE:
            return self._search_ck_test(parsed)
        elif parsed.kind == KernelKind.ATEN_NATIVE:
            return self._search_aten_test(parsed)
        elif parsed.kind == KernelKind.HIP_CPP:
            return self._search_hip_test(parsed, source)
        elif parsed.kind == KernelKind.AITER:
            return self._search_aiter_test(parsed, source)
        
        return None

    # ------------------------------------------------------------------
    # Baseline (PyTorch / Triton) reference search
    # ------------------------------------------------------------------
    # ======================================================================
    # Convention-driven baseline-ref + Triton-ref discovery.
    #
    # We do NOT maintain a (kernel_name -> ref_symbol) lookup table. Instead:
    #   - For the eager baseline, we reuse `search_test()` to locate the
    #     existing test file (which Magpie already routes via per-category
    #     keyword tables in `_search_*_test`), then ripgrep that file for
    #     functions named `run_torch`/`ref_*`/`torch_*`/`*_native`/etc.
    #   - For the Triton ref, we map KernelCategory -> small list of triton-
    #     kernels root dirs (5 entries), then ripgrep for `@triton.jit\ndef`.
    # ======================================================================

    def search_baseline_ref(
        self,
        parsed: ParsedKernelName,
        source: Optional[SourceMatch] = None,
        category: Optional[KernelCategory] = None,
        test_match: Optional[TestMatch] = None,
    ) -> Optional[BaselineRefMatch]:
        """Resolve a PyTorch eager baseline by convention-driven discovery.

        Strategy:
          1. Special structural cases (Memcpy markers, ATen, Inductor,
             annotations) -> hardcoded structural answers, NOT per-kernel.
          2. Otherwise, get the test file from `search_test()` (or use the
             one already computed and passed in), then scan it for known
             ref-fn name patterns and emit (file, symbol, kind).

        Args:
            parsed: Parsed kernel information.
            source: Optional source match (unused; kept for API compat).
            category: Optional KernelCategory (used for some structural
                fallbacks).
            test_match: Optional precomputed TestMatch from `search_test()`.
                If absent, we recompute it.

        Returns:
            BaselineRefMatch or None.
        """
        if parsed.kind == KernelKind.ANNOTATION:
            return None

        # Structural answer: ATen kernels ARE the eager op.
        if parsed.kind == KernelKind.ATEN_NATIVE:
            symbol = parsed.function_name or parsed.original_name
            return BaselineRefMatch(
                ref_file="(ATen kernel is itself the PyTorch eager op)",
                ref_symbol=f"torch.* equivalent of {symbol[:64]}",
                repo_var="",
                kind="eager_fn",
                notes="No separate baseline: this is PyTorch eager. Compare against a "
                      "torch.* call with the same inputs.",
            )

        # Structural answer: inductor codegen -> dump recipe.
        if parsed.kind == KernelKind.INDUCTOR:
            return BaselineRefMatch(
                ref_file="(torch.compile generated; dump with TORCH_LOGS=output_code)",
                ref_symbol="(eager pre-fusion graph)",
                repo_var="",
                kind="none",
                notes="No canned baseline. Re-run with TORCH_LOGS='output_code,inductor' "
                      "and TORCHINDUCTOR_TRACE_DIR=<dir> to dump the eager pre-fusion "
                      "graph as the baseline.",
            )

        # Profiler markers (e.g. Memcpy DtoD) -- not a kernel at all.
        original_lc = parsed.original_name.lower()
        if original_lc.startswith("memcpy") or "memcpy " in original_lc:
            return BaselineRefMatch(
                ref_file="",
                ref_symbol="(profiler marker, not a kernel)",
                kind="none",
                notes="Memcpy markers correspond to torch.Tensor.copy_() / hipMemcpyAsync; "
                      "no separate baseline kernel exists",
            )

        # Primary path: ripgrep the test file Magpie already routed us to.
        if test_match is None:
            test_match = self.search_test(parsed, source)
        if test_match is not None and test_match.test_file:
            m = self._discover_eager_ref_in_file(test_match.display_path)
            if m is not None:
                return m

        # Category fallback: covers kernels whose KernelKind the parser
        # couldn't pin down (most often because the runtime stripped the
        # `.kd` / `[clone .kd]` suffix). We try the per-category candidate
        # test files in order; first one that yields a discovered ref wins.
        cat = category or KernelCategory.UNKNOWN
        for test_display in CATEGORY_TO_TEST_FILES.get(cat, []):
            m = self._discover_eager_ref_in_file(test_display)
            if m is not None:
                return m

        return None

    def search_triton_ref(
        self,
        parsed: ParsedKernelName,
        source: Optional[SourceMatch] = None,
        category: Optional[KernelCategory] = None,
    ) -> Optional[TritonRefMatch]:
        """Resolve the canonical Triton kernel implementation for this kernel.

        Strategy: map `category` -> list of `_triton_kernels/<op>/` dirs,
        then ripgrep those dirs for `@triton.jit` -> `def NAME(` and pick
        the function whose name has the strongest substring overlap with the
        profiled kernel name.

        Returns None when no Triton implementation exists for this category
        (e.g. ATen elementwise, profiler markers, plain copy/index kernels).
        """
        if parsed.kind == KernelKind.ANNOTATION:
            return None
        if parsed.kind == KernelKind.ATEN_NATIVE:
            return None

        cat = category or KernelCategory.UNKNOWN
        dirs = CATEGORY_TO_TRITON_DIRS.get(cat)
        if not dirs:
            return None

        return self._discover_triton_ref_in_dirs(
            dirs, parsed.original_name, category=cat
        )

    # ------------------------------------------------------------------
    # Path resolution and file-scanning helpers (convention-driven)
    # ------------------------------------------------------------------
    def _resolve_repo_path(self, display_path: str) -> Optional[Path]:
        """Turn a `$REPO_VAR/relative/path` string into an absolute Path.

        Returns None if the repo var is unknown or the path does not exist.
        """
        if not display_path:
            return None
        if display_path.startswith("$"):
            parts = display_path.split("/", 1)
            var = parts[0]
            rel = parts[1] if len(parts) > 1 else ""
            base = self._repo_var_map.get(var)
            if not base:
                return None
            p = Path(base) / rel
        else:
            p = Path(display_path)
        return p if p.exists() else None

    @staticmethod
    def _scan_eager_refs(file_path: Path) -> List[tuple]:
        """Scan a Python test file for eager-reference function defs.

        Returns a list of (priority_idx, name, has_perftest, lineno) tuples,
        one per candidate function. Lower priority_idx = better match.
        """
        try:
            text = file_path.read_text(errors="ignore")
        except Exception:
            return []
        lines = text.splitlines()

        def_re = re.compile(r"^def\s+(\w+)\s*\(")
        out: List[tuple] = []
        for i, line in enumerate(lines):
            m = def_re.match(line)
            if not m:
                continue
            name = m.group(1)
            prio = None
            for idx, pat in enumerate(EAGER_REF_NAME_PRIORITY):
                if pat.match(name):
                    prio = idx
                    break
            if prio is None:
                continue
            # Walk backwards to detect any @perftest decorator.
            has_perftest = False
            for j in range(i - 1, max(-1, i - 12), -1):
                stripped = lines[j].strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("@"):
                    if "perftest" in stripped:
                        has_perftest = True
                    continue
                break
            out.append((prio, name, has_perftest, i + 1))
        out.sort(key=lambda t: (t[0], t[3]))  # priority asc, then line asc
        return out

    def _discover_eager_ref_in_file(self, display_path: str) -> Optional[BaselineRefMatch]:
        """Open the file behind `display_path` and return its best eager ref."""
        abs_path = self._resolve_repo_path(display_path)
        if abs_path is None or abs_path.is_dir():
            # display_path can be a *directory* (e.g. "op_tests/") when the
            # test searcher couldn't pin a single file. We can't scan dirs
            # for a "the" ref symbol, so return None.
            return None

        candidates = self._scan_eager_refs(abs_path)
        if not candidates:
            return None

        prio, name, has_perftest, lineno = candidates[0]
        if name.startswith("test_"):
            kind = "inline_in_test"
            notes = (
                f"Reference is built inline inside the parametrized test body. "
                f"Read {abs_path.name} starting at line {lineno}."
            )
        elif has_perftest:
            kind = "perftest_wrapper"
            notes = (
                f"@perftest-decorated wrapper at line {lineno}; importable, gives "
                f"A/B timing for free."
            )
        else:
            kind = "eager_fn"
            notes = f"Pure-PyTorch eager reference at line {lineno}."

        # Reconstruct the display path with $REPO_VAR prefix preserved.
        return BaselineRefMatch(
            ref_file=display_path.split("/", 1)[1] if display_path.startswith("$") else display_path,
            ref_symbol=name,
            repo_var=display_path.split("/", 1)[0] if display_path.startswith("$") else "",
            kind=kind,
            notes=notes,
        )

    def _discover_triton_ref_in_dirs(
        self,
        dir_display_paths: List[str],
        kernel_name: str,
        category: Optional[KernelCategory] = None,
    ) -> Optional[TritonRefMatch]:
        """Pick the best @triton.jit kernel from the candidate directories.

        Uses the tokenized scorer documented above (`_score_candidate`):
          - IDF-weighted token-set intersection over the candidate dir
          - hard reject on dtype mismatch
          - helper penalty when a higher-arg-count kernel lives in the
            same file
          - returns None below `MIN_REF_SCORE` rather than emitting a
            misleading "best-effort" pointer

        See the module-level comment block (above `_DTYPE_TOKENS`) for the
        list of failure modes this replaces.
        """
        kernel_tokens = _tokenize_symbol(kernel_name)
        if not kernel_tokens:
            return None
        kernel_compute = _extract_compute_dtypes(kernel_name)
        kernel_quant = _extract_quant_dtypes(kernel_name)
        # "quant intent" -- does the kernel mention quantization at all?
        # Used by `_score_candidate` to reject quantized triton candidates
        # for pure-compute kernels (e.g. CK MoE FlatMM is pure bf16; should
        # not match a int4-quantizing triton MoE GEMM).
        kernel_has_quant_intent = bool(kernel_quant) or bool(
            re.search(r"(?i)quant", kernel_name)
        )
        primary_tokens = (
            _category_primary_tokens(category) if category is not None else frozenset()
        )
        blocklist_tokens = (
            _category_blocklist(category) if category is not None else frozenset()
        )

        # Pass 1: collect every candidate across all candidate dirs. We
        # need the full corpus before we can compute idf weights.
        candidates: List[tuple] = []
        # tuple shape: (file_path, file_display, sym, lineno, num_args,
        #               sym_tokens, sym_compute, sym_quant)
        for dir_display in dir_display_paths:
            dir_path = self._resolve_repo_path(dir_display)
            if dir_path is None or not dir_path.is_dir():
                continue
            var_prefix = dir_display.split("/", 1)[0]
            base_path = self._repo_var_map.get(var_prefix)
            if not base_path:
                continue
            for py_file, sym, lineno, num_args in _scan_triton_dir(dir_path):
                try:
                    rel = py_file.relative_to(base_path)
                    file_display = f"{var_prefix}/{rel}"
                except ValueError:
                    file_display = str(py_file)
                # Dtype tags come from BOTH the symbol name AND its file
                # path (helper kernels in `gemm_afp4wfp4.py` inherit fp4
                # from the filename even if their own symbol doesn't say
                # so explicitly).
                name_and_file = sym + " " + py_file.name
                sym_compute = _extract_compute_dtypes(name_and_file)
                sym_quant = _extract_quant_dtypes(name_and_file)
                candidates.append(
                    (py_file, file_display, sym, lineno, num_args,
                     _tokenize_symbol(sym), sym_compute, sym_quant)
                )

        if not candidates:
            return None

        # Compute idf over candidate symbol tokens.
        import math
        N = len(candidates)
        df: Dict[str, int] = {}
        for _, _, _, _, _, sym_tokens, _, _ in candidates:
            for t in set(sym_tokens):
                df[t] = df.get(t, 0) + 1
        idf: Dict[str, float] = {}
        for t, count in df.items():
            if t in _STOPWORD_TOKENS:
                idf[t] = 0.1
            else:
                w = math.log((N + 1) / (count + 0.5)) + 0.5
                idf[t] = max(0.5, min(4.0, w))

        files_with_main: set = set()
        for py_file, _, _, _, num_args, _, _, _ in candidates:
            if num_args >= _MAIN_KERNEL_MIN_ARGS:
                files_with_main.add(py_file)

        best: Optional[tuple] = None  # (score, file_display, sym, lineno)
        for (py_file, file_display, sym, lineno, num_args,
             sym_tokens, sym_compute, sym_quant) in candidates:
            score = _score_candidate(
                kernel_tokens=kernel_tokens,
                kernel_compute=kernel_compute,
                kernel_quant=kernel_quant,
                kernel_has_quant_intent=kernel_has_quant_intent,
                sym_tokens=sym_tokens,
                sym_compute=sym_compute,
                sym_quant=sym_quant,
                idf=idf,
                num_args=num_args,
                has_main_in_file=(py_file in files_with_main),
                primary_tokens=primary_tokens,
                blocklist_tokens=blocklist_tokens,
            )
            if score is None:
                continue  # hard-rejected (e.g. dtype mismatch)
            if best is None or score > best[0]:
                best = (score, file_display, sym, lineno)

        if best is None:
            return None

        score, file_display, sym, lineno = best
        if score < MIN_REF_SCORE:
            # No candidate cleared the confidence floor; better to leave
            # the column empty than to emit a misleading pointer.
            return None

        note = (
            f"Canonical Triton @triton.jit kernel `{sym}` at line {lineno} "
            f"(score={score:.2f})."
        )
        return TritonRefMatch(
            ref_file=file_display,
            ref_symbol=sym,
            repo_var="",  # ref_file already includes the $..._DIR prefix
            notes=note,
        )
    
    def _run_ripgrep(self, pattern: str, search_path: str, 
                     file_types: List[str] = None, max_results: int = 5) -> List[str]:
        """Run ripgrep and return matching files."""
        if not Path(search_path).exists():
            return []
        
        cmd = ["rg", "-l", "--max-count", "1"]
        
        if file_types:
            for ft in file_types:
                cmd.extend(["--type", ft])
        
        cmd.extend([pattern, search_path])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                files = result.stdout.strip().split('\n')
                return [f for f in files if f][:max_results]
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.warning(f"ripgrep failed: {e}")
        
        return []
    
    def _search_triton_source(self, parsed: ParsedKernelName) -> Optional[SourceMatch]:
        """Search for Triton JIT kernel source."""
        function_name = parsed.function_name
        
        # Known kernel mappings for common kernels
        # $TRITON_KERNELS_DIR = triton/python/triton_kernels/
        known_mappings = {
            "_matmul_ogs": ("triton_kernels/matmul_details/_matmul.py", "$TRITON_KERNELS_DIR"),
            "_matmul": ("triton_kernels/matmul_details/_matmul.py", "$TRITON_KERNELS_DIR"),
            "_reduce": ("triton_kernels/reduce.py", "$TRITON_KERNELS_DIR"),
            "kernel_unified_attention": ("vllm/v1/attention/ops/triton_unified_attention.py", "$VLLM_DIR"),
            "_topk_forward": ("triton_kernels/topk_details/_topk_forward.py", "$TRITON_KERNELS_DIR"),
            "_topk_backward": ("triton_kernels/topk_details/_topk_backward.py", "$TRITON_KERNELS_DIR"),
            "_bitmatrix_metadata": ("triton_kernels/tensor_details/", "$TRITON_KERNELS_DIR"),
            "_ragged_tensor_metadata": ("triton_kernels/tensor_details/", "$TRITON_KERNELS_DIR"),
            "_sum_bitmatrix_rows": ("triton_kernels/tensor_details/", "$TRITON_KERNELS_DIR"),
            "_fused_add_rmsnorm": ("triton_kernels/swiglu_details/", "$TRITON_KERNELS_DIR"),
            "_swiglu": ("triton_kernels/swiglu_details/", "$TRITON_KERNELS_DIR"),
            "_compaction": ("triton_kernels/compaction_details/", "$TRITON_KERNELS_DIR"),
        }
        
        # Check known mappings first
        for key, (path, repo_var) in known_mappings.items():
            if key in function_name:
                return SourceMatch(
                    file_path=path,
                    symbol=function_name,
                    repo_name="triton_kernels",
                    repo_var=repo_var,
                )
        
        # Search patterns
        patterns = [
            f"def {function_name}",
            f"@triton.jit.*\\n.*def {function_name}",
            f'def {function_name}\\(',
        ]
        
        # Search in triton repos
        triton_path = self._repo_var_map.get("$TRITON_DIR")
        if triton_path:
            for pattern in patterns:
                files = self._run_ripgrep(pattern, triton_path, ["py"])
                if files:
                    rel_path = os.path.relpath(files[0], triton_path)
                    return SourceMatch(
                        file_path=rel_path,
                        symbol=function_name,
                        repo_name="triton",
                        repo_var="$TRITON_DIR",
                    )
        
        # Search in rocm-libraries (triton_kernels might be there)
        rocm_libs = self._repo_var_map.get("$ROCM_LIBRARIES_DIR")
        if rocm_libs:
            for pattern in patterns:
                files = self._run_ripgrep(pattern, rocm_libs, ["py"])
                if files:
                    rel_path = os.path.relpath(files[0], rocm_libs)
                    return SourceMatch(
                        file_path=rel_path,
                        symbol=function_name,
                        repo_name="rocm-libraries",
                        repo_var="$ROCM_LIBRARIES_DIR",
                    )
        
        # Default fallback for triton kernels
        return SourceMatch(
            file_path="(search in triton_kernels or vllm)",
            symbol=function_name,
            repo_name="triton",
            repo_var="$TRITON_DIR",
        )
    
    def _search_tensile_source(self, parsed: ParsedKernelName) -> Optional[SourceMatch]:
        """Search for Tensile GEMM source (logic YAML files)."""
        rocm_libs = self._repo_var_map.get("$ROCM_LIBRARIES_DIR")
        if not rocm_libs:
            return None
        
        # Tensile kernels are generated, point to logic files
        tensile_logic_path = Path(rocm_libs) / "projects/rocblas/library/src/blas3/Tensile/Logic"
        if tensile_logic_path.exists():
            return SourceMatch(
                file_path="projects/rocblas/library/src/blas3/Tensile/Logic/asm_full/",
                symbol="Tensile-generated kernel (asm)",
                repo_name="rocm-libraries",
                repo_var="$ROCM_LIBRARIES_DIR",
            )
        
        return None
    
    def _search_ck_source(self, parsed: ParsedKernelName) -> Optional[SourceMatch]:
        """Search for Composable Kernel source."""
        rocm_libs = self._repo_var_map.get("$ROCM_LIBRARIES_DIR")
        if not rocm_libs:
            return None
        
        ck_path = Path(rocm_libs) / "projects/composablekernel"
        if not ck_path.exists():
            return None
        
        # Map operation name to directory and kernel file
        op_name = parsed.function_name.lower()
        op_info = {
            "rmsnorm2dfwd": ("rmsnorm2d", "kernel/rmsnorm2d_fwd_kernel.hpp"),
            "rmsnorm": ("rmsnorm2d", "kernel/rmsnorm2d_fwd_kernel.hpp"),
            "fmha": ("fmha", "kernel/fmha_fwd_kernel.hpp"),
            "softmax": ("softmax", "kernel/softmax_kernel.hpp"),
            "gemm": ("gemm", "kernel/gemm_kernel.hpp"),
            "layernorm": ("layernorm2d", "kernel/layernorm2d_fwd_kernel.hpp"),
            "moe": ("moe_sorting_topk", "kernel/moe_sorting_kernel.hpp"),
        }
        
        for op_key, (op_dir, kernel_file) in op_info.items():
            if op_key in op_name:
                # Try specific kernel file first
                kernel_path = f"projects/composablekernel/include/ck_tile/ops/{op_dir}/{kernel_file}"
                if (Path(rocm_libs) / kernel_path).exists():
                    return SourceMatch(
                        file_path=kernel_path,
                        symbol=f"ck_tile::{op_dir}_kernel",
                        repo_name="rocm-libraries",
                        repo_var="$ROCM_LIBRARIES_DIR",
                    )
                # Fall back to directory
                op_path = f"projects/composablekernel/include/ck_tile/ops/{op_dir}/"
                if (Path(rocm_libs) / op_path).exists():
                    return SourceMatch(
                        file_path=op_path,
                        symbol=parsed.function_name,
                        repo_name="rocm-libraries",
                        repo_var="$ROCM_LIBRARIES_DIR",
                    )
        
        # Generic CK search
        return SourceMatch(
            file_path="projects/composablekernel/include/ck_tile/ops/",
            symbol=parsed.function_name,
            repo_name="rocm-libraries",
            repo_var="$ROCM_LIBRARIES_DIR",
        )
    
    def _search_aten_source(self, parsed: ParsedKernelName) -> Optional[SourceMatch]:
        """Search for ATen native kernel source."""
        # ATen kernels are in PyTorch, provide standard path
        kernel_type = parsed.extra.get('kernel_type', '')
        functor = parsed.extra.get('functor', '')
        
        # Map to known files
        file_mapping = {
            'FillFunctor': 'Fill.cu',
            'CompareEqFunctor': 'CompareEQKernel.cu',
            'div_trunc': 'BinaryDivTruncKernel.cu',
            'copy': 'Copy.cu',
            'argmax': 'ReduceArgMaxKernel.cu',
        }
        
        for key, filename in file_mapping.items():
            if key in parsed.original_name:
                return SourceMatch(
                    file_path=f"aten/src/ATen/native/cuda/{filename}",
                    symbol=parsed.function_name,
                    repo_name="pytorch",
                    repo_var="$PYTORCH_DIR",
                )
        
        return SourceMatch(
            file_path="aten/src/ATen/native/cuda/",
            symbol=parsed.function_name,
            repo_name="pytorch",
            repo_var="$PYTORCH_DIR",
        )
    
    def _search_hip_source(self, parsed: ParsedKernelName) -> Optional[SourceMatch]:
        """Search for HIP/CUDA C++ kernel source."""
        namespace = parsed.namespace
        function_name = parsed.function_name
        original_name = parsed.original_name
        
        # Known HIP kernel mappings
        known_hip_mappings = {
            # WMMA / Matrix core kernels (hipBLASLt / rocWMMA)
            "wvSplitK": ("projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/", "hipBLASLt WMMA kernel", "$ROCM_LIBRARIES_DIR"),
            "wvSpltK": ("projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/", "hipBLASLt WMMA kernel", "$ROCM_LIBRARIES_DIR"),
            "DeviceGemmWmma": ("projects/composablekernel/include/ck/tensor_operation/gpu/device/impl/", "CK WMMA GEMM", "$ROCM_LIBRARIES_DIR"),
            # vLLM kernels
            "reshape_and_cache": ("csrc/cache_kernels.cu", "reshape_and_cache_flash_kernel", "$VLLM_DIR"),
            "paged_attention": ("csrc/attention/paged_attention_v1.cu", "paged_attention_kernel", "$VLLM_DIR"),
            "rotary_embedding": ("csrc/pos_encoding_kernels.cu", "rotary_embedding_kernel", "$VLLM_DIR"),
            "rms_norm": ("csrc/layernorm_kernels.cu", "rms_norm_kernel", "$VLLM_DIR"),
            "silu_and_mul": ("csrc/activation_kernels.cu", "silu_and_mul_kernel", "$VLLM_DIR"),
            "gelu": ("csrc/activation_kernels.cu", "gelu_kernel", "$VLLM_DIR"),
            # rocBLAS / BLAS
            "rocblas": ("projects/rocblas/library/src/", "rocBLAS kernel", "$ROCM_LIBRARIES_DIR"),
        }
        
        # Check for ROCm runtime kernels (in rocm-systems super-repo)
        if "__amd_rocclr" in original_name or "rocclr_copy" in original_name:
            return SourceMatch(
                file_path="projects/clr/rocclr/device/blit.cpp",
                symbol="ROCm runtime blit kernel",
                repo_name="rocm-systems",
                repo_var="$ROCM_SYSTEMS_DIR",
            )
        
        # HIP memory copy operations (internal runtime)
        if original_name.startswith("MEMORY_COPY"):
            return SourceMatch(
                file_path="projects/clr/hipamd/src/hip_memory.cpp",
                symbol="HIP memory copy",
                repo_name="rocm-systems",
                repo_var="$ROCM_SYSTEMS_DIR",
            )
        
        # Check known mappings
        for key, (path, symbol, repo_var) in known_hip_mappings.items():
            if key in original_name or key in function_name:
                repo_name = "vllm" if repo_var == "$VLLM_DIR" else "rocm-libraries"
                return SourceMatch(
                    file_path=path,
                    symbol=symbol,
                    repo_name=repo_name,
                    repo_var=repo_var,
                )
        
        # Check vLLM kernels by namespace
        if namespace == "vllm" or "vllm" in original_name.lower():
            return SourceMatch(
                file_path="csrc/",
                symbol=function_name,
                repo_name="vllm",
                repo_var="$VLLM_DIR",
            )
        
        # Search in rocm-libraries
        rocm_libs = self._repo_var_map.get("$ROCM_LIBRARIES_DIR")
        if rocm_libs:
            pattern = f"void.*{function_name}"
            files = self._run_ripgrep(pattern, rocm_libs, ["cpp"])
            if files:
                rel_path = os.path.relpath(files[0], rocm_libs)
                return SourceMatch(
                    file_path=rel_path,
                    symbol=function_name,
                    repo_name="rocm-libraries",
                    repo_var="$ROCM_LIBRARIES_DIR",
                )
        
        return None
    
    def _search_inductor_source(self, parsed: ParsedKernelName) -> Optional[SourceMatch]:
        """Search for torch.inductor generated kernel."""
        return SourceMatch(
            file_path="torch/_inductor/codegen/triton.py",
            symbol=parsed.function_name,
            repo_name="pytorch",
            repo_var="$PYTORCH_DIR",
        )
    
    def _search_triton_test(self, parsed: ParsedKernelName, 
                           source: Optional[SourceMatch]) -> Optional[TestMatch]:
        """Search for Triton kernel tests."""
        function_name = parsed.function_name
        
        # If source is from aiter, use aiter test mappings
        if source and source.repo_name == "aiter":
            aiter_test_mappings = {
                "rmsnorm": ("op_tests/test_rmsnorm2d.py", "cd $AITER_DIR && pytest op_tests/test_rmsnorm2d.py -v"),
                "layernorm": ("op_tests/test_layernorm.py", "cd $AITER_DIR && pytest op_tests/test_layernorm.py -v"),
                "attention": ("op_tests/test_mha.py", "cd $AITER_DIR && pytest op_tests/test_mha.py -v"),
                "mha": ("op_tests/test_mha.py", "cd $AITER_DIR && pytest op_tests/test_mha.py -v"),
                "moe": ("op_tests/test_moe.py", "cd $AITER_DIR && pytest op_tests/test_moe.py -v"),
                "quant": ("op_tests/test_quant.py", "cd $AITER_DIR && pytest op_tests/test_quant.py -v"),
                "gemm": ("op_tests/test_gemm_a8w8.py", "cd $AITER_DIR && pytest op_tests/test_gemm_a8w8.py -v"),
                "rope": ("op_tests/test_rope.py", "cd $AITER_DIR && pytest op_tests/test_rope.py -v"),
            }
            
            fn_lower = function_name.lower()
            for key, (test_file, test_cmd) in aiter_test_mappings.items():
                if key in fn_lower:
                    return TestMatch(
                        test_file=test_file,
                        test_cmd=test_cmd,
                        repo_var="$AITER_DIR",
                    )
            
            # Default aiter test
            return TestMatch(
                test_file="op_tests/",
                test_cmd="cd $AITER_DIR && pytest op_tests/ -v",
                repo_var="$AITER_DIR",
            )
        
        # Known test mappings for common kernels
        # Note: $TRITON_KERNELS_DIR = triton/python/triton_kernels/
        known_test_mappings = {
            "_matmul_ogs": ("tests/test_matmul.py", "cd $TRITON_KERNELS_DIR && pytest tests/test_matmul.py -v", "$TRITON_KERNELS_DIR"),
            "_matmul": ("tests/test_matmul.py", "cd $TRITON_KERNELS_DIR && pytest tests/test_matmul.py -v", "$TRITON_KERNELS_DIR"),
            "_reduce": ("tests/test_reduce.py", "cd $TRITON_KERNELS_DIR && pytest tests/test_reduce.py -v", "$TRITON_KERNELS_DIR"),
            "kernel_unified_attention": ("tests/v1/attention/", "cd $VLLM_DIR && pytest tests/v1/ -v -k attention", "$VLLM_DIR"),
            "_topk_forward": ("tests/test_topk.py", "cd $TRITON_KERNELS_DIR && pytest tests/test_topk.py -v", "$TRITON_KERNELS_DIR"),
            "_topk_backward": ("tests/test_topk.py", "cd $TRITON_KERNELS_DIR && pytest tests/test_topk.py -v", "$TRITON_KERNELS_DIR"),
            "_bitmatrix": ("tests/test_tensor.py", "cd $TRITON_KERNELS_DIR && pytest tests/test_tensor.py -v", "$TRITON_KERNELS_DIR"),
            "_ragged_tensor": ("tests/test_tensor.py", "cd $TRITON_KERNELS_DIR && pytest tests/test_tensor.py -v", "$TRITON_KERNELS_DIR"),
            "_sum_bitmatrix_rows": ("tests/test_tensor.py", "cd $TRITON_KERNELS_DIR && pytest tests/test_tensor.py -v", "$TRITON_KERNELS_DIR"),
            "_fused_add_rmsnorm": ("tests/test_swiglu.py", "cd $TRITON_KERNELS_DIR && pytest tests/test_swiglu.py -v", "$TRITON_KERNELS_DIR"),
            "_swiglu": ("tests/test_swiglu.py", "cd $TRITON_KERNELS_DIR && pytest tests/test_swiglu.py -v", "$TRITON_KERNELS_DIR"),
            "_compaction": ("tests/test_compaction.py", "cd $TRITON_KERNELS_DIR && pytest tests/test_compaction.py -v", "$TRITON_KERNELS_DIR"),
        }
        
        # Check known mappings first
        for key, (test_file, test_cmd, repo_var) in known_test_mappings.items():
            if key in function_name:
                return TestMatch(
                    test_file=test_file,
                    test_cmd=test_cmd,
                    repo_var=repo_var,
                )
        
        triton_path = self._repo_var_map.get("$TRITON_DIR")
        if triton_path:
            test_path = Path(triton_path) / "python/test/unit/language"
            if test_path.exists():
                # Search for test files mentioning the function
                pattern = f"def test.*{function_name}|{function_name}"
                files = self._run_ripgrep(pattern, str(test_path), ["py"])
                if files:
                    rel_path = os.path.relpath(files[0], triton_path)
                    return TestMatch(
                        test_file=rel_path,
                        test_cmd=f"cd $TRITON_DIR && pytest {rel_path} -q",
                        repo_var="$TRITON_DIR",
                    )
        
        return None
    
    def _search_tensile_test(self, parsed: ParsedKernelName) -> Optional[TestMatch]:
        """Search for Tensile GEMM tests."""
        rocm_libs = self._repo_var_map.get("$ROCM_LIBRARIES_DIR")
        if rocm_libs:
            return TestMatch(
                test_file="projects/rocblas/clients/gtest/blas3/gemm_gtest.cpp",
                test_cmd="cd $ROCM_LIBRARIES_DIR/projects/rocblas/build/release && ./clients/staging/rocblas-bench -f gemm_ex --a_type bf16_r --b_type bf16_r --compute_type f32_r",
                repo_var="$ROCM_LIBRARIES_DIR",
            )
        return None
    
    def _search_ck_test(self, parsed: ParsedKernelName) -> Optional[TestMatch]:
        """Search for CK tile tests."""
        op_name = parsed.function_name.lower()
        original_name = parsed.original_name.lower()
        
        # Map operation to example/test directory
        # CK examples are at: projects/composablekernel/example/ck_tile/
        if "rmsnorm" in op_name or "rmsnorm" in original_name:
            return TestMatch(
                test_file="projects/composablekernel/example/ck_tile/10_rmsnorm2d/",
                test_cmd="cd $ROCM_LIBRARIES_DIR/projects/composablekernel/build && cmake --build . -j --target tile_example_rmsnorm2d_fwd && ./bin/tile_example_rmsnorm2d_fwd -m 1024 -n 2048",
                repo_var="$ROCM_LIBRARIES_DIR",
            )
        elif "fmha" in op_name or "fmha" in original_name:
            return TestMatch(
                test_file="projects/composablekernel/example/ck_tile/01_fmha/",
                test_cmd="cd $ROCM_LIBRARIES_DIR/projects/composablekernel/build && cmake --build . -j --target tile_example_fmha_fwd && ./bin/tile_example_fmha_fwd",
                repo_var="$ROCM_LIBRARIES_DIR",
            )
        elif "layernorm" in op_name or "layernorm" in original_name:
            return TestMatch(
                test_file="projects/composablekernel/example/ck_tile/02_layernorm2d/",
                test_cmd="cd $ROCM_LIBRARIES_DIR/projects/composablekernel/build && cmake --build . -j --target tile_example_layernorm2d_fwd && ./bin/tile_example_layernorm2d_fwd",
                repo_var="$ROCM_LIBRARIES_DIR",
            )
        elif "gemm" in op_name or "gemm" in original_name:
            return TestMatch(
                test_file="projects/composablekernel/example/ck_tile/03_gemm/",
                test_cmd="cd $ROCM_LIBRARIES_DIR/projects/composablekernel/build && cmake --build . -j --target tile_example_gemm && ./bin/tile_example_gemm",
                repo_var="$ROCM_LIBRARIES_DIR",
            )
        elif "topk" in op_name or "softmax" in op_name:
            return TestMatch(
                test_file="projects/composablekernel/example/ck_tile/09_topk_softmax/",
                test_cmd="cd $ROCM_LIBRARIES_DIR/projects/composablekernel/build && cmake --build . -j --target tile_example_topk_softmax && ./bin/tile_example_topk_softmax",
                repo_var="$ROCM_LIBRARIES_DIR",
            )
        # MoE sorting + MoE FlatMM (top-bottleneck on MI355X gpt-oss/MoE traces).
        # Added in refrence_torch follow-up; previously the CK searcher returned
        # None for any MoeSorting* / MoeFlatmm* kernel and so the gap_analysis
        # CSV had no test entry for ~37%+ of GPU time on MoE workloads.
        elif "moesorting" in op_name or "moe_sorting" in op_name \
             or "moesorting" in original_name or "moe_sorting" in original_name:
            return TestMatch(
                test_file="projects/composablekernel/example/ck_tile/13_moe_sorting/",
                test_cmd="cd $ROCM_LIBRARIES_DIR/projects/composablekernel/build && cmake --build . -j --target tile_example_moe_sorting && ./bin/tile_example_moe_sorting",
                repo_var="$ROCM_LIBRARIES_DIR",
            )
        elif "moeflatmm" in op_name or "moe_flatmm" in op_name \
             or "moeflatmm" in original_name or "moe_flatmm" in original_name \
             or "flatmm" in original_name:
            return TestMatch(
                test_file="projects/composablekernel/example/ck_tile/18_flatmm/",
                test_cmd="cd $ROCM_LIBRARIES_DIR/projects/composablekernel/build && cmake --build . -j --target tile_example_flatmm && ./bin/tile_example_flatmm",
                repo_var="$ROCM_LIBRARIES_DIR",
            )
        elif "fused_moe" in op_name or "fused_moe" in original_name:
            return TestMatch(
                test_file="projects/composablekernel/example/ck_tile/15_fused_moe/",
                test_cmd="cd $ROCM_LIBRARIES_DIR/projects/composablekernel/build && cmake --build . -j --target tile_example_fused_moe && ./bin/tile_example_fused_moe",
                repo_var="$ROCM_LIBRARIES_DIR",
            )

        return None
    
    def _search_aten_test(self, parsed: ParsedKernelName) -> Optional[TestMatch]:
        """Search for ATen native tests."""
        # Map operation to test file
        test_mapping = {
            'fill': 'test_torch.py',
            'eq': 'test_binary_ufuncs.py',
            'div': 'test_binary_ufuncs.py',
            'copy': 'test_torch.py',
            'argmax': 'test_reductions.py',
        }
        
        for op, test_file in test_mapping.items():
            if op in parsed.function_name.lower():
                return TestMatch(
                    test_file=f"test/{test_file}",
                    test_cmd=f"pytest $PYTORCH_DIR/test/{test_file} -q -k {op}",
                    repo_var="$PYTORCH_DIR",
                )
        
        return TestMatch(
            test_file="test/test_torch.py",
            test_cmd="pytest $PYTORCH_DIR/test/test_torch.py -q",
            repo_var="$PYTORCH_DIR",
        )
    
    def _search_hip_test(self, parsed: ParsedKernelName,
                        source: Optional[SourceMatch]) -> Optional[TestMatch]:
        """Search for HIP/CUDA kernel tests."""
        namespace = parsed.namespace
        function_name = parsed.function_name
        original_name = parsed.original_name
        
        # Known HIP kernel test mappings
        known_hip_test_mappings = {
            # WMMA / Matrix core kernels (hipBLASLt)
            "wvSplitK": ("projects/hipblaslt/clients/tests/", "cd $ROCM_LIBRARIES_DIR/projects/hipblaslt/build/release && ./clients/staging/hipblaslt-bench -f gemm_ex --a_type bf16_r --b_type bf16_r --compute_type f32_r -m 1024 -n 1024 -k 1024", "$ROCM_LIBRARIES_DIR"),
            "wvSpltK": ("projects/hipblaslt/clients/tests/", "cd $ROCM_LIBRARIES_DIR/projects/hipblaslt/build/release && ./clients/staging/hipblaslt-bench -f gemm_ex --a_type bf16_r --b_type bf16_r --compute_type f32_r -m 1024 -n 1024 -k 1024", "$ROCM_LIBRARIES_DIR"),
            # vLLM kernels
            "reshape_and_cache": ("tests/kernels/test_cache.py", "cd $VLLM_DIR && pytest tests/kernels/test_cache.py -v", "$VLLM_DIR"),
            "paged_attention": ("tests/kernels/test_attention.py", "cd $VLLM_DIR && pytest tests/kernels/test_attention.py -v", "$VLLM_DIR"),
            "rotary_embedding": ("tests/kernels/test_pos_encoding.py", "cd $VLLM_DIR && pytest tests/kernels/test_pos_encoding.py -v", "$VLLM_DIR"),
            "rms_norm": ("tests/kernels/test_layernorm.py", "cd $VLLM_DIR && pytest tests/kernels/test_layernorm.py -v", "$VLLM_DIR"),
            "silu_and_mul": ("tests/kernels/test_activation.py", "cd $VLLM_DIR && pytest tests/kernels/test_activation.py -v", "$VLLM_DIR"),
        }
        
        # Check for ROCm runtime kernels (in rocm-systems)
        if "__amd_rocclr" in original_name or "rocclr_copy" in original_name or original_name.startswith("MEMORY_COPY"):
            return TestMatch(
                test_file="projects/hip-tests/catch/unit/memory/",
                test_cmd="cd $ROCM_SYSTEMS_DIR && ctest -R hipMemcpy",
                repo_var="$ROCM_SYSTEMS_DIR",
            )
        
        # Check known mappings
        for key, (test_file, test_cmd, repo_var) in known_hip_test_mappings.items():
            if key in original_name or key in function_name:
                return TestMatch(
                    test_file=test_file,
                    test_cmd=test_cmd,
                    repo_var=repo_var,
                )
        
        if namespace == "vllm" or "vllm" in original_name.lower():
            # Map to vLLM test directories
            if "cache" in function_name.lower():
                return TestMatch(
                    test_file="tests/kernels/attention/test_cache.py",
                    test_cmd="cd $VLLM_DIR && pytest tests/kernels/attention/test_cache.py -q",
                    repo_var="$VLLM_DIR",
                )
            return TestMatch(
                test_file="tests/kernels/",
                test_cmd="cd $VLLM_DIR && pytest tests/kernels/ -q",
                repo_var="$VLLM_DIR",
            )
        
        return None
    
    def _search_aiter_source(self, parsed: ParsedKernelName) -> Optional[SourceMatch]:
        """Search for aiter kernel source."""
        function_name = parsed.function_name
        original_name = parsed.original_name
        extra = parsed.extra or {}
        category = extra.get('category', '')
        
        # Known aiter kernel mappings
        known_mappings = {
            # Quantization kernels
            'dynamic_per_group_scaled_quant': ('aiter/ops/quant.py', 'dynamic_per_group_scaled_quant'),
            'dynamic_per_token_scaled_quant': ('aiter/ops/quant.py', 'dynamic_per_token_scaled_quant'),
            'group_fp8_quant': ('aiter/ops/quant.py', 'group_fp8_quant'),
            # MoE kernels
            'fmoe': ('aiter/fused_moe.py', 'fused_moe'),
            'moe_sorting': ('aiter/ops/moe_sorting.py', 'moe_sorting'),
            'moe_align': ('csrc/kernels/moe_align_block_size_kernels.cu', 'moe_align'),
            # GEMM kernels
            'gemm_a8w8': ('aiter/ops/gemm_op_a8w8.py', 'gemm_a8w8'),
            'gemm_a4w4': ('aiter/ops/gemm_op_a4w4.py', 'gemm_a4w4'),
            'batched_gemm': ('aiter/ops/batched_gemm_op_a8w8.py', 'batched_gemm'),
            # Attention kernels
            'mha': ('aiter/ops/mha.py', 'mha'),
            'mla': ('aiter/mla.py', 'mla'),
            'paged_attention': ('aiter/paged_attn.py', 'paged_attn'),
            # Norm kernels
            'rmsnorm': ('aiter/ops/rmsnorm.py', 'rmsnorm'),
            'groupnorm': ('aiter/ops/groupnorm.py', 'groupnorm'),
            # Rope kernels
            'rotary': ('aiter/rotary_embedding.py', 'rotary_embedding'),
            'rope': ('aiter/ops/rope.py', 'rope'),
        }
        
        # Check known mappings
        for key, (path, symbol) in known_mappings.items():
            if key in function_name.lower() or key in original_name.lower():
                return SourceMatch(
                    file_path=path,
                    symbol=symbol,
                    repo_name="aiter",
                    repo_var="$AITER_DIR",
                )
        
        # Fall back based on category
        category_paths = {
            'quant': 'aiter/ops/quant.py',
            'moe': 'aiter/fused_moe.py',
            'gemm': 'aiter/ops/gemm_op_a8w8.py',
            'attention': 'aiter/ops/mha.py',
            'norm': 'aiter/ops/rmsnorm.py',
        }
        
        if category in category_paths:
            return SourceMatch(
                file_path=category_paths[category],
                symbol=function_name,
                repo_name="aiter",
                repo_var="$AITER_DIR",
            )
        
        # Default: search in aiter/ops
        return SourceMatch(
            file_path="aiter/ops/",
            symbol=function_name,
            repo_name="aiter",
            repo_var="$AITER_DIR",
        )
    
    def _search_aiter_test(self, parsed: ParsedKernelName,
                           source: Optional[SourceMatch]) -> Optional[TestMatch]:
        """Search for aiter kernel tests."""
        function_name = parsed.function_name
        original_name = parsed.original_name
        extra = parsed.extra or {}
        category = extra.get('category', '')
        
        # Known test mappings
        test_mappings = {
            'quant': ('op_tests/test_quant.py', 'quant'),
            'moe': ('op_tests/test_moe.py', 'moe'),
            'gemm': ('op_tests/test_gemm_a8w8.py', 'gemm'),
            'attention': ('op_tests/test_mha.py', 'mha'),
            'norm': ('op_tests/test_rmsnorm2d.py', 'rmsnorm'),
            'rope': ('op_tests/test_rope.py', 'rope'),
        }
        
        # Check by category
        if category in test_mappings:
            test_file, keyword = test_mappings[category]
            return TestMatch(
                test_file=test_file,
                test_cmd=f"cd $AITER_DIR && pytest {test_file} -v -k {keyword}",
                repo_var="$AITER_DIR",
            )
        
        # Check by keywords in function name
        if 'quant' in function_name.lower():
            return TestMatch(
                test_file="op_tests/test_quant.py",
                test_cmd="cd $AITER_DIR && pytest op_tests/test_quant.py -v",
                repo_var="$AITER_DIR",
            )
        elif 'moe' in function_name.lower() or 'fmoe' in function_name.lower():
            return TestMatch(
                test_file="op_tests/test_moe.py",
                test_cmd="cd $AITER_DIR && pytest op_tests/test_moe.py -v",
                repo_var="$AITER_DIR",
            )
        elif 'gemm' in function_name.lower():
            return TestMatch(
                test_file="op_tests/test_gemm_a8w8.py",
                test_cmd="cd $AITER_DIR && pytest op_tests/test_gemm_a8w8.py -v",
                repo_var="$AITER_DIR",
            )
        elif 'mha' in function_name.lower() or 'attention' in function_name.lower():
            return TestMatch(
                test_file="op_tests/test_mha.py",
                test_cmd="cd $AITER_DIR && pytest op_tests/test_mha.py -v",
                repo_var="$AITER_DIR",
            )
        
        # Default test
        return TestMatch(
            test_file="op_tests/",
            test_cmd="cd $AITER_DIR && pytest op_tests/ -v",
            repo_var="$AITER_DIR",
        )
