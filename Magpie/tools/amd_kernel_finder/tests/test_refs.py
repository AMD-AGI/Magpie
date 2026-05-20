###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Regression tests for the Triton-reference ranker in
``tools.amd_kernel_finder.searcher.KernelSourceSearcher.search_triton_ref``
and the syntactic invariants that every reference Magpie emits in a
``gap_analysis.csv`` must satisfy.

Three test classes:

  * ``TestTokenizer`` -- unit tests for the tokenizer + dtype extractor.
    Pure functions, no filesystem access. Fast.

  * ``TestRankerGoldenCorpus`` -- end-to-end ranker test against a small,
    hand-curated corpus of (kernel_name, category) -> expected outcome
    tuples covering every failure mode we have caught in production
    (see also ``REFERENCE_VALIDATION.md`` for the GPT-OSS-20B audit
    that motivated each entry). Requires the magpie repo cache to
    contain ``aiter`` + ``vllm`` clones. Skipped (not failed) if the
    cache is missing.

  * ``TestEmittedRefsResolve`` -- if a CSV path is provided via the
    ``MAGPIE_GAP_CSV`` env var, walk every row and assert that the
    ``baseline_ref_file`` / ``triton_ref_file`` columns each resolve to
    a real ``def NAME(`` in a real file (and that every triton symbol
    carries an ``@triton.jit`` decorator within 5 lines above). This
    is the syntactic invariant the validation script in
    ``REFERENCE_VALIDATION.md`` checked manually.

Run with:

    pytest -q tools/amd_kernel_finder/tests/test_refs.py
"""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Optional

import pytest

from tools.amd_kernel_finder.models import KernelCategory
from tools.amd_kernel_finder.parser import KernelNameParser
from tools.amd_kernel_finder.searcher import (
    KernelSourceSearcher,
    _tokenize_symbol,
    _extract_compute_dtypes,
    _extract_quant_dtypes,
)


REPO_BASE = Path(os.environ.get(
    "MAGPIE_REPO_CACHE",
    str(Path.home() / ".cache" / "magpie" / "repos"),
))


def _repo_cache_available() -> bool:
    """True iff the aiter clone exists in the magpie repo cache."""
    return (REPO_BASE / "aiter").is_dir()


needs_repo_cache = pytest.mark.skipif(
    not _repo_cache_available(),
    reason=f"magpie repo cache missing under {REPO_BASE}; "
           f"run `magpie sync-repos` first or set MAGPIE_REPO_CACHE",
)


# ----------------------------------------------------------------------------
# 1. Pure-function unit tests (no filesystem)
# ----------------------------------------------------------------------------

class TestTokenizer:
    """Unit tests for ``_tokenize_symbol`` and dtype extraction."""

    def test_underscore_split(self):
        assert _tokenize_symbol("_fused_add_rmsnorm_pad") == [
            "fused", "add", "rmsnorm", "rms", "norm", "pad",
        ]

    def test_camelcase_split(self):
        # MoeSortingKernel -> [moe, sorting, kernel]
        assert _tokenize_symbol("MoeSortingKernel") == [
            "moe", "sorting", "kernel",
        ]
        # wvSplitK -> [wv, split, k]
        assert "split" in _tokenize_symbol("wvSplitK")

    def test_stem_aliases(self):
        # rmsnorm should ALSO emit rms + norm
        t = _tokenize_symbol("rmsnorm")
        assert "rmsnorm" in t and "rms" in t and "norm" in t
        # bfloat16 should also emit bf16
        t = _tokenize_symbol("__hip_bfloat16")
        assert "bfloat16" in t and "bf16" in t

    def test_compute_dtype_extraction(self):
        # CK encoding
        assert "bf16" in _extract_compute_dtypes(
            "_ZN7ck_tile6kentryILi1ENS_12Rmsnorm2dFwd"
            "ProblemEDF16bDF16bE")
        # hip bfloat16 template arg
        assert "bf16" in _extract_compute_dtypes(
            "void vllm::reshape_and_cache_flash_kernel<__hip_bfloat16, 16>")
        # fp4 family from afp4wfp4
        assert "fp4" in _extract_compute_dtypes(
            "_fused_gemm_afp4wfp4_split_cat")

    def test_quant_dtype_extraction(self):
        # underscores must NOT block fp8 detection
        # (the bug that earlier let \bfp8\b miss `_fp8_`)
        assert "fp8" in _extract_quant_dtypes(
            "_fused_rms_fp8_per_tensor_static_quant_kernel")
        # int8 + blockscale combo
        q = _extract_quant_dtypes("_fused_gemm_a8w8_blockscale_split_cat")
        assert "int8" in q and "blockscale" in q
        # int4 from a16w4
        assert "int4" in _extract_quant_dtypes("_moe_gemm_a16w4")
        # plain bf16 GEMM has NO quant tag
        assert _extract_quant_dtypes("_gemm_a16_w16_kernel") == frozenset()


# ----------------------------------------------------------------------------
# 2. End-to-end golden corpus
# ----------------------------------------------------------------------------

# Each entry: (kernel_name, category, expected_outcome)
# where expected_outcome is one of:
#   ("symbol", "expected_symbol")             -- exact symbol match
#   ("file",   "expected_filename_substring") -- any symbol in this file is OK
#   ("none",   None)                          -- ranker MUST return None
#
# Every entry below is grounded in either:
#   * a real failure mode we caught in REFERENCE_VALIDATION.md, or
#   * a known-good case that the original ranker had right and we MUST
#     preserve.
GOLDEN_CORPUS = [
    # ----- Original-correct cases that MUST stay correct -----
    ("kernel_unified_attention_2d.kd",
     KernelCategory.ATTENTION,
     ("symbol", "kernel_unified_attention_2d")),

    ("kernel_unified_attention_3d.kd",
     KernelCategory.ATTENTION,
     ("symbol", "kernel_unified_attention_3d")),

    ("reduce_segments.kd",
     KernelCategory.ATTENTION,
     ("symbol", "reduce_segments")),

    ("_kernel_paged_attention_2d.kd",
     KernelCategory.ATTENTION,
     ("symbol", "_kernel_paged_attention_2d")),

    ("_gemm_a16_w16_kernel_BLOCK_SIZE_M_64_BLOCK_SIZE_N_128.kd",
     KernelCategory.GEMM,
     ("symbol", "_gemm_a16_w16_kernel")),

    ("_fused_add_rmsnorm_pad.kd",
     KernelCategory.LAYERNORM,
     ("symbol", "_fused_add_rmsnorm_pad")),

    # ----- The three categorical-error cases the new ranker fixes -----

    # P0.1 helper penalty: the original CSV had ALL MoE FlatMM rows
    # pointing at `unshuffle_weights` (a 3-arg helper that lives in the
    # same file as the actual GEMM). With the helper penalty + dtype
    # awareness, the production F8xMXF4 / F16xMXF4 variants resolve to
    # either `_fused_moe_kernel_mxfp4` (the swiglu/non-fused-silu form)
    # or `_fused_moe_kernel_mxfp4_silu` (the moe-silu form), both in
    # the moe/ dir. We assert file-level correctness; exact symbol
    # depends on whether the variant fuses silu.
    (
        "_ZN7ck_tile6kentryILi2ENS_15MoeFlatmmKernelINS_33GemmSpatially"
        "LocalTilePartitionerINS_13TileGemmShapeINS_8sequenceIJLi32ELi"
        "256ELi256EEEENS4_IJLi1ELi4ELi1EEEENS4_IJLi16ELi16ELi128EEEELb"
        "0ELb0EEELi1ELi1EEENS_37F8xMXF4FlatmmPipelineAGmemBGmemCRegV1"
        "INS_28F8xMXF4FlatmmPipelineProblemIDB8_NS_16pk_float4_e2m1_t"
        "EfS8_NS_23TileGemmUniversalTraitsILb0ELb0ELb0ELb0ENS_13tensor"
        "_layout4gemm8RowMajorENSG_11ColumnMajorESH_Lb0ELb0ELb0ELi1ELb"
        "1ELi16EEELNS_21GemmPipelineSchedulerE0ELb1ELNS_10TailNumberE1"
        "ELNS_25amd_buffer_coherence_enumE0ELb0ESC_EENS_33F8xMXF4Flat"
        "mmPipelineAgBgCrPolicyEEENS_16CShuffleEpilogueINS_23CShuffle"
        "EpilogueProblemISC_SC_NS_5tupleIJEEEfDF16bST_SH_NS_12element"
        "_wise11PassThroughELi32ELi256ELi1ELi4ELi16ELi16ELi128ELb0ELi"
        "1ELb0ELi1ELi2ELb0EEEvEELNS_13MoeFlatmmKindE2ENS_3moe7MoeSilu"
        "EEEJNS11_19MoeFlatmmKernelArgsINS_18FlatmmScalePointerILi1EL"
        "i32ENS_11e8m0_bexp_tEEES15_NS13_ILi1ELi0EfEEEEEEEvDpT1_",
        KernelCategory.MOE_GEMM,
        ("file", "moe_op_mxfp4_silu_fused.py"),
    ),

    # P0.1 dtype mismatch: previously matched
    # `_fused_gemm_a8w8_blockscale_split_cat` purely on the literal
    # substring "split". A BF16 wvSplitK kernel must not be matched to
    # an a8w8/blockscale or fp4 triton candidate.
    (
        "void wvSplitK_hf_sml_<__hip_bfloat16, 64, 4, 16, 8, 1, 4>(int)",
        KernelCategory.GEMM,
        ("none", None),
    ),

    # P0.2 ROUTER root-dir fallback: previously matched
    # `_kernel_paged_attention_2d` because the searcher walked the whole
    # `_triton_kernels/` root and `kernel` was a shared lexical token.
    # Now must resolve to one of the moe_align_block_size_* kernels.
    (
        "void ck_tile::kentry<2, ck_tile::MoeSortingMultiPhaseKernel_P23"
        "<ck_tile::MoeSorting>>",
        KernelCategory.ROUTER,
        ("file", "moe_align_block_size.py"),
    ),
    (
        "void ck_tile::kentry<2, ck_tile::MoeSortingKernel<"
        "ck_tile::MoeSortingProblemEx<int>>>(...)",
        KernelCategory.ROUTER,
        ("file", "moe_align_block_size.py"),
    ),

    # ----- Additional discriminative cases -----

    # bf16 input + quant intent + fp8 quant candidate exists -- the
    # quant-asymmetry rule must NOT reject the fp8 candidate just because
    # the kernel doesn't have an explicit fp8 dtype tag.
    (
        "aiter::add_rmsnorm_quant_kernel<__hip_bfloat16, 8>(...)",
        KernelCategory.LAYERNORM,
        # both `_quant_fused_add_rmsnorm_kernel` and
        # `_fused_rms_fp8_per_tensor_static_quant_kernel` are
        # acceptable -- accept any in the layernorm/quant cluster.
        ("file_any", ["rmsnorm.py", "fused_fp8_quant.py"]),
    ),

    # vllm v1 reshape_and_cache_flash MUST resolve via the
    # `vllm/v1/attention/ops/` path (P0.2 dir fix).
    (
        "void vllm::reshape_and_cache_flash_kernel<__hip_bfloat16, 16>(...)",
        KernelCategory.KV_CACHE,
        ("file", "triton_reshape_and_cache_flash.py"),
    ),

    # ATen reduce kernel: in profiler output this often ends up
    # categorised UNKNOWN with the parser; the ranker should return None
    # rather than guess.
    (
        "void at::native::reduce_kernel<512, 1, "
        "at::native::ReduceOp<...>>(...)",
        KernelCategory.REDUCE,
        # REDUCE dir scan may emit a low-confidence weak match; the
        # only invariant we enforce here is "do not raise". Marked as
        # "any" -- correctness of the score threshold for REDUCE is
        # a follow-up.
        ("any", None),
    ),
]


@needs_repo_cache
class TestRankerGoldenCorpus:
    """End-to-end ranker test against the hand-curated corpus above."""

    @pytest.fixture(scope="class")
    def searcher(self):
        repos = [
            str(REPO_BASE / "aiter"),
            str(REPO_BASE / "vllm"),
            str(REPO_BASE / "pytorch"),
        ]
        return KernelSourceSearcher(repos=repos, auto_install_ripgrep=False)

    @pytest.fixture(scope="class")
    def parser(self):
        return KernelNameParser()

    @pytest.mark.parametrize(
        "kernel_name,category,expected",
        GOLDEN_CORPUS,
        ids=[
            f"{c.value}::{n[:40]}"
            for n, c, _ in GOLDEN_CORPUS
        ],
    )
    def test_corpus(self, searcher, parser, kernel_name, category, expected):
        parsed = parser.parse(kernel_name)
        result = searcher.search_triton_ref(parsed, category=category)

        kind, want = expected

        if kind == "none":
            assert result is None, (
                f"expected no triton ref for {kernel_name!r}, got "
                f"{result.ref_symbol} @ {result.ref_file}"
            )
            return

        if kind == "any":
            return  # only contract: don't raise

        assert result is not None, (
            f"expected ranker to find a triton ref for {kernel_name!r}, "
            f"got None"
        )

        if kind == "symbol":
            assert result.ref_symbol == want, (
                f"wrong symbol for {kernel_name!r}: "
                f"got {result.ref_symbol}, expected {want}"
            )
        elif kind == "file":
            assert want in result.ref_file, (
                f"wrong file for {kernel_name!r}: "
                f"got {result.ref_file}, expected substring {want!r}"
            )
        elif kind == "file_any":
            assert any(s in result.ref_file for s in want), (
                f"file {result.ref_file} matched none of {want}"
            )
        else:
            pytest.fail(f"unknown expected-outcome kind {kind!r}")


# ----------------------------------------------------------------------------
# 3. CSV-resolution invariants
# ----------------------------------------------------------------------------

# Path env var override; if unset, the test class self-skips.
_CSV_ENV = "MAGPIE_GAP_CSV"


def _resolve_csv_path(p: str) -> Optional[Path]:
    if not p:
        return None
    pp = Path(p).expanduser()
    return pp if pp.exists() else None


def _expand_repo_var(path: str) -> Optional[Path]:
    if not path:
        return None
    if path.startswith("$"):
        var, _, rel = path.partition("/")
        # $AITER_DIR -> aiter ; $VLLM_DIR -> vllm ; ...
        sub = var.lstrip("$").lower().replace("_dir", "")
        cand = REPO_BASE / sub / rel
        return cand if cand.exists() else None
    p = Path(path)
    return p if p.exists() else None


@pytest.mark.skipif(
    _resolve_csv_path(os.environ.get(_CSV_ENV, "")) is None,
    reason=f"set {_CSV_ENV}=<path/to/gap_analysis.csv> to enable",
)
class TestEmittedRefsResolve:
    """
    Walk a real gap_analysis.csv and assert every emitted ref column
    resolves to a real symbol on disk.

    This is the syntactic-validation step that REFERENCE_VALIDATION.md
    did by hand; running it in CI ensures no CSV emission ever ships
    with dangling pointers.
    """

    @pytest.fixture(scope="class")
    def rows(self):
        path = _resolve_csv_path(os.environ[_CSV_ENV])
        with open(path) as f:
            # strip leading comment-only lines from the metadata header
            data = [ln for ln in f if not ln.lstrip().startswith("#") and ln.strip()]
        return list(csv.DictReader(data))

    def _assert_def_present(self, file_path: str, symbol: str):
        if not file_path or not symbol:
            return  # row carries no ref -- nothing to validate
        # Skip placeholder / sentinel values used by the emitter for
        # ATen / inductor / profiler-marker rows.
        if file_path.startswith("(") or symbol.startswith("("):
            return
        if "torch.*" in symbol:
            return
        resolved = _expand_repo_var(file_path)
        assert resolved is not None, (
            f"ref_file {file_path!r} does not resolve to an existing path "
            f"under {REPO_BASE}"
        )
        body = resolved.read_text(errors="ignore")
        assert re.search(rf"\bdef\s+{re.escape(symbol)}\s*\(", body), (
            f"def {symbol}(...) not found in {file_path}"
        )

    def test_baseline_refs_resolve(self, rows):
        for r in rows:
            self._assert_def_present(
                r.get("baseline_ref_file", "").strip(),
                r.get("baseline_ref_symbol", "").strip(),
            )

    def test_triton_refs_resolve(self, rows):
        for r in rows:
            self._assert_def_present(
                r.get("triton_ref_file", "").strip(),
                r.get("triton_ref_symbol", "").strip(),
            )

    def test_triton_refs_carry_jit_decorator(self, rows):
        """
        Every non-empty triton_ref_symbol must have an `@triton.jit`
        decorator within 5 lines above its `def NAME(`.
        """
        for r in rows:
            file_path = r.get("triton_ref_file", "").strip()
            symbol = r.get("triton_ref_symbol", "").strip()
            if not file_path or not symbol:
                continue
            if file_path.startswith("(") or symbol.startswith("("):
                continue
            resolved = _expand_repo_var(file_path)
            if resolved is None:
                continue  # the resolve test above will already fail
            body = resolved.read_text(errors="ignore")
            # @triton.jit ... (up to 5 intervening lines) ... def SYM(
            pat = (
                r"@triton\.jit[^\n]*\n"
                r"(?:[^\n]*\n){0,5}"
                rf"\s*def\s+{re.escape(symbol)}\s*\("
            )
            assert re.search(pat, body), (
                f"{symbol} in {file_path} is not within 5 lines of an "
                f"@triton.jit decorator"
            )
