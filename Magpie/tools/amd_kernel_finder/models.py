###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Data models for kernel source finder.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # Forward references will work with __future__ annotations


class KernelKind(Enum):
    """Classification of kernel types."""
    TRITON_JIT = "triton_jit"
    TENSILE_GEMM = "tensile_gemm"
    CK_TILE = "ck_tile"
    ATEN_NATIVE = "aten_native"
    HIP_CPP = "hip_cpp"
    INDUCTOR = "inductor"
    AITER = "aiter"
    ANNOTATION = "annotation"
    UNKNOWN = "unknown"


class KernelCategory(Enum):
    """Classification of kernel operation categories."""
    ATTENTION = "attention"
    GEMM = "gemm"
    MOE_GEMM = "moe_gemm"
    LAYERNORM = "layernorm"
    SOFTMAX = "softmax"
    COPY = "copy"
    ELEMENTWISE = "elementwise"
    INDEXING = "indexing"
    REDUCE = "reduce"
    ROUTER = "router"
    KV_CACHE = "kv_cache"
    BLIT = "blit"
    ANNOTATION = "annotation"
    UNKNOWN = "unknown"


@dataclass
class KernelSourceInfo:
    """Kernel source and test information."""
    
    # Classification
    kind: str = ""
    category: str = ""
    
    # Source location
    source_repo: str = ""
    source_file: str = ""
    upstream_url: str = ""
    
    # Test information
    test_file: str = ""
    test_cmd: str = ""

    # PyTorch eager baseline used for correctness & perf comparison when
    # optimizing this kernel. Points at an existing reference impl inside the
    # libraries (aiter op_tests, vllm tests, pytorch ATen, etc.). When
    # possible we point at the importable eager symbol (e.g.
    # `aiter.fused_moe.torch_moe`) rather than a thin @perftest wrapper.
    baseline_ref_file: str = ""
    baseline_ref_symbol: str = ""
    # One of: "eager_fn" (importable pure-torch fn), "perftest_wrapper" (thin
    # @perftest-decorated wrapper around a pure-torch fn; importable; gives
    # A/B timing for free), "inline_in_test" (reference is built inside a
    # parametrized pytest body, NOT importable -- read the test body), or
    # "none" (no baseline found).
    baseline_ref_kind: str = ""

    # Canonical Triton implementation reference (separate from the eager
    # baseline above). Lets a developer compare an optimized kernel against a
    # known-good Triton impl that lives inside aiter/vllm.
    triton_ref_file: str = ""
    triton_ref_symbol: str = ""

    # Additional context
    notes: str = ""
    
    def to_list(self) -> List[str]:
        """Convert to list of values for CSV output."""
        return [
            self.kind,
            self.category,
            self.source_repo,
            self.source_file,
            self.upstream_url,
            self.test_file,
            self.test_cmd,
            self.baseline_ref_file,
            self.baseline_ref_symbol,
            self.baseline_ref_kind,
            self.triton_ref_file,
            self.triton_ref_symbol,
            self.notes,
        ]
    
    @staticmethod
    def csv_headers() -> List[str]:
        """Return CSV header names for source info columns."""
        return [
            "kind",
            "category", 
            "source_repo",
            "source_file",
            "upstream_url",
            "test_file",
            "test_cmd",
            "baseline_ref_file",
            "baseline_ref_symbol",
            "baseline_ref_kind",
            "triton_ref_file",
            "triton_ref_symbol",
            "notes",
        ]


@dataclass
class ParsedKernelName:
    """Parsed information from a kernel name."""
    
    original_name: str
    kind: KernelKind
    function_name: str = ""
    config: str = ""
    namespace: str = ""
    dtype: str = ""
    tile_sizes: Optional[dict] = None
    extra: dict = field(default_factory=dict)


@dataclass
class SourceMatch:
    """A matched source file location."""
    
    file_path: str
    symbol: str = ""
    line_number: Optional[int] = None
    repo_name: str = ""
    repo_var: str = ""  # e.g., $TRITON_DIR
    
    @property
    def display_path(self) -> str:
        """Return path with repo variable prefix."""
        if self.repo_var:
            return f"{self.repo_var}/{self.file_path}"
        return self.file_path


@dataclass 
class TestMatch:
    """A matched test file and command."""
    
    test_file: str
    test_cmd: str
    repo_var: str = ""
    
    @property
    def display_path(self) -> str:
        """Return test path with repo variable prefix."""
        if self.repo_var:
            return f"{self.repo_var}/{self.test_file}"
        return self.test_file


@dataclass
class BaselineRefMatch:
    """A matched baseline PyTorch eager reference implementation.

    Used by gap analysis so that a developer optimizing a given kernel knows
    exactly which eager function to import and run side-by-side for
    correctness checks (torch.allclose) and apples-to-apples timing.
    """

    ref_file: str
    ref_symbol: str
    repo_var: str = ""
    # One of: "eager_fn", "perftest_wrapper", "inline_in_test", "none".
    # See KernelSourceInfo.baseline_ref_kind for definitions.
    kind: str = "eager_fn"
    notes: str = ""

    @property
    def display_path(self) -> str:
        """Return reference file path with repo variable prefix."""
        if self.repo_var:
            return f"{self.repo_var}/{self.ref_file}"
        return self.ref_file


@dataclass
class TritonRefMatch:
    """A matched canonical Triton kernel implementation.

    Lets a developer compare an optimized kernel against a known-good Triton
    implementation that lives inside aiter / vllm / triton_kernels.
    """

    ref_file: str
    ref_symbol: str
    repo_var: str = ""
    notes: str = ""

    @property
    def display_path(self) -> str:
        if self.repo_var:
            return f"{self.repo_var}/{self.ref_file}"
        return self.ref_file
