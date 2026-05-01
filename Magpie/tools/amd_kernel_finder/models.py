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
