###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
AMD Kernel Source Finder - Find source code and tests for AMD GPU kernels.

This module provides tools to identify kernel source locations and test cases
from profiler kernel names on AMD GPUs (ROCm/HIP), supporting:
- Triton JIT kernels (on ROCm)
- Tensile GEMM kernels (rocBLAS)
- Composable Kernel (CK) tiles
- hipBLASLt WMMA kernels
- ATen native kernels (HIP backend)
- HIP C++ kernels

Features:
- Dynamic kernel indexing for fast lookups
- Auto-cloning of missing repositories on-demand
- Python-native search fallback when ripgrep unavailable
- Repository structure auto-discovery
"""

from .finder import KernelSourceFinder
from .models import KernelSourceInfo, KernelKind
from .indexer import KernelIndex
from .repo_manager import RepoManager

__all__ = [
    "KernelSourceFinder",
    "KernelSourceInfo",
    "KernelKind",
    "KernelIndex",
    "RepoManager",
]
