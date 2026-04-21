###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Benchmark mode for framework-level profiling.

This module provides:
- BenchmarkMode: Main class for running vLLM/SGLang benchmarks
- BenchmarkConfig: Configuration for benchmark runs
- BenchmarkResult: Results from benchmark execution
"""

from .config import (
    BenchmarkConfig,
    DEFAULT_SHARED_STORAGE_PATH,
    ProfilerConfig,
    TorchProfilerConfig,
    SystemProfilerConfig,
    GapAnalysisConfig,
)
from .benchmarker import BenchmarkMode
from .result import BenchmarkResult
from .workspace import WorkspaceManager
from .image_selector import ImageSelector
from .inferencex import InferenceXManager, ensure_inferencex_available
from .gap_analysis import GapAnalyzer, GapAnalysisResult

__all__ = [
    "BenchmarkMode",
    "BenchmarkConfig",
    "DEFAULT_SHARED_STORAGE_PATH",
    "BenchmarkResult",
    "ProfilerConfig",
    "TorchProfilerConfig",
    "SystemProfilerConfig",
    "GapAnalysisConfig",
    "GapAnalyzer",
    "GapAnalysisResult",
    "WorkspaceManager",
    "ImageSelector",
    "InferenceXManager",
    "ensure_inferencex_available",
]


