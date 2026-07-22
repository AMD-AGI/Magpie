###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Magpie: A general-purpose evaluation framework for GPU kernel implementations.

This framework provides tools for measuring correctness, robustness, efficiency,
and performance of GPU kernels, independent of how the kernel code is produced.

Modules:
    - config: Configuration classes for the evaluation pipeline
    - core: Core evaluation engine components
    - eval: Evaluation modules (compiling, correctness, performance)
    - modes: Different evaluation modes (analyze, compare)
    - utils: Utility functions
"""

__version__ = "0.2.0"
__author__ = "Magpie Team"

# Re-export commonly used classes for convenience
from .config import (
    KernelType,
    EvalMode,
    PipelineConfig,
    KernelEvalConfig,
    CorrectnessMode,
    CorrectnessConfig,
    PerfBackend,
    PerformanceConfig,
)

from .eval import (
    Evaluator,
    EvaluationState,
    Compiling,
    CompilingResult,
    Correctness,
    CorrectnessResult,
    Performance,
    PerformanceResult,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Config
    "KernelType",
    "EvalMode",
    "PipelineConfig",
    "KernelEvalConfig",
    "CorrectnessMode",
    "CorrectnessConfig",
    "PerfBackend",
    "PerformanceConfig",
    # Eval
    "Evaluator",
    "EvaluationState",
    "Compiling",
    "CompilingResult",
    "Correctness",
    "CorrectnessResult",
    "Performance",
    "PerformanceResult",
]
