###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Evaluation modules for Magpie.

This module contains the core evaluation components:
- Evaluator: Main evaluation pipeline orchestrator
- Compiling: Kernel compilation
- Correctness: Correctness verification
- Performance: Performance measurement
"""

from .evaluator import (
    Evaluator,
    EvaluationState,
    BaseKind,
)
from .compiling import Compiling, CompilingResult
from .correctness import (
    Correctness,
    CorrectnessResult,
    MetricResult,
)
from .performance import Performance, PerformanceResult
from .latency import Latency, LatencyResult

__all__ = [
    # Evaluator
    "Evaluator",
    "EvaluationState",
    "BaseKind",
    # Compiling
    "Compiling",
    "CompilingResult",
    # Correctness
    "Correctness",
    "CorrectnessResult",
    "MetricResult",
    # Performance
    "Performance",
    "PerformanceResult",
    # Latency
    "Latency",
    "LatencyResult",
]
