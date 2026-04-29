###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Pipeline configuration for kernel evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from .correctness import CorrectnessConfig
from .latency import LatencyConfig
from .performance import PerformanceConfig


class KernelType(Enum):
    """Kernel language type."""

    PYTORCH = auto()  # PyTorch kernel
    HIP = auto()  # HIP/ROCm kernel
    CUDA = auto()  # CUDA kernel
    TRITON = auto()  # Triton kernel (.py, JIT-compiled)


class EvalMode(Enum):
    """Evaluation mode."""

    ANALYZE = auto()  # Analyze kernel (requires testcase)
    COMPARE = auto()  # Compare multiple kernels


@dataclass
class CompilingConfig:
    """
    Configuration for kernel compilation.

    Attributes:
        enable_default_compile: If True, attempt default compilation (hipcc/nvcc)
                               when no compile_command is provided.
                               If False (default), skip compilation for pre-compiled kernels.
    """

    enable_default_compile: bool = False


@dataclass
class PipelineConfig:
    """
    Main pipeline configuration (framework config).

    Attributes:
        mode: Evaluation mode (analyze or compare)
        kernel_type: Type of kernel (pytorch, hip, cuda)
        gpu_arch: Target GPU architecture (auto-detected if None)
        compiling_config: Configuration for compilation
        correctness_config: Configuration for correctness evaluation
        performance_config: Configuration for performance evaluation
        output_dir: Directory for output files
        verbose: Enable verbose logging
    """

    mode: EvalMode = EvalMode.ANALYZE
    kernel_type: KernelType = KernelType.HIP
    gpu_arch: Optional[str] = None  # Auto-detect if None
    compiling_config: Optional[CompilingConfig] = None
    correctness_config: Optional[CorrectnessConfig] = None
    performance_config: Optional[PerformanceConfig] = None
    latency_config: Optional[LatencyConfig] = None
    output_dir: str = "./results"
    verbose: bool = False

    def __post_init__(self):
        # Auto-detect GPU architecture first — needed by PerformanceConfig
        # to choose the right profiler for cross-platform kernels (Triton).
        if self.gpu_arch is None:
            self.gpu_arch = self._detect_gpu_arch()

        if self.compiling_config is None:
            self.compiling_config = CompilingConfig()
        if self.correctness_config is None:
            self.correctness_config = CorrectnessConfig()
        if self.performance_config is None:
            self.performance_config = PerformanceConfig(
                kernel_type=self.kernel_type,
                gpu_arch=self.gpu_arch,
            )
        if self.latency_config is None:
            self.latency_config = LatencyConfig(
                kernel_type=self.kernel_type,
                gpu_arch=self.gpu_arch,
            )
        else:
            # Backfill kernel_type / gpu_arch when caller built LatencyConfig
            # without them so resolve_method() works.
            if self.latency_config.kernel_type is None:
                self.latency_config.kernel_type = self.kernel_type
            if self.latency_config.gpu_arch is None:
                self.latency_config.gpu_arch = self.gpu_arch

    def _detect_gpu_arch(self) -> str:
        """Auto-detect GPU architecture."""
        try:
            from ..utils import detect_gpu

            vendor, arch = detect_gpu()
            if arch:
                return arch
            raise RuntimeError("GPU detected but architecture is None")
        except Exception as e:
            raise RuntimeError(
                f"Failed to detect GPU architecture for kernel type '{self.kernel_type.name}'. "
                "Please specify 'gpu_arch' explicitly or ensure GPU tools (rocminfo/nvidia-smi) are available."
            ) from e
