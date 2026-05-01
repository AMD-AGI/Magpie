###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Analyze mode for single kernel analysis.

In analyze mode:
- A testcase command is required
- The kernel is compiled, testcase is run, and performance is measured
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...config import (
    KernelType,
    EvalMode,
    PipelineConfig,
    KernelEvalConfig,
    CompilingConfig,
    CorrectnessConfig,
    CorrectnessMode,
    LatencyConfig,
    PerformanceConfig,
)
from ...config.performance import RocprofComputeConfig, NcuConfig, MetrixConfig, PerfBackend
from ...eval import Evaluator, EvaluationState

logger = logging.getLogger(__name__)


@dataclass
class AnalyzeConfig:
    """
    Configuration for analyze mode.

    Attributes:
        kernel_type: Default kernel type
        gpu_arch: GPU architecture
        enable_default_compile: Enable default compilation when no compile_command
        check_performance: Whether to run performance profiling
        timeout_seconds: Timeout for profiling operations
        profiler_args: Additional arguments for the profiler (legacy)
        rocprof_config: rocprof-compute configuration dict
        ncu_config: ncu configuration dict
    """

    kernel_type: KernelType = KernelType.HIP
    gpu_arch: Optional[str] = None
    enable_default_compile: bool = False
    check_performance: bool = True
    timeout_seconds: float = 300.0
    profiler_args: List[str] = field(default_factory=list)
    rocprof_config: Dict[str, Any] = field(default_factory=dict)
    ncu_config: Dict[str, Any] = field(default_factory=dict)
    metrix_config: Dict[str, Any] = field(default_factory=dict)
    correctness_config: Dict[str, Any] = field(default_factory=dict)
    latency_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.gpu_arch is None:
            self._detect_gpu_arch()

    def _detect_gpu_arch(self) -> None:
        try:
            from ...utils import detect_gpu
            _, arch = detect_gpu()
            if arch:
                self.gpu_arch = arch
        except Exception:
            pass


class AnalyzeMode:
    """
    Analyzer for individual kernel evaluation.

    Requires testcase_command to be provided in KernelEvalConfig.
    """

    def __init__(self, config: Optional[AnalyzeConfig] = None):
        self.config = config or AnalyzeConfig()

    def analyze(self, kernel_cfg: KernelEvalConfig) -> EvaluationState:
        """
        Analyze a single kernel.

        Args:
            kernel_cfg: Kernel configuration (must include testcase_command)

        Returns:
            EvaluationState with analysis results
        """
        # Validate that testcase is provided
        if not kernel_cfg.has_testcase():
            logger.error("Analyze mode requires testcase_command")
            state = EvaluationState()
            state.errors.append("Analyze mode requires testcase_command")
            return state

        # Build rocprof-compute config if provided
        rocprof_cfg = None
        if self.config.rocprof_config:
            rocprof_cfg = RocprofComputeConfig(
                output_dir=self.config.rocprof_config.get("output_dir"),
                workload_dir=self.config.rocprof_config.get(
                    "workload_dir", "./workloads"
                ),
                metric_blocks=self.config.rocprof_config.get(
                    "metric_blocks", ["1", "2", "5", "10", "11", "12", "14", "16", "17"]
                ),
                output_format=self.config.rocprof_config.get("output_format", "csv"),
                profile_args=self.config.rocprof_config.get("profile_args", []),
                analyze_args=self.config.rocprof_config.get("analyze_args", []),
            )

        # Build ncu config if provided
        ncu_cfg = None
        if self.config.ncu_config:
            ncu_cfg = NcuConfig(
                args=self.config.ncu_config.get("args", []),
                metrics=self.config.ncu_config.get("metrics", []),
            )

        # Build metrix config if provided
        metrix_cfg = None
        explicit_backend = None
        if self.config.metrix_config:
            metrix_cfg = MetrixConfig(
                output_dir=self.config.metrix_config.get("output_dir"),
                profile=self.config.metrix_config.get("profile"),
                metrics=self.config.metrix_config.get("metrics", []),
                kernel_filter=self.config.metrix_config.get("kernel_filter"),
                num_replays=self.config.metrix_config.get("num_replays", 1),
                timeout_seconds=self.config.metrix_config.get("timeout_seconds", 60),
                extra_args=self.config.metrix_config.get("extra_args", []),
            )
            if self.config.metrix_config.get("backend") == "metrix":
                explicit_backend = PerfBackend.METRIX

        # Build correctness config
        corr_cfg = self._build_correctness_config()

        # Build latency config from dict
        lat_cfg = LatencyConfig.from_dict(
            self.config.latency_config,
            kernel_type=kernel_cfg.kernel_type,
            gpu_arch=self.config.gpu_arch,
        )

        # Build pipeline config for analyze mode
        pipeline_cfg = PipelineConfig(
            mode=EvalMode.ANALYZE,
            kernel_type=kernel_cfg.kernel_type,
            gpu_arch=self.config.gpu_arch,
            compiling_config=CompilingConfig(
                enable_default_compile=self.config.enable_default_compile,
            ),
            correctness_config=corr_cfg,
            performance_config=PerformanceConfig(
                enabled=self.config.check_performance,
                backend=explicit_backend,
                kernel_type=kernel_cfg.kernel_type,
                gpu_arch=self.config.gpu_arch,
                timeout_seconds=self.config.timeout_seconds,
                profiler_args=self.config.profiler_args,
                rocprof_config=rocprof_cfg,
                ncu_config=ncu_cfg,
                metrix_config=metrix_cfg,
            ),
            latency_config=lat_cfg,
        )

        evaluator = Evaluator(pipeline_cfg)
        state = evaluator.evaluate(kernel_cfg)

        self._log_summary(kernel_cfg, state)
        return state

    def analyze_batch(
        self, kernel_configs: List[KernelEvalConfig]
    ) -> List[EvaluationState]:
        """
        Analyze multiple kernels sequentially.

        Args:
            kernel_configs: List of kernel configurations to analyze

        Returns:
            List of EvaluationState results
        """
        if not kernel_configs:
            return []

        logger.info(f"Analyzing {len(kernel_configs)} kernels sequentially")

        results = []
        for i, kernel_cfg in enumerate(kernel_configs):
            logger.info(
                f"Analyzing kernel {i + 1}/{len(kernel_configs)}: {kernel_cfg.kernel_id}"
            )
            state = self.analyze(kernel_cfg)
            results.append(state)

        return results

    def _build_correctness_config(self) -> CorrectnessConfig:
        """Build CorrectnessConfig from the correctness_config dict."""
        return CorrectnessConfig.from_dict(
            self.config.correctness_config, mode=CorrectnessMode.TESTCASE
        )

    def _log_summary(
        self, kernel_cfg: KernelEvalConfig, state: EvaluationState
    ) -> None:
        """Log analysis summary."""
        logger.info(f"Analysis complete: {kernel_cfg.kernel_id}")
        logger.info(f"  Compiling: {state.compiling_state.name}")
        logger.info(f"  Correctness: {state.correctness_state.name}")
        logger.info(f"  Performance: {state.performance_state.name}")
        logger.info(f"  Score: {state.score:.2f}")
