###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Compare mode for kernel comparison.

In compare mode:
- If testcase_command is provided, use it
- Otherwise, for PyTorch kernels, generate inputs and compare outputs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ...config import (
    EvalMode,
    PipelineConfig,
    KernelEvalConfig,
    KernelType,
    CompilingConfig,
    CorrectnessConfig,
    CorrectnessMode,
    LatencyConfig,
    PerformanceConfig,
)
from ...config.performance import RocprofComputeConfig, NcuConfig, MetrixConfig, PerfBackend
from ...eval import Evaluator, EvaluationState, BaseKind

logger = logging.getLogger(__name__)


@dataclass
class CompareConfig:
    """
    Configuration for compare mode.

    Attributes:
        baseline_index: Index of baseline kernel for comparison
        gpu_arch: GPU architecture
        enable_default_compile: Enable default compilation when no compile_command
        check_performance: Whether to run performance profiling
        timeout_seconds: Timeout for profiling operations
        profiler_args: Additional arguments for the profiler (legacy)
        rocprof_config: rocprof-compute configuration dict
        ncu_config: ncu configuration dict
    """

    baseline_index: int = 0
    gpu_arch: Optional[str] = None
    enable_default_compile: bool = False

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
    check_performance: bool = True
    timeout_seconds: float = 300.0
    profiler_args: List[str] = field(default_factory=list)
    rocprof_config: Dict[str, Any] = field(default_factory=dict)
    ncu_config: Dict[str, Any] = field(default_factory=dict)
    metrix_config: Dict[str, Any] = field(default_factory=dict)
    correctness_config: Dict[str, Any] = field(default_factory=dict)
    latency_config: Dict[str, Any] = field(default_factory=dict)
    # Winner selection strategy: "correctness_first" or "perf_score"
    winner_strategy: str = "perf_score"
    # Per-backend scoring weights
    perf_weights_rocprof: Dict[str, float] = field(default_factory=dict)
    perf_weights_ncu: Dict[str, float] = field(default_factory=dict)
    perf_weights_metrix: Dict[str, float] = field(default_factory=dict)
    # Metrics where lower values are better (e.g., duration)
    perf_lower_is_better: List[str] = field(
        default_factory=lambda: ["duration_ns_total", "LDS_Conflicts"]
    )


@dataclass
class ComparisonResult:
    """
    Result of comparing kernel implementations.
    """

    kernel_results: List[EvaluationState] = field(default_factory=list)
    comparison_metrics: Dict[str, Any] = field(default_factory=dict)
    winner: Optional[int] = None
    rankings: List[Tuple[int, float]] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kernel_results": [r.to_dict() for r in self.kernel_results],
            "comparison_metrics": self.comparison_metrics,
            "winner": self.winner,
            "rankings": self.rankings,
            "summary": self.summary,
        }


class CompareMode:
    """
    Comparator for multiple kernel implementations.
    """

    def __init__(self, config: Optional[CompareConfig] = None):
        self.config = config or CompareConfig()

    def compare(self, kernel_configs: List[KernelEvalConfig]) -> ComparisonResult:
        """
        Compare multiple kernel implementations.

        Args:
            kernel_configs: List of kernel configurations to compare

        Returns:
            ComparisonResult with comparison results
        """
        if len(kernel_configs) < 2:
            raise ValueError("Need at least 2 kernels for comparison")

        logger.info(f"Comparing {len(kernel_configs)} kernels sequentially")
        results = self._compare_sequential(kernel_configs)

        # Build comparison
        comparison = self._build_comparison(results, kernel_configs)
        self._log_summary(comparison)

        return comparison

    def _compare_sequential(
        self, kernel_configs: List[KernelEvalConfig]
    ) -> List[EvaluationState]:
        """Evaluate kernels sequentially."""
        results = []
        for i, cfg in enumerate(kernel_configs):
            logger.info(
                f"Evaluating kernel {i + 1}/{len(kernel_configs)}: {cfg.kernel_id}"
            )

            # Determine correctness mode
            if cfg.has_testcase():
                corr_mode = CorrectnessMode.TESTCASE
            else:
                corr_mode = CorrectnessMode.RESULT_COMPARISON

            # Build rocprof-compute config if provided
            rocprof_cfg = None
            if self.config.rocprof_config:
                rocprof_cfg = RocprofComputeConfig(
                    output_dir=self.config.rocprof_config.get("output_dir"),
                    workload_dir=self.config.rocprof_config.get(
                        "workload_dir", "./workloads"
                    ),
                    metric_blocks=self.config.rocprof_config.get(
                        "metric_blocks",
                        ["1", "2", "5", "10", "11", "12", "14", "16", "17"],
                    ),
                    output_format=self.config.rocprof_config.get(
                        "output_format", "csv"
                    ),
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
            corr_cfg = self._build_correctness_config(corr_mode)

            # Build latency config from dict
            lat_cfg = LatencyConfig.from_dict(
                self.config.latency_config,
                kernel_type=cfg.kernel_type,
                gpu_arch=self.config.gpu_arch,
            )

            # Build pipeline config
            pipeline_cfg = PipelineConfig(
                mode=EvalMode.COMPARE,
                kernel_type=cfg.kernel_type,
                gpu_arch=self.config.gpu_arch,
                compiling_config=CompilingConfig(
                    enable_default_compile=self.config.enable_default_compile,
                ),
                correctness_config=corr_cfg,
                performance_config=PerformanceConfig(
                    enabled=self.config.check_performance,
                    backend=explicit_backend,
                    kernel_type=cfg.kernel_type,
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
            state = evaluator.evaluate(cfg)
            results.append(state)

        return results

    def _build_correctness_config(
        self, corr_mode: CorrectnessMode
    ) -> CorrectnessConfig:
        """Build CorrectnessConfig from the correctness_config dict."""
        return CorrectnessConfig.from_dict(
            self.config.correctness_config, mode=corr_mode
        )

    def _build_comparison(
        self, results: List[EvaluationState], kernel_configs: List[KernelEvalConfig]
    ) -> ComparisonResult:
        """Build comparison from individual results."""
        comparison = ComparisonResult(kernel_results=results)

        # Check correctness
        correctness_results = []
        for r in results:
            is_correct = (
                r.correctness_state == BaseKind.SUCCESS
                and r.correctness_result is not None
                and r.correctness_result.success
            )
            correctness_results.append(is_correct)

        comparison.comparison_metrics["correctness"] = correctness_results
        comparison.comparison_metrics["all_correct"] = all(correctness_results)

        # Performance-based scoring (optional)
        kernel_type = kernel_configs[0].kernel_type if kernel_configs else KernelType.HIP
        perf_scores = self._compute_perf_scores(
            results, correctness_results, kernel_type
        )
        comparison.comparison_metrics["perf_scores"] = perf_scores
        comparison.rankings = sorted(
            [(i, s) for i, s in enumerate(perf_scores) if s is not None],
            key=lambda x: x[1],
            reverse=True,
        )

        if self.config.winner_strategy == "perf_score" and comparison.rankings:
            comparison.winner = comparison.rankings[0][0]
        else:
            # Fallback: first kernel that passes correctness
            for i, correct in enumerate(correctness_results):
                if correct:
                    comparison.winner = i
                    break

        # Generate summary
        comparison.summary = self._generate_summary(comparison, kernel_configs)

        return comparison

    def _get_perf_weights(self, kernel_type: KernelType) -> Dict[str, float]:
        """Select per-backend weights, with sane fallbacks."""
        # If metrix backend is explicitly selected, prefer metrix weights
        is_metrix = (
            self.config.metrix_config
            and self.config.metrix_config.get("backend") == "metrix"
        )
        if is_metrix and self.config.perf_weights_metrix:
            return self.config.perf_weights_metrix

        if kernel_type == KernelType.CUDA:
            if self.config.perf_weights_ncu:
                return self.config.perf_weights_ncu
            return self.config.perf_weights_rocprof
        if self.config.perf_weights_rocprof:
            return self.config.perf_weights_rocprof
        return self.config.perf_weights_ncu

    def _get_perf_value(
        self, state: EvaluationState, metric: str
    ) -> Optional[float]:
        perf = state.performance_result
        if perf is None or not perf.success:
            return None

        # Summary metrics (FLOPs, bandwidth, utilization, etc.)
        summary = perf.get_summary_metrics()
        entry = summary.get(metric)
        if entry:
            if entry.get("pct_of_peak") is not None:
                return float(entry["pct_of_peak"])
            val = entry.get("value")
            return float(val) if val is not None else None

        # Synthetic metric: total kernel duration (ns)
        if metric == "duration_ns_total":
            total = 0.0
            found = False
            for kernel in perf.get_kernel_summary():
                dur = kernel.get("duration_ns", {}).get("total")
                if dur is not None:
                    total += float(dur)
                    found = True
            return total if found else None

        return None

    def _compute_perf_scores(
        self,
        results: List[EvaluationState],
        correctness_results: List[bool],
        kernel_type: KernelType,
    ) -> List[Optional[float]]:
        weights = self._get_perf_weights(kernel_type)
        if not weights:
            return [None for _ in results]

        # Gather values per metric for normalization
        metric_values: Dict[str, List[Optional[float]]] = {
            metric: [] for metric in weights
        }
        for state, correct in zip(results, correctness_results):
            for metric in weights:
                val = self._get_perf_value(state, metric) if correct else None
                metric_values[metric].append(val)

        scores: List[Optional[float]] = []
        for idx, correct in enumerate(correctness_results):
            if not correct:
                scores.append(None)
                continue

            score = 0.0
            used_any = False
            for metric, weight in weights.items():
                values = [v for v in metric_values[metric] if v is not None]
                if not values:
                    continue
                val = metric_values[metric][idx]
                if val is None:
                    continue

                if metric in self.config.perf_lower_is_better:
                    denom = val if val != 0 else None
                    norm = (min(values) / denom) if denom else 0.0
                else:
                    denom = max(values)
                    norm = (val / denom) if denom else 0.0

                score += float(weight) * float(norm)
                used_any = True

            scores.append(score if used_any else None)

        return scores

    def _generate_summary(
        self, comparison: ComparisonResult, kernel_configs: List[KernelEvalConfig]
    ) -> str:
        """Generate human-readable comparison summary."""
        lines = [
            f"Compared {len(comparison.kernel_results)} kernel implementations",
            "",
        ]

        # Correctness summary
        correct_count = sum(comparison.comparison_metrics.get("correctness", []))
        total_count = len(comparison.kernel_results)
        lines.append(f"Correctness: {correct_count}/{total_count} passed")

        # Individual results
        lines.append("")
        lines.append("Results:")
        for i, (cfg, result) in enumerate(
            zip(kernel_configs, comparison.kernel_results)
        ):
            status = "✓" if comparison.comparison_metrics["correctness"][i] else "✗"
            lines.append(f"  {status} Kernel {i}: {cfg.kernel_id}")

        # Winner
        if comparison.winner is not None:
            lines.append("")
            lines.append(
                f"Winner: Kernel {comparison.winner} ({kernel_configs[comparison.winner].kernel_id})"
            )

        return "\n".join(lines)

    def _log_summary(self, comparison: ComparisonResult) -> None:
        """Log comparison summary."""
        logger.info("Comparison complete")
        for line in comparison.summary.split("\n"):
            if line.strip():
                logger.info(f"  {line}")
