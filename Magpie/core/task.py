###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Task definitions for kernel evaluation.

This module defines Task and TaskResult for unified task management.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..config import KernelEvalConfig


class TaskStatus(Enum):
    """Status of a task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModeType(Enum):
    """Type of evaluation mode."""

    ANALYZE = "analyze"
    COMPARE = "compare"
    BENCHMARK = "benchmark"


@dataclass
class ModeConfig:
    """
    Configuration for the evaluation mode.

    Attributes:
        mode_type: Type of mode (analyze, compare, or benchmark)
        enable_default_compile: Enable default compilation when no compile_command
        check_performance: Whether to run performance profiling
        gpu_arch: GPU architecture
        timeout_seconds: Timeout for profiling operations
        profiler_args: Additional arguments for the profiler (legacy)
        rocprof_config: rocprof-compute configuration dict
        ncu_config: ncu configuration dict
        baseline_index: Baseline kernel index for compare mode
        benchmark_config: Configuration for benchmark mode
    """

    mode_type: ModeType = ModeType.ANALYZE
    enable_default_compile: bool = False
    check_performance: bool = True
    gpu_arch: Optional[str] = None
    timeout_seconds: float = 300.0
    profiler_args: List[str] = field(default_factory=list)
    rocprof_config: Dict[str, Any] = field(default_factory=dict)
    ncu_config: Dict[str, Any] = field(default_factory=dict)
    metrix_config: Dict[str, Any] = field(default_factory=dict)
    correctness_config: Dict[str, Any] = field(default_factory=dict)
    latency_config: Dict[str, Any] = field(default_factory=dict)
    baseline_index: int = 0  # For compare mode
    compare_config: Dict[str, Any] = field(default_factory=dict)
    benchmark_config: Dict[str, Any] = field(default_factory=dict)  # For benchmark mode


@dataclass
class Task:
    """
    A task to be executed by the Executor.

    Attributes:
        task_id: Unique identifier for the task
        kernel_configs: Kernel configurations to evaluate
        mode_config: Mode configuration
        status: Current status of the task
        priority: Task priority (higher = more important)
        metadata: Additional metadata
    """

    kernel_configs: List[KernelEvalConfig]
    mode_config: ModeConfig = field(default_factory=ModeConfig)
    task_id: str = field(default_factory=lambda: str(uuid4())[:8])
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = TaskStatus(self.status)
        if isinstance(self.mode_config, dict):
            self.mode_config = ModeConfig(**self.mode_config)

    @property
    def mode_type(self) -> ModeType:
        """Get the mode type of this task."""
        return self.mode_config.mode_type

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "kernel_configs": [
                cfg.to_dict() if hasattr(cfg, "to_dict") else str(cfg)
                for cfg in self.kernel_configs
            ],
            "mode_config": {
                "mode_type": self.mode_config.mode_type.value,
                "enable_default_compile": self.mode_config.enable_default_compile,
                "check_performance": self.mode_config.check_performance,
                "gpu_arch": self.mode_config.gpu_arch,
                "timeout_seconds": self.mode_config.timeout_seconds,
                "profiler_args": self.mode_config.profiler_args,
                "rocprof_config": self.mode_config.rocprof_config,
                "ncu_config": self.mode_config.ncu_config,
                "metrix_config": self.mode_config.metrix_config,
                "correctness_config": self.mode_config.correctness_config,
                "latency_config": self.mode_config.latency_config,
                "baseline_index": self.mode_config.baseline_index,
                "compare_config": self.mode_config.compare_config,
                "benchmark_config": self.mode_config.benchmark_config,
            },
            "status": self.status.value,
            "priority": self.priority,
            "metadata": self.metadata,
        }


@dataclass
class TaskResult:
    """
    Result of a task execution.

    Attributes:
        task_id: ID of the task
        status: Final status of the task
        results: Evaluation results (EvaluationState list or ComparisonResult)
        errors: List of errors encountered
        execution_time: Time taken to execute the task in seconds
        metadata: Additional metadata
    """

    task_id: str
    status: TaskStatus = TaskStatus.COMPLETED
    results: Any = None
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = TaskStatus(self.status)

    @property
    def success(self) -> bool:
        """Check if task completed successfully."""
        return self.status == TaskStatus.COMPLETED and not self.errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        results_dict = None
        if self.results is not None:
            if isinstance(self.results, list):
                results_dict = [
                    r.to_dict() if hasattr(r, "to_dict") else r for r in self.results
                ]
            elif hasattr(self.results, "to_dict"):
                results_dict = self.results.to_dict()
            else:
                results_dict = self.results

        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "success": self.success,
            "results": results_dict,
            "errors": self.errors,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }
