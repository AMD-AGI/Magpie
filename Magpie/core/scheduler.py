###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Task scheduler for kernel evaluation.

This module provides the Scheduler class which:
- Receives task requests from CLI/MCP
- Creates and manages Executors (Local, Container, future: Distributed)
- Distributes tasks to executors
- Manages task lifecycle
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .task import Task, TaskResult, TaskStatus, ModeType, ModeConfig
from .executor import (
    BaseExecutor,
    ExecutorConfig,
    ExecutorType,
    create_executor,
)
from ..config import KernelEvalConfig
from ..modes.benchmark.config import DEFAULT_SHARED_STORAGE_PATH

logger = logging.getLogger(__name__)


class EnvironmentType(Enum):
    """Type of execution environment."""

    LOCAL = "local"
    CONTAINER = "container"
    RAY = "ray"


@dataclass
class SchedulerConfig:
    """
    Configuration for the scheduler.

    Attributes:
        environment_type: Type of execution environment
        max_workers: Maximum number of concurrent workers
        docker_image: Docker image for container environment
        gpu_devices: List of GPU device IDs to use
        timeout_seconds: Default timeout for task execution
        pre_hooks: Functions to run before task execution
        post_hooks: Functions to run after task execution
    """

    environment_type: EnvironmentType = EnvironmentType.LOCAL
    max_workers: int = 1
    docker_image: Optional[str] = None
    gpu_devices: List[int] = field(default_factory=lambda: [0])
    timeout_seconds: float = 300.0
    ray_cluster_address: Optional[str] = None
    ray_shared_storage_path: Optional[str] = None  # shared FS root on Ray workers
    pre_hooks: List[Callable[[Task], None]] = field(default_factory=list)
    post_hooks: List[Callable[[Task, TaskResult], None]] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.environment_type, str):
            self.environment_type = EnvironmentType(self.environment_type)


class Scheduler:
    """
    Task scheduler for kernel evaluation.

    The scheduler is the main entry point for CLI/MCP to submit tasks.
    It handles:
    - Creating appropriate executors based on environment type
    - Managing task queue and distribution
    - Running pre/post execution hooks
    - Tracking task status and results
    """

    def __init__(self, config: Optional[SchedulerConfig] = None):
        """
        Initialize the scheduler.

        Args:
            config: Scheduler configuration
        """
        self.config = config or SchedulerConfig()
        self._executor: Optional[BaseExecutor] = None
        self._task_queue: List[Task] = []
        self._completed_tasks: Dict[str, TaskResult] = {}
        self._is_initialized = False

    def initialize(self) -> bool:
        """
        Initialize the scheduler and create executor.

        Returns:
            True if initialization succeeded, False otherwise
        """
        if self._is_initialized:
            logger.warning("Scheduler is already initialized")
            return True

        logger.info(
            f"Initializing scheduler with {self.config.environment_type.value} environment"
        )

        # Create executor config
        executor_config = self._create_executor_config()

        # Create executor (pass ray_config for RAY environment)
        extra_kwargs: Dict[str, Any] = {}
        if self.config.environment_type == EnvironmentType.RAY:
            from ..modes.benchmark.config import RayConfig
            extra_kwargs["ray_config"] = RayConfig(
                cluster_address=self.config.ray_cluster_address or "auto",
                shared_storage_path=(
                    self.config.ray_shared_storage_path
                    or DEFAULT_SHARED_STORAGE_PATH
                ),
            )
        self._executor = create_executor(executor_config, **extra_kwargs)

        # Start executor
        if not self._executor.start():
            logger.error("Failed to start executor")
            return False

        self._is_initialized = True
        logger.info("Scheduler initialized successfully")
        return True

    def _create_executor_config(self) -> ExecutorConfig:
        """Create executor configuration based on scheduler config."""
        if self.config.environment_type == EnvironmentType.LOCAL:
            executor_type = ExecutorType.LOCAL
        elif self.config.environment_type == EnvironmentType.CONTAINER:
            executor_type = ExecutorType.CONTAINER
        elif self.config.environment_type == EnvironmentType.RAY:
            executor_type = ExecutorType.RAY
        else:
            executor_type = ExecutorType.LOCAL

        return ExecutorConfig(
            executor_type=executor_type,
            max_workers=self.config.max_workers,
            gpu_devices=self.config.gpu_devices,
            docker_image=self.config.docker_image,
            timeout_seconds=self.config.timeout_seconds,
        )

    def create_task(
        self,
        kernel_configs: List[KernelEvalConfig],
        mode_type: ModeType = ModeType.ANALYZE,
        enable_default_compile: bool = False,
        check_performance: bool = True,
        gpu_arch: Optional[str] = None,
        timeout_seconds: float = 300.0,
        profiler_args: Optional[List[str]] = None,
        rocprof_config: Optional[Dict[str, Any]] = None,
        ncu_config: Optional[Dict[str, Any]] = None,
        metrix_config: Optional[Dict[str, Any]] = None,
        correctness_config: Optional[Dict[str, Any]] = None,
        baseline_index: int = 0,
        compare_config: Optional[Dict[str, Any]] = None,
        benchmark_config: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """
        Create a new task.

        Args:
            kernel_configs: Kernel configurations to evaluate
            mode_type: Type of evaluation mode
            enable_default_compile: Enable default compilation when no compile_command
            check_performance: Whether to run performance profiling
            gpu_arch: GPU architecture
            timeout_seconds: Timeout for profiling operations
            profiler_args: Additional arguments for the profiler (legacy)
            rocprof_config: rocprof-compute configuration dict
            ncu_config: ncu configuration dict
            baseline_index: Baseline kernel index for compare mode
            priority: Task priority
            metadata: Additional metadata

        Returns:
            Created Task object
        """
        mode_config = ModeConfig(
            mode_type=mode_type,
            enable_default_compile=enable_default_compile,
            check_performance=check_performance,
            gpu_arch=gpu_arch,
            timeout_seconds=timeout_seconds,
            profiler_args=profiler_args or [],
            rocprof_config=rocprof_config or {},
            ncu_config=ncu_config or {},
            metrix_config=metrix_config or {},
            correctness_config=correctness_config or {},
            baseline_index=baseline_index,
            compare_config=compare_config or {},
            benchmark_config=benchmark_config or {},
        )

        task = Task(
            kernel_configs=kernel_configs,
            mode_config=mode_config,
            priority=priority,
            metadata=metadata or {},
        )

        logger.debug(f"Created task {task.task_id} with mode {mode_type.value}")
        return task

    def submit(self, task: Task) -> str:
        """
        Submit a task for execution.

        The task will be added to the queue and executed
        when a worker becomes available.

        Args:
            task: Task to execute

        Returns:
            Task ID
        """
        if not self._is_initialized:
            raise RuntimeError("Scheduler is not initialized. Call initialize() first.")

        # Run pre-hooks
        self._run_pre_hooks(task)

        # Add to queue
        self._task_queue.append(task)

        # Submit to executor
        assert self._executor is not None
        task_id = self._executor.submit(task)

        logger.info(f"Task {task_id} submitted to scheduler")
        return task_id

    def execute(self, task: Task) -> TaskResult:
        """
        Execute a task synchronously.

        This method blocks until the task completes.

        Args:
            task: Task to execute

        Returns:
            TaskResult with execution results
        """
        if not self._is_initialized:
            raise RuntimeError("Scheduler is not initialized. Call initialize() first.")

        # Run pre-hooks
        self._run_pre_hooks(task)

        # Execute directly
        assert self._executor is not None
        result = self._executor.execute(task)

        # Run post-hooks
        self._run_post_hooks(task, result)

        # Store result
        self._completed_tasks[task.task_id] = result

        logger.info(f"Task {task.task_id} completed with status: {result.status.value}")
        return result

    def run_analyze(
        self,
        kernel_configs: List[KernelEvalConfig],
        enable_default_compile: bool = False,
        check_performance: bool = True,
        gpu_arch: Optional[str] = None,
        timeout_seconds: float = 300.0,
        profiler_args: Optional[List[str]] = None,
        rocprof_config: Optional[Dict[str, Any]] = None,
        ncu_config: Optional[Dict[str, Any]] = None,
        metrix_config: Optional[Dict[str, Any]] = None,
        correctness_config: Optional[Dict[str, Any]] = None,
    ) -> TaskResult:
        """
        Convenience method to run analyze mode.

        Args:
            kernel_configs: Kernel configurations to analyze
            enable_default_compile: Enable default compilation when no compile_command
            check_performance: Whether to run performance profiling
            gpu_arch: GPU architecture
            timeout_seconds: Timeout for profiling operations
            profiler_args: Additional arguments for the profiler (legacy)
            rocprof_config: rocprof-compute configuration dict
            ncu_config: ncu configuration dict
            metrix_config: Metrix configuration dict

        Returns:
            TaskResult with analysis results
        """
        task = self.create_task(
            kernel_configs=kernel_configs,
            mode_type=ModeType.ANALYZE,
            enable_default_compile=enable_default_compile,
            check_performance=check_performance,
            gpu_arch=gpu_arch,
            timeout_seconds=timeout_seconds,
            profiler_args=profiler_args,
            rocprof_config=rocprof_config,
            ncu_config=ncu_config,
            metrix_config=metrix_config,
            correctness_config=correctness_config,
        )
        return self.execute(task)

    def run_compare(
        self,
        kernel_configs: List[KernelEvalConfig],
        baseline_index: int = 0,
        enable_default_compile: bool = False,
        check_performance: bool = True,
        gpu_arch: Optional[str] = None,
        timeout_seconds: float = 300.0,
        profiler_args: Optional[List[str]] = None,
        rocprof_config: Optional[Dict[str, Any]] = None,
        ncu_config: Optional[Dict[str, Any]] = None,
        metrix_config: Optional[Dict[str, Any]] = None,
        correctness_config: Optional[Dict[str, Any]] = None,
        compare_config: Optional[Dict[str, Any]] = None,
    ) -> TaskResult:
        """
        Convenience method to run compare mode.

        Args:
            kernel_configs: Kernel configurations to compare
            baseline_index: Index of baseline kernel
            enable_default_compile: Enable default compilation when no compile_command
            check_performance: Whether to run performance profiling
            gpu_arch: GPU architecture
            timeout_seconds: Timeout for profiling operations
            profiler_args: Additional arguments for the profiler (legacy)
            rocprof_config: rocprof-compute configuration dict
            ncu_config: ncu configuration dict
            metrix_config: Metrix configuration dict

        Returns:
            TaskResult with comparison results
        """
        task = self.create_task(
            kernel_configs=kernel_configs,
            mode_type=ModeType.COMPARE,
            enable_default_compile=enable_default_compile,
            baseline_index=baseline_index,
            check_performance=check_performance,
            gpu_arch=gpu_arch,
            timeout_seconds=timeout_seconds,
            profiler_args=profiler_args,
            rocprof_config=rocprof_config,
            ncu_config=ncu_config,
            metrix_config=metrix_config,
            correctness_config=correctness_config,
            compare_config=compare_config,
        )
        return self.execute(task)

    def wait_for_task(
        self, task_id: str, timeout: Optional[float] = None
    ) -> TaskResult:
        """
        Wait for a specific task to complete.

        Args:
            task_id: Task ID to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            TaskResult when task completes
        """
        if task_id in self._completed_tasks:
            return self._completed_tasks[task_id]

        # Both LocalExecutor and RayJobExecutor support wait_for_task
        if self._executor and hasattr(self._executor, "wait_for_task"):
            result = self._executor.wait_for_task(task_id, timeout)

            # Find task and run post-hooks
            for task in self._task_queue:
                if task.task_id == task_id:
                    self._run_post_hooks(task, result)
                    break

            self._completed_tasks[task_id] = result
            return result

        raise ValueError(f"Task {task_id} not found")

    def wait_all(self, timeout: Optional[float] = None) -> List[TaskResult]:
        """
        Wait for all submitted tasks to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            List of TaskResults
        """
        results = []

        if self._executor and hasattr(self._executor, "wait_all"):
            results = self._executor.wait_all(timeout)

            # Run post-hooks for all tasks
            for result in results:
                for task in self._task_queue:
                    if task.task_id == result.task_id:
                        self._run_post_hooks(task, result)
                        self._completed_tasks[task.task_id] = result
                        break

        return results

    def submit_batch(self, tasks: List[Task]) -> List[str]:
        """
        Submit multiple tasks for parallel execution.

        Args:
            tasks: List of tasks to submit

        Returns:
            List of task IDs
        """
        task_ids = []
        for task in tasks:
            task_id = self.submit(task)
            task_ids.append(task_id)
        return task_ids

    def execute_batch(self, tasks: List[Task]) -> List[TaskResult]:
        """
        Execute multiple tasks in parallel and wait for all results.

        This is the recommended way to run multiple tasks in parallel.
        Uses the executor's ProcessPoolExecutor for parallel execution.

        Args:
            tasks: List of tasks to execute

        Returns:
            List of TaskResults in the same order as input tasks
        """
        if not self._is_initialized:
            raise RuntimeError("Scheduler is not initialized. Call initialize() first.")

        if not tasks:
            return []

        # Submit all tasks
        logger.info(f"Submitting {len(tasks)} tasks for parallel execution")
        task_ids = self.submit_batch(tasks)

        # Wait for all to complete
        results = self.wait_all()

        # Sort results to match input order
        result_map = {r.task_id: r for r in results}
        missing = [tid for tid in task_ids if tid not in result_map]
        if missing:
            raise RuntimeError(f"Missing results for task IDs: {missing}")
        ordered_results = [result_map[tid] for tid in task_ids]

        return ordered_results

    def run_analyze_batch(
        self,
        kernel_configs_list: List[List[KernelEvalConfig]],
        enable_default_compile: bool = False,
        check_performance: bool = True,
        gpu_arch: Optional[str] = None,
        timeout_seconds: float = 300.0,
        profiler_args: Optional[List[str]] = None,
        rocprof_config: Optional[Dict[str, Any]] = None,
        ncu_config: Optional[Dict[str, Any]] = None,
        metrix_config: Optional[Dict[str, Any]] = None,
    ) -> List[TaskResult]:
        """
        Run multiple analyze tasks in parallel at Scheduler level.

        Each item in kernel_configs_list becomes a separate Task that runs in parallel.

        Args:
            kernel_configs_list: List of kernel config lists (each becomes a Task)
            Other args same as run_analyze

        Returns:
            List of TaskResults

        Example:
            # Run 4 analyze tasks in parallel
            results = scheduler.run_analyze_batch([
                [kernel_config_1],
                [kernel_config_2],
                [kernel_config_3],
                [kernel_config_4],
            ])
        """
        tasks = []
        for kernel_configs in kernel_configs_list:
            task = self.create_task(
                kernel_configs=kernel_configs,
                mode_type=ModeType.ANALYZE,
                enable_default_compile=enable_default_compile,
                check_performance=check_performance,
                gpu_arch=gpu_arch,
                timeout_seconds=timeout_seconds,
                profiler_args=profiler_args,
                rocprof_config=rocprof_config,
                ncu_config=ncu_config,
                metrix_config=metrix_config,
            )
            tasks.append(task)

        return self.execute_batch(tasks)

    def run_compare_batch(
        self,
        kernel_configs_list: List[List[KernelEvalConfig]],
        baseline_index: int = 0,
        enable_default_compile: bool = False,
        check_performance: bool = True,
        gpu_arch: Optional[str] = None,
        timeout_seconds: float = 300.0,
        profiler_args: Optional[List[str]] = None,
        rocprof_config: Optional[Dict[str, Any]] = None,
        ncu_config: Optional[Dict[str, Any]] = None,
        metrix_config: Optional[Dict[str, Any]] = None,
    ) -> List[TaskResult]:
        """
        Run multiple compare tasks in parallel at Scheduler level.

        Each item in kernel_configs_list becomes a separate Task that runs in parallel.

        Args:
            kernel_configs_list: List of kernel config lists (each becomes a Compare Task)
            Other args same as run_compare

        Returns:
            List of TaskResults

        Example:
            # Run 2 compare tasks in parallel
            results = scheduler.run_compare_batch([
                [baseline_1, optimized_1a, optimized_1b],  # Compare set 1
                [baseline_2, optimized_2a, optimized_2b],  # Compare set 2
            ])
        """
        tasks = []
        for kernel_configs in kernel_configs_list:
            task = self.create_task(
                kernel_configs=kernel_configs,
                mode_type=ModeType.COMPARE,
                enable_default_compile=enable_default_compile,
                baseline_index=baseline_index,
                check_performance=check_performance,
                gpu_arch=gpu_arch,
                timeout_seconds=timeout_seconds,
                profiler_args=profiler_args,
                rocprof_config=rocprof_config,
                ncu_config=ncu_config,
                metrix_config=metrix_config,
            )
            tasks.append(task)

        return self.execute_batch(tasks)

    def run_benchmark(
        self,
        benchmark_config: Dict[str, Any],
        gpu_arch: Optional[str] = None,
        timeout_seconds: float = 3600.0,
    ) -> TaskResult:
        """
        Run benchmark mode.

        Benchmark mode always uses container environment for execution.
        Uses InferenceX as backend for vLLM/SGLang benchmarks.

        Args:
            benchmark_config: Benchmark configuration dict containing:
                - framework: "vllm" or "sglang"
                - model: Model name or path
                - precision: "fp8", "fp16", "bf16"
                - params: Dict with TP, CONC, ISL, OSL, etc.
                - profiler: Profiler configuration
                - docker_image: Optional image override
            gpu_arch: GPU architecture (auto-detected if not specified)
            timeout_seconds: Benchmark timeout

        Returns:
            TaskResult with benchmark results
        """
        # Benchmark mode forces container environment
        if self.config.environment_type != EnvironmentType.CONTAINER:
            logger.info("Benchmark mode: forcing container environment")
            # Note: We don't actually switch executor here, the benchmark
            # is executed directly via BenchmarkMode which handles its own
            # Docker execution
        
        task = self.create_task(
            kernel_configs=[],  # Benchmark mode doesn't use kernel configs
            mode_type=ModeType.BENCHMARK,
            gpu_arch=gpu_arch,
            timeout_seconds=timeout_seconds,
            benchmark_config=benchmark_config,
        )
        return self.execute(task)

    def run_benchmark_ray(
        self,
        benchmark_config: Dict[str, Any],
        ray_cluster_address: str = "auto",
        ray_shared_storage_path: str = DEFAULT_SHARED_STORAGE_PATH,
        gpu_arch: Optional[str] = None,
        timeout_seconds: float = 3600.0,
    ) -> TaskResult:
        """
        Submit a benchmark to a remote Ray cluster.

        Unlike run_benchmark() which blocks, this submits asynchronously
        and returns a TaskResult whose metadata contains the task_id.

        Args:
            benchmark_config: Benchmark configuration dict
            ray_cluster_address: Ray head address
            ray_shared_storage_path: Shared filesystem root on workers (HF cache, results)
            gpu_arch: GPU architecture
            timeout_seconds: Timeout

        Returns:
            TaskResult with task_id in metadata
        """
        benchmark_config["run_mode"] = "ray"
        benchmark_config["ray_config"] = {
            "cluster_address": ray_cluster_address,
            "shared_storage_path": ray_shared_storage_path,
        }

        task = self.create_task(
            kernel_configs=[],
            mode_type=ModeType.BENCHMARK,
            gpu_arch=gpu_arch,
            timeout_seconds=timeout_seconds,
            benchmark_config=benchmark_config,
        )

        # For Ray, we use BenchmarkMode directly (it handles its own executor)
        from ..modes.benchmark import BenchmarkMode, BenchmarkConfig
        config = BenchmarkConfig.from_dict(benchmark_config)
        benchmarker = BenchmarkMode(config)
        bench_result = benchmarker.run(task_id=task.task_id)

        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.COMPLETED if bench_result.success else TaskStatus.FAILED,
            results=bench_result.to_dict(),
            errors=bench_result.errors,
            metadata=bench_result.metadata or {},
        )

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Get the status of a task.

        Args:
            task_id: Task ID

        Returns:
            Task status or None if not found
        """
        # Check completed tasks
        if task_id in self._completed_tasks:
            return self._completed_tasks[task_id].status

        # Check executor
        if self._executor:
            return self._executor.get_task_status(task_id)

        # Check queue
        for task in self._task_queue:
            if task.task_id == task_id:
                return task.status

        return None

    def get_pending_tasks(self) -> List[Task]:
        """
        Get list of pending tasks.

        Returns:
            List of pending tasks
        """
        return [t for t in self._task_queue if t.status == TaskStatus.PENDING]

    def get_completed_results(self) -> Dict[str, TaskResult]:
        """
        Get all completed task results.

        Returns:
            Dictionary mapping task IDs to results
        """
        return self._completed_tasks.copy()

    def clear_tasks(self) -> None:
        """Clear all pending tasks from queue."""
        self._task_queue = [
            t for t in self._task_queue if t.status == TaskStatus.RUNNING
        ]
        logger.info("Cleared pending tasks")

    def _run_pre_hooks(self, task: Task) -> None:
        """Run pre-execution hooks."""
        for hook in self.config.pre_hooks:
            try:
                hook(task)
            except Exception as e:
                logger.warning(f"Pre-hook failed: {e}")

    def _run_post_hooks(self, task: Task, result: TaskResult) -> None:
        """Run post-execution hooks."""
        for hook in self.config.post_hooks:
            try:
                hook(task, result)
            except Exception as e:
                logger.warning(f"Post-hook failed: {e}")

    def is_initialized(self) -> bool:
        """Check if scheduler is initialized."""
        return self._is_initialized

    def shutdown(self) -> None:
        """Shutdown the scheduler and cleanup resources."""
        if self._executor:
            self._executor.stop()
            self._executor = None

        self._task_queue.clear()
        self._completed_tasks.clear()
        self._is_initialized = False
        logger.info("Scheduler shutdown completed")
