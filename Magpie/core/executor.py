###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Executor implementations for kernel evaluation.

This module provides:
- BaseExecutor: Abstract base class for executors
- LocalExecutor: Execute tasks locally with multiprocessing
- ContainerExecutor: Execute tasks in Docker containers
"""

import logging
import multiprocessing
import os
import subprocess
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .task import Task, TaskResult, TaskStatus, ModeType
from ..config import KernelEvalConfig, KernelType
from ..utils.gpu import get_gpu_count

logger = logging.getLogger(__name__)


def _local_pool_worker_init(counter, lock, gpu_devices: Tuple[int, ...]) -> None:
    """ProcessPool initializer: round-robin bind this worker to one physical GPU.

    Each pool process runs once at startup; ``counter`` assigns 0,1,... modulo
    ``len(gpu_devices)``. Visible-device env makes that GPU appear as index 0
    to CUDA/HIP stacks.
    """
    from multiprocessing import current_process

    if not gpu_devices:
        return
    with lock:
        idx = counter.value % len(gpu_devices)
        counter.value += 1
        phys_id = int(gpu_devices[idx])
    dev = str(phys_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = dev
    os.environ["HIP_VISIBLE_DEVICES"] = dev
    os.environ["ROCR_VISIBLE_DEVICES"] = dev
    logger.info(
        "LocalExecutor worker %s bound to physical GPU %s (round-robin)",
        current_process().name,
        phys_id,
    )


class ExecutorType(Enum):
    """Type of executor."""

    LOCAL = "local"
    CONTAINER = "container"
    RAY = "ray"


@dataclass
class ExecutorConfig:
    """
    Configuration for executors.

    Attributes:
        executor_type: Type of executor
        max_workers: Maximum number of concurrent workers
        gpu_devices: List of GPU device IDs to use
        docker_image: Docker image for container executor
        timeout_seconds: Default timeout for task execution
    """

    executor_type: ExecutorType = ExecutorType.LOCAL
    max_workers: int = 1
    gpu_devices: List[int] = field(default_factory=lambda: [0])
    docker_image: Optional[str] = None
    timeout_seconds: float = 300.0

    def __post_init__(self):
        if isinstance(self.executor_type, str):
            self.executor_type = ExecutorType(self.executor_type)


class BaseExecutor(ABC):
    """
    Abstract base class for task executors.

    Executors are responsible for:
    - Executing tasks (running evaluation modes)
    - Managing worker processes
    - Handling task lifecycle
    """

    def __init__(self, config: ExecutorConfig):
        """
        Initialize the executor.

        Args:
            config: Executor configuration
        """
        self.config = config
        self._is_running = False
        self._pending_tasks: Dict[str, Task] = {}
        self._futures: Dict[str, Future] = {}

    @abstractmethod
    def start(self) -> bool:
        """
        Start the executor.

        Returns:
            True if started successfully, False otherwise
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the executor and cleanup resources."""
        pass

    @abstractmethod
    def submit(self, task: Task) -> str:
        """
        Submit a task for execution.

        Args:
            task: Task to execute

        Returns:
            Task ID
        """
        pass

    @abstractmethod
    def execute(self, task: Task) -> TaskResult:
        """
        Execute a task synchronously.

        Args:
            task: Task to execute

        Returns:
            TaskResult with execution results
        """
        pass

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Get the status of a task.

        Args:
            task_id: Task ID

        Returns:
            Task status or None if not found
        """
        if task_id in self._pending_tasks:
            return self._pending_tasks[task_id].status
        return None

    def is_running(self) -> bool:
        """Check if executor is running."""
        return self._is_running


class LocalExecutor(BaseExecutor):
    """
    Local executor using multiprocessing.

    Executes tasks locally using a ProcessPoolExecutor for
    parallel execution of multiple tasks.

    When ``config.gpu_devices`` is non-empty, each pool worker sets
    ``CUDA_VISIBLE_DEVICES`` / ``HIP_VISIBLE_DEVICES`` / ``ROCR_VISIBLE_DEVICES``
    at process start using round-robin over that list (worker 0 → index 0, …).
    Synchronous ``execute()`` still runs in the parent process and does not apply
    this binding.
    """

    def __init__(self, config: ExecutorConfig):
        """
        Initialize the local executor.

        Args:
            config: Executor configuration
        """
        super().__init__(config)
        self._pool: Optional[ProcessPoolExecutor] = None
        self._available_gpus: List[int] = []
        self._gpu_rr_counter: Optional[Any] = None
        self._gpu_rr_lock: Optional[Any] = None

    def start(self) -> bool:
        """
        Start the executor.

        Verifies GPU availability and creates the process pool.

        Returns:
            True if started successfully, False otherwise
        """
        if self._is_running:
            logger.warning("Executor is already running")
            return True

        # Verify GPU availability
        _ = self._check_gpu_availability()

        # Create process pool (optional per-worker GPU binding via initializer)
        try:
            gpu_tuple: Tuple[int, ...] = tuple()
            if self.config.gpu_devices:
                gpu_tuple = tuple(int(x) for x in self.config.gpu_devices)
            pool_kw: Dict[str, Any] = {"max_workers": self.config.max_workers}
            if gpu_tuple:
                self._gpu_rr_counter = multiprocessing.Value("i", 0)
                self._gpu_rr_lock = multiprocessing.Lock()
                pool_kw["initializer"] = _local_pool_worker_init
                pool_kw["initargs"] = (
                    self._gpu_rr_counter,
                    self._gpu_rr_lock,
                    gpu_tuple,
                )
                logger.info(
                    "LocalExecutor GPU round-robin over %s (%s worker(s))",
                    list(gpu_tuple),
                    self.config.max_workers,
                )
            self._pool = ProcessPoolExecutor(**pool_kw)
            if (
                self.config.max_workers > 1
                and len(gpu_tuple) == 1
                and len(self._available_gpus) > 1
            ):
                logger.info(
                    "Parallel local runs all use gpu_devices=%s; set scheduler.gpu_devices "
                    "to e.g. %s to spread workers across GPUs.",
                    list(gpu_tuple),
                    self._available_gpus[: self.config.max_workers],
                )
            self._is_running = True
            logger.info(f"LocalExecutor started with {self.config.max_workers} workers")
            return True
        except Exception as e:
            logger.error(f"Failed to start executor: {e}")
            return False

    def stop(self) -> None:
        """Stop the executor and cleanup resources."""
        if self._pool:
            self._pool.shutdown(wait=True)
            self._pool = None
        self._gpu_rr_counter = None
        self._gpu_rr_lock = None
        self._is_running = False
        self._pending_tasks.clear()
        self._futures.clear()
        logger.info("LocalExecutor stopped")

    def submit(self, task: Task) -> str:
        """
        Submit a task for asynchronous execution.

        Args:
            task: Task to execute

        Returns:
            Task ID
        """
        if not self._is_running:
            raise RuntimeError("Executor is not running")

        task.status = TaskStatus.RUNNING
        self._pending_tasks[task.task_id] = task

        # Submit to process pool
        assert self._pool is not None
        future = self._pool.submit(_execute_task_worker, task.to_dict())
        self._futures[task.task_id] = future

        logger.info(f"Task {task.task_id} submitted")
        return task.task_id

    def execute(self, task: Task) -> TaskResult:
        """
        Execute a task synchronously.

        Args:
            task: Task to execute

        Returns:
            TaskResult with execution results
        """
        start_time = time.time()
        task.status = TaskStatus.RUNNING

        try:
            # Execute directly in current process
            result_dict = _execute_task_worker(task.to_dict())

            # Reconstruct TaskResult
            result = TaskResult(
                task_id=result_dict["task_id"],
                status=TaskStatus(result_dict["status"]),
                results=result_dict["results"],
                errors=result_dict["errors"],
                execution_time=result_dict["execution_time"],
                metadata=result_dict.get("metadata", {}),
            )

            task.status = result.status
            return result

        except Exception as e:
            task.status = TaskStatus.FAILED
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                errors=[str(e)],
                execution_time=time.time() - start_time,
            )

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
        if task_id not in self._futures:
            raise ValueError(f"Task {task_id} not found")

        future = self._futures[task_id]
        try:
            result_dict = future.result(timeout=timeout)
            result = TaskResult(
                task_id=result_dict["task_id"],
                status=TaskStatus(result_dict["status"]),
                results=result_dict["results"],
                errors=result_dict["errors"],
                execution_time=result_dict["execution_time"],
                metadata=result_dict.get("metadata", {}),
            )

            # Update task status
            if task_id in self._pending_tasks:
                self._pending_tasks[task_id].status = result.status

            return result

        except Exception as e:
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                errors=[str(e)],
            )

    def wait_all(self, timeout: Optional[float] = None) -> List[TaskResult]:
        """
        Wait for all submitted tasks to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            List of TaskResults
        """
        results = []
        for task_id in list(self._futures.keys()):
            result = self.wait_for_task(task_id, timeout)
            results.append(result)
        return results

    def _check_gpu_availability(self) -> bool:
        """Check GPU availability. Exit with error if no GPU is found."""
        gpu_count = get_gpu_count()

        if gpu_count > 0:
            self._available_gpus = list(range(gpu_count))
            logger.info(f"Found {gpu_count} GPU(s)")
            if self.config.gpu_devices:
                bad = [
                    g
                    for g in self.config.gpu_devices
                    if int(g) < 0 or int(g) >= gpu_count
                ]
                if bad:
                    logger.warning(
                        "scheduler.gpu_devices %s are out of range for %s GPU(s) — "
                        "binding may fail at runtime",
                        bad,
                        gpu_count,
                    )
            return True
        else:
            logger.error(
                "No GPU detected. Please ensure GPU drivers (AMD ROCm or NVIDIA CUDA) are installed."
            )
            raise RuntimeError("GPU check failed: No GPU detected")


class ContainerExecutor(BaseExecutor):
    """
    Container executor using Docker.

    Executes tasks inside Docker containers for isolation
    and reproducibility.
    """

    def __init__(self, config: ExecutorConfig):
        """
        Initialize the container executor.

        Args:
            config: Executor configuration (must include docker_image)
        """
        super().__init__(config)
        if not config.docker_image:
            raise ValueError("docker_image is required for ContainerExecutor")
        self._docker_image: str = config.docker_image
        self._container_ids: Dict[str, str] = {}

    def start(self) -> bool:
        """
        Start the executor.

        Verifies Docker availability and pulls the image.

        Returns:
            True if started successfully, False otherwise
        """
        if self._is_running:
            logger.warning("Executor is already running")
            return True

        # Check Docker availability
        if not self._check_docker_availability():
            logger.error("Docker is not available")
            return False

        # Pull the image
        if not self._pull_image():
            logger.error(f"Failed to pull image: {self._docker_image}")
            return False

        self._is_running = True
        logger.info(f"ContainerExecutor started with image: {self._docker_image}")
        return True

    def stop(self) -> None:
        """Stop the executor and cleanup containers."""
        # Stop all running containers
        for task_id, container_id in self._container_ids.items():
            try:
                subprocess.run(
                    ["docker", "stop", container_id], capture_output=True, timeout=30
                )
                subprocess.run(
                    ["docker", "rm", container_id], capture_output=True, timeout=30
                )
            except Exception as e:
                logger.warning(f"Failed to stop container {container_id}: {e}")

        self._container_ids.clear()
        self._pending_tasks.clear()
        self._is_running = False
        logger.info("ContainerExecutor stopped")

    def submit(self, task: Task) -> str:
        """
        Submit a task for execution in a container.

        Args:
            task: Task to execute

        Returns:
            Task ID
        """
        if not self._is_running:
            raise RuntimeError("Executor is not running")

        task.status = TaskStatus.RUNNING
        self._pending_tasks[task.task_id] = task

        # Start container in background
        # Note: This is a simplified implementation
        # A real implementation would use docker-py or similar
        logger.info(f"Task {task.task_id} submitted to container")
        return task.task_id

    def execute(self, task: Task) -> TaskResult:
        """
        Execute a task synchronously in a container.

        Args:
            task: Task to execute

        Returns:
            TaskResult with execution results
        """
        start_time = time.time()
        task.status = TaskStatus.RUNNING
        errors = []

        try:
            # Build docker run command
            gpu_args = self._build_gpu_args()
            container_name = f"magpie-{task.task_id}"

            # For now, execute locally but in container
            # This is a simplified implementation
            # A full implementation would:
            # 1. Mount necessary files into container
            # 2. Run the evaluation inside container
            # 3. Collect results

            cmd = [
                "docker",
                "run",
                "--rm",
                "--name",
                container_name,
                *gpu_args,
                self._docker_image,
                "python",
                "-c",
                f"print('Task {task.task_id} executed')",
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.config.timeout_seconds
            )

            if result.returncode != 0:
                errors.append(f"Container execution failed: {result.stderr}")
                status = TaskStatus.FAILED
            else:
                status = TaskStatus.COMPLETED

            # Note: In a real implementation, we would parse the actual
            # evaluation results from the container output

            task.status = status
            return TaskResult(
                task_id=task.task_id,
                status=status,
                results=None,  # Would contain actual results
                errors=errors,
                execution_time=time.time() - start_time,
            )

        except subprocess.TimeoutExpired:
            task.status = TaskStatus.FAILED
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                errors=["Container execution timed out"],
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            task.status = TaskStatus.FAILED
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                errors=[str(e)],
                execution_time=time.time() - start_time,
            )

    def _check_docker_availability(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "version"], capture_output=True, timeout=10
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _pull_image(self) -> bool:
        """Pull the Docker image."""
        try:
            result = subprocess.run(
                ["docker", "pull", self._docker_image], capture_output=True, timeout=300
            )
            return result.returncode == 0
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    def _build_gpu_args(self) -> List[str]:
        """Build GPU arguments for docker run."""
        if not self.config.gpu_devices:
            return []

        # Use NVIDIA Container Toolkit
        gpu_ids = ",".join(str(g) for g in self.config.gpu_devices)
        return ["--gpus", f'"device={gpu_ids}"']


def create_executor(config: ExecutorConfig, **kwargs) -> BaseExecutor:
    """
    Factory function to create an executor.

    Args:
        config: Executor configuration
        **kwargs: Extra arguments forwarded to specific executor constructors.
            For RAY executor, pass ``ray_config`` (a RayConfig instance).

    Returns:
        Appropriate executor instance
    """
    if config.executor_type == ExecutorType.LOCAL:
        return LocalExecutor(config)
    elif config.executor_type == ExecutorType.CONTAINER:
        return ContainerExecutor(config)
    elif config.executor_type == ExecutorType.RAY:
        from .ray_executor import RayJobExecutor
        ray_config = kwargs.get("ray_config")
        if ray_config is None:
            raise ValueError("ray_config is required for RayJobExecutor")
        return RayJobExecutor(config, ray_config=ray_config)
    else:
        raise ValueError(f"Unknown executor type: {config.executor_type}")


def _execute_task_worker(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function for executing tasks in separate processes.

    This function is pickled and sent to worker processes.

    Args:
        task_dict: Serialized task data

    Returns:
        Serialized TaskResult
    """
    import time
    from .task import TaskResult, TaskStatus
    from ..modes import AnalyzeMode, CompareMode
    from ..modes.analyze_eval.analyzer import AnalyzeConfig
    from ..modes.compare_eval.comparator import CompareConfig

    start_time = time.time()
    task_id = task_dict["task_id"]
    errors = []
    results: Any = None

    try:
        # Reconstruct kernel configs
        kernel_configs = []
        for cfg_data in task_dict["kernel_configs"]:
            if isinstance(cfg_data, dict):
                kernel_configs.append(KernelEvalConfig.from_dict(cfg_data))
            elif isinstance(cfg_data, KernelEvalConfig):
                kernel_configs.append(cfg_data)
            else:
                # Skip invalid config data
                logger.warning(f"Skipping invalid kernel config: {type(cfg_data)}")
                continue

        # Get mode config
        mode_cfg = task_dict["mode_config"]
        mode_type = ModeType(mode_cfg["mode_type"])

        if mode_type == ModeType.ANALYZE:
            # Get kernel type from first config
            kernel_type = (
                kernel_configs[0].kernel_type if kernel_configs else KernelType.HIP
            )

            # Create analyzer
            analyze_config = AnalyzeConfig(
                kernel_type=kernel_type,
                enable_default_compile=mode_cfg.get("enable_default_compile", False),
                check_performance=mode_cfg.get("check_performance", True),
                timeout_seconds=mode_cfg.get("timeout_seconds", 300.0),
                profiler_args=mode_cfg.get("profiler_args", []),
                rocprof_config=mode_cfg.get("rocprof_config", {}),
                ncu_config=mode_cfg.get("ncu_config", {}),
                metrix_config=mode_cfg.get("metrix_config", {}),
                correctness_config=mode_cfg.get("correctness_config", {}),
                latency_config=mode_cfg.get("latency_config", {}),
                gpu_arch=mode_cfg.get("gpu_arch", None),
            )
            analyzer = AnalyzeMode(analyze_config)

            # Execute analysis
            analysis_results: List[Any] = []
            for kernel_cfg in kernel_configs:
                result = analyzer.analyze(kernel_cfg)
                analysis_results.append(result)
            results = analysis_results

        elif mode_type == ModeType.COMPARE:
            # Create comparator
            compare_cfg = mode_cfg.get("compare_config", {})
            compare_config = CompareConfig(
                baseline_index=mode_cfg.get("baseline_index", 0),
                enable_default_compile=mode_cfg.get("enable_default_compile", False),
                check_performance=mode_cfg.get("check_performance", True),
                timeout_seconds=mode_cfg.get("timeout_seconds", 300.0),
                profiler_args=mode_cfg.get("profiler_args", []),
                rocprof_config=mode_cfg.get("rocprof_config", {}),
                ncu_config=mode_cfg.get("ncu_config", {}),
                metrix_config=mode_cfg.get("metrix_config", {}),
                correctness_config=mode_cfg.get("correctness_config", {}),
                latency_config=mode_cfg.get("latency_config", {}),
                gpu_arch=mode_cfg.get("gpu_arch", None),
                winner_strategy=compare_cfg.get("winner_strategy", "perf_score"),
                perf_weights_rocprof=compare_cfg.get("perf_weights_rocprof", {}),
                perf_weights_ncu=compare_cfg.get("perf_weights_ncu", {}),
                perf_weights_metrix=compare_cfg.get("perf_weights_metrix", {}),
            )
            comparator = CompareMode(compare_config)

            # Execute comparison
            results = comparator.compare(kernel_configs)

        elif mode_type == ModeType.BENCHMARK:
            # Import benchmark mode
            from ..modes.benchmark import BenchmarkMode, BenchmarkConfig
            
            # Get benchmark config
            bench_cfg = mode_cfg.get("benchmark_config", {})
            
            # Create benchmark config
            benchmark_config = BenchmarkConfig.from_dict(bench_cfg)
            
            # Override GPU arch and timeout if specified in mode config
            if mode_cfg.get("gpu_arch"):
                benchmark_config.gpu_arch = mode_cfg["gpu_arch"]
            if mode_cfg.get("timeout_seconds"):
                benchmark_config.timeout_seconds = mode_cfg["timeout_seconds"]
            
            # Create and run benchmarker
            benchmarker = BenchmarkMode(benchmark_config)
            benchmark_result = benchmarker.run(task_id=task_id)
            
            # Convert to dict for serialization
            results = benchmark_result.to_dict()

        status = TaskStatus.COMPLETED

    except Exception as e:
        errors.append(str(e))
        status = TaskStatus.FAILED
        logger.exception(f"Task {task_id} failed with error: {e}")

    execution_time = time.time() - start_time

    # Return serializable result
    task_result = TaskResult(
        task_id=task_id,
        status=status,
        results=results,
        errors=errors,
        execution_time=execution_time,
    )
    return task_result.to_dict()
