###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Ray executor for remote benchmark / analyze / compare execution.

Dispatches tasks to GPU workers via ``ray.init()`` + ``@ray.remote()``.
Only requires the Ray GCS (port 6379) or Ray Client (port 10001) —
**no Dashboard / Job Submission API needed**.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .executor import BaseExecutor, EndpointClientCleanupError, ExecutorConfig
from .task import Task, TaskResult, TaskStatus

logger = logging.getLogger(__name__)


def di_cleanup_timeout_seconds(benchmark_config: dict) -> int:
    """Allow sequential supervisor cleanup, rank verification and client unwind."""
    topology = benchmark_config.get("disaggregation") or {}
    instances = sum(
        max(1, int((topology.get(role) or {}).get("instances", 1)))
        for role in ("prefill", "decode")
    )
    # Inspection and stop each have a 30s deadline per supervisor; rank
    # verification adds 30s.
    # Leave another 30s for the local client and result/journal handoff.
    return 60 + 60 * instances


def di_evaluation_timeout_seconds(benchmark_config: dict, ray_config: dict) -> float:
    """Budget serving admission/startup separately from the benchmark script."""
    effective_ray = {**ray_config, **(benchmark_config.get("ray_config") or {})}
    return (
        float(benchmark_config.get("timeout_seconds", 3600))
        + float(effective_ray.get("placement_timeout_seconds", 120))
        + float((benchmark_config.get("disaggregation") or {}).get("startup_timeout", 1800))
        + di_cleanup_timeout_seconds(benchmark_config)
    )


class RayJobExecutor(BaseExecutor):
    """
    Executor that dispatches tasks to a Ray cluster via
    ``ray.init()`` and ``@ray.remote(num_gpus=...)``.

    Only the core Ray GCS / Ray Client is required (works with
    ``--minimal`` KubeRay clusters that have no Dashboard).

    Prerequisites:
        * ``ray`` must be installed (``pip install ray``)
        * The Ray cluster must be reachable at the configured address
    """

    def __init__(self, config: ExecutorConfig, ray_config: Any, *, namespace: str | None = None):
        super().__init__(config)
        self._ray_config = ray_config
        self._namespace = namespace
        self._ray_inited = False
        self._di_tasks = set()
        self._di_task_configs: Dict[str, dict] = {}
        self._di_cancel_requested = set()
        self.check_cancelled = lambda: None
        # task_id -> ray.ObjectRef
        self._obj_refs: Dict[str, Any] = {}
        # task_id -> cached result (filled after ray.get)
        self._results_cache: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Connect to the Ray cluster via ``ray.init()``."""
        if self._is_running:
            logger.warning("RayJobExecutor is already running")
            return True

        try:
            import ray
        except ImportError:
            logger.error("ray is not installed.  Run: pip install ray")
            return False

        addr = self._ray_config.cluster_address
        try:
            if ray.is_initialized():
                if self._namespace is not None and ray.get_runtime_context().namespace != self._namespace:
                    raise RuntimeError("existing Ray namespace differs from the DI evaluation namespace")
                logger.info("Ray is already initialised, reusing context")
            else:
                options = {"namespace": self._namespace} if self._namespace is not None else {}
                ray.init(address=addr, logging_level=logging.WARNING, **options)
            logger.info(f"Connected to Ray cluster (address={addr})")
        except Exception as e:
            logger.error(f"ray.init(address={addr!r}) failed: {e}")
            return False

        self._ray_inited = True
        self._is_running = True
        logger.info(
            "RayJobExecutor started (install_magpie=%s)",
            self._ray_config.install_magpie,
        )
        return True

    def stop(self) -> None:
        """Release resources and disconnect from Ray."""
        cleanup_errors = []
        for task_id in sorted(self._di_tasks):
            if task_id not in self._obj_refs:
                cleanup_errors.append(
                    f"{task_id}: Ray DI evaluation submission has no task reference; "
                    "cleanup requires reconciliation before serving resources can be reused"
                )
                continue
            try:
                if not self.cancel_task(task_id):
                    detail = self._results_cache.get(task_id, {}).get("error")
                    cleanup_errors.append(
                        f"{task_id}: Ray DI evaluation cleanup was not confirmed"
                        + (f": {detail}" if detail else "")
                    )
            except Exception as error:  # noqa: BLE001 - attempt every tracked DI evaluation
                cleanup_errors.append(f"{task_id}: {error}")
        if cleanup_errors:
            raise EndpointClientCleanupError("; ".join(cleanup_errors))
        self._obj_refs.clear()
        self._results_cache.clear()
        self._di_task_configs.clear()
        self._di_cancel_requested.clear()
        self._pending_tasks.clear()
        self._is_running = False

        if self._ray_inited:
            self._ray_inited = False
            try:
                import ray
                if ray.is_initialized():
                    ray.shutdown()
            except Exception as e:
                logger.debug(f"ray.shutdown() error (non-fatal): {e}")

        logger.info("RayJobExecutor stopped")

    # ------------------------------------------------------------------
    # Submit (async — returns immediately)
    # ------------------------------------------------------------------

    def submit(self, task: Task) -> str:
        """Dispatch a task to a GPU worker via ``@ray.remote``.

        Returns the Magpie task ID immediately.
        """
        if not self._is_running:
            raise RuntimeError("RayJobExecutor is not running")

        task.status = TaskStatus.RUNNING
        self._pending_tasks[task.task_id] = task

        obj_ref = self._submit_ray_task(task)
        self._obj_refs[task.task_id] = obj_ref

        logger.info(f"Task {task.task_id} dispatched to Ray worker")
        return task.task_id

    # ------------------------------------------------------------------
    # Execute (synchronous — blocks until done)
    # ------------------------------------------------------------------

    def execute(self, task: Task) -> TaskResult:
        """Submit and block until the Ray task completes."""
        timeout = self.config.timeout_seconds
        benchmark = getattr(task.mode_config, "benchmark_config", None)
        if (task.mode_config.mode_type.value == "benchmark"
                and isinstance(benchmark, dict)
                and benchmark.get("disaggregation") is not None):
            timeout = di_evaluation_timeout_seconds(benchmark, self._ray_config.to_dict())
        self.submit(task)
        return self.wait_for_task(task.task_id, timeout=timeout)

    # ------------------------------------------------------------------
    # Waiting / polling
    # ------------------------------------------------------------------

    def wait_for_task(
        self, task_id: str, timeout: Optional[float] = None
    ) -> TaskResult:
        """Block until the remote task finishes or *timeout* elapses."""
        import ray

        if task_id not in self._obj_refs:
            raise ValueError(f"Task {task_id} not found")

        obj_ref = self._obj_refs[task_id]
        di_evaluation = task_id in self._di_tasks or task_id in self._di_task_configs

        try:
            if di_evaluation:
                import time
                deadline = None if timeout is None else time.monotonic() + timeout
                # Always poll once, including timeout=0. A missing deadline means
                # unbounded waiting, with bounded polls so cancellation stays live.
                while True:
                    self.check_cancelled()
                    remaining = None if deadline is None else max(0, deadline - time.monotonic())
                    ready, _ = ray.wait(
                        [obj_ref], num_returns=1,
                        timeout=1 if remaining is None else min(1, remaining),
                    )
                    if ready or (deadline is not None and time.monotonic() >= deadline):
                        break
            else:
                ready, _ = ray.wait([obj_ref], num_returns=1, timeout=timeout)
            if not ready:
                if task_id in self._di_tasks and not self.cancel_task(task_id):
                    raise EndpointClientCleanupError("timed-out DI evaluation cleanup was not confirmed")
                logger.warning(f"Timeout waiting for task {task_id}")
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    errors=[f"Timeout after {timeout}s"],
                )

            result_dict: dict = ray.get(obj_ref)
            if di_evaluation and result_dict.get("di_cleanup_unconfirmed"):
                raise EndpointClientCleanupError(result_dict.get("error") or "remote DI evaluation cleanup was not confirmed")
            self._results_cache[task_id] = result_dict

            magpie_status = TaskStatus.COMPLETED
            errors: List[str] = []
            if (result_dict.get("status") == "failed" or "error" in result_dict
                    or (di_evaluation and result_dict.get("success") is False)):
                magpie_status = TaskStatus.FAILED
                if di_evaluation:
                    errors.extend(result_dict.get("errors") or [])
                if result_dict.get("error"):
                    errors.append(result_dict["error"])

        except ray.exceptions.TaskCancelledError:
            magpie_status = TaskStatus.CANCELLED
            result_dict = {}
            errors = ["Task was cancelled"]
        except EndpointClientCleanupError:
            raise
        except Exception as e:
            logger.exception(f"ray.get failed for task {task_id}: {e}")
            magpie_status = TaskStatus.FAILED
            result_dict = {}
            errors = [str(e)]

        if task_id in self._pending_tasks:
            self._pending_tasks[task_id].status = magpie_status
        if magpie_status == TaskStatus.COMPLETED:
            self._di_tasks.discard(task_id)

        return TaskResult(
            task_id=task_id,
            status=magpie_status,
            results=result_dict if result_dict else None,
            errors=errors,
            execution_time=result_dict.get("execution_time", 0.0) if result_dict else 0.0,
            metadata={"task_id": task_id},
        )

    def wait_all(self, timeout: Optional[float] = None) -> List[TaskResult]:
        """Wait for all submitted tasks."""
        results = []
        for task_id in list(self._obj_refs.keys()):
            results.append(self.wait_for_task(task_id, timeout))
        return results

    # ------------------------------------------------------------------
    # Job management helpers (used by MCP tools)
    # ------------------------------------------------------------------

    def get_task_status_ray(self, task_id: str) -> str:
        """Check if a remote task is still running."""
        import ray

        di_evaluation = task_id in self._di_tasks or task_id in self._di_task_configs
        if task_id in self._results_cache:
            result = self._results_cache[task_id]
            if di_evaluation and (
                result.get("status") == "failed" or "error" in result
                or result.get("success") is False
                or result.get("di_cleanup_unconfirmed")
            ):
                return "FAILED"
            return "SUCCEEDED"

        obj_ref = self._obj_refs.get(task_id)
        if obj_ref is None:
            return "UNKNOWN"

        ready, _ = ray.wait([obj_ref], num_returns=1, timeout=0)
        if ready:
            try:
                result = ray.get(obj_ref)
                self._results_cache[task_id] = result
                if (result.get("status") == "failed" or "error" in result
                        or (di_evaluation and (
                            result.get("success") is False
                            or result.get("di_cleanup_unconfirmed")
                        ))):
                    return "FAILED"
                return "SUCCEEDED"
            except Exception:
                return "FAILED"
        return "RUNNING"

    def get_task_result(self, task_id: str) -> Optional[dict]:
        """Return the cached result dict for a completed task."""
        return self._results_cache.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running Ray task."""
        import ray

        obj_ref = self._obj_refs.get(task_id)
        if obj_ref is None:
            return False
        try:
            di_evaluation = (
                task_id in self._di_tasks
                or task_id in self._di_task_configs
                or task_id in self._di_cancel_requested
            )
            if di_evaluation and task_id not in self._di_tasks:
                return True  # This DI evaluation already confirmed cleanup.
            if not di_evaluation:
                ray.cancel(obj_ref, force=True)
            elif task_id not in self._di_cancel_requested:
                # A repeated Ray cancellation can interrupt the worker's finally
                # cleanup. Delivery uncertainty is not permission to resend it.
                self._di_cancel_requested.add(task_id)
                # Child actor RPCs must remain runnable to finish serving cleanup.
                ray.cancel(obj_ref, force=False, recursive=False)
            if di_evaluation:
                budget = di_cleanup_timeout_seconds(self._di_task_configs.get(task_id, {}))
                ready, _ = ray.wait([obj_ref], timeout=budget)
                if not ready:
                    raise RuntimeError("DI evaluation did not stop; do not reuse serving resources")
                try:
                    result = ray.get(obj_ref)
                except ray.exceptions.TaskCancelledError:
                    pass
                else:
                    if not isinstance(result, dict) or result.get("di_cleanup_unconfirmed"):
                        detail = ""
                        if isinstance(result, dict):
                            self._results_cache[task_id] = result
                            detail = result.get("error", "")
                        raise EndpointClientCleanupError(
                            "remote DI evaluation process cleanup was not confirmed"
                            + (f": {detail}" if detail else "")
                        )
                self._di_tasks.discard(task_id)
            logger.info(f"Cancelled task {task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False

    def list_tasks(self) -> Dict[str, str]:
        """Return a dict of ``{task_id: status}`` for all tracked tasks."""
        out: Dict[str, str] = {}
        for tid in self._obj_refs:
            out[tid] = self.get_task_status_ray(tid)
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_magpie_root() -> str:
        """Auto-detect the Magpie project root."""
        pkg_dir = Path(__file__).resolve().parent.parent  # Magpie/core -> Magpie
        for candidate in [pkg_dir.parent, pkg_dir]:
            if (candidate / "pyproject.toml").exists() or (candidate / "setup.py").exists():
                return str(candidate)
        return str(pkg_dir.parent)

    @staticmethod
    def _collect_pip_packages(rc: Any) -> List[str]:
        """Build the full pip package list for ``runtime_env``."""
        pkgs: List[str] = list(rc.pip_packages)
        if rc.install_magpie:
            magpie_root = rc.magpie_install_path or RayJobExecutor._find_magpie_root()
            req_file = Path(magpie_root) / "requirements.txt"
            if req_file.exists():
                with open(req_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            pkgs.append(line)
            pkgs.append(magpie_root)
        return pkgs

    def _build_runtime_env(self, ray_config: Any = None) -> Dict[str, Any]:
        """Construct the ``runtime_env`` dict for the remote function."""
        rc = self._ray_config if ray_config is None else ray_config

        env_vars: Dict[str, str] = {
            "HF_HOME": rc.hf_cache_dir,
            "TRANSFORMERS_CACHE": f"{rc.hf_cache_dir}/hub",
        }
        if rc.multi_node:
            env_vars["RAY_ADDRESS"] = "auto"
            env_vars["MAGPIE_TOTAL_GPUS"] = str(rc.total_num_gpus)
        env_vars.update(rc.env_vars)

        runtime_env: Dict[str, Any] = {"env_vars": env_vars}

        pip_pkgs = self._collect_pip_packages(rc)
        if pip_pkgs:
            runtime_env["pip"] = pip_pkgs

        return runtime_env

    @staticmethod
    def _find_gpu_node() -> Optional[str]:
        """Return the Ray node ID of a GPU-equipped **worker** node.

        Prefers non-head nodes so that heavy benchmark workloads do not
        destabilise the cluster head.  Falls back to the head node if no
        dedicated GPU worker is available.
        """
        import ray
        head_gpu_node: Optional[str] = None
        for node in ray.nodes():
            if not node.get("Alive"):
                continue
            if node.get("Resources", {}).get("GPU", 0) <= 0:
                continue
            if "node:__internal_head__" in node.get("Resources", {}):
                head_gpu_node = node["NodeID"]
            else:
                return node["NodeID"]
        return head_gpu_node

    def _submit_ray_task(self, task: Task) -> Any:
        """Build a payload, wrap it in a ``@ray.remote`` call, and dispatch.

        The outer task claims **no GPUs** (``num_gpus=0``) so that
        frameworks like vLLM can allocate GPU resources internally via
        Ray for tensor parallelism.  A ``NodeAffinitySchedulingStrategy``
        ensures the task lands on a node that actually has GPUs.
        """
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        from Magpie.remote.tasks import run_task

        rc = self._ray_config

        merged_ray: Dict[str, Any] = dict(rc.to_dict())
        bc = getattr(task.mode_config, "benchmark_config", None)
        if isinstance(bc, dict):
            br = bc.get("ray_config")
            if isinstance(br, dict):
                merged_ray.update(br)

        job_payload = {
            "task_id": task.task_id,
            "mode_type": task.mode_config.mode_type.value,
            "mode_config": task.to_dict()["mode_config"],
            "ray_config": merged_ray,
        }

        di_evaluation = (
            job_payload["mode_type"] == "benchmark"
            and isinstance(bc, dict)
            and bc.get("disaggregation") is not None
        )
        if di_evaluation:
            from Magpie.modes.benchmark.config import RayConfig

            requested_namespace = merged_ray.get("namespace")
            if (
                requested_namespace is not None
                and requested_namespace != ray.get_runtime_context().namespace
            ):
                raise ValueError(
                    "DI evaluation namespace differs from the connected Ray namespace; "
                    "connect an executor to the requested namespace before submitting"
                )
            # A reused executor may have different defaults from this task.
            # Match resource admission/runtime setup to the merged worker payload.
            rc = RayConfig.from_dict(merged_ray)
            runtime_env = self._build_runtime_env(rc)
        else:
            runtime_env = self._build_runtime_env()
        if di_evaluation:
            for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                if os.environ.get(name):
                    runtime_env["env_vars"].setdefault(name, os.environ[name])

        gpu_node_id = self._find_gpu_node()
        if gpu_node_id is None:
            raise RuntimeError("No GPU node found in the Ray cluster")

        options = {
            "num_gpus": 0,
            "num_cpus": rc.entrypoint_num_cpus,
            "runtime_env": runtime_env,
            "scheduling_strategy": NodeAffinitySchedulingStrategy(
                node_id=gpu_node_id,
                soft=True,
            ),
        }
        if di_evaluation:
            # One evaluation owns a serving lifecycle; Ray must not retry it.
            options["max_retries"] = 0
        remote_fn = ray.remote(**options)(run_task)
        if di_evaluation:
            # Once RPC delivery starts, a transport error cannot prove that no
            # remote evaluation (and therefore no serving session) exists.
            self._di_tasks.add(task.task_id)
            self._di_task_configs[task.task_id] = bc
            try:
                obj_ref = remote_fn.remote(job_payload)
            except BaseException as error:
                raise EndpointClientCleanupError(
                    "Ray DI evaluation submission is unconfirmed; reconcile its "
                    "task and serving resources before reuse"
                ) from error
        else:
            obj_ref = remote_fn.remote(job_payload)
        logger.info(
            f"Dispatched task {task.task_id} to GPU node {gpu_node_id[:12]}... "
            f"(num_gpus=0 [managed by framework], "
            f"num_cpus={rc.entrypoint_num_cpus})"
        )
        return obj_ref
