###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Task functions executed on Ray GPU workers via ``@ray.remote``.

Dispatched by ``RayJobExecutor._submit_ray_task()`` in ``ray_executor.py``.
Supports benchmark, analyze, and compare modes.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict

from Magpie.modes.benchmark.config import DEFAULT_SHARED_STORAGE_PATH

logger = logging.getLogger("magpie.remote.tasks")

# Registry: mode_type -> runner function
_MODE_RUNNERS: Dict[str, Callable[..., dict]] = {}


def _register(mode_type: str):
    """Decorator to register a mode runner."""
    def wrapper(fn: Callable[..., dict]):
        _MODE_RUNNERS[mode_type] = fn
        return fn
    return wrapper


# ---------------------------------------------------------------------------
# Entry point (called by @ray.remote)
# ---------------------------------------------------------------------------

def run_task(config: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level dispatcher executed on a Ray GPU worker.

    Receives a plain dict, routes to the registered mode runner, and
    returns a plain dict (JSON-serialisable).
    """
    task_id = config.get("task_id", "unknown")
    mode_type = config.get("mode_type", "benchmark")
    mode_config = config.get("mode_config", {})
    ray_config = config.get("ray_config", {})

    logger.info(f"Starting remote execution: task={task_id} mode={mode_type}")
    start_time = time.time()

    try:
        _setup_env(ray_config)
        _clear_hidden_gpus()

        if mode_type == "benchmark":
            _configure_tp_isolation(mode_config, ray_config)

        runner = _MODE_RUNNERS.get(mode_type)
        if runner is None:
            result = {"error": f"Unsupported mode_type: {mode_type}"}
        else:
            result = runner(mode_config, ray_config, task_id)

        result["execution_time"] = time.time() - start_time

    except Exception as e:
        logger.exception(f"Remote execution failed: {e}")
        result = {
            "task_id": task_id,
            "status": "failed",
            "error": str(e),
            "execution_time": time.time() - start_time,
        }

    return result


# ---------------------------------------------------------------------------
# Common setup
# ---------------------------------------------------------------------------

def _setup_env(ray_config: dict) -> None:
    """Set HF cache paths from shared storage config."""
    shared_storage_root = ray_config.get(
        "shared_storage_path", DEFAULT_SHARED_STORAGE_PATH
    )
    os.environ.setdefault("HF_HOME", f"{shared_storage_root}/hf_cache")
    os.environ.setdefault(
        "TRANSFORMERS_CACHE", f"{shared_storage_root}/hf_cache/hub"
    )


def _clear_hidden_gpus() -> None:
    """Remove Ray-imposed empty GPU visibility vars.

    When the outer Ray task uses ``num_gpus=0``, Ray sets
    ``HIP_VISIBLE_DEVICES=""`` etc.  Remove these so the benchmark
    subprocess can see and manage GPUs directly.
    """
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES"):
        if var in os.environ and os.environ[var] == "":
            logger.info(f"Clearing Ray-imposed {var}='' to expose GPUs")
            del os.environ[var]


# ---------------------------------------------------------------------------
# TP isolation (benchmark-only)
# ---------------------------------------------------------------------------

def _get_local_gpu_count() -> int:
    """Return the number of GPUs on the node this task is running on."""
    try:
        import ray
        current_node_id = ray.get_runtime_context().get_node_id()
        for node in ray.nodes():
            if node.get("NodeID") == current_node_id and node.get("Alive"):
                return int(node.get("Resources", {}).get("GPU", 0))
    except Exception as exc:
        logger.warning(f"Could not detect local GPU count via Ray: {exc}")
    return 8


def _configure_tp_isolation(mode_config: dict, ray_config: dict) -> None:
    """Configure TP backend for the benchmark subprocess.

    * **TP <= local GPUs** — clear ``RAY_ADDRESS`` so the framework
      uses local multiprocessing.
    * **TP > local GPUs** — keep ``RAY_ADDRESS`` and inject the Ray
      backend flag for cross-node TP.

    Framework-specific:

    * **vLLM**: ``--distributed-executor-backend mp|ray``
    * **SGLang**: ``--use-ray --nnodes N`` (PR #17684)
    """
    bench_cfg = mode_config.get("benchmark_config", {})
    envs = bench_cfg.get("envs", {})
    tp = int(envs.get("TP", 1))
    local_gpus = _get_local_gpu_count()
    framework = bench_cfg.get("framework", "vllm").lower()
    extra_key = _extra_args_key(framework)

    if tp <= local_gpus:
        if "RAY_ADDRESS" in os.environ:
            logger.info(
                f"TP={tp} fits on one node ({local_gpus} GPUs). "
                "Clearing RAY_ADDRESS to isolate subprocess."
            )
            del os.environ["RAY_ADDRESS"]
        if framework == "vllm":
            _ensure_extra_arg(envs, extra_key, "--distributed-executor-backend mp")
            _commit_envs(envs, bench_cfg, mode_config)
    else:
        num_nodes = (tp + local_gpus - 1) // local_gpus
        logger.info(
            f"TP={tp} exceeds local GPUs ({local_gpus}). "
            f"Keeping RAY_ADDRESS for cross-node TP ({num_nodes} nodes)."
        )
        if framework == "vllm":
            _ensure_extra_arg(envs, extra_key, "--distributed-executor-backend ray")
        elif framework == "sglang":
            _ensure_extra_arg(envs, extra_key, f"--use-ray --nnodes {num_nodes}")
        else:
            logger.warning(f"Unknown framework '{framework}'; skipping auto-config")
            return
        _commit_envs(envs, bench_cfg, mode_config)


def _ensure_extra_arg(envs: dict, extra_key: str, fragment: str) -> None:
    """Append *fragment* to EXTRA_*_ARGS unless the key flag already present."""
    key_flag = fragment.split()[0]
    existing = envs.get(extra_key, "")
    if key_flag in existing:
        return
    envs[extra_key] = f"{existing} {fragment}".strip()
    logger.info(f"Auto-appended '{fragment}' → {extra_key}='{envs[extra_key]}'")


def _extra_args_key(framework: str) -> str:
    return {
        "vllm": "EXTRA_VLLM_ARGS",
        "sglang": "EXTRA_SGLANG_ARGS",
    }.get(framework, f"EXTRA_{framework.upper()}_ARGS")


def _commit_envs(envs: dict, bench_cfg: dict, mode_config: dict) -> None:
    bench_cfg["envs"] = envs
    mode_config["benchmark_config"] = bench_cfg


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------

@_register("benchmark")
def _run_benchmark(mode_config: dict, ray_config: dict, task_id: str) -> dict:
    """Execute benchmark mode on a GPU worker."""
    from Magpie.modes.benchmark import BenchmarkMode, BenchmarkConfig

    bench_cfg_dict = mode_config.get("benchmark_config", {})

    if bench_cfg_dict.get("run_mode") == "ray":
        bench_cfg_dict["run_mode"] = "local"

    shared_storage_root = ray_config.get(
        "shared_storage_path", DEFAULT_SHARED_STORAGE_PATH
    )
    if not bench_cfg_dict.get("inferencex_path"):
        bench_cfg_dict["inferencex_path"] = f"{shared_storage_root}/InferenceX"
    if not bench_cfg_dict.get("hf_cache_path"):
        bench_cfg_dict["hf_cache_path"] = f"{shared_storage_root}/hf_cache"

    job_dir = str(
        Path(
            ray_config.get(
                "results_dir", f"{shared_storage_root}/results"
            )
        )
        / task_id
    )

    benchmark_config = BenchmarkConfig.from_dict(bench_cfg_dict)
    benchmarker = BenchmarkMode(benchmark_config, output_dir=job_dir)
    result = benchmarker.run(task_id=task_id)
    return result.to_dict()


@_register("analyze")
def _run_analyze(mode_config: dict, ray_config: dict, task_id: str) -> dict:
    """Execute analyze mode on a GPU worker."""
    from Magpie.modes import AnalyzeMode
    from Magpie.modes.analyze_eval.analyzer import AnalyzeConfig
    from Magpie.config import KernelEvalConfig, KernelType

    kernel_configs = [
        KernelEvalConfig.from_dict(c)
        for c in mode_config.get("kernel_configs", [])
        if isinstance(c, dict)
    ]
    kernel_type = kernel_configs[0].kernel_type if kernel_configs else KernelType.HIP

    analyze_config = AnalyzeConfig(
        kernel_type=kernel_type,
        enable_default_compile=mode_config.get("enable_default_compile", False),
        check_performance=mode_config.get("check_performance", True),
        timeout_seconds=mode_config.get("timeout_seconds", 300.0),
        profiler_args=mode_config.get("profiler_args", []),
        rocprof_config=mode_config.get("rocprof_config", {}),
        ncu_config=mode_config.get("ncu_config", {}),
        metrix_config=mode_config.get("metrix_config", {}),
        gpu_arch=mode_config.get("gpu_arch"),
    )
    analyzer = AnalyzeMode(analyze_config)
    results = [
        r.to_dict() if hasattr(r, "to_dict") else r
        for r in (analyzer.analyze(k) for k in kernel_configs)
    ]
    return {"task_id": task_id, "status": "completed", "results": results}


@_register("compare")
def _run_compare(mode_config: dict, ray_config: dict, task_id: str) -> dict:
    """Execute compare mode on a GPU worker."""
    from Magpie.modes import CompareMode
    from Magpie.modes.compare_eval.comparator import CompareConfig
    from Magpie.config import KernelEvalConfig

    kernel_configs = [
        KernelEvalConfig.from_dict(c)
        for c in mode_config.get("kernel_configs", [])
        if isinstance(c, dict)
    ]
    compare_cfg = mode_config.get("compare_config", {})
    compare_config = CompareConfig(
        baseline_index=mode_config.get("baseline_index", 0),
        enable_default_compile=mode_config.get("enable_default_compile", False),
        check_performance=mode_config.get("check_performance", True),
        timeout_seconds=mode_config.get("timeout_seconds", 300.0),
        profiler_args=mode_config.get("profiler_args", []),
        rocprof_config=mode_config.get("rocprof_config", {}),
        ncu_config=mode_config.get("ncu_config", {}),
        metrix_config=mode_config.get("metrix_config", {}),
        gpu_arch=mode_config.get("gpu_arch"),
        winner_strategy=compare_cfg.get("winner_strategy", "perf_score"),
        perf_weights_rocprof=compare_cfg.get("perf_weights_rocprof", {}),
        perf_weights_ncu=compare_cfg.get("perf_weights_ncu", {}),
        perf_weights_metrix=compare_cfg.get("perf_weights_metrix", {}),
    )
    comparator = CompareMode(compare_config)
    result = comparator.compare(kernel_configs)
    result_dict = result.to_dict() if hasattr(result, "to_dict") else result
    return {"task_id": task_id, "status": "completed", "results": result_dict}
