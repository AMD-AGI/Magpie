#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Magpie MCP Server

Single MCP server that exposes multiple GPU kernel evaluation tools.
Tools are automatically discovered by MCP clients.

Kernel Tools:
  - hardware_spec: Get GPU hardware specifications
  - analyze: Analyze kernel correctness and performance
  - compare: Compare multiple kernels
  - configure_gpu: Configure GPU power/frequency settings
  - discover_kernels: Discover analyzable kernels in a project
  - suggest_optimizations: Get optimization suggestions based on analysis results
  - create_kernel_config: Generate kernel config YAML for CLI use

Benchmark Tools:
  - benchmark: Run vLLM/SGLang framework benchmark (Docker, local, or Ray)
  - gap_analysis: Run standalone gap analysis on existing torch profiler traces
  - list_benchmark_images: List available Docker images per framework/arch
  - list_benchmark_results: List previous benchmark workspaces and summaries
  - get_benchmark_result: Read detailed results from a specific benchmark run
  - compare_benchmark_reports: Compare TraceLens reports across benchmark runs

Ray Remote Execution Tools:
  - ray_task_status: Check status of a Ray-dispatched task
  - ray_task_result: Retrieve result from a completed Ray task
  - ray_task_cancel: Cancel a running Ray task
  - ray_task_list: List all tracked Ray tasks
  - benchmark_batch: Submit multiple Ray benchmarks (sequential by default; optional async submit)

Usage:
    python -m Magpie.mcp
"""

import asyncio
import json
import logging
import os
import yaml  # type: ignore[import-untyped]
from pathlib import Path
from typing import List, Dict, Any, TYPE_CHECKING, Optional, Tuple

from ..modes.benchmark.config import DEFAULT_SHARED_STORAGE_PATH

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from .discovery import discover_project_kernels

# Initialize MCP server
mcp = FastMCP(
    "magpie",
    host=os.getenv("MAGPIE_HOST", "0.0.0.0"),
    port=int(os.getenv("MAGPIE_PORT", "8000")),
)


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from ..core import SchedulerConfig


def _load_framework_config() -> Dict[str, Any]:
    """
    Load framework configuration from config.yaml.

    Searches for config.yaml in:
    1. Current directory
    2. Magpie package directory

    Returns:
        Configuration dictionary
    """
    # Try current directory first
    config_paths = [
        Path.cwd() / "config.yaml",
        Path.cwd() / "Magpie" / "config.yaml",
        Path(__file__).parent.parent / "config.yaml",  # Magpie/config.yaml
    ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

    return {}


def _get_scheduler_config_from_yaml(environment: str = "local") -> "SchedulerConfig":
    """
    Get scheduler configuration from framework config.yaml.

    Args:
        environment: Execution environment override

    Returns:
        SchedulerConfig with settings from config.yaml
    """
    from ..core import SchedulerConfig, EnvironmentType

    config = _load_framework_config()
    sched_cfg = config.get("scheduler", {})

    # Get settings from config, with defaults
    max_workers = sched_cfg.get("max_workers", 1)
    docker_image = sched_cfg.get("docker_image", None)

    # Environment can be overridden by parameter
    env_type = EnvironmentType(environment.lower())

    return SchedulerConfig(
        environment_type=env_type,
        max_workers=max_workers,
        docker_image=docker_image,
    )


def _get_correctness_settings_from_yaml() -> Dict[str, Any]:
    """Read correctness settings from framework config.yaml."""
    from ..main import _get_correctness_config

    return _get_correctness_config(_load_framework_config())


def _get_perf_settings_from_yaml() -> Dict[str, Any]:
    """Read performance profiler settings from framework config.yaml.

    Returns dict with timeout_seconds, profiler_args, rocprof_config,
    ncu_config, and metrix_config.
    Both rocprof and ncu configs are always populated so that cross-platform
    kernels (Triton) get the correct settings regardless of which GPU is used.
    """
    config = _load_framework_config()
    perf_cfg = config.get("performance", {})

    backend_str = perf_cfg.get("backend")
    rocprof_cfg = perf_cfg.get("rocprof_compute", {})
    ncu_cfg = perf_cfg.get("ncu", {})
    mtx_cfg = perf_cfg.get("metrix", {})

    return {
        "timeout_seconds": perf_cfg.get("timeout_seconds", 300.0),
        "profiler_args": ncu_cfg.get("args", []),
        "rocprof_config": {
            "workload_dir": rocprof_cfg.get("workload_dir", "./workloads"),
            "metric_blocks": rocprof_cfg.get(
                "metric_blocks", ["1", "2", "5", "10", "11", "12", "14", "16", "17"]
            ),
            "output_format": rocprof_cfg.get("output_format", "csv"),
            "profile_args": rocprof_cfg.get("profile_args", []),
            "analyze_args": rocprof_cfg.get("analyze_args", []),
        },
        "ncu_config": {
            "args": ncu_cfg.get("args", []),
            "metrics": ncu_cfg.get("metrics", []),
        },
        "metrix_config": {
            "profile": mtx_cfg.get("profile"),
            "metrics": mtx_cfg.get("metrics", []),
            "kernel_filter": mtx_cfg.get("kernel_filter"),
            "num_replays": mtx_cfg.get("num_replays", 1),
            "timeout_seconds": mtx_cfg.get("timeout_seconds", 60),
            "extra_args": mtx_cfg.get("extra_args", []),
            "backend": backend_str,
        },
    }


# =============================================================================
# Tool 1: hardware_spec
# =============================================================================
@mcp.tool()
def hardware_spec(
    device_id: int = 0,
    include_all: bool = False,
) -> str:
    """
    Get GPU hardware specifications.

    Args:
        device_id: GPU device ID to query (default: 0)
        include_all: If True, returns info for all available GPUs

    Returns:
        JSON with GPU info: vendor, architecture, power, clocks, temperature, memory
    """
    from ..utils import GPUController, MultiGPUController, get_gpu_count

    try:
        if include_all:
            multi_controller = MultiGPUController()
            all_info = multi_controller.get_all_hardware_info()
            result = {
                "gpu_count": len(all_info),
                "gpus": {str(d): info.to_dict() for d, info in all_info.items()},
            }
        else:
            controller = GPUController(device_id=device_id)
            info = controller.get_hardware_info()
            result = {"gpu_count": get_gpu_count(), "gpu": info.to_dict()}
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Tool 2: analyze
# =============================================================================
@mcp.tool()
def analyze(
    kernel_path: str,
    testcase_command: str,
    kernel_type: str = "hip",
    working_dir: str = "",
    compile_command: str = "",
    check_performance: bool = True,
    environment: str = "local",
    performance_backend: str = "",
    correctness_backend: str = "",
    accordo_kernel_name: str = "",
    accordo_reference_binary: str = "",
    accordo_optimized_binary: str = "",
    accordo_atol: float = 1e-6,
    accordo_rtol: float = 1e-5,
    accordo_equal_nan: bool = False,
    accordo_timeout_seconds: int = 30,
) -> str:
    """
    Analyze a GPU kernel for correctness and performance.

    Args:
        kernel_path: Path to kernel source file (.hip, .cu, .py)
        testcase_command: Command to run the test case
        kernel_type: "hip", "cuda", "pytorch", or "triton"
        working_dir: Working directory (default: kernel's parent dir)
        compile_command: Custom compile command (optional)
        check_performance: Run performance profiling (default: True)
        environment: Execution environment "local" or "container"
        performance_backend: Profiling backend override: "metrix", "rocprof_compute", or "ncu" (default: auto)
        correctness_backend: Correctness backend override: "accordo" or "testcase" (default: auto from config)
        accordo_kernel_name: Kernel function name for Accordo validation (required when correctness_backend="accordo")
        accordo_reference_binary: Reference binary for Accordo comparison
        accordo_optimized_binary: Optimized binary for Accordo comparison
        accordo_atol: Absolute tolerance for Accordo np.allclose (default: 1e-6)
        accordo_rtol: Relative tolerance (default: 1e-5, torch-style)
        accordo_equal_nan: If True, NaN compares equal
        accordo_timeout_seconds: Timeout per snapshot capture in seconds (default: 30)

    Returns:
        JSON with comprehensive analysis results including:
        - compiling_state, correctness_state, performance_state
        - score (0.0 - 1.0)
        - compiling_result: compilation details (if applicable)
        - correctness_result: testcase execution results
        - performance_result: detailed performance metrics
          - summary: key GPU metrics (VALU_FLOPs, MFMA_FLOPs, utilization, etc.)
          - kernels: per-kernel statistics (dispatch count, duration)
          - workload_dir: path to raw profiler data
    """
    from ..config import KernelType, KernelEvalConfig
    from ..core import Scheduler

    try:
        # Parse kernel type
        type_map = {
            "hip": KernelType.HIP,
            "cuda": KernelType.CUDA,
            "pytorch": KernelType.PYTORCH,
            "torch": KernelType.PYTORCH,
            "triton": KernelType.TRITON,
        }
        ktype = type_map.get(kernel_type.lower(), KernelType.HIP)

        kernel_file = Path(kernel_path)
        kernel_config = KernelEvalConfig(
            kernel_id=kernel_file.stem,
            kernel_type=ktype,
            source_file_path=[str(kernel_file)],
            testcase_command=testcase_command.split(),
            compiling_command=compile_command.split() if compile_command else None,
            working_dir=working_dir or str(kernel_file.parent),
        )

        # Create scheduler with config from config.yaml
        scheduler_config = _get_scheduler_config_from_yaml(environment)
        scheduler = Scheduler(scheduler_config)

        if not scheduler.initialize():
            return json.dumps({"error": "Failed to initialize scheduler"})

        # Build kernel_config dict for output
        kernel_config_dict = {
            "kernel_id": kernel_config.kernel_id,
            "kernel_type": ktype.name,
            "source_file_path": kernel_config.source_file_path,
            "testcase_command": testcase_command,
            "working_dir": kernel_config.working_dir,
        }
        if compile_command:
            kernel_config_dict["compile_command"] = compile_command

        try:
            perf_settings = _get_perf_settings_from_yaml()
            corr_settings = _get_correctness_settings_from_yaml()

            # Apply per-call backend override
            if performance_backend:
                from ..main import _apply_perf_overrides

                perf_settings = _apply_perf_overrides(
                    perf_settings, {"backend": performance_backend}, ktype
                )

            if correctness_backend:
                from ..main import _apply_correctness_overrides

                overrides: Dict[str, Any] = {"backend": correctness_backend}
                if correctness_backend == "accordo":
                    overrides["accordo"] = {
                        "kernel_name": accordo_kernel_name or None,
                        "reference_binary": accordo_reference_binary or None,
                        "optimized_binary": accordo_optimized_binary or None,
                        "atol": accordo_atol,
                        "rtol": accordo_rtol,
                        "equal_nan": accordo_equal_nan,
                        "timeout_seconds": accordo_timeout_seconds,
                        "working_directory": working_dir or None,
                    }
                corr_settings = _apply_correctness_overrides(corr_settings, overrides)

            task_result = scheduler.run_analyze(
                kernel_configs=[kernel_config],
                check_performance=check_performance,
                timeout_seconds=perf_settings["timeout_seconds"],
                profiler_args=perf_settings["profiler_args"],
                rocprof_config=perf_settings["rocprof_config"],
                ncu_config=perf_settings["ncu_config"],
                metrix_config=perf_settings["metrix_config"],
                correctness_config=corr_settings,
            )

            if task_result.success and task_result.results:
                result = task_result.results[0]
                formatted = _format_analysis_result(result, kernel_config, ktype)
                formatted["kernel_config"] = kernel_config_dict
                return json.dumps(formatted, indent=2)
            else:
                return json.dumps(
                    {
                        "error": "Analysis failed",
                        "errors": task_result.errors,
                        "kernel_config": kernel_config_dict,
                    }
                )
        finally:
            scheduler.shutdown()

    except Exception as e:
        return json.dumps(
            {
                "error": str(e),
                "kernel_path": kernel_path,
                "kernel_config": {
                    "kernel_path": kernel_path,
                    "kernel_type": kernel_type,
                    "testcase_command": testcase_command,
                    "working_dir": working_dir or str(Path(kernel_path).parent),
                    "compile_command": compile_command or None,
                },
            }
        )


def _format_analysis_result(result: dict, kernel_config, ktype) -> dict:
    """
    Format analysis result into a comprehensive response.

    Args:
        result: Raw result from analyzer (dict or object)
        kernel_config: Kernel configuration
        ktype: Kernel type enum

    Returns:
        Formatted result dictionary
    """
    if isinstance(result, dict):
        # Result is already a dict
        formatted = {
            "kernel_id": kernel_config.kernel_id,
            "kernel_type": ktype.name,
            "compiling_state": result.get("compiling_state", "UNKNOWN"),
            "correctness_state": result.get("correctness_state", "UNKNOWN"),
            "performance_state": result.get("performance_state", "UNKNOWN"),
            "score": result.get("score", 0.0),
            "errors": result.get("errors", []),
        }

        # Add compiling result if available
        compiling_result = result.get("compiling_result")
        if compiling_result:
            formatted["compiling_result"] = _format_compiling_result(compiling_result)

        # Add correctness result if available
        correctness_result = result.get("correctness_result")
        if correctness_result:
            formatted["correctness_result"] = _format_correctness_result(
                correctness_result
            )

        # Add performance result if available
        performance_result = result.get("performance_result")
        if performance_result:
            formatted["performance_result"] = _format_performance_result(
                performance_result
            )

        return formatted
    else:
        # Result is an object (EvaluationState)
        formatted = {
            "kernel_id": kernel_config.kernel_id,
            "kernel_type": ktype.name,
            "compiling_state": result.compiling_state.name
            if hasattr(result.compiling_state, "name")
            else str(result.compiling_state),
            "correctness_state": result.correctness_state.name
            if hasattr(result.correctness_state, "name")
            else str(result.correctness_state),
            "performance_state": result.performance_state.name
            if hasattr(result.performance_state, "name")
            else str(result.performance_state),
            "score": result.score,
            "errors": result.errors or [],
        }

        if hasattr(result, "compiling_result") and result.compiling_result:
            formatted["compiling_result"] = _format_compiling_result(
                result.compiling_result
            )

        if hasattr(result, "correctness_result") and result.correctness_result:
            formatted["correctness_result"] = _format_correctness_result(
                result.correctness_result
            )

        if hasattr(result, "performance_result") and result.performance_result:
            formatted["performance_result"] = _format_performance_result(
                result.performance_result
            )

        return formatted


def _format_compiling_result(result) -> dict:
    """Format compiling result."""
    if isinstance(result, dict):
        return {
            "success": result.get("success", False),
            "compile_time_seconds": result.get("compile_time_seconds"),
            "errors": result.get("errors"),
        }
    elif hasattr(result, "to_dict"):
        return result.to_dict()
    else:
        return {
            "success": getattr(result, "success", False),
            "compile_time_seconds": getattr(result, "compile_time_seconds", None),
            "errors": getattr(result, "errors", None),
        }


def _format_correctness_result(result) -> dict:
    """Format correctness result."""
    if isinstance(result, dict):
        return {
            "success": result.get("success", False),
            "errors": result.get("errors"),
        }
    elif hasattr(result, "to_dict"):
        return result.to_dict()
    else:
        return {
            "success": getattr(result, "success", False),
            "errors": getattr(result, "errors", None),
        }


def _format_performance_result(result) -> dict:
    """
    Format performance result with summary metrics and kernel statistics.

    Returns a structured dict with:
    - success: bool
    - errors: str or None
    - workload_dir: path to raw profiler data
    - summary: key performance metrics (dict of metric_name -> {value, unit, peak, pct_of_peak})
    - kernels: list of kernel summaries (kernel_name, dispatch_count, duration_ns stats)
    """
    if isinstance(result, dict):
        return {
            "success": result.get("success", False),
            "errors": result.get("errors"),
            "workload_dir": result.get("workload_dir"),
            "summary": result.get("summary", {}),
            "kernels": result.get("kernels", []),
        }
    elif hasattr(result, "to_dict"):
        return result.to_dict()
    else:
        # Handle PerformanceResult object directly
        formatted = {
            "success": getattr(result, "success", False),
            "errors": getattr(result, "errors", None),
            "workload_dir": getattr(result, "workload_dir", None),
        }

        # Extract summary metrics
        if hasattr(result, "get_summary_metrics"):
            formatted["summary"] = result.get_summary_metrics()
        elif hasattr(result, "metrics"):
            summary = {}
            for m in result.metrics:
                summary[m.name] = {
                    "value": m.value,
                    "unit": m.unit,
                }
                if hasattr(m, "peak") and m.peak is not None:
                    summary[m.name]["peak"] = m.peak
                if hasattr(m, "pct_of_peak") and m.pct_of_peak is not None:
                    summary[m.name]["pct_of_peak"] = m.pct_of_peak
            formatted["summary"] = summary
        else:
            formatted["summary"] = {}

        # Extract kernel statistics
        if hasattr(result, "get_kernel_summary"):
            formatted["kernels"] = result.get_kernel_summary()
        else:
            formatted["kernels"] = []

        return formatted


# =============================================================================
# Tool 3: compare
# =============================================================================
@mcp.tool()
def compare(
    kernel_paths: List[str],
    testcase_commands: List[str],
    kernel_type: str = "hip",
    baseline_index: int = 0,
    check_performance: bool = True,
    environment: str = "local",
    performance_backend: str = "",
    correctness_backend: str = "",
    accordo_kernel_name: str = "",
    accordo_reference_binary: str = "",
    accordo_optimized_binary: str = "",
    accordo_atol: float = 1e-6,
    accordo_rtol: float = 1e-5,
    accordo_equal_nan: bool = False,
    accordo_timeout_seconds: int = 30,
) -> str:
    """
    Compare multiple GPU kernels for performance and correctness.

    IMPORTANT: testcase_commands is REQUIRED and must have the same length as kernel_paths.
    Each kernel needs its own test command to run.

    Args:
        kernel_paths: List of kernel source file paths (minimum 2)
        testcase_commands: List of test commands for each kernel (REQUIRED, must match kernel_paths length)
        kernel_type: "hip", "cuda", "pytorch", or "triton"
        baseline_index: Index of baseline kernel for comparison (default: 0)
        check_performance: Run performance profiling (default: True)
        environment: Execution environment "local" or "container"
        performance_backend: Profiling backend override: "metrix", "rocprof_compute", or "ncu" (default: auto)
        correctness_backend: Correctness backend override: "accordo" or "testcase" (default: auto from config)
        accordo_kernel_name: Kernel function name for Accordo validation (required when correctness_backend="accordo")
        accordo_reference_binary: Reference binary for Accordo comparison
        accordo_optimized_binary: Optimized binary for Accordo comparison
        accordo_atol: Absolute tolerance for Accordo (default: 1e-6)
        accordo_rtol: Relative tolerance (default: 1e-5, torch-style)
        accordo_equal_nan: If True, NaN compares equal
        accordo_timeout_seconds: Timeout per snapshot capture in seconds (default: 30)

    Returns:
        JSON with comparison results, ranking, and summary

    Example:
        compare(
            kernel_paths=["/path/kernel_v1.hip", "/path/kernel_v2.hip"],
            testcase_commands=["./test_v1", "./test_v2"],
            baseline_index=0
        )
    """
    from ..config import KernelType, KernelEvalConfig
    from ..core import Scheduler

    try:
        if len(kernel_paths) < 2:
            return json.dumps({"error": "Compare requires at least 2 kernels"})

        if len(kernel_paths) != len(testcase_commands):
            return json.dumps(
                {
                    "error": f"kernel_paths ({len(kernel_paths)}) and testcase_commands ({len(testcase_commands)}) must have the same length"
                }
            )

        type_map = {
            "hip": KernelType.HIP,
            "cuda": KernelType.CUDA,
            "pytorch": KernelType.PYTORCH,
            "triton": KernelType.TRITON,
        }
        ktype = type_map.get(kernel_type.lower(), KernelType.HIP)

        kernel_configs = []
        kernel_configs_dict = []  # For output
        for i, (path, cmd) in enumerate(zip(kernel_paths, testcase_commands)):
            kernel_file = Path(path)
            kernel_id = f"kernel_{i}_{kernel_file.stem}"
            kernel_configs.append(
                KernelEvalConfig(
                    kernel_id=kernel_id,
                    kernel_type=ktype,
                    source_file_path=[str(kernel_file)],
                    testcase_command=cmd.split() if cmd else None,
                    working_dir=str(kernel_file.parent),
                )
            )
            # Build config dict for output
            kernel_configs_dict.append(
                {
                    "kernel_id": kernel_id,
                    "kernel_type": ktype.name,
                    "source_file_path": str(kernel_file),
                    "testcase_command": cmd or None,
                    "working_dir": str(kernel_file.parent),
                    "is_baseline": i == baseline_index,
                }
            )

        # Create scheduler with config from config.yaml
        scheduler_config = _get_scheduler_config_from_yaml(environment)
        scheduler = Scheduler(scheduler_config)

        if not scheduler.initialize():
            return json.dumps(
                {
                    "error": "Failed to initialize scheduler",
                    "kernel_configs": kernel_configs_dict,
                }
            )

        try:
            perf_settings = _get_perf_settings_from_yaml()
            corr_settings = _get_correctness_settings_from_yaml()

            # Apply per-call backend override
            if performance_backend:
                from ..main import _apply_perf_overrides

                perf_settings = _apply_perf_overrides(
                    perf_settings, {"backend": performance_backend}, ktype
                )

            if correctness_backend:
                from ..main import _apply_correctness_overrides

                corr_overrides: Dict[str, Any] = {"backend": correctness_backend}
                if correctness_backend == "accordo":
                    first_wd = kernel_configs[0].working_dir if kernel_configs else None
                    corr_overrides["accordo"] = {
                        "kernel_name": accordo_kernel_name or None,
                        "reference_binary": accordo_reference_binary or None,
                        "optimized_binary": accordo_optimized_binary or None,
                        "atol": accordo_atol,
                        "rtol": accordo_rtol,
                        "equal_nan": accordo_equal_nan,
                        "timeout_seconds": accordo_timeout_seconds,
                        "working_directory": first_wd,
                    }
                corr_settings = _apply_correctness_overrides(
                    corr_settings, corr_overrides
                )

            task_result = scheduler.run_compare(
                kernel_configs=kernel_configs,
                baseline_index=baseline_index,
                check_performance=check_performance,
                timeout_seconds=perf_settings["timeout_seconds"],
                profiler_args=perf_settings["profiler_args"],
                rocprof_config=perf_settings["rocprof_config"],
                ncu_config=perf_settings["ncu_config"],
                metrix_config=perf_settings["metrix_config"],
                correctness_config=corr_settings,
            )

            if task_result.success and task_result.results:
                comparison = task_result.results
                if isinstance(comparison, dict):
                    comparison["kernel_configs"] = kernel_configs_dict
                    return json.dumps(comparison, indent=2)
                elif hasattr(comparison, "to_dict"):
                    result_dict = comparison.to_dict()
                    result_dict["kernel_configs"] = kernel_configs_dict
                    return json.dumps(result_dict, indent=2)
                else:
                    return json.dumps(
                        {
                            "result": str(comparison),
                            "kernel_configs": kernel_configs_dict,
                        },
                        indent=2,
                    )
            else:
                return json.dumps(
                    {
                        "error": "Comparison failed",
                        "errors": task_result.errors,
                        "kernel_configs": kernel_configs_dict,
                    }
                )
        finally:
            scheduler.shutdown()

    except Exception as e:
        return json.dumps(
            {
                "error": str(e),
                "kernel_paths": kernel_paths,
                "kernel_configs": [
                    {
                        "kernel_path": p,
                        "kernel_type": kernel_type,
                        "testcase_command": testcase_commands[i]
                        if i < len(testcase_commands)
                        else None,
                        "is_baseline": i == baseline_index,
                    }
                    for i, p in enumerate(kernel_paths)
                ],
            }
        )


# =============================================================================
# Tool 4: configure_gpu
# =============================================================================
@mcp.tool()
def configure_gpu(
    device_ids: Optional[List[int]] = None,
    power_limit_watts: Optional[int] = None,
    gpu_clock_mhz: Optional[List[int]] = None,
    mem_clock_mhz: Optional[List[int]] = None,
    reset: bool = False,
) -> str:
    """
    Configure GPU hardware settings. Requires root/sudo permissions.

    Args:
        device_ids: GPU device IDs to configure (default: all GPUs)
        power_limit_watts: Power limit in watts (e.g., 300)
        gpu_clock_mhz: GPU clock range [min, max] MHz (e.g., [1500, 1800])
        mem_clock_mhz: Memory clock range [min, max] MHz
        reset: Reset GPUs to default settings

    Returns:
        JSON with configuration results for each GPU
    """
    from ..utils import MultiGPUController, MultiGPUConfig, GPUConfig

    try:
        controller = MultiGPUController(device_ids=device_ids)

        if reset:
            results = controller.reset_all()
            return json.dumps(
                {
                    "action": "reset",
                    "results": {str(k): v for k, v in results.items()},
                },
                indent=2,
            )

        gpu_clock_tuple: Optional[Tuple[int, int]] = None
        if gpu_clock_mhz:
            if len(gpu_clock_mhz) != 2:
                raise ValueError("gpu_clock_mhz must be [min, max]")
            gpu_clock_tuple = (gpu_clock_mhz[0], gpu_clock_mhz[1])

        mem_clock_tuple: Optional[Tuple[int, int]] = None
        if mem_clock_mhz:
            if len(mem_clock_mhz) != 2:
                raise ValueError("mem_clock_mhz must be [min, max]")
            mem_clock_tuple = (mem_clock_mhz[0], mem_clock_mhz[1])

        default_config = GPUConfig(
            power_limit_watts=power_limit_watts,
            gpu_clock_mhz=gpu_clock_tuple,
            mem_clock_mhz=mem_clock_tuple,
        )

        multi_config = MultiGPUConfig(
            default_config=default_config,
            device_ids=device_ids,
            parallel=True,
        )

        results = controller.apply_config(multi_config)
        return json.dumps(
            {
                "action": "configure",
                "config": {
                    "power_limit_watts": power_limit_watts,
                    "gpu_clock_mhz": gpu_clock_mhz,
                    "mem_clock_mhz": mem_clock_mhz,
                },
                "results": {str(k): v for k, v in results.items()},
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Tool 5: discover_kernels
# =============================================================================


@mcp.tool()
def discover_kernels(
    project_path: str,
    kernel_type: str = "hip",
    include_tests: bool = True,
    include_examples: bool = True,
) -> str:
    """
    Discover analyzable GPU kernels in a project directory.

    Scans the project for kernel source files and attempts to find
    corresponding test binaries or executables.

    Args:
        project_path: Root path of the project to scan
        kernel_type: Type of kernels to find: "hip", "cuda", "triton", or "all"
        include_tests: Include test directories in search
        include_examples: Include example directories in search

    Returns:
        JSON with list of discovered kernels, each containing:
        - source_file: Path to kernel source
        - possible_binaries: List of potential test/example binaries
        - suggested_config: Suggested kernel configuration for analyze()
    """
    try:
        result = discover_project_kernels(
            project_path=project_path,
            kernel_type=kernel_type,
            include_tests=include_tests,
            include_examples=include_examples,
        )
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Tool 6: suggest_optimizations
# =============================================================================
@mcp.tool()
def suggest_optimizations(
    analysis_result: str,
) -> str:
    """
    Analyze performance results and suggest optimizations.

    Takes the JSON output from analyze() and provides actionable
    optimization suggestions based on the performance metrics.

    Args:
        analysis_result: JSON string from analyze() output

    Returns:
        JSON with optimization suggestions including:
        - bottlenecks: Identified performance bottlenecks
        - suggestions: Specific optimization recommendations
        - priority: Ranked list of optimizations by potential impact
    """
    try:
        data = json.loads(analysis_result)

        suggestions = []
        bottlenecks = []

        # Check if analysis was successful
        if data.get("performance_state") != "SUCCESS":
            return json.dumps(
                {
                    "error": "Performance analysis not available",
                    "reason": data.get("errors", ["Unknown error"]),
                }
            )

        perf = data.get("performance_result", {})
        summary = perf.get("summary", {})
        kernels = perf.get("kernels", [])

        # Analyze CU utilization
        active_cus = summary.get("Active CUs", {})
        if active_cus:
            cu_pct = active_cus.get("pct_of_peak", 0)
            if cu_pct < 50:
                bottlenecks.append(
                    {
                        "type": "low_cu_occupancy",
                        "severity": "high" if cu_pct < 25 else "medium",
                        "value": f"{cu_pct}% CU utilization",
                        "description": "Low Compute Unit occupancy indicates underutilization of GPU resources",
                    }
                )
                suggestions.append(
                    {
                        "category": "parallelism",
                        "suggestion": "Increase thread block size or launch more thread blocks",
                        "impact": "high",
                        "details": f"Current CU utilization is only {cu_pct}%. Consider increasing occupancy by adjusting block dimensions or using more registers efficiently.",
                    }
                )

        # Analyze MFMA utilization
        mfma_util = summary.get("MFMA_Util", {})
        if mfma_util:
            mfma_pct = mfma_util.get("value", 0)
            if mfma_pct < 50:
                # Check if this is a matrix-heavy workload
                mfma_f16 = summary.get("MFMA_FLOPs_F16", {}).get("value", 0)
                mfma_bf16 = summary.get("MFMA_FLOPs_BF16", {}).get("value", 0)
                if mfma_f16 > 0 or mfma_bf16 > 0:
                    bottlenecks.append(
                        {
                            "type": "low_mfma_utilization",
                            "severity": "medium",
                            "value": f"{mfma_pct}% MFMA utilization",
                            "description": "Matrix operations not fully utilizing MFMA units",
                        }
                    )
                    suggestions.append(
                        {
                            "category": "matrix_ops",
                            "suggestion": "Optimize matrix tile sizes for MFMA instructions",
                            "impact": "high",
                            "details": "Consider using tile sizes that match MFMA instruction requirements (16x16, 32x32, etc.)",
                        }
                    )

        # Analyze VALU utilization
        valu_util = summary.get("VALU_Util", {})
        if valu_util:
            valu_pct = valu_util.get("value", 0)
            if valu_pct > 50:
                # High VALU but potentially low MFMA could indicate scalar bottleneck
                mfma_pct = mfma_util.get("value", 0) if mfma_util else 0
                if mfma_pct < 10:
                    suggestions.append(
                        {
                            "category": "vectorization",
                            "suggestion": "Replace scalar operations with matrix operations where possible",
                            "impact": "medium",
                            "details": f"High VALU utilization ({valu_pct}%) with low MFMA ({mfma_pct}%) suggests potential for matrix optimization",
                        }
                    )

        # Analyze memory bandwidth
        vmem_util = summary.get("VMEM_Util", {})
        if vmem_util:
            vmem_pct = vmem_util.get("value", 0)
            if vmem_pct > 70:
                bottlenecks.append(
                    {
                        "type": "memory_bound",
                        "severity": "high",
                        "value": f"{vmem_pct}% VMEM utilization",
                        "description": "Kernel is memory bandwidth limited",
                    }
                )
                suggestions.append(
                    {
                        "category": "memory",
                        "suggestion": "Optimize memory access patterns and use LDS caching",
                        "impact": "high",
                        "details": "Consider coalescing memory accesses, using shared memory (LDS), or reducing memory traffic through computation reuse",
                    }
                )

        # Analyze kernel dispatch overhead
        if kernels:
            short_kernels = [
                k for k in kernels if k.get("duration_ns", {}).get("avg", 0) < 10000
            ]  # < 10µs
            if len(short_kernels) > len(kernels) * 0.5:
                bottlenecks.append(
                    {
                        "type": "kernel_launch_overhead",
                        "severity": "medium",
                        "value": f"{len(short_kernels)} short kernels (<10µs)",
                        "description": "Many short kernel launches can be dominated by launch overhead",
                    }
                )
                suggestions.append(
                    {
                        "category": "fusion",
                        "suggestion": "Fuse multiple short kernels into larger kernels",
                        "impact": "medium",
                        "details": "Consider kernel fusion to reduce launch overhead and improve data locality",
                    }
                )

        # Prioritize suggestions
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda x: priority_order.get(x.get("impact", "low"), 2))

        return json.dumps(
            {
                "kernel_id": data.get("kernel_id", "unknown"),
                "overall_score": data.get("score", 0),
                "bottlenecks": bottlenecks,
                "suggestions": suggestions,
                "summary": {
                    "total_bottlenecks": len(bottlenecks),
                    "total_suggestions": len(suggestions),
                    "high_impact_suggestions": len(
                        [s for s in suggestions if s.get("impact") == "high"]
                    ),
                },
            },
            indent=2,
        )

    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Tool 7: create_kernel_config
# =============================================================================
@mcp.tool()
def create_kernel_config(
    kernel_id: str,
    kernel_path: str,
    testcase_command: str,
    kernel_type: str = "hip",
    working_dir: str = "",
    compile_command: str = "",
    output_path: str = "",
) -> str:
    """
    Create a kernel configuration YAML content for use with Magpie CLI.

    This tool generates a properly formatted kernel configuration that
    can be used with 'Magpie analyze --kernel-config <file>'.

    NOTE: This tool is READ-ONLY and does not write files to disk.
    The agent should use the returned config_content directly with
    the analyze() tool, or save it manually if needed.

    Args:
        kernel_id: Unique identifier for the kernel
        kernel_path: Path to the kernel source file
        testcase_command: Command to run the test case
        kernel_type: "hip", "cuda", "pytorch", or "triton"
        working_dir: Working directory for execution
        compile_command: Custom compile command (optional)
        output_path: DEPRECATED - file saving is disabled for safety

    Returns:
        JSON with:
        - config_content: The YAML configuration content
        - usage: Example command to use the config
    """
    try:
        import yaml

        # Build kernel config
        kernel_config = {
            "kernel": {
                "id": kernel_id,
                "type": kernel_type,
                "source_files": [kernel_path],
                "testcase_command": testcase_command,
            }
        }

        if working_dir:
            kernel_config["kernel"]["working_dir"] = working_dir

        if compile_command:
            kernel_config["kernel"]["compile_command"] = compile_command

        # Generate YAML content
        yaml_content = f"# Kernel Configuration for {kernel_id}\n"
        yaml_content += "# Generated by Magpie MCP\n\n"
        yaml_content += yaml.dump(
            kernel_config, default_flow_style=False, sort_keys=False
        )

        result = {
            "config_content": yaml_content,
            "usage": "python -m Magpie analyze --kernel-config <config_file>",
        }

        # Note: File saving is disabled by default to avoid unintended side effects.
        # The agent should use the config_content directly or save it manually.
        # To enable file saving, uncomment the code below:
        #
        # if output_path:
        #     output_file = Path(output_path)
        #     if output_file.exists():
        #         result["warning"] = f"File {output_path} already exists, not overwriting"
        #     else:
        #         output_file.parent.mkdir(parents=True, exist_ok=True)
        #         output_file.write_text(yaml_content)
        #         result["saved_to"] = str(output_file)
        #         result["usage"] = f"python -m Magpie analyze --kernel-config {output_path}"

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Tool 8: benchmark
# =============================================================================
@mcp.tool()
async def benchmark(
    framework: str,
    model: str,
    precision: str = "fp8",
    run_mode: str = "docker",
    tp: int = 1,
    concurrency: int = 32,
    input_len: int = 1024,
    output_len: int = 512,
    torch_profiler: bool = True,
    system_profiler: bool = False,
    tracelens: bool = False,
    tracelens_export_format: str = "csv",
    gap_analysis_enabled: bool = False,
    gap_analysis_top_k: int = 20,
    gap_analysis_start_pct: float = 0.0,
    gap_analysis_end_pct: float = 100.0,
    gap_analysis_min_duration_us: float = 0.0,
    gap_analysis_categories: Optional[List[str]] = None,
    gap_analysis_ignore_categories: Optional[List[str]] = None,
    docker_image: Optional[str] = None,
    gpu_arch: Optional[str] = None,
    inferencex_path: str = "",
    hf_cache_path: Optional[str] = None,
    benchmark_script: Optional[str] = None,
    runner_type: Optional[str] = None,
    timeout_seconds: float = 3600.0,
    output_dir: str = "./results",
    extra_envs: Optional[Dict[str, Any]] = None,
    ray_cluster_address: str = "",
    ray_shared_storage_path: str = "",
    ray_num_gpus: int = 0,
    ray_multi_node: bool = False,
    ray_total_num_gpus: int = 8,
    ray_num_nodes: int = 1,
) -> str:
    """
    Run a framework-level LLM inference benchmark (vLLM or SGLang).

    Launches the inference server using InferenceX scripts, runs a benchmark
    client, and collects throughput/latency metrics. Optionally collects torch
    profiler traces, runs TraceLens analysis, and performs gap analysis.

    InferenceX is auto-cloned if not present.

    Supports three execution modes:
    - "docker": Run inside a Docker container (default)
    - "local": Run directly on the host
    - "ray": Submit to a remote Ray cluster (returns immediately with job_id)

    Args:
        framework: "vllm" or "sglang"
        model: HuggingFace model name (e.g., "deepseek-ai/DeepSeek-R1-0528")
        precision: Model precision - "fp8", "fp16", "bf16", or "fp4" (default: "fp8")
        run_mode: Execution mode - "docker" (default), "local", or "ray"
        tp: Tensor parallelism / number of GPUs (default: 1)
        concurrency: Request concurrency (default: 32)
        input_len: Input sequence length (default: 1024)
        output_len: Output sequence length (default: 512)
        torch_profiler: Enable PyTorch profiler traces (default: True)
        system_profiler: Enable system profiler - rocprof (AMD) or ncu (NVIDIA) (default: False)
        tracelens: Enable TraceLens trace analysis on host after benchmark (default: False)
        tracelens_export_format: TraceLens export format - "csv" or "excel" (default: "csv")
        gap_analysis_enabled: Run gap analysis on torch traces after benchmark (default: False).
            Requires torch_profiler=True.
        gap_analysis_top_k: Number of top bottleneck kernels to report (default: 20)
        gap_analysis_start_pct: Start of analysis window as % of trace (0-100, default: 0)
        gap_analysis_end_pct: End of analysis window as % of trace (0-100, default: 100)
        gap_analysis_min_duration_us: Minimum event duration filter in microseconds (default: 0)
        gap_analysis_categories: Event categories to include (e.g., ["kernel", "gpu"]).
            Default: ["kernel", "gpu"]
        gap_analysis_ignore_categories: Event categories to exclude.
            Default: ["gpu_user_annotation"]
        docker_image: Override automatic Docker image selection (optional)
        gpu_arch: Force GPU architecture, e.g. "gfx942" (auto-detected if omitted)
        inferencex_path: Path to InferenceX installation (auto-cloned if empty)
        hf_cache_path: HuggingFace cache directory (default: ~/.cache/huggingface)
        benchmark_script: Override benchmark script name (e.g., "dsr1_fp8_mi300x.sh")
        runner_type: Hardware runner type (e.g., "mi300x", "h100") - auto-detected if omitted
        timeout_seconds: Benchmark timeout in seconds (default: 3600)
        output_dir: Base directory for results (default: "./results")
        extra_envs: Additional environment variables passed to the benchmark
        ray_cluster_address: Ray cluster address for run_mode="ray".
            "auto" (default) for head-node usage, or "ray://<ip>:10001"
            for remote Ray Client connection.
        ray_shared_storage_path: Shared filesystem root on worker nodes (HF cache,
            InferenceX, results; same mount on driver + workers). Default is
            ``DEFAULT_SHARED_STORAGE_PATH`` from ``Magpie.modes.benchmark.config``.
        ray_num_gpus: Number of GPUs to request for the Ray job entrypoint (default: 0)
        ray_multi_node: Whether the benchmark needs multiple nodes (default: False)
        ray_total_num_gpus: Total GPUs across nodes for multi-node (default: 8)
        ray_num_nodes: Number of nodes for multi-node (default: 1)

    Returns:
        JSON with benchmark results. For run_mode="ray", returns immediately
        with task_id and status="PENDING". Use ray_task_status() to track.
        For other modes, blocks until complete and returns full results:
        - success: bool
        - framework, model: identifiers
        - throughput: request_throughput (req/s), output_throughput (tok/s), etc.
        - latency: TTFT, TPOT, ITL, E2EL (mean/median/p99/std in ms)
        - kernel_summary: top GPU kernels by time
        - top_bottlenecks: names of top 10 bottleneck kernels
        - gap_analysis: kernel bottleneck analysis (if gap_analysis_enabled)
        - tracelens_analysis: TraceLens output files (if enabled)
        - workspace_dir: path to all output files
        - execution_time: total wall time in seconds
        - errors: list of error messages (if any)
    """
    from ..modes.benchmark import BenchmarkMode, BenchmarkConfig

    try:
        envs: Dict[str, Any] = {
            "TP": tp,
            "CONC": concurrency,
            "ISL": input_len,
            "OSL": output_len,
            "RANDOM_RANGE_RATIO": 0.5,
        }
        if extra_envs:
            envs.update(extra_envs)

        profiler_cfg = {
            "torch_profiler": {"enabled": torch_profiler},
            "system_profiler": {"enabled": system_profiler},
            "tracelens": {
                "enabled": tracelens,
                "export_format": tracelens_export_format,
                "perf_report_enabled": True,
                "multi_rank_report_enabled": tp > 1,
            },
        }

        gap_analysis_cfg = {
            "enabled": gap_analysis_enabled,
            "top_k": gap_analysis_top_k,
            "trace_start_pct": gap_analysis_start_pct,
            "trace_end_pct": gap_analysis_end_pct,
            "min_duration_us": gap_analysis_min_duration_us,
            "categories": gap_analysis_categories or ["kernel", "gpu"],
            "ignore_categories": gap_analysis_ignore_categories
            or ["gpu_user_annotation"],
        }

        # Build Ray config if run_mode is "ray"
        ray_config_dict = None
        if run_mode == "ray":
            ray_config_dict = {
                "cluster_address": ray_cluster_address or "auto",
                "shared_storage_path": ray_shared_storage_path
                or DEFAULT_SHARED_STORAGE_PATH,
                "entrypoint_num_gpus": ray_num_gpus,
                "multi_node": ray_multi_node,
                "total_num_gpus": ray_total_num_gpus,
                "num_nodes": ray_num_nodes,
                "gpus_per_node": ray_total_num_gpus // max(ray_num_nodes, 1),
            }

        config_dict = {
            "framework": framework,
            "model": model,
            "precision": precision,
            "run_mode": run_mode,
            "envs": envs,
            "profiler": profiler_cfg,
            "gap_analysis": gap_analysis_cfg,
            "docker_image": docker_image,
            "gpu_arch": gpu_arch,
            "timeout_seconds": timeout_seconds,
            "inferencex_path": inferencex_path,
            "hf_cache_path": hf_cache_path,
            "benchmark_script": benchmark_script,
            "runner_type": runner_type,
        }
        if ray_config_dict:
            config_dict["ray_config"] = ray_config_dict

        benchmark_config = BenchmarkConfig.from_dict(config_dict)

        benchmarker = BenchmarkMode(
            config=benchmark_config,
            output_dir=output_dir,
        )

        try:
            result = await asyncio.to_thread(benchmarker.run)
        finally:
            benchmarker.cleanup()

        response = result.to_dict()
        response["summary_text"] = result.get_summary()
        return json.dumps(response, indent=2)

    except Exception as e:
        logger.exception(f"Benchmark failed: {e}")
        return json.dumps(
            {
                "error": str(e),
                "framework": framework,
                "model": model,
            }
        )


# =============================================================================
# Tool 9: gap_analysis
# =============================================================================
@mcp.tool()
def gap_analysis(
    trace_dir: str,
    start_pct: float = 0.0,
    end_pct: float = 100.0,
    top_k: int = 20,
    min_duration_us: float = 0.0,
    categories: Optional[List[str]] = None,
    ignore_categories: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    generate_clamped_traces: bool = False,
) -> str:
    """
    Run gap analysis on existing torch profiler traces to identify GPU kernel bottlenecks.

    Parses Chrome-trace JSON files produced by PyTorch profiler, aggregates
    kernel statistics across ranks, and identifies the top time-consuming
    kernels. Works on traces from any source (Magpie benchmark, manual
    profiling, etc.).

    Args:
        trace_dir: Path to the directory containing torch profiler trace files
            (.json or .json.gz). If the path contains a torch_trace/ subdirectory,
            it is used automatically.
        start_pct: Start of the analysis time window as percentage of total
            trace duration (0-100, default: 0)
        end_pct: End of the analysis time window as percentage of total
            trace duration (0-100, default: 100)
        top_k: Number of top bottleneck kernels to report (default: 20)
        min_duration_us: Filter out events shorter than this threshold
            in microseconds (default: 0)
        categories: Event categories to include using case-insensitive
            substring matching (e.g., ["kernel", "gpu"]). None = all categories.
        ignore_categories: Event categories to exclude (e.g.,
            ["gpu_user_annotation"]). None = exclude nothing.
        output_dir: Directory for output CSV files. Defaults to
            gap_analysis/ next to the trace directory.
        generate_clamped_traces: If True, also generate time-windowed
            (clamped) trace files for visualization (default: False)

    Returns:
        JSON with gap analysis results:
        - num_ranks: number of ranks analyzed
        - total_duration_us: total GPU time in microseconds
        - top_kernels: list of top-k kernels with name, calls,
          self_cuda_total_us, avg_time_us, pct_total
        - output_dir: path to CSV output files
        - csv_file: path to the main gap_analysis.csv
        - rank_csv_files: paths to per-rank CSV files (multi-rank only)
        - clamped_trace_files: paths to clamped traces (if requested)
        - errors: any warnings or errors during analysis
    """
    from ..modes.benchmark.gap_analysis import GapAnalyzer
    from ..modes.benchmark.config import GapAnalysisConfig

    try:
        trace_path = Path(trace_dir).resolve()

        if (trace_path / "torch_trace").is_dir():
            trace_path = trace_path / "torch_trace"

        if not trace_path.is_dir():
            return json.dumps({"error": f"Trace directory not found: {trace_dir}"})

        gap_config = GapAnalysisConfig(
            enabled=True,
            trace_start_pct=start_pct,
            trace_end_pct=end_pct,
            top_k=top_k,
            min_duration_us=min_duration_us,
            categories=categories,
            ignore_categories=ignore_categories,
        )

        base_dir = Path(output_dir) if output_dir else trace_path.parent
        gap_dir = base_dir / "gap_analysis"
        gap_dir.mkdir(parents=True, exist_ok=True)

        analyzer = GapAnalyzer(gap_config)
        result = analyzer.analyze(trace_path)

        response: Dict[str, Any] = {
            "num_ranks": len(result.rank_results),
            "total_duration_us": result.total_duration_us,
            "top_kernels": result.to_dict().get("top_kernels", []),
            "output_dir": str(gap_dir),
            "errors": result.errors,
        }

        if result.merged_kernels:
            csv_path = result.to_csv(gap_dir / "gap_analysis.csv")
            response["csv_file"] = str(csv_path)

            if len(result.rank_results) > 1:
                rank_paths = result.to_rank_csv(gap_dir)
                response["rank_csv_files"] = [str(p) for p in rank_paths]
        else:
            response["warning"] = "No kernel events found in traces"

        if generate_clamped_traces:
            clamped_paths = analyzer.generate_clamped_traces(
                trace_path,
                output_dir=gap_dir,
            )
            response["clamped_trace_files"] = [str(p) for p in clamped_paths]

        return json.dumps(response, indent=2)

    except Exception as e:
        logger.exception(f"Gap analysis failed: {e}")
        return json.dumps({"error": str(e), "trace_dir": trace_dir})


# =============================================================================
# Tool 10: list_benchmark_images
# =============================================================================
@mcp.tool()
def list_benchmark_images(
    framework: Optional[str] = None,
) -> str:
    """
    List available Docker images for benchmarking.

    Shows the framework -> GPU architecture -> Docker image mapping used
    by benchmark mode to auto-select containers.

    Args:
        framework: Filter by framework ("vllm" or "sglang"). If None, list all.

    Returns:
        JSON with image mapping: { framework: { gpu_arch: docker_image } }
    """
    from ..modes.benchmark import ImageSelector

    try:
        selector = ImageSelector()
        images = selector.list_available_images(framework=framework)

        runner_info = {}
        for fw, arch_map in images.items():
            runner_info[fw] = {}
            for arch, image in arch_map.items():
                try:
                    runner = selector.get_runner_type(arch)
                except (ValueError, Exception):
                    runner = "unknown"
                runner_info[fw][arch] = {
                    "image": image,
                    "runner_type": runner,
                }

        return json.dumps(
            {
                "images": images,
                "details": runner_info,
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Tool 11: list_benchmark_results
# =============================================================================
@mcp.tool()
def list_benchmark_results(
    output_dir: str = "./results",
    limit: int = 20,
) -> str:
    """
    List previous benchmark runs and their summary results.

    Scans the output directory for benchmark workspaces and returns a
    summary of each run (framework, model, status, throughput, latency).

    Args:
        output_dir: Base directory where benchmark results are stored (default: "./results")
        limit: Maximum number of results to return, most recent first (default: 20)

    Returns:
        JSON with list of benchmark runs, each containing:
        - workspace: directory path
        - timestamp: extracted from directory name
        - framework: vllm or sglang
        - config: benchmark configuration snapshot
        - metrics: throughput and latency summary (if available)
        - has_torch_trace: whether torch traces exist
        - has_tracelens: whether TraceLens output exists
    """
    from ..modes.benchmark import WorkspaceManager

    try:
        workspaces = WorkspaceManager.list_workspaces(base_dir=output_dir)
        runs = []

        for ws_path in workspaces[:limit]:
            ws = Path(ws_path)
            entry: Dict[str, Any] = {
                "workspace": str(ws),
                "name": ws.name,
            }

            # Extract framework and timestamp from directory name
            # Format: benchmark_{framework}_{timestamp}
            parts = ws.name.split("_", 2)
            if len(parts) >= 3:
                entry["framework"] = parts[1]
                entry["timestamp"] = parts[2]

            # Read config snapshot
            config_file = ws / "config.yaml"
            if config_file.exists():
                try:
                    with open(config_file) as f:
                        entry["config"] = yaml.safe_load(f) or {}
                except Exception:
                    entry["config"] = None

            # Read benchmark report for summary metrics
            report_file = ws / "benchmark_report.json"
            if report_file.exists():
                try:
                    with open(report_file) as f:
                        report = json.load(f)
                    entry["success"] = report.get("success", False)
                    entry["model"] = report.get("model", "")
                    entry["execution_time"] = report.get("execution_time", 0)
                    entry["throughput"] = report.get("throughput")
                    entry["latency"] = report.get("latency")
                    entry["top_bottlenecks"] = report.get("top_bottlenecks", [])[:5]
                    entry["errors"] = report.get("errors", [])
                except Exception:
                    entry["report_error"] = "Failed to parse benchmark_report.json"
            else:
                entry["success"] = None

            # Check for trace and analysis artifacts
            entry["has_torch_trace"] = (
                (ws / "torch_trace").exists() and any((ws / "torch_trace").iterdir())
                if (ws / "torch_trace").exists()
                else False
            )
            entry["has_tracelens"] = (ws / "tracelens_rank0_csvs").exists() or (
                ws / "tracelens_collective_csvs"
            ).exists()

            runs.append(entry)

        return json.dumps(
            {
                "output_dir": output_dir,
                "total_found": len(workspaces),
                "showing": len(runs),
                "runs": runs,
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Tool 12: get_benchmark_result
# =============================================================================
@mcp.tool()
def get_benchmark_result(
    workspace_dir: str,
    include_kernel_summary: bool = True,
    include_raw_result: bool = False,
    include_tracelens_files: bool = True,
) -> str:
    """
    Read detailed results from a specific benchmark run.

    Loads all available data from a benchmark workspace directory
    including the report, config, kernel summary, and TraceLens output file listing.

    Args:
        workspace_dir: Path to the benchmark workspace directory
        include_kernel_summary: Include per-kernel profiling breakdown (default: True)
        include_raw_result: Include raw InferenceX JSON output (default: False)
        include_tracelens_files: List TraceLens output files (default: True)

    Returns:
        JSON with full benchmark details:
        - config: benchmark configuration
        - success, framework, model
        - throughput, latency metrics
        - kernel_summary: per-kernel time/percentage/calls
        - tracelens_files: list of TraceLens output file paths
        - errors
    """
    try:
        ws = Path(workspace_dir)
        if not ws.exists():
            return json.dumps({"error": f"Workspace not found: {workspace_dir}"})

        result: Dict[str, Any] = {"workspace": str(ws)}

        # Load config
        config_file = ws / "config.yaml"
        if config_file.exists():
            with open(config_file) as f:
                result["config"] = yaml.safe_load(f)

        # Load main report
        report_file = ws / "benchmark_report.json"
        if report_file.exists():
            with open(report_file) as f:
                report = json.load(f)

            result["success"] = report.get("success", False)
            result["framework"] = report.get("framework", "")
            result["model"] = report.get("model", "")
            result["execution_time"] = report.get("execution_time", 0)
            result["throughput"] = report.get("throughput")
            result["latency"] = report.get("latency")
            result["top_bottlenecks"] = report.get("top_bottlenecks", [])
            result["errors"] = report.get("errors", [])

            if include_kernel_summary:
                result["kernel_summary"] = report.get("kernel_summary", [])

            if include_raw_result:
                # Load raw InferenceX result
                raw_file = ws / "inferencex_result.json"
                if raw_file.exists():
                    with open(raw_file) as f:
                        result["raw_inferencex_result"] = json.load(f)
        else:
            result["error"] = "benchmark_report.json not found in workspace"

        # Load summary text
        summary_file = ws / "summary.txt"
        if summary_file.exists():
            with open(summary_file) as f:
                result["summary_text"] = f.read()

        # List TraceLens outputs
        if include_tracelens_files:
            tracelens_files: Dict[str, List[str]] = {}

            rank0_dir = ws / "tracelens_rank0_csvs"
            if rank0_dir.exists():
                tracelens_files["rank0_csvs"] = sorted(
                    str(f) for f in rank0_dir.iterdir() if f.is_file()
                )

            collective_dir = ws / "tracelens_collective_csvs"
            if collective_dir.exists():
                tracelens_files["collective_csvs"] = sorted(
                    str(f) for f in collective_dir.iterdir() if f.is_file()
                )

            if tracelens_files:
                result["tracelens_files"] = tracelens_files

        # List torch trace files
        trace_dir = ws / "torch_trace"
        if trace_dir.exists():
            result["torch_trace_files"] = sorted(
                str(f) for f in trace_dir.iterdir() if f.is_file()
            )

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Tool 13: compare_benchmark_reports
# =============================================================================
@mcp.tool()
def compare_benchmark_reports(
    workspace_dirs: List[str],
    labels: Optional[List[str]] = None,
    output_dir: str = "./results/comparisons",
    export_format: str = "csv",
) -> str:
    """
    Compare TraceLens performance reports from multiple benchmark runs.

    Uses TraceLens_compare_perf_reports_pytorch to generate a side-by-side
    comparison of kernel-level performance across different benchmark runs.
    Useful for A/B testing framework versions, models, or configurations.

    Requires at least 2 benchmark workspaces that have TraceLens rank0 CSV output.

    Args:
        workspace_dirs: List of benchmark workspace directories to compare (minimum 2)
        labels: Display labels for each run (e.g., ["baseline", "optimized"]).
                If None, workspace directory names are used.
        output_dir: Directory for comparison output files (default: "./results/comparisons")
        export_format: "csv" or "excel" (default: "csv")

    Returns:
        JSON with comparison results:
        - output_files: list of generated comparison CSV/Excel files
        - run_summaries: throughput/latency summary for each run
        - errors: any errors during comparison
    """
    from ..modes.benchmark.config import TraceLensConfig
    from ..modes.benchmark.tracelens import TraceLensAnalyzer

    try:
        if len(workspace_dirs) < 2:
            return json.dumps({"error": "At least 2 workspace directories required"})

        # Validate workspaces and collect summaries
        report_dirs = []
        run_summaries = []
        for i, ws_path in enumerate(workspace_dirs):
            ws = Path(ws_path)
            rank0_dir = ws / "tracelens_rank0_csvs"
            if not rank0_dir.exists():
                return json.dumps(
                    {
                        "error": f"No TraceLens rank0 CSVs found in {ws_path}. "
                        f"Run benchmark with tracelens=True first.",
                    }
                )
            report_dirs.append(rank0_dir)

            # Collect summary for context
            summary: Dict[str, Any] = {
                "workspace": str(ws),
                "label": labels[i] if labels and i < len(labels) else ws.name,
            }
            report_file = ws / "benchmark_report.json"
            if report_file.exists():
                with open(report_file) as f:
                    report = json.load(f)
                summary["framework"] = report.get("framework", "")
                summary["model"] = report.get("model", "")
                summary["throughput"] = report.get("throughput")
                summary["latency"] = report.get("latency")
            run_summaries.append(summary)

        # Create TraceLens config for comparison
        tl_config = TraceLensConfig(
            enabled=True,
            export_format=export_format,
        )

        analyzer = TraceLensAnalyzer(tl_config)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        comparison = analyzer.compare_reports(
            report_dirs=report_dirs,
            output_dir=output_path,
            labels=labels or [ws.name for ws in (Path(d) for d in workspace_dirs)],
        )

        return json.dumps(
            {
                "success": comparison.get("error") is None,
                "output_dir": str(output_path),
                "output_files": comparison.get("files", []),
                "run_summaries": run_summaries,
                "errors": [comparison["error"]] if comparison.get("error") else [],
            },
            indent=2,
        )

    except Exception as e:
        logger.exception(f"Benchmark comparison failed: {e}")
        return json.dumps({"error": str(e)})


# =============================================================================
# Tool 14: ray_task_status
# =============================================================================

_ray_executor_instance: Any = None
_ray_executor_key: Optional[Tuple[str, str]] = None


def _get_ray_executor(
    cluster_address: str = "auto",
    shared_storage_path: str = DEFAULT_SHARED_STORAGE_PATH,
) -> Any:
    """Return or lazily create a singleton RayJobExecutor for MCP tools."""
    global _ray_executor_instance, _ray_executor_key
    addr = cluster_address or "auto"
    path = shared_storage_path or DEFAULT_SHARED_STORAGE_PATH
    key = (addr, path)
    if (
        _ray_executor_instance is not None
        and _ray_executor_key == key
        and _ray_executor_instance.is_running()
    ):
        return _ray_executor_instance
    if _ray_executor_instance is not None:
        try:
            _ray_executor_instance.stop()
        except Exception:
            logger.debug("Stopping previous RayJobExecutor for MCP", exc_info=True)
    from ..core.ray_executor import RayJobExecutor
    from ..core.executor import ExecutorConfig, ExecutorType
    from ..modes.benchmark.config import RayConfig

    rc = RayConfig(cluster_address=addr, shared_storage_path=path)
    cfg = ExecutorConfig(executor_type=ExecutorType.RAY)
    _ray_executor_key = key
    _ray_executor_instance = RayJobExecutor(cfg, ray_config=rc)
    _ray_executor_instance.start()
    return _ray_executor_instance


@mcp.tool()
def ray_task_status(
    task_id: str,
    ray_cluster_address: str = "",
    ray_shared_storage_path: str = "",
) -> str:
    """
    Check the status of a Ray-dispatched benchmark/analysis task.

    Args:
        task_id: Magpie task ID returned by the benchmark tool
        ray_cluster_address: Ray cluster address ("auto" or "ray://<ip>:10001")
        ray_shared_storage_path: Must match ``benchmark_batch`` shared-storage path

    Returns:
        JSON with task status (RUNNING, SUCCEEDED, FAILED, or UNKNOWN).
    """
    try:
        executor = _get_ray_executor(
            ray_cluster_address or "auto",
            ray_shared_storage_path or DEFAULT_SHARED_STORAGE_PATH,
        )
        status = executor.get_task_status_ray(task_id)
        return json.dumps(
            {
                "task_id": task_id,
                "status": status,
            },
            indent=2,
        )
    except Exception as e:
        logger.exception(f"ray_task_status failed: {e}")
        return json.dumps({"error": str(e), "task_id": task_id})


# =============================================================================
# Tool 15: ray_task_result
# =============================================================================
@mcp.tool()
def ray_task_result(
    task_id: str,
    ray_cluster_address: str = "",
    ray_shared_storage_path: str = "",
) -> str:
    """
    Retrieve the result of a completed Ray task.

    Only works after the task has reached SUCCEEDED status.

    Args:
        task_id: Magpie task ID
        ray_cluster_address: Ray cluster address
        ray_shared_storage_path: Must match ``benchmark_batch`` shared-storage path

    Returns:
        JSON with the full benchmark/analysis result dict.
    """
    try:
        executor = _get_ray_executor(
            ray_cluster_address or "auto",
            ray_shared_storage_path or DEFAULT_SHARED_STORAGE_PATH,
        )
        result = executor.get_task_result(task_id)
        if result is None:
            status = executor.get_task_status_ray(task_id)
            return json.dumps(
                {
                    "error": f"No result available for task {task_id}",
                    "status": status,
                    "hint": "Task may still be running. Check ray_task_status first.",
                }
            )
        result["_task_id"] = task_id
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.exception(f"ray_task_result failed: {e}")
        return json.dumps({"error": str(e), "task_id": task_id})


# =============================================================================
# Tool 16: ray_task_cancel
# =============================================================================
@mcp.tool()
def ray_task_cancel(
    task_id: str,
    ray_cluster_address: str = "",
    ray_shared_storage_path: str = "",
) -> str:
    """
    Cancel a running Ray task.

    Args:
        task_id: Magpie task ID to cancel
        ray_cluster_address: Ray cluster address
        ray_shared_storage_path: Must match ``benchmark_batch`` shared-storage path

    Returns:
        JSON with cancellation result.
    """
    try:
        executor = _get_ray_executor(
            ray_cluster_address or "auto",
            ray_shared_storage_path or DEFAULT_SHARED_STORAGE_PATH,
        )
        ok = executor.cancel_task(task_id)
        return json.dumps(
            {
                "task_id": task_id,
                "cancelled": ok,
                "message": f"Task {task_id} cancel {'requested' if ok else 'failed'}",
            },
            indent=2,
        )
    except Exception as e:
        logger.exception(f"ray_task_cancel failed: {e}")
        return json.dumps({"error": str(e), "task_id": task_id})


# =============================================================================
# Tool 17: ray_task_list
# =============================================================================
@mcp.tool()
def ray_task_list(
    ray_cluster_address: str = "",
    ray_shared_storage_path: str = "",
) -> str:
    """
    List all tracked Ray tasks in this session with their status.

    Args:
        ray_cluster_address: Ray cluster address
        ray_shared_storage_path: Must match ``benchmark_batch`` shared-storage path

    Returns:
        JSON with list of {task_id, status} entries.
    """
    try:
        executor = _get_ray_executor(
            ray_cluster_address or "auto",
            ray_shared_storage_path or DEFAULT_SHARED_STORAGE_PATH,
        )
        tasks = executor.list_tasks()
        jobs = [{"task_id": tid, "status": s} for tid, s in tasks.items()]
        return json.dumps(
            {
                "total": len(jobs),
                "tasks": jobs,
            },
            indent=2,
        )
    except Exception as e:
        logger.exception(f"ray_task_list failed: {e}")
        return json.dumps({"error": str(e)})


# =============================================================================
# Tool 19: benchmark_batch
# =============================================================================
@mcp.tool()
async def benchmark_batch(
    configs: List[Dict[str, Any]],
    ray_cluster_address: str = "",
    ray_shared_storage_path: str = "",
    parallel: bool = False,
) -> str:
    """
    Submit multiple independent benchmarks to a Ray cluster.

    When ``parallel`` is False (default), each benchmark runs to completion
    before the next starts (sequential). When ``parallel`` is True, each
    config is submitted asynchronously via the shared MCP ``RayJobExecutor``;
    use ``ray_task_list``, ``ray_task_status``, and ``ray_task_result`` to
    track and retrieve outcomes.

    Args:
        configs: List of benchmark configuration dicts.  Each dict should have
            at least "framework" and "model".  Other fields default to the
            same values as the benchmark() tool.
        ray_cluster_address: Ray cluster address ("auto" or "ray://<ip>:10001")
        ray_shared_storage_path: Shared filesystem root on worker nodes (HF cache, results)
        parallel: If True, submit all jobs without waiting (non-blocking).
            If False, run benchmarks one after another until each finishes.

    Returns:
        JSON with job entries and, for sequential mode, full run metadata;
        for parallel mode, ``task_id`` / ``ray_job_id`` per job with
        ``status``: ``SUBMITTED``.
    """
    from ..modes.benchmark import BenchmarkMode, BenchmarkConfig

    def _run_batch() -> str:
        try:
            if not configs:
                return json.dumps({"error": "configs list is empty"})

            results: List[Dict[str, Any]] = []
            addr = ray_cluster_address or "auto"
            shared_storage_root = ray_shared_storage_path or DEFAULT_SHARED_STORAGE_PATH

            if parallel:
                executor = _get_ray_executor(addr, shared_storage_root)
                for i, raw in enumerate(configs):
                    try:
                        cfg = dict(raw)
                        cfg["run_mode"] = "ray"
                        if "ray_config" not in cfg:
                            cfg["ray_config"] = {
                                "cluster_address": addr,
                                "shared_storage_path": shared_storage_root,
                            }

                        benchmark_config = BenchmarkConfig.from_dict(cfg)
                        benchmarker = BenchmarkMode(
                            config=benchmark_config,
                            output_dir=cfg.get("output_dir", "./results"),
                        )
                        sub = benchmarker.submit_ray_benchmark(executor)
                        tid = (sub.metadata or {}).get("task_id", "unknown")
                        if sub.success:
                            results.append(
                                {
                                    "index": i,
                                    "framework": cfg.get("framework"),
                                    "model": cfg.get("model"),
                                    "task_id": tid,
                                    "ray_job_id": tid,
                                    "status": "SUBMITTED",
                                    "submitted": True,
                                }
                            )
                        else:
                            results.append(
                                {
                                    "index": i,
                                    "framework": cfg.get("framework"),
                                    "model": cfg.get("model"),
                                    "error": "; ".join(sub.errors)
                                    if sub.errors
                                    else "submit failed",
                                }
                            )
                    except Exception as e:
                        results.append(
                            {
                                "index": i,
                                "framework": raw.get("framework"),
                                "model": raw.get("model"),
                                "error": str(e),
                            }
                        )
            else:
                for i, raw in enumerate(configs):
                    try:
                        cfg = dict(raw)
                        cfg["run_mode"] = "ray"
                        if "ray_config" not in cfg:
                            cfg["ray_config"] = {
                                "cluster_address": addr,
                                "shared_storage_path": shared_storage_root,
                            }

                        benchmark_config = BenchmarkConfig.from_dict(cfg)
                        benchmarker = BenchmarkMode(
                            config=benchmark_config,
                            output_dir=cfg.get("output_dir", "./results"),
                        )
                        result = benchmarker.run()
                        ray_job_id = (result.metadata or {}).get(
                            "ray_job_id", "unknown"
                        )
                        results.append(
                            {
                                "index": i,
                                "framework": cfg.get("framework"),
                                "model": cfg.get("model"),
                                "ray_job_id": ray_job_id,
                                "status": "PENDING",
                                "workspace_dir": result.workspace_dir,
                            }
                        )
                    except Exception as e:
                        results.append(
                            {
                                "index": i,
                                "framework": raw.get("framework"),
                                "model": raw.get("model"),
                                "error": str(e),
                            }
                        )

            ok = [r for r in results if "error" not in r]
            return json.dumps(
                {
                    "parallel": parallel,
                    "submitted": len(ok),
                    "failed": len([r for r in results if "error" in r]),
                    "jobs": results,
                },
                indent=2,
            )

        except Exception as e:
            logger.exception(f"benchmark_batch failed: {e}")
            return json.dumps({"error": str(e)})

    return await asyncio.to_thread(_run_batch)


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    mcp.run()
