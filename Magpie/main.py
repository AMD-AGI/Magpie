###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Magpie: GPU Kernel Evaluation Framework

Usage:
    # Run from project root
    python -m Magpie analyze kernel.hip -t "./test.sh"

    # Analyze with kernel config file
    python -m Magpie analyze --kernel-config my_kernel.yaml

    # Compare kernels
    python -m Magpie compare kernel1.hip kernel2.hip

    # Compare with kernel config file
    python -m Magpie compare --kernel-config kernels.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # type: ignore[import-untyped]

from .config import KernelType, KernelEvalConfig
from .core import Scheduler, SchedulerConfig, EnvironmentType
from .eval import EvaluationState, BaseKind
from .utils import get_gpu_info

logger = logging.getLogger(__name__)


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file."""
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def setup_logging(config: Dict[str, Any], verbose: bool = False) -> None:
    """Configure logging."""
    log_cfg = config.get("logging", {})
    level_str = "DEBUG" if verbose else log_cfg.get("level", "INFO")
    level = getattr(logging, level_str.upper(), logging.INFO)

    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def parse_kernel_type(type_str: str) -> KernelType:
    """Parse kernel type from string."""
    type_map = {
        "hip": KernelType.HIP,
        "cuda": KernelType.CUDA,
        "pytorch": KernelType.PYTORCH,
        "torch": KernelType.PYTORCH,
        "py": KernelType.PYTORCH,
        "triton": KernelType.TRITON,
    }
    normalized = type_str.lower()
    if normalized not in type_map:
        supported = ", ".join(sorted(type_map))
        raise ValueError(
            f"Unsupported kernel type '{type_str}'. Supported types: {supported}"
        )
    return type_map[normalized]


def load_kernel_config(
    kernel_config_path: Path,
) -> tuple[List[KernelEvalConfig], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Load kernel configuration from YAML file.

    The YAML may optionally contain:
    - ``performance:`` — overrides framework-level profiler settings
    - ``correctness:`` — overrides framework-level correctness settings
    - ``ray_config:`` — Ray cluster settings (implies ``environment: ray``)
    - ``scheduler:`` — scheduler-level overrides (environment, workers, …)

    Returns:
        Tuple of (kernel configs, performance overrides, correctness overrides,
        scheduler overrides). Override dicts are empty when absent.
    """
    data = load_yaml(kernel_config_path)
    configs = []

    # Single kernel
    if "kernel" in data:
        cfg = _parse_kernel_entry(data["kernel"])
        if cfg:
            configs.append(cfg)

    # Multiple kernels
    if "kernels" in data:
        for entry in data["kernels"]:
            cfg = _parse_kernel_entry(entry)
            if cfg:
                configs.append(cfg)

    # Performance overrides (optional section in kernel config)
    perf_overrides = _expand_env_vars(data.get("performance", {}))

    corr_overrides = _expand_env_vars(data.get("correctness", {}))

    sched_overrides: Dict[str, Any] = dict(data.get("scheduler", {}))
    if "ray_config" in data:
        sched_overrides["ray_config"] = data["ray_config"]
        sched_overrides.setdefault("environment", "ray")

    return configs, perf_overrides, corr_overrides, sched_overrides


def _parse_command_list(cmd_entry) -> Optional[List]:
    """
    Parse a command entry which can be:

    Single command formats:
    - A string: "make build" -> ["make", "build"]
    - A list of strings: ["make", "build"] -> ["make", "build"]

    Multiple commands format (list of lists):
    - [["make", "clean"], ["make", "build"]] -> [["make", "clean"], ["make", "build"]]
    - In YAML:
        compile_command:
          - ["cmake", "-B", "build"]
          - ["cmake", "--build", "build"]
      Or:
        compile_command:
          - - cmake
            - -B
            - build
          - - cmake
            - --build
            - build

    Returns:
        - Single command as List[str]
        - Multiple commands as List[List[str]]
        - None if no command
    """
    if cmd_entry is None:
        return None

    cmd_entry = _expand_env_vars(cmd_entry)

    if isinstance(cmd_entry, str):
        # Single string command: "make build" -> ["make", "build"]
        return cmd_entry.split()

    if isinstance(cmd_entry, list):
        if len(cmd_entry) == 0:
            return None

        # Check if it's a list of lists (multiple commands)
        # e.g., [["make", "clean"], ["make", "build"]]
        if isinstance(cmd_entry[0], list):
            return cmd_entry

        # It's a list of strings - treat as single command
        # e.g., ["make", "build"]
        if all(isinstance(item, str) for item in cmd_entry):
            return cmd_entry

    return None


def _expand_env_vars(value):
    """Recursively expand environment variables in strings."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_vars(val) for key, val in value.items()}
    return value


def _parse_kernel_entry(entry: Dict[str, Any]) -> Optional[KernelEvalConfig]:
    """Parse a single kernel entry from config."""
    if not entry:
        return None

    kernel_type = parse_kernel_type(entry.get("type", "hip"))

    source_files = _expand_env_vars(entry.get("source_files", []))
    working_dir = _expand_env_vars(entry.get("working_dir"))
    env = _expand_env_vars(entry.get("env"))

    # Parse testcase command(s)
    testcase_cmd = _parse_command_list(entry.get("testcase_command"))

    # Parse compile command(s)
    compile_cmd = _parse_command_list(entry.get("compile_command"))

    # Parse prof command(s)
    prof_cmd = _parse_command_list(entry.get("prof_command"))

    return KernelEvalConfig(
        kernel_id=entry.get("id", "kernel"),
        kernel_type=kernel_type,
        source_file_path=source_files,
        working_dir=working_dir,
        env=env,
        testcase_command=testcase_cmd,
        compiling_command=compile_cmd,
        prof_command=prof_cmd,
        get_inputs_func=entry.get("get_inputs_func", "get_inputs"),
        get_init_inputs_func=entry.get("get_init_inputs_func", "get_init_inputs"),
    )


def _get_compiling_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get compiling configuration from framework config.

    Args:
        config: Framework config dict

    Returns:
        Dict with enable_default_compile
    """
    compile_cfg = config.get("compiling", {})
    return {
        "enable_default_compile": compile_cfg.get("enable_default_compile", False),
    }


def _get_correctness_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get correctness configuration from framework config.

    Args:
        config: Framework config dict

    Returns:
        Dict with backend and accordo settings.
    """
    corr_cfg = config.get("correctness", {})
    backend_str = corr_cfg.get("backend")

    acc_cfg = corr_cfg.get("accordo", {})
    accordo_settings = {
        "atol": acc_cfg.get("atol", 1e-6),
        "rtol": acc_cfg.get("rtol", 1e-5),
        "equal_nan": acc_cfg.get("equal_nan", False),
        "timeout_seconds": acc_cfg.get("timeout_seconds", 30),
    }

    result: Dict[str, Any] = {"accordo": accordo_settings}
    if backend_str:
        result["backend"] = backend_str

    return result


def _apply_correctness_overrides(
    corr_settings: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge kernel-config-level ``correctness:`` overrides into *corr_settings*.

    Supported override keys (all optional):
      - ``backend``: ``"testcase"`` | ``"accordo"``
      - ``accordo``: dict of Accordo-specific settings
    """
    settings = dict(corr_settings)

    if "backend" in overrides:
        settings["backend"] = overrides["backend"]

    if "accordo" in overrides:
        acc = dict(settings.get("accordo", {}))
        acc.update(overrides["accordo"])
        settings["accordo"] = acc

    return settings


def _get_performance_config(
    config: Dict[str, Any], kernel_type: KernelType
) -> Dict[str, Any]:
    """
    Get performance configuration from framework config.

    Args:
        config: Framework config dict
        kernel_type: Kernel type to determine which profiler args to use

    Returns:
        Dict with timeout_seconds, profiler_args, rocprof_config, ncu_config,
        and metrix_config.
    """
    perf_cfg = config.get("performance", {})

    # Get timeout
    timeout = perf_cfg.get("timeout_seconds", 300.0)

    # Explicit backend override (e.g. "metrix" for AMD GPUs)
    backend_str = perf_cfg.get("backend")

    # Get profiler args and config based on kernel type
    profiler_args = []
    rocprof_config = {}
    ncu_config = {}
    metrix_config = {}

    if kernel_type in (KernelType.HIP, KernelType.TRITON):
        rocprof_cfg = perf_cfg.get("rocprof_compute", {})
        rocprof_config = {
            "workload_dir": rocprof_cfg.get("workload_dir", "./workloads"),
            "metric_blocks": rocprof_cfg.get(
                "metric_blocks", ["1", "2", "5", "10", "11", "12", "14", "16", "17"]
            ),
            "output_format": rocprof_cfg.get("output_format", "csv"),
            "profile_args": rocprof_cfg.get("profile_args", []),
            "analyze_args": rocprof_cfg.get("analyze_args", []),
        }

        # Metrix config (available for HIP/Triton on AMD)
        mtx_cfg = perf_cfg.get("metrix", {})
        metrix_config = {
            "profile": mtx_cfg.get("profile"),
            "metrics": mtx_cfg.get("metrics", []),
            "kernel_filter": mtx_cfg.get("kernel_filter"),
            "num_replays": mtx_cfg.get("num_replays", 1),
            "timeout_seconds": mtx_cfg.get("timeout_seconds", 60),
            "extra_args": mtx_cfg.get("extra_args", []),
            "backend": backend_str,
        }

    if kernel_type in (KernelType.CUDA, KernelType.TRITON):
        ncu_cfg = perf_cfg.get("ncu", {})
        profiler_args = ncu_cfg.get("args", [])
        ncu_config = {
            "args": ncu_cfg.get("args", []),
            "metrics": ncu_cfg.get("metrics", []),
        }

    return {
        "timeout_seconds": timeout,
        "profiler_args": profiler_args,
        "rocprof_config": rocprof_config,
        "ncu_config": ncu_config,
        "metrix_config": metrix_config,
    }


def _apply_perf_overrides(
    perf_settings: Dict[str, Any],
    overrides: Dict[str, Any],
    kernel_type: "KernelType",
) -> Dict[str, Any]:
    """Merge kernel-config-level ``performance:`` overrides into *perf_settings*.

    Supported override keys (all optional):
      - ``backend``: ``"metrix"`` | ``"rocprof_compute"`` | ``"ncu"``
      - ``timeout_seconds``: int
      - ``metrix``: dict of Metrix-specific settings
      - ``rocprof_compute`` / ``ncu``: dict of backend-specific settings
    """
    settings = dict(perf_settings)

    backend_str = overrides.get("backend")

    if "timeout_seconds" in overrides:
        settings["timeout_seconds"] = int(overrides["timeout_seconds"])

    # Metrix overrides
    if "metrix" in overrides or backend_str == "metrix":
        mtx = overrides.get("metrix", {})
        settings["metrix_config"] = {
            "profile": mtx.get("profile"),
            "metrics": mtx.get("metrics", []),
            "kernel_filter": mtx.get("kernel_filter"),
            "num_replays": mtx.get("num_replays", 1),
            "timeout_seconds": mtx.get(
                "timeout_seconds", settings.get("timeout_seconds", 600)
            ),
            "backend": backend_str or "metrix",
        }

    # rocprof_compute overrides
    if "rocprof_compute" in overrides:
        rpc = overrides["rocprof_compute"]
        existing = settings.get("rocprof_config", {})
        existing.update(rpc)
        settings["rocprof_config"] = existing

    # ncu overrides
    if "ncu" in overrides:
        ncu = overrides["ncu"]
        existing = settings.get("ncu_config", {})
        existing.update(ncu)
        settings["ncu_config"] = existing

    # Propagate explicit backend into metrix_config even if only backend was set
    if backend_str and backend_str != "metrix":
        settings.setdefault("metrix_config", {})
        settings["metrix_config"]["backend"] = None

    return settings


def _get_compare_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get compare configuration from framework config.

    Args:
        config: Framework config dict

    Returns:
        Dict with compare settings
    """
    return config.get("compare", {})


def _get_scheduler_config(
    config: Dict[str, Any],
    args,
    sched_overrides: Optional[Dict[str, Any]] = None,
) -> SchedulerConfig:
    """
    Get scheduler configuration from framework config, CLI args, and
    optional per-YAML scheduler overrides.

    Priority (highest → lowest):
        CLI args  >  kernel-config YAML overrides  >  framework config.yaml

    Args:
        config: Framework config dict
        args: CLI arguments
        sched_overrides: Scheduler overrides from kernel config YAML
            (may contain ``environment``, ``ray_config``, ``max_workers``, …)

    Returns:
        SchedulerConfig
    """
    sched_cfg = config.get("scheduler", {})
    overrides = sched_overrides or {}

    # Determine environment type: CLI > YAML override > framework config
    env_type_str = (
        getattr(args, "environment", None)
        or overrides.get("environment")
        or sched_cfg.get("environment", "local")
    )
    env_type = EnvironmentType(env_type_str.lower())

    # Get worker count
    max_workers = getattr(args, "workers", None) or sched_cfg.get("max_workers", 1)

    # Get GPU devices
    gpu_devices = sched_cfg.get("gpu_devices", [0])

    # Get docker image
    docker_image = getattr(args, "docker_image", None) or sched_cfg.get("docker_image")

    # Ray settings from YAML ray_config section
    ray_cfg = overrides.get("ray_config", {})
    ray_cluster_address = ray_cfg.get("cluster_address")
    ray_shared_storage_path = ray_cfg.get("shared_storage_path")

    return SchedulerConfig(
        environment_type=env_type,
        max_workers=max_workers,
        gpu_devices=gpu_devices,
        docker_image=docker_image,
        ray_cluster_address=ray_cluster_address,
        ray_shared_storage_path=ray_shared_storage_path,
    )


def run_analyze(args, config: Dict[str, Any]) -> int:
    """Run analyze mode."""
    # Load kernel configs
    kernel_configs = []
    perf_overrides: Dict[str, Any] = {}
    corr_overrides: Dict[str, Any] = {}
    sched_overrides: Dict[str, Any] = {}

    if args.kernel_config:
        # Load from kernel config file
        try:
            kernel_configs, perf_overrides, corr_overrides, sched_overrides = (
                load_kernel_config(args.kernel_config)
            )
        except ValueError as e:
            logger.error(f"Invalid kernel config: {e}")
            return 1
        if not kernel_configs:
            logger.error(f"No kernels found in {args.kernel_config}")
            return 1
    elif args.kernels:
        # Build from CLI arguments
        if not args.testcase:
            logger.error("Analyze mode requires --testcase")
            print("Error: --testcase is required for analyze mode")
            return 1

        try:
            kernel_type = parse_kernel_type(args.type)
        except ValueError as e:
            logger.error(str(e))
            return 1

        for path in args.kernels:
            if not path.exists():
                logger.error(f"Kernel not found: {path}")
                continue

            kernel_configs.append(
                KernelEvalConfig(
                    kernel_id=path.stem,
                    kernel_type=kernel_type,
                    source_file_path=[str(path)],
                    testcase_command=args.testcase.split(),
                    compiling_command=args.compile_cmd.split()
                    if args.compile_cmd
                    else None,
                    working_dir=str(path.parent),
                )
            )
    else:
        logger.error("No kernels specified")
        return 1

    # Get kernel type for configuration
    kernel_type = kernel_configs[0].kernel_type if kernel_configs else KernelType.HIP

    # Get config from framework config
    compile_settings = _get_compiling_config(config)
    perf_settings = _get_performance_config(config, kernel_type)
    corr_settings = _get_correctness_config(config)

    # Apply per-config overrides (from kernel config YAML)
    if perf_overrides:
        perf_settings = _apply_perf_overrides(perf_settings, perf_overrides, kernel_type)
    if corr_overrides:
        corr_settings = _apply_correctness_overrides(corr_settings, corr_overrides)

    # Create workspace before profiling so profiler writes directly there
    label = kernel_configs[0].kernel_id if kernel_configs else ""
    ws_path = _create_workspace(args.output_dir, "analyze", label)
    _save_config_snapshot(ws_path, kernel_configs)

    perf_dir = str(ws_path / "performance")
    perf_settings["rocprof_config"]["output_dir"] = perf_dir
    perf_settings["metrix_config"]["output_dir"] = perf_dir

    corr_settings["workspace_path"] = str(ws_path)

    scheduler_config = _get_scheduler_config(config, args, sched_overrides)
    scheduler = Scheduler(scheduler_config)

    if not scheduler.initialize():
        logger.error("Failed to initialize scheduler")
        return 1

    try:
        # Run analysis via scheduler
        result = scheduler.run_analyze(
            kernel_configs=kernel_configs,
            enable_default_compile=compile_settings["enable_default_compile"],
            check_performance=not args.no_perf,
            timeout_seconds=perf_settings["timeout_seconds"],
            profiler_args=perf_settings["profiler_args"],
            rocprof_config=perf_settings["rocprof_config"],
            ncu_config=perf_settings["ncu_config"],
            metrix_config=perf_settings["metrix_config"],
            correctness_config=corr_settings,
        )

        # Unwrap Ray result format: {'task_id': ..., 'results': [...]}
        raw = result.results
        if isinstance(raw, dict) and "results" in raw:
            eval_states = raw["results"]
        else:
            eval_states = raw

        # Print and save results
        if result.success and eval_states:
            for kernel_cfg, state in zip(kernel_configs, eval_states):
                if isinstance(state, dict):
                    state = _dict_to_eval_state(state)
                _print_result(kernel_cfg, state)

            _save_results(eval_states, ws_path, "analyze")
            print(f"\nWorkspace: {ws_path}")
        else:
            logger.error(f"Analysis failed: {result.errors}")
            return 1

        # Count failures
        failed = 0
        for state in eval_states:
            if isinstance(state, dict):
                correctness = state.get("correctness_state", "UNKNOWN")
                if correctness != "SUCCESS":
                    failed += 1
            elif hasattr(state, "correctness_state"):
                if state.correctness_state != BaseKind.SUCCESS:
                    failed += 1

        return 1 if failed > 0 else 0

    finally:
        scheduler.shutdown()


def run_compare(args, config: Dict[str, Any]) -> int:
    """Run compare mode."""
    # Load kernel configs
    kernel_configs = []
    perf_overrides: Dict[str, Any] = {}
    corr_overrides: Dict[str, Any] = {}
    sched_overrides: Dict[str, Any] = {}

    if args.kernel_config:
        try:
            kernel_configs, perf_overrides, corr_overrides, sched_overrides = (
                load_kernel_config(args.kernel_config)
            )
        except ValueError as e:
            logger.error(f"Invalid kernel config: {e}")
            return 1
    elif args.kernels:
        try:
            kernel_type = parse_kernel_type(args.type)
        except ValueError as e:
            logger.error(str(e))
            return 1

        for i, path in enumerate(args.kernels):
            if not path.exists():
                logger.error(f"Kernel not found: {path}")
                return 1

            kernel_configs.append(
                KernelEvalConfig(
                    kernel_id=f"kernel_{i}_{path.stem}",
                    kernel_type=kernel_type,
                    source_file_path=[str(path)],
                    testcase_command=args.testcase.split() if args.testcase else None,
                    working_dir=str(path.parent),
                )
            )

    if len(kernel_configs) < 2:
        logger.error("Compare mode requires at least 2 kernels")
        return 1

    # Get kernel type for configuration
    kernel_type = kernel_configs[0].kernel_type if kernel_configs else KernelType.HIP

    # Get config from framework config
    compile_settings = _get_compiling_config(config)
    perf_settings = _get_performance_config(config, kernel_type)
    corr_settings = _get_correctness_config(config)
    compare_settings = _get_compare_config(config)

    # Apply per-config overrides (from kernel config YAML)
    if perf_overrides:
        perf_settings = _apply_perf_overrides(perf_settings, perf_overrides, kernel_type)
    if corr_overrides:
        corr_settings = _apply_correctness_overrides(corr_settings, corr_overrides)

    # Create workspace before profiling so profiler writes directly there
    ws_path = _create_workspace(args.output_dir, "compare")
    _save_config_snapshot(ws_path, kernel_configs)

    perf_dir = str(ws_path / "performance")
    perf_settings["rocprof_config"]["output_dir"] = perf_dir
    perf_settings["metrix_config"]["output_dir"] = perf_dir

    corr_settings["workspace_path"] = str(ws_path)

    scheduler_config = _get_scheduler_config(config, args, sched_overrides)
    scheduler = Scheduler(scheduler_config)

    if not scheduler.initialize():
        logger.error("Failed to initialize scheduler")
        return 1

    try:
        # Run comparison via scheduler
        result = scheduler.run_compare(
            kernel_configs=kernel_configs,
            baseline_index=args.baseline,
            enable_default_compile=compile_settings["enable_default_compile"],
            check_performance=not args.no_perf,
            timeout_seconds=perf_settings["timeout_seconds"],
            profiler_args=perf_settings["profiler_args"],
            rocprof_config=perf_settings["rocprof_config"],
            ncu_config=perf_settings["ncu_config"],
            metrix_config=perf_settings["metrix_config"],
            correctness_config=corr_settings,
            compare_config=compare_settings,
        )

        # Print results
        if result.success and result.results:
            comparison = result.results

            print(f"\n{'=' * 60}")
            print("COMPARISON RESULTS")
            print(f"{'=' * 60}")

            summary_text = ""
            if isinstance(comparison, dict):
                summary_text = comparison.get("summary", "No summary available")
            elif hasattr(comparison, "summary"):
                summary_text = comparison.summary
            print(summary_text)

            print(f"{'=' * 60}\n")

            _save_comparison(comparison, ws_path)
            print(f"Workspace: {ws_path}")
        else:
            logger.error(f"Comparison failed: {result.errors}")
            return 1

        return 0

    finally:
        scheduler.shutdown()


def _dict_to_eval_state(state_dict: Dict[str, Any]) -> EvaluationState:
    """Convert a dictionary back to EvaluationState."""
    state = EvaluationState()

    if "compiling_state" in state_dict:
        state.compiling_state = BaseKind[state_dict["compiling_state"]]
    if "correctness_state" in state_dict:
        state.correctness_state = BaseKind[state_dict["correctness_state"]]
    if "performance_state" in state_dict:
        state.performance_state = BaseKind[state_dict["performance_state"]]
    if "score" in state_dict:
        state.score = state_dict["score"]
    if "errors" in state_dict:
        state.errors = state_dict["errors"]
    if "extra" in state_dict:
        state.extra = state_dict["extra"]

    return state


def _print_result(kernel_cfg: KernelEvalConfig, result: EvaluationState) -> None:
    """Print evaluation result."""
    print(f"\n{'=' * 60}")
    print(f"Kernel: {kernel_cfg.kernel_id}")
    print(f"Type: {kernel_cfg.kernel_type.name}")

    if isinstance(result, dict):
        print(f"Compiling: {result.get('compiling_state', 'UNKNOWN')}")
        print(f"Correctness: {result.get('correctness_state', 'UNKNOWN')}")
        print(f"Performance: {result.get('performance_state', 'UNKNOWN')}")
        print(f"Score: {result.get('score', 0.0):.2f}")
        errors = result.get("errors", [])
    else:
        print(f"Compiling: {result.compiling_state.name}")
        print(f"Correctness: {result.correctness_state.name}")
        print(f"Performance: {result.performance_state.name}")
        print(f"Score: {result.score:.2f}")
        errors = result.errors

    if errors:
        print("Errors:")
        for err in errors:
            print(f"  - {err}")
    print(f"{'=' * 60}")


def _create_workspace(
    base_dir: Path, mode: str, label: str = ""
) -> Path:
    """Create a timestamped workspace directory for analyze/compare results.

    Structure:
        <base_dir>/<mode>_<label>_<timestamp>/
            performance/
            config.yaml  (written later)
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label) if label else ""
    parts = [mode, safe_label, timestamp] if safe_label else [mode, timestamp]
    ws_name = "_".join(parts)

    ws_path = (base_dir / ws_name).resolve()
    ws_path.mkdir(parents=True, exist_ok=True)
    (ws_path / "performance").mkdir(exist_ok=True)

    logger.info(f"Created {mode} workspace: {ws_path}")
    return ws_path


def _save_config_snapshot(
    ws_path: Path, kernel_configs: List, extra: Optional[Dict[str, Any]] = None
) -> None:
    """Save a YAML config snapshot into the workspace."""
    snapshot: Dict[str, Any] = {"kernels": []}
    for cfg in kernel_configs:
        if hasattr(cfg, "to_dict"):
            snapshot["kernels"].append(cfg.to_dict())
        else:
            snapshot["kernels"].append(str(cfg))
    if extra:
        snapshot.update(extra)

    config_file = ws_path / "config.yaml"
    try:
        with open(config_file, "w") as f:
            yaml.dump(snapshot, f, default_flow_style=False)
    except Exception as e:
        logger.warning(f"Failed to save config snapshot: {e}")


def _save_results(results: List, ws_path: Path, mode: str) -> None:
    """Save analysis/compare results into the workspace as a JSON report."""
    import json
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = ws_path / f"{mode}_report.json"

    serialized_results: List[Any] = []
    for r in results:
        if isinstance(r, dict):
            serialized_results.append(r)
        elif hasattr(r, "to_dict"):
            serialized_results.append(r.to_dict())
        else:
            serialized_results.append(str(r))

    with open(report_file, "w") as f:
        json.dump(
            {"mode": mode, "timestamp": timestamp, "results": serialized_results},
            f,
            indent=2,
        )

    logger.info(f"Results saved to {report_file}")


def _save_comparison(comparison: Any, ws_path: Path) -> None:
    """Save comparison results into the workspace as a JSON report."""
    import json
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = ws_path / "compare_report.json"

    if isinstance(comparison, dict):
        comparison_data = comparison
    elif hasattr(comparison, "to_dict"):
        comparison_data = comparison.to_dict()
    else:
        comparison_data = {"result": str(comparison)}

    with open(report_file, "w") as f:
        json.dump(
            {
                "mode": "compare",
                "timestamp": timestamp,
                "results": {
                    "kernel_results": comparison_data.get("kernel_results", []),
                    "comparison_metrics": comparison_data.get("comparison_metrics", {}),
                    "winner": comparison_data.get("winner"),
                    "rankings": comparison_data.get("rankings", []),
                    "summary": comparison_data.get("summary", ""),
                },
            },
            f,
            indent=2,
        )

    logger.info(f"Comparison saved to {report_file}")


def load_benchmark_config(benchmark_config_path: Path) -> Dict[str, Any]:
    """
    Load benchmark configuration from YAML file.

    Returns:
        Dictionary with benchmark configuration
    """
    data = load_yaml(benchmark_config_path)
    return data.get("benchmark", {})


def run_gap_analysis_standalone(args) -> int:
    """Run standalone gap analysis on existing torch traces."""
    from .modes.benchmark.gap_analysis import GapAnalyzer
    from .modes.benchmark.config import GapAnalysisConfig

    trace_dir = args.trace_dir.resolve()

    # Auto-detect: if user passed a workspace dir, look for torch_trace/ inside
    if (trace_dir / "torch_trace").is_dir():
        trace_dir = trace_dir / "torch_trace"

    if not trace_dir.is_dir():
        print(f"Error: trace directory not found: {trace_dir}")
        return 1

    top_k = getattr(args, "top_k", 20)

    # Build config kwargs, only including non-None values to preserve defaults
    gap_kwargs = {
        "enabled": True,
        "trace_start_pct": args.start_pct,
        "trace_end_pct": args.end_pct,
        "top_k": top_k,
        "min_duration_us": args.min_duration_us,
    }
    categories = getattr(args, "categories", None)
    if categories is not None:
        gap_kwargs["categories"] = categories
    ignore_categories = getattr(args, "ignore_categories", None)
    if ignore_categories is not None:
        gap_kwargs["ignore_categories"] = ignore_categories

    gap_config = GapAnalysisConfig(**gap_kwargs)

    # All output goes into a gap_analysis/ subfolder
    base_dir = getattr(args, "output_dir", None) or trace_dir.parent
    gap_dir = Path(base_dir) / "gap_analysis"
    gap_dir.mkdir(parents=True, exist_ok=True)

    analyzer = GapAnalyzer(gap_config)
    result = analyzer.analyze(trace_dir)

    if result.errors:
        for err in result.errors:
            print(f"Warning: {err}")

    if not result.merged_kernels:
        print("No kernel events found in traces.")
        return 1

    # Get kernel source options
    find_kernel_sources = getattr(args, "find_kernel_sources", False)
    kernel_source_repos = getattr(args, "kernel_source_repos", None)

    result.to_csv(
        gap_dir / "gap_analysis.csv",
        find_kernel_sources=find_kernel_sources,
        kernel_source_repos=kernel_source_repos,
        auto_clone_repos=True,
    )
    if len(result.rank_results) > 1 and not getattr(args, "no_rank_csv", False):
        result.to_rank_csv(gap_dir)

    # Print summary
    start = gap_config.trace_start_pct
    end = gap_config.trace_end_pct
    print(f"\nGap Analysis ({start}%-{end}% window, category-filtered)")
    print(f"{'=' * 60}")
    print(f"Ranks analyzed: {len(result.rank_results)}")
    if gap_config.categories:
        print(f"Categories: {', '.join(gap_config.categories)}")
    if gap_config.ignore_categories:
        print(f"Ignore: {', '.join(gap_config.ignore_categories)}")
    print(f"Total duration: {result.total_duration_us:.2f} us")
    print(f"\n{'Name':50s} {'Calls':>6s} {'Self CUDA (us)':>15s} {'Avg (us)':>12s} {'% Total':>8s}")
    print("-" * 95)
    for k in result.merged_kernels:
        pct = (
            k.total_duration_us / result.total_duration_us * 100.0
            if result.total_duration_us > 0 else 0.0
        )
        name = k.name if len(k.name) <= 50 else k.name[:47] + "..."
        print(f"  {name:50s} {k.calls:6d} {k.total_duration_us:15.2f} {k.avg_us:12.2f} {pct:7.2f}%")
    print(f"\nOutput: {gap_dir}")

    return 0


def run_benchmark(args, config: Dict[str, Any]) -> int:
    """Run benchmark mode."""
    from .modes.benchmark import BenchmarkMode, BenchmarkConfig

    # Handle standalone gap-analysis (--trace-dir without framework)
    if getattr(args, "trace_dir", None) is not None:
        return run_gap_analysis_standalone(args)

    # Build benchmark config
    benchmark_cfg = {}
    
    if args.benchmark_config:
        # Load from config file
        benchmark_cfg = load_benchmark_config(args.benchmark_config)
        if not benchmark_cfg:
            logger.error(f"No benchmark config found in {args.benchmark_config}")
            return 1
    elif args.framework and args.model:
        # Build from CLI arguments
        benchmark_cfg = {
            "framework": args.framework,
            "model": args.model,
            "precision": args.precision,
            "envs": {
                "TP": args.tp,
                "CONC": args.concurrency,
                "ISL": args.input_len,
                "OSL": args.output_len,
                "RANDOM_RANGE_RATIO": 0.5,
            },
            "profiler": {
                "torch_profiler": {
                    "enabled": args.torch_profiler,
                },
                "system_profiler": {
                    "enabled": args.system_profiler,
                },
            },
            "docker_image": args.docker_image,
            "inferencex_path": args.inferencex_path,
            "benchmark_script": args.benchmark_script,
            "timeout_seconds": args.timeout,
        }
        if args.run_mode:
            benchmark_cfg["run_mode"] = args.run_mode
    else:
        logger.error("Benchmark mode requires either --benchmark-config or (framework and --model)")
        print("Error: Specify --benchmark-config or provide framework and --model")
        print("Example: python -m Magpie benchmark sglang --model meta-llama/Llama-2-7b-hf")
        return 1
    
    # CLI --run-mode overrides YAML config
    run_mode = getattr(args, "run_mode", None)
    if run_mode:
        benchmark_cfg["run_mode"] = run_mode
    
    # Get benchmark settings from framework config
    bench_settings = config.get("benchmark", {})
    
    # Merge with framework config defaults (auto-clone handled in BenchmarkMode)
    if "inferencex_path" not in benchmark_cfg or not benchmark_cfg["inferencex_path"]:
        benchmark_cfg["inferencex_path"] = bench_settings.get("inferencex_path", "")
    
    # Create benchmark config object
    try:
        benchmark_config = BenchmarkConfig.from_dict(benchmark_cfg)
    except Exception as e:
        logger.error(f"Invalid benchmark configuration: {e}")
        return 1
    
    # Run benchmark directly (benchmark mode handles its own Docker execution)
    logger.info(f"Starting benchmark: {benchmark_config.framework} / {benchmark_config.model}")
    
    try:
        benchmarker = BenchmarkMode(
            config=benchmark_config,
            output_dir=str(args.output_dir),
        )
        result = benchmarker.run()
        
        # Print results
        print(result.get_summary())
        
        if result.success:
            logger.info("Benchmark completed successfully")
            logger.info(f"Results saved to: {result.workspace_dir}")
            return 0
        else:
            logger.error(f"Benchmark failed: {result.errors}")
            return 1
            
    except Exception as e:
        logger.exception(f"Benchmark failed with error: {e}")
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Magpie: GPU Kernel Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Default config: use config.yaml relative to Magpie package
    default_config = Path(__file__).parent / "config.yaml"

    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=default_config,
        help="Framework configuration file",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--gpu-info", action="store_true", help="Show detected GPU info and exit"
    )
    parser.add_argument(
        "--environment",
        "-e",
        type=str,
        choices=["local", "container", "ray"],
        help="Execution environment (default: local)",
    )
    parser.add_argument(
        "--workers", "-w", type=int, help="Number of concurrent workers"
    )
    parser.add_argument(
        "--docker-image", type=str, help="Docker image for container environment"
    )

    subparsers = parser.add_subparsers(dest="mode", help="Evaluation mode")

    # Analyze subcommand
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze kernel(s) - requires testcase"
    )
    analyze_parser.add_argument(
        "kernels", type=Path, nargs="*", help="Kernel file(s) to analyze"
    )
    analyze_parser.add_argument(
        "--kernel-config", "-k", type=Path, help="Kernel configuration file"
    )
    analyze_parser.add_argument("--testcase", "-t", type=str, help="Testcase command")
    analyze_parser.add_argument(
        "--type",
        type=str,
        default="hip",
        choices=["hip", "cuda", "pytorch", "triton"],
        help="Kernel type",
    )
    analyze_parser.add_argument(
        "--compile-cmd", type=str, help="Custom compile command"
    )
    analyze_parser.add_argument(
        "--no-perf", action="store_true", help="Skip performance profiling"
    )
    analyze_parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("./results"),
        help="Output directory",
    )

    # Compare subcommand
    compare_parser = subparsers.add_parser("compare", help="Compare multiple kernels")
    compare_parser.add_argument(
        "kernels", type=Path, nargs="*", help="Kernel files to compare"
    )
    compare_parser.add_argument(
        "--kernel-config", "-k", type=Path, help="Kernel configuration file"
    )
    compare_parser.add_argument(
        "--testcase", "-t", type=str, help="Testcase command (optional)"
    )
    compare_parser.add_argument(
        "--type",
        type=str,
        default="hip",
        choices=["hip", "cuda", "pytorch", "triton"],
        help="Kernel type",
    )
    compare_parser.add_argument(
        "--baseline", type=int, default=0, help="Baseline kernel index"
    )
    compare_parser.add_argument(
        "--no-perf", action="store_true", help="Skip performance profiling"
    )
    compare_parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("./results"),
        help="Output directory",
    )

    # Benchmark subcommand
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Run framework benchmark (vLLM/SGLang)"
    )
    benchmark_parser.add_argument(
        "framework",
        type=str,
        nargs="?",
        choices=["vllm", "sglang"],
        help="Framework to benchmark",
    )
    benchmark_parser.add_argument(
        "--benchmark-config", "-b", type=Path, help="Benchmark configuration file"
    )
    benchmark_parser.add_argument(
        "--model", "-m", type=str, help="Model name or path"
    )
    benchmark_parser.add_argument(
        "--precision", "-p", type=str, default="fp8",
        choices=["fp8", "fp16", "bf16", "fp4"],
        help="Model precision (default: fp8)"
    )
    benchmark_parser.add_argument(
        "--tp", type=int, default=1, help="Tensor parallel size"
    )
    benchmark_parser.add_argument(
        "--concurrency", type=int, default=32, help="Request concurrency"
    )
    benchmark_parser.add_argument(
        "--input-len", type=int, default=1024, help="Input sequence length"
    )
    benchmark_parser.add_argument(
        "--output-len", type=int, default=512, help="Output sequence length"
    )
    benchmark_parser.add_argument(
        "--torch-profiler", action="store_true",
        help="Enable torch profiler"
    )
    benchmark_parser.add_argument(
        "--system-profiler", action="store_true",
        help="Enable system profiler (rocprof/ncu)"
    )
    benchmark_parser.add_argument(
        "--run-mode", type=str, default=None,
        choices=["docker", "local"],
        help="Execution mode: 'docker' (default) runs inside a container; "
             "'local' runs directly on the host (useful inside pods/containers)"
    )
    benchmark_parser.add_argument(
        "--docker-image", type=str, help="Override Docker image"
    )
    benchmark_parser.add_argument(
        "--inferencex-path", type=str,
        default="",
        help="Path to InferenceX installation (auto-cloned if not specified)"
    )
    benchmark_parser.add_argument(
        "--benchmark-script", type=str,
        help="InferenceX benchmark script name"
    )
    benchmark_parser.add_argument(
        "--timeout", type=int, default=3600, help="Benchmark timeout in seconds"
    )
    benchmark_parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("./results"),
        help="Output directory",
    )

    # Standalone gap-analysis on existing traces (replaces sub-subparser to
    # avoid argparse conflict with the positional `framework` argument)
    benchmark_parser.add_argument(
        "--trace-dir", type=Path, default=None,
        help="Run standalone gap analysis on existing traces instead of a "
             "benchmark. Pass the path to a torch_trace directory (or a "
             "benchmark workspace containing one).",
    )
    benchmark_parser.add_argument(
        "--top-k", type=int, default=20,
        help="Gap analysis: number of top bottleneck kernels (default: 20)",
    )
    benchmark_parser.add_argument(
        "--start-pct", type=float, default=0.0,
        help="Gap analysis: start of window (0-100, default: 0)",
    )
    benchmark_parser.add_argument(
        "--end-pct", type=float, default=100.0,
        help="Gap analysis: end of window (0-100, default: 100)",
    )
    benchmark_parser.add_argument(
        "--min-duration-us", type=float, default=0.0,
        help="Gap analysis: minimum event duration in microseconds (default: 0)",
    )
    benchmark_parser.add_argument(
        "--categories", type=str, nargs="*", default=None,
        help="Gap analysis: event categories to include (e.g. kernel gpu). None = all.",
    )
    benchmark_parser.add_argument(
        "--ignore-categories", type=str, nargs="*", default=None,
        help="Gap analysis: event categories to exclude (e.g. gpu_user_annotation)",
    )
    benchmark_parser.add_argument(
        "--no-rank-csv", action="store_true",
        help="Gap analysis: skip generating per-rank CSV files",
    )
    benchmark_parser.add_argument(
        "--find-kernel-sources", action="store_true",
        help="Gap analysis: find kernel source files and test commands (AMD kernels)",
    )
    benchmark_parser.add_argument(
        "--kernel-source-repos", type=str, nargs="*", default=None,
        help="Gap analysis: repository paths to search for kernel sources",
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle --gpu-info
    if args.gpu_info:
        info = get_gpu_info()
        print("GPU Information:")
        print(f"  Vendor: {info['vendor']}")
        print(f"  Architecture: {info['architecture']}")
        if info["detected"]:
            print(f"  Compiler: {info.get('compiler', 'N/A')}")
            print(f"  Profiler: {info.get('profiler', 'N/A')}")
        else:
            print("  (No GPU detected)")
        return 0

    if not args.mode:
        parser.print_help()
        return 0

    # Load framework config
    config = load_yaml(args.config)
    setup_logging(config, args.verbose)

    if args.mode == "analyze":
        return run_analyze(args, config)
    elif args.mode == "compare":
        return run_compare(args, config)
    elif args.mode == "benchmark":
        return run_benchmark(args, config)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
