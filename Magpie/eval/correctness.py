###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Correctness evaluation module.

This module handles correctness verification of GPU kernels.

- Analyze mode: Run testcase command and check pass/fail
- Compare mode: Compare outputs between multiple kernels
  - For PyTorch: Generate test inputs and compare outputs
  - For HIP/CUDA: Run kernels and compare stdout/results
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import random
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, List, Optional, TYPE_CHECKING

from ..utils import get_updated_env
from ..config import (
    EvalMode,
    KernelType,
    PipelineConfig,
    KernelEvalConfig,
    CorrectnessConfig,
    CorrectnessBackend,
    AccordoConfig,
)

# Torch is imported lazily to avoid initializing HIP/CUDA at module load time,
# which would break Accordo's HSA interception (IPC memory handles).
HAS_TORCH = False
torch = None


def _ensure_torch():
    global HAS_TORCH, torch
    if torch is not None or HAS_TORCH:
        return
    try:
        import torch as _torch

        torch = _torch
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =========================
# ------- Data Models -----
# =========================


@dataclass
class MetricResult:
    """A single metric outcome for one comparison."""

    name: str
    success: bool
    value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class CorrectnessResult:
    """Result of correctness evaluation."""

    success: bool
    metrics: List[MetricResult] = field(default_factory=list)
    errors: Optional[str] = None


# =========================
# ------- Correctness -----
# =========================


class Correctness:
    """
    Correctness evaluation orchestrator.

    - Analyze mode: Runs testcase command
    - Compare mode: Compares outputs between kernels
    """

    def __init__(self, pipeline_cfg: PipelineConfig) -> None:
        """
        Initialize correctness evaluator.

        Args:
            pipeline_cfg: Pipeline configuration
        """
        self.pipeline_cfg = pipeline_cfg
        self.corr_cfg: CorrectnessConfig = (
            pipeline_cfg.correctness_config or CorrectnessConfig()
        )

    def run(self, eval_state: Any, kernel_cfg: KernelEvalConfig) -> CorrectnessResult:
        """
        Run correctness evaluation.

        Args:
            eval_state: Current evaluation state
            kernel_cfg: Kernel configuration

        Returns:
            CorrectnessResult with pass/fail status
        """
        try:
            if self.corr_cfg.backend == CorrectnessBackend.ACCORDO:
                return self._run_accordo_mode(eval_state, kernel_cfg)

            mode = self.pipeline_cfg.mode

            if mode == EvalMode.ANALYZE:
                return self._run_analyze_mode(eval_state, kernel_cfg)
            elif mode == EvalMode.COMPARE:
                return self._run_compare_mode(eval_state, kernel_cfg)
            else:
                return CorrectnessResult(
                    success=False, errors=f"Unknown evaluation mode: {mode}"
                )

        except Exception as e:
            logger.error(f"Correctness evaluation failed: {e}")
            return CorrectnessResult(success=False, errors=str(e))

    def _run_analyze_mode(
        self, eval_state: Any, kernel_cfg: KernelEvalConfig
    ) -> CorrectnessResult:
        """
        Run analyze mode - execute testcase command.

        Requires testcase_command to be provided.
        """
        if not kernel_cfg.has_testcase():
            return CorrectnessResult(
                success=False, errors="Analyze mode requires testcase_command"
            )

        return self._run_testcase(kernel_cfg)

    def _run_compare_mode(
        self, eval_state: Any, kernel_cfg: KernelEvalConfig
    ) -> CorrectnessResult:
        """
        Run compare mode - compare outputs.

        If testcase command is provided, use it.
        Otherwise, for PyTorch kernels, generate inputs and compare outputs.
        """
        # If testcase is provided, use it
        if kernel_cfg.has_testcase():
            return self._run_testcase(kernel_cfg)

        # Otherwise, for PyTorch, generate inputs and compare
        if kernel_cfg.kernel_type == KernelType.PYTORCH:
            return self._run_pytorch_comparison(eval_state, kernel_cfg)

        # Triton kernels: run script directly, check exit code and PASS
        if kernel_cfg.kernel_type == KernelType.TRITON:
            return self._run_triton_comparison(eval_state, kernel_cfg)

        # For HIP/CUDA without testcase, we need exec to compare outputs
        return CorrectnessResult(
            success=False,
            errors="Compare mode for HIP/CUDA requires testcase_command or executable comparison",
        )

    def _run_testcase(self, kernel_cfg: KernelEvalConfig) -> CorrectnessResult:
        """
        Run testcase command(s) and check result.

        Supports both single command and multiple commands executed in order.
        """
        commands = kernel_cfg.get_testcase_commands()
        working_dir = kernel_cfg.working_dir
        env = get_updated_env(kernel_cfg.env)

        n_iter = self.corr_cfg.iteration_count or 1
        all_metrics = []

        try:
            for i in range(n_iter):
                for cmd_idx, cmd in enumerate(commands):
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, env=env, cwd=working_dir
                    )

                    if result.returncode != 0:
                        errors = result.stderr if result.stderr else result.stdout
                        all_metrics.append(
                            MetricResult(
                                name=f"testcase_cmd{cmd_idx + 1}",
                                success=False,
                                value=float(result.returncode),
                            )
                        )
                        cmd_info = (
                            f"command {cmd_idx + 1}/{len(commands)}"
                            if len(commands) > 1
                            else "command"
                        )
                        return CorrectnessResult(
                            success=False,
                            metrics=all_metrics,
                            errors=f"Testcase {cmd_info} failed (iteration {i + 1}): {errors}",
                        )

                    all_metrics.append(
                        MetricResult(
                            name=f"testcase_cmd{cmd_idx + 1}_iter{i + 1}", success=True
                        )
                    )

            return CorrectnessResult(
                success=True, metrics=[MetricResult(name="testcase", success=True)]
            )

        except Exception as e:
            return CorrectnessResult(success=False, errors=str(e))

    def _run_pytorch_comparison(
        self, eval_state: Any, kernel_cfg: KernelEvalConfig
    ) -> CorrectnessResult:
        """
        Run PyTorch kernel comparison.

        Generate test inputs using get_inputs function and compare outputs.
        """
        _ensure_torch()
        if not HAS_TORCH:
            return CorrectnessResult(
                success=False, errors="PyTorch not available for comparison"
            )

        source_files = kernel_cfg.get_source_file_paths()
        if not source_files:
            return CorrectnessResult(success=False, errors="No source files provided")

        try:
            # Load the module
            module = self._load_module(source_files[0])

            # Get input generation function
            get_inputs_fn = getattr(module, kernel_cfg.get_inputs_func, None)
            if get_inputs_fn is None:
                return CorrectnessResult(
                    success=False,
                    errors=f"Function '{kernel_cfg.get_inputs_func}' not found in module",
                )

            # Get model classes (assuming Model and NewModel pattern)
            model_class = getattr(module, "Model", None)

            if model_class is None:
                return CorrectnessResult(
                    success=False, errors="Model class not found in module"
                )

            # Generate inputs and run comparison
            n_iter = self.corr_cfg.iteration_count or 1
            all_metrics = []

            for i in range(n_iter):
                seed = random.randint(1, 10000)
                torch.manual_seed(seed)

                # Generate inputs
                inputs = get_inputs_fn()
                if not isinstance(inputs, (list, tuple)):
                    inputs = [inputs]

                # Move to GPU if available
                if torch.cuda.is_available():
                    inputs = [
                        inp.cuda() if isinstance(inp, torch.Tensor) else inp
                        for inp in inputs
                    ]

                # Get init inputs if available
                get_init_inputs_fn = getattr(
                    module, kernel_cfg.get_init_inputs_func, None
                )
                init_inputs = get_init_inputs_fn() if get_init_inputs_fn else []

                # Create and run model
                if init_inputs:
                    model = model_class(*init_inputs)
                else:
                    model = model_class()

                if torch.cuda.is_available():
                    model = model.cuda()

                model.eval()
                with torch.no_grad():
                    output = model(*inputs)

                # Check for NaN/Inf
                if self.corr_cfg.check_nan and torch.isnan(output).any():
                    all_metrics.append(MetricResult(name="nan_check", success=False))
                    return CorrectnessResult(
                        success=False,
                        metrics=all_metrics,
                        errors=f"NaN detected in output (iteration {i + 1})",
                    )

                if self.corr_cfg.check_inf and torch.isinf(output).any():
                    all_metrics.append(MetricResult(name="inf_check", success=False))
                    return CorrectnessResult(
                        success=False,
                        metrics=all_metrics,
                        errors=f"Inf detected in output (iteration {i + 1})",
                    )

                all_metrics.append(MetricResult(name=f"run_{i + 1}", success=True))

            return CorrectnessResult(success=True, metrics=all_metrics)

        except Exception as e:
            return CorrectnessResult(success=False, errors=str(e))

    def _run_triton_comparison(
        self, eval_state: Any, kernel_cfg: KernelEvalConfig
    ) -> CorrectnessResult:
        """
        Run Triton kernel correctness check.

        Triton kernels are standalone .py scripts with a __main__ block that:
        - Runs the kernel with test inputs
        - Compares output against a reference (baseline or torch)
        - Prints "PASS" on success
        - Exits with code 0 on success, non-zero on failure

        This method runs each source file and checks for success.
        """
        source_files = kernel_cfg.get_source_file_paths()
        if not source_files:
            return CorrectnessResult(success=False, errors="No source files provided")

        working_dir = kernel_cfg.working_dir
        env = get_updated_env(kernel_cfg.env)
        n_iter = self.corr_cfg.iteration_count or 1
        all_metrics = []

        try:
            import sys as _sys

            python_bin = _sys.executable or "python3"

            for source_file in source_files:
                abs_source = str(Path(source_file).resolve())
                for i in range(n_iter):
                    result = subprocess.run(
                        [python_bin, abs_source],
                        capture_output=True,
                        text=True,
                        env=env,
                        cwd=working_dir,
                        timeout=120,
                    )

                    if result.returncode != 0:
                        all_metrics.append(
                            MetricResult(
                                name=f"triton_run_{i + 1}",
                                success=False,
                                value=float(result.returncode),
                            )
                        )
                        stderr_snippet = (result.stderr or result.stdout)[:500]
                        return CorrectnessResult(
                            success=False,
                            metrics=all_metrics,
                            errors=f"Triton kernel failed (iteration {i + 1}): {stderr_snippet}",
                        )

                    stdout_lower = result.stdout.lower()
                    if "fail" in stdout_lower or "error" in stdout_lower:
                        if "pass" not in stdout_lower:
                            all_metrics.append(
                                MetricResult(
                                    name=f"triton_run_{i + 1}", success=False
                                )
                            )
                            return CorrectnessResult(
                                success=False,
                                metrics=all_metrics,
                                errors=f"Triton kernel reported failure (iteration {i + 1}): {result.stdout[:500]}",
                            )

                    all_metrics.append(
                        MetricResult(name=f"triton_run_{i + 1}", success=True)
                    )

            return CorrectnessResult(success=True, metrics=all_metrics)

        except subprocess.TimeoutExpired:
            return CorrectnessResult(
                success=False, errors="Triton kernel timed out (120s)"
            )
        except Exception as e:
            return CorrectnessResult(success=False, errors=str(e))

    def _run_accordo_mode(
        self, eval_state: Any, kernel_cfg: KernelEvalConfig
    ) -> CorrectnessResult:
        """
        Run Accordo HSA-level correctness validation via the ``accordo`` CLI.

        Invokes ``accordo validate`` as a subprocess so that HSA interception,
        signal-based timeouts, and ``os.chdir`` are fully isolated from the
        Magpie host process (important for Ray / multi-worker execution).
        """
        if not shutil.which("accordo"):
            return CorrectnessResult(
                success=False,
                errors=(
                    "'accordo' CLI not found on PATH. Install IntelliKit Accordo "
                    '(pip install -e ".[accordo]" from a Magpie checkout, or '
                    'pip install "magpie-eval[accordo] @ '
                    'git+https://github.com/AMD-AGI/Magpie.git@<tag-or-sha>").'
                ),
            )

        accordo_cfg: AccordoConfig = self.corr_cfg.accordo_config or AccordoConfig()

        kernel_name = accordo_cfg.kernel_name
        if not kernel_name:
            return CorrectnessResult(
                success=False,
                errors="Accordo backend requires 'kernel_name' to be set in accordo config.",
            )

        ref_binary = accordo_cfg.reference_binary
        opt_binary = accordo_cfg.optimized_binary
        if not ref_binary or not opt_binary:
            return CorrectnessResult(
                success=False,
                errors="Accordo backend requires both 'reference_binary' and "
                "'optimized_binary' to be set in accordo config.",
            )

        perf_timeout = getattr(
            self.pipeline_cfg.performance_config, "timeout_seconds", None
        )
        timeout = accordo_cfg.timeout_seconds or perf_timeout or 300

        binary_working_dir = (
            accordo_cfg.working_directory or kernel_cfg.working_dir or "."
        )
        binary_working_dir = str(Path(binary_working_dir).resolve())

        workspace_dir = accordo_cfg.workspace_path or binary_working_dir

        cmd = [
            "accordo", "validate",
            "--kernel-name", kernel_name,
            "--ref-binary", ref_binary,
            "--opt-binary", opt_binary,
            "--atol", str(accordo_cfg.atol),
            "--rtol", str(accordo_cfg.rtol),
            "--timeout", str(int(timeout)),
            "--working-dir", binary_working_dir,
            "--log-level", "INFO",
        ]
        if accordo_cfg.equal_nan:
            cmd.extend(["--equal-nan"])

        if accordo_cfg.kernel_args:
            args_str = ",".join(
                f"{name}:{typ}" for name, typ in accordo_cfg.kernel_args
            )
            cmd.extend(["--kernel-args", args_str])

        logger.info("Accordo CLI: %s (cwd=%s)", " ".join(cmd), workspace_dir)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=workspace_dir,
                timeout=int(timeout) * 2 + 60,
            )
        except subprocess.TimeoutExpired:
            return CorrectnessResult(
                success=False,
                errors=f"Accordo CLI timed out after {int(timeout) * 2 + 60}s",
            )
        except FileNotFoundError:
            return CorrectnessResult(
                success=False,
                errors="'accordo' CLI not found. Is it installed?",
            )

        if proc.returncode != 0:
            stdout = proc.stdout.strip()
            try:
                err_data = json.loads(stdout)
                error_msg = err_data.get("error", stdout)
            except (json.JSONDecodeError, ValueError):
                error_msg = stdout or proc.stderr.strip()
            return CorrectnessResult(
                success=False, errors=f"Accordo error: {error_msg}"
            )

        try:
            raw = proc.stdout.strip()
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            if json_start < 0 or json_end <= json_start:
                raise ValueError("No JSON object found in output")
            output = json.loads(raw[json_start:json_end])
        except (json.JSONDecodeError, ValueError):
            return CorrectnessResult(
                success=False,
                errors=f"Accordo CLI returned invalid JSON: {proc.stdout[:500]}",
            )

        is_valid = output.get("is_valid", False)
        metrics = []

        if is_valid:
            metrics.append(
                MetricResult(
                    name="accordo_validation",
                    success=True,
                    value=float(output.get("num_arrays_validated", 0)),
                )
            )
            for arr_name in output.get("matched_arrays", {}):
                metrics.append(
                    MetricResult(
                        name=f"accordo_array_{arr_name}",
                        success=True,
                        threshold=accordo_cfg.atol,
                    )
                )
        else:
            metrics.append(
                MetricResult(
                    name="accordo_validation",
                    success=False,
                    value=float(output.get("num_mismatches", 0)),
                )
            )
            for mismatch in output.get("mismatches", []):
                argn = mismatch.get("arg_name", "?")
                disp_idx = mismatch.get("dispatch_index")
                mm_name = (
                    f"accordo_mismatch_d{disp_idx}_{argn}"
                    if disp_idx is not None
                    else f"accordo_mismatch_{argn}"
                )
                metrics.append(
                    MetricResult(
                        name=mm_name,
                        success=False,
                        value=mismatch.get("max_difference"),
                        threshold=accordo_cfg.atol,
                    )
                )

        return CorrectnessResult(
            success=is_valid,
            metrics=metrics,
            errors=output.get("error_message") if not is_valid else None,
        )

    def _load_module(self, file_path: str) -> ModuleType:
        """Load a Python module from file."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        salt = hashlib.md5(str(p.resolve()).encode()).hexdigest()[:8]
        mod_name = f"_dyn_module_{p.stem}_{salt}"

        spec = importlib.util.spec_from_file_location(mod_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
