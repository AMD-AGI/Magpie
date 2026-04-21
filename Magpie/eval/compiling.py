###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Kernel compilation module.

This module handles compilation of GPU kernels based on kernel type.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

from ..config import KernelType, PipelineConfig, KernelEvalConfig
from ..utils.common import compile_hip, get_compilation_output_stem, get_updated_env


@dataclass
class CompilingResult:
    """
    Result of kernel compilation.

    Attributes:
        success: Whether compilation succeeded
        output_file_path: Path to compiled output file
        so_file_path: Path to shared object file (if applicable)
        errors: Error message if compilation failed
    """

    success: bool
    output_file_path: Optional[str] = None
    so_file_path: Optional[str] = None
    errors: Optional[str] = None


class Compiling:
    """
    Kernel compilation handler.

    Supports different kernel types:
    - HIP: Compile using hipcc
    - CUDA: Compile using nvcc
    - PyTorch: No compilation needed (Python module)
    """

    def __init__(self, pipeline_cfg: PipelineConfig) -> None:
        """
        Initialize the compiler.

        Args:
            pipeline_cfg: Pipeline configuration
        """
        self.pipeline_cfg = pipeline_cfg

    def run(self, kernel_cfg: KernelEvalConfig) -> Optional[CompilingResult]:
        """
        Compile the kernel code based on the provided configuration.

        Args:
            kernel_cfg: Configuration for the kernel evaluation

        Returns:
            CompilingResult with success status and compiled output paths,
            or None if compilation should be skipped
        """
        try:
            kernel_type = kernel_cfg.kernel_type

            # PyTorch and Triton kernels never need ahead-of-time compilation
            if kernel_type in (KernelType.PYTORCH, KernelType.TRITON):
                return None

            # If custom compile command is provided, use it
            if kernel_cfg.has_compile_command():
                return self._compile_with_command(kernel_cfg)

            # No compile_command provided - check if default compile is enabled
            compiling_config = self.pipeline_cfg.compiling_config
            if compiling_config and compiling_config.enable_default_compile:
                # Default compile enabled - compile based on kernel type
                if kernel_type == KernelType.HIP:
                    return self._compile_hip(kernel_cfg)
                elif kernel_type == KernelType.CUDA:
                    return self._compile_cuda(kernel_cfg)

            # Default compile disabled or unsupported type - skip compilation
            # (assume pre-compiled binary exists)
            return None

        except Exception as e:
            return CompilingResult(success=False, errors=str(e))

    def _compile_with_command(self, kernel_cfg: KernelEvalConfig) -> CompilingResult:
        """
        Compile the kernel using custom command(s).

        Supports both single command and multiple commands executed in order.
        """
        if not kernel_cfg.compiling_command:
            raise ValueError("No compiling command provided")

        commands = kernel_cfg.get_compile_commands()
        working_dir = kernel_cfg.working_dir or tempfile.mkdtemp(prefix="kerneleval_")
        env = get_updated_env(kernel_cfg.env)

        try:
            for i, cmd in enumerate(commands):
                result = subprocess.run(
                    cmd, capture_output=True, text=True, env=env, cwd=working_dir
                )
                if result.returncode != 0:
                    errors = result.stderr if result.stderr else result.stdout
                    return CompilingResult(
                        success=False,
                        errors=f"Compile command {i + 1}/{len(commands)} failed: {errors}",
                    )

            return CompilingResult(success=True)
        except Exception as e:
            return CompilingResult(success=False, errors=str(e))

    def _compile_hip(self, kernel_cfg: KernelEvalConfig) -> CompilingResult:
        """
        Compile HIP kernel code.
        """
        if shutil.which("hipcc") is None:
            return CompilingResult(
                success=False,
                errors="hipcc not found. Please install ROCm HIP compiler.",
            )

        working_dir = kernel_cfg.working_dir or tempfile.mkdtemp(prefix="kerneleval_")
        source_files = kernel_cfg.get_source_file_paths()
        env = get_updated_env(kernel_cfg.env)
        gpu_arch = self.pipeline_cfg.gpu_arch
        if gpu_arch is None:
            raise ValueError(
                "gpu_arch is not set; please configure pipeline_cfg.gpu_arch"
            )

        try:
            out_file_path, _, errors = compile_hip(
                hip_file_path=source_files,
                working_dir=working_dir,
                gpu_arch=gpu_arch,
                with_so=False,
                env=env,
            )

            if errors:
                return CompilingResult(success=False, errors=errors)

            return CompilingResult(success=True, output_file_path=out_file_path)

        except Exception as e:
            return CompilingResult(success=False, errors=str(e))

    def _compile_cuda(self, kernel_cfg: KernelEvalConfig) -> CompilingResult:
        """
        Compile CUDA kernel code.
        """
        if shutil.which("nvcc") is None:
            return CompilingResult(
                success=False, errors="nvcc not found. Please install CUDA toolkit."
            )

        working_dir = kernel_cfg.working_dir or tempfile.mkdtemp(prefix="kerneleval_")
        source_files = kernel_cfg.get_source_file_paths()
        env = get_updated_env(kernel_cfg.env)

        base_name = get_compilation_output_stem(source_files)
        out_file_path = os.path.join(working_dir, base_name + ".out")

        try:
            cmd = ["nvcc", "-O2", "-std=c++17", *source_files, "-o", out_file_path]

            result = subprocess.run(
                cmd, capture_output=True, text=True, env=env, cwd=working_dir
            )

            if result.returncode != 0:
                errors = result.stderr if result.stderr else result.stdout
                return CompilingResult(success=False, errors=errors)

            return CompilingResult(success=True, output_file_path=out_file_path)

        except Exception as e:
            return CompilingResult(success=False, errors=str(e))
