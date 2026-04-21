###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Common utility functions for kernel evaluation.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple


def get_updated_env(env: Optional[Dict[str, str]]) -> Dict[str, str]:
    """
    Merge custom environment variables with os.environ.
    For PATH-like variables, prepend custom values to existing ones.

    Args:
        env: Custom environment variables to merge

    Returns:
        Merged environment dictionary
    """
    # Start with a copy of system environment variables
    updated_env = os.environ.copy()

    if env is not None:
        for key, value in env.items():
            # For PATH-like variables, prepend to existing value
            if key.upper() in ["LD_LIBRARY_PATH", "PATH", "PYTHONPATH", "LIBRARY_PATH"]:
                existing_value = updated_env.get(key, "")
                if existing_value:
                    updated_env[key] = f"{value}:{existing_value}"
                else:
                    updated_env[key] = value
            else:
                # For other variables, simply override
                updated_env[key] = value

    return updated_env


def get_compilation_output_stem(source_file_paths: List[str]) -> str:
    """Create a deterministic output stem from one or more source paths."""
    if len(source_file_paths) == 1:
        return os.path.splitext(os.path.basename(source_file_paths[0]))[0]

    source_manifest = "\0".join(source_file_paths)
    digest = hashlib.blake2b(
        source_manifest.encode("utf-8"), digest_size=8
    ).hexdigest()
    return f"kernel_{digest}"


def compile_hip(
    hip_file_path: List[str],
    working_dir: str,
    gpu_arch: str,
    with_so: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Compile HIP kernel code and return file paths and errors.

    Args:
        hip_file_path: List of HIP source file paths
        working_dir: Working directory for compilation
        gpu_arch: Target GPU architecture (e.g., "gfx90a")
        with_so: Whether to also create a shared object
        env: Environment variables for compilation

    Returns:
        Tuple of (out_file_path, so_file_path, errors)
        - out_file_path: Path to compiled binary, None on failure
        - so_file_path: Path to shared object, None if not requested or on failure
        - errors: Error message if compilation failed, None on success
    """
    if shutil.which("hipcc") is None:
        raise RuntimeError("hipcc not found. Please install ROCm HIP compiler.")

    source_file_name = get_compilation_output_stem(hip_file_path)

    out_file_path = os.path.join(working_dir, source_file_name + ".out")
    so_file_path = (
        os.path.join(working_dir, source_file_name + ".so") if with_so else None
    )

    if env is None:
        env = os.environ.copy()

    try:
        # Compile to binary
        out_cmd = [
            "hipcc",
            "-O2",
            "-std=c++17",
            f"--offload-arch={gpu_arch}",
            "-g",
            *hip_file_path,
            "-o",
            out_file_path,
        ]

        bin_result = subprocess.run(
            out_cmd, capture_output=True, text=True, env=env, cwd=working_dir
        )

        if bin_result.returncode != 0:
            errors = bin_result.stderr if bin_result.stderr else bin_result.stdout
            return None, None, errors

        # Optionally compile to shared object
        if with_so and so_file_path:
            so_cmd = ["hipcc", "-shared", "-fPIC", *hip_file_path, "-o", so_file_path]
            so_result = subprocess.run(
                so_cmd, capture_output=True, text=True, env=env, cwd=working_dir
            )
            if so_result.returncode != 0:
                errors = so_result.stderr if so_result.stderr else so_result.stdout
                return out_file_path, None, errors

        # Compilation successful
        return out_file_path, so_file_path, None

    except Exception as e:
        return None, None, str(e)
