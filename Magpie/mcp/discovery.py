###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Kernel discovery helpers for the Magpie MCP server.

This module keeps project scanning and Triton kernel detection separate from
the MCP transport layer so the logic is easier to test and maintain.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any, Optional

# Directories to always skip during kernel discovery.
SKIP_DIRS = frozenset(
    {
        ".git",
        ".svn",
        ".hg",
        "node_modules",
        "__pycache__",
        ".cache",
        "venv",
        ".venv",
        "env",
        ".env",
        ".tox",
        ".nox",
        ".pytest_cache",
        "dist",
        "third_party",
        "external",
        "deps",
        "vendor",
    }
)

# Build-like directories often contain generated sources. Keep skipping them for
# HIP/CUDA discovery, but allow Triton discovery so generated Triton kernels can
# still be surfaced.
BUILD_LIKE_DIRS = frozenset(
    {
        "build",
        "bin",
        "out",
        "results",
        "cmake-build-release",
        "cmake-build-debug",
    }
)

_TRITON_DECORATOR_NAMES = frozenset({"jit", "autotune", "heuristics"})
_TRITON_DECORATOR_MARKERS = (
    ".jit",
    "@jit",
    ".autotune",
    "@autotune",
    ".heuristics",
    "@heuristics",
)


def _decorator_root_name(node: ast.AST) -> Optional[str]:
    """Return the root name for a decorator expression like triton.jit."""
    current = node
    while isinstance(current, ast.Attribute):
        current = current.value
    if isinstance(current, ast.Name):
        return current.id
    return None


def _should_descend_into_dir(dir_name: str, kernel_type: str) -> bool:
    """Return whether discovery should recurse into a directory."""
    if dir_name.startswith(".") or dir_name in SKIP_DIRS:
        return False
    if dir_name in BUILD_LIKE_DIRS:
        return kernel_type in ("triton", "all")
    return True


def _is_build_like_path(path: Path) -> bool:
    """Return whether a relative path contains a build-like directory."""
    return any(part in BUILD_LIKE_DIRS for part in path.parts)


def is_triton_kernel_file(source_file: Path) -> bool:
    """
    Heuristically detect Triton Python source files.

    We first do a cheap textual scan, then use the AST to confirm that the file
    defines at least one function decorated with Triton kernel decorators.
    """
    try:
        source = source_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False

    if not any(marker in source for marker in _TRITON_DECORATOR_MARKERS):
        return False

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Keep discovery useful for broken work-in-progress kernels: if the file
        # clearly contains Triton decorators, still surface it as a candidate.
        return True

    triton_aliases = set()
    triton_decorator_aliases = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "triton":
                    triton_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "triton":
            for alias in node.names:
                if alias.name == "*":
                    triton_decorator_aliases.update(_TRITON_DECORATOR_NAMES)
                elif alias.name in _TRITON_DECORATOR_NAMES:
                    triton_decorator_aliases.add(alias.asname or alias.name)

    if not triton_aliases and not triton_decorator_aliases:
        return False

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        for decorator in node.decorator_list:
            decorator_node = (
                decorator.func if isinstance(decorator, ast.Call) else decorator
            )

            if isinstance(decorator_node, ast.Name):
                if decorator_node.id in triton_decorator_aliases:
                    return True
                continue

            if isinstance(decorator_node, ast.Attribute):
                if decorator_node.attr not in _TRITON_DECORATOR_NAMES:
                    continue
                if _decorator_root_name(decorator_node) in triton_aliases:
                    return True

    return False


def discover_project_kernels(
    project_path: str | Path,
    kernel_type: str = "hip",
    include_tests: bool = True,
    include_examples: bool = True,
    max_results: int = 50,
) -> dict[str, Any]:
    """
    Scan a project and return discovered kernel candidates.

    Args:
        project_path: Root path of the project to scan
        kernel_type: "hip", "cuda", "triton", or "all"
        include_tests: Include test directories in search
        include_examples: Include example directories in search
        max_results: Maximum number of returned entries

    Returns:
        Discovery result payload used by the MCP tool response.
    """
    project = Path(project_path)
    if not project.exists():
        raise FileNotFoundError(f"Project path does not exist: {project_path}")

    extensions: set[str] = set()
    if kernel_type in ("hip", "all"):
        extensions.update({".hip", ".cpp"})
    if kernel_type in ("cuda", "all"):
        extensions.update({".cu", ".cuh"})
    if kernel_type in ("triton", "all"):
        extensions.update({".py"})

    discovered = []
    build_dirs = sorted(BUILD_LIKE_DIRS)

    for root, dirs, files in os.walk(project):
        dirs[:] = sorted(d for d in dirs if _should_descend_into_dir(d, kernel_type))

        for filename in sorted(files):
            ext = os.path.splitext(filename)[1]
            if ext not in extensions:
                continue

            source_file = Path(root) / filename
            rel_path = source_file.relative_to(project)

            if _is_build_like_path(rel_path) and ext != ".py":
                continue
            if ext == ".py" and not is_triton_kernel_file(source_file):
                continue

            rel_path_lower = str(rel_path).lower()

            is_test = "test" in rel_path_lower
            is_example = "example" in rel_path_lower

            if not include_tests and is_test:
                continue
            if not include_examples and is_example:
                continue

            suggested_config = {
                "kernel_path": str(source_file),
                "kernel_type": "hip"
                if ext in (".hip", ".cpp")
                else ("triton" if ext == ".py" else "cuda"),
                "working_dir": str(project / "build")
                if (project / "build").exists()
                else str(project),
            }

            discovered.append(
                {
                    "source_file": str(source_file),
                    "name": source_file.stem,
                    "is_test": is_test,
                    "is_example": is_example,
                    "possible_binaries": [],
                    "suggested_config": suggested_config,
                }
            )

            if len(discovered) >= max_results * 2:
                break

        if len(discovered) >= max_results * 2:
            break

    discovered.sort(key=lambda x: (not x["is_test"], not x["is_example"], x["name"]))

    for entry in discovered[:max_results]:
        stem = entry["name"]
        possible_binaries = []

        for build_dir in build_dirs:
            build_path = project / build_dir
            if not build_path.exists():
                continue

            for binary in build_path.rglob(stem):
                if binary.is_file() and os.access(binary, os.X_OK):
                    possible_binaries.append(str(binary))
                    if len(possible_binaries) >= 3:
                        break
            if possible_binaries:
                break

        entry["possible_binaries"] = possible_binaries[:5]
        if possible_binaries:
            entry["suggested_config"]["testcase_command"] = possible_binaries[0]

    return {
        "project_path": str(project),
        "kernel_type": kernel_type,
        "total_found": len(discovered),
        "kernels": discovered[:max_results],
    }
