###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Kernel evaluation configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .pipeline import KernelType


@dataclass
class KernelEvalConfig:
    """
    Configuration for evaluating a specific kernel.

    Attributes:
        kernel_id: Unique identifier for the kernel
        kernel_type: Type of kernel (pytorch, hip, cuda)
        source_file_path: Path(s) to kernel source file(s)
        working_dir: Working directory for compilation and execution
        env: Environment variables for execution

        # Compilation
        compiling_command: Custom compilation command(s) (optional)
            - Can be a single command: ["make", "build"]
            - Or a list of commands executed in order: [["make", "clean"], ["make", "build"]]

        # Correctness (for analyze mode)
        testcase_command: Custom testcase command(s) (required for analyze mode)
            - Can be a single command: ["./test.sh"]
            - Or a list of commands executed in order: [["./setup.sh"], ["./test.sh"]]

        # Performance profiling
        prof_command: Custom profiling command(s) (optional)
            - If provided, replaces the built-in profiler backend
            - Can be a single command or a list of commands

        # Input generation (for compare mode with pytorch)
        get_inputs_func: Name of function to generate inputs
        get_init_inputs_func: Name of function to generate init inputs

        # Additional
        input_shapes: Input tensor shapes for evaluation
        dtype: Data type for computation
        extra: Additional kernel-specific parameters
    """

    kernel_id: str = ""
    kernel_type: KernelType = KernelType.HIP
    source_file_path: List[str] = field(default_factory=list)
    working_dir: Optional[str] = None
    env: Optional[Dict[str, str]] = None

    # Compilation - supports single command or list of commands
    # Single: ["make", "build"] or List: [["make", "clean"], ["make", "build"]]
    compiling_command: Optional[List] = None

    # Correctness - supports single command or list of commands
    testcase_command: Optional[List] = None

    # Performance profiling - custom command(s) to replace built-in profiler
    prof_command: Optional[List] = None

    # Latency benchmarking (Triton/PyTorch/CUDA in-process harness)
    # Optional per-kernel BenchTarget spec ({"module", "callable", "get_inputs"}).
    # Wins over LatencyConfig.bench_target when both are set.
    bench_target: Optional[Dict[str, Any]] = None

    # Input generation (for KenrelBench)
    get_inputs_func: str = "get_inputs"
    get_init_inputs_func: str = "get_init_inputs"

    # Additional
    input_shapes: List[tuple] = field(default_factory=list)
    dtype: str = "float32"
    extra: Dict[str, Any] = field(default_factory=dict)

    def get_source_file_paths(self) -> List[str]:
        """Get source file paths as a list."""
        if isinstance(self.source_file_path, str):
            return [self.source_file_path]
        return self.source_file_path

    def has_compile_command(self) -> bool:
        """Check if custom compile command is provided."""
        return self.compiling_command is not None and len(self.compiling_command) > 0

    def has_testcase(self) -> bool:
        """Check if testcase command is provided."""
        return self.testcase_command is not None and len(self.testcase_command) > 0

    def has_prof_command(self) -> bool:
        """Check if custom profiling command is provided."""
        return self.prof_command is not None and len(self.prof_command) > 0

    def get_compile_commands(self) -> List[List[str]]:
        """
        Get compilation commands as a list of commands.

        Returns:
            List of commands, where each command is a list of strings.
        """
        if self.compiling_command is None:
            return []
        # Check if it's a list of commands or a single command
        if self.compiling_command and isinstance(self.compiling_command[0], list):
            return self.compiling_command
        # Single command - wrap in a list
        return [self.compiling_command]

    def get_testcase_commands(self) -> List[List[str]]:
        """
        Get testcase commands as a list of commands.

        Returns:
            List of commands, where each command is a list of strings.
        """
        if self.testcase_command is None:
            return []
        # Check if it's a list of commands or a single command
        if self.testcase_command and isinstance(self.testcase_command[0], list):
            return self.testcase_command
        # Single command - wrap in a list
        return [self.testcase_command]

    def get_prof_commands(self) -> List[List[str]]:
        """
        Get profiling commands as a list of commands.

        Returns:
            List of commands, where each command is a list of strings.
        """
        if self.prof_command is None:
            return []
        # Check if it's a list of commands or a single command
        if self.prof_command and isinstance(self.prof_command[0], list):
            return self.prof_command
        # Single command - wrap in a list
        return [self.prof_command]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "kernel_id": self.kernel_id,
            "kernel_type": self.kernel_type.name
            if hasattr(self.kernel_type, "name")
            else str(self.kernel_type),
            "source_file_path": self.source_file_path,
            "working_dir": self.working_dir,
            "env": self.env,
            "compiling_command": self.compiling_command,
            "testcase_command": self.testcase_command,
            "prof_command": self.prof_command,
            "bench_target": self.bench_target,
            "get_inputs_func": self.get_inputs_func,
            "get_init_inputs_func": self.get_init_inputs_func,
            "input_shapes": self.input_shapes,
            "dtype": self.dtype,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KernelEvalConfig":
        """Create from dictionary."""
        # Handle kernel_type conversion - use name (e.g., "HIP") not value
        kernel_type = data.get("kernel_type", "HIP")
        if isinstance(kernel_type, str):
            # Try to get enum by name
            kernel_type = KernelType[kernel_type.upper()]
        elif isinstance(kernel_type, int):
            # Fallback: get by value
            kernel_type = KernelType(kernel_type)

        return cls(
            kernel_id=data.get("kernel_id", ""),
            kernel_type=kernel_type,
            source_file_path=data.get("source_file_path", []),
            working_dir=data.get("working_dir"),
            env=data.get("env"),
            compiling_command=data.get("compiling_command"),
            testcase_command=data.get("testcase_command"),
            prof_command=data.get("prof_command"),
            bench_target=data.get("bench_target"),
            get_inputs_func=data.get("get_inputs_func", "get_inputs"),
            get_init_inputs_func=data.get("get_init_inputs_func", "get_init_inputs"),
            input_shapes=data.get("input_shapes", []),
            dtype=data.get("dtype", "float32"),
            extra=data.get("extra", {}),
        )
