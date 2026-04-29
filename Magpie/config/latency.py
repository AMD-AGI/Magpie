###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Latency evaluation configuration.

Defines a 0-overhead in-process timing harness that complements the
HW-counter-based ``Performance`` stage:

- ``cuda_graph``       — dispatch-inclusive wall-clock latency via
                          ``do_bench_cudagraph`` (warmup -> capture -> unrolled
                          replay -> median across retries).
- ``kernel_trace``     — kernel-only timing free of dispatch noise; the harness
                          runs in ``--profile`` mode (tight loop, no graph) and
                          the outer ``rocprofv3 --kernel-trace`` produces HW
                          per-dispatch durations.
- ``rocprof_timestamps`` — reuse ``pmc_perf.csv`` already produced by the
                          ``Performance`` stage; no extra subprocess.
- ``both``             — run both wall-clock and kernel-only and emit
                          ``dispatch_overhead_us = wall - kernel``.

For kernel-config autotuning (BLOCK sizes, num_warps, num_stages) the dispatch
overhead is roughly constant across configs and dominates wall-clock numbers
when the kernel runs in microseconds — use ``primary_metric=kernel_median_ms``
to rank by the dispatch-free measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import KernelType


# ---------------------------------------------------------------------------
# Sentinel string literals (not Enums to keep YAML/JSON round-tripping trivial)
# ---------------------------------------------------------------------------

LATENCY_METHODS = (
    "auto",
    "cuda_graph",
    "kernel_trace",
    "rocprof_timestamps",
    "both",
    "none",
)

PRIMARY_METRICS = ("wall_median_ms", "kernel_median_ms")


@dataclass
class BenchTarget:
    """
    Import-based benchmark target.

    The runner imports ``module``, looks up ``callable`` and ``get_inputs``,
    materializes inputs by calling ``get_inputs()`` (which must return either
    a tuple ``(args, kwargs)`` or a single args tuple/list), then times
    ``callable(*args, **kwargs)`` with ``magpie.bench.do_bench_cudagraph``.

    Attributes:
        module: Importable module path (e.g. ``my_kernels.scaled_mm``).
        callable: Attribute name of the callable inside the module.
        get_inputs: Attribute name of the inputs factory inside the module.
                    Must return ``(args, kwargs)`` or a positional tuple.
    """

    module: str
    callable: str
    get_inputs: str = "get_inputs"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "callable": self.callable,
            "get_inputs": self.get_inputs,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["BenchTarget"]:
        if not data:
            return None
        if "module" not in data or "callable" not in data:
            return None
        return cls(
            module=str(data["module"]),
            callable=str(data["callable"]),
            get_inputs=str(data.get("get_inputs", "get_inputs")),
        )


@dataclass
class LatencyConfig:
    """
    Configuration for the Latency evaluation stage.

    Attributes:
        enabled: Master switch. ``False`` skips the stage entirely.
        method: One of ``LATENCY_METHODS``. ``auto`` selects ``both`` for
                ``TRITON``/``PYTORCH``/``CUDA`` and ``rocprof_timestamps`` for
                ``HIP`` (since HIP testcases are native binaries that don't
                import torch).
        primary_metric: Which median is reported as the headline number and
                        used by ``compare`` mode for ranking.
        rep_ms: Target measurement window in milliseconds. ``do_bench_cudagraph``
                picks ``n_repeat = max(1, int(rep_ms / estimate_ms))``.
        n_retries: How many independent measurements to take (median is reported).
        estimate_reps: How many ``fn()`` calls to capture for the initial
                       cost-estimate graph.
        warmup_iters: Eager warmup iterations before any graph capture / timing.
        seed: Seed passed to ``torch.manual_seed`` / ``torch.cuda.manual_seed_all``
              before inputs are materialized — guarantees reproducible tensor
              contents and sizes across runs.
        kernel_filter: Optional regex applied to per-dispatch kernel names when
                       aggregating ``rocprof_timestamps`` / ``kernel_trace``
                       results.
        kernel_type: Auto-selection input. Set by ``PipelineConfig`` so that
                     ``method=auto`` can pick the right backend.
        gpu_arch: Auto-selection input (``gfx*`` -> AMD, ``sm_*`` -> NVIDIA).
        bench_target: Import-based target (sub-mode A). When ``None``, the
                      Latency stage falls back to running the user's
                      ``testcase_command`` and parsing a ``MAGPIE_LATENCY_JSON:``
                      stdout marker (sub-mode B).
        pythonpath: Extra absolute paths prepended to ``PYTHONPATH`` of the
                    benchmark subprocess so non-installed user packages import
                    cleanly.
        timeout_seconds: Per-subprocess timeout.
        output_dir: Where ``kernel_trace`` mode writes its rocprofv3 CSV.
                    ``None`` -> a sibling ``latency/`` folder under the
                    workload dir.
    """

    enabled: bool = True
    method: str = "auto"
    primary_metric: str = "wall_median_ms"
    rep_ms: int = 20
    n_retries: int = 5
    estimate_reps: int = 5
    warmup_iters: int = 5
    seed: int = 42
    kernel_filter: Optional[str] = None

    kernel_type: Optional["KernelType"] = None
    gpu_arch: Optional[str] = None

    bench_target: Optional[BenchTarget] = None
    pythonpath: List[str] = field(default_factory=list)

    timeout_seconds: float = 120.0
    output_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if self.method not in LATENCY_METHODS:
            raise ValueError(
                f"latency.method must be one of {LATENCY_METHODS}, got {self.method!r}"
            )
        if self.primary_metric not in PRIMARY_METRICS:
            raise ValueError(
                f"latency.primary_metric must be one of {PRIMARY_METRICS}, "
                f"got {self.primary_metric!r}"
            )

    # ------------------------------------------------------------------
    # Method resolution
    # ------------------------------------------------------------------

    def resolve_method(self) -> str:
        """
        Resolve ``method=auto`` into a concrete method based on the configured
        kernel type and GPU architecture.

        Selection table:
          - HIP                            -> ``rocprof_timestamps``
          - TRITON / PYTORCH / CUDA        -> ``both``
          - unknown / no kernel type       -> ``cuda_graph`` (best portable default)
        """
        if self.method != "auto":
            return self.method

        from .pipeline import KernelType

        if self.kernel_type == KernelType.HIP:
            return "rocprof_timestamps"
        if self.kernel_type in (KernelType.TRITON, KernelType.PYTORCH, KernelType.CUDA):
            return "both"
        return "cuda_graph"

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "method": self.method,
            "primary_metric": self.primary_metric,
            "rep_ms": self.rep_ms,
            "n_retries": self.n_retries,
            "estimate_reps": self.estimate_reps,
            "warmup_iters": self.warmup_iters,
            "seed": self.seed,
            "kernel_filter": self.kernel_filter,
            "bench_target": self.bench_target.to_dict() if self.bench_target else None,
            "pythonpath": list(self.pythonpath),
            "timeout_seconds": self.timeout_seconds,
            "output_dir": self.output_dir,
        }

    @classmethod
    def from_dict(
        cls,
        data: Optional[Dict[str, Any]],
        kernel_type: Optional["KernelType"] = None,
        gpu_arch: Optional[str] = None,
    ) -> "LatencyConfig":
        data = dict(data or {})
        bench_target = BenchTarget.from_dict(data.get("bench_target"))
        return cls(
            enabled=bool(data.get("enabled", True)),
            method=str(data.get("method", "auto")),
            primary_metric=str(data.get("primary_metric", "wall_median_ms")),
            rep_ms=int(data.get("rep_ms", 20)),
            n_retries=int(data.get("n_retries", 5)),
            estimate_reps=int(data.get("estimate_reps", 5)),
            warmup_iters=int(data.get("warmup_iters", 5)),
            seed=int(data.get("seed", 42)),
            kernel_filter=data.get("kernel_filter"),
            kernel_type=kernel_type,
            gpu_arch=gpu_arch,
            bench_target=bench_target,
            pythonpath=list(data.get("pythonpath", []) or []),
            timeout_seconds=float(data.get("timeout_seconds", 120.0)),
            output_dir=data.get("output_dir"),
        )
