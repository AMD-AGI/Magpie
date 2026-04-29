###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Latency evaluation stage.

Sits next to ``Performance`` (HW counters) in the pipeline and produces
0-overhead wall-clock and/or kernel-only latency for the kernel under
evaluation. See ``Magpie.config.latency`` for the method semantics.

This module never imports ``torch`` at module level — all heavyweight work
happens inside short-lived subprocesses spawned by :class:`Latency`.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..bench import LatencyStats, MAGPIE_LATENCY_JSON_MARKER
from ..config import (
    BenchTarget,
    KernelEvalConfig,
    LatencyConfig,
    PipelineConfig,
)
from ..utils import get_updated_env

logger = logging.getLogger(__name__)


# Filter out HIP / CUDA runtime kernels from per-kernel aggregations.
_RUNTIME_KERNEL_PREFIXES = ("__amd_rocclr_", "__hip_", "cuLaunchKernel")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class LatencyResult:
    """
    Result of the Latency evaluation stage.

    Fields are populated based on the resolved ``method``:

      - ``cuda_graph``         -> ``wall_stats`` only
      - ``kernel_trace``       -> ``kernel_stats`` + per-kernel breakdown
      - ``rocprof_timestamps`` -> ``kernel_stats`` + per-kernel breakdown
      - ``both``               -> both, plus ``dispatch_overhead_us``
    """

    success: bool
    method: str = "none"
    primary_metric: str = "wall_median_ms"
    wall_stats: Optional[LatencyStats] = None
    kernel_stats: Optional[LatencyStats] = None
    per_kernel: Dict[str, LatencyStats] = field(default_factory=dict)
    dispatch_overhead_us: Optional[float] = None
    crosscheck_vs_rocprof_ratio: Optional[float] = None
    crosscheck_warning: Optional[str] = None
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    command: Optional[str] = None
    output_dir: Optional[str] = None
    raw_output: Optional[str] = None
    errors: Optional[str] = None

    def get_primary_value(self) -> Optional[float]:
        """Return the headline number used by ``compare`` rankings."""
        if self.primary_metric == "kernel_median_ms" and self.kernel_stats:
            return self.kernel_stats.median_ms
        if self.wall_stats:
            return self.wall_stats.median_ms
        if self.kernel_stats:
            return self.kernel_stats.median_ms
        return None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "success": self.success,
            "method": self.method,
            "primary_metric": self.primary_metric,
            "primary_value_ms": self.get_primary_value(),
            "config": self.config_snapshot,
            "command": self.command,
            "output_dir": self.output_dir,
            "errors": self.errors,
        }
        if self.wall_stats is not None:
            d["wall_stats"] = self.wall_stats.to_dict()
        if self.kernel_stats is not None:
            d["kernel_stats"] = self.kernel_stats.to_dict()
        if self.per_kernel:
            d["per_kernel"] = {k: v.to_dict() for k, v in self.per_kernel.items()}
        if self.dispatch_overhead_us is not None:
            d["dispatch_overhead_us"] = self.dispatch_overhead_us
        if self.crosscheck_vs_rocprof_ratio is not None:
            d["crosscheck_vs_rocprof_ratio"] = self.crosscheck_vs_rocprof_ratio
        if self.crosscheck_warning:
            d["crosscheck_warning"] = self.crosscheck_warning
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_runtime_kernel(name: str) -> bool:
    return name.startswith(_RUNTIME_KERNEL_PREFIXES)


def _aggregate_per_kernel_durations_ns(
    per_kernel_ns: Dict[str, List[float]]
) -> Dict[str, LatencyStats]:
    """Build per-kernel ``LatencyStats`` from raw ns durations."""
    out: Dict[str, LatencyStats] = {}
    for name, samples_ns in per_kernel_ns.items():
        samples_ms = [v / 1e6 for v in samples_ns if v > 0]
        if not samples_ms:
            continue
        out[name] = LatencyStats.from_samples(
            samples_ms, n_repeat=1, n_retries=len(samples_ms)
        )
    return out


def _summary_stats_from_per_kernel(
    per_kernel: Dict[str, LatencyStats],
) -> Optional[LatencyStats]:
    """
    Build a single ``kernel_stats`` summary from per-kernel breakdown:

    - For each per-kernel timeline, take the median.
    - The summary's ``median_ms`` is the SUM of per-kernel medians (i.e. the
      median total kernel time per "iteration"), with min/max/p99 derived
      from the collated set of all dispatch durations.
    """
    if not per_kernel:
        return None

    total_median_ms = sum(stats.median_ms for stats in per_kernel.values())
    all_samples: List[float] = []
    for stats in per_kernel.values():
        all_samples.extend(stats.samples_ms)

    if not all_samples:
        return None

    base = LatencyStats.from_samples(
        all_samples, n_repeat=1, n_retries=len(all_samples)
    )
    # Override median with the per-kernel-summed median (truer "per iter" cost)
    base.median_ms = total_median_ms
    base.p50_ms = total_median_ms
    return base


def _merge_bench_target(
    cfg: LatencyConfig, kernel_cfg: KernelEvalConfig
) -> Optional[BenchTarget]:
    """Per-kernel ``bench_target`` wins over LatencyConfig default."""
    if kernel_cfg.bench_target:
        return BenchTarget.from_dict(kernel_cfg.bench_target)
    return cfg.bench_target


def _build_runner_env(
    cfg: LatencyConfig,
    bench_target: BenchTarget,
    kernel_env: Optional[Dict[str, str]],
) -> Dict[str, str]:
    """Construct env for the runner subprocess (no leakage of MAGPIE state)."""
    extra: Dict[str, str] = dict(kernel_env or {})

    extra["MAGPIE_BENCH_MODULE"] = bench_target.module
    extra["MAGPIE_BENCH_CALLABLE"] = bench_target.callable
    extra["MAGPIE_BENCH_INPUTS_FUNC"] = bench_target.get_inputs
    extra["MAGPIE_BENCH_REP_MS"] = str(cfg.rep_ms)
    extra["MAGPIE_BENCH_N_RETRIES"] = str(cfg.n_retries)
    extra["MAGPIE_BENCH_ESTIMATE_REPS"] = str(cfg.estimate_reps)
    extra["MAGPIE_BENCH_WARMUP_ITERS"] = str(cfg.warmup_iters)
    extra["MAGPIE_BENCH_SEED"] = str(cfg.seed)

    if cfg.pythonpath:
        extra["PYTHONPATH"] = ":".join(cfg.pythonpath)

    return get_updated_env(extra)


def _runner_module_args() -> List[str]:
    """Command-line invocation of the runner as a module."""
    return [sys.executable, "-m", "Magpie.bench._runner"]


def _parse_marker_line(stdout: str) -> Optional[Dict[str, Any]]:
    """Find and parse the ``MAGPIE_LATENCY_JSON: {...}`` line in *stdout*."""
    if not stdout:
        return None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith(MAGPIE_LATENCY_JSON_MARKER):
            payload = line[len(MAGPIE_LATENCY_JSON_MARKER):].strip()
            try:
                return json.loads(payload)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse MAGPIE_LATENCY_JSON line: {e}")
                return None
    return None


# ---------------------------------------------------------------------------
# rocprofv3 / pmc_perf parsing
# ---------------------------------------------------------------------------


def _parse_rocprofv3_kernel_trace_csv(
    csv_path: Path,
    kernel_filter_re: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Parse a rocprofv3 ``--kernel-trace`` CSV and return per-kernel ns durations.

    rocprofv3 emits columns like ``Kernel_Name``, ``Start_Timestamp``,
    ``End_Timestamp`` (units = ns). Different rocprofv3 versions also use
    ``KernelName`` / ``Start_Time`` / ``End_Time`` — we accept both spellings.
    """
    import re

    rx = re.compile(kernel_filter_re) if kernel_filter_re else None

    per_kernel: Dict[str, List[float]] = {}
    if not csv_path.exists():
        return per_kernel

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (
                    row.get("Kernel_Name")
                    or row.get("KernelName")
                    or row.get("kernel_name")
                    or "unknown"
                )
                if _is_runtime_kernel(name):
                    continue
                if rx and not rx.search(name):
                    continue
                start = (
                    row.get("Start_Timestamp")
                    or row.get("Start_Time")
                    or row.get("start_timestamp")
                )
                end = (
                    row.get("End_Timestamp")
                    or row.get("End_Time")
                    or row.get("end_timestamp")
                )
                if not start or not end:
                    continue
                try:
                    duration_ns = float(end) - float(start)
                except (TypeError, ValueError):
                    continue
                if duration_ns <= 0:
                    continue
                per_kernel.setdefault(name, []).append(duration_ns)
    except Exception as e:
        logger.warning(f"Failed to parse rocprofv3 kernel-trace CSV {csv_path}: {e}")

    return per_kernel


def _find_rocprofv3_csv(out_dir: Path) -> Optional[Path]:
    """Locate the kernel-trace CSV inside *out_dir*.

    rocprofv3 layouts seen in the wild:
      - ``<out_dir>/kernel_trace.csv``
      - ``<out_dir>/<name>_kernel_trace.csv``
      - ``<out_dir>/<host>/<pid>_kernel_trace.csv``  (default in 7.x+)
    Walks recursively to find the first matching file.
    """
    if not out_dir.exists():
        return None

    direct = out_dir / "kernel_trace.csv"
    if direct.exists():
        return direct

    # Recursive glob so ``<host>/<pid>_kernel_trace.csv`` works.
    for pattern in ("*kernel_trace*.csv", "*_kernel_trace.csv"):
        for hit in sorted(out_dir.rglob(pattern)):
            # Skip the agent_info / per-process metadata files.
            if "agent_info" in hit.name:
                continue
            return hit
    return None


def _parse_pmc_perf_csv_for_durations(
    csv_path: Path,
    kernel_filter_re: Optional[str] = None,
) -> Dict[str, List[float]]:
    """Parse rocprof-compute's ``pmc_perf.csv`` for per-dispatch ns durations."""
    import re

    rx = re.compile(kernel_filter_re) if kernel_filter_re else None

    per_kernel: Dict[str, List[float]] = {}
    if not csv_path.exists():
        return per_kernel

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (
                    row.get("Kernel_Name")
                    or row.get("KernelName")
                    or "unknown"
                )
                if _is_runtime_kernel(name):
                    continue
                if rx and not rx.search(name):
                    continue
                try:
                    start = float(row.get("Start_Timestamp", 0))
                    end = float(row.get("End_Timestamp", 0))
                except (TypeError, ValueError):
                    continue
                if end <= start:
                    continue
                per_kernel.setdefault(name, []).append(end - start)
    except Exception as e:
        logger.warning(f"Failed to parse pmc_perf.csv {csv_path}: {e}")

    return per_kernel


# ---------------------------------------------------------------------------
# Latency stage
# ---------------------------------------------------------------------------


class Latency:
    """
    Latency evaluation handler.

    Methods:
      - ``cuda_graph``        : in-process ``do_bench_cudagraph`` (subprocess)
      - ``kernel_trace``      : runner ``--profile`` + rocprofv3 ``--kernel-trace``
      - ``rocprof_timestamps``: reuse pmc_perf.csv from the Performance stage
      - ``both``              : run both wall-clock and a kernel-only method
    """

    def __init__(self, pipeline_cfg: PipelineConfig) -> None:
        self.pipeline_cfg = pipeline_cfg
        self.lat_cfg: LatencyConfig = (
            pipeline_cfg.latency_config
            or LatencyConfig(
                kernel_type=pipeline_cfg.kernel_type,
                gpu_arch=pipeline_cfg.gpu_arch,
            )
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        eval_state: Any,
        kernel_cfg: KernelEvalConfig,
    ) -> Optional[LatencyResult]:
        """
        Run the Latency stage.

        Returns:
          - ``None`` when the stage is skipped (disabled or method=``none``).
          - ``LatencyResult`` otherwise.
        """
        if not self.lat_cfg.enabled:
            return None

        method = self.lat_cfg.resolve_method()
        if method == "none":
            return None

        snapshot = self._config_snapshot(method)

        # rocprof_timestamps reuses the Performance stage's pmc_perf.csv
        if method == "rocprof_timestamps":
            return self._run_rocprof_timestamps(eval_state, snapshot)

        # cuda_graph / kernel_trace / both — need either a bench_target OR a
        # testcase_command harness that emits MAGPIE_LATENCY_JSON.
        bench_target = _merge_bench_target(self.lat_cfg, kernel_cfg)
        has_harness_cmd = kernel_cfg.has_testcase()

        try:
            if method == "cuda_graph":
                return self._run_cuda_graph(
                    bench_target, kernel_cfg, has_harness_cmd, snapshot
                )
            if method == "kernel_trace":
                return self._run_kernel_trace(
                    bench_target, kernel_cfg, snapshot
                )
            if method == "both":
                return self._run_both(
                    bench_target, kernel_cfg, has_harness_cmd, eval_state, snapshot
                )
        except Exception as e:  # pragma: no cover — defensive
            logger.exception(f"Latency stage failed: {e}")
            return LatencyResult(
                success=False,
                method=method,
                primary_metric=self.lat_cfg.primary_metric,
                config_snapshot=snapshot,
                errors=str(e),
            )

        return LatencyResult(
            success=False,
            method=method,
            primary_metric=self.lat_cfg.primary_metric,
            config_snapshot=snapshot,
            errors=f"Unknown latency method: {method}",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _config_snapshot(self, resolved_method: str) -> Dict[str, Any]:
        snap = self.lat_cfg.to_dict()
        snap["resolved_method"] = resolved_method
        snap["gpu_arch"] = self.lat_cfg.gpu_arch
        snap["kernel_type"] = (
            self.lat_cfg.kernel_type.name if self.lat_cfg.kernel_type else None
        )
        return snap

    # ----- cuda_graph -------------------------------------------------

    def _run_cuda_graph(
        self,
        bench_target: Optional[BenchTarget],
        kernel_cfg: KernelEvalConfig,
        has_harness_cmd: bool,
        snapshot: Dict[str, Any],
    ) -> LatencyResult:
        if bench_target is None and not has_harness_cmd:
            return LatencyResult(
                success=False,
                method="cuda_graph",
                primary_metric=self.lat_cfg.primary_metric,
                config_snapshot=snapshot,
                errors=(
                    "cuda_graph requires either a bench_target (import-based) "
                    "or testcase_command (user harness emitting "
                    "MAGPIE_LATENCY_JSON: {...})."
                ),
            )

        if bench_target is not None:
            return self._exec_runner_cuda_graph(bench_target, kernel_cfg, snapshot)

        return self._exec_user_harness(kernel_cfg, snapshot)

    def _exec_runner_cuda_graph(
        self,
        bench_target: BenchTarget,
        kernel_cfg: KernelEvalConfig,
        snapshot: Dict[str, Any],
    ) -> LatencyResult:
        env = _build_runner_env(self.lat_cfg, bench_target, kernel_cfg.env)
        cmd = _runner_module_args()
        return self._invoke_runner(cmd, env, kernel_cfg.working_dir, snapshot, "cuda_graph")

    def _exec_user_harness(
        self,
        kernel_cfg: KernelEvalConfig,
        snapshot: Dict[str, Any],
    ) -> LatencyResult:
        cmds = kernel_cfg.get_testcase_commands()
        if not cmds:
            return LatencyResult(
                success=False,
                method="cuda_graph",
                primary_metric=self.lat_cfg.primary_metric,
                config_snapshot=snapshot,
                errors="No testcase command available for user harness",
            )
        env = get_updated_env(kernel_cfg.env)
        # Use the LAST testcase command — earlier commands are usually setup
        cmd = cmds[-1]
        return self._invoke_runner(cmd, env, kernel_cfg.working_dir, snapshot, "cuda_graph")

    def _invoke_runner(
        self,
        cmd: List[str],
        env: Dict[str, str],
        cwd: Optional[str],
        snapshot: Dict[str, Any],
        method: str,
    ) -> LatencyResult:
        cmd_str = " ".join(cmd)
        logger.info(f"[Latency:{method}] running: {cmd_str}")
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd=cwd,
                timeout=self.lat_cfg.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return LatencyResult(
                success=False,
                method=method,
                primary_metric=self.lat_cfg.primary_metric,
                config_snapshot=snapshot,
                command=cmd_str,
                errors=f"Latency runner timed out after {self.lat_cfg.timeout_seconds}s",
            )

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        marker = _parse_marker_line(stdout)

        if proc.returncode != 0 or marker is None:
            return LatencyResult(
                success=False,
                method=method,
                primary_metric=self.lat_cfg.primary_metric,
                config_snapshot=snapshot,
                command=cmd_str,
                raw_output=stdout,
                errors=(
                    marker.get("error")
                    if isinstance(marker, dict)
                    else (stderr.strip() or stdout.strip() or "no MAGPIE_LATENCY_JSON marker")
                ),
            )

        if marker.get("error"):
            return LatencyResult(
                success=False,
                method=method,
                primary_metric=self.lat_cfg.primary_metric,
                config_snapshot=snapshot,
                command=cmd_str,
                raw_output=stdout,
                errors=str(marker.get("error")),
            )

        wall_stats = LatencyStats.from_dict(marker.get("stats"))

        return LatencyResult(
            success=True,
            method=method,
            primary_metric=self.lat_cfg.primary_metric,
            wall_stats=wall_stats,
            config_snapshot=snapshot,
            command=cmd_str,
            raw_output=stdout,
        )

    # ----- kernel_trace ----------------------------------------------

    def _run_kernel_trace(
        self,
        bench_target: Optional[BenchTarget],
        kernel_cfg: KernelEvalConfig,
        snapshot: Dict[str, Any],
    ) -> LatencyResult:
        if shutil.which("rocprofv3") is None:
            return LatencyResult(
                success=False,
                method="kernel_trace",
                primary_metric=self.lat_cfg.primary_metric,
                config_snapshot=snapshot,
                errors=(
                    "rocprofv3 not found. kernel_trace requires ROCm rocprofv3 "
                    "on PATH. Use method=cuda_graph for a torch-only fallback."
                ),
            )

        if bench_target is None:
            # User harness: just wrap the testcase command with rocprofv3.
            cmds = kernel_cfg.get_testcase_commands()
            if not cmds:
                return LatencyResult(
                    success=False,
                    method="kernel_trace",
                    primary_metric=self.lat_cfg.primary_metric,
                    config_snapshot=snapshot,
                    errors=(
                        "kernel_trace requires either a bench_target or a "
                        "testcase_command to wrap with rocprofv3."
                    ),
                )
            inner = cmds[-1]
            env = get_updated_env(kernel_cfg.env)
        else:
            inner = _runner_module_args() + ["--profile"]
            env = _build_runner_env(self.lat_cfg, bench_target, kernel_cfg.env)

        out_dir = self._kernel_trace_output_dir(kernel_cfg)
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "rocprofv3",
            "--kernel-trace",
            "--output-format",
            "csv",
            "-d",
            str(out_dir),
            "--",
            *inner,
        ]
        cmd_str = " ".join(cmd)
        logger.info(f"[Latency:kernel_trace] running: {cmd_str}")

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd=kernel_cfg.working_dir,
                timeout=self.lat_cfg.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return LatencyResult(
                success=False,
                method="kernel_trace",
                primary_metric=self.lat_cfg.primary_metric,
                config_snapshot=snapshot,
                command=cmd_str,
                output_dir=str(out_dir),
                errors=f"rocprofv3 timed out after {self.lat_cfg.timeout_seconds}s",
            )

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        if proc.returncode != 0:
            return LatencyResult(
                success=False,
                method="kernel_trace",
                primary_metric=self.lat_cfg.primary_metric,
                config_snapshot=snapshot,
                command=cmd_str,
                output_dir=str(out_dir),
                raw_output=stdout,
                errors=stderr.strip() or stdout.strip() or "rocprofv3 failed",
            )

        csv_path = _find_rocprofv3_csv(out_dir)
        if csv_path is None:
            return LatencyResult(
                success=False,
                method="kernel_trace",
                primary_metric=self.lat_cfg.primary_metric,
                config_snapshot=snapshot,
                command=cmd_str,
                output_dir=str(out_dir),
                raw_output=stdout,
                errors=f"No kernel-trace CSV found under {out_dir}",
            )

        per_kernel_ns = _parse_rocprofv3_kernel_trace_csv(
            csv_path, self.lat_cfg.kernel_filter
        )
        per_kernel = _aggregate_per_kernel_durations_ns(per_kernel_ns)

        # Heuristic: when no kernel_filter is set, drop "outlier" kernels
        # that fired only a handful of times (typical for torch setup like
        # randn / fill / kaiming) so they don't pollute the sum-of-medians
        # summary. Always emit a warning when many distinct kernels were
        # captured so users know to set kernel_filter for tighter numbers.
        if not self.lat_cfg.kernel_filter and len(per_kernel) > 1:
            max_n = max(len(s.samples_ms) for s in per_kernel.values())
            cutoff = max(2, max_n // 10)
            dropped = {k: len(s.samples_ms) for k, s in per_kernel.items()
                       if len(s.samples_ms) < cutoff}
            if dropped:
                logger.warning(
                    "[Latency:kernel_trace] dropping %d low-dispatch kernels "
                    "from kernel_stats summary (set latency.kernel_filter to "
                    "silence): %s",
                    len(dropped),
                    ", ".join(f"{k} ({n} dispatches)" for k, n in dropped.items()),
                )
                per_kernel = {
                    k: s for k, s in per_kernel.items()
                    if len(s.samples_ms) >= cutoff
                }

        kernel_stats = _summary_stats_from_per_kernel(per_kernel)

        if kernel_stats is None:
            return LatencyResult(
                success=False,
                method="kernel_trace",
                primary_metric=self.lat_cfg.primary_metric,
                config_snapshot=snapshot,
                command=cmd_str,
                output_dir=str(out_dir),
                raw_output=stdout,
                errors=(
                    f"rocprofv3 produced no kernel timings in {csv_path} "
                    "(check kernel_filter regex)"
                ),
            )

        return LatencyResult(
            success=True,
            method="kernel_trace",
            primary_metric=self.lat_cfg.primary_metric,
            kernel_stats=kernel_stats,
            per_kernel=per_kernel,
            config_snapshot=snapshot,
            command=cmd_str,
            output_dir=str(out_dir),
            raw_output=stdout,
        )

    def _kernel_trace_output_dir(self, kernel_cfg: KernelEvalConfig) -> Path:
        base = self.lat_cfg.output_dir or os.path.join(
            kernel_cfg.working_dir or os.getcwd(), "latency"
        )
        # Use a per-kernel subdir + timestamp to avoid clobbering across runs
        kid = (
            kernel_cfg.kernel_id
            if kernel_cfg.kernel_id
            else "kernel"
        )
        safe_kid = "".join(c if c.isalnum() or c in "-_" else "_" for c in kid)
        ts = int(time.time())
        return Path(base) / f"kernel_trace_{safe_kid}_{ts}"

    # ----- rocprof_timestamps ----------------------------------------

    def _run_rocprof_timestamps(
        self,
        eval_state: Any,
        snapshot: Dict[str, Any],
    ) -> LatencyResult:
        """
        Reuse ``pmc_perf.csv`` produced by a prior Performance stage.

        Looks at ``eval_state.performance_result.workload_dir`` to find the
        rocprof-compute workload directory. If it's missing (e.g. the
        Performance stage was skipped), returns a soft failure suggesting
        the user run with profiling enabled.
        """
        perf_result = getattr(eval_state, "performance_result", None)
        workload_dir = getattr(perf_result, "workload_dir", None) if perf_result else None

        if not workload_dir:
            return LatencyResult(
                success=False,
                method="rocprof_timestamps",
                primary_metric=self.lat_cfg.primary_metric,
                config_snapshot=snapshot,
                errors=(
                    "rocprof_timestamps requires the Performance stage to have "
                    "produced a workload_dir (rocprof-compute). Re-run with "
                    "performance enabled, or use method=kernel_trace."
                ),
            )

        pmc_csv = Path(workload_dir) / "pmc_perf.csv"
        if not pmc_csv.exists():
            return LatencyResult(
                success=False,
                method="rocprof_timestamps",
                primary_metric=self.lat_cfg.primary_metric,
                config_snapshot=snapshot,
                output_dir=str(workload_dir),
                errors=f"No pmc_perf.csv found in {workload_dir}",
            )

        per_kernel_ns = _parse_pmc_perf_csv_for_durations(
            pmc_csv, self.lat_cfg.kernel_filter
        )
        per_kernel = _aggregate_per_kernel_durations_ns(per_kernel_ns)
        kernel_stats = _summary_stats_from_per_kernel(per_kernel)

        if kernel_stats is None:
            return LatencyResult(
                success=False,
                method="rocprof_timestamps",
                primary_metric=self.lat_cfg.primary_metric,
                config_snapshot=snapshot,
                output_dir=str(workload_dir),
                errors=(
                    f"No kernel dispatch durations parsed from {pmc_csv} "
                    "(check kernel_filter regex)"
                ),
            )

        return LatencyResult(
            success=True,
            method="rocprof_timestamps",
            primary_metric=self.lat_cfg.primary_metric,
            kernel_stats=kernel_stats,
            per_kernel=per_kernel,
            config_snapshot=snapshot,
            output_dir=str(workload_dir),
        )

    # ----- both --------------------------------------------------------

    def _run_both(
        self,
        bench_target: Optional[BenchTarget],
        kernel_cfg: KernelEvalConfig,
        has_harness_cmd: bool,
        eval_state: Any,
        snapshot: Dict[str, Any],
    ) -> LatencyResult:
        wall = self._run_cuda_graph(bench_target, kernel_cfg, has_harness_cmd, snapshot)

        # Pick the kernel-only method based on environment availability
        if shutil.which("rocprofv3") is not None and bench_target is not None:
            kern = self._run_kernel_trace(bench_target, kernel_cfg, snapshot)
        else:
            # Try rocprof_timestamps as a fallback; only works when
            # Performance has produced a workload_dir.
            kern = self._run_rocprof_timestamps(eval_state, snapshot)

        wall_ok = wall.success and wall.wall_stats is not None
        kern_ok = kern.success and kern.kernel_stats is not None

        if not wall_ok and not kern_ok:
            errs = [
                e for e in (wall.errors, kern.errors) if e
            ]
            return LatencyResult(
                success=False,
                method="both",
                primary_metric=self.lat_cfg.primary_metric,
                config_snapshot=snapshot,
                errors="; ".join(errs) or "both methods failed",
            )

        merged = LatencyResult(
            success=True,
            method="both",
            primary_metric=self.lat_cfg.primary_metric,
            wall_stats=wall.wall_stats if wall_ok else None,
            kernel_stats=kern.kernel_stats if kern_ok else None,
            per_kernel=kern.per_kernel if kern_ok else {},
            config_snapshot=snapshot,
            command=" && ".join(
                c for c in (wall.command, kern.command) if c
            ) or None,
            output_dir=kern.output_dir,
            raw_output=wall.raw_output,
        )

        if wall_ok and kern_ok and wall.wall_stats and kern.kernel_stats:
            wall_us = wall.wall_stats.median_ms * 1000.0
            kern_us = kern.kernel_stats.median_ms * 1000.0
            merged.dispatch_overhead_us = wall_us - kern_us
            if kern_us > 0:
                merged.crosscheck_vs_rocprof_ratio = wall_us / kern_us
                if not (0.5 <= merged.crosscheck_vs_rocprof_ratio <= 2.0):
                    merged.crosscheck_warning = (
                        f"wall/kernel ratio {merged.crosscheck_vs_rocprof_ratio:.2f} "
                        "is outside [0.5, 2.0]; check warmup, timer pollution, "
                        "or kernel_filter."
                    )

        if not wall_ok:
            # Surface kernel-only error so users see why wall failed
            merged.errors = wall.errors
        elif not kern_ok:
            merged.errors = kern.errors

        return merged
