###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
``magpie.bench`` — sanctioned 0-overhead kernel timing helpers.

Two entry points:

- ``do_bench_cudagraph(fn, rep=20, n_retries=5, estimate_reps=5)`` —
  dispatch-inclusive wall-clock latency measured via CUDA graph
  estimate-then-unrolled-replay. Works on AMD (HIP graphs through
  ``torch.cuda.CUDAGraph``) and NVIDIA without code changes.
- ``LatencyStats`` — dataclass returned by ``do_bench_cudagraph`` carrying
  median / p50 / p99 / min / max / std and the meta-parameters used.

The marker line ``MAGPIE_LATENCY_JSON: { ... }`` printed by ``_runner.py``
is the contract used by ``Magpie/eval/latency.py`` to ingest results from
either the bundled runner subprocess or any user-provided harness.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, cast


__all__ = [
    "LatencyStats",
    "do_bench_cudagraph",
    "MAGPIE_LATENCY_JSON_MARKER",
]


MAGPIE_LATENCY_JSON_MARKER = "MAGPIE_LATENCY_JSON:"


@dataclass
class LatencyStats:
    """
    Summary statistics for a latency measurement.

    All time fields are in milliseconds.

    Attributes:
        median_ms: Median across ``n_retries`` independent measurements.
        p50_ms / p99_ms: Percentiles across the same set.
        min_ms / max_ms: Min/max across the same set.
        std_ms: Sample standard deviation (0.0 if ``n_retries < 2``).
        samples_ms: Raw per-retry measurements (``len == n_retries``).
        n_repeat: Number of unrolled ``fn()`` calls inside the timed graph.
        n_retries: Number of independent graph-replay measurements taken.
        estimate_ms: Per-call cost estimated from the small estimate graph
                     (used to size ``n_repeat``).
    """

    median_ms: float
    p50_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    samples_ms: List[float] = field(default_factory=list)
    n_repeat: int = 0
    n_retries: int = 0
    estimate_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["LatencyStats"]:
        if not data:
            return None
        return cls(
            median_ms=float(data.get("median_ms", 0.0)),
            p50_ms=float(data.get("p50_ms", 0.0)),
            p99_ms=float(data.get("p99_ms", 0.0)),
            min_ms=float(data.get("min_ms", 0.0)),
            max_ms=float(data.get("max_ms", 0.0)),
            std_ms=float(data.get("std_ms", 0.0)),
            samples_ms=list(data.get("samples_ms", []) or []),
            n_repeat=int(data.get("n_repeat", 0)),
            n_retries=int(data.get("n_retries", 0)),
            estimate_ms=float(data.get("estimate_ms", 0.0)),
        )

    @classmethod
    def from_samples(
        cls,
        samples_ms: List[float],
        *,
        n_repeat: int,
        n_retries: int,
        estimate_ms: float = 0.0,
    ) -> "LatencyStats":
        """Compute summary statistics from a list of per-retry latency samples."""
        if not samples_ms:
            return cls(
                median_ms=0.0,
                p50_ms=0.0,
                p99_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                std_ms=0.0,
                samples_ms=[],
                n_repeat=n_repeat,
                n_retries=n_retries,
                estimate_ms=estimate_ms,
            )

        sorted_ms = sorted(samples_ms)
        n = len(sorted_ms)
        # Linear-interpolation percentile (matches numpy's default)
        def _pct(p: float) -> float:
            if n == 1:
                return sorted_ms[0]
            rank = p * (n - 1) / 100.0
            lo = int(rank)
            hi = min(lo + 1, n - 1)
            frac = rank - lo
            return sorted_ms[lo] + (sorted_ms[hi] - sorted_ms[lo]) * frac

        std_ms = statistics.stdev(samples_ms) if n >= 2 else 0.0

        return cls(
            median_ms=statistics.median(samples_ms),
            p50_ms=_pct(50.0),
            p99_ms=_pct(99.0),
            min_ms=sorted_ms[0],
            max_ms=sorted_ms[-1],
            std_ms=std_ms,
            samples_ms=list(samples_ms),
            n_repeat=n_repeat,
            n_retries=n_retries,
            estimate_ms=estimate_ms,
        )


def do_bench_cudagraph(
    fn: Callable[[], Any],
    rep: int = 20,
    n_retries: int = 5,
    estimate_reps: int = 5,
) -> LatencyStats:
    """
    Benchmark ``fn`` via CUDA-graph estimate-then-unrolled-replay.

    Algorithm (mirrors the user-attached snippet byte-for-byte):

      1. Warmup: call ``fn()`` once on a side stream.
      2. Capture an "estimate" graph containing ``estimate_reps`` calls of
         ``fn``; replay it once to get ``estimate_ms`` per-call.
      3. Compute ``n_repeat = max(1, int(rep / estimate_ms))`` so the timed
         graph runs for roughly ``rep`` milliseconds.
      4. Capture a fresh graph with ``n_repeat`` unrolled calls.
      5. Replay the timed graph ``n_retries`` times, each time bracketed by
         a fresh pair of ``torch.cuda.Event`` records, and divide the
         elapsed time by ``n_repeat``.
      6. Report ``statistics.median`` of the ``n_retries`` per-call samples
         along with min/max/p50/p99/std.

    The dispatch overhead of each ``fn()`` call is amortized across
    ``n_repeat`` replays inside the captured graph, so per-call latency
    closely tracks the kernel time *plus* one graph-launch's worth of
    overhead (typically tens of microseconds).

    Args:
        fn: Zero-arg callable that issues the workload onto the current
            CUDA stream. Must be safe to capture inside ``torch.cuda.graph``.
        rep: Target measurement window in milliseconds.
        n_retries: Number of independent replay measurements.
        estimate_reps: Number of ``fn()`` calls inside the small estimate graph.

    Returns:
        ``LatencyStats`` with per-call median latency in ms.

    Raises:
        ImportError: If ``torch`` is not installed.
        RuntimeError: If CUDA / HIP is not available.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "magpie.bench.do_bench_cudagraph requires PyTorch. "
            "Install torch (or torch+rocm) and retry."
        ) from e

    if not torch.cuda.is_available():
        raise RuntimeError(
            "magpie.bench.do_bench_cudagraph requires a CUDA / HIP capable GPU; "
            "torch.cuda.is_available() is False."
        )

    stream = cast(torch.cuda.Stream, torch.cuda.Stream())
    stream.wait_stream(cast(torch.cuda.Stream, torch.cuda.current_stream()))
    with torch.cuda.stream(stream):
        torch.cuda.synchronize()
        # Warmup
        fn()

        # Step 1: capture initial estimate graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(estimate_reps):
                fn()
        torch.cuda.synchronize()

        # Step 2: estimate per-call device time
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        g.replay()
        end_event.record(stream)
        torch.cuda.synchronize()

        estimate_ms = start_event.elapsed_time(end_event) / estimate_reps
        if estimate_ms == 0:
            n_repeat = 1000
        else:
            n_repeat = max(1, int(rep / estimate_ms))

        # Step 3: capture timed graph with n_repeat unrolled calls
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n_repeat):
                fn()
        torch.cuda.synchronize()

        # Step 4: measure n_retries replays
        samples_ms: List[float] = []
        for _ in range(n_retries):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)
            g.replay()
            end_event.record(stream)
            torch.cuda.synchronize()
            samples_ms.append(start_event.elapsed_time(end_event) / n_repeat)

    return LatencyStats.from_samples(
        samples_ms,
        n_repeat=n_repeat,
        n_retries=n_retries,
        estimate_ms=estimate_ms,
    )
