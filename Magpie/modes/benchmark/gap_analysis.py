###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Gap analysis for torch profiler traces.

Matches the behavior of workloads-inference gen_kstats_clamped_traces.py:
  - Kernel stats CSV uses ALL events (no time window), category-filtered only
  - Clamped trace files use time window, no category filter
  - Category matching is case-insensitive substring matching

CSV columns: Name, Calls, Self CUDA total (us), Avg time (us), % Total
"""

import csv
import gzip
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import stdev
from typing import Any, Dict, List, Optional, Tuple

from .config import GapAnalysisConfig

logger = logging.getLogger(__name__)


@dataclass
class KernelStat:
    """Aggregated statistics for a single kernel/function name."""
    name: str
    total_duration_us: float = 0.0
    calls: int = 0
    durations_us: List[float] = field(default_factory=list)

    @property
    def avg_us(self) -> float:
        return self.total_duration_us / self.calls if self.calls else 0.0

    @property
    def min_us(self) -> float:
        return min(self.durations_us) if self.durations_us else 0.0

    @property
    def max_us(self) -> float:
        return max(self.durations_us) if self.durations_us else 0.0

    @property
    def std_us(self) -> float:
        return stdev(self.durations_us) if len(self.durations_us) > 1 else 0.0


@dataclass
class RankResult:
    """Gap analysis result for a single rank."""
    rank: int
    trace_file: str
    total_duration_us: float = 0.0
    kernels: List[KernelStat] = field(default_factory=list)


@dataclass
class GapAnalysisResult:
    """Complete gap analysis result across all ranks."""
    config: Dict[str, Any] = field(default_factory=dict)
    rank_results: List[RankResult] = field(default_factory=list)
    merged_kernels: List[KernelStat] = field(default_factory=list)
    total_duration_us: float = 0.0
    clamped_trace_paths: List[Path] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config": self.config,
            "total_duration_us": self.total_duration_us,
            "num_ranks": len(self.rank_results),
            "top_kernels": [
                {
                    "name": k.name,
                    "calls": k.calls,
                    "self_cuda_total_us": k.total_duration_us,
                    "avg_time_us": k.avg_us,
                    "pct_total": (
                        k.total_duration_us / self.total_duration_us * 100.0
                        if self.total_duration_us > 0 else 0.0
                    ),
                }
                for k in self.merged_kernels
            ],
            "clamped_trace_paths": [str(p) for p in self.clamped_trace_paths],
            "errors": self.errors,
        }

    def to_csv(self, output_path: Path) -> Path:
        """Write merged kernel stats to CSV."""
        output_path = Path(output_path)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Name", "Calls", "Self CUDA total (us)",
                "Avg time (us)", "% Total",
            ])
            for k in self.merged_kernels:
                pct = (
                    k.total_duration_us / self.total_duration_us * 100.0
                    if self.total_duration_us > 0 else 0.0
                )
                writer.writerow([
                    k.name,
                    k.calls,
                    f"{k.total_duration_us:.2f}",
                    f"{k.avg_us:.2f}",
                    f"{pct:.2f}",
                ])
        logger.info(f"Wrote gap analysis CSV: {output_path}")
        return output_path

    def to_rank_csv(self, output_dir: Path) -> List[Path]:
        """Write per-rank CSV files. Returns list of written paths."""
        paths: List[Path] = []
        for rr in self.rank_results:
            rank_path = Path(output_dir) / f"gap_analysis_rank{rr.rank}.csv"
            with open(rank_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Name", "Calls", "Self CUDA total (us)",
                    "Avg time (us)", "% Total",
                ])
                for k in rr.kernels:
                    pct = (
                        k.total_duration_us / rr.total_duration_us * 100.0
                        if rr.total_duration_us > 0 else 0.0
                    )
                    writer.writerow([
                        k.name,
                        k.calls,
                        f"{k.total_duration_us:.2f}",
                        f"{k.avg_us:.2f}",
                        f"{pct:.2f}",
                    ])
            paths.append(rank_path)
            logger.debug(f"Wrote per-rank CSV: {rank_path}")
        return paths


class GapAnalyzer:
    """
    Analyzes torch profiler Chrome-trace files, producing kernel stats CSVs.

    Pipeline:
      1. Apply time window (trace_start_pct – trace_end_pct)
      2. Filter by category (case-insensitive substring matching)
      3. Aggregate stats and rank by total duration
    """

    def __init__(self, config: Optional[GapAnalysisConfig] = None):
        self.config = config or GapAnalysisConfig(enabled=True)

    def analyze(self, trace_dir: Path) -> GapAnalysisResult:
        """
        Run gap analysis on all rank traces in *trace_dir*.

        Produces kernel stats (category-filtered, no time window).
        Does NOT generate clamped traces — call
        :meth:`generate_clamped_traces` separately if needed.

        Returns a GapAnalysisResult with per-rank and merged kernel stats.
        """
        trace_dir = Path(trace_dir)
        result = GapAnalysisResult(config=self.config.to_dict())

        if not trace_dir.exists():
            result.errors.append(f"Trace directory not found: {trace_dir}")
            return result

        rank_files = self.detect_trace_files(trace_dir)
        if not rank_files:
            result.errors.append(f"No trace files found in {trace_dir}")
            return result

        logger.info(
            f"Found {len(rank_files)} trace file(s) in {trace_dir}"
        )

        for rank, trace_file in rank_files:
            try:
                _data, events = self._load_trace_data(trace_file)

                rr = self._analyze_single_rank(rank, trace_file, events)
                result.rank_results.append(rr)

            except Exception as e:
                msg = f"Failed to analyze rank {rank} ({trace_file.name}): {e}"
                logger.warning(msg)
                result.errors.append(msg)

        if result.rank_results:
            result.merged_kernels = self._merge_ranks(result.rank_results)
            result.total_duration_us = sum(
                rr.total_duration_us for rr in result.rank_results
            )
            result.merged_kernels = result.merged_kernels[: self.config.top_k]
            for rr in result.rank_results:
                rr.kernels = rr.kernels[: self.config.top_k]

        return result

    def generate_clamped_traces(
        self,
        trace_dir: Path,
        output_dir: Optional[Path] = None,
    ) -> List[Path]:
        """
        Generate time-windowed (clamped) trace files for each rank.

        This is a separate step from :meth:`analyze` — call it only
        when you explicitly need clamped trace output.

        Args:
            trace_dir: Directory containing torch profiler trace files.
            output_dir: Where to write clamped traces.
                        Defaults to *trace_dir* itself.

        Returns:
            List of paths to generated clamped trace files.
        """
        trace_dir = Path(trace_dir)
        dest = Path(output_dir) if output_dir else trace_dir
        dest.mkdir(parents=True, exist_ok=True)

        rank_files = self.detect_trace_files(trace_dir)
        paths: List[Path] = []

        for _rank, trace_file in rank_files:
            try:
                data, events = self._load_trace_data(trace_file)
                p = self._generate_clamped_trace(
                    data, events, trace_file, output_dir=dest,
                )
                if p:
                    paths.append(p)
            except Exception as e:
                logger.warning(
                    f"Failed to generate clamped trace for "
                    f"{trace_file.name}: {e}"
                )

        return paths

    # -- trace file discovery -----------------------------------------------

    @staticmethod
    def detect_trace_files(trace_dir: Path) -> List[Tuple[int, Path]]:
        """
        Discover per-rank trace files in *trace_dir*.

        Rank traces match vLLM-style ``*rankN*.json.gz`` or the older
        ``*-rank-N.*.json.gz`` naming. Async coordinator traces such as
        ``async_llm`` are excluded because they are not per-rank GPU traces.
        Falls back to non-async ``.json.gz`` / ``.json`` if no rank pattern
        is found.
        """
        rank_files: List[Tuple[int, Path]] = []

        def _candidate_paths(pattern: str) -> List[Path]:
            return sorted(
                path for path in trace_dir.glob(pattern)
                if "async_llm" not in path.name
            )

        for pattern in ("*rank*.pt.trace.json.gz", "*rank*.pt.trace.json"):
            for path in _candidate_paths(pattern):
                rank = _extract_rank(path.name)
                if rank is not None:
                    rank_files.append((rank, path))
            if rank_files:
                return sorted(rank_files, key=lambda x: x[0])

        # Fallback: any non-async trace file.
        for idx, gz in enumerate(_candidate_paths("*.json.gz")):
            rank_files.append((idx, gz))
        for idx, jf in enumerate(_candidate_paths("*.json"), start=len(rank_files)):
            rank_files.append((idx, jf))

        return sorted(rank_files, key=lambda x: x[0])

    # -- single-rank analysis -----------------------------------------------

    def _analyze_single_rank(
        self,
        rank: int,
        trace_file: Path,
        events: List[Dict[str, Any]],
    ) -> RankResult:
        """
        Build kernel stats for one rank.

        Applies time window first, then category filter.
        """
        logger.debug(
            f"Rank {rank}: loaded {len(events)} events from {trace_file.name}"
        )

        windowed = self._apply_time_window(events)
        logger.debug(
            f"Rank {rank}: {len(windowed)} events in time window "
            f"({self.config.trace_start_pct}%-{self.config.trace_end_pct}%)"
        )

        filtered = self._filter_by_category(windowed)
        logger.debug(
            f"Rank {rank}: {len(filtered)} events after category filter"
        )

        if self.config.min_duration_us > 0:
            filtered = [
                (n, d) for n, d in filtered
                if d >= self.config.min_duration_us
            ]

        kernels = self._aggregate_stats(filtered)
        total_us = sum(k.total_duration_us for k in kernels)

        return RankResult(
            rank=rank,
            trace_file=str(trace_file),
            total_duration_us=total_us,
            kernels=kernels,
        )

    # -- time window --------------------------------------------------------

    def _apply_time_window(
        self, events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Keep only events that overlap with the configured time window.

        If the window is 0%-100% (default), returns all events unchanged.
        """
        start_pct = self.config.trace_start_pct
        end_pct = self.config.trace_end_pct

        if start_pct <= 0 and end_pct >= 100:
            return events

        t_min = float("inf")
        t_max = float("-inf")
        for e in events:
            ts = e.get("ts")
            if ts is None:
                continue
            dur = e.get("dur", 0)
            t_min = min(t_min, ts)
            t_max = max(t_max, ts + dur)

        if t_min == float("inf"):
            return events

        span = t_max - t_min
        t_start = t_min + span * (start_pct / 100.0)
        t_end = t_min + span * (end_pct / 100.0)

        result = []
        for e in events:
            ts = e.get("ts")
            if ts is None:
                continue
            dur = e.get("dur", 0)
            if (ts + dur) >= t_start and ts <= t_end:
                result.append(e)

        return result

    # -- trace loading ------------------------------------------------------

    @staticmethod
    def _load_trace_data(
        trace_file: Path,
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        """Load Chrome-trace data and events from .json or .json.gz."""
        if trace_file.name.endswith(".json.gz"):
            opener = gzip.open(trace_file, "rt")
        else:
            opener = open(trace_file, "r")

        with opener as f:
            data = json.load(f)

        if isinstance(data, dict):
            events = data.get("traceEvents", [])
        elif isinstance(data, list):
            events = data
        else:
            events = []

        return data, events

    # -- category filtering -------------------------------------------------

    def _filter_by_category(
        self, events: List[Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        """
        Filter events by category using case-insensitive substring matching.

        Returns list of (name, duration_us) tuples.
        """
        allowed = self.config.categories
        ignored = self.config.ignore_categories

        result: List[Tuple[str, float]] = []
        for ev in events:
            cat = (ev.get("cat") or "").lower()

            if ignored and any(ig.lower() in cat for ig in ignored):
                continue

            if allowed and not any(c.lower() in cat for c in allowed):
                continue

            name = ev.get("name", "<?>")
            dur_us = ev.get("dur", 0.0)
            result.append((name, dur_us))

        return result

    # -- clamped trace generation -------------------------------------------

    def _generate_clamped_trace(
        self,
        data: Any,
        events: List[Dict[str, Any]],
        trace_file: Path,
        output_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Generate a time-windowed (clamped) trace file.

        No category filtering — all events within the window are included.
        Events overlapping the window boundary are clamped to fit.
        Matches gen_kstats_clamped_traces.py generate_clamped_trace().

        Args:
            output_dir: Directory to write the clamped trace into.
                        Defaults to the same directory as *trace_file*.
        """
        start_pct = self.config.trace_start_pct
        end_pct = self.config.trace_end_pct

        t_min = float("inf")
        t_max = float("-inf")
        for e in events:
            ts = e.get("ts")
            if ts is None:
                continue
            dur = e.get("dur", 0)
            t_min = min(t_min, ts)
            t_max = max(t_max, ts + dur)

        if t_min == float("inf"):
            return None

        span = t_max - t_min
        t_start = t_min + span * (start_pct / 100.0)
        t_end = t_min + span * (end_pct / 100.0)

        filtered = []
        for e in events:
            ts = e.get("ts")
            if ts is None:
                continue
            dur = e.get("dur", 0)
            if (ts + dur) < t_start or ts > t_end:
                continue

            e_new = dict(e)
            clamped_start = max(ts, t_start)
            clamped_end = min(ts + dur, t_end)
            e_new["ts"] = clamped_start
            e_new["dur"] = clamped_end - clamped_start
            filtered.append(e_new)

        # Build output filename: <stem>_clamped_<start>_<end>.trace.json.gz
        stem = trace_file.name
        for suffix in (".json.gz", ".json"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break

        dest = output_dir if output_dir else trace_file.parent
        output_path = dest / f"{stem}_clamped_{start_pct}_{end_pct}.trace.json.gz"

        if isinstance(data, dict) and "traceEvents" in data:
            data_out = dict(data)
            data_out["traceEvents"] = filtered
        else:
            data_out = filtered

        with gzip.open(output_path, "wt") as f:
            json.dump(data_out, f)

        logger.info(
            f"Wrote clamped trace ({len(filtered)}/{len(events)} events): "
            f"{output_path}"
        )
        return output_path

    # -- aggregation --------------------------------------------------------

    @staticmethod
    def _aggregate_stats(
        events: List[Tuple[str, float]],
    ) -> List[KernelStat]:
        """Aggregate events by name, sorted by total duration descending."""
        by_name: Dict[str, KernelStat] = {}
        for name, dur_us in events:
            if name not in by_name:
                by_name[name] = KernelStat(name=name)
            ks = by_name[name]
            ks.total_duration_us += dur_us
            ks.calls += 1
            ks.durations_us.append(dur_us)

        return sorted(by_name.values(), key=lambda k: -k.total_duration_us)

    # -- merge ranks --------------------------------------------------------

    @staticmethod
    def _merge_ranks(rank_results: List[RankResult]) -> List[KernelStat]:
        """Merge kernel stats across ranks."""
        merged: Dict[str, KernelStat] = {}
        for rr in rank_results:
            for ks in rr.kernels:
                if ks.name not in merged:
                    merged[ks.name] = KernelStat(name=ks.name)
                m = merged[ks.name]
                m.total_duration_us += ks.total_duration_us
                m.calls += ks.calls
                m.durations_us.extend(ks.durations_us)

        return sorted(merged.values(), key=lambda k: -k.total_duration_us)


def _extract_rank(filename: str) -> Optional[int]:
    """Extract rank number from vLLM trace filenames."""
    m = re.search(r"(?:-rank-|_rank)(\d+)(?:\.|_)", filename)
    return int(m.group(1)) if m else None
