###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Tests for the Latency evaluation stage.

Coverage:
  * LatencyConfig.method == "auto" selection table per KernelType
  * BenchTarget round-trip through to_dict / from_dict
  * LatencyStats summary derivation from raw samples
  * rocprofv3 --kernel-trace CSV parser
  * pmc_perf.csv parser (rocprof_timestamps method)
  * Skip-on-no-torch integration test for do_bench_cudagraph
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from Magpie.bench import LatencyStats, MAGPIE_LATENCY_JSON_MARKER
from Magpie.config import (
    BenchTarget,
    KernelType,
    LatencyConfig,
)
from Magpie.eval.latency import (
    _aggregate_per_kernel_durations_ns,
    _find_rocprofv3_csv,
    _parse_marker_line,
    _parse_pmc_perf_csv_for_durations,
    _parse_rocprofv3_kernel_trace_csv,
    _summary_stats_from_per_kernel,
)


# ---------------------------------------------------------------------------
# LatencyConfig.method == "auto" selection table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kernel_type,expected",
    [
        (KernelType.HIP, "rocprof_timestamps"),
        (KernelType.TRITON, "both"),
        (KernelType.PYTORCH, "both"),
        (KernelType.CUDA, "both"),
        (None, "cuda_graph"),
    ],
)
def test_latency_config_auto_method_selection(kernel_type, expected):
    cfg = LatencyConfig(method="auto", kernel_type=kernel_type)
    assert cfg.resolve_method() == expected


def test_latency_config_explicit_method_passthrough():
    cfg = LatencyConfig(method="kernel_trace", kernel_type=KernelType.TRITON)
    assert cfg.resolve_method() == "kernel_trace"

    cfg2 = LatencyConfig(method="none", kernel_type=KernelType.HIP)
    assert cfg2.resolve_method() == "none"


def test_latency_config_validates_method():
    with pytest.raises(ValueError):
        LatencyConfig(method="bogus")


def test_latency_config_validates_primary_metric():
    with pytest.raises(ValueError):
        LatencyConfig(primary_metric="bogus")


def test_latency_config_round_trip_from_dict():
    cfg = LatencyConfig.from_dict(
        {
            "enabled": False,
            "method": "kernel_trace",
            "primary_metric": "kernel_median_ms",
            "rep_ms": 30,
            "n_retries": 9,
            "seed": 7,
            "pythonpath": ["/a", "/b"],
            "bench_target": {
                "module": "m.foo",
                "callable": "f",
                "get_inputs": "ginps",
            },
        },
        kernel_type=KernelType.TRITON,
        gpu_arch="gfx942",
    )

    assert cfg.enabled is False
    assert cfg.method == "kernel_trace"
    assert cfg.primary_metric == "kernel_median_ms"
    assert cfg.rep_ms == 30
    assert cfg.n_retries == 9
    assert cfg.seed == 7
    assert cfg.pythonpath == ["/a", "/b"]
    assert cfg.bench_target is not None
    assert cfg.bench_target.module == "m.foo"
    assert cfg.bench_target.callable == "f"
    assert cfg.bench_target.get_inputs == "ginps"
    assert cfg.kernel_type is KernelType.TRITON
    assert cfg.gpu_arch == "gfx942"


# ---------------------------------------------------------------------------
# BenchTarget
# ---------------------------------------------------------------------------


def test_bench_target_from_dict_requires_module_and_callable():
    assert BenchTarget.from_dict(None) is None
    assert BenchTarget.from_dict({}) is None
    assert BenchTarget.from_dict({"module": "m"}) is None
    assert BenchTarget.from_dict({"callable": "c"}) is None

    bt = BenchTarget.from_dict({"module": "m", "callable": "c"})
    assert bt is not None
    assert bt.module == "m"
    assert bt.callable == "c"
    assert bt.get_inputs == "get_inputs"


# ---------------------------------------------------------------------------
# LatencyStats
# ---------------------------------------------------------------------------


def test_latency_stats_from_samples_basic():
    stats = LatencyStats.from_samples(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        n_repeat=10,
        n_retries=5,
        estimate_ms=2.5,
    )
    assert stats.median_ms == 3.0
    assert stats.min_ms == 1.0
    assert stats.max_ms == 5.0
    assert stats.p50_ms == pytest.approx(3.0)
    assert stats.p99_ms == pytest.approx(5.0 - (5.0 - 4.0) * (1 - 0.96))  # interp
    assert stats.std_ms > 0
    assert stats.n_repeat == 10
    assert stats.n_retries == 5
    assert stats.estimate_ms == 2.5
    assert len(stats.samples_ms) == 5


def test_latency_stats_from_samples_empty():
    stats = LatencyStats.from_samples([], n_repeat=1, n_retries=0)
    assert stats.median_ms == 0.0
    assert stats.samples_ms == []


def test_latency_stats_dict_round_trip():
    stats = LatencyStats.from_samples([1.0, 2.0], n_repeat=4, n_retries=2)
    restored = LatencyStats.from_dict(stats.to_dict())
    assert restored is not None
    assert restored.median_ms == stats.median_ms
    assert restored.n_repeat == 4
    assert restored.n_retries == 2


# ---------------------------------------------------------------------------
# rocprofv3 --kernel-trace CSV parsing
# ---------------------------------------------------------------------------


def _write_rocprofv3_csv(path: Path, rows):
    header = "Kernel_Name,Start_Timestamp,End_Timestamp\n"
    body = "\n".join(",".join(str(c) for c in row) for row in rows)
    path.write_text(header + body + "\n")


def test_parse_rocprofv3_kernel_trace_csv_basic(tmp_path: Path):
    csv = tmp_path / "kernel_trace.csv"
    _write_rocprofv3_csv(
        csv,
        [
            ("triton_scaled_mm_kernel", 1000, 2000),  # 1000 ns
            ("triton_scaled_mm_kernel", 3000, 4500),  # 1500 ns
            ("triton_scaled_mm_kernel", 5000, 6000),  # 1000 ns
            ("__hip_some_runtime_thunk", 7000, 7100),  # filtered out
            ("other_kernel", 8000, 9000),  # 1000 ns
        ],
    )

    per_kernel_ns = _parse_rocprofv3_kernel_trace_csv(csv)
    assert "triton_scaled_mm_kernel" in per_kernel_ns
    assert "__hip_some_runtime_thunk" not in per_kernel_ns
    assert per_kernel_ns["triton_scaled_mm_kernel"] == [1000.0, 1500.0, 1000.0]
    assert per_kernel_ns["other_kernel"] == [1000.0]


def test_parse_rocprofv3_kernel_trace_csv_kernel_filter(tmp_path: Path):
    csv = tmp_path / "kernel_trace.csv"
    _write_rocprofv3_csv(
        csv,
        [
            ("triton_scaled_mm_kernel", 1000, 2000),
            ("other_kernel", 3000, 4000),
        ],
    )

    per_kernel_ns = _parse_rocprofv3_kernel_trace_csv(
        csv, kernel_filter_re=r"triton_"
    )
    assert "triton_scaled_mm_kernel" in per_kernel_ns
    assert "other_kernel" not in per_kernel_ns


def test_parse_rocprofv3_kernel_trace_csv_missing_file(tmp_path: Path):
    csv = tmp_path / "nope.csv"
    assert _parse_rocprofv3_kernel_trace_csv(csv) == {}


def test_find_rocprofv3_csv_locates_kernel_trace(tmp_path: Path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    target = out_dir / "myrun_kernel_trace.csv"
    target.write_text("Kernel_Name,Start_Timestamp,End_Timestamp\n")
    found = _find_rocprofv3_csv(out_dir)
    assert found == target


def test_find_rocprofv3_csv_returns_none_when_empty(tmp_path: Path):
    out_dir = tmp_path / "empty"
    out_dir.mkdir()
    assert _find_rocprofv3_csv(out_dir) is None


# ---------------------------------------------------------------------------
# pmc_perf.csv parsing (rocprof_timestamps)
# ---------------------------------------------------------------------------


def test_parse_pmc_perf_csv_for_durations(tmp_path: Path):
    csv = tmp_path / "pmc_perf.csv"
    csv.write_text(
        "Kernel_Name,Start_Timestamp,End_Timestamp,Other\n"
        "vector_add,1000,5000,x\n"
        "vector_add,6000,9000,x\n"
        "vector_add,abc,def,x\n"
        "__hip_thunk,1,2,x\n"
    )

    per_kernel_ns = _parse_pmc_perf_csv_for_durations(csv)
    assert "vector_add" in per_kernel_ns
    assert per_kernel_ns["vector_add"] == [4000.0, 3000.0]
    assert "__hip_thunk" not in per_kernel_ns


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def test_aggregate_per_kernel_durations_ns_to_stats():
    per_kernel_ns = {
        "k1": [1_000_000.0, 2_000_000.0, 3_000_000.0],  # 1, 2, 3 ms
        "k2": [],  # dropped
        "k3": [-5.0, 0.0],  # dropped (no positive samples)
    }
    out = _aggregate_per_kernel_durations_ns(per_kernel_ns)
    assert "k1" in out
    assert "k2" not in out
    assert "k3" not in out
    assert out["k1"].median_ms == 2.0
    assert out["k1"].min_ms == 1.0
    assert out["k1"].max_ms == 3.0


def test_summary_stats_from_per_kernel():
    per_kernel = {
        "a": LatencyStats.from_samples([1.0, 2.0, 3.0], n_repeat=1, n_retries=3),
        "b": LatencyStats.from_samples([0.5, 0.5, 0.5], n_repeat=1, n_retries=3),
    }
    summary = _summary_stats_from_per_kernel(per_kernel)
    assert summary is not None
    # median is sum-of-medians = 2.0 + 0.5 = 2.5
    assert summary.median_ms == pytest.approx(2.5)
    assert summary.p50_ms == pytest.approx(2.5)
    # min/max derived from the merged sample set
    assert summary.min_ms == 0.5
    assert summary.max_ms == 3.0


def test_summary_stats_from_empty_per_kernel():
    assert _summary_stats_from_per_kernel({}) is None


# ---------------------------------------------------------------------------
# MAGPIE_LATENCY_JSON marker parsing
# ---------------------------------------------------------------------------


def test_parse_marker_line_picks_up_payload():
    payload = {"stats": {"median_ms": 0.42}, "module": "m"}
    output = textwrap.dedent(
        f"""\
        garbage line 1
        unrelated output
        {MAGPIE_LATENCY_JSON_MARKER} {json.dumps(payload)}
        trailing log line
        """
    )
    assert _parse_marker_line(output) == payload


def test_parse_marker_line_no_marker():
    assert _parse_marker_line("nothing here") is None
    assert _parse_marker_line("") is None


def test_parse_marker_line_picks_last_marker():
    p1 = {"stats": {"median_ms": 1.0}}
    p2 = {"stats": {"median_ms": 2.0}}
    output = (
        f"{MAGPIE_LATENCY_JSON_MARKER} {json.dumps(p1)}\n"
        f"{MAGPIE_LATENCY_JSON_MARKER} {json.dumps(p2)}\n"
    )
    # The last marker wins (closest to "final result")
    assert _parse_marker_line(output) == p2


# ---------------------------------------------------------------------------
# Optional torch integration test
# ---------------------------------------------------------------------------


def test_do_bench_cudagraph_smoke():
    """
    Smoke test that exercises the real do_bench_cudagraph code path on a
    trivial elementwise add. Skipped automatically when torch is missing or
    no GPU is present (so the suite stays green on CPU-only CI).
    """
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("torch.cuda.is_available() is False")

    from Magpie.bench import do_bench_cudagraph

    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")

    def fn():
        torch.add(a, b, out=a)

    stats = do_bench_cudagraph(fn, rep=5, n_retries=3, estimate_reps=3)

    assert stats.median_ms > 0
    assert stats.n_retries == 3
    assert stats.n_repeat >= 1
    assert len(stats.samples_ms) == 3
