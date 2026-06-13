import gzip
import json

import pytest
import yaml

from Magpie.modes.benchmark.benchmarker import BenchmarkMode
from Magpie.modes.benchmark.config import (
    BenchmarkConfig,
    GapAnalysisConfig,
    ProfilerConfig,
    RayConfig,
    ServerLifecycleConfig,
    TorchProfilerConfig,
    TraceLensConfig,
)
from Magpie.modes.benchmark.image_selector import ImageSelector
from Magpie.modes.benchmark.result import BenchmarkResult, ResultParser
from Magpie.utils.gpu import GPUVendor


def test_benchmark_server_lifecycle_requires_local_runtime():
    with pytest.raises(ValueError, match="server_lifecycle"):
        BenchmarkConfig(
            framework="vllm",
            model="demo",
            run_mode="docker",
            envs={
                "TP": 1,
                "CONC": 32,
                "ISL": 1024,
                "OSL": 512,
                "RANDOM_RANGE_RATIO": 0.5,
            },
            profiler=ProfilerConfig(
                torch_profiler=TorchProfilerConfig(enabled=False),
            ),
            server_lifecycle=ServerLifecycleConfig(enabled=True),
        )


def test_benchmark_server_lifecycle_rejects_profiler_without_cleanup():
    with pytest.raises(ValueError, match="torch_profiler"):
        BenchmarkConfig(
            framework="vllm",
            model="demo",
            run_mode="local",
            envs={
                "TP": 1,
                "CONC": 32,
                "ISL": 1024,
                "OSL": 512,
                "RANDOM_RANGE_RATIO": 0.5,
            },
            profiler=ProfilerConfig(
                torch_profiler=TorchProfilerConfig(enabled=True),
            ),
            server_lifecycle=ServerLifecycleConfig(enabled=True, cleanup=False),
        )


def test_benchmark_server_lifecycle_sweep_conflict():
    with pytest.raises(ValueError, match="sweep_matrix"):
        BenchmarkConfig.from_dict(
            {
                "framework": "vllm",
                "model": "demo",
                "run_mode": "local",
                "profiler": {"torch_profiler": {"enabled": False}},
                "envs": {
                    "TP": 1,
                    "CONC": 32,
                    "ISL": 1024,
                    "OSL": 512,
                    "RANDOM_RANGE_RATIO": 0.5,
                },
                "server_lifecycle": {"enabled": True},
                "sweep_matrix": {"cases": [{"CONC": 2}]},
            }
        )


def test_benchmark_server_lifecycle_from_dict_ok():
    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "vllm",
            "model": "demo",
            "run_mode": "local",
            "profiler": {"torch_profiler": {"enabled": False}},
            "envs": {"PORT": 8899},
            "server_lifecycle": {"enabled": True},
        }
    )

    assert cfg.is_server_lifecycle is True


def test_tracelens_config_supports_legacy_export_flags():
    cfg = TraceLensConfig.from_dict({"enabled": True, "export_excel": True})

    assert cfg.enabled is True
    assert cfg.export_format == "excel"
    assert cfg.export_excel is True
    assert cfg.export_csv is False


def test_gap_analysis_config_validates_window():
    with pytest.raises(ValueError):
        GapAnalysisConfig(trace_start_pct=80, trace_end_pct=80)

    with pytest.raises(ValueError):
        GapAnalysisConfig(trace_start_pct=-1, trace_end_pct=50)


def test_benchmark_config_from_dict_normalizes_nested_sections():
    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "VLLM",
            "model": "test-model",
            "run_mode": "ray",
            "profiler": {
                "torch_profiler": {"enabled": False},
                "tracelens": {"enabled": True, "export_format": "csv"},
            },
            "gap_analysis": {
                "enabled": True,
                "trace_start_pct": 10,
                "trace_end_pct": 90,
            },
            "ray_config": {"cluster_address": "auto", "num_nodes": 2},
            "inferencemax_path": "/tmp/inferencex",
        }
    )

    assert cfg.framework == "vllm"
    assert cfg.is_ray is True
    assert cfg.profiler.torch_profiler.enabled is False
    assert cfg.profiler.tracelens.enabled is True
    assert cfg.gap_analysis.trace_start_pct == 10
    assert isinstance(cfg.ray_config, RayConfig)
    assert cfg.ray_config.num_nodes == 2
    assert cfg.inferencex_path == "/tmp/inferencex"
    assert cfg.get_env_vars()["MODEL"] == "test-model"


def test_benchmark_config_sets_defaults_and_script_name():
    cfg = BenchmarkConfig(framework="sglang", model="demo")

    assert cfg.envs["TP"] == 1
    assert cfg.envs["CONC"] == 32
    assert cfg.get_benchmark_script_name() == "generic_fp8_mi300x.sh"

    cfg.runner_type = "h100"
    cfg.precision = "bf16"
    assert cfg.get_benchmark_script_name() == "generic_bf16_h100.sh"


def test_benchmark_config_accepts_atom_framework():
    cfg = BenchmarkConfig(framework="atom", model="demo")

    assert cfg.framework == "atom"
    assert cfg.envs["TP"] == 1

    cfg_from_dict = BenchmarkConfig.from_dict(
        {
            "framework": "ATOM",
            "model": "demo",
            "run_mode": "local",
            "profiler": {"torch_profiler": {"enabled": False}},
        }
    )
    assert cfg_from_dict.framework == "atom"


@pytest.mark.parametrize("runner", ["atom_mi300x.sh", "atom_mi355x.sh"])
def test_atom_launch_script_wires_profile_to_torch_profiler_dir(runner):
    """PROFILE=1 must build --torch-profiler-dir from ATOM_TORCH_PROFILER_DIR
    (with a workspace fallback) and pass it to the atom server."""
    from pathlib import Path

    script = (
        Path(__file__).resolve().parent.parent
        / "Magpie"
        / "scripts"
        / "benchmark"
        / runner
    )
    text = script.read_text(encoding="utf-8")
    assert "PROFILER_ARGS+=(--torch-profiler-dir" in text
    assert "ATOM_TORCH_PROFILER_DIR" in text
    assert "WORKSPACE_DIR/torch_trace" in text
    assert '"${PROFILER_ARGS[@]}"' in text


@pytest.mark.parametrize("runner", ["atom_mi300x.sh", "atom_mi355x.sh"])
def test_atom_launch_script_forwards_max_model_len(runner):
    """MAX_MODEL_LEN is configured by the example and must reach the server."""
    from pathlib import Path

    script = (
        Path(__file__).resolve().parent.parent
        / "Magpie"
        / "scripts"
        / "benchmark"
        / runner
    )
    text = script.read_text(encoding="utf-8")
    assert "MAX_MODEL_LEN=${MAX_MODEL_LEN:-" in text
    assert '--max-model-len "$MAX_MODEL_LEN"' in text


def test_benchmark_script_copy_is_atomic_on_failure(tmp_path, monkeypatch):
    source = tmp_path / "source.sh"
    target = tmp_path / "target.sh"
    source.write_text("new content\n", encoding="utf-8")
    target.write_text("old content\n", encoding="utf-8")

    def fail_copy(src, dst):
        raise RuntimeError("copy failed")

    monkeypatch.setattr(
        "Magpie.modes.benchmark.benchmarker.shutil.copy2",
        fail_copy,
    )

    with pytest.raises(RuntimeError, match="copy failed"):
        BenchmarkMode._copy_benchmark_script_atomic(source, target)

    assert target.read_text(encoding="utf-8") == "old content\n"
    assert not list(tmp_path.glob(".target.sh.*.tmp"))


def test_benchmark_script_copy_sets_executable_bit(tmp_path):
    source = tmp_path / "source.sh"
    target = tmp_path / "target.sh"
    source.write_text("#!/usr/bin/env bash\necho ok\n", encoding="utf-8")

    BenchmarkMode._copy_benchmark_script_atomic(source, target)

    assert target.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert target.stat().st_mode & 0o111


def test_image_selector_selects_override_and_arch_mapping(tmp_path, monkeypatch):
    config_path = tmp_path / "images.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "vllm": {"gfx942": "amd/vllm:mi300x", "sm_90": "nvidia/vllm:h100"},
                "sglang": {"gfx950": "amd/sglang:mi355x"},
                "atom": {"gfx942": "amd/atom:mi300x"},
            }
        ),
        encoding="utf-8",
    )
    selector = ImageSelector(str(config_path))

    assert (
        selector.select_image("vllm", override_image="custom:image") == "custom:image"
    )
    assert selector.select_image("vllm", gpu_arch="gfx942") == "amd/vllm:mi300x"

    monkeypatch.setattr(
        "Magpie.modes.benchmark.image_selector.detect_gpu",
        lambda: (GPUVendor.AMD, "gfx950"),
    )
    assert selector.select_image("sglang") == "amd/sglang:mi355x"
    assert selector.select_image("atom", gpu_arch="gfx942") == "amd/atom:mi300x"
    assert selector.get_runner_type("sm_90") == "h100"

    with pytest.raises(ValueError):
        selector.select_image("unknown", gpu_arch="gfx942")

    with pytest.raises(ValueError):
        selector.get_runner_type("unknown_arch")


@pytest.mark.parametrize("gpu_arch", ["gfx942", "gfx950"])
def test_real_benchmark_images_yaml_atom_resolves_to_rocm_atom_image(gpu_arch):
    """atom must resolve to the dedicated rocm/atom image on AMD arches."""
    selector = ImageSelector()
    image = selector.select_image("atom", gpu_arch=gpu_arch)
    assert image == "rocm/atom:latest"
    assert "vllm" not in image.lower()


def test_real_benchmark_images_yaml_atom_nvidia_arch_raises():
    """atom is AMD-only; NVIDIA arches must error out, not fall back to vLLM."""
    selector = ImageSelector()
    for sm_arch in ("sm_80", "sm_90", "sm_100"):
        with pytest.raises(ValueError, match="No image found for GPU architecture"):
            selector.select_image("atom", gpu_arch=sm_arch)


def test_result_parser_parses_inferencex_json_and_missing_file(tmp_path):
    missing = ResultParser.parse_inferencex_result(tmp_path / "missing.json")
    assert missing.success is False
    assert missing.errors

    result_path = tmp_path / "inferencex_result.json"
    result_path.write_text(
        json.dumps(
            {
                "request_throughput": 12.5,
                "output_throughput": 512.0,
                "total_token_throughput": 768.0,
                "completed": 32,
                "total_input_tokens": 4096,
                "total_output_tokens": 8192,
                "duration": 10.0,
                "mean_ttft_ms": 3.5,
                "p99_e2el_ms": 42.0,
                "model_id": "from-file-model",
            }
        ),
        encoding="utf-8",
    )

    parsed = ResultParser.parse_inferencex_result(result_path, framework="vllm")

    assert parsed.success is True
    assert parsed.framework == "vllm"
    assert parsed.model == "from-file-model"
    assert parsed.throughput.request_throughput == 12.5
    assert parsed.latency.ttft_mean == 3.5
    assert parsed.latency.e2el_p99 == 42.0


def test_result_parser_aggregates_first_torch_trace_file(tmp_path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()

    trace = {
        "traceEvents": [
            {"cat": "kernel", "name": "kernel_a", "dur": 2000},
            {"cat": "kernel", "name": "kernel_a", "dur": 1000},
            {"cat": "kernel", "name": "kernel_b", "dur": 500},
            {"cat": "cpu_op", "name": "ignored", "dur": 999},
        ]
    }

    with gzip.open(trace_dir / "rank0.json.gz", "wt") as f:
        json.dump(trace, f)

    kernels = ResultParser.parse_torch_trace(trace_dir)

    assert [k.name for k in kernels] == ["kernel_a", "kernel_b"]
    assert kernels[0].time_ms == 3.0
    assert kernels[0].calls == 2
    assert pytest.approx(kernels[0].percent, rel=1e-6) == (3.0 / 3.5) * 100


def test_result_parser_finds_atom_nested_rank_traces(tmp_path):
    """parse_torch_trace must recurse into per-rank subdirs (atom layout)."""
    trace_dir = tmp_path / "torch_trace"
    rank_dir = trace_dir / "rank_0"
    rank_dir.mkdir(parents=True)

    trace = {
        "traceEvents": [
            {"cat": "kernel", "name": "atom_moe_kernel", "dur": 1500},
            {"cat": "kernel", "name": "atom_moe_kernel", "dur": 1500},
            {"cat": "cpu_op", "name": "ignored", "dur": 999},
        ]
    }
    with gzip.open(rank_dir / "atom_ts_20260528_120000_001.pt.trace.json.gz", "wt") as f:
        json.dump(trace, f)

    kernels = ResultParser.parse_torch_trace(trace_dir)

    assert [k.name for k in kernels] == ["atom_moe_kernel"]
    assert kernels[0].time_ms == 3.0
    assert kernels[0].calls == 2


def test_result_parser_skips_vllm_warmup_trace(tmp_path):
    """parse_torch_trace must ignore CUDA-graph warmup snapshots (issue #38).

    vLLM writes a warmup trace under capture_traces/. The parser must select the
    real benchmark trace instead, so the dominant kernels (e.g. the all-reduce)
    appear in the summary. Here the warmup file is made larger and kernel-bearing
    so the capture_traces/ skip is the only thing that can exclude it.
    """
    trace_dir = tmp_path / "torch_trace"
    capture_dir = trace_dir / "capture_traces"
    capture_dir.mkdir(parents=True)

    # The warmup trace is deliberately the LARGEST file and contains valid
    # kernel events. Neither the largest-first sort nor the kernel-presence
    # check would reject it, so only the capture_traces/ skip prevents it from
    # shadowing the real trace. This keeps the test from passing vacuously.
    # Names are unique so the file does not gzip-compress down below the real
    # trace and break the size precondition asserted below.
    warmup_trace = {
        "traceEvents": [
            {"cat": "kernel", "name": f"warmup_kernel_{i}", "dur": 100}
            for i in range(2000)
        ]
    }
    warmup_file = capture_dir / "graph_capture_rank_0.json.gz"
    with gzip.open(warmup_file, "wt") as f:
        json.dump(warmup_trace, f)

    real_trace = {
        "traceEvents": [
            {"cat": "kernel", "name": "vllm::cross_device_reduce_1stage", "dur": 9000},
            {"cat": "kernel", "name": "gemm_kernel", "dur": 1000},
            {"cat": "cpu_op", "name": "ignored", "dur": 999},
        ]
    }
    real_file = trace_dir / "dp0_pp0_tp0_rank0.json.gz"
    with gzip.open(real_file, "wt") as f:
        json.dump(real_trace, f)

    # Guard the precondition: the warmup snapshot must be the larger file, so
    # the test genuinely exercises the capture_traces/ skip rather than the
    # size heuristic.
    assert warmup_file.stat().st_size > real_file.stat().st_size

    kernels = ResultParser.parse_torch_trace(trace_dir)

    names = [k.name for k in kernels]
    assert not any(n.startswith("warmup_kernel") for n in names)
    assert names[0] == "vllm::cross_device_reduce_1stage"


def test_result_parser_falls_back_when_largest_trace_is_corrupt(tmp_path):
    """A truncated/corrupt largest trace must fall back to the next valid one."""
    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()

    # Largest file by size, but corrupt (not valid gzip/json).
    corrupt = trace_dir / "rank0_corrupt.json.gz"
    corrupt.write_bytes(b"\x00not-a-valid-gzip-trace" * 1000)

    valid_trace = {
        "traceEvents": [
            {"cat": "kernel", "name": "gemm_kernel", "dur": 4000},
        ]
    }
    with gzip.open(trace_dir / "rank1_valid.json.gz", "wt") as f:
        json.dump(valid_trace, f)

    assert corrupt.stat().st_size > (trace_dir / "rank1_valid.json.gz").stat().st_size

    kernels = ResultParser.parse_torch_trace(trace_dir)

    assert [k.name for k in kernels] == ["gemm_kernel"]
    assert kernels[0].time_ms == 4.0


def test_result_parser_skips_trace_without_kernel_events(tmp_path):
    """A larger trace with no kernel events must not shadow a smaller real one."""
    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()

    # Larger file, valid JSON, but only CPU ops (no kernel-category events).
    cpu_only = {
        "traceEvents": [
            {"cat": "cpu_op", "name": f"op_{i}", "dur": 5} for i in range(500)
        ]
    }
    with gzip.open(trace_dir / "rank0_cpu_only.json.gz", "wt") as f:
        json.dump(cpu_only, f)

    real_trace = {
        "traceEvents": [
            {"cat": "kernel", "name": "gemm_kernel", "dur": 4000},
        ]
    }
    with gzip.open(trace_dir / "rank1_real.json.gz", "wt") as f:
        json.dump(real_trace, f)

    # Ensure the kernel-less file is the larger one, so we genuinely exercise
    # the fall-through (largest-first) path rather than passing trivially.
    assert (trace_dir / "rank0_cpu_only.json.gz").stat().st_size > (
        trace_dir / "rank1_real.json.gz"
    ).stat().st_size

    kernels = ResultParser.parse_torch_trace(trace_dir)

    assert [k.name for k in kernels] == ["gemm_kernel"]


def test_tracelens_find_trace_files_walks_atom_rank_subdirs(tmp_path):
    """_find_trace_files must pick up both flat and per-rank nested traces."""
    from Magpie.modes.benchmark.config import TraceLensConfig
    from Magpie.modes.benchmark.tracelens import TraceLensAnalyzer

    trace_dir = tmp_path / "torch_trace"
    flat_path = trace_dir / "vllm_style.json.gz"
    nested_dir = trace_dir / "rank_0"
    nested_path = nested_dir / "atom_ts_20260528.pt.trace.json.gz"

    nested_dir.mkdir(parents=True)
    flat_path.write_bytes(b"")
    nested_path.write_bytes(b"")

    cfg = TraceLensConfig(enabled=True)
    analyzer = TraceLensAnalyzer(cfg)
    found = analyzer._find_trace_files(trace_dir)

    found_set = {p.resolve() for p in found}
    assert flat_path.resolve() in found_set
    assert nested_path.resolve() in found_set


def test_tracelens_detect_pattern_wildcards_atom_rank_dir(tmp_path):
    """_detect_trace_pattern must wildcard the rank_<N>/ directory component
    for the atom layout, not collapse the traces to a flat path."""
    from Magpie.modes.benchmark.config import TraceLensConfig
    from Magpie.modes.benchmark.tracelens import TraceLensAnalyzer

    trace_dir = tmp_path / "torch_trace"
    files = []
    for rank in (0, 1):
        rank_dir = trace_dir / f"rank_{rank}"
        rank_dir.mkdir(parents=True)
        f = rank_dir / "Qwen-Qwen3-8B_ts_20260528_120000.pt.trace.json.gz"
        f.write_bytes(b"")
        files.append(f)

    analyzer = TraceLensAnalyzer(TraceLensConfig(enabled=True))
    pattern = analyzer._detect_trace_pattern(trace_dir, files)

    assert pattern is not None
    assert "rank_*" in pattern
    assert pattern == str(
        trace_dir / "rank_*" / "Qwen-Qwen3-8B_ts_20260528_120000.pt.trace.json.gz"
    )


def test_tracelens_detect_pattern_wildcards_flat_filename_rank(tmp_path):
    """The vllm/sglang flat layout (rank in the filename) still works."""
    from Magpie.modes.benchmark.config import TraceLensConfig
    from Magpie.modes.benchmark.tracelens import TraceLensAnalyzer

    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()
    f = trace_dir / "trace-rank-0.pt.trace.json.gz"
    f.write_bytes(b"")

    analyzer = TraceLensAnalyzer(TraceLensConfig(enabled=True))
    pattern = analyzer._detect_trace_pattern(trace_dir, [f])

    assert pattern == str(trace_dir / "trace-rank-*.pt.trace.json.gz")


def test_gap_analysis_detect_trace_files_handles_atom_rank_dirs(tmp_path):
    """GapAnalyzer.detect_trace_files must recurse into rank_<N>/ subdirs and
    read the rank from the directory name (atom layout)."""
    from Magpie.modes.benchmark.gap_analysis import GapAnalyzer

    trace_dir = tmp_path / "torch_trace"
    expected = {}
    for rank in (0, 1, 2):
        rank_dir = trace_dir / f"rank_{rank}"
        rank_dir.mkdir(parents=True)
        f = rank_dir / "Qwen-Qwen3-8B_ts_20260528.pt.trace.json.gz"
        f.write_bytes(b"")
        expected[rank] = f.resolve()

    found = GapAnalyzer.detect_trace_files(trace_dir)

    assert [r for r, _ in found] == [0, 1, 2]
    assert {r: p.resolve() for r, p in found} == expected


def test_gap_analysis_detect_trace_files_handles_flat_rank_filenames(tmp_path):
    """The filename-encoded rank layout (vllm/sglang) still works."""
    from Magpie.modes.benchmark.gap_analysis import GapAnalyzer

    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()
    (trace_dir / "trace-rank-1.pt.trace.json.gz").write_bytes(b"")
    (trace_dir / "trace-rank-0.pt.trace.json.gz").write_bytes(b"")

    found = GapAnalyzer.detect_trace_files(trace_dir)

    assert [r for r, _ in found] == [0, 1]


def test_benchmark_result_summary_includes_sections():
    result = BenchmarkResult(success=True, framework="vllm", model="demo-model")
    result.errors.append("example warning")

    summary = result.get_summary()

    assert "Benchmark Result: VLLM" in summary
    assert "Status: SUCCESS" in summary
    assert "Errors:" in summary
