import gzip
import json

import pytest
import yaml

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


# ---------------------------------------------------------------------------
# atom PROFILE=1 wiring (atom_mi*x.sh)
# ---------------------------------------------------------------------------
# The atom launch script translates PROFILE=1 to atom's
# --torch-profiler-dir CLI flag (added to the openai_server.py invocation)
# and points the directory at $ATOM_TORCH_PROFILER_DIR (set by Magpie's
# benchmarker.py for parity with VLLM_/SGLANG_TORCH_PROFILER_DIR) with a
# workspace-local fallback. These tests are static content checks on the
# script files because spawning bash + atom would require a GPU, but the
# checks catch the regression "someone reverted to the no-op warning".
@pytest.mark.parametrize("runner", ["atom_mi300x.sh", "atom_mi355x.sh"])
def test_atom_launch_script_wires_profile_to_torch_profiler_dir(runner):
    from pathlib import Path

    script = (
        Path(__file__).resolve().parent.parent
        / "Magpie"
        / "scripts"
        / "benchmark"
        / runner
    )
    text = script.read_text(encoding="utf-8")
    # The old "PROFILE=1 ... ignoring" warning must not regress.
    assert "not yet implemented; ignoring" not in text, (
        f"{runner} regressed to the pre-wiring no-op warning"
    )
    # PROFILE=1 branch builds a PROFILER_ARGS array...
    assert 'PROFILER_ARGS+=(--torch-profiler-dir' in text, (
        f"{runner} missing --torch-profiler-dir flag construction"
    )
    # ...defaulted to ATOM_TORCH_PROFILER_DIR (parity with vLLM/SGLang)
    # or workspace torch_trace/...
    assert "ATOM_TORCH_PROFILER_DIR" in text, (
        f"{runner} should honour ATOM_TORCH_PROFILER_DIR for parity with the "
        "other frameworks (benchmarker.py exports it)"
    )
    assert "WORKSPACE_DIR/torch_trace" in text, (
        f"{runner} should fall back to $WORKSPACE_DIR/torch_trace so "
        "inference_optimizer's _candidate_trace_dirs probe finds the trace"
    )
    # ...and the array is expanded into the atom server launch command.
    # (The PROFILER_ARGS expansion must appear *before* the EXTRA_ATOM_ARGS
    # passthrough so EXTRA_ATOM_ARGS values can still override.)
    assert '"${PROFILER_ARGS[@]}"' in text, (
        f"{runner} builds PROFILER_ARGS but never expands it into the server "
        "launch command — the profiler dir flag won't reach atom"
    )


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
    """atom writes torch traces to ``<trace_dir>/rank_<N>/*.pt.trace.json.gz``
    (per atom/model_engine/model_runner.py::start_profiler — config
    torch_profiler_dir is joined with a per-rank subdir). A non-recursive
    glob would miss them entirely. Regression guard: drop a trace inside
    a rank_0/ subdirectory and verify ResultParser.parse_torch_trace
    still picks it up."""
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

    assert [k.name for k in kernels] == ["atom_moe_kernel"], (
        "parse_torch_trace must rglob through rank_<N>/ subdirs to pick "
        "up atom-style nested traces"
    )
    assert kernels[0].time_ms == 3.0
    assert kernels[0].calls == 2


def test_tracelens_find_trace_files_walks_atom_rank_subdirs(tmp_path):
    """Same nested-layout regression guard, but for
    TraceLensAnalyzer._find_trace_files. Without a recursive glob, the
    TraceLens analyze pipeline gets an empty trace list and emits an
    empty perf report when the user runs PROFILE=1 + framework=atom."""
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
    assert flat_path.resolve() in found_set, (
        "_find_trace_files dropped the flat vllm/sglang trace — recursive "
        "glob must still match top-level files"
    )
    assert nested_path.resolve() in found_set, (
        "_find_trace_files dropped the nested atom rank_<N>/ trace — "
        "recursive glob (rglob) is required for atom support"
    )


def test_benchmark_result_summary_includes_sections():
    result = BenchmarkResult(success=True, framework="vllm", model="demo-model")
    result.errors.append("example warning")

    summary = result.get_summary()

    assert "Benchmark Result: VLLM" in summary
    assert "Status: SUCCESS" in summary
    assert "Errors:" in summary
