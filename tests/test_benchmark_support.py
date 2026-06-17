import csv
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
from Magpie.modes.benchmark.tracelens_inference import (
    SGLANG_SHAPE_DISCOVERY_FLAG,
    TraceLensInferencePipeline,
    append_flag_value_args,
    compute_steady_state_iters,
    is_tracelens_patched_sglang_image,
    trace_arch_platform_from_runner,
)
from Magpie.modes.benchmark.tracelens_runtime import (
    derive_tracelens_image_tag,
    infer_sglang_patch_version,
    infer_vllm_patch_version,
    is_tracelens_ready_runtime_image,
    prepare_tracelens_runtime_image,
    runner_type_to_gpu_type,
)
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
    assert cfg.analysis_mode == "inference"
    assert cfg.analysis_stages == ["prefilldecode", "decode", "prefill"]
    assert cfg.cli_timeout_seconds == 1800
    assert cfg.auto_patch_runtime is True
    assert cfg.export_format == "excel"
    assert cfg.export_excel is True
    assert cfg.export_csv is False


def test_tracelens_config_normalizes_inference_stages():
    cfg = TraceLensConfig.from_dict(
        {
            "enabled": True,
            "analysis_stages": ["decode", "mixed", "prefill"],
            "cli_timeout_seconds": 2400,
            "auto_patch_runtime": False,
        }
    )

    assert cfg.analysis_stages == ["decode", "prefilldecode", "prefill"]
    assert cfg.cli_timeout_seconds == 2400
    assert cfg.auto_patch_runtime is False

    cfg = TraceLensConfig.from_dict(
        {"enabled": True, "analysis_mode": "classic", "analysis_stages": "all"}
    )

    assert cfg.analysis_mode == "pytorch"
    assert cfg.analysis_stages == ["prefilldecode", "decode", "prefill"]


def test_tracelens_inference_iteration_and_arg_helpers():
    max_iters, delay_iters = compute_steady_state_iters(1024, 64, 1)

    assert max_iters == 256
    assert delay_iters == 6016

    envs = {"EXTRA_VLLM_ARGS": "--existing true"}
    append_flag_value_args(
        envs,
        "EXTRA_VLLM_ARGS",
        [
            ("--existing", "false"),
            ("--new-flag", "value"),
        ],
    )

    assert envs["EXTRA_VLLM_ARGS"] == "--existing true --new-flag value"
    assert trace_arch_platform_from_runner("mi355x") == "MI355X"
    assert trace_arch_platform_from_runner("mi300") == "MI300X"
    assert is_tracelens_patched_sglang_image("tracelens-sglang:0.5.12")
    assert not is_tracelens_patched_sglang_image("lmsysorg/sglang:latest")

    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "sglang",
            "model": "demo",
            "envs": {"TL_EXTENSION": "TraceLens_NDA"},
        }
    )
    assert not TraceLensInferencePipeline(cfg)._should_enable_sglang_shape_discovery()


def test_tracelens_runtime_image_helpers():
    assert infer_sglang_patch_version("lmsysorg/sglang:v0.5.12-rocm720-mi35x") == "0.5.12"
    assert infer_sglang_patch_version("lmsysorg/sglang:v0.5.8") is None
    assert infer_vllm_patch_version("vllm/vllm-openai-rocm:v0.19.1") == "v19"
    assert infer_vllm_patch_version("vllm/vllm-openai-rocm:v0.13.0") is None
    assert runner_type_to_gpu_type("mi355x") == "mi355"
    assert is_tracelens_ready_runtime_image("sglang", "tracelens-sglang:0.5.12")
    assert not is_tracelens_ready_runtime_image("vllm", "lmsysorg/sglang:v0.5.12")

    tag = derive_tracelens_image_tag(
        "sglang",
        "lmsysorg/sglang:v0.5.12-rocm720-mi35x",
        "mi355x",
        "0.5.12",
    )
    assert tag.startswith("magpie-tracelens-sglang:0_5_12-mi355x-")


def test_prepare_tracelens_runtime_image_reuses_existing_derived_image(
    tmp_path,
    monkeypatch,
):
    tracelens_repo = tmp_path / "TraceLens"
    workflow_dir = tracelens_repo / "examples/custom_workflows/inference_analysis"
    workflow_dir.mkdir(parents=True)
    monkeypatch.setenv("TRACELENS_REPO_PATH", str(tracelens_repo))
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.docker_image_exists",
        lambda _image: True,
    )

    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "sglang",
            "model": "demo",
            "docker_image": "lmsysorg/sglang:v0.5.12-rocm720-mi35x",
            "profiler": {"tracelens": {"enabled": True}},
        }
    )

    result = prepare_tracelens_runtime_image(
        cfg,
        base_image=cfg.docker_image,
        runner_type="mi355x",
    )

    assert result["built"] is False
    assert result["reason"] == "derived image already exists"
    assert result["image"].startswith("magpie-tracelens-sglang:0_5_12-mi355x-")


def test_tracelens_inference_prepare_patches_and_restores(tmp_path):
    inferencex = tmp_path / "InferenceX"
    bench_dir = inferencex / "benchmarks"
    serving_dir = inferencex / "utils" / "bench_serving"
    bench_dir.mkdir(parents=True)
    serving_dir.mkdir(parents=True)

    benchmark_lib = bench_dir / "benchmark_lib.sh"
    benchmark_lib.write_text(
        'if [[ "${PROFILE:-}" == "1" ]]; then\n'
        '    num_prompts="$max_concurrency"\n'
        "fi\n",
        encoding="utf-8",
    )

    benchmark_serving = serving_dir / "benchmark_serving.py"
    benchmark_serving.write_text(
        'extra_body={"num_steps": 1, "merge_profiles": True, '
        '"profile_by_stage": True},\n',
        encoding="utf-8",
    )

    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "sglang",
            "model": "demo",
            "inferencex_path": str(inferencex),
            "envs": {
                "CONC": 64,
                "OSL": 1024,
                "RANDOM_RANGE_RATIO": 1,
            },
            "profiler": {"tracelens": {"enabled": True}},
        }
    )
    pipeline = TraceLensInferencePipeline(cfg)

    result = pipeline.prepare(tmp_path / "workspace")

    assert result["max_iterations"] == 256
    assert result["delay_iterations"] == 6016
    assert 'num_prompts="$num_prompts"' in benchmark_lib.read_text(encoding="utf-8")
    patched_serving = benchmark_serving.read_text(encoding="utf-8")
    assert '"roofline_annotations": True' in patched_serving
    assert '"start_step": 6016' in patched_serving
    assert '"num_steps": 256' in patched_serving
    assert cfg.envs["SGLANG_PROFILE_WITH_STACK"] == "True"
    assert "--enable-profile-cuda-graph" in cfg.envs["EXTRA_SGLANG_ARGS"]
    assert SGLANG_SHAPE_DISCOVERY_FLAG not in cfg.envs["EXTRA_SGLANG_ARGS"]

    restore = pipeline.restore()

    assert str(benchmark_lib) in restore["restored_files"]
    assert str(benchmark_serving) in restore["restored_files"]
    assert 'num_prompts="$max_concurrency"' in benchmark_lib.read_text(encoding="utf-8")
    assert "roofline_annotations" not in benchmark_serving.read_text(encoding="utf-8")


def test_tracelens_inference_prepare_enables_sglang_shape_discovery_for_patched_image(
    tmp_path,
):
    inferencex = tmp_path / "InferenceX"
    bench_dir = inferencex / "benchmarks"
    serving_dir = inferencex / "utils" / "bench_serving"
    bench_dir.mkdir(parents=True)
    serving_dir.mkdir(parents=True)

    (bench_dir / "benchmark_lib.sh").write_text(
        'if [[ "${PROFILE:-}" == "1" ]]; then\n'
        '    num_prompts="$max_concurrency"\n'
        "fi\n",
        encoding="utf-8",
    )
    (serving_dir / "benchmark_serving.py").write_text(
        'extra_body={"num_steps": 1, "merge_profiles": True, '
        '"profile_by_stage": True},\n',
        encoding="utf-8",
    )

    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "sglang",
            "model": "demo",
            "docker_image": "tracelens-sglang:0.5.12-mi355-fix",
            "inferencex_path": str(inferencex),
            "envs": {
                "CONC": 64,
                "OSL": 1024,
                "RANDOM_RANGE_RATIO": 1,
            },
            "profiler": {"tracelens": {"enabled": True}},
        }
    )

    result = TraceLensInferencePipeline(cfg).prepare(tmp_path / "workspace")

    assert SGLANG_SHAPE_DISCOVERY_FLAG in cfg.envs["EXTRA_SGLANG_ARGS"]
    assert SGLANG_SHAPE_DISCOVERY_FLAG in result["env_updates"]["EXTRA_SGLANG_ARGS"]


def test_tracelens_inference_sglang_step_marker_fallback(tmp_path):
    trace_path = tmp_path / "rank0.trace.json.gz"
    trace = {
        "traceEvents": [
            {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "rank0"}},
            {
                "cat": "user_annotation",
                "name": "step[DECODE bs=32]",
                "ts": 100.0,
                "dur": 10.0,
            },
            {
                "cat": "user_annotation",
                "name": "step[DECODE bs=64]",
                "ts": 200.0,
                "dur": 20.0,
            },
            {
                "cat": "user_annotation",
                "name": "step[EXTEND bs=4 toks=4096]",
                "ts": 300.0,
                "dur": 30.0,
            },
            {"cat": "cpu_op", "name": "inside_decode", "ts": 205.0, "dur": 2.0},
            {"cat": "kernel", "name": "decode_kernel", "ts": 225.0, "dur": 5.0},
            {"cat": "cpu_op", "name": "outside", "ts": 500.0, "dur": 2.0},
        ]
    }
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        json.dump(trace, handle)

    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "sglang",
            "model": "demo",
            "envs": {"CONC": 64, "OSL": 1024},
            "profiler": {"tracelens": {"enabled": True}},
        }
    )
    pipeline = TraceLensInferencePipeline(cfg)
    split_dir = tmp_path / "split"

    warnings = pipeline._split_sglang_step_markers(trace_path, split_dir)

    assert "wrote 2 trace window" in warnings[-1]
    rows = list(csv.DictReader((split_dir / "execution_details.csv").open()))
    assert {row["stage"] for row in rows} == {"decode", "prefill"}
    assert {
        row["phase_avg_bs"] for row in rows
    } == {"64.0", "4.0"}
    decode_trace = split_dir / "decode_only_step.trace.json.gz"
    with gzip.open(decode_trace, "rt", encoding="utf-8") as handle:
        decode_events = json.load(handle)["traceEvents"]
    assert any(event.get("name") == "inside_decode" for event in decode_events)
    assert any(event.get("name") == "decode_kernel" for event in decode_events)
    assert not any(event.get("name") == "outside" for event in decode_events)


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


def test_image_selector_selects_override_and_arch_mapping(tmp_path, monkeypatch):
    config_path = tmp_path / "images.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "vllm": {"gfx942": "amd/vllm:mi300x", "sm_90": "nvidia/vllm:h100"},
                "sglang": {"gfx950": "amd/sglang:mi355x"},
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


def test_benchmark_result_summary_includes_sections():
    result = BenchmarkResult(success=True, framework="vllm", model="demo-model")
    result.errors.append("example warning")

    summary = result.get_summary()

    assert "Benchmark Result: VLLM" in summary
    assert "Status: SUCCESS" in summary
    assert "Errors:" in summary
