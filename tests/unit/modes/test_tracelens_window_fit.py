"""Tests for fitting the inference profiler window to short runs."""

import pytest

from Magpie.modes.benchmark.config import BenchmarkConfig
from Magpie.modes.benchmark.tracelens_inference import (
    TraceLensInferencePipeline,
    compute_steady_state_iters,
    fit_window_to_run,
)


def test_fit_window_to_run_keeps_window_that_fits():
    max_iters, delay_iters = compute_steady_state_iters(512, 8, 1)
    assert (max_iters, delay_iters) == (1024, 2560)

    assert fit_window_to_run(max_iters, delay_iters, 512, 8, 80, 1) == (1024, 2560, None)
    assert fit_window_to_run(256, 6016, 1024, 64, 640) == (256, 6016, None)
    assert fit_window_to_run(1024, 2560, 512, 8, "many") == (1024, 2560, None)
    assert fit_window_to_run(1024, 2560, 512, 0, 16) == (1024, 2560, None)


def test_fit_window_to_run_moves_window_inside_short_run():
    max_iters, delay_iters, warning = fit_window_to_run(1024, 2560, 512, 8, 16, 1)

    assert (max_iters, delay_iters) == (512, 256)
    assert "steps 2560..3584" in warning
    assert "about 1024 decode steps" in warning
    assert "using steps 256..768" in warning

    assert fit_window_to_run(1024, 2560, 512, 8, 8)[:2] == (256, 128)
    assert fit_window_to_run(1024, 1024, 512, 8, 16, 0.5)[:2] == (384, 192)


def test_fit_window_to_run_keeps_window_that_opens_before_run_ends():
    assert fit_window_to_run(1024, 2560, 512, 8, 48, 1) == (1024, 2560, None)
    assert fit_window_to_run(512, 0, 32, 1, 2, 0) == (512, 0, None)
    assert fit_window_to_run(512, 0, 32, 1, 10, 1) == (512, 0, None)


def _prepare(tmp_path, framework, num_prompts=None):
    inferencex = tmp_path / framework / "InferenceX"
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
    benchmark_serving = serving_dir / "benchmark_serving.py"
    benchmark_serving.write_text(
        'extra_body={"num_steps": 1, "merge_profiles": True, '
        '"profile_by_stage": True},\n',
        encoding="utf-8",
    )
    envs = {"CONC": 8, "OSL": 512, "RANDOM_RANGE_RATIO": 1}
    if num_prompts is not None:
        envs["NUM_PROMPTS"] = num_prompts
    cfg = BenchmarkConfig.from_dict(
        {
            "framework": framework,
            "model": "demo",
            "inferencex_path": str(inferencex),
            "envs": envs,
            "profiler": {"tracelens": {"enabled": True}},
        }
    )
    result = TraceLensInferencePipeline(cfg).prepare(tmp_path / "workspace")
    return result, cfg, benchmark_serving.read_text(encoding="utf-8")


def test_tracelens_inference_prepare_fits_sglang_window_to_short_run(tmp_path):
    result, _, patched_serving = _prepare(tmp_path, "sglang", 16)

    assert result["max_iterations"] == 512
    assert result["delay_iterations"] == 256
    assert '"start_step": 256' in patched_serving
    assert '"num_steps": 512' in patched_serving
    assert len(result["warnings"]) == 1
    assert "starts after this run" in result["warnings"][0]


def test_tracelens_inference_prepare_fits_vllm_window_to_short_run(tmp_path):
    result, cfg, _ = _prepare(tmp_path, "vllm", 16)

    extra_args = cfg.envs["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.delay_iterations 256" in extra_args
    assert "--profiler-config.max_iterations 512" in extra_args
    assert len(result["warnings"]) == 1


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
def test_tracelens_inference_prepare_keeps_window_for_default_num_prompts(
    tmp_path, framework
):
    result, cfg, _ = _prepare(tmp_path, framework)

    assert cfg.envs["NUM_PROMPTS"] == "80"
    assert result["max_iterations"] == 1024
    assert result["delay_iterations"] == 2560
    assert result["warnings"] == []
