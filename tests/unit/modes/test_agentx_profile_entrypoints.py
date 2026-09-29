"""Profiler diagnostics retain their identity through public Magpie entrypoints."""

import asyncio
import gzip
import json
import os
from pathlib import Path
import sys

import pytest
import yaml

from Magpie import main
from Magpie.mcp import server
from Magpie.modes.benchmark import benchmarker
from Magpie.modes.benchmark.agentx_launch import prepare_server_launch
from Magpie.modes.benchmark.agentx_profile_config import (
    profile_server_spec,
    profile_settings,
)
from Magpie.modes.benchmark.benchmarker import BenchmarkMode
from Magpie.modes.benchmark.config import BenchmarkConfig
from Magpie.modes.benchmark.result import BenchmarkResult, ResultParser


def _aggregate():
    return {
        "scenario_type": "agentic-coding",
        "recipe_fingerprint": "a" * 64,
        "num_requests_total": 5,
        "num_requests_successful": 5,
        "request_accounting": {
            "records_total": 5,
            "records_profiled": 5,
            "records_error_dropped": 0,
            "records_warmup_dropped": 0,
        },
        "request_metrics": {
            "qps": {"mean": 0.5},
            "throughput": {
                "input": {"tokens_per_second": 100},
                "output": {"tokens_per_second": 20},
                "total": {"tokens_per_second": 120},
                "duration_seconds": 3600,
            },
            "latency": {"ttft": {"mean": 0.5}},
        },
    }


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
@pytest.mark.parametrize("trace_suffix", [".json", ".json.gz"])
@pytest.mark.parametrize(
    ("capture_status", "client_ok"),
    [("complete", True), ("failed", True), (None, True), ("complete", False)],
)
def test_public_mode_keeps_capture_manifest_after_real_agentx_result_parsing(
    monkeypatch, tmp_path, framework, trace_suffix, capture_status, client_ok
):
    config = BenchmarkConfig.from_dict(
        {
            "framework": framework,
            "model": "test/model",
            "agentx": True,
            "run_mode": "local",
            "runner_type": "mi355x",
            "inferencex_path": str(tmp_path),
            "profiler": {"torch_profiler": {"enabled": True, "num_steps": 7}},
        }
    )
    mode = BenchmarkMode(config, output_dir=str(tmp_path / "results"))
    monkeypatch.setattr(benchmarker, "ensure_inferencex_available", lambda path: path)
    monkeypatch.setattr(benchmarker, "ensure_agentx_dependencies", lambda path: None)
    monkeypatch.setattr(mode, "_apply_gpu_selection", lambda: None)
    monkeypatch.setattr(mode, "_get_benchmark_script", lambda runner: "srt_agentic.sh")

    def resolve(config, *_args, **_kwargs):
        config.benchmark_script = "srt_agentic.sh"
        config.agentx.resolved = {
            "server-launch-spec": {
                "framework": framework,
                "argv": [sys.executable, "-m", f"{framework}.launch_server"],
                "env": {},
                "client_env": {},
            }
        }

    monkeypatch.setattr(benchmarker, "resolve_agentx_recipe", resolve)
    captures = []

    def execute(config, workspace, _runner, **_kwargs):
        raw_path = workspace / "inferencex_result.json"
        raw_path.write_text(json.dumps(_aggregate()))
        unprofiled = ResultParser.parse_inferencex_result(
            raw_path, scenario="agentx", agentx_mode="canonical"
        )
        assert unprofiled.benchmark_valid is True
        assert unprofiled.publishable is True

        spec = profile_server_spec(
            config.agentx.resolved["server-launch-spec"],
            profile_settings(config),
            workspace,
            "b" * 32,
        )
        prepare_server_launch(
            spec["argv"],
            {**os.environ, **spec["env"]},
            None,
            framework,
            workspace,
            server_spec=spec,
        )
        capture = {
            "version": 1,
            "capture_id": "b" * 32,
            "status": capture_status,
            "framework": framework,
            "num_steps": 7,
            "expected_ranks": 1,
            "phase_start_ns": 123456789,
            "trace_files": [],
        }
        if capture_status == "complete":
            trace = Path(spec["torch_profiler"]["trace_dir"]) / f"rank0{trace_suffix}"
            trace.parent.mkdir(parents=True)
            opener = gzip.open if trace_suffix.endswith(".gz") else open
            with opener(trace, "wt") as stream:
                json.dump(
                    {"traceEvents": [{"cat": "kernel", "name": "gemm", "dur": 100}]},
                    stream,
                )
            with gzip.open(
                workspace / "torch_trace" / "old-rank0.json.gz", "wt"
            ) as stream:
                json.dump(
                    {
                        "traceEvents": [
                            {"cat": "kernel", "name": "old", "dur": i}
                            for i in range(200)
                        ]
                    },
                    stream,
                )
            capture["trace_files"] = [{"path": str(trace), "rank": 0}]
        elif capture_status == "failed":
            capture["error"] = "TimeoutError: rank trace did not flush"
        captures.append(capture)
        return (
            BenchmarkResult(
                success=client_ok,
                agentx_metrics={"profile_capture": capture} if capture_status else {},
            ),
            "",
            "",
        )

    monkeypatch.setattr(benchmarker, "execute_agentx", execute)
    result = mode.run("profile-entrypoint")
    assert result.benchmark_valid is False
    assert result.publishable is False
    assert result.success is (capture_status == "complete" and client_ok)
    assert result.profiling_enabled is True
    assert result.agentx_metrics["diagnostic_only"] is True
    assert result.agentx_metrics["execution_owner"] == "magpie"
    expected_capture = captures[0] if capture_status else {"status": "failed"}
    assert result.agentx_metrics["profile_capture"] == expected_capture
    assert result.agentx_metrics["requests"]["successful"] == 5
    if capture_status != "complete":
        assert any("capture did not complete" in error for error in result.errors)
    if result.success:
        assert result.top_bottlenecks == ["gemm"]
    saved = json.loads(
        (Path(result.workspace_dir) / "benchmark_report.json").read_text()
    )
    assert saved["agentx_metrics"]["profile_capture"] == expected_capture
    assert saved["benchmark_valid"] is False
    assert saved["publishable"] is False
    assert saved["success"] == result.success


@pytest.fixture
def entrypoint_configs(monkeypatch, tmp_path):
    configs = []

    class Mode:
        def __init__(self, config, output_dir):
            configs.append(config)

        def run(self):
            return BenchmarkResult(success=True, workspace_dir=str(tmp_path))

        def cleanup(self):
            pass

    monkeypatch.setattr("Magpie.modes.benchmark.BenchmarkMode", Mode)
    return configs


@pytest.mark.parametrize(
    ("options", "enabled", "steps"),
    [
        ([], False, 20),
        (["--torch-profiler"], True, 20),
        (["--torch-profiler", "--torch-profiler-steps", "9"], True, 9),
    ],
)
def test_agentx_cli_profiling_is_explicit_and_forwards_step_count(
    entrypoint_configs, options, enabled, steps
):
    args = main.create_parser().parse_args(
        ["benchmark", "sglang", "--model", "test/model", "--agentx", *options]
    )
    assert main.run_benchmark(args, {}) == 0
    settings = entrypoint_configs[0].profiler.torch_profiler
    assert settings.enabled is enabled
    assert settings.num_steps == steps


@pytest.mark.parametrize("steps", ["0", "-1"])
def test_agentx_cli_rejects_invalid_step_count_before_running(
    entrypoint_configs, steps
):
    args = main.create_parser().parse_args(
        [
            "benchmark",
            "sglang",
            "--model",
            "test/model",
            "--agentx",
            "--torch-profiler",
            "--torch-profiler-steps",
            steps,
        ]
    )
    assert main.run_benchmark(args, {}) == 1
    assert entrypoint_configs == []


def test_agentx_cli_preserves_yaml_capture_timeouts(entrypoint_configs, tmp_path):
    path = tmp_path / "profile.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "sglang",
                    "model": "test/model",
                    "agentx": True,
                    "profiler": {
                        "torch_profiler": {
                            "enabled": True,
                            "num_steps": 11,
                            "capture_timeout_seconds": 41.5,
                            "flush_timeout_seconds": 123,
                        }
                    },
                }
            }
        )
    )
    args = main.create_parser().parse_args(
        ["benchmark", "--benchmark-config", str(path)]
    )
    assert main.run_benchmark(args, {}) == 0
    assert entrypoint_configs[0].profiler.torch_profiler.to_dict() == {
        "enabled": True,
        "num_steps": 11,
        "capture_timeout_seconds": 41.5,
        "flush_timeout_seconds": 123.0,
    }


def test_ordinary_cli_default_stays_unprofiled(entrypoint_configs):
    args = main.create_parser().parse_args(
        ["benchmark", "sglang", "--model", "test/model"]
    )
    assert main.run_benchmark(args, {}) == 0
    assert entrypoint_configs[0].profiler.torch_profiler.enabled is False


@pytest.mark.parametrize(
    ("options", "enabled", "steps", "trace_mode"),
    [
        ({"agentx": True}, False, 20, "pytorch"),
        (
            {
                "agentx": True,
                "torch_profiler": True,
                "torch_profiler_steps": 9,
                "tracelens": True,
            },
            True,
            9,
            "pytorch",
        ),
        ({}, True, 20, "inference"),
        ({"torch_profiler": False}, False, 20, "inference"),
    ],
)
def test_mcp_profiler_defaults_and_explicit_diagnostics(
    entrypoint_configs, options, enabled, steps, trace_mode
):
    response = json.loads(
        asyncio.run(server.benchmark("sglang", "test/model", **options))
    )
    assert response["success"] is True
    config = entrypoint_configs[0]
    assert config.profiler.torch_profiler.enabled is enabled
    assert config.profiler.torch_profiler.num_steps == steps
    assert config.profiler.tracelens.analysis_mode == trace_mode


@pytest.mark.parametrize("steps", [True, 0, -1, 1.5])
def test_mcp_rejects_invalid_step_count_before_running(entrypoint_configs, steps):
    response = json.loads(
        asyncio.run(
            server.benchmark(
                "sglang",
                "test/model",
                agentx=True,
                torch_profiler=True,
                torch_profiler_steps=steps,
            )
        )
    )
    assert "positive integer" in response["error"]
    assert entrypoint_configs == []


def test_mcp_tracelens_cannot_implicitly_enable_agentx_capture(entrypoint_configs):
    response = json.loads(
        asyncio.run(
            server.benchmark(
                "sglang",
                "test/model",
                agentx=True,
                tracelens=True,
            )
        )
    )
    assert "torch_profiler.enabled=true" in response["error"]
    assert entrypoint_configs == []
