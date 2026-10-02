"""Managed AgentX dispatch, persistence, and report gates at the public mode."""

import json
import os
import sys
from pathlib import Path

import pytest

from Magpie.modes.benchmark import benchmarker
from Magpie.modes.benchmark.agentx_launch import prepare_server_launch
from Magpie.modes.benchmark.benchmarker import BenchmarkMode
from Magpie.modes.benchmark.config import BenchmarkConfig
from Magpie.modes.benchmark.result import BenchmarkResult


def _config(tmp_path, **updates):
    return BenchmarkConfig.from_dict(
        {
            "model": "test/model",
            "framework": "sglang",
            "precision": "bf16",
            "agentx": True,
            "run_mode": "local",
            "runner_type": "mi355x",
            "inferencex_path": str(tmp_path),
            **updates,
        }
    )


def _spec():
    return {
        "argv": [sys.executable, "-m", "sglang.launch_server"],
        "client_env": {},
        "model_metadata": {"native_context_length": 8192},
    }


def test_extra_arguments_are_literal_and_consumed_once_across_config_reload(tmp_path):
    config = _config(
        tmp_path,
        envs={"EXTRA_SGLANG_ARGS": '--json-setting \'{"value":"a b"}\' --flag'},
    )
    config.agentx.resolved = {"server-launch-spec": _spec()}
    mode = BenchmarkMode(config, output_dir=str(tmp_path / "results"))
    mode._normalize_agentx_extra_args()
    expected = ["--json-setting", '{"value":"a b"}', "--flag"]
    assert config.agentx.launch_overrides["append_args"] == expected
    assert "EXTRA_SGLANG_ARGS" not in config.envs
    restored = BenchmarkConfig.from_dict(config.to_dict())
    restored.agentx.resolved = {"server-launch-spec": _spec()}
    BenchmarkMode(restored)._normalize_agentx_extra_args()
    assert restored.agentx.launch_overrides["append_args"] == expected


@pytest.mark.parametrize(
    "envs",
    [
        {"EXTRA_VLLM_ARGS": "--flag"},
        {"EXTRA_SGLANG_ARGS": "--flag", "extra_sglang_args": "--other"},
        {"EXTRA_SGLANG_ARGS": ["--flag"]},
        {"EXTRA_SGLANG_ARGS": "--port 9999"},
    ],
)
def test_extra_arguments_cannot_bypass_framework_or_protocol_checks(tmp_path, envs):
    config = _config(tmp_path, envs=envs)
    config.agentx.resolved = {"server-launch-spec": _spec()}
    with pytest.raises(ValueError):
        BenchmarkMode(config)._normalize_agentx_extra_args()


@pytest.mark.parametrize(
    "receipt,client_ok,gate_ok",
    [
        (True, True, True),
        (False, True, True),
        (True, False, True),
        (True, True, False),
    ],
)
def test_managed_dispatch_requires_client_gate_and_current_launch_evidence(
    monkeypatch, tmp_path, receipt, client_ok, gate_ok
):
    config = _config(tmp_path)
    mode = BenchmarkMode(config, output_dir=str(tmp_path / "results"))
    monkeypatch.setattr(benchmarker, "ensure_inferencex_available", lambda path: path)
    monkeypatch.setattr(benchmarker, "ensure_agentx_dependencies", lambda path: None)

    def resolve(config, *_args, **_kwargs):
        config.agentx.resolved = {"server-launch-spec": _spec()}
        config.benchmark_script = "srt_agentic.sh"

    monkeypatch.setattr(benchmarker, "resolve_agentx_recipe", resolve)
    selected = []
    monkeypatch.setattr(mode, "_apply_gpu_selection", lambda: selected.append(True))
    monkeypatch.setattr(mode, "_get_benchmark_script", lambda runner: "srt_agentic.sh")

    def unexpected(*_args, **_kwargs):
        pytest.fail("Managed AgentX must not call a legacy launcher or global cleanup")

    for name in (
        "_prepare_benchmark_scripts",
        "_execute_local_benchmark",
        "_cleanup_server_processes",
    ):
        monkeypatch.setattr(mode, name, unexpected)

    def execute(config, workspace, runner, **kwargs):
        assert runner == "mi355x"
        assert kwargs["docker_image"] is None
        (workspace / "inferencex_result.json").write_text("{}")
        if receipt:
            spec = config.agentx.resolved["server-launch-spec"]
            prepare_server_launch(
                spec["argv"],
                dict(os.environ),
                None,
                "sglang",
                workspace,
                server_spec=spec,
            )
        return (
            BenchmarkResult(success=client_ok),
            "client output",
            "client failure" if not client_ok else "",
        )

    monkeypatch.setattr(benchmarker, "execute_agentx", execute)

    def parse(*_args, **_kwargs):
        return BenchmarkResult(
            success=gate_ok,
            scenario="agentx",
            benchmark_valid=gate_ok,
            publishable=gate_ok,
            agentx_metrics={},
        )

    monkeypatch.setattr(benchmarker.ResultParser, "parse_inferencex_result", parse)
    monkeypatch.setattr(mode, "_validate_results", lambda result: True)
    result = mode.run("managed-test")
    assert selected == [True]
    assert result.success is (receipt and client_ok and gate_ok)
    assert result.benchmark_valid is (receipt and client_ok and gate_ok)
    assert result.publishable is (receipt and client_ok and gate_ok)
    saved = json.loads(
        (Path(result.workspace_dir) / "benchmark_report.json").read_text()
    )
    assert saved["success"] == result.success
    if receipt:
        assert result.agentx_metrics["execution_owner"] == "magpie"
        assert result.agentx_metrics["model_context"]["native_context_length"] == 8192
    else:
        assert any("launch evidence rejected" in error for error in result.errors)


def test_managed_lifecycle_requires_cleanup_and_disallows_reuse(tmp_path):
    config = _config(
        tmp_path,
        server_lifecycle={"enabled": True, "cleanup": True, "force_reuse": False},
    )
    assert config.server_lifecycle.cleanup
    for change in ({"cleanup": False}, {"force_reuse": True}):
        with pytest.raises(ValueError, match="AgentX cannot"):
            _config(
                tmp_path, server_lifecycle={"enabled": True, "cleanup": True, **change}
            )
