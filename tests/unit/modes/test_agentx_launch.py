"""The native launch extension must be explicit and backed by run evidence."""

import json

import pytest
from test_agentx import _fake_inferencex, _minimal_config

from Magpie.modes.benchmark import benchmarker
from Magpie.modes.benchmark.agentx import resolve_agentx_recipe
from Magpie.modes.benchmark.agentx_launch import (
    digest,
    launch_environment,
    read_launch_evidence,
    validate_overrides,
)
from Magpie.modes.benchmark.benchmarker import BenchmarkMode
from Magpie.modes.benchmark.config import AgentXConfig, BenchmarkConfig
from Magpie.modes.benchmark.result import BenchmarkResult

RECIPE = "dsv4-fp4-mi355x-sglang-agentic-mtp"
SCRIPT = "single_node/agentic/dsv4_fp4_mi355x_sglang_mtp.sh"


def _declare_launcher(root, **changes):
    row = {
        "framework": "sglang",
        "runner_type": "mi355x",
        "model": "deepseek-ai/DeepSeek-V4-Pro-0813",
        "precision": "fp4",
        "benchmark_script": SCRIPT,
        "launch_overrides_version": 1,
        **changes,
    }
    path = root / "benchmarks" / SCRIPT
    path.parent.mkdir(parents=True)
    path.write_text("#!/bin/bash\n")
    (root / "configs/agentx-launchers.json").write_text(
        json.dumps({"version": 1, "recipes": {RECIPE: row}})
    )


def _config(**overrides):
    return _minimal_config(agentx={"launch_overrides": {"version": 1, **overrides}})


def _evidence(config, workspace):
    overrides = config.agentx.launch_overrides
    names = set(overrides["env"]) | set(overrides["unset_env"])
    evidence = {
        "version": 1,
        "framework": "sglang",
        "base_argv": ["python3", "-m", "sglang.launch_server"],
        "effective_argv": [
            overrides["executable"] or "python3",
            "-m",
            "sglang.launch_server",
        ],
        "base_env": {name: "original" for name in names},
        "effective_env": {name: overrides["env"].get(name) for name in names},
        "source_files": overrides["source_files"],
        "absent_source_files": overrides["absent_source_files"],
        "overrides_sha256": digest(overrides),
        "resolved_executable": "/usr/bin/python3",
        "runtime_environment": {"PATH": "/usr/bin"},
    }
    _write_evidence(workspace, evidence)
    return evidence


def _write_evidence(workspace, evidence):
    evidence["evidence_sha256"] = digest(
        {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    )
    (workspace / "agentx_server_launch.json").write_text(json.dumps(evidence))


def test_launcher_and_image_resolve_from_declared_recipe_without_user_pins(tmp_path):
    root = _fake_inferencex(tmp_path)
    _declare_launcher(root)
    config = _minimal_config(
        benchmark_script=None,
        docker_image=None,
        agentx={"launch_overrides": {"version": 1}},
    )
    spec = resolve_agentx_recipe(config, str(root), runner_type="mi355x")
    assert config.benchmark_script == SCRIPT
    assert config.docker_image == "sglang:test"
    assert spec.entry["benchmark-script"] == SCRIPT
    assert spec.entry["launch-overrides-version"] == 1
    assert config.to_dict()["benchmark_script"] == SCRIPT


def test_old_checkout_requires_explicit_launcher_and_cannot_ignore_overrides(tmp_path):
    root = _fake_inferencex(tmp_path)
    with pytest.raises(ValueError, match="no declared AgentX launcher"):
        resolve_agentx_recipe(
            _minimal_config(benchmark_script=None), str(root), runner_type="mi355x"
        )
    with pytest.raises(ValueError, match="does not declare launch_overrides"):
        resolve_agentx_recipe(_config(), str(root), runner_type="mi355x")
    resolve_agentx_recipe(_minimal_config(), str(root), runner_type="mi355x")


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"framework": "vllm"}, "framework mismatch"),
        ({"model": "wrong/model"}, "model mismatch"),
        ({"precision": "bf16"}, "precision mismatch"),
        ({"runner_type": "mi300x"}, "runner mismatch"),
        ({"benchmark_script": "../../escape.sh"}, "Invalid AgentX launcher path"),
        ({"launch_overrides_version": None}, "does not declare"),
    ],
)
def test_launcher_manifest_cannot_silently_change_identity(tmp_path, changes, match):
    root = _fake_inferencex(tmp_path)
    _declare_launcher(root, **changes)
    with pytest.raises(ValueError, match=match):
        resolve_agentx_recipe(_config(), str(root), runner_type="mi355x")


def test_explicit_launcher_cannot_disagree_with_declared_recipe(tmp_path):
    root = _fake_inferencex(tmp_path)
    _declare_launcher(root)
    with pytest.raises(ValueError, match="declares launcher"):
        resolve_agentx_recipe(
            _minimal_config(benchmark_script="wrong.sh"),
            str(root),
            runner_type="mi355x",
        )


def test_config_roundtrip_and_saved_request_have_identical_normalized_hash(tmp_path):
    config = _config(
        append_args=["--json", '{"x": "a b"}'], env={"SGLANG_USE_AITER": "0"}
    )
    serialized = config.to_dict()
    restored = BenchmarkConfig.from_dict(serialized)
    assert restored.agentx.launch_overrides == config.agentx.launch_overrides
    environment = launch_environment(config, tmp_path)
    request = json.loads((tmp_path / "agentx_launch_overrides.json").read_text())
    assert request == serialized["agentx"]["launch_overrides"]
    assert environment["AGENTX_LAUNCH_OVERRIDES_SHA256"] == digest(request)
    assert request["replace_args"] is False
    assert request["executable"] is None
    assert request["absent_source_files"] == []


def test_docker_paths_refer_to_mounted_workspace_and_remove_stale_receipt(tmp_path):
    (tmp_path / "agentx_server_launch.json").write_text("stale")
    env = launch_environment(_config(), tmp_path, docker=True)
    assert (
        env["AGENTX_LAUNCH_OVERRIDES_FILE"] == "/workspace/agentx_launch_overrides.json"
    )
    assert env["AGENTX_SERVER_LAUNCH_FILE"] == "/workspace/agentx_server_launch.json"
    assert not (tmp_path / "agentx_server_launch.json").exists()


def test_no_extension_clears_ambient_protocol_paths_without_creating_request(tmp_path):
    env = launch_environment(_minimal_config(), tmp_path)
    assert set(env.values()) == {""}
    assert not list(tmp_path.iterdir())


def test_valid_evidence_matches_exact_request_environment_and_sources(tmp_path):
    config = _config(
        env={"SGLANG_USE_AITER": "0"},
        unset_env=["OMP_NUM_THREADS"],
        source_files={"/candidate/kernel.py": "a" * 64},
        absent_source_files=["/candidate/deleted.py"],
    )
    launch_environment(config, tmp_path)
    evidence = _evidence(config, tmp_path)
    assert read_launch_evidence(config, tmp_path) == evidence


@pytest.mark.parametrize(
    "field,value",
    [
        ("overrides_sha256", "wrong"),
        ("source_files", {}),
        ("absent_source_files", []),
        ("effective_env", {}),
        ("base_env", {}),
        ("framework", "vllm"),
        ("version", 2),
        ("effective_argv", []),
        ("base_argv", "not-an-array"),
    ],
)
def test_rehashed_but_mismatched_receipt_is_rejected(tmp_path, field, value):
    config = _config(
        env={"SGLANG_USE_AITER": "0"},
        source_files={"/x.py": "a" * 64},
        absent_source_files=["/gone.py"],
    )
    launch_environment(config, tmp_path)
    evidence = _evidence(config, tmp_path)
    evidence[field] = value
    _write_evidence(tmp_path, evidence)
    with pytest.raises(ValueError):
        read_launch_evidence(config, tmp_path)


def test_missing_or_tampered_launch_receipt_cannot_pass(tmp_path):
    config = _config()
    launch_environment(config, tmp_path)
    with pytest.raises(FileNotFoundError):
        read_launch_evidence(config, tmp_path)
    evidence = _evidence(config, tmp_path)
    evidence["evidence_sha256"] = "bad"
    (tmp_path / "agentx_server_launch.json").write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="hash mismatch"):
        read_launch_evidence(config, tmp_path)


@pytest.mark.parametrize(
    "value", [{}, {"version": True}, {"version": 2}, {"version": 1, "unknown": True}]
)
def test_invalid_extension_versions_and_fields_rejected(value):
    with pytest.raises(ValueError):
        AgentXConfig(launch_overrides=value)


def test_source_and_argument_validation_is_applied_at_config_boundary():
    with pytest.raises(ValueError, match="absolute paths"):
        _config(source_files={"relative": "a" * 64})
    with pytest.raises(ValueError, match="protocol option"):
        _config(remove_args=["--tp"])
    assert validate_overrides({"version": 1})["remove_args"] == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"append_args": "--x"},
        {"remove_args": ["--x", "--x"]},
        {"remove_args": ["-x"]},
        {"env": []},
        {"env": {"A": 2}},
        {"env": {"A-B": "x"}},
        {"env": {"PORT": "1"}},
        {"env": {"A": "x"}, "unset_env": ["A"]},
        {"replace_args": 1},
        {"executable": "python"},
        {"absent_source_files": ["relative.py"]},
    ],
)
def test_malformed_requests_are_rejected_before_benchmark(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)


def test_changed_config_or_persisted_request_cannot_rebind_evidence(tmp_path):
    config = _config()
    config.agentx.launch_overrides.pop("executable")
    with pytest.raises(ValueError, match="normalization"):
        launch_environment(config, tmp_path)
    config = _config()
    launch_environment(config, tmp_path)
    (tmp_path / "agentx_launch_overrides.json").write_text("{}")
    with pytest.raises(ValueError, match="differs"):
        read_launch_evidence(config, tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("resolved_executable", None),
        ("runtime_environment", None),
        ("runtime_environment", {"PATH": 123}),
        ("effective_argv", ["wrong-executable"]),
    ],
)
def test_incomplete_runtime_and_executable_evidence_rejected(tmp_path, field, value):
    config = _config(executable="/candidate/bin/python")
    launch_environment(config, tmp_path)
    evidence = _evidence(config, tmp_path)
    evidence[field] = value
    _write_evidence(tmp_path, evidence)
    with pytest.raises(ValueError):
        read_launch_evidence(config, tmp_path)


@pytest.mark.parametrize(
    "manifest", ["not json", "[]", '{"version": 2, "recipes": {}}']
)
def test_invalid_launcher_manifest_is_never_silently_ignored(tmp_path, manifest):
    root = _fake_inferencex(tmp_path)
    (root / "configs/agentx-launchers.json").write_text(manifest)
    with pytest.raises(ValueError):
        resolve_agentx_recipe(_minimal_config(), str(root), runner_type="mi355x")


def test_declared_launcher_must_exist(tmp_path):
    root = _fake_inferencex(tmp_path)
    _declare_launcher(root)
    (root / "benchmarks" / SCRIPT).unlink()
    with pytest.raises(ValueError, match="missing or escapes"):
        resolve_agentx_recipe(_config(), str(root), runner_type="mi355x")


@pytest.mark.parametrize("receipt_mode", ["valid", "missing", "wrong-request"])
def test_benchmark_result_requires_this_runs_launch_evidence(
    monkeypatch, tmp_path, receipt_mode
):
    root = _fake_inferencex(tmp_path)
    _declare_launcher(root)
    config = _minimal_config(
        inferencex_path=str(root),
        run_mode="local",
        runner_type="mi355x",
        agentx={"launch_overrides": {"version": 1}},
    )
    mode = BenchmarkMode(config, output_dir=str(tmp_path / "results"))
    monkeypatch.setattr(benchmarker, "ensure_inferencex_available", lambda path: path)
    monkeypatch.setattr(benchmarker, "ensure_agentx_dependencies", lambda path: None)
    monkeypatch.setattr(mode, "_apply_gpu_selection", lambda: None)
    monkeypatch.setattr(mode, "_prepare_benchmark_scripts", lambda: None)
    monkeypatch.setattr(mode, "_cleanup_server_processes", lambda framework: None)

    def execute(_cmd, env, workspace):
        assert env["AGENTX_LAUNCH_OVERRIDES_FILE"] == str(
            workspace / "agentx_launch_overrides.json"
        )
        aggregate = {
            "scenario_type": "agentic-coding",
            "recipe_fingerprint": "a" * 64,
            "num_requests_total": 10,
            "num_requests_successful": 10,
            "request_accounting": {
                "records_total": 10,
                "records_profiled": 10,
                "records_error_dropped": 0,
            },
            "request_metrics": {
                "qps": {"mean": 1},
                "throughput": {
                    "input": {"tokens_per_second": 100},
                    "output": {"tokens_per_second": 20},
                    "total": {"tokens_per_second": 120},
                    "duration_seconds": 3600,
                },
                "latency": {"ttft": {"mean": 1}, "tpot": {"mean": 0.01}},
            },
        }
        (workspace / "inferencex_result.json").write_text(json.dumps(aggregate))
        if receipt_mode != "missing":
            evidence = _evidence(config, workspace)
            if receipt_mode == "wrong-request":
                evidence["overrides_sha256"] = "other-run"
                _write_evidence(workspace, evidence)
        return BenchmarkResult(success=True), "", ""

    monkeypatch.setattr(mode, "_execute_local_benchmark", execute)
    result = mode.run("launch-extension-test")
    assert result.success is (receipt_mode == "valid")
    if receipt_mode == "valid":
        assert result.agentx_metrics["server_launch"]["overrides_sha256"] == digest(
            config.agentx.launch_overrides
        )
    else:
        assert result.benchmark_valid is False
        assert result.publishable is False
        assert any("launch evidence rejected" in error for error in result.errors)
