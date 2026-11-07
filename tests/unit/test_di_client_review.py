"""DI review regressions at client launch, interruption and script selection."""

import os
import signal
from types import SimpleNamespace

import pytest
import yaml

from Magpie.core.executor import EndpointClientCleanupError
from Magpie.di.native_ray import NativeRayDISession
from Magpie.modes.benchmark import BenchmarkConfig, BenchmarkMode
from Magpie.modes.benchmark import benchmarker, endpoint_client


def _config(tmp_path, di=True):
    raw = {
        "framework": "sglang", "model": "test", "run_mode": "ray",
        "runner_type": "mi355x", "inferencex_path": str(tmp_path),
    }
    if di:
        raw["disaggregation"] = {"prefill": {}, "decode": {}}
    return raw


def _process(monkeypatch):
    calls = []
    process = SimpleNamespace(
        pid=12345678, returncode=0,
        wait=lambda **kwargs: calls.append("wait"),
        communicate=lambda **kwargs: ("output", ""),
    )
    monkeypatch.setattr(endpoint_client.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(endpoint_client.os, "killpg", lambda *args: calls.append("stop"))
    monkeypatch.setattr(endpoint_client, "_group_exists", lambda pgid: False)
    return process, calls


def test_sigint_during_launch_is_delivered_after_process_ownership(monkeypatch):
    process, calls = _process(monkeypatch)

    def launch(*args, **kwargs):
        os.kill(os.getpid(), signal.SIGINT)
        return process

    monkeypatch.setattr(endpoint_client.subprocess, "Popen", launch)
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with pytest.raises(KeyboardInterrupt):
            endpoint_client.execute(["client"], {}, 1, lambda: None)
        assert calls == ["stop", "wait", "wait"]
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    finally:
        signal.signal(signal.SIGINT, previous)


def test_launch_interruption_with_unknown_process_retains_ownership(monkeypatch, tmp_path):
    session = NativeRayDISession(_config(tmp_path), tmp_path / "session")
    session.state["status"] = "ready"

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("interrupted before ownership returned")

    monkeypatch.setattr(endpoint_client.subprocess, "Popen", interrupted)
    with pytest.raises(EndpointClientCleanupError, match="launch ownership"):
        with session.client_run("evaluation", tmp_path):
            endpoint_client.execute(["client"], {}, 1, lambda: None)
    assert session.state["client_cleanup_pending"]["task_id"] == "evaluation"
    with pytest.raises(RuntimeError, match="client cleanup"):
        session.stop()


def test_cleanup_interruption_retains_client_marker(monkeypatch, tmp_path):
    process, _ = _process(monkeypatch)
    session = NativeRayDISession(_config(tmp_path), tmp_path / "session")
    session.state["status"] = "ready"

    def interrupted(**kwargs):
        raise KeyboardInterrupt("interrupted cleanup")

    process.wait = interrupted
    with pytest.raises(EndpointClientCleanupError, match="cleanup was not confirmed"):
        with session.client_run("evaluation", tmp_path):
            endpoint_client.execute(["client"], {}, 1, lambda: None)
    assert session.state["client_cleanup_pending"]


def test_sigint_during_cleanup_is_replayed_only_after_confirmed_absence(monkeypatch):
    process, calls = _process(monkeypatch)
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)

    def wait(**kwargs):
        if not calls.count("interrupt"):
            calls.append("interrupt")
            os.kill(os.getpid(), signal.SIGINT)
        calls.append("wait")

    process.wait = wait
    try:
        with pytest.raises(KeyboardInterrupt):
            endpoint_client.execute(["client"], {}, 1, lambda: None)
        assert calls == ["stop", "interrupt", "wait", "wait"]
    finally:
        signal.signal(signal.SIGINT, previous)


def test_di_default_uses_wrapper_while_non_di_keeps_native_priority(tmp_path):
    scripts = tmp_path / "benchmarks"
    scripts.mkdir()
    (scripts / "dsr1_fp8_mi355x.sh").write_text("launch ordinary engine")
    (scripts / "sglang_mi355x.sh").write_text("MAGPIE_RUN_PHASE BENCHMARK_BASE_URL")
    for di, expected in ((True, "sglang_mi355x.sh"), (False, "dsr1_fp8_mi355x.sh")):
        mode = BenchmarkMode(BenchmarkConfig.from_dict(_config(tmp_path, di)), output_dir=str(tmp_path / str(di)))
        assert mode._get_benchmark_script("mi355x") == "benchmarks/" + expected


def test_incompatible_explicit_script_rejected_before_serving(tmp_path, monkeypatch):
    scripts = tmp_path / "benchmarks"
    scripts.mkdir()
    (scripts / "custom.sh").write_text("launch ordinary engine")
    raw = _config(tmp_path)
    raw["benchmark_script"] = "custom.sh"
    mode = BenchmarkMode(BenchmarkConfig.from_dict(raw), output_dir=str(tmp_path / "output"))
    monkeypatch.setattr(benchmarker, "ensure_inferencex_available", lambda path: path)
    monkeypatch.setattr(mode, "_prepare_benchmark_scripts", lambda: None)
    monkeypatch.setattr(mode, "_get_runner_type", lambda: "mi355x")
    with pytest.raises(ValueError, match="must support"):
        mode._execute_native_di_benchmark()
    assert not list((tmp_path / "output").rglob("native-session.json"))


def test_documented_di_yaml_supplies_required_client_settings():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    document = yaml.safe_load((root / "examples/benchmarks/benchmark_sglang_dsr1_di_ray.yaml").read_text())
    env = BenchmarkConfig.from_dict(document["benchmark"]).get_env_vars()
    assert float(env["RANDOM_RANGE_RATIO"]) == 0.5
    assert all(env[key] for key in ("MODEL", "CONC", "ISL", "OSL"))
    assert "RANDOM_RANGE_RATIO: 0.5" in (root / "docs/how-to/di-session.md").read_text()


@pytest.mark.parametrize("failure", ["missing", "permission"])
def test_confirmed_exec_failure_clears_client_ownership(tmp_path, failure):
    target = tmp_path / "client"
    error_type = FileNotFoundError
    if failure == "permission":
        target.write_text("#!/bin/sh\nexit 0\n")
        target.chmod(0o600)
        error_type = PermissionError
    session = NativeRayDISession(_config(tmp_path), tmp_path / "session")
    session.state["status"] = "ready"
    with pytest.raises(error_type):
        with session.client_run("evaluation", tmp_path):
            endpoint_client.execute([str(target)], {}, 1, lambda: None)
    assert session.state["client_cleanup_pending"] is None
    session._require_client_cleanup()


def test_pending_sigint_does_not_replace_cleanup_error():
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            with endpoint_client._defer_sigint():
                os.kill(os.getpid(), signal.SIGINT)
                raise RuntimeError("cleanup failed")
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    finally:
        signal.signal(signal.SIGINT, previous)
