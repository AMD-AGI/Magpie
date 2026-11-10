"""DI admission, invocation isolation and accuracy-only client regressions."""

import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from Magpie.core.executor import EndpointClientCleanupError
from Magpie.modes.benchmark import BenchmarkConfig, BenchmarkMode, benchmarker
from Magpie.modes.benchmark.workspace import WorkspaceManager


def mode(tmp_path, **updates):
    raw = {
        "framework": "sglang",
        "model": "test",
        "precision": "bf16",
        "run_mode": "ray",
        "runner_type": "mi355x",
        "inferencex_path": str(tmp_path / "InferenceX"),
        "disaggregation": {"prefill": {}, "decode": {}},
        "envs": {"CONC": 1, "ISL": 32, "OSL": 8, "RANDOM_RANGE_RATIO": 0.5},
    }
    raw.update(updates)
    return BenchmarkMode(
        BenchmarkConfig.from_dict(raw), output_dir=str(tmp_path / "results")
    )


def test_di_invocations_have_distinct_task_ids_in_same_second(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmarker.time, "time", lambda: 1)
    modes = [mode(tmp_path), mode(tmp_path)]
    for item in modes:
        monkeypatch.setattr(item, "_execute_ray_benchmark", lambda: None)
        item.run()
    assert modes[0]._task_id != modes[1]._task_id
    modes[0].run(task_id="caller-id")
    assert modes[0]._task_id == "caller-id"


def test_di_workspace_creation_is_atomic_and_unique(tmp_path):
    def create(_):
        manager = WorkspaceManager(str(tmp_path), "sglang")
        directory = manager.create({"candidate": "same"}, unique=True)
        (directory / "di-session").mkdir()
        return directory

    with ThreadPoolExecutor(max_workers=8) as pool:
        directories = list(pool.map(create, range(16)))
    assert len(set(directories)) == 16
    assert all((path / "config.yaml").is_file() for path in directories)
    # The existing ordinary timestamp naming and reuse policy is untouched.
    ordinary = WorkspaceManager(str(tmp_path), "vllm").create()
    assert len(ordinary.name.split("_")) == 4


@pytest.mark.parametrize("missing", ["CONC", "ISL", "OSL", "RANDOM_RANGE_RATIO"])
def test_stock_di_client_environment_is_checked_before_session(
    monkeypatch, tmp_path, missing
):
    item = mode(tmp_path)
    del item.config.envs[missing]
    monkeypatch.setattr(benchmarker, "ensure_inferencex_available", lambda path: path)
    monkeypatch.setattr(item, "_prepare_benchmark_scripts", lambda: None)
    monkeypatch.setattr(
        item, "_get_benchmark_script", lambda _: "benchmarks/sglang_mi355x.sh"
    )
    monkeypatch.setattr(
        "Magpie.di.native_ray.NativeRayDISession",
        lambda *a, **k: pytest.fail("serving started"),
    )
    with pytest.raises(ValueError, match=missing):
        item._execute_native_di_benchmark()
    assert item.workspace_mgr.workspace_path is None


@pytest.mark.parametrize("flag", ["true", "on", "YES"])
def test_di_accuracy_only_requires_evaluator_before_session(
    monkeypatch, tmp_path, flag
):
    item = mode(tmp_path)
    item.config.envs["EVAL_ONLY"] = flag
    monkeypatch.setattr(benchmarker, "ensure_inferencex_available", lambda path: path)
    monkeypatch.setattr(item, "_prepare_benchmark_scripts", lambda: None)
    monkeypatch.setattr(
        item, "_get_benchmark_script", lambda _: "benchmarks/sglang_mi355x.sh"
    )
    with pytest.raises(ValueError, match="RUN_EVAL"):
        item._execute_native_di_benchmark()
    assert item.workspace_mgr.workspace_path is None


@pytest.mark.parametrize("primary", [True, False])
def test_stop_failure_keeps_original_di_cleanup_journal(monkeypatch, tmp_path, primary):
    item = mode(tmp_path)
    original = EndpointClientCleanupError("inspect /shared/session/native-session.json")
    secondary = EndpointClientCleanupError("executor cleanup unconfirmed")
    calls = []

    class Executor:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return True

        def execute(self, task):
            if primary:
                raise original
            return SimpleNamespace(
                status=SimpleNamespace(value="failed"),
                results=None,
                errors=["ordinary failure"],
                execution_time=0,
            )

        def stop(self):
            calls.append("stop")
            raise secondary

    monkeypatch.setattr("Magpie.core.ray_executor.RayJobExecutor", Executor)
    with pytest.raises(EndpointClientCleanupError) as error:
        item._execute_ray_benchmark()
    assert error.value is (original if primary else secondary)
    assert calls == ["stop"]


@pytest.mark.parametrize("runner", ["mi300x", "mi355x"])
@pytest.mark.parametrize("accuracy_only", [True, False])
def test_real_stock_wrapper_accuracy_only_skips_failing_throughput(
    tmp_path, runner, accuracy_only
):
    source = (
        Path(benchmarker.__file__).parents[2]
        / "scripts"
        / "benchmark"
        / f"sglang_{runner}.sh"
    )
    wrapper = tmp_path / source.name
    shutil.copyfile(source, wrapper)
    # Exercise the actual wrapper branches. A throughput invocation fails, so
    # accuracy-only succeeds only if it reaches the evaluator directly.
    (tmp_path / "benchmark_lib.sh").write_text("check_env_vars() { :; }\n")
    (tmp_path / "server_cleanup.sh").write_text("")
    (tmp_path / "magpie_bench_remote_compat.sh").write_text(
        'magpie_eval_truthy() { [[ "$1" == true ]]; }\n'
        "magpie_run_benchmark_serving_remote_direct() { echo throughput; return 23; }\n"
        "magpie_run_eval_remote_direct() { echo accuracy; }\n"
    )
    result = subprocess.run(
        ["bash", str(wrapper)],
        check=False,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "BENCHMARK_BASE_URL": "http://router",
            "MAGPIE_RUN_PHASE": "client",
            "EVAL_ONLY": "true" if accuracy_only else "false",
            "RUN_EVAL": "true",
        },
    )
    assert result.returncode == (0 if accuracy_only else 23), result.stderr
    assert ("accuracy\n" in result.stdout) is accuracy_only
    assert ("throughput\n" in result.stdout) is not accuracy_only


def test_di_accuracy_only_uses_existing_truthy_contract(tmp_path):
    item = mode(tmp_path)
    item.config.envs["EVAL_ONLY"] = "on"
    (tmp_path / "accuracy_report.json").write_text('{"status":"COMPLETED"}')
    assert item._eval_only_accuracy_ok(tmp_path)
    item.config.disaggregation = None
    assert not item._eval_only_accuracy_ok(tmp_path)
