"""DI clients preserve outcomes, artifacts and caller-owned lifetimes."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from Magpie.modes.benchmark import BenchmarkConfig, BenchmarkMode
from Magpie.modes.benchmark.config import RayConfig
from Magpie.modes.benchmark.endpoint_client import execute
from Magpie.modes.benchmark.result import BenchmarkResult


def client(tmp_path):
    config = BenchmarkConfig.from_dict(
        {
            "framework": "sglang",
            "model": "test",
            "run_mode": "ray",
            "disaggregation": {"prefill": {}, "decode": {}},
            "envs": {
                "TP": 8,
            },
            "ray_config": {"install_magpie": False},
        }
    )
    return BenchmarkMode(config, output_dir=str(tmp_path / "invocation"))


@pytest.mark.parametrize("success", [True, False])
def test_ray_client_materializes_artifacts_without_changing_outcome(tmp_path, success):
    remote = tmp_path / "remote"
    remote.mkdir()
    report = {
        "success": success,
        "workspace_dir": str(remote),
        "errors": [] if success else ["accuracy backend failed"],
        "metadata": {"candidate": "sha256:exact"},
        "quality_gate": {"passed": success, "checks": {"accuracy": success}},
    }
    (remote / "benchmark_report.json").write_text(json.dumps(report))
    (remote / "accuracy_report.json").write_text('{"passed":false}')
    (remote / "raw-sample.json").write_text('{"requests":16}')
    mode = client(tmp_path)
    mode._task_id = "point-1"
    result = BenchmarkResult()
    mode._populate_result_from_ray(result, report, RayConfig())
    assert result.success is success
    assert result.errors == report["errors"]
    assert result.quality_gate == report["quality_gate"]
    assert Path(result.workspace_dir).is_relative_to(tmp_path / "invocation")
    assert (
        Path(result.workspace_dir) / "accuracy_report.json"
    ).read_text() == '{"passed":false}'
    assert result.metadata["candidate"] == "sha256:exact"
    assert (
        json.loads((Path(result.workspace_dir) / "benchmark_report.json").read_text())[
            "success"
        ]
        is success
    )
    assert json.loads((Path(result.workspace_dir) / "benchmark_report.json").read_text())["quality_gate"] == report["quality_gate"]


def test_remote_path_without_shared_artifacts_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="not accessible"):
        client(tmp_path)._populate_result_from_ray(
            BenchmarkResult(),
            {"success": True, "workspace_dir": str(tmp_path / "missing")},
            RayConfig(),
        )


def test_whole_di_task_skips_gpu_visibility_and_tp_mutation(monkeypatch):
    from Magpie.remote import tasks

    def forbidden(*_):
        pytest.fail("endpoint client attempted to configure serving GPUs")

    monkeypatch.setattr(tasks, "_clear_hidden_gpus", forbidden)
    monkeypatch.setattr(tasks, "_configure_tp_isolation", forbidden)
    monkeypatch.setitem(tasks._MODE_RUNNERS, "benchmark", lambda *_: {"success": False})
    result = tasks.run_task(
        {
            "mode_type": "benchmark",
            "mode_config": {
                "benchmark_config": {
                    "disaggregation": {"prefill": {}, "decode": {}},
                    "envs": {
                        "BENCHMARK_BASE_URL": "http://router",
                        "MAGPIE_DI_CLIENT_HOST": "router",
                        "TP": 16,
                    },
                }
            },
        }
    )
    assert result["success"] is False


def test_client_timeout_stops_owned_process_group(tmp_path):
    marker = tmp_path / "ticks"
    program = (
        "import pathlib,time; p=pathlib.Path(" + repr(str(marker)) + "); "
        "exec('while True:\\n p.write_text(str(time.time()))\\n time.sleep(.03)')"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        execute([sys.executable, "-c", program], os.environ.copy(), 0.4, lambda: None)
    before = marker.read_text()
    time.sleep(0.15)
    assert marker.read_text() == before


def test_ray_executor_reports_typed_cleanup_failure_without_clearing_ownership(
    monkeypatch,
):
    from Magpie.core.executor import EndpointClientCleanupError, ExecutorConfig
    from Magpie.core.ray_executor import RayJobExecutor

    executor = RayJobExecutor(ExecutorConfig(), RayConfig())
    executor._di_tasks.add("client")
    reference = object()
    executor._obj_refs["client"] = reference
    monkeypatch.setattr(executor, "cancel_task", lambda task_id: False)
    with pytest.raises(EndpointClientCleanupError):
        executor.stop()
    assert executor._obj_refs["client"] is reference
    assert "client" in executor._di_tasks


def test_process_cleanup_failure_is_typed(monkeypatch):
    from Magpie.core.executor import EndpointClientCleanupError
    from Magpie.modes.benchmark import endpoint_client

    def denied(*_):
        raise PermissionError("cannot establish process-group absence")

    monkeypatch.setattr(endpoint_client.os, "killpg", denied)
    with pytest.raises(EndpointClientCleanupError, match="cleanup was not confirmed"):
        execute([sys.executable, "-c", "pass"], os.environ.copy(), 3, lambda: None)


@pytest.mark.parametrize("timeout", [None, 0, -1, float("nan"), float("inf")])
def test_invalid_client_timeout_cannot_start_an_unowned_process(monkeypatch, timeout):
    from Magpie.modes.benchmark import endpoint_client

    monkeypatch.setattr(
        endpoint_client.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("started a client"),
    )
    with pytest.raises((TypeError, ValueError)):
        execute([sys.executable, "-c", "pass"], {}, timeout, lambda: None)


@pytest.mark.parametrize(
    "outcome,confirmed",
    [
        ({"success": True}, True),
        ({"success": False, "error": "ordinary failed gate"}, True),
        ({"di_cleanup_unconfirmed": True}, False),
        ("cancelled", True),
        ("worker-lost", False),
    ],
)
def test_cancel_reads_remote_cleanup_outcome_before_discarding_ownership(
    monkeypatch, outcome, confirmed
):
    from types import SimpleNamespace

    from Magpie.core.executor import ExecutorConfig
    from Magpie.core.ray_executor import RayJobExecutor

    class Cancelled(Exception):
        pass

    reference = object()

    def get(_):
        if outcome == "cancelled":
            raise Cancelled
        if outcome == "worker-lost":
            raise RuntimeError(
                "Ray worker disappeared before process cleanup confirmation"
            )
        return outcome

    ray = SimpleNamespace(
        cancel=lambda *_args, **_kwargs: None,
        wait=lambda *_args, **_kwargs: ([reference], []),
        get=get,
        exceptions=SimpleNamespace(TaskCancelledError=Cancelled),
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    executor = RayJobExecutor(ExecutorConfig(), RayConfig())
    executor._di_tasks.add("client")
    executor._obj_refs["client"] = reference
    assert executor.cancel_task("client") is confirmed
    assert ("client" not in executor._di_tasks) is confirmed
    assert executor._obj_refs["client"] is reference


def test_timeout_does_not_swallow_remote_cleanup_uncertainty(monkeypatch):
    from types import SimpleNamespace

    from Magpie.core.executor import EndpointClientCleanupError, ExecutorConfig
    from Magpie.core.ray_executor import RayJobExecutor

    class Cancelled(Exception):
        pass

    reference = object()
    ray = SimpleNamespace(
        cancel=lambda *_args, **_kwargs: None,
        wait=lambda *_args, **kwargs: (
            [] if kwargs.get("timeout") == 0 else [reference],
            [],
        ),
        get=lambda _: {
            "di_cleanup_unconfirmed": True,
            "error": "group still active",
        },
        exceptions=SimpleNamespace(TaskCancelledError=Cancelled),
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    executor = RayJobExecutor(ExecutorConfig(), RayConfig())
    executor._di_tasks.add("client")
    executor._obj_refs["client"] = reference
    with pytest.raises(EndpointClientCleanupError, match="cleanup was not confirmed"):
        executor.wait_for_task("client", timeout=0)
    assert "client" in executor._di_tasks
    with pytest.raises(EndpointClientCleanupError):
        executor.stop()


def test_ready_result_does_not_swallow_remote_cleanup_uncertainty(monkeypatch):
    from types import SimpleNamespace

    from Magpie.core.executor import EndpointClientCleanupError, ExecutorConfig
    from Magpie.core.ray_executor import RayJobExecutor

    class Cancelled(Exception):
        pass

    reference = object()
    ray = SimpleNamespace(
        wait=lambda *_args, **_kwargs: ([reference], []),
        get=lambda _: {
            "di_cleanup_unconfirmed": True,
            "error": "group still active",
        },
        exceptions=SimpleNamespace(TaskCancelledError=Cancelled),
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    executor = RayJobExecutor(ExecutorConfig(), RayConfig())
    executor._di_tasks.add("client")
    executor._obj_refs["client"] = reference
    with pytest.raises(EndpointClientCleanupError, match="group still active"):
        executor.wait_for_task("client", timeout=1)
    assert "client" in executor._di_tasks


def test_di_client_receives_hf_auth_without_saving_it_in_task_config(monkeypatch):
    from copy import deepcopy
    from types import ModuleType, SimpleNamespace

    from Magpie.core.executor import ExecutorConfig
    from Magpie.core.ray_executor import RayJobExecutor

    captured = {}
    ray = ModuleType("ray")
    ray.nodes = lambda: [
        {
            "Alive": True,
            "NodeID": "allocated-a",
            "NodeManagerAddress": "127.0.0.1",
            "Resources": {"GPU": 8},
        }
    ]

    def remote(**options):
        captured["options"] = options

        def bind(_function):
            return SimpleNamespace(
                remote=lambda payload: captured.update(payload=payload)
            )

        return bind

    ray.remote = remote
    strategies = ModuleType("ray.util.scheduling_strategies")
    strategies.NodeAffinitySchedulingStrategy = lambda **kwargs: SimpleNamespace(
        **kwargs
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", strategies)
    monkeypatch.setenv("HF_TOKEN", "dummy-private-model-token")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-be-forwarded")
    benchmark_config = {"envs": {}, "disaggregation": {"prefill": {}, "decode": {}}}
    original = deepcopy(benchmark_config)
    task = SimpleNamespace(
        task_id="client-auth",
        mode_config=SimpleNamespace(
            mode_type=SimpleNamespace(value="benchmark"),
            benchmark_config=benchmark_config,
        ),
        to_dict=lambda: {"mode_config": {"benchmark_config": benchmark_config}},
    )
    executor = RayJobExecutor(ExecutorConfig(), RayConfig(install_magpie=False))
    executor._submit_ray_task(task)
    env = captured["options"]["runtime_env"]["env_vars"]
    assert env["HF_TOKEN"] == "dummy-private-model-token"
    assert "UNRELATED_SECRET" not in env
    assert "dummy-private-model-token" not in json.dumps(captured["payload"])
    assert benchmark_config == original


def _endpoint_submission_probe(monkeypatch, *, valid_host=True, fail_delivery=True):
    """Exercise the real submission boundary without starting Ray or a client."""
    from types import SimpleNamespace

    from Magpie.core.executor import ExecutorConfig
    from Magpie.core.ray_executor import RayJobExecutor

    events = []
    reference = object()

    def deliver(payload):
        events.append(("delivery", payload["task_id"]))
        if fail_delivery:
            raise RuntimeError(
                "connection lost after endpoint request may have been accepted"
            )
        return reference

    ray = SimpleNamespace(
        nodes=lambda: (
            [
                {
                    "Alive": True,
                    "NodeID": "allocated-a",
                    "NodeManagerAddress": "192.0.2.1",
                    "Resources": {"CPU": 2, "GPU": 8},
                }
            ]
            if valid_host
            else []
        ),
        remote=lambda **kwargs: lambda function: SimpleNamespace(remote=deliver),
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.util", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "ray.util.scheduling_strategies",
        SimpleNamespace(
            NodeAffinitySchedulingStrategy=lambda **kwargs: SimpleNamespace(**kwargs)
        ),
    )
    benchmark = {"envs": {}, "disaggregation": {"prefill": {}, "decode": {}}}
    task = SimpleNamespace(
        task_id="endpoint-delivery",
        status=None,
        mode_config=SimpleNamespace(
            benchmark_config=benchmark, mode_type=SimpleNamespace(value="benchmark")
        ),
        to_dict=lambda: {"mode_config": {"benchmark_config": benchmark}},
    )
    executor = RayJobExecutor(
        ExecutorConfig(), RayConfig(install_magpie=False, entrypoint_num_cpus=1)
    )
    executor._is_running = True
    monkeypatch.setattr(executor, "_build_runtime_env", lambda *_args: {"env_vars": {}})
    return executor, task, events, reference


def test_ambiguous_endpoint_submission_retains_ownership_without_object_ref(
    monkeypatch,
):
    from Magpie.core.executor import EndpointClientCleanupError

    executor, task, events, _ = _endpoint_submission_probe(monkeypatch)
    with pytest.raises(EndpointClientCleanupError, match="submission.*unconfirmed"):
        executor.submit(task)
    assert events == [("delivery", task.task_id)]
    assert task.task_id in executor._di_tasks
    assert task.task_id not in executor._obj_refs
    with pytest.raises(EndpointClientCleanupError, match="reference"):
        executor.stop()
    assert task.task_id in executor._di_tasks
    assert executor._is_running


def test_endpoint_local_validation_failure_does_not_create_delivery_uncertainty(
    monkeypatch,
):
    executor, task, events, _ = _endpoint_submission_probe(
        monkeypatch, valid_host=False
    )
    with pytest.raises(RuntimeError, match="No GPU node"):
        executor.submit(task)
    assert events == []
    assert executor._di_tasks == set()
    executor.stop()
    assert not executor._is_running


def test_successful_endpoint_submission_retains_returned_object_ref(monkeypatch):
    executor, task, events, reference = _endpoint_submission_probe(
        monkeypatch, fail_delivery=False
    )
    assert executor.submit(task) == task.task_id
    assert events == [("delivery", task.task_id)]
    assert executor._obj_refs[task.task_id] is reference
    assert task.task_id in executor._di_tasks


def test_non_di_submission_keeps_existing_error_behavior(monkeypatch):
    from Magpie.core.executor import EndpointClientCleanupError

    executor, task, events, _ = _endpoint_submission_probe(monkeypatch)
    task.mode_config.benchmark_config.pop("disaggregation")
    monkeypatch.setattr(executor, "_find_gpu_node", lambda: "allocated-a")
    with pytest.raises(RuntimeError, match="connection lost") as caught:
        executor.submit(task)
    assert not isinstance(caught.value, EndpointClientCleanupError)
    assert events == [("delivery", task.task_id)]
    assert executor._di_tasks == set()
    executor.stop()


def test_di_cleanup_does_not_stop_host_docker(monkeypatch, tmp_path):
    mode = client(tmp_path)
    mode._task_id = "native-di-client"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("Ray cleanup invoked host Docker"),
    )
    mode.cleanup()


def test_endpoint_timeout_retains_client_diagnostics(tmp_path):
    mode = client(tmp_path)
    mode.config.run_mode = "local"
    script = "import sys,time; print('client progress',flush=True); print('client diagnostic',file=sys.stderr,flush=True); time.sleep(30)"
    result, stdout, stderr = mode._execute_local_benchmark(
        [sys.executable, "-c", script],
        os.environ.copy(),
        tmp_path,
        timeout_seconds=0.4,
        managed_client=True,
    )
    assert result.success is False
    assert "client progress" in stdout
    assert "client diagnostic" in stderr
    assert "timed out" in result.errors[0]


def test_nonbenchmark_ignores_unused_di_metadata(monkeypatch):
    from types import SimpleNamespace

    executor, task, events, _ = _endpoint_submission_probe(
        monkeypatch, fail_delivery=False
    )
    task.mode_config.mode_type = SimpleNamespace(value="analyze")
    task.mode_config.benchmark_config["envs"] = None
    monkeypatch.setattr(executor, "_find_gpu_node", lambda: "allocated-a")
    assert executor.submit(task) == task.task_id
    assert events == [("delivery", task.task_id)]
    assert executor._di_tasks == set()


def test_whole_di_evaluation_is_one_ordinary_ray_submission_without_endpoint(
    monkeypatch,
):
    executor, task, events, reference = _endpoint_submission_probe(
        monkeypatch, fail_delivery=False
    )
    captured = {}

    def remote(**options):
        captured.update(options)
        return lambda function: type(
            "Remote",
            (),
            {
                "remote": lambda self, payload: (
                    events.append(("delivery", payload["task_id"])),
                    reference,
                )[1]
            },
        )()

    sys.modules["ray"].remote = remote
    assert task.mode_config.benchmark_config["envs"] == {}
    assert executor.submit(task) == task.task_id
    assert events == [("delivery", task.task_id)]
    assert captured["num_gpus"] == 0
    assert captured["max_retries"] == 0
    assert captured["scheduling_strategy"].node_id == "allocated-a"
    assert captured["scheduling_strategy"].soft is True
    assert task.task_id in executor._di_tasks


def test_endpoint_url_alone_does_not_change_ordinary_ray_submission(monkeypatch):
    executor, task, events, _ = _endpoint_submission_probe(monkeypatch)
    task.mode_config.benchmark_config.pop("disaggregation")
    task.mode_config.benchmark_config["envs"] = {
        "BENCHMARK_BASE_URL": "http://not-a-ray-node",
        "MAGPIE_DI_CLIENT_HOST": "not-a-ray-node",
    }
    with pytest.raises(RuntimeError, match="connection lost"):
        executor.submit(task)
    assert events == [("delivery", task.task_id)]
    assert not executor._di_tasks


@pytest.mark.parametrize("disaggregation", [{}, {"prefill": {}, "decode": {}}])
def test_present_disaggregation_routes_whole_evaluation_even_before_validation(
    monkeypatch, disaggregation
):
    from Magpie.core.executor import EndpointClientCleanupError

    executor, task, _, _ = _endpoint_submission_probe(monkeypatch)
    task.mode_config.benchmark_config["disaggregation"] = disaggregation
    with pytest.raises(EndpointClientCleanupError):
        executor.submit(task)
    assert task.task_id in executor._di_tasks
