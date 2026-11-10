"""DI executor ownership remains recoverable across cancellation and reuse."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from Magpie.core.executor import EndpointClientCleanupError, ExecutorConfig
from Magpie.core.ray_executor import RayJobExecutor, di_cleanup_timeout_seconds
from Magpie.modes.benchmark.config import RayConfig


def test_mcp_retains_executor_after_unconfirmed_di_cleanup(monkeypatch):
    from Magpie.mcp import server

    owner = Mock()
    owner.stop.side_effect = EndpointClientCleanupError("inspect /shared/di-session")
    key = ("ray://old", "/shared")
    monkeypatch.setattr(server, "_ray_executor_instance", owner)
    monkeypatch.setattr(server, "_ray_executor_key", key)
    replacement = Mock()
    monkeypatch.setattr("Magpie.core.ray_executor.RayJobExecutor", replacement)

    with pytest.raises(EndpointClientCleanupError, match="/shared/di-session"):
        server._get_ray_executor("ray://new", "/other")

    assert server._ray_executor_instance is owner
    assert server._ray_executor_key == key
    replacement.assert_not_called()


def test_mcp_ordinary_stop_failure_retains_existing_replacement_policy(monkeypatch):
    from Magpie.mcp import server

    owner = Mock()
    owner.stop.side_effect = RuntimeError("ordinary stop failed")
    monkeypatch.setattr(server, "_ray_executor_instance", owner)
    monkeypatch.setattr(server, "_ray_executor_key", ("old", "/old"))
    replacement = Mock()
    monkeypatch.setattr("Magpie.core.ray_executor.RayJobExecutor", replacement)

    assert server._get_ray_executor("new", "/new") is replacement.return_value
    replacement.return_value.start.assert_called_once_with()


def test_di_cleanup_budget_covers_inspection_then_stop_for_every_supervisor():
    config = {
        "disaggregation": {"prefill": {"instances": 1}, "decode": {"instances": 2}}
    }
    # Three inspections and three stops near their 30s deadlines, followed by
    # rank verification and local client/result handoff, fit the owner budget.
    assert di_cleanup_timeout_seconds(config) == 3 * 30 + 3 * 30 + 30 + 30


def test_cancel_and_stop_preserve_worker_recovery_path(monkeypatch, caplog):
    reference = object()
    error = "DI cleanup failed; inspect /shared/results/run/di-session/session.json"
    payload = {"di_cleanup_unconfirmed": True, "error": error}
    cancel = Mock()
    ray = SimpleNamespace(
        cancel=cancel,
        wait=lambda *_args, **_kwargs: ([reference], []),
        get=lambda _ref: payload,
        exceptions=SimpleNamespace(
            TaskCancelledError=type("Cancelled", (Exception,), {})
        ),
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    executor = RayJobExecutor(ExecutorConfig(), RayConfig())
    executor._di_tasks.add("run")
    executor._obj_refs["run"] = reference
    executor._di_task_configs["run"] = {}

    assert executor.cancel_task("run") is False
    assert error in caplog.text
    assert executor._results_cache["run"] == payload
    with pytest.raises(
        EndpointClientCleanupError, match="/shared/results/run/di-session"
    ):
        executor.stop()
    assert "run" in executor._di_tasks
    assert executor._obj_refs["run"] is reference
    cancel.assert_called_once_with(reference, force=False, recursive=False)


@pytest.mark.parametrize(
    "namespace,di,dispatch",
    [("other", True, False), ("connected", True, True), ("other", False, True)],
)
def test_di_namespace_checked_before_dispatch(monkeypatch, namespace, di, dispatch):
    ray = ModuleType("ray")
    ray.get_runtime_context = lambda: SimpleNamespace(namespace="connected")
    remote = Mock(
        return_value=lambda _function: SimpleNamespace(
            remote=Mock(return_value="reference")
        )
    )
    ray.remote = remote
    strategies = ModuleType("ray.util.scheduling_strategies")
    strategies.NodeAffinitySchedulingStrategy = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", strategies)
    benchmark_config = {"ray_config": {"namespace": namespace}}
    if di:
        benchmark_config["disaggregation"] = {"prefill": {}, "decode": {}}
    task = SimpleNamespace(
        task_id="run",
        mode_config=SimpleNamespace(
            mode_type=SimpleNamespace(value="benchmark"),
            benchmark_config=benchmark_config,
        ),
        to_dict=lambda: {"mode_config": {"benchmark_config": benchmark_config}},
    )
    executor = RayJobExecutor(ExecutorConfig(), RayConfig(install_magpie=False))
    runtime = Mock(return_value={"env_vars": {}})
    monkeypatch.setattr(executor, "_build_runtime_env", runtime)
    monkeypatch.setattr(executor, "_find_gpu_node", lambda: "gpu-node")

    if dispatch:
        assert executor._submit_ray_task(task) == "reference"
        remote.assert_called_once()
    else:
        with pytest.raises(ValueError, match="connected Ray namespace"):
            executor._submit_ray_task(task)
        runtime.assert_not_called()
        remote.assert_not_called()
        assert not executor._di_tasks
        assert not executor._di_task_configs
