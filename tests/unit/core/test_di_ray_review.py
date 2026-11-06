"""DI executor review regressions; ordinary Ray contracts remain unchanged."""

import sys
from types import SimpleNamespace

import pytest

from Magpie.core.executor import ExecutorConfig
from Magpie.core.ray_executor import (
    RayJobExecutor,
    di_cleanup_timeout_seconds,
    di_evaluation_timeout_seconds,
)
from Magpie.core.task import ModeType
from Magpie.modes.benchmark.config import RayConfig


class FakeRay:
    class exceptions:
        class TaskCancelledError(Exception):
            pass

    def __init__(self, result=None):
        self.result = result or {}
        self.cancelled, self.waits, self.options, self.payloads = [], [], [], []
        self.ready = True

    def wait(self, refs, **kwargs):
        self.waits.append(kwargs)
        return (refs if self.ready else [], [])

    def get(self, _ref):
        return self.result

    def cancel(self, ref, force=False):
        self.cancelled.append((ref, force))

    def remote(self, **options):
        self.options.append(options)

        def submit(payload):
            self.payloads.append(payload)
            return "ref"

        return lambda _function: SimpleNamespace(remote=submit)


def install_ray(monkeypatch, ray):
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.util", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "ray.util.scheduling_strategies",
        SimpleNamespace(NodeAffinitySchedulingStrategy=lambda **kwargs: kwargs),
    )


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("result", [
    {"success": False},
    {"success": True, "di_cleanup_unconfirmed": True},
    {"status": "failed", "errors": ["accuracy failed"]},
])
def test_di_async_failure_stays_failed_on_fresh_and_cached_poll(monkeypatch, cached, result):
    ray = FakeRay(result)
    install_ray(monkeypatch, ray)
    executor = RayJobExecutor(ExecutorConfig("ray"), RayConfig(install_magpie=False))
    executor._obj_refs["di"] = "ref"
    executor._di_task_configs["di"] = {"disaggregation": {}}
    # Even after cooperative cleanup removed active membership, its result is DI.
    if cached:
        executor._results_cache["di"] = result
    assert executor.get_task_status_ray("di") == "FAILED"
    assert executor.get_task_status_ray("di") == "FAILED"


def test_non_di_status_cache_and_success_false_keep_upstream_behavior(monkeypatch):
    ray = FakeRay({"success": False})
    install_ray(monkeypatch, ray)
    executor = RayJobExecutor(ExecutorConfig("ray"), RayConfig())
    executor._obj_refs["ordinary"] = "ref"
    assert executor.get_task_status_ray("ordinary") == "SUCCEEDED"
    executor._results_cache["ordinary"] = {"status": "failed"}
    assert executor.get_task_status_ray("ordinary") == "SUCCEEDED"


@pytest.mark.parametrize("di", [True, False])
def test_reused_executor_uses_task_ray_resources_and_environment_only_for_di(
    monkeypatch, task_factory, di,
):
    ray = FakeRay()
    install_ray(monkeypatch, ray)
    base = RayConfig(
        entrypoint_num_cpus=16, install_magpie=False,
        shared_storage_path="/base", env_vars={"SCOPE": "base"},
    )
    executor = RayJobExecutor(ExecutorConfig("ray"), base)
    monkeypatch.setattr(executor, "_find_gpu_node", lambda: "worker")
    config = {"ray_config": {
        "entrypoint_num_cpus": 1, "shared_storage_path": "/task",
        "env_vars": {"SCOPE": "task"}, "pip_packages": ["task-dependency==1"],
    }}
    if di:
        config["disaggregation"] = {"prefill": {"instances": 1}, "decode": {"instances": 1}}
    executor._submit_ray_task(task_factory(ModeType.BENCHMARK, benchmark_config=config))
    options = ray.options[-1]
    assert options["num_cpus"] == (1 if di else 16)
    assert options["runtime_env"]["env_vars"]["SCOPE"] == ("task" if di else "base")
    assert options["runtime_env"]["env_vars"]["HF_HOME"] == ("/task/hf_cache" if di else "/base/hf_cache")
    assert options["runtime_env"].get("pip") == (["task-dependency==1"] if di else None)
    assert ray.payloads[-1]["ray_config"]["entrypoint_num_cpus"] == 1
    assert base.entrypoint_num_cpus == 16
    assert base.env_vars == {"SCOPE": "base"}


def test_di_cancellation_retries_wait_without_interrupting_cleanup_again(monkeypatch):
    ray = FakeRay({"success": True})
    ray.ready = False
    install_ray(monkeypatch, ray)
    executor = RayJobExecutor(ExecutorConfig("ray"), RayConfig())
    config = {"disaggregation": {"prefill": {"instances": 2}, "decode": {"instances": 3}}}
    executor._obj_refs["di"] = "ref"
    executor._di_tasks.add("di")
    executor._di_task_configs["di"] = config
    assert not executor.cancel_task("di")
    assert "di" in executor._di_tasks
    ray.ready = True
    executor.stop()
    assert ray.cancelled == [("ref", False)]
    assert [call["timeout"] for call in ray.waits] == [210, 210]
    assert not executor._di_tasks


def test_di_cancel_delivery_error_is_not_resent(monkeypatch):
    ray = FakeRay({"success": True})
    install_ray(monkeypatch, ray)
    executor = RayJobExecutor(ExecutorConfig("ray"), RayConfig())
    executor._obj_refs["di"] = "ref"
    executor._di_tasks.add("di")

    def uncertain_delivery(ref, force=False):
        ray.cancelled.append((ref, force))
        raise RuntimeError("transport interrupted")

    ray.cancel = uncertain_delivery
    assert not executor.cancel_task("di")
    assert executor.cancel_task("di")
    assert executor.cancel_task("di")  # Confirmed cleanup is also idempotent.
    assert ray.cancelled == [("ref", False)]


def test_di_execute_uses_whole_evaluation_budget_and_ordinary_keeps_config(
    monkeypatch, task_factory,
):
    executor = RayJobExecutor(
        ExecutorConfig("ray", timeout_seconds=15),
        RayConfig(install_magpie=False),
    )
    monkeypatch.setattr(executor, "submit", lambda _task: None)
    monkeypatch.setattr(executor, "wait_for_task", lambda task_id, timeout: timeout)
    config = {
        "timeout_seconds": 900,
        "ray_config": {"placement_timeout_seconds": 70},
        "disaggregation": {"startup_timeout": 800, "prefill": {"instances": 2}, "decode": {"instances": 1}},
    }
    assert di_cleanup_timeout_seconds({"disaggregation": {}}) == 120
    assert di_cleanup_timeout_seconds(config) == 150
    assert di_evaluation_timeout_seconds(config, {"placement_timeout_seconds": 5}) == 1920
    assert executor.execute(task_factory(ModeType.BENCHMARK, benchmark_config=config)) == 1920
    assert executor.execute(task_factory(ModeType.BENCHMARK, benchmark_config={"timeout_seconds": 900})) == 15
    assert executor.execute(task_factory()) == 15
