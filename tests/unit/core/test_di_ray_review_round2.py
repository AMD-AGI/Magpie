"""Multi-evaluation cleanup and persistent DI result-identity regressions."""

import sys

import pytest

from Magpie.core.executor import EndpointClientCleanupError, ExecutorConfig
from Magpie.core.ray_executor import RayJobExecutor
from Magpie.core.task import TaskStatus


class FakeRay:
    class exceptions:
        class TaskCancelledError(Exception):
            pass

    def __init__(self, results):
        self.results = results
        self.cancelled = []
        self.shutdown_calls = 0

    def wait(self, refs, **_kwargs):
        return refs, []

    def get(self, ref):
        result = self.results[ref]
        if isinstance(result, Exception):
            raise result
        return result

    def cancel(self, ref, force=False, recursive=True):
        self.cancelled.append((ref, force))

    def is_initialized(self):
        return True

    def shutdown(self):
        self.shutdown_calls += 1


def test_stop_attempts_every_di_evaluation_and_retains_uncertain_ownership(
    monkeypatch, ray_config,
):
    ray = FakeRay({
        "uncertain": {"di_cleanup_unconfirmed": True},
        "clean": {"success": False, "errors": ["accuracy failed"]},
        "errored": RuntimeError("remote transport error"),
    })
    monkeypatch.setitem(sys.modules, "ray", ray)
    executor = RayJobExecutor(ExecutorConfig("ray"), ray_config)
    executor._is_running = executor._ray_inited = True
    executor._di_tasks.update({"a-missing", "b-uncertain", "c-clean", "d-errored"})
    executor._di_task_configs = {key: {} for key in executor._di_tasks}
    executor._obj_refs = {"b-uncertain": "uncertain", "c-clean": "clean", "d-errored": "errored"}
    executor._results_cache["saved"] = {"success": True}
    executor._pending_tasks["saved"] = object()

    with pytest.raises(EndpointClientCleanupError) as failure:
        executor.stop()

    assert ray.cancelled == [("uncertain", False), ("clean", False), ("errored", False)]
    message = str(failure.value)
    assert "a-missing" in message and "no task reference" in message
    assert "b-uncertain" in message and "d-errored" in message
    assert "c-clean" not in message
    assert executor._di_tasks == {"a-missing", "b-uncertain", "d-errored"}
    assert set(executor._obj_refs) == {"b-uncertain", "c-clean", "d-errored"}
    assert set(executor._di_task_configs) == {"a-missing", "b-uncertain", "c-clean", "d-errored"}
    assert executor._results_cache["saved"] == {"success": True}
    assert executor._results_cache["b-uncertain"] == {"di_cleanup_unconfirmed": True}
    assert "saved" in executor._pending_tasks
    assert executor._is_running and executor._ray_inited
    assert ray.shutdown_calls == 0

    # Retry waits for already requested cancellations, but never interrupts
    # their cleanup a second time or revisits the confirmed evaluation.
    with pytest.raises(EndpointClientCleanupError):
        executor.stop()
    assert len(ray.cancelled) == 3
    assert ray.shutdown_calls == 0


def test_stop_continues_after_unexpected_cancellation_exception(monkeypatch, ray_config):
    executor = RayJobExecutor(ExecutorConfig("ray"), ray_config)
    executor._di_tasks.update({"a-failure", "b-clean"})
    executor._obj_refs = {key: key for key in executor._di_tasks}
    calls = []

    def cancel(task_id):
        calls.append(task_id)
        if task_id == "a-failure":
            raise RuntimeError("cancellation dispatch failed")
        executor._di_tasks.discard(task_id)
        return True

    monkeypatch.setattr(executor, "cancel_task", cancel)
    with pytest.raises(EndpointClientCleanupError, match="cancellation dispatch failed"):
        executor.stop()
    assert calls == ["a-failure", "b-clean"]
    assert executor._di_tasks == {"a-failure"}


@pytest.mark.parametrize("wait_all", [False, True])
def test_post_cancel_wait_preserves_di_failure_and_error_details(
    monkeypatch, ray_config, wait_all,
):
    result = {"success": False, "errors": ["accuracy failed"], "execution_time": 1.5}
    ray = FakeRay({"ref": result})
    monkeypatch.setitem(sys.modules, "ray", ray)
    executor = RayJobExecutor(ExecutorConfig("ray"), ray_config)
    executor._obj_refs["di"] = "ref"
    executor._di_tasks.add("di")
    executor._di_task_configs["di"] = {}
    assert executor.cancel_task("di")
    assert "di" not in executor._di_tasks

    actual = executor.wait_all()[0] if wait_all else executor.wait_for_task("di")
    assert actual.status is TaskStatus.FAILED
    assert actual.errors == ["accuracy failed"]
    assert actual.results == result
    assert actual.execution_time == 1.5
    assert ray.cancelled == [("ref", False)]


@pytest.mark.parametrize("wait_all", [False, True])
def test_persistent_di_identity_rejects_unconfirmed_cleanup(monkeypatch, ray_config, wait_all):
    ray = FakeRay({"ref": {"success": False, "di_cleanup_unconfirmed": True, "error": "cleanup pending"}})
    monkeypatch.setitem(sys.modules, "ray", ray)
    executor = RayJobExecutor(ExecutorConfig("ray"), ray_config)
    executor._obj_refs["di"] = "ref"
    executor._di_task_configs["di"] = {}
    # Identity survives active-set removal; an uncertain result cannot become
    # an ordinary completed evaluation through a later synchronous read.
    with pytest.raises(EndpointClientCleanupError, match="cleanup pending"):
        executor.wait_all() if wait_all else executor.wait_for_task("di")
    assert not ray.cancelled


def test_ordinary_wait_success_false_retains_upstream_behavior(monkeypatch, ray_config):
    ray = FakeRay({"ref": {"success": False, "errors": ["ordinary payload"]}})
    monkeypatch.setitem(sys.modules, "ray", ray)
    executor = RayJobExecutor(ExecutorConfig("ray"), ray_config)
    executor._obj_refs["ordinary"] = "ref"
    result = executor.wait_for_task("ordinary")
    assert result.status is TaskStatus.COMPLETED
    assert result.errors == []


@pytest.mark.parametrize("di", [False, True])
def test_di_cancellation_preserves_child_cleanup_rpcs_without_changing_ordinary(
    monkeypatch, ray_config, di,
):
    ray = FakeRay({"ref": {"success": True}})
    calls = []
    monkeypatch.setitem(sys.modules, "ray", ray)
    ray.cancel = lambda ref, **kwargs: calls.append((ref, kwargs))
    executor = RayJobExecutor(ExecutorConfig("ray"), ray_config)
    executor._obj_refs["evaluation"] = "ref"
    if di:
        executor._di_tasks.add("evaluation")
        executor._di_task_configs["evaluation"] = {}
    assert executor.cancel_task("evaluation")
    assert calls == [("ref", {"force": False, "recursive": False} if di else {"force": True})]
