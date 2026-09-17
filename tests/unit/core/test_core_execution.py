import json
import os
import subprocess
import sys
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from Magpie.config import KernelEvalConfig, KernelType
from Magpie.core.executor import (
    ContainerExecutor,
    ExecutorConfig,
    ExecutorType,
    LocalExecutor,
    _execute_task_worker,
    _local_pool_worker_init,
    create_executor,
)
from Magpie.core.job_store import JobRecord, JobStore
from Magpie.core.ray_executor import RayJobExecutor
from Magpie.core.task import ModeConfig, ModeType, Task, TaskResult, TaskStatus


class Lock:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class FakePool:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.future = Future()
        self.shutdown_calls = []

    def submit(self, function, payload):
        self.submission = (function, payload)
        return self.future

    def shutdown(self, wait=True):
        self.shutdown_calls.append(wait)


def completed_payload(task_id="task-analyze"):
    return {
        "task_id": task_id,
        "status": "completed",
        "success": True,
        "results": {"value": 1},
        "errors": [],
        "execution_time": 0.1,
        "metadata": {"source": "unit"},
    }


def test_task_and_result_serialization(kernel_config):
    task = Task(
        task_id="task",
        kernel_configs=[kernel_config, "invalid"],
        mode_config={"mode_type": ModeType.COMPARE},
        status="running",
        metadata={"owner": "test"},
    )
    assert task.status is TaskStatus.RUNNING
    assert task.mode_type is ModeType.COMPARE
    payload = task.to_dict()
    assert payload["kernel_configs"][0]["kernel_id"] == "unit-kernel"
    assert payload["kernel_configs"][1] == "invalid"

    value = SimpleNamespace(to_dict=lambda: {"score": 2})
    list_result = TaskResult("task", results=[value, {"raw": True}])
    assert list_result.success is True
    assert list_result.to_dict()["results"] == [{"score": 2}, {"raw": True}]
    assert TaskResult("task", results=value).to_dict()["results"] == {"score": 2}
    assert TaskResult("task", results="raw").to_dict()["results"] == "raw"
    failed = TaskResult("task", status="failed", errors=["boom"])
    assert failed.success is False


def test_job_store_crud_filters_and_invalid_metadata(tmp_path):
    store = JobStore(str(tmp_path / "jobs" / "store.db"))
    first = JobRecord(
        ray_job_id="ray-1",
        magpie_task_id="task-1",
        mode_type="benchmark",
        ray_cluster="auto",
        config_path="config.json",
        result_path="result.json",
        submitted_at=1,
        metadata={"model": "demo"},
    )
    second = JobRecord(
        ray_job_id="ray-2",
        magpie_task_id="task-2",
        mode_type="analyze",
        ray_cluster="auto",
        config_path="two.json",
        result_path="two-result.json",
        submitted_at=2,
        status="RUNNING",
    )
    store.add(first)
    store.add(second)
    assert store.get("missing") is None
    assert store.get("ray-1").metadata == {"model": "demo"}
    assert store.get_by_task_id("task-2").ray_job_id == "ray-2"
    assert store.get_by_task_id("missing") is None
    assert [r.ray_job_id for r in store.list_jobs()] == ["ray-2", "ray-1"]
    assert [r.ray_job_id for r in store.list_jobs(status="RUNNING")] == ["ray-2"]
    assert [r.ray_job_id for r in store.list_jobs(mode_type="benchmark")] == ["ray-1"]
    store.update_status("ray-1", "SUCCEEDED")
    assert store.get("ray-1").status == "SUCCEEDED"
    store._conn.execute(
        "UPDATE jobs SET metadata = ? WHERE ray_job_id = ?", ("{", "ray-1")
    )
    store._conn.commit()
    assert store.get("ray-1").metadata == {}
    store.delete("ray-1")
    assert store.get("ray-1") is None
    store.close()


def test_local_worker_gpu_binding(monkeypatch):
    counter = SimpleNamespace(value=1)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _local_pool_worker_init(counter, Lock(), (3, 7))
    assert counter.value == 2
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"
    assert os.environ["HIP_VISIBLE_DEVICES"] == "7"
    assert os.environ["ROCR_VISIBLE_DEVICES"] == "7"
    _local_pool_worker_init(counter, Lock(), ())


def test_local_executor_lifecycle_submit_wait_and_execute(task_factory, monkeypatch):
    pools = []

    def pool(**kwargs):
        pools.append(FakePool(**kwargs))
        return pools[-1]

    monkeypatch.setattr("Magpie.core.executor.get_gpu_count", lambda: 2)
    monkeypatch.setattr("Magpie.core.executor.ProcessPoolExecutor", pool)
    monkeypatch.setattr(
        "Magpie.core.executor.multiprocessing.Value",
        lambda *_: SimpleNamespace(value=0),
    )
    monkeypatch.setattr("Magpie.core.executor.multiprocessing.Lock", Lock)
    executor = LocalExecutor(
        ExecutorConfig(max_workers=2, gpu_devices=[0], timeout_seconds=1)
    )
    assert executor.is_running() is False
    assert executor.start() is True
    assert executor.start() is True
    assert executor._available_gpus == [0, 1]
    task = task_factory()
    assert executor.get_task_status(task.task_id) is None
    assert executor.submit(task) == task.task_id
    assert executor.get_task_status(task.task_id) is TaskStatus.RUNNING
    pools[0].future.set_result(completed_payload(task.task_id))
    result = executor.wait_for_task(task.task_id, timeout=1)
    assert result.success is True
    assert executor.wait_all()[0].success is True

    monkeypatch.setattr(
        "Magpie.core.executor._execute_task_worker",
        lambda _: completed_payload(task.task_id),
    )
    assert executor.execute(task).success is True
    executor.stop()
    assert pools[0].shutdown_calls == [True]
    assert executor.is_running() is False


def test_local_executor_errors_and_gpu_validation(task_factory, monkeypatch):
    executor = LocalExecutor(ExecutorConfig(gpu_devices=[]))
    with pytest.raises(RuntimeError, match="not running"):
        executor.submit(task_factory())
    with pytest.raises(ValueError, match="not found"):
        executor.wait_for_task("missing")

    future = Future()
    future.set_exception(ValueError("worker failed"))
    executor._futures["bad"] = future
    assert executor.wait_for_task("bad").status is TaskStatus.FAILED

    monkeypatch.setattr(
        "Magpie.core.executor._execute_task_worker",
        lambda _: (_ for _ in ()).throw(ValueError("bad")),
    )
    assert executor.execute(task_factory()).status is TaskStatus.FAILED
    monkeypatch.setattr("Magpie.core.executor.get_gpu_count", lambda: 0)
    with pytest.raises(RuntimeError, match="No GPU"):
        executor._check_gpu_availability()


def test_local_executor_start_failure(monkeypatch):
    monkeypatch.setattr("Magpie.core.executor.get_gpu_count", lambda: 1)
    monkeypatch.setattr(
        "Magpie.core.executor.ProcessPoolExecutor",
        lambda **_: (_ for _ in ()).throw(OSError("no pool")),
    )
    assert LocalExecutor(ExecutorConfig()).start() is False


def test_container_executor_lifecycle_and_commands(task_factory, monkeypatch):
    with pytest.raises(ValueError, match="docker_image"):
        ContainerExecutor(ExecutorConfig(executor_type="container"))
    executor = ContainerExecutor(
        ExecutorConfig(
            executor_type="container", docker_image="image:tag", gpu_devices=[0, 2]
        )
    )
    monkeypatch.setattr(executor, "_check_docker_availability", lambda: True)
    monkeypatch.setattr(executor, "_pull_image", lambda: True)
    assert executor.start() is True
    assert executor.start() is True
    task = task_factory()
    assert executor.submit(task) == task.task_id

    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="ok")

    monkeypatch.setattr("Magpie.core.executor.subprocess.run", run)
    assert executor.execute(task).success is True
    assert '"device=0,2"' in calls[0][0]
    executor._container_ids[task.task_id] = "container-id"
    executor.stop()
    assert calls[-2][0][:2] == ["docker", "stop"]
    assert calls[-1][0][:2] == ["docker", "rm"]


def test_container_executor_failure_timeout_and_helpers(task_factory, monkeypatch):
    executor = ContainerExecutor(ExecutorConfig(docker_image="image", gpu_devices=[]))
    with pytest.raises(RuntimeError, match="not running"):
        executor.submit(task_factory())
    assert executor._build_gpu_args() == []

    monkeypatch.setattr(
        "Magpie.core.executor.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 2, stderr="bad"),
    )
    assert executor._check_docker_availability() is False
    assert executor._pull_image() is False
    assert executor.execute(task_factory()).status is TaskStatus.FAILED

    monkeypatch.setattr(
        "Magpie.core.executor.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired(a[0], 1)),
    )
    assert executor._check_docker_availability() is False
    assert executor._pull_image() is False
    assert executor.execute(task_factory()).errors == ["Container execution timed out"]

    monkeypatch.setattr(
        "Magpie.core.executor.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(OSError("docker gone")),
    )
    assert executor.execute(task_factory()).errors == ["docker gone"]


def test_executor_factory(task_factory, ray_config):
    assert isinstance(create_executor(ExecutorConfig()), LocalExecutor)
    assert isinstance(
        create_executor(ExecutorConfig("container", docker_image="image")),
        ContainerExecutor,
    )
    with pytest.raises(ValueError, match="ray_config"):
        create_executor(ExecutorConfig("ray"))
    assert isinstance(
        create_executor(ExecutorConfig("ray"), ray_config=ray_config), RayJobExecutor
    )
    cfg = ExecutorConfig()
    cfg.executor_type = object()
    with pytest.raises(ValueError, match="Unknown"):
        create_executor(cfg)


def test_worker_analyze_compare_benchmark_and_failure(monkeypatch, task_factory):
    class Analyzer:
        def __init__(self, config):
            self.config = config

        def analyze(self, kernel):
            return {"kernel": kernel.kernel_id}

    class Comparator:
        def __init__(self, config):
            self.config = config

        def compare(self, kernels):
            return {"count": len(kernels)}

    class Benchmarker:
        def __init__(self, config):
            self.config = config

        def run(self, task_id):
            return SimpleNamespace(to_dict=lambda: {"task_id": task_id})

    monkeypatch.setattr("Magpie.modes.AnalyzeMode", Analyzer)
    monkeypatch.setattr("Magpie.modes.CompareMode", Comparator)
    monkeypatch.setattr("Magpie.modes.benchmark.BenchmarkMode", Benchmarker)
    assert _execute_task_worker(task_factory().to_dict())["status"] == "completed"
    assert (
        _execute_task_worker(task_factory(ModeType.COMPARE).to_dict())["status"]
        == "completed"
    )
    bench = task_factory(
        ModeType.BENCHMARK, benchmark_config={"framework": "vllm", "model": "demo"}
    )
    assert _execute_task_worker(bench.to_dict())["results"]["task_id"] == bench.task_id
    payload = task_factory().to_dict()
    payload["mode_config"]["mode_type"] = "invalid"
    assert _execute_task_worker(payload)["status"] == "failed"


class FakeRay:
    class exceptions:
        class TaskCancelledError(Exception):
            pass

    def __init__(self):
        self.initialized = False
        self.ready = True
        self.result = {"status": "ok", "execution_time": 2}
        self.cancelled = []

    def is_initialized(self):
        return self.initialized

    def init(self, **kwargs):
        self.initialized = True
        self.init_kwargs = kwargs

    def shutdown(self):
        self.initialized = False

    def wait(self, refs, **kwargs):
        return (refs if self.ready else [], [])

    def get(self, ref):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def cancel(self, ref, force=False):
        self.cancelled.append((ref, force))

    def nodes(self):
        return []


def test_ray_executor_lifecycle_wait_status_and_cancel(
    task_factory, ray_config, monkeypatch
):
    ray = FakeRay()
    monkeypatch.setitem(sys.modules, "ray", ray)
    executor = RayJobExecutor(ExecutorConfig("ray"), ray_config)
    assert executor.start() is True
    assert executor.start() is True
    monkeypatch.setattr(executor, "_submit_ray_task", lambda task: "ref")
    task = task_factory()
    assert executor.submit(task) == task.task_id
    assert executor.execute(task).success is True
    assert executor.get_task_status_ray(task.task_id) == "SUCCEEDED"
    assert executor.get_task_result(task.task_id)["status"] == "ok"
    assert executor.cancel_task(task.task_id) is True
    assert executor.cancel_task("missing") is False
    assert executor.list_tasks()[task.task_id] == "SUCCEEDED"
    executor.stop()
    assert executor.is_running() is False


def test_ray_executor_wait_variants(task_factory, ray_config, monkeypatch):
    ray = FakeRay()
    monkeypatch.setitem(sys.modules, "ray", ray)
    executor = RayJobExecutor(ExecutorConfig("ray"), ray_config)
    executor._is_running = True
    task = task_factory()
    executor._pending_tasks[task.task_id] = task
    executor._obj_refs[task.task_id] = "ref"
    with pytest.raises(ValueError, match="not found"):
        executor.wait_for_task("missing")
    ray.ready = False
    assert executor.wait_for_task(task.task_id, 0).status is TaskStatus.FAILED
    assert executor.get_task_status_ray(task.task_id) == "RUNNING"
    ray.ready = True
    ray.result = {"status": "failed", "error": "remote"}
    assert executor.wait_for_task(task.task_id).errors == ["remote"]
    executor._results_cache.clear()
    assert executor.get_task_status_ray(task.task_id) == "FAILED"
    ray.result = ValueError("ray get")
    assert executor.wait_for_task(task.task_id).status is TaskStatus.FAILED
    executor._results_cache.clear()
    assert executor.get_task_status_ray(task.task_id) == "FAILED"
    executor._obj_refs["unknown"] = None
    assert executor.get_task_status_ray("absent") == "UNKNOWN"


def test_ray_runtime_helpers(tmp_path, ray_config, monkeypatch):
    root = tmp_path / "magpie"
    root.mkdir()
    (root / "requirements.txt").write_text("# comment\nPyYAML>=6\n\n")
    ray_config.install_magpie = True
    ray_config.magpie_install_path = str(root)
    ray_config.pip_packages = ["pytest"]
    ray_config.multi_node = True
    ray_config.total_num_gpus = 8
    ray_config.env_vars = {"CUSTOM": "1"}
    executor = RayJobExecutor(ExecutorConfig("ray"), ray_config)
    assert RayJobExecutor._collect_pip_packages(ray_config) == [
        "pytest",
        "PyYAML>=6",
        str(root),
    ]
    env = executor._build_runtime_env()
    assert env["env_vars"]["RAY_ADDRESS"] == "auto"
    assert env["env_vars"]["MAGPIE_TOTAL_GPUS"] == "8"
    assert env["env_vars"]["CUSTOM"] == "1"
    assert env["pip"][-1] == str(root)

    ray = FakeRay()
    ray.nodes = lambda: [
        {"Alive": False, "Resources": {"GPU": 1}, "NodeID": "dead"},
        {
            "Alive": True,
            "Resources": {"GPU": 1, "node:__internal_head__": 1},
            "NodeID": "head",
        },
        {"Alive": True, "Resources": {"GPU": 2}, "NodeID": "worker"},
    ]
    monkeypatch.setitem(sys.modules, "ray", ray)
    assert RayJobExecutor._find_gpu_node() == "worker"
    ray.nodes = lambda: [
        {
            "Alive": True,
            "Resources": {"GPU": 1, "node:__internal_head__": 1},
            "NodeID": "head",
        }
    ]
    assert RayJobExecutor._find_gpu_node() == "head"


def test_ray_executor_failure_and_cancelled_paths(
    task_factory, ray_config, monkeypatch
):
    ray = FakeRay()
    ray.init = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("connect failed"))
    monkeypatch.setitem(sys.modules, "ray", ray)
    executor = RayJobExecutor(ExecutorConfig("ray"), ray_config)
    assert executor.start() is False
    with pytest.raises(RuntimeError, match="not running"):
        executor.submit(task_factory())

    executor._is_running = True
    task = task_factory()
    executor._pending_tasks[task.task_id] = task
    executor._obj_refs[task.task_id] = "ref"
    ray.get = lambda ref: (_ for _ in ()).throw(FakeRay.exceptions.TaskCancelledError())
    cancelled = executor.wait_for_task(task.task_id)
    assert cancelled.status is TaskStatus.CANCELLED
    assert task.status is TaskStatus.CANCELLED
    assert executor.wait_all()[-1].status is TaskStatus.CANCELLED

    ray.cancel = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("cancel failed")
    )
    assert executor.cancel_task(task.task_id) is False


def test_ray_executor_stop_and_runtime_env_minimal(ray_config, monkeypatch):
    ray = FakeRay()
    ray.initialized = True
    monkeypatch.setitem(sys.modules, "ray", ray)
    executor = RayJobExecutor(ExecutorConfig("ray"), ray_config)
    executor._ray_inited = True
    executor._is_running = True
    executor._obj_refs["task"] = "ref"
    executor._results_cache["task"] = {"ok": True}
    executor.stop()
    assert executor.is_running() is False
    assert executor._obj_refs == {}
    assert ray.initialized is False

    ray_config.install_magpie = False
    ray_config.pip_packages = []
    ray_config.multi_node = False
    runtime = executor._build_runtime_env()
    assert "pip" not in runtime
    assert "RAY_ADDRESS" not in runtime["env_vars"]


def test_ray_executor_builds_and_submits_remote_payload(
    task_factory, ray_config, monkeypatch
):
    captured = {}

    class RemoteFunction:
        def remote(self, payload):
            captured["payload"] = payload
            return "object-ref"

    ray = FakeRay()

    def remote(**options):
        captured["options"] = options

        def decorate(function):
            captured["function"] = function
            return RemoteFunction()

        return decorate

    ray.remote = remote
    monkeypatch.setitem(sys.modules, "ray", ray)

    class NodeAffinitySchedulingStrategy:
        def __init__(self, node_id, soft):
            self.node_id = node_id
            self.soft = soft

    scheduling = SimpleNamespace(
        NodeAffinitySchedulingStrategy=NodeAffinitySchedulingStrategy
    )
    monkeypatch.setitem(sys.modules, "ray.util", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", scheduling)

    ray_config.env_vars = {"CUSTOM": "yes"}
    executor = RayJobExecutor(ExecutorConfig("ray"), ray_config)
    monkeypatch.setattr(executor, "_find_gpu_node", lambda: "worker-node")
    task = task_factory(
        ModeType.BENCHMARK,
        benchmark_config={
            "framework": "vllm",
            "ray_config": {"cluster_address": "ray://override"},
        },
    )
    assert executor._submit_ray_task(task) == "object-ref"
    assert captured["options"]["num_gpus"] == 0
    assert captured["options"]["scheduling_strategy"].node_id == "worker-node"
    assert captured["payload"]["task_id"] == task.task_id
    assert captured["payload"]["ray_config"]["cluster_address"] == "ray://override"

    monkeypatch.setattr(executor, "_find_gpu_node", lambda: None)
    with pytest.raises(RuntimeError, match="No GPU node"):
        executor._submit_ray_task(task)
