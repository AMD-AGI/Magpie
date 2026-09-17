from types import SimpleNamespace

import pytest

from Magpie.core.executor import ExecutorType
from Magpie.core.scheduler import EnvironmentType, Scheduler, SchedulerConfig
from Magpie.core.task import ModeType, TaskResult, TaskStatus


class FakeExecutor:
    def __init__(self, start=True):
        self.start_result = start
        self.tasks = {}
        self.stopped = False

    def start(self):
        return self.start_result

    def stop(self):
        self.stopped = True

    def submit(self, task):
        task.status = TaskStatus.RUNNING
        self.tasks[task.task_id] = task
        return task.task_id

    def execute(self, task):
        return TaskResult(task.task_id, results={"mode": task.mode_type.value})

    def wait_for_task(self, task_id, timeout=None):
        return TaskResult(task_id, results={"timeout": timeout})

    def wait_all(self, timeout=None):
        return [TaskResult(task_id) for task_id in reversed(list(self.tasks))]

    def get_task_status(self, task_id):
        task = self.tasks.get(task_id)
        return task.status if task else None


def initialized_scheduler(config=None):
    scheduler = Scheduler(config)
    scheduler._executor = FakeExecutor()
    scheduler._is_initialized = True
    return scheduler


@pytest.mark.parametrize(
    ("environment", "executor_type"),
    [
        (EnvironmentType.LOCAL, ExecutorType.LOCAL),
        (EnvironmentType.CONTAINER, ExecutorType.CONTAINER),
        (EnvironmentType.RAY, ExecutorType.RAY),
    ],
)
def test_scheduler_config_and_executor_mapping(environment, executor_type):
    cfg = SchedulerConfig(environment_type=environment.value, max_workers=3)
    scheduler = Scheduler(cfg)
    executor_cfg = scheduler._create_executor_config()
    assert cfg.environment_type is environment
    assert executor_cfg.executor_type is executor_type
    assert executor_cfg.max_workers == 3


def test_scheduler_initialize_success_failure_and_repeat(monkeypatch):
    created = []

    def create(config, **kwargs):
        created.append((config, kwargs))
        return FakeExecutor()

    monkeypatch.setattr("Magpie.core.scheduler.create_executor", create)
    scheduler = Scheduler(SchedulerConfig(environment_type="ray"))
    assert scheduler.initialize() is True
    assert scheduler.initialize() is True
    assert created[0][0].executor_type is ExecutorType.RAY
    assert created[0][1]["ray_config"].cluster_address == "auto"

    monkeypatch.setattr(
        "Magpie.core.scheduler.create_executor", lambda *a, **k: FakeExecutor(False)
    )
    assert Scheduler().initialize() is False


def test_scheduler_create_submit_execute_wait_and_hooks(kernel_config):
    events = []

    def bad_pre(_task):
        raise ValueError("ignored")

    def bad_post(_task, _result):
        raise ValueError("ignored")

    cfg = SchedulerConfig(
        pre_hooks=[lambda task: events.append(("pre", task.task_id)), bad_pre],
        post_hooks=[
            lambda task, result: events.append(("post", result.task_id)),
            bad_post,
        ],
    )
    scheduler = initialized_scheduler(cfg)
    task = scheduler.create_task(
        [kernel_config],
        mode_type=ModeType.COMPARE,
        enable_default_compile=True,
        check_performance=False,
        gpu_arch="gfx942",
        profiler_args=["--x"],
        rocprof_config={"blocks": [1]},
        ncu_config={"set": "full"},
        metrix_config={"profile": "quick"},
        correctness_config={"backend": "accordo"},
        baseline_index=1,
        compare_config={"winner_strategy": "correctness"},
        priority=5,
        metadata={"owner": "unit"},
    )
    assert task.mode_config.baseline_index == 1
    assert task.priority == 5
    assert scheduler.submit(task) == task.task_id
    assert scheduler.get_task_status(task.task_id) is TaskStatus.RUNNING
    waited = scheduler.wait_for_task(task.task_id, 7)
    assert waited.results == {"timeout": 7}
    assert scheduler.wait_for_task(task.task_id) is waited

    direct = scheduler.create_task([kernel_config])
    assert scheduler.execute(direct).success is True
    assert scheduler.get_task_status(direct.task_id) is TaskStatus.COMPLETED
    assert events[0][0] == "pre"
    assert any(event[0] == "post" for event in events)


def test_scheduler_requires_initialization(task_factory):
    scheduler = Scheduler()
    with pytest.raises(RuntimeError, match="initialize"):
        scheduler.submit(task_factory())
    with pytest.raises(RuntimeError, match="initialize"):
        scheduler.execute(task_factory())
    with pytest.raises(RuntimeError, match="initialize"):
        scheduler.execute_batch([task_factory()])
    with pytest.raises(ValueError, match="not found"):
        scheduler.wait_for_task("missing")
    assert scheduler.wait_all() == []


def test_scheduler_batch_execution_and_order(kernel_config):
    scheduler = initialized_scheduler()
    tasks = [
        scheduler.create_task([kernel_config], metadata={"index": index})
        for index in range(3)
    ]
    results = scheduler.execute_batch(tasks)
    assert [result.task_id for result in results] == [task.task_id for task in tasks]
    assert scheduler.execute_batch([]) == []

    scheduler.wait_all = lambda timeout=None: []
    with pytest.raises(RuntimeError, match="Missing results"):
        scheduler.execute_batch([scheduler.create_task([kernel_config])])


def test_scheduler_convenience_methods(kernel_config, monkeypatch):
    scheduler = initialized_scheduler()
    seen = []

    def execute(task):
        seen.append(task)
        return TaskResult(task.task_id)

    monkeypatch.setattr(scheduler, "execute", execute)
    assert scheduler.run_analyze([kernel_config]).success is True
    assert seen[-1].mode_type is ModeType.ANALYZE
    assert scheduler.run_compare([kernel_config], baseline_index=0).success is True
    assert seen[-1].mode_type is ModeType.COMPARE
    assert scheduler.run_benchmark({"framework": "vllm", "model": "demo"}).success
    assert seen[-1].mode_type is ModeType.BENCHMARK

    batch_modes = []

    def execute_batch(tasks):
        batch_modes.extend(task.mode_type for task in tasks)
        return [TaskResult(task.task_id) for task in tasks]

    monkeypatch.setattr(scheduler, "execute_batch", execute_batch)
    assert len(scheduler.run_analyze_batch([[kernel_config], [kernel_config]])) == 2
    assert batch_modes == [ModeType.ANALYZE, ModeType.ANALYZE]
    batch_modes.clear()
    assert len(scheduler.run_compare_batch([[kernel_config], [kernel_config]])) == 2
    assert batch_modes == [ModeType.COMPARE, ModeType.COMPARE]


def test_scheduler_ray_benchmark_success_and_failure(monkeypatch):
    class Benchmarker:
        success = True

        def __init__(self, _config):
            pass

        def run(self, task_id):
            return SimpleNamespace(
                success=self.success,
                errors=[] if self.success else ["failed"],
                metadata={"ray": True},
                to_dict=lambda: {"task_id": task_id},
            )

    monkeypatch.setattr("Magpie.modes.benchmark.BenchmarkMode", Benchmarker)
    scheduler = Scheduler()
    config = {"framework": "vllm", "model": "demo"}
    result = scheduler.run_benchmark_ray(config, ray_cluster_address="ray://head")
    assert result.success is True
    assert config["run_mode"] == "ray"
    assert config["ray_config"]["cluster_address"] == "ray://head"
    Benchmarker.success = False
    assert scheduler.run_benchmark_ray(config).status is TaskStatus.FAILED


def test_scheduler_status_queue_clear_and_shutdown(task_factory):
    scheduler = Scheduler()
    pending = task_factory()
    running = task_factory(ModeType.COMPARE)
    running.task_id = "running"
    running.status = TaskStatus.RUNNING
    scheduler._task_queue = [pending, running]
    assert scheduler.get_task_status(pending.task_id) is TaskStatus.PENDING
    assert scheduler.get_pending_tasks() == [pending]
    scheduler.clear_tasks()
    assert scheduler._task_queue == [running]

    result = TaskResult("done")
    scheduler._completed_tasks["done"] = result
    assert scheduler.get_completed_results() == {"done": result}
    assert scheduler.get_task_status("done") is TaskStatus.COMPLETED
    assert scheduler.get_task_status("missing") is None
    executor = FakeExecutor()
    scheduler._executor = executor
    scheduler._is_initialized = True
    assert scheduler.is_initialized() is True
    scheduler.shutdown()
    assert executor.stopped is True
    assert scheduler.is_initialized() is False
