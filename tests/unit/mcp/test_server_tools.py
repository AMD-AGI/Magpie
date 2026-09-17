import asyncio
import json
from types import SimpleNamespace

import pytest

from Magpie.config import KernelEvalConfig, KernelType
from Magpie.eval import BaseKind, EvaluationState
from Magpie.mcp import server


def decoded(value):
    return json.loads(value)


def test_health_and_framework_config(monkeypatch, tmp_path):
    response = asyncio.run(server.health_check(None))
    assert response.status_code == 200
    config = tmp_path / "config.yaml"
    config.write_text(
        "scheduler:\n  max_workers: 3\nperformance:\n  timeout_seconds: 12\n"
    )
    monkeypatch.chdir(tmp_path)
    assert server._load_framework_config()["scheduler"]["max_workers"] == 3
    scheduler = server._get_scheduler_config_from_yaml("local")
    assert scheduler.max_workers == 3
    assert server._get_perf_settings_from_yaml()["timeout_seconds"] == 12
    assert "accordo" in server._get_correctness_settings_from_yaml()


def test_hardware_spec_single_all_and_error(monkeypatch):
    info = SimpleNamespace(to_dict=lambda: {"name": "gpu"})

    class Controller:
        def __init__(self, device_id=0):
            self.device_id = device_id

        def get_hardware_info(self):
            return info

    class Multi:
        def get_all_hardware_info(self):
            return {0: info, 1: info}

    monkeypatch.setattr("Magpie.utils.GPUController", Controller)
    monkeypatch.setattr("Magpie.utils.MultiGPUController", Multi)
    monkeypatch.setattr("Magpie.utils.get_gpu_count", lambda: 2)
    assert decoded(server.hardware_spec())["gpu_count"] == 2
    assert len(decoded(server.hardware_spec(include_all=True))["gpus"]) == 2
    monkeypatch.setattr(
        Controller,
        "get_hardware_info",
        lambda self: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    assert decoded(server.hardware_spec())["error"] == "offline"


def test_format_analysis_dict_and_object():
    config = KernelEvalConfig(kernel_id="demo", kernel_type=KernelType.HIP)
    raw = {
        "compiling_state": "SUCCESS",
        "correctness_state": "SUCCESS",
        "performance_state": "SUCCESS",
        "score": 1,
        "compiling_result": {"success": True, "compile_time_seconds": 1},
        "correctness_result": {"success": True},
        "performance_result": {"success": True, "summary": {"util": 90}},
    }
    formatted = server._format_analysis_result(raw, config, KernelType.HIP)
    assert formatted["performance_result"]["summary"] == {"util": 90}

    state = EvaluationState(
        compiling_state=BaseKind.SUCCESS,
        correctness_state=BaseKind.SUCCESS,
        performance_state=BaseKind.SUCCESS,
        score=2,
    )
    formatted = server._format_analysis_result(state, config, KernelType.HIP)
    assert formatted["score"] == 2


def test_result_formatters_cover_protocols():
    serializable = SimpleNamespace(to_dict=lambda: {"serialized": True})
    assert server._format_compiling_result(serializable) == {"serialized": True}
    assert server._format_correctness_result(serializable) == {"serialized": True}
    assert server._format_performance_result(serializable) == {"serialized": True}
    plain = SimpleNamespace(success=True, errors="none", compile_time_seconds=2)
    assert server._format_compiling_result(plain)["compile_time_seconds"] == 2
    assert server._format_correctness_result(plain)["success"] is True

    metrics = [
        SimpleNamespace(name="util", value=80, unit="%", peak=100, pct_of_peak=80)
    ]
    perf = SimpleNamespace(
        success=True, errors=None, workload_dir="work", metrics=metrics
    )
    formatted = server._format_performance_result(perf)
    assert formatted["summary"]["util"]["pct_of_peak"] == 80
    assert formatted["kernels"] == []


def test_discover_kernels_success_and_error(monkeypatch):
    monkeypatch.setattr(
        server,
        "discover_project_kernels",
        lambda **kwargs: {"kernels": [kwargs["kernel_type"]]},
    )
    assert decoded(server.discover_kernels(".", "cuda"))["kernels"] == ["cuda"]
    monkeypatch.setattr(
        server,
        "discover_project_kernels",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("scan failed")),
    )
    assert decoded(server.discover_kernels("."))["error"] == "scan failed"


def test_suggest_optimizations_all_rules_and_errors():
    unavailable = decoded(
        server.suggest_optimizations(
            json.dumps({"performance_state": "FAILED", "errors": ["bad"]})
        )
    )
    assert unavailable["error"] == "Performance analysis not available"
    data = {
        "kernel_id": "demo",
        "score": 0.4,
        "performance_state": "SUCCESS",
        "performance_result": {
            "summary": {
                "Active CUs": {"pct_of_peak": 20},
                "MFMA_Util": {"value": 5},
                "MFMA_FLOPs_F16": {"value": 1},
                "VALU_Util": {"value": 80},
                "VMEM_Util": {"value": 90},
            },
            "kernels": [
                {"duration_ns": {"avg": 1000}},
                {"duration_ns": {"avg": 2000}},
                {"duration_ns": {"avg": 30000}},
            ],
        },
    }
    result = decoded(server.suggest_optimizations(json.dumps(data)))
    assert result["summary"]["total_bottlenecks"] == 4
    assert result["summary"]["high_impact_suggestions"] == 3
    assert decoded(server.suggest_optimizations("{"))["error"] == "Invalid JSON input"


def test_create_kernel_config_variants(monkeypatch):
    result = decoded(
        server.create_kernel_config(
            "demo", "kernel.hip", "./test", working_dir="/work", compile_command="make"
        )
    )
    assert "working_dir: /work" in result["config_content"]
    assert "compile_command: make" in result["config_content"]
    monkeypatch.setattr(
        "yaml.dump", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("yaml failed"))
    )
    assert (
        decoded(server.create_kernel_config("demo", "x", "test"))["error"]
        == "yaml failed"
    )


class FakeScheduler:
    initialized = True
    response = None
    instances = []

    def __init__(self, config):
        self.config = config
        self.shutdown_called = False
        self.__class__.instances.append(self)

    def initialize(self):
        return self.initialized

    def run_analyze(self, **kwargs):
        self.kwargs = kwargs
        return self.response

    def run_compare(self, **kwargs):
        self.kwargs = kwargs
        return self.response

    def shutdown(self):
        self.shutdown_called = True


def test_analyze_success_failure_and_initialization(monkeypatch):
    monkeypatch.setattr("Magpie.core.Scheduler", FakeScheduler)
    FakeScheduler.instances.clear()
    FakeScheduler.initialized = True
    FakeScheduler.response = SimpleNamespace(
        success=True,
        results=[
            {
                "compiling_state": "SUCCESS",
                "correctness_state": "SUCCESS",
                "performance_state": "SUCCESS",
                "score": 1,
            }
        ],
        errors=[],
    )
    result = decoded(
        server.analyze(
            "kernel.hip",
            "./test --quick",
            compile_command="make build",
            performance_backend="metrix",
            correctness_backend="accordo",
            accordo_kernel_name="kernel",
        )
    )
    assert result["score"] == 1
    assert result["kernel_config"]["compile_command"] == "make build"
    assert FakeScheduler.instances[-1].shutdown_called is True

    FakeScheduler.response = SimpleNamespace(
        success=False, results=[], errors=["failed"]
    )
    assert decoded(server.analyze("kernel.hip", "./test"))["error"] == "Analysis failed"
    FakeScheduler.initialized = False
    assert (
        decoded(server.analyze("kernel.hip", "./test"))["error"]
        == "Failed to initialize scheduler"
    )


def test_analyze_handles_scheduler_exception(monkeypatch):
    class Broken:
        def __init__(self, config):
            raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr("Magpie.core.Scheduler", Broken)
    result = decoded(server.analyze("kernel.hip", "./test"))
    assert result["error"] == "scheduler unavailable"
    assert result["kernel_config"]["kernel_path"] == "kernel.hip"


def test_compare_success_shapes_failures_and_validation(monkeypatch):
    monkeypatch.setattr("Magpie.core.Scheduler", FakeScheduler)
    assert "at least 2" in decoded(server.compare(["one"], ["test"]))["error"]
    assert "same length" in decoded(server.compare(["one", "two"], ["test"]))["error"]

    FakeScheduler.initialized = True
    FakeScheduler.response = SimpleNamespace(
        success=True, results={"winner": 0}, errors=[]
    )
    result = decoded(
        server.compare(
            ["one.hip", "two.hip"],
            ["./one", "./two"],
            performance_backend="rocprof_compute",
            correctness_backend="accordo",
            accordo_kernel_name="kernel",
        )
    )
    assert result["winner"] == 0
    assert len(result["kernel_configs"]) == 2

    FakeScheduler.response = SimpleNamespace(
        success=True,
        results=SimpleNamespace(to_dict=lambda: {"winner": 1}),
        errors=[],
    )
    assert decoded(server.compare(["one", "two"], ["a", "b"]))["winner"] == 1
    FakeScheduler.response = SimpleNamespace(success=True, results="plain", errors=[])
    assert decoded(server.compare(["one", "two"], ["a", "b"]))["result"] == "plain"
    FakeScheduler.response = SimpleNamespace(
        success=False, results=None, errors=["bad"]
    )
    assert (
        decoded(server.compare(["one", "two"], ["a", "b"]))["error"]
        == "Comparison failed"
    )
    FakeScheduler.initialized = False
    assert (
        decoded(server.compare(["one", "two"], ["a", "b"]))["error"]
        == "Failed to initialize scheduler"
    )


def test_configure_gpu_reset_config_and_validation(monkeypatch):
    class Controller:
        def __init__(self, device_ids=None):
            self.device_ids = device_ids

        def reset_all(self):
            return {0: True}

        def apply_config(self, config):
            return {0: True, 1: False}

    monkeypatch.setattr("Magpie.utils.MultiGPUController", Controller)
    assert decoded(server.configure_gpu(reset=True))["action"] == "reset"
    result = decoded(
        server.configure_gpu(
            [0, 1],
            power_limit_watts=300,
            gpu_clock_mhz=[1000, 1500],
            mem_clock_mhz=[800, 1000],
        )
    )
    assert result["action"] == "configure"
    assert result["config"]["power_limit_watts"] == 300
    assert "must be" in decoded(server.configure_gpu(gpu_clock_mhz=[1]))["error"]
    assert "must be" in decoded(server.configure_gpu(mem_clock_mhz=[1]))["error"]


def test_benchmark_tool_local_ray_and_error(monkeypatch, tmp_path):
    created = []

    class Benchmarker:
        def __init__(self, config, output_dir):
            self.config = config
            self.output_dir = output_dir
            created.append(self)

        def run(self):
            return SimpleNamespace(
                to_dict=lambda: {"success": True},
                get_summary=lambda: "complete",
            )

        def cleanup(self):
            self.cleaned = True

    monkeypatch.setattr("Magpie.modes.benchmark.BenchmarkMode", Benchmarker)
    result = decoded(
        asyncio.run(
            server.benchmark(
                "vllm",
                "demo",
                run_mode="ray",
                tp=4,
                extra_envs={"CUSTOM": "yes"},
                ray_num_nodes=2,
                ray_total_num_gpus=8,
                output_dir=str(tmp_path),
            )
        )
    )
    assert result["success"] is True
    assert result["summary_text"] == "complete"
    assert created[-1].config.ray_config.gpus_per_node == 4
    assert created[-1].cleaned is True

    class Broken(Benchmarker):
        def run(self):
            raise RuntimeError("benchmark failed")

    monkeypatch.setattr("Magpie.modes.benchmark.BenchmarkMode", Broken)
    result = decoded(asyncio.run(server.benchmark("vllm", "demo")))
    assert result["error"] == "benchmark failed"


def test_gap_analysis_tool_success_empty_missing_and_error(monkeypatch, tmp_path):
    traces = tmp_path / "run" / "torch_trace"
    traces.mkdir(parents=True)

    class Result:
        rank_results = [{}, {}]
        total_duration_us = 10
        merged_kernels = ["kernel"]
        errors = []

        def to_dict(self):
            return {"top_kernels": [{"name": "gemm"}]}

        def to_csv(self, path):
            path.write_text("csv")
            return path

        def to_rank_csv(self, directory):
            return [directory / "rank0.csv"]

    class Analyzer:
        def __init__(self, config):
            self.config = config

        def analyze(self, path):
            return Result()

        def generate_clamped_traces(self, path, output_dir):
            return [output_dir / "clamped.json"]

    monkeypatch.setattr("Magpie.modes.benchmark.gap_analysis.GapAnalyzer", Analyzer)
    result = decoded(
        server.gap_analysis(str(tmp_path / "run"), generate_clamped_traces=True)
    )
    assert result["num_ranks"] == 2
    assert result["top_kernels"][0]["name"] == "gemm"
    assert result["rank_csv_files"]
    assert result["clamped_trace_files"]
    assert (
        "not found" in decoded(server.gap_analysis(str(tmp_path / "missing")))["error"]
    )

    monkeypatch.setattr(
        Analyzer,
        "analyze",
        lambda self, path: (_ for _ in ()).throw(RuntimeError("parse failed")),
    )
    assert decoded(server.gap_analysis(str(traces)))["error"] == "parse failed"


def test_list_images_success_unknown_and_error(monkeypatch):
    class Selector:
        def list_available_images(self, framework=None):
            return {"vllm": {"gfx942": "image", "odd": "other"}}

        def get_runner_type(self, arch):
            if arch == "odd":
                raise ValueError("unknown")
            return "mi300x"

    monkeypatch.setattr("Magpie.modes.benchmark.ImageSelector", Selector)
    result = decoded(server.list_benchmark_images("vllm"))
    assert result["details"]["vllm"]["gfx942"]["runner_type"] == "mi300x"
    assert result["details"]["vllm"]["odd"]["runner_type"] == "unknown"
    monkeypatch.setattr(
        Selector,
        "list_available_images",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("missing config")),
    )
    assert decoded(server.list_benchmark_images())["error"] == "missing config"


def test_list_and_get_benchmark_results(monkeypatch, tmp_path):
    workspace = tmp_path / "benchmark_vllm_20260101"
    (workspace / "torch_trace").mkdir(parents=True)
    (workspace / "torch_trace" / "trace.json").write_text("{}")
    (workspace / "tracelens_rank0_csvs").mkdir()
    (workspace / "tracelens_rank0_csvs" / "summary.csv").write_text("csv")
    (workspace / "config.yaml").write_text("model: demo\n")
    report = {
        "success": True,
        "framework": "vllm",
        "model": "demo",
        "execution_time": 2,
        "throughput": {"request_throughput": 3},
        "latency": {"ttft_mean": 4},
        "top_bottlenecks": ["a", "b"],
        "kernel_summary": [{"name": "gemm"}],
        "errors": [],
    }
    (workspace / "benchmark_report.json").write_text(json.dumps(report))
    (workspace / "inferencex_result.json").write_text('{"raw": true}')
    (workspace / "summary.txt").write_text("complete")
    monkeypatch.setattr(
        "Magpie.modes.benchmark.WorkspaceManager.list_workspaces",
        lambda base_dir: [str(workspace)],
    )
    listed = decoded(server.list_benchmark_results(str(tmp_path)))
    assert listed["runs"][0]["has_torch_trace"] is True
    assert listed["runs"][0]["has_tracelens"] is True
    detailed = decoded(
        server.get_benchmark_result(
            str(workspace), include_raw_result=True, include_tracelens_files=True
        )
    )
    assert detailed["raw_inferencex_result"] == {"raw": True}
    assert detailed["tracelens_files"]["rank0_csvs"]
    assert detailed["torch_trace_files"]
    assert (
        "not found"
        in decoded(server.get_benchmark_result(str(tmp_path / "missing")))["error"]
    )


def test_list_results_handles_invalid_and_outer_error(monkeypatch, tmp_path):
    workspace = tmp_path / "benchmark_atom_date"
    workspace.mkdir()
    (workspace / "config.yaml").write_text("{")
    (workspace / "benchmark_report.json").write_text("{")
    monkeypatch.setattr(
        "Magpie.modes.benchmark.WorkspaceManager.list_workspaces",
        lambda base_dir: [str(workspace)],
    )
    entry = decoded(server.list_benchmark_results(str(tmp_path)))["runs"][0]
    assert entry["config"] is None
    assert entry["report_error"]
    monkeypatch.setattr(
        "Magpie.modes.benchmark.WorkspaceManager.list_workspaces",
        lambda base_dir: (_ for _ in ()).throw(RuntimeError("scan failed")),
    )
    assert (
        decoded(server.list_benchmark_results(str(tmp_path)))["error"] == "scan failed"
    )


def test_compare_benchmark_reports_success_validation_and_error(monkeypatch, tmp_path):
    workspaces = []
    for name in ("baseline", "optimized"):
        workspace = tmp_path / name
        (workspace / "tracelens_rank0_csvs").mkdir(parents=True)
        (workspace / "benchmark_report.json").write_text(
            json.dumps(
                {"framework": "vllm", "model": name, "throughput": {}, "latency": {}}
            )
        )
        workspaces.append(str(workspace))

    class Analyzer:
        def __init__(self, config):
            self.config = config

        def compare_reports(self, **kwargs):
            return {"files": ["comparison.csv"], "error": None}

    monkeypatch.setattr("Magpie.modes.benchmark.tracelens.TraceLensAnalyzer", Analyzer)
    result = decoded(
        server.compare_benchmark_reports(
            workspaces, ["before", "after"], str(tmp_path / "out")
        )
    )
    assert result["success"] is True
    assert len(result["run_summaries"]) == 2
    assert (
        "At least 2"
        in decoded(server.compare_benchmark_reports(workspaces[:1]))["error"]
    )
    empty = tmp_path / "empty"
    empty.mkdir()
    assert (
        "No TraceLens"
        in decoded(server.compare_benchmark_reports([workspaces[0], str(empty)]))[
            "error"
        ]
    )
    monkeypatch.setattr(
        Analyzer,
        "compare_reports",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("compare failed")),
    )
    assert (
        decoded(server.compare_benchmark_reports(workspaces))["error"]
        == "compare failed"
    )


class FakeRayExecutor:
    def __init__(self):
        self.running = True
        self.stopped = False

    def is_running(self):
        return self.running

    def stop(self):
        self.stopped = True

    def get_task_status_ray(self, task_id):
        return "SUCCEEDED"

    def get_task_result(self, task_id):
        return {"success": True} if task_id == "done" else None

    def cancel_task(self, task_id):
        return task_id == "running"

    def list_tasks(self):
        return {"done": "SUCCEEDED", "running": "RUNNING"}


def test_ray_task_tools_success_and_errors(monkeypatch):
    executor = FakeRayExecutor()
    monkeypatch.setattr(server, "_get_ray_executor", lambda *args: executor)
    assert decoded(server.ray_task_status("done"))["status"] == "SUCCEEDED"
    assert decoded(server.ray_task_result("done"))["_task_id"] == "done"
    assert decoded(server.ray_task_result("waiting"))["status"] == "SUCCEEDED"
    assert decoded(server.ray_task_cancel("running"))["cancelled"] is True
    assert decoded(server.ray_task_cancel("done"))["cancelled"] is False
    assert decoded(server.ray_task_list())["total"] == 2

    monkeypatch.setattr(
        server,
        "_get_ray_executor",
        lambda *args: (_ for _ in ()).throw(RuntimeError("ray down")),
    )
    assert decoded(server.ray_task_status("x"))["error"] == "ray down"
    assert decoded(server.ray_task_result("x"))["error"] == "ray down"
    assert decoded(server.ray_task_cancel("x"))["error"] == "ray down"
    assert decoded(server.ray_task_list())["error"] == "ray down"


def test_ray_executor_cache_and_replacement(monkeypatch):
    current = FakeRayExecutor()
    server._ray_executor_instance = current
    server._ray_executor_key = ("auto", "/shared")
    assert server._get_ray_executor("auto", "/shared") is current

    created = []

    class NewExecutor(FakeRayExecutor):
        def __init__(self, config, ray_config):
            super().__init__()
            created.append((config, ray_config))

        def start(self):
            self.started = True

    monkeypatch.setattr("Magpie.core.ray_executor.RayJobExecutor", NewExecutor)
    replacement = server._get_ray_executor("ray://host", "/new")
    assert current.stopped is True
    assert replacement.started is True
    assert created[-1][1].cluster_address == "ray://host"


def test_benchmark_batch_parallel_sequential_empty_and_item_errors(monkeypatch):
    class Mode:
        calls = 0

        def __init__(self, config, output_dir):
            self.config = config

        def submit_ray_benchmark(self, executor):
            self.__class__.calls += 1
            if self.config.model == "bad":
                return SimpleNamespace(
                    success=False, errors=["submit failed"], metadata={}
                )
            return SimpleNamespace(
                success=True, errors=[], metadata={"task_id": "task-1"}
            )

        def run(self):
            if self.config.model == "explode":
                raise RuntimeError("run failed")
            return SimpleNamespace(
                metadata={"ray_job_id": "job-1"}, workspace_dir="/results"
            )

    monkeypatch.setattr("Magpie.modes.benchmark.BenchmarkMode", Mode)
    monkeypatch.setattr(server, "_get_ray_executor", lambda *args: FakeRayExecutor())
    assert (
        decoded(asyncio.run(server.benchmark_batch([])))["error"]
        == "configs list is empty"
    )
    configs = [
        {"framework": "vllm", "model": "demo"},
        {"framework": "vllm", "model": "bad"},
    ]
    parallel = decoded(asyncio.run(server.benchmark_batch(configs, parallel=True)))
    assert parallel["submitted"] == 1
    assert parallel["failed"] == 1
    sequential = decoded(
        asyncio.run(
            server.benchmark_batch(
                [configs[0], {"framework": "vllm", "model": "explode"}],
                parallel=False,
            )
        )
    )
    assert sequential["submitted"] == 1
    assert sequential["failed"] == 1
