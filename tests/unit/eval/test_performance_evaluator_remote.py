import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from Magpie.config import (
    KernelEvalConfig,
    KernelType,
    MetrixConfig,
    NcuConfig,
    PerfBackend,
    PerformanceConfig,
    PipelineConfig,
    RocprofComputeConfig,
)
from Magpie.eval.compiling import CompilingResult
from Magpie.eval.correctness import CorrectnessResult
from Magpie.eval.evaluator import BaseKind, EvaluationState, Evaluator
from Magpie.eval.performance import (
    KernelMetrics,
    MetricResult,
    Performance,
    PerformanceResult,
)
from Magpie.remote import tasks as remote_tasks


def performance(backend=PerfBackend.NONE, **kwargs):
    perf = PerformanceConfig(backend=backend, **kwargs)
    cfg = PipelineConfig(
        kernel_type=KernelType.HIP,
        gpu_arch="gfx942",
        performance_config=perf,
    )
    return Performance(cfg)


def test_performance_result_models_and_summaries():
    metric = MetricResult("FLOPS", 12, "TF/s", peak=20, pct_of_peak=60)
    kernels = [
        KernelMetrics("gemm", 0, duration_ns=10),
        KernelMetrics("gemm", 1, duration_ns=20),
        KernelMetrics("other", 2, duration_ns=5),
        KernelMetrics("__amd_rocclr_internal", 3, duration_ns=100),
        KernelMetrics("no-duration", 4),
    ]
    result = PerformanceResult(True, metrics=[metric], kernel_metrics=kernels)
    assert metric.to_dict() == {
        "name": "FLOPS",
        "value": 12,
        "unit": "TF/s",
        "peak": 20,
        "pct_of_peak": 60,
    }
    assert kernels[0].to_dict()["kernel_name"] == "gemm"
    data = result.to_dict()
    assert data["summary"]["FLOPS"]["peak"] == 20
    assert data["kernels"][0]["kernel_name"] == "gemm"
    assert data["kernels"][0]["dispatch_count"] == 2


def test_performance_run_dispatch_and_disabled(kernel_config, monkeypatch):
    runner = performance(PerfBackend.NCU)
    runner.perf_cfg.enabled = False
    assert runner.run(None, kernel_config) is None
    runner.perf_cfg.enabled = True
    no_commands = KernelEvalConfig()
    assert runner.run(None, no_commands) is None
    custom = KernelEvalConfig(prof_command=["profile"])
    monkeypatch.setattr(runner, "_run_custom_profiler", lambda _: "custom")
    assert runner.run(None, custom) == "custom"
    for backend, method, marker in [
        (PerfBackend.NCU, "_run_ncu_on_testcase", "ncu"),
        (PerfBackend.ROCPROF_COMPUTE, "_run_rocprof_compute_on_testcase", "rocprof"),
        (PerfBackend.METRIX, "_run_metrix_on_testcase", "metrix"),
    ]:
        runner.perf_cfg.backend = backend
        monkeypatch.setattr(runner, method, lambda _, value=marker: value)
        assert runner.run(None, kernel_config) == marker
    runner.perf_cfg.backend = PerfBackend.NONE
    assert runner.run(None, kernel_config) is None
    runner.perf_cfg.backend = PerfBackend.NCU
    monkeypatch.setattr(
        runner,
        "_run_ncu_on_testcase",
        lambda _: (_ for _ in ()).throw(ValueError("bad")),
    )
    assert runner.run(None, kernel_config).errors == "bad"


def test_custom_profiler_success_failure_timeout_exception(monkeypatch):
    runner = performance()
    kernel = KernelEvalConfig(prof_command=[["one"], ["two"]])
    monkeypatch.setattr(
        "Magpie.eval.performance.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="ok"),
    )
    assert runner._run_custom_profiler(kernel).success is True
    monkeypatch.setattr(
        "Magpie.eval.performance.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd, 2, stdout="", stderr="failed"
        ),
    )
    assert "1/2 failed" in runner._run_custom_profiler(kernel).errors
    monkeypatch.setattr(
        "Magpie.eval.performance.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("x", 1)),
    )
    assert "timed out" in runner._run_custom_profiler(kernel).errors
    monkeypatch.setattr(
        "Magpie.eval.performance.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(OSError("spawn")),
    )
    assert runner._run_custom_profiler(kernel).errors == "spawn"


def test_rocprof_csv_parsers(tmp_path):
    runner = performance(PerfBackend.ROCPROF_COMPUTE)
    pmc = tmp_path / "pmc_perf.csv"
    pmc.write_text(
        "Kernel_Name,Dispatch_ID,Start_Timestamp,End_Timestamp,SQ_INSTS,TEXT\n"
        "gemm,2,10,30,44,nope\n"
        "bad,bad,x,y,bad,nope\n"
    )
    kernels = runner._parse_pmc_perf_csv(pmc)
    assert kernels[0].duration_ns == 20
    assert kernels[0].metrics[0].value == 44
    assert kernels[1].dispatch_id == 0

    top = tmp_path / "pmc_kernel_top.csv"
    top.write_text("KernelName,DispatchID,Duration,Counter,Text\n" "gemm,1,30,7,nope\n")
    parsed = runner._parse_kernel_top_csv(top)
    assert parsed[0].duration_ns == 30
    assert any(metric.name == "Counter" for metric in parsed[0].metrics)

    table = tmp_path / "2.1_Speed-of-Light.csv"
    table.write_text(
        "Metric,Avg,Unit,Peak,Pct of Peak\n"
        "VALU Utilization,50,%,100,50\n"
        "missing,,%,,\n"
    )
    metrics = runner._parse_metric_table_csv(table)
    assert metrics[0].name == "VALU_Util"
    assert metrics[0].peak == 100
    assert metrics[0].pct_of_peak == 50
    assert runner._parse_speed_of_light_csv(tmp_path)

    instruction = tmp_path / "10.1_Overall_Instruction_Mix.csv"
    instruction.write_text("Metric,Avg,Unit\nVALU,10,count\n")
    memory = tmp_path / "12.2_LDS_Stats.csv"
    memory.write_text("Metric,Avg,Unit\nLDS,20,count\n")
    assert runner._parse_instruction_mix_csv(tmp_path)[0].name == "Inst_VALU"
    assert runner._parse_memory_metrics_csv(tmp_path)[0].name == "Inst_LDS"
    summary, workload_kernels = runner._parse_rocprof_workload(tmp_path)
    assert summary
    assert workload_kernels


def test_ncu_profile_and_parser(monkeypatch, kernel_config):
    runner = performance(
        PerfBackend.NCU,
        ncu_config=NcuConfig(
            metrics=["sm__cycles"], kernel_filter="gemm", args=["--csv"]
        ),
    )
    monkeypatch.setattr("Magpie.eval.performance.shutil.which", lambda _: None)
    assert "not found" in runner._run_ncu_on_testcase(kernel_config).errors
    monkeypatch.setattr("Magpie.eval.performance.shutil.which", lambda _: "/bin/ncu")
    assert "No testcase" in runner._run_ncu_on_testcase(KernelEvalConfig()).errors
    monkeypatch.setattr(
        "Magpie.eval.performance.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd, 0, stdout="sm__cycles cycle 1,234\n"
        ),
    )
    result = runner._run_ncu_on_testcase(kernel_config)
    assert result.success is True
    assert result.metrics[0].value == 1234
    monkeypatch.setattr(
        "Magpie.eval.performance.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd, 2, stdout="", stderr="ncu failed"
        ),
    )
    assert "ncu failed" in runner._run_ncu_on_testcase(kernel_config).errors
    assert runner._parse_ncu_output("not metrics")[0].name == "profiling_complete"


def test_metrix_profile_and_json_parser(tmp_path, monkeypatch, kernel_config):
    config = MetrixConfig(output_dir=str(tmp_path), profile="quick")
    runner = performance(PerfBackend.METRIX, metrix_config=config)
    monkeypatch.setattr("Magpie.eval.performance.shutil.which", lambda _: None)
    assert "not found" in runner._run_metrix_on_testcase(kernel_config).errors
    monkeypatch.setattr("Magpie.eval.performance.shutil.which", lambda _: "/bin/metrix")
    assert "No testcase" in runner._run_metrix_on_testcase(KernelEvalConfig()).errors
    monkeypatch.setattr(
        "Magpie.eval.performance.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="profiled"),
    )
    assert runner._run_metrix_on_testcase(kernel_config).success is True

    data = {
        "gemm": {
            "duration_us": {"avg": 2},
            "metrics": {
                "memory.l2_hit_rate": {"avg": 80},
                "ignored": {"avg": None},
            },
        },
        "gemm-2": {"metrics": {"memory.l2_hit_rate": {"avg": 100}}},
    }
    path = tmp_path / "metrix.json"
    path.write_text(json.dumps(data))
    metrics, kernels = runner._parse_metrix_json_output(path)
    assert metrics[0].name == "L2_HitRate"
    assert metrics[0].value == 90
    assert kernels[0].duration_ns == 2000
    missing_metrics, missing_kernels = runner._parse_metrix_json_output(
        tmp_path / "missing"
    )
    assert missing_metrics == missing_kernels == []


def test_rocprof_profile_success_and_failures(tmp_path, monkeypatch, kernel_config):
    rocprof = RocprofComputeConfig(output_dir=str(tmp_path))
    runner = performance(PerfBackend.ROCPROF_COMPUTE, rocprof_config=rocprof)
    monkeypatch.setattr("Magpie.eval.performance.shutil.which", lambda _: None)
    assert "not found" in runner._run_rocprof_compute_on_testcase(kernel_config).errors
    monkeypatch.setattr(
        "Magpie.eval.performance.shutil.which", lambda _: "/bin/rocprof"
    )
    assert (
        "No testcase"
        in runner._run_rocprof_compute_on_testcase(KernelEvalConfig()).errors
    )
    workload = tmp_path / "rocprof_compute_output"

    def run(cmd, **kwargs):
        workload.mkdir(exist_ok=True)
        (workload / "pmc_perf.csv").write_text(
            "Kernel_Name,Dispatch_ID,Start_Timestamp,End_Timestamp\n" "gemm,0,0,10\n"
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="ok")

    monkeypatch.setattr("Magpie.eval.performance.subprocess.run", run)
    result = runner._run_rocprof_compute_on_testcase(kernel_config)
    assert result.success is True
    assert result.kernel_metrics[0].kernel_name == "gemm"
    monkeypatch.setattr(
        "Magpie.eval.performance.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 2, stderr="profile bad"),
    )
    assert (
        "profile failed"
        in runner._run_rocprof_compute_on_testcase(kernel_config).errors
    )


def test_evaluation_state_and_pipeline(kernel_config, monkeypatch):
    state = EvaluationState(
        compiling_result=CompilingResult(True),
        correctness_result=CorrectnessResult(True),
        performance_result=PerformanceResult(True),
        score=1,
        extra={"x": 1},
    )
    restored = EvaluationState.from_dict(state.to_dict())
    assert restored.score == 1
    assert restored.compiling_result.success is True
    assert restored.correctness_result.success is True
    assert restored.performance_result.success is True

    evaluator = Evaluator(PipelineConfig(kernel_type=KernelType.HIP, gpu_arch="gfx942"))
    monkeypatch.setattr(evaluator.compiling, "run", lambda _: None)
    monkeypatch.setattr(
        evaluator.correctness, "run", lambda *a: CorrectnessResult(True)
    )
    monkeypatch.setattr(evaluator.performance, "run", lambda *a: None)
    result = evaluator.evaluate(kernel_config)
    assert result.score == 1
    assert result.compiling_state is BaseKind.SKIPPED
    assert result.performance_state is BaseKind.SKIPPED


@pytest.mark.parametrize(
    ("stage", "error_prefix"),
    [
        ("compiling", "Compilation error"),
        ("correctness", "Correctness error"),
        ("performance", "Performance error"),
    ],
)
def test_evaluator_stage_exceptions(kernel_config, stage, error_prefix, monkeypatch):
    evaluator = Evaluator(PipelineConfig(kernel_type=KernelType.HIP, gpu_arch="gfx942"))
    monkeypatch.setattr(
        getattr(evaluator, stage),
        "run",
        lambda *a: (_ for _ in ()).throw(ValueError("broken")),
    )
    state = EvaluationState()
    method = {
        "compiling": evaluator._compile,
        "correctness": evaluator._check_correctness,
        "performance": evaluator._check_performance,
    }[stage]
    assert error_prefix in method(state, kernel_config).errors[0]


def test_evaluator_failed_results_and_scores(kernel_config, monkeypatch):
    evaluator = Evaluator(PipelineConfig(kernel_type=KernelType.HIP, gpu_arch="gfx942"))
    monkeypatch.setattr(
        evaluator.compiling, "run", lambda _: CompilingResult(False, errors="compile")
    )
    assert evaluator.evaluate(kernel_config).score == 0
    state = EvaluationState(compiling_state=BaseKind.SUCCESS)
    monkeypatch.setattr(
        evaluator.correctness,
        "run",
        lambda *a: CorrectnessResult(False, errors="wrong"),
    )
    assert (
        evaluator._check_correctness(state, kernel_config).correctness_state
        is BaseKind.FAILED
    )
    monkeypatch.setattr(
        evaluator.performance, "run", lambda *a: PerformanceResult(False, errors="slow")
    )
    assert (
        evaluator._check_performance(state, kernel_config).performance_state
        is BaseKind.FAILED
    )
    state.compiling_state = BaseKind.FAILED
    assert evaluator._calculate_score(state).score == 0


def test_remote_task_helpers_and_dispatch(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    remote_tasks._setup_env({"shared_storage_path": "/shared"})
    assert os.environ["HF_HOME"] == "/shared/hf_cache"
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    remote_tasks._clear_hidden_gpus()
    assert "HIP_VISIBLE_DEVICES" not in os.environ
    envs = {}
    remote_tasks._ensure_extra_arg(envs, "EXTRA", "--flag value")
    remote_tasks._ensure_extra_arg(envs, "EXTRA", "--flag other")
    assert envs["EXTRA"] == "--flag value"
    assert remote_tasks._extra_args_key("vllm") == "EXTRA_VLLM_ARGS"
    assert remote_tasks._extra_args_key("custom") == "EXTRA_CUSTOM_ARGS"

    monkeypatch.setitem(
        remote_tasks._MODE_RUNNERS,
        "unit",
        lambda mode, ray, task: {"task_id": task, "status": "completed"},
    )
    result = remote_tasks.run_task({"task_id": "t", "mode_type": "unit"})
    assert result["status"] == "completed"
    assert "execution_time" in result
    assert "Unsupported" in remote_tasks.run_task({"mode_type": "missing"})["error"]


@pytest.mark.parametrize(
    ("framework", "tp", "local", "expected"),
    [
        ("vllm", 2, 8, "--distributed-executor-backend mp"),
        ("vllm", 16, 8, "--distributed-executor-backend ray"),
        ("sglang", 16, 8, "--use-ray --nnodes 2"),
    ],
)
def test_remote_tp_isolation(framework, tp, local, expected, monkeypatch):
    mode = {"benchmark_config": {"framework": framework, "envs": {"TP": tp}}}
    monkeypatch.setattr(remote_tasks, "_get_local_gpu_count", lambda: local)
    monkeypatch.setenv("RAY_ADDRESS", "auto")
    remote_tasks._configure_tp_isolation(mode, {})
    key = remote_tasks._extra_args_key(framework)
    assert expected in mode["benchmark_config"]["envs"][key]


def test_remote_gpu_count_and_dispatch_failure(monkeypatch):
    ray = SimpleNamespace(
        get_runtime_context=lambda: SimpleNamespace(get_node_id=lambda: "node"),
        nodes=lambda: [{"NodeID": "node", "Alive": True, "Resources": {"GPU": 4}}],
    )
    monkeypatch.setitem(__import__("sys").modules, "ray", ray)
    assert remote_tasks._get_local_gpu_count() == 4
    ray.nodes = lambda: (_ for _ in ()).throw(RuntimeError("ray unavailable"))
    assert remote_tasks._get_local_gpu_count() == 8

    monkeypatch.setitem(
        remote_tasks._MODE_RUNNERS,
        "broken",
        lambda *args: (_ for _ in ()).throw(ValueError("worker broke")),
    )
    result = remote_tasks.run_task({"task_id": "broken", "mode_type": "broken"})
    assert result["status"] == "failed"
    assert result["error"] == "worker broke"


@pytest.mark.parametrize("framework", ["atom", "unknown"])
def test_remote_tp_isolation_skips_unsupported_multinode(framework, monkeypatch):
    mode = {"benchmark_config": {"framework": framework, "envs": {"TP": 16}}}
    monkeypatch.setattr(remote_tasks, "_get_local_gpu_count", lambda: 8)
    remote_tasks._configure_tp_isolation(mode, {})
    assert (
        remote_tasks._extra_args_key(framework) not in mode["benchmark_config"]["envs"]
    )


def test_remote_benchmark_runner_sets_shared_paths(monkeypatch):
    class Benchmarker:
        def __init__(self, config, output_dir):
            self.config = config
            self.output_dir = output_dir

        def run(self, task_id):
            return SimpleNamespace(
                to_dict=lambda: {
                    "task_id": task_id,
                    "run_mode": self.config.run_mode,
                    "inferencex": self.config.inferencex_path,
                    "cache": self.config.hf_cache_path,
                    "output_dir": self.output_dir,
                }
            )

    monkeypatch.setattr("Magpie.modes.benchmark.BenchmarkMode", Benchmarker)
    mode = {
        "benchmark_config": {
            "framework": "vllm",
            "model": "demo",
            "run_mode": "ray",
        }
    }
    result = remote_tasks._run_benchmark(
        mode, {"shared_storage_path": "/shared"}, "task"
    )
    assert result["run_mode"] == "local"
    assert result["inferencex"] == "/shared/InferenceX"
    assert result["cache"] == "/shared/hf_cache"
    assert result["output_dir"] == "/shared/results/task"


def test_remote_analyze_and_compare_runners(monkeypatch, kernel_config):
    class Analyzer:
        def __init__(self, config):
            self.config = config

        def analyze(self, kernel):
            return SimpleNamespace(to_dict=lambda: {"kernel": kernel.kernel_id})

    class Comparator:
        def __init__(self, config):
            self.config = config

        def compare(self, kernels):
            return SimpleNamespace(to_dict=lambda: {"count": len(kernels)})

    monkeypatch.setattr("Magpie.modes.AnalyzeMode", Analyzer)
    monkeypatch.setattr("Magpie.modes.CompareMode", Comparator)
    mode = {
        "kernel_configs": [kernel_config.to_dict(), "ignored"],
        "gpu_arch": "gfx942",
        "compare_config": {"winner_strategy": "correctness"},
    }
    analyzed = remote_tasks._run_analyze(mode, {}, "analyze")
    compared = remote_tasks._run_compare(mode, {}, "compare")
    assert analyzed["results"] == [{"kernel": "unit-kernel"}]
    assert compared["results"] == {"count": 1}
