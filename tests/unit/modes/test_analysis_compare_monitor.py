import subprocess
from types import SimpleNamespace

import pytest

from Magpie.config import KernelEvalConfig, KernelType
from Magpie.eval import BaseKind, EvaluationState
from Magpie.eval.correctness import CorrectnessResult
from Magpie.eval.performance import KernelMetrics, MetricResult, PerformanceResult
from Magpie.modes.analyze_eval.analyzer import AnalyzeConfig, AnalyzeMode
from Magpie.modes.compare_eval.comparator import CompareConfig, CompareMode
from Magpie.utils.gpu_monitor import GPUMonitor, GPUMonitorStats, GPUSample


def successful_state(metric=80, duration=10):
    return EvaluationState(
        correctness_state=BaseKind.SUCCESS,
        correctness_result=CorrectnessResult(True),
        performance_state=BaseKind.SUCCESS,
        performance_result=PerformanceResult(
            True,
            metrics=[MetricResult("util", metric, "%", pct_of_peak=metric)],
            kernel_metrics=[KernelMetrics("gemm", 0, duration_ns=duration)],
        ),
        score=1,
    )


def test_analyze_mode_validation_config_and_batch(monkeypatch, kernel_config):
    monkeypatch.setattr("Magpie.utils.detect_gpu", lambda: ("amd", "gfx942"))
    config = AnalyzeConfig(
        gpu_arch=None,
        rocprof_config={"metric_blocks": ["1"]},
        ncu_config={"metrics": ["cycles"]},
        metrix_config={"backend": "metrix", "profile": "quick"},
        correctness_config={"backend": "testcase"},
    )
    assert config.gpu_arch == "gfx942"
    mode = AnalyzeMode(config)
    assert "requires testcase" in mode.analyze(KernelEvalConfig()).errors[0]
    seen = []

    class FakeEvaluator:
        def __init__(self, pipeline):
            seen.append(pipeline)

        def evaluate(self, kernel):
            return successful_state()

    monkeypatch.setattr("Magpie.modes.analyze_eval.analyzer.Evaluator", FakeEvaluator)
    result = mode.analyze(kernel_config)
    assert result.score == 1
    assert seen[0].performance_config.get_backend().name == "METRIX"
    assert mode.analyze_batch([]) == []
    assert len(mode.analyze_batch([kernel_config, kernel_config])) == 2


def test_compare_mode_evaluation_scoring_and_summary(monkeypatch, kernel_config):
    configs = [
        kernel_config,
        KernelEvalConfig(**{**kernel_config.__dict__, "kernel_id": "two"}),
    ]
    states = [successful_state(80, 10), successful_state(40, 20)]

    class FakeEvaluator:
        def __init__(self, _pipeline):
            pass

        def evaluate(self, _kernel):
            return states.pop(0)

    config = CompareConfig(
        gpu_arch="gfx942",
        perf_weights_rocprof={"util": 0.5, "duration_ns_total": 0.5},
        rocprof_config={"metric_blocks": ["1"]},
        ncu_config={"metrics": ["cycles"]},
        metrix_config={"profile": "quick"},
    )
    mode = CompareMode(config)
    monkeypatch.setattr("Magpie.modes.compare_eval.comparator.Evaluator", FakeEvaluator)
    comparison = mode.compare(configs)
    assert comparison.winner == 0
    assert comparison.rankings[0][0] == 0
    assert "Correctness: 2/2 passed" in comparison.summary
    assert comparison.to_dict()["winner"] == 0
    with pytest.raises(ValueError, match="at least 2"):
        mode.compare([kernel_config])


def test_compare_weights_values_and_fallbacks(kernel_config):
    config = CompareConfig(
        gpu_arch="sm_90",
        perf_weights_ncu={"util": 1},
        perf_weights_rocprof={"duration_ns_total": 1},
        perf_weights_metrix={"L2": 1},
        metrix_config={"backend": "metrix"},
    )
    mode = CompareMode(config)
    assert mode._get_perf_weights(KernelType.HIP) == {"L2": 1}
    config.metrix_config = {}
    assert mode._get_perf_weights(KernelType.CUDA) == {"util": 1}
    state = successful_state(75, 12)
    assert mode._get_perf_value(state, "util") == 75
    assert mode._get_perf_value(state, "duration_ns_total") == 12
    assert mode._get_perf_value(state, "missing") is None
    assert mode._get_perf_value(EvaluationState(), "util") is None

    failed = EvaluationState(
        correctness_state=BaseKind.FAILED,
        correctness_result=CorrectnessResult(False),
    )
    scores = mode._compute_perf_scores([state, failed], [True, False], KernelType.CUDA)
    assert scores == [1.0, None]
    config.perf_weights_ncu = {}
    config.perf_weights_rocprof = {}
    assert mode._compute_perf_scores([state], [True], KernelType.HIP) == [None]


def test_compare_correctness_first_fallback(kernel_config):
    mode = CompareMode(
        CompareConfig(gpu_arch="gfx942", winner_strategy="correctness_first")
    )
    states = [
        EvaluationState(correctness_state=BaseKind.FAILED),
        successful_state(),
    ]
    configs = [kernel_config, KernelEvalConfig(kernel_id="second")]
    comparison = mode._build_comparison(states, configs)
    assert comparison.winner == 1


def test_gpu_monitor_vendor_detection_and_start(monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "rocm-smi":
            return subprocess.CompletedProcess(cmd, 0, stdout="GPU[0]")
        return subprocess.CompletedProcess(cmd, 1, stdout="")

    monkeypatch.setattr("Magpie.utils.gpu_monitor.subprocess.run", run)
    monitor = GPUMonitor(interval_sec=0.1)
    assert monitor.vendor == "amd"
    assert monitor.interval == 0.5
    monkeypatch.setattr(
        "Magpie.utils.gpu_monitor.threading.Thread",
        lambda **kwargs: SimpleNamespace(start=lambda: None, is_alive=lambda: False),
    )
    assert monitor.start() is True
    assert monitor.start() is True
    assert monitor.is_running is True
    assert monitor.stop().sample_count == 0
    assert GPUMonitor(vendor="unknown").start() is False


def test_gpu_monitor_collects_amd_and_nvidia_samples(monkeypatch):
    monitor = GPUMonitor(vendor="amd")
    amd_output = (
        "Temperature junction 55.5\n"
        "sclk 1200 Mhz\n"
        "mclk 900 Mhz\n"
        "Current Graphics Power: 220.5\n"
    )
    monkeypatch.setattr(
        "Magpie.utils.gpu_monitor.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout=amd_output),
    )
    sample = monitor._collect_sample()
    assert sample.temperature_c == 55.5
    assert sample.gpu_clock_mhz == 1200
    assert sample.mem_clock_mhz == 900
    assert sample.power_watts == 220.5

    monitor.vendor = "nvidia"
    monkeypatch.setattr(
        "Magpie.utils.gpu_monitor.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd, 0, stdout="60, 1500, 1000, 250\n"
        ),
    )
    sample = monitor._collect_sample()
    assert sample.temperature_c == 60
    assert sample.power_watts == 250
    monitor.vendor = "other"
    assert monitor._collect_sample() is None


@pytest.mark.parametrize("vendor", ["amd", "nvidia"])
def test_gpu_monitor_command_failures(vendor, monkeypatch):
    monitor = GPUMonitor(vendor=vendor)
    monkeypatch.setattr(
        "Magpie.utils.gpu_monitor.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 2, stdout=""),
    )
    assert monitor._collect_sample() is None
    monkeypatch.setattr(
        "Magpie.utils.gpu_monitor.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("gpu", 5)),
    )
    assert monitor._collect_sample() is None


def test_gpu_monitor_statistics_and_properties():
    monitor = GPUMonitor(vendor="amd")
    monitor._start_time = 1
    monitor._samples = [
        GPUSample(2, 40, 1000, 800, 200),
        GPUSample(3, 60, 1200, 1000, 240),
    ]
    stats = monitor._compute_stats()
    assert stats.sample_count == 2
    assert stats.duration_sec == 2
    assert stats.temp_avg == 50
    assert stats.gpu_clock_avg == 1100
    assert stats.mem_clock_avg == 900
    assert stats.power_avg == 220
    assert stats.to_dict()["temperature_c"]["max"] == 60
    assert monitor.sample_count == 2
    assert GPUMonitorStats().to_dict()["sample_count"] == 0
