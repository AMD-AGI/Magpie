import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from Magpie import main
from Magpie.config import KernelEvalConfig, KernelType
from Magpie.core import EnvironmentType
from Magpie.eval import BaseKind, EvaluationState


def args(**changes):
    values = {
        "environment": None,
        "workers": None,
        "docker_image": None,
    }
    values.update(changes)
    return argparse.Namespace(**values)


def test_yaml_kernel_loading_and_environment_expansion(monkeypatch, tmp_path):
    monkeypatch.setenv("MAGPIE_TEST_ROOT", str(tmp_path))
    path = tmp_path / "kernels.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "kernel": {
                    "id": "first",
                    "type": "torch",
                    "source_files": ["$MAGPIE_TEST_ROOT/kernel.py"],
                    "working_dir": "$MAGPIE_TEST_ROOT",
                    "env": {"ROOT": "$MAGPIE_TEST_ROOT"},
                    "testcase_command": "python test.py",
                    "compile_command": [["make", "clean"], ["make"]],
                },
                "kernels": [{"id": "second", "type": "hip"}],
                "performance": {"backend": "$MAGPIE_BACKEND"},
                "correctness": {"backend": "testcase"},
                "ray_config": {"cluster_address": "ray://host"},
            }
        )
    )
    monkeypatch.setenv("MAGPIE_BACKEND", "metrix")
    configs, perf, corr, sched = main.load_kernel_config(path)
    assert [cfg.kernel_id for cfg in configs] == ["first", "second"]
    assert configs[0].testcase_command == ["python", "test.py"]
    assert configs[0].compiling_command == [["make", "clean"], ["make"]]
    assert perf["backend"] == "metrix"
    assert corr["backend"] == "testcase"
    assert sched["environment"] == "ray"
    assert main.load_yaml(tmp_path / "missing") == {}


def test_setup_logging_uses_config_and_verbose(monkeypatch):
    calls = []
    monkeypatch.setattr(
        main.logging, "basicConfig", lambda **kwargs: calls.append(kwargs)
    )
    main.setup_logging({"logging": {"level": "warning"}})
    assert calls[-1]["level"] == main.logging.WARNING
    main.setup_logging({}, verbose=True)
    assert calls[-1]["level"] == main.logging.DEBUG


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        (None, None),
        ([], None),
        ("make build", ["make", "build"]),
        (["make", "build"], ["make", "build"]),
        ([["make"], ["test"]], [["make"], ["test"]]),
        (["make", 1], None),
        ({"command": "make"}, None),
    ],
)
def test_parse_command_list(entry, expected):
    assert main._parse_command_list(entry) == expected


def test_kernel_types_and_empty_entry():
    assert main.parse_kernel_type("PY") is KernelType.PYTORCH
    assert main.parse_kernel_type("triton") is KernelType.TRITON
    with pytest.raises(ValueError, match="Unsupported kernel type"):
        main.parse_kernel_type("metal")
    assert main._parse_kernel_entry({}) is None


def test_framework_settings_and_overrides():
    config = {
        "compiling": {"enable_default_compile": True},
        "correctness": {"backend": "accordo", "accordo": {"atol": 0.1}},
        "performance": {
            "timeout_seconds": 12,
            "backend": "metrix",
            "rocprof_compute": {"roofline": False, "profile_args": "--foo"},
            "metrix": {"profile": "quick", "num_replays": 2},
            "ncu": {"args": ["--set", "full"], "metrics": ["cycles"]},
        },
        "compare": {"winner_strategy": "weighted"},
    }
    assert main._get_compiling_config(config)["enable_default_compile"] is True
    correctness = main._get_correctness_config(config)
    assert correctness["backend"] == "accordo"
    assert correctness["accordo"]["atol"] == 0.1
    correctness = main._apply_correctness_overrides(
        correctness, {"backend": "testcase", "accordo": {"rtol": 0.2}}
    )
    assert correctness["backend"] == "testcase"
    assert correctness["accordo"]["rtol"] == 0.2

    hip = main._get_performance_config(config, KernelType.HIP)
    assert hip["rocprof_config"]["profile_args"] == ["--foo", "--no-roof"]
    cuda = main._get_performance_config(config, KernelType.CUDA)
    assert cuda["profiler_args"] == ["--set", "full"]
    triton = main._get_performance_config(config, KernelType.TRITON)
    assert triton["rocprof_config"] and triton["ncu_config"]
    assert main._get_compare_config(config)["winner_strategy"] == "weighted"

    merged = main._apply_perf_overrides(
        hip,
        {
            "backend": "metrix",
            "timeout_seconds": "33",
            "metrix": {"metrics": ["L2"]},
            "rocprof_compute": {"roofline": True},
            "ncu": {"metrics": ["dram"]},
        },
        KernelType.HIP,
    )
    assert merged["timeout_seconds"] == 33
    assert merged["metrix_config"]["metrics"] == ["L2"]
    assert "--no-roof" not in merged["rocprof_config"]["profile_args"]
    non_metrix = main._apply_perf_overrides({}, {"backend": "ncu"}, KernelType.CUDA)
    assert non_metrix["metrix_config"]["backend"] is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (" YES ", True), ("off", False), (1, True), (0, False)],
)
def test_coerce_bool(value, expected):
    assert main._coerce_bool(value) is expected


def test_scheduler_config_precedence():
    config = {
        "scheduler": {
            "environment": "docker",
            "max_workers": 2,
            "gpu_devices": [1],
            "docker_image": "base",
        }
    }
    scheduler = main._get_scheduler_config(
        config,
        args(environment="local", workers=4, docker_image="cli"),
        {
            "environment": "ray",
            "ray_config": {
                "cluster_address": "ray://host",
                "shared_storage_path": "/shared",
            },
        },
    )
    assert scheduler.environment_type is EnvironmentType.LOCAL
    assert scheduler.max_workers == 4
    assert scheduler.docker_image == "cli"
    assert scheduler.ray_cluster_address == "ray://host"


def test_state_print_workspace_and_serialization(tmp_path, capsys):
    state = main._dict_to_eval_state(
        {
            "compiling_state": "SUCCESS",
            "correctness_state": "FAILED",
            "performance_state": "SKIPPED",
            "score": 0.5,
            "errors": ["bad"],
            "extra": {"key": "value"},
        }
    )
    assert state.correctness_state is BaseKind.FAILED
    config = KernelEvalConfig(kernel_id="demo", kernel_type=KernelType.HIP)
    main._print_result(config, state)
    main._print_result(config, {"score": 1.0, "errors": ["warning"]})
    assert "Kernel: demo" in capsys.readouterr().out

    workspace = main._create_workspace(tmp_path, "analyze", "unsafe label!")
    main._save_config_snapshot(workspace, [config, "raw"], {"extra": True})
    assert yaml.safe_load((workspace / "config.yaml").read_text())["extra"] is True
    main._save_results([state, {"raw": True}, "plain"], workspace, "analyze")
    report = next(workspace.glob("analyze_report.json"))
    assert len(json.loads(report.read_text())["results"]) == 3

    main._save_comparison(
        {"winner": 0, "summary": "first", "rankings": [[0, 1.0]]}, workspace
    )
    assert (
        json.loads((workspace / "compare_report.json").read_text())["results"]["winner"]
        == 0
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
        return self.response

    def run_compare(self, **kwargs):
        return self.response

    def shutdown(self):
        self.shutdown_called = True


def test_run_analyze_success_and_failures(monkeypatch, tmp_path):
    source = tmp_path / "kernel.hip"
    source.write_text("kernel")
    cli = args(
        kernel_config=None,
        kernels=[source],
        testcase="./test.sh",
        compile_cmd=None,
        type="hip",
        output_dir=tmp_path / "out",
        no_perf=False,
    )
    state = EvaluationState(
        compiling_state=BaseKind.SUCCESS,
        correctness_state=BaseKind.SUCCESS,
        performance_state=BaseKind.SUCCESS,
    )
    FakeScheduler.instances.clear()
    FakeScheduler.initialized = True
    FakeScheduler.response = SimpleNamespace(success=True, results=[state], errors=[])
    monkeypatch.setattr(main, "Scheduler", FakeScheduler)
    assert main.run_analyze(cli, {}) == 0
    assert FakeScheduler.instances[-1].shutdown_called is True

    cli.testcase = None
    assert main.run_analyze(cli, {}) == 1
    cli.kernels = None
    assert main.run_analyze(cli, {}) == 1


def test_run_analyze_kernel_config_validation(tmp_path):
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("kernel:\n  id: broken\n  type: unsupported\n")
    cli = args(
        kernel_config=invalid,
        kernels=None,
        testcase=None,
        compile_cmd=None,
        type="hip",
        output_dir=tmp_path / "out",
        no_perf=False,
    )
    assert main.run_analyze(cli, {}) == 1

    empty = tmp_path / "empty.yaml"
    empty.write_text("kernels: []\n")
    cli.kernel_config = empty
    assert main.run_analyze(cli, {}) == 1


def test_run_compare_success_and_validation(monkeypatch, tmp_path):
    sources = [tmp_path / "one.hip", tmp_path / "two.hip"]
    for source in sources:
        source.write_text("kernel")
    cli = args(
        kernel_config=None,
        kernels=sources,
        testcase="./test.sh",
        type="hip",
        output_dir=tmp_path / "out",
        no_perf=False,
        baseline=0,
    )
    FakeScheduler.instances.clear()
    FakeScheduler.initialized = True
    FakeScheduler.response = SimpleNamespace(
        success=True,
        results={"winner": 0, "summary": "winner", "rankings": []},
        errors=[],
    )
    monkeypatch.setattr(main, "Scheduler", FakeScheduler)
    assert main.run_compare(cli, {}) == 0
    assert FakeScheduler.instances[-1].shutdown_called is True
    cli.kernels = sources[:1]
    assert main.run_compare(cli, {}) == 1


def test_parser_and_load_benchmark_config(tmp_path):
    parser = main.create_parser()
    parsed = parser.parse_args(["analyze", "kernel.hip", "--testcase", "./test.sh"])
    assert parsed.mode == "analyze"
    path = tmp_path / "benchmark.yaml"
    path.write_text("benchmark:\n  framework: vllm\n  model: demo\n")
    config = main.load_benchmark_config(path)
    assert config["framework"] == "vllm"


def test_standalone_gap_analysis_success_empty_and_missing(
    monkeypatch, tmp_path, capsys
):
    trace_dir = tmp_path / "run" / "torch_trace"
    trace_dir.mkdir(parents=True)
    kernel = SimpleNamespace(name="a" * 60, calls=2, total_duration_us=8, avg_us=4)

    class Result:
        errors = ["partial trace"]
        merged_kernels = [kernel]
        rank_results = [{}, {}]
        total_duration_us = 10

        def to_csv(self, path, **kwargs):
            path.write_text("csv")

        def to_rank_csv(self, path):
            (path / "rank.csv").write_text("csv")

    class Analyzer:
        empty = False

        def __init__(self, config):
            self.config = config

        def analyze(self, path):
            result = Result()
            if self.empty:
                result.merged_kernels = []
            return result

    monkeypatch.setattr("Magpie.modes.benchmark.gap_analysis.GapAnalyzer", Analyzer)
    cli = args(
        trace_dir=tmp_path / "run",
        start_pct=10,
        end_pct=90,
        top_k=5,
        min_duration_us=1,
        categories=["kernel"],
        ignore_categories=["annotation"],
        output_dir=tmp_path / "output",
        find_kernel_sources=True,
        kernel_source_repos=["repo"],
        no_rank_csv=False,
    )
    assert main.run_gap_analysis_standalone(cli) == 0
    output = capsys.readouterr().out
    assert "Ranks analyzed: 2" in output
    assert "partial trace" in output
    assert (tmp_path / "output" / "gap_analysis" / "gap_analysis.csv").exists()
    Analyzer.empty = True
    assert main.run_gap_analysis_standalone(cli) == 1
    cli.trace_dir = tmp_path / "missing"
    assert main.run_gap_analysis_standalone(cli) == 1


def benchmark_args(tmp_path, **changes):
    values = {
        "trace_dir": None,
        "benchmark_config": None,
        "framework": "vllm",
        "model": "demo",
        "precision": "fp8",
        "tp": 2,
        "concurrency": 4,
        "input_len": 16,
        "output_len": 8,
        "torch_profiler": False,
        "system_profiler": False,
        "docker_image": "image:test",
        "inferencex_path": "",
        "benchmark_script": None,
        "timeout": 30,
        "run_mode": "local",
        "output_dir": tmp_path,
    }
    values.update(changes)
    return args(**values)


def test_run_benchmark_cli_success_failure_and_validation(monkeypatch, tmp_path):
    created = []

    class Mode:
        success = True
        raises = False

        def __init__(self, config, output_dir):
            self.config = config
            created.append(self)

        def run(self):
            if self.raises:
                raise RuntimeError("runtime failed")
            return SimpleNamespace(
                success=self.success,
                workspace_dir="/results/run",
                errors=[] if self.success else ["failed"],
                get_summary=lambda: "summary",
            )

    monkeypatch.setattr("Magpie.modes.benchmark.BenchmarkMode", Mode)
    cli = benchmark_args(tmp_path)
    assert main.run_benchmark(cli, {"benchmark": {"inferencex_path": "/default"}}) == 0
    assert created[-1].config.inferencex_path == "/default"
    Mode.success = False
    assert main.run_benchmark(cli, {}) == 1
    Mode.raises = True
    assert main.run_benchmark(cli, {}) == 1
    Mode.raises = False

    cli.framework = None
    assert main.run_benchmark(cli, {}) == 1
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("benchmark: {}\n")
    cli.benchmark_config = config_path
    assert main.run_benchmark(cli, {}) == 1


def test_main_dispatch_gpu_info_help_and_modes(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        main,
        "get_gpu_info",
        lambda: {
            "vendor": "amd",
            "architecture": "gfx942",
            "detected": True,
            "compiler": "hipcc",
            "profiler": "rocprof",
        },
    )
    monkeypatch.setattr(
        main,
        "create_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: args(gpu_info=True), print_help=lambda: None
        ),
    )
    assert main.main() == 0
    assert "gfx942" in capsys.readouterr().out

    monkeypatch.setattr(
        main,
        "create_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: args(gpu_info=False, mode=None), print_help=lambda: None
        ),
    )
    assert main.main() == 0

    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}")
    for mode_name, function_name, expected in [
        ("analyze", "run_analyze", 2),
        ("compare", "run_compare", 3),
        ("benchmark", "run_benchmark", 4),
    ]:
        monkeypatch.setattr(main, function_name, lambda *a, value=expected, **k: value)
        monkeypatch.setattr(
            main,
            "create_parser",
            lambda m=mode_name: SimpleNamespace(
                parse_args=lambda: args(
                    gpu_info=False, mode=m, config=config_path, verbose=False
                ),
                print_help=lambda: None,
            ),
        )
        assert main.main() == expected
    monkeypatch.setattr(
        main,
        "create_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: args(
                gpu_info=False, mode="unknown", config=config_path, verbose=False
            ),
            print_help=lambda: None,
        ),
    )
    assert main.main() == 1
