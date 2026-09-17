import json
import subprocess
from pathlib import Path

import pytest

from Magpie.modes.benchmark.image_selector import ImageSelector
from Magpie.modes.benchmark.inferencex import (
    InferenceXManager,
    _resolve_default_inferencex_dir,
    ensure_inferencex_available,
    get_manager,
)
from Magpie.modes.benchmark.result import (
    BenchmarkResult,
    KernelMetrics,
    LatencyMetrics,
    ResultParser,
    ThroughputMetrics,
)
from Magpie.modes.benchmark.workspace import WorkspaceManager
from Magpie.utils.gpu import GPUVendor


def test_benchmark_result_serialization_and_full_summary(tmp_path):
    result = BenchmarkResult(
        success=True,
        framework="vllm",
        model="demo",
        throughput=ThroughputMetrics(
            request_throughput=12.5,
            output_throughput=100,
            total_token_throughput=150,
            completed_requests=20,
            duration_seconds=2,
        ),
        latency=LatencyMetrics(
            ttft_mean=1,
            ttft_p99=2,
            tpot_mean=3,
            tpot_p99=4,
            itl_mean=5,
            itl_p99=6,
            e2el_mean=7,
            e2el_p99=8,
        ),
        kernel_summary=[KernelMetrics("gemm", 2.5, 100, 3)],
        top_bottlenecks=[f"kernel-{index}" for index in range(7)],
        tracelens_analysis={
            "output_files": [
                str(tmp_path / f"report-{index}.csv") for index in range(5)
            ],
            "errors": ["partial report"],
        },
        gap_analysis={
            "config": {
                "trace_start_pct": 10,
                "trace_end_pct": 90,
                "categories": ["gemm", "communication"],
            },
            "top_kernels": [
                {
                    "name": f"kernel-{index}",
                    "pct_total": 20,
                    "self_cuda_total_us": 5,
                    "calls": 2,
                }
                for index in range(7)
            ],
        },
        gpu_monitor={
            "sample_count": 2,
            "duration_sec": 1,
            "temperature_c": {"min": 40, "max": 50, "avg": 45},
            "gpu_clock_mhz": {"min": 1000, "max": 1200, "avg": 1100},
            "power_watts": {"min": 200, "max": 250, "avg": 225},
        },
        errors=["example warning"],
        workload_kind="diffusion",
        throughput_unit="img/s",
        quality_gate={"passed": True},
        latency_s=0.5,
    )

    serialized = result.to_dict()
    summary = result.get_summary()

    assert serialized["kernel_summary"] == [
        {"name": "gemm", "time_ms": 2.5, "percent": 100, "calls": 3}
    ]
    assert serialized["workload_kind"] == "diffusion"
    assert serialized["throughput_unit"] == "img/s"
    assert serialized["quality_gate"] == {"passed": True}
    assert serialized["latency_s"] == 0.5
    assert "Request throughput: 12.50 req/s" in summary
    assert "TTFT (mean/p99): 1.00ms / 2.00ms" in summary
    assert "Top Bottleneck Kernels:" in summary
    assert "... and 2 more" in summary
    assert "TraceLens Analysis:" in summary
    assert "Warning: partial report" in summary
    assert "Window: 10%-90%" in summary
    assert "Categories: gemm, communication" in summary
    assert "GPU Hardware Monitoring:" in summary
    assert "Temperature: 40.0°C - 50.0°C" in summary
    assert "Errors:" in summary


def test_result_parser_handles_missing_invalid_and_failed_quality_gate(tmp_path):
    missing = ResultParser.parse_inferencex_result(tmp_path / "missing.json")
    assert missing.success is False
    assert "Result file not found" in missing.errors[0]

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{")
    parsed = ResultParser.parse_inferencex_result(invalid)
    assert parsed.success is False
    assert "Failed to parse JSON" in parsed.errors[0]

    failed_gate = tmp_path / "failed-gate.json"
    failed_gate.write_text(
        json.dumps({"model_id": "demo", "quality_gate": {"passed": False}})
    )
    parsed = ResultParser.parse_inferencex_result(failed_gate)
    assert parsed.success is False
    assert parsed.model == "demo"
    assert "Quality gate failed" in parsed.errors[0]


def test_workspace_manager_lifecycle_and_result_collection(tmp_path):
    manager = WorkspaceManager(str(tmp_path), framework="vllm")
    assert manager.workspace_path is None
    assert manager.torch_trace_dir is None
    assert manager.system_profile_dir is None
    assert manager.get_result_file_path() is None
    assert manager.collect_results() == {}
    manager.save_report({"ignored": True})
    manager.save_summary("ignored")
    manager.cleanup()

    workspace = manager.create({"model": "demo"})
    assert workspace.is_absolute()
    assert manager.workspace_path == workspace
    assert manager.torch_trace_dir == workspace / "torch_trace"
    assert manager.system_profile_dir == workspace / "system_profile"
    assert manager.get_result_file_path("custom.json") == workspace / "custom.json"
    assert (workspace / "config.yaml").read_text().strip() == "model: demo"

    manager.save_report({"success": True})
    manager.save_summary("all good")
    (workspace / "inferencex_result.json").write_text('{"throughput": 12}')
    (workspace / "torch_trace" / "trace.json").write_text("{}")
    (workspace / "system_profile" / "metrics.csv").write_text("metric,value")
    (workspace / "server.log").write_text("ready")
    (workspace / "worker.pid").write_text("123")

    results = manager.collect_results()
    assert results["inferencex_result"] == {"throughput": 12}
    assert results["torch_trace_files"] == [
        str(workspace / "torch_trace" / "trace.json")
    ]
    assert results["system_profile_files"] == [
        str(workspace / "system_profile" / "metrics.csv")
    ]
    assert results["server_log"] == "ready"
    assert json.loads((workspace / "benchmark_report.json").read_text()) == {
        "success": True
    }
    assert (workspace / "summary.txt").read_text() == "all good"

    manager.cleanup(keep_results=True)
    assert not (workspace / "worker.pid").exists()
    assert workspace.exists()
    assert manager.workspace_path is None


def test_workspace_manager_removes_workspace_and_lists_only_benchmarks(tmp_path):
    first = WorkspaceManager(str(tmp_path), framework="atom")
    workspace = first.create()
    (tmp_path / "unrelated").mkdir()
    (tmp_path / "plain-file").write_text("x")

    listed = WorkspaceManager.list_workspaces(str(tmp_path))
    assert listed == [str(workspace)]
    assert WorkspaceManager.list_workspaces(str(tmp_path / "missing")) == []

    first.cleanup(keep_results=False)
    assert not workspace.exists()
    assert first.workspace_path is None


def test_workspace_manager_handles_file_io_failures(tmp_path, monkeypatch):
    manager = WorkspaceManager(str(tmp_path), framework="vllm")
    workspace = manager.create()
    (workspace / "inferencex_result.json").touch()
    (workspace / "server.log").touch()

    def fail_open(*_args, **_kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr("builtins.open", fail_open)
    manager._save_config_snapshot(workspace, {"model": "demo"})
    manager.save_report({"success": True})
    manager.save_summary("summary")
    results = manager.collect_results()

    assert results["inferencex_result"] is None
    assert results["server_log"] is None


def test_image_selector_loads_mapping_and_selects_images(tmp_path, monkeypatch):
    config = tmp_path / "images.yaml"
    config.write_text("vllm:\n  gfx942: example/vllm:latest\n")
    selector = ImageSelector(str(config))

    assert selector.select_image("VLLM", "gfx942") == "example/vllm:latest"
    assert (
        selector.select_image("missing", override_image="override:image")
        == "override:image"
    )
    assert selector.list_available_images() == {
        "vllm": {"gfx942": "example/vllm:latest"}
    }
    assert selector.list_available_images("VLLM") == {
        "VLLM": {"gfx942": "example/vllm:latest"}
    }

    monkeypatch.setattr(
        "Magpie.modes.benchmark.image_selector.detect_gpu",
        lambda: (GPUVendor.AMD, "gfx942"),
    )
    assert selector.select_image("vllm") == "example/vllm:latest"
    assert selector.get_runner_type() == "mi300x"


@pytest.mark.parametrize(
    ("arch", "runner"),
    [
        ("gfx942", "mi300x"),
        ("gfx950", "mi355x"),
        ("gfx1100", "mi325x"),
        ("gfx1151", "radeon8060s"),
        ("sm_80", "a100"),
        ("sm_90", "h100"),
        ("sm_90a", "h200"),
        ("sm_100", "b200"),
    ],
)
def test_image_selector_runner_mapping(tmp_path, arch, runner):
    selector = ImageSelector(str(tmp_path / "missing.yaml"))
    assert selector.get_runner_type(arch) == runner


def test_image_selector_reports_missing_framework_arch_and_bad_yaml(tmp_path):
    missing = ImageSelector(str(tmp_path / "missing.yaml"))
    assert missing.list_available_images() == {}
    with pytest.raises(ValueError, match="No image mapping"):
        missing.select_image("vllm", "gfx942")

    config = tmp_path / "images.yaml"
    config.write_text("vllm:\n  gfx942: image\n")
    selector = ImageSelector(str(config))
    with pytest.raises(ValueError, match="No image found"):
        selector.select_image("vllm", "gfx950")
    with pytest.raises(ValueError, match="No runner type"):
        selector.get_runner_type("unknown")

    config.write_text("not: [valid")
    assert ImageSelector(str(config)).list_available_images() == {}


def test_inferencex_default_resolution_prefers_environment(monkeypatch, tmp_path):
    override = tmp_path / "explicit"
    monkeypatch.setenv("MAGPIE_INFERENCEX_PATH", str(override))
    assert _resolve_default_inferencex_dir() == str(override)


def test_inferencex_manager_reuses_configured_and_default_paths(tmp_path):
    configured = tmp_path / "configured"
    configured.mkdir()
    manager = InferenceXManager(default_dir=str(tmp_path / "default"))
    assert manager.ensure_available(str(configured)) == str(configured)
    assert manager._is_placeholder(None) is True
    assert manager._is_placeholder("") is True
    assert manager._is_placeholder("YOUR_INFERENCEX_PATH") is True
    assert manager._is_placeholder(str(configured)) is False

    default = Path(manager.default_dir)
    default.mkdir()
    assert manager.ensure_available() == str(default)
    assert manager._validate_installation(str(default)) is False
    (default / "benchmarks").mkdir()
    assert manager._validate_installation(str(default)) is True
    assert manager.ensure_available() == str(default)


def test_inferencex_manager_clones_repository(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "InferenceX"
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="ok")

    monkeypatch.setattr("Magpie.modes.benchmark.inferencex.subprocess.run", run)
    manager = InferenceXManager(
        repo_url="https://example/repo.git", default_dir=str(target)
    )

    assert manager.ensure_available() == str(target)
    assert target.parent.exists()
    assert calls[0][0] == [
        "git",
        "clone",
        "https://example/repo.git",
        str(target),
    ]
    assert calls[0][1]["timeout"] == 300


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (subprocess.TimeoutExpired("git", 300), "timed out"),
        (FileNotFoundError(), "git is not installed"),
        (OSError("read-only"), "Failed to setup"),
    ],
)
def test_inferencex_manager_translates_clone_exceptions(
    tmp_path, monkeypatch, failure, message
):
    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr("Magpie.modes.benchmark.inferencex.subprocess.run", fail)
    manager = InferenceXManager(default_dir=str(tmp_path / "InferenceX"))

    with pytest.raises(RuntimeError, match=message):
        manager.ensure_available()


def test_inferencex_manager_reports_git_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "Magpie.modes.benchmark.inferencex.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 2, stderr="authentication failed"
        ),
    )
    manager = InferenceXManager(default_dir=str(tmp_path / "InferenceX"))

    with pytest.raises(RuntimeError, match="authentication failed"):
        manager.ensure_available()


def test_inferencex_module_level_helpers_delegate(monkeypatch, tmp_path):
    import Magpie.modes.benchmark.inferencex as inferencex

    inferencex._manager = None
    first = get_manager()
    assert get_manager() is first

    class StubManager:
        def ensure_available(self, configured_path):
            return f"resolved:{configured_path}"

    monkeypatch.setattr(inferencex, "_manager", StubManager())
    assert ensure_inferencex_available("/configured") == "resolved:/configured"
