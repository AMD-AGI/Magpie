import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from Magpie.modes.benchmark import benchmarker
from Magpie.modes.benchmark.benchmarker import BenchmarkMode
from Magpie.modes.benchmark.config import (
    BenchmarkConfig,
    GapAnalysisConfig,
    GpuSelectionConfig,
    ProfilerConfig,
    RayConfig,
    ServerLifecycleConfig,
    TorchProfilerConfig,
)
from Magpie.modes.benchmark.result import (
    BenchmarkResult,
    KernelMetrics,
    LatencyMetrics,
    ThroughputMetrics,
)
from Magpie.utils.gpu import GPUVendor


def make_config(tmp_path, **changes):
    values = {
        "framework": "vllm",
        "model": "demo",
        "run_mode": "local",
        "runner_type": "mi300x",
        "inferencex_path": str(tmp_path / "InferenceX"),
        "profiler": ProfilerConfig(torch_profiler=TorchProfilerConfig(enabled=False)),
    }
    values.update(changes)
    return BenchmarkConfig(**values)


def successful_result():
    return BenchmarkResult(
        success=True,
        throughput=ThroughputMetrics(request_throughput=2),
        latency=LatencyMetrics(ttft_mean=3),
    )


def test_run_local_happy_path(monkeypatch, tmp_path):
    config = make_config(tmp_path)
    mode = BenchmarkMode(config, output_dir=str(tmp_path / "results"))
    monkeypatch.setattr(benchmarker, "ensure_inferencex_available", lambda path: path)
    monkeypatch.setattr(mode, "_apply_gpu_selection", lambda: None)
    monkeypatch.setattr(mode, "_prepare_benchmark_scripts", lambda: None)
    monkeypatch.setattr(
        mode, "_get_benchmark_script", lambda runner: "benchmarks/run.sh"
    )
    monkeypatch.setattr(mode, "_cleanup_server_processes", lambda framework: None)

    def execute(_cmd, _env, workspace):
        (workspace / "inferencex_result.json").write_text("{}")
        return successful_result(), "stdout", ""

    monkeypatch.setattr(mode, "_execute_local_benchmark", execute)
    monkeypatch.setattr(
        benchmarker.ResultParser,
        "parse_inferencex_result",
        lambda *a, **k: successful_result(),
    )
    result = mode.run("local-task")
    assert result.success is True
    assert result.framework == "vllm"
    assert result.workspace_dir
    assert mode._task_id == "local-task"


def test_run_docker_with_trace_analysis(monkeypatch, tmp_path):
    profile = ProfilerConfig()
    profile.tracelens.enabled = True
    profile.tracelens.analysis_mode = "pytorch"
    config = make_config(
        tmp_path, run_mode="docker", docker_image="image:test", profiler=profile
    )
    mode = BenchmarkMode(config, output_dir=str(tmp_path / "results"))
    monkeypatch.setattr(benchmarker, "ensure_inferencex_available", lambda path: path)
    monkeypatch.setattr(mode, "_apply_gpu_selection", lambda: None)
    monkeypatch.setattr(mode, "_prepare_benchmark_scripts", lambda: None)
    monkeypatch.setattr(
        mode, "_get_benchmark_script", lambda runner: "benchmarks/run.sh"
    )
    monkeypatch.setattr(mode, "_select_image", lambda: "image:test")
    monkeypatch.setattr(mode, "_build_docker_command", lambda **kwargs: ["docker"])

    def execute(_cmd, workspace):
        (workspace / "inferencex_result.json").write_text("{}")
        (workspace / "torch_trace" / "rank0.json.gz").write_text("trace")
        return successful_result(), "", ""

    monkeypatch.setattr(mode, "_execute_benchmark", execute)
    monkeypatch.setattr(
        benchmarker.ResultParser,
        "parse_inferencex_result",
        lambda *a, **k: successful_result(),
    )
    monkeypatch.setattr(
        benchmarker.ResultParser,
        "parse_torch_trace",
        lambda path: [KernelMetrics("gemm", 1, 100, 1)],
    )
    monkeypatch.setattr(
        mode, "_run_tracelens_analysis", lambda *args: {"enabled": True}
    )
    result = mode.run()
    assert result.success is True
    assert result.top_bottlenecks == ["gemm"]
    assert result.tracelens_analysis == {"enabled": True}


def test_run_docker_dispatches_to_server_lifecycle(monkeypatch, tmp_path):
    config = make_config(
        tmp_path,
        run_mode="docker",
        docker_image="image:test",
        benchmark_script="sglang_mi355x.sh",
        server_lifecycle=ServerLifecycleConfig(
            enabled=True,
            pid_dir=str(tmp_path / "state"),
        ),
    )
    mode = BenchmarkMode(config, output_dir=str(tmp_path / "results"))
    monkeypatch.setattr(benchmarker, "ensure_inferencex_available", lambda path: path)
    monkeypatch.setattr(mode, "_reuse_http_healthy", lambda port: False)
    monkeypatch.setattr(mode, "_apply_gpu_selection", lambda: None)
    monkeypatch.setattr(mode, "_prepare_benchmark_scripts", lambda: None)
    monkeypatch.setattr(
        mode, "_get_benchmark_script", lambda runner: "benchmarks/sglang_mi355x.sh"
    )
    monkeypatch.setattr(mode, "_select_image", lambda: "image:test")
    calls = []

    def execute(**kwargs):
        calls.append(kwargs)
        (kwargs["workspace"] / "inferencex_result.json").write_text("{}")
        return successful_result(), "", ""

    monkeypatch.setattr(mode, "_execute_docker_benchmark_with_reuse", execute)
    monkeypatch.setattr(
        benchmarker.ResultParser,
        "parse_inferencex_result",
        lambda *a, **k: successful_result(),
    )

    result = mode.run("docker-lifecycle")

    assert result.success is True
    assert calls[0]["docker_image"] == "image:test"
    assert calls[0]["runner_type"] == "mi300x"


def inference_profile():
    profile = ProfilerConfig()
    profile.tracelens.enabled = True
    profile.tracelens.analysis_mode = "inference"
    profile.gpu_monitor.enabled = False
    return profile


def prepare_run_mode(monkeypatch, mode):
    monkeypatch.setattr(benchmarker, "ensure_inferencex_available", lambda path: path)
    monkeypatch.setattr(mode, "_apply_gpu_selection", lambda: None)
    monkeypatch.setattr(mode, "_prepare_benchmark_scripts", lambda: None)
    monkeypatch.setattr(
        mode, "_get_benchmark_script", lambda runner: "benchmarks/run.sh"
    )
    monkeypatch.setattr(mode, "_select_image", lambda: "image:test")


def test_run_tracelens_runtime_setup_failure(monkeypatch, tmp_path):
    mode = BenchmarkMode(
        make_config(
            tmp_path,
            run_mode="docker",
            docker_image="image:test",
            profiler=inference_profile(),
        ),
        output_dir=str(tmp_path / "results"),
    )
    prepare_run_mode(monkeypatch, mode)
    monkeypatch.setattr(
        benchmarker,
        "prepare_tracelens_runtime_image",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("patch failed")),
    )
    result = mode.run()
    assert result.success is False
    assert "runtime image setup failed" in result.errors[0]
    assert result.tracelens_analysis["runtime_error"] == "patch failed"


def test_run_tracelens_preprocess_failure(monkeypatch, tmp_path):
    mode = BenchmarkMode(
        make_config(tmp_path, profiler=inference_profile()),
        output_dir=str(tmp_path / "results"),
    )
    prepare_run_mode(monkeypatch, mode)

    class Pipeline:
        def __init__(self, config):
            pass

        def prepare(self, workspace, runtime=None):
            raise RuntimeError("prepare failed")

        def restore(self):
            return {"warnings": ["restored partially"]}

    monkeypatch.setattr(benchmarker, "TraceLensInferencePipeline", Pipeline)
    result = mode.run()
    assert result.success is False
    assert "preprocess failed" in result.errors[0]
    assert result.tracelens_analysis["restore"]["warnings"]


def test_run_missing_result_collects_stderr_and_server_errors(monkeypatch, tmp_path):
    mode = BenchmarkMode(make_config(tmp_path), output_dir=str(tmp_path / "results"))
    prepare_run_mode(monkeypatch, mode)
    monkeypatch.setattr(mode, "_cleanup_server_processes", lambda framework: None)

    def execute(_cmd, _env, workspace):
        (workspace / "server.log").write_text("ready\nRuntimeError: server crashed\n")
        return successful_result(), "", "diagnostic"

    monkeypatch.setattr(mode, "_execute_local_benchmark", execute)
    result = mode.run()
    assert result.success is False
    assert any("not found" in error for error in result.errors)
    assert any("diagnostic" in error for error in result.errors)
    assert any("server.log errors" in error for error in result.errors)


def test_run_inference_trace_gap_monitor_and_restore(monkeypatch, tmp_path):
    profile = inference_profile()
    profile.gpu_monitor.enabled = True
    config = make_config(
        tmp_path,
        profiler=profile,
        gap_analysis=GapAnalysisConfig(enabled=True),
        envs={"HIP_VISIBLE_DEVICES": "3"},
    )
    mode = BenchmarkMode(config, output_dir=str(tmp_path / "results"))
    prepare_run_mode(monkeypatch, mode)
    monkeypatch.setattr(mode, "_cleanup_server_processes", lambda framework: None)

    class Pipeline:
        def __init__(self, config):
            pass

        def prepare(self, workspace, runtime=None):
            return {"warnings": ["prepare warning"]}

        def restore(self):
            return {"warnings": ["restore warning"]}

    class Monitor:
        def __init__(self, **kwargs):
            assert kwargs["device_id"] == 3

        def start(self):
            return True

        def stop(self):
            return SimpleNamespace(sample_count=2, to_dict=lambda: {"sample_count": 2})

    monkeypatch.setattr(benchmarker, "TraceLensInferencePipeline", Pipeline)
    monkeypatch.setattr(benchmarker, "GPUMonitor", Monitor)

    def execute(_cmd, _env, workspace):
        (workspace / "inferencex_result.json").write_text("{}")
        (workspace / "torch_trace" / "rank0.json.gz").write_text("trace")
        return successful_result(), "", ""

    monkeypatch.setattr(mode, "_execute_local_benchmark", execute)
    monkeypatch.setattr(
        benchmarker.ResultParser,
        "parse_inferencex_result",
        lambda *a, **k: successful_result(),
    )
    monkeypatch.setattr(
        benchmarker.ResultParser,
        "parse_torch_trace",
        lambda path: [KernelMetrics("gemm", 1, 100, 1)],
    )
    monkeypatch.setattr(
        mode,
        "_run_tracelens_inference_analysis",
        lambda **kwargs: {"output_files": ["trace.csv"]},
    )
    monkeypatch.setattr(
        mode, "_run_gap_analysis", lambda *args: {"top_kernels": [{"name": "gemm"}]}
    )
    result = mode.run()
    assert result.success is True
    assert result.gpu_monitor == {"sample_count": 2}
    assert result.gap_analysis["top_kernels"] == [{"name": "gemm"}]
    assert result.tracelens_analysis["preprocess"]["warnings"]
    assert result.tracelens_analysis["restore"]["warnings"]


@pytest.mark.parametrize("failure", ["inferencex", "gpu", "script"])
def test_run_setup_failures(monkeypatch, tmp_path, failure):
    mode = BenchmarkMode(make_config(tmp_path), output_dir=str(tmp_path / failure))
    if failure == "inferencex":
        monkeypatch.setattr(
            benchmarker,
            "ensure_inferencex_available",
            lambda path: (_ for _ in ()).throw(RuntimeError("clone failed")),
        )
    else:
        monkeypatch.setattr(
            benchmarker, "ensure_inferencex_available", lambda path: path
        )
        if failure == "gpu":
            monkeypatch.setattr(
                mode,
                "_apply_gpu_selection",
                lambda: (_ for _ in ()).throw(RuntimeError("no gpu")),
            )
        else:
            monkeypatch.setattr(mode, "_apply_gpu_selection", lambda: None)
            monkeypatch.setattr(mode, "_prepare_benchmark_scripts", lambda: None)
            monkeypatch.setattr(
                mode,
                "_get_benchmark_script",
                lambda runner: (_ for _ in ()).throw(FileNotFoundError("missing")),
            )
    result = mode.run()
    assert result.success is False
    assert result.errors


@pytest.mark.parametrize(
    ("throughput", "latency", "expected"),
    [
        (ThroughputMetrics(request_throughput=1), None, True),
        (ThroughputMetrics(output_throughput=1), None, True),
        (ThroughputMetrics(completed_requests=1), None, True),
        (None, LatencyMetrics(ttft_mean=1), True),
        (None, LatencyMetrics(e2el_mean=1), True),
        (ThroughputMetrics(), LatencyMetrics(), False),
    ],
)
def test_validate_results(tmp_path, throughput, latency, expected):
    mode = BenchmarkMode(make_config(tmp_path))
    assert (
        mode._validate_results(BenchmarkResult(throughput=throughput, latency=latency))
        is expected
    )


def test_gpu_selection_amd_nvidia_and_skips(monkeypatch, tmp_path):
    disabled = make_config(tmp_path, gpu_selection=GpuSelectionConfig(auto=False))
    BenchmarkMode(disabled)._apply_gpu_selection()

    manual = make_config(
        tmp_path,
        gpu_selection=GpuSelectionConfig(auto=True),
        envs={"HIP_VISIBLE_DEVICES": "3"},
    )
    BenchmarkMode(manual)._apply_gpu_selection()
    assert manual.envs == {"HIP_VISIBLE_DEVICES": "3"}

    amd = make_config(
        tmp_path, gpu_selection=GpuSelectionConfig(auto=True, count=2), envs={}
    )
    monkeypatch.setattr(benchmarker, "find_idle_gpus", lambda **kwargs: [2, 4])
    monkeypatch.setattr(benchmarker, "detect_gpu", lambda: (GPUVendor.AMD, "gfx942"))
    BenchmarkMode(amd)._apply_gpu_selection()
    assert amd.envs["ROCR_VISIBLE_DEVICES"] == "2,4"
    assert amd.envs["HIP_VISIBLE_DEVICES"] == "0,1"

    nvidia = make_config(
        tmp_path, gpu_selection=GpuSelectionConfig(auto=True), envs={"TP": "bad"}
    )
    monkeypatch.setattr(benchmarker, "find_idle_gpus", lambda **kwargs: [7])
    monkeypatch.setattr(benchmarker, "detect_gpu", lambda: (GPUVendor.NVIDIA, "sm_90"))
    BenchmarkMode(nvidia)._apply_gpu_selection()
    assert nvidia.envs["CUDA_VISIBLE_DEVICES"] == "7"
    assert nvidia.envs["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_gpu_selection_errors(monkeypatch, tmp_path):
    config = make_config(tmp_path, gpu_selection=GpuSelectionConfig(auto=True, count=2))
    mode = BenchmarkMode(config)
    monkeypatch.setattr(benchmarker, "find_idle_gpus", lambda **kwargs: [0])
    with pytest.raises(RuntimeError, match="needed 2"):
        mode._apply_gpu_selection()
    monkeypatch.setattr(
        benchmarker,
        "find_idle_gpus",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("busy")),
    )
    with pytest.raises(RuntimeError, match="Free a GPU"):
        mode._apply_gpu_selection()


def test_build_commands_and_script_resolution(monkeypatch, tmp_path):
    inferencex = tmp_path / "InferenceX"
    benchmarks = inferencex / "benchmarks"
    nested = benchmarks / "single_node"
    nested.mkdir(parents=True)
    native = nested / "gptoss_fp8_mi300x.sh"
    native.write_text("#!/bin/bash")
    config = make_config(tmp_path, inferencex_path=str(inferencex), envs={"TP": 2})
    mode = BenchmarkMode(config)
    mode._task_id = "task"
    assert (
        mode._get_benchmark_script("mi300x")
        == "benchmarks/single_node/gptoss_fp8_mi300x.sh"
    )
    config.benchmark_script = native.name
    assert mode._get_benchmark_script("mi300x").endswith(native.name)

    monkeypatch.setattr(benchmarker, "detect_gpu", lambda: (GPUVendor.AMD, "gfx942"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    docker = mode._build_docker_command("image:test", workspace, "mi300x")
    assert "--device=/dev/kfd" in docker
    assert "--device=/dev/dri" in docker
    assert "--device=/dev/mem" not in docker
    assert docker[-1].startswith("cd /opt/InferenceX")

    server_docker = mode._build_docker_command(
        "image:test",
        workspace,
        "mi300x",
        phase="server",
        container_name="magpie-server-test",
        detach=True,
    )
    assert "--detach" in server_docker
    assert "--rm" not in server_docker
    assert "magpie-server-test" in server_docker
    assert "MAGPIE_RUN_PHASE=server" in server_docker
    assert "MAGPIE_KEEP_CONTAINER_ALIVE=1" in server_docker
    assert "com.amd.magpie.server-lifecycle=true" in server_docker

    command, env = mode._build_local_command(
        workspace, "mi300x", "server", workspace / "pid"
    )
    assert command[0] == "bash"
    assert env["MAGPIE_RUN_PHASE"] == "server"
    assert env["MAGPIE_SERVER_PID_FILE"].endswith("pid")


def test_prepare_scripts_select_runner_and_docker_nvidia_mounts(monkeypatch, tmp_path):
    inferencex = tmp_path / "InferenceX"
    config = make_config(
        tmp_path,
        inferencex_path=str(inferencex),
        docker_image="image:test",
        hf_cache_path=str(tmp_path / "hf"),
        model=str(tmp_path / "model"),
        envs={"CUSTOM": "yes"},
    )
    (tmp_path / "hf").mkdir()
    (tmp_path / "model").mkdir()
    mode = BenchmarkMode(config)
    mode._prepare_benchmark_scripts()
    copied = list((inferencex / "benchmarks").glob("*.sh"))
    assert copied
    assert all(path.stat().st_mode & 0o111 for path in copied)
    assert mode._get_runner_type() == "mi300x"
    assert mode._select_image() == "image:test"

    monkeypatch.setattr(benchmarker, "detect_gpu", lambda: (GPUVendor.NVIDIA, "sm_90"))
    monkeypatch.setattr(
        mode, "_get_benchmark_script", lambda runner: "benchmarks/run.sh"
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = mode._build_docker_command("image:test", workspace, "h100")
    assert "--gpus" in command
    assert any(str(tmp_path / "hf") in item for item in command)
    assert any(str(tmp_path / "model") in item for item in command)
    assert "-e" in command


def test_fix_workspace_ownership_helper_and_error(monkeypatch, tmp_path):
    mode = BenchmarkMode(make_config(tmp_path, docker_image="image:test"))
    calls = []
    monkeypatch.setattr(benchmarker.os, "getuid", lambda: 501)
    monkeypatch.setattr(benchmarker.os, "getgid", lambda: 20)
    monkeypatch.setattr(
        benchmarker.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd)
    )
    mode._fix_workspace_ownership(tmp_path)
    assert calls[0][-2:] == ["501:20", "/workspace"]
    monkeypatch.setattr(
        benchmarker.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("docker failed")),
    )
    mode._fix_workspace_ownership(tmp_path)


def test_script_resolution_fallback_and_errors(tmp_path):
    inferencex = tmp_path / "InferenceX"
    benchmarks = inferencex / "benchmarks"
    benchmarks.mkdir(parents=True)
    generic = benchmarks / "vllm_mi300x.sh"
    generic.write_text("#!/bin/bash")
    config = make_config(tmp_path, inferencex_path=str(inferencex))
    mode = BenchmarkMode(config)
    assert mode._get_benchmark_script("mi300x") == "benchmarks/vllm_mi300x.sh"
    generic.unlink()
    with pytest.raises(FileNotFoundError, match="No benchmark script"):
        mode._get_benchmark_script("mi300x")
    config.benchmark_script = "missing.sh"
    with pytest.raises(FileNotFoundError, match="Specified benchmark_script"):
        mode._get_benchmark_script("mi300x")


@pytest.mark.parametrize("behavior", ["success", "failed", "timeout", "exception"])
def test_execute_benchmark_paths(monkeypatch, tmp_path, behavior):
    mode = BenchmarkMode(make_config(tmp_path, docker_image="image:test"))
    mode._task_id = "task"
    monkeypatch.setattr(mode, "_fix_workspace_ownership", lambda workspace: None)
    if behavior == "success":
        runner = lambda *a, **k: subprocess.CompletedProcess(a[0], 0, "out", "")
    elif behavior == "failed":
        runner = lambda *a, **k: subprocess.CompletedProcess(a[0], 2, "", "bad")
    elif behavior == "timeout":
        runner = lambda *a, **k: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(a[0], 1, output=b"partial", stderr=b"late")
        )
    else:
        runner = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setattr(mode, "_run_streaming_command", runner)
    result, stdout, stderr = mode._execute_benchmark(["docker"], tmp_path)
    assert result.success is (behavior == "success")
    if behavior != "success":
        assert result.errors


def test_logs_cleanup_and_symlink_helpers(monkeypatch, tmp_path):
    mode = BenchmarkMode(make_config(tmp_path))
    mode._save_logs(tmp_path, "out", "err")
    assert (tmp_path / "benchmark_stdout.log").read_text() == "out"
    assert (tmp_path / "benchmark_stderr.log").read_text() == "err"
    assert mode._create_workspace_symlink(tmp_path) is None
    BenchmarkMode._remove_workspace_symlink(None)
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "target")
    BenchmarkMode._remove_workspace_symlink(link)
    assert not link.exists()

    monkeypatch.setattr(benchmarker.os, "getuid", lambda: 0)
    mode._fix_workspace_ownership(tmp_path)


def lifecycle_mode(tmp_path, **lifecycle_changes):
    lifecycle = ServerLifecycleConfig(
        enabled=True,
        pid_dir=str(tmp_path / "pids"),
        **lifecycle_changes,
    )
    config = make_config(
        tmp_path,
        envs={"PORT": "9000", "TP": 2, "EXTRA_VLLM_ARGS": "--fast"},
        server_lifecycle=lifecycle,
    )
    return BenchmarkMode(config)


def docker_lifecycle_mode(tmp_path, **lifecycle_changes):
    lifecycle = ServerLifecycleConfig(
        enabled=True,
        pid_dir=str(tmp_path / "pids"),
        **lifecycle_changes,
    )
    config = make_config(
        tmp_path,
        run_mode="docker",
        docker_image="image:test",
        envs={"PORT": "9000", "TP": 2, "EXTRA_VLLM_ARGS": "--fast"},
        server_lifecycle=lifecycle,
    )
    return BenchmarkMode(config)


def test_reuse_metadata_ports_paths_and_health(monkeypatch, tmp_path):
    mode = lifecycle_mode(tmp_path)
    assert mode._reuse_benchmark_port() == 9000
    mode.config.envs["PORT"] = "invalid"
    assert mode._reuse_benchmark_port() == 8888
    desired = mode._desired_reuse_server_meta(8888)
    assert desired["framework"] == "vllm"
    assert desired["tp"] == "2"
    assert mode._reuse_meta_mismatch(None, desired)
    assert mode._reuse_meta_mismatch(desired, desired) is None
    changed = dict(desired, model="other")
    assert "model" in mode._reuse_meta_mismatch(changed, desired)
    mode.config.envs["CONC"] = 64
    mode.config.envs["ISL"] = 8192
    client_changed = mode._desired_reuse_server_meta(8888)
    assert mode._reuse_meta_mismatch(desired, client_changed) is None
    mode.config.envs["ARTEMIS_ENABLED"] = 1
    server_changed = mode._desired_reuse_server_meta(8888)
    assert "server_env" in mode._reuse_meta_mismatch(desired, server_changed)

    directory, pid_file, meta_file = mode._reuse_server_paths(8888)
    assert directory.is_dir()
    assert pid_file.name == "vllm_8888.pid"
    mode._reuse_write_meta(meta_file, desired)
    assert mode._reuse_read_meta(meta_file) == desired
    meta_file.write_text("[]")
    assert mode._reuse_read_meta(meta_file) is None
    meta_file.write_text("{")
    assert mode._reuse_read_meta(meta_file) is None

    class Response:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    response = Response()
    monkeypatch.setattr(benchmarker.urllib.request, "urlopen", lambda *a, **k: response)
    assert mode._reuse_http_healthy(9000) is True
    monkeypatch.setattr(
        benchmarker.urllib.request,
        "urlopen",
        lambda *a, **k: (_ for _ in ()).throw(
            benchmarker.urllib.error.URLError("down")
        ),
    )
    assert mode._reuse_http_healthy(9000) is False


def test_reuse_attach_decisions_and_wait(monkeypatch, tmp_path):
    mode = lifecycle_mode(tmp_path)
    desired = mode._desired_reuse_server_meta(9000)
    _, _, meta_file = mode._reuse_server_paths(9000)
    mode._reuse_write_meta(meta_file, desired)
    monkeypatch.setattr(mode, "_reuse_http_healthy", lambda port: True)
    assert mode._reuse_will_attach_to_existing_server() is True
    mode.config.server_lifecycle.force_reuse = True
    meta_file.write_text("{")
    assert mode._reuse_will_attach_to_existing_server() is True
    monkeypatch.setattr(mode, "_reuse_http_healthy", lambda port: False)
    assert mode._reuse_will_attach_to_existing_server() is False

    times = iter([0, 1])
    monkeypatch.setattr(benchmarker.time, "time", lambda: next(times))
    monkeypatch.setattr(mode, "_reuse_http_healthy", lambda port: True)
    assert mode._reuse_wait_health(9000, 2) is True


def test_reuse_existing_server_client_and_cleanup(monkeypatch, tmp_path):
    mode = lifecycle_mode(tmp_path, cleanup=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        mode, "_get_benchmark_script", lambda runner: "benchmarks/vllm_mi300x.sh"
    )
    monkeypatch.setattr(mode, "_reuse_http_healthy", lambda port: True)
    monkeypatch.setattr(mode, "_reuse_meta_mismatch", lambda stored, desired: None)
    _, pid_file, meta_file = mode._reuse_server_paths(9000)
    pid_file.write_text("123")
    mode._reuse_write_meta(meta_file, {"server_pid": 123})
    monkeypatch.setattr(
        mode, "_build_local_command", lambda **kwargs: ([kwargs["phase"]], {})
    )
    monkeypatch.setattr(
        mode,
        "_execute_local_benchmark",
        lambda *a, **k: (successful_result(), "client out", "client err"),
    )
    terminated = []
    monkeypatch.setattr(
        mode, "_reuse_terminate_persistent_server", lambda pid: terminated.append(pid)
    )
    monkeypatch.setattr(mode, "_cleanup_server_processes", lambda framework: None)
    result, stdout, stderr = mode._execute_local_benchmark_with_reuse(
        workspace, "mi300x"
    )
    assert result.success is True
    assert "Using existing" in stdout
    assert "client err" in stderr
    assert terminated == [123]
    assert not pid_file.exists()
    assert not meta_file.exists()


def test_reuse_existing_server_mismatch_and_non_builtin(monkeypatch, tmp_path):
    mode = lifecycle_mode(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        mode, "_get_benchmark_script", lambda runner: "benchmarks/custom.sh"
    )
    result, _, _ = mode._execute_local_benchmark_with_reuse(workspace, "mi300x")
    assert "built-in" in result.errors[0]

    monkeypatch.setattr(
        mode, "_get_benchmark_script", lambda runner: "benchmarks/vllm_mi300x.sh"
    )
    monkeypatch.setattr(mode, "_reuse_http_healthy", lambda port: True)
    monkeypatch.setattr(
        mode, "_reuse_meta_mismatch", lambda stored, desired: "model differs"
    )
    result, _, _ = mode._execute_local_benchmark_with_reuse(workspace, "mi300x")
    assert "metadata mismatch" in result.errors[0]


def test_reuse_spawns_server_then_runs_client(monkeypatch, tmp_path):
    mode = lifecycle_mode(tmp_path, cleanup=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        mode, "_get_benchmark_script", lambda runner: "benchmarks/vllm_mi300x.sh"
    )
    monkeypatch.setattr(mode, "_reuse_http_healthy", lambda port: False)
    monkeypatch.setattr(mode, "_reuse_clear_stale_artifacts", lambda *args: None)

    def build(**kwargs):
        if kwargs["phase"] == "server":
            kwargs["server_pid_file"].write_text("321")
        return [kwargs["phase"]], {}

    monkeypatch.setattr(mode, "_build_local_command", build)
    monkeypatch.setattr(
        mode,
        "_run_streaming_command",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, "server out", ""),
    )
    monkeypatch.setattr(mode, "_reuse_wait_health", lambda port, deadline: True)
    monkeypatch.setattr(
        mode,
        "_execute_local_benchmark",
        lambda *a, **k: (successful_result(), "client out", ""),
    )
    result, stdout, _ = mode._execute_local_benchmark_with_reuse(workspace, "mi300x")
    assert result.success is True
    assert "Spawned persistent server PID 321" in stdout
    _, pid_file, meta_file = mode._reuse_server_paths(9000)
    assert pid_file.read_text().strip() == "321"
    assert mode._reuse_read_meta(meta_file)["server_pid"] == 321


def test_docker_reuse_spawns_server_then_runs_client(monkeypatch, tmp_path):
    mode = docker_lifecycle_mode(tmp_path, cleanup=False)
    mode._task_id = "docker-reuse"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        mode, "_get_benchmark_script", lambda runner: "benchmarks/vllm_mi300x.sh"
    )
    monkeypatch.setattr(mode, "_reuse_http_healthy", lambda port: False)
    monkeypatch.setattr(mode, "_reuse_clear_stale_artifacts", lambda *args: None)
    monkeypatch.setattr(mode, "_reuse_wait_docker_health", lambda *args: True)

    phases = []

    def build(**kwargs):
        phases.append(kwargs["phase"])
        return [kwargs["phase"]]

    monkeypatch.setattr(mode, "_build_docker_command", build)
    monkeypatch.setattr(
        mode,
        "_run_streaming_command",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, "container-123\n", ""),
    )
    monkeypatch.setattr(
        mode,
        "_execute_benchmark",
        lambda *a, **k: (successful_result(), "client out", ""),
    )

    result, stdout, _ = mode._execute_docker_benchmark_with_reuse(
        workspace, "mi300x", "image:test"
    )
    assert result.success is True
    assert phases == ["server", "client"]
    assert "Spawned Docker server container-123" in stdout
    _, handle_file, meta_file = mode._reuse_server_paths(9000)
    assert handle_file.read_text().strip() == "container-123"
    meta = mode._reuse_read_meta(meta_file)
    assert meta["container_id"] == "container-123"
    assert meta["docker_image"] == "image:test"


def test_docker_reuse_existing_server_client_and_cleanup(monkeypatch, tmp_path):
    mode = docker_lifecycle_mode(tmp_path, cleanup=True)
    mode._task_id = "docker-reuse"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        mode, "_get_benchmark_script", lambda runner: "benchmarks/vllm_mi300x.sh"
    )
    monkeypatch.setattr(mode, "_reuse_http_healthy", lambda port: True)
    monkeypatch.setattr(mode, "_docker_container_running", lambda ref: True)
    _, handle_file, meta_file = mode._reuse_server_paths(9000)
    desired = mode._desired_reuse_server_meta(9000, docker_image="image:test")
    mode._reuse_write_meta(
        meta_file,
        {
            **desired,
            "container_id": "container-123",
            "visible_devices": {"ROCR_VISIBLE_DEVICES": "4,5"},
        },
    )
    handle_file.write_text("container-123\n")

    phases = []
    monkeypatch.setattr(
        mode,
        "_build_docker_command",
        lambda **kwargs: phases.append(kwargs["phase"]) or [kwargs["phase"]],
    )
    monkeypatch.setattr(
        mode,
        "_execute_benchmark",
        lambda *a, **k: (successful_result(), "client out", "client err"),
    )
    monkeypatch.setattr(mode, "_docker_container_logs", lambda ref: "server logs")
    removed = []
    monkeypatch.setattr(
        mode, "_docker_remove_container", lambda ref: removed.append(ref)
    )

    result, stdout, stderr = mode._execute_docker_benchmark_with_reuse(
        workspace, "mi300x", "image:test"
    )
    assert result.success is True
    assert phases == ["client"]
    assert "Using Docker server container-123" in stdout
    assert "client err" in stderr
    assert mode.config.envs["ROCR_VISIBLE_DEVICES"] == "4,5"
    assert removed == ["container-123"]
    assert not handle_file.exists() and not meta_file.exists()
    assert (workspace / "reuse_server_container.log").read_text() == "server logs"


def test_docker_reuse_rejects_missing_and_custom_scripts(monkeypatch, tmp_path):
    mode = docker_lifecycle_mode(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def missing_script(_runner):
        raise FileNotFoundError("benchmark script is missing")

    monkeypatch.setattr(mode, "_get_benchmark_script", missing_script)
    result, stdout, stderr = mode._execute_docker_benchmark_with_reuse(
        workspace, "mi300x", "image:test"
    )
    assert result.success is False
    assert result.errors == ["benchmark script is missing"]
    assert stdout == stderr == ""

    monkeypatch.setattr(
        mode, "_get_benchmark_script", lambda _runner: "benchmarks/custom.sh"
    )
    result, stdout, stderr = mode._execute_docker_benchmark_with_reuse(
        workspace, "mi300x", "image:test"
    )
    assert result.success is False
    assert "built-in InferenceX benchmark script" in result.errors[0]
    assert stdout == stderr == ""


def test_docker_reuse_rejects_unowned_and_incompatible_servers(monkeypatch, tmp_path):
    mode = docker_lifecycle_mode(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        mode, "_get_benchmark_script", lambda _runner: "benchmarks/vllm_mi300x.sh"
    )
    monkeypatch.setattr(mode, "_reuse_http_healthy", lambda _port: True)

    result, _, _ = mode._execute_docker_benchmark_with_reuse(
        workspace, "mi300x", "image:test"
    )
    assert result.success is False
    assert "not owned by the recorded Magpie Docker lifecycle" in result.errors[0]

    _, handle_file, meta_file = mode._reuse_server_paths(9000)
    desired = mode._desired_reuse_server_meta(9000, docker_image="image:test")
    mode._reuse_write_meta(meta_file, {**desired, "container_id": "container-123"})
    handle_file.write_text("container-123\n")
    monkeypatch.setattr(mode, "_docker_container_running", lambda _ref: True)
    monkeypatch.setattr(
        mode, "_reuse_meta_mismatch", lambda _stored, _desired: "model differs"
    )

    result, _, _ = mode._execute_docker_benchmark_with_reuse(
        workspace, "mi300x", "image:test"
    )
    assert result.success is False
    assert "metadata mismatch" in result.errors[0]


@pytest.mark.parametrize("behavior", ["timeout", "exception", "nonzero"])
def test_docker_reuse_reports_container_launch_failures(
    monkeypatch, tmp_path, behavior
):
    mode = docker_lifecycle_mode(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        mode, "_get_benchmark_script", lambda _runner: "benchmarks/vllm_mi300x.sh"
    )
    monkeypatch.setattr(mode, "_reuse_http_healthy", lambda _port: False)
    monkeypatch.setattr(mode, "_reuse_clear_stale_artifacts", lambda *args: None)
    monkeypatch.setattr(mode, "_build_docker_command", lambda **kwargs: ["docker"])

    if behavior == "timeout":

        def launch(*args, **kwargs):
            raise subprocess.TimeoutExpired(
                args[0], 120, output=b"partial output", stderr=b"late error"
            )

    elif behavior == "exception":

        def launch(*args, **kwargs):
            raise RuntimeError("docker unavailable")

    else:

        def launch(*args, **kwargs):
            return subprocess.CompletedProcess(args[0], 125, "launch output", "bad")

    monkeypatch.setattr(mode, "_run_streaming_command", launch)
    result, stdout, stderr = mode._execute_docker_benchmark_with_reuse(
        workspace, "mi300x", "image:test"
    )
    assert result.success is False
    assert result.errors
    if behavior == "timeout":
        assert stdout == "partial output"
        assert stderr == "late error"
    elif behavior == "exception":
        assert stderr == "docker unavailable"
    else:
        assert stdout == "launch output"
        assert stderr == "bad"


def test_docker_reuse_cleans_up_when_handle_cannot_be_written(monkeypatch, tmp_path):
    mode = docker_lifecycle_mode(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        mode, "_get_benchmark_script", lambda _runner: "benchmarks/vllm_mi300x.sh"
    )
    monkeypatch.setattr(mode, "_reuse_http_healthy", lambda _port: False)
    monkeypatch.setattr(mode, "_reuse_clear_stale_artifacts", lambda *args: None)
    monkeypatch.setattr(mode, "_build_docker_command", lambda **kwargs: ["docker"])
    monkeypatch.setattr(
        mode,
        "_run_streaming_command",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, "container-123\n", ""
        ),
    )
    removed = []
    monkeypatch.setattr(
        mode, "_docker_remove_container", lambda ref: removed.append(ref)
    )
    original_write_text = Path.write_text

    def write_text(path, *args, **kwargs):
        if path.suffix == ".pid":
            raise OSError("read-only state directory")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_text)
    result, _, _ = mode._execute_docker_benchmark_with_reuse(
        workspace, "mi300x", "image:test"
    )
    assert result.success is False
    assert "Could not persist Docker reuse handle" in result.errors[0]
    assert removed == ["container-123"]


def test_docker_reuse_cleans_up_when_metadata_cannot_be_written(monkeypatch, tmp_path):
    mode = docker_lifecycle_mode(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        mode, "_get_benchmark_script", lambda _runner: "benchmarks/vllm_mi300x.sh"
    )
    monkeypatch.setattr(mode, "_reuse_http_healthy", lambda _port: False)
    monkeypatch.setattr(mode, "_reuse_clear_stale_artifacts", lambda *args: None)
    monkeypatch.setattr(mode, "_build_docker_command", lambda **kwargs: ["docker"])
    monkeypatch.setattr(
        mode,
        "_run_streaming_command",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, "container-123\n", ""
        ),
    )
    monkeypatch.setattr(mode, "_reuse_write_meta", lambda *args: False)
    removed = []
    monkeypatch.setattr(
        mode, "_docker_remove_container", lambda ref: removed.append(ref)
    )

    result, _, _ = mode._execute_docker_benchmark_with_reuse(
        workspace, "mi300x", "image:test"
    )
    assert result.success is False
    assert "Could not persist Docker reuse metadata" in result.errors[0]
    assert removed == ["container-123"]


def test_docker_reuse_cleans_up_when_server_never_becomes_healthy(
    monkeypatch, tmp_path
):
    mode = docker_lifecycle_mode(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "server.log").write_text("server startup failed")
    monkeypatch.setattr(
        mode, "_get_benchmark_script", lambda _runner: "benchmarks/vllm_mi300x.sh"
    )
    monkeypatch.setattr(mode, "_reuse_http_healthy", lambda _port: False)
    monkeypatch.setattr(mode, "_reuse_clear_stale_artifacts", lambda *args: None)
    monkeypatch.setattr(mode, "_build_docker_command", lambda **kwargs: ["docker"])
    monkeypatch.setattr(
        mode,
        "_run_streaming_command",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, "container-123\n", ""
        ),
    )
    monkeypatch.setattr(mode, "_reuse_wait_docker_health", lambda *args: False)
    monkeypatch.setattr(mode, "_docker_container_logs", lambda _ref: "container failed")
    removed = []
    monkeypatch.setattr(
        mode, "_docker_remove_container", lambda ref: removed.append(ref)
    )

    result, _, _ = mode._execute_docker_benchmark_with_reuse(
        workspace, "mi300x", "image:test"
    )
    assert result.success is False
    assert "server startup failed" in result.errors[0]
    assert "container failed" in result.errors[0]
    assert removed == ["container-123"]
    _, handle_file, meta_file = mode._reuse_server_paths(9000)
    assert not handle_file.exists() and not meta_file.exists()


def test_reuse_state_and_visible_device_helpers(monkeypatch, tmp_path):
    mode = docker_lifecycle_mode(tmp_path)
    _, handle_file, meta_file = mode._reuse_server_paths(9000)
    handle_file.write_text("container-from-handle\n")

    assert mode._reuse_docker_container_ref(None, handle_file) == (
        "container-from-handle"
    )
    assert (
        mode._reuse_docker_container_ref(
            {"container_name": "container-from-meta"}, handle_file
        )
        == "container-from-meta"
    )

    mode.config.envs.update(
        {"rocr_visible_devices": "2,3", "CUDA_VISIBLE_DEVICES": "", "OTHER": "x"}
    )
    assert mode._reuse_visible_devices() == {"ROCR_VISIBLE_DEVICES": "2,3"}
    mode._reuse_restore_visible_devices(None)
    mode._reuse_restore_visible_devices({"visible_devices": "invalid"})

    assert mode._reuse_write_meta(tmp_path, {"invalid": True}) is False

    meta_file.write_text('{"container_id": "container-from-meta"}')
    removed = []
    monkeypatch.setattr(
        mode, "_docker_remove_container", lambda ref: removed.append(ref)
    )
    mode._reuse_clear_stale_artifacts(handle_file, meta_file, 9000)
    assert removed == ["container-from-meta"]
    assert not handle_file.exists() and not meta_file.exists()

    local = lifecycle_mode(tmp_path)
    _, pid_file, local_meta_file = local._reuse_server_paths(9001)
    pid_file.write_text("not-a-pid")
    local_meta_file.write_text("{}")
    cleaned = []
    monkeypatch.setattr(
        local, "_cleanup_server_processes", lambda framework: cleaned.append(framework)
    )
    local._reuse_clear_stale_artifacts(pid_file, local_meta_file, 9001)
    assert cleaned == ["vllm"]


def test_docker_process_helpers(monkeypatch, tmp_path):
    mode = docker_lifecycle_mode(tmp_path)
    assert mode._docker_container_running("") is False
    assert mode._docker_container_logs("") == ""
    mode._docker_remove_container("")

    calls = []

    def successful_run(command, **kwargs):
        calls.append(command)
        if command[1] == "inspect":
            return subprocess.CompletedProcess(command, 0, "true\n", "")
        if command[1] == "logs":
            return subprocess.CompletedProcess(command, 0, "stdout", "stderr")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(benchmarker.subprocess, "run", successful_run)
    assert mode._docker_container_running("container-123") is True
    assert mode._docker_container_logs("container-123") == "stdoutstderr"
    mode._docker_remove_container("container-123")
    assert [command[1] for command in calls] == ["inspect", "logs", "stop", "rm"]

    monkeypatch.setattr(
        benchmarker.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(args[0], 1)
        ),
    )
    assert mode._docker_container_running("container-123") is False
    assert mode._docker_container_logs("container-123") == ""
    mode._docker_remove_container("container-123")

    assert mode._reuse_server_log_tail(tmp_path) == ""


def test_reuse_wait_docker_health_paths(monkeypatch, tmp_path):
    mode = docker_lifecycle_mode(tmp_path)
    times = iter([0, 1])
    monkeypatch.setattr(benchmarker.time, "time", lambda: next(times))
    monkeypatch.setattr(mode, "_docker_container_running", lambda _ref: False)
    assert mode._reuse_wait_docker_health("container-123", 9000, 2) is False

    times = iter([0, 1])
    monkeypatch.setattr(benchmarker.time, "time", lambda: next(times))
    monkeypatch.setattr(mode, "_docker_container_running", lambda _ref: True)
    monkeypatch.setattr(mode, "_reuse_http_healthy", lambda _port: True)
    assert mode._reuse_wait_docker_health("container-123", 9000, 2) is True


@pytest.mark.parametrize("behavior", ["success", "failure", "timeout", "exception"])
def test_execute_local_benchmark_paths(monkeypatch, tmp_path, behavior):
    mode = BenchmarkMode(make_config(tmp_path))
    if behavior == "success":
        runner = lambda *a, **k: subprocess.CompletedProcess(a[0], 0, "out", "")
    elif behavior == "failure":
        runner = lambda *a, **k: subprocess.CompletedProcess(a[0], 3, "", "bad")
    elif behavior == "timeout":
        runner = lambda *a, **k: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(a[0], 1, output=b"partial", stderr=b"late")
        )
    else:
        runner = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setattr(mode, "_run_streaming_command", runner)
    result, stdout, stderr = mode._execute_local_benchmark(["run"], {}, tmp_path)
    assert result.success is (behavior == "success")
    if behavior != "success":
        assert result.errors


def test_clear_stale_artifacts(monkeypatch, tmp_path):
    mode = lifecycle_mode(tmp_path)
    _, pid_file, meta_file = mode._reuse_server_paths(9000)
    pid_file.write_text("123 extra")
    meta_file.write_text("{}")
    terminated = []
    monkeypatch.setattr(
        mode, "_reuse_terminate_persistent_server", lambda pid: terminated.append(pid)
    )
    monkeypatch.setattr(mode, "_cleanup_server_processes", lambda framework: None)
    mode._reuse_clear_stale_artifacts(pid_file, meta_file, 9000)
    assert terminated == [123]
    assert not pid_file.exists() and not meta_file.exists()


def test_postprocessing_helpers_success_and_errors(monkeypatch, tmp_path):
    config = make_config(
        tmp_path,
        envs={"TP": 2},
        gap_analysis=GapAnalysisConfig(enabled=True),
    )
    mode = BenchmarkMode(config)

    class TraceAnalyzer:
        def __init__(self, config):
            pass

        def analyze(self, **kwargs):
            return {"output_files": ["report.csv"], "errors": ["warning"]}

    monkeypatch.setattr(benchmarker, "TraceLensAnalyzer", TraceAnalyzer)
    assert mode._run_tracelens_analysis(tmp_path, tmp_path)["output_files"]
    monkeypatch.setattr(
        TraceAnalyzer,
        "analyze",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("trace failed")),
    )
    assert mode._run_tracelens_analysis(tmp_path, tmp_path)["error"] == "trace failed"

    pipeline = SimpleNamespace(
        analyze=lambda **kwargs: {
            "output_files": ["host.csv"],
            "warnings": ["warn"],
            "errors": [],
        },
        analyze_in_container=lambda **kwargs: {
            "output_files": ["docker.csv"],
            "warnings": [],
            "errors": ["partial"],
        },
    )
    assert mode._run_tracelens_inference_analysis(
        tmp_path, tmp_path, "mi300x", pipeline
    )["output_files"] == ["host.csv"]
    mode.config.run_mode = "docker"
    mode.config.docker_image = "image:test"
    assert mode._run_tracelens_inference_analysis(
        tmp_path, tmp_path, "mi300x", pipeline
    )["output_files"] == ["docker.csv"]
    pipeline.analyze_in_container = lambda **kwargs: (_ for _ in ()).throw(
        RuntimeError("inference failed")
    )
    assert (
        mode._run_tracelens_inference_analysis(tmp_path, tmp_path, "mi300x", pipeline)[
            "error"
        ]
        == "inference failed"
    )


def test_gap_postprocessing_success_and_error(monkeypatch, tmp_path):
    mode = BenchmarkMode(
        make_config(tmp_path, gap_analysis=GapAnalysisConfig(enabled=True))
    )

    class Result:
        rank_results = [{}, {}]
        merged_kernels = ["gemm"]
        errors = ["warning"]

        def to_csv(self, path, **kwargs):
            path.write_text("csv")

        def to_rank_csv(self, directory):
            return [directory / "rank0.csv"]

        def to_dict(self):
            return {"top_kernels": [{"name": "gemm"}]}

    class Analyzer:
        def __init__(self, config):
            pass

        def analyze(self, path):
            return Result()

    monkeypatch.setattr("Magpie.modes.benchmark.gap_analysis.GapAnalyzer", Analyzer)
    result = mode._run_gap_analysis(tmp_path, tmp_path)
    assert result["top_kernels"][0]["name"] == "gemm"
    assert (tmp_path / "gap_analysis" / "gap_analysis.csv").exists()
    monkeypatch.setattr(
        Analyzer,
        "analyze",
        lambda self, path: (_ for _ in ()).throw(RuntimeError("gap failed")),
    )
    assert mode._run_gap_analysis(tmp_path, tmp_path)["error"] == "gap failed"


def test_ray_task_build_submit_and_population(monkeypatch, tmp_path):
    local = BenchmarkMode(make_config(tmp_path))
    result = local.submit_ray_benchmark(SimpleNamespace())
    assert result.success is False
    assert "run_mode='ray'" in result.errors[0]

    ray_config = make_config(
        tmp_path,
        run_mode="ray",
        ray_config=RayConfig(
            cluster_address="ray://host", shared_storage_path="/shared"
        ),
    )
    mode = BenchmarkMode(ray_config)
    task, error = mode._build_ray_benchmark_task()
    assert error is None
    assert task.task_id == mode._task_id
    executor = SimpleNamespace(submit=lambda task: "task-remote")
    submitted = mode.submit_ray_benchmark(executor)
    assert submitted.success is True
    assert submitted.metadata["task_id"] == "task-remote"
    failed = mode.submit_ray_benchmark(
        SimpleNamespace(
            submit=lambda task: (_ for _ in ()).throw(RuntimeError("submit failed"))
        )
    )
    assert failed.success is False

    result = BenchmarkResult()
    mode._populate_result_from_ray(
        result,
        {
            "workspace_dir": "/results/run",
            "throughput": {"request_throughput": 12},
            "latency": {"ttft": {"mean_ms": 1, "p99_ms": 2}},
            "kernel_summary": ["gemm"],
            "top_bottlenecks": ["gemm"],
            "gap_analysis": {"enabled": True},
            "tracelens_analysis": {"enabled": True},
            "errors": [],
        },
        ray_config.ray_config,
    )
    assert result.success is True
    assert result.throughput.request_throughput == 12
    assert result.latency.ttft_mean == 1
    assert result.metadata["ray_cluster"] == "ray://host"


def test_execute_ray_benchmark_success_start_failure_and_exception(
    monkeypatch, tmp_path
):
    config = make_config(
        tmp_path,
        run_mode="ray",
        ray_config=RayConfig(
            cluster_address="ray://host", shared_storage_path="/shared"
        ),
    )
    mode = BenchmarkMode(config)

    class Executor:
        start_value = True
        execute_value = SimpleNamespace(
            status=SimpleNamespace(value="completed"),
            results={"throughput": {"request_throughput": 5}},
            execution_time=2,
            errors=[],
        )

        def __init__(self, config, ray_config):
            self.stopped = False

        def start(self):
            return self.start_value

        def execute(self, task):
            return self.execute_value

        def stop(self):
            self.stopped = True

    monkeypatch.setattr("Magpie.core.ray_executor.RayJobExecutor", Executor)
    result = mode._execute_ray_benchmark()
    assert result.success is True
    assert result.throughput.request_throughput == 5

    Executor.start_value = False
    result = mode._execute_ray_benchmark()
    assert "Failed to connect" in result.errors[0]
    Executor.start_value = True
    Executor.execute_value = SimpleNamespace(
        status=SimpleNamespace(value="failed"),
        results=None,
        execution_time=1,
        errors=["remote failed"],
    )
    assert mode._execute_ray_benchmark().errors == ["remote failed"]
    monkeypatch.setattr(
        "Magpie.core.ray_executor.RayJobExecutor",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ray unavailable")),
    )
    assert "ray unavailable" in mode._execute_ray_benchmark().errors[0]


def test_cleanup_local_and_docker(monkeypatch, tmp_path):
    local = BenchmarkMode(make_config(tmp_path))
    cleaned = []
    monkeypatch.setattr(
        local, "_cleanup_server_processes", lambda framework: cleaned.append(framework)
    )
    local.cleanup()
    assert cleaned == ["vllm"]
    docker = BenchmarkMode(
        make_config(tmp_path, run_mode="docker", docker_image="image")
    )
    docker._task_id = "task"
    calls = []
    monkeypatch.setattr(
        benchmarker.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd)
    )
    docker.cleanup()
    assert calls[0][-1] == "magpie-benchmark-task"
