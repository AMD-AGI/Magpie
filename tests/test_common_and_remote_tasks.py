import os
import subprocess

import pytest

from Magpie.config import KernelEvalConfig, KernelType, PipelineConfig
from Magpie.eval.compiling import Compiling
from Magpie.remote.tasks import (
    _clear_hidden_gpus,
    _commit_envs,
    _configure_tp_isolation,
    _ensure_extra_arg,
    _extra_args_key,
)
from Magpie.utils.common import (
    compile_hip,
    get_compilation_output_stem,
    get_updated_env,
)


def test_get_updated_env_prepends_path_like_variables(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib")
    monkeypatch.setenv("HOME", "/root")

    env = get_updated_env(
        {
            "PATH": "/custom/bin",
            "LD_LIBRARY_PATH": "/custom/lib",
            "HOME": "/tmp/home",
            "MAGPIE_FLAG": "1",
        }
    )

    assert env["PATH"] == "/custom/bin:/usr/bin"
    assert env["LD_LIBRARY_PATH"] == "/custom/lib:/usr/lib"
    assert env["HOME"] == "/tmp/home"
    assert env["MAGPIE_FLAG"] == "1"


def test_compile_hip_raises_when_hipcc_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("Magpie.utils.common.shutil.which", lambda _: None)

    with pytest.raises(RuntimeError, match="hipcc not found"):
        compile_hip([str(tmp_path / "kernel.hip")], str(tmp_path), "gfx942")


def test_compile_hip_returns_binary_and_shared_object(monkeypatch, tmp_path):
    commands = []

    def fake_run(cmd, capture_output, text, env, cwd):
        commands.append((cmd, cwd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("Magpie.utils.common.shutil.which", lambda _: "/usr/bin/hipcc")
    monkeypatch.setattr("Magpie.utils.common.subprocess.run", fake_run)

    out_file, so_file, errors = compile_hip(
        [str(tmp_path / "vector_add.hip")],
        str(tmp_path),
        "gfx942",
        with_so=True,
        env={"MAGPIE_FLAG": "1"},
    )

    assert errors is None
    assert out_file == str(tmp_path / "vector_add.out")
    assert so_file == str(tmp_path / "vector_add.so")
    assert commands[0][0][:4] == ["hipcc", "-O2", "-std=c++17", "--offload-arch=gfx942"]
    assert commands[1][0][:3] == ["hipcc", "-shared", "-fPIC"]


def test_compile_hip_returns_stderr_on_failure(monkeypatch, tmp_path):
    def fake_run(cmd, capture_output, text, env, cwd):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="compile failed")

    monkeypatch.setattr("Magpie.utils.common.shutil.which", lambda _: "/usr/bin/hipcc")
    monkeypatch.setattr("Magpie.utils.common.subprocess.run", fake_run)

    out_file, so_file, errors = compile_hip(
        [str(tmp_path / "broken.hip")], str(tmp_path), "gfx942"
    )

    assert out_file is None
    assert so_file is None
    assert errors == "compile failed"


def test_compile_hip_uses_deterministic_name_for_multiple_sources(
    monkeypatch, tmp_path
):
    commands = []

    def fake_run(cmd, capture_output, text, env, cwd):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    source_files = [
        str(tmp_path / "part_a.hip"),
        str(tmp_path / "part_b.hip"),
    ]
    expected_stem = get_compilation_output_stem(source_files)

    monkeypatch.setattr("Magpie.utils.common.shutil.which", lambda _: "/usr/bin/hipcc")
    monkeypatch.setattr("Magpie.utils.common.subprocess.run", fake_run)

    out_file, so_file, errors = compile_hip(source_files, str(tmp_path), "gfx942")

    assert errors is None
    assert so_file is None
    assert out_file == str(tmp_path / f"{expected_stem}.out")
    assert commands[0][-1] == str(tmp_path / f"{expected_stem}.out")


def test_compile_cuda_uses_deterministic_name_for_multiple_sources(
    monkeypatch, tmp_path
):
    commands = []

    def fake_run(cmd, capture_output, text, env, cwd):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    source_files = [
        str(tmp_path / "part_a.cu"),
        str(tmp_path / "part_b.cu"),
    ]
    expected_stem = get_compilation_output_stem(source_files)
    compiler = Compiling(
        PipelineConfig(kernel_type=KernelType.CUDA, gpu_arch="sm_90")
    )
    kernel_cfg = KernelEvalConfig(
        kernel_id="cuda_kernel",
        kernel_type=KernelType.CUDA,
        source_file_path=source_files,
        working_dir=str(tmp_path),
    )

    monkeypatch.setattr("Magpie.eval.compiling.shutil.which", lambda _: "/usr/bin/nvcc")
    monkeypatch.setattr("Magpie.eval.compiling.subprocess.run", fake_run)

    result = compiler._compile_cuda(kernel_cfg)

    assert result.success is True
    assert result.output_file_path == str(tmp_path / f"{expected_stem}.out")
    assert commands[0][-1] == str(tmp_path / f"{expected_stem}.out")


def test_remote_task_helpers_manage_extra_args_and_envs():
    envs = {"EXTRA_VLLM_ARGS": "--tensor-parallel-size 1"}
    _ensure_extra_arg(envs, "EXTRA_VLLM_ARGS", "--distributed-executor-backend mp")
    _ensure_extra_arg(envs, "EXTRA_VLLM_ARGS", "--distributed-executor-backend mp")

    assert envs["EXTRA_VLLM_ARGS"].count("--distributed-executor-backend") == 1
    assert _extra_args_key("vllm") == "EXTRA_VLLM_ARGS"
    assert _extra_args_key("custom") == "EXTRA_CUSTOM_ARGS"

    bench_cfg = {"envs": {}}
    mode_config = {"benchmark_config": {}}
    _commit_envs(envs, bench_cfg, mode_config)

    assert bench_cfg["envs"] is envs
    assert mode_config["benchmark_config"] is bench_cfg


def test_clear_hidden_gpus_only_removes_empty_values(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "")

    _clear_hidden_gpus()

    assert "HIP_VISIBLE_DEVICES" not in os.environ
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert "ROCR_VISIBLE_DEVICES" not in os.environ


def test_configure_tp_isolation_switches_between_mp_and_ray(monkeypatch):
    monkeypatch.setattr("Magpie.remote.tasks._get_local_gpu_count", lambda: 4)

    monkeypatch.setenv("RAY_ADDRESS", "ray://cluster")
    mode_config = {
        "benchmark_config": {"framework": "vllm", "envs": {"TP": 2, "CONC": 8}}
    }
    ray_config = {}

    _configure_tp_isolation(mode_config, ray_config)

    assert "RAY_ADDRESS" not in os.environ
    assert (
        mode_config["benchmark_config"]["envs"]["EXTRA_VLLM_ARGS"]
        == "--distributed-executor-backend mp"
    )

    mode_config = {
        "benchmark_config": {"framework": "sglang", "envs": {"TP": 9, "CONC": 8}}
    }
    monkeypatch.setenv("RAY_ADDRESS", "ray://cluster")

    _configure_tp_isolation(mode_config, ray_config)

    assert os.environ["RAY_ADDRESS"] == "ray://cluster"
    assert (
        mode_config["benchmark_config"]["envs"]["EXTRA_SGLANG_ARGS"]
        == "--use-ray --nnodes 3"
    )
