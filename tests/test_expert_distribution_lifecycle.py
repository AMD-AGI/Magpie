"""Keep opt-in SGLang recording portable across reused-server samples."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from Magpie.modes.benchmark import benchmarker
from Magpie.modes.benchmark.benchmarker import BenchmarkMode
from Magpie.modes.benchmark.config import BenchmarkConfig
from Magpie.modes.benchmark.result import BenchmarkResult, ThroughputMetrics
from Magpie.utils.gpu import GPUVendor

ROOT = Path(__file__).resolve().parents[1]


def _mode(tmp_path, *, envs=None, framework="sglang", run_mode="local"):
    checkout = tmp_path / "InferenceX"
    (checkout / "benchmarks").mkdir(parents=True, exist_ok=True)
    (checkout / "benchmarks" / f"{framework}_mi355x.sh").touch()
    config = BenchmarkConfig(
        framework=framework,
        model="test/model",
        runner_type="mi355x",
        run_mode=run_mode,
        inferencex_path=str(checkout),
        envs=envs or {},
        profiler={"torch_profiler": {"enabled": False}},
    )
    return BenchmarkMode(config, output_dir=str(tmp_path / "results"))


@pytest.mark.parametrize("runtime", ["local", "docker"])
def test_recorder_dir_is_stable_across_workspaces(tmp_path, monkeypatch, runtime):
    mode = _mode(
        tmp_path,
        envs={"MAGPIE_RECORD_EXPERT_DISTRIBUTION": 1, "PORT": 8991},
        run_mode=runtime,
    )
    monkeypatch.setattr(benchmarker, "detect_gpu", lambda: (GPUVendor.AMD, "gfx950"))
    environments = []
    for workspace_name, phase in [("first", "server"), ("second", "client")]:
        workspace = tmp_path / workspace_name
        if runtime == "local":
            _, env = mode._build_local_command(workspace, "mi355x", phase=phase)
        else:
            cmd = mode._build_docker_command("test-image", workspace, "mi355x", phase=phase)
            env = dict(cmd[i + 1].split("=", 1) for i, arg in enumerate(cmd[:-1]) if arg == "-e")
            assert f"{mode.config.inferencex_path}:/opt/InferenceX" in cmd
        environments.append(env)
    first, second = environments
    directory = first["SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR"]
    assert directory == second["SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR"]
    assert "/.magpie-expert-distribution/" in directory
    assert directory.endswith(f"_{runtime}_8991")
    assert "--expert-distribution-recorder-mode stat" in first["EXTRA_SGLANG_ARGS"]
    assert "--expert-distribution-recorder-buffer-size 1" in first["EXTRA_SGLANG_ARGS"]
    assert "--enable-eplb" not in first["EXTRA_SGLANG_ARGS"]


def test_disabled_recorder_does_not_change_launch_or_report(tmp_path):
    mode = _mode(tmp_path)
    env = mode.config.get_env_vars()
    mode._set_expert_distribution_env(env, mode.config.inferencex_path)
    assert "EXTRA_SGLANG_ARGS" not in env
    assert "SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR" not in env
    assert "expert_load_distribution" not in BenchmarkResult().to_dict()


@pytest.mark.parametrize(
    "args",
    ["--enable-eplb", "--enable-eplb=true", "--expert-distribution-recorder-mode per_pass"],
)
def test_incompatible_recorder_options_fail_explicitly(tmp_path, args):
    mode = _mode(
        tmp_path,
        envs={"MAGPIE_RECORD_EXPERT_DISTRIBUTION": 1, "EXTRA_SGLANG_ARGS": args},
    )
    with pytest.raises(ValueError):
        mode._build_local_command(tmp_path / "workspace", "mi355x")


@pytest.mark.parametrize(
    "function",
    ["magpie_run_benchmark_serving", "magpie_run_benchmark_serving_remote_direct"],
)
@pytest.mark.parametrize(
    ("start_rc", "finish_rc", "client_exit", "expected_rc"),
    [
        (0, 0, "return 0", 0),
        (23, 0, "return 0", 23),
        (0, 29, "return 0", 29),
        (0, 0, "return 17", 17),
        (0, 29, "return 17", 17),
        (0, 0, "sh -c 'kill -TERM \"$PPID\"'; return 0", 143),
        (0, 29, "sh -c 'kill -INT \"$PPID\"'; return 0", 130),
    ],
)
def test_wrapper_stops_before_eval_and_preserves_client_exit(
    tmp_path, function, start_rc, finish_rc, client_exit, expected_rc
):
    compat = tmp_path / "magpie_bench_remote_compat.sh"
    shutil.copy2(ROOT / "Magpie/scripts/benchmark/magpie_bench_remote_compat.sh", compat)
    (tmp_path / "expert_distribution.py").write_text(
        "import os, sys\n"
        "with open(os.environ['SEQUENCE'], 'a') as out: out.write(sys.argv[-1] + '\\n')\n"
        "raise SystemExit(int(os.environ['START_RC'] if sys.argv[-1] == 'start' else os.environ['FINISH_RC']))\n"
    )
    native_python = tmp_path / "bin" / "python3"
    native_python.parent.mkdir()
    native_python.symlink_to(sys.executable)
    env = {
        **os.environ,
        "PATH": f"{native_python.parent}:{os.environ['PATH']}",
        "MAGPIE_RECORD_EXPERT_DISTRIBUTION": "1",
        "SEQUENCE": str(tmp_path / "sequence"),
        "START_RC": str(start_rc),
        "FINISH_RC": str(finish_rc),
        "COMPAT": str(compat),
    }
    command = f"""source "$COMPAT"
trap 'echo caller_cleanup >> "$SEQUENCE"' EXIT
_{function}() {{ echo client >> "$SEQUENCE"; {client_exit}; }}
{function} --example || exit $?
echo eval >> "$SEQUENCE"
"""
    result = subprocess.run(["bash", "-e", "-c", command], env=env, capture_output=True, text=True)
    assert result.returncode == expected_rc, result.stderr
    sequence = ["start"]
    if not start_rc:
        sequence += ["client", "finish"]
    if not expected_rc:
        sequence += ["eval"]
    assert (tmp_path / "sequence").read_text().splitlines() == [*sequence, "caller_cleanup"]


@pytest.mark.parametrize("sidecar", ["available", "missing", "malformed", "unavailable"])
@pytest.mark.parametrize("eval_only", [False, True])
@pytest.mark.parametrize("recording", [False, True])
def test_only_requested_capture_affects_success(tmp_path, monkeypatch, sidecar, eval_only, recording):
    mode = _mode(tmp_path, envs={"MAGPIE_RECORD_EXPERT_DISTRIBUTION": int(recording)})
    monkeypatch.setattr(benchmarker, "ensure_inferencex_available", lambda path: path)
    monkeypatch.setattr(mode, "_apply_gpu_selection", lambda: None)
    monkeypatch.setattr(mode, "_prepare_benchmark_scripts", lambda: None)
    monkeypatch.setattr(mode, "_get_benchmark_script", lambda runner: "benchmarks/run.sh")
    monkeypatch.setattr(mode, "_cleanup_server_processes", lambda framework: None)
    monkeypatch.setattr(mode, "_eval_only_accuracy_ok", lambda _: eval_only)
    stop_attempts = []

    def endpoint_unavailable(request, timeout):
        stop_attempts.append((request.full_url, request.method, timeout))
        raise OSError("server is unreachable")

    monkeypatch.setattr(benchmarker.urllib.request, "urlopen", endpoint_unavailable)
    metadata = {
        "status": "available",
        "scope": "client_including_warmup",
        "artifact_path": "expert_distribution.json",
        "rank_count": 2,
        "record_count": 4,
    }

    def execute(_cmd, _env, workspace):
        (workspace / "inferencex_result.json").write_text("{}")
        if sidecar != "missing":
            (workspace / "expert_load_distribution.json").write_text(
                "{broken" if sidecar == "malformed" else json.dumps(
                    metadata if sidecar == "available" else {"status": "unavailable"}
                )
            )
        return BenchmarkResult(success=True), "", ""

    monkeypatch.setattr(mode, "_execute_local_benchmark", execute)
    monkeypatch.setattr(
        benchmarker.ResultParser,
        "parse_inferencex_result",
        lambda *a, **kw: BenchmarkResult(
            success=True, throughput=ThroughputMetrics(output_throughput=100),
            benchmark_valid=True, publishable=True,
        ),
    )
    result = mode.run("expert-recording")
    assert result.success == (not recording or sidecar == "available")
    report = json.loads((Path(result.workspace_dir) / "benchmark_report.json").read_text())
    if not recording:
        assert result.expert_load_distribution is None
        assert "expert_load_distribution" not in report
        assert stop_attempts == []
    elif sidecar == "available":
        assert result.expert_load_distribution == report["expert_load_distribution"] == metadata
        assert stop_attempts == []
    else:
        assert result.expert_load_distribution is None
        assert "expert_load_distribution" not in report
        assert any("Expert capture failed" in error for error in result.errors)
        assert result.benchmark_valid is result.publishable is False
        assert stop_attempts == [
            ("http://127.0.0.1:8888/stop_expert_distribution_record", "POST", 3)
        ]


@pytest.mark.parametrize(
    "args",
    [
        "--expert-distribution-recorder-mode=stat --expert-distribution-recorder-buffer-size=1",
        "--expert-distribution-recorder-mode stat --expert-distribution-recorder-buffer-size 1",
    ],
)
def test_existing_bounded_recorder_options_are_not_duplicated(tmp_path, args):
    mode = _mode(tmp_path, envs={"MAGPIE_RECORD_EXPERT_DISTRIBUTION": 1, "EXTRA_SGLANG_ARGS": args})
    env = mode.config.get_env_vars()
    mode._set_expert_distribution_env(env, mode.config.inferencex_path)
    assert env["EXTRA_SGLANG_ARGS"] == args


@pytest.mark.parametrize(
    "args",
    [
        "--expert-distribution-recorder-mode stat --expert-distribution-recorder-mode per_pass",
        "--expert-distribution-recorder-buffer-size=64",
        "--expert-distribution-recorder-buffer-size",
    ],
)
def test_all_duplicate_and_malformed_recorder_options_are_validated(tmp_path, args):
    mode = _mode(tmp_path, envs={"MAGPIE_RECORD_EXPERT_DISTRIBUTION": 1, "EXTRA_SGLANG_ARGS": args})
    with pytest.raises(ValueError):
        mode._set_expert_distribution_env(mode.config.get_env_vars(), mode.config.inferencex_path)


def test_plugin_source_is_copied_with_inferencex_support_files(tmp_path):
    mode = _mode(tmp_path)
    mode._prepare_benchmark_scripts()
    plugin = Path(mode.config.inferencex_path) / "benchmarks" / "expert_distribution_plugin"
    source = ROOT / "Magpie" / "scripts" / "benchmark" / "expert_distribution_plugin"
    for name in ("pyproject.toml", "magpie_expert_totals.py"):
        assert (plugin / name).read_bytes() == (source / name).read_bytes()


@pytest.mark.parametrize("selector", [None, "", "other", "other,magpie_expert_totals"])
@pytest.mark.parametrize("install_rc", [0, 19])
def test_isolated_plugin_installer_is_offline_and_preserves_discovery(
    tmp_path, selector, install_rc
):
    fake_python = tmp_path / "bin" / "python3"
    fake_python.parent.mkdir()
    fake_python.write_text(
        '#!/usr/bin/env bash\nif [[ "$1" != "-m" ]]; then exit 0; fi\nprintf "%s\\n" "$@" > "$PIP_ARGS"\nexit "$INSTALL_RC"\n'
    )
    fake_python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_python.parent}:{os.environ['PATH']}",
        "TMPDIR": str(tmp_path),
        "MAGPIE_RECORD_EXPERT_DISTRIBUTION": "1",
        "PIP_ARGS": str(tmp_path / "pip_args"),
        "INSTALL_RC": str(install_rc),
        "PYTHONPATH": "/existing/pythonpath",
    }
    env.pop("SGLANG_PLUGINS", None)
    if selector is not None:
        env["SGLANG_PLUGINS"] = selector
    compat = ROOT / "Magpie/scripts/benchmark/magpie_bench_remote_compat.sh"
    command = f'''source "{compat}"
magpie_prepare_expert_distribution_plugin
"{sys.executable}" -c 'import json,os; print(json.dumps(dict(pythonpath=os.environ.get("PYTHONPATH"),plugins=os.environ.get("SGLANG_PLUGINS"))))'
'''
    result = subprocess.run(["bash", "-e", "-c", command], env=env, text=True, capture_output=True)
    assert result.returncode == (1 if install_rc else 0), result.stderr
    args = (tmp_path / "pip_args").read_text().splitlines()
    assert args[:3] == ["-m", "pip", "install"]
    assert all(
        flag in args for flag in ("--no-index", "--no-deps", "--no-build-isolation", "--target")
    )
    assert args[-1].endswith("/source")
    if install_rc:
        assert result.stdout == ""
        assert "plugin installation failed" in result.stderr
    else:
        installed = json.loads(result.stdout)
        target = args[args.index("--target") + 1]
        assert installed["pythonpath"] == target + ":/existing/pythonpath"
        expected = "other,magpie_expert_totals" if selector == "other" else selector
        assert installed["plugins"] == expected


def test_disabled_plugin_installer_does_not_invoke_pip_or_change_environment(tmp_path):
    compat = ROOT / "Magpie/scripts/benchmark/magpie_bench_remote_compat.sh"
    env = {
        **os.environ,
        "MAGPIE_RECORD_EXPERT_DISTRIBUTION": "0",
        "PYTHONPATH": "/existing/pythonpath",
        "SGLANG_PLUGINS": "other",
    }
    command = f'''source "{compat}"
python3() {{ echo unexpected_pip; return 23; }}
magpie_prepare_expert_distribution_plugin
printf '%s\\n' "$PYTHONPATH" "$SGLANG_PLUGINS"
'''
    result = subprocess.run(["bash", "-e", "-c", command], env=env, text=True, capture_output=True)
    assert result.returncode == 0
    assert result.stdout.splitlines() == ["/existing/pythonpath", "other"]


@pytest.mark.parametrize("script", ["sglang_mi300x.sh", "sglang_mi355x.sh", "sglang_radeon8060s.sh"])
def test_expert_setup_failure_prevents_server_launch(tmp_path, script):
    shutil.copy2(ROOT / "Magpie/scripts/benchmark" / script, tmp_path / script)
    (tmp_path / "benchmark_lib.sh").write_text("check_env_vars() { :; }\n")
    (tmp_path / "server_cleanup.sh").write_text("")
    (tmp_path / "magpie_bench_remote_compat.sh").write_text(
        "magpie_prepare_expert_distribution_plugin() { return 19; }\n"
    )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, body in {"hf": ":", "rocm-smi": "echo 'MEC 177'", "setsid": "exit 99"}.items():
        command = bindir / name
        command.write_text("#!/bin/bash\n" + body + "\n")
        command.chmod(0o755)
    pid_file = tmp_path / "server.pid"
    result = subprocess.run(
        ["bash", str(tmp_path / script)], capture_output=True, text=True, timeout=10,
        env={
            "PATH": f"{bindir}:/usr/bin:/bin", "MODEL": "test-model", "TP": "1",
            "PYTORCH_ROCM_ARCH": "gfx1151", "MAGPIE_RUN_PHASE": "server",
            "MAGPIE_RECORD_EXPERT_DISTRIBUTION": "1",
            "MAGPIE_SERVER_PID_FILE": str(pid_file), "RESULT_DIR": str(tmp_path),
        }, check=False,
    )
    assert result.returncode == 19, result.stderr
    assert not pid_file.exists()
    assert not (tmp_path / "server.log").exists()
