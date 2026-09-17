import csv
import gzip
import json
import io
import os
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from Magpie.modes.benchmark.benchmarker import BenchmarkMode
from Magpie.modes.benchmark.config import (
    BenchmarkConfig,
    GapAnalysisConfig,
    ProfilerConfig,
    RayConfig,
    ServerLifecycleConfig,
    TorchProfilerConfig,
    TraceLensConfig,
)
from Magpie.modes.benchmark.image_selector import ImageSelector
from Magpie.modes.benchmark.result import BenchmarkResult, ResultParser
from Magpie.modes.benchmark.tracelens_inference import (
    SGLANG_SHAPE_DISCOVERY_FLAG,
    TraceLensInferencePipeline,
    append_flag_value_args,
    compute_steady_state_iters,
    host_atom_supports_detailed_annotation,
    is_tracelens_patched_sglang_image,
    resolve_tl_extension,
    trace_arch_platform_from_runner,
)
from Magpie.modes.benchmark.tracelens_runtime import (
    available_tracelens_sglang_patch_versions,
    available_tracelens_vllm_patch_versions,
    derive_tracelens_extension_image_tag,
    derive_tracelens_image_tag,
    docker_image_probe,
    docker_image_package_version,
    infer_sglang_patch_version,
    infer_vllm_patch_version,
    inspect_tracelens_extension_wheel,
    is_tracelens_ready_runtime_image,
    prepare_tracelens_runtime_image,
    resolve_tracelens_repo_path,
    runner_type_to_gpu_type,
    vllm_profiler_options_are_upstream,
)
from Magpie.modes.benchmark.workspace import WorkspaceManager
from Magpie.utils.gpu import GPUVendor


@pytest.mark.parametrize("runtime", ["local", "docker"])
def test_benchmark_streaming_crosses_real_nested_processes(runtime, tmp_path):
    release = tmp_path / "release"
    observed = tmp_path / "observed"
    child = (
        "import os, pathlib, sys, time\n"
        "release, observed = map(pathlib.Path, sys.argv[1:])\n"
        "os.write(1, b'progress')\n"
        "os.write(2, b'warning')\n"
        "deadline = time.monotonic() + 3\n"
        "while not release.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        "observed.write_text(str(release.exists()))\n"
        "os.write(1, b' done')\n"
    )
    parent = (
        "import json, os, pathlib, sys\n"
        "from types import SimpleNamespace\n"
        "from Magpie.modes.benchmark.benchmarker import BenchmarkMode\n"
        "mode = BenchmarkMode.__new__(BenchmarkMode)\n"
        "mode.config = SimpleNamespace(timeout_seconds=10)\n"
        "mode._fix_workspace_ownership = lambda _: None\n"
        "runtime, workspace, child, release, observed = sys.argv[1:]\n"
        "workspace = pathlib.Path(workspace)\n"
        "cmd = [sys.executable, '-u', '-c', child, release, observed]\n"
        "if runtime == 'local':\n"
        "    result, out, err = mode._execute_local_benchmark(cmd, os.environ.copy(), workspace)\n"
        "else:\n"
        "    result, out, err = mode._execute_benchmark(cmd, workspace)\n"
        "(workspace / 'result.json').write_text(json.dumps([result.success, out, err]))\n"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            parent,
            runtime,
            str(tmp_path),
            child,
            str(release),
            str(observed),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=Path(__file__).resolve().parents[1],
    )
    captured = [bytearray(), bytearray()]
    arrived = [threading.Event(), threading.Event()]

    def observe(pipe, output, event):
        while chunk := pipe.read1(65536):
            output.extend(chunk)
            event.set()
        pipe.close()

    readers = [
        threading.Thread(target=observe, args=args, daemon=True)
        for args in zip((process.stdout, process.stderr), captured, arrived)
    ]
    try:
        for reader in readers:
            reader.start()
        assert all(event.wait(10) for event in arrived)
        assert process.poll() is None, "the Magpie parent exited before forwarding"
        release.write_text("continue")
        assert process.wait(timeout=10) == 0
        for reader in readers:
            reader.join(timeout=2)
        assert observed.read_text() == "True", "nested output stayed buffered"
        assert captured == [b"progress done", b"warning"]
        assert json.loads((tmp_path / "result.json").read_text()) == [
            True,
            "progress done",
            "warning",
        ]
    finally:
        release.write_text("continue")
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        for reader in readers:
            reader.join(timeout=5)


@pytest.mark.parametrize("runtime", ["local", "docker"])
@pytest.mark.parametrize("attach", [False, True])
def test_benchmark_streaming_reuse_server_and_client(
    runtime, attach, tmp_path, monkeypatch
):
    mode = BenchmarkMode.__new__(BenchmarkMode)
    mode.config = SimpleNamespace(
        timeout_seconds=10,
        framework="vllm",
        server_lifecycle=SimpleNamespace(
            cleanup=False, force_reuse=False, server_ready_timeout_s=1
        ),
    )
    mode._task_id = "stream-test"
    handle = tmp_path / "server.pid"
    meta = tmp_path / "server.json"
    monkeypatch.setattr(mode, "_get_benchmark_script", lambda _: "vllm_mi300x.sh")
    monkeypatch.setattr(mode, "_reuse_benchmark_port", lambda: 9000)
    monkeypatch.setattr(mode, "_reuse_server_paths", lambda _: (tmp_path, handle, meta))
    monkeypatch.setattr(mode, "_desired_reuse_server_meta", lambda *a, **k: {})
    monkeypatch.setattr(mode, "_reuse_http_healthy", lambda _: attach)
    monkeypatch.setattr(
        mode, "_reuse_read_meta", lambda _: {"container_id": "owned-container"}
    )
    monkeypatch.setattr(mode, "_reuse_meta_mismatch", lambda *a: None)
    monkeypatch.setattr(mode, "_reuse_clear_stale_artifacts", lambda *a: None)
    monkeypatch.setattr(mode, "_reuse_wait_health", lambda *a: True)
    monkeypatch.setattr(mode, "_reuse_wait_docker_health", lambda *a: True)
    monkeypatch.setattr(mode, "_reuse_write_meta", lambda *a: True)
    monkeypatch.setattr(mode, "_reuse_visible_devices", lambda: {})
    monkeypatch.setattr(mode, "_reuse_restore_visible_devices", lambda _: None)
    monkeypatch.setattr(mode, "_docker_container_running", lambda _: True)
    monkeypatch.setattr(mode, "_fix_workspace_ownership", lambda _: None)
    phases = []
    errors = io.StringIO()
    monkeypatch.setattr(
        mode, "_reuse_docker_container_name", lambda _: "owned-container"
    )

    def build(**kwargs):
        phase = kwargs["phase"]
        phases.append(phase)
        release = tmp_path / (phase + "-release")
        observed = tmp_path / (phase + "-observed")
        monkeypatch.setattr(sys, "stdout", _LiveBenchmarkOutput(release))
        monkeypatch.setattr(sys, "stderr", errors)
        output_by_phase[phase] = sys.stdout
        if runtime == "local" and phase == "server":
            kwargs["server_pid_file"].write_text("123")
        code = (
            "import os, pathlib, sys, time\n"
            "release, observed = map(pathlib.Path, sys.argv[1:])\n"
            f"os.write(1, b'{phase}-progress')\n"
            f"os.write(2, b'{phase}-warning')\n"
            "deadline = time.monotonic() + 1.5\n"
            "while not release.exists() and time.monotonic() < deadline:\n"
            "    time.sleep(0.01)\n"
            "observed.write_text(str(release.exists()))\n"
            "os.write(1, b' done\\n')\n"
        )
        cmd = [sys.executable, "-u", "-c", code, str(release), str(observed)]
        return (cmd, os.environ.copy()) if runtime == "local" else cmd

    output_by_phase = {}
    monkeypatch.setattr(mode, "_build_local_command", build)
    monkeypatch.setattr(mode, "_build_docker_command", build)
    if runtime == "local":
        result, stdout, stderr = mode._execute_local_benchmark_with_reuse(
            tmp_path, "mi300x"
        )
    else:
        result, stdout, stderr = mode._execute_docker_benchmark_with_reuse(
            tmp_path, "mi300x", "image:test"
        )
    assert result.success, result.errors
    assert phases == (["client"] if attach else ["server", "client"])
    for phase in phases:
        assert (tmp_path / (phase + "-observed")).read_text() == "True"
        assert output_by_phase[phase].getvalue() == phase + "-progress done\n"
        assert phase + "-progress done\n" in stdout
        assert phase + "-warning" in stderr
    assert errors.getvalue() == "".join(phase + "-warning" for phase in phases)
    assert (tmp_path / "benchmark_stdout.log").read_text() == stdout
    assert (tmp_path / "benchmark_stderr.log").read_text() == stderr


@pytest.mark.parametrize("runtime", ["local", "docker"])
@pytest.mark.parametrize("behavior", ["nonzero", "timeout"])
def test_benchmark_streaming_reuse_launch_failure_saves_logs(
    runtime, behavior, tmp_path, monkeypatch
):
    mode = BenchmarkMode.__new__(BenchmarkMode)
    mode.config = SimpleNamespace(
        framework="vllm",
        server_lifecycle=SimpleNamespace(server_ready_timeout_s=1),
    )
    monkeypatch.setattr(mode, "_get_benchmark_script", lambda _: "vllm_mi300x.sh")
    monkeypatch.setattr(mode, "_reuse_benchmark_port", lambda: 9000)
    monkeypatch.setattr(
        mode,
        "_reuse_server_paths",
        lambda _: (tmp_path, tmp_path / "server.pid", tmp_path / "server.json"),
    )
    monkeypatch.setattr(mode, "_desired_reuse_server_meta", lambda *a, **k: {})
    monkeypatch.setattr(mode, "_reuse_http_healthy", lambda _: False)
    monkeypatch.setattr(mode, "_reuse_clear_stale_artifacts", lambda *a: None)
    monkeypatch.setattr(
        mode, "_reuse_docker_container_name", lambda _: "owned-container"
    )
    code = (
        "import os, sys, time\n"
        "os.write(1, 'partial:\\u20ac'.encode('utf-8'))\n"
        "os.write(2, b'launch-error')\n"
        + ("time.sleep(10)\n" if behavior == "timeout" else "sys.exit(7)\n")
    )
    cmd = [sys.executable, "-u", "-c", code]
    monkeypatch.setattr(
        mode, "_build_local_command", lambda **k: (cmd, os.environ.copy())
    )
    monkeypatch.setattr(mode, "_build_docker_command", lambda **k: cmd)
    run = mode._run_streaming_command
    monkeypatch.setattr(
        mode,
        "_run_streaming_command",
        lambda cmd, timeout, env=None: run(cmd, timeout=0.5, env=env),
    )
    output, errors = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "stderr", errors)
    if runtime == "local":
        result, stdout, stderr = mode._execute_local_benchmark_with_reuse(
            tmp_path, "mi300x"
        )
    else:
        result, stdout, stderr = mode._execute_docker_benchmark_with_reuse(
            tmp_path, "mi300x", "image:test"
        )
    assert not result.success
    assert (
        "timed out" in result.errors[0]
        if behavior == "timeout"
        else "7" in result.errors[0]
    )
    assert stdout == output.getvalue() == "partial:€"
    assert stderr == errors.getvalue() == "launch-error"
    assert (tmp_path / "benchmark_stdout.log").read_text(encoding="utf-8") == stdout
    assert (tmp_path / "benchmark_stderr.log").read_text(encoding="utf-8") == stderr


class _LiveBenchmarkOutput(io.StringIO):
    def __init__(self, release):
        super().__init__()
        self.release = release
        self.flushed = threading.Event()

    def flush(self):
        self.flushed.set()
        self.release.write_text("continue", encoding="utf-8")
        super().flush()


@pytest.mark.parametrize("runtime", ["local", "docker"])
def test_benchmark_streams_without_newline_before_child_exit(
    runtime, tmp_path, monkeypatch
):
    release = tmp_path / "release"
    observed = tmp_path / "observed"
    output = _LiveBenchmarkOutput(release)
    errors = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "stderr", errors)
    mode = BenchmarkMode.__new__(BenchmarkMode)
    mode.config = SimpleNamespace(timeout_seconds=5)
    monkeypatch.setattr(mode, "_fix_workspace_ownership", lambda _: None)
    code = (
        "import os, pathlib, sys, time\n"
        "release, observed = map(pathlib.Path, sys.argv[1:])\n"
        "os.write(1, b'progress')\n"
        "os.write(2, b'warning')\n"
        "deadline = time.monotonic() + 1.5\n"
        "while not release.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        "observed.write_text(str(release.exists()))\n"
        "os.write(1, b' done')\n"
    )
    cmd = [sys.executable, "-u", "-c", code, str(release), str(observed)]
    if runtime == "local":
        result, stdout, stderr = mode._execute_local_benchmark(
            cmd, os.environ.copy(), tmp_path
        )
    else:
        result, stdout, stderr = mode._execute_benchmark(cmd, tmp_path)

    assert observed.read_text() == "True", "output stayed buffered until child exit"
    assert output.flushed.is_set()
    assert result.success
    assert stdout == output.getvalue() == "progress done"
    assert stderr == errors.getvalue() == "warning"
    assert (tmp_path / "benchmark_stdout.log").read_text() == stdout
    assert (tmp_path / "benchmark_stderr.log").read_text() == stderr


@pytest.mark.parametrize("binary_parent", [False, True])
def test_benchmark_streaming_preserves_split_utf8_and_environment(
    binary_parent, tmp_path, monkeypatch
):
    monkeypatch.setenv("MAGPIE_STREAM_TEST", "inherited")
    expected_out = "inherited\u20ac\U0001d11e done"
    expected_err = "warning:\u00e9"
    raw_out, raw_err = io.BytesIO(), io.BytesIO()
    output = (
        io.TextIOWrapper(raw_out, encoding="utf-8") if binary_parent else io.StringIO()
    )
    errors = (
        io.TextIOWrapper(raw_err, encoding="utf-8") if binary_parent else io.StringIO()
    )
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "stderr", errors)
    code = (
        "import os, time\n"
        "os.write(1, os.environ['MAGPIE_STREAM_TEST'].encode())\n"
        "for b in '\\u20ac\\U0001d11e done'.encode('utf-8'):\n"
        "    os.write(1, bytes([b]))\n"
        "    time.sleep(0.01)\n"
        "for b in 'warning:\\u00e9'.encode('utf-8'):\n"
        "    os.write(2, bytes([b]))\n"
        "    time.sleep(0.01)\n"
    )
    mode = BenchmarkMode.__new__(BenchmarkMode)
    mode.config = SimpleNamespace(timeout_seconds=5)
    result, stdout, stderr = mode._execute_local_benchmark(
        [sys.executable, "-u", "-c", code], os.environ.copy(), tmp_path
    )
    assert result.success
    assert stdout == expected_out
    assert stderr == expected_err
    if binary_parent:
        assert raw_out.getvalue().decode("utf-8") == expected_out
        assert raw_err.getvalue().decode("utf-8") == expected_err
    else:
        assert output.getvalue() == expected_out
        assert errors.getvalue() == expected_err
    assert (tmp_path / "benchmark_stdout.log").read_text(encoding="utf-8") == stdout
    assert (tmp_path / "benchmark_stderr.log").read_text(encoding="utf-8") == stderr


def test_benchmark_streaming_forwards_incomplete_utf8_bytes(tmp_path, monkeypatch):
    release = tmp_path / "release"
    observed = tmp_path / "observed"

    class BinaryOutput(io.BytesIO):
        def flush(self):
            if self.getvalue() == b"\xe2":
                release.write_text("continue")
            super().flush()

    raw = BinaryOutput()
    output = io.TextIOWrapper(raw, encoding="utf-8")
    monkeypatch.setattr(sys, "stdout", output)
    code = (
        "import os, pathlib, sys, time\n"
        "release, observed = map(pathlib.Path, sys.argv[1:])\n"
        "os.write(1, b'\\xe2')\n"
        "deadline = time.monotonic() + 1.5\n"
        "while not release.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        "observed.write_text(str(release.exists()))\n"
        "os.write(1, b'\\x82\\xac')\n"
    )
    process = BenchmarkMode._run_streaming_command(
        [sys.executable, "-u", "-c", code, str(release), str(observed)], timeout=5
    )
    assert observed.read_text() == "True"
    assert process.stdout == "\u20ac"
    assert raw.getvalue() == "\u20ac".encode("utf-8")


@pytest.mark.parametrize("returncode", [0, 7])
@pytest.mark.parametrize("runtime", ["local", "docker"])
def test_benchmark_streaming_preserves_exit_contract(
    returncode, runtime, tmp_path, monkeypatch
):
    output, errors = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "stderr", errors)
    monkeypatch.setenv("MAGPIE_STREAM_TEST", "inherited")
    mode = BenchmarkMode.__new__(BenchmarkMode)
    mode.config = SimpleNamespace(timeout_seconds=5)
    monkeypatch.setattr(mode, "_fix_workspace_ownership", lambda _: None)
    code = (
        "import os, sys\n"
        "for _ in range(32):\n"
        "    os.write(1, b'o' * 8192)\n"
        "    os.write(2, b'e' * 8192)\n"
        "os.write(1, b'\\r\\n' + os.environ['MAGPIE_STREAM_TEST'].encode())\n"
        f"sys.exit({returncode})\n"
    )
    cmd = [sys.executable, "-u", "-c", code]
    if runtime == "local":
        result, stdout, stderr = mode._execute_local_benchmark(
            cmd, os.environ.copy(), tmp_path
        )
        prefix = "Benchmark process"
    else:
        result, stdout, stderr = mode._execute_benchmark(cmd, tmp_path)
        prefix = "Docker command"
    assert result.success == (returncode == 0)
    assert stdout == "o" * (32 * 8192) + "\ninherited"
    assert stderr == "e" * (32 * 8192)
    assert output.getvalue().replace("\r\n", "\n") == stdout
    assert errors.getvalue() == stderr
    if returncode:
        assert result.errors == [
            f"{prefix} failed with code {returncode}",
            f"stderr: {stderr[:1000]}",
        ]
    else:
        assert not result.errors


@pytest.mark.parametrize("runtime", ["local", "docker"])
def test_benchmark_streaming_timeout_preserves_partial_logs(
    runtime, tmp_path, monkeypatch
):
    output, errors = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "stderr", errors)
    mode = BenchmarkMode.__new__(BenchmarkMode)
    mode.config = SimpleNamespace(timeout_seconds=0.5 if runtime == "docker" else 9)
    mode._task_id = "cpu-test"
    stopped = []
    real_run = subprocess.run

    def fake_docker_stop(cmd, **kwargs):
        if cmd[0] == "docker":
            stopped.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_docker_stop)
    code = (
        "import os, time\n"
        "os.write(1, 'partial:\\u20ac'.encode('utf-8'))\n"
        "os.write(2, b'partial-error')\n"
        "time.sleep(5)\n"
    )
    cmd = [sys.executable, "-u", "-c", code]
    started = time.monotonic()
    if runtime == "local":
        result, stdout, stderr = mode._execute_local_benchmark(
            cmd, os.environ.copy(), tmp_path, timeout_seconds=0.5
        )
    else:
        result, stdout, stderr = mode._execute_benchmark(cmd, tmp_path)
    # Windows allows five seconds for taskkill and two for bounded pipe drain,
    # in addition to the benchmark's own deadline.
    cleanup_bound = 8 if os.name == "nt" else 3
    assert time.monotonic() - started <= cleanup_bound
    assert not result.success
    assert result.errors == ["Benchmark timed out after 0.5s"]
    assert stdout == output.getvalue() == "partial:\u20ac"
    assert stderr == errors.getvalue() == "partial-error"
    assert (tmp_path / "benchmark_stdout.log").read_text(encoding="utf-8") == stdout
    assert (tmp_path / "benchmark_stderr.log").read_text(encoding="utf-8") == stderr
    assert stopped == (
        [["docker", "stop", "magpie-benchmark-cpu-test"]] if runtime == "docker" else []
    )


def test_benchmark_streaming_timeout_stops_only_owned_descendants(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    marker = tmp_path / "descendant-survived"
    diagnostics = []
    real_run = subprocess.run
    started = time.monotonic()

    def record_cleanup(cmd, **kwargs):
        cleanup_started = time.monotonic()
        result = real_run(cmd, **kwargs)
        diagnostics.append(
            {
                "command": cmd,
                "returncode": result.returncode,
                "duration": time.monotonic() - cleanup_started,
                "marker_after_cleanup": marker.exists(),
            }
        )
        return result

    monkeypatch.setattr(subprocess, "run", record_cleanup)
    descendant = (
        "import os, pathlib, sys, time\n"
        "os.write(2, b'descendant-ready')\n"
        "time.sleep(1.5)\n"
        "pathlib.Path(sys.argv[1]).write_text('still alive')\n"
    )
    code = (
        "import os, subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-u', '-c', sys.argv[1], sys.argv[2]])\n"
        "os.write(1, b'parent-ready')\n"
        "time.sleep(5)\n"
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with pytest.raises(subprocess.TimeoutExpired) as exc:
            BenchmarkMode._run_streaming_command(
                [sys.executable, "-u", "-c", code, descendant, str(marker)], timeout=0.5
            )
        diagnostics.append(
            {
                "elapsed_at_return": time.monotonic() - started,
                "marker_at_return": marker.exists(),
            }
        )
        assert exc.value.stdout == "parent-ready"
        assert exc.value.stderr == "descendant-ready"
        # Cleanup can outlast the worker's delay. Only post-return activity
        # violates the bounded-cleanup contract; an earlier marker does not.
        marker_at_return = marker.exists()
        time.sleep(1.6)
        assert marker.exists() == marker_at_return, diagnostics
        for entry in diagnostics:
            if entry.get("returncode", 0):
                assert f"exit code {entry['returncode']}" in caplog.text
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait()


def test_benchmark_streaming_timeout_stops_heartbeat_after_return(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    heartbeat = tmp_path / "heartbeat"
    stop = tmp_path / "stop"
    diagnostics = []
    real_run = subprocess.run

    def record_cleanup(cmd, **kwargs):
        started = time.monotonic()
        result = real_run(cmd, **kwargs)
        diagnostics.append(
            {
                "returncode": result.returncode,
                "duration": time.monotonic() - started,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        return result

    monkeypatch.setattr(subprocess, "run", record_cleanup)
    worker = (
        "import os, pathlib, sys, time\n"
        "heartbeat, stop = map(pathlib.Path, sys.argv[1:])\n"
        "os.write(2, b'heartbeat-ready')\n"
        "deadline = time.monotonic() + 10\n"
        "with heartbeat.open('ab', buffering=0) as handle:\n"
        "    while not stop.exists() and time.monotonic() < deadline:\n"
        "        handle.write(b'.')\n"
        "        time.sleep(0.02)\n"
    )
    code = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-u', '-c', *sys.argv[1:]])\n"
        "time.sleep(15)\n"
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(15)"])
    try:
        with pytest.raises(subprocess.TimeoutExpired) as exc:
            BenchmarkMode._run_streaming_command(
                [sys.executable, "-u", "-c", code, worker, str(heartbeat), str(stop)],
                timeout=0.5,
            )
        assert exc.value.stderr == "heartbeat-ready"
        count_at_return = heartbeat.stat().st_size
        assert count_at_return > 0
        time.sleep(0.3)
        assert heartbeat.stat().st_size == count_at_return, diagnostics
        assert unrelated.poll() is None
        if os.name == "nt":
            assert len(diagnostics) == 1
            if diagnostics[0]["returncode"]:
                assert f"exit code {diagnostics[0]['returncode']}" in caplog.text
        print("heartbeat cleanup:", diagnostics, file=sys.__stdout__, flush=True)
    finally:
        stop.write_text("stop")
        unrelated.kill()
        unrelated.wait()


def test_benchmark_streaming_reports_failed_windows_tree_cleanup(monkeypatch, caplog):
    from Magpie.modes.benchmark import benchmarker

    class FakeProcess:
        pid = 54321
        stdout = io.BytesIO(b"partial")
        stderr = io.BytesIO()
        returncode = None

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = 1

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired(["benchmark"], timeout)
            return self.returncode

    process = FakeProcess()
    monkeypatch.setattr(benchmarker, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(benchmarker.subprocess, "Popen", lambda *a, **kw: process)

    def fail_cleanup(cmd, **kwargs):
        assert cmd == ["taskkill", "/F", "/T", "/PID", "54321"]
        return subprocess.CompletedProcess(cmd, 255, b"", b"cleanup denied")

    monkeypatch.setattr(benchmarker.subprocess, "run", fail_cleanup)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    with pytest.raises(subprocess.TimeoutExpired) as exc:
        BenchmarkMode._run_streaming_command(["benchmark"], timeout=0.5)
    assert exc.value.stdout == "partial"
    assert "exit code 255" in caplog.text
    assert "cleanup denied" in caplog.text
    assert process.returncode == 1


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_benchmark_streaming_timeout_cleans_group_after_leader_exit(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    marker = tmp_path / "orphan-survived"
    worker = (
        "import pathlib, sys, time; time.sleep(1); "
        "pathlib.Path(sys.argv[1]).write_text('still alive')"
    )
    code = "import subprocess, sys; subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]])"
    with pytest.raises(subprocess.TimeoutExpired):
        BenchmarkMode._run_streaming_command(
            [sys.executable, "-c", code, worker, str(marker)], timeout=0.3
        )
    time.sleep(1.1)
    assert not marker.exists()


def test_benchmark_streaming_posix_timeout_scopes_cleanup(monkeypatch):
    from Magpie.modes.benchmark import benchmarker

    calls = []
    env = {"MAGPIE_STREAM_TEST": "explicit"}

    class FakeProcess:
        pid = 54321
        stdout = io.BytesIO(b"partial-output")
        stderr = io.BytesIO(b"partial-error")
        returncode = None

        def wait(self, timeout=None):
            if not calls:
                raise subprocess.TimeoutExpired(["benchmark"], timeout)
            self.returncode = -9
            return self.returncode

    def fake_popen(cmd, **kwargs):
        assert cmd == ["benchmark"]
        assert kwargs == {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "env": env,
            "start_new_session": True,
        }
        return FakeProcess()

    monkeypatch.setattr(benchmarker.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        benchmarker,
        "os",
        SimpleNamespace(name="posix", killpg=lambda pid, sig: calls.append((pid, sig))),
    )
    monkeypatch.setattr(benchmarker, "signal", SimpleNamespace(SIGKILL=9))
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    with pytest.raises(subprocess.TimeoutExpired) as exc:
        BenchmarkMode._run_streaming_command(["benchmark"], timeout=0.5, env=env)
    assert calls == [(54321, 9)]
    assert exc.value.timeout == 0.5
    assert exc.value.stdout == "partial-output"
    assert exc.value.stderr == "partial-error"


@pytest.mark.parametrize("target_state", ["closed", "missing"])
def test_benchmark_streaming_drains_when_parent_console_is_unavailable(
    target_state, monkeypatch
):
    output = io.StringIO()
    if target_state == "closed":
        output.close()
    else:
        output = None
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "stderr", output)
    process = BenchmarkMode._run_streaming_command(
        [
            sys.executable,
            "-c",
            "import os; os.write(1, b'o' * 100000); os.write(2, b'e' * 100000)",
        ],
        timeout=None,
    )
    assert process.returncode == 0
    assert process.stdout == "o" * 100000
    assert process.stderr == "e" * 100000


def test_benchmark_streaming_spawn_error_keeps_result_contract(tmp_path):
    mode = BenchmarkMode.__new__(BenchmarkMode)
    mode.config = SimpleNamespace(timeout_seconds=5)
    result, stdout, stderr = mode._execute_local_benchmark(
        [str(tmp_path / "missing-executable")], os.environ.copy(), tmp_path
    )
    assert not result.success
    assert stdout == stderr == ""
    assert len(result.errors) == 1
    assert result.errors[0].startswith("Benchmark execution error:")


def test_workspace_manager_makes_docker_mounts_container_writable(tmp_path):
    workspace = WorkspaceManager(
        base_dir=str(tmp_path),
        framework="vllm",
        container_writable=True,
    ).create()

    assert workspace.stat().st_mode & 0o777 == 0o777
    assert (workspace / "torch_trace").stat().st_mode & 0o777 == 0o777
    assert (workspace / "system_profile").stat().st_mode & 0o777 == 0o777


def test_benchmark_mode_only_requests_container_writable_workspace_for_docker(
    tmp_path,
):
    docker_mode = BenchmarkMode(
        BenchmarkConfig(framework="vllm", model="demo", run_mode="docker"),
        output_dir=str(tmp_path / "docker"),
    )
    local_mode = BenchmarkMode(
        BenchmarkConfig(framework="vllm", model="demo", run_mode="local"),
        output_dir=str(tmp_path / "local"),
    )

    assert docker_mode.workspace_mgr.container_writable is True
    assert local_mode.workspace_mgr.container_writable is False


def test_benchmark_server_lifecycle_supports_docker_but_rejects_ray():
    docker = BenchmarkConfig(
        framework="vllm",
        model="demo",
        run_mode="docker",
        envs={
            "TP": 1,
            "CONC": 32,
            "ISL": 1024,
            "OSL": 512,
            "RANDOM_RANGE_RATIO": 0.5,
        },
        profiler=ProfilerConfig(
            torch_profiler=TorchProfilerConfig(enabled=False),
        ),
        server_lifecycle=ServerLifecycleConfig(enabled=True),
    )
    assert docker.is_server_lifecycle is True

    with pytest.raises(ValueError, match="Ray executions cannot reuse"):
        BenchmarkConfig(
            framework="vllm",
            model="demo",
            run_mode="ray",
            profiler=ProfilerConfig(
                torch_profiler=TorchProfilerConfig(enabled=False),
            ),
            server_lifecycle=ServerLifecycleConfig(enabled=True),
        )


def test_benchmark_server_lifecycle_rejects_profiler_without_cleanup():
    with pytest.raises(ValueError, match="torch_profiler"):
        BenchmarkConfig(
            framework="vllm",
            model="demo",
            run_mode="local",
            envs={
                "TP": 1,
                "CONC": 32,
                "ISL": 1024,
                "OSL": 512,
                "RANDOM_RANGE_RATIO": 0.5,
            },
            profiler=ProfilerConfig(
                torch_profiler=TorchProfilerConfig(enabled=True),
            ),
            server_lifecycle=ServerLifecycleConfig(enabled=True, cleanup=False),
        )


def test_benchmark_server_lifecycle_rejects_remote_endpoint():
    with pytest.raises(ValueError, match="BENCHMARK_BASE_URL"):
        BenchmarkConfig(
            framework="sglang",
            model="demo",
            run_mode="docker",
            envs={"BENCHMARK_BASE_URL": "http://model-server:8888"},
            profiler=ProfilerConfig(
                torch_profiler=TorchProfilerConfig(enabled=False),
            ),
            server_lifecycle=ServerLifecycleConfig(enabled=True),
        )


def test_benchmark_server_lifecycle_sweep_conflict():
    with pytest.raises(ValueError, match="sweep_matrix"):
        BenchmarkConfig.from_dict(
            {
                "framework": "vllm",
                "model": "demo",
                "run_mode": "local",
                "profiler": {"torch_profiler": {"enabled": False}},
                "envs": {
                    "TP": 1,
                    "CONC": 32,
                    "ISL": 1024,
                    "OSL": 512,
                    "RANDOM_RANGE_RATIO": 0.5,
                },
                "server_lifecycle": {"enabled": True},
                "sweep_matrix": {"cases": [{"CONC": 2}]},
            }
        )


def test_benchmark_server_lifecycle_from_dict_ok():
    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "vllm",
            "model": "demo",
            "run_mode": "local",
            "profiler": {"torch_profiler": {"enabled": False}},
            "envs": {"PORT": 8899},
            "server_lifecycle": {"enabled": True},
        }
    )

    assert cfg.is_server_lifecycle is True


def test_tracelens_config_supports_legacy_export_flags():
    cfg = TraceLensConfig.from_dict({"enabled": True, "export_excel": True})

    assert cfg.enabled is True
    assert cfg.analysis_mode == "inference"
    assert cfg.analysis_stages == ["prefilldecode", "decode", "prefill"]
    assert cfg.cli_timeout_seconds == 1800
    assert cfg.auto_patch_runtime is True
    assert cfg.export_format == "excel"
    assert cfg.export_excel is True
    assert cfg.export_csv is False


def test_tracelens_config_round_trips_extension_wheel_path():
    cfg = TraceLensConfig.from_dict(
        {
            "enabled": True,
            "extension_wheel_path": "/secure/extensions/custom.whl",
        }
    )

    assert cfg.extension_wheel_path == "/secure/extensions/custom.whl"
    assert cfg.to_dict()["extension_wheel_path"] == ("/secure/extensions/custom.whl")


def test_tracelens_config_normalizes_inference_stages():
    cfg = TraceLensConfig.from_dict(
        {
            "enabled": True,
            "analysis_stages": ["decode", "mixed", "prefill"],
            "cli_timeout_seconds": 2400,
            "auto_patch_runtime": False,
        }
    )

    assert cfg.analysis_stages == ["decode", "prefilldecode", "prefill"]
    assert cfg.cli_timeout_seconds == 2400
    assert cfg.auto_patch_runtime is False

    cfg = TraceLensConfig.from_dict(
        {"enabled": True, "analysis_mode": "classic", "analysis_stages": "all"}
    )

    assert cfg.analysis_mode == "pytorch"
    assert cfg.analysis_stages == ["prefilldecode", "decode", "prefill"]


def test_tracelens_inference_iteration_and_arg_helpers():
    max_iters, delay_iters = compute_steady_state_iters(1024, 64, 1)

    assert max_iters == 256
    assert delay_iters == 6016

    envs = {"EXTRA_VLLM_ARGS": "--existing true"}
    append_flag_value_args(
        envs,
        "EXTRA_VLLM_ARGS",
        [
            ("--existing", "false"),
            ("--new-flag", "value"),
        ],
    )

    assert envs["EXTRA_VLLM_ARGS"] == "--existing true --new-flag value"
    assert trace_arch_platform_from_runner("mi355x") == "MI355X"
    assert trace_arch_platform_from_runner("mi300") == "MI300X"
    assert is_tracelens_patched_sglang_image("tracelens-sglang:0.5.12")
    assert not is_tracelens_patched_sglang_image("lmsysorg/sglang:latest")

    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "sglang",
            "model": "demo",
            "envs": {"TL_EXTENSION": "TraceLens_NDA"},
        }
    )
    assert not TraceLensInferencePipeline(cfg)._should_enable_sglang_shape_discovery()


def test_host_atom_detailed_annotation_detection(tmp_path, monkeypatch):
    atom_package = tmp_path / "atom"
    envs_file = atom_package / "utils" / "envs.py"
    envs_file.parent.mkdir(parents=True)
    envs_file.write_text("ATOM_ENABLE_DETAILED_ANNOTATION = False\n")
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_inference.importlib.util.find_spec",
        lambda _name: SimpleNamespace(origin=str(atom_package / "__init__.py")),
    )

    assert host_atom_supports_detailed_annotation() is True


def test_host_atom_detailed_annotation_detection_handles_missing_install(monkeypatch):
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_inference.importlib.util.find_spec",
        lambda _name: None,
    )

    assert host_atom_supports_detailed_annotation() is False


def test_tracelens_auto_selects_only_a_supported_gpu_platform():
    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "sglang",
            "model": "demo",
            "profiler": {"tracelens": {"enabled": True}},
        }
    )
    pipeline = TraceLensInferencePipeline(cfg)

    candidate, selected, warning = pipeline._select_gpu_arch_platform(
        "mi355x",
        lambda _candidate: subprocess.CompletedProcess(
            [], 0, "available=MI300X,MI355X\n", ""
        ),
        "test TraceLens",
    )

    assert candidate == "MI355X"
    assert selected == "MI355X"
    assert warning is None

    candidate, selected, warning = pipeline._select_gpu_arch_platform(
        "mi355x",
        lambda _candidate: subprocess.CompletedProcess(
            [], 3, "available=MI300X,MI325X\n", ""
        ),
        "test TraceLens",
    )

    assert candidate == "MI355X"
    assert selected is None
    assert "MI355X is not supported" in warning
    assert "available=MI300X,MI325X" in warning


def test_tracelens_explicit_gpu_arch_config_skips_platform_probe(tmp_path):
    arch_config = tmp_path / "mi355x.json"
    arch_config.write_text("{}", encoding="utf-8")
    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "sglang",
            "model": "demo",
            "profiler": {
                "tracelens": {
                    "enabled": True,
                    "gpu_arch_config": str(arch_config),
                }
            },
        }
    )

    def unexpected_probe(_candidate):
        raise AssertionError("platform probe should not run")

    candidate, selected, warning = TraceLensInferencePipeline(
        cfg
    )._select_gpu_arch_platform("mi355x", unexpected_probe, "test TraceLens")

    assert candidate == "MI355X"
    assert selected is None
    assert warning is None


def test_tracelens_container_platform_probe_uses_extension(
    tmp_path,
    monkeypatch,
):
    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "sglang",
            "model": "demo",
            "run_mode": "docker",
            "envs": {"TL_EXTENSION": "TraceLens_NDA"},
            "profiler": {"tracelens": {"enabled": True}},
        }
    )
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, "available=MI300X,MI355X\n", "")

    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_inference.subprocess.run",
        fake_run,
    )

    proc = TraceLensInferencePipeline(cfg)._probe_container_gpu_arch_platform(
        "tracelens-sglang:test",
        tmp_path,
        "MI355X",
    )

    assert proc.returncode == 0
    assert len(commands) == 1
    cmd, kwargs = commands[0]
    assert "TL_EXTENSION=TraceLens_NDA" in cmd
    assert "MI355X" in cmd[-1]
    assert "list_platforms" in cmd[-1]
    assert kwargs["timeout"] == 60


def test_resolve_tl_extension_merges_host_and_config(monkeypatch):
    monkeypatch.setenv("TL_EXTENSION", "HostExtension:SharedExtension")

    assert (
        resolve_tl_extension({"TL_EXTENSION": "ConfigExtension:SharedExtension"})
        == "HostExtension:SharedExtension:ConfigExtension"
    )


def test_tracelens_runtime_image_helpers():
    assert (
        infer_sglang_patch_version("lmsysorg/sglang:v0.5.12-rocm720-mi35x") == "0.5.12"
    )
    assert (
        infer_sglang_patch_version("lmsysorg/sglang:v0.5.13-rocm720-mi35x") == "0.5.13"
    )
    assert (
        infer_sglang_patch_version(
            "internal/sglang:rocm",
            installed_version="0.5.13+rocm720",
        )
        == "0.5.13"
    )
    assert infer_sglang_patch_version("lmsysorg/sglang:latest") is None
    assert infer_vllm_patch_version("vllm/vllm-openai-rocm:v0.19.1") == "v19"
    assert infer_vllm_patch_version("vllm/vllm-openai-rocm:v0.23.0") == "v23"
    assert (
        infer_vllm_patch_version(
            "internal/vllm:rocm",
            installed_version="0.22.0+rocm722",
        )
        == "v22"
    )
    assert infer_vllm_patch_version("vllm/vllm-openai-rocm:nightly") is None
    assert runner_type_to_gpu_type("mi355x") == "mi355"
    assert is_tracelens_ready_runtime_image("sglang", "tracelens-sglang:0.5.12")
    assert not is_tracelens_ready_runtime_image("vllm", "lmsysorg/sglang:v0.5.12")

    tag = derive_tracelens_image_tag(
        "sglang",
        "lmsysorg/sglang:v0.5.12-rocm720-mi35x",
        "mi355x",
        "0.5.12",
    )
    assert tag.startswith("magpie-tracelens-sglang:0_5_12-mi355x-")
    assert derive_tracelens_extension_image_tag(
        tag,
        "abcdef0123456789",
    ).endswith("-ext-abcdef012345")


def test_inspect_tracelens_extension_wheel_infers_module_and_normalizes_name(
    tmp_path,
):
    wheel = tmp_path / "TraceLens_Ext-0.1.0.dev20260529+gacb7fbc6-py3-none-any 1.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("TraceLens_Ext/__init__.py", "")
        archive.writestr(
            "TraceLens_Ext/Agent/Analysis/utils/arch/ExampleGPU.json",
            '{"name": "ExampleGPU"}',
        )
        archive.writestr(
            "TraceLens_Ext/Agent/Analysis/utils/agent_extension.py",
            "",
        )

    inspected = inspect_tracelens_extension_wheel(wheel)

    assert inspected["module"] == "TraceLens_Ext"
    assert inspected["filename"] == (
        "TraceLens_Ext-0.1.0.dev20260529+gacb7fbc6-py3-none-any.whl"
    )
    assert len(inspected["sha256"]) == 64


def test_inspect_tracelens_extension_wheel_rejects_ambiguous_packages(tmp_path):
    wheel = tmp_path / "ambiguous-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for module in ("ExtensionOne", "ExtensionTwo"):
            archive.writestr(
                f"{module}/Agent/Analysis/utils/arch/ExampleGPU.json",
                "{}",
            )

    with pytest.raises(ValueError, match="exactly one top-level"):
        inspect_tracelens_extension_wheel(wheel)


def test_sglang_runtime_support_is_read_from_tracelens_checkout(tmp_path):
    tracelens_repo = tmp_path / "TraceLens"
    workflow_dir = tracelens_repo / "examples/custom_workflows/inference_analysis"
    patch_root = workflow_dir / "sglang_roofline_patches"
    (patch_root / "sglang_0_5_13").mkdir(parents=True)
    (patch_root / "sglang_0_5_14").mkdir(parents=True)
    (workflow_dir / "build_docker_sglang.sh").write_text(
        "normalize_version() {\n"
        '    case "$1" in\n'
        "        0.5.12|v0512|0512|5.12)\n"
        '            echo "0.5.12"\n'
        "            ;;\n"
        "        0.5.13|v0513|0513|5.13)\n"
        '            echo "0.5.13"\n'
        "            ;;\n"
        "    esac\n"
        "}\n"
    )

    assert available_tracelens_sglang_patch_versions(tracelens_repo) == ["0.5.13"]
    assert (
        infer_sglang_patch_version(
            "lmsysorg/sglang:v0.5.13-rocm720-mi35x",
            tracelens_repo,
        )
        == "0.5.13"
    )
    assert (
        infer_sglang_patch_version(
            "lmsysorg/sglang:v0.5.14-rocm720-mi35x",
            tracelens_repo,
        )
        is None
    )


def test_vllm_runtime_support_is_read_from_tracelens_checkout(tmp_path):
    tracelens_repo = tmp_path / "TraceLens"
    workflow_dir = tracelens_repo / "examples/custom_workflows/inference_analysis"
    patch_dir = workflow_dir / "vllm_patches"
    patch_dir.mkdir(parents=True)
    (workflow_dir / "build_docker_vllm.sh").write_text(
        "case ${VLLM_VERSION} in\n"
        "    v22)\n"
        "        ;;\n"
        "    v23)\n"
        "        ;;\n"
        "esac\n"
    )
    (patch_dir / "config_vllm_v0.23.0.patch").write_text("patch")
    (patch_dir / "config_vllm_v0.24.0.patch").write_text("patch")

    assert available_tracelens_vllm_patch_versions(tracelens_repo) == ["v23"]
    assert (
        infer_vllm_patch_version(
            "vllm/vllm-openai-rocm:v0.23.0",
            tracelens_repo,
        )
        == "v23"
    )
    assert (
        infer_vllm_patch_version(
            "vllm/vllm-openai-rocm:v0.24.0",
            tracelens_repo,
        )
        is None
    )


def test_vllm_profiler_options_are_upstream_from_v026():
    assert vllm_profiler_options_are_upstream("vllm/vllm-openai-rocm:v0.26.0")
    assert vllm_profiler_options_are_upstream("vllm/vllm-openai-rocm:v0.27.1")
    assert vllm_profiler_options_are_upstream(
        "internal/vllm:rocm",
        installed_version="0.26.1+rocm722",
    )
    assert not vllm_profiler_options_are_upstream("vllm/vllm-openai-rocm:v0.25.0")
    assert not vllm_profiler_options_are_upstream("vllm/vllm-openai-rocm:nightly")


def _upstream_vllm_config(tmp_path, monkeypatch, *, has_tracelens):
    """Build a v0.26 vLLM config, which needs no TraceLens framework patch."""
    tracelens_repo = tmp_path / "TraceLens"
    (tracelens_repo / "examples/custom_workflows/inference_analysis").mkdir(
        parents=True
    )
    monkeypatch.setenv("TRACELENS_REPO_PATH", str(tracelens_repo))
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.docker_image_package_version",
        lambda _image, _package: None,
    )
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.docker_image_probe",
        lambda _image, _script: has_tracelens,
    )
    return BenchmarkConfig.from_dict(
        {
            "framework": "vllm",
            "model": "demo",
            "docker_image": "vllm/vllm-openai-rocm:v0.26.0",
            "profiler": {"tracelens": {"enabled": True}},
        }
    )


def test_prepare_tracelens_runtime_image_skips_patch_for_upstream_vllm(
    tmp_path,
    monkeypatch,
):
    cfg = _upstream_vllm_config(tmp_path, monkeypatch, has_tracelens=True)

    result = prepare_tracelens_runtime_image(
        cfg,
        base_image=cfg.docker_image,
        runner_type="mi355x",
    )

    assert result["built"] is False
    assert result["framework_patch_required"] is False
    assert result["image"] == "vllm/vllm-openai-rocm:v0.26.0"
    assert result["reason"] == (
        "no framework patch needed; image already has TraceLens"
    )


def test_prepare_tracelens_runtime_image_installs_tracelens_when_missing(
    tmp_path,
    monkeypatch,
):
    cfg = _upstream_vllm_config(tmp_path, monkeypatch, has_tracelens=False)
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.docker_image_exists",
        lambda _image: False,
    )
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["dockerfile"] = kwargs.get("input")
        return subprocess.CompletedProcess(cmd, 0, stdout="built")

    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.subprocess.run",
        fake_run,
    )

    result = prepare_tracelens_runtime_image(
        cfg,
        base_image=cfg.docker_image,
        runner_type="mi355x",
    )

    assert result["built"] is True
    assert result["framework_patch_required"] is False
    assert result["image"].startswith("magpie-tracelens-vllm:tlonly-mi355x-")
    assert seen["cmd"][:4] == ["docker", "build", "-f", "-"]
    assert "FROM vllm/vllm-openai-rocm:v0.26.0" in seen["dockerfile"]
    assert "pip install --no-cache-dir /tmp/TraceLens" in seen["dockerfile"]


def test_prepare_tracelens_runtime_image_honors_custom_overlay_tag(
    tmp_path, monkeypatch
):
    cfg = _upstream_vllm_config(tmp_path, monkeypatch, has_tracelens=False)
    cfg.profiler.tracelens.runtime_patch_image_tag = "example/custom-tracelens:latest"
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.docker_image_exists",
        lambda image: image == "example/custom-tracelens:latest",
    )

    result = prepare_tracelens_runtime_image(
        cfg,
        base_image=cfg.docker_image,
        runner_type="mi355x",
    )

    assert result["image"] == "example/custom-tracelens:latest"
    assert result["built"] is False


def test_tracelens_inference_prepare_sets_vllm_capture_torch_profiler_flag(tmp_path):
    inferencex = tmp_path / "InferenceX"
    bench_dir = inferencex / "benchmarks"
    serving_dir = inferencex / "utils" / "bench_serving"
    bench_dir.mkdir(parents=True)
    serving_dir.mkdir(parents=True)
    (bench_dir / "benchmark_lib.sh").write_text(
        'if [[ "${PROFILE:-}" == "1" ]]; then\n'
        '    num_prompts="$max_concurrency"\n'
        "fi\n",
        encoding="utf-8",
    )

    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "vllm",
            "model": "demo",
            "inferencex_path": str(inferencex),
            "envs": {"CONC": 64, "OSL": 1024, "RANDOM_RANGE_RATIO": 1},
            "profiler": {"tracelens": {"enabled": True}},
        }
    )

    TraceLensInferencePipeline(cfg).prepare(tmp_path / "workspace")

    extra_args = cfg.envs["EXTRA_VLLM_ARGS"]
    assert "--profiler-config.capture_torch_profiler True" in extra_args
    assert "capture_torch_profiler_dir" not in extra_args
    assert "--profiler-config.detailed_trace_annotation True" in extra_args


def test_prepare_tracelens_runtime_image_probes_atom_graph_capture(
    tmp_path,
    monkeypatch,
):
    tracelens_repo = tmp_path / "TraceLens"
    (tracelens_repo / "examples/custom_workflows/inference_analysis").mkdir(
        parents=True
    )
    monkeypatch.setenv("TRACELENS_REPO_PATH", str(tracelens_repo))
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.docker_image_probe",
        lambda _image, script: "ATOM_ENABLE_DETAILED_ANNOTATION" in script,
    )
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.docker_image_exists",
        lambda _image: False,
    )
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="built"),
    )
    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "atom",
            "model": "demo",
            "docker_image": "rocm/atom:nightly-20260801",
            "profiler": {"tracelens": {"enabled": True}},
        }
    )

    result = prepare_tracelens_runtime_image(
        cfg,
        base_image=cfg.docker_image,
        runner_type="mi355x",
    )

    assert result["atom_detailed_annotation"] is True
    assert result["framework_patch_required"] is False
    assert result["image"].startswith("magpie-tracelens-atom:tlonly-mi355x-")


def _atom_config(tmp_path):
    inferencex = tmp_path / "InferenceX"
    bench_dir = inferencex / "benchmarks"
    bench_dir.mkdir(parents=True)
    (bench_dir / "benchmark_lib.sh").write_text(
        'num_prompts="$max_concurrency"\n',
        encoding="utf-8",
    )
    return BenchmarkConfig.from_dict(
        {
            "framework": "atom",
            "model": "demo",
            "inferencex_path": str(inferencex),
            "envs": {"CONC": 32, "OSL": 512},
            "profiler": {"tracelens": {"enabled": True}},
        }
    )


def test_tracelens_inference_prepare_enables_atom_graph_capture(tmp_path):
    cfg = _atom_config(tmp_path)

    result = TraceLensInferencePipeline(cfg).prepare(
        tmp_path / "workspace",
        runtime={"atom_detailed_annotation": True},
    )

    assert cfg.envs["ATOM_PROFILER_MORE"] == "1"
    assert cfg.envs["ATOM_ENABLE_DETAILED_ANNOTATION"] == "1"
    assert "--mark-trace" in cfg.envs["EXTRA_ATOM_ARGS"]
    # ATOM uses its own runner, so InferenceX benchmark_lib.sh must stay untouched.
    assert result["patched_files"] == []
    bench_lib = Path(cfg.inferencex_path) / "benchmarks" / "benchmark_lib.sh"
    assert bench_lib.read_text(encoding="utf-8") == 'num_prompts="$max_concurrency"\n'


def test_tracelens_inference_prepare_warns_for_atom_without_graph_capture(tmp_path):
    cfg = _atom_config(tmp_path)

    result = TraceLensInferencePipeline(cfg).prepare(
        tmp_path / "workspace",
        runtime={"atom_detailed_annotation": False},
    )

    assert cfg.envs["ATOM_PROFILER_MORE"] == "1"
    assert "ATOM_ENABLE_DETAILED_ANNOTATION" not in cfg.envs
    # --mark-trace is stock ATOM, so it stays even without the annotation patch.
    assert "--mark-trace" in cfg.envs["EXTRA_ATOM_ARGS"]
    assert any("2026-07-22" in warning for warning in result["warnings"])


def test_tracelens_inference_prepare_does_not_duplicate_atom_mark_trace(tmp_path):
    cfg = _atom_config(tmp_path)
    cfg.envs["EXTRA_ATOM_ARGS"] = "--kv_cache_dtype fp8 --mark-trace"

    TraceLensInferencePipeline(cfg).prepare(
        tmp_path / "workspace",
        runtime={"atom_detailed_annotation": True},
    )

    assert cfg.envs["EXTRA_ATOM_ARGS"].count("--mark-trace") == 1


def test_tracelens_inference_prepare_probes_local_atom_when_runtime_unknown(
    tmp_path, monkeypatch
):
    cfg = _atom_config(tmp_path)
    cfg.run_mode = "local"
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_inference.host_atom_supports_detailed_annotation",
        lambda: True,
    )

    TraceLensInferencePipeline(cfg).prepare(tmp_path / "workspace")

    assert cfg.envs["ATOM_ENABLE_DETAILED_ANNOTATION"] == "1"


def test_tracelens_inference_prepare_probes_docker_atom_when_runtime_unknown(
    tmp_path, monkeypatch
):
    cfg = _atom_config(tmp_path)
    cfg.docker_image = "example/atom:latest"
    seen = {}

    def probe(image, script):
        seen.update(image=image, script=script)
        return True

    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_inference.docker_image_probe",
        probe,
    )

    TraceLensInferencePipeline(cfg).prepare(tmp_path / "workspace")

    assert seen["image"] == "example/atom:latest"
    assert "ATOM_ENABLE_DETAILED_ANNOTATION" in seen["script"]
    assert cfg.envs["ATOM_ENABLE_DETAILED_ANNOTATION"] == "1"


def test_tracelens_inference_resolves_atom_rank0_trace_and_capture_folder(tmp_path):
    torch_trace = tmp_path / "torch_trace"
    for rank in range(2):
        capture = torch_trace / f"rank_{rank}" / "capture_traces"
        capture.mkdir(parents=True)
        (capture / f"bs_1_rank{rank}.json.gz").write_bytes(b"")
        (
            torch_trace / f"rank_{rank}" / f"demo_ts_2026_{rank}.pt.trace.json.gz"
        ).write_bytes(b"")

    pipeline = TraceLensInferencePipeline(_atom_config(tmp_path))
    rank0_trace = pipeline._locate_rank0_trace(torch_trace)

    assert rank0_trace == torch_trace / "rank_0" / "demo_ts_2026_0.pt.trace.json.gz"
    assert pipeline._capture_folder(torch_trace, rank0_trace) == (
        torch_trace / "rank_0" / "capture_traces"
    )


@pytest.mark.parametrize(
    "rank0_dir, sibling_dir",
    [
        ("dp0_tp0", "dp1_tp0"),
        ("pp0_rank_0", "pp1_rank_0"),
        ("pp0_dp0_tp0", "pp0_dp1_tp0"),
    ],
)
def test_tracelens_inference_resolves_atom_parallel_layouts(
    tmp_path, rank0_dir, sibling_dir
):
    """atom names the rank dir dp<n>_tp<n> / pp<n>_... once DP or PP is on."""
    torch_trace = tmp_path / "torch_trace"
    for name in (rank0_dir, sibling_dir):
        capture = torch_trace / name / "capture_traces"
        capture.mkdir(parents=True)
        (capture / "bs_1_rank0.json.gz").write_bytes(b"")
        (torch_trace / name / f"demo_ts_2026_{name}.pt.trace.json.gz").write_bytes(b"")

    pipeline = TraceLensInferencePipeline(_atom_config(tmp_path))
    rank0_trace = pipeline._locate_rank0_trace(torch_trace)

    assert rank0_trace == (
        torch_trace / rank0_dir / f"demo_ts_2026_{rank0_dir}.pt.trace.json.gz"
    )
    assert pipeline._capture_folder(torch_trace, rank0_trace) == (
        torch_trace / rank0_dir / "capture_traces"
    )


def test_tracelens_inference_prefers_tp_only_layout_over_dp(tmp_path):
    """rank_0/ stays the first choice when both layouts somehow coexist."""
    torch_trace = tmp_path / "torch_trace"
    for name in ("rank_0", "dp0_tp0"):
        (torch_trace / name).mkdir(parents=True)
        (torch_trace / name / "demo.pt.trace.json.gz").write_bytes(b"")

    pipeline = TraceLensInferencePipeline(_atom_config(tmp_path))

    assert pipeline._locate_rank0_trace(torch_trace) == (
        torch_trace / "rank_0" / "demo.pt.trace.json.gz"
    )


def test_tracelens_inference_capture_folder_falls_back_to_flat_layout(tmp_path):
    torch_trace = tmp_path / "torch_trace"
    (torch_trace / "capture_traces").mkdir(parents=True)
    rank0_trace = torch_trace / "demo.pt.trace.json.gz"
    rank0_trace.write_bytes(b"")

    pipeline = TraceLensInferencePipeline(_atom_config(tmp_path))

    assert pipeline._capture_folder(torch_trace, rank0_trace) == (
        torch_trace / "capture_traces"
    )


def test_docker_image_package_version_reads_importlib_metadata(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="0.22.0+rocm722\n")

    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.subprocess.run",
        fake_run,
    )

    assert docker_image_package_version("internal/vllm:latest", "vllm") == (
        "0.22.0+rocm722"
    )
    assert seen["cmd"][:6] == [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "python3",
        "internal/vllm:latest",
    ]
    assert "importlib.metadata" in seen["cmd"][7]
    assert seen["cmd"][8] == "vllm"
    assert seen["kwargs"]["timeout"] == 120


@pytest.mark.parametrize(
    ("completed", "expected"),
    [
        (subprocess.CompletedProcess([], 0, stdout="noise\nTrue\n"), True),
        (subprocess.CompletedProcess([], 0, stdout="False\n"), False),
        (subprocess.CompletedProcess([], 2, stdout="True\n"), False),
        (None, False),
    ],
)
def test_docker_image_probe_interprets_probe_result(monkeypatch, completed, expected):
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime._run_python_in_image",
        lambda *_args: completed,
    )

    assert docker_image_probe("example/image:latest", "print(True)") is expected


def test_docker_image_probe_handles_docker_errors(monkeypatch):
    def fail(*_args, **_kwargs):
        raise OSError("docker unavailable")

    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.subprocess.run",
        fail,
    )

    assert docker_image_probe("example/image:latest", "print(True)") is False


def test_resolve_tracelens_repo_path_clones_main_when_unconfigured(
    tmp_path,
    monkeypatch,
):
    cache_path = tmp_path / "cache" / "magpie" / "TraceLens"
    clone_calls = []

    monkeypatch.delenv("TRACELENS_REPO_PATH", raising=False)
    monkeypatch.delenv("TRACELENS_PATH", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    def fake_run(cmd, **kwargs):
        clone_calls.append((cmd, kwargs))
        checkout = Path(cmd[-1])
        (checkout / "examples/custom_workflows/inference_analysis").mkdir(parents=True)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.subprocess.run",
        fake_run,
    )

    assert resolve_tracelens_repo_path() == cache_path.resolve()
    assert clone_calls[0][0][:7] == [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        "main",
        "https://github.com/AMD-AGI/TraceLens.git",
    ]
    assert clone_calls[0][1]["timeout"] == 300


def test_resolve_tracelens_repo_path_does_not_clone_for_invalid_explicit_path(
    tmp_path,
    monkeypatch,
):
    configured_path = tmp_path / "invalid-tracelens"

    monkeypatch.delenv("TRACELENS_REPO_PATH", raising=False)
    monkeypatch.delenv("TRACELENS_PATH", raising=False)
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.subprocess.run",
        lambda *args, **kwargs: pytest.fail("git clone must not run"),
    )

    with pytest.raises(RuntimeError, match="path is invalid"):
        resolve_tracelens_repo_path(str(configured_path))


def test_prepare_tracelens_runtime_image_reuses_existing_derived_image(
    tmp_path,
    monkeypatch,
):
    tracelens_repo = tmp_path / "TraceLens"
    workflow_dir = tracelens_repo / "examples/custom_workflows/inference_analysis"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "build_docker_sglang.sh").write_text(
        "normalize_version() {\n"
        '    case "$1" in\n'
        "        0.5.12|v0512|0512|5.12)\n"
        '            echo "0.5.12"\n'
        "            ;;\n"
        "    esac\n"
        "}\n"
    )
    (workflow_dir / "sglang_roofline_patches" / "sglang_0_5_12").mkdir(parents=True)
    monkeypatch.setenv("TRACELENS_REPO_PATH", str(tracelens_repo))
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.docker_image_exists",
        lambda _image: True,
    )
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.docker_image_package_version",
        lambda _image, _package: None,
    )

    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "sglang",
            "model": "demo",
            "docker_image": "lmsysorg/sglang:v0.5.12-rocm720-mi35x",
            "profiler": {"tracelens": {"enabled": True}},
        }
    )

    result = prepare_tracelens_runtime_image(
        cfg,
        base_image=cfg.docker_image,
        runner_type="mi355x",
    )

    assert result["built"] is False
    assert result["reason"] == "derived image already exists"
    assert result["image"].startswith("magpie-tracelens-sglang:0_5_12-mi355x-")


def test_prepare_tracelens_runtime_image_prefers_installed_package_version(
    tmp_path,
    monkeypatch,
):
    tracelens_repo = tmp_path / "TraceLens"
    workflow_dir = tracelens_repo / "examples/custom_workflows/inference_analysis"
    patch_dir = workflow_dir / "vllm_patches"
    patch_dir.mkdir(parents=True)
    (workflow_dir / "build_docker_vllm.sh").write_text(
        "case ${VLLM_VERSION} in\n" "    v22)\n" "        ;;\n" "esac\n"
    )
    (patch_dir / "config_vllm_v0.22.0.patch").write_text("patch")
    monkeypatch.setenv("TRACELENS_REPO_PATH", str(tracelens_repo))
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.docker_image_exists",
        lambda _image: True,
    )
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.docker_image_package_version",
        lambda _image, package: "0.22.0+rocm722" if package == "vllm" else None,
    )

    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "vllm",
            "model": "demo",
            "docker_image": "internal/vllm-rocm:latest",
            "profiler": {"tracelens": {"enabled": True}},
        }
    )

    result = prepare_tracelens_runtime_image(
        cfg,
        base_image=cfg.docker_image,
        runner_type="mi355x",
    )

    assert result["built"] is False
    assert result["patch_version"] == "v22"
    assert result["patch_version_source"] == "package"
    assert result["runtime_package_version"] == "0.22.0+rocm722"
    assert result["image"].startswith("magpie-tracelens-vllm:v22-mi355x-")


def test_prepare_tracelens_runtime_image_builds_extension_overlay(
    tmp_path,
    monkeypatch,
):
    tracelens_repo = tmp_path / "TraceLens"
    workflow_dir = tracelens_repo / "examples/custom_workflows/inference_analysis"
    patch_dir = workflow_dir / "vllm_patches"
    patch_dir.mkdir(parents=True)
    (workflow_dir / "build_docker_vllm.sh").write_text(
        "case ${VLLM_VERSION} in\n" "    v21)\n" "        ;;\n" "esac\n"
    )
    (patch_dir / "config_vllm_v0.21.0.patch").write_text("patch")

    extension_wheel = (
        tmp_path / "TraceLens_Ext-0.1.0.dev20260529+gacb7fbc6-py3-none-any 1.whl"
    )
    with zipfile.ZipFile(extension_wheel, "w") as archive:
        archive.writestr("TraceLens_Ext/__init__.py", "")
        archive.writestr(
            "TraceLens_Ext/Agent/Analysis/utils/arch/ExampleGPU.json",
            '{"name": "ExampleGPU"}',
        )

    monkeypatch.setenv("TRACELENS_REPO_PATH", str(tracelens_repo))
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.docker_image_exists",
        lambda _image: False,
    )
    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.docker_image_package_version",
        lambda _image, _package: None,
    )

    calls = []

    def fake_run(cmd, **kwargs):
        call = {"cmd": cmd, "kwargs": kwargs}
        if cmd[:2] == ["docker", "build"]:
            build_context = Path(cmd[-1])
            call["dockerfile"] = (build_context / "Dockerfile").read_text()
            call["context_files"] = sorted(
                path.name for path in build_context.iterdir()
            )
        calls.append(call)
        return subprocess.CompletedProcess(cmd, 0, stdout="built\n")

    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.subprocess.run",
        fake_run,
    )

    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "vllm",
            "model": "demo",
            "docker_image": "vllm/vllm-openai-rocm:v0.21.0",
            "envs": {"TL_EXTENSION": "ExistingExtension"},
            "profiler": {
                "tracelens": {
                    "enabled": True,
                    "extension_wheel_path": str(extension_wheel),
                }
            },
        }
    )

    result = prepare_tracelens_runtime_image(
        cfg,
        base_image=cfg.docker_image,
        runner_type="mi355x",
    )

    assert result["built"] is True
    assert result["public_runtime_built"] is True
    assert result["extension_built"] is True
    assert result["extension_module"] == "TraceLens_Ext"
    assert result["image"].endswith(f"-ext-{result['extension_wheel_sha256'][:12]}")
    assert cfg.envs["TL_EXTENSION"] == "ExistingExtension:TraceLens_Ext"
    assert calls[0]["cmd"][0] == "bash"
    assert calls[1]["cmd"][:2] == ["docker", "build"]
    assert "ENV TL_EXTENSION=TraceLens_Ext" in calls[1]["dockerfile"]
    assert (
        "TraceLens_Ext-0.1.0.dev20260529+gacb7fbc6-py3-none-any.whl"
        in calls[1]["context_files"]
    )


def test_prepare_tracelens_runtime_image_overlays_ready_image(
    tmp_path,
    monkeypatch,
):
    extension_wheel = tmp_path / "Custom_Ext-1.0-py3-none-any.whl"
    with zipfile.ZipFile(extension_wheel, "w") as archive:
        archive.writestr("Custom_Ext/__init__.py", "")
        archive.writestr(
            "Custom_Ext/Agent/Analysis/utils/agent_extension.py",
            "",
        )

    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.docker_image_exists",
        lambda _image: False,
    )
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="built\n")

    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_runtime.subprocess.run",
        fake_run,
    )

    base_image = "magpie-tracelens-vllm:v21-mi355x-public"
    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "vllm",
            "model": "demo",
            "docker_image": base_image,
            "profiler": {
                "tracelens": {
                    "enabled": True,
                    "extension_wheel_path": str(extension_wheel),
                }
            },
        }
    )

    result = prepare_tracelens_runtime_image(
        cfg,
        base_image=base_image,
        runner_type="mi355x",
    )

    assert result["public_runtime_built"] is False
    assert result["extension_built"] is True
    assert result["public_runtime_image"] == base_image
    assert result["extension_module"] == "Custom_Ext"
    assert calls[0][:2] == ["docker", "build"]


def test_tracelens_inference_prepare_patches_and_restores(tmp_path):
    inferencex = tmp_path / "InferenceX"
    bench_dir = inferencex / "benchmarks"
    serving_dir = inferencex / "utils" / "bench_serving"
    bench_dir.mkdir(parents=True)
    serving_dir.mkdir(parents=True)

    benchmark_lib = bench_dir / "benchmark_lib.sh"
    benchmark_lib.write_text(
        'if [[ "${PROFILE:-}" == "1" ]]; then\n'
        '    num_prompts="$max_concurrency"\n'
        "fi\n",
        encoding="utf-8",
    )

    benchmark_serving = serving_dir / "benchmark_serving.py"
    benchmark_serving.write_text(
        'extra_body={"num_steps": 1, "merge_profiles": True, '
        '"profile_by_stage": True},\n',
        encoding="utf-8",
    )

    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "sglang",
            "model": "demo",
            "inferencex_path": str(inferencex),
            "envs": {
                "CONC": 64,
                "OSL": 1024,
                "RANDOM_RANGE_RATIO": 1,
            },
            "profiler": {"tracelens": {"enabled": True}},
        }
    )
    pipeline = TraceLensInferencePipeline(cfg)

    result = pipeline.prepare(tmp_path / "workspace")

    assert result["max_iterations"] == 256
    assert result["delay_iterations"] == 6016
    assert 'num_prompts="$num_prompts"' in benchmark_lib.read_text(encoding="utf-8")
    patched_serving = benchmark_serving.read_text(encoding="utf-8")
    assert '"shape_discovery": True' in patched_serving
    assert '"detailed_annotations": True' in patched_serving
    assert '"start_step": 6016' in patched_serving
    assert '"num_steps": 256' in patched_serving
    assert cfg.envs["SGLANG_PROFILE_WITH_STACK"] == "True"
    assert cfg.envs["SGLANG_PROFILE_RECORD_SHAPES"] == "True"
    assert cfg.envs["SGLANG_GRAPH_BATCH_CAPTURE"] == "True"
    assert "SGLANG_PROFILE_RECORD_SHAPE" not in cfg.envs
    assert "--enable-profile-cuda-graph" in cfg.envs["EXTRA_SGLANG_ARGS"]
    assert SGLANG_SHAPE_DISCOVERY_FLAG not in cfg.envs["EXTRA_SGLANG_ARGS"]

    restore = pipeline.restore()

    assert str(benchmark_lib) in restore["restored_files"]
    assert str(benchmark_serving) in restore["restored_files"]
    assert 'num_prompts="$max_concurrency"' in benchmark_lib.read_text(encoding="utf-8")
    assert "detailed_annotations" not in benchmark_serving.read_text(encoding="utf-8")


def test_tracelens_inference_prepare_enables_sglang_shape_discovery_for_patched_image(
    tmp_path,
):
    inferencex = tmp_path / "InferenceX"
    bench_dir = inferencex / "benchmarks"
    serving_dir = inferencex / "utils" / "bench_serving"
    bench_dir.mkdir(parents=True)
    serving_dir.mkdir(parents=True)

    (bench_dir / "benchmark_lib.sh").write_text(
        'if [[ "${PROFILE:-}" == "1" ]]; then\n'
        '    num_prompts="$max_concurrency"\n'
        "fi\n",
        encoding="utf-8",
    )
    (serving_dir / "benchmark_serving.py").write_text(
        'extra_body={"num_steps": 1, "merge_profiles": True, '
        '"profile_by_stage": True},\n',
        encoding="utf-8",
    )

    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "sglang",
            "model": "demo",
            "docker_image": "tracelens-sglang:0.5.12-mi355-fix",
            "inferencex_path": str(inferencex),
            "envs": {
                "CONC": 64,
                "OSL": 1024,
                "RANDOM_RANGE_RATIO": 1,
            },
            "profiler": {"tracelens": {"enabled": True}},
        }
    )

    result = TraceLensInferencePipeline(cfg).prepare(tmp_path / "workspace")

    assert SGLANG_SHAPE_DISCOVERY_FLAG in cfg.envs["EXTRA_SGLANG_ARGS"]
    assert SGLANG_SHAPE_DISCOVERY_FLAG in result["env_updates"]["EXTRA_SGLANG_ARGS"]


def test_tracelens_inference_sglang_step_marker_fallback(tmp_path):
    trace_path = tmp_path / "rank0.trace.json.gz"
    trace = {
        "traceEvents": [
            {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "rank0"}},
            {
                "cat": "user_annotation",
                "name": "step[DECODE bs=32]",
                "ts": 100.0,
                "dur": 10.0,
            },
            {
                "cat": "user_annotation",
                "name": "step[DECODE bs=64]",
                "ts": 200.0,
                "dur": 20.0,
            },
            {
                "cat": "user_annotation",
                "name": "step[EXTEND bs=4 toks=4096]",
                "ts": 300.0,
                "dur": 30.0,
            },
            {"cat": "cpu_op", "name": "inside_decode", "ts": 205.0, "dur": 2.0},
            {"cat": "kernel", "name": "decode_kernel", "ts": 225.0, "dur": 5.0},
            {"cat": "cpu_op", "name": "outside", "ts": 500.0, "dur": 2.0},
        ]
    }
    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        json.dump(trace, handle)

    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "sglang",
            "model": "demo",
            "envs": {"CONC": 64, "OSL": 1024},
            "profiler": {"tracelens": {"enabled": True}},
        }
    )
    pipeline = TraceLensInferencePipeline(cfg)
    split_dir = tmp_path / "split"

    warnings = pipeline._split_sglang_step_markers(trace_path, split_dir)

    assert "wrote 2 trace window" in warnings[-1]
    rows = list(csv.DictReader((split_dir / "execution_details.csv").open()))
    assert {row["stage"] for row in rows} == {"decode", "prefill"}
    assert {row["phase_avg_bs"] for row in rows} == {"64.0", "4.0"}
    decode_trace = split_dir / "decode_only_step.trace.json.gz"
    with gzip.open(decode_trace, "rt", encoding="utf-8") as handle:
        decode_events = json.load(handle)["traceEvents"]
    assert any(event.get("name") == "inside_decode" for event in decode_events)
    assert any(event.get("name") == "decode_kernel" for event in decode_events)
    assert not any(event.get("name") == "outside" for event in decode_events)


def test_tracelens_inference_skips_empty_gpu_trace_candidates(tmp_path):
    split_dir = tmp_path / "trace_split"
    split_dir.mkdir()
    empty_decode = split_dir / "annotation_iteration_0_decode.trace.json.gz"
    valid_decode = split_dir / "annotation_iteration_511_decode.trace.json.gz"
    for trace_path in (empty_decode, valid_decode):
        with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
            json.dump({"traceEvents": []}, handle)

    with (split_dir / "execution_details.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "output_path",
                "num_steps",
                "phase_avg_bs",
                "phase_avg_conc",
                "phase_num_prefilldecode",
                "phase_num_decode",
                "phase_num_prefill",
                "num_gpu_events",
                "gpu_duration",
                "gpu_busy_duration",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "output_path": str(empty_decode),
                "num_steps": "1",
                "phase_avg_bs": "32",
                "phase_avg_conc": "32",
                "phase_num_prefilldecode": "0",
                "phase_num_decode": "1",
                "phase_num_prefill": "0",
                "num_gpu_events": "0",
                "gpu_duration": "0",
                "gpu_busy_duration": "0",
            }
        )
        writer.writerow(
            {
                "output_path": str(valid_decode),
                "num_steps": "1",
                "phase_avg_bs": "32",
                "phase_avg_conc": "32",
                "phase_num_prefilldecode": "0",
                "phase_num_decode": "1",
                "phase_num_prefill": "0",
                "num_gpu_events": "1921",
                "gpu_duration": "22082.06",
                "gpu_busy_duration": "18000.5",
            }
        )

    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "vllm",
            "model": "demo",
            "envs": {"CONC": 32, "OSL": 1024},
            "profiler": {"tracelens": {"enabled": True}},
        }
    )

    picks = TraceLensInferencePipeline(cfg)._pick_largest_batch_traces(
        split_dir / "execution_details.csv"
    )

    decode_pick = picks["decode"]
    assert decode_pick.trace_path == valid_decode
    assert decode_pick.row_index == 1
    assert decode_pick.num_gpu_events == 1921
    assert decode_pick.gpu_duration == 22082.06
    assert decode_pick.gpu_busy_duration == 18000.5
    assert decode_pick.selection_reason is not None
    assert "valid single-iteration traces with GPU work" in decode_pick.selection_reason


def test_tracelens_inference_prefers_representative_gpu_work(tmp_path):
    split_dir = tmp_path / "trace_split"
    split_dir.mkdir()
    tiny_decode = split_dir / "annotation_iteration_254_decode.trace.json.gz"
    full_decode = split_dir / "annotation_iteration_255_decode.trace.json.gz"
    for trace_path in (tiny_decode, full_decode):
        with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
            json.dump({"traceEvents": []}, handle)

    with (split_dir / "execution_details.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "output_path",
                "num_steps",
                "phase_avg_bs",
                "phase_avg_conc",
                "phase_num_prefilldecode",
                "phase_num_decode",
                "phase_num_prefill",
                "num_gpu_events",
                "gpu_duration",
                "gpu_busy_duration",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "output_path": str(tiny_decode),
                "num_steps": "1",
                "phase_avg_bs": "256",
                "phase_avg_conc": "256",
                "phase_num_prefilldecode": "0",
                "phase_num_decode": "1",
                "phase_num_prefill": "0",
                "num_gpu_events": "5",
                "gpu_duration": "626.76",
                "gpu_busy_duration": "617.32",
            }
        )
        writer.writerow(
            {
                "output_path": str(full_decode),
                "num_steps": "1",
                "phase_avg_bs": "256",
                "phase_avg_conc": "256",
                "phase_num_prefilldecode": "0",
                "phase_num_decode": "1",
                "phase_num_prefill": "0",
                "num_gpu_events": "1861",
                "gpu_duration": "45538.17",
                "gpu_busy_duration": "45507.27",
            }
        )

    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "vllm",
            "model": "demo",
            "envs": {"CONC": 256, "OSL": 1024},
            "profiler": {"tracelens": {"enabled": True}},
        }
    )

    picks = TraceLensInferencePipeline(cfg)._pick_largest_batch_traces(
        split_dir / "execution_details.csv"
    )

    decode_pick = picks["decode"]
    assert decode_pick.trace_path == full_decode
    assert decode_pick.num_gpu_events == 1861
    assert "median gpu_busy_duration" in (decode_pick.selection_reason or "")


def test_tracelens_inference_analysis_runs_in_cpu_only_container(
    tmp_path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    gpu_arch_config = tmp_path / "mi355x.json"
    gpu_arch_config.write_text('{"mem_bw_gbps": 8000}', encoding="utf-8")
    torch_trace_dir = workspace / "torch_trace"
    capture_dir = torch_trace_dir / "capture_traces"
    capture_dir.mkdir(parents=True)

    rank0_trace = torch_trace_dir / "rank0-TP-0.trace.json.gz"
    with gzip.open(rank0_trace, "wt", encoding="utf-8") as handle:
        json.dump({"traceEvents": []}, handle)
    with gzip.open(
        capture_dir / "bs_64_rank0.json.gz",
        "wt",
        encoding="utf-8",
    ) as handle:
        json.dump({"traceEvents": []}, handle)

    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "sglang",
            "model": "demo",
            "run_mode": "docker",
            "docker_image": "tracelens-sglang:0.5.12-mi355-fix",
            "envs": {
                "ISL": 8192,
                "CONC": 64,
                "OSL": 1024,
                "RANDOM_RANGE_RATIO": 1,
                "TL_EXTENSION": "TraceLens_NDA",
            },
            "profiler": {
                "tracelens": {
                    "enabled": True,
                    "analysis_stages": ["decode"],
                    "gpu_arch_config": str(gpu_arch_config),
                }
            },
        }
    )
    docker_cmds = []

    def fake_run(cmd, **_kwargs):
        docker_cmds.append(cmd)
        bash_cmd = cmd[-1]
        if "TraceLens_split_inference_trace" in bash_cmd:
            split_dir = torch_trace_dir / "trace_split"
            split_dir.mkdir(parents=True, exist_ok=True)
            decode_trace = split_dir / "decode.trace.json.gz"
            with gzip.open(decode_trace, "wt", encoding="utf-8") as handle:
                json.dump({"traceEvents": []}, handle)
            with (split_dir / "execution_details.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "output_path",
                        "num_steps",
                        "phase_avg_bs",
                        "phase_num_prefilldecode",
                        "phase_num_decode",
                        "phase_num_prefill",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "output_path": (
                            "/workspace/torch_trace/trace_split/" "decode.trace.json.gz"
                        ),
                        "num_steps": "1",
                        "phase_avg_bs": "64",
                        "phase_num_prefilldecode": "0",
                        "phase_num_decode": "1",
                        "phase_num_prefill": "0",
                    }
                )
        elif "TraceLens_generate_perf_report_pytorch_inference" in bash_cmd:
            out_dir = workspace / "tracelens" / "decode_only"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "summary.csv").write_text(
                "name,value\nok,1\n",
                encoding="utf-8",
            )
            with (out_dir / "unified_perf_summary.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "name",
                        "op category",
                        "operation_count",
                        "Kernel Time (µs)_sum",
                        "Percentage (%)",
                        "Kernel Time (µs)_mean",
                        "GFLOPS",
                        "Data Moved (MB)",
                        "FLOPS/Byte",
                        "TFLOPS/s_mean",
                        "TB/s_mean",
                        "Compute Spec",
                        "Roofline Bound",
                        "Pct Roofline_mean",
                        "Roofline Time (µs)_first",
                        "has_perf_model",
                        "Input Dims",
                        "Input type",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "name": "vllm::rocm_unquantized_gemm",
                        "op category": "GEMM",
                        "operation_count": "61",
                        "Kernel Time (µs)_sum": "1500",
                        "Percentage (%)": "12.5",
                        "Kernel Time (µs)_mean": "24.59",
                        "GFLOPS": "1.25",
                        "Data Moved (MB)": "32",
                        "FLOPS/Byte": "40",
                        "TFLOPS/s_mean": "100",
                        "TB/s_mean": "2.5",
                        "Compute Spec": "matrix_bf16",
                        "Roofline Bound": "MEMORY_BOUND",
                        "Pct Roofline_mean": "70",
                        "Roofline Time (µs)_first": "20",
                        "has_perf_model": "True",
                        "Input Dims": "[[8192, 7168], [7168, 512]]",
                        "Input type": "['bf16', 'bf16']",
                    }
                )
            with (out_dir / "GEMM.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "name",
                        "param: M",
                        "param: N",
                        "param: K",
                        "param: dtype_A_B",
                        "num_kernels",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "name": "vllm::rocm_unquantized_gemm",
                        "param: M": "8192",
                        "param: N": "512",
                        "param: K": "7168",
                        "param: dtype_A_B": "bf16",
                        "num_kernels": "2",
                    }
                )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(
        "Magpie.modes.benchmark.tracelens_inference.subprocess.run",
        fake_run,
    )

    result = TraceLensInferencePipeline(cfg).analyze_in_container(
        torch_trace_dir=torch_trace_dir,
        output_dir=workspace,
        runner_type="mi355x",
        docker_image=cfg.docker_image,
        workspace=workspace,
    )

    assert result["errors"] == []
    assert result["analysis_stages"] == ["decode"]
    assert result["tl_extension"] == "TraceLens_NDA"
    assert result["gpu_arch_platform_candidate"] == "MI355X"
    assert result["gpu_arch_platform"] is None
    assert result["postprocess_runtime"]["mode"] == "docker"
    assert (
        str(workspace / "tracelens" / "decode_only" / "summary.csv")
        in result["output_files"]
    )
    simple_summary = (
        workspace
        / "tracelens"
        / "decode_only_ISL8192_OSL1024_CONC64_kernel_roofline_simple.csv"
    )
    assert str(simple_summary) in result["output_files"]
    assert result["stage_results"]["decode"]["simple_summary_file"] == str(
        simple_summary
    )
    assert len(docker_cmds) == 2
    assert all(
        cmd[:5] == ["docker", "run", "--rm", "--network", "none"] for cmd in docker_cmds
    )
    assert all(
        "--gpus" not in cmd and "--device=/dev/kfd" not in cmd for cmd in docker_cmds
    )
    assert any("TL_EXTENSION=TraceLens_NDA" in cmd for cmd in docker_cmds)
    assert any(
        "--gpu_arch_json_path /workspace/tracelens/mi355x.json" in cmd[-1]
        for cmd in docker_cmds
    )
    assert all("--gpu_arch_platform" not in cmd[-1] for cmd in docker_cmds)
    assert (workspace / "tracelens" / "mi355x.json").read_text(
        encoding="utf-8"
    ) == gpu_arch_config.read_text(encoding="utf-8")

    rows = list(
        csv.DictReader(
            (torch_trace_dir / "trace_split" / "execution_details.csv").open()
        )
    )
    assert rows[0]["output_path"] == str(
        torch_trace_dir / "trace_split" / "decode.trace.json.gz"
    )

    simple_reader = csv.DictReader(simple_summary.open())
    assert simple_reader.fieldnames == [
        "source_category",
        "op_name",
        "param_signature",
        "operation_count",
        "num_kernels",
        "kernel_time_ms_sum",
        "time_pct",
        "kernel_time_us_mean",
        "gflops",
        "data_moved_mb",
        "arithmetic_intensity_flops_per_byte",
        "achieved_tflops_mean",
        "achieved_tbps_mean",
        "compute_spec",
        "roofline_bound",
        "pct_roofline_mean",
        "roofline_time_us",
        "has_perf_model",
        "params_json",
        "input_dims",
        "input_type",
    ]
    simple_rows = list(simple_reader)
    assert simple_rows == [
        {
            "source_category": "GEMM",
            "op_name": "vllm::rocm_unquantized_gemm",
            "param_signature": "M=8192,N=512,K=7168,dtype_A_B=bf16",
            "operation_count": "61",
            "num_kernels": "2",
            "kernel_time_ms_sum": "1.5",
            "time_pct": "12.5",
            "kernel_time_us_mean": "24.59",
            "gflops": "1.25",
            "data_moved_mb": "32",
            "arithmetic_intensity_flops_per_byte": "40",
            "achieved_tflops_mean": "100",
            "achieved_tbps_mean": "2.5",
            "compute_spec": "matrix_bf16",
            "roofline_bound": "MEMORY_BOUND",
            "pct_roofline_mean": "70",
            "roofline_time_us": "20",
            "has_perf_model": "True",
            "params_json": ('{"K":"7168","M":"8192","N":"512","dtype_A_B":"bf16"}'),
            "input_dims": "[[8192, 7168], [7168, 512]]",
            "input_type": "['bf16', 'bf16']",
        }
    ]


def test_tracelens_simple_summary_omits_arch_columns_without_arch_json(tmp_path):
    output_dir = tmp_path / "tracelens" / "decode_only"
    output_dir.mkdir(parents=True)
    with (output_dir / "unified_perf_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "op category",
                "operation_count",
                "Kernel Time (µs)_sum",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "name": "aten::add",
                "op category": "ELEMENTWISE",
                "operation_count": "3",
                "Kernel Time (µs)_sum": "250",
            }
        )

    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "vllm",
            "model": "demo",
            "profiler": {"tracelens": {"enabled": True}},
        }
    )
    summary = TraceLensInferencePipeline(cfg)._write_simple_roofline_summary(
        "decode",
        output_dir,
    )

    assert summary is not None
    reader = csv.DictReader(summary.open())
    assert "roofline_bound" not in reader.fieldnames
    assert "pct_roofline_mean" not in reader.fieldnames
    assert "roofline_time_us" not in reader.fieldnames
    assert list(reader)[0]["kernel_time_ms_sum"] == "0.25"


def test_benchmark_mode_uses_container_for_docker_tracelens_inference(tmp_path):
    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "sglang",
            "model": "demo",
            "run_mode": "docker",
            "docker_image": "tracelens-sglang:0.5.12-mi355-fix",
            "profiler": {"tracelens": {"enabled": True}},
        }
    )
    mode = BenchmarkMode(cfg)

    class FakePipeline:
        def analyze(self, **_kwargs):
            raise AssertionError("host TraceLens analysis should not run")

        def analyze_in_container(self, **kwargs):
            return {
                "enabled": True,
                "analysis_mode": "inference",
                "postprocess_runtime": {
                    "mode": "docker",
                    "image": kwargs["docker_image"],
                },
                "output_files": [],
                "warnings": [],
                "errors": [],
            }

    result = mode._run_tracelens_inference_analysis(
        torch_trace_dir=tmp_path / "torch_trace",
        workspace=tmp_path,
        runner_type="mi355x",
        pipeline=FakePipeline(),
    )

    assert result["postprocess_runtime"] == {
        "mode": "docker",
        "image": "tracelens-sglang:0.5.12-mi355-fix",
    }


def test_gap_analysis_config_validates_window():
    with pytest.raises(ValueError):
        GapAnalysisConfig(trace_start_pct=80, trace_end_pct=80)

    with pytest.raises(ValueError):
        GapAnalysisConfig(trace_start_pct=-1, trace_end_pct=50)


def test_benchmark_config_from_dict_normalizes_nested_sections():
    cfg = BenchmarkConfig.from_dict(
        {
            "framework": "VLLM",
            "model": "test-model",
            "run_mode": "ray",
            "profiler": {
                "torch_profiler": {"enabled": False},
                "tracelens": {"enabled": True, "export_format": "csv"},
            },
            "gap_analysis": {
                "enabled": True,
                "trace_start_pct": 10,
                "trace_end_pct": 90,
            },
            "ray_config": {"cluster_address": "auto", "num_nodes": 2},
            "inferencemax_path": "/tmp/inferencex",
        }
    )

    assert cfg.framework == "vllm"
    assert cfg.is_ray is True
    assert cfg.profiler.torch_profiler.enabled is False
    assert cfg.profiler.tracelens.enabled is True
    assert cfg.gap_analysis.trace_start_pct == 10
    assert isinstance(cfg.ray_config, RayConfig)
    assert cfg.ray_config.num_nodes == 2
    assert cfg.inferencex_path == "/tmp/inferencex"
    assert cfg.get_env_vars()["MODEL"] == "test-model"


def test_benchmark_config_sets_defaults_and_script_name():
    cfg = BenchmarkConfig(framework="sglang", model="demo")

    assert cfg.envs["TP"] == 1
    assert cfg.envs["CONC"] == 32
    assert cfg.get_benchmark_script_name() == "generic_fp8_mi300x.sh"

    cfg.runner_type = "h100"
    cfg.precision = "bf16"
    assert cfg.get_benchmark_script_name() == "generic_bf16_h100.sh"


def test_benchmark_config_accepts_atom_framework():
    cfg = BenchmarkConfig(framework="atom", model="demo")

    assert cfg.framework == "atom"
    assert cfg.envs["TP"] == 1

    cfg_from_dict = BenchmarkConfig.from_dict(
        {
            "framework": "ATOM",
            "model": "demo",
            "run_mode": "local",
            "profiler": {"torch_profiler": {"enabled": False}},
        }
    )
    assert cfg_from_dict.framework == "atom"


@pytest.mark.parametrize("runner", ["atom_mi300x.sh", "atom_mi355x.sh"])
def test_atom_launch_script_wires_profile_to_torch_profiler_dir(runner):
    """PROFILE=1 must build --torch-profiler-dir from ATOM_TORCH_PROFILER_DIR
    (with a workspace fallback) and pass it to the atom server."""
    from pathlib import Path

    script = (
        Path(__file__).resolve().parent.parent
        / "Magpie"
        / "scripts"
        / "benchmark"
        / runner
    )
    text = script.read_text(encoding="utf-8")
    assert "PROFILER_ARGS+=(--torch-profiler-dir" in text
    assert "ATOM_TORCH_PROFILER_DIR" in text
    assert "WORKSPACE_DIR/torch_trace" in text
    assert '"${PROFILER_ARGS[@]}"' in text


@pytest.mark.parametrize("runner", ["atom_mi300x.sh", "atom_mi355x.sh"])
def test_atom_launch_script_forwards_max_model_len(runner):
    """MAX_MODEL_LEN is configured by the example and must reach the server."""
    from pathlib import Path

    script = (
        Path(__file__).resolve().parent.parent
        / "Magpie"
        / "scripts"
        / "benchmark"
        / runner
    )
    text = script.read_text(encoding="utf-8")
    assert "MAX_MODEL_LEN=${MAX_MODEL_LEN:-" in text
    assert '--max-model-len "$MAX_MODEL_LEN"' in text


def test_benchmark_script_copy_is_atomic_on_failure(tmp_path, monkeypatch):
    source = tmp_path / "source.sh"
    target = tmp_path / "target.sh"
    source.write_text("new content\n", encoding="utf-8")
    target.write_text("old content\n", encoding="utf-8")

    def fail_copy(src, dst):
        raise RuntimeError("copy failed")

    monkeypatch.setattr(
        "Magpie.modes.benchmark.benchmarker.shutil.copy2",
        fail_copy,
    )

    with pytest.raises(RuntimeError, match="copy failed"):
        BenchmarkMode._copy_benchmark_script_atomic(source, target)

    assert target.read_text(encoding="utf-8") == "old content\n"
    assert not list(tmp_path.glob(".target.sh.*.tmp"))


def test_benchmark_script_copy_sets_executable_bit(tmp_path):
    source = tmp_path / "source.sh"
    target = tmp_path / "target.sh"
    source.write_text("#!/usr/bin/env bash\necho ok\n", encoding="utf-8")

    BenchmarkMode._copy_benchmark_script_atomic(source, target)

    assert target.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert target.stat().st_mode & 0o111


def test_image_selector_selects_override_and_arch_mapping(tmp_path, monkeypatch):
    config_path = tmp_path / "images.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "vllm": {"gfx942": "amd/vllm:mi300x", "sm_90": "nvidia/vllm:h100"},
                "sglang": {"gfx950": "amd/sglang:mi355x"},
                "atom": {"gfx942": "amd/atom:mi300x"},
            }
        ),
        encoding="utf-8",
    )
    selector = ImageSelector(str(config_path))

    assert (
        selector.select_image("vllm", override_image="custom:image") == "custom:image"
    )
    assert selector.select_image("vllm", gpu_arch="gfx942") == "amd/vllm:mi300x"

    monkeypatch.setattr(
        "Magpie.modes.benchmark.image_selector.detect_gpu",
        lambda: (GPUVendor.AMD, "gfx950"),
    )
    assert selector.select_image("sglang") == "amd/sglang:mi355x"
    assert selector.select_image("atom", gpu_arch="gfx942") == "amd/atom:mi300x"
    assert selector.get_runner_type("sm_90") == "h100"

    with pytest.raises(ValueError):
        selector.select_image("unknown", gpu_arch="gfx942")

    with pytest.raises(ValueError):
        selector.get_runner_type("unknown_arch")


@pytest.mark.parametrize("gpu_arch", ["gfx942", "gfx950"])
def test_real_benchmark_images_yaml_atom_resolves_to_rocm_atom_image(gpu_arch):
    """atom must resolve to the dedicated rocm/atom image on AMD arches."""
    selector = ImageSelector()
    image = selector.select_image("atom", gpu_arch=gpu_arch)
    assert image == "rocm/atom:latest"
    assert "vllm" not in image.lower()


def test_real_benchmark_images_yaml_atom_nvidia_arch_raises():
    """atom is AMD-only; NVIDIA arches must error out, not fall back to vLLM."""
    selector = ImageSelector()
    for sm_arch in ("sm_80", "sm_90", "sm_100"):
        with pytest.raises(ValueError, match="No image found for GPU architecture"):
            selector.select_image("atom", gpu_arch=sm_arch)


def test_result_parser_parses_inferencex_json_and_missing_file(tmp_path):
    missing = ResultParser.parse_inferencex_result(tmp_path / "missing.json")
    assert missing.success is False
    assert missing.errors

    result_path = tmp_path / "inferencex_result.json"
    result_path.write_text(
        json.dumps(
            {
                "request_throughput": 12.5,
                "output_throughput": 512.0,
                "total_token_throughput": 768.0,
                "completed": 32,
                "total_input_tokens": 4096,
                "total_output_tokens": 8192,
                "duration": 10.0,
                "mean_ttft_ms": 3.5,
                "p99_e2el_ms": 42.0,
                "model_id": "from-file-model",
            }
        ),
        encoding="utf-8",
    )

    parsed = ResultParser.parse_inferencex_result(result_path, framework="vllm")

    assert parsed.success is True
    assert parsed.framework == "vllm"
    assert parsed.model == "from-file-model"
    assert parsed.throughput.request_throughput == 12.5
    assert parsed.latency.ttft_mean == 3.5
    assert parsed.latency.e2el_p99 == 42.0


def test_result_parser_aggregates_first_torch_trace_file(tmp_path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()

    trace = {
        "traceEvents": [
            {"cat": "kernel", "name": "kernel_a", "dur": 2000},
            {"cat": "kernel", "name": "kernel_a", "dur": 1000},
            {"cat": "kernel", "name": "kernel_b", "dur": 500},
            {"cat": "cpu_op", "name": "ignored", "dur": 999},
        ]
    }

    with gzip.open(trace_dir / "rank0.json.gz", "wt") as f:
        json.dump(trace, f)

    kernels = ResultParser.parse_torch_trace(trace_dir)

    assert [k.name for k in kernels] == ["kernel_a", "kernel_b"]
    assert kernels[0].time_ms == 3.0
    assert kernels[0].calls == 2
    assert pytest.approx(kernels[0].percent, rel=1e-6) == (3.0 / 3.5) * 100


def test_result_parser_finds_atom_nested_rank_traces(tmp_path):
    """parse_torch_trace must recurse into per-rank subdirs (atom layout)."""
    trace_dir = tmp_path / "torch_trace"
    rank_dir = trace_dir / "rank_0"
    rank_dir.mkdir(parents=True)

    trace = {
        "traceEvents": [
            {"cat": "kernel", "name": "atom_moe_kernel", "dur": 1500},
            {"cat": "kernel", "name": "atom_moe_kernel", "dur": 1500},
            {"cat": "cpu_op", "name": "ignored", "dur": 999},
        ]
    }
    with gzip.open(
        rank_dir / "atom_ts_20260528_120000_001.pt.trace.json.gz", "wt"
    ) as f:
        json.dump(trace, f)

    kernels = ResultParser.parse_torch_trace(trace_dir)

    assert [k.name for k in kernels] == ["atom_moe_kernel"]
    assert kernels[0].time_ms == 3.0
    assert kernels[0].calls == 2


def test_result_parser_skips_vllm_warmup_trace(tmp_path):
    """parse_torch_trace must ignore CUDA-graph warmup snapshots (issue #38).

    vLLM writes a warmup trace under capture_traces/. The parser must select the
    real benchmark trace instead, so the dominant kernels (e.g. the all-reduce)
    appear in the summary. Here the warmup file is made larger and kernel-bearing
    so the capture_traces/ skip is the only thing that can exclude it.
    """
    trace_dir = tmp_path / "torch_trace"
    capture_dir = trace_dir / "capture_traces"
    capture_dir.mkdir(parents=True)

    # The warmup trace is deliberately the LARGEST file and contains valid
    # kernel events. Neither the largest-first sort nor the kernel-presence
    # check would reject it, so only the capture_traces/ skip prevents it from
    # shadowing the real trace. This keeps the test from passing vacuously.
    # Names are unique so the file does not gzip-compress down below the real
    # trace and break the size precondition asserted below.
    warmup_trace = {
        "traceEvents": [
            {"cat": "kernel", "name": f"warmup_kernel_{i}", "dur": 100}
            for i in range(2000)
        ]
    }
    warmup_file = capture_dir / "graph_capture_rank_0.json.gz"
    with gzip.open(warmup_file, "wt") as f:
        json.dump(warmup_trace, f)

    real_trace = {
        "traceEvents": [
            {"cat": "kernel", "name": "vllm::cross_device_reduce_1stage", "dur": 9000},
            {"cat": "kernel", "name": "gemm_kernel", "dur": 1000},
            {"cat": "cpu_op", "name": "ignored", "dur": 999},
        ]
    }
    real_file = trace_dir / "dp0_pp0_tp0_rank0.json.gz"
    with gzip.open(real_file, "wt") as f:
        json.dump(real_trace, f)

    # Guard the precondition: the warmup snapshot must be the larger file, so
    # the test genuinely exercises the capture_traces/ skip rather than the
    # size heuristic.
    assert warmup_file.stat().st_size > real_file.stat().st_size

    kernels = ResultParser.parse_torch_trace(trace_dir)

    names = [k.name for k in kernels]
    assert not any(n.startswith("warmup_kernel") for n in names)
    assert names[0] == "vllm::cross_device_reduce_1stage"


def test_result_parser_falls_back_when_largest_trace_is_corrupt(tmp_path):
    """A truncated/corrupt largest trace must fall back to the next valid one."""
    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()

    # Largest file by size, but corrupt (not valid gzip/json).
    corrupt = trace_dir / "rank0_corrupt.json.gz"
    corrupt.write_bytes(b"\x00not-a-valid-gzip-trace" * 1000)

    valid_trace = {
        "traceEvents": [
            {"cat": "kernel", "name": "gemm_kernel", "dur": 4000},
        ]
    }
    with gzip.open(trace_dir / "rank1_valid.json.gz", "wt") as f:
        json.dump(valid_trace, f)

    assert corrupt.stat().st_size > (trace_dir / "rank1_valid.json.gz").stat().st_size

    kernels = ResultParser.parse_torch_trace(trace_dir)

    assert [k.name for k in kernels] == ["gemm_kernel"]
    assert kernels[0].time_ms == 4.0


def test_result_parser_skips_trace_without_kernel_events(tmp_path):
    """A larger trace with no kernel events must not shadow a smaller real one."""
    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()

    # Larger file, valid JSON, but only CPU ops (no kernel-category events).
    cpu_only = {
        "traceEvents": [
            {"cat": "cpu_op", "name": f"op_{i}", "dur": 5} for i in range(500)
        ]
    }
    with gzip.open(trace_dir / "rank0_cpu_only.json.gz", "wt") as f:
        json.dump(cpu_only, f)

    real_trace = {
        "traceEvents": [
            {"cat": "kernel", "name": "gemm_kernel", "dur": 4000},
        ]
    }
    with gzip.open(trace_dir / "rank1_real.json.gz", "wt") as f:
        json.dump(real_trace, f)

    # Ensure the kernel-less file is the larger one, so we genuinely exercise
    # the fall-through (largest-first) path rather than passing trivially.
    assert (trace_dir / "rank0_cpu_only.json.gz").stat().st_size > (
        trace_dir / "rank1_real.json.gz"
    ).stat().st_size

    kernels = ResultParser.parse_torch_trace(trace_dir)

    assert [k.name for k in kernels] == ["gemm_kernel"]


def test_tracelens_find_trace_files_walks_atom_rank_subdirs(tmp_path):
    """_find_trace_files must pick up both flat and per-rank nested traces."""
    from Magpie.modes.benchmark.config import TraceLensConfig
    from Magpie.modes.benchmark.tracelens import TraceLensAnalyzer

    trace_dir = tmp_path / "torch_trace"
    flat_path = trace_dir / "vllm_style.json.gz"
    nested_dir = trace_dir / "rank_0"
    nested_path = nested_dir / "atom_ts_20260528.pt.trace.json.gz"

    nested_dir.mkdir(parents=True)
    flat_path.write_bytes(b"")
    nested_path.write_bytes(b"")

    cfg = TraceLensConfig(enabled=True)
    analyzer = TraceLensAnalyzer(cfg)
    found = analyzer._find_trace_files(trace_dir)

    found_set = {p.resolve() for p in found}
    assert flat_path.resolve() in found_set
    assert nested_path.resolve() in found_set


def test_tracelens_detect_pattern_wildcards_atom_rank_dir(tmp_path):
    """_detect_trace_pattern must wildcard the rank_<N>/ directory component
    for the atom layout, not collapse the traces to a flat path."""
    from Magpie.modes.benchmark.config import TraceLensConfig
    from Magpie.modes.benchmark.tracelens import TraceLensAnalyzer

    trace_dir = tmp_path / "torch_trace"
    files = []
    for rank in (0, 1):
        rank_dir = trace_dir / f"rank_{rank}"
        rank_dir.mkdir(parents=True)
        f = rank_dir / "Qwen-Qwen3-8B_ts_20260528_120000.pt.trace.json.gz"
        f.write_bytes(b"")
        files.append(f)

    analyzer = TraceLensAnalyzer(TraceLensConfig(enabled=True))
    pattern = analyzer._detect_trace_pattern(trace_dir, files)

    assert pattern is not None
    assert "rank_*" in pattern
    assert pattern == str(
        trace_dir / "rank_*" / "Qwen-Qwen3-8B_ts_20260528_120000.pt.trace.json.gz"
    )


def test_tracelens_detect_pattern_wildcards_flat_filename_rank(tmp_path):
    """The vllm/sglang flat layout (rank in the filename) still works."""
    from Magpie.modes.benchmark.config import TraceLensConfig
    from Magpie.modes.benchmark.tracelens import TraceLensAnalyzer

    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()
    f = trace_dir / "trace-rank-0.pt.trace.json.gz"
    f.write_bytes(b"")

    analyzer = TraceLensAnalyzer(TraceLensConfig(enabled=True))
    pattern = analyzer._detect_trace_pattern(trace_dir, [f])

    assert pattern == str(trace_dir / "trace-rank-*.pt.trace.json.gz")


def test_gap_analysis_detect_trace_files_handles_atom_rank_dirs(tmp_path):
    """GapAnalyzer.detect_trace_files must recurse into rank_<N>/ subdirs and
    read the rank from the directory name (atom layout)."""
    from Magpie.modes.benchmark.gap_analysis import GapAnalyzer

    trace_dir = tmp_path / "torch_trace"
    expected = {}
    for rank in (0, 1, 2):
        rank_dir = trace_dir / f"rank_{rank}"
        rank_dir.mkdir(parents=True)
        f = rank_dir / "Qwen-Qwen3-8B_ts_20260528.pt.trace.json.gz"
        f.write_bytes(b"")
        expected[rank] = f.resolve()

    found = GapAnalyzer.detect_trace_files(trace_dir)

    assert [r for r, _ in found] == [0, 1, 2]
    assert {r: p.resolve() for r, p in found} == expected


def test_gap_analysis_detect_trace_files_handles_atom_dp_rank_dirs(tmp_path):
    """Under data parallel atom names the dir dp<n>_tp<n>, so the rank must be
    read from the tp component instead of falling through to enumeration."""
    from Magpie.modes.benchmark.gap_analysis import GapAnalyzer

    trace_dir = tmp_path / "torch_trace"
    for dp in (0, 1):
        for tp in (0, 1):
            rank_dir = trace_dir / f"dp{dp}_tp{tp}"
            capture = rank_dir / "capture_traces"
            capture.mkdir(parents=True)
            # Graph-capture snapshots must not be picked up as rank traces.
            (capture / f"bs_1_rank{tp}.json.gz").write_bytes(b"")
            (rank_dir / "Qwen-Qwen3-8B_ts_20260528.pt.trace.json.gz").write_bytes(b"")

    found = GapAnalyzer.detect_trace_files(trace_dir)

    assert [r for r, _ in found] == [0, 0, 1, 1]
    assert all(p.name.endswith(".pt.trace.json.gz") for _, p in found)


def test_gap_analysis_detect_trace_files_handles_flat_rank_filenames(tmp_path):
    """The filename-encoded rank layout (vllm/sglang) still works."""
    from Magpie.modes.benchmark.gap_analysis import GapAnalyzer

    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()
    (trace_dir / "trace-rank-1.pt.trace.json.gz").write_bytes(b"")
    (trace_dir / "trace-rank-0.pt.trace.json.gz").write_bytes(b"")

    found = GapAnalyzer.detect_trace_files(trace_dir)

    assert [r for r, _ in found] == [0, 1]


def test_benchmark_result_summary_includes_sections():
    result = BenchmarkResult(success=True, framework="vllm", model="demo-model")
    result.errors.append("example warning")

    summary = result.get_summary()

    assert "Benchmark Result: VLLM" in summary
    assert "Status: SUCCESS" in summary
    assert "Errors:" in summary


def test_benchmark_config_accepts_xdit_scriptable_framework():
    cfg = BenchmarkConfig(
        framework="XDIT", model="/models/FLUX.2-dev", run_mode="local"
    )
    assert cfg.framework == "xdit"
    assert cfg.is_scriptable is True


def test_benchmark_config_serving_frameworks_not_scriptable():
    for fw in ("vllm", "sglang", "atom"):
        cfg = BenchmarkConfig(framework=fw, model="demo")
        assert cfg.is_scriptable is False


def test_benchmark_config_rejects_unknown_framework():
    with pytest.raises(ValueError):
        BenchmarkConfig(framework="rust-burn", model="demo")


def test_benchmark_config_xdit_requires_local_run_mode():
    # xDiT is server-less (scriptable) and has no Docker image, so docker/ray
    # must be rejected at config time.
    for bad_mode in ("docker", "ray"):
        with pytest.raises(ValueError):
            BenchmarkConfig(
                framework="xdit", model="/models/FLUX.2-dev", run_mode=bad_mode
            )


def test_result_parser_fails_on_quality_gate_regression(tmp_path):
    # A scriptable quality gate with passed=False must fail the benchmark so a
    # faster-but-degraded config is never accepted.
    result_path = tmp_path / "inferencex_result.json"
    result_path.write_text(
        json.dumps(
            {
                "framework": "xdit",
                "workload_kind": "scriptable",
                "throughput_unit": "img/s",
                "output_throughput": 0.42,
                "latency_s": 2.38,
                "quality_gate": {"passed": False, "ssim": 0.40, "lpips": 0.55},
            }
        ),
        encoding="utf-8",
    )

    parsed = ResultParser.parse_inferencex_result(
        result_path, framework="xdit", is_scriptable=True
    )

    assert parsed.success is False
    assert any("Quality gate not passed" in e for e in parsed.errors)


def test_result_parser_fails_on_missing_quality_gate_scriptable(tmp_path):
    # Scriptable (xDiT) runs report an image-quality gate in place of an eval;
    # the gate is the only correctness signal, so a missing gate must fail the
    # benchmark (fail-closed) rather than silently pass.
    result_path = tmp_path / "inferencex_result.json"
    result_path.write_text(
        json.dumps(
            {
                "framework": "xdit",
                "workload_kind": "scriptable",
                "throughput_unit": "img/s",
                "output_throughput": 0.42,
                "latency_s": 2.38,
            }
        ),
        encoding="utf-8",
    )

    parsed = ResultParser.parse_inferencex_result(
        result_path, framework="xdit", is_scriptable=True
    )

    assert parsed.success is False
    assert any("Quality gate missing" in e for e in parsed.errors)


def test_result_parser_fails_on_passed_absent_scriptable(tmp_path):
    # A scriptable gate dict that omits the ``passed`` flag is ambiguous and
    # must not be treated as a pass.
    result_path = tmp_path / "inferencex_result.json"
    result_path.write_text(
        json.dumps(
            {
                "framework": "xdit",
                "workload_kind": "scriptable",
                "throughput_unit": "img/s",
                "output_throughput": 0.42,
                "quality_gate": {"ssim": 0.97, "lpips": 0.01},
            }
        ),
        encoding="utf-8",
    )

    parsed = ResultParser.parse_inferencex_result(
        result_path, framework="xdit", is_scriptable=True
    )

    assert parsed.success is False
    assert any("Quality gate not passed" in e for e in parsed.errors)


def test_result_parser_serving_allows_missing_gate(tmp_path):
    # Serving frameworks (vLLM/SGLang/Atom) legitimately have no quality gate
    # (eval may not have run), so a missing gate must not fail the benchmark.
    result_path = tmp_path / "inferencex_result.json"
    result_path.write_text(
        json.dumps({"output_throughput": 123.0, "completed": 10, "duration": 5.0}),
        encoding="utf-8",
    )

    parsed = ResultParser.parse_inferencex_result(
        result_path, framework="vllm", is_scriptable=False
    )

    assert parsed.success is True
    assert not any("Quality gate" in e for e in parsed.errors)


def test_result_to_dict_omits_scriptable_fields_for_serving(tmp_path):
    # vLLM/SGLang/Atom results have no scriptable extras, so to_dict() must not
    # add workload_kind / throughput_unit / quality_gate / latency_s keys.
    result_path = tmp_path / "inferencex_result.json"
    result_path.write_text(
        json.dumps({"output_throughput": 123.0, "completed": 10, "duration": 5.0}),
        encoding="utf-8",
    )

    parsed = ResultParser.parse_inferencex_result(result_path, framework="vllm")
    d = parsed.to_dict()

    assert parsed.latency_s is None
    for key in ("workload_kind", "throughput_unit", "quality_gate", "latency_s"):
        assert key not in d


def test_result_parser_carries_scriptable_quality_gate(tmp_path):
    result_path = tmp_path / "inferencex_result.json"
    result_path.write_text(
        json.dumps(
            {
                "framework": "xdit",
                "workload_kind": "scriptable",
                "throughput_unit": "img/s",
                "output_throughput": 0.287,
                "completed": 25,
                "duration": 90.0,
                "latency_s": 3.48,
                "quality_gate": {"passed": True, "ssim": 0.97, "lpips": 0.01},
            }
        ),
        encoding="utf-8",
    )

    parsed = ResultParser.parse_inferencex_result(
        result_path, framework="xdit", is_scriptable=True
    )

    assert parsed.success is True
    assert parsed.workload_kind == "scriptable"
    assert parsed.throughput_unit == "img/s"
    assert parsed.latency_s == 3.48
    assert parsed.quality_gate == {"passed": True, "ssim": 0.97, "lpips": 0.01}
    # to_dict surfaces the extras at top level for Hyperloom to consume.
    d = parsed.to_dict()
    assert d["workload_kind"] == "scriptable"
    assert d["quality_gate"]["passed"] is True
