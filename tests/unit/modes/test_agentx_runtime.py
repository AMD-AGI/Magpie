"""Magpie owns real server processes while upstream owns only the replay client."""

import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from Magpie.modes.benchmark import agentx_runtime as runtime
from Magpie.modes.benchmark.agentx_launch import read_launch_evidence

SERVER = """import http.server, json, os, sys
from pathlib import Path
args = sys.argv[1:]
port = int(args[args.index('--port') + 1])
Path(os.environ['SERVER_RECORD']).write_text(json.dumps({'pid':os.getpid(),'argv':args,'env':os.environ.get('SGLANG_CANDIDATE')}))
if os.environ.get('SERVER_FAIL'):
    raise SystemExit(7)
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{}')
    def log_message(self, *args): pass
http.server.HTTPServer(('127.0.0.1',port),Handler).serve_forever()
"""
CLIENT = """import json, os, sys, time, urllib.request
from pathlib import Path
workspace = Path(os.environ['RESULT_DIR'])
receipt = (workspace / 'agentx_server_launch.json').read_bytes()
assert urllib.request.urlopen(os.environ['AIPERF_SERVER_URL'] + '/health').status == 200
assert 'AGENTX_LAUNCH_OVERRIDES_FILE' not in os.environ
assert 'CONC_LIST' not in os.environ
assert 'SGLANG_CANDIDATE' not in os.environ
workspace.joinpath('client.json').write_text(json.dumps({'url':os.environ['AIPERF_SERVER_URL'],'conc':os.environ['CONC'],'receipt':receipt.decode()}))
behavior = os.environ.get('CLIENT_BEHAVIOR')
if behavior == 'timeout': time.sleep(60)
if behavior == 'failure': raise SystemExit(9)
workspace.joinpath('inferencex_result.json').write_text('{"fixture":true}')
assert receipt == (workspace / 'agentx_server_launch.json').read_bytes()
"""


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def config(tmp_path):
    project = tmp_path / "InferenceX project"
    benchmarks = project / "benchmarks"
    benchmarks.mkdir(parents=True)
    server = tmp_path / "server.py"
    server.write_text(SERVER)
    client = tmp_path / "client.py"
    client.write_text(CLIENT)
    (benchmarks / "srt_agentic.sh").write_text(f'exec "$CLIENT_PYTHON" "{client}"\n')
    spec = {
        "version": 1,
        "framework": "sglang",
        "argv": [sys.executable, str(server), "--port", str(_free_port())],
        "env": {"SERVER_RECORD": str(tmp_path / "server.json")},
        "setup_commands": [],
        "health_path": "/health",
        "model_metadata": None,
        "source_files": {},
        "client_env": {},
    }
    spec["port"] = int(spec["argv"][-1])
    env = {"CONC": "1", "CLIENT_PYTHON": sys.executable, "MODEL": "fixture-model"}
    return SimpleNamespace(
        framework="sglang",
        model="fixture-model",
        run_mode="local",
        inferencex_path=str(project),
        envs=env,
        get_env_vars=lambda: dict(env),
        agentx=SimpleNamespace(
            resolved={"server-launch-spec": spec}, launch_overrides=None
        ),
        server_lifecycle=SimpleNamespace(server_ready_timeout_s=3),
        timeout_seconds=3,
        hf_cache_path=None,
        docker_image="image:fixed",
    )


def _assert_stopped(path):
    pid = json.loads(path.read_text())["pid"]
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def _git(path, *arguments):
    return subprocess.run(
        ["git", "-C", str(path), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _aiperf_checkout(config, tmp_path):
    path = Path(config.inferencex_path) / "utils/aiperf"
    path.mkdir(parents=True)
    metadata = tmp_path / "parent-git-metadata/aiperf"
    metadata.parent.mkdir()
    _git(path, "init", f"--separate-git-dir={metadata}")
    _git(
        path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--allow-empty",
        "-m",
        "first",
    )
    revision = _git(path, "rev-parse", "HEAD")
    config.agentx.resolved["server-launch-spec"]["aiperf_revision"] = revision
    return path, metadata, revision


def test_actual_aiperf_git_pin_is_checked_before_setup(config, tmp_path):
    path, _, _ = _aiperf_checkout(config, tmp_path)
    _git(
        path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--allow-empty",
        "-m",
        "changed",
    )
    marker = tmp_path / "setup-ran"
    config.agentx.resolved["server-launch-spec"]["setup_commands"] = [
        [
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ]
    ]
    result, _, error = runtime.execute_agentx(config, tmp_path / "results", "mi355x")
    assert not result.success and "AIPerf revision changed" in error
    assert not marker.exists()
    assert not (tmp_path / "server.json").exists()


def test_actual_aiperf_git_pin_is_rechecked_after_setup(config, tmp_path):
    path, _, _ = _aiperf_checkout(config, tmp_path)
    config.agentx.resolved["server-launch-spec"]["setup_commands"] = [
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-m",
            "setup changed harness",
        ]
    ]
    result, _, error = runtime.execute_agentx(config, tmp_path / "results", "mi355x")
    assert not result.success and "AIPerf revision changed" in error
    assert not (tmp_path / "server.json").exists()
    assert not (tmp_path / "results/agentx_server_launch.json").exists()


def test_docker_invalid_pin_returns_failure_without_launch(
    config, tmp_path, monkeypatch
):
    _aiperf_checkout(config, tmp_path)
    config.run_mode = "docker"
    config.agentx.resolved["server-launch-spec"]["aiperf_revision"] = "invalid"
    calls = []
    monkeypatch.setattr(
        runtime,
        "_run_owned",
        lambda *args, **kwargs: pytest.fail("must not start Docker"),
    )
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command)
        or subprocess.CompletedProcess(command, 1, "", "No such container"),
    )
    result, _, error = runtime.execute_agentx(config, tmp_path / "results", "mi355x")
    assert not result.success and "complete Git commit" in error
    assert len(calls) == 1 and calls[0][:3] == ["docker", "rm", "--force"]


def test_actual_aiperf_pin_runs_and_docker_preserves_separate_git_metadata(
    config, tmp_path
):
    _, metadata, _ = _aiperf_checkout(config, tmp_path)
    result, _, error = runtime.execute_agentx(config, tmp_path / "results", "mi355x")
    assert result.success, error
    request = _request(config, tmp_path / "docker-results")
    command = runtime._docker_command(
        request,
        image="fixture",
        name="owned",
        request_path=tmp_path / "request.json",
        runner_type="mi355x",
    )
    assert f"{metadata}:{metadata}:ro" in command
    assert not any(str(metadata.parent) + ":" in item for item in command)


def test_public_runtime_applies_literal_candidate_and_retains_receipt(
    config, tmp_path, monkeypatch
):
    marker = tmp_path / "must-not-be-created"
    literal = json.dumps({"command": f"$(touch {marker})", "text": "value with spaces"})
    config.agentx.launch_overrides = {
        "version": 1,
        "append_args": ["--json", literal],
        "env": {"SGLANG_CANDIDATE": "literal $VALUE"},
    }
    monkeypatch.setenv("AGENTX_LAUNCH_OVERRIDES_FILE", "/stale/foreign/request")
    monkeypatch.setenv("CONC_LIST", "1,2,4")
    workspace = tmp_path / "results"
    result, stdout, stderr = runtime.execute_agentx(config, workspace, "mi355x")
    assert result.success, stderr
    observed = json.loads((tmp_path / "server.json").read_text())
    assert observed["argv"][-2:] == ["--json", literal]
    assert observed["env"] == "literal $VALUE"
    assert not marker.exists()
    evidence = read_launch_evidence(config, workspace)
    assert evidence["owner"] == "magpie"
    assert (
        json.loads((workspace / "client.json").read_text())["receipt"]
        == (workspace / "agentx_server_launch.json").read_text()
    )
    _assert_stopped(tmp_path / "server.json")
    first = observed["pid"]
    result, _, error = runtime.execute_agentx(config, tmp_path / "second", "mi355x")
    assert result.success, error
    assert json.loads((tmp_path / "server.json").read_text())["pid"] != first
    _assert_stopped(tmp_path / "server.json")


@pytest.mark.parametrize("behavior", ["failure", "timeout", "server_failure"])
def test_failure_tears_down_only_owned_server(config, tmp_path, behavior):
    other = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True
    )
    try:
        if behavior == "server_failure":
            config.agentx.resolved["server-launch-spec"]["env"]["SERVER_FAIL"] = "1"
        else:
            config.envs["CLIENT_BEHAVIOR"] = behavior
        if behavior == "timeout":
            config.timeout_seconds = 0.2
        result, _, error = runtime.execute_agentx(
            config, tmp_path / "results", "mi355x"
        )
        assert not result.success
        assert (
            "failed with exit code 9" in error
            or "deadline" in error
            or "before becoming healthy" in error
        )
        _assert_stopped(tmp_path / "server.json")
        assert other.poll() is None
    finally:
        os.killpg(other.pid, signal.SIGKILL)
        other.wait()


def test_occupied_port_is_not_reused_or_killed(config, tmp_path):
    with socket.socket() as unrelated:
        unrelated.bind(
            ("127.0.0.1", config.agentx.resolved["server-launch-spec"]["port"])
        )
        unrelated.listen()
        result, _, error = runtime.execute_agentx(
            config, tmp_path / "results", "mi355x"
        )
        assert not result.success
        assert "existing servers are not reused" in error
        assert unrelated.fileno() >= 0
    assert not (tmp_path / "server.json").exists()


def test_failed_setup_invalidates_stale_receipt_without_launch(config, tmp_path):
    spec = config.agentx.resolved["server-launch-spec"]
    spec["setup_commands"] = [[sys.executable, "-c", "raise SystemExit(12)"]]
    workspace = tmp_path / "results"
    workspace.mkdir()
    (workspace / "agentx_server_launch.json").write_text('{"stale":true}')
    result, _, error = runtime.execute_agentx(config, workspace, "mi355x")
    assert not result.success and "preparation failed" in error
    assert not (workspace / "agentx_server_launch.json").exists()
    assert not (tmp_path / "server.json").exists()


def test_source_attestation_runs_after_setup(config, tmp_path):
    source = tmp_path / "candidate.py"
    source.write_text("old")
    spec = config.agentx.resolved["server-launch-spec"]
    spec["setup_commands"] = [
        [
            sys.executable,
            "-c",
            f'from pathlib import Path;Path({str(source)!r}).write_text("new")',
        ]
    ]
    config.agentx.launch_overrides = {
        "version": 1,
        "source_files": {str(source): hashlib.sha256(b"new").hexdigest()},
    }
    result, _, error = runtime.execute_agentx(config, tmp_path / "results", "mi355x")
    assert result.success, error
    _assert_stopped(tmp_path / "server.json")


def test_source_mismatch_prevents_spawn(config, tmp_path):
    source = tmp_path / "candidate.py"
    source.write_text("wrong")
    config.agentx.launch_overrides = {
        "version": 1,
        "source_files": {str(source): "a" * 64},
    }
    result, _, error = runtime.execute_agentx(config, tmp_path / "results", "mi355x")
    assert not result.success and "source changed" in error
    assert not (tmp_path / "server.json").exists()


def test_legacy_adapter_only_calls_client_functions(config, tmp_path):
    root = Path(config.inferencex_path)
    (root / "benchmarks/srt_agentic.sh").unlink()
    (root / "benchmarks/benchmark_lib.sh").write_text("""
resolve_trace_source() { echo resolve; }
install_agentic_deps() { echo install; }
build_replay_cmd() { echo build; }
run_agentic_replay_and_write_outputs() { echo replay; }
""")
    result, output, error = runtime.execute_agentx(
        config, tmp_path / "results", "mi355x"
    )
    assert result.success, error
    assert output.splitlines() == ["resolve", "install", "build", "replay"]
    _assert_stopped(tmp_path / "server.json")


def test_docker_uses_unique_owned_container_and_in_container_worker(
    config, tmp_path, monkeypatch
):
    calls = []
    config.run_mode = "docker"

    def run(command, **kwargs):
        calls.append(command)
        path = Path(command[-1])
        request = json.loads(path.read_text())
        assert "--entrypoint" in command and command[-3].endswith("agentx_runtime.py")
        assert request["server_spec"] == config.agentx.resolved["server-launch-spec"]
        assert "--privileged" not in command
        (Path(request["workspace"]) / "agentx_runtime_result.json").write_text(
            json.dumps({"success": True, "errors": []})
        )
        return 0

    def cleanup(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runtime, "_run_owned", run)
    monkeypatch.setattr(runtime.subprocess, "run", cleanup)
    for index in range(2):
        result, _, error = runtime.execute_agentx(
            config, tmp_path / str(index), "mi355x"
        )
        assert result.success, error
    names = [calls[i][calls[i].index("--name") + 1] for i in (0, 2)]
    assert names[0] != names[1]
    assert calls[1] == ["docker", "rm", "--force", names[0]]
    assert calls[3] == ["docker", "rm", "--force", names[1]]


def test_docker_timeout_still_removes_only_its_container(config, tmp_path, monkeypatch):
    calls = []
    config.run_mode = "docker"

    def timeout(command, **kwargs):
        calls.append(command)
        raise TimeoutError("deadline")

    monkeypatch.setattr(runtime, "_run_owned", timeout)
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda command, **kw: calls.append(command)
        or subprocess.CompletedProcess(command, 0, "", ""),
    )
    result, _, error = runtime.execute_agentx(config, tmp_path / "results", "mi355x")
    assert not result.success and "deadline" in error
    name = calls[0][calls[0].index("--name") + 1]
    assert calls[1] == ["docker", "rm", "--force", name]


@pytest.mark.parametrize(
    "failure", ["missing_result", "invalid_result", "container_exit", "cleanup"]
)
def test_docker_incomplete_execution_or_cleanup_cannot_pass(
    config, tmp_path, monkeypatch, failure
):
    config.run_mode = "docker"
    cleanup_calls = []

    def run(command, **kwargs):
        payload = json.loads(Path(command[-1]).read_text())
        result = Path(payload["workspace"]) / "agentx_runtime_result.json"
        if failure == "invalid_result":
            result.write_text("not JSON")
        elif failure != "missing_result":
            result.write_text('{"success":true,"errors":[]}')
        return 9 if failure == "container_exit" else 0

    def cleanup(command, **kwargs):
        cleanup_calls.append(command)
        return subprocess.CompletedProcess(
            command,
            1 if failure == "cleanup" else 0,
            "",
            "failed to remove owned container",
        )

    monkeypatch.setattr(runtime, "_run_owned", run)
    monkeypatch.setattr(runtime.subprocess, "run", cleanup)
    result, _, error = runtime.execute_agentx(config, tmp_path / "results", "mi355x")
    assert not result.success
    assert error
    assert len(cleanup_calls) == 1
    assert cleanup_calls[0][:3] == ["docker", "rm", "--force"]
    assert cleanup_calls[0][-1].startswith("magpie-agentx-")


def test_health_deadline_stops_owned_unready_server(config, tmp_path):
    spec = config.agentx.resolved["server-launch-spec"]
    server = Path(spec["argv"][1])
    server.write_text(
        SERVER.replace("self.send_response(200)", "self.send_response(503)")
    )
    config.server_lifecycle.server_ready_timeout_s = 0.25
    result, _, error = runtime.execute_agentx(config, tmp_path / "results", "mi355x")
    assert not result.success and "did not become healthy" in error
    _assert_stopped(tmp_path / "server.json")
    assert not (tmp_path / "results/client.json").exists()


def _request(config, workspace):
    workspace.mkdir()
    return {
        "workspace": str(workspace),
        "inferencex_path": config.inferencex_path,
        "server_spec": config.agentx.resolved["server-launch-spec"],
        "overrides": config.agentx.launch_overrides,
        "client_env": config.get_env_vars(),
        "ready_timeout": 3,
        "client_timeout": 5,
        "gpu_env": {},
    }


def test_standalone_container_worker_needs_only_standard_library(config, tmp_path):
    workspace = tmp_path / "worker"
    payload = _request(config, workspace)
    path = workspace / "request.json"
    path.write_text(json.dumps(payload))
    completed = subprocess.run(
        [sys.executable, "-S", str(Path(runtime.__file__)), "--worker", str(path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads((workspace / "agentx_runtime_result.json").read_text())["success"]
    _assert_stopped(tmp_path / "server.json")


def test_worker_interrupt_stops_server_and_client(config, tmp_path):
    import time

    config.envs["CLIENT_BEHAVIOR"] = "timeout"
    workspace = tmp_path / "worker"
    path = workspace / "request.json"
    path.parent.mkdir()
    payload = {
        "workspace": str(workspace),
        "inferencex_path": config.inferencex_path,
        "server_spec": config.agentx.resolved["server-launch-spec"],
        "overrides": None,
        "client_env": config.get_env_vars(),
        "ready_timeout": 3,
        "client_timeout": 60,
        "gpu_env": {},
    }
    path.write_text(json.dumps(payload))
    worker = subprocess.Popen(
        [sys.executable, "-S", str(Path(runtime.__file__)), "--worker", str(path)],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not (workspace / "client.json").is_file() and time.monotonic() < deadline:
            assert worker.poll() is None
            time.sleep(0.05)
        assert (workspace / "client.json").is_file()
        worker.terminate()
        assert worker.wait(timeout=5) == 1
        outcome = json.loads((workspace / "agentx_runtime_result.json").read_text())
        assert not outcome["success"] and "signal" in outcome["errors"][0]
        _assert_stopped(tmp_path / "server.json")
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()


def test_tampered_setup_is_not_executed(config, tmp_path):
    marker = tmp_path / "executed"
    setup = tmp_path / "setup.py"
    setup.write_text(f"from pathlib import Path;Path({str(marker)!r}).touch()")
    spec = config.agentx.resolved["server-launch-spec"]
    spec["setup_commands"] = [[sys.executable, str(setup)]]
    spec["source_files"] = {str(setup): "a" * 64}
    result, _, error = runtime.execute_agentx(config, tmp_path / "results", "mi355x")
    assert not result.success and "recipe input changed" in error
    assert not marker.exists()


def test_metadata_checks_actual_local_bytes_and_context(config, tmp_path):
    path = tmp_path / "config.json"
    contents = b'{"max_position_embeddings":8192}'
    path.write_bytes(contents)
    spec = copy.deepcopy(config.agentx.resolved["server-launch-spec"])
    spec["model_metadata"] = {
        "source": str(path),
        "sha256": hashlib.sha256(contents).hexdigest(),
        "native_context_length": 8192,
        "max_model_len": 4096,
    }
    runtime._verify_model_metadata(spec)
    spec["model_metadata"]["max_model_len"] = 16384
    with pytest.raises(ValueError, match="context limits"):
        runtime._verify_model_metadata(spec)
    path.write_text("{}")
    with pytest.raises(ValueError, match="metadata changed"):
        runtime._verify_model_metadata(spec)


def test_remote_metadata_is_public_bounded_and_matches_served_revision(
    config, monkeypatch
):
    import io

    revision = "a" * 40
    content = b'{"max_position_embeddings":8192}'
    spec = copy.deepcopy(config.agentx.resolved["server-launch-spec"])
    spec["argv"] += ["--revision", revision]
    spec["model_metadata"] = {
        "source": f"https://huggingface.co/org/model/resolve/{revision}/config.json",
        "sha256": hashlib.sha256(content).hexdigest(),
        "revision": revision,
        "native_context_length": 8192,
        "max_model_len": 4096,
    }
    calls = []

    class Response(io.BytesIO):
        def read(self, size=-1):
            assert size == 4 * 1024 * 1024 + 1
            return super().read(size)

    def open_public(url, timeout):
        assert isinstance(url, str) and timeout == 30
        calls.append(url)
        return Response(content)

    monkeypatch.setattr(
        runtime.urllib.request,
        "build_opener",
        lambda *args: SimpleNamespace(open=open_public),
    )
    runtime._verify_model_metadata(spec)
    assert calls == [spec["model_metadata"]["source"]]
    spec["argv"][-1] = "b" * 40
    with pytest.raises(ValueError, match="server revision"):
        runtime._verify_model_metadata(spec)
    assert len(calls) == 1


def test_remote_metadata_rejects_external_redirect():
    with pytest.raises(ValueError, match="left the public"):
        runtime._PublicModelRedirect().redirect_request(
            None, None, 302, "", {}, "https://example.com/config.json"
        )


def test_server_overlay_is_not_injected_into_client(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("PYTHONPATH", "/ambient/framework")
    request = {
        "client_env": {
            "PATH": "/candidate/bin",
            "PYTHONPATH": "/candidate/overlay",
            "CONC": "4",
            "HIP_VISIBLE_DEVICES": "2",
        }
    }
    spec = {
        "env": {
            "PATH": "/candidate/bin",
            "PYTHONPATH": "/candidate/overlay",
            "HIP_VISIBLE_DEVICES": "2",
        },
        "port": 8888,
    }
    env = runtime._client_environment(request, spec, tmp_path, tmp_path)
    assert env["PATH"] == "/usr/bin:/bin"
    assert env["PYTHONPATH"] == str(tmp_path)
    assert env["CONC"] == "4"
    assert env["HIP_VISIBLE_DEVICES"] == "2"
