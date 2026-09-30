"""AgentX diagnostics collect framework traces from a real managed process."""

import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from Magpie.modes.benchmark import agentx_runtime as runtime
from Magpie.modes.benchmark.agentx_launch import (
    digest,
    prepare_server_launch,
    read_launch_evidence,
)
from Magpie.modes.benchmark.config import TorchProfilerConfig

SERVER = r"""
import gzip, http.server, json, os, sys, threading, time
from pathlib import Path

fixture = Path(os.environ['PROFILE_FIXTURE'])
port = int(sys.argv[sys.argv.index('--port') + 1])
profiler_config = (
    json.loads(sys.argv[sys.argv.index('--profiler-config') + 1])
    if '--profiler-config' in sys.argv else None
)
record = {'pid': os.getpid(), 'argv': sys.argv[1:],
          'trace_dir': (profiler_config['torch_profiler_dir'] if profiler_config
                        else os.environ.get('SGLANG_TORCH_PROFILER_DIR'))}
fixture.joinpath('server.json').write_text(json.dumps(record))
with fixture.joinpath('server_launches.jsonl').open('a') as handle:
    handle.write(json.dumps(record) + '\n')
if os.environ.get('PROFILE_BEHAVIOR') == 'startup_trace':
    startup = Path(record['trace_dir']) / 'graph_capture_profile'
    startup.mkdir(parents=True, exist_ok=True)
    with gzip.open(startup / 'cuda_graph_capture-fixture-TP-0.json.gz', 'wt') as handle:
        json.dump({'traceEvents': [
            {'cat': 'kernel', 'ph': 'X', 'name': 'startup_only_kernel',
             'ts': 1, 'dur': 999999, 'pid': 0, 'tid': 0}
        ]}, handle)
remaining = None
profile_active = False
capture_index = 0
flushed_index = 0
capture_directory = Path(record['trace_dir'])
lock = threading.Lock()

def export_traces(index, directory):
    global profile_active, flushed_index
    directory.mkdir(parents=True, exist_ok=True)
    behavior = os.environ.get('PROFILE_BEHAVIOR', '')
    for rank in range(2):
        if rank == 1 and behavior == 'missing_rank':
            return
        path = directory / ('rank' + str(rank) + '.trace.json')
        path.write_text('{"traceEvents":[')
        time.sleep(0.15)
        if behavior == 'truncated':
            return
        path.write_text(json.dumps({'traceEvents': [
            {'cat': 'kernel', 'ph': 'X', 'name': 'fixture_gpu_kernel_' + str(index),
             'ts': 1000, 'dur': 20, 'pid': rank, 'tid': 0,
             'args': {'capture_index': index}}
        ]}))
    with lock:
        flushed_index = index
        if profiler_config is None:
            profile_active = False
    fixture.joinpath('trace_flushed').write_text(str(time.time_ns()))

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{}')

    def do_POST(self):
        global remaining, profile_active, capture_index, capture_directory
        body = self.rfile.read(int(self.headers.get('Content-Length', '0')))
        if self.path == '/start_profile':
            payload = json.loads(body or '{}')
            if os.environ.get('PROFILE_BEHAVIOR') == 'reject':
                self.send_response(503)
                self.end_headers()
                return
            with lock:
                if profile_active:
                    self.send_response(409)
                    self.end_headers()
                    self.wfile.write(b'profiler still active; stop must reset it')
                    return
                capture_index += 1
                profile_active = True
                remaining = (profiler_config['max_iterations'] if profiler_config
                             else payload['num_steps'])
                capture_directory = Path(
                    record['trace_dir'] if profiler_config else payload['output_dir']
                )
                start = {'body': payload, 'pid': os.getpid(),
                         'capture_index': capture_index,
                         'phase': fixture.joinpath('phase').read_text(),
                         'time_ns': time.time_ns()}
                fixture.joinpath('start.json').write_text(json.dumps(start))
                with fixture.joinpath('starts.jsonl').open('a') as handle:
                    handle.write(json.dumps(start) + '\n')
        elif self.path == '/fixture_step':
            with lock:
                if remaining is not None and remaining > 0:
                    remaining -= 1
                    if remaining == 0:
                        threading.Thread(target=export_traces,
                                         args=(capture_index, capture_directory),
                                         daemon=True).start()
        elif self.path == '/stop_profile':
            with lock:
                with fixture.joinpath('stops.jsonl').open('a') as handle:
                    handle.write(json.dumps({'capture_index': capture_index,
                                             'flushed_index': flushed_index,
                                             'time_ns': time.time_ns()}) + '\n')
                profile_active = False
                remaining = None
            fixture.joinpath('stop_profile').write_text(str(time.time_ns()))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{}')

    def log_message(self, *args):
        pass

http.server.ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()
"""

CLIENT = r"""
import http.server, json, os, threading, time, urllib.request
from pathlib import Path

fixture = Path(os.environ['PROFILE_FIXTURE'])
workspace = Path(os.environ['RESULT_DIR'])
fixture.joinpath('client.json').write_text(json.dumps({'pid': os.getpid()}))
with fixture.joinpath('client_launches.jsonl').open('a') as handle:
    handle.write(json.dumps({'pid': os.getpid()}) + '\n')
assert 'SGLANG_TORCH_PROFILER_DIR' not in os.environ
phase = {'warmup': {'start_ns': time.time_ns(), 'requests_end_ns': None}}
fixture.joinpath('phase').write_text('warmup')

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({'phases': phase}).encode())

    def log_message(self, *args):
        pass

api = http.server.ThreadingHTTPServer((os.environ['AIPERF_API_SERVER_HOST'],
                                     int(os.environ['AIPERF_API_SERVER_PORT'])), Handler)
threading.Thread(target=api.serve_forever, daemon=True).start()
time.sleep(0.25)
assert not fixture.joinpath('start.json').exists(), 'profiler started during warmup'
phase['warmup']['requests_end_ns'] = time.time_ns()
fixture.joinpath('phase').write_text('profiling')
phase['profiling'] = {'start_ns': time.time_ns(), 'sent_end_ns': None,
                      'requests_end_ns': None, 'requests_completed': 0}
deadline = time.monotonic() + float(os.environ.get('PROFILE_TEST_DURATION', '1.5'))
while time.monotonic() < deadline:
    end_after_starts = int(os.environ.get('PROFILE_TEST_END_AFTER_STARTS', '0'))
    starts_file = fixture.joinpath('starts.jsonl')
    if end_after_starts and starts_file.exists():
        if len(starts_file.read_text().splitlines()) >= end_after_starts:
            break
    request = urllib.request.Request(os.environ['AIPERF_SERVER_URL'] + '/fixture_step',
                                     data=b'{}', method='POST')
    with urllib.request.urlopen(request, timeout=1) as response:
        assert response.status == 200
    phase['profiling']['requests_completed'] += 1
    time.sleep(0.025)
phase['profiling']['sent_end_ns'] = time.time_ns()
time.sleep(0.1)
phase['profiling']['requests_end_ns'] = time.time_ns()
fixture.joinpath('client_complete').write_text(str(time.time_ns()))
workspace.joinpath('inferencex_result.json').write_text('{"fixture":true}')
api.shutdown()
"""


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def diagnostic_config(tmp_path):
    project = tmp_path / "InferenceX project"
    benchmarks = project / "benchmarks"
    benchmarks.mkdir(parents=True)
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    server = tmp_path / "server.py"
    server.write_text(SERVER)
    client = tmp_path / "client.py"
    client.write_text(CLIENT)
    (benchmarks / "srt_agentic.sh").write_text(f'exec "$CLIENT_PYTHON" "{client}"\n')
    port = _free_port()
    spec = {
        "version": 1,
        "framework": "sglang",
        "argv": [sys.executable, str(server), "--port", str(port), "--tp", "2"],
        "env": {"PROFILE_FIXTURE": str(fixture)},
        "setup_commands": [],
        "port": port,
        "health_path": "/health",
        "model_metadata": None,
        "source_files": {},
        "client_env": {"PROFILE_FIXTURE": str(fixture)},
    }
    env = {
        "CONC": "1",
        "TP": "2",
        "CLIENT_PYTHON": sys.executable,
        "MODEL": "fixture-model",
        "PROFILE_FIXTURE": str(fixture),
    }
    return SimpleNamespace(
        framework="sglang",
        model="fixture-model",
        run_mode="local",
        inferencex_path=str(project),
        envs=env,
        get_env_vars=lambda: dict(env),
        agentx=SimpleNamespace(
            resolved={"server-launch-spec": spec, "tp": 2},
            launch_overrides=None,
        ),
        profiler=SimpleNamespace(
            torch_profiler=TorchProfilerConfig(
                enabled=True,
                num_steps=3,
                capture_timeout_seconds=2,
                flush_timeout_seconds=2,
            )
        ),
        server_lifecycle=SimpleNamespace(server_ready_timeout_s=3),
        timeout_seconds=6,
        hf_cache_path=None,
        docker_image="image:fixed",
    )


def _fixture(config):
    return Path(config.envs["PROFILE_FIXTURE"])


def _assert_owned_processes_stopped(config):
    for name in ("server.json", "client.json"):
        pid = json.loads((_fixture(config) / name).read_text())["pid"]
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def _capture(result):
    assert result.profiling_enabled
    assert result.benchmark_valid is False
    assert result.publishable is False
    return result.agentx_metrics["profile_capture"]


def test_real_managed_profile_waits_for_measurement_and_complete_rank_traces(
    diagnostic_config, tmp_path
):
    config = diagnostic_config
    original_spec = copy.deepcopy(config.agentx.resolved["server-launch-spec"])
    workspace = tmp_path / "results"
    result, _, error = runtime.execute_agentx(config, workspace, "mi355x")
    assert result.success, error
    assert _capture(result)["status"] == "complete"
    fixture = _fixture(config)
    start = json.loads((fixture / "start.json").read_text())
    assert start["phase"] == "profiling"
    assert start["body"]["num_steps"] == 3
    assert int((fixture / "trace_flushed").read_text()) < int(
        (fixture / "client_complete").read_text()
    )
    receipt = read_launch_evidence(config, workspace)
    trace_dir = Path(receipt["torch_profiler"]["trace_dir"])
    assert trace_dir.parent == workspace / "torch_trace"
    for rank in range(2):
        trace = json.loads((trace_dir / f"rank{rank}.trace.json").read_text())
        assert trace["traceEvents"][0]["cat"] == "kernel"
    assert config.agentx.resolved["server-launch-spec"] == original_spec
    _assert_owned_processes_stopped(config)


def test_repeated_profile_uses_new_capture_directory(diagnostic_config, tmp_path):
    config = diagnostic_config
    workspace = tmp_path / "repeated"
    captures = []
    for _ in range(2):
        for marker in ("start.json", "trace_flushed", "client_complete"):
            (_fixture(config) / marker).unlink(missing_ok=True)
        result, _, error = runtime.execute_agentx(config, workspace, "mi355x")
        assert result.success, error
        assert _capture(result)["status"] == "complete"
        captures.append(read_launch_evidence(config, workspace)["torch_profiler"])
        _assert_owned_processes_stopped(config)
    assert captures[0]["capture_id"] != captures[1]["capture_id"]
    assert captures[0]["trace_dir"] != captures[1]["trace_dir"]
    assert all(Path(capture["trace_dir"]).is_dir() for capture in captures)


def _json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _multiple_profiles(config, *, framework="sglang", interval_seconds=0.05):
    config.framework = framework
    config.agentx.resolved["server-launch-spec"]["framework"] = framework
    config.profiler.torch_profiler.num_profiles = 3
    config.profiler.torch_profiler.interval_seconds = interval_seconds
    config.envs["PROFILE_TEST_DURATION"] = "4"


def _assert_single_server_and_client(config, captures):
    fixture = _fixture(config)
    servers = _json_lines(fixture / "server_launches.jsonl")
    clients = _json_lines(fixture / "client_launches.jsonl")
    assert len(servers) == len(clients) == 1
    assert {capture["pid"] for capture in captures} == {servers[0]["pid"]}
    _assert_owned_processes_stopped(config)


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
def test_multiple_profiles_share_live_processes_and_preserve_every_capture(
    diagnostic_config, tmp_path, framework
):
    config = diagnostic_config
    _multiple_profiles(config, framework=framework)
    workspace = tmp_path / "multiple"
    result, _, error = runtime.execute_agentx(config, workspace, "mi355x")
    assert result.success, error
    capture = _capture(result)
    assert capture["status"] == "complete"
    assert capture["requested_profiles"] == capture["completed_profiles"] == 3
    assert len(capture["trace_files"]) == 6
    starts = _json_lines(_fixture(config) / "starts.jsonl")
    assert [start["capture_index"] for start in starts] == [1, 2, 3]
    assert all(start["phase"] == "profiling" for start in starts)
    receipt = read_launch_evidence(config, workspace)
    root = Path(receipt["torch_profiler"]["trace_dir"])
    assert receipt["torch_profiler"]["num_profiles"] == 3
    assert receipt["torch_profiler"]["interval_seconds"] == 0.05
    for index, profile in enumerate(capture["profiles"], 1):
        directory = root / f"profile_{index:03d}"
        assert Path(profile["trace_dir"]) == directory
        assert profile["status"] == "complete"
        assert len(profile["trace_files"]) == 2
        assert json.loads((directory / "capture.json").read_text()) == profile
        for path in profile["trace_files"]:
            assert Path(path).parent == directory
            event = json.loads(Path(path).read_text())["traceEvents"][0]
            assert event["args"]["capture_index"] == index
    assert not (root / "active").exists()
    if framework == "vllm":
        # This fake retains the real vLLM active state after its step limit.
        # Another start returns HTTP 409 unless stop reset that state first.
        stops = _json_lines(_fixture(config) / "stops.jsonl")
        assert [stop["capture_index"] for stop in stops] == [1, 2, 3]
        assert all(stop["flushed_index"] == stop["capture_index"] for stop in stops)
        for index in range(2):
            assert stops[index]["time_ns"] < starts[index + 1]["time_ns"]
    _assert_single_server_and_client(config, starts)


def test_early_replay_end_preserves_completed_profiles_and_failed_capture(
    diagnostic_config, tmp_path
):
    config = diagnostic_config
    _multiple_profiles(config)
    config.envs["PROFILE_TEST_END_AFTER_STARTS"] = "2"
    workspace = tmp_path / "partial"
    result, _, error = runtime.execute_agentx(config, workspace, "mi355x")
    assert not result.success
    assert error
    capture = _capture(result)
    assert capture["status"] == "failed"
    assert capture["requested_profiles"] == 3
    assert capture["completed_profiles"] == 1
    assert len(capture["trace_files"]) == 2
    assert all(Path(path).is_file() for path in capture["trace_files"])
    root = Path(read_launch_evidence(config, workspace)["torch_profiler"]["trace_dir"])
    failed = json.loads((root / "active" / "capture.json").read_text())
    assert failed["status"] == "failed"
    assert failed["cleanup"] == "stop_requested"
    assert (_fixture(config) / "client_complete").exists()
    starts = _json_lines(_fixture(config) / "starts.jsonl")
    assert len(starts) == 2
    _assert_single_server_and_client(config, starts)


def test_client_exit_during_interval_fails_before_waiting_for_next_capture(
    diagnostic_config, tmp_path
):
    config = diagnostic_config
    _multiple_profiles(config, interval_seconds=10)
    config.envs["PROFILE_TEST_DURATION"] = "1.0"
    started = time.monotonic()
    result, _, error = runtime.execute_agentx(config, tmp_path / "interval", "mi355x")
    elapsed = time.monotonic() - started
    assert not result.success
    assert "replay ended" in error.lower()
    assert elapsed < 4, "an exited replay must not wait out the configured interval"
    capture = _capture(result)
    assert capture["status"] == "failed"
    assert capture["completed_profiles"] == 1
    assert all(Path(path).is_file() for path in capture["trace_files"])
    assert (_fixture(config) / "client_complete").exists()
    starts = _json_lines(_fixture(config) / "starts.jsonl")
    assert len(starts) == 1
    _assert_single_server_and_client(config, starts)


def test_server_startup_graph_trace_is_excluded_from_measured_capture(
    diagnostic_config, tmp_path
):
    config = diagnostic_config
    config.agentx.resolved["server-launch-spec"]["env"][
        "PROFILE_BEHAVIOR"
    ] = "startup_trace"
    workspace = tmp_path / "startup-trace"
    result, _, error = runtime.execute_agentx(config, workspace, "mi355x")
    assert result.success, error
    capture = _capture(result)
    assert capture["status"] == "complete"
    receipt = read_launch_evidence(config, workspace)
    trace_dir = Path(receipt["torch_profiler"]["trace_dir"])
    assert (
        trace_dir / "graph_capture_profile" / "cuda_graph_capture-fixture-TP-0.json.gz"
    ).is_file()
    assert {Path(path).name for path in capture["trace_files"]} == {
        "rank0.trace.json",
        "rank1.trace.json",
    }
    assert all("graph_capture_profile" not in path for path in capture["trace_files"])
    _assert_owned_processes_stopped(config)


@pytest.mark.parametrize("behavior", ["reject", "missing_rank", "truncated"])
def test_profile_failure_cannot_be_reported_as_success(
    diagnostic_config, tmp_path, behavior
):
    config = diagnostic_config
    config.agentx.resolved["server-launch-spec"]["env"]["PROFILE_BEHAVIOR"] = behavior
    result, _, error = runtime.execute_agentx(config, tmp_path / "failed", "mi355x")
    assert not result.success
    assert _capture(result)["status"] == "failed"
    assert error
    assert (_fixture(config) / "stop_profile").is_file()
    _assert_owned_processes_stopped(config)


def test_profile_worker_runs_without_site_packages(
    diagnostic_config, tmp_path, monkeypatch
):
    def standalone_worker(request):
        workspace = Path(request["workspace"])
        request_file = workspace / "request.json"
        request_file.write_text(json.dumps(request))
        completed = subprocess.run(
            [
                sys.executable,
                "-S",
                str(Path(runtime.__file__)),
                "--worker",
                str(request_file),
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return json.loads((workspace / "agentx_runtime_result.json").read_text())

    monkeypatch.setattr(runtime, "_execute_local", standalone_worker)
    result, _, error = runtime.execute_agentx(
        diagnostic_config, tmp_path / "stdlib-worker", "mi355x"
    )
    assert result.success, error
    assert _capture(result)["status"] == "complete"
    _assert_owned_processes_stopped(diagnostic_config)


def test_profile_receipt_rejects_rehashed_capture_setting_tampering(
    diagnostic_config, tmp_path
):
    config = diagnostic_config
    workspace = tmp_path / "receipt"
    result, _, error = runtime.execute_agentx(config, workspace, "mi355x")
    assert result.success, error
    receipt = read_launch_evidence(config, workspace)
    receipt["torch_profiler"]["num_steps"] = 99
    receipt["evidence_sha256"] = digest(
        {name: value for name, value in receipt.items() if name != "evidence_sha256"}
    )
    (workspace / "agentx_server_launch.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="profil|capture|specification"):
        read_launch_evidence(config, workspace)
    _assert_owned_processes_stopped(config)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--profiler_config", '{"max_iterations":1}'],
        ['--profiler_config={"max_iterations":1}'],
    ],
)
def test_vllm_profiler_underscore_alias_cannot_override_capture(tmp_path, arguments):
    argv = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--profiler-config",
        '{"profiler":"torch","max_iterations":20}',
    ]
    with pytest.raises(ValueError):
        prepare_server_launch(
            argv,
            {},
            {"version": 1, "append_args": arguments},
            "vllm",
            tmp_path,
        )
    assert not (tmp_path / "agentx_server_launch.json").exists()
