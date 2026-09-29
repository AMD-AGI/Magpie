"""Measured-phase trace captures require complete, fresh GPU rank evidence."""

from __future__ import annotations

import gzip
import http.server
import json
import threading
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from Magpie.modes.benchmark import agentx_profiling as profiling


class Clock:
    def __init__(self):
        self.now = 0.0
        self.on_sleep = lambda: None

    def monotonic(self):
        return self.now

    def time_ns(self):
        return 1_000_000_000 + int(self.now * 1e9)

    def sleep(self, seconds):
        self.now += seconds
        self.on_sleep()


def trace(path, rank=0, *, kernel=True, compress=True, event=None):
    payload = {
        "distributedInfo": {"rank": rank},
        "traceEvents": [event or {"cat": "kernel" if kernel else "cpu_op", "ph": "X"}],
    }
    encoded = json.dumps(payload).encode()
    path.write_bytes(gzip.compress(encoded) if compress else encoded)


@pytest.fixture
def harness(monkeypatch, tmp_path):
    clock = Clock()
    monkeypatch.setattr(profiling, "time", clock)
    directory = tmp_path / "capture-unique"
    calls = []
    phases = [{"start_ns": 100, "requests_end_ns": None}]
    on_start = lambda: trace(directory / "worker-rank0.pt.trace.json.gz")
    state = SimpleNamespace(
        clock=clock,
        directory=directory,
        calls=calls,
        phases=phases,
        on_start=on_start,
        check_alive=lambda: None,
    )

    def http(url, *, timeout, body=None):
        calls.append((url, body))
        assert timeout > 0
        if url.endswith("/api/progress"):
            stats = state.phases.pop(0) if len(state.phases) > 1 else state.phases[0]
            if isinstance(stats, BaseException):
                raise stats
            return json.dumps({"phases": {"profiling": stats}}).encode()
        if url.endswith("/start_profile"):
            state.on_start()
        return b"ok"

    monkeypatch.setattr(profiling, "_http", http)

    def run(**kwargs):
        defaults = {
            "framework": "sglang",
            "server_url": "http://127.0.0.1:8888",
            "progress_url": "http://127.0.0.1:9999/api/progress",
            "trace_dir": directory,
            "settings": {
                "num_steps": 20,
                "capture_timeout_seconds": 1,
                "flush_timeout_seconds": 2,
            },
            "expected_ranks": 1,
            "phase_timeout_seconds": 1,
            "check_alive": state.check_alive,
            "client_started_ns": 100,
        }
        defaults.update(kwargs)
        return profiling.capture_profile(**defaults)

    state.run = run
    return state


def manifest(harness):
    return json.loads((harness.directory / "capture.json").read_text())


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
def test_capture_waits_for_measured_phase_then_verifies_complete_trace(
    harness, framework
):
    harness.phases = [
        {"start_ns": None, "requests_end_ns": None},
        {"start_ns": 101, "sent_end_ns": 102, "requests_end_ns": None},
    ]
    result = harness.run(framework=framework)
    assert result["status"] == "complete"
    assert result["phase_start_ns"] == 101
    assert result["num_steps"] == 20
    assert len(result["trace_files"]) == 1
    assert manifest(harness) == result
    assert [url.rsplit("/", 1)[-1] for url, _ in harness.calls] == [
        "progress",
        "progress",
        "start_profile",
    ]
    body = harness.calls[-1][1]
    if framework == "sglang":
        assert body == {
            "num_steps": 20,
            "output_dir": str(harness.directory),
            "activities": ["CPU", "GPU"],
            "profile_prefix": harness.directory.name,
        }
    else:
        assert body == {}


@pytest.mark.parametrize(
    "stats",
    [
        {"start_ns": 101, "requests_end_ns": 102},
        {"start_ns": 99, "requests_end_ns": None},
        {"start_ns": True, "requests_end_ns": None},
        None,
    ],
)
def test_finished_stale_or_malformed_phase_never_triggers_profiler(harness, stats):
    harness.phases = [stats]
    with pytest.raises((RuntimeError, TimeoutError)):
        harness.run()
    assert not any(url.endswith("/start_profile") for url, _ in harness.calls)
    assert manifest(harness)["status"] == "failed"


def test_transient_progress_connection_failure_retries_without_profiling_warmup(
    harness,
):
    harness.phases = [ConnectionRefusedError("starting"), {"start_ns": 101}]
    assert harness.run()["status"] == "complete"
    assert len(harness.calls) == 3


@pytest.mark.parametrize("start_ns", [None, 101])
def test_cancelled_phase_never_triggers_profiler(harness, start_ns):
    harness.phases = [{"start_ns": start_ns, "was_cancelled": True}]
    with pytest.raises(RuntimeError, match="was cancelled"):
        harness.run()
    assert not any(url.endswith("/start_profile") for url, _ in harness.calls)
    assert manifest(harness)["status"] == "failed"


def test_stale_cancelled_phase_is_ignored_while_current_client_starts(harness):
    harness.phases = [
        {"start_ns": 99, "was_cancelled": True},
        {"start_ns": 101, "was_cancelled": False},
    ]
    assert harness.run()["status"] == "complete"


@pytest.mark.parametrize("when", ["phase", "capture", "flush"])
def test_child_failure_and_cancellation_stop_pending_capture_and_record_failure(
    harness, when
):
    harness.on_start = lambda: None
    if when == "flush":
        harness.on_start = lambda: (
            harness.directory / "rank0.trace.json.gz"
        ).write_bytes(b"partial")

    def check_alive():
        if when == "phase" or any(
            url.endswith("/start_profile") for url, _ in harness.calls
        ):
            raise RuntimeError("AIPerf exited with status 9")

    harness.check_alive = check_alive
    with pytest.raises(RuntimeError, match="status 9"):
        harness.run()
    assert manifest(harness)["status"] == "failed"
    stopped = any(url.endswith("/stop_profile") for url, _ in harness.calls)
    assert stopped is (when != "phase")


def test_keyboard_interrupt_persists_capture_failure_and_attempts_stop(harness):
    harness.on_start = lambda: None

    def check_alive():
        if any(url.endswith("/start_profile") for url, _ in harness.calls):
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        harness.run(check_alive=check_alive)
    assert manifest(harness)["error"].startswith("KeyboardInterrupt:")
    assert harness.calls[-1][0].endswith("/stop_profile")


def test_http_rejection_cannot_be_reported_as_success(harness, monkeypatch):
    original = profiling._http

    def rejected(url, **kwargs):
        if url.endswith("/start_profile"):
            raise urllib.error.HTTPError(url, 400, "unsupported profiler", {}, None)
        return original(url, **kwargs)

    monkeypatch.setattr(profiling, "_http", rejected)
    with pytest.raises(urllib.error.HTTPError):
        harness.run()
    assert manifest(harness)["status"] == "failed"
    assert harness.calls[-1][0].endswith("/stop_profile")


def test_no_trace_times_out_and_cleanup_rejection_keeps_original_failure(
    harness, monkeypatch
):
    harness.on_start = lambda: None
    original = profiling._http

    def already_stopped(url, **kwargs):
        if url.endswith("/stop_profile"):
            raise urllib.error.HTTPError(url, 400, "already auto-stopped", {}, None)
        return original(url, **kwargs)

    monkeypatch.setattr(profiling, "_http", already_stopped)
    with pytest.raises(TimeoutError, match="no trace"):
        harness.run()
    assert "HTTPError" in manifest(harness)["cleanup"]
    assert "no trace" in manifest(harness)["error"]


def test_start_request_has_full_capture_budget_for_scheduler_ack(harness, monkeypatch):
    original = profiling._http
    timeouts = []

    def http(url, *, timeout, body=None):
        if url.endswith("/start_profile"):
            timeouts.append(timeout)
            harness.clock.now += 3
        return original(url, timeout=timeout, body=body)

    monkeypatch.setattr(profiling, "_http", http)
    assert harness.run(settings={"capture_timeout_seconds": 10})["status"] == "complete"
    assert timeouts == [10]


@pytest.mark.parametrize("ack_delay", [0.7, 1.2])
def test_start_request_time_counts_toward_capture_deadline(harness, ack_delay):
    def start():
        harness.clock.now += ack_delay
        if ack_delay > 1:
            trace(harness.directory / "rank0.trace.json.gz")

    harness.on_start = start
    message = "start exceeded" if ack_delay > 1 else "no trace"
    with pytest.raises(TimeoutError, match=message):
        harness.run()
    assert harness.clock.now == max(1, ack_delay)
    assert manifest(harness)["status"] == "failed"
    assert harness.calls[-1][0].endswith("/stop_profile")


@pytest.mark.parametrize(
    "broken",
    [
        "missing-rank",
        "duplicate-rank",
        "cpu-only",
        "truncated-gzip",
        "bad-json",
        "trailing-json",
    ],
)
def test_incomplete_or_invalid_rank_traces_cannot_complete(harness, broken):
    expected = 1

    def write():
        path = harness.directory / "rank0.trace.json.gz"
        trace(path)
        if broken == "missing-rank":
            return
        if broken == "duplicate-rank":
            trace(harness.directory / "copy-rank0.trace.json.gz")
        elif broken == "cpu-only":
            trace(path, kernel=False)
        elif broken == "truncated-gzip":
            path.write_bytes(path.read_bytes()[:-8])
        elif broken == "bad-json":
            path.write_bytes(
                gzip.compress(b'{"traceEvents": [{"cat":"kernel","ph":"X"}')
            )
        elif broken == "trailing-json":
            path.write_bytes(
                gzip.compress(b'{"traceEvents":[{"cat":"kernel","ph":"X"}]} garbage')
            )

    if broken in {"missing-rank", "duplicate-rank"}:
        expected = 2
    harness.on_start = write
    with pytest.raises(TimeoutError, match="complete GPU traces"):
        harness.run(expected_ranks=expected)
    assert manifest(harness)["status"] == "failed"


def test_flush_has_its_own_budget_and_waits_for_all_ranks(harness):
    def begin():
        (harness.directory / "rank0.trace.json.gz").write_bytes(b"partial")

    def finish():
        if harness.clock.now >= 1.2:
            trace(harness.directory / "rank0.trace.json.gz")
            trace(harness.directory / "rank1.trace.json.gz", rank=1)

    harness.on_start = begin
    harness.clock.on_sleep = finish
    result = harness.run(expected_ranks=2)
    assert result["status"] == "complete"
    assert len(result["trace_files"]) == 2
    assert (
        harness.clock.now > 1
    )  # Capture bound has expired, but flush remains allowed.


def test_sglang_local_tp_names_use_distributed_global_rank(harness):
    def write():
        trace(harness.directory / "capture-TP-0-PP-0.trace.json.gz", rank=0)
        trace(harness.directory / "capture-TP-0-PP-1.trace.json.gz", rank=1)

    harness.on_start = write
    result = harness.run(expected_ranks=2)
    assert result["status"] == "complete"
    assert len(result["trace_files"]) == 2


def test_several_complete_stage_traces_per_rank_preserve_rank_coverage(harness):
    def write():
        trace(harness.directory / "capture-TP-0-EXTEND.trace.json.gz", rank=0)
        trace(harness.directory / "capture-TP-0-DECODE.trace.json.gz", rank=0)
        trace(harness.directory / "capture-TP-1-DECODE.trace.json.gz", rank=1)

    harness.on_start = write
    result = harness.run(expected_ranks=2)
    assert result["status"] == "complete"
    assert len(result["trace_files"]) == 3


def test_explicit_global_filename_rank_conflicting_with_metadata_is_rejected(harness):
    harness.on_start = lambda: trace(
        harness.directory / "dp0_pp0_tp0_rank0.trace.json.gz", rank=1
    )
    with pytest.raises(TimeoutError, match="complete GPU traces"):
        harness.run(expected_ranks=2)


def test_complete_cpu_companion_and_graph_capture_are_not_worker_rank_evidence(harness):
    def write():
        trace(harness.directory / "rank0.trace.json.gz")
        trace(harness.directory / "frontend.trace.json.gz", kernel=False)
        trace(harness.directory / "graph_capture_rank99.trace.json.gz", rank=99)

    harness.on_start = write
    result = harness.run()
    assert [Path(item).name for item in result["trace_files"]] == [
        "rank0.trace.json.gz"
    ]


@pytest.mark.parametrize(
    "startup_directory", ["graph_capture_profile", "capture_traces"]
)
def test_server_startup_trace_is_allowed_but_excluded_from_capture(
    harness, startup_directory
):
    startup = harness.directory / startup_directory
    startup.mkdir(parents=True)
    # Actual SGLang startup files use .json.gz, and other exports can use the
    # normal trace suffix. Neither must enter measured-phase rank evidence.
    trace(startup / "cuda_graph_capture-CudaGraphRunner-TP-0.json.gz")
    trace(startup / "rank99.trace.json.gz", rank=99)
    (startup / "incomplete.trace.json.gz").write_bytes(b"unfinished startup trace")
    result = harness.run()
    assert [Path(item).name for item in result["trace_files"]] == [
        "worker-rank0.pt.trace.json.gz"
    ]
    assert result["status"] == "complete"
    assert harness.calls[-1][0].endswith("/start_profile")


@pytest.mark.parametrize(
    "startup_directory", ["graph_capture_profile", "capture_traces"]
)
def test_startup_traces_alone_cannot_complete_capture(harness, startup_directory):
    startup = harness.directory / startup_directory
    startup.mkdir(parents=True)
    trace(startup / "rank0.trace.json.gz")
    harness.on_start = lambda: None
    with pytest.raises(TimeoutError, match="no trace"):
        harness.run()
    assert manifest(harness)["status"] == "failed"
    assert manifest(harness)["trace_files"] == []


def test_empty_server_created_subdirectories_are_allowed(harness):
    (harness.directory / "worker-output" / "nested").mkdir(parents=True)
    assert harness.run()["status"] == "complete"


@pytest.mark.parametrize("subdirectory", ["", "worker-output", "trace_split"])
def test_stale_capture_directory_is_rejected_without_overwriting_it(
    harness, subdirectory
):
    parent = harness.directory / subdirectory
    parent.mkdir(parents=True)
    trace(parent / "rank0.trace.json.gz")
    with pytest.raises(ValueError, match="fresh unique"):
        harness.run()
    assert not (harness.directory / "capture.json").exists()
    assert harness.calls == []


def test_streaming_validation_handles_oversized_event_fields(harness):
    event = {"args": {"stack": "long data " * 150_000}, "cat": "kernel", "ph": "X"}
    harness.on_start = lambda: trace(
        harness.directory / "rank0.trace.json.gz", event=event
    )
    assert harness.run()["status"] == "complete"


@pytest.mark.parametrize(
    "change",
    [
        {"settings": {"num_steps": True}},
        {"settings": {"num_steps": 0}},
        {"settings": {"flush_timeout_seconds": float("nan")}},
        {"settings": {"capture_timeout_seconds": 0}},
        {"settings": {"capture_timeout_seconds": 10**10000}},
        {"phase_timeout_seconds": -1},
        {"expected_ranks": 0},
        {"framework": "atom"},
        {"client_started_ns": False},
    ],
)
def test_invalid_capture_controls_are_rejected_before_network(harness, change):
    with pytest.raises(ValueError):
        harness.run(**change)
    assert harness.calls == []


def test_real_http_controller_uses_progress_get_and_json_profile_post(tmp_path):
    directory = tmp_path / "real-capture"
    calls = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(("GET", self.path, None))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                b'{"phases":{"profiling":{"start_ns":101,"requests_end_ns":null}}}'
            )

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(("POST", self.path, body))
            trace(Path(body["output_dir"]) / "rank0.trace.json.gz")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"started")

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        result = profiling.capture_profile(
            framework="sglang",
            server_url=url,
            progress_url=url + "/api/progress",
            trace_dir=directory,
            settings={"num_steps": 5},
            expected_ranks=1,
            phase_timeout_seconds=2,
            check_alive=lambda: None,
            client_started_ns=100,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert result["status"] == "complete"
    assert [(method, path) for method, path, _ in calls] == [
        ("GET", "/api/progress"),
        ("POST", "/start_profile"),
    ]
    assert calls[-1][2]["num_steps"] == 5
