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

    def run(*, multiple=False, **kwargs):
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
        capture = profiling.capture_profiles if multiple else profiling.capture_profile
        return capture(**defaults)

    state.run = run
    return state


def manifest(harness):
    return json.loads((harness.directory / "capture.json").read_text())


def test_first_capture_delay_starts_after_measurement_not_warmup(harness):
    phase_start = 1_400_000_000
    harness.phases[:] = [{}, {}, {"start_ns": phase_start}]
    result = harness.run(settings={"start_seconds": 0.5}, phase_timeout_seconds=3)
    assert result["capture_started_ns"] >= phase_start + 500_000_000
    assert result["start_seconds"] == 0.5


def test_replay_ending_before_first_delayed_capture_is_a_failure(harness):
    harness.phases[:] = [
        {"start_ns": 1_000_000_000},
        {"start_ns": 1_000_000_000, "sent_end_ns": 1_200_000_000},
    ]
    with pytest.raises(profiling.ReplayFinished):
        harness.run(settings={"start_seconds": 0.5})
    assert manifest(harness)["status"] == "failed"
    assert not any(url.endswith("/start_profile") for url, _ in harness.calls)


@pytest.mark.parametrize("field", ["roofline_annotations", "detailed_annotations"])
@pytest.mark.parametrize("shape", [False, True])
def test_sglang_enhanced_request_uses_probed_field(harness, field, shape):
    capabilities = {
        "annotation_field": field,
        "shape_discovery": shape,
        "graph_capture": True,
        "graph_shape_discovery": True,
    }
    result = harness.run(
        settings={"detailed_annotations": True, "capabilities": capabilities}
    )
    body = next(body for url, body in harness.calls if url.endswith("/start_profile"))
    assert body[field] is True
    if shape:
        assert body["shape_discovery"] is True
    else:
        assert "shape_discovery" not in body
    assert body["record_shapes"] is True
    assert body["with_stack"] is True
    assert result["capabilities"] == capabilities
    assert manifest(harness)["capabilities"] == capabilities


def test_rank_trace_mapping_uses_validated_global_rank_with_repeated_local_tp(harness):
    paths = [
        harness.directory / f"fixture-TP-0-EP-{rank}.trace.json.gz" for rank in range(2)
    ]
    harness.on_start = lambda: [trace(path, rank) for rank, path in enumerate(paths)]
    result = harness.run(expected_ranks=2)
    assert result["rank_trace_files"] == {
        str(rank): [str(path)] for rank, path in enumerate(paths)
    }


def test_single_unranked_trace_is_mapped_to_validated_rank_zero(harness):
    path = harness.directory / "unranked.trace.json"
    harness.on_start = lambda: path.write_text(
        '{"traceEvents":[{"cat":"kernel","ph":"X"}]}'
    )
    result = harness.run()
    assert result["rank_trace_files"] == {"0": [str(path)]}


@pytest.mark.parametrize("location", ["source", "shared", "nested"])
def test_shared_graph_evidence_rejects_external_links(tmp_path, location):
    root = tmp_path / "capture"
    archive = root / "profile_001"
    archive.mkdir(parents=True)
    external = tmp_path / "previous-run"
    external.mkdir()
    if location == "source":
        (archive / "capture_traces").symlink_to(external, target_is_directory=True)
    elif location == "shared":
        (root / "capture_traces").symlink_to(external, target_is_directory=True)
    else:
        (archive / "capture_traces").mkdir()
        (archive / "capture_traces" / "rank0").symlink_to(
            external, target_is_directory=True
        )
    with pytest.raises(ValueError, match="external links"):
        profiling._share_graph_traces(archive, root)


@pytest.fixture
def series(harness):
    harness.start_times = []

    def start():
        harness.start_times.append(harness.clock.now)
        trace(harness.directory / "active" / "worker-rank0.pt.trace.json.gz")

    harness.on_start = start

    def run(**kwargs):
        defaults = {
            "multiple": True,
            "phase_timeout_seconds": 5,
            "settings": {
                "num_profiles": 3,
                "interval_seconds": 0.4,
                "capture_timeout_seconds": 1,
                "flush_timeout_seconds": 2,
            },
        }
        defaults.update(kwargs)
        return harness.run(**defaults)

    harness.run_series = run
    return harness


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
def test_capture_waits_for_measured_phase_then_verifies_complete_trace(
    harness, framework
):
    harness.phases = [
        {"start_ns": None, "requests_end_ns": None},
        {"start_ns": 101, "sent_end_ns": None, "requests_end_ns": None},
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


def test_series_default_retains_single_capture_layout_and_manifest(harness):
    result = harness.run(multiple=True)
    assert result["status"] == "complete"
    assert "profiles" not in result
    assert "requested_profiles" not in result
    assert Path(result["trace_files"][0]).parent == harness.directory
    assert not (harness.directory / "active").exists()


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
def test_series_archives_every_complete_capture_and_rewrites_manifests(
    series, framework
):
    result = series.run_series(framework=framework)
    assert result == manifest(series)
    assert result["status"] == "complete"
    assert result["capture_id"] == series.directory.name
    assert result["completed_profiles"] == result["requested_profiles"] == 3
    assert result["interval_seconds"] == 0.4
    assert len(result["trace_files"]) == 3
    assert series.start_times == pytest.approx([0, 0.4, 0.8])
    assert not (series.directory / "active").exists()
    for index, capture in enumerate(result["profiles"], start=1):
        archived = series.directory / f"profile_{index:03d}"
        assert capture["profile_index"] == index
        assert capture["capture_id"] == series.directory.name
        assert capture["trace_dir"] == str(archived)
        assert json.loads((archived / "capture.json").read_text()) == capture
        assert all(Path(path).is_file() for path in capture["trace_files"])
        assert all(Path(path).parent == archived for path in capture["trace_files"])
    if framework == "sglang":
        bodies = [body for url, body in series.calls if url.endswith("/start_profile")]
        assert {body["output_dir"] for body in bodies} == {
            str(series.directory / "active")
        }
    else:
        controls = [url.rsplit("/", 1)[-1] for url, _ in series.calls]
        assert controls == ["progress", "start_profile", "stop_profile"] * 3


def test_vllm_series_reset_uses_flush_budget(series, monkeypatch):
    original = profiling._http
    timeouts = []

    def http(url, *, timeout, body=None):
        if url.endswith("/stop_profile"):
            timeouts.append(timeout)
        return original(url, timeout=timeout, body=body)

    monkeypatch.setattr(profiling, "_http", http)
    result = series.run_series(
        framework="vllm",
        settings={
            "num_profiles": 2,
            "interval_seconds": 0,
            "flush_timeout_seconds": 30,
        },
    )
    assert result["status"] == "complete"
    assert timeouts == [30, 30]


def test_vllm_reset_failure_preserves_completed_trace_and_fails_series(
    series, monkeypatch
):
    original = profiling._http

    def http(url, **kwargs):
        if url.endswith("/stop_profile"):
            raise urllib.error.HTTPError(url, 500, "worker reset failed", {}, None)
        return original(url, **kwargs)

    monkeypatch.setattr(profiling, "_http", http)
    with pytest.raises(urllib.error.HTTPError):
        series.run_series(framework="vllm")
    result = manifest(series)
    assert result["status"] == "failed"
    assert result["completed_profiles"] == 0
    assert result["failed_profile"] == 1
    assert result["active_trace_dir"] == str(series.directory / "active")
    assert "worker reset failed" in result["error"]
    active = series.directory / "active"
    assert json.loads((active / "capture.json").read_text())["status"] == "complete"
    assert (active / "worker-rank0.pt.trace.json.gz").is_file()
    assert len(series.start_times) == 1


@pytest.mark.parametrize("last_flush_end", [6.5, 8])
def test_vllm_reset_cannot_extend_shared_capture_and_flush_budget(
    series, monkeypatch, last_flush_end
):
    captures = 0
    timeouts = []

    def completed_capture(**kwargs):
        nonlocal captures
        captures += 1
        active = kwargs["trace_dir"]
        active.mkdir(parents=True)
        path = active / "rank0.trace.json.gz"
        trace(path)
        series.clock.now = 0 if captures == 1 else last_flush_end
        return {
            "status": "complete",
            "trace_files": [str(path)],
            "rank_trace_files": {"0": [str(path)]},
        }

    def http(url, *, timeout, body=None):
        assert url.endswith("/stop_profile")
        timeouts.append(timeout)
        return b"ok"

    monkeypatch.setattr(profiling, "capture_profile", completed_capture)
    monkeypatch.setattr(profiling, "_http", http)
    kwargs = {
        "framework": "vllm",
        "settings": {
            "num_profiles": 2,
            "interval_seconds": 0,
            "capture_timeout_seconds": 1,
            "flush_timeout_seconds": 2,
        },
        "phase_timeout_seconds": 5,
    }
    if last_flush_end == 8:
        with pytest.raises(TimeoutError, match="reset exceeded the overall budget"):
            series.run_series(**kwargs)
        assert timeouts == [2]
        assert manifest(series)["completed_profiles"] == 1
        assert manifest(series)["failed_profile"] == 2
        assert (series.directory / "active" / "rank0.trace.json.gz").is_file()
    else:
        assert series.run_series(**kwargs)["status"] == "complete"
        assert timeouts == [2, 1.5]


def test_interval_sleep_is_never_negative_if_liveness_check_takes_time(series):
    original_sleep = series.clock.sleep

    def sleep(seconds):
        assert seconds >= 0
        original_sleep(seconds)

    def replay_alive():
        if (series.directory / "profile_001").exists() and series.clock.now == 0:
            series.clock.now += 1

    series.clock.sleep = sleep
    assert series.run_series(check_replay_alive=replay_alive)["status"] == "complete"
    assert series.start_times[1] == 1


def test_default_series_interval_is_two_hundred_seconds(series):
    result = series.run_series(settings={"num_profiles": 2}, phase_timeout_seconds=500)
    assert result["interval_seconds"] == 200
    assert series.start_times == pytest.approx([0, 200])


def test_series_interval_begins_after_complete_flush(series):
    def start():
        series.start_times.append(series.clock.now)
        (series.directory / "active" / "rank0.trace.json.gz").write_bytes(b"partial")

    def finish():
        if series.start_times and series.clock.now >= series.start_times[-1] + 0.6:
            path = series.directory / "active" / "rank0.trace.json.gz"
            if path.parent.exists():
                trace(path)

    series.on_start = start
    series.clock.on_sleep = finish
    result = series.run_series()
    assert result["status"] == "complete"
    assert series.start_times[1] >= 1.0
    assert series.start_times[2] >= 2.0


def test_later_profile_failure_preserves_archived_capture_and_records_failure(series):
    original = series.on_start

    def start():
        original()
        if len(series.start_times) == 2:
            path = series.directory / "active" / "worker-rank0.pt.trace.json.gz"
            path.write_bytes(path.read_bytes()[:-8])

    series.on_start = start
    with pytest.raises(TimeoutError, match="complete GPU traces"):
        series.run_series()
    result = manifest(series)
    assert result["status"] == "failed"
    assert result["completed_profiles"] == 1
    assert result["requested_profiles"] == 3
    assert len(result["profiles"]) == 1
    assert all(Path(path).is_file() for path in result["trace_files"])
    active = series.directory / "active"
    assert json.loads((active / "capture.json").read_text())["status"] == "failed"


def test_replay_exit_during_interval_preserves_completed_profiles(series):
    def replay_alive():
        if series.clock.now >= 0.2:
            raise RuntimeError(
                "AIPerf exited successfully before all profiles completed"
            )

    with pytest.raises(RuntimeError, match="exited successfully"):
        series.run_series(check_replay_alive=replay_alive)
    assert manifest(series)["completed_profiles"] == 1
    assert manifest(series)["status"] == "failed"
    assert len(series.start_times) == 1
    assert series.clock.now == 0.2


def test_replay_exit_before_measured_phase_stops_waiting_immediately(harness):
    harness.phases = [{"start_ns": None}]

    def replay_alive():
        if harness.clock.now >= 0.2:
            raise RuntimeError("AIPerf exited successfully before profiling")

    with pytest.raises(RuntimeError, match="exited successfully"):
        harness.run(check_replay_alive=replay_alive)
    assert harness.clock.now == 0.2
    assert not any(url.endswith("/start_profile") for url, _ in harness.calls)


def test_successful_replay_exit_during_last_flush_does_not_cancel_capture(series):
    original = series.on_start

    def replay_alive():
        if len(series.start_times) == 3:
            raise RuntimeError("replay ended during final trace flush")

    def start():
        original()
        if len(series.start_times) == 3:
            path = series.directory / "active" / "worker-rank0.pt.trace.json.gz"
            path.write_bytes(b"partial")
            series.clock.on_sleep = lambda: trace(path)

    series.on_start = start
    assert series.run_series(check_replay_alive=replay_alive)["status"] == "complete"


def test_replay_budget_is_shared_across_all_capture_intervals(series):
    with pytest.raises(TimeoutError, match="replay budget"):
        series.run_series(phase_timeout_seconds=0.6)
    assert series.start_times == pytest.approx([0, 0.4])
    assert series.clock.now == 0.6
    assert manifest(series)["completed_profiles"] == 2
    assert manifest(series)["status"] == "failed"


def test_each_capture_rechecks_that_measured_phase_has_not_ended(series):
    series.phases = [
        {"start_ns": 101},
        {"start_ns": 101, "requests_end_ns": 102},
    ]
    result = series.run_series()
    assert result["status"] == "complete"
    assert result["effective_profiles"] == 1
    assert result["stop_reason"] == "replay_finished"
    assert manifest(series)["completed_profiles"] == 1
    assert len(series.start_times) == 1


def test_series_cancellation_during_interval_keeps_completed_evidence(series):
    def alive():
        if series.clock.now >= 0.2:
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        series.run_series(check_alive=alive)
    assert manifest(series)["completed_profiles"] == 1
    assert manifest(series)["error"].startswith("KeyboardInterrupt:")


@pytest.mark.parametrize(
    "settings",
    [
        {"num_profiles": 0},
        {"num_profiles": True},
        {"num_profiles": 1.5},
        {"interval_seconds": -1},
        {"interval_seconds": True},
        {"interval_seconds": float("nan")},
        {"interval_seconds": float("inf")},
        {"interval_seconds": 10**10000},
    ],
)
def test_invalid_series_controls_are_rejected_before_network(series, settings):
    with pytest.raises(ValueError):
        series.run_series(settings=settings)
    assert series.calls == []


def test_series_allows_startup_graph_traces_in_active_directory(series):
    startup = series.directory / "active" / "graph_capture_profile"
    startup.mkdir(parents=True)
    trace(startup / "rank99.trace.json.gz", rank=99)
    result = series.run_series()
    assert result["status"] == "complete"
    assert len(result["trace_files"]) == 3
    assert (series.directory / "profile_001" / "graph_capture_profile").is_dir()


def test_series_refuses_to_overwrite_an_existing_archive(series):
    archive = series.directory / "profile_001"
    archive.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="archive already exists"):
        series.run_series()
    assert archive.is_dir()
    assert manifest(series)["completed_profiles"] == 0
    assert series.calls == []


def profile_plan(requested=3, planned=3, maximum=3, duration=1):
    return {
        "requested_profiles": requested,
        "max_profiles": maximum,
        "planned_profiles": planned,
        "measurement_duration_seconds": duration,
    }


def test_static_cap_keeps_multi_profile_layout_when_only_one_capture_is_planned(series):
    series.phases = [{"start_ns": series.clock.time_ns()}]
    settings = {"num_profiles": 99, "interval_seconds": 200}
    result = series.run_series(
        settings=settings,
        profile_plan=profile_plan(requested=99, planned=1, maximum=1, duration=120),
    )
    assert settings["num_profiles"] == 99
    assert result["requested_profiles"] == 99
    assert result["max_profiles"] == result["planned_profiles"] == 1
    assert result["effective_profiles"] == result["completed_profiles"] == 1
    assert result["status"] == "complete"
    assert result["stop_reason"] == "duration_cap"
    assert result["measurement_duration_seconds"] == 120
    assert (series.directory / "profile_001" / "capture.json").is_file()
    assert len(series.start_times) == 1


def test_single_requested_profile_keeps_layout_and_records_plan(harness):
    harness.phases = [{"start_ns": harness.clock.time_ns()}]
    result = harness.run(
        multiple=True,
        profile_plan=profile_plan(requested=1, planned=1, maximum=1, duration=120),
    )
    assert result["planned_profiles"] == result["effective_profiles"] == 1
    assert result["completed_profiles"] == 1
    assert "profiles" not in result
    assert "stop_reason" not in result
    assert Path(result["trace_files"][0]).parent == harness.directory
    assert manifest(harness) == result


def test_capture_time_reduces_effective_count_without_extending_measurement(series):
    series.phases = [{"start_ns": series.clock.time_ns()}]
    original = series.on_start

    def start():
        original()
        series.clock.now += 0.3

    series.on_start = start
    result = series.run_series(profile_plan=profile_plan(duration=0.9))
    assert result["status"] == "complete"
    assert result["planned_profiles"] == 3
    assert result["effective_profiles"] == result["completed_profiles"] == 2
    assert result["stop_reason"] == "insufficient_measurement_time"
    assert series.start_times == pytest.approx([0, 0.7])
    assert series.clock.now == pytest.approx(1.0)


def test_interval_must_fit_strictly_before_measurement_ends(series):
    series.phases = [{"start_ns": series.clock.time_ns()}]
    original = series.on_start

    def start():
        original()
        series.clock.now += 0.6

    series.on_start = start
    result = series.run_series(profile_plan=profile_plan(duration=1))
    assert result["effective_profiles"] == 1
    assert result["stop_reason"] == "insufficient_measurement_time"
    assert series.clock.now == 0.6


@pytest.mark.parametrize("when", ["before-first", "interval", "next-start"])
def test_typed_natural_replay_end_truncates_only_after_a_complete_capture(series, when):
    def replay_alive():
        if (
            when == "before-first"
            or (when == "interval" and series.clock.now >= 0.2)
            or (when == "next-start" and series.clock.now >= 0.4)
        ):
            raise profiling.ReplayFinished("AIPerf exited with status 0")

    if when == "before-first":
        with pytest.raises(profiling.ReplayFinished):
            series.run_series(check_replay_alive=replay_alive)
        assert manifest(series)["status"] == "failed"
        assert manifest(series)["completed_profiles"] == 0
    else:
        result = series.run_series(check_replay_alive=replay_alive)
        assert result["status"] == "complete"
        assert result["effective_profiles"] == 1
        assert result["stop_reason"] == "replay_finished"
        assert "error" not in result


@pytest.mark.parametrize("end_field", ["sent_end_ns", "requests_end_ns"])
def test_duration_timeout_drain_is_natural_end_before_the_next_capture(
    series, end_field
):
    series.phases = [
        {"start_ns": 101},
        {"start_ns": 101, end_field: 102, "timeout_triggered": True},
    ]
    result = series.run_series()
    assert result["status"] == "complete"
    assert result["effective_profiles"] == 1
    assert result["stop_reason"] == "replay_finished"
    assert len(series.start_times) == 1


def test_cancelled_phase_is_still_a_failure_after_complete_captures(series):
    series.phases = [
        {"start_ns": 101},
        {"start_ns": 101, "sent_end_ns": 102, "was_cancelled": True},
    ]
    with pytest.raises(RuntimeError, match="was cancelled"):
        series.run_series()
    assert manifest(series)["status"] == "failed"
    assert manifest(series)["completed_profiles"] == 1


def test_natural_end_reported_during_capture_is_not_converted_to_success(series):
    def alive():
        if len(series.start_times) == 2:
            raise profiling.ReplayFinished("unexpected replay end inside capture")

    with pytest.raises(profiling.ReplayFinished) as caught:
        series.run_series(check_alive=alive)
    assert caught.value.capture_started is True
    assert manifest(series)["status"] == "failed"
    assert manifest(series)["completed_profiles"] == 1


def test_no_capture_starts_after_the_actual_measurement_window(harness):
    harness.phases = [{"start_ns": harness.clock.time_ns()}]
    harness.clock.now = 2
    with pytest.raises(profiling.ReplayFinished, match="measurement window ended"):
        harness.run(measurement_duration_seconds=1)
    assert not any(url.endswith("/start_profile") for url, _ in harness.calls)
    assert manifest(harness)["status"] == "failed"


def test_window_is_rechecked_after_pre_start_liveness_callback(harness):
    harness.phases = [{"start_ns": harness.clock.time_ns()}]
    checks = 0

    def replay_alive():
        nonlocal checks
        checks += 1
        if checks == 2:
            harness.clock.now = 1

    with pytest.raises(profiling.ReplayFinished, match="measurement window ended"):
        harness.run(measurement_duration_seconds=1, check_replay_alive=replay_alive)
    assert checks == 2
    assert not any(url.endswith("/start_profile") for url, _ in harness.calls)
    assert "capture_started_ns" not in manifest(harness)


def test_zero_interval_has_no_static_cap_and_stops_when_replay_finishes(series):
    series.phases = [
        {"start_ns": series.clock.time_ns()},
        {"start_ns": series.clock.time_ns()},
        {"start_ns": series.clock.time_ns(), "sent_end_ns": 2_000_000_000},
    ]
    result = series.run_series(
        settings={"num_profiles": 3, "interval_seconds": 0},
        profile_plan=profile_plan(maximum=None, duration=120),
    )
    assert result["max_profiles"] is None
    assert result["effective_profiles"] == 2
    assert result["stop_reason"] == "replay_finished"
    assert series.start_times == [0, 0]


@pytest.mark.parametrize(
    "change",
    [
        {"requested_profiles": 9},
        {"planned_profiles": 0},
        {"planned_profiles": 4},
        {"planned_profiles": True},
        {"max_profiles": 2},
        {"max_profiles": True},
        {"measurement_duration_seconds": 0},
        {"measurement_duration_seconds": float("inf")},
    ],
)
def test_inconsistent_profile_plan_is_rejected_before_network(series, change):
    plan = profile_plan()
    plan.update(change)
    with pytest.raises(ValueError):
        series.run_series(profile_plan=plan)
    assert series.calls == []


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
