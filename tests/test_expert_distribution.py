"""Portable expert-count capture keeps source-rank data and isolates samples."""

import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "Magpie/scripts/benchmark/expert_distribution.py"
spec = importlib.util.spec_from_file_location("expert_distribution", SCRIPT)
expert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(expert)


def _server_info(**overrides):
    return {
        "tp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
        "enable_dp_attention": False,
        "enable_eplb": False,
        "expert_distribution_recorder_mode": "stat",
        "expert_distribution_recorder_buffer_size": 1,
        **overrides,
    }


@pytest.fixture(autouse=True)
def _mock_server_info(monkeypatch):
    monkeypatch.setattr(expert, "_get_server_info", lambda _: _server_info())
    monkeypatch.setattr(expert, "DUMP_WAIT_SECONDS", 0)


def _fake_clock(monkeypatch, on_sleep=lambda: None):
    ticks = [0.0]

    def sleep(seconds):
        ticks[0] += seconds
        on_sleep()

    monkeypatch.setattr(expert, "time", SimpleNamespace(monotonic=lambda: ticks[0], sleep=sleep))
    monkeypatch.setattr(expert, "DUMP_WAIT_SECONDS", 1)
    return ticks


def _legacy_dump(rank=0):
    return {
        "last_physical_to_logical_map": [[0, 1, 1]],
        "records": [
            {
                "forward_pass_id": 17,
                "rank": rank,
                "gatherer_key": "primary",
                "global_physical_count": [[2 + rank, 0, 4]],
            }
        ],
    }


def _dump(rank=0, passes=1):
    return {
        "schema_version": 2,
        "aggregation": "sum_over_capture",
        "gatherer_key": "primary",
        "rank": rank,
        "forward_pass_count": passes,
        "first_forward_pass_id": 17 if passes else None,
        "last_forward_pass_id": 17 + passes - 1 if passes else None,
        "global_physical_count": [[2 + rank, 0, 4]] if passes else [[0, 0, 0]],
        "last_physical_to_logical_map": [[0, 1, 1]],
    }


@pytest.mark.parametrize("bad", [[], [[1], [2, 3]], [[-1, 0]], [[0.5, 1]]])
def test_invalid_count_matrix_does_not_publish_artifact(tmp_path, bad):
    dump = _dump()
    dump["global_physical_count"] = bad
    destination = tmp_path / expert.ARTIFACT
    with pytest.raises(ValueError):
        expert.export_aggregate_dumps([tmp_path / "rank0.pt"], destination, lambda _: dump)
    assert not destination.exists()
    assert not destination.with_suffix(".json.tmp").exists()


def test_safe_cpu_tensor_load(monkeypatch, tmp_path):
    calls = []
    fake = SimpleNamespace(load=lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setitem(sys.modules, "torch", fake)
    path = tmp_path / "dump.pt"
    expert._load_dump(path)
    assert calls == [((path,), {"map_location": "cpu", "weights_only": True})]


def test_capture_excludes_previous_sample_dumps(monkeypatch, tmp_path):
    native = tmp_path / "native"
    native.mkdir()
    stale = native / "expert_distribution_recorder_previous_0.pt"
    stale.touch()
    output = tmp_path / "output"
    calls = []

    def request(url, action):
        calls.append(action)
        if action == "dump":
            (native / "expert_distribution_recorder_current_0.pt").touch()

    loaded = []
    monkeypatch.setattr(expert, "_request", request)
    monkeypatch.setattr(expert, "_load_dump", lambda path: loaded.append(path) or _dump())
    expert.capture("start", "http://localhost:30000", output, native)
    result = expert.capture("finish", "http://localhost:30000", output, native)
    assert calls == ["start", "stop", "dump", "start", "stop", "dump"]
    assert [path.name for path in loaded] == [
        "expert_distribution_recorder_current_0.pt",
        "expert_distribution_recorder_current_0.pt",
    ]
    assert result["status"] == "available"
    assert result["artifact_path"] == expert.ARTIFACT
    assert result["scope"] == "client_including_warmup"
    assert stale.exists()
    assert not loaded[0].exists()
    assert json.loads((output / expert.SUMMARY).read_text()) == result


def test_failed_start_still_stops_and_does_not_publish_state(monkeypatch, tmp_path):
    calls = []

    def request(url, action):
        calls.append(action)
        if action == "start":
            raise TimeoutError("test timeout")

    monkeypatch.setattr(expert, "_request", request)
    with pytest.raises(TimeoutError, match="test timeout"):
        expert.capture("start", "http://localhost:30000", tmp_path, tmp_path)
    assert not (tmp_path / expert.STATE).exists()
    assert not (tmp_path / expert.SUMMARY).exists()
    with pytest.raises(FileNotFoundError):
        expert.capture("finish", "http://localhost:30000", tmp_path, tmp_path)
    assert calls == ["start", "stop"]


def test_no_worker_dumps_fails_the_probe(monkeypatch, tmp_path):
    monkeypatch.setattr(expert, "_request", lambda *args: None)
    with pytest.raises(TimeoutError, match="Expected 1 complete worker dumps; found 0"):
        expert.capture("start", "http://localhost:30000", tmp_path, tmp_path)
    assert not (tmp_path / expert.SUMMARY).exists()


@pytest.mark.parametrize("failure", ["payload", "metadata_write"])
def test_failed_export_keeps_native_dump(monkeypatch, tmp_path, failure):
    dumps = []

    def request(url, action):
        if action == "dump":
            path = tmp_path / f"expert_distribution_recorder_{len(dumps)}.pt"
            path.touch()
            dumps.append(path)

    monkeypatch.setattr(expert, "_request", request)
    monkeypatch.setattr(expert, "_load_dump", lambda path: _dump(passes=0))
    expert.capture("start", "http://localhost:30000", tmp_path, tmp_path)
    monkeypatch.setattr(expert, "_load_dump", lambda path: {} if failure == "payload" else _dump())
    original_write = expert._write_json

    def write(path, value):
        if failure == "metadata_write" and path.name == expert.SUMMARY:
            raise OSError("metadata disk full")
        original_write(path, value)

    monkeypatch.setattr(expert, "_write_json", write)
    with pytest.raises((ValueError, OSError)):
        expert.capture("finish", "http://localhost:30000", tmp_path, tmp_path)
    assert dumps[-1].exists()
    assert not (tmp_path / expert.SUMMARY).exists()


def test_start_invalidates_previous_capture_before_probing(monkeypatch, tmp_path):
    for name in (expert.ARTIFACT, expert.STATE, expert.SUMMARY):
        (tmp_path / name).write_text("stale")
    monkeypatch.setattr(expert, "_request", lambda *args: None)
    with pytest.raises(TimeoutError):
        expert.capture("start", "http://localhost:30000", tmp_path, tmp_path)
    assert all(not (tmp_path / name).exists() for name in (expert.ARTIFACT, expert.STATE, expert.SUMMARY))


def test_failed_stop_gets_one_cleanup_attempt(monkeypatch, tmp_path):
    calls = []

    def request(url, action):
        calls.append(action)
        if action == "stop":
            raise TimeoutError("test stop timeout")

    monkeypatch.setattr(expert, "_request", request)
    expert._write_json(tmp_path / expert.STATE, {"existing": [], "expected_count": 1})
    with pytest.raises(TimeoutError, match="test stop timeout"):
        expert.capture("finish", "http://localhost:30000", tmp_path, tmp_path)
    assert calls == ["stop", "stop"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"pp_size": 2},
        {"dp_size": 2},
        {"tp_size": 0},
        {"expert_distribution_recorder_mode": "per_pass"},
        {"expert_distribution_recorder_buffer_size": 2},
        {"enable_eplb": True},
    ],
)
def test_unsupported_runtime_fails_before_start(monkeypatch, tmp_path, overrides):
    calls = []
    monkeypatch.setattr(expert, "_get_server_info", lambda _: _server_info(**overrides))
    monkeypatch.setattr(expert, "_request", lambda *args: calls.append(args))
    with pytest.raises(ValueError):
        expert.capture("start", "http://localhost:30000", tmp_path, tmp_path)
    assert calls == []


def test_dp_attention_uses_actual_tp_worker_count():
    info = _server_info(tp_size=8, dp_size=8, enable_dp_attention=True)
    assert expert._expected_dump_count(info) == 8
    assert expert._expected_dump_count({"server_args": info}) == 8


def test_delayed_worker_dump_and_repeated_native_rank_ids(monkeypatch, tmp_path):
    first = tmp_path / "expert_distribution_recorder_first_0.pt"
    second = tmp_path / "expert_distribution_recorder_second_0.pt"
    first.touch()
    ticks = _fake_clock(monkeypatch, on_sleep=second.touch)
    monkeypatch.setattr(expert, "_load_dump", lambda _: _dump(rank=0))
    destination = tmp_path / expert.ARTIFACT
    metadata, paths = expert._export_when_ready(tmp_path, set(), destination, 2)
    artifact = json.loads(destination.read_text())
    assert metadata == {"rank_count": 2, "record_count": 2}
    assert ticks[0] == 0.25
    assert set(paths) == {first, second}
    assert [stream["rank"] for stream in artifact["ranks"]] == [0, 0]
    assert artifact["stream_identity"] == "source_file"
    assert artifact["rank_identity"] == "native_recorder_rank"
    assert len({stream["source_file"] for stream in artifact["ranks"]}) == 2


def test_partial_torch_save_is_retried(monkeypatch, tmp_path):
    path = tmp_path / "expert_distribution_recorder_current_0.pt"
    path.touch()
    calls = []

    def load(path):
        calls.append(path)
        if len(calls) == 1:
            raise RuntimeError("PytorchStreamReader failed finding central directory")
        return _dump()

    ticks = _fake_clock(monkeypatch)
    monkeypatch.setattr(expert, "_load_dump", load)
    metadata, paths = expert._export_when_ready(tmp_path, set(), tmp_path / expert.ARTIFACT, 1)
    assert metadata["rank_count"] == 1
    assert ticks[0] == 0.25
    assert calls == [path, path]


def test_missing_worker_timeout_preserves_native_evidence(monkeypatch, tmp_path):
    path = tmp_path / "expert_distribution_recorder_current_0.pt"

    def request(url, action):
        if action == "dump":
            path.touch()

    ticks = _fake_clock(monkeypatch)
    monkeypatch.setattr(expert, "_request", request)
    expert._write_json(tmp_path / expert.STATE, {"existing": [], "expected_count": 2})
    with pytest.raises(TimeoutError, match="Expected 2 complete worker dumps; found 1"):
        expert.capture("finish", "http://localhost:30000", tmp_path, tmp_path)
    assert ticks[0] == 1
    assert path.exists()
    assert not (tmp_path / expert.ARTIFACT).exists()


def test_permanent_load_error_is_not_retried(monkeypatch, tmp_path):
    path = tmp_path / "expert_distribution_recorder_current_0.pt"
    path.touch()
    ticks = _fake_clock(monkeypatch)

    def load(path):
        raise RuntimeError("Unsupported checkpoint contents")

    monkeypatch.setattr(expert, "_load_dump", load)
    with pytest.raises(RuntimeError, match="Unsupported checkpoint contents"):
        expert._export_when_ready(tmp_path, set(), tmp_path / expert.ARTIFACT, 1)
    assert ticks[0] == 0


def test_rpc_timeout_is_fixed_and_only_dump_gets_the_completion_budget(monkeypatch):
    calls = []

    def urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return nullcontext(SimpleNamespace(read=lambda limit: b"{}"))

    monkeypatch.setattr(expert, "urlopen", urlopen)
    for action in ("start", "stop", "dump"):
        expert._request("http://localhost:30000", action)
    assert [timeout for _, timeout in calls] == [30, 30, 40]


def test_aggregate_export_keeps_totals_pass_ids_and_worker_identity(tmp_path):
    paths = [tmp_path / "worker0.pt", tmp_path / "worker1.pt"]
    destination = tmp_path / expert.ARTIFACT
    result = expert.export_aggregate_dumps(paths, destination, lambda _: _dump(passes=7))
    artifact = json.loads(destination.read_text())
    assert result == {"rank_count": 2, "record_count": 14}
    assert artifact["schema_version"] == 2
    assert artifact["recorder_mode"] == "aggregate"
    assert artifact["aggregation"] == "sum_over_capture"
    assert artifact["rank_reduction"] == "none"
    assert artifact["ranks"][0]["global_physical_count"] == [[2, 0, 4]]
    assert artifact["ranks"][0]["last_physical_to_logical_map"] == [[0, 1, 1]]
    assert artifact["ranks"][0]["forward_pass_count"] == 7
    assert artifact["ranks"][0]["first_forward_pass_id"] == 17
    assert artifact["ranks"][0]["last_forward_pass_id"] == 23
    assert [worker["rank"] for worker in artifact["ranks"]] == [0, 0]
    assert len({worker["source_file"] for worker in artifact["ranks"]}) == 2
    assert all("records" not in worker for worker in artifact["ranks"])


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"schema_version": 1}, id="legacy"),
        pytest.param({"aggregation": "mean"}, id="aggregation"),
        pytest.param({"gatherer_key": "secondary"}, id="gatherer"),
        pytest.param({"last_physical_to_logical_map": [[0]]}, id="map-shape"),
        pytest.param({"global_physical_count": [[-1, 0, 4]]}, id="negative-count"),
        pytest.param({"global_physical_count": [[1.2, 0, 4]]}, id="float-count"),
        pytest.param({"forward_pass_count": 0, "first_forward_pass_id": None,
                      "last_forward_pass_id": None, "global_physical_count": [[0]*3]},
                     id="empty-capture"),
        pytest.param({"forward_pass_count": True}, id="bool-passes"),
        pytest.param({"rank": -1}, id="negative-rank"),
        pytest.param({"last_forward_pass_id": 18}, id="short-id-span"),
        pytest.param({"last_forward_pass_id": 1}, id="reversed-ids"),
        pytest.param({"global_physical_count": [[0]*3]}, id="zero-assignments"),
    ],
)
def test_aggregate_export_fails_closed_on_invalid_plugin_payload(tmp_path, change):
    dump = {**_dump(passes=3), **change}
    destination = tmp_path / expert.ARTIFACT
    with pytest.raises((ValueError, KeyError)):
        expert.export_aggregate_dumps([tmp_path / "worker.pt"], destination, lambda _: dump)
    assert not destination.exists()



def test_skipped_native_forward_ids_remain_valid_when_monotonic_count_fits(tmp_path):
    dump = _dump(passes=2)
    dump["last_forward_pass_id"] = 21
    assert (
        expert.export_aggregate_dumps(
            [tmp_path / "worker.pt"],
            tmp_path / expert.ARTIFACT,
            lambda _: dump,
        )["record_count"]
        == 2
    )


def test_probe_verifies_all_workers_then_resets_before_actual_capture(monkeypatch, tmp_path):
    native = tmp_path / "native"
    native.mkdir()
    stale = native / "expert_distribution_recorder_stale_0.pt"
    stale.touch()
    calls, dumps = [], [0]

    def request(url, action):
        calls.append(action)
        if action == "dump":
            dumps[0] += 1
            for worker in range(2):
                (native / f"expert_distribution_recorder_capture{dumps[0]}_{worker}.pt").touch()

    def load(path):
        assert path != stale
        return _dump(rank=0, passes=0 if "capture1" in path.name else 4)

    monkeypatch.setattr(expert, "_get_server_info", lambda _: _server_info(tp_size=2))
    monkeypatch.setattr(expert, "_request", request)
    monkeypatch.setattr(expert, "_load_dump", load)
    output = tmp_path / "output"
    expert.capture("start", "http://server", output, native)
    assert calls == ["start", "stop", "dump", "start"]
    assert list(native.glob("*capture1*")) == []
    assert stale.exists()
    assert not (output / expert.ARTIFACT).exists()
    assert json.loads((output / expert.STATE).read_text()) == {
        "existing": [stale.name], "expected_count": 2,
    }
    assert not (output / expert.SUMMARY).exists()
    result = expert.capture("finish", "http://server", output, native)
    assert calls == ["start", "stop", "dump", "start", "stop", "dump"]
    assert result["status"] == "available" and result["record_count"] == 8
    artifact = json.loads((output / expert.ARTIFACT).read_text())
    assert [worker["forward_pass_count"] for worker in artifact["ranks"]] == [4, 4]
    assert list(native.iterdir()) == [stale]
    assert not (output / ".expert_distribution_probe.json").exists()


def test_mixed_or_missing_plugin_never_starts_actual_capture(monkeypatch, tmp_path):
    calls = []

    def request(url, action):
        calls.append(action)
        if action == "dump":
            for worker in range(2):
                (tmp_path / f"expert_distribution_recorder_probe_{worker}.pt").touch()

    monkeypatch.setattr(expert, "_get_server_info", lambda _: _server_info(tp_size=2))
    monkeypatch.setattr(expert, "_request", request)
    monkeypatch.setattr(
        expert,
        "_load_dump",
        lambda p: _legacy_dump() if p.name.endswith("_1.pt") else _dump(passes=0),
    )
    with pytest.raises(ValueError, match="plugin is unavailable"):
        expert.capture("start", "http://server", tmp_path, tmp_path)
    assert calls == ["start", "stop", "dump", "stop"]
    assert len(list(tmp_path.glob("*.pt"))) == 2  # Retain failed-probe evidence.
    assert not (tmp_path / expert.STATE).exists()
    assert not (tmp_path / expert.SUMMARY).exists()
    with pytest.raises(FileNotFoundError):
        expert.capture("finish", "http://server", tmp_path, tmp_path)
    assert calls.count("start") == 1


def test_dp_attention_idle_worker_zero_matrix_is_preserved(tmp_path):
    active, idle = _dump(rank=0), _dump(rank=1)
    idle["global_physical_count"] = [[0, 0, 0]]
    destination = tmp_path / expert.ARTIFACT
    expert.export_aggregate_dumps(
        [tmp_path / "active.pt", tmp_path / "idle.pt"],
        destination,
        lambda path: active if path.name == "active.pt" else idle,
    )
    assert json.loads(destination.read_text())["ranks"][1]["global_physical_count"] == [[0, 0, 0]]


def test_zero_pass_idle_worker_is_valid_with_global_assignments(tmp_path):
    destination = tmp_path / expert.ARTIFACT
    metadata = expert.export_aggregate_dumps(
        [tmp_path / "active.pt", tmp_path / "idle.pt"],
        destination,
        lambda path: _dump() if path.name == "active.pt" else _dump(rank=1, passes=0),
    )
    assert metadata == {"rank_count": 2, "record_count": 1}
    idle = json.loads(destination.read_text())["ranks"][1]
    assert idle["forward_pass_count"] == 0
    assert idle["first_forward_pass_id"] is idle["last_forward_pass_id"] is None
    assert idle["global_physical_count"] == [[0, 0, 0]]
