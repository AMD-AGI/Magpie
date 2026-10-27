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
    monkeypatch.delenv("MAGPIE_EXPERT_DISTRIBUTION_DUMP_WAIT_SECONDS", raising=False)
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


def test_unavailable_start_still_stops_and_does_not_dump(monkeypatch, tmp_path):
    calls = []

    def request(url, action):
        calls.append(action)
        if action == "start":
            raise TimeoutError("test timeout")

    monkeypatch.setattr(expert, "_request", request)
    result = expert.capture("start", "http://localhost:30000", tmp_path, tmp_path)
    assert result["status"] == "unavailable"
    assert "TimeoutError" in result["reason"]
    result = expert.capture("finish", "http://localhost:30000", tmp_path, tmp_path)
    assert result["status"] == "unavailable"
    assert calls == ["start", "stop"]


def test_no_records_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(expert, "_request", lambda *args: None)
    expert.capture("start", "http://localhost:30000", tmp_path, tmp_path)
    result = expert.capture("finish", "http://localhost:30000", tmp_path, tmp_path)
    assert result["status"] == "unavailable"
    assert "Expected 1 complete worker dumps; found 0" in result["reason"]
    assert "artifact_path" not in result


def test_failed_export_keeps_native_dump(monkeypatch, tmp_path):
    path = tmp_path / "expert_distribution_recorder_current_0.pt"

    def request(url, action):
        if action == "dump":
            path.touch()

    monkeypatch.setattr(expert, "_request", request)
    monkeypatch.setattr(expert, "_load_dump", lambda path: {})
    expert.capture("start", "http://localhost:30000", tmp_path, tmp_path)
    result = expert.capture("finish", "http://localhost:30000", tmp_path, tmp_path)
    assert result["status"] == "unavailable"
    assert path.exists()


def test_start_removes_previous_portable_artifact(monkeypatch, tmp_path):
    artifact = tmp_path / expert.ARTIFACT
    artifact.write_text("stale")
    monkeypatch.setattr(expert, "_request", lambda *args: None)
    expert.capture("start", "http://localhost:30000", tmp_path, tmp_path)
    assert not artifact.exists()


def test_failed_stop_gets_one_cleanup_attempt(monkeypatch, tmp_path):
    calls = []

    def request(url, action):
        calls.append(action)
        if action == "stop":
            raise TimeoutError("test stop timeout")

    monkeypatch.setattr(expert, "_request", request)
    expert.capture("start", "http://localhost:30000", tmp_path, tmp_path)
    result = expert.capture("finish", "http://localhost:30000", tmp_path, tmp_path)
    assert result["status"] == "unavailable"
    assert calls == ["start", "stop", "stop"]


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
def test_unsupported_runtime_is_unavailable_before_start(monkeypatch, tmp_path, overrides):
    calls = []
    monkeypatch.setattr(expert, "_get_server_info", lambda _: _server_info(**overrides))
    monkeypatch.setattr(expert, "_request", lambda *args: calls.append(args))
    result = expert.capture("start", "http://localhost:30000", tmp_path, tmp_path)
    assert result["status"] == "unavailable"
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
    monkeypatch.setattr(expert, "_get_server_info", lambda _: _server_info(tp_size=2))
    expert.capture("start", "http://localhost:30000", tmp_path, tmp_path)
    result = expert.capture("finish", "http://localhost:30000", tmp_path, tmp_path)
    assert result["status"] == "unavailable"
    assert "Expected 2 complete worker dumps; found 1" in result["reason"]
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


def test_configured_wait_accepts_worker_completing_after_default_deadline(monkeypatch, tmp_path):
    first = tmp_path / "expert_distribution_recorder_first_0.pt"
    second = tmp_path / "expert_distribution_recorder_second_1.pt"
    ticks = _fake_clock(monkeypatch)
    original_sleep = expert.time.sleep

    def sleep(seconds):
        original_sleep(seconds)
        if ticks[0] >= 12:
            second.touch()

    expert.time.sleep = sleep
    monkeypatch.setenv("MAGPIE_EXPERT_DISTRIBUTION_DUMP_WAIT_SECONDS", "300")
    monkeypatch.setattr(expert, "_get_server_info", lambda _: _server_info(tp_size=2))
    monkeypatch.setattr(expert, "_load_dump", lambda _: _dump())
    monkeypatch.setattr(
        expert, "_request", lambda url, action: first.touch() if action == "dump" else None
    )
    expert.capture("start", "http://localhost:30000", tmp_path, tmp_path)
    result = expert.capture("finish", "http://localhost:30000", tmp_path, tmp_path)
    assert result["status"] == "available" and result["rank_count"] == 2
    assert ticks[0] == 12.25
    assert not first.exists() and not second.exists()


def test_configured_wait_timeout_remains_bounded_and_keeps_native_dump(monkeypatch, tmp_path):
    ticks = _fake_clock(monkeypatch)
    monkeypatch.setenv("MAGPIE_EXPERT_DISTRIBUTION_DUMP_WAIT_SECONDS", "300")
    path = tmp_path / "expert_distribution_recorder_current_0.pt"
    monkeypatch.setattr(expert, "_get_server_info", lambda _: _server_info(tp_size=2))
    monkeypatch.setattr(
        expert, "_request", lambda url, action: path.touch() if action == "dump" else None
    )
    expert.capture("start", "http://localhost:30000", tmp_path, tmp_path)
    result = expert.capture("finish", "http://localhost:30000", tmp_path, tmp_path)
    assert ticks[0] == 300
    assert result["status"] == "unavailable" and "TimeoutError" in result["reason"]
    assert path.exists() and not (tmp_path / expert.ARTIFACT).exists()


def test_configured_wait_extends_only_dump_rpc_timeout(monkeypatch):
    monkeypatch.setenv("MAGPIE_EXPERT_DISTRIBUTION_DUMP_WAIT_SECONDS", "300")
    calls = []

    def urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return nullcontext(SimpleNamespace(read=lambda limit: b"{}"))

    monkeypatch.setattr(expert, "urlopen", urlopen)
    for action in ("start", "stop", "dump"):
        expert._request("http://localhost:30000", action)
    assert [timeout for _, timeout in calls] == [30, 30, 330]


@pytest.mark.parametrize("value", ["", "not-a-number", "0", "-1", "nan", "inf", "3601"])
def test_invalid_configured_wait_stops_before_dump_and_reports_unavailable(
    monkeypatch, tmp_path, value
):
    monkeypatch.setenv("MAGPIE_EXPERT_DISTRIBUTION_DUMP_WAIT_SECONDS", value)
    calls = []
    monkeypatch.setattr(expert, "_request", lambda url, action: calls.append(action))
    expert.capture("start", "http://localhost:30000", tmp_path, tmp_path)
    result = expert.capture("finish", "http://localhost:30000", tmp_path, tmp_path)
    assert calls == ["start", "stop", "dump", "stop"]
    assert result["status"] == "unavailable"
    assert "MAGPIE_EXPERT_DISTRIBUTION_DUMP_WAIT_SECONDS" in result["reason"]
    assert "artifact_path" not in result


@pytest.mark.parametrize("value,expected", [("0.1", 0.1), ("3600", 3600)])
def test_configured_wait_accepts_positive_finite_boundaries(monkeypatch, value, expected):
    monkeypatch.setenv("MAGPIE_EXPERT_DISTRIBUTION_DUMP_WAIT_SECONDS", value)
    assert expert._dump_wait_seconds() == expected


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
        "native_legacy",
        "aggregation",
        "secondary_gatherer",
        "map_shape",
        "negative_count",
        "float_count",
        "zero_passes",
        "bool_passes",
        "negative_rank",
        "id_span_too_short",
        "reversed_ids",
        "zero_counts_with_passes",
    ],
)
def test_aggregate_export_fails_closed_on_invalid_plugin_payload(tmp_path, change):
    dump = _dump(passes=3)
    if change == "native_legacy":
        dump = _legacy_dump()
    elif change == "aggregation":
        dump["aggregation"] = "mean"
    elif change == "secondary_gatherer":
        dump["gatherer_key"] = "secondary"
    elif change == "map_shape":
        dump["last_physical_to_logical_map"] = [[0]]
    elif change == "negative_count":
        dump["global_physical_count"][0][0] = -1
    elif change == "float_count":
        dump["global_physical_count"][0][0] = 1.2
    elif change == "zero_passes":
        dump = _dump(passes=0)
    elif change == "bool_passes":
        dump["forward_pass_count"] = True
    elif change == "negative_rank":
        dump["rank"] = -1
    elif change == "id_span_too_short":
        dump["last_forward_pass_id"] = dump["first_forward_pass_id"] + 1
    elif change == "reversed_ids":
        dump["last_forward_pass_id"] = 1
    else:
        dump["global_physical_count"] = [[0, 0, 0]]
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
    assert json.loads((output / expert.STATE).read_text())["started"] is True
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
    result = expert.capture("start", "http://server", tmp_path, tmp_path)
    assert result["status"] == "unavailable"
    assert "plugin is unavailable" in result["reason"]
    assert calls == ["start", "stop", "dump", "stop"]
    assert len(list(tmp_path.glob("*.pt"))) == 2  # Retain failed-probe evidence.
    assert json.loads((tmp_path / expert.STATE).read_text())["started"] is False
    result = expert.capture("finish", "http://server", tmp_path, tmp_path)
    assert result["status"] == "unavailable" and "plugin is unavailable" in result["reason"]
    assert "artifact_path" not in result
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
