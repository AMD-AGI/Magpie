"""Diagnostic instrumentation remains separate from accepted launch inputs."""

import copy
import json

import pytest

from Magpie.modes.benchmark.agentx_profile_config import (
    profile_plan,
    profile_server_spec,
)
from Magpie.modes.benchmark.agentx_runtime import _check_profiled_replay_result
from Magpie.modes.benchmark.config import TraceLensConfig
from Magpie.modes.benchmark.tracelens import TraceLensAnalyzer

SETTINGS = {
    "num_steps": 12,
    "capture_timeout_seconds": 300.0,
    "flush_timeout_seconds": 1800.0,
}
CAPTURE_ID = "a" * 32


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
def test_diagnostic_launch_preserves_candidate_and_binds_step_limit(
    tmp_path, framework
):
    spec = {
        "framework": framework,
        "argv": ["python", "--model", "test/model"],
        "env": {},
    }
    original = copy.deepcopy(spec)
    derived = profile_server_spec(spec, SETTINGS, tmp_path, CAPTURE_ID)
    directory = str(tmp_path / "torch_trace" / CAPTURE_ID)
    assert spec == original
    assert derived["torch_profiler"] == {
        **SETTINGS,
        "capture_id": CAPTURE_ID,
        "trace_dir": directory,
    }
    if framework == "vllm":
        assert derived["argv"][:-2] == spec["argv"]
        assert derived["argv"][-2] == "--profiler-config"
        assert json.loads(derived["argv"][-1]) == {
            "profiler": "torch",
            "torch_profiler_dir": directory,
            "max_iterations": 12,
            "ignore_frontend": True,
            "torch_profiler_use_gzip": True,
        }
    else:
        assert derived["argv"] == spec["argv"]
        assert derived["env"]["SGLANG_TORCH_PROFILER_DIR"] == directory


@pytest.mark.parametrize(
    "flag",
    ["--profiler-config", "--profiler_config", "--profiler-config.max_iterations=1"],
)
def test_rejects_preexisting_profiler_configuration(tmp_path, flag):
    with pytest.raises(ValueError, match="owned by Magpie"):
        profile_server_spec(
            {"framework": "vllm", "argv": ["vllm", flag]},
            SETTINGS,
            tmp_path,
            CAPTURE_ID,
        )


@pytest.mark.parametrize("capture_id", ["../old", "", None, "A" * 32])
def test_rejects_invalid_capture_directory(tmp_path, capture_id):
    with pytest.raises(ValueError, match="unique capture id"):
        profile_server_spec({}, SETTINGS, tmp_path, capture_id)


@pytest.mark.parametrize("steps", [0, -1, True, 1.5, "20"])
def test_rejects_invalid_step_limit(tmp_path, steps):
    with pytest.raises(ValueError, match="positive integer"):
        profile_server_spec({}, {**SETTINGS, "num_steps": steps}, tmp_path, CAPTURE_ID)


@pytest.mark.parametrize(
    "timeout", [0, True, "20", float("nan"), float("inf"), 10**400]
)
def test_rejects_unbounded_capture_timeout(tmp_path, timeout):
    with pytest.raises(ValueError, match="positive and finite"):
        profile_server_spec(
            {}, {**SETTINGS, "capture_timeout_seconds": timeout}, tmp_path, CAPTURE_ID
        )


def test_rejects_unsupported_framework(tmp_path):
    with pytest.raises(ValueError, match="SGLang and vLLM"):
        profile_server_spec(
            {"framework": "atom", "argv": []}, SETTINGS, tmp_path, CAPTURE_ID
        )


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
def test_repeated_captures_write_to_fixed_active_directory(tmp_path, framework):
    derived = profile_server_spec(
        {"framework": framework, "argv": ["python", "--model", "fixture"]},
        {**SETTINGS, "num_profiles": 3, "interval_seconds": 200},
        tmp_path,
        CAPTURE_ID,
    )
    root = tmp_path / "torch_trace" / CAPTURE_ID
    assert derived["torch_profiler"]["trace_dir"] == str(root)
    assert derived["torch_profiler"]["num_profiles"] == 3
    if framework == "sglang":
        assert derived["env"]["SGLANG_TORCH_PROFILER_DIR"] == str(root / "active")
    else:
        options = json.loads(derived["argv"][-1])
        assert options["torch_profiler_dir"] == str(root / "active")
        assert options["max_iterations"] == 12


@pytest.mark.parametrize("count", [0, -1, 1.5, True, "3"])
def test_repeated_capture_count_must_be_positive_integer(tmp_path, count):
    with pytest.raises(ValueError, match="num_profiles must be a positive integer"):
        profile_server_spec(
            {}, {**SETTINGS, "num_profiles": count}, tmp_path, CAPTURE_ID
        )


@pytest.mark.parametrize(
    "interval", [-1, True, "200", float("nan"), float("inf"), 10**400]
)
def test_repeat_interval_rejects_invalid_or_infinite_wait(tmp_path, interval):
    with pytest.raises(
        ValueError, match="interval_seconds must be nonnegative and finite"
    ):
        profile_server_spec(
            {}, {**SETTINGS, "interval_seconds": interval}, tmp_path, CAPTURE_ID
        )


def test_tracelens_ignores_capture_manifest(tmp_path):
    (tmp_path / "capture.json").write_text('{"status":"complete"}')
    startup = tmp_path / "graph_capture_profile"
    startup.mkdir()
    (startup / "startup.trace.json").write_text('{"traceEvents":[]}')
    trace = tmp_path / "rank0.trace.json"
    trace.write_text('{"traceEvents":[]}')
    analyzer = TraceLensAnalyzer(TraceLensConfig(analysis_mode="pytorch"))
    assert analyzer._find_trace_files(tmp_path) == [trace]


@pytest.mark.parametrize(
    ("env", "interval", "maximum", "duration"),
    [
        ({"DURATION": "3600"}, 200, 18, 3600),
        ({"DURATION": "3600", "AIPERF_EXPERIMENTAL_FAST": "1"}, 200, 6, 1200),
        ({"DURATION": "600", "AIPERF_EXPERIMENTAL_FAST": "true"}, 200, 3, 600),
        ({"DURATION": "600.1"}, 200, 4, 600.1),
        ({"DURATION": "0.9"}, 0.3, 3, 0.9),
        ({"DURATION": "3600"}, 5000, 1, 3600),
        ({"DURATION": "3600"}, 0, None, 3600),
        ({}, 200, 18, 3600),
    ],
)
def test_capture_plan_uses_measurement_window_not_process_timeout(
    env, interval, maximum, duration
):
    settings = {**SETTINGS, "num_profiles": 100, "interval_seconds": interval}
    original = copy.deepcopy(settings)
    plan = profile_plan(settings, env)
    assert plan == {
        "requested_profiles": 100,
        "max_profiles": maximum,
        "planned_profiles": min(100, maximum) if maximum is not None else 100,
        "measurement_duration_seconds": duration,
        "start_seconds": 0.0,
    }
    assert settings == original


def test_plan_never_increases_a_smaller_requested_count():
    assert (
        profile_plan({"num_profiles": 3}, {"DURATION": "3600"})["planned_profiles"] == 3
    )


@pytest.mark.parametrize(
    ("duration", "start", "interval", "maximum"),
    [
        (3600, 1800, 200, 9),
        (3600, 3599, 200, 1),
        (0.9, 0.3, 0.3, 2),
        (0.7, 0.1, 0.2, 3),
        (1200, 500, 0, None),
    ],
)
def test_delayed_capture_plan_uses_exact_remaining_measurement_window(
    duration, start, interval, maximum
):
    plan = profile_plan(
        {"num_profiles": 100, "start_seconds": start, "interval_seconds": interval},
        {"DURATION": str(duration)},
    )
    assert plan["start_seconds"] == start
    assert plan["max_profiles"] == maximum


@pytest.mark.parametrize("start", [1200, 1800, 3600])
def test_fast_mode_rejects_first_capture_outside_its_window(start):
    with pytest.raises(ValueError, match="less than the measurement duration"):
        profile_plan({"start_seconds": start}, {"AIPERF_EXPERIMENTAL_FAST": "1"})


@pytest.mark.parametrize("start", [-1, True, "1", float("nan"), float("inf"), 10**400])
def test_invalid_start_seconds_cannot_enter_plan_or_launch(tmp_path, start):
    settings = {**SETTINGS, "start_seconds": start}
    with pytest.raises(ValueError, match="start_seconds"):
        profile_plan(settings, {})
    with pytest.raises(ValueError, match="start_seconds"):
        profile_server_spec({}, settings, tmp_path, CAPTURE_ID)


@pytest.mark.parametrize("field", ["roofline_annotations", "detailed_annotations"])
def test_sglang_enhanced_launch_binds_detected_capabilities(tmp_path, field):
    capabilities = {
        "annotation_field": field,
        "shape_discovery": True,
        "graph_capture": True,
        "graph_shape_discovery": True,
    }
    spec = {"framework": "sglang", "argv": ["python", "--model", "fixture"]}
    derived = profile_server_spec(
        spec,
        {**SETTINGS, "detailed_annotations": True},
        tmp_path,
        CAPTURE_ID,
        capabilities=capabilities,
    )
    assert "--enable-profile-cuda-graph" in derived["argv"]
    assert "--enable-shape-discovery-for-cuda-graph-profile" in derived["argv"]
    assert derived["env"]["SGLANG_PROFILE_RECORD_SHAPES"] == "True"
    assert derived["torch_profiler"]["capabilities"] == capabilities
    eager = profile_server_spec(
        spec,
        {**SETTINGS, "detailed_annotations": True},
        tmp_path,
        CAPTURE_ID,
        capabilities=capabilities,
        launch_overrides={"version": 1, "append_args": ["--disable-cuda-graph"]},
    )
    assert "--enable-profile-cuda-graph" not in eager["argv"]


@pytest.mark.parametrize(
    "field", ["capture_torch_profiler_dir", "capture_torch_profiler"]
)
def test_vllm_enhanced_launch_supports_new_and_old_capture_fields(tmp_path, field):
    spec = {"framework": "vllm", "argv": ["python", "--model", "fixture"]}
    capabilities = {
        "annotation_field": "detailed_trace_annotation",
        "capture_field": field,
    }
    derived = profile_server_spec(
        spec,
        {**SETTINGS, "detailed_annotations": True},
        tmp_path,
        CAPTURE_ID,
        capabilities=capabilities,
    )
    options = json.loads(derived["argv"][-1])
    assert options["detailed_trace_annotation"] is True
    assert options["torch_profiler_record_shapes"] is True
    assert options[field] == (
        str(tmp_path / "torch_trace" / CAPTURE_ID / "capture_traces")
        if field.endswith("_dir")
        else True
    )
    eager = profile_server_spec(
        spec,
        {**SETTINGS, "detailed_annotations": True},
        tmp_path,
        CAPTURE_ID,
        capabilities=capabilities,
        launch_overrides={"version": 1, "append_args": ["--enforce-eager"]},
    )
    assert field not in json.loads(eager["argv"][-1])


@pytest.mark.parametrize(
    "flags,eager",
    [
        (["--enforce-eager", "--no-enforce-eager"], False),
        (["--no-enforce-eager", "--enforce-eager"], True),
    ],
)
def test_vllm_annotation_graph_options_follow_final_eager_setting(
    tmp_path, flags, eager
):
    derived = profile_server_spec(
        {"framework": "vllm", "argv": ["python", "--model", "fixture", *flags]},
        {**SETTINGS, "detailed_annotations": True},
        tmp_path,
        CAPTURE_ID,
        capabilities={
            "annotation_field": "detailed_trace_annotation",
            "capture_field": "capture_torch_profiler_dir",
        },
    )
    assert (
        "capture_torch_profiler_dir" in json.loads(derived["argv"][-1])
    ) is not eager


def test_plan_handles_extreme_finite_duration_interval_ratio():
    plan = profile_plan(
        {"num_profiles": 100, "interval_seconds": 5e-324}, {"DURATION": "1e308"}
    )
    assert plan["max_profiles"] == 2 * 10**631
    assert plan["planned_profiles"] == 100


@pytest.mark.parametrize("duration", ["0", "-1", "inf", "nan", "invalid", True, None])
def test_plan_rejects_invalid_effective_duration(duration):
    with pytest.raises(ValueError, match="replay duration"):
        profile_plan(SETTINGS, {"DURATION": duration})


@pytest.mark.parametrize("interval", [-1, True, "200", float("inf"), 10**400])
def test_plan_rejects_invalid_interval(interval):
    with pytest.raises(ValueError, match="finite interval"):
        profile_plan({**SETTINGS, "interval_seconds": interval}, {})


@pytest.mark.parametrize("count", [0, True, "3"])
def test_plan_rejects_invalid_count(count):
    with pytest.raises(ValueError, match="positive integer"):
        profile_plan({**SETTINGS, "num_profiles": count}, {})


@pytest.mark.parametrize(
    "payload", [None, "not json", "[]", "{}", '{"was_cancelled":"false"}']
)
def test_zero_exit_requires_an_unambiguous_replay_result(tmp_path, payload):
    artifact = tmp_path / "aiperf_artifacts" / "profile_export_aiperf.json"
    if payload is not None:
        artifact.parent.mkdir()
        artifact.write_text(payload)
    with pytest.raises(RuntimeError, match="cannot verify replay completion"):
        _check_profiled_replay_result(tmp_path, 1)


def test_previous_replay_result_cannot_prove_current_completion(tmp_path):
    artifact = tmp_path / "aiperf_artifacts" / "profile_export_aiperf.json"
    artifact.parent.mkdir()
    artifact.write_text('{"was_cancelled":false}')
    with pytest.raises(RuntimeError, match="earlier client"):
        _check_profiled_replay_result(tmp_path, artifact.stat().st_mtime_ns + 1)
