"""Diagnostic instrumentation remains separate from accepted launch inputs."""

import copy
import json

import pytest

from Magpie.modes.benchmark.agentx_profile_config import profile_server_spec
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
