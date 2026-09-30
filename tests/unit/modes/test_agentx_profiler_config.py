"""Public configuration contracts for AgentX diagnostic trace capture."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from Magpie.modes.benchmark.config import (
    BenchmarkConfig,
    ProfilerConfig,
    TorchProfilerConfig,
)


def _config(**overrides):
    return BenchmarkConfig.from_dict(
        {"framework": "sglang", "model": "test/model", "agentx": "enable", **overrides}
    )


def test_agentx_profiler_settings_survive_config_round_trip():
    settings = {
        "enabled": True,
        "num_steps": 12,
        "capture_timeout_seconds": 45.5,
        "flush_timeout_seconds": 600,
        "num_profiles": 3,
        "interval_seconds": 0,
    }
    config = _config(profiler={"torch_profiler": settings})
    restored = BenchmarkConfig.from_dict(config.to_dict())
    assert restored.profiler.torch_profiler.to_dict() == settings


def test_agentx_profiling_is_explicit_and_ordinary_defaults_are_preserved():
    assert _config().profiler.torch_profiler.enabled is False
    ordinary = BenchmarkConfig.from_dict({"framework": "sglang", "model": "test/model"})
    assert ordinary.profiler.torch_profiler.enabled is True
    assert TorchProfilerConfig().to_dict() == {
        "enabled": True,
        "num_steps": 20,
        "capture_timeout_seconds": 300.0,
        "flush_timeout_seconds": 1800.0,
        "num_profiles": 1,
        "interval_seconds": 200.0,
    }


@pytest.mark.parametrize("direct", [True, False])
@pytest.mark.parametrize(
    "sections",
    [
        {},
        {"profiler": None},
        {"profiler": {}},
        {"profiler": {"torch_profiler": {}}},
        {"profiler": {"torch_profiler": {"num_steps": 9}}},
        {"profiler": {"torch_profiler": {"num_profiles": 3}}},
        {"profiler": {"torch_profiler": {"interval_seconds": 0}}},
        {"profiler": {"gpu_monitor": {"interval_sec": 1.5}}},
    ],
)
def test_agentx_partial_profiler_config_never_enables_diagnostics(direct, sections):
    values = {
        "framework": "sglang",
        "model": "test/model",
        "agentx": True,
        **sections,
    }
    before = deepcopy(values)
    config = BenchmarkConfig(**values) if direct else BenchmarkConfig.from_dict(values)
    assert config.profiler.torch_profiler.enabled is False
    assert config.profiler.gpu_monitor.enabled is False
    assert values == before
    if (sections.get("profiler") or {}).get("torch_profiler", {}).get("num_steps"):
        assert config.profiler.torch_profiler.num_steps == 9


@pytest.mark.parametrize("direct", [True, False])
@pytest.mark.parametrize("enabled", [True, False])
def test_agentx_explicit_profiler_dict_retains_its_switches(direct, enabled):
    values = {
        "framework": "sglang",
        "model": "test/model",
        "agentx": True,
        "profiler": {
            "torch_profiler": {"enabled": enabled, "num_steps": 9},
            "gpu_monitor": {"enabled": enabled},
        },
    }
    config = BenchmarkConfig(**values) if direct else BenchmarkConfig.from_dict(values)
    assert config.profiler.torch_profiler.enabled is enabled
    assert config.profiler.torch_profiler.num_steps == 9
    assert config.profiler.gpu_monitor.enabled is enabled


@pytest.mark.parametrize("enabled", [True, False])
def test_direct_agentx_preserves_explicit_profiler_object(enabled):
    profiler = ProfilerConfig(torch_profiler=TorchProfilerConfig(enabled=enabled))
    config = BenchmarkConfig(
        framework="sglang", model="test/model", agentx=True, profiler=profiler
    )
    assert config.profiler is profiler
    assert config.profiler.torch_profiler.enabled is enabled
    assert config.profiler.gpu_monitor.enabled is True


@pytest.mark.parametrize("direct", [True, False])
@pytest.mark.parametrize("profiler", [None, {}, {"torch_profiler": {"num_steps": 9}}])
def test_ordinary_partial_profiler_config_keeps_enabled_defaults(direct, profiler):
    values = {"framework": "sglang", "model": "test/model", "profiler": profiler}
    config = BenchmarkConfig(**values) if direct else BenchmarkConfig.from_dict(values)
    assert config.profiler.torch_profiler.enabled is True
    assert config.profiler.gpu_monitor.enabled is True


@pytest.mark.parametrize("field", ["num_steps", "num_profiles"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 2.5, "20", None])
def test_torch_profiler_rejects_invalid_capture_counts(field, value):
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        TorchProfilerConfig.from_dict({field: value})


@pytest.mark.parametrize("field", ["capture_timeout_seconds", "flush_timeout_seconds"])
@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        0,
        -1,
        float("nan"),
        float("inf"),
        -float("inf"),
        "30",
        None,
        10**400,
    ],
)
def test_torch_profiler_rejects_invalid_timeout(field, value):
    with pytest.raises(ValueError, match=f"{field} must be a finite positive number"):
        TorchProfilerConfig.from_dict({field: value})


@pytest.mark.parametrize("value", [0, 0.25, 200])
def test_torch_profiler_allows_zero_and_fractional_intervals(value):
    config = TorchProfilerConfig(num_profiles=3, interval_seconds=value)
    assert config.interval_seconds == value
    assert TorchProfilerConfig.from_dict(config.to_dict()).interval_seconds == value


@pytest.mark.parametrize(
    "value",
    [True, False, -1, float("nan"), float("inf"), -float("inf"), "200", None, 10**400],
)
def test_torch_profiler_rejects_invalid_interval(value):
    with pytest.raises(
        ValueError, match="interval_seconds must be a finite non-negative number"
    ):
        TorchProfilerConfig.from_dict({"interval_seconds": value})


@pytest.mark.parametrize("mode", ["pytorch", "classic"])
def test_agentx_allows_tracelens_postprocess_after_torch_capture(mode):
    config = _config(
        profiler={
            "torch_profiler": {"enabled": True},
            "tracelens": {"enabled": True, "analysis_mode": mode},
        }
    )
    assert config.profiler.tracelens.analysis_mode == "pytorch"


def test_agentx_tracelens_requires_explicit_torch_capture():
    with pytest.raises(
        ValueError, match="requires profiler.torch_profiler.enabled=true"
    ):
        _config(profiler={"tracelens": {"enabled": True, "analysis_mode": "pytorch"}})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"profiler": {"tracelens": {"enabled": True}}}, "analysis_mode=inference"),
        ({"profiler": {"system_profiler": {"enabled": True}}}, "system_profiler"),
        ({"gap_analysis": {"enabled": True}}, "gap_analysis"),
    ],
)
def test_agentx_rejects_unsupported_analysis_modes(overrides, message):
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


def test_agentx_profiler_example_keeps_canonical_example_unprofiled():
    examples = Path(__file__).resolve().parents[3] / "examples" / "benchmarks"
    name = "benchmark_sglang_deepseek_v4_pro_fp4_mi355x_agentx"
    ordinary = BenchmarkConfig.from_dict(
        yaml.safe_load((examples / f"{name}.yaml").read_text())["benchmark"]
    )
    diagnostic = BenchmarkConfig.from_dict(
        yaml.safe_load((examples / f"{name}_profile.yaml").read_text())["benchmark"]
    )
    assert ordinary.profiler.torch_profiler.enabled is False
    assert diagnostic.profiler.torch_profiler.enabled is True
    assert diagnostic.profiler.tracelens.analysis_mode == "pytorch"
    assert diagnostic.model == ordinary.model
    assert diagnostic.envs["CONC"] == ordinary.envs["CONC"]
