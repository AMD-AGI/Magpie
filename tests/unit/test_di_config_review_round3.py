"""DI precision/model declarations must agree with actual serving/client inputs."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from Magpie.di.native_config import normalize_native_di
from Magpie.di.native_ray import NativeRayDISession
from Magpie.modes.benchmark import BenchmarkConfig


def configuration(precision="bf16", extra="--context-length 8192"):
    return {
        "framework": "sglang",
        "precision": precision,
        "model": "/prepared/model",
        "run_mode": "ray",
        "envs": {"TP": 1, "EXTRA_SGLANG_ARGS": extra},
        "ray_config": {"env_vars": {}},
        "disaggregation": {"prefill": {}, "decode": {"envs": {}}},
    }


@pytest.mark.parametrize("precision,dtype", [("fp16", "float16"), ("bf16", "bfloat16")])
def test_precision_controls_dtype_for_both_roles_without_mutating_caller(
    precision, dtype
):
    raw = configuration(precision)
    before = deepcopy(raw)
    normalized = normalize_native_di(raw)
    for instance in normalized["instances"]:
        arguments = instance["server_args"]
        assert arguments[arguments.index("--dtype") + 1] == dtype
    assert raw == before
    assert BenchmarkConfig.from_dict(raw).to_dict()["precision"] == precision


@pytest.mark.parametrize(
    "precision,extra,expected",
    [
        ("fp16", "--dtype half", ["--dtype", "float16"]),
        ("fp16", "--dtype=float16", ["--dtype=float16"]),
        ("bf16", "--dtype bfloat16", ["--dtype", "bfloat16"]),
    ],
)
def test_matching_dtype_aliases_are_canonical_for_readiness(precision, extra, expected):
    assert all(
        instance["server_args"] == expected
        for instance in normalize_native_di(configuration(precision, extra))[
            "instances"
        ]
    )


@pytest.mark.parametrize(
    "precision,extra",
    [
        ("fp16", "--dtype bfloat16"),
        ("bf16", "--dtype=half"),
        ("bf16", "--dtype auto"),
        ("fp16", "--dtype float32"),
        ("bf16", "--dtype"),
        ("bf16", "--dtype="),
    ],
)
def test_conflicting_or_missing_dtype_rejected_before_session_directory(
    tmp_path, precision, extra
):
    with pytest.raises(ValueError, match="dtype"):
        NativeRayDISession(configuration(precision, extra), tmp_path / "must-not-exist")
    assert not list(tmp_path.iterdir())


def test_role_dtype_override_cannot_conflict_with_shared_precision():
    raw = configuration("bf16")
    raw["disaggregation"]["decode"]["envs"]["EXTRA_SGLANG_ARGS"] = "--dtype half"
    with pytest.raises(ValueError, match="precision conflicts"):
        normalize_native_di(raw)


@pytest.mark.parametrize(
    "extra", ["", "--dtype bfloat16", "--quantization awq", "--quantization=fp4"]
)
def test_fp8_requires_explicit_matching_quantization_without_auto_conversion(extra):
    with pytest.raises(
        ValueError, match="precision=fp8 requires explicit --quantization fp8"
    ):
        normalize_native_di(configuration("fp8", extra))


@pytest.mark.parametrize(
    "extra", ["--quantization fp8", "--dtype bfloat16 --quantization=fp8"]
)
def test_explicit_fp8_arguments_are_preserved(extra):
    assert all(
        instance["server_args"] == extra.split()
        for instance in normalize_native_di(configuration("fp8", extra))["instances"]
    )


def test_default_fp8_also_requires_explicit_quantization():
    raw = configuration()
    raw.pop("precision")
    with pytest.raises(ValueError, match="precision=fp8 requires explicit"):
        BenchmarkConfig.from_dict(raw)


def test_role_argument_replacement_must_retain_fp8_quantization_flag():
    raw = configuration("fp8", "--quantization fp8")
    raw["disaggregation"]["decode"]["envs"]["EXTRA_SGLANG_ARGS"] = (
        "--context-length 4096"
    )
    with pytest.raises(ValueError, match="every role"):
        normalize_native_di(raw)


@pytest.mark.parametrize("precision", ["fp4", "int8", None])
def test_unsupported_di_precision_fails_early(precision):
    with pytest.raises(ValueError, match="DI precision must be"):
        normalize_native_di(configuration(precision))


@pytest.mark.parametrize("placement", ["common", "ray"])
@pytest.mark.parametrize("name", ["MODEL", "model"])
def test_mismatched_model_override_cannot_change_client_identity(
    tmp_path, placement, name
):
    raw = configuration()
    target = raw["envs"] if placement == "common" else raw["ray_config"]["env_vars"]
    target[name] = "other/model"
    with pytest.raises(ValueError, match="MODEL must match benchmark.model"):
        NativeRayDISession(raw, tmp_path / "must-not-exist")
    with pytest.raises(ValueError, match="MODEL must match benchmark.model"):
        BenchmarkConfig.from_dict(raw)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("placement", ["common", "ray"])
def test_matching_model_alias_preserves_one_canonical_identity(placement):
    raw = configuration()
    target = raw["envs"] if placement == "common" else raw["ray_config"]["env_vars"]
    target["model"] = raw["model"]
    spec = normalize_native_di(raw)
    assert spec["model"] == raw["model"]
    assert all("MODEL" not in instance["serving_env"] for instance in spec["instances"])
    assert BenchmarkConfig.from_dict(raw).get_env_vars()["MODEL"] == raw["model"]


def test_conflicting_case_model_aliases_are_not_silently_accepted():
    raw = configuration()
    raw["envs"].update(MODEL="different", model=raw["model"])
    with pytest.raises(ValueError, match="MODEL must match"):
        normalize_native_di(raw)


@pytest.mark.parametrize("placement", ["common", "ray"])
def test_mismatched_precision_alias_rejected_and_matching_alias_accepted(placement):
    raw = configuration("bf16")
    env = raw["envs"] if placement == "common" else raw["ray_config"]["env_vars"]
    env["precision"] = "fp16"
    with pytest.raises(ValueError, match="PRECISION must match"):
        normalize_native_di(raw)
    env["precision"] = "BF16"
    assert normalize_native_di(raw)["instances"][0]["server_args"][-1] == "bfloat16"


@pytest.mark.parametrize("key", ["MODEL", "PRECISION"])
def test_roles_cannot_override_client_model_or_precision(key):
    raw = configuration()
    raw["disaggregation"]["decode"]["envs"][key] = "other"
    with pytest.raises(ValueError, match="client settings"):
        normalize_native_di(raw)


def test_public_fp8_example_explicitly_declares_the_serving_quantization():
    example = (
        Path(__file__).resolve().parents[2]
        / "examples/benchmarks/benchmark_sglang_dsr1_di_ray.yaml"
    )
    spec = normalize_native_di(yaml.safe_load(example.read_text()))
    for instance in spec["instances"]:
        args = instance["server_args"]
        assert args[args.index("--quantization") + 1] == "fp8"
        assert args[args.index("--dtype") + 1] == "bfloat16"


def test_non_di_model_and_precision_env_overrides_keep_existing_behavior():
    raw = configuration("fp16")
    raw.pop("disaggregation")
    raw["envs"].update(MODEL="ordinary/override", PRECISION="other-label")
    env = BenchmarkConfig.from_dict(raw).get_env_vars()
    assert env["MODEL"] == "ordinary/override" and env["PRECISION"] == "other-label"
