"""Inline DI credentials must never reach durable workspace configuration."""

from copy import deepcopy

import pytest

from Magpie.di.native_config import normalize_native_di
from Magpie.di.native_ray import NativeRayDISession
from Magpie.modes.benchmark import BenchmarkConfig


def configuration():
    return {
        "framework": "sglang",
        "precision": "bf16",
        "run_mode": "ray",
        "model": "/prepared/model",
        "envs": {"TP": 1, "EP": 1},
        "ray_config": {"cluster_address": "auto", "env_vars": {}},
        "disaggregation": {"prefill": {}, "decode": {"envs": {}}},
    }


@pytest.mark.parametrize(
    "key", ["HF_TOKEN", "hf_token", "HUGGING_FACE_HUB_TOKEN", "hugging_face_hub_token"]
)
@pytest.mark.parametrize("placement", ["common", "role", "ray"])
@pytest.mark.parametrize("entrypoint", ["session", "yaml-session", "benchmark-config"])
def test_inline_credentials_fail_before_any_di_workspace_or_journal(
    tmp_path, key, placement, entrypoint
):
    raw = configuration()
    env = {
        "common": raw["envs"],
        "role": raw["disaggregation"]["decode"]["envs"],
        "ray": raw["ray_config"]["env_vars"],
    }[placement]
    secret = "dummy-review-credential-never-echo"
    env[key] = secret
    original = deepcopy(raw)
    with pytest.raises(
        ValueError, match="use worker-side environment credentials"
    ) as error:
        if entrypoint == "benchmark-config":
            BenchmarkConfig.from_dict(raw)
        else:
            supplied = {"benchmark": raw} if entrypoint == "yaml-session" else raw
            NativeRayDISession(supplied, tmp_path / "must-not-create")
    assert secret not in str(error.value)
    assert raw == original
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("key", ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"])
def test_empty_inline_credential_cannot_shadow_worker_authentication(key):
    raw = configuration()
    raw["envs"][key] = ""
    with pytest.raises(ValueError, match="Inline model credentials"):
        normalize_native_di(raw)


def test_existing_worker_environment_credentials_remain_outside_di_journal(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HF_TOKEN", "dummy-runtime-token")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "dummy-runtime-legacy-token")
    session = NativeRayDISession(configuration(), tmp_path / "session")
    text = session.path.read_text()
    assert "dummy-runtime-token" not in text
    assert "dummy-runtime-legacy-token" not in text
    assert session.state["status"] == "created"


@pytest.mark.parametrize("placement", ["common", "ray"])
def test_non_di_config_retains_existing_inline_credential_behavior(placement):
    raw = configuration()
    raw.pop("disaggregation")
    target = raw["envs"] if placement == "common" else raw["ray_config"]["env_vars"]
    target["HF_TOKEN"] = "dummy-ordinary-token"
    config = BenchmarkConfig.from_dict(raw)
    saved = config.to_dict()
    target = saved["envs"] if placement == "common" else saved["ray_config"]["env_vars"]
    assert target["HF_TOKEN"] == "dummy-ordinary-token"
