"""Custom AgentX uses declared native launchers and verified model metadata."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from Magpie.modes.benchmark.agentx import _recipe_fingerprint, resolve_agentx_recipe
from Magpie.modes.benchmark.agentx_custom import native_context_length
from Magpie.modes.benchmark.config import BenchmarkConfig


@pytest.fixture
def custom_config(tmp_path):
    root = tmp_path / "InferenceX"
    (root / "configs").mkdir(parents=True)
    (root / "configs/amd-master.yaml").write_text("{}\n")
    manifest = {"version": 1, "recipes": {}, "generic": {}}
    for framework in ("sglang", "vllm"):
        script = f"single_node/agentic/generic_{framework}.sh"
        target = root / "benchmarks" / script
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("#!/bin/bash\n")
        for runner in ("mi300x", "mi325x", "mi355x"):
            manifest["generic"][f"custom-{framework}-{runner}"] = {
                "framework": framework,
                "runner_type": runner,
                "benchmark_script": script,
                "launch_overrides_version": 1,
                "max_gpus": 8,
            }
    (root / "configs/agentx-launchers.json").write_text(json.dumps(manifest))
    model = tmp_path / "model with spaces"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"max_position_embeddings": 40960}))
    data = {
        "model": "Qwen/Qwen3-0.6B",
        "framework": "sglang",
        "precision": "bf16",
        "docker_image": "sglang:pinned",
        "agentx": {"enabled": True, "launch_overrides": {"version": 1}},
        "envs": {"MODEL_PATH": str(model), "TP": 1, "EP_SIZE": 1, "CONC": 64},
    }
    return root, data


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
@pytest.mark.parametrize("runner", ["mi300x", "mi325x", "mi355x"])
def test_custom_recipe_resolves_and_roundtrips_without_registered_matrix(
    custom_config, framework, runner
):
    root, data = custom_config
    data["framework"] = framework
    config = BenchmarkConfig.from_dict(data)
    spec = resolve_agentx_recipe(config, str(root), runner)
    assert spec.recipe == f"custom-{framework}-{runner}"
    assert config.benchmark_script == f"single_node/agentic/generic_{framework}.sh"
    assert spec.entry["custom"] is True
    assert spec.entry["native-context-length"] == 40960
    assert spec.entry["max-model-len"] == 40960
    assert config.envs["RECIPE_FINGERPRINT"] == spec.entry["recipe-fingerprint"]
    fingerprint_entry = {
        k: v
        for k, v in spec.entry.items()
        if k not in {"benchmark-script", "launch-overrides-version"}
    }
    assert _recipe_fingerprint(fingerprint_entry) == spec.entry["recipe-fingerprint"]
    assert (
        config.envs["WEKA_LOADER_OVERRIDE"] == "semianalysis_cc_traces_weka_062126_256k"
    )
    snapshot = config.to_dict()
    reloaded = BenchmarkConfig.from_dict(deepcopy(snapshot))
    assert resolve_agentx_recipe(reloaded, str(root), runner).entry == spec.entry
    assert reloaded.to_dict() == snapshot


def test_custom_context_and_metadata_are_fingerprinted(custom_config):
    root, data = custom_config
    data["envs"]["MAX_MODEL_LEN"] = 8192
    config = BenchmarkConfig.from_dict(data)
    entry = resolve_agentx_recipe(config, str(root), "mi355x").entry
    assert entry["max-model-len"] == 8192
    assert config.envs["AGENTX_MAX_MODEL_LEN"] == 8192
    path = Path(data["envs"]["MODEL_PATH"]) / "config.json"
    assert entry["model-config-sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text('{"max_position_embeddings":32768}')
    changed = resolve_agentx_recipe(config, str(root), "mi355x").entry
    assert changed["recipe-fingerprint"] != entry["recipe-fingerprint"]


@pytest.mark.parametrize(
    "env",
    [
        {"MAX_MODEL_LEN": 50000},
        {"MAX_MODEL_LEN": 0},
        {"TP": 16},
        {"TP": 4, "EP_SIZE": 3},
        {"PP_SIZE": 2},
        {"NNODES": 2},
        {"DP_SIZE": 2},
        {"DISAGG": True},
        {"DP_ATTENTION": True},
        {"KV_OFFLOADING": "dram"},
        {"SPEC_DECODING": "mtp"},
        {"EP": 2},
    ],
)
def test_unsupported_custom_point_fails_closed(custom_config, env):
    root, data = custom_config
    data["envs"].update(env)
    with pytest.raises(ValueError):
        resolve_agentx_recipe(BenchmarkConfig.from_dict(data), str(root), "mi355x")


def test_custom_requires_explicit_image(custom_config):
    root, data = custom_config
    data.pop("docker_image")
    with pytest.raises(ValueError, match="explicit docker_image"):
        resolve_agentx_recipe(BenchmarkConfig.from_dict(data), str(root), "mi355x")


def test_custom_vllm_does_not_infer_partial_ep(custom_config):
    root, data = custom_config
    data["framework"] = "vllm"
    data["envs"].update(TP=4, EP_SIZE=2)
    with pytest.raises(ValueError, match="EP=1 or EP=TP"):
        resolve_agentx_recipe(BenchmarkConfig.from_dict(data), str(root), "mi355x")


@pytest.mark.parametrize(
    "metadata, expected",
    [
        ({"max_position_embeddings": 40960}, 40960),
        ({"text_config": {"max_position_embeddings": 32768}}, 32768),
        ({"max_position_embeddings": 40960, "seq_length": 8192}, 8192),
    ],
)
def test_context_is_derived_without_model_code(metadata, expected):
    assert native_context_length(metadata) == expected


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"max_position_embeddings": True},
        {"max_position_embeddings": -1},
        {"text_config": []},
    ],
)
def test_unknown_native_context_is_not_guessed(metadata):
    with pytest.raises(ValueError):
        native_context_length(metadata)


def test_hf_metadata_request_uses_only_config_and_does_not_persist_auth(
    custom_config, monkeypatch
):
    import io

    from Magpie.modes.benchmark import agentx_custom

    root, data = custom_config
    data["envs"].pop("MODEL_PATH")
    requests = []

    def fetch(request, timeout):
        requests.append((request, timeout))
        return io.BytesIO(b'{"max_position_embeddings":40960}')

    monkeypatch.setattr(agentx_custom, "urlopen", fetch)
    monkeypatch.setenv("HF_TOKEN", "secret-test-value")
    entry = resolve_agentx_recipe(
        BenchmarkConfig.from_dict(data), str(root), "mi355x"
    ).entry
    assert (
        requests[0][0].full_url
        == "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json"
    )
    assert requests[0][0].get_header("Authorization") == "Bearer secret-test-value"
    assert "secret-test-value" not in json.dumps(entry)


@pytest.mark.parametrize(
    "changes, valid",
    [
        ({}, True),
        ({"max_model_len": 50000}, False),
        ({"model_config_sha256": "bad"}, False),
        ({"native_context_length": None}, False),
    ],
)
def test_custom_result_preserves_and_checks_model_metadata(tmp_path, changes, valid):
    from Magpie.modes.benchmark.result import ResultParser

    aggregate = {
        "scenario_type": "agentic-coding",
        "recipe_fingerprint": "a" * 64,
        "num_requests_total": 2,
        "num_requests_successful": 2,
        "request_accounting": {
            "records_total": 2,
            "records_profiled": 2,
            "records_error_dropped": 0,
            "records_warmup_dropped": 0,
        },
        "request_metrics": {
            "throughput": {"total": {"tokens_per_second": 20}, "duration_seconds": 3600}
        },
        "custom_recipe": True,
        "native_context_length": 40960,
        "max_model_len": 8192,
        "model_config_sha256": "b" * 64,
        **changes,
    }
    path = tmp_path / "aggregate.json"
    path.write_text(json.dumps(aggregate))
    result = ResultParser.parse_inferencex_result(path, scenario="agentx")
    assert result.benchmark_valid is valid
    assert result.publishable is valid
    assert result.agentx_metrics["recipe"]["custom_recipe"] is True
    assert (
        result.agentx_metrics["recipe"]["max_model_len"] == aggregate["max_model_len"]
    )
