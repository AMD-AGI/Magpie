"""Candidate execution evidence is produced at Magpie's server boundary."""

import hashlib
import json
import os
import sys

import pytest

from Magpie.modes.benchmark.agentx_launch import (
    digest,
    prepare_server_launch,
    read_launch_evidence,
)
from Magpie.modes.benchmark.config import BenchmarkConfig


def _config(spec, overrides=None):
    config = BenchmarkConfig.from_dict(
        {
            "framework": "sglang",
            "model": "test/model",
            "precision": "fp16",
            "agentx": True,
        }
    )
    config.agentx.resolved = {"server-launch-spec": spec}
    if overrides is not None:
        from Magpie.modes.benchmark.agentx_launch import validate_overrides

        config.agentx.launch_overrides = validate_overrides(overrides)
    return config


def test_server_preparation_applies_candidate_without_changing_client_environment(
    tmp_path,
):
    source = tmp_path / "kernel.py"
    source.write_text("candidate = 1\n")
    overrides = {
        "version": 1,
        "remove_args": ["--chunked-prefill-size"],
        "append_args": [
            "--chunked-prefill-size",
            "2048",
            "--json-setting",
            '{"a": "x y"}',
        ],
        "env": {"SGLANG_OPT": "$(must-stay-literal)"},
        "unset_env": ["OLD_OPT"],
        "source_files": {str(source): hashlib.sha256(source.read_bytes()).hexdigest()},
    }
    spec = {
        "argv": [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            "test/model",
            "--chunked-prefill-size",
            "1024",
        ]
    }
    environment = {
        "PATH": os.environ["PATH"],
        "OLD_OPT": "old",
        "HF_TOKEN": "redacted-test-value",
    }
    argv, server_env, evidence = prepare_server_launch(
        spec["argv"], environment, overrides, "sglang", tmp_path, server_spec=spec
    )
    assert argv[-4:] == [
        "--chunked-prefill-size",
        "2048",
        "--json-setting",
        '{"a": "x y"}',
    ]
    assert server_env["SGLANG_OPT"] == "$(must-stay-literal)"
    assert "OLD_OPT" not in server_env
    assert environment["OLD_OPT"] == "old"
    assert "SGLANG_OPT" not in environment
    assert "HF_TOKEN" not in evidence["runtime_environment"]
    assert read_launch_evidence(_config(spec, overrides), tmp_path) == evidence


def test_empty_candidate_still_attests_managed_server(tmp_path):
    spec = {"argv": [sys.executable, "-m", "sglang.launch_server"]}
    _, _, evidence = prepare_server_launch(
        spec["argv"], dict(os.environ), None, "sglang", tmp_path, server_spec=spec
    )
    assert evidence["owner"] == "magpie"
    assert evidence["server_spec_sha256"] == digest(spec)
    assert read_launch_evidence(_config(spec), tmp_path) == evidence


@pytest.mark.parametrize(
    "failure", ["changed-source", "reappeared-source", "missing-executable"]
)
def test_failed_server_preparation_removes_previous_receipt(tmp_path, failure):
    source = tmp_path / "source.py"
    source.write_text("changed")
    receipt = tmp_path / "agentx_server_launch.json"
    receipt.write_text('{"stale": true}')
    request = {"version": 1}
    if failure == "changed-source":
        request["source_files"] = {str(source): "0" * 64}
    elif failure == "reappeared-source":
        request["absent_source_files"] = [str(source)]
    else:
        request["executable"] = str(tmp_path / "missing-python")
    with pytest.raises(ValueError):
        prepare_server_launch(
            [sys.executable], dict(os.environ), request, "sglang", tmp_path
        )
    assert not receipt.exists()


@pytest.mark.parametrize("replacement", ["spec", "argv", "owner"])
def test_rehashed_receipt_cannot_change_candidate_identity(tmp_path, replacement):
    spec = {"argv": [sys.executable, "-m", "sglang.launch_server"]}
    _, _, evidence = prepare_server_launch(
        spec["argv"], dict(os.environ), None, "sglang", tmp_path, server_spec=spec
    )
    if replacement == "spec":
        evidence["server_spec_sha256"] = "0" * 64
    elif replacement == "argv":
        evidence["effective_argv"] += ["--extra-option", "changed"]
    else:
        evidence["owner"] = "upstream"
    evidence["evidence_sha256"] = digest(
        {k: v for k, v in evidence.items() if k != "evidence_sha256"}
    )
    (tmp_path / "agentx_server_launch.json").write_text(json.dumps(evidence))
    with pytest.raises(ValueError):
        read_launch_evidence(_config(spec), tmp_path)


@pytest.mark.parametrize(
    "option", ["--model-path=other", "--port", "--tensor-parallel-size=8"]
)
def test_managed_launch_rejects_protocol_changes(tmp_path, option):
    with pytest.raises(ValueError, match="model, topology, host or port"):
        prepare_server_launch(
            [sys.executable, "-m", "sglang.launch_server"],
            dict(os.environ),
            {"version": 1, "append_args": [option]},
            "sglang",
            tmp_path,
        )


@pytest.mark.parametrize(
    "change", ["acceptance", "token-budget", "drop", "duplicate", "not-object"]
)
def test_candidate_cannot_override_vllm_golden_acceptance(tmp_path, change):
    golden = {
        "method": "mtp",
        "num_speculative_tokens": 3,
        "rejection_sample_method": "synthetic",
        "synthetic_acceptance_length": 2.4,
    }
    argv = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--speculative-config",
        json.dumps(golden),
    ]
    spec = {
        "argv": argv,
        "framework": "vllm",
        "golden_acceptance_overrides": ["official"],
    }
    replacement = dict(golden)
    if change == "acceptance":
        replacement["synthetic_acceptance_length"] = 999
    if change == "token-budget":
        replacement["num_speculative_tokens"] = 20
    request = {
        "version": 1,
        "remove_args": ["--speculative-config"],
        "append_args": ["--speculative-config", json.dumps(replacement)],
    }
    if change == "drop":
        request["append_args"] = []
    elif change == "duplicate":
        request["remove_args"] = []
    elif change == "not-object":
        request["append_args"][-1] = "[]"
    with pytest.raises(ValueError, match="golden|speculative-config"):
        prepare_server_launch(
            argv, dict(os.environ), request, "vllm", tmp_path, server_spec=spec
        )
    assert not (tmp_path / "agentx_server_launch.json").exists()


def test_vllm_candidate_can_tune_other_speculative_settings(tmp_path):
    golden = {
        "method": "mtp",
        "num_speculative_tokens": 3,
        "rejection_sample_method": "synthetic",
        "synthetic_acceptance_length": 2.4,
    }
    argv = [sys.executable, "--speculative-config", json.dumps(golden)]
    request = {
        "version": 1,
        "remove_args": ["--speculative-config"],
        "append_args": [
            "--speculative-config",
            json.dumps({**golden, "enforce_eager": True}),
        ],
    }
    spec = {
        "argv": argv,
        "framework": "vllm",
        "golden_acceptance_overrides": ["official"],
    }
    actual, _, _ = prepare_server_launch(
        argv, dict(os.environ), request, "vllm", tmp_path, server_spec=spec
    )
    assert json.loads(actual[-1])["enforce_eager"] is True


def test_sglang_candidate_cannot_change_golden_curve_inputs(tmp_path):
    argv = [
        sys.executable,
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
    ]
    spec = {
        "argv": argv,
        "framework": "sglang",
        "golden_acceptance_overrides": ["official"],
    }
    with pytest.raises(ValueError, match="golden"):
        prepare_server_launch(
            argv,
            dict(os.environ),
            {
                "version": 1,
                "remove_args": ["--speculative-num-steps"],
                "append_args": ["--speculative-num-steps", "10"],
            },
            "sglang",
            tmp_path,
            server_spec=spec,
        )
