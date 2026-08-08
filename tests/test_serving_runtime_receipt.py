import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from Magpie.main import load_benchmark_config_with_sha256, run_benchmark
from Magpie.modes.benchmark.benchmarker import BenchmarkMode
from Magpie.modes.benchmark.config import BenchmarkConfig
from Magpie.modes.benchmark.result import BenchmarkResult
from Magpie.modes.benchmark.serving_runtime import (
    SERVING_RUNTIME_KEYS,
    SERVING_RUNTIME_SCHEMA,
    canonical_docker_argv_sha256,
    pending_serving_runtime_receipt,
    resolve_docker_image_id,
    sha256_bytes,
)


def _docker_command(container_name: str, image_id: str) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--entrypoint",
        "/bin/bash",
        image_id,
        "-c",
        "true",
    ]


def _tracelens_runtime(
    *,
    base_image: str,
    base_image_id: str,
    derived_image: str,
    derived_image_id: str,
) -> dict[str, object]:
    return {
        "enabled": True,
        "framework": "vllm",
        "runtime_schema": "magpie.tracelens-vllm-runtime/v1",
        "base_image": base_image,
        "base_image_id": base_image_id,
        "base_image_locator": base_image_id,
        "image": derived_image,
        "public_runtime_image": derived_image,
        "public_runtime_image_id": derived_image_id,
        "tracelens_source_commit": "1" * 40,
        "tracelens_source_tree": "2" * 40,
        "patch_version": "v19",
        "tracelens_patch_path": (
            "examples/custom_workflows/inference_analysis/vllm_patches/"
            "config_vllm_v0.19.0.patch"
        ),
        "tracelens_patch_sha256": "3" * 64,
        "dependency_wheel_manifest_sha256": "4" * 64,
        "public_runtime_validation": {
            "valid": True,
            "image_id": derived_image_id,
        },
    }


def test_cli_hashes_the_exact_yaml_bytes_it_parses(tmp_path):
    config_path = tmp_path / "benchmark.yaml"
    raw = (
        b"# byte identity matters\r\n"
        b"benchmark:\r\n"
        b"  framework: vllm\r\n"
        b"  model: example/model\r\n"
    )
    config_path.write_bytes(raw)

    config, digest = load_benchmark_config_with_sha256(config_path)

    assert config == {"framework": "vllm", "model": "example/model"}
    assert digest == sha256_bytes(raw)


def test_cli_passes_raw_config_digest_to_benchmark_mode(tmp_path, monkeypatch):
    config_path = tmp_path / "benchmark.yaml"
    raw = (
        b"benchmark:\n"
        b"  framework: vllm\n"
        b"  model: example/model\n"
        b"  run_mode: local\n"
    )
    config_path.write_bytes(raw)
    captured = {}

    class FakeBenchmarkMode:
        def __init__(self, *, config, output_dir, input_config_sha256):
            captured["config"] = config
            captured["output_dir"] = output_dir
            captured["input_config_sha256"] = input_config_sha256

        def run(self):
            return BenchmarkResult(success=True)

    monkeypatch.setattr(
        "Magpie.modes.benchmark.BenchmarkMode",
        FakeBenchmarkMode,
    )
    args = SimpleNamespace(
        trace_dir=None,
        benchmark_config=config_path,
        framework=None,
        model=None,
        run_mode=None,
        run_kind=None,
        output_dir=tmp_path / "results",
    )

    assert run_benchmark(args, {}) == 0
    assert captured["input_config_sha256"] == sha256_bytes(raw)
    assert captured["config"].model == "example/model"


def test_resolve_tag_uses_fixed_inspect_argv(monkeypatch):
    image_id = "sha256:" + "1" * 64
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, image_id + "\n", "")

    monkeypatch.setattr(
        "Magpie.modes.benchmark.serving_runtime.subprocess.run",
        fake_run,
    )

    resolved, errors = resolve_docker_image_id("example/vllm:fixed")

    assert resolved == image_id
    assert errors == ()
    assert calls == [
        (
            [
                "docker",
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                "--",
                "example/vllm:fixed",
            ],
            {
                "capture_output": True,
                "text": True,
                "timeout": 30,
                "check": False,
            },
        )
    ]


def test_raw_image_id_needs_no_mutable_lookup(monkeypatch):
    image_id = "sha256:" + "2" * 64
    monkeypatch.setattr(
        "Magpie.modes.benchmark.serving_runtime.subprocess.run",
        lambda *args, **kwargs: pytest.fail("raw image ID must not be inspected"),
    )

    assert resolve_docker_image_id(image_id) == (image_id, ())


def test_image_inspect_failure_is_bounded_and_does_not_copy_stderr(monkeypatch):
    secret = "hf_secret_that_must_not_be_retained"

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 125, "", secret)

    monkeypatch.setattr(
        "Magpie.modes.benchmark.serving_runtime.subprocess.run",
        fake_run,
    )

    resolved, errors = resolve_docker_image_id("missing/image:fixed")

    assert resolved == ""
    assert errors == ("Docker image inspection failed with code 125",)
    assert secret not in json.dumps(errors)


def test_pending_receipt_rejects_image_slot_mismatch():
    expected = "sha256:" + "3" * 64
    wrong = "sha256:" + "4" * 64
    command = _docker_command("magpie-benchmark-case", wrong)

    receipt = pending_serving_runtime_receipt(
        execution_mode="docker",
        input_config_sha256="5" * 64,
        framework="vllm",
        input_image="example/vllm:fixed",
        input_image_id=expected,
        requested_image="example/vllm:fixed",
        resolved_image_id=expected,
        container_name="magpie-benchmark-case",
        docker_argv=command,
    )

    assert receipt["verified"] is False
    assert receipt["process_succeeded"] is False
    assert receipt["docker_argv_sha256"] == canonical_docker_argv_sha256(
        command
    )
    assert receipt["errors"] == [
        "Docker argv image does not match the resolved image ID"
    ]


def test_tracelens_receipt_binds_input_image_to_validated_derived_runtime():
    input_image = "sha256:" + "a" * 64
    derived_image = "magpie-tracelens-vllm:v19-candidate"
    derived_image_id = "sha256:" + "b" * 64
    container_name = "magpie-benchmark-tracelens"
    command = _docker_command(container_name, derived_image_id)
    runtime = _tracelens_runtime(
        base_image=input_image,
        base_image_id=input_image,
        derived_image=derived_image,
        derived_image_id=derived_image_id,
    )

    receipt = pending_serving_runtime_receipt(
        execution_mode="docker",
        input_config_sha256="5" * 64,
        framework="vllm",
        input_image=input_image,
        input_image_id=input_image,
        requested_image=derived_image,
        resolved_image_id=derived_image_id,
        container_name=container_name,
        docker_argv=command,
        tracelens_runtime=runtime,
    )

    assert receipt["errors"] == []
    assert receipt["input_image"] == input_image
    assert receipt["input_image_id"] == input_image
    assert receipt["requested_image"] == derived_image
    assert receipt["resolved_image_id"] == derived_image_id
    assert receipt["image_derivation"] == {
        "kind": "tracelens-derived",
        "framework": "vllm",
        "runtime_schema": "magpie.tracelens-vllm-runtime/v1",
        "base_image": input_image,
        "base_image_id": input_image,
        "base_image_locator": input_image,
        "derived_image": derived_image,
        "derived_image_id": derived_image_id,
        "tracelens_source_commit": "1" * 40,
        "tracelens_source_tree": "2" * 40,
        "patch_version": "v19",
        "patch_path": (
            "examples/custom_workflows/inference_analysis/vllm_patches/"
            "config_vllm_v0.19.0.patch"
        ),
        "patch_sha256": "3" * 64,
        "dependency_wheel_manifest_sha256": "4" * 64,
        "validator": "vllm-tracelens-runtime-validation/v1",
        "verified": True,
    }


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            lambda runtime: runtime["public_runtime_validation"].update(valid=False),
            "serving image derivation is not verified",
        ),
        (
            lambda runtime: runtime.update(base_image_id="sha256:" + "c" * 64),
            "serving image derivation is not verified",
        ),
        (
            lambda runtime: runtime.update(dependency_wheel_manifest_sha256="bad"),
            "TraceLens wheel manifest SHA-256 is invalid",
        ),
        (
            lambda runtime: runtime.update(base_image_locator="mutable:tag"),
            "TraceLens base image locator is missing",
        ),
        (
            lambda runtime: runtime.update(
                tracelens_patch_path=(
                    "examples/custom_workflows/inference_analysis/vllm_patches/"
                    "config_vllm_v0.20.0.patch"
                )
            ),
            "TraceLens patch path is invalid",
        ),
    ],
)
def test_tracelens_receipt_rejects_unverified_or_malformed_lineage(
    mutation,
    expected_error,
):
    input_image = "sha256:" + "a" * 64
    derived_image = "magpie-tracelens-vllm:v19-candidate"
    derived_image_id = "sha256:" + "b" * 64
    runtime = _tracelens_runtime(
        base_image=input_image,
        base_image_id=input_image,
        derived_image=derived_image,
        derived_image_id=derived_image_id,
    )
    mutation(runtime)

    receipt = pending_serving_runtime_receipt(
        execution_mode="docker",
        input_config_sha256="5" * 64,
        framework="vllm",
        input_image=input_image,
        input_image_id=input_image,
        requested_image=derived_image,
        resolved_image_id=derived_image_id,
        container_name="magpie-benchmark-tracelens-invalid",
        docker_argv=_docker_command(
            "magpie-benchmark-tracelens-invalid",
            derived_image_id,
        ),
        tracelens_runtime=runtime,
    )

    assert receipt["verified"] is False
    assert receipt["image_derivation"]["verified"] is False
    assert expected_error in receipt["errors"]


def test_success_receipt_binds_command_without_persisting_token(
    tmp_path,
    monkeypatch,
):
    input_digest = "6" * 64
    image_id = "sha256:" + "7" * 64
    container_name = "magpie-benchmark-success"
    secret = "hf_a_unique_secret_value"
    command = _docker_command(container_name, image_id)
    command[5:5] = ["-e", f"HF_TOKEN={secret}"]
    mode = BenchmarkMode(
        BenchmarkConfig(framework="vllm", model="demo", run_mode="docker"),
        output_dir=str(tmp_path / "results"),
        input_config_sha256=input_digest,
    )
    mode._task_id = "success"
    mode._requested_docker_image = "example/vllm:fixed"
    mode._resolved_docker_image = image_id
    mode._serving_runtime_workspace = tmp_path
    mode._serving_runtime_receipt = pending_serving_runtime_receipt(
        execution_mode="docker",
        input_config_sha256=input_digest,
        framework="vllm",
        input_image="example/vllm:fixed",
        input_image_id=image_id,
        requested_image="example/vllm:fixed",
        resolved_image_id=image_id,
        container_name=container_name,
        docker_argv=command,
    )
    monkeypatch.setattr(mode, "_fix_workspace_ownership", lambda workspace: None)
    monkeypatch.setattr(
        "Magpie.modes.benchmark.benchmarker.subprocess.run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, "ok", ""),
    )

    result, _stdout, _stderr = mode._execute_benchmark(command, tmp_path)

    receipt = result.serving_runtime_receipt
    assert result.success is True
    assert tuple(receipt) == SERVING_RUNTIME_KEYS
    assert receipt == json.loads(
        (tmp_path / "serving_runtime_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["schema"] == SERVING_RUNTIME_SCHEMA
    assert receipt["input_config_sha256"] == input_digest
    assert receipt["input_image"] == "example/vllm:fixed"
    assert receipt["input_image_id"] == image_id
    assert receipt["requested_image"] == "example/vllm:fixed"
    assert receipt["resolved_image_id"] == image_id
    assert receipt["image_derivation"] == {
        "kind": "direct",
        "framework": "vllm",
        "runtime_schema": None,
        "base_image": "example/vllm:fixed",
        "base_image_id": image_id,
        "base_image_locator": "example/vllm:fixed",
        "derived_image": "example/vllm:fixed",
        "derived_image_id": image_id,
        "tracelens_source_commit": None,
        "tracelens_source_tree": None,
        "patch_version": None,
        "patch_path": None,
        "patch_sha256": None,
        "dependency_wheel_manifest_sha256": None,
        "validator": "docker-image-id",
        "verified": True,
    }
    assert receipt["container_name"] == container_name
    assert receipt["docker_argv_sha256"] == canonical_docker_argv_sha256(
        command
    )
    assert receipt["process_succeeded"] is True
    assert receipt["verified"] is True
    assert receipt["errors"] == []
    assert secret not in json.dumps(receipt)
    assert secret not in (tmp_path / "serving_runtime_receipt.json").read_text(
        encoding="utf-8"
    )
    assert BenchmarkResult(serving_runtime_receipt=receipt).to_dict()[
        "serving_runtime_receipt"
    ] == receipt


@pytest.mark.parametrize("tracelens_derived", [False, True])
def test_benchmark_run_reports_resolved_immutable_docker_runtime(
    tmp_path,
    monkeypatch,
    tracelens_derived,
):
    source = tmp_path / "InferenceX"
    source.mkdir()
    requested = "example/vllm:fixed"
    image_id = "sha256:" + "a" * 64
    derived_image = "magpie-tracelens-vllm:v19-test"
    derived_image_id = "sha256:" + "e" * 64
    input_digest = "b" * 64
    config = BenchmarkConfig(
        framework="vllm",
        model="example/model",
        run_mode="docker",
        run_kind="diagnostic" if tracelens_derived else "measurement",
        docker_image=requested,
        inferencex_path=str(source),
        gpu_selection={"auto": False},
        profiler={
            "torch_profiler": {"enabled": False},
            "gpu_monitor": {"enabled": False},
            "tracelens": {"enabled": tracelens_derived},
        },
    )
    mode = BenchmarkMode(
        config,
        output_dir=str(tmp_path / "results"),
        input_config_sha256=input_digest,
    )
    monkeypatch.setattr(
        "Magpie.modes.benchmark.benchmarker.ensure_inferencex_available",
        lambda path: str(source),
    )
    monkeypatch.setattr(
        "Magpie.modes.benchmark.benchmarker.materialize_inferencex_runtime",
        lambda source_path, workspace: SimpleNamespace(
            root=source_path,
            receipt={"source_commit": "c" * 40, "source_tree": "d" * 40},
        ),
    )
    monkeypatch.setattr(mode, "_prepare_benchmark_scripts", lambda: None)
    monkeypatch.setattr(mode, "_get_runner_type", lambda: "mi355x")
    monkeypatch.setattr(
        mode,
        "_get_benchmark_script",
        lambda runner_type: "benchmarks/vllm_mi355x.sh",
    )
    monkeypatch.setattr(
        "Magpie.modes.benchmark.benchmarker.detect_gpu",
        lambda: ("unknown", ""),
    )
    if tracelens_derived:
        runtime = _tracelens_runtime(
            base_image=requested,
            base_image_id=image_id,
            derived_image=derived_image,
            derived_image_id=derived_image_id,
        )
        monkeypatch.setattr(
            "Magpie.modes.benchmark.benchmarker.prepare_tracelens_runtime_image",
            lambda **_kwargs: runtime,
        )

        class FakeTraceLensPipeline:
            def __init__(self, _config):
                pass

            def prepare(self, _workspace):
                return {"warnings": []}

            def restore(self):
                return {"warnings": []}

        monkeypatch.setattr(
            "Magpie.modes.benchmark.benchmarker.TraceLensInferencePipeline",
            FakeTraceLensPipeline,
        )
    main_commands = []

    def fake_benchmark_run(command, **kwargs):
        if command[:3] == ["docker", "image", "inspect"]:
            inspected = command[-1]
            inspected_id = (
                derived_image_id if inspected == derived_image else image_id
            )
            return subprocess.CompletedProcess(
                command,
                0,
                inspected_id + "\n",
                "",
            )
        main_commands.append(command)
        workspace = mode.workspace_mgr.workspace_path
        if "--name" in command:
            (workspace / "inferencex_result.json").write_text(
                json.dumps(
                    {
                        "request_throughput": 1.0,
                        "output_throughput": 10.0,
                        "completed": 1,
                    }
                ),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(
        "Magpie.modes.benchmark.benchmarker.subprocess.run",
        fake_benchmark_run,
    )

    result = mode.run(task_id="full-success")

    assert result.success is True
    receipt = result.serving_runtime_receipt
    assert receipt["verified"] is True
    assert receipt["input_image"] == requested
    assert receipt["input_image_id"] == image_id
    assert receipt["requested_image"] == (
        derived_image if tracelens_derived else requested
    )
    assert receipt["resolved_image_id"] == (
        derived_image_id if tracelens_derived else image_id
    )
    assert receipt["image_derivation"]["kind"] == (
        "tracelens-derived" if tracelens_derived else "direct"
    )
    assert receipt["input_config_sha256"] == input_digest
    benchmark_command = next(
        command for command in main_commands if "--name" in command
    )
    entrypoint = benchmark_command.index("--entrypoint")
    assert benchmark_command[entrypoint + 2] == (
        derived_image_id if tracelens_derived else image_id
    )
    report = json.loads(
        (
            tmp_path
            / "results"
            / Path(result.workspace_dir).name
            / "benchmark_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["serving_runtime_receipt"] == receipt


def test_command_digest_mismatch_fails_before_process_launch(
    tmp_path,
    monkeypatch,
):
    image_id = "sha256:" + "8" * 64
    container_name = "magpie-benchmark-command-swap"
    prepared_command = _docker_command(container_name, image_id)
    changed_command = list(prepared_command)
    changed_command[-1] = "false"
    mode = BenchmarkMode(
        BenchmarkConfig(framework="vllm", model="demo", run_mode="docker"),
        output_dir=str(tmp_path / "results"),
        input_config_sha256="9" * 64,
    )
    mode._task_id = "command-swap"
    mode._requested_docker_image = "example/vllm:fixed"
    mode._resolved_docker_image = image_id
    mode._serving_runtime_workspace = tmp_path
    mode._serving_runtime_receipt = pending_serving_runtime_receipt(
        execution_mode="docker",
        input_config_sha256="9" * 64,
        framework="vllm",
        input_image="example/vllm:fixed",
        input_image_id=image_id,
        requested_image="example/vllm:fixed",
        resolved_image_id=image_id,
        container_name=container_name,
        docker_argv=prepared_command,
    )
    monkeypatch.setattr(
        "Magpie.modes.benchmark.benchmarker.subprocess.run",
        lambda *args, **kwargs: pytest.fail("mismatched command must not launch"),
    )

    result, _stdout, _stderr = mode._execute_benchmark(
        changed_command,
        tmp_path,
    )

    assert result.success is False
    assert result.errors == [
        "Docker serving runtime command binding failed before launch"
    ]
    assert result.serving_runtime_receipt["process_succeeded"] is False
    assert result.serving_runtime_receipt["verified"] is False
    assert any(
        "prepared digest" in error
        for error in result.serving_runtime_receipt["errors"]
    )
