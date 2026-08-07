import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from Magpie.modes.benchmark.benchmarker import BenchmarkMode
from Magpie.modes.benchmark.config import BenchmarkConfig, LmEvalRuntimeConfig
from Magpie.modes.benchmark.lm_eval_runtime import (
    LM_EVAL_EVIDENCE_SCHEMA,
    LM_EVAL_MANIFEST_FILENAME,
    LM_EVAL_MANIFEST_SCHEMA,
    LM_EVAL_RECEIPT_FILENAME,
    collect_lm_eval_runtime_evidence,
    snapshot_runtime_manifest,
    validate_lm_eval_runtime,
)
from Magpie.modes.benchmark.result import BenchmarkResult
from Magpie.utils.gpu import GPUVendor


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "Magpie/scripts/benchmark/lm_eval_runtime.sh"
BASE_IMAGE = (
    "vllm/vllm-openai-rocm@sha256:"
    "c3457ab4702a5bd665b06d7ba57e6105fe98adc4f5b3d4afcf98ec45551988e0"
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_sha256(identity, files) -> str:
    canonical = json.dumps(
        {"identity": identity, "files": files},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _build_runtime(
    tmp_path: Path,
    *,
    python_abi: str | None = None,
) -> tuple[Path, LmEvalRuntimeConfig]:
    root = tmp_path / "runtime"
    site_packages = root / "site-packages"
    package = site_packages / "lm_eval"
    dist_info = site_packages / "lm_eval-0.4.9.2.dist-info"
    package.mkdir(parents=True)
    dist_info.mkdir()
    (package / "__init__.py").write_text("VALUE = 'locked'\n", encoding="utf-8")
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: lm_eval\nVersion: 0.4.9.2\n",
        encoding="utf-8",
    )

    for path in (package / "__init__.py", dist_info / "METADATA"):
        path.chmod(0o444)
    for path in (package, dist_info, site_packages):
        path.chmod(0o555)

    files = []
    for path in sorted(site_packages.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(site_packages).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "mode": stat.S_IMODE(path.stat().st_mode),
                    "sha256": _file_sha256(path),
                }
            )
    identity = {
        "lm_eval_commit": "a" * 40,
        "lm_eval_tree": "b" * 40,
        "lm_eval_version": "0.4.9.2",
        "python_abi": python_abi or sys.implementation.cache_tag,
        "base_image_id": "sha256:" + "c" * 64,
        "base_image_repo_digest": "example/image@sha256:" + "d" * 64,
        "inferencex_commit": "e" * 40,
        "inferencex_tree": "f" * 40,
        "lock_sha256": "1" * 64,
    }
    runtime_sha256 = _runtime_sha256(identity, files)
    manifest = {
        "schema": LM_EVAL_MANIFEST_SCHEMA,
        "runtime_sha256": runtime_sha256,
        "site_packages": "site-packages",
        "identity": identity,
        "files": files,
    }
    manifest_path = root / LM_EVAL_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path.chmod(0o444)
    root.chmod(0o555)
    return root, LmEvalRuntimeConfig(
        path=str(root),
        sha256=runtime_sha256,
        identity=identity,
    )


def _run_helper(root: Path, config: LmEvalRuntimeConfig, workspace: Path):
    workspace.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "MAGPIE_LM_EVAL_RUNTIME_ROOT": str(root),
            "MAGPIE_LM_EVAL_RUNTIME_SHA256": config.sha256,
            "MAGPIE_LM_EVAL_RUNTIME_RECEIPT": str(
                workspace / LM_EVAL_RECEIPT_FILENAME
            ),
            "MAGPIE_LM_EVAL_EXECUTION_MODE": "local",
            "MAGPIE_LM_EVAL_REQUIRE_READONLY_MOUNT": "0",
        }
    )
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" && _install_lm_eval_deps',
            "bash",
            str(HELPER),
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def _inferencex_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "InferenceX"
    (repo / "benchmarks").mkdir(parents=True)
    (repo / "benchmarks/benchmark_lib.sh").write_text(
        "run_benchmark_serving() { return 0; }\n",
        encoding="utf-8",
    )
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "initial")
    return repo


def test_config_round_trip_preserves_nested_runtime(tmp_path):
    _, runtime_config = _build_runtime(tmp_path)
    config = BenchmarkConfig(
        framework="vllm",
        model="example/model",
        lm_eval_runtime=runtime_config,
    )

    restored = BenchmarkConfig.from_dict(config.to_dict())

    assert restored.lm_eval_runtime == runtime_config
    assert restored.to_dict()["lm_eval_runtime"] == runtime_config.to_dict()


def test_validate_runtime_and_helper_emit_bound_evidence(tmp_path):
    root, runtime_config = _build_runtime(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime = validate_lm_eval_runtime(runtime_config)
    snapshot_runtime_manifest(runtime, workspace)

    completed = _run_helper(root, runtime_config, workspace)

    assert completed.returncode == 0, completed.stderr
    evidence = collect_lm_eval_runtime_evidence(
        workspace,
        requested=True,
        config=runtime_config,
        execution_mode="local",
    )
    assert evidence["schema"] == LM_EVAL_EVIDENCE_SCHEMA
    assert evidence["status"] == "verified"
    assert evidence["runtime_sha256"] == runtime_config.sha256
    assert evidence["identity"] == runtime_config.identity
    assert evidence["mount_mode"] == "local"
    assert evidence["manifest_artifact"]["path"] == LM_EVAL_MANIFEST_FILENAME
    assert evidence["receipt_artifact"]["path"] == LM_EVAL_RECEIPT_FILENAME


def test_validate_runtime_rejects_byte_tampering(tmp_path):
    root, runtime_config = _build_runtime(tmp_path)
    target = root / "site-packages/lm_eval/__init__.py"
    target.chmod(0o644)
    target.write_text("VALUE = 'tampered'\n", encoding="utf-8")
    target.chmod(0o444)

    with pytest.raises(ValueError, match="size mismatch|content digest mismatch"):
        validate_lm_eval_runtime(runtime_config)


def test_validate_runtime_rejects_symlink_and_hardlink(tmp_path):
    root, runtime_config = _build_runtime(tmp_path)
    site_packages = root / "site-packages"
    root.chmod(0o755)
    site_packages.chmod(0o755)
    source = site_packages / "lm_eval/__init__.py"
    hardlink = site_packages / "hardlink.py"
    os.link(source, hardlink)
    hardlink.chmod(0o444)
    site_packages.chmod(0o555)
    root.chmod(0o555)

    with pytest.raises(ValueError, match="nlink=1|exactly match"):
        validate_lm_eval_runtime(runtime_config)


def test_run_eval_without_runtime_fails_before_launch(tmp_path):
    inferencex = tmp_path / "InferenceX"
    (inferencex / "benchmarks").mkdir(parents=True)
    config = BenchmarkConfig(
        framework="vllm",
        model="example/model",
        run_mode="local",
        run_kind="measurement",
        envs={"TP": 1, "RUN_EVAL": "true"},
        profiler={
            "torch_profiler": {"enabled": False},
            "gpu_monitor": {"enabled": False},
        },
        gpu_selection={"auto": False},
        inferencex_path=str(inferencex),
    )
    mode = BenchmarkMode(config, output_dir=str(tmp_path / "results"))

    result = mode.run(task_id="missing-evaluator")

    assert result.success is False
    assert result.lm_eval_runtime_receipt["status"] == "invalid"
    assert any("RUN_EVAL=true requires" in item for item in result.errors)


def test_run_eval_ray_fails_closed_before_remote_dispatch(tmp_path, monkeypatch):
    config = BenchmarkConfig(
        framework="vllm",
        model="example/model",
        run_mode="ray",
        envs={"RUN_EVAL": "true"},
    )
    mode = BenchmarkMode(config, output_dir=str(tmp_path / "results"))
    monkeypatch.setattr(
        mode,
        "_execute_ray_benchmark",
        lambda: pytest.fail("Ray dispatch must not run"),
    )

    result = mode.run(task_id="ray-locked-evaluator")

    assert result.success is False
    assert result.lm_eval_runtime_receipt["status"] == "unsupported"
    assert any("Ray benchmark refused" in item for item in result.errors)
    task, error = mode._build_ray_benchmark_task()
    assert task is None
    assert "unsupported in Ray mode" in error


def test_run_eval_rejects_native_or_custom_benchmark_script(tmp_path):
    config = BenchmarkConfig(
        framework="vllm",
        model="example/model",
        run_mode="local",
        envs={"RUN_EVAL": "true"},
    )
    mode = BenchmarkMode(config, output_dir=str(tmp_path / "results"))

    with pytest.raises(RuntimeError, match="native and custom"):
        mode._validate_lm_eval_benchmark_script(
            "benchmarks/single_node/gptoss_fp8_mi355x.sh"
        )
    mode._validate_lm_eval_benchmark_script("benchmarks/vllm_mi355x.sh")


def test_runtime_inferencex_identity_mismatch_fails_before_launch(tmp_path):
    inferencex = _inferencex_repo(tmp_path)
    _, runtime_config = _build_runtime(tmp_path)
    config = BenchmarkConfig(
        framework="vllm",
        model="example/model",
        run_mode="local",
        run_kind="measurement",
        envs={"TP": 1, "RUN_EVAL": "true"},
        profiler={"torch_profiler": {"enabled": False}},
        gpu_selection={"auto": False},
        inferencex_path=str(inferencex),
        benchmark_script="vllm_mi355x.sh",
        lm_eval_runtime=runtime_config,
    )
    mode = BenchmarkMode(config, output_dir=str(tmp_path / "results"))

    result = mode.run(task_id="mismatched-inferencex")

    assert result.success is False
    assert result.lm_eval_runtime_receipt["status"] == "invalid"
    assert any("commit/tree does not match" in item for item in result.errors)


def test_docker_command_mounts_runtime_read_only(tmp_path, monkeypatch):
    _, runtime_config = _build_runtime(tmp_path)
    config = BenchmarkConfig(
        framework="vllm",
        model="example/model",
        run_mode="docker",
        run_kind="measurement",
        envs={"TP": 1, "RUN_EVAL": "true"},
        profiler={"torch_profiler": {"enabled": False}},
        gpu_selection={"auto": False},
        inferencex_path=str(tmp_path / "InferenceX"),
        benchmark_script="vllm_mi355x.sh",
        lm_eval_runtime=runtime_config,
    )
    (tmp_path / "InferenceX").mkdir()
    mode = BenchmarkMode(config, output_dir=str(tmp_path / "results"))
    mode._task_id = "read-only-runtime"
    mode._lm_eval_runtime = validate_lm_eval_runtime(runtime_config)
    monkeypatch.setattr(
        "Magpie.modes.benchmark.benchmarker.detect_gpu",
        lambda: (GPUVendor.UNKNOWN, ""),
    )
    monkeypatch.setattr(
        mode,
        "_get_benchmark_script",
        lambda runner_type: "benchmarks/vllm_mi355x.sh",
    )

    command = mode._build_docker_command(
        "example/image@sha256:" + "a" * 64,
        tmp_path / "workspace",
        "mi355x",
    )

    mount = f"{Path(runtime_config.path).resolve()}:/opt/apex/lm-eval-runtime:ro"
    assert mount in command
    assert "MAGPIE_LM_EVAL_RUNTIME_ROOT=/opt/apex/lm-eval-runtime" in command
    assert f"MAGPIE_LM_EVAL_RUNTIME_SHA256={runtime_config.sha256}" in command
    assert "MAGPIE_LM_EVAL_EXECUTION_MODE=docker" in command
    assert "MAGPIE_LM_EVAL_REQUIRE_READONLY_MOUNT=1" in command


def test_helper_contains_no_mutable_or_network_install_path():
    source = HELPER.read_text(encoding="utf-8")

    assert "_install_lm_eval_deps()" in source
    for forbidden in ("pip install", "git+https://", "curl ", "wget "):
        assert forbidden not in source


def test_owned_evaluator_splits_context_and_output_budget(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    argv_path = tmp_path / "argv.txt"
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$ARGV_PATH\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    include_path = tmp_path / "InferenceX" / "utils" / "evals"
    include_path.mkdir(parents=True)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "ARGV_PATH": str(argv_path),
            "MODEL": "Qwen/example",
            "RESULT_DIR": str(tmp_path / "workspace"),
            "MAGPIE_EVAL_POLICY_ID": "qwen3-next-gsm8k-v1",
            "MAGPIE_EVAL_PRIMARY_METRIC": "exact_match,strict-match",
            "MAGPIE_EVAL_MAX_LENGTH": "2248",
            "MAGPIE_EVAL_MAX_GEN_TOKENS": "480",
            "MAGPIE_EVAL_INCLUDE_PATH": str(include_path),
        }
    )
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; _install_lm_eval_deps() { return 0; }; '
            "magpie_run_lm_eval --port 8888",
            "bash",
            str(HELPER),
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    argv = argv_path.read_text(encoding="utf-8")
    assert "max_length=2248" in argv
    assert "max_tokens=480" in argv
    assert "max_tokens=1124" not in argv
    assert "--log_samples" in argv
    assert "--include_path" in argv
    assert str(include_path) in argv


@pytest.mark.parametrize(
    ("max_length", "max_tokens"),
    (("", "480"), ("2248", ""), ("2248", "2248"), ("bad", "480")),
)
def test_owned_evaluator_rejects_invalid_budget(max_length, max_tokens, tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "MODEL": "Qwen/example",
            "RESULT_DIR": str(tmp_path),
            "MAGPIE_EVAL_POLICY_ID": "qwen3-next-gsm8k-v1",
            "MAGPIE_EVAL_PRIMARY_METRIC": "exact_match,strict-match",
            "MAGPIE_EVAL_MAX_LENGTH": max_length,
            "MAGPIE_EVAL_MAX_GEN_TOKENS": max_tokens,
        }
    )
    completed = subprocess.run(
        ["bash", "-c", 'source "$1"; magpie_run_lm_eval', "bash", str(HELPER)],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 42


def test_helper_failure_terminates_ignoring_upstream_caller():
    env = os.environ.copy()
    for name in (
        "MAGPIE_LM_EVAL_RUNTIME_ROOT",
        "MAGPIE_LM_EVAL_RUNTIME_SHA256",
        "MAGPIE_LM_EVAL_RUNTIME_RECEIPT",
        "MAGPIE_LM_EVAL_EXECUTION_MODE",
        "MAGPIE_LM_EVAL_REQUIRE_READONLY_MOUNT",
    ):
        env.pop(name, None)

    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; _install_lm_eval_deps; echo UNSAFE_CONTINUATION',
            "bash",
            str(HELPER),
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode != 0
    assert "UNSAFE_CONTINUATION" not in completed.stdout
    assert "refusing to run" in completed.stderr


def _base_image_is_local() -> bool:
    if shutil.which("docker") is None:
        return False
    inspected = subprocess.run(
        ["docker", "image", "inspect", BASE_IMAGE],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    return inspected.returncode == 0


@pytest.mark.skipif(
    not _base_image_is_local(),
    reason="pinned vLLM base image is not present locally",
)
def test_helper_import_smoke_is_offline_and_mount_is_read_only(tmp_path):
    root, runtime_config = _build_runtime(tmp_path, python_abi="cpython-312")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    helper_copy = tmp_path / HELPER.name
    shutil.copy2(HELPER, helper_copy)
    snapshot_runtime_manifest(validate_lm_eval_runtime(runtime_config), workspace)

    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "-v",
            f"{root}:/opt/apex/lm-eval-runtime:ro",
            "-v",
            f"{helper_copy}:/opt/apex/lm_eval_runtime.sh:ro",
            "-v",
            f"{workspace}:/workspace",
            "-e",
            "MAGPIE_LM_EVAL_RUNTIME_ROOT=/opt/apex/lm-eval-runtime",
            "-e",
            f"MAGPIE_LM_EVAL_RUNTIME_SHA256={runtime_config.sha256}",
            "-e",
            "MAGPIE_LM_EVAL_RUNTIME_RECEIPT=/workspace/lm_eval_runtime_receipt.json",
            "-e",
            "MAGPIE_LM_EVAL_EXECUTION_MODE=docker",
            "-e",
            "MAGPIE_LM_EVAL_REQUIRE_READONLY_MOUNT=1",
            "--entrypoint",
            "bash",
            BASE_IMAGE,
            "-c",
            "source /opt/apex/lm_eval_runtime.sh && _install_lm_eval_deps",
        ],
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    evidence = collect_lm_eval_runtime_evidence(
        workspace,
        requested=True,
        config=runtime_config,
        execution_mode="docker",
    )
    assert evidence["status"] == "verified"
    assert evidence["mount_mode"] == "read_only"


def test_tampered_runtime_receipt_is_not_reportable(tmp_path):
    root, runtime_config = _build_runtime(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    snapshot_runtime_manifest(validate_lm_eval_runtime(runtime_config), workspace)
    completed = _run_helper(root, runtime_config, workspace)
    assert completed.returncode == 0, completed.stderr
    receipt_path = workspace / LM_EVAL_RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["runtime_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    evidence = collect_lm_eval_runtime_evidence(
        workspace,
        requested=True,
        config=runtime_config,
        execution_mode="local",
    )

    assert evidence["verified"] is False
    assert evidence["status"] == "invalid"
    assert "receipt digest" in evidence["errors"][0]


def test_benchmark_result_serializes_runtime_evidence():
    evidence = {
        "schema": LM_EVAL_EVIDENCE_SCHEMA,
        "status": "verified",
        "runtime_sha256": "a" * 64,
        "mount_mode": "read_only",
    }
    result = BenchmarkResult(lm_eval_runtime_receipt=evidence)

    assert result.to_dict()["lm_eval_runtime_receipt"] == evidence
    assert "lm-eval runtime evidence:" in result.get_summary()
