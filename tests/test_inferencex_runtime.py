import json
import subprocess
from pathlib import Path

import pytest

from Magpie.modes.benchmark.benchmarker import BenchmarkMode
from Magpie.modes.benchmark.config import BenchmarkConfig
from Magpie.modes.benchmark.inferencex_runtime import (
    INFERENCEX_RUNTIME_RECEIPT_FILENAME,
    INFERENCEX_RUNTIME_RECEIPT_SCHEMA,
    materialize_inferencex_runtime,
)
from Magpie.modes.benchmark.result import BenchmarkResult


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def _create_inferencex_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "InferenceX"
    benchmarks = repo / "benchmarks"
    benchmarks.mkdir(parents=True)
    (benchmarks / "benchmark_lib.sh").write_text(
        "run_benchmark_serving() { return 0; }\n",
        encoding="utf-8",
    )
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "initial")
    return repo


def test_materialize_inferencex_runtime_exports_commit_without_source_writes(
    tmp_path,
):
    source = _create_inferencex_repo(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    status_before = _git(source, "status", "--porcelain=v1", "--untracked-files=all")
    commit = _git(source, "rev-parse", "HEAD")
    tree = _git(source, "rev-parse", "HEAD^{tree}")

    runtime = materialize_inferencex_runtime(source, workspace)

    status_after = _git(source, "status", "--porcelain=v1", "--untracked-files=all")
    assert status_after == status_before == ""
    assert runtime.root == workspace / "inferencex_runtime"
    assert not (runtime.root / ".git").exists()
    assert runtime.receipt == json.loads(
        (workspace / INFERENCEX_RUNTIME_RECEIPT_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert runtime.receipt["schema"] == INFERENCEX_RUNTIME_RECEIPT_SCHEMA
    assert runtime.receipt["source_commit"] == commit
    assert runtime.receipt["source_tree"] == tree
    assert runtime.receipt["source_clean"] is True
    assert runtime.receipt["source_status_unchanged"] is True
    assert runtime.receipt["materialization_method"] == (
        "git_private_index_checkout"
    )

    runtime_lib = runtime.root / "benchmarks/benchmark_lib.sh"
    runtime_lib.write_text("runtime-only change\n", encoding="utf-8")
    assert (source / "benchmarks/benchmark_lib.sh").read_text(
        encoding="utf-8"
    ).startswith("run_benchmark_serving")
    assert _git(source, "status", "--porcelain=v1", "--untracked-files=all") == ""


def test_materialize_inferencex_runtime_ignores_dirty_and_untracked_source(
    tmp_path,
):
    source = _create_inferencex_repo(tmp_path)
    tracked = source / "benchmarks/benchmark_lib.sh"
    committed_content = tracked.read_text(encoding="utf-8")
    tracked.write_text("unstaged source change\n", encoding="utf-8")
    (source / "benchmarks/untracked.sh").write_text(
        "untracked\n", encoding="utf-8"
    )
    status_before = _git(source, "status", "--porcelain=v1", "--untracked-files=all")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    runtime = materialize_inferencex_runtime(source, workspace)

    assert (runtime.root / "benchmarks/benchmark_lib.sh").read_text(
        encoding="utf-8"
    ) == committed_content
    assert not (runtime.root / "benchmarks/untracked.sh").exists()
    assert runtime.receipt["source_clean"] is False
    assert _git(
        source, "status", "--porcelain=v1", "--untracked-files=all"
    ) == status_before


def test_benchmark_mode_adds_scripts_only_to_disposable_runtime(
    tmp_path,
    monkeypatch,
):
    source = _create_inferencex_repo(tmp_path)
    source_status = _git(
        source, "status", "--porcelain=v1", "--untracked-files=all"
    )
    config = BenchmarkConfig(
        framework="vllm",
        model="example/model",
        run_mode="local",
        run_kind="measurement",
        envs={"TP": 1},
        profiler={
            "torch_profiler": {"enabled": False},
            "gpu_monitor": {"enabled": False},
        },
        gpu_selection={"auto": False},
        inferencex_path=str(source),
        runner_type="mi355x",
        benchmark_script="vllm_mi355x.sh",
    )
    mode = BenchmarkMode(config, output_dir=str(tmp_path / "results"))
    monkeypatch.setattr(mode, "_cleanup_server_processes", lambda framework: None)

    def execute(local_cmd, local_env, workspace):
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
        return BenchmarkResult(success=True), "", ""

    monkeypatch.setattr(mode, "_execute_local_benchmark", execute)

    result = mode.run(task_id="non-mutating-inferencex")

    assert result.success is True
    assert _git(
        source, "status", "--porcelain=v1", "--untracked-files=all"
    ) == source_status
    assert not (source / "benchmarks/vllm_mi355x.sh").exists()
    runtime_root = Path(result.workspace_dir) / "inferencex_runtime"
    assert (runtime_root / "benchmarks/vllm_mi355x.sh").is_file()
    assert config.inferencex_path == str(runtime_root)
    assert result.inferencex_runtime_receipt["source_commit"] == _git(
        source, "rev-parse", "HEAD"
    )
    assert result.inferencex_runtime_receipt["source_tree"] == _git(
        source, "rev-parse", "HEAD^{tree}"
    )
    report = json.loads(
        (Path(result.workspace_dir) / "benchmark_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["inferencex_runtime_receipt"] == (
        result.inferencex_runtime_receipt
    )


def test_prepare_benchmark_scripts_refuses_source_checkout(tmp_path):
    source = _create_inferencex_repo(tmp_path)
    config = BenchmarkConfig(
        framework="vllm",
        model="example/model",
        run_mode="local",
        inferencex_path=str(source),
    )
    mode = BenchmarkMode(config, output_dir=str(tmp_path / "results"))
    mode._inferencex_source_path = str(source.resolve())
    source_status = _git(
        source, "status", "--porcelain=v1", "--untracked-files=all"
    )

    with pytest.raises(RuntimeError, match="refusing to install"):
        mode._prepare_benchmark_scripts()

    assert _git(
        source, "status", "--porcelain=v1", "--untracked-files=all"
    ) == source_status
