import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from Magpie.modes.benchmark.benchmarker import BenchmarkMode
from Magpie.modes.benchmark.config import BenchmarkConfig
from Magpie.modes.benchmark.model_revision import (
    MODEL_REVISION_EVIDENCE_SCHEMA,
    MODEL_REVISION_RECEIPT_SCHEMA,
    collect_model_revision_evidence,
)
from Magpie.modes.benchmark.result import BenchmarkResult


REPO_ROOT = Path(__file__).resolve().parents[1]
VLLM_MI355X_SCRIPT = REPO_ROOT / "Magpie/scripts/benchmark/vllm_mi355x.sh"
LM_EVAL_RUNTIME_HELPER = REPO_ROOT / "Magpie/scripts/benchmark/lm_eval_runtime.sh"
MODEL = "example-org/example-model"
REVISION = "a" * 40


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _script_sandbox(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    scripts = tmp_path / "benchmarks"
    commands = tmp_path / "bin"
    workspace = tmp_path / "workspace"
    snapshots = tmp_path / "snapshots"
    scripts.mkdir()
    commands.mkdir()
    workspace.mkdir()
    snapshots.mkdir()
    script = scripts / VLLM_MI355X_SCRIPT.name
    shutil.copy2(VLLM_MI355X_SCRIPT, script)
    shutil.copy2(LM_EVAL_RUNTIME_HELPER, scripts / LM_EVAL_RUNTIME_HELPER.name)

    (scripts / "benchmark_lib.sh").write_text(
        """
check_env_vars() {
  local name
  for name in "$@"; do
    [[ -n "${!name:-}" ]] || return 1
  done
}
wait_for_server_ready() {
  local attempt
  for attempt in $(seq 1 100); do
    [[ -s "${VLLM_ARGS_LOG:-/nonexistent}" ]] && return 0
    sleep 0.01
  done
  return 1
}
run_benchmark_serving() { return 0; }
magpie_mark_lm_eval_start() { return 0; }
run_eval() {
  printf '%s|%s\n' "${EVAL_CONCURRENT_REQUESTS:-}" "$*" > "$EVAL_LOG"
}
magpie_preserve_lm_eval_artifacts() { return 0; }
append_lm_eval_summary() { return 0; }
""",
        encoding="utf-8",
    )
    (scripts / "server_cleanup.sh").write_text(
        "magpie_stop_benchmark_server_stack() { return 0; }\n",
        encoding="utf-8",
    )

    _write_executable(
        commands / "hf",
        """#!/usr/bin/env bash
printf '%s\n' "$@" > "$HF_ARGS_LOG"
[[ "${HF_FAIL:-0}" == "1" ]] && exit 9
resolved="${HF_RESOLVED_REVISION:-$MODEL_REVISION}"
snapshot="$HF_SNAPSHOT_ROOT/$resolved"
mkdir -p "$snapshot"
printf '%s\n' "$snapshot"
""",
    )
    _write_executable(
        commands / "vllm",
        """#!/usr/bin/env bash
printf '%s\n' "$@" > "$VLLM_ARGS_LOG"
""",
    )
    _write_executable(
        commands / "setsid",
        """#!/usr/bin/env bash
exec "$@"
""",
    )
    _write_executable(commands / "rocm-smi", "#!/usr/bin/env bash\nexit 1\n")

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{commands}:{env['PATH']}",
            "MODEL": MODEL,
            "MODEL_REVISION": REVISION,
            "TP": "1",
            "RESULT_DIR": str(workspace),
            "SERVER_LOG": str(workspace / "server.log"),
            "MAGPIE_RUN_PHASE": "server",
            "MAGPIE_SERVER_PID_FILE": str(workspace / "server.pid"),
            "HF_ARGS_LOG": str(tmp_path / "hf.args"),
            "HF_SNAPSHOT_ROOT": str(snapshots),
            "VLLM_ARGS_LOG": str(tmp_path / "vllm.args"),
            "EVAL_LOG": str(tmp_path / "eval.args"),
            "SLURM_JOB_ID": "",
            "ROCR_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
        }
    )
    return script, env


def _run_script(script: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script)],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


def _receipt_payload(**overrides):
    payload = {
        "schema": MODEL_REVISION_RECEIPT_SCHEMA,
        "model": MODEL,
        "requested_revision": REVISION,
        "resolved_revision": REVISION,
        "snapshot_path": f"/cache/snapshots/{REVISION}",
        "verified": True,
    }
    payload.update(overrides)
    return payload


def test_vllm_mi355x_binds_exact_revision_and_writes_receipt(tmp_path):
    script, env = _script_sandbox(tmp_path)

    completed = _run_script(script, env)

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "hf.args").read_text(encoding="utf-8").splitlines() == [
        "download",
        MODEL,
        "--revision",
        REVISION,
        "--format",
        "quiet",
    ]
    vllm_args = (tmp_path / "vllm.args").read_text(encoding="utf-8").splitlines()
    assert vllm_args[:2] == ["serve", MODEL]
    revision_index = vllm_args.index("--revision")
    assert vllm_args[revision_index + 1] == REVISION

    receipt = json.loads(
        (tmp_path / "workspace/model_revision_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt == _receipt_payload(
        snapshot_path=str((tmp_path / "snapshots" / REVISION).resolve())
    )


@pytest.mark.parametrize(
    "revision",
    ["a" * 39, "A" * 40, "a" * 40 + "0", "main"],
)
def test_vllm_mi355x_rejects_non_exact_revision_before_download(
    tmp_path, revision
):
    script, env = _script_sandbox(tmp_path)
    env["MODEL_REVISION"] = revision

    completed = _run_script(script, env)

    assert completed.returncode == 4
    assert "exact lowercase 40-hex" in completed.stderr
    assert not (tmp_path / "hf.args").exists()
    assert not (tmp_path / "vllm.args").exists()


def test_vllm_mi355x_fails_closed_when_download_fails(tmp_path):
    script, env = _script_sandbox(tmp_path)
    env["HF_FAIL"] = "1"

    completed = _run_script(script, env)

    assert completed.returncode == 9
    assert not (tmp_path / "workspace/model_revision_receipt.json").exists()
    assert not (tmp_path / "vllm.args").exists()


def test_vllm_mi355x_fails_closed_when_resolved_snapshot_differs(tmp_path):
    script, env = _script_sandbox(tmp_path)
    env["HF_RESOLVED_REVISION"] = "b" * 40

    completed = _run_script(script, env)

    assert completed.returncode != 0
    assert "does not match MODEL_REVISION" in completed.stderr
    assert not (tmp_path / "workspace/model_revision_receipt.json").exists()
    assert not (tmp_path / "vllm.args").exists()


def test_vllm_mi355x_passes_eval_concurrency_through_environment(tmp_path):
    script, env = _script_sandbox(tmp_path)
    env.update(
        {
            "MAGPIE_RUN_PHASE": "client",
            "CONC": "16",
            "ISL": "128",
            "OSL": "64",
            "RANDOM_RANGE_RATIO": "1",
            "RESULT_FILENAME": "inferencex_result",
            "RUN_EVAL": "true",
        }
    )
    env.pop("MODEL_REVISION")

    completed = _run_script(script, env)

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "eval.args").read_text(encoding="utf-8").strip() == (
        "16|--framework lm-eval --port 8888"
    )


def test_model_revision_evidence_verifies_workspace_receipt(tmp_path):
    receipt_path = tmp_path / "model_revision_receipt.json"
    receipt_path.write_text(json.dumps(_receipt_payload()), encoding="utf-8")

    evidence = collect_model_revision_evidence(
        tmp_path,
        model=MODEL,
        requested_revision=REVISION,
    )

    assert evidence["schema"] == MODEL_REVISION_EVIDENCE_SCHEMA
    assert evidence["status"] == "verified"
    assert evidence["verified"] is True
    assert evidence["resolved_revision"] == REVISION
    assert evidence["receipt_artifact"]["path"] == receipt_path.name
    assert len(evidence["receipt_artifact"]["sha256"]) == 64


def test_model_revision_evidence_missing_requested_receipt_fails_closed(tmp_path):
    evidence = collect_model_revision_evidence(
        tmp_path,
        model=MODEL,
        requested_revision=REVISION,
    )

    assert evidence["status"] == "missing"
    assert evidence["verified"] is False
    assert evidence["errors"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": "wrong/model"},
        {"resolved_revision": "b" * 40},
        {"snapshot_path": "/cache/snapshots/not-the-revision"},
        {"verified": False},
        {"unexpected": "field"},
    ],
)
def test_model_revision_evidence_rejects_tampered_receipt(tmp_path, overrides):
    (tmp_path / "model_revision_receipt.json").write_text(
        json.dumps(_receipt_payload(**overrides)),
        encoding="utf-8",
    )

    evidence = collect_model_revision_evidence(
        tmp_path,
        model=MODEL,
        requested_revision=REVISION,
    )

    assert evidence["status"] == "invalid"
    assert evidence["verified"] is False
    assert evidence["errors"]


def test_benchmark_result_serializes_model_revision_evidence():
    evidence = collect_model_revision_evidence(
        Path("/nonexistent"),
        model=MODEL,
        requested_revision=None,
    )
    result = BenchmarkResult(model_revision_receipt=evidence)

    report = result.to_dict()

    assert report["model_revision_receipt"]["status"] == "not_requested"
    assert "Model revision evidence:" in result.get_summary()


@pytest.mark.parametrize(
    ("write_receipt", "expected_success", "expected_status"),
    [(True, True, "verified"), (False, False, "missing")],
)
def test_benchmark_mode_report_enforces_requested_revision_receipt(
    tmp_path,
    monkeypatch,
    write_receipt,
    expected_success,
    expected_status,
):
    inferencex = tmp_path / "InferenceX"
    (inferencex / "benchmarks").mkdir(parents=True)
    config = BenchmarkConfig(
        framework="vllm",
        model=MODEL,
        run_mode="local",
        run_kind="measurement",
        envs={"MODEL_REVISION": REVISION, "TP": 1},
        profiler={
            "torch_profiler": {"enabled": False},
            "gpu_monitor": {"enabled": False},
        },
        gpu_selection={"auto": False},
        inferencex_path=str(inferencex),
    )
    mode = BenchmarkMode(config, output_dir=str(tmp_path / "results"))

    monkeypatch.setattr(
        "Magpie.modes.benchmark.benchmarker.ensure_inferencex_available",
        lambda path: str(inferencex),
    )
    monkeypatch.setattr(mode, "_prepare_benchmark_scripts", lambda: None)
    monkeypatch.setattr(mode, "_get_runner_type", lambda: "mi355x")
    monkeypatch.setattr(
        mode,
        "_get_benchmark_script",
        lambda runner_type: "benchmarks/vllm_mi355x.sh",
    )
    monkeypatch.setattr(
        mode,
        "_build_local_command",
        lambda workspace, runner_type: (["true"], {}),
    )
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
        if write_receipt:
            (workspace / "model_revision_receipt.json").write_text(
                json.dumps(_receipt_payload()),
                encoding="utf-8",
            )
        return BenchmarkResult(success=True), "", ""

    monkeypatch.setattr(mode, "_execute_local_benchmark", execute)

    result = mode.run(task_id="revision-contract")

    assert result.success is expected_success
    assert result.model_revision_receipt["status"] == expected_status
    report_path = Path(result.workspace_dir) / "benchmark_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["model_revision_receipt"]["status"] == expected_status
    if not expected_success:
        assert any(
            "Model revision evidence gate failed" in error
            for error in result.errors
        )
