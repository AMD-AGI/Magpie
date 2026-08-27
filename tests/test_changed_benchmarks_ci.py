import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "ci" / "run_changed_benchmarks.py"
WORKFLOW = (
    Path(__file__).parents[1]
    / ".github"
    / "workflows"
    / "benchmark-e2e-cd.yml"
)
SPEC = importlib.util.spec_from_file_location("run_changed_benchmarks", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_gpu_job_requires_manual_dispatch_or_internal_pr_approval():
    workflow = WORKFLOW.read_text()

    assert "github.event_name == 'workflow_dispatch'" in workflow
    assert (
        "github.event.pull_request.head.repo.full_name == github.repository"
        in workflow
    )
    assert "vars.MI355X_PR_BENCHMARK_ENABLED == 'true'" in workflow
    assert "environment:\n      name: mi355x-benchmark" in workflow


def _config(
    path: Path, model: str = "org/model", gpu_arch: str | None = "gfx950"
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = f"benchmark:\n  framework: sglang\n  model: {model}\n"
    if gpu_arch is not None:
        contents += f"  gpu_arch: {gpu_arch}\n"
    path.write_text(contents)
    return path


def test_discover_configs_includes_only_added_modified_examples(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    changed = _config(tmp_path / "examples/benchmarks/changed.yaml")
    untracked = _config(tmp_path / "examples/benchmarks/nested/new.yml")
    ignored = _config(tmp_path / "elsewhere/not-an-example.yaml")

    def fake_git(*args):
        if args[0] == "diff":
            return [
                str(changed.relative_to(tmp_path)),
                str(ignored.relative_to(tmp_path)),
            ]
        return [str(untracked.relative_to(tmp_path))]

    monkeypatch.setattr(MODULE, "_git_lines", fake_git)
    assert MODULE.discover_configs("base", include_untracked=True) == [
        changed.relative_to(tmp_path),
        untracked.relative_to(tmp_path),
    ]


def test_run_configs_dry_run_validates_and_writes_summary(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path / "examples/benchmarks/example.yaml")
    output = tmp_path / "results"

    assert MODULE.run_configs([config.relative_to(tmp_path)], output, dry_run=True) == 0
    results = json.loads((output / "results.json").read_text())
    assert results[0]["status"] == "VALID"
    assert results[0]["runner"] == "mi355x"
    assert "org/model" in (output / "summary.md").read_text()


def test_run_and_tee_streams_and_saves_output(tmp_path, capsys):
    log_path = tmp_path / "runner.log"

    returncode = MODULE._run_and_tee(
        [
            MODULE.sys.executable,
            "-c",
            (
                "import sys; print('live benchmark log', flush=True); "
                "print('live benchmark error', file=sys.stderr, flush=True); "
                "raise SystemExit(7)"
            ),
        ],
        log_path,
    )

    expected = "live benchmark log\nlive benchmark error\n"
    assert returncode == 7
    assert capsys.readouterr().out == expected
    assert log_path.read_text() == expected


def test_run_configs_executes_benchmark_and_summarizes_report(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path / "examples/benchmarks/example.yaml")
    output = tmp_path / "results"

    def fake_run_and_tee(command, log_path):
        config_output = Path(command[command.index("--output-dir") + 1])
        workspace = config_output / "benchmark_sglang_20260825_120000"
        workspace.mkdir(parents=True)
        log_path.write_text("streamed benchmark output\n")
        (workspace / "benchmark_report.json").write_text(
            json.dumps(
                {
                    "success": True,
                    "throughput": {
                        "request_throughput": 12.5,
                        "output_throughput": 6400.0,
                    },
                    "latency": {
                        "ttft": {"mean_ms": 20.0},
                        "tpot": {"mean_ms": 4.0},
                    },
                }
            )
        )
        return 0

    monkeypatch.setattr(MODULE, "_run_and_tee", fake_run_and_tee)
    assert (
        MODULE.run_configs([config.relative_to(tmp_path)], output, dry_run=False) == 0
    )
    results = json.loads((output / "results.json").read_text())
    assert results[0]["status"] == "PASSED"
    summary = (output / "summary.md").read_text()
    assert "12.5" in summary
    assert "20.0" in summary
    assert "4.0" in summary


def test_summary_supports_legacy_flat_latency_metrics():
    summary = MODULE._summary_table(
        [
            {
                "config": "examples/benchmarks/example.yaml",
                "model": "org/model",
                "runner": "mi355x",
                "status": "PASSED",
                "report": {
                    "latency": {"ttft_mean": 21.0, "tpot_mean": 5.0},
                },
            }
        ]
    )

    assert "21.0" in summary
    assert "5.0" in summary


def test_github_matrix_only_schedules_supported_hardware(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    mi355x = _config(tmp_path / "examples/benchmarks/mi355x.yaml")
    unsupported = _config(
        tmp_path / "examples/benchmarks/mi300x.yaml", gpu_arch="gfx942"
    )
    output = tmp_path / "results"
    github_output = tmp_path / "github-output"

    assert (
        MODULE.run_configs(
            [mi355x.relative_to(tmp_path), unsupported.relative_to(tmp_path)],
            output,
            dry_run=True,
            github_output=github_output,
        )
        == 0
    )
    lines = dict(line.split("=", 1) for line in github_output.read_text().splitlines())
    matrix = json.loads(lines["matrix"])
    assert lines["has_benchmarks"] == "true"
    assert matrix["include"] == [
        {
            "config": "examples/benchmarks/mi355x.yaml",
            "runner": "mi355x",
            "id": "examples-benchmarks-mi355x",
        }
    ]


def test_missing_hardware_target_defaults_to_mi355x(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = _config(
        tmp_path / "examples/benchmarks/default.yaml", gpu_arch=None
    ).relative_to(tmp_path)

    assert MODULE.select_runner(MODULE.validate_config(config)) == "mi355x"


@pytest.mark.parametrize(
    "contents,error",
    [
        ("framework: sglang\n", "top-level 'benchmark'"),
        ("benchmark:\n  framework: sglang\n", "model"),
    ],
)
def test_validate_config_rejects_invalid_examples(
    monkeypatch, tmp_path, contents, error
):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "examples/benchmarks/invalid.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(contents)
    with pytest.raises(ValueError, match=error):
        MODULE.validate_config(config.relative_to(tmp_path))
