import builtins
import subprocess
from pathlib import Path

import pytest

from Magpie.modes.benchmark.config import TraceLensConfig
from Magpie.modes.benchmark import tracelens


def config(**changes):
    values = {
        "enabled": True,
        "analysis_mode": "pytorch",
        "export_format": "csv",
    }
    values.update(changes)
    return TraceLensConfig(**values)


def completed(cmd, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)


@pytest.mark.parametrize(
    "outcome", ["cli", "import", "installed", "failure", "timeout"]
)
def test_ensure_tracelens_installed(monkeypatch, outcome):
    monkeypatch.setattr(
        tracelens.shutil,
        "which",
        lambda _cmd: "/bin/tool" if outcome == "cli" else None,
    )
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "TraceLens":
            if outcome == "import":
                return object()
            raise ImportError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    if outcome == "timeout":
        monkeypatch.setattr(
            tracelens.subprocess,
            "run",
            lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("pip", 1)),
        )
    else:
        monkeypatch.setattr(
            tracelens.subprocess,
            "run",
            lambda *a, **k: completed(
                a[0], 0 if outcome == "installed" else 1, stderr="bad"
            ),
        )

    assert tracelens.ensure_tracelens_installed() is (
        outcome in {"cli", "import", "installed"}
    )


def test_availability_is_cached_and_checks_cli(monkeypatch):
    analyzer = tracelens.TraceLensAnalyzer(config())
    monkeypatch.setattr(tracelens, "ensure_tracelens_installed", lambda: False)
    assert analyzer.is_available() is False
    monkeypatch.setattr(tracelens, "ensure_tracelens_installed", lambda: True)
    assert analyzer.is_available() is False

    analyzer = tracelens.TraceLensAnalyzer(config())
    monkeypatch.setattr(tracelens.shutil, "which", lambda _cmd: "/bin/tool")
    assert analyzer.is_available() is True


def test_analyze_disabled_unavailable_and_missing_traces(monkeypatch, tmp_path):
    assert tracelens.TraceLensAnalyzer(config(enabled=False)).analyze(
        tmp_path, tmp_path
    ) == {"enabled": False}
    analyzer = tracelens.TraceLensAnalyzer(config())
    monkeypatch.setattr(analyzer, "is_available", lambda: False)
    assert analyzer.analyze(tmp_path, tmp_path)["error"] == "TraceLens not installed"
    monkeypatch.setattr(analyzer, "is_available", lambda: True)
    assert "No trace files" in analyzer.analyze(tmp_path, tmp_path)["errors"][0]


def test_analyze_runs_single_and_multi_reports(monkeypatch, tmp_path):
    trace_dir = tmp_path / "traces"
    output_dir = tmp_path / "out"
    for rank in range(2):
        rank_dir = trace_dir / f"rank_{rank}"
        rank_dir.mkdir(parents=True)
        (rank_dir / "trace.json.gz").write_text("trace")

    analyzer = tracelens.TraceLensAnalyzer(config())
    monkeypatch.setattr(analyzer, "is_available", lambda: True)
    monkeypatch.setattr(
        analyzer,
        "_run_generate_report",
        lambda **kwargs: {"files": ["single.csv"], "error": "single warning"},
    )
    monkeypatch.setattr(
        analyzer,
        "_run_multi_rank_collective",
        lambda **kwargs: {"files": ["multi.csv"], "error": None},
    )
    result = analyzer.analyze(trace_dir, output_dir, num_ranks=2)
    assert result["output_files"] == ["single.csv", "multi.csv"]
    assert result["errors"] == ["single warning"]
    assert (output_dir / "tracelens_rank0_csvs").is_dir()
    assert (output_dir / "tracelens_collective_csvs").is_dir()


def test_find_trace_files_filters_async_llm(tmp_path):
    (tmp_path / "async_llm.json").write_text("trace")
    (tmp_path / "worker.json").write_text("trace")
    analyzer = tracelens.TraceLensAnalyzer(config())
    assert [path.name for path in analyzer._find_trace_files(tmp_path)] == [
        "worker.json"
    ]


@pytest.mark.parametrize(
    ("runner", "expected"),
    [
        (lambda cmd, **kwargs: completed(cmd, 1, stderr="bad"), "CLI failed"),
        (
            lambda cmd, **kwargs: (_ for _ in ()).throw(
                subprocess.TimeoutExpired(cmd, 1)
            ),
            "timed out",
        ),
        (lambda cmd, **kwargs: (_ for _ in ()).throw(FileNotFoundError()), "not found"),
        (lambda cmd, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")), "boom"),
    ],
)
def test_generate_report_failures(monkeypatch, tmp_path, runner, expected):
    analyzer = tracelens.TraceLensAnalyzer(config())
    assert "At least one" in analyzer._run_generate_report(tmp_path / "trace")["error"]
    monkeypatch.setattr(tracelens.subprocess, "run", runner)
    result = analyzer._run_generate_report(tmp_path / "trace", tmp_path / "csv")
    assert expected in result["error"]


def test_generate_report_command_and_outputs(monkeypatch, tmp_path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    (csv_dir / "summary.csv").write_text("ok")
    workbook = tmp_path / "report.xlsx"
    workbook.write_text("ok")
    calls = []
    monkeypatch.setattr(
        tracelens.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append(cmd) or completed(cmd, stdout="done"),
    )
    analyzer = tracelens.TraceLensAnalyzer(config(gpu_arch_config="arch.json"))
    result = analyzer._run_generate_report(tmp_path / "trace.json", csv_dir, workbook)
    assert len(result["files"]) == 2
    assert "--enable_kernel_summary" in calls[0]
    assert calls[0][-2:] == ["--gpu_arch_json_path", "arch.json"]


@pytest.mark.parametrize(
    ("runner", "expected"),
    [
        (lambda cmd, **kwargs: completed(cmd, 1, stderr="bad"), "failed"),
        (
            lambda cmd, **kwargs: (_ for _ in ()).throw(
                subprocess.TimeoutExpired(cmd, 1)
            ),
            "timed out",
        ),
        (lambda cmd, **kwargs: (_ for _ in ()).throw(FileNotFoundError()), "not found"),
        (lambda cmd, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")), "boom"),
    ],
)
def test_multi_rank_failures(monkeypatch, tmp_path, runner, expected):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    for rank in range(2):
        (trace_dir / f"rank-{rank}.json").write_text("trace")
    analyzer = tracelens.TraceLensAnalyzer(config())
    monkeypatch.setattr(tracelens.subprocess, "run", runner)
    result = analyzer._run_multi_rank_collective(
        trace_dir, tmp_path / "csv", num_ranks=2
    )
    assert expected in result["error"]


def test_multi_rank_validation_success_and_patterns(monkeypatch, tmp_path):
    analyzer = tracelens.TraceLensAnalyzer(config())
    assert "At least one" in analyzer._run_multi_rank_collective(tmp_path)["error"]
    single_dir = tmp_path / "single"
    single_dir.mkdir()
    (single_dir / "only.json").write_text("trace")
    assert (
        "at least 2"
        in analyzer._run_multi_rank_collective(
            single_dir, tmp_path / "csv", num_ranks=8
        )["error"]
    )

    (tmp_path / "rank-0.json").write_text("trace")
    (tmp_path / "rank-1.json").write_text("trace")
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir(exist_ok=True)
    (csv_dir / "collective.csv").write_text("ok")
    calls = []
    monkeypatch.setattr(
        tracelens.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append(cmd) or completed(cmd, stdout="done"),
    )
    result = analyzer._run_multi_rank_collective(tmp_path, csv_dir, num_ranks=2)
    assert result["files"] == [str(csv_dir / "collective.csv")]
    assert "--trace_pattern" in calls[0]
    assert analyzer._detect_trace_pattern(tmp_path, []) is None
    assert analyzer._detect_trace_pattern(
        tmp_path, [Path("outside/gpu2.json")]
    ).endswith("gpu*.json")
    plain = tmp_path / "plain.json"
    assert analyzer._detect_trace_pattern(tmp_path, [plain]) is None


@pytest.mark.parametrize(
    ("runner", "expected"),
    [
        (lambda cmd, **kwargs: completed(cmd, 1, stderr="bad"), "failed"),
        (
            lambda cmd, **kwargs: (_ for _ in ()).throw(
                subprocess.TimeoutExpired(cmd, 1)
            ),
            "timed out",
        ),
        (lambda cmd, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")), "boom"),
    ],
)
def test_compare_reports_validation_and_failures(
    monkeypatch, tmp_path, runner, expected
):
    analyzer = tracelens.TraceLensAnalyzer(config())
    assert "At least 2" in analyzer.compare_reports([tmp_path], tmp_path)["error"]
    monkeypatch.setattr(analyzer, "is_available", lambda: False)
    assert (
        analyzer.compare_reports([tmp_path, tmp_path], tmp_path)["error"]
        == "TraceLens not installed"
    )
    monkeypatch.setattr(analyzer, "is_available", lambda: True)
    monkeypatch.setattr(tracelens.subprocess, "run", runner)
    assert expected in analyzer.compare_reports([tmp_path, tmp_path], tmp_path)["error"]


def test_compare_reports_success_and_convenience_functions(monkeypatch, tmp_path):
    output = tmp_path / "output"
    csv_dir = output / "tracelens_comparison_csvs"

    def run(cmd, **kwargs):
        csv_dir.mkdir(parents=True, exist_ok=True)
        (csv_dir / "comparison.csv").write_text("ok")
        (output / "tracelens_comparison.xlsx").write_text("ok")
        return completed(cmd)

    monkeypatch.setattr(tracelens.subprocess, "run", run)
    analyzer = tracelens.TraceLensAnalyzer(config(export_format="excel"))
    monkeypatch.setattr(analyzer, "is_available", lambda: True)
    result = analyzer.compare_reports(
        [tmp_path / "one", tmp_path / "two"], output, ["one", "two"]
    )
    assert len(result["files"]) == 2

    monkeypatch.setattr(
        tracelens.TraceLensAnalyzer, "analyze", lambda self, *args: {"ran": args}
    )
    assert (
        tracelens.run_tracelens_analysis(config(), tmp_path, output, 2)["ran"][-1] == 2
    )
    monkeypatch.setattr(
        tracelens.TraceLensAnalyzer,
        "compare_reports",
        lambda self, *args: {"compared": args},
    )
    assert tracelens.compare_tracelens_reports(
        config(), [tmp_path, output], output, ["a", "b"]
    )["compared"][-1] == ["a", "b"]
