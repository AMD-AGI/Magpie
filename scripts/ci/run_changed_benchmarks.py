#!/usr/bin/env python3
"""Run added or modified benchmark example YAML files and summarize results."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

BENCHMARK_ROOT = Path("examples/benchmarks")
BENCHMARK_SUFFIXES = {".yaml", ".yml"}
RUNNER_BY_GPU_ARCH = {"gfx950": "mi355x"}


def _git_lines(*args: str) -> list[str]:
    proc = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def is_benchmark_example(path: Path) -> bool:
    try:
        path.relative_to(BENCHMARK_ROOT)
    except ValueError:
        return False
    return path.suffix.lower() in BENCHMARK_SUFFIXES


def discover_configs(
    base: str | None,
    head: str = "HEAD",
    include_untracked: bool = False,
) -> list[Path]:
    """Return added/modified benchmark examples between two revisions."""
    paths: set[Path] = set()
    if base:
        for name in _git_lines(
            "diff",
            "--name-only",
            "--diff-filter=AMR",
            base,
            head,
            "--",
            str(BENCHMARK_ROOT),
        ):
            path = Path(name)
            if is_benchmark_example(path) and path.is_file():
                paths.add(path)

    if include_untracked:
        for name in _git_lines(
            "ls-files", "--others", "--exclude-standard", "--", str(BENCHMARK_ROOT)
        ):
            path = Path(name)
            if is_benchmark_example(path) and path.is_file():
                paths.add(path)

    return sorted(paths)


def validate_config(path: Path) -> dict[str, Any]:
    """Load a benchmark example and validate the fields needed by the CLI."""
    if not is_benchmark_example(path):
        raise ValueError(f"config is outside {BENCHMARK_ROOT}: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("benchmark"), dict):
        raise ValueError("missing top-level 'benchmark' mapping")
    benchmark = data["benchmark"]
    missing = [key for key in ("framework", "model") if not benchmark.get(key)]
    if missing:
        raise ValueError(f"missing required benchmark field(s): {', '.join(missing)}")
    return benchmark


def select_runner(benchmark: dict[str, Any]) -> str:
    """Map an explicit benchmark hardware target to a GitHub runner label."""
    runner_type = str(benchmark.get("runner_type", "")).lower()
    if runner_type:
        return runner_type if runner_type == "mi355x" else ""

    gpu_arch = str(benchmark.get("gpu_arch", "")).lower()
    if gpu_arch:
        return RUNNER_BY_GPU_ARCH.get(gpu_arch, "")

    return "mi355x"


def _slug(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", path.with_suffix("").as_posix())


def _run_and_tee(command: list[str], log_path: Path) -> int:
    """Run a command while streaming combined output to stdout and a log file."""
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as proc:
            if proc.stdout is None:
                raise RuntimeError("benchmark process stdout pipe was not created")
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return proc.wait()


def _latest_report(output_dir: Path) -> Path | None:
    reports = sorted(
        output_dir.glob("benchmark_*/benchmark_report.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return reports[0] if reports else None


def _metric(report: dict[str, Any], group: str, key: str) -> Any:
    value = report.get(group) or {}
    return value.get(key, "") if isinstance(value, dict) else ""


def _summary_table(results: Iterable[dict[str, Any]]) -> str:
    rows = list(results)
    lines = [
        "# Changed benchmark results",
        "",
        "| Config | Model | Runner | Status | Requests/s | Output tokens/s | Mean TTFT (ms) | Mean TPOT (ms) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    if not rows:
        lines.append(
            "| _No added or modified benchmark example YAML files_ | | | Skipped | | | | |"
        )
    for result in rows:
        report = result.get("report") or {}
        lines.append(
            "| {config} | {model} | {runner} | {status} | {request_tp} | {output_tp} | "
            "{ttft} | {tpot} |".format(
                config=result["config"],
                model=result.get("model", ""),
                runner=result.get("runner", "unsupported"),
                status=result["status"],
                request_tp=_metric(report, "throughput", "request_throughput"),
                output_tp=_metric(report, "throughput", "output_throughput"),
                ttft=_metric(report, "latency", "ttft_mean"),
                tpot=_metric(report, "latency", "tpot_mean"),
            )
        )
    lines.append("")
    for result in rows:
        if result.get("error"):
            lines.extend(
                [f"## {result['config']}", "", f"Error: `{result['error']}`", ""]
            )
    return "\n".join(lines)


def _write_github_matrix(results: list[dict[str, Any]], output_path: Path) -> None:
    include = [
        {
            "config": result["config"],
            "runner": result["runner"],
            "id": _slug(Path(result["config"])),
        }
        for result in results
        if result["status"] == "VALID" and result.get("runner")
    ]
    with output_path.open("a", encoding="utf-8") as output:
        output.write(
            f"matrix={json.dumps({'include': include}, separators=(',', ':'))}\n"
        )
        output.write(f"has_benchmarks={'true' if include else 'false'}\n")


def run_configs(
    configs: list[Path],
    output_root: Path,
    dry_run: bool,
    github_output: Path | None = None,
) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for config in configs:
        entry: dict[str, Any] = {"config": config.as_posix(), "status": "FAILED"}
        try:
            benchmark = validate_config(config)
            entry["model"] = benchmark["model"]
            entry["gpu_arch"] = benchmark.get("gpu_arch", "")
            entry["runner"] = select_runner(benchmark)
        except Exception as exc:
            entry["error"] = f"invalid config: {exc}"
            results.append(entry)
            continue

        if dry_run:
            entry["status"] = "VALID"
            results.append(entry)
            continue

        config_output = output_root / _slug(config)
        config_output.mkdir(parents=True, exist_ok=True)
        log_path = config_output / "runner.log"
        command = [
            sys.executable,
            "-m",
            "Magpie",
            "benchmark",
            "--benchmark-config",
            str(config),
            "--output-dir",
            str(config_output),
        ]
        returncode = _run_and_tee(command, log_path)

        report_path = _latest_report(config_output)
        if report_path:
            try:
                entry["report"] = json.loads(report_path.read_text(encoding="utf-8"))
                entry["report_path"] = str(report_path)
            except (OSError, json.JSONDecodeError) as exc:
                entry["error"] = f"cannot read benchmark report: {exc}"
        if returncode == 0 and entry.get("report", {}).get("success") is True:
            entry["status"] = "PASSED"
        else:
            entry.setdefault("error", f"benchmark exited with code {returncode}")
        results.append(entry)

    (output_root / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "summary.md").write_text(_summary_table(results), encoding="utf-8")
    if github_output:
        _write_github_matrix(results, github_output)
    return 1 if any(row["status"] == "FAILED" for row in results) else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", type=Path)
    parser.add_argument("--base", help="base Git revision used for change detection")
    parser.add_argument("--head", default="HEAD", help="head Git revision")
    parser.add_argument("--include-untracked", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-ci-results"))
    parser.add_argument(
        "--github-output",
        type=Path,
        help="write the selected runner matrix to a GitHub Actions output file",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="validate without GPU execution"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configs = list(args.configs)
    if not configs:
        configs = discover_configs(args.base, args.head, args.include_untracked)
    invalid_paths = [path for path in configs if not is_benchmark_example(path)]
    if invalid_paths:
        print(
            f"Refusing non-example config(s): {', '.join(map(str, invalid_paths))}",
            file=sys.stderr,
        )
        return 2
    print("Benchmark configs:", *(str(path) for path in configs), sep="\n  ")
    return run_configs(
        sorted(set(configs)), args.output_dir, args.dry_run, args.github_output
    )


if __name__ == "__main__":
    raise SystemExit(main())
