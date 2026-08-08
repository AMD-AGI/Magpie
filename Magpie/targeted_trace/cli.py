"""CLI adapter for offline Torch-profiler conversion and trace validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml  # type: ignore[import-untyped]

from .config import TargetedTraceConfig
from .postprocess import postprocess_trace_dir
from .torch_profiler import adapt_torch_profiler_traces


def add_targeted_trace_parser(subparsers: Any) -> None:
    """Register the ``targeted-trace`` CLI without coupling it to benchmark code."""

    parser = subparsers.add_parser(
        "targeted-trace",
        help="Adapt or validate diagnostic TargetedKernelTrace artifacts",
    )
    commands = parser.add_subparsers(dest="targeted_trace_command", required=True)

    adapt = commands.add_parser(
        "adapt-torch", help="Stream selected events from Torch profiler traces"
    )
    adapt.add_argument("--trace", type=Path, action="append", default=[])
    adapt.add_argument(
        "--trace-dir",
        type=Path,
        help="Recursively discover .json/.json.gz Torch profiler traces",
    )
    adapt.add_argument("--target-config", type=Path, required=True)
    adapt.add_argument("--output-dir", "-o", type=Path, required=True)
    adapt.add_argument("--run-id", required=True)
    adapt.add_argument("--framework", required=True)
    adapt.add_argument("--framework-version")
    adapt.add_argument("--image")

    postprocess = commands.add_parser(
        "postprocess", help="Stream-validate shards and write a bounded summary"
    )
    postprocess.add_argument("--trace-dir", type=Path, required=True)
    postprocess.add_argument("--output", "-o", type=Path)
    postprocess.add_argument("--strict", action="store_true")


def _load_target_config(path: Path) -> TargetedTraceConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("target config root must be an object")
    raw = data.get("targeted_trace", data)
    if not isinstance(raw, dict):
        raise ValueError("targeted_trace config must be an object")
    raw = dict(raw)
    raw["enabled"] = True
    return TargetedTraceConfig.from_dict(raw)


def _discover_traces(explicit: Iterable[Path], trace_dir: Path | None) -> List[Path]:
    paths = [Path(path) for path in explicit]
    if trace_dir is not None:
        paths.extend(trace_dir.rglob("*.json"))
        paths.extend(trace_dir.rglob("*.json.gz"))
    unique = sorted({path.resolve() for path in paths if path.is_file()})
    if not unique:
        raise ValueError("no Torch profiler trace files were found")
    return unique


def run_targeted_trace(args: argparse.Namespace) -> int:
    """Execute a parsed targeted-trace command."""

    if args.targeted_trace_command == "adapt-torch":
        try:
            config = _load_target_config(args.target_config)
            traces = _discover_traces(args.trace, args.trace_dir)
            manifest = adapt_torch_profiler_traces(
                traces,
                args.output_dir,
                config=config,
                run_id=args.run_id,
                framework=args.framework,
                framework_version=args.framework_version,
                image=args.image,
            )
        except (OSError, ValueError) as exc:
            print(f"Error: {exc}")
            return 1
        print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.targeted_trace_command == "postprocess":
        try:
            summary = postprocess_trace_dir(
                args.trace_dir,
                output_path=args.output,
                strict=args.strict,
            )
        except (OSError, ValueError) as exc:
            print(f"Error: {exc}")
            return 1
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["valid"] else 1
    raise ValueError(f"unknown targeted trace command: {args.targeted_trace_command}")
