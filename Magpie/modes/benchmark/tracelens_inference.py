###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
TraceLens inference-mode integration for benchmark mode.

This module ports the Magpie-facing preprocess/postprocess workflow into
Magpie so TraceLens public inference analysis can be driven directly by
``magpie benchmark``. TL_EXTENSION is passed through to TraceLens without
Magpie interpreting the extension value.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import BenchmarkConfig
from .tracelens import ensure_tracelens_installed
from ...utils.gpu import detect_gpu

logger = logging.getLogger(__name__)

BACKUP_SUFFIX = ".tracelens.bak"
CLI_SPLIT_INFERENCE_TRACE = "TraceLens_split_inference_trace"
CLI_INFERENCE_REPORT = "TraceLens_generate_perf_report_pytorch_inference"
SGLANG_PROFILE_CUDA_GRAPH_FLAG = "--enable-profile-cuda-graph"
SGLANG_SHAPE_DISCOVERY_FLAG = "--enable-shape-discovery-for-cuda-graph-profile"


@dataclass
class InferencePhasePick:
    """Selected single-iteration trace for one inference phase."""

    stage: str
    csv_kind: str
    output_label: str
    batch_size: float
    trace_path: Optional[Path]
    row_index: Optional[int] = None
    phase_avg_conc: Optional[float] = None
    num_gpu_events: Optional[int] = None
    gpu_duration: Optional[float] = None
    gpu_busy_duration: Optional[float] = None
    selection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "csv_kind": self.csv_kind,
            "output_label": self.output_label,
            "batch_size": self.batch_size,
            "trace_path": str(self.trace_path) if self.trace_path else None,
            "row_index": self.row_index,
            "phase_avg_conc": self.phase_avg_conc,
            "num_gpu_events": self.num_gpu_events,
            "gpu_duration": self.gpu_duration,
            "gpu_busy_duration": self.gpu_busy_duration,
            "selection_reason": self.selection_reason,
        }


@dataclass
class _TraceCandidate:
    kind: str
    path: Path
    batch_size: float
    row_index: int
    phase_avg_conc: Optional[float]
    num_gpu_events: Optional[int]
    gpu_duration: Optional[float]
    gpu_busy_duration: Optional[float]


def resolve_tl_extension(envs: Dict[str, Any]) -> Optional[str]:
    """Resolve TL_EXTENSION from host env first, then benchmark envs."""
    host_value = os.environ.get("TL_EXTENSION", "").strip()
    if host_value:
        return host_value

    cfg_value = str(envs.get("TL_EXTENSION", "") or "").strip()
    return cfg_value or None


def is_tracelens_patched_sglang_image(image_name: Optional[str]) -> bool:
    """Detect image names that are expected to carry TraceLens SGLang patches."""
    if not image_name:
        return False

    lowered = image_name.lower()
    return "tracelens" in lowered and "sglang" in lowered


def host_sglang_supports_shape_discovery() -> bool:
    """Detect whether the host SGLang install supports the patched flag."""
    try:
        from sglang.srt.server_args import ServerArgs  # type: ignore[import-not-found]
    except Exception:
        return False
    return hasattr(ServerArgs, "enable_shape_discovery_for_cuda_graph_profile")


def compute_steady_state_iters(
    osl: Any,
    conc: Any,
    random_range_ratio: Any = 1.0,
) -> Tuple[int, int]:
    """Compute TraceLens-friendly steady-state profiler iteration bounds."""
    osl_i = int(float(osl))
    conc_i = int(float(conc))
    ratio = float(random_range_ratio)
    if osl_i <= 0 or conc_i <= 0:
        raise ValueError(
            f"OSL and CONC must be positive, got OSL={osl}, CONC={conc}"
        )

    max_iters = min(1024, max(256, (osl_i * 16) // conc_i))
    delay_iters = int(osl_i * (ratio + 1.0) * 3 - max_iters / 2)
    return int(max_iters), max(0, delay_iters)


def append_flag_value_args(
    envs: Dict[str, Any],
    key: str,
    pairs: Sequence[Tuple[str, Any]],
) -> None:
    """Append flag/value pairs to an EXTRA_*_ARGS env without duplicating flags."""
    current = str(envs.get(key, "") or "").strip()
    parts = current.split()
    appended: List[str] = []
    for flag, value in pairs:
        if flag in parts or flag in appended:
            continue
        appended.append(flag)
        if value not in (None, ""):
            appended.append(str(value))

    if not appended:
        return

    addition = " ".join(appended)
    envs[key] = f"{current} {addition}".strip() if current else addition


def trace_arch_platform_from_runner(runner_type: Optional[str]) -> Optional[str]:
    """Map Magpie runner/GPU architecture naming to TraceLens platform names."""
    if runner_type:
        runner = runner_type.lower()
        aliases = {
            "mi300": "MI300X",
            "mi300x": "MI300X",
            "mi325": "MI325X",
            "mi325x": "MI325X",
            "mi350": "MI350X",
            "mi350x": "MI350X",
            "mi355": "MI355X",
            "mi355x": "MI355X",
            "mi455": "MI455X",
            "mi455x": "MI455X",
        }
        return aliases.get(runner, runner_type.upper())

    try:
        _, arch = detect_gpu()
    except Exception as exc:
        logger.warning("Could not auto-detect GPU arch for TraceLens: %s", exc)
        return None

    arch_map = {
        "gfx942": "MI300X",
        "gfx950": "MI355X",
        "gfx1100": "MI325X",
    }
    return arch_map.get(arch)


class TraceLensInferencePipeline:
    """Preprocess and postprocess TraceLens inference benchmark traces."""

    def __init__(self, benchmark_config: BenchmarkConfig):
        self.config = benchmark_config
        self.tl_config = benchmark_config.profiler.tracelens
        self.tl_extension = resolve_tl_extension(benchmark_config.envs)
        self._created_backups: List[Path] = []

    def prepare(self, workspace: Path) -> Dict[str, Any]:
        """Patch config/envs and mutable InferenceX files before benchmark."""
        result: Dict[str, Any] = {
            "enabled": True,
            "analysis_mode": self.tl_config.analysis_mode,
            "tl_extension": self.tl_extension,
            "patched_files": [],
            "env_updates": {},
            "warnings": [],
        }

        envs = self.config.envs
        self.config.profiler.torch_profiler.enabled = True
        if self.tl_extension:
            envs["TL_EXTENSION"] = self.tl_extension

        max_iters, delay_iters = compute_steady_state_iters(
            envs.get("OSL", 512),
            envs.get("CONC", 32),
            envs.get("RANDOM_RANGE_RATIO", 1.0),
        )
        result["max_iterations"] = max_iters
        result["delay_iterations"] = delay_iters

        if "NUM_PROMPTS" not in envs:
            envs["NUM_PROMPTS"] = str(int(float(envs.get("CONC", 32))) * 10)

        self._patch_benchmark_lib(result)

        trace_root = self._runtime_torch_trace_dir(workspace)
        capture_dir = f"{trace_root}/capture_traces"

        if self.config.framework == "vllm":
            self._prepare_vllm_env(envs, capture_dir, max_iters, delay_iters)
        elif self.config.framework == "sglang":
            self._prepare_sglang_env(envs, max_iters, delay_iters, result)

        result["env_updates"] = {
            key: envs.get(key)
            for key in sorted(
                {
                    "NUM_PROMPTS",
                    "TL_EXTENSION",
                    "EXTRA_VLLM_ARGS",
                    "EXTRA_SGLANG_ARGS",
                    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS",
                    "SGLANG_PROFILE_WITH_STACK",
                    "SGLANG_PROFILE_RECORD_SHAPE",
                }
            )
            if key in envs
        }
        return result

    def analyze(
        self,
        torch_trace_dir: Path,
        output_dir: Path,
        runner_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Split inference traces and run TraceLens perf reports per stage."""
        results: Dict[str, Any] = {
            "enabled": True,
            "analysis_mode": "inference",
            "analysis_stages": self.tl_config.analysis_stages,
            "tl_extension": self.tl_extension,
            "trace_dir": str(torch_trace_dir),
            "output_files": [],
            "stage_results": {},
            "errors": [],
            "warnings": [],
        }

        if not ensure_tracelens_installed():
            results["errors"].append("TraceLens is not installed")
            return results

        missing_cli = [
            name
            for name in (CLI_SPLIT_INFERENCE_TRACE, CLI_INFERENCE_REPORT)
            if shutil.which(name) is None
        ]
        if missing_cli:
            results["errors"].append(
                "Required TraceLens inference CLI command(s) not found on PATH: "
                + ", ".join(missing_cli)
            )
            return results

        split_dir = torch_trace_dir / "trace_split"
        capture_folder = torch_trace_dir / "capture_traces"
        output_csvs_dir = output_dir / "tracelens"
        gpu_arch_platform = trace_arch_platform_from_runner(runner_type)

        results["split_dir"] = str(split_dir)
        results["capture_folder"] = str(capture_folder)
        results["output_dir"] = str(output_csvs_dir)
        results["gpu_arch_platform"] = gpu_arch_platform

        rank0_trace = self._locate_rank0_trace(torch_trace_dir)
        if rank0_trace is None:
            results["errors"].append(
                f"Could not locate rank-0 trace for framework={self.config.framework} "
                f"in {torch_trace_dir}"
            )
            return results
        results["rank0_trace"] = str(rank0_trace)

        split_error = self._run_splitter(rank0_trace, split_dir)
        if split_error:
            results["errors"].append(split_error)
            return results

        validation_warnings = self._validate_trace_layout(torch_trace_dir, split_dir)
        results["warnings"].extend(validation_warnings)

        execution_csv = split_dir / "execution_details.csv"
        if not execution_csv.exists():
            if self.config.framework == "sglang":
                fallback_warning = (
                    "TraceLens splitter did not produce execution_details.csv; "
                    "trying SGLang step marker fallback"
                )
                logger.warning(fallback_warning)
                results["warnings"].append(fallback_warning)
                results["warnings"].extend(
                    self._split_sglang_step_markers(rank0_trace, split_dir)
                )

        execution_csv = split_dir / "execution_details.csv"
        if not execution_csv.exists():
            results["errors"].append(f"Missing TraceLens split CSV: {execution_csv}")
            return results

        picks = self._pick_largest_batch_traces(execution_csv)
        results["phase_picks"] = {
            stage: pick.to_dict()
            for stage, pick in picks.items()
        }

        for stage in self.tl_config.analysis_stages:
            pick = picks.get(stage)
            if pick is None or pick.trace_path is None:
                warning = f"{stage}: no candidate trace found; skipping perf report"
                logger.warning(warning)
                results["warnings"].append(warning)
                continue

            stage_result = self._run_perf_report(
                stage=stage,
                trace_path=pick.trace_path,
                capture_folder=capture_folder,
                output_root=output_csvs_dir,
                output_label=pick.output_label,
                gpu_arch_platform=gpu_arch_platform,
            )
            stage_result["phase_pick"] = pick.to_dict()
            results["stage_results"][stage] = stage_result
            results["output_files"].extend(stage_result.get("files", []))
            if stage_result.get("error"):
                results["errors"].append(stage_result["error"])

        return results

    def analyze_in_container(
        self,
        torch_trace_dir: Path,
        output_dir: Path,
        runner_type: Optional[str],
        docker_image: str,
        workspace: Path,
    ) -> Dict[str, Any]:
        """Run TraceLens inference postprocess inside a TraceLens-ready image.

        Docker benchmark traces are produced inside the runtime container, so
        the matching TraceLens CLI should be resolved from that same runtime
        image instead of the host environment. This method starts short-lived,
        CPU-only containers after the benchmark container has exited and writes
        all outputs back to the mounted benchmark workspace.
        """
        results = self._base_analysis_result(torch_trace_dir)
        results["postprocess_runtime"] = {
            "mode": "docker",
            "image": docker_image,
            "workspace_mount": "/workspace",
        }

        if not docker_image:
            results["errors"].append("TraceLens Docker image is not configured")
            return results

        workspace = workspace.resolve()
        torch_trace_dir = torch_trace_dir.resolve()
        output_dir = output_dir.resolve()

        split_dir = torch_trace_dir / "trace_split"
        capture_folder = torch_trace_dir / "capture_traces"
        output_csvs_dir = output_dir / "tracelens"
        gpu_arch_platform = trace_arch_platform_from_runner(runner_type)

        results["split_dir"] = str(split_dir)
        results["capture_folder"] = str(capture_folder)
        results["output_dir"] = str(output_csvs_dir)
        results["gpu_arch_platform"] = gpu_arch_platform

        rank0_trace = self._locate_rank0_trace(torch_trace_dir)
        if rank0_trace is None:
            results["errors"].append(
                f"Could not locate rank-0 trace for framework={self.config.framework} "
                f"in {torch_trace_dir}"
            )
            return results
        results["rank0_trace"] = str(rank0_trace)

        split_error = self._run_splitter_in_container(
            docker_image=docker_image,
            workspace=workspace,
            rank0_trace=rank0_trace.resolve(),
            split_dir=split_dir,
        )
        if split_error:
            results["errors"].append(split_error)
            return results

        self._rewrite_container_split_paths(
            split_dir / "execution_details.csv",
            workspace,
        )

        validation_warnings = self._validate_trace_layout(torch_trace_dir, split_dir)
        results["warnings"].extend(validation_warnings)

        execution_csv = split_dir / "execution_details.csv"
        if not execution_csv.exists():
            if self.config.framework == "sglang":
                fallback_warning = (
                    "TraceLens splitter did not produce execution_details.csv; "
                    "trying SGLang step marker fallback"
                )
                logger.warning(fallback_warning)
                results["warnings"].append(fallback_warning)
                results["warnings"].extend(
                    self._split_sglang_step_markers(rank0_trace, split_dir)
                )

        execution_csv = split_dir / "execution_details.csv"
        if not execution_csv.exists():
            results["errors"].append(f"Missing TraceLens split CSV: {execution_csv}")
            return results

        picks = self._pick_largest_batch_traces(execution_csv)
        results["phase_picks"] = {
            stage: pick.to_dict()
            for stage, pick in picks.items()
        }

        for stage in self.tl_config.analysis_stages:
            pick = picks.get(stage)
            if pick is None or pick.trace_path is None:
                warning = f"{stage}: no candidate trace found; skipping perf report"
                logger.warning(warning)
                results["warnings"].append(warning)
                continue

            stage_result = self._run_perf_report_in_container(
                docker_image=docker_image,
                workspace=workspace,
                stage=stage,
                trace_path=pick.trace_path.resolve(),
                capture_folder=capture_folder,
                output_root=output_csvs_dir,
                output_label=pick.output_label,
                gpu_arch_platform=gpu_arch_platform,
            )
            stage_result["phase_pick"] = pick.to_dict()
            results["stage_results"][stage] = stage_result
            results["output_files"].extend(stage_result.get("files", []))
            if stage_result.get("error"):
                results["errors"].append(stage_result["error"])

        return results

    def restore(self) -> Dict[str, Any]:
        """Restore files backed up during prepare()."""
        result = {"restored_files": [], "warnings": []}
        if not self.tl_config.restore_patches:
            result["warnings"].append("restore_patches=false; leaving patched files in place")
            return result

        for path in reversed(self._created_backups):
            backup = Path(str(path) + BACKUP_SUFFIX)
            if not backup.exists():
                result["warnings"].append(f"backup missing for restore: {backup}")
                continue
            shutil.move(str(backup), str(path))
            result["restored_files"].append(str(path))
            logger.info("Restored TraceLens preprocess patch: %s", path)
        return result

    def _runtime_torch_trace_dir(self, workspace: Path) -> str:
        if self.config.is_local:
            return str(workspace / "torch_trace")
        return "/workspace/torch_trace"

    def _prepare_vllm_env(
        self,
        envs: Dict[str, Any],
        capture_dir: str,
        max_iters: int,
        delay_iters: int,
    ) -> None:
        append_flag_value_args(
            envs,
            "EXTRA_VLLM_ARGS",
            [
                ("--profiler-config.capture_torch_profiler_dir", capture_dir),
                ("--profiler-config.detailed_trace_annotation", "True"),
                ("--profiler-config.delay_iterations", delay_iters),
                ("--profiler-config.max_iterations", max_iters),
                ("--profiler-config.ignore_frontend", "True"),
            ],
        )
        envs.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "1200")

    def _prepare_sglang_env(
        self,
        envs: Dict[str, Any],
        max_iters: int,
        delay_iters: int,
        result: Dict[str, Any],
    ) -> None:
        envs["SGLANG_PROFILE_WITH_STACK"] = "True"
        envs["SGLANG_PROFILE_RECORD_SHAPE"] = "True"
        sglang_args: List[Tuple[str, Any]] = [
            (SGLANG_PROFILE_CUDA_GRAPH_FLAG, None),
        ]
        if self._should_enable_sglang_shape_discovery():
            sglang_args.append((SGLANG_SHAPE_DISCOVERY_FLAG, None))

        append_flag_value_args(
            envs,
            "EXTRA_SGLANG_ARGS",
            sglang_args,
        )
        self._patch_sglang_benchmark_serving(max_iters, delay_iters, result)

    def _should_enable_sglang_shape_discovery(self) -> bool:
        """Only enable TraceLens-patched SGLang flags for known patched runtimes."""
        return is_tracelens_patched_sglang_image(
            self.config.docker_image
        ) or (self.config.is_local and host_sglang_supports_shape_discovery())

    def _patch_benchmark_lib(self, result: Dict[str, Any]) -> None:
        path = Path(self.config.inferencex_path) / "benchmarks" / "benchmark_lib.sh"
        if not path.exists():
            warning = f"benchmark_lib.sh not found, skipping TraceLens preprocess patch: {path}"
            logger.warning(warning)
            result["warnings"].append(warning)
            return

        text = path.read_text(encoding="utf-8")
        old = 'num_prompts="$max_concurrency"'
        new = 'num_prompts="$num_prompts"'
        if new in text:
            return
        if old not in text:
            warning = f"benchmark_lib.sh profile num_prompts anchor not found: {path}"
            logger.warning(warning)
            result["warnings"].append(warning)
            return

        self._backup_file(path)
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        result["patched_files"].append(str(path))
        logger.info("Patched benchmark_lib.sh for TraceLens inference profiling")

    def _patch_sglang_benchmark_serving(
        self,
        max_iters: int,
        delay_iters: int,
        result: Dict[str, Any],
    ) -> None:
        path = (
            Path(self.config.inferencex_path)
            / "utils"
            / "bench_serving"
            / "benchmark_serving.py"
        )
        if not path.exists():
            warning = f"benchmark_serving.py not found, skipping SGLang TraceLens patch: {path}"
            logger.warning(warning)
            result["warnings"].append(warning)
            return

        text = path.read_text(encoding="utf-8")
        common_anchor = '"num_steps": 1, "merge_profiles": True, "profile_by_stage": True'
        common_repl = (
            '"shape_discovery": True, "roofline_annotations": True, '
            '"num_steps": 1, "merge_profiles": True, "profile_by_stage": True'
        )
        steady_repl = (
            f'"start_step": {delay_iters}, "num_steps": {max_iters}, '
            '"merge_profiles": False, "profile_by_stage": False'
        )

        if steady_repl in text and "roofline_annotations" in text:
            return

        if "roofline_annotations" not in text:
            if common_anchor not in text:
                warning = f"SGLang benchmark_serving.py common patch anchor not found: {path}"
                logger.warning(warning)
                result["warnings"].append(warning)
                return
            self._backup_file(path)
            text = text.replace(common_anchor, common_repl, 1)

        if common_anchor not in text:
            warning = f"SGLang benchmark_serving.py steady-state patch anchor not found: {path}"
            logger.warning(warning)
            result["warnings"].append(warning)
            return

        self._backup_file(path)
        text = text.replace(common_anchor, steady_repl, 1)
        path.write_text(text, encoding="utf-8")
        result["patched_files"].append(str(path))
        logger.info("Patched SGLang benchmark_serving.py for TraceLens inference profiling")

    def _backup_file(self, path: Path) -> None:
        if path in self._created_backups:
            return
        backup = Path(str(path) + BACKUP_SUFFIX)
        if backup.exists():
            logger.warning(
                "TraceLens backup already exists; keeping existing restore point: %s",
                backup,
            )
            return
        shutil.copy2(path, backup)
        self._created_backups.append(path)

    def _locate_rank0_trace(self, torch_trace_dir: Path) -> Optional[Path]:
        if self.config.framework == "vllm":
            patterns = [
                "*-rank_0.*trace.json.gz",
                "*rank0*.pt.trace.json.gz",
                "*rank-0*.trace.json.gz",
                "*.trace.json.gz",
                "*.json.gz",
            ]
        else:
            patterns = [
                "merged-*.trace.json.gz",
                "*TP-0*.trace.json.gz",
                "*-TP-0-DECODE.trace.json.gz",
                "*-TP-0-EXTEND.trace.json.gz",
                "*.trace.json.gz",
                "*.json.gz",
            ]

        for pattern in patterns:
            matches = sorted(torch_trace_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    def _run_splitter(self, rank0_trace: Path, split_dir: Path) -> Optional[str]:
        envs = self.config.envs
        split_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            CLI_SPLIT_INFERENCE_TRACE,
            str(rank0_trace),
            "-o",
            str(split_dir),
            "--find-steady-state",
            "--store-single-iteration",
            "--num-steps",
            str(self.tl_config.num_steps),
            "--CONC",
            str(envs.get("CONC", 32)),
            "--OSL",
            str(envs.get("OSL", 512)),
            "--R",
            str(envs.get("RANDOM_RANGE_RATIO", 1.0)),
        ]
        logger.info("Running TraceLens inference splitter: %s", " ".join(cmd))
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.tl_config.cli_timeout_seconds,
            env=self._subprocess_env(),
        )
        if proc.returncode != 0:
            return f"TraceLens inference split failed: {proc.stderr or proc.stdout}"
        return None

    def _run_splitter_in_container(
        self,
        docker_image: str,
        workspace: Path,
        rank0_trace: Path,
        split_dir: Path,
    ) -> Optional[str]:
        envs = self.config.envs
        split_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            CLI_SPLIT_INFERENCE_TRACE,
            self._container_path(rank0_trace, workspace),
            "-o",
            self._container_path(split_dir, workspace),
            "--find-steady-state",
            "--store-single-iteration",
            "--num-steps",
            str(self.tl_config.num_steps),
            "--CONC",
            str(envs.get("CONC", 32)),
            "--OSL",
            str(envs.get("OSL", 512)),
            "--R",
            str(envs.get("RANDOM_RANGE_RATIO", 1.0)),
        ]
        logger.info(
            "Running TraceLens inference splitter in container %s: %s",
            docker_image,
            " ".join(cmd),
        )
        proc = self._run_container_command(docker_image, workspace, cmd)
        if proc.returncode != 0:
            return (
                "TraceLens inference split failed in container "
                f"{docker_image}: {proc.stderr or proc.stdout}"
            )
        return None

    def _validate_trace_layout(self, torch_trace_dir: Path, split_dir: Path) -> List[str]:
        warnings: List[str] = []
        capture_dir = torch_trace_dir / "capture_traces"
        if not capture_dir.is_dir() or not any(capture_dir.iterdir()):
            warnings.append(f"capture_traces directory is missing or empty: {capture_dir}")
        if not (split_dir / "execution_details.csv").exists():
            warnings.append(f"execution_details.csv missing under split dir: {split_dir}")
        return warnings

    def _split_sglang_step_markers(
        self,
        rank0_trace: Path,
        split_dir: Path,
    ) -> List[str]:
        """Fallback split for SGLang traces annotated with step[...] spans."""
        warnings: List[str] = []
        split_dir.mkdir(parents=True, exist_ok=True)

        trace = self._load_trace(rank0_trace)
        events = trace.get("traceEvents", [])
        if not isinstance(events, list):
            return ["SGLang fallback splitter: traceEvents is not a list"]

        picks = self._pick_sglang_step_events(events)
        if not picks:
            return ["SGLang fallback splitter: no step[...] user annotations found"]

        rows: List[Dict[str, Any]] = []
        for kind, event in picks.items():
            stage, output_label = {
                "DECODE": ("decode", "decode_only"),
                "PREFILL": ("prefill", "prefill_only"),
                "PD": ("prefilldecode", "prefilldecode"),
            }[kind]
            marker_name = str(event.get("name", ""))
            ts = float(event.get("ts", 0.0) or 0.0)
            dur = float(event.get("dur", 0.0) or 0.0)
            if dur <= 0:
                warnings.append(
                    f"SGLang fallback splitter: {marker_name} has no duration"
                )
                continue

            end_ts = self._next_sglang_step_ts(events, ts) or (ts + dur)
            out_path = split_dir / f"{output_label}_step.trace.json.gz"
            self._write_trace_window(trace, out_path, ts, end_ts)
            rows.append(
                {
                    "output_path": str(out_path),
                    "num_steps": 1,
                    "phase_avg_bs": self._sglang_step_batch_size(marker_name),
                    "phase_num_prefilldecode": 1 if kind == "PD" else 0,
                    "phase_num_decode": 1 if kind == "DECODE" else 0,
                    "phase_num_prefill": 1 if kind == "PREFILL" else 0,
                    "stage": stage,
                    "source_marker": marker_name,
                    "start_ts": ts,
                    "end_ts": end_ts,
                }
            )

        if not rows:
            return warnings or ["SGLang fallback splitter: no trace windows written"]

        execution_csv = split_dir / "execution_details.csv"
        with execution_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        warnings.append(
            f"SGLang fallback splitter wrote {len(rows)} trace window(s) to {split_dir}"
        )
        return warnings

    @staticmethod
    def _load_trace(trace_path: Path) -> Dict[str, Any]:
        opener = gzip.open if trace_path.suffix == ".gz" else open
        with opener(trace_path, "rt", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _pick_sglang_step_events(
        events: Sequence[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        candidates: Dict[str, List[Tuple[float, float, Dict[str, Any]]]] = {
            "DECODE": [],
            "PREFILL": [],
            "PD": [],
        }
        for event in events:
            if str(event.get("cat", "")) != "user_annotation":
                continue
            marker_name = str(event.get("name", ""))
            if not marker_name.startswith("step["):
                continue
            kind = TraceLensInferencePipeline._classify_sglang_step_name(marker_name)
            if kind is None:
                continue
            batch_size = TraceLensInferencePipeline._sglang_step_batch_size(marker_name)
            ts = float(event.get("ts", 0.0) or 0.0)
            candidates[kind].append((batch_size, ts, event))

        picks: Dict[str, Dict[str, Any]] = {}
        for kind, kind_candidates in candidates.items():
            if not kind_candidates:
                continue
            max_bs = max(item[0] for item in kind_candidates)
            best_batch = [
                item for item in kind_candidates
                if item[0] == max_bs
            ]
            best_batch.sort(key=lambda item: item[1])
            picks[kind] = best_batch[len(best_batch) // 2][2]
        return picks

    @staticmethod
    def _classify_sglang_step_name(name: str) -> Optional[str]:
        upper = name.upper()
        if "DECODE" in upper:
            return "DECODE"
        if "PREFILLDECODE" in upper or "PREFILL_DECODE" in upper or "MIX" in upper:
            return "PD"
        if "EXTEND" in upper or "PREFILL" in upper:
            return "PREFILL"
        return None

    @staticmethod
    def _sglang_step_batch_size(name: str) -> float:
        match = re.search(r"\bbs=(\d+(?:\.\d+)?)", name)
        if match:
            return float(match.group(1))
        return 0.0

    @staticmethod
    def _next_sglang_step_ts(
        events: Sequence[Dict[str, Any]],
        start_ts: float,
    ) -> Optional[float]:
        next_ts: Optional[float] = None
        for event in events:
            if str(event.get("cat", "")) != "user_annotation":
                continue
            if not str(event.get("name", "")).startswith("step["):
                continue
            try:
                ts = float(event.get("ts", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if ts <= start_ts:
                continue
            if next_ts is None or ts < next_ts:
                next_ts = ts
        return next_ts

    @staticmethod
    def _write_trace_window(
        trace: Dict[str, Any],
        out_path: Path,
        start_ts: float,
        end_ts: float,
    ) -> None:
        window_events = []
        for event in trace.get("traceEvents", []):
            ts_raw = event.get("ts")
            if ts_raw is None:
                window_events.append(event)
                continue
            try:
                ts = float(ts_raw)
                dur = float(event.get("dur", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            event_end = ts + max(dur, 0.0)
            if ts <= end_ts and event_end >= start_ts:
                window_events.append(event)

        out_trace = dict(trace)
        out_trace["traceEvents"] = window_events
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(out_path, "wt", encoding="utf-8") as handle:
            json.dump(out_trace, handle)

    def _pick_largest_batch_traces(
        self,
        execution_csv: Path,
    ) -> Dict[str, InferencePhasePick]:
        by_kind: Dict[str, List[_TraceCandidate]] = {}
        split_dir = execution_csv.parent

        with execution_csv.open(newline="", encoding="utf-8") as handle:
            for row_index, row in enumerate(csv.DictReader(handle)):
                output_path = row.get("output_path", "")
                if not output_path or not self._is_single_iteration_row(row):
                    continue

                kind = self._classify_split_row(row)
                if kind is None:
                    continue

                try:
                    batch_size = float(row.get("phase_avg_bs") or 0.0)
                except ValueError:
                    batch_size = 0.0

                path = Path(output_path)
                if not path.is_absolute():
                    path = split_dir / path
                if not path.exists():
                    continue

                if not self._row_has_gpu_work(row):
                    continue

                by_kind.setdefault(kind, []).append(
                    _TraceCandidate(
                        kind=kind,
                        path=path,
                        batch_size=batch_size,
                        row_index=row_index,
                        phase_avg_conc=self._optional_float(row, "phase_avg_conc"),
                        num_gpu_events=self._optional_int(row, "num_gpu_events"),
                        gpu_duration=self._optional_float(row, "gpu_duration"),
                        gpu_busy_duration=self._optional_float(
                            row,
                            "gpu_busy_duration",
                        ),
                    )
                )

        kind_to_stage = {
            "PD": ("prefilldecode", "prefilldecode"),
            "DECODE": ("decode", "decode_only"),
            "PREFILL": ("prefill", "prefill_only"),
        }
        picks: Dict[str, InferencePhasePick] = {}
        for kind, (stage, output_label) in kind_to_stage.items():
            candidate = self._select_trace_candidate(stage, by_kind.get(kind, []))
            picks[stage] = InferencePhasePick(
                stage=stage,
                csv_kind=kind,
                output_label=output_label,
                batch_size=candidate.batch_size if candidate else 0.0,
                trace_path=candidate.path if candidate else None,
                row_index=candidate.row_index if candidate else None,
                phase_avg_conc=candidate.phase_avg_conc if candidate else None,
                num_gpu_events=candidate.num_gpu_events if candidate else None,
                gpu_duration=candidate.gpu_duration if candidate else None,
                gpu_busy_duration=candidate.gpu_busy_duration if candidate else None,
                selection_reason=(
                    self._selection_reason(stage, candidate)
                    if candidate
                    else "no valid single-iteration trace with GPU work"
                ),
            )
        return picks

    def _select_trace_candidate(
        self,
        stage: str,
        candidates: Sequence[_TraceCandidate],
    ) -> Optional[_TraceCandidate]:
        if not candidates:
            return None

        max_batch = max(candidate.batch_size for candidate in candidates)
        best_batch = [
            candidate
            for candidate in candidates
            if candidate.batch_size == max_batch
        ]

        target_conc = self._optional_float(self.config.envs, "CONC")
        if stage == "decode" and target_conc is not None:
            with_conc = [
                candidate
                for candidate in best_batch
                if candidate.phase_avg_conc is not None
            ]
            if with_conc:
                closest_delta = min(
                    abs(candidate.phase_avg_conc - target_conc)
                    for candidate in with_conc
                    if candidate.phase_avg_conc is not None
                )
                best_batch = [
                    candidate
                    for candidate in with_conc
                    if candidate.phase_avg_conc is not None
                    and abs(candidate.phase_avg_conc - target_conc) == closest_delta
                ]

        with_gpu_busy_duration = [
            candidate
            for candidate in best_batch
            if candidate.gpu_busy_duration is not None
        ]
        if with_gpu_busy_duration:
            with_gpu_busy_duration.sort(
                key=lambda candidate: (
                    candidate.gpu_busy_duration or 0.0,
                    candidate.row_index,
                )
            )
            return with_gpu_busy_duration[len(with_gpu_busy_duration) // 2]

        best_batch.sort(key=lambda candidate: candidate.row_index)
        return best_batch[len(best_batch) // 2]

    def _selection_reason(
        self,
        stage: str,
        candidate: _TraceCandidate,
    ) -> str:
        reason = (
            "largest phase_avg_bs among valid single-iteration traces with GPU work"
        )
        target_conc = self._optional_float(self.config.envs, "CONC")
        if (
            stage == "decode"
            and target_conc is not None
            and candidate.phase_avg_conc is not None
        ):
            reason += f"; closest phase_avg_conc to CONC={target_conc:g}"
        if candidate.gpu_busy_duration is not None:
            reason += "; median gpu_busy_duration tie-break"
        else:
            reason += "; median row_index tie-break"
        return reason

    @staticmethod
    def _optional_float(row: Dict[str, Any], key: str) -> Optional[float]:
        raw = row.get(key)
        if raw in (None, ""):
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_int(row: Dict[str, Any], key: str) -> Optional[int]:
        value = TraceLensInferencePipeline._optional_float(row, key)
        if value is None:
            return None
        return int(value)

    @staticmethod
    def _row_has_gpu_work(row: Dict[str, str]) -> bool:
        metric_keys = ("num_gpu_events", "gpu_duration", "gpu_busy_duration")
        for key in metric_keys:
            raw = row.get(key)
            if raw in (None, ""):
                continue
            try:
                if float(raw) <= 0:
                    return False
            except (TypeError, ValueError):
                return False
        return True

    @staticmethod
    def _is_single_iteration_row(row: Dict[str, str]) -> bool:
        base = os.path.basename(row.get("output_path", ""))
        if base.startswith(
            ("mixed_steady_state_", "prefilldecode_steady_state_", "decode_only_steady_state_")
        ):
            return False
        try:
            return int(float(row.get("num_steps") or 1)) <= 1
        except ValueError:
            return True

    @staticmethod
    def _classify_split_row(row: Dict[str, str]) -> Optional[str]:
        def col(name: str) -> int:
            try:
                return int(float(row.get(name) or 0))
            except ValueError:
                return 0

        if col("phase_num_prefilldecode") == 1:
            return "PD"
        if col("phase_num_decode") == 1:
            return "DECODE"
        if col("phase_num_prefill") == 1:
            return "PREFILL"
        return None

    def _run_perf_report(
        self,
        stage: str,
        trace_path: Path,
        capture_folder: Path,
        output_root: Path,
        output_label: str,
        gpu_arch_platform: Optional[str],
    ) -> Dict[str, Any]:
        out_dir = output_root / output_label
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            CLI_INFERENCE_REPORT,
            "--profile_json_path",
            str(trace_path),
            "--output_csvs_dir",
            str(out_dir),
            "--group_by_parent_module",
            "--enable_pseudo_ops",
            "--group_by_num_kernels",
        ]
        if capture_folder.is_dir() and any(capture_folder.iterdir()):
            cmd.extend(["--capture_folder", str(capture_folder)])
        else:
            logger.warning(
                "TraceLens capture folder missing or empty; "
                "running report without it: %s",
                capture_folder,
            )
        if gpu_arch_platform:
            cmd.extend(["--gpu_arch_platform", gpu_arch_platform])

        logger.info("Running TraceLens inference perf report (%s): %s", stage, " ".join(cmd))
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.tl_config.cli_timeout_seconds,
            env=self._subprocess_env(),
        )

        result: Dict[str, Any] = {
            "stage": stage,
            "trace_path": str(trace_path),
            "output_dir": str(out_dir),
            "files": [str(path) for path in sorted(out_dir.glob("*.csv"))],
        }
        if proc.returncode != 0:
            result["error"] = (
                f"TraceLens inference perf report failed for {stage}: "
                f"{proc.stderr or proc.stdout}"
            )
        return result

    def _run_perf_report_in_container(
        self,
        docker_image: str,
        workspace: Path,
        stage: str,
        trace_path: Path,
        capture_folder: Path,
        output_root: Path,
        output_label: str,
        gpu_arch_platform: Optional[str],
    ) -> Dict[str, Any]:
        out_dir = output_root / output_label
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            CLI_INFERENCE_REPORT,
            "--profile_json_path",
            self._container_path(trace_path, workspace),
            "--output_csvs_dir",
            self._container_path(out_dir, workspace),
            "--group_by_parent_module",
            "--enable_pseudo_ops",
            "--group_by_num_kernels",
        ]
        if capture_folder.is_dir() and any(capture_folder.iterdir()):
            cmd.extend(
                [
                    "--capture_folder",
                    self._container_path(capture_folder, workspace),
                ]
            )
        else:
            logger.warning(
                "TraceLens capture folder missing or empty; "
                "running report without it: %s",
                capture_folder,
            )
        if gpu_arch_platform:
            cmd.extend(["--gpu_arch_platform", gpu_arch_platform])

        logger.info(
            "Running TraceLens inference perf report in container %s (%s): %s",
            docker_image,
            stage,
            " ".join(cmd),
        )
        proc = self._run_container_command(docker_image, workspace, cmd)

        result: Dict[str, Any] = {
            "stage": stage,
            "trace_path": str(trace_path),
            "output_dir": str(out_dir),
            "files": [str(path) for path in sorted(out_dir.glob("*.csv"))],
        }
        if proc.returncode != 0:
            result["error"] = (
                f"TraceLens inference perf report failed for {stage} in container "
                f"{docker_image}: {proc.stderr or proc.stdout}"
            )
        return result

    def _run_container_command(
        self,
        docker_image: str,
        workspace: Path,
        cmd: Sequence[str],
    ) -> subprocess.CompletedProcess[str]:
        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{workspace}:/workspace",
            "-w",
            "/workspace",
        ]
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            docker_cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        docker_cmd.extend(["-e", "HOME=/tmp", "-e", "PYTHONUNBUFFERED=1"])
        if self.tl_extension:
            docker_cmd.extend(["-e", f"TL_EXTENSION={self.tl_extension}"])
        docker_cmd.extend(
            ["--entrypoint", "/bin/bash", docker_image, "-lc", shlex.join(cmd)]
        )

        logger.debug("Running TraceLens container command: %s", " ".join(docker_cmd))
        try:
            return subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=self.tl_config.cli_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return subprocess.CompletedProcess(
                docker_cmd,
                124,
                stdout=exc.stdout or "",
                stderr=(
                    exc.stderr
                    or f"TraceLens container command timed out after "
                    f"{self.tl_config.cli_timeout_seconds}s"
                ),
            )
        except FileNotFoundError as exc:
            return subprocess.CompletedProcess(
                docker_cmd,
                127,
                stdout="",
                stderr=f"Docker command not found: {exc}",
            )

    def _base_analysis_result(self, torch_trace_dir: Path) -> Dict[str, Any]:
        return {
            "enabled": True,
            "analysis_mode": "inference",
            "analysis_stages": self.tl_config.analysis_stages,
            "tl_extension": self.tl_extension,
            "trace_dir": str(torch_trace_dir),
            "output_files": [],
            "stage_results": {},
            "errors": [],
            "warnings": [],
        }

    def _container_path(self, path: Path, workspace: Path) -> str:
        path = path.resolve()
        workspace = workspace.resolve()
        try:
            rel = path.relative_to(workspace)
        except ValueError as exc:
            raise ValueError(
                f"TraceLens container postprocess path must be inside workspace: "
                f"{path} (workspace={workspace})"
            ) from exc
        return "/workspace" if str(rel) == "." else f"/workspace/{rel.as_posix()}"

    @staticmethod
    def _rewrite_container_split_paths(execution_csv: Path, workspace: Path) -> None:
        if not execution_csv.exists():
            return
        with execution_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = reader.fieldnames
        if not fieldnames:
            return

        changed = False
        for row in rows:
            output_path = row.get("output_path", "")
            if output_path == "/workspace":
                row["output_path"] = str(workspace)
                changed = True
            elif output_path.startswith("/workspace/"):
                rel = output_path[len("/workspace/"):]
                row["output_path"] = str(workspace / rel)
                changed = True

        if not changed:
            return
        with execution_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _subprocess_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        if self.tl_extension:
            env["TL_EXTENSION"] = self.tl_extension
        return env


def is_tracelens_inference_enabled(config: BenchmarkConfig) -> bool:
    """Return True when benchmark config requests TraceLens inference mode."""
    tl_config = config.profiler.tracelens
    return bool(tl_config.enabled and tl_config.is_inference_mode)


__all__ = [
    "CLI_INFERENCE_REPORT",
    "CLI_SPLIT_INFERENCE_TRACE",
    "InferencePhasePick",
    "SGLANG_PROFILE_CUDA_GRAPH_FLAG",
    "SGLANG_SHAPE_DISCOVERY_FLAG",
    "TraceLensInferencePipeline",
    "append_flag_value_args",
    "compute_steady_state_iters",
    "host_sglang_supports_shape_discovery",
    "is_tracelens_inference_enabled",
    "is_tracelens_patched_sglang_image",
    "resolve_tl_extension",
    "trace_arch_platform_from_runner",
]
