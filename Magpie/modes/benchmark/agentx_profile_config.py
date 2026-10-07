###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################
"""Derive a diagnostic launch without mutating the accepted AgentX candidate."""

from __future__ import annotations

import copy
import json
import math
import re
from fractions import Fraction
from pathlib import Path
from typing import Any


def profile_settings(config) -> dict[str, Any] | None:
    profiler = getattr(getattr(config, "profiler", None), "torch_profiler", None)
    if profiler is None or not profiler.enabled:
        return None
    return {
        "num_steps": profiler.num_steps,
        "num_profiles": profiler.num_profiles,
        "interval_seconds": profiler.interval_seconds,
        "start_seconds": profiler.start_seconds,
        "detailed_annotations": profiler.detailed_annotations,
        "capture_timeout_seconds": profiler.capture_timeout_seconds,
        "flush_timeout_seconds": profiler.flush_timeout_seconds,
    }


def profile_plan(
    settings: dict[str, Any], client_env: dict[str, str]
) -> dict[str, Any]:
    """Bound capture starts by the replay window, without predicting step cost."""
    count = settings.get("num_profiles", 1)
    if type(count) is not int or count <= 0:
        raise ValueError("AgentX num_profiles must be a positive integer")
    interval = settings.get("interval_seconds", 200.0)
    start = settings.get("start_seconds", 0.0)
    raw_duration = (
        1200
        if client_env.get("AIPERF_EXPERIMENTAL_FAST") == "1"
        else client_env.get("DURATION", 3600)
    )
    try:
        duration = float(raw_duration)
        valid = (
            not isinstance(raw_duration, bool)
            and math.isfinite(duration)
            and duration > 0
            and type(interval) in (int, float)
            and math.isfinite(interval)
            and interval >= 0
            and type(start) in (int, float)
            and math.isfinite(start)
            and start >= 0
        )
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(
            "AgentX profiling needs a positive finite replay duration and nonnegative finite interval/start_seconds"
        )
    if start >= duration:
        raise ValueError(
            "AgentX profiler start_seconds must be less than the measurement duration"
        )
    maximum = None
    if interval > 0:
        # A start at exactly the end of measurement is too late. Decimal
        # rational arithmetic avoids overflow and ceil errors near multiples.
        remaining = Fraction(str(duration)) - Fraction(str(start))
        ratio = remaining / Fraction(str(interval))
        maximum = -(-ratio.numerator // ratio.denominator)
    return {
        "requested_profiles": count,
        "max_profiles": maximum,
        "planned_profiles": min(count, maximum) if maximum is not None else count,
        "measurement_duration_seconds": duration,
        "start_seconds": float(start),
    }


def profile_server_spec(
    spec: dict[str, Any],
    settings: dict[str, Any],
    workspace: Path,
    capture_id: str,
    *,
    capabilities: dict[str, Any] | None = None,
    launch_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind framework instrumentation and its output directory into launch evidence."""
    if not isinstance(capture_id, str) or not re.fullmatch(r"[0-9a-f]{32}", capture_id):
        raise ValueError("AgentX profiler requires a unique capture id")
    if type(settings.get("num_steps")) is not int or settings["num_steps"] <= 0:
        raise ValueError("AgentX profiler num_steps must be a positive integer")
    annotations = settings.get("detailed_annotations", False)
    if type(annotations) is not bool:
        raise ValueError("AgentX profiler detailed_annotations must be a boolean")
    if capabilities is not None:
        if not annotations:
            raise ValueError("Unexpected AgentX annotation capabilities")
        if __package__:
            from .agentx_profile_capabilities import validate_profile_capabilities
        else:
            from agentx_profile_capabilities import validate_profile_capabilities
        capabilities = validate_profile_capabilities(spec["framework"], capabilities)
        if __package__:
            from .agentx_launch import apply_launch_args
        else:
            from agentx_launch import apply_launch_args
        effective_args = apply_launch_args(
            spec["argv"], launch_overrides or {"version": 1}, spec["framework"]
        )
        effective_flags = {
            arg.split("=", 1)[0].replace("_", "-") for arg in effective_args
        }
        enforce_eager = False
        for arg in effective_args:
            name = arg.replace("_", "-")
            if name in {"--enforce-eager", "--no-enforce-eager"}:
                enforce_eager = name == "--enforce-eager"
    count = settings.get("num_profiles", 1)
    if type(count) is not int or count <= 0:
        raise ValueError("AgentX profiler num_profiles must be a positive integer")
    interval = settings.get("interval_seconds", 200.0)
    try:
        valid_interval = (
            type(interval) in (int, float) and math.isfinite(interval) and interval >= 0
        )
    except OverflowError:
        valid_interval = False
    if not valid_interval:
        raise ValueError(
            "AgentX profiler interval_seconds must be nonnegative and finite"
        )
    start = settings.get("start_seconds", 0.0)
    try:
        valid_start = (
            type(start) in (int, float) and math.isfinite(start) and start >= 0
        )
    except OverflowError:
        valid_start = False
    if not valid_start:
        raise ValueError("AgentX profiler start_seconds must be nonnegative and finite")
    for name in ("capture_timeout_seconds", "flush_timeout_seconds"):
        value = settings.get(name)
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"AgentX profiler {name} must be positive and finite")
    derived = copy.deepcopy(spec)
    directory = workspace.resolve() / "torch_trace" / capture_id
    output_directory = directory / "active" if count > 1 else directory
    instrumentation = {
        **settings,
        "capture_id": capture_id,
        "trace_dir": str(directory),
    }
    if capabilities is not None:
        instrumentation["capabilities"] = capabilities
    if any(
        arg.replace("_", "-").startswith("--profiler-config") for arg in derived["argv"]
    ):
        raise ValueError("AgentX profiler configuration must be owned by Magpie")
    if derived["framework"] == "vllm":
        options = {
            "profiler": "torch",
            "torch_profiler_dir": str(output_directory),
            "max_iterations": settings["num_steps"],
            "ignore_frontend": True,
            "torch_profiler_use_gzip": True,
        }
        if capabilities is not None:
            options.update(
                detailed_trace_annotation=True,
                torch_profiler_record_shapes=True,
                torch_profiler_with_stack=True,
            )
            if not enforce_eager:
                field = capabilities["capture_field"]
                options[field] = (
                    str(output_directory / "capture_traces")
                    if field == "capture_torch_profiler_dir"
                    else True
                )
        derived["argv"] += [
            "--profiler-config",
            json.dumps(
                options,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ]
    elif derived["framework"] == "sglang":
        derived.setdefault("env", {})["SGLANG_TORCH_PROFILER_DIR"] = str(
            output_directory
        )
        if capabilities is not None:
            derived["env"].update(
                SGLANG_PROFILE_WITH_STACK="True",
                SGLANG_PROFILE_RECORD_SHAPES="True",
                # Prefer the combined export on runtimes that support it. It
                # takes precedence over per-batch capture, whose fixed batch
                # list can overflow when one batch has multiple graph variants.
                # Keep the old switch for runtimes with only per-batch export.
                SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE="True",
                SGLANG_GRAPH_BATCH_CAPTURE="True",
            )
            if "--disable-cuda-graph" not in effective_flags:
                if not capabilities["graph_capture"]:
                    raise ValueError(
                        "SGLang detailed annotations with CUDA graphs require graph profiling support"
                    )
                for name, flag in (
                    ("graph_capture", "--enable-profile-cuda-graph"),
                    (
                        "graph_shape_discovery",
                        "--enable-shape-discovery-for-cuda-graph-profile",
                    ),
                ):
                    if capabilities[name] and flag not in effective_flags:
                        derived["argv"].append(flag)
    else:
        raise ValueError("AgentX torch profiling supports SGLang and vLLM")
    derived["torch_profiler"] = instrumentation
    return derived
