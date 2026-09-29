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
from pathlib import Path
from typing import Any


def profile_settings(config) -> dict[str, Any] | None:
    profiler = getattr(getattr(config, "profiler", None), "torch_profiler", None)
    if profiler is None or not profiler.enabled:
        return None
    return {
        "num_steps": profiler.num_steps,
        "capture_timeout_seconds": profiler.capture_timeout_seconds,
        "flush_timeout_seconds": profiler.flush_timeout_seconds,
    }


def profile_server_spec(
    spec: dict[str, Any], settings: dict[str, Any], workspace: Path, capture_id: str
) -> dict[str, Any]:
    """Bind framework instrumentation and its output directory into launch evidence."""
    if not isinstance(capture_id, str) or not re.fullmatch(r"[0-9a-f]{32}", capture_id):
        raise ValueError("AgentX profiler requires a unique capture id")
    if type(settings.get("num_steps")) is not int or settings["num_steps"] <= 0:
        raise ValueError("AgentX profiler num_steps must be a positive integer")
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
    instrumentation = {
        **settings,
        "capture_id": capture_id,
        "trace_dir": str(directory),
    }
    if any(
        arg.replace("_", "-").startswith("--profiler-config") for arg in derived["argv"]
    ):
        raise ValueError("AgentX profiler configuration must be owned by Magpie")
    if derived["framework"] == "vllm":
        derived["argv"] += [
            "--profiler-config",
            json.dumps(
                {
                    "profiler": "torch",
                    "torch_profiler_dir": str(directory),
                    "max_iterations": settings["num_steps"],
                    "ignore_frontend": True,
                    "torch_profiler_use_gzip": True,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        ]
    elif derived["framework"] == "sglang":
        derived.setdefault("env", {})["SGLANG_TORCH_PROFILER_DIR"] = str(directory)
    else:
        raise ValueError("AgentX torch profiling supports SGLang and vLLM")
    derived["torch_profiler"] = instrumentation
    return derived
