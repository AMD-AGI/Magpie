###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################
"""Inspect profiling support with the interpreter that will launch the server.

This module and its child probe use only the standard library. Importing the
framework's configuration classes does not instantiate a model or a GPU worker.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from typing import Any

_RESULT_PREFIX = "MAGPIE_PROFILE_CAPABILITIES="
_PYTHON_NAME = re.compile(r"(?:python|pypy)(?:\d+(?:\.\d+)*)?(?:\.exe)?$")
_PROBE_SCRIPT = r"""
import dataclasses
import importlib
import json
import sys

framework, script_directory = sys.argv[1:3]
if script_directory and not (getattr(sys.flags, "safe_path", False) or sys.flags.isolated):
    sys.path[0] = script_directory

def fields(cls):
    names = set()
    for base in cls.__mro__:
        names.update(getattr(base, "__annotations__", {}))
    if dataclasses.is_dataclass(cls):
        names.update(field.name for field in dataclasses.fields(cls))
    for attribute in ("model_fields", "__fields__", "__struct_fields__"):
        names.update(getattr(cls, attribute, {}) or {})
    return names

if framework == "sglang":
    server = importlib.import_module("sglang.srt.server_args")
    requests = importlib.import_module("sglang.srt.managers.io_struct")
    request_types = [getattr(requests, name) for name in
                     ("ProfileReqInput", "ProfileReq") if hasattr(requests, name)]
    if not request_types:
        raise RuntimeError("SGLang lacks ProfileReqInput and ProfileReq")
    request_fields = set.intersection(*(fields(cls) for cls in request_types))
    server_fields = fields(server.ServerArgs)
    value = {
        "annotation_field": next((name for name in
            ("roofline_annotations", "detailed_annotations")
            if name in request_fields), None),
        "shape_discovery": "shape_discovery" in request_fields,
        "graph_capture": "enable_profile_cuda_graph" in server_fields,
        "graph_shape_discovery":
            "enable_shape_discovery_for_cuda_graph_profile" in server_fields,
    }
else:
    config = importlib.import_module("vllm.config")
    cls = getattr(config, "ProfilerConfig", None)
    if cls is None:
        cls = importlib.import_module("vllm.config.profiler").ProfilerConfig
    names = fields(cls)
    value = {
        "annotation_field": ("detailed_trace_annotation"
                             if "detailed_trace_annotation" in names else None),
        "capture_field": next((name for name in
            ("capture_torch_profiler_dir", "capture_torch_profiler")
            if name in names), None),
    }
print("MAGPIE_PROFILE_CAPABILITIES=" + json.dumps(value, sort_keys=True))
"""


def validate_profile_capabilities(framework: str, value: Any) -> dict[str, Any]:
    """Validate and canonicalize capabilities persisted in launch evidence.

    Graph support is reported separately because an eager candidate need not
    support graph capture. The caller must require it before enabling graph flags.
    """
    if framework not in {"sglang", "vllm"}:
        raise ValueError("AgentX detailed profiling supports SGLang and vLLM")
    if not isinstance(value, dict):
        raise ValueError("AgentX profile capabilities must be an object")
    expected = (
        {
            "annotation_field",
            "shape_discovery",
            "graph_capture",
            "graph_shape_discovery",
        }
        if framework == "sglang"
        else {"annotation_field", "capture_field"}
    )
    if value.keys() - expected:
        raise ValueError(
            f"Unknown AgentX profile capability fields: {sorted(value.keys() - expected)}"
        )
    if expected - value.keys():
        raise ValueError(
            f"Missing AgentX profile capability fields: {sorted(expected - value.keys())}"
        )
    if framework == "sglang":
        if value["annotation_field"] not in (
            "roofline_annotations",
            "detailed_annotations",
        ):
            raise ValueError(
                "SGLang lacks profile request roofline_annotations/detailed_annotations"
            )
        if value["shape_discovery"] is not True:
            raise ValueError("SGLang lacks profile request shape_discovery")
        for name in ("graph_capture", "graph_shape_discovery"):
            if type(value[name]) is not bool:
                raise ValueError(f"SGLang profile capability {name} must be boolean")
        return {
            "annotation_field": value["annotation_field"],
            "shape_discovery": True,
            "graph_capture": value["graph_capture"],
            "graph_shape_discovery": value["graph_shape_discovery"],
        }
    if value["annotation_field"] != "detailed_trace_annotation":
        raise ValueError("vLLM ProfilerConfig lacks detailed_trace_annotation")
    if value["capture_field"] not in (
        "capture_torch_profiler_dir",
        "capture_torch_profiler",
    ):
        raise ValueError(
            "vLLM ProfilerConfig lacks capture_torch_profiler_dir/capture_torch_profiler"
        )
    return {
        "annotation_field": "detailed_trace_annotation",
        "capture_field": value["capture_field"],
    }


def _resolve_executable(executable: str, env: dict[str, str], cwd: Path) -> str:
    if os.sep in executable:
        path = Path(executable)
        path = path if path.is_absolute() else cwd / path
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"Server executable is unavailable: {path}")
        # Preserve virtualenv symlinks: resolving them can select another install.
        return str(path.absolute())
    search_path = os.pathsep.join(
        str(Path(part) if Path(part).is_absolute() else cwd / part)
        for part in env.get("PATH", os.defpath).split(os.pathsep)
    )
    resolved = shutil.which(executable, path=search_path)
    if resolved is None:
        raise ValueError(f"Server executable is unavailable on its PATH: {executable}")
    return resolved


def _python_prefix(
    argv: list[str], env: dict[str, str], cwd: Path
) -> tuple[list[str], str]:
    executable = _resolve_executable(argv[0], env, cwd)
    arguments = argv[1:]
    script_directory = ""
    if not _PYTHON_NAME.fullmatch(Path(executable).name):
        with open(executable, "rb") as handle:
            line = handle.readline(4096)
        if not line.startswith(b"#!"):
            raise ValueError(f"Server executable has no Python shebang: {executable}")
        try:
            shebang = shlex.split(line[2:].decode("utf-8").strip())
        except (UnicodeError, ValueError) as exc:
            raise ValueError(f"Invalid server Python shebang: {executable}") from exc
        if shebang and Path(shebang[0]).name == "env":
            shebang = shebang[1:]
            if shebang[:1] == ["-S"]:
                shebang = shebang[1:]
        if not shebang or not _PYTHON_NAME.fullmatch(Path(shebang[0]).name):
            raise ValueError(f"Unsupported server Python shebang: {executable}")
        script_directory = str(Path(executable).parent)
        executable = _resolve_executable(shebang[0], env, cwd)
        arguments = shebang[1:]

    prefix = [executable]
    index = 0
    while index < len(arguments):
        arg = arguments[index]
        if arg in {"-m", "-c"}:
            break
        if arg == "--" or not arg.startswith("-"):
            if arg == "--" and index + 1 >= len(arguments):
                raise ValueError(f"Missing Python script after --: {executable}")
            script = arguments[index + 1] if arg == "--" else arg
            if script != "-":
                path = Path(script)
                script_directory = str(
                    (path if path.is_absolute() else cwd / path).parent
                )
            break
        if arg in {"-W", "-X", "--check-hash-based-pycs"}:
            if index + 1 >= len(arguments):
                raise ValueError(f"Incomplete Python option {arg}: {executable}")
            prefix.extend(arguments[index : index + 2])
            index += 2
            continue
        if re.fullmatch(r"-[bBdEHiIOPqRsSuv]+", arg) or arg.startswith(("-W", "-X")):
            # -i would hang the probe after -c. Its import semantics are unchanged.
            retained = arg.replace("i", "") if not arg.startswith(("-W", "-X")) else arg
            if retained != "-":
                prefix.append(retained)
            index += 1
            continue
        raise ValueError(f"Unsupported server Python option {arg!r}: {executable}")
    return prefix, script_directory


def probe_profile_capabilities(
    framework: str,
    argv: list[str],
    env: dict[str, str],
    cwd: str | Path,
    timeout: float,
) -> dict[str, Any]:
    """Probe the selected server runtime; never substitute the Magpie interpreter."""
    if framework not in {"sglang", "vllm"}:
        raise ValueError("AgentX detailed profiling supports SGLang and vLLM")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(arg, str) or "\0" in arg for arg in argv)
        or not argv[0]
    ):
        raise ValueError("AgentX profile probe requires the resolved server argv")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("AgentX profile probe timeout must be positive and finite")
    location = Path(cwd).absolute()
    try:
        prefix, script_directory = _python_prefix(argv, env, location)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Cannot select profile probe interpreter for {argv[0]!r}: {exc}"
        ) from exc
    label = shlex.join(prefix)
    try:
        result = subprocess.run(
            [*prefix, "-c", _PROBE_SCRIPT, framework, script_directory],
            env=env,
            cwd=location,
            timeout=timeout,
            text=True,
            capture_output=True,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"AgentX {framework} profile capability probe timed out using {label}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"AgentX {framework} profile capability probe failed using {label}: {exc}"
        ) from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise RuntimeError(
            f"AgentX {framework} profile capability probe failed using {label}: {detail}"
        )
    lines = [
        line[len(_RESULT_PREFIX) :]
        for line in result.stdout.splitlines()
        if line.startswith(_RESULT_PREFIX)
    ]
    try:
        if len(lines) != 1:
            raise ValueError("probe did not return exactly one capability object")
        return validate_profile_capabilities(framework, json.loads(lines[0]))
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            f"AgentX {framework} detailed profiling unavailable using {label}: {exc}"
        ) from exc
