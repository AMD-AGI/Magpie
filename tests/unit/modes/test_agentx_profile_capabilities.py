"""Capability probes use the selected runtime without starting a GPU server."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from Magpie.modes.benchmark import agentx_profile_capabilities as capabilities


def _module(root, name, text):
    path = root.joinpath(*name.split(".")).with_suffix(".py")
    path.parent.mkdir(parents=True, exist_ok=True)
    for parent in path.parents:
        if parent == root:
            break
        (parent / "__init__.py").touch()
    path.write_text(text)


def _environment(root):
    return {**os.environ, "PYTHONPATH": str(root), "MAGPIE_PROBE_OVERLAY": "selected"}


def _probe(root, framework, argv=None, **kwargs):
    return capabilities.probe_profile_capabilities(
        framework,
        argv or [sys.executable, "-S", "-m", framework],
        _environment(root),
        root,
        kwargs.get("timeout", 5),
    )


def _sglang(root, annotation="roofline_annotations", shape=True, graph=True):
    _module(
        root,
        "sglang.srt.server_args",
        (
            "from dataclasses import dataclass\n"
            "@dataclass\nclass ServerArgs:\n"
            + (
                "    enable_profile_cuda_graph: bool = False\n"
                "    enable_shape_discovery_for_cuda_graph_profile: bool = False\n"
                if graph
                else "    unrelated: bool = False\n"
            )
        ),
    )
    _module(
        root,
        "sglang.srt.managers.io_struct",
        (
            "import os\n"
            "assert os.environ['MAGPIE_PROBE_OVERLAY'] == 'selected'\n"
            "print('framework import diagnostic')\n"
            "class RequestBase:\n"
            f"    {annotation}: bool = False\n"
            + ("    shape_discovery: bool = False\n" if shape else "")
            + "class ProfileReq(RequestBase):\n    pass\n"
            "class ProfileReqInput(RequestBase):\n    pass\n"
        ),
    )


def _vllm(
    root, capture="capture_torch_profiler_dir", style="annotations", exported=True
):
    names = ["detailed_trace_annotation", capture]
    if style == "annotations":
        body = "\n".join(f"    {name}: bool = False" for name in names)
    elif style == "dataclass":
        body = "\n".join(f"    {name}: bool = False" for name in names)
    else:
        body = f"    {style} = {dict.fromkeys(names)!r}"
    text = (
        "from dataclasses import dataclass\n@dataclass\n"
        if style == "dataclass"
        else ""
    ) + f"class ProfilerConfig:\n{body}\n"
    if exported:
        _module(root, "vllm.config", text)
    else:
        _module(root, "vllm.config.profiler", text)


@pytest.mark.parametrize("annotation", ["roofline_annotations", "detailed_annotations"])
def test_sglang_request_variants_and_runtime_environment(tmp_path, annotation):
    _sglang(tmp_path, annotation)
    assert _probe(tmp_path, "sglang") == {
        "annotation_field": annotation,
        "shape_discovery": True,
        "graph_capture": True,
        "graph_shape_discovery": True,
    }


def test_sglang_allows_eager_runtime_without_graph_capabilities(tmp_path):
    _sglang(tmp_path, graph=False)
    result = _probe(tmp_path, "sglang")
    assert result["graph_capture"] is False
    assert result["graph_shape_discovery"] is False


@pytest.mark.parametrize(
    "capture", ["capture_torch_profiler_dir", "capture_torch_profiler"]
)
@pytest.mark.parametrize(
    "style", ["annotations", "dataclass", "model_fields", "__fields__"]
)
def test_vllm_config_field_variants(tmp_path, capture, style):
    _vllm(tmp_path, capture, style)
    assert _probe(tmp_path, "vllm") == {
        "annotation_field": "detailed_trace_annotation",
        "capture_field": capture,
    }


def test_vllm_profiler_submodule_and_preferred_capture_field(tmp_path):
    _vllm(tmp_path, exported=False)
    path = tmp_path / "vllm/config/profiler.py"
    with path.open("a") as handle:
        handle.write("    capture_torch_profiler: bool = False\n")
    assert _probe(tmp_path, "vllm")["capture_field"] == "capture_torch_profiler_dir"


@pytest.mark.parametrize(
    "annotation,shape,missing",
    [
        ("other", True, "roofline_annotations/detailed_annotations"),
        ("detailed_annotations", False, "shape_discovery"),
    ],
)
def test_missing_sglang_capability_reports_interpreter(
    tmp_path, annotation, shape, missing
):
    _sglang(tmp_path, annotation, shape)
    with pytest.raises(RuntimeError) as error:
        _probe(tmp_path, "sglang")
    assert missing in str(error.value)
    assert sys.executable in str(error.value)


def test_sglang_requires_frontend_and_internal_request_support(tmp_path):
    _sglang(tmp_path)
    with (tmp_path / "sglang/srt/managers/io_struct.py").open("a") as handle:
        handle.write("class ProfileReqInput:\n    shape_discovery: bool = False\n")
    with pytest.raises(RuntimeError, match="roofline_annotations/detailed_annotations"):
        _probe(tmp_path, "sglang")


@pytest.mark.parametrize(
    "field", ["detailed_trace_annotation", "capture_torch_profiler_dir"]
)
def test_missing_vllm_capability_is_explicit(tmp_path, field):
    _vllm(tmp_path)
    path = tmp_path / "vllm/config.py"
    path.write_text(path.read_text().replace(field, "unrelated"))
    with pytest.raises(RuntimeError, match=field):
        _probe(tmp_path, "vllm")


@pytest.mark.parametrize("kind", ["absolute", "env", "env_split"])
def test_console_script_selects_its_shebang_python(tmp_path, kind):
    _vllm(tmp_path)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    interpreter = bindir / "python3"
    interpreter.symlink_to(sys.executable)
    script = bindir / "vllm"
    shebang = {
        "absolute": f"#!{interpreter} -S",
        "env": "#!/usr/bin/env python3",
        "env_split": "#!/usr/bin/env -S python3 -S",
    }[kind]
    script.write_text(shebang + "\nraise RuntimeError('server must not be executed')\n")
    script.chmod(0o755)
    env = {**_environment(tmp_path), "PATH": "bin"}
    result = capabilities.probe_profile_capabilities(
        "vllm", ["vllm", "serve", "model"], env, tmp_path, 5
    )
    assert result["capture_field"] == "capture_torch_profiler_dir"


def test_console_script_import_path_matches_server(tmp_path):
    script_root = tmp_path / "runtime"
    _vllm(script_root)
    script = script_root / "serve"
    script.write_text(
        f"#!{sys.executable} -S\nraise RuntimeError('must not execute')\n"
    )
    script.chmod(0o755)
    env = {**os.environ, "PYTHONPATH": ""}
    result = capabilities.probe_profile_capabilities(
        "vllm", [str(script)], env, tmp_path, 5
    )
    assert result["annotation_field"] == "detailed_trace_annotation"


def test_python_script_preserves_startup_options_and_import_path(tmp_path):
    script_root = tmp_path / "runtime"
    _vllm(script_root)
    script = script_root / "serve.py"
    script.write_text("raise RuntimeError('must not execute')\n")
    result = capabilities.probe_profile_capabilities(
        "vllm",
        [sys.executable, "-S", "-W", "ignore", str(script), "--model", "x"],
        {**os.environ, "PYTHONPATH": ""},
        tmp_path,
        5,
    )
    assert result["capture_field"] == "capture_torch_profiler_dir"


def test_missing_shebang_interpreter_never_falls_back_to_host(tmp_path):
    _vllm(tmp_path)
    script = tmp_path / "serve"
    script.write_text("#!/missing/runtime/python3\n")
    script.chmod(0o755)
    with pytest.raises(RuntimeError, match="/missing/runtime/python3"):
        _probe(tmp_path, "vllm", [str(script)])


def test_non_python_shebang_is_rejected(tmp_path):
    script = tmp_path / "serve"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o755)
    with pytest.raises(RuntimeError, match="Unsupported server Python shebang"):
        _probe(tmp_path, "vllm", [str(script)])


@pytest.mark.parametrize(
    "code,expected",
    [
        ("raise RuntimeError('framework import failed')\n", "framework import failed"),
        ("import time\ntime.sleep(10)\n", "timed out"),
    ],
)
def test_probe_import_failure_and_timeout(tmp_path, code, expected):
    _module(tmp_path, "vllm.config", code)
    with pytest.raises(RuntimeError) as error:
        _probe(tmp_path, "vllm", timeout=0.2)
    assert expected in str(error.value)
    assert sys.executable in str(error.value)


@pytest.mark.parametrize("output", ["", "MAGPIE_PROFILE_CAPABILITIES=not-json\n"])
def test_invalid_probe_response_is_rejected(tmp_path, monkeypatch, output):
    monkeypatch.setattr(
        capabilities.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, output, ""),
    )
    with pytest.raises(RuntimeError, match="using"):
        _probe(tmp_path, "vllm")


@pytest.mark.parametrize(
    "change",
    [
        {"unexpected": True},
        {"capture_field": "arbitrary"},
        {"annotation_field": "arbitrary"},
    ],
)
def test_receipt_capabilities_reject_unknown_or_unsupported_values(change):
    value = {
        "annotation_field": "detailed_trace_annotation",
        "capture_field": "capture_torch_profiler",
    }
    with pytest.raises(ValueError):
        capabilities.validate_profile_capabilities("vllm", {**value, **change})


def test_receipt_capabilities_require_boolean_flags_and_all_fields():
    value = {
        "annotation_field": "roofline_annotations",
        "shape_discovery": True,
        "graph_capture": True,
        "graph_shape_discovery": True,
    }
    for name in ("shape_discovery", "graph_capture", "graph_shape_discovery"):
        with pytest.raises(ValueError):
            capabilities.validate_profile_capabilities("sglang", {**value, name: 1})
        with pytest.raises(ValueError, match="Missing"):
            capabilities.validate_profile_capabilities(
                "sglang", {key: item for key, item in value.items() if key != name}
            )


def test_module_imports_with_python_no_site_packages():
    path = Path(capabilities.__file__)
    code = (
        "import importlib.util, sys; "
        "spec = importlib.util.spec_from_file_location('probe', sys.argv[1]); "
        "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "print(module.__name__)"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", code, str(path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "probe"
