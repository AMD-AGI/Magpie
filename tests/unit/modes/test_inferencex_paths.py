"""Supported checkout layouts and client runtime boundaries."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from Magpie.modes.benchmark.config import BenchmarkConfig
from Magpie.modes.benchmark.inferencex import (
    resolve_benchmark_serving,
    resolve_inferencex_root,
)
from Magpie.modes.benchmark.tracelens_inference import TraceLensInferencePipeline

COMPAT = (
    Path(__file__).resolve().parents[3]
    / "Magpie/scripts/benchmark/magpie_bench_remote_compat.sh"
)
PROFILE_LIB = (
    'if [[ "${PROFILE:-}" == "1" ]]; then\n num_prompts="$max_concurrency"\nfi\n'
)
CAPTURE = """import json, os, sys
from pathlib import Path

def capture():
    Path(os.environ['CAPTURE']).write_text(json.dumps({
        'argv': sys.argv[1:], 'cwd': os.getcwd(), 'executable': sys.executable,
        'pythonpath': os.environ.get('PYTHONPATH'),
    }))
"""


def _layout(root, packaged):
    (root / "benchmarks").mkdir(parents=True)
    (root / "benchmarks/benchmark_lib.sh").write_text(PROFILE_LIB)
    client = root / ("infx" if packaged else "utils") / "bench_serving"
    client.mkdir(parents=True)
    if packaged:
        (root / "pyproject.toml").write_text('[project]\nname = "infx"\n')
        (client / "capture.py").write_text(CAPTURE)
        (client / "benchmark_serving.py").write_text(
            "from .capture import capture\nif __name__ == '__main__': capture()\n"
        )
    else:
        (client / "benchmark_serving.py").write_text(CAPTURE + "\ncapture()\n")
    return client / "benchmark_serving.py"


@pytest.mark.parametrize("packaged", [False, True])
def test_layout_resolution_preserves_existing_checkout(tmp_path, packaged):
    root = tmp_path / "checkout with spaces"
    project = root / "inferencex-e2e" if packaged else root
    client = _layout(project, packaged)
    before = {p: p.read_bytes() for p in project.rglob("*") if p.is_file()}
    assert resolve_inferencex_root(str(root)) == str(project.resolve())
    assert resolve_inferencex_root(str(project)) == str(project.resolve())
    assert resolve_benchmark_serving(str(root)) == client.resolve()
    assert before == {p: p.read_bytes() for p in project.rglob("*") if p.is_file()}


def test_layout_rejects_copied_scripts_and_ambiguous_roots(tmp_path):
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "benchmarks/benchmark_lib.sh").write_text("# copied scripts only")
    with pytest.raises(RuntimeError, match="supported InferenceX"):
        resolve_inferencex_root(str(tmp_path))
    (tmp_path / "benchmarks/benchmark_lib.sh").unlink()
    (tmp_path / "benchmarks").rmdir()
    _layout(tmp_path, False)
    _layout(tmp_path / "inferencex-e2e", True)
    with pytest.raises(RuntimeError, match="Expected one"):
        resolve_inferencex_root(str(tmp_path))


def _run_shell(tmp_path, root, body, **overrides):
    env = {
        **os.environ,
        "MAGPIE_INFERENCEX_ROOT": str(root),
        "MAGPIE_BENCHMARK_PYTHON": sys.executable,
        "CAPTURE": str(tmp_path / "captured.json"),
        "MODEL": "test/model",
        "CONC": "4",
        "ISL": "128",
        "OSL": "64",
        "RANDOM_RANGE_RATIO": "1",
        "RESULT_FILENAME": "result",
        "RESULT_DIR": str(tmp_path / "result dir"),
        "BENCHMARK_BASE_URL": "http://localhost:30000",
        **overrides,
    }
    return subprocess.run(
        ["bash", "-c", f"source {shlex.quote(str(COMPAT))}\n{body}"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


@pytest.mark.parametrize("packaged", [False, True])
def test_remote_client_supports_both_layouts_and_profile_workload(tmp_path, packaged):
    if packaged and sys.version_info < (3, 12):
        pytest.skip("packaged InferenceX client integration requires Python 3.12")
    repo = tmp_path / "Inference X"
    project = repo / "inferencex-e2e" if packaged else repo
    _layout(project, packaged)
    result = _run_shell(
        tmp_path,
        repo,
        'before="$PWD:$PATH"; magpie_run_benchmark_serving_remote_direct trust || exit $?; '
        '[[ "$before" == "$PWD:$PATH" ]]',
        PROFILE="1",
        NUM_PROMPTS="40",
        PYTHONPATH=str(tmp_path / "server-only-packages"),
    )
    assert result.returncode == 0, result.stderr
    captured = json.loads((tmp_path / "captured.json").read_text())
    assert captured["cwd"] == str(project.resolve())
    if packaged:
        assert captured["pythonpath"] == str(project.resolve())
    args = captured["argv"]
    assert args[args.index("--num-prompts") + 1] == "40"
    assert args[args.index("--max-concurrency") + 1] == "4"
    assert "--profile" in args and "--trust-remote-code" in args
    assert args[args.index("--base-url") + 1] == "http://localhost:30000"


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="isolated client integration uses Python 3.12"
)
def test_packaged_local_client_preserves_upstream_args_and_parent_environment(tmp_path):
    project = tmp_path / "repo/inferencex-e2e"
    _layout(project, True)
    result = _run_shell(
        tmp_path,
        project,
        """run_benchmark_serving() {
  env PYTHONPATH="$PWD" python3 -m infx.bench_serving.benchmark_serving "$@"
}
before="$PWD:$PATH:${PYTHONPATH:-}"
magpie_run_benchmark_serving --model 'a model' --num-prompts 17 || exit $?
[[ "$before" == "$PWD:$PATH:${PYTHONPATH:-}" ]]
""",
    )
    assert result.returncode == 0, result.stderr
    captured = json.loads((tmp_path / "captured.json").read_text())
    assert captured["argv"] == ["--model", "a model", "--num-prompts", "17"]
    assert captured["cwd"] == str(project.resolve())
    assert Path(captured["executable"]).resolve() == Path(sys.executable).resolve()


def test_legacy_local_client_uses_original_function_without_bootstrap(tmp_path):
    _layout(tmp_path / "repo", False)
    result = _run_shell(
        tmp_path,
        tmp_path / "repo",
        """magpie_infx_client_python() { echo "unexpected bootstrap" >&2; return 8; }
run_benchmark_serving() { printf '%s\\n' "$@"; return 7; }
magpie_run_benchmark_serving --model legacy""",
    )
    assert result.returncode == 7
    assert result.stdout.splitlines() == ["--model", "legacy"]
    assert "unexpected bootstrap" not in result.stderr


def test_legacy_local_client_preserves_upstream_shell_state(tmp_path):
    _layout(tmp_path / "repo", False)
    result = _run_shell(
        tmp_path,
        tmp_path / "repo",
        "run_benchmark_serving() { benchmark_finished=yes; }; "
        'magpie_run_benchmark_serving && [[ "$benchmark_finished" == yes ]]',
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="isolated client integration uses Python 3.12"
)
def test_client_bootstrap_installs_only_into_private_environment(tmp_path):
    project = tmp_path / "repo/inferencex-e2e"
    _layout(project, True)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    uv = bindir / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys\n"
        "with open(os.environ['UV_CALLS'], 'a') as f: f.write(repr(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1] == 'venv':\n"
        " p = pathlib.Path(sys.argv[-1]) / 'bin/python'\n"
        " p.parent.mkdir(parents=True)\n"
        f" p.write_text('#!/bin/bash\\nexec {shlex.quote(sys.executable)} \\\"$@\\\"\\n')\n"
        " p.chmod(0o755)\n"
    )
    uv.chmod(0o755)
    result = _run_shell(
        tmp_path,
        project,
        """before="$PATH"
first="$(magpie_infx_client_python "$MAGPIE_INFERENCEX_ROOT")" || exit $?
second="$(magpie_infx_client_python "$MAGPIE_INFERENCEX_ROOT")" || exit $?
[[ "$first" == "$second" && "$before" == "$PATH" ]]
""",
        MAGPIE_BENCHMARK_PYTHON="",
        XDG_CACHE_HOME=str(tmp_path / "client-cache"),
        PATH=f"{bindir}:{os.environ['PATH']}",
        UV_CALLS=str(tmp_path / "uv-calls"),
    )
    assert result.returncode == 0, result.stderr
    calls = (tmp_path / "uv-calls").read_text().splitlines()
    assert len(calls) == 2  # setup reused by the second benchmark
    assert "'venv', '--python', '3.12'" in calls[0]
    assert "'pip', 'install', '--python'" in calls[1]
    assert "venv-build." in calls[1]
    assert "vllm" not in calls[1] and "sglang" not in calls[1]


def test_explicit_incompatible_client_fails_without_mutating_server(tmp_path):
    _layout(tmp_path / "repo", True)
    result = _run_shell(
        tmp_path,
        tmp_path / "repo",
        "magpie_run_benchmark_serving_remote_direct",
        MAGPIE_BENCHMARK_PYTHON="false",
    )
    assert result.returncode != 0
    assert "Python >=3.12" in result.stderr
    assert not (tmp_path / "captured.json").exists()


@pytest.mark.parametrize("packaged", [False, True])
def test_tracelens_patches_both_client_formats_and_restores(tmp_path, packaged):
    project = tmp_path / "repo/inferencex-e2e" if packaged else tmp_path / "repo"
    client = _layout(project, packaged)
    original = (
        'extra_body={\n "num_steps": 1,\n "merge_profiles": True,\n'
        ' "profile_by_stage": True,\n}\n'
        if packaged
        else 'extra_body={"num_steps": 1, "merge_profiles": True, "profile_by_stage": True}\n'
    )
    client.write_text(original)
    config = BenchmarkConfig.from_dict(
        {
            "framework": "sglang",
            "model": "demo",
            "inferencex_path": str(tmp_path / "repo"),
            "envs": {"CONC": 64, "OSL": 1024, "RANDOM_RANGE_RATIO": 1},
            "profiler": {"tracelens": {"enabled": True}},
        }
    )
    pipeline = TraceLensInferencePipeline(config)
    pipeline.prepare(tmp_path / "workspace")
    patched = client.read_text()
    assert '"num_steps": 256' in patched and '"start_step": 6016' in patched
    assert '"detailed_annotations": True' in patched
    pipeline.restore()
    assert client.read_text() == original
    assert (project / "benchmarks/benchmark_lib.sh").read_text() == PROFILE_LIB


def test_tracelens_unknown_profile_protocol_fails_and_restores_prior_patch(tmp_path):
    client = _layout(tmp_path / "repo", True)
    client.write_text('extra_body={"different_protocol": True}\n')
    config = BenchmarkConfig.from_dict(
        {
            "framework": "sglang",
            "model": "demo",
            "inferencex_path": str(tmp_path / "repo"),
            "profiler": {"tracelens": {"enabled": True}},
        }
    )
    with pytest.raises(RuntimeError, match="profile request anchor"):
        TraceLensInferencePipeline(config).prepare(tmp_path / "workspace")
    assert (tmp_path / "repo/benchmarks/benchmark_lib.sh").read_text() == PROFILE_LIB
