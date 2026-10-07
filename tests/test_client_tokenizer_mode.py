"""Generic launchers forward optional tokenizer selection only to the client."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "Magpie/scripts/benchmark"
SCRIPT_NAMES = (
    "atom_mi300x.sh",
    "atom_mi355x.sh",
    "vllm_gfx12.sh",
    "vllm_mi300x.sh",
    "vllm_mi355x.sh",
    "vllm_radeon8060s.sh",
)
CAPTURE = """import json, os, sys
from pathlib import Path

def capture():
    Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))
"""
LIBRARY = r"""
check_env_vars() { :; }
hf() { :; }
rocm-smi() { printf 'MEC 999\n'; }
setsid() { "$TEST_PYTHON" "$SERVER_CAPTURE_SCRIPT" "$@"; }
wait_for_server_ready() { wait "$SERVER_PID"; }
run_benchmark_serving() {
    if [[ -f infx/bench_serving/benchmark_serving.py ]]; then
        env PYTHONPATH="$PWD" python3 -m infx.bench_serving.benchmark_serving "$@"
    else
        "$TEST_PYTHON" utils/bench_serving/benchmark_serving.py "$@"
    fi
}
"""


def _fixture(tmp_path, packaged):
    project = tmp_path / "InferenceX checkout"
    scripts = project / "benchmarks"
    shutil.copytree(SCRIPTS, scripts)
    (scripts / "benchmark_lib.sh").write_text(LIBRARY)
    client = project / ("infx" if packaged else "utils") / "bench_serving"
    client.mkdir(parents=True)
    if packaged:
        (project / "pyproject.toml").write_text('[project]\nname = "infx"\n')
        (project / "transformers.py").write_text('__version__ = "5.17.0"\n')
        (client / "capture.py").write_text(CAPTURE)
        (client / "benchmark_serving.py").write_text(
            "from .capture import capture\nif __name__ == '__main__': capture()\n"
        )
    else:
        (client / "benchmark_serving.py").write_text(CAPTURE + "\ncapture()\n")
    capture_script = tmp_path / "capture-server.py"
    capture_script.write_text(CAPTURE + "\ncapture()\n")
    results = tmp_path / "result dir"
    results.mkdir()
    env = {
        "PATH": os.environ["PATH"],
        "PYTHONDONTWRITEBYTECODE": "1",
        "TEST_PYTHON": sys.executable,
        "SERVER_CAPTURE_SCRIPT": str(capture_script),
        "MAGPIE_BENCHMARK_PYTHON": sys.executable,
        "MAGPIE_INFERENCEX_ROOT": str(project),
        "MAGPIE_RUN_PHASE": "client",
        "MAGPIE_SERVER_PID_FILE": str(tmp_path / "server.pid"),
        "CAPTURE": str(tmp_path / "captured.json"),
        "MODEL": "test/model with spaces",
        "TP": "2",
        "CONC": "4",
        "ISL": "128",
        "OSL": "64",
        "RANDOM_RANGE_RATIO": "1",
        "RESULT_FILENAME": "result",
        "RESULT_DIR": str(results),
        "SERVER_LOG": str(tmp_path / "server.log"),
        "RUN_EVAL": "false",
    }
    return project, env


def _run(project, script_name, env):
    result = subprocess.run(
        ["bash", str(project / "benchmarks" / script_name)],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(Path(env["CAPTURE"]).read_text())


@pytest.mark.parametrize("script_name", SCRIPT_NAMES)
@pytest.mark.parametrize("packaged", [False, True])
@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize(
    ("mode_env", "expected_mode"),
    [
        ({}, None),
        ({"MAGPIE_CLIENT_TOKENIZER_MODE": "deepseek_v4"}, "deepseek_v4"),
        ({"HYPERLOOM_CLIENT_TOKENIZER_MODE": "deepseek_v4"}, "deepseek_v4"),
        (
            {
                "MAGPIE_CLIENT_TOKENIZER_MODE": "auto",
                "HYPERLOOM_CLIENT_TOKENIZER_MODE": "deepseek_v4",
            },
            "auto",
        ),
        (
            {
                "MAGPIE_CLIENT_TOKENIZER_MODE": "",
                "HYPERLOOM_CLIENT_TOKENIZER_MODE": "deepseek_v4",
            },
            "deepseek_v4",
        ),
    ],
    ids=["default", "magpie", "hyperloom", "magpie-precedence", "empty-fallback"],
)
def test_generic_client_tokenizer_argv(
    tmp_path, script_name, packaged, remote, mode_env, expected_mode
):
    if packaged and sys.version_info < (3, 12):
        pytest.skip("packaged InferenceX requires Python 3.12")
    project, env = _fixture(tmp_path, packaged)
    env.update(mode_env)
    if remote:
        env["BENCHMARK_BASE_URL"] = "http://127.0.0.1:30000"
        expected = [
            "--model",
            env["MODEL"],
            "--backend",
            "vllm",
            "--base-url",
            env["BENCHMARK_BASE_URL"],
            "--endpoint",
            "/v1/completions",
            "--dataset-name",
            "random",
            "--random-input-len",
            "128",
            "--random-output-len",
            "64",
            "--random-range-ratio",
            "1",
            "--num-prompts",
            "40",
            "--max-concurrency",
            "4",
            "--request-rate",
            "inf",
            "--ignore-eos",
            "--save-result",
            "--num-warmups",
            "8",
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
            "--result-dir",
            env["RESULT_DIR"],
            "--result-filename",
            "result.json",
            "--trust-remote-code",
        ]
        if expected_mode:
            expected += ["--tokenizer-mode", expected_mode]
    else:
        expected = [
            "--model",
            env["MODEL"],
            "--port",
            "8888",
            "--backend",
            "vllm",
            "--input-len",
            "128",
            "--output-len",
            "64",
            "--random-range-ratio",
            "1",
            "--num-prompts",
            "40",
            "--max-concurrency",
            "4",
            "--result-filename",
            "result",
            "--result-dir",
            env["RESULT_DIR"] + "/",
        ]
        if expected_mode:
            expected += ["--tokenizer-mode", expected_mode]
        expected += ["--trust-remote-code"]
    assert _run(project, script_name, env) == expected
    assert not (tmp_path / "server.pid").exists()


@pytest.mark.parametrize(
    "script_name", (*SCRIPT_NAMES, "sglang_mi300x.sh", "sglang_mi355x.sh")
)
def test_tokenizer_selection_does_not_change_server_argv(tmp_path, script_name):
    project, env = _fixture(tmp_path, packaged=False)
    env["MAGPIE_RUN_PHASE"] = "server"
    original = _run(project, script_name, env)
    assert original[:2] == (
        ["python3", "-m"]
        if script_name.startswith(("atom_", "sglang_"))
        else ["vllm", "serve"]
    )
    env.update(
        MAGPIE_CLIENT_TOKENIZER_MODE="auto",
        HYPERLOOM_CLIENT_TOKENIZER_MODE="deepseek_v4",
        MAGPIE_TRUST_REMOTE_CODE="1",
    )
    assert _run(project, script_name, env) == original
    assert "--tokenizer-mode" not in original


def test_client_tokenizer_value_is_one_argument(tmp_path):
    project, env = _fixture(tmp_path, packaged=False)
    env["MAGPIE_CLIENT_TOKENIZER_MODE"] = "mode with spaces; false"
    args = _run(project, "vllm_mi300x.sh", env)
    assert (
        args[args.index("--tokenizer-mode") + 1] == env["MAGPIE_CLIENT_TOKENIZER_MODE"]
    )


@pytest.mark.parametrize("script_name", ["sglang_mi300x.sh", "sglang_mi355x.sh"])
@pytest.mark.parametrize("packaged", [False, True])
@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("trust_mode", [None, "0", "1", "true"])
def test_sglang_client_trust_preserves_defaults_and_explicit_opt_in(
    tmp_path, script_name, packaged, remote, trust_mode
):
    if packaged and sys.version_info < (3, 12):
        pytest.skip("packaged InferenceX requires Python 3.12")
    project, env = _fixture(tmp_path, packaged)
    if remote:
        env["BENCHMARK_BASE_URL"] = "http://127.0.0.1:30000"
    baseline = _run(project, script_name, env)
    if trust_mode is not None:
        env["MAGPIE_TRUST_REMOTE_CODE"] = trust_mode
    actual = _run(project, script_name, env)
    local_always_trust = script_name == "sglang_mi355x.sh" and not remote
    assert baseline.count("--trust-remote-code") == int(local_always_trust)
    assert actual.count("--trust-remote-code") == int(
        local_always_trust or trust_mode == "1"
    )
    assert [arg for arg in actual if arg != "--trust-remote-code"] == [
        arg for arg in baseline if arg != "--trust-remote-code"
    ]
    assert "--tokenizer-mode" not in actual
