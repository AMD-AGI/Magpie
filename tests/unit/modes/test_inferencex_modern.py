"""Modern single-node client boundaries, including externally owned serving."""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import pytest

from Magpie.modes.benchmark import agentx_runtime as runtime
from Magpie.modes.benchmark.agentx import ensure_agentx_dependencies
from Magpie.modes.benchmark.agentx_recipe import _server_environment
from Magpie.modes.benchmark.benchmarker import BenchmarkMode
from Magpie.modes.benchmark.config import BenchmarkConfig
from Magpie.modes.benchmark.inferencex import (
    InferenceXManager,
    inferencex_provenance,
    resolve_inferencex_root,
)

ROOT = Path(__file__).parents[3]
SCRIPTS = ROOT / "Magpie/scripts/benchmark"


def test_external_model_alias_and_tokenizer_reach_client(tmp_path):
    config = BenchmarkConfig(
        model="original/model",
        framework="sglang",
        envs={"SERVED_MODEL_NAME": "endpoint-alias", "TOKENIZER": "/local/tokenizer"},
    )
    env = runtime._client_environment(
        {"client_env": config.get_env_vars()},
        {"env": _server_environment(config), "port": 8888},
        tmp_path,
        tmp_path,
    )
    assert env["SERVED_MODEL_NAME"] == "endpoint-alias"
    assert env["TOKENIZER"] == "/local/tokenizer"


@pytest.fixture
def modern(tmp_path):
    root = tmp_path / "Inference X/inferencex-e2e"
    for name in (
        "pyproject.toml",
        "infx/bench/__main__.py",
        "infx/bench/fixed_seq.py",
        "infx/bench/agentic/run.py",
        "infx/bench_serving/benchmark_serving.py",
        "infx/results/agentic/process_agentic_result.py",
        "infx/results/agentic/validate_agentic_result.py",
    ):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
    (root / "infx/bench/agentic/requirements.txt").write_text(
        "agentx-harness==1.0.14\n"
    )
    return root


def test_modern_checkout_resolves_without_shell_library_or_submodule(modern):
    before = {p: p.read_bytes() for p in modern.rglob("*") if p.is_file()}
    assert resolve_inferencex_root(str(modern.parent)) == str(modern)
    assert InferenceXManager().ensure_available(str(modern)) == str(modern)
    ensure_agentx_dependencies(str(modern))
    assert inferencex_provenance(str(modern))["layout"] == "modern"
    assert before == {p: p.read_bytes() for p in modern.rglob("*") if p.is_file()}


def test_modern_generic_scripts_execute_from_magpie_without_copying(modern, tmp_path):
    mode = BenchmarkMode(
        BenchmarkConfig(
            model="test/model", framework="sglang", inferencex_path=str(modern)
        ),
        output_dir=str(tmp_path / "out"),
    )
    mode._prepare_benchmark_scripts()
    script = mode._get_benchmark_script("mi355x")
    assert Path(script) == SCRIPTS / "sglang_mi355x.sh"
    assert not (modern / "benchmarks").exists()


def test_modern_runner_missing_inputs_stop_before_launch():
    completed = subprocess.run(
        [
            "bash",
            "-c",
            (
                f'source "{SCRIPTS}/magpie_bench_common.sh"; '
                "unset MODEL; check_env_vars MODEL; echo server-would-start"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "MODEL" in completed.stderr
    assert "server-would-start" not in completed.stdout


def test_modern_runner_stops_when_server_readiness_fails(modern, tmp_path):
    (modern / "infx/bench/__main__.py").write_text("raise SystemExit(9)\n")
    completed = subprocess.run(
        [
            "bash",
            "-c",
            (
                f'source "{SCRIPTS}/magpie_bench_common.sh"; '
                f'source "{SCRIPTS}/magpie_bench_remote_compat.sh"; '
                "wait_for_server_ready --port 12345; echo client-would-start"
            ),
        ],
        env={**os.environ, "MAGPIE_INFERENCEX_ROOT": str(modern)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 9
    assert "client-would-start" not in completed.stdout


def test_modern_external_docker_command_does_not_expose_gpus(
    modern, tmp_path, monkeypatch
):
    from Magpie.modes.benchmark import benchmarker

    mode = BenchmarkMode(
        BenchmarkConfig(
            framework="sglang",
            model="test/model",
            inferencex_path=str(modern),
            envs={"BENCHMARK_BASE_URL": "http://server:8000"},
        )
    )
    mode._task_id = "client-test"

    def unexpected():
        pytest.fail("an external HTTP client must not detect GPUs")

    monkeypatch.setattr(benchmarker, "detect_gpu", unexpected)
    command = mode._build_docker_command("client-image", tmp_path, "mi355x")
    assert "--gpus" not in command and "--device=/dev/kfd" not in command
    assert f"{SCRIPTS}:/opt/magpie-benchmark:ro" in command
    assert "/opt/magpie-benchmark/sglang_mi355x.sh" in command[-1]


def test_explicit_ref_is_verified_without_changing_checkout(modern):
    subprocess.run(["git", "init", str(modern.parent)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(modern.parent), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(modern.parent),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )
    sha = subprocess.check_output(
        ["git", "-C", str(modern.parent), "rev-parse", "HEAD"], text=True
    ).strip()
    manager = InferenceXManager()
    assert manager.ensure_available(str(modern), ref=sha) == str(modern)
    with pytest.raises(RuntimeError, match="never changed"):
        manager.ensure_available(str(modern), ref="0" * 40)
    assert inferencex_provenance(str(modern))["modified"] is False
    (modern / "infx/bench/fixed_seq.py").write_text("# local edit\n")
    assert inferencex_provenance(str(modern))["modified"] is True
    assert (modern / "infx/bench/fixed_seq.py").read_text() == "# local edit\n"
    target = modern.parent.parent / "managed clone"
    cloned = InferenceXManager(
        repo_url=str(modern.parent), default_dir=str(target)
    ).ensure_available(ref=sha)
    assert Path(cloned) == target / "inferencex-e2e"
    assert inferencex_provenance(cloned)["commit"] == sha
    assert inferencex_provenance(cloned)["modified"] is False


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="The packaged InferenceX client requires Python 3.12 or newer",
)
def test_new_generic_adapter_preserves_profile_workload_and_client_exit(
    modern, tmp_path
):
    (modern / "transformers.py").write_text("__version__='5.22.1'\n")
    (modern / "infx/bench_serving/benchmark_serving.py").write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        "if __name__ == '__main__':\n"
        " Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
        " raise SystemExit(7)\n"
    )
    body = f"""source '{SCRIPTS}/magpie_bench_common.sh'
source '{SCRIPTS}/magpie_bench_remote_compat.sh'
magpie_run_benchmark_serving --model 'served model' --port 8000 --backend vllm \\
 --input-len 128 --output-len 64 --random-range-ratio 1 --num-prompts 40 \\
 --max-concurrency 4 --result-filename result --result-dir '{tmp_path}' --tokenizer 'tokenizer/model'
"""
    result = subprocess.run(
        ["bash", "-c", body],
        cwd=modern,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "MAGPIE_INFERENCEX_ROOT": str(modern),
            "MAGPIE_BENCHMARK_PYTHON": sys.executable,
            "PROFILE": "1",
            "CAPTURE": str(tmp_path / "capture.json"),
        },
    )
    assert result.returncode == 7, result.stderr
    argv = json.loads((tmp_path / "capture.json").read_text())
    for flag, value in (
        ("--num-prompts", "40"),
        ("--num-warmups", "8"),
        ("--model", "served model"),
        ("--tokenizer", "tokenizer/model"),
        ("--result-filename", "result.json"),
    ):
        assert argv[argv.index(flag) + 1] == value
    assert "--profile" in argv


@pytest.fixture
def endpoint():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "behavior,success", [("success", True), ("failure", False), ("timeout", False)]
)
def test_external_client_never_owns_server(
    modern, tmp_path, endpoint, behavior, success
):
    (modern / "infx/bench/agentic/run.py").write_text(
        "class Plan:\n @classmethod\n def from_env(cls, env):\n  assert env['AIPERF_SERVER_URL'].startswith('http')\n"
    )
    benchmarks = modern / "benchmarks"
    benchmarks.mkdir()
    client = tmp_path / "client.py"
    client.write_text(
        "import json, os, time, urllib.request\nfrom pathlib import Path\n"
        "assert urllib.request.urlopen(os.environ['AIPERF_SERVER_URL']+'/health').status == 200\n"
        "behavior=os.environ['TEST_BEHAVIOR']\n"
        "if behavior=='failure': raise SystemExit(9)\n"
        "if behavior=='timeout': time.sleep(30)\n"
        "p=Path(os.environ['RESULT_DIR'])/'aiperf_artifacts';p.mkdir()\n"
        "(p/'profile_export_aiperf.json').write_text(json.dumps({'was_cancelled':False}))\n"
    )
    (benchmarks / "srt_agentic.sh").write_text(f'exec "{sys.executable}" "{client}"\n')
    workspace = tmp_path / "workspace"
    python = workspace / ".agentx-runtime/venv/bin/python"
    # A real isolated interpreter with fixture package metadata exercises the
    # runtime-version check without downloading or running the AgentX harness.
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(python.parents[1])],
        check=True,
    )
    site = next((python.parents[1] / "lib").glob("python*/site-packages"))
    dist = site / "agentx_harness-1.0.14.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: agentx-harness\nVersion: 1.0.14\n"
    )
    spec = {
        "version": 1,
        "framework": "sglang",
        "port": 12345,
        "argv": ["this-server-must-never-start"],
        "env": {},
        "client_env": {},
        "setup_commands": [["this-setup-must-never-run"]],
    }
    env = {"BENCHMARK_BASE_URL": endpoint, "TEST_BEHAVIOR": behavior}
    config = SimpleNamespace(
        framework="sglang",
        model="test/model",
        run_mode="local",
        inferencex_path=str(modern),
        envs=env,
        get_env_vars=lambda: env,
        agentx=SimpleNamespace(
            resolved={"server-launch-spec": spec}, launch_overrides=None
        ),
        server_lifecycle=None,
        timeout_seconds=0.5 if behavior == "timeout" else 5,
        hf_cache_path=None,
        profiler=None,
    )
    result, _, _ = runtime.execute_agentx(config, workspace, "mi355x")
    assert result.success is success, result.errors
    assert result.publishable is False
    assert urlopen(endpoint + "/health").status == 200
    assert not (workspace / "agentx_server_launch.json").exists()


@pytest.mark.parametrize(
    "cancelled,reasons", [(True, []), (False, ["invalid_submission"])]
)
def test_modern_client_rejects_cancelled_or_invalid_replay(
    modern, tmp_path, cancelled, reasons
):
    artifacts = tmp_path / "aiperf_artifacts"
    artifacts.mkdir()
    (artifacts / "profile_export_aiperf.json").write_text(
        json.dumps(
            {
                "was_cancelled": cancelled,
                "metadata": {"submission_invalid_reasons": reasons},
            }
        )
    )
    with pytest.raises(RuntimeError):
        runtime._verify_modern_result(modern, tmp_path, 0)
