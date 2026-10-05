"""Generic serving must satisfy the pinned upstream eval entry point."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from Magpie.modes.benchmark.benchmarker import BenchmarkMode
from Magpie.modes.benchmark.config import BenchmarkConfig
from Magpie.utils.gpu import GPUVendor

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {"EVAL_ONLY": "false", "IS_AGENTIC": "0", "OPENAI_API_KEY": "EMPTY"}


def _mode(tmp_path, *, envs=None, agentx=None, custom=False, framework="sglang"):
    checkout = tmp_path / "inferencex"
    scripts = checkout / "benchmarks"
    scripts.mkdir(parents=True)
    script_name = "custom.sh" if custom else f"{framework}_mi355x.sh"
    (scripts / script_name).touch()
    config = BenchmarkConfig(
        framework=framework,
        model="test/model",
        precision="fp4",
        run_mode="local",
        inferencex_path=str(checkout),
        benchmark_script=script_name if custom else None,
        envs={
            "TP": 4,
            "CONC": 8,
            "ISL": 512,
            "OSL": 128,
            "RANDOM_RANGE_RATIO": 1,
            "RUN_EVAL": "true",
            **(envs or {}),
        },
        profiler={"torch_profiler": {"enabled": False}},
        agentx=agentx,
    )
    return BenchmarkMode(config, output_dir=str(tmp_path / "results"))


def _clean_eval_env(monkeypatch):
    for key in (*DEFAULTS, "BENCHMARK_BASE_URL", "MAGPIE_EVAL_TASKS"):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("framework", ["sglang", "vllm", "atom"])
@pytest.mark.parametrize("runtime", ["local", "docker"])
def test_builtin_frameworks_supply_upstream_eval_defaults(
    tmp_path, monkeypatch, framework, runtime
):
    _clean_eval_env(monkeypatch)
    mode = _mode(tmp_path, framework=framework)
    if runtime == "local":
        _, env = mode._build_local_command(tmp_path / "workspace", "mi355x")
    else:
        monkeypatch.setattr(
            "Magpie.modes.benchmark.benchmarker.detect_gpu",
            lambda: (GPUVendor.AMD, "gfx950"),
        )
        cmd = mode._build_docker_command("test-image", tmp_path / "workspace", "mi355x")
        env = dict(
            cmd[i + 1].split("=", 1) for i, arg in enumerate(cmd[:-1]) if arg == "-e"
        )
    assert {key: env[key] for key in DEFAULTS} == DEFAULTS


def test_generic_client_reaches_pinned_upstream_eval_without_workflow_env(
    tmp_path, monkeypatch
):
    """Only the model client is stubbed; upstream run_eval/run_lm_eval are real."""
    _clean_eval_env(monkeypatch)
    mode = _mode(tmp_path, envs={"EVAL_LIMIT": 8, "EVAL_MAX_MODEL_LEN": 8192})
    checkout = Path(mode.config.inferencex_path)
    scripts = checkout / "benchmarks"
    for name in (
        "sglang_mi355x.sh",
        "server_cleanup.sh",
        "magpie_bench_remote_compat.sh",
    ):
        shutil.copy2(ROOT / "Magpie/scripts/benchmark" / name, scripts / name)
    client = checkout / "utils/bench_serving/benchmark_serving.py"
    client.parent.mkdir(parents=True)
    client.touch()
    upstream = (
        ROOT / "tests/fixtures/inferencex_408c015/benchmarks/benchmark_lib_eval.sh"
    )
    (scripts / "benchmark_lib.sh").write_text(
        upstream.read_text()
        + "\nrun_benchmark_serving() { return 0; }\n"
        + "run_server_client() { python3 - \"$@\" <<'PY'\n"
        + "import json, os, sys\nfrom pathlib import Path\n"
        + "Path(os.environ['RESULT_DIR'], 'eval-invocation.json').write_text(json.dumps({\n"
        + " 'argv': sys.argv[1:], 'eval_only': os.environ['EVAL_ONLY'],\n"
        + " 'is_agentic': os.environ['IS_AGENTIC'], 'api_key': os.environ['OPENAI_API_KEY']}))\n"
        + "Path(os.environ['EVAL_RESULT_DIR'], 'results.json').write_text(json.dumps({\n"
        + " 'results': {'gsm8k': {'exact_match,strict-match': 1.0}},\n"
        + " 'n-samples': {'gsm8k': {'effective': 8}}}))\nPY\n}\n"
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cmd, env = mode._build_local_command(workspace, "mi355x", phase="client")
    env["INFERENCEX_REPO_ROOT"] = str(checkout)
    # The CPU boundary test does not install dependencies or contact a model.
    env["INFERENCEX_LM_EVAL_RUNTIME_READY"] = "true"
    result = subprocess.run(
        cmd, env=env, text=True, capture_output=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr
    invocation = json.loads((workspace / "eval-invocation.json").read_text())
    assert invocation["eval_only"] == "false"
    assert invocation["is_agentic"] == "0"
    assert invocation["api_key"] == "EMPTY"
    assert invocation["argv"][:4] == ["python3", "-m", "lm_eval", "--model"]
    assert invocation["argv"][-2:] == ["--limit", "8"]
    accuracy = json.loads((workspace / "accuracy_report.json").read_text())
    assert accuracy["status"] == "COMPLETED"
    assert accuracy["samples"] == 8
    assert not any(key in mode.config.envs for key in DEFAULTS)


@pytest.mark.parametrize("source", ["parent", "config"])
def test_local_explicit_eval_environment_is_preserved(
    tmp_path, monkeypatch, caplog, source
):
    _clean_eval_env(monkeypatch)
    values = {
        "EVAL_ONLY": "true",
        "IS_AGENTIC": "0",
        "OPENAI_API_KEY": "explicit-test-secret",
    }
    if source == "parent":
        for key, value in values.items():
            monkeypatch.setenv(key, value)
    mode = _mode(tmp_path, envs=values if source == "config" else None)
    cmd, env = mode._build_local_command(tmp_path / "workspace", "mi355x")
    assert {key: env[key] for key in values} == values
    assert "explicit-test-secret" not in " ".join(cmd)
    assert "explicit-test-secret" not in caplog.text


@pytest.mark.parametrize("explicit", [False, True])
def test_docker_defaults_use_only_configured_credentials(
    tmp_path, monkeypatch, explicit
):
    _clean_eval_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "host-secret-must-stay-out")
    monkeypatch.setattr(
        "Magpie.modes.benchmark.benchmarker.detect_gpu",
        lambda: (GPUVendor.AMD, "gfx950"),
    )
    values = {
        "EVAL_ONLY": "true",
        "IS_AGENTIC": "0",
        "OPENAI_API_KEY": "configured-test-key",
    }
    mode = _mode(tmp_path, envs=values if explicit else None)
    cmd = mode._build_docker_command("test-image", tmp_path / "workspace", "mi355x")
    env = dict(
        cmd[i + 1].split("=", 1) for i, arg in enumerate(cmd[:-1]) if arg == "-e"
    )
    assert {key: env[key] for key in DEFAULTS} == (values if explicit else DEFAULTS)
    assert "host-secret-must-stay-out" not in " ".join(cmd)
    if not explicit:
        assert not any(key in mode.config.envs for key in DEFAULTS)


@pytest.mark.parametrize("kind", ["agentx", "custom"])
def test_native_and_custom_launches_receive_no_generic_defaults(
    tmp_path, monkeypatch, kind
):
    _clean_eval_env(monkeypatch)
    mode = _mode(
        tmp_path, agentx=True if kind == "agentx" else None, custom=kind == "custom"
    )
    _, env = mode._build_local_command(tmp_path / "workspace", "mi355x")
    assert not any(key in env for key in DEFAULTS)
