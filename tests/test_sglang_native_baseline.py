"""Exercise the opt-in native baseline without a GPU or a SGLang installation."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "Magpie/scripts/benchmark"
HELPER = SCRIPTS / "sglang_native_benchmark.py"

# This stand-in is launched by the real helper, so the tests cover argument
# quoting, process status, persisted outputs, and phase ordering together.
FAKE_SERVING = r"""
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
def option(name):
    return args[args.index(name) + 1]

output = Path(option("--output-file"))
phase = output.stem
with Path(os.environ["NATIVE_INVOCATIONS"]).open("a") as stream:
    stream.write(json.dumps({"phase": phase, "argv": args}) + "\n")
print("native " + phase + " log", flush=True)
if os.environ.get("NATIVE_FAIL_PHASE") == phase:
    sys.exit(int(os.environ.get("NATIVE_FAIL_CODE", "19")))

count = int(option("--num-prompts"))
result = {
    "completed": count,
    "failed": 0,
    "errors": [""] * count,
    "total_input": count * int(option("--random-input-len")),
    "total_output": count * int(option("--random-output-len")),
    "total_throughput": 123.5,
    "request_throughput": 2.5,
    "output_throughput": 99.0,
    "mean_ttft_ms": 11.0,
    "mean_tpot_ms": 3.0,
    "mean_itl_ms": 4.0,
    "mean_e2e_latency_ms": 21.0,
    "duration": 5.0,
    "input_lens": [int(option("--random-input-len"))] * count,
    "output_lens": [int(option("--random-output-len"))] * count,
}
if os.environ.get("NATIVE_INVALID_PHASE") == phase:
    result.update(json.loads(os.environ["NATIVE_RESULT_OVERRIDE"]))
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result) + "\n")
if os.environ.get("NATIVE_COMPILE_PHASE") == phase:
    with Path(os.environ["SERVER_LOG"]).open("a") as stream:
        stream.write(os.environ["NATIVE_COMPILE_MESSAGE"] + "\n")
"""


@pytest.fixture
def native_env(tmp_path):
    modules = tmp_path / "fake modules"
    package = modules / "sglang" / "benchmark"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").touch()
    (package / "__init__.py").touch()
    (package / "serving.py").write_text(FAKE_SERVING)
    (package.parent / "launch_server.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['SERVER_CAPTURE']).write_text(json.dumps({\n"
        " 'argv': sys.argv[1:], 'env': {name: os.environ.get(name) for name in\n"
        " ('HSA_NO_SCRATCH_RECLAIM', 'SGLANG_USE_AITER', 'NCCL_MIN_NCHANNELS')}}))\n"
    )
    results = tmp_path / "result dir"
    results.mkdir()
    server_log = tmp_path / "server.log"
    server_log.write_text("Server ready\n")
    return {
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": str(modules),
        "PYTHONDONTWRITEBYTECODE": "1",
        "MODEL": "test/model with spaces",
        "CONC": "8",
        "ISL": "8192",
        "OSL": "1024",
        "RANDOM_RANGE_RATIO": "1",
        "RESULT_DIR": str(results),
        "RESULT_FILENAME": "baseline",
        "SERVER_LOG": str(server_log),
        "NATIVE_INVOCATIONS": str(tmp_path / "native-invocations.jsonl"),
    }


def run_helper(env, *args):
    return subprocess.run(
        [sys.executable, str(HELPER), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def invocations(env):
    path = Path(env["NATIVE_INVOCATIONS"])
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def option(args, name):
    return args[args.index(name) + 1]


def test_native_client_runs_full_warmup_then_measures_and_normalizes(native_env):
    result = run_helper(native_env, "client")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = invocations(native_env)
    assert [call["phase"] for call in calls] == ["warmup", "measurement"]
    assert [option(call["argv"], "--num-prompts") for call in calls] == ["16", "80"]
    expected = {
        "--backend": "sglang",
        "--base-url": "http://127.0.0.1:8888",
        "--model": native_env["MODEL"],
        "--tokenizer": native_env["MODEL"],
        "--dataset-name": "random-ids",
        "--random-input-len": "8192",
        "--random-output-len": "1024",
        "--random-range-ratio": "1.0",
        "--max-concurrency": "8",
        "--request-rate": "inf",
        "--warmup-requests": "8",
        "--seed": "42",
    }
    for call in calls:
        args = call["argv"]
        assert {name: option(args, name) for name in expected} == expected
        assert float(option(args, "--temperature")) == 0
        assert {"--tokenize-prompt", "--output-details", "--disable-tqdm"} <= set(args)
    results = Path(native_env["RESULT_DIR"])
    raw = results / "baseline.native"
    assert {path.name for path in raw.iterdir()} >= {
        "warmup.jsonl",
        "measurement.jsonl",
        "warmup.log",
        "measurement.log",
    }
    assert "native warmup log" in (raw / "warmup.log").read_text()
    assert "native measurement log" in (raw / "measurement.log").read_text()
    measured = json.loads((raw / "measurement.jsonl").read_text())
    normalized = json.loads((results / "baseline.json").read_text())
    assert normalized["completed"] == 80
    assert normalized["total_input_tokens"] == 80 * 8192
    assert normalized["total_output_tokens"] == 80 * 1024
    assert normalized["total_token_throughput"] == measured["total_throughput"]
    for name in ("mean_ttft_ms", "mean_tpot_ms", "mean_itl_ms", "mean_e2e_latency_ms"):
        assert normalized[name] == measured[name]
    assert "total_input_tokens" not in measured


def test_native_overrides_and_existing_json_suffix(native_env):
    native_env.update(
        NUM_PROMPTS="24",
        TOKENIZER="test/tokenizer with spaces",
        BENCHMARK_BASE_URL="http://remote.example:30000",
        RESULT_FILENAME="custom.json",
        SGLANG_NATIVE_FULL_WARMUP_MULTIPLIER="3",
        SGLANG_NATIVE_FULL_WARMUP_MINIMUM="40",
        SGLANG_BENCHMARK_SEED="11",
        SGLANG_BENCHMARK_TEMPERATURE="0.25",
    )
    result = run_helper(native_env, "client")
    assert result.returncode == 0, result.stderr
    calls = invocations(native_env)
    assert [option(call["argv"], "--num-prompts") for call in calls] == ["40", "24"]
    for call in calls:
        args = call["argv"]
        assert option(args, "--base-url") == native_env["BENCHMARK_BASE_URL"]
        assert option(args, "--tokenizer") == native_env["TOKENIZER"]
        assert option(args, "--seed") == "11"
        assert float(option(args, "--temperature")) == 0.25
    results = Path(native_env["RESULT_DIR"])
    assert (results / "custom.native" / "measurement.jsonl").exists()
    assert (results / "custom.json").exists()
    assert not (results / "custom.json.json").exists()


def test_native_port_override(native_env):
    native_env["PORT"] = "32100"
    result = run_helper(native_env, "client")
    assert result.returncode == 0, result.stderr
    assert all(
        option(call["argv"], "--base-url") == "http://127.0.0.1:32100"
        for call in invocations(native_env)
    )


@pytest.mark.parametrize("phase,expected_calls", [("warmup", 1), ("measurement", 2)])
def test_native_subprocess_failure_stops_without_publishing_result(
    native_env, phase, expected_calls
):
    native_env.update(NATIVE_FAIL_PHASE=phase, NATIVE_FAIL_CODE="19")
    result = run_helper(native_env, "client")
    assert result.returncode == 19, result.stdout + result.stderr
    assert len(invocations(native_env)) == expected_calls
    results = Path(native_env["RESULT_DIR"])
    assert not (results / "baseline.json").exists()
    assert (results / "baseline.native" / f"{phase}.log").exists()


@pytest.mark.parametrize("phase", ["warmup", "measurement"])
@pytest.mark.parametrize(
    "override",
    [
        {"completed": 1},
        {"failed": 1},
        {"error": 1},
        {"errors": None},
        {"errors": ["request failed"]},
    ],
)
def test_native_incomplete_or_failed_requests_reject_result(
    native_env, phase, override
):
    native_env.update(
        NATIVE_INVALID_PHASE=phase, NATIVE_RESULT_OVERRIDE=json.dumps(override)
    )
    result = run_helper(native_env, "client")
    assert result.returncode != 0
    assert len(invocations(native_env)) == (1 if phase == "warmup" else 2)
    results = Path(native_env["RESULT_DIR"])
    assert not (results / "baseline.json").exists()
    assert (results / "baseline.native" / f"{phase}.jsonl").exists()


@pytest.mark.parametrize("phase,count", [("warmup", 16), ("measurement", 80)])
@pytest.mark.parametrize(
    "fault", ["short_output", "wrong_input", "input_total", "output_total"]
)
def test_successful_http_requests_must_match_requested_token_lengths(
    native_env, phase, count, fault
):
    overrides = {
        "short_output": {"output_lens": [1024] * (count - 1) + [17]},
        "wrong_input": {"input_lens": [8192] * (count - 1) + [8191]},
        "input_total": {"total_input": count * 8192 - 1},
        "output_total": {"total_output": count * 1024 - 1},
    }
    native_env.update(
        NATIVE_INVALID_PHASE=phase,
        NATIVE_RESULT_OVERRIDE=json.dumps(overrides[fault]),
    )
    result = run_helper(native_env, "client")
    assert result.returncode != 0
    assert len(invocations(native_env)) == (1 if phase == "warmup" else 2)
    results = Path(native_env["RESULT_DIR"])
    assert not (results / "baseline.json").exists()
    assert (results / "baseline.native" / f"{phase}.jsonl").exists()


@pytest.mark.parametrize(
    "message",
    [
        "Compiling Triton kernel",
        "[aiter] compiling module_moe",
        "start build [module_moe] under /tmp/aiter/module_moe",
        "\x1b[32mfinish build \x1b[0m[module_moe], cost 1.25s",
        "failed build [module_moe]",
        "[aiter] build [module_test] under /tmp/build failed !!!!!!",
        "start jit build [module_moe] under /tmp/aiter/module_moe",
    ],
)
def test_compilation_during_measurement_rejects_but_retains_raw_results(
    native_env, message
):
    native_env.update(
        NATIVE_COMPILE_PHASE="measurement",
        NATIVE_COMPILE_MESSAGE=message,
    )
    result = run_helper(native_env, "client")
    assert result.returncode != 0
    results = Path(native_env["RESULT_DIR"])
    assert not (results / "baseline.json").exists()
    assert (results / "baseline.native" / "measurement.jsonl").exists()
    assert len(invocations(native_env)) == 2


def test_benign_build_message_does_not_invalidate_measurement(native_env):
    native_env.update(
        NATIVE_COMPILE_PHASE="measurement",
        NATIVE_COMPILE_MESSAGE="finish build request batch, 8 requests ready",
    )
    result = run_helper(native_env, "client")
    assert result.returncode == 0, result.stderr
    assert (Path(native_env["RESULT_DIR"]) / "baseline.json").exists()


def test_compilation_before_measurement_is_allowed(native_env):
    Path(native_env["SERVER_LOG"]).write_text(
        "Compiling Triton kernel before benchmark\n"
    )
    native_env.update(
        NATIVE_COMPILE_PHASE="warmup",
        NATIVE_COMPILE_MESSAGE="[aiter] compiling module_moe during full warmup",
    )
    result = run_helper(native_env, "client")
    assert result.returncode == 0, result.stderr
    assert (Path(native_env["RESULT_DIR"]) / "baseline.json").exists()


def test_existing_native_output_cannot_be_mixed_with_a_retry(native_env):
    first = run_helper(native_env, "client")
    assert first.returncode == 0, first.stderr
    results = Path(native_env["RESULT_DIR"])
    original = (results / "baseline.json").read_bytes()
    second = run_helper(native_env, "client")
    assert second.returncode != 0
    assert len(invocations(native_env)) == 2
    assert (results / "baseline.json").read_bytes() == original


@pytest.mark.parametrize(
    "override",
    [
        {"RANDOM_RANGE_RATIO": "0.8"},
        {"RESULT_FILENAME": "../outside"},
        {"RESULT_FILENAME": "/tmp/outside"},
        {"SGLANG_NATIVE_FULL_WARMUP_MULTIPLIER": "-1"},
        {"SGLANG_NATIVE_FULL_WARMUP_MINIMUM": "-1"},
    ],
)
def test_invalid_native_settings_fail_before_running_client(native_env, override):
    native_env.update(override)
    result = run_helper(native_env, "client")
    assert result.returncode != 0
    assert invocations(native_env) == []


def test_server_environment_override_order_and_literal_values(native_env, tmp_path):
    sentinel = tmp_path / "must-not-exist"
    literal = f"$(touch '{sentinel}') ; `touch '{sentinel}'` $HOME\nquoted value"
    native_env.update(
        KEEP_ME="inherited",
        HSA_NO_SCRATCH_RECLAIM="1",
        REMOVE_ME="inherited",
        RESTORE_ME="inherited",
        SGLANG_UNSET_ENVS="HSA_NO_SCRATCH_RECLAIM\n RESTORE_ME",
        SGLANG_SERVER_ENVS_JSON=json.dumps(
            {"REMOVE_ME": None, "RESTORE_ME": "restored", "LITERAL": literal}
        ),
    )
    keys = ["KEEP_ME", "HSA_NO_SCRATCH_RECLAIM", "REMOVE_ME", "RESTORE_ME", "LITERAL"]
    code = (
        "import json,os,sys; print(json.dumps({'env':{k:os.environ.get(k) for k in "
        + repr(keys)
        + "},'argv':sys.argv[1:]}))"
    )
    result = run_helper(
        native_env,
        "server",
        sys.executable,
        "-c",
        code,
        "argument with spaces",
        "$(literal)",
    )
    assert result.returncode == 0, result.stderr
    captured = json.loads(result.stdout)
    assert captured["env"] == {
        "KEEP_ME": "inherited",
        "HSA_NO_SCRATCH_RECLAIM": None,
        "REMOVE_ME": None,
        "RESTORE_ME": "restored",
        "LITERAL": literal,
    }
    assert captured["argv"] == ["argument with spaces", "$(literal)"]
    assert not sentinel.exists()


@pytest.mark.parametrize(
    "value", ["[]", '{"BAD": 1}', '{"BAD": true}', '{"BAD": []}', "invalid json"]
)
def test_server_rejects_invalid_runtime_environment(native_env, tmp_path, value):
    sentinel = tmp_path / "server-was-started"
    native_env["SGLANG_SERVER_ENVS_JSON"] = value
    result = run_helper(
        native_env,
        "server",
        sys.executable,
        "-c",
        "from pathlib import Path; Path(" + repr(str(sentinel)) + ").touch()",
    )
    assert result.returncode != 0
    assert not sentinel.exists()


def launcher_fixture(tmp_path, native_env):
    scripts = tmp_path / "copied scripts"
    scripts.mkdir()
    for name in ("sglang_mi355x.sh", "sglang_native_benchmark.py"):
        shutil.copy2(SCRIPTS / name, scripts / name)
    (scripts / "benchmark_lib.sh").write_text(
        "check_env_vars() { :; }\n"
        'rocm-smi() { printf "MEC %s\\n" "${TEST_MEC_VERSION:-999}"; }\n'
        "hf() { :; }\n"
        'setsid() { "$@"; }\n'
        'wait_for_server_ready() { wait "$SERVER_PID"; }\n'
        'magpie_run_benchmark_serving() { printf "%s\\n" "$@" > "$GENERIC_CAPTURE"; }\n'
        'run_benchmark_serving() { magpie_run_benchmark_serving "$@"; }\n'
    )
    (scripts / "server_cleanup.sh").touch()
    (scripts / "magpie_bench_remote_compat.sh").write_text(
        'magpie_run_benchmark_serving_remote_direct() { printf "remote\\n" > "$GENERIC_CAPTURE"; }\n'
    )
    native_env.update(
        MAGPIE_RUN_PHASE="client",
        RUN_EVAL="false",
        GENERIC_CAPTURE=str(tmp_path / "generic-argv"),
    )
    return scripts / "sglang_mi355x.sh"


@pytest.mark.parametrize("remote", [False, True])
def test_generic_client_remains_default(tmp_path, native_env, remote):
    launcher = launcher_fixture(tmp_path, native_env)
    if remote:
        native_env["BENCHMARK_BASE_URL"] = "http://remote.example:30000"
    result = subprocess.run(
        ["bash", str(launcher)],
        env=native_env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(native_env["GENERIC_CAPTURE"]).exists()
    assert invocations(native_env) == []
    if not remote:
        args = Path(native_env["GENERIC_CAPTURE"]).read_text().splitlines()
        assert option(args, "--backend") == "vllm"
        assert option(args, "--num-prompts") == "80"


def test_launcher_selects_native_client(tmp_path, native_env):
    launcher = launcher_fixture(tmp_path, native_env)
    native_env["SGLANG_BENCHMARK_CLIENT"] = "sglang-native"
    result = subprocess.run(
        ["bash", str(launcher)],
        env=native_env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(invocations(native_env)) == 2
    assert not Path(native_env["GENERIC_CAPTURE"]).exists()


def test_launcher_rejects_profiling_native_client(tmp_path, native_env):
    launcher = launcher_fixture(tmp_path, native_env)
    native_env.update(SGLANG_BENCHMARK_CLIENT="sglang-native", PROFILE="1")
    result = subprocess.run(
        ["bash", str(launcher)],
        env=native_env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode != 0
    assert invocations(native_env) == []
    assert not Path(native_env["GENERIC_CAPTURE"]).exists()


def test_server_overrides_apply_after_firmware_and_aiter_defaults(tmp_path, native_env):
    launcher = launcher_fixture(tmp_path, native_env)
    capture = tmp_path / "server-capture.json"
    native_env.update(
        MAGPIE_RUN_PHASE="server",
        MAGPIE_SERVER_PID_FILE=str(tmp_path / "server.pid"),
        TEST_MEC_VERSION="100",
        TP="8",
        SERVER_CAPTURE=str(capture),
        NCCL_MIN_NCHANNELS="8",
        SGLANG_UNSET_ENVS="HSA_NO_SCRATCH_RECLAIM NCCL_MIN_NCHANNELS",
        SGLANG_SERVER_ENVS_JSON=json.dumps({"SGLANG_USE_AITER": "0"}),
    )
    result = subprocess.run(
        ["bash", str(launcher)],
        env=native_env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(capture.read_text())
    assert observed["env"] == {
        "HSA_NO_SCRATCH_RECLAIM": None,
        "SGLANG_USE_AITER": "0",
        "NCCL_MIN_NCHANNELS": None,
    }
    assert "--tensor-parallel-size=8" in observed["argv"]
    assert invocations(native_env) == []


def test_benchmark_preparation_copies_a_runnable_native_helper(tmp_path, native_env):
    from types import SimpleNamespace

    from Magpie.modes.benchmark.benchmarker import BenchmarkMode

    checkout = tmp_path / "InferenceX checkout"
    mode = object.__new__(BenchmarkMode)
    mode.config = SimpleNamespace(inferencex_path=str(checkout))
    mode._prepare_benchmark_scripts()
    copied = checkout / "benchmarks" / HELPER.name
    assert copied.read_bytes() == HELPER.read_bytes()
    result = subprocess.run(
        [sys.executable, str(copied), "client"],
        env=native_env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(invocations(native_env)) == 2
