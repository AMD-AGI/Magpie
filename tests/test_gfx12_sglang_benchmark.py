"""The gfx12 (R9700) SGLang runner: selection, launch defaults and overrides."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from Magpie.modes.benchmark.benchmarker import MAGPIE_BUILTIN_SCRIPTS, BenchmarkMode
from Magpie.modes.benchmark.config import BenchmarkConfig
from Magpie.modes.benchmark.image_selector import ImageSelector

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "Magpie" / "scripts" / "benchmark" / "sglang_gfx12.sh"
GFX1201_SGLANG_IMAGE = "rocm/sgl-dev:v0.5.15.post1-ubuntu24.04-py3.14-rocm10.0.0"


def _stage(tmp_path: Path) -> Path:
    script_dir = tmp_path / "benchmarks"
    script_dir.mkdir()
    staged = script_dir / SCRIPT.name
    shutil.copy2(SCRIPT, staged)
    (script_dir / "benchmark_lib.sh").write_text(
        """check_env_vars() {
  local name
  for name in "$@"; do
    [[ -n "${!name:-}" ]] || { echo "missing required variable: $name" >&2; return 1; }
  done
}
# The fake python3 writes CAPTURE_PATH, so its presence marks the server as up.
wait_for_server_ready() {
  local waited=0
  while [[ ! -s "${CAPTURE_PATH:-}" && "$waited" -lt 100 ]]; do
    sleep 0.05
    waited=$((waited + 1))
  done
}
""",
        encoding="utf-8",
    )
    (script_dir / "server_cleanup.sh").write_text("magpie_stop_benchmark_server_stack() { :; }\n", encoding="utf-8")
    (script_dir / "magpie_bench_remote_compat.sh").write_text(
        """magpie_run_benchmark_serving_remote_direct() { printf '%s\\0' "$@" > "$CLIENT_CAPTURE_PATH"; }
magpie_run_benchmark_serving() { printf '%s\\0' "$@" > "$CLIENT_CAPTURE_PATH"; }
magpie_run_eval_remote_direct() { : > "$EVAL_MARKER"; }
magpie_run_eval_persisted() { : > "$EVAL_MARKER"; }
""",
        encoding="utf-8",
    )
    return staged


def _fake_bin(tmp_path: Path) -> Path:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    python3 = bindir / "python3"
    python3.write_text(
        """#!/usr/bin/env bash
printf 'SGLANG_USE_AITER=%s\\nSGLANG_AITER_MLA_PERSIST=%s\\nHSA_OVERRIDE_GFX_VERSION=%s\\n' \\
  "${SGLANG_USE_AITER-<unset>}" "${SGLANG_AITER_MLA_PERSIST-<unset>}" \\
  "${HSA_OVERRIDE_GFX_VERSION-<unset>}" > "$CAPTURE_ENV_PATH"
printf '%s\\0' "$@" > "$CAPTURE_PATH"
""",
        encoding="utf-8",
    )
    python3.chmod(0o755)
    hf = bindir / "hf"
    hf.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    hf.chmod(0o755)
    return bindir


def _run(tmp_path: Path, phase: str, **overrides: str) -> subprocess.CompletedProcess[str]:
    staged = _stage(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{_fake_bin(tmp_path)}:{os.environ['PATH']}",
        "CAPTURE_PATH": str(tmp_path / "server_args.bin"),
        "CAPTURE_ENV_PATH": str(tmp_path / "server_env.txt"),
        "CLIENT_CAPTURE_PATH": str(tmp_path / "client_args.bin"),
        "EVAL_MARKER": str(tmp_path / "eval_called"),
        "MAGPIE_RUN_PHASE": phase,
        "MAGPIE_SERVER_PID_FILE": str(tmp_path / "server.pid"),
        "MODEL": "Qwen/Qwen3.5-9B",
        "TP": "1",
        "CONC": "15",
        "ISL": "512",
        "OSL": "128",
        "RANDOM_RANGE_RATIO": "1.0",
        "RESULT_FILENAME": "result",
        "RESULT_DIR": str(tmp_path),
        "RUN_EVAL": "false",
        "PROFILE": "0",
        "BENCHMARK_BASE_URL": "",
        "EXTRA_SGLANG_ARGS": "",
    }
    for key in ("SGLANG_USE_AITER", "SGLANG_AITER_MLA_PERSIST", "HSA_OVERRIDE_GFX_VERSION"):
        env.pop(key, None)
    env.update(overrides)
    return subprocess.run(["bash", str(staged)], env=env, capture_output=True, text=True, check=False, cwd=tmp_path)


def _captured_args(path: Path) -> list[str]:
    return [value.decode() for value in path.read_bytes().split(b"\0") if value]


def _server(tmp_path: Path, **overrides: str) -> tuple[list[str], dict[str, str]]:
    completed = _run(tmp_path, "server", **overrides)
    assert completed.returncode == 0, completed.stderr
    env_lines = (tmp_path / "server_env.txt").read_text(encoding="utf-8").splitlines()
    return _captured_args(tmp_path / "server_args.bin"), dict(line.split("=", 1) for line in env_lines)


def test_gfx1201_resolves_to_the_gfx12_sglang_runner_and_image(tmp_path: Path):
    inferencex = tmp_path / "InferenceX"
    (inferencex / "benchmarks").mkdir(parents=True)
    config = BenchmarkConfig(framework="sglang", model="demo", gpu_arch="gfx1201", inferencex_path=str(inferencex))
    mode = BenchmarkMode(config, output_dir=str(tmp_path / "results"))
    mode._prepare_benchmark_scripts()

    selector = ImageSelector()
    runner = selector.get_runner_type(config.gpu_arch)
    assert runner == "gfx12"
    assert selector.select_image("sglang", config.gpu_arch) == GFX1201_SGLANG_IMAGE
    assert mode._get_benchmark_script(runner) == "benchmarks/sglang_gfx12.sh"
    assert "sglang_gfx12.sh" in MAGPIE_BUILTIN_SCRIPTS


def test_server_applies_gfx12_defaults_and_opts_out_of_aiter(tmp_path: Path):
    args, env = _server(tmp_path, SGLANG_USE_AITER="1", SGLANG_AITER_MLA_PERSIST="1", HSA_OVERRIDE_GFX_VERSION="12.0.1")

    assert args == [
        "-m",
        "sglang.launch_server",
        "--model-path",
        "Qwen/Qwen3.5-9B",
        "--host",
        "0.0.0.0",
        "--port",
        "8888",
        "--trust-remote-code",
        "--tensor-parallel-size",
        "1",
        "--mem-fraction-static=0.9",
        "--disable-radix-cache",
        "--attention-backend=triton",
    ]
    assert env == {
        "SGLANG_USE_AITER": "0",
        "SGLANG_AITER_MLA_PERSIST": "<unset>",
        "HSA_OVERRIDE_GFX_VERSION": "<unset>",
    }


@pytest.mark.parametrize(
    ("extra", "absent"),
    [
        ("--mem-fraction-static 0.8", "--mem-fraction-static=0.9"),
        ("--mem-fraction-static=0.8", "--mem-fraction-static=0.9"),
        ("--attention-backend wave", "--attention-backend=triton"),
        ("--enable-hierarchical-cache", "--disable-radix-cache"),
    ],
)
def test_server_default_yields_to_an_explicit_flag(tmp_path: Path, extra: str, absent: str):
    args, _env = _server(tmp_path, EXTRA_SGLANG_ARGS=extra)

    assert absent not in args
    assert args[-len(extra.split()) :] == extra.split()


def test_server_keeps_multiline_and_glob_extra_args_literal(tmp_path: Path):
    (tmp_path / "star-match.txt").write_text("bait\n", encoding="utf-8")
    args, _env = _server(tmp_path, EXTRA_SGLANG_ARGS="--max-running-requests 64\n--served-model-name *\r\n")

    assert args[-4:] == ["--max-running-requests", "64", "--served-model-name", "*"]
    assert "star-match.txt" not in args


def test_remote_client_trusts_remote_code_and_skips_the_server(tmp_path: Path):
    completed = _run(tmp_path, "all", BENCHMARK_BASE_URL="http://127.0.0.1:8888", RUN_EVAL="True")

    assert completed.returncode == 0, completed.stderr
    assert not (tmp_path / "server_args.bin").exists()
    assert _captured_args(tmp_path / "client_args.bin") == ["trust"]
    assert (tmp_path / "eval_called").is_file()


def test_rejects_a_non_numeric_concurrency(tmp_path: Path):
    completed = _run(tmp_path, "client", CONC="15x")

    assert completed.returncode == 2
    assert "CONC must be an unsigned integer" in completed.stderr


def test_fails_fast_without_benchmark_lib(tmp_path: Path):
    staged = tmp_path / SCRIPT.name
    shutil.copy2(SCRIPT, staged)

    completed = subprocess.run(["bash", str(staged)], capture_output=True, text=True, check=False)

    assert completed.returncode == 3
    assert "benchmark_lib.sh" in completed.stderr
