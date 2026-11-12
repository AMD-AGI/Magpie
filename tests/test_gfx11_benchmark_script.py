import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT_DIR = ROOT / "Magpie" / "scripts" / "benchmark"
SCRIPT = SCRIPT_DIR / "vllm_gfx11.sh"
UNSET = "<unset>"
CAPTURED_VARS = (
    "VLLM_ROCM_USE_AITER",
    "VLLM_ROCM_USE_AITER_RMSNORM",
    "FLASH_ATTENTION_TRITON_AMD_ENABLE",
    "HSA_OVERRIDE_GFX_VERSION",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
)
CAPTURE_ENV = "".join(
    f'printf "{name}=%s\\n" "${{{name}-{UNSET}}}" >> "$CAPTURE_ENV_PATH"\n'
    for name in CAPTURED_VARS
)


def _stage_script(tmp_path: Path) -> Path:
    script_dir = tmp_path / "benchmarks"
    script_dir.mkdir(exist_ok=True)
    staged_script = script_dir / SCRIPT.name
    shutil.copy2(SCRIPT, staged_script)
    (script_dir / "benchmark_lib.sh").write_text(
        f"""check_env_vars() {{
  local name
  for name in "$@"; do
    if [[ -z "${{!name:-}}" ]]; then
      echo "missing required variable: $name" >&2
      return 1
    fi
  done
}}

_capture_client() {{
  printf '%s\\0' "$@" > "$CLIENT_CAPTURE_PATH"
  if [[ -n "${{CAPTURE_ENV_PATH:-}}" ]]; then
    : > "$CAPTURE_ENV_PATH"
{CAPTURE_ENV}  fi
}}

run_benchmark_serving() {{
  _capture_client "$@"
}}

# The fake vllm writes CAPTURE_PATH, so its presence marks the server as up.
wait_for_server_ready() {{
  local waited=0
  while [[ ! -s "${{CAPTURE_PATH:-}}" && "$waited" -lt 100 ]]; do
    sleep 0.05
    waited=$((waited + 1))
  done
}}
""",
        encoding="utf-8",
    )
    (script_dir / "server_cleanup.sh").write_text(
        "magpie_stop_benchmark_server_stack() { :; }\n",
        encoding="utf-8",
    )
    (script_dir / "magpie_bench_remote_compat.sh").write_text(
        """magpie_run_eval_remote_direct() { echo remote > "$EVAL_MARKER"; }
magpie_run_eval_persisted() { printf '%s\\n' "$@" > "$EVAL_MARKER"; }
magpie_run_benchmark_serving_remote_direct() { _capture_client remote "$@"; }
magpie_run_benchmark_serving() { run_benchmark_serving "$@"; }
""",
        encoding="utf-8",
    )
    shutil.copy2(
        SCRIPT_DIR / "magpie_split_extra_vllm_args.sh",
        script_dir / "magpie_split_extra_vllm_args.sh",
    )
    return staged_script


def _fake_bin(tmp_path: Path) -> Path:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    fake_vllm = fake_bin / "vllm"
    fake_vllm.write_text(
        f"""#!/usr/bin/env bash
printf '%s\\0' "$@" > "$CAPTURE_PATH"
if [[ -n "${{CAPTURE_ENV_PATH:-}}" ]]; then
  : > "$CAPTURE_ENV_PATH"
{CAPTURE_ENV}fi
""",
        encoding="utf-8",
    )
    fake_vllm.chmod(0o755)
    fake_hf = fake_bin / "hf"
    fake_hf.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_hf.chmod(0o755)
    return fake_bin


def _run_script(
    tmp_path: Path,
    *,
    phase: str,
    overrides: dict[str, str] | None = None,
    unset: tuple[str, ...] = (),
    staged_script: Path | None = None,
) -> subprocess.CompletedProcess:
    staged_script = staged_script or _stage_script(tmp_path)
    fake_bin = _fake_bin(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CAPTURE_PATH": str(tmp_path / "server_args.bin"),
        "CLIENT_CAPTURE_PATH": str(tmp_path / "client_args.bin"),
        "CAPTURE_ENV_PATH": str(tmp_path / "captured_env.txt"),
        "EVAL_MARKER": str(tmp_path / "eval_called"),
        "MAGPIE_RUN_PHASE": phase,
        "MAGPIE_SERVER_PID_FILE": str(tmp_path / "server.pid"),
        "MODEL": "/models/Qwen3.5-9B",
        "TP": "1",
        "CONC": "15",
        "ISL": "512",
        "OSL": "128",
        "RANDOM_RANGE_RATIO": "1",
        "NUM_PROMPTS": "150",
        "RESULT_FILENAME": "result",
        "RESULT_DIR": str(tmp_path),
        "ROCR_VISIBLE_DEVICES": "2",
        "EXTRA_VLLM_ARGS": "--language-model-only --dtype bfloat16",
        "PROFILE": "0",
        "RUN_EVAL": "false",
        "BENCHMARK_BASE_URL": "",
        "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
    }
    for key in (
        "TARGET_GPU_TYPE",
        "HIP_VISIBLE_DEVICES",
        "VLLM_ROCM_USE_AITER",
        "VLLM_ROCM_USE_AITER_RMSNORM",
        "FLASH_ATTENTION_TRITON_AMD_ENABLE",
        "MAX_MODEL_LEN",
        "GPU_MEMORY_UTILIZATION",
        "MAGPIE_KEEP_CONTAINER_ALIVE",
        "MAGPIE_CLIENT_TOKENIZER_MODE",
        "HYPERLOOM_CLIENT_TOKENIZER_MODE",
        *unset,
    ):
        env.pop(key, None)
    env.update(overrides or {})
    return subprocess.run(
        ["bash", str(staged_script)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _args(path: Path) -> list[str]:
    return [value.decode() for value in path.read_bytes().split(b"\0") if value]


def _env(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines()
    )


def test_server_launch_uses_gfx11_defaults(tmp_path: Path):
    completed = _run_script(tmp_path, phase="server")

    assert completed.returncode == 0, completed.stderr
    assert _args(tmp_path / "server_args.bin") == [
        "serve",
        "/models/Qwen3.5-9B",
        "--port",
        "8888",
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.9",
        "--max-model-len",
        "4096",
        "--trust-remote-code",
        "--language-model-only",
        "--dtype",
        "bfloat16",
    ]
    assert _env(tmp_path / "captured_env.txt") == {
        "VLLM_ROCM_USE_AITER": "0",
        "VLLM_ROCM_USE_AITER_RMSNORM": UNSET,
        "FLASH_ATTENTION_TRITON_AMD_ENABLE": "TRUE",
        "HSA_OVERRIDE_GFX_VERSION": UNSET,
        "HIP_VISIBLE_DEVICES": "0",
        "ROCR_VISIBLE_DEVICES": "2",
    }
    assert (tmp_path / "server.pid").read_text().strip().isdigit()
    assert not (tmp_path / "client_args.bin").exists()


def test_hip_devices_follow_the_rocr_count(tmp_path: Path):
    completed = _run_script(
        tmp_path, phase="server", overrides={"ROCR_VISIBLE_DEVICES": "1,3", "TP": "2"}
    )

    assert completed.returncode == 0, completed.stderr
    assert _env(tmp_path / "captured_env.txt")["HIP_VISIBLE_DEVICES"] == "0,1"


def test_explicit_settings_are_preserved(tmp_path: Path):
    completed = _run_script(
        tmp_path,
        phase="server",
        overrides={
            "VLLM_ROCM_USE_AITER": "1",
            "FLASH_ATTENTION_TRITON_AMD_ENABLE": "FALSE",
            "HIP_VISIBLE_DEVICES": "3",
            "MAX_MODEL_LEN": "8192",
            "GPU_MEMORY_UTILIZATION": "0.8",
        },
    )

    assert completed.returncode == 0, completed.stderr
    args = _args(tmp_path / "server_args.bin")
    assert args[args.index("--max-model-len") + 1] == "8192"
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.8"
    env = _env(tmp_path / "captured_env.txt")
    assert env["VLLM_ROCM_USE_AITER"] == "1"
    assert env["VLLM_ROCM_USE_AITER_RMSNORM"] == UNSET
    assert env["FLASH_ATTENTION_TRITON_AMD_ENABLE"] == "FALSE"
    assert env["HIP_VISIBLE_DEVICES"] == "3"


def test_local_client_sends_the_requested_workload(tmp_path: Path):
    completed = _run_script(tmp_path, phase="client")

    assert completed.returncode == 0, completed.stderr
    assert not (tmp_path / "server_args.bin").exists()
    assert _args(tmp_path / "client_args.bin") == [
        "--model",
        "/models/Qwen3.5-9B",
        "--port",
        "8888",
        "--backend",
        "vllm",
        "--input-len",
        "512",
        "--output-len",
        "128",
        "--random-range-ratio",
        "1",
        "--num-prompts",
        "150",
        "--max-concurrency",
        "15",
        "--result-filename",
        "result",
        "--result-dir",
        f"{tmp_path}/",
        "--trust-remote-code",
    ]


def test_num_prompts_defaults_to_ten_per_concurrent_request(tmp_path: Path):
    completed = _run_script(
        tmp_path, phase="client", overrides={"CONC": "4"}, unset=("NUM_PROMPTS",)
    )

    assert completed.returncode == 0, completed.stderr
    args = _args(tmp_path / "client_args.bin")
    assert args[args.index("--num-prompts") + 1] == "40"


def test_all_phase_launches_server_then_client(tmp_path: Path):
    completed = _run_script(tmp_path, phase="all")

    assert completed.returncode == 0, completed.stderr
    assert _args(tmp_path / "server_args.bin")[:2] == ["serve", "/models/Qwen3.5-9B"]
    client_args = _args(tmp_path / "client_args.bin")
    assert "--server-pid" in client_args
    assert not (tmp_path / "eval_called").exists()


@pytest.mark.parametrize("phase", ["all", "server"])
def test_remote_server_forces_the_client_phase(tmp_path: Path, phase: str):
    completed = _run_script(
        tmp_path,
        phase=phase,
        overrides={"BENCHMARK_BASE_URL": "http://127.0.0.1:8888", "RUN_EVAL": "true"},
    )

    assert completed.returncode == 0, completed.stderr
    assert f"forcing PHASE=client (was {phase})" in completed.stdout
    assert not (tmp_path / "server_args.bin").exists()
    assert _args(tmp_path / "client_args.bin")[:2] == ["remote", "trust"]
    assert (tmp_path / "eval_called").read_text().strip() == "remote"


@pytest.mark.parametrize("run_eval", ["true", "True", "TRUE"])
def test_eval_runs_after_a_local_client(tmp_path: Path, run_eval: str):
    completed = _run_script(
        tmp_path, phase="client", overrides={"RUN_EVAL": run_eval}
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "eval_called").read_text().split() == [
        "--framework",
        "lm-eval",
        "--port",
        "8888",
    ]


def test_server_phase_never_runs_eval(tmp_path: Path):
    completed = _run_script(tmp_path, phase="server", overrides={"RUN_EVAL": "true"})

    assert completed.returncode == 0, completed.stderr
    assert not (tmp_path / "eval_called").exists()


def test_invalid_phase_exits_2(tmp_path: Path):
    completed = _run_script(tmp_path, phase="warmup")

    assert completed.returncode == 2
    assert "Invalid MAGPIE_RUN_PHASE='warmup'" in completed.stderr


@pytest.mark.parametrize(
    "dependency",
    [
        "benchmark_lib.sh",
        "server_cleanup.sh",
        "magpie_bench_remote_compat.sh",
        "magpie_split_extra_vllm_args.sh",
    ],
)
def test_missing_dependency_exits_3(tmp_path: Path, dependency: str):
    staged_script = _stage_script(tmp_path)
    (staged_script.parent / dependency).unlink()

    completed = _run_script(tmp_path, phase="server", staged_script=staged_script)

    assert completed.returncode == 3
    assert f"Required benchmark dependency is missing: {staged_script.parent / dependency}" in (
        completed.stderr
    )


def test_script_needs_no_r9700_policy():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "magpie_r9700_vllm_policy" not in text
    assert "magpie_apply_r9700" not in text
    assert os.access(SCRIPT, os.X_OK)
