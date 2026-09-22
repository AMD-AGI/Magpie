import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from Magpie.modes.benchmark.benchmarker import (
    MAGPIE_BUILTIN_SCRIPTS,
    BenchmarkMode,
)
from Magpie.modes.benchmark.config import BenchmarkConfig
from Magpie.modes.benchmark.image_selector import ImageSelector


ROOT = Path(__file__).parents[1]
SCRIPT_DIR = ROOT / "Magpie" / "scripts" / "benchmark"
TEXT_SCRIPT = SCRIPT_DIR / "vllm_gfx12.sh"
SCRIPT = SCRIPT_DIR / "vllm_gfx12_mm.sh"
SINGLE_USER_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "benchmark_vllm_r9700_qwen35_9b_mm_single_user.yaml"
)
MAX_THROUGHPUT_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "benchmark_vllm_r9700_qwen35_9b_mm_max_throughput.yaml"
)
TRACELENS_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "benchmark_vllm_r9700_qwen35_9b_mm_tracelens.yaml"
)


def _stage_client_script(tmp_path: Path) -> Path:
    script_dir = tmp_path / "benchmarks"
    script_dir.mkdir(exist_ok=True)
    staged_script = script_dir / SCRIPT.name
    shutil.copy2(SCRIPT, staged_script)
    (script_dir / "benchmark_lib.sh").write_text(
        """check_env_vars() {
  local name
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      echo "missing required variable: $name" >&2
      return 1
    fi
  done
}
""",
        encoding="utf-8",
    )
    (script_dir / "server_cleanup.sh").write_text(
        "magpie_stop_benchmark_server_stack() { :; }\n",
        encoding="utf-8",
    )
    (script_dir / "magpie_bench_remote_compat.sh").write_text(
        """magpie_run_eval_remote_direct() { : > "$EVAL_MARKER"; }
magpie_run_eval_persisted() { : > "$EVAL_MARKER"; }
""",
        encoding="utf-8",
    )
    return staged_script


def _capture_client_args(
    tmp_path: Path,
    *,
    num_prompts: int | None = 10,
    concurrency: int = 1,
    profile: bool = False,
    run_eval: str = "false",
) -> list[str]:
    staged_script = _stage_client_script(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    fake_vllm = fake_bin / "vllm"
    fake_vllm.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\0\' "$@" > "$CAPTURE_PATH"\n',
        encoding="utf-8",
    )
    fake_vllm.chmod(0o755)

    capture_path = tmp_path / (
        "profile_args.bin" if profile else "benchmark_args.bin"
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CAPTURE_PATH": str(capture_path),
        "MAGPIE_RUN_PHASE": "client",
        "BENCHMARK_BASE_URL": "http://127.0.0.1:8888",
        "MODEL": "Qwen/Qwen3.5-9B",
        "CONC": str(concurrency),
        "ISL": "512",
        "OSL": "128",
        "RANDOM_RANGE_RATIO": "0.0",
        "REQUEST_RATE": "inf",
        "MM_BASE_ITEMS_PER_REQUEST": "1",
        "MM_ITEMS_RANGE_RATIO": "0.0",
        "MM_LIMIT_IMAGES": "3",
        "MM_LIMIT_VIDEOS": "0",
        "MM_IMAGE_HEIGHT": "256",
        "MM_IMAGE_WIDTH": "256",
        "MM_IMAGE_NUM_FRAMES": "1",
        "SEED": "42",
        "RESULT_DIR": str(tmp_path),
        "RESULT_FILENAME": "inferencex_result",
        "RUN_EVAL": run_eval,
        "EVAL_MARKER": str(tmp_path / "eval_called"),
        "PROFILE": "1" if profile else "0",
    }
    if num_prompts is not None:
        env["NUM_PROMPTS"] = str(num_prompts)
    completed = subprocess.run(
        ["bash", str(staged_script)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "No such file or directory" not in completed.stderr
    assert "command not found" not in completed.stderr
    return [
        value.decode()
        for value in capture_path.read_bytes().split(b"\0")
        if value
    ]


@pytest.mark.parametrize(("num_prompts", "concurrency"), [(10, 1), (150, 15)])
def test_gfx12_mm_client_uses_acceptance_workload(
    tmp_path: Path,
    num_prompts: int,
    concurrency: int,
):
    args = _capture_client_args(
        tmp_path,
        num_prompts=num_prompts,
        concurrency=concurrency,
    )

    assert args == [
        "bench",
        "serve",
        "--model",
        "Qwen/Qwen3.5-9B",
        "--backend",
        "openai-chat",
        "--endpoint",
        "/v1/chat/completions",
        "--base-url",
        "http://127.0.0.1:8888",
        "--dataset-name",
        "random-mm",
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(concurrency),
        "--random-input-len",
        "512",
        "--random-output-len",
        "128",
        "--random-range-ratio",
        "0.0",
        "--random-mm-base-items-per-request",
        "1",
        "--random-mm-num-mm-items-range-ratio",
        "0.0",
        "--random-mm-limit-mm-per-prompt",
        '{"image": 3, "video": 0}',
        "--random-mm-bucket-config",
        "{(256, 256, 1): 1.0}",
        "--request-rate",
        "inf",
        "--num-warmups",
        str(min(concurrency, 8)),
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--ignore-eos",
        "--seed",
        "42",
        "--save-result",
        "--result-dir",
        f"{tmp_path}/",
        "--result-filename",
        "inferencex_result.json",
        "--trust-remote-code",
    ]
    assert (tmp_path / "client_command.sh").is_file()


def test_gfx12_mm_profiling_only_adds_profile_flag(tmp_path: Path):
    benchmark_args = _capture_client_args(tmp_path)
    profile_args = _capture_client_args(tmp_path, profile=True)

    assert "--profile" in profile_args
    profile_args.remove("--profile")
    assert profile_args == benchmark_args


def test_gfx12_mm_defaults_num_prompts_from_concurrency(tmp_path: Path):
    args = _capture_client_args(
        tmp_path,
        num_prompts=None,
        concurrency=3,
    )

    assert args[args.index("--num-prompts") + 1] == "30"


def test_gfx12_mm_normalizes_config_boolean_for_eval(tmp_path: Path):
    _capture_client_args(tmp_path, run_eval="True")

    assert (tmp_path / "eval_called").is_file()


@pytest.mark.parametrize("source", [TEXT_SCRIPT, SCRIPT], ids=lambda path: path.name)
def test_gfx12_scripts_fail_fast_without_benchmark_lib(
    tmp_path: Path,
    source: Path,
):
    staged_script = tmp_path / source.name
    shutil.copy2(source, staged_script)

    completed = subprocess.run(
        ["bash", str(staged_script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 3
    assert "Required benchmark dependency is missing" in completed.stderr
    assert "benchmark_lib.sh" in completed.stderr


def test_gfx1201_auto_resolves_to_gfx12_script(tmp_path: Path):
    inferencex = tmp_path / "InferenceX"
    (inferencex / "benchmarks").mkdir(parents=True)
    config = BenchmarkConfig(
        framework="vllm",
        model="demo",
        gpu_arch="gfx1201",
        inferencex_path=str(inferencex),
    )
    mode = BenchmarkMode(config, output_dir=str(tmp_path / "results"))
    mode._prepare_benchmark_scripts()

    selector = ImageSelector()
    runner = selector.get_runner_type(config.gpu_arch)
    assert runner == "gfx12"
    assert selector.select_image("vllm", config.gpu_arch) == (
        "rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_"
        "pytorch_2.12.0_vllm_0.27.0"
    )
    assert mode._get_benchmark_script(runner) == "benchmarks/vllm_gfx12.sh"


def test_gfx12_mm_configs_pin_acceptance_cases_and_reuse_server(tmp_path: Path):
    single_data = yaml.safe_load(SINGLE_USER_CONFIG.read_text())["benchmark"]
    max_data = yaml.safe_load(MAX_THROUGHPUT_CONFIG.read_text())["benchmark"]
    single = BenchmarkConfig.from_dict(single_data)
    maximum = BenchmarkConfig.from_dict(max_data)

    assert "vllm_gfx12_mm.sh" in MAGPIE_BUILTIN_SCRIPTS
    assert single.benchmark_script == maximum.benchmark_script == "vllm_gfx12_mm.sh"
    assert single.model == maximum.model == "Qwen/Qwen3.5-9B"
    assert single.precision == maximum.precision == "bf16"
    assert single.envs["CONC"] == 1
    assert single.envs["NUM_PROMPTS"] == 10
    assert maximum.envs["CONC"] == 15
    assert maximum.envs["NUM_PROMPTS"] == 150

    common_single = {
        key: value
        for key, value in single.envs.items()
        if key not in {"CONC", "NUM_PROMPTS"}
    }
    common_maximum = {
        key: value
        for key, value in maximum.envs.items()
        if key not in {"CONC", "NUM_PROMPTS"}
    }
    assert common_single == common_maximum
    assert common_single == {
        "TARGET_GPU_TYPE": "r9700",
        "TP": 1,
        "PORT": 8888,
        "ISL": 512,
        "OSL": 128,
        "RANDOM_RANGE_RATIO": 0.0,
        "REQUEST_RATE": "inf",
        "MM_BASE_ITEMS_PER_REQUEST": 1,
        "MM_ITEMS_RANGE_RATIO": 0.0,
        "MM_LIMIT_IMAGES": 3,
        "MM_LIMIT_VIDEOS": 0,
        "MM_IMAGE_HEIGHT": 256,
        "MM_IMAGE_WIDTH": 256,
        "MM_IMAGE_NUM_FRAMES": 1,
        "SEED": 42,
        "MAX_MODEL_LEN": 4096,
        "GPU_MEMORY_UTILIZATION": 0.9,
        "VLLM_ROCM_USE_AITER": 0,
        "EXTRA_VLLM_ARGS": "--dtype bfloat16",
        "RUN_EVAL": "false",
    }
    assert single.get_env_vars()["RUN_EVAL"] == "false"
    assert maximum.get_env_vars()["RUN_EVAL"] == "false"

    assert single.server_lifecycle is not None
    assert maximum.server_lifecycle is not None
    assert single.server_lifecycle.cleanup is False
    assert maximum.server_lifecycle.cleanup is True

    single.inferencex_path = str(tmp_path / "InferenceX")
    maximum.inferencex_path = single.inferencex_path
    image = single.docker_image
    single_meta = BenchmarkMode(single)._desired_reuse_server_meta(
        8888, docker_image=image
    )
    maximum_meta = BenchmarkMode(maximum)._desired_reuse_server_meta(
        8888, docker_image=image
    )
    assert single_meta == maximum_meta


def test_gfx12_mm_tracelens_config_preserves_concrete_target():
    data = yaml.safe_load(TRACELENS_CONFIG.read_text())["benchmark"]
    config = BenchmarkConfig.from_dict(data)

    assert config.gpu_arch == "gfx1201"
    assert config.runner_type == "gfx12"
    assert config.benchmark_script == "vllm_gfx12_mm.sh"
    assert config.envs["TARGET_GPU_TYPE"] == "r9700"
    assert config.envs["ENABLE_PROFILE"] == "true"
    assert config.profiler.torch_profiler.enabled is True
    assert config.profiler.tracelens.enabled is True
    assert config.profiler.tracelens.analysis_stages == [
        "prefilldecode",
        "decode",
        "prefill",
    ]
    assert config.profiler.tracelens.perf_report_enabled is True
    assert config.profiler.tracelens.multi_rank_report_enabled is False
