import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
LOCAL_EVAL_SCRIPTS = [
    "atom_mi300x.sh",
    "atom_mi355x.sh",
    "sglang_mi300x.sh",
    "sglang_mi355x.sh",
    "sglang_radeon8060s.sh",
    "vllm_mi300x.sh",
    "vllm_mi355x.sh",
    "vllm_radeon8060s.sh",
]
CONCURRENCY_EXPORT = (
    'export EVAL_CONCURRENT_REQUESTS="${MAGPIE_EVAL_CONCURRENCY:-'
    '${EVAL_CONCURRENT_REQUESTS:-$CONC}}"'
)


def test_radeon_scripts_are_registered_for_server_lifecycle():
    from Magpie.modes.benchmark.benchmarker import MAGPIE_BUILTIN_SCRIPTS

    assert {"vllm_radeon8060s.sh", "sglang_radeon8060s.sh"}.issubset(MAGPIE_BUILTIN_SCRIPTS)


def test_radeon_scripts_pin_the_qualified_attention_routes():
    scripts = ROOT / "Magpie" / "scripts" / "benchmark"
    vllm = (scripts / "vllm_radeon8060s.sh").read_text(encoding="utf-8")
    sglang = (scripts / "sglang_radeon8060s.sh").read_text(encoding="utf-8")

    for contents in (vllm, sglang):
        assert "export PYTORCH_ROCM_ARCH=gfx1151" in contents
        assert "unset HSA_OVERRIDE_GFX_VERSION" in contents
    assert "export VLLM_ROCM_USE_AITER=0" in vllm
    assert '"--attention-backend=triton"' in sglang
    assert '"--disable-cuda-graph"' in sglang
    assert "export HYPERLOOM_GFX1151_LOWBIT_BRIDGE=0" in sglang


def _run_radeon_client(tmp_path: Path, script_name: str, **overrides: str) -> subprocess.CompletedProcess[str]:
    source = ROOT / "Magpie" / "scripts" / "benchmark" / script_name
    script = tmp_path / script_name
    shutil.copy2(source, script)
    (tmp_path / "benchmark_lib.sh").write_text(
        "check_env_vars() { :; }\n"
        "run_benchmark_serving() { printf 'ARCH=%s VLLM_AITER=%s SGLANG_AITER=%s BRIDGE=%s\\n' "
        '"${PYTORCH_ROCM_ARCH:-}" "${VLLM_ROCM_USE_AITER:-}" "${SGLANG_USE_AITER:-}" '
        '"${HYPERLOOM_GFX1151_LOWBIT_BRIDGE:-}"; }\n',
        encoding="utf-8",
    )
    (tmp_path / "server_cleanup.sh").write_text("magpie_stop_benchmark_server_stack() { :; }\n", encoding="utf-8")
    (tmp_path / "magpie_bench_remote_compat.sh").write_text("", encoding="utf-8")
    env = {
        **os.environ,
        "MAGPIE_RUN_PHASE": "client",
        "MODEL": "test-model",
        "TP": "1",
        "CONC": "1",
        "ISL": "64",
        "OSL": "16",
        "RANDOM_RANGE_RATIO": "1.0",
        "RESULT_FILENAME": "result",
        "RESULT_DIR": str(tmp_path),
        "RUN_EVAL": "false",
        "BENCHMARK_BASE_URL": "",
        "EXTRA_VLLM_ARGS": "",
        "EXTRA_SGLANG_ARGS": "",
        "SLURM_JOB_ID": "",
        **overrides,
    }
    return subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, check=False)


def test_radeon_clients_force_qualified_environment(tmp_path: Path):
    vllm = _run_radeon_client(tmp_path, "vllm_radeon8060s.sh", VLLM_ROCM_USE_AITER="1")
    assert vllm.returncode == 0, vllm.stderr
    assert "ARCH=gfx1151 VLLM_AITER=0" in vllm.stdout

    sglang = _run_radeon_client(
        tmp_path,
        "sglang_radeon8060s.sh",
        SGLANG_USE_AITER="1",
        HYPERLOOM_GFX1151_LOWBIT_BRIDGE="1",
    )
    assert sglang.returncode == 0, sglang.stderr
    assert "ARCH=gfx1151 VLLM_AITER= SGLANG_AITER=0 BRIDGE=0" in sglang.stdout


def test_radeon_clients_reject_wrong_arch_and_conflicting_attention(tmp_path: Path):
    wrong_arch = _run_radeon_client(tmp_path, "vllm_radeon8060s.sh", PYTORCH_ROCM_ARCH="gfx950")
    assert wrong_arch.returncode == 2
    assert "requires PYTORCH_ROCM_ARCH=gfx1151" in wrong_arch.stderr

    conflict = _run_radeon_client(
        tmp_path,
        "sglang_radeon8060s.sh",
        EXTRA_SGLANG_ARGS="--attention-backend aiter",
    )
    assert conflict.returncode == 2
    assert "conflicting arg '--attention-backend'" in conflict.stderr


@pytest.mark.parametrize("script_name", LOCAL_EVAL_SCRIPTS)
def test_local_eval_scripts_use_environment_for_concurrency(script_name: str):
    script = ROOT / "Magpie" / "scripts" / "benchmark" / script_name
    contents = script.read_text(encoding="utf-8")

    assert CONCURRENCY_EXPORT in contents
    assert 'magpie_run_eval_persisted --framework lm-eval --port "$PORT"' in contents
    assert "declare -F magpie_run_eval_persisted" not in contents
    assert "--concurrent-requests" not in contents


@pytest.mark.parametrize("script_name", LOCAL_EVAL_SCRIPTS)
def test_local_scripts_keep_docker_server_container_alive(script_name: str):
    script = ROOT / "Magpie" / "scripts" / "benchmark" / script_name
    contents = script.read_text(encoding="utf-8")

    assert '"${MAGPIE_KEEP_CONTAINER_ALIVE:-0}" == "1"' in contents
    assert "wait \"$SERVER_PID\"" in contents
    assert "magpie_stop_benchmark_server_stack" in contents


def _compat_script() -> Path:
    return ROOT / "Magpie" / "scripts" / "benchmark" / "magpie_bench_remote_compat.sh"


def _lm_eval_python_stub(tmp_path: Path, args_file: Path) -> Path:
    stub = tmp_path / "lm_eval_python"
    stub.write_text(
        "#!/usr/bin/python3\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "args_file = Path(os.environ['LM_EVAL_ARGS_FILE'])\n"
        "if len(sys.argv) >= 2 and sys.argv[1] == '-c':\n"
        "    code = sys.argv[2] if len(sys.argv) > 2 else ''\n"
        "    raise SystemExit(1 if 'import lm_eval' in code and os.environ.get('LM_EVAL_MISSING') == '1' else 0)\n"
        "if len(sys.argv) >= 3 and sys.argv[1] == '-m' and sys.argv[2] == 'pip':\n"
        "    with args_file.open('a', encoding='utf-8') as handle:\n"
        "        handle.write(' '.join(sys.argv[1:]) + '\\n')\n"
        "    marker = os.environ.get('LM_EVAL_PIP_MARKER')\n"
        "    if marker:\n"
        "        Path(marker).write_text('installed\\n', encoding='utf-8')\n"
        "    raise SystemExit(int(os.environ.get('LM_EVAL_PIP_RC', '0')))\n"
        "fail_conc = os.environ.get('LM_EVAL_FAIL_CONC')\n"
        "if fail_conc and any(\n"
        "    f'num_concurrent={fail_conc}' in arg for arg in sys.argv\n"
        "):\n"
        "    with args_file.open('a', encoding='utf-8') as handle:\n"
        "        handle.write(' '.join(sys.argv[1:]) + '\\n')\n"
        "    raise SystemExit(int(os.environ.get('LM_EVAL_FAIL_RC', '17')))\n"
        "with args_file.open('a', encoding='utf-8') as handle:\n"
        "    handle.write(' '.join(sys.argv[1:]) + '\\n')\n"
        "    handle.write('HF_ALLOW_CODE_EVAL=' + os.environ.get('HF_ALLOW_CODE_EVAL', '') + '\\n')\n"
        "out_dir = Path('.')\n"
        "for index, arg in enumerate(sys.argv):\n"
        "    if arg == '--output_path' and index + 1 < len(sys.argv):\n"
        "        out_dir = Path(sys.argv[index + 1])\n"
        "        break\n"
        "payload = {\n"
        "    'lm_eval_version': '0.4.8',\n"
        "    'results': {\n"
        "        'gsm8k': {'exact_match,strict-match': 0.9},\n"
        "        'mmlu': {'acc,none': 0.8},\n"
        "        'hellaswag': {'acc,none': 0.85},\n"
        "        'humaneval_instruct': {'pass@1,create_test': 0.7},\n"
        "    }\n"
        "}\n"
        "out_dir.mkdir(parents=True, exist_ok=True)\n"
        "(out_dir / 'results.json').write_text(json.dumps(payload) + '\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


def test_remote_eval_prefers_independent_accuracy_concurrency():
    contents = _compat_script().read_text(encoding="utf-8")

    assert "magpie_eval_concurrency_values()" in contents
    assert (
        'local raw="${MAGPIE_EVAL_CONCURRENCY:-'
        '${EVAL_CONCURRENT_REQUESTS:-${CONC:-8}}}"'
    ) in contents
    assert "--model local-completions" in contents
    assert "--model local-chat-completions" not in contents
    assert "unset -f python3" not in contents
    assert "Crystal" not in contents


def test_persisted_eval_writes_raw_and_formatted_results(tmp_path: Path):
    result_payload = {
        "lm_eval_version": "0.4.8",
        "results": {
            "gsm8k": {
                "exact_match,strict-match": 0.98,
                "exact_match,flexible-extract": 0.99,
            }
        },
        "n-samples": {"gsm8k": {"original": 1319, "effective": 100}},
    }
    shell = r'''
source "$MAGPIE_COMPAT"
run_eval() {
  mkdir -p "$EVAL_RESULT_DIR/model"
  printf '%s\n' "$FAKE_EVAL_RESULT" > "$EVAL_RESULT_DIR/model/results_123.json"
}
append_lm_eval_summary() {
  return 99
}
_write_lm_eval_meta_json() {
  printf '{"model":"test","conc":%s}\n' "$3" > "$1"
}
unset MAGPIE_EVAL_TASKS
magpie_run_eval_persisted --framework lm-eval --port 8888
'''
    source_dir = tmp_path / "source"
    env = {
        **os.environ,
        "MAGPIE_COMPAT": str(_compat_script()),
        "RESULT_DIR": str(tmp_path),
        "EVAL_RESULT_DIR": str(source_dir),
        "FAKE_EVAL_RESULT": json.dumps(result_payload),
        "CONC": "64",
        "EVAL_CONCURRENT_REQUESTS": "8",
    }
    env.pop("MAGPIE_EVAL_TASKS", None)
    subprocess.run(["bash", "-c", shell], check=True, env=env)

    eval_dir = tmp_path / "lm_eval"
    assert json.loads((source_dir / "model" / "results_123.json").read_text()) == result_payload
    assert (source_dir / "meta_env.json").is_file()
    assert json.loads((source_dir / "meta_env.json").read_text())["conc"] == 8
    assert json.loads((eval_dir / "results_123.json").read_text()) == result_payload
    assert (eval_dir / "meta_env.json").is_file()
    summary = json.loads((tmp_path / "accuracy_report.json").read_text())
    assert summary["status"] == "COMPLETED"
    assert summary["task"] == "gsm8k"
    assert summary["metric"] == "exact_match,strict-match"
    assert summary["score"] == 0.98
    assert summary["samples"] == 100
    assert summary["source_result"] == "lm_eval/results_123.json"


def test_persisted_eval_collects_batched_concurrency_results(tmp_path: Path):
    shell = r'''
source "$MAGPIE_COMPAT"
run_eval() {
  local conc
  for conc in $EVAL_CONCURRENT_REQUESTS; do
    printf '{"results":{"gsm8k":{"exact_match,strict-match":0.%s}}}\n' \
      "$conc" > "results_conc${conc}.json"
  done
  export EVAL_BATCHED_CONCS="$EVAL_CONCURRENT_REQUESTS"
  export EVAL_BATCHED_COMPLETED_CONCS="$EVAL_CONCURRENT_REQUESTS"
  export EVAL_BATCHED_FAILED_CONCS=""
  export EVAL_RESULT_DIR=""
}
append_lm_eval_summary() {
  printf '%s\n' '{"eval_concs":[2,4]}' > ./meta_env.json
}
_write_lm_eval_meta_json() {
  return 99
}
unset MAGPIE_EVAL_TASKS
magpie_run_eval_persisted --framework lm-eval --port 8888
'''
    env = {
        **os.environ,
        "MAGPIE_COMPAT": str(_compat_script()),
        "RESULT_DIR": str(tmp_path),
        "EVAL_CONCURRENT_REQUESTS": "2 4",
    }
    env.pop("MAGPIE_EVAL_TASKS", None)
    subprocess.run(["bash", "-c", shell], check=True, env=env)

    eval_dir = tmp_path / "lm_eval"
    assert (eval_dir / "results_conc2.json").is_file()
    assert (eval_dir / "results_conc4.json").is_file()
    assert json.loads((eval_dir / "meta_env.json").read_text())["eval_concs"] == [
        2,
        4,
    ]
    summary = json.loads((tmp_path / "accuracy_report.json").read_text())
    assert summary["status"] == "COMPLETED"
    assert summary["task"] == "gsm8k"


def test_persisted_eval_drives_local_completions_for_multi_task(tmp_path: Path):
    args_file = tmp_path / "lm_eval.args"
    python_stub = _lm_eval_python_stub(tmp_path, args_file)
    shell = r'''
source "$MAGPIE_COMPAT"
run_eval() {
  printf 'run_eval_should_not_run\n' > "$EVAL_RESULT_DIR/run_eval.txt"
  return 99
}
_write_lm_eval_meta_json() {
  printf '{"model":"test","conc":%s}\n' "$3" > "$1"
}
magpie_run_eval_persisted --framework lm-eval --port 7777
'''
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    env = {
        **os.environ,
        "MAGPIE_COMPAT": str(_compat_script()),
        "RESULT_DIR": str(tmp_path),
        "EVAL_RESULT_DIR": str(source_dir),
        "MAGPIE_EVAL_PYTHON": str(python_stub),
        "MAGPIE_ACCURACY_REPORT_PYTHON": sys.executable,
        "MAGPIE_EVAL_TASKS": "gsm8k,mmlu,hellaswag",
        "MAGPIE_EVAL_LIMIT": "100",
        "EVAL_CONCURRENT_REQUESTS": "8",
        "LM_EVAL_ARGS_FILE": str(args_file),
        "MODEL": "test-model",
    }
    env.pop("HF_ALLOW_CODE_EVAL", None)
    subprocess.run(["bash", "-c", shell], check=True, env=env)
    assert not (source_dir / "run_eval.txt").exists()
    args = args_file.read_text()
    assert "-m lm_eval" in args
    assert "--model local-completions" in args
    assert "local-chat-completions" not in args
    assert "--tasks gsm8k mmlu hellaswag" in args
    assert "--tasks gsm8k,mmlu,hellaswag" not in args
    assert "--limit 100" in args
    assert "base_url=http://127.0.0.1:7777/v1/completions" in args
    assert "num_concurrent=8" in args
    assert "--output_path" in args
    assert str(source_dir / "conc8") in args
    assert (source_dir / "results_conc8.json").is_file()
    assert json.loads((source_dir / "results_conc8.json").read_text())["lm_eval_version"] == "0.4.8"
    summary = json.loads((tmp_path / "accuracy_report.json").read_text())
    assert summary["status"] == "COMPLETED"
    assert "gsm8k" in summary["tasks"]
    assert "mmlu" in summary["tasks"]
    assert "hellaswag" in summary["tasks"]


def test_persisted_eval_writes_per_concurrency_result_files(tmp_path: Path):
    args_file = tmp_path / "lm_eval.args"
    python_stub = _lm_eval_python_stub(tmp_path, args_file)
    shell = r'''
source "$MAGPIE_COMPAT"
_write_lm_eval_meta_json() {
  printf '{"model":"test","conc":%s}\n' "$3" > "$1"
}
append_lm_eval_summary() {
  printf '%s\n' '{"eval_concs":[8,64]}' > ./meta_env.json
}
magpie_run_eval_persisted --framework lm-eval --port 8888
'''
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    env = {
        **os.environ,
        "MAGPIE_COMPAT": str(_compat_script()),
        "RESULT_DIR": str(tmp_path),
        "EVAL_RESULT_DIR": str(source_dir),
        "MAGPIE_EVAL_PYTHON": str(python_stub),
        "MAGPIE_ACCURACY_REPORT_PYTHON": sys.executable,
        "MAGPIE_EVAL_TASKS": "gsm8k,mmlu",
        "MAGPIE_EVAL_CONCURRENCY": "8 64",
        "LM_EVAL_ARGS_FILE": str(args_file),
        "MODEL": "test-model",
    }
    subprocess.run(["bash", "-c", shell], check=True, env=env)
    args = args_file.read_text()
    assert "num_concurrent=8" in args
    assert "num_concurrent=64" in args
    assert (source_dir / "results_conc8.json").is_file()
    assert (source_dir / "results_conc64.json").is_file()
    eval_dir = tmp_path / "lm_eval"
    assert (eval_dir / "results_conc8.json").is_file()
    assert (eval_dir / "results_conc64.json").is_file()


def test_persisted_eval_adds_unsafe_flag_and_code_eval_env_for_humaneval(tmp_path: Path):
    args_file = tmp_path / "lm_eval.args"
    python_stub = _lm_eval_python_stub(tmp_path, args_file)
    shell = r'''
source "$MAGPIE_COMPAT"
run_eval() {
  printf 'run_eval_should_not_run\n' > "$EVAL_RESULT_DIR/run_eval.txt"
  return 99
}
_write_lm_eval_meta_json() {
  printf '{"model":"test","conc":8}\n' > "$1"
}
magpie_run_eval_persisted --framework lm-eval --port 8888
'''
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    env = {
        **os.environ,
        "MAGPIE_COMPAT": str(_compat_script()),
        "RESULT_DIR": str(tmp_path),
        "EVAL_RESULT_DIR": str(source_dir),
        "MAGPIE_EVAL_PYTHON": str(python_stub),
        "MAGPIE_ACCURACY_REPORT_PYTHON": sys.executable,
        "MAGPIE_EVAL_TASKS": "gsm8k,humaneval_instruct",
        "EVAL_CONCURRENT_REQUESTS": "8",
        "LM_EVAL_ARGS_FILE": str(args_file),
        "MODEL": "test-model",
    }
    env.pop("HF_ALLOW_CODE_EVAL", None)
    subprocess.run(["bash", "-c", shell], check=True, env=env)
    assert not (source_dir / "run_eval.txt").exists()
    args = args_file.read_text()
    assert "--confirm_run_unsafe_code" in args
    assert "--model local-completions" in args
    assert "HF_ALLOW_CODE_EVAL=1" in args
    assert "unset -f python3" not in _compat_script().read_text(encoding="utf-8")


def test_remote_eval_adds_unsafe_flag_and_code_eval_env_for_humaneval(tmp_path: Path):
    args_file = tmp_path / "lm_eval.args"
    python_stub = _lm_eval_python_stub(tmp_path, args_file)
    shell = r'''
source "$MAGPIE_COMPAT"
magpie_write_accuracy_result() { return 0; }
magpie_run_eval_remote_direct
'''
    env = {
        **os.environ,
        "MAGPIE_COMPAT": str(_compat_script()),
        "RESULT_DIR": str(tmp_path),
        "BENCHMARK_BASE_URL": "http://127.0.0.1:8888",
        "MAGPIE_EVAL_PYTHON": str(python_stub),
        "MAGPIE_EVAL_TASKS": "gsm8k,humaneval_instruct",
        "LM_EVAL_ARGS_FILE": str(args_file),
        "MODEL": "test-model",
    }
    env.pop("HF_ALLOW_CODE_EVAL", None)
    subprocess.run(["bash", "-c", shell], check=True, env=env)
    args = args_file.read_text()
    assert "--confirm_run_unsafe_code" in args
    assert "--model local-completions" in args
    assert "--tasks gsm8k humaneval_instruct" in args
    assert "--tasks gsm8k,humaneval_instruct" not in args
    assert "HF_ALLOW_CODE_EVAL=1" in args


def _per_task_lm_eval_stub(tmp_path: Path, args_file: Path) -> Path:
    """lm-eval stub that scores only the tasks it was asked for."""
    stub = tmp_path / "per_task_python"
    stub.write_text(
        "#!/usr/bin/python3\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "args_file = Path(os.environ['LM_EVAL_ARGS_FILE'])\n"
        "if len(sys.argv) >= 2 and sys.argv[1] == '-c':\n"
        "    raise SystemExit(0)\n"
        "with args_file.open('a', encoding='utf-8') as handle:\n"
        "    handle.write(' '.join(sys.argv[1:]) + '\\n')\n"
        "tasks = []\n"
        "out_dir = Path('.')\n"
        "for index, arg in enumerate(sys.argv):\n"
        "    if arg == '--output_path' and index + 1 < len(sys.argv):\n"
        "        out_dir = Path(sys.argv[index + 1])\n"
        "    if arg == '--tasks':\n"
        "        for value in sys.argv[index + 1:]:\n"
        "            if value.startswith('--'):\n"
        "                break\n"
        "            tasks.append(value)\n"
        "payload = {'results': {task: {'acc,none': 0.5} for task in tasks}}\n"
        "out_dir.mkdir(parents=True, exist_ok=True)\n"
        "(out_dir / ('results_' + tasks[0] + '.json')).write_text(\n"
        "    json.dumps(payload) + '\\n', encoding='utf-8'\n"
        ")\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


def test_remote_eval_applies_stock_include_per_task(tmp_path: Path):
    args_file = tmp_path / "lm_eval.args"
    python_stub = _per_task_lm_eval_stub(tmp_path, args_file)
    include_dir = tmp_path / "utils" / "evals"
    include_dir.mkdir(parents=True)
    shell = r'''
source "$MAGPIE_COMPAT"
magpie_write_accuracy_result() { return 0; }
magpie_run_eval_remote_direct
'''
    env = {
        **os.environ,
        "MAGPIE_COMPAT": str(_compat_script()),
        "RESULT_DIR": str(tmp_path),
        "BENCHMARK_BASE_URL": "http://127.0.0.1:8888",
        "MAGPIE_EVAL_PYTHON": str(python_stub),
        "MAGPIE_ACCURACY_REPORT_PYTHON": sys.executable,
        "MAGPIE_EVAL_TASKS": "gsm8k,mmlu,gpqa_diamond_cot_n_shot",
        "MAGPIE_EVAL_TASK_PATH": str(include_dir),
        "EVAL_CONCURRENT_REQUESTS": "8",
        "LM_EVAL_ARGS_FILE": str(args_file),
        "MODEL": "test-model",
    }
    subprocess.run(["bash", "-c", shell], check=True, env=env)

    invocations = [line for line in args_file.read_text().splitlines() if "--tasks" in line]
    assert len(invocations) == 2
    plain = next(line for line in invocations if "gpqa_diamond_cot_n_shot" in line)
    stock = next(line for line in invocations if "--tasks gsm8k" in line)
    assert "--tasks mmlu gpqa_diamond_cot_n_shot" in plain
    assert "--include_path" not in plain
    assert f"--include_path {include_dir}" in stock
    assert "gsm8k" not in plain.split("--model_args")[0].replace("--tasks", "")

    merged = json.loads((tmp_path / "lm_eval" / "results_conc8.json").read_text())
    assert set(merged["results"]) == {"gsm8k", "mmlu", "gpqa_diamond_cot_n_shot"}


def test_remote_eval_keeps_single_invocation_without_stock_include(tmp_path: Path):
    args_file = tmp_path / "lm_eval.args"
    python_stub = _per_task_lm_eval_stub(tmp_path, args_file)
    shell = r'''
source "$MAGPIE_COMPAT"
magpie_write_accuracy_result() { return 0; }
magpie_run_eval_remote_direct
'''
    env = {
        **os.environ,
        "MAGPIE_COMPAT": str(_compat_script()),
        "RESULT_DIR": str(tmp_path),
        "BENCHMARK_BASE_URL": "http://127.0.0.1:8888",
        "MAGPIE_EVAL_PYTHON": str(python_stub),
        "MAGPIE_ACCURACY_REPORT_PYTHON": sys.executable,
        "MAGPIE_EVAL_TASKS": "gsm8k,mmlu",
        "EVAL_CONCURRENT_REQUESTS": "8",
        "LM_EVAL_ARGS_FILE": str(args_file),
        "MODEL": "test-model",
    }
    subprocess.run(["bash", "-c", shell], check=True, env=env)

    invocations = [line for line in args_file.read_text().splitlines() if "--tasks" in line]
    assert len(invocations) == 1
    assert "--tasks gsm8k mmlu" in invocations[0]
    assert "--include_path" not in invocations[0]


def test_remote_eval_propagates_accuracy_report_failure(tmp_path: Path):
    compat = (
        ROOT
        / "Magpie"
        / "scripts"
        / "benchmark"
        / "magpie_bench_remote_compat.sh"
    )
    shell = r'''
source "$MAGPIE_COMPAT"
magpie_write_accuracy_result() {
  return 73
}
magpie_run_eval_remote_direct
'''
    env = {
        **os.environ,
        "MAGPIE_COMPAT": str(compat),
        "RESULT_DIR": str(tmp_path),
        "BENCHMARK_BASE_URL": "http://127.0.0.1:8888",
        "MAGPIE_EVAL_PYTHON": "true",
        "MODEL": "test-model",
        "CONC": "8",
    }
    completed = subprocess.run(["bash", "-c", shell], check=False, env=env)

    assert completed.returncode == 73


def test_persisted_eval_calls_inferencex_dep_installer(tmp_path: Path):
    args_file = tmp_path / "lm_eval.args"
    python_stub = _lm_eval_python_stub(tmp_path, args_file)
    marker = tmp_path / "deps.txt"
    shell = r'''
source "$MAGPIE_COMPAT"
_install_lm_eval_deps() {
  printf 'installed\n' > "$DEPS_MARKER"
}
_write_lm_eval_meta_json() {
  printf '{"model":"test","conc":%s}\n' "$3" > "$1"
}
magpie_run_eval_persisted --framework lm-eval --port 8888
'''
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    env = {
        **os.environ,
        "MAGPIE_COMPAT": str(_compat_script()),
        "RESULT_DIR": str(tmp_path),
        "EVAL_RESULT_DIR": str(source_dir),
        "MAGPIE_EVAL_PYTHON": str(python_stub),
        "MAGPIE_ACCURACY_REPORT_PYTHON": sys.executable,
        "MAGPIE_EVAL_TASKS": "gsm8k",
        "EVAL_CONCURRENT_REQUESTS": "8",
        "LM_EVAL_ARGS_FILE": str(args_file),
        "DEPS_MARKER": str(marker),
        "MODEL": "test-model",
    }
    subprocess.run(["bash", "-c", shell], check=True, env=env)
    assert marker.read_text() == "installed\n"
    assert "-m lm_eval" in args_file.read_text()


def test_persisted_eval_applies_inferencex_lm_eval_patch(tmp_path: Path):
    args_file = tmp_path / "lm_eval.args"
    python_stub = _lm_eval_python_stub(tmp_path, args_file)
    marker = tmp_path / "patch.txt"
    shell = r'''
source "$MAGPIE_COMPAT"
_install_lm_eval_deps() { :; }
_patch_lm_eval() {
  printf 'patched\n' > "$PATCH_MARKER"
}
_write_lm_eval_meta_json() {
  printf '{"model":"test","conc":%s}\n' "$3" > "$1"
}
magpie_run_eval_persisted --framework lm-eval --port 8888
'''
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    env = {
        **os.environ,
        "MAGPIE_COMPAT": str(_compat_script()),
        "RESULT_DIR": str(tmp_path),
        "EVAL_RESULT_DIR": str(source_dir),
        "MAGPIE_EVAL_PYTHON": str(python_stub),
        "MAGPIE_ACCURACY_REPORT_PYTHON": sys.executable,
        "MAGPIE_EVAL_TASKS": "gsm8k",
        "EVAL_CONCURRENT_REQUESTS": "8",
        "LM_EVAL_ARGS_FILE": str(args_file),
        "PATCH_MARKER": str(marker),
        "MODEL": "test-model",
    }
    subprocess.run(["bash", "-c", shell], check=True, env=env)
    assert marker.read_text() == "patched\n"

def test_persisted_eval_installs_lm_eval_when_missing(tmp_path: Path):
    args_file = tmp_path / "lm_eval.args"
    python_stub = _lm_eval_python_stub(tmp_path, args_file)
    pip_marker = tmp_path / "pip.txt"
    shell = r'''
source "$MAGPIE_COMPAT"
_write_lm_eval_meta_json() {
  printf '{"model":"test","conc":%s}\n' "$3" > "$1"
}
magpie_run_eval_persisted --framework lm-eval --port 8888
'''
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    env = {
        **os.environ,
        "MAGPIE_COMPAT": str(_compat_script()),
        "RESULT_DIR": str(tmp_path),
        "EVAL_RESULT_DIR": str(source_dir),
        "MAGPIE_EVAL_PYTHON": str(python_stub),
        "MAGPIE_ACCURACY_REPORT_PYTHON": sys.executable,
        "MAGPIE_EVAL_TASKS": "gsm8k",
        "EVAL_CONCURRENT_REQUESTS": "8",
        "LM_EVAL_ARGS_FILE": str(args_file),
        "LM_EVAL_MISSING": "1",
        "LM_EVAL_PIP_MARKER": str(pip_marker),
        "MODEL": "test-model",
    }
    subprocess.run(["bash", "-c", shell], check=True, env=env)
    assert pip_marker.read_text() == "installed\n"
    args = args_file.read_text()
    assert "-m pip install" in args
    assert "lm-eval[api]" in args
    assert "-m lm_eval" in args


def test_persisted_eval_forwards_context_and_generation_limits(tmp_path: Path):
    args_file = tmp_path / "lm_eval.args"
    python_stub = _lm_eval_python_stub(tmp_path, args_file)
    shell = r'''
source "$MAGPIE_COMPAT"
_write_lm_eval_meta_json() {
  printf '{"model":"test","conc":%s}\n' "$3" > "$1"
}
magpie_run_eval_persisted --framework lm-eval --port 8888
'''
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    env = {
        **os.environ,
        "MAGPIE_COMPAT": str(_compat_script()),
        "RESULT_DIR": str(tmp_path),
        "EVAL_RESULT_DIR": str(source_dir),
        "MAGPIE_EVAL_PYTHON": str(python_stub),
        "MAGPIE_ACCURACY_REPORT_PYTHON": sys.executable,
        "MAGPIE_EVAL_TASKS": "gsm8k",
        "EVAL_CONCURRENT_REQUESTS": "8",
        "EVAL_MAX_MODEL_LEN": "16384",
        "EVAL_MAX_GEN_TOKS": "512",
        "LM_EVAL_ARGS_FILE": str(args_file),
        "MODEL": "test-model",
    }
    env.pop("OSL", None)
    env.pop("MAX_MODEL_LEN", None)
    subprocess.run(["bash", "-c", shell], check=True, env=env)
    args = args_file.read_text()
    assert "max_length=16384" in args
    assert "max_gen_toks=512" in args
    assert "--gen_kwargs" in args
    assert "max_tokens=512" in args
    assert "--model local-completions" in args


def test_persisted_eval_derives_generation_budget_from_context(tmp_path: Path):
    args_file = tmp_path / "lm_eval.args"
    python_stub = _lm_eval_python_stub(tmp_path, args_file)
    shell = r'''
source "$MAGPIE_COMPAT"
_write_lm_eval_meta_json() {
  printf '{"model":"test","conc":%s}\n' "$3" > "$1"
}
magpie_run_eval_persisted --framework lm-eval --port 8888
'''
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    env = {
        **os.environ,
        "MAGPIE_COMPAT": str(_compat_script()),
        "RESULT_DIR": str(tmp_path),
        "EVAL_RESULT_DIR": str(source_dir),
        "MAGPIE_EVAL_PYTHON": str(python_stub),
        "MAGPIE_ACCURACY_REPORT_PYTHON": sys.executable,
        "MAGPIE_EVAL_TASKS": "gsm8k",
        "EVAL_CONCURRENT_REQUESTS": "8",
        "EVAL_MAX_MODEL_LEN": "16384",
        "LM_EVAL_ARGS_FILE": str(args_file),
        "MODEL": "test-model",
    }
    env.pop("OSL", None)
    env.pop("MAX_MODEL_LEN", None)
    env.pop("EVAL_MAX_GEN_TOKS", None)
    env.pop("MAGPIE_EVAL_MAX_GEN_TOKS", None)
    subprocess.run(["bash", "-c", shell], check=True, env=env)
    args = args_file.read_text()
    assert "max_length=16384" in args
    assert "max_gen_toks=12288" in args
    assert "max_tokens=12288" in args

def test_persisted_eval_records_failed_concurrencies_separately(tmp_path: Path):
    args_file = tmp_path / "lm_eval.args"
    python_stub = _lm_eval_python_stub(tmp_path, args_file)
    shell = r'''
set -e
source "$MAGPIE_COMPAT"
append_lm_eval_summary() {
  python3 - <<'PY'
import json, os
from pathlib import Path

def numbers(raw):
    return [int(part) for part in raw.split() if part]

Path("meta_env.json").write_text(
    json.dumps(
        {
            "eval_concs": numbers(os.environ.get("EVAL_BATCHED_CONCS", "")),
            "completed_eval_concs": numbers(
                os.environ.get("EVAL_BATCHED_COMPLETED_CONCS", "")
            ),
            "failed_eval_concs": numbers(
                os.environ.get("EVAL_BATCHED_FAILED_CONCS", "")
            ),
        }
    )
    + "\n",
    encoding="utf-8",
)
PY
}
magpie_run_eval_persisted --framework lm-eval --port 8888
'''
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    env = {
        **os.environ,
        "MAGPIE_COMPAT": str(_compat_script()),
        "RESULT_DIR": str(tmp_path),
        "EVAL_RESULT_DIR": str(source_dir),
        "MAGPIE_EVAL_PYTHON": str(python_stub),
        "MAGPIE_ACCURACY_REPORT_PYTHON": sys.executable,
        "MAGPIE_EVAL_TASKS": "gsm8k",
        "MAGPIE_EVAL_CONCURRENCY": "8 64",
        "LM_EVAL_ARGS_FILE": str(args_file),
        "LM_EVAL_FAIL_CONC": "64",
        "LM_EVAL_FAIL_RC": "17",
        "MODEL": "test-model",
    }
    completed = subprocess.run(["bash", "-c", shell], check=False, env=env)
    assert completed.returncode == 17
    meta = json.loads((source_dir / "meta_env.json").read_text())
    assert meta["eval_concs"] == [8, 64]
    assert meta["completed_eval_concs"] == [8]
    assert meta["failed_eval_concs"] == [64]
    eval_dir = tmp_path / "lm_eval"
    copied = json.loads((eval_dir / "meta_env.json").read_text())
    assert copied["completed_eval_concs"] == [8]
    assert copied["failed_eval_concs"] == [64]
    assert (source_dir / "results_conc8.json").is_file()
    assert not (source_dir / "results_conc64.json").is_file()


def test_persisted_eval_fallback_meta_records_failed_concs(tmp_path: Path):
    args_file = tmp_path / "lm_eval.args"
    python_stub = _lm_eval_python_stub(tmp_path, args_file)
    shell = r'''
source "$MAGPIE_COMPAT"
magpie_run_eval_persisted --framework lm-eval --port 8888
'''
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    env = {
        **os.environ,
        "MAGPIE_COMPAT": str(_compat_script()),
        "RESULT_DIR": str(tmp_path),
        "EVAL_RESULT_DIR": str(source_dir),
        "MAGPIE_EVAL_PYTHON": str(python_stub),
        "MAGPIE_ACCURACY_REPORT_PYTHON": sys.executable,
        "MAGPIE_EVAL_TASKS": "gsm8k",
        "MAGPIE_EVAL_CONCURRENCY": "8 64",
        "LM_EVAL_ARGS_FILE": str(args_file),
        "LM_EVAL_FAIL_CONC": "64",
        "LM_EVAL_FAIL_RC": "17",
        "MODEL": "test-model",
    }
    completed = subprocess.run(["bash", "-c", shell], check=False, env=env)
    assert completed.returncode == 17
    meta = json.loads((source_dir / "meta_env.json").read_text())
    assert meta["eval_concs"] == [8, 64]
    assert meta["completed_eval_concs"] == [8]
    assert meta["failed_eval_concs"] == [64]


def test_persisted_eval_absolutizes_relative_include_directory(tmp_path: Path):
    args_file = tmp_path / "lm_eval.args"
    python_stub = _lm_eval_python_stub(tmp_path, args_file)
    work = tmp_path / "work"
    include = work / "custom-tasks"
    include.mkdir(parents=True)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    shell = r'''
source "$MAGPIE_COMPAT"
_write_lm_eval_meta_json() {
  printf '{"model":"test","conc":%s}\n' "$3" > "$1"
}
magpie_run_eval_persisted --framework lm-eval --port 8888
'''
    env = {
        **os.environ,
        "MAGPIE_COMPAT": str(_compat_script()),
        "RESULT_DIR": str(tmp_path),
        "EVAL_RESULT_DIR": str(source_dir),
        "MAGPIE_EVAL_PYTHON": str(python_stub),
        "MAGPIE_ACCURACY_REPORT_PYTHON": sys.executable,
        "MAGPIE_EVAL_TASKS": "gsm8k",
        "MAGPIE_EVAL_INCLUDE_PATH": "custom-tasks",
        "EVAL_CONCURRENT_REQUESTS": "8",
        "LM_EVAL_ARGS_FILE": str(args_file),
        "MODEL": "test-model",
    }
    subprocess.run(["bash", "-c", shell], check=True, env=env, cwd=work)
    args = args_file.read_text()
    resolved = str(include.resolve())
    assert f"--include_path {resolved}" in args
    assert "--include_path custom-tasks\n" not in args
    assert "--include_path custom-tasks " not in args


def test_persisted_eval_skips_stock_inferencex_include_for_builtins(tmp_path: Path):
    args_file = tmp_path / "lm_eval.args"
    python_stub = _lm_eval_python_stub(tmp_path, args_file)
    include = tmp_path / "utils" / "evals"
    include.mkdir(parents=True)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    shell = r'''
source "$MAGPIE_COMPAT"
_write_lm_eval_meta_json() {
  printf '{"model":"test","conc":%s}\n' "$3" > "$1"
}
magpie_run_eval_persisted --framework lm-eval --port 8888
'''
    env = {
        **os.environ,
        "MAGPIE_COMPAT": str(_compat_script()),
        "RESULT_DIR": str(tmp_path),
        "EVAL_RESULT_DIR": str(source_dir),
        "MAGPIE_EVAL_PYTHON": str(python_stub),
        "MAGPIE_ACCURACY_REPORT_PYTHON": sys.executable,
        "MAGPIE_EVAL_TASKS": "gpqa_diamond_cot_n_shot",
        "MAGPIE_EVAL_TASK_PATH": str(include),
        "MAGPIE_EVAL_INCLUDE_PATH": str(include),
        "EVAL_CONCURRENT_REQUESTS": "8",
        "LM_EVAL_ARGS_FILE": str(args_file),
        "MODEL": "test-model",
    }
    subprocess.run(["bash", "-c", shell], check=True, env=env)
    args = args_file.read_text()
    assert "--tasks gpqa_diamond_cot_n_shot" in args
    assert "--include_path" not in args


def test_accuracy_report_prefers_humaneval_create_test_metric(tmp_path: Path):
    args_file = tmp_path / "lm_eval.args"
    python_stub = _lm_eval_python_stub(tmp_path, args_file)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    shell = r'''
source "$MAGPIE_COMPAT"
_write_lm_eval_meta_json() {
  printf '{"model":"test","conc":%s}\n' "$3" > "$1"
}
magpie_run_eval_persisted --framework lm-eval --port 8888
'''
    env = {
        **os.environ,
        "MAGPIE_COMPAT": str(_compat_script()),
        "RESULT_DIR": str(tmp_path),
        "EVAL_RESULT_DIR": str(source_dir),
        "MAGPIE_EVAL_PYTHON": str(python_stub),
        "MAGPIE_ACCURACY_REPORT_PYTHON": sys.executable,
        "MAGPIE_EVAL_TASKS": "humaneval_instruct",
        "EVAL_CONCURRENT_REQUESTS": "8",
        "LM_EVAL_ARGS_FILE": str(args_file),
        "MODEL": "test-model",
    }
    env.pop("HF_ALLOW_CODE_EVAL", None)
    subprocess.run(["bash", "-c", shell], check=True, env=env)
    summary = json.loads((tmp_path / "accuracy_report.json").read_text())
    assert summary["status"] == "COMPLETED"
    assert summary["task"] == "humaneval_instruct"
    assert summary["metric"] == "pass@1,create_test"
    assert summary["score"] == 0.7
