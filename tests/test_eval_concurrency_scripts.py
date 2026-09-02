import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
LOCAL_EVAL_SCRIPTS = [
    "atom_mi300x.sh",
    "atom_mi355x.sh",
    "sglang_mi300x.sh",
    "sglang_mi355x.sh",
    "vllm_mi300x.sh",
    "vllm_mi355x.sh",
]
CONCURRENCY_EXPORT = (
    'export EVAL_CONCURRENT_REQUESTS="${MAGPIE_EVAL_CONCURRENCY:-'
    '${EVAL_CONCURRENT_REQUESTS:-$CONC}}"'
)


@pytest.mark.parametrize("script_name", LOCAL_EVAL_SCRIPTS)
def test_local_eval_scripts_use_environment_for_concurrency(script_name: str):
    script = ROOT / "Magpie" / "scripts" / "benchmark" / script_name
    contents = script.read_text(encoding="utf-8")

    assert CONCURRENCY_EXPORT in contents
    assert 'magpie_run_eval_persisted --framework lm-eval --port "$PORT"' in contents
    assert "declare -F magpie_run_eval_persisted" not in contents
    assert "--concurrent-requests" not in contents


def test_remote_eval_prefers_independent_accuracy_concurrency():
    script = (
        ROOT
        / "Magpie"
        / "scripts"
        / "benchmark"
        / "magpie_bench_remote_compat.sh"
    )
    contents = script.read_text(encoding="utf-8")

    assert (
        'local conc="${MAGPIE_EVAL_CONCURRENCY:-'
        '${EVAL_CONCURRENT_REQUESTS:-${CONC:-8}}}"'
    ) in contents


def test_persisted_eval_writes_raw_and_formatted_results(tmp_path: Path):
    compat = (
        ROOT
        / "Magpie"
        / "scripts"
        / "benchmark"
        / "magpie_bench_remote_compat.sh"
    )
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
magpie_run_eval_persisted --framework lm-eval --port 8888
'''
    source_dir = tmp_path / "source"
    env = {
        **os.environ,
        "MAGPIE_COMPAT": str(compat),
        "RESULT_DIR": str(tmp_path),
        "EVAL_RESULT_DIR": str(source_dir),
        "FAKE_EVAL_RESULT": json.dumps(result_payload),
        "MAGPIE_EVAL_TASKS": "gsm8k",
        "CONC": "64",
        "EVAL_CONCURRENT_REQUESTS": "8",
    }
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
    compat = (
        ROOT
        / "Magpie"
        / "scripts"
        / "benchmark"
        / "magpie_bench_remote_compat.sh"
    )
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
magpie_run_eval_persisted --framework lm-eval --port 8888
'''
    env = {
        **os.environ,
        "MAGPIE_COMPAT": str(compat),
        "RESULT_DIR": str(tmp_path),
        "EVAL_CONCURRENT_REQUESTS": "2 4",
        "MAGPIE_EVAL_TASKS": "gsm8k",
    }
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
