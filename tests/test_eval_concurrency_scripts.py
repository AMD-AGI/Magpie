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
    assert 'run_eval --framework lm-eval --port "$PORT" || exit $?' in contents
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


@pytest.mark.parametrize("script_name", LOCAL_EVAL_SCRIPTS)
def test_local_eval_scripts_persist_accuracy_results(script_name: str):
    script = ROOT / "Magpie" / "scripts" / "benchmark" / script_name
    contents = script.read_text(encoding="utf-8")

    assert "magpie_run_eval_persisted --framework lm-eval" in contents


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
  printf '%s\n' '{"model":"test"}' > "$EVAL_RESULT_DIR/meta_env.json"
  find "$EVAL_RESULT_DIR" -type f -name '*.json*' -exec mv -f {} . \;
  rm -rf "$EVAL_RESULT_DIR"
}
magpie_run_eval_persisted --framework lm-eval --port 8888
'''
    env = {
        **os.environ,
        "MAGPIE_COMPAT": str(compat),
        "RESULT_DIR": str(tmp_path),
        "FAKE_EVAL_RESULT": json.dumps(result_payload),
        "MAGPIE_EVAL_TASKS": "gsm8k",
    }
    subprocess.run(["bash", "-c", shell], check=True, env=env)

    eval_dir = tmp_path / "lm_eval"
    assert json.loads((eval_dir / "results_123.json").read_text()) == result_payload
    assert (eval_dir / "meta_env.json").is_file()
    summary = json.loads((eval_dir / "accuracy_result.json").read_text())
    assert summary["status"] == "COMPLETED"
    assert summary["task"] == "gsm8k"
    assert summary["metric"] == "exact_match,strict-match"
    assert summary["score"] == 0.98
    assert summary["samples"] == 100
    assert summary["source_result"] == "results_123.json"
