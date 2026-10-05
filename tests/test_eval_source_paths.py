"""Generic scripts keep sourced eval resources valid after persisted eval changes cwd."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "Magpie" / "scripts" / "benchmark"
EVAL_SCRIPTS = [
    path.name
    for path in sorted(SCRIPTS.glob("*.sh"))
    if '/benchmark_lib.sh"' in path.read_text(encoding="utf-8")
]

# InferenceX 3d5581562 resolves these paths inside functions, after Magpie's
# persisted eval wrapper changes cwd. Keep real Bash source-stack resolution;
# a stub that captures the root at source time would conceal the regression.
BENCHMARK_LIB = r'''
_eval_patches_dir() {
    cd "$(dirname "${BASH_SOURCE[0]}")/../utils/evals/patches" && pwd
}

run_eval() {
    local _repo_root
    _repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 1
    [[ "$_repo_root" == "$EXPECTED_REPO_ROOT" ]] || return 2
    [[ "$PWD" == "$EVAL_RESULT_DIR" ]] || return 3
    cp "$(_eval_patches_dir)/lm_eval_sitecustomize.py" "$EVAL_RESULT_DIR/sitecustomize.py" || return 4
    cp "$_repo_root/utils/evals/gsm8k.yaml" "$EVAL_RESULT_DIR/task.yaml" || return 5
    printf '%s\n' '{"results":{"gsm8k":{"exact_match,strict-match":0.5}}}' > "$EVAL_RESULT_DIR/results_fixture.json"
}

# Exercise the entrypoint's actual dependency loading and persisted eval
# wrapper, exiting before any server, GPU command, or throughput client runs.
check_env_vars() {
    magpie_run_eval_persisted --framework lm-eval --port 8888
    local eval_rc=$?
    [[ "$PWD" == "$EXPECTED_REPO_ROOT" ]] || exit 6
    exit "$eval_rc"
}
'''


@pytest.mark.parametrize("script_name", EVAL_SCRIPTS)
@pytest.mark.parametrize("absolute_entrypoint", [False, True])
def test_persisted_eval_keeps_deferred_source_paths(
    tmp_path: Path, script_name: str, absolute_entrypoint: bool
):
    checkout = tmp_path / "InferenceX checkout"
    scripts = checkout / "benchmarks"
    shutil.copytree(SCRIPTS, scripts)
    library = scripts / "benchmark_lib.sh"
    library.write_text(BENCHMARK_LIB, encoding="utf-8")
    patches = checkout / "utils" / "evals" / "patches"
    patches.mkdir(parents=True)
    (patches / "lm_eval_sitecustomize.py").write_text("# eval patch fixture\n")
    (patches.parent / "gsm8k.yaml").write_text("task: gsm8k\n")
    result_dir = tmp_path / "session results"
    eval_dir = result_dir / "eval output"
    env = {
        "PATH": os.environ["PATH"],
        "MAGPIE_RUN_PHASE": "client",
        "MAGPIE_ACCURACY_REPORT_PYTHON": sys.executable,
        "RESULT_DIR": str(result_dir),
        "EVAL_RESULT_DIR": str(eval_dir),
        "EXPECTED_REPO_ROOT": str(checkout),
        "MODEL": "fixture-model",
        "CONC": "1",
    }
    entrypoint = scripts / script_name
    if not absolute_entrypoint:
        entrypoint = entrypoint.relative_to(checkout)
    completed = subprocess.run(
        ["bash", str(entrypoint)],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (eval_dir / "sitecustomize.py").read_text() == "# eval patch fixture\n"
    assert (eval_dir / "task.yaml").read_text() == "task: gsm8k\n"
    summary = json.loads((result_dir / "accuracy_report.json").read_text())
    assert summary["status"] == "COMPLETED"
    assert summary["score"] == 0.5
    assert (result_dir / summary["source_result"]).is_file()
    assert library.read_text() == BENCHMARK_LIB
    assert not (checkout / "results_fixture.json").exists()
