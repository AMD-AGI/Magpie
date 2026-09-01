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
