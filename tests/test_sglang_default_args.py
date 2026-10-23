"""The SGLang scripts' built-in server defaults must not contradict the caller's own flags."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "Magpie" / "scripts" / "benchmark"


def _server_argv(tmp_path: Path, script_name: str, extra_args: str) -> list[str]:
    """Run the script's server phase with a stub ``setsid`` and return the argv it would launch."""
    script = tmp_path / script_name
    shutil.copy2(SCRIPTS / script_name, script)
    (tmp_path / "benchmark_lib.sh").write_text(
        "check_env_vars() { :; }\nwait_for_server_ready() { wait; }\n", encoding="utf-8"
    )
    (tmp_path / "server_cleanup.sh").write_text(
        "magpie_stop_benchmark_server_stack() { :; }\n", encoding="utf-8"
    )
    (tmp_path / "magpie_bench_remote_compat.sh").write_text("", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    argv_file = tmp_path / "argv"
    setsid = bin_dir / "setsid"
    setsid.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "{argv_file}"\n', encoding="utf-8"
    )
    setsid.chmod(0o755)
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "MAGPIE_RUN_PHASE": "server",
        "MAGPIE_SERVER_PID_FILE": str(tmp_path / "server.pid"),
        "MODEL": "test-model",
        "TP": "4",
        "RESULT_DIR": str(tmp_path),
        "EXTRA_SGLANG_ARGS": extra_args,
    }
    subprocess.run(
        ["bash", str(script)],
        env=env,
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return argv_file.read_text(encoding="utf-8").split()


@pytest.mark.parametrize("script_name", ["sglang_mi300x.sh", "sglang_mi355x.sh"])
def test_hierarchical_cache_recipe_does_not_get_disable_radix_cache(
    tmp_path, script_name
):
    argv = _server_argv(
        tmp_path, script_name, "--enable-hierarchical-cache --hicache-ratio 1.0"
    )
    assert "--enable-hierarchical-cache" in argv
    assert "--disable-radix-cache" not in argv


@pytest.mark.parametrize("script_name", ["sglang_mi300x.sh", "sglang_mi355x.sh"])
def test_plain_recipe_keeps_the_disable_radix_cache_default(tmp_path, script_name):
    argv = _server_argv(tmp_path, script_name, "--kv-cache-dtype fp8_e4m3")
    assert "--disable-radix-cache" in argv
    assert "--mem-fraction-static=0.8" in argv
