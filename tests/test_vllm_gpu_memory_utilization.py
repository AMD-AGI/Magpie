"""The MI300X/MI355X vLLM launchers honor the caller's GPU memory utilization."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "Magpie/scripts/benchmark"
SCRIPT_NAMES = ("vllm_mi300x.sh", "vllm_mi355x.sh")
CAPTURE = """import json, os, sys
from pathlib import Path

Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))
"""
LIBRARY = r"""
check_env_vars() { :; }
hf() { :; }
rocm-smi() { printf 'MEC 999\n'; }
setsid() { "$TEST_PYTHON" "$SERVER_CAPTURE_SCRIPT" "$@"; }
wait_for_server_ready() { wait "$SERVER_PID"; }
"""


def _server_argv(tmp_path, script_name, overrides):
    scripts = tmp_path / "benchmarks"
    shutil.copytree(SCRIPTS, scripts)
    (scripts / "benchmark_lib.sh").write_text(LIBRARY)
    capture_script = tmp_path / "capture-server.py"
    capture_script.write_text(CAPTURE)
    env = {
        "PATH": os.environ["PATH"],
        "PYTHONDONTWRITEBYTECODE": "1",
        "TEST_PYTHON": sys.executable,
        "SERVER_CAPTURE_SCRIPT": str(capture_script),
        "MAGPIE_RUN_PHASE": "server",
        "MAGPIE_SERVER_PID_FILE": str(tmp_path / "server.pid"),
        "CAPTURE": str(tmp_path / "captured.json"),
        "MODEL": "test-model",
        "TP": "2",
        "RESULT_DIR": str(tmp_path),
        "SERVER_LOG": str(tmp_path / "server.log"),
        **overrides,
    }
    result = subprocess.run(
        ["bash", str(scripts / script_name)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((tmp_path / "captured.json").read_text())


@pytest.mark.parametrize("script_name", SCRIPT_NAMES)
@pytest.mark.parametrize(
    ("overrides", "expected"),
    [({}, "0.95"), ({"GPU_MEMORY_UTILIZATION": "0.8"}, "0.8")],
    ids=["default", "explicit"],
)
def test_server_uses_gpu_memory_utilization(tmp_path, script_name, overrides, expected):
    argv = _server_argv(tmp_path, script_name, overrides)

    assert argv[:2] == ["vllm", "serve"]
    assert argv[argv.index("--gpu-memory-utilization") + 1] == expected
