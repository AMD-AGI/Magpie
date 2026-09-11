"""Container metadata must not exclude a multi-architecture gfx1151 build."""

import shutil
import subprocess
from pathlib import Path

import pytest

from Magpie.modes.benchmark.image_selector import ImageSelector

SCRIPTS = Path(__file__).resolve().parents[1] / "Magpie" / "scripts" / "benchmark"
PUBLISHED_ARCHES = (
    "gfx90a;gfx942;gfx950;gfx1100;gfx1101;gfx1200;gfx1201;gfx1150;gfx1151"
)


@pytest.fixture(params=("vllm", "sglang"))
def runner(tmp_path, request):
    """Run the real shell script with CPU-only serving/benchmark boundaries."""
    framework = request.param
    script = tmp_path / f"{framework}_radeon8060s.sh"
    shutil.copy2(SCRIPTS / script.name, script)
    (tmp_path / "benchmark_lib.sh").write_text(
        'check_env_vars() { for name in "$@"; do '
        '[[ -n "${!name}" ]] || return 2; done; }\n'
        'wait_for_server_ready() { wait "${@: -1}"; }\n'
        "run_benchmark_serving() { "
        "printf 'CLIENT_ARCH=%s\\n' \"${PYTORCH_ROCM_ARCH:-}\"; }\n",
        encoding="utf-8",
    )
    (tmp_path / "server_cleanup.sh").write_text(
        "magpie_stop_benchmark_server_stack() { :; }\n", encoding="utf-8"
    )
    (tmp_path / "magpie_bench_remote_compat.sh").write_text("", encoding="utf-8")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("vllm", "python3", "hf"):
        command = bindir / name
        command.write_text(
            "#!/bin/bash\nprintf 'SERVER_ARCH=%s\\n' \"${PYTORCH_ROCM_ARCH:-}\"\n",
            encoding="utf-8",
        )
        command.chmod(0o755)

    def run(phase, arch):
        # Deliberately do not inherit caller architecture, remote URL, or args.
        env = {
            "PATH": f"{bindir}:/usr/bin:/bin",
            "MAGPIE_RUN_PHASE": phase,
            "MAGPIE_SERVER_PID_FILE": str(tmp_path / "server.pid"),
            "MODEL": "test-model",
            "TP": "1",
            "CONC": "1",
            "ISL": "64",
            "OSL": "16",
            "RANDOM_RANGE_RATIO": "1.0",
            "RESULT_FILENAME": "result",
            "RESULT_DIR": str(tmp_path),
            "RUN_EVAL": "false",
        }
        if arch is not None:
            env["PYTORCH_ROCM_ARCH"] = arch
        result = subprocess.run(
            ["bash", str(script)],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        log = tmp_path / "server.log"
        return result, log.read_text(encoding="utf-8") if log.exists() else ""

    return run


@pytest.mark.parametrize("phase", ("server", "client", "all"))
@pytest.mark.parametrize(
    "arch",
    (None, "", "gfx1151", PUBLISHED_ARCHES, "gfx1151;gfx950", "gfx950;gfx1151;gfx942"),
)
def test_runner_accepts_and_preserves_gfx1151_build_lists(runner, phase, arch):
    result, log = runner(phase, arch)
    assert result.returncode == 0, result.stderr
    expected_arch = arch or "gfx1151"
    if phase in ("server", "all"):
        assert f"SERVER_ARCH={expected_arch}\n" in log
    if phase in ("client", "all"):
        assert f"CLIENT_ARCH={expected_arch}\n" in result.stdout


@pytest.mark.parametrize("phase", ("server", "client", "all"))
@pytest.mark.parametrize("arch", ("gfx950", "gfx11510", "notgfx1151", "gfx950;gfx942"))
def test_runner_rejects_build_lists_without_exact_gfx1151_entry(runner, phase, arch):
    result, log = runner(phase, arch)
    assert result.returncode == 2, result.stderr
    assert "requires PYTORCH_ROCM_ARCH=gfx1151" in result.stderr
    assert "SERVER_ARCH=" not in log
    assert "CLIENT_ARCH=" not in result.stdout


def test_default_vllm_image_supports_gfx1151():
    selector = ImageSelector()
    assert selector.select_image("vllm", "gfx1151") == "vllm/vllm-openai-rocm:v0.23.0"
    assert selector.get_runner_type("gfx1151") == "radeon8060s"


def test_sglang_gfx1151_requires_explicit_image():
    with pytest.raises(ValueError, match="use --docker-image to override"):
        ImageSelector().select_image("sglang", "gfx1151")


def test_sglang_gfx1151_accepts_explicit_image():
    image = "example/sglang-gfx1151:test"
    assert (
        ImageSelector().select_image("sglang", "gfx1151", override_image=image) == image
    )
