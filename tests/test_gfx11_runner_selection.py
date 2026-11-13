"""gfx1100 selects the gfx11 runner by ISA; only the Radeon PRO W7900 is validated.

This host class includes real W7900 boards, so every test injects the detected
ISA and the rocminfo boards; the autouse fixture makes a live probe fail loudly.
"""

import logging
from pathlib import Path

import pytest
import yaml

from Magpie.modes.benchmark import benchmarker as benchmarker_module
from Magpie.modes.benchmark import image_selector as image_selector_module
from Magpie.modes.benchmark import tracelens_inference as tracelens_inference_module
from Magpie.modes.benchmark.benchmarker import MAGPIE_BUILTIN_SCRIPTS, BenchmarkMode
from Magpie.modes.benchmark.config import BenchmarkConfig
from Magpie.modes.benchmark.image_selector import ImageSelector
from Magpie.modes.benchmark.tracelens_inference import trace_arch_platform_from_runner
from Magpie.utils.gpu import GPUComputeSpec, GPUVendor

IMAGES_YAML = Path(__file__).resolve().parents[1] / "Magpie" / "benchmark_images.yaml"
W7900_IMAGE = "rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0"
W7900 = "AMD Radeon Pro W7900"
W7900_DS = "AMD Radeon Pro W7900 Dual Slot"
W7800 = "AMD Radeon Pro W7800"
RX_7900_XTX = "AMD Radeon RX 7900 XTX"
R9700 = "AMD Radeon AI PRO R9700"
VALIDATION_WARNING = "The gfx11 runner is validated only on the Radeon PRO W7900"


def board(name, isa="gfx1100"):
    return GPUComputeSpec(marketing_name=name, isa_name=f"amdgcn-amd-amdhsa--{isa}")


@pytest.fixture(autouse=True)
def isolate_from_host_gpu(monkeypatch):
    def live_probe(*_args, **_kwargs):
        raise AssertionError("a test probed the host GPU")

    for module in (image_selector_module, benchmarker_module, tracelens_inference_module):
        monkeypatch.setattr(module, "detect_gpu", live_probe)
    monkeypatch.setattr(benchmarker_module, "get_amd_compute_specs", live_probe)


def use_host(monkeypatch, *boards, arch="gfx1100"):
    """Detect ``arch`` and list ``boards`` from rocminfo; return the refresh flags read."""
    refreshes = []

    def compute_specs(force_refresh=False):
        refreshes.append(force_refresh)
        return list(boards)

    for module in (image_selector_module, benchmarker_module):
        monkeypatch.setattr(module, "detect_gpu", lambda: (GPUVendor.AMD, arch))
    monkeypatch.setattr(benchmarker_module, "get_amd_compute_specs", compute_specs)
    return refreshes


def benchmark_mode(tmp_path, framework="vllm", **config):
    mode = BenchmarkMode(
        BenchmarkConfig.from_dict(
            {
                "framework": framework,
                "model": "demo",
                "run_mode": "docker",
                "inferencex_path": str(tmp_path / "InferenceX"),
                "profiler": {
                    "torch_profiler": {"enabled": False},
                    "gpu_monitor": {"enabled": False},
                },
                **config,
            }
        ),
        output_dir=str(tmp_path / "results"),
    )
    (tmp_path / "InferenceX" / "benchmarks").mkdir(parents=True, exist_ok=True)
    mode._prepare_benchmark_scripts()
    return mode


def validation_warnings(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and VALIDATION_WARNING in record.getMessage()
    ]


@pytest.fixture
def warnings_log(caplog):
    caplog.set_level(logging.WARNING, logger=benchmarker_module.__name__)
    return caplog


# --- Runner and image -------------------------------------------------------


def test_gfx1100_maps_to_the_gfx11_runner():
    selector = ImageSelector(str(IMAGES_YAML))

    assert selector.get_runner_type("gfx1100") == "gfx11"


def test_vllm_on_gfx1100_uses_the_r9700_image():
    images = yaml.safe_load(IMAGES_YAML.read_text())
    selector = ImageSelector(str(IMAGES_YAML))

    assert images["vllm"]["gfx1100"] == images["vllm"]["gfx1201"] == W7900_IMAGE
    assert selector.select_image("vllm", "gfx1100") == W7900_IMAGE


def test_detected_gfx1100_selects_the_gfx11_script_image_and_runner(monkeypatch, tmp_path):
    use_host(monkeypatch, board(W7900_DS))
    mode = benchmark_mode(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    runner = mode._get_runner_type()
    image = mode._select_image()
    command = mode._build_docker_command(image, workspace, runner)

    assert runner == "gfx11"
    assert "vllm_gfx11.sh" in MAGPIE_BUILTIN_SCRIPTS
    assert mode._get_benchmark_script(runner) == "benchmarks/vllm_gfx11.sh"
    assert image == W7900_IMAGE
    assert "RUNNER_TYPE=gfx11" in command
    assert not any(item.startswith("TARGET_GPU_TYPE=") for item in command)


@pytest.mark.parametrize("framework", ["sglang", "atom"])
def test_sglang_and_atom_are_refused_on_gfx11(monkeypatch, tmp_path, framework):
    use_host(monkeypatch, board(W7900_DS))
    monkeypatch.setattr(benchmarker_module, "ensure_inferencex_available", lambda path: path)
    mode = benchmark_mode(tmp_path, framework=framework, gpu_selection={"auto": False})
    for name in ("_select_image", "_build_docker_command", "_execute_benchmark"):
        monkeypatch.setattr(
            mode, name, lambda *a, _name=name, **k: pytest.fail(f"{_name} ran after a refusal")
        )

    result = mode.run()

    assert result.success is False
    assert any(
        f"No benchmark script found for framework={framework}" in error and "gpu=gfx11" in error
        for error in result.errors
    ), result.errors
    images = yaml.safe_load(IMAGES_YAML.read_text())
    assert "gfx1100" not in images[framework]


# --- Validation warning -----------------------------------------------------


@pytest.mark.parametrize(
    "boards",
    [
        [board(W7900)],
        [board(W7900_DS)],
        [board("AMD Radeon PRO W7900")],
        [board(W7900), board(W7900_DS)],
        [board(R9700, isa="gfx1201"), board(W7900_DS)],
    ],
    ids=["w7900", "w7900-dual-slot", "uppercase-pro", "both-variants", "beside-an-r9700"],
)
def test_a_w7900_runs_without_a_warning(monkeypatch, tmp_path, warnings_log, boards):
    refreshes = use_host(monkeypatch, *boards)

    assert benchmark_mode(tmp_path)._get_runner_type() == "gfx11"
    assert refreshes == [True]
    assert validation_warnings(warnings_log) == []


@pytest.mark.parametrize(
    ("boards", "found"),
    [
        ([board(W7800)], W7800),
        ([board(RX_7900_XTX)], RX_7900_XTX),
        ([board(None)], "an unnamed board"),
        ([board(W7900_DS), board(W7800)], f"{W7800}, {W7900_DS}"),
        ([], "no gfx1100 board in rocminfo"),
    ],
    ids=["w7800", "rx-7900-xtx", "unnamed", "w7900-and-w7800", "no-rocminfo"],
)
def test_any_other_gfx1100_board_runs_with_a_warning(
    monkeypatch, tmp_path, warnings_log, boards, found
):
    use_host(monkeypatch, *boards)

    assert benchmark_mode(tmp_path)._get_runner_type() == "gfx11"
    assert validation_warnings(warnings_log) == [f"{VALIDATION_WARNING}; found {found}."]


def test_an_explicit_gfx11_runner_is_checked_too(monkeypatch, tmp_path, warnings_log):
    use_host(monkeypatch, board(W7800), arch=None)

    assert benchmark_mode(tmp_path, runner_type="gfx11")._get_runner_type() == "gfx11"
    assert validation_warnings(warnings_log) == [f"{VALIDATION_WARNING}; found {W7800}."]


def test_other_runners_never_read_the_boards(monkeypatch, tmp_path, warnings_log):
    use_host(monkeypatch, board(R9700, isa="gfx1201"), arch="gfx1201")
    monkeypatch.setattr(
        benchmarker_module,
        "get_amd_compute_specs",
        lambda **_kwargs: pytest.fail("only the gfx11 runner reads the boards"),
    )

    assert benchmark_mode(tmp_path)._get_runner_type() == "gfx12"
    assert validation_warnings(warnings_log) == []


# --- TraceLens --------------------------------------------------------------


@pytest.mark.parametrize("runner", ["gfx11", "gfx1100", "GFX11"])
def test_tracelens_has_no_platform_for_the_runner_or_the_isa(runner):
    assert trace_arch_platform_from_runner(runner) is None


def test_tracelens_autodetected_gfx1100_has_no_platform(monkeypatch):
    monkeypatch.setattr(
        tracelens_inference_module, "detect_gpu", lambda: (GPUVendor.AMD, "gfx1100")
    )

    assert trace_arch_platform_from_runner(None) is None
