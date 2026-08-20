import subprocess

import pytest

import Magpie.utils.gpu as gpu
from Magpie.utils.gpu import (
    GPUComputeSpec,
    GPUConfig,
    GPUController,
    GPUHardwareInfo,
    GPUVendor,
    MultiGPUConfig,
    MultiGPUController,
)

ROCMINFO = """
***** Agent 1 *****
  Device Type: GPU
  Marketing Name: MI300X
  Uuid: GPU-abc
  Chip ID: 123(0x74a1)
  Compute Unit: 304
  SIMDs per CU: 4
  Shader Engines: 8
  Wavefront Size: 64
  Workgroup Max Size: 1024
  Max Waves Per CU: 32
  Max Work-item Per CU: 2048
  Cacheline Size: 64
  Workgroup Max Size per Dimension:
    x 1024 foo
    y 1024 foo
    z 1024
  Grid Max Size: 4294967295
  Grid Max Size per Dimension:
    x 1 foo
    y 2 foo
    z 3
  Cache Info:
    L1: 32
    L2: 16384
    L3: 262144
  Segment: GROUP
    Size: 64
  Name: amdgcn-amd-amdhsa--gfx942
***** Agent 2 *****
  Device Type: CPU
"""


def test_gpu_models_and_rocminfo_parser():
    specs = gpu._parse_rocminfo_gpu_agents(ROCMINFO)
    assert len(specs) == 1
    spec = specs[0]
    assert spec.compute_units == 304
    assert spec.max_workgroup_size_xyz == [1024, 1024, 1024]
    assert spec.grid_max_size_xyz == [1, 2, 3]
    assert spec.l2_cache_kb == 16384
    assert spec.lds_size_kb == 64
    assert spec.isa_name.endswith("gfx942")
    assert GPUComputeSpec(compute_units=1).to_dict() == {"compute_units": 1}
    info = GPUHardwareInfo(GPUVendor.AMD, compute_spec=spec)
    assert info.to_dict()["compute_spec"]["marketing_name"] == "MI300X"
    restored = gpu._spec_from_cache_dict(spec.to_dict())
    assert restored.compute_units == 304


def test_compute_spec_disk_memory_and_live_caches(tmp_path, monkeypatch):
    cache_dir = tmp_path / ".magpie"
    cache_file = cache_dir / "specs.json"
    monkeypatch.setattr(gpu, "_MAGPIE_CACHE_DIR", cache_dir)
    monkeypatch.setattr(gpu, "_COMPUTE_SPEC_CACHE_FILE", cache_file)
    monkeypatch.setattr(gpu, "_rocminfo_specs", None)
    spec = GPUComputeSpec(compute_units=8)
    gpu._save_compute_spec_cache([spec])
    assert gpu._load_compute_spec_cache()["gpu_count"] == 1
    assert gpu.get_amd_compute_specs()[0].compute_units == 8
    monkeypatch.setattr(gpu, "_rocminfo_specs", None)
    cache_file.unlink()
    monkeypatch.setattr(
        gpu.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout=ROCMINFO),
    )
    assert gpu.get_amd_compute_specs(force_refresh=True)[0].marketing_name == "MI300X"


def controller(vendor, monkeypatch, device_id=0):
    monkeypatch.setattr(gpu, "detect_gpu", lambda: (vendor, "gfx942"))
    return GPUController(device_id)


def test_gpu_controller_hardware_info_amd(monkeypatch):
    ctrl = controller(GPUVendor.AMD, monkeypatch)
    monkeypatch.setattr(
        gpu, "get_amd_compute_specs", lambda: [GPUComputeSpec(marketing_name="MI300X")]
    )
    outputs = {
        "--showpower": "Average Power (W): 150.0",
        "--showclocks": "sclk 1200\nmclk 900",
        "--showtemp": "Temperature: 55.5",
        "--showmeminfo": f"Total {16 * 1024**3}\nUsed {4 * 1024**3}",
    }

    def run(cmd, **kwargs):
        key = next(key for key in outputs if key in cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=outputs[key])

    monkeypatch.setattr(gpu.subprocess, "run", run)
    info = ctrl.get_hardware_info()
    assert info.device_name == "MI300X"
    assert info.power_current_watts == 150
    assert info.gpu_clock_current == 1200
    assert info.mem_clock_current == 900
    assert info.temperature == 55.5
    assert info.memory_total_gb == 16


def test_gpu_controller_hardware_info_nvidia_and_unknown(monkeypatch):
    ctrl = controller(GPUVendor.NVIDIA, monkeypatch)
    values = "H100,100,700,800,1200,1800,900,1200,50,81920,1024"
    monkeypatch.setattr(
        gpu.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout=values),
    )
    info = ctrl.get_hardware_info()
    assert info.device_name == "H100"
    assert info.memory_total_gb == 80
    ctrl.vendor = GPUVendor.UNKNOWN
    assert ctrl.get_hardware_info().vendor is GPUVendor.UNKNOWN


@pytest.mark.parametrize("vendor", [GPUVendor.AMD, GPUVendor.NVIDIA])
def test_gpu_controller_apply_and_reset(vendor, monkeypatch):
    ctrl = controller(vendor, monkeypatch)
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok")

    monkeypatch.setattr(gpu.subprocess, "run", run)
    config = GPUConfig(
        power_limit_watts=300,
        gpu_clock_mhz=(1000, 1200),
        mem_clock_mhz=(800, 900),
        gpu_clock_level=4,
        mem_clock_level=2,
    )
    assert ctrl.apply_config(config) is True
    assert ctrl.reset_config() is True
    assert len(calls) >= 4 if vendor is GPUVendor.NVIDIA else len(calls) >= 4
    monkeypatch.setattr(
        gpu.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(OSError("permission")),
    )
    assert ctrl.apply_config(config) is False
    assert ctrl.reset_config() is False
    ctrl.vendor = GPUVendor.UNKNOWN
    assert ctrl.apply_config(config) is False
    assert ctrl.reset_config() is False


def test_gpu_detection_info_and_counts(monkeypatch):
    monkeypatch.setattr(gpu, "_detect_amd_gpu", lambda: "gfx942")
    assert gpu.detect_gpu() == (GPUVendor.AMD, "gfx942")
    assert gpu.get_gpu_info()["compiler"] == "hipcc"
    monkeypatch.setattr(gpu, "_detect_amd_gpu", lambda: None)
    monkeypatch.setattr(gpu, "_detect_nvidia_gpu", lambda: "sm_90")
    assert gpu.detect_gpu() == (GPUVendor.NVIDIA, "sm_90")
    assert gpu.get_gpu_info()["compiler"] == "nvcc"
    monkeypatch.setattr(gpu, "_detect_nvidia_gpu", lambda: None)
    assert gpu.detect_gpu() == (GPUVendor.UNKNOWN, None)

    responses = iter(
        [
            subprocess.CompletedProcess([], 0, stdout="1\n1\n"),
            subprocess.CompletedProcess([], 1, stdout=""),
            subprocess.CompletedProcess([], 0, stdout="GPU[0]\nGPU[1]\nGPU[1]"),
        ]
    )
    monkeypatch.setattr(gpu.subprocess, "run", lambda *a, **k: next(responses))
    assert gpu.get_gpu_count() == 2
    assert gpu.get_gpu_count() == 2


def test_low_level_detection_parsers(monkeypatch):
    monkeypatch.setattr(
        gpu.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd, 0, stdout="Name: gfx950" if cmd[0] == "rocminfo" else "9.0\n"
        ),
    )
    assert gpu._detect_amd_gpu() == "gfx950"
    assert gpu._detect_nvidia_gpu() == "sm_90"


def test_busy_and_memory_parsers(monkeypatch):
    outputs = {
        "--showpidgpus": "PID 10\n10 device(s): 0 2\n",
        "--showmeminfo": f"device,total,used\nGPU[0],{8 * 1024**3},{2 * 1024**3}\n",
        "--query-gpu=index,uuid": "0, GPU-a\n1, GPU-b\n",
        "--query-compute-apps=gpu_uuid,pid": "GPU-b, 123\n",
        "--query-gpu=index,memory.used,memory.total": "0, 1024, 8192\n",
    }

    def run(cmd, **kwargs):
        key = next(key for key in outputs if key in " ".join(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout=outputs[key])

    monkeypatch.setattr(gpu.subprocess, "run", run)
    assert gpu._amd_busy_gpu_ids() == {0, 2}
    assert gpu._amd_gpu_memory_usage()[0] == (2, 8)
    assert gpu._nvidia_busy_gpu_ids() == {1}
    assert gpu._nvidia_gpu_memory_usage()[0] == (1, 8)


def test_find_idle_gpus_paths(monkeypatch):
    monkeypatch.setattr(gpu, "detect_gpu", lambda: (GPUVendor.AMD, "gfx942"))
    monkeypatch.setattr(gpu, "get_gpu_count", lambda: 3)
    monkeypatch.setattr(gpu, "_amd_busy_gpu_ids", lambda: {1})
    monkeypatch.setattr(
        gpu,
        "_amd_gpu_memory_usage",
        lambda: {0: (2, 8), 1: (1, 8), 2: (7.5, 8)},
    )
    assert gpu.find_idle_gpus(1) == [0]
    assert gpu.find_idle_gpus(0) == []
    with pytest.raises(RuntimeError, match="no idle"):
        gpu.find_idle_gpus(1, min_free_memory_gb=7)
    monkeypatch.setattr(gpu, "get_gpu_count", lambda: 0)
    with pytest.raises(RuntimeError, match="no GPUs"):
        gpu.find_idle_gpus()


def test_find_idle_gpus_nvidia_candidates_shortage_and_vendor(monkeypatch):
    monkeypatch.setattr(gpu, "detect_gpu", lambda: (GPUVendor.NVIDIA, "sm_90"))
    monkeypatch.setattr(gpu, "get_gpu_count", lambda: 4)
    monkeypatch.setattr(gpu, "_nvidia_busy_gpu_ids", lambda: {1})
    monkeypatch.setattr(
        gpu,
        "_nvidia_gpu_memory_usage",
        lambda: {0: (7, 8), 1: (0, 8), 2: (2, 8), 3: (1, 8)},
    )
    assert gpu.find_idle_gpus(3, min_free_memory_gb=2, candidates=[-1, 0, 1, 2, 9]) == [
        2
    ]
    monkeypatch.setattr(gpu, "detect_gpu", lambda: (GPUVendor.UNKNOWN, ""))
    with pytest.raises(RuntimeError, match="unsupported"):
        gpu.find_idle_gpus()


@pytest.mark.parametrize(
    "helper", ["amd_busy", "amd_memory", "nvidia_busy", "nvidia_memory"]
)
def test_gpu_process_memory_helpers_command_failures(monkeypatch, helper):
    if helper == "nvidia_busy":
        monkeypatch.setattr(
            gpu.subprocess,
            "run",
            lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()),
        )
        assert gpu._nvidia_busy_gpu_ids() == set()
    elif helper == "nvidia_memory":
        monkeypatch.setattr(
            gpu.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0], 1, stdout=""),
        )
        assert gpu._nvidia_gpu_memory_usage() == {}
    elif helper == "amd_busy":
        monkeypatch.setattr(
            gpu.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(
                a[0], 0, stdout="No KFD PIDs currently running"
            ),
        )
        assert gpu._amd_busy_gpu_ids() == set()
    else:
        monkeypatch.setattr(
            gpu.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(
                a[0], 0, stdout="device,total,used\ninvalid\nGPU[x],1,1\nGPU[0],bad,1\n"
            ),
        )
        assert gpu._amd_gpu_memory_usage() == {}


class FakeController:
    def __init__(self, device_id):
        self.device_id = device_id

    def get_hardware_info(self):
        return GPUHardwareInfo(GPUVendor.AMD, device_id=self.device_id)

    def apply_config(self, _config):
        return True

    def reset_config(self):
        return True


def test_multi_gpu_configuration_and_controller(monkeypatch, capsys):
    monkeypatch.setattr(gpu, "get_gpu_count", lambda: 2)
    monkeypatch.setattr(gpu, "GPUController", FakeController)
    controller = MultiGPUController()
    assert set(controller.get_all_hardware_info(parallel=False)) == {0, 1}
    config = MultiGPUConfig(
        default_config=GPUConfig(power_limit_watts=300), parallel=False
    )
    assert config.get_config_for_device(1).device_id == 1
    assert controller.apply_config(config) == {0: True, 1: True}
    assert controller.reset_all(parallel=False) == {0: True, 1: True}
    controller.print_summary()
    assert "GPU Summary" in capsys.readouterr().out


def test_multi_gpu_parallel_paths_invalid_devices_and_errors(monkeypatch):
    class SometimesBroken(FakeController):
        def get_hardware_info(self):
            if self.device_id == 1:
                raise RuntimeError("query failed")
            return super().get_hardware_info()

        def apply_config(self, config):
            return self.device_id == 0

    monkeypatch.setattr(gpu, "get_gpu_count", lambda: 2)
    monkeypatch.setattr(gpu, "GPUController", SometimesBroken)
    controller = MultiGPUController(device_ids=[-1, 0, 1, 5])
    infos = controller.get_all_hardware_info(parallel=True)
    assert infos[1].vendor is GPUVendor.UNKNOWN
    config = MultiGPUConfig(
        default_config=GPUConfig(power_limit_watts=300),
        device_ids=[0, 1, 5],
        parallel=True,
    )
    assert controller.apply_config(config) == {0: True, 1: False, 5: False}
    assert controller.reset_all(parallel=True) == {0: True, 1: True}
    assert MultiGPUConfig().get_config_for_device(0) is None


def test_list_gpus_empty_and_populated(monkeypatch):
    monkeypatch.setattr(gpu, "get_gpu_count", lambda: 0)
    assert gpu.list_gpus() == []
    monkeypatch.setattr(gpu, "get_gpu_count", lambda: 2)
    monkeypatch.setattr(gpu, "GPUController", FakeController)
    assert [info.device_id for info in gpu.list_gpus()] == [0, 1]


def test_load_gpu_config_and_reset_default(monkeypatch):
    monkeypatch.setattr(gpu, "get_gpu_count", lambda: 2)
    assert gpu.load_gpu_config_from_dict({}) == ([0, 1], None)
    config = {
        "gpu": {
            "device_ids": [0, 1],
            "parallel": False,
            "hardware": {
                "enabled": True,
                "reset_after_benchmark": False,
                "default": {
                    "power": {"limit_watts": 300},
                    "frequency": {
                        "gpu_clock_mhz": [1000, 1200],
                        "mem_clock_mhz": [800, 900],
                    },
                },
                "per_gpu": {"1": {"power": {"limit_watts": 250}}},
            },
        }
    }
    devices, multi = gpu.load_gpu_config_from_dict(config)
    assert devices == [0, 1]
    assert multi.default_config.gpu_clock_mhz == (1000, 1200)
    assert multi.gpu_configs[1].power_limit_watts == 250
    assert gpu.get_reset_after_benchmark(config) is False
    assert gpu.get_reset_after_benchmark({}) is True
