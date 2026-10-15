import subprocess
import time

import pytest

import Magpie.utils.gpu_monitor as gpu_monitor
from Magpie.utils.gpu_monitor import GPUMonitor, GPUMonitorStats, GPUSample


def test_detect_vendor_amd_when_amd_smi_lists_a_gpu(monkeypatch):
    monkeypatch.setattr(
        gpu_monitor.amd_smi, "run_json", lambda *a, **k: {"gpu_data": [{"gpu": 0}]}
    )
    monitor = GPUMonitor.__new__(GPUMonitor)
    assert monitor._detect_vendor() == "amd"


def test_detect_vendor_nvidia_when_amd_smi_absent(monkeypatch):
    monkeypatch.setattr(gpu_monitor.amd_smi, "run_json", lambda *a, **k: None)
    monkeypatch.setattr(
        gpu_monitor.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, stdout="GPU 0: H100"),
    )
    monitor = GPUMonitor.__new__(GPUMonitor)
    assert monitor._detect_vendor() == "nvidia"


def test_detect_vendor_unknown_when_neither_tool_reports_a_gpu(monkeypatch):
    monkeypatch.setattr(gpu_monitor.amd_smi, "run_json", lambda *a, **k: None)
    monkeypatch.setattr(
        gpu_monitor.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()),
    )
    monitor = GPUMonitor.__new__(GPUMonitor)
    assert monitor._detect_vendor() == "unknown"


def test_detect_vendor_amd_smi_empty_rows_falls_through_to_nvidia(monkeypatch):
    # amd-smi ran but enumerated no GPUs (e.g. "gpu_data": []): not "amd".
    monkeypatch.setattr(
        gpu_monitor.amd_smi, "run_json", lambda *a, **k: {"gpu_data": []}
    )
    monkeypatch.setattr(
        gpu_monitor.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, stdout="GPU 0: H100"),
    )
    monitor = GPUMonitor.__new__(GPUMonitor)
    assert monitor._detect_vendor() == "nvidia"


def _monitor(monkeypatch, vendor="amd"):
    monkeypatch.setattr(GPUMonitor, "_detect_vendor", lambda self: vendor)
    return GPUMonitor(device_id=0, interval_sec=0.5, vendor="auto")


def test_start_skips_unsupported_vendor(monkeypatch):
    monitor = _monitor(monkeypatch, vendor="unknown")
    assert monitor.start() is False
    assert monitor.is_running is False


def test_collect_amd_sample_discrete_gpu_shape(monkeypatch):
    """Discrete-GPU-style payload: per-domain clock entries and 'hotspot' temperature."""
    monitor = _monitor(monkeypatch)
    payload = {
        "gpu_data": [
            {
                "gpu": 0,
                "temperature": {"hotspot": 62, "edge": 50},
                "clock": {
                    "gfx_0": {"clk": {"value": 1500, "unit": "MHz"}},
                    "mem_0": {"clk": {"value": 1200, "unit": "MHz"}},
                },
                "power": {"socket_power": {"value": 300, "unit": "W"}},
            }
        ]
    }
    monkeypatch.setattr(gpu_monitor.amd_smi, "run_json", lambda *a, **k: payload)
    sample = monitor._collect_amd_sample()
    assert sample.temperature_c == 62
    assert sample.gpu_clock_mhz == 1500
    assert sample.mem_clock_mhz == 1200
    assert sample.power_watts == 300


def test_collect_amd_sample_apu_shape(monkeypatch):
    """APU-style payload: per-domain clock/temperature read "N/A"; only the apu_average_*
    aggregate keys and "socket_power" carry real numbers (observed on gfx1151)."""
    monitor = _monitor(monkeypatch)
    payload = {
        "gpu_data": [
            {
                "gpu": 0,
                "temperature": {
                    "hotspot": "N/A",
                    "edge": 45,
                    "apu_temperature_gfx": 45,
                },
                "clock": {
                    "gfx_0": {"clk": "N/A"},
                    "mem_0": {"clk": "N/A"},
                    "apu_average_gfxclk_frequency": {"value": 627, "unit": "MHz"},
                    "apu_average_uclk_frequency": {"value": 800, "unit": "MHz"},
                },
                "power": {"socket_power": "N/A", "apu_average_socket_power": 11.0},
            }
        ]
    }
    monkeypatch.setattr(gpu_monitor.amd_smi, "run_json", lambda *a, **k: payload)
    sample = monitor._collect_amd_sample()
    assert sample.temperature_c == 45
    assert sample.gpu_clock_mhz == 627
    assert sample.mem_clock_mhz == 800
    assert sample.power_watts == 11.0


def test_collect_amd_sample_everything_na_reports_none(monkeypatch):
    """26.2.2 / ROCm 7.2.3 shape observed on an APU: power and clock are entirely "N/A",
    temperature only has "edge". Must degrade to the GPUSample defaults, not raise."""
    monitor = _monitor(monkeypatch)
    payload = {
        "gpu_data": [
            {
                "gpu": 0,
                "temperature": {"edge": 52, "hotspot": "N/A", "mem": "N/A"},
                "clock": {"gfx_0": {"clk": "N/A"}, "mem_0": {"clk": "N/A"}},
                "power": {"socket_power": "N/A"},
            }
        ]
    }
    monkeypatch.setattr(gpu_monitor.amd_smi, "run_json", lambda *a, **k: payload)
    sample = monitor._collect_amd_sample()
    assert sample.temperature_c == 52
    assert sample.gpu_clock_mhz == 0
    assert sample.mem_clock_mhz == 0
    assert sample.power_watts == 0.0


def test_collect_amd_sample_returns_none_when_amd_smi_unavailable(monkeypatch):
    monitor = _monitor(monkeypatch)
    monkeypatch.setattr(gpu_monitor.amd_smi, "run_json", lambda *a, **k: None)
    assert monitor._collect_amd_sample() is None


def test_collect_amd_sample_returns_none_when_no_rows(monkeypatch):
    monitor = _monitor(monkeypatch)
    monkeypatch.setattr(
        gpu_monitor.amd_smi, "run_json", lambda *a, **k: {"gpu_data": []}
    )
    assert monitor._collect_amd_sample() is None


def test_collect_nvidia_sample_parses_csv(monkeypatch):
    monitor = _monitor(monkeypatch, vendor="nvidia")
    monkeypatch.setattr(
        gpu_monitor.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            [], 0, stdout="65, 1400, 1100, 250.5"
        ),
    )
    sample = monitor._collect_nvidia_sample()
    assert sample.temperature_c == 65.0
    assert sample.gpu_clock_mhz == 1400
    assert sample.mem_clock_mhz == 1100
    assert sample.power_watts == 250.5


def test_start_stop_lifecycle_collects_samples(monkeypatch):
    monitor = _monitor(monkeypatch)
    monkeypatch.setattr(
        monitor,
        "_collect_amd_sample",
        lambda: GPUSample(timestamp=time.time(), temperature_c=50.0),
    )
    assert monitor.start() is True
    assert monitor.start() is True  # already running: no-op, still True
    time.sleep(1.2)
    stats = monitor.stop()
    assert isinstance(stats, GPUMonitorStats)
    assert stats.sample_count >= 1
    assert stats.temp_avg == 50.0
    assert monitor.is_running is False
