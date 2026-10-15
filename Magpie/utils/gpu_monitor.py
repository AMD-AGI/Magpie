###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
GPU monitoring for benchmark mode.

Provides background monitoring of GPU temperature, frequency, and power
during benchmark execution.
"""

import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import amd_smi

logger = logging.getLogger(__name__)

#: ``metric --temperature`` keys a representative die temperature has appeared under, hottest
#: sensor first; "junction" is rocm-smi's old name for what amd-smi calls "hotspot".
_TEMPERATURE_KEYS = ("hotspot", "junction", "edge", "apu_temperature_gfx")

#: ``metric --clock`` domain the GPU core clock has appeared under; APUs only populate the
#: aggregate key, discrete GPUs only populate the per-domain one.
_GPU_CLOCK_KEYS = ("gfx_0", "apu_average_gfxclk_frequency")

#: ``metric --clock`` domain the memory clock has appeared under, same APU/discrete split.
_MEM_CLOCK_KEYS = ("mem_0", "apu_average_uclk_frequency")

#: ``metric --power`` keys the socket draw has appeared under.
_SOCKET_POWER_KEYS = ("socket_power", "apu_average_socket_power")


@dataclass
class GPUSample:
    """Single GPU sample data point."""
    timestamp: float
    temperature_c: float = 0.0
    gpu_clock_mhz: int = 0
    mem_clock_mhz: int = 0
    power_watts: float = 0.0


@dataclass
class GPUMonitorStats:
    """GPU monitoring statistics computed from samples."""
    sample_count: int = 0
    duration_sec: float = 0.0
    
    # Temperature statistics (Celsius)
    temp_min: float = 0.0
    temp_max: float = 0.0
    temp_avg: float = 0.0
    
    # GPU clock statistics (MHz)
    gpu_clock_min: int = 0
    gpu_clock_max: int = 0
    gpu_clock_avg: float = 0.0
    
    # Memory clock statistics (MHz)
    mem_clock_min: int = 0
    mem_clock_max: int = 0
    mem_clock_avg: float = 0.0
    
    # Power statistics (Watts)
    power_min: float = 0.0
    power_max: float = 0.0
    power_avg: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sample_count": self.sample_count,
            "duration_sec": round(self.duration_sec, 2),
            "temperature_c": {
                "min": round(self.temp_min, 1),
                "max": round(self.temp_max, 1),
                "avg": round(self.temp_avg, 1),
            },
            "gpu_clock_mhz": {
                "min": self.gpu_clock_min,
                "max": self.gpu_clock_max,
                "avg": round(self.gpu_clock_avg, 1),
            },
            "mem_clock_mhz": {
                "min": self.mem_clock_min,
                "max": self.mem_clock_max,
                "avg": round(self.mem_clock_avg, 1),
            },
            "power_watts": {
                "min": round(self.power_min, 1),
                "max": round(self.power_max, 1),
                "avg": round(self.power_avg, 1),
            },
        }


class GPUMonitor:
    """
    Background GPU monitor for collecting hardware metrics during benchmark.
    
    Collects temperature, GPU/memory clock frequencies, and power consumption
    at regular intervals using amd-smi (AMD) or nvidia-smi (NVIDIA).
    
    Example:
        monitor = GPUMonitor(device_id=0, interval_sec=2.0)
        monitor.start()
        
        # ... run benchmark ...
        
        stats = monitor.stop()
        print(f"Avg temp: {stats.temp_avg}°C, Max power: {stats.power_max}W")
    """
    
    def __init__(
        self,
        device_id: int = 0,
        interval_sec: float = 2.0,
        vendor: str = "auto",
    ):
        """
        Initialize GPU monitor.
        
        Args:
            device_id: GPU device index to monitor
            interval_sec: Sampling interval in seconds (default 2.0)
            vendor: GPU vendor ("amd", "nvidia", or "auto" for detection)
        """
        self.device_id = device_id
        self.interval = max(0.5, interval_sec)  # Minimum 0.5s
        self.vendor = self._detect_vendor() if vendor == "auto" else vendor.lower()
        
        self._samples: List[GPUSample] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: float = 0.0
        self._lock = threading.Lock()
    
    def _detect_vendor(self) -> str:
        """Auto-detect GPU vendor."""
        payload = amd_smi.run_json(["list"], timeout_s=5.0)
        if payload is not None and amd_smi.gpu_rows(payload):
            return "amd"

        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and "GPU" in result.stdout:
                return "nvidia"
        except Exception:
            pass
        
        return "unknown"
    
    def start(self) -> bool:
        """
        Start background monitoring.
        
        Returns:
            True if monitoring started successfully, False if skipped (unsupported vendor)
        """
        if self._running:
            logger.warning("GPU monitor already running")
            return True
        
        # Only AMD GPUs are supported currently
        if self.vendor != "amd":
            logger.warning(
                f"GPU monitor: vendor '{self.vendor}' not supported, skipping. "
                "Currently only AMD GPUs (amd-smi) are supported."
            )
            return False
        
        self._samples.clear()
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.debug(f"GPU monitor started (device={self.device_id}, interval={self.interval}s)")
        return True
    
    def stop(self) -> GPUMonitorStats:
        """
        Stop monitoring and return statistics.
        
        Returns:
            GPUMonitorStats with computed min/max/avg values
        """
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.interval + 1.0)
        
        stats = self._compute_stats()
        logger.debug(f"GPU monitor stopped: {stats.sample_count} samples collected")
        return stats
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            sample = self._collect_sample()
            if sample:
                with self._lock:
                    self._samples.append(sample)
            time.sleep(self.interval)
    
    def _collect_sample(self) -> Optional[GPUSample]:
        """Collect a single GPU sample."""
        try:
            if self.vendor == "amd":
                return self._collect_amd_sample()
            elif self.vendor == "nvidia":
                return self._collect_nvidia_sample()
            else:
                return None
        except Exception as e:
            logger.debug(f"Failed to collect GPU sample: {e}")
            return None
    
    def _collect_amd_sample(self) -> Optional[GPUSample]:
        """Collect sample from AMD GPU using amd-smi."""
        sample = GPUSample(timestamp=time.time())

        payload = amd_smi.run_json(
            ["metric", "--temperature", "--clock", "--power", "-g", str(self.device_id)],
            timeout_s=5.0,
        )
        if payload is None:
            return None
        rows = amd_smi.gpu_rows(payload)
        if not rows:
            return None
        row = rows[0]

        temp_c = amd_smi.scalar(amd_smi.lookup(row.get("temperature"), *_TEMPERATURE_KEYS))
        if temp_c is not None:
            sample.temperature_c = temp_c

        gpu_clk = amd_smi.clock_mhz(row.get("clock"), *_GPU_CLOCK_KEYS)
        if gpu_clk is not None:
            sample.gpu_clock_mhz = round(gpu_clk)

        mem_clk = amd_smi.clock_mhz(row.get("clock"), *_MEM_CLOCK_KEYS)
        if mem_clk is not None:
            sample.mem_clock_mhz = round(mem_clk)

        power_w = amd_smi.scalar(amd_smi.lookup(row.get("power"), *_SOCKET_POWER_KEYS))
        if power_w is not None:
            sample.power_watts = power_w

        return sample
    
    def _collect_nvidia_sample(self) -> Optional[GPUSample]:
        """Collect sample from NVIDIA GPU using nvidia-smi."""
        sample = GPUSample(timestamp=time.time())
        
        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", str(self.device_id),
                 "--query-gpu=temperature.gpu,clocks.gr,clocks.mem,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode != 0:
                return None
            
            # Parse CSV output: temp, gpu_clock, mem_clock, power
            values = result.stdout.strip().split(",")
            if len(values) >= 4:
                sample.temperature_c = float(values[0].strip())
                sample.gpu_clock_mhz = int(values[1].strip())
                sample.mem_clock_mhz = int(values[2].strip())
                sample.power_watts = float(values[3].strip())
            
            return sample
            
        except subprocess.TimeoutExpired:
            logger.debug("nvidia-smi timed out")
            return None
        except Exception as e:
            logger.debug(f"Error collecting NVIDIA sample: {e}")
            return None
    
    def _compute_stats(self) -> GPUMonitorStats:
        """Compute statistics from collected samples."""
        with self._lock:
            samples = list(self._samples)
        
        if not samples:
            return GPUMonitorStats()
        
        temps = [s.temperature_c for s in samples if s.temperature_c > 0]
        gpu_clocks = [s.gpu_clock_mhz for s in samples if s.gpu_clock_mhz > 0]
        mem_clocks = [s.mem_clock_mhz for s in samples if s.mem_clock_mhz > 0]
        powers = [s.power_watts for s in samples if s.power_watts > 0]
        
        duration = samples[-1].timestamp - self._start_time if samples else 0.0
        
        stats = GPUMonitorStats(
            sample_count=len(samples),
            duration_sec=duration,
        )
        
        if temps:
            stats.temp_min = min(temps)
            stats.temp_max = max(temps)
            stats.temp_avg = sum(temps) / len(temps)
        
        if gpu_clocks:
            stats.gpu_clock_min = min(gpu_clocks)
            stats.gpu_clock_max = max(gpu_clocks)
            stats.gpu_clock_avg = sum(gpu_clocks) / len(gpu_clocks)
        
        if mem_clocks:
            stats.mem_clock_min = min(mem_clocks)
            stats.mem_clock_max = max(mem_clocks)
            stats.mem_clock_avg = sum(mem_clocks) / len(mem_clocks)
        
        if powers:
            stats.power_min = min(powers)
            stats.power_max = max(powers)
            stats.power_avg = sum(powers) / len(powers)
        
        return stats
    
    @property
    def is_running(self) -> bool:
        """Check if monitor is currently running."""
        return self._running
    
    @property
    def sample_count(self) -> int:
        """Get current number of collected samples."""
        with self._lock:
            return len(self._samples)
