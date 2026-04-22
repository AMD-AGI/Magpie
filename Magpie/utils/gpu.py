###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
GPU detection and hardware control utilities.
"""

from __future__ import annotations

import json as _json
import subprocess
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

_MAGPIE_CACHE_DIR = Path.home() / ".magpie"
_COMPUTE_SPEC_CACHE_FILE = _MAGPIE_CACHE_DIR / "gpu_compute_specs.json"


class GPUVendor(Enum):
    """GPU vendor type."""

    AMD = auto()
    NVIDIA = auto()
    UNKNOWN = auto()


@dataclass
class GPUComputeSpec:
    """Static GPU compute architecture specifications (from rocminfo / nvidia-smi).

    These are immutable hardware properties that do not change at runtime.
    """

    # Compute resources
    compute_units: Optional[int] = None
    simds_per_cu: Optional[int] = None
    shader_engines: Optional[int] = None
    wavefront_size: Optional[int] = None
    max_workgroup_size: Optional[int] = None
    max_workgroup_size_xyz: Optional[List[int]] = None
    max_waves_per_cu: Optional[int] = None
    max_workitems_per_cu: Optional[int] = None

    # Grid dispatch limits
    grid_max_size: Optional[int] = None
    grid_max_size_xyz: Optional[List[int]] = None

    # Cache hierarchy (KB)
    l1_cache_kb: Optional[int] = None
    l2_cache_kb: Optional[int] = None
    l3_cache_kb: Optional[int] = None

    # Cacheline
    cacheline_bytes: Optional[int] = None

    # Local Data Share / Shared Memory (KB)
    lds_size_kb: Optional[int] = None

    # Memory subsystem
    memory_bus_width_bits: Optional[int] = None
    memory_bandwidth_gbs: Optional[float] = None

    # ISA / device identity
    isa_name: Optional[str] = None
    marketing_name: Optional[str] = None
    chip_id: Optional[str] = None
    gpu_uuid: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary, omitting None values for cleaner output."""
        return {k: v for k, v in {
            "compute_units": self.compute_units,
            "simds_per_cu": self.simds_per_cu,
            "shader_engines": self.shader_engines,
            "wavefront_size": self.wavefront_size,
            "max_workgroup_size": self.max_workgroup_size,
            "max_workgroup_size_xyz": self.max_workgroup_size_xyz,
            "max_waves_per_cu": self.max_waves_per_cu,
            "max_workitems_per_cu": self.max_workitems_per_cu,
            "grid_max_size": self.grid_max_size,
            "grid_max_size_xyz": self.grid_max_size_xyz,
            "l1_cache_kb": self.l1_cache_kb,
            "l2_cache_kb": self.l2_cache_kb,
            "l3_cache_kb": self.l3_cache_kb,
            "cacheline_bytes": self.cacheline_bytes,
            "lds_size_kb": self.lds_size_kb,
            "memory_bus_width_bits": self.memory_bus_width_bits,
            "memory_bandwidth_gbs": self.memory_bandwidth_gbs,
            "isa_name": self.isa_name,
            "marketing_name": self.marketing_name,
            "chip_id": self.chip_id,
            "gpu_uuid": self.gpu_uuid,
        }.items() if v is not None}


@dataclass
class GPUHardwareInfo:
    """GPU hardware information."""

    vendor: GPUVendor
    architecture: Optional[str] = None
    device_id: int = 0
    device_name: Optional[str] = None

    # Power information
    power_current_watts: Optional[float] = None
    power_limit_watts: Optional[float] = None
    power_max_watts: Optional[float] = None

    # Frequency information (MHz)
    gpu_clock_current: Optional[int] = None
    gpu_clock_max: Optional[int] = None
    mem_clock_current: Optional[int] = None
    mem_clock_max: Optional[int] = None

    # Temperature (Celsius)
    temperature: Optional[float] = None

    # Memory (GB)
    memory_total_gb: Optional[float] = None
    memory_used_gb: Optional[float] = None

    # Static compute architecture specs
    compute_spec: Optional[GPUComputeSpec] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = {
            "vendor": self.vendor.name,
            "architecture": self.architecture,
            "device_id": self.device_id,
            "device_name": self.device_name,
            "power_current_watts": self.power_current_watts,
            "power_limit_watts": self.power_limit_watts,
            "power_max_watts": self.power_max_watts,
            "gpu_clock_current": self.gpu_clock_current,
            "gpu_clock_max": self.gpu_clock_max,
            "mem_clock_current": self.mem_clock_current,
            "mem_clock_max": self.mem_clock_max,
            "temperature": self.temperature,
            "memory_total_gb": self.memory_total_gb,
            "memory_used_gb": self.memory_used_gb,
        }
        if self.compute_spec is not None:
            d["compute_spec"] = self.compute_spec.to_dict()
        return d


@dataclass
class GPUConfig:
    """GPU configuration for power and frequency control."""

    device_id: int = 0
    # Power limit in watts (None = don't change)
    power_limit_watts: Optional[int] = None
    # GPU clock frequency in MHz (min, max), None = don't change
    gpu_clock_mhz: Optional[Tuple[int, int]] = None
    # Memory clock frequency in MHz (min, max), None = don't change
    mem_clock_mhz: Optional[Tuple[int, int]] = None
    # Lock clocks at specific levels (AMD specific: 0-7)
    gpu_clock_level: Optional[int] = None
    mem_clock_level: Optional[int] = None


def _parse_rocminfo_gpu_agents(raw_output: str) -> List[GPUComputeSpec]:
    """Parse rocminfo output and return a GPUComputeSpec per GPU agent.

    Splits the output into Agent blocks, filters GPU agents, and extracts
    static hardware properties from each.
    """
    agent_blocks: List[str] = re.split(r"^\*{5,}", raw_output, flags=re.MULTILINE)
    specs: List[GPUComputeSpec] = []

    for block in agent_blocks:
        if "Device Type:" not in block:
            continue
        type_match = re.search(r"Device Type:\s+(\w+)", block)
        if not type_match or type_match.group(1) != "GPU":
            continue

        spec = GPUComputeSpec()

        def _int(pattern: str) -> Optional[int]:
            m = re.search(pattern, block)
            return int(m.group(1)) if m else None

        spec.marketing_name = (
            re.search(r"Marketing Name:\s+(.+)", block).group(1).strip()
            if re.search(r"Marketing Name:\s+(.+)", block)
            else None
        )
        uuid_m = re.search(r"Uuid:\s+(GPU-[\w]+)", block)
        spec.gpu_uuid = uuid_m.group(1) if uuid_m else None
        chip_m = re.search(r"Chip ID:\s+\d+\((0x\w+)\)", block)
        spec.chip_id = chip_m.group(1) if chip_m else None

        spec.compute_units = _int(r"Compute Unit:\s+(\d+)")
        spec.simds_per_cu = _int(r"SIMDs per CU:\s+(\d+)")
        spec.shader_engines = _int(r"Shader Engines:\s+(\d+)")
        spec.wavefront_size = _int(r"Wavefront Size:\s+(\d+)")
        spec.max_workgroup_size = _int(r"Workgroup Max Size:\s+(\d+)")
        spec.max_waves_per_cu = _int(r"Max Waves Per CU:\s+(\d+)")
        spec.max_workitems_per_cu = _int(r"Max Work-item Per CU:\s+(\d+)")
        spec.cacheline_bytes = _int(r"Cacheline Size:\s+(\d+)")

        # Workgroup Max Size per Dimension (x, y, z)
        wg_dim = re.search(
            r"Workgroup Max Size per Dimension:\s*\n"
            r"\s+x\s+(\d+).*\n"
            r"\s+y\s+(\d+).*\n"
            r"\s+z\s+(\d+)",
            block,
        )
        if wg_dim:
            spec.max_workgroup_size_xyz = [
                int(wg_dim.group(1)), int(wg_dim.group(2)), int(wg_dim.group(3))
            ]

        # Grid Max Size
        spec.grid_max_size = _int(r"Grid Max Size:\s+(\d+)")
        grid_dim = re.search(
            r"Grid Max Size per Dimension:\s*\n"
            r"\s+x\s+(\d+).*\n"
            r"\s+y\s+(\d+).*\n"
            r"\s+z\s+(\d+)",
            block,
        )
        if grid_dim:
            spec.grid_max_size_xyz = [
                int(grid_dim.group(1)), int(grid_dim.group(2)), int(grid_dim.group(3))
            ]

        # Cache hierarchy – rocminfo lists L1, L2, L3 under "Cache Info:"
        cache_section = re.search(
            r"Cache Info:\s*\n((?:\s+L\d.*\n)+)", block
        )
        if cache_section:
            cache_text = cache_section.group(1)
            l1 = re.search(r"L1:\s+(\d+)", cache_text)
            l2 = re.search(r"L2:\s+(\d+)", cache_text)
            l3 = re.search(r"L3:\s+(\d+)", cache_text)
            spec.l1_cache_kb = int(l1.group(1)) if l1 else None
            spec.l2_cache_kb = int(l2.group(1)) if l2 else None
            spec.l3_cache_kb = int(l3.group(1)) if l3 else None

        # LDS = GROUP segment size
        group_match = re.search(
            r"Segment:\s+GROUP\s*\n\s*Size:\s+(\d+)", block
        )
        if group_match:
            spec.lds_size_kb = int(group_match.group(1))

        # ISA name (first ISA entry)
        isa_match = re.search(r"Name:\s+(amdgcn[^\s]+)", block)
        spec.isa_name = isa_match.group(1) if isa_match else None

        specs.append(spec)

    return specs


def _load_compute_spec_cache() -> Optional[Dict]:
    """Load cached compute specs from ~/.magpie/gpu_compute_specs.json."""
    try:
        if _COMPUTE_SPEC_CACHE_FILE.exists():
            data = _json.loads(_COMPUTE_SPEC_CACHE_FILE.read_text())
            return data
    except Exception as e:
        logger.debug(f"Failed to read compute spec cache: {e}")
    return None


def _save_compute_spec_cache(specs: List[GPUComputeSpec]) -> None:
    """Save compute specs to ~/.magpie/gpu_compute_specs.json."""
    try:
        _MAGPIE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gpu_count": len(specs),
            "devices": {str(i): specs[i].to_dict() for i in range(len(specs))},
        }
        _COMPUTE_SPEC_CACHE_FILE.write_text(
            _json.dumps(payload, indent=2) + "\n"
        )
        logger.info(
            f"Cached {len(specs)} GPU compute specs to {_COMPUTE_SPEC_CACHE_FILE}"
        )
    except Exception as e:
        logger.warning(f"Failed to write compute spec cache: {e}")


def _spec_from_cache_dict(d: dict) -> GPUComputeSpec:
    """Reconstruct a GPUComputeSpec from a cached dictionary."""
    return GPUComputeSpec(
        compute_units=d.get("compute_units"),
        simds_per_cu=d.get("simds_per_cu"),
        shader_engines=d.get("shader_engines"),
        wavefront_size=d.get("wavefront_size"),
        max_workgroup_size=d.get("max_workgroup_size"),
        max_workgroup_size_xyz=d.get("max_workgroup_size_xyz"),
        max_waves_per_cu=d.get("max_waves_per_cu"),
        max_workitems_per_cu=d.get("max_workitems_per_cu"),
        grid_max_size=d.get("grid_max_size"),
        grid_max_size_xyz=d.get("grid_max_size_xyz"),
        l1_cache_kb=d.get("l1_cache_kb"),
        l2_cache_kb=d.get("l2_cache_kb"),
        l3_cache_kb=d.get("l3_cache_kb"),
        cacheline_bytes=d.get("cacheline_bytes"),
        lds_size_kb=d.get("lds_size_kb"),
        memory_bus_width_bits=d.get("memory_bus_width_bits"),
        memory_bandwidth_gbs=d.get("memory_bandwidth_gbs"),
        isa_name=d.get("isa_name"),
        marketing_name=d.get("marketing_name"),
        chip_id=d.get("chip_id"),
        gpu_uuid=d.get("gpu_uuid"),
    )


# Module-level in-memory cache so rocminfo is parsed at most once per process.
_rocminfo_specs: Optional[List[GPUComputeSpec]] = None


def get_amd_compute_specs(force_refresh: bool = False) -> List[GPUComputeSpec]:
    """Get compute specs for all AMD GPUs, using a two-level cache.

    Level 1: in-memory (process lifetime).
    Level 2: ~/.magpie/gpu_compute_specs.json (persists across runs).

    Args:
        force_refresh: bypass caches and re-parse rocminfo.
    """
    global _rocminfo_specs

    if not force_refresh and _rocminfo_specs is not None:
        return _rocminfo_specs

    if not force_refresh:
        cached = _load_compute_spec_cache()
        if cached and "devices" in cached:
            specs = [
                _spec_from_cache_dict(cached["devices"][str(i)])
                for i in range(cached.get("gpu_count", 0))
                if str(i) in cached["devices"]
            ]
            if specs:
                _rocminfo_specs = specs
                logger.info(
                    f"Loaded {len(specs)} GPU specs from cache "
                    f"(cached at {cached.get('cached_at', 'unknown')})"
                )
                return specs

    # Parse live from rocminfo
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            specs = _parse_rocminfo_gpu_agents(result.stdout)
            if specs:
                _rocminfo_specs = specs
                _save_compute_spec_cache(specs)
                return specs
    except FileNotFoundError:
        logger.debug("rocminfo not found")
    except subprocess.TimeoutExpired:
        logger.warning("rocminfo timed out")
    except Exception as e:
        logger.warning(f"Failed to parse rocminfo: {e}")

    return []


class GPUController:
    """GPU hardware controller for AMD and NVIDIA GPUs."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.vendor, self.arch = detect_gpu()

    def get_hardware_info(self) -> GPUHardwareInfo:
        """Get detailed hardware information."""
        if self.vendor == GPUVendor.AMD:
            return self._get_amd_info()
        elif self.vendor == GPUVendor.NVIDIA:
            return self._get_nvidia_info()
        return GPUHardwareInfo(vendor=GPUVendor.UNKNOWN)

    def _get_amd_info(self) -> GPUHardwareInfo:
        """Get AMD GPU info using rocm-smi and rocminfo."""
        info = GPUHardwareInfo(
            vendor=GPUVendor.AMD, architecture=self.arch, device_id=self.device_id
        )

        # Populate static compute specs (cached)
        specs = get_amd_compute_specs()
        if self.device_id < len(specs):
            spec = specs[self.device_id]
            info.compute_spec = spec
            if spec.marketing_name and not info.device_name:
                info.device_name = spec.marketing_name

        try:
            # Get power info
            result = subprocess.run(
                ["rocm-smi", "-d", str(self.device_id), "--showpower"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                # Parse: "GPU[0]		: Average Graphics Package Power (W): 150.0"
                for line in result.stdout.split("\n"):
                    if "Power" in line and "W" in line:
                        match = re.search(r"(\d+\.?\d*)\s*$", line)
                        if match:
                            info.power_current_watts = float(match.group(1))
                            break

            # Get clock info
            result = subprocess.run(
                ["rocm-smi", "-d", str(self.device_id), "--showclocks"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "sclk" in line.lower():
                        match = re.search(r"(\d+)", line)
                        if match:
                            info.gpu_clock_current = int(match.group(1))
                    elif "mclk" in line.lower():
                        match = re.search(r"(\d+)", line)
                        if match:
                            info.mem_clock_current = int(match.group(1))

            # Get temperature
            result = subprocess.run(
                ["rocm-smi", "-d", str(self.device_id), "--showtemp"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "Temperature" in line:
                        match = re.search(r"(\d+\.?\d*)", line)
                        if match:
                            info.temperature = float(match.group(1))
                            break

            # Get memory info
            result = subprocess.run(
                ["rocm-smi", "-d", str(self.device_id), "--showmeminfo", "vram"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "Total" in line:
                        match = re.search(r"(\d+)", line)
                        if match:
                            info.memory_total_gb = int(match.group(1)) / (1024**3)
                    elif "Used" in line:
                        match = re.search(r"(\d+)", line)
                        if match:
                            info.memory_used_gb = int(match.group(1)) / (1024**3)

        except Exception as e:
            logger.warning(f"Error getting AMD GPU info: {e}")

        return info

    def _get_nvidia_info(self) -> GPUHardwareInfo:
        """Get NVIDIA GPU info using nvidia-smi."""
        info = GPUHardwareInfo(
            vendor=GPUVendor.NVIDIA, architecture=self.arch, device_id=self.device_id
        )

        try:
            # Query multiple properties at once
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(self.device_id),
                    "--query-gpu=name,power.draw,power.limit,power.max_limit,"
                    "clocks.current.graphics,clocks.max.graphics,"
                    "clocks.current.memory,clocks.max.memory,"
                    "temperature.gpu,memory.total,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                values = [v.strip() for v in result.stdout.strip().split(",")]
                if len(values) >= 11:
                    info.device_name = values[0]
                    info.power_current_watts = (
                        float(values[1]) if values[1] != "[N/A]" else None
                    )
                    info.power_limit_watts = (
                        float(values[2]) if values[2] != "[N/A]" else None
                    )
                    info.power_max_watts = (
                        float(values[3]) if values[3] != "[N/A]" else None
                    )
                    info.gpu_clock_current = (
                        int(values[4]) if values[4] != "[N/A]" else None
                    )
                    info.gpu_clock_max = (
                        int(values[5]) if values[5] != "[N/A]" else None
                    )
                    info.mem_clock_current = (
                        int(values[6]) if values[6] != "[N/A]" else None
                    )
                    info.mem_clock_max = (
                        int(values[7]) if values[7] != "[N/A]" else None
                    )
                    info.temperature = (
                        float(values[8]) if values[8] != "[N/A]" else None
                    )
                    info.memory_total_gb = (
                        float(values[9]) / 1024 if values[9] != "[N/A]" else None
                    )
                    info.memory_used_gb = (
                        float(values[10]) / 1024 if values[10] != "[N/A]" else None
                    )

        except Exception as e:
            logger.warning(f"Error getting NVIDIA GPU info: {e}")

        return info

    def apply_config(self, config: GPUConfig) -> bool:
        """Apply GPU configuration (power limit, frequencies)."""
        if self.vendor == GPUVendor.AMD:
            return self._apply_amd_config(config)
        elif self.vendor == GPUVendor.NVIDIA:
            return self._apply_nvidia_config(config)
        return False

    def _apply_amd_config(self, config: GPUConfig) -> bool:
        """Apply configuration to AMD GPU using rocm-smi."""
        success = True
        device = str(config.device_id)

        try:
            # Set power limit (overdrive percentage, not direct watts on some GPUs)
            if config.power_limit_watts is not None:
                result = subprocess.run(
                    [
                        "rocm-smi",
                        "-d",
                        device,
                        "--setpoweroverdrive",
                        str(config.power_limit_watts),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    logger.error(f"Failed to set power limit: {result.stderr}")
                    success = False
                else:
                    logger.info(f"Set power limit to {config.power_limit_watts}W")

            # Set GPU clock level
            if config.gpu_clock_level is not None:
                result = subprocess.run(
                    [
                        "rocm-smi",
                        "-d",
                        device,
                        "--setsclk",
                        str(config.gpu_clock_level),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    logger.error(f"Failed to set GPU clock: {result.stderr}")
                    success = False
                else:
                    logger.info(f"Set GPU clock level to {config.gpu_clock_level}")

            # Set memory clock level
            if config.mem_clock_level is not None:
                result = subprocess.run(
                    [
                        "rocm-smi",
                        "-d",
                        device,
                        "--setmclk",
                        str(config.mem_clock_level),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    logger.error(f"Failed to set memory clock: {result.stderr}")
                    success = False
                else:
                    logger.info(f"Set memory clock level to {config.mem_clock_level}")

        except Exception as e:
            logger.error(f"Error applying AMD config: {e}")
            success = False

        return success

    def _apply_nvidia_config(self, config: GPUConfig) -> bool:
        """Apply configuration to NVIDIA GPU using nvidia-smi."""
        success = True
        device = str(config.device_id)

        try:
            # Set power limit
            if config.power_limit_watts is not None:
                result = subprocess.run(
                    ["nvidia-smi", "-i", device, "-pl", str(config.power_limit_watts)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    logger.error(f"Failed to set power limit: {result.stderr}")
                    success = False
                else:
                    logger.info(f"Set power limit to {config.power_limit_watts}W")

            # Lock GPU clocks
            if config.gpu_clock_mhz is not None:
                min_clk, max_clk = config.gpu_clock_mhz
                result = subprocess.run(
                    ["nvidia-smi", "-i", device, "-lgc", f"{min_clk},{max_clk}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    logger.error(f"Failed to lock GPU clocks: {result.stderr}")
                    success = False
                else:
                    logger.info(f"Locked GPU clocks to {min_clk}-{max_clk} MHz")

            # Lock memory clocks
            if config.mem_clock_mhz is not None:
                min_clk, max_clk = config.mem_clock_mhz
                result = subprocess.run(
                    ["nvidia-smi", "-i", device, "-lmc", f"{min_clk},{max_clk}"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    logger.error(f"Failed to lock memory clocks: {result.stderr}")
                    success = False
                else:
                    logger.info(f"Locked memory clocks to {min_clk}-{max_clk} MHz")

        except Exception as e:
            logger.error(f"Error applying NVIDIA config: {e}")
            success = False

        return success

    def reset_config(self) -> bool:
        """Reset GPU to default settings."""
        if self.vendor == GPUVendor.AMD:
            try:
                subprocess.run(
                    ["rocm-smi", "-d", str(self.device_id), "--resetclocks"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                logger.info(f"Reset AMD GPU {self.device_id} clocks")
                return True
            except Exception as e:
                logger.error(f"Error resetting AMD GPU: {e}")
                return False
        elif self.vendor == GPUVendor.NVIDIA:
            try:
                # Reset GPU clocks
                subprocess.run(
                    ["nvidia-smi", "-i", str(self.device_id), "-rgc"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                # Reset memory clocks
                subprocess.run(
                    ["nvidia-smi", "-i", str(self.device_id), "-rmc"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                logger.info(f"Reset NVIDIA GPU {self.device_id} clocks")
                return True
            except Exception as e:
                logger.error(f"Error resetting NVIDIA GPU: {e}")
                return False
        return False


def detect_gpu() -> Tuple[GPUVendor, Optional[str]]:
    """
    Auto-detect GPU vendor and architecture.

    Returns:
        Tuple of (GPUVendor, architecture_string)
        - For AMD: ("gfx90a", "gfx942", etc.)
        - For NVIDIA: ("sm_80", "sm_90", etc.)
    """
    # Try AMD GPU first
    amd_arch = _detect_amd_gpu()
    if amd_arch:
        return GPUVendor.AMD, amd_arch

    # Try NVIDIA GPU
    nvidia_arch = _detect_nvidia_gpu()
    if nvidia_arch:
        return GPUVendor.NVIDIA, nvidia_arch

    logger.warning("No GPU detected")
    return GPUVendor.UNKNOWN, None


def _detect_amd_gpu() -> Optional[str]:
    """Detect AMD GPU architecture using rocminfo."""
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            # Parse rocminfo output for gfx architecture
            for line in result.stdout.split("\n"):
                if "gfx" in line.lower():
                    # Extract gfx version (e.g., gfx90a, gfx942)
                    match = re.search(r"(gfx\w+)", line, re.IGNORECASE)
                    if match:
                        arch = match.group(1).lower()
                        logger.info(f"Detected AMD GPU: {arch}")
                        return arch
    except FileNotFoundError:
        pass  # rocminfo not available
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        logger.debug(f"AMD GPU detection failed: {e}")

    return None


def _detect_nvidia_gpu() -> Optional[str]:
    """Detect NVIDIA GPU architecture using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # Parse compute capability (e.g., "8.0" -> "sm_80")
            compute_cap = result.stdout.strip().split("\n")[0]
            if compute_cap:
                major, minor = compute_cap.split(".")
                arch = f"sm_{major}{minor}"
                logger.info(f"Detected NVIDIA GPU: {arch}")
                return arch
    except FileNotFoundError:
        pass  # nvidia-smi not available
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        logger.debug(f"NVIDIA GPU detection failed: {e}")

    return None


def get_gpu_info() -> dict:
    """
    Get detailed GPU information.

    Returns:
        Dictionary with GPU details
    """
    vendor, arch = detect_gpu()

    info = {
        "vendor": vendor.name,
        "architecture": arch,
        "detected": vendor != GPUVendor.UNKNOWN,
    }

    if vendor == GPUVendor.AMD:
        info["profiler"] = "rocprofv3 & rocprof-compute"
        info["compiler"] = "hipcc"
    elif vendor == GPUVendor.NVIDIA:
        info["profiler"] = "ncu"
        info["compiler"] = "nvcc"

    return info


def _amd_busy_gpu_ids() -> set:
    """Return AMD GPU device IDs that currently have KFD compute processes.

    Uses ``rocm-smi --showpidgpus`` which prints, for each running KFD PID,
    the list of GPU indices it is using. When idle, rocm-smi reports
    "No KFD PIDs currently running" and this returns an empty set.
    """
    busy: set = set()
    try:
        result = subprocess.run(
            ["rocm-smi", "--showpidgpus"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return busy
        out = result.stdout
        if "No KFD PIDs currently running" in out:
            return busy
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            if "device(s):" in line:
                tail = line.split("device(s):", 1)[1].strip()
                for tok in tail.split():
                    if tok.isdigit():
                        busy.add(int(tok))
            elif line.startswith("PID "):
                continue
            else:
                for tok in line.split():
                    if tok.isdigit():
                        busy.add(int(tok))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception as e:
        logger.debug(f"rocm-smi --showpidgpus parse failed: {e}")
    return busy


def _amd_gpu_memory_usage() -> Dict[int, Tuple[float, float]]:
    """Return ``{device_id: (used_gb, total_gb)}`` for all AMD GPUs."""
    usage: Dict[int, Tuple[float, float]] = {}
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return usage
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.lower().startswith("device"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            m = re.search(r"(\d+)", parts[0])
            if not m:
                continue
            dev_id = int(m.group(1))
            try:
                total_b = int(parts[1])
                used_b = int(parts[2])
            except ValueError:
                continue
            usage[dev_id] = (used_b / (1024 ** 3), total_b / (1024 ** 3))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception as e:
        logger.debug(f"rocm-smi --showmeminfo parse failed: {e}")
    return usage


def _nvidia_busy_gpu_ids() -> set:
    """Return set of NVIDIA GPU indices with active compute processes."""
    busy: set = set()
    uuid_to_idx: Dict[str, int] = {}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[0].isdigit():
                    uuid_to_idx[parts[1]] = int(parts[0])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return busy

    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if not parts or not parts[0]:
                    continue
                idx = uuid_to_idx.get(parts[0])
                if idx is not None:
                    busy.add(idx)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return busy


def _nvidia_gpu_memory_usage() -> Dict[int, Tuple[float, float]]:
    """Return ``{device_id: (used_gb, total_gb)}`` for all NVIDIA GPUs."""
    usage: Dict[int, Tuple[float, float]] = {}
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return usage
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3 or not parts[0].isdigit():
                continue
            dev_id = int(parts[0])
            try:
                used_mb = float(parts[1])
                total_mb = float(parts[2])
            except ValueError:
                continue
            usage[dev_id] = (used_mb / 1024.0, total_mb / 1024.0)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return usage


def find_idle_gpus(
    count: int = 1,
    min_free_memory_gb: float = 1.0,
    candidates: Optional[List[int]] = None,
) -> List[int]:
    """Return up to ``count`` GPU indices that look idle.

    A GPU is considered idle when **both** are true:

      * No compute / KFD process is currently running on it.
      * Free VRAM is at least ``min_free_memory_gb``.

    Args:
        count: Number of idle GPUs to return.
        min_free_memory_gb: Reject GPUs with less free VRAM than this.
        candidates: Optional restricted device id pool. When ``None``,
            all GPUs reported by ``get_gpu_count()`` are considered.

    Returns:
        Device ids ordered most-free first, possibly shorter than
        ``count`` if not enough GPUs are idle.

    Raises:
        RuntimeError: when zero GPUs satisfy the criteria.
    """
    if count <= 0:
        return []

    vendor, _ = detect_gpu()
    total = get_gpu_count()
    if total == 0:
        raise RuntimeError("find_idle_gpus: no GPUs detected on this host")

    if candidates is None:
        candidates = list(range(total))
    else:
        candidates = [d for d in candidates if 0 <= d < total]

    if vendor == GPUVendor.AMD:
        busy = _amd_busy_gpu_ids()
        mem = _amd_gpu_memory_usage()
    elif vendor == GPUVendor.NVIDIA:
        busy = _nvidia_busy_gpu_ids()
        mem = _nvidia_gpu_memory_usage()
    else:
        raise RuntimeError("find_idle_gpus: unsupported GPU vendor")

    scored: List[Tuple[float, int]] = []
    rejected: List[str] = []
    for dev_id in candidates:
        if dev_id in busy:
            rejected.append(f"GPU {dev_id}: has running compute process")
            continue
        used_gb, total_gb = mem.get(dev_id, (0.0, 0.0))
        free_gb = max(0.0, total_gb - used_gb)
        if total_gb > 0 and free_gb < min_free_memory_gb:
            rejected.append(
                f"GPU {dev_id}: only {free_gb:.1f} GB free "
                f"(< {min_free_memory_gb:.1f} GB threshold)"
            )
            continue
        scored.append((free_gb, dev_id))

    scored.sort(key=lambda x: (-x[0], x[1]))
    chosen = [dev_id for _, dev_id in scored[:count]]

    if not chosen:
        msg = "find_idle_gpus: no idle GPU available"
        if rejected:
            msg += " (" + "; ".join(rejected) + ")"
        raise RuntimeError(msg)

    if len(chosen) < count:
        logger.warning(
            "find_idle_gpus: requested %d but only %d idle (%s); rejected: %s",
            count, len(chosen), chosen, rejected,
        )

    logger.info(
        "find_idle_gpus: selected %s (free GB: %s)",
        chosen,
        [f"{free:.1f}" for free, _ in scored[:count]],
    )
    return chosen


def get_gpu_count() -> int:
    """
    Get the number of available GPUs.

    Returns:
        Number of GPUs detected (0 if none)
    """
    # Try NVIDIA first
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # nvidia-smi returns one line per GPU
            lines = [
                line.strip()
                for line in result.stdout.strip().split("\n")
                if line.strip()
            ]
            return len(lines)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try AMD
    try:
        result = subprocess.run(
            ["rocm-smi", "--showid"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            # Count unique GPU IDs (each GPU has multiple lines in output)
            gpu_ids = set()
            for line in result.stdout.split("\n"):
                match = re.match(r"^\s*GPU\[(\d+)\]", line)
                if match:
                    gpu_ids.add(int(match.group(1)))
            if gpu_ids:
                return len(gpu_ids)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return 0


def list_gpus() -> List[GPUHardwareInfo]:
    """
    List all available GPUs with their hardware information.

    Returns:
        List of GPUHardwareInfo for each GPU
    """
    count = get_gpu_count()
    if count == 0:
        return []

    gpus = []
    for device_id in range(count):
        controller = GPUController(device_id=device_id)
        info = controller.get_hardware_info()
        gpus.append(info)

    return gpus


@dataclass
class MultiGPUConfig:
    """Configuration for multiple GPUs."""

    # Default configuration applied to all GPUs unless overridden
    default_config: Optional[GPUConfig] = None
    # Per-GPU configurations (device_id -> config)
    gpu_configs: Dict[int, GPUConfig] = field(default_factory=dict)
    # List of device IDs to manage (None = all available GPUs)
    device_ids: Optional[List[int]] = None
    # Whether to apply configurations in parallel
    parallel: bool = True

    def get_config_for_device(self, device_id: int) -> Optional[GPUConfig]:
        """Get configuration for a specific device."""
        if device_id in self.gpu_configs:
            return self.gpu_configs[device_id]
        if self.default_config:
            # Create a copy with the correct device_id
            config = GPUConfig(
                device_id=device_id,
                power_limit_watts=self.default_config.power_limit_watts,
                gpu_clock_mhz=self.default_config.gpu_clock_mhz,
                mem_clock_mhz=self.default_config.mem_clock_mhz,
                gpu_clock_level=self.default_config.gpu_clock_level,
                mem_clock_level=self.default_config.mem_clock_level,
            )
            return config
        return None


class MultiGPUController:
    """Controller for managing multiple GPUs."""

    def __init__(self, device_ids: Optional[List[int]] = None):
        """
        Initialize multi-GPU controller.

        Args:
            device_ids: List of GPU device IDs to manage.
                       If None, manages all available GPUs.
        """
        self.gpu_count = get_gpu_count()

        if device_ids is None:
            self.device_ids = list(range(self.gpu_count))
        else:
            # Validate device IDs
            self.device_ids = [d for d in device_ids if 0 <= d < self.gpu_count]

        # Create controllers for each GPU
        self.controllers: Dict[int, GPUController] = {}
        for device_id in self.device_ids:
            self.controllers[device_id] = GPUController(device_id=device_id)

    def get_all_hardware_info(
        self, parallel: bool = True
    ) -> Dict[int, GPUHardwareInfo]:
        """
        Get hardware information for all managed GPUs.

        Args:
            parallel: Whether to query GPUs in parallel

        Returns:
            Dictionary mapping device_id to GPUHardwareInfo
        """
        results = {}

        if parallel and len(self.device_ids) > 1:
            with ThreadPoolExecutor(max_workers=len(self.device_ids)) as executor:
                futures = {
                    executor.submit(ctrl.get_hardware_info): device_id
                    for device_id, ctrl in self.controllers.items()
                }
                for future in as_completed(futures):
                    device_id = futures[future]
                    try:
                        results[device_id] = future.result()
                    except Exception as e:
                        logger.error(f"Error getting info for GPU {device_id}: {e}")
                        results[device_id] = GPUHardwareInfo(
                            vendor=GPUVendor.UNKNOWN, device_id=device_id
                        )
        else:
            for device_id, controller in self.controllers.items():
                try:
                    results[device_id] = controller.get_hardware_info()
                except Exception as e:
                    logger.error(f"Error getting info for GPU {device_id}: {e}")
                    results[device_id] = GPUHardwareInfo(
                        vendor=GPUVendor.UNKNOWN, device_id=device_id
                    )

        return results

    def apply_config(self, config: MultiGPUConfig) -> Dict[int, bool]:
        """
        Apply configuration to multiple GPUs.

        Args:
            config: Multi-GPU configuration

        Returns:
            Dictionary mapping device_id to success status
        """
        results = {}
        device_ids = config.device_ids if config.device_ids else self.device_ids

        def apply_single(device_id: int) -> Tuple[int, bool]:
            if device_id not in self.controllers:
                return device_id, False
            gpu_config = config.get_config_for_device(device_id)
            if gpu_config is None:
                return device_id, True  # No config to apply
            return device_id, self.controllers[device_id].apply_config(gpu_config)

        if config.parallel and len(device_ids) > 1:
            with ThreadPoolExecutor(max_workers=len(device_ids)) as executor:
                futures = [executor.submit(apply_single, d) for d in device_ids]
                for future in as_completed(futures):
                    device_id, success = future.result()
                    results[device_id] = success
        else:
            for device_id in device_ids:
                _, success = apply_single(device_id)
                results[device_id] = success

        return results

    def reset_all(self, parallel: bool = True) -> Dict[int, bool]:
        """
        Reset all managed GPUs to default settings.

        Args:
            parallel: Whether to reset GPUs in parallel

        Returns:
            Dictionary mapping device_id to success status
        """
        results = {}

        def reset_single(device_id: int) -> Tuple[int, bool]:
            return device_id, self.controllers[device_id].reset_config()

        if parallel and len(self.device_ids) > 1:
            with ThreadPoolExecutor(max_workers=len(self.device_ids)) as executor:
                futures = [executor.submit(reset_single, d) for d in self.device_ids]
                for future in as_completed(futures):
                    device_id, success = future.result()
                    results[device_id] = success
        else:
            for device_id in self.device_ids:
                _, success = reset_single(device_id)
                results[device_id] = success

        return results

    def print_summary(self) -> None:
        """Print a summary of all managed GPUs."""
        infos = self.get_all_hardware_info()

        print(f"\n{'=' * 70}")
        print(f"GPU Summary ({len(infos)} GPUs)")
        print(f"{'=' * 70}")

        for device_id in sorted(infos.keys()):
            info = infos[device_id]
            print(f"\nGPU {device_id}: {info.device_name or 'Unknown'}")
            print(f"  Vendor: {info.vendor.name}")
            print(f"  Architecture: {info.architecture or 'N/A'}")
            print(
                f"  Power: {info.power_current_watts or 'N/A'}W / "
                f"{info.power_limit_watts or 'N/A'}W (limit)"
            )
            print(f"  GPU Clock: {info.gpu_clock_current or 'N/A'} MHz")
            print(f"  Mem Clock: {info.mem_clock_current or 'N/A'} MHz")
            print(f"  Temperature: {info.temperature or 'N/A'}°C")
            print(
                f"  Memory: {info.memory_used_gb or 0:.1f} / "
                f"{info.memory_total_gb or 0:.1f} GB"
            )
            if info.compute_spec:
                cs = info.compute_spec
                print(f"  Compute Units: {cs.compute_units or 'N/A'}")
                print(f"  SIMDs/CU: {cs.simds_per_cu or 'N/A'}")
                print(f"  Wavefront: {cs.wavefront_size or 'N/A'}")
                print(f"  LDS: {cs.lds_size_kb or 'N/A'} KB")
                if cs.l1_cache_kb:
                    print(f"  L1: {cs.l1_cache_kb} KB")
                if cs.l2_cache_kb:
                    print(f"  L2: {cs.l2_cache_kb} KB")
                if cs.l3_cache_kb:
                    print(f"  L3: {cs.l3_cache_kb} KB")

        print(f"\n{'=' * 70}\n")


def load_gpu_config_from_dict(
    config: Dict,
) -> Tuple[List[int], Optional[MultiGPUConfig]]:
    """
    Load GPU configuration from a config dictionary (parsed from YAML).

    This is the unified way to get GPU settings from config.yaml.
    Both scheduler and hardware controller should use this.

    Args:
        config: Dictionary containing 'gpu' section from config.yaml

    Returns:
        Tuple of (device_ids, MultiGPUConfig or None if hardware control disabled)

    Example:
        config = yaml.safe_load(open('config.yaml'))
        device_ids, gpu_config = load_gpu_config_from_dict(config)
    """
    gpu_section = config.get("gpu", {})

    # Get device IDs (unified for scheduler and hardware control)
    device_ids = gpu_section.get("device_ids")
    if device_ids is None:
        # Auto-detect all GPUs
        device_ids = list(range(get_gpu_count()))

    # Check if hardware control is enabled
    hardware = gpu_section.get("hardware", {})
    if not hardware.get("enabled", False):
        return device_ids, None

    # Parse hardware configuration
    default_section = hardware.get("default", {})
    per_gpu_section = hardware.get("per_gpu", {})

    # Build default config
    default_config = None
    if default_section:
        power = default_section.get("power", {})
        freq = default_section.get("frequency", {})

        gpu_clock_mhz = freq.get("gpu_clock_mhz")
        if gpu_clock_mhz and isinstance(gpu_clock_mhz, list):
            gpu_clock_mhz = tuple(gpu_clock_mhz)

        mem_clock_mhz = freq.get("mem_clock_mhz")
        if mem_clock_mhz and isinstance(mem_clock_mhz, list):
            mem_clock_mhz = tuple(mem_clock_mhz)

        default_config = GPUConfig(
            power_limit_watts=power.get("limit_watts"),
            gpu_clock_mhz=gpu_clock_mhz,
            mem_clock_mhz=mem_clock_mhz,
            gpu_clock_level=freq.get("gpu_clock_level"),
            mem_clock_level=freq.get("mem_clock_level"),
        )

    # Build per-GPU configs
    gpu_configs = {}
    for device_id_str, gpu_cfg in per_gpu_section.items():
        device_id = int(device_id_str)
        power = gpu_cfg.get("power", {})
        freq = gpu_cfg.get("frequency", {})

        gpu_clock_mhz = freq.get("gpu_clock_mhz")
        if gpu_clock_mhz and isinstance(gpu_clock_mhz, list):
            gpu_clock_mhz = tuple(gpu_clock_mhz)

        mem_clock_mhz = freq.get("mem_clock_mhz")
        if mem_clock_mhz and isinstance(mem_clock_mhz, list):
            mem_clock_mhz = tuple(mem_clock_mhz)

        gpu_configs[device_id] = GPUConfig(
            device_id=device_id,
            power_limit_watts=power.get("limit_watts"),
            gpu_clock_mhz=gpu_clock_mhz,
            mem_clock_mhz=mem_clock_mhz,
            gpu_clock_level=freq.get("gpu_clock_level"),
            mem_clock_level=freq.get("mem_clock_level"),
        )

    multi_gpu_config = MultiGPUConfig(
        default_config=default_config,
        gpu_configs=gpu_configs,
        device_ids=device_ids,
        parallel=gpu_section.get("parallel", True),
    )

    return device_ids, multi_gpu_config


def get_reset_after_benchmark(config: Dict) -> bool:
    """
    Check if GPUs should be reset after benchmark.

    Args:
        config: Dictionary containing 'gpu' section from config.yaml

    Returns:
        True if reset is enabled
    """
    gpu_section = config.get("gpu", {})
    hardware = gpu_section.get("hardware", {})
    return hardware.get("reset_after_benchmark", True)
