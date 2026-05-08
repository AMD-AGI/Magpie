###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Configuration classes for benchmark mode.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class BenchmarkFramework(Enum):
    """Supported benchmark frameworks."""

    VLLM = "vllm"
    SGLANG = "sglang"


class BenchmarkRunMode(Enum):
    """Benchmark execution mode."""

    DOCKER = "docker"
    LOCAL = "local"
    RAY = "ray"


class TraceLensExportFormat(Enum):
    """TraceLens export format options."""

    CSV = "csv"
    EXCEL = "excel"


# Default root on Ray workers for HF cache, InferenceX, and benchmark results.
# Use the same mount on driver and workers (NFS, Lustre, parallel filesystem, etc.).
DEFAULT_SHARED_STORAGE_PATH = "/shared_nfs/magpie"


@dataclass
class TorchProfilerConfig:
    """
    PyTorch Profiler configuration.

    Attributes:
        enabled: Whether torch_profiler is enabled (default: True)
    """

    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {"enabled": self.enabled}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TorchProfilerConfig":
        """Create from dictionary."""
        return cls(enabled=data.get("enabled", True))


@dataclass
class SystemProfilerConfig:
    """
    System-level profiler configuration (rocprof-compute / ncu).

    Attributes:
        enabled: Whether system profiler is enabled (default: False)
        profile_args: Additional arguments for profiler
    """

    enabled: bool = False
    profile_args: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "profile_args": self.profile_args,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemProfilerConfig":
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            profile_args=data.get("profile_args", []),
        )


@dataclass
class TraceLensConfig:
    """
    TraceLens trace analysis configuration.

    Supports two CLI commands:
    - TraceLens_generate_perf_report_pytorch: Single rank performance report
    - TraceLens_generate_multi_rank_collective_report_pytorch: Multi-rank collective analysis

    Attributes:
        enabled: Master switch for TraceLens analysis (default: False)
        export_format: Export format - "csv" or "excel" (default: "csv")
        perf_report_enabled: Enable single-rank performance report (default: True)
        multi_rank_report_enabled: Enable multi-rank collective report (default: True)
        gpu_arch_config: Path to GPU architecture JSON config for roofline (optional)
    """

    enabled: bool = False
    export_format: str = "csv"  # "csv" or "excel"

    # Command-specific enable/disable
    perf_report_enabled: bool = True
    multi_rank_report_enabled: bool = True

    # GPU architecture config (for roofline analysis)
    gpu_arch_config: Optional[str] = None

    def __post_init__(self):
        """Validate export format."""
        if self.export_format not in ["csv", "excel"]:
            raise ValueError(
                f"Invalid export_format: {self.export_format}. Use 'csv' or 'excel'."
            )

    @property
    def export_csv(self) -> bool:
        """Check if CSV export is enabled."""
        return self.export_format == "csv"

    @property
    def export_excel(self) -> bool:
        """Check if Excel export is enabled."""
        return self.export_format == "excel"

    # Internal defaults (not exposed to user config)
    # These follow TraceLens CLI defaults
    @property
    def collective_analysis(self) -> bool:
        """Collective analysis is enabled by default in TraceLens."""
        return True

    @property
    def short_kernel_study(self) -> bool:
        """Short kernel study is disabled by default in TraceLens."""
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "export_format": self.export_format,
            "perf_report_enabled": self.perf_report_enabled,
            "multi_rank_report_enabled": self.multi_rank_report_enabled,
            "gpu_arch_config": self.gpu_arch_config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceLensConfig":
        """Create from dictionary."""
        # Handle legacy config format
        export_format = data.get("export_format", "csv")
        if "export_format" not in data and (
            "export_csv" in data or "export_excel" in data
        ):
            # Legacy: export_csv=True means csv, export_excel=True means excel
            if data.get("export_excel", False):
                export_format = "excel"
            else:
                export_format = "csv"

        return cls(
            enabled=data.get("enabled", False),
            export_format=export_format,
            perf_report_enabled=data.get("perf_report_enabled", True),
            multi_rank_report_enabled=data.get("multi_rank_report_enabled", True),
            gpu_arch_config=data.get("gpu_arch_config"),
        )


@dataclass
class GPUMonitorConfig:
    """
    GPU hardware monitoring configuration.

    Collects temperature, clock frequencies, and power consumption
    during benchmark execution.

    Attributes:
        enabled: Whether GPU monitoring is enabled (default: True)
        interval_sec: Sampling interval in seconds (default: 2.0)
        device_id: GPU device to monitor (default: 0, or auto from benchmark)
    """

    enabled: bool = True
    interval_sec: float = 2.0
    device_id: Optional[int] = None  # None = auto-detect from benchmark

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "interval_sec": self.interval_sec,
            "device_id": self.device_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GPUMonitorConfig":
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            interval_sec=data.get("interval_sec", 2.0),
            device_id=data.get("device_id"),
        )


@dataclass
class ProfilerConfig:
    """
    Complete profiler configuration.

    Attributes:
        torch_profiler: PyTorch profiler settings (default enabled)
        system_profiler: System profiler settings (default disabled)
        tracelens: TraceLens trace analysis settings (default disabled)
        gpu_monitor: GPU hardware monitoring settings (default enabled)
    """

    torch_profiler: TorchProfilerConfig = field(default_factory=TorchProfilerConfig)
    system_profiler: SystemProfilerConfig = field(default_factory=SystemProfilerConfig)
    tracelens: TraceLensConfig = field(default_factory=TraceLensConfig)
    gpu_monitor: GPUMonitorConfig = field(default_factory=GPUMonitorConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "torch_profiler": self.torch_profiler.to_dict(),
            "system_profiler": self.system_profiler.to_dict(),
            "tracelens": self.tracelens.to_dict(),
            "gpu_monitor": self.gpu_monitor.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProfilerConfig":
        """Create from dictionary."""
        torch_cfg = data.get("torch_profiler", {})
        sys_cfg = data.get("system_profiler", {})
        tracelens_cfg = data.get("tracelens", {})
        gpu_monitor_cfg = data.get("gpu_monitor", {})
        return cls(
            torch_profiler=TorchProfilerConfig.from_dict(torch_cfg)
            if torch_cfg
            else TorchProfilerConfig(),
            system_profiler=SystemProfilerConfig.from_dict(sys_cfg)
            if sys_cfg
            else SystemProfilerConfig(),
            tracelens=TraceLensConfig.from_dict(tracelens_cfg)
            if tracelens_cfg
            else TraceLensConfig(),
            gpu_monitor=GPUMonitorConfig.from_dict(gpu_monitor_cfg)
            if gpu_monitor_cfg
            else GPUMonitorConfig(),
        )


@dataclass
class GapAnalysisConfig:
    """
    Gap analysis configuration for torch profiler trace analysis.

    Analyzes a time window of the trace to identify kernel-level bottlenecks.

    Attributes:
        enabled: Whether gap analysis is enabled (default: False)
        trace_start_pct: Start of analysis window as percentage of trace duration (0-100)
        trace_end_pct: End of analysis window as percentage of trace duration (0-100)
        top_k: Number of top bottleneck events to include in the report
        min_duration_us: Filter out events shorter than this (microseconds)
        categories: Event categories to include (e.g., ["kernel", "gpu"]). None = all.
        ignore_categories: Event categories to exclude (default: ["gpu_user_annotation", "user_annotation"])
        find_kernel_sources: Whether to find kernel source files and tests (default: False)
        kernel_source_repos: List of repository paths to search for kernel sources.
            If None and auto_clone_repos is True, repos are cloned on-demand based on kernel types.
        auto_clone_repos: Whether to auto-clone missing repositories (default: True).
            When enabled, required repos are detected from kernel names and cloned automatically.
        repos_base_dir: Base directory for auto-cloned repos. Defaults to ~/.cache/magpie/repos/
    """

    enabled: bool = False
    trace_start_pct: float = 0.0
    trace_end_pct: float = 100.0
    top_k: int = 20
    min_duration_us: float = 0.0
    categories: Optional[List[str]] = field(default_factory=lambda: ["kernel", "gpu"])
    ignore_categories: Optional[List[str]] = field(
        default_factory=lambda: ["gpu_user_annotation", "user_annotation"]
    )
    find_kernel_sources: bool = False
    kernel_source_repos: Optional[List[str]] = None
    auto_clone_repos: bool = True
    repos_base_dir: Optional[str] = None

    def __post_init__(self):
        """Validate percentage range."""
        if not (0.0 <= self.trace_start_pct <= 100.0):
            raise ValueError(
                f"trace_start_pct must be 0-100, got {self.trace_start_pct}"
            )
        if not (0.0 <= self.trace_end_pct <= 100.0):
            raise ValueError(f"trace_end_pct must be 0-100, got {self.trace_end_pct}")
        if self.trace_start_pct >= self.trace_end_pct:
            raise ValueError(
                f"trace_start_pct ({self.trace_start_pct}) must be less than "
                f"trace_end_pct ({self.trace_end_pct})"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "trace_start_pct": self.trace_start_pct,
            "trace_end_pct": self.trace_end_pct,
            "top_k": self.top_k,
            "min_duration_us": self.min_duration_us,
            "categories": self.categories,
            "ignore_categories": self.ignore_categories,
            "find_kernel_sources": self.find_kernel_sources,
            "kernel_source_repos": self.kernel_source_repos,
            "auto_clone_repos": self.auto_clone_repos,
            "repos_base_dir": self.repos_base_dir,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GapAnalysisConfig":
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            trace_start_pct=data.get("trace_start_pct", 0.0),
            trace_end_pct=data.get("trace_end_pct", 100.0),
            top_k=data.get("top_k", 20),
            min_duration_us=data.get("min_duration_us", 0.0),
            categories=data.get("categories", ["kernel", "gpu"]),
            ignore_categories=data.get("ignore_categories", ["gpu_user_annotation", "user_annotation"]),
            find_kernel_sources=data.get("find_kernel_sources", False),
            kernel_source_repos=data.get("kernel_source_repos"),
            auto_clone_repos=data.get("auto_clone_repos", True),
            repos_base_dir=data.get("repos_base_dir"),
        )


@dataclass
class RayConfig:
    """
    Configuration for Ray remote execution.

    Tasks are dispatched to GPU workers via ``ray.init()`` +
    ``@ray.remote(num_gpus=...)``.  Only the Ray GCS (port 6379) or
    Ray Client (port 10001) is required — **no Dashboard needed**.

    Attributes:
        cluster_address: How to connect to the Ray cluster.
            ``"auto"`` — on the head node (connects via local GCS).
            ``"ray://<host>:10001"`` — from a remote machine via Ray Client.
        shared_storage_path: Shared filesystem path on **worker** nodes for HF
            model cache and InferenceX (same mount on driver + workers).
        entrypoint_num_gpus: GPU resources requested per task.
        entrypoint_num_cpus: CPU resources requested per task.
        multi_node: Whether the benchmark requires multiple nodes.
        total_num_gpus: Total GPUs needed across all nodes (multi-node).
        num_nodes: Number of nodes required (multi-node).
        gpus_per_node: GPUs per node (multi-node resource calculation).
        pip_packages: Extra pip packages for the Ray ``runtime_env``.
        env_vars: Extra environment variables for the Ray job.
        metadata: Metadata tags attached to the Ray job.
        install_magpie: Auto-install Magpie + requirements on workers.
        magpie_install_path: Explicit Magpie project root for pip install.
    """

    cluster_address: str = "auto"
    shared_storage_path: str = DEFAULT_SHARED_STORAGE_PATH
    entrypoint_num_gpus: int = 0
    entrypoint_num_cpus: int = 16
    multi_node: bool = False
    total_num_gpus: int = 8
    num_nodes: int = 1
    gpus_per_node: int = 8
    pip_packages: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)
    install_magpie: bool = True
    magpie_install_path: Optional[str] = None

    @property
    def results_dir(self) -> str:
        """Directory for benchmark results on the shared storage."""
        return f"{self.shared_storage_path}/results"

    @property
    def hf_cache_dir(self) -> str:
        """HuggingFace model cache on the shared storage."""
        return f"{self.shared_storage_path}/hf_cache"

    @property
    def inferencex_dir(self) -> str:
        """InferenceX installation on the shared storage."""
        return f"{self.shared_storage_path}/InferenceX"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cluster_address": self.cluster_address,
            "shared_storage_path": self.shared_storage_path,
            "entrypoint_num_gpus": self.entrypoint_num_gpus,
            "entrypoint_num_cpus": self.entrypoint_num_cpus,
            "multi_node": self.multi_node,
            "total_num_gpus": self.total_num_gpus,
            "num_nodes": self.num_nodes,
            "gpus_per_node": self.gpus_per_node,
            "pip_packages": self.pip_packages,
            "env_vars": self.env_vars,
            "metadata": self.metadata,
            "install_magpie": self.install_magpie,
            "magpie_install_path": self.magpie_install_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RayConfig":
        """Create from dictionary."""
        return cls(
            cluster_address=data.get("cluster_address", "auto"),
            shared_storage_path=data.get(
                "shared_storage_path", DEFAULT_SHARED_STORAGE_PATH
            ),
            entrypoint_num_gpus=data.get("entrypoint_num_gpus", 0),
            entrypoint_num_cpus=data.get("entrypoint_num_cpus", 16),
            multi_node=data.get("multi_node", False),
            total_num_gpus=data.get("total_num_gpus", 8),
            num_nodes=data.get("num_nodes", 1),
            gpus_per_node=data.get("gpus_per_node", 8),
            pip_packages=data.get("pip_packages", []),
            env_vars=data.get("env_vars", {}),
            metadata=data.get("metadata", {}),
            install_magpie=data.get("install_magpie", True),
            magpie_install_path=data.get("magpie_install_path"),
        )


@dataclass
class GpuSelectionConfig:
    """
    Auto-select idle GPU(s) before launching the benchmark.

    When enabled, Magpie scans the host (rocm-smi / nvidia-smi) for GPUs
    with no compute/KFD processes and at least ``min_free_memory_gb`` of
    free VRAM, then injects ``HIP_VISIBLE_DEVICES`` /
    ``CUDA_VISIBLE_DEVICES`` / ``ROCR_VISIBLE_DEVICES`` into the benchmark
    environment so vLLM/SGLang only sees the chosen device(s).

    Attributes:
        auto: If True (default), perform the selection. Set to False to
            keep the legacy behaviour where every GPU on the host is
            visible and the framework picks device 0.
        min_free_memory_gb: Reject a GPU whose free VRAM is below this
            threshold. Default 8 GB (any almost-empty GPU is acceptable).
        count: Number of GPUs to select. ``None`` means use ``envs.TP``.
        candidates: Optional whitelist of physical GPU indices to consider.
    """
    auto: bool = True
    min_free_memory_gb: float = 8.0
    count: Optional[int] = None
    candidates: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "auto": self.auto,
            "min_free_memory_gb": self.min_free_memory_gb,
            "count": self.count,
            "candidates": self.candidates,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GpuSelectionConfig":
        return cls(
            auto=bool(data.get("auto", True)),
            min_free_memory_gb=float(data.get("min_free_memory_gb", 8.0)),
            count=data.get("count"),
            candidates=data.get("candidates"),
        )


@dataclass
class ServerLifecycleConfig:
    """
    Persist a benchmark inference server across multiple local runs.

    When enabled, ``timeout_seconds`` applies to the client (benchmark serving)
    phase only; server startup is gated by ``server_ready_timeout_s``.
    Server processes are only recycled when ``cleanup`` is True for a run.

    Intended for ``run_mode: local`` with Magpie built-in benchmarks scripts
    (``vllm_*.sh`` / ``sglang_*.sh``) that honour ``MAGPIE_RUN_PHASE``.
    """

    enabled: bool = False
    cleanup: bool = False
    force_reuse: bool = False
    pid_dir: Optional[str] = None
    server_ready_timeout_s: int = 2700

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "cleanup": self.cleanup,
            "force_reuse": self.force_reuse,
            "pid_dir": self.pid_dir,
            "server_ready_timeout_s": self.server_ready_timeout_s,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ServerLifecycleConfig":
        if not data:
            return cls(enabled=False)
        return cls(
            enabled=bool(data.get("enabled", False)),
            cleanup=bool(data.get("cleanup", False)),
            force_reuse=bool(data.get("force_reuse", False)),
            pid_dir=data.get("pid_dir"),
            server_ready_timeout_s=int(data.get("server_ready_timeout_s", 2700)),
        )


@dataclass
class BenchmarkConfig:
    """
    Configuration for benchmark mode.

    Attributes:
        framework: Benchmark framework ("vllm" or "sglang")
        model: Model name or path (e.g., "meta-llama/Llama-2-7b-hf")
        precision: Model precision ("fp8", "fp16", "bf16", "fp4")
        run_mode: Execution mode - "docker" (default), "local", or "ray"
        envs: Environment variables for benchmark (TP, CONC, ISL, OSL, etc.)
        profiler: Profiler configuration
        docker_image: Override automatic image selection
        gpu_arch: GPU architecture (auto-detected if not specified)
        timeout_seconds: Benchmark timeout
        inferencex_path: Path to InferenceX installation
        hf_cache_path: HuggingFace cache directory
        runner_type: Hardware runner type for InferenceX (e.g., "mi300x", "h100")
        server_lifecycle: Optional persisted-server settings (local-only)
    """

    framework: str
    model: str
    precision: str = "fp8"

    # Execution mode: "docker", "local", or "ray"
    run_mode: str = "docker"

    # Environment variables for benchmark
    envs: Dict[str, Any] = field(default_factory=dict)

    # Profiler configuration
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)

    # Docker/execution settings
    docker_image: Optional[str] = None
    gpu_arch: Optional[str] = None
    timeout_seconds: float = 3600.0

    # Paths
    # Empty/None triggers auto-resolution in InferenceXManager:
    #   1) $MAGPIE_INFERENCEX_PATH  2) ./InferenceX next to Magpie repo
    #   3) ~/.cache/magpie/InferenceX
    inferencex_path: str = ""
    hf_cache_path: Optional[str] = None

    # Gap analysis
    gap_analysis: GapAnalysisConfig = field(default_factory=GapAnalysisConfig)

    # InferenceX specific
    runner_type: Optional[str] = None
    benchmark_script: Optional[str] = None
    
    # GPU auto-selection (skip cards with running processes / low free VRAM)
    gpu_selection: GpuSelectionConfig = field(default_factory=GpuSelectionConfig)

    # Ray remote execution configuration (used when run_mode="ray")
    ray_config: Optional[RayConfig] = None

    # Persist inference server across local benchmark runs (opt-in).
    server_lifecycle: Optional[ServerLifecycleConfig] = None

    def __post_init__(self):
        """Validate and set defaults."""
        # Normalize framework name
        self.framework = self.framework.lower()
        if self.framework not in ["vllm", "sglang"]:
            raise ValueError(
                f"Unsupported framework: {self.framework}. Use 'vllm' or 'sglang'."
            )

        # Validate run_mode
        self.run_mode = self.run_mode.lower()
        if self.run_mode not in ("docker", "local", "ray"):
            raise ValueError(
                f"Unsupported run_mode: {self.run_mode}. Use 'docker', 'local', or 'ray'."
            )

        # Set default envs if not provided
        if not self.envs:
            self.envs = {
                "TP": 1,
                "CONC": 32,
                "ISL": 1024,
                "OSL": 512,
                "RANDOM_RANGE_RATIO": 0.5,
            }

        # Convert profiler dict to ProfilerConfig if needed
        if isinstance(self.profiler, dict):
            self.profiler = ProfilerConfig.from_dict(self.profiler)

        # Convert gap_analysis dict to GapAnalysisConfig if needed
        if isinstance(self.gap_analysis, dict):
            self.gap_analysis = GapAnalysisConfig.from_dict(self.gap_analysis)
        
        # Convert gpu_selection dict to GpuSelectionConfig if needed
        if isinstance(self.gpu_selection, dict):
            self.gpu_selection = GpuSelectionConfig.from_dict(self.gpu_selection)

        # Convert ray_config dict to RayConfig if needed
        if isinstance(self.ray_config, dict):
            self.ray_config = RayConfig.from_dict(self.ray_config)

        # Ensure ray_config exists when run_mode is "ray"
        if self.run_mode == "ray" and self.ray_config is None:
            self.ray_config = RayConfig()

        if isinstance(self.server_lifecycle, dict):
            self.server_lifecycle = ServerLifecycleConfig.from_dict(
                self.server_lifecycle
            )

        if self.is_server_lifecycle:
            if self.run_mode != "local":
                raise ValueError(
                    "server_lifecycle.enabled requires run_mode='local'. "
                    "Docker/Ray executions cannot reuse a server process "
                    "across Magpie invocations."
                )
            lc = self.server_lifecycle
            assert lc is not None
            if self.profiler.torch_profiler.enabled and not lc.cleanup:
                raise ValueError(
                    "server_lifecycle is incompatible with "
                    "profiler.torch_profiler.enabled=true when cleanup=false. "
                    "Disable torch_profiler for reuse runs or set cleanup=true "
                    "so the profiled server terminates with the benchmark."
                )

    def get_env_vars(self) -> Dict[str, str]:
        """
        Get environment variables for InferenceX.

        Returns:
            Dictionary of environment variable names to values
        """
        env = {
            "MODEL": self.model,
            "PRECISION": self.precision,
        }

        # Add all envs as environment variables
        for key, value in self.envs.items():
            env[key.upper()] = str(value)

        # Add runner type if specified
        if self.runner_type:
            env["RUNNER_TYPE"] = self.runner_type

        return env

    def get_benchmark_script_name(self) -> str:
        """
        Determine the InferenceX benchmark script name.

        Returns:
            Script name like "dsr1_fp8_mi300x.sh"
        """
        if self.benchmark_script:
            return self.benchmark_script

        # Auto-generate based on config
        runner = self.runner_type or "mi300x"
        # Format: {exp_name}_{precision}_{runner}.sh
        # For now, use a generic experiment name
        return f"generic_{self.precision}_{runner}.sh"

    @property
    def is_local(self) -> bool:
        """Check if running in local mode (no Docker)."""
        return self.run_mode == "local"

    @property
    def is_ray(self) -> bool:
        """Check if running in Ray remote execution mode."""
        return self.run_mode == "ray"

    @property
    def is_server_lifecycle(self) -> bool:
        """Reuse a shared inference server across local benchmark tasks."""
        return self.server_lifecycle is not None and bool(self.server_lifecycle.enabled)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d: Dict[str, Any] = {
            "framework": self.framework,
            "model": self.model,
            "precision": self.precision,
            "run_mode": self.run_mode,
            "envs": self.envs,
            "profiler": self.profiler.to_dict(),
            "gap_analysis": self.gap_analysis.to_dict(),
            "docker_image": self.docker_image,
            "gpu_arch": self.gpu_arch,
            "timeout_seconds": self.timeout_seconds,
            "inferencex_path": self.inferencex_path,
            "hf_cache_path": self.hf_cache_path,
            "runner_type": self.runner_type,
            "benchmark_script": self.benchmark_script,
            "gpu_selection": self.gpu_selection.to_dict(),
        }
        if self.ray_config is not None:
            d["ray_config"] = self.ray_config.to_dict()
        if self.server_lifecycle is not None:
            d["server_lifecycle"] = self.server_lifecycle.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkConfig":
        """Create from dictionary."""
        profiler_data = data.get("profiler", {})
        profiler = (
            ProfilerConfig.from_dict(profiler_data)
            if profiler_data
            else ProfilerConfig()
        )

        gap_data = data.get("gap_analysis", {})
        gap_analysis = (
            GapAnalysisConfig.from_dict(gap_data) if gap_data else GapAnalysisConfig()
        )

        ray_data = data.get("ray_config")
        ray_config = RayConfig.from_dict(ray_data) if ray_data else None

        sl_raw = data.get("server_lifecycle")
        server_lifecycle = (
            ServerLifecycleConfig.from_dict(sl_raw)
            if isinstance(sl_raw, dict)
            else (sl_raw if isinstance(sl_raw, ServerLifecycleConfig) else None)
        )
        sweep_matrix = data.get("sweep_matrix")
        cases_forbidden = isinstance(sweep_matrix, dict) and bool(
            sweep_matrix.get("cases")
        )
        if server_lifecycle and server_lifecycle.enabled and cases_forbidden:
            raise ValueError(
                "server_lifecycle cannot be combined with sweep_matrix "
                "(non-empty sweep_matrix.cases)."
            )

        return cls(
            framework=data.get("framework", "sglang"),
            model=data.get("model", ""),
            precision=data.get("precision", "fp8"),
            run_mode=data.get("run_mode", "docker"),
            envs=data.get("envs", {}),
            profiler=profiler,
            gap_analysis=gap_analysis,
            docker_image=data.get("docker_image"),
            gpu_arch=data.get("gpu_arch"),
            timeout_seconds=data.get("timeout_seconds", 3600.0),
            inferencex_path=(
                data.get("inferencex_path")
                or data.get("inferencemax_path")
                or ""
            ),
            hf_cache_path=data.get("hf_cache_path"),
            runner_type=data.get("runner_type"),
            benchmark_script=data.get("benchmark_script"),
            gpu_selection=GpuSelectionConfig.from_dict(data.get("gpu_selection") or {}),
            ray_config=ray_config,
            server_lifecycle=server_lifecycle,
        )
