###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Core benchmarker for benchmark mode.

Orchestrates benchmark execution using InferenceX as backend.
"""

import logging
import os
import signal
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.ray_executor import RayJobExecutor
    from ...core.task import Task

from .config import BenchmarkConfig
from .image_selector import ImageSelector
from .inferencex import ensure_inferencex_available
from .workspace import WorkspaceManager
from .result import BenchmarkResult, LatencyMetrics, ResultParser, ThroughputMetrics
from .tracelens import TraceLensAnalyzer

from ...utils.gpu import detect_gpu, find_idle_gpus, GPUVendor
from ...utils.gpu_monitor import GPUMonitor

logger = logging.getLogger(__name__)


class BenchmarkMode:
    """
    Benchmark mode for framework-level profiling.
    
    Uses InferenceX as backend to run vLLM/SGLang benchmarks
    either inside Docker containers (run_mode=docker), directly
    on the host (run_mode=local), or on a remote Ray cluster
    (run_mode=ray).
    """
    
    def __init__(
        self,
        config: BenchmarkConfig,
        image_config_path: Optional[str] = None,
        output_dir: str = "./results",
    ):
        """
        Initialize benchmark mode.
        
        Args:
            config: Benchmark configuration
            image_config_path: Path to benchmark_images.yaml
            output_dir: Base directory for results
        """
        self.config = config
        self.image_selector = ImageSelector(image_config_path)
        self.workspace_mgr = WorkspaceManager(
            base_dir=output_dir,
            framework=config.framework,
        )
        self._task_id: Optional[str] = None
    
    def run(self, task_id: Optional[str] = None) -> BenchmarkResult:
        """
        Run benchmark.
        
        Args:
            task_id: Optional task identifier for container naming
        
        Returns:
            BenchmarkResult with metrics and profiling data
        """
        self._task_id = task_id or f"bench_{int(time.time())}"
        start_time = time.time()
        
        logger.info(f"Starting benchmark: {self.config.framework} / {self.config.model}")
        
        # Ray mode: delegate to remote cluster and return when complete
        if self.config.is_ray:
            return self._execute_ray_benchmark()
        
        # 0. Ensure InferenceX is available (auto-clone if needed)
        try:
            self.config.inferencex_path = ensure_inferencex_available(
                self.config.inferencex_path
            )
        except RuntimeError as e:
            result = BenchmarkResult()
            result.success = False
            result.errors.append(f"Failed to setup InferenceX: {e}")
            result.errors.append(
                "Please clone manually: "
                "git clone https://github.com/SemiAnalysisAI/InferenceX.git"
            )
            return result
        
        # 0b. Optionally pick idle GPU(s) and pin VISIBLE_DEVICES
        try:
            self._apply_gpu_selection()
        except RuntimeError as e:
            result = BenchmarkResult()
            result.success = False
            result.errors.append(str(e))
            return result

        # 1. Copy Magpie generic scripts to InferenceX/benchmarks/
        self._prepare_benchmark_scripts()
        
        # 2. Determine runner type from GPU
        runner_type = self._get_runner_type()
        
        # 3. Find and validate benchmark script BEFORE container starts
        try:
            benchmark_script = self._get_benchmark_script(runner_type)
            logger.info(f"Selected benchmark script: {benchmark_script}")
        except FileNotFoundError as e:
            result = BenchmarkResult()
            result.success = False
            result.errors.append(str(e))
            return result
        
        # 4. Create workspace
        workspace = self.workspace_mgr.create(self.config.to_dict())
        
        # 4b. Start GPU monitor if enabled
        gpu_monitor = None
        gpu_monitor_stats = None
        if self.config.profiler.gpu_monitor.enabled:
            monitor_cfg = self.config.profiler.gpu_monitor
            device_id = monitor_cfg.device_id
            if device_id is None:
                # Auto-detect from GPU selection or default to 0
                envs = self.config.envs or {}
                visible = envs.get("HIP_VISIBLE_DEVICES") or envs.get("CUDA_VISIBLE_DEVICES") or envs.get("ROCR_VISIBLE_DEVICES")
                if visible:
                    device_id = int(visible.split(",")[0])
                else:
                    device_id = 0
            gpu_monitor = GPUMonitor(
                device_id=device_id,
                interval_sec=monitor_cfg.interval_sec,
            )
            if gpu_monitor.start():
                logger.info(f"GPU monitor started (device={device_id}, interval={monitor_cfg.interval_sec}s)")
            else:
                # Unsupported vendor, disable monitor
                gpu_monitor = None
        
        # 5-7. Build and execute (Docker or local)
        if self.config.is_local:
            local_cmd, local_env = self._build_local_command(
                workspace=workspace,
                runner_type=runner_type,
            )
            logger.info("Running benchmark locally (no Docker)")
            logger.debug(f"Local command: {' '.join(local_cmd)}")

            symlink = self._create_workspace_symlink(workspace)
            try:
                result, stdout, stderr = self._execute_local_benchmark(
                    local_cmd, local_env, workspace,
                )
            finally:
                self._remove_workspace_symlink(symlink)
                self._cleanup_server_processes(self.config.framework)
        else:
            docker_image = self._select_image()
            docker_cmd = self._build_docker_command(
                docker_image=docker_image,
                workspace=workspace,
                runner_type=runner_type,
            )
            logger.info(f"Running benchmark in container with image: {docker_image}")
            logger.debug(f"Docker command: {' '.join(docker_cmd)}")
            result, stdout, stderr = self._execute_benchmark(docker_cmd, workspace)
        
        # 7b. Stop GPU monitor and collect stats
        if gpu_monitor is not None:
            gpu_monitor_stats = gpu_monitor.stop()
            logger.info(f"GPU monitor stopped: {gpu_monitor_stats.sample_count} samples collected")
        
        # 8. Collect results
        result.workspace_dir = str(workspace)
        result.execution_time = time.time() - start_time
        result.framework = self.config.framework
        result.model = self.config.model
        result.profiling_enabled = self.config.profiler.torch_profiler.enabled
        
        # Add GPU monitor stats
        if gpu_monitor_stats is not None:
            result.gpu_monitor = gpu_monitor_stats.to_dict()
        
        # Parse InferenceX output
        result_file = workspace / "inferencex_result.json"
        if result_file.exists():
            parsed = ResultParser.parse_inferencex_result(
                result_file,
                framework=self.config.framework,
                model=self.config.model,
            )
            # Merge parsed results
            result.throughput = parsed.throughput
            result.latency = parsed.latency
            result.raw_result = parsed.raw_result
            if parsed.errors:
                result.errors.extend(parsed.errors)
        else:
            result.success = False
            mode_label = "locally" if self.config.is_local else "inside container"
            result.errors.append(
                f"inferencex_result.json not found - benchmark may have failed {mode_label}"
            )
            if stderr:
                result.errors.append(f"stderr (last 500 chars): {stderr[-500:]}")

            server_log = workspace / "server.log"
            if server_log.exists():
                try:
                    lines = server_log.read_text().splitlines()
                    error_lines = [
                        line
                        for line in lines[-50:]
                        if any(
                            kw in line
                            for kw in [
                                "Error",
                                "Exception",
                                "FAILED",
                                "Traceback",
                                "RuntimeError",
                            ]
                        )
                    ]
                    if error_lines:
                        result.errors.append(
                            f"server.log errors: {chr(10).join(error_lines[-5:])}"
                        )
                except Exception:
                    pass
        
        # Validate that we got actual results
        if result.success and not self._validate_results(result):
            result.success = False
            result.errors.append("Benchmark produced no valid throughput/latency metrics")
        
        # Parse torch trace if available
        if self.config.profiler.torch_profiler.enabled:
            torch_trace_dir = workspace / "torch_trace"
            trace_files = list(torch_trace_dir.glob("*.json.gz")) if torch_trace_dir.is_dir() else []
            has_traces = len(trace_files) > 0

            if not result.success or not has_traces:
                if not has_traces:
                    logger.warning("No torch trace files found, skipping trace analysis / gap analysis")
                else:
                    logger.warning("Benchmark failed, skipping trace analysis / gap analysis")
            else:
                kernels = ResultParser.parse_torch_trace(torch_trace_dir)
                result.kernel_summary = kernels
                result.top_bottlenecks = [k.name for k in kernels[:10]]

                if self.config.profiler.tracelens.enabled:
                    tracelens_result = self._run_tracelens_analysis(
                        torch_trace_dir, workspace
                    )
                    result.tracelens_analysis = tracelens_result

                if self.config.gap_analysis.enabled:
                    gap_result = self._run_gap_analysis(torch_trace_dir, workspace)
                    result.gap_analysis = gap_result

        # Save report
        self.workspace_mgr.save_report(result.to_dict())
        self.workspace_mgr.save_summary(result.get_summary())
        
        logger.info(f"Benchmark completed in {result.execution_time:.2f}s")
        
        return result
    
    def _apply_gpu_selection(self) -> None:
        """Resolve idle GPU(s) and inject the selection into config.envs.

        No-op when ``self.config.gpu_selection.auto`` is False, when the
        caller has already pinned ``HIP_VISIBLE_DEVICES`` /
        ``CUDA_VISIBLE_DEVICES`` / ``ROCR_VISIBLE_DEVICES`` in the yaml,
        or when run_mode is ``ray`` (Ray manages its own scheduling).
        """
        sel = self.config.gpu_selection
        if not sel.auto:
            return
        if self.config.is_ray:
            logger.info("gpu_selection.auto ignored under run_mode=ray")
            return

        # Honor manual override from yaml envs
        envs_upper = {str(k).upper(): v for k, v in self.config.envs.items()}
        for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                    "ROCR_VISIBLE_DEVICES"):
            if envs_upper.get(var) not in (None, ""):
                logger.info(
                    "gpu_selection.auto skipped: %s already set in envs (%s)",
                    var, envs_upper.get(var),
                )
                return

        # Determine how many GPUs we need
        count = sel.count
        if count is None:
            try:
                count = max(1, int(envs_upper.get("TP", 1)))
            except (TypeError, ValueError):
                count = 1

        try:
            chosen = find_idle_gpus(
                count=count,
                min_free_memory_gb=sel.min_free_memory_gb,
                candidates=sel.candidates,
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"gpu_selection.auto failed: {e}. "
                "Free a GPU, raise gpu_selection.min_free_memory_gb, or set "
                "HIP_VISIBLE_DEVICES manually in envs."
            ) from e

        if len(chosen) < count:
            raise RuntimeError(
                f"gpu_selection.auto: needed {count} idle GPU(s) but only "
                f"found {len(chosen)} ({chosen})"
            )

        dev_str = ",".join(str(d) for d in chosen)

        # AMD: set ROCR_VISIBLE_DEVICES (shares index space with rocm-smi);
        #      launcher scripts remap HIP to the post-filter 0..N-1 range.
        # NVIDIA: force PCI_BUS_ID order so CUDA_VISIBLE_DEVICES matches nvidia-smi.
        vendor, _ = detect_gpu()
        if vendor == GPUVendor.AMD:
            self.config.envs["ROCR_VISIBLE_DEVICES"] = dev_str
        else:
            self.config.envs["CUDA_VISIBLE_DEVICES"] = dev_str
            self.config.envs.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

        logger.info(
            "gpu_selection.auto: pinned benchmark to physical GPU(s) %s "
            "(vendor=%s)", dev_str, vendor.name,
        )

    def _select_image(self) -> str:
        """Select Docker image based on configuration."""
        return self.image_selector.select_image(
            framework=self.config.framework,
            gpu_arch=self.config.gpu_arch,
            override_image=self.config.docker_image,
        )
    
    def _get_runner_type(self) -> str:
        """Get InferenceX runner type."""
        if self.config.runner_type:
            return self.config.runner_type
        return self.image_selector.get_runner_type(self.config.gpu_arch)
    
    def _prepare_benchmark_scripts(self) -> None:
        """
        Copy Magpie generic benchmark scripts to InferenceX/benchmarks/.
        
        This allows using Magpie's generic scripts while still leveraging
        InferenceX's benchmark_lib.sh and other utilities.
        
        Always overwrites to keep scripts in sync with Magpie source.
        """
        # Magpie scripts location: Magpie/scripts/benchmark/
        magpie_scripts = Path(__file__).parent.parent.parent / "scripts" / "benchmark"
        target_dir = Path(self.config.inferencex_path) / "benchmarks"
        
        if not magpie_scripts.exists():
            logger.debug(f"Magpie scripts directory not found: {magpie_scripts}")
            return
        
        # Ensure target directory exists
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all .sh scripts (always overwrite to keep in sync with Magpie source)
        for script in magpie_scripts.glob("*.sh"):
            target_file = target_dir / script.name
            shutil.copy2(script, target_file)
            target_file.chmod(0o755)
            logger.info(f"Copied Magpie script {script.name} to {target_dir}")
    
    def _validate_results(self, result: BenchmarkResult) -> bool:
        """
        Validate that benchmark produced meaningful results.
        
        Args:
            result: BenchmarkResult to validate
        
        Returns:
            True if results are valid, False otherwise
        """
        # Check if we have throughput metrics with actual values
        if result.throughput:
            if result.throughput.request_throughput > 0:
                return True
            if result.throughput.output_throughput > 0:
                return True
            if result.throughput.completed_requests > 0:
                return True
        
        # Check if we have latency metrics
        if result.latency:
            if result.latency.ttft_mean > 0 or result.latency.e2el_mean > 0:
                return True
        
        return False
    
    def _build_docker_command(
        self,
        docker_image: str,
        workspace: Path,
        runner_type: str,
    ) -> List[str]:
        """
        Build Docker run command.
        
        Args:
            docker_image: Docker image to use
            workspace: Workspace directory path
            runner_type: InferenceX runner type
        
        Returns:
            Docker command as list of strings
        """
        # Detect GPU vendor for device flags
        vendor, arch = detect_gpu()
        
        cmd = [
            "docker", "run", "--rm",
            "--ipc=host", "--shm-size=16g", "--network=host",
            "--name", f"magpie-benchmark-{self._task_id}",
        ]
        
        # Add GPU-specific flags
        if vendor == GPUVendor.AMD:
            cmd.extend([
                "--privileged",
                "--cap-add=CAP_SYS_ADMIN",
                "--device=/dev/kfd",
                "--device=/dev/dri",
                "--device=/dev/mem",
                "--cap-add=SYS_PTRACE",
                "--security-opt", "seccomp=unconfined",
            ])
        elif vendor == GPUVendor.NVIDIA:
            cmd.extend([
                "--gpus", "all",
            ])
        
        # HuggingFace cache mount
        hf_cache = self.config.hf_cache_path or os.path.expanduser("~/.cache/huggingface")
        if os.path.exists(hf_cache):
            cmd.extend(["-v", f"{hf_cache}:/root/.cache/huggingface"])
        
        # InferenceX mount
        inferencex_path = self.config.inferencex_path
        if os.path.exists(inferencex_path):
            cmd.extend(["-v", f"{inferencex_path}:/opt/InferenceX"])

        # Model directory mount — if the model path is a local directory, mount it
        # so the container can access the weights (e.g. /mnt/dcgpuval/datasets/...)
        model_path = self.config.model
        if model_path and os.path.isdir(model_path):
            cmd.extend(["-v", f"{model_path}:{model_path}"])
        
        # Workspace mount — map directly to /workspace so all output
        # (server.log, results, torch_trace/) lands on the host automatically
        cmd.extend(["-v", f"{workspace}:/workspace"])
        
        # Environment variables
        env_vars = self.config.get_env_vars()
        env_vars["RESULT_FILENAME"] = "inferencex_result"
        env_vars["RESULT_DIR"] = "/workspace"
        env_vars["RUNNER_TYPE"] = runner_type
        
        # torch_profiler environment (matches official InferenceX: PROFILE=1)
        if self.config.profiler.torch_profiler.enabled:
            env_vars["PROFILE"] = "1"
            env_vars["VLLM_TORCH_PROFILER_DIR"] = "/workspace/torch_trace"
            env_vars["SGLANG_TORCH_PROFILER_DIR"] = "/workspace/torch_trace"

        
        # HuggingFace token from environment
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            env_vars["HF_TOKEN"] = hf_token
        
        for key, value in env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])
        
        # Working directory
        cmd.extend(["-w", "/opt/InferenceX"])
        
        # Image and entrypoint - always override to bash for script compatibility
        cmd.extend(["--entrypoint", "/bin/bash"])
        cmd.append(docker_image)
        
        # Build the benchmark command
        benchmark_script = self._get_benchmark_script(runner_type)
        
        # With --entrypoint /bin/bash, pass -c as first arg
        cmd.extend([
            "-c",
            f"cd /opt/InferenceX && bash {benchmark_script}"
        ])
        
        return cmd
    
    def _create_workspace_symlink(self, workspace: Path) -> Optional[Path]:
        """
        Create /workspace symlink pointing to the real workspace directory.
        
        Skipped: RESULT_DIR and SERVER_LOG are already passed via env vars
        in _build_local_command, so InferenceX scripts resolve paths correctly
        without a /workspace symlink. Creating symlinks in pods causes stale
        references when benchmarks are interrupted.
        
        Returns:
            Always None (symlink creation disabled).
        """
        logger.debug(
            f"Skipping /workspace symlink (RESULT_DIR={workspace} is passed via env)"
        )
        return None

    @staticmethod
    def _remove_workspace_symlink(symlink: Optional[Path]) -> None:
        """Remove the /workspace symlink created by _create_workspace_symlink."""
        if symlink is None:
            return
        try:
            if symlink.is_symlink():
                symlink.unlink()
                logger.info(f"Removed symlink {symlink}")
        except OSError as e:
            logger.warning(f"Could not remove symlink {symlink}: {e}")

    def _build_local_command(
        self,
        workspace: Path,
        runner_type: str,
    ) -> tuple:
        """
        Build command and environment for local (non-Docker) execution.
        
        Runs the benchmark script directly on the host via bash.
        Useful when already inside a container/pod with the required
        runtime (vLLM/SGLang) installed.
        
        Args:
            workspace: Workspace directory path
            runner_type: InferenceX runner type
        
        Returns:
            Tuple of (command list, environment dict)
        """
        benchmark_script = self._get_benchmark_script(runner_type)
        inferencex_path = str(Path(self.config.inferencex_path).resolve())

        cmd = [
            "bash", "-c",
            f"cd {inferencex_path} && bash {benchmark_script}",
        ]

        env = os.environ.copy()

        env_vars = self.config.get_env_vars()
        env_vars["RESULT_FILENAME"] = "inferencex_result"
        env_vars["RESULT_DIR"] = str(workspace)
        env_vars["RUNNER_TYPE"] = runner_type

        if self.config.profiler.torch_profiler.enabled:
            torch_trace_dir = workspace / "torch_trace"
            torch_trace_dir.mkdir(parents=True, exist_ok=True)
            env_vars["PROFILE"] = "1"
            env_vars["VLLM_TORCH_PROFILER_DIR"] = str(torch_trace_dir)
            env_vars["SGLANG_TORCH_PROFILER_DIR"] = str(torch_trace_dir)

        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            env_vars["HF_TOKEN"] = hf_token

        env_vars["SERVER_LOG"] = str(workspace / "server.log")

        for key, value in env_vars.items():
            env[key] = str(value)

        return cmd, env

    def _execute_local_benchmark(
        self,
        cmd: List[str],
        env: Dict[str, str],
        workspace: Path,
    ) -> tuple:
        """
        Execute benchmark locally (no Docker).
        
        Args:
            cmd: Shell command list
            env: Environment variables
            workspace: Workspace directory for saving logs
        
        Returns:
            Tuple of (BenchmarkResult, stdout, stderr)
        """
        result = BenchmarkResult()
        stdout = ""
        stderr = ""

        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                env=env,
            )

            stdout = process.stdout or ""
            stderr = process.stderr or ""

            self._save_logs(workspace, stdout, stderr)

            if process.returncode == 0:
                result.success = True
                logger.info("Local benchmark completed successfully")
            else:
                result.success = False
                result.errors.append(
                    f"Benchmark process failed with code {process.returncode}"
                )
                if stderr:
                    result.errors.append(f"stderr: {stderr[:1000]}")
                logger.error(f"Benchmark failed: {process.returncode}")
                logger.error(f"stderr: {stderr[:500]}")

            if stdout:
                logger.debug(f"stdout (last 500 chars): {stdout[-500:]}")

        except subprocess.TimeoutExpired as e:
            result.errors.append(
                f"Benchmark timed out after {self.config.timeout_seconds}s"
            )
            logger.error("Benchmark timed out")

            if hasattr(e, "stdout") and e.stdout:
                stdout = e.stdout if isinstance(e.stdout, str) else e.stdout.decode()
            if hasattr(e, "stderr") and e.stderr:
                stderr = e.stderr if isinstance(e.stderr, str) else e.stderr.decode()

            self._save_logs(workspace, stdout, stderr)

        except Exception as e:
            result.errors.append(f"Benchmark execution error: {str(e)}")
            logger.exception(f"Benchmark execution failed: {e}")

        return result, stdout, stderr

    @staticmethod
    def _cleanup_server_processes(framework: str) -> None:
        """Kill leftover inference server processes after benchmark completes.

        Targets vllm/sglang server processes and their workers (TP workers,
        schedulers, etc.) that may survive the shell EXIT trap — e.g. when
        torch profiler stop hangs or NCCL cleanup stalls.
        """
        patterns = {
            "vllm": [
                "vllm.entrypoints",
                "vllm serve",
                "vllm.v1.executor",
                "vllm.v1.worker",
            ],
            "sglang": [
                "sglang.launch_server",
                "sglang.srt.managers",
                "sglang.srt.server",
            ],
        }

        targets = patterns.get(framework, [])
        if not targets:
            return

        killed = 0
        for proc_dir in Path("/proc").iterdir():
            if not proc_dir.name.isdigit():
                continue
            pid = int(proc_dir.name)
            if pid == os.getpid():
                continue
            try:
                cmdline = (proc_dir / "cmdline").read_bytes().decode(
                    errors="replace"
                ).replace("\x00", " ")
                if any(pat in cmdline for pat in targets):
                    os.kill(pid, signal.SIGKILL)
                    killed += 1
            except (ProcessLookupError, PermissionError, FileNotFoundError, OSError):
                continue

        if killed:
            logger.info(
                "cleanup: killed %d leftover %s process(es)", killed, framework
            )
            time.sleep(1)

    def _find_script_in_benchmarks(self, benchmarks_dir: Path, script_name: str) -> Optional[Path]:
        """
        Search for a script by filename inside benchmarks/ and its subdirectories.
        
        Checks the top-level directory first, then searches subdirectories.
        Returns the first match or None.
        """
        # Top-level first
        top_level = benchmarks_dir / script_name
        if top_level.exists():
            return top_level
        
        # Recursive search in subdirectories
        for match in benchmarks_dir.rglob(script_name):
            if match.is_file():
                return match
        
        return None
    
    def _get_benchmark_script(self, runner_type: str) -> str:
        """
        Get InferenceX benchmark script path with 3-tier priority.
        
        Searches benchmarks/ and its subdirectories (e.g. single_node/, multi_node/).
        
        Priority:
            1. User-specified script (benchmark_script config) - must exist
            2. InferenceX native scripts: {prefix}_{precision}_{runner}.sh
               - sglang -> dsr1_
               - vllm -> gptoss_
            3. Magpie generic scripts: {framework}_{runner}.sh
        
        Args:
            runner_type: Runner type (e.g., "mi300x", "h100", "b200")
        
        Returns:
            Relative path to benchmark script (from InferenceX root)
            
        Raises:
            FileNotFoundError: If no suitable script found
        """
        benchmarks_dir = Path(self.config.inferencex_path) / "benchmarks"
        inferencex_root = Path(self.config.inferencex_path)
        
        # Priority 1: User-specified script (must exist)
        if self.config.benchmark_script:
            found = self._find_script_in_benchmarks(benchmarks_dir, self.config.benchmark_script)
            if not found:
                raise FileNotFoundError(
                    f"Specified benchmark_script not found: {self.config.benchmark_script}\n"
                    f"Searched in: {benchmarks_dir} (including subdirectories)\n"
                    f"Please ensure the file exists or remove the benchmark_script config."
                )
            rel_path = found.relative_to(inferencex_root)
            logger.info(f"Using user-specified script: {rel_path}")
            return str(rel_path)
        
        # Priority 2: InferenceX native scripts
        # Mapping: sglang -> dsr1_, vllm -> gptoss_
        prefix_map = {"sglang": "dsr1", "vllm": "gptoss"}
        prefix = prefix_map.get(self.config.framework)
        
        if prefix:
            native_script = f"{prefix}_{self.config.precision}_{runner_type}.sh"
            found = self._find_script_in_benchmarks(benchmarks_dir, native_script)
            if found:
                rel_path = found.relative_to(inferencex_root)
                logger.info(f"Using InferenceX native script: {rel_path}")
                return str(rel_path)
        
        # Priority 3: Magpie generic scripts (top-level only)
        generic_script = f"{self.config.framework}_{runner_type}.sh"
        if (benchmarks_dir / generic_script).exists():
            logger.info(f"Using Magpie generic script: {generic_script}")
            return f"benchmarks/{generic_script}"
        
        # No script found - raise error with helpful message
        raise FileNotFoundError(
            f"No benchmark script found for framework={self.config.framework}, "
            f"precision={self.config.precision}, gpu={runner_type}.\n"
            f"Searched in: {benchmarks_dir} (including subdirectories)\n"
            f"Expected one of:\n"
            f"  - {prefix}_{self.config.precision}_{runner_type}.sh (InferenceX native)\n"
            f"  - {generic_script} (Magpie generic)\n"
            f"Please create the script or specify benchmark_script in config."
        )
    
    def _execute_benchmark(self, cmd: List[str], workspace: Path) -> tuple:
        """
        Execute Docker benchmark command.
        
        Args:
            cmd: Docker command
            workspace: Workspace directory for saving logs
        
        Returns:
            Tuple of (BenchmarkResult, stdout, stderr)
        """
        result = BenchmarkResult()
        stdout = ""
        stderr = ""
        
        try:
            # Run Docker command
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
            
            stdout = process.stdout or ""
            stderr = process.stderr or ""

            # Fix ownership: files written by root inside the container
            # (server.log, inferencex_result.json, torch_trace/*, ...) appear
            # as UID 0 on the host. chown them back to the invoking user so
            # non-root users can read/modify/delete the output.
            self._fix_workspace_ownership(workspace)

            # Save logs to workspace for debugging
            self._save_logs(workspace, stdout, stderr)
            
            if process.returncode == 0:
                result.success = True
                logger.info("Benchmark completed successfully")
            else:
                result.success = False
                result.errors.append(f"Docker command failed with code {process.returncode}")
                if stderr:
                    result.errors.append(f"stderr: {stderr[:1000]}")
                logger.error(f"Benchmark failed: {process.returncode}")
                logger.error(f"stderr: {stderr[:500]}")
            
            # Log stdout for debugging
            if stdout:
                logger.debug(f"stdout (last 500 chars): {stdout[-500:]}")
                
        except subprocess.TimeoutExpired as e:
            result.errors.append(f"Benchmark timed out after {self.config.timeout_seconds}s")
            logger.error("Benchmark timed out")
            
            # Capture partial output if available
            if hasattr(e, 'stdout') and e.stdout:
                stdout = e.stdout if isinstance(e.stdout, str) else e.stdout.decode()
            if hasattr(e, 'stderr') and e.stderr:
                stderr = e.stderr if isinstance(e.stderr, str) else e.stderr.decode()
            
            self._save_logs(workspace, stdout, stderr)
            
            # Try to stop the container
            try:
                subprocess.run(
                    ["docker", "stop", f"magpie-benchmark-{self._task_id}"],
                    capture_output=True,
                    timeout=30,
                )
            except Exception:
                pass
                
        except Exception as e:
            result.errors.append(f"Benchmark execution error: {str(e)}")
            logger.exception(f"Benchmark execution failed: {e}")
        
        return result, stdout, stderr
    
    def _fix_workspace_ownership(self, workspace: Path) -> None:
        """chown workspace files back to the invoking user after docker run.

        The benchmark container writes as root, so /workspace outputs land
        as root-owned on the host. We can't chown them from this process
        (non-root can't change another user's files), so we spawn a short-
        lived helper container that runs as root, mounts the same workspace,
        and chowns to the invoking UID/GID. Best-effort.
        """
        try:
            uid = os.getuid()
            gid = os.getgid()
        except AttributeError:
            return  # non-POSIX

        if uid == 0:
            return  # nothing to fix

        image = self.config.docker_image or "busybox"
        try:
            subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-v", f"{workspace}:/workspace",
                    "--entrypoint", "chown",
                    image, "-R", f"{uid}:{gid}", "/workspace",
                ],
                capture_output=True,
                timeout=60,
                check=False,
            )
        except Exception as e:
            logger.warning(f"Failed to fix workspace ownership: {e}")

    def _save_logs(self, workspace: Path, stdout: str, stderr: str) -> None:
        """Save benchmark subprocess stdout/stderr to workspace for debugging."""
        try:
            if stdout:
                out_file = workspace / "benchmark_stdout.log"
                with open(out_file, 'w') as f:
                    f.write(stdout)
                logger.debug(f"Saved stdout to {out_file}")
            
            if stderr:
                err_file = workspace / "benchmark_stderr.log"
                with open(err_file, 'w') as f:
                    f.write(stderr)
                logger.debug(f"Saved stderr to {err_file}")
        except Exception as e:
            logger.warning(f"Failed to save logs: {e}")
    
    def _run_tracelens_analysis(
        self,
        torch_trace_dir: Path,
        workspace: Path,
    ) -> Dict[str, Any]:
        """
        Run TraceLens analysis on torch profiler traces.
        
        NOTE: This runs on the HOST machine (not in container) after the
        Docker benchmark completes. TraceLens will be auto-installed from
        https://github.com/AMD-AIG-AIMA/TraceLens.git if not present.
        
        Args:
            torch_trace_dir: Directory containing torch trace files
            workspace: Workspace directory for output
        
        Returns:
            TraceLens analysis results dictionary
        """
        logger.info("Running TraceLens analysis on host...")
        
        try:
            analyzer = TraceLensAnalyzer(self.config.profiler.tracelens)
            
            # Get number of ranks from TP config
            num_ranks = int(self.config.envs.get("TP", 8))
            
            results = analyzer.analyze(
                trace_dir=torch_trace_dir,
                output_dir=workspace,
                num_ranks=num_ranks,
            )
            
            if results.get("output_files"):
                logger.info(
                    f"TraceLens analysis complete: {len(results['output_files'])} output files"
                )
            
            if results.get("errors"):
                for error in results["errors"]:
                    logger.warning(f"TraceLens warning: {error}")
            
            return results
            
        except Exception as e:
            logger.exception(f"TraceLens analysis failed: {e}")
            return {"enabled": True, "error": str(e)}
    
    def _run_gap_analysis(
        self,
        torch_trace_dir: Path,
        workspace: Path,
    ) -> Dict[str, Any]:
        """
        Run gap analysis on torch profiler traces.
        
        Analyzes a time window of the trace to identify kernel-level
        bottlenecks and writes a CSV report.
        
        Args:
            torch_trace_dir: Directory containing torch trace files
            workspace: Workspace directory for output
        
        Returns:
            Gap analysis results dictionary
        """
        from .gap_analysis import GapAnalyzer

        logger.info("Running gap analysis on torch traces...")
        
        try:
            gap_dir = workspace / "gap_analysis"
            gap_dir.mkdir(parents=True, exist_ok=True)

            analyzer = GapAnalyzer(self.config.gap_analysis)
            result = analyzer.analyze(torch_trace_dir)
            
            # Write merged CSV (with optional kernel source enrichment)
            csv_path = gap_dir / "gap_analysis.csv"
            result.to_csv(
                csv_path,
                find_kernel_sources=self.config.gap_analysis.find_kernel_sources,
                kernel_source_repos=self.config.gap_analysis.kernel_source_repos,
                auto_clone_repos=self.config.gap_analysis.auto_clone_repos,
                repos_base_dir=self.config.gap_analysis.repos_base_dir,
            )
            logger.info(f"Wrote gap analysis CSV: {csv_path}")
            
            # Write per-rank CSVs if multiple ranks
            if len(result.rank_results) > 1:
                rank_paths = result.to_rank_csv(gap_dir)
                for rp in rank_paths:
                    logger.info(f"Wrote per-rank CSV: {rp}")
            
            ga_dict = result.to_dict()
            ga_dict["csv_path"] = str(csv_path)
            ga_dict["output_dir"] = str(gap_dir)
            
            if result.errors:
                for err in result.errors:
                    logger.warning(f"Gap analysis warning: {err}")
            else:
                n = len(result.merged_kernels)
                logger.info(f"Gap analysis complete: {n} kernels in CSV")
            
            return ga_dict
            
        except Exception as e:
            logger.exception(f"Gap analysis failed: {e}")
            return {"enabled": True, "error": str(e)}
    
    def _build_ray_benchmark_task(self) -> Tuple[Optional["Task"], Optional[str]]:
        """Build the ``Task`` for a Ray benchmark; sets ``self._task_id`` if unset."""
        from ...core.task import Task, ModeType, ModeConfig

        if self.config.ray_config is None:
            return None, "ray_config is required when run_mode='ray'"

        t_id = self._task_id or f"bench_{uuid.uuid4().hex[:12]}"
        self._task_id = t_id

        mode_config = ModeConfig(
            mode_type=ModeType.BENCHMARK,
            gpu_arch=self.config.gpu_arch,
            timeout_seconds=self.config.timeout_seconds,
            benchmark_config=self.config.to_dict(),
        )
        task = Task(
            kernel_configs=[],
            mode_config=mode_config,
            task_id=t_id,
        )
        return task, None

    def submit_ray_benchmark(self, executor: "RayJobExecutor") -> BenchmarkResult:
        """
        Submit a Ray benchmark without waiting for completion.

        The remote task runs asynchronously; poll with MCP ``ray_task_*``
        tools using the returned ``task_id``.

        Args:
            executor: A **started** ``RayJobExecutor`` (e.g. MCP singleton).

        Returns:
            ``BenchmarkResult`` with ``metadata`` containing ``task_id``,
            ``ray_job_id``, and ``submitted: True`` on success.
        """
        result = BenchmarkResult()
        result.framework = self.config.framework
        result.model = self.config.model

        if not self.config.is_ray:
            result.success = False
            result.errors.append("submit_ray_benchmark requires run_mode='ray'")
            return result

        task, err = self._build_ray_benchmark_task()
        if err:
            result.success = False
            result.errors.append(err)
            return result

        try:
            assert task is not None
            tid = executor.submit(task)
            logger.info(f"Submitted Ray benchmark task {tid} (async)")
            result.success = True
            result.metadata = {
                "task_id": tid,
                "ray_job_id": tid,
                "submitted": True,
            }
            return result
        except Exception as e:
            result.success = False
            result.errors.append(f"Ray submit failed: {e}")
            logger.exception(f"submit_ray_benchmark failed: {e}")
            return result

    def _execute_ray_benchmark(self) -> BenchmarkResult:
        """
        Submit benchmark to a remote Ray cluster.

        Dispatches the benchmark to a Ray GPU worker via ``ray.remote()``
        and blocks until the task completes.  Returns a BenchmarkResult
        with the full results from the worker.
        """
        from ...core.ray_executor import RayJobExecutor
        from ...core.executor import ExecutorConfig, ExecutorType

        result = BenchmarkResult()
        rc = self.config.ray_config
        if rc is None:
            result.success = False
            result.errors.append("ray_config is required when run_mode='ray'")
            return result

        executor = None
        executor_started = False
        try:
            executor_config = ExecutorConfig(
                executor_type=ExecutorType.RAY,
                timeout_seconds=self.config.timeout_seconds,
            )
            executor = RayJobExecutor(executor_config, ray_config=rc)

            if not executor.start():
                executor.stop()
                result.success = False
                result.errors.append(
                    f"Failed to connect to Ray cluster at {rc.cluster_address}"
                )
                return result

            executor_started = True

            task, err = self._build_ray_benchmark_task()
            if err:
                result.success = False
                result.errors.append(err)
                return result
            assert task is not None

            logger.info(f"Dispatching benchmark to Ray GPU worker (task={self._task_id})...")
            task_result = executor.execute(task)

            result.framework = self.config.framework
            result.model = self.config.model
            result.execution_time = task_result.execution_time

            if task_result.status.value == "completed" and task_result.results:
                self._populate_result_from_ray(result, task_result.results, rc)
            else:
                result.success = False
                result.errors = task_result.errors or ["Ray task failed"]
                result.metadata = {"task_id": self._task_id}

            logger.info(f"Ray benchmark task {self._task_id} finished: {task_result.status.value}")
            return result

        except Exception as e:
            result.success = False
            result.errors.append(f"Ray submission failed: {e}")
            logger.exception(f"Ray benchmark submission failed: {e}")
            return result
        finally:
            if executor_started and executor is not None:
                executor.stop()

    def _populate_result_from_ray(
        self, result: BenchmarkResult, ray_result: dict, rc
    ) -> None:
        """Fill *result* from the dict returned by the Ray worker."""
        result.success = True
        result.workspace_dir = ray_result.get("workspace_dir", "")

        tp = ray_result.get("throughput")
        if isinstance(tp, dict):
            result.throughput = ThroughputMetrics(**{
                k: tp[k] for k in ThroughputMetrics.__dataclass_fields__
                if k in tp
            })
        else:
            result.throughput = tp

        lp = ray_result.get("latency")
        if isinstance(lp, dict):
            flat: dict = {}
            for group in ("ttft", "tpot", "itl", "e2el"):
                sub = lp.get(group, {})
                if isinstance(sub, dict):
                    flat[f"{group}_mean"] = sub.get("mean_ms", 0.0)
                    flat[f"{group}_median"] = sub.get("median_ms", 0.0)
                    flat[f"{group}_p99"] = sub.get("p99_ms", 0.0)
                    flat[f"{group}_std"] = sub.get("std_ms", 0.0)
                else:
                    flat[f"{group}_mean"] = lp.get(f"{group}_mean", 0.0)
                    flat[f"{group}_median"] = lp.get(f"{group}_median", 0.0)
                    flat[f"{group}_p99"] = lp.get(f"{group}_p99", 0.0)
                    flat[f"{group}_std"] = lp.get(f"{group}_std", 0.0)
            result.latency = LatencyMetrics(**flat)
        else:
            result.latency = lp

        result.kernel_summary = ray_result.get("kernel_summary", [])
        result.top_bottlenecks = ray_result.get("top_bottlenecks", [])
        result.gap_analysis = ray_result.get("gap_analysis")
        result.tracelens_analysis = ray_result.get("tracelens_analysis")
        result.errors = ray_result.get("errors", [])
        result.metadata = {
            "task_id": self._task_id,
            "ray_cluster": rc.cluster_address,
        }

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.config.is_local:
            self._cleanup_server_processes(self.config.framework)
        elif self._task_id:
            try:
                subprocess.run(
                    ["docker", "stop", f"magpie-benchmark-{self._task_id}"],
                    capture_output=True,
                    timeout=30,
                )
            except Exception:
                pass
