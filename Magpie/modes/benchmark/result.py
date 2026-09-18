###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Result classes for benchmark mode.

Parses and structures benchmark results from InferenceX output.
"""

import gzip
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ThroughputMetrics:
    """Throughput metrics from benchmark."""
    request_throughput: float = 0.0  # requests/second
    output_throughput: float = 0.0  # tokens/second
    total_token_throughput: float = 0.0  # tokens/second (input + output)
    completed_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_throughput": self.request_throughput,
            "output_throughput": self.output_throughput,
            "total_token_throughput": self.total_token_throughput,
            "completed_requests": self.completed_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class LatencyMetrics:
    """Latency metrics from benchmark."""
    # Time to First Token (ms)
    ttft_mean: float = 0.0
    ttft_median: float = 0.0
    ttft_p99: float = 0.0
    ttft_std: float = 0.0
    
    # Time per Output Token (ms)
    tpot_mean: float = 0.0
    tpot_median: float = 0.0
    tpot_p99: float = 0.0
    tpot_std: float = 0.0
    
    # Inter-token Latency (ms)
    itl_mean: float = 0.0
    itl_median: float = 0.0
    itl_p99: float = 0.0
    itl_std: float = 0.0
    
    # End-to-end Latency (ms)
    e2el_mean: float = 0.0
    e2el_median: float = 0.0
    e2el_p99: float = 0.0
    e2el_std: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ttft": {
                "mean_ms": self.ttft_mean,
                "median_ms": self.ttft_median,
                "p99_ms": self.ttft_p99,
                "std_ms": self.ttft_std,
            },
            "tpot": {
                "mean_ms": self.tpot_mean,
                "median_ms": self.tpot_median,
                "p99_ms": self.tpot_p99,
                "std_ms": self.tpot_std,
            },
            "itl": {
                "mean_ms": self.itl_mean,
                "median_ms": self.itl_median,
                "p99_ms": self.itl_p99,
                "std_ms": self.itl_std,
            },
            "e2el": {
                "mean_ms": self.e2el_mean,
                "median_ms": self.e2el_median,
                "p99_ms": self.e2el_p99,
                "std_ms": self.e2el_std,
            },
        }


@dataclass
class KernelMetrics:
    """Kernel-level metrics from profiling."""
    name: str = ""
    time_ms: float = 0.0
    percent: float = 0.0
    calls: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "time_ms": self.time_ms,
            "percent": self.percent,
            "calls": self.calls,
        }


@dataclass
class BenchmarkResult:
    """
    Complete benchmark result.
    
    Aggregates results from InferenceX benchmark execution including
    throughput, latency, and optional profiling data.
    """
    success: bool = False
    framework: str = ""
    model: str = ""
    
    # Metrics
    throughput: Optional[ThroughputMetrics] = None
    latency: Optional[LatencyMetrics] = None
    
    # Kernel profiling (from torch_profiler or system_profiler)
    kernel_summary: List[KernelMetrics] = field(default_factory=list)
    top_bottlenecks: List[str] = field(default_factory=list)
    
    # TraceLens analysis results
    tracelens_analysis: Optional[Dict[str, Any]] = None
    
    # Gap analysis results
    gap_analysis: Optional[Dict[str, Any]] = None
    
    # GPU hardware monitoring (temperature, frequency, power)
    gpu_monitor: Optional[Dict[str, Any]] = None
    
    # Execution info
    workspace_dir: str = ""
    execution_time: float = 0.0
    profiling_enabled: bool = False
    
    # Errors
    errors: List[str] = field(default_factory=list)
    
    # Raw data
    raw_result: Optional[Dict[str, Any]] = None

    # Workload identity and AgentX-specific normalized data. ``success`` means
    # the run and validation passed; ``publishable`` additionally requires a
    # canonical (non-fast) AgentX run.
    scenario: str = "fixed-seq-len"
    benchmark_valid: Optional[bool] = None
    publishable: Optional[bool] = None
    agentx_metrics: Optional[Dict[str, Any]] = None

    # Scriptable (server-less) extras — e.g. xDiT diffusion. ``workload_kind``
    # distinguishes serving vs scriptable runs; ``quality_gate`` carries the
    # image-quality result (LPIPS/SSIM/MSE) in place of a GSM8K eval;
    # ``throughput_unit`` is "img/s" for diffusion; ``latency_s`` is the E2E
    # per-image latency the throughput was derived from.
    workload_kind: str = ""
    throughput_unit: str = ""
    quality_gate: Optional[Dict[str, Any]] = None
    latency_s: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "success": self.success,
            "framework": self.framework,
            "model": self.model,
            "throughput": self.throughput.to_dict() if self.throughput else None,
            "latency": self.latency.to_dict() if self.latency else None,
            "kernel_summary": [k.to_dict() for k in self.kernel_summary],
            "top_bottlenecks": self.top_bottlenecks,
            "tracelens_analysis": self.tracelens_analysis,
            "gap_analysis": self.gap_analysis,
            "gpu_monitor": self.gpu_monitor,
            "workspace_dir": self.workspace_dir,
            "execution_time": self.execution_time,
            "profiling_enabled": self.profiling_enabled,
            "errors": self.errors,
        }
        # Scriptable (server-less) extras — e.g. xDiT diffusion. Only emit when
        # populated so the report schema is unchanged for vLLM/SGLang/Atom.
        if self.workload_kind:
            d["workload_kind"] = self.workload_kind
        if self.throughput_unit:
            d["throughput_unit"] = self.throughput_unit
        if self.quality_gate is not None:
            d["quality_gate"] = self.quality_gate
        if self.latency_s is not None:
            d["latency_s"] = self.latency_s
        if self.scenario != "fixed-seq-len":
            d["scenario"] = self.scenario
        if self.benchmark_valid is not None:
            d["benchmark_valid"] = self.benchmark_valid
        if self.publishable is not None:
            d["publishable"] = self.publishable
        if self.agentx_metrics is not None:
            d["agentx_metrics"] = self.agentx_metrics
        return d
    
    def get_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"{'=' * 60}",
            f"Benchmark Result: {self.framework.upper()}",
            f"{'=' * 60}",
            f"Model: {self.model}",
            f"Status: {'SUCCESS' if self.success else 'FAILED'}",
        ]

        if self.scenario == "agentx" and self.agentx_metrics:
            metrics = self.agentx_metrics
            requests = metrics.get("requests", {})
            throughput = metrics.get("throughput", {})
            lines.extend(
                [
                    "",
                    "AgentX:",
                    f"  Mode: {metrics.get('mode', 'unknown')}",
                    f"  Benchmark valid: {bool(self.benchmark_valid)}",
                    f"  Publishable: {bool(self.publishable)}",
                    "  Requests: "
                    f"{requests.get('successful', 0)}/"
                    f"{requests.get('profiled_total', requests.get('total', 0))} "
                    f"successful (error rate {requests.get('error_rate', 0):.2%})",
                    f"  Mean QPS: {throughput.get('request_throughput', 0):.2f}",
                    "  Output throughput: "
                    f"{throughput.get('output_tokens_per_second', 0):.2f} tok/s",
                    "  Total throughput: "
                    f"{throughput.get('total_tokens_per_second', 0):.2f} tok/s",
                ]
            )

        
        if self.throughput and self.scenario != "agentx":
            lines.extend([
                "",
                "Throughput:",
                f"  Request throughput: {self.throughput.request_throughput:.2f} req/s",
                f"  Output throughput: {self.throughput.output_throughput:.2f} tok/s",
                f"  Total throughput: {self.throughput.total_token_throughput:.2f} tok/s",
                f"  Completed requests: {self.throughput.completed_requests}",
                f"  Duration: {self.throughput.duration_seconds:.2f}s",
            ])
        
        if self.latency and self.scenario != "agentx":
            lines.extend([
                "",
                "Latency:",
                f"  TTFT (mean/p99): {self.latency.ttft_mean:.2f}ms / {self.latency.ttft_p99:.2f}ms",
                f"  TPOT (mean/p99): {self.latency.tpot_mean:.2f}ms / {self.latency.tpot_p99:.2f}ms",
                f"  ITL (mean/p99): {self.latency.itl_mean:.2f}ms / {self.latency.itl_p99:.2f}ms",
                f"  E2EL (mean/p99): {self.latency.e2el_mean:.2f}ms / {self.latency.e2el_p99:.2f}ms",
            ])
        
        if self.top_bottlenecks:
            lines.extend([
                "",
                "Top Bottleneck Kernels:",
            ])
            for i, kernel in enumerate(self.top_bottlenecks[:5], 1):
                lines.append(f"  {i}. {kernel}")
        
        if self.tracelens_analysis:
            lines.extend([
                "",
                "TraceLens Analysis:",
            ])
            if self.tracelens_analysis.get("output_files"):
                lines.append(f"  Output files: {len(self.tracelens_analysis['output_files'])}")
                # Show first few files
                for f in self.tracelens_analysis["output_files"][:3]:
                    lines.append(f"    - {Path(f).name}")
                if len(self.tracelens_analysis["output_files"]) > 3:
                    lines.append(f"    ... and {len(self.tracelens_analysis['output_files']) - 3} more")
            if self.tracelens_analysis.get("errors"):
                for err in self.tracelens_analysis["errors"]:
                    lines.append(f"  Warning: {err}")
        
        if self.gap_analysis:
            lines.extend(["", "Gap Analysis:"])
            ga = self.gap_analysis
            cfg = ga.get("config", {})
            start_pct = cfg.get("trace_start_pct", 0)
            end_pct = cfg.get("trace_end_pct", 100)
            lines.append(f"  Window: {start_pct}%-{end_pct}%")
            cats = cfg.get("categories")
            if cats:
                lines.append(f"  Categories: {', '.join(cats)}")
            top_kernels = ga.get("top_kernels", [])
            if top_kernels:
                for i, k in enumerate(top_kernels[:5], 1):
                    lines.append(
                        f"  {i}. {k['name']} - "
                        f"{k.get('pct_total', 0):.1f}% "
                        f"({k.get('self_cuda_total_us', 0):.2f}us, "
                        f"{k.get('calls', 0)} calls)"
                    )
                if len(top_kernels) > 5:
                    lines.append(f"  ... and {len(top_kernels) - 5} more")

        if self.gpu_monitor:
            lines.extend(["", "GPU Hardware Monitoring:"])
            gm = self.gpu_monitor
            lines.append(f"  Samples: {gm.get('sample_count', 0)} ({gm.get('duration_sec', 0):.1f}s)")
            temp = gm.get("temperature_c", {})
            if temp:
                lines.append(f"  Temperature: {temp.get('min', 0):.1f}°C - {temp.get('max', 0):.1f}°C (avg: {temp.get('avg', 0):.1f}°C)")
            gpu_clk = gm.get("gpu_clock_mhz", {})
            if gpu_clk:
                lines.append(f"  GPU Clock: {gpu_clk.get('min', 0)} - {gpu_clk.get('max', 0)} MHz (avg: {gpu_clk.get('avg', 0):.0f})")
            power = gm.get("power_watts", {})
            if power:
                lines.append(f"  Power: {power.get('min', 0):.1f} - {power.get('max', 0):.1f} W (avg: {power.get('avg', 0):.1f})")

        if self.errors:
            lines.extend([
                "",
                "Errors:",
            ])
            for err in self.errors:
                lines.append(f"  - {err}")
        
        lines.extend([
            "",
            f"Workspace: {self.workspace_dir}",
            f"Execution time: {self.execution_time:.2f}s",
            f"{'=' * 60}",
        ])
        
        return "\n".join(lines)

def _number(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _as_non_negative_int(value: Any) -> int:
    number = _number(value)
    return max(0, int(number))


def _nested_number(data: Dict[str, Any], key: str, nested_key: str) -> float:
    nested = data.get(key)
    return _number(nested.get(nested_key)) if isinstance(nested, dict) else 0.0


def _latency_stat_ms(latency: Dict[str, Any], metric: str, stat: str) -> float:
    values = latency.get(metric)
    if not isinstance(values, dict):
        return 0.0
    return _number(values.get(stat)) * 1000.0


def _agentx_latency_metrics(latency: Dict[str, Any]) -> LatencyMetrics:
    return LatencyMetrics(
        ttft_mean=_latency_stat_ms(latency, "ttft", "mean"),
        ttft_median=_latency_stat_ms(latency, "ttft", "p50"),
        ttft_std=_latency_stat_ms(latency, "ttft", "std"),
        tpot_mean=_latency_stat_ms(latency, "tpot", "mean"),
        tpot_median=_latency_stat_ms(latency, "tpot", "p50"),
        tpot_std=_latency_stat_ms(latency, "tpot", "std"),
        itl_mean=_latency_stat_ms(latency, "itl", "mean"),
        itl_median=_latency_stat_ms(latency, "itl", "p50"),
        itl_std=_latency_stat_ms(latency, "itl", "std"),
        e2el_mean=_latency_stat_ms(latency, "e2el", "mean"),
        e2el_median=_latency_stat_ms(latency, "e2el", "p50"),
        e2el_std=_latency_stat_ms(latency, "e2el", "std"),
    )



class ResultParser:
    """
    Parses InferenceX benchmark output into structured results.
    """
    
    @staticmethod
    def parse_inferencex_result(
        result_file: Path,
        framework: str = "",
        model: str = "",
        is_scriptable: bool = False,
        scenario: str = "fixed-seq-len",
        agentx_mode: str = "canonical",
        failed_request_threshold: float = 0.10,
    ) -> BenchmarkResult:
        """
        Parse InferenceX result JSON file.
        
        Args:
            result_file: Path to inferencex_result.json
            framework: Framework name
            model: Model name
            is_scriptable: Whether this is a server-less (scriptable) workload
                (e.g. xDiT diffusion). For scriptable runs the image-quality
                gate is the only correctness signal, so a missing/un-passed
                gate fails the benchmark instead of silently passing.
        
        Returns:
            Parsed BenchmarkResult
        """
        result = BenchmarkResult(
            framework=framework,
            model=model,
            scenario=scenario,
        )
        
        if not result_file.exists():
            result.errors.append(f"Result file not found: {result_file}")
            return result
        
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)

            if scenario == "agentx":
                return ResultParser._parse_agentx_data(
                    data,
                    framework=framework,
                    model=model,
                    mode=agentx_mode,
                    failed_request_threshold=failed_request_threshold,
                )
            
            result.raw_result = data
            result.success = True
            
            # Parse throughput metrics
            result.throughput = ThroughputMetrics(
                request_throughput=data.get("request_throughput", 0.0),
                output_throughput=data.get("output_throughput", 0.0),
                total_token_throughput=data.get("total_token_throughput", 0.0),
                completed_requests=data.get("completed", 0),
                total_input_tokens=data.get("total_input_tokens", 0),
                total_output_tokens=data.get("total_output_tokens", 0),
                duration_seconds=data.get("duration", 0.0),
            )
            
            # Parse latency metrics
            result.latency = LatencyMetrics(
                ttft_mean=data.get("mean_ttft_ms", 0.0),
                ttft_median=data.get("median_ttft_ms", 0.0),
                ttft_p99=data.get("p99_ttft_ms", 0.0),
                ttft_std=data.get("std_ttft_ms", 0.0),
                tpot_mean=data.get("mean_tpot_ms", 0.0),
                tpot_median=data.get("median_tpot_ms", 0.0),
                tpot_p99=data.get("p99_tpot_ms", 0.0),
                tpot_std=data.get("std_tpot_ms", 0.0),
                itl_mean=data.get("mean_itl_ms", 0.0),
                itl_median=data.get("median_itl_ms", 0.0),
                itl_p99=data.get("p99_itl_ms", 0.0),
                itl_std=data.get("std_itl_ms", 0.0),
                e2el_mean=data.get("mean_e2el_ms", 0.0),
                e2el_median=data.get("median_e2el_ms", 0.0),
                e2el_p99=data.get("p99_e2el_ms", 0.0),
                e2el_std=data.get("std_e2el_ms", 0.0),
            )
            
            # Scriptable (server-less) extras — carried verbatim from the bench
            # script's result JSON (e.g. xDiT diffusion image-quality gate).
            result.workload_kind = data.get("workload_kind", "")
            result.throughput_unit = data.get("throughput_unit", "")
            result.quality_gate = data.get("quality_gate")
            result.latency_s = data.get("latency_s")

            # Enforce the scriptable image-quality gate so tuning/leaderboard
            # runs never accept a faster config that regressed image quality.
            #
            # Serving (vLLM/SGLang/Atom): only a populated gate with
            # passed=False fails (a missing gate is legitimate — no eval ran).
            #
            # Scriptable (xDiT diffusion): the gate is the ONLY correctness
            # signal, so it is required — a missing/non-dict gate, or one whose
            # ``passed`` is not exactly True, fails the benchmark (fail-closed)
            # rather than silently passing.
            if is_scriptable:
                gate = result.quality_gate
                if not isinstance(gate, dict):
                    result.success = False
                    result.errors.append(
                        "Quality gate missing for scriptable workload: "
                        "the image-quality gate is required but was not "
                        f"reported (got {gate!r})"
                    )
                elif gate.get("passed") is not True:
                    result.success = False
                    result.errors.append(
                        f"Quality gate not passed: {gate}"
                    )
            elif isinstance(result.quality_gate, dict) and \
                    result.quality_gate.get("passed") is False:
                result.success = False
                result.errors.append(
                    f"Quality gate failed: {result.quality_gate}"
                )

            # Extract model info if not provided
            if not result.model and "model_id" in data:
                result.model = data["model_id"]
            
            logger.info(f"Parsed benchmark result: {result.throughput.request_throughput:.2f} req/s")
            
        except json.JSONDecodeError as e:
            result.errors.append(f"Failed to parse JSON: {e}")
        except Exception as e:
            result.errors.append(f"Failed to parse result: {e}")
        
        return result
    
    @staticmethod
    def _parse_agentx_data(
        data: Dict[str, Any],
        *,
        framework: str,
        model: str,
        mode: str,
        failed_request_threshold: float,
    ) -> BenchmarkResult:
        """Normalize an InferenceX AgentX aggregate into Magpie metrics."""

        result = BenchmarkResult(
            framework=framework,
            model=model,
            scenario="agentx",
            raw_result=data,
        )
        if data.get("scenario_type") != "agentic-coding":
            result.benchmark_valid = False
            result.publishable = False
            result.errors.append(
                "AgentX result must contain scenario_type='agentic-coding'"
            )
            return result

        request_metrics = data.get("request_metrics")
        accounting = data.get("request_accounting")
        if not isinstance(request_metrics, dict) or not isinstance(accounting, dict):
            result.benchmark_valid = False
            result.publishable = False
            result.errors.append(
                "AgentX result is missing request_metrics or request_accounting"
            )
            return result

        successful = _as_non_negative_int(data.get("num_requests_successful"))
        total = _as_non_negative_int(
            data.get("num_requests_total", accounting.get("records_total"))
        )
        errors = _as_non_negative_int(accounting.get("records_error_dropped"))
        completed = successful + errors
        error_rate = errors / completed if completed else 1.0

        throughput_data = request_metrics.get("throughput")
        qps_data = request_metrics.get("qps")
        latency_data = request_metrics.get("latency")
        throughput_data = throughput_data if isinstance(throughput_data, dict) else {}
        qps_data = qps_data if isinstance(qps_data, dict) else {}
        latency_data = latency_data if isinstance(latency_data, dict) else {}

        input_tps = _nested_number(throughput_data, "input", "tokens_per_second")
        output_tps = _nested_number(throughput_data, "output", "tokens_per_second")
        total_tps = _nested_number(throughput_data, "total", "tokens_per_second")
        duration = _number(throughput_data.get("duration_seconds"))
        request_tput = _number(qps_data.get("mean"))
        recipe_fingerprint = data.get("recipe_fingerprint")
        fingerprint_valid = (
            isinstance(recipe_fingerprint, str)
            and len(recipe_fingerprint) == 64
            and all(
                character in "0123456789abcdef"
                for character in recipe_fingerprint
            )
        )

        result.throughput = ThroughputMetrics(
            request_throughput=request_tput,
            output_throughput=output_tps,
            total_token_throughput=total_tps,
            completed_requests=successful,
            duration_seconds=duration,
        )
        result.latency = _agentx_latency_metrics(latency_data)

        valid = successful > 0 and error_rate <= failed_request_threshold
        if successful <= 0:
            result.errors.append("AgentX completed zero successful requests")
        if error_rate > failed_request_threshold:
            result.errors.append(
                "AgentX request error rate exceeded the configured limit: "
                f"{error_rate:.3%} > {failed_request_threshold:.3%}"
            )
        if duration <= 0 or total_tps <= 0:
            valid = False
            result.errors.append(
                "AgentX result is missing a positive duration or total throughput"
            )

        result.success = valid
        result.benchmark_valid = valid
        result.publishable = valid and mode == "canonical" and fingerprint_valid
        result.agentx_metrics = {
            "mode": mode,
            "scenario_type": "agentic-coding",
            "recipe_fingerprint_valid": fingerprint_valid,
            "requests": {
                "total": total,
                "records_total": total,
                "profiled_total": completed,
                "successful": successful,
                "errors": errors,
                "warmup_dropped": _as_non_negative_int(
                    accounting.get("records_warmup_dropped")
                ),
                "error_rate": error_rate,
                "threshold": failed_request_threshold,
            },
            "throughput": {
                "request_throughput": request_tput,
                "input_tokens_per_second": input_tps,
                "output_tokens_per_second": output_tps,
                "total_tokens_per_second": total_tps,
                "duration_seconds": duration,
            },
            "latency_seconds": latency_data,
            "dataset": data.get("dataset"),
            "server_metrics": data.get("server_metrics"),
            "request_accounting": accounting,
            "recipe": {
                key: data.get(key)
                for key in (
                    "hw",
                    "conc",
                    "image",
                    "recipe_fingerprint",
                    "model",
                    "infmax_model_prefix",
                    "framework",
                    "precision",
                    "spec_decoding",
                    "tp",
                    "pp",
                    "ep",
                    "dp_attention",
                    "kv_offloading",
                    "kv_offload_backend",
                    "allocated_cpu_dram_gb",
                    "router",
                    "kv_p2p_transfer",
                )
                if key in data
            },
        }
        return result

    @staticmethod
    def parse_torch_trace(trace_dir: Path) -> List[KernelMetrics]:
        """
        Parse PyTorch profiler trace for kernel metrics.

        Searches ``trace_dir`` recursively, ignores CUDA-graph warmup snapshots
        (e.g. vLLM's ``capture_traces/``), and parses the largest real trace so
        the summary reflects the actual benchmark workload (see issue #38).

        Args:
            trace_dir: Directory containing torch trace files

        Returns:
            List of kernel metrics, ordered by descending GPU time
        """
        kernels = []

        if not trace_dir.exists():
            return kernels

        # Recursive: atom writes per-rank traces under rank_<N>/ subdirs
        trace_files = sorted(trace_dir.rglob("*.json.gz")) + sorted(
            trace_dir.rglob("*.json")
        )

        # Skip CUDA-graph warmup snapshots (e.g. vLLM writes a small warmup
        # trace under capture_traces/). Parsing those instead of the real
        # benchmark trace yields a misleading kernel summary that misses the
        # dominant kernels. See issue #38.
        candidate_files = [
            f
            for f in trace_files
            if "capture_traces" not in (part.lower() for part in f.parts)
        ]
        if not candidate_files:
            candidate_files = trace_files

        # The real per-rank benchmark trace is by far the largest file; the
        # warmup/auxiliary snapshots are tiny by comparison. Process candidates
        # largest-first and use the first one that parses successfully, so a
        # truncated/corrupt largest trace falls back to the next-best file.
        def _file_size(path: Path) -> int:
            try:
                return path.stat().st_size
            except OSError:
                return 0

        candidate_files = sorted(
            candidate_files, key=lambda f: (-_file_size(f), f.name)
        )

        for trace_file in candidate_files:
            try:
                if trace_file.suffix == ".gz":
                    with gzip.open(trace_file, "rt") as f:
                        trace_data = json.load(f)
                else:
                    with open(trace_file, "r") as f:
                        trace_data = json.load(f)

                events = trace_data.get("traceEvents", [])
                kernel_times: Dict[str, float] = {}
                kernel_counts: Dict[str, int] = {}

                for event in events:
                    if event.get("cat") == "kernel":
                        name = event.get("name", "unknown")
                        dur = event.get("dur", 0) / 1000.0
                        if name in kernel_times:
                            kernel_times[name] += dur
                            kernel_counts[name] += 1
                        else:
                            kernel_times[name] = dur
                            kernel_counts[name] = 1

                total_time = sum(kernel_times.values())
                for name, time_ms in sorted(kernel_times.items(), key=lambda x: -x[1]):
                    percent = (time_ms / total_time * 100) if total_time > 0 else 0
                    kernels.append(
                        KernelMetrics(
                            name=name,
                            time_ms=time_ms,
                            percent=percent,
                            calls=kernel_counts.get(name, 0),
                        )
                    )

                # Use the first trace that actually contains kernel events;
                # otherwise fall through to the next-largest candidate.
                if kernels:
                    break

            except Exception as e:
                logger.warning(f"Failed to parse trace file {trace_file}: {e}")

        return kernels
