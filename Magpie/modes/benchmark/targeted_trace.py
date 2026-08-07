"""Benchmark-workspace adapter for the generic TargetedKernelTrace contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from ...targeted_trace.postprocess import postprocess_trace_dir
from ...targeted_trace.torch_profiler import adapt_torch_profiler_traces
from .config import BenchmarkConfig


def run_targeted_trace_analysis(
    *,
    config: BenchmarkConfig,
    trace_files: Iterable[Path],
    workspace: Path,
    run_id: str,
    resolved_image: Optional[str] = None,
) -> Dict[str, Any]:
    """Materialize and validate selected target evidence for one diagnostic run."""

    targeted_dir = Path(workspace) / "targeted_trace"
    manifest = adapt_torch_profiler_traces(
        sorted(Path(path) for path in trace_files),
        targeted_dir,
        config=config.profiler.targeted_trace,
        run_id=run_id,
        framework=config.framework,
        framework_version=str(config.envs.get("FRAMEWORK_VERSION", "")) or None,
        image=resolved_image or config.docker_image,
        provenance={
            "model": config.model,
            "precision": config.precision,
            "gpu_arch": config.gpu_arch,
            "benchmark_script": config.benchmark_script,
            "run_kind": config.run_kind,
        },
    )
    summary_path = targeted_dir / "summary.json"
    summary = postprocess_trace_dir(targeted_dir, output_path=summary_path)
    coverage = manifest.coverage.to_dict()
    return {
        "valid": bool(summary["valid"] and coverage["written"] > 0),
        "reward_eligible": False,
        "manifest_path": str(targeted_dir / "manifest.json"),
        "summary_path": str(summary_path),
        "coverage": coverage,
        "events": summary["events"],
        "integrity_failures_by_reason": summary[
            "integrity_failures_by_reason"
        ],
        "issues": summary["issues"],
    }
