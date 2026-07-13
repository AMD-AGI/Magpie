---
myst:
    html_meta:
        "description": "Run framework-level benchmarks for vLLM, SGLang, and Atom with Magpie. Covers Docker and local run modes, TraceLens analysis, gap analysis, and GPU selection."
        "keywords": "Magpie, benchmark, vLLM, SGLang, Atom, TraceLens, gap analysis, ROCm, AMD Instinct, GPU, LLM inference"
---

# Benchmark frameworks with Magpie

Magpie's benchmark mode runs end-to-end performance tests against LLM inference frameworks—vLLM, SGLang, and Atom—and collects throughput and latency metrics in a structured JSON report. Benchmarks can run inside a Docker container, directly on the host, or on a remote Ray cluster, and optionally capture torch profiler traces for deeper analysis with TraceLens and gap analysis. 

TraceLens is an AMD tool for visualizing profiler traces; it installs automatically on first use, but can also be installed manually—see [TraceLens installation](../../reference/troubleshooting.md#benchmarking-mode) if the auto-install fails. 

Magpie uses [InferenceX](https://github.com/SemiAnalysisAI/InferenceX) as its benchmarking backend; InferenceX is a collection of benchmark scripts for LLM inference frameworks and is cloned automatically on first run. Use this mode to measure inference performance on AMD Instinct™ GPUs and identify the GPU kernels that dominate runtime.

Review these topics for more information:

- [Magpie benchmarking mode architecture](../../conceptual/benchmarking-architecture.md) — how the benchmark pipeline is designed and how the components interact
- [Magpie benchmark mode configuration](../../reference/benchmark-config.md) — full YAML schema with all available options and defaults
- [Magpie troubleshooting](../../reference/troubleshooting.md) — solutions for common benchmark errors

```{toctree}
:maxdepth: 1
:hidden:

automatic-gpu
persistent-server-reuse
profiling-options
```

## Quick start

### Before you begin

- **Magpie installed** — see [Install Magpie](../../install/install.md).
- **Docker installed and running** — benchmark mode defaults to `run_mode: docker`. Verify with `docker info`.
- **ROCm-compatible GPU with sufficient VRAM** — the example configs target AMD Instinct™ GPUs (MI300X/MI355X). DeepSeek-R1 requires 8 GPUs at fp8; smaller models need less. Magpie [selects idle GPUs automatically](automatic-gpu.md).
- **HuggingFace token** — required for gated models. Set `HF_TOKEN` in your environment before running.
- **InferenceX** — cloned automatically on first run; no manual install needed.

### Commands

The following commands cover the most common benchmark invocations. Examples use `python -m Magpie`; the `magpie` CLI entrypoint is equivalent—see [Install Magpie](../../install/install.md) for details.

```bash
# Basic vLLM benchmark (paths are under examples/benchmarks/)
python -m Magpie benchmark --benchmark-config examples/benchmarks/benchmark_vllm_dsr1.yaml

# vLLM with TraceLens analysis
python -m Magpie benchmark --benchmark-config examples/benchmarks/benchmark_vllm_tracelens.yaml

# vLLM with gap analysis (kernel bottleneck report)
python -m Magpie benchmark --benchmark-config examples/benchmarks/benchmark_vllm_kimi_k2.yaml

# Standalone gap analysis on existing traces
python -m Magpie benchmark --trace-dir results/benchmark_vllm_<timestamp>/

# SGLang benchmark
python -m Magpie benchmark --benchmark-config examples/benchmarks/benchmark_sglang_dsr1.yaml

# Ad-hoc CLI without a YAML file (framework + model; optional torch profiler)
python -m Magpie benchmark vllm --model deepseek-ai/DeepSeek-R1-0528 --torch-profiler
```

## Output structure

A successful benchmark run creates a timestamped workspace directory with the following layout.

```
results/benchmark_vllm_<timestamp>/
├── benchmark_report.json      # Main benchmark results
├── summary.txt                # Human-readable summary
├── config.yaml                # Snapshot of benchmark configuration
├── container_stdout.log       # Container stdout
├── container_stderr.log       # Container stderr
├── inferencex_result.json   # Raw InferenceX output
├── torch_trace/               # Raw torch profiler traces
│   ├── *-rank-0.*.pt.trace.json.gz
│   ├── *-rank-1.*.pt.trace.json.gz
│   └── ...
├── gap_analysis/              # Gap analysis output (if enabled)
│   ├── gap_analysis.csv       # Merged kernel stats across all ranks
│   ├── gap_analysis_rank0.csv # Per-rank kernel stats
│   ├── gap_analysis_rank1.csv
│   └── ...
├── tracelens_rank0_csvs/      # Single-rank TraceLens analysis
│   ├── gpu_timeline.csv
│   ├── ops_summary.csv
│   └── ...
└── tracelens_collective_csvs/ # Multi-rank TraceLens analysis
    └── ...
```

## Benchmark report

The primary summary file is **`benchmark_report.json`**, written to the run workspace directory. It aggregates throughput, latency, and optional `gap_analysis` and `tracelens_analysis` sections. A typical shape (abbreviated, with `...` marking elided values):

```text
{
  "success": true,
  "framework": "vllm",
  "model": "amd/Kimi-K2-Thinking-MXFP4",
  "throughput": {
    "request_throughput": 0.16,
    "output_throughput": 1.13,
    "total_token_throughput": 1192.76,
    "completed_requests": 40
  },
  "latency": {
    "ttft": { "mean_ms": 1185.44, "p99_ms": 1969.59 },
    "tpot": { "mean_ms": 131.09, "p99_ms": 282.21 }
  },
  "gap_analysis": {
    "config": { "trace_start_pct": 50, "trace_end_pct": 80, "categories": ["kernel", "gpu"] },
    "csv_path": "results/.../gap_analysis/gap_analysis.csv",
    "top_kernels": [
      { "name": "rcclGenericKernel<...>", "calls": 19620, "self_cuda_total_us": 28999961.95, "pct_total": 44.0 },
      { "name": "kernel_moe_mxgemm_2lds<...>", "calls": 9360, "self_cuda_total_us": 12495324.68, "pct_total": 18.9 }
    ]
  },
  "tracelens_analysis": { "output_files": [...] }
}
```

## Related topics

See the following pages for related concepts, configuration, and reference material.

- [Automatic GPU selection in Magpie's benchmark mode](automatic-gpu.md) — how Magpie picks idle GPUs before launching and how to override or disable selection
- [Persistent server reuse (local) in Magpie's benchmark mode](persistent-server-reuse.md) — keep a server alive across runs to avoid model reload overhead
- [Profiling options in Magpie's benchmark mode](profiling-options.md) — configure torch profiler, TraceLens, and gap analysis
- [Analyze and compare kernels with Magpie](../analyze-compare.md) — kernel evaluation modes independent of benchmark mode
- [Run Magpie on a Ray cluster](../ray.md) — optional remote benchmark scheduling