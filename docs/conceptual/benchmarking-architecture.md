---
myst:
    html_meta:
        "description": "Learn how Magpie's benchmark mode pipeline is structured, including components, execution flow, and integration with TraceLens and gap analysis."
        "keywords": "Magpie, benchmark architecture, BenchmarkMode, TraceLens, gap analysis, vLLM, SGLang, ROCm, GPU, LLM inference"
---

# Magpie benchmarking mode architecture

Magpie's benchmark mode drives end-to-end performance evaluation of LLM inference frameworks—vLLM, SGLang, and Atom—by launching a server, running a client workload, and collecting throughput and latency metrics into a structured JSON report. Benchmarks can run inside a Docker container (the default), directly on the host, or on a remote Ray cluster, and they optionally capture torch profiler traces for downstream analysis with TraceLens and gap analysis. This page describes the components that make up the benchmark pipeline, the execution flow from configuration to report generation, and how the pieces connect.

## Architecture

Magpie benchmark mode is composed of the following key components that work together to run, profile, and analyze inference framework benchmarks.

### Components

Benchmark mode consists of the following Python modules.

| Component | File | Description |
|-----------|------|-------------|
| `BenchmarkMode` | `benchmarker.py` | Main orchestrator |
| `BenchmarkConfig` | `config.py` | Configuration dataclasses |
| `TraceLensAnalyzer` | `tracelens.py` | TraceLens CLI integration |
| `TraceLensInferencePipeline` | `tracelens_inference.py` | Inference-aware TraceLens split/report flow and simple roofline summaries |
| `GapAnalyzer` | `gap_analysis.py` | Kernel bottleneck analysis |
| `BenchmarkResult` | `result.py` | Result data structures |

### Execution flow

Each benchmark run proceeds through the following stages.

1. **Configuration Loading**: Parse YAML config into `BenchmarkConfig`
2. **Runtime Setup**: For `run_mode: docker`, prepare a container with InferenceX; for `local`, use the host environment
3. **Server Launch**: Start vLLM/SGLang server (in container or on host per `run_mode`)
4. **Client Execution**: Run benchmark client with profiling enabled
5. **Trace Collection**: Torch profiler traces saved to workspace
6. **TraceLens Analysis**: Split inference traces, run TraceLens CLI commands
   inside the runtime image for Docker inference mode or on host for
   local/classic mode, and write per-stage simple roofline summaries (if enabled)
7. **Gap Analysis**: Analyze kernel bottlenecks within time window (if enabled)
8. **Result Generation**: Aggregate metrics and generate reports

### Architecture diagram

The following diagram shows how Magpie orchestrates the benchmark pipeline.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Benchmark Mode                               │
├─────────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐    ┌───────────────┐    ┌────────────────────┐   │
│  │BenchmarkConfig│  → │ BenchmarkMode │ →  │  BenchmarkResult   │   │
│  │  (YAML)       │    │               │    │  (JSON + CSV)      │   │
│  └───────────────┘    └───────────────┘    └────────────────────┘   │
│                               │                                     │
│                               ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Runtime: docker │ local │ ray                               │   │
│  │  ┌─────────────┐        ┌─────────────────────────────────┐  │   │
│  │  │ InferenceX  │  →     │ vLLM / SGLang Server + Client   │  │   │
│  │  │ scripts     │        │ + Torch Profiler                │  │   │
│  │  └─────────────┘        └─────────────────────────────────┘  │   │
│  │  Ray: Magpie driver → RayJobExecutor → GPU worker runs the   │   │
│  │        same stack (local/docker on worker; NFS for cache/    │   │
│  │        results). See ray.md                                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                               │                                     │
│                      ┌────────┴────────┐                            │
│                      ▼                 ▼                            │
│  ┌────────────────────────┐  ┌─────────────────────────────────┐    │
│  │  Gap Analysis          │  │  TraceLens Analysis             │    │
│  │  • Time window filter  │  │  • Inference stage reports      │    │
│  │  • Category filter     │  │  • Simple roofline summaries    │    │
│  │  • Kernel stats CSV    │  │  • Legacy per-rank reports      │    │
│  └────────────────────────┘  └─────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

## Related topics

- [Benchmark frameworks with Magpie](../how-to/benchmarking/benchmark.md): how-to guide covering configuration, run modes, TraceLens, gap analysis, and examples
- [Magpie benchmark mode configuration](../reference/benchmark-config.md): full YAML schema with all available options and defaults
- [Run Magpie on a Ray cluster](../how-to/ray.md): running benchmarks on remote GPU nodes using Ray
- [Find kernel sources with Magpie](../how-to/kernel-source-finder.md): mapping kernel names from gap analysis output to source files
- [Magpie troubleshooting](../reference/troubleshooting.md): solutions for common benchmark errors
