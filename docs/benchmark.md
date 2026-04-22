# Benchmark Mode

Benchmark mode enables framework-level performance benchmarking for LLM inference engines (vLLM, SGLang) with integrated trace analysis capabilities.

**Execution:** Benchmarks use `run_mode`: **`docker`** (default), **`local`** (host / in-pod, via YAML or `--run-mode local`), or **`ray`** (driver submits `RayJobExecutor`; a **GPU worker** runs the same InferenceX → vLLM/SGLang flow—see [Magpie + Ray](ray-magpie.md)). InferenceX is cloned automatically when `inferencex_path` is empty (see `Magpie/config.yaml` `benchmark.inferencex_path`).

## Overview

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
│  │  Runtime: docker │ local │ ray                                │   │
│  │  ┌─────────────┐        ┌─────────────────────────────────┐  │   │
│  │  │ InferenceX  │  →    │ vLLM / SGLang Server + Client   │  │   │
│  │  │ scripts     │        │ + Torch Profiler                │  │   │
│  │  └─────────────┘        └─────────────────────────────────┘  │   │
│  │  Ray: Magpie driver → RayJobExecutor → GPU worker runs the   │   │
│  │        same stack (local/docker on worker; NFS for cache/     │   │
│  │        results). See docs/ray-magpie.md                       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                               │                                     │
│                      ┌────────┴────────┐                            │
│                      ▼                 ▼                            │
│  ┌────────────────────────┐  ┌─────────────────────────────────┐   │
│  │  Gap Analysis          │  │  TraceLens Analysis             │   │
│  │  • Time window filter  │  │  • Perf report (per-rank)       │   │
│  │  • Category filter     │  │  • Multi-rank collective report │   │
│  │  • Kernel stats CSV    │  │                                 │   │
│  └────────────────────────┘  └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Basic vLLM benchmark (paths are under examples/benchmarks/)
python -m Magpie benchmark --benchmark-config examples/benchmarks/benchmark_vllm_dsr1.yaml

# vLLM with TraceLens analysis
python -m Magpie benchmark --benchmark-config examples/benchmarks/benchmark_vllm_tracelens.yaml

# vLLM with gap analysis (kernel bottleneck report)
python -m Magpie benchmark --benchmark-config examples/benchmarks/benchmark_vllm_kimi_k2.yaml

# Standalone gap analysis on existing traces
python -m Magpie benchmark gap-analysis --trace-dir results/benchmark_vllm_<timestamp>/

# SGLang benchmark
python -m Magpie benchmark --benchmark-config examples/benchmarks/benchmark_sglang_dsr1.yaml

# Ad-hoc CLI without a YAML file (framework + model; optional torch profiler)
python -m Magpie benchmark vllm --model deepseek-ai/DeepSeek-R1-0528 --torch-profiler
```

## Configuration

### Minimal Example

```yaml
benchmark:
  framework: vllm              # "vllm" or "sglang"
  model: deepseek-ai/DeepSeek-R1-0528
  precision: fp8               # "fp8", "fp16", "bf16"
  
  envs:
    TP: 8                      # Tensor parallelism
    CONC: 32                   # Concurrency (num_prompts = CONC * 10)
    ISL: 1024                  # Input sequence length
    OSL: 1024                  # Output sequence length
    
  profiler:
    torch_profiler:
      enabled: true            # Generate torch profiling traces
      
  timeout_seconds: 3600
```

### Full Configuration Reference

```yaml
benchmark:
  # Framework selection
  framework: vllm              # Required: "vllm" or "sglang"
  model: <model_name>          # Required: HuggingFace model name/path
  precision: fp8               # Optional: "fp8" (default), "fp16", "bf16"
  
  # Benchmark parameters
  envs:
    TP: 8                      # Tensor parallelism (GPU count)
    CONC: 32                   # Request concurrency
    ISL: 1024                  # Input sequence length
    OSL: 1024                  # Output sequence length
    RANDOM_RANGE_RATIO: 1      # Length randomization (0-1)
    MAX_MODEL_LEN: 131072      # Max model context length
    GPU_MEM_UTIL: 0.95         # GPU memory utilization (0-1)
    ENABLE_PROFILE: "true"     # Enable profiling in benchmark script
    
  # Profiler configuration
  profiler:
    # PyTorch profiler (generates JSON traces)
    torch_profiler:
      enabled: true            # Sets VLLM_TORCH_PROFILER_DIR
      
    # System profiler (rocprof-compute / ncu)
    system_profiler:
      enabled: false
      profile_args: []         # Additional profiler arguments
      
    # TraceLens trace analysis
    tracelens:
      enabled: true            # Enable TraceLens analysis
      export_format: csv       # "csv" or "excel"
      perf_report_enabled: true      # Single-rank performance report
      multi_rank_report_enabled: true # Multi-rank collective report
      gpu_arch_config: null    # Optional: GPU arch config for roofline

  # Gap analysis (kernel bottleneck report)
  gap_analysis:
    enabled: true              # Enable gap analysis after benchmark
    trace_start_pct: 50        # Start of analysis window (0-100)
    trace_end_pct: 80          # End of analysis window (0-100)
    top_k: 20                  # Number of top kernels in report
    min_duration_us: 0.0       # Filter out events shorter than this (us)
    categories:                # Event category whitelist (default: [kernel, gpu])
      - kernel
      - gpu
    ignore_categories:         # Event category blacklist (default: [gpu_user_annotation])
      - gpu_user_annotation
      
  # Auto-pick idle GPU(s) before launching (enabled by default).
  # See "Automatic GPU Selection" below for details.
  gpu_selection:
    auto: true                 # Default: true. Set false to disable.
    min_free_memory_gb: 8.0    # Reject GPUs with less free VRAM
    count: null                # Number of GPUs; null -> use envs.TP
    candidates: null           # Optional whitelist of physical GPU ids

  # Execution settings
  run_mode: docker             # "docker" (default) or "local" (host / in-container)
  docker_image: null           # Optional: override auto-selected image
  gpu_arch: null               # Optional: force GPU architecture
  timeout_seconds: 3600        # Benchmark timeout
  
  # Paths
  inferencex_path: /path/to/InferenceX  # InferenceX installation
  hf_cache_path: null          # HuggingFace cache directory
  
  # InferenceX specific
  runner_type: mi300x          # Hardware runner type
  benchmark_script: null       # Override benchmark script
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TP` | Tensor parallelism (number of GPUs) | 1 |
| `CONC` | Request concurrency | 32 |
| `ISL` | Input sequence length | 1024 |
| `OSL` | Output sequence length | 512 |
| `RANDOM_RANGE_RATIO` | Length randomization ratio | 0.5 |
| `MAX_MODEL_LEN` | Maximum model context length | - |
| `GPU_MEM_UTIL` | GPU memory utilization | 0.95 |
| `ENABLE_PROFILE` | Enable torch profiler | "false" |
| `EXTRA_VLLM_ARGS` | Additional arguments passed to `vllm serve` | "" |

## Automatic GPU Selection

Before launching the benchmark, Magpie scans the host (`rocm-smi` / `nvidia-smi`),
picks the least-busy GPU(s) with enough free VRAM, and pins the run via
vendor-specific environment variables:

- **AMD**: `ROCR_VISIBLE_DEVICES=<ids>` (the launcher script remaps
  `HIP_VISIBLE_DEVICES` to the post-filter logical range `0..N-1`).
- **NVIDIA**: `CUDA_VISIBLE_DEVICES=<ids>` + `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

GPU ids use the same index space as `rocm-smi` / `nvidia-smi`. By default the
selector asks for `envs.TP` idle GPUs; override with `gpu_selection.count`.

**Config knobs** (all optional; `gpu_selection` block is enabled by default):

| Field | Default | Description |
|-------|---------|-------------|
| `auto` | `true` | Set `false` to disable and let the framework see every GPU |
| `min_free_memory_gb` | `8.0` | Reject GPUs with less free VRAM than this |
| `count` | `null` | Number of GPUs to pin; `null` → `envs.TP` |
| `candidates` | `null` | Optional whitelist of physical GPU ids to consider |

**Manual override**: setting `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` /
`ROCR_VISIBLE_DEVICES` in `envs:` pins to specific cards and skips auto-selection:

```yaml
  envs:
    TP: 1
    ROCR_VISIBLE_DEVICES: "3"    # AMD: pin to rocm-smi GPU[3]
    # CUDA_VISIBLE_DEVICES: "3"  # NVIDIA alternative
    # CUDA_DEVICE_ORDER: PCI_BUS_ID
```

**Ray mode** (`run_mode: ray`): `gpu_selection` is ignored — Ray schedules
devices itself via `num_gpus`. To restrict the cluster to specific cards,
export `ROCR_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` in the shell before
starting `ray start`.

## Profiling Options

### Torch Profiler

When `torch_profiler.enabled: true`:
- Sets `VLLM_TORCH_PROFILER_DIR` automatically
- Generates JSON trace files for each GPU rank
- Traces saved to: `results/benchmark_<framework>_<timestamp>/torch_trace/`

### TraceLens Analysis

TraceLens provides automated analysis of torch profiler traces:

| Command | Description | Output |
|---------|-------------|--------|
| `TraceLens_generate_perf_report_pytorch` | Single-rank performance report | `tracelens_rank0_csvs/` |
| `TraceLens_generate_multi_rank_collective_report_pytorch` | Multi-rank collective analysis | `tracelens_collective_csvs/` |

#### TraceLens Output Files

**Single-rank report (`tracelens_rank0_csvs/`):**
- `gpu_timeline.csv` - GPU kernel timeline
- `ops_summary.csv` - Operation summary
- `ops_summary_by_category.csv` - Operations by category
- `coll_analysis.csv` - Collective communication analysis
- `kernel_summary.csv` - Kernel summary statistics

**Multi-rank collective report (`tracelens_collective_csvs/`):**
- Aggregated statistics across all GPU ranks
- Communication pattern analysis
- Load balancing metrics

### Gap Analysis

Gap analysis identifies GPU kernel bottlenecks from torch profiler traces. It applies a configurable time window to focus on the steady-state portion of the trace, then aggregates kernel durations by category.

**Pipeline:**
1. Apply time window (`trace_start_pct` – `trace_end_pct`) to isolate steady-state events
2. Filter by category (case-insensitive substring matching on the event `cat` field)
3. Aggregate stats per kernel name, rank by total duration

**CSV output columns:** `Name, Calls, Self CUDA total (us), Avg time (us), % Total`

**Defaults (no YAML needed):**
- `categories`: `["kernel", "gpu"]`
- `ignore_categories`: `["gpu_user_annotation"]`

**Minimal config:**
```yaml
  gap_analysis:
    enabled: true
    trace_start_pct: 50
    trace_end_pct: 80
```

#### Standalone CLI

Run gap analysis on existing trace directories without re-running the benchmark:

```bash
# Basic usage (CLI defaults: --start-pct 0 --end-pct 100 unless you override)
python -m Magpie benchmark gap-analysis \
    --trace-dir results/benchmark_vllm_<timestamp>/

# With custom window and categories (align with YAML gap_analysis window if desired)
python -m Magpie benchmark gap-analysis \
    --trace-dir results/benchmark_vllm_<timestamp>/torch_trace \
    --start-pct 50 --end-pct 80 \
    --top-k 15 \
    --categories kernel gpu \
    --ignore-categories gpu_user_annotation
```

The `--trace-dir` argument accepts either a benchmark workspace directory (auto-detects `torch_trace/` inside) or a direct path to the trace directory.

Output is written to a `gap_analysis/` subfolder under the trace directory's parent.

## Output Structure

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

The primary summary file is **`benchmark_report.json`** in the run workspace (see `WorkspaceManager.save_report`). It aggregates throughput, latency, and optional `gap_analysis` / `tracelens_analysis` sections. A typical shape:

```json
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

## Examples

### Quick Profiling Run

Minimal configuration for fast trace collection:

```yaml
benchmark:
  framework: vllm
  model: deepseek-ai/DeepSeek-R1-0528
  precision: fp8
  
  envs:
    TP: 8
    CONC: 4                    # Small concurrency for quick run
    ISL: 128
    OSL: 64
    GPU_MEM_UTIL: 0.85
    
  profiler:
    torch_profiler:
      enabled: true
    tracelens:
      enabled: true
      export_format: csv
      multi_rank_report_enabled: false  # Skip multi-rank for speed
      
  timeout_seconds: 1200
```

### Full Production Benchmark

```yaml
benchmark:
  framework: vllm
  model: deepseek-ai/DeepSeek-R1-0528
  precision: fp8
  
  envs:
    TP: 8
    CONC: 64
    ISL: 2048
    OSL: 2048
    MAX_MODEL_LEN: 131072
    
  profiler:
    torch_profiler:
      enabled: true
    tracelens:
      enabled: true
      export_format: csv
      perf_report_enabled: true
      multi_rank_report_enabled: true
      
  timeout_seconds: 7200
```

### SGLang Benchmark

```yaml
benchmark:
  framework: sglang
  model: meta-llama/Llama-3.1-70B-Instruct
  precision: fp16
  
  envs:
    TP: 4
    CONC: 32
    ISL: 1024
    OSL: 512
    
  profiler:
    torch_profiler:
      enabled: true
      
  timeout_seconds: 3600
```

## Troubleshooting

### Common Issues

**1. GPU Memory Error**
```
ValueError: Free memory on device (...) is less than desired GPU memory utilization
```
Solution: Reduce `GPU_MEM_UTIL` in config (e.g., 0.85)

**2. Docker Permission Error**
```
docker: permission denied
```
Solution: Add user to docker group or run with sudo

**3. TraceLens Not Found**
```
TraceLens CLI command 'TraceLens_generate_perf_report_pytorch' not found
```
Solution: TraceLens will be auto-installed. If issues persist:
```bash
pip install git+https://github.com/AMD-AIG-AIMA/TraceLens.git
```

**4. Timeout During Model Loading**

Large models (e.g., DeepSeek-R1) may need longer timeouts:
```yaml
timeout_seconds: 7200  # 2 hours
```

**5. `gpu_selection.auto failed: ...`**

Not enough idle GPUs on the host. Either free a GPU, lower
`gpu_selection.min_free_memory_gb`, narrow `gpu_selection.candidates`, or pin
manually via `envs.ROCR_VISIBLE_DEVICES` (AMD) / `envs.CUDA_VISIBLE_DEVICES`
(NVIDIA). See [Automatic GPU Selection](#automatic-gpu-selection).

### Debug Mode

Enable verbose logging:
```bash
python -m Magpie benchmark --benchmark-config config.yaml --log-level DEBUG
```

## Architecture

### Components

| Component | File | Description |
|-----------|------|-------------|
| `BenchmarkMode` | `benchmarker.py` | Main orchestrator |
| `BenchmarkConfig` | `config.py` | Configuration dataclasses |
| `TraceLensAnalyzer` | `tracelens.py` | TraceLens CLI integration |
| `GapAnalyzer` | `gap_analysis.py` | Kernel bottleneck analysis |
| `BenchmarkResult` | `result.py` | Result data structures |

### Execution Flow

1. **Configuration Loading**: Parse YAML config into `BenchmarkConfig`
2. **Runtime Setup**: For `run_mode: docker`, prepare a container with InferenceX; for `local`, use the host environment
3. **Server Launch**: Start vLLM/SGLang server (in container or on host per `run_mode`)
4. **Client Execution**: Run benchmark client with profiling enabled
5. **Trace Collection**: Torch profiler traces saved to workspace
6. **TraceLens Analysis**: Run TraceLens CLI commands on host (if enabled)
7. **Gap Analysis**: Analyze kernel bottlenecks within time window (if enabled)
8. **Result Generation**: Aggregate metrics and generate reports

## Related

- [Analyze vs Compare](analysis_compare.md) — kernel evaluation modes (orthogonal to Benchmark)
- [TraceLens](https://github.com/AMD-AIG-AIMA/TraceLens) — Trace analysis library
- [InferenceX](https://github.com/SemiAnalysisAI/InferenceX) — Benchmark scripts (auto-clone target in default config)
- [vLLM](https://github.com/vllm-project/vllm) — LLM inference engine
- [SGLang](https://github.com/sgl-project/sglang) — LLM serving framework
- [Ray + Magpie](ray-magpie.md) — optional remote benchmark scheduling

