---
myst:
    html_meta:
        "description": "Full YAML configuration reference for Magpie benchmark mode, including profiler settings, gap analysis, GPU selection, run modes, and environment variable options."
        "keywords": "Magpie, benchmark configuration, YAML, TraceLens, gap analysis, GPU selection, vLLM, SGLang, ROCm, profiler"
---

# Magpie benchmark mode configuration

Magpie benchmark mode is configured entirely through YAML files, which control the inference framework, model, request shape, profiling backends, GPU selection, and execution environment. All settings live under a top-level `benchmark:` key and are passed to the CLI with `--benchmark-config`; you can also set individual values directly on the command line for quick experimentation. This page provides a minimal starting configuration, a full annotated reference with every available option, a table of environment variables for request shape and profiling, and ready-to-run examples for common scenarios.

## Configuration

All benchmark settings live under a top-level `benchmark:` key in a YAML file passed to `--benchmark-config`.

### Minimal example

The following configuration runs a basic vLLM benchmark with torch profiling enabled.

```yaml
benchmark:
  framework: vllm              # "vllm", "sglang", or "atom"
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

### Full configuration reference

The following shows every available option with its default or a representative value.

```yaml
benchmark:
  # Framework selection
  framework: vllm              # Required: "vllm", "sglang", or "atom"
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
      enabled: true                 # Enable TraceLens analysis
      analysis_mode: inference      # Optional, default: inference
      analysis_stages: all          # Optional, default: all
      auto_patch_runtime: true      # Optional, default: true for Docker runs
      tracelens_repo_path: null     # Optional public TraceLens source checkout
      extension_wheel_path: null    # Optional local TraceLens extension wheel
      cli_timeout_seconds: 2400     # TraceLens postprocess timeout per command
      export_format: csv            # "csv" or "excel"
      perf_report_enabled: true           # Single-rank performance report
      multi_rank_report_enabled: true     # Multi-rank collective report
      gpu_arch_config: null         # Optional: GPU arch JSON passed to TraceLens

  # Gap analysis (kernel bottleneck report)
  gap_analysis:
    enabled: true              # Enable gap analysis after benchmark
    trace_start_pct: 50        # Start of analysis window (0-100)
    trace_end_pct: 80          # End of analysis window (0-100)
    top_k: 20                  # Number of top kernels in report
    min_duration_us: 0.0       # Filter out events shorter than this (us)
    categories:                # Event category allowlist (default: [kernel, gpu])
      - kernel
      - gpu
    ignore_categories:         # Event category denylist (default: [gpu_user_annotation])
      - gpu_user_annotation
      
  # Automatically selects idle GPU(s) before launching (enabled by default).
  # See "Automatic GPU Selection" below for details.
  gpu_selection:
    auto: true                 # Default: true. Set false to disable.
    min_free_memory_gb: 8.0    # Reject GPUs with less free VRAM
    count: null                # Number of GPUs; null -> use envs.TP
    candidates: null           # Optional allowlist of physical GPU ids

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

## Environment variables

Pass these variables under `benchmark.envs:` to control request shape, concurrency, memory usage, and profiling behavior.

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

## Examples

The following example configurations cover common benchmark scenarios.

### Quick profiling run

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
      # analysis_mode defaults to inference
      # analysis_stages defaults to all (prefilldecode, decode, prefill)
      # auto_patch_runtime defaults to true for Docker runs
      # tracelens_repo_path can point to a public TraceLens checkout
      # extension_wheel_path can add a local TraceLens extension to the image
      # cli_timeout_seconds defaults to 1800
      export_format: csv
      multi_rank_report_enabled: false  # Skip multi-rank for speed
      
  timeout_seconds: 1200
```

### Full production benchmark

Full configuration with TraceLens inference analysis enabled across all stages:

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
      analysis_mode: inference
      analysis_stages: all
      auto_patch_runtime: true
      # tracelens_repo_path: /path/to/TraceLens
      # extension_wheel_path: /secure/path/to/TraceLens_extension.whl
      cli_timeout_seconds: 2400
      export_format: csv
      perf_report_enabled: true
      multi_rank_report_enabled: true
      
  timeout_seconds: 7200
```

When `profiler.tracelens.analysis_mode: inference` is enabled, Magpie writes
the full TraceLens CSV reports under stage subdirectories such as
`tracelens/prefilldecode/`, `tracelens/decode_only/`, and
`tracelens/prefill_only/`. It also creates one compact roofline review file per
stage in the `tracelens/` root:

```text
tracelens/prefilldecode_ISL1024_OSL1024_CONC64_kernel_roofline_simple.csv
tracelens/decode_only_ISL1024_OSL1024_CONC64_kernel_roofline_simple.csv
tracelens/prefill_only_ISL1024_OSL1024_CONC64_kernel_roofline_simple.csv
```

These simple files are generated from each stage's `unified_perf_summary.csv`
and category-specific `param:*` CSVs. They are designed for quick review and
include operation category, operation name, `param_signature`, `params_json`,
kernel time, total time percentage, arithmetic intensity, achieved TFLOP/s,
achieved TB/s, roofline bound, and percent of roofline. The filename records
the benchmark `ISL`, `OSL`, and `CONC` values. Magpie passes
`gpu_arch_config` to the inference CLI as `--gpu_arch_json_path`; when it is
not configured, architecture-dependent roofline columns are omitted from the
simple summary. Magpie does not pass TraceLens' `--gpu_arch_platform` by
default because bundled platform support can lag new hardware; use
`gpu_arch_config` for hardware-specific roofline data.

### SGLang benchmark

Basic SGLang benchmark with torch profiler enabled:

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

## Related topics

See the following pages for related concepts, how-to guidance, and reference material.

- [Benchmark frameworks with Magpie](../how-to/benchmarking/benchmark.md): how-to guide covering run modes, TraceLens analysis, gap analysis, and automatic GPU selection
- [Magpie benchmarking mode architecture](../conceptual/benchmarking-architecture.md): how the benchmark pipeline components interact
- [Run Magpie on a Ray cluster](../how-to/ray.md): running benchmarks on remote GPU nodes using `run_mode: ray`
- [Magpie API reference](api-reference.md): CLI options for `magpie benchmark` and standalone gap analysis
- [Magpie troubleshooting](troubleshooting.md): solutions for common benchmark errors
