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

  # Optional InferenceX AgentX trace replay workload switch.
  agentx: disabled              # true/enable/enabled also enable it
  
  # Benchmark parameters
  envs:
    TP: 8                      # Tensor parallelism (GPU count)
    CONC: 32                   # Request concurrency
    ISL: 1024                  # Input sequence length (not used by AgentX)
    OSL: 1024                  # Output sequence length (not used by AgentX)
    RANDOM_RANGE_RATIO: 1      # Length randomization (0-1; not used by AgentX)
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
                                    # Patches vLLM v0.14-v0.25 and SGLang. vLLM
                                    # v0.26+ and ATOM ship the profiler options
                                    # upstream and only get TraceLens installed.
      tracelens_repo_path: null     # Optional checkout; otherwise clone main
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

## AgentX trace replay

AgentX workload semantics are enabled with one line. The Docker image defaults
to the resolved recipe image. The InferenceX launcher path is resolved from
`configs/agentx-launchers.json` when that checkout declares the recipe; an
explicit `benchmark_script` must match this mapping. Older checkouts without
the manifest still require an explicit launcher. The recipe and launcher
contents still come from the checkout at `inferencex_path`; pin that checkout
to a commit outside Magpie when exact source reproducibility is required.
Magpie detects the GPU, matches `model`/`framework`/`precision` against that
checkout's AgentX recipe, then executes the requested launcher under
`InferenceX/benchmarks/`. Model prefix, TP/EP, speculative decoding, and
KV-offload settings come from the recipe rather than from Magpie defaults.

```yaml
benchmark:
  framework: sglang
  model: deepseek-ai/DeepSeek-V4-Pro-0813
  precision: fp4
  agentx: enable
  docker_image: lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260914
  benchmark_script: single_node/agentic/dsv4_fp4_mi355x_sglang_mtp.sh
```

AgentX replays traces: input lengths and target output lengths come from the
trace dataset and vary by request. Do not set `ISL`, `OSL`, or
`RANDOM_RANGE_RATIO` for AgentX. Magpie warns and discards these values if they
are supplied under `benchmark.envs`; they are not passed to the launcher or
retained in the saved configuration. The CLI similarly warns and discards
explicit `--input-len` and `--output-len` values when `--agentx` is enabled,
and does not inject the ordinary benchmark length defaults.

Concurrency remains configurable and defaults to 32. Set `CONC` when a different
point from the InferenceX recipe is required:

```yaml
benchmark:
  framework: sglang
  model: deepseek-ai/DeepSeek-V4-Pro-0813
  precision: fp4
  agentx: enable
  docker_image: lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260914
  benchmark_script: single_node/agentic/dsv4_fp4_mi355x_sglang_mtp.sh
  envs:
    CONC: 16
```

The object form exposes optional controls:

```yaml
benchmark:
  framework: sglang
  model: deepseek-ai/DeepSeek-V4-Pro-0813
  precision: fp4
  agentx:
    enabled: true
    mode: canonical            # canonical, or fast for a non-publishable check
    # recipe: dsv4-fp4-mi355x-sglang-agentic-mtp  # ambiguity override only
    # selector: {tp: 8, kv_offloading: dram}       # recipe-arm override only
  docker_image: lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260914
  benchmark_script: single_node/agentic/dsv4_fp4_mi355x_sglang_mtp.sh
```

`MODEL_PREFIX`, `KV_OFFLOADING`, `KV_OFFLOAD_BACKEND`,
and `TOTAL_CPU_DRAM_GB` are not AgentX YAML requirements. They are resolved
from the matching InferenceX recipe. `benchmark_script` and `docker_image`
can explicitly select the launcher path and runtime image; they do not pin
the InferenceX checkout. Missing or ambiguous recipes/arms fail resolution.
`agentx.recipe` remains an advanced ambiguity override.

### Verified server launch overrides

For launchers declaring `launch_overrides_version: 1` in the InferenceX
manifest, `agentx.launch_overrides` applies a structured server-only extension:

```yaml
agentx:
  enabled: true
  launch_overrides:
    version: 1
    append_args: [--mem-fraction-static, '0.8']
    remove_args: [--mem-fraction-static]
    replace_args: false
    env: {SGLANG_USE_AITER: '0'}
    unset_env: []
    executable: null
    source_files: {}           # absolute path -> expected lowercase SHA256
    absent_source_files: []   # absolute paths that must remain absent
```

Arguments are tokens, never shell expressions. Removal must match exactly one
existing long option and removes its following values up to the next long
option. Ambiguous short-option forms fail. `replace_args: true` retains the
executable, positionals, and model/host/port/topology options while replacing
the other serving options. Protocol options and replay environment controls
cannot be overridden. `executable`, when supplied, must be an absolute executable
path: use the Python interpreter for SGLang or the vLLM entrypoint for vLLM.
Environment changes affect the server process, not the replay/router process.

Source hashes and required absences are checked after launcher setup and
immediately before server start. The exact normalized request is saved as
`agentx_launch_overrides.json`. `agentx_server_launch.json` records base/effective
argv, changed environment, filtered runtime controls, resolved executable, and
verified source identities. Both request and evidence have canonical JSON
SHA256 identities. Magpie rejects missing, corrupt, or mismatched evidence and
publishes verified evidence under `agentx_metrics.server_launch`. An empty
`{version: 1}` request records evidence without changing the canonical command.
Omitting `launch_overrides` preserves the older execution contract.

When no registered recipe matches, a checkout with the manifest's `generic`
capabilities can run a custom SGLang or vLLM model on MI300X/MI325X/MI355X:

```yaml
benchmark:
  model: Qwen/Qwen3-0.6B
  framework: sglang
  precision: bf16
  docker_image: your-tested-sglang-image@sha256:your-digest
  agentx:
    enabled: true
    launch_overrides: {version: 1}
  envs:
    MODEL_PATH: /models/Qwen3-0.6B
    TP: 1
    EP_SIZE: 1
    CONC: 64
```

No matching recipe triggers this path; multiple registered matches still require
an explicit selection. The resolved public name is `custom-<framework>-<runner>`.
An explicit runtime image and TP/EP are required. TP must fit one eight-GPU node;
SGLang EP must divide TP, while vLLM supports EP=1 or EP=TP. PP/DCP/PCP remain 1,
with no DP attention, disaggregation, KV offload, or inferred speculative decoding.
Quantized precisions require the model's own `quantization_config`.

Custom replay uses the same `inferencex-agentx-mvp` scenario and canonical duration.
Magpie reads local `MODEL_PATH/config.json`, or only the remote HuggingFace
`config.json` metadata when MODEL_PATH is absent. It never executes model code
or downloads weights during resolution. The native context, metadata SHA256,
fixed trace loader, and optional `MAX_MODEL_LEN` cap become recipe identity.
The cap cannot exceed confirmed native context. The launcher verifies the same
metadata bytes before server start and passes the fixed cap to both server and
replay. Launch overrides cannot change that context. Custom results carry
`custom_recipe`, `native_context_length`, `max_model_len`, and
`model_config_sha256` in raw output and `agentx_metrics.recipe`; results with
different model/context/metadata identities are different workloads.

Magpie AgentX v1 is single-node and supports Docker or local execution. Its
own trace-replay loop is incompatible with Ray, persistent-server reuse,
PyTorch/system profiling, TraceLens, and gap analysis. Those profilers default
to disabled when AgentX is enabled. A successful `fast` run is marked
`benchmark_valid: true` but `publishable: false`; canonical mode is required
for a publishable result.

The `profile` in `aiperf profile` means workload measurement, not PyTorch
profiling. AgentX v1 collects request-level AIPerf data, server metrics, and
power artifacts, but it does not create `torch_trace/` profiler files.

## Environment variables

Pass these variables under `benchmark.envs:` to control request shape, concurrency, memory usage, and profiling behavior.

| Variable | Description | Default |
|----------|-------------|---------|
| `TP` | Tensor parallelism (number of GPUs) | 1 |
| `CONC` | Request concurrency | 32 |
| `ISL` | Input sequence length (ordinary benchmarks only; ignored with a warning for AgentX) | 1024 |
| `OSL` | Output sequence length (ordinary benchmarks only; ignored with a warning for AgentX) | 512 |
| `RANDOM_RANGE_RATIO` | Length randomization ratio (ordinary benchmarks only; ignored with a warning for AgentX) | 0.5 |
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
      # tracelens_repo_path can select a checkout; otherwise Magpie clones main
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
the benchmark `ISL`, `OSL`, and `CONC` values. Without an explicit
`gpu_arch_config`, Magpie infers a candidate platform from `runner_type` and
checks it against `list_platforms()` in the actual TraceLens post-processing
environment. The check includes `TL_EXTENSION`, so an extension wheel can add
platform support such as `MI355X`. Magpie passes `--gpu_arch_platform` only
when the candidate is supported; otherwise it warns and continues without
architecture-specific roofline data. An explicit `gpu_arch_config` takes
priority and is passed as `--gpu_arch_json_path`.

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
