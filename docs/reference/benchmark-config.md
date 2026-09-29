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
      # The following settings apply only to AgentX diagnostic capture:
      num_steps: 20             # Positive integer framework step count
      capture_timeout_seconds: 300  # Start capture and wait for the first trace
      flush_timeout_seconds: 1800   # Wait for all rank traces
      
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

Enable AgentX with one line. For a new installation, Magpie checks out
InferenceX commit `408c015be4b22d14c69518643609669405507077` and initializes
only the AIPerf client submodule. It does not install or submit through Slurm.
`inferencex_path` accepts either the old repository root, the new repository
root containing `inferencex-e2e/`, or that project directory directly. Magpie
validates the upstream library and client sources and leaves existing checkouts
unchanged; choose an explicit checkout to use another revision.

On the new layout, Magpie matches `model`/`framework`/`precision` and the
selected concurrency to a single-node YAML recipe. It preserves TP/EP,
speculative decoding, KV offload, setup dependencies, and upstream golden
acceptance settings. Magpie starts the server and runs the official
`srt_agentic.sh` client against it. Each point has a fresh server; warmup and
measurement share that point's server. Unknown deployment fields, ambiguous
variants, routers, zipped sweeps, and multi-node recipes fail explicitly.

```yaml
benchmark:
  framework: sglang
  model: deepseek-ai/DeepSeek-V4-Pro-0813
  precision: fp4
  agentx: enable
  docker_image: lmsysorg/sglang-rocm:v0.5.20-rocm720-mi35x-20260926
  envs:
    CONC: 32  # Active agent session trees, including their subagents.
```

The image may be omitted to use the recipe image. On the new layout an explicit
image must match the registered recipe, and `benchmark_script`, if supplied,
must be `srt_agentic.sh`. Old checkouts retain the old launcher compatibility
path: `configs/agentx-launchers.json` selects a launcher when available;
otherwise provide the old `benchmark_script` explicitly.

AgentX replays traces: input lengths and target output lengths come from the
trace dataset and vary by request. Do not set `ISL`, `OSL`, or
`RANDOM_RANGE_RATIO` for AgentX. Magpie warns and discards these values if they
are supplied under `benchmark.envs`; they are not passed to the launcher or
retained in the saved configuration. The CLI similarly warns and discards
explicit `--input-len` and `--output-len` values when `--agentx` is enabled,
and does not inject the ordinary benchmark length defaults.

`benchmark.envs.CONC` sets the number of active agent session trees, including
their subagents. It does not fix the number of simultaneous HTTP requests.
The examples set it explicitly; omitting it defaults to 32. Choose a concurrency
point supported by the InferenceX recipe, for example:

```yaml
benchmark:
  framework: sglang
  model: deepseek-ai/DeepSeek-V4-Pro-0813
  precision: fp4
  agentx: enable
  docker_image: lmsysorg/sglang-rocm:v0.5.20-rocm720-mi35x-20260926
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
  docker_image: lmsysorg/sglang-rocm:v0.5.20-rocm720-mi35x-20260926
  envs:
    CONC: 32  # Active agent session trees, including their subagents.
```

`MODEL_PREFIX`, `KV_OFFLOADING`, `KV_OFFLOAD_BACKEND`,
and `TOTAL_CPU_DRAM_GB` are not AgentX YAML requirements. They are resolved
from the matching InferenceX recipe. The checkout determines recipe and client source versions; the Docker image
determines the serving framework version. Missing or ambiguous recipes/arms fail resolution.
`agentx.recipe` remains an advanced ambiguity override.

### Verified server launch overrides

Magpie-managed serving supports `agentx.launch_overrides` directly, without
a modified InferenceX launcher or manifest. Legacy launchers must explicitly
declare version 1 support. The extension applies only to the server:

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
For recipes with golden acceptance, its acceptance values and curve inputs
(speculative method, draft model, and token budget) cannot be changed by a
candidate override. Select and resolve a corresponding recipe to change them;
other serving optimizations remain available.

Source hashes and required absences are checked after server setup and
immediately before server start. The exact normalized request is saved as
`agentx_launch_overrides.json`. `agentx_server_launch.json` records base/effective
argv, changed environment, filtered runtime controls, resolved executable, and
verified source identities. Both request and evidence have canonical JSON
SHA256 identities. Magpie rejects missing, corrupt, or mismatched evidence and
publishes verified evidence under `agentx_metrics.server_launch`. An empty
`{version: 1}` request records evidence without changing the canonical command.
Managed runs always record the recipe/server specification and client source
hashes. Omitting `launch_overrides` on the legacy path preserves its older
execution contract. `EXTRA_SGLANG_ARGS` or `EXTRA_VLLM_ARGS` is converted once
to literal override tokens on the managed path, with the same protocol guards.

When no registered recipe matches, the new layout supports a Magpie-managed
custom SGLang or vLLM model on MI300X/MI325X/MI355X. Older checkouts require
the corresponding generic capability declaration:

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
`config.json` metadata when MODEL_PATH is absent. Public metadata is resolved
to an immutable HuggingFace revision, which is also passed to the server. This
metadata request does not forward credentials; use a downloaded local
`MODEL_PATH` for gated models. Resolution never executes model code
or downloads weights during resolution. The native context, metadata SHA256,
fixed trace loader, and optional `MAX_MODEL_LEN` cap become recipe identity.
The cap cannot exceed confirmed native context. The runtime verifies the same
metadata bytes before server start and passes the fixed cap to both server and
replay. Launch overrides cannot change that context. Custom results carry
`custom_recipe`, `native_context_length`, `max_model_len`, and
`model_config_sha256` in raw output and `agentx_metrics.recipe`; results with
different model/context/metadata identities are different workloads.

Magpie AgentX v1 is single-node and supports Docker or local execution. Its
trace-replay measurement is incompatible with Ray, persistent-server reuse,
system profiling, TraceLens inference mode, and gap analysis. Torch profiling
defaults to disabled when AgentX is enabled. A successful unprofiled `fast` run is marked
`benchmark_valid: true` but `publishable: false`; canonical mode is required
for a publishable result. Managed AgentX needs no `server_lifecycle` flag. If
provided, `cleanup: true` and `force_reuse: false` are required;
`server_ready_timeout_s` controls startup independently of the client timeout.
Legacy launchers do not support `server_lifecycle`.

The `profile` in `aiperf profile` means workload measurement, not PyTorch
profiling. Managed AgentX can optionally collect framework traces under
`torch_trace/` using `profiler.torch_profiler.enabled: true`. Magpie waits for
AIPerf's profiling phase before requesting a framework capture; the framework
stops capture after `num_steps`. Magpie waits for every rank's trace to finish
before cleanup. `num_steps` must be a positive integer, and both timeouts must
be finite positive numbers. These three settings apply only to AgentX
diagnostics; ordinary benchmark profiling keeps its existing behavior.

TraceLens post-processing is supported with explicit
`profiler.tracelens.analysis_mode: pytorch` and torch capture enabled.
`analysis_mode: inference` is rejected because its preprocessing modifies the
upstream checkout. Legacy AgentX shell launchers do not support this diagnostic
capture path.

Every profiled AgentX run is diagnostic: it sets `benchmark_valid: false` and
`publishable: false`, even when all traces are captured. Do not use its metrics
for baseline/candidate comparison or KEEP decisions. Use a separate unprofiled
run for those decisions. See the
[diagnostic example and command](../how-to/benchmarking/profiling-options.md#agentx-diagnostic-traces).
GPU execution of the diagnostic path has not yet been validated.

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
