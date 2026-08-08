---
myst:
    html_meta:
        "description": "Configure Magpie's profiling backends for benchmark mode: torch profiler for trace capture, TraceLens for inference analysis, and gap analysis for kernel bottleneck reporting."
        "keywords": "Magpie, profiling, torch profiler, TraceLens, gap analysis, benchmark, ROCm, vLLM, SGLang, kernel bottleneck"
---

# Profiling options in Magpie's benchmark mode

Magpie supports three profiling backends that can be enabled independently and combined in a single benchmark run: the torch profiler captures per-rank JSON traces, TraceLens runs inference-aware analysis on those traces to produce prefill/decode performance reports, and gap analysis aggregates kernel durations from the traces to identify the operations that dominate runtime. Each backend is configured under the `profiler:` key in your benchmark YAML and writes its output to a dedicated subdirectory in the benchmark workspace. Enable only the backends you need — torch profiler is a prerequisite for both TraceLens and gap analysis, but TraceLens and gap analysis are independent of each other.

```{note}
Profiling is an optional feature of Magpie's benchmark mode. For a full overview
of benchmark mode, including run modes, configuration, and output structure, see
[Benchmark frameworks with Magpie](benchmark.md).
```

## Torch profiler

When `torch_profiler.enabled: true`, Magpie takes the following actions.

- Sets `VLLM_TORCH_PROFILER_DIR` automatically
- Generates JSON trace files for each GPU rank
- Traces saved to: `results/benchmark_<framework>_<timestamp>/torch_trace/`

## TraceLens analysis

TraceLens provides automated analysis of torch profiler traces:

| Command | Description | Output |
|---------|-------------|--------|
| `TraceLens_`<br>`split_`<br>`inference`<br>`_trace` | Split vLLM/SGLang inference traces into phase windows | `torch_trace/trace_split/` |
| `TraceLens_`<br>`generate_perf`<br>`_report_pytorch`<br>`_inference` | Inference-aware prefill/decode reports and compact roofline summaries | `tracelens/` |
| `TraceLens_`<br>`generate_perf`<br>`_report_pytorch` | Single-rank performance report | `tracelens_rank0_csvs/` |
| `TraceLens_`<br>`generate_multi`<br>`_rank_collective`<br>`_report_pytorch` | Multi-rank collective analysis | `tracelens_collective_csvs/` |

`analysis_mode` defaults to `inference`, which is the recommended mode for
vLLM/SGLang benchmarks. It automatically enables the torch profiler, patches the
needed InferenceX profiling helpers for the run, splits the rank-0 trace, and
runs reports for all inference stages. Use `analysis_mode: pytorch` to keep the
legacy direct PyTorch report flow.

For Docker benchmarks, `auto_patch_runtime` defaults to `true`. When TraceLens
inference mode is enabled and the selected runtime image is not already
TraceLens-ready, Magpie builds a derived image from supported official
vLLM/SGLang tags. SGLang continues to use the public TraceLens workflow script.
For vLLM, Magpie archives the selected committed TraceLens tree, builds its
wheel without dependencies in the exact base image, and installs only pinned
Matplotlib diagnostic wheels with `--no-deps`. It deliberately does not install
`xprof`, `gcsfs`, or `grpcio-status`, and verifies that the base image's `vllm`
and `grpcio` versions remain unchanged. The final Docker build is network-free.
The derived image is tagged locally as `magpie-tracelens-<framework>:...` and
reused on later runs only after validation.

When the selected base is a locally derived image without a repository digest,
Magpie first gives the exact image ID a stable, content-addressed `localhost/`
retention tag, then binds it to a unique owned build-only tag instead of writing
`FROM sha256:...` (which BuildKit interprets as a registry tag). The retention
tag keeps a formerly unnamed parent inspectable after Docker removes the unique
tag. The final build disables pulls, verifies both bindings before and after the
build, and removes only the unique tag created by that build. Repository-digest
parents keep using their immutable digest directly and need no retention tag.

The vLLM image carries OCI labels and a runtime identity document containing the
base image ID, TraceLens source commit and tree, patch hash, pinned wheel hashes,
and preserved package versions. A same-name local image with missing or stale
identity, changed base ancestry, forbidden packages, import failures, or patch
marker failures is rejected and rebuilt. These fields are also returned in the
benchmark runtime metadata. TraceLens' upstream wheel metadata still declares
features outside Magpie's CSV diagnostic path, so a whole-environment
`pip check` can report intentionally omitted packages; Magpie instead validates
the exact splitter/report/import path it executes.

If no TraceLens source path is configured, Magpie shallow clones the official
TraceLens `main` branch to
`$XDG_CACHE_HOME/magpie/TraceLens` (or `~/.cache/magpie/TraceLens`) and reuses
that checkout. It does not scan other local directories for TraceLens. Set
`profiler.tracelens.tracelens_repo_path` or `TRACELENS_REPO_PATH` to use a
specific checkout instead. An invalid explicit path is reported as an error and
does not fall back to `main`.

To install an optional local TraceLens extension in the derived image, set
`extension_wheel_path`:

```yaml
profiler:
  tracelens:
    enabled: true
    auto_patch_runtime: true
    extension_wheel_path: /secure/path/to/TraceLens_extension.whl
```

Magpie infers the extension's import module from its wheel layout, installs the
wheel with `--no-deps` as a final image layer, and adds the module to
`TL_EXTENSION`. The wheel SHA-256 is included in the derived image tag, so a
changed wheel creates a new image instead of silently reusing an older one.
The wheel remains in a temporary Docker build context and is not copied into
the public TraceLens checkout. This option also works when `docker_image`
already names a TraceLens-ready image.

For `run_mode: docker`, TraceLens inference post-processing also runs inside the
resolved runtime image after the benchmark container exits. The post-processing
container is CPU-only, mounts the benchmark workspace at `/workspace`, and writes
CSV outputs under `tracelens/`. Host Python only needs Docker; it does not need
the TraceLens CLI on `PATH`.

`analysis_stages` defaults to `all`:

```yaml
profiler:
  tracelens:
    enabled: true
    analysis_stages: all
```

To run only selected stages:

```yaml
profiler:
  tracelens:
    enabled: true
    analysis_stages: [prefill, decode]
```

Supported stage names are `prefilldecode` (alias: `mixed`), `prefill`, and
`decode`. Magpie infers a candidate TraceLens architecture from the benchmark
runner (for example, `mi355x` maps to `MI355X`) and asks the TraceLens
installation used for post-processing whether that platform is supported. It
passes `--gpu_arch_platform` only when the candidate appears in TraceLens'
`list_platforms()` result. This check runs with `TL_EXTENSION`, so platforms
provided by an extension wheel are included. If the platform is unavailable,
Magpie logs a warning and continues without architecture-specific roofline
data. An explicit `tracelens.gpu_arch_config` takes priority and is passed as
`--gpu_arch_json_path`.

In particular, the public TraceLens commit may not yet bundle `MI355X` even
though the vLLM instrumentation patch supports an MI355X workload. In that
case the trace, stage splitting, timing tables, kernel attribution, and CSV
reports remain valid; only architecture-dependent roofline columns are omitted.
Provide a reviewed `gpu_arch_config` or extension wheel to enable those columns.

For SGLang, TraceLens inference mode automatically adds
`--enable-profile-cuda-graph`. It also adds
`--enable-shape-discovery-for-cuda-graph-profile` when the configured Docker
image name looks like a TraceLens-patched SGLang image, such as
`tracelens-sglang:*` or `magpie-tracelens-sglang:*`. For local runs, Magpie also
detects whether the installed SGLang exposes the patched server argument. For
other SGLang builds, keep patched-runtime-only flags explicit in
`EXTRA_SGLANG_ARGS`.

Each TraceLens inference postprocess command uses `cli_timeout_seconds`, which
defaults to `1800`. Increase it for long-output runs where splitting the full
decode trace can take longer:

```yaml
profiler:
  tracelens:
    enabled: true
    cli_timeout_seconds: 2400
```

To enable an internal TraceLens extension, set `TL_EXTENSION` either in the
shell environment or under benchmark envs. Magpie does not interpret the value;
it only passes the variable through to the benchmark and TraceLens
post-processing commands:

```yaml
benchmark:
  envs:
    TL_EXTENSION: "TraceLens_NDA"
```

### TraceLens output files

TraceLens writes results into the following directories under the benchmark workspace.

**Inference reports (`tracelens/`):**
- `prefilldecode/` - Mixed prefill+decode phase report
- `decode_only/` - Pure decode phase report
- `prefill_only/` - Pure prefill phase report
- `prefilldecode_ISL1024_OSL1024_CONC64_kernel_roofline_simple.csv` - Compact roofline summary for the mixed phase
- `decode_only_ISL1024_OSL1024_CONC64_kernel_roofline_simple.csv` - Compact roofline summary for decode
- `prefill_only_ISL1024_OSL1024_CONC64_kernel_roofline_simple.csv` - Compact roofline summary for prefill, when a prefill trace is available

TraceLens inference mode runs this post-processing flow:

1. Capture torch profiler traces during the benchmark run.
2. Split the rank-0 trace into representative inference phase windows under `torch_trace/trace_split/`.
3. Run TraceLens perf reports for the configured `analysis_stages`.
4. Write the full TraceLens CSV set into one subdirectory per stage.
5. Write one stage-level `*_ISL*_OSL*_CONC*_kernel_roofline_simple.csv` file in the `tracelens/` root.

The simple roofline files are intended as the first file to open when reviewing
TraceLens output. They keep the most useful roofline and timing columns from
`unified_perf_summary.csv` and add matched `param:*` metadata from the
category-specific TraceLens CSVs. The stage and benchmark `ISL`, `OSL`, and
`CONC` values are encoded in the filename, so the CSV itself does not include a
`stage` column.

Simple roofline columns:

| Column | Meaning |
|--------|---------|
| `source_category` | TraceLens op category, such as `GEMM`, `MoE_unfused`, or `InferenceAttention` |
| `op_name` | Operation or pseudo-op name |
| `param_signature` | Human-readable parameter summary from category CSV `param:*` columns |
| `operation_count` | Number of operation instances aggregated into this row |
| `num_kernels` | Number of GPU kernels associated with the matched category row, when available |
| `kernel_time_ms_sum` | Total kernel time for the row, converted from microseconds to milliseconds |
| `time_pct` | Percent of total kernel time in the current stage |
| `kernel_time_us_mean` | Mean kernel time per operation instance, in microseconds |
| `gflops` | Estimated operation work in GFLOPs |
| `data_moved_mb` | Estimated data movement in MB |
| `arithmetic_intensity_flops_per_byte` | Arithmetic intensity, the roofline x-axis |
| `achieved_tflops_mean` | Mean achieved compute throughput in TFLOP/s |
| `achieved_tbps_mean` | Mean achieved memory bandwidth in TB/s |
| `compute_spec` | Compute capability/model used by TraceLens, for example `matrix_bf16` or `matrix_fp4` |
| `roofline_bound` | TraceLens roofline classification, such as `COMPUTE_BOUND` or `MEMORY_BOUND` |
| `pct_roofline_mean` | Mean percentage of the modeled roofline achieved |
| `roofline_time_us` | Modeled roofline time in microseconds |
| `has_perf_model` | Whether TraceLens had a performance model for the row |
| `params_json` | Machine-readable JSON copy of the matched params |
| `input_dims` | Input shapes recorded by TraceLens |
| `input_type` | Input dtypes/types recorded by TraceLens |

Sort the simple CSV by `kernel_time_ms_sum` or `time_pct` to find the dominant
operations first, then use `roofline_bound`, arithmetic intensity, achieved
TFLOP/s, achieved TB/s, and `pct_roofline_mean` to decide whether to investigate
compute efficiency, memory traffic, or missing performance-model coverage.

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

## Gap analysis

Gap analysis identifies GPU kernel bottlenecks from torch profiler traces. It applies a configurable time window to focus on the steady-state portion of the trace, then aggregates kernel durations by category.

The analysis pipeline runs the following steps.

1. Apply time window (`trace_start_pct` – `trace_end_pct`) to isolate steady-state events
2. Filter by category (case-insensitive substring matching on the event `cat` field)
3. Aggregate stats per kernel name, rank by total duration

**CSV output columns**: `Name, Calls, Self CUDA total (us), Avg time (us), % Total`

**Defaults (no YAML needed)**:
- `categories`: `["kernel", "gpu"]`
- `ignore_categories`: `["gpu_user_annotation"]`

**Minimal config**:
```yaml
  gap_analysis:
    enabled: true
    trace_start_pct: 50
    trace_end_pct: 80
```

### Standalone CLI

Run gap analysis on existing trace directories without re-running the benchmark.

```bash
# Basic usage (CLI defaults: --start-pct 0 --end-pct 100 unless you override)
python -m Magpie benchmark \
    --trace-dir results/benchmark_vllm_<timestamp>/

# With custom window and categories (align with YAML gap_analysis window if desired)
python -m Magpie benchmark \
    --trace-dir results/benchmark_vllm_<timestamp>/torch_trace \
    --start-pct 50 --end-pct 80 \
    --top-k 15 \
    --categories kernel gpu \
    --ignore-categories gpu_user_annotation
```

The `--trace-dir` argument accepts either a benchmark workspace directory (auto-detects `torch_trace/` inside) or a direct path to the trace directory.

Output is written to a `gap_analysis/` subfolder under the trace directory's parent.
