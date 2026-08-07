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

- [Magpie benchmarking mode architecture](../../conceptual/benchmarking-architecture.md): how the benchmark pipeline is designed and how the components interact
- [Magpie benchmark mode configuration](../../reference/benchmark-config.md): full YAML schema with all available options and defaults
- [Magpie troubleshooting](../../reference/troubleshooting.md): solutions for common benchmark errors

```{toctree}
:maxdepth: 1
:hidden:

automatic-gpu
persistent-server-reuse
profiling-options
```

## Quick start

### Before you begin

- **Magpie installed**: see [Install Magpie](../../install/install.md).
- **Docker installed and running**: benchmark mode defaults to `run_mode: docker`. Verify with `docker info`.
- **ROCm-compatible GPU with sufficient VRAM**: the example configs target AMD Instinct™ GPUs (MI300X/MI355X). DeepSeek-R1 requires 8 GPUs at fp8; smaller models need less. Magpie [selects idle GPUs automatically](automatic-gpu.md).
- **HuggingFace token**: required for gated models. Set `HF_TOKEN` in your environment before running.
- **InferenceX**: cloned automatically on first run; no manual install needed.
- **Pinned lm-eval runtime for accuracy runs**: `RUN_EVAL=true` requires a
  caller-built `benchmark.lm_eval_runtime`. Magpie never resolves or installs
  evaluator packages during a benchmark.

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
├── inferencex_result.json     # Raw InferenceX output
├── inferencex_runtime_receipt.json # Exact source/runtime identity
├── inferencex_runtime/        # Private run-scoped InferenceX tree
├── model_revision_receipt.json # Requested/resolved HF snapshot (when pinned)
├── lm_eval_runtime_manifest.json # Preserved content/identity manifest
├── lm_eval_runtime_receipt.json  # In-container ABI/import verification
├── lm_eval/                   # Preserved serving accuracy artifacts (RUN_EVAL=true)
├── torch_trace/               # Raw torch profiler traces
│   ├── *-rank-0.*.pt.trace.json.gz
│   ├── *-rank-1.*.pt.trace.json.gz
│   └── ...
├── targeted_trace/            # Diagnostic selected-kernel evidence (if enabled)
│   ├── manifest.json          # Schema/provenance/coverage and shard receipts
│   ├── summary.json           # Streaming integrity/coverage summary
│   └── shards/                # Checksummed PID/rank JSONL shards
├── gap_analysis/              # Gap analysis output (if enabled)
│   ├── gap_analysis.csv       # Merged kernel stats across all ranks
│   ├── gap_analysis_rank0.csv # Per-rank kernel stats
│   ├── gap_analysis_rank1.csv
│   └── ...
├── tracelens/                 # TraceLens inference reports (if enabled)
│   ├── prefilldecode/         # Full TraceLens CSVs for mixed prefill+decode
│   ├── decode_only/           # Full TraceLens CSVs for decode
│   ├── prefill_only/          # Full TraceLens CSVs for prefill, when available
│   ├── prefilldecode_ISL1024_OSL1024_CONC64_kernel_roofline_simple.csv
│   ├── decode_only_ISL1024_OSL1024_CONC64_kernel_roofline_simple.csv
│   └── prefill_only_ISL1024_OSL1024_CONC64_kernel_roofline_simple.csv
├── tracelens_rank0_csvs/      # Legacy/direct PyTorch single-rank report
└── tracelens_collective_csvs/ # Legacy/direct PyTorch multi-rank collective report
```

For TraceLens inference runs, the
`*_ISL*_OSL*_CONC*_kernel_roofline_simple.csv` files are the fastest starting
point for review. Each file covers one inference stage, encodes the benchmark
`ISL`, `OSL`, and `CONC` values in the filename, keeps the most important
roofline and timing columns, and includes both a compact `param_signature` and
machine-readable `params_json` for matched TraceLens `param:*` metadata.

## Benchmark report

The primary summary file is **`benchmark_report.json`**, written to the run workspace directory. It aggregates throughput, latency, and optional `gap_analysis` and `tracelens_analysis` sections.

Every report declares `run_kind` and `reward_eligible`. A
`run_kind: measurement` run rejects heavy profilers; diagnostic runs and all
TargetedKernelTrace artifacts have `reward_eligible: false`. When `RUN_EVAL=true`,
raw lm-eval files remain under `lm_eval/` and `quality_gate` exposes each task's
strictly ordered primary metric, a content-bound `outcome_digest`, the raw
artifact receipts, and a `sample_set_digest`. The same run must provide the
nested runtime configuration shown below.

Apex's reviewed Qwen view also
sets `MAGPIE_EVAL_MAX_LENGTH=2248` and `MAGPIE_EVAL_MAX_GEN_TOKENS=480`:
the former is evaluator request admission, while the latter is the independent
generation budget. `MAX_MODEL_LEN` remains the serving context limit. The
locked Magpie helper constructs this argv; it does not patch InferenceX.

```yaml
benchmark:
  envs:
    RUN_EVAL: "true"
  lm_eval_runtime:
    path: /absolute/path/to/content-addressed/runtime
    sha256: <64-lowercase-hex-runtime-digest>
    identity:
      lm_eval_commit: <40-hex-commit>
      lm_eval_tree: <40-hex-tree>
      lm_eval_version: 0.4.9.2
      python_abi: cpython-312
      base_image_id: sha256:<64-hex-image-id>
      base_image_repo_digest: image/name@sha256:<64-hex-repo-digest>
      inferencex_commit: <40-hex-commit>
      inferencex_tree: <40-hex-tree>
```

The runtime root contains only `lm_eval_runtime_manifest.json` and
`site-packages/`. Magpie validates the exact identity, sorted file manifest,
permissions, link counts, and every file digest on the host. Docker runs mount
the root at `/opt/apex/lm-eval-runtime:ro`; the benchmark helper independently
recomputes the digest, checks the actual Python ABI and `lm_eval` version, and
proves that `lm_eval` imported from that mount. The validated report field
`lm_eval_runtime_receipt` binds the full identity and runtime digest to hashes
of the preserved manifest and receipt. The runtime's InferenceX commit/tree
must match the materialized benchmark checkout. `RUN_EVAL=true` currently
supports local or Docker execution through Magpie's built-in vLLM, SGLang, or
Atom MI300X/MI355X scripts; Ray and native/custom scripts fail closed.
Missing runtime, mutation, ABI/version mismatch, or missing receipt fails the
benchmark. There is no package-manager or network fallback.

The MI355X vLLM script accepts `envs.MODEL_REVISION` as an exact lowercase
40-hex Hugging Face commit. When set, both `hf download` and `vllm serve` are
pinned to it. Magpie then validates `model_revision_receipt.json` and exposes a
bounded `model_revision_receipt` section in the report. A missing, malformed,
or mismatched requested receipt fails the benchmark. Without `MODEL_REVISION`,
the report uses `status: not_requested`; consumers requiring reproducible model
provenance must reject that status.

Magpie never installs its benchmark scripts into the configured InferenceX
checkout. For a Git checkout it exports the exact `HEAD` commit into the
workspace through a private Git index, records the commit and unchanged source
status in `inferencex_runtime_receipt.json`, and modifies only that disposable
tree. A non-Git InferenceX directory uses a compatibility filesystem copy and
is explicitly marked unpinned in the receipt.

Targeted trace selection uses portable symbol glob patterns under
`profiler.targeted_trace.targets`; it does not depend on a fixed container-image
registry. See `Magpie/targeted_trace/README.md` for its artifact contract and
standalone conversion commands.

A typical report shape (abbreviated, with `...` marking elided values):

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

- [Automatic GPU selection in Magpie's benchmark mode](automatic-gpu.md): how Magpie picks idle GPUs before launching and how to override or disable selection
- [Persistent server reuse (local) in Magpie's benchmark mode](persistent-server-reuse.md): keep a server alive across runs to avoid model reload overhead
- [Profiling options in Magpie's benchmark mode](profiling-options.md): configure torch profiler, TraceLens, and gap analysis
- [Analyze and compare kernels with Magpie](../analyze-compare.md): kernel evaluation modes independent of benchmark mode
- [Run Magpie on a Ray cluster](../ray.md): optional remote benchmark scheduling
