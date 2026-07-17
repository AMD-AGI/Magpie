---
myst:
    html_meta:
        "description": "Step-by-step Magpie examples for analyzing HIP kernels, comparing implementations, benchmarking vLLM with TraceLens, and running standalone gap analysis on GPU traces."
        "keywords": "Magpie, examples, HIP kernel, compare kernels, vLLM benchmark, TraceLens, gap analysis, ROCm, CUDA, GPU"
---

# Magpie examples

This topic provides end-to-end, step-by-step examples for common Magpie use
cases. Each example lists the prerequisites, the exact commands to run, and the
expected output. All example configuration files referenced here live in the
[`examples/`](https://github.com/AMD-AGI/Magpie/tree/main/examples) directory of
the Magpie repository.

Run every command from the Magpie repository root unless noted otherwise.

## Analyze a simple HIP kernel

This example analyzes a minimal HIP `vector_add` kernel for correctness using a
testcase command.

**Prerequisites**

- A working AMD ROCm/HIP toolchain (`hipcc` on your `PATH`).
- Magpie installed (see [Install Magpie](../install/install.md)).

**Steps**

1. Build the sample kernel binary:

   ```bash
   cd examples/simple_hip_test
   hipcc -g -O2 -o vector_add vector_add.hip
   cd ../..
   ```

2. Run the analysis, skipping performance profiling:

   ```bash
   python -m Magpie analyze \
     --kernel-config examples/simple_hip_test/analyze_default.yaml \
     --no-perf
   ```

   The config defines a single kernel and its testcase:

   ```yaml
   kernel:
     id: "vector_add"
     type: hip
     source_files:
       - "./vector_add.hip"
     testcase_command: "./vector_add"
     working_dir: "examples/simple_hip_test"
   ```

**Expected output**

Magpie reports a passing correctness state and writes a JSON report to
`./results`. The summary indicates the kernel compiled and the testcase passed,
with an overall `score` of `1.0` when correctness succeeds and profiling is
skipped.

## Compare two kernel implementations

This example compares BF16 and FP16 grouped GEMM kernels from Composable Kernel
and ranks them by performance.

**Prerequisites**

- AMD ROCm/HIP toolchain and a built
  [Composable Kernel](https://github.com/ROCm/composable_kernel) checkout.
- `rocprof-compute` installed if you want performance metrics.

**Steps**

1. Clone and point Magpie at Composable Kernel:

   ```bash
   git clone https://github.com/ROCm/composable_kernel.git
   export CK_HOME=/path/to/composable_kernel
   ```

2. Build the required example targets (see the comments in the config), then run
   the comparison:

   ```bash
   python -m Magpie compare \
     --kernel-config examples/ck_grouped_gemm_compare.yaml
   ```

   The config lists the two kernels to compare:

   ```yaml
   kernels:
     - id: "grouped_gemm_xdl_bf16"
       type: hip
       source_files:
         - "${CK_HOME}/example/15_grouped_gemm/grouped_gemm_xdl_bf16.cpp"
       working_dir: "${CK_HOME}/build"
       testcase_command: "bin/example_grouped_gemm_xdl_bf16"
     - id: "grouped_gemm_xdl_fp16"
       type: hip
       source_files:
         - "${CK_HOME}/example/15_grouped_gemm/grouped_gemm_xdl_fp16.cpp"
       working_dir: "${CK_HOME}/build"
       testcase_command: "bin/example_grouped_gemm_xdl_fp16"
   ```

**Expected output**

Magpie evaluates both kernels, prints a ranked comparison against the baseline
(index `0` by default), and writes a JSON report identifying the winning
implementation. See [Analyze and compare kernels](../how-to/analyze-compare.md)
for how scores and rankings are computed.

## Benchmark vLLM with TraceLens analysis

This example runs a framework-level benchmark of vLLM and analyzes the resulting
traces.

**Prerequisites**

- Docker (for the default `docker` run mode), or run with `--run-mode local`
  inside a prepared container.
- Access to the model referenced by the benchmark config.

**Steps**

1. Run the benchmark from a config file:

   ```bash
   python -m Magpie benchmark \
     --benchmark-config examples/benchmarks/benchmark_vllm_dsr1.yaml
   ```

2. To include TraceLens trace analysis, use the TraceLens example config:

   ```bash
   python -m Magpie benchmark \
     --benchmark-config examples/benchmarks/benchmark_vllm_tracelens.yaml
   ```

**Expected output**

Magpie launches the benchmark, collects throughput and latency metrics, and (for
the TraceLens config) produces a trace analysis report under the benchmark
workspace in `./results`. In TraceLens inference mode, the full per-stage CSVs
are written under `tracelens/prefilldecode/`, `tracelens/decode_only/`, and
`tracelens/prefill_only/` when those stages are available. Magpie also writes
compact review files such as `tracelens/decode_only_kernel_roofline_simple.csv`
and `tracelens/prefilldecode_kernel_roofline_simple.csv`; sort them by
`kernel_time_ms_sum` or `time_pct` to find the dominant operations quickly.
See [Benchmark frameworks with Magpie](../how-to/benchmarking/benchmark.md) for
the full result layout and metric descriptions.

## Standalone gap analysis on existing traces

If you already have torch profiler traces, you can run gap analysis without
launching a benchmark to find the kernels that dominate runtime.

**Steps**

```bash
python -m Magpie benchmark \
  --trace-dir /path/to/torch_trace \
  --top-k 20
```

**Expected output**

Magpie writes a `gap_analysis/gap_analysis.csv` file (plus optional per-rank
CSVs) under the trace directory, listing the top bottleneck kernels by
aggregated duration. Add `--find-kernel-sources` to also locate kernel source
files and test commands for AMD kernels; see
[Find kernel sources with Magpie](../how-to/kernel-source-finder.md).

## Additional benchmark configurations

The [`examples/benchmarks/`](https://github.com/AMD-AGI/Magpie/tree/main/examples/benchmarks)
directory contains many additional ready-to-run benchmark configurations,
including SGLang, Atom, Ray-based runs, and a range of models and precisions.
