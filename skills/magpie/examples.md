# Magpie examples

Run these commands from the Magpie repository root. Replace paths, models, images, and testcase commands with values for the target environment.

## Inspect the environment

```bash
magpie --gpu-info
magpie --help
magpie benchmark --help
```

## Analyze one kernel

HIP:

```bash
magpie analyze ./kernels/matmul.hip \
  --type hip \
  --testcase "./tests/run_matmul.sh" \
  --output-dir ./results
```

Triton:

```bash
magpie analyze ./kernels/matmul.py \
  --type triton \
  --testcase "python ./tests/test_matmul.py" \
  --output-dir ./results
```

Use a config for repeatable runs:

```bash
magpie analyze --kernel-config ./configs/kernel.yaml
```

## Compare optimized variants

```bash
magpie compare \
  ./kernels/baseline.hip \
  ./kernels/candidate_a.hip \
  ./kernels/candidate_b.hip \
  --type hip \
  --testcase "./tests/run_correctness.sh" \
  --baseline 0 \
  --output-dir ./results
```

For candidates with different compile commands, environments, or working directories, put each entry in a `kernels:` YAML list and run:

```bash
magpie compare --kernel-config ./configs/compare.yaml --baseline 0
```

## Benchmark an inference workload

Use a repository config when available:

```bash
magpie benchmark \
  --benchmark-config ./examples/benchmarks/benchmark_vllm_dsr1.yaml \
  --output-dir ./results
```

Quick local override:

```bash
magpie benchmark vllm \
  --model <model-name-or-path> \
  --precision fp8 \
  --tp 8 \
  --concurrency 32 \
  --input-len 1024 \
  --output-len 512 \
  --run-mode local \
  --output-dir ./results
```

Create a separate profiled run instead of using it as the clean baseline:

```yaml
# ./configs/profiled-benchmark.yaml
benchmark:
  # Keep framework, model, precision, envs, and workload settings aligned with
  # the clean baseline.
  profiler:
    torch_profiler:
      enabled: true
    tracelens:
      enabled: true
      analysis_mode: inference
      analysis_stages: all
      export_format: csv
```

```bash
magpie benchmark \
  --benchmark-config ./configs/profiled-benchmark.yaml \
  --output-dir ./profiled-results
```

Review TraceLens post-processing before selecting a concrete kernel:

```bash
find ./profiled-results/<benchmark-workspace>/tracelens \
  -name '*_kernel_roofline_simple.csv' -print
```

Start with the stage rows that dominate `kernel_time_ms_sum` or `time_pct`, use the roofline columns to form a compute/memory hypothesis, and then use gap analysis to identify and map the underlying kernels.

## Find bottlenecks and kernel source

```bash
magpie benchmark \
  --trace-dir ./profiled-results/<benchmark-workspace>/torch_trace \
  --start-pct 20 \
  --end-pct 80 \
  --top-k 20 \
  --find-kernel-sources \
  --kernel-source-repos ./rocm-libraries ./vllm
```

Omit `--find-kernel-sources` when only timing aggregation is needed. Use `--no-rank-csv` only when per-rank imbalance is irrelevant.

## End-to-end optimization sequence

```bash
# 1. Clean baseline
magpie benchmark --benchmark-config ./configs/baseline.yaml --output-dir ./baseline-results

# 2. Profile the same workload and run TraceLens post-processing
magpie benchmark --benchmark-config ./configs/profiled.yaml --output-dir ./profiled-results

# 3. Review per-stage TraceLens roofline summaries
find ./profiled-results/<benchmark-workspace>/tracelens \
  -name '*_kernel_roofline_simple.csv' -print

# 4. Find top kernels and source candidates
magpie benchmark \
  --trace-dir ./profiled-results/<benchmark-workspace>/torch_trace \
  --top-k 20 \
  --find-kernel-sources \
  --kernel-source-repos ./rocm-libraries ./vllm

# 5. Evaluate and rank isolated candidates
magpie analyze --kernel-config ./configs/candidate.yaml --output-dir ./kernel-results
magpie compare --kernel-config ./configs/compare.yaml --baseline 0 --output-dir ./kernel-results

# 6. Re-run the clean benchmark with the selected implementation
magpie benchmark --benchmark-config ./configs/optimized.yaml --output-dir ./optimized-results
```

Keep `baseline.yaml` and `optimized.yaml` equivalent except for the selected implementation or build. Compare `benchmark_report.json` files and report correctness status, kernel-level change, end-to-end change, and run-to-run variance.
