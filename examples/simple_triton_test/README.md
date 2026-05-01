# Triton 0-overhead Latency Examples

Self-contained Triton vector-add example exercising the new `Latency`
stage end-to-end. See [`docs/latency.md`](../../docs/latency.md) for the
full design.

## Files

| File | Purpose |
| --- | --- |
| `triton_vector_add.py` | Triton kernel + 2 BLOCK_SIZE variants + `get_inputs` factory + `--check` / `--bench` CLI. |
| `analyze_triton_latency.yaml` | Single-kernel analyze with import-based `bench_target` and `method: auto`. |
| `compare_triton_blocksize.yaml` | Compare 2 BLOCK_SIZE configs ranking by `kernel_median_ms` (the autotuning recipe). |

## Setup

```bash
pip install triton torch
# rocprofv3 needed for the kernel-only half of method=auto / method=both
```

## Run

```bash
cd /path/to/Magpie

# Single-kernel analyze (wall-clock + kernel-only)
python -m Magpie analyze -k examples/simple_triton_test/analyze_triton_latency.yaml

# Compare two BLOCK_SIZE configs — primary_metric: kernel_median_ms
python -m Magpie compare -k examples/simple_triton_test/compare_triton_blocksize.yaml
```

## What you should see

`results/<run>/analyze_report.json` will contain a top-level summary
block per kernel:

```json
{
  "summary": [
    {
      "kernel_id": "triton_vector_add",
      "latency_state": "SUCCESS",
      "latency": {
        "method": "both",
        "primary_metric": "kernel_median_ms",
        "primary_value_ms": 0.012,
        "wall_median_ms":   0.044,
        "kernel_median_ms": 0.012,
        "dispatch_overhead_us": 32.0,
        "crosscheck_vs_rocprof_ratio": 3.67
      }
    }
  ]
}
```

(`dispatch_overhead_us` and the high `crosscheck_vs_rocprof_ratio`
illustrate exactly why we need `kernel_median_ms` for autotuning small
kernels — wall-clock is dominated by the ~30 µs dispatch path.)

## Without Magpie (sanity / dev loop)

```bash
# Correctness only
python examples/simple_triton_test/triton_vector_add.py --check

# User-harness benchmark (prints MAGPIE_LATENCY_JSON line directly)
python examples/simple_triton_test/triton_vector_add.py --bench
```
