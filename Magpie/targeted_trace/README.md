# TargetedKernelTrace

`Magpie.targeted_trace` is Magpie's versioned acquisition contract for kernel
evidence needed by optimization agents. It is deliberately independent of Apex,
TraceLens internals, container tags, and image-specific source registries.

## Ownership and run separation

- Magpie owns acquisition, integrity receipts, deterministic sampling, and loss
  accounting.
- TraceLens remains the authoritative system/GPU analysis layer.
- Consumers such as Apex read `manifest.json` plus checksummed shard artifacts.
- Targeted traces are diagnostic artifacts and always carry
  `reward_eligible: false`. A benchmark explicitly declared as `measurement`
  rejects Torch profiler, TraceLens, system profiler, gap analysis, and targeted
  tracing.

## Artifact contract

Each `shards/trace_pid<PID>_rank<RANK>.jsonl` contains:

1. A header with run/rank/PID, sampling seed/rate, and finite record budget.
2. Typed event envelopes with a monotonic sequence and chained SHA-256 checksum.
3. An end sentinel with `seen`, `sampled`, `written`, `dropped`, and
   `dropped_by_reason` counters.

`manifest.json` records schema/version, targets, provenance, aggregate coverage,
and per-shard file/checksum receipts. Unsupported schema versions fail fast.
Postprocessing reads one JSONL line at a time and reports corrupt or missing tails;
it never silently skips them.

## Evidence fidelity

The explicit `TargetedTraceRecorder` API captures Python-visible Triton and HIP
wrapper calls: launch source/hash, Python grid, tensor shape/dtype/stride, named
scalars, constexpr values, and meta parameters. It does not read tensor contents or
store raw data pointers.

The Torch profiler adapter contributes runtime symbol/grid/block/stream/duration,
rank/stage/graph context, correlation IDs, and any tensor metadata present in the
trace. Missing fields remain null/empty with warnings; symbol/count/order is not
treated as a globally stable CPU-to-GPU join.

Sampling uses only `{run_seed, stable_event_key}` through SHA-256. Python's
process-randomized `hash()` and mutable PRNG state are not used.

## Benchmark configuration

Target selection uses portable glob patterns rather than a hardcoded image map:

```yaml
benchmark:
  run_kind: diagnostic
  profiler:
    torch_profiler:
      enabled: true
    targeted_trace:
      enabled: true
      backend: torch_profiler
      run_seed: qwen-diagnostic-1
      sample_rate: 0.1
      max_records_per_shard: 10000
      targets:
        - target_id: aiter.fused_moe
          name_patterns:
            - "*fused_moe*"
          package: aiter
```

The benchmark workspace receives `targeted_trace/manifest.json`, shards, and
`summary.json`; `benchmark_report.json` contains the bounded summary and manifest
path.

## Standalone CLI

Convert existing Torch profiler traces:

```bash
magpie targeted-trace adapt-torch \
  --trace-dir ./torch_trace \
  --target-config targets.yaml \
  --output-dir ./targeted_trace \
  --run-id diagnostic-001 \
  --framework vllm
```

Validate and aggregate artifacts:

```bash
magpie targeted-trace postprocess \
  --trace-dir ./targeted_trace \
  --output ./targeted_trace/summary.json \
  --strict
```

## Runtime probes

Framework integration code can use the generic API at a known launch boundary:

```python
from Magpie.targeted_trace import TargetedTraceRecorder

with TargetedTraceRecorder(
    "/workspace/targeted_trace",
    run_id="diagnostic-001",
    run_seed="qwen-diagnostic-1",
    framework="vllm",
    rank=0,
) as trace:
    trace.record_triton_launch(
        target_id="aiter.fused_moe",
        kernel_name="fused_moe_kernel",
        args=(x, weights),
        positional_names=("x", "weights"),
        kwargs={"BLOCK_SIZE": 256},
        constexpr_names=("BLOCK_SIZE",),
        grid=(128, 1, 1),
        source_path=__file__,
        source_line=42,
    )
```

This module intentionally does not ship an image registry or mutate framework
packages. Source discovery and temporary instrumentation are separate adapters;
the durable evidence format remains generic.
