---
myst:
    html_meta:
        "description": "Learn how Magpie automatically selects idle GPUs before launching a benchmark, and how to override GPU selection manually or disable it for Ray and server-reuse runs."
        "keywords": "Magpie, GPU selection, ROCR_VISIBLE_DEVICES, CUDA_VISIBLE_DEVICES, benchmark, ROCm, AMD Instinct, idle GPU, gpu_selection"
---

# Automatic GPU selection in Magpie's benchmark mode

Before launching the benchmark, Magpie scans the host (`rocm-smi` / `nvidia-smi`),
picks the least-busy GPU(s) with enough free VRAM, and pins the run via
vendor-specific environment variables. 

```{note}
For a full overview of benchmark mode and how GPU selection fits into the broader pipeline, see [Benchmark frameworks with Magpie](benchmark.md).
```

- **AMD**: `ROCR_VISIBLE_DEVICES=<ids>` (the launcher script remaps
  `HIP_VISIBLE_DEVICES` to the post-filter logical range `0..N-1`).
- **NVIDIA**: `CUDA_VISIBLE_DEVICES=<ids>` + `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

GPU IDs use the same index space as `rocm-smi` / `nvidia-smi`. By default the
selector asks for `envs.TP` idle GPUs; override with `gpu_selection.count`.

## Configuration

All `gpu_selection` fields are optional; the block is enabled by default.

| Field | Default | Description |
|-------|---------|-------------|
| `auto` | `true` | Set `false` to disable and let the framework see every GPU |
| `min_free_memory_gb` | `8.0` | Reject GPUs with less free VRAM than this |
| `count` | `null` | Number of GPUs to pin; `null` → `envs.TP` |
| `candidates` | `null` | Optional allowlist of physical GPU IDs to consider |

## Manual override

Setting `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` /
`ROCR_VISIBLE_DEVICES` in `envs:` pins to specific cards and skips auto-selection:

```yaml
  envs:
    TP: 1
    ROCR_VISIBLE_DEVICES: "3"    # AMD: pin to rocm-smi GPU[3]
    # CUDA_VISIBLE_DEVICES: "3"  # NVIDIA alternative
    # CUDA_DEVICE_ORDER: PCI_BUS_ID
```

## Ray mode

In Ray mode (`run_mode: ray`), `gpu_selection` is ignored — Ray schedules
devices itself via `num_gpus`. To restrict the cluster to specific cards,
export `ROCR_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` in the shell before
starting `ray start`.

## Server lifecycle reuse

When `server_lifecycle.enabled: true` and the run attaches to an existing
healthy server matching reuse metadata (`force_reuse` skips mismatch checks),
`gpu_selection.auto` is skipped entirely for that invocation so pinned devices
do not churn between chain runs. See
[Persistent server reuse](persistent-server-reuse.md) for details.
