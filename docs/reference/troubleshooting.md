---
myst:
    html_meta:
        "description": "Troubleshoot common Magpie issues including GPU memory errors, Docker permission problems, TraceLens installation, and timeout configuration."
        "keywords": "Magpie, troubleshooting, GPU memory error, Docker, TraceLens, timeout, ROCm, benchmark, debug"
---

# Magpie troubleshooting

This topic covers errors and debugging techniques. Each section presents symptoms and their solutions in a table so you can quickly find the issue you're seeing. For benchmark configuration problems not listed here, enable verbose logging with `--log-level DEBUG` and check the output before filing a bug report.

## Benchmarking mode

### Common issues

The following errors are frequently reported in benchmark mode.

| Error | Solution |
|-------|----------|
| `ValueError: Free memory on device (...) is less than desired GPU memory utilization` | Reduce `GPU_MEM_UTIL` in config (for example, `0.85`). |
| `docker: permission denied` | Add your user to the docker group or run with sudo. |
| `Required TraceLens inference CLI command(s) not found on PATH` | Applies to `run_mode: local` or classic host post-processing. TraceLens auto-installs on first run. If issues persist, run: `pip install git+https://github.com/AMD-AIG-AIMA/TraceLens.git`. If `TL_EXTENSION=TraceLens_NDA` is set, install the matching internal extension package. For `run_mode: docker`, commands are resolved from the runtime image. |
| Timeout during model loading | Large models (for example, DeepSeek-R1) might need longer timeouts. Set `timeout_seconds: 7200` in your benchmark config. |
| `gpu_selection.auto failed: ...` | Not enough idle GPUs on the host. Free a GPU, lower `gpu_selection.min_free_memory_gb`, narrow `gpu_selection.candidates`, or pin manually via `envs.ROCR_VISIBLE_DEVICES` (AMD) / `envs.CUDA_VISIBLE_DEVICES` (NVIDIA). See [Automatic GPU selection in Magpie's benchmark mode](../how-to/benchmarking/automatic-gpu.md). |

### Debug mode

Enable verbose logging to get more detailed output from the benchmark run.

```bash
python -m Magpie benchmark --benchmark-config config.yaml --log-level DEBUG
```

## Ray on Magpie

| Error | Solution |
|---------|------------------|
| `ray.init` fails | Firewall, wrong address, Ray version mismatch; try `ray://host:10001` from remote drivers. |
| `No GPU node found in the Ray cluster` | Workers not started with GPUs; head-only cluster; GPU resources zero in `ray.nodes()`. |
| Analyze fails on worker: missing sources | `${CK_HOME}` or paths not on worker or NFS; build artifacts not present on worker. |
| Worker import errors for Magpie | Set `install_magpie: true` or bake Magpie into the worker image; check `runtime_env` pip logs. |
| Benchmark TP / Ray backend wrong | Inspect `_configure_tp_isolation` logs; set `EXTRA_VLLM_ARGS` / `EXTRA_SGLANG_ARGS` explicitly. |
| Empty GPU visibility in child | Should be fixed by `_clear_hidden_gpus`; if not, inspect env in InferenceX subprocess. |