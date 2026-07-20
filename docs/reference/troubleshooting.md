---
myst:
    html_meta:
        "description": "Troubleshoot common Magpie issues including GPU memory errors, Docker permission problems, TraceLens installation, and timeout configuration."
        "keywords": "Magpie, troubleshooting, GPU memory error, Docker, TraceLens, timeout, ROCm, benchmark, debug"
---

# Magpie troubleshooting

This topic covers errors and debugging techniques. Each section presents symptoms and their solutions in a table so you can quickly find the issue you're seeing. For benchmark configuration problems not listed here, enable debug logging with the global `--verbose` (`-v`) flag and check the output before filing a bug report.

## Analyze and compare mode

### Common issues

The following errors are frequently reported in analyze and compare mode.

| Error | Solution |
|-------|----------|
| `Analyze mode requires testcase_command` | `testcase_command` is mandatory for analyze mode. Add it to your kernel config under `correctness.testcase_command`. |
| `Testcase command failed (iteration N): <stderr>` | The testcase exited non-zero. Check the quoted stderr for the root cause. Common causes: wrong working directory, missing environment variables, or a path that does not exist on the host. Set `kernel.working_dir` and any required `env` entries in your config. |
| `Compile command N/N failed: <stderr>` | The compile step returned non-zero. The stderr is included in the error. For HIP kernels, verify `gpu_arch` is set in your pipeline config and matches the installed ROCm target (for example, `gfx942`). |
| `hipcc not found. Please install ROCm HIP compiler.` | `hipcc` is not on `PATH`. Install ROCm and ensure `/opt/rocm/bin` is in `PATH`, or set a full path in `compiling_command`. |
| `nvcc not found. Please install CUDA toolkit.` | `nvcc` is not on `PATH`. Install the CUDA toolkit and ensure its `bin` directory is in `PATH`, or set a full path in `compiling_command`. |
| `gpu_arch is not set` | Add `gpu_arch` to your pipeline config (for example, `gpu_arch: gfx942` for MI300X). Run `rocminfo | grep gfx` to find the correct value for your GPU. |
| `PyTorch not available for comparison` | PyTorch is required for `correctness_mode: pytorch`. Install it with `pip install torch` or use `pip install "magpie-eval[all]"`. |

### Debug mode

Enable verbose logging to get more detailed output from analyze and compare runs.

```bash
python -m Magpie analyze --config kernel_config.yaml --verbose
```

## Benchmarking mode

### Common issues

The following errors are frequently reported in benchmark mode.

| Error | Solution |
|-------|----------|
| `ValueError: Free memory on device (...) is less than desired GPU memory utilization` | Reduce `GPU_MEM_UTIL` in config (for example, `0.85`). |
| `docker: permission denied` | Add your user to the docker group or run with sudo. |
| `Required TraceLens inference CLI command(s) not found on PATH` | Applies to `run_mode: local` or classic host post-processing. TraceLens auto-installs on first run. If issues persist, run: `pip install git+https://github.com/AMD-AGI/TraceLens.git`. If `TL_EXTENSION=TraceLens_NDA` is set, install the matching internal extension package. For `run_mode: docker`, commands are resolved from the runtime image. |
| Timeout during model loading | Large models (for example, DeepSeek-R1) might need longer timeouts. Set `timeout_seconds: 7200` in your benchmark config. |
| `gpu_selection.auto failed: ...` | Not enough idle GPUs on the host. Free a GPU, lower `gpu_selection.min_free_memory_gb`, narrow `gpu_selection.candidates`, or pin manually using `envs.ROCR_VISIBLE_DEVICES` (AMD) or `envs.CUDA_VISIBLE_DEVICES` (NVIDIA). See [Automatic GPU selection in Magpie's benchmark mode](../how-to/benchmarking/automatic-gpu.md). |

### Debug mode

Enable verbose logging to get more detailed output from the benchmark run.

```bash
python -m Magpie benchmark --benchmark-config config.yaml --verbose
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