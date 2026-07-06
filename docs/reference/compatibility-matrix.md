---
myst:
    html_meta:
        "description": "Verified hardware and software compatibility for Magpie, including Python versions, ROCm and CUDA toolchains, profilers, GPU hardware, and supported inference frameworks."
        "keywords": "Magpie, compatibility matrix, ROCm, CUDA, AMD Instinct, Python, vLLM, SGLang, rocprof-compute, GPU requirements"
---

# Compatibility matrix

This topic lists the known version requirements for Magpie. It covers hardware
and software requirements and is intended to capture only versions that have
been verified and tested.

```{note}
The entries marked **TODO (verify)** must be confirmed by the Magpie team
against tested configurations before publication. Only verified versions should
remain in this table.
```

## Software requirements

The following Python packages and operating systems are required or tested with Magpie.

| Component | Supported / tested versions | Notes |
| --- | --- | --- |
| Python | 3.10, 3.11, 3.12, 3.13 | Declared in `pyproject.toml`. Minimum is 3.10. |
| Operating system | Linux | TODO (verify): list tested distributions and versions. |
| PyYAML | >= 6.0 | Core dependency. |
| NumPy | >= 1.24.0 | Core dependency. |
| MCP (`mcp`) | >= 1.0.0 | Required only for the MCP server. |

## GPU toolchains

The following GPU compute toolchains are supported for kernel compilation and profiling.

| Toolchain | Supported / tested versions | Notes |
| --- | --- | --- |
| AMD ROCm (HIP) | TODO (verify) | Required for HIP kernel compilation and profiling on AMD GPUs. |
| NVIDIA CUDA | TODO (verify) | Required for CUDA kernel compilation and profiling on NVIDIA GPUs. |

## Profilers and optional tools

The following profiling tools and optional packages extend Magpie's capabilities.

| Tool | Supported / tested versions | Notes |
| --- | --- | --- |
| `rocprof-compute` | >= 3.40 | AMD performance profiling. See the [install guide](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/install/core-install.html). |
| `ncu` (Nsight Compute) | TODO (verify) | NVIDIA performance profiling. |
| IntelliKit Metrix | TODO (verify) | Optional human-readable AMD profiling metrics. |
| IntelliKit Accordo | TODO (verify) | Optional AMD kernel correctness validation. |
| Docker | TODO (verify) | Required for Benchmark mode when `run_mode` is `docker`. |
| Ray | TODO (verify) | Required for the Ray execution environment. |

## Hardware

Magpie has been tested on the following GPU hardware.

| Hardware | Supported / tested | Notes |
| --- | --- | --- |
| AMD Instinct™ GPUs | MI300X, MI355X | Verified via the bundled benchmark scripts in `Magpie/scripts/benchmark/`. |
| NVIDIA GPUs | TODO (verify) | List verified architectures. |

## Framework benchmark compatibility

Benchmark mode runs against external inference frameworks. List the verified
versions of each framework and the supported model precisions.

| Framework | Tested versions | Supported precisions | Notes |
| --- | --- | --- | --- |
| vLLM | TODO (verify) | fp8, fp16, bf16, fp4 | Default precision is fp8. |
| SGLang | TODO (verify) | fp8, fp16, bf16, fp4 | |
| Atom | TODO (verify) | fp8, fp16, bf16, fp4 | Single-node v1. |

```{note}
As a Hyperloom component, the Hyperloom compatibility matrix should record which
version of Magpie is captured in each Hyperloom release.
```
