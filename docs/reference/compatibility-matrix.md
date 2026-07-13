---
myst:
    html_meta:
        "description": "Verified hardware and software compatibility for Magpie, including Python versions, profilers, GPU hardware, and supported inference frameworks."
        "keywords": "Magpie, compatibility matrix, AMD Instinct, Python, vLLM, SGLang, rocprof-compute, GPU requirements"
---

# Compatibility matrix

This topic lists the known version requirements for Magpie. It covers hardware
and software requirements and captures only versions that have been verified and
tested.

## Software requirements

The following Python packages and operating systems are required or tested with Magpie.

| Component | Supported / tested versions | Notes |
| --- | --- | --- |
| Python | 3.10, 3.11, 3.12, 3.13 | Declared in `pyproject.toml`. Minimum is 3.10. |
| Operating system | Linux | Linux only. |
| PyYAML | >= 6.0 | Core dependency. |
| NumPy | >= 1.24.0 | Core dependency. |
| MCP (`mcp`) | >= 1.0.0 | Required only for the MCP server. |
| IntelliKit integrations | `3f45cd314d455b652a1246678511b40547fe521e` | Optional. Install with `.[intellikit]` or individual extras. Some components require ROCm/HIP build tools. |

## Profilers and optional tools

The following profiling tools and optional packages extend Magpie's capabilities.

| Tool | Supported / tested versions | Notes |
| --- | --- | --- |
| `rocprof-compute` | >= 3.40 | AMD performance profiling. See the [install guide](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/install/core-install.html). |
| IntelliKit Metrix | `3f45cd314d455b652a1246678511b40547fe521e` | Optional AMD profiling integration. Install with `.[metrix]` or `.[intellikit]`. |
| IntelliKit Accordo | `3f45cd314d455b652a1246678511b40547fe521e` | Optional AMD correctness integration. Install with `.[accordo]` or `.[intellikit]`. |

## Hardware

Magpie has been tested on the following GPU hardware.

| Hardware | Supported / tested | Notes |
| --- | --- | --- |
| AMD Instinct™ GPUs | MI300X, MI355X | Verified using the bundled benchmark scripts in `Magpie/scripts/benchmark/`. |

## Framework benchmark compatibility

Benchmark mode runs against external inference frameworks with the following
supported model precisions.

| Framework | Supported precisions | Notes |
| --- | --- | --- |
| vLLM | fp8, fp16, bf16, fp4 | Default precision is fp8. |
| SGLang | fp8, fp16, bf16, fp4 | |
| Atom | fp8, fp16, bf16, fp4 | Single-node v1. |

```{note}
As a Hyperloom component, the Hyperloom compatibility matrix should record which
version of Magpie is captured in each Hyperloom release.
```
