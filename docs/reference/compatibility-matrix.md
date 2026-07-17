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

| AMD Instinct GPU | ROCm version | Python | OS | PyYAML | NumPy | MCP (`mcp`) | IntelliKit integrations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MI300X, MI355X | >=7.0.0 | 3.10, 3.11, 3.12, 3.13 | Linux | >= 6.0 | >= 1.24.0 | >= 1.0.0 | `3f45cd314d455b652a1246678511b40547fe521e`|

- GPUs are verified using the bundled benchmark scripts in `Magpie/scripts/benchmark/`.
- Python versions are declared in `pyproject.toml`; the minimum is 3.10.
- MCP (`mcp`) is only required for the MCP server.
- IntelliKit operations are optional. Install with `.[intellikit]` or individual extras. Some components require ROCm/HIP build tools.

## Profilers and optional tools

The following profiling tools and optional packages extend Magpie's capabilities.

| Tool | Version| Notes |
| --- | --- | --- |
| `rocprof-compute` | >= 3.40 | AMD performance profiling. See the [install guide](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/install/core-install.html). |
| IntelliKit Metrix | `3f45cd314d455b652a1246678511b40547fe521e` | Optional AMD profiling integration. Install with `.[metrix]` or `.[intellikit]`. |
| IntelliKit Accordo | `3f45cd314d455b652a1246678511b40547fe521e` | Optional AMD correctness integration. Install with `.[accordo]` or `.[intellikit]`. |

## Framework benchmark compatibility

Benchmark mode runs against external inference frameworks with the following
supported model precisions.

| Framework | Supported precisions | Notes |
| --- | --- | --- |
| vLLM | fp8, fp16, bf16, fp4 | Default precision is fp8. |
| SGLang | fp8, fp16, bf16, fp4 | |
| Atom | fp8, fp16, bf16, fp4 | Single-node v1. |

