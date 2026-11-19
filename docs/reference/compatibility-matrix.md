---
myst:
    html_meta:
        "description": "Verified hardware and software compatibility for Magpie, including Python versions, profilers, GPU hardware, and supported inference frameworks."
        "keywords": "Magpie, compatibility matrix, AMD Instinct, Python, vLLM, SGLang, rocprof-compute, GPU requirements"
---

# Magpie compatibility matrix

This topic lists the known version requirements for Magpie. It covers hardware
and software requirements and captures only versions that have been verified and
tested.

| GPU                              | ROCm                                 | Python    | Ubuntu       | PyYAML | NumPy     | MCP      |
| -------------------------------- | ------------------------------------ | --------- | ------------ | ------ | --------- | -------- |
| MI300X,<br>MI325X,<br>MI355X     | 7.0.x,<br>7.1.x,<br>7.2.x,<br>10.0.0 | 3.10–3.13 | 22.04, 24.04 | >= 6.0 | >= 1.24.0 | >= 1.0.0 |
| Radeon 8060S (`gfx1151`, bounded vLLM/SGLang runner qualification) | 10.0.0 | 3.14 | 24.04 | >= 6.0 | >= 1.24.0 | >= 1.0.0 |

- GPUs are verified using the bundled benchmark scripts in `Magpie/scripts/benchmark/`.
- Python versions are declared in `pyproject.toml`; the minimum is 3.10.
- MCP (`mcp`) is required only for the MCP server.
- IntelliKit integrations are optional (pinned commit `2f61453a779980b00504ea3b772ff4a1a1c3f4ad`). 
- The Radeon 8060S row records bounded runner/serving/evaluation qualification of custom ROCm 10 builds (vLLM `0.27.0+hyperloom.gfx1151.rocm10`, SGLang `0.5.15`), not every profiler or precision route. It does not describe the software versions in the default published vLLM image.
- The default gfx1151 vLLM image is `vllm/vllm-openai-rocm:v0.23.0`, separately [tested on a Radeon 8060S by a collaborator](https://github.com/AMD-AGI/Magpie/pull/90#issuecomment-5600409716). SGLang requires an explicit compatible custom image or local installation; no default upstream gfx1151 image is mapped.
- The Radeon 8060S SGLang route is qualified with Triton attention, CUDA graphs disabled, and AITER disabled. The vLLM route also pins AITER off.

## Profilers and optional tools

The following profiling tools and optional packages extend Magpie's capabilities.

| Tool | Version | Notes |
| --- | --- | --- |
| `rocprof-compute` | >= 3.2.3 | AMD performance profiling. See the [install guide](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/install/core-install.html). |
| IntelliKit Metrix | `2f61453a779980b00504ea3b772ff4a1a1c3f4ad` | Optional AMD profiling integration. Install with `.[metrix]` or `.[intellikit]`. |
| IntelliKit Accordo | `2f61453a779980b00504ea3b772ff4a1a1c3f4ad` | Optional AMD correctness integration. Install with `.[accordo]` or `.[intellikit]`. |

```{note}
The IntelliKit integrations (Accordo and Metrix) are optional/experimental. Magpie primarily relies on ROCm libraries.
```

## Framework benchmark compatibility

Benchmark mode runs against external inference frameworks with the following
supported model precisions.

| Framework | Supported precisions | Notes |
| --- | --- | --- |
| vLLM | fp8, fp16, bf16, fp4 | Default precision is fp8. |
| SGLang | fp8, fp16, bf16, fp4 | |
| Atom | fp8, fp16, bf16, fp4 | Single-node v1. |


## InferenceX client interfaces

| Revision | Interface | Qualification |
| --- | --- | --- |
| `408c015be4b22d14c69518643609669405507077` | Existing packaged client with shell benchmark library and AIPerf submodule | Default pin retained; legacy regression coverage. |
| `404c7e1ad2350dce1db7aa99acb5cd87a583a38c` | Python `infx.bench` clients and published AgentX harness | Opt-in single-node adapter; real-upstream contract/HTTP checks and bounded MI355X SGLang GPU qualification below. |

On 2026-11-18, a Crusoe MI355X passed single-GPU Qwen3-8B BF16
fixed-sequence runs with and without the torch profiler (20/20 requests in each
run). The serving image was
`lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261112`
(SGLang `0.5.23.dev20261112+gcabb43afd5`, PyTorch `2.11.0+rocm10.0.0`).
The profiled run used Triton attention and produced prefill, decode and merged
traces containing GPU events. CUDA graphs were disabled. This is a functional
qualification, not a performance comparison or qualification of every framework,
precision, profiler or GPU topology. The previous default pin is retained.

The same node also completed the official 1,200-second AgentX fast replay with
the local 27B BF16 Qwen3-family checkpoint, TP=1, concurrency=1 and a
262,144-token context limit. All 71 measured requests passed upstream validation
with zero request errors (two warmup records excluded). The isolated client used Python 3.11.18
and `agentx-harness==1.0.14`. The report is explicitly non-publishable in fast
mode. This run used a Magpie-managed server; externally managed server lifecycle
boundaries have HTTP contract coverage. AgentX torch profiling, canonical replay,
power telemetry and vLLM GPU execution were not qualified by this run.

See [modern single-node configuration](benchmark-config.md#modern-inferencex-single-node-clients)
for version selection, runtime isolation and external service ownership.
