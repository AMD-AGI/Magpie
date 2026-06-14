# Release notes

This page summarizes the features included in each Magpie release. For the
initial release, the notes provide an overview of all features available in the
tool.

## Magpie 0.1.0 (initial release)

The initial public release of Magpie establishes a lightweight, general-purpose
framework for evaluating GPU kernel correctness and performance across AMD and
NVIDIA hardware.

### Evaluation modes

- **Analyze**: Single-kernel evaluation against a testcase, with optional
  performance profiling.
- **Compare**: Multi-kernel comparison and ranking against a configurable
  baseline to identify the fastest correct implementation.
- **Benchmark**: Framework-level benchmarking for vLLM, SGLang, and Atom, with
  optional torch profiler and system profiler runs.

### Hardware and execution

- Support for AMD (HIP/ROCm) and NVIDIA (CUDA) GPUs.
- Three execution environments: local host, sandboxed container, and remote
  Ray cluster.
- Hardware-aware evaluation with optional GPU power and frequency control.
- Automatic idle-GPU selection in Benchmark mode for both AMD and NVIDIA
  devices.

### Kernel types

- HIP, CUDA, PyTorch, and Triton kernels.

### Profiling and trace analysis

- Pluggable performance profiler backends: `rocprof-compute` (AMD), `ncu`
  (NVIDIA), and IntelliKit Metrix.
- Optional correctness validation via testcase or IntelliKit Accordo.
- TraceLens integration for performance trace analysis.
- Kernel-level gap analysis from torch profiler traces, including a standalone
  mode that runs on existing traces.
- Kernel source finder to locate kernel source files and test commands for AMD
  kernels.

### Integration

- Model Context Protocol (MCP) server exposing Magpie capabilities to AI
  agents.
- Agent skill packaging for environments without MCP.
- Structured JSON reports for pipeline integration.

### Configuration

- Framework-level configuration via `config.yaml`.
- Per-evaluation kernel configuration files for analyze and compare modes.
- Benchmark configuration files for framework benchmarks.

```{note}
Update this page with each new Magpie release, listing added, changed, and
fixed features for that version.
```
