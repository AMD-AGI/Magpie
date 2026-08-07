---
myst:
    html_meta:
        "description": "Release notes for Magpie, a GPU kernel evaluation framework. Lists features added in each release, including evaluation modes, profiling backends, and MCP integration."
        "keywords": "Magpie, release notes, changelog, GPU kernel, ROCm, HIP, CUDA, vLLM, SGLang, TraceLens, MCP server"
---

# Magpie release notes

This topic summarizes the features available in each Magpie release. For the hardware and software versions validated for a release, see the [Compatibility matrix](compatibility-matrix.md).

## Magpie 0.2.0

Released July 21, 2026, Magpie 0.2.0 improves TraceLens inference analysis and
refreshes the documentation and compatibility guidance introduced in the
initial beta.

### Release highlights

- TraceLens inference runs now produce compact, per-stage simple roofline CSV
  summaries for mixed prefill/decode, prefill, and decode analysis.
- Summary filenames include the input sequence length, output sequence length,
  and concurrency, making results from benchmark sweeps easier to identify.
- TraceLens runtime patch selection is now based on the framework version
  installed in the runtime image and the patches available in the selected
  TraceLens checkout, instead of a hard-coded version list.
- Derived TraceLens vLLM images now use committed source plus a minimal pinned
  wheel set, preserve the base vLLM/grpcio versions, record OCI identity labels,
  and reject stale same-name images before reuse.
- Trace processing now selects non-empty inference traces, improving report
  generation when profiler output contains empty trace files.
- The compatibility matrix and the benchmarking, profiling, Ray, MCP, and
  troubleshooting documentation have been updated for the 0.2.0 release.

### Packaging and installation

- The Python package, importable package version, and documentation version are
  now `0.2.0`.
- Reproducible GitHub install examples now pin the `v0.2.0` release tag.
- Magpie remains beta software and retains the lightweight core install and
  optional extras introduced in 0.1.0.

## Magpie 0.1.0

The initial public beta release of Magpie establishes a lightweight,
general-purpose framework for evaluating GPU kernel correctness and performance
across AMD and NVIDIA hardware.

### Release highlights

This release delivers three evaluation modes, support for AMD and NVIDIA hardware, Ray-based remote execution, TraceLens trace analysis, gap analysis, a Model Context Protocol (MCP) server, and structured JSON reporting.

#### Packaging and installation

- The package version is `0.1.0`, and the Python package classifier marks
  Magpie as beta software.
- Core installs remain lightweight. `pip install` installs only Magpie's core
  Python dependencies.
- IntelliKit components are pinned to a fixed commit and moved behind optional
  extras instead of being installed by default.
- The `intellikit` extra installs the pinned IntelliKit integrations:
  `metrix` and `accordo`.
- The `all` extra installs broadly supported optional integrations. It
  currently includes MCP support.

#### Documentation

- Installation docs now distinguish core, MCP-only, IntelliKit, and full
  optional installs.
- Ray documentation now clarifies that worker `requirements.txt` installation
  does not automatically install pyproject extras.
- Missing-tool messages for Metrix and Accordo now point to the matching Magpie
  extras.

#### Evaluation modes

Magpie 0.1.0 ships with three evaluation modes.

- **Analyze**: Single-kernel evaluation against a testcase, with optional
  performance profiling.
- **Compare**: Multi-kernel comparison and ranking against a configurable
  baseline to identify the fastest correct implementation.
- **Benchmark**: Framework-level benchmarking for vLLM, SGLang, and Atom, with
  optional torch profiler and system profiler runs.

#### Hardware and execution

This release supports the following hardware and execution environments.

- Support for AMD (HIP/ROCm) and NVIDIA (CUDA) GPUs.
- Three execution environments: local host, sandboxed container, and remote
  Ray cluster.
- Hardware-aware evaluation with optional GPU power and frequency control.
- Automatic idle-GPU selection in Benchmark mode for both AMD and NVIDIA
  devices.

#### Kernel types

The following kernel types are supported for compilation and evaluation.

- HIP, CUDA, PyTorch, and Triton kernels.

#### Profiling and trace analysis

The following profiling and trace analysis capabilities are included.

- Pluggable performance profiler backends: `rocprof-compute` (AMD), `ncu`
  (NVIDIA), and IntelliKit Metrix.
- Optional correctness validation using a testcase or IntelliKit Accordo.
- TraceLens integration for performance trace analysis.
- TraceLens inference runs emit per-stage simple roofline summary CSVs for
  quick kernel/op review.
- Kernel-level gap analysis from torch profiler traces, including a standalone
  mode that runs on existing traces.
- Kernel source finder to locate kernel source files and test commands for AMD
  kernels.

#### Integration

Magpie integrates with the following external systems and workflows.

- Model Context Protocol (MCP) server exposing Magpie capabilities to AI
  agents.
- Agent skill packaging for environments without MCP.
- Structured JSON reports for pipeline integration.

#### Configuration

Magpie provides the following configuration mechanisms.

- Framework-level configuration using `config.yaml`.
- Per-evaluation kernel configuration files for analyze and compare modes.
- Benchmark configuration files for framework benchmarks.
