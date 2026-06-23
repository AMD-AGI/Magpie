.. meta::
   :description: Magpie is a lightweight, general-purpose framework for evaluating GPU kernel correctness and performance on AMD and NVIDIA GPUs.
   :keywords: Magpie, ROCm, GPU, kernel, evaluation, benchmark, HIP, CUDA, profiling, AMD

***************
What is Magpie?
***************

Magpie is a lightweight, general-purpose framework for evaluating GPU kernel
correctness and performance. It provides a single, reproducible workflow for
checking that a kernel is correct, comparing competing implementations, and
benchmarking full inference frameworks, on both AMD (HIP/ROCm) and NVIDIA
(CUDA) hardware.

Magpie is a component of the Hyperloom toolkit. The Magpie source code is
hosted in the `AMD-AGI/Magpie <https://github.com/AMD-AGI/Magpie>`_ GitHub
repository.

What Magpie does
================

Magpie organizes kernel evaluation into three modes:

* **Analyze** -- Evaluate a single kernel against a testcase for correctness,
  then optionally profile its performance.
* **Compare** -- Evaluate and rank multiple kernel implementations against a
  baseline to find the fastest correct variant.
* **Benchmark** -- Run framework-level benchmarks (vLLM, SGLang, Atom) with
  optional torch and system profiling, including TraceLens trace analysis and
  kernel-level gap analysis.

Key features
============

Magpie provides the following capabilities:

* **Three evaluation modes**: analyze, compare, and benchmark.
* **Heterogeneous hardware**: AMD (HIP) and NVIDIA (CUDA) GPUs.
* **Multiple execution environments**: local host, sandboxed container, and
  remote Ray cluster.
* **Hardware-aware evaluation**: controlled execution with optional power and
  frequency settings.
* **Automatic GPU selection**: benchmark mode picks idle GPUs before launching.
* **Trace analysis**: TraceLens integration for performance profiling and gap
  analysis.
* **MCP server**: Model Context Protocol integration for AI agents.
* **Structured reports**: JSON output for pipeline integration.

Use cases
=========

* Validate hand-written or AI-generated GPU kernels for correctness before
  promoting them.
* Rank competing kernel implementations to pick the fastest correct one.
* Benchmark and profile LLM inference frameworks on AMD GPUs and locate the
  kernels that dominate runtime.

Next steps
==========

To get started with Magpie, see the following pages.

* :doc:`Install Magpie <install/install>` — get up and running in minutes.
* :doc:`Analyze and compare kernels <how-to/analyze-compare>` — validate a kernel and rank competing implementations.
* :doc:`Benchmark frameworks <how-to/benchmark>` — run vLLM, SGLang, or Atom benchmarks with trace analysis.
* :doc:`MCP server and agent skills <how-to/mcp-and-skills>` — drive Magpie from an AI agent.