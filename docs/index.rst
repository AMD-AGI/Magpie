.. meta::
   :description: Magpie is a lightweight, general-purpose framework for evaluating GPU kernel correctness and performance on AMD and NVIDIA GPUs.
   :keywords: Magpie, ROCm, GPU, kernel, evaluation, benchmark, HIP, CUDA, profiling, AMD

********************
Magpie documentation
********************

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
* Drive kernel evaluation from an AI agent through the MCP server.

Documentation
=============

The Magpie documentation is organized into the following categories.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Install

      * :doc:`Install Magpie <install/install>`

   .. grid-item-card:: Reference

      * :doc:`Release notes <reference/release-notes>`
      * :doc:`Compatibility matrix <reference/compatibility-matrix>`
      * :doc:`API reference <reference/api-reference>`

   .. grid-item-card:: How to

      * :doc:`Analyze and compare kernels <how-to/analyze-compare>`
      * :doc:`Benchmark frameworks <how-to/benchmark>`
      * :doc:`Run on a Ray cluster <how-to/ray>`
      * :doc:`MCP server and agent skills <how-to/mcp-and-skills>`
      * :doc:`Find kernel sources <how-to/kernel-source-finder>`

   .. grid-item-card:: Examples

      * :doc:`Examples <examples/examples>`

   .. grid-item-card:: About

      * :doc:`License <about/license>`

To contribute to the documentation, see the
`Magpie GitHub repository <https://github.com/AMD-AGI/Magpie>`_.

Magpie is released under the MIT license. For details, see the
:doc:`License <about/license>` page.
