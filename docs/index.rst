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

Magpie is a component of the Hyperloom toolkit (AMD's multi-tool GPU
evaluation platform), but can also be used as a standalone tool. The Magpie
source code is hosted in the
`AMD-AGI/Magpie <https://github.com/AMD-AGI/Magpie>`_ GitHub repository.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Install

      * :doc:`Install Magpie <install/install>`

   .. grid-item-card:: How to

      * :doc:`Analyze and compare kernels <how-to/analyze-compare>`
      * :doc:`Benchmark frameworks <how-to/benchmarking/benchmark>`
      * :doc:`Run on a Ray cluster <how-to/ray>`
      * :doc:`MCP server and agent skills <how-to/mcp-and-skills>`
      * :doc:`Find kernel sources <how-to/kernel-source-finder>`
  
   .. grid-item-card:: Conceptual

      * :doc:`Benchmarking architecture <conceptual/benchmarking-architecture>`
      * :doc:`Ray architecture <conceptual/ray-architecture>`

   .. grid-item-card:: Examples

      * :doc:`Examples <examples/examples>`

   .. grid-item-card:: Reference

      * :doc:`API reference <reference/api-reference>`
      * :doc:`Benchmark configuration <reference/benchmark-config>`
      * :doc:`Troubleshooting <reference/troubleshooting>`

To contribute to the documentation, see the
`Magpie GitHub repository <https://github.com/AMD-AGI/Magpie>`_.

Magpie is released under the MIT license. For details, see the
:doc:`License <about/license>` page.
