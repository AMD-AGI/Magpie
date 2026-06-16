###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Magpie MCP (Model Context Protocol) Server.

This module provides MCP tools for GPU kernel evaluation and
framework-level benchmarking that can be used by AI agents
and other MCP clients.

Kernel Tools:
- hardware_spec: Get GPU hardware specifications
- analyze: Analyze kernel correctness and performance
- compare: Compare multiple kernels
- configure_gpu: Configure GPU power/frequency settings
- discover_kernels: Discover analyzable kernels in a project
- suggest_optimizations: Get optimization suggestions from analysis results
- create_kernel_config: Generate kernel config YAML for CLI use

Benchmark Tools:
- benchmark: Run vLLM/SGLang/Atom framework benchmark in Docker
- gap_analysis: Run standalone gap analysis on existing torch profiler traces
- list_benchmark_images: List available Docker images per framework/arch
- list_benchmark_results: List previous benchmark workspaces and summaries
- get_benchmark_result: Read detailed results from a specific benchmark run
- compare_benchmark_reports: Compare TraceLens reports across benchmark runs
"""

from .server import mcp

__all__ = ["mcp"]
