# Magpie

A lightweight, general-purpose framework for evaluating GPU kernel correctness and performance.

## Features

- **Three Evaluation Modes**: Analyze, Compare, Benchmark
- **Heterogeneous Hardware**: AMD (HIP) and NVIDIA (CUDA) GPUs
- **Execution Environments**: Local, Sandbox Container, and Remote Ray Cluster
- **Hardware Control**: Hardware-aware kernel evaluation under controlled execution settings
- **Auto GPU Selection**: Benchmark mode picks idle GPU(s) before launching (AMD + NVIDIA)
- **Trace Analysis**: TraceLens integration with per-stage roofline summaries for benchmark traces
- **MCP Server**: Model Context Protocol integration for AI agents
- **Structured Reports**: JSON output for pipeline integration

## Requirements

- Python 3.10+
- AMD ROCm (HIP) or NVIDIA CUDA toolchain (for kernel compilation/profiling)
- `rocprof-compute` (AMD) or `ncu` (NVIDIA) if you enable performance profiling
- Docker (default for Benchmark mode when `run_mode` is `docker`; host execution uses `run_mode: local`)

## Installation

### From GitHub (Recommended)

```bash
# Basic installation
pip install git+https://github.com/AMD-AGI/Magpie.git

# Optional MCP support
pip install "magpie-eval[all] @ git+https://github.com/AMD-AGI/Magpie.git"
```

### From Source (Development)

```bash
git clone https://github.com/AMD-AGI/Magpie.git
cd Magpie

# Editable install (recommended for development)
pip install -e .

# Optional MCP support
pip install -e ".[all]"

# Or use make for a managed dependency virtualenv
make install
```

## Quick Start

```bash
# Analyze a kernel using a config file
magpie analyze --kernel-config Magpie/kernel_config.yaml.example

# Compare kernels using a config file
magpie compare --kernel-config examples/ck_grouped_gemm_compare.yaml

# Benchmark vLLM (see examples/benchmarks/*.yaml)
magpie benchmark --benchmark-config examples/benchmarks/benchmark_vllm_dsr1.yaml

# GPU / toolchain summary
magpie --gpu-info

# Run MCP server
python -m Magpie.mcp
```

> **Note:** You can use `python -m Magpie` instead of the `magpie` CLI for the same subcommands.

## Evaluation Modes

| Mode | Description | Status |
|------|-------------|--------|
| **Analyze** | Single kernel evaluation with testcase | ✅ |
| **Compare** | Multi-kernel comparison and ranking | ✅ |
| **Benchmark** | Framework-level benchmarking (vLLM/SGLang/Atom) with trace analysis | ✅ |

> 📖 See [Benchmark mode](docs/how-to/benchmarking/benchmark.md) for vLLM/SGLang/Atom usage.  
> 📖 See [Analyze vs Compare](docs/how-to/analyze-compare.md) for kernel evaluation modes.

### SGLang expert-load capture

Set `MAGPIE_RECORD_EXPERT_DISTRIBUTION: "1"` in benchmark `envs` to record
per-worker expert totals for the client, including its warmup requests. Magpie
uses SGLang's plugin hook to accumulate one `int64[layer, physical_expert]` matrix
per worker throughout the capture. Storage and recorder memory do not grow with
the number of forward passes. The opt-in launch requires native recorder mode
`stat`, buffer size `1`, and EPLB disabled; incompatible explicit options fail.

The standalone plugin installs offline into a private `PYTHONPATH` target using
`--no-deps --no-build-isolation`; Magpie is not installed into the serving runtime.
Existing SGLang plugin discovery is preserved. Before each sample, Magpie probes
start/stop/dump on every worker, checks fresh aggregate markers, then resets for
the real capture. Plugin, recording, or export failures fail the requested profile;
ordinary benchmarks remain unchanged when capture is disabled. Supported capture
layouts currently require PP1 and either DP1 or DP attention within TP.

`benchmark_report.json` links `expert_distribution.json` (schema version 2),
containing each worker's source identity, native rank, physical-expert totals,
forward-pass count/range, and final expert map. Worker totals are not summed or
assumed independent. Native balancedness reporting remains available.

Reused server/client containers share a stable recorder directory under the
InferenceX mount. An explicit `SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR` override
must resolve to the same directory in every server and client process.

## Configuration

### Framework Config (`Magpie/config.yaml`)

Key categories:
- `gpu`: device selection and hardware control (power/frequency).
- `scheduler`: local, container, or Ray execution and worker settings.
- `compiling` / `correctness`: default compile behavior, testcase vs Accordo, tolerances.
- `performance`: profiler backend (rocprof-compute, ncu, Metrix), timeouts, metric blocks.
- `compare`: perf metric weights and winner selection for compare mode.
- `benchmark`: InferenceX path, image mapping, default profiler flags.
- `logging`: log levels and optional file output.

### Kernel Config

See [`Magpie/kernel_config.yaml.example`](./Magpie/kernel_config.yaml.example) for full examples.

### Example Configs

Example configs live in `examples/`:

| Mode | Config File | Description |
|------|-------------|-------------|
| Analyze | `examples/ck_gemm_add.yaml` | Single kernel evaluation |
| Analyze | `examples/simple_hip_test/analyze_default.yaml` | Minimal HIP example |
| Compare | `examples/ck_grouped_gemm_compare.yaml` | Multi-kernel comparison |
| Benchmark | `examples/benchmarks/benchmark_vllm_dsr1.yaml` | vLLM (DeepSeek-R1-style) |
| Benchmark | `examples/benchmarks/benchmark_vllm_tracelens.yaml` | vLLM + TraceLens |
| Benchmark | `examples/benchmarks/benchmark_vllm_kimi_k2.yaml` | vLLM + gap analysis example |
| Benchmark | `examples/benchmarks/benchmark_sglang_dsr1.yaml` | SGLang benchmark |
| Benchmark | `examples/benchmarks/benchmark_atom_dsr1.yaml` | Atom benchmark (single-node v1) |
| Benchmark | `examples/benchmarks/benchmark_vllm_*_ray.yaml` | vLLM on Ray|

## MCP Server

MCP configuration example: [`Magpie/mcp/config.json`](./Magpie/mcp/config.json)

Available tools:
- `analyze` - Analyze kernel correctness and performance
- `compare` - Compare multiple kernel implementations
- `hardware_spec` - Query GPU hardware specifications
- `configure_gpu` - Configure GPU power and frequency
- `discover_kernels` - Scan a project and suggest analyzable kernels/configs
- `suggest_optimizations` - Suggest performance optimizations from analyze output
- `create_kernel_config` - Generate a kernel config YAML for analyze
- `benchmark` - Run vLLM/SGLang/Atom framework benchmark with optional profiling
- `gap_analysis` - Run gap analysis on existing torch profiler traces
- `list_benchmark_images` - List available Docker images per framework/arch
- `list_benchmark_results` - List previous benchmark workspaces and summaries
- `get_benchmark_result` - Read detailed results from a specific benchmark run
- `compare_benchmark_reports` - Compare TraceLens reports across benchmark runs

For environments without MCP, install the Magpie skill; see [docs/how-to/mcp-and-skills.md](docs/how-to/mcp-and-skills.md).

## Development

```bash
make install-dev
make lint
make format
```

## Project Structure

```
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml       # Package configuration (pip install)
├── requirements.txt
├── Makefile
├── examples/            # Example configurations
├── docs/                # Documentation (ROCm docs site source)
│   ├── index.rst             # Overview / landing page
│   ├── install/              # Installation instructions
│   ├── reference/            # Release notes, compatibility, API reference
│   ├── how-to/               # How-to guides (benchmark, analyze/compare, ray, MCP/skills)
│   ├── examples/             # Step-by-step examples
│   ├── about/                # License
│   └── images/               # Architecture diagrams
└── Magpie/
    ├── __init__.py          # Package initialization
    ├── __main__.py          # Entry point for python -m Magpie
    ├── main.py              # CLI implementation
    ├── config.yaml          # Framework configuration
    ├── kernel_config.yaml.example
    ├── config/              # Configuration classes
    ├── core/                # Core engine components
    ├── eval/                # Evaluation pipeline
    ├── modes/               # Evaluation modes
    │   ├── analyze_eval/    # Single kernel analysis
    │   ├── compare_eval/    # Multi-kernel comparison
    │   └── benchmark/       # Framework-level benchmarking
    │       ├── benchmarker.py   # Benchmark orchestration
    │       ├── config.py        # Benchmark configuration
    │       ├── tracelens.py     # TraceLens integration
    │       ├── gap_analysis.py  # Kernel bottleneck report from torch traces
    │       └── result.py        # Result data structures
    ├── mcp/                 # MCP Server
    │   ├── __init__.py
    │   ├── __main__.py      # Entry point for python -m Magpie.mcp
    │   ├── server.py        # MCP server implementation
    │   └── config.json      # MCP client configuration
    └── utils/               # Utility functions
```

## Overall Architecture Diagram

![Overall Architecture](docs/images/overall-architecture.png)

## Eval Pipeline

### Analyze & Compare

![Analyze & Compare Pipeline](docs/images/analyze-compare-pipeline.png)

### Benchmark

![Benchmark Pipeline](docs/images/benchmark-pipeline.png)

## License

MIT License. See `LICENSE`.
