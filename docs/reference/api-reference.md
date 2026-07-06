---
myst:
    html_meta:
        "description": "Reference documentation for the Magpie CLI, YAML configuration files, and MCP server tools, including command options, parameters, and usage examples."
        "keywords": "Magpie, API reference, CLI, MCP server, YAML configuration, magpie analyze, magpie compare, magpie benchmark, GPU kernel"
---

# Magpie API reference

Magpie exposes three public interfaces: a command-line interface (CLI), a set of
configuration files, and a Model Context Protocol (MCP) server for AI agents.
This page documents each interface, including command names, parameters, and
expected behavior.

```{note}
You can invoke the CLI either as `magpie <command>` or as
`python -m Magpie <command>`. Both forms accept the same arguments.
```

## Command-line interface

### Global options

The following options apply to every command and are passed before the subcommand.

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--config`, `-c` | path | `Magpie/config.yaml` | Framework configuration file |
| `--verbose`, `-v` | flag | off | Enable verbose (DEBUG) logging |
| `--gpu-info` | flag | off | Print detected GPU and toolchain info, then exit |
| `--environment`, `-e` | `local` \| `container` \| `ray` | `local` | Execution environment |
| `--workers`, `-w` | int | from config | Number of concurrent workers |
| `--docker-image` | string | from config | Docker image for the container environment |

Example:

```bash
# Show detected GPU information and exit
magpie --gpu-info
```

### `magpie analyze`

Evaluate a single kernel for correctness against a testcase, then optionally
profile its performance.

```bash
magpie analyze [kernels ...] [options]
```

| Argument / option | Type | Default | Description |
| --- | --- | --- | --- |
| `kernels` | path(s) | none | Kernel source file(s) to analyze |
| `--kernel-config`, `-k` | path | none | Kernel configuration file (alternative to positional kernels) |
| `--testcase`, `-t` | string | none | Testcase command used to verify correctness |
| `--type` | `hip` \| `cuda` \| `pytorch` \| `triton` | `hip` | Kernel type |
| `--compile-cmd` | string | auto | Custom compile command |
| `--no-perf` | flag | off | Skip performance profiling |
| `--output-dir`, `-o` | path | `./results` | Output directory for reports |

Example:

```bash
magpie analyze --kernel-config Magpie/kernel_config.yaml.example
```

### `magpie compare`

Evaluate and rank multiple kernel implementations against a baseline.

```bash
magpie compare [kernels ...] [options]
```

| Argument / option | Type | Default | Description |
| --- | --- | --- | --- |
| `kernels` | path(s) | none | Kernel files to compare |
| `--kernel-config`, `-k` | path | none | Kernel configuration file |
| `--testcase`, `-t` | string | none | Testcase command (optional) |
| `--type` | `hip` \| `cuda` \| `pytorch` \| `triton` | `hip` | Kernel type |
| `--baseline` | int | `0` | Index of the baseline kernel |
| `--no-perf` | flag | off | Skip performance profiling |
| `--output-dir`, `-o` | path | `./results` | Output directory for reports |

Example:

```bash
magpie compare --kernel-config examples/ck_grouped_gemm_compare.yaml
```

### `magpie benchmark`

Run a framework-level benchmark (vLLM, SGLang, or Atom) with optional profiling,
or run standalone gap analysis on existing traces.

```bash
magpie benchmark [framework] [options]
```

| Argument / option | Type | Default | Description |
| --- | --- | --- | --- |
| `framework` | `vllm` \| `sglang` \| `atom` | none | Framework to benchmark |
| `--benchmark-config`, `-b` | path | none | Benchmark configuration file |
| `--model`, `-m` | string | none | Model name or path |
| `--precision`, `-p` | `fp8` \| `fp16` \| `bf16` \| `fp4` | `fp8` | Model precision |
| `--tp` | int | `1` | Tensor parallel size |
| `--concurrency` | int | `32` | Request concurrency |
| `--input-len` | int | `1024` | Input sequence length |
| `--output-len` | int | `512` | Output sequence length |
| `--torch-profiler` | flag | off | Enable the torch profiler |
| `--system-profiler` | flag | off | Enable the system profiler (rocprof/ncu) |
| `--run-mode` | `docker` \| `local` | `docker` | Run in a container or directly on the host |
| `--docker-image` | string | from config | Override the Docker image |
| `--inferencex-path` | string | auto-clone | Path to an InferenceX installation |
| `--benchmark-script` | string | none | InferenceX benchmark script name |
| `--timeout` | int | `3600` | Benchmark timeout in seconds |
| `--output-dir`, `-o` | path | `./results` | Output directory for reports |

#### Standalone gap analysis options

When `--trace-dir` is provided, `benchmark` runs gap analysis on existing torch
traces instead of launching a benchmark.

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `--trace-dir` | path | none | Run gap analysis on this `torch_trace` directory (or a workspace containing one) |
| `--top-k` | int | `20` | Number of top bottleneck kernels to report |
| `--start-pct` | float | `0.0` | Start of the analysis window (0-100) |
| `--end-pct` | float | `100.0` | End of the analysis window (0-100) |
| `--min-duration-us` | float | `0.0` | Minimum event duration to include, in microseconds |
| `--categories` | string(s) | all | Event categories to include (for example, `kernel gpu`) |
| `--ignore-categories` | string(s) | none | Event categories to exclude |
| `--no-rank-csv` | flag | off | Skip per-rank CSV generation |
| `--find-kernel-sources` | flag | off | Find kernel source files and test commands (AMD kernels) |
| `--kernel-source-repos` | string(s) | none | Repository paths to search for kernel sources |

Example:

```bash
magpie benchmark --benchmark-config examples/benchmarks/benchmark_vllm_dsr1.yaml
```

## Configuration reference

Magpie is driven by YAML configuration files. See
[`Magpie/config.yaml`](https://github.com/AMD-AGI/Magpie/blob/main/Magpie/config.yaml)
and
[`Magpie/kernel_config.yaml.example`](https://github.com/AMD-AGI/Magpie/blob/main/Magpie/kernel_config.yaml.example)
for complete, commented examples.

### Framework configuration (`config.yaml`)

| Section | Purpose |
| --- | --- |
| `gpu` | Device selection and hardware control (power/frequency) |
| `scheduler` | Execution environment (local, container, or Ray) and worker settings |
| `compiling` | Default compile behavior |
| `correctness` | Correctness backend (testcase or Accordo) and tolerances |
| `performance` | Profiler backend (`rocprof-compute`, `ncu`, Metrix), timeouts, and metric blocks |
| `compare` | Performance metric weights and winner selection for compare mode |
| `benchmark` | InferenceX path, image mapping, and default profiler flags |
| `logging` | Log level (`level`: `DEBUG`, `INFO`, `WARNING`, `ERROR`) and optional `file` output. For a one-off debug run, use the global `--verbose` (`-v`) flag, which forces `DEBUG`. |

### Kernel configuration

A kernel configuration file can define a single `kernel:` or multiple
`kernels:`, plus optional override sections:

| Key | Purpose |
| --- | --- |
| `kernel` | A single kernel entry (path, type, testcase, compile command) |
| `kernels` | A list of kernel entries for compare mode |
| `performance` | Overrides framework-level profiler settings |
| `correctness` | Overrides framework-level correctness settings |
| `ray_config` | Ray cluster settings (implies `environment: ray`) |
| `scheduler` | Scheduler-level overrides (environment, workers, and so on) |

## MCP server

Start the MCP server with:

```bash
python -m Magpie.mcp
```

A sample client configuration is available at
[`Magpie/mcp/config.json`](https://github.com/AMD-AGI/Magpie/blob/main/Magpie/mcp/config.json).
All tools return JSON-formatted strings.

### Available tools

The MCP server exposes the following tools to AI agents.

| Tool | Description |
| --- | --- |
| `analyze` | Analyze a kernel for correctness and performance |
| `compare` | Compare multiple kernel implementations |
| `hardware_spec` | Query GPU hardware specifications |
| `configure_gpu` | Configure GPU power and frequency |
| `discover_kernels` | Scan a project and suggest analyzable kernels/configs |
| `suggest_optimizations` | Suggest performance optimizations from analyze output |
| `create_kernel_config` | Generate a kernel config YAML for analyze |
| `benchmark` | Run a vLLM/SGLang/Atom benchmark with optional profiling |
| `gap_analysis` | Run gap analysis on existing torch profiler traces |
| `list_benchmark_images` | List available Docker images per framework/architecture |
| `list_benchmark_results` | List previous benchmark workspaces and summaries |
| `get_benchmark_result` | Read detailed results from a specific benchmark run |
| `compare_benchmark_reports` | Compare TraceLens reports across benchmark runs |

### Selected tool signatures

The following examples show key parameters. All parameters have sensible
defaults unless noted.

**`hardware_spec`**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `device_id` | int | `0` | GPU device ID to query |
| `include_all` | bool | `false` | Return info for all available GPUs |

Returns JSON with vendor, architecture, power, clocks, temperature, and memory.

**`analyze`**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `kernel_path` | string | required | Path to the kernel source file (`.hip`, `.cu`, `.py`) |
| `testcase_command` | string | required | Command to run the testcase |
| `kernel_type` | string | `hip` | `hip`, `cuda`, `pytorch`, or `triton` |
| `working_dir` | string | kernel's dir | Working directory |
| `compile_command` | string | auto | Custom compile command |
| `check_performance` | bool | `true` | Run performance profiling |
| `environment` | string | `local` | `local` or `container` |
| `performance_backend` | string | auto | `metrix`, `rocprof_compute`, or `ncu` |
| `correctness_backend` | string | auto | `accordo` or `testcase` |
| `accordo_*` | various | see defaults | Accordo kernel name, binaries, tolerances, and timeout |

Returns JSON with `compiling_state`, `correctness_state`, `performance_state`, a
`score` (0.0-1.0), and detailed per-stage results.

```{note}
For the full parameter list of every tool, inspect the tool docstrings in
[`Magpie/mcp/server.py`](https://github.com/AMD-AGI/Magpie/blob/main/Magpie/mcp/server.py),
which are surfaced to MCP clients automatically.
```
