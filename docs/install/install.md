# Install Magpie

This page provides step-by-step instructions to install Magpie and verify the
installation on your system. Choose the method that best fits your workflow:
install directly from GitHub (recommended for most users), install from source
(recommended for development), or install into a managed virtual environment
with `make`.

## Prerequisites

Before installing Magpie, make sure your system meets the following
requirements:

- Python 3.10 or newer
- `pip` (and optionally `git` for source/GitHub installs)
- An AMD ROCm (HIP) or NVIDIA CUDA toolchain, if you plan to compile or profile
  GPU kernels
- (Optional) `rocprof-compute` (AMD) or `ncu` (NVIDIA) for performance profiling
- (Optional) Docker, used by Benchmark mode when `run_mode` is `docker`

For the full list of verified hardware and software versions, see the
[Compatibility matrix](../reference/compatibility-matrix.md).

## Install from GitHub (recommended)

This installs the latest published Magpie package and its core dependencies.

```bash
pip install git+https://github.com/AMD-AGI/Magpie.git
```

To also install the optional Model Context Protocol (MCP) server dependencies,
request the `mcp` extra:

```bash
pip install "magpie-eval[mcp] @ git+https://github.com/AMD-AGI/Magpie.git"
```

## Install from source (development)

Use an editable install when you want to modify Magpie or follow the latest
changes on a branch.

```bash
# Clone the repository
git clone https://github.com/AMD-AGI/Magpie.git
cd Magpie

# Editable install (recommended for development)
pip install -e .

# Optional: include the MCP server extra
pip install -e ".[mcp]"

# Optional: include development tools (pytest, black, isort, mypy)
pip install -e ".[dev]"
```

## Install with make (managed virtual environment)

The provided `Makefile` creates a `.venv` virtual environment and installs the
runtime dependencies from `requirements.txt`.

```bash
git clone https://github.com/AMD-AGI/Magpie.git
cd Magpie

# Create .venv and install runtime dependencies
make install

# Or install runtime + development tools
make install-dev

# Activate the environment
source .venv/bin/activate
```

## Optional dependencies

Some Magpie features rely on additional packages that are not installed by
default. Install them based on the features you need:

- **GPU frameworks**: `torch`, `triton`, `vllm`, or `sglang` for kernel
  compilation and framework benchmarks.
- **AMD profiling**: `rocprof-compute` (version 3.40 or newer). See the
  [rocprofiler-compute install guide](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/install/core-install.html).
- **NVIDIA profiling**: `ncu` (Nsight Compute). See the
  [Nsight Compute CLI documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html).
- **IntelliKit Metrix** (human-readable AMD profiling metrics):

  ```bash
  pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=metrix"
  ```

- **IntelliKit Accordo** (AMD kernel correctness validation via HSA
  interception):

  ```bash
  pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=accordo"
  ```

## Verify the installation

After installing, confirm that the package imports and the CLI is available:

```bash
# Confirm the package imports
python -c "import Magpie; print('Magpie import OK')"

# Confirm the CLI is on your PATH
magpie --help

# Show detected GPU and toolchain information
magpie --gpu-info
```

```{note}
You can use `python -m Magpie` in place of the `magpie` CLI for the same
subcommands.
```

A successful `magpie --gpu-info` run prints the detected GPU vendor,
architecture, and device count. If no GPU is detected, kernel compilation and
profiling features are unavailable, but the CLI still installs and runs.

## Next steps

- Run your first evaluation with the [Analyze and compare](../how-to/analyze-compare.md)
  guide.
- Benchmark an inference framework with the [Benchmark](../how-to/benchmark.md)
  guide.
- Browse the [Examples](../examples/examples.md) for end-to-end walkthroughs.
