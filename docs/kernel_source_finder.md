# Kernel Source Finder

Automatically maps GPU kernel names from profiler traces to their source code and test files.

## Overview

```
Profiler Trace → Kernel Name → Parser → Searcher → Source & Test Info
                                 │           │
                            Classify     Search in
                            kernel type  cloned repos
```

## Supported Kernel Types

| Type | Pattern | Source Repository |
|------|---------|-------------------|
| **Triton JIT** | `*.kd` (e.g., `_matmul_ogs_NNT.kd`) | triton, triton_kernels |
| **CK Tile** | `_ZN7ck_tile*` | rocm-libraries/composablekernel |
| **Tensile GEMM** | `Cijk_*` | rocm-libraries/rocblas |
| **ATen Native** | `void at::native::*` | pytorch |
| **HIP C++** | `wvSplitK*`, `DeviceGemmWmma*` | rocm-libraries, vllm |
| **AITER** | `_ZN5aiter*` | aiter |
| **Inductor** | `triton_*_fused_*` | pytorch |

## How It Works

### Step 1: Auto-Clone Repositories

When gap analysis runs, it automatically clones required repos to `~/.cache/magpie/repos/`:

```
~/.cache/magpie/repos/
├── rocm-libraries/    # CK Tile, Tensile, hipBLASLt
├── triton/            # Triton compiler
├── pytorch/           # ATen kernels
├── vllm/              # vLLM custom kernels
└── aiter/             # AITER kernels
```

### Step 2: Parse Kernel Name

The parser extracts structured info from kernel names:

```python
# Input: "_matmul_ogs_NNT_bf16xbf16xmxfp4_32x256x128x1.kd"
# Output:
ParsedKernelName(
    kind = TRITON_JIT,
    function_name = "_matmul_ogs_NNT",
    dtype = "bf16",
    config = "bf16xbf16xmxfp4_32x256x128x1"
)
```

### Step 3: Search Source & Test

The searcher looks up source files using:
- **ripgrep**: Fast regex search across repos
- **Static mappings**: Known paths for Tensile, CK Tile examples
- **Kernel index**: Pre-built index for faster lookups

### Step 4: Generate Output

Results are written to `gap_analysis.csv`:

```csv
Name,Calls,Self CUDA total (us),...,kind,category,source_repo,source_file,upstream_url,test_file,test_cmd,notes
_matmul_ogs_NNT_bf16.kd,24552,5631747.87,...,triton_jit,gemm,triton_kernels,$TRITON_KERNELS_DIR/matmul_details/_matmul.py,https://github.com/...,$TRITON_KERNELS_DIR/tests/test_matmul.py,cd $TRITON_KERNELS_DIR && pytest tests/test_matmul.py -v,dtype=bf16
```

## Usage

### Run Gap Analysis with Kernel Source Finding

```bash
python3 -m Magpie benchmark \
    --trace-dir /path/to/torch_trace \
    --output-dir /path/to/output \
    --find-kernel-sources
```

### Output Fields

| Field | Description |
|-------|-------------|
| `kind` | Kernel type (triton_jit, ck_tile, tensile_gemm, etc.) |
| `category` | Operation category (gemm, attention, layernorm, etc.) |
| `source_repo` | Repository name |
| `source_file` | Path to source file (uses `$REPO_DIR` variables) |
| `upstream_url` | GitHub URL to source |
| `test_file` | Path to test file |
| `test_cmd` | Command to run tests |
| `notes` | Additional info (dtype, tile sizes, etc.) |

### Path Variables

The CSV header includes path mappings:

```
# $TRITON_DIR=./triton
# $ROCM_LIBRARIES_DIR=./rocm-libraries
# $CK_DIR=./rocm-libraries/projects/composablekernel
# $AITER_DIR=./aiter
```

Base directory: `~/.cache/magpie/repos/`

## Example Output

For a CK Tile RMSNorm kernel:

```
Name: _ZN7ck_tile6kentryILi1ENS_12Rmsnorm2dFwd...
kind: ck_tile
category: layernorm
source_repo: rocm-libraries
source_file: $ROCM_LIBRARIES_DIR/projects/composablekernel/include/ck_tile/ops/rmsnorm2d/kernel/rmsnorm2d_fwd_kernel.hpp
test_file: $ROCM_LIBRARIES_DIR/projects/composablekernel/example/ck_tile/10_rmsnorm2d/
test_cmd: cd $ROCM_LIBRARIES_DIR/projects/composablekernel/build && cmake --build . -j --target tile_example_rmsnorm2d_fwd
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    KernelSourceFinder                       │
│                    (finder.py)                              │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ RepoManager  │  │ KernelName   │  │ KernelSource     │  │
│  │              │  │ Parser       │  │ Searcher         │  │
│  │ - auto clone │  │              │  │                  │  │
│  │ - 5 repos    │  │ - classify   │  │ - ripgrep search │  │
│  │              │  │ - parse info │  │ - static mapping │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│         │                 │                   │             │
│         ▼                 ▼                   ▼             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                   KernelSourceInfo                    │  │
│  │  (kind, category, source_file, test_file, test_cmd)  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Adding New Kernel Types

1. Add pattern to `parser.py`:
   ```python
   MY_PATTERN = re.compile(r'^my_kernel_prefix')
   ```

2. Add search methods to `searcher.py`:
   ```python
   def _search_my_source(self, parsed):
       # Search logic
   
   def _search_my_test(self, parsed, source):
       # Test search logic
   ```

3. Add repo URL to `repo_manager.py`:
   ```python
   REPO_URLS = {
       "my-repo": "https://github.com/org/my-repo.git",
   }
   ```
