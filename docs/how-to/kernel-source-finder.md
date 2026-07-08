---
myst:
    html_meta:
        "description": "Automatically map GPU kernel names from profiler traces to their source code and test files using Magpie's kernel source finder for AMD and NVIDIA kernels."
        "keywords": "Magpie, kernel source finder, GPU kernels, Triton, CK Tile, Tensile, gap analysis, ROCm, profiler traces"
---

# Find kernel sources with Magpie

When gap analysis identifies the GPU kernels dominating your benchmark runtime, the kernel source finder maps those mangled kernel names back to their human-readable source files and runnable test commands. It clones the relevant upstream repositories automatically, parses the kernel name to determine its type and origin, and writes source file paths, GitHub URLs, and test commands directly into the gap analysis CSV. Use this feature to quickly locate the code behind a bottleneck kernel and reproduce it in isolation.

## Pipeline overview

The kernel source finder follows a four-step pipeline.

```
Profiler Trace → Kernel Name → Parser → Searcher → Source & Test Info
                                 │           │
                            Classify     Search in
                            kernel type  cloned repos
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    KernelSourceFinder                       │
│                    (finder.py)                              │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ RepoManager  │  │ KernelName   │  │ KernelSource     │   │
│  │              │  │ Parser       │  │ Searcher         │   │
│  │ - auto clone │  │              │  │                  │   │
│  │ - 5 repos    │  │ - classify   │  │ - ripgrep search │   │
│  │              │  │ - parse info │  │ - static mapping │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│         │                 │                   │             │
│         ▼                 ▼                   ▼             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   KernelSourceInfo                   │   │
│  │  (kind, category, source_file, test_file, test_cmd)  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Supported kernel types

The kernel source finder recognizes the following kernel types.

| Type | Pattern | Source Repository |
|------|---------|-------------------|
| **Triton JIT** | `*.kd` (for example, `_matmul_ogs_NNT.kd`) | [triton-lang/triton](https://github.com/triton-lang/triton/tree/main/python/triton_kernels) |
| **CK Tile** | `_ZN7ck_tile*` | [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries/tree/develop/projects/composablekernel) |
| **Tensile GEMM** | `Cijk_*` | [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocblas) |
| **ATen Native** | `void at::native::*` | [pytorch/pytorch](https://github.com/pytorch/pytorch) |
| **HIP C++** | `wvSplitK*`, `DeviceGemmWmma*` | [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries), [vllm-project/vllm](https://github.com/vllm-project/vllm) |
| **AITER** | `_ZN5aiter*` | [ROCm/aiter](https://github.com/ROCm/aiter) |
| **Inductor** | `triton_*_fused_*` | [pytorch/pytorch](https://github.com/pytorch/pytorch) |

## Workflow

The finder runs four sequential steps to map kernel names to source files.

### Step 1: Auto-clone repositories

When gap analysis runs, it automatically clones required repos to `~/.cache/magpie/repos/`:

```
~/.cache/magpie/repos/
├── rocm-libraries/    # CK Tile, Tensile, hipBLASLt
├── triton/            # Triton compiler
├── pytorch/           # ATen kernels
├── vllm/              # vLLM custom kernels
└── aiter/             # AITER kernels
```

### Step 2: Parse kernel name

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

### Step 3: Search source and test

The searcher looks up source files using:
- **ripgrep**: Fast regex search across repos
- **Static mappings**: Known paths for Tensile, CK Tile examples
- **Kernel index**: Pre-built index for faster lookups

### Step 4: Generate output

Results are written to `gap_analysis.csv`:

```text
Name,Calls,Self CUDA total (us),...,kind,category,source_repo,source_file,upstream_url,test_file,test_cmd,notes
_matmul_ogs_NNT_bf16.kd,24552,5631747.87,...,triton_jit,gemm,triton_kernels,$TRITON_KERNELS_DIR/matmul_details/_matmul.py,https://github.com/...,$TRITON_KERNELS_DIR/tests/test_matmul.py,cd $TRITON_KERNELS_DIR && pytest tests/test_matmul.py -v,dtype=bf16
```

## Usage

### Run gap analysis with kernel source finding

Pass `--find-kernel-sources` to enable source lookup during gap analysis.

```bash
python3 -m Magpie benchmark \
    --trace-dir /path/to/torch_trace \
    --output-dir /path/to/output \
    --find-kernel-sources
```

### Output fields

The following fields are added to `gap_analysis.csv` when kernel source finding is enabled.

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

### Path variables

The CSV header includes path mappings:

```
# $TRITON_DIR=./triton
# $ROCM_LIBRARIES_DIR=./rocm-libraries
# $CK_DIR=./rocm-libraries/projects/composablekernel
# $AITER_DIR=./aiter
```

Base directory: `~/.cache/magpie/repos/`

## Example output

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

## Add new kernel types

To add support for a new kernel type, update these three files:

- Add pattern to `parser.py`:
   ```python
   MY_PATTERN = re.compile(r'^my_kernel_prefix')
   ```

- Add search methods to `searcher.py`:
   ```python
   def _search_my_source(self, parsed):
       # Search logic
   
   def _search_my_test(self, parsed, source):
       # Test search logic
   ```

- Add repo URL to `repo_manager.py`:
   ```python
   REPO_URLS = {
       "my-repo": "https://github.com/org/my-repo.git",
   }
   ```
