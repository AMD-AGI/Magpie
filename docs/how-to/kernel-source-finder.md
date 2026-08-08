---
myst:
    html_meta:
        "description": "Automatically map GPU kernel names from profiler traces to their source code and test files using Magpie's kernel source finder for AMD and NVIDIA kernels."
        "keywords": "Magpie, kernel source finder, GPU kernels, Triton, CK Tile, Tensile, gap analysis, ROCm, profiler traces"
---

# Find kernel sources with Magpie

When gap analysis identifies the GPU kernels dominating your benchmark runtime, the kernel source finder maps those mangled kernel names back to their human-readable source files and runnable test commands. It parses the kernel name, searches caller-supplied or explicitly cloned repository roots, and writes source file paths, GitHub URLs, and test commands directly into the gap analysis CSV. Source resolution is fail-closed: a missing or ambiguous definition leaves the source fields empty and records a machine-readable error instead of emitting a plausible placeholder.

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
- **Verified mappings**: Known paths are emitted only when the file exists in a supplied root
- **Kernel index**: Pre-built index for faster lookups

For reproducible benchmark runs, set `gap_analysis.kernel_source_repos` to the exact locked vLLM and AITER roots. Triton source lookup searches these roots explicitly, along with supplied Triton and ROCm Libraries roots. It does not consult ambient `VLLM_DIR` or `AITER_DIR` values and does not fall back to a text placeholder.

The index accepts only a unique source-level identifier compatible with the parser's repository and kernel-kind constraints. Empty or mangled names, cross-repository matches, and ambiguous prefix matches remain unresolved.

Composable Kernel source embedded in AITER is usable only when `3rdparty/composable_kernel` is materialized. If the gitlink is present but its source tree is absent, the row reports `source_resolution=unsupported` with `source_error=ck_submodule_not_materialized`; a generic CK directory is never treated as an exact source file.

### Step 4: Generate output

Results are written to `gap_analysis.csv`:

```text
Name,Calls,Self CUDA total (us),...,kind,category,source_repo,source_file,...,notes,source_resolution,source_error
kernel_paged_attention_2d.kd,12288,1679000.00,...,triton_jit,attention,vllm,$VLLM_DIR/vllm/v1/attention/ops/chunked_prefill_paged_decode.py,...,,resolved,
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
| `source_resolution` | `resolved`, `unresolved`, or `unsupported` |
| `source_error` | Stable fail-closed reason such as `triton_source_not_found`, `triton_source_ambiguous`, or `ck_submodule_not_materialized` |

Only `source_resolution=resolved` is patchable source evidence. Empty source fields are intentional for the other statuses.

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
