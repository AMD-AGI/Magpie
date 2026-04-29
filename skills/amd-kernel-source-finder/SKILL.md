---
name: amd-kernel-source-finder
description: Find kernel source code, test files, and test commands for AMD GPU kernels identified in profiler traces. Use when the user wants to locate kernel implementations, find tests for specific kernels, or enrich gap_analysis results with source information. Supports Triton JIT (ROCm), Tensile GEMM (rocBLAS), CK Tile, hipBLASLt, and HIP kernels.
---

# AMD Kernel Source Finder

Find source code locations and test cases for **AMD GPU kernels** (ROCm/HIP) identified in profiler traces.

> **Note:** This skill is specifically designed for AMD GPUs. It supports kernels from ROCm libraries including Composable Kernel (CK), hipBLASLt, rocBLAS, and Triton on ROCm.

## Workflow

This skill works with Magpie's gap_analysis output. The workflow is:

### Step 1: Run Profiling and Gap Analysis

First, generate a `gap_analysis.csv` with kernel statistics:

```bash
# Use Magpie CLI
magpie benchmark --config benchmark.yaml
```

### Step 2: Find Kernel Sources

**Option A: Automatic Enrichment (Recommended)**

Enable `find_kernel_sources` in your benchmark config:

```yaml
gap_analysis:
  enabled: true
  find_kernel_sources: true
  kernel_source_repos:
    - <path-to-rocm-libraries>
    - <path-to-triton>
```

**Option B: Manual Search by Agent**

If you already have a `gap_analysis.csv`, ask the agent to find sources for specific kernels.

## When to Use This Skill

- User has a `gap_analysis.csv` and wants source/test info for specific kernels
- User asks "where is this kernel defined?" with a kernel name
- User wants to run tests for kernels identified as performance bottlenecks

## Prerequisites

Clone the relevant repositories for source searching:

| Repository | URL | Contains |
|------------|-----|----------|
| rocm-libraries | `https://github.com/ROCm/rocm-libraries` | CK, hipBLASLt, rocBLAS |
| triton | `https://github.com/triton-lang/triton` | Triton JIT kernels |
| vllm | `https://github.com/vllm-project/vllm` | vLLM kernels |
| pytorch | `https://github.com/pytorch/pytorch` | ATen native kernels |

## Kernel Classification

### 1. Triton JIT Kernels (`triton_jit`)

**Pattern:** `<function_name>.kd` or `<function_name>_<config>.kd`

**Examples:**
- `_matmul_ogs_NNT_bf16xbf16xmxfp4_32x256x128x1_swiglu.kd` → function: `_matmul_ogs`
- `kernel_unified_attention_3d.kd` → function: `kernel_unified_attention`
- `triton_poi_fused_0.kd` → torch.compile/inductor generated

**Source locations:**
- `triton/python/triton_kernels/triton_kernels/` - triton_kernels package
- `vllm/vllm/` - vLLM Triton kernels

**Test locations:**
- `triton/python/triton_kernels/tests/test_*.py`
- `vllm/tests/kernels/`

### 2. Tensile GEMM Kernels (`tensile_gemm`)

**Pattern:** `Cijk_<layout>_<options>_MT<M>x<N>x<K>_..._WG<x>_<y>_<z>.kd`

**Example:**
- `Cijk_Alik_Bljk_BBS_BH_MT32x64x256_MI16x16x1_WG16_8_2.kd`

**Key tokens:**
- `Alik` = A transposed, `Bljk` = B not transposed
- `MT<M>x<N>x<K>` = tile sizes
- `MI<m>x<n>x<k>` = MFMA instruction
- `WG<x>_<y>_<z>` = workgroup size

**Source:** `rocm-libraries/projects/rocblas/` or Tensile generated

**Test:** `rocblas-bench -f gemm_ex --a_type bf16_r --b_type bf16_r`

### 3. Composable Kernel Tiles (`ck_tile`)

**Pattern:** `_ZN7ck_tile<len>kentryILi<N>ENS_<OpName>...`

**Examples:**
- `_ZN7ck_tile6kentryILi1ENS_12Rmsnorm2dFwd...` → Op: `Rmsnorm2dFwd`
- Contains `Fmha` → Flash attention
- Contains `DF16b` → bf16 dtype

**Source:** `rocm-libraries/projects/composablekernel/include/ck_tile/ops/`

**Test:** `rocm-libraries/projects/composablekernel/example/ck_tile/`

### 4. ATen Native Kernels (`aten_native`)

**Pattern:** `void at::native::<kernel_template><...>(...) [clone .kd]`

**Examples:**
- `vectorized_elementwise_kernel<4, FillFunctor...` → Fill kernel
- `reduce_kernel<512, 1, ...` → Reduce kernel

**Key tokens:**
- `FillFunctor` → `torch.fill_`, `torch.zeros`
- `CompareEqFunctor` → `torch.eq`
- `direct_copy_kernel_cuda` → `torch.Tensor.to`

**Source:** `pytorch/aten/src/ATen/native/cuda/`

**Test:** `pytest pytorch/test/test_torch.py -k <op_name>`

### 5. HIP/CUDA C++ Kernels (`hip_cpp`)

**Pattern:** `void <namespace>::<kernel_name><templates>(...) [clone .kd]`

**Examples:**
- `void vllm::reshape_and_cache_flash_kernel<...` → vLLM cache kernel
- `void wvSplitK_hf_sml_<__hip_bfloat16, 64...` → hipBLASLt WMMA kernel

**Source locations:**
- `vllm/csrc/*.cu` - vLLM HIP kernels
- `rocm-libraries/projects/hipblaslt/` - hipBLASLt kernels

## Agent Instructions

### If user has NO gap_analysis.csv:
1. Guide them to run profiling first:
   ```bash
   magpie benchmark --config <config.yaml>
   ```
2. Suggest enabling `find_kernel_sources: true` for automatic source finding

### If user HAS a gap_analysis.csv:
1. Read the CSV to identify top kernels by `% Total` or `Self CUDA total (us)`
2. Classify each kernel using the patterns above
3. Ask user for repo paths if not provided
4. Search and provide source file + test command

### If user asks about a specific kernel name:
1. Classify by name pattern
2. Identify the likely source repository
3. Provide: source file path, test command

## Category Classification

| Category | Keywords in kernel name |
|----------|------------------------|
| `attention` | `attention`, `fmha`, `Fmha` |
| `gemm` | `gemm`, `matmul`, `Cijk_`, `wvSplitK` |
| `moe_gemm` | `matmul_ogs`, `_ogs_` |
| `layernorm` | `rmsnorm`, `layernorm`, `Rmsnorm` |
| `softmax` | `softmax`, `Softmax` |
| `elementwise` | `elementwise`, `Fill`, `copy` |
| `reduce` | `reduce`, `sum`, `argmax` |
| `router` | `topk`, `bitmatrix`, `routing` |
| `kv_cache` | `reshape_and_cache`, `cache` |

## Programmatic Usage

```python
from Magpie.tools.amd_kernel_finder import KernelSourceFinder

# Initialize with user's repo paths
finder = KernelSourceFinder(repos=[
    "<path-to-rocm-libraries>",
    "<path-to-triton>",
])

# Search for a kernel
result = finder.search("_matmul_ogs_NNT_bf16xbf16xmxfp4.kd")
print(f"Kind: {result.kind}")
print(f"Category: {result.category}")
print(f"Source: {result.source_file}")
print(f"Test: {result.test_cmd}")
```

## YAML Config

```yaml
gap_analysis:
  enabled: true
  find_kernel_sources: true
  kernel_source_repos:
    - <path-to-rocm-libraries>
    - <path-to-triton>
    # Add other repos as needed
```
