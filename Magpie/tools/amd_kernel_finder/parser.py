###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Kernel name parser - extract structured information from profiler kernel names.
"""

import re
from typing import Optional

from .models import KernelKind, KernelCategory, ParsedKernelName


class KernelNameParser:
    """Parse kernel names from profiler output."""
    
    # Patterns for classification
    TRITON_PATTERN = re.compile(r'^[_a-zA-Z][\w]*\.kd$|\.k\.d$')
    TENSILE_PATTERN = re.compile(r'^Cijk_')
    CK_PATTERN = re.compile(r'^_ZN7ck_tile|ck_tile::')
    ATEN_PATTERN = re.compile(r'void at::native::')
    INDUCTOR_PATTERN = re.compile(r'triton_\w+_fused_')
    HIPBLASLT_PATTERN = re.compile(r'wvSplitK|wvSpltK|DeviceGemmWmma')
    AITER_PATTERN = re.compile(r'^_ZN5aiter|aiter::')
    ROCM_RUNTIME_PATTERN = re.compile(r'^__amd_rocclr_|^MEMORY_COPY_')
    
    # Category keywords
    CATEGORY_KEYWORDS = {
        KernelCategory.ATTENTION: ['attention', 'fmha', 'Fmha', 'unified_attention', 'paged_attention'],
        KernelCategory.GEMM: ['Cijk_', 'gemm', 'Gemm', 'wvSplitK', 'wvSpltK', 'DeviceGemmWmma', 'matmul'],
        KernelCategory.MOE_GEMM: ['matmul_ogs', '_ogs_', 'moe'],
        KernelCategory.LAYERNORM: ['rmsnorm', 'layernorm', 'Rmsnorm', 'Layernorm', 'rms_norm'],
        KernelCategory.SOFTMAX: ['softmax', 'Softmax'],
        KernelCategory.COPY: ['copy', 'Copy', 'direct_copy'],
        KernelCategory.ELEMENTWISE: ['elementwise', 'Fill', 'FillFunctor', 'vectorized_elementwise', 'silu_and_mul', 'gelu'],
        KernelCategory.INDEXING: ['index', 'Index', 'gather', 'scatter'],
        KernelCategory.REDUCE: ['reduce', 'Reduce', 'argmax', 'argmin', 'sum_bitmatrix'],
        KernelCategory.ROUTER: ['topk', 'bitmatrix', 'routing', 'ragged_tensor'],
        KernelCategory.KV_CACHE: ['reshape_and_cache', 'cache'],
        KernelCategory.BLIT: ['rocclr_copy', 'Blit', 'blit', '__amd_rocclr'],
        KernelCategory.ANNOTATION: ['execute_context', 'CompiledFxGraph', '## Call'],
    }
    
    def parse(self, name: str) -> ParsedKernelName:
        """Parse a kernel name and extract structured information."""
        kind = self._classify_kind(name)
        
        if kind == KernelKind.TRITON_JIT:
            return self._parse_triton(name)
        elif kind == KernelKind.TENSILE_GEMM:
            return self._parse_tensile(name)
        elif kind == KernelKind.CK_TILE:
            return self._parse_ck_tile(name)
        elif kind == KernelKind.ATEN_NATIVE:
            return self._parse_aten(name)
        elif kind == KernelKind.HIP_CPP:
            return self._parse_hip(name)
        elif kind == KernelKind.INDUCTOR:
            return self._parse_inductor(name)
        elif kind == KernelKind.AITER:
            return self._parse_aiter(name)
        elif kind == KernelKind.ANNOTATION:
            return ParsedKernelName(
                original_name=name,
                kind=kind,
                function_name=name,
            )
        else:
            return ParsedKernelName(
                original_name=name,
                kind=kind,
                function_name=name,
            )
    
    def _classify_kind(self, name: str) -> KernelKind:
        """Classify the kernel type based on name pattern."""
        # Check for annotations first
        if name.startswith('execute_context') or name.startswith('## Call'):
            return KernelKind.ANNOTATION
        
        # Check for Tensile GEMM
        if self.TENSILE_PATTERN.match(name):
            return KernelKind.TENSILE_GEMM
        
        # Check for hipBLASLt WMMA kernels (before general HIP check)
        if self.HIPBLASLT_PATTERN.search(name):
            return KernelKind.HIP_CPP
        
        # Check for aiter kernels (before CK and Triton checks)
        if self.AITER_PATTERN.search(name):
            return KernelKind.AITER
        
        # Check for CK tile (handles both mangled and readable names)
        if self.CK_PATTERN.search(name):
            return KernelKind.CK_TILE
        
        # Check for ATen native
        if self.ATEN_PATTERN.search(name):
            return KernelKind.ATEN_NATIVE
        
        # Check for inductor generated
        if self.INDUCTOR_PATTERN.search(name):
            return KernelKind.INDUCTOR
        
        # Check for ROCm runtime kernels (before Triton check)
        if self.ROCM_RUNTIME_PATTERN.search(name):
            return KernelKind.HIP_CPP
        
        # Check for Triton JIT (ends with .kd or .k.d)
        if name.endswith('.kd') or name.endswith('.k.d'):
            # Check if it looks like a HIP/CUDA kernel (has void prefix)
            if name.startswith('void '):
                return KernelKind.HIP_CPP
            return KernelKind.TRITON_JIT
        
        # Check for other HIP/CUDA kernels
        if 'void ' in name and '[clone .kd]' in name:
            return KernelKind.HIP_CPP
        
        return KernelKind.UNKNOWN
    
    def classify_category(self, name: str) -> KernelCategory:
        """Classify the operation category based on kernel name."""
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in name:
                    return category
        return KernelCategory.UNKNOWN
    
    def _parse_triton(self, name: str) -> ParsedKernelName:
        """Parse Triton JIT kernel name."""
        # Remove .kd suffix
        base = name[:-3] if name.endswith('.kd') else name
        
        # Common patterns:
        # _matmul_ogs_NNT_bf16xbf16xmxfp4_32x256x128x1_swiglu
        # kernel_unified_attention_3d
        # _reduce
        
        parts = base.split('_')
        function_name = base
        config = ""
        dtype = ""
        
        # Try to find where config starts (dtype patterns)
        dtype_patterns = ['bf16', 'fp16', 'fp32', 'fp8', 'mxfp4', 'int8']
        for i, part in enumerate(parts):
            for dt in dtype_patterns:
                if dt in part.lower():
                    function_name = '_'.join(parts[:i])
                    config = '_'.join(parts[i:])
                    dtype = dt
                    break
            if config:
                break
        
        # If no config found, function name is the whole base
        if not function_name:
            function_name = base
        
        return ParsedKernelName(
            original_name=name,
            kind=KernelKind.TRITON_JIT,
            function_name=function_name,
            config=config,
            dtype=dtype,
        )
    
    def _parse_tensile(self, name: str) -> ParsedKernelName:
        """Parse Tensile GEMM kernel name."""
        # Example: Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x64x256_MI16x16x1_...
        
        extra = {}
        
        # Extract transpose info
        if 'Alik' in name:
            extra['trans_a'] = 'T'
        if 'Bljk' in name:
            extra['trans_b'] = 'N'
        
        # Extract tile sizes: MT<M>x<N>x<K>
        mt_match = re.search(r'MT(\d+)x(\d+)x(\d+)', name)
        if mt_match:
            extra['tile_m'] = int(mt_match.group(1))
            extra['tile_n'] = int(mt_match.group(2))
            extra['tile_k'] = int(mt_match.group(3))
        
        # Extract MFMA: MI<m>x<n>x<k>
        mi_match = re.search(r'MI(\d+)x(\d+)x(\d+)', name)
        if mi_match:
            extra['mfma'] = f"{mi_match.group(1)}x{mi_match.group(2)}x{mi_match.group(3)}"
        
        # Extract workgroup: WG<x>_<y>_<z>
        wg_match = re.search(r'WG(\d+)_(\d+)_(\d+)', name)
        if wg_match:
            extra['workgroup'] = f"{wg_match.group(1)}x{wg_match.group(2)}x{wg_match.group(3)}"
        
        return ParsedKernelName(
            original_name=name,
            kind=KernelKind.TENSILE_GEMM,
            function_name="Tensile GEMM",
            extra=extra,
        )
    
    def _parse_ck_tile(self, name: str) -> ParsedKernelName:
        """Parse Composable Kernel tile name."""
        # Example: _ZN7ck_tile6kentryILi1ENS_12Rmsnorm2dFwd...
        
        extra = {}
        
        # Extract operation name
        op_patterns = [
            (r'Rmsnorm2dFwd', 'Rmsnorm2dFwd'),
            (r'Fmha', 'Fmha'),
            (r'Softmax', 'Softmax'),
            (r'Gemm', 'Gemm'),
        ]
        
        op_name = "Unknown CK Op"
        for pattern, op in op_patterns:
            if pattern in name:
                op_name = op
                break
        
        # Extract dtype
        if 'DF16b' in name:
            extra['dtype'] = 'bf16'
        elif 'DF16' in name:
            extra['dtype'] = 'fp16'
        elif 'DF32' in name:
            extra['dtype'] = 'fp32'
        
        # Extract block shape
        shape_match = re.search(r'sequenceIJLi(\d+)ELi(\d+)E', name)
        if shape_match:
            extra['block_shape'] = f"{shape_match.group(1)}x{shape_match.group(2)}"
        
        # Check for fused operations
        if 'FusedAdd' in name:
            extra['fused_add'] = True
        if 'FusedQuant' in name:
            extra['fused_quant'] = True
        
        return ParsedKernelName(
            original_name=name,
            kind=KernelKind.CK_TILE,
            function_name=op_name,
            extra=extra,
        )
    
    def _parse_aten(self, name: str) -> ParsedKernelName:
        """Parse ATen native kernel name."""
        # Example: void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>...
        
        extra = {}
        
        # Extract kernel type
        if 'vectorized_elementwise_kernel' in name:
            extra['kernel_type'] = 'elementwise'
        elif 'reduce_kernel' in name:
            extra['kernel_type'] = 'reduce'
        elif 'index_elementwise_kernel' in name:
            extra['kernel_type'] = 'indexing'
        
        # Extract functor
        functor_match = re.search(r'at::native::(\w+Functor)', name)
        if functor_match:
            extra['functor'] = functor_match.group(1)
        
        # Extract operation from lambda/functor
        op_patterns = [
            (r'FillFunctor', 'fill'),
            (r'CompareEqFunctor', 'eq'),
            (r'div_trunc_kernel_cuda', 'div_trunc'),
            (r'direct_copy_kernel_cuda', 'copy'),
            (r'ArgMaxOps', 'argmax'),
        ]
        
        op_name = "ATen Op"
        for pattern, op in op_patterns:
            if pattern in name:
                op_name = op
                break
        
        return ParsedKernelName(
            original_name=name,
            kind=KernelKind.ATEN_NATIVE,
            function_name=op_name,
            extra=extra,
        )
    
    def _parse_hip(self, name: str) -> ParsedKernelName:
        """Parse HIP/CUDA C++ kernel name."""
        # Example: void vllm::reshape_and_cache_flash_kernel<__hip_bfloat16...
        # Example: void wvSplitK_hf_sml_<__hip_bfloat16, 64, 2...
        
        extra = {}
        
        # Extract namespace and kernel name
        match = re.search(r'void (\w+)::(\w+)', name)
        if match:
            extra['namespace'] = match.group(1)
            function_name = match.group(2)
        else:
            # Try without namespace
            match = re.search(r'void (\w+)<', name)
            if match:
                function_name = match.group(1)
            else:
                function_name = "HIP kernel"
        
        # Extract dtype
        if '__hip_bfloat16' in name:
            extra['dtype'] = 'bf16'
        elif 'float' in name:
            extra['dtype'] = 'fp32'
        elif '__half' in name:
            extra['dtype'] = 'fp16'
        
        return ParsedKernelName(
            original_name=name,
            kind=KernelKind.HIP_CPP,
            function_name=function_name,
            namespace=extra.get('namespace', ''),
            extra=extra,
        )
    
    def _parse_inductor(self, name: str) -> ParsedKernelName:
        """Parse torch.inductor generated kernel name."""
        # Example: triton_poi_fused_0.kd
        
        base = name[:-3] if name.endswith('.kd') else name
        
        return ParsedKernelName(
            original_name=name,
            kind=KernelKind.INDUCTOR,
            function_name=base,
            extra={'generated': True},
        )
    
    def _parse_aiter(self, name: str) -> ParsedKernelName:
        """Parse aiter kernel name."""
        # Examples:
        # _ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32EEEvPT0_PfPKT_PKfiliibPKii.kd
        # _ZN5aiter50fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256E.kd
        
        extra = {}
        function_name = "aiter kernel"
        
        # Extract function name from mangled name
        # Pattern: _ZN5aiter<len><function_name>...
        match = re.search(r'_ZN5aiter\d+(\w+)', name)
        if match:
            function_name = match.group(1)
        
        # Detect kernel category
        if 'quant' in name.lower():
            extra['category'] = 'quant'
        elif 'fmoe' in name.lower() or 'moe' in name.lower():
            extra['category'] = 'moe'
        elif 'gemm' in name.lower():
            extra['category'] = 'gemm'
        elif 'attention' in name.lower() or 'mha' in name.lower():
            extra['category'] = 'attention'
        elif 'norm' in name.lower():
            extra['category'] = 'norm'
        
        # Extract dtype
        if 'bf16' in name.lower() or 'DF16b' in name:
            extra['dtype'] = 'bf16'
        elif 'fp8' in name.lower() or 'Fp8' in name:
            extra['dtype'] = 'fp8'
        elif 'fp16' in name.lower() or 'DF16' in name:
            extra['dtype'] = 'fp16'
        elif 'fp4' in name.lower() or 'mxfp4' in name.lower():
            extra['dtype'] = 'fp4'
        
        # Extract config info from name
        config_match = re.search(r'_(\d+x\d+)', name)
        if config_match:
            extra['config'] = config_match.group(1)
        
        return ParsedKernelName(
            original_name=name,
            kind=KernelKind.AITER,
            function_name=function_name,
            namespace="aiter",
            extra=extra,
        )
