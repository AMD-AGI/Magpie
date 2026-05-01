###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Source file searcher with ripgrep and Python fallback.
"""

import glob
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Dict

from .models import KernelKind, ParsedKernelName, SourceMatch, TestMatch
from .repo_config import RepoConfig, SUBPROJECT_MAPPINGS, GITHUB_URL_TEMPLATES

logger = logging.getLogger(__name__)


class KernelSourceSearcher:
    """Search for kernel source files in repositories."""
    
    def __init__(self, repos: List[str], repo_configs: Dict[str, RepoConfig] = None,
                 auto_install_ripgrep: bool = True):
        """
        Initialize searcher with repository paths.
        
        Args:
            repos: List of repository root paths
            repo_configs: Optional custom repo configurations
            auto_install_ripgrep: If True, attempt to install ripgrep if missing
        """
        self.repos = repos
        self.repo_configs = repo_configs or {}
        self._repo_var_map: Dict[str, str] = {}
        
        # Check/install ripgrep
        self._has_ripgrep = self._check_ripgrep()
        if not self._has_ripgrep and auto_install_ripgrep:
            self._has_ripgrep = self._ensure_ripgrep()
        
        if not self._has_ripgrep:
            logger.info("ripgrep not available, using Python fallback for searches")
        
        # Build repo variable map
        self._build_repo_var_map()
    
    def _check_ripgrep(self) -> bool:
        """Check if ripgrep is available."""
        try:
            result = subprocess.run(
                ["rg", "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _ensure_ripgrep(self) -> bool:
        """Attempt to install ripgrep if missing."""
        logger.info("ripgrep not found, attempting to install...")
        
        install_cmds = [
            (["apt-get", "update"], ["apt-get", "install", "-y", "ripgrep"]),
            (None, ["yum", "install", "-y", "ripgrep"]),
            (None, ["dnf", "install", "-y", "ripgrep"]),
            (None, ["brew", "install", "ripgrep"]),
            (None, ["cargo", "install", "ripgrep"]),
        ]
        
        for pre_cmd, install_cmd in install_cmds:
            try:
                if pre_cmd:
                    subprocess.run(pre_cmd, capture_output=True, timeout=60)
                
                result = subprocess.run(
                    install_cmd,
                    capture_output=True,
                    timeout=300,
                )
                
                if result.returncode == 0 and self._check_ripgrep():
                    logger.info(f"ripgrep installed successfully via {install_cmd[0]}")
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
                continue
        
        logger.warning("Could not install ripgrep automatically")
        return False
    
    def _run_python_search(self, pattern: str, search_path: str,
                           file_extensions: List[str] = None,
                           max_results: int = 5) -> List[str]:
        """
        Python-native search fallback using glob + re.
        
        Args:
            pattern: Regex pattern to search for
            search_path: Directory to search in
            file_extensions: List of extensions (e.g., ["py", "cpp"])
            max_results: Maximum results to return
            
        Returns:
            List of matching file paths
        """
        if not Path(search_path).exists():
            return []
        
        results = []
        extensions = file_extensions or ["py", "cpp", "cu", "hip", "hpp"]
        
        try:
            compiled_pattern = re.compile(pattern, re.MULTILINE)
        except re.error:
            pattern = re.escape(pattern)
            compiled_pattern = re.compile(pattern, re.MULTILINE)
        
        for ext in extensions:
            glob_pattern = f"{search_path}/**/*.{ext}"
            for filepath in glob.iglob(glob_pattern, recursive=True):
                if ".git" in filepath or "__pycache__" in filepath:
                    continue
                
                try:
                    with open(filepath, 'r', errors='ignore') as f:
                        content = f.read()
                        if compiled_pattern.search(content):
                            results.append(filepath)
                            if len(results) >= max_results:
                                return results
                except (IOError, OSError):
                    continue
        
        return results
    
    def _search_files(self, pattern: str, search_path: str,
                      file_types: List[str] = None,
                      max_results: int = 5) -> List[str]:
        """
        Search files using ripgrep with Python fallback.
        
        Args:
            pattern: Search pattern
            search_path: Directory to search
            file_types: List of file type filters for ripgrep
            max_results: Maximum results
            
        Returns:
            List of matching file paths
        """
        if self._has_ripgrep:
            results = self._run_ripgrep(pattern, search_path, file_types, max_results)
            if results:
                return results
        
        ext_map = {
            "py": "py",
            "cpp": "cpp",
            "cu": "cu",
            "hip": "hip",
            "hpp": "hpp",
        }
        extensions = [ext_map.get(ft, ft) for ft in (file_types or [])]
        return self._run_python_search(pattern, search_path, extensions, max_results)
    
    def _build_repo_var_map(self):
        """Build mapping from repo variable names to actual paths."""
        for repo_path in self.repos:
            from .repo_config import detect_repo_type, REPO_CONFIGS
            repo_type = detect_repo_type(repo_path)
            if repo_type and repo_type in REPO_CONFIGS:
                config = REPO_CONFIGS[repo_type]
                self._repo_var_map[config.var_name] = repo_path
                
                # Add subproject mappings
                for subvar, (parent_var, subpath) in SUBPROJECT_MAPPINGS.items():
                    if config.var_name == parent_var:
                        self._repo_var_map[subvar] = str(Path(repo_path) / subpath)
    
    def get_repo_paths(self) -> Dict[str, str]:
        """Get mapping of repo variable names to actual paths."""
        return self._repo_var_map.copy()
    
    def search_source(self, parsed: ParsedKernelName) -> Optional[SourceMatch]:
        """
        Search for kernel source file.
        
        Args:
            parsed: Parsed kernel name information
            
        Returns:
            SourceMatch if found, None otherwise
        """
        if parsed.kind == KernelKind.ANNOTATION:
            return None
        
        if parsed.kind == KernelKind.TRITON_JIT:
            return self._search_triton_source(parsed)
        elif parsed.kind == KernelKind.TENSILE_GEMM:
            return self._search_tensile_source(parsed)
        elif parsed.kind == KernelKind.CK_TILE:
            return self._search_ck_source(parsed)
        elif parsed.kind == KernelKind.ATEN_NATIVE:
            return self._search_aten_source(parsed)
        elif parsed.kind == KernelKind.HIP_CPP:
            return self._search_hip_source(parsed)
        elif parsed.kind == KernelKind.INDUCTOR:
            return self._search_inductor_source(parsed)
        elif parsed.kind == KernelKind.AITER:
            return self._search_aiter_source(parsed)
        
        return None
    
    def search_test(self, parsed: ParsedKernelName, source: Optional[SourceMatch] = None) -> Optional[TestMatch]:
        """
        Search for test files and generate test command.
        
        Args:
            parsed: Parsed kernel name information
            source: Optional source match for context
            
        Returns:
            TestMatch if found, None otherwise
        """
        if parsed.kind == KernelKind.ANNOTATION:
            return None
        
        if parsed.kind == KernelKind.TRITON_JIT:
            return self._search_triton_test(parsed, source)
        elif parsed.kind == KernelKind.TENSILE_GEMM:
            return self._search_tensile_test(parsed)
        elif parsed.kind == KernelKind.CK_TILE:
            return self._search_ck_test(parsed)
        elif parsed.kind == KernelKind.ATEN_NATIVE:
            return self._search_aten_test(parsed)
        elif parsed.kind == KernelKind.HIP_CPP:
            return self._search_hip_test(parsed, source)
        elif parsed.kind == KernelKind.AITER:
            return self._search_aiter_test(parsed, source)
        
        return None
    
    def _run_ripgrep(self, pattern: str, search_path: str, 
                     file_types: List[str] = None, max_results: int = 5) -> List[str]:
        """Run ripgrep and return matching files."""
        if not Path(search_path).exists():
            return []
        
        cmd = ["rg", "-l", "--max-count", "1"]
        
        if file_types:
            for ft in file_types:
                cmd.extend(["--type", ft])
        
        cmd.extend([pattern, search_path])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                files = result.stdout.strip().split('\n')
                return [f for f in files if f][:max_results]
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.warning(f"ripgrep failed: {e}")
        
        return []
    
    def _search_triton_source(self, parsed: ParsedKernelName) -> Optional[SourceMatch]:
        """Search for Triton JIT kernel source."""
        function_name = parsed.function_name
        
        # Known kernel mappings for common kernels
        # $TRITON_KERNELS_DIR = triton/python/triton_kernels/
        known_mappings = {
            "_matmul_ogs": ("triton_kernels/matmul_details/_matmul.py", "$TRITON_KERNELS_DIR"),
            "_matmul": ("triton_kernels/matmul_details/_matmul.py", "$TRITON_KERNELS_DIR"),
            "_reduce": ("triton_kernels/reduce.py", "$TRITON_KERNELS_DIR"),
            "kernel_unified_attention": ("vllm/v1/attention/ops/triton_unified_attention.py", "$VLLM_DIR"),
            "_topk_forward": ("triton_kernels/topk_details/_topk_forward.py", "$TRITON_KERNELS_DIR"),
            "_topk_backward": ("triton_kernels/topk_details/_topk_backward.py", "$TRITON_KERNELS_DIR"),
            "_bitmatrix_metadata": ("triton_kernels/tensor_details/", "$TRITON_KERNELS_DIR"),
            "_ragged_tensor_metadata": ("triton_kernels/tensor_details/", "$TRITON_KERNELS_DIR"),
            "_sum_bitmatrix_rows": ("triton_kernels/tensor_details/", "$TRITON_KERNELS_DIR"),
            "_fused_add_rmsnorm": ("triton_kernels/swiglu_details/", "$TRITON_KERNELS_DIR"),
            "_swiglu": ("triton_kernels/swiglu_details/", "$TRITON_KERNELS_DIR"),
            "_compaction": ("triton_kernels/compaction_details/", "$TRITON_KERNELS_DIR"),
        }
        
        # Check known mappings first
        for key, (path, repo_var) in known_mappings.items():
            if key in function_name:
                return SourceMatch(
                    file_path=path,
                    symbol=function_name,
                    repo_name="triton_kernels",
                    repo_var=repo_var,
                )
        
        # Search patterns
        patterns = [
            f"def {function_name}",
            f"@triton.jit.*\\n.*def {function_name}",
            f'def {function_name}\\(',
        ]
        
        # Search in triton repos
        triton_path = self._repo_var_map.get("$TRITON_DIR")
        if triton_path:
            for pattern in patterns:
                files = self._run_ripgrep(pattern, triton_path, ["py"])
                if files:
                    rel_path = os.path.relpath(files[0], triton_path)
                    return SourceMatch(
                        file_path=rel_path,
                        symbol=function_name,
                        repo_name="triton",
                        repo_var="$TRITON_DIR",
                    )
        
        # Search in rocm-libraries (triton_kernels might be there)
        rocm_libs = self._repo_var_map.get("$ROCM_LIBRARIES_DIR")
        if rocm_libs:
            for pattern in patterns:
                files = self._run_ripgrep(pattern, rocm_libs, ["py"])
                if files:
                    rel_path = os.path.relpath(files[0], rocm_libs)
                    return SourceMatch(
                        file_path=rel_path,
                        symbol=function_name,
                        repo_name="rocm-libraries",
                        repo_var="$ROCM_LIBRARIES_DIR",
                    )
        
        # Default fallback for triton kernels
        return SourceMatch(
            file_path="(search in triton_kernels or vllm)",
            symbol=function_name,
            repo_name="triton",
            repo_var="$TRITON_DIR",
        )
    
    def _search_tensile_source(self, parsed: ParsedKernelName) -> Optional[SourceMatch]:
        """Search for Tensile GEMM source (logic YAML files)."""
        rocm_libs = self._repo_var_map.get("$ROCM_LIBRARIES_DIR")
        if not rocm_libs:
            return None
        
        # Tensile kernels are generated, point to logic files
        tensile_logic_path = Path(rocm_libs) / "projects/rocblas/library/src/blas3/Tensile/Logic"
        if tensile_logic_path.exists():
            return SourceMatch(
                file_path="projects/rocblas/library/src/blas3/Tensile/Logic/asm_full/",
                symbol="Tensile-generated kernel (asm)",
                repo_name="rocm-libraries",
                repo_var="$ROCM_LIBRARIES_DIR",
            )
        
        return None
    
    def _search_ck_source(self, parsed: ParsedKernelName) -> Optional[SourceMatch]:
        """Search for Composable Kernel source."""
        rocm_libs = self._repo_var_map.get("$ROCM_LIBRARIES_DIR")
        if not rocm_libs:
            return None
        
        ck_path = Path(rocm_libs) / "projects/composablekernel"
        if not ck_path.exists():
            return None
        
        # Map operation name to directory and kernel file
        op_name = parsed.function_name.lower()
        op_info = {
            "rmsnorm2dfwd": ("rmsnorm2d", "kernel/rmsnorm2d_fwd_kernel.hpp"),
            "rmsnorm": ("rmsnorm2d", "kernel/rmsnorm2d_fwd_kernel.hpp"),
            "fmha": ("fmha", "kernel/fmha_fwd_kernel.hpp"),
            "softmax": ("softmax", "kernel/softmax_kernel.hpp"),
            "gemm": ("gemm", "kernel/gemm_kernel.hpp"),
            "layernorm": ("layernorm2d", "kernel/layernorm2d_fwd_kernel.hpp"),
            "moe": ("moe_sorting_topk", "kernel/moe_sorting_kernel.hpp"),
        }
        
        for op_key, (op_dir, kernel_file) in op_info.items():
            if op_key in op_name:
                # Try specific kernel file first
                kernel_path = f"projects/composablekernel/include/ck_tile/ops/{op_dir}/{kernel_file}"
                if (Path(rocm_libs) / kernel_path).exists():
                    return SourceMatch(
                        file_path=kernel_path,
                        symbol=f"ck_tile::{op_dir}_kernel",
                        repo_name="rocm-libraries",
                        repo_var="$ROCM_LIBRARIES_DIR",
                    )
                # Fall back to directory
                op_path = f"projects/composablekernel/include/ck_tile/ops/{op_dir}/"
                if (Path(rocm_libs) / op_path).exists():
                    return SourceMatch(
                        file_path=op_path,
                        symbol=parsed.function_name,
                        repo_name="rocm-libraries",
                        repo_var="$ROCM_LIBRARIES_DIR",
                    )
        
        # Generic CK search
        return SourceMatch(
            file_path="projects/composablekernel/include/ck_tile/ops/",
            symbol=parsed.function_name,
            repo_name="rocm-libraries",
            repo_var="$ROCM_LIBRARIES_DIR",
        )
    
    def _search_aten_source(self, parsed: ParsedKernelName) -> Optional[SourceMatch]:
        """Search for ATen native kernel source."""
        # ATen kernels are in PyTorch, provide standard path
        kernel_type = parsed.extra.get('kernel_type', '')
        functor = parsed.extra.get('functor', '')
        
        # Map to known files
        file_mapping = {
            'FillFunctor': 'Fill.cu',
            'CompareEqFunctor': 'CompareEQKernel.cu',
            'div_trunc': 'BinaryDivTruncKernel.cu',
            'copy': 'Copy.cu',
            'argmax': 'ReduceArgMaxKernel.cu',
        }
        
        for key, filename in file_mapping.items():
            if key in parsed.original_name:
                return SourceMatch(
                    file_path=f"aten/src/ATen/native/cuda/{filename}",
                    symbol=parsed.function_name,
                    repo_name="pytorch",
                    repo_var="$PYTORCH_DIR",
                )
        
        return SourceMatch(
            file_path="aten/src/ATen/native/cuda/",
            symbol=parsed.function_name,
            repo_name="pytorch",
            repo_var="$PYTORCH_DIR",
        )
    
    def _search_hip_source(self, parsed: ParsedKernelName) -> Optional[SourceMatch]:
        """Search for HIP/CUDA C++ kernel source."""
        namespace = parsed.namespace
        function_name = parsed.function_name
        original_name = parsed.original_name
        
        # Known HIP kernel mappings
        known_hip_mappings = {
            # WMMA / Matrix core kernels (hipBLASLt / rocWMMA)
            "wvSplitK": ("projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/", "hipBLASLt WMMA kernel", "$ROCM_LIBRARIES_DIR"),
            "wvSpltK": ("projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/", "hipBLASLt WMMA kernel", "$ROCM_LIBRARIES_DIR"),
            "DeviceGemmWmma": ("projects/composablekernel/include/ck/tensor_operation/gpu/device/impl/", "CK WMMA GEMM", "$ROCM_LIBRARIES_DIR"),
            # vLLM kernels
            "reshape_and_cache": ("csrc/cache_kernels.cu", "reshape_and_cache_flash_kernel", "$VLLM_DIR"),
            "paged_attention": ("csrc/attention/paged_attention_v1.cu", "paged_attention_kernel", "$VLLM_DIR"),
            "rotary_embedding": ("csrc/pos_encoding_kernels.cu", "rotary_embedding_kernel", "$VLLM_DIR"),
            "rms_norm": ("csrc/layernorm_kernels.cu", "rms_norm_kernel", "$VLLM_DIR"),
            "silu_and_mul": ("csrc/activation_kernels.cu", "silu_and_mul_kernel", "$VLLM_DIR"),
            "gelu": ("csrc/activation_kernels.cu", "gelu_kernel", "$VLLM_DIR"),
            # rocBLAS / BLAS
            "rocblas": ("projects/rocblas/library/src/", "rocBLAS kernel", "$ROCM_LIBRARIES_DIR"),
        }
        
        # Check for ROCm runtime kernels (in rocm-systems super-repo)
        if "__amd_rocclr" in original_name or "rocclr_copy" in original_name:
            return SourceMatch(
                file_path="projects/clr/rocclr/device/blit.cpp",
                symbol="ROCm runtime blit kernel",
                repo_name="rocm-systems",
                repo_var="$ROCM_SYSTEMS_DIR",
            )
        
        # HIP memory copy operations (internal runtime)
        if original_name.startswith("MEMORY_COPY"):
            return SourceMatch(
                file_path="projects/clr/hipamd/src/hip_memory.cpp",
                symbol="HIP memory copy",
                repo_name="rocm-systems",
                repo_var="$ROCM_SYSTEMS_DIR",
            )
        
        # Check known mappings
        for key, (path, symbol, repo_var) in known_hip_mappings.items():
            if key in original_name or key in function_name:
                repo_name = "vllm" if repo_var == "$VLLM_DIR" else "rocm-libraries"
                return SourceMatch(
                    file_path=path,
                    symbol=symbol,
                    repo_name=repo_name,
                    repo_var=repo_var,
                )
        
        # Check vLLM kernels by namespace
        if namespace == "vllm" or "vllm" in original_name.lower():
            return SourceMatch(
                file_path="csrc/",
                symbol=function_name,
                repo_name="vllm",
                repo_var="$VLLM_DIR",
            )
        
        # Search in rocm-libraries
        rocm_libs = self._repo_var_map.get("$ROCM_LIBRARIES_DIR")
        if rocm_libs:
            pattern = f"void.*{function_name}"
            files = self._run_ripgrep(pattern, rocm_libs, ["cpp"])
            if files:
                rel_path = os.path.relpath(files[0], rocm_libs)
                return SourceMatch(
                    file_path=rel_path,
                    symbol=function_name,
                    repo_name="rocm-libraries",
                    repo_var="$ROCM_LIBRARIES_DIR",
                )
        
        return None
    
    def _search_inductor_source(self, parsed: ParsedKernelName) -> Optional[SourceMatch]:
        """Search for torch.inductor generated kernel."""
        return SourceMatch(
            file_path="torch/_inductor/codegen/triton.py",
            symbol=parsed.function_name,
            repo_name="pytorch",
            repo_var="$PYTORCH_DIR",
        )
    
    def _search_triton_test(self, parsed: ParsedKernelName, 
                           source: Optional[SourceMatch]) -> Optional[TestMatch]:
        """Search for Triton kernel tests."""
        function_name = parsed.function_name
        
        # If source is from aiter, use aiter test mappings
        if source and source.repo_name == "aiter":
            aiter_test_mappings = {
                "rmsnorm": ("op_tests/test_rmsnorm2d.py", "cd $AITER_DIR && pytest op_tests/test_rmsnorm2d.py -v"),
                "layernorm": ("op_tests/test_layernorm.py", "cd $AITER_DIR && pytest op_tests/test_layernorm.py -v"),
                "attention": ("op_tests/test_mha.py", "cd $AITER_DIR && pytest op_tests/test_mha.py -v"),
                "mha": ("op_tests/test_mha.py", "cd $AITER_DIR && pytest op_tests/test_mha.py -v"),
                "moe": ("op_tests/test_moe.py", "cd $AITER_DIR && pytest op_tests/test_moe.py -v"),
                "quant": ("op_tests/test_quant.py", "cd $AITER_DIR && pytest op_tests/test_quant.py -v"),
                "gemm": ("op_tests/test_gemm_a8w8.py", "cd $AITER_DIR && pytest op_tests/test_gemm_a8w8.py -v"),
                "rope": ("op_tests/test_rope.py", "cd $AITER_DIR && pytest op_tests/test_rope.py -v"),
            }
            
            fn_lower = function_name.lower()
            for key, (test_file, test_cmd) in aiter_test_mappings.items():
                if key in fn_lower:
                    return TestMatch(
                        test_file=test_file,
                        test_cmd=test_cmd,
                        repo_var="$AITER_DIR",
                    )
            
            # Default aiter test
            return TestMatch(
                test_file="op_tests/",
                test_cmd="cd $AITER_DIR && pytest op_tests/ -v",
                repo_var="$AITER_DIR",
            )
        
        # Known test mappings for common kernels
        # Note: $TRITON_KERNELS_DIR = triton/python/triton_kernels/
        known_test_mappings = {
            "_matmul_ogs": ("tests/test_matmul.py", "cd $TRITON_KERNELS_DIR && pytest tests/test_matmul.py -v", "$TRITON_KERNELS_DIR"),
            "_matmul": ("tests/test_matmul.py", "cd $TRITON_KERNELS_DIR && pytest tests/test_matmul.py -v", "$TRITON_KERNELS_DIR"),
            "_reduce": ("tests/test_reduce.py", "cd $TRITON_KERNELS_DIR && pytest tests/test_reduce.py -v", "$TRITON_KERNELS_DIR"),
            "kernel_unified_attention": ("tests/v1/attention/", "cd $VLLM_DIR && pytest tests/v1/ -v -k attention", "$VLLM_DIR"),
            "_topk_forward": ("tests/test_topk.py", "cd $TRITON_KERNELS_DIR && pytest tests/test_topk.py -v", "$TRITON_KERNELS_DIR"),
            "_topk_backward": ("tests/test_topk.py", "cd $TRITON_KERNELS_DIR && pytest tests/test_topk.py -v", "$TRITON_KERNELS_DIR"),
            "_bitmatrix": ("tests/test_tensor.py", "cd $TRITON_KERNELS_DIR && pytest tests/test_tensor.py -v", "$TRITON_KERNELS_DIR"),
            "_ragged_tensor": ("tests/test_tensor.py", "cd $TRITON_KERNELS_DIR && pytest tests/test_tensor.py -v", "$TRITON_KERNELS_DIR"),
            "_sum_bitmatrix_rows": ("tests/test_tensor.py", "cd $TRITON_KERNELS_DIR && pytest tests/test_tensor.py -v", "$TRITON_KERNELS_DIR"),
            "_fused_add_rmsnorm": ("tests/test_swiglu.py", "cd $TRITON_KERNELS_DIR && pytest tests/test_swiglu.py -v", "$TRITON_KERNELS_DIR"),
            "_swiglu": ("tests/test_swiglu.py", "cd $TRITON_KERNELS_DIR && pytest tests/test_swiglu.py -v", "$TRITON_KERNELS_DIR"),
            "_compaction": ("tests/test_compaction.py", "cd $TRITON_KERNELS_DIR && pytest tests/test_compaction.py -v", "$TRITON_KERNELS_DIR"),
        }
        
        # Check known mappings first
        for key, (test_file, test_cmd, repo_var) in known_test_mappings.items():
            if key in function_name:
                return TestMatch(
                    test_file=test_file,
                    test_cmd=test_cmd,
                    repo_var=repo_var,
                )
        
        triton_path = self._repo_var_map.get("$TRITON_DIR")
        if triton_path:
            test_path = Path(triton_path) / "python/test/unit/language"
            if test_path.exists():
                # Search for test files mentioning the function
                pattern = f"def test.*{function_name}|{function_name}"
                files = self._run_ripgrep(pattern, str(test_path), ["py"])
                if files:
                    rel_path = os.path.relpath(files[0], triton_path)
                    return TestMatch(
                        test_file=rel_path,
                        test_cmd=f"cd $TRITON_DIR && pytest {rel_path} -q",
                        repo_var="$TRITON_DIR",
                    )
        
        return None
    
    def _search_tensile_test(self, parsed: ParsedKernelName) -> Optional[TestMatch]:
        """Search for Tensile GEMM tests."""
        rocm_libs = self._repo_var_map.get("$ROCM_LIBRARIES_DIR")
        if rocm_libs:
            return TestMatch(
                test_file="projects/rocblas/clients/gtest/blas3/gemm_gtest.cpp",
                test_cmd="cd $ROCM_LIBRARIES_DIR/projects/rocblas/build/release && ./clients/staging/rocblas-bench -f gemm_ex --a_type bf16_r --b_type bf16_r --compute_type f32_r",
                repo_var="$ROCM_LIBRARIES_DIR",
            )
        return None
    
    def _search_ck_test(self, parsed: ParsedKernelName) -> Optional[TestMatch]:
        """Search for CK tile tests."""
        op_name = parsed.function_name.lower()
        original_name = parsed.original_name.lower()
        
        # Map operation to example/test directory
        # CK examples are at: projects/composablekernel/example/ck_tile/
        if "rmsnorm" in op_name or "rmsnorm" in original_name:
            return TestMatch(
                test_file="projects/composablekernel/example/ck_tile/10_rmsnorm2d/",
                test_cmd="cd $ROCM_LIBRARIES_DIR/projects/composablekernel/build && cmake --build . -j --target tile_example_rmsnorm2d_fwd && ./bin/tile_example_rmsnorm2d_fwd -m 1024 -n 2048",
                repo_var="$ROCM_LIBRARIES_DIR",
            )
        elif "fmha" in op_name or "fmha" in original_name:
            return TestMatch(
                test_file="projects/composablekernel/example/ck_tile/01_fmha/",
                test_cmd="cd $ROCM_LIBRARIES_DIR/projects/composablekernel/build && cmake --build . -j --target tile_example_fmha_fwd && ./bin/tile_example_fmha_fwd",
                repo_var="$ROCM_LIBRARIES_DIR",
            )
        elif "layernorm" in op_name or "layernorm" in original_name:
            return TestMatch(
                test_file="projects/composablekernel/example/ck_tile/02_layernorm2d/",
                test_cmd="cd $ROCM_LIBRARIES_DIR/projects/composablekernel/build && cmake --build . -j --target tile_example_layernorm2d_fwd && ./bin/tile_example_layernorm2d_fwd",
                repo_var="$ROCM_LIBRARIES_DIR",
            )
        elif "gemm" in op_name or "gemm" in original_name:
            return TestMatch(
                test_file="projects/composablekernel/example/ck_tile/03_gemm/",
                test_cmd="cd $ROCM_LIBRARIES_DIR/projects/composablekernel/build && cmake --build . -j --target tile_example_gemm && ./bin/tile_example_gemm",
                repo_var="$ROCM_LIBRARIES_DIR",
            )
        elif "topk" in op_name or "softmax" in op_name:
            return TestMatch(
                test_file="projects/composablekernel/example/ck_tile/09_topk_softmax/",
                test_cmd="cd $ROCM_LIBRARIES_DIR/projects/composablekernel/build && cmake --build . -j --target tile_example_topk_softmax && ./bin/tile_example_topk_softmax",
                repo_var="$ROCM_LIBRARIES_DIR",
            )
        
        return None
    
    def _search_aten_test(self, parsed: ParsedKernelName) -> Optional[TestMatch]:
        """Search for ATen native tests."""
        # Map operation to test file
        test_mapping = {
            'fill': 'test_torch.py',
            'eq': 'test_binary_ufuncs.py',
            'div': 'test_binary_ufuncs.py',
            'copy': 'test_torch.py',
            'argmax': 'test_reductions.py',
        }
        
        for op, test_file in test_mapping.items():
            if op in parsed.function_name.lower():
                return TestMatch(
                    test_file=f"test/{test_file}",
                    test_cmd=f"pytest $PYTORCH_DIR/test/{test_file} -q -k {op}",
                    repo_var="$PYTORCH_DIR",
                )
        
        return TestMatch(
            test_file="test/test_torch.py",
            test_cmd="pytest $PYTORCH_DIR/test/test_torch.py -q",
            repo_var="$PYTORCH_DIR",
        )
    
    def _search_hip_test(self, parsed: ParsedKernelName,
                        source: Optional[SourceMatch]) -> Optional[TestMatch]:
        """Search for HIP/CUDA kernel tests."""
        namespace = parsed.namespace
        function_name = parsed.function_name
        original_name = parsed.original_name
        
        # Known HIP kernel test mappings
        known_hip_test_mappings = {
            # WMMA / Matrix core kernels (hipBLASLt)
            "wvSplitK": ("projects/hipblaslt/clients/tests/", "cd $ROCM_LIBRARIES_DIR/projects/hipblaslt/build/release && ./clients/staging/hipblaslt-bench -f gemm_ex --a_type bf16_r --b_type bf16_r --compute_type f32_r -m 1024 -n 1024 -k 1024", "$ROCM_LIBRARIES_DIR"),
            "wvSpltK": ("projects/hipblaslt/clients/tests/", "cd $ROCM_LIBRARIES_DIR/projects/hipblaslt/build/release && ./clients/staging/hipblaslt-bench -f gemm_ex --a_type bf16_r --b_type bf16_r --compute_type f32_r -m 1024 -n 1024 -k 1024", "$ROCM_LIBRARIES_DIR"),
            # vLLM kernels
            "reshape_and_cache": ("tests/kernels/test_cache.py", "cd $VLLM_DIR && pytest tests/kernels/test_cache.py -v", "$VLLM_DIR"),
            "paged_attention": ("tests/kernels/test_attention.py", "cd $VLLM_DIR && pytest tests/kernels/test_attention.py -v", "$VLLM_DIR"),
            "rotary_embedding": ("tests/kernels/test_pos_encoding.py", "cd $VLLM_DIR && pytest tests/kernels/test_pos_encoding.py -v", "$VLLM_DIR"),
            "rms_norm": ("tests/kernels/test_layernorm.py", "cd $VLLM_DIR && pytest tests/kernels/test_layernorm.py -v", "$VLLM_DIR"),
            "silu_and_mul": ("tests/kernels/test_activation.py", "cd $VLLM_DIR && pytest tests/kernels/test_activation.py -v", "$VLLM_DIR"),
        }
        
        # Check for ROCm runtime kernels (in rocm-systems)
        if "__amd_rocclr" in original_name or "rocclr_copy" in original_name or original_name.startswith("MEMORY_COPY"):
            return TestMatch(
                test_file="projects/hip-tests/catch/unit/memory/",
                test_cmd="cd $ROCM_SYSTEMS_DIR && ctest -R hipMemcpy",
                repo_var="$ROCM_SYSTEMS_DIR",
            )
        
        # Check known mappings
        for key, (test_file, test_cmd, repo_var) in known_hip_test_mappings.items():
            if key in original_name or key in function_name:
                return TestMatch(
                    test_file=test_file,
                    test_cmd=test_cmd,
                    repo_var=repo_var,
                )
        
        if namespace == "vllm" or "vllm" in original_name.lower():
            # Map to vLLM test directories
            if "cache" in function_name.lower():
                return TestMatch(
                    test_file="tests/kernels/attention/test_cache.py",
                    test_cmd="cd $VLLM_DIR && pytest tests/kernels/attention/test_cache.py -q",
                    repo_var="$VLLM_DIR",
                )
            return TestMatch(
                test_file="tests/kernels/",
                test_cmd="cd $VLLM_DIR && pytest tests/kernels/ -q",
                repo_var="$VLLM_DIR",
            )
        
        return None
    
    def _search_aiter_source(self, parsed: ParsedKernelName) -> Optional[SourceMatch]:
        """Search for aiter kernel source."""
        function_name = parsed.function_name
        original_name = parsed.original_name
        extra = parsed.extra or {}
        category = extra.get('category', '')
        
        # Known aiter kernel mappings
        known_mappings = {
            # Quantization kernels
            'dynamic_per_group_scaled_quant': ('aiter/ops/quant.py', 'dynamic_per_group_scaled_quant'),
            'dynamic_per_token_scaled_quant': ('aiter/ops/quant.py', 'dynamic_per_token_scaled_quant'),
            'group_fp8_quant': ('aiter/ops/quant.py', 'group_fp8_quant'),
            # MoE kernels
            'fmoe': ('aiter/fused_moe.py', 'fused_moe'),
            'moe_sorting': ('aiter/ops/moe_sorting.py', 'moe_sorting'),
            'moe_align': ('csrc/kernels/moe_align_block_size_kernels.cu', 'moe_align'),
            # GEMM kernels
            'gemm_a8w8': ('aiter/ops/gemm_op_a8w8.py', 'gemm_a8w8'),
            'gemm_a4w4': ('aiter/ops/gemm_op_a4w4.py', 'gemm_a4w4'),
            'batched_gemm': ('aiter/ops/batched_gemm_op_a8w8.py', 'batched_gemm'),
            # Attention kernels
            'mha': ('aiter/ops/mha.py', 'mha'),
            'mla': ('aiter/mla.py', 'mla'),
            'paged_attention': ('aiter/paged_attn.py', 'paged_attn'),
            # Norm kernels
            'rmsnorm': ('aiter/ops/rmsnorm.py', 'rmsnorm'),
            'groupnorm': ('aiter/ops/groupnorm.py', 'groupnorm'),
            # Rope kernels
            'rotary': ('aiter/rotary_embedding.py', 'rotary_embedding'),
            'rope': ('aiter/ops/rope.py', 'rope'),
        }
        
        # Check known mappings
        for key, (path, symbol) in known_mappings.items():
            if key in function_name.lower() or key in original_name.lower():
                return SourceMatch(
                    file_path=path,
                    symbol=symbol,
                    repo_name="aiter",
                    repo_var="$AITER_DIR",
                )
        
        # Fall back based on category
        category_paths = {
            'quant': 'aiter/ops/quant.py',
            'moe': 'aiter/fused_moe.py',
            'gemm': 'aiter/ops/gemm_op_a8w8.py',
            'attention': 'aiter/ops/mha.py',
            'norm': 'aiter/ops/rmsnorm.py',
        }
        
        if category in category_paths:
            return SourceMatch(
                file_path=category_paths[category],
                symbol=function_name,
                repo_name="aiter",
                repo_var="$AITER_DIR",
            )
        
        # Default: search in aiter/ops
        return SourceMatch(
            file_path="aiter/ops/",
            symbol=function_name,
            repo_name="aiter",
            repo_var="$AITER_DIR",
        )
    
    def _search_aiter_test(self, parsed: ParsedKernelName,
                           source: Optional[SourceMatch]) -> Optional[TestMatch]:
        """Search for aiter kernel tests."""
        function_name = parsed.function_name
        original_name = parsed.original_name
        extra = parsed.extra or {}
        category = extra.get('category', '')
        
        # Known test mappings
        test_mappings = {
            'quant': ('op_tests/test_quant.py', 'quant'),
            'moe': ('op_tests/test_moe.py', 'moe'),
            'gemm': ('op_tests/test_gemm_a8w8.py', 'gemm'),
            'attention': ('op_tests/test_mha.py', 'mha'),
            'norm': ('op_tests/test_rmsnorm2d.py', 'rmsnorm'),
            'rope': ('op_tests/test_rope.py', 'rope'),
        }
        
        # Check by category
        if category in test_mappings:
            test_file, keyword = test_mappings[category]
            return TestMatch(
                test_file=test_file,
                test_cmd=f"cd $AITER_DIR && pytest {test_file} -v -k {keyword}",
                repo_var="$AITER_DIR",
            )
        
        # Check by keywords in function name
        if 'quant' in function_name.lower():
            return TestMatch(
                test_file="op_tests/test_quant.py",
                test_cmd="cd $AITER_DIR && pytest op_tests/test_quant.py -v",
                repo_var="$AITER_DIR",
            )
        elif 'moe' in function_name.lower() or 'fmoe' in function_name.lower():
            return TestMatch(
                test_file="op_tests/test_moe.py",
                test_cmd="cd $AITER_DIR && pytest op_tests/test_moe.py -v",
                repo_var="$AITER_DIR",
            )
        elif 'gemm' in function_name.lower():
            return TestMatch(
                test_file="op_tests/test_gemm_a8w8.py",
                test_cmd="cd $AITER_DIR && pytest op_tests/test_gemm_a8w8.py -v",
                repo_var="$AITER_DIR",
            )
        elif 'mha' in function_name.lower() or 'attention' in function_name.lower():
            return TestMatch(
                test_file="op_tests/test_mha.py",
                test_cmd="cd $AITER_DIR && pytest op_tests/test_mha.py -v",
                repo_var="$AITER_DIR",
            )
        
        # Default test
        return TestMatch(
            test_file="op_tests/",
            test_cmd="cd $AITER_DIR && pytest op_tests/ -v",
            repo_var="$AITER_DIR",
        )
