###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Repository configuration for kernel source finder.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class RepoConfig:
    """Configuration for a single repository."""
    
    name: str
    var_name: str  # Environment variable name, e.g., $ROCM_LIBRARIES_DIR
    path: str = ""
    github_base: str = ""
    
    # Source search paths by kernel kind
    source_paths: Dict[str, List[str]] = field(default_factory=dict)
    
    # Test search paths by kernel kind
    test_paths: Dict[str, List[str]] = field(default_factory=dict)
    
    # File patterns to search
    source_patterns: List[str] = field(default_factory=lambda: ["*.py", "*.cpp", "*.cu", "*.hip", "*.hpp"])
    test_patterns: List[str] = field(default_factory=lambda: ["test_*.py", "*_test.py", "*_gtest.cpp"])


# Default repository configurations
REPO_CONFIGS = {
    "rocm-libraries": RepoConfig(
        name="rocm-libraries",
        var_name="$ROCM_LIBRARIES_DIR",
        github_base="https://github.com/ROCm/rocm-libraries",
        source_paths={
            "ck_tile": [
                "projects/composablekernel/include/ck_tile/ops/",
                "projects/composablekernel/library/",
            ],
            "tensile": [
                "shared/tensile/Tensile/",
                "projects/rocblas/library/src/blas3/Tensile/",
            ],
            "rocblas": [
                "projects/rocblas/library/src/",
            ],
            "miopen": [
                "projects/miopen/src/solver/",
                "projects/miopen/src/kernels/",
            ],
            "hipblaslt": [
                "projects/hipblaslt/library/src/",
            ],
        },
        test_paths={
            "ck_tile": [
                "projects/composablekernel/test/ck_tile/",
                "projects/composablekernel/example/ck_tile/",
            ],
            "tensile": [
                "shared/tensile/Tensile/Tests/",
            ],
            "rocblas": [
                "projects/rocblas/clients/gtest/",
            ],
            "miopen": [
                "projects/miopen/test/",
            ],
        },
    ),
    "triton": RepoConfig(
        name="triton",
        var_name="$TRITON_DIR",
        github_base="https://github.com/triton-lang/triton",
        source_paths={
            "triton_jit": [
                "python/triton/",
                "python/tutorials/",
                "third_party/amd/",
            ],
        },
        test_paths={
            "triton_jit": [
                "python/test/unit/language/",
                "python/test/unit/",
            ],
        },
    ),
    "rocm-systems": RepoConfig(
        name="rocm-systems",
        var_name="$ROCM_SYSTEMS_DIR",
        github_base="https://github.com/ROCm/rocm-systems",
        source_paths={
            "hip": [
                "projects/hip/",
                "projects/clr/",
            ],
            "rccl": [
                "projects/rccl/src/device/",
            ],
        },
        test_paths={
            "hip": [
                "projects/hip-tests/catch/",
            ],
            "rccl": [
                "projects/rccl-tests/",
            ],
        },
    ),
    "aiter": RepoConfig(
        name="aiter",
        var_name="$AITER_DIR",
        github_base="https://github.com/ROCm/aiter",
        source_paths={
            "aiter_quant": [
                "aiter/ops/quant.py",
                "aiter/ops/triton/quant/",
                "csrc/kernels/quant_kernels.cu",
            ],
            "aiter_moe": [
                "aiter/ops/moe_op.py",
                "aiter/fused_moe.py",
                "aiter/ops/triton/moe/",
                "csrc/kernels/moe_fused_gate.cu",
                "csrc/kernels/moe_align_block_size_kernels.cu",
                "csrc/ck_tile_gemm_moe_2stages/",
            ],
            "aiter_gemm": [
                "aiter/ops/gemm_op_a8w8.py",
                "aiter/ops/gemm_op_a4w4.py",
                "aiter/ops/batched_gemm_op_a8w8.py",
                "aiter/ops/triton/gemm/",
                "csrc/ck_gemm_a8w8/",
                "csrc/ck_gemm_a8w8_blockscale/",
                "csrc/ck_gemm_a4w4_blockscale/",
            ],
            "aiter_attention": [
                "aiter/ops/attention.py",
                "aiter/ops/mha.py",
                "aiter/paged_attn.py",
                "aiter/mla.py",
                "aiter/ops/triton/attention/",
                "csrc/kernels/attention.cu",
                "csrc/kernels/mla/",
            ],
            "aiter_triton": [
                "aiter/ops/triton/",
                "aiter/ops/triton/_triton_kernels/",
            ],
            "aiter_rope": [
                "aiter/ops/rope.py",
                "aiter/rotary_embedding.py",
                "aiter/ops/triton/rope/",
                "csrc/kernels/rope/",
                "csrc/kernels/pos_encoding_kernels.cu",
            ],
            "aiter_norm": [
                "aiter/ops/norm.py",
                "aiter/ops/rmsnorm.py",
                "aiter/ops/triton/normalization/",
                "csrc/kernels/rmsnorm_kernels.cu",
                "csrc/kernels/groupnorm.cu",
            ],
            "aiter_cache": [
                "aiter/ops/cache.py",
                "csrc/kernels/cache_kernels.cu",
            ],
            "aiter_jit": [
                "aiter/jit/",
            ],
        },
        test_paths={
            "aiter_quant": [
                "op_tests/test_quant.py",
                "op_tests/test_smoothquant.py",
            ],
            "aiter_moe": [
                "op_tests/test_moe.py",
                "op_tests/test_moe_*.py",
            ],
            "aiter_gemm": [
                "op_tests/test_gemm_*.py",
                "op_tests/test_batched_gemm_*.py",
            ],
            "aiter_attention": [
                "op_tests/test_mha*.py",
                "op_tests/test_pa*.py",
                "op_tests/test_mla*.py",
                "op_tests/test_batch_prefill.py",
            ],
            "aiter_rope": [
                "op_tests/test_rope.py",
            ],
            "aiter_norm": [
                "op_tests/test_rmsnorm*.py",
                "op_tests/test_layernorm*.py",
                "op_tests/test_groupnorm.py",
            ],
            "aiter_cache": [
                "op_tests/test_kvcache*.py",
            ],
            "aiter_triton": [
                "op_tests/triton_tests/",
            ],
        },
    ),
}


# Subproject mappings (for repos within repos)
SUBPROJECT_MAPPINGS = {
    "$CK_DIR": ("$ROCM_LIBRARIES_DIR", "projects/composablekernel"),
    "$ROCBLAS_DIR": ("$ROCM_LIBRARIES_DIR", "projects/rocblas"),
    "$MIOPEN_DIR": ("$ROCM_LIBRARIES_DIR", "projects/miopen"),
    "$TENSILE_DIR": ("$ROCM_LIBRARIES_DIR", "shared/tensile"),
    "$HIPBLASLT_DIR": ("$ROCM_LIBRARIES_DIR", "projects/hipblaslt"),
    "$TRITON_KERNELS_DIR": ("$TRITON_DIR", "python/triton_kernels"),
    "$VLLM_DIR": ("$VLLM_DIR", ""),  # vllm is its own repo
    "$PYTORCH_DIR": ("$PYTORCH_DIR", ""),  # pytorch is its own repo
    "$AITER_DIR": ("$AITER_DIR", ""),  # aiter is its own repo
    "$ROCM_SYSTEMS_DIR": ("$ROCM_SYSTEMS_DIR", ""),  # rocm-systems super-repo (clr, hip, rocprofiler, etc)
}


# GitHub URL templates
GITHUB_URL_TEMPLATES = {
    "rocm-libraries": "https://github.com/ROCm/rocm-libraries/blob/main/{path}",
    "triton": "https://github.com/triton-lang/triton/blob/main/{path}",
    "rocm-systems": "https://github.com/ROCm/rocm-systems/blob/develop/{path}",  # ROCm super-repo (clr, hip, rocprofiler, etc)
    "vllm": "https://github.com/vllm-project/vllm/blob/main/{path}",
    "pytorch": "https://github.com/pytorch/pytorch/blob/main/{path}",
    "aiter": "https://github.com/ROCm/aiter/blob/main/{path}",
}


def detect_repo_type(repo_path: str) -> Optional[str]:
    """Detect the type of repository based on directory structure."""
    path = Path(repo_path)
    
    if not path.exists():
        return None
    
    # Check for rocm-libraries
    if (path / "projects" / "composablekernel").exists():
        return "rocm-libraries"
    
    # Check for triton
    if (path / "python" / "triton").exists():
        return "triton"
    
    # Check for rocm-systems
    if (path / "projects" / "hip").exists() or (path / "projects" / "rccl").exists():
        return "rocm-systems"
    
    # Check for aiter
    if (path / "aiter").exists() and (path / "aiter" / "ops").exists() and (path / "csrc" / "kernels").exists():
        return "aiter"
    
    # Check for vllm
    if (path / "vllm").exists() and (path / "csrc").exists():
        return "vllm"
    
    # Check for pytorch
    if (path / "aten").exists() and (path / "torch").exists():
        return "pytorch"
    
    return None


def get_repo_config(repo_path: str) -> Optional[RepoConfig]:
    """Get repository configuration for a given path."""
    repo_type = detect_repo_type(repo_path)
    
    if repo_type and repo_type in REPO_CONFIGS:
        config = REPO_CONFIGS[repo_type]
        config.path = repo_path
        return config
    
    return None


# Test command templates
TEST_CMD_TEMPLATES = {
    "triton_jit": "cd {repo_path} && pytest {test_file} -q -k {function}",
    "tensile_gemm": "cd {repo_path}/build/release && ./clients/staging/rocblas-bench -f gemm_ex --a_type bf16_r --b_type bf16_r --compute_type f32_r",
    "ck_tile": "cd {repo_path}/build && cmake --build . -j --target tile_example_{op}_fwd && ./bin/tile_example_{op}_fwd",
    "aten_native": "pytest {repo_path}/test/test_{module}.py -q",
    "hip_cpp": "cd {repo_path} && pytest tests/kernels/ -q",
    "aiter": "cd {repo_path} && pytest op_tests/{test_file} -q -k {function}",
}


class RepoDiscovery:
    """
    Auto-discover repository structure.
    
    Scans a repository to find key directories and files,
    adapting to different repo layouts.
    """
    
    MARKER_FILES = {
        "triton_kernels": [
            "triton_kernels/__init__.py",
            "triton_kernels/matmul.py",
        ],
        "ck_tile": [
            "include/ck_tile/ops",
            "example/ck_tile",
        ],
        "hipblaslt": [
            "library/src/amd_detail",
            "clients/tests",
        ],
        "vllm": [
            "vllm/__init__.py",
            "csrc/attention",
        ],
        "pytorch_aten": [
            "aten/src/ATen/native/cuda",
        ],
        "aiter": [
            "aiter/__init__.py",
            "aiter/ops",
            "csrc/kernels",
            "op_tests",
        ],
    }
    
    @classmethod
    def discover(cls, repo_path: str) -> Dict[str, str]:
        """
        Discover structure and return path mappings.
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Dictionary mapping component names to their paths
        """
        structure = {}
        path = Path(repo_path)
        
        if not path.exists():
            return structure
        
        # Find triton_kernels location
        for candidate in path.rglob("triton_kernels/__init__.py"):
            parent = candidate.parent.parent
            structure["triton_kernels"] = str(parent)
            structure["triton_kernels_tests"] = str(parent / "tests")
            break
        
        # Find CK ops location
        for candidate in path.rglob("ck_tile/ops"):
            if "include" in str(candidate):
                structure["ck_ops"] = str(candidate)
                ck_root = candidate.parent.parent.parent
                structure["ck_examples"] = str(ck_root / "example" / "ck_tile")
                break
        
        # Find hipBLASLt
        for candidate in path.rglob("hipblaslt"):
            if (candidate / "library").exists():
                structure["hipblaslt"] = str(candidate)
                structure["hipblaslt_tests"] = str(candidate / "clients" / "tests")
                break
        
        # Find vLLM csrc
        for candidate in path.rglob("csrc"):
            if (candidate / "attention").exists() or (candidate / "cache_kernels.cu").exists():
                structure["vllm_csrc"] = str(candidate)
                vllm_root = candidate.parent
                structure["vllm_tests"] = str(vllm_root / "tests" / "kernels")
                break
        
        # Find PyTorch ATen
        for candidate in path.rglob("ATen/native/cuda"):
            structure["aten_cuda"] = str(candidate)
            pytorch_root = candidate.parent.parent.parent.parent.parent
            structure["pytorch_tests"] = str(pytorch_root / "test")
            break
        
        # Find aiter
        for candidate in path.rglob("aiter/ops"):
            if candidate.is_dir() and (candidate / "triton").exists():
                aiter_root = candidate.parent.parent
                structure["aiter_ops"] = str(candidate)
                structure["aiter_triton"] = str(candidate / "triton")
                structure["aiter_csrc"] = str(aiter_root / "csrc" / "kernels")
                structure["aiter_ck"] = str(aiter_root / "csrc")
                structure["aiter_tests"] = str(aiter_root / "op_tests")
                break
        
        return structure
    
    @classmethod
    def find_tests_dir(cls, repo_path: str, component: str = None) -> Optional[str]:
        """
        Find test directory in a repository.
        
        Args:
            repo_path: Path to repository
            component: Optional component name to find tests for
            
        Returns:
            Path to test directory if found
        """
        path = Path(repo_path)
        
        # Common test directory patterns
        test_patterns = [
            "tests",
            "test",
            "python/*/tests",
            "clients/tests",
            "example",
        ]
        
        for pattern in test_patterns:
            for candidate in path.glob(pattern):
                if candidate.is_dir():
                    return str(candidate)
        
        return None
    
    @classmethod
    def find_source_dir(cls, repo_path: str, kernel_kind: str) -> Optional[str]:
        """
        Find source directory for a kernel kind.
        
        Args:
            repo_path: Path to repository
            kernel_kind: Kind of kernel (triton_jit, ck_tile, etc.)
            
        Returns:
            Path to source directory if found
        """
        structure = cls.discover(repo_path)
        
        kind_to_component = {
            "triton_jit": ["triton_kernels", "aiter_triton"],
            "ck_tile": ["ck_ops", "aiter_ck"],
            "hip_cpp": ["hipblaslt", "vllm_csrc", "aiter_csrc"],
            "aten_native": "aten_cuda",
            "aiter": ["aiter_ops", "aiter_triton", "aiter_csrc", "aiter_ck"],
        }
        
        components = kind_to_component.get(kernel_kind, [])
        if isinstance(components, str):
            components = [components]
        
        for comp in components:
            if comp in structure:
                return structure[comp]
        
        return None
