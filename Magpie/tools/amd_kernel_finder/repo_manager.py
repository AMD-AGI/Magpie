###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Repository manager for auto-cloning kernel source repositories.

Automatically clones missing repositories needed for kernel source searching.
Supports shallow cloning for faster downloads.
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


REPO_URLS = {
    "rocm-libraries": "https://github.com/ROCm/rocm-libraries.git",
    "triton": "https://github.com/triton-lang/triton.git",
    "vllm": "https://github.com/vllm-project/vllm.git",
    "pytorch": "https://github.com/pytorch/pytorch.git",
    "rocm-systems": "https://github.com/ROCm/rocm-systems.git",  # ROCm super-repo (clr, hip, rocprofiler, etc)
    "aiter": "https://github.com/ROCm/aiter.git",
}

KERNEL_REPO_MAP = {
    "triton_jit": ["triton"],
    "ck_tile": ["rocm-libraries", "aiter"],  # CK Tile has aiter wrapper
    "tensile_gemm": ["rocm-libraries"],
    "hip_cpp": ["rocm-libraries", "vllm"],  # HIP kernels may be in vllm
    "aten_native": ["pytorch"],
    "inductor": ["pytorch"],
    "aiter": ["aiter"],
}

# All repos to clone when force_all is True
ALL_REPOS = ["rocm-libraries", "triton", "vllm", "pytorch", "aiter", "rocm-systems"]



class RepoManager:
    """
    Manage repository cloning and updates.
    
    Automatically clones missing repositories to a cache directory,
    using shallow clones by default for faster downloads.
    """
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the repository manager.
        
        Args:
            base_dir: Directory to store cloned repos. 
                     Defaults to ~/.cache/magpie/repos/
        """
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = Path.home() / ".cache" / "magpie" / "repos"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self._cloned_repos: Dict[str, str] = {}
    
    def ensure_repo(self, repo_name: str, shallow: bool = True) -> str:
        """
        Ensure a repository is cloned and return its path.
        
        Args:
            repo_name: Name of the repository (e.g., "rocm-libraries")
            shallow: If True, use shallow clone (--depth 1) for speed
            
        Returns:
            Path to the cloned repository
            
        Raises:
            ValueError: If repo_name is unknown
            RuntimeError: If clone fails
        """
        if repo_name in self._cloned_repos:
            return self._cloned_repos[repo_name]
        
        repo_path = self.base_dir / repo_name
        
        if repo_path.exists() and (repo_path / ".git").exists():
            logger.info(f"Repo {repo_name} already exists at {repo_path}")
            self._cloned_repos[repo_name] = str(repo_path)
            return str(repo_path)
        
        if repo_name not in REPO_URLS:
            raise ValueError(f"Unknown repository: {repo_name}. "
                           f"Known repos: {list(REPO_URLS.keys())}")
        
        url = REPO_URLS[repo_name]
        logger.info(f"Cloning {repo_name} from {url}...")
        
        cmd = ["git", "clone"]
        if shallow:
            cmd.extend(["--depth", "1"])
        cmd.extend([url, str(repo_path)])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Clone failed: {result.stderr}")
            
            logger.info(f"Successfully cloned {repo_name} to {repo_path}")
            self._cloned_repos[repo_name] = str(repo_path)
            return str(repo_path)
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Clone timed out for {repo_name}")
        except Exception as e:
            raise RuntimeError(f"Clone failed for {repo_name}: {e}")
    
    def get_required_repos(self, kernel_kinds: List[str]) -> List[str]:
        """
        Determine which repos are needed for given kernel kinds.
        
        Args:
            kernel_kinds: List of kernel kinds (e.g., ["triton_jit", "ck_tile"])
            
        Returns:
            List of repo names needed
        """
        repos = set()
        for kind in kernel_kinds:
            if kind in KERNEL_REPO_MAP:
                repos.update(KERNEL_REPO_MAP[kind])
        return list(repos)
    
    def get_repos_for_kernels(self, kernel_names: List[str], force_all: bool = False) -> List[str]:
        """
        Analyze kernel names and determine which repos to clone.
        
        Args:
            kernel_names: List of kernel names from profiler
            force_all: If True, return all repos regardless of kernel types
            
        Returns:
            List of repo names needed
        """
        if force_all:
            return ALL_REPOS.copy()
        
        from .parser import KernelNameParser
        
        parser = KernelNameParser()
        kinds = set()
        has_vllm = False
        
        for name in kernel_names:
            parsed = parser.parse(name)
            kinds.add(parsed.kind.value)
            
            if "vllm::" in name or "vllm" in name.lower():
                has_vllm = True
        
        repos = set()
        for kind in kinds:
            if kind in KERNEL_REPO_MAP:
                repos.update(KERNEL_REPO_MAP[kind])
        
        if has_vllm:
            repos.add("vllm")
        
        return list(repos)
    
    def ensure_all_repos(self, shallow: bool = True) -> List[str]:
        """
        Ensure all known repositories are cloned.
        
        Args:
            shallow: If True, use shallow clone
            
        Returns:
            List of repo paths
        """
        logger.info(f"Cloning all repos: {ALL_REPOS}")
        
        paths = []
        for repo_name in ALL_REPOS:
            try:
                path = self.ensure_repo(repo_name, shallow=shallow)
                paths.append(path)
            except Exception as e:
                logger.warning(f"Failed to clone {repo_name}: {e}")
        
        return paths
    
    def ensure_repos_for_kernels(self, kernel_names: List[str], 
                                  shallow: bool = True) -> List[str]:
        """
        Ensure all repos needed for the given kernels are available.
        
        Args:
            kernel_names: List of kernel names from profiler
            shallow: If True, use shallow clone
            
        Returns:
            List of repo paths
        """
        needed_repos = self.get_repos_for_kernels(kernel_names)
        logger.info(f"Kernel analysis requires repos: {needed_repos}")
        
        paths = []
        for repo_name in needed_repos:
            try:
                path = self.ensure_repo(repo_name, shallow=shallow)
                paths.append(path)
            except Exception as e:
                logger.warning(f"Failed to ensure repo {repo_name}: {e}")
        
        return paths
    
    def get_repo_path(self, repo_name: str) -> Optional[str]:
        """Get path to a repo if it exists."""
        if repo_name in self._cloned_repos:
            return self._cloned_repos[repo_name]
        
        repo_path = self.base_dir / repo_name
        if repo_path.exists():
            self._cloned_repos[repo_name] = str(repo_path)
            return str(repo_path)
        
        return None
    
    def list_available_repos(self) -> List[str]:
        """List all available (cloned) repositories."""
        available = []
        for repo_name in REPO_URLS:
            repo_path = self.base_dir / repo_name
            if repo_path.exists():
                available.append(repo_name)
        return available
    
    def update_repo(self, repo_name: str) -> bool:
        """
        Update (git pull) a repository.
        
        Args:
            repo_name: Name of the repository
            
        Returns:
            True if update succeeded
        """
        repo_path = self.get_repo_path(repo_name)
        if not repo_path:
            logger.warning(f"Repo {repo_name} not found, cannot update")
            return False
        
        try:
            result = subprocess.run(
                ["git", "pull", "--ff-only"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=120,
            )
            
            if result.returncode == 0:
                logger.info(f"Updated {repo_name}")
                return True
            else:
                logger.warning(f"Update failed for {repo_name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.warning(f"Update failed for {repo_name}: {e}")
            return False
