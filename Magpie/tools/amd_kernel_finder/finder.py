###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Main kernel source finder - combines parser, searcher, and enrichment.

Supports:
- Dynamic kernel indexing for faster lookups
- Auto-cloning missing repositories
- Python-native search fallback when ripgrep unavailable
- Repository structure auto-discovery
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import KernelKind, KernelSourceInfo
from .parser import KernelNameParser
from .searcher import KernelSourceSearcher
from .repo_config import GITHUB_URL_TEMPLATES, RepoDiscovery
from .indexer import KernelIndex
from .repo_manager import RepoManager

logger = logging.getLogger(__name__)


class KernelSourceFinder:
    """Find kernel source code and tests from profiler kernel names."""
    
    def __init__(
        self,
        repos: List[str] = None,
        auto_clone: bool = True,
        repos_base_dir: str = None,
        use_index: bool = True,
        auto_install_ripgrep: bool = True,
    ):
        """
        Initialize the kernel source finder.
        
        Args:
            repos: List of repository paths to search. If None, repos are 
                   auto-cloned on demand based on detected kernel types.
            auto_clone: If True, automatically clone missing repos when needed.
            repos_base_dir: Base directory for auto-cloned repos.
            use_index: If True, build dynamic kernel index for faster lookups.
            auto_install_ripgrep: If True, try to install ripgrep if missing.
        """
        self.repos = repos or []
        self.auto_clone = auto_clone
        self.use_index = use_index
        
        self.parser = KernelNameParser()
        self.repo_manager = RepoManager(base_dir=repos_base_dir) if auto_clone else None
        self.index = KernelIndex() if use_index else None
        
        # Searcher will be initialized after repos are resolved
        self._searcher: Optional[KernelSourceSearcher] = None
        self._auto_install_ripgrep = auto_install_ripgrep
        self._repos_initialized = False
    
    def _ensure_repos(self, kernel_names: List[str] = None) -> None:
        """Ensure repositories are available, cloning on-demand if needed."""
        if self._repos_initialized:
            return
        
        # If explicit repos provided, use them
        if self.repos:
            self._searcher = KernelSourceSearcher(
                self.repos,
                auto_install_ripgrep=self._auto_install_ripgrep,
            )
            if self.use_index and self.index:
                self.index.build(self.repos)
            self._repos_initialized = True
            return
        
        # Auto-clone mode: determine needed repos from kernel names
        if self.auto_clone and self.repo_manager and kernel_names:
            needed_repos = self.repo_manager.get_repos_for_kernels(kernel_names)
            logger.info(f"Auto-detecting repos for kernels: {needed_repos}")
            
            cloned_paths = []
            for repo_name in needed_repos:
                try:
                    path = self.repo_manager.ensure_repo(repo_name, shallow=True)
                    cloned_paths.append(path)
                except Exception as e:
                    logger.warning(f"Failed to clone {repo_name}: {e}")
            
            self.repos = cloned_paths
        
        # Initialize searcher and index
        self._searcher = KernelSourceSearcher(
            self.repos,
            auto_install_ripgrep=self._auto_install_ripgrep,
        )
        
        if self.use_index and self.index and self.repos:
            self.index.build(self.repos)
        
        self._repos_initialized = True
    
    @property
    def searcher(self) -> KernelSourceSearcher:
        """Get searcher, initializing repos if needed."""
        if self._searcher is None:
            self._ensure_repos()
        return self._searcher
    
    def search(self, kernel_name: str) -> KernelSourceInfo:
        """
        Search for source and test information for a kernel.
        
        Args:
            kernel_name: Kernel name from profiler
            
        Returns:
            KernelSourceInfo with all found information
        """
        # Ensure repos initialized for single kernel search
        self._ensure_repos([kernel_name])
        
        # Parse the kernel name
        parsed = self.parser.parse(kernel_name)
        category = self.parser.classify_category(kernel_name)
        
        # Try index lookup first for fast results
        source_match = None
        if self.use_index and self.index:
            index_result = self.index.lookup(kernel_name)
            if index_result:
                from .models import SourceMatch
                source_match = SourceMatch(
                    file_path=index_result.file_path,
                    symbol=index_result.symbol or index_result.name,
                    repo_var=f"${index_result.repo_name.upper().replace('-', '_')}_DIR",
                    repo_name=index_result.repo_name,
                )
        
        # Fall back to searcher if index miss
        if not source_match:
            source_match = self.searcher.search_source(parsed)
        
        # Search for test
        test_match = self.searcher.search_test(parsed, source_match)
        
        # Build upstream URL
        upstream_url = ""
        if source_match and source_match.repo_name in GITHUB_URL_TEMPLATES:
            upstream_url = GITHUB_URL_TEMPLATES[source_match.repo_name].format(
                path=source_match.file_path
            )
        
        # Build notes with more details
        notes = self._build_notes(parsed)
        
        return KernelSourceInfo(
            kind=parsed.kind.value,
            category=category.value,
            source_repo=source_match.repo_name if source_match else "",
            source_file=source_match.display_path if source_match else "",
            source_symbol=source_match.symbol if source_match else parsed.function_name,
            upstream_url=upstream_url,
            test_file=test_match.display_path if test_match else "",
            test_cmd=test_match.test_cmd if test_match else "",
            notes=notes,
        )
    
    def _build_notes(self, parsed) -> str:
        """Build notes from parsed information."""
        notes = []
        
        extra = parsed.extra or {}
        
        # Add dtype info
        if parsed.dtype:
            notes.append(f"dtype={parsed.dtype}")
        
        # Tensile GEMM specific
        if parsed.kind == KernelKind.TENSILE_GEMM:
            trans_info = []
            if extra.get('trans_a'):
                trans_info.append(f"A={extra['trans_a']}")
            if extra.get('trans_b'):
                trans_info.append(f"B={extra['trans_b']}")
            if trans_info:
                notes.append(f"Transpose: {', '.join(trans_info)}")
            if 'tile_m' in extra:
                notes.append(f"tile={extra['tile_m']}x{extra['tile_n']}x{extra['tile_k']}")
            if 'mfma' in extra:
                notes.append(f"MFMA={extra['mfma']}")
            if 'workgroup' in extra:
                notes.append(f"WG={extra['workgroup']}")
        
        # CK tile specific
        if parsed.kind == KernelKind.CK_TILE:
            if extra.get('fused_add'):
                notes.append("Fused residual add")
            if extra.get('fused_quant'):
                notes.append("Fused quantization")
            if 'block_shape' in extra:
                notes.append(f"block={extra['block_shape']}")
        
        # HIP kernels
        if parsed.kind == KernelKind.HIP_CPP:
            if 'wvSplitK' in parsed.original_name or 'wvSpltK' in parsed.original_name:
                notes.append("WMMA Split-K GEMM kernel")
            if parsed.namespace:
                notes.append(f"namespace={parsed.namespace}")
        
        # Inductor generated
        if parsed.kind == KernelKind.INDUCTOR:
            notes.append("torch.compile generated")
        
        # Triton config
        if parsed.kind == KernelKind.TRITON_JIT and parsed.config:
            # Extract key config details
            config_parts = parsed.config.split('_')
            if len(config_parts) >= 2:
                notes.append(f"config: {parsed.config[:60]}")
        
        return "; ".join(notes)
    
    def get_repo_paths(self) -> Dict[str, str]:
        """Get mapping of repo variable names to actual paths."""
        return self.searcher.get_repo_paths()
    
    def enrich_csv(self, input_path: str, output_path: str = None) -> str:
        """
        Enrich a gap_analysis CSV with kernel source information.
        
        Args:
            input_path: Path to input gap_analysis.csv
            output_path: Path for output (defaults to overwriting input)
            
        Returns:
            Path to the output file
        """
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path
        else:
            output_path = Path(output_path)
        
        # Read existing data
        rows = []
        headers = []
        
        with open(input_path, 'r', newline='') as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                rows.append(row)
        
        # Extract all kernel names for auto-clone detection
        kernel_names = [row[0] for row in rows if row]
        
        # Initialize repos based on all kernel names (triggers on-demand clone)
        self._ensure_repos(kernel_names)
        
        # Get repo paths for metadata
        repo_paths = self.get_repo_paths()
        
        # Add new headers
        new_headers = headers + KernelSourceInfo.csv_headers()
        
        # Enrich each row
        enriched_rows = []
        for row in rows:
            kernel_name = row[0] if row else ""
            source_info = self.search(kernel_name)
            enriched_rows.append(row + source_info.to_list())
        
        # Write output (no metadata rows - clean CSV)
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(new_headers)
            writer.writerows(enriched_rows)
        
        logger.info(f"Enriched CSV written to {output_path}")
        return str(output_path)
    
    def enrich_kernel_stats(self, kernel_stats: List) -> List:
        """
        Enrich a list of KernelStat objects with source information.
        
        Args:
            kernel_stats: List of KernelStat dataclass instances
            
        Returns:
            Same list with source_info populated
        """
        for stat in kernel_stats:
            source_info = self.search(stat.name)
            stat.source_info = source_info
        
        return kernel_stats
