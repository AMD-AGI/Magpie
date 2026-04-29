###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Dynamic kernel index for fast source lookups.

Scans repositories for kernel definitions and builds a searchable index,
replacing hardcoded mappings with dynamically discovered kernel locations.
"""

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class KernelDefinition:
    """A kernel definition found in source code."""
    
    name: str
    file_path: str
    repo_name: str
    repo_path: str
    kind: str
    line_number: int = 0
    symbol: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "KernelDefinition":
        return cls(**d)


class KernelIndex:
    """
    Dynamic index of kernel definitions.
    
    Scans repositories for kernel definitions and builds a searchable index.
    Supports caching to avoid rescanning.
    """
    
    KERNEL_PATTERNS = {
        "triton_jit": [
            (r"@triton\.jit[^\n]*\s*\ndef\s+(\w+)", "py"),
            (r"@triton\.autotune[^\n]*\s*@triton\.jit[^\n]*\s*\ndef\s+(\w+)", "py"),
        ],
        "hip_cpp": [
            (r"__global__\s+void\s+(\w+)\s*[<(]", "cpp,cu,hip"),
        ],
        "ck_tile": [
            (r"template\s*<[^>]*>\s*__global__\s+void\s+kentry", "hpp,cpp"),
        ],
    }
    
    FILE_EXTENSIONS = {
        "triton_jit": [".py"],
        "hip_cpp": [".cpp", ".cu", ".hip", ".hpp"],
        "ck_tile": [".hpp", ".cpp"],
        "aten_native": [".cu", ".cpp"],
    }
    
    SKIP_DIRS = {
        ".git", "__pycache__", "node_modules", "build", "dist",
        ".tox", ".eggs", "venv", ".venv",
    }
    
    def __init__(self, cache_dir: str = None):
        self.index: Dict[str, KernelDefinition] = {}
        self.name_to_keys: Dict[str, List[str]] = {}
        
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "magpie" / "kernel_index"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def build(self, repos: List[str], force_rebuild: bool = False) -> None:
        """Build index from repositories."""
        for repo_path in repos:
            repo_path = str(repo_path)
            if not Path(repo_path).exists():
                logger.warning(f"Repo path does not exist: {repo_path}")
                continue
            
            cache_file = self._get_cache_file(repo_path)
            if not force_rebuild and cache_file.exists():
                if self._load_cache(cache_file, repo_path):
                    logger.info(f"Loaded index from cache for {repo_path}")
                    continue
            
            logger.info(f"Scanning repository: {repo_path}")
            self._scan_repo(repo_path)
            self._save_cache(cache_file, repo_path)
        
        self._build_name_index()
        logger.info(f"Index built with {len(self.index)} kernel definitions")
    
    def _get_cache_file(self, repo_path: str) -> Path:
        path_hash = hashlib.md5(repo_path.encode()).hexdigest()[:12]
        repo_name = Path(repo_path).name
        return self.cache_dir / f"{repo_name}_{path_hash}.json"
    
    def _load_cache(self, cache_file: Path, repo_path: str) -> bool:
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            cached_mtime = data.get("mtime", 0)
            current_mtime = Path(repo_path).stat().st_mtime
            
            if abs(current_mtime - cached_mtime) > 86400:
                logger.info(f"Cache expired for {repo_path}")
                return False
            
            for key, def_dict in data.get("definitions", {}).items():
                self.index[key] = KernelDefinition.from_dict(def_dict)
            
            return True
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False
    
    def _save_cache(self, cache_file: Path, repo_path: str) -> None:
        try:
            repo_defs = {
                k: v.to_dict() for k, v in self.index.items()
                if v.repo_path == repo_path
            }
            
            data = {
                "mtime": Path(repo_path).stat().st_mtime,
                "definitions": repo_defs,
            }
            
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(repo_defs)} definitions to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _scan_repo(self, repo_path: str) -> None:
        repo_path = Path(repo_path)
        repo_name = self._detect_repo_name(repo_path)
        
        files_to_scan: Dict[str, List[Path]] = {}
        
        for kind, extensions in self.FILE_EXTENSIONS.items():
            files_to_scan[kind] = []
            for ext in extensions:
                for file_path in repo_path.rglob(f"*{ext}"):
                    if any(skip in file_path.parts for skip in self.SKIP_DIRS):
                        continue
                    files_to_scan[kind].append(file_path)
        
        for kind, patterns in self.KERNEL_PATTERNS.items():
            for pattern, file_types in patterns:
                allowed_exts = [f".{ft}" for ft in file_types.split(",")]
                
                for file_path in files_to_scan.get(kind, []):
                    if file_path.suffix not in allowed_exts:
                        continue
                    
                    self._scan_file(file_path, pattern, kind, repo_name, str(repo_path))
    
    def _scan_file(self, file_path: Path, pattern: str, kind: str,
                   repo_name: str, repo_path: str) -> None:
        try:
            content = file_path.read_text(errors='ignore')
            
            for match in re.finditer(pattern, content, re.MULTILINE):
                name = match.group(1) if match.groups() else None
                if not name:
                    continue
                
                line_num = content[:match.start()].count('\n') + 1
                rel_path = str(file_path.relative_to(repo_path))
                key = f"{repo_name}:{rel_path}:{name}"
                
                self.index[key] = KernelDefinition(
                    name=name,
                    file_path=rel_path,
                    repo_name=repo_name,
                    repo_path=repo_path,
                    kind=kind,
                    line_number=line_num,
                    symbol=match.group(0)[:200],
                )
        except Exception as e:
            logger.debug(f"Error scanning {file_path}: {e}")
    
    def _detect_repo_name(self, repo_path: Path) -> str:
        if (repo_path / "projects" / "composablekernel").exists():
            return "rocm-libraries"
        if (repo_path / "python" / "triton").exists():
            return "triton"
        if (repo_path / "vllm").exists() and (repo_path / "csrc").exists():
            return "vllm"
        if (repo_path / "aten").exists():
            return "pytorch"
        return repo_path.name
    
    def _build_name_index(self) -> None:
        self.name_to_keys.clear()
        for key, defn in self.index.items():
            name = defn.name
            if name not in self.name_to_keys:
                self.name_to_keys[name] = []
            self.name_to_keys[name].append(key)
    
    def lookup(self, kernel_name: str) -> Optional[KernelDefinition]:
        """Look up a kernel by name."""
        function_name = self._extract_function_name(kernel_name)
        
        if function_name in self.name_to_keys:
            keys = self.name_to_keys[function_name]
            if keys:
                return self.index[keys[0]]
        
        for name, keys in self.name_to_keys.items():
            if function_name.startswith(name) or name.startswith(function_name):
                return self.index[keys[0]]
        
        return None
    
    def _extract_function_name(self, kernel_name: str) -> str:
        name = kernel_name.replace(".kd", "")
        parts = name.split("_")
        
        for i, part in enumerate(parts):
            if part and (part[0].isupper() or part.isdigit() or
                        part in ("bf16", "fp16", "fp32", "int8")):
                return "_".join(parts[:i])
        
        return name
    
    def get_all_definitions(self, kind: str = None) -> List[KernelDefinition]:
        if kind:
            return [d for d in self.index.values() if d.kind == kind]
        return list(self.index.values())
    
    def clear(self) -> None:
        self.index.clear()
        self.name_to_keys.clear()
