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

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Collection, Dict, List, Optional

from .repo_config import detect_repo_type

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

    CACHE_SCHEMA_VERSION = 2
    
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

            if data.get("schema_version") != self.CACHE_SCHEMA_VERSION:
                logger.info(f"Cache schema is stale for {repo_path}")
                return False
            
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
                "schema_version": self.CACHE_SCHEMA_VERSION,
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
        # Repository checkouts are frequently materialized under versioned or
        # evidence-specific directory names (for example
        # ``aiter-v0.1.10.post2``). Use the shared known-repository structural
        # detector so emitted repo variables stay canonical (``$AITER_DIR``),
        # independent of the checkout directory name.
        detected = detect_repo_type(str(repo_path))
        if detected:
            return detected

        # Preserve the looser legacy PyTorch fixture/layout detection. All
        # other known repositories are identified through repo_config.
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
    
    def lookup(
        self,
        kernel_name: str,
        *,
        expected_repo_names: Optional[Collection[str]] = None,
        expected_kinds: Optional[Collection[str]] = None,
    ) -> Optional[KernelDefinition]:
        """Look up a kernel without guessing across provenance boundaries.

        ``expected_repo_names`` and ``expected_kinds`` are supplied by the
        parser/finder boundary.  They keep an identical symbol in an unrelated
        repository or language from being accepted as source evidence.  Any
        empty, mangled, ambiguous, or otherwise unparseable name fails closed.
        """
        function_name = self._extract_function_name(kernel_name)
        if not function_name:
            logger.debug("Kernel index rejected unparseable name: %r", kernel_name)
            return None

        repo_names = self._normalize_filter(expected_repo_names)
        kinds = self._normalize_filter(expected_kinds)

        exact = self._compatible_definitions(
            self.name_to_keys.get(function_name, []),
            repo_names=repo_names,
            kinds=kinds,
        )
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            logger.warning(
                "Kernel index rejected ambiguous exact match for %r (%d candidates)",
                kernel_name,
                len(exact),
            )
            return None

        prefix_keys: List[str] = []
        for name, keys in self.name_to_keys.items():
            if self._is_token_boundary_prefix(function_name, name):
                prefix_keys.extend(keys)
        prefix_matches = self._compatible_definitions(
            prefix_keys,
            repo_names=repo_names,
            kinds=kinds,
        )
        if len(prefix_matches) == 1:
            return prefix_matches[0]
        if prefix_matches:
            logger.warning(
                "Kernel index rejected ambiguous prefix match for %r (%d candidates)",
                kernel_name,
                len(prefix_matches),
            )
        return None

    @staticmethod
    def _normalize_filter(
        values: Optional[Collection[str]],
    ) -> Optional[frozenset[str]]:
        if values is None:
            return None
        normalized = frozenset(
            str(value).strip() for value in values if str(value).strip()
        )
        return normalized

    def _compatible_definitions(
        self,
        keys: Collection[str],
        *,
        repo_names: Optional[frozenset[str]],
        kinds: Optional[frozenset[str]],
    ) -> List[KernelDefinition]:
        matches: List[KernelDefinition] = []
        seen = set()
        for key in keys:
            definition = self.index.get(key)
            if definition is None:
                continue
            if repo_names is not None and definition.repo_name not in repo_names:
                continue
            if kinds is not None and definition.kind not in kinds:
                continue
            identity = (
                definition.repo_path,
                definition.file_path,
                definition.name,
                definition.kind,
            )
            if identity in seen:
                continue
            seen.add(identity)
            matches.append(definition)
        return matches

    @staticmethod
    def _is_token_boundary_prefix(query: str, indexed_name: str) -> bool:
        """Return true only for a unique underscore-delimited extension.

        The previous symmetric ``startswith`` accepted an empty query and
        returned whichever definition happened to be inserted first.  It also
        let short incidental prefixes win.  Configured Triton symbols use
        underscore-delimited suffixes, so that is the only inexact relation we
        retain.
        """

        if not query or not indexed_name or query == indexed_name:
            return False
        return query.startswith(f"{indexed_name}_")

    def _extract_function_name(self, kernel_name: str) -> Optional[str]:
        if not isinstance(kernel_name, str):
            return None
        name = kernel_name.strip()
        if not name:
            return None
        if name.endswith(".k.d"):
            name = name[:-4]
        elif name.endswith(".kd"):
            name = name[:-3]
        name = re.sub(r"\s*\[clone \.kd\]\s*$", "", name).strip()
        # The index contains source-level identifiers, not a demangler.  An
        # Itanium symbol must be resolved by the kind-aware searcher instead
        # of being truncated to an empty string and prefix-matched.
        if name.startswith("_Z"):
            return None
        if not re.fullmatch(r"[A-Za-z_]\w*", name):
            return None
        parts = name.split("_")

        for i, part in enumerate(parts):
            if part and (part[0].isupper() or part.isdigit() or
                        part in ("bf16", "fp16", "fp32", "int8")):
                candidate = "_".join(parts[:i])
                return candidate or None

        return name
    
    def get_all_definitions(self, kind: str = None) -> List[KernelDefinition]:
        if kind:
            return [d for d in self.index.values() if d.kind == kind]
        return list(self.index.values())
    
    def clear(self) -> None:
        self.index.clear()
        self.name_to_keys.clear()
