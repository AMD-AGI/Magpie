"""User-facing configuration for generic target selection and trace budgets."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class TargetSpec:
    """A target selected by portable symbol patterns, not image registries."""

    target_id: str
    name_patterns: tuple[str, ...]
    variant_id: str = "baseline"
    package: Optional[str] = None
    source: Optional[Mapping[str, Any]] = None
    source_hashes: Mapping[str, str] = field(default_factory=dict)
    provenance_hashes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.target_id.strip():
            raise ValueError("target_id must be non-empty")
        if not self.name_patterns or any(not pattern for pattern in self.name_patterns):
            raise ValueError(f"target {self.target_id!r} requires name_patterns")

    def matches(self, symbol: str) -> bool:
        """Return whether a profiler/runtime symbol belongs to this target."""

        return any(fnmatch.fnmatchcase(symbol, pattern) for pattern in self.name_patterns)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "name_patterns": list(self.name_patterns),
            "variant_id": self.variant_id,
            "package": self.package,
            "source": dict(self.source) if self.source else None,
            "source_hashes": dict(self.source_hashes),
            "provenance_hashes": dict(self.provenance_hashes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TargetSpec":
        patterns = data.get("name_patterns", data.get("names", []))
        if isinstance(patterns, str):
            patterns = [patterns]
        return cls(
            target_id=str(data.get("target_id", data.get("id", ""))),
            name_patterns=tuple(str(item) for item in patterns),
            variant_id=str(data.get("variant_id", "baseline")),
            package=str(data["package"]) if data.get("package") is not None else None,
            source=dict(data["source"]) if isinstance(data.get("source"), Mapping) else None,
            source_hashes=dict(data.get("source_hashes", {})),
            provenance_hashes=dict(data.get("provenance_hashes", {})),
        )


@dataclass
class TargetedTraceConfig:
    """Diagnostic-only targeted trace settings."""

    enabled: bool = False
    backend: str = "torch_profiler"
    run_seed: str = "magpie-targeted-trace"
    sample_rate: float = 1.0
    max_records_per_shard: int = 100_000
    targets: List[TargetSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.backend = self.backend.lower().strip()
        if self.backend not in {"torch_profiler"}:
            raise ValueError(
                f"unsupported targeted trace backend {self.backend!r}; "
                "use 'torch_profiler'"
            )
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError("targeted_trace.sample_rate must be between 0 and 1")
        if self.max_records_per_shard < 0:
            raise ValueError(
                "targeted_trace.max_records_per_shard must be non-negative"
            )
        converted: List[TargetSpec] = []
        for target in self.targets:
            converted.append(
                TargetSpec.from_dict(target) if isinstance(target, Mapping) else target
            )
        self.targets = converted
        if self.enabled and not self.targets:
            raise ValueError("enabled targeted_trace requires at least one target")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "run_seed": self.run_seed,
            "sample_rate": self.sample_rate,
            "max_records_per_shard": self.max_records_per_shard,
            "targets": [target.to_dict() for target in self.targets],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TargetedTraceConfig":
        return cls(
            enabled=bool(data.get("enabled", False)),
            backend=str(data.get("backend", "torch_profiler")),
            run_seed=str(data.get("run_seed", "magpie-targeted-trace")),
            sample_rate=float(data.get("sample_rate", 1.0)),
            max_records_per_shard=int(data.get("max_records_per_shard", 100_000)),
            targets=[
                TargetSpec.from_dict(item)
                for item in data.get("targets", [])
                if isinstance(item, Mapping)
            ],
        )
