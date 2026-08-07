"""Versioned data contract for targeted kernel trace artifacts.

The schema intentionally describes evidence, not a particular framework patching
mechanism.  Runtime probes, Torch profiler adapters, and future TraceLens adapters
all emit the same records and shard receipts.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_NAME = "magpie.targeted-kernel-trace"
SCHEMA_VERSION = "1.0.0"
ZERO_CHECKSUM = "0" * 64
ENVELOPE_TYPES = frozenset({"header", "event", "end"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class TraceValidationError(ValueError):
    """Raised when a trace artifact violates the semantic contract."""


def utc_now() -> str:
    """Return an RFC3339 timestamp in UTC."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    """Serialize *value* deterministically for hashing and golden fixtures."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    """Return the SHA-256 digest of canonical JSON."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TraceValidationError(f"{field_name} must be a non-empty string")


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise TraceValidationError(
            f"{field_name} must be a lowercase SHA-256 hex string"
        )


def _json_copy(value: Any) -> Any:
    """Validate JSON compatibility and return a detached canonical copy."""

    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise TraceValidationError(f"value is not canonical JSON: {exc}") from exc


@dataclass(frozen=True)
class SourceEvidence:
    """Python-visible launch or wrapper source evidence."""

    path: str
    line: Optional[int] = None
    function: Optional[str] = None
    sha256: Optional[str] = None

    def __post_init__(self) -> None:
        _require_nonempty(self.path, "source.path")
        if self.line is not None and self.line <= 0:
            raise TraceValidationError("source.line must be positive")
        if self.sha256 is not None:
            _require_sha256(self.sha256, "source.sha256")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceEvidence":
        return cls(
            path=str(data.get("path", "")),
            line=int(data["line"]) if data.get("line") is not None else None,
            function=(
                str(data["function"]) if data.get("function") is not None else None
            ),
            sha256=str(data["sha256"]) if data.get("sha256") is not None else None,
        )


@dataclass(frozen=True)
class TensorEvidence:
    """Host-visible tensor metadata; tensor contents and raw pointers are excluded."""

    name: str
    shape: Tuple[Any, ...]
    dtype: str
    stride: Optional[Tuple[Any, ...]] = None
    device: Optional[str] = None
    layout: Optional[str] = None
    requires_grad: Optional[bool] = None

    def __post_init__(self) -> None:
        _require_nonempty(self.name, "tensor.name")
        _require_nonempty(self.dtype, "tensor.dtype")
        _json_copy(list(self.shape))
        if self.stride is not None:
            _json_copy(list(self.stride))

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["shape"] = list(self.shape)
        if self.stride is not None:
            data["stride"] = list(self.stride)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TensorEvidence":
        stride = data.get("stride")
        return cls(
            name=str(data.get("name", "")),
            shape=tuple(data.get("shape", [])),
            dtype=str(data.get("dtype", "unknown")),
            stride=tuple(stride) if isinstance(stride, (list, tuple)) else None,
            device=str(data["device"]) if data.get("device") is not None else None,
            layout=str(data["layout"]) if data.get("layout") is not None else None,
            requires_grad=(
                bool(data["requires_grad"])
                if data.get("requires_grad") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class TraceIdentity:
    """Stable identity and provenance for one target observation."""

    run_id: str
    target_id: str
    variant_id: str = "baseline"
    package: Optional[str] = None
    image: Optional[str] = None
    source_hashes: Mapping[str, str] = field(default_factory=dict)
    provenance_hashes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty(self.run_id, "identity.run_id")
        _require_nonempty(self.target_id, "identity.target_id")
        _require_nonempty(self.variant_id, "identity.variant_id")
        _json_copy(dict(self.source_hashes))
        _json_copy(dict(self.provenance_hashes))

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["source_hashes"] = dict(self.source_hashes)
        data["provenance_hashes"] = dict(self.provenance_hashes)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TraceIdentity":
        return cls(
            run_id=str(data.get("run_id", "")),
            target_id=str(data.get("target_id", "")),
            variant_id=str(data.get("variant_id", "baseline")),
            package=str(data["package"]) if data.get("package") is not None else None,
            image=str(data["image"]) if data.get("image") is not None else None,
            source_hashes=dict(data.get("source_hashes", {})),
            provenance_hashes=dict(data.get("provenance_hashes", {})),
        )


@dataclass(frozen=True)
class TraceContext:
    """Execution context used for rank/stage/graph attribution."""

    framework: str
    rank: int
    pid: int
    framework_version: Optional[str] = None
    stage: str = "unknown"
    execution_mode: str = "unknown"
    graph_id: Optional[str] = None
    world_size: Optional[int] = None

    def __post_init__(self) -> None:
        _require_nonempty(self.framework, "context.framework")
        if self.rank < 0:
            raise TraceValidationError("context.rank must be non-negative")
        if self.pid < 0:
            raise TraceValidationError("context.pid must be non-negative")
        if self.world_size is not None and self.world_size <= 0:
            raise TraceValidationError("context.world_size must be positive")
        if self.world_size is not None and self.rank >= self.world_size:
            raise TraceValidationError("context.rank must be less than world_size")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TraceContext":
        return cls(
            framework=str(data.get("framework", "unknown")),
            rank=int(data.get("rank", 0)),
            pid=int(data.get("pid", 0)),
            framework_version=(
                str(data["framework_version"])
                if data.get("framework_version") is not None
                else None
            ),
            stage=str(data.get("stage", "unknown")),
            execution_mode=str(data.get("execution_mode", "unknown")),
            graph_id=(
                str(data["graph_id"]) if data.get("graph_id") is not None else None
            ),
            world_size=(
                int(data["world_size"])
                if data.get("world_size") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class LaunchSemantics:
    """Python-visible invocation semantics needed to reproduce a launch."""

    source: Optional[SourceEvidence] = None
    tensors: Tuple[TensorEvidence, ...] = ()
    named_scalars: Mapping[str, Any] = field(default_factory=dict)
    constexpr: Mapping[str, Any] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)
    python_grid: Any = None

    def __post_init__(self) -> None:
        _json_copy(dict(self.named_scalars))
        _json_copy(dict(self.constexpr))
        _json_copy(dict(self.meta))
        _json_copy(self.python_grid)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.to_dict() if self.source else None,
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "named_scalars": _json_copy(dict(self.named_scalars)),
            "constexpr": _json_copy(dict(self.constexpr)),
            "meta": _json_copy(dict(self.meta)),
            "python_grid": _json_copy(self.python_grid),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LaunchSemantics":
        source = data.get("source")
        return cls(
            source=(
                SourceEvidence.from_dict(source)
                if isinstance(source, Mapping)
                else None
            ),
            tensors=tuple(
                TensorEvidence.from_dict(item)
                for item in data.get("tensors", [])
                if isinstance(item, Mapping)
            ),
            named_scalars=dict(data.get("named_scalars", {})),
            constexpr=dict(data.get("constexpr", {})),
            meta=dict(data.get("meta", {})),
            python_grid=data.get("python_grid"),
        )


@dataclass(frozen=True)
class RuntimeEvidence:
    """Profiler/runtime evidence.  Unknown fields remain ``None``, never zero."""

    cpu_uid: Optional[str] = None
    correlation_id: Optional[str] = None
    gpu_uid: Optional[str] = None
    gpu_symbol: Optional[str] = None
    grid: Optional[Tuple[int, ...]] = None
    block: Optional[Tuple[int, ...]] = None
    stream: Optional[str] = None
    duration_us: Optional[float] = None
    timestamp_us: Optional[float] = None

    def __post_init__(self) -> None:
        if self.duration_us is not None and self.duration_us < 0:
            raise TraceValidationError("runtime.duration_us must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.grid is not None:
            data["grid"] = list(self.grid)
        if self.block is not None:
            data["block"] = list(self.block)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeEvidence":
        grid = data.get("grid")
        block = data.get("block")
        return cls(
            cpu_uid=str(data["cpu_uid"]) if data.get("cpu_uid") is not None else None,
            correlation_id=(
                str(data["correlation_id"])
                if data.get("correlation_id") is not None
                else None
            ),
            gpu_uid=str(data["gpu_uid"]) if data.get("gpu_uid") is not None else None,
            gpu_symbol=(
                str(data["gpu_symbol"])
                if data.get("gpu_symbol") is not None
                else None
            ),
            grid=(
                tuple(int(value) for value in grid)
                if isinstance(grid, Sequence)
                and not isinstance(grid, (str, bytes, bytearray))
                else None
            ),
            block=(
                tuple(int(value) for value in block)
                if isinstance(block, Sequence)
                and not isinstance(block, (str, bytes, bytearray))
                else None
            ),
            stream=str(data["stream"]) if data.get("stream") is not None else None,
            duration_us=(
                float(data["duration_us"])
                if data.get("duration_us") is not None
                else None
            ),
            timestamp_us=(
                float(data["timestamp_us"])
                if data.get("timestamp_us") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class TargetedTraceRecord:
    """One semantic and/or runtime observation for a selected target."""

    kind: str
    stable_event_key: str
    identity: TraceIdentity
    context: TraceContext
    semantics: LaunchSemantics = field(default_factory=LaunchSemantics)
    runtime: RuntimeEvidence = field(default_factory=RuntimeEvidence)
    timestamp_ns: Optional[int] = None
    warnings: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty(self.kind, "record.kind")
        _require_nonempty(self.stable_event_key, "record.stable_event_key")
        if self.timestamp_ns is not None and self.timestamp_ns < 0:
            raise TraceValidationError("record.timestamp_ns must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "stable_event_key": self.stable_event_key,
            "identity": self.identity.to_dict(),
            "context": self.context.to_dict(),
            "semantics": self.semantics.to_dict(),
            "runtime": self.runtime.to_dict(),
            "timestamp_ns": self.timestamp_ns,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TargetedTraceRecord":
        try:
            identity = data["identity"]
            context = data["context"]
        except KeyError as exc:
            raise TraceValidationError(f"record missing {exc.args[0]}") from exc
        if not isinstance(identity, Mapping) or not isinstance(context, Mapping):
            raise TraceValidationError("record identity/context must be objects")
        semantics = data.get("semantics", {})
        runtime = data.get("runtime", {})
        return cls(
            kind=str(data.get("kind", "")),
            stable_event_key=str(data.get("stable_event_key", "")),
            identity=TraceIdentity.from_dict(identity),
            context=TraceContext.from_dict(context),
            semantics=LaunchSemantics.from_dict(
                semantics if isinstance(semantics, Mapping) else {}
            ),
            runtime=RuntimeEvidence.from_dict(
                runtime if isinstance(runtime, Mapping) else {}
            ),
            timestamp_ns=(
                int(data["timestamp_ns"])
                if data.get("timestamp_ns") is not None
                else None
            ),
            warnings=tuple(str(item) for item in data.get("warnings", [])),
        )


@dataclass
class ShardCounters:
    """Loss-accounting counters for one PID/rank shard."""

    seen: int = 0
    sampled: int = 0
    written: int = 0
    dropped: int = 0
    dropped_by_reason: Dict[str, int] = field(default_factory=dict)

    def note_drop(self, reason: str) -> None:
        _require_nonempty(reason, "drop reason")
        self.dropped += 1
        self.dropped_by_reason[reason] = self.dropped_by_reason.get(reason, 0) + 1

    def validate(self) -> None:
        values = (self.seen, self.sampled, self.written, self.dropped)
        if any(not isinstance(value, int) or value < 0 for value in values):
            raise TraceValidationError("coverage counters must be non-negative integers")
        if sum(self.dropped_by_reason.values()) != self.dropped:
            raise TraceValidationError("dropped_by_reason does not sum to dropped")
        if self.seen != self.written + self.dropped:
            raise TraceValidationError("seen must equal written + dropped")
        unsampled = self.dropped_by_reason.get("sampling", 0)
        if self.sampled != self.written + self.dropped - unsampled:
            raise TraceValidationError(
                "sampled must equal written + non-sampling drops"
            )

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "seen": self.seen,
            "sampled": self.sampled,
            "written": self.written,
            "dropped": self.dropped,
            "dropped_by_reason": dict(sorted(self.dropped_by_reason.items())),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ShardCounters":
        counters = cls(
            seen=int(data.get("seen", 0)),
            sampled=int(data.get("sampled", 0)),
            written=int(data.get("written", 0)),
            dropped=int(data.get("dropped", 0)),
            dropped_by_reason={
                str(key): int(value)
                for key, value in dict(data.get("dropped_by_reason", {})).items()
            },
        )
        counters.validate()
        return counters

    @classmethod
    def aggregate(cls, counters: Iterable["ShardCounters"]) -> "ShardCounters":
        result = cls()
        for item in counters:
            result.seen += item.seen
            result.sampled += item.sampled
            result.written += item.written
            result.dropped += item.dropped
            for reason, count in item.dropped_by_reason.items():
                result.dropped_by_reason[reason] = (
                    result.dropped_by_reason.get(reason, 0) + count
                )
        result.validate()
        return result


@dataclass(frozen=True)
class ShardReceipt:
    """Integrity and coverage receipt for a completed shard."""

    path: str
    rank: int
    pid: int
    sequence_end: int
    chain_checksum: str
    file_sha256: str
    byte_count: int
    counters: ShardCounters
    complete: bool = True

    def __post_init__(self) -> None:
        _require_nonempty(self.path, "receipt.path")
        if self.rank < 0 or self.pid < 0 or self.sequence_end < 0:
            raise TraceValidationError("receipt rank/pid/sequence are invalid")
        _require_sha256(self.chain_checksum, "receipt.chain_checksum")
        _require_sha256(self.file_sha256, "receipt.file_sha256")
        if self.byte_count < 0:
            raise TraceValidationError("receipt.byte_count must be non-negative")
        self.counters.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "rank": self.rank,
            "pid": self.pid,
            "sequence_end": self.sequence_end,
            "chain_checksum": self.chain_checksum,
            "file_sha256": self.file_sha256,
            "byte_count": self.byte_count,
            "counters": self.counters.to_dict(),
            "complete": self.complete,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ShardReceipt":
        counters = data.get("counters", {})
        return cls(
            path=str(data.get("path", "")),
            rank=int(data.get("rank", 0)),
            pid=int(data.get("pid", 0)),
            sequence_end=int(data.get("sequence_end", 0)),
            chain_checksum=str(data.get("chain_checksum", "")),
            file_sha256=str(data.get("file_sha256", "")),
            byte_count=int(data.get("byte_count", 0)),
            counters=ShardCounters.from_dict(
                counters if isinstance(counters, Mapping) else {}
            ),
            complete=bool(data.get("complete", False)),
        )


@dataclass(frozen=True)
class TargetedTraceManifest:
    """Run-level manifest consumed by Apex and other evidence clients."""

    run_id: str
    acquisition_backend: str
    targets: Tuple[Mapping[str, Any], ...]
    shards: Tuple[ShardReceipt, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    pass_kind: str = "diagnostic"
    reward_eligible: bool = False
    schema_name: str = SCHEMA_NAME
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_nonempty(self.run_id, "manifest.run_id")
        _require_nonempty(self.acquisition_backend, "manifest.acquisition_backend")
        if self.schema_name != SCHEMA_NAME or self.schema_version != SCHEMA_VERSION:
            raise TraceValidationError(
                f"unsupported schema {self.schema_name}@{self.schema_version}"
            )
        if self.pass_kind != "diagnostic":
            raise TraceValidationError("targeted trace artifacts must be diagnostic")
        if self.reward_eligible:
            raise TraceValidationError("targeted trace artifacts cannot be reward eligible")
        _json_copy([dict(item) for item in self.targets])
        _json_copy(dict(self.provenance))
        receipt_keys = [
            (receipt.rank, receipt.pid, receipt.path) for receipt in self.shards
        ]
        if len(receipt_keys) != len(set(receipt_keys)):
            raise TraceValidationError("manifest contains duplicate shard receipts")

    @property
    def coverage(self) -> ShardCounters:
        return ShardCounters.aggregate(receipt.counters for receipt in self.shards)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "pass_kind": self.pass_kind,
            "reward_eligible": self.reward_eligible,
            "acquisition_backend": self.acquisition_backend,
            "targets": [_json_copy(dict(item)) for item in self.targets],
            "provenance": _json_copy(dict(self.provenance)),
            "coverage": self.coverage.to_dict(),
            "shards": [receipt.to_dict() for receipt in self.shards],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TargetedTraceManifest":
        schema_name = str(data.get("schema_name", ""))
        schema_version = str(data.get("schema_version", ""))
        if schema_name != SCHEMA_NAME or schema_version != SCHEMA_VERSION:
            raise TraceValidationError(
                f"unsupported schema {schema_name or '<missing>'}@"
                f"{schema_version or '<missing>'}; expected "
                f"{SCHEMA_NAME}@{SCHEMA_VERSION}"
            )
        raw_targets = data.get("targets")
        raw_shards = data.get("shards")
        if not isinstance(raw_targets, list) or any(
            not isinstance(item, Mapping) for item in raw_targets
        ):
            raise TraceValidationError("manifest targets must be a list of objects")
        if not isinstance(raw_shards, list) or any(
            not isinstance(item, Mapping) for item in raw_shards
        ):
            raise TraceValidationError("manifest shards must be a list of objects")
        manifest = cls(
            run_id=str(data.get("run_id", "")),
            acquisition_backend=str(data.get("acquisition_backend", "")),
            targets=tuple(dict(item) for item in raw_targets),
            shards=tuple(ShardReceipt.from_dict(item) for item in raw_shards),
            provenance=dict(data.get("provenance", {})),
            created_at=str(data.get("created_at", utc_now())),
            pass_kind=str(data.get("pass_kind", "")),
            reward_eligible=bool(data.get("reward_eligible", True)),
            schema_name=schema_name,
            schema_version=schema_version,
        )
        expected_coverage = data.get("coverage")
        if not isinstance(expected_coverage, Mapping):
            raise TraceValidationError("manifest coverage must be an object")
        if ShardCounters.from_dict(expected_coverage).to_dict() != manifest.coverage.to_dict():
            raise TraceValidationError("manifest coverage does not match shard receipts")
        return manifest


def build_envelope(
    *, record_type: str, sequence: int, previous_checksum: str, payload: Mapping[str, Any]
) -> Dict[str, Any]:
    """Build one checksummed JSONL envelope."""

    if record_type not in ENVELOPE_TYPES:
        raise TraceValidationError(f"unknown envelope record_type: {record_type}")
    if sequence < 0:
        raise TraceValidationError("envelope.sequence must be non-negative")
    _require_sha256(previous_checksum, "previous_checksum")
    body = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "record_type": record_type,
        "sequence": sequence,
        "previous_checksum": previous_checksum,
        "payload": _json_copy(dict(payload)),
    }
    body["checksum"] = sha256_json(body)
    return body


def validate_envelope(
    data: Mapping[str, Any], *, expected_sequence: int, previous_checksum: str
) -> Dict[str, Any]:
    """Validate one shard envelope and return a detached dictionary."""

    schema_name = data.get("schema_name")
    schema_version = data.get("schema_version")
    if schema_name != SCHEMA_NAME or schema_version != SCHEMA_VERSION:
        raise TraceValidationError(
            f"unsupported envelope schema {schema_name}@{schema_version}"
        )
    record_type = data.get("record_type")
    if record_type not in ENVELOPE_TYPES:
        raise TraceValidationError(f"invalid record_type {record_type!r}")
    if data.get("sequence") != expected_sequence:
        raise TraceValidationError(
            f"sequence mismatch: expected {expected_sequence}, got {data.get('sequence')}"
        )
    if data.get("previous_checksum") != previous_checksum:
        raise TraceValidationError("previous checksum mismatch")
    checksum = data.get("checksum")
    if not isinstance(checksum, str):
        raise TraceValidationError("missing envelope checksum")
    body = dict(data)
    body.pop("checksum", None)
    try:
        observed_checksum = sha256_json(body)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TraceValidationError("envelope is not canonical JSON") from exc
    if observed_checksum != checksum:
        raise TraceValidationError("envelope checksum mismatch")
    if not isinstance(data.get("payload"), Mapping):
        raise TraceValidationError("envelope payload must be an object")
    return _json_copy(dict(data))
