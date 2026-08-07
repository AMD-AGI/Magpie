"""Loss-accounted per-rank shard writer for targeted trace records."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Mapping, Optional

from .sampling import should_sample
from .schema import (
    ZERO_CHECKSUM,
    ShardCounters,
    ShardReceipt,
    TargetedTraceManifest,
    TargetedTraceRecord,
    TraceValidationError,
    build_envelope,
    canonical_json,
    utc_now,
)


class TraceShardWriter:
    """Write one checksummed JSONL shard.

    Header and end-sentinel envelopes are outside the event budget.  Every
    observed target event becomes either one written event or one named drop,
    making sampling/cap/serialization loss visible to downstream consumers.
    """

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        rank: int,
        pid: int,
        run_seed: str,
        sample_rate: float = 1.0,
        max_records: int = 100_000,
        header_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if rank < 0 or pid < 0:
            raise ValueError("rank and pid must be non-negative")
        if max_records < 0:
            raise ValueError("max_records must be non-negative")
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("sample_rate must be between 0 and 1")
        self.path = Path(path)
        self.run_id = run_id
        self.rank = rank
        self.pid = pid
        self.run_seed = str(run_seed)
        self.sample_rate = float(sample_rate)
        self.max_records = int(max_records)
        self.counters = ShardCounters()
        self._sequence = 0
        self._previous_checksum = ZERO_CHECKSUM
        self._file_hash = hashlib.sha256()
        self._byte_count = 0
        self._closed = False
        self._io_failed = False
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A shard is an append-once artifact. Reusing a PID/rank path would
        # silently mix or erase evidence, so require a fresh output directory.
        self._file = self.path.open("xb")
        try:
            self._write_envelope(
                "header",
                {
                    "run_id": run_id,
                    "rank": rank,
                    "pid": pid,
                    "run_seed": self.run_seed,
                    "sample_rate": self.sample_rate,
                    "max_records": self.max_records,
                    "metadata": dict(header_metadata or {}),
                },
            )
        except Exception:
            self._file.close()
            raise

    def _write_envelope(self, record_type: str, payload: Mapping[str, Any]) -> None:
        envelope = build_envelope(
            record_type=record_type,
            sequence=self._sequence,
            previous_checksum=self._previous_checksum,
            payload=payload,
        )
        raw = (canonical_json(envelope) + "\n").encode("utf-8")
        written = self._file.write(raw)
        if written is None:
            written = 0
        self._file_hash.update(raw[:written])
        self._byte_count += written
        if written != len(raw):
            raise OSError(f"short trace shard write: {written}/{len(raw)} bytes")
        self._previous_checksum = str(envelope["checksum"])
        self._sequence += 1

    def submit(self, record: TargetedTraceRecord) -> bool:
        """Observe and conditionally write *record*; return whether it was written."""

        with self._lock:
            if self._closed:
                raise RuntimeError("cannot submit to a closed trace shard")
            self.counters.seen += 1

            if not should_sample(
                self.run_seed, record.stable_event_key, self.sample_rate
            ):
                self.counters.note_drop("sampling")
                return False

            self.counters.sampled += 1
            if self._io_failed:
                self.counters.note_drop("io_error")
                return False
            if self.counters.written >= self.max_records:
                self.counters.note_drop("cap")
                return False
            if record.identity.run_id != self.run_id:
                self.counters.note_drop("invalid_record")
                return False
            if record.context.rank != self.rank or record.context.pid != self.pid:
                self.counters.note_drop("invalid_record")
                return False

            try:
                payload = record.to_dict()
            except Exception:
                self.counters.note_drop("serialization_error")
                return False

            try:
                self._write_envelope("event", payload)
            except (OSError, ValueError, TypeError):
                self._io_failed = True
                self.counters.note_drop("io_error")
                return False
            self.counters.written += 1
            return True

    def note_failed_observation(self, reason: str, *, sampled: bool = True) -> None:
        """Account for an event that failed before a typed record could be built."""

        with self._lock:
            if self._closed:
                raise RuntimeError("cannot update a closed trace shard")
            self.counters.seen += 1
            if sampled:
                self.counters.sampled += 1
                self.counters.note_drop(reason)
            else:
                self.counters.note_drop("sampling")

    def close(self) -> ShardReceipt:
        """Write the end sentinel, fsync, and return an integrity receipt."""

        with self._lock:
            if self._closed:
                if not hasattr(self, "_receipt"):
                    raise RuntimeError("closed trace shard has no receipt")
                return self._receipt

            self.counters.validate()
            complete = not self._io_failed
            if complete:
                try:
                    self._write_envelope(
                        "end",
                        {
                            "run_id": self.run_id,
                            "rank": self.rank,
                            "pid": self.pid,
                            "counters": self.counters.to_dict(),
                            "end_reason": "complete",
                        },
                    )
                    self._file.flush()
                    os.fsync(self._file.fileno())
                except OSError:
                    complete = False
                    self._io_failed = True
            self._file.close()
            self._closed = True
            self._receipt = ShardReceipt(
                path=str(self.path),
                rank=self.rank,
                pid=self.pid,
                sequence_end=max(0, self._sequence - 1),
                chain_checksum=self._previous_checksum,
                file_sha256=self._file_hash.hexdigest(),
                byte_count=self._byte_count,
                counters=self.counters,
                complete=complete,
            )
            return self._receipt

    def __enter__(self) -> "TraceShardWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def write_manifest(path: Path, manifest: TargetedTraceManifest) -> None:
    """Atomically write a run manifest."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(
        canonical_json(manifest.to_dict()) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def merge_runtime_manifest(
    path: Path,
    *,
    run_id: str,
    receipt: ShardReceipt,
    targets: list[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> TargetedTraceManifest:
    """Lock/merge one runtime shard into a multi-process run manifest."""

    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        existing: Optional[TargetedTraceManifest] = None
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping):
                raise TraceValidationError("runtime manifest root must be an object")
            existing = TargetedTraceManifest.from_dict(raw)
            if existing.run_id != run_id:
                raise TraceValidationError(
                    f"runtime manifest run mismatch: {existing.run_id} != {run_id}"
                )

        merged_provenance = dict(existing.provenance) if existing else {}
        for key, value in provenance.items():
            if (
                key in merged_provenance
                and merged_provenance[key] is not None
                and value is not None
                and merged_provenance[key] != value
            ):
                raise TraceValidationError(
                    f"runtime manifest provenance conflict for {key!r}"
                )
            if value is not None:
                merged_provenance[key] = value

        target_map: dict[tuple[str, str], dict[str, Any]] = {}
        for raw_target in [*(existing.targets if existing else ()), *targets]:
            target = dict(raw_target)
            key = (
                str(target.get("target_id", "")),
                str(target.get("variant_id", "baseline")),
            )
            if key not in target_map:
                target_map[key] = target
                continue
            current = target_map[key]
            for field_name in ("package", "image", "source"):
                old_value = current.get(field_name)
                new_value = target.get(field_name)
                if (
                    old_value is not None
                    and new_value is not None
                    and old_value != new_value
                ):
                    raise TraceValidationError(
                        f"runtime target {key!r} conflicts on {field_name}"
                    )
                if old_value is None and new_value is not None:
                    current[field_name] = new_value
            current["name_patterns"] = sorted(
                {
                    *current.get("name_patterns", []),
                    *target.get("name_patterns", []),
                }
            )
            for field_name in ("source_hashes", "provenance_hashes"):
                merged_hashes = dict(current.get(field_name, {}))
                for hash_name, digest in dict(target.get(field_name, {})).items():
                    if hash_name in merged_hashes and merged_hashes[hash_name] != digest:
                        raise TraceValidationError(
                            f"runtime target {key!r} conflicts on {field_name}."
                            f"{hash_name}"
                        )
                    merged_hashes[hash_name] = digest
                current[field_name] = merged_hashes

        receipt_map = {
            (item.rank, item.pid, Path(item.path).name): item
            for item in (existing.shards if existing else ())
        }
        receipt_map[(receipt.rank, receipt.pid, Path(receipt.path).name)] = receipt
        manifest = TargetedTraceManifest(
            run_id=run_id,
            acquisition_backend="python_runtime",
            targets=tuple(target_map[key] for key in sorted(target_map)),
            shards=tuple(receipt_map[key] for key in sorted(receipt_map)),
            provenance=merged_provenance,
            created_at=existing.created_at if existing else utc_now(),
        )
        write_manifest(path, manifest)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return manifest


def default_shard_path(output_dir: Path, *, rank: int, pid: int) -> Path:
    """Return the conventional unique shard path."""

    if rank < 0 or pid < 0:
        raise TraceValidationError("rank and pid must be non-negative")
    return Path(output_dir) / "shards" / f"trace_pid{pid}_rank{rank}.jsonl"
