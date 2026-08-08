"""Streaming integrity validation and bounded aggregation for trace shards."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from .schema import (
    ZERO_CHECKSUM,
    ShardCounters,
    ShardReceipt,
    TargetedTraceManifest,
    TargetedTraceRecord,
    TraceValidationError,
    canonical_json,
    validate_envelope,
)


MAX_JSONL_LINE_BYTES = 16 * 1024 * 1024


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


@dataclass
class ShardValidationResult:
    """Validation outcome for one streamed shard."""

    path: str
    valid: bool = False
    complete: bool = False
    event_count: int = 0
    byte_count: int = 0
    file_sha256: str = ""
    sequence_end: int = -1
    chain_checksum: str = ZERO_CHECKSUM
    counters: Optional[ShardCounters] = None
    rank: Optional[int] = None
    pid: Optional[int] = None
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "valid": self.valid,
            "complete": self.complete,
            "event_count": self.event_count,
            "byte_count": self.byte_count,
            "file_sha256": self.file_sha256,
            "sequence_end": self.sequence_end,
            "chain_checksum": self.chain_checksum,
            "counters": self.counters.to_dict() if self.counters else None,
            "rank": self.rank,
            "pid": self.pid,
            "issues": list(self.issues),
        }


def _receipt_mismatches(
    result: ShardValidationResult, receipt: ShardReceipt
) -> List[str]:
    mismatches: List[str] = []
    checks = {
        "rank": (result.rank, receipt.rank),
        "pid": (result.pid, receipt.pid),
        "sequence_end": (result.sequence_end, receipt.sequence_end),
        "chain_checksum": (result.chain_checksum, receipt.chain_checksum),
        "file_sha256": (result.file_sha256, receipt.file_sha256),
        "byte_count": (result.byte_count, receipt.byte_count),
        "complete": (result.complete, receipt.complete),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            mismatches.append(
                f"receipt {name} mismatch: expected {expected!r}, got {actual!r}"
            )
    if result.counters and result.counters.to_dict() != receipt.counters.to_dict():
        mismatches.append("receipt counters mismatch")
    return mismatches


def validate_shard(
    path: Path,
    *,
    expected_receipt: Optional[ShardReceipt] = None,
    on_event: Optional[Callable[[TargetedTraceRecord], None]] = None,
) -> ShardValidationResult:
    """Validate *path* one line at a time, including sequence/checksum/sentinel."""

    path = Path(path)
    result = ShardValidationResult(path=str(path))
    expected_sequence = 0
    previous_checksum = ZERO_CHECKSUM
    file_hash = hashlib.sha256()
    saw_header = False
    saw_end = False

    try:
        stream = path.open("rb")
    except OSError as exc:
        result.issues.append(f"open failed: {exc}")
        return result

    with stream:
        for line_number, raw_line in enumerate(stream, 1):
            result.byte_count += len(raw_line)
            file_hash.update(raw_line)
            if len(raw_line) > MAX_JSONL_LINE_BYTES:
                result.issues.append(
                    f"line {line_number}: exceeds {MAX_JSONL_LINE_BYTES} byte limit"
                )
                break
            if saw_end:
                result.issues.append(f"line {line_number}: data after end sentinel")
                break
            try:
                raw = json.loads(raw_line, parse_constant=_reject_json_constant)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                result.issues.append(f"line {line_number}: corrupt JSON tail: {exc}")
                break
            if not isinstance(raw, Mapping):
                result.issues.append(f"line {line_number}: envelope is not an object")
                break
            try:
                envelope = validate_envelope(
                    raw,
                    expected_sequence=expected_sequence,
                    previous_checksum=previous_checksum,
                )
            except (TraceValidationError, TypeError, ValueError, OverflowError) as exc:
                result.issues.append(f"line {line_number}: {exc}")
                break

            record_type = envelope["record_type"]
            payload = envelope["payload"]
            if expected_sequence == 0 and record_type != "header":
                result.issues.append("line 1: first envelope is not a header")
                break
            if record_type == "header":
                if saw_header or expected_sequence != 0:
                    result.issues.append(f"line {line_number}: duplicate header")
                    break
                saw_header = True
                try:
                    result.rank = int(payload["rank"])
                    result.pid = int(payload["pid"])
                except (KeyError, TypeError, ValueError):
                    result.issues.append(f"line {line_number}: invalid header rank/pid")
                    break
            elif record_type == "event":
                if not saw_header:
                    result.issues.append(f"line {line_number}: event before header")
                    break
                try:
                    record = TargetedTraceRecord.from_dict(payload)
                except (TraceValidationError, TypeError, ValueError, OverflowError) as exc:
                    result.issues.append(f"line {line_number}: invalid event: {exc}")
                    break
                if record.context.rank != result.rank or record.context.pid != result.pid:
                    result.issues.append(
                        f"line {line_number}: event rank/pid differs from header"
                    )
                    break
                result.event_count += 1
                if on_event:
                    on_event(record)
            elif record_type == "end":
                if not saw_header:
                    result.issues.append(f"line {line_number}: end before header")
                    break
                try:
                    result.counters = ShardCounters.from_dict(payload["counters"])
                except (KeyError, TypeError, TraceValidationError) as exc:
                    result.issues.append(f"line {line_number}: invalid end counters: {exc}")
                    break
                if result.counters.written != result.event_count:
                    result.issues.append(
                        "end counters written does not match event envelope count"
                    )
                    break
                saw_end = True

            result.sequence_end = expected_sequence
            result.chain_checksum = str(envelope["checksum"])
            previous_checksum = result.chain_checksum
            expected_sequence += 1

    result.file_sha256 = file_hash.hexdigest()
    result.complete = saw_end
    if not saw_header:
        result.issues.append("missing header")
    if not saw_end:
        result.issues.append("missing end sentinel")
    if expected_receipt is not None:
        result.issues.extend(_receipt_mismatches(result, expected_receipt))
    result.valid = not result.issues and result.complete
    return result


def postprocess_trace_dir(
    trace_dir: Path,
    *,
    output_path: Optional[Path] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Validate/aggregate a targeted trace directory without loading its events."""

    trace_dir = Path(trace_dir)
    manifest_path = trace_dir / "manifest.json"
    manifest: Optional[TargetedTraceManifest] = None
    manifest_error: Optional[str] = None
    if manifest_path.is_file():
        try:
            raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(raw_manifest, Mapping):
                raise TraceValidationError("manifest root must be an object")
            manifest = TargetedTraceManifest.from_dict(raw_manifest)
        except (OSError, json.JSONDecodeError, TraceValidationError) as exc:
            manifest_error = str(exc)
    else:
        manifest_error = "manifest.json is missing"

    receipts: Dict[str, ShardReceipt] = {}
    if manifest is not None:
        for receipt in manifest.shards:
            receipt_path = Path(receipt.path)
            receipts[str(receipt_path.resolve())] = receipt
            receipts[receipt_path.name] = receipt

    aggregates: Dict[str, Dict[str, int]] = {
        "by_target": {},
        "by_kind": {},
        "by_rank": {},
    }
    semantic_missing = {
        "phase": 0,
        "source": 0,
        "grid": 0,
        "shape": 0,
        "correlation": 0,
    }
    complete_semantic_records = 0
    torch_profiler_records = 0

    def observe(record: TargetedTraceRecord) -> None:
        nonlocal complete_semantic_records, torch_profiler_records
        target_id = record.identity.target_id
        aggregates["by_target"][target_id] = (
            aggregates["by_target"].get(target_id, 0) + 1
        )
        aggregates["by_kind"][record.kind] = (
            aggregates["by_kind"].get(record.kind, 0) + 1
        )
        rank = str(record.context.rank)
        aggregates["by_rank"][rank] = aggregates["by_rank"].get(rank, 0) + 1

        missing = []
        if not record.context.stage or record.context.stage.lower() == "unknown":
            missing.append("phase")
        if record.semantics.source is None:
            missing.append("source")
        if record.runtime.grid is None and record.semantics.python_grid is None:
            missing.append("grid")
        if not record.semantics.tensors:
            missing.append("shape")
        if record.kind == "torch_profiler_kernel":
            torch_profiler_records += 1
            if record.runtime.correlation_id is None:
                missing.append("correlation")
        for field_name in missing:
            semantic_missing[field_name] += 1
        if not missing:
            complete_semantic_records += 1

    shard_paths = sorted((trace_dir / "shards").glob("*.jsonl"))
    if not shard_paths:
        shard_paths = sorted(trace_dir.glob("*.jsonl"))
    validations: List[ShardValidationResult] = []
    for path in shard_paths:
        receipt = receipts.get(str(path.resolve())) or receipts.get(path.name)
        validations.append(
            validate_shard(path, expected_receipt=receipt, on_event=observe)
        )

    counter_items = [item.counters for item in validations if item.counters]
    coverage = (
        ShardCounters.aggregate(counter_items).to_dict()
        if counter_items
        else ShardCounters().to_dict()
    )
    issues: List[str] = []
    if manifest_error:
        issues.append(f"manifest: {manifest_error}")
    if manifest is not None:
        adapter_warnings = manifest.provenance.get("adapter_warnings", [])
        if isinstance(adapter_warnings, list):
            issues.extend(f"acquisition: {warning}" for warning in adapter_warnings)
    if manifest is not None:
        observed_names = {Path(item.path).name for item in validations}
        declared_names = {Path(item.path).name for item in manifest.shards}
        for receipt in manifest.shards:
            if Path(receipt.path).name not in observed_names:
                issues.append(f"manifest shard missing: {receipt.path}")
        for undeclared in sorted(observed_names - declared_names):
            issues.append(f"undeclared trace shard: {undeclared}")
    for item in validations:
        issues.extend(f"{item.path}: {issue}" for issue in item.issues)
    if not shard_paths:
        issues.append(f"no trace shards found under {trace_dir}")

    integrity_failures: Dict[str, int] = {}
    for issue in issues:
        if "corrupt JSON tail" in issue:
            reason = "corrupt_tail"
        elif "missing end sentinel" in issue:
            reason = "missing_end_sentinel"
        elif "checksum" in issue:
            reason = "checksum"
        elif "receipt" in issue:
            reason = "receipt"
        elif issue.startswith("acquisition:"):
            reason = "acquisition"
        elif issue.startswith("manifest:"):
            reason = "manifest"
        else:
            reason = "other"
        integrity_failures[reason] = integrity_failures.get(reason, 0) + 1

    seen = int(coverage["seen"])
    written = int(coverage["written"])
    dropped = int(coverage["dropped"])
    record_coverage_fraction = written / seen if seen else 0.0
    lossless_record_coverage = seen > 0 and dropped == 0 and written == seen
    complete_semantic_coverage = (
        written > 0 and complete_semantic_records == written
    )
    semantic_coverage_claimed = (
        not issues and lossless_record_coverage and complete_semantic_coverage
    )
    unresolved_reasons = []
    if not seen:
        unresolved_reasons.append("no_records")
    if dropped:
        unresolved_reasons.extend(
            f"dropped:{reason}"
            for reason in sorted(coverage["dropped_by_reason"])
        )
    unresolved_reasons.extend(
        f"missing:{field_name}"
        for field_name, count in semantic_missing.items()
        if count
    )
    if issues:
        unresolved_reasons.append("integrity_validation_failed")

    evidence_quality = {
        "evidence_class": "diagnostic_only",
        "resolution_status": (
            "resolved" if semantic_coverage_claimed else "unresolved"
        ),
        "semantic_coverage_claimed": semantic_coverage_claimed,
        "record_coverage_fraction": record_coverage_fraction,
        "lossless_record_coverage": lossless_record_coverage,
        "records_evaluated": written,
        "records_with_complete_semantics": complete_semantic_records,
        "missing_by_field": semantic_missing,
        # The postprocessor never synthesizes a CPU/source-to-GPU join.  A
        # correlation ID merely reports that a future trusted consumer has a
        # join key available.
        "cross_event_join": "not_performed",
        "join_eligible_records": (
            torch_profiler_records - semantic_missing["correlation"]
        ),
        "unresolved_reasons": unresolved_reasons,
    }

    summary: Dict[str, Any] = {
        "schema_name": manifest.schema_name if manifest else None,
        "schema_version": manifest.schema_version if manifest else None,
        "run_id": manifest.run_id if manifest else None,
        "valid": not issues and all(item.valid for item in validations),
        "streaming": True,
        "coverage": coverage,
        "evidence_quality": evidence_quality,
        "events": aggregates,
        "integrity_failures_by_reason": dict(sorted(integrity_failures.items())),
        "shards": [item.to_dict() for item in validations],
        "issues": issues,
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if strict and issues:
        raise TraceValidationError("; ".join(issues))
    return summary
