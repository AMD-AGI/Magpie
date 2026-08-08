"""Validate model revision evidence emitted by serving benchmark scripts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple


MODEL_REVISION_RECEIPT_FILENAME = "model_revision_receipt.json"
MODEL_REVISION_RECEIPT_SCHEMA = "magpie.model-revision-receipt/v1"
MODEL_REVISION_EVIDENCE_SCHEMA = "magpie.model-revision-evidence/v1"
MAX_RECEIPT_SIZE_BYTES = 64 * 1024

_EXACT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "model",
        "requested_revision",
        "resolved_revision",
        "snapshot_path",
        "verified",
    }
)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence(
    *,
    model: str,
    requested_revision: str | None,
    status: str,
    error: str | None = None,
) -> Dict[str, Any]:
    requested = requested_revision is not None
    return {
        "schema": MODEL_REVISION_EVIDENCE_SCHEMA,
        "requested": requested,
        "status": status,
        "verified": status == "verified",
        "evidence_present": False,
        "model": model,
        "requested_revision": requested_revision,
        "resolved_revision": None,
        "snapshot_path": None,
        "receipt_artifact": None,
        "errors": [error] if error else [],
    }


def _validate_payload(
    payload: Mapping[str, Any],
    *,
    model: str,
    requested_revision: str,
) -> str | None:
    keys = set(payload)
    if keys != _RECEIPT_KEYS:
        missing = sorted(_RECEIPT_KEYS - keys)
        unexpected = sorted(keys - _RECEIPT_KEYS)
        return (
            "receipt keys do not match the v1 contract "
            f"(missing={missing}, unexpected={unexpected})"
        )
    if payload.get("schema") != MODEL_REVISION_RECEIPT_SCHEMA:
        return f"unsupported receipt schema: {payload.get('schema')!r}"
    if payload.get("model") != model:
        return (
            "receipt model does not match the benchmark config: "
            f"{payload.get('model')!r} != {model!r}"
        )
    if payload.get("requested_revision") != requested_revision:
        return "receipt requested_revision does not match MODEL_REVISION"

    resolved_revision = payload.get("resolved_revision")
    if not isinstance(resolved_revision, str) or not _EXACT_COMMIT_RE.fullmatch(
        resolved_revision
    ):
        return "receipt resolved_revision is not an exact lowercase 40-hex commit"
    if resolved_revision != requested_revision:
        return "resolved model revision does not match MODEL_REVISION"

    snapshot_path = payload.get("snapshot_path")
    if not isinstance(snapshot_path, str) or not snapshot_path.strip():
        return "receipt snapshot_path must be a non-empty string"
    if not Path(snapshot_path).is_absolute():
        return "receipt snapshot_path must be absolute"
    if Path(snapshot_path).name != resolved_revision:
        return "receipt snapshot_path does not name the resolved revision"
    if payload.get("verified") is not True:
        return "receipt verified must be true"
    return None


def collect_model_revision_evidence(
    workspace: Path,
    *,
    model: str,
    requested_revision: Any,
) -> Dict[str, Any]:
    """Collect a bounded, validated model revision receipt from ``workspace``.

    No revision request is a supported legacy mode and returns ``not_requested``.
    Once ``MODEL_REVISION`` is supplied, every malformed or absent receipt is a
    failed evidence gate; callers must not treat agent text or server logs as a
    substitute.
    """

    if requested_revision in (None, ""):
        return _evidence(
            model=model,
            requested_revision=None,
            status="not_requested",
        )

    requested = str(requested_revision).strip()
    if not _EXACT_COMMIT_RE.fullmatch(requested):
        return _evidence(
            model=model,
            requested_revision=requested,
            status="invalid",
            error="MODEL_REVISION must be an exact lowercase 40-hex commit",
        )

    evidence = _evidence(
        model=model,
        requested_revision=requested,
        status="missing",
        error=(
            f"{MODEL_REVISION_RECEIPT_FILENAME} is missing for requested "
            "MODEL_REVISION"
        ),
    )
    receipt_path = Path(workspace) / MODEL_REVISION_RECEIPT_FILENAME
    try:
        if receipt_path.is_symlink():
            raise ValueError("receipt must be a regular file, not a symlink")
        if not receipt_path.is_file():
            return evidence
        size_bytes = receipt_path.stat().st_size
        if size_bytes <= 0 or size_bytes > MAX_RECEIPT_SIZE_BYTES:
            raise ValueError(
                "receipt size must be between 1 and "
                f"{MAX_RECEIPT_SIZE_BYTES} bytes, got {size_bytes}"
            )
        payload = json.loads(
            receipt_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
        if not isinstance(payload, Mapping):
            raise ValueError("receipt root must be a JSON object")
        validation_error = _validate_payload(
            payload,
            model=model,
            requested_revision=requested,
        )
        if validation_error:
            raise ValueError(validation_error)
        receipt_sha256 = _sha256(receipt_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return _evidence(
            model=model,
            requested_revision=requested,
            status="invalid",
            error=str(exc),
        )

    evidence.update(
        {
            "status": "verified",
            "verified": True,
            "evidence_present": True,
            "resolved_revision": payload["resolved_revision"],
            "snapshot_path": payload["snapshot_path"],
            "receipt_artifact": {
                "path": MODEL_REVISION_RECEIPT_FILENAME,
                "size_bytes": size_bytes,
                "sha256": receipt_sha256,
            },
            "errors": [],
        }
    )
    return evidence
