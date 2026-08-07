"""Validate and attest caller-supplied, hash-locked lm-eval runtimes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .config import LmEvalRuntimeConfig


LM_EVAL_MANIFEST_FILENAME = "lm_eval_runtime_manifest.json"
LM_EVAL_MANIFEST_SCHEMA = "apex.lm-eval-runtime/v1"
LM_EVAL_RECEIPT_FILENAME = "lm_eval_runtime_receipt.json"
LM_EVAL_RECEIPT_SCHEMA = "magpie.lm-eval-runtime-receipt/v1"
LM_EVAL_EVIDENCE_SCHEMA = "magpie.lm-eval-runtime-evidence/v1"
MAX_MANIFEST_SIZE_BYTES = 64 * 1024 * 1024
MAX_RECEIPT_SIZE_BYTES = 256 * 1024

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_REPO_DIGEST = re.compile(r"^(?:[^\s@]+@)?sha256:[0-9a-f]{64}$")
_PYTHON_ABI = re.compile(r"^[a-z0-9_]+-[a-z0-9_]+$")
_MANIFEST_KEYS = frozenset(
    {"schema", "runtime_sha256", "site_packages", "identity", "files"}
)
_FILE_KEYS = frozenset({"path", "size_bytes", "mode", "sha256"})
_IDENTITY_KEYS = frozenset(
    {
        "lm_eval_commit",
        "lm_eval_tree",
        "lm_eval_version",
        "python_abi",
        "base_image_id",
        "base_image_repo_digest",
        "inferencex_commit",
        "inferencex_tree",
    }
)
_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "runtime_sha256",
        "identity",
        "manifest_sha256",
        "site_packages",
        "python_abi",
        "lm_eval_version",
        "lm_eval_module",
        "execution_mode",
        "read_only_mount",
        "verified",
    }
)


@dataclass(frozen=True)
class LmEvalRuntime:
    """A fully verified host runtime ready for local or read-only Docker use."""

    root: Path
    site_packages: Path
    runtime_sha256: str
    identity: Dict[str, Any]
    manifest_bytes: bytes
    manifest_sha256: str


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _read_json_file(
    path: Path,
    *,
    limit: int,
    require_readonly: bool = False,
) -> tuple[Mapping[str, Any], bytes]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"{path.name} must be a regular file with nlink=1")
    if require_readonly and info.st_mode & 0o222:
        raise ValueError(f"{path.name} must not have writable permission bits")
    if info.st_size <= 0 or info.st_size > limit:
        raise ValueError(f"{path.name} has invalid size {info.st_size}")
    raw = path.read_bytes()
    payload = json.loads(
        raw.decode("utf-8"),
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path.name} root must be a JSON object")
    return payload, raw


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(identity: Mapping[str, Any], files: Sequence[Any]) -> str:
    encoded = json.dumps(
        {"identity": identity, "files": files},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _validate_identity(identity: Any) -> Dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise ValueError("manifest identity must be a JSON object")
    missing = sorted(_IDENTITY_KEYS - set(identity))
    if missing:
        raise ValueError(f"manifest identity is missing required keys: {missing}")
    result = dict(identity)
    for key in ("lm_eval_commit", "lm_eval_tree", "inferencex_commit", "inferencex_tree"):
        if not isinstance(result[key], str) or not _HEX40.fullmatch(result[key]):
            raise ValueError(f"identity.{key} must be an exact lowercase 40-hex id")
    version = result["lm_eval_version"]
    if not isinstance(version, str) or not version.strip():
        raise ValueError("identity.lm_eval_version must be a non-empty string")
    abi = result["python_abi"]
    if not isinstance(abi, str) or not _PYTHON_ABI.fullmatch(abi):
        raise ValueError("identity.python_abi must be sys.implementation.cache_tag")
    if not isinstance(result["base_image_id"], str) or not _IMAGE_SHA256.fullmatch(
        result["base_image_id"]
    ):
        raise ValueError("identity.base_image_id must be an immutable image ID")
    repo_digest = result["base_image_repo_digest"]
    if not isinstance(repo_digest, str) or not _REPO_DIGEST.fullmatch(repo_digest):
        raise ValueError("identity.base_image_repo_digest must contain a sha256 digest")
    return result


def _validate_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("file record path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or ".." in path.parts:
        raise ValueError(f"file record path is not canonical and relative: {value!r}")
    return value


def _validate_manifest_records(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("manifest files must be a non-empty list")
    records: List[Dict[str, Any]] = []
    paths: List[str] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _FILE_KEYS:
            raise ValueError("each file record must have path/size_bytes/mode/sha256")
        path = _validate_relative_path(item.get("path"))
        size = item.get("size_bytes")
        mode = item.get("mode")
        digest = item.get("sha256")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"invalid size_bytes for {path}")
        if not isinstance(mode, int) or isinstance(mode, bool) or not 0 <= mode <= 0o7777:
            raise ValueError(f"invalid mode for {path}")
        if mode & 0o222:
            raise ValueError(f"file record is writable: {path}")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"invalid sha256 for {path}")
        paths.append(path)
        records.append(dict(item))
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("manifest file records must be unique and path-sorted")
    return records


def _verify_site_packages(site_packages: Path, records: Sequence[Mapping[str, Any]]) -> None:
    actual_files: List[str] = []
    for path in sorted(site_packages.rglob("*"), key=lambda item: item.as_posix()):
        info = path.lstat()
        relative = path.relative_to(site_packages).as_posix()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"runtime symlink is forbidden: {relative}")
        if stat.S_ISDIR(info.st_mode):
            if info.st_mode & 0o222:
                raise ValueError(f"runtime directory is writable: {relative}")
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"runtime entry must be a regular nlink=1 file: {relative}")
        actual_files.append(relative)
    expected_files = [str(item["path"]) for item in records]
    if actual_files != expected_files:
        raise ValueError("site-packages files do not exactly match the manifest")
    for item in records:
        path = site_packages / str(item["path"])
        info = path.lstat()
        if info.st_size != item["size_bytes"]:
            raise ValueError(f"size mismatch for {item['path']}")
        if stat.S_IMODE(info.st_mode) != item["mode"]:
            raise ValueError(f"mode mismatch for {item['path']}")
        if _sha256_file(path) != item["sha256"]:
            raise ValueError(f"content digest mismatch for {item['path']}")


def validate_lm_eval_runtime(config: LmEvalRuntimeConfig) -> LmEvalRuntime:
    """Fail closed unless the configured runtime exactly matches its manifest."""

    root = Path(config.path)
    if not root.is_absolute():
        raise ValueError("lm_eval_runtime.path must be absolute")
    root_info = root.lstat()
    if not stat.S_ISDIR(root_info.st_mode) or root.is_symlink():
        raise ValueError("lm_eval_runtime.path must be a real directory")
    if root_info.st_mode & 0o222:
        raise ValueError("lm_eval_runtime root must not be writable")
    entries = {path.name for path in root.iterdir()}
    if entries != {LM_EVAL_MANIFEST_FILENAME, "site-packages"}:
        raise ValueError("lm_eval_runtime root must contain only manifest and site-packages")

    site_packages = root / "site-packages"
    site_info = site_packages.lstat()
    if not stat.S_ISDIR(site_info.st_mode) or site_packages.is_symlink():
        raise ValueError("lm_eval_runtime site-packages must be a real directory")
    if site_info.st_mode & 0o222:
        raise ValueError("lm_eval_runtime site-packages must not be writable")

    manifest_path = root / LM_EVAL_MANIFEST_FILENAME
    manifest, manifest_bytes = _read_json_file(
        manifest_path,
        limit=MAX_MANIFEST_SIZE_BYTES,
        require_readonly=True,
    )
    if set(manifest) != _MANIFEST_KEYS:
        raise ValueError("lm_eval runtime manifest keys do not match the v1 contract")
    if manifest.get("schema") != LM_EVAL_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported lm_eval runtime schema: {manifest.get('schema')!r}")
    if manifest.get("site_packages") != "site-packages":
        raise ValueError("manifest site_packages must be exactly 'site-packages'")
    identity = _validate_identity(manifest.get("identity"))
    if identity != config.identity:
        raise ValueError("manifest identity does not exactly match benchmark config")
    records = _validate_manifest_records(manifest.get("files"))
    computed = _canonical_sha256(identity, records)
    declared = manifest.get("runtime_sha256")
    if not isinstance(declared, str) or not _SHA256.fullmatch(declared):
        raise ValueError("manifest runtime_sha256 is invalid")
    if computed != declared or declared != config.sha256:
        raise ValueError("lm_eval runtime digest does not match manifest/config")
    _verify_site_packages(site_packages, records)
    return LmEvalRuntime(
        root=root.resolve(),
        site_packages=site_packages.resolve(),
        runtime_sha256=declared,
        identity=identity,
        manifest_bytes=manifest_bytes,
        manifest_sha256=_sha256_bytes(manifest_bytes),
    )


def snapshot_runtime_manifest(runtime: LmEvalRuntime, workspace: Path) -> Path:
    """Atomically preserve the exact consumed manifest in the run workspace."""

    destination = Path(workspace) / LM_EVAL_MANIFEST_FILENAME
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(runtime.manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _base_evidence(
    config: LmEvalRuntimeConfig | None,
    status: str,
    *,
    requested: bool,
) -> Dict[str, Any]:
    return {
        "schema": LM_EVAL_EVIDENCE_SCHEMA,
        "requested": requested,
        "status": status,
        "verified": status == "verified",
        "evidence_present": False,
        "runtime_sha256": config.sha256 if config else None,
        "identity": dict(config.identity) if config else None,
        "mount_mode": None,
        "manifest_artifact": None,
        "receipt_artifact": None,
        "errors": [],
    }


def invalid_runtime_evidence(
    config: LmEvalRuntimeConfig | None,
    error: str,
    *,
    status: str = "invalid",
) -> Dict[str, Any]:
    evidence = _base_evidence(config, status, requested=True)
    evidence["errors"] = [error]
    return evidence


def collect_lm_eval_runtime_evidence(
    workspace: Path,
    *,
    requested: bool,
    config: LmEvalRuntimeConfig | None,
    execution_mode: str,
) -> Dict[str, Any]:
    """Validate the in-run receipt and preserved manifest for report evidence."""

    if not requested:
        return _base_evidence(config, "not_requested", requested=False)
    if config is None:
        return invalid_runtime_evidence(
            None,
            "RUN_EVAL=true requires benchmark.lm_eval_runtime",
        )
    evidence = _base_evidence(config, "missing", requested=True)
    receipt_path = Path(workspace) / LM_EVAL_RECEIPT_FILENAME
    manifest_path = Path(workspace) / LM_EVAL_MANIFEST_FILENAME
    try:
        payload, receipt_bytes = _read_json_file(
            receipt_path,
            limit=MAX_RECEIPT_SIZE_BYTES,
        )
        manifest, manifest_bytes = _read_json_file(
            manifest_path,
            limit=MAX_MANIFEST_SIZE_BYTES,
        )
        if set(payload) != _RECEIPT_KEYS:
            raise ValueError("runtime receipt keys do not match the v1 contract")
        if payload.get("schema") != LM_EVAL_RECEIPT_SCHEMA:
            raise ValueError("unsupported lm_eval runtime receipt schema")
        if payload.get("verified") is not True:
            raise ValueError("runtime receipt verified must be true")
        if payload.get("runtime_sha256") != config.sha256:
            raise ValueError("runtime receipt digest does not match benchmark config")
        if payload.get("identity") != config.identity:
            raise ValueError("runtime receipt identity does not match benchmark config")
        if payload.get("manifest_sha256") != _sha256_bytes(manifest_bytes):
            raise ValueError("runtime receipt manifest digest does not match artifact")
        if manifest.get("runtime_sha256") != config.sha256:
            raise ValueError("manifest artifact runtime digest does not match config")
        if manifest.get("identity") != config.identity:
            raise ValueError("manifest artifact identity does not match config")
        if payload.get("site_packages") != "site-packages":
            raise ValueError("runtime receipt site_packages is invalid")
        if payload.get("python_abi") != config.identity.get("python_abi"):
            raise ValueError("actual Python ABI does not match runtime identity")
        if payload.get("lm_eval_version") != config.identity.get("lm_eval_version"):
            raise ValueError("actual lm_eval version does not match runtime identity")
        if payload.get("execution_mode") != execution_mode:
            raise ValueError("runtime receipt execution mode does not match benchmark")
        read_only_mount = payload.get("read_only_mount")
        if not isinstance(read_only_mount, bool):
            raise ValueError("runtime receipt read_only_mount must be boolean")
        if execution_mode == "docker" and read_only_mount is not True:
            raise ValueError("Docker evaluator runtime was not mounted read-only")
        module = payload.get("lm_eval_module")
        if not isinstance(module, str) or not module.startswith("site-packages/lm_eval/"):
            raise ValueError("lm_eval module did not import from the supplied runtime")
        manifest_info = manifest_path.stat()
        receipt_info = receipt_path.stat()
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return invalid_runtime_evidence(config, str(exc))

    evidence.update(
        {
            "status": "verified",
            "verified": True,
            "evidence_present": True,
            "mount_mode": "read_only" if execution_mode == "docker" else "local",
            "manifest_artifact": {
                "path": LM_EVAL_MANIFEST_FILENAME,
                "size_bytes": manifest_info.st_size,
                "sha256": _sha256_bytes(manifest_bytes),
            },
            "receipt_artifact": {
                "path": LM_EVAL_RECEIPT_FILENAME,
                "size_bytes": receipt_info.st_size,
                "sha256": _sha256_bytes(receipt_bytes),
            },
            "errors": [],
        }
    )
    return evidence
