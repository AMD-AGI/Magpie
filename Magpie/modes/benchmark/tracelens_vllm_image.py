###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Deterministic, minimal TraceLens image derivation for vLLM.

The upstream TraceLens installer intentionally serves many workflows and pulls
large, loosely resolved dependencies (including xprof).  Magpie's vLLM
inference path needs a much smaller surface: the trace splitter, the CSV report
generator, Matplotlib (eagerly imported by the architecture helper), and the
vLLM instrumentation patch.  This module builds exactly that surface without
changing packages already present in the vLLM base image.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional, Sequence

VLLM_TRACELENS_REQUIREMENTS = (
    ("contourpy", "1.3.3"),
    ("cycler", "0.12.1"),
    ("fonttools", "4.63.0"),
    ("kiwisolver", "1.5.0"),
    ("matplotlib", "3.11.1"),
    ("pyparsing", "3.3.2"),
)
VLLM_TRACELENS_FORBIDDEN = ("xprof", "gcsfs", "grpcio-status")
VLLM_TRACELENS_SCHEMA = "magpie.tracelens-vllm-runtime/v1"
LABEL_PREFIX = "io.magpie.tracelens"
LABEL_SCHEMA = f"{LABEL_PREFIX}.schema"
LABEL_BASE_ID = f"{LABEL_PREFIX}.base-image-id"
LABEL_BASE_LOCATOR = f"{LABEL_PREFIX}.base-image-locator"
LABEL_VLLM_VERSION = f"{LABEL_PREFIX}.vllm-version"
LABEL_GRPCIO_VERSION = f"{LABEL_PREFIX}.grpcio-version"
LABEL_SOURCE_COMMIT = f"{LABEL_PREFIX}.source-commit"
LABEL_SOURCE_TREE = f"{LABEL_PREFIX}.source-tree"
LABEL_PATCH_VERSION = f"{LABEL_PREFIX}.patch-version"
LABEL_PATCH_SHA256 = f"{LABEL_PREFIX}.patch-sha256"
LABEL_WHEEL_MANIFEST = f"{LABEL_PREFIX}.wheel-manifest"
LABEL_WHEEL_MANIFEST_SHA256 = f"{LABEL_PREFIX}.wheel-manifest-sha256"
LABEL_DEPENDENCY_POLICY = f"{LABEL_PREFIX}.dependency-policy"
DEPENDENCY_POLICY = "minimal-pinned-wheels-no-deps"

_SOURCE_PATHS = (
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "setup.py",
    "TraceLens",
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_ID_RE = re.compile(r"^sha256:([0-9a-f]{64})$")
_REPO_DIGEST_RE = re.compile(r"^[a-z0-9][a-z0-9._:/-]*@sha256:[0-9a-f]{64}$")
_LOCAL_BASE_REPOSITORY = "localhost/magpie-tracelens-vllm-base"


@dataclass(frozen=True)
class VllmTraceLensIdentity:
    """Immutable inputs that define a derived TraceLens vLLM image."""

    base_image: str
    base_image_id: str
    base_image_locator: str
    vllm_version: str
    grpcio_version: str
    source_commit: str
    source_tree: str
    patch_version: str
    patch_path: str
    patch_sha256: str
    patch_bytes: bytes = field(repr=False)

    def labels(self) -> Dict[str, str]:
        return {
            LABEL_SCHEMA: VLLM_TRACELENS_SCHEMA,
            LABEL_BASE_ID: self.base_image_id,
            LABEL_BASE_LOCATOR: self.base_image_locator,
            LABEL_VLLM_VERSION: self.vllm_version,
            LABEL_GRPCIO_VERSION: self.grpcio_version,
            LABEL_SOURCE_COMMIT: self.source_commit,
            LABEL_SOURCE_TREE: self.source_tree,
            LABEL_PATCH_VERSION: self.patch_version,
            LABEL_PATCH_SHA256: self.patch_sha256,
            LABEL_DEPENDENCY_POLICY: DEPENDENCY_POLICY,
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "base_image_id": self.base_image_id,
            "base_image_locator": self.base_image_locator,
            "runtime_package_version": self.vllm_version,
            "base_grpcio_version": self.grpcio_version,
            "tracelens_source_commit": self.source_commit,
            "tracelens_source_tree": self.source_tree,
            "tracelens_patch_path": self.patch_path,
            "tracelens_patch_sha256": self.patch_sha256,
            "dependency_policy": DEPENDENCY_POLICY,
        }


@dataclass(frozen=True)
class _BuildBaseReference:
    """A build-only Docker reference bound to one locally inspectable image ID."""

    locator: str
    image_id: str
    kind: str
    owns_temporary_tag: bool = False
    retained_locator: Optional[str] = None
    retained_locator_created: bool = False


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completed_output(proc: subprocess.CompletedProcess[Any]) -> str:
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    return (stderr or stdout).strip()


def _git_text(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Could not inspect TraceLens source identity: {_completed_output(proc)}"
        )
    return (proc.stdout or "").strip()


def _git_bytes(repo: Path, *args: str) -> bytes:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Could not read committed TraceLens content: {_completed_output(proc)}"
        )
    return bytes(proc.stdout or b"")


def docker_image_record(image: str) -> Optional[Dict[str, Any]]:
    """Return Docker's inspect record for ``image``, or None when unavailable."""
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        records = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return None
    if not isinstance(records, list) or len(records) != 1:
        return None
    return records[0] if isinstance(records[0], dict) else None


def docker_image_id(image: str) -> Optional[str]:
    record = docker_image_record(image)
    image_id = record.get("Id") if record else None
    return image_id if isinstance(image_id, str) and image_id else None


def _require_expected_image_id(image: str, expected_id: str, *, role: str) -> None:
    actual_id = docker_image_id(image)
    if actual_id != expected_id:
        raise RuntimeError(
            f"{role} is not bound to the expected local Docker image ID: "
            f"reference={image!r}, expected={expected_id!r}, actual={actual_id!r}"
        )


def _temporary_local_base_tag(
    image_id: str,
    *,
    nonce: Optional[str] = None,
) -> str:
    match = _IMAGE_ID_RE.fullmatch(image_id)
    if not match:
        raise RuntimeError(f"Invalid local Docker image ID: {image_id!r}")
    unique = nonce or secrets.token_hex(16)
    if not re.fullmatch(r"[0-9a-f]{32}", unique):
        raise RuntimeError(f"Invalid local Docker tag nonce: {unique!r}")
    return f"{_LOCAL_BASE_REPOSITORY}:sha256-{match.group(1)}-{unique}"


def _retained_local_base_tag(image_id: str) -> str:
    """Return the stable local name that keeps an unnamed parent inspectable."""
    match = _IMAGE_ID_RE.fullmatch(image_id)
    if not match:
        raise RuntimeError(f"Invalid local Docker image ID: {image_id!r}")
    return f"{_LOCAL_BASE_REPOSITORY}:sha256-{match.group(1)}"


def _docker_tag_image(image_id: str, tag: str) -> None:
    try:
        proc = subprocess.run(
            ["docker", "image", "tag", image_id, tag],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"Could not create local Docker base tag {tag!r}: {exc}"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"Could not create local Docker base tag {tag!r}: "
            f"{_completed_output(proc)}"
        )


def _docker_remove_tag(tag: str) -> None:
    try:
        proc = subprocess.run(
            ["docker", "image", "rm", tag],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"Could not remove local Docker base tag {tag!r}: {exc}"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"Could not remove local Docker base tag {tag!r}: "
            f"{_completed_output(proc)}"
        )


def _acquire_build_base_reference(
    identity: VllmTraceLensIdentity,
) -> _BuildBaseReference:
    """Resolve an exact build base without treating an image ID as a tag."""
    expected_id = identity.base_image_id
    if not _IMAGE_ID_RE.fullmatch(expected_id):
        raise RuntimeError(f"Invalid local Docker image ID: {expected_id!r}")
    _require_expected_image_id(expected_id, expected_id, role="base image ID")

    if _REPO_DIGEST_RE.fullmatch(identity.base_image_locator):
        _require_expected_image_id(
            identity.base_image_locator,
            expected_id,
            role="repository-digest build base",
        )
        return _BuildBaseReference(
            locator=identity.base_image_locator,
            image_id=expected_id,
            kind="repository-digest",
        )

    retained_tag = _retained_local_base_tag(expected_id)
    retained_id = docker_image_id(retained_tag)
    retained_created = False
    if retained_id is None:
        _docker_tag_image(expected_id, retained_tag)
        _require_expected_image_id(
            retained_tag,
            expected_id,
            role="retained local base tag",
        )
        retained_created = True
    elif retained_id != expected_id:
        raise RuntimeError(
            "Content-addressed local TraceLens base tag resolves to a different "
            "image ID: "
            f"tag={retained_tag!r}, expected={expected_id!r}, actual={retained_id!r}"
        )

    tag = _temporary_local_base_tag(expected_id)
    existing_id = docker_image_id(tag)
    if existing_id is not None:
        raise RuntimeError(
            "Unique local TraceLens base tag already exists; refusing to reuse "
            "a tag this build does not own: "
            f"tag={tag!r}, expected={expected_id!r}, actual={existing_id!r}"
        )

    _docker_tag_image(expected_id, tag)
    try:
        _require_expected_image_id(tag, expected_id, role="temporary build base tag")
    except RuntimeError:
        if docker_image_id(tag) == expected_id:
            _docker_remove_tag(tag)
        raise
    return _BuildBaseReference(
        locator=tag,
        image_id=expected_id,
        kind="temporary-local-tag",
        owns_temporary_tag=True,
        retained_locator=retained_tag,
        retained_locator_created=retained_created,
    )


def _verify_build_base_reference(reference: _BuildBaseReference) -> None:
    if reference.kind == "repository-digest":
        _require_expected_image_id(
            reference.locator,
            reference.image_id,
            role="repository-digest build base",
        )
    else:
        _require_expected_image_id(
            reference.locator,
            reference.image_id,
            role="local build base tag",
        )
        if reference.retained_locator is None:
            raise RuntimeError("Local build base is missing its retained locator")
        _require_expected_image_id(
            reference.retained_locator,
            reference.image_id,
            role="retained local base tag",
        )
    _require_expected_image_id(
        reference.image_id,
        reference.image_id,
        role="base image ID",
    )


def _release_build_base_reference(reference: _BuildBaseReference) -> None:
    if not reference.owns_temporary_tag:
        return
    if reference.retained_locator is None:
        raise RuntimeError("Owned temporary base tag has no retained locator")
    _require_expected_image_id(
        reference.retained_locator,
        reference.image_id,
        role="retained local base tag",
    )
    _require_expected_image_id(
        reference.locator,
        reference.image_id,
        role="owned temporary build base tag",
    )
    _docker_remove_tag(reference.locator)
    remaining_id = docker_image_id(reference.locator)
    if remaining_id is not None:
        raise RuntimeError(
            "Owned temporary build base tag still exists after cleanup: "
            f"tag={reference.locator!r}, actual={remaining_id!r}"
        )
    _require_expected_image_id(
        reference.retained_locator,
        reference.image_id,
        role="retained local base tag",
    )
    _require_expected_image_id(
        reference.image_id,
        reference.image_id,
        role="base image ID after temporary-tag cleanup",
    )


def resolve_vllm_tracelens_identity(
    *,
    base_image: str,
    vllm_version: str,
    grpcio_version: str,
    tracelens_repo: Path,
    patch_version: str,
) -> VllmTraceLensIdentity:
    """Resolve source, patch, and immutable base-image identity."""
    base_record = docker_image_record(base_image)
    base_id = base_record.get("Id") if base_record else None
    if not isinstance(base_id, str) or not _IMAGE_ID_RE.fullmatch(base_id):
        raise RuntimeError(f"Could not resolve Docker image ID for {base_image!r}")
    repo_digests = base_record.get("RepoDigests") or []
    base_locator = (
        repo_digests[0]
        if isinstance(repo_digests, list)
        and repo_digests
        and isinstance(repo_digests[0], str)
        else base_image
    )

    source_commit = _git_text(tracelens_repo, "rev-parse", "HEAD")
    source_tree = _git_text(tracelens_repo, "rev-parse", "HEAD^{tree}")
    minor = patch_version.removeprefix("v")
    if not minor.isdigit():
        raise RuntimeError(f"Invalid TraceLens vLLM patch version: {patch_version!r}")
    patch_path = (
        "examples/custom_workflows/inference_analysis/vllm_patches/"
        f"config_vllm_v0.{minor}.0.patch"
    )
    patch_bytes = _git_bytes(tracelens_repo, "show", f"{source_commit}:{patch_path}")
    if not patch_bytes:
        raise RuntimeError(f"Committed TraceLens patch is empty: {patch_path}")

    return VllmTraceLensIdentity(
        base_image=base_image,
        base_image_id=base_id,
        base_image_locator=base_locator,
        vllm_version=vllm_version,
        grpcio_version=grpcio_version,
        source_commit=source_commit,
        source_tree=source_tree,
        patch_version=patch_version,
        patch_path=patch_path,
        patch_sha256=_sha256_bytes(patch_bytes),
        patch_bytes=patch_bytes,
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _wheel_distribution(path: Path) -> str:
    return path.name.split("-", 1)[0].replace("_", "-").lower()


def _wheel_manifest(wheelhouse: Path) -> list[Dict[str, str]]:
    expected_versions = dict(VLLM_TRACELENS_REQUIREMENTS)
    records = []
    for wheel in sorted(wheelhouse.glob("*.whl")):
        distribution = _wheel_distribution(wheel)
        version = expected_versions.get(distribution, "source-commit")
        records.append(
            {
                "distribution": distribution,
                "filename": wheel.name,
                "version": version,
                "sha256": _sha256_file(wheel),
            }
        )
    expected = set(expected_versions) | {"tracelens"}
    found = {record["distribution"] for record in records}
    if found != expected:
        raise RuntimeError(
            "TraceLens wheelhouse is incomplete or contains unexpected wheels: "
            f"expected={sorted(expected)}, found={sorted(found)}"
        )
    return records


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"Unsafe path in git archive: {member.name!r}")
            if member.issym() or member.islnk():
                raise RuntimeError(
                    f"Links are not allowed in the TraceLens build archive: "
                    f"{member.name!r}"
                )
        archive.extractall(destination)


def _stage_committed_source(
    identity: VllmTraceLensIdentity,
    tracelens_repo: Path,
    destination: Path,
) -> None:
    archive_path = destination.parent / "tracelens-source.tar"
    cmd = [
        "git",
        "-C",
        str(tracelens_repo),
        "archive",
        "--format=tar",
        "--output",
        str(archive_path),
        identity.source_commit,
        "--",
        *_SOURCE_PATHS,
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Could not stage committed TraceLens source: {_completed_output(proc)}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    _safe_extract_tar(archive_path, destination)


def _mounted(path: Path, container_path: str, read_only: bool = False) -> str:
    suffix = ":ro" if read_only else ""
    return f"{path.resolve()}:{container_path}{suffix}"


def _build_source_wheel(
    identity: VllmTraceLensIdentity,
    source_dir: Path,
    wheelhouse: Path,
) -> list[str]:
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        "--entrypoint",
        "python3",
        "-v",
        _mounted(source_dir, "/src"),
        "-v",
        _mounted(wheelhouse, "/wheels"),
        identity.base_image_id,
        "-m",
        "pip",
        "wheel",
        "--disable-pip-version-check",
        "--no-deps",
        "--no-build-isolation",
        "--wheel-dir",
        "/wheels",
        "/src",
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Could not build the pinned TraceLens wheel in the base runtime: "
            f"{(proc.stdout or '')[-4000:]}"
        )
    return cmd


def _download_requirement_wheels(
    identity: VllmTraceLensIdentity,
    wheelhouse: Path,
) -> list[str]:
    requirements = [
        f"{name}=={version}" for name, version in VLLM_TRACELENS_REQUIREMENTS
    ]
    cmd = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "python3",
        "-v",
        _mounted(wheelhouse, "/wheels"),
        identity.base_image_id,
        "-m",
        "pip",
        "download",
        "--disable-pip-version-check",
        "--dest",
        "/wheels",
        "--only-binary=:all:",
        "--no-deps",
        *requirements,
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Could not download pinned TraceLens diagnostic wheels: "
            f"{(proc.stdout or '')[-4000:]}"
        )
    return cmd


def _dockerfile_labels(labels: Mapping[str, str]) -> str:
    return "\n".join(
        f"LABEL {key}={json.dumps(value)}" for key, value in sorted(labels.items())
    )


def _verification_script() -> str:
    requirements = _canonical_json(dict(VLLM_TRACELENS_REQUIREMENTS))
    forbidden = _canonical_json(list(VLLM_TRACELENS_FORBIDDEN))
    return f"""\
import importlib.metadata as metadata
import json
import pathlib
import py_compile

expected = json.loads({requirements!r})
forbidden = json.loads({forbidden!r})
actual = {{name: metadata.version(name) for name in expected}}
assert actual == expected, (actual, expected)
for name in forbidden:
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        continue
    raise AssertionError(f"forbidden package installed: {{name}}")

import TraceLens
from TraceLens.Agent.Analysis.utils.arch_utils import list_platforms
from TraceLens.Reporting import generate_perf_report_pytorch_inference
from TraceLens.TraceUtils import split_inference_trace_annotation
import vllm
from vllm.config.profiler import ProfilerConfig

fields = ProfilerConfig.__dataclass_fields__
assert "capture_torch_profiler_dir" in fields
assert "detailed_trace_annotation" in fields
root = pathlib.Path(vllm.__file__).resolve().parent
files = [
    root / "config/profiler.py",
    root / "v1/worker/gpu_model_runner.py",
    root / "v1/worker/gpu_worker.py",
]
for path in files:
    py_compile.compile(str(path), doraise=True)
runner = files[1].read_text(encoding="utf-8")
worker = files[2].read_text(encoding="utf-8")
assert "capture_torch_profiler_dir" in runner and "profiler=profiler" in runner
assert "detailed_trace_annotation" in worker and "c_sqsk" in worker
print(json.dumps({{
    "grpcio_version": metadata.version("grpcio"),
    "platforms": list_platforms(),
    "tracelens_version": metadata.version("TraceLens"),
    "vllm_version": metadata.version("vllm"),
}}, sort_keys=True))
"""


def _write_build_context(
    context: Path,
    identity: VllmTraceLensIdentity,
    wheel_manifest: Sequence[Mapping[str, str]],
    *,
    build_base_locator: str,
) -> Dict[str, str]:
    wheel_manifest_json = _canonical_json(list(wheel_manifest))
    labels = identity.labels()
    labels.update(
        {
            LABEL_WHEEL_MANIFEST: wheel_manifest_json,
            LABEL_WHEEL_MANIFEST_SHA256: _sha256_bytes(
                wheel_manifest_json.encode("utf-8")
            ),
        }
    )
    identity_document = {
        "schema": VLLM_TRACELENS_SCHEMA,
        **identity.metadata(),
        "wheels": list(wheel_manifest),
    }
    (context / "identity.json").write_text(
        json.dumps(identity_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (context / "verify.py").write_text(_verification_script(), encoding="utf-8")
    (context / "patch.diff").write_bytes(identity.patch_bytes)
    dockerfile = f"""\
FROM {build_base_locator}
{_dockerfile_labels(labels)}
COPY wheels/ /tmp/tracelens-wheels/
COPY patch.diff /tmp/tracelens-vllm.patch
COPY verify.py /tmp/verify-tracelens-runtime.py
COPY identity.json /opt/magpie/tracelens-runtime-identity.json
RUN python3 -m pip install --disable-pip-version-check --no-index --no-deps \\
      /tmp/tracelens-wheels/*.whl \\
    && VLLM_PACKAGE="$(python3 -c 'import os,vllm; print(os.path.dirname(vllm.__file__))')" \\
    && cd "$(dirname "$VLLM_PACKAGE")" \\
    && git apply --check /tmp/tracelens-vllm.patch \\
    && git apply /tmp/tracelens-vllm.patch \\
    && python3 /tmp/verify-tracelens-runtime.py \\
    && rm -rf /tmp/tracelens-wheels /tmp/tracelens-vllm.patch /tmp/verify-tracelens-runtime.py
WORKDIR /workspace
"""
    (context / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    return labels


def build_vllm_tracelens_image(
    *,
    identity: VllmTraceLensIdentity,
    tracelens_repo: Path,
    derived_image: str,
) -> Dict[str, Any]:
    """Build and verify a minimal TraceLens image from immutable inputs."""
    with tempfile.TemporaryDirectory(prefix="magpie-tracelens-vllm-") as temp_dir:
        context = Path(temp_dir)
        source_dir = context / "source"
        wheelhouse = context / "wheels"
        wheelhouse.mkdir()
        _stage_committed_source(identity, tracelens_repo, source_dir)
        source_wheel_command = _build_source_wheel(identity, source_dir, wheelhouse)
        download_command = _download_requirement_wheels(identity, wheelhouse)
        wheel_manifest = _wheel_manifest(wheelhouse)
        shutil.rmtree(source_dir)
        base_reference = _acquire_build_base_reference(identity)
        try:
            labels = _write_build_context(
                context,
                identity,
                wheel_manifest,
                build_base_locator=base_reference.locator,
            )
            archive = context / "tracelens-source.tar"
            if archive.exists():
                archive.unlink()
            _verify_build_base_reference(base_reference)
            cmd = [
                "docker",
                "build",
                "--network",
                "none",
                "--pull=false",
                "--no-cache",
                "--provenance=false",
                "-t",
                derived_image,
                str(context),
            ]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "TraceLens vLLM runtime image build failed with exit code "
                    f"{proc.returncode}. Image: {derived_image}\n"
                    f"{(proc.stdout or '')[-4000:]}"
                )
            _verify_build_base_reference(base_reference)
        finally:
            _release_build_base_reference(base_reference)

    validation = validate_vllm_tracelens_image(derived_image, identity)
    if not validation["valid"]:
        raise RuntimeError(
            "Built TraceLens vLLM image failed identity validation: "
            f"{validation['reason']}"
        )
    record = docker_image_record(derived_image) or {}
    return {
        "command": cmd[:-1] + ["<temporary-build-context>"],
        "source_wheel_command": source_wheel_command[:-1] + ["<staged-source>"],
        "requirements_download_command": download_command,
        "base_binding": {
            "image_id": base_reference.image_id,
            "provenance_locator": identity.base_image_locator,
            "build_reference_kind": base_reference.kind,
            "temporary_tag_removed": base_reference.owns_temporary_tag,
            "retained_local_reference": base_reference.retained_locator,
            "retained_local_reference_created": (
                base_reference.retained_locator_created
            ),
        },
        "image_id": record.get("Id"),
        "image_labels": labels,
        "dependency_wheels": list(wheel_manifest),
        "dependency_wheel_manifest_sha256": labels[LABEL_WHEEL_MANIFEST_SHA256],
        "validation": validation,
    }


def _validate_wheel_labels(labels: Mapping[str, Any]) -> Optional[str]:
    manifest_text = labels.get(LABEL_WHEEL_MANIFEST)
    manifest_sha = labels.get(LABEL_WHEEL_MANIFEST_SHA256)
    if not isinstance(manifest_text, str) or not isinstance(manifest_sha, str):
        return "missing wheel-manifest labels"
    if _sha256_bytes(manifest_text.encode("utf-8")) != manifest_sha:
        return "wheel-manifest label digest mismatch"
    try:
        manifest = json.loads(manifest_text)
    except json.JSONDecodeError:
        return "wheel-manifest label is not valid JSON"
    if not isinstance(manifest, list):
        return "wheel-manifest label is not a list"
    expected = dict(VLLM_TRACELENS_REQUIREMENTS)
    found: Dict[str, str] = {}
    for item in manifest:
        if not isinstance(item, dict):
            return "wheel-manifest entry is not an object"
        distribution = item.get("distribution")
        digest = item.get("sha256")
        version = item.get("version")
        if not isinstance(distribution, str) or not isinstance(digest, str):
            return "wheel-manifest entry is incomplete"
        if not _HASH_RE.fullmatch(digest):
            return f"invalid wheel digest for {distribution}"
        if distribution != "tracelens":
            found[distribution] = str(version)
    if found != expected:
        return f"pinned wheel versions mismatch: expected={expected}, found={found}"
    if not any(item.get("distribution") == "tracelens" for item in manifest):
        return "TraceLens source wheel is absent from wheel-manifest"
    return None


def _runtime_probe(image: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "python3",
            image,
            "-c",
            _verification_script(),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
        check=False,
    )


def _patch_reverse_probe(
    image: str,
    patch_bytes: bytes,
) -> subprocess.CompletedProcess[bytes]:
    script = (
        "VLLM_PACKAGE=$(python3 -c 'import os,vllm; "
        "print(os.path.dirname(vllm.__file__))'); "
        'cd "$(dirname "$VLLM_PACKAGE")"; '
        "git apply --reverse --check -"
    )
    return subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "bash",
            "-i",
            image,
            "-lc",
            script,
        ],
        input=patch_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )


def validate_vllm_tracelens_image(
    image: str,
    identity: VllmTraceLensIdentity,
) -> Dict[str, Any]:
    """Validate labels, base-layer ancestry, packages, imports, and patch markers."""
    record = docker_image_record(image)
    if not record:
        return {"valid": False, "reason": "image is not locally inspectable"}
    labels = (record.get("Config") or {}).get("Labels") or {}
    if not isinstance(labels, dict):
        return {"valid": False, "reason": "image labels are unavailable"}
    mismatches = {
        key: {"expected": value, "actual": labels.get(key)}
        for key, value in identity.labels().items()
        if labels.get(key) != value
    }
    if mismatches:
        return {
            "valid": False,
            "reason": "identity label mismatch",
            "label_mismatches": mismatches,
        }
    wheel_error = _validate_wheel_labels(labels)
    if wheel_error:
        return {"valid": False, "reason": wheel_error}

    base_record = docker_image_record(identity.base_image_id)
    base_layers = ((base_record or {}).get("RootFS") or {}).get("Layers") or []
    image_layers = (record.get("RootFS") or {}).get("Layers") or []
    if not base_layers or image_layers[: len(base_layers)] != base_layers:
        return {"valid": False, "reason": "base image layer ancestry mismatch"}

    try:
        probe = _runtime_probe(image)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"valid": False, "reason": f"runtime probe failed: {exc}"}
    if probe.returncode != 0:
        return {
            "valid": False,
            "reason": "runtime package/import/patch probe failed",
            "probe_output": _completed_output(probe)[-2000:],
        }
    lines = (probe.stdout or "").strip().splitlines()
    try:
        probe_result = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        return {"valid": False, "reason": "runtime probe returned invalid JSON"}
    if probe_result.get("vllm_version") != identity.vllm_version:
        return {"valid": False, "reason": "vLLM version changed in derived image"}
    if probe_result.get("grpcio_version") != identity.grpcio_version:
        return {"valid": False, "reason": "grpcio version changed in derived image"}
    try:
        patch_probe = _patch_reverse_probe(image, identity.patch_bytes)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"valid": False, "reason": f"exact patch probe failed: {exc}"}
    if patch_probe.returncode != 0:
        return {
            "valid": False,
            "reason": "installed vLLM files do not match the committed patch",
            "probe_output": _completed_output(patch_probe)[-2000:],
        }
    return {
        "valid": True,
        "reason": ("identity, ancestry, packages, imports, and exact patch verified"),
        "image_id": record.get("Id"),
        "runtime_probe": probe_result,
        "dependency_wheels": json.loads(labels[LABEL_WHEEL_MANIFEST]),
        "dependency_wheel_manifest_sha256": labels[LABEL_WHEEL_MANIFEST_SHA256],
    }


__all__ = [
    "VLLM_TRACELENS_FORBIDDEN",
    "VLLM_TRACELENS_REQUIREMENTS",
    "VllmTraceLensIdentity",
    "build_vllm_tracelens_image",
    "docker_image_id",
    "docker_image_record",
    "resolve_vllm_tracelens_identity",
    "validate_vllm_tracelens_image",
]
