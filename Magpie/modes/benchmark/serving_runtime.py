###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Evidence binding a Docker serving benchmark to its immutable runtime."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

SERVING_RUNTIME_SCHEMA = "magpie.serving-runtime-receipt/v2"
SERVING_RUNTIME_RECEIPT = "serving_runtime_receipt.json"
SERVING_RUNTIME_KEYS = (
    "schema",
    "execution_mode",
    "input_config_sha256",
    "input_image",
    "input_image_id",
    "requested_image",
    "resolved_image_id",
    "image_derivation",
    "container_name",
    "docker_argv_sha256",
    "process_succeeded",
    "verified",
    "errors",
)
SERVING_IMAGE_DERIVATION_KEYS = (
    "kind",
    "framework",
    "runtime_schema",
    "base_image",
    "base_image_id",
    "base_image_locator",
    "derived_image",
    "derived_image_id",
    "tracelens_source_commit",
    "tracelens_source_tree",
    "patch_version",
    "patch_path",
    "patch_sha256",
    "dependency_wheel_manifest_sha256",
    "validator",
    "verified",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}$")
_PATCH_VERSION_RE = re.compile(r"^v[0-9]+$")
_PATCH_PATH_RE = re.compile(
    r"^examples/custom_workflows/inference_analysis/vllm_patches/"
    r"config_vllm_v0\.([0-9]+)\.0\.patch$"
)
_REPO_DIGEST_RE = re.compile(r"^[a-z0-9][a-z0-9._:/-]*@sha256:[0-9a-f]{64}$")
_TRACELENS_VLLM_RUNTIME_SCHEMA = "magpie.tracelens-vllm-runtime/v1"
_DIRECT_VALIDATOR = "docker-image-id"
_TRACELENS_VALIDATOR = "vllm-tracelens-runtime-validation/v1"
_MAX_ERRORS = 8
_MAX_ERROR_LENGTH = 240


def sha256_bytes(content: bytes) -> str:
    """Return the lowercase SHA-256 digest of exact input bytes."""

    return hashlib.sha256(content).hexdigest()


def canonical_docker_argv_sha256(argv: Sequence[str]) -> str:
    """Hash an unambiguous JSON encoding of an argv vector.

    Only the digest is persisted. In particular, environment values such as a
    Hugging Face token that may be present in the process argv are never copied
    into the receipt.
    """

    encoded = json.dumps(
        [str(item) for item in argv],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(encoded)


def resolve_docker_image_id(requested_image: str) -> Tuple[str, Tuple[str, ...]]:
    """Resolve a tag or digest reference to the local immutable Docker image ID.

    A raw ``sha256:<64 hex>`` ID is already immutable and is used directly.
    Other references are resolved with one fixed, non-shell Docker invocation.
    No Docker stderr is retained in the receipt.
    """

    requested = str(requested_image).strip()
    if _IMAGE_ID_RE.fullmatch(requested):
        return requested, ()
    if not requested:
        return "", ("requested Docker image is empty",)

    command = [
        "docker",
        "image",
        "inspect",
        "--format",
        "{{.Id}}",
        "--",
        requested,
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "", ("Docker image inspection timed out",)
    except OSError as exc:
        return "", (
            f"Docker image inspection could not start ({type(exc).__name__})",
        )

    if completed.returncode != 0:
        return "", (
            f"Docker image inspection failed with code {completed.returncode}",
        )
    lines = [line.strip() for line in (completed.stdout or "").splitlines()]
    identities = [line for line in lines if line]
    if len(identities) != 1 or not _IMAGE_ID_RE.fullmatch(identities[0]):
        return "", ("Docker image inspection returned an invalid image ID",)
    return identities[0], ()


def image_derivation_receipt(
    *,
    framework: str,
    input_image: str,
    input_image_id: str,
    requested_image: str,
    resolved_image_id: str,
    tracelens_runtime: Optional[Mapping[str, Any]] = None,
) -> Tuple[dict[str, Any], Tuple[str, ...]]:
    """Bind the frozen input image to the exact image selected for execution."""

    input_ref = str(input_image or "").strip()
    input_id = str(input_image_id or "").strip()
    runtime_ref = str(requested_image or "").strip()
    runtime_id = str(resolved_image_id or "").strip()
    framework_name = str(framework or "").strip().lower()
    if input_ref == runtime_ref:
        derivation = _direct_derivation(
            framework=framework_name,
            image=input_ref,
            image_id=input_id,
        )
    else:
        derivation = _tracelens_derivation(
            framework=framework_name,
            input_image=input_ref,
            input_image_id=input_id,
            requested_image=runtime_ref,
            resolved_image_id=runtime_id,
            runtime=tracelens_runtime,
        )
    errors = _image_derivation_errors(
        derivation,
        input_image=input_ref,
        input_image_id=input_id,
        requested_image=runtime_ref,
        resolved_image_id=runtime_id,
    )
    if errors and derivation.get("verified") is True:
        derivation = dict(derivation)
        derivation["verified"] = False
        errors = _image_derivation_errors(
            derivation,
            input_image=input_ref,
            input_image_id=input_id,
            requested_image=runtime_ref,
            resolved_image_id=runtime_id,
        )
    return derivation, errors


def _direct_derivation(
    *,
    framework: str,
    image: str,
    image_id: str,
) -> dict[str, Any]:
    return _ordered_derivation(
        {
            "kind": "direct",
            "framework": framework,
            "runtime_schema": None,
            "base_image": image,
            "base_image_id": image_id,
            "base_image_locator": image,
            "derived_image": image,
            "derived_image_id": image_id,
            "tracelens_source_commit": None,
            "tracelens_source_tree": None,
            "patch_version": None,
            "patch_path": None,
            "patch_sha256": None,
            "dependency_wheel_manifest_sha256": None,
            "validator": _DIRECT_VALIDATOR,
            "verified": True,
        }
    )


def _tracelens_derivation(
    *,
    framework: str,
    input_image: str,
    input_image_id: str,
    requested_image: str,
    resolved_image_id: str,
    runtime: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    data = runtime if isinstance(runtime, Mapping) else {}
    validation = data.get("public_runtime_validation")
    validation = validation if isinstance(validation, Mapping) else {}
    metadata = {
        "kind": "tracelens-derived",
        "framework": framework,
        "runtime_schema": _string(data.get("runtime_schema")),
        "base_image": input_image,
        "base_image_id": input_image_id,
        "base_image_locator": _string(data.get("base_image_locator")),
        "derived_image": requested_image,
        "derived_image_id": resolved_image_id,
        "tracelens_source_commit": _string(data.get("tracelens_source_commit")),
        "tracelens_source_tree": _string(data.get("tracelens_source_tree")),
        "patch_version": _string(data.get("patch_version")),
        "patch_path": _string(data.get("tracelens_patch_path")),
        "patch_sha256": _string(data.get("tracelens_patch_sha256")),
        "dependency_wheel_manifest_sha256": _string(
            data.get("dependency_wheel_manifest_sha256")
        ),
        "validator": _TRACELENS_VALIDATOR,
        "verified": False,
    }
    runtime_matches = (
        data.get("enabled") is True
        and data.get("framework") == framework
        and data.get("base_image") == input_image
        and data.get("base_image_id") == input_image_id
        and data.get("image") == requested_image
        and data.get("public_runtime_image") == requested_image
        and data.get("public_runtime_image_id") == resolved_image_id
        and validation.get("valid") is True
        and validation.get("image_id") == resolved_image_id
    )
    metadata["verified"] = bool(runtime_matches)
    return _ordered_derivation(metadata)


def _ordered_derivation(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.get(key) for key in SERVING_IMAGE_DERIVATION_KEYS}


def _image_derivation_errors(
    value: object,
    *,
    input_image: str,
    input_image_id: str,
    requested_image: str,
    resolved_image_id: str,
) -> Tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ("serving image derivation is missing",)
    errors = []
    if tuple(value.keys()) != SERVING_IMAGE_DERIVATION_KEYS:
        errors.append("serving image derivation has an invalid shape")
    if not _IMAGE_ID_RE.fullmatch(input_image_id):
        errors.append("input Docker image ID is missing or invalid")
    if not _IMAGE_ID_RE.fullmatch(resolved_image_id):
        errors.append("resolved Docker image ID is missing or invalid")
    if not input_image or not requested_image:
        errors.append("serving image references are missing")
    if value.get("framework") not in {"vllm", "sglang", "atom"}:
        errors.append("serving image derivation framework is invalid")
    if (
        value.get("base_image") != input_image
        or value.get("base_image_id") != input_image_id
        or value.get("derived_image") != requested_image
        or value.get("derived_image_id") != resolved_image_id
    ):
        errors.append("serving image derivation does not match its receipt")

    if value.get("kind") == "direct":
        errors.extend(
            _direct_derivation_errors(
                value,
                input_image=input_image,
                input_image_id=input_image_id,
                requested_image=requested_image,
                resolved_image_id=resolved_image_id,
            )
        )
    elif value.get("kind") == "tracelens-derived":
        errors.extend(_tracelens_derivation_errors(value))
    else:
        errors.append("serving image derivation kind is invalid")
    if value.get("verified") is not True:
        errors.append("serving image derivation is not verified")
    return tuple(_bounded_errors(errors))


def _direct_derivation_errors(
    value: Mapping[str, Any],
    *,
    input_image: str,
    input_image_id: str,
    requested_image: str,
    resolved_image_id: str,
) -> list[str]:
    errors = []
    nullable = SERVING_IMAGE_DERIVATION_KEYS[2:3] + SERVING_IMAGE_DERIVATION_KEYS[8:14]
    if any(value.get(key) is not None for key in nullable):
        errors.append("direct image derivation carries TraceLens identity")
    if (
        input_image != requested_image
        or input_image_id != resolved_image_id
        or value.get("base_image_locator") != input_image
        or value.get("validator") != _DIRECT_VALIDATOR
    ):
        errors.append("direct image derivation changed the configured image")
    return errors


def _tracelens_derivation_errors(value: Mapping[str, Any]) -> list[str]:
    errors = []
    patch_path = value.get("patch_path")
    patch_match = None
    if isinstance(patch_path, str) and patch_path:
        parsed = PurePosixPath(patch_path)
        patch_match = (
            _PATCH_PATH_RE.fullmatch(patch_path)
            if not parsed.is_absolute() and ".." not in parsed.parts
            else None
        )
    if value.get("framework") != "vllm":
        errors.append("verified TraceLens derivation currently requires vLLM")
    if value.get("runtime_schema") != _TRACELENS_VLLM_RUNTIME_SCHEMA:
        errors.append("TraceLens runtime schema is invalid")
    base_locator = value.get("base_image_locator")
    base_id = value.get("base_image_id")
    locator_valid = bool(
        isinstance(base_locator, str)
        and (
            (_IMAGE_ID_RE.fullmatch(base_locator) and base_locator == base_id)
            or _REPO_DIGEST_RE.fullmatch(base_locator)
        )
    )
    if not locator_valid:
        errors.append("TraceLens base image locator is missing")
    if not _GIT_OBJECT_RE.fullmatch(str(value.get("tracelens_source_commit") or "")):
        errors.append("TraceLens source commit is invalid")
    if not _GIT_OBJECT_RE.fullmatch(str(value.get("tracelens_source_tree") or "")):
        errors.append("TraceLens source tree is invalid")
    patch_version = str(value.get("patch_version") or "")
    if not _PATCH_VERSION_RE.fullmatch(patch_version):
        errors.append("TraceLens patch version is invalid")
    if patch_match is None or patch_version != f"v{int(patch_match.group(1))}":
        errors.append("TraceLens patch path is invalid")
    if not _SHA256_RE.fullmatch(str(value.get("patch_sha256") or "")):
        errors.append("TraceLens patch SHA-256 is invalid")
    if not _SHA256_RE.fullmatch(
        str(value.get("dependency_wheel_manifest_sha256") or "")
    ):
        errors.append("TraceLens wheel manifest SHA-256 is invalid")
    if value.get("validator") != _TRACELENS_VALIDATOR:
        errors.append("TraceLens image validator is invalid")
    return errors


def pending_serving_runtime_receipt(
    *,
    execution_mode: str,
    input_config_sha256: str,
    framework: str,
    input_image: str,
    input_image_id: str,
    requested_image: str,
    resolved_image_id: str,
    container_name: str,
    docker_argv: Sequence[str],
    tracelens_runtime: Optional[Mapping[str, Any]] = None,
    prior_errors: Iterable[str] = (),
) -> dict[str, Any]:
    """Build the pre-execution receipt and validate its command bindings."""

    errors = list(prior_errors)
    config_digest = str(input_config_sha256 or "")
    input_id = str(input_image_id or "")
    image_id = str(resolved_image_id or "")
    argv_digest = canonical_docker_argv_sha256(docker_argv) if docker_argv else ""
    derivation, derivation_errors = image_derivation_receipt(
        framework=framework,
        input_image=input_image,
        input_image_id=input_id,
        requested_image=requested_image,
        resolved_image_id=image_id,
        tracelens_runtime=tracelens_runtime,
    )

    if not _SHA256_RE.fullmatch(config_digest):
        errors.append("input config SHA-256 is missing or invalid")
    errors.extend(derivation_errors)
    errors.extend(
        docker_command_binding_errors(
            docker_argv,
            expected_container_name=container_name,
            expected_image_id=image_id,
        )
    )
    return _receipt(
        execution_mode=execution_mode,
        input_config_sha256=config_digest,
        input_image=input_image,
        input_image_id=input_id,
        requested_image=requested_image,
        resolved_image_id=image_id,
        image_derivation=derivation,
        container_name=container_name,
        docker_argv_sha256=argv_digest,
        process_succeeded=False,
        verified=False,
        errors=errors,
    )


def unresolved_serving_runtime_receipt(
    *,
    input_config_sha256: str,
    framework: str,
    input_image: str,
    input_image_id: str,
    requested_image: str,
    resolved_image_id: str,
    container_name: str,
    tracelens_runtime: Optional[Mapping[str, Any]] = None,
    errors: Iterable[str],
) -> dict[str, Any]:
    """Build a receipt when an immutable image could not be resolved."""

    combined = list(errors)
    if not _SHA256_RE.fullmatch(str(input_config_sha256 or "")):
        combined.append("input config SHA-256 is missing or invalid")
    derivation, derivation_errors = image_derivation_receipt(
        framework=framework,
        input_image=input_image,
        input_image_id=input_image_id,
        requested_image=requested_image,
        resolved_image_id=resolved_image_id,
        tracelens_runtime=tracelens_runtime,
    )
    combined.extend(derivation_errors)
    return _receipt(
        execution_mode="docker",
        input_config_sha256=str(input_config_sha256 or ""),
        input_image=input_image,
        input_image_id=input_image_id,
        requested_image=requested_image,
        resolved_image_id=resolved_image_id,
        image_derivation=derivation,
        container_name=container_name,
        docker_argv_sha256="",
        process_succeeded=False,
        verified=False,
        errors=combined,
    )


def validate_prepared_command(
    receipt: Mapping[str, Any],
    docker_argv: Sequence[str],
) -> Tuple[str, ...]:
    """Reject a command that differs from the prepared runtime receipt."""

    errors = []
    if tuple(receipt.keys()) != SERVING_RUNTIME_KEYS:
        errors.append("serving runtime receipt has an invalid shape")
    if receipt.get("schema") != SERVING_RUNTIME_SCHEMA:
        errors.append("serving runtime receipt schema is invalid")
    if receipt.get("execution_mode") != "docker":
        errors.append("serving runtime execution mode is invalid")
    expected_digest = receipt.get("docker_argv_sha256")
    if expected_digest != canonical_docker_argv_sha256(docker_argv):
        errors.append("Docker argv does not match its prepared digest")
    errors.extend(
        docker_command_binding_errors(
            docker_argv,
            expected_container_name=str(receipt.get("container_name", "")),
            expected_image_id=str(receipt.get("resolved_image_id", "")),
        )
    )
    errors.extend(
        _image_derivation_errors(
            receipt.get("image_derivation"),
            input_image=str(receipt.get("input_image", "")),
            input_image_id=str(receipt.get("input_image_id", "")),
            requested_image=str(receipt.get("requested_image", "")),
            resolved_image_id=str(receipt.get("resolved_image_id", "")),
        )
    )
    errors.extend(str(item) for item in receipt.get("errors", []))
    return tuple(_bounded_errors(errors))


def finalize_serving_runtime_receipt(
    receipt: Mapping[str, Any],
    *,
    process_succeeded: bool,
    process_error: str = "",
) -> dict[str, Any]:
    """Finalize process status while preserving the prepared identity fields."""

    errors = list(receipt.get("errors", []))
    if process_error:
        errors.append(process_error)
    errors.extend(
        _image_derivation_errors(
            receipt.get("image_derivation"),
            input_image=str(receipt.get("input_image", "")),
            input_image_id=str(receipt.get("input_image_id", "")),
            requested_image=str(receipt.get("requested_image", "")),
            resolved_image_id=str(receipt.get("resolved_image_id", "")),
        )
    )
    bounded = _bounded_errors(errors)
    succeeded = bool(process_succeeded)
    return _receipt(
        execution_mode=str(receipt.get("execution_mode", "docker")),
        input_config_sha256=str(receipt.get("input_config_sha256", "")),
        input_image=str(receipt.get("input_image", "")),
        input_image_id=str(receipt.get("input_image_id", "")),
        requested_image=str(receipt.get("requested_image", "")),
        resolved_image_id=str(receipt.get("resolved_image_id", "")),
        image_derivation=receipt.get("image_derivation"),
        container_name=str(receipt.get("container_name", "")),
        docker_argv_sha256=str(receipt.get("docker_argv_sha256", "")),
        process_succeeded=succeeded,
        verified=succeeded and not bounded,
        errors=bounded,
    )


def write_serving_runtime_receipt(
    workspace: Path,
    receipt: Mapping[str, Any],
) -> None:
    """Atomically persist the bounded receipt beside the benchmark report."""

    destination = Path(workspace) / SERVING_RUNTIME_RECEIPT
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(dict(receipt), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def docker_command_binding_errors(
    docker_argv: Sequence[str],
    *,
    expected_container_name: str,
    expected_image_id: str,
) -> Tuple[str, ...]:
    """Validate the exact container-name and image slots in ``docker run``."""

    argv = [str(item) for item in docker_argv]
    errors = []
    if argv[:2] != ["docker", "run"]:
        errors.append("Docker argv is not a docker run command")

    name_positions = [index for index, item in enumerate(argv) if item == "--name"]
    if len(name_positions) != 1 or name_positions[0] + 1 >= len(argv):
        errors.append("Docker argv does not contain one container name")
    elif argv[name_positions[0] + 1] != expected_container_name:
        errors.append("Docker argv container name does not match the receipt")

    entry_positions = [
        index for index, item in enumerate(argv) if item == "--entrypoint"
    ]
    if len(entry_positions) != 1 or entry_positions[0] + 2 >= len(argv):
        errors.append("Docker argv does not contain one image slot")
    elif argv[entry_positions[0] + 2] != expected_image_id:
        errors.append("Docker argv image does not match the resolved image ID")
    return tuple(_bounded_errors(errors))


def _receipt(
    *,
    execution_mode: str,
    input_config_sha256: str,
    input_image: str,
    input_image_id: str,
    requested_image: str,
    resolved_image_id: str,
    image_derivation: object,
    container_name: str,
    docker_argv_sha256: str,
    process_succeeded: bool,
    verified: bool,
    errors: Iterable[str],
) -> dict[str, Any]:
    return {
        "schema": SERVING_RUNTIME_SCHEMA,
        "execution_mode": execution_mode,
        "input_config_sha256": input_config_sha256,
        "input_image": input_image,
        "input_image_id": input_image_id,
        "requested_image": requested_image,
        "resolved_image_id": resolved_image_id,
        "image_derivation": _ordered_derivation(
            image_derivation if isinstance(image_derivation, Mapping) else {}
        ),
        "container_name": container_name,
        "docker_argv_sha256": docker_argv_sha256,
        "process_succeeded": process_succeeded,
        "verified": verified,
        "errors": _bounded_errors(errors),
    }


def _bounded_errors(errors: Iterable[str]) -> list[str]:
    bounded = []
    for error in errors:
        text = " ".join(str(error).split())[:_MAX_ERROR_LENGTH]
        if text and text not in bounded:
            bounded.append(text)
        if len(bounded) == _MAX_ERRORS:
            break
    return bounded


def _string(value: object) -> Optional[str]:
    return value if isinstance(value, str) and value else None
