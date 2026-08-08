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
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Tuple

SERVING_RUNTIME_SCHEMA = "magpie.serving-runtime-receipt/v1"
SERVING_RUNTIME_RECEIPT = "serving_runtime_receipt.json"
SERVING_RUNTIME_KEYS = (
    "schema",
    "execution_mode",
    "input_config_sha256",
    "requested_image",
    "resolved_image_id",
    "container_name",
    "docker_argv_sha256",
    "process_succeeded",
    "verified",
    "errors",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
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


def pending_serving_runtime_receipt(
    *,
    execution_mode: str,
    input_config_sha256: str,
    requested_image: str,
    resolved_image_id: str,
    container_name: str,
    docker_argv: Sequence[str],
    prior_errors: Iterable[str] = (),
) -> dict[str, Any]:
    """Build the pre-execution receipt and validate its command bindings."""

    errors = list(prior_errors)
    config_digest = str(input_config_sha256 or "")
    image_id = str(resolved_image_id or "")
    argv_digest = canonical_docker_argv_sha256(docker_argv) if docker_argv else ""

    if not _SHA256_RE.fullmatch(config_digest):
        errors.append("input config SHA-256 is missing or invalid")
    if not _IMAGE_ID_RE.fullmatch(image_id):
        errors.append("resolved Docker image ID is missing or invalid")
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
        requested_image=requested_image,
        resolved_image_id=image_id,
        container_name=container_name,
        docker_argv_sha256=argv_digest,
        process_succeeded=False,
        verified=False,
        errors=errors,
    )


def unresolved_serving_runtime_receipt(
    *,
    input_config_sha256: str,
    requested_image: str,
    container_name: str,
    errors: Iterable[str],
) -> dict[str, Any]:
    """Build a receipt when an immutable image could not be resolved."""

    combined = list(errors)
    if not _SHA256_RE.fullmatch(str(input_config_sha256 or "")):
        combined.append("input config SHA-256 is missing or invalid")
    return _receipt(
        execution_mode="docker",
        input_config_sha256=str(input_config_sha256 or ""),
        requested_image=requested_image,
        resolved_image_id="",
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
    bounded = _bounded_errors(errors)
    succeeded = bool(process_succeeded)
    return _receipt(
        execution_mode=str(receipt.get("execution_mode", "docker")),
        input_config_sha256=str(receipt.get("input_config_sha256", "")),
        requested_image=str(receipt.get("requested_image", "")),
        resolved_image_id=str(receipt.get("resolved_image_id", "")),
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
    requested_image: str,
    resolved_image_id: str,
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
        "requested_image": requested_image,
        "resolved_image_id": resolved_image_id,
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
