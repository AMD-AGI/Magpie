###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
TraceLens-ready benchmark runtime image preparation.

TraceLens inference mode needs framework runtime patches for some vLLM/SGLang
versions. This module uses the public TraceLens workflow build scripts to derive
patched Docker images from supported official runtime images.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from .config import BenchmarkConfig

logger = logging.getLogger(__name__)


TRACELENS_INFERENCE_WORKFLOW = Path("examples/custom_workflows/inference_analysis")
SUPPORTED_SGLANG_PATCH_VERSIONS = ("0.5.9", "0.5.11", "0.5.12")
SUPPORTED_VLLM_PATCH_VERSIONS = tuple(range(14, 22))


def is_tracelens_ready_runtime_image(framework: str, image_name: Optional[str]) -> bool:
    """Return whether the Docker image name already looks TraceLens-ready."""
    if not image_name:
        return False

    lowered = image_name.lower()
    return "tracelens" in lowered and framework.lower() in lowered


def infer_sglang_patch_version(image_name: str) -> Optional[str]:
    """Infer supported TraceLens SGLang patch version from an image tag."""
    lowered = image_name.lower()
    for version in SUPPORTED_SGLANG_PATCH_VERSIONS:
        if version in lowered:
            return version
    return None


def infer_vllm_patch_version(image_name: str) -> Optional[str]:
    """Infer TraceLens vLLM patch shorthand such as v19 from an image tag."""
    lowered = image_name.lower()
    match = re.search(r"(?:^|[^0-9])v?0\.(1[4-9]|2[0-1])(?:\.\d+)?", lowered)
    if not match:
        return None

    minor = int(match.group(1))
    if minor not in SUPPORTED_VLLM_PATCH_VERSIONS:
        return None
    return f"v{minor}"


def runner_type_to_gpu_type(runner_type: str) -> str:
    """Map Magpie/InferenceX runner type to TraceLens build script gpu type."""
    normalized = (runner_type or "").lower()
    if normalized.startswith("mi300"):
        return "mi300"
    if normalized.startswith("mi350"):
        return "mi350"
    if normalized.startswith("mi355"):
        return "mi355"
    raise ValueError(
        "TraceLens runtime auto patch currently supports AMD MI300/MI350/MI355 "
        f"runners, got runner_type={runner_type!r}"
    )


def resolve_tracelens_repo_path(configured_path: Optional[str] = None) -> Path:
    """Find a public TraceLens source checkout containing inference workflows."""
    candidates = []
    if configured_path:
        candidates.append(Path(configured_path).expanduser())

    for env_key in ("TRACELENS_REPO_PATH", "TRACELENS_PATH"):
        env_value = os.environ.get(env_key)
        if env_value:
            candidates.append(Path(env_value).expanduser())

    magpie_repo = Path(__file__).resolve().parents[3]
    candidates.extend(
        [
            magpie_repo / "TraceLens",
            magpie_repo.parent / "TraceLens",
            Path.cwd() / "TraceLens",
            Path.home() / "TraceLens",
        ]
    )

    for candidate in candidates:
        workflow_dir = candidate / TRACELENS_INFERENCE_WORKFLOW
        if workflow_dir.is_dir():
            return candidate.resolve()

    searched = ", ".join(str(path) for path in candidates)
    raise RuntimeError(
        "TraceLens auto_patch_runtime requires a public TraceLens source checkout "
        "with examples/custom_workflows/inference_analysis. Set "
        "profiler.tracelens.tracelens_repo_path or TRACELENS_REPO_PATH. "
        f"Searched: {searched}"
    )


def derive_tracelens_image_tag(
    framework: str,
    base_image: str,
    runner_type: str,
    patch_version: str,
) -> str:
    """Create a deterministic local Docker tag for a derived runtime image."""
    digest = hashlib.sha256(base_image.encode("utf-8")).hexdigest()[:10]
    safe_base = re.sub(r"[^a-z0-9_.-]+", "-", base_image.lower()).strip(".-")
    safe_base = safe_base[:72].strip(".-") or framework
    safe_runner = re.sub(r"[^a-z0-9_.-]+", "-", runner_type.lower()).strip(".-")
    safe_version = patch_version.replace(".", "_")
    return (
        f"magpie-tracelens-{framework}:"
        f"{safe_version}-{safe_runner}-{digest}-{safe_base}"
    )


def docker_image_exists(image_tag: str) -> bool:
    """Return True when Docker can inspect the image tag locally."""
    proc = subprocess.run(
        ["docker", "image", "inspect", image_tag],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return proc.returncode == 0


def _build_command(
    config: BenchmarkConfig,
    base_image: str,
    runner_type: str,
    derived_image: str,
    tracelens_repo: Path,
) -> list[str]:
    workflow_dir = tracelens_repo / TRACELENS_INFERENCE_WORKFLOW
    if config.framework == "sglang":
        patch_version = infer_sglang_patch_version(base_image)
        if patch_version is None:
            raise RuntimeError(
                "TraceLens auto_patch_runtime cannot infer a supported SGLang "
                f"version from image {base_image!r}. Supported versions: "
                f"{', '.join(SUPPORTED_SGLANG_PATCH_VERSIONS)}. Set docker_image "
                "to a supported official SGLang tag, use a TraceLens-ready image, "
                "or set auto_patch_runtime=false."
            )
        return [
            "bash",
            str(workflow_dir / "build_docker_sglang.sh"),
            str(tracelens_repo),
            "--sglang-version",
            patch_version,
            "--gpu-type",
            runner_type_to_gpu_type(runner_type),
            "--base-image",
            base_image,
            "-t",
            derived_image,
        ]

    if config.framework == "vllm":
        patch_version = infer_vllm_patch_version(base_image)
        if patch_version is None:
            raise RuntimeError(
                "TraceLens auto_patch_runtime cannot infer a supported vLLM "
                f"version from image {base_image!r}. Supported versions: "
                "v0.14.x through v0.21.x. Set docker_image to a supported "
                "official vLLM tag, use a TraceLens-ready image, or set "
                "auto_patch_runtime=false."
            )
        return [
            "bash",
            str(workflow_dir / "build_docker_vllm.sh"),
            patch_version,
            str(tracelens_repo),
            "--base-image",
            base_image,
            "-t",
            derived_image,
        ]

    raise RuntimeError(f"Unsupported TraceLens runtime framework: {config.framework}")


def prepare_tracelens_runtime_image(
    config: BenchmarkConfig,
    base_image: str,
    runner_type: str,
) -> Dict[str, Any]:
    """
    Ensure the Docker runtime image is TraceLens-ready when requested.

    Returns a serializable metadata dict. The caller should use the returned
    ``image`` value for the benchmark Docker run.
    """
    tl_config = config.profiler.tracelens
    result: Dict[str, Any] = {
        "enabled": bool(tl_config.enabled and tl_config.is_inference_mode),
        "auto_patch_runtime": tl_config.auto_patch_runtime,
        "framework": config.framework,
        "base_image": base_image,
        "image": base_image,
        "built": False,
        "skipped": True,
    }

    if not result["enabled"]:
        result["reason"] = "TraceLens inference mode is not enabled"
        return result

    if config.is_local or config.is_ray:
        result["reason"] = f"run_mode={config.run_mode} does not use Docker image patching"
        return result

    if is_tracelens_ready_runtime_image(config.framework, base_image):
        result["reason"] = "image already appears TraceLens-ready"
        return result

    if not tl_config.auto_patch_runtime:
        result["reason"] = "auto_patch_runtime=false"
        return result

    tracelens_repo = resolve_tracelens_repo_path(tl_config.tracelens_repo_path)

    patch_version = (
        infer_sglang_patch_version(base_image)
        if config.framework == "sglang"
        else infer_vllm_patch_version(base_image)
    )
    if patch_version is None:
        # Let _build_command raise the framework-specific error text.
        patch_version = "unknown"

    derived_image = tl_config.runtime_patch_image_tag or derive_tracelens_image_tag(
        framework=config.framework,
        base_image=base_image,
        runner_type=runner_type,
        patch_version=patch_version,
    )

    result.update(
        {
            "image": derived_image,
            "tracelens_repo_path": str(tracelens_repo),
            "patch_version": patch_version,
        }
    )

    if docker_image_exists(derived_image) and not tl_config.runtime_patch_force_rebuild:
        result["reason"] = "derived image already exists"
        return result

    cmd = _build_command(
        config=config,
        base_image=base_image,
        runner_type=runner_type,
        derived_image=derived_image,
        tracelens_repo=tracelens_repo,
    )
    result["command"] = cmd
    result["skipped"] = False

    logger.info(
        "Building TraceLens-ready %s image from %s as %s",
        config.framework,
        base_image,
        derived_image,
    )
    proc = subprocess.run(
        cmd,
        cwd=str(tracelens_repo),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        tail = (proc.stdout or "")[-4000:]
        raise RuntimeError(
            "TraceLens runtime image build failed with exit code "
            f"{proc.returncode}. Command: {' '.join(cmd)}\n{tail}"
        )

    result["built"] = True
    result["reason"] = "derived image built"
    return result


__all__ = [
    "derive_tracelens_image_tag",
    "docker_image_exists",
    "infer_sglang_patch_version",
    "infer_vllm_patch_version",
    "is_tracelens_ready_runtime_image",
    "prepare_tracelens_runtime_image",
    "resolve_tracelens_repo_path",
    "runner_type_to_gpu_type",
]
