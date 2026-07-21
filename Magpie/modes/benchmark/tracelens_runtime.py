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
SGLANG_PATCH_ROOT = TRACELENS_INFERENCE_WORKFLOW / "sglang_roofline_patches"
VLLM_PATCH_DIR = TRACELENS_INFERENCE_WORKFLOW / "vllm_patches"
FRAMEWORK_PACKAGE_NAMES = {
    "sglang": "sglang",
    "vllm": "vllm",
}

PACKAGE_VERSION_SCRIPT = r"""
import importlib.metadata as metadata
import sys

print(metadata.version(sys.argv[1]))
"""


def is_tracelens_ready_runtime_image(framework: str, image_name: Optional[str]) -> bool:
    """Return whether the Docker image name already looks TraceLens-ready."""
    if not image_name:
        return False

    lowered = image_name.lower()
    return "tracelens" in lowered and framework.lower() in lowered


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version))


def _sort_semver_versions(versions: set[str]) -> list[str]:
    return sorted(versions, key=_version_key)


def available_tracelens_sglang_patch_versions(tracelens_repo: Path) -> list[str]:
    """Return SGLang versions supported by the selected TraceLens checkout."""
    workflow_dir = tracelens_repo / TRACELENS_INFERENCE_WORKFLOW
    patch_root = tracelens_repo / SGLANG_PATCH_ROOT

    patch_versions = {
        match.group(1).replace("_", ".")
        for patch_dir in patch_root.glob("sglang_*")
        if patch_dir.is_dir()
        and (match := re.match(r"sglang_((?:\d+_)+\d+)$", patch_dir.name))
    }

    build_script = workflow_dir / "build_docker_sglang.sh"
    if not build_script.is_file():
        return []

    script_versions = {
        match.group(1)
        for match in re.finditer(
            r"^\s*(\d+(?:\.\d+)+)(?:[)|])",
            build_script.read_text(encoding="utf-8", errors="replace"),
            re.MULTILINE,
        )
    }

    return _sort_semver_versions(patch_versions & script_versions)


def _parse_sglang_patch_version(version_text: str) -> Optional[str]:
    match = re.search(r"(?:^|[^0-9])v?(\d+\.\d+\.\d+)(?:[^0-9]|$)", version_text)
    if not match:
        return None
    return match.group(1)


def infer_sglang_patch_version(
    image_name: str,
    tracelens_repo: Optional[Path] = None,
    installed_version: Optional[str] = None,
) -> Optional[str]:
    """Infer TraceLens SGLang patch version from package version or image tag."""
    patch_version = (
        _parse_sglang_patch_version(installed_version)
        if installed_version
        else None
    )
    if patch_version is None:
        patch_version = _parse_sglang_patch_version(image_name.lower())
    if patch_version is None:
        return None

    if (
        tracelens_repo is not None
        and patch_version
        not in available_tracelens_sglang_patch_versions(tracelens_repo)
    ):
        return None
    return patch_version


def _sort_vllm_patch_versions(versions: set[str]) -> list[str]:
    return sorted(versions, key=lambda version: int(version.removeprefix("v")))


def available_tracelens_vllm_patch_versions(tracelens_repo: Path) -> list[str]:
    """Return vLLM versions supported by the selected TraceLens checkout."""
    workflow_dir = tracelens_repo / TRACELENS_INFERENCE_WORKFLOW
    patch_dir = tracelens_repo / VLLM_PATCH_DIR

    patch_versions = {
        f"v{match.group(1)}"
        for patch_file in patch_dir.glob("config_vllm_v0.*.0.patch")
        if (match := re.match(r"config_vllm_v0\.(\d+)\.0\.patch$", patch_file.name))
    }

    build_script = workflow_dir / "build_docker_vllm.sh"
    if not build_script.is_file():
        return []

    script_versions = {
        f"v{match.group(1)}"
        for match in re.finditer(
            r"^\s*v(\d+)\)",
            build_script.read_text(encoding="utf-8", errors="replace"),
            re.MULTILINE,
        )
    }

    return _sort_vllm_patch_versions(patch_versions & script_versions)


def _parse_vllm_patch_version(version_text: str) -> Optional[str]:
    match = re.search(r"(?:^|[^0-9])v?0\.(\d+)(?:\.\d+)?", version_text.lower())
    if not match:
        return None
    return f"v{int(match.group(1))}"


def infer_vllm_patch_version(
    image_name: str,
    tracelens_repo: Optional[Path] = None,
    installed_version: Optional[str] = None,
) -> Optional[str]:
    """Infer TraceLens vLLM patch shorthand such as v19."""
    patch_version = (
        _parse_vllm_patch_version(installed_version)
        if installed_version
        else None
    )
    if patch_version is None:
        patch_version = _parse_vllm_patch_version(image_name)
    if patch_version is None:
        return None

    if (
        tracelens_repo is not None
        and patch_version not in available_tracelens_vllm_patch_versions(tracelens_repo)
    ):
        return None
    return patch_version


def _vllm_patch_error(
    base_image: str,
    tracelens_repo: Path,
    installed_version: Optional[str] = None,
) -> str:
    inferred = infer_vllm_patch_version(base_image, installed_version=installed_version)
    if inferred is None:
        return (
            "TraceLens auto_patch_runtime cannot infer a vLLM version from "
            f"image {base_image!r} or installed package version "
            f"{installed_version!r}. Set docker_image to a versioned official "
            "vLLM tag, use an image with vLLM installed, use a TraceLens-ready "
            "image, or set "
            "auto_patch_runtime=false."
        )

    supported = available_tracelens_vllm_patch_versions(tracelens_repo)
    supported_text = ", ".join(supported) if supported else "none found"
    return (
        "TraceLens auto_patch_runtime cannot find matching TraceLens support "
        f"for vLLM {inferred} from image {base_image!r}. Available vLLM patch "
        f"versions in {tracelens_repo}: {supported_text}. Update TraceLens, "
        "use a TraceLens-ready image, or set auto_patch_runtime=false."
    )


def _sglang_patch_error(
    base_image: str,
    tracelens_repo: Path,
    installed_version: Optional[str] = None,
) -> str:
    inferred = infer_sglang_patch_version(
        base_image,
        installed_version=installed_version,
    )
    if inferred is None:
        return (
            "TraceLens auto_patch_runtime cannot infer an SGLang version from "
            f"image {base_image!r} or installed package version "
            f"{installed_version!r}. Set docker_image to a versioned official "
            "SGLang tag, use an image with SGLang installed, use a "
            "TraceLens-ready image, or set "
            "auto_patch_runtime=false."
        )

    supported = available_tracelens_sglang_patch_versions(tracelens_repo)
    supported_text = ", ".join(supported) if supported else "none found"
    return (
        "TraceLens auto_patch_runtime cannot find matching TraceLens support "
        f"for SGLang {inferred} from image {base_image!r}. Available SGLang "
        f"patch versions in {tracelens_repo}: {supported_text}. Update "
        "TraceLens, use a TraceLens-ready image, or set "
        "auto_patch_runtime=false."
    )


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


def docker_image_package_version(image_tag: str, package_name: str) -> Optional[str]:
    """Read an installed Python package version from a Docker image."""
    try:
        proc = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "python3",
                image_tag,
                "-c",
                PACKAGE_VERSION_SCRIPT,
                package_name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "Could not read %s package version from Docker image %s: %s",
            package_name,
            image_tag,
            exc,
        )
        return None

    if proc.returncode == 0:
        version = (proc.stdout or "").strip().splitlines()
        if version:
            return version[-1].strip()

    logger.warning(
        "Could not read %s package version from Docker image %s: %s",
        package_name,
        image_tag,
        (proc.stderr or proc.stdout or "").strip()[-1000:],
    )
    return None


def _build_command(
    config: BenchmarkConfig,
    base_image: str,
    runner_type: str,
    derived_image: str,
    tracelens_repo: Path,
    patch_version: str,
) -> list[str]:
    workflow_dir = tracelens_repo / TRACELENS_INFERENCE_WORKFLOW
    if config.framework == "sglang":
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
    package_name = FRAMEWORK_PACKAGE_NAMES.get(config.framework)
    installed_version = (
        docker_image_package_version(base_image, package_name)
        if package_name
        else None
    )
    if installed_version:
        result["runtime_package_version"] = installed_version

    if config.framework == "sglang":
        patch_version = infer_sglang_patch_version(
            base_image,
            tracelens_repo,
            installed_version=installed_version,
        )
        if patch_version is None:
            raise RuntimeError(
                _sglang_patch_error(base_image, tracelens_repo, installed_version)
            )
    elif config.framework == "vllm":
        patch_version = infer_vllm_patch_version(
            base_image,
            tracelens_repo,
            installed_version=installed_version,
        )
        if patch_version is None:
            raise RuntimeError(
                _vllm_patch_error(base_image, tracelens_repo, installed_version)
            )
    else:
        patch_version = None
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
            "patch_version_source": "package" if installed_version else "image",
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
        patch_version=patch_version,
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
    "available_tracelens_sglang_patch_versions",
    "available_tracelens_vllm_patch_versions",
    "derive_tracelens_image_tag",
    "docker_image_package_version",
    "docker_image_exists",
    "infer_sglang_patch_version",
    "infer_vllm_patch_version",
    "is_tracelens_ready_runtime_image",
    "prepare_tracelens_runtime_image",
    "resolve_tracelens_repo_path",
    "runner_type_to_gpu_type",
]
