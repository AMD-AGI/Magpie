###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
InferenceX repository management.

Handles automatic cloning and validation of the InferenceX repository
for benchmark execution.
"""

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# InferenceX repository configuration
INFERENCEX_REPO_URL = "https://github.com/SemiAnalysisAI/InferenceX.git"
INFERENCEX_DEFAULT_REF = "408c015be4b22d14c69518643609669405507077"
# Default directory resolution order (first writable candidate wins):
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
INFERENCEX_DEFAULT_DIR = str(_PROJECT_ROOT / "InferenceX")

# Placeholder values that indicate the path is not configured
PLACEHOLDER_VALUES = {
    "YOUR_INFERENCEX_PATH",
    "",
}


def resolve_inferencex_root(path: str) -> str:
    """Return a supported InferenceX project root without altering the checkout.

    A benchmarks directory alone is not an installation: Magpie may have copied
    its own scripts there. Require the upstream library and client source too.
    """
    root = Path(path).expanduser().resolve()
    candidates = []
    for candidate in (root, root / "inferencex-e2e"):
        if not (candidate / "benchmarks" / "benchmark_lib.sh").is_file():
            continue
        legacy = candidate / "utils" / "bench_serving" / "benchmark_serving.py"
        packaged = candidate / "infx" / "bench_serving" / "benchmark_serving.py"
        if legacy.is_file() or (
            packaged.is_file() and (candidate / "pyproject.toml").is_file()
        ):
            candidates.append(candidate)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one supported InferenceX project at {root}; provide the "
            "repository root or its inferencex-e2e directory containing "
            "benchmarks/benchmark_lib.sh and the upstream benchmark client"
        )
    return str(candidates[0])


def resolve_benchmark_serving(path: str) -> Path:
    """Locate the supported client source for profiling and runtime adapters."""
    root = Path(resolve_inferencex_root(path))
    packaged = root / "infx" / "bench_serving" / "benchmark_serving.py"
    if packaged.is_file() and (root / "pyproject.toml").is_file():
        return packaged
    return root / "utils" / "bench_serving" / "benchmark_serving.py"


def _resolve_default_inferencex_dir() -> str:
    """Pick the first writable default location for InferenceX.

    Precedence:
      1. $MAGPIE_INFERENCEX_PATH (explicit override)
      2. <magpie-repo>/../InferenceX (sibling of Magpie checkout)
      3. $XDG_CACHE_HOME/magpie/InferenceX or ~/.cache/magpie/InferenceX
      4. /root/workspace/InferenceX (legacy, for root-in-container setups)
    """
    env_override = os.environ.get("MAGPIE_INFERENCEX_PATH")
    if env_override:
        return env_override

    candidates = [
        INFERENCEX_DEFAULT_DIR,
        str(
            Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
            / "magpie"
            / "InferenceX"
        ),
        "/root/workspace/InferenceX",
    ]
    for candidate in candidates:
        parent = os.path.dirname(candidate) or "/"
        # Reuse if it already exists
        if os.path.isdir(candidate):
            return candidate
        # Otherwise pick the first candidate whose parent we can create/write to
        try:
            os.makedirs(parent, exist_ok=True)
            if os.access(parent, os.W_OK):
                return candidate
        except OSError:
            continue
    return INFERENCEX_DEFAULT_DIR


class InferenceXManager:
    """
    Manages InferenceX repository for benchmark execution.

    Handles:
    - Validation of existing installation
    - Automatic cloning if not present
    - Path resolution for placeholder values
    """

    def __init__(
        self,
        repo_url: str = INFERENCEX_REPO_URL,
        default_dir: str | None = None,
    ):
        """
        Initialize InferenceX manager.

        Args:
            repo_url: Git repository URL for InferenceX
            default_dir: Default directory to clone into. If None, resolves
                a writable default at call time (see _resolve_default_inferencex_dir).
        """
        self.repo_url = repo_url
        self.default_dir = default_dir or _resolve_default_inferencex_dir()

    def ensure_available(self, configured_path: str | None = None) -> str:
        """
        Ensure InferenceX is available at the specified path.

        If the path doesn't exist or is a placeholder value, automatically
        clone InferenceX from the repository.

        Args:
            configured_path: Configured path to InferenceX (may be None or placeholder)

        Returns:
            Valid path to InferenceX installation

        Raises:
            RuntimeError: If unable to clone or validate InferenceX
        """
        # Determine if path is a placeholder or not configured
        is_placeholder = self._is_placeholder(configured_path)

        # Determine target directory
        target_dir = (
            self.default_dir
            if is_placeholder or not configured_path
            else configured_path
        )
        target_dir = str(Path(target_dir).expanduser().resolve())

        # Check if already exists at target directory
        if os.path.exists(target_dir):
            project_root = resolve_inferencex_root(target_dir)
            logger.info(f"InferenceX already exists at: {project_root}")
            return project_root

        # Clone the repository
        return self._clone_repository(target_dir)

    def _is_placeholder(self, path: str | None) -> bool:
        """Check if the path is a placeholder value."""
        if path is None:
            return True
        return path in PLACEHOLDER_VALUES

    def _validate_installation(self, path: str) -> bool:
        """
        Validate that the path contains a valid InferenceX installation.

        Args:
            path: Path to validate

        Returns:
            True if valid InferenceX installation
        """
        try:
            resolve_inferencex_root(path)
        except (OSError, RuntimeError):
            return False
        return True

    def _clone_repository(self, target_dir: str) -> str:
        """
        Clone InferenceX repository.

        Args:
            target_dir: Directory to clone into

        Returns:
            Path to cloned repository

        Raises:
            RuntimeError: If clone fails
        """
        logger.info(f"InferenceX not found. Cloning from {self.repo_url}...")
        logger.info(f"Clone destination: {target_dir}")

        try:
            # Ensure parent directory exists
            parent_dir = os.path.dirname(target_dir)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            commands = [
                ["git", "clone", "--no-checkout", self.repo_url, target_dir],
                [
                    "git",
                    "-C",
                    target_dir,
                    "checkout",
                    "--detach",
                    INFERENCEX_DEFAULT_REF,
                ],
                [
                    "git",
                    "-C",
                    target_dir,
                    "submodule",
                    "update",
                    "--init",
                    "--recursive",
                    "--",
                    "inferencex-e2e/utils/aiperf",
                ],
            ]
            for command in commands:
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=300, check=False
                )
                if result.returncode != 0:
                    error_msg = (
                        result.stderr.strip() if result.stderr else "Unknown error"
                    )
                    raise RuntimeError(
                        f"InferenceX {' '.join(command)} failed: {error_msg}"
                    )

            logger.info(f"Successfully cloned InferenceX to: {target_dir}")
            return resolve_inferencex_root(target_dir)

        except subprocess.TimeoutExpired:
            logger.error("InferenceX installation command timed out after 5 minutes")
            raise RuntimeError("InferenceX git command timed out after 5 minutes")
        except FileNotFoundError:
            logger.error("git command not found. Please install git.")
            raise RuntimeError("git is not installed. Please install git first.")
        except OSError as e:
            logger.error(f"Failed to create directory or clone: {e}")
            raise RuntimeError(f"Failed to setup InferenceX: {e}")


# Module-level instance for convenience
_manager: InferenceXManager | None = None


def get_manager() -> InferenceXManager:
    """Get the global InferenceX manager instance."""
    global _manager
    if _manager is None:
        _manager = InferenceXManager()
    return _manager


def ensure_inferencex_available(configured_path: str | None = None) -> str:
    """
    Convenience function to ensure InferenceX is available.

    Args:
        configured_path: Configured path to InferenceX

    Returns:
        Valid path to InferenceX installation
    """
    return get_manager().ensure_available(configured_path)
