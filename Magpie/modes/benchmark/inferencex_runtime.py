"""Create run-scoped InferenceX trees without mutating the source checkout."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


INFERENCEX_RUNTIME_RECEIPT_FILENAME = "inferencex_runtime_receipt.json"
INFERENCEX_RUNTIME_RECEIPT_SCHEMA = "magpie.inferencex-runtime-receipt/v1"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class InferenceXRuntime:
    """A run-scoped InferenceX tree and its source/materialization receipt."""

    source_root: Path
    root: Path
    receipt: Dict[str, Any]


def _run_git(
    source_root: Path,
    args: Sequence[str],
    *,
    env: Optional[Dict[str, str]] = None,
) -> str:
    command = ["git", "-C", str(source_root), *args]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not run {' '.join(command[:3])}: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "git command failed").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail[-1000:]}")
    return completed.stdout.strip()


def _git_root(source_root: Path) -> Optional[Path]:
    try:
        top_level = _run_git(source_root, ["rev-parse", "--show-toplevel"])
    except RuntimeError:
        return None
    resolved = Path(top_level).resolve()
    return resolved if resolved == source_root else None


def _git_status(source_root: Path) -> str:
    return _run_git(
        source_root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
    )


def _status_sha256(status: str) -> str:
    return hashlib.sha256(status.encode("utf-8")).hexdigest()


def _checkout_commit(
    source_root: Path,
    runtime_root: Path,
    workspace: Path,
    commit: str,
) -> None:
    """Export ``commit`` through a private temporary Git index.

    This reads blobs from the source object database but never changes its
    working tree, index, refs, or worktree metadata. Unlike copying the source
    filesystem, staged and unstaged changes cannot leak into the runtime tree.
    """

    index_path = workspace / f".inferencex-index-{uuid.uuid4().hex}"
    index_lock = Path(f"{index_path}.lock")
    runtime_root.mkdir(parents=True)
    git_env = os.environ.copy()
    git_env["GIT_INDEX_FILE"] = str(index_path)
    try:
        _run_git(source_root, ["read-tree", commit], env=git_env)
        prefix = f"{runtime_root}{os.sep}"
        _run_git(
            source_root,
            ["checkout-index", "--all", f"--prefix={prefix}"],
            env=git_env,
        )
    finally:
        for temporary in (index_path, index_lock):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _write_receipt(workspace: Path, payload: Dict[str, Any]) -> None:
    receipt_path = workspace / INFERENCEX_RUNTIME_RECEIPT_FILENAME
    temporary = workspace / f".{receipt_path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, receipt_path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def materialize_inferencex_runtime(
    source_root: Path,
    workspace: Path,
) -> InferenceXRuntime:
    """Materialize a private InferenceX tree under a benchmark workspace.

    Git repositories are exported from the exact ``HEAD`` commit through a
    private index. Non-Git directories retain a compatibility path using a
    filesystem copy, clearly marked as unpinned in the receipt.
    """

    source_root = Path(source_root).resolve()
    workspace = Path(workspace).resolve()
    runtime_root = workspace / "inferencex_runtime"
    if not source_root.is_dir():
        raise RuntimeError(f"InferenceX source is not a directory: {source_root}")
    if not (source_root / "benchmarks").is_dir():
        raise RuntimeError(
            f"InferenceX source has no benchmarks directory: {source_root}"
        )
    if workspace == source_root or source_root in workspace.parents:
        raise RuntimeError("benchmark workspace must be outside the InferenceX source")
    if runtime_root.exists():
        raise RuntimeError(f"InferenceX runtime already exists: {runtime_root}")

    git_root = _git_root(source_root)
    if git_root is None and (source_root / ".git").exists():
        raise RuntimeError(
            "InferenceX has Git metadata but its repository root/HEAD could not "
            "be resolved"
        )
    if git_root is not None:
        commit = _run_git(source_root, ["rev-parse", "--verify", "HEAD^{commit}"])
        if not _COMMIT_RE.fullmatch(commit):
            raise RuntimeError(f"InferenceX HEAD is not an exact commit: {commit!r}")
        tree = _run_git(source_root, ["rev-parse", "--verify", "HEAD^{tree}"])
        if not _COMMIT_RE.fullmatch(tree):
            raise RuntimeError(f"InferenceX HEAD tree is not exact: {tree!r}")
        status_before = _git_status(source_root)
        _checkout_commit(source_root, runtime_root, workspace, commit)
        status_after = _git_status(source_root)
        if status_after != status_before:
            raise RuntimeError(
                "InferenceX source status changed while materializing runtime"
            )
        method = "git_private_index_checkout"
        source_is_git = True
        status_digest = _status_sha256(status_before)
        source_clean = not status_before
    else:
        shutil.copytree(
            source_root,
            runtime_root,
            symlinks=True,
            ignore=shutil.ignore_patterns(".git"),
        )
        commit = None
        tree = None
        method = "filesystem_copy"
        source_is_git = False
        status_digest = None
        source_clean = None

    if not (runtime_root / "benchmarks").is_dir():
        raise RuntimeError(
            "materialized InferenceX runtime has no benchmarks directory"
        )

    receipt: Dict[str, Any] = {
        "schema": INFERENCEX_RUNTIME_RECEIPT_SCHEMA,
        "source_root": str(source_root),
        "source_is_git": source_is_git,
        "source_commit": commit,
        "source_tree": tree,
        "source_clean": source_clean,
        "source_status_sha256": status_digest,
        "source_status_unchanged": True,
        "runtime_path": runtime_root.name,
        "materialization_method": method,
    }
    _write_receipt(workspace, receipt)
    return InferenceXRuntime(
        source_root=source_root,
        root=runtime_root,
        receipt=receipt,
    )
