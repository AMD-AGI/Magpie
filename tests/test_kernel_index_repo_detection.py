import json
from pathlib import Path

from Magpie.tools.amd_kernel_finder.indexer import KernelIndex


def _write_triton_kernel(root: Path, relative_path: str, name: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""import triton

@triton.jit
def {name}(x):
    return x
""",
        encoding="utf-8",
    )


def test_kernel_index_canonicalizes_versioned_aiter_root(tmp_path):
    repo = tmp_path / "aiter-v0.1.10.post2"
    (repo / "aiter/ops").mkdir(parents=True)
    (repo / "csrc/kernels").mkdir(parents=True)
    _write_triton_kernel(
        repo,
        "aiter/ops/triton/versioned_kernel.py",
        "versioned_aiter_kernel",
    )
    index = KernelIndex(cache_dir=str(tmp_path / "cache"))

    index.build([str(repo)], force_rebuild=True)
    definition = index.lookup("versioned_aiter_kernel.kd")

    assert definition is not None
    assert definition.repo_name == "aiter"
    assert definition.file_path == "aiter/ops/triton/versioned_kernel.py"
    repo_var = f"${definition.repo_name.upper().replace('-', '_')}_DIR"
    assert f"{repo_var}/{definition.file_path}" == (
        "$AITER_DIR/aiter/ops/triton/versioned_kernel.py"
    )


def test_kernel_index_keeps_versioned_vllm_root_canonical(tmp_path):
    repo = tmp_path / "vllm-v0.19.1"
    (repo / "vllm").mkdir(parents=True)
    (repo / "csrc").mkdir()
    _write_triton_kernel(
        repo,
        "vllm/lora/ops/triton_ops/versioned_kernel.py",
        "versioned_vllm_kernel",
    )
    index = KernelIndex(cache_dir=str(tmp_path / "cache"))

    index.build([str(repo)], force_rebuild=True)
    definition = index.lookup("versioned_vllm_kernel.kd")

    assert definition is not None
    assert definition.repo_name == "vllm"
    assert definition.file_path == (
        "vllm/lora/ops/triton_ops/versioned_kernel.py"
    )
    repo_var = f"${definition.repo_name.upper().replace('-', '_')}_DIR"
    assert repo_var == "$VLLM_DIR"


def test_kernel_index_rebuilds_pre_canonicalization_cache(tmp_path):
    repo = tmp_path / "aiter-v0.1.10.post2"
    (repo / "aiter/ops").mkdir(parents=True)
    (repo / "csrc/kernels").mkdir(parents=True)
    _write_triton_kernel(repo, "aiter/ops/cached.py", "cached_aiter_kernel")
    index = KernelIndex(cache_dir=str(tmp_path / "cache"))
    cache_file = index._get_cache_file(str(repo))
    cache_file.write_text(
        json.dumps(
            {
                "mtime": repo.stat().st_mtime,
                "definitions": {
                    "stale": {
                        "name": "cached_aiter_kernel",
                        "file_path": "aiter/ops/cached.py",
                        "repo_name": "aiter-v0.1.10.post2",
                        "repo_path": str(repo),
                        "kind": "triton_jit",
                        "line_number": 1,
                        "symbol": "stale",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    index.build([str(repo)])
    definition = index.lookup("cached_aiter_kernel.kd")

    assert definition is not None
    assert definition.repo_name == "aiter"
    refreshed = json.loads(cache_file.read_text(encoding="utf-8"))
    assert refreshed["schema_version"] == KernelIndex.CACHE_SCHEMA_VERSION
