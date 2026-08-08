from pathlib import Path

from Magpie.tools.amd_kernel_finder.finder import KernelSourceFinder
from Magpie.tools.amd_kernel_finder.indexer import KernelDefinition, KernelIndex
from Magpie.tools.amd_kernel_finder.models import KernelKind, KernelSourceInfo
from Magpie.tools.amd_kernel_finder.parser import KernelNameParser


def _definition(
    root: Path,
    *,
    name: str,
    repo_name: str,
    kind: str,
    relative: str,
) -> KernelDefinition:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"def {name}():\n    pass\n", encoding="utf-8")
    return KernelDefinition(
        name=name,
        file_path=relative,
        repo_name=repo_name,
        repo_path=str(root),
        kind=kind,
    )


def _load_definitions(index: KernelIndex, definitions: list[KernelDefinition]) -> None:
    for offset, definition in enumerate(definitions):
        index.index[f"definition-{offset}"] = definition
    index._build_name_index()


def _write_triton_source(root: Path, relative: str, function_name: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "import triton\n\n"
        "@triton.jit\n"
        f"def {function_name}(x):\n"
        "    return x\n",
        encoding="utf-8",
    )


def _make_vllm_root(tmp_path: Path) -> Path:
    root = tmp_path / "vllm-v0.19.1"
    (root / "vllm").mkdir(parents=True)
    (root / "csrc").mkdir()
    return root


def _make_aiter_root(tmp_path: Path) -> Path:
    root = tmp_path / "aiter-v0.1.10.post2"
    (root / "aiter/ops").mkdir(parents=True)
    (root / "csrc/kernels").mkdir(parents=True)
    return root


def test_index_rejects_empty_or_mangled_name_instead_of_first_prefix(tmp_path):
    index = KernelIndex(cache_dir=str(tmp_path / "cache"))
    definition = _definition(
        tmp_path / "vllm",
        name="unrelated_kernel",
        repo_name="vllm",
        kind="triton_jit",
        relative="kernel.py",
    )
    _load_definitions(index, [definition])

    mangled = (
        "_ZN5aiter37dynamic_per_group_scaled_quant_kernel"
        "IDF16bDB8_Li32EEEvPT0_PfPKT_PKfiliibPKii.kd"
    )
    assert index.lookup(mangled) is None
    assert index.lookup("") is None
    assert index.lookup("void aiter::kernel<int>()") is None


def test_source_resolution_fields_preserve_csv_alignment():
    info = KernelSourceInfo(
        kind="triton_jit",
        notes="diagnostic",
        source_resolution="unresolved",
        source_error="triton_source_not_found",
    )

    row = dict(zip(info.csv_headers(), info.to_list(), strict=True))

    assert row["notes"] == "diagnostic"
    assert row["source_resolution"] == "unresolved"
    assert row["source_error"] == "triton_source_not_found"


def test_index_enforces_repo_kind_and_ambiguity_constraints(tmp_path):
    shared = "shared_kernel"
    vllm = _definition(
        tmp_path / "vllm",
        name=shared,
        repo_name="vllm",
        kind="triton_jit",
        relative="vllm_kernel.py",
    )
    aiter = _definition(
        tmp_path / "aiter",
        name=shared,
        repo_name="aiter",
        kind="hip_cpp",
        relative="aiter_kernel.cu",
    )
    index = KernelIndex(cache_dir=str(tmp_path / "cache"))
    _load_definitions(index, [vllm, aiter])

    assert index.lookup(f"{shared}.kd") is None
    assert index.lookup(
        f"{shared}.kd",
        expected_repo_names={"vllm"},
        expected_kinds={"triton_jit"},
    ) == vllm
    assert index.lookup(
        f"{shared}.kd",
        expected_repo_names={"aiter"},
        expected_kinds={"triton_jit"},
    ) is None


def test_index_rejects_ambiguous_token_boundary_prefix(tmp_path):
    root = tmp_path / "vllm"
    shorter = _definition(
        root,
        name="paged_kernel",
        repo_name="vllm",
        kind="triton_jit",
        relative="short.py",
    )
    longer = _definition(
        root,
        name="paged_kernel_decode",
        repo_name="vllm",
        kind="triton_jit",
        relative="long.py",
    )
    index = KernelIndex(cache_dir=str(tmp_path / "cache"))
    _load_definitions(index, [shorter, longer])

    assert index.lookup(
        "paged_kernel_decode_config_64.kd",
        expected_repo_names={"vllm"},
        expected_kinds={"triton_jit"},
    ) is None


def test_triton_search_uses_explicit_vllm_and_aiter_roots(tmp_path):
    vllm = _make_vllm_root(tmp_path)
    aiter = _make_aiter_root(tmp_path)
    _write_triton_source(
        vllm,
        "vllm/v1/attention/ops/chunked_prefill_paged_decode.py",
        "kernel_paged_attention_2d",
    )
    _write_triton_source(
        aiter,
        "aiter/ops/triton/_triton_kernels/quant/aiter_quant.py",
        "aiter_quant_kernel",
    )
    finder = KernelSourceFinder(
        repos=[str(vllm), str(aiter)],
        auto_clone=False,
        use_index=False,
        auto_install_ripgrep=False,
    )

    page = finder.search("kernel_paged_attention_2d.kd")
    quant = finder.search("aiter_quant_kernel.kd")

    assert page.source_resolution == "resolved"
    assert page.source_file == (
        "$VLLM_DIR/vllm/v1/attention/ops/"
        "chunked_prefill_paged_decode.py"
    )
    assert quant.source_resolution == "resolved"
    assert quant.source_file == (
        "$AITER_DIR/aiter/ops/triton/_triton_kernels/quant/aiter_quant.py"
    )


def test_triton_cache_miss_emits_unresolved_without_placeholder(
    tmp_path,
    monkeypatch,
):
    vllm = _make_vllm_root(tmp_path)
    rogue = _make_vllm_root(tmp_path / "rogue")
    _write_triton_source(rogue, "vllm/rogue.py", "missing_kernel")
    monkeypatch.setenv("VLLM_DIR", str(rogue))
    finder = KernelSourceFinder(
        repos=[str(vllm)],
        auto_clone=False,
        use_index=False,
        auto_install_ripgrep=False,
    )

    result = finder.search("missing_kernel.kd")

    assert result.source_resolution == "unresolved"
    assert result.source_error == "triton_source_not_found"
    assert result.source_file == ""
    assert result.source_repo == ""
    assert "search in" not in result.notes


def test_duplicate_triton_definition_across_trusted_roots_fails_closed(tmp_path):
    vllm = _make_vllm_root(tmp_path)
    aiter = _make_aiter_root(tmp_path)
    _write_triton_source(vllm, "vllm/shared.py", "shared_kernel")
    _write_triton_source(aiter, "aiter/ops/shared.py", "shared_kernel")
    finder = KernelSourceFinder(
        repos=[str(vllm), str(aiter)],
        auto_clone=False,
        use_index=False,
        auto_install_ripgrep=False,
    )

    result = finder.search("shared_kernel.kd")

    assert result.source_resolution == "unresolved"
    assert result.source_error == "triton_source_ambiguous"
    assert result.source_file == ""


def test_unmaterialized_aiter_ck_submodule_is_explicitly_unsupported(tmp_path):
    aiter = _make_aiter_root(tmp_path)
    (aiter / ".gitmodules").write_text(
        "[submodule \"3rdparty/composable_kernel\"]\n"
        "    path = 3rdparty/composable_kernel\n"
        "    url = https://github.com/ROCm/composable_kernel.git\n",
        encoding="utf-8",
    )
    (aiter / "3rdparty/composable_kernel").mkdir(parents=True)
    finder = KernelSourceFinder(
        repos=[str(aiter)],
        auto_clone=False,
        use_index=False,
        auto_install_ripgrep=False,
    )
    kernel_name = (
        "kernel_gemm_xdl_cshuffle_v3<"
        "GridwiseGemmMultiD_ABScale<bf16>>.kd"
    )

    assert KernelNameParser().parse(kernel_name).kind == KernelKind.CK_TILE
    result = finder.search(kernel_name)

    assert result.source_resolution == "unsupported"
    assert result.source_error == "ck_submodule_not_materialized"
    assert result.source_file == ""
    assert result.source_repo == ""
    assert result.test_file == ""
    assert "source_resolution=unsupported" in result.notes


def test_materialized_aiter_ck_source_is_bound_to_exact_file(tmp_path):
    aiter = _make_aiter_root(tmp_path)
    ck_source = (
        aiter
        / "3rdparty/composable_kernel/include/ck_tile/ops/gemm/kernel/"
        "gemm_kernel.hpp"
    )
    ck_source.parent.mkdir(parents=True)
    ck_source.write_text("// fixture\n", encoding="utf-8")
    finder = KernelSourceFinder(
        repos=[str(aiter)],
        auto_clone=False,
        use_index=False,
        auto_install_ripgrep=False,
    )

    result = finder.search(
        "kernel_gemm_xdl_cshuffle_v3<GridwiseGemmMultiD_ABScale>.kd"
    )

    assert result.source_resolution == "resolved"
    assert result.source_error == ""
    assert result.source_file == (
        "$AITER_DIR/3rdparty/composable_kernel/"
        "include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
    )
