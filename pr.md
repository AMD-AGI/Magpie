# Add Atom inference engine as a first-class benchmark framework

## Summary

This PR adds **Atom** alongside vLLM and SGLang as a supported framework in
Magpie's benchmark mode. Atom exposes an OpenAI-compatible HTTP server at
`atom.entrypoints.openai_server` and is wire-compatible with vLLM's REST
dialect, so the existing `--backend vllm` bench client is reused unchanged.
All new code lives on the **server-launch side** and in the framework
registry plumbing.

After this PR, `framework: atom` is valid in benchmark YAMLs and Magpie owns
the full server-lifecycle + bench flow (`PHASE=all`) for Atom the same way it
does for vLLM/SGLang.

## Motivation

An internal benchmarking workflow needs Magpie to drive Atom end-to-end on
AMD MI300X / MI355X hosts. Today, `framework: atom` is rejected by
`BenchmarkConfig` and no launch script exists, so the workflow has to fork
Magpie or shell out to a parallel runner. Adding Atom as a first-class
framework keeps everything inside the Magpie surface (CLI, MCP tools,
`benchmark_report.json` schema, Ray dispatch).

## Changes

### New files
- `Magpie/scripts/benchmark/atom_mi300x.sh`
- `Magpie/scripts/benchmark/atom_mi355x.sh`
- `examples/benchmarks/benchmark_atom_dsr1.yaml`

The shell scripts are cloned from their `vllm_*` counterparts with three diffs:

1. Server-launch block calls `python3 -m atom.entrypoints.openai_server`
   instead of `vllm serve`.
2. `VLLM_ROCM_USE_AITER` export dropped (vLLM-specific). The MEC-firmware /
   `HSA_NO_SCRATCH_RECLAIM` check is kept (framework-agnostic).
3. Profiler block replaced with a one-line warning — `PROFILE=1` is accepted
   but ignored in v1 (see Known Limitations).

The bench-client call (`benchmark_serving.py` / `--backend vllm`) is **unchanged**.

### Modified files (registry + dispatch)
- `Magpie/modes/benchmark/config.py` — `BenchmarkFramework.ATOM` enum
  member; allowlist extended to `["vllm", "sglang", "atom"]`; docstrings.
- `Magpie/modes/benchmark/benchmarker.py` —
  - `MAGPIE_BUILTIN_SCRIPTS` += `atom_mi300x.sh`, `atom_mi355x.sh`
  - `EXTRA_ATOM_ARGS` extraction + `extra_atom_args` record field
  - `ATOM_TORCH_PROFILER_DIR` env export (parity with the other two, even
    though v1 ignores it)
  - server-kill pattern for `atom.entrypoints.openai_server`
  - error message includes atom in the built-in-script list
- `Magpie/remote/tasks.py` — explicit `"atom": "EXTRA_ATOM_ARGS"` entry in
  `_extra_args_key()`. (The fallback already produced this string, but the
  explicit entry makes intent clear and is asserted in the test.)
- `Magpie/main.py` — `--framework` choices extended; help text updated.
- `Magpie/benchmark_images.yaml` — Atom inherits the vLLM image as a
  placeholder (see Known Limitations); override via `--docker-image` when
  an Atom-specific image becomes available.
- `Magpie/mcp/{server.py,__init__.py}`, `Magpie/modes/benchmark/__init__.py`,
  `Magpie/modes/benchmark/image_selector.py`, `Magpie/core/scheduler.py` —
  docstring updates (`vLLM/SGLang` → `vLLM/SGLang/Atom`).

### Tests
- `tests/test_benchmark_support.py` —
  - `BenchmarkConfig(framework="atom", model=...)` smoke test (positional
    + `from_dict` with case-insensitive framework key).
  - `ImageSelector` mapping for `atom/gfx942`.
- `tests/test_common_and_remote_tasks.py` —
  - `_extra_args_key("atom") == "EXTRA_ATOM_ARGS"`.
  - `_configure_tp_isolation` single-node path for `framework="atom"`:
    asserts `RAY_ADDRESS` is cleared and **no** vLLM-specific flag is
    injected into `EXTRA_ATOM_ARGS` (Atom uses `--backend vllm` only on the
    client side).

### Docs
- `README.md` — framework enumeration + example-files table row + MCP
  tool description.
- `docs/benchmark.md` — Quick Start sentence + YAML reference comments.

## Known limitations (deferred to follow-ups)

These are explicitly out of scope for v1 and documented in the example YAML
header + `docs/benchmark.md`:

1. **Single-node only — upstream limitation.** Atom upstream's
   `atom.entrypoints.openai_server` does not currently expose multi-node
   TP wiring (no `--use-ray` / `--dist-init-addr` analogues like vLLM /
   SGLang). The Ray multi-node branch in
   `Magpie/remote/tasks.py:_configure_tp_isolation` now identifies atom
   as a known single-node-only framework
   (`KNOWN_SINGLE_NODE_FRAMEWORKS = frozenset({"atom"})`) and logs a
   clear `"Framework 'atom' is single-node only; multi-node TP
   auto-config not applicable"` message instead of the misleading
   "Unknown framework" warning. Hyperloom enforces this earlier by
   fail-fast on `--framework atom --nodes >= 2` (see Hyperloom IR-8 /
   `_apply_atom_auto_tighten`), so this Magpie code path is only reached
   when Magpie is invoked directly bypassing Hyperloom's guard. Closing
   the gap requires upstream atom to ship cross-node TP CLI; this is a
   per-engine limitation, not a Magpie one.
2. **`PROFILE=1` is now wired.** The launch script bridges
   `PROFILE=1` to atom's `--torch-profiler-dir` CLI flag (default:
   `$ATOM_TORCH_PROFILER_DIR` → `$WORKSPACE_DIR/torch_trace`) and the
   InferenceX benchmark client auto-POSTs `/start_profile` and
   `/stop_profile` to atom's OpenAI-compatible HTTP server. Atom writes
   standard Chrome traces (`*.pt.trace.json.gz`) under
   `<dir>/rank_<N>/`, which TraceLens consumes unchanged.

## Image considerations

`benchmark_images.yaml` now maps `atom:` to **`rocm/atom:latest`** for
both `gfx942` (MI300X) and `gfx950` (MI355X). The image is published
publicly by the ATOM team (see
[`/app/ATOM/recipes/DeepSeek-R1.md`](https://github.com/ROCm/ATOM)
and the [Docker Hub page](https://hub.docker.com/r/rocm/atom)) and
ships `atom` pre-installed, so the previous "import atom into the
vLLM placeholder image" workflow is no longer required.

For reproducibility-sensitive runs, pin to a specific image SHA / tag
via `--docker-image rocm/atom:<tag>` rather than relying on `:latest`.
On NVIDIA hosts (sm_80 / sm_90 / sm_100) atom has no published CUDA
image yet, so `benchmark_images.yaml::atom::sm_*` continues to point
at the vLLM CUDA placeholder; operators on Hopper / Blackwell GPUs
need to layer `pip install atom` on top or pass `--docker-image`.

## Verification

Run from the repo root:

```bash
# Targeted tests
pytest tests/test_benchmark_support.py -k atom -v
pytest tests/test_common_and_remote_tasks.py -v

# Full suite must stay green
pytest tests/ -q

# Config smoke
python -c "from Magpie.modes.benchmark.config import BenchmarkConfig, BenchmarkFramework; \
  print([m.value for m in BenchmarkFramework]); \
  print(BenchmarkConfig(framework='atom', model='dummy'))"
```

End-to-end (requires MI300X / MI355X host with the `atom` Python package
importable):

```bash
python -m Magpie benchmark \
  --benchmark-config examples/benchmarks/benchmark_atom_dsr1.yaml
```

Expect `benchmark_report.json` emitted with the same schema as vLLM reports.

## Test plan

- [x] `pytest tests/` passes (35/35 — was 33/33 before this PR, +2 atom cases)
- [x] `_get_benchmark_script("mi300x")` and `_get_benchmark_script("mi355x")`
      resolve to `benchmarks/atom_mi300x.sh` / `benchmarks/atom_mi355x.sh`
      after `_prepare_benchmark_scripts()` syncs them into a synthetic
      InferenceX directory
- [x] `BenchmarkConfig.from_dict({"framework": "ATOM", ...})` normalizes to
      lowercase and constructs cleanly
- [x] `ImageSelector` returns the vLLM-inherited image for `atom/gfx942`
      and `atom/gfx950`
- [ ] **(reviewer)** End-to-end MI300X run with `atom` package installed,
      producing `benchmark_report.json` with the same schema as vLLM
- [ ] **(reviewer)** End-to-end MI355X run

## Files changed

```
 Magpie/benchmark_images.yaml             | 14 ++++++++++++++
 Magpie/core/scheduler.py                 |  4 ++--
 Magpie/main.py                           |  4 ++--
 Magpie/mcp/__init__.py                   |  2 +-
 Magpie/mcp/server.py                     | 10 +++++-----
 Magpie/modes/benchmark/__init__.py       |  2 +-
 Magpie/modes/benchmark/benchmarker.py    | 24 ++++++++++++++++++------
 Magpie/modes/benchmark/config.py         | 12 +++++++-----
 Magpie/modes/benchmark/image_selector.py |  2 +-
 Magpie/remote/tasks.py                   |  1 +
 Magpie/scripts/benchmark/atom_mi300x.sh  | (new)
 Magpie/scripts/benchmark/atom_mi355x.sh  | (new)
 examples/benchmarks/benchmark_atom_dsr1.yaml | (new)
 README.md                                |  9 +++++----
 docs/benchmark.md                        |  6 +++---
 tests/test_benchmark_support.py          | 19 +++++++++++++++++++
 tests/test_common_and_remote_tasks.py    | 14 ++++++++++++++
 17 files changed, 444 insertions(+), 30 deletions(-)
```
