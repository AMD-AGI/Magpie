# Magpie documentation

This directory contains the source for the Magpie documentation site, which is
intended for publication on [rocm.docs.amd.com](https://rocm.docs.amd.com/) as a
component of the Hyperloom toolkit. It follows the same structure as other ROCm
toolkit component docs (for example,
[ROCm-LLMExt](https://rocm.docs.amd.com/projects/rocm-llmext/en/latest/index.html)).

The site is built with [Sphinx](https://www.sphinx-doc.org/) and
[rocm-docs-core](https://github.com/ROCm/rocm-docs-core). Both Markdown (MyST,
`.md`) and reStructuredText (`.rst`) sources are supported and can be mixed.

## Building locally

```bash
# From the repository root
python3 -m venv .docvenv
source .docvenv/bin/activate
pip install -r docs/sphinx/requirements.txt

python -m sphinx -T -b html docs docs/_build/html
# Open docs/_build/html/index.html
```

## Layout

| Path | Required page | Notes |
| --- | --- | --- |
| `index.rst` | Overview | Landing page; feature summary, use cases, links to all subpages and the GitHub repo. |
| `install/install.md` | Installation Instructions | pip-from-GitHub, source/editable, and `make` methods, plus verification. |
| `reference/release-notes.md` | Release Notes | Per-release feature breakdown, packaging changes, and release notes. |
| `reference/compatibility-matrix.md` | Compatibility Matrix | Verified hardware/software versions. Contains `TODO (verify)` markers. |
| `reference/api-reference.md` | API Reference | CLI commands and options, configuration schema, and MCP tools. |
| `how-to/analyze-compare.md` | How-to | Analyze vs compare kernel modes. |
| `how-to/benchmarking/benchmark.md` | How-to | vLLM/SGLang/Atom benchmarking, TraceLens, gap analysis. |
| `how-to/ray.md` | How-to | Remote execution on a Ray cluster. |
| `how-to/mcp-and-skills.md` | How-to | MCP server and agent skill installation. |
| `how-to/kernel-source-finder.md` | How-to | Locating kernel sources from traces. |
| `examples/examples.md` | Examples | Step-by-step walkthroughs with expected output. |
| `about/license.md` | License | Full MIT license text (mirrors the repo `LICENSE`). |

Navigation (the left sidebar) is defined in `sphinx/_toc.yml.in`.

## Configuration files

| File | Purpose |
| --- | --- |
| `../.readthedocs.yaml` | Read the Docs build configuration. |
| `conf.py` | Sphinx configuration (rocm-docs-core, MyST, mermaid). |
| `sphinx/_toc.yml.in` | Table of contents / sidebar navigation. |
| `sphinx/requirements.in` | Top-level documentation dependencies. |
| `sphinx/requirements.txt` | Pinned documentation dependencies used by the build. |

## Open items for the Magpie team

These must be resolved before publication:

- **Compatibility matrix**: replace every `TODO (verify)` entry in
  `reference/compatibility-matrix.md` with verified, tested versions (ROCm/CUDA,
  GPU architectures, framework versions, Docker, Ray, profilers).
- **About / Resources**: add the public Magpie blog URL once available (the
  ROCm-LLMExt site links a per-component blog under "Resources").
- **API reference depth**: decide whether to keep the hand-written CLI/MCP/config
  reference or augment it with `autodoc`/`autosummary` generated from Python
  docstrings.
- **Project registration**: during ROCm onboarding, register `Magpie` in the
  rocm-docs-core project registry. Until then, a local build prints a benign
  `Current project 'Magpie' not found in projects` warning.

## Notes on the local build

A local build outside AMD infrastructure prints benign warnings for intersphinx
inventory fetches (for example, `instinct.docs.amd.com/objects.inv`); these
resolve on the ROCm build infrastructure. The build itself succeeds.
