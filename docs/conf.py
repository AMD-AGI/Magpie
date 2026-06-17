# Configuration file for the Sphinx documentation builder.
#
# Magpie documentation is built with rocm-docs-core, which configures the
# theme, navigation, MyST Markdown support, and shared ROCm options. Both
# Markdown (.md, via MyST) and reStructuredText (.rst) source files build out
# of the box.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# https://rocm.docs.amd.com/projects/rocm-docs-core/en/latest/

# -- Project information ------------------------------------------------------

project = "Magpie"
author = "Advanced Micro Devices, Inc."
copyright = "2026, Advanced Micro Devices, Inc."

# Single-sourced version. Update alongside pyproject.toml / package version.
version = "0.1.0"
release = version

# -- General configuration ----------------------------------------------------

extensions = ["rocm_docs", "sphinxcontrib.mermaid"]

# Render fenced ```mermaid code blocks in Markdown as diagrams.
myst_fence_as_directive = ["mermaid"]

external_toc_path = "./sphinx/_toc.yml"

# docs/README.md documents the build process for contributors and is not a
# published page; keep it out of the source build so it is not treated as an
# orphan document.
exclude_patterns = ["README.md"]

# rocm-docs-core options.
html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm-docs-home"}
