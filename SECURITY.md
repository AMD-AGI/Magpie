# Security Policy

## Reporting a Vulnerability

**Do not open a public GitHub issue.** Report privately via one of:

- **GitHub Private Vulnerability Reporting:** [Report a vulnerability](https://github.com/AMD-AGI/Magpie/security/advisories/new)
- **AMD Product Security portal:** https://www.amd.com/en/resources/product-security.html

Please include: description and impact, steps to reproduce, and affected versions or commits.

We aim to acknowledge reports within 1 business day.

## Scope

This policy covers code and configuration in this repository — the Magpie kernel evaluation framework, including the compilation, correctness, and performance grading pipeline.

Because Magpie compiles and executes user-supplied kernels, please flag any sandbox-escape, arbitrary-execution, or resource-exhaustion issues privately.

For issues in third-party dependencies (ROCm, PyTorch, Triton, HIP/CUDA toolchains) report upstream. For AMD product issues unrelated to this repo, use the [AMD Product Security portal](https://www.amd.com/en/resources/product-security.html).
