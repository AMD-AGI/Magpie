# Analyze vs Compare

Magpie’s **Analyze** and **Compare** modes both evaluate GPU kernels (HIP, CUDA, PyTorch, Triton) through the same underlying pipeline—compile (optional), correctness, and optional performance profiling—but they differ in how many kernels you evaluate and how a “winner” is chosen.

## At a glance

| | **Analyze** | **Compare** |
|---|-------------|-------------|
| **Goal** | Validate one implementation end-to-end | Rank two or more implementations |
| **Kernels** | One (or multiple independent runs from one config) | At least two |
| **Testcase** | **Required** (CLI or YAML `testcase_command`) | Optional per kernel; if omitted, PyTorch can use **result comparison** between variants |
| **Outcome** | Per-kernel `EvaluationState` | `ComparisonResult`: correctness vector, perf scores, rankings, `winner` |
| **CLI** | `magpie analyze …` | `magpie compare …` |
| **Report file** | `analyze_report.json` | `compare_report.json` |

For architecture and diagrams, see the [README](https://github.com/AMD-AGI/Magpie#readme) (Analyze & Compare pipeline image).

## When to use which

- **Analyze** when you have a single kernel (or a small set you want to evaluate independently) and a clear test command (build + run test, or script that exits non-zero on failure).
- **Compare** when you have multiple source variants (e.g. v1 vs v2 HIP, or several PyTorch implementations) and want Magpie to run them in sequence, check correctness, optionally profile each, and produce a **ranking** and **winner** using configured scoring rules.

## Correctness behavior

### Analyze

- `AnalyzeMode` **requires** `testcase_command` in the effective `KernelEvalConfig`. Without it, analysis stops with an error.
- Use this mode when your validation story is “run this command and trust exit status / Accordo backend output.”

### Compare

- If a kernel has `testcase_command`, correctness uses the **testcase** path (same idea as analyze).
- If **no** testcase is provided, compare can use **result comparison** mode for PyTorch-style workflows (outputs compared across variants).
- You need **at least two** kernel entries (from CLI paths or YAML `kernels:` list).

## CLI quick reference

```bash
# Analyze: kernel file(s) + testcase (required without --kernel-config)
magpie analyze path/to/kernel.hip -t "./run_test.sh"

# Analyze from YAML (kernel: or kernels: — analyze still needs testcase per entry)
magpie analyze --kernel-config Magpie/kernel_config.yaml.example

# Compare: at least two kernels
magpie compare kernel_a.hip kernel_b.hip

# Optional shared testcase on CLI (applied when using positional kernel files)
magpie compare k1.hip k2.hip -t "./run_both.sh"

# Compare from YAML (e.g. examples/ck_grouped_gemm_compare.yaml)
magpie compare --kernel-config examples/ck_grouped_gemm_compare.yaml

# Skip profiling (both modes)
magpie analyze ... --no-perf
magpie compare ... --no-perf

# Baseline index for compare (YAML / framework config still controls winner strategy)
magpie compare k0.hip k1.hip --baseline 0
```

Kernel types: `--type` accepts `hip`, `cuda`, `pytorch`, `triton`.

## YAML configuration

Shared rules:

- **`kernel:`** — single kernel block.
- **`kernels:`** — list of kernel blocks (typical for compare).
- Optional sections in the same file: `performance`, `correctness`, `scheduler`, `ray_config` (see `load_kernel_config` in `Magpie/main.py`).

Examples in-repo:

| Mode | Example file |
|------|----------------|
| Analyze | `examples/ck_gemm_add.yaml`, `examples/simple_hip_test/analyze_default.yaml` |
| Compare | `examples/ck_grouped_gemm_compare.yaml` |

## Framework config (`Magpie/config.yaml`)

Both modes read **GPU**, **scheduler**, **compiling**, **performance**, and **correctness** settings.

**Compare-only** (`compare:`):

- **`perf_weights_rocprof` / `perf_weights_ncu` / `perf_weights_metrix`** — weights used when turning profiler summaries into scalar **perf scores**.
- **`perf_lower_is_better`** — metric names where lower values score better (defaults include duration-style metrics).
- **`winner_strategy`** (passed through from config to `CompareConfig`): typically `perf_score` (highest score wins among ranked kernels) or `correctness_first` (first kernel that passes correctness). Default in code is `perf_score`.

Tune these when your comparison should emphasize different hardware metrics or when correctness should override raw performance.

## Workspace output

Analyze and compare runs create timestamped workspaces under `--output-dir` (default `./results`):

- **Analyze:** `analyze_report.json` plus profiler output under `performance/` when profiling is enabled; config snapshot and correctness artifacts as configured.
- **Compare:** `compare_report.json` with `kernel_results`, `comparison_metrics` (including `correctness`, `perf_scores`, `all_correct`), `rankings`, `winner`, and `summary`.

## Related documentation

- [Benchmark mode](benchmark.md) — vLLM/SGLang framework benchmarks (separate from kernel analyze/compare).
- [Skills install](mcp-and-skills.md) — using Magpie without MCP.
- [Ray scheduling (EN)](ray.md) — remote execution when `scheduler.environment: ray`.
