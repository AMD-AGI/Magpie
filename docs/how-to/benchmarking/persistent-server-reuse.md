---
myst:
    html_meta:
        "description": "Keep a Magpie inference server alive across successive benchmark runs using server_lifecycle.enabled to avoid model reload overhead between client invocations."
        "keywords": "Magpie, persistent server, server reuse, server_lifecycle, vLLM, benchmark, local run mode, ROCm"
---

# Persistent server reuse (local) in Magpie's benchmark mode

Setting `server_lifecycle.enabled: true` with `run_mode: local` keeps one
detached inference server alive across successive `python -m Magpie benchmark`
runs on the same port, avoiding the model reload overhead on each invocation.
This is useful when running many client-only benchmark sweeps against the same
model and configuration.

```{note}
Persistent server reuse is a `run_mode: local` feature of Magpie's benchmark mode.
For a full overview of benchmark mode, including run modes, configuration, and output
structure, see [Benchmark frameworks with Magpie](benchmark.md).
```

## How it works

- `timeout_seconds` applies to the client subprocess (`benchmark_serving.py`)
  only — it does not stop the shared HTTP server afterward.
- `server_lifecycle.cleanup`: Magpie terminates the persisted process group
  (writes `SIGTERM`, then kills stragglers) only when `cleanup: true`; it also
  removes the associated `*.pid` and `*.json` artifacts under
  `~/.cache/magpie/server/` (or `server_lifecycle.pid_dir`).
- Compatibility gate: reuse checks JSON metadata versus `MODEL`, `TP`,
  `EXTRA_VLLM_ARGS`, `EXTRA_SGLANG_ARGS`, `MAX_MODEL_LEN`, InferenceX resolved
  path, framework, and `PORT`. Set `force_reuse: true` to bypass the mismatch
  errors.
- Scripts: requires Magpie built-in InferenceX wrappers that implement
  `MAGPIE_RUN_PHASE=server|client` (for example, `vllm_mi355x.sh`). Native
  InferenceX `gptoss_*` / `dsr1_*` scripts reject this flag path by design until
  they are updated upstream — point `benchmark_script` at one of the Magpie
  `*.sh` files.
- Profiling: torch profiler + `cleanup: false` is rejected (profiler state is
  tied to surviving workers). Configure `profiler.torch_profiler.enabled: false`
  for warmed servers, or set `cleanup: true`.

## GPU selection and server reuse

Before each run Magpie probes `http://127.0.0.1:$PORT/health` and compares
reuse metadata against the chosen config (`force_reuse: true` skips the
comparison). When the probe indicates the existing server should be reused
(eligible client-only path), `gpu_selection.auto` is skipped (`find_idle_gpus`
is not run), so visible-device env vars are not re-randomized relative to the
server's physical GPUs on the reuse chain. When the probe fails (cold start or
stale server after crash), idle-GPU selection runs as usual and Magpie launches a
new server phase. For `profiler.gpu_monitor` while reusing without auto-selection,
pin GPUs in `envs` or set `gpu_monitor.device_id` if you care which card is
sampled.

## Example

See `examples/benchmarks/benchmark_vllm_reuse.yaml` for a complete working
configuration.
