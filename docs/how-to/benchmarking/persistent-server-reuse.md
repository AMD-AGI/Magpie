---
myst:
    html_meta:
        "description": "Keep a Magpie inference server alive across successive benchmark runs using server_lifecycle.enabled to avoid model reload overhead between client invocations."
        "keywords": "Magpie, persistent server, Docker server reuse, server_lifecycle, vLLM, SGLang, benchmark, ROCm"
---

# Persistent server reuse in Magpie's benchmark mode

Setting `server_lifecycle.enabled: true` with `run_mode: local` or
`run_mode: docker` keeps one inference server alive across successive
`python -m Magpie benchmark` runs on the same host and port. This avoids model
reload overhead when running multiple client workloads against one server
configuration.

```{note}
For a full overview of benchmark mode, including run modes, configuration, and
output structure, see [Benchmark frameworks with Magpie](benchmark.md).
```

## How it works

- `timeout_seconds` applies to the client subprocess (`benchmark_serving.py`)
  only — it does not stop the shared HTTP server afterward.
- Local mode persists a server process group. Docker mode persists a detached,
  Magpie-named container and runs each benchmark client in a short-lived
  container using the same image.
- `server_lifecycle.cleanup`: Magpie stops the persisted local process group or
  removes the recorded Docker container only when `cleanup: true`. It also
  removes the associated state under `~/.cache/magpie/server/` (or
  `server_lifecycle.pid_dir`).
- Compatibility gate: reuse checks JSON metadata versus `MODEL`, `TP`,
  `EXTRA_VLLM_ARGS`, `EXTRA_SGLANG_ARGS`, `MAX_MODEL_LEN`, InferenceX resolved
  path, framework, and `PORT`. Docker mode additionally checks the image. Set
  `force_reuse: true` to bypass configuration mismatch errors. Docker mode still
  requires the server to be owned by a recorded Magpie container.
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
is not run). In Docker mode Magpie records and restores the first run's visible
device mapping for subsequent client containers. When the probe fails (cold
start or stale server after crash), idle-GPU selection runs as usual and Magpie
launches a new server phase. For `profiler.gpu_monitor` while reusing without
auto-selection, pin GPUs in `envs` or set `gpu_monitor.device_id` if you care
which card is sampled.

## Docker lifecycle

For `run_mode: docker`, the first invocation starts a detached server container
and waits for its health endpoint. It then starts a separate client container.
Later invocations with compatible server settings skip server startup and run
only the client container. Client-only values such as `CONC`, `ISL`, `OSL`, and
`NUM_PROMPTS` may change between invocations; server-side settings must remain
identical.

Set `cleanup: false` on intermediate runs and `cleanup: true` on the final run.
The final run captures the server container output as
`reuse_server_container.log` before removing the container. The application
server log remains in the workspace in which the server was first started.

```yaml
benchmark:
  framework: sglang
  model: Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
  run_mode: docker
  docker_image: your-image:version
  benchmark_script: sglang_mi355x.sh
  envs:
    TP: 4
    PORT: 8888
    CONC: 1
    ISL: 1024
    OSL: 1024
    RANDOM_RANGE_RATIO: 1
    EXTRA_SGLANG_ARGS: "--context-length 131072"
  profiler:
    torch_profiler:
      enabled: false
  server_lifecycle:
    enabled: true
    cleanup: false
    server_ready_timeout_s: 2700
```

Change `CONC` for the next invocation while retaining the same server-side
fields. On the final invocation, change only `server_lifecycle.cleanup` to
`true`.

## Example

See `examples/benchmarks/benchmark_vllm_reuse.yaml` for a local example and
`examples/benchmarks/benchmark_sglang_docker_reuse.yaml` for a Docker example.
