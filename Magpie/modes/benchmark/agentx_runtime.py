###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################
"""Own one AgentX server and its client, without invoking an upstream launcher."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Direct execution inside the serving image does not import Magpie's optional
# analysis/evaluation dependencies. Both files are mounted from the same build.
if __package__:
    from .agentx_custom import native_context_length
    from .agentx_launch import prepare_server_launch
    from .agentx_profile_config import (
        profile_plan,
        profile_server_spec,
        profile_settings,
    )
else:
    from agentx_custom import native_context_length
    from agentx_launch import prepare_server_launch
    from agentx_profile_config import (
        profile_plan,
        profile_server_spec,
        profile_settings,
    )


class RuntimeInterrupted(RuntimeError):
    """A signal interrupted this point; its owned children must be reaped."""


class _PublicModelRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlparse(newurl)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "huggingface.co"
            or parsed.port not in {None, 443}
            or parsed.username
            or parsed.password
        ):
            raise ValueError(
                "AgentX model metadata redirect left the public HuggingFace host"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _signal_interrupt(signum, _frame):
    raise RuntimeInterrupted(f"AgentX interrupted by signal {signum}")


def _stop_owned(process: subprocess.Popen) -> None:
    """Stop only the process group created by this runtime, including descendants."""
    if process.pid <= 1 or process.pid == os.getpgrp():
        raise RuntimeError("Refusing to terminate an unowned process group")
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # The root may exit before its workers, so leader exit alone is not cleanup.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def _spawn(
    argv: list[str], *, env: dict[str, str], cwd: Path, output
) -> subprocess.Popen:
    return subprocess.Popen(
        argv,
        env=env,
        cwd=cwd,
        stdout=output,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _run_owned(argv, *, env, cwd, output, timeout, server=None) -> int:
    process = _spawn(argv, env=env, cwd=cwd, output=output)
    deadline = time.monotonic() + timeout
    try:
        while process.poll() is None:
            if server is not None and server.poll() is not None:
                raise RuntimeError(
                    f"AgentX server exited while the client was running ({server.returncode})"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(f"AgentX command exceeded its {timeout:g}s deadline")
            time.sleep(0.1)
        if server is not None and server.poll() is not None:
            raise RuntimeError(
                f"AgentX server exited before the client completed ({server.returncode})"
            )
        return process.returncode
    finally:
        _stop_owned(process)


def _check_profiled_replay_result(workspace: Path, client_started_ns: int) -> None:
    """AIPerf can exit zero after cancellation; require its final completion flag."""
    result_path = workspace / "aiperf_artifacts" / "profile_export_aiperf.json"
    try:
        if result_path.stat().st_mtime_ns < client_started_ns:
            raise ValueError("AIPerf result belongs to an earlier client")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            not isinstance(result, dict)
            or type(result.get("was_cancelled")) is not bool
        ):
            raise ValueError("AIPerf result has no boolean was_cancelled flag")
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"AgentX cannot verify replay completion from {result_path}: {exc}"
        ) from exc
    if result["was_cancelled"]:
        raise RuntimeError("AIPerf replay was cancelled")


def _run_profiled_client(request, spec, *, server, output, outcome) -> int:
    """Bracket framework step profiling while the official replay remains unchanged."""
    if __package__:
        from .agentx_profiling import ReplayFinished, capture_profiles
    else:
        from agentx_profiling import ReplayFinished, capture_profiles

    workspace = Path(request["workspace"])
    root = Path(request["inferencex_path"])
    trace_dir = Path(spec["torch_profiler"]["trace_dir"])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        progress_port = probe.getsockname()[1]
    env = _client_environment(request, spec, root, workspace)
    env.update(
        AIPERF_API_SERVER_HOST="127.0.0.1", AIPERF_API_SERVER_PORT=str(progress_port)
    )
    plan = profile_plan(request["profile"], env)
    output.write("Magpie profiler plan: " + json.dumps(plan, sort_keys=True) + "\n")
    output.flush()
    started_ns = time.time_ns()
    deadline = time.monotonic() + request["client_timeout"]
    overall_deadline = (
        deadline
        + request["profile"]["capture_timeout_seconds"]
        + request["profile"]["flush_timeout_seconds"]
    )
    client = _spawn(_client_command(root), env=env, cwd=root, output=output)
    replay_verified = False

    def check_alive():
        nonlocal replay_verified
        if time.monotonic() >= overall_deadline:
            raise TimeoutError("AgentX profiling exceeded its overall deadline")
        if server.poll() is not None:
            raise RuntimeError("AgentX server exited during profiler capture")
        return_code = client.poll()
        if return_code not in (None, 0):
            raise RuntimeError(
                f"AgentX client failed during profiler capture ({client.returncode})"
            )
        if return_code == 0 and not replay_verified:
            _check_profiled_replay_result(workspace, started_ns)
            replay_verified = True

    def check_replay_alive():
        check_alive()
        if client.poll() == 0:
            raise ReplayFinished(
                "AgentX replay ended before all requested profiler captures completed"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError("AgentX client deadline reached before the next capture")

    try:
        outcome["profile_capture"] = capture_profiles(
            framework=spec["framework"],
            server_url=f"http://127.0.0.1:{spec['port']}",
            progress_url=f"http://127.0.0.1:{progress_port}/api/progress",
            trace_dir=trace_dir,
            settings=request["profile"],
            expected_ranks=request["profile_ranks"],
            phase_timeout_seconds=max(0.001, deadline - time.monotonic()),
            check_alive=check_alive,
            check_replay_alive=check_replay_alive,
            profile_plan=plan,
            client_started_ns=started_ns,
        )
        while client.poll() is None:
            check_alive()
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "AgentX client exceeded its deadline after profiler capture"
                )
            time.sleep(0.1)
        check_alive()
        return client.returncode
    finally:
        try:
            manifest = trace_dir / "capture.json"
            if manifest.is_file():
                outcome["profile_capture"] = json.loads(
                    manifest.read_text(encoding="utf-8")
                )
        finally:
            _stop_owned(client)


def _require_free_port(port: int) -> None:
    # Do not reuse an unrelated healthy endpoint. A new server owns every point.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(
                f"AgentX server port {port} is unavailable; existing servers are not reused"
            ) from exc


def _wait_health(server, url: str, deadline: float) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(
                f"AgentX server exited before becoming healthy ({server.returncode})"
            )
        try:
            with opener.open(
                url, timeout=min(2.0, max(0.1, deadline - time.monotonic()))
            ) as response:
                healthy = response.status == 200
        except (OSError, urllib.error.URLError):
            healthy = False
        if healthy:
            time.sleep(0.1)
            if server.poll() is not None:
                raise RuntimeError(
                    f"AgentX server exited during its health check ({server.returncode})"
                )
            return
        time.sleep(0.1)
    raise TimeoutError(
        "AgentX server did not become healthy before its startup deadline"
    )


def _verify_model_metadata(spec: dict[str, Any]) -> None:
    metadata = spec.get("model_metadata")
    if not metadata:
        return
    source = str(metadata.get("source") or "")
    if source.startswith("https://"):
        revision = str(metadata.get("revision") or "")
        parsed = urlparse(source)
        if (
            parsed.hostname != "huggingface.co"
            or parsed.port not in {None, 443}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or not re.fullmatch(r"[a-f0-9]{40}", revision)
            or not parsed.path.endswith(f"/resolve/{revision}/config.json")
        ):
            raise ValueError(
                "Remote AgentX metadata requires an immutable public HuggingFace revision"
            )
        argv = spec["argv"]
        if "--revision" not in argv or argv[argv.index("--revision") + 1 :][:1] != [
            revision
        ]:
            raise ValueError("AgentX server revision does not match its model metadata")
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _PublicModelRedirect()
        )
        try:
            with opener.open(source, timeout=30) as response:
                contents = response.read(4 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                raise ValueError(
                    "Gated AgentX models require an already downloaded local MODEL_PATH/config.json"
                ) from exc
            raise
    else:
        path = Path(source)
        if not path.is_absolute() or not path.is_file():
            raise ValueError(
                "AgentX model metadata requires its resolved local config.json"
            )
        contents = path.read_bytes()
    if len(contents) > 4 * 1024 * 1024:
        raise ValueError("AgentX model configuration exceeds the 4 MiB metadata limit")
    if hashlib.sha256(contents).hexdigest() != metadata.get("sha256"):
        raise ValueError("AgentX model metadata changed after recipe resolution")
    native = native_context_length(json.loads(contents))
    maximum = metadata.get("max_model_len")
    if (
        native != metadata.get("native_context_length")
        or type(maximum) is not int
        or not 0 < maximum <= native
    ):
        raise ValueError(
            "AgentX model context limits differ from the resolved metadata"
        )


def _verify_recipe_inputs(spec: dict[str, Any]) -> None:
    for name, expected in spec.get("source_files", {}).items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"AgentX recipe input changed before setup: {name}")


def _aiperf_git(root: Path, *arguments: str) -> str:
    path = root / "utils/aiperf"
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={path}", "-C", str(path), *arguments],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode:
        raise ValueError("AgentX requires the resolved AIPerf Git checkout")
    return completed.stdout.strip()


def _verify_aiperf_revision(root: Path, spec: dict[str, Any]) -> None:
    expected = spec.get("aiperf_revision")
    if expected is None:
        return
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError("AgentX AIPerf revision must be a complete Git commit")
    if (
        Path(_aiperf_git(root, "rev-parse", "--show-toplevel")).resolve()
        != (root / "utils/aiperf").resolve()
    ):
        raise ValueError("AgentX AIPerf submodule is not initialized")
    if _aiperf_git(root, "rev-parse", "HEAD") != expected:
        raise ValueError("AgentX AIPerf revision changed after recipe resolution")


def _validate_spec(spec: Any) -> dict[str, Any]:
    if (
        not isinstance(spec, dict)
        or type(spec.get("version")) is not int
        or spec["version"] != 1
    ):
        raise ValueError(
            "AgentX runtime requires a resolved server-launch-spec version 1"
        )
    if spec.get("framework") not in {"sglang", "vllm"}:
        raise ValueError("Magpie-owned AgentX supports SGLang and vLLM")
    setup = spec.get("setup_commands", [])
    if not isinstance(setup, list):
        raise ValueError("AgentX setup_commands must be a list of argv token lists")
    for argv in [spec.get("argv"), *setup]:
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(token, str) or "\0" in token for token in argv)
        ):
            raise ValueError(
                "AgentX server/setup commands must be literal argv token lists"
            )
    for key in ("env", "client_env"):
        mapping = spec.get(key, {})
        if not isinstance(mapping, dict) or any(
            not isinstance(name, str)
            or not isinstance(value, str)
            or "=" in name
            or "\0" in name + value
            for name, value in mapping.items()
        ):
            raise ValueError(
                f"AgentX {key} must contain literal string environment values"
            )
    port = spec.get("port")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("AgentX server port must be between 1 and 65535")
    health_path = spec.get("health_path", "/health")
    if (
        not isinstance(health_path, str)
        or not health_path.startswith("/")
        or health_path.startswith("//")
    ):
        raise ValueError("AgentX health_path must be a local absolute URL path")
    return spec


def _client_command(root: Path) -> list[str]:
    current = root / "benchmarks" / "srt_agentic.sh"
    if current.is_file():
        return ["bash", str(current)]
    if (root / "benchmarks" / "benchmark_lib.sh").is_file():
        adapter = Path(__file__).parents[2] / "scripts" / "agentx" / "client_legacy.sh"
        return ["bash", str(adapter)]
    raise ValueError("InferenceX project has no supported AgentX client entrypoint")


def _client_environment(request, spec, root, workspace) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AIPERF_", "AGENTIC_", "AGENTX_", "SRT_", "SRTCTL_"))
        and key not in {"CONC_LIST", "BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS"}
    }
    server_only = set(spec.get("env", {})) - {
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    }
    env.update(
        {
            key: value
            for key, value in request["client_env"].items()
            if key not in server_only
        }
    )
    env.update(spec.get("client_env", {}))
    # A candidate framework overlay must not change the measurement client.
    env["PYTHONPATH"] = str(root)
    env.pop("PYTHONHOME", None)
    env.update(
        {
            "INFMAX_CONTAINER_WORKSPACE": str(root),
            "RESULT_DIR": str(workspace),
            "AGENTIC_OUTPUT_DIR": str(workspace),
            "RESULT_FILENAME": "inferencex_result",
            "SERVER_LOG": str(workspace / "server.log"),
            "AIPERF_RUNTIME_DIR": str(workspace / ".agentx-runtime"),
            "AIPERF_SERVER_URL": f"http://127.0.0.1:{spec['port']}",
            "AIPERF_SERVER_METRICS_URLS": f"http://127.0.0.1:{spec['port']}/metrics",
            "PORT": str(spec["port"]),
            "IS_MULTINODE": "false",
            "IS_AGENTIC": "1",
            "SCENARIO_TYPE": "agentic-coding",
            "EVAL_ONLY": "false",
            "RUN_EVAL": "false",
        }
    )
    # Client adapters must not alter or delete the server receipt.
    for name in tuple(env):
        if name.startswith(("AGENTX_LAUNCH_", "SRT_", "SRTCTL_")) or name in {
            "AGENTX_SERVER_LAUNCH_FILE",
            "CONC_LIST",
        }:
            env.pop(name)
    return env


def _preflight_client_environment(env: dict[str, str]) -> None:
    """Validate the pinned native client's single-node replay contract.

    These are the check_env_vars inputs in srt_agentic.sh and its dependency,
    replay, and power helpers at InferenceX 408c015. Keep this contract in sync
    when changing the client pin; do not execute the upstream shell to check it.
    """
    required = {
        "RESULT_DIR",
        "EVAL_ONLY",
        "MODEL",
        "MODEL_PREFIX",
        "FRAMEWORK",
        "PRECISION",
        "CONC",
        "RESULT_FILENAME",
        "DURATION",
        "PORT",
        "INFMAX_CONTAINER_WORKSPACE",
        "AIPERF_PYTHON_VERSION",
        "AIPERF_FAILED_REQUEST_THRESHOLD",
        "AIPERF_LIVE_FAILED_REQUEST_THRESHOLD",
        "AIPERF_TRACE_IDLE_GAP_CAP_SECONDS",
        "AGENTIC_WARMUP_GRACE_PERIOD",
        "AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES",
        "AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS",
        "AIPERF_EXPERIMENTAL_FAST",
        "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID",
        "AIPERF_UNSAFE_OVERRIDE",
        "AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING",
        "AIPERF_WARMUP_REQUESTS_PER_LANE",
        "ENABLE_AGENTX_POWER",
        "IS_MULTINODE",
        "REQUIRE_POWER",
    }
    if env.get("ENABLE_AGENTX_POWER", "").lower() in {"1", "true", "yes"}:
        required.update({"TP", "PP_SIZE", "PCP_SIZE"})
    missing = sorted(name for name in required if not env.get(name))
    if missing:
        raise ValueError(
            "AgentX client environment is missing required values before server startup: "
            + ", ".join(missing)
        )


def _execute_local(request: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    workspace = Path(request["workspace"])
    root = Path(request["inferencex_path"])
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "agentx_server_launch.json").unlink(missing_ok=True)
    server = None
    outcome = {"success": False, "errors": [], "execution_time": 0.0}
    try:
        spec = _validate_spec(request["server_spec"])
        _verify_recipe_inputs(spec)
        _verify_aiperf_revision(root, spec)
        command = _client_command(root)
        if "recipe_variant" in spec and Path(command[-1]).name == "srt_agentic.sh":
            _preflight_client_environment(
                _client_environment(request, spec, root, workspace)
            )
        if request.get("profile"):
            # Reserve this invocation before startup: frameworks may write
            # CUDA-graph warmup traces while the service is coming online.
            Path(spec["torch_profiler"]["trace_dir"]).mkdir(
                parents=True, exist_ok=False
            )
        _require_free_port(spec["port"])
        deadline = time.monotonic() + request["ready_timeout"]
        # Acceptance simulation must come from the resolved recipe, including
        # when this recipe intentionally has no simulation settings.
        server_env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("SGLANG_SIMULATE_ACC_")
        }
        server_env.update(request.get("gpu_env", {}))
        server_env.update(spec.get("env", {}))
        for key in ("BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS"):
            server_env.pop(key, None)
        with (workspace / "agentx_setup.log").open("w", encoding="utf-8") as setup_log:
            for setup in spec.get("setup_commands", []):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "AgentX preparation exceeded its startup deadline"
                    )
                rc = _run_owned(
                    setup, env=server_env, cwd=root, output=setup_log, timeout=remaining
                )
                if rc:
                    raise RuntimeError(
                        f"AgentX server preparation failed with exit code {rc}"
                    )
        _verify_aiperf_revision(root, spec)
        _verify_model_metadata(spec)
        argv, server_env, _evidence = prepare_server_launch(
            spec["argv"],
            server_env,
            request.get("overrides"),
            spec["framework"],
            workspace,
            server_spec=spec,
        )
        with (workspace / "server.log").open("w", encoding="utf-8") as server_log:
            server = _spawn(argv, env=server_env, cwd=root, output=server_log)
            _wait_health(
                server,
                f"http://127.0.0.1:{spec['port']}{spec.get('health_path', '/health')}",
                deadline,
            )
            with (workspace / "agentx_client.log").open(
                "w", encoding="utf-8"
            ) as client_log:
                if request.get("profile"):
                    rc = _run_profiled_client(
                        request, spec, server=server, output=client_log, outcome=outcome
                    )
                else:
                    rc = _run_owned(
                        command,
                        env=_client_environment(request, spec, root, workspace),
                        cwd=root,
                        output=client_log,
                        timeout=request["client_timeout"],
                        server=server,
                    )
            if rc:
                raise RuntimeError(
                    f"AgentX benchmark client failed with exit code {rc}"
                )
        outcome["success"] = True
    except (
        OSError,
        ValueError,
        RuntimeError,
        TimeoutError,
        subprocess.SubprocessError,
        http.client.HTTPException,
    ) as exc:
        outcome["errors"].append(str(exc))
    finally:
        if server is not None:
            try:
                _stop_owned(server)
            except (OSError, subprocess.SubprocessError) as exc:
                outcome["success"] = False
                outcome["errors"].append(f"AgentX server cleanup failed: {exc}")
        outcome["execution_time"] = time.monotonic() - started
    return outcome


def _docker_command(
    request, *, image: str, name: str, request_path: Path, runner_type: str
) -> list[str]:
    command = [
        "docker",
        "run",
        "--init",
        "--name",
        name,
        "--network=host",
        "--ipc=host",
        "--shm-size=16g",
    ]
    if runner_type.lower().startswith(("mi", "gfx", "radeon")):
        command += [
            "--device=/dev/kfd",
            "--device=/dev/dri",
            "--group-add=video",
            "--cap-add=SYS_PTRACE",
            "--security-opt=seccomp=unconfined",
        ]
    else:
        command += ["--gpus=all"]
    package = Path(__file__).resolve().parents[2]
    mounts = {
        str(package): "ro",
        request["inferencex_path"]: "rw",
        request["workspace"]: "rw",
    }
    if request["server_spec"].get("aiperf_revision") is not None:
        root = Path(request["inferencex_path"])
        _verify_aiperf_revision(root, request["server_spec"])
        # A submodule's .git file points outside the mounted project directory.
        # Preserve that exact path without exposing the parent checkout.
        git_directory = Path(_aiperf_git(root, "rev-parse", "--absolute-git-dir"))
        mounts[str(git_directory.resolve())] = "ro"
    model_path = request["client_env"].get(
        "MODEL_PATH", request["client_env"].get("MODEL", "")
    )
    if model_path and Path(model_path).is_dir():
        mounts[str(Path(model_path).resolve())] = "ro"
    cache = request.get("hf_cache_path")
    if cache and Path(cache).is_dir():
        mounts[str(Path(cache).resolve())] = "rw"
        command += ["--env", f"HF_HOME={Path(cache).resolve()}"]
    for source, mode in mounts.items():
        if ":" in source or "\n" in source:
            raise ValueError(
                "AgentX Docker mount paths cannot contain ':' or a newline"
            )
        command += ["--volume", f"{source}:{source}:{mode}"]
    for key, value in request.get("gpu_env", {}).items():
        command += ["--env", f"{key}={value}"]
    if os.environ.get("HF_TOKEN"):
        command += ["--env", "HF_TOKEN"]
    command += [
        "--entrypoint",
        "python3",
        "--",
        image,
        str(Path(__file__).resolve()),
        "--worker",
        str(request_path),
    ]
    return command


def _execute_docker(request, *, image: str, runner_type: str) -> dict[str, Any]:
    workspace = Path(request["workspace"])
    request_path = workspace / "agentx_runtime_request.json"
    result_path = workspace / "agentx_runtime_result.json"
    result_path.unlink(missing_ok=True)
    request_path.write_text(json.dumps(request), encoding="utf-8")
    name = f"magpie-agentx-{uuid.uuid4().hex}"
    outcome = {"success": False, "errors": []}
    started = time.monotonic()
    try:
        command = _docker_command(
            request,
            image=image,
            name=name,
            request_path=request_path,
            runner_type=runner_type,
        )
        with (workspace / "agentx_container.log").open("w", encoding="utf-8") as output:
            rc = _run_owned(
                command,
                env=dict(os.environ),
                cwd=workspace,
                output=output,
                timeout=(
                    request["ready_timeout"]
                    + request["client_timeout"]
                    + 60
                    + (request.get("profile") or {}).get("capture_timeout_seconds", 0)
                    + (request.get("profile") or {}).get("flush_timeout_seconds", 0)
                ),
            )
        if result_path.is_file():
            outcome = json.loads(result_path.read_text(encoding="utf-8"))
        if rc:
            outcome["success"] = False
            outcome["errors"].append(f"AgentX container exited with code {rc}")
        elif not result_path.is_file():
            outcome["errors"].append("AgentX container produced no runtime result")
    except (
        OSError,
        ValueError,
        RuntimeError,
        TimeoutError,
        subprocess.SubprocessError,
    ) as exc:
        outcome["errors"].append(str(exc))
    finally:
        try:
            removed = subprocess.run(
                ["docker", "rm", "--force", name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if removed.returncode and "No such container" not in removed.stderr:
                raise RuntimeError(removed.stderr.strip())
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            outcome["success"] = False
            outcome["errors"].append(f"AgentX container cleanup failed: {exc}")
    outcome["execution_time"] = time.monotonic() - started
    return outcome


def execute_agentx(
    config, workspace: Path, runner_type: str, *, docker_image: str | None = None
):
    """Run one fresh server/client point and return Magpie's normal execution tuple."""
    from .result import BenchmarkResult

    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    resolved = config.agentx.resolved if config.agentx else {}
    spec = _validate_spec(resolved.get("server-launch-spec"))
    if spec["framework"] != config.framework:
        raise ValueError("AgentX server framework differs from the benchmark")
    lifecycle = config.server_lifecycle
    request = {
        "workspace": str(workspace),
        "inferencex_path": str(Path(config.inferencex_path).resolve()),
        "server_spec": spec,
        "overrides": config.agentx.launch_overrides,
        "client_env": config.get_env_vars(),
        "hf_cache_path": config.hf_cache_path
        or os.environ.get("HF_HOME")
        or str(Path.home() / ".cache/huggingface"),
        "ready_timeout": float(
            lifecycle.server_ready_timeout_s
            if lifecycle
            else spec.get("health_timeout_seconds", 2700)
        ),
        "client_timeout": float(config.timeout_seconds),
        "gpu_env": {
            key: str(value)
            for key, value in config.envs.items()
            if key
            in {"ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"}
        },
    }
    settings = profile_settings(config)
    if settings is not None:
        request["profile"] = settings
        request["server_spec"] = profile_server_spec(
            spec, settings, workspace, uuid.uuid4().hex
        )
        ranks = 1
        for entry_name, env_name in (
            ("tp", "TP"),
            ("pp", "PP_SIZE"),
            ("pcp-size", "PCP_SIZE"),
        ):
            value = resolved.get(entry_name, config.envs.get(env_name, 1))
            if (
                not isinstance(value, (str, int))
                or isinstance(value, bool)
                or int(value) <= 0
            ):
                raise ValueError(
                    "AgentX profiling requires positive resolved GPU topology"
                )
            ranks *= int(value)
        request["profile_ranks"] = ranks
    if any(
        not math.isfinite(request[key]) or request[key] <= 0
        for key in ("ready_timeout", "client_timeout")
    ):
        raise ValueError(
            "AgentX startup and client timeouts must be positive finite durations"
        )
    if config.run_mode == "docker":
        image = docker_image or config.docker_image
        if not image:
            raise ValueError("AgentX Docker execution requires a pinned image")
        outcome = _execute_docker(request, image=image, runner_type=runner_type)
    elif config.run_mode == "local":
        outcome = _execute_local(request)
    else:
        raise ValueError("Magpie-owned AgentX supports local and Docker execution")
    result = BenchmarkResult(
        success=bool(outcome["success"]),
        framework=config.framework,
        model=config.model,
        scenario="agentx",
        workspace_dir=str(workspace),
        execution_time=outcome["execution_time"],
        errors=outcome["errors"],
    )
    if settings is not None:
        result.profiling_enabled = True
        result.benchmark_valid = False
        result.publishable = False
        result.agentx_metrics = {
            "diagnostic_only": True,
            "profile_capture": outcome.get("profile_capture", {"status": "failed"}),
        }
    log = workspace / "agentx_client.log"
    stdout = (
        log.read_text(encoding="utf-8", errors="replace")[-131072:]
        if log.is_file()
        else ""
    )
    return result, stdout, "\n".join(outcome["errors"])


def _worker_main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.worker.read_text(encoding="utf-8"))
    signal.signal(signal.SIGTERM, _signal_interrupt)
    signal.signal(signal.SIGINT, _signal_interrupt)
    outcome = _execute_local(request)
    (Path(request["workspace"]) / "agentx_runtime_result.json").write_text(
        json.dumps(outcome), encoding="utf-8"
    )
    return 0 if outcome["success"] else 1


if __name__ == "__main__":
    raise SystemExit(_worker_main())
