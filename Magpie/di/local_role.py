"""Node-local SGLang role lifecycle for a controller-assigned instance.

This module never allocates nodes or executes on another host. The enrolled worker
must validate placement/GPU ownership before constructing a role. A successful
router probe establishes a routed request, not all-path xPyD or RDMA qualification.
Readiness uses /health; SGLang validates its launch arguments. This module does
not require a particular /get_server_info response schema.

Router argv follows SGLang v0.5.18's router_args.py: repeat --prefill URL [PORT]
and --decode URL for each instance. Other router versions must be qualified by
callers; list construction alone is not a real multi-instance qualification.
Upstream contract: https://github.com/sgl-project/sglang/blob/v0.5.18/
sgl-model-gateway/bindings/python/src/sglang_router/router_args.py
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlsplit

_LABEL = "magpie.di.local-owner"
_SPEC_LABEL = "magpie.di.local-spec"
_DOCKER = ["docker", "--host", "unix:///var/run/docker.sock"]
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_IMAGE = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _port(value):
    if type(value) is not int or not 0 < value < 65536:
        raise ValueError("ports must be integers in 1..65535")
    return value


def _path(value):
    if not isinstance(value, str) or not value.startswith("/") or any(c in value for c in ",\x00\n\r"):
        raise ValueError("prepared paths must be absolute and cannot contain commas or control characters")
    return value


def validate_local_spec(spec: dict) -> dict:
    """Reject ambiguous placement and topology overrides before side effects."""
    if not isinstance(spec, dict):
        raise TypeError("role spec must be an object")
    s = deepcopy(spec)
    allowed = {
        "run_id",
        "attempt_id",
        "group_id",
        "round",
        "instance_id",
        "role",
        "image",
        "model_path",
        "served_model",
        "host",
        "port",
        "gpus",
        "tp",
        "nic",
        "env",
        "server_args",
        "mounts",
        "startup_timeout",
        "bootstrap_port",
        "prefill",
        "decode",
    }
    if set(s) - allowed:
        raise ValueError(f"unsupported local role options: {sorted(set(s) - allowed)}")
    if "round" in s and (type(s["round"]) is not int or s["round"] < 0):
        raise ValueError("round must be a nonnegative integer")
    for key in ("run_id", "attempt_id", "instance_id", *(("group_id",) if "group_id" in s else ())):
        if key == "attempt_id" and type(s.get(key)) is int:
            s[key] = str(s[key])
        if not isinstance(s.get(key), str) or not _IDENTIFIER.fullmatch(s[key]):
            raise ValueError(f"invalid {key}")
    if s.get("role") not in {"prefill", "decode", "router"}:
        raise ValueError("role must be prefill, decode or router")
    if not isinstance(s.get("image"), str) or not _IMAGE.fullmatch(s["image"]):
        raise ValueError("image must be an immutable local sha256 image ID")
    _path(s.get("model_path"))
    if (
        not isinstance(s.get("served_model"), str)
        or not s["served_model"]
        or any(c in s["served_model"] for c in "\x00\r\n")
    ):
        raise ValueError("served_model is required")
    if not isinstance(s.get("host"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", s["host"]):
        raise ValueError("host must be an advertised hostname or IPv4 address")
    _port(s.get("port"))
    timeout = s.setdefault("startup_timeout", 5400)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("startup_timeout must be finite and positive")
    env = s.setdefault("env", {})
    if not isinstance(env, dict) or any(
        not isinstance(k, str)
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k)
        or not isinstance(v, (str, int, float))
        or "\x00" in str(v)
        for k, v in env.items()
    ):
        raise ValueError("env must contain valid environment keys and scalar values")
    if any(k in env for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")):
        raise ValueError("GPU visibility is owned by the assignment")
    if str(env.get("MC_FORCE_TCP", "0")) not in ("0", "", "false", "False"):
        raise ValueError("TCP override conflicts with RDMA")
    if env.get("MOONCAKE_PROTOCOL", "rdma") != "rdma":
        raise ValueError("Mooncake protocol must be rdma")
    env.pop("MC_FORCE_TCP", None)
    env["MOONCAKE_PROTOCOL"] = "rdma"
    args = s.setdefault("server_args", [])
    if not isinstance(args, list) or any(not isinstance(x, str) or "\x00" in x for x in args):
        raise ValueError("server_args must be an argv list")
    reserved = {
        "--model-path",
        "--model",
        "--served-model-name",
        "--host",
        "--port",
        "--tp",
        "--tp-size",
        "--tensor-parallel-size",
        "--nnodes",
        "--node-rank",
        "--dist-init-addr",
        "--nccl-init-addr",
        "--config",
        "--use-ray",
    }
    if any(
        x.startswith("--")
        and (
            any(flag.startswith(x.split("=", 1)[0]) for flag in reserved)
            or x.startswith("--disaggregation-")
            or "--disaggregation-".startswith(x.split("=", 1)[0])
        )
        for x in args
    ):
        raise ValueError("server_args cannot override the local role topology")
    mounts = s.setdefault("mounts", [])
    if not isinstance(mounts, list):
        raise TypeError("mounts must be a list of readonly bind objects")
    destinations = {"/model"}
    for mount in mounts:
        if not isinstance(mount, dict) or set(mount) != {"source", "target"}:
            raise ValueError("mounts require source and target; all mounts are readonly")
        source, target = _path(mount["source"]), _path(mount["target"])
        if (
            target in destinations
            or target in {"/", "/proc", "/sys", "/dev", "/run", "/var/run"}
            or "docker.sock" in source
            or "docker.sock" in target
        ):
            raise ValueError("unsafe or duplicate mount target")
        destinations.add(target)
    if s["role"] == "router":
        if any(k in s for k in ("gpus", "tp", "nic", "bootstrap_port")) or args:
            raise ValueError("router does not accept serving role options")
        for role in ("prefill", "decode"):
            entries = s.get(role)
            if not isinstance(entries, list) or not entries:
                raise ValueError("router needs nonempty prefill and decode lists")
            seen = set()
            for entry in entries:
                if not isinstance(entry, dict) or set(entry) - (
                    {"url", "bootstrap_port"} if role == "prefill" else {"url"}
                ):
                    raise ValueError("invalid router role entry")
                url = entry.get("url")
                if not isinstance(url, str):
                    raise TypeError("role URL is required")
                parsed = urlsplit(url)
                if (
                    parsed.scheme != "http"
                    or not parsed.hostname
                    or parsed.username
                    or parsed.password
                    or parsed.path
                    or parsed.query
                    or parsed.fragment
                ):
                    raise ValueError("role URL must be http://host:port without credentials or a path")
                _port(parsed.port)
                if url in seen:
                    raise ValueError("duplicate router endpoint")
                seen.add(url)
                if "bootstrap_port" in entry:
                    _port(entry["bootstrap_port"])
    else:
        if "prefill" in s or "decode" in s:
            raise ValueError("serving role cannot own other role endpoints")
        gpus = s.get("gpus")
        if (
            not isinstance(gpus, list)
            or not gpus
            or any(type(g) is not int or g < 0 for g in gpus)
            or len(set(gpus)) != len(gpus)
        ):
            raise ValueError("gpus must be distinct nonnegative integer IDs")
        s.setdefault("tp", len(gpus))
        if type(s["tp"]) is not int or s["tp"] != len(gpus):
            raise ValueError("one role must fit its assigned local GPUs")
        if not isinstance(s.get("nic"), str) or not re.fullmatch(r"[A-Za-z0-9_,.:+-]+", s["nic"]):
            raise ValueError("nic is required")
        _port(s.setdefault("bootstrap_port", 8998))
        if s["bootstrap_port"] == s["port"]:
            raise ValueError("role and bootstrap ports must differ")
    return s


class LocalRole:
    """One durable local role. Replaying the same spec never creates another owner."""

    def __init__(
        self,
        spec: dict,
        state_dir: Path,
        check_cancelled: Callable[[], None] | None = None,
    ):
        self.spec = validate_local_spec(spec)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.state_dir / "local-role.json"
        self.check_cancelled = check_cancelled or (lambda: None)
        identity = [self.spec[k] for k in ("run_id", "attempt_id", "instance_id")]
        if "group_id" in self.spec:
            identity.append(self.spec["group_id"])
        if "round" in self.spec:
            identity.append(self.spec["round"])
        self.owner = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        self.spec_hash = hashlib.sha256(
            json.dumps(self.spec, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.name = f"magpie-di-{self.owner[:32]}-{self.spec['role']}"
        self.endpoint = f"http://{self.spec['host']}:{self.spec['port']}"
        with self._locked():
            if self.path.exists():
                self._reload()
            else:
                self.state = {
                    "version": 1,
                    "spec": self.spec,
                    "owner": self.owner,
                    "spec_hash": self.spec_hash,
                    "container_name": self.name,
                    "endpoint": self.endpoint,
                    "status": "created",
                    "cleanup_verified": False,
                }
                self._save()

    @classmethod
    def load(cls, state_dir: Path, check_cancelled=None):
        path = Path(state_dir) / "local-role.json"
        return cls(json.loads(path.read_text())["spec"], state_dir, check_cancelled)

    @contextmanager
    def _locked(self):
        with (self.state_dir / "local-role.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _reload(self):
        self.state = json.loads(self.path.read_text())
        if (
            self.state.get("owner") != self.owner
            or self.state.get("spec_hash") != self.spec_hash
            or self.state.get("spec") != self.spec
            or self.state.get("container_name") != self.name
        ):
            raise ValueError("local role ownership/spec changed; use a new attempt and state directory")

    def _save(self):
        fd, temporary = tempfile.mkstemp(prefix=".local-role-", dir=self.state_dir)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(self.state, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.state_dir, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _run(self, argv, *, timeout=60, cancellable=True):
        # A Docker context or DOCKER_HOST inherited by the worker must not turn
        # this local primitive into remote execution.
        process = subprocess.Popen([*_DOCKER, *argv], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + timeout
        try:
            while True:
                if cancellable:
                    self.check_cancelled()
                try:
                    out, err = process.communicate(timeout=min(1, max(0.01, deadline - time.monotonic())))
                    return process.returncode, out, err
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"local Docker {argv[0]} timed out")
        except BaseException:
            process.kill()
            process.communicate()
            raise

    def command(self):
        s = self.spec
        cmd = [
            "run",
            "--detach",
            "--name",
            self.name,
            "--label",
            f"{_LABEL}={self.owner}",
            "--label",
            f"{_SPEC_LABEL}={self.spec_hash}",
            "--network",
            "host",
            "--ipc",
            "host",
            "--log-opt",
            "max-size=512m",
            "--log-opt",
            "max-file=2",
            "--mount",
            f"type=bind,src={s['model_path']},dst=/model,readonly",
        ]
        env = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", **s["env"]}
        for m in s["mounts"]:
            cmd += [
                "--mount",
                f"type=bind,src={m['source']},dst={m['target']},readonly",
            ]
        if s["role"] != "router":
            cmd += [
                "--shm-size",
                "64g",
                "--ulimit",
                "memlock=-1",
                "--ulimit",
                "stack=67108864",
                "--group-add",
                "video",
            ]
            for device in ("/dev/kfd", "/dev/dri", "/dev/infiniband"):
                cmd += ["--device", device]
            visible = ",".join(map(str, s["gpus"]))
            env.update(CUDA_VISIBLE_DEVICES=visible, HIP_VISIBLE_DEVICES=visible)
        for key, value in sorted(env.items()):
            cmd += ["--env", f"{key}={value}"]
        cmd += ["--entrypoint", "python3", s["image"]]
        if s["role"] == "router":
            cmd += ["-m", "sglang_router.launch_router", "--pd-disaggregation"]
            for entry in s["prefill"]:
                cmd += ["--prefill", entry["url"]]
                if "bootstrap_port" in entry:
                    cmd.append(str(entry["bootstrap_port"]))
            for entry in s["decode"]:
                cmd += ["--decode", entry["url"]]
            cmd += [
                "--host",
                "0.0.0.0",
                "--port",
                str(s["port"]),
                "--tokenizer-path",
                "/model",
            ]
        else:
            cmd += [
                "-m",
                "sglang.launch_server",
                *s["server_args"],
                "--model-path",
                "/model",
                "--served-model-name",
                s["served_model"],
                "--tensor-parallel-size",
                str(s["tp"]),
                "--host",
                "0.0.0.0",
                "--port",
                str(s["port"]),
                "--disaggregation-mode",
                s["role"],
                "--disaggregation-transfer-backend",
                "mooncake",
                "--disaggregation-bootstrap-port",
                str(s["bootstrap_port"]),
                "--disaggregation-ib-device",
                s["nic"],
            ]
        return cmd

    def _inspect(self, target=None, *, cancellable=False, timeout=60):
        target = target or self.name
        rc, out, err = self._run(["container", "inspect", target], cancellable=cancellable, timeout=timeout)
        if rc:
            # A daemon/permission/transport error is never evidence of absence.
            missing = (
                f"Error: No such container: {target}",
                f"Error response from daemon: No such container: {target}",
                f"Error: No such object: {target}",
            )
            if err.strip() in missing and out.strip() in ("", "[]"):
                return None
            raise RuntimeError(f"cannot inspect owned local role: {err[-1000:]}")
        values = json.loads(out)
        if not isinstance(values, list) or len(values) != 1:
            raise RuntimeError("invalid container inspection")
        item = values[0]
        labels = item.get("Config", {}).get("Labels", {}) or {}
        if (
            labels.get(_LABEL) != self.owner
            or labels.get(_SPEC_LABEL) != self.spec_hash
            or item.get("Image") != self.spec["image"]
        ):
            raise RuntimeError("container ownership or immutable image mismatch")
        if not re.fullmatch(r"[0-9a-f]{64}", item.get("Id", "")):
            raise RuntimeError("invalid owned container ID")
        return item

    def inspect(self, *, cancellable=False):
        with self._locked():
            self._reload()
            item = self._inspect(cancellable=cancellable)
            return {
                **deepcopy(self.state),
                "container": item,
                "present": item is not None,
            }

    def _ports_free(self):
        ports = [self.spec["port"]]
        if self.spec["role"] != "router":
            ports.append(self.spec["bootstrap_port"])
        for port in ports:
            with socket.socket() as sock:
                # Match server rebinding after shutdown: TIME_WAIT is not a live listener.
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(("0.0.0.0", port))
                except OSError as exc:
                    raise OSError(exc.errno, f"local {self.spec['role']} port {port} unavailable: {exc.strerror}") from exc

    def launch(self):
        with self._locked():
            self._reload()
            self.check_cancelled()
            if self.state["status"] in {"stopped", "stopping", "cleanup_failed"}:
                raise RuntimeError("a stopped/cleaning attempt cannot be relaunched")
            item = self._inspect()
            if item:
                if not item.get("State", {}).get("Running"):
                    raise RuntimeError("owned role exited; cleanup is required")
                self.state["container_id"] = item["Id"]
                self.state["launch_uncertain"] = False
                self.state["status"] = "ready" if self.state["status"] == "ready" else "running"
                self._save()
                return deepcopy(self.state)
            if self.state.get("launch_uncertain"):
                raise RuntimeError("launch outcome is unresolved; retain the assignment for reconciliation")
            if self.state["status"] in {"running", "ready"}:
                raise RuntimeError("previously running role disappeared; start a new attempt")
            rc, out, err = self._run(["image", "inspect", self.spec["image"], "--format", "{{.Id}}"])
            if rc or out.strip() != self.spec["image"]:
                raise RuntimeError(f"prepared immutable image is unavailable: {err[-1000:]}")
            self._ports_free()
            self.state.update(status="launching", cleanup_verified=False, launch_uncertain=True)
            self._save()  # Durable owner precedes the first container mutation.
            rc, _, err = self._run(self.command())
            if rc:
                # A failed CLI reply does not prove the daemon rejected creation.
                raise RuntimeError(f"local role launch failed; cleanup required: {err[-1000:]}")
            self.state["launch_uncertain"] = False
            self._save()
            item = self._inspect(timeout=2)
            if not item or not item.get("State", {}).get("Running"):
                raise RuntimeError("role did not remain running after launch")
            self.state.update(status="running", container_id=item["Id"])
            self._save()
            return deepcopy(self.state)

    def _http(self, path, data=None, timeout=5):
        url = f"http://127.0.0.1:{self.spec['port']}{path}"
        request = urllib.request.Request(
            url,
            data=json.dumps(data).encode() if data is not None else None,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout) as response:
            raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise RuntimeError("readiness response exceeds 1 MiB")
            if path == "/health":
                return {}  # Health responses may be plain text or empty.
            return json.loads(raw) if raw.strip() else {}

    def _probe(self, timeout=180):
        deadline = time.monotonic() + timeout
        done, result = threading.Event(), {}

        def request():
            try:
                result["value"] = self._http(
                    "/v1/completions",
                    {
                        "model": self.spec["served_model"],
                        "prompt": "The capital of France is",
                        "max_tokens": 8,
                        "temperature": 0,
                    },
                    timeout=timeout,
                )
            except Exception as exc:  # noqa: BLE001 - propagate the original thread failure
                result["error"] = exc
            finally:
                done.set()

        threading.Thread(target=request, daemon=True).start()
        while not done.wait(max(0, min(1, deadline - time.monotonic()))):
            self.check_cancelled()
            if time.monotonic() >= deadline:
                raise TimeoutError("routed probe timed out")
        self.check_cancelled()
        if "error" in result:
            raise result["error"]
        value = result["value"]
        if (
            not isinstance(value, dict)
            or not value.get("choices")
            or value.get("usage", {}).get("completion_tokens", 0) <= 0
        ):
            raise RuntimeError("router probe did not produce completion tokens")
        return value

    def wait_ready(self):
        deadline = time.monotonic() + self.spec["startup_timeout"]
        last_error = "no readiness response"
        while time.monotonic() < deadline:
            self.check_cancelled()
            observation = self.inspect(cancellable=True)
            item = observation["container"]
            if (
                not item
                or not item.get("State", {}).get("Running")
                or observation["status"] in {"stopped", "stopping", "cleanup_failed"}
            ):
                raise RuntimeError("role stopped before readiness")
            try:
                self._http(
                    "/health",
                    timeout=min(1, max(0.01, deadline - time.monotonic())),
                )
                evidence = {}
                if self.spec["role"] == "router":
                    evidence = {
                        "routed_ready": True,
                        "probe": self._probe(timeout=min(180, max(0.01, deadline - time.monotonic()))),
                    }
            except urllib.error.HTTPError as exc:
                if exc.code != 503:
                    raise
                last_error = str(exc)
                exc.close()
                self.check_cancelled()
                time.sleep(min(1, max(0, deadline - time.monotonic())))
                continue
            except OSError as exc:
                last_error = str(exc)
                self.check_cancelled()
                time.sleep(min(1, max(0, deadline - time.monotonic())))
                continue
            with self._locked():
                self._reload()
                item = self._inspect()
                if (
                    not item
                    or not item.get("State", {}).get("Running")
                    or self.state["status"] in {"stopped", "stopping", "cleanup_failed"}
                ):
                    raise RuntimeError("role stopped while readiness was checked")
                self.state.update(status="ready", **evidence)
                self._save()
                return deepcopy(self.state)
        raise TimeoutError(f"local role readiness timed out: {last_error}")

    def check_health(self, *, check_http=True):
        """Check ready ownership/process; optionally probe HTTP availability."""
        with self._locked():
            self._reload()
            if self.state["status"] != "ready":
                return
            item = self._inspect(timeout=2)
            if not item or not item.get("State", {}).get("Running"):
                raise RuntimeError("ready local role is no longer running")
            if check_http:
                self._http("/health", timeout=2)

    def start(self):
        self.launch()
        return self.wait_ready()

    def _collect(self, item):
        if item:
            rc, out, err = self._run(["logs", "--tail", "10000", item["Id"]], cancellable=False)
            if rc:
                raise RuntimeError(f"cannot collect owned role logs: {err[-1000:]}")
            (self.state_dir / "role.log").write_text(out + err)
            self.state["log_path"] = str(self.state_dir / "role.log")
            self._save()

    def collect(self):
        with self._locked():
            self._reload()
            self._collect(self._inspect())
            return deepcopy(self.state)

    def stop(self):
        # Cleanup must work even after the task's cancellation callback fires.
        with self._locked():
            self._reload()
            try:
                item = self._inspect()
                if item is None and self.state.get("launch_uncertain"):
                    raise RuntimeError("launch outcome is unresolved; absence does not yet prove cleanup")
                if item is not None:
                    self.state["launch_uncertain"] = False
                self.state.update(status="stopping", cleanup_verified=False)
                self._save()
                if item:
                    try:
                        self._collect(item)
                    except Exception as exc:  # noqa: BLE001 - retain evidence failure and still clean up
                        # Evidence failure is reported, but cannot prevent cleanup.
                        self.state["collection_error"] = str(exc)
                        self._save()
                    identifier = item["Id"]
                    rc, _, err = self._run(["rm", "--force", identifier], cancellable=False)
                    if self._inspect(identifier) is not None:
                        raise RuntimeError(
                            f"owned container still exists after removal: {err[-1000:] if rc else identifier}"
                        )
                if self._inspect() is not None:
                    raise RuntimeError("owned name still exists after cleanup")
                self.state.update(status="stopped", cleanup_verified=True)
                self.state.pop("cleanup_error", None)
                self._save()
                return deepcopy(self.state)
            except Exception as exc:
                self.state.update(
                    status="cleanup_failed",
                    cleanup_verified=False,
                    cleanup_error=str(exc),
                )
                self._save()
                raise
