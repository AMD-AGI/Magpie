"""DI session ownership around SGLang's Ray engines in prepared workers."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from ..core.executor import EndpointClientCleanupError


@contextmanager
def cancellation_signals():
    """Allow the benchmark driver to finish DI cleanup on SIGINT or SIGTERM."""
    cancelled = threading.Event()

    def check():
        if cancelled.is_set():
            raise RuntimeError("managed DI benchmark interrupted")

    def handle(_signum, _frame):
        cancelled.set()

    previous = {}
    if threading.current_thread() is threading.main_thread():
        for name in (signal.SIGTERM, signal.SIGINT):
            previous[name] = signal.signal(name, handle)
    try:
        yield check
    finally:
        for name, handler in previous.items():
            signal.signal(name, handler)


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _http(url, data=None):
    request = urllib.request.Request(
        url,
        data=None if data is None else json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        if url.endswith("/health"):
            return {"healthy": response.status == 200}
        payload = response.read()
        return json.loads(payload) if payload else {}


def _rank_layout(instances, nodes, available):
    """Map balanced engine ranks onto distinct existing Ray nodes."""
    remaining = {
        node["NodeID"]: node for node in nodes
        if node.get("Alive") and node.get("Resources", {}).get("GPU", 0) >= 1
    }
    assignments = {}
    # Fit larger engines first; leave larger nodes for engines that need them.
    for instance in sorted(instances, key=lambda item: item["tp"] * item["pp"], reverse=True):
        world = instance["tp"] * instance["pp"]
        chosen = None
        for count in range(1, len(remaining) + 1):
            if world % count:
                continue
            per_node = world // count
            candidates = sorted(
                (node for key, node in remaining.items()
                 if available.get(key, {}).get("GPU", 0) >= per_node),
                key=lambda node: (available[node["NodeID"]]["GPU"], node["NodeID"]),
            )
            frontend = next(
                (node for node in candidates
                 if available[node["NodeID"]].get("CPU", 0) >= 1), None,
            )
            if len(candidates) >= count and frontend is not None:
                chosen = [frontend, *(node for node in candidates if node != frontend)][:count]
                break
        if chosen is None:
            raise ValueError("available Ray nodes cannot fit the DI engine topology")
        assignments[instance["id"]] = (chosen, per_node)
        for node in chosen:
            remaining.pop(node["NodeID"])
    bundles, layout = [], {}
    for instance in instances:
        selected, per_node = assignments[instance["id"]]
        indices = []
        for node in selected:
            resources = node["Resources"]
            keys = [key for key in resources if key.startswith("node:") and key != "node:__internal_head__"]
            preferred = "node:" + node.get("NodeManagerAddress", "")
            key = preferred if preferred in keys else keys[0] if len(keys) == 1 else None
            if key is None:
                raise RuntimeError("Ray node has no unambiguous node placement resource")
            for _rank in range(per_node):
                bundle = {"GPU": 1, key: 0.001}
                if not indices:
                    bundle["CPU"] = 1
                indices.append(len(bundles))
                bundles.append(bundle)
        layout[instance["id"]] = {
            "nodes": [node["NodeID"] for node in selected],
            "bundle_indices": indices,
        }
    return bundles, layout


class _NativeRole:
    """A CPU-only actor supervises its Ray HTTP frontend and optional router."""

    def __init__(self, owner, instance, directory):
        self.owner, self.instance = owner, instance
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.processes = {}
        self.records = {}

    def identity(self):
        import ray

        return {
            "owner": self.owner,
            "instance": self.instance["id"],
            "node_id": ray.get_runtime_context().get_node_id(),
            "host": ray.util.get_node_ip_address(),
            "uid": os.getuid(),
            "python": sys.executable,
            "gpu_ids": sorted(
                str(g)
                for g in ray.get_runtime_context().get_accelerator_ids().get("GPU", [])
            ),
        }

    def launch(self, name, argv, env, ports):
        if name in self.records:
            raise RuntimeError("process launch already attempted")
        # Reject stale endpoints before starting this generation's processes.
        for port in ports:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("0.0.0.0", port))
        record = {"argv": argv, "status": "launching"}
        self.records[name] = record
        _write(self.directory / "processes.json", self.records)
        log = self.directory / f"{name}.log"
        with log.open("ab") as stream:
            process = subprocess.Popen(
                argv,
                env={**os.environ, **env},
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.processes[name] = process
        record.update(pid=process.pid, pgid=process.pid, status="started", log=str(log))
        _write(self.directory / "processes.json", self.records)
        return deepcopy(record)

    def inspect(self):
        return {
            name: {**record, "returncode": self.processes[name].poll()}
            for name, record in self.records.items()
            if name in self.processes
        }

    def request(self, name, path, port, data=None):
        process = self.processes[name]
        if process.poll() is not None:
            raise RuntimeError(
                f"{name} exited with {process.returncode}; inspect its log"
            )
        return _http(f"http://127.0.0.1:{port}{path}", data)

    def stop(self):
        from ..modes.benchmark.endpoint_client import _group_exists

        for name, process in reversed(list(self.processes.items())):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            if _group_exists(process.pid):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
            deadline = time.monotonic() + 5
            while _group_exists(process.pid):
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"{name} process group cleanup is unconfirmed")
                time.sleep(0.05)
            self.records[name]["status"] = "stopped"
        _write(self.directory / "processes.json", self.records)
        return {"cleanup_verified": True, "processes": deepcopy(self.records)}


class NativeRayDISession:
    """Own native role processes and their Ray reservations as one evaluation.

    Capacity and the worker runtime image belong to the caller. A lost launcher
    or uncertain endpoint client keeps the session unavailable for reuse.
    """

    def __init__(self, config, state_dir, check_cancelled=None):
        from .native_config import normalize_native_di

        if not isinstance(config, dict):
            raise TypeError("native DI requires a benchmark mapping")
        self.raw = deepcopy(config.get("benchmark", config))
        self.spec = normalize_native_di(self.raw)
        self.state_dir = Path(state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.state_dir / "native-session.json"
        if self.path.exists():
            raise ValueError("native session exists; inspect/reconcile it before reuse")
        self.check_cancelled = check_cancelled or (lambda: None)
        self.actors, self.pg = {}, None
        self._recovered = False
        self.state = {
            "status": "created",
            "generations": [],
            "client_cleanup_pending": None,
        }
        self._generation()

    @property
    def id(self):
        return self.state["generation"]["id"]

    @property
    def endpoint(self):
        return self.state["generation"].get("endpoint")

    def _save(self):
        _write(self.path, {"raw": self.raw, "spec": self.spec, **self.state})

    def _generation(self):
        generation = {
            "id": uuid.uuid4().hex,
            "actors": {},
            "identities": {},
            "cleanup_verified": False,
            "launch_intents": [],
            "status": "created",
            "rank_layout": {},
            "schedulers": {},
        }
        generation["placement_group"] = f"magpie-native-di-{generation['id']}"
        generation["directory"] = str(self.state_dir / generation["id"])
        Path(generation["directory"]).mkdir()
        self.state["generation"] = generation
        self.state["generations"].append(generation)
        self.state["status"] = "created"
        self.actors, self.pg = {}, None
        self._save()

    def _connect(self):
        import ray

        config = self.spec["ray_config"]
        if not ray.is_initialized():
            handlers = {}
            if threading.current_thread() is threading.main_thread():
                for sig in (signal.SIGTERM, signal.SIGINT):
                    handler = signal.getsignal(sig)
                    if handler not in (
                        None,
                        signal.SIG_DFL,
                        signal.default_int_handler,
                    ):
                        handlers[sig] = handler
            try:
                ray.init(
                    address=config.get("cluster_address", "auto"),
                    namespace=config.get("namespace", "magpie-di"),
                )
            finally:
                # Keep the caller's cooperative cancellation hook after Ray init.
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
        namespace = ray.get_runtime_context().namespace
        if config.get("namespace", namespace) != namespace:
            raise ValueError("existing Ray context has a different DI namespace")
        expected = config.get("expected_node_ids")
        if expected is not None and {
            n["NodeID"] for n in ray.nodes() if n.get("Alive")
        } != set(expected):
            raise ValueError(
                "Ray cluster membership differs from the expected allocation"
            )
        config["namespace"] = namespace
        self._save()
        return ray

    def _rpc(self, reference, timeout=30, cancellable=True):
        import ray

        deadline = time.monotonic() + timeout
        while True:
            if cancellable:
                self.check_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "native Ray operation timed out; reconcile before reuse"
                )
            ready, _ = ray.wait([reference], timeout=min(1, remaining))
            if ready:
                return ray.get(reference)

    def _assert_mutable(self):
        if self._recovered:
            raise RuntimeError("recovered sessions allow inspection and cleanup only")
        self._require_client_cleanup()

    def _require_client_cleanup(self):
        if self.state.get("client_cleanup_pending"):
            raise EndpointClientCleanupError(
                "native endpoint client cleanup unconfirmed; retain reservations"
            )

    def _engine_command(self, instance):
        generation = self.state["generation"]
        python = generation["identities"][instance["id"]]["python"]
        layout = generation["rank_layout"][instance["id"]]
        command = [
            python,
            "-m",
            "Magpie.di.sglang_ray",
            "--pg-name", generation["placement_group"],
            "--pg-id", generation["placement_group_id"],
            "--namespace", self.spec["ray_config"]["namespace"],
            "--bundle-indices", ",".join(map(str, layout["bundle_indices"])),
            "--journal-path", str(Path(generation["directory"]) / instance["id"] / "ray-engine.json"),
            "--",
            "--use-ray",
            "--nnodes", str(len(layout["nodes"])),
            *instance["server_args"],
            "--model-path",
            self.spec["model"],
            "--served-model-name",
            self.spec["model"],
            "--tensor-parallel-size",
            str(instance["tp"]),
            "--ep-size",
            str(instance["ep"]),
            "--pp-size",
            str(instance["pp"]),
            "--host",
            "0.0.0.0",
            "--port",
            str(self.spec["server_port"]),
            "--disaggregation-mode",
            instance["role"],
            "--disaggregation-transfer-backend",
            "mooncake",
            "--disaggregation-bootstrap-port",
            str(self.spec["bootstrap_port"]),
        ]
        if self.spec.get("nic"):
            command += ["--disaggregation-ib-device", self.spec["nic"]]
        return command

    def _router_command(self):
        first = next(i for i in self.spec["instances"] if i["role"] == "prefill")
        python = self.state["generation"]["identities"][first["id"]]["python"]
        command = [
            python,
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--host",
            "0.0.0.0",
            "--port",
            str(self.spec["router_port"]),
            "--tokenizer-path",
            self.spec["model"],
        ]
        identities = self.state["generation"]["identities"]
        for instance in self.spec["instances"]:
            url = f"http://{identities[instance['id']]['host']}:{self.spec['server_port']}"
            command += ["--" + instance["role"], url]
            if instance["role"] == "prefill":
                command += [str(self.spec["bootstrap_port"])]
        return command

    def _ready(self, instance, deadline):
        actor = self.actors[instance["id"]]
        expected = {
            "model_path": self.spec["model"],
            "disaggregation_mode": instance["role"],
            "tp_size": instance["tp"],
            "ep_size": instance["ep"],
            "pp_size": instance["pp"],
            "nnodes": len(self.state["generation"]["rank_layout"][instance["id"]]["nodes"]),
            "disaggregation_transfer_backend": "mooncake",
        }
        arguments = instance["server_args"]
        for flag, field in (
            ("--context-length", "context_length"),
            ("--dtype", "dtype"),
            ("--quantization", "quantization"),
        ):
            for index, value in enumerate(arguments):
                if value == flag:
                    expected[field] = arguments[index + 1]
                elif value.startswith(flag + "="):
                    expected[field] = value.split("=", 1)[1]
        last = "no response"
        while time.monotonic() < deadline:
            self.check_cancelled()
            try:
                info = self._rpc(
                    actor.request.remote(
                        "engine", "/get_server_info", self.spec["server_port"]
                    )
                )
            except (urllib.error.URLError, TimeoutError) as error:
                last = str(error)
                time.sleep(0.2)
                continue
            if any(str(info.get(key)) != str(value) for key, value in expected.items()):
                raise RuntimeError(
                    f"{instance['id']} configuration mismatch: expected {expected}; got {info}"
                )
            return info
        raise TimeoutError(f"{instance['id']} not ready: {last}")

    def start(self):
        self._assert_mutable()
        if self.state["status"] != "created":
            raise RuntimeError("native session already started")
        from ray._private.state import available_resources_per_node
        from ray.util.placement_group import placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        from .sglang_ray import scheduler_names

        generation = self.state["generation"]
        ray = self._connect()
        self.state["status"] = generation["status"] = "starting"
        # Names and placement intent are durable before detached resources exist.
        generation["actors"] = {
            i["id"]: f"{generation['placement_group']}-{i['id']}"
            for i in self.spec["instances"]
        }
        self._save()
        try:
            timeout = self.spec["ray_config"].get("placement_timeout_seconds", 120)
            admission_deadline = time.monotonic() + timeout
            while True:
                self.check_cancelled()
                try:
                    bundles, layout = _rank_layout(
                        self.spec["instances"], ray.nodes(), available_resources_per_node()
                    )
                    break
                except ValueError as error:
                    if time.monotonic() >= admission_deadline:
                        raise TimeoutError(str(error)) from error
                    time.sleep(0.2)
            generation["rank_layout"] = layout
            generation["placement_requested"] = True
            self._save()
            self.pg = placement_group(
                bundles,
                strategy="PACK",
                name=generation["placement_group"],
                lifetime="detached",
            )
            generation["placement_created"] = True
            generation["placement_group_id"] = self.pg.id.hex()
            self._save()
            self._rpc(
                self.pg.ready(),
                max(0, admission_deadline - time.monotonic()),
            )
            actor_class = ray.remote(_NativeRole)
            for instance in self.spec["instances"]:
                assignment = layout[instance["id"]]
                actor = actor_class.options(
                    num_cpus=1,
                    num_gpus=0,
                    name=generation["actors"][instance["id"]],
                    namespace=self.spec["ray_config"]["namespace"],
                    lifetime="detached",
                    max_restarts=0,
                    max_task_retries=0,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=self.pg,
                        placement_group_bundle_index=assignment["bundle_indices"][0],
                    ),
                ).remote(
                    self.id,
                    instance,
                    str(Path(generation["directory"]) / instance["id"]),
                )
                self.actors[instance["id"]] = actor
                identity = self._rpc(actor.identity.remote())
                if (
                    identity["owner"] != self.id
                    or identity["instance"] != instance["id"]
                    or identity["gpu_ids"]
                    or identity["node_id"] != assignment["nodes"][0]
                ):
                    raise RuntimeError("DI frontend identity/placement mismatch")
                generation["identities"][instance["id"]] = identity
                generation["schedulers"][instance["id"]] = scheduler_names(
                    generation["placement_group_id"], identity["host"],
                    assignment["bundle_indices"], instance["tp"],
                )
                self._save()
            for instance in self.spec["instances"]:
                generation["launch_intents"].append(instance["id"])
                self._save()
                self._rpc(
                    self.actors[instance["id"]].launch.remote(
                        "engine",
                        self._engine_command(instance),
                        instance["serving_env"],
                        [self.spec["server_port"], self.spec["bootstrap_port"]],
                    )
                )
            deadline = time.monotonic() + self.spec["startup_timeout"]
            generation["server_info"] = {
                i["id"]: self._ready(i, deadline) for i in self.spec["instances"]
            }
            first = next(i for i in self.spec["instances"] if i["role"] == "prefill")
            generation["router_actor"] = first["id"]
            self._rpc(
                self.actors[first["id"]].launch.remote(
                    "router",
                    self._router_command(),
                    self.spec["serving_env"],
                    [self.spec["router_port"]],
                )
            )
            generation["endpoint"] = (
                f"http://{generation['identities'][first['id']]['host']}:{self.spec['router_port']}"
            )
            while True:
                self.check_cancelled()
                try:
                    actor = self.actors[first["id"]]
                    self._rpc(
                        actor.request.remote(
                            "router", "/health", self.spec["router_port"]
                        )
                    )
                    probe = self._rpc(
                        actor.request.remote(
                            "router",
                            "/v1/completions",
                            self.spec["router_port"],
                            {
                                "model": self.spec["model"],
                                "prompt": "Hello",
                                "max_tokens": 8,
                                "temperature": 0,
                            },
                        )
                    )
                    if not probe.get("choices") or not probe.get("usage", {}).get(
                        "completion_tokens"
                    ):
                        raise RuntimeError(
                            "routed readiness returned no completion tokens"
                        )
                    break
                except Exception:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.2)
            generation["readiness"] = {"routed_completion": probe}
            self.state["status"] = generation["status"] = "ready"
            self._save()
            return self.endpoint
        except BaseException:
            self.state["status"] = generation["status"] = "failed"
            self._save()
            try:
                self.collect()
            except Exception as error:  # noqa: BLE001 - preserve startup failure if evidence collection fails
                generation["collection_error"] = str(error)
                self._save()
            self.stop()
            raise

    def inspect(self):
        result = deepcopy(self.state)
        result["processes"] = {
            key: self._rpc(actor.inspect.remote(), cancellable=False)
            for key, actor in self.actors.items()
        }
        return result

    def collect(self):
        _write(
            Path(self.state["generation"]["directory"]) / "inspection.json",
            self.inspect(),
        )
        return Path(self.state["generation"]["directory"])

    def _stop_schedulers(self):
        """Verify SGLang's owned GPU actors stopped before releasing capacity."""
        import ray

        from .sglang_ray import scheduler_names

        generation = self.state["generation"]
        namespace = self.spec["ray_config"]["namespace"]
        pending = {}
        for instance in self.spec["instances"]:
            key = instance["id"]
            names = generation.get("schedulers", {}).get(key, [])
            if not names:
                if key in generation.get("launch_intents", []):
                    raise RuntimeError("launched DI engine has no scheduler ownership journal")
                continue
            assignment = generation["rank_layout"][key]
            expected = scheduler_names(
                generation["placement_group_id"], generation["identities"][key]["host"],
                assignment["bundle_indices"], instance["tp"],
            )
            if names != expected:
                raise RuntimeError("DI scheduler ownership journal is inconsistent")
            journal = Path(generation["directory"]) / key / "ray-engine.json"
            ids = []
            if journal.exists():
                record = json.loads(journal.read_text())
                if record.get("compatibility_error"):
                    raise RuntimeError("SGLang scheduler compatibility needs reconciliation")
                if (
                    record.get("pg_id") != generation["placement_group_id"]
                    or record.get("namespace") != namespace
                    or record.get("scheduler_names") != names
                ):
                    raise RuntimeError("SGLang scheduler journal identity changed")
                ids = record.get("actor_ids", [])
                if ids and len(ids) != len(names):
                    raise RuntimeError("SGLang scheduler actor identities are incomplete")
            for index, name in enumerate(names):
                try:
                    actor = ray.get_actor(name, namespace=namespace)
                except ValueError:
                    continue
                if ids and actor._actor_id.hex() != ids[index]:
                    raise RuntimeError("SGLang scheduler actor identity changed")
                pending[name] = actor
                ray.kill(actor, no_restart=True)
        deadline = time.monotonic() + 30
        while pending:
            for name in list(pending):
                try:
                    actor = ray.get_actor(name, namespace=namespace)
                except ValueError:
                    pending.pop(name)
                    continue
                if actor._actor_id != pending[name]._actor_id:
                    raise RuntimeError("SGLang scheduler changed during cleanup")
            if pending:
                if time.monotonic() >= deadline:
                    raise RuntimeError("SGLang GPU scheduler cleanup was not confirmed")
                time.sleep(0.1)
        generation["scheduler_cleanup_verified"] = True
        self._save()

    def _validate_rank_layout(self):
        generation = self.state["generation"]
        layout = generation.get("rank_layout", {})
        if not layout and not generation.get("launch_intents"):
            return
        instances = self.spec["instances"]
        if set(layout) != {instance["id"] for instance in instances}:
            raise RuntimeError("DI rank layout has missing or unknown instances")
        indices, nodes = set(), set()
        for instance in instances:
            assignment = layout[instance["id"]]
            ranks, selected = assignment["bundle_indices"], assignment["nodes"]
            if (
                len(ranks) != instance["tp"] * instance["pp"]
                or len(set(ranks)) != len(ranks)
                or any(type(rank) is not int for rank in ranks)
                or indices.intersection(ranks)
                or not selected
                or len(set(selected)) != len(selected)
                or len(ranks) % len(selected)
                or nodes.intersection(selected)
            ):
                raise RuntimeError("DI rank layout has overlapping or invalid ownership")
            indices.update(ranks)
            nodes.update(selected)
        if indices != set(range(sum(instance["tp"] * instance["pp"] for instance in instances))):
            raise RuntimeError("DI rank bundle indices are out of range")

    def stop(self):
        self._require_client_cleanup()
        generation = self.state["generation"]
        if self.state["status"] == "stopped":
            return
        if not generation["cleanup_verified"]:
            if "rank_layout" in generation:
                self._validate_rank_layout()
            # Missing launchers cannot prove their children absent.
            missing = set(generation["launch_intents"]) - set(self.actors)
            if missing:
                raise RuntimeError(
                    f"native launchers missing; reconcile before reuse: {sorted(missing)}"
                )
            receipts, errors = {}, []
            for key, actor in self.actors.items():
                try:
                    receipts[key] = self._rpc(actor.stop.remote(), cancellable=False)
                    if not receipts[key].get("cleanup_verified"):
                        errors.append(f"{key}: process cleanup unconfirmed")
                except Exception as error:  # noqa: BLE001 - aggregate RPC failures without releasing reservations
                    errors.append(f"{key}: {error}")
            if not errors:
                try:
                    self._stop_schedulers()
                except Exception as error:  # noqa: BLE001 - retain the group on uncertain GPU cleanup
                    errors.append(str(error))
            if errors:
                self.state["status"] = generation["status"] = "cleanup_failed"
                generation["cleanup_errors"] = errors
                self._save()
                raise RuntimeError("; ".join(errors))
            generation["cleanup_receipts"] = receipts
            generation["cleanup_verified"] = True
            self._save()
        import ray
        from ray.util.placement_group import get_placement_group, remove_placement_group

        if self.pg is None and generation.get("placement_requested"):
            try:
                self.pg = get_placement_group(generation["placement_group"])
                if generation.get("placement_group_id") and self.pg.id.hex() != generation["placement_group_id"]:
                    raise RuntimeError("DI placement group identity changed before cleanup")
                generation["placement_created"] = True
                self._save()
            except ValueError:
                if not generation.get("placement_created"):
                    self.state["status"] = generation["status"] = "cleanup_failed"
                    generation["cleanup_errors"] = [
                        "placement group creation is unconfirmed; reconcile its journaled name before reuse"
                    ]
                    self._save()
                    raise RuntimeError(generation["cleanup_errors"][0]) from None
        for actor in list(self.actors.values()):
            ray.kill(actor, no_restart=True)
        self.actors.clear()
        if self.pg is not None:
            remove_placement_group(self.pg)
            self.pg = None
        generation["cleanup_errors"] = []
        self.state["status"] = generation["status"] = "stopped"
        self._save()

    @classmethod
    def load(cls, state_dir):
        self = cls.__new__(cls)
        self.state_dir = Path(state_dir).resolve()
        self.path = self.state_dir / "native-session.json"
        record = json.loads(self.path.read_text())
        from .native_config import normalize_native_di

        raw, spec = record.pop("raw", None), record.pop("spec", None)
        if not isinstance(raw, dict) or not isinstance(spec, dict):
            raise TypeError(
                "native session journal requires raw benchmark and normalized spec"
            )
        self.raw = deepcopy(raw.get("benchmark", raw))
        normalized = normalize_native_di(self.raw)
        # The connected namespace is added after normalization. Everything else
        # defining the live serving topology must still match the saved source.
        if any(
            normalized[key] != spec.get(key)
            for key in normalized
            if key != "ray_config"
        ):
            raise ValueError("native session journal raw benchmark and spec disagree")
        self.spec = spec
        self.state = record
        current, history = record.get("generation"), record.get("generations")
        if (
            not isinstance(current, dict)
            or not isinstance(history, list)
            or not history
            or not current.get("id")
            or sum(
                isinstance(item, dict) and item.get("id") == current["id"]
                for item in history
            )
            != 1
            or history[-1] != current
        ):
            raise ValueError(
                "native session generation history is ambiguous or contradictory"
            )
        # JSON duplicates dictionaries; recovered mutations must update both
        # the current view and its canonical tail in the history journal.
        self.state["generation"] = history[-1]
        if "rank_layout" in self.state["generation"]:
            self._validate_rank_layout()
        self.check_cancelled = lambda: None
        self._recovered, self.actors, self.pg = True, {}, None
        ray = self._connect()
        from ray.util.placement_group import get_placement_group

        generation = self.state["generation"]
        if self.state["status"] == "stopped":
            return self
        try:
            self.pg = get_placement_group(generation["placement_group"])
        except ValueError:
            pass
        if self.pg is not None and generation.get("placement_group_id") and self.pg.id.hex() != generation["placement_group_id"]:
            raise RuntimeError("recovered DI placement group identity changed")
        for key, name in generation["actors"].items():
            try:
                actor = ray.get_actor(
                    name, namespace=self.spec["ray_config"]["namespace"]
                )
            except ValueError:
                continue
            identity = self._rpc(actor.identity.remote(), cancellable=False)
            if generation["identities"].get(key) != identity:
                raise RuntimeError("recovered native launcher identity changed")
            self.actors[key] = actor
        return self

    @contextmanager
    def client_run(self, task_id, output_dir):
        """Keep reservations until the owning worker confirms client cleanup."""
        self._assert_mutable()
        if self.state["status"] != "ready":
            raise RuntimeError("client execution requires ready native DI")
        self.check_cancelled()
        self.state["client_cleanup_pending"] = {
            "task_id": task_id,
            "output_dir": str(Path(output_dir).resolve()),
        }
        self._save()
        try:
            yield
        finally:
            # The managed client primitive cleans its group on every exit,
            # including Ray cancellation. Its typed error is the sole exception.
            if not isinstance(sys.exc_info()[1], EndpointClientCleanupError):
                self.state["client_cleanup_pending"] = None
                self._save()

    def replace(self, config):
        self._assert_mutable()
        from .native_config import normalize_native_di

        candidate = normalize_native_di(config.get("benchmark", config))
        if self.state["status"] != "ready":
            raise RuntimeError("replacement requires ready native DI")
        previous_raw, previous_spec = deepcopy(self.raw), deepcopy(self.spec)
        self.collect()
        self.stop()
        try:
            self.raw, self.spec = deepcopy(config.get("benchmark", config)), candidate
            self._generation()
            return self.start()
        except Exception as error:
            if self.state["status"] != "stopped":
                raise
            self.check_cancelled()
            self.raw, self.spec = previous_raw, previous_spec
            self._generation()
            self.start()
            self.state["replacement"] = {"restored": True, "error": str(error)}
            self._save()
            raise RuntimeError(
                f"candidate failed; previous candidate restored: {error}"
            ) from error

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_args):
        try:
            self.collect()
        finally:
            self.stop()
