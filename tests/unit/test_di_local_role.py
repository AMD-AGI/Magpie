import argparse
import errno
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from Magpie.di.local_role import LocalRole, validate_local_spec

IMAGE = "sha256:" + "a" * 64
CID = "b" * 64


def spec(role="prefill", **changes):
    value = dict(  # noqa: C408
        run_id="run",
        attempt_id=1,
        instance_id=role,
        role=role,
        image=IMAGE,
        model_path="/models/model",
        served_model="model",
        host="node-a",
        port=30000,
        startup_timeout=3,
    )
    if role == "router":
        value.update(
            prefill=[{"url": "http://node-a:30000", "bootstrap_port": 8998}],
            decode=[{"url": "http://node-b:30001"}],
            port=18000,
        )
    else:
        value.update(gpus=[0, 1], tp=2, nic="ionic_0")
    value.update(changes)
    return value


class Docker:
    def __init__(self):
        self.item = None
        self.calls = []
        self.remove_fails = False
        self.daemon_error = False
        self.lost_launch_reply = False
        self.role = None

    def run(self, argv, **kwargs):
        self.calls.append(argv)
        if argv[:2] == ["container", "inspect"]:
            if self.daemon_error:
                return 1, "", "Cannot connect to the Docker daemon"
            if not self.item:
                return 1, "[]\n", "Error: No such container: " + argv[2]
            return 0, json.dumps([self.item]), ""
        if argv[:2] == ["image", "inspect"]:
            return 0, IMAGE + "\n", ""
        if argv[0] == "run":
            state = json.loads(self.role.path.read_text())
            assert state["status"] == "launching" and not state["cleanup_verified"]
            labels = dict(
                argv[i + 1].split("=", 1)
                for i, arg in enumerate(argv)
                if arg == "--label"
            )
            self.item = {
                "Id": CID,
                "Image": IMAGE,
                "Config": {"Labels": labels},
                "State": {"Running": True},
            }
            if self.lost_launch_reply:
                raise TimeoutError(
                    "client connection lost after Docker accepted request"
                )
            return 0, CID, ""
        if argv[:2] == ["rm", "--force"]:
            assert argv[2] == CID
            if self.remove_fails:
                return 1, "", "removal failed"
            self.item = None
            return 0, CID, ""
        if argv[0] == "logs":
            return 0, "Mooncake rdma role log\n", "stderr\n"
        raise AssertionError(argv)


def make(tmp_path, monkeypatch, value=None, docker=None, callback=None):
    role = LocalRole(value or spec(), tmp_path, callback)
    docker = docker or Docker()
    docker.role = role
    monkeypatch.setattr(role, "_run", docker.run)
    monkeypatch.setattr(role, "_ports_free", lambda: None)
    return role, docker


@pytest.mark.parametrize(
    "changes",
    [
        {"image": "candidate:latest"},
        {"command_prefix": ["ssh", "other"]},
        {"server_args": ["--nnodes=2"]},
        {"server_args": ["--nccl-init-addr", "other:1234"]},
        {"server_args": ["--nccl-init", "other:1234"]},
        {"server_args": ["--config", "/config.yaml"]},
        {"server_args": ["--disaggregation-mode=decode"]},
        {"server_args": ["--use-ray"]},
        {"server_args": ["--nnod=2"]},
        {"env": {"HIP_VISIBLE_DEVICES": "7"}},
        {"env": {"MC_FORCE_TCP": "1"}},
        {"gpus": [0, 0]},
        {"tp": 3},
        {"model_path": "relative"},
        {"port": True},
        {"startup_timeout": float("nan")},
        {"startup_timeout": float("inf")},
        {"startup_timeout": 0},
        {"startup_timeout": -1},
        {"mounts": [{"source": "/var/run/docker.sock", "target": "/x"}]},
    ],
)
def test_validation_rejects_ambiguous_or_remote_assignment(changes):
    with pytest.raises(ValueError):
        validate_local_spec(spec(**changes))


def test_local_spec_does_not_impose_scheduling_or_name_length_limits():
    value = spec(
        "router", run_id="r" * 129, served_model="m" * 513,
        startup_timeout=86401,
        prefill=[{"url": f"http://prefill-{i}:30000"} for i in range(65)],
    )
    validated = validate_local_spec(value)
    assert validated["run_id"] == value["run_id"]
    assert validated["served_model"] == value["served_model"]
    assert validated["startup_timeout"] == value["startup_timeout"]
    assert validated["prefill"] == value["prefill"]


def test_role_command_immutable_local_topology(tmp_path):
    role = LocalRole(spec(), tmp_path)
    command = role.command()
    assert command[:2] == ["run", "--detach"]
    assert (
        "--tensor-parallel-size" in command
        and command[command.index("--tensor-parallel-size") + 1] == "2"
    )
    assert IMAGE in command
    assert "HIP_VISIBLE_DEVICES=0,1" in command
    assert not any(word in command for word in ("ssh", "srun", "ray", "--privileged"))
    assert command[command.index("--disaggregation-mode") + 1] == "prefill"


def test_port_preflight_accepts_closed_server_time_wait(tmp_path):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=2) as client, listener.accept()[0] as accepted:
            accepted.settimeout(2)
            # Server closes first, leaving its local port in TIME_WAIT.
            accepted.shutdown(socket.SHUT_WR)
            assert client.recv(1) == b""
            client.shutdown(socket.SHUT_WR)
            assert accepted.recv(1) == b""
    with socket.socket() as plain_probe, pytest.raises(OSError) as error:
        plain_probe.bind(("0.0.0.0", port))
    assert error.value.errno == errno.EADDRINUSE
    LocalRole(spec("router", port=port), tmp_path)._ports_free()


@pytest.mark.parametrize("address", [
    pytest.param("127.0.0.1", marks=pytest.mark.skipif(
        sys.platform != "linux",
        reason="Linux/ROCm bind semantics: macOS permits reusable wildcard bind beside a loopback listener",
    )),
    "0.0.0.0",
])
@pytest.mark.parametrize("reuse", [False, True])
def test_port_preflight_rejects_live_listener(tmp_path, address, reuse):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, int(reuse))
        listener.bind((address, 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(OSError, match=f"port {port} unavailable") as error:
            LocalRole(spec("router", port=port), tmp_path)._ports_free()
        assert error.value.errno == errno.EADDRINUSE


def test_router_multiple_instances_follow_pinned_parser_shape(tmp_path):
    role = LocalRole(
        spec(
            "router",
            prefill=[
                {"url": "http://p1:1", "bootstrap_port": 8998},
                {"url": "http://p2:2"},
            ],
            decode=[{"url": "http://d1:3"}, {"url": "http://d2:4"}],
        ),
        tmp_path,
    )
    # These argument definitions are the v0.5.18 upstream router contract.
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill", nargs="+", action="append")
    parser.add_argument("--decode", nargs=1, action="append")
    args, _ = parser.parse_known_args(role.command())
    assert args.prefill == [["http://p1:1", "8998"], ["http://p2:2"]]
    assert args.decode == [["http://d1:3"], ["http://d2:4"]]
    assert "--device" not in role.command()


@pytest.mark.parametrize(
    "entries",
    [
        [],
        [{"url": "http://u:p@host:3"}],
        [{"url": "http://host:3/path"}],
        [{"url": "http://host:3"}, {"url": "http://host:3"}],
    ],
)
def test_router_rejects_invalid_lists(entries):
    with pytest.raises(ValueError):
        validate_local_spec(spec("router", prefill=entries))


def test_launch_replay_after_lost_reply_adopts_one_container(tmp_path, monkeypatch):
    role, docker = make(tmp_path, monkeypatch)
    docker.lost_launch_reply = True
    with pytest.raises(TimeoutError):
        role.launch()
    assert json.loads(role.path.read_text())["status"] == "launching"
    recovered, _ = make(tmp_path, monkeypatch, docker=docker)
    assert recovered.launch()["status"] == "running"
    assert len([x for x in docker.calls if x[0] == "run"]) == 1
    assert recovered.launch()["container_id"] == CID
    assert len([x for x in docker.calls if x[0] == "run"]) == 1


def test_changed_spec_does_not_adopt_existing_owner(tmp_path):
    LocalRole(spec(), tmp_path)
    with pytest.raises(ValueError, match="changed"):
        LocalRole(spec(image="sha256:" + "c" * 64), tmp_path)


def test_cleanup_survives_cancel_and_is_idempotent(tmp_path, monkeypatch):
    role, docker = make(tmp_path, monkeypatch)
    role.launch()
    role.check_cancelled = lambda: (_ for _ in ()).throw(RuntimeError("cancelled"))
    role.collect()
    assert (tmp_path / "role.log").read_text() == "Mooncake rdma role log\nstderr\n"
    assert role.stop()["cleanup_verified"] is True
    assert role.stop()["status"] == "stopped"
    assert len([x for x in docker.calls if x[0] == "rm"]) == 1
    recovered = LocalRole.load(tmp_path)
    assert recovered.state["cleanup_verified"]
    with pytest.raises(RuntimeError, match="relaunched"):
        make(tmp_path, monkeypatch, docker=docker)[0].launch()


@pytest.mark.parametrize("tamper", ["owner", "image"])
def test_cleanup_never_removes_foreign_container(tmp_path, monkeypatch, tamper):
    role, docker = make(tmp_path, monkeypatch)
    role.launch()
    if tamper == "owner":
        docker.item["Config"]["Labels"]["magpie.di.local-owner"] = "foreign"
    else:
        docker.item["Image"] = "sha256:" + "f" * 64
    with pytest.raises(RuntimeError, match="mismatch"):
        role.stop()
    assert not any(x[0] == "rm" for x in docker.calls)
    assert not json.loads(role.path.read_text())["cleanup_verified"]


@pytest.mark.parametrize("failure", ["daemon", "remove"])
def test_cleanup_uncertainty_keeps_reservation_required(tmp_path, monkeypatch, failure):
    role, docker = make(tmp_path, monkeypatch)
    role.launch()
    docker.daemon_error = failure == "daemon"
    docker.remove_fails = failure == "remove"
    with pytest.raises(RuntimeError):
        role.stop()
    assert json.loads(role.path.read_text())["status"] == "cleanup_failed"
    assert not role.state["cleanup_verified"]
    docker.daemon_error = docker.remove_fails = False
    assert role.stop()["cleanup_verified"]


def test_running_role_disappearance_does_not_restart_attempt(tmp_path, monkeypatch):
    role, docker = make(tmp_path, monkeypatch)
    role.launch()
    docker.item = None
    with pytest.raises(RuntimeError, match="disappeared"):
        role.launch()


def test_immutable_image_must_exist_before_container_effect(tmp_path, monkeypatch):
    role, docker = make(tmp_path, monkeypatch)
    original = docker.run
    monkeypatch.setattr(
        role,
        "_run",
        lambda argv, **kw: (
            (0, "sha256:" + "c" * 64, "")
            if argv[:2] == ["image", "inspect"]
            else original(argv, **kw)
        ),
    )
    with pytest.raises(RuntimeError, match="immutable image"):
        role.launch()
    assert not any(x[0] == "run" for x in docker.calls)


def test_unhealthy_serving_role_never_ready(tmp_path, monkeypatch):
    role, _ = make(tmp_path, monkeypatch, spec(startup_timeout=0.01))
    role.launch()
    def unhealthy(path, **kwargs):
        assert path == "/health"
        raise ConnectionRefusedError("server still starting")
    monkeypatch.setattr(role, "_http", unhealthy)
    with pytest.raises(TimeoutError, match="readiness timed out"):
        role.wait_ready()
    assert role.state["status"] != "ready"


@pytest.mark.parametrize("role_name", ["prefill", "decode", "router"])
def test_http_health_readiness_and_router_completion(tmp_path, monkeypatch, role_name):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            # Serving readiness must not depend on /get_server_info support.
            if self.path != "/health":
                self.send_error(404)
                return
            if len(calls) == 1:
                self.send_error(503, "Loading model")
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, body))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                b'{"choices":[{"text":"Paris"}],"usage":{"completion_tokens":1}}'
            )

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    role, _ = make(tmp_path, monkeypatch, spec(role_name, port=server.server_port))
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    try:
        result = role.start()
        assert result["status"] == "ready"
        assert calls[:2] == ["/health", "/health"]
        if role_name == "router":
            assert result["routed_ready"]
            assert calls[2][0] == "/v1/completions" and calls[2][1]["max_tokens"] == 8
        else:
            assert len(calls) == 2
            assert "server_info" not in result
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_probe_cancellation_is_responsive(tmp_path, monkeypatch):
    role, _ = make(tmp_path, monkeypatch, spec("router"))
    blocked = threading.Event()
    monkeypatch.setattr(role, "_http", lambda *a, **k: blocked.wait(3))
    role.check_cancelled = lambda: (_ for _ in ()).throw(RuntimeError("cancelled"))
    start = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="cancelled"):
            role._probe()
        assert time.monotonic() - start < 1.5
    finally:
        blocked.set()


def test_local_socket_forces_local_docker_despite_environment(tmp_path, monkeypatch):
    role = LocalRole(spec(), tmp_path)
    captured = []

    class Process:
        returncode = 0

        def __init__(self, argv, **kwargs):
            captured.extend(argv)

        def communicate(self, **kwargs):
            return "", ""

    monkeypatch.setenv("DOCKER_HOST", "ssh://foreign-host")
    monkeypatch.setattr("Magpie.di.local_role.subprocess.Popen", Process)
    role._run(["version"])
    assert captured == ["docker", "--host", "unix:///var/run/docker.sock", "version"]


@pytest.mark.parametrize("failure", ["timeout", "nonzero"])
def test_uncertain_launch_without_observed_container_cannot_release_or_relaunch(
    tmp_path, monkeypatch, failure
):
    role, docker = make(tmp_path, monkeypatch)
    original = docker.run

    def interrupted(argv, **kwargs):
        if argv[0] == "run":
            if failure == "nonzero":
                return 1, "", "Docker daemon connection lost before response"
            raise TimeoutError("Docker request may still be pending in daemon")
        return original(argv, **kwargs)

    monkeypatch.setattr(role, "_run", interrupted)
    with pytest.raises(TimeoutError if failure == "timeout" else RuntimeError):
        role.launch()
    with pytest.raises(RuntimeError, match="unresolved"):
        role.launch()
    with pytest.raises(RuntimeError, match="unresolved"):
        role.stop()
    assert not role.state["cleanup_verified"]
    assert role.state["launch_uncertain"]
    # Once the actual late container is observed, cleanup is safe and scoped.
    docker.item = {
        "Id": CID,
        "Image": IMAGE,
        "Config": {
            "Labels": {
                "magpie.di.local-owner": role.owner,
                "magpie.di.local-spec": role.spec_hash,
            }
        },
        "State": {"Running": True},
    }
    assert role.stop()["cleanup_verified"]


def test_cancel_during_readiness_preserves_cleanup(tmp_path, monkeypatch):
    role, _ = make(tmp_path, monkeypatch)
    role.launch()
    role.check_cancelled = lambda: (_ for _ in ()).throw(RuntimeError("cancelled"))
    with pytest.raises(RuntimeError, match="cancelled"):
        role.wait_ready()
    assert role.stop()["cleanup_verified"]


def test_collection_failure_is_visible_but_does_not_prevent_cleanup(
    tmp_path, monkeypatch
):
    role, docker = make(tmp_path, monkeypatch)
    role.launch()
    original = docker.run
    monkeypatch.setattr(
        role,
        "_run",
        lambda argv, **kw: (
            (1, "", "logs unavailable") if argv[0] == "logs" else original(argv, **kw)
        ),
    )
    result = role.stop()
    assert result["cleanup_verified"]
    assert "logs unavailable" in result["collection_error"]


def test_router_health_alone_is_not_ready(tmp_path, monkeypatch):
    role, _ = make(tmp_path, monkeypatch, spec("router"))
    monkeypatch.setattr(role, "_http", lambda *a, **kw: {})
    with pytest.raises(RuntimeError, match="completion tokens"):
        role.start()
    assert role.state["status"] == "running"


@pytest.mark.parametrize("args", [
    ["--context-length", "131072", "--ep-size=2"],
    ["--context-length=131072", "--ep-size", "2"],
])
def test_serving_arguments_pass_through_without_runtime_schema_checks(tmp_path, monkeypatch, args):
    role, docker = make(tmp_path, monkeypatch, spec(server_args=args))
    calls = []
    monkeypatch.setattr(role, "_http", lambda path, **kw: calls.append(path) or {})
    assert role.start()["status"] == "ready"
    command = next(command for command in docker.calls if command[0] == "run")
    start = command.index("sglang.launch_server") + 1
    assert command[start:start + len(args)] == args
    assert calls == ["/health"]


def test_two_processes_replay_one_durable_owner(tmp_path):
    import os
    import subprocess
    import sys

    fake = tmp_path / "fake_docker.py"
    fake.write_text("""import json, pathlib, sys, time
root = pathlib.Path(sys.argv[1]); argv = sys.argv[2:]; state = root / 'container.json'
if argv[:2] == ['container', 'inspect']:
    if state.exists(): print('[' + state.read_text() + ']')
    else:
        print('[]'); print('Error: No such container: ' + argv[2], file=sys.stderr); sys.exit(1)
elif argv[:2] == ['image', 'inspect']: print(argv[2])
elif argv[0] == 'run':
    with (root / 'launches').open('a') as stream: stream.write('launch\\n')
    time.sleep(.2)
    labels = dict(argv[i+1].split('=', 1) for i, value in enumerate(argv) if value == '--label')
    image = argv[argv.index('--entrypoint') + 2]
    state.write_text(json.dumps({'Id': 'b'*64, 'Image': image, 'Config': {'Labels': labels}, 'State': {'Running': True}}))
    print('b'*64)
else: raise RuntimeError(argv)
""")
    script = """import json, sys
import Magpie.di.local_role as module
module._DOCKER = [sys.executable, sys.argv[1], sys.argv[2]]
role = module.LocalRole(json.loads(sys.argv[3]), sys.argv[4])
role._ports_free = lambda: None
role.launch()
"""
    args = [
        sys.executable,
        "-c",
        script,
        str(fake),
        str(tmp_path),
        json.dumps(spec()),
        str(tmp_path / "owner"),
    ]
    env = {
        **os.environ,
        "PYTHONPATH": str(__import__("pathlib").Path(__file__).resolve().parents[2]),
    }
    first = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
    )
    second = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
    )
    for process in (first, second):
        out, err = process.communicate(timeout=20)
        assert process.returncode == 0, (out, err)
    assert (tmp_path / "launches").read_text().splitlines() == ["launch"]


def test_recovered_groups_in_same_attempt_have_distinct_owners(tmp_path):
    first = LocalRole(spec(group_id="group-one"), tmp_path / "one")
    second = LocalRole(spec(group_id="group-two"), tmp_path / "two")
    assert first.name != second.name
    assert first.owner != second.owner
    assert first.spec["attempt_id"] == second.spec["attempt_id"] == "1"


def test_round_changes_local_container_identity(tmp_path):
    first = LocalRole(spec(group_id="group", round=0), tmp_path / "first")
    second = LocalRole(spec(group_id="group", round=1), tmp_path / "second")
    assert first.name != second.name
    with pytest.raises(ValueError, match="round"):
        LocalRole(spec(round=-1), tmp_path / "bad")


def test_continuous_health_checks_owned_process_and_http_without_inference(tmp_path, monkeypatch):
    role, docker = make(tmp_path, monkeypatch)
    role.launch()
    role.state['status'] = 'ready'
    role._save()
    calls = []
    monkeypatch.setattr(role, '_http', lambda path, **kwargs: calls.append(path) or {})
    role.check_health()
    assert calls == ['/health']
    docker.item['State']['Running'] = False
    with pytest.raises(RuntimeError, match='no longer running'):
        role.check_health()
    docker.item['State']['Running'] = True
    docker.item['Config']['Labels'] = {}
    with pytest.raises(RuntimeError):
        role.check_health()


@pytest.mark.parametrize("failure", ["missing", "exited", "ownership"])
def test_process_health_skips_busy_http_but_still_rejects_dead_or_foreign_container(tmp_path, monkeypatch, failure):
    role, docker = make(tmp_path, monkeypatch)
    role.launch()
    role.state["status"] = "ready"
    role._save()
    calls = []
    def busy_http(*args, **kwargs):
        calls.append(1)
        raise TimeoutError("cold JIT blocks HTTP")
    monkeypatch.setattr(role, "_http", busy_http)
    role.check_health(check_http=False)
    assert calls == []
    with pytest.raises(TimeoutError, match="cold JIT"):
        role.check_health()
    if failure == "missing":
        docker.item = None
    elif failure == "exited":
        docker.item["State"]["Running"] = False
    else:
        docker.item["Config"]["Labels"] = {}
    with pytest.raises(RuntimeError, match="no longer running|ownership"):
        role.check_health(check_http=False)
    assert calls == [1]
