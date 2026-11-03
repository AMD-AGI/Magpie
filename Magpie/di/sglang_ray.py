"""Pinned SGLang Ray HTTP bridge using an existing DI placement group."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import socket
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def scheduler_names(pg_id, host, bundle_indices, tp):
    """Match SGLang v0.5.18's custom-group scheduler names (DP=1)."""
    return [
        f"sglang_scheduler_node{host}_dp0_pp{rank // tp}_tp{rank % tp}"
        f"_pg{pg_id[:8]}_bundle{bundle}"
        for rank, bundle in enumerate(bundle_indices)
    ]


def _write(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def launch(pg_name, pg_id, namespace, bundle_indices, journal_path, argv):
    """Run stock HTTP serving with SGLang-owned GPU scheduler actors."""
    import ray
    from ray.util.placement_group import get_placement_group

    from Magpie.remote.tasks import _clear_hidden_gpus

    # Same visibility preparation as the ordinary zero-GPU Ray entry task.
    _clear_hidden_gpus()
    os.environ["SGLANG_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
    owns_connection = not ray.is_initialized()
    if owns_connection:
        ray.init(address="auto", namespace=namespace)
    actors, record = [], None
    path = Path(journal_path)
    try:
        if ray.get_runtime_context().namespace != namespace:
            raise ValueError("SGLang frontend has a different Ray namespace")
        pg = get_placement_group(pg_name)
        if pg.id.hex() != pg_id:
            raise ValueError("SGLang frontend placement group identity changed")
        try:
            installed = version("sglang")
        except PackageNotFoundError as error:
            raise RuntimeError("DI Ray HTTP adapter requires SGLang 0.5.18") from error
        if installed != "0.5.18":
            raise RuntimeError(
                f"DI Ray HTTP adapter requires SGLang 0.5.18; found {installed}"
            )
        from sglang.srt.plugins import load_plugins
        from sglang.srt.server_args import prepare_server_args

        load_plugins()
        server_args = prepare_server_args(argv)
        host = ray.util.get_node_ip_address()
        # PortArgs reads this before RayEngine computes scheduler rendezvous.
        if server_args.nnodes > 1 and not server_args.dist_init_addr:
            family = socket.AF_INET6 if ":" in host else socket.AF_INET
            with socket.socket(family) as listener:
                listener.bind((host, 0))
                port = listener.getsockname()[1]
            server_args.dist_init_addr = (
                f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
            )
        if (
            not server_args.use_ray
            or server_args.dp_size != 1
            or server_args.encoder_only
            or server_args.smg_grpc_mode
            or server_args.grpc_mode
            or server_args.grpc_port is not None
        ):
            raise ValueError("DI frontend requires SGLang Ray HTTP serving with DP=1")
        if len(bundle_indices) != server_args.tp_size * server_args.pp_size:
            raise ValueError("SGLang rank bundle count differs from TP * PP")
        from sglang.srt.entrypoints.engine import (
            init_tokenizer_manager,
            run_detokenizer_process,
            run_scheduler_process,
        )
        from sglang.srt.entrypoints.http_server import (
            _execute_server_warmup,
            _setup_and_run_http_server,
        )
        from sglang.srt.ray.engine import RayEngine
        from sglang.srt.utils import kill_process_tree

        try:
            signature = inspect.signature(RayEngine._launch_subprocesses)
            if "placement_group" not in signature.parameters:
                raise TypeError("missing placement_group")
            signature.bind(
                server_args,
                init_tokenizer_manager_func=init_tokenizer_manager,
                run_scheduler_process_func=run_scheduler_process,
                run_detokenizer_process_func=run_detokenizer_process,
                placement_group=pg,
            )
            inspect.signature(_setup_and_run_http_server).bind(
                server_args,
                None,
                None,
                None,
                [],
                None,
                execute_warmup_func=_execute_server_warmup,
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "SGLang Ray HTTP API is incompatible: explicit placement-group HTTP support is required (v0.5.18)"
            ) from error
        job_id = ray.get_runtime_context().get_job_id()
        record = {
            "job_id": job_id.hex() if hasattr(job_id, "hex") else str(job_id),
            "pg_id": pg_id,
            "namespace": namespace,
            "scheduler_names": scheduler_names(
                pg_id, host, bundle_indices, server_args.tp_size
            ),
            "actor_ids": [],
            "status": "starting",
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        _write(path, record)
        try:
            components = RayEngine._launch_subprocesses(
                server_args,
                init_tokenizer_manager_func=init_tokenizer_manager,
                run_scheduler_process_func=run_scheduler_process,
                run_detokenizer_process_func=run_detokenizer_process,
                placement_group=pg,
            )
            if not isinstance(components, tuple) or len(components) != 6:
                raise RuntimeError(
                    "SGLang Ray HTTP startup result is incompatible (v0.5.18)"
                )
            tokenizer, templates, ports, schedulers, watchdog, _daemons = components
            actors = schedulers.scheduler_actors
            record["actor_ids"] = [actor._actor_id.hex() for actor in actors]
            _write(path, record)
            try:
                if len(actors) != len(record["scheduler_names"]):
                    raise ValueError("scheduler count differs from reserved ranks")
                for name, actor in zip(record["scheduler_names"], actors):
                    registered = ray.get_actor(name, namespace=namespace)
                    if registered._actor_id.hex() != actor._actor_id.hex():
                        raise ValueError("scheduler name identifies a different actor")
            except (ValueError, AttributeError) as error:
                record["compatibility_error"] = str(error)
                _write(path, record)
                raise RuntimeError(
                    "SGLang 0.5.18 scheduler ownership API is incompatible"
                ) from error
            record["status"] = "http_running"
            _write(path, record)
            _setup_and_run_http_server(
                server_args,
                tokenizer,
                templates,
                ports,
                schedulers.scheduler_infos,
                watchdog,
                execute_warmup_func=_execute_server_warmup,
            )
        finally:
            errors = []
            for actor in actors:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception as error:  # noqa: BLE001 - attempt every owned scheduler
                    errors.append(str(error))
            try:
                kill_process_tree(os.getpid(), include_parent=False)
            finally:
                # Requests are not a cleanup receipt: the session verifies actor
                # termination before releasing the shared placement group.
                record.update(
                    status="exited", cleanup_requested=True, cleanup_errors=errors
                )
                _write(path, record)
            if errors:
                raise RuntimeError(
                    "SGLang scheduler cleanup requests failed: " + "; ".join(errors)
                )
    finally:
        if owns_connection:
            ray.shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pg-name", required=True)
    parser.add_argument("--pg-id", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--bundle-indices", required=True)
    parser.add_argument("--journal-path", required=True)
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    server_args = args.server_args
    if server_args[:1] == ["--"]:
        server_args = server_args[1:]
    indices = [int(value) for value in args.bundle_indices.split(",")]
    launch(
        args.pg_name,
        args.pg_id,
        args.namespace,
        indices,
        args.journal_path,
        server_args,
    )


if __name__ == "__main__":
    main()
