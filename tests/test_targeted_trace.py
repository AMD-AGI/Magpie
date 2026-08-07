"""Hermetic semantic tests for TargetedKernelTrace."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from Magpie.modes.benchmark.config import (
    BenchmarkConfig,
    ProfilerConfig,
    TorchProfilerConfig,
)
from Magpie.modes.benchmark.targeted_trace import run_targeted_trace_analysis
from Magpie.targeted_trace import (
    RuntimeEvidence,
    TargetSpec,
    TargetedTraceConfig,
    TargetedTraceRecord,
    TargetedTraceRecorder,
    TraceContext,
    TraceIdentity,
    TraceValidationError,
    adapt_torch_profiler_traces,
    iter_trace_events,
    postprocess_trace_dir,
    validate_shard,
)
from Magpie.targeted_trace.sampling import should_sample, stable_key
from Magpie.targeted_trace.schema import TargetedTraceManifest
from Magpie.targeted_trace.schema import canonical_json
from Magpie.targeted_trace.writer import TraceShardWriter, default_shard_path, write_manifest


FIXTURE = Path(__file__).parent / "fixtures" / "targeted_trace" / "torch_trace.json"


class FixtureTensor:
    """Torch-free tensor metadata fixture."""

    shape = (4, 8)
    dtype = "torch.float16"
    device = "cuda:0"
    layout = "torch.strided"
    requires_grad = False

    def stride(self):
        return (8, 1)


def make_record(run_id: str, *, rank: int = 0, pid: int = 10, key: str = "event"):
    return TargetedTraceRecord(
        kind="torch_profiler_kernel",
        stable_event_key=key,
        identity=TraceIdentity(run_id=run_id, target_id="target"),
        context=TraceContext(framework="vllm", rank=rank, pid=pid),
        runtime=RuntimeEvidence(gpu_symbol="kernel", duration_us=4.0),
    )


def test_sampling_is_reproducible_and_seeded():
    key = stable_key({"target": "moe", "shape": [4, 128]}, occurrence=3)
    decisions = [should_sample("run-a", key, 0.5) for _ in range(20)]
    assert len(set(decisions)) == 1
    assert should_sample("run-a", key, 0.0) is False
    assert should_sample("run-a", key, 1.0) is True
    assert stable_key({"shape": [4, 128], "target": "moe"}, occurrence=3) == key


def test_shard_round_trip_checksum_sentinel_and_cap(tmp_path):
    path = default_shard_path(tmp_path, rank=0, pid=10)
    writer = TraceShardWriter(
        path,
        run_id="run-1",
        rank=0,
        pid=10,
        run_seed="seed",
        max_records=1,
    )
    assert writer.submit(make_record("run-1", key="one")) is True
    assert writer.submit(make_record("run-1", key="two")) is False
    receipt = writer.close()

    assert receipt.complete is True
    assert receipt.counters.to_dict() == {
        "seen": 2,
        "sampled": 2,
        "written": 1,
        "dropped": 1,
        "dropped_by_reason": {"cap": 1},
    }
    validated = validate_shard(path, expected_receipt=receipt)
    assert validated.valid is True
    assert validated.event_count == 1


def test_sampling_drop_is_loss_accounted(tmp_path):
    path = default_shard_path(tmp_path, rank=0, pid=10)
    writer = TraceShardWriter(
        path,
        run_id="run-sampling",
        rank=0,
        pid=10,
        run_seed="seed",
        sample_rate=0.0,
    )
    assert writer.submit(make_record("run-sampling")) is False
    receipt = writer.close()

    assert receipt.counters.to_dict() == {
        "seen": 1,
        "sampled": 0,
        "written": 0,
        "dropped": 1,
        "dropped_by_reason": {"sampling": 1},
    }
    assert validate_shard(path).valid is True


def test_writer_refuses_to_overwrite_existing_shard(tmp_path):
    path = default_shard_path(tmp_path, rank=0, pid=10)
    writer = TraceShardWriter(
        path, run_id="first", rank=0, pid=10, run_seed="seed"
    )
    writer.close()

    with pytest.raises(FileExistsError):
        TraceShardWriter(
            path, run_id="second", rank=0, pid=10, run_seed="seed"
        )


def test_corrupt_tail_is_reported_not_silently_skipped(tmp_path):
    path = default_shard_path(tmp_path, rank=0, pid=10)
    writer = TraceShardWriter(
        path, run_id="run-1", rank=0, pid=10, run_seed="seed"
    )
    writer.submit(make_record("run-1"))
    writer.close()
    raw = path.read_bytes()
    path.write_bytes(raw[:-20])

    validated = validate_shard(path)
    assert validated.valid is False
    assert validated.complete is False
    assert any("corrupt JSON tail" in issue for issue in validated.issues)
    assert any("missing end sentinel" in issue for issue in validated.issues)


def test_checksum_tampering_is_reported(tmp_path):
    path = default_shard_path(tmp_path, rank=0, pid=10)
    writer = TraceShardWriter(
        path, run_id="run-1", rank=0, pid=10, run_seed="seed"
    )
    writer.submit(make_record("run-1"))
    writer.close()
    lines = path.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[1])
    event["payload"]["runtime"]["gpu_symbol"] = "tampered"
    lines[1] = canonical_json(event)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    validated = validate_shard(path)
    assert validated.valid is False
    assert any("checksum mismatch" in issue for issue in validated.issues)


def test_nonfinite_envelope_is_reported_without_crashing(tmp_path):
    path = default_shard_path(tmp_path, rank=0, pid=10)
    writer = TraceShardWriter(
        path, run_id="run-nan", rank=0, pid=10, run_seed="seed"
    )
    writer.submit(make_record("run-nan"))
    writer.close()
    lines = path.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[1])
    event["payload"]["runtime"]["duration_us"] = float("nan")
    lines[1] = json.dumps(event, allow_nan=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    validated = validate_shard(path)

    assert validated.valid is False
    assert any("non-finite JSON" in issue for issue in validated.issues)


def test_runtime_capture_records_triton_tensor_grid_meta_and_source(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text("def launch():\n    pass\n", encoding="utf-8")
    recorder = TargetedTraceRecorder(
        tmp_path / "trace",
        run_id="runtime-1",
        run_seed="seed",
        framework="vllm",
        rank=2,
        pid=22,
    )
    assert recorder.record_triton_launch(
        target_id="fused_moe",
        kernel_name="fused_moe_kernel",
        args=(FixtureTensor(),),
        positional_names=("x",),
        kwargs={"BLOCK_SIZE": 256, "num_warps": 8},
        constexpr_names=("BLOCK_SIZE",),
        meta_names=("num_warps",),
        grid=(32, 1, 1),
        source_path=str(source),
        source_line=1,
        source_function="launch",
    )
    receipt = recorder.close()

    records = []
    validated = validate_shard(Path(receipt.path), on_event=records.append)
    assert validated.valid is True
    record = records[0]
    assert record.semantics.tensors[0].shape == (4, 8)
    assert record.semantics.tensors[0].stride == (8, 1)
    assert record.semantics.python_grid["items"] == [32, 1, 1]
    assert record.semantics.constexpr == {"BLOCK_SIZE": 256}
    assert record.semantics.meta == {"num_warps": 8}
    assert record.semantics.source.sha256
    summary = postprocess_trace_dir(tmp_path / "trace")
    assert summary["valid"] is True
    assert summary["events"]["by_target"] == {"fused_moe": 1}


def test_runtime_recorders_merge_rank_shards_into_one_manifest(tmp_path):
    output = tmp_path / "trace"
    for rank, pid in ((0, 20), (1, 21)):
        recorder = TargetedTraceRecorder(
            output,
            run_id="multi-rank",
            run_seed="seed",
            framework="vllm",
            rank=rank,
            pid=pid,
            world_size=2,
        )
        recorder.record_python_hip_launch(
            target_id="aiter.hip_op",
            kernel_name="hip_op",
            args=(FixtureTensor(),),
        )
        recorder.close()

    manifest = TargetedTraceManifest.from_dict(
        json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    )
    assert len(manifest.shards) == 2
    assert manifest.coverage.written == 2
    assert {receipt.rank for receipt in manifest.shards} == {0, 1}
    assert postprocess_trace_dir(output)["valid"] is True


def test_runtime_capture_records_python_hip_grid_meta_scalars_and_source(tmp_path):
    source = tmp_path / "hip_wrapper.py"
    source.write_text("def launch():\n    pass\n", encoding="utf-8")
    recorder = TargetedTraceRecorder(
        tmp_path / "trace",
        run_id="hip-runtime",
        run_seed="seed",
        framework="vllm",
        rank=0,
        pid=23,
    )
    recorder.record_python_hip_launch(
        target_id="aiter.hip_op",
        kernel_name="hip_op",
        args=(FixtureTensor(), 7),
        positional_names=("x", "split_k"),
        kwargs={"algorithm": 3},
        grid=(64, 1, 1),
        meta_names=("algorithm",),
        constexpr_names=("split_k",),
        source_path=str(source),
        source_line=1,
        source_function="launch",
        runtime=RuntimeEvidence(
            gpu_symbol="hip_op_kernel",
            grid=(64, 1, 1),
            block=(256, 1, 1),
        ),
    )
    receipt = recorder.close()

    records = []
    assert validate_shard(Path(receipt.path), on_event=records.append).valid is True
    record = records[0]
    assert record.semantics.tensors[0].shape == (4, 8)
    assert record.semantics.tensors[0].stride == (8, 1)
    assert record.semantics.python_grid["items"] == [64, 1, 1]
    assert record.semantics.constexpr == {"split_k": 7}
    assert record.semantics.meta == {"algorithm": 3}
    assert record.semantics.source.sha256
    assert record.runtime.grid == (64, 1, 1)
    assert record.runtime.block == (256, 1, 1)


def test_torch_profiler_stream_adapter_and_manifest(tmp_path):
    config = TargetedTraceConfig(
        enabled=True,
        run_seed="fixture-seed",
        targets=[
            TargetSpec(
                target_id="aiter.fused_moe",
                name_patterns=("*fused_moe*",),
                package="aiter",
                source={"path": "aiter/moe.py", "line": 42},
            )
        ],
    )
    output = tmp_path / "targeted"
    manifest = adapt_torch_profiler_traces(
        [FIXTURE],
        output,
        config=config,
        run_id="fixture-run",
        framework="vllm",
        framework_version="0.19.1",
        image="example/image:tag",
    )

    assert manifest.reward_eligible is False
    assert manifest.pass_kind == "diagnostic"
    assert manifest.coverage.written == 1
    records = []
    receipt = manifest.shards[0]
    validated = validate_shard(
        Path(receipt.path), expected_receipt=receipt, on_event=records.append
    )
    assert validated.valid is True
    record = records[0]
    assert record.context.rank == 1
    assert record.context.stage == "decode"
    assert record.context.execution_mode == "graph"
    assert record.runtime.grid == (32, 2, 1)
    assert record.runtime.block == (256, 1, 1)
    assert record.runtime.stream == "7"
    assert record.runtime.correlation_id == "99"
    assert record.semantics.tensors[0].shape == (4, 128)
    assert record.semantics.tensors[0].stride == (128, 1)
    assert record.semantics.source.path == "aiter/moe.py"

    summary = postprocess_trace_dir(output, output_path=output / "summary.json")
    assert summary["valid"] is True
    assert summary["streaming"] is True
    assert summary["events"]["by_target"] == {"aiter.fused_moe": 1}


def test_benchmark_adapter_materializes_bounded_valid_evidence(tmp_path):
    targeted = TargetedTraceConfig(
        enabled=True,
        run_seed="benchmark-seed",
        targets=[
            TargetSpec(target_id="moe", name_patterns=("*fused_moe*",))
        ],
    )
    config = BenchmarkConfig(
        framework="vllm",
        model="model",
        profiler=ProfilerConfig(
            torch_profiler=TorchProfilerConfig(enabled=True),
            targeted_trace=targeted,
        ),
    )

    result = run_targeted_trace_analysis(
        config=config,
        trace_files=[FIXTURE],
        workspace=tmp_path,
        run_id="benchmark-run",
        resolved_image="example/image@sha256:fixture",
    )

    assert result["valid"] is True
    assert result["reward_eligible"] is False
    assert result["coverage"] == {
        "seen": 1,
        "sampled": 1,
        "written": 1,
        "dropped": 0,
        "dropped_by_reason": {},
    }
    assert Path(result["manifest_path"]).is_file()
    assert Path(result["summary_path"]).is_file()


def test_adapter_is_byte_stable_for_same_trace_seed_and_run_id(tmp_path):
    config = TargetedTraceConfig(
        enabled=True,
        run_seed="same-seed",
        targets=[TargetSpec(target_id="moe", name_patterns=("*fused_moe*",))],
    )
    outputs = []
    for name in ("first", "second"):
        output = tmp_path / name
        manifest = adapt_torch_profiler_traces(
            [FIXTURE],
            output,
            config=config,
            run_id="same-run",
            framework="vllm",
        )
        outputs.append(Path(manifest.shards[0].path).read_bytes())

    assert outputs[0] == outputs[1]


def test_adapter_surfaces_invalid_input_trace_as_acquisition_failure(tmp_path):
    invalid = tmp_path / "not_a_trace.json"
    invalid.write_text('{"metadata": {}}\n', encoding="utf-8")
    config = TargetedTraceConfig(
        enabled=True,
        targets=[TargetSpec(target_id="moe", name_patterns=("*fused_moe*",))],
    )
    output = tmp_path / "output"
    adapt_torch_profiler_traces(
        [FIXTURE, invalid],
        output,
        config=config,
        run_id="invalid-input",
        framework="vllm",
    )

    summary = postprocess_trace_dir(output)
    assert summary["valid"] is False
    assert summary["integrity_failures_by_reason"]["acquisition"] == 1


def test_torch_trace_iterator_supports_gzip(tmp_path):
    compressed = tmp_path / "rank_1_trace.json.gz"
    with gzip.open(compressed, "wb") as stream:
        stream.write(FIXTURE.read_bytes())
    events = list(iter_trace_events(compressed, chunk_size=37))
    assert [event["name"] for event in events] == [
        "cpu_only_event",
        "triton_fused_moe_kernel",
        "unselected_kernel",
    ]


def test_adapter_ignores_capture_warmup_when_real_trace_exists(tmp_path):
    capture_dir = tmp_path / "capture_traces"
    real_dir = tmp_path / "rank_1"
    capture_dir.mkdir()
    real_dir.mkdir()
    warmup = capture_dir / "warmup.json"
    real = real_dir / "profile.json"
    warmup.write_bytes(FIXTURE.read_bytes())
    real.write_bytes(FIXTURE.read_bytes())
    config = TargetedTraceConfig(
        enabled=True,
        targets=[TargetSpec(target_id="moe", name_patterns=("*fused_moe*",))],
    )

    manifest = adapt_torch_profiler_traces(
        [warmup, real],
        tmp_path / "output",
        config=config,
        run_id="no-warmup",
        framework="vllm",
    )

    assert manifest.coverage.written == 1
    assert manifest.provenance["input_traces"] == [str(real)]


def test_postprocess_streams_shards_without_path_read_text(tmp_path, monkeypatch):
    output = tmp_path / "targeted"
    path = default_shard_path(output, rank=0, pid=10)
    writer = TraceShardWriter(
        path, run_id="run-stream", rank=0, pid=10, run_seed="seed"
    )
    for index in range(1000):
        writer.submit(make_record("run-stream", key=f"event-{index}"))
    receipt = writer.close()
    manifest = TargetedTraceManifest(
        run_id="run-stream",
        acquisition_backend="fixture",
        targets=({"target_id": "target"},),
        shards=(receipt,),
    )
    write_manifest(output / "manifest.json", manifest)

    original = Path.read_text

    def guarded_read_text(self, *args, **kwargs):
        if self.suffix == ".jsonl":
            raise AssertionError("streaming postprocess must not read_text a shard")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    summary = postprocess_trace_dir(output)
    assert summary["valid"] is True
    assert summary["coverage"]["written"] == 1000


def test_postprocess_rejects_shard_not_bound_by_manifest(tmp_path):
    output = tmp_path / "targeted"
    first = TraceShardWriter(
        default_shard_path(output, rank=0, pid=10),
        run_id="bound-run",
        rank=0,
        pid=10,
        run_seed="seed",
    )
    first.submit(make_record("bound-run", pid=10))
    receipt = first.close()
    extra = TraceShardWriter(
        default_shard_path(output, rank=1, pid=11),
        run_id="bound-run",
        rank=1,
        pid=11,
        run_seed="seed",
    )
    extra.submit(make_record("bound-run", rank=1, pid=11))
    extra.close()
    write_manifest(
        output / "manifest.json",
        TargetedTraceManifest(
            run_id="bound-run",
            acquisition_backend="fixture",
            targets=({"target_id": "target"},),
            shards=(receipt,),
        ),
    )

    summary = postprocess_trace_dir(output)

    assert summary["valid"] is False
    assert any("undeclared trace shard" in issue for issue in summary["issues"])


def test_unsupported_manifest_version_fails_fast():
    with pytest.raises(TraceValidationError, match="unsupported schema"):
        TargetedTraceManifest.from_dict(
            {
                "schema_name": "magpie.targeted-kernel-trace",
                "schema_version": "2.0.0",
            }
        )


def test_measurement_and_diagnostic_lanes_are_explicit():
    measurement = BenchmarkConfig(
        framework="vllm",
        model="model",
        run_kind="measurement",
        profiler=ProfilerConfig(
            torch_profiler=TorchProfilerConfig(enabled=False)
        ),
    )
    assert measurement.reward_eligible is True

    diagnostic = BenchmarkConfig(
        framework="vllm",
        model="model",
        profiler=ProfilerConfig(),
    )
    assert diagnostic.run_kind == "diagnostic"
    assert diagnostic.reward_eligible is False

    with pytest.raises(ValueError, match="measurement.*requires"):
        BenchmarkConfig(
            framework="vllm",
            model="model",
            run_kind="measurement",
            profiler=ProfilerConfig(),
        )


def test_targeted_trace_requires_torch_profiler():
    targeted = TargetedTraceConfig(
        enabled=True,
        targets=[TargetSpec(target_id="kernel", name_patterns=("*kernel*",))],
    )
    with pytest.raises(ValueError, match="requires.*torch_profiler"):
        BenchmarkConfig(
            framework="vllm",
            model="model",
            profiler=ProfilerConfig(
                torch_profiler=TorchProfilerConfig(enabled=False),
                targeted_trace=targeted,
            ),
        )
