import os
from types import SimpleNamespace

from Magpie.modes.benchmark import tracelens_inference as module
from Magpie.modes.benchmark.config import BenchmarkConfig
from Magpie.modes.benchmark.tracelens_inference import (
    InferencePhasePick,
    TraceLensInferencePipeline,
)


def make_pipeline(framework="vllm", stages=None):
    config = BenchmarkConfig.from_dict(
        {
            "framework": framework,
            "model": "demo",
            "run_mode": "local",
            "profiler": {
                "tracelens": {
                    "enabled": True,
                    "analysis_mode": "inference",
                    "analysis_stages": stages or ["decode"],
                }
            },
        }
    )
    return TraceLensInferencePipeline(config)


def test_analyze_validation_failures(monkeypatch, tmp_path):
    pipeline = make_pipeline()
    monkeypatch.setattr(module, "ensure_tracelens_installed", lambda: False)
    assert pipeline.analyze(tmp_path, tmp_path)["errors"] == [
        "TraceLens is not installed"
    ]

    monkeypatch.setattr(module, "ensure_tracelens_installed", lambda: True)
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    result = pipeline.analyze(tmp_path, tmp_path)
    assert "Required TraceLens" in result["errors"][0]


def test_analyze_missing_rank_split_and_execution_csv(monkeypatch, tmp_path):
    pipeline = make_pipeline(framework="vllm")
    monkeypatch.setattr(module, "ensure_tracelens_installed", lambda: True)
    monkeypatch.setattr(module.shutil, "which", lambda name: "/bin/tool")
    monkeypatch.setattr(
        pipeline,
        "_select_gpu_arch_platform",
        lambda *args: ("mi300x", None, "platform warning"),
    )
    monkeypatch.setattr(pipeline, "_locate_rank0_trace", lambda path: None)
    result = pipeline.analyze(tmp_path, tmp_path)
    assert "Could not locate" in result["errors"][0]
    assert "platform warning" in result["warnings"]

    trace = tmp_path / "rank0.json"
    trace.write_text("{}")
    monkeypatch.setattr(pipeline, "_locate_rank0_trace", lambda path: trace)
    monkeypatch.setattr(pipeline, "_capture_folder", lambda *args: tmp_path)
    monkeypatch.setattr(pipeline, "_run_splitter", lambda *args: "split failed")
    assert pipeline.analyze(tmp_path, tmp_path)["errors"][-1] == "split failed"

    monkeypatch.setattr(pipeline, "_run_splitter", lambda *args: None)
    monkeypatch.setattr(
        pipeline, "_validate_trace_layout", lambda *args: ["layout warning"]
    )
    result = pipeline.analyze(tmp_path, tmp_path / "out")
    assert "Missing TraceLens split CSV" in result["errors"][-1]
    assert "layout warning" in result["warnings"]


def test_analyze_stage_reports_and_skips(monkeypatch, tmp_path):
    pipeline = make_pipeline(stages=["decode", "prefill"])
    monkeypatch.setattr(module, "ensure_tracelens_installed", lambda: True)
    monkeypatch.setattr(module.shutil, "which", lambda name: "/bin/tool")
    monkeypatch.setattr(
        pipeline, "_select_gpu_arch_platform", lambda *args: ("mi300x", "mi300x", None)
    )
    trace = tmp_path / "rank0.json"
    trace.write_text("{}")
    monkeypatch.setattr(pipeline, "_locate_rank0_trace", lambda path: trace)
    monkeypatch.setattr(pipeline, "_capture_folder", lambda *args: tmp_path)

    def split(_trace, split_dir):
        split_dir.mkdir(parents=True)
        (split_dir / "execution_details.csv").write_text("stage,output_path\n")
        return None

    monkeypatch.setattr(pipeline, "_run_splitter", split)
    monkeypatch.setattr(pipeline, "_validate_trace_layout", lambda *args: [])
    pick = InferencePhasePick(
        stage="decode",
        csv_kind="execution",
        batch_size=8,
        trace_path=trace,
        output_label="decode_bs8",
    )
    monkeypatch.setattr(
        pipeline, "_pick_largest_batch_traces", lambda path: {"decode": pick}
    )
    monkeypatch.setattr(
        pipeline,
        "_run_perf_report",
        lambda **kwargs: {"files": ["decode.csv"], "error": "partial"},
    )
    result = pipeline.analyze(tmp_path, tmp_path / "out")
    assert result["output_files"] == ["decode.csv"]
    assert result["errors"] == ["partial"]
    assert any("prefill: no candidate" in warning for warning in result["warnings"])


def test_sglang_fallback_creates_execution_csv(monkeypatch, tmp_path):
    pipeline = make_pipeline(framework="sglang")
    monkeypatch.setattr(module, "ensure_tracelens_installed", lambda: True)
    monkeypatch.setattr(module.shutil, "which", lambda name: "/bin/tool")
    monkeypatch.setattr(
        pipeline, "_select_gpu_arch_platform", lambda *args: (None, None, None)
    )
    trace = tmp_path / "rank0.json"
    trace.write_text("{}")
    monkeypatch.setattr(pipeline, "_locate_rank0_trace", lambda path: trace)
    monkeypatch.setattr(pipeline, "_capture_folder", lambda *args: tmp_path)
    monkeypatch.setattr(pipeline, "_run_splitter", lambda *args: None)
    monkeypatch.setattr(pipeline, "_validate_trace_layout", lambda *args: [])

    def fallback(_trace, split_dir):
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "execution_details.csv").write_text("stage,output_path\n")
        return ["fallback used"]

    monkeypatch.setattr(pipeline, "_split_sglang_step_markers", fallback)
    monkeypatch.setattr(pipeline, "_pick_largest_batch_traces", lambda path: {})
    result = pipeline.analyze(tmp_path, tmp_path / "out")
    assert any("fallback" in warning for warning in result["warnings"])


def test_container_paths_rewrite_and_subprocess_environment(monkeypatch, tmp_path):
    pipeline = make_pipeline()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    nested = workspace / "trace" / "rank0.json"
    nested.parent.mkdir()
    nested.write_text("{}")
    assert pipeline._container_path(workspace, workspace) == "/workspace"
    assert pipeline._container_path(nested, workspace) == "/workspace/trace/rank0.json"
    try:
        pipeline._container_path(tmp_path / "outside", workspace)
    except ValueError as error:
        assert "inside workspace" in str(error)

    csv_file = workspace / "execution_details.csv"
    csv_file.write_text("stage,output_path\ndecode,/workspace/trace/rank0.json\n")
    pipeline._rewrite_container_split_paths(csv_file, workspace)
    assert str(workspace / "trace" / "rank0.json") in csv_file.read_text()
    pipeline.tl_extension = "extension.module"
    assert pipeline._subprocess_env()["TL_EXTENSION"] == "extension.module"


def _sglang_graph_captures(torch_trace, batches=(1, 8), ranks=(0, 1)):
    source = torch_trace / "graph_capture_profile"
    source.mkdir(parents=True)
    for batch in batches:
        for rank in ranks:
            name = f"DecodeCudaGraphRunner_bs_{batch}_rank{rank}.json.gz"
            (source / name).write_bytes(b"rank %d" % rank)
    return source


def test_capture_folder_links_sglang_rank0_graph_captures(tmp_path):
    torch_trace = tmp_path / "torch_trace"
    source = _sglang_graph_captures(torch_trace)
    (source / "cuda_graph_runner_memory_usage.pickle").write_bytes(b"")
    rank0_trace = torch_trace / "demo-TP-0.trace.json.gz"
    rank0_trace.write_bytes(b"")
    pipeline = make_pipeline(framework="sglang")

    folder = pipeline._capture_folder(torch_trace, rank0_trace)

    assert folder == torch_trace / "capture_traces"
    assert sorted(path.name for path in folder.iterdir()) == [
        "DecodeCudaGraphRunner_bs_1_rank0.json.gz",
        "DecodeCudaGraphRunner_bs_8_rank0.json.gz",
    ]
    link = folder / "DecodeCudaGraphRunner_bs_8_rank0.json.gz"
    assert link.read_bytes() == b"rank 0"
    if link.is_symlink():
        assert not os.path.isabs(os.readlink(link))
    assert len(list(source.iterdir())) == 5
    assert pipeline._validate_trace_layout(folder, tmp_path) == [
        f"execution_details.csv missing under split dir: {tmp_path}"
    ]
    assert pipeline._capture_folder(torch_trace, rank0_trace) == folder
    assert len(list(folder.iterdir())) == 2


def test_capture_folder_keeps_existing_capture_traces(tmp_path):
    torch_trace = tmp_path / "torch_trace"
    _sglang_graph_captures(torch_trace)
    capture = torch_trace / "capture_traces"
    capture.mkdir()
    (capture / "graph_capture_rank_0.json.gz").write_bytes(b"")
    pipeline = make_pipeline(framework="sglang")

    folder = pipeline._capture_folder(torch_trace, torch_trace / "demo.json.gz")

    assert folder == capture
    assert [path.name for path in folder.iterdir()] == [
        "graph_capture_rank_0.json.gz"
    ]


def test_capture_folder_fills_existing_empty_capture_traces(tmp_path):
    torch_trace = tmp_path / "torch_trace"
    _sglang_graph_captures(torch_trace)
    capture = torch_trace / "capture_traces"
    capture.mkdir()
    rank0_trace = torch_trace / "demo-TP-0.trace.json.gz"
    rank0_trace.write_bytes(b"")
    pipeline = make_pipeline(framework="sglang")

    folder = pipeline._capture_folder(torch_trace, rank0_trace)

    assert folder == capture
    assert sorted(path.name for path in folder.iterdir()) == [
        "DecodeCudaGraphRunner_bs_1_rank0.json.gz",
        "DecodeCudaGraphRunner_bs_8_rank0.json.gz",
    ]
    assert pipeline._validate_trace_layout(folder, tmp_path) == [
        f"execution_details.csv missing under split dir: {tmp_path}"
    ]


def test_capture_folder_without_rank0_graph_captures(tmp_path):
    torch_trace = tmp_path / "torch_trace"
    _sglang_graph_captures(torch_trace, ranks=(1,))
    pipeline = make_pipeline(framework="sglang")

    folder = pipeline._capture_folder(torch_trace, torch_trace / "demo.json.gz")

    assert folder == torch_trace / "capture_traces"
    assert not folder.exists()
    assert pipeline._validate_trace_layout(folder, tmp_path)[0].startswith(
        "capture_traces directory is missing or empty"
    )
