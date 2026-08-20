import gzip
import json

from Magpie.modes.benchmark.config import GapAnalysisConfig
from Magpie.modes.benchmark.gap_analysis import (
    GapAnalysisResult,
    GapAnalyzer,
    KernelStat,
    RankResult,
    _extract_rank,
)


def trace_events():
    return [
        {
            "cat": "cpu_op",
            "name": "aten::mm",
            "ts": 0,
            "dur": 5,
            "args": {"External id": 1, "Input Dims": [[2, 3], [3, 4]]},
        },
        {
            "cat": "kernel",
            "name": "gemm",
            "ts": 10,
            "dur": 20,
            "args": {"External id": 1},
        },
        {
            "cat": "kernel",
            "name": "gemm",
            "ts": 35,
            "dur": 10,
            "args": {"External id": 1},
        },
        {"cat": "ignored", "name": "noise", "ts": 50, "dur": 5},
        {"name": "metadata"},
    ]


def write_trace(path, events=None, compressed=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"traceEvents": events if events is not None else trace_events(), "meta": 1}
    if compressed:
        with gzip.open(path, "wt") as handle:
            json.dump(data, handle)
    else:
        path.write_text(json.dumps(data))
    return path


def config(**kwargs):
    values = {
        "enabled": True,
        "categories": ["kernel"],
        "ignore_categories": ["ignored"],
        "trace_start_pct": 0,
        "trace_end_pct": 100,
        "top_k": 10,
    }
    values.update(kwargs)
    return GapAnalysisConfig(**values)


def test_kernel_stats_and_result_serialization(tmp_path):
    stat = KernelStat(
        "gemm",
        total_duration_us=30,
        calls=2,
        durations_us=[10, 20],
        shapes=["x", "x", "y"],
    )
    assert stat.avg_us == 15
    assert stat.min_us == 10
    assert stat.max_us == 20
    assert stat.std_us > 0
    assert stat.unique_shapes == "x; y"
    result = GapAnalysisResult(
        config={"top_k": 10},
        rank_results=[RankResult(0, "trace", 30, [stat])],
        merged_kernels=[stat],
        total_duration_us=30,
        clamped_trace_paths=[tmp_path / "clamped.json.gz"],
    )
    assert result.to_dict()["top_kernels"][0]["pct_total"] == 100
    csv_path = result.to_csv(tmp_path / "summary.csv")
    assert "gemm" in csv_path.read_text()
    rank_paths = result.to_rank_csv(tmp_path)
    assert "gemm" in rank_paths[0].read_text()


def test_gap_analyzer_end_to_end_and_clamping(tmp_path):
    trace_dir = tmp_path / "traces"
    first = write_trace(trace_dir / "trace-rank-0.pt.trace.json.gz")
    second = write_trace(
        trace_dir / "rank_1" / "trace.pt.trace.json.gz",
        trace_events() + [{"cat": "kernel", "name": "other", "ts": 60, "dur": 40}],
    )
    analyzer = GapAnalyzer(
        config(trace_start_pct=10, trace_end_pct=90, min_duration_us=5)
    )
    found = analyzer.detect_trace_files(trace_dir)
    assert found == [(0, first), (1, second)]
    result = analyzer.analyze(trace_dir)
    assert len(result.rank_results) == 2
    assert result.merged_kernels[0].name == "gemm"
    assert result.total_duration_us > 0
    paths = analyzer.generate_clamped_traces(trace_dir, tmp_path / "clamped")
    assert len(paths) == 2
    with gzip.open(paths[0], "rt") as handle:
        clamped = json.load(handle)
    assert clamped["traceEvents"]


def test_gap_analyzer_missing_empty_and_bad_traces(tmp_path):
    analyzer = GapAnalyzer(config())
    assert "not found" in analyzer.analyze(tmp_path / "missing").errors[0]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "No trace" in analyzer.analyze(empty).errors[0]
    bad = empty / "trace-rank-0.pt.trace.json"
    bad.write_text("{")
    result = analyzer.analyze(empty)
    assert "Failed to analyze" in result.errors[0]
    assert analyzer.generate_clamped_traces(empty) == []


def test_gap_analysis_helpers_and_formats(tmp_path):
    analyzer = GapAnalyzer(config(trace_start_pct=25, trace_end_pct=75))
    events = trace_events()
    shapes = analyzer._build_shape_map(events)
    assert shapes == {1: "[2,3]x[3,4]"}
    assert analyzer._format_input_dims([]) == ""
    assert analyzer._format_input_dims([[], "bad"]) == ""
    window = analyzer._apply_time_window(events)
    assert len(window) < len(events)
    assert analyzer._apply_time_window([{"name": "no timestamps"}]) == [
        {"name": "no timestamps"}
    ]
    filtered = analyzer._filter_by_category(events, shapes)
    assert [item[0] for item in filtered] == ["gemm", "gemm"]
    assert filtered[0][2] == "[2,3]x[3,4]"
    stats = analyzer._aggregate_stats(filtered)
    assert stats[0].calls == 2
    rank = analyzer._analyze_single_rank(0, tmp_path / "trace", events)
    assert rank.kernels[0].name == "gemm"
    merged = analyzer._merge_ranks([rank, rank])
    assert merged[0].calls == rank.kernels[0].calls * 2


def test_trace_loading_list_fallback_and_rank_extraction(tmp_path):
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps(trace_events()))
    data, events = GapAnalyzer._load_trace_data(plain)
    assert isinstance(data, list)
    assert len(events) == len(trace_events())
    odd = tmp_path / "odd.json"
    odd.write_text("42")
    assert GapAnalyzer._load_trace_data(odd)[1] == []
    assert _extract_rank(tmp_path / "trace-rank-7.json") == 7
    assert _extract_rank(tmp_path / "pp0_dp2_tp3" / "trace.json") == 3
    assert _extract_rank(tmp_path / "trace.json") is None


def test_generate_clamped_trace_no_timestamps_and_list_output(tmp_path):
    analyzer = GapAnalyzer(config(trace_start_pct=0, trace_end_pct=50))
    assert (
        analyzer._generate_clamped_trace([], [{"name": "meta"}], tmp_path / "x.json")
        is None
    )
    path = analyzer._generate_clamped_trace(
        trace_events(), trace_events(), tmp_path / "flat.json", output_dir=tmp_path
    )
    with gzip.open(path, "rt") as handle:
        assert isinstance(json.load(handle), list)
