from Magpie.modes.benchmark import tracelens_inference as module
from Magpie.modes.benchmark.config import BenchmarkConfig
from Magpie.modes.benchmark.tracelens_inference import (
    BACKUP_SUFFIX,
    TraceLensInferencePipeline,
)

CLIENT = """            extra_body={
                "num_steps": 1,
                "merge_profiles": True,
                "profile_by_stage": True,
            },
"""
LIB = 'if profile; then\n    num_prompts="$max_concurrency"\nfi\n'


def _files(tmp_path, monkeypatch):
    client = tmp_path / "utils" / "bench_serving" / "benchmark_serving.py"
    lib = tmp_path / "benchmarks" / "benchmark_lib.sh"
    for path, text in ((client, CLIENT), (lib, LIB)):
        path.parent.mkdir(parents=True)
        path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(module, "resolve_benchmark_serving", lambda root: client)
    monkeypatch.setattr(module, "resolve_inferencex_root", lambda root: str(tmp_path))
    return client, lib


def _pipeline(tmp_path):
    config = BenchmarkConfig.from_dict(
        {
            "framework": "sglang",
            "model": "demo",
            "run_mode": "local",
            "inferencex_path": str(tmp_path),
            "profiler": {
                "tracelens": {"enabled": True, "analysis_mode": "inference"}
            },
        }
    )
    return TraceLensInferencePipeline(config)


def _patch(pipeline, max_iters, delay_iters):
    result = {"patched_files": [], "warnings": []}
    pipeline._patch_benchmark_lib(result)
    pipeline._patch_sglang_benchmark_serving(max_iters, delay_iters, result)
    return result


def test_interrupted_patch_is_restored_before_a_different_window(tmp_path, monkeypatch):
    client, lib = _files(tmp_path, monkeypatch)
    _patch(_pipeline(tmp_path), 1024, 2560)
    assert '"start_step": 2560, "num_steps": 1024' in client.read_text()

    pipeline = _pipeline(tmp_path)
    result = _patch(pipeline, 128, 256)

    text = client.read_text()
    assert '"start_step": 256, "num_steps": 128' in text
    assert "2560" not in text
    assert result["patched_files"] == [str(lib), str(client)]
    assert len(result["warnings"]) == 2
    assert all("earlier run" in warning for warning in result["warnings"])

    restored = pipeline.restore()
    assert restored["restored_files"] == [str(client), str(lib)]
    assert client.read_text() == CLIENT
    assert lib.read_text() == LIB
    assert not client.with_name(client.name + BACKUP_SUFFIX).exists()
    assert not lib.with_name(lib.name + BACKUP_SUFFIX).exists()


def test_interrupted_patch_with_the_same_window_is_restored(tmp_path, monkeypatch):
    client, lib = _files(tmp_path, monkeypatch)
    _patch(_pipeline(tmp_path), 1024, 2560)

    pipeline = _pipeline(tmp_path)
    result = _patch(pipeline, 1024, 2560)

    assert result["patched_files"] == [str(lib), str(client)]
    pipeline.restore()
    assert client.read_text() == CLIENT
    assert lib.read_text() == LIB


def test_repeated_patch_keeps_its_own_backup(tmp_path, monkeypatch):
    client, lib = _files(tmp_path, monkeypatch)
    pipeline = _pipeline(tmp_path)
    _patch(pipeline, 1024, 2560)

    result = _patch(pipeline, 1024, 2560)

    assert result == {"patched_files": [], "warnings": []}
    pipeline.restore()
    assert client.read_text() == CLIENT
    assert lib.read_text() == LIB
