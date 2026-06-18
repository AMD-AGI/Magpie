"""Unit tests for the offline diffusion (xDiT) benchmark path.

Covers the InferenceX-free offline runner added for xDiT:
  - BenchmarkConfig offline validation / defaults / round-trip
  - ResultParser.parse_diffusion_result (timings.json -> diffusion metrics)
  - BenchmarkMode._build_offline_command (output_directory/profile injection)
  - benchmark_images.yaml xdit entry
"""

import json

import pytest
import yaml

from Magpie.modes.benchmark.benchmarker import BenchmarkMode
from Magpie.modes.benchmark.config import (
    SUPPORTED_FRAMEWORKS,
    BenchmarkConfig,
    BenchmarkFramework,
    ProfilerConfig,
    TorchProfilerConfig,
)
from Magpie.modes.benchmark.result import ResultParser


# --------------------------------------------------------------------------- #
# BenchmarkConfig — offline framework handling
# --------------------------------------------------------------------------- #
class TestOfflineBenchmarkConfig:
    def test_xdit_is_registered_framework(self):
        assert BenchmarkFramework.XDIT.value == "xdit"
        assert "xdit" in SUPPORTED_FRAMEWORKS

    def test_xdit_requires_run_cmd(self):
        with pytest.raises(ValueError, match="run_cmd"):
            BenchmarkConfig(framework="xdit", model="flux")

    def test_xdit_minimal_valid_config(self):
        cfg = BenchmarkConfig(
            framework="xdit", model="flux", run_cmd="xdit --model flux"
        )
        assert cfg.is_offline is True
        assert cfg.is_command_launched is True
        assert cfg.is_diffusion is True
        assert cfg.framework == "xdit"
        assert cfg.run_cmd == "xdit --model flux"

    def test_llm_framework_axes_are_defaults(self):
        cfg = BenchmarkConfig(framework="sglang", model="demo")
        assert cfg.is_offline is False
        assert cfg.is_command_launched is False
        assert cfg.is_diffusion is False

    def test_xdit_docker_run_mode_downgraded_to_local(self):
        cfg = BenchmarkConfig(
            framework="xdit",
            model="flux",
            run_cmd="xdit --model flux",
            run_mode="docker",
        )
        assert cfg.run_mode == "local"

    def test_xdit_does_not_inject_llm_env_defaults(self):
        cfg = BenchmarkConfig(
            framework="xdit", model="flux", run_cmd="xdit --model flux"
        )
        # No TP/CONC/ISL/OSL — those are LLM serving knobs.
        assert cfg.envs == {}

    def test_online_framework_still_gets_llm_env_defaults(self):
        cfg = BenchmarkConfig(framework="sglang", model="demo")
        assert cfg.is_offline is False
        assert cfg.envs["TP"] == 1
        assert cfg.envs["CONC"] == 32

    def test_xdit_name_is_case_insensitive(self):
        cfg = BenchmarkConfig(
            framework="XDIT", model="flux", run_cmd="xdit --model flux"
        )
        assert cfg.framework == "xdit"

    def test_unknown_framework_still_rejected(self):
        with pytest.raises(ValueError, match="Unsupported framework"):
            BenchmarkConfig(framework="nope", model="demo")

    def test_to_dict_and_from_dict_roundtrip_run_cmd(self):
        cfg = BenchmarkConfig(
            framework="xdit",
            model="flux",
            run_cmd="xdit --model flux --ulysses_degree 2",
        )
        d = cfg.to_dict()
        assert d["run_cmd"] == "xdit --model flux --ulysses_degree 2"
        restored = BenchmarkConfig.from_dict(d)
        assert restored.run_cmd == cfg.run_cmd
        assert restored.is_offline is True

    def test_from_dict_offline_without_run_cmd_raises(self):
        with pytest.raises(ValueError, match="run_cmd"):
            BenchmarkConfig.from_dict({"framework": "xdit", "model": "flux"})


# --------------------------------------------------------------------------- #
# ResultParser.parse_diffusion_result
# --------------------------------------------------------------------------- #
class TestParseDiffusionResult:
    def _write_timings(self, tmp_path, timings):
        f = tmp_path / "timings.json"
        f.write_text(json.dumps(timings), encoding="utf-8")
        return f

    def test_missing_file_returns_failure(self, tmp_path):
        result = ResultParser.parse_diffusion_result(tmp_path / "missing.json")
        assert result.success is False
        assert result.errors

    def test_empty_timings_returns_failure(self, tmp_path):
        f = self._write_timings(tmp_path, [])
        result = ResultParser.parse_diffusion_result(f)
        assert result.success is False
        assert result.errors

    def test_malformed_json_returns_failure(self, tmp_path):
        f = tmp_path / "timings.json"
        f.write_text("{not json", encoding="utf-8")
        result = ResultParser.parse_diffusion_result(f)
        assert result.success is False
        assert result.errors

    def test_single_timing_no_batch(self, tmp_path):
        f = self._write_timings(tmp_path, [2.0])
        result = ResultParser.parse_diffusion_result(
            f, framework="xdit", model="flux", run_cmd="xdit --model flux"
        )
        assert result.success is True
        # latency/image = mean = 2.0s; images/sec = 1/2 = 0.5
        assert result.raw_result["latency_per_image_s"] == pytest.approx(2.0)
        assert result.raw_result["images_per_sec"] == pytest.approx(0.5)
        assert result.raw_result["batch_size"] == 1

    def test_batch_size_parsed_from_run_cmd(self, tmp_path):
        f = self._write_timings(tmp_path, [2.0])
        result = ResultParser.parse_diffusion_result(
            f, run_cmd="xdit --model flux --batch_size 4"
        )
        assert result.raw_result["batch_size"] == 4
        # latency/image = mean/batch = 2.0/4 = 0.5; images/sec = 4/2 = 2.0
        assert result.raw_result["latency_per_image_s"] == pytest.approx(0.5)
        assert result.raw_result["images_per_sec"] == pytest.approx(2.0)

    def test_batch_size_dash_alias(self, tmp_path):
        f = self._write_timings(tmp_path, [1.0])
        result = ResultParser.parse_diffusion_result(
            f, run_cmd="xdit --batch-size 2 --model flux"
        )
        assert result.raw_result["batch_size"] == 2

    def test_steps_per_sec_from_num_inference_steps(self, tmp_path):
        f = self._write_timings(tmp_path, [2.0])
        result = ResultParser.parse_diffusion_result(
            f, run_cmd="xdit --model flux --num_inference_steps 50"
        )
        # steps/sec = 50 / mean(2.0) = 25
        assert result.raw_result["steps_per_sec"] == pytest.approx(25.0)
        assert result.raw_result["num_inference_steps"] == 50

    def test_steps_per_sec_zero_when_steps_absent(self, tmp_path):
        f = self._write_timings(tmp_path, [2.0])
        result = ResultParser.parse_diffusion_result(f, run_cmd="xdit --model flux")
        assert result.raw_result["steps_per_sec"] == 0.0
        assert result.raw_result["num_inference_steps"] is None

    def test_images_per_sec_mapped_onto_output_throughput(self, tmp_path):
        """The gain/objective machinery ranks on output_throughput; for xDiT
        that must carry images/sec (higher = better)."""
        f = self._write_timings(tmp_path, [2.0])
        result = ResultParser.parse_diffusion_result(
            f, run_cmd="xdit --model flux --batch_size 2"
        )
        assert result.throughput.output_throughput == pytest.approx(1.0)
        assert result.throughput.request_throughput == pytest.approx(1.0)

    def test_latency_mapped_onto_e2el_ms(self, tmp_path):
        f = self._write_timings(tmp_path, [1.0, 1.0])
        result = ResultParser.parse_diffusion_result(f, run_cmd="xdit --model flux")
        # mean iteration = 1.0s -> 1000 ms on e2el_mean
        assert result.latency.e2el_mean == pytest.approx(1000.0)

    def test_does_not_double_drop_first_iteration(self, tmp_path):
        """xDiT already drops the first iteration in timings.json; the parser
        must average ALL entries it receives (no second drop)."""
        f = self._write_timings(tmp_path, [1.0, 3.0])
        result = ResultParser.parse_diffusion_result(f, run_cmd="xdit --model flux")
        assert result.raw_result["iterations"] == 2
        assert result.raw_result["mean_iteration_s"] == pytest.approx(2.0)

    def test_non_numeric_entries_filtered(self, tmp_path):
        f = self._write_timings(tmp_path, [1.0, "bad", 3.0, None])
        result = ResultParser.parse_diffusion_result(f, run_cmd="xdit --model flux")
        assert result.success is True
        assert result.raw_result["iterations"] == 2

    def test_invalid_batch_size_falls_back_to_one(self, tmp_path):
        f = self._write_timings(tmp_path, [2.0])
        result = ResultParser.parse_diffusion_result(
            f, run_cmd="xdit --batch_size notanint --model flux"
        )
        assert result.raw_result["batch_size"] == 1


# --------------------------------------------------------------------------- #
# BenchmarkMode._build_offline_command
# --------------------------------------------------------------------------- #
class TestBuildOfflineCommand:
    def _mode(self, run_cmd, profile=False):
        cfg = BenchmarkConfig(
            framework="xdit",
            model="flux",
            run_cmd=run_cmd,
            run_mode="local",
            profiler=ProfilerConfig(
                torch_profiler=TorchProfilerConfig(enabled=profile)
            ),
        )
        return BenchmarkMode(config=cfg, output_dir="/tmp")

    def test_injects_output_directory(self, tmp_path):
        mode = self._mode("xdit --model flux --ulysses_degree 2")
        cmd = mode._build_offline_command(tmp_path)
        assert cmd[-2:] == ["--output_directory", str(tmp_path)]
        assert "xdit" in cmd and "--ulysses_degree" in cmd

    def test_strips_user_output_directory_space_form(self, tmp_path):
        mode = self._mode("xdit --model flux --output_directory /user/dir")
        cmd = mode._build_offline_command(tmp_path)
        assert "/user/dir" not in cmd
        assert cmd.count("--output_directory") == 1
        assert cmd[-1] == str(tmp_path)

    def test_strips_user_output_directory_equals_form(self, tmp_path):
        mode = self._mode("xdit --model flux --output_directory=/user/dir")
        cmd = mode._build_offline_command(tmp_path)
        assert "/user/dir" not in cmd
        assert "--output_directory=/user/dir" not in cmd
        assert cmd.count("--output_directory") == 1

    def test_strips_user_output_directory_dash_alias(self, tmp_path):
        mode = self._mode("xdit --model flux --output-directory /user/dir")
        cmd = mode._build_offline_command(tmp_path)
        assert "/user/dir" not in cmd

    def test_profile_off_omits_profile_flag(self, tmp_path):
        mode = self._mode("xdit --model flux", profile=False)
        cmd = mode._build_offline_command(tmp_path)
        assert "--profile" not in cmd

    def test_profile_on_adds_profile_flag(self, tmp_path):
        mode = self._mode("xdit --model flux", profile=True)
        cmd = mode._build_offline_command(tmp_path)
        assert "--profile" in cmd

    def test_profile_on_strips_user_profile_then_readds(self, tmp_path):
        mode = self._mode("xdit --model flux --profile", profile=True)
        cmd = mode._build_offline_command(tmp_path)
        assert cmd.count("--profile") == 1

    def test_profile_off_strips_user_profile(self, tmp_path):
        mode = self._mode("xdit --model flux --profile", profile=False)
        cmd = mode._build_offline_command(tmp_path)
        assert "--profile" not in cmd

    def test_preserves_quoted_prompt(self, tmp_path):
        mode = self._mode('xdit --model flux --prompt "a cat sitting"')
        cmd = mode._build_offline_command(tmp_path)
        assert "a cat sitting" in cmd

    def test_empty_run_cmd_raises(self, tmp_path):
        mode = self._mode("xdit --model flux")
        mode.config.run_cmd = "   "
        with pytest.raises(ValueError, match="empty"):
            mode._build_offline_command(tmp_path)


# --------------------------------------------------------------------------- #
# Full offline run() — InferenceX-free path (subprocess mocked)
# --------------------------------------------------------------------------- #
class TestOfflineRunEndToEnd:
    def _make_mode(self, tmp_path, run_cmd, profile=False):
        from Magpie.modes.benchmark.config import GpuSelectionConfig

        cfg = BenchmarkConfig(
            framework="xdit",
            model="flux",
            run_cmd=run_cmd,
            run_mode="local",
            profiler=ProfilerConfig(
                torch_profiler=TorchProfilerConfig(enabled=profile)
            ),
            gpu_selection=GpuSelectionConfig(auto=False),
        )
        return BenchmarkMode(config=cfg, output_dir=str(tmp_path / "results"))

    def _patch_subprocess_to_emit_timings(self, monkeypatch, timings, capture):
        """Patch subprocess.run so the 'benchmark' writes timings.json into the
        workspace passed via --output_directory, then reports success."""
        import subprocess

        def fake_run(cmd, *args, **kwargs):
            capture["cmd"] = cmd
            # Resolve --output_directory from the command.
            out_dir = None
            for i, tok in enumerate(cmd):
                if tok == "--output_directory" and i + 1 < len(cmd):
                    out_dir = cmd[i + 1]
            assert out_dir is not None, "output_directory not injected"
            from pathlib import Path

            (Path(out_dir) / "timings.json").write_text(
                json.dumps(timings), encoding="utf-8"
            )

            class _CP:
                returncode = 0
                stdout = "ok"
                stderr = ""

            return _CP()

        monkeypatch.setattr(
            "Magpie.modes.benchmark.benchmarker.subprocess.run", fake_run
        )

    def test_offline_run_never_touches_inferencex(self, tmp_path, monkeypatch):
        """The whole point: offline run must NOT call ensure_inferencex_available."""
        called = {"inferencex": False}

        def boom(*a, **k):
            called["inferencex"] = True
            raise AssertionError("ensure_inferencex_available must not be called")

        monkeypatch.setattr(
            "Magpie.modes.benchmark.benchmarker.ensure_inferencex_available",
            boom,
        )
        capture = {}
        self._patch_subprocess_to_emit_timings(monkeypatch, [1.0, 1.0], capture)

        mode = self._make_mode(tmp_path, "xdit --model flux --batch_size 1")
        result = mode.run()

        assert called["inferencex"] is False
        assert result.success is True

    def test_offline_run_parses_timings_into_result(self, tmp_path, monkeypatch):
        capture = {}
        self._patch_subprocess_to_emit_timings(
            monkeypatch, [2.0, 2.0], capture
        )
        mode = self._make_mode(
            tmp_path, "xdit --model flux --batch_size 1 --num_inference_steps 10"
        )
        result = mode.run()

        assert result.success is True
        assert result.framework == "xdit"
        assert result.throughput.output_throughput == pytest.approx(0.5)
        assert result.raw_result["latency_per_image_s"] == pytest.approx(2.0)
        assert result.raw_result["steps_per_sec"] == pytest.approx(5.0)

    def test_offline_run_missing_timings_marks_failure(
        self, tmp_path, monkeypatch
    ):
        import subprocess

        def fake_run(cmd, *args, **kwargs):
            class _CP:
                returncode = 0
                stdout = ""
                stderr = "boom"

            return _CP()

        monkeypatch.setattr(
            "Magpie.modes.benchmark.benchmarker.subprocess.run", fake_run
        )
        mode = self._make_mode(tmp_path, "xdit --model flux")
        result = mode.run()

        assert result.success is False
        assert any("timings.json" in e for e in result.errors)

    def test_offline_run_writes_report(self, tmp_path, monkeypatch):
        capture = {}
        self._patch_subprocess_to_emit_timings(monkeypatch, [1.0], capture)
        mode = self._make_mode(tmp_path, "xdit --model flux")
        result = mode.run()
        assert result.success is True
        # Report written into the workspace.
        from pathlib import Path

        report = Path(result.workspace_dir) / "benchmark_report.json"
        assert report.exists()


# --------------------------------------------------------------------------- #
# benchmark_images.yaml — xdit entry exists
# --------------------------------------------------------------------------- #
class TestBenchmarkImagesYaml:
    def test_xdit_entry_present(self):
        from pathlib import Path

        images_path = (
            Path(__file__).resolve().parent.parent
            / "Magpie"
            / "benchmark_images.yaml"
        )
        data = yaml.safe_load(images_path.read_text(encoding="utf-8"))
        assert "xdit" in data
        assert "gfx942" in data["xdit"]
        assert "gfx950" in data["xdit"]
