import subprocess

import pytest

import Magpie.utils.amd_smi as amd_smi


def test_run_json_success(monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout='{"gpu_data": [{"gpu": 0}]}')

    monkeypatch.setattr(amd_smi.subprocess, "run", run)
    payload = amd_smi.run_json(["metric", "--power"])
    assert payload == {"gpu_data": [{"gpu": 0}]}
    assert calls[0] == ["amd-smi", "metric", "--power", "--json"]


@pytest.mark.parametrize(
    "effect",
    [
        FileNotFoundError(),
        subprocess.TimeoutExpired(cmd="amd-smi", timeout=1),
        OSError("boom"),
    ],
)
def test_run_json_handles_run_failures(monkeypatch, effect):
    monkeypatch.setattr(
        amd_smi.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(effect)
    )
    assert amd_smi.run_json(["list"]) is None


def test_run_json_handles_nonzero_exit_and_bad_json(monkeypatch):
    monkeypatch.setattr(
        amd_smi.subprocess,
        "run",
        lambda cmd, **k: subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="no GPU"
        ),
    )
    assert amd_smi.run_json(["list"]) is None

    monkeypatch.setattr(
        amd_smi.subprocess,
        "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout="not json"),
    )
    assert amd_smi.run_json(["list"]) is None


def test_gpu_rows_handles_both_wrapped_and_bare_shapes():
    assert amd_smi.gpu_rows({"gpu_data": [{"gpu": 0}, "garbage", {"gpu": 1}]}) == [
        {"gpu": 0},
        {"gpu": 1},
    ]
    assert amd_smi.gpu_rows([{"gpu": 0}]) == [{"gpu": 0}]
    assert amd_smi.gpu_rows(None) == []
    assert amd_smi.gpu_rows({"gpu_data": None}) == []


@pytest.mark.parametrize(
    "block,expected",
    [
        (11, 11.0),
        (11.5, 11.5),
        ({"value": 11, "unit": "W"}, 11.0),
        ("N/A", None),
        (None, None),
        (True, None),  # bool must not be read as 0/1
        ({"value": "N/A"}, None),
    ],
)
def test_scalar_handles_bare_wrapped_and_na(block, expected):
    assert amd_smi.scalar(block) == expected


def test_lookup_skips_na_and_missing_and_prefers_first_match():
    mapping = {"hotspot": "N/A", "edge": {"value": 31, "unit": "C"}, "junction": 99}
    assert amd_smi.lookup(mapping, "hotspot", "junction", "edge") == 99
    assert amd_smi.lookup(mapping, "hotspot", "missing", "edge") == {
        "value": 31,
        "unit": "C",
    }
    assert amd_smi.lookup(mapping, "hotspot") is None
    assert amd_smi.lookup("not a dict", "hotspot") is None


@pytest.mark.parametrize(
    "block,expected_mib",
    [
        ({"value": 2048, "unit": "MB"}, 2048.0),
        ({"value": 2, "unit": "GB"}, 2048.0),
        ({"value": 49152, "unit": "MB"}, 49152.0),
        ({"value": 1, "unit": "GiB"}, 1024.0),
        ("N/A", None),
        ({"value": 0, "unit": "bogus"}, None),
    ],
)
def test_mib_converts_by_exact_unit(block, expected_mib):
    assert amd_smi.mib(block) == expected_mib


def test_clock_mhz_prefers_per_domain_then_falls_back_to_apu_aggregate():
    # Discrete GPU shape: the per-domain block carries the real reading.
    discrete = {"gfx_0": {"clk": {"value": 1500, "unit": "MHz"}, "min_clk": "N/A"}}
    assert (
        amd_smi.clock_mhz(discrete, "gfx_0", "apu_average_gfxclk_frequency") == 1500.0
    )

    # APU shape: the per-domain block is "N/A" for every sub-field, only the aggregate reports.
    apu = {
        "gfx_0": {"clk": "N/A", "min_clk": "N/A"},
        "apu_average_gfxclk_frequency": {"value": 627, "unit": "MHz"},
    }
    assert amd_smi.clock_mhz(apu, "gfx_0", "apu_average_gfxclk_frequency") == 627.0

    # Neither domain reports: None, not a crash.
    assert (
        amd_smi.clock_mhz(
            {"gfx_0": {"clk": "N/A"}}, "gfx_0", "apu_average_gfxclk_frequency"
        )
        is None
    )
    assert amd_smi.clock_mhz(None, "gfx_0") is None


def test_gpu_count_reads_list_rows(monkeypatch):
    monkeypatch.setattr(amd_smi, "run_json", lambda *a, **k: [{"gpu": 0}, {"gpu": 1}])
    assert amd_smi.gpu_count() == 2
    monkeypatch.setattr(amd_smi, "run_json", lambda *a, **k: None)
    assert amd_smi.gpu_count() == 0
