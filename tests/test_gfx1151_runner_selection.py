"""gfx1151 (Radeon 8060S) is selected end to end, not only shipped as scripts."""

from pathlib import Path

from Magpie.modes.benchmark.image_selector import ImageSelector

SCRIPTS = Path(__file__).resolve().parents[1] / "Magpie" / "scripts" / "benchmark"


def test_gfx1151_maps_to_radeon8060s_runner(tmp_path):
    selector = ImageSelector(str(tmp_path / "missing.yaml"))
    assert selector.get_runner_type("gfx1151") == "radeon8060s"


def test_radeon8060s_runner_has_a_builtin_script_for_each_framework():
    for framework in ("vllm", "sglang"):
        script = SCRIPTS / f"{framework}_radeon8060s.sh"
        assert script.is_file(), script
        text = script.read_text()
        assert "MAGPIE_RUN_PHASE" in text, "runner must honour the server/client split like the MI scripts"
