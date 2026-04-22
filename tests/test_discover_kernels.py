from Magpie.mcp.discovery import discover_project_kernels


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _discover(project_path, **kwargs):
    return discover_project_kernels(str(project_path), **kwargs)


def test_discover_kernels_filters_plain_python_files_for_triton(tmp_path):
    _write(
        tmp_path / "kernels" / "softmax.py",
        """
import triton

@triton.jit
def softmax_kernel(x_ptr):
    pass
""",
    )
    _write(
        tmp_path / "scripts" / "task_runner.py",
        """
import triton

def main():
    print(triton.__version__)
""",
    )
    _write(
        tmp_path / "helpers" / "plain.py",
        """
def helper():
    return 1
""",
    )

    data = _discover(tmp_path, kernel_type="triton")
    found = {entry["source_file"] for entry in data["kernels"]}

    assert str(tmp_path / "kernels" / "softmax.py") in found
    assert str(tmp_path / "scripts" / "task_runner.py") not in found
    assert str(tmp_path / "helpers" / "plain.py") not in found


def test_discover_kernels_accepts_triton_aliases(tmp_path):
    _write(
        tmp_path / "source" / "alias_kernel.py",
        """
import triton as tr

@tr.jit
def alias_kernel(x_ptr):
    pass
""",
    )
    _write(
        tmp_path / "source" / "from_import_kernel.py",
        """
from triton import jit

@jit
def imported_kernel(x_ptr):
    pass
""",
    )

    data = _discover(tmp_path, kernel_type="triton")
    found = {entry["source_file"] for entry in data["kernels"]}

    assert str(tmp_path / "source" / "alias_kernel.py") in found
    assert str(tmp_path / "source" / "from_import_kernel.py") in found


def test_discover_kernels_includes_generated_triton_from_build_directories(tmp_path):
    _write(
        tmp_path / "build" / "generated_kernel.py",
        """
import triton

@triton.jit
def generated_kernel(x_ptr):
    pass
""",
    )
    _write(
        tmp_path / "src" / "real_kernel.py",
        """
import triton

@triton.jit
def real_kernel(x_ptr):
    pass
""",
    )

    data = _discover(tmp_path, kernel_type="triton")
    found = {entry["source_file"] for entry in data["kernels"]}

    assert str(tmp_path / "src" / "real_kernel.py") in found
    assert str(tmp_path / "build" / "generated_kernel.py") in found


def test_discover_kernels_skips_non_triton_generated_sources_in_build_dirs(tmp_path):
    _write(
        tmp_path / "build" / "generated_kernel.cpp",
        """
extern "C" __global__ void generated_kernel(float* x) {}
""",
    )
    _write(
        tmp_path / "src" / "real_kernel.cpp",
        """
extern "C" __global__ void real_kernel(float* x) {}
""",
    )
    _write(
        tmp_path / "build" / "plain.py",
        """
def helper():
    return 1
""",
    )
    _write(
        tmp_path / "build" / "generated_kernel.py",
        """
import triton

@triton.jit
def generated_kernel(x_ptr):
    pass
""",
    )

    data = _discover(tmp_path, kernel_type="all")
    found = {entry["source_file"] for entry in data["kernels"]}

    assert str(tmp_path / "src" / "real_kernel.cpp") in found
    assert str(tmp_path / "build" / "generated_kernel.cpp") not in found
    assert str(tmp_path / "build" / "plain.py") not in found
    assert str(tmp_path / "build" / "generated_kernel.py") in found
