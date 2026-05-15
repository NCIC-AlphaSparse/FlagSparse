"""CPU-only checks for README command examples and workflow references."""

from pathlib import Path


README = Path(__file__).resolve().parents[2] / "README.md"
TEXT = README.read_text(encoding="utf-8")


def test_readme_install_command_is_present():
    assert "pip install . --no-deps --no-build-isolation" in TEXT


def test_readme_cpu_smoke_examples_are_present():
    for snippet in [
        "pytest tests/pytest --mode quick",
        "python run_flagsparse_pytest.py --mode quick --ops gather,spmv_csr,spmm_csr --gpus 0",
        "python tests/test_spmv.py <dir_or_file.mtx>",
        "python tests/test_spmm.py --synthetic",
        "python tests/test_spgemm.py <dir_or_file.mtx> --input-mode auto",
        "python tests/test_spsm.py --synthetic --n 512 --rhs 32",
    ]:
        assert snippet in TEXT


def test_readme_ci_cd_section_mentions_workflows():
    for snippet in [
        ".github/workflows/ci.yml",
        ".github/workflows/nightly-cpu.yml",
        ".github/workflows/release.yml",
        ".github/workflows/triton-smoke.yml",
        ".github/workflows/gpu-benchmark.yml",
    ]:
        assert snippet in TEXT


def test_readme_documents_standardized_command_policy():
    for snippet in [
        "documented invocation standard",
        "CPU-only install, build, help-text, and smoke paths are checked in CI",
        "`make ci` / `make check`",
        "`make release-check` / `make release`",
        "`make triton-smoke` and `make triton-deps`",
    ]:
        assert snippet in TEXT
