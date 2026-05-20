"""CPU-only smoke tests for CI packaging and public exports."""

import flagsparse


def test_package_version_is_exposed():
    assert flagsparse.__version__ == "1.0.0"


def test_public_exports_are_listed():
    exported = set(dir(flagsparse))
    assert "flagsparse_spmv_csr" in exported
    assert "create_csr_matrix" in exported
    assert "read_mtx_file" in exported
