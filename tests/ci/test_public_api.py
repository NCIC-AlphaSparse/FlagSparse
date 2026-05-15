"""CPU-only checks for the public package surface."""

import flagsparse


EXPECTED_TOP_LEVEL = {
    "flagsparse_gather",
    "flagsparse_scatter",
    "flagsparse_spmv_csr",
    "flagsparse_spmv_coo",
    "flagsparse_spmm_csr",
    "flagsparse_spmm_coo",
    "flagsparse_spgemm_csr",
    "flagsparse_sddmm_csr",
    "flagsparse_spsv_csr",
    "flagsparse_spsv_coo",
    "flagsparse_spsm_csr",
    "flagsparse_spsm_coo",
    "create_csr_matrix",
    "create_coo_matrix",
    "read_mtx_file",
}


def test_public_surface_exposes_core_entry_points():
    exported = set(dir(flagsparse))
    assert EXPECTED_TOP_LEVEL <= exported
    assert EXPECTED_TOP_LEVEL <= set(flagsparse.__all__)
