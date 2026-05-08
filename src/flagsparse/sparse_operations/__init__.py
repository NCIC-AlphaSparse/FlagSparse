"""FlagSparse sparse operations (gather, scatter, SpMV, SpMM, SpGEMM, SDDMM, SpSM)."""

from ._common import SUPPORTED_INDEX_DTYPES, SUPPORTED_VALUE_DTYPES, cp, cpx_sparse
from .alpha_spmm_alg1 import (
    PreparedAlphaSpmmAlg1,
    flagsparse_alpha_spmm_alg1,
    prepare_alpha_spmm_alg1,
)
from .gather_scatter import (
    cusparse_spmv_gather,
    cusparse_spmv_scatter,
    flagsparse_gather,
    flagsparse_scatter,
    pytorch_index_gather,
    pytorch_index_scatter,
    triton_cusparse_gather,
    triton_cusparse_scatter,
)
from .sddmm_csr import SDDMMPrepared, benchmark_sddmm_case, flagsparse_sddmm_csr, prepare_sddmm_csr
from .spgemm_csr import SpGEMMPrepared, benchmark_spgemm_case, flagsparse_spgemm_csr, prepare_spgemm_csr
from .spmm_coo import flagsparse_spmm_coo
from .spmm_csr import (
    PreparedCsrSpmmOpt,
    benchmark_spmm_case,
    benchmark_spmm_opt_case,
    comprehensive_spmm_test,
    flagsparse_spmm_csr,
    flagsparse_spmm_csr_opt,
    prepare_spmm_csr_opt,
)
from .spmm_csr_opt_alg2 import (
    PreparedCsrSpmmOptAlg2,
    benchmark_spmm_opt_alg2_case,
    flagsparse_spmm_csr_opt_alg2,
    flagsparse_spmm_csr_opt_alg2_symbolic,
    prepare_spmm_csr_opt_alg2,
    prepare_spmm_csr_opt_alg2_symbolic,
)
from .spmv_coo import PreparedCoo, flagsparse_spmv_coo, prepare_spmv_coo
from .spmv_csr import (
    PreparedCsrSpmv,
    flagsparse_spmv_coo_tocsr,
    flagsparse_spmv_csr,
    prepare_spmv_coo_tocsr,
    prepare_spmv_csr,
)
from .spsm import benchmark_spsm_case, flagsparse_spsm_coo, flagsparse_spsm_csr
from .spsv import flagsparse_spsv_coo, flagsparse_spsv_csr

_BENCHMARK_EXPORTS = {
    "benchmark_gather_case",
    "benchmark_performance",
    "benchmark_scatter_case",
    "benchmark_spmv_case",
    "comprehensive_gather_test",
    "comprehensive_scatter_test",
    "comprehensive_spsm_test",
}

__all__ = [
    "PreparedCoo",
    "PreparedAlphaSpmmAlg1",
    "PreparedCsrSpmv",
    "PreparedCsrSpmmOpt",
    "PreparedCsrSpmmOptAlg2",
    "SDDMMPrepared",
    "SpGEMMPrepared",
    "SUPPORTED_INDEX_DTYPES",
    "SUPPORTED_VALUE_DTYPES",
    "benchmark_gather_case",
    "benchmark_performance",
    "benchmark_scatter_case",
    "benchmark_sddmm_case",
    "benchmark_spgemm_case",
    "benchmark_spmm_case",
    "benchmark_spmm_opt_case",
    "benchmark_spmm_opt_alg2_case",
    "benchmark_spmv_case",
    "benchmark_spsm_case",
    "comprehensive_gather_test",
    "comprehensive_scatter_test",
    "comprehensive_spmm_test",
    "comprehensive_spsm_test",
    "cusparse_spmv_gather",
    "cusparse_spmv_scatter",
    "flagsparse_gather",
    "flagsparse_alpha_spmm_alg1",
    "flagsparse_sddmm_csr",
    "flagsparse_spgemm_csr",
    "flagsparse_spmm_coo",
    "flagsparse_spmm_csr",
    "flagsparse_spmm_csr_opt",
    "flagsparse_spmm_csr_opt_alg2",
    "flagsparse_spmm_csr_opt_alg2_symbolic",
    "flagsparse_spmv_coo",
    "flagsparse_spmv_coo_tocsr",
    "flagsparse_spmv_csr",
    "flagsparse_spsm_coo",
    "flagsparse_spsm_csr",
    "flagsparse_spsv_coo",
    "flagsparse_spsv_csr",
    "prepare_sddmm_csr",
    "prepare_alpha_spmm_alg1",
    "prepare_spgemm_csr",
    "prepare_spmm_csr_opt",
    "prepare_spmm_csr_opt_alg2",
    "prepare_spmm_csr_opt_alg2_symbolic",
    "prepare_spmv_coo",
    "prepare_spmv_coo_tocsr",
    "prepare_spmv_csr",
    "pytorch_index_gather",
    "pytorch_index_scatter",
    "triton_cusparse_gather",
    "triton_cusparse_scatter",
]


def __getattr__(name):
    if name in _BENCHMARK_EXPORTS:
        from . import benchmarks as _benchmarks

        return getattr(_benchmarks, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
