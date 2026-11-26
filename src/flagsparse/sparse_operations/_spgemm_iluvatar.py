"""Reuse bounded symbolic SpGEMM product maps on Iluvatar."""

from ._rocm_sparse_cache import TensorPlanCache
from ._spgemm_symbolic import build_symbolic, evaluate_symbolic


_SYMBOLIC_CACHE = TensorPlanCache(maxsize=1)
_MAX_PRODUCTS = 128_000_000


def _build_symbolic(prepared):
    return build_symbolic(prepared, _MAX_PRODUCTS)


def compute(prepared):
    sources = prepared.symbolic_sources
    if sources is None:
        sources = (prepared.a_indices, prepared.a_indptr, prepared.b_indices, prepared.b_indptr)
    if (prepared.n_rows * prepared.n_cols >= 2**63
            or max(prepared.a_data.numel(), prepared.b_data.numel()) >= 2**31
            or _SYMBOLIC_CACHE._signature(sources) is None):
        return None
    options = (prepared.a_shape, prepared.b_shape)
    symbolic = _SYMBOLIC_CACHE.get(sources, options)
    if symbolic is None:
        symbolic = _build_symbolic(prepared)
        _SYMBOLIC_CACHE.put(sources, options, symbolic)
    if not symbolic:
        return None
    return evaluate_symbolic(prepared, symbolic)
