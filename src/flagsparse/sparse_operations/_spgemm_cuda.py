# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded reuse of validated CUDA SpGEMM structure, with live operands."""

import torch

from ._rocm_sparse_cache import TensorPlanCache


_STRUCTURES = TensorPlanCache(maxsize=2)


def prepare(a, ai, ap, ashape, b, bi, bp, bshape):
    from .spgemm_csr import SpGEMMPrepared, prepare_spgemm_csr

    # Tensor metadata is checked on every call. Device contents only need
    # revalidation when an index/pointer version, storage or shape changes.
    sources = (ai, ap, bi, bp)
    options = (
        tuple(ashape),
        tuple(bshape),
        a.dtype,
        b.dtype,
        a.device,
        b.device,
        tuple(a.shape),
        tuple(b.shape),
        torch.cuda.current_stream(a.device).cuda_stream,
    )
    capture = torch.cuda.is_current_stream_capturing()
    plan = None if capture else _STRUCTURES.get(sources, options)
    if plan is None:
        prepared = prepare_spgemm_csr(a, ai, ap, ashape, b, bi, bp, bshape)
        if not capture:
            _STRUCTURES.put(
                sources,
                options,
                (
                    prepared.a_indices,
                    prepared.a_indptr,
                    prepared.b_indices,
                    prepared.b_indptr,
                    prepared.a_row_work,
                    prepared.a_pref,
                    prepared.row_work_ready,
                ),
            )
        return prepared
    ai, ap, bi, bp, work, pref, ready = plan
    # These prepared objects stay internal to the convenience call. In
    # particular, a caller cannot overwrite a cached plan through prepare().
    return SpGEMMPrepared(
        a.contiguous(),
        ai,
        ap,
        ashape,
        b.contiguous(),
        bi,
        bp,
        bshape,
        work,
        pref,
        ready,
    )
