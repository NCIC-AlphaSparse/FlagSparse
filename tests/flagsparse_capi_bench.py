# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""C API SpMM timing on the same torch buffers as the Python Q4 benchmark."""
import ctypes as ct
from pathlib import Path

import torch


def prepare_spmm(values, cols, ptr, shape, b, c, op_a="trans", op_b="non"):
    lib = ct.CDLL(str(Path(__file__).resolve().parents[1] / "capi/build/libflagsparse.so"))
    p, i, n = ct.c_void_p, ct.c_int, ct.c_int64
    pp = ct.POINTER(p)
    signatures = {
        "flagsparseCreate": [pp], "flagsparseSetStream": [p, p],
        "flagsparseCreateCsr": [pp, n, n, n, p, p, p, i, i, i, i],
        "flagsparseCreateDnMat": [pp, n, n, n, p, i, i],
        "flagsparseSpMM_bufferSize": [p, i, i, p, p, p, p, p, i, i, ct.POINTER(ct.c_size_t)],
        "flagsparseSpMM_preprocess": [p, i, i, p, p, p, p, p, i, i, p],
        "flagsparseSpMM": [p, i, i, p, p, p, p, p, i, i, p],
        "flagsparseDestroySpMat": [p], "flagsparseDestroyDnMat": [p], "flagsparseDestroy": [p],
    }
    for name, signature in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = signature, i
    def check(status):
        if status:
            raise RuntimeError(f"FlagSparse C API status={status}")
    handle, a, bd, cd = p(), p(), p(), p()
    check(lib.flagsparseCreate(ct.byref(handle)))
    check(lib.flagsparseSetStream(handle, p(torch.cuda.current_stream().cuda_stream)))
    check(lib.flagsparseCreateCsr(ct.byref(a), *shape, values.numel(), p(ptr.data_ptr()),
        p(cols.data_ptr()), p(values.data_ptr()), 2 if ptr.dtype == torch.int32 else 3,
        2 if cols.dtype == torch.int32 else 3, 0, 0))
    for tensor, desc in ((b, bd), (c, cd)):
        order, ld = (2, tensor.stride(0)) if tensor.stride(1) == 1 else (1, tensor.stride(1))
        check(lib.flagsparseCreateDnMat(ct.byref(desc), *tensor.shape, ld, p(tensor.data_ptr()), 0, order))
    one, zero = ct.c_float(1), ct.c_float(0)
    ops = {"non": 0, "trans": 1, "conj": 2}
    args = (handle, ops[op_a], ops[op_b], ct.byref(one), a, bd, ct.byref(zero), cd, 0, 0)
    size = ct.c_size_t()
    check(lib.flagsparseSpMM_bufferSize(*args, ct.byref(size)))
    work = torch.empty(max(1, size.value), dtype=torch.uint8, device=values.device)
    check(lib.flagsparseSpMM_preprocess(*args, p(work.data_ptr())))
    def run():
        check(lib.flagsparseSpMM(*args, p(work.data_ptr())))
        return c
    def close():
        lib.flagsparseDestroyDnMat(cd)
        lib.flagsparseDestroyDnMat(bd)
        lib.flagsparseDestroySpMat(a)
        lib.flagsparseDestroy(handle)
    return run, close
