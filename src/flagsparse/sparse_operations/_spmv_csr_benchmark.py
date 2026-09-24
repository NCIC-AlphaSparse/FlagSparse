"""Shared CSR SpMV measurements; setup and diagnostics never change the denominator."""

import torch

from . import _common as common
from .spmv_csr import flagsparse_spmv_csr_run, _spmv_device_context


def _filtered_avg_ms(times):
    if not times:
        return None
    values = [float(t) for t in times]
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    n = len(ordered)
    if n % 2 == 0:
        median = (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    else:
        median = ordered[n // 2]
    if median == 0.0:
        kept = [t for t in ordered if t == 0.0]
    else:
        lo = median * 0.9
        hi = median * 1.1
        kept = [t for t in ordered if lo <= t <= hi]
    return sum(kept) / len(kept) if kept else median


def event_benchmark(fn, warmup, iters):
    if warmup < 0 or iters < 1:
        raise ValueError("require warmup >= 0 and iters >= 1")
    # Always compile before measuring, including --warmup=0.
    value = fn()
    for _ in range(warmup):
        value = fn()
    common._ACCEL.synchronize()
    samples = []
    for _ in range(iters):
        start = common._ACCEL.Event(enable_timing=True)
        end = common._ACCEL.Event(enable_timing=True)
        start.record()
        value = fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return value, _filtered_avg_ms(samples)


def measure_route(prepared, x, warmup=10, iters=50, timing=False):
    with _spmv_device_context(prepared.data.device):
        # Keep the timed operation equivalent to the prepared hipSPARSE path,
        # whose DnVec output is allocated during descriptor setup.  The CSR
        # kernels overwrite every output element, so this buffer needs no
        # per-iteration initialization.
        output_size = prepared.n_cols if prepared.transpose else prepared.n_rows
        out = torch.empty(
            output_size, dtype=prepared.data.dtype, device=prepared.data.device
        )
        # hipSPARSE's reference window invokes an already prepared SpMV
        # descriptor directly.  A non-transpose row_tile has no runtime plan,
        # conversion, or fallback work, so time its prepared kernel launch
        # directly as well.  Starting an event before the public Python API
        # otherwise charges its validation/dispatch latency (~32us on gfx936)
        # to the GPU event while the stream is empty.  Keep the generic route
        # for paths that genuinely perform runtime processing.
        direct_row_tile = (
            prepared.alg == "row_tile"
            and not prepared.transpose
            and not x.is_conj()
            and not prepared.data.is_conj()
        )
        if direct_row_tile:
            from . import _spmv_csr_kernels as kernels

            timed_op = lambda: kernels.compute(
                prepared, x, out, prepared.alg, prepared.config, plan=None
            )
        else:
            timed_op = lambda: flagsparse_spmv_csr_run(prepared, x, out=out)
        value, gpu_ms = event_benchmark(
            timed_op, warmup, iters
        )
        # A separate invocation collects metadata and optional phase events.
        _, meta = flagsparse_spmv_csr_run(
            prepared, x, out=out, return_meta=True, timing=timing
        )
        meta.update(gpu_ms=gpu_ms, op_gpu_ms=gpu_ms)
        meta["ms"] = meta["process_cpu_ms"] + gpu_ms
        meta["op_total_ms"] = meta["ms"]
        return value, meta


def golden_csr(data, indices, indptr, x, shape, op="non"):
    """Correctness-only CPU FP64/complex128 scatter reference, cast to output dtype."""
    dtype = torch.complex128 if data.is_complex() else torch.float64
    a = data.detach().to(device="cpu", dtype=dtype)
    vector = x.detach().to(device="cpu", dtype=dtype)
    ptr = indptr.detach().to(device="cpu", dtype=torch.int64)
    col = indices.detach().to(device="cpu", dtype=torch.int64)
    rows = torch.repeat_interleave(torch.arange(shape[0]), ptr[1:] - ptr[:-1])
    if op == "non":
        output = torch.zeros(shape[0], dtype=dtype)
        output.index_add_(0, rows, a * vector[col])
    else:
        output = torch.zeros(shape[1], dtype=dtype)
        output.index_add_(0, col, (a.conj() if op == "conj" else a) * vector[rows])
    return output.to(data.dtype)


def check_result(value, reference, rtol, atol):
    actual = value.detach().cpu()
    dtype = torch.complex128 if actual.is_complex() else torch.float64
    actual, reference = actual.to(dtype), reference.to(dtype)
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    error = float((actual - reference).abs().max()) if actual.numel() else 0.0
    return finite and torch.allclose(actual, reference, rtol=rtol, atol=atol), error


def _pytorch_spmv(matrix, x_2d, op):
    if op == "non":
        return torch.sparse.mm(matrix, x_2d).squeeze(1)
    if op == "trans":
        return torch.sparse.mm(matrix.transpose(0, 1), x_2d).squeeze(1)
    if op == "conj":
        if matrix.is_complex():
            matrix = matrix.conj()
        return torch.sparse.mm(matrix.transpose(0, 1), x_2d).squeeze(1)
    raise ValueError(f"unsupported sparse operation: {op}")


def _measure_pytorch_spmv(data, indices, indptr, x, shape, op, warmup, iters):
    """Time MACA's working torch.sparse format after one out-of-band probe."""
    x_2d = x.unsqueeze(1)
    _, sparse_format = common._pytorch_sparse_mm(
        data, indices, indptr, shape, x_2d, op=op
    )
    if sparse_format == "COO":
        matrix = common._pytorch_sparse_coo_matrix(data, indices, indptr, shape)
    else:
        matrix, _ = common._pytorch_sparse_matrix(data, indices, indptr, shape)
    return (*event_benchmark(lambda: _pytorch_spmv(matrix, x_2d, op), warmup, iters), sparse_format)


def measure_vendor(data, indices, indptr, x, shape, op, warmup, iters):
    """Native same-device CSR baseline, with descriptors outside event measurement."""
    with _spmv_device_context(data.device):
        backend, reason = common._spmv_csr_sparse_ref_backend(
            data.dtype, indices.dtype, op=op
        )
        result = {
            "vendor_backend": backend or "N/A",
            "vendor_alg": "N/A",
            "vendor_ms": None,
            "vendor_reason": reason,
            "values": None,
        }
        if backend is None:
            return result
        if backend == "hipsparse":
            state = common._prepare_spmv_csr_ref_hipsparse(
                data, indices, indptr, x, shape, op=op
            )
            try:
                if not state.get("empty"):
                    from .gather_scatter import _set_hipsparse_stream

                    reason = _set_hipsparse_stream(state["handle"])
                    if reason:
                        result["vendor_reason"] = reason
                        return result
                value, ms = event_benchmark(
                    lambda: common._run_spmv_csr_ref_hipsparse_prepared(state),
                    warmup,
                    iters,
                )
                # The binding's selected enum is the source of truth.
                result["vendor_alg"] = str(state.get("alg", "empty"))
            finally:
                common._destroy_spmv_csr_ref_hipsparse_prepared(state)
        elif backend in ("cupy_cusparse", "iluvatar_legacy_cusparse"):
            if op != "non":
                result["vendor_reason"] = (
                    "CuPy transpose changes CSR storage; no native CSR transpose baseline exposed"
                )
                return result
            if common._is_iluvatar_runtime():
                # The CoreX package has no cupyx.cusparse module, which makes
                # common's generic-CuPy optional import unavailable.  Import
                # only the verified legacy path here; other backends retain
                # their existing shared imports and generic implementation.
                import cupy as cp
                import cupy.cusparse as cupy_cusparse
                import cupyx.scipy.sparse as cpx_sparse
            else:
                cp = common.cp
                cpx_sparse = common.cpx_sparse
            # Use the actual Torch stream on the input device for construction and timing.
            with cp.cuda.Device(data.device.index or 0):
                stream = common._torch_current_stream_ptr()
                if stream is None:
                    result["vendor_reason"] = "cannot identify the execution stream"
                    return result
                with cp.cuda.ExternalStream(stream):
                    if common._is_iluvatar_runtime():
                        # The verified CoreX legacy API consumes int32 CSR row
                        # offsets.  This setup conversion is outside events and
                        # leaves the operator's original index tensors intact.
                        legacy_indptr = (
                            indptr
                            if indptr.dtype == torch.int32
                            else indptr.to(torch.int32)
                        )
                        matrix_args = (
                            cp.from_dlpack(torch.utils.dlpack.to_dlpack(data)),
                            cp.from_dlpack(torch.utils.dlpack.to_dlpack(indices)),
                            cp.from_dlpack(torch.utils.dlpack.to_dlpack(legacy_indptr)),
                        )
                        vector = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x))
                    else:
                        matrix_args = (
                            common._cupy_from_torch(data),
                            common._cupy_from_torch(indices),
                            common._cupy_from_torch(indptr),
                        )
                        vector = common._cupy_from_torch(x)
                    matrix = cpx_sparse.csr_matrix(matrix_args, shape=shape)
                    if common._is_iluvatar_runtime():
                        # CoreX generic SpMV has a broken buffer-size query.  Its
                        # legacy csrmv entry point is independently validated for
                        # fp32/int32/non and takes a reusable output buffer.
                        output_cp = cp.empty(shape[0], dtype=vector.dtype)
                        value_cp, ms = event_benchmark(
                            lambda: cupy_cusparse.csrmv(
                                matrix, vector, y=output_cp
                            ),
                            warmup,
                            iters,
                        )
                        result["vendor_alg"] = "cupy_legacy_csrmv"
                    else:
                        value_cp, ms = event_benchmark(
                            lambda: matrix @ vector, warmup, iters
                        )
                        result["vendor_alg"] = "cupy_csr_matvec (library selected)"
                    if common._is_iluvatar_runtime():
                        value = torch.utils.dlpack.from_dlpack(value_cp.toDlpack())
                    else:
                        value = common._torch_from_cupy(value_cp)
        elif backend == "torch":
            value, ms, sparse_format = _measure_pytorch_spmv(
                data, indices, indptr, x, shape, op, warmup, iters
            )
            result["vendor_alg"] = f"torch_sparse_{sparse_format.lower()}_matvec"
        else:
            result["vendor_reason"] = f"no event-timed CSR baseline for {backend}"
            return result
        result.update(values=value, vendor_ms=ms, vendor_reason=None)
        return result
