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
        value, gpu_ms = event_benchmark(
            lambda: flagsparse_spmv_csr_run(prepared, x), warmup, iters
        )
        # A separate invocation collects metadata and optional phase events.
        _, meta = flagsparse_spmv_csr_run(prepared, x, return_meta=True, timing=timing)
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
        elif backend == "cupy_cusparse":
            if op != "non":
                result["vendor_reason"] = (
                    "CuPy transpose changes CSR storage; no native CSR transpose baseline exposed"
                )
                return result
            cp = common.cp
            # Use the actual Torch stream on the input device for construction and timing.
            with cp.cuda.Device(data.device.index or 0):
                stream = common._torch_current_stream_ptr()
                if stream is None:
                    result["vendor_reason"] = "cannot identify the execution stream"
                    return result
                with cp.cuda.ExternalStream(stream):
                    matrix = common.cpx_sparse.csr_matrix(
                        (
                            common._cupy_from_torch(data),
                            common._cupy_from_torch(indices),
                            common._cupy_from_torch(indptr),
                        ),
                        shape=shape,
                    )
                    vector = common._cupy_from_torch(x)
                    value_cp, ms = event_benchmark(
                        lambda: matrix @ vector, warmup, iters
                    )
                    value = common._torch_from_cupy(value_cp)
            result["vendor_alg"] = "cupy_csr_matvec (library selected)"
        else:
            result["vendor_reason"] = f"no event-timed CSR baseline for {backend}"
            return result
        result.update(values=value, vendor_ms=ms, vendor_reason=None)
        return result
