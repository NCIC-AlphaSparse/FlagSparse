# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CUDA reuse must preserve live operands, metadata and stream semantics."""

import pytest
import torch

import flagsparse as fs
from flagsparse.sparse_operations import spmm_csc, vector_ops
from flagsparse.sparse_operations._common import _backend_name


pytestmark = pytest.mark.skipif(
    _backend_name() != "cuda" or not torch.cuda.is_available(),
    reason="requires NVIDIA CUDA",
)


def _wide(dtype):
    return torch.complex128 if dtype.is_complex else torch.float64


_SPMV_CASES = [
    ("csr", torch.float16, torch.float16, torch.float16, "non"),
    ("csr", torch.bfloat16, torch.bfloat16, torch.bfloat16, "non"),
    ("csr", torch.float32, torch.float32, torch.float32, "non"),
    ("csr", torch.float64, torch.float64, torch.float64, "non"),
    ("csr", torch.complex64, torch.complex64, torch.complex64, "non"),
    ("coo", torch.float16, torch.float16, torch.float16, "non"),
    ("coo", torch.float16, torch.float16, torch.float32, "non"),
    ("coo", torch.int8, torch.int8, torch.int32, "non"),
    ("coo", torch.int8, torch.int8, torch.float32, "non"),
    ("coo", torch.float32, torch.complex64, torch.complex64, "non"),
    ("coo", torch.complex64, torch.complex64, torch.complex64, "non"),
    ("csc", torch.float32, torch.float32, torch.float32, "trans"),
    ("csc", torch.complex64, torch.complex64, torch.complex64, "conj"),
    ("csc", torch.int8, torch.int8, torch.int32, "non"),
]


@pytest.mark.parametrize("fmt,dtype,xdtype,odtype,op", _SPMV_CASES)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_cuda_spmv_live_prepared_inputs(fmt, dtype, xdtype, odtype, op, index_dtype):
    values = torch.tensor([1, -2, 3, 4, -1, 2], dtype=dtype, device="cuda")
    if dtype.is_complex:
        values.add_(0.5j)
    cols = torch.tensor([0, 3, 2, 4, 1, 0], dtype=index_dtype, device="cuda")
    ptr = torch.tensor([0, 2, 3, 5, 6], dtype=index_dtype, device="cuda")
    shape = (4, 5)
    if fmt == "csr":
        plan = fs.prepare_spmv_csr(values, cols, ptr, shape)
        v, ci, rp = plan.data, plan.kernel_indices, plan.kernel_indptr

        def call(x, **kw):
            return fs.flagsparse_spmv_csr(prepared=plan, x=x, **kw)
    elif fmt == "coo":
        rows = torch.tensor([0, 0, 1, 2, 2, 3], dtype=index_dtype, device="cuda")
        plan = fs.prepare_spmv_coo(values, rows, cols, shape)
        v, ri, ci = plan.data_non, plan.row_non, plan.col_non

        def call(x, **kw):
            return fs.flagsparse_spmv_coo(prepared=plan, x=x, **kw)
    else:
        shape = (5, 4)
        plan = (
            None
            if dtype == torch.int8
            else fs.prepare_spmv_csc(values, cols, ptr, shape, op=op)
        )
        v, ci, rp = (
            (values, cols, ptr)
            if plan is None
            else (plan.data, plan.kernel_indices, plan.kernel_indptr)
        )
        call = (
            (lambda x, **kw: fs.flagsparse_spmv_csc(v, ci, rp, x, shape, op=op, **kw))
            if plan is None
            else (lambda x, **kw: fs.flagsparse_spmv_csc(prepared=plan, x=x, **kw))
        )
    n_in = shape[1] if op == "non" else shape[0]
    x = torch.arange(1, n_in + 1, dtype=torch.float32, device="cuda").to(xdtype)
    if xdtype.is_complex:
        x.add_(0.25j)
    wide = _wide(xdtype)
    previous = None
    for step in range(7):
        if step == 1:
            v.mul_(2)
        if step == 2:
            x.add_(1)
        if step == 3:
            ci[0] = 1
        if step == 4:
            if fmt == "coo":
                ri[0] = 2  # A formerly sorted pattern becomes unsorted.
            else:
                rp[1] = 1
        if step == 5:
            v.set_(v.clone().add_(1))
        if fmt == "coo":
            rr, cc = ri.cpu().long(), ci.cpu().long()
        else:
            rr = torch.repeat_interleave(
                torch.arange(4), (rp[1:] - rp[:-1]).cpu().long()
            )
            cc = ci.cpu().long()
            if fmt == "csc":
                rr, cc = cc, rr
        dense = torch.zeros(shape, dtype=wide)
        dense.index_put_((rr, cc), v.cpu().to(wide), accumulate=True)
        if op != "non":
            dense = dense.t().conj() if op == "conj" else dense.t()
        expected = (dense @ x.cpu().to(wide)).to(odtype)
        kw = (
            {}
            if odtype == (torch.int32 if dtype == torch.int8 else xdtype)
            else {"out_dtype": odtype}
        )
        actual = call(x, **kw)
        tol = (
            1e-2
            if dtype == torch.bfloat16
            else 2e-3
            if dtype == torch.float16
            else 1e-5
        )
        torch.testing.assert_close(actual.cpu(), expected, rtol=tol, atol=tol)
        if previous is not None:
            torch.testing.assert_close(previous[0].cpu(), previous[1])
        previous = actual, actual.cpu().clone()


@pytest.mark.parametrize(
    "dtype,op,odtype",
    [
        (torch.float16, "non", torch.float32),
        (torch.float32, "non", torch.float32),
        (torch.int8, "non", torch.int32),
        (torch.int8, "non", torch.float32),
        (torch.complex64, "non", torch.complex64),
        (torch.complex64, "conj", torch.complex64),
    ],
)
@pytest.mark.parametrize("nnz", [0, 17, 1024, 4096, 16384, 65536])
def test_cuda_spvv_live_inputs_and_independent_results(dtype, op, odtype, nnz):
    torch.manual_seed(42)
    n = max(1, nnz * 2)
    idx = torch.randperm(n, device="cuda")[:nnz].int()
    v = torch.randint(-4, 5, (nnz,), device="cuda").to(dtype)
    y = torch.randint(-4, 5, (n,), device="cuda").to(dtype)
    if dtype.is_complex:
        v.add_(0.5j)
        y.add_(0.25j)
    kw = {} if dtype != torch.int8 or odtype == torch.int32 else {"out_dtype": odtype}
    previous = []
    for step in range(5):
        if step == 2:
            v.add_(1)
            y.mul_(2)
        if step == 3:
            idx.copy_(idx.roll(1))
        lhs = v.cpu().to(_wide(dtype))
        if op == "conj":
            lhs = lhs.conj()
        expected = (lhs * y.cpu().to(_wide(dtype))[idx.cpu().long()]).sum().to(odtype)
        actual = fs.flagsparse_spvv(v, idx, y, op=op, validate=False, **kw)
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)
        previous.append((actual, expected.clone()))
    for actual, expected in previous:
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)


def test_cuda_spvv_scratch_isolated_between_streams_and_graph_replay():
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    v = (
        torch.arange(65536, device="cuda", dtype=torch.float32)
        .remainder(7)
        .to(torch.complex64)
    )
    idx = torch.arange(v.numel(), device="cuda", dtype=torch.int32)
    y = torch.full_like(v, 1 + 0.25j)
    torch.cuda.synchronize()
    expected = (
        (v.cpu().to(torch.complex128) * y.cpu().to(torch.complex128)).sum().to(v.dtype)
    )
    results = []
    for _ in range(20):
        for stream in streams:
            with torch.cuda.stream(stream):
                results.append(fs.flagsparse_spvv(v, idx, y, validate=False))
    torch.cuda.synchronize()
    for result in results:
        torch.testing.assert_close(result.cpu(), expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fs.flagsparse_spvv(v, idx, y, validate=False)
    for scale in (2, 3):
        v.mul_(scale)
        graph.replay()
        torch.testing.assert_close(captured.cpu(), expected.mul_(scale))


def test_cuda_spmv_fp32_cancellation_overflow_and_subnormal_products():
    v = torch.tensor(
        [1e20, -1e20, 1e-30, 1e-30, 1e8, 1, -1e8, 1e6, 1, -1e6], device="cuda"
    )
    ci = torch.tensor([0, 0, 2, 2, 1, 1, 1, 1, 1, 1], dtype=torch.int32, device="cuda")
    rp = torch.tensor([0, 2, 4, 7, 10], dtype=torch.int32, device="cuda")
    x = torch.tensor([1e20, 1, 1e-10], device="cuda")
    plan = fs.prepare_spmv_csr(v, ci, rp, (4, 3))
    matrix = torch.sparse_csr_tensor(
        rp.cpu().long(), ci.cpu().long(), v.cpu().double(), (4, 3)
    )
    expected = (matrix @ x.cpu().double()).float()
    for _ in range(3):
        torch.testing.assert_close(
            fs.flagsparse_spmv_csr(prepared=plan, x=x).cpu(), expected, rtol=0, atol=0
        )


@pytest.mark.parametrize("fmt", ["csr", "coo"])
@pytest.mark.parametrize(
    "dtype,odtype",
    [
        (torch.float16, torch.float32),
        (torch.int8, torch.int32),
        (torch.complex64, torch.complex64),
    ],
)
@pytest.mark.parametrize("n", [7, 32])
def test_cuda_spmm_reads_live_values_structure_and_column_major_rhs(
    fmt, dtype, odtype, n
):
    values = torch.tensor([1, 2, -3, 4, 2, -1], dtype=dtype, device="cuda")
    if dtype.is_complex:
        values.add_(0.5j)
    cols = torch.tensor([0, 3, 2, 4, 1, 0], dtype=torch.int64, device="cuda")
    ptr = torch.tensor([0, 2, 3, 5, 6], dtype=torch.int64, device="cuda")
    rows = torch.tensor([0, 0, 1, 2, 2, 3], dtype=torch.int64, device="cuda")
    shape = (4, 5)
    if dtype.is_complex:
        if fmt == "csr":
            plan = fs.prepare_spmm_csr_route(values, cols, ptr, shape)
            v, ci, rp = plan.data, plan.kernel_indices, plan.kernel_indptr

            def call(b, **kw):
                return fs.flagsparse_spmm_csr_run(plan, b, **kw)
        else:
            plan = fs.prepare_spmm_coo_route(values, rows, cols, shape)
            v, ri, ci = plan.data, plan.row, plan.col

            def call(b, **kw):
                return fs.flagsparse_spmm_coo_run(plan, b, **kw)
    else:
        v, ci, rp, ri = values, cols, ptr, rows
        if fmt == "csr":

            def call(b, **kw):
                return fs.flagsparse_spmm_csr(
                    v, ci, rp, b, shape, out_dtype=odtype, **kw
                )
        else:

            def call(b, **kw):
                return fs.flagsparse_spmm_coo(
                    v, ri, ci, b, shape, out_dtype=odtype, **kw
                )

    b = torch.arange(5 * n, device="cuda").remainder(7).reshape(5, n).to(dtype)
    b = b.t().contiguous().t()
    previous = []
    for step in range(6):
        if step == 1:
            v.mul_(2)
        if step == 2:
            b.add_(1)
        if step == 3:
            ci[0] = 1
        if step == 4:
            if fmt == "csr":
                rp[1] = 1
            else:
                ri[0] = 2
        wide = _wide(dtype)
        dense = torch.zeros(shape, dtype=wide)
        rr = (
            torch.repeat_interleave(torch.arange(4), (rp[1:] - rp[:-1]).cpu().long())
            if fmt == "csr"
            else ri.cpu().long()
        )
        dense.index_put_((rr, ci.cpu().long()), v.cpu().to(wide), accumulate=True)
        expected = (dense @ b.cpu().to(wide)).to(odtype)
        actual = call(b)
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)
        target = torch.empty_like(actual)
        assert call(b, out=target) is target
        torch.testing.assert_close(target.cpu(), expected, rtol=1e-5, atol=1e-5)
        previous.append((actual, expected.clone()))
    for result, expected in previous:
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-5, atol=1e-5)


def test_cuda_coo_spmm_graph_replay_refreshes_reordered_values():
    v = torch.tensor([1, -2, 3, 4], dtype=torch.float16, device="cuda")
    ri = torch.tensor([2, 0, 1, 0], dtype=torch.int32, device="cuda")
    ci = torch.tensor([0, 2, 1, 3], dtype=torch.int32, device="cuda")
    b = torch.ones((4, 32), dtype=torch.float16, device="cuda")
    target = torch.empty((3, 32), device="cuda")

    def run():
        return fs.flagsparse_spmm_coo(
            v, ri, ci, b, (3, 4), out=target, out_dtype=torch.float32
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        v.mul_(2)
        b.add_(1)
        graph.replay()
        dense = torch.zeros((3, 4), dtype=torch.float64)
        dense.index_put_(
            (ri.cpu().long(), ci.cpu().long()), v.cpu().double(), accumulate=True
        )
        torch.testing.assert_close(target.cpu(), (dense @ b.cpu().double()).float())


@pytest.mark.parametrize("k", [7, 32, 64])
@pytest.mark.parametrize("strided", [False, True])
def test_cuda_sddmm_row_reuse_reads_current_operands_and_beta(k, strided):
    torch.manual_seed(51)
    cols = torch.tensor([0, 3, 2, 4, 1, 0], dtype=torch.int32, device="cuda")
    ptr = torch.tensor([0, 2, 3, 5, 6], dtype=torch.int32, device="cuda")
    plan = fs.prepare_sddmm_csr(cols, ptr, (4, 5), k_hint=k)
    a = torch.randn((4, k), dtype=torch.complex64, device="cuda")
    b = torch.randn((5, k), dtype=torch.complex64, device="cuda")
    if strided:
        a = a.t().contiguous().t()
        b = b.t().contiguous().t()
    data = torch.randn(6, dtype=torch.complex64, device="cuda")
    target = torch.empty_like(data)
    rows = torch.tensor([0, 0, 1, 2, 2, 3])
    for _ in range(4):
        a.add_(0.25 + 0.5j)
        b.mul_(0.5)
        data.add_(1)
        expected = (
            1.25
            * (
                a.cpu().to(torch.complex128)[rows]
                * b.cpu().to(torch.complex128)[cols.cpu().long()]
            ).sum(1)
            - 0.5 * data.cpu()
        ).to(data.dtype)
        actual = fs.flagsparse_sddmm_csr(
            prepared=plan, x=a, y=b, data=data, alpha=1.25, beta=-0.5, out=target
        )
        assert actual is target
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("indptr_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("op_a", ["non", "trans"])
@pytest.mark.parametrize("op_b", ["non", "trans"])
@pytest.mark.parametrize("layout", ["row", "col"])
def test_cuda_sddmm_complex_irregular_rows_and_layouts(indptr_dtype, op_a, op_b, layout):
    # Exercise row/column tile tails, empty rows and a long row in the same tile.
    torch.manual_seed(93)
    lengths = torch.tensor([0, 1, 7, 8, 9, 17, 0, 65, 2, 31, 3, 16, 0])
    m, n, k = lengths.numel(), 80, 32
    ptr = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
    rows = torch.repeat_interleave(torch.arange(m), lengths)
    cols = torch.randint(n, (rows.numel(),), dtype=torch.int32, device="cuda")
    plan = fs.prepare_sddmm_csr(cols, ptr.to(device="cuda", dtype=indptr_dtype), (m, n))
    a = torch.randn((m, k) if op_a == "non" else (k, m), device="cuda", dtype=torch.complex64)
    b = torch.randn((k, n) if op_b == "non" else (n, k), device="cuda", dtype=torch.complex64)
    if layout == "col":
        a, b = a.t().contiguous().t(), b.t().contiguous().t()
    data = torch.randn(rows.numel(), device="cuda", dtype=torch.complex64)
    target = torch.empty_like(data)
    for step in range(3):
        if step:
            a.add_(0.25 + 0.5j)
            b.mul_(0.5)
            data.add_(1)
            cols[0] = (cols[0] + 1) % n
        aa = a.cpu().to(torch.complex128)
        bb = b.cpu().to(torch.complex128)
        aa = aa if op_a == "non" else aa.t()
        bb = bb if op_b == "non" else bb.t()
        sampled = (aa[rows] * bb.t()[cols.cpu().long()]).sum(1)
        # Also cover input/output aliasing and the data-free allocation path.
        source = data if step < 2 else None
        output = target if step == 0 else data if step == 1 else None
        alpha, beta = 1.25, -0.5 if source is not None else 0.0
        expected = alpha * sampled
        if source is not None:
            expected += beta * source.cpu().to(torch.complex128)
        actual = fs.flagsparse_sddmm_csr(
            prepared=plan, x=a, y=b, data=source, alpha=alpha, beta=beta,
            out=output, op_a=op_a, op_b=op_b, dense_layout=layout,
        )
        if output is not None:
            assert actual is output
        torch.testing.assert_close(actual.cpu(), expected.to(actual.dtype), rtol=1e-5, atol=1e-5)


def test_cuda_csr_spmv_hot_launch_rechecks_output_alignment():
    v = torch.tensor([1, 2, 3, 4], dtype=torch.float32, device="cuda")
    ci = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device="cuda")
    rp = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    plan = fs.prepare_spmv_csr(v, ci, rp, (2, 2))
    x = torch.tensor([2, 3], dtype=torch.float32, device="cuda")
    for offset in (0, 1, 2, 3, 0):
        target = torch.empty(2 + offset, device="cuda")[offset:]
        for _ in range(2):
            assert fs.flagsparse_spmv_csr(prepared=plan, x=x, out=target) is target
            torch.testing.assert_close(target, torch.tensor([8.0, 18.0], device="cuda"))


def test_cuda_mixed_transpose_spmv_reads_aliased_rhs_before_output_copy():
    v = torch.arange(1, 9, device="cuda", dtype=torch.float16)
    ci = torch.arange(8, device="cuda", dtype=torch.int32).remainder(4)
    rp = torch.arange(9, device="cuda", dtype=torch.int32)
    x = torch.arange(1, 9, device="cuda", dtype=torch.float16)
    target = x.view(torch.float32)
    expected = torch.zeros(4, dtype=torch.float64)
    expected.index_add_(0, ci.cpu().long(), v.cpu().double() * x.cpu().double())
    assert (
        fs.flagsparse_spmv_csr(v, ci, rp, x, (8, 4), op="trans", out=target) is target
    )
    torch.testing.assert_close(target.cpu(), expected.float())


@pytest.mark.parametrize("m,n,nnz", [(24, 10, 3), (4096, 2048, 2048)])
def test_cuda_gemvi_cached_launches_read_live_values_and_revalidate(m, n, nnz):
    torch.manual_seed(71)
    a = torch.randn((m, n), device="cuda").t().contiguous().t()
    v = torch.randn(nnz, device="cuda")
    ci = torch.arange(nnz, device="cuda", dtype=torch.int32)
    target = torch.zeros(m, device="cuda")
    for step in range(5):
        if step == 2:
            a.mul_(0.5)
            v.add_(1)
            ci.copy_(ci.roll(1))
        if step == 3:
            v.set_(v.clone().mul_(2))
        alpha, beta = (1.25, -0.5) if step < 4 else (0.75, 0.25)
        expected = (
            alpha * (a.cpu().double()[:, ci.cpu().long()] @ v.cpu().double())
            + beta * target.cpu().double()
        ).float()
        actual = fs.flagsparse_gemvi(
            a, v, ci, target, alpha=alpha, beta=beta, validate=False
        )
        assert actual is target
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)
    v.resize_(nnz + 1)
    with pytest.raises(ValueError, match="same length"):
        fs.flagsparse_gemvi(a, v, ci, target, validate=False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.complex64])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("other_stream", [False, True])
def test_cuda_axpby_reuses_only_structure(dtype, index_dtype, other_stream):
    stream = torch.cuda.Stream() if other_stream else torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        x = torch.tensor([1, -2, 3], dtype=dtype, device="cuda")
        y = torch.arange(11, dtype=torch.float32, device="cuda").to(dtype)
        indices = torch.tensor([1, 4, 7], dtype=index_dtype, device="cuda")
        if dtype.is_complex:
            x.add_(0.5j)
            y.add_(0.25j)
        wide = torch.complex128 if dtype.is_complex else torch.float64
        for step in range(6):
            if step == 1:
                x.mul_(2)
            if step == 2:
                indices.copy_(torch.tensor([0, 3, 8], device="cuda"))
            if step == 3:
                y.add_(1)
            if step == 4:
                x.set_(x.clone().add_(1))
                y.set_(y.clone().mul_(2))
            alpha, beta = (
                (1.5, 0.5)
                if step < 5
                else ((0.5 + 0.25j, -0.25j) if dtype.is_complex else (-0.5, 0.25))
            )
            expected = y.cpu().to(wide) * beta
            expected.index_add_(0, indices.cpu().long(), x.cpu().to(wide) * alpha)
            result = fs.flagsparse_axpby(
                x, indices, y, alpha=alpha, beta=beta, validate=False
            )
            assert result is y
            torch.testing.assert_close(
                result.cpu(),
                expected.to(dtype),
                rtol=2e-3 if dtype == torch.float16 else 1e-5,
                atol=2e-3 if dtype == torch.float16 else 1e-5,
            )
        # Repeated calls with unchanged metadata reach the compiled hot entry.
        for _ in range(2):
            expected = y.cpu().to(wide) * beta
            expected.index_add_(0, indices.cpu().long(), x.cpu().to(wide) * alpha)
            fs.flagsparse_axpby(x, indices, y, alpha=alpha, beta=beta, validate=False)
            torch.testing.assert_close(
                y.cpu(), expected.to(dtype), rtol=2e-3, atol=2e-3
            )
    stream.synchronize()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.complex64])
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_cuda_csc_reuse_tracks_values_structure_and_rhs(dtype, op, index_dtype):
    values = torch.tensor([1, -2, 3, 4], dtype=dtype, device="cuda")
    if dtype.is_complex:
        values.add_(0.5j)
    rows = torch.tensor([0, 2, 1, 3], dtype=index_dtype, device="cuda")
    ptr = torch.tensor([0, 2, 3, 4], dtype=index_dtype, device="cuda")
    shape = (4, 3)
    k = shape[1] if op == "non" else shape[0]
    b = torch.arange(k * 8, dtype=torch.float32, device="cuda").reshape(k, 8).to(dtype)
    b = b.t().contiguous().t()
    wide = torch.complex128 if dtype.is_complex else torch.float64
    previous = None
    for step in range(6):
        if step == 1:
            b.add_(1)
        if step == 2:
            values.mul_(2)
        if step == 3:
            rows[0] = 1
        if step == 4:
            ptr[1] = 1
        if step == 5:
            b.set_(b.clone().mul_(0.5))
        dense = torch.sparse_csc_tensor(
            ptr.cpu().long(),
            rows.cpu().long(),
            values.cpu().to(wide),
            shape,
        ).to_dense()
        if op != "non":
            dense = dense.t().conj() if op == "conj" else dense.t()
        expected = (dense @ b.cpu().to(wide)).to(dtype)
        result = fs.flagsparse_spmm_csc(values, rows, ptr, b, shape, op=op)
        torch.testing.assert_close(result.cpu(), expected, rtol=2e-3, atol=2e-3)
        if previous is not None:
            torch.testing.assert_close(previous[0].cpu(), previous[1])
        previous = result, result.cpu().clone()
        target = torch.empty_like(result)
        returned = fs.flagsparse_spmm_csc(
            values, rows, ptr, b, shape, op=op, out=target
        )
        assert returned is target
        torch.testing.assert_close(returned.cpu(), expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize(
    "backend", ["rocm", "metax", "mthreads", "ascend", "xpu", "gcu", "iluvatar"]
)
def test_cuda_reuse_is_disabled_for_other_backend_selectors(monkeypatch, backend):
    from flagsparse.sparse_operations import _vector_cuda

    def forbidden(*args, **kwargs):
        pytest.fail("CUDA reuse was selected for another backend")

    monkeypatch.setattr(vector_ops, "_backend_name", lambda: backend)
    monkeypatch.setattr(spmm_csc, "_backend_name", lambda: backend)
    monkeypatch.setattr(_vector_cuda, "axpby", forbidden)
    monkeypatch.setattr(_vector_cuda, "try_axpby", forbidden)
    monkeypatch.setattr(spmm_csc._CUDA_CSC_PREPARED_CACHE, "get", forbidden)
    monkeypatch.setattr(spmm_csc._CUDA_CSC_HALF_VALUES_CACHE, "get", forbidden)
    x = torch.ones(2, device="cuda")
    indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    y = torch.ones(2, device="cuda")
    fs.flagsparse_axpby(x, indices, y, validate=False)
    torch.testing.assert_close(y, torch.full_like(y, 2))
    ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    b = torch.ones((2, 8), device="cuda")
    torch.testing.assert_close(fs.flagsparse_spmm_csc(x, indices, ptr, b, (2, 2)), b)


def test_cuda_axpby_hot_entry_revalidates_resized_operands():
    x = torch.ones(2, device="cuda")
    indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    y = torch.ones(4, device="cuda")
    fs.flagsparse_axpby(x, indices, y, validate=False)
    x.resize_(3)
    with pytest.raises(ValueError, match="same length"):
        fs.flagsparse_axpby(x, indices, y, validate=False)


def test_cuda_axpby_overlap_keeps_existing_update_order():
    y = torch.arange(8, dtype=torch.float32, device="cuda")
    x = y[:3]
    indices = torch.tensor([1, 3, 4], dtype=torch.int32, device="cuda")
    expected = y.cpu() * 0.5
    expected.index_add_(0, indices.cpu().long(), expected[:3].clone() * 1.5)
    fs.flagsparse_axpby(x, indices, y, alpha=1.5, beta=0.5, validate=False)
    torch.testing.assert_close(y.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.complex64])
def test_cuda_axpby_empty_and_inference_inputs(dtype):
    with torch.inference_mode():
        y = torch.ones(17, device="cuda", dtype=dtype)
        indices = torch.empty(0, device="cuda", dtype=torch.int32)
        x = torch.empty(0, device="cuda", dtype=dtype)
        for _ in range(2):
            fs.flagsparse_axpby(
                x, indices, y, alpha=float("nan"), beta=0.5, validate=False
            )
        torch.testing.assert_close(y, torch.full_like(y, 0.25))


def test_cuda_axpby_capture_replays_live_values():
    x = torch.ones(3, device="cuda")
    indices = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    y = torch.zeros(7, device="cuda")
    fs.flagsparse_axpby(x, indices, y, alpha=1.5, beta=0.5, validate=False)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fs.flagsparse_axpby(x, indices, y, alpha=1.5, beta=0.5, validate=False)
    for factor in (1, 2):
        x.fill_(factor)
        expected = y.cpu() * 0.5
        expected.index_add_(0, indices.cpu().long(), x.cpu() * 1.5)
        graph.replay()
        torch.testing.assert_close(y.cpu(), expected)


def test_cuda_axpby_index_preparation_is_private_to_each_stream():
    x = torch.ones(3, device="cuda")
    indices = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    first = torch.zeros(7, device="cuda")
    second = torch.zeros_like(first)
    fs.flagsparse_axpby(x, indices, first, validate=False)
    first.zero_()
    torch.cuda.synchronize()
    stream1, stream2 = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(stream1):
        torch.cuda._sleep(5_000_000)
        fs.flagsparse_axpby(x, indices, first, validate=False)
    with torch.cuda.stream(stream2):
        fs.flagsparse_axpby(x, indices, second, validate=False)
    stream2.synchronize()
    expected = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    torch.testing.assert_close(second.cpu(), expected)
    stream1.synchronize()
    torch.testing.assert_close(first.cpu(), expected)


def test_cuda_prepared_csc_gather_is_private_to_each_stream():
    values = torch.tensor([1 + 2j, 3 - 1j], device="cuda")
    rows = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
    ptr = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
    b = torch.ones((2, 8), device="cuda", dtype=torch.complex64)
    prepared = fs.prepare_spmm_csc_route(values, rows, ptr, (2, 2))
    fs.flagsparse_spmm_csc_run(prepared, b)
    values.mul_(2)
    torch.cuda.synchronize()
    stream1, stream2 = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(stream1):
        torch.cuda._sleep(5_000_000)
        first = fs.flagsparse_spmm_csc_run(prepared, b)
    with torch.cuda.stream(stream2):
        second = fs.flagsparse_spmm_csc_run(prepared, b)
    stream2.synchronize()
    expected = torch.tensor([[6 - 2j] * 8, [2 + 4j] * 8])
    torch.testing.assert_close(second.cpu(), expected)
    stream1.synchronize()
    torch.testing.assert_close(first.cpu(), expected)


def test_cuda_prepared_csc_inference_reads_live_values():
    with torch.inference_mode():
        values = torch.tensor([1 + 2j, 3 - 1j], device="cuda")
        rows = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
        ptr = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
        b = torch.ones((2, 8), device="cuda", dtype=torch.complex64)
        prepared = fs.prepare_spmm_csc_route(values, rows, ptr, (2, 2))
        for _ in range(2):
            values.mul_(2)
            b.add_(1)
            result = fs.flagsparse_spmm_csc_run(prepared, b)
            expected = torch.stack((values[1] * b[1], values[0] * b[0]))
            torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_cuda_scatter_reset_reads_live_values_and_indices(dtype, index_dtype):
    from flagsparse.sparse_operations.gather_scatter import _triton_scatter_impl

    v = torch.tensor([1 + 2j, 3 - 4j, 5 + 6j], dtype=dtype, device="cuda")
    idx = torch.tensor([0, 4, 8], dtype=index_dtype, device="cuda")
    target = torch.empty(11, dtype=dtype, device="cuda")
    for step in range(5):
        if step == 1:
            v.mul_(2)
        if step == 2:
            idx[0] = 2
        if step == 3:
            idx.set_(torch.tensor([1, 3, 5], dtype=index_dtype, device="cuda"))
        expected = torch.zeros_like(target)
        expected[idx.long()] = v.conj()
        _triton_scatter_impl(v.conj(), idx, 11, out=target)
        torch.testing.assert_close(target, expected, rtol=0, atol=0)

    # No-reset and explicit launch controls retain the public semantics.
    expected.fill_(7)
    expected[idx.long()] = v
    target.fill_(7)
    _triton_scatter_impl(v, idx, 11, out=target, reset_output=False)
    torch.testing.assert_close(target, expected, rtol=0, atol=0)


def test_cuda_scatter_cold_capture_does_not_publish_uninitialised_ownership():
    from flagsparse.sparse_operations.gather_scatter import _triton_scatter_impl

    # Compile before capture, using distinct indices so ownership remains cold.
    v = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex128, device="cuda")
    idx = torch.tensor([1, 4], dtype=torch.int32, device="cuda")
    target = torch.empty(7, dtype=v.dtype, device="cuda")
    _triton_scatter_impl(v, idx.clone(), 7, out=target)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _triton_scatter_impl(v, idx, 7, out=target)
    eager = _triton_scatter_impl(v, idx, 7)
    expected = torch.zeros_like(target)
    expected[idx.long()] = v
    torch.testing.assert_close(eager, expected, rtol=0, atol=0)
    v.mul_(2)
    graph.replay()
    torch.testing.assert_close(target, expected * 2, rtol=0, atol=0)


@pytest.mark.parametrize("n", [0, 7, 32, 65])
def test_cuda_half_coo_spmm_preserves_wide_duplicate_sums_and_live_plan(n):
    v = torch.tensor([2048, 1, -2048, 2, 3], dtype=torch.float16, device="cuda")
    ri = torch.tensor([0, 0, 0, 2, 2], dtype=torch.int32, device="cuda")
    ci = torch.tensor([0, 0, 1, 0, 2], dtype=torch.int32, device="cuda")
    plan = fs.prepare_spmm_coo_route(v, ri, ci, (4, 3))
    b = torch.ones((n, 3), dtype=v.dtype, device="cuda").t()
    for step in range(4):
        if step == 1:
            b.mul_(2)
        if step == 2:
            plan.data.add_(1)
        if step == 3:
            plan.row[0] = 3
        dense = torch.zeros((4, 3), dtype=torch.float64)
        dense.index_put_(
            (plan.row.cpu().long(), plan.col.cpu().long()),
            plan.data.cpu().double(),
            accumulate=True,
        )
        expected = (dense @ b.cpu().double()).half()
        result = fs.flagsparse_spmm_coo_run(plan, b)
        assert result.dtype == torch.float16
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    if n:
        # A live canonical fp32 sum (2049) must survive through the kernel.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = fs.flagsparse_spmm_coo_run(plan, b)
        plan.data.mul_(2)
        graph.replay()
        torch.testing.assert_close(result.cpu(), expected * 2, rtol=0, atol=0)


@pytest.mark.parametrize("pointer_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("inference", [False, True])
def test_cuda_spgemm_full_call_invalidates_structure_and_reads_live_values(
    pointer_dtype, inference
):
    with torch.inference_mode(inference):
        a = torch.tensor([1, 2, -3, 4], dtype=torch.float32, device="cuda")
        ai = torch.tensor([0, 2, 1, 2], dtype=torch.int32, device="cuda")
        ap = torch.tensor([0, 2, 2, 4], dtype=pointer_dtype, device="cuda")
        b = torch.tensor([1, -2, 3, 5], dtype=torch.float32, device="cuda")
        bi = torch.tensor([0, 3, 2, 1], dtype=torch.int32, device="cuda")
        bp = torch.tensor([0, 2, 3, 4], dtype=pointer_dtype, device="cuda")
        for step in range(8):
            if step == 1:
                a.mul_(2)
                b.add_(1)
            if step == 2:
                ai[0] = 1
            if step == 3:
                ap[1] = 1
            if step == 4:
                bi.set_(torch.tensor([1, 1, 3, 2], dtype=bi.dtype, device="cuda"))
            if step == 5:
                bp[1] = 1
            if step == 6:
                a.set_(a.clone().add_(2))
            left = torch.sparse_csr_tensor(ap.cpu(), ai.cpu(), a.cpu(), (3, 3))
            right = torch.sparse_csr_tensor(bp.cpu(), bi.cpu(), b.cpu(), (3, 4))
            expected = left.to_dense() @ right.to_dense()
            result = fs.flagsparse_spgemm_csr(a, ai, ap, (3, 3), b, bi, bp, (3, 4))
            got = torch.sparse_csr_tensor(
                result[2].cpu(), result[1].cpu(), result[0].cpu(), result[3]
            ).to_dense()
            torch.testing.assert_close(got, expected)
            # Caller-owned outputs cannot corrupt private preparation metadata.
            result[0].zero_()
            result[1].zero_()
            result[2].zero_()
        ai[0] = 3
        with pytest.raises(IndexError, match="a_indices out of range"):
            fs.flagsparse_spgemm_csr(a, ai, ap, (3, 3), b, bi, bp, (3, 4))


def test_cuda_spgemm_preparation_is_private_to_each_stream():
    a = torch.tensor([1.0, 2.0], device="cuda")
    ai = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    ap = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    b = torch.tensor([3.0, 4.0], device="cuda")
    bi = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    bp = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")

    def run():
        return fs.flagsparse_spgemm_csr(a, ai, ap, (2, 2), b, bi, bp, (2, 2))

    run()
    torch.cuda.synchronize()
    outputs = []
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        with torch.cuda.stream(stream):
            outputs.append(run())
    for stream in streams:
        stream.synchronize()
    for result in outputs:
        got = torch.sparse_csr_tensor(
            result[2].cpu(), result[1].cpu(), result[0].cpu(), result[3]
        ).to_dense()
        torch.testing.assert_close(got, torch.tensor([[0.0, 3.0], [8.0, 0.0]]))
