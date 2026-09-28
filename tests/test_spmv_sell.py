# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark sliced-ELLPACK SpMV over .mtx matrices.

Each matrix is read as CSR, converted once to cuSPARSE SELL (``csr_to_sell``,
outside the timed window, like cuSPARSE's own descriptor setup), and timed as:

  triton_ms    flagsparse_spmv_sell
  cusparse_ms  cusparseSpMV with CUSPARSE_SPMV_SELL_ALG1 (CUDA only; empty elsewhere)
  pytorch_ms   torch.sparse CSR @ x on the same matrix (a different format, so it
               is a reference point, not a like-for-like baseline)

``--dtypes`` takes ``value[:out]`` tokens: ``float16:float32`` is f16 in, f32 out,
``int8`` is int8 in, int32 out. Accuracy is checked against a CPU float64 /
complex128 SpMV; ``status`` is PASS / FAIL per row, ERROR when FlagSparse raised.
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.is_dir():
    sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from benchmark_utils import accelerator_device  # noqa: E402
from mtx_fast import read_scipy_csr  # noqa: E402

import cusparse_generic_baseline as cusparse_baseline  # noqa: E402
import flagsparse as fs  # noqa: E402
from flagsparse.sparse_operations._common import _benchmark_cuda_op  # noqa: E402

# The q4 list's SELL variants: f32, f16, c32, i8 -> i32 (plus f16 -> f32).
DEFAULT_DTYPES = "float32,float16,complex64,int8,float16:float32"
_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    "int8": torch.int8,
    "int32": torch.int32,
}
_DEFAULT_OUT = {torch.int8: torch.int32}
_TOL = {torch.float16: 2e-3, torch.bfloat16: 1e-2, torch.float32: 1e-4, torch.complex64: 1e-4}
_TAGS = {torch.float16: "f16", torch.bfloat16: "bf16", torch.float32: "f32",
         torch.float64: "f64", torch.complex64: "c32", torch.complex128: "c64",
         torch.int8: "i8", torch.int32: "i32"}
FIELDS = [
    "variant",
    "matrix",
    "value_dtype",
    "out_dtype",
    "index_dtype",
    "op",
    "n_rows",
    "n_cols",
    "nnz",
    "slice_size",
    "sell_values",
    "triton_ms",
    "cusparse_ms",
    "pytorch_ms",
    "triton_speedup_vs_cusparse",
    "triton_speedup_vs_pytorch",
    "triton_max_error",
    "cusparse_max_error",
    "cusparse_reason",
    "pytorch_reason",
    "status",
    "error",
]


def _name(dtype):
    return str(dtype).replace("torch.", "")


def _parse_dtypes(raw):
    out = []
    for token in (t.strip() for t in str(raw).split(",") if t.strip()):
        value, _, result = token.partition(":")
        if value not in _DTYPES or (result and result not in _DTYPES):
            raise ValueError(f"unsupported dtype token: {token}")
        vdt = _DTYPES[value]
        out.append((vdt, _DTYPES[result] if result else _DEFAULT_OUT.get(vdt, vdt)))
    return out


def _rand_like(n, dtype, gen):
    if dtype == torch.int8:
        return torch.randint(-3, 4, (n,), dtype=dtype, generator=gen)
    if dtype.is_complex:
        return torch.randn(n, dtype=dtype, generator=gen)
    return torch.randn(n, generator=gen).to(dtype)


def _values(csr, dtype, gen):
    """Matrix values in ``dtype``; int8 cannot hold arbitrary .mtx values, so it is drawn."""
    if dtype == torch.int8:
        return torch.randint(-3, 4, (csr.nnz,), dtype=dtype, generator=gen)
    data = csr.data
    if np.iscomplexobj(data) and not dtype.is_complex:
        data = data.real
    if dtype.is_complex and not np.iscomplexobj(data):
        imag = torch.randn(csr.nnz, dtype=torch.float64, generator=gen)
        return torch.complex(torch.from_numpy(np.asarray(data, dtype=np.float64)), imag).to(dtype)
    t = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float64))
    if dtype in (torch.float16, torch.bfloat16):
        t = t / max(1.0, float(t.abs().max()))  # keep half-precision products finite
    return t.to(dtype)


def _wide(t):
    return t.detach().cpu().to(torch.complex128 if t.is_complex() else torch.float64)


def _error(got, ref):
    return float((_wide(got) - ref).abs().max() / max(1.0, float(ref.abs().max())))


def _passes(out_dtype, err):
    if not (out_dtype.is_floating_point or out_dtype.is_complex):
        return err == 0.0
    return err <= _TOL.get(out_dtype, 1e-10)


def _ratio(base, ours):
    return None if base is None or not ours else base / ours


def _run(path, csr, vdt, odt, slice_size, warmup, iters, gen):
    dev = accelerator_device()
    m, k = csr.shape
    # Variant naming: one token when the output keeps the value type, two when it
    # widens (f16f32, i8i32); int8's default int32 output is still "i8i32".
    tag = _TAGS[vdt] if odt == vdt else _TAGS[vdt] + _TAGS[odt]
    row = {
        "variant": f"spmv_sell_{tag}_int_non",
        "matrix": os.path.basename(path),
        "value_dtype": _name(vdt),
        "out_dtype": _name(odt),
        "index_dtype": "int32",
        "op": "non",
        "n_rows": m,
        "n_cols": k,
        "nnz": csr.nnz,
        "slice_size": slice_size,
    }
    vals_cpu = _values(csr, vdt, gen)
    x_cpu = _rand_like(k, vdt, gen)
    ptr = torch.from_numpy(csr.indptr.astype(np.int32))
    cols = torch.from_numpy(csr.indices.astype(np.int32))
    ref_csr = torch.sparse_csr_tensor(ptr.long(), cols.long(), _wide(vals_cpu), (m, k))
    ref = ref_csr @ _wide(x_cpu)

    data, cols_d, ptr_d, x = (t.to(dev) for t in (vals_cpu, cols, ptr, x_cpu))
    values, sell_cols, offsets = fs.csr_to_sell(data, cols_d, ptr_d, m, slice_size)
    row["sell_values"] = int(values.numel())
    out_arg = None if odt == _DEFAULT_OUT.get(vdt, vdt) else odt

    def ours():
        return fs.flagsparse_spmv_sell(
            values, sell_cols, offsets, x, (m, k), slice_size=slice_size,
            out_dtype=out_arg, validate=False,
        )

    first = fs.flagsparse_spmv_sell(values, sell_cols, offsets, x, (m, k), slice_size=slice_size, out_dtype=out_arg)
    row["triton_max_error"] = _error(first, ref)
    _, row["triton_ms"] = _benchmark_cuda_op(ours, warmup, iters)

    try:
        compute = torch.float32 if vdt in (torch.float16, torch.bfloat16, torch.int8) else vdt
        A = torch.sparse_csr_tensor(ptr_d.long(), cols_d.long(), data.to(compute), (m, k))
        xc = x.to(compute)
        _, row["pytorch_ms"] = _benchmark_cuda_op(lambda: A @ xc, warmup, iters)
    except Exception as exc:  # e.g. MUSA registers no sparse matmul
        row["pytorch_reason"] = f"{type(exc).__name__}: {exc}"

    reason = cusparse_baseline.skip_reason()
    if reason is None:
        try:
            y = torch.empty(m, dtype=odt, device=dev)
            compute = {torch.float16: torch.float32, torch.bfloat16: torch.float32}.get(vdt, odt)
            plan = cusparse_baseline.prepare_spmv_sell(values, sell_cols, offsets, x, y, (m, k), slice_size, compute)
            plan.run()
            row["cusparse_max_error"] = _error(y, ref)
            _, row["cusparse_ms"] = _benchmark_cuda_op(plan.run, warmup, iters)
            plan.close()
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
    row["cusparse_reason"] = reason
    row["triton_speedup_vs_cusparse"] = _ratio(row.get("cusparse_ms"), row["triton_ms"])
    row["triton_speedup_vs_pytorch"] = _ratio(row.get("pytorch_ms"), row["triton_ms"])
    row["status"] = "PASS" if _passes(odt, row["triton_max_error"]) else "FAIL"
    return row


def _matrix_paths(inputs):
    paths = []
    for item in inputs or [str(PROJECT_ROOT / "tests" / "data")]:
        if os.path.isdir(item):
            paths.extend(sorted(glob.glob(os.path.join(item, "*.mtx"))))
        elif item.endswith(".mtx"):
            paths.append(item)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mtx", nargs="*", help=".mtx files or directories (default tests/data)")
    parser.add_argument("--csv", required=True, help="output CSV path")
    parser.add_argument("--dtypes", default=DEFAULT_DTYPES, help="value[:out],...")
    parser.add_argument("--slice-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    dtypes = _parse_dtypes(args.dtypes)
    paths = _matrix_paths(args.mtx)
    if not paths:
        parser.error("no .mtx files found")
    gen = torch.Generator().manual_seed(args.seed)
    rows = []
    for path in paths:
        print(f"RUNNING: {os.path.basename(path)}", flush=True)
        csr = read_scipy_csr(path)
        for vdt, odt in dtypes:
            try:
                row = _run(path, csr, vdt, odt, args.slice_size, args.warmup, args.iters, gen)
            except Exception as exc:
                row = {
                    "matrix": os.path.basename(path),
                    "value_dtype": _name(vdt),
                    "out_dtype": _name(odt),
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            print(
                f"  {row['value_dtype']:>9s}->{row['out_dtype']:<9s} {row['status']:5s} "
                f"ours={row.get('triton_ms')} cusparse={row.get('cusparse_ms')} torch={row.get('pytorch_ms')}",
                flush=True,
            )
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in FIELDS})
    return 1 if any(r.get("status") == "ERROR" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
