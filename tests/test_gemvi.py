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

"""Benchmark flagsparse_gemvi: dense A times a sparse vector (cusparse<t>gemvi).

Synthetic cases (m, n, nnz) like tests/test_vector_ops.py, since the operator
takes a dense matrix and a sparse vector, not a matrix file. A is column-major,
as cusparse<t>gemvi requires. Each row carries:

  triton_ms    FlagSparse
  cusparse_ms  native cusparseSgemvi / cusparseCgemvi (CUDA only; empty
               elsewhere, with the reason in cusparse_reason)
  pytorch_ms   A.index_select + matrix-vector product in PyTorch

and the speedup columns the unified runner reads (vendor first, then PyTorch).
The timed FlagSparse call passes ``validate=False`` (no index range check, a host
sync neither baseline performs); the accuracy check runs once with it on, against
a CPU float64 / complex128 golden. ``status`` is PASS or FAIL per row, ERROR when
FlagSparse raised.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.is_dir():
    sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from benchmark_utils import accelerator_device  # noqa: E402

import cusparse_generic_baseline as cusparse_baseline  # noqa: E402
import flagsparse as fs  # noqa: E402
from flagsparse.sparse_operations._common import _benchmark_cuda_op  # noqa: E402

DEFAULT_CASES = "8192:4096:256,16384:8192:2048,65536:4096:1024,4096:65536:4096"
# (value dtype, op): the delivered variants gemvi_f32_int_non / _trans, gemvi_c32_int_non.
DEFAULT_VARIANTS = "float32:non,float32:trans,complex64:non"
_DTYPES = {"float32": torch.float32, "float64": torch.float64,
           "complex64": torch.complex64, "complex128": torch.complex128}
_TAGS = {"float32": "f32", "float64": "f64", "complex64": "c32", "complex128": "c64"}
_TOL = {torch.float32: 1e-4, torch.complex64: 1e-4}
FIELDS = [
    "variant",
    "case_id",
    "op",
    "value_dtype",
    "index_dtype",
    "m",
    "n",
    "nnz",
    "triton_ms",
    "cusparse_ms",
    "pytorch_ms",
    "triton_speedup_vs_cusparse",
    "triton_speedup_vs_pytorch",
    "triton_max_error",
    "cusparse_max_error",
    "pytorch_max_error",
    "cusparse_reason",
    "pytorch_reason",
    "status",
    "error",
]


def _wide(t):
    return t.detach().cpu().to(torch.complex128 if t.is_complex() else torch.float64)


def _error(got, ref):
    return float((_wide(got) - ref).abs().max() / max(1.0, float(ref.abs().max())))


def _ratio(base, ours):
    return None if base is None or not ours else base / ours


def _run_case(dtype, op, m, n, nnz, warmup, iters, gen):
    dev = accelerator_device()
    name = str(dtype).replace("torch.", "")
    n_in, n_out = (n, m) if op == "non" else (m, n)
    row = {
        "variant": f"gemvi_{_TAGS[name]}_int_{op}",
        "case_id": f"gemvi|{name}|{op}|m={m}|n={n}|nnz={nnz}",
        "op": op,
        "value_dtype": name,
        "index_dtype": "int32",
        "m": m,
        "n": n,
        "nnz": nnz,
    }
    A_cpu = torch.randn((m, n), dtype=dtype, generator=gen)
    idx_cpu = torch.randperm(n_in, generator=gen)[:nnz].to(torch.int32)
    x_cpu = torch.randn((nnz,), dtype=dtype, generator=gen)
    A = A_cpu.to(dev).t().contiguous().t()  # column-major
    idx, x = idx_cpu.to(dev), x_cpu.to(dev)
    y = torch.empty(n_out, dtype=dtype, device=dev)
    opA = _wide(A_cpu) if op == "non" else _wide(A_cpu).t()
    ref = opA[:, idx_cpu.long()] @ _wide(x_cpu)

    first = fs.flagsparse_gemvi(A, x, idx, op=op)
    row["triton_max_error"] = _error(first, ref)
    _, row["triton_ms"] = _benchmark_cuda_op(
        lambda: fs.flagsparse_gemvi(A, x, idx, y, op=op, validate=False), warmup, iters
    )

    gather_dim = 1 if op == "non" else 0
    idx64 = idx.to(torch.int64)

    def torch_op():
        picked = A.index_select(gather_dim, idx64)
        return picked @ x if op == "non" else x @ picked

    try:
        row["pytorch_max_error"] = _error(torch_op(), ref)
        _, row["pytorch_ms"] = _benchmark_cuda_op(torch_op, warmup, iters)
    except Exception as exc:
        row["pytorch_reason"] = f"{type(exc).__name__}: {exc}"

    reason = cusparse_baseline.skip_reason()
    if reason is None:
        plan = None
        try:
            y_cu = torch.zeros(n_out, dtype=dtype, device=dev)
            plan = cusparse_baseline.prepare_gemvi(A, x, idx, y_cu, op, 1.0, 0.0)
            row["cusparse_max_error"] = _error(plan.run(), ref)
            if row["cusparse_max_error"] > _TOL.get(dtype, 1e-10):
                raise RuntimeError(f"result off by {row['cusparse_max_error']:.3g}; baseline not used")
            _, row["cusparse_ms"] = _benchmark_cuda_op(plan.run, warmup, iters)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
        finally:
            if plan is not None:
                plan.close()
    row["cusparse_reason"] = reason
    row["triton_speedup_vs_cusparse"] = _ratio(row.get("cusparse_ms"), row["triton_ms"])
    row["triton_speedup_vs_pytorch"] = _ratio(row.get("pytorch_ms"), row["triton_ms"])
    row["status"] = "PASS" if row["triton_max_error"] <= _TOL.get(dtype, 1e-10) else "FAIL"
    return row


def _parse_cases(raw):
    cases = []
    for item in str(raw).split(","):
        m, n, nnz = (int(v) for v in item.split(":"))
        if m <= 0 or n <= 0 or nnz <= 0:
            raise ValueError(f"case {item}: need m, n, nnz > 0")
        cases.append((m, n, nnz))
    return cases


def _parse_variants(raw):
    variants = []
    for item in str(raw).split(","):
        name, _, op = item.strip().partition(":")
        if name not in _DTYPES or op not in ("non", "trans"):
            raise ValueError(f"variant {item!r}: need <dtype>:<non|trans>")
        variants.append((name, op))
    return variants


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv-summary", required=True, help="output CSV path")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS, help="dtype:op,...")
    parser.add_argument("--cases", default=DEFAULT_CASES, help="m:n:nnz,...")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        variants, cases = _parse_variants(args.variants), _parse_cases(args.cases)
    except ValueError as exc:
        parser.error(str(exc))
    gen = torch.Generator().manual_seed(args.seed)
    rows = []
    for name, op in variants:
        for m, n, nnz in cases:
            if nnz > (n if op == "non" else m):
                continue
            try:
                row = _run_case(_DTYPES[name], op, m, n, nnz, args.warmup, args.iters, gen)
            except Exception as exc:
                row = {
                    "variant": f"gemvi_{_TAGS[name]}_int_{op}",
                    "case_id": f"gemvi|{name}|{op}|m={m}|n={n}|nnz={nnz}",
                    "op": op,
                    "value_dtype": name,
                    "m": m,
                    "n": n,
                    "nnz": nnz,
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            print(
                f"{row['case_id']:52s} {row.get('status'):5s} "
                f"ours={row.get('triton_ms')} cusparse={row.get('cusparse_ms')} "
                f"torch={row.get('pytorch_ms')}"
            )
    os.makedirs(os.path.dirname(os.path.abspath(args.csv_summary)) or ".", exist_ok=True)
    with open(args.csv_summary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in FIELDS})
    return 1 if any(r.get("status") == "ERROR" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
