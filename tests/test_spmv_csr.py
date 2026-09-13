#!/usr/bin/env python3
"""CSR SpMV benchmark entry point (not a pytest accuracy module)."""

import argparse
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

__test__ = False
ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "matrix",
    "dtype",
    "index_dtype",
    "indptr_dtype",
    "op",
    "alg",
    "m",
    "n",
    "nnz",
    "status",
    "reason",
    "ref",
    "max_error",
    "ms",
    "gpu_ms",
    "process_cpu_ms",
    "vendor_ms",
    "vendor_backend",
    "vendor_alg",
    "vendor_status",
    "vendor_reason",
    "vendor_max_error",
    "speedup_vs_vendor",
    "alg_requested",
    "alg_resolved",
    "implementation_version",
    "backend",
    "target",
    "arch",
    "config",
    "config_source",
    "config_rejections",
    "compute_dtype",
    "input_indices_dtype",
    "input_indptr_dtype",
    "indices_dtype",
    "actual_indptr_dtype",
    "index_fallback_applied",
    "index_fallback_reason",
    "device",
    "torch_version",
    "triton_version",
    "hip_version",
    "cuda_version",
    "python_version",
    "commit",
    "warmup",
    "iters",
]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="*", help="MatrixMarket files or directories")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--dtypes", "--dtype", default="float32,float64")
    p.add_argument("--index-dtypes", "--index-dtype", default="int32,int64")
    p.add_argument(
        "--indptr-dtypes",
        default=None,
        help="defaults to index dtype; comma-separated for mixed indices",
    )
    p.add_argument("--ops", default="non", help="non,trans,conj or all")
    p.add_argument(
        "--alg",
        default="auto",
        help="auto, compare, or comma-separated concrete algorithms",
    )
    p.add_argument(
        "--config",
        type=json.loads,
        default=None,
        help="explicit JSON launch configuration",
    )
    p.add_argument("--csv-csr", "--csv", dest="csv_csr")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--timing", action="store_true")
    p.add_argument(
        "--no-vendor", "--no-cusparse", dest="no_vendor", action="store_true"
    )
    p.add_argument("--fail-fast", action="store_true")
    return p


def choices(value, allowed):
    names = list(allowed) if value == "all" else value.replace(" ", ",").split(",")
    names = list(dict.fromkeys(n for n in names if n))
    if not names or any(n not in allowed for n in names):
        raise ValueError(
            f"expected a nonempty selection from {tuple(allowed)}, got {value!r}"
        )
    return names


def load_mtx_to_csr_torch(file_path, dtype=None, device=None):
    import torch
    from tests.mtx_fast import load_csr

    return load_csr(
        file_path, dtype=torch.float32 if dtype is None else dtype, device=device
    )


def synthetic_cases(torch, dtype, device):
    # Boundary rows, irregular tails, empties, and a long row requiring >256 partials.
    cases = [
        ("empty", [], 11),
        ("empty_rows", [0] * 19, 13),
        ("short_tail", [0, 1, 7, 8, 9, 31, 32] * 3, 127),
        ("mixed_boundaries", [0, 33, 1024, 1, 1025, 32, 2049, 7], 4099),
        ("single_long", [1024 * 257 + 1], 1024 * 257 + 7),
    ]
    for name, lengths, n in cases:
        ptr = torch.tensor([0] + lengths, dtype=torch.int64).cumsum(0)
        count = int(ptr[-1])
        col = torch.arange(count, dtype=torch.int64) % n
        a = torch.randn(count, dtype=dtype)
        yield name, (a.to(device), col.to(device), ptr.to(device), (len(lengths), n))


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.warmup < 0 or args.iters < 1:
        p.error("require warmup >= 0 and iters >= 1")
    if not args.paths and not args.synthetic:
        p.error("supply .mtx paths or --synthetic")
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    import torch
    import triton
    import flagsparse as fs
    from flagsparse.sparse_operations import _common as common
    from flagsparse.sparse_operations._spmv_csr_benchmark import (
        measure_route,
        measure_vendor,
        golden_csr,
        check_result,
    )
    from tests.pytest.accuracy_utils import close_tolerances

    dtypes = choices(
        args.dtypes,
        ("float16", "bfloat16", "float32", "float64", "complex64", "complex128"),
    )
    indices = choices(args.index_dtypes, ("int32", "int64"))
    ptrtypes = (
        choices(args.indptr_dtypes, ("int32", "int64")) if args.indptr_dtypes else None
    )
    ops = choices(args.ops, ("non", "trans", "conj"))
    if not common._ACCEL.is_available():
        raise RuntimeError(f"no accelerator available for {common._backend_name()}")
    device = torch.device(common._ACCEL_DEVICE_TYPE)
    paths = []
    for item in args.paths:
        path = Path(item)
        paths.extend(sorted(path.rglob("*.mtx")) if path.is_dir() else [path])
    paths = list(dict.fromkeys(paths))
    if args.paths and not paths:
        p.error("no .mtx files found")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        if subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip():
            commit += "+dirty"
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    environment = dict(
        device=common._ACCEL.get_device_name(device),
        torch_version=torch.__version__,
        triton_version=triton.__version__,
        hip_version=torch.version.hip,
        cuda_version=torch.version.cuda,
        python_version=platform.python_version(),
        commit=commit,
        warmup=args.warmup,
        iters=args.iters,
    )
    print(json.dumps(environment, ensure_ascii=False))
    print(
        "Native CSR SpMV; Ref=CPU FP64/complex128 (correctness only). ms=process_cpu_ms+gpu_ms; phase diagnostics run separately. Vendor=N/A when native same-device CSR is unavailable."
    )
    fields = FIELDS + (["process_gpu_ms", "compute_ms"] if args.timing else [])
    csv_file = None
    writer = None
    failures = 0
    if args.csv_csr:
        csv_file = open(args.csv_csr, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()

    def emit(row):
        if writer:
            writer.writerow(
                {
                    key: (
                        "N/A"
                        if value is None
                        else (
                            json.dumps(value, sort_keys=True)
                            if isinstance(value, (dict, list))
                            else value
                        )
                    )
                    for key, value in row.items()
                }
            )
            csv_file.flush()
        print(
            f"{row['matrix']} {row['dtype']} {row['index_dtype']}/{row['indptr_dtype']} {row['op']} {row['alg']}: {row['status']} ms={row.get('ms', 'N/A')} {row.get('reason') or ''}"
        )

    try:
        for dtype_name in dtypes:
            dtype = getattr(torch, dtype_name)
            torch.manual_seed(2026)

            def inputs():
                if args.synthetic:
                    yield from synthetic_cases(torch, dtype, device)
                for path in paths:
                    try:
                        yield str(path), load_mtx_to_csr_torch(path, dtype, device)
                    except Exception as exc:
                        yield str(path), exc

            for name, source in inputs():
                for index_name in indices:
                    for ptr_name in ptrtypes or (index_name,):
                        for op in ops:
                            base = dict(
                                environment,
                                matrix=name,
                                dtype=dtype_name,
                                index_dtype=index_name,
                                indptr_dtype=ptr_name,
                                op=op,
                                ref="CPU FP64/complex128 scatter (correctness only)",
                            )
                            algorithms = (
                                fs.list_spmv_csr_algorithms(
                                    op, dtype, common._backend_name()
                                )
                                if args.alg == "compare"
                                else choices(
                                    args.alg, ("auto",) + fs.list_spmv_csr_algorithms()
                                )
                            )
                            try:
                                if isinstance(source, Exception):
                                    raise source
                                data, ci, rp, shape = source
                                for tensor, requested in (
                                    (ci, index_name),
                                    (rp, ptr_name),
                                ):
                                    if (
                                        requested == "int32"
                                        and tensor.numel()
                                        and int(tensor.max()) > 2147483647
                                    ):
                                        raise ValueError(
                                            "input indices cannot be represented safely as int32"
                                        )
                                ci, rp = ci.to(getattr(torch, index_name)), rp.to(
                                    getattr(torch, ptr_name)
                                )
                                x = torch.randn(
                                    shape[1] if op == "non" else shape[0],
                                    dtype=dtype,
                                    device=device,
                                )
                                ref = golden_csr(data, ci, rp, x, shape, op)
                                rtol, atol = close_tolerances(dtype)
                                base.update(m=shape[0], n=shape[1], nnz=data.numel())
                                vendor = dict(
                                    vendor_ms=None,
                                    vendor_backend="N/A",
                                    vendor_alg="N/A",
                                    vendor_status="N/A",
                                    vendor_reason=(
                                        "disabled" if args.no_vendor else None
                                    ),
                                )
                                if not args.no_vendor:
                                    try:
                                        vendor.update(
                                            measure_vendor(
                                                data,
                                                ci,
                                                rp,
                                                x,
                                                shape,
                                                op,
                                                args.warmup,
                                                args.iters,
                                            )
                                        )
                                        value = vendor.pop("values", None)
                                        if value is not None:
                                            ok, err = check_result(
                                                value, ref, rtol, atol
                                            )
                                            vendor.update(
                                                vendor_status="PASS" if ok else "FAIL",
                                                vendor_max_error=err,
                                            )
                                            if not ok:
                                                vendor["vendor_reason"] = (
                                                    "vendor correctness check failed"
                                                )
                                    except Exception as exc:
                                        vendor.update(
                                            vendor_status="N/A", vendor_reason=str(exc)
                                        )
                                base.update(vendor)
                            except Exception as exc:
                                for alg in algorithms:
                                    emit(
                                        dict(
                                            base,
                                            alg=alg,
                                            status="FAIL",
                                            reason=str(exc),
                                        )
                                    )
                                    failures += 1
                                if args.fail_fast:
                                    raise
                                continue
                            for alg in algorithms:
                                row = dict(base, alg=alg)
                                try:
                                    spec = (
                                        fs.get_spmv_csr_algorithm_spec(alg)
                                        if alg != "auto"
                                        else None
                                    )
                                    if (
                                        args.alg == "compare"
                                        and spec
                                        and index_name not in spec["index_dtypes"]
                                    ):
                                        continue
                                    prepared = fs.prepare_spmv_csr(
                                        data,
                                        ci,
                                        rp,
                                        shape,
                                        op=op,
                                        alg=alg,
                                        config=args.config,
                                    )
                                    value, meta = measure_route(
                                        prepared,
                                        x,
                                        args.warmup,
                                        args.iters,
                                        args.timing,
                                    )
                                    ok, error = check_result(value, ref, rtol, atol)
                                    row.update(
                                        meta,
                                        indptr_dtype=ptr_name,
                                        actual_indptr_dtype=meta["indptr_dtype"],
                                        max_error=error,
                                        status="PASS" if ok else "FAIL",
                                        reason=(
                                            None if ok else "correctness check failed"
                                        ),
                                    )
                                    row["speedup_vs_vendor"] = (
                                        row["vendor_ms"] / row["ms"]
                                        if ok
                                        and row["vendor_status"] == "PASS"
                                        and row["ms"] > 0
                                        else None
                                    )
                                    failures += not ok
                                except NotImplementedError as exc:
                                    row.update(status="SKIP", reason=str(exc))
                                except Exception as exc:
                                    row.update(status="FAIL", reason=str(exc))
                                    failures += 1
                                emit(row)
                                if args.fail_fast and row["status"] == "FAIL":
                                    raise RuntimeError(row["reason"])
    finally:
        if csv_file:
            csv_file.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
