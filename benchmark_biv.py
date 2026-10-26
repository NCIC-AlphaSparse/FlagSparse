#!/usr/bin/env python3
"""Run the BI-V accuracy/performance suite through the shared pytest runner."""

from __future__ import annotations

import os
import sys
from datetime import date

from run_flagsparse_pytest import main


def run() -> int:
    os.environ.setdefault("FLAGSPARSE_BACKEND", "iluvatar")
    os.environ.setdefault("FLAGSPARSE_ILUVATAR_VENDOR", "cupy_cusparse")
    # Time spmv_sell through prepare_spmv_sell, like spmv_csr / spmv_coo: unset, the
    # delivered number depended on whoever launched the run (0.81 vs 0.62 on 10-26/27).
    os.environ.setdefault("FLAGSPARSE_SELL_PREPARED", "auto")
    os.environ["FLAGSPARSE_BASELINE_POLICY"] = "biv"
    if "--help" not in sys.argv and "-h" not in sys.argv:
        defaults = [
                "--ops",
                "gather,scatter,axpby,spvv,spmv_sell,spmv_csr,spmv_coo,spmv_csc,spmm_csr,spmm_coo,spmm_csc,spgemm_csr,sddmm_csr",
                "--phase",
                "both",
                "--mode",
                "normal",
                "--gpus",
                "0",
                "--timeout",
                "3600",
                "--benchmark-input",
                "tests/data",
                "--benchmark-warmup",
                "5",
                "--benchmark-iters",
                "20",
                "--h800-reference",
                "--pytest-args",
                "-k 'test_q4_variant or (((float or half or f16 or f32) and not (double or float64 or bfloat16 or complex64 or complex128)) or (complex64 and (gather or scatter)))'",
                "--op-benchmark-args",
                "gather=--value-dtypes float16,float32,complex64",
                "--op-benchmark-args",
                "scatter=--value-dtypes float16,float32,complex64,int8",
                "--op-benchmark-args",
                "spmv_csr=--dtypes float32 --alg auto",
                "--op-benchmark-args",
                "spmv_coo=--dtypes float32",
                "--op-benchmark-args",
                "spmv_csc=--dtypes float32,complex64",
                "--op-benchmark-args",
                "spmm_csr=--dtypes float16,float32",
                "--op-benchmark-args",
                "spmm_coo=--dtypes float16,float32",
                "--op-benchmark-args",
                "spmm_csc=--dtypes float32,complex64",
                "--op-benchmark-args",
                "sddmm_csr=--dtype float32",
                "--op-benchmark-args",
                "spgemm_csr=--dtypes float32",
                "--results-dir",
                f"results_biv_65_{date.today().strftime('%Y%m%d')}",
            ]
        sys.argv[1:] = defaults + sys.argv[1:]
    return main(
        description=(
            "Run all FlagSparse delivery variants on Iluvatar BI-V with "
            "verified local baselines and H800 fallback."
        )
    )


if __name__ == "__main__":
    raise SystemExit(run())
