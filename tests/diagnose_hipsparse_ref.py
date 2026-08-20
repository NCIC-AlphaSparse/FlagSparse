#!/usr/bin/env python3
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

"""Phase-by-phase smoke probe for DCU hip-python/hipSPARSE references.

This is intentionally tiny and verbose. If a benchmark prints only its table
header and then stalls, run one operation here and the last printed phase shows
which hipSPARSE wrapper call or synchronization point is hanging.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def phase(message: str) -> None:
    print(f"[phase] {message}", flush=True)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception as exc:
        return f"unavailable ({exc.__class__.__name__}: {exc})"


def sync(torch_module, label: str) -> None:
    phase(f"{label}: torch.cuda.synchronize begin")
    torch_module.cuda.synchronize()
    phase(f"{label}: torch.cuda.synchronize done")


def install_call_trace() -> None:
    from hip import hip, hipsparse

    def wrap(module, symbol: str) -> None:
        if not hasattr(module, symbol):
            return
        original = getattr(module, symbol)
        if getattr(original, "_flagsparse_trace_wrapped", False):
            return

        def traced(*args, **kwargs):
            phase(f"{symbol} begin")
            try:
                result = original(*args, **kwargs)
            except Exception as exc:
                phase(f"{symbol} raised {exc.__class__.__name__}: {exc}")
                raise
            phase(f"{symbol} done -> {result}")
            return result

        traced._flagsparse_trace_wrapped = True
        setattr(module, symbol, traced)

    for symbol in (
        "hipMalloc",
        "hipFree",
        "hipEventCreate",
        "hipEventRecord",
        "hipEventSynchronize",
        "hipEventElapsedTime",
        "hipEventDestroy",
    ):
        wrap(hip, symbol)
    for symbol in (
        "hipsparseCreate",
        "hipsparseDestroy",
        "hipsparseCreateCsr",
        "hipsparseCreateCoo",
        "hipsparseCreateDnVec",
        "hipsparseDestroyDnVec",
        "hipsparseCreateDnMat",
        "hipsparseDestroyDnMat",
        "hipsparseCreateSpVec",
        "hipsparseDestroySpVec",
        "hipsparseDestroySpMat",
        "hipsparseSpMV_bufferSize",
        "hipsparseSpMV",
        "hipsparseSpMM_bufferSize",
        "hipsparseSpMM_preprocess",
        "hipsparseSpMM",
        "hipsparseGather",
        "hipsparseScatter",
    ):
        wrap(hipsparse, symbol)


def print_environment() -> None:
    phase("import torch")
    import torch

    phase("import hip-python modules")
    from flagsparse.sparse_operations import _common

    if not _common._is_hipsparse_available():
        raise SystemExit(
            "hip-python/hipSPARSE is unavailable, so there is nothing for this DCU probe "
            f"to exercise: {_common._hipsparse_unavailable_reason()}"
        )
    from hip import hip, hipsparse

    print(f"torch={getattr(torch, '__version__', '<unknown>')}", flush=True)
    print(f"torch.version.hip={getattr(torch.version, 'hip', None)}", flush=True)
    print(f"torch.version.cuda={getattr(torch.version, 'cuda', None)}", flush=True)
    print(f"hip-python={package_version('hip-python')}", flush=True)
    print(f"hip module={getattr(hip, '__file__', '<builtin>')}", flush=True)
    print(f"hipsparse module={getattr(hipsparse, '__file__', '<builtin>')}", flush=True)
    print(f"hipSPARSE available={_common._is_hipsparse_available()}", flush=True)
    print(f"hipSPARSE unavailable reason={_common._hipsparse_unavailable_reason()}", flush=True)
    for symbol in (
        "hipsparseCreate",
        "hipsparseCreateCsr",
        "hipsparseCreateCoo",
        "hipsparseCreateDnVec",
        "hipsparseCreateDnMat",
        "hipsparseCreateSpVec",
        "hipsparseSpMV_bufferSize",
        "hipsparseSpMV",
        "hipsparseSpMM_bufferSize",
        "hipsparseSpMM_preprocess",
        "hipsparseSpMM",
        "hipsparseGather",
        "hipsparseScatter",
    ):
        print(f"symbol {symbol}={hasattr(hipsparse, symbol)}", flush=True)
    for fmt in ("csr", "coo"):
        try:
            print(f"SpMM alg {fmt}={_common._hipsparse_spmm_algorithm(fmt)}", flush=True)
        except Exception as exc:
            print(f"SpMM alg {fmt}=ERROR {exc.__class__.__name__}: {exc}", flush=True)


def run_timing_only() -> None:
    import torch
    from hip import hip
    from flagsparse.sparse_operations import _common

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device is not available")
    if not _common._hip_runtime_event_available():
        raise RuntimeError("HIP runtime event API is unavailable")

    start_evt = None
    stop_evt = None
    try:
        phase("hipEventCreate(start) begin")
        start_evt = _common._hip_check_result(
            hip.hipEventCreate(), "hipEventCreate(start)"
        )
        phase(f"hipEventCreate(start) done -> {start_evt}")
        phase("hipEventCreate(stop) begin")
        stop_evt = _common._hip_check_result(
            hip.hipEventCreate(), "hipEventCreate(stop)"
        )
        phase(f"hipEventCreate(stop) done -> {stop_evt}")

        sync(torch, "timing-only pre-record")
        phase("hipEventRecord(start, 0) begin")
        _common._hip_check_result(
            hip.hipEventRecord(start_evt, 0), "hipEventRecord(start)"
        )
        phase("hipEventRecord(start, 0) done")
        scratch = torch.empty(1, device="cuda")
        scratch.add_(1.0)
        phase("hipEventRecord(stop, 0) begin")
        _common._hip_check_result(
            hip.hipEventRecord(stop_evt, 0), "hipEventRecord(stop)"
        )
        phase("hipEventRecord(stop, 0) done")
        phase("hipEventSynchronize(stop) begin")
        _common._hip_check_result(
            hip.hipEventSynchronize(stop_evt), "hipEventSynchronize(stop)"
        )
        phase("hipEventSynchronize(stop) done")
        phase("hipEventElapsedTime raw begin")
        raw = hip.hipEventElapsedTime(start_evt, stop_evt)
        phase(f"hipEventElapsedTime raw done -> {raw}")
        parsed_ms = _common._hip_event_elapsed_ms(start_evt, stop_evt)
        print(f"hip event elapsed ms={parsed_ms}", flush=True)
    finally:
        phase("hip event destroy begin")
        _common._destroy_hip_event(stop_evt)
        _common._destroy_hip_event(start_evt)
        phase("hip event destroy done")


def run_spmv_csr() -> None:
    import torch
    from flagsparse.sparse_operations import _common

    device = torch.device("cuda")
    data = torch.tensor([1.0, 2.0], device=device)
    indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    indptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    x = torch.tensor([3.0, 4.0], device=device)
    state = None
    try:
        phase("spmv_csr prepare begin")
        state = _common._prepare_spmv_csr_ref_hipsparse(data, indices, indptr, x, (2, 2))
        phase(f"spmv_csr prepare done buffer_size={state.get('buffer_size')}")
        phase("spmv_csr run begin")
        y = _common._run_spmv_csr_ref_hipsparse_prepared(state)
        phase("spmv_csr run done")
        sync(torch, "spmv_csr")
        print(f"spmv_csr result={y.detach().cpu().tolist()}", flush=True)
    finally:
        if state is not None:
            phase("spmv_csr destroy begin")
            _common._destroy_spmv_csr_ref_hipsparse_prepared(state)
            phase("spmv_csr destroy done")


def run_spmv_coo() -> None:
    import torch
    from flagsparse.sparse_operations import _common

    device = torch.device("cuda")
    data = torch.tensor([1.0, 2.0], device=device)
    row = torch.tensor([0, 1], dtype=torch.int32, device=device)
    col = torch.tensor([0, 1], dtype=torch.int32, device=device)
    x = torch.tensor([3.0, 4.0], device=device)
    state = None
    try:
        phase("spmv_coo prepare begin")
        state = _common._prepare_spmv_coo_ref_hipsparse(data, row, col, x, (2, 2))
        phase(f"spmv_coo prepare done buffer_size={state.get('buffer_size')}")
        phase("spmv_coo run begin")
        y = _common._run_spmv_coo_ref_hipsparse_prepared(state)
        phase("spmv_coo run done")
        sync(torch, "spmv_coo")
        print(f"spmv_coo result={y.detach().cpu().tolist()}", flush=True)
    finally:
        if state is not None:
            phase("spmv_coo destroy begin")
            _common._destroy_spmv_coo_ref_hipsparse_prepared(state)
            phase("spmv_coo destroy done")


def run_spmm_csr() -> None:
    import torch
    import flagsparse.sparse_operations.spmm_csr as spmm_csr

    device = torch.device("cuda")
    data = torch.tensor([1.0, 2.0], device=device)
    indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    indptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    B = torch.tensor([[3.0, 5.0], [4.0, 6.0]], device=device)
    state = None
    try:
        phase("spmm_csr prepare begin")
        state = spmm_csr._prepare_spmm_csr_ref_hipsparse(data, indices, indptr, B, (2, 2))
        phase(f"spmm_csr prepare done buffer_size={state.get('buffer_size')} alg={state.get('alg')}")
        phase("spmm_csr run begin")
        C = spmm_csr._run_spmm_csr_ref_hipsparse_prepared(state)
        phase("spmm_csr run done")
        sync(torch, "spmm_csr")
        print(f"spmm_csr result={C.detach().cpu().tolist()}", flush=True)
    finally:
        if state is not None:
            phase("spmm_csr destroy begin")
            spmm_csr._destroy_spmm_csr_ref_hipsparse_prepared(state)
            phase("spmm_csr destroy done")


def run_spmm_coo() -> None:
    import torch
    import flagsparse.sparse_operations.spmm_coo as spmm_coo

    device = torch.device("cuda")
    data = torch.tensor([1.0, 2.0], device=device)
    row = torch.tensor([0, 1], dtype=torch.int32, device=device)
    col = torch.tensor([0, 1], dtype=torch.int32, device=device)
    B = torch.tensor([[3.0, 5.0], [4.0, 6.0]], device=device)
    state = None
    try:
        phase("spmm_coo prepare begin")
        state = spmm_coo._prepare_spmm_coo_ref_hipsparse(data, row, col, B, (2, 2))
        phase(f"spmm_coo prepare done buffer_size={state.get('buffer_size')} alg={state.get('alg')}")
        phase("spmm_coo run begin")
        C = spmm_coo._run_spmm_coo_ref_hipsparse_prepared(state)
        phase("spmm_coo run done")
        sync(torch, "spmm_coo")
        print(f"spmm_coo result={C.detach().cpu().tolist()}", flush=True)
    finally:
        if state is not None:
            phase("spmm_coo destroy begin")
            spmm_coo._destroy_spmm_coo_ref_hipsparse_prepared(state)
            phase("spmm_coo destroy done")


def run_gather() -> None:
    import torch
    import flagsparse.sparse_operations.gather_scatter as gather_scatter

    device = torch.device("cuda")
    dense = torch.tensor([10.0, 20.0, 30.0], device=device)
    indices = torch.tensor([2, 0], dtype=torch.int32, device=device)
    state = None
    try:
        phase("gather prepare begin")
        state = gather_scatter._prepare_hipsparse_gather(dense, indices)
        phase("gather prepare done")
        phase("gather run begin")
        values = gather_scatter._run_hipsparse_gather_prepared(state)
        phase("gather run done")
        sync(torch, "gather")
        print(f"gather result={values.detach().cpu().tolist()}", flush=True)
    finally:
        if state is not None:
            phase("gather destroy begin")
            gather_scatter._destroy_hipsparse_gather_prepared(state)
            phase("gather destroy done")


def run_scatter() -> None:
    import torch
    import flagsparse.sparse_operations.gather_scatter as gather_scatter

    device = torch.device("cuda")
    values = torch.tensor([30.0, 10.0], device=device)
    indices = torch.tensor([2, 0], dtype=torch.int32, device=device)
    state = None
    try:
        phase("scatter prepare begin")
        state = gather_scatter._prepare_hipsparse_scatter(values, indices, 3)
        phase("scatter prepare done")
        phase("scatter run begin")
        dense = gather_scatter._run_hipsparse_scatter_prepared(state)
        phase("scatter run done")
        sync(torch, "scatter")
        print(f"scatter result={dense.detach().cpu().tolist()}", flush=True)
    finally:
        if state is not None:
            phase("scatter destroy begin")
            gather_scatter._destroy_hipsparse_scatter_prepared(state)
            phase("scatter destroy done")


OPS = {
    "spmv-csr": run_spmv_csr,
    "spmv-coo": run_spmv_coo,
    "spmm-csr": run_spmm_csr,
    "spmm-coo": run_spmm_coo,
    "gather": run_gather,
    "scatter": run_scatter,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--op",
        choices=tuple(OPS) + ("all", "env"),
        default="env",
        help="Operation to probe. Use env first, then one operation at a time.",
    )
    parser.add_argument(
        "--timing-only",
        action="store_true",
        help="Probe only the HIP runtime event timing chain.",
    )
    args = parser.parse_args()

    print_environment()
    if args.timing_only:
        install_call_trace()
        run_timing_only()
        return
    if args.op == "env":
        return
    install_call_trace()
    if args.op == "all":
        for name, fn in OPS.items():
            phase(f"{name} begin")
            fn()
            phase(f"{name} done")
        return
    OPS[args.op]()


if __name__ == "__main__":
    main()
