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

"""SpMV bottleneck probes for step 0 of docs/debug/ILUVATAR_SELL_COO_PLAN.md.

Written for Iluvatar BI-V150; runs on any FlagSparse backend. For every matrix and
dtype it times, interleaved in one process (median of ``--rounds``), with the
device-event timing the BI-V delivery uses:

  P1 stream     contiguous read + sum (once per run): the rate Triton reaches here
  P2 sell_api   flagsparse_spmv_sell(validate=False) -- the delivered code path
     sell       today's real SELL kernel with this backend's tile, re-launched bare
  P3 no_gather  P2 with x[col] replaced by x[min(row, n_cols - 1)], an address
                fixed per lane: no random x reads (wrong product; only time matters)
  P4 no_dep     P2 with the value load masked by slot, not by col >= 0
  P5 tile       P2 over (GROUP, BLOCK_R, num_warps); programs per SM are printed
  P6 sigma      prepare_spmv_sell + flagsparse_spmv_sell(prepared=...) over sigma
                and (c, group, num_warps), plus paired loads for complex
  P7 rcm        square matrices reordered by reverse Cuthill-McKee in prepare:
                prepare_spmv_sell(reorder="rcm") with the x permutation inside the timed
                call, plus today's SELL call on the reordered matrix and the bare x
                permutation (x_perm) for the breakdown

P3-P5 are real-valued (complex runs P2 sell_api and P6). Rates: ``useful`` counts
nnz * (value + int32 index) + rows * out + cols * x bytes; ``stored`` counts every
stored slot instead of nnz. The closing hints follow section 4 of the plan.

  python3 tools/biv_spmv_probe.py tests/data --matrices NACA0015,auto,ASIC_680ks \\
      --dtypes float32,float16,int8,complex64 --csv results_biv_probe.csv
"""

import argparse
import csv
import glob
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "tests"):
    if extra.is_dir():
        sys.path.insert(0, str(extra))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from benchmark_utils import accelerator_device  # noqa: E402
from mtx_fast import read_scipy_csr  # noqa: E402

import flagsparse as fs  # noqa: E402
from flagsparse.sparse_operations import _common  # noqa: E402

_ACCEL = _common._ACCEL
_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "int8": torch.int8,
    "int32": torch.int32,
}
_DEFAULT_OUT = {torch.int8: torch.int32}
_ACC = {torch.float16: tl.float32, torch.bfloat16: tl.float32, torch.float32: tl.float32,
        torch.float64: tl.float64, torch.int8: tl.int32}
_TOL = {torch.float16: 2e-2, torch.bfloat16: 2e-2, torch.float32: 1e-4, torch.float64: 1e-10,
        torch.complex64: 1e-4}
FIELDS = ["backend", "device", "matrix", "n_rows", "n_cols", "nnz", "dtype", "out_dtype",
          "probe", "config", "ms", "useful_gbps", "stored_gbps", "stored_slots", "status",
          "max_error", "note"]


@triton.jit
def _stream_kernel(src, partial, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(src + offs, mask=offs < n, other=0.0)
    tl.store(partial + pid, tl.sum(v, axis=0))


@triton.jit
def _probe_sell_kernel(
    y_ptr, v_ptr, c_ptr, off_ptr, x_ptr, n_rows, n_cols,
    SLICE: tl.constexpr, BLOCK_R: tl.constexpr, ACC: tl.constexpr, GROUP: tl.constexpr,
    GATHER: tl.constexpr, VDEP: tl.constexpr,
):
    # GATHER=1, VDEP=1 is spmv_sell._spmv_sell_real_kernel line for line.
    s = tl.program_id(0) * GROUP + tl.arange(0, GROUP)[:, None]
    r = tl.arange(0, BLOCK_R)[None, :]
    smask = s * SLICE < n_rows
    rmask = (r < SLICE) & smask
    start = tl.load(off_ptr + s, mask=smask, other=0)
    width = (tl.load(off_ptr + s + 1, mask=smask, other=0) - start) // SLICE
    row = s * SLICE + r
    fixed = tl.minimum(row, n_cols - 1)  # hoisted: no per-slot integer work for P3
    acc = tl.zeros((GROUP, BLOCK_R), dtype=ACC)
    for j in range(0, tl.max(tl.max(width, 0), 0)):
        pos = start + j * SLICE + r
        inside = rmask & (j < width)
        col = tl.load(c_ptr + pos, mask=inside, other=-1)
        valid = col >= 0
        if VDEP:
            v = tl.load(v_ptr + pos, mask=valid, other=0).to(ACC)
        else:
            v = tl.load(v_ptr + pos, mask=inside, other=0).to(ACC)
        if GATHER:
            xv = tl.load(x_ptr + col, mask=valid, other=0).to(ACC)
        else:
            # Unmasked: the address is always in range and v is already 0 on padding.
            # A masked load of a per-lane fixed address ran 3x slower than the real
            # gather on ASIC_680ks (5090), which would invert this probe.
            xv = tl.load(x_ptr + fixed).to(ACC)
        acc += v * xv
    tl.store(y_ptr + row, acc.to(y_ptr.dtype.element_ty), mask=rmask & (row < n_rows))


def _library_tile(n_slices, slice_size, dtype):
    """(GROUP, BLOCK_R, num_warps) that flagsparse_spmv_sell's generic real path uses."""
    block_r = triton.next_power_of_2(slice_size)
    group = max(1, min(8, (256 if _common._is_rocm_runtime() else 128) // block_r))
    warps = 4
    if _common._is_rocm_runtime():
        group, warps = (4, 1) if dtype == torch.float32 else (8, 2)
    elif _common._is_maca_runtime() and dtype in (torch.float16, torch.float32, torch.int8):
        group, warps = 1, 1
    if _common._is_iluvatar_runtime():
        group, warps = (1 if n_slices <= 32768 else 2), 1
        if slice_size <= 32:
            block_r = 64
    return group, block_r, warps


def _time(op, warmup, iters):
    for _ in range(warmup):
        op()
    _ACCEL.synchronize()
    start = _ACCEL.Event(enable_timing=True)
    stop = _ACCEL.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        op()
    stop.record()
    _ACCEL.synchronize()
    return start.elapsed_time(stop) / iters


def _random(n, dtype, rng):
    if dtype == torch.int8:
        return torch.from_numpy(rng.integers(-3, 4, n).astype(np.int8))
    if dtype.is_complex:
        return torch.from_numpy((rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64))
    scale = 0.5 if dtype in (torch.float16, torch.bfloat16) else 1.0
    return torch.from_numpy(scale * rng.standard_normal(n)).to(dtype)


def _wide(t):
    return t.detach().cpu().to(torch.complex128 if t.is_complex() else torch.float64).numpy()


def _error(got, ref):
    return float(np.abs(_wide(got) - ref).max() / max(1.0, float(np.abs(ref).max())))


def _passes(out_dtype, err):
    if not (out_dtype.is_floating_point or out_dtype.is_complex):
        return err == 0.0
    return err <= _TOL.get(out_dtype, 1e-4)


def _parse_dtypes(raw):
    pairs = []
    for token in (t.strip() for t in raw.split(",") if t.strip()):
        value, _, result = token.partition(":")
        vdt = _DTYPES[value]
        pairs.append((vdt, _DTYPES[result] if result else _DEFAULT_OUT.get(vdt, vdt)))
    return pairs


def _parse_configs(raw):
    return [tuple(int(v) for v in t.lower().split("x")) for t in raw.split(",") if t.strip()]


def _matrix_paths(inputs, names):
    paths = []
    for item in inputs or [str(ROOT / "tests" / "data")]:
        paths += sorted(glob.glob(os.path.join(item, "*.mtx"))) if os.path.isdir(item) else [item]
    if names:
        wanted = {n.strip().removesuffix(".mtx") for n in names.split(",") if n.strip()}
        paths = [p for p in paths if Path(p).stem in wanted]
    return paths


def _sm_count(dev):
    try:
        return int(_ACCEL.get_device_properties(dev).multi_processor_count)
    except Exception:  # not every backend exposes it
        return None


class Probe:
    def __init__(self, probe, config, op, useful, stored, slots, check=None, note=""):
        self.probe, self.config, self.op = probe, config, op
        self.useful, self.stored, self.slots = useful, stored, slots
        self.check, self.note, self.times = check, note, []


def _stream_probes(dev, mb):
    n = (mb << 20) // 4
    src = torch.ones(n, dtype=torch.float32, device=dev)
    probes = []
    for block in (512, 2048, 8192):
        partial = torch.empty(triton.cdiv(n, block), dtype=torch.float32, device=dev)
        for warps in (1, 2, 4, 8):
            def op(block=block, warps=warps, partial=partial):
                _stream_kernel[(triton.cdiv(n, block),)](src, partial, n, BLOCK=block, num_warps=warps)
            probes.append(Probe("P1 stream", f"BLOCK={block} warps={warps}", op, n * 4, n * 4, n))
    return probes


def _matrix_probes(path, csr, vdt, odt, args, dev, sm):
    rng = np.random.default_rng(args.seed)
    m, n = csr.shape
    nnz = int(csr.nnz)
    vals = _random(nnz, vdt, rng)
    x_cpu = _random(n, vdt, rng)
    ref = sp.csr_matrix((_wide(vals), csr.indices, csr.indptr), shape=csr.shape) @ _wide(x_cpu)
    data, x = vals.to(dev), x_cpu.to(dev)
    cols = torch.from_numpy(csr.indices.astype(np.int32)).to(dev)
    ptr = torch.from_numpy(csr.indptr.astype(np.int32)).to(dev)
    values, sell_cols, offsets = fs.csr_to_sell(data, cols, ptr, m, args.slice_size)
    n_slices = (m + args.slice_size - 1) // args.slice_size
    y = torch.empty(m, dtype=odt, device=dev)
    vb, ob = torch.empty((), dtype=vdt).element_size(), torch.empty((), dtype=odt).element_size()
    useful = nnz * (vb + 4) + m * ob + n * vb

    def stored(slots):
        return slots * (vb + 4) + m * ob + n * vb

    def check():
        return _error(y, ref)

    out_arg = None if odt == _DEFAULT_OUT.get(vdt, vdt) else odt
    probes = [Probe(
        "P2 sell_api", "flagsparse_spmv_sell(validate=False)",
        lambda: fs.flagsparse_spmv_sell(values, sell_cols, offsets, x, (m, n), slice_size=args.slice_size,
                                        out=y, out_dtype=out_arg, validate=False),
        useful, stored(values.numel()), values.numel(), check,
    )]
    real = not vdt.is_complex
    if real and "P2" not in args.skip:
        group, block_r, warps = _library_tile(n_slices, args.slice_size, vdt)

        def bare(group, block_r, warps, gather=1, vdep=1):
            def op():
                _probe_sell_kernel[(triton.cdiv(n_slices, group),)](
                    y, values, sell_cols, offsets, x, m, n, SLICE=args.slice_size, BLOCK_R=block_r,
                    ACC=_ACC[vdt] if not (vdt == torch.int8 and odt == torch.float32) else tl.float32,
                    GROUP=group, GATHER=gather, VDEP=vdep, num_warps=warps,
                )
            return op

        base = f"GROUP={group} BLOCK_R={block_r} warps={warps}"
        probes.append(Probe("P2 sell", base, bare(group, block_r, warps), useful,
                            stored(values.numel()), values.numel(), check))
        if "P3" not in args.skip:
            probes.append(Probe("P3 no_gather", base, bare(group, block_r, warps, gather=0), useful,
                                stored(values.numel()), values.numel(), None,
                                "x[min(row, n_cols-1)]; wrong on purpose"))
        if "P4" not in args.skip:
            probes.append(Probe("P4 no_dep", base, bare(group, block_r, warps, vdep=0), useful,
                                stored(values.numel()), values.numel(), check))
        if "P5" not in args.skip:
            for g, b, w in _parse_configs(args.tiles):
                if b < args.slice_size:
                    continue
                programs = triton.cdiv(n_slices, g)
                note = f"{programs / sm:.1f} programs per SM" if sm else ""
                probes.append(Probe("P5 tile", f"GROUP={g} BLOCK_R={b} warps={w}", bare(g, b, w), useful,
                                    stored(values.numel()), values.numel(), check, note))
    if "P6" not in args.skip:
        for sigma in [s.strip() for s in args.sigmas.split(",") if s.strip()]:
            for c, g, w in _parse_configs(args.configs):
                for paired in ((False, True) if vdt.is_complex else (False,)):
                    plan = fs.prepare_spmv_sell(
                        values, sell_cols, offsets, (m, n), slice_size=args.slice_size, out_dtype=odt,
                        route="sigma", sigma=sigma, c=c, group=g, num_warps=w, paired=paired,
                    )
                    config = f"sigma={plan.sigma}{'(auto)' if sigma == 'auto' else ''} c={c} group={g} warps={w}"
                    config += " paired" if paired else ""
                    probes.append(Probe(
                        "P6 sigma", config,
                        lambda plan=plan: fs.flagsparse_spmv_sell(prepared=plan, x=x, out=y),
                        useful, stored(plan.n_values), plan.n_values, check,
                        f"stored/nnz {plan.n_values / max(1, nnz):.2f}",
                    ))
    if "P7" not in args.skip and m == n and m > 1:
        probes += _rcm_probes(csr, data, x, y, ref, odt, out_arg, stored, check, args, dev,
                              (values, sell_cols, offsets))
    return probes


def _rcm_probes(csr, data, x, y, ref, odt, out_arg, stored, check, args, dev, sell_arrays):
    """Reverse Cuthill-McKee on the symmetrised pattern: x locality bought in prepare."""
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    m = csr.shape[0]
    values_in, cols_in, offsets_in = sell_arrays
    pattern = sp.csr_matrix((np.ones(csr.nnz, dtype=np.int8), csr.indices, csr.indptr), shape=csr.shape)
    order = reverse_cuthill_mckee((pattern + pattern.T).tocsr(), symmetric_mode=True)
    perm = torch.from_numpy(order.astype(np.int64))
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(m)
    rows = torch.repeat_interleave(torch.arange(m), torch.from_numpy(np.diff(csr.indptr).astype(np.int64)))
    new_rows = inv[rows]
    new_cols = inv[torch.from_numpy(csr.indices.astype(np.int64))]
    key = torch.argsort(new_rows * m + new_cols)
    new_rows, new_cols = new_rows[key], new_cols[key]
    ptr = torch.zeros(m + 1, dtype=torch.int64)
    ptr[1:] = torch.cumsum(torch.bincount(new_rows, minlength=m), 0)
    pdata = data.index_select(0, key.to(dev))
    pv, pc, po = fs.csr_to_sell(pdata, new_cols.to(torch.int32).to(dev), ptr.to(torch.int32).to(dev),
                                m, args.slice_size)
    perm_d = perm.to(dev)
    xp = x.index_select(0, perm_d)
    spread = float(np.median(np.abs(new_cols.numpy() - new_rows.numpy())))
    ref_perm = ref[order]

    def check_perm():
        return _error(y, ref_perm)

    nnz = int(csr.nnz)
    probes = [Probe(
        "P7 rcm", "flagsparse_spmv_sell on the RCM-reordered matrix",
        lambda: fs.flagsparse_spmv_sell(pv, pc, po, xp, csr.shape, slice_size=args.slice_size,
                                        out=y, out_dtype=out_arg, validate=False),
        stored(nnz), stored(pv.numel()), pv.numel(), check_perm,
        f"median |col-row| after RCM {spread:.0f}",
    )]
    plan = fs.prepare_spmv_sell(values_in, cols_in, offsets_in, csr.shape, slice_size=args.slice_size,
                                out_dtype=odt, route="sigma", reorder="rcm")
    probes.append(Probe(
        "P7 rcm", f"prepare_spmv_sell(reorder='rcm') sigma={plan.sigma}(auto) c={plan.c} "
                  f"group={plan.group} warps={plan.num_warps}",
        lambda: fs.flagsparse_spmv_sell(prepared=plan, x=x, out=y),
        stored(nnz), stored(plan.n_values), plan.n_values, check,
        "x permutation included in the timed call",
    ))
    xbuf = torch.empty_like(x)
    vb = torch.empty((), dtype=x.dtype).element_size()
    probes.append(Probe(
        "P7 x_perm", "x.index_select(perm) per call", lambda: torch.index_select(x, 0, perm_d, out=xbuf),
        2 * m * vb, 2 * m * vb, m, None, "add to a P7 rcm row for the full per-call cost",
    ))
    return probes


def _run_probes(probes, args):
    for _ in range(args.rounds):
        for p in probes:
            p.times.append(_time(p.op, args.warmup, args.iters))
    rows = []
    for p in probes:
        ms = statistics.median(p.times)
        row = {"probe": p.probe, "config": p.config, "ms": ms,
               "useful_gbps": p.useful / ms / 1e6, "stored_gbps": p.stored / ms / 1e6,
               "stored_slots": p.slots, "note": p.note, "status": "n/a"}
        if p.check is not None:
            p.op()
            _ACCEL.synchronize()
            err = p.check()
            row["max_error"] = err
            row["status"] = "PASS" if _passes(args._out_dtype, err) else "FAIL"
        rows.append(row)
    return rows


def _best(rows, probe):
    hits = [r for r in rows if r["probe"] == probe]
    return min(hits, key=lambda r: r["ms"]) if hits else None


def _hints(rows, stream_peak):
    api, base = _best(rows, "P2 sell_api"), _best(rows, "P2 sell")
    out = []
    if stream_peak:
        out.append(f"P1: streaming peaks at {stream_peak:.0f} GB/s"
                   + (" -- Triton itself is the ceiling; favour fewer bytes (sigma, compression)"
                      if stream_peak < 500 else ""))
    if base:
        for probe, threshold, meaning in (
            ("P3 no_gather", 1.3, "random x reads dominate: try a smaller sigma / x locality"),
            ("P4 no_dep", 1.1, "the col -> value dependency is exposed (P6 already masks by row length)"),
            ("P5 tile", 1.15, "tile / occupancy matters: adopt the best P5 tile"),
        ):
            hit = _best(rows, probe)
            if hit:
                gain = base["ms"] / hit["ms"]
                out.append(f"{probe}: {gain:.2f}x vs P2 ({hit['config']})"
                           + (f" -- {meaning}" if gain >= threshold else ""))
    best6 = _best([r for r in rows if r["status"] != "FAIL"], "P6 sigma")
    if api and best6:
        out.append(f"P6: best {best6['config']} at {best6['ms'] * 1e3:.1f} us = "
                   f"{api['ms'] / best6['ms']:.2f}x of the delivered path ({api['ms'] * 1e3:.1f} us)")
    rcm = [r for r in rows if r["probe"] == "P7 rcm" and r["config"].startswith("prepare_spmv_sell")]
    if rcm and rcm[0]["status"] != "FAIL" and (best6 or api):
        # The question is whether to reorder on top of the prepared path, so compare
        # with the best unreordered P6 plan when there is one.
        ref, what = (best6, "the best P6 plan") if best6 else (api, "the delivered path")
        gain = ref["ms"] / rcm[0]["ms"]
        out.append(f"P7: prepared RCM (x permutation included) {rcm[0]['ms'] * 1e3:.1f} us = "
                   f"{gain:.2f}x of {what} ({ref['ms'] * 1e3:.1f} us)"
                   + (" -- reordering in prepare pays off: try FLAGSPARSE_SELL_PREP_REORDER=auto"
                      if gain >= 1.15 else ""))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mtx", nargs="*", help=".mtx files or directories (default tests/data)")
    parser.add_argument("--matrices", default="NACA0015,auto,ASIC_680ks",
                        help="comma-separated matrix names to keep ('' = all)")
    parser.add_argument("--dtypes", default="float32,float16,int8,complex64", help="value[:out],...")
    parser.add_argument("--slice-size", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--stream-mb", type=int, default=512)
    parser.add_argument("--sigmas", default="32,256,1024,4096,auto")
    parser.add_argument("--configs", default="64x1x1,32x2x1,32x1x1,32x4x1,32x4x4",
                        help="P6 c x group x num_warps list")
    parser.add_argument("--tiles", default="1x32x1,1x64x1,2x32x1,2x64x1,4x32x1,4x32x2,4x32x4,8x32x4",
                        help="P5 GROUP x BLOCK_R x num_warps list")
    parser.add_argument("--skip", default="", help="probes to skip, e.g. P1,P5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args(argv)
    args.skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}

    dev = accelerator_device()
    sm = _sm_count(dev)
    backend = _common._backend_name()
    try:
        device_name = _ACCEL.get_device_name(dev)
    except Exception:
        device_name = str(dev)
    print(f"backend={backend} device={device_name} SMs={sm} triton={triton.__version__}", flush=True)
    meta = {"backend": backend, "device": device_name}
    rows = []
    stream_peak = None
    if "P1" not in args.skip:
        args._out_dtype = torch.float32
        for row in _run_probes(_stream_probes(dev, args.stream_mb), args):
            rows.append({**meta, **row, "matrix": "-", "dtype": "float32", "status": "n/a"})
        stream_peak = max(r["useful_gbps"] for r in rows)
        best = max(rows, key=lambda r: r["useful_gbps"])
        print(f"P1 stream: peak {stream_peak:.0f} GB/s ({best['config']})", flush=True)

    paths = _matrix_paths(args.mtx, args.matrices)
    if not paths:
        parser.error("no matching .mtx files")
    for path in paths:
        csr = read_scipy_csr(path)
        for vdt, odt in _parse_dtypes(args.dtypes):
            args._out_dtype = odt
            label = f"{Path(path).stem} {str(vdt)[6:]}->{str(odt)[6:]}"
            try:
                results = _run_probes(_matrix_probes(path, csr, vdt, odt, args, dev, sm), args)
            except Exception as exc:  # keep going: one failing dtype must not hide the rest
                print(f"== {label}: ERROR {type(exc).__name__}: {exc}", flush=True)
                rows.append({**meta, "matrix": Path(path).name, "dtype": str(vdt)[6:], "status": "ERROR",
                             "note": f"{type(exc).__name__}: {exc}"})
                continue
            print(f"== {label} (nnz={csr.nnz})", flush=True)
            for r in results:
                print(f"   {r['probe']:12s} {r['ms'] * 1e3:9.1f} us  useful {r['useful_gbps']:6.0f} GB/s"
                      f"  stored {r['stored_gbps']:6.0f} GB/s  {r['status']:4s} {r['config']}"
                      + (f"  [{r['note']}]" if r['note'] else ""), flush=True)
            for line in _hints(results, stream_peak):
                print(f"   -> {line}", flush=True)
            for r in results:
                rows.append({**meta, **r, "matrix": Path(path).name, "n_rows": csr.shape[0],
                             "n_cols": csr.shape[1], "nnz": csr.nnz, "dtype": str(vdt)[6:],
                             "out_dtype": str(odt)[6:]})
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in FIELDS})
    print(f"wrote {len(rows)} rows to {args.csv}", flush=True)
    return 1 if any(r.get("status") in ("FAIL", "ERROR") for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
