#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Judge rows with no vendor baseline against an H800 reference, scaled by peaks.

When a backend has no native sparse library to compare with, a row carries no
speedup. The acceptance scheme in ``modified/基线缺失方案.md`` then asks that the
vendor's utilisation of the NVIDIA bottleneck unit be at least ``--ratio``
(default 0.8) of NVIDIA's. With the same workload on both sides that is a time
bound on the vendor's FlagSparse time::

    T' <= T * (P_ref / P_vendor) / ratio

``T`` is the VENDOR LIBRARY's time (cuSPARSE: the ``cusparse_ms`` / ``vendor_ms`` /
``*sparse_ms`` column) for the same case on the reference card (H800 by default) --
the baseline the missing one is standing in for. ``P`` is the peak of the bottleneck
unit: memory bandwidth, CUDA-core FLOP/s for the dtype, or Tensor-core FLOP/s for the
dtype. Example: cuSPARSE takes 10 ms on an H800 at 3050 GB/s (measured), the vendor
card has 1525 GB/s -> its library would take 20 ms (``T' scaled``), bound 20 / 0.8 =
25 ms. The speedup is that scaled library time divided by FlagSparse's own time on
the vendor card; 1.0 means FlagSparse is exactly as fast as the scaled library.
``--ref-column`` names a different reference column (e.g. ``triton_ms`` to scale
FlagSparse's own H800 time instead).

Inputs
    * the reference run: a results directory (``<dir>/<op>/performance.csv``) or
      one ``performance.csv``. Only rows whose ``status`` is PASS are used -- the
      reference must itself be a verified result -- and whose vendor-library result
      matched FlagSparse (no FAIL in ``cu_status`` / ``vendor_status``);
    * the reference peaks: built in for ``--reference h800-sxm`` (default) and
      ``h800-pcie``, or the ``reference`` side of a ``--peaks`` JSON;
    * the vendor peaks: detected from the vendor run's ``summary.json`` when it has
      one (the runner records the backend and device name), or ``--vendor-card``
      (built-in measured bandwidth for C550,
      S5000, BW1000, BI-V150, Ascend 910B/910C), ``--vendor-bw-gbs`` (either is enough for
      the default memory-bound unit), or the ``vendor`` side of ``--peaks``;
    * optionally the vendor run (``--vendor``, same layout as the reference). Each
      vendor row with no usable vendor-library baseline is judged PASS/FAIL; a row
      that has one is HAS_BASELINE and left to its measured speedup (``--all-rows``
      judges it anyway); a vendor row the reference run lacks is NO_REFERENCE.

The FlagSparse time column is picked per CSV from the runner's
``PERFORMANCE_SPEEDUP_SCHEMAS``: the first vendor-library schema whose FlagSparse
column the CSV carries (``triton_ms``, ``ms``, ``opt_ms``, ...). The two sides may
differ -- CUDA spmv_csr writes ``triton_ms``, ROCm writes ``ms``.

The bottleneck unit of a row is, in order: its ``bottleneck`` column
(``mem``/``cuda``/``tensor``), the largest of its ``u_cuda``/``u_tensor``/``u_mem``
columns (Nsight Compute utilisation, %), else ``--default-resource`` (``mem``).
The last case is an assumption -- sparse kernels are usually memory-bound, but
nothing measured it -- and is labelled ``assumed`` in the output.

Exit status is 0 when every row has a bound (and, with ``--vendor``, passes or
has its own baseline), 1 otherwise.

The reference may be a results directory, one performance.csv, or the bundled
``conf/h800_reference.json`` (see ``tools/h800_reference.py``); when it is left out the
bundled file is used.

Usage:
    python3 tools/baseline_bound.py --vendor vendor_results/ --vendor-card dcu-bw1000
    python3 tools/baseline_bound.py h800_results/ --vendor-bw-gbs 1600
    python3 tools/baseline_bound.py h800_results/ --vendor-card dcu-bw1000 \
        --vendor vendor_results/ [--markdown] [--csv out.csv]
    python3 tools/baseline_bound.py --print-template > peaks.json
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:  # `python3 tools/baseline_bound.py` puts tools/ first
    sys.path.insert(0, str(_ROOT))
from tools import h800_reference as _h800_reference  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_flagsparse_pytest.py"
RESOURCES = ("mem", "cuda", "tensor")

# NVIDIA H800 peaks. Memory bandwidth on h800-sxm is MEASURED (3050 GB/s; the
# datasheet says 3350) so that it divides the vendors' measured figures below
# like for like -- a datasheet over a measurement would loosen every bound by
# ~10%. h800-pcie has no measurement and keeps its datasheet 2000. The compute
# figures are datasheet (dense; the datasheet's Tensor figures are quoted
# with 2:4 sparsity and are halved here). H800 is H100 with FP64 cut to ~1
# TFLOPS -- a compute-bound fp64 row therefore gets a very tight bound, since
# almost any vendor card out-runs it. The datasheet gives no non-Tensor FP16/BF16
# figure; Hopper runs them at 2x FP32 (H100 whitepaper: 133.8 vs 66.9 TFLOPS),
# so those two entries are derived, not quoted.
REFERENCE_PEAKS = {
    "h800-sxm": {
        "name": "NVIDIA H800 SXM5 80GB HBM3",
        "mem_bw_gbs": 3050.0,
        "cuda_tflops": {"fp64": 1.0, "fp32": 67.0, "fp16": 134.0, "bf16": 134.0},
        "tensor_tflops": {"fp64": 1.0, "tf32": 494.5, "fp16": 989.5, "bf16": 989.5},
    },
    "h800-pcie": {
        "name": "NVIDIA H800 PCIe 80GB HBM2e",
        "mem_bw_gbs": 2000.0,
        "cuda_tflops": {"fp64": 0.8, "fp32": 51.0, "fp16": 102.0, "bf16": 102.0},
        "tensor_tflops": {"fp64": 0.8, "tf32": 378.0, "fp16": 756.5, "bf16": 756.5},
    },
}

# Measured memory bandwidth per vendor card, GB/s (2026-09-27, same method as the
# H800 3050). --vendor-bw-gbs overrides.
VENDOR_BANDWIDTH_GBS = {
    "maca-c550": 1440.0,  # MetaX C550
    "musa-s5000": 1370.0,  # Moore Threads S5000
    "dcu-bw1000": 1530.0,  # Hygon BW1000
    "iluvatar-biv150": 1150.0,  # Iluvatar CoreX BI-V150
    # Ascend delivery tests run on a 910B. No measurement exists for it, so this is
    # the PUBLISHED 1.6 TB/s -- the one entry that is not measured; a published peak
    # usually exceeds a measured one, so the bound is somewhat tighter than the rest.
    "ascend-910b": 1600.0,
    # Measured on an Atlas 800T A3 (910C, two dies per package). Not the tested chip.
    "ascend-910c": 3070.0,
}

# Device tokens -> the VENDOR_BANDWIDTH_GBS entry, for auto-detection from a results
# directory's summary.json (its env records the backend and the device name). Checked
# against the device name first, then the backend, so a box with two cards of the same
# vendor is still told apart.
_DEVICE_TOKEN_CARDS = (
    ("bi-v150", "iluvatar-biv150"),
    ("biv150", "iluvatar-biv150"),
    ("c550", "maca-c550"),
    ("s5000", "musa-s5000"),
    ("bw1000", "dcu-bw1000"),
    ("910b", "ascend-910b"),
    ("910c", "ascend-910c"),
)
_BACKEND_CARDS = {
    "iluvatar": "iluvatar-biv150",
    "metax": "maca-c550",
    "mthreads": "musa-s5000",
    "ascend": "ascend-910b",
}
# Reference peaks are a card's; a reference run from a different card silently
# rescales every bound, so the detected name is checked against these tokens.
_REFERENCE_TOKENS = {"h800-sxm": ("h800",), "h800-pcie": ("h800",)}


# A column holding a vendor sparse library's time: cusparse_ms, vendor_ms, and the
# SpSV/SpSM spellings cuSPARSE_ms, hipSPARSE_ms, CuPy/cuSPARSE_ms. PyTorch is not
# one -- on Ascend it is the same library FlagSparse falls back to.
_BASELINE_EXACT = {"vendor_ms"}
_BASELINE_SUFFIX = "sparse_ms"
# SpSV/SpSM name their FlagSparse column outside the runner's schema table.
_EXTRA_TIME_COLUMNS = ("FlagSparse_ms",)

# Complex values are computed on the real unit of the matching precision.
_UNIT_DTYPE = {
    "float16": "fp16",
    "half": "fp16",
    "bfloat16": "bf16",
    "float32": "fp32",
    "float": "fp32",
    "float64": "fp64",
    "double": "fp64",
    "complex64": "fp32",
    "complex128": "fp64",
    "tf32": "tf32",
}
_MATRIX_KEYS = ("matrix", "path", "name", "case_id", "case")
_DTYPE_KEYS = (
    "dtype",
    "value_dtype",
    "value_dtype_req",
    "value_dtype_compute",
    "data_dtype",
)
_OP_KEYS = ("op", "opa", "transpose")
# Case axes joined into the key when BOTH sides carry the column; one side alone
# (ROCm spmv_csr has `alg`, CUDA does not) would make every key miss.
_OPTIONAL_KEYS = ("alg", "layout", "mode", "dense_cols", "k")
_PASS = {"PASS", "PASSED", "OK", "SUCCESS"}
_TEMPLATE_SIDE = {
    "name": None,
    "mem_bw_gbs": None,
    "cuda_tflops": {"fp16": None, "fp32": None, "fp64": None},
    "tensor_tflops": {"fp16": None, "bf16": None, "tf32": None},
}


def _template() -> dict:
    return {
        "_note": (
            "Fill in measured peaks. Units only need to agree between the reference "
            "and vendor sides of the same field; the ratio is what is used. Delete "
            "dtype entries you do not need. cuda_tflops on the vendor side is the "
            "vector-unit peak. Omit 'reference' to use the built-in --reference."
        ),
        "reference": json.loads(json.dumps(REFERENCE_PEAKS["h800-sxm"])),
        "vendor": json.loads(json.dumps(_TEMPLATE_SIDE)),
    }


def _schemas() -> list[tuple[str, str | None, str | None]]:
    """The runner's (speedup, baseline, FlagSparse) column table, read statically.

    Importing the runner costs seconds and pulls in torch; the literal is enough.
    """

    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                getattr(t, "id", "") == "PERFORMANCE_SPEEDUP_SCHEMAS" for t in targets
            ):
                return list(ast.literal_eval(node.value))
    raise SystemExit(f"PERFORMANCE_SPEEDUP_SCHEMAS not found in {RUNNER}")


def _is_baseline_column(name: str | None) -> bool:
    lowered = str(name or "").lower()
    if lowered == "flagsparse_ms":  # ends in "sparse_ms" but is FlagSparse itself
        return False
    return lowered in _BASELINE_EXACT or lowered.endswith(_BASELINE_SUFFIX)


def _time_column(fields: list[str]) -> str | None:
    for _, base, latency in _schemas():
        if latency and latency in fields and _is_baseline_column(base):
            return latency
    for name in _EXTRA_TIME_COLUMNS:
        if name in fields:
            return name
    return None


def _read_csv(path) -> tuple[list[str], list[dict[str, str]]]:
    if isinstance(path, _h800_reference.ReferenceTable):
        return path.read()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _collect(target: str) -> dict:
    """Map operator -> performance.csv (or bundled table) for a results directory, a
    single CSV, or the bundled ``conf/h800_reference.json``."""

    path = Path(target)
    if _h800_reference.is_reference_file(path):
        return _h800_reference.tables(path)
    if path.is_file():
        return {path.parent.name or path.stem: path}
    if path.is_dir():
        found = {p.parent.name: p for p in sorted(path.glob("*/performance.csv"))}
        if (path / "performance.csv").is_file():
            found.setdefault(path.name, path / "performance.csv")
        if found:
            return found
        raise SystemExit(f"no */performance.csv under {path}")
    raise SystemExit(f"no such file or directory: {path}")


def _run_env(target: str) -> tuple[str, str] | None:
    """(backend, device name) from a results directory's summary.json, else None.

    A single CSV or a directory the runner did not write has no env; detection is
    skipped rather than guessed.
    """
    path = Path(target)
    if _h800_reference.is_reference_file(path):
        ref = _h800_reference.reference_info(path)
        device = str(ref.get("device") or "").strip()
        return ("nvidia", device) if device else None
    if not path.is_dir():
        return None
    for name in ("summary.json", "summary_split.json"):
        candidate = path / name
        if not candidate.is_file():
            continue
        try:
            env = json.loads(candidate.read_text(encoding="utf-8")).get("env") or {}
        except (OSError, ValueError):
            return None
        gems = env.get("flag_gems") if isinstance(env.get("flag_gems"), dict) else {}
        torch_env = env.get("torch") if isinstance(env.get("torch"), dict) else {}
        backend = str(gems.get("vendor") or gems.get("device") or "").strip().lower()
        device = str(torch_env.get("device_name") or "").strip()
        if backend or device:
            return backend, device
    return None


def _detect_card(env: tuple[str, str] | None) -> str | None:
    """The VENDOR_BANDWIDTH_GBS key this run's hardware matches, if any."""
    if env is None:
        return None
    backend, device = env
    lowered = device.lower()
    for token, card in _DEVICE_TOKEN_CARDS:
        if token in lowered:
            return card
    return _BACKEND_CARDS.get(backend)


def _num(value: object) -> float | None:
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


def _lower_fields(row: dict[str, str]) -> dict[str, str]:
    return {str(k).lower(): v for k, v in row.items() if k is not None}


def _matrix_key(row: dict[str, str]) -> str:
    for key in _MATRIX_KEYS:
        value = str(row.get(key) or "").strip()
        if value:
            return Path(value).name.lower()
    return ""


def _dtype(row: dict[str, str]) -> str:
    for key in _DTYPE_KEYS:
        value = str(row.get(key) or "").strip()
        if value:
            return value.replace("torch.", "").lower()
    return "unknown"


def _op(row: dict[str, str]) -> str:
    low = _lower_fields(row)
    for key in _OP_KEYS:
        value = str(low.get(key) or "").strip().lower()
        if value:
            return value
    return ""


def _row_key(row: dict[str, str], extra: list[str]) -> tuple[str, ...]:
    parts = [
        _matrix_key(row),
        _dtype(row),
        str(row.get("index_dtype") or "").strip().lower(),
        _op(row),
    ]
    parts.extend(str(row.get(col) or "").strip().lower() for col in extra)
    return tuple(parts)


def _label(key: tuple[str, ...]) -> str:
    return "|".join(part for part in key if part)


def _index_rows(
    rows: list[dict[str, str]], extra: list[str], source: str
) -> dict[tuple[str, ...], dict[str, str]]:
    indexed: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = _row_key(row, extra)
        if key in indexed:
            raise ValueError(
                f"{source}: two rows share the key {_label(key)!r}; add a column that "
                "tells them apart with --extra-key (e.g. --extra-key alg)"
            )
        indexed[key] = row
    return indexed


def _passed(row: dict[str, str]) -> bool:
    status = str(row.get("status") or row.get("matrix_status") or "").strip().upper()
    return status in _PASS


def _baseline_ok(row: dict[str, str]) -> bool:
    """False when the row's own vendor-library check failed (its time is not a baseline)."""
    for key, value in row.items():
        if str(key).lower() in ("cu_status", "vendor_status", "cusparse_status"):
            if any(bad in str(value).upper() for bad in ("FAIL", "ERROR")):
                return False
    return True


def _reference_time(
    row: dict[str, str], ref_col: str | None
) -> tuple[float | None, str]:
    """The reference row's baseline time and, if it has none, why.

    Default: the vendor library's time on that row. ``--ref-column`` names another.
    """
    if ref_col:
        return _num(row.get(ref_col)), f"no usable reference {ref_col}"
    if not _baseline_ok(row):
        return None, "the reference's vendor-library result did not match FlagSparse"
    return _baseline_ms(row), "the reference row has no vendor-library (cuSPARSE) time"


def _baseline_ms(row: dict[str, str]) -> float | None:
    for key, value in row.items():
        if _is_baseline_column(key):
            number = _num(value)
            if number is not None and number > 0:
                return number
    return None


def _check_reference_run(args) -> None:
    """Warn when the reference run is not the card whose peaks are being used."""
    if args.peaks:
        return  # explicit peaks: the caller stated both sides
    env = _run_env(args.reference_run)
    if env is None:
        return
    backend, device = env
    tokens = _REFERENCE_TOKENS.get(args.reference, ())
    if any(token in device.lower() for token in tokens):
        return
    print(
        f"WARNING: --reference {args.reference} peaks are being applied to a reference "
        f"run recorded on {device or backend!r}. T is that card's time, so every bound "
        "is scaled by the wrong ratio. Re-run the reference on the card, or pass "
        "--peaks with its measured peaks.",
        file=sys.stderr,
    )


def _load_peaks(args) -> dict:
    peaks: dict = {}
    if args.peaks:
        try:
            peaks = json.loads(Path(args.peaks).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SystemExit(f"cannot read peaks file {args.peaks}: {exc}")
    reference = peaks.get("reference") or peaks.get("nvidia")
    if not isinstance(reference, dict):
        reference = json.loads(json.dumps(REFERENCE_PEAKS[args.reference]))
    vendor = peaks.get("vendor")
    vendor = dict(vendor) if isinstance(vendor, dict) else {}
    detected = _detect_card(_run_env(args.vendor)) if args.vendor else None
    card = args.vendor_card or detected
    if detected and args.vendor_card and args.vendor_card != detected:
        print(
            f"WARNING: --vendor-card {args.vendor_card} overrides {detected}, which the "
            "vendor run's summary.json reports. A wrong peak moves every bound.",
            file=sys.stderr,
        )
    elif detected and not args.vendor_card:
        print(
            f"note: vendor card {detected} detected from the vendor run's summary.json",
            file=sys.stderr,
        )
    if card:
        vendor["name"] = card
        vendor["mem_bw_gbs"] = VENDOR_BANDWIDTH_GBS[card]
    if args.vendor_bw_gbs is not None:
        vendor["mem_bw_gbs"] = args.vendor_bw_gbs
    if _num(vendor.get("mem_bw_gbs")) is None and not any(
        isinstance(vendor.get(f), dict) for f in ("cuda_tflops", "tensor_tflops")
    ):
        raise SystemExit("no vendor peaks: pass --vendor-bw-gbs or a --peaks JSON")
    return {"reference": reference, "vendor": vendor}


def _peak(side: dict, resource: str, dtype: str) -> float:
    unit = _UNIT_DTYPE.get(dtype, dtype)
    if resource == "mem":
        value, what = side.get("mem_bw_gbs"), "mem_bw_gbs"
    else:
        field = "cuda_tflops" if resource == "cuda" else "tensor_tflops"
        table = side.get(field)
        value = table.get(unit) if isinstance(table, dict) else None
        what = f"{field}[{unit}]"
    number = _num(value)
    if number is None or number <= 0:
        raise KeyError(what)
    return number


def _resource_for(row: dict[str, str], forced: str, default: str) -> tuple[str, str]:
    if forced != "auto":
        return forced, "forced"
    named = str(row.get("bottleneck") or "").strip().lower()
    if named in RESOURCES:
        return named, "column"
    util = {r: _num(row.get(f"u_{r}")) for r in RESOURCES}
    util = {r: u for r, u in util.items() if u is not None}
    if util:
        return max(util, key=util.get), "profile"
    return default, "assumed"


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4g}"


def _result(op: str, key: str, **fields) -> dict:
    out = {
        "op": op,
        "key": key,
        "dtype": "",
        "resource": "",
        "source": "",
        "t_ref_ms": None,
        "expected_ms": None,
        "bound_ms": None,
        "t_vendor_ms": None,
        "speedup": None,
        "margin": None,
        "verdict": "",
        "note": "",
    }
    out.update(fields)
    return out


def _pick(fields: list[str], wanted: str | None, side: str, path: Path) -> str:
    if wanted:
        if wanted not in fields:
            raise ValueError(f"{side} {path}: column {wanted!r} is not in the header")
        return wanted
    column = _time_column(fields)
    if column is None:
        times = [f for f in fields if f.lower().endswith("ms")]
        raise ValueError(
            f"{side} {path}: no FlagSparse time column found; pass --ref-column / "
            f"--vendor-column. Time-like columns here: {', '.join(times) or 'none'}"
        )
    return column


def _bound(out: dict, row: dict[str, str], ref_col: str | None, args, peaks) -> bool:
    """Fill ``out`` with the reference time and its bound; False when there is none."""

    t_ref, why = _reference_time(row, ref_col)
    out["t_ref_ms"] = t_ref
    if not _passed(row):
        out["verdict"] = "N/A"
        out["note"] = f"reference row is not PASS ({row.get('status') or 'no status'})"
        return False
    if t_ref is None or t_ref <= 0:
        out["verdict"], out["note"] = "N/A", why
        return False
    try:
        p_ref = _peak(peaks["reference"], out["resource"], out["dtype"])
        p_v = _peak(peaks["vendor"], out["resource"], out["dtype"])
    except KeyError as exc:
        out["verdict"], out["note"] = "N/A", f"peak missing: {exc.args[0]}"
        return False
    # The H800 time rescaled to the vendor card: what the vendor should take if it
    # used its bottleneck unit as well as the H800 did. `--ratio` only sets the
    # pass line around it.
    out["expected_ms"] = t_ref * (p_ref / p_v)
    out["bound_ms"] = out["expected_ms"] / args.ratio
    return True


# Columns scaled_baseline_rows() adds to a performance CSV. The runner's
# PERFORMANCE_SPEEDUP_SCHEMAS reads the last three as (speedup, baseline, latency), so
# they have to exist under exactly these names.
SCALED_COLUMNS = (
    "h800_vendor_ms",
    "h800_scaled_ms",
    "h800_scaled_fs_ms",
    "speedup_vs_h800_scaled",
    "h800_scaled_card",
)


def _usable(row: dict[str, str]) -> bool:
    """The runner's rule: no status column, or a passing one."""
    status = str(row.get("status") or row.get("matrix_status") or "").strip().upper()
    return not status or status in _PASS


def scaled_baseline_rows(
    op: str,
    rows: list[dict[str, str]],
    fields: list[str],
    reference_run: str,
    card: str,
    *,
    reference: str = "h800-sxm",
    needs_baseline=lambda row: True,
) -> tuple[list[dict[str, str]], dict]:
    """Fill a speedup for rows that have no vendor or PyTorch baseline.

    For a row ``needs_baseline`` accepts, the same case's VENDOR-LIBRARY time (cuSPARSE)
    on the reference card (H800) is rescaled by the bottleneck-unit peak ratio to the
    vendor card; THAT time stands in for the missing baseline (speedup 1.0), and the
    row's own FlagSparse time is compared with it::

        h800_scaled_ms          = T_cusparse_h800 * (P_h800 / P_vendor)
        speedup_vs_h800_scaled  = h800_scaled_ms / T_flagsparse_vendor

    Returns ``(rows, info)``: ``rows`` are copies with ``SCALED_COLUMNS`` added (blank
    where nothing could be filled), ``info`` counts what happened and why not.
    """
    info = {
        "op": op,
        "reference": str(reference_run),
        "reference_card": reference,
        "card": card,
        "rows": len(rows),
        "filled": 0,
        "eligible": 0,
        "skipped": {},
    }

    def skip(reason: str, count: int = 1) -> None:
        info["skipped"][reason] = info["skipped"].get(reason, 0) + count

    out = [dict(row) for row in rows]
    for row in out:
        for column in SCALED_COLUMNS:
            row.setdefault(column, "")
    try:
        ref_path = _collect(reference_run).get(op)
    except SystemExit as exc:
        info["error"] = str(exc)
        return out, info
    if ref_path is None:
        info["error"] = f"no {op}/performance.csv in the reference run"
        return out, info
    ref_fields, ref_rows = _read_csv(ref_path)
    own_col = _time_column(fields)
    if own_col is None:
        info["error"] = "no FlagSparse time column recognised"
        return out, info
    extra = [c for c in _OPTIONAL_KEYS if c in ref_fields and c in fields]
    ref_index: dict[tuple[str, ...], dict[str, str]] = {}
    ambiguous: set[tuple[str, ...]] = set()
    for row in ref_rows:
        key = _row_key(row, extra)
        if key in ref_index:
            ambiguous.add(key)
        ref_index[key] = row
    peaks = {
        "reference": REFERENCE_PEAKS[reference],
        "vendor": {"mem_bw_gbs": VENDOR_BANDWIDTH_GBS[card]},
    }
    for row in out:
        if not needs_baseline(row):
            continue
        info["eligible"] += 1
        t_own = _num(row.get(own_col))
        if not _usable(row) or t_own is None or t_own <= 0:
            skip("this row has no usable FlagSparse time")
            continue
        key = _row_key(row, extra)
        if key in ambiguous:
            skip("the reference has several rows for this case")
            continue
        ref = ref_index.get(key)
        if ref is None:
            skip("no such case in the reference run")
            continue
        if not _usable(ref):
            skip("the reference row did not pass")
            continue
        t_ref, why = _reference_time(ref, None)
        if t_ref is None or t_ref <= 0:
            skip(why)
            continue
        resource, _ = _resource_for(ref, "auto", "mem")
        try:
            p_ref = _peak(peaks["reference"], resource, _dtype(ref))
            p_v = _peak(peaks["vendor"], resource, _dtype(row))
        except KeyError as exc:
            skip(f"peak missing: {exc.args[0]}")
            continue
        scaled = t_ref * (p_ref / p_v)
        row["h800_vendor_ms"] = repr(t_ref)
        row["h800_scaled_ms"] = repr(scaled)
        row["h800_scaled_fs_ms"] = repr(t_own)
        row["speedup_vs_h800_scaled"] = repr(scaled / t_own)
        row["h800_scaled_card"] = card
        info["filled"] += 1
    return out, info


def _evaluate_op(op: str, ref_path: Path, vendor_path: Path | None, args, peaks):
    ref_fields, ref_rows = _read_csv(ref_path)
    ref_col = (
        _pick(ref_fields, args.ref_column, "reference", ref_path)
        if args.ref_column
        else None
    )
    vendor_fields, vendor_rows = _read_csv(vendor_path) if vendor_path else ([], [])
    # A header-only CSV (a benchmark that recorded nothing) would otherwise add no
    # rows and vanish from the table.
    if not ref_rows or (vendor_path and not vendor_rows):
        empty = ref_path if not ref_rows else vendor_path
        return [_result(op, "*", verdict="EMPTY", note=f"{empty} has no rows")]
    if vendor_path is None:
        results = []
        # No vendor run to intersect columns with: use the reference's own case axes
        # (a k / layout / alg sweep would otherwise collapse into duplicate keys).
        extra = list(args.extra_key) + [
            c for c in _OPTIONAL_KEYS if c in ref_fields and c not in args.extra_key
        ]
        for key, row in _index_rows(ref_rows, extra, str(ref_path)).items():
            resource, source = _resource_for(row, args.resource, args.default_resource)
            out = _result(
                op, _label(key), dtype=_dtype(row), resource=resource, source=source
            )
            if _bound(out, row, ref_col, args, peaks):
                out["verdict"] = "BOUND"
            results.append(out)
        return results

    vendor_col = _pick(vendor_fields, args.vendor_column, "vendor", vendor_path)
    extra = list(args.extra_key) + [
        c
        for c in _OPTIONAL_KEYS
        if c in ref_fields and c in vendor_fields and c not in args.extra_key
    ]
    ref_index = _index_rows(ref_rows, extra, str(ref_path))
    vendor_index = _index_rows(vendor_rows, extra, str(vendor_path))

    # Driven by the vendor run: its rows are what is being accepted, and a delivery
    # run sweeps fewer axes than a full reference run (int32/non only), so the
    # reference rows it skipped are not failures. Completeness of the vendor run
    # itself is tools/delivery_table.py's job.
    results = []
    for key, vrow in vendor_index.items():
        out = _result(op, _label(key), dtype=_dtype(vrow))
        results.append(out)
        if not args.all_rows and _baseline_ms(vrow) is not None:
            out["verdict"] = "HAS_BASELINE"
            out["note"] = "vendor row has its own library baseline"
            continue
        row = ref_index.get(key)
        if row is None:
            out["verdict"], out["note"] = "NO_REFERENCE", "no reference row"
            continue
        out["resource"], out["source"] = _resource_for(
            row, args.resource, args.default_resource
        )
        if not _bound(out, row, ref_col, args, peaks):
            continue
        t_v = _num(vrow.get(vendor_col))
        if not _passed(vrow):
            out["verdict"] = "FAIL"
            out["note"] = (
                f"vendor row is not PASS ({vrow.get('status') or 'no status'})"
            )
            continue
        if t_v is None or t_v <= 0:
            out["verdict"], out["note"] = "FAIL", f"no usable vendor {vendor_col}"
            continue
        out["t_vendor_ms"] = t_v
        out["margin"] = out["bound_ms"] / t_v
        # 1.0 = exactly the rescaled H800 time; >= --ratio passes.
        out["speedup"] = out["expected_ms"] / t_v
        out["verdict"] = "PASS" if t_v <= out["bound_ms"] else "FAIL"
    return results


def _evaluate(args, peaks: dict) -> list[dict]:
    ref_ops = _collect(args.reference_run)
    vendor_ops = _collect(args.vendor) if args.vendor else None
    wanted = {o.strip() for o in (args.ops or "").split(",") if o.strip()}
    single = len(ref_ops) == 1 and vendor_ops is not None and len(vendor_ops) == 1
    driver = vendor_ops if vendor_ops is not None else ref_ops
    results = []
    for op in driver:
        if wanted and op not in wanted:
            continue
        # Two bare CSVs pair up whatever their parent directories are called.
        ref_path = next(iter(ref_ops.values())) if single else ref_ops.get(op)
        if ref_path is None:
            results.append(
                _result(
                    op, "*", verdict="NO_REFERENCE", note="reference run has no such op"
                )
            )
            continue
        vendor_path = vendor_ops[op] if vendor_ops is not None else None
        try:
            results.extend(_evaluate_op(op, ref_path, vendor_path, args, peaks))
        except ValueError as exc:
            results.append(_result(op, "*", verdict="ERROR", note=str(exc)))
    return results


_HEADERS = (
    ("op", "op"),
    ("key", "case"),
    ("dtype", "dtype"),
    ("resource", "unit"),
    ("source", "unit from"),
    ("t_ref_ms", "T_ref ms"),
    ("expected_ms", "T' scaled ms"),
    ("bound_ms", "T' max ms"),
    ("t_vendor_ms", "T' ms"),
    ("speedup", "speedup"),
    ("margin", "margin"),
    ("verdict", "verdict"),
    ("note", "note"),
)


def _cell(row: dict, name: str) -> str:
    value = row[name]
    return _fmt(value) if isinstance(value, float) or value is None else str(value)


def _render(results: list[dict], markdown: bool, with_vendor: bool) -> str:
    columns = [
        (n, h)
        for n, h in _HEADERS
        if with_vendor or n not in ("t_vendor_ms", "margin", "speedup")
    ]
    table = [[h for _, h in columns]]
    table += [[_cell(r, n) for n, _ in columns] for r in results]
    if markdown:
        # Case labels are joined with "|" (and gather case ids contain it), which
        # would otherwise add columns to the row.
        table = [[cell.replace("|", "\\|") for cell in row] for row in table]
        lines = ["| " + " | ".join(table[0]) + " |"]
        lines.append("|" + "|".join("---" for _ in columns) + "|")
        lines += ["| " + " | ".join(r) + " |" for r in table[1:]]
        return "\n".join(lines)
    widths = [max(len(r[i]) for r in table) for i in range(len(columns))]
    return "\n".join(
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)).rstrip()
        for r in table
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "reference_run",
        nargs="?",
        help=(
            "H800 results dir, performance.csv or reference JSON "
            f"(default: the bundled {_h800_reference.DEFAULT_PATH.relative_to(_ROOT)})"
        ),
    )
    parser.add_argument(
        "--reference",
        choices=sorted(REFERENCE_PEAKS),
        default="h800-sxm",
        help="built-in reference card peaks (default h800-sxm)",
    )
    parser.add_argument(
        "--vendor-bw-gbs", type=float, help="vendor peak memory bandwidth, GB/s"
    )
    parser.add_argument(
        "--vendor-card",
        choices=sorted(VENDOR_BANDWIDTH_GBS),
        help="use this card's built-in measured bandwidth",
    )
    parser.add_argument("--peaks", help="peaks JSON (see --print-template)")
    parser.add_argument("--print-template", action="store_true")
    parser.add_argument("--vendor", help="vendor results dir or performance.csv")
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="also judge vendor rows that have their own library baseline",
    )
    parser.add_argument("--ops", help="comma-separated operators to evaluate")
    parser.add_argument(
        "--ref-column",
        help=(
            "reference time column. Default: the reference row's vendor-library "
            "(cuSPARSE) time; name e.g. triton_ms to scale FlagSparse's own H800 time"
        ),
    )
    parser.add_argument("--vendor-column", help="vendor time column (default: auto)")
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument(
        "--resource",
        choices=("auto", *RESOURCES),
        default="auto",
        help="force the bottleneck unit for every row (default: per row, see above)",
    )
    parser.add_argument(
        "--default-resource",
        choices=RESOURCES,
        default="mem",
        help="unit for rows nothing identifies a bottleneck for (default mem)",
    )
    parser.add_argument(
        "--extra-key",
        action="append",
        default=[],
        metavar="COL",
        help="extra column that makes rows unique (repeatable)",
    )
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--csv", help="also write the result table here")
    args = parser.parse_args(argv)

    if args.print_template:
        print(json.dumps(_template(), indent=2, ensure_ascii=False))
        return 0
    if not args.reference_run:
        if not _h800_reference.DEFAULT_PATH.is_file():
            parser.error("the reference run is required (or use --print-template)")
        args.reference_run = str(_h800_reference.DEFAULT_PATH)
    if not 0 < args.ratio <= 1:
        parser.error("--ratio must be in (0, 1]")

    _check_reference_run(args)
    peaks = _load_peaks(args)
    results = _evaluate(args, peaks)
    print(_render(results, args.markdown, bool(args.vendor)))

    counts: dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no rows"
    ref_name = peaks["reference"].get("name") or args.reference
    print(
        f"\n{len(results)} rows: {summary}; reference={ref_name}; ratio={args.ratio}",
        file=sys.stderr,
    )
    assumed = sum(1 for r in results if r["source"] == "assumed" and r["bound_ms"])
    if assumed:
        print(
            f"note: {assumed} row(s) use the assumed '{args.default_resource}' "
            "bottleneck; add u_cuda/u_tensor/u_mem or a bottleneck column to measure it",
            file=sys.stderr,
        )

    if args.csv:
        with Path(args.csv).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[n for n, _ in _HEADERS])
            writer.writeheader()
            writer.writerows(results)
    bad = [r for r in results if r["verdict"] not in ("BOUND", "PASS", "HAS_BASELINE")]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
