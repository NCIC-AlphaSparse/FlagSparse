# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""tools/baseline_bound.py: T' <= T_h800 * (P_h800 / P_vendor) / ratio."""

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "baseline_bound.py"
sys.path.insert(0, str(ROOT))
from tools.baseline_bound import REFERENCE_PEAKS  # noqa: E402

H800_BW = REFERENCE_PEAKS["h800-sxm"]["mem_bw_gbs"]
HALF_BW = str(H800_BW / 2)

PEAKS = {
    "reference": {
        "mem_bw_gbs": 1000.0,
        "cuda_tflops": {"fp32": 40.0, "fp64": 1.0},
        "tensor_tflops": {"fp16": 200.0},
    },
    "vendor": {
        "mem_bw_gbs": 500.0,
        "cuda_tflops": {"fp32": 10.0, "fp64": 0.5},
        "tensor_tflops": {"fp16": 100.0},
    },
}


def _write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            fields += [k for k in row if k not in fields]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _ref(matrix="a.mtx", dtype="float32", t=1.0, status="PASS", **extra):
    # An H800 row: FlagSparse time, the cuSPARSE baseline it was accepted against.
    return {
        "matrix": matrix,
        "value_dtype": dtype,
        "triton_ms": t,
        "cusparse_ms": t,
        "status": status,
        **extra,
    }


def _ven(matrix="a.mtx", dtype="float32", t=1.0, status="PASS", **extra):
    # A vendor row with no library baseline: cusparse_ms is present but empty.
    return {
        "matrix": matrix,
        "value_dtype": dtype,
        "triton_ms": t,
        "cusparse_ms": "",
        "status": status,
        **extra,
    }


def _call(tmp_path, *args):
    cmd = [sys.executable, str(TOOL), *map(str, args), "--csv", tmp_path / "out.csv"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    out = tmp_path / "out.csv"
    rows = list(csv.DictReader(out.open(encoding="utf-8"))) if out.exists() else []
    return proc, rows


def _run(tmp_path, ref_rows, vendor_rows=None, *extra):
    _write_csv(tmp_path / "ref.csv", ref_rows)
    args = [tmp_path / "ref.csv", *extra]
    if not {"--peaks", "--vendor-bw-gbs", "--vendor-card"} & set(extra):
        (tmp_path / "peaks.json").write_text(json.dumps(PEAKS), encoding="utf-8")
        args += ["--peaks", tmp_path / "peaks.json"]
    if vendor_rows is not None:
        _write_csv(tmp_path / "vendor.csv", vendor_rows)
        args += ["--vendor", tmp_path / "vendor.csv"]
    return _call(tmp_path, *args)


def test_h800_sxm_peaks_match_the_datasheet():
    h800 = REFERENCE_PEAKS["h800-sxm"]
    assert h800["mem_bw_gbs"] == 3050.0  # measured, not the 3350 datasheet
    assert h800["cuda_tflops"]["fp32"] == 67.0 and h800["cuda_tflops"]["fp64"] == 1.0
    pcie = REFERENCE_PEAKS["h800-pcie"]
    assert pcie["mem_bw_gbs"] == 2000.0 and pcie["cuda_tflops"]["fp32"] == 51.0


def test_template_carries_h800_and_leaves_the_vendor_blank():
    proc = subprocess.run(
        [sys.executable, str(TOOL), "--print-template"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    tpl = json.loads(proc.stdout)
    assert tpl["reference"]["mem_bw_gbs"] == H800_BW
    assert tpl["vendor"]["mem_bw_gbs"] is None
    assert all(v is None for v in tpl["vendor"]["cuda_tflops"].values())


def test_half_the_h800_bandwidth_allows_twice_the_time_over_0_8(tmp_path):
    # 10 ms on H800; vendor has half the bandwidth -> 20 ms expected, 25 ms allowed.
    ref = [_ref("a.mtx", t=10.0), _ref("b.mtx", t=10.0), _ref("c.mtx", t=10.0)]
    vendor = [_ven("a.mtx", t=24.0), _ven("b.mtx", t=25.0), _ven("c.mtx", t=26.0)]
    proc, rows = _run(tmp_path, ref, vendor, "--vendor-bw-gbs", HALF_BW)
    assert proc.returncode == 1
    by = {r["key"].split("|")[0]: r for r in rows}
    assert {k: float(r["bound_ms"]) for k, r in by.items()} == {
        "a.mtx": 25.0,
        "b.mtx": 25.0,
        "c.mtx": 25.0,
    }
    # exactly at the bound counts
    assert {k: r["verdict"] for k, r in by.items()} == {
        "a.mtx": "PASS",
        "b.mtx": "PASS",
        "c.mtx": "FAIL",
    }
    assert by["a.mtx"]["resource"] == "mem" and by["a.mtx"]["source"] == "assumed"
    assert "H800 SXM" in proc.stderr and "assumed" in proc.stderr


def test_bound_only_mode_needs_no_vendor_run(tmp_path):
    proc, rows = _run(tmp_path, [_ref(t=10.0)], None, "--vendor-bw-gbs", HALF_BW)
    assert proc.returncode == 0, proc.stderr
    assert rows[0]["verdict"] == "BOUND" and float(rows[0]["bound_ms"]) == 25.0


def test_vendor_card_uses_its_measured_bandwidth(tmp_path):
    # MetaX C550 1440 GB/s against H800 3050: 1.0 * 3050 / 1440 / 0.8
    proc, rows = _run(tmp_path, [_ref(t=1.0)], None, "--vendor-card", "maca-c550")
    assert proc.returncode == 0, proc.stderr
    assert abs(float(rows[0]["bound_ms"]) - 3050 / 1440 / 0.8) < 1e-3
    # an explicit bandwidth wins over the card table
    proc, rows = _run(
        tmp_path,
        [_ref(t=1.0)],
        None,
        "--vendor-card",
        "maca-c550",
        "--vendor-bw-gbs",
        "3050",
    )
    assert float(rows[0]["bound_ms"]) == 1.25


def test_pcie_reference_changes_the_ratio(tmp_path):
    proc, rows = _run(
        tmp_path,
        [_ref(t=1.0)],
        None,
        "--reference",
        "h800-pcie",
        "--vendor-bw-gbs",
        1000,
    )
    assert proc.returncode == 0, proc.stderr
    assert float(rows[0]["bound_ms"]) == 2.5  # 1.0 * 2000 / 1000 / 0.8


def test_a_row_with_its_own_vendor_baseline_is_not_judged(tmp_path):
    vendor = [_ven("a.mtx", t=100.0, cusparse_ms=90.0)]
    proc, rows = _run(tmp_path, [_ref("a.mtx", t=1.0)], vendor)
    assert proc.returncode == 0, proc.stderr
    assert rows[0]["verdict"] == "HAS_BASELINE" and rows[0]["bound_ms"] == ""
    proc, rows = _run(tmp_path, [_ref("a.mtx", t=1.0)], vendor, "--all-rows")
    assert proc.returncode == 1 and rows[0]["verdict"] == "FAIL"


def test_the_reference_row_must_itself_have_passed(tmp_path):
    proc, rows = _run(tmp_path, [_ref(status="FAIL")], [_ven()])
    assert proc.returncode == 1
    assert rows[0]["verdict"] == "N/A" and "not PASS" in rows[0]["note"]


def test_a_failed_vendor_row_fails_even_when_fast(tmp_path):
    proc, rows = _run(tmp_path, [_ref(t=1.0)], [_ven(t=0.1, status="FAIL")])
    assert proc.returncode == 1 and rows[0]["verdict"] == "FAIL"


def test_a_vendor_row_the_reference_lacks_is_no_reference(tmp_path):
    ref = [_ref("a.mtx"), _ref("wide-sweep-only.mtx")]
    vendor = [_ven("a.mtx", t=1.0), _ven("b.mtx", t=1.0)]
    proc, rows = _run(tmp_path, ref, vendor)
    assert proc.returncode == 1
    # Driven by the vendor run: the reference's extra row is not reported.
    assert {r["key"].split("|")[0]: r["verdict"] for r in rows} == {
        "a.mtx": "PASS",
        "b.mtx": "NO_REFERENCE",
    }


def test_time_columns_are_picked_per_side_from_the_runner_schemas(tmp_path):
    # CUDA spmv_csr writes triton_ms/cusparse_ms, ROCm writes ms/vendor_ms.
    ref = [_ref(t=1.0, pytorch_ms=0.1)]
    vendor = [
        {
            "matrix": "a.mtx",
            "dtype": "float32",
            "ms": 2.4,
            "vendor_ms": "",
            "status": "PASS",
        }
    ]
    proc, rows = _run(tmp_path, ref, vendor)
    assert proc.returncode == 0, proc.stderr
    assert float(rows[0]["t_ref_ms"]) == 1.0 and float(rows[0]["t_vendor_ms"]) == 2.4
    # spmv_coo: the delivered kernel is opt_ms, not base_ms.
    ref = [
        {
            "matrix": "a.mtx",
            "value_dtype": "float32",
            "base_ms": 9.0,
            "opt_ms": 1.0,
            "cusparse_ms": 1.0,
            "status": "PASS",
        }
    ]
    proc, rows = _run(tmp_path, ref, None)
    assert float(rows[0]["t_ref_ms"]) == 1.0


def test_spsv_columns_are_recognised(tmp_path):
    ref = [
        {
            "matrix": "a.mtx",
            "value_dtype": "float32",
            "opA": "NON",
            "FlagSparse_ms": 1.0,
            "CuPy/cuSPARSE_ms": 1.0,
            "status": "PASS",
        }
    ]
    vendor = [
        {
            "matrix": "a.mtx",
            "value_dtype": "float32",
            "opA": "NON",
            "FlagSparse_ms": 2.0,
            "hipSPARSE_ms": "",
            "status": "PASS",
        }
    ]
    proc, rows = _run(tmp_path, ref, vendor)
    assert proc.returncode == 0, proc.stderr
    assert rows[0]["verdict"] == "PASS" and rows[0]["key"] == "a.mtx|float32|non"


def test_bottleneck_follows_the_largest_nsight_utilisation(tmp_path):
    rows_in = [
        _ref("mem.mtx", u_cuda=30, u_tensor=5, u_mem=70),
        _ref("cuda.mtx", u_cuda=80, u_tensor=1, u_mem=20),
        _ref("named.mtx", bottleneck="mem", u_cuda=99),
    ]
    proc, rows = _run(tmp_path, rows_in)
    assert proc.returncode == 0, proc.stderr
    by = {r["key"].split("|")[0]: r for r in rows}
    assert (by["mem.mtx"]["resource"], by["mem.mtx"]["source"]) == ("mem", "profile")
    assert float(by["mem.mtx"]["bound_ms"]) == 2.5
    # fp32 vector peaks 40 vs 10: 1.0 * 4 / 0.8 = 5.0
    assert (by["cuda.mtx"]["resource"], by["cuda.mtx"]["source"]) == ("cuda", "profile")
    assert float(by["cuda.mtx"]["bound_ms"]) == 5.0
    assert (by["named.mtx"]["resource"], by["named.mtx"]["source"]) == ("mem", "column")


def test_complex_dtype_uses_the_real_unit_of_the_same_precision(tmp_path):
    # complex128 -> fp64 vector peaks 1.0 vs 0.5: 1.0 * 2 / 0.8 = 2.5
    proc, rows = _run(tmp_path, [_ref(dtype="complex128")], None, "--resource", "cuda")
    assert proc.returncode == 0, proc.stderr
    assert float(rows[0]["bound_ms"]) == 2.5


def test_ratio_is_configurable(tmp_path):
    proc, rows = _run(tmp_path, [_ref()], None, "--ratio", "0.5")
    assert proc.returncode == 0, proc.stderr
    assert float(rows[0]["bound_ms"]) == 4.0


def test_missing_peak_is_reported_not_guessed(tmp_path):
    proc, rows = _run(tmp_path, [_ref()], None, "--resource", "tensor")
    assert proc.returncode == 1
    assert rows[0]["verdict"] == "N/A" and "tensor_tflops[fp32]" in rows[0]["note"]


def test_no_vendor_peaks_is_an_error(tmp_path):
    _write_csv(tmp_path / "ref.csv", [_ref()])
    proc = subprocess.run(
        [sys.executable, str(TOOL), str(tmp_path / "ref.csv")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode != 0 and "--vendor-bw-gbs" in proc.stderr


def test_results_directories_pair_up_by_operator(tmp_path):
    ref_dir, ven_dir = tmp_path / "h800", tmp_path / "vendor"
    _write_csv(ref_dir / "gather" / "performance.csv", [_ref(t=10.0)])
    _write_csv(ref_dir / "spmm_csr" / "performance.csv", [_ref(t=1.0)])
    _write_csv(ref_dir / "spmv_csr" / "performance.csv", [_ref(t=1.0)])
    _write_csv(ven_dir / "gather" / "performance.csv", [_ven(t=20.0)])
    _write_csv(ven_dir / "spmv_csr" / "performance.csv", [], fields=["matrix", "ms"])
    _write_csv(ven_dir / "sddmm_csr" / "performance.csv", [_ven()])
    proc, rows = _call(
        tmp_path, ref_dir, "--vendor", ven_dir, "--vendor-bw-gbs", HALF_BW
    )
    assert proc.returncode == 1
    assert {r["op"]: r["verdict"] for r in rows} == {
        "gather": "PASS",
        "sddmm_csr": "NO_REFERENCE",
        "spmv_csr": "EMPTY",  # header-only CSV must not vanish from the table
    }


def test_duplicate_keys_are_an_error_until_disambiguated(tmp_path):
    rows_in = [_ref(alg="1"), _ref(alg="2", t=2.0)]
    proc, rows = _run(tmp_path, rows_in)
    assert proc.returncode == 1
    assert rows[0]["verdict"] == "ERROR" and "--extra-key" in rows[0]["note"]
    proc, rows = _run(tmp_path, rows_in, None, "--extra-key", "alg")
    assert proc.returncode == 0, proc.stderr
    assert sorted(float(r["bound_ms"]) for r in rows) == [2.5, 5.0]


def test_a_case_axis_both_sides_carry_joins_the_key(tmp_path):
    ref = [_ref(alg="1", t=1.0), _ref(alg="2", t=2.0)]
    vendor = [_ven(alg="1", t=2.5), _ven(alg="2", t=2.5)]
    proc, rows = _run(tmp_path, ref, vendor)
    assert proc.returncode == 0, proc.stderr
    # alg 1: 1.0 * 2 / 0.8 = 2.5 (at the bound); alg 2: bound 5.0
    assert sorted((r["key"], r["bound_ms"], r["verdict"]) for r in rows) == [
        ("a.mtx|float32|1", "2.5", "PASS"),
        ("a.mtx|float32|2", "5.0", "PASS"),
    ]


def test_markdown_table_is_well_formed(tmp_path):
    proc, _ = _run(tmp_path, [_ref()], None, "--markdown")
    lines = proc.stdout.strip().splitlines()
    assert lines[0].startswith("| op | case |") and set(lines[1]) <= {"|", "-"}
    # "|" inside a cell (case labels are joined with it) must be escaped, or the row
    # grows extra columns.
    assert len({len(re.split(r"(?<!\\)\|", line)) for line in lines}) == 1
    assert "a.mtx\\|float32" in lines[2]
