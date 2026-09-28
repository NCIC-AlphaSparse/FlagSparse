"""The bundled H800 reference (conf/h800_reference.json) and the tool that makes it."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools import baseline_bound, h800_reference  # noqa: E402
from tools.delivery_variants import load_delivery_variants  # noqa: E402

BUNDLED = h800_reference.DEFAULT_PATH

# The operators the 20 delivery variants (and the q4 variants of them) are benchmarked with.
DELIVERY_PARENTS = (
    "gather",
    "scatter",
    "spmv_csr",
    "spmv_coo",
    "spmm_csr",
    "spmm_coo",
    "sddmm_csr",
)


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sddmm_row(matrix="a.mtx", k="32", cusparse="2.0", triton="1.0", **extra):
    return {
        "matrix": matrix,
        "value_dtype": "float32",
        "index_dtype": "torch.int32",
        "n_rows": "10",
        "n_cols": "10",
        "nnz": "30",
        "k": k,
        "triton_ms": triton,
        "cusparse_ms": cusparse,
        "pytorch_ms": "9.0",
        "triton_speedup_vs_cusparse": "2.0",
        "status": "PASS",
        "cu_status": "OK",
        # noise the bundle must drop
        "process_cpu_ms": "5.5",
        "gpu_ms": "1.0",
        "prepare_ms": "0.3",
        "max_rel_err": "1e-7",
        "cu_reason": "long free text",
        "alpha": "1.0",
        **extra,
    }


def _synthetic_run(tmp_path, rows=None):
    root = tmp_path / "pytest_results_fake"
    _write_csv(root / "sddmm_csr" / "performance.csv", rows or [_sddmm_row()])
    (root / "summary.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-09-22 18:01:14",
                "env": {
                    "torch": {
                        "version": "2.8",
                        "device_name": "NVIDIA H800",
                        "device_count": 8,
                    },
                    "triton": {"version": "3.6.0"},
                    "flag_gems": {"vendor": "nvidia", "device": "cuda"},
                },
                "result": {
                    "sddmm_csr_f32_int_non_non_row": {
                        "accuracy": {"status": "Passed", "passed": 2, "total": 2},
                        "performance": {
                            "status": "Passed",
                            "data": {"fp32": {"speedup": 3.25}},
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return root


# ---------------------------------------------------------------------------
# the generator
# ---------------------------------------------------------------------------


def test_the_bundle_keeps_case_keys_times_and_status_and_drops_the_noise(tmp_path):
    doc = h800_reference.build(_synthetic_run(tmp_path))
    table = doc["operators"]["sddmm_csr"]
    fields = table["fields"]
    assert {
        "matrix",
        "value_dtype",
        "index_dtype",
        "k",
        "triton_ms",
        "cusparse_ms",
        "pytorch_ms",
        "triton_speedup_vs_cusparse",
        "status",
        "cu_status",
    } <= set(fields)
    for dropped in (
        "process_cpu_ms",
        "gpu_ms",
        "prepare_ms",
        "max_rel_err",
        "cu_reason",
        "alpha",
    ):
        assert dropped not in fields


def test_axis_columns_stay_strings_and_times_become_numbers(tmp_path):
    doc = h800_reference.build(_synthetic_run(tmp_path))
    table = doc["operators"]["sddmm_csr"]
    row = dict(zip(table["fields"], table["rows"][0]))
    assert (
        row["k"] == "32" and row["nnz"] == "30" and row["index_dtype"] == "torch.int32"
    )
    assert row["cusparse_ms"] == 2.0 and isinstance(row["triton_ms"], float)


def test_an_empty_time_is_null_and_reads_back_as_an_empty_cell(tmp_path):
    root = _synthetic_run(tmp_path, [_sddmm_row(cusparse="")])
    out = tmp_path / "ref.json"
    h800_reference.write(h800_reference.build(root), out)
    fields, rows = h800_reference.tables(out)["sddmm_csr"].read()
    assert rows[0]["cusparse_ms"] == ""
    assert float(rows[0]["triton_ms"]) == 1.0


def test_the_reference_block_records_where_the_numbers_came_from(tmp_path):
    ref = h800_reference.build(_synthetic_run(tmp_path))["reference"]
    assert ref["device"] == "NVIDIA H800" and ref["device_count"] == 8
    assert ref["source_dir"] == "pytest_results_fake"
    assert ref["run_timestamp"] == "2026-09-22 18:01:14"


def test_the_commit_comes_from_the_csv_when_the_env_has_none(tmp_path):
    root = _synthetic_run(tmp_path, [_sddmm_row(commit="abc123+dirty")])
    assert (
        h800_reference.build(root)["reference"]["flagsparse_commit"] == "abc123+dirty"
    )


def test_variant_results_are_summarised(tmp_path):
    doc = h800_reference.build(_synthetic_run(tmp_path))
    v = doc["variants"]["sddmm_csr_f32_int_non_non_row"]
    assert v["accuracy"] == "Passed" and v["performance"] == "Passed"
    assert v["speedup"] == {"fp32": 3.25}


def test_a_directory_without_a_summary_is_refused(tmp_path):
    (tmp_path / "x").mkdir()
    with pytest.raises(SystemExit):
        h800_reference.build(tmp_path / "x")


def test_cli_writes_the_json_and_the_markdown(tmp_path):
    root = _synthetic_run(tmp_path)
    out, doc = tmp_path / "ref.json", tmp_path / "ref.md"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "h800_reference.py"),
            str(root),
            "--out",
            str(out),
            "--doc",
            str(doc),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (
        json.loads(out.read_text(encoding="utf-8"))["schema"] == h800_reference.SCHEMA
    )
    text = doc.read_text(encoding="utf-8")
    assert "sddmm_csr_f32_int_non_non_row" in text and "NVIDIA H800" in text


def test_a_file_that_is_not_a_reference_is_refused(tmp_path):
    bad = tmp_path / "x.json"
    bad.write_text(json.dumps({"schema": 99}), encoding="utf-8")
    with pytest.raises(SystemExit):
        h800_reference.load(bad)


# ---------------------------------------------------------------------------
# the file and the directory give the same baseline
# ---------------------------------------------------------------------------


def _scaled(reference, rows):
    fields = list(rows[0])
    return baseline_bound.scaled_baseline_rows(
        "sddmm_csr",
        rows,
        fields,
        str(reference),
        "iluvatar-biv150",
        needs_baseline=lambda row: True,
    )


def test_scaled_baseline_is_identical_from_the_file_and_from_the_directory(tmp_path):
    root = _synthetic_run(
        tmp_path,
        [
            _sddmm_row("a.mtx", k="32", cusparse="2.0"),
            _sddmm_row("a.mtx", k="64", cusparse="4.0"),
            _sddmm_row("b.mtx", k="32", cusparse="3.0"),
        ],
    )
    out = h800_reference.write(h800_reference.build(root), tmp_path / "ref.json")
    ours = [
        {
            "matrix": "a.mtx",
            "value_dtype": "float32",
            "index_dtype": "torch.int32",
            "k": "32",
            "triton_ms": "1.0",
            "cusparse_ms": "",
        },
        {
            "matrix": "a.mtx",
            "value_dtype": "float32",
            "index_dtype": "torch.int32",
            "k": "64",
            "triton_ms": "8.0",
            "cusparse_ms": "",
        },
        {
            "matrix": "b.mtx",
            "value_dtype": "float32",
            "index_dtype": "torch.int32",
            "k": "32",
            "triton_ms": "1.0",
            "cusparse_ms": "",
        },
    ]
    from_dir, info_dir = _scaled(root, [dict(r) for r in ours])
    from_file, info_file = _scaled(out, [dict(r) for r in ours])
    assert info_file["filled"] == info_dir["filled"] == 3
    assert from_file == from_dir
    # the baseline is the H800 cuSPARSE time (2 ms), not FlagSparse's own (1 ms), rescaled
    assert float(from_file[0]["h800_vendor_ms"]) == 2.0
    assert float(from_file[0]["h800_scaled_ms"]) == pytest.approx(2.0 * 3050 / 1150)


def test_the_tool_detects_the_reference_card_from_the_file(tmp_path):
    out = h800_reference.write(
        h800_reference.build(_synthetic_run(tmp_path)), tmp_path / "r.json"
    )
    assert baseline_bound._run_env(str(out)) == ("nvidia", "NVIDIA H800")


def test_a_reference_file_on_the_wrong_card_is_flagged(tmp_path, capsys):
    doc = h800_reference.build(_synthetic_run(tmp_path))
    doc["reference"]["device"] = "Iluvatar BI-V150"
    out = h800_reference.write(doc, tmp_path / "r.json")
    ns = type(
        "A", (), {"peaks": None, "reference_run": str(out), "reference": "h800-sxm"}
    )
    baseline_bound._check_reference_run(ns)
    assert "WARNING" in capsys.readouterr().err


def test_bound_only_cli_defaults_to_the_bundled_file(tmp_path):
    if not BUNDLED.is_file():
        pytest.skip("bundled reference not generated")
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "baseline_bound.py"),
            "--vendor-card",
            "iluvatar-biv150",
            "--ops",
            "sddmm_csr",
            "--markdown",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "sddmm_csr" in proc.stdout


# ---------------------------------------------------------------------------
# the shipped file
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bundled():
    if not BUNDLED.is_file():
        pytest.skip("conf/h800_reference.json not generated")
    return h800_reference.load(BUNDLED)


def test_the_bundled_file_is_an_h800_run(bundled):
    ref = bundled["reference"]
    assert "H800" in ref["device"]
    assert ref["peaks"] == "h800-sxm"
    assert ref["flagsparse_commit"]


def test_the_bundled_file_has_every_delivery_operator(bundled):
    missing = [op for op in DELIVERY_PARENTS if op not in bundled["operators"]]
    assert not missing, missing


def test_every_bundled_table_has_a_case_key_a_time_and_a_status(bundled):
    for op, table in bundled["operators"].items():
        fields = table["fields"]
        assert table["rows"], op
        assert all(len(row) == len(fields) for row in table["rows"]), op
        assert "status" in fields, op
        assert any(f in fields for f in ("matrix", "case_id")), op
        assert any(h800_reference._is_time(f) for f in fields), op


def test_every_delivery_operator_has_a_usable_library_time_column(bundled):
    for op in DELIVERY_PARENTS:
        fields, rows = h800_reference.tables(BUNDLED)[op].read()
        usable = [r for r in rows if baseline_bound._baseline_ms(r)]
        assert usable, f"{op}: no row carries a vendor-library time ({fields})"


def test_the_bundled_variants_are_the_20_delivery_variants(bundled):
    assert set(bundled["variants"]) == {v["id"] for v in load_delivery_variants()}
    assert all(v["accuracy"] == "Passed" for v in bundled["variants"].values())


def test_the_bundled_file_matches_its_source_directory_when_present(bundled):
    source = ROOT / bundled["reference"]["source_dir"]
    if not source.is_dir():
        pytest.skip("the original results directory is not on this machine")
    fresh = h800_reference.build(source)
    assert fresh["operators"] == bundled["operators"]
    assert fresh["variants"] == bundled["variants"]


def test_the_bundled_file_stays_small(bundled):
    assert BUNDLED.stat().st_size < 2 * 1024 * 1024
