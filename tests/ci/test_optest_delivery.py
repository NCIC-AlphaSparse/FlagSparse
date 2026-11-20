# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Cover optest's delivery selection, projection and exit status without a GPU."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tools.delivery_variants import load_delivery_matrices, load_delivery_variants
from tools.optest_delivery import _write_delivery_summary, run_delivery

ROOT = Path(__file__).resolve().parents[2]


def test_optest_delivery_schedules_once_per_parent():
    catalog = yaml.safe_load((ROOT / "conf/operators.yaml").read_text())["ops"]
    variants = load_delivery_variants()
    seen = []
    optest = SimpleNamespace(ROOT=ROOT, CFG=SimpleNamespace())
    optest.get_ops_from_inventory = lambda: catalog

    def select():
        entries = optest.get_ops_from_inventory()
        assert [entry["id"] for entry in entries] == [v["id"] for v in variants]
        assert all(entry["description"] and entry["stages"] for entry in entries)
        return [entry["id"] for entry in entries]

    def main():
        seen.extend(optest.get_ops_to_test())
        inputs = Path(os.environ["FLAGSPARSE_OPTEST_BENCHMARK_INPUT"])
        assert {p.stem for p in inputs.glob("*.mtx")} == set(load_delivery_matrices())
        return 3

    optest.get_ops_to_test = select
    optest.main = main
    assert run_delivery(optest, None) == 3
    assert seen == list(dict.fromkeys(v["operator"] for v in variants))
    assert optest.get_ops_to_test is select


@pytest.mark.parametrize("missing", [False, True])
def test_optest_delivery_keeps_classic_and_tagged_variants_distinct(tmp_path, missing):
    op = "spmv_csr"
    classic, transposed = "spmv_csr_f32_int_non", "spmv_csr_f32_int_trans"
    variants = [v for v in load_delivery_variants() if v["id"] in {classic, transposed}]
    accuracy_path = tmp_path / "accuracy_result.json"
    accuracy_path.write_text(
        json.dumps(
            {
                "test_spmv_csr_accuracy.py::test_values[float32-int32-non]": {
                    "params": {
                        "dtype": "torch.float32",
                        "index_dtype": "torch.int32",
                        "op": "non",
                    },
                    "result": "passed",
                },
                f"test_q4_variants_accuracy.py::test_variant[{transposed}-small]": {
                    "params": {"name": transposed},
                    "result": "passed",
                },
            }
        )
    )
    row = {"dtype": "float32", "latency_base": "2", "latency": "1", "speedup": "2"}
    rows = [dict(row, op="non")]
    if not missing:
        rows.append(
            dict(row, variant=transposed, op="trans", latency="4", speedup="0.5")
        )
    (tmp_path / f"{op}.json").write_text(
        json.dumps({"status": "Passed", "records": rows})
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "result": {
                    op: {
                        "accuracy": {"data_file": accuracy_path.name},
                        "performance": {},
                    }
                }
            }
        )
    )
    optest = SimpleNamespace(
        CFG=SimpleNamespace(output_dir=tmp_path), count_op_statuses=lambda *_: {}
    )
    assert _write_delivery_summary(optest, variants, tmp_path) == int(missing)
    results = json.loads(summary_path.read_text())["result"]
    assert set(results) == {classic, transposed}
    assert results[classic]["accuracy"]["passed"] == 1
    assert results[classic]["performance"]["data"]["fp32"]["speedup"] == 2
    assert results[transposed]["accuracy"]["passed"] == 1
    performance = results[transposed]["performance"]
    if missing:
        assert performance["status"] == "NotFound"
        assert not performance["data"]
    else:
        assert performance["data"]["fp32"]["speedup"] == 0.5
