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

"""The q4 operators reach the unified runner's performance phase on every backend."""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
Q4_OPS = ("axpby", "spvv", "spmv_sell")


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location(
        "_runner_q4", ROOT / "run_flagsparse_pytest.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["_runner_q4"] = module
    spec.loader.exec_module(module)
    return module


def _script_fields(script):
    source = (ROOT / script).read_text(encoding="utf-8")
    block = re.search(r"FIELDS = \[(.*?)\]", source, re.S).group(1)
    return source, set(re.findall(r'"([a-z_0-9]+)"', block))


@pytest.mark.parametrize("op", Q4_OPS)
def test_q4_op_has_accuracy_marker_and_performance_command(runner, op):
    config = runner.OP_TEST_CONFIGS[op]
    assert config.accuracy_marker == op
    assert config.performance_cmd and (ROOT / config.performance_cmd[0]).is_file()
    # Ascend reuses the same backend-neutral script instead of reporting NOT_CONFIGURED.
    assert runner.ASCEND_PERFORMANCE_COMMANDS[op] == config.performance_cmd


@pytest.mark.parametrize("op", Q4_OPS)
def test_q4_script_accepts_every_flag_and_writes_a_known_speedup_schema(runner, op):
    command = runner.OP_TEST_CONFIGS[op].performance_cmd
    source, fields = _script_fields(command[0])
    for flag in (arg for arg in command if arg.startswith("--")):
        assert f'"{flag}"' in source, f"{command[0]} has no {flag}"
    schemas = [s for s in runner.PERFORMANCE_SPEEDUP_SCHEMAS if set(s) <= fields]
    # Vendor first, then PyTorch -- the order the runner resolves a row in.
    assert schemas[0] == ("triton_speedup_vs_cusparse", "cusparse_ms", "triton_ms")
    assert ("triton_speedup_vs_pytorch", "pytorch_ms", "triton_ms") in schemas
    assert {"status", "cusparse_reason"} <= fields


def test_q4_rows_without_a_vendor_baseline_fall_back_to_pytorch(runner):
    row = {
        "value_dtype": "float32",
        "triton_ms": "0.02",
        "cusparse_ms": "",
        "pytorch_ms": "0.04",
        "triton_speedup_vs_cusparse": "",
        "triton_speedup_vs_pytorch": "2.0",
        "status": "PASS",
    }
    assert runner._performance_schema(row) == (
        "triton_speedup_vs_pytorch",
        "pytorch_ms",
        "triton_ms",
    )
    assert runner._performance_row_has_complete_speedup(row)


Q4_VARIANT_OPS = (
    "spmv_csr",
    "spmv_coo",
    "spmv_csc",
    "spmm_csr",
    "spmm_coo",
    "sddmm_csr",
)


@pytest.fixture(scope="module")
def variant_bench():
    pytest.importorskip("torch", reason="q4_variant_bench imports torch")
    sys.path.insert(0, str(ROOT / "tests"))
    sys.path.insert(0, str(ROOT / "src"))
    import q4_variant_bench

    return q4_variant_bench


@pytest.mark.parametrize("op", Q4_VARIANT_OPS)
def test_existing_script_runs_its_q4_variants_in_the_runner(runner, op):
    command = runner.OP_TEST_CONFIGS[op].performance_cmd
    assert "--q4-variants" in command
    source = (ROOT / command[0]).read_text(encoding="utf-8")
    assert '"--q4-variants"' in source and "run_and_append(" in source


@pytest.mark.parametrize("op", Q4_VARIANT_OPS)
@pytest.mark.parametrize("with_vendor", [True, False])
def test_q4_rows_resolve_to_the_scripts_own_speedup_schema(
    runner, variant_bench, op, with_vendor
):
    generic = {
        "variant": variant_bench.VARIANTS[op][0].name,
        "value_dtype": "float32",
        "ours_ms": 0.02,
        "vendor_ms": 0.03 if with_vendor else None,
        "pytorch_ms": 0.04,
        "speedup_vs_vendor": 1.5 if with_vendor else None,
        "speedup_vs_pytorch": 2.0,
        "status": "PASS",
    }
    mapping = variant_bench.COLUMN_MAPS[op]
    row = {mapping.get(k, k): ("" if v is None else str(v)) for k, v in generic.items()}
    assert runner._performance_row_has_complete_speedup(row)
    speedup_key, base_key, latency_key = runner._performance_schema(row)
    assert float(row[speedup_key]) == (1.5 if with_vendor else 2.0)
    assert base_key == mapping.get(
        "vendor_ms" if with_vendor else "pytorch_ms",
        "vendor_ms" if with_vendor else "pytorch_ms",
    )
    assert latency_key == mapping.get("ours_ms", "ours_ms")


def test_every_listed_existing_op_variant_is_covered(variant_bench):
    names = [v.name for vs in variant_bench.VARIANTS.values() for v in vs]
    assert len(names) == len(set(names)) == 30


def test_registry_holds_20_delivery_and_45_q4_variants():
    sys.path.insert(0, str(ROOT))
    from tools.delivery_variants import load_delivery_variants, load_q4_variants

    delivery, q4 = load_delivery_variants(), load_q4_variants()
    assert len(delivery) == 20 and len(q4) == 45
    assert len({v["id"] for v in delivery + q4}) == 65


def test_q4_accuracy_slice_is_exactly_the_cases_carrying_the_variant_id(
    runner, tmp_path
):
    q4_file = "tests/pytest/test_q4_variants_accuracy.py"
    raw = {
        f"{q4_file}::test_q4_variant_matches_cpu_golden[spmv_csr_f32_int_trans-64x48x16]": {
            "result": "passed",
            "params": {},
        },
        f"{q4_file}::test_q4_variant_matches_cpu_golden[spmv_csr_f16f32_int_non-64x48x16]": {
            "result": "failed",
            "params": {},
        },
        "tests/pytest/test_spmv_csr_accuracy.py::test_x[float32-int32-non]": {
            "result": "passed",
            "params": {"dtype": "float32", "index_dtype": "int32", "op": "non"},
        },
    }
    path = tmp_path / "accuracy_result.json"
    path.write_text(__import__("json").dumps(raw), encoding="utf-8")
    phase = {"result_path": str(path), "status": "PASS"}

    q4 = runner._delivery_accuracy_phase(phase, "f32", "spmv_csr_f32_int_trans")
    assert (q4["passed"], q4.get("failed", 0)) == (1, 0)
    # The delivered spmv_csr_f32_int_non does not absorb the q4 trans case.
    delivered = runner._delivery_accuracy_phase(phase, "f32")
    assert delivered["passed"] == 1 and delivered.get("failed", 0) == 0
    missing = runner._delivery_accuracy_phase(phase, "i8i32", "spmv_csr_i8i32_int_non")
    assert missing["status"] == "NOT_CONFIGURED"


def test_q4_performance_slice_takes_only_rows_tagged_with_the_variant(runner):
    tagged = {
        "variant": "spmv_csr_f32_int_trans",
        "dtype": "float32",
        "op": "trans",
        "ms": "0.02",
        "vendor_ms": "0.01",
        "speedup_vs_vendor": "0.5",
        "status": "PASS",
        "matrix": "a.mtx",
    }
    own = dict(tagged, variant="", op="non", speedup_vs_vendor="2.0", vendor_ms="0.04")
    phase = {"status": "PASS", "records": [tagged, own]}
    q4 = runner._q4_performance_phase(phase, "spmv_csr_f32_int_trans")
    assert q4["delivery_row_count"] == 1
    assert [round(v["speedup"], 3) for v in q4["data"].values()] == [0.5]
    delivered = runner._delivery_performance_phase(phase, "f32")
    assert delivered["delivery_row_count"] == 1
    assert [round(v["speedup"], 3) for v in delivered["data"].values()] == [2.0]


def test_delivery_shaped_q4_variant_falls_back_to_the_untagged_rows(runner):
    """spmm_bsr/bell/csc f32 and gather/scatter int8 are measured by the scripts'
    own rows, which carry no ``variant`` tag."""
    rows = [
        {
            "dtype": "float32",
            "op": "non",
            "index_dtype": "int32",
            "ms": "0.02",
            "cusparse_ms": "0.04",
            "cusparse_vs_alg_speedup": "2.0",
            "status": "PASS",
        },
        {
            "dtype": "float32",
            "op": "trans",
            "index_dtype": "int32",
            "ms": "0.5",
            "cusparse_ms": "0.1",
            "cusparse_vs_alg_speedup": "0.2",
            "status": "PASS",
        },
        {
            "dtype": "float64",
            "op": "non",
            "index_dtype": "int32",
            "ms": "0.04",
            "cusparse_ms": "0.08",
            "cusparse_vs_alg_speedup": "2.0",
            "status": "PASS",
        },
    ]
    phase = {"status": "PASS", "records": rows}
    result = runner._q4_performance_phase(phase, "spmm_bsr_f32_int_non_non_row", "f32")
    assert result["status"] != "NOT_CONFIGURED"
    assert "untagged rows" in result.get("matched_by", "")
    # Only the f32 non row: the trans row and the f64 row are off the delivery axes.
    assert [round(v["speedup"], 3) for v in result["data"].values()] == [2.0]

    # A variant that is not delivery-shaped gets no fallback -- it would silently
    # report the `non` rows as if they were the transposed ones.
    assert (
        runner._q4_performance_phase(phase, "spmv_csr_f32_int_trans", "f32")["status"]
        == "NOT_CONFIGURED"
    )


def test_int8_reaches_the_gather_scatter_benchmarks_and_compares_exactly():
    sys.path.insert(0, str(ROOT / "src"))
    torch = pytest.importorskip("torch")
    from flagsparse.sparse_operations._common import (
        _build_random_dense,
        _tolerance_for_dtype,
    )

    for script in ("test_gather.py", "test_scatter.py"):
        source = (ROOT / "tests" / script).read_text(encoding="utf-8")
        default = re.search(r'DEFAULT_VALUE_DTYPES = "([^"]+)"', source).group(1)
        assert "int8" in default.split(","), script
        assert '"int8",' in source, script  # accepted by the --value-dtypes parser
    # gather resolves the token itself; scatter defers to the shared resolver.
    assert '"int8": torch.int8' in (ROOT / "tests" / "test_gather.py").read_text(
        encoding="utf-8"
    )
    from flagsparse.sparse_operations._common import _resolve_scatter_value_dtype

    assert _resolve_scatter_value_dtype("int8")[0] == torch.int8
    values = _build_random_dense(64, torch.int8, torch.device("cpu"))
    assert values.dtype == torch.int8 and values.numel() == 64
    # Moved bytes are compared exactly; a float tolerance would hide a wrong byte.
    assert _tolerance_for_dtype(torch.int8) == (0.0, 0.0)
