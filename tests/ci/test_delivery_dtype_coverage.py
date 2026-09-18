# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""A benchmark script must cover every delivery dtype it is responsible for.

THE INCIDENT THIS GUARDS. scatter declares five delivery variants -- f16, f32,
f64, c32, c64 -- and `tests/test_scatter.py` ran only the three real ones, while
its sibling `test_gather.py` had carried complex all along. Accuracy passed 8/8
for scatter_c32_int and scatter_c64_int, so nothing looked wrong; the only
symptom was a performance row reading

    NOT_CONFIGURED  the benchmark recorded no c32 rows

which is exactly the failure shape this repo keeps running into: an empty result
reads like a pass. The kernel was fine -- forcing the dtypes by hand produced 16
cases, all PASS, with a working cuSPARSE baseline. The gap was this one list.

Only scripts that DECLARE ``DEFAULT_VALUE_DTYPES`` can be checked statically, so
the test discovers them rather than hardcoding a list: a script that adopts the
constant later is picked up automatically, and one that never does is out of
scope here (its dtype selection lives somewhere this test cannot read).
"""

import ast
import importlib.util
import sys
from pathlib import Path

from tools.delivery_variants import load_delivery_variants

ROOT = Path(__file__).resolve().parents[2]

# The registry's dtype tags, in the spelling the benchmark CLIs use.
TAG_TO_TORCH_NAME = {
    "f16": "float16",
    "bf16": "bfloat16",
    "f32": "float32",
    "f64": "float64",
    "c32": "complex64",
    "c64": "complex128",
}


def _declared_default_dtypes(path):
    """The script's DEFAULT_VALUE_DTYPES, read without importing it.

    Static parsing on purpose: importing a benchmark CLI drags in torch and runs
    its sys.path surgery, neither of which belongs in a CPU-only policy test.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "DEFAULT_VALUE_DTYPES":
                return {item.strip() for item in str(node.value.value).split(",")}
    return None


def _operator_to_script():
    spec = importlib.util.spec_from_file_location(
        "_runner_for_dtype_coverage", ROOT / "run_flagsparse_pytest.py"
    )
    runner = importlib.util.module_from_spec(spec)
    sys.modules["_runner_for_dtype_coverage"] = runner
    spec.loader.exec_module(runner)
    mapping = {}
    for op, config in runner.OP_TEST_CONFIGS.items():
        command = getattr(config, "performance_cmd", None)
        if command:
            mapping[op] = command[0]
    return mapping


def test_declared_default_dtypes_cover_their_delivery_variants():
    scripts = _operator_to_script()
    variants = load_delivery_variants(ROOT / "conf" / "operators.yaml")

    wanted = {}
    for variant in variants:
        script = scripts.get(variant["operator"])
        if script is None:
            continue
        wanted.setdefault(script, set()).add(variant["dtype"])

    checked = 0
    for script, dtype_tags in sorted(wanted.items()):
        declared = _declared_default_dtypes(ROOT / script)
        if declared is None:
            continue  # selects dtypes some other way; out of this test's reach
        checked += 1
        required = {TAG_TO_TORCH_NAME[tag] for tag in dtype_tags}
        missing = required - declared
        assert not missing, (
            f"{script} declares DEFAULT_VALUE_DTYPES without {sorted(missing)}, "
            f"so the delivery variants needing them report NOT_CONFIGURED with "
            f"'the benchmark recorded no <dtype> rows' -- an empty performance row, "
            f"not a failing one"
        )

    # The discovery must actually find something; a rename of the constant would
    # otherwise turn this test into one that passes by checking nothing.
    assert checked >= 2, f"expected to check at least gather and scatter, got {checked}"
