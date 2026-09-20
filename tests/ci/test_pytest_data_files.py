# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""tests/pytest may only read data files that are actually in the tree.

``test_spmv_csr_accuracy.py`` reads ``tests/data/spmv_csr_regressions.json`` at
import time. When a revert deleted that file, the runner's accuracy phase died in
collection for *every* operator on *every* backend (``exit_code=2``, ``total=0``)
and no test under tests/ci noticed, because none of them collects tests/pytest.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTEST_DIR = ROOT / "tests" / "pytest"
DATA_DIR = ROOT / "tests" / "data"
# ``Path(__file__).resolve().parents[1] / "data/foo.json"`` and friends.
DATA_LITERAL = re.compile(r'"data/([^"\s]+)"')


def _referenced():
    found = []
    for source in sorted(PYTEST_DIR.glob("*.py")):
        for name in DATA_LITERAL.findall(source.read_text(encoding="utf-8")):
            found.append((source.name, name))
    return found


def test_the_scan_sees_the_known_regression_manifest():
    # If this stops matching, the pattern above no longer covers how the tests
    # spell the path and the real check below would pass vacuously.
    assert ("test_spmv_csr_accuracy.py", "spmv_csr_regressions.json") in _referenced()


def test_every_data_file_read_by_tests_pytest_exists():
    missing = [
        f"{source} reads tests/data/{name}"
        for source, name in _referenced()
        if not (DATA_DIR / name).is_file()
    ]
    assert not missing, "\n".join(missing)


def test_the_regression_manifest_lists_matrices():
    import json

    manifest = json.loads((DATA_DIR / "spmv_csr_regressions.json").read_text())
    assert manifest["matrices"], "the parametrisation would collect zero cases"
    assert all(name.endswith(".mtx") for name in manifest["matrices"])
