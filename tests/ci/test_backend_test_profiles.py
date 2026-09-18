# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Keep backend-owned test profiles aligned with dispatch and CTest."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUITES = ROOT / "tests" / "backends"
CAPI_PROFILES = ROOT / "capi" / "ctest" / "backends"
EXPECTED = {
    "cuda": {"selector": "cuda", "capi_backend": "CUDA", "capi_buildable": True},
    "rocm": {"selector": "rocm", "capi_backend": "HCU", "capi_buildable": False},
    "maca": {"selector": "metax", "capi_backend": "MACA", "capi_buildable": False},
    "musa": {"selector": "mthreads", "capi_backend": "MUSA", "capi_buildable": True},
    "ascend": {"selector": "ascend", "capi_backend": "NPU", "capi_buildable": False},
    "xpu": {"selector": "xpu", "capi_backend": "XPU", "capi_buildable": False},
}


def test_every_dispatch_backend_owns_a_python_and_capi_test_profile():
    assert {path.parent.name for path in SUITES.glob("*/suite.json")} == set(EXPECTED)
    assert {path.stem for path in CAPI_PROFILES.glob("*.cmake")} == set(EXPECTED)


def test_backend_test_profiles_have_the_expected_runtime_and_capi_state():
    for backend, expected in EXPECTED.items():
        suite_path = SUITES / backend / "suite.json"
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        assert suite["backend"] == backend
        assert {key: suite[key] for key in expected} == expected
        assert suite["pytest_paths"] == ["tests/pytest"]
        assert suite["benchmark_runner"] == "run_flagsparse_pytest.py"

        profile = (CAPI_PROFILES / f"{backend}.cmake").read_text(encoding="utf-8")
        assert f'set(FLAGSPARSE_CTEST_PROFILE "{backend}")' in profile
        selector = expected["selector"]
        assert f'set(FLAGSPARSE_CTEST_SELECTOR "{selector}")' in profile
        buildable = "ON" if expected["capi_buildable"] else "OFF"
        assert f"set(FLAGSPARSE_CTEST_CAPI_BUILDABLE {buildable})" in profile


def test_only_cuda_and_musa_can_register_capi_cases_today():
    assert {
        backend for backend, profile in EXPECTED.items() if profile["capi_buildable"]
    } == {"cuda", "musa"}
