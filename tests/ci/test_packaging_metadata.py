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

"""CPU-only packaging metadata checks."""

import re
import sys
from pathlib import Path

import pytest

import flagsparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.ci.check_installed_wheel import normalize_version  # noqa: E402


def _read_text(path):
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def _extract_version_from_pyproject():
    text = _read_text("pyproject.toml")
    match = re.search(r'^\s*version\s*=\s*"([^"]+)"\s*$', text, re.MULTILINE)
    assert match is not None, "pyproject.toml is missing project.version"
    return match.group(1)


def _extract_version_from_setup_py():
    text = _read_text("setup.py")
    match = re.search(r'^\s*version\s*=\s*"([^"]+)"\s*,\s*$', text, re.MULTILINE)
    assert match is not None, "setup.py is missing setup(version=...)"
    return match.group(1)


def _extract_requires_python_from_pyproject():
    text = _read_text("pyproject.toml")
    match = re.search(r'^\s*requires-python\s*=\s*"([^"]+)"\s*$', text, re.MULTILINE)
    assert match is not None, "pyproject.toml is missing project.requires-python"
    return match.group(1)


def _extract_python_requires_from_setup_py():
    text = _read_text("setup.py")
    match = re.search(
        r'^\s*python_requires\s*=\s*"([^"]+)"\s*,\s*$', text, re.MULTILINE
    )
    assert match is not None, "setup.py is missing setup(python_requires=...)"
    return match.group(1)


def test_package_version_matches_metadata():
    declared = _extract_version_from_pyproject()
    # The two declarations are compared as written; the installed package reports
    # the PEP 440 form of them (0.4.0-dev1 is installed as 0.4.0.dev1).
    assert declared == _extract_version_from_setup_py()
    assert flagsparse.__version__ == normalize_version(declared)


def test_python_requires_matches_metadata():
    assert _extract_requires_python_from_pyproject() == ">=3.10"
    assert (
        _extract_python_requires_from_setup_py()
        == _extract_requires_python_from_pyproject()
    )


def test_license_metadata_is_apache_2():
    pyproject = _read_text("pyproject.toml")
    assert 'license = "Apache-2.0"' in pyproject
    assert 'license-files = ["LICENSE"]' in pyproject
    assert "Apache (Version 2.0)" in _read_text("README.md")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0.4.0", "0.4.0"),
        ("0.4.0-dev1", "0.4.0.dev1"),
        ("0.4.0.dev1", "0.4.0.dev1"),
        ("1.2.3rc1", "1.2.3rc1"),
    ],
)
def test_declared_versions_normalize_like_a_built_wheel(raw, expected):
    assert normalize_version(raw) == expected


def test_normalization_survives_a_missing_packaging_module(monkeypatch):
    # packaging is not a declared dependency; the fallback must agree for the common case.
    monkeypatch.setitem(sys.modules, "packaging", None)
    monkeypatch.setitem(sys.modules, "packaging.version", None)
    assert normalize_version("0.4.0-dev1") == "0.4.0.dev1"
    assert normalize_version("0.4.0") == "0.4.0"
