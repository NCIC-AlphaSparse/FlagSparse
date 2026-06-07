"""CPU-only packaging metadata checks."""

import re
from pathlib import Path

import flagsparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    version = flagsparse.__version__
    assert version == _extract_version_from_pyproject()
    assert version == _extract_version_from_setup_py()


def test_python_requires_matches_metadata():
    assert _extract_requires_python_from_pyproject() == ">=3.10"
    assert (
        _extract_python_requires_from_setup_py()
        == _extract_requires_python_from_pyproject()
    )


def test_license_metadata_is_apache_2():
    pyproject = _read_text("pyproject.toml")
    assert 'license = { file = "LICENSE" }' in pyproject
    assert "Apache (Version 2.0)" in _read_text("README.md")
