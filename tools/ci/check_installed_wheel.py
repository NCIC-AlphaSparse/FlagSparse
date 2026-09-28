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

"""Validate that CI is exercising the installed wheel, not the source tree."""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import List, Optional


@dataclass(frozen=True)
class WheelImportCheck:
    """Result of validating an installed wheel import."""

    version: str
    module_path: pathlib.Path


def normalize_version(raw: str) -> str:
    """The PEP 440 form a built wheel reports for a version written in pyproject.toml.

    ``0.4.0-dev1`` is written that way but installs as ``0.4.0.dev1``; comparing the
    raw text is what broke the wheel check when the dev version landed.
    """
    try:
        from packaging.version import Version

        return str(Version(raw))
    except ImportError:  # packaging is not a declared dependency
        return re.sub(r"[-_.]?(dev|a|b|rc)(\d+)", r".\1\2", raw)


def project_version(project_root: Optional[pathlib.Path] = None) -> str:
    """The version pyproject.toml declares, normalized as a wheel reports it.

    pyproject.toml stays the only place the version lives.
    """
    root = project_root or pathlib.Path(__file__).resolve().parents[2]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^\[project\]\s.*?^version\s*=\s*"([^"]+)"', text, re.S | re.M)
    if match is None:
        raise AssertionError("no [project] version in pyproject.toml")
    return normalize_version(match.group(1))


def validate_installed_wheel(
    expected_version: Optional[str] = None,
) -> WheelImportCheck:
    """Import flagsparse from a clean interpreter and verify the module path."""
    project_root = pathlib.Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    code = (
        "import flagsparse, pathlib; "
        "print(flagsparse.__version__); "
        "print(pathlib.Path(flagsparse.__file__).resolve())"
    )
    with TemporaryDirectory() as tmpdir:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=tmpdir,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )

    lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    if len(lines) < 2:
        raise AssertionError(proc.stdout)

    version = lines[0]
    if expected_version == "auto":
        expected_version = project_version(project_root)
    if expected_version is not None and version != expected_version:
        raise AssertionError(f"expected version {expected_version!r}, got {version!r}")

    module_path = pathlib.Path(lines[1]).resolve()
    if project_root in module_path.parents:
        raise AssertionError(f"installed wheel resolved into repo tree: {module_path}")
    if not module_path.is_file():
        raise AssertionError(f"installed wheel path is not a file: {module_path}")

    return WheelImportCheck(version=version, module_path=module_path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-version",
        default=None,
        help="version the installed wheel must report, or 'auto' for pyproject.toml's",
    )
    args = parser.parse_args(argv)

    result = validate_installed_wheel(expected_version=args.expected_version)
    print(result.version)
    print(result.module_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
