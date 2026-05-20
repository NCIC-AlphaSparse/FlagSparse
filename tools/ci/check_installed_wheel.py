"""Validate that CI is exercising the installed wheel, not the source tree."""

from __future__ import annotations

import argparse
import os
import pathlib
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
    parser.add_argument("--expected-version", default=None)
    args = parser.parse_args(argv)

    result = validate_installed_wheel(expected_version=args.expected_version)
    print(result.version)
    print(result.module_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
