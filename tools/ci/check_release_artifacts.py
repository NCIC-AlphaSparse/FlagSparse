"""Validate release artifacts produced by the CI/CD pipeline."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


WHEEL_RE = re.compile(r"^flagsparse-(?P<version>[^-]+)-py3-none-any\.whl$")
SDIST_RE = re.compile(r"^flagsparse-(?P<version>[^-]+)\.tar\.gz$")


def _extract_version(path: Path, pattern: re.Pattern[str]) -> str:
    match = pattern.match(path.name)
    if match is None:
        raise AssertionError(f"unexpected artifact name: {path.name}")
    return match.group("version")


def validate_release_artifacts(dist_dir: Path) -> tuple[Path, Path]:
    """Return the wheel and sdist after validating their names and versions."""
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))

    if len(wheels) != 1:
        raise AssertionError(f"expected one wheel, found {len(wheels)}: {wheels}")
    if len(sdists) != 1:
        raise AssertionError(f"expected one sdist, found {len(sdists)}: {sdists}")

    wheel = wheels[0]
    sdist = sdists[0]
    wheel_version = _extract_version(wheel, WHEEL_RE)
    sdist_version = _extract_version(sdist, SDIST_RE)
    if wheel_version != sdist_version:
        raise AssertionError(
            f"artifact versions do not match: wheel={wheel_version!r}, sdist={sdist_version!r}"
        )

    return wheel, sdist


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", nargs="?", default="dist")
    args = parser.parse_args(argv)

    wheel, sdist = validate_release_artifacts(Path(args.dist_dir))
    print(wheel)
    print(sdist)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
