"""Write a SHA256 manifest for release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import List, Optional


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(dist_dir: Path, manifest_name: str = "SHA256SUMS") -> Path:
    artifacts = sorted(
        path
        for path in dist_dir.iterdir()
        if path.is_file() and path.name != manifest_name
    )
    if not artifacts:
        raise AssertionError(f"no artifacts found in {dist_dir}")

    manifest_path = dist_dir / manifest_name
    lines = [f"{_sha256(path)}  {path.name}" for path in artifacts]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def verify_manifest(dist_dir: Path, manifest_name: str = "SHA256SUMS") -> Path:
    manifest_path = dist_dir / manifest_name
    if not manifest_path.is_file():
        raise AssertionError(f"missing checksum manifest: {manifest_path}")

    expected = {}
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        digest, filename = line.split("  ", 1)
        expected[filename] = digest

    for filename, digest in expected.items():
        artifact = dist_dir / filename
        if not artifact.is_file():
            raise AssertionError(
                f"missing release artifact listed in manifest: {artifact}"
            )
        if _sha256(artifact) != digest:
            raise AssertionError(f"checksum mismatch for {artifact}")

    return manifest_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", nargs="?", default="dist")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)

    dist_dir = Path(args.dist_dir)
    manifest_path = (
        verify_manifest(dist_dir) if args.verify else write_manifest(dist_dir)
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
