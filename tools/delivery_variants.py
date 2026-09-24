#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Load and validate the shared Python/C API delivery variant registry."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "conf" / "operators.yaml"
REQUIRED_KEYS = frozenset({"id", "operator", "format", "dtype"})


def load_delivery_variants(path: Path | None = None) -> list[dict[str, str]]:
    """Return ordered, validated delivery variants from the canonical manifest."""
    manifest = path or DEFAULT_MANIFEST
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    variants = raw.get("delivery_variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError(f"{manifest}: delivery_variants must be a non-empty list")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(variants):
        if not isinstance(item, dict):
            raise ValueError(
                f"{manifest}: delivery_variants[{index}] must be a mapping"
            )
        missing = REQUIRED_KEYS.difference(item)
        if missing:
            raise ValueError(
                f"{manifest}: delivery_variants[{index}] lacks {sorted(missing)}"
            )
        variant = {key: str(item[key]) for key in REQUIRED_KEYS}
        if variant["id"] in seen:
            raise ValueError(f"{manifest}: duplicate variant id {variant['id']!r}")
        seen.add(variant["id"])
        result.append(variant)
    return result


def load_delivery_matrices(path: Path | None = None) -> list[str]:
    """Return the matrix names (without `.mtx`) a delivery run measures."""
    manifest = path or DEFAULT_MANIFEST
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    matrices = raw.get("delivery_matrices")
    if not isinstance(matrices, list) or not matrices:
        raise ValueError(f"{manifest}: delivery_matrices must be a non-empty list")
    names = [str(item) for item in matrices]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{manifest}: delivery_matrices repeats {duplicates}")
    return names


def cleanup_delivery_matrices(dest_dir: Path) -> None:
    """Remove the symlink directory `select_delivery_matrices` built, if it is ours.

    The directory is an intermediate: the benchmark scripts glob it during the
    run and nothing reads it afterwards, while `performance.csv`'s `matrix`
    column already records which matrices a row used.  Leaving it behind is
    worse than clutter -- the entries are symlinks into the source corpus, so a
    results directory that gets copied or archived carries dangling links that
    look like missing data.

    Only ever removes symlinks this module creates, and then `rmdir`s, which
    fails on a non-empty directory.  That is the safety property: pass it a real
    matrix directory (which happens when the filter was NOT applied and the
    caller got the source back) and it removes nothing.
    """
    if not dest_dir.is_dir():
        return
    for entry in dest_dir.glob("*.mtx"):
        if entry.is_symlink():
            entry.unlink()
    try:
        dest_dir.rmdir()
    except OSError:
        # Something else lives here; leave it alone rather than guess.
        pass


def select_delivery_matrices(
    source_dir: Path, dest_dir: Path, names: list[str] | None = None
) -> tuple[Path, str | None]:
    """Filter a matrix directory down to the delivery matrices.

    Returns `(directory, note)`. When `source_dir` holds every delivery matrix, the
    directory is `dest_dir`, filled with symlinks to just those files (every benchmark
    script takes a directory and globs `*.mtx` itself, so presenting a directory that
    holds only the chosen matrices filters all of them at once), and `note` is None.
    When it does not -- a smoke run on tests/data, a single file, a partial copy -- the
    input is returned unchanged with a `note` saying so and naming what was missing:
    that runs a superset, never quietly fewer matrices than a delivery run.
    """
    wanted = names if names is not None else load_delivery_matrices()
    if not source_dir.is_dir():
        return source_dir, f"{source_dir} is not a directory; using it as given"
    missing = [name for name in wanted if not (source_dir / f"{name}.mtx").is_file()]
    if missing:
        return source_dir, (
            f"{source_dir} lacks delivery matrices {missing}; using every .mtx in it"
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    for stale in dest_dir.glob("*.mtx"):
        stale.unlink()
    for name in wanted:
        (dest_dir / f"{name}.mtx").symlink_to((source_dir / f"{name}.mtx").resolve())
    return dest_dir, None
