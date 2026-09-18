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
