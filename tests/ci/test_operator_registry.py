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

"""Checks for the FlagSparse operator registry."""

import re
from pathlib import Path

import flagsparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = PROJECT_ROOT / "conf" / "operators.yaml"


def _registry_text():
    return REGISTRY.read_text(encoding="utf-8")


def _operator_ids(text):
    return re.findall(r"^  - id: ([a-z0-9_]+)$", text, flags=re.MULTILINE)


def _registered_interfaces(text):
    return re.findall(r"^      - ([A-Za-z_][A-Za-z0-9_]*)$", text, flags=re.MULTILINE)


def test_operator_registry_uses_flagext_style_shape():
    text = _registry_text()
    # Strip copyright/comment lines and leading blanks before checking
    lines = [line for line in text.splitlines() if not line.startswith("#")]
    while lines and lines[0] == "":
        lines.pop(0)
    cleaned = chr(10).join(lines)
    assert cleaned.startswith("ops:\n")
    for snippet in [
        "description: |",
        "for:",
        "labels:",
        "kind:",
        "stages:",
    ]:
        assert snippet in text


def test_operator_registry_ids_are_unique():
    text = _registry_text()
    ids = _operator_ids(text)
    assert ids
    assert len(ids) == len(set(ids))


def test_operator_registry_interfaces_are_public_exports():
    text = _registry_text()
    exported = set(flagsparse.__all__)
    ignored_yaml_values = {
        "flagsparse",
        "sparse",
        "triton",
        "public-api",
        "csr",
        "coo",
        "csc",
        "bsr",
        "bell",
        "sell",
        "optimized",
        "alpha-sparse",
        "descriptor",
        "format",
        "Sparse",
        "SparseLinearAlg",
        "SparseSolver",
        "SparseFormat",
        "Utility",
    }
    interfaces = {
        name for name in _registered_interfaces(text) if name not in ignored_yaml_values
    }
    assert interfaces
    assert interfaces <= exported
