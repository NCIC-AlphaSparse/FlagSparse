# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""A delivery run measures the matrices listed in conf/operators.yaml.

`--delivery-only` filters the --benchmark-input directory down to them, so every
backend passes the same directory and still measures the same 10 matrices.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from tools.delivery_variants import load_delivery_matrices, select_delivery_matrices

ROOT = Path(__file__).resolve().parents[2]

DELIVERY_MATRICES = [
    "GL7d14",
    "auto",
    "NACA0015",
    "amazon0601",
    "filter3D",
    "cage12",
    "roadNet-TX",
    "wave",
    "ASIC_680ks",
    "cfd2",
]


def _corpus(tmp_path, names):
    source = tmp_path / "corpus"
    source.mkdir()
    for name in names:
        (source / f"{name}.mtx").write_text(f"matrix {name}\n")
    return source


def test_the_registry_names_the_ten_delivery_matrices():
    assert load_delivery_matrices() == DELIVERY_MATRICES


def test_a_list_that_is_empty_or_repeats_is_rejected(tmp_path):
    empty = tmp_path / "a.yaml"
    empty.write_text("delivery_matrices: []\n")
    with pytest.raises(ValueError, match="non-empty"):
        load_delivery_matrices(empty)
    repeated = tmp_path / "b.yaml"
    repeated.write_text("delivery_matrices: [a, b, a]\n")
    with pytest.raises(ValueError, match="repeats"):
        load_delivery_matrices(repeated)


def test_the_filter_keeps_only_the_delivery_matrices(tmp_path):
    source = _corpus(tmp_path, [*DELIVERY_MATRICES, "net150", "trdheim", "mip1"])
    (source / "wave.mtx.hicsr").write_text("sidecar")
    picked, note = select_delivery_matrices(source, tmp_path / "out")
    assert note is None
    assert sorted(p.stem for p in picked.glob("*.mtx")) == sorted(DELIVERY_MATRICES)
    assert (picked / "wave.mtx").read_text() == "matrix wave\n"
    assert not list(picked.glob("*.hicsr"))


def test_refiltering_replaces_an_earlier_selection(tmp_path):
    source = _corpus(tmp_path, ["a", "b", "c"])
    dest = tmp_path / "out"
    select_delivery_matrices(source, dest, ["a", "b"])
    select_delivery_matrices(source, dest, ["c"])
    assert [p.stem for p in dest.glob("*.mtx")] == ["c"]


def test_a_directory_without_them_is_used_as_given_and_says_so(tmp_path):
    """tests/data has three small matrices: a smoke run must still run."""
    source = _corpus(tmp_path, ["net150", "trdheim", "wave"])
    picked, note = select_delivery_matrices(source, tmp_path / "out")
    assert picked == source
    assert "lacks delivery matrices" in note and "GL7d14" in note
    assert not (tmp_path / "out").exists()


def test_one_missing_matrix_never_quietly_shrinks_the_run(tmp_path):
    source = _corpus(tmp_path, [n for n in DELIVERY_MATRICES if n != "cfd2"])
    picked, note = select_delivery_matrices(source, tmp_path / "out")
    assert picked == source and "['cfd2']" in note


def test_a_single_file_input_is_used_as_given(tmp_path):
    single = tmp_path / "wave.mtx"
    single.write_text("x")
    picked, note = select_delivery_matrices(single, tmp_path / "out")
    assert picked == single and "not a directory" in note


def test_the_runner_filters_under_delivery_only_without_any_extra_option(tmp_path):
    source = _corpus(tmp_path, [*DELIVERY_MATRICES, "net150"])
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_flagsparse_pytest.py"),
            "--phase",
            "performance",
            "--delivery-only",
            "--ops",
            "gather",
            "--benchmark-input",
            str(source),
            "--results-dir",
            str(tmp_path / "res"),
            "--timeout",
            "60",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert "matrices from" in done.stdout, done.stdout + done.stderr
    kept = sorted(
        p.stem for p in (tmp_path / "res" / "delivery_matrices").glob("*.mtx")
    )
    assert kept == sorted(DELIVERY_MATRICES)
