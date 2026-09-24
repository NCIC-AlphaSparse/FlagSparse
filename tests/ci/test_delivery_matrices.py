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

from tools import delivery_variants

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
    assert delivery_variants.load_delivery_matrices() == DELIVERY_MATRICES


def test_a_list_that_is_empty_or_repeats_is_rejected(tmp_path):
    empty = tmp_path / "a.yaml"
    empty.write_text("delivery_matrices: []\n")
    with pytest.raises(ValueError, match="non-empty"):
        delivery_variants.load_delivery_matrices(empty)
    repeated = tmp_path / "b.yaml"
    repeated.write_text("delivery_matrices: [a, b, a]\n")
    with pytest.raises(ValueError, match="repeats"):
        delivery_variants.load_delivery_matrices(repeated)


def test_the_filter_keeps_only_the_delivery_matrices(tmp_path):
    source = _corpus(tmp_path, [*DELIVERY_MATRICES, "net150", "trdheim", "mip1"])
    (source / "wave.mtx.hicsr").write_text("sidecar")
    picked, note = delivery_variants.select_delivery_matrices(source, tmp_path / "out")
    assert note is None
    assert sorted(p.stem for p in picked.glob("*.mtx")) == sorted(DELIVERY_MATRICES)
    assert (picked / "wave.mtx").read_text() == "matrix wave\n"
    assert not list(picked.glob("*.hicsr"))


def test_refiltering_replaces_an_earlier_selection(tmp_path):
    source = _corpus(tmp_path, ["a", "b", "c"])
    dest = tmp_path / "out"
    delivery_variants.select_delivery_matrices(source, dest, ["a", "b"])
    delivery_variants.select_delivery_matrices(source, dest, ["c"])
    assert [p.stem for p in dest.glob("*.mtx")] == ["c"]


def test_a_directory_without_them_is_used_as_given_and_says_so(tmp_path):
    """tests/data has three small matrices: a smoke run must still run."""
    source = _corpus(tmp_path, ["net150", "trdheim", "wave"])
    picked, note = delivery_variants.select_delivery_matrices(source, tmp_path / "out")
    assert picked == source
    assert "lacks delivery matrices" in note and "GL7d14" in note
    assert not (tmp_path / "out").exists()


def test_one_missing_matrix_never_quietly_shrinks_the_run(tmp_path):
    source = _corpus(tmp_path, [n for n in DELIVERY_MATRICES if n != "cfd2"])
    picked, note = delivery_variants.select_delivery_matrices(source, tmp_path / "out")
    assert picked == source and "['cfd2']" in note


def test_a_single_file_input_is_used_as_given(tmp_path):
    single = tmp_path / "wave.mtx"
    single.write_text("x")
    picked, note = delivery_variants.select_delivery_matrices(single, tmp_path / "out")
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
    # WHICH matrices the filter picks is pinned by
    # test_the_filter_keeps_only_the_delivery_matrices, at the unit level where the
    # directory still exists.  Here the point is the runner reaching that code path
    # and then clearing up after itself: the entries are symlinks into the source
    # corpus, so a results directory that outlives them carries dangling links.
    delivery_matrices = tmp_path / "res" / "delivery_matrices"
    assert not delivery_matrices.exists(), "delivery matrix symlink directory survived"
    assert sorted(p.stem for p in source.glob("*.mtx")) == sorted(
        [*DELIVERY_MATRICES, "net150"]
    ), "cleanup must never touch the source corpus"


def test_cleanup_only_removes_the_symlinks_it_made(tmp_path):
    """Pointing cleanup at a real corpus, or a dirtied dir, must remove nothing."""
    source = _corpus(tmp_path, DELIVERY_MATRICES)
    dest = tmp_path / "out"
    delivery_variants.select_delivery_matrices(source, dest)

    delivery_variants.cleanup_delivery_matrices(source)
    assert sorted(p.stem for p in source.glob("*.mtx")) == sorted(DELIVERY_MATRICES)
    assert source.is_dir()

    (dest / "notes.txt").write_text("keep me", encoding="utf-8")
    delivery_variants.cleanup_delivery_matrices(dest)
    assert dest.is_dir(), "a directory holding anything else must survive"
    assert list(dest.glob("*.mtx")) == []
    assert (dest / "notes.txt").exists()

    (dest / "notes.txt").unlink()
    delivery_variants.cleanup_delivery_matrices(dest)
    assert not dest.exists()
