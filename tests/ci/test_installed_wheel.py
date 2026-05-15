"""CPU-only check that CI validates the installed wheel, not the source tree."""

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_installed_wheel_import_resolves_outside_repo_tree():
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
    assert lines, proc.stdout
    assert lines[0] == "1.0.0"
    module_path = Path(lines[1]).resolve()
    assert PROJECT_ROOT not in module_path.parents
    assert module_path.is_file()

