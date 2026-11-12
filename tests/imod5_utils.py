"""Shared by the tests that compare against the iMOD5 executable (``test_*_imod5.py``)."""

import os
import subprocess
from pathlib import Path

import pytest

# If the value of IMOD5_EXE does not point to an existing file, ALL tests in a module using
# IMOD5_MARKS will be skipped. The path can be overridden by setting the IMOD5_EXE environment
# variable.
IMOD5_EXE = Path(
    os.environ.get(
        "IMOD5_EXE", r"C:\Development\bin\iMOD5\iMOD 5.6.1\iMOD_V5_6_1.exe"
    )
)
IMOD5_MARKS = [
    pytest.mark.imod5,
    pytest.mark.skipif(
        not IMOD5_EXE.exists(),
        reason=f"{IMOD5_EXE} not found. Set the IMOD5_EXE environment variable to the path of the iMOD5 executable.",
    ),
]

TIMEOUT_SECONDS = 300


def run_imod5(inifile: Path) -> subprocess.CompletedProcess:
    """Run iMOD5 on the ini file, in its folder. The console output is kept next to it."""
    result = subprocess.run(
        [str(IMOD5_EXE), str(inifile)],
        cwd=inifile.parent,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=TIMEOUT_SECONDS,
        check=False,
    )
    (inifile.parent / "imod5_console.txt").write_text(
        f"returncode: {result.returncode}\n"
        f"--- stdout\n{result.stdout}\n"
        f"--- stderr\n{result.stderr}\n",
        encoding="utf-8",
    )
    return result
