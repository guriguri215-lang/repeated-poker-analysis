#!/usr/bin/env python3
"""Run the minimum MVP checks: the test suite and the key example scripts.

This is a developer-facing helper. It runs the checks in sequence, stops at the
first failure, and returns that failure's exit code (or 1). It uses only the
Python standard library and has no network or file-output side effects. The
script itself makes no version-control calls; the commands it runs may read the
local git-commit hash while building run manifests (best effort), and nothing
writes version-control state.

It can be run from any directory: the repository root is resolved from the
script's own location, and every command runs with that root as the working
directory and with ``src`` and ``examples`` on the Python path.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMANDS = [
    [sys.executable, "-m", "pytest", "-q"],
    [sys.executable, "examples/analysis_pipeline.py"],
    [sys.executable, "examples/nuts_chop_river.py"],
    [sys.executable, "examples/candidate_filters.py"],
    [
        sys.executable,
        "scripts/run_stt_pushfold_analysis.py",
        "examples/stt_pushfold_2x2.json",
        "--no-markdown",
    ],
    [sys.executable, "examples/aiof_real_card_workflow.py"],
    [sys.executable, "examples/stage_plan_diagnostic_workflow.py"],
]


def _build_env() -> dict:
    """Return an environment with ``src`` and ``examples`` on PYTHONPATH."""

    env = os.environ.copy()
    extra = os.pathsep.join([str(ROOT / "src"), str(ROOT / "examples")])
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = extra if not existing else extra + os.pathsep + existing
    return env


def main() -> int:
    env = _build_env()
    total = len(COMMANDS)
    for index, command in enumerate(COMMANDS, start=1):
        printable = " ".join(command)
        print(f"[{index}/{total}] running: {printable}", flush=True)
        result = subprocess.run(command, cwd=str(ROOT), env=env)
        if result.returncode != 0:
            print(
                f"[{index}/{total}] FAILED (exit {result.returncode}): {printable}",
                flush=True,
            )
            return result.returncode or 1
        print(f"[{index}/{total}] OK: {printable}", flush=True)

    print(f"All {total} MVP checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
