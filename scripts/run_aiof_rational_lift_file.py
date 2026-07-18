"""Run one strict AIoF rational-lift v1 JSON document."""

from __future__ import annotations

import argparse
from pathlib import Path

from repeated_poker.aiof_rational_lift_file_workflow import (
    AiofRationalLiftFileError,
    AiofRationalLiftFileResult,
    AiofRationalLiftFileStatus,
    aiof_rational_lift_file_json,
    run_aiof_rational_lift_file,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a bounded exact AIoF rational-lift JSON document."
    )
    parser.add_argument("input", type=Path)
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
    except OSError:
        result = AiofRationalLiftFileResult(
            AiofRationalLiftFileStatus.INVALID_INPUT,
            None,
            AiofRationalLiftFileError("input", "cannot read input file", None),
        )
    else:
        result = run_aiof_rational_lift_file(raw)
    print(aiof_rational_lift_file_json(result))
    return 0 if result.status is AiofRationalLiftFileStatus.SUCCESS else 2


if __name__ == "__main__":
    raise SystemExit(main())
