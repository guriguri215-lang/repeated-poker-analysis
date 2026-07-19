"""Inspect or run one strict stage-plan diagnostic file-v1 JSON document."""

from __future__ import annotations

import argparse
from pathlib import Path

from repeated_poker.stage_plan_diagnostic_file_workflow import (
    StagePlanDiagnosticFileError,
    StagePlanDiagnosticFileResult,
    StagePlanDiagnosticFileStatus,
    process_stage_plan_diagnostic_file,
    stage_plan_diagnostic_file_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect or run a bounded exact-rational stage-plan diagnostic document."
    )
    parser.add_argument("input", type=Path)
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
    except OSError:
        result = StagePlanDiagnosticFileResult(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            None,
            StagePlanDiagnosticFileError("input", "cannot read input file", None),
        )
    else:
        result = process_stage_plan_diagnostic_file(raw)
    print(stage_plan_diagnostic_file_json(result))
    return 0 if result.status is StagePlanDiagnosticFileStatus.SUCCESS else 2


if __name__ == "__main__":
    raise SystemExit(main())
