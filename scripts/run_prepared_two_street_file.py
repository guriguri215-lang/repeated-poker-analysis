"""Inspect or run a prepared two-street versioned JSON file."""

from __future__ import annotations

import argparse
from pathlib import Path

from repeated_poker.prepared_two_street_file_workflow import (
    PreparedFileWorkflowError,
    PreparedFileWorkflowResult,
    PreparedFileWorkflowStatus,
    inspect_prepared_two_street_file,
    prepared_file_workflow_json,
    run_prepared_two_street_file,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect or run a bounded prepared one-/two-street JSON file."
    )
    parser.add_argument("operation", choices=("inspect", "run"))
    parser.add_argument("input", type=Path)
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
    except OSError:
        result = PreparedFileWorkflowResult(
            PreparedFileWorkflowStatus.INVALID_INPUT,
            None,
            PreparedFileWorkflowError("input", "cannot read input file", None),
        )
    else:
        result = (
            inspect_prepared_two_street_file(raw)
            if args.operation == "inspect"
            else run_prepared_two_street_file(raw)
        )
    print(prepared_file_workflow_json(result))
    return 0 if result.status is PreparedFileWorkflowStatus.SUCCESS else 2


if __name__ == "__main__":
    raise SystemExit(main())
