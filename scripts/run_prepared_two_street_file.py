"""Inspect or run a prepared two-street versioned JSON file."""

from __future__ import annotations

import argparse
from pathlib import Path

from repeated_poker.prepared_two_street_file_workflow import (
    PreparedFileWorkflowStatus,
    inspect_prepared_two_street_file,
    prepared_file_workflow_json,
    run_prepared_two_street_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect or run a bounded prepared one-/two-street JSON file."
    )
    parser.add_argument("operation", choices=("inspect", "run"))
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    try:
        raw = args.input.read_bytes()
    except OSError:
        raw = b""
    result = (
        inspect_prepared_two_street_file(raw)
        if args.operation == "inspect"
        else run_prepared_two_street_file(raw)
    )
    print(prepared_file_workflow_json(result))
    return 0 if result.status is PreparedFileWorkflowStatus.SUCCESS else 2


if __name__ == "__main__":
    raise SystemExit(main())
