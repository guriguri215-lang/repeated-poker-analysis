#!/usr/bin/env python3
"""Validate river scenario JSON inputs before running any analysis.

Usage:

    python scripts/validate_river_scenario.py examples/scenarios
    python scripts/validate_river_scenario.py a.json b.json
    python scripts/validate_river_scenario.py examples/scenarios --continue-on-error

It expands the inputs (a directory's ``*.json`` in filename order, or the given
files in order), and for each scenario loads the JSON, parses it, and builds the
game tree -- but stops there. It does not generate candidates, run the
exact-response solver, or run the analysis pipeline. The point is a fast,
traceback-free pre-flight check that an input is well formed and a view of which
model kind it is interpreted as.

It prints a table-like summary to stdout (one row per scenario, with the ok /
error counts) and can save the rows as JSON with ``--output-json``. Without
``--continue-on-error`` the first bad file aborts with a short ``error: ...``
message and a non-zero exit, never a traceback.

For the full candidate analysis of a scenario, use
``scripts/run_river_scenario_analysis.py``; for a batch comparison, use
``scripts/run_scenario_batch.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repeated_poker import (  # noqa: E402  (path is set up above)
    ScenarioValidationConfig,
    validate_river_scenario_inputs,
    write_validation_json,
)

_STDOUT_COLUMNS = [
    "source_path",
    "ok",
    "scenario_id",
    "model_kind",
    "hero_info_set_count",
    "villain_info_set_count",
    "terminal_count",
    "error_type",
]


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Validate river scenario JSON inputs before running analysis."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="scenario JSON files, or a directory whose *.json files are read",
    )
    parser.add_argument(
        "--continue-on-error",
        dest="continue_on_error",
        action="store_true",
        help="record failing scenarios in the summary instead of stopping",
    )
    parser.add_argument(
        "--output-json",
        dest="output_json",
        default=None,
        help="save the validation rows as JSON to this path",
    )
    parser.add_argument(
        "--strict-json",
        dest="strict_json",
        action="store_true",
        help=(
            "emit RFC 8259-compatible JSON, mapping non-finite floats to null "
            "(applies to --output-json)"
        ),
    )
    return parser.parse_args(argv)


def _print_summary(validation) -> None:
    print(
        f"scenarios: {len(validation.rows)} "
        f"(ok: {validation.ok_count}, errors: {validation.error_count})"
    )
    print(" | ".join(_STDOUT_COLUMNS))
    for row in validation.rows:
        row_dict = row.to_dict()
        cells = [
            "" if row_dict.get(column) is None else str(row_dict.get(column))
            for column in _STDOUT_COLUMNS
        ]
        print(" | ".join(cells))
        # Show the short error message on its own line so it is easy to read.
        if not row.ok and row.error_message:
            print(f"    error: {row.error_type}: {row.error_message}")


def main(argv) -> int:
    args = _parse_args(argv)
    config = ScenarioValidationConfig(continue_on_error=args.continue_on_error)
    try:
        validation = validate_river_scenario_inputs(args.inputs, config)
    except Exception as exc:  # noqa: BLE001 - report fail-fast errors cleanly
        # Without --continue-on-error the first bad file aborts validation; print
        # a short message to stderr and exit non-zero rather than dumping a
        # traceback. --continue-on-error never reaches here.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_summary(validation)
    if args.output_json is not None:
        write_validation_json(validation, args.output_json, strict=args.strict_json)
        print(f"saved JSON to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
