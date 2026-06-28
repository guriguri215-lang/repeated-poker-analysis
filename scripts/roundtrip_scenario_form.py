#!/usr/bin/env python3
"""Round-trip a JSON scenario through its form model and write it back out.

Usage:

    python scripts/roundtrip_scenario_form.py examples/scenarios/range_matrix_steal_bet98.json
    python scripts/roundtrip_scenario_form.py a.json --output reports/a.json --force
    python scripts/roundtrip_scenario_form.py a.json --output - --strict-json

This is a pre-GUI developer tool that mimics a future GUI "save": it loads one
scenario JSON, detects its form-model mode, runs that mode's
``*_form_from_dict`` / ``validate_*_form`` / ``*_form_to_dict``, and -- only when
the form validates cleanly and the ``to_dict`` output re-parses and rebuilds --
emits that round-tripped JSON. With ``--output PATH`` it writes the file (refusing
to overwrite without ``--force``, creating parent directories as needed);
otherwise, or with ``--output -``, it prints the JSON to stdout with no other
output mixed in.

It does not edit, analyse, or run the candidate pipeline; for inspection without
writing use ``scripts/inspect_scenario_form.py``. It reuses the existing form
model, parser, and game builder as the source of truth and adds no new model or
analysis logic.

Validation messages, a failed round-trip, or any write problem print
``error: ...`` to stderr and exit non-zero, never a traceback, and never leave a
partially valid file from a form that failed the checks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for the inspect helpers

from repeated_poker import (  # noqa: E402  (path is set up above)
    build_river_steal_game_from_scenario,
    detect_scenario_form_mode,
    river_scenario_from_dict,
)
from repeated_poker.report_export import _dump_json  # noqa: E402

# Reuse the scenario loader and the mode -> (from_dict, validate, to_dict) mapping
# from the inspection CLI rather than duplicating them (the same script-to-script
# reuse wizard_run_scenario uses for wizard_create_scenario).
from inspect_scenario_form import (  # noqa: E402
    _MODE_HELPERS,
    _load_scenario_dict,
)


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Round-trip a JSON scenario through its form model and write it out."
    )
    parser.add_argument("scenario", help="path to a scenario JSON file")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="write JSON to this path; omit or use '-' for stdout",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite the output file if it already exists",
    )
    parser.add_argument(
        "--strict-json",
        dest="strict_json",
        action="store_true",
        help=(
            "emit RFC 8259-compatible JSON, mapping non-finite floats to null "
            "(a valid scenario has none, so this only changes lenient edge cases)"
        ),
    )
    return parser.parse_args(argv)


def _write_output(text: str, output: str, force: bool, print_func) -> None:
    """Write ``text`` to the file ``output`` (not stdout), raising ValueError cleanly."""

    target = Path(output)
    if target.is_dir():
        raise ValueError(f"output path is a directory: {output}")
    if target.exists() and not force:
        raise ValueError(f"refusing to overwrite existing file {output}; pass --force")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not write {output}: {exc}")
    print_func(f"wrote {output}")


def roundtrip_scenario_form(
    path: str,
    output=None,
    force: bool = False,
    strict: bool = False,
    print_func=print,
) -> int:
    """Round-trip ``path`` through its form model and write/print the JSON.

    Returns ``0`` on success. Raises :class:`ValueError` (caught by :func:`main`)
    when the scenario cannot be loaded, the form has validation messages, the
    round-trip parse/build fails, or the output cannot be written -- in every case
    no output file is produced.
    """

    data = _load_scenario_dict(path)
    # Detection and from_dict raise ValueError for a non-dict / wrong-mode /
    # structurally invalid scenario.
    mode = detect_scenario_form_mode(data)
    from_dict, validate, to_dict = _MODE_HELPERS[mode]
    form = from_dict(data)

    messages = validate(form)
    if messages:
        detail = "\n".join(
            f"- [{m.severity}] {m.field}: {m.message}" for m in messages
        )
        raise ValueError(
            f"form has {len(messages)} validation message(s); not writing:\n{detail}"
        )

    # Only emit JSON once the form's to_dict output re-parses and rebuilds, so a
    # written file is always a scenario the parser and builder accept.
    try:
        round_trip_dict = to_dict(form)
        scenario = river_scenario_from_dict(round_trip_dict)
        build_river_steal_game_from_scenario(scenario)
    except Exception as exc:  # noqa: BLE001 - surface as a clean round-trip error
        raise ValueError(f"round-trip failed, not writing: {exc}")

    # Reuse report_export's strict-JSON serialiser so the non-finite handling is
    # identical to the analysis/batch/validation exporters.
    text = _dump_json(round_trip_dict, strict)

    if output is None or output == "-":
        print_func(text)
    else:
        _write_output(text, output, force, print_func)
    return 0


def main(argv) -> int:
    args = _parse_args(argv)
    try:
        return roundtrip_scenario_form(
            args.scenario,
            output=args.output,
            force=args.force,
            strict=args.strict_json,
        )
    except Exception as exc:  # noqa: BLE001 - report cleanly, never a traceback
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
