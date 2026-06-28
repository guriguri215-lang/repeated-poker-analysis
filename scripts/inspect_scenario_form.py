#!/usr/bin/env python3
"""Inspect a JSON scenario through its form model (developer utility).

Usage:

    python scripts/inspect_scenario_form.py examples/scenarios/range_matrix_steal_bet98.json

This is a pre-GUI developer tool. It loads one scenario JSON, detects which of the
five form-model modes it is, runs that mode's ``*_form_from_dict`` /
``validate_*_form`` / ``*_form_to_dict`` helpers, and then re-parses and rebuilds
the round-tripped dict, printing a short report to stdout::

    Scenario form inspection
    scenario_id: range_matrix_steal_bet98
    mode: showdown-matrix
    form: ShowdownMatrixScenarioForm
    validation: ok
    round_trip_parse: ok
    round_trip_build: ok

If the form has field-level validation messages they are listed::

    validation: 2 message(s)
    - [error] hero_buckets[0].hand_id: hand_id must be a non-empty string

It does not edit, analyse, or run the candidate pipeline; for that use
``scripts/run_river_scenario_analysis.py``. It uses the existing form model and
parser as the source of truth and adds no new model or analysis logic.

Exit code: ``0`` when the form loads, validation is clean, and the round-trip
re-parses and rebuilds; ``1`` when validation reports messages or a round-trip
step fails. A scenario that cannot be loaded at all (missing file, invalid JSON,
or a scenario the parser rejects) prints ``error: ...`` to stderr and exits ``1``
with no traceback.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repeated_poker import (  # noqa: E402  (path is set up above)
    betting_tree_form_from_dict,
    betting_tree_form_to_dict,
    build_river_steal_game_from_scenario,
    detect_scenario_form_mode,
    equity_matrix_form_from_dict,
    equity_matrix_form_to_dict,
    hero_range_form_from_dict,
    hero_range_form_to_dict,
    river_scenario_from_dict,
    showdown_matrix_form_from_dict,
    showdown_matrix_form_to_dict,
    single_hand_form_from_dict,
    single_hand_form_to_dict,
    validate_betting_tree_form,
    validate_equity_matrix_form,
    validate_hero_range_form,
    validate_showdown_matrix_form,
    validate_single_hand_form,
)

# Per mode: the (from_dict, validate, to_dict) helpers. Detection of the mode is
# shared with a future GUI loader via repeated_poker.detect_scenario_form_mode.
_MODE_HELPERS = {
    "single-hand": (
        single_hand_form_from_dict,
        validate_single_hand_form,
        single_hand_form_to_dict,
    ),
    "hero-range": (
        hero_range_form_from_dict,
        validate_hero_range_form,
        hero_range_form_to_dict,
    ),
    "showdown-matrix": (
        showdown_matrix_form_from_dict,
        validate_showdown_matrix_form,
        showdown_matrix_form_to_dict,
    ),
    "equity-matrix": (
        equity_matrix_form_from_dict,
        validate_equity_matrix_form,
        equity_matrix_form_to_dict,
    ),
    "betting-tree": (
        betting_tree_form_from_dict,
        validate_betting_tree_form,
        betting_tree_form_to_dict,
    ),
}


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Inspect a JSON scenario through its form model."
    )
    parser.add_argument("scenario", help="path to a scenario JSON file")
    return parser.parse_args(argv)


def _load_scenario_dict(path: str) -> dict:
    """Read and JSON-decode ``path``, raising ValueError with a short message."""

    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ValueError(f"file not found: {path}")
    except OSError as exc:
        raise ValueError(f"could not read {path}: {exc}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}")
    return data


def inspect_scenario_form(path: str, print_func=print) -> int:
    """Print the form-model inspection report for ``path`` and return an exit code.

    Returns ``0`` when the form loads, validation is clean, and the round-trip
    re-parses and rebuilds; ``1`` otherwise. Raises :class:`ValueError` (caught by
    :func:`main`) when the scenario cannot be loaded into a form at all.
    """

    data = _load_scenario_dict(path)
    # Both detection and from_dict raise ValueError for a non-dict / wrong-mode /
    # structurally invalid scenario; main turns that into a clean error exit.
    mode = detect_scenario_form_mode(data)
    from_dict, validate, to_dict = _MODE_HELPERS[mode]
    form = from_dict(data)

    print_func("Scenario form inspection")
    print_func(f"scenario_id: {form.scenario_id}")
    print_func(f"mode: {mode}")
    print_func(f"form: {type(form).__name__}")

    ok = True

    messages = validate(form)
    if messages:
        ok = False
        print_func(f"validation: {len(messages)} message(s)")
        for message in messages:
            print_func(f"- [{message.severity}] {message.field}: {message.message}")
    else:
        print_func("validation: ok")

    # Round-trip: the form's to_dict output must re-parse and rebuild. to_dict can
    # itself raise on an invalid form (for example an unknown betting-tree
    # matrix_type), so guard it as part of the parse step.
    scenario = None
    try:
        round_trip_dict = to_dict(form)
        scenario = river_scenario_from_dict(round_trip_dict)
        print_func("round_trip_parse: ok")
    except Exception as exc:  # noqa: BLE001 - report as a failed round-trip line
        ok = False
        print_func(f"round_trip_parse: failed: {exc}")

    if scenario is None:
        print_func("round_trip_build: skipped (parse failed)")
        ok = False
    else:
        try:
            build_river_steal_game_from_scenario(scenario)
            print_func("round_trip_build: ok")
        except Exception as exc:  # noqa: BLE001 - report as a failed round-trip line
            ok = False
            print_func(f"round_trip_build: failed: {exc}")

    return 0 if ok else 1


def main(argv) -> int:
    args = _parse_args(argv)
    try:
        return inspect_scenario_form(args.scenario)
    except Exception as exc:  # noqa: BLE001 - report cleanly, never a traceback
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
