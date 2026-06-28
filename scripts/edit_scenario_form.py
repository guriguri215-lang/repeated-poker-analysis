#!/usr/bin/env python3
"""Edit a single-hand scenario through its form model and write it back out.

Usage:

    python scripts/edit_scenario_form.py <scenario.json> \
        --set scenario_id=edited --set bet_size=50 --set horizons=10,20 \
        --output reports/edited.json --force

This is a pre-GUI developer tool: the smallest "load a form, edit some fields,
validate, and save" flow, restricted to single-hand mode. It loads one
single-hand scenario JSON into a ``SingleHandScenarioForm``, applies each
``--set FIELD=VALUE`` edit, and -- only when the edited form validates cleanly and
its ``to_dict`` output re-parses and rebuilds -- emits that JSON. With
``--output PATH`` it writes the file (refusing to overwrite without ``--force``,
creating parent directories); otherwise, or with ``--output -``, it prints JSON to
stdout with no other output mixed in, so it composes with
``inspect_scenario_form.py`` / ``roundtrip_scenario_form.py``.

Only single-hand mode is supported; range / matrix / betting-tree scenarios are
rejected. It does not analyse or run the candidate pipeline and adds no new model
or analysis logic; it reuses the existing form model, parser, and game builder as
the source of truth.

Editable fields (the flat ``SingleHandScenarioForm`` fields), with dotted aliases:

    scenario_id, description, showdown
    rake_rate            (alias rake.rate)
    rake_cap             (alias rake.cap; none/null/empty -> no cap)
    initial_commitment_hero      (alias initial_commitment.hero)
    initial_commitment_villain   (alias initial_commitment.villain)
    bet_size
    baseline_call_probability    (alias baseline.call)
    baseline_fold_probability    (alias baseline.fold)
    shift_amounts        (comma-separated floats, e.g. 0.25,0.5,1.0)
    horizons             (comma-separated ints, e.g. 10,20,100)
    discount

Unknown fields, a malformed ``--set`` (no ``=``), a bad value, a non-single-hand
scenario, validation messages, a failed round-trip, or a write problem print
``error: ...`` to stderr and exit non-zero, never a traceback, and never leave a
file from a form that failed the checks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for the sibling CLIs

from repeated_poker import (  # noqa: E402  (path is set up above)
    build_river_steal_game_from_scenario,
    detect_scenario_form_mode,
    river_scenario_from_dict,
    single_hand_form_from_dict,
    single_hand_form_to_dict,
    validate_single_hand_form,
)
from repeated_poker.report_export import _dump_json  # noqa: E402

# Reuse the loader and the safe file writer from the sibling form CLIs rather than
# duplicating them (the same script-to-script reuse wizard_run_scenario uses).
from inspect_scenario_form import _load_scenario_dict  # noqa: E402
from roundtrip_scenario_form import _write_output  # noqa: E402

# Fields handled as plain strings (final validity is checked by the validator /
# parser, e.g. showdown must be hero/villain/chop).
_STR_FIELDS = frozenset({"scenario_id", "description", "showdown"})
# Fields parsed as a single float.
_FLOAT_FIELDS = frozenset(
    {
        "rake_rate",
        "initial_commitment_hero",
        "initial_commitment_villain",
        "bet_size",
        "baseline_call_probability",
        "baseline_fold_probability",
        "discount",
    }
)
# All editable flat field names (str + float + the specially-parsed ones).
_KNOWN_FIELDS = _STR_FIELDS | _FLOAT_FIELDS | {"rake_cap", "shift_amounts", "horizons"}
# Dotted aliases mapping onto the flat field names.
_ALIASES = {
    "rake.rate": "rake_rate",
    "rake.cap": "rake_cap",
    "initial_commitment.hero": "initial_commitment_hero",
    "initial_commitment.villain": "initial_commitment_villain",
    "baseline.call": "baseline_call_probability",
    "baseline.fold": "baseline_fold_probability",
}
# Values that clear rake_cap (no cap).
_NO_CAP_VALUES = frozenset({"none", "null", ""})


def _to_float(field: str, value: str) -> float:
    try:
        return float(value.strip())
    except ValueError:
        raise ValueError(f"invalid number for {field}: {value!r}")


def _to_number_list(field: str, value: str, cast):
    items = [piece.strip() for piece in value.split(",") if piece.strip() != ""]
    result = []
    for item in items:
        try:
            result.append(cast(item))
        except ValueError:
            raise ValueError(f"invalid {field} entry: {item!r}")
    return result


def _convert_field(field: str, value: str):
    """Convert a raw ``--set`` string value for ``field`` to the form's type."""

    if field in _STR_FIELDS:
        return value
    if field in _FLOAT_FIELDS:
        return _to_float(field, value)
    if field == "rake_cap":
        if value.strip().lower() in _NO_CAP_VALUES:
            return None
        return _to_float(field, value)
    if field == "shift_amounts":
        return _to_number_list(field, value, float)
    if field == "horizons":
        return _to_number_list(field, value, int)
    # Unreachable: callers check membership in _KNOWN_FIELDS first.
    raise ValueError(f"unknown field {field!r}")


def _apply_set(form, raw: str) -> None:
    """Apply one ``FIELD=VALUE`` edit to ``form`` in place, raising ValueError."""

    if "=" not in raw:
        raise ValueError(f"invalid --set {raw!r}; expected FIELD=VALUE")
    name, value = raw.split("=", 1)
    field = _ALIASES.get(name.strip(), name.strip())
    if field not in _KNOWN_FIELDS:
        raise ValueError(
            f"unknown field {name.strip()!r}; editable fields are "
            f"{sorted(_KNOWN_FIELDS)}"
        )
    setattr(form, field, _convert_field(field, value))


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Edit a single-hand scenario through its form model and write it out."
    )
    parser.add_argument("scenario", help="path to a single-hand scenario JSON file")
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="set a form field; may be given multiple times",
    )
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
        help="emit RFC 8259-compatible JSON (same serialiser as the exporters)",
    )
    return parser.parse_args(argv)


def edit_scenario_form(
    path: str,
    sets=(),
    output=None,
    force: bool = False,
    strict: bool = False,
    print_func=print,
) -> int:
    """Load, edit, validate, round-trip, and write/print a single-hand scenario.

    Returns ``0`` on success. Raises :class:`ValueError` (caught by :func:`main`)
    when the scenario is not single-hand, a ``--set`` is malformed, a field is
    unknown or has a bad value, the edited form has validation messages, the
    round-trip parse/build fails, or the output cannot be written -- in every case
    no output file is produced.
    """

    data = _load_scenario_dict(path)
    mode = detect_scenario_form_mode(data)
    if mode != "single-hand":
        raise ValueError(
            f"edit CLI supports single-hand mode only; this scenario is {mode} mode"
        )
    # single_hand_form_from_dict re-checks the mode and applies the parser's
    # structural validation; mode was checked above only for a clearer message.
    form = single_hand_form_from_dict(data)

    for raw in sets:
        _apply_set(form, raw)

    messages = validate_single_hand_form(form)
    if messages:
        detail = "\n".join(
            f"- [{m.severity}] {m.field}: {m.message}" for m in messages
        )
        raise ValueError(
            f"edited form has {len(messages)} validation message(s); not writing:\n{detail}"
        )

    # Only emit JSON once the edited form's to_dict re-parses and rebuilds, so a
    # written file is always a scenario the parser and builder accept.
    try:
        out_dict = single_hand_form_to_dict(form)
        scenario = river_scenario_from_dict(out_dict)
        build_river_steal_game_from_scenario(scenario)
    except Exception as exc:  # noqa: BLE001 - surface as a clean round-trip error
        raise ValueError(f"edited form failed round-trip, not writing: {exc}")

    text = _dump_json(out_dict, strict)

    if output is None or output == "-":
        print_func(text)
    else:
        _write_output(text, output, force, print_func)
    return 0


def main(argv) -> int:
    args = _parse_args(argv)
    try:
        return edit_scenario_form(
            args.scenario,
            sets=args.sets,
            output=args.output,
            force=args.force,
            strict=args.strict_json,
        )
    except Exception as exc:  # noqa: BLE001 - report cleanly, never a traceback
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
