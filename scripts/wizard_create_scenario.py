#!/usr/bin/env python3
"""Interactive wizard that builds a starter river-scenario JSON file.

This is a thin, dependency-light layer over
:func:`repeated_poker.create_scenario_template`: it starts from a template for
the chosen kind and asks a few questions to fill in the common top-level fields
(scenario id, description, rake, initial commitment, bet size, repeated horizons
and discount, and the output path). It does not edit range buckets or matrices;
those keep the template's abstract toy values, and the wizard prints a reminder
to edit the JSON by hand if a real range/matrix is needed.

Usage:

    python scripts/wizard_create_scenario.py
    python scripts/wizard_create_scenario.py --kind single-hand --output reports/my_scenario.json
    python scripts/wizard_create_scenario.py --kind range-matrix-equity --non-interactive --output reports/x.json

Anything supplied on the command line (``--kind`` / ``--scenario-id`` /
``--description`` / ``--output``) is used as-is and not asked for. By default the
generated scenario is validated at the parser/build level before it is written;
pass ``--no-validate`` to skip that. An existing output file is not overwritten
unless ``--force`` is given (or the overwrite prompt is confirmed). Generated
scenarios are abstract toy examples, not strategic recommendations, and always
include ``"format_version": "1"``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repeated_poker import (  # noqa: E402  (path is set up above)
    available_scenario_template_kinds,
    build_river_steal_game_from_scenario,
    create_scenario_template,
    river_scenario_from_dict,
)

_TOY_VALUES_NOTE = (
    "Note: range buckets / matrices are still starter toy values; "
    "edit the JSON manually if you need a real range or matrix."
)


class _WizardError(Exception):
    """A clean, user-facing error (reported without a traceback)."""


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Interactively build a starter river-scenario JSON file."
    )
    parser.add_argument(
        "--kind",
        choices=available_scenario_template_kinds(),
        help="template kind to start from (asked interactively if omitted)",
    )
    parser.add_argument("--scenario-id", dest="scenario_id", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument(
        "--output",
        default=None,
        help="write the scenario to this path (asked interactively if omitted)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite the --output file if it already exists",
    )
    parser.add_argument(
        "--non-interactive",
        dest="non_interactive",
        action="store_true",
        help="ask nothing; use template defaults plus any provided flags",
    )
    parser.add_argument(
        "--validate",
        dest="validate",
        action="store_true",
        default=True,
        help="validate the scenario at the parser/build level (default)",
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="skip the parser/build validation",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Interactive input helpers (only used when interactive)
# ---------------------------------------------------------------------------


def _ask_text(input_func, label, default):
    raw = input_func(f"{label} [{default}]: ").strip()
    return raw or default


def _ask_float(input_func, print_func, label, default, *, predicate=None, message=""):
    while True:
        raw = input_func(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            print_func(f"  please enter a number (got {raw!r})")
            continue
        if predicate is not None and not predicate(value):
            print_func(f"  {message}")
            continue
        return value


def _ask_cap(input_func, print_func, default):
    # Empty keeps the current value; "none"/"null" clears the cap to null.
    while True:
        raw = input_func(f"rake cap (number, or 'none' for no cap) [{default}]: ").strip()
        if not raw:
            return default
        if raw.lower() in ("none", "null"):
            return None
        try:
            value = float(raw)
        except ValueError:
            print_func(f"  please enter a number or 'none' (got {raw!r})")
            continue
        if value < 0:
            print_func("  rake cap must be non-negative")
            continue
        return value


def _ask_horizons(input_func, print_func, default):
    default_text = ",".join(str(h) for h in default)
    while True:
        raw = input_func(f"repeated horizons (comma-separated) [{default_text}]: ").strip()
        if not raw:
            return list(default)
        try:
            horizons = _parse_horizons(raw)
        except ValueError as exc:
            print_func(f"  {exc}")
            continue
        return horizons


def _ask_yes_no(input_func, label):
    raw = input_func(f"{label} [y/N]: ").strip().lower()
    return raw in ("y", "yes")


# ---------------------------------------------------------------------------
# Pure parsing / building helpers
# ---------------------------------------------------------------------------


def _parse_horizons(text):
    horizons = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            raise ValueError(f"horizons must be integers (got {part!r})")
        if value < 1:
            raise ValueError(f"each horizon must be at least 1 (got {value})")
        horizons.append(value)
    if not horizons:
        raise ValueError("at least one horizon is required")
    return horizons


def _resolve_kind(args, interactive, input_func, print_func):
    if args.kind is not None:
        return args.kind
    if not interactive:
        raise _WizardError("--kind is required in non-interactive mode")
    kinds = available_scenario_template_kinds()
    print_func("Available template kinds:")
    for kind in kinds:
        print_func(f"  - {kind}")
    while True:
        raw = input_func(f"kind [{kinds[0]}]: ").strip()
        choice = raw or kinds[0]
        if choice in kinds:
            return choice
        print_func(f"  unknown kind {choice!r}; choose one of {kinds}")


def _build_template(args, interactive, input_func, print_func):
    kind = _resolve_kind(args, interactive, input_func, print_func)

    base = create_scenario_template(kind)
    default_id = base["scenario_id"]
    if args.scenario_id is not None:
        scenario_id = args.scenario_id
    elif interactive:
        scenario_id = _ask_text(input_func, "scenario_id", default_id)
    else:
        scenario_id = default_id

    try:
        template = create_scenario_template(kind, scenario_id=scenario_id)
    except ValueError as exc:
        raise _WizardError(str(exc))

    if args.description is not None:
        template["description"] = args.description
    elif interactive:
        template["description"] = _ask_text(
            input_func, "description", template["description"]
        )

    if interactive:
        template["rake"]["rate"] = _ask_float(
            input_func, print_func, "rake rate", template["rake"]["rate"],
            predicate=lambda v: 0.0 <= v <= 1.0,
            message="rake rate must be within [0, 1]",
        )
        template["rake"]["cap"] = _ask_cap(input_func, print_func, template["rake"]["cap"])
        template["initial_commitment"]["hero"] = _ask_float(
            input_func, print_func, "initial commitment hero",
            template["initial_commitment"]["hero"],
            predicate=lambda v: v >= 0.0, message="must be non-negative",
        )
        template["initial_commitment"]["villain"] = _ask_float(
            input_func, print_func, "initial commitment villain",
            template["initial_commitment"]["villain"],
            predicate=lambda v: v >= 0.0, message="must be non-negative",
        )
        if "bet_size" in template:
            template["bet_size"] = _ask_float(
                input_func, print_func, "bet size", template["bet_size"],
                predicate=lambda v: v > 0.0, message="bet size must be positive",
            )
        template["repeated"]["horizons"] = _ask_horizons(
            input_func, print_func, template["repeated"]["horizons"]
        )
        template["repeated"]["discount"] = _ask_float(
            input_func, print_func, "discount", template["repeated"]["discount"],
            predicate=lambda v: 0.0 < v <= 1.0,
            message="discount must satisfy 0 < discount <= 1",
        )

    return template


def _resolve_output(args, interactive, input_func):
    if args.output is not None:
        return args.output
    if not interactive:
        return None
    raw = input_func("output path (empty to print to stdout): ").strip()
    return raw or None


def _write_output(template, output, args, interactive, input_func, print_func):
    text = json.dumps(template, indent=2, ensure_ascii=False)
    if output is None:
        print_func(text)
        print_func(_TOY_VALUES_NOTE)
        return 0

    target = Path(output)
    if target.exists() and not args.force:
        if interactive:
            if not _ask_yes_no(input_func, f"{output} exists; overwrite?"):
                raise _WizardError(f"not overwriting existing file {output}")
        else:
            raise _WizardError(
                f"{output} already exists; pass --force to overwrite"
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text + "\n", encoding="utf-8")
    print_func(f"saved scenario to {output}")
    print_func(_TOY_VALUES_NOTE)
    return 0


def main(argv=None, input_func=input, print_func=print) -> int:
    args = _parse_args(argv)
    interactive = not args.non_interactive

    try:
        template = _build_template(args, interactive, input_func, print_func)

        if args.validate:
            scenario = river_scenario_from_dict(template)
            build_river_steal_game_from_scenario(scenario)

        output = _resolve_output(args, interactive, input_func)
        return _write_output(template, output, args, interactive, input_func, print_func)
    except _WizardError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (ValueError, KeyError) as exc:
        # Validation or build problems are reported cleanly, without a traceback.
        print(f"error: generated scenario failed validation: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
