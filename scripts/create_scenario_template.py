#!/usr/bin/env python3
"""Generate a starter river-scenario JSON template.

Usage:

    python scripts/create_scenario_template.py --list-kinds
    python scripts/create_scenario_template.py --kind single-hand
    python scripts/create_scenario_template.py --kind range-matrix-equity-betting-tree --output reports/template.json

It prints a minimal, well-formed scenario JSON to stdout (or writes it with
``--output``) so a user can start from a working file instead of an empty one.
The templates are abstract toy examples, not strategic recommendations; every
template includes ``"format_version": "1"`` and follows
``docs/scenario_format_reference.md``.

By default the generated scenario is validated at the parser/build level
(``river_scenario_from_dict`` then ``build_river_steal_game_from_scenario``); pass
``--no-validate`` to skip it. With ``--output`` an existing file is not
overwritten unless ``--force`` is given. After generating, edit the file and
re-check it with ``python scripts/validate_river_scenario.py <file>``.
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


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Generate a starter river-scenario JSON template."
    )
    parser.add_argument(
        "--kind",
        choices=available_scenario_template_kinds(),
        help="which template to generate (see --list-kinds)",
    )
    parser.add_argument(
        "--scenario-id",
        dest="scenario_id",
        default=None,
        help="override the generated scenario_id (default depends on --kind)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="write the template to this path instead of stdout",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite the --output file if it already exists",
    )
    parser.add_argument(
        "--list-kinds",
        dest="list_kinds",
        action="store_true",
        help="list the available template kinds and exit",
    )
    parser.add_argument(
        "--validate",
        dest="validate",
        action="store_true",
        default=True,
        help="validate the generated template at the parser/build level (default)",
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="skip the parser/build validation of the generated template",
    )
    return parser.parse_args(argv)


def _validate_template(template: dict) -> None:
    """Run the parser/build validation, raising on the first problem."""

    scenario = river_scenario_from_dict(template)
    build_river_steal_game_from_scenario(scenario)


def main(argv) -> int:
    args = _parse_args(argv)

    if args.list_kinds:
        for kind in available_scenario_template_kinds():
            print(kind)
        return 0

    if args.kind is None:
        print("error: --kind is required (or use --list-kinds)", file=sys.stderr)
        return 2

    try:
        template = create_scenario_template(args.kind, args.scenario_id)
    except ValueError as exc:
        # For example an empty or non-string --scenario-id; report cleanly rather
        # than dumping a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.validate:
        try:
            _validate_template(template)
        except Exception as exc:  # noqa: BLE001 - report cleanly, no traceback
            print(f"error: generated template failed validation: {exc}", file=sys.stderr)
            return 1

    text = json.dumps(template, indent=2, ensure_ascii=False)

    if args.output is None:
        print(text)
        return 0

    target = Path(args.output)
    if target.exists() and not args.force:
        print(
            f"error: {args.output} already exists; pass --force to overwrite",
            file=sys.stderr,
        )
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text + "\n", encoding="utf-8")
    print(f"saved template to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
