#!/usr/bin/env python3
"""Guided end-to-end scenario workflow: create or pick a scenario, then analyse it.

This ties the existing pieces together into the path a user actually walks:

1. *create-and-run mode* (``--kind``, or interactively): build a starter scenario
   from a template (reusing the interactive scenario wizard's helpers), save it,
   then analyse it; or
2. *existing-file mode* (``--scenario PATH``): validate and analyse an existing
   scenario file without modifying it.

In both modes it validates at the parser/build level, runs
``run_river_scenario_analysis``, prints a short summary, and can save the
analysis as Markdown / JSON via the existing ``report_export`` writers. It adds
no new solver or game-theory model; it only sequences existing functionality.

Examples:

    python scripts/wizard_run_scenario.py --scenario examples/scenarios/nuts_chop_steal_bet98.json
    python scripts/wizard_run_scenario.py --kind single-hand --scenario-output reports/my.json --non-interactive
    python scripts/wizard_run_scenario.py --scenario examples/scenarios/nuts_chop_steal_bet98.json \
        --output-json reports/result.json --strict-json --output-markdown reports/result.md

Errors (validation, analysis, or output problems) are reported as a short
``error: ...`` line with a non-zero exit, never a traceback.
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for the wizard helpers

from repeated_poker import (  # noqa: E402  (path is set up above)
    RiverScenarioAnalysisConfig,
    available_scenario_template_kinds,
    build_river_steal_game_from_scenario,
    load_river_scenario_json,
    river_scenario_from_dict,
    run_river_scenario_analysis,
    write_analysis_json,
    write_analysis_markdown,
)

# Reuse the interactive wizard's building blocks rather than re-implementing the
# prompts or the hardened file-output handling.
from wizard_create_scenario import (  # noqa: E402
    _TOY_VALUES_NOTE,
    _WizardError,
    _build_template,
    _write_output,
)


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Create or pick a scenario, then validate, analyse, and export it."
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="existing scenario JSON to analyse (existing-file mode)",
    )
    parser.add_argument(
        "--kind",
        choices=available_scenario_template_kinds(),
        help="template kind to create and analyse (create-and-run mode)",
    )
    parser.add_argument(
        "--scenario-output",
        dest="scenario_output",
        default=None,
        help="where to save the created scenario JSON (create-and-run mode)",
    )
    parser.add_argument("--scenario-id", dest="scenario_id", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite the --scenario-output file if it already exists",
    )
    parser.add_argument(
        "--non-interactive",
        dest="non_interactive",
        action="store_true",
        help="ask nothing; use template defaults plus any provided flags",
    )
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--discount", type=float, default=None)
    parser.add_argument("--output-json", dest="output_json", default=None)
    parser.add_argument("--output-markdown", dest="output_markdown", default=None)
    parser.add_argument(
        "--strict-json",
        dest="strict_json",
        action="store_true",
        help="emit RFC 8259-compatible JSON (applies to --output-json)",
    )
    return parser.parse_args(argv)


def _resolve_scenario_output(args, interactive, input_func):
    if args.scenario_output is not None:
        if not args.scenario_output.strip():
            raise _WizardError("--scenario-output must not be empty")
        return args.scenario_output
    if not interactive:
        raise _WizardError(
            "--scenario-output is required in non-interactive create-and-run mode"
        )
    raw = input_func("scenario output path: ").strip()
    if not raw:
        raise _WizardError("a scenario output path is required to create a scenario")
    return raw


def _create_scenario(args, interactive, input_func, print_func):
    """Build a scenario from a template, save it, and return (scenario, path, created)."""

    create_args = Namespace(
        kind=args.kind,
        scenario_id=args.scenario_id,
        description=args.description,
        force=args.force,
    )
    template = _build_template(create_args, interactive, input_func, print_func)
    scenario_output = _resolve_scenario_output(args, interactive, input_func)
    # Validate (parser/build) before writing, like the create wizard: a template
    # that fails validation must not leave a scenario file behind.
    scenario = river_scenario_from_dict(template)
    build_river_steal_game_from_scenario(scenario)
    # ``_write_output`` validates the path (empty / directory / OSError) and
    # honours --force, printing "saved scenario to ..." and the toy-values note.
    _write_output(template, scenario_output, create_args, interactive, input_func, print_func)
    return scenario, scenario_output, True


def _load_existing_scenario(path):
    try:
        scenario = load_river_scenario_json(path)
    except FileNotFoundError:
        raise _WizardError(f"scenario file not found: {path}")
    except OSError as exc:
        raise _WizardError(f"could not read scenario file {path}: {exc.strerror or exc}")
    return scenario


def _run_workflow(args, input_func, print_func) -> int:
    interactive = not args.non_interactive

    if args.scenario is not None and args.kind is not None:
        raise _WizardError("--scenario and --kind cannot be used together")
    if args.scenario is not None:
        # These flags only configure create-and-run mode; with an existing
        # scenario they would be silently ignored, so reject them explicitly.
        create_only = [
            ("--scenario-output", args.scenario_output is not None),
            ("--scenario-id", args.scenario_id is not None),
            ("--description", args.description is not None),
            ("--force", args.force),
        ]
        offending = [name for name, present in create_only if present]
        if offending:
            raise _WizardError(
                f"{', '.join(offending)} only apply to create-and-run mode, "
                "not --scenario"
            )

    if args.scenario is not None:
        scenario = _load_existing_scenario(args.scenario)
        scenario_path = args.scenario
        created = False
    elif args.kind is not None or interactive:
        scenario, scenario_path, created = _create_scenario(
            args, interactive, input_func, print_func
        )
    else:
        raise _WizardError(
            "either --scenario or --kind is required in non-interactive mode"
        )

    # Parser/build validation (a clean, separate step before analysis).
    build_river_steal_game_from_scenario(scenario)

    need_markdown = args.output_markdown is not None
    config = RiverScenarioAnalysisConfig(
        horizon=args.horizon,
        discount=args.discount,
        markdown=need_markdown,
    )
    result = run_river_scenario_analysis(scenario, config)

    counts = result.pipeline_result.filter_result.summary_counts
    print_func(f"scenario_id: {result.scenario_id}")
    print_func(f"scenario: {scenario_path}")
    print_func("validation: ok")
    print_func(f"horizon: {result.horizon} / discount: {result.discount}")
    print_func(
        f"candidates: generated={len(result.pipeline_result.generated_candidates)} "
        f"kept={counts.kept} excluded={counts.excluded}"
    )

    if args.output_json is not None:
        write_analysis_json(result, args.output_json, strict=args.strict_json)
        print_func(f"saved JSON to {args.output_json}")
    if args.output_markdown is not None:
        write_analysis_markdown(result, args.output_markdown)
        print_func(f"saved Markdown to {args.output_markdown}")

    if created:
        # Make the starter-values caveat visible at the end of the run too.
        print_func(_TOY_VALUES_NOTE)
    return 0


def main(argv=None, input_func=input, print_func=print) -> int:
    args = _parse_args(argv)
    try:
        return _run_workflow(args, input_func, print_func)
    except _WizardError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {exc.strerror or exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
