#!/usr/bin/env python3
"""Run several river scenarios through the analysis pipeline and compare them.

Usage:

    python scripts/run_scenario_batch.py examples/scenarios
    python scripts/run_scenario_batch.py a.json b.json --rank-by t_deadline

It expands the inputs (a directory's ``*.json`` in filename order, or the given
files in order), runs the existing single-scenario analysis on each, prints a
comparison summary to stdout, and optionally saves JSON / CSV / Markdown reports.
It adds no new analysis maths; it is a batch wrapper over
``run_river_scenario_analysis``.

For a single scenario with its full Markdown summary, use
``scripts/run_river_scenario_analysis.py`` instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repeated_poker import (  # noqa: E402  (path is set up above)
    BatchScenarioAnalysisConfig,
    RiverScenarioAnalysisConfig,
    run_batch_scenario_analysis,
    write_batch_csv,
    write_batch_json,
    write_batch_markdown,
)

_STDOUT_COLUMNS = [
    "scenario_id",
    "model_kind",
    "horizon",
    "generated_candidates",
    "kept_candidates",
    "eligible_candidates",
    "top_candidate_id",
    "top_candidate_t_deadline",
    "error",
]


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Run several river scenarios through the analysis pipeline."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="scenario JSON files, or a directory whose *.json files are read",
    )
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--discount", type=float, default=None)
    parser.add_argument("--rank-by", dest="rank_by", default=None)
    parser.add_argument("--top-k", dest="top_k", type=int, default=None)
    parser.add_argument(
        "--continue-on-error",
        dest="continue_on_error",
        action="store_true",
        help="record failing scenarios in the summary instead of stopping",
    )
    parser.add_argument(
        "--no-markdown",
        dest="markdown",
        action="store_false",
        help="skip per-scenario Markdown generation",
    )
    parser.add_argument("--output-json", dest="output_json", default=None)
    parser.add_argument("--output-csv", dest="output_csv", default=None)
    parser.add_argument("--output-markdown", dest="output_markdown", default=None)
    return parser.parse_args(argv)


def _print_summary(batch) -> None:
    print(
        f"scenarios: {len(batch.rows)} "
        f"(ok: {batch.ok_count}, errors: {batch.error_count})"
    )
    print(" | ".join(_STDOUT_COLUMNS))
    for row in batch.rows:
        row_dict = row.to_dict()
        cells = [
            "" if row_dict.get(column) is None else str(row_dict.get(column))
            for column in _STDOUT_COLUMNS
        ]
        print(" | ".join(cells))


def _write_outputs(batch, args) -> None:
    if args.output_json is not None:
        write_batch_json(batch, args.output_json)
        print(f"saved JSON to {args.output_json}")
    if args.output_csv is not None:
        write_batch_csv(batch, args.output_csv)
        print(f"saved CSV to {args.output_csv}")
    if args.output_markdown is not None:
        write_batch_markdown(batch, args.output_markdown)
        print(f"saved Markdown to {args.output_markdown}")


def main(argv) -> int:
    args = _parse_args(argv)
    analysis = RiverScenarioAnalysisConfig(
        horizon=args.horizon,
        discount=args.discount,
        markdown=args.markdown,
        ranking_criterion=args.rank_by,
        ranking_top_k=args.top_k,
    )
    config = BatchScenarioAnalysisConfig(
        analysis=analysis, continue_on_error=args.continue_on_error
    )
    try:
        batch = run_batch_scenario_analysis(args.inputs, config)
    except Exception as exc:  # noqa: BLE001 - report fail-fast errors cleanly
        # Without --continue-on-error a failing scenario aborts the batch; print a
        # short message to stderr and exit non-zero rather than dumping a
        # traceback. --continue-on-error never reaches here.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_summary(batch)
    _write_outputs(batch, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
