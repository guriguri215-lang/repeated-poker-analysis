#!/usr/bin/env python3
"""Run a JSON river scenario through the full candidate-analysis pipeline.

Usage:

    python scripts/run_river_scenario_analysis.py examples/scenarios/nuts_chop_steal_bet98.json

It loads and validates the scenario, builds the game, runs
``run_river_scenario_analysis`` (candidate generation, comparison, reporting, and
Markdown rendering), and prints the scenario id, the resolved horizon/discount,
the candidate counts, the Markdown summary (unless disabled), and an optional
ranking section.

By default it writes no file. The optional ``--output-json``, ``--output-markdown``,
and ``--output-csv`` flags save the result to the given paths, creating missing
parent directories and overwriting existing files. ``--output-markdown`` forces
Markdown generation even with ``--no-markdown`` (``--no-markdown`` then only
suppresses the stdout summary, not the saved file). It uses only the package and
the standard library, and it does no network work. Saved outputs embed a run
manifest (scenario SHA-256, package version, best-effort local git commit or
null, UTC timestamp, and effective parameters) for reproducibility.

For a quick terminal-EV / baseline / ``T_deadline`` sanity check of a scenario
without the full pipeline, use ``scripts/run_river_scenario.py`` instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repeated_poker import (  # noqa: E402  (path is set up above)
    RiverScenarioAnalysisConfig,
    run_river_scenario_analysis,
    write_analysis_csv,
    write_analysis_json,
    write_analysis_markdown,
)


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Run a JSON river scenario through the candidate-analysis pipeline."
    )
    parser.add_argument("scenario", help="path to a JSON river scenario file")
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="repeated-game horizon (default: max of the scenario's horizons)",
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=None,
        help="repeated-game discount (default: the scenario's discount)",
    )
    parser.add_argument(
        "--rank-by",
        dest="rank_by",
        default=None,
        help="rank the report rows by this criterion (for example t_deadline)",
    )
    parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=None,
        help="limit the ranking section to the top K rows",
    )
    parser.add_argument(
        "--detection-log-likelihood-threshold",
        dest="detection_log_likelihood_threshold",
        type=float,
        default=None,
        help="enable T_detect with this log-likelihood threshold in nats",
    )
    parser.add_argument(
        "--detection-occurrence-probability-per-opportunity",
        dest="detection_occurrence_probability_per_opportunity",
        type=float,
        default=None,
        help="local_v0 only: convert required observations to opportunities",
    )
    parser.add_argument(
        "--detection-comparable-spot-occurrence-probability-per-physical-hand",
        dest="detection_comparable_spot_occurrence_probability_per_physical_hand",
        type=float,
        default=None,
        help=(
            "diagnostic only: convert comparable opportunities to physical "
            "dealt hands using a supplied spot frequency"
        ),
    )
    parser.add_argument(
        "--detection-method",
        dest="detection_method",
        default="local_v0",
        choices=("local_v0", "reach_weighted_v1"),
        help="T_detect method (default: local_v0)",
    )
    parser.add_argument(
        "--detection-observation-model",
        dest="detection_observation_model",
        default=None,
        choices=("actions_only", "showdown_reveal"),
        help="reach_weighted_v1 observation model (default: actions_only)",
    )
    parser.add_argument(
        "--max-detection-terminals",
        dest="max_detection_terminals",
        type=int,
        default=100000,
        help="safety limit for reach_weighted_v1 terminal enumeration",
    )
    parser.add_argument(
        "--no-markdown",
        dest="markdown",
        action="store_false",
        help="do not print the Markdown summary to stdout",
    )
    parser.add_argument(
        "--output-json",
        dest="output_json",
        default=None,
        help="save the full analysis result as JSON to this path",
    )
    parser.add_argument(
        "--output-markdown",
        dest="output_markdown",
        default=None,
        help="save the Markdown summary to this path (forces Markdown generation)",
    )
    parser.add_argument(
        "--output-csv",
        dest="output_csv",
        default=None,
        help="save one CSV row per candidate to this path",
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


def _print_ranking(result) -> None:
    ranking = result.ranking_result
    if ranking is None:
        return
    print()
    print(f"## Ranking by {ranking.criterion} (descending={ranking.descending})")
    for ranked in ranking.ranked_rows:
        print(
            f"  {ranked.rank}. {ranked.row.candidate_id} "
            f"({ranking.criterion}={ranked.sort_key})"
        )


def _write_outputs(result, args) -> None:
    if args.output_json is not None:
        write_analysis_json(result, args.output_json, strict=args.strict_json)
        print(f"saved JSON to {args.output_json}")
    if args.output_markdown is not None:
        write_analysis_markdown(result, args.output_markdown)
        print(f"saved Markdown to {args.output_markdown}")
    if args.output_csv is not None:
        write_analysis_csv(result, args.output_csv)
        print(f"saved CSV to {args.output_csv}")


def main(argv) -> int:
    args = _parse_args(argv)
    # ``--output-markdown`` needs a Markdown summary even with ``--no-markdown``,
    # which then only suppresses the stdout summary, not the saved file.
    generate_markdown = args.markdown or args.output_markdown is not None
    config = RiverScenarioAnalysisConfig(
        horizon=args.horizon,
        discount=args.discount,
        markdown=generate_markdown,
        ranking_criterion=args.rank_by,
        ranking_top_k=args.top_k,
        detection_log_likelihood_threshold=args.detection_log_likelihood_threshold,
        detection_occurrence_probability_per_opportunity=(
            args.detection_occurrence_probability_per_opportunity
        ),
        detection_comparable_spot_occurrence_probability_per_physical_hand=(
            args.detection_comparable_spot_occurrence_probability_per_physical_hand
        ),
        detection_method=args.detection_method,
        detection_observation_model=args.detection_observation_model,
        max_detection_terminals=args.max_detection_terminals,
    )
    result = run_river_scenario_analysis(args.scenario, config)

    counts = result.pipeline_result.filter_result.summary_counts
    print(f"scenario_id: {result.scenario_id}")
    print(f"horizon: {result.horizon} / discount: {result.discount}")
    print(
        f"candidates: generated={len(result.pipeline_result.generated_candidates)} "
        f"kept={counts.kept} excluded={counts.excluded}"
    )

    if args.markdown and result.markdown_summary is not None:
        print()
        print(result.markdown_summary)

    _print_ranking(result)
    _write_outputs(result, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
