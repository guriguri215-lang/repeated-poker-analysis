#!/usr/bin/env python3
"""Run a JSON river scenario through the full candidate-analysis pipeline.

Usage:

    python scripts/run_river_scenario_analysis.py examples/scenarios/nuts_chop_steal_bet98.json

It loads and validates the scenario, builds the game, runs
``run_river_scenario_analysis`` (candidate generation, comparison, reporting, and
Markdown rendering), and prints the scenario id, the resolved horizon/discount,
the candidate counts, the Markdown summary (unless disabled), and an optional
ranking section. It uses only the package and the standard library, and it does
not write any file.

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
        "--no-markdown",
        dest="markdown",
        action="store_false",
        help="do not render the Markdown summary",
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


def main(argv) -> int:
    args = _parse_args(argv)
    config = RiverScenarioAnalysisConfig(
        horizon=args.horizon,
        discount=args.discount,
        markdown=args.markdown,
        ranking_criterion=args.rank_by,
        ranking_top_k=args.top_k,
    )
    result = run_river_scenario_analysis(args.scenario, config)

    counts = result.pipeline_result.filter_result.summary_counts
    print(f"scenario_id: {result.scenario_id}")
    print(f"horizon: {result.horizon} / discount: {result.discount}")
    print(
        f"candidates: generated={len(result.pipeline_result.generated_candidates)} "
        f"kept={counts.kept} excluded={counts.excluded}"
    )

    if result.markdown_summary is not None:
        print()
        print(result.markdown_summary)

    _print_ranking(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
