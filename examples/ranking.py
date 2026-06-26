"""Worked example: rank analysis report rows by diagnostic criteria.

Runs the full pipeline on the nuts-chop tree, then sorts the resulting analysis
rows by a few criteria and prints the top entries. This is a *diagnostic*
ranking only: it does not auto-select a candidate and makes no optimality claim.
It does not write any file.
"""

from __future__ import annotations

from candidate_library import baseline_villain_strategy
from nuts_chop_river import build_nuts_chop_river, default_hero_strategy

from repeated_poker import (
    RANK_BY_FIXED_VILLAIN_EV,
    RANK_BY_POST_RESPONSE_HERO_EV_WORST_DIFF,
    RANK_BY_T_DETECT,
    CandidateGenerationConfig,
    rank_candidate_rows,
    run_candidate_analysis_pipeline,
)


def _print_ranking(report, criterion, top_k) -> None:
    result = rank_candidate_rows(report, criterion, top_k=top_k)
    print(f"\n# Ranked by {result.criterion} (descending={result.descending}):")
    for ranked in result.ranked_rows:
        print(
            f"  {ranked.rank}. {ranked.row.candidate_id} "
            f"-> {ranked.sort_key}"
        )


def main() -> None:
    tree = build_nuts_chop_river()
    baseline_hero = default_hero_strategy()
    baseline_villain = baseline_villain_strategy()

    result = run_candidate_analysis_pipeline(
        tree,
        baseline_hero,
        baseline_villain,
        generation=CandidateGenerationConfig(shift_amounts=[0.1, 0.2]),
        horizon=5,
        detection_log_likelihood_threshold=3.0,
        detection_occurrence_probability_per_opportunity=0.5,
        render_markdown=False,
    )
    report = result.analysis_report

    print("# Diagnostic ranking only (no automatic selection, no optimality claim).")
    _print_ranking(report, RANK_BY_POST_RESPONSE_HERO_EV_WORST_DIFF, top_k=3)
    _print_ranking(report, RANK_BY_FIXED_VILLAIN_EV, top_k=3)
    _print_ranking(report, RANK_BY_T_DETECT, top_k=3)


if __name__ == "__main__":
    main()
