"""Worked example: run the full candidate-analysis pipeline.

Reuses the nuts-chop tree and the baseline mixed Villain strategy, runs
``run_candidate_analysis_pipeline`` with a pre-filter and detection enabled, and
prints the filter summary counts and the Markdown summary to stdout. It does not
write any file.
"""

from __future__ import annotations

from candidate_library import baseline_villain_strategy
from nuts_chop_river import build_nuts_chop_river, default_hero_strategy

from repeated_poker import (
    CandidateFilterConfig,
    CandidateGenerationConfig,
    run_candidate_analysis_pipeline,
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
        profit_tolerance=-2.0,
        max_selection_l1_distance=0.3,
        detection_log_likelihood_threshold=3.0,
        detection_occurrence_probability_per_opportunity=0.5,
        filtering=CandidateFilterConfig(max_l1_distance=0.3, min_required_observations=5),
        markdown_max_rows=10,
    )

    counts = result.filter_result.summary_counts
    print(
        f"generated={len(result.generated_candidates)} "
        f"kept={counts.kept} excluded={counts.excluded}"
    )
    print(f"compared={len(result.comparison_report.comparisons)}")
    print()
    print(result.markdown_summary)


if __name__ == "__main__":
    main()
