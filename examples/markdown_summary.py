"""Worked example: render a candidate analysis report as Markdown.

Reuses the nuts-chop candidate library, builds a consolidated analysis report
with detection enabled, and prints a human-readable Markdown summary to stdout.
It does not write any file.
"""

from __future__ import annotations

from candidate_library import baseline_villain_strategy
from nuts_chop_river import build_nuts_chop_river, default_hero_strategy

from repeated_poker import (
    build_candidate_analysis_report,
    compare_candidates,
    format_candidate_analysis_markdown,
    generate_shift_candidates,
)


def main() -> None:
    tree = build_nuts_chop_river()
    baseline_hero = default_hero_strategy()
    baseline_villain = baseline_villain_strategy()

    candidates = generate_shift_candidates(tree, baseline_hero, shift_amounts=[0.1, 0.2])
    comparison_report = compare_candidates(
        tree, baseline_hero, baseline_villain, candidates
    )

    report = build_candidate_analysis_report(
        comparison_report,
        horizon=5,
        profit_tolerance=-2.0,
        max_l1_distance=0.3,
        baseline_hero_strategy=baseline_hero,
        detection_log_likelihood_threshold=3.0,
        detection_occurrence_probability_per_opportunity=0.5,
    )

    print(format_candidate_analysis_markdown(report))


if __name__ == "__main__":
    main()
