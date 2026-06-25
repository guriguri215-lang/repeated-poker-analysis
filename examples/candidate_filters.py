"""Worked example: pre-filter generated Hero candidates before comparison.

Reuses the nuts-chop tree, generates Hero candidates, and applies
``filter_candidates`` with an L1-distance cap and a local-detection minimum.
It prints the kept/excluded counts and the exclusion reasons to stdout. It does
not write any file and does not run the comparison or selection stages.
"""

from __future__ import annotations

from nuts_chop_river import build_nuts_chop_river, default_hero_strategy

from repeated_poker import filter_candidates, generate_shift_candidates


def main() -> None:
    tree = build_nuts_chop_river()
    baseline_hero = default_hero_strategy()
    candidates = generate_shift_candidates(tree, baseline_hero, shift_amounts=[0.1, 0.2])

    result = filter_candidates(
        candidates,
        max_l1_distance=0.3,
        min_required_observations=5,
        baseline_hero_strategy=baseline_hero,
        detection_log_likelihood_threshold=3.0,
    )

    counts = result.summary_counts
    print(f"total={counts.total} kept={counts.kept} excluded={counts.excluded}")

    print("\nkept:")
    for candidate in result.kept:
        print(f"  {candidate.candidate_id}")

    print("\nexcluded:")
    for excluded in result.excluded:
        reasons = ", ".join(excluded.reasons)
        print(f"  {excluded.candidate.candidate_id}: {reasons}")


if __name__ == "__main__":
    main()
