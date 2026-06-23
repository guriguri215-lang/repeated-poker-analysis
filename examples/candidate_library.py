"""Worked example: a Hero candidate library compared against a baseline profile.

Reuses the nuts-chop river tree.  It fixes a baseline Hero strategy and a
baseline *mixed* Villain strategy, generates single-shift Hero candidates, and
compares each candidate against the baseline profile: value against the fixed
baseline Villain, EV differences, the exact Villain best response, and the
post-response Hero-EV interval with a robust-profitability flag.

No candidate is auto-selected and no repeated-game timing is computed here.
"""

from __future__ import annotations

import json

from nuts_chop_river import build_nuts_chop_river, default_hero_strategy

from repeated_poker import (
    VillainStrategy,
    compare_candidates,
    generate_shift_candidates,
)


def baseline_villain_strategy() -> VillainStrategy:
    """A fixed mixed Villain baseline for the nuts-chop river tree."""

    return VillainStrategy(
        probabilities={
            "V_check_a": {"check": 0.8, "bet": 0.2},
            "V_check_b": {"check": 0.8, "bet": 0.2},
            "V_bet_a": {"fold": 0.2, "call": 0.7, "raise": 0.1},
            "V_bet_b": {"fold": 0.2, "call": 0.7, "raise": 0.1},
        }
    )


def main() -> None:
    tree = build_nuts_chop_river()
    baseline_hero = default_hero_strategy()
    baseline_villain = baseline_villain_strategy()

    candidates = generate_shift_candidates(
        tree, baseline_hero, shift_amounts=[0.1, 0.2]
    )
    report = compare_candidates(tree, baseline_hero, baseline_villain, candidates)

    print("# Baseline profile value:")
    print(json.dumps(report.baseline_value.to_dict(), indent=2))

    print(f"\n# {len(report.comparisons)} candidate comparison(s):")
    for comparison in report.comparisons:
        summary = {
            "candidate_id": comparison.candidate.candidate_id,
            "l1_distance": comparison.candidate.l1_distance,
            "fixed_profile_value": comparison.fixed_profile_value.to_dict(),
            "villain_ev_diff_from_baseline": comparison.villain_ev_diff_from_baseline,
            "hero_ev_diff_from_baseline": comparison.hero_ev_diff_from_baseline,
            "post_response_hero_ev_worst_diff": (
                comparison.post_response_hero_ev_worst_diff
            ),
            "post_response_hero_ev_best_diff": (
                comparison.post_response_hero_ev_best_diff
            ),
            "robustly_profitable": comparison.robustly_profitable,
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
