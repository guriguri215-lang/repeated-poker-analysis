"""Compare a Hero candidate library against a fixed baseline profile.

For each Hero candidate this reports:

* its expected value against the fixed baseline Villain strategy;
* how that moves Hero and Villain EV away from the baseline profile;
* Villain's exact best-response correspondence to the locked candidate; and
* the post-response Hero-EV worst/best gap versus the baseline profile, plus a
  robust-profitability flag.

This stage deliberately does *not* auto-select a candidate and does not compute
any repeated-game timing measure (``T_deadline`` / ``T_detect``).  It exposes a
Python data structure for inspection; it adds no JSON/CSV or CLI surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .candidates import HeroStrategyCandidate
from .exact_response import (
    DEFAULT_MAX_PURE_STRATEGIES,
    BestResponseResult,
    solve_exact_response,
)
from .fixed_profile import FixedProfileValue, evaluate_fixed_profile
from .game import GameTree, HeroStrategy, VillainStrategy


@dataclass(frozen=True)
class CandidateComparison:
    """The comparison record for a single Hero candidate."""

    candidate: HeroStrategyCandidate
    fixed_profile_value: FixedProfileValue
    villain_ev_diff_from_baseline: float
    hero_ev_diff_from_baseline: float
    best_response: BestResponseResult
    post_response_hero_ev_worst_diff: float
    post_response_hero_ev_best_diff: float
    robustly_profitable: bool


@dataclass(frozen=True)
class CandidateComparisonReport:
    """Baseline profile value plus one comparison per Hero candidate."""

    baseline_value: FixedProfileValue
    comparisons: List[CandidateComparison]


def compare_candidates(
    tree: GameTree,
    baseline_hero_strategy: HeroStrategy,
    baseline_villain_strategy: VillainStrategy,
    candidates: Sequence[HeroStrategyCandidate],
    tolerance: float = 1e-9,
    max_pure_strategies: int = DEFAULT_MAX_PURE_STRATEGIES,
    *,
    allow_negative_residual: bool = False,
) -> CandidateComparisonReport:
    """Compare each Hero candidate against the fixed baseline profile.

    The baseline profile value (baseline Hero vs baseline Villain) is computed
    once.  For every candidate, the value against the fixed baseline Villain and
    the exact best response to the locked candidate are computed, and the EV
    differences from the baseline profile are recorded.  ``robustly_profitable``
    is ``True`` when the post-response worst-case Hero EV strictly exceeds the
    baseline profile's Hero EV. ``allow_negative_residual`` is forwarded to the
    validators used by the fixed-profile and exact-response stages.
    """

    baseline_value = evaluate_fixed_profile(
        tree,
        baseline_hero_strategy,
        baseline_villain_strategy,
        tolerance=tolerance,
        allow_negative_residual=allow_negative_residual,
    )

    comparisons: List[CandidateComparison] = []
    for candidate in candidates:
        fixed_profile_value = evaluate_fixed_profile(
            tree,
            candidate.hero_strategy,
            baseline_villain_strategy,
            tolerance=tolerance,
            allow_negative_residual=allow_negative_residual,
        )
        best_response = solve_exact_response(
            tree,
            candidate.hero_strategy,
            tolerance=tolerance,
            max_pure_strategies=max_pure_strategies,
            allow_negative_residual=allow_negative_residual,
        )
        comparisons.append(
            CandidateComparison(
                candidate=candidate,
                fixed_profile_value=fixed_profile_value,
                villain_ev_diff_from_baseline=(
                    fixed_profile_value.villain_ev - baseline_value.villain_ev
                ),
                hero_ev_diff_from_baseline=(
                    fixed_profile_value.hero_ev - baseline_value.hero_ev
                ),
                best_response=best_response,
                post_response_hero_ev_worst_diff=(
                    best_response.ev_h_worst - baseline_value.hero_ev
                ),
                post_response_hero_ev_best_diff=(
                    best_response.ev_h_best - baseline_value.hero_ev
                ),
                robustly_profitable=best_response.ev_h_worst > baseline_value.hero_ev,
            )
        )

    return CandidateComparisonReport(
        baseline_value=baseline_value, comparisons=comparisons
    )
