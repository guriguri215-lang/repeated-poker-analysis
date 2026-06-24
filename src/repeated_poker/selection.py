"""Screen and rank Hero candidates from a CandidateComparisonReport.

This is a candidate-selection / screening stage built on top of the
already-computed :class:`~repeated_poker.comparison.CandidateComparison`
records.  It does three things:

* mark each candidate eligible or excluded (with explicit English reasons);
* return the eligible candidate(s) that minimise Villain EV while Villain keeps
  the fixed baseline mixed strategy; and
* return the Pareto frontier over Villain EV, post-response robust Hero EV, and
  Hero strategy-space L1 distance.

It deliberately does *not* auto-select a single candidate, and it computes no
repeated-game timing (``T_deadline`` / ``T_detect``), observable-distance, or
external-solver quantity.  The ``l1_distance`` used here is a plain strategy-
space distance, not an observable behavioural distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .comparison import CandidateComparison, CandidateComparisonReport
from .game import require_finite, require_valid_tolerance

# Exclusion-reason codes (stable English identifiers).
NOT_ROBUSTLY_PROFITABLE = "not_robustly_profitable"
L1_DISTANCE_EXCEEDS_LIMIT = "l1_distance_exceeds_limit"


@dataclass(frozen=True)
class ExcludedCandidate:
    """An ineligible candidate together with its English exclusion reasons."""

    comparison: CandidateComparison
    reasons: List[str]


@dataclass(frozen=True)
class CandidateSelectionReport:
    """Screening and ranking of a candidate library.

    ``eligible`` and ``excluded`` partition the comparisons.
    ``minimum_villain_ev_candidates`` holds every eligible candidate whose
    value against the fixed baseline Villain strategy ties for the lowest
    Villain EV.  ``pareto_frontier`` holds the non-dominated eligible
    candidates over the three trade-off objectives.  No single candidate is
    auto-selected.
    """

    eligible: List[CandidateComparison]
    excluded: List[ExcludedCandidate]
    minimum_villain_ev_candidates: List[CandidateComparison]
    pareto_frontier: List[CandidateComparison]
    profit_tolerance: float
    max_l1_distance: Optional[float]
    tolerance: float

    @property
    def has_eligible_candidates(self) -> bool:
        return bool(self.eligible)


def _validate_selection_thresholds(
    profit_tolerance: float, max_l1_distance: Optional[float], tolerance: float
) -> None:
    require_valid_tolerance(tolerance)
    require_finite(profit_tolerance, "profit_tolerance")
    if max_l1_distance is not None:
        require_finite(max_l1_distance, "max_l1_distance")
        if max_l1_distance < 0:
            raise ValueError(
                f"max_l1_distance must be non-negative, got {max_l1_distance!r}"
            )


def candidate_exclusion_reasons(
    comparison: CandidateComparison,
    profit_tolerance: float = 0.0,
    max_l1_distance: Optional[float] = None,
    tolerance: float = 1e-9,
) -> List[str]:
    """Return the exclusion reasons for one candidate (empty means eligible).

    A candidate is eligible when its post-response worst-case Hero EV gap
    strictly exceeds ``profit_tolerance`` (beyond the comparison ``tolerance``)
    and, when ``max_l1_distance`` is given, its Hero strategy-space L1 distance
    does not exceed that limit.
    """

    _validate_selection_thresholds(profit_tolerance, max_l1_distance, tolerance)

    reasons: List[str] = []
    if comparison.post_response_hero_ev_worst_diff <= profit_tolerance + tolerance:
        reasons.append(NOT_ROBUSTLY_PROFITABLE)
    if (
        max_l1_distance is not None
        and comparison.candidate.l1_distance > max_l1_distance + tolerance
    ):
        reasons.append(L1_DISTANCE_EXCEEDS_LIMIT)
    return reasons


def select_minimum_villain_ev(
    eligible: Sequence[CandidateComparison], tolerance: float = 1e-9
) -> List[CandidateComparison]:
    """Return eligible candidates that minimise baseline-Villain EV.

    The minimised quantity is ``fixed_profile_value.villain_ev``: Villain's EV
    *while Villain keeps the fixed baseline mixed strategy* against the locked
    candidate, not Villain's best-response EV.  Every candidate tying for the
    minimum within ``tolerance`` is returned.  An empty input yields an empty
    list.
    """

    require_valid_tolerance(tolerance)
    if not eligible:
        return []
    minimum = min(c.fixed_profile_value.villain_ev for c in eligible)
    return [
        c
        for c in eligible
        if c.fixed_profile_value.villain_ev <= minimum + tolerance
    ]


def _objective_vector(comparison: CandidateComparison) -> Tuple[float, float, float]:
    """Lower-is-better objective vector for Pareto comparison.

    Components: baseline-Villain EV (lower better), the negated post-response
    worst-case Hero EV gap (so that higher Hero EV becomes lower), and the
    Hero strategy-space L1 distance (lower better).
    """

    return (
        comparison.fixed_profile_value.villain_ev,
        -comparison.post_response_hero_ev_worst_diff,
        comparison.candidate.l1_distance,
    )


def _dominates(
    a: CandidateComparison, b: CandidateComparison, tolerance: float
) -> bool:
    """Return ``True`` if ``a`` Pareto-dominates ``b`` within ``tolerance``."""

    va, vb = _objective_vector(a), _objective_vector(b)
    no_worse = all(va[k] <= vb[k] + tolerance for k in range(len(va)))
    strictly_better = any(va[k] < vb[k] - tolerance for k in range(len(va)))
    return no_worse and strictly_better


def pareto_frontier(
    eligible: Sequence[CandidateComparison], tolerance: float = 1e-9
) -> List[CandidateComparison]:
    """Return the non-dominated eligible candidates over the three objectives.

    Objectives: ``fixed_profile_value.villain_ev`` (lower is better),
    ``post_response_hero_ev_worst_diff`` (higher is better), and
    ``candidate.l1_distance`` (lower is better).  Domination uses ``tolerance``,
    so candidates with equal objective values never dominate one another and
    are all retained; none is dropped arbitrarily by candidate id.
    """

    require_valid_tolerance(tolerance)
    frontier: List[CandidateComparison] = []
    for b in eligible:
        if not any(_dominates(a, b, tolerance) for a in eligible if a is not b):
            frontier.append(b)
    return frontier


def select_candidates(
    report: CandidateComparisonReport,
    profit_tolerance: float = 0.0,
    max_l1_distance: Optional[float] = None,
    tolerance: float = 1e-9,
) -> CandidateSelectionReport:
    """Screen and rank the candidates in ``report``.

    Partitions the report's comparisons into eligible and excluded (with
    reasons), then computes the minimum-baseline-Villain-EV candidate set and
    the three-objective Pareto frontier over the eligible candidates.  No single
    candidate is auto-selected.
    """

    _validate_selection_thresholds(profit_tolerance, max_l1_distance, tolerance)

    eligible: List[CandidateComparison] = []
    excluded: List[ExcludedCandidate] = []
    for comparison in report.comparisons:
        reasons = candidate_exclusion_reasons(
            comparison, profit_tolerance, max_l1_distance, tolerance
        )
        if reasons:
            excluded.append(ExcludedCandidate(comparison=comparison, reasons=reasons))
        else:
            eligible.append(comparison)

    return CandidateSelectionReport(
        eligible=eligible,
        excluded=excluded,
        minimum_villain_ev_candidates=select_minimum_villain_ev(eligible, tolerance),
        pareto_frontier=pareto_frontier(eligible, tolerance),
        profit_tolerance=profit_tolerance,
        max_l1_distance=max_l1_distance,
        tolerance=tolerance,
    )
