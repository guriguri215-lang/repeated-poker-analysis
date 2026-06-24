"""Repeated-game timing: the adaptation deadline ``T_deadline``.

This is the first time-axis measure for the repeated-game analysis.  Given
Hero's per-opportunity value under three regimes -- baseline ``b``, locked but
before Villain adapts ``a``, and after Villain adapts ``l`` -- and a horizon of
``N`` opportunities with discount ``delta``, it reports for every switching
opportunity ``m`` the total locked Hero value

``V_lock(m) = sum(t=1..m-1, delta**(t-1) * a) + sum(t=m..N, delta**(t-1) * l)``

against the baseline total ``V_base = sum(t=1..N, delta**(t-1) * b)``.

``T_deadline`` is the latest opportunity ``m`` in ``1..N`` at which Villain may
adapt while the locked policy stays at least as valuable as the baseline:

``T_deadline = max { m in [1, N] : V_lock(m) >= V_base }`` (``None`` if no such m).

This is a *sensitivity analysis* over an assumed switching opportunity ``m``.
It does **not** estimate when Villain will actually switch, identify the
commitment, or model any learning probability; that behavioural-identification
measure (``T_detect``) is deliberately out of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .comparison import CandidateComparison, CandidateComparisonReport
from .game import require_finite, require_valid_tolerance

# Villain post-adaptation response modes for the candidate integration.
RESPONSE_MODE_WORST = "worst"
RESPONSE_MODE_BEST = "best"
_RESPONSE_MODES = (RESPONSE_MODE_WORST, RESPONSE_MODE_BEST)

# Conservative default ceiling on the horizon, because the result materialises
# one timing row per opportunity (m = 1 .. N+1).
DEFAULT_MAX_HORIZON = 100_000


@dataclass(frozen=True)
class AdaptationTimingRow:
    """One switching opportunity ``m`` and its locked total Hero value."""

    adaptation_opportunity: int
    locked_total_hero_ev: float
    delta_from_baseline: float
    is_at_least_baseline: bool


@dataclass(frozen=True)
class AdaptationDeadlineResult:
    """The adaptation-deadline analysis for one ``(b, a, l, N, delta)`` setting.

    ``timing`` holds rows for ``m = 1 .. N+1``.  The ``m = N+1`` row is the
    diagnostic "Villain never adapts within N opportunities" case
    (``locked_total_hero_ev == never_adapts_total_hero_ev``); it is reported but
    is never a candidate for ``t_deadline``, whose search range is ``1 .. N``.
    """

    horizon: int
    discount: float
    baseline_total_hero_ev: float
    pre_adaptation_hero_ev: float
    post_adaptation_hero_ev: float
    timing: List[AdaptationTimingRow]
    t_deadline: Optional[int]
    never_adapts_total_hero_ev: float


def _geometric_sum(count: int, discount: float) -> float:
    """Return ``sum(i=0..count-1, discount**i)`` by stable accumulation.

    For ``discount == 1`` this is exactly ``count``.  Accumulation avoids the
    cancellation that the closed form ``(1 - discount**count) / (1 - discount)``
    can suffer when ``discount`` is very close to one.
    """

    total = 0.0
    term = 1.0
    for _ in range(count):
        total += term
        term *= discount
    return total


def _validate_horizon(horizon: int, max_horizon: int) -> None:
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    if isinstance(max_horizon, bool) or not isinstance(max_horizon, int):
        raise ValueError(f"max_horizon must be a positive integer, got {max_horizon!r}")
    if max_horizon < 1:
        raise ValueError(f"max_horizon must be at least 1, got {max_horizon}")
    if horizon > max_horizon:
        raise ValueError(
            f"horizon {horizon} exceeds the safety limit max_horizon={max_horizon}; "
            "raise the limit deliberately if you truly intend this many rows."
        )


def calculate_adaptation_deadline(
    baseline_hero_ev: float,
    pre_adaptation_hero_ev: float,
    post_adaptation_hero_ev: float,
    horizon: int,
    discount: float = 1.0,
    tolerance: float = 1e-9,
    max_horizon: int = DEFAULT_MAX_HORIZON,
) -> AdaptationDeadlineResult:
    """Compute the adaptation deadline for a fixed ``(b, a, l, N, delta)`` setting.

    ``baseline_hero_ev`` (``b``), ``pre_adaptation_hero_ev`` (``a``), and
    ``post_adaptation_hero_ev`` (``l``) are Hero's per-opportunity values.
    ``horizon`` (``N``) is a positive integer; ``discount`` (``delta``) is finite
    with ``0 < discount <= 1``.  The ``V_lock(m) >= V_base`` test uses
    ``tolerance`` (finite, non-negative).  ``t_deadline`` is the largest
    ``m in [1, N]`` that passes the test, scanning every ``m`` (monotonicity is
    not assumed), or ``None`` if none passes.
    """

    require_finite(baseline_hero_ev, "baseline_hero_ev")
    require_finite(pre_adaptation_hero_ev, "pre_adaptation_hero_ev")
    require_finite(post_adaptation_hero_ev, "post_adaptation_hero_ev")
    require_valid_tolerance(tolerance)
    require_finite(discount, "discount")
    if not 0.0 < discount <= 1.0:
        raise ValueError(f"discount must satisfy 0 < discount <= 1, got {discount!r}")
    _validate_horizon(horizon, max_horizon)

    total_weight = _geometric_sum(horizon, discount)
    baseline_total = baseline_hero_ev * total_weight

    rows: List[AdaptationTimingRow] = []
    t_deadline: Optional[int] = None

    prefix_weight = 0.0  # G(m-1): discounted weight of the pre-adaptation block
    term = 1.0  # discount**(m-1)
    for opportunity in range(1, horizon + 2):  # m = 1 .. N+1
        pre = pre_adaptation_hero_ev * prefix_weight
        post = post_adaptation_hero_ev * (total_weight - prefix_weight)
        locked_total = pre + post
        delta_from_baseline = locked_total - baseline_total
        is_at_least_baseline = delta_from_baseline >= -tolerance

        rows.append(
            AdaptationTimingRow(
                adaptation_opportunity=opportunity,
                locked_total_hero_ev=locked_total,
                delta_from_baseline=delta_from_baseline,
                is_at_least_baseline=is_at_least_baseline,
            )
        )
        if opportunity <= horizon and is_at_least_baseline:
            t_deadline = opportunity  # ascending scan keeps the largest passing m

        prefix_weight += term
        term *= discount

    never_adapts_total = rows[-1].locked_total_hero_ev  # m = N+1 row: all 'a'

    return AdaptationDeadlineResult(
        horizon=horizon,
        discount=discount,
        baseline_total_hero_ev=baseline_total,
        pre_adaptation_hero_ev=pre_adaptation_hero_ev,
        post_adaptation_hero_ev=post_adaptation_hero_ev,
        timing=rows,
        t_deadline=t_deadline,
        never_adapts_total_hero_ev=never_adapts_total,
    )


@dataclass(frozen=True)
class CandidateAdaptationDeadline:
    """An adaptation-deadline result tied to a candidate and response mode."""

    candidate_id: str
    response_mode: str
    result: AdaptationDeadlineResult


def _post_adaptation_hero_ev(
    comparison: CandidateComparison, response_mode: str
) -> float:
    if response_mode == RESPONSE_MODE_WORST:
        return comparison.best_response.ev_h_worst
    if response_mode == RESPONSE_MODE_BEST:
        return comparison.best_response.ev_h_best
    raise ValueError(
        f"unknown response_mode {response_mode!r}; expected one of {_RESPONSE_MODES}"
    )


def calculate_candidate_adaptation_deadline(
    report: CandidateComparisonReport,
    comparison: CandidateComparison,
    horizon: int,
    discount: float = 1.0,
    response_mode: str = RESPONSE_MODE_WORST,
    tolerance: float = 1e-9,
    max_horizon: int = DEFAULT_MAX_HORIZON,
) -> CandidateAdaptationDeadline:
    """Compute the adaptation deadline for one candidate against the baseline.

    The baseline value ``b`` is ``report.baseline_value.hero_ev``; the
    pre-adaptation value ``a`` is ``comparison.fixed_profile_value.hero_ev``
    (the candidate locked while Villain keeps the baseline strategy); and the
    post-adaptation value ``l`` is Villain's exact best-response Hero EV chosen
    by ``response_mode`` (``"worst"`` by default, or ``"best"``).

    This is a sensitivity analysis over an assumed switching opportunity ``m``;
    it does not estimate when or whether Villain actually adapts.
    """

    baseline_hero_ev = report.baseline_value.hero_ev
    pre_adaptation = comparison.fixed_profile_value.hero_ev
    post_adaptation = _post_adaptation_hero_ev(comparison, response_mode)

    result = calculate_adaptation_deadline(
        baseline_hero_ev=baseline_hero_ev,
        pre_adaptation_hero_ev=pre_adaptation,
        post_adaptation_hero_ev=post_adaptation,
        horizon=horizon,
        discount=discount,
        tolerance=tolerance,
        max_horizon=max_horizon,
    )
    return CandidateAdaptationDeadline(
        candidate_id=comparison.candidate.candidate_id,
        response_mode=response_mode,
        result=result,
    )


def calculate_candidate_adaptation_deadlines(
    report: CandidateComparisonReport,
    horizon: int,
    discount: float = 1.0,
    response_mode: str = RESPONSE_MODE_WORST,
    tolerance: float = 1e-9,
    max_horizon: int = DEFAULT_MAX_HORIZON,
) -> List[CandidateAdaptationDeadline]:
    """Compute adaptation deadlines for every comparison in ``report``."""

    return [
        calculate_candidate_adaptation_deadline(
            report,
            comparison,
            horizon=horizon,
            discount=discount,
            response_mode=response_mode,
            tolerance=tolerance,
            max_horizon=max_horizon,
        )
        for comparison in report.comparisons
    ]
