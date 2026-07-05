"""Assemble a per-candidate analysis report from existing analyses.

This is a *reporting* layer that gathers the already-computed pieces into one
consistent, JSON-serialisable structure per candidate:

* the candidate-vs-baseline comparison (Hero/Villain/rake EV and diffs);
* the selection labels (eligible, exclusion reasons, minimum-Villain-EV,
  Pareto frontier);
* the adaptation deadline ``T_deadline`` for the candidate; and
* an optional detection-time ``T_detect`` diagnostic for the candidate.

It runs :func:`~repeated_poker.selection.select_candidates` and
:func:`~repeated_poker.repeated.calculate_candidate_adaptation_deadlines` with
the *same* parameters so the labels and the deadline are mutually consistent.
It does not rank candidates and never auto-selects a single one; it only
arranges the existing results in the original candidate order.

The ``l1_distance`` reported here is a strategy-space L1 distance, not an
observable behavioural distance.  ``t_deadline`` is a sensitivity analysis over
an assumed switching opportunity ``m`` (when Villain adapts to the locked
candidate); it is not ``T_detect`` and not an opponent learning probability.

When detection is enabled, the report carries either the default local
``local_v0`` estimate or the opt-in per-hand ``reach_weighted_v1`` estimate (see
:mod:`repeated_poker.detection`). ``local_v0`` is conditional on reaching the
candidate's information set; ``reach_weighted_v1`` builds public observation
distributions from root-to-terminal path probabilities. Neither is a real
opponent-learning model. ``T_detect`` and ``T_deadline`` are distinct measures
and must not be conflated.

Two different detection-vs-deadline reads are reported, and they must not be
confused:

* ``t_detect_is_no_later_than_t_deadline`` is a *pure time comparison*
  (``estimated_opportunities <= t_deadline``).  Because ``t_deadline`` is the
  latest passing opportunity and Hero EV need not be monotone in the switching
  opportunity, being no later than the deadline does **not** imply Hero is at or
  above baseline.  It is not an economic-safety statement.
* ``detected_adaptation_is_at_least_baseline`` is the economic read: it maps the
  estimated detection opportunity onto the deadline timing rows (clamped to the
  ``m = N+1`` never-adapts row beyond the horizon) and reports whether Hero is at
  least at baseline EV if Villain adapts exactly then.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Dict, List, Optional, Sequence, Tuple

from .comparison import CandidateComparison, CandidateComparisonReport
from .detection import (
    DEFAULT_MAX_DETECTION_TERMINALS,
    DETECTION_METHOD_LOCAL_V0,
    DETECTION_METHOD_REACH_WEIGHTED_V1,
    DetectionResult,
    ReachWeightedDetectionResult,
    TerminalReveals,
    calculate_candidate_local_detection,
    calculate_candidate_reach_weighted_detection,
    candidate_observation_distance,
    resolve_detection_observation_model,
    validate_detection_parameters,
    validate_max_detection_terminals,
)
from .fixed_profile import FixedProfileValue
from .game import (
    GameTree,
    HeroStrategy,
    VillainStrategy,
    require_finite,
    require_valid_tolerance,
)
from .repeated import (
    DEFAULT_MAX_HORIZON,
    RESPONSE_MODE_WORST,
    CandidateAdaptationDeadline,
    _validate_response_mode,
    calculate_candidate_adaptation_deadlines,
    validate_deadline_parameters,
)
from .selection import (
    CandidateSelectionReport,
    _validate_selection_thresholds,
    select_candidates,
)


@dataclass(frozen=True)
class SelectionConfiguration:
    """The screening parameters used to build the report."""

    profit_tolerance: float
    max_l1_distance: Optional[float]
    tolerance: float

    def to_dict(self) -> dict:
        return {
            "profit_tolerance": self.profit_tolerance,
            "max_l1_distance": self.max_l1_distance,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True)
class DeadlineConfiguration:
    """The adaptation-deadline parameters used to build the report."""

    horizon: int
    discount: float
    response_mode: str
    tolerance: float
    max_horizon: int

    def to_dict(self) -> dict:
        return {
            "horizon": self.horizon,
            "discount": self.discount,
            "response_mode": self.response_mode,
            "tolerance": self.tolerance,
            "max_horizon": self.max_horizon,
        }


@dataclass(frozen=True)
class DetectionConfiguration:
    """The local detection-time parameters used to build the report.

    When ``enabled`` is ``False`` no detection information is computed and every
    row's detection fields are ``None``. ``method`` defaults to the historical
    local model; ``reach_weighted_v1`` is opt-in and reports per-hand fields.
    ``comparable_spot_occurrence_probability_per_physical_hand`` is an optional
    report-side diagnostic conversion from comparable opportunities to physical
    dealt hands; it is not part of the detection math.
    """

    enabled: bool
    log_likelihood_threshold: Optional[float]
    occurrence_probability_per_opportunity: Optional[float]
    tolerance: float
    method: str = DETECTION_METHOD_LOCAL_V0
    observation_model: Optional[str] = None
    comparable_spot_occurrence_probability_per_physical_hand: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "method": self.method,
            "observation_model": self.observation_model,
            "log_likelihood_threshold": self.log_likelihood_threshold,
            "occurrence_probability_per_opportunity": (
                self.occurrence_probability_per_opportunity
            ),
            "comparable_spot_occurrence_probability_per_physical_hand": (
                self.comparable_spot_occurrence_probability_per_physical_hand
            ),
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True)
class SelectionSummaryCounts:
    """Counts of candidates by selection outcome.

    ``pareto_frontier`` counts the strategy-space selection frontier
    (baseline-Villain EV, post-response robust Hero EV, strategy-space L1).
    ``ev_observation_deadline_pareto_frontier`` counts the separate M2-T2
    trade-off frontier over post-response worst-case Hero EV, observation
    distance, and ``T_deadline``; it is ``None`` when that frontier could not be
    computed (no baseline Hero strategy was available to measure observation
    distance).
    """

    total: int
    eligible: int
    excluded: int
    minimum_villain_ev: int
    pareto_frontier: int
    ev_observation_deadline_pareto_frontier: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "eligible": self.eligible,
            "excluded": self.excluded,
            "minimum_villain_ev": self.minimum_villain_ev,
            "pareto_frontier": self.pareto_frontier,
            "ev_observation_deadline_pareto_frontier": (
                self.ev_observation_deadline_pareto_frontier
            ),
        }


@dataclass(frozen=True)
class CandidateAnalysisRow:
    """A single candidate's consolidated analysis with English keys."""

    # Candidate metadata. For a single-shift candidate the scalar shift fields
    # (``info_set`` / ``source_action`` / ``target_action`` / ``shift_amount``)
    # describe the shift. For a multi-shift candidate (M2-T2) there is no single
    # shift, so those scalars are ``None`` and ``shifts`` carries the full
    # per-information-set breakdown (it is always populated, with one entry per
    # changed information set).
    candidate_id: str
    info_set: Optional[str]
    source_action: Optional[str]
    target_action: Optional[str]
    shift_amount: Optional[float]
    shifts: List[dict]
    l1_distance: float
    # Fixed-baseline values (candidate locked, Villain at baseline strategy).
    fixed_hero_ev: float
    fixed_villain_ev: float
    fixed_house_rake: float
    hero_ev_diff_from_baseline: float
    villain_ev_diff_from_baseline: float
    # Post-response values (Villain best response to the locked candidate).
    post_response_hero_ev_worst: float
    post_response_hero_ev_best: float
    post_response_hero_ev_worst_diff: float
    post_response_hero_ev_best_diff: float
    robustly_profitable: bool
    # Observable-distribution distance between baseline and candidate Hero action
    # distributions at the changed information set(s) (the max over sets for a
    # multi-shift candidate). ``None`` only when no baseline Hero strategy was
    # available. It is an observable distance, distinct from ``l1_distance``.
    observation_distance: Optional[float]
    # Selection labels.
    is_eligible: bool
    exclusion_reasons: List[str]
    is_minimum_villain_ev_candidate: bool
    is_pareto_frontier_candidate: bool
    # M2-T2 trade-off Pareto frontier over (post_response_hero_ev_worst higher is
    # better, observation_distance lower is better, t_deadline higher is better).
    # ``None`` when the frontier could not be computed (no observation distance).
    is_ev_observation_deadline_pareto_candidate: Optional[bool]
    # Deadline values.
    response_mode: str
    t_deadline: Optional[int]
    baseline_total_hero_ev: float
    never_adapts_total_hero_ev: float
    never_adapts_delta_from_baseline: float
    # Local detection values (all ``None`` when detection is disabled).
    detection_total_variation_distance: Optional[float]
    detection_kl_divergence_nats: Optional[float]
    detection_required_observations: Optional[int]
    detection_estimated_opportunities: Optional[int]
    # T_deadline vs T_detect comparison.
    t_detect_estimated_opportunities: Optional[int]
    # Pure time comparison only: NOT a statement about Hero's economic safety.
    t_detect_is_no_later_than_t_deadline: Optional[bool]
    # The estimated detection opportunity mapped onto the deadline timing rows,
    # clamped to ``horizon + 1`` (the never-adapts diagnostic row) when the
    # estimate exceeds the horizon.
    detected_adaptation_opportunity: Optional[int]
    detected_adaptation_delta_from_baseline: Optional[float]
    # Whether Hero is at least at baseline EV if Villain adapts at the estimated
    # detection opportunity. This is the economic-safety read, not the pure time
    # comparison above.
    detected_adaptation_is_at_least_baseline: Optional[bool]
    # Reach-weighted v1 detection values (all ``None`` for local_v0 / disabled).
    detection_kl_per_hand_nats: Optional[float] = None
    detection_tv_per_hand: Optional[float] = None
    detection_baseline_impossible_mass_per_hand: Optional[float] = None
    t_detect_hands: Optional[int] = None
    detection_time_basis: Optional[str] = None
    # Optional report-side diagnostic conversion from comparable opportunities to
    # physical dealt hands.
    t_detect_estimated_physical_hands: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "info_set": self.info_set,
            "source_action": self.source_action,
            "target_action": self.target_action,
            "shift_amount": self.shift_amount,
            "shifts": [dict(component) for component in self.shifts],
            "l1_distance": self.l1_distance,
            "fixed_hero_ev": self.fixed_hero_ev,
            "fixed_villain_ev": self.fixed_villain_ev,
            "fixed_house_rake": self.fixed_house_rake,
            "hero_ev_diff_from_baseline": self.hero_ev_diff_from_baseline,
            "villain_ev_diff_from_baseline": self.villain_ev_diff_from_baseline,
            "post_response_hero_ev_worst": self.post_response_hero_ev_worst,
            "post_response_hero_ev_best": self.post_response_hero_ev_best,
            "post_response_hero_ev_worst_diff": self.post_response_hero_ev_worst_diff,
            "post_response_hero_ev_best_diff": self.post_response_hero_ev_best_diff,
            "robustly_profitable": self.robustly_profitable,
            "observation_distance": self.observation_distance,
            "is_eligible": self.is_eligible,
            "exclusion_reasons": list(self.exclusion_reasons),
            "is_minimum_villain_ev_candidate": self.is_minimum_villain_ev_candidate,
            "is_pareto_frontier_candidate": self.is_pareto_frontier_candidate,
            "is_ev_observation_deadline_pareto_candidate": (
                self.is_ev_observation_deadline_pareto_candidate
            ),
            "response_mode": self.response_mode,
            "t_deadline": self.t_deadline,
            "baseline_total_hero_ev": self.baseline_total_hero_ev,
            "never_adapts_total_hero_ev": self.never_adapts_total_hero_ev,
            "never_adapts_delta_from_baseline": self.never_adapts_delta_from_baseline,
            "detection_total_variation_distance": (
                self.detection_total_variation_distance
            ),
            "detection_kl_divergence_nats": self.detection_kl_divergence_nats,
            "detection_required_observations": self.detection_required_observations,
            "detection_estimated_opportunities": (
                self.detection_estimated_opportunities
            ),
            "detection_kl_per_hand_nats": self.detection_kl_per_hand_nats,
            "detection_tv_per_hand": self.detection_tv_per_hand,
            "detection_baseline_impossible_mass_per_hand": (
                self.detection_baseline_impossible_mass_per_hand
            ),
            "t_detect_hands": self.t_detect_hands,
            "detection_time_basis": self.detection_time_basis,
            "t_detect_estimated_opportunities": self.t_detect_estimated_opportunities,
            "t_detect_estimated_physical_hands": (
                self.t_detect_estimated_physical_hands
            ),
            "t_detect_is_no_later_than_t_deadline": (
                self.t_detect_is_no_later_than_t_deadline
            ),
            "detected_adaptation_opportunity": self.detected_adaptation_opportunity,
            "detected_adaptation_delta_from_baseline": (
                self.detected_adaptation_delta_from_baseline
            ),
            "detected_adaptation_is_at_least_baseline": (
                self.detected_adaptation_is_at_least_baseline
            ),
        }


@dataclass(frozen=True)
class CandidateAnalysisReport:
    """A consolidated, JSON-serialisable analysis over all candidates."""

    baseline_value: FixedProfileValue
    selection_configuration: SelectionConfiguration
    deadline_configuration: DeadlineConfiguration
    detection_configuration: DetectionConfiguration
    rows: List[CandidateAnalysisRow]
    summary_counts: SelectionSummaryCounts

    def summary_rows(self) -> List[dict]:
        """Return one JSON-serialisable dict per candidate, in report order."""

        return [row.to_dict() for row in self.rows]

    def to_dict(self) -> dict:
        """Return a JSON-serialisable summary of the whole report.

        ``detection_kl_divergence_nats`` in a row may be ``inf`` when the
        candidate places probability on an action the baseline never plays; the
        standard library ``json`` module serialises this as ``Infinity``.
        """

        return {
            "baseline_value": self.baseline_value.to_dict(),
            "selection_configuration": self.selection_configuration.to_dict(),
            "deadline_configuration": self.deadline_configuration.to_dict(),
            "detection_configuration": self.detection_configuration.to_dict(),
            "summary_counts": self.summary_counts.to_dict(),
            "candidate_rows": self.summary_rows(),
        }


def ev_observation_deadline_pareto_ids(
    objectives: Sequence[Tuple[str, float, Optional[float], Optional[int]]],
    tolerance: float = 1e-9,
) -> Optional[set]:
    """Return the candidate ids on the M2-T2 trade-off Pareto frontier.

    ``objectives`` is one ``(candidate_id, post_response_hero_ev_worst,
    observation_distance, t_deadline)`` tuple per candidate. The three objectives
    and their better-directions are:

    * ``post_response_hero_ev_worst`` -- higher is better (robust profitability);
    * ``observation_distance`` -- lower is better (a smaller observable deviation
      is treated as more valuable, i.e. harder to detect);
    * ``t_deadline`` -- higher is better (a later safe adaptation deadline). A
      ``None`` deadline means no opportunity keeps Hero at least at baseline, so it
      is treated as the worst on this axis.

    A candidate is on the frontier when no other candidate is no worse on all
    three objectives and strictly better on at least one, using ``tolerance`` the
    same way as :func:`repeated_poker.selection.pareto_frontier` (so ties are all
    retained and none is dropped by candidate id).

    Returns ``None`` -- the frontier is undefined -- when any candidate has no
    ``observation_distance`` (no baseline Hero strategy was available to measure
    it), so the caller can report the flag as ``None`` rather than guess.

    The observation-distance direction and the ``None``-deadline handling are
    deliberate modelling choices, not equilibrium or optimality claims.
    """

    require_valid_tolerance(tolerance)
    if any(observation is None for _, _, observation, _ in objectives):
        return None

    vectors: List[Tuple[str, Tuple[float, float, float]]] = []
    for candidate_id, ev_worst, observation, t_deadline in objectives:
        # Fold all three objectives into a lower-is-better vector.
        deadline_value = t_deadline if t_deadline is not None else -math.inf
        vectors.append((candidate_id, (-ev_worst, observation, -deadline_value)))

    frontier: set = set()
    for candidate_id_b, vector_b in vectors:
        dominated = False
        for candidate_id_a, vector_a in vectors:
            if candidate_id_a == candidate_id_b:
                continue
            no_worse = all(
                vector_a[k] <= vector_b[k] + tolerance for k in range(3)
            )
            strictly_better = any(
                vector_a[k] < vector_b[k] - tolerance for k in range(3)
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.add(candidate_id_b)
    return frontier


def _require_unique_candidate_ids(report: CandidateComparisonReport) -> None:
    candidate_ids = [c.candidate.candidate_id for c in report.comparisons]
    duplicates = sorted(
        {cid for cid in candidate_ids if candidate_ids.count(cid) > 1}
    )
    if duplicates:
        raise ValueError(f"duplicate candidate_id(s) in report: {duplicates}")


def _validate_comparable_spot_occurrence_probability_per_physical_hand(
    value: Optional[float],
) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        raise ValueError(
            "comparable_spot_occurrence_probability_per_physical_hand must be "
            f"a finite number, got {value!r}"
        )
    if not isinstance(value, Real):
        raise ValueError(
            "comparable_spot_occurrence_probability_per_physical_hand must be "
            f"a finite number, got {value!r}"
        )
    require_finite(value, "comparable_spot_occurrence_probability_per_physical_hand")
    if not 0.0 < value <= 1.0:
        raise ValueError(
            "comparable_spot_occurrence_probability_per_physical_hand must "
            f"satisfy 0 < p <= 1, got {value!r}"
        )


def _validate_physical_hand_conversion_configuration(
    *,
    detection_enabled: bool,
    detection_method: str,
    detection_occurrence_probability_per_opportunity: Optional[float],
    comparable_spot_occurrence_probability_per_physical_hand: Optional[float],
) -> None:
    if comparable_spot_occurrence_probability_per_physical_hand is None:
        return
    _validate_comparable_spot_occurrence_probability_per_physical_hand(
        comparable_spot_occurrence_probability_per_physical_hand
    )
    if not detection_enabled:
        raise ValueError(
            "detection_comparable_spot_occurrence_probability_per_physical_hand "
            "requires detection_log_likelihood_threshold"
        )
    if (
        detection_method == DETECTION_METHOD_LOCAL_V0
        and detection_occurrence_probability_per_opportunity is None
    ):
        raise ValueError(
            "detection_occurrence_probability_per_opportunity is required with "
            "local_v0 physical-hand conversion"
        )


def _estimate_physical_hands(
    t_detect_estimated_opportunities: Optional[int],
    comparable_spot_occurrence_probability_per_physical_hand: Optional[float],
) -> Optional[int]:
    if (
        t_detect_estimated_opportunities is None
        or comparable_spot_occurrence_probability_per_physical_hand is None
    ):
        return None
    return math.ceil(
        t_detect_estimated_opportunities
        / comparable_spot_occurrence_probability_per_physical_hand
    )


def _build_row(
    comparison: CandidateComparison,
    deadline: CandidateAdaptationDeadline,
    detection: Optional[object],
    comparable_spot_occurrence_probability_per_physical_hand: Optional[float],
    observation_distance: Optional[float],
    eligible_ids,
    exclusion_reasons_by_id: Dict[str, List[str]],
    minimum_ids,
    pareto_ids,
    ev_observation_deadline_pareto_ids: Optional[set],
) -> CandidateAnalysisRow:
    candidate = comparison.candidate
    candidate_id = candidate.candidate_id
    result = deadline.result
    never_adapts_delta = (
        result.never_adapts_total_hero_ev - result.baseline_total_hero_ev
    )

    if detection is None:
        detection_tv: Optional[float] = None
        detection_kl: Optional[float] = None
        detection_required: Optional[int] = None
        detection_estimated: Optional[int] = None
        t_detect_estimated: Optional[int] = None
        detection_kl_per_hand: Optional[float] = None
        detection_tv_per_hand: Optional[float] = None
        detection_baseline_impossible_mass: Optional[float] = None
        t_detect_hands: Optional[int] = None
        detection_time_basis: Optional[str] = None
    elif isinstance(detection, ReachWeightedDetectionResult):
        detection_tv = None
        detection_kl = None
        detection_required = None
        detection_estimated = None
        t_detect_estimated = detection.t_detect_hands
        detection_kl_per_hand = detection.kl_per_hand_nats
        detection_tv_per_hand = detection.total_variation_per_hand
        detection_baseline_impossible_mass = (
            detection.baseline_impossible_mass_per_hand
        )
        t_detect_hands = detection.t_detect_hands
        detection_time_basis = detection.detection_time_basis
    else:
        detection_tv = detection.total_variation_distance
        detection_kl = detection.kl_divergence_nats
        detection_required = detection.required_observations
        detection_estimated = detection.estimated_opportunities
        t_detect_estimated = detection.estimated_opportunities
        detection_kl_per_hand = None
        detection_tv_per_hand = None
        detection_baseline_impossible_mass = None
        t_detect_hands = None
        detection_time_basis = None

    t_detect_estimated_physical_hands = _estimate_physical_hands(
        t_detect_estimated,
        comparable_spot_occurrence_probability_per_physical_hand,
    )
    t_deadline = result.t_deadline

    # Pure time comparison only; it does not say whether Hero is at baseline.
    if t_deadline is not None and t_detect_estimated is not None:
        t_detect_is_no_later: Optional[bool] = t_detect_estimated <= t_deadline
    else:
        t_detect_is_no_later = None

    # Map the estimated detection opportunity onto the deadline timing rows.
    # Beyond the horizon it becomes the m = N+1 never-adapts diagnostic row.
    if t_detect_estimated is None:
        detected_opportunity: Optional[int] = None
        detected_delta: Optional[float] = None
        detected_at_least_baseline: Optional[bool] = None
    else:
        if t_detect_estimated > result.horizon:
            detected_opportunity = result.horizon + 1
        else:
            detected_opportunity = t_detect_estimated
        timing_row = result.timing[detected_opportunity - 1]
        detected_delta = timing_row.delta_from_baseline
        detected_at_least_baseline = timing_row.is_at_least_baseline

    is_ev_obs_deadline_pareto: Optional[bool]
    if ev_observation_deadline_pareto_ids is None:
        is_ev_obs_deadline_pareto = None
    else:
        is_ev_obs_deadline_pareto = candidate_id in ev_observation_deadline_pareto_ids

    return CandidateAnalysisRow(
        candidate_id=candidate_id,
        info_set=candidate.info_set,
        source_action=candidate.source_action,
        target_action=candidate.target_action,
        shift_amount=candidate.shift_amount,
        shifts=[component.to_dict() for component in candidate.shifts],
        l1_distance=candidate.l1_distance,
        fixed_hero_ev=comparison.fixed_profile_value.hero_ev,
        fixed_villain_ev=comparison.fixed_profile_value.villain_ev,
        fixed_house_rake=comparison.fixed_profile_value.house_rake,
        hero_ev_diff_from_baseline=comparison.hero_ev_diff_from_baseline,
        villain_ev_diff_from_baseline=comparison.villain_ev_diff_from_baseline,
        post_response_hero_ev_worst=comparison.best_response.ev_h_worst,
        post_response_hero_ev_best=comparison.best_response.ev_h_best,
        post_response_hero_ev_worst_diff=comparison.post_response_hero_ev_worst_diff,
        post_response_hero_ev_best_diff=comparison.post_response_hero_ev_best_diff,
        robustly_profitable=comparison.robustly_profitable,
        observation_distance=observation_distance,
        is_eligible=candidate_id in eligible_ids,
        exclusion_reasons=exclusion_reasons_by_id.get(candidate_id, []),
        is_minimum_villain_ev_candidate=candidate_id in minimum_ids,
        is_pareto_frontier_candidate=candidate_id in pareto_ids,
        is_ev_observation_deadline_pareto_candidate=is_ev_obs_deadline_pareto,
        response_mode=deadline.response_mode,
        t_deadline=t_deadline,
        baseline_total_hero_ev=result.baseline_total_hero_ev,
        never_adapts_total_hero_ev=result.never_adapts_total_hero_ev,
        never_adapts_delta_from_baseline=never_adapts_delta,
        detection_total_variation_distance=detection_tv,
        detection_kl_divergence_nats=detection_kl,
        detection_required_observations=detection_required,
        detection_estimated_opportunities=detection_estimated,
        detection_kl_per_hand_nats=detection_kl_per_hand,
        detection_tv_per_hand=detection_tv_per_hand,
        detection_baseline_impossible_mass_per_hand=(
            detection_baseline_impossible_mass
        ),
        t_detect_hands=t_detect_hands,
        detection_time_basis=detection_time_basis,
        t_detect_estimated_opportunities=t_detect_estimated,
        t_detect_estimated_physical_hands=t_detect_estimated_physical_hands,
        t_detect_is_no_later_than_t_deadline=t_detect_is_no_later,
        detected_adaptation_opportunity=detected_opportunity,
        detected_adaptation_delta_from_baseline=detected_delta,
        detected_adaptation_is_at_least_baseline=detected_at_least_baseline,
    )


def build_candidate_analysis_report(
    comparison_report: CandidateComparisonReport,
    horizon: int,
    discount: float = 1.0,
    response_mode: str = RESPONSE_MODE_WORST,
    profit_tolerance: float = 0.0,
    max_l1_distance: Optional[float] = None,
    tolerance: float = 1e-9,
    max_horizon: int = DEFAULT_MAX_HORIZON,
    baseline_hero_strategy: Optional[HeroStrategy] = None,
    tree: Optional[GameTree] = None,
    baseline_villain_strategy: Optional[VillainStrategy] = None,
    detection_log_likelihood_threshold: Optional[float] = None,
    detection_occurrence_probability_per_opportunity: Optional[float] = None,
    detection_method: str = DETECTION_METHOD_LOCAL_V0,
    detection_observation_model: Optional[str] = None,
    terminal_reveals: Optional[TerminalReveals] = None,
    max_detection_terminals: int = DEFAULT_MAX_DETECTION_TERMINALS,
    detection_comparable_spot_occurrence_probability_per_physical_hand: Optional[
        float
    ] = None,
) -> CandidateAnalysisReport:
    """Consolidate selection, adaptation-deadline, and optional detection analyses.

    Runs candidate screening and the per-candidate adaptation deadline with the
    same ``tolerance`` (and the shared ``response_mode``), then emits one
    analysis row per candidate in the original report order.  No candidate is
    ranked or auto-selected.

    Detection is *disabled* unless ``detection_log_likelihood_threshold`` is
    given.  When enabled, ``baseline_hero_strategy`` is required and each
    candidate's local ``T_detect`` is computed with
    :func:`~repeated_poker.detection.calculate_candidate_local_detection` from
    the Hero action distributions at the candidate's own information set; this is
    a local, reach-conditional sensitivity analysis (see the module docstring).
    ``detection_comparable_spot_occurrence_probability_per_physical_hand`` is
    an optional report-side conversion from already-estimated comparable
    opportunities to physical dealt hands. It is diagnostic metadata only and is
    not used by the detection math, filtering, ranking, or ``T_deadline``.

    Raises :class:`ValueError` if the report contains duplicate candidate ids, or
    if any threshold/horizon/discount/response-mode/detection argument is
    invalid.  All scalar arguments are validated up front with the same shared
    helpers used by the underlying APIs, so an empty candidate report is still
    fully validated (no rule depends on the presence of candidates).
    """

    _require_unique_candidate_ids(comparison_report)
    _validate_selection_thresholds(profit_tolerance, max_l1_distance, tolerance)
    validate_deadline_parameters(horizon, discount, tolerance, max_horizon)
    _validate_response_mode(response_mode)
    resolved_observation_model = resolve_detection_observation_model(
        detection_method, detection_observation_model
    )
    validate_max_detection_terminals(max_detection_terminals)
    if (
        detection_method == DETECTION_METHOD_REACH_WEIGHTED_V1
        and detection_occurrence_probability_per_opportunity is not None
    ):
        raise ValueError(
            "detection_occurrence_probability_per_opportunity is only valid with "
            "detection_method='local_v0'"
        )

    detection_enabled = detection_log_likelihood_threshold is not None
    _validate_physical_hand_conversion_configuration(
        detection_enabled=detection_enabled,
        detection_method=detection_method,
        detection_occurrence_probability_per_opportunity=(
            detection_occurrence_probability_per_opportunity
        ),
        comparable_spot_occurrence_probability_per_physical_hand=(
            detection_comparable_spot_occurrence_probability_per_physical_hand
        ),
    )
    if detection_enabled:
        if baseline_hero_strategy is None:
            raise ValueError(
                "baseline_hero_strategy is required when "
                "detection_log_likelihood_threshold is given"
            )
        if detection_method == DETECTION_METHOD_REACH_WEIGHTED_V1:
            if tree is None:
                raise ValueError(
                    "tree is required when detection_method='reach_weighted_v1'"
                )
            if baseline_villain_strategy is None:
                raise ValueError(
                    "baseline_villain_strategy is required when "
                    "detection_method='reach_weighted_v1'"
                )
            validate_detection_parameters(
                detection_log_likelihood_threshold, None, tolerance
            )
        else:
            validate_detection_parameters(
                detection_log_likelihood_threshold,
                detection_occurrence_probability_per_opportunity,
                tolerance,
            )

    selection_report: CandidateSelectionReport = select_candidates(
        comparison_report,
        profit_tolerance=profit_tolerance,
        max_l1_distance=max_l1_distance,
        tolerance=tolerance,
    )
    deadlines = calculate_candidate_adaptation_deadlines(
        comparison_report,
        horizon=horizon,
        discount=discount,
        response_mode=response_mode,
        tolerance=tolerance,
        max_horizon=max_horizon,
    )

    if detection_enabled and detection_method == DETECTION_METHOD_REACH_WEIGHTED_V1:
        detections = [
            calculate_candidate_reach_weighted_detection(
                tree,
                baseline_hero_strategy,
                comparison.candidate,
                baseline_villain_strategy,
                log_likelihood_threshold=detection_log_likelihood_threshold,
                observation_model=resolved_observation_model,
                terminal_reveals=terminal_reveals,
                max_detection_terminals=max_detection_terminals,
                tolerance=tolerance,
            )
            for comparison in comparison_report.comparisons
        ]
    elif detection_enabled:
        detections: List[Optional[DetectionResult]] = [
            calculate_candidate_local_detection(
                baseline_hero_strategy,
                comparison.candidate,
                log_likelihood_threshold=detection_log_likelihood_threshold,
                occurrence_probability_per_opportunity=(
                    detection_occurrence_probability_per_opportunity
                ),
                tolerance=tolerance,
            )
            for comparison in comparison_report.comparisons
        ]
    else:
        detections = [None for _ in comparison_report.comparisons]

    # Observation distance (M2-T2 Pareto axis) is computed whenever a baseline
    # Hero strategy is available, independently of the optional detection-time
    # analysis, so the trade-off frontier is defined even when detection is off.
    if baseline_hero_strategy is not None:
        observation_distances: List[Optional[float]] = [
            candidate_observation_distance(
                baseline_hero_strategy, comparison.candidate, tolerance=tolerance
            )
            for comparison in comparison_report.comparisons
        ]
    else:
        observation_distances = [None for _ in comparison_report.comparisons]

    ev_observation_deadline_ids = ev_observation_deadline_pareto_ids(
        [
            (
                comparison.candidate.candidate_id,
                comparison.best_response.ev_h_worst,
                observation,
                deadline.result.t_deadline,
            )
            for comparison, deadline, observation in zip(
                comparison_report.comparisons, deadlines, observation_distances
            )
        ],
        tolerance=tolerance,
    )

    eligible_ids = {c.candidate.candidate_id for c in selection_report.eligible}
    exclusion_reasons_by_id = {
        excluded.comparison.candidate.candidate_id: excluded.reasons
        for excluded in selection_report.excluded
    }
    minimum_ids = {
        c.candidate.candidate_id
        for c in selection_report.minimum_villain_ev_candidates
    }
    pareto_ids = {c.candidate.candidate_id for c in selection_report.pareto_frontier}

    rows = [
        _build_row(
            comparison,
            deadline,
            detection,
            detection_comparable_spot_occurrence_probability_per_physical_hand,
            observation,
            eligible_ids,
            exclusion_reasons_by_id,
            minimum_ids,
            pareto_ids,
            ev_observation_deadline_ids,
        )
        for comparison, deadline, detection, observation in zip(
            comparison_report.comparisons,
            deadlines,
            detections,
            observation_distances,
        )
    ]

    summary_counts = SelectionSummaryCounts(
        total=len(comparison_report.comparisons),
        eligible=len(selection_report.eligible),
        excluded=len(selection_report.excluded),
        minimum_villain_ev=len(selection_report.minimum_villain_ev_candidates),
        pareto_frontier=len(selection_report.pareto_frontier),
        ev_observation_deadline_pareto_frontier=(
            None
            if ev_observation_deadline_ids is None
            else len(ev_observation_deadline_ids)
        ),
    )

    return CandidateAnalysisReport(
        baseline_value=comparison_report.baseline_value,
        selection_configuration=SelectionConfiguration(
            profit_tolerance=profit_tolerance,
            max_l1_distance=max_l1_distance,
            tolerance=tolerance,
        ),
        deadline_configuration=DeadlineConfiguration(
            horizon=horizon,
            discount=discount,
            response_mode=response_mode,
            tolerance=tolerance,
            max_horizon=max_horizon,
        ),
        detection_configuration=DetectionConfiguration(
            enabled=detection_enabled,
            log_likelihood_threshold=detection_log_likelihood_threshold,
            occurrence_probability_per_opportunity=(
                detection_occurrence_probability_per_opportunity
            ),
            comparable_spot_occurrence_probability_per_physical_hand=(
                detection_comparable_spot_occurrence_probability_per_physical_hand
            ),
            tolerance=tolerance,
            method=detection_method,
            observation_model=resolved_observation_model,
        ),
        rows=rows,
        summary_counts=summary_counts,
    )
