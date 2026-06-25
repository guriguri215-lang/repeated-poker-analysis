"""Assemble a per-candidate analysis report from existing analyses.

This is a *reporting* layer that gathers the already-computed pieces into one
consistent, JSON-serialisable structure per candidate:

* the candidate-vs-baseline comparison (Hero/Villain/rake EV and diffs);
* the selection labels (eligible, exclusion reasons, minimum-Villain-EV,
  Pareto frontier);
* the adaptation deadline ``T_deadline`` for the candidate; and
* an optional local detection-time ``T_detect`` for the candidate.

It runs :func:`~repeated_poker.selection.select_candidates` and
:func:`~repeated_poker.repeated.calculate_candidate_adaptation_deadlines` with
the *same* parameters so the labels and the deadline are mutually consistent.
It does not rank candidates and never auto-selects a single one; it only
arranges the existing results in the original candidate order.

The ``l1_distance`` reported here is a strategy-space L1 distance, not an
observable behavioural distance.  ``t_deadline`` is a sensitivity analysis over
an assumed switching opportunity ``m`` (when Villain adapts to the locked
candidate); it is not ``T_detect`` and not an opponent learning probability.

When detection is enabled, the report also carries a *local* ``T_detect`` for
each candidate, computed from the Hero action distributions at the candidate's
own information set (see :mod:`repeated_poker.detection`).  This is conditional
on reaching that information set and observing an action there; it ignores tree
reach probability, and it does not guarantee how a real opponent learns or
adapts.  ``T_detect`` and ``T_deadline`` are distinct measures and must not be
conflated.

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

from dataclasses import dataclass
from typing import Dict, List, Optional

from .comparison import CandidateComparison, CandidateComparisonReport
from .detection import (
    DetectionResult,
    calculate_candidate_local_detection,
    validate_detection_parameters,
)
from .fixed_profile import FixedProfileValue
from .game import HeroStrategy
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
    row's detection fields are ``None``.
    """

    enabled: bool
    log_likelihood_threshold: Optional[float]
    occurrence_probability_per_opportunity: Optional[float]
    tolerance: float

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "log_likelihood_threshold": self.log_likelihood_threshold,
            "occurrence_probability_per_opportunity": (
                self.occurrence_probability_per_opportunity
            ),
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True)
class SelectionSummaryCounts:
    """Counts of candidates by selection outcome."""

    total: int
    eligible: int
    excluded: int
    minimum_villain_ev: int
    pareto_frontier: int

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "eligible": self.eligible,
            "excluded": self.excluded,
            "minimum_villain_ev": self.minimum_villain_ev,
            "pareto_frontier": self.pareto_frontier,
        }


@dataclass(frozen=True)
class CandidateAnalysisRow:
    """A single candidate's consolidated analysis with English keys."""

    # Candidate metadata.
    candidate_id: str
    info_set: str
    source_action: str
    target_action: str
    shift_amount: float
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
    # Selection labels.
    is_eligible: bool
    exclusion_reasons: List[str]
    is_minimum_villain_ev_candidate: bool
    is_pareto_frontier_candidate: bool
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

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "info_set": self.info_set,
            "source_action": self.source_action,
            "target_action": self.target_action,
            "shift_amount": self.shift_amount,
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
            "is_eligible": self.is_eligible,
            "exclusion_reasons": list(self.exclusion_reasons),
            "is_minimum_villain_ev_candidate": self.is_minimum_villain_ev_candidate,
            "is_pareto_frontier_candidate": self.is_pareto_frontier_candidate,
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
            "t_detect_estimated_opportunities": self.t_detect_estimated_opportunities,
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


def _require_unique_candidate_ids(report: CandidateComparisonReport) -> None:
    candidate_ids = [c.candidate.candidate_id for c in report.comparisons]
    duplicates = sorted(
        {cid for cid in candidate_ids if candidate_ids.count(cid) > 1}
    )
    if duplicates:
        raise ValueError(f"duplicate candidate_id(s) in report: {duplicates}")


def _build_row(
    comparison: CandidateComparison,
    deadline: CandidateAdaptationDeadline,
    detection: Optional[DetectionResult],
    eligible_ids,
    exclusion_reasons_by_id: Dict[str, List[str]],
    minimum_ids,
    pareto_ids,
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
    else:
        detection_tv = detection.total_variation_distance
        detection_kl = detection.kl_divergence_nats
        detection_required = detection.required_observations
        detection_estimated = detection.estimated_opportunities

    t_deadline = result.t_deadline

    # Pure time comparison only; it does not say whether Hero is at baseline.
    if t_deadline is not None and detection_estimated is not None:
        t_detect_is_no_later: Optional[bool] = detection_estimated <= t_deadline
    else:
        t_detect_is_no_later = None

    # Map the estimated detection opportunity onto the deadline timing rows.
    # Beyond the horizon it becomes the m = N+1 never-adapts diagnostic row.
    if detection_estimated is None:
        detected_opportunity: Optional[int] = None
        detected_delta: Optional[float] = None
        detected_at_least_baseline: Optional[bool] = None
    else:
        if detection_estimated > result.horizon:
            detected_opportunity = result.horizon + 1
        else:
            detected_opportunity = detection_estimated
        timing_row = result.timing[detected_opportunity - 1]
        detected_delta = timing_row.delta_from_baseline
        detected_at_least_baseline = timing_row.is_at_least_baseline

    return CandidateAnalysisRow(
        candidate_id=candidate_id,
        info_set=candidate.info_set,
        source_action=candidate.source_action,
        target_action=candidate.target_action,
        shift_amount=candidate.shift_amount,
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
        is_eligible=candidate_id in eligible_ids,
        exclusion_reasons=exclusion_reasons_by_id.get(candidate_id, []),
        is_minimum_villain_ev_candidate=candidate_id in minimum_ids,
        is_pareto_frontier_candidate=candidate_id in pareto_ids,
        response_mode=deadline.response_mode,
        t_deadline=t_deadline,
        baseline_total_hero_ev=result.baseline_total_hero_ev,
        never_adapts_total_hero_ev=result.never_adapts_total_hero_ev,
        never_adapts_delta_from_baseline=never_adapts_delta,
        detection_total_variation_distance=detection_tv,
        detection_kl_divergence_nats=detection_kl,
        detection_required_observations=detection_required,
        detection_estimated_opportunities=detection_estimated,
        t_detect_estimated_opportunities=detection_estimated,
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
    detection_log_likelihood_threshold: Optional[float] = None,
    detection_occurrence_probability_per_opportunity: Optional[float] = None,
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

    detection_enabled = detection_log_likelihood_threshold is not None
    if detection_enabled:
        if baseline_hero_strategy is None:
            raise ValueError(
                "baseline_hero_strategy is required when "
                "detection_log_likelihood_threshold is given"
            )
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

    if detection_enabled:
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
            eligible_ids,
            exclusion_reasons_by_id,
            minimum_ids,
            pareto_ids,
        )
        for comparison, deadline, detection in zip(
            comparison_report.comparisons, deadlines, detections
        )
    ]

    summary_counts = SelectionSummaryCounts(
        total=len(comparison_report.comparisons),
        eligible=len(selection_report.eligible),
        excluded=len(selection_report.excluded),
        minimum_villain_ev=len(selection_report.minimum_villain_ev_candidates),
        pareto_frontier=len(selection_report.pareto_frontier),
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
            tolerance=tolerance,
        ),
        rows=rows,
        summary_counts=summary_counts,
    )
