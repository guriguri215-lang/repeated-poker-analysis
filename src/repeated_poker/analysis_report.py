"""Assemble a per-candidate analysis report from existing analyses.

This is a *reporting* layer that gathers the already-computed pieces into one
consistent, JSON-serialisable structure per candidate:

* the candidate-vs-baseline comparison (Hero/Villain/rake EV and diffs);
* the selection labels (eligible, exclusion reasons, minimum-Villain-EV,
  Pareto frontier); and
* the adaptation deadline ``T_deadline`` for the candidate.

It runs :func:`~repeated_poker.selection.select_candidates` and
:func:`~repeated_poker.repeated.calculate_candidate_adaptation_deadlines` with
the *same* parameters so the labels and the deadline are mutually consistent.
It does not rank candidates and never auto-selects a single one; it only
arranges the existing results in the original candidate order.

The ``l1_distance`` reported here is a strategy-space L1 distance, not an
observable behavioural distance.  ``t_deadline`` is a sensitivity analysis over
an assumed switching opportunity ``m`` (when Villain adapts to the locked
candidate); it is not ``T_detect`` and not an opponent learning probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .comparison import CandidateComparison, CandidateComparisonReport
from .fixed_profile import FixedProfileValue
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
        }


@dataclass(frozen=True)
class CandidateAnalysisReport:
    """A consolidated, JSON-serialisable analysis over all candidates."""

    baseline_value: FixedProfileValue
    selection_configuration: SelectionConfiguration
    deadline_configuration: DeadlineConfiguration
    rows: List[CandidateAnalysisRow]
    summary_counts: SelectionSummaryCounts

    def summary_rows(self) -> List[dict]:
        """Return one JSON-serialisable dict per candidate, in report order."""

        return [row.to_dict() for row in self.rows]

    def to_dict(self) -> dict:
        """Return a JSON-serialisable summary of the whole report."""

        return {
            "baseline_value": self.baseline_value.to_dict(),
            "selection_configuration": self.selection_configuration.to_dict(),
            "deadline_configuration": self.deadline_configuration.to_dict(),
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
        t_deadline=result.t_deadline,
        baseline_total_hero_ev=result.baseline_total_hero_ev,
        never_adapts_total_hero_ev=result.never_adapts_total_hero_ev,
        never_adapts_delta_from_baseline=never_adapts_delta,
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
) -> CandidateAnalysisReport:
    """Consolidate selection and adaptation-deadline analyses per candidate.

    Runs candidate screening and the per-candidate adaptation deadline with the
    same ``tolerance`` (and the shared ``response_mode``), then emits one
    analysis row per candidate in the original report order.  No candidate is
    ranked or auto-selected.

    Raises :class:`ValueError` if the report contains duplicate candidate ids,
    or if any threshold/horizon/discount/response-mode argument is invalid.  All
    arguments are validated up front with the same shared helpers used by the
    selection and deadline APIs, so an empty candidate report is still fully
    validated (no rule depends on the presence of candidates).
    """

    _require_unique_candidate_ids(comparison_report)
    _validate_selection_thresholds(profit_tolerance, max_l1_distance, tolerance)
    validate_deadline_parameters(horizon, discount, tolerance, max_horizon)
    _validate_response_mode(response_mode)

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
            eligible_ids,
            exclusion_reasons_by_id,
            minimum_ids,
            pareto_ids,
        )
        for comparison, deadline in zip(comparison_report.comparisons, deadlines)
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
        rows=rows,
        summary_counts=summary_counts,
    )
