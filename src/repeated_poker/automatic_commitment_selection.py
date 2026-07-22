"""Bounded automatic Hero commitment selection by adaptation opportunity.

The selector in this module consumes an already-computed
:class:`~repeated_poker.comparison.CandidateComparisonReport`.  For every
``m = 1 .. N+1`` it reuses the existing repeated-game deadline API in
``response_mode="worst"`` and selects the candidate with the largest total
Hero-EV delta from baseline.  A commitment is returned only when that best
delta strictly exceeds the configured minimum uplift beyond ``tolerance``.

The result is an adaptation-opportunity-conditional optimum over the declared
finite candidate library only.  It is not a continuous/global optimum, an
equilibrium result, a prediction of when an opponent adapts, or strategy or
profitability advice.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from numbers import Real
from typing import List, Optional, Sequence, Tuple

from .candidates import DEFAULT_MAX_CANDIDATES
from .comparison import CandidateComparison, CandidateComparisonReport
from .fixed_profile import FixedProfileValue
from .game import require_finite
from .repeated import (
    DEFAULT_MAX_HORIZON,
    RESPONSE_MODE_WORST,
    AdaptationDeadlineResult,
    calculate_adaptation_deadline,
    calculate_candidate_adaptation_deadlines,
    validate_deadline_parameters,
)

AUTOMATIC_COMMITMENT_SELECTION_CONTRACT_VERSION = (
    "automatic-commitment-selection-v1"
)
AUTOMATIC_COMMITMENT_SELECTION_STATUS_COMPLETE = "COMPLETE"
AUTOMATIC_COMMITMENT_SELECTION_ROW_STATUS_SELECTED = "SELECTED"
NO_BENEFICIAL_COMMITMENT = "NO_BENEFICIAL_COMMITMENT"
AUTOMATIC_COMMITMENT_RESPONSE_SEMANTICS = (
    "villain_exact_best_response_hero_worst"
)
AUTOMATIC_COMMITMENT_CLAIM_SCOPE = (
    "bounded_finite_candidate_library_adaptation_opportunity_conditional_optimum"
)
AUTOMATIC_COMMITMENT_BASELINE_IDENTITY_ALGORITHM = (
    "baseline-fixed-profile-value-sha256-canonical-json-v1"
)

# The existing repeated API materialises N+1 timing rows for every candidate.
# This combined cap is checked before that materialisation begins.
DEFAULT_MAX_AUTOMATIC_TIMING_ROWS = 1_000_000


@dataclass(frozen=True)
class AutomaticCommitmentSelectionConfig:
    """Selector-specific threshold and allocation caps.

    ``minimum_total_uplift`` is non-negative.  A row selects a commitment only
    when its best total Hero-EV delta is strictly greater than
    ``minimum_total_uplift + tolerance``.  ``max_candidates`` bounds the kept
    comparison universe, and ``max_timing_rows`` bounds
    ``kept_candidates * (horizon + 1)`` before the existing repeated rows are
    materialised.
    """

    minimum_total_uplift: float = 0.0
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    max_timing_rows: int = DEFAULT_MAX_AUTOMATIC_TIMING_ROWS

    def to_dict(self) -> dict:
        return {
            "minimum_total_uplift": self.minimum_total_uplift,
            "max_candidates": self.max_candidates,
            "max_timing_rows": self.max_timing_rows,
        }


@dataclass(frozen=True)
class AutomaticCommitmentSearchCoverage:
    """Declared finite search coverage and optional pipeline provenance.

    ``input_candidate_ids`` is the pre-filter generated library and
    ``kept_candidate_ids`` is the exact universe present in
    ``CandidateComparisonReport.comparisons``.  Direct selector callers that do
    not supply pipeline provenance report the comparison universe as both sets.
    """

    input_candidate_ids: Tuple[str, ...]
    kept_candidate_ids: Tuple[str, ...]
    source: str = "comparison_report"
    shift_amounts: Optional[Tuple[float, ...]] = None
    max_simultaneous_info_sets: Optional[int] = None
    generation_max_candidates: Optional[int] = None
    filtering_applied: bool = False
    filter_allowed_info_sets: Optional[Tuple[str, ...]] = None
    filter_max_l1_distance: Optional[float] = None
    filter_min_required_observations: Optional[int] = None

    def to_dict(self) -> dict:
        generation_configuration: Optional[dict]
        if self.shift_amounts is None:
            generation_configuration = None
        else:
            generation_configuration = {
                "shift_amounts": list(self.shift_amounts),
                "max_simultaneous_info_sets": self.max_simultaneous_info_sets,
                "max_candidates": self.generation_max_candidates,
            }

        filter_configuration: Optional[dict]
        if not self.filtering_applied:
            filter_configuration = None
        else:
            filter_configuration = {
                "allowed_info_sets": (
                    None
                    if self.filter_allowed_info_sets is None
                    else list(self.filter_allowed_info_sets)
                ),
                "max_l1_distance": self.filter_max_l1_distance,
                "min_required_observations": self.filter_min_required_observations,
            }

        return {
            "source": self.source,
            "input_candidate_count": len(self.input_candidate_ids),
            "input_candidate_ids": list(self.input_candidate_ids),
            "kept_candidate_count": len(self.kept_candidate_ids),
            "kept_candidate_ids": list(self.kept_candidate_ids),
            "generation_configuration": generation_configuration,
            "filter_configuration": filter_configuration,
        }


@dataclass(frozen=True)
class AutomaticCommitmentValueCandidate:
    """Domain-neutral scalar inputs for one bounded commitment candidate.

    The selector needs only the fixed-opponent value, the conservative and
    optimistic post-response Hero values, their conservative baseline
    difference, and the candidate's declared L1 distance.  Domain adapters
    retain their native response provenance; they must not manufacture a
    generic tree response merely to call the selector.
    """

    candidate_id: str
    fixed_profile_value: FixedProfileValue
    post_response_hero_ev_worst: float
    post_response_hero_ev_best: float
    post_response_hero_ev_worst_diff: float
    l1_distance: float

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "fixed_profile_value": self.fixed_profile_value.to_dict(),
            "post_response_hero_ev_worst": self.post_response_hero_ev_worst,
            "post_response_hero_ev_best": self.post_response_hero_ev_best,
            "post_response_hero_ev_worst_diff": (
                self.post_response_hero_ev_worst_diff
            ),
            "l1_distance": self.l1_distance,
        }


@dataclass(frozen=True)
class AutomaticCommitmentTieBreakEvidence:
    """One primary-tied candidate's deterministic secondary evidence."""

    candidate_id: str
    total_hero_ev_delta: float
    post_response_hero_ev_worst_diff: float
    l1_distance: float
    within_best_post_response_tolerance: bool
    within_best_l1_tolerance: bool
    is_primary_tie_display_candidate: bool

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "total_hero_ev_delta": self.total_hero_ev_delta,
            "post_response_hero_ev_worst_diff": (
                self.post_response_hero_ev_worst_diff
            ),
            "l1_distance": self.l1_distance,
            "within_best_post_response_tolerance": (
                self.within_best_post_response_tolerance
            ),
            "within_best_l1_tolerance": self.within_best_l1_tolerance,
            "is_primary_tie_display_candidate": (
                self.is_primary_tie_display_candidate
            ),
        }


@dataclass(frozen=True)
class AutomaticCommitmentSelectionRow:
    """The conditional selection for one adaptation opportunity ``m``."""

    adaptation_opportunity: int
    status: str
    best_total_hero_ev_delta: Optional[float]
    selected_candidate_id: Optional[str]
    primary_tie_candidate_ids: Tuple[str, ...]
    primary_tie_display_candidate_id: Optional[str]
    tie_break_evidence: Tuple[AutomaticCommitmentTieBreakEvidence, ...]

    def to_dict(self) -> dict:
        return {
            "adaptation_opportunity": self.adaptation_opportunity,
            "status": self.status,
            "best_total_hero_ev_delta": self.best_total_hero_ev_delta,
            "selected_candidate_id": self.selected_candidate_id,
            "primary_tie_candidate_ids": list(self.primary_tie_candidate_ids),
            "primary_tie_display_candidate_id": (
                self.primary_tie_display_candidate_id
            ),
            "tie_break_evidence": [
                evidence.to_dict() for evidence in self.tie_break_evidence
            ],
        }


@dataclass(frozen=True)
class AutomaticCommitmentSelectionReport:
    """Deterministic, JSON-serialisable conditional selections for all ``m``."""

    status: str
    baseline_identity: str
    baseline_hero_ev: float
    baseline_villain_ev: float
    baseline_house_rake: float
    horizon: int
    discount: float
    tolerance: float
    max_horizon: int
    configuration: AutomaticCommitmentSelectionConfig
    search_coverage: AutomaticCommitmentSearchCoverage
    timing_row_evaluation_count: int
    rows: Tuple[AutomaticCommitmentSelectionRow, ...]

    def summary_rows(self) -> List[dict]:
        return [row.to_dict() for row in self.rows]

    def to_dict(self) -> dict:
        return {
            "contract_version": AUTOMATIC_COMMITMENT_SELECTION_CONTRACT_VERSION,
            "status": self.status,
            "response_semantics": AUTOMATIC_COMMITMENT_RESPONSE_SEMANTICS,
            "claim_scope": AUTOMATIC_COMMITMENT_CLAIM_SCOPE,
            "baseline_identity_algorithm": (
                AUTOMATIC_COMMITMENT_BASELINE_IDENTITY_ALGORITHM
            ),
            "baseline_identity": self.baseline_identity,
            "baseline_value": {
                "hero_ev": self.baseline_hero_ev,
                "villain_ev": self.baseline_villain_ev,
                "house_rake": self.baseline_house_rake,
            },
            "selection_configuration": {
                "horizon": self.horizon,
                "discount": self.discount,
                "tolerance": self.tolerance,
                "max_horizon": self.max_horizon,
                **self.configuration.to_dict(),
            },
            "search_coverage": self.search_coverage.to_dict(),
            "timing_row_evaluation_count": self.timing_row_evaluation_count,
            "rows": self.summary_rows(),
        }


@dataclass(frozen=True)
class _CandidateScore:
    candidate: AutomaticCommitmentValueCandidate
    total_hero_ev_delta: float


@dataclass(frozen=True)
class _CandidateValueDeadline:
    candidate_id: str
    response_mode: str
    result: AdaptationDeadlineResult


def _require_finite_number(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    require_finite(value, name)


def _validate_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value!r}")


def validate_automatic_commitment_selection_parameters(
    *,
    horizon: int,
    discount: float = 1.0,
    tolerance: float = 1e-9,
    max_horizon: int = DEFAULT_MAX_HORIZON,
    configuration: AutomaticCommitmentSelectionConfig = (
        AutomaticCommitmentSelectionConfig()
    ),
) -> None:
    """Validate selector parameters independently of candidate presence."""

    if not isinstance(configuration, AutomaticCommitmentSelectionConfig):
        raise ValueError(
            "configuration must be an AutomaticCommitmentSelectionConfig, got "
            f"{configuration!r}"
        )
    _require_finite_number(discount, "discount")
    _require_finite_number(tolerance, "tolerance")
    validate_deadline_parameters(horizon, discount, tolerance, max_horizon)
    _require_finite_number(
        configuration.minimum_total_uplift, "minimum_total_uplift"
    )
    if configuration.minimum_total_uplift < 0:
        raise ValueError(
            "minimum_total_uplift must be non-negative, got "
            f"{configuration.minimum_total_uplift!r}"
        )
    _validate_positive_int(configuration.max_candidates, "max_candidates")
    _validate_positive_int(configuration.max_timing_rows, "max_timing_rows")


def _validate_unique_ids(ids: Sequence[str], name: str) -> Tuple[str, ...]:
    if isinstance(ids, (str, bytes)):
        raise ValueError(
            f"{name} must be a sequence of non-empty string candidate ids, "
            "not a bare str or bytes container"
        )
    canonical: List[str] = []
    for candidate_id in ids:
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(
                f"{name} must contain non-empty string candidate ids, got "
                f"{candidate_id!r}"
            )
        canonical.append(candidate_id)
    duplicates = sorted(
        value for value, count in Counter(canonical).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate candidate_id(s) in {name}: {duplicates}")
    return tuple(sorted(canonical))


def _validate_fixed_profile_value(value: object, name: str) -> FixedProfileValue:
    if not isinstance(value, FixedProfileValue):
        raise ValueError(f"{name} must be a FixedProfileValue, got {value!r}")
    for field_name, scalar in (
        ("hero_ev", value.hero_ev),
        ("villain_ev", value.villain_ev),
        ("house_rake", value.house_rake),
    ):
        _require_finite_number(scalar, f"{name}.{field_name}")
    return value


def _validate_value_candidates(
    baseline_value: FixedProfileValue,
    candidates: Sequence[AutomaticCommitmentValueCandidate],
    *,
    tolerance: float,
    require_diff_consistency: bool,
) -> Tuple[AutomaticCommitmentValueCandidate, ...]:
    _validate_fixed_profile_value(baseline_value, "baseline_value")
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise ValueError(
            "candidates must be a sequence of AutomaticCommitmentValueCandidate"
        )
    candidate_ids = []
    for candidate in candidates:
        if not isinstance(candidate, AutomaticCommitmentValueCandidate):
            raise ValueError(
                "candidates must contain AutomaticCommitmentValueCandidate "
                f"objects, got {candidate!r}"
            )
        candidate_ids.append(candidate.candidate_id)
    _validate_unique_ids(candidate_ids, "candidates")

    for candidate in candidates:
        candidate_id = candidate.candidate_id
        _validate_fixed_profile_value(
            candidate.fixed_profile_value,
            f"candidate {candidate_id!r} fixed_profile_value",
        )
        for name, value in (
            ("post_response_hero_ev_worst", candidate.post_response_hero_ev_worst),
            ("post_response_hero_ev_best", candidate.post_response_hero_ev_best),
            (
                "post_response_hero_ev_worst_diff",
                candidate.post_response_hero_ev_worst_diff,
            ),
            ("l1_distance", candidate.l1_distance),
        ):
            _require_finite_number(value, f"candidate {candidate_id!r} {name}")
        if candidate.l1_distance < 0:
            raise ValueError(
                f"candidate {candidate_id!r} l1_distance must be non-negative, "
                f"got {candidate.l1_distance!r}"
            )
        if candidate.post_response_hero_ev_worst > candidate.post_response_hero_ev_best:
            raise ValueError(
                f"candidate {candidate_id!r} post-response Hero worst value "
                "must not exceed the best value"
            )
        if require_diff_consistency:
            expected_diff = (
                candidate.post_response_hero_ev_worst - baseline_value.hero_ev
            )
            _require_finite_number(
                expected_diff,
                f"candidate {candidate_id!r} expected post-response difference",
            )
            if abs(candidate.post_response_hero_ev_worst_diff - expected_diff) > tolerance:
                raise ValueError(
                    f"candidate {candidate_id!r} post-response Hero worst "
                    "difference is inconsistent with the baseline"
                )

    return tuple(sorted(candidates, key=lambda item: item.candidate_id))


def _validate_comparison_report(
    report: CandidateComparisonReport,
) -> Tuple[CandidateComparison, ...]:
    if not isinstance(report, CandidateComparisonReport):
        raise ValueError(
            "report must be a CandidateComparisonReport, got " f"{report!r}"
        )

    for name, value in (
        ("baseline_value.hero_ev", report.baseline_value.hero_ev),
        ("baseline_value.villain_ev", report.baseline_value.villain_ev),
        ("baseline_value.house_rake", report.baseline_value.house_rake),
    ):
        _require_finite_number(value, name)

    candidate_ids = [
        comparison.candidate.candidate_id for comparison in report.comparisons
    ]
    _validate_unique_ids(candidate_ids, "report.comparisons")

    for comparison in report.comparisons:
        candidate_id = comparison.candidate.candidate_id
        for name, value in (
            ("candidate.l1_distance", comparison.candidate.l1_distance),
            ("fixed_profile_value.hero_ev", comparison.fixed_profile_value.hero_ev),
            (
                "fixed_profile_value.villain_ev",
                comparison.fixed_profile_value.villain_ev,
            ),
            (
                "fixed_profile_value.house_rake",
                comparison.fixed_profile_value.house_rake,
            ),
            (
                "villain_ev_diff_from_baseline",
                comparison.villain_ev_diff_from_baseline,
            ),
            ("hero_ev_diff_from_baseline", comparison.hero_ev_diff_from_baseline),
            ("best_response.villain_max_ev", comparison.best_response.villain_max_ev),
            ("best_response.ev_h_worst", comparison.best_response.ev_h_worst),
            ("best_response.ev_h_best", comparison.best_response.ev_h_best),
            (
                "best_response.expected_house_rake_worst",
                comparison.best_response.expected_house_rake_worst,
            ),
            (
                "best_response.expected_house_rake_best",
                comparison.best_response.expected_house_rake_best,
            ),
            (
                "post_response_hero_ev_worst_diff",
                comparison.post_response_hero_ev_worst_diff,
            ),
            (
                "post_response_hero_ev_best_diff",
                comparison.post_response_hero_ev_best_diff,
            ),
        ):
            _require_finite_number(value, f"candidate {candidate_id!r} {name}")
        if comparison.candidate.l1_distance < 0:
            raise ValueError(
                f"candidate {candidate_id!r} l1_distance must be non-negative, "
                f"got {comparison.candidate.l1_distance!r}"
            )

    return tuple(sorted(report.comparisons, key=lambda item: item.candidate.candidate_id))


def _canonicalize_coverage(
    coverage: Optional[AutomaticCommitmentSearchCoverage],
    kept_candidate_ids: Tuple[str, ...],
) -> AutomaticCommitmentSearchCoverage:
    if coverage is None:
        return AutomaticCommitmentSearchCoverage(
            input_candidate_ids=kept_candidate_ids,
            kept_candidate_ids=kept_candidate_ids,
        )
    if not isinstance(coverage, AutomaticCommitmentSearchCoverage):
        raise ValueError(
            "search_coverage must be an AutomaticCommitmentSearchCoverage or "
            f"None, got {coverage!r}"
        )
    if not isinstance(coverage.source, str) or not coverage.source:
        raise ValueError("search_coverage.source must be a non-empty string")
    input_ids = _validate_unique_ids(
        coverage.input_candidate_ids, "search_coverage.input_candidate_ids"
    )
    declared_kept_ids = _validate_unique_ids(
        coverage.kept_candidate_ids, "search_coverage.kept_candidate_ids"
    )
    if declared_kept_ids != kept_candidate_ids:
        raise ValueError(
            "search_coverage.kept_candidate_ids must exactly match "
            "report.comparisons"
        )
    if not set(kept_candidate_ids).issubset(input_ids):
        raise ValueError(
            "search_coverage input candidate ids must contain every kept candidate"
        )
    if not isinstance(coverage.filtering_applied, bool):
        raise ValueError("search_coverage.filtering_applied must be a bool")

    shift_amounts: Optional[Tuple[float, ...]]
    if coverage.shift_amounts is None:
        shift_amounts = None
    else:
        shift_amounts = tuple(coverage.shift_amounts)
        for shift in shift_amounts:
            _require_finite_number(shift, "search_coverage.shift_amount")
            if shift <= 0:
                raise ValueError(
                    "search_coverage.shift_amounts must be strictly positive"
                )
        if coverage.max_simultaneous_info_sets not in (1, 2):
            raise ValueError(
                "search_coverage.max_simultaneous_info_sets must be 1 or 2"
            )
        _validate_positive_int(
            coverage.generation_max_candidates,
            "search_coverage.generation_max_candidates",
        )

    allowed_info_sets: Optional[Tuple[str, ...]]
    if coverage.filter_allowed_info_sets is None:
        allowed_info_sets = None
    else:
        raw_allowed_info_sets = coverage.filter_allowed_info_sets
        if isinstance(raw_allowed_info_sets, (str, bytes)):
            raise ValueError(
                "search_coverage.filter_allowed_info_sets must be a sequence of "
                "non-empty strings, not a bare str or bytes container"
            )
        canonical_allowed_info_sets: List[str] = []
        for value in raw_allowed_info_sets:
            if not isinstance(value, str) or not value:
                raise ValueError(
                    "search_coverage.filter_allowed_info_sets must contain only "
                    f"non-empty strings, got {value!r}"
                )
            canonical_allowed_info_sets.append(value)
        duplicate_info_sets = sorted(
            value
            for value, count in Counter(canonical_allowed_info_sets).items()
            if count > 1
        )
        if duplicate_info_sets:
            raise ValueError(
                "duplicate info set(s) in "
                "search_coverage.filter_allowed_info_sets: "
                f"{duplicate_info_sets}"
            )
        allowed_info_sets = tuple(sorted(canonical_allowed_info_sets))

    if coverage.filter_max_l1_distance is not None:
        _require_finite_number(
            coverage.filter_max_l1_distance,
            "search_coverage.filter_max_l1_distance",
        )
        if coverage.filter_max_l1_distance < 0:
            raise ValueError(
                "search_coverage.filter_max_l1_distance must be non-negative"
            )
    if coverage.filter_min_required_observations is not None:
        _validate_positive_int(
            coverage.filter_min_required_observations,
            "search_coverage.filter_min_required_observations",
        )

    return AutomaticCommitmentSearchCoverage(
        input_candidate_ids=input_ids,
        kept_candidate_ids=declared_kept_ids,
        source=coverage.source,
        shift_amounts=shift_amounts,
        max_simultaneous_info_sets=coverage.max_simultaneous_info_sets,
        generation_max_candidates=coverage.generation_max_candidates,
        filtering_applied=coverage.filtering_applied,
        filter_allowed_info_sets=allowed_info_sets,
        filter_max_l1_distance=coverage.filter_max_l1_distance,
        filter_min_required_observations=(
            coverage.filter_min_required_observations
        ),
    )


def _baseline_identity_from_value(baseline_value: FixedProfileValue) -> str:
    payload = {
        "algorithm": AUTOMATIC_COMMITMENT_BASELINE_IDENTITY_ALGORITHM,
        "baseline_value": baseline_value.to_dict(),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _validate_deadline_outputs(deadlines, horizon: int) -> None:
    for deadline in deadlines:
        if deadline.response_mode != RESPONSE_MODE_WORST:
            raise ValueError(
                "automatic selection requires response_mode='worst' deadline rows"
            )
        if len(deadline.result.timing) != horizon + 1:
            raise ValueError(
                f"candidate {deadline.candidate_id!r} returned an invalid timing "
                "row count"
            )
        for expected_opportunity, row in enumerate(deadline.result.timing, start=1):
            if row.adaptation_opportunity != expected_opportunity:
                raise ValueError(
                    f"candidate {deadline.candidate_id!r} returned non-canonical "
                    "adaptation opportunities"
                )
            _require_finite_number(
                row.locked_total_hero_ev,
                f"candidate {deadline.candidate_id!r} locked_total_hero_ev",
            )
            _require_finite_number(
                row.delta_from_baseline,
                f"candidate {deadline.candidate_id!r} delta_from_baseline",
            )


def _select_row(
    scores: Sequence[_CandidateScore],
    opportunity: int,
    minimum_total_uplift: float,
    tolerance: float,
) -> AutomaticCommitmentSelectionRow:
    if not scores:
        return AutomaticCommitmentSelectionRow(
            adaptation_opportunity=opportunity,
            status=NO_BENEFICIAL_COMMITMENT,
            best_total_hero_ev_delta=None,
            selected_candidate_id=None,
            primary_tie_candidate_ids=(),
            primary_tie_display_candidate_id=None,
            tie_break_evidence=(),
        )

    best_delta = max(score.total_hero_ev_delta for score in scores)
    primary_ties = tuple(
        score
        for score in scores
        if score.total_hero_ev_delta >= best_delta - tolerance
    )

    best_post_response_diff = max(
        score.candidate.post_response_hero_ev_worst_diff for score in primary_ties
    )
    post_response_ties = tuple(
        score
        for score in primary_ties
        if score.candidate.post_response_hero_ev_worst_diff
        >= best_post_response_diff - tolerance
    )
    best_l1_distance = min(
        score.candidate.l1_distance for score in post_response_ties
    )
    l1_ties = tuple(
        score
        for score in post_response_ties
        if score.candidate.l1_distance <= best_l1_distance + tolerance
    )
    display_candidate_id = min(
        score.candidate.candidate_id for score in l1_ties
    )

    post_response_ids = {
        score.candidate.candidate_id for score in post_response_ties
    }
    l1_ids = {score.candidate.candidate_id for score in l1_ties}
    evidence = tuple(
        AutomaticCommitmentTieBreakEvidence(
            candidate_id=score.candidate.candidate_id,
            total_hero_ev_delta=score.total_hero_ev_delta,
            post_response_hero_ev_worst_diff=(
                score.candidate.post_response_hero_ev_worst_diff
            ),
            l1_distance=score.candidate.l1_distance,
            within_best_post_response_tolerance=(
                score.candidate.candidate_id in post_response_ids
            ),
            within_best_l1_tolerance=(
                score.candidate.candidate_id in l1_ids
            ),
            is_primary_tie_display_candidate=(
                score.candidate.candidate_id == display_candidate_id
            ),
        )
        for score in primary_ties
    )

    beneficial = best_delta > minimum_total_uplift + tolerance
    return AutomaticCommitmentSelectionRow(
        adaptation_opportunity=opportunity,
        status=(
            AUTOMATIC_COMMITMENT_SELECTION_ROW_STATUS_SELECTED
            if beneficial
            else NO_BENEFICIAL_COMMITMENT
        ),
        best_total_hero_ev_delta=best_delta,
        selected_candidate_id=display_candidate_id if beneficial else None,
        primary_tie_candidate_ids=tuple(
            score.candidate.candidate_id for score in primary_ties
        ),
        primary_tie_display_candidate_id=display_candidate_id,
        tie_break_evidence=evidence,
    )


def _select_automatic_commitment_values_validated(
    baseline_value: FixedProfileValue,
    candidates: Tuple[AutomaticCommitmentValueCandidate, ...],
    *,
    horizon: int,
    discount: float,
    tolerance: float,
    max_horizon: int,
    configuration: AutomaticCommitmentSelectionConfig,
    coverage: AutomaticCommitmentSearchCoverage,
    timing_row_count: int,
) -> AutomaticCommitmentSelectionReport:
    deadlines = tuple(
        _CandidateValueDeadline(
            candidate_id=candidate.candidate_id,
            response_mode=RESPONSE_MODE_WORST,
            result=calculate_adaptation_deadline(
                baseline_hero_ev=baseline_value.hero_ev,
                pre_adaptation_hero_ev=candidate.fixed_profile_value.hero_ev,
                post_adaptation_hero_ev=candidate.post_response_hero_ev_worst,
                horizon=horizon,
                discount=discount,
                tolerance=tolerance,
                max_horizon=max_horizon,
            ),
        )
        for candidate in candidates
    )
    _validate_deadline_outputs(deadlines, horizon)
    deadlines_by_id = {deadline.candidate_id: deadline for deadline in deadlines}
    kept_candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
    if tuple(sorted(deadlines_by_id)) != kept_candidate_ids:
        raise ValueError(
            "automatic selection deadline candidate ids do not match the candidates"
        )

    rows: List[AutomaticCommitmentSelectionRow] = []
    for opportunity in range(1, horizon + 2):
        scores = tuple(
            _CandidateScore(
                candidate=candidate,
                total_hero_ev_delta=deadlines_by_id[
                    candidate.candidate_id
                ].result.timing[opportunity - 1].delta_from_baseline,
            )
            for candidate in candidates
        )
        rows.append(
            _select_row(
                scores,
                opportunity,
                configuration.minimum_total_uplift,
                tolerance,
            )
        )

    return AutomaticCommitmentSelectionReport(
        status=AUTOMATIC_COMMITMENT_SELECTION_STATUS_COMPLETE,
        baseline_identity=_baseline_identity_from_value(baseline_value),
        baseline_hero_ev=baseline_value.hero_ev,
        baseline_villain_ev=baseline_value.villain_ev,
        baseline_house_rake=baseline_value.house_rake,
        horizon=horizon,
        discount=discount,
        tolerance=tolerance,
        max_horizon=max_horizon,
        configuration=configuration,
        search_coverage=coverage,
        timing_row_evaluation_count=timing_row_count,
        rows=tuple(rows),
    )


def select_automatic_commitment_values(
    baseline_value: FixedProfileValue,
    candidates: Sequence[AutomaticCommitmentValueCandidate],
    *,
    horizon: int,
    discount: float = 1.0,
    tolerance: float = 1e-9,
    max_horizon: int = DEFAULT_MAX_HORIZON,
    configuration: AutomaticCommitmentSelectionConfig = (
        AutomaticCommitmentSelectionConfig()
    ),
    search_coverage: Optional[AutomaticCommitmentSearchCoverage] = None,
) -> AutomaticCommitmentSelectionReport:
    """Select commitments from validated domain-neutral scalar values.

    Domain adapters keep their native response representation and provide only
    the finite values required by the shared M27 selection kernel.  The input
    is bounded and fully validated before any repeated timing rows are built.
    """

    validate_automatic_commitment_selection_parameters(
        horizon=horizon,
        discount=discount,
        tolerance=tolerance,
        max_horizon=max_horizon,
        configuration=configuration,
    )
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise ValueError(
            "candidates must be a sequence of AutomaticCommitmentValueCandidate"
        )
    candidate_count = len(candidates)
    if candidate_count > configuration.max_candidates:
        raise ValueError(
            f"automatic selection has {candidate_count} kept candidates, exceeding "
            f"max_candidates={configuration.max_candidates}"
        )
    timing_row_count = candidate_count * (horizon + 1)
    if timing_row_count > configuration.max_timing_rows:
        raise ValueError(
            "automatic selection would materialise "
            f"{timing_row_count} candidate timing rows, exceeding "
            f"max_timing_rows={configuration.max_timing_rows}"
        )

    validated_candidates = _validate_value_candidates(
        baseline_value,
        candidates,
        tolerance=tolerance,
        require_diff_consistency=True,
    )
    kept_candidate_ids = tuple(
        candidate.candidate_id for candidate in validated_candidates
    )
    coverage = _canonicalize_coverage(search_coverage, kept_candidate_ids)

    return _select_automatic_commitment_values_validated(
        baseline_value,
        validated_candidates,
        horizon=horizon,
        discount=discount,
        tolerance=tolerance,
        max_horizon=max_horizon,
        configuration=configuration,
        coverage=coverage,
        timing_row_count=timing_row_count,
    )


def select_automatic_commitments(
    report: CandidateComparisonReport,
    *,
    horizon: int,
    discount: float = 1.0,
    tolerance: float = 1e-9,
    max_horizon: int = DEFAULT_MAX_HORIZON,
    configuration: AutomaticCommitmentSelectionConfig = (
        AutomaticCommitmentSelectionConfig()
    ),
    search_coverage: Optional[AutomaticCommitmentSearchCoverage] = None,
) -> AutomaticCommitmentSelectionReport:
    """Select the bounded Hero commitment conditionally for every ``m``.

    This public generic-tree adapter preserves the original validation and cap
    order, converts every comparison losslessly to the domain-neutral value
    contract, and delegates to the same selection kernel as domain adapters.
    """

    validate_automatic_commitment_selection_parameters(
        horizon=horizon,
        discount=discount,
        tolerance=tolerance,
        max_horizon=max_horizon,
        configuration=configuration,
    )
    if not isinstance(report, CandidateComparisonReport):
        raise ValueError(
            "report must be a CandidateComparisonReport, got " f"{report!r}"
        )
    candidate_count = len(report.comparisons)
    if candidate_count > configuration.max_candidates:
        raise ValueError(
            f"automatic selection has {candidate_count} kept candidates, exceeding "
            f"max_candidates={configuration.max_candidates}"
        )
    timing_row_count = candidate_count * (horizon + 1)
    if timing_row_count > configuration.max_timing_rows:
        raise ValueError(
            "automatic selection would materialise "
            f"{timing_row_count} candidate timing rows, exceeding "
            f"max_timing_rows={configuration.max_timing_rows}"
        )

    comparisons = _validate_comparison_report(report)
    kept_candidate_ids = tuple(
        comparison.candidate.candidate_id for comparison in comparisons
    )
    coverage = _canonicalize_coverage(search_coverage, kept_candidate_ids)
    value_candidates = tuple(
        AutomaticCommitmentValueCandidate(
            candidate_id=comparison.candidate.candidate_id,
            fixed_profile_value=comparison.fixed_profile_value,
            post_response_hero_ev_worst=comparison.best_response.ev_h_worst,
            post_response_hero_ev_best=comparison.best_response.ev_h_best,
            post_response_hero_ev_worst_diff=(
                comparison.post_response_hero_ev_worst_diff
            ),
            l1_distance=comparison.candidate.l1_distance,
        )
        for comparison in comparisons
    )

    return _select_automatic_commitment_values_validated(
        report.baseline_value,
        value_candidates,
        horizon=horizon,
        discount=discount,
        tolerance=tolerance,
        max_horizon=max_horizon,
        configuration=configuration,
        coverage=coverage,
        timing_row_count=timing_row_count,
    )
