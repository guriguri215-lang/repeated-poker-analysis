"""Lightweight pre-comparison pruning for generated Hero candidates.

``filter_candidates`` reduces a list of generated
:class:`~repeated_poker.candidates.HeroStrategyCandidate` objects *before* the
expensive comparison / selection stages.  It is a cheap pre-filter, not a
replacement for :func:`~repeated_poker.comparison.compare_candidates` or
:func:`~repeated_poker.selection.select_candidates`.

The optional detection-based filter uses ``local_v0`` by default, preserving the
historical local observable-distribution ``T_detect`` model.  When explicitly
asked to use ``reach_weighted_v1``, it applies the same public-observation
diagnostic used by the report path and interprets the minimum as a threshold on
finite per-hand ``t_detect_hands``.  Neither method models real opponent
learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

from .candidates import HeroStrategyCandidate
from .detection import (
    DEFAULT_MAX_DETECTION_TERMINALS,
    DETECTION_METHOD_LOCAL_V0,
    DETECTION_METHOD_REACH_WEIGHTED_V1,
    TerminalReveals,
    calculate_candidate_local_detection,
    calculate_candidate_reach_weighted_detection,
    resolve_detection_observation_model,
    validate_detection_parameters,
    validate_max_detection_terminals,
)
from .game import (
    GameTree,
    HeroStrategy,
    VillainStrategy,
    require_finite,
    require_valid_tolerance,
)
from .selection import L1_DISTANCE_EXCEEDS_LIMIT

# Exclusion-reason codes (stable English identifiers).  L1_DISTANCE_EXCEEDS_LIMIT
# is reused from :mod:`repeated_poker.selection` to avoid a duplicate definition.
INFO_SET_NOT_ALLOWED = "info_set_not_allowed"
REQUIRED_OBSERVATIONS_BELOW_LIMIT = "required_observations_below_limit"


@dataclass(frozen=True)
class ExcludedGeneratedCandidate:
    """A generated candidate removed by the pre-filter, with its reasons."""

    candidate: HeroStrategyCandidate
    reasons: List[str]

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate.candidate_id,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class CandidateFilterSummaryCounts:
    """Counts of candidates kept and excluded by the pre-filter."""

    total: int
    kept: int
    excluded: int

    def to_dict(self) -> dict:
        return {"total": self.total, "kept": self.kept, "excluded": self.excluded}


@dataclass(frozen=True)
class CandidateFilterResult:
    """The kept and excluded candidates from a pre-filter pass."""

    kept: List[HeroStrategyCandidate]
    excluded: List[ExcludedGeneratedCandidate]
    summary_counts: CandidateFilterSummaryCounts

    def to_dict(self) -> dict:
        return {
            "summary_counts": self.summary_counts.to_dict(),
            "kept_candidate_ids": [c.candidate_id for c in self.kept],
            "excluded": [e.to_dict() for e in self.excluded],
        }


def _validate_allowed_info_sets(allowed_info_sets) -> Optional[Set[str]]:
    if allowed_info_sets is None:
        return None
    # A bare string is iterable but is almost never the intended set of info
    # sets (it would be split into its characters), so reject it explicitly.
    if isinstance(allowed_info_sets, (str, bytes)):
        raise ValueError(
            "allowed_info_sets must be a collection of strings, not a bare "
            f"string; got {allowed_info_sets!r}"
        )
    try:
        allowed = set(allowed_info_sets)
    except TypeError:
        raise ValueError(
            "allowed_info_sets must be an iterable of strings or None, got "
            f"{allowed_info_sets!r}"
        ) from None
    for value in allowed:
        if not isinstance(value, str):
            raise ValueError(
                f"allowed_info_sets must contain only strings, got {value!r}"
            )
    return allowed


def filter_candidates(
    candidates: Sequence[HeroStrategyCandidate],
    *,
    allowed_info_sets: Optional[Set[str]] = None,
    max_l1_distance: Optional[float] = None,
    min_required_observations: Optional[int] = None,
    baseline_hero_strategy: Optional[HeroStrategy] = None,
    tree: Optional[GameTree] = None,
    baseline_villain_strategy: Optional[VillainStrategy] = None,
    detection_log_likelihood_threshold: Optional[float] = None,
    detection_method: str = DETECTION_METHOD_LOCAL_V0,
    detection_observation_model: Optional[str] = None,
    terminal_reveals: Optional[TerminalReveals] = None,
    max_detection_terminals: int = DEFAULT_MAX_DETECTION_TERMINALS,
    tolerance: float = 1e-9,
) -> CandidateFilterResult:
    """Pre-filter generated candidates, returning kept and excluded sets.

    Filters applied per candidate (a candidate may collect several reasons):

    * ``allowed_info_sets`` (when given): exclude a candidate whose ``info_set``
      is not in the set -> ``INFO_SET_NOT_ALLOWED``.  An empty set excludes all.
    * ``max_l1_distance`` (when given): exclude when ``l1_distance`` exceeds it
      beyond ``tolerance`` -> ``L1_DISTANCE_EXCEEDS_LIMIT``.
    * ``min_required_observations`` (when given): compute the candidate's
      ``T_detect`` with ``detection_method`` and exclude finite estimates below
      the limit -> ``REQUIRED_OBSERVATIONS_BELOW_LIMIT``.  For ``local_v0`` the
      estimate is ``required_observations``; for ``reach_weighted_v1`` it is the
      per-hand ``t_detect_hands``.  ``None`` (no signal under the selected model)
      is never excluded.

    Using ``min_required_observations`` requires both ``baseline_hero_strategy``
    and ``detection_log_likelihood_threshold``.  ``reach_weighted_v1`` also
    requires ``tree`` and ``baseline_villain_strategy``.  Input order is
    preserved, the candidate objects are never mutated, and ``kept`` plus
    ``excluded`` always sum to the input total.
    """

    require_valid_tolerance(tolerance)
    allowed = _validate_allowed_info_sets(allowed_info_sets)
    resolved_observation_model = resolve_detection_observation_model(
        detection_method, detection_observation_model
    )
    validate_max_detection_terminals(max_detection_terminals)

    if max_l1_distance is not None:
        require_finite(max_l1_distance, "max_l1_distance")
        if max_l1_distance < 0:
            raise ValueError(
                f"max_l1_distance must be non-negative, got {max_l1_distance!r}"
            )

    detection_enabled = min_required_observations is not None
    if detection_enabled:
        if isinstance(min_required_observations, bool) or not isinstance(
            min_required_observations, int
        ):
            raise ValueError(
                "min_required_observations must be a positive integer, got "
                f"{min_required_observations!r}"
            )
        if min_required_observations < 1:
            raise ValueError(
                "min_required_observations must be at least 1, got "
                f"{min_required_observations}"
            )
        if baseline_hero_strategy is None:
            raise ValueError(
                "baseline_hero_strategy is required when min_required_observations "
                "is given"
            )
        if detection_log_likelihood_threshold is None:
            raise ValueError(
                "detection_log_likelihood_threshold is required when "
                "min_required_observations is given"
            )
        if detection_method == DETECTION_METHOD_REACH_WEIGHTED_V1:
            if tree is None:
                raise ValueError(
                    "tree is required when detection_method='reach_weighted_v1' "
                    "and min_required_observations is given"
                )
            if baseline_villain_strategy is None:
                raise ValueError(
                    "baseline_villain_strategy is required when "
                    "detection_method='reach_weighted_v1' and "
                    "min_required_observations is given"
                )
        validate_detection_parameters(detection_log_likelihood_threshold, None, tolerance)

    kept: List[HeroStrategyCandidate] = []
    excluded: List[ExcludedGeneratedCandidate] = []

    for candidate in candidates:
        reasons: List[str] = []

        # A multi-shift candidate changes several information sets; it is allowed
        # only when every changed information set is in ``allowed`` (a single-shift
        # candidate has exactly one, so this matches the original behaviour).
        if allowed is not None and any(
            info_set not in allowed for info_set in candidate.info_sets
        ):
            reasons.append(INFO_SET_NOT_ALLOWED)

        if (
            max_l1_distance is not None
            and candidate.l1_distance > max_l1_distance + tolerance
        ):
            reasons.append(L1_DISTANCE_EXCEEDS_LIMIT)

        if detection_enabled:
            if detection_method == DETECTION_METHOD_REACH_WEIGHTED_V1:
                detection = calculate_candidate_reach_weighted_detection(
                    tree,
                    baseline_hero_strategy,
                    candidate,
                    baseline_villain_strategy,
                    log_likelihood_threshold=detection_log_likelihood_threshold,
                    observation_model=resolved_observation_model,
                    terminal_reveals=terminal_reveals,
                    max_detection_terminals=max_detection_terminals,
                    tolerance=tolerance,
                )
                required = detection.t_detect_hands
            else:
                detection = calculate_candidate_local_detection(
                    baseline_hero_strategy,
                    candidate,
                    log_likelihood_threshold=detection_log_likelihood_threshold,
                    tolerance=tolerance,
                )
                required = detection.required_observations
            if required is not None and required < min_required_observations:
                reasons.append(REQUIRED_OBSERVATIONS_BELOW_LIMIT)

        if reasons:
            excluded.append(
                ExcludedGeneratedCandidate(candidate=candidate, reasons=reasons)
            )
        else:
            kept.append(candidate)

    summary_counts = CandidateFilterSummaryCounts(
        total=len(candidates), kept=len(kept), excluded=len(excluded)
    )
    return CandidateFilterResult(
        kept=kept, excluded=excluded, summary_counts=summary_counts
    )
