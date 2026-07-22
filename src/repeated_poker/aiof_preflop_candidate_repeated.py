"""Exact real-card AIoF candidate, response, and repeated-value bridge.

This module composes the existing M13 real-card supplied-profile APIs with the
M27 domain-neutral automatic selector.  It searches a declared, bounded set of
one- or two-exact-combo probability shifts in heads-up fee-zero net ChipEV.
It does not solve for an endogenous baseline, certify equilibrium, optimize a
continuous/global strategy space, model ICM, or make a profitability claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from numbers import Real
from typing import Any, Optional, Tuple

from .aiof_cards import (
    AiofContractError,
    AiofLimits,
    AiofStatus,
    PreparedRanges,
    RangeSpec,
    canonicalize_exact_combo,
    prepare_compatible_ranges,
)
from .aiof_chip_ev import (
    ComboActionProbability,
    HeadsUpChipEvGame,
    PushFoldAnalysis,
    PushFoldRequest,
    SuppliedProfile,
    UnilateralBestResponse,
    analyze_pushfold,
)
from .aiof_equity import EquityAlgorithm
from .automatic_commitment_selection import (
    DEFAULT_MAX_AUTOMATIC_TIMING_ROWS,
    AutomaticCommitmentSearchCoverage,
    AutomaticCommitmentSelectionConfig,
    AutomaticCommitmentSelectionReport,
    AutomaticCommitmentValueCandidate,
    select_automatic_commitment_values,
    validate_automatic_commitment_selection_parameters,
)
from .candidates import DEFAULT_MAX_CANDIDATES
from .fixed_profile import FixedProfileValue
from .repeated import DEFAULT_MAX_HORIZON


__all__ = [
    "AIOF_PREFLOP_CANDIDATE_REPEATED_CONTRACT_VERSION",
    "AiofPreflopCandidateRepeatedLimits",
    "AiofPreflopCandidateRepeatedRequest",
    "AiofPreflopShift",
    "AiofPreflopWorkloadProjection",
    "AiofPreflopCandidateCoverage",
    "AiofPreflopBaselineRecord",
    "AiofPreflopCandidateRecord",
    "AiofPreflopCandidateRepeatedPayload",
    "AiofPreflopCandidateRepeatedResult",
    "analyze_aiof_preflop_candidate_repeated",
]


AIOF_PREFLOP_CANDIDATE_REPEATED_CONTRACT_VERSION = (
    "aiof-preflop-candidate-repeated-v1"
)
AIOF_PREFLOP_BASELINE_IDENTITY_ALGORITHM = (
    "aiof-preflop-baseline-sha256-canonical-json-v1"
)
AIOF_PREFLOP_CANDIDATE_IDENTITY_ALGORITHM = (
    "aiof-preflop-candidate-sha256-canonical-json-v1"
)
AIOF_PREFLOP_ANALYSIS_IDENTITY_ALGORITHM = (
    "aiof-preflop-analysis-sha256-canonical-json-v1"
)
AIOF_PREFLOP_UTILITY_CONTRACT = "heads-up-fee0-net-chipev-v1"
AIOF_PREFLOP_UNIT = "net_chip_delta_before_mandatory_posts"
AIOF_PREFLOP_CLAIM_SCOPE = (
    "bounded_exact_combo_candidate_library_adaptation_opportunity_conditional_decision"
)

MAX_BRIDGE_CANDIDATES = DEFAULT_MAX_CANDIDATES
MAX_BRIDGE_TOTAL_BOARD_EVALUATIONS = 10_000_000
MAX_BRIDGE_RESPONSE_ROWS = 1_000_000
MAX_BRIDGE_TIMING_ROWS = DEFAULT_MAX_AUTOMATIC_TIMING_ROWS


@dataclass(frozen=True)
class AiofPreflopCandidateRepeatedLimits:
    """Caller-lowerable bridge caps checked before candidate materialization."""

    max_candidates: int = MAX_BRIDGE_CANDIDATES
    max_total_board_evaluations: int = MAX_BRIDGE_TOTAL_BOARD_EVALUATIONS
    max_response_rows: int = MAX_BRIDGE_RESPONSE_ROWS
    max_timing_rows: int = MAX_BRIDGE_TIMING_ROWS

    def to_dict(self) -> dict:
        return {
            "max_candidates": self.max_candidates,
            "max_total_board_evaluations": self.max_total_board_evaluations,
            "max_response_rows": self.max_response_rows,
            "max_timing_rows": self.max_timing_rows,
        }


@dataclass(frozen=True)
class AiofPreflopCandidateRepeatedRequest:
    """One exact in-memory real-card candidate/repeated analysis request."""

    game: HeadsUpChipEvGame
    sb_range: RangeSpec
    bb_range: RangeSpec
    dead_cards: tuple[str, ...]
    baseline_profile: SuppliedProfile
    hero_seat: str
    shift_amounts: tuple[float, ...]
    max_shifted_combos: int
    horizon: int
    discount: float = 1.0
    tolerance: float = 1e-9
    minimum_total_uplift: float = 0.0
    aiof_limits: AiofLimits = AiofLimits()
    bridge_limits: AiofPreflopCandidateRepeatedLimits = (
        AiofPreflopCandidateRepeatedLimits()
    )
    max_horizon: int = DEFAULT_MAX_HORIZON
    algorithm: EquityAlgorithm = EquityAlgorithm.EXACT_EXHAUSTIVE
    seed: int | None = None
    samples: int | None = None
    expected_baseline_identity: str | None = None


@dataclass(frozen=True)
class AiofPreflopShift:
    """One canonical probability shift on an exact Hero combo."""

    combo: str
    source_action: str
    target_action: str
    shift_amount: float
    baseline_active_probability: float
    candidate_active_probability: float

    def to_dict(self) -> dict:
        return {
            "combo": self.combo,
            "source_action": self.source_action,
            "target_action": self.target_action,
            "shift_amount": self.shift_amount,
            "baseline_active_probability": self.baseline_active_probability,
            "candidate_active_probability": self.candidate_active_probability,
        }


@dataclass(frozen=True)
class AiofPreflopWorkloadProjection:
    """Integer workload projection completed before candidate construction."""

    feasible_single_shift_count: int
    feasible_shift_counts_by_combo: tuple[tuple[str, int], ...]
    candidate_count: int
    exact_board_evaluations_per_analysis: int
    total_exact_board_evaluations: int
    opponent_support_count: int
    response_row_count: int
    timing_row_count: int

    def to_dict(self) -> dict:
        return {
            "feasible_single_shift_count": self.feasible_single_shift_count,
            "feasible_shift_counts_by_combo": [
                {"combo": combo, "count": count}
                for combo, count in self.feasible_shift_counts_by_combo
            ],
            "candidate_count": self.candidate_count,
            "exact_board_evaluations_per_analysis": (
                self.exact_board_evaluations_per_analysis
            ),
            "total_exact_board_evaluations": self.total_exact_board_evaluations,
            "opponent_support_count": self.opponent_support_count,
            "response_row_count": self.response_row_count,
            "timing_row_count": self.timing_row_count,
        }


@dataclass(frozen=True)
class AiofPreflopCandidateCoverage:
    """Complete unfiltered coverage of the declared candidate universe."""

    generated_candidate_ids: tuple[str, ...]
    kept_candidate_ids: tuple[str, ...]
    shift_amounts: tuple[float, ...]
    max_shifted_combos: int
    filtering_applied: bool = False

    def to_dict(self) -> dict:
        return {
            "generated_candidate_count": len(self.generated_candidate_ids),
            "generated_candidate_ids": list(self.generated_candidate_ids),
            "kept_candidate_count": len(self.kept_candidate_ids),
            "kept_candidate_ids": list(self.kept_candidate_ids),
            "shift_amounts": list(self.shift_amounts),
            "max_shifted_combos": self.max_shifted_combos,
            "filtering_applied": self.filtering_applied,
        }


@dataclass(frozen=True)
class AiofPreflopBaselineRecord:
    """Baseline supplied profile, exact value, and native opponent response."""

    profile_identity: str
    profile: SuppliedProfile
    analysis_input_identity: str
    fixed_profile_value: FixedProfileValue
    opponent_response: UnilateralBestResponse
    baseline_hero_ev: float

    def to_dict(self) -> dict:
        return {
            "profile_identity": self.profile_identity,
            "profile": _profile_to_dict(self.profile),
            "analysis_input_identity": self.analysis_input_identity,
            "fixed_profile_value": self.fixed_profile_value.to_dict(),
            "opponent_response": _response_to_dict(self.opponent_response),
            "b": self.baseline_hero_ev,
        }


@dataclass(frozen=True)
class AiofPreflopCandidateRecord:
    """One full candidate profile and its exact native response provenance."""

    candidate_id: str
    shifts: tuple[AiofPreflopShift, ...]
    l1_distance: float
    profile_identity: str
    profile: SuppliedProfile
    analysis_input_identity: str
    fixed_profile_value: FixedProfileValue
    opponent_response: UnilateralBestResponse
    baseline_hero_ev: float
    fixed_opponent_hero_ev: float
    post_response_hero_ev_worst: float
    post_response_hero_ev_best: float
    post_response_hero_ev_worst_diff: float

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "shifts": [shift.to_dict() for shift in self.shifts],
            "l1_distance": self.l1_distance,
            "profile_identity": self.profile_identity,
            "profile": _profile_to_dict(self.profile),
            "analysis_input_identity": self.analysis_input_identity,
            "fixed_profile_value": self.fixed_profile_value.to_dict(),
            "factorized_opponent_response": _response_to_dict(
                self.opponent_response
            ),
            "values": {
                "b": self.baseline_hero_ev,
                "a": self.fixed_opponent_hero_ev,
                "l_worst": self.post_response_hero_ev_worst,
                "l_best": self.post_response_hero_ev_best,
                "post_response_hero_ev_worst_diff": (
                    self.post_response_hero_ev_worst_diff
                ),
            },
        }


@dataclass(frozen=True)
class AiofPreflopCandidateRepeatedPayload:
    """Complete successful bridge payload; failures return no instance."""

    hero_seat: str
    opponent_seat: str
    hero_active_action: str
    opponent_active_action: str
    prepared_ranges: PreparedRanges
    baseline_identity: str
    analysis_identity: str
    baseline: AiofPreflopBaselineRecord
    workload: AiofPreflopWorkloadProjection
    coverage: AiofPreflopCandidateCoverage
    candidates: tuple[AiofPreflopCandidateRecord, ...]
    automatic_selection: AutomaticCommitmentSelectionReport
    request: AiofPreflopCandidateRepeatedRequest

    def to_dict(self) -> dict:
        return {
            "contract_version": AIOF_PREFLOP_CANDIDATE_REPEATED_CONTRACT_VERSION,
            "claim_scope": AIOF_PREFLOP_CLAIM_SCOPE,
            "accounting": AIOF_PREFLOP_UTILITY_CONTRACT,
            "unit": AIOF_PREFLOP_UNIT,
            "algorithm": self.request.algorithm.value,
            "hero_seat": self.hero_seat,
            "opponent_seat": self.opponent_seat,
            "hero_active_action": self.hero_active_action,
            "opponent_active_action": self.opponent_active_action,
            "prepared_ranges": _prepared_ranges_to_dict(self.prepared_ranges),
            "baseline_identity_algorithm": (
                AIOF_PREFLOP_BASELINE_IDENTITY_ALGORITHM
            ),
            "baseline_identity": self.baseline_identity,
            "analysis_identity_algorithm": (
                AIOF_PREFLOP_ANALYSIS_IDENTITY_ALGORITHM
            ),
            "analysis_identity": self.analysis_identity,
            "baseline": self.baseline.to_dict(),
            "candidate_count_projection": self.workload.candidate_count,
            "candidate_count": len(self.candidates),
            "workload_projection": self.workload.to_dict(),
            "coverage": self.coverage.to_dict(),
            "caps": {
                "aiof": _dataclass_plain_dict(self.request.aiof_limits),
                "bridge": self.request.bridge_limits.to_dict(),
                "max_horizon": self.request.max_horizon,
            },
            "repeated_configuration": {
                "horizon": self.request.horizon,
                "discount": self.request.discount,
                "tolerance": self.request.tolerance,
                "minimum_total_uplift": self.request.minimum_total_uplift,
            },
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "automatic_selection": self.automatic_selection.to_dict(),
        }


@dataclass(frozen=True)
class AiofPreflopCandidateRepeatedResult:
    """Existing-status no-partial wrapper for the public bridge."""

    status: AiofStatus
    payload: AiofPreflopCandidateRepeatedPayload | None
    error_message: str | None

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "payload": None if self.payload is None else self.payload.to_dict(),
            "error": self.error_message,
        }


@dataclass(frozen=True)
class _MaterializedCandidate:
    candidate_id: str
    shifts: tuple[AiofPreflopShift, ...]
    l1_distance: float
    profile_identity: str
    profile: SuppliedProfile


def _require_finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, f"{name} must be a finite number"
        )
    number = float(value)
    if not math.isfinite(number):
        raise AiofContractError(AiofStatus.INVALID_INPUT, f"{name} must be finite")
    return number


def _validate_positive_cap(value: object, name: str, ceiling: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, f"{name} must be a positive integer"
        )
    if value > ceiling:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            f"{name}={value} exceeds hard ceiling {ceiling}",
        )
    return value


def _validate_bridge_limits(limits: object) -> AiofPreflopCandidateRepeatedLimits:
    if not isinstance(limits, AiofPreflopCandidateRepeatedLimits):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            "bridge_limits must be AiofPreflopCandidateRepeatedLimits",
        )
    _validate_positive_cap(
        limits.max_candidates, "max_candidates", MAX_BRIDGE_CANDIDATES
    )
    _validate_positive_cap(
        limits.max_total_board_evaluations,
        "max_total_board_evaluations",
        MAX_BRIDGE_TOTAL_BOARD_EVALUATIONS,
    )
    _validate_positive_cap(
        limits.max_response_rows,
        "max_response_rows",
        MAX_BRIDGE_RESPONSE_ROWS,
    )
    _validate_positive_cap(
        limits.max_timing_rows,
        "max_timing_rows",
        MAX_BRIDGE_TIMING_ROWS,
    )
    return limits


def _canonical_identity_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AiofContractError(
                AiofStatus.NUMERIC_FAILURE, "identity contains non-finite float"
            )
        return {"float_hex": value.hex()}
    if is_dataclass(value):
        return {
            field.name: _canonical_identity_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {
            str(key): _canonical_identity_value(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_identity_value(item) for item in value]
    return value


def _sha256_identity(payload: object) -> str:
    encoded = json.dumps(
        _canonical_identity_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _dataclass_plain_dict(value: object) -> dict:
    if not is_dataclass(value):
        raise TypeError("value must be a dataclass instance")
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _profile_to_dict(profile: SuppliedProfile) -> dict:
    return {
        "sb_shove": [
            {"combo": row.combo, "probability": row.probability}
            for row in profile.sb_shove
        ],
        "bb_call": [
            {"combo": row.combo, "probability": row.probability}
            for row in profile.bb_call
        ],
    }


def _response_to_dict(response: UnilateralBestResponse) -> dict:
    return {
        "seat": response.seat,
        "supplied_profile_value": response.supplied_profile_value,
        "best_response_value": response.best_response_value,
        "raw_gain": response.raw_gain,
        "rows": [
            {
                "combo": row.combo,
                "compatible_probability": row.compatible_probability,
                "information_reach_probability": row.information_reach_probability,
                "action_values": [
                    {"action": action, "value": value}
                    for action, value in row.action_values
                ],
                "best_actions": list(row.best_actions),
                "supplied_action_probability": row.supplied_action_probability,
                "raw_gain": row.raw_gain,
            }
            for row in response.rows
        ],
    }


def _prepared_ranges_to_dict(prepared: PreparedRanges) -> dict:
    return {
        "identity": prepared.content_identity,
        "dead_cards": list(prepared.dead_cards),
        "compatible_pair_count": prepared.compatible_pair_count,
        "compatible_raw_joint_mass": prepared.compatible_raw_joint_mass,
        "sb": {
            "range_identity": prepared.sb_range.content_identity,
            "support": [item.combo for item in prepared.sb_range.combos],
            "marginals": [
                {"combo": item.combo, "probability": item.probability}
                for item in prepared.sb_marginals
            ],
        },
        "bb": {
            "range_identity": prepared.bb_range.content_identity,
            "support": [item.combo for item in prepared.bb_range.combos],
            "marginals": [
                {"combo": item.combo, "probability": item.probability}
                for item in prepared.bb_marginals
            ],
        },
    }


def _canonical_strategy_rows(
    rows: object, expected_combos: tuple[str, ...], name: str
) -> tuple[ComboActionProbability, ...]:
    if not isinstance(rows, tuple):
        raise AiofContractError(
            AiofStatus.INVALID_STRATEGY, f"{name} must be a tuple"
        )
    probabilities: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, ComboActionProbability):
            raise AiofContractError(
                AiofStatus.INVALID_STRATEGY, f"{name} contains an invalid row"
            )
        try:
            combo = canonicalize_exact_combo(row.combo)
        except AiofContractError as exc:
            raise AiofContractError(AiofStatus.INVALID_STRATEGY, str(exc)) from exc
        try:
            probability = _require_finite_number(row.probability, f"{name}[{combo}]")
        except AiofContractError as exc:
            raise AiofContractError(AiofStatus.INVALID_STRATEGY, str(exc)) from exc
        if not 0.0 <= probability <= 1.0:
            raise AiofContractError(
                AiofStatus.INVALID_STRATEGY,
                f"{name}[{combo}] probability must be in [0, 1]",
            )
        if combo in probabilities:
            raise AiofContractError(
                AiofStatus.INVALID_STRATEGY, f"duplicate strategy combo {combo}"
            )
        probabilities[combo] = probability
    if set(probabilities) != set(expected_combos):
        missing = sorted(set(expected_combos) - set(probabilities))
        extra = sorted(set(probabilities) - set(expected_combos))
        raise AiofContractError(
            AiofStatus.INVALID_STRATEGY,
            f"strategy support mismatch; missing={missing}, extra={extra}",
        )
    return tuple(
        ComboActionProbability(combo, probabilities[combo])
        for combo in expected_combos
    )


def _canonical_profile(
    profile: object, prepared: PreparedRanges
) -> SuppliedProfile:
    if not isinstance(profile, SuppliedProfile):
        raise AiofContractError(
            AiofStatus.INVALID_STRATEGY, "baseline_profile must be SuppliedProfile"
        )
    sb_expected = tuple(item.combo for item in prepared.sb_marginals)
    bb_expected = tuple(item.combo for item in prepared.bb_marginals)
    return SuppliedProfile(
        _canonical_strategy_rows(profile.sb_shove, sb_expected, "sb_shove"),
        _canonical_strategy_rows(profile.bb_call, bb_expected, "bb_call"),
    )


def _canonical_shift_amounts(value: object) -> tuple[float, ...]:
    if not isinstance(value, tuple):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, "shift_amounts must be a tuple"
        )
    shifts = tuple(_require_finite_number(item, "shift_amount") for item in value)
    if any(item <= 0.0 for item in shifts):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, "shift amounts must be strictly positive"
        )
    if len(set(shifts)) != len(shifts):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, "shift amounts must be unique"
        )
    return tuple(sorted(shifts))


def _validate_request(
    request: object,
) -> tuple[
    AiofPreflopCandidateRepeatedRequest,
    tuple[float, ...],
    AutomaticCommitmentSelectionConfig,
]:
    if not isinstance(request, AiofPreflopCandidateRepeatedRequest):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            "request must be AiofPreflopCandidateRepeatedRequest",
        )
    if not isinstance(request.game, HeadsUpChipEvGame):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, "game must be HeadsUpChipEvGame"
        )
    if not isinstance(request.sb_range, RangeSpec) or not isinstance(
        request.bb_range, RangeSpec
    ):
        raise AiofContractError(
            AiofStatus.INVALID_RANGE, "sb_range and bb_range must be RangeSpec"
        )
    if not isinstance(request.dead_cards, tuple):
        raise AiofContractError(
            AiofStatus.INVALID_CARD_INPUT, "dead_cards must be a tuple"
        )
    if not isinstance(request.aiof_limits, AiofLimits):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, "aiof_limits must be AiofLimits"
        )
    bridge_limits = _validate_bridge_limits(request.bridge_limits)
    if request.hero_seat not in ("sb", "bb"):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, "hero_seat must be 'sb' or 'bb'"
        )
    if isinstance(request.max_shifted_combos, bool) or request.max_shifted_combos not in (
        1,
        2,
    ):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, "max_shifted_combos must be 1 or 2"
        )
    if request.algorithm is not EquityAlgorithm.EXACT_EXHAUSTIVE:
        raise AiofContractError(
            AiofStatus.UNSUPPORTED_MODEL,
            "v1 supports exact exhaustive equity only",
        )
    if request.seed is not None or request.samples is not None:
        raise AiofContractError(
            AiofStatus.UNSUPPORTED_MODEL,
            "v1 does not accept seed or samples controls",
        )
    if request.expected_baseline_identity is not None and (
        not isinstance(request.expected_baseline_identity, str)
        or not request.expected_baseline_identity
    ):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            "expected_baseline_identity must be a non-empty string or None",
        )
    shifts = _canonical_shift_amounts(request.shift_amounts)
    configuration = AutomaticCommitmentSelectionConfig(
        minimum_total_uplift=request.minimum_total_uplift,
        max_candidates=bridge_limits.max_candidates,
        max_timing_rows=bridge_limits.max_timing_rows,
    )
    try:
        validate_automatic_commitment_selection_parameters(
            horizon=request.horizon,
            discount=request.discount,
            tolerance=request.tolerance,
            max_horizon=request.max_horizon,
            configuration=configuration,
        )
    except ValueError as exc:
        raise AiofContractError(AiofStatus.INVALID_INPUT, str(exc)) from exc
    return request, shifts, configuration


def _active_probability_map(
    profile: SuppliedProfile, hero_seat: str
) -> dict[str, float]:
    rows = profile.sb_shove if hero_seat == "sb" else profile.bb_call
    return {row.combo: row.probability for row in rows}


def _feasible_shift_counts(
    active_probabilities: dict[str, float], shift_amounts: tuple[float, ...]
) -> tuple[tuple[str, int], ...]:
    counts = []
    for combo in sorted(active_probabilities):
        probability = active_probabilities[combo]
        count = 0
        for shift in shift_amounts:
            if probability + shift <= 1.0:
                count += 1
            if probability - shift >= 0.0:
                count += 1
        counts.append((combo, count))
    return tuple(counts)


def _project_candidate_count(
    counts: tuple[tuple[str, int], ...], max_shifted_combos: int
) -> tuple[int, int]:
    single_count = sum(count for _, count in counts)
    if max_shifted_combos == 1:
        return single_count, single_count
    sum_squares = sum(count * count for _, count in counts)
    return single_count, single_count + (
        single_count * single_count - sum_squares
    ) // 2


def _exact_board_evaluations_per_analysis(prepared: PreparedRanges) -> int:
    remaining = 52 - len(prepared.dead_card_ids) - 4
    if remaining < 5:
        raise AiofContractError(
            AiofStatus.INVALID_CARD_INPUT, "fewer than five board cards remain"
        )
    return prepared.compatible_pair_count * math.comb(remaining, 5)


def _project_workload(
    request: AiofPreflopCandidateRepeatedRequest,
    prepared: PreparedRanges,
    counts: tuple[tuple[str, int], ...],
) -> AiofPreflopWorkloadProjection:
    single_count, candidate_count = _project_candidate_count(
        counts, request.max_shifted_combos
    )
    if candidate_count > request.bridge_limits.max_candidates:
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            f"projected {candidate_count} candidates exceed max_candidates",
        )
    per_analysis = _exact_board_evaluations_per_analysis(prepared)
    if per_analysis > request.aiof_limits.max_exact_board_evaluations:
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            "projected exact board evaluations per analysis exceed the M13 cap",
        )
    total_boards = (candidate_count + 1) * per_analysis
    if total_boards > request.bridge_limits.max_total_board_evaluations:
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            "projected total exact board evaluations exceed the bridge cap",
        )
    opponent_support_count = (
        len(prepared.bb_marginals)
        if request.hero_seat == "sb"
        else len(prepared.sb_marginals)
    )
    response_rows = (candidate_count + 1) * opponent_support_count
    if response_rows > request.bridge_limits.max_response_rows:
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            "projected retained response rows exceed the bridge cap",
        )
    timing_rows = candidate_count * (request.horizon + 1)
    if timing_rows > request.bridge_limits.max_timing_rows:
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            "projected selector timing rows exceed the bridge cap",
        )
    return AiofPreflopWorkloadProjection(
        feasible_single_shift_count=single_count,
        feasible_shift_counts_by_combo=counts,
        candidate_count=candidate_count,
        exact_board_evaluations_per_analysis=per_analysis,
        total_exact_board_evaluations=total_boards,
        opponent_support_count=opponent_support_count,
        response_row_count=response_rows,
        timing_row_count=timing_rows,
    )


def _baseline_identity(
    request: AiofPreflopCandidateRepeatedRequest,
    prepared: PreparedRanges,
    profile: SuppliedProfile,
) -> str:
    return _sha256_identity(
        {
            "algorithm": AIOF_PREFLOP_BASELINE_IDENTITY_ALGORITHM,
            "contract_version": AIOF_PREFLOP_CANDIDATE_REPEATED_CONTRACT_VERSION,
            "game": request.game,
            "accounting": AIOF_PREFLOP_UTILITY_CONTRACT,
            "unit": AIOF_PREFLOP_UNIT,
            "sb_prepared_range_identity": prepared.sb_range.content_identity,
            "bb_prepared_range_identity": prepared.bb_range.content_identity,
            "sb_support": tuple(item.combo for item in prepared.sb_range.combos),
            "bb_support": tuple(item.combo for item in prepared.bb_range.combos),
            "dead_cards": prepared.dead_cards,
            "compatible_pair_count": prepared.compatible_pair_count,
            "hero_seat": request.hero_seat,
            "baseline_profile": profile,
            "equity_algorithm": request.algorithm,
        }
    )


def _shift_sort_key(shift: AiofPreflopShift) -> tuple[str, str, str, str]:
    return (
        shift.combo,
        shift.source_action,
        shift.target_action,
        shift.shift_amount.hex(),
    )


def _candidate_identity(
    baseline_identity: str,
    hero_seat: str,
    shifts: tuple[AiofPreflopShift, ...],
) -> str:
    return _sha256_identity(
        {
            "algorithm": AIOF_PREFLOP_CANDIDATE_IDENTITY_ALGORITHM,
            "baseline_identity": baseline_identity,
            "hero_seat": hero_seat,
            "shifts": shifts,
        }
    )


def _candidate_profile(
    baseline: SuppliedProfile,
    hero_seat: str,
    shifts: tuple[AiofPreflopShift, ...],
) -> SuppliedProfile:
    sb = {row.combo: row.probability for row in baseline.sb_shove}
    bb = {row.combo: row.probability for row in baseline.bb_call}
    target = sb if hero_seat == "sb" else bb
    for shift in shifts:
        target[shift.combo] = shift.candidate_active_probability
    return SuppliedProfile(
        tuple(ComboActionProbability(combo, sb[combo]) for combo in sorted(sb)),
        tuple(ComboActionProbability(combo, bb[combo]) for combo in sorted(bb)),
    )


def _materialize_candidates(
    request: AiofPreflopCandidateRepeatedRequest,
    baseline_profile: SuppliedProfile,
    baseline_identity: str,
    shift_amounts: tuple[float, ...],
) -> tuple[_MaterializedCandidate, ...]:
    active_action = "shove" if request.hero_seat == "sb" else "call"
    active = _active_probability_map(baseline_profile, request.hero_seat)
    options_by_combo: dict[str, tuple[AiofPreflopShift, ...]] = {}
    for combo in sorted(active):
        probability = active[combo]
        options = []
        for shift in shift_amounts:
            if probability + shift <= 1.0:
                options.append(
                    AiofPreflopShift(
                        combo,
                        "fold",
                        active_action,
                        shift,
                        probability,
                        probability + shift,
                    )
                )
            if probability - shift >= 0.0:
                options.append(
                    AiofPreflopShift(
                        combo,
                        active_action,
                        "fold",
                        shift,
                        probability,
                        probability - shift,
                    )
                )
        options_by_combo[combo] = tuple(sorted(options, key=_shift_sort_key))

    shift_sets: list[tuple[AiofPreflopShift, ...]] = [
        (shift,)
        for combo in sorted(options_by_combo)
        for shift in options_by_combo[combo]
    ]
    if request.max_shifted_combos == 2:
        combos = sorted(options_by_combo)
        for first_index, first_combo in enumerate(combos):
            for second_combo in combos[first_index + 1 :]:
                for first in options_by_combo[first_combo]:
                    for second in options_by_combo[second_combo]:
                        shift_sets.append(tuple(sorted((first, second), key=_shift_sort_key)))
    shift_sets.sort(key=lambda group: tuple(_shift_sort_key(item) for item in group))

    materialized = []
    for shifts in shift_sets:
        candidate_id = _candidate_identity(
            baseline_identity, request.hero_seat, shifts
        )
        profile = _candidate_profile(baseline_profile, request.hero_seat, shifts)
        l1_distance = 2.0 * math.fsum(shift.shift_amount for shift in shifts)
        if not math.isfinite(l1_distance):
            raise AiofContractError(
                AiofStatus.NUMERIC_FAILURE, "candidate L1 distance is non-finite"
            )
        profile_identity = _sha256_identity(
            {
                "baseline_identity": baseline_identity,
                "candidate_id": candidate_id,
                "profile": profile,
            }
        )
        materialized.append(
            _MaterializedCandidate(
                candidate_id,
                shifts,
                l1_distance,
                profile_identity,
                profile,
            )
        )
    ids = [candidate.candidate_id for candidate in materialized]
    if len(ids) != len(set(ids)):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, "candidate identity collision detected"
        )
    return tuple(materialized)


def _analysis_identity(
    request: AiofPreflopCandidateRepeatedRequest,
    baseline_identity: str,
    candidates: tuple[_MaterializedCandidate, ...],
    shifts: tuple[float, ...],
    workload: AiofPreflopWorkloadProjection,
) -> str:
    return _sha256_identity(
        {
            "algorithm": AIOF_PREFLOP_ANALYSIS_IDENTITY_ALGORITHM,
            "baseline_identity": baseline_identity,
            "candidate_ids": tuple(candidate.candidate_id for candidate in candidates),
            "generation": {
                "shift_amounts": shifts,
                "max_shifted_combos": request.max_shifted_combos,
            },
            "horizon": request.horizon,
            "discount": request.discount,
            "tolerance": request.tolerance,
            "minimum_total_uplift": request.minimum_total_uplift,
            "aiof_limits": request.aiof_limits,
            "bridge_limits": request.bridge_limits,
            "max_horizon": request.max_horizon,
            "equity_algorithm": request.algorithm,
            "coverage": {
                "generated_candidate_ids": tuple(
                    candidate.candidate_id for candidate in candidates
                ),
                "kept_candidate_ids": tuple(
                    candidate.candidate_id for candidate in candidates
                ),
                "filtering_applied": False,
            },
            "workload": workload,
        }
    )


def _pushfold_request(
    request: AiofPreflopCandidateRepeatedRequest,
    profile: SuppliedProfile,
    opponent_seat: str,
) -> PushFoldRequest:
    return PushFoldRequest(
        sb_range=request.sb_range,
        bb_range=request.bb_range,
        dead_cards=request.dead_cards,
        algorithm=request.algorithm,
        limits=request.aiof_limits,
        requested_trace_points=0,
        game=request.game,
        profile=profile,
        best_response_seats=(opponent_seat,),
        deviation_tolerance=request.tolerance,
        seed=None,
        samples=None,
    )


def _run_exact_analysis(
    request: AiofPreflopCandidateRepeatedRequest,
    profile: SuppliedProfile,
    prepared: PreparedRanges,
    opponent_seat: str,
) -> PushFoldAnalysis:
    result = analyze_pushfold(_pushfold_request(request, profile, opponent_seat))
    if result.status is not AiofStatus.SUCCESS:
        raise AiofContractError(
            result.status, result.error_message or result.status.value
        )
    if result.analysis is None or result.error_message is not None:
        raise AiofContractError(
            AiofStatus.ORACLE_MISMATCH,
            "malformed successful supplied-profile analysis",
        )
    analysis = result.analysis
    if analysis.algorithm is not EquityAlgorithm.EXACT_EXHAUSTIVE:
        raise AiofContractError(
            AiofStatus.ORACLE_MISMATCH, "analysis algorithm changed unexpectedly"
        )
    if analysis.chip_accounting_id != AIOF_PREFLOP_UTILITY_CONTRACT:
        raise AiofContractError(
            AiofStatus.ACCOUNTING_MISMATCH, "unexpected ChipEV accounting contract"
        )
    if analysis.prepared_ranges_identity != prepared.content_identity:
        raise AiofContractError(
            AiofStatus.ORACLE_MISMATCH, "prepared-range identity mismatch"
        )
    if analysis.compatible_pair_count != prepared.compatible_pair_count:
        raise AiofContractError(
            AiofStatus.ORACLE_MISMATCH, "compatible-pair count mismatch"
        )
    if len(analysis.best_responses) != 1 or analysis.best_responses[0].seat != opponent_seat:
        raise AiofContractError(
            AiofStatus.ORACLE_MISMATCH, "opponent response projection mismatch"
        )
    return analysis


def _extract_exact_values(
    analysis: PushFoldAnalysis,
    hero_seat: str,
    tolerance: float,
) -> tuple[FixedProfileValue, UnilateralBestResponse, float]:
    hero_ev = analysis.profile_value_sb if hero_seat == "sb" else analysis.profile_value_bb
    opponent_ev = analysis.profile_value_bb if hero_seat == "sb" else analysis.profile_value_sb
    if not all(math.isfinite(value) for value in (hero_ev, opponent_ev)):
        raise AiofContractError(
            AiofStatus.NUMERIC_FAILURE, "profile value is non-finite"
        )
    scale = max(1.0, abs(hero_ev), abs(opponent_ev))
    numeric_tolerance = max(tolerance, 1e-12 * scale)
    if abs(hero_ev + opponent_ev) > numeric_tolerance:
        raise AiofContractError(
            AiofStatus.ACCOUNTING_MISMATCH, "fee-zero heads-up zero-sum check failed"
        )
    response = analysis.best_responses[0]
    if not all(
        math.isfinite(value)
        for value in (
            response.supplied_profile_value,
            response.best_response_value,
            response.raw_gain,
        )
    ):
        raise AiofContractError(
            AiofStatus.NUMERIC_FAILURE, "opponent response value is non-finite"
        )
    if abs(response.supplied_profile_value - opponent_ev) > numeric_tolerance:
        raise AiofContractError(
            AiofStatus.ORACLE_MISMATCH,
            "opponent supplied-profile value is inconsistent",
        )
    post_response_hero_ev = -response.best_response_value
    if not math.isfinite(post_response_hero_ev):
        raise AiofContractError(
            AiofStatus.NUMERIC_FAILURE, "post-response Hero value is non-finite"
        )
    return FixedProfileValue(hero_ev, opponent_ev, 0.0), response, post_response_hero_ev


def _execute(
    request: AiofPreflopCandidateRepeatedRequest,
    shift_amounts: tuple[float, ...],
    selector_configuration: AutomaticCommitmentSelectionConfig,
) -> AiofPreflopCandidateRepeatedPayload:
    prepared = prepare_compatible_ranges(
        request.sb_range,
        request.bb_range,
        request.dead_cards,
        request.aiof_limits,
    )
    baseline_profile = _canonical_profile(request.baseline_profile, prepared)
    baseline_identity = _baseline_identity(request, prepared, baseline_profile)
    if (
        request.expected_baseline_identity is not None
        and request.expected_baseline_identity != baseline_identity
    ):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, "expected baseline identity mismatch"
        )
    active_probabilities = _active_probability_map(
        baseline_profile, request.hero_seat
    )
    feasible_counts = _feasible_shift_counts(active_probabilities, shift_amounts)
    workload = _project_workload(request, prepared, feasible_counts)

    materialized = _materialize_candidates(
        request, baseline_profile, baseline_identity, shift_amounts
    )
    if len(materialized) != workload.candidate_count:
        raise AiofContractError(
            AiofStatus.ORACLE_MISMATCH,
            "candidate materialization count differs from projection",
        )
    analysis_identity = _analysis_identity(
        request, baseline_identity, materialized, shift_amounts, workload
    )

    opponent_seat = "bb" if request.hero_seat == "sb" else "sb"
    hero_active_action = "shove" if request.hero_seat == "sb" else "call"
    opponent_active_action = "call" if opponent_seat == "bb" else "shove"
    baseline_analysis = _run_exact_analysis(
        request, baseline_profile, prepared, opponent_seat
    )
    baseline_value, baseline_response, _ = _extract_exact_values(
        baseline_analysis, request.hero_seat, request.tolerance
    )
    baseline_hero_ev = baseline_value.hero_ev
    baseline_record = AiofPreflopBaselineRecord(
        profile_identity=_sha256_identity(
            {
                "baseline_identity": baseline_identity,
                "profile": baseline_profile,
            }
        ),
        profile=baseline_profile,
        analysis_input_identity=baseline_analysis.input_identity,
        fixed_profile_value=baseline_value,
        opponent_response=baseline_response,
        baseline_hero_ev=baseline_hero_ev,
    )

    candidate_records = []
    value_candidates = []
    for candidate in materialized:
        analysis = _run_exact_analysis(
            request, candidate.profile, prepared, opponent_seat
        )
        fixed_value, response, post_response_hero_ev = _extract_exact_values(
            analysis, request.hero_seat, request.tolerance
        )
        post_response_diff = post_response_hero_ev - baseline_hero_ev
        if not math.isfinite(post_response_diff):
            raise AiofContractError(
                AiofStatus.NUMERIC_FAILURE,
                "post-response Hero difference is non-finite",
            )
        record = AiofPreflopCandidateRecord(
            candidate_id=candidate.candidate_id,
            shifts=candidate.shifts,
            l1_distance=candidate.l1_distance,
            profile_identity=candidate.profile_identity,
            profile=candidate.profile,
            analysis_input_identity=analysis.input_identity,
            fixed_profile_value=fixed_value,
            opponent_response=response,
            baseline_hero_ev=baseline_hero_ev,
            fixed_opponent_hero_ev=fixed_value.hero_ev,
            post_response_hero_ev_worst=post_response_hero_ev,
            post_response_hero_ev_best=post_response_hero_ev,
            post_response_hero_ev_worst_diff=post_response_diff,
        )
        candidate_records.append(record)
        value_candidates.append(
            AutomaticCommitmentValueCandidate(
                candidate_id=candidate.candidate_id,
                fixed_profile_value=fixed_value,
                post_response_hero_ev_worst=post_response_hero_ev,
                post_response_hero_ev_best=post_response_hero_ev,
                post_response_hero_ev_worst_diff=post_response_diff,
                l1_distance=candidate.l1_distance,
            )
        )

    candidate_ids = tuple(candidate.candidate_id for candidate in materialized)
    coverage = AiofPreflopCandidateCoverage(
        generated_candidate_ids=candidate_ids,
        kept_candidate_ids=candidate_ids,
        shift_amounts=shift_amounts,
        max_shifted_combos=request.max_shifted_combos,
        filtering_applied=False,
    )
    selector_coverage = AutomaticCommitmentSearchCoverage(
        input_candidate_ids=candidate_ids,
        kept_candidate_ids=candidate_ids,
        source="aiof_preflop_exact_combo_generated",
        shift_amounts=shift_amounts,
        max_simultaneous_info_sets=request.max_shifted_combos,
        generation_max_candidates=request.bridge_limits.max_candidates,
        filtering_applied=False,
    )
    try:
        automatic_selection = select_automatic_commitment_values(
            baseline_value,
            tuple(value_candidates),
            horizon=request.horizon,
            discount=request.discount,
            tolerance=request.tolerance,
            max_horizon=request.max_horizon,
            configuration=selector_configuration,
            search_coverage=selector_coverage,
        )
    except ValueError as exc:
        raise AiofContractError(AiofStatus.INVALID_INPUT, str(exc)) from exc

    return AiofPreflopCandidateRepeatedPayload(
        hero_seat=request.hero_seat,
        opponent_seat=opponent_seat,
        hero_active_action=hero_active_action,
        opponent_active_action=opponent_active_action,
        prepared_ranges=prepared,
        baseline_identity=baseline_identity,
        analysis_identity=analysis_identity,
        baseline=baseline_record,
        workload=workload,
        coverage=coverage,
        candidates=tuple(candidate_records),
        automatic_selection=automatic_selection,
        request=request,
    )


def _clean_error(message: str, fallback: str) -> str:
    cleaned = " ".join((message or fallback).replace("\r", " ").replace("\n", " ").split())
    return cleaned[:500] or fallback


def analyze_aiof_preflop_candidate_repeated(
    request: AiofPreflopCandidateRepeatedRequest,
) -> AiofPreflopCandidateRepeatedResult:
    """Run the exact bounded bridge with no fallback or partial failure payload.

    Success returns the complete baseline, every declared candidate, native
    factorized exact response rows, repeated values, and automatic selections.
    Any controlled or unexpected failure returns ``payload=None``.
    """

    try:
        validated, shifts, selector_configuration = _validate_request(request)
        payload = _execute(validated, shifts, selector_configuration)
        return AiofPreflopCandidateRepeatedResult(
            AiofStatus.SUCCESS, payload, None
        )
    except AiofContractError as exc:
        return AiofPreflopCandidateRepeatedResult(
            exc.status,
            None,
            _clean_error(str(exc), exc.status.value),
        )
    except (ArithmeticError, OverflowError) as exc:
        return AiofPreflopCandidateRepeatedResult(
            AiofStatus.NUMERIC_FAILURE,
            None,
            _clean_error(str(exc), "numeric failure"),
        )
    except Exception:
        return AiofPreflopCandidateRepeatedResult(
            AiofStatus.NUMERIC_FAILURE,
            None,
            "unexpected bridge failure",
        )
