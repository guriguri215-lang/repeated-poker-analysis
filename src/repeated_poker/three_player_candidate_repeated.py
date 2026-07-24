"""Bounded robust-all candidate and repeated-value integration for M31.

This isolated in-memory adapter completely generates one- and two-information-
set exact Hero policy shifts, evaluates the baseline and every candidate with
the public M31 exact river/rake evaluator, and passes only the resulting
domain-neutral scalar values to the public M27 selection kernel.

The supported claim is deliberately narrow: O1 and O2 adapt simultaneously to
the complete non-cooperative exact response correspondence, and Hero safety is
the correspondence-wide ``response.hero_worst`` value.  Current CFR state,
individual witnesses, the pure-response subset, the joint-plan stress
diagnostic, and ``hero_best`` are never safety substitutes.  O1+O2 is used only
to fill M27's historical accounting field; it is not a coalition claim.

Default/hard ceilings are respectively: shifts ``4/16``; simultaneous Hero
information sets ``2/2``; generated candidates ``4/16``; total M31 runs
``5/17``; aggregate joint profiles ``80/612``; aggregate terminal evaluations
``10240/156672``; aggregate support pairs ``1125/67473``; aggregate exact
systems ``50000/4250000``; horizon ``1000/100000``; timing evaluations
``4004/1600016``; rational numerator/denominator bits ``256/1024``; identity
records ``100000/1000000``; projected embedded M31 bytes
``20000000/256000000``; outer records ``500000/2000000``; and outer bytes
``32000000/256000000``.  Caps are checked before candidate materialisation,
before the first candidate M31 call, before timing rows, and before output.

There is no truncation, sampling, fallback, skipped candidate, clamping,
probability repair, or partial result.  The module is intentionally not
exported from :mod:`repeated_poker`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, fields
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any, Mapping, Sequence

from . import automatic_commitment_selection as _m27
from . import three_player_response as _m30
from . import three_player_river_rake as _m31
from .fixed_profile import FixedProfileValue


CONTRACT_VERSION = "m32-three-player-candidate-repeated-robust-all-v1"
ALGORITHM_VERSION = "complete-exact-shift-m31-m27-integration-v1"
PROJECTION_VERSION = "exact-rational-to-safe-binary64-v1"

EXACT_THREE_PLAYER_CANDIDATE_REPEATED_COMPLETE = (
    "EXACT_THREE_PLAYER_CANDIDATE_REPEATED_COMPLETE"
)
INVALID_INPUT = "INVALID_INPUT"
UNSUPPORTED_MODE = "UNSUPPORTED_MODE"
STALE_INPUT = "STALE_INPUT"
CAP_EXCEEDED = "CAP_EXCEEDED"
BASELINE_RESPONSE_FAILURE = "BASELINE_RESPONSE_FAILURE"
CANDIDATE_RESPONSE_FAILURE = "CANDIDATE_RESPONSE_FAILURE"
SCALAR_PROJECTION_FAILURE = "SCALAR_PROJECTION_FAILURE"
SELECTOR_FAILURE = "SELECTOR_FAILURE"
NUMERIC_FAILURE = "NUMERIC_FAILURE"
INTERNAL_FAILURE = "INTERNAL_FAILURE"

ROBUST_ALL = "robust_all"
SIMULTANEOUS_O1_O2 = "simultaneous_o1_o2"
RESPONSE_SEMANTICS = (
    "o1_o2_noncooperative_exact_correspondence_hero_worst"
)
SELECTOR_KERNEL = _m27.AUTOMATIC_COMMITMENT_SELECTION_CONTRACT_VERSION
SELECTOR_TRANSPORT_OPPONENT_VALUE = (
    "o1_plus_o2_accounting_only_not_a_coalition"
)

DEFAULT_MAX_SHIFT_AMOUNTS = 4
DEFAULT_MAX_SIMULTANEOUS_INFO_SETS = 2
DEFAULT_MAX_GENERATED_CANDIDATES = 4
DEFAULT_MAX_TOTAL_M31_RUNS = 5
DEFAULT_MAX_AGGREGATE_JOINT_PROFILES = 80
DEFAULT_MAX_AGGREGATE_TERMINAL_EVALUATIONS = 10_240
DEFAULT_MAX_AGGREGATE_SUPPORT_PAIRS = 1_125
DEFAULT_MAX_AGGREGATE_EXACT_LINEAR_SYSTEMS = 50_000
DEFAULT_MAX_HORIZON = 1_000
DEFAULT_MAX_TIMING_ROW_EVALUATIONS = 4_004
DEFAULT_MAX_RATIONAL_NUMERATOR_BITS = 256
DEFAULT_MAX_RATIONAL_DENOMINATOR_BITS = 256
DEFAULT_MAX_IDENTITY_RECORDS = 100_000
DEFAULT_MAX_EMBEDDED_M31_OUTPUT_BYTES = 20_000_000
DEFAULT_MAX_OUTER_OUTPUT_RECORDS = 500_000
DEFAULT_MAX_OUTER_OUTPUT_BYTES = 32_000_000

HARD_MAX_SHIFT_AMOUNTS = 16
HARD_MAX_SIMULTANEOUS_INFO_SETS = 2
HARD_MAX_GENERATED_CANDIDATES = 16
HARD_MAX_TOTAL_M31_RUNS = 17
HARD_MAX_AGGREGATE_JOINT_PROFILES = 612
HARD_MAX_AGGREGATE_TERMINAL_EVALUATIONS = 156_672
HARD_MAX_AGGREGATE_SUPPORT_PAIRS = 67_473
HARD_MAX_AGGREGATE_EXACT_LINEAR_SYSTEMS = 4_250_000
HARD_MAX_HORIZON = 100_000
HARD_MAX_TIMING_ROW_EVALUATIONS = 1_600_016
HARD_MAX_RATIONAL_NUMERATOR_BITS = 1_024
HARD_MAX_RATIONAL_DENOMINATOR_BITS = 1_024
HARD_MAX_IDENTITY_RECORDS = 1_000_000
HARD_MAX_EMBEDDED_M31_OUTPUT_BYTES = 256_000_000
HARD_MAX_OUTER_OUTPUT_RECORDS = 2_000_000
HARD_MAX_OUTER_OUTPUT_BYTES = 256_000_000

_INTEGER_RE = re.compile(r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)")
_FRACTION_RE = re.compile(r"(-?[1-9][0-9]*)/([1-9][0-9]*)")
_IDENTITY_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ThreePlayerCandidateGenerationConfig:
    """Complete exact Hero-shift universe configuration."""

    shift_amounts: tuple[str, ...] = ("1/2",)
    max_simultaneous_info_sets: int = 2
    search_mode: str = ROBUST_ALL
    adaptation_mode: str = SIMULTANEOUS_O1_O2


@dataclass(frozen=True)
class ThreePlayerRepeatedConfig:
    """Simultaneous adaptation sensitivity configuration."""

    horizon: int = 1
    discount: float = 1.0
    tolerance: float = 1e-9


@dataclass(frozen=True)
class ThreePlayerCandidateRepeatedLimits:
    """Caller-lowerable M32 bounds with immutable hard ceilings."""

    max_shift_amounts: int = DEFAULT_MAX_SHIFT_AMOUNTS
    max_simultaneous_info_sets: int = DEFAULT_MAX_SIMULTANEOUS_INFO_SETS
    max_generated_candidates: int = DEFAULT_MAX_GENERATED_CANDIDATES
    max_total_m31_runs: int = DEFAULT_MAX_TOTAL_M31_RUNS
    max_aggregate_joint_profiles: int = (
        DEFAULT_MAX_AGGREGATE_JOINT_PROFILES
    )
    max_aggregate_terminal_evaluations: int = (
        DEFAULT_MAX_AGGREGATE_TERMINAL_EVALUATIONS
    )
    max_aggregate_support_pairs: int = DEFAULT_MAX_AGGREGATE_SUPPORT_PAIRS
    max_aggregate_exact_linear_systems: int = (
        DEFAULT_MAX_AGGREGATE_EXACT_LINEAR_SYSTEMS
    )
    max_horizon: int = DEFAULT_MAX_HORIZON
    max_timing_row_evaluations: int = (
        DEFAULT_MAX_TIMING_ROW_EVALUATIONS
    )
    max_rational_numerator_bits: int = (
        DEFAULT_MAX_RATIONAL_NUMERATOR_BITS
    )
    max_rational_denominator_bits: int = (
        DEFAULT_MAX_RATIONAL_DENOMINATOR_BITS
    )
    max_identity_records: int = DEFAULT_MAX_IDENTITY_RECORDS
    max_embedded_m31_output_bytes: int = (
        DEFAULT_MAX_EMBEDDED_M31_OUTPUT_BYTES
    )
    max_outer_output_records: int = DEFAULT_MAX_OUTER_OUTPUT_RECORDS
    max_outer_output_bytes: int = DEFAULT_MAX_OUTER_OUTPUT_BYTES


@dataclass(frozen=True)
class ThreePlayerCandidateRepeatedPins:
    """Optional semantic pins; every supplied pin must match exactly."""

    scenario_identity: str | None = None
    tree_structure_identity: str | None = None
    baseline_fixed_hero_identity: str | None = None
    initial_profile_identity: str | None = None
    candidate_generation_config_identity: str | None = None
    candidate_universe_identity: str | None = None
    ordered_candidate_ids: tuple[str, ...] | None = None
    repeated_config_identity: str | None = None
    selector_transport_identity: str | None = None
    candidate_fixed_hero_identities: tuple[tuple[str, str], ...] | None = None
    candidate_response_game_identities: tuple[tuple[str, str], ...] | None = None
    candidate_response_run_identities: tuple[tuple[str, str], ...] | None = None
    candidate_run_identities: tuple[tuple[str, str], ...] | None = None
    m32_run_identity: str | None = None


@dataclass(frozen=True)
class HeroPolicyEdit:
    """One exact source-to-target probability shift."""

    information_set_id: str
    source_action_id: str
    target_action_id: str
    shift_amount: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ThreePlayerHeroCandidate:
    """One complete exact fixed-Hero candidate."""

    candidate_id: str
    resulting_policy_identity: str
    edits: tuple[HeroPolicyEdit, ...]
    policy: _m31.ExactBehaviorPolicy
    l1_distance_exact: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "resulting_policy_identity": self.resulting_policy_identity,
            "edits": [edit.to_dict() for edit in self.edits],
            "complete_fixed_hero_policy": {
                information_set: dict(distribution)
                for information_set, distribution in self.policy.probabilities.items()
            },
            "l1_distance_exact": self.l1_distance_exact,
        }


@dataclass(frozen=True)
class ExactScalarProjection:
    """An exact rational source and its audited binary64 projection."""

    source_exact: str
    binary64: float
    binary64_exact_rational: str
    absolute_error_exact: str
    exact_in_binary64: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ThreePlayerCandidateEvaluation:
    """One candidate, its full M31 object, scalar sources, and identities."""

    candidate: ThreePlayerHeroCandidate
    scenario_response: _m31.ScenarioResponseResult
    exact_values: dict[str, Any]
    scalar_projections: dict[str, ExactScalarProjection]
    candidate_run_identity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "m31_scenario_response": self.scenario_response.to_dict(),
            "exact_values": self.exact_values,
            "scalar_projections": {
                name: projection.to_dict()
                for name, projection in self.scalar_projections.items()
            },
            "candidate_run_identity": self.candidate_run_identity,
        }


@dataclass(frozen=True)
class ThreePlayerCandidateRepeatedAnalysis:
    """Complete successful M32 analysis, retaining native M31/M27 objects."""

    baseline_response: _m31.ScenarioResponseResult
    baseline_exact_values: dict[str, str]
    baseline_scalar_projections: dict[str, ExactScalarProjection]
    candidates: tuple[ThreePlayerCandidateEvaluation, ...]
    selector_report: _m27.AutomaticCommitmentSelectionReport
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        output = dict(self.payload)
        output["baseline"] = {
            "m31_scenario_response": self.baseline_response.to_dict(),
            "exact_initial_profile_values": self.baseline_exact_values,
            "scalar_projections": {
                name: projection.to_dict()
                for name, projection in self.baseline_scalar_projections.items()
            },
        }
        output["candidates"] = [
            candidate.to_dict() for candidate in self.candidates
        ]
        output["selector"] = _selector_output(self.selector_report)
        return output


@dataclass(frozen=True)
class ThreePlayerCandidateRepeatedError:
    """Bounded failure metadata without partial analysis."""

    phase: str
    message: str
    cause_status: str | None = None
    candidate_id: str | None = None


@dataclass(frozen=True)
class ThreePlayerCandidateRepeatedResult:
    """Exclusive complete-success or fail-closed M32 result."""

    status: str
    analysis: ThreePlayerCandidateRepeatedAnalysis | None
    error: ThreePlayerCandidateRepeatedError | None
    partial_result: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "analysis": (
                None if self.analysis is None else self.analysis.to_dict()
            ),
            "error": None if self.error is None else asdict(self.error),
            "partial_result": self.partial_result,
        }


@dataclass(frozen=True)
class _EditSpec:
    information_set_id: str
    source_action_id: str
    target_action_id: str
    shift: Fraction

    def key(self) -> tuple[Any, ...]:
        return (
            self.information_set_id,
            self.source_action_id,
            self.target_action_id,
            self.shift,
        )

    def projection(self) -> dict[str, str]:
        return {
            "information_set_id": self.information_set_id,
            "source_action_id": self.source_action_id,
            "target_action_id": self.target_action_id,
            "shift_amount": _rational_text(self.shift),
        }


class _M32Failure(ValueError):
    def __init__(
        self,
        status: str,
        phase: str,
        message: str,
        *,
        cause_status: str | None = None,
        candidate_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.phase = phase
        self.cause_status = cause_status
        self.candidate_id = candidate_id


_HARD_LIMITS = {
    "max_shift_amounts": HARD_MAX_SHIFT_AMOUNTS,
    "max_simultaneous_info_sets": HARD_MAX_SIMULTANEOUS_INFO_SETS,
    "max_generated_candidates": HARD_MAX_GENERATED_CANDIDATES,
    "max_total_m31_runs": HARD_MAX_TOTAL_M31_RUNS,
    "max_aggregate_joint_profiles": HARD_MAX_AGGREGATE_JOINT_PROFILES,
    "max_aggregate_terminal_evaluations": (
        HARD_MAX_AGGREGATE_TERMINAL_EVALUATIONS
    ),
    "max_aggregate_support_pairs": HARD_MAX_AGGREGATE_SUPPORT_PAIRS,
    "max_aggregate_exact_linear_systems": (
        HARD_MAX_AGGREGATE_EXACT_LINEAR_SYSTEMS
    ),
    "max_horizon": HARD_MAX_HORIZON,
    "max_timing_row_evaluations": HARD_MAX_TIMING_ROW_EVALUATIONS,
    "max_rational_numerator_bits": HARD_MAX_RATIONAL_NUMERATOR_BITS,
    "max_rational_denominator_bits": HARD_MAX_RATIONAL_DENOMINATOR_BITS,
    "max_identity_records": HARD_MAX_IDENTITY_RECORDS,
    "max_embedded_m31_output_bytes": HARD_MAX_EMBEDDED_M31_OUTPUT_BYTES,
    "max_outer_output_records": HARD_MAX_OUTER_OUTPUT_RECORDS,
    "max_outer_output_bytes": HARD_MAX_OUTER_OUTPUT_BYTES,
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _identity(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _rational_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _clean_text(value: str, maximum: int) -> str:
    return value.replace("\r", " ").replace("\n", " ")[:maximum]


def _failure(exc: _M32Failure) -> ThreePlayerCandidateRepeatedResult:
    return ThreePlayerCandidateRepeatedResult(
        status=exc.status,
        analysis=None,
        error=ThreePlayerCandidateRepeatedError(
            phase=_clean_text(exc.phase, 96),
            message=_clean_text(str(exc), 600),
            cause_status=(
                None
                if exc.cause_status is None
                else _clean_text(exc.cause_status, 96)
            ),
            candidate_id=(
                None
                if exc.candidate_id is None
                else _clean_text(exc.candidate_id, 128)
            ),
        ),
        partial_result=False,
    )


def _checked_product(left: int, right: int, maximum: int, phase: str) -> int:
    if left < 0 or right < 0 or left > maximum:
        raise _M32Failure(CAP_EXCEEDED, phase, "checked product cap exceeded")
    if right and left > maximum // right:
        raise _M32Failure(CAP_EXCEEDED, phase, "checked product cap exceeded")
    return left * right


def _checked_sum(left: int, right: int, maximum: int, phase: str) -> int:
    if left < 0 or right < 0 or left > maximum - right:
        raise _M32Failure(CAP_EXCEEDED, phase, "checked sum cap exceeded")
    return left + right


def _validate_limits(limits: ThreePlayerCandidateRepeatedLimits) -> None:
    if type(limits) is not ThreePlayerCandidateRepeatedLimits:
        raise _M32Failure(
            INVALID_INPUT,
            "limits",
            "limits must be ThreePlayerCandidateRepeatedLimits",
        )
    for field in fields(ThreePlayerCandidateRepeatedLimits):
        value = getattr(limits, field.name)
        hard = _HARD_LIMITS[field.name]
        if type(value) is not int or value < 1 or value > hard:
            raise _M32Failure(
                INVALID_INPUT,
                "limits",
                f"{field.name} must be an int in [1,{hard}]",
            )


def _check_fraction_bits(
    value: Fraction,
    limits: ThreePlayerCandidateRepeatedLimits,
    phase: str,
) -> Fraction:
    if abs(value.numerator).bit_length() > limits.max_rational_numerator_bits:
        raise _M32Failure(
            CAP_EXCEEDED, phase, "max_rational_numerator_bits exceeded"
        )
    if value.denominator.bit_length() > limits.max_rational_denominator_bits:
        raise _M32Failure(
            CAP_EXCEEDED, phase, "max_rational_denominator_bits exceeded"
        )
    return value


def _parse_rational(
    value: Any,
    limits: ThreePlayerCandidateRepeatedLimits,
    phase: str,
) -> Fraction:
    if type(value) is not str:
        raise _M32Failure(
            INVALID_INPUT, phase, "value must be a canonical rational string"
        )
    if _INTEGER_RE.fullmatch(value) is not None:
        numerator_text, denominator_text = value, "1"
    else:
        match = _FRACTION_RE.fullmatch(value)
        if match is None:
            raise _M32Failure(INVALID_INPUT, phase, "noncanonical rational")
        numerator_text, denominator_text = match.groups()
    maximum_num_digits = (
        math.ceil(limits.max_rational_numerator_bits * math.log10(2)) + 1
    )
    maximum_den_digits = (
        math.ceil(limits.max_rational_denominator_bits * math.log10(2)) + 1
    )
    if len(numerator_text.lstrip("-")) > maximum_num_digits:
        raise _M32Failure(
            CAP_EXCEEDED, phase, "max_rational_numerator_bits exceeded"
        )
    if len(denominator_text) > maximum_den_digits:
        raise _M32Failure(
            CAP_EXCEEDED, phase, "max_rational_denominator_bits exceeded"
        )
    numerator = int(numerator_text)
    denominator = int(denominator_text)
    if denominator != 1 and (
        numerator == 0 or math.gcd(abs(numerator), denominator) != 1
    ):
        raise _M32Failure(
            INVALID_INPUT, phase, "fraction must be nonzero and reduced"
        )
    parsed = Fraction(numerator, denominator)
    if _rational_text(parsed) != value:
        raise _M32Failure(INVALID_INPUT, phase, "rational is not canonical")
    return _check_fraction_bits(parsed, limits, phase)


def _validate_identity(value: Any, phase: str, *, optional: bool = True) -> None:
    if optional and value is None:
        return
    if type(value) is not str or _IDENTITY_RE.fullmatch(value) is None:
        raise _M32Failure(
            INVALID_INPUT,
            phase,
            "identity must be a lowercase 64-character SHA-256 string",
        )


def _validate_pin_pairs(
    value: Any, phase: str
) -> tuple[tuple[str, str], ...] | None:
    if value is None:
        return None
    if type(value) is not tuple:
        raise _M32Failure(INVALID_INPUT, phase, "pin mapping must be a tuple")
    ids: list[str] = []
    for item in value:
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or not item[0]
        ):
            raise _M32Failure(
                INVALID_INPUT, phase, "pin mapping rows must be (candidate_id, identity)"
            )
        _validate_identity(item[1], phase, optional=False)
        ids.append(item[0])
    if len(ids) != len(set(ids)):
        raise _M32Failure(
            INVALID_INPUT, phase, "pin mapping candidate IDs must be unique"
        )
    return value


def _validate_pins(pins: ThreePlayerCandidateRepeatedPins) -> None:
    if type(pins) is not ThreePlayerCandidateRepeatedPins:
        raise _M32Failure(
            INVALID_INPUT,
            "pins",
            "pins must be ThreePlayerCandidateRepeatedPins",
        )
    for name in (
        "scenario_identity",
        "tree_structure_identity",
        "baseline_fixed_hero_identity",
        "initial_profile_identity",
        "candidate_generation_config_identity",
        "candidate_universe_identity",
        "repeated_config_identity",
        "selector_transport_identity",
        "m32_run_identity",
    ):
        _validate_identity(getattr(pins, name), f"pins.{name}")
    if pins.ordered_candidate_ids is not None:
        if (
            type(pins.ordered_candidate_ids) is not tuple
            or any(
                type(value) is not str or not value
                for value in pins.ordered_candidate_ids
            )
            or len(pins.ordered_candidate_ids)
            != len(set(pins.ordered_candidate_ids))
        ):
            raise _M32Failure(
                INVALID_INPUT,
                "pins.ordered_candidate_ids",
                "ordered candidate IDs must be a unique tuple of strings",
            )
    for name in (
        "candidate_fixed_hero_identities",
        "candidate_response_game_identities",
        "candidate_response_run_identities",
        "candidate_run_identities",
    ):
        _validate_pin_pairs(getattr(pins, name), f"pins.{name}")


def _check_pin(
    pins: ThreePlayerCandidateRepeatedPins, name: str, actual: Any
) -> None:
    expected = getattr(pins, name)
    if expected is not None and expected != actual:
        raise _M32Failure(STALE_INPUT, f"pins.{name}", f"stale {name}")


def _validate_configuration(
    generation: ThreePlayerCandidateGenerationConfig,
    repeated: ThreePlayerRepeatedConfig,
    selector_configuration: _m27.AutomaticCommitmentSelectionConfig,
    limits: ThreePlayerCandidateRepeatedLimits,
) -> tuple[tuple[Fraction, ...], str, str, str]:
    if type(generation) is not ThreePlayerCandidateGenerationConfig:
        raise _M32Failure(
            INVALID_INPUT,
            "candidate_generation",
            "generation must be ThreePlayerCandidateGenerationConfig",
        )
    if generation.search_mode != ROBUST_ALL:
        raise _M32Failure(
            UNSUPPORTED_MODE,
            "candidate_generation.search_mode",
            "only search_mode='robust_all' is supported",
        )
    if generation.adaptation_mode != SIMULTANEOUS_O1_O2:
        raise _M32Failure(
            UNSUPPORTED_MODE,
            "candidate_generation.adaptation_mode",
            "only adaptation_mode='simultaneous_o1_o2' is supported",
        )
    if (
        type(generation.max_simultaneous_info_sets) is not int
        or generation.max_simultaneous_info_sets not in (1, 2)
        or generation.max_simultaneous_info_sets
        > limits.max_simultaneous_info_sets
    ):
        raise _M32Failure(
            INVALID_INPUT,
            "candidate_generation.max_simultaneous_info_sets",
            "max_simultaneous_info_sets must be 1 or 2 within M32 limits",
        )
    if type(generation.shift_amounts) is not tuple:
        raise _M32Failure(
            INVALID_INPUT,
            "candidate_generation.shift_amounts",
            "shift_amounts must be a tuple",
        )
    if len(generation.shift_amounts) > limits.max_shift_amounts:
        raise _M32Failure(
            CAP_EXCEEDED,
            "candidate_generation.shift_amounts",
            "max_shift_amounts exceeded before candidate generation",
        )
    shifts = tuple(
        _parse_rational(value, limits, f"shift_amounts[{index}]")
        for index, value in enumerate(generation.shift_amounts)
    )
    if any(value <= 0 or value > 1 for value in shifts):
        raise _M32Failure(
            INVALID_INPUT,
            "candidate_generation.shift_amounts",
            "shift amounts must satisfy 0 < shift <= 1",
        )
    if len(set(shifts)) != len(shifts):
        raise _M32Failure(
            INVALID_INPUT,
            "candidate_generation.shift_amounts",
            "duplicate numeric shift amount",
        )
    shifts = tuple(sorted(shifts))
    if type(repeated) is not ThreePlayerRepeatedConfig:
        raise _M32Failure(
            INVALID_INPUT,
            "repeated",
            "repeated must be ThreePlayerRepeatedConfig",
        )
    if type(repeated.horizon) is not int or repeated.horizon < 1:
        raise _M32Failure(
            INVALID_INPUT,
            "repeated.horizon",
            "horizon must be a positive int",
        )
    if repeated.horizon > limits.max_horizon:
        raise _M32Failure(
            CAP_EXCEEDED,
            "repeated.horizon",
            "horizon exceeds max_horizon",
        )
    if type(selector_configuration) is not _m27.AutomaticCommitmentSelectionConfig:
        raise _M32Failure(
            INVALID_INPUT,
            "selector_configuration",
            "selector configuration has the wrong type",
        )
    try:
        _m27.validate_automatic_commitment_selection_parameters(
            horizon=repeated.horizon,
            discount=repeated.discount,
            tolerance=repeated.tolerance,
            max_horizon=limits.max_horizon,
            configuration=selector_configuration,
        )
    except ValueError as exc:
        raise _M32Failure(
            INVALID_INPUT, "repeated_selector_configuration", str(exc)
        ) from None
    generation_identity = _identity(
        {
            "contract_version": CONTRACT_VERSION,
            "search_mode": generation.search_mode,
            "adaptation_mode": generation.adaptation_mode,
            "shift_amounts": [_rational_text(value) for value in shifts],
            "max_simultaneous_info_sets": (
                generation.max_simultaneous_info_sets
            ),
        }
    )
    repeated_identity = _identity(
        {
            "contract_version": CONTRACT_VERSION,
            "adaptation_mode": generation.adaptation_mode,
            "horizon": repeated.horizon,
            "discount": repeated.discount,
            "tolerance": repeated.tolerance,
        }
    )
    selector_transport_identity = _identity(
        {
            "contract_version": CONTRACT_VERSION,
            "selector_kernel": SELECTOR_KERNEL,
            "response_semantics": RESPONSE_SEMANTICS,
            "opponent_value_transport": SELECTOR_TRANSPORT_OPPONENT_VALUE,
            "configuration": selector_configuration.to_dict(),
            "projection_version": PROJECTION_VERSION,
        }
    )
    return (
        shifts,
        generation_identity,
        repeated_identity,
        selector_transport_identity,
    )


def _hero_action_order(
    scenario: _m31.ThreePlayerRiverRakeScenario,
) -> dict[str, tuple[str, ...]]:
    if type(scenario) is not _m31.ThreePlayerRiverRakeScenario:
        raise _M32Failure(
            INVALID_INPUT,
            "scenario",
            "scenario must be ThreePlayerRiverRakeScenario",
        )
    action_order: dict[str, tuple[str, ...]] = {}
    active: set[int] = set()

    def visit(node: _m31.RiverNode) -> None:
        marker = id(node)
        if marker in active:
            raise _M32Failure(INVALID_INPUT, "scenario.tree", "tree cycle")
        active.add(marker)
        try:
            if type(node) is _m31.RiverDecisionNode:
                if node.owner == "H":
                    actions = tuple(action.action_id for action in node.actions)
                    prior = action_order.get(node.information_set_id)
                    if prior is not None and prior != actions:
                        raise _M32Failure(
                            INVALID_INPUT,
                            "scenario.tree",
                            "inconsistent Hero information-set action order",
                        )
                    action_order[node.information_set_id] = actions
                for action in node.actions:
                    if type(action) is not _m31.RiverAction:
                        raise _M32Failure(
                            INVALID_INPUT,
                            "scenario.tree",
                            "tree action must be RiverAction",
                        )
                    visit(action.child)
            elif type(node) is _m31.RiverChanceNode:
                for outcome in node.outcomes:
                    if type(outcome) is not _m31.RiverChanceOutcome:
                        raise _M32Failure(
                            INVALID_INPUT,
                            "scenario.tree",
                            "chance outcome has the wrong type",
                        )
                    visit(outcome.child)
            elif type(node) is not _m31.RiverTerminalNode:
                raise _M32Failure(
                    INVALID_INPUT, "scenario.tree", "unsupported tree node"
                )
        finally:
            active.remove(marker)

    visit(scenario.root)
    return {key: action_order[key] for key in sorted(action_order)}


def _parse_complete_hero_policy(
    policy: _m31.ExactBehaviorPolicy,
    action_order: Mapping[str, tuple[str, ...]],
    limits: ThreePlayerCandidateRepeatedLimits,
) -> tuple[dict[str, dict[str, Fraction]], list[dict[str, Any]]]:
    if type(policy) is not _m31.ExactBehaviorPolicy:
        raise _M32Failure(
            INVALID_INPUT,
            "baseline_fixed_hero_policy",
            "baseline policy must be ExactBehaviorPolicy",
        )
    raw = policy.probabilities
    if not isinstance(raw, Mapping) or set(raw) != set(action_order):
        raise _M32Failure(
            INVALID_INPUT,
            "baseline_fixed_hero_policy",
            "baseline Hero policy must be complete",
        )
    parsed: dict[str, dict[str, Fraction]] = {}
    projection: list[dict[str, Any]] = []
    for information_set in sorted(action_order):
        distribution = raw[information_set]
        actions = action_order[information_set]
        if not isinstance(distribution, Mapping) or set(distribution) != set(actions):
            raise _M32Failure(
                INVALID_INPUT,
                "baseline_fixed_hero_policy",
                "policy action keys must exactly equal ordered legal actions",
            )
        values: dict[str, Fraction] = {}
        total = Fraction(0)
        for action in actions:
            probability = _parse_rational(
                distribution[action],
                limits,
                f"baseline_fixed_hero_policy.{information_set}.{action}",
            )
            if probability < 0 or probability > 1:
                raise _M32Failure(
                    INVALID_INPUT,
                    "baseline_fixed_hero_policy",
                    "policy probabilities must be in [0,1]",
                )
            values[action] = probability
            total = _check_fraction_bits(
                total + probability, limits, "baseline_fixed_hero_policy"
            )
        if total != 1:
            raise _M32Failure(
                INVALID_INPUT,
                "baseline_fixed_hero_policy",
                "policy probabilities must sum exactly to one",
            )
        parsed[information_set] = values
        projection.append(
            {
                "information_set_id": information_set,
                "probabilities": {
                    action: _rational_text(values[action]) for action in actions
                },
            }
        )
    return parsed, projection


def _validate_initial_types(
    initial_profile: _m31.OpponentInitialProfile,
    attestation: _m31.PerfectRecallAttestation,
    m31_limits: _m31.RiverRakeLimits,
    m30_limits: _m30.ExactResponseLimits,
    m31_pins: _m31.RiverRakeIdentityPins,
    m30_response_pins: _m30.ResponseIdentityPins,
) -> None:
    if (
        type(initial_profile) is not _m31.OpponentInitialProfile
        or initial_profile.o1_probabilities is None
        or initial_profile.o2_probabilities is None
    ):
        raise _M32Failure(
            INVALID_INPUT,
            "initial_profile",
            "complete O1 and O2 initial profiles are mandatory",
        )
    if type(attestation) is not _m31.PerfectRecallAttestation:
        raise _M32Failure(
            INVALID_INPUT,
            "attestation",
            "complete PerfectRecallAttestation is mandatory",
        )
    if type(m31_limits) is not _m31.RiverRakeLimits:
        raise _M32Failure(INVALID_INPUT, "m31_limits", "wrong M31 limits type")
    if type(m30_limits) is not _m30.ExactResponseLimits:
        raise _M32Failure(INVALID_INPUT, "m30_limits", "wrong M30 limits type")
    if type(m31_pins) is not _m31.RiverRakeIdentityPins:
        raise _M32Failure(INVALID_INPUT, "m31_pins", "wrong M31 pins type")
    if type(m30_response_pins) is not _m30.ResponseIdentityPins:
        raise _M32Failure(
            INVALID_INPUT, "m30_response_pins", "wrong M30 pins type"
        )
    for name in (
        "fixed_hero_identity",
        "payoff_table_identity",
        "evaluator_config_identity",
    ):
        if getattr(m31_pins, name) is not None:
            raise _M32Failure(
                INVALID_INPUT,
                f"m31_pins.{name}",
                f"{name} varies by candidate; use M32 per-run pins",
            )
    for name in (
        "fixed_hero_identity",
        "payoff_table_identity",
        "candidate_identity",
        "config_identity",
        "response_game_identity",
        "response_run_identity",
    ):
        if getattr(m30_response_pins, name) is not None:
            raise _M32Failure(
                INVALID_INPUT,
                f"m30_response_pins.{name}",
                f"{name} varies by candidate; use M32 per-run pins",
            )


def _validate_m31_result(
    result: Any,
    limits: ThreePlayerCandidateRepeatedLimits,
    *,
    baseline: bool,
    candidate_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    status = BASELINE_RESPONSE_FAILURE if baseline else CANDIDATE_RESPONSE_FAILURE
    phase = "baseline_response" if baseline else "candidate_response"
    if type(result) is not _m31.ScenarioResponseResult:
        raise _M32Failure(
            status,
            phase,
            "M31 returned an invalid result object",
            candidate_id=candidate_id,
        )
    if (
        result.status != _m31.EXACT_SCENARIO_RESPONSE_COMPLETE
        or result.partial_result
        or result.error is not None
        or result.scenario_evaluation is None
        or result.payoff_table is None
        or result.response is None
    ):
        raise _M32Failure(
            status,
            phase,
            "M31 did not return a complete exact scenario response",
            cause_status=result.status,
            candidate_id=candidate_id,
        )
    scenario_evaluation = result.scenario_evaluation
    response = result.response
    handoff = scenario_evaluation.get("m32_handoff")
    initial = scenario_evaluation.get("initial_profile_comparison")
    if (
        scenario_evaluation.get("status")
        != _m31.EXACT_SCENARIO_RESPONSE_COMPLETE
        or not isinstance(handoff, dict)
        or handoff.get("m32_usable") is not True
        or handoff.get("partial") is not False
        or not isinstance(initial, dict)
        or not isinstance(initial.get("utility"), dict)
    ):
        raise _M32Failure(
            status,
            phase,
            "M31 result is not M32-usable or lacks initial-profile utility",
            candidate_id=candidate_id,
        )
    required_response_keys = {
        "response_game_identity",
        "response_run_identity",
        "content_identities",
        "counts",
        "support_cells",
        "support_pair_audit",
        "pure_profile_unilateral_stability",
        "utility_extrema",
        "hero_worst",
        "hero_best",
        "hero_worst_witnesses",
        "hero_best_witnesses",
        "hero_min_joint_plan_stress",
        "independent_verification",
        "limits",
        "ordering",
    }
    if (
        response.get("status") != _m30.EXACT_CORRESPONDENCE_COMPLETE
        or response.get("coverage") != "complete"
        or response.get("partial_response") is not False
        or not required_response_keys.issubset(response)
        or not isinstance(response["support_cells"], list)
        or not isinstance(response["support_pair_audit"], list)
        or not isinstance(response["hero_worst_witnesses"], list)
    ):
        raise _M32Failure(
            status,
            phase,
            "M30 correspondence is incomplete or missing lossless evidence",
            cause_status=response.get("status"),
            candidate_id=candidate_id,
        )
    for name in ("H", "O1", "O2", "R"):
        _parse_rational(initial["utility"].get(name), limits, f"{phase}.initial.{name}")
        extrema = response["utility_extrema"].get(name)
        if not isinstance(extrema, dict):
            raise _M32Failure(
                status, phase, f"missing {name} extrema", candidate_id=candidate_id
            )
    worst = _parse_rational(response["hero_worst"], limits, f"{phase}.hero_worst")
    best = _parse_rational(response["hero_best"], limits, f"{phase}.hero_best")
    if (
        worst > best
        or response["utility_extrema"]["H"].get("minimum")
        != response["hero_worst"]
        or response["utility_extrema"]["H"].get("maximum")
        != response["hero_best"]
        or response["utility_extrema"]["H"].get("minimum_witnesses")
        != response["hero_worst_witnesses"]
    ):
        raise _M32Failure(
            status,
            phase,
            "Hero extrema or all-witness correspondence mismatch",
            candidate_id=candidate_id,
        )
    identities = scenario_evaluation.get("identities")
    counts = scenario_evaluation.get("counts")
    if not isinstance(identities, dict) or not isinstance(counts, dict):
        raise _M32Failure(
            status, phase, "M31 identities/counts missing", candidate_id=candidate_id
        )
    for name in (
        "scenario",
        "tree_structure",
        "fixed_hero",
        "initial_profile",
        "rake_convention",
        "perfect_recall_evidence",
        "payoff_table",
        "evaluator_config",
        "response_game",
        "response_run",
    ):
        _validate_identity(identities.get(name), f"{phase}.identities.{name}", optional=False)
    return scenario_evaluation, response, initial["utility"]


def _feasible_counts(
    baseline: Mapping[str, Mapping[str, Fraction]],
    action_order: Mapping[str, tuple[str, ...]],
    shifts: tuple[Fraction, ...],
    maximum_info_sets: int,
    limits: ThreePlayerCandidateRepeatedLimits,
) -> tuple[dict[str, int], int, int, int]:
    q: dict[str, int] = {}
    for information_set in sorted(action_order):
        count = 0
        for source in action_order[information_set]:
            for target in action_order[information_set]:
                if source == target:
                    continue
                for shift in shifts:
                    if baseline[information_set][source] >= shift:
                        count = _checked_sum(
                            count,
                            1,
                            limits.max_generated_candidates,
                            "candidate_count.q_i",
                        )
        q[information_set] = count
    c1 = 0
    for count in q.values():
        c1 = _checked_sum(
            c1,
            count,
            limits.max_generated_candidates,
            "candidate_count.C1",
        )
    c2 = 0
    if maximum_info_sets == 2:
        info_sets = sorted(q)
        for left_index, left in enumerate(info_sets):
            for right in info_sets[left_index + 1 :]:
                product = _checked_product(
                    q[left],
                    q[right],
                    limits.max_generated_candidates,
                    "candidate_count.C2.product",
                )
                c2 = _checked_sum(
                    c2,
                    product,
                    limits.max_generated_candidates,
                    "candidate_count.C2.sum",
                )
    total = _checked_sum(
        c1,
        c2,
        limits.max_generated_candidates,
        "candidate_count.C",
    )
    identity_projection = (
        64
        + len(action_order) * 8
        + sum(len(actions) for actions in action_order.values()) * 4
        + len(shifts) * 2
        + total * 24
    )
    if identity_projection > limits.max_identity_records:
        raise _M32Failure(
            CAP_EXCEEDED,
            "candidate_count.identity_records",
            "max_identity_records exceeded before candidate materialisation",
        )
    return q, c1, c2, total


def _component_universe(
    baseline: Mapping[str, Mapping[str, Fraction]],
    action_order: Mapping[str, tuple[str, ...]],
    shifts: tuple[Fraction, ...],
) -> dict[str, tuple[_EditSpec, ...]]:
    output: dict[str, tuple[_EditSpec, ...]] = {}
    for information_set in sorted(action_order):
        components = []
        for source in action_order[information_set]:
            for target in action_order[information_set]:
                if source == target:
                    continue
                for shift in shifts:
                    if baseline[information_set][source] >= shift:
                        components.append(
                            _EditSpec(information_set, source, target, shift)
                        )
        output[information_set] = tuple(components)
    return output


def _policy_projection(
    policy: Mapping[str, Mapping[str, Fraction]],
    action_order: Mapping[str, tuple[str, ...]],
) -> list[dict[str, Any]]:
    return [
        {
            "information_set_id": information_set,
            "probabilities": {
                action: _rational_text(policy[information_set][action])
                for action in action_order[information_set]
            },
        }
        for information_set in sorted(action_order)
    ]


def _materialize_candidate_policy(
    baseline: Mapping[str, Mapping[str, Fraction]],
    action_order: Mapping[str, tuple[str, ...]],
    edits: tuple[_EditSpec, ...],
    tree_structure_identity: str,
    universe_identity: str,
    baseline_fixed_hero_identity: str,
    limits: ThreePlayerCandidateRepeatedLimits,
) -> ThreePlayerHeroCandidate:
    if (
        not edits
        or len(edits) > 2
        or len({edit.information_set_id for edit in edits}) != len(edits)
    ):
        raise _M32Failure(
            INTERNAL_FAILURE,
            "candidate_materialisation",
            "candidate edits must cover one or two distinct information sets",
        )
    mutable = {
        information_set: dict(distribution)
        for information_set, distribution in baseline.items()
    }
    shift_total = Fraction(0)
    for edit in edits:
        source = mutable[edit.information_set_id][edit.source_action_id]
        target = mutable[edit.information_set_id][edit.target_action_id]
        if source < edit.shift:
            raise _M32Failure(
                INTERNAL_FAILURE,
                "candidate_materialisation",
                "infeasible component reached materialisation",
            )
        mutable[edit.information_set_id][edit.source_action_id] = (
            _check_fraction_bits(
                source - edit.shift, limits, "candidate_materialisation"
            )
        )
        mutable[edit.information_set_id][edit.target_action_id] = (
            _check_fraction_bits(
                target + edit.shift, limits, "candidate_materialisation"
            )
        )
        shift_total = _check_fraction_bits(
            shift_total + edit.shift, limits, "candidate_l1"
        )
    projection = _policy_projection(mutable, action_order)
    resulting_policy_identity = _identity(
        {
            "tree_structure_identity": tree_structure_identity,
            "complete_policy": projection,
        }
    )
    candidate_projection = {
        "contract_version": CONTRACT_VERSION,
        "baseline_fixed_hero_identity": baseline_fixed_hero_identity,
        "candidate_universe_identity": universe_identity,
        "edits": [edit.projection() for edit in edits],
        "complete_policy": projection,
    }
    candidate_id = _identity(candidate_projection)
    policy = _m31.ExactBehaviorPolicy(
        {
            item["information_set_id"]: dict(item["probabilities"])
            for item in projection
        }
    )
    return ThreePlayerHeroCandidate(
        candidate_id=candidate_id,
        resulting_policy_identity=resulting_policy_identity,
        edits=tuple(
            HeroPolicyEdit(**edit.projection()) for edit in edits
        ),
        policy=policy,
        l1_distance_exact=_rational_text(
            _check_fraction_bits(
                Fraction(2) * shift_total, limits, "candidate_l1"
            )
        ),
    )


def _materialize_candidates(
    baseline: Mapping[str, Mapping[str, Fraction]],
    action_order: Mapping[str, tuple[str, ...]],
    components: Mapping[str, tuple[_EditSpec, ...]],
    maximum_info_sets: int,
    tree_structure_identity: str,
    universe_identity: str,
    baseline_fixed_hero_identity: str,
    limits: ThreePlayerCandidateRepeatedLimits,
) -> tuple[ThreePlayerHeroCandidate, ...]:
    edit_sets: list[tuple[_EditSpec, ...]] = [
        (component,)
        for information_set in sorted(components)
        for component in components[information_set]
    ]
    if maximum_info_sets == 2:
        info_sets = sorted(components)
        for left_index, left in enumerate(info_sets):
            for right in info_sets[left_index + 1 :]:
                edit_sets.extend(
                    (left_component, right_component)
                    for left_component in components[left]
                    for right_component in components[right]
                )
    candidates = []
    seen_policies: set[str] = set()
    for edits in edit_sets:
        candidate = _materialize_candidate_policy(
            baseline,
            action_order,
            edits,
            tree_structure_identity,
            universe_identity,
            baseline_fixed_hero_identity,
            limits,
        )
        if candidate.resulting_policy_identity in seen_policies:
            continue
        seen_policies.add(candidate.resulting_policy_identity)
        candidates.append(candidate)
    candidates.sort(
        key=lambda candidate: (
            tuple(
                (
                    edit.information_set_id,
                    action_order[edit.information_set_id].index(
                        edit.source_action_id
                    ),
                    action_order[edit.information_set_id].index(
                        edit.target_action_id
                    ),
                    Fraction(edit.shift_amount),
                )
                for edit in candidate.edits
            ),
            candidate.candidate_id,
        )
    )
    return tuple(candidates)


def _phase_b_caps(
    candidate_count: int,
    baseline_scenario: Mapping[str, Any],
    baseline_response: Mapping[str, Any],
    repeated: ThreePlayerRepeatedConfig,
    m31_limits: _m31.RiverRakeLimits,
    limits: ThreePlayerCandidateRepeatedLimits,
) -> dict[str, int]:
    runs = _checked_sum(
        candidate_count,
        1,
        limits.max_total_m31_runs,
        "response_preflight.total_m31_runs",
    )
    counts = baseline_scenario["counts"]
    joint = counts.get("joint_profiles")
    terminals = counts.get("terminals")
    support_pairs = baseline_response["counts"].get("support_pairs_total")
    effective_m30 = baseline_scenario["limits"].get("effective_m30")
    if (
        type(joint) is not int
        or type(terminals) is not int
        or type(support_pairs) is not int
        or not isinstance(effective_m30, dict)
        or type(effective_m30.get("max_exact_linear_systems")) is not int
    ):
        raise _M32Failure(
            BASELINE_RESPONSE_FAILURE,
            "response_preflight",
            "baseline workload counts are incomplete",
        )
    aggregate_joint = _checked_product(
        runs,
        joint,
        limits.max_aggregate_joint_profiles,
        "response_preflight.aggregate_joint_profiles",
    )
    m31_output_limits = baseline_scenario["limits"].get("m31")
    if (
        not isinstance(m31_output_limits, dict)
        or type(m31_output_limits.get("max_terminal_evaluations")) is not int
    ):
        raise _M32Failure(
            BASELINE_RESPONSE_FAILURE,
            "response_preflight",
            "baseline M31 terminal-evaluation limit is missing",
        )
    terminal_upper = m31_output_limits["max_terminal_evaluations"]
    aggregate_terminal = _checked_product(
        runs,
        terminal_upper,
        limits.max_aggregate_terminal_evaluations,
        "response_preflight.aggregate_terminal_evaluations",
    )
    aggregate_support = _checked_product(
        runs,
        support_pairs,
        limits.max_aggregate_support_pairs,
        "response_preflight.aggregate_support_pairs",
    )
    aggregate_systems = _checked_product(
        runs,
        effective_m30["max_exact_linear_systems"],
        limits.max_aggregate_exact_linear_systems,
        "response_preflight.aggregate_exact_linear_systems",
    )
    projected_embedded_bytes = _checked_product(
        runs,
        m31_limits.max_output_bytes,
        limits.max_embedded_m31_output_bytes,
        "response_preflight.embedded_m31_output_bytes",
    )
    projected_embedded_records = _checked_product(
        runs,
        m31_limits.max_output_records,
        limits.max_outer_output_records,
        "response_preflight.embedded_m31_output_records",
    )
    projected_timing = candidate_count * (repeated.horizon + 1)
    projected_wrapper_records = (
        1_024 + candidate_count * 256 + projected_timing * 8
    )
    if (
        projected_embedded_records
        > limits.max_outer_output_records - projected_wrapper_records
    ):
        raise _M32Failure(
            CAP_EXCEEDED,
            "response_preflight.outer_output_records",
            "projected wrapper/output records exceed M32 cap",
        )
    projected_wrapper_bytes = (
        65_536 + candidate_count * 8_192 + projected_timing * 1_024
    )
    if (
        projected_embedded_bytes
        > limits.max_outer_output_bytes - projected_wrapper_bytes
    ):
        raise _M32Failure(
            CAP_EXCEEDED,
            "response_preflight.outer_output_bytes",
            "projected wrapper/output bytes exceed M32 cap",
        )
    return {
        "total_m31_runs": runs,
        "aggregate_joint_profiles": aggregate_joint,
        "aggregate_terminal_evaluations": aggregate_terminal,
        "aggregate_support_pairs": aggregate_support,
        "aggregate_exact_linear_systems": aggregate_systems,
        "projected_embedded_m31_output_bytes": projected_embedded_bytes,
        "projected_embedded_m31_output_records": projected_embedded_records,
        "projected_wrapper_records": projected_wrapper_records,
        "projected_wrapper_bytes": projected_wrapper_bytes,
    }


def _exact_initial_values(
    utility: Mapping[str, Any],
    limits: ThreePlayerCandidateRepeatedLimits,
    phase: str,
) -> tuple[dict[str, str], dict[str, Fraction]]:
    texts: dict[str, str] = {}
    parsed: dict[str, Fraction] = {}
    for name in ("H", "O1", "O2", "R"):
        value = _parse_rational(utility.get(name), limits, f"{phase}.{name}")
        texts[name] = _rational_text(value)
        parsed[name] = value
    return texts, parsed


def _project_fraction(value: Fraction, phase: str) -> ExactScalarProjection:
    try:
        projection = float(value)
    except (OverflowError, ValueError):
        raise _M32Failure(
            SCALAR_PROJECTION_FAILURE,
            phase,
            "exact rational cannot be projected to finite binary64",
        ) from None
    if not math.isfinite(projection):
        raise _M32Failure(
            SCALAR_PROJECTION_FAILURE,
            phase,
            "exact rational projection is non-finite",
        )
    if value != 0 and projection == 0.0:
        raise _M32Failure(
            SCALAR_PROJECTION_FAILURE,
            phase,
            "nonzero exact rational underflows to binary64 zero",
        )
    roundtrip = Fraction.from_float(projection)
    error = abs(roundtrip - value)
    return ExactScalarProjection(
        source_exact=_rational_text(value),
        binary64=projection,
        binary64_exact_rational=_rational_text(roundtrip),
        absolute_error_exact=_rational_text(error),
        exact_in_binary64=(roundtrip == value),
    )


def _phase_c_timing_cap(
    candidate_count: int,
    repeated: ThreePlayerRepeatedConfig,
    selector_configuration: _m27.AutomaticCommitmentSelectionConfig,
    limits: ThreePlayerCandidateRepeatedLimits,
) -> int:
    rows = _checked_product(
        candidate_count,
        repeated.horizon + 1,
        limits.max_timing_row_evaluations,
        "timing_preflight.rows",
    )
    if candidate_count > selector_configuration.max_candidates:
        raise _M32Failure(
            CAP_EXCEEDED,
            "timing_preflight.selector_candidates",
            "M27 max_candidates exceeded before timing rows",
        )
    if rows > selector_configuration.max_timing_rows:
        raise _M32Failure(
            CAP_EXCEEDED,
            "timing_preflight.selector_rows",
            "M27 max_timing_rows exceeded before timing rows",
        )
    worst_tie_records = rows * (candidate_count + 2)
    if worst_tie_records > limits.max_outer_output_records:
        raise _M32Failure(
            CAP_EXCEEDED,
            "timing_preflight.tie_records",
            "worst-case full tie evidence exceeds output record cap",
        )
    return rows


def _float_deadline_scores(
    baseline: float,
    candidates: Sequence[tuple[str, float, float]],
    repeated: ThreePlayerRepeatedConfig,
) -> dict[int, dict[str, float]]:
    total_weight = 0.0
    term = 1.0
    for _ in range(repeated.horizon):
        total_weight += term
        term *= repeated.discount
    baseline_total = baseline * total_weight
    scores = {
        opportunity: {} for opportunity in range(1, repeated.horizon + 2)
    }
    for candidate_id, pre, post in candidates:
        prefix = 0.0
        term = 1.0
        for opportunity in range(1, repeated.horizon + 2):
            locked = pre * prefix + post * (total_weight - prefix)
            delta = locked - baseline_total
            if not math.isfinite(delta):
                raise _M32Failure(
                    NUMERIC_FAILURE,
                    "scalar_projection.repeated",
                    "derived binary64 repeated value is non-finite",
                )
            scores[opportunity][candidate_id] = delta
            prefix += term
            term *= repeated.discount
    return scores


def _decimal_deadline_scores(
    baseline: Fraction,
    candidates: Sequence[tuple[str, Fraction, Fraction]],
    repeated: ThreePlayerRepeatedConfig,
    limits: ThreePlayerCandidateRepeatedLimits,
) -> dict[int, dict[str, Decimal]]:
    precision = max(
        160,
        math.ceil(
            max(
                limits.max_rational_numerator_bits,
                limits.max_rational_denominator_bits,
            )
            * math.log10(2)
        )
        * 2
        + 80,
    )

    def decimal_fraction(value: Fraction) -> Decimal:
        return Decimal(value.numerator) / Decimal(value.denominator)

    with localcontext() as context:
        context.prec = precision
        discount = Decimal.from_float(repeated.discount)
        total_weight = Decimal(0)
        term = Decimal(1)
        for _ in range(repeated.horizon):
            total_weight += term
            term *= discount
        baseline_total = decimal_fraction(baseline) * total_weight
        scores = {
            opportunity: {}
            for opportunity in range(1, repeated.horizon + 2)
        }
        for candidate_id, pre_value, post_value in candidates:
            pre = decimal_fraction(pre_value)
            post = decimal_fraction(post_value)
            prefix = Decimal(0)
            term = Decimal(1)
            for opportunity in range(1, repeated.horizon + 2):
                scores[opportunity][candidate_id] = (
                    pre * prefix
                    + post * (total_weight - prefix)
                    - baseline_total
                )
                prefix += term
                term *= discount
        return scores


def _classification(
    scores: Mapping[str, Any],
    post_diffs: Mapping[str, Any],
    l1_values: Mapping[str, Any],
    tolerance: Any,
    threshold: Any,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], str | None, bool]:
    if not scores:
        return (), (), (), None, False
    candidate_ids = tuple(sorted(scores))
    best = max(scores.values())
    primary = tuple(
        candidate_id
        for candidate_id in candidate_ids
        if scores[candidate_id] >= best - tolerance
    )
    best_post = max(post_diffs[candidate_id] for candidate_id in primary)
    post_ties = tuple(
        candidate_id
        for candidate_id in primary
        if post_diffs[candidate_id] >= best_post - tolerance
    )
    best_l1 = min(l1_values[candidate_id] for candidate_id in post_ties)
    l1_ties = tuple(
        candidate_id
        for candidate_id in post_ties
        if l1_values[candidate_id] <= best_l1 + tolerance
    )
    return primary, post_ties, l1_ties, min(l1_ties), best > threshold + tolerance


def _validate_projection_safety(
    baseline: Fraction,
    candidate_exact: Sequence[tuple[str, Fraction, Fraction, Fraction]],
    baseline_projection: ExactScalarProjection,
    candidate_projections: Mapping[str, tuple[float, float, float]],
    repeated: ThreePlayerRepeatedConfig,
    selector_configuration: _m27.AutomaticCommitmentSelectionConfig,
    limits: ThreePlayerCandidateRepeatedLimits,
) -> dict[int, dict[str, float]]:
    float_inputs = [
        (
            candidate_id,
            candidate_projections[candidate_id][0],
            candidate_projections[candidate_id][1],
        )
        for candidate_id, _pre, _post, _l1 in candidate_exact
    ]
    float_scores = _float_deadline_scores(
        baseline_projection.binary64, float_inputs, repeated
    )
    decimal_scores = _decimal_deadline_scores(
        baseline,
        [
            (candidate_id, pre, post)
            for candidate_id, pre, post, _l1 in candidate_exact
        ],
        repeated,
        limits,
    )
    float_post = {
        candidate_id: candidate_projections[candidate_id][1]
        - baseline_projection.binary64
        for candidate_id, _pre, _post, _l1 in candidate_exact
    }
    float_l1 = {
        candidate_id: candidate_projections[candidate_id][2]
        for candidate_id, _pre, _post, _l1 in candidate_exact
    }
    precision = max(
        160,
        math.ceil(
            max(
                limits.max_rational_numerator_bits,
                limits.max_rational_denominator_bits,
            )
            * math.log10(2)
        )
        * 2
        + 80,
    )
    with localcontext() as context:
        context.prec = precision
        baseline_decimal = (
            Decimal(baseline.numerator) / Decimal(baseline.denominator)
        )
        decimal_post = {
            candidate_id: (
                Decimal(post.numerator) / Decimal(post.denominator)
                - baseline_decimal
            )
            for candidate_id, _pre, post, _l1 in candidate_exact
        }
        decimal_l1 = {
            candidate_id: Decimal(l1.numerator) / Decimal(l1.denominator)
            for candidate_id, _pre, _post, l1 in candidate_exact
        }
        decimal_tolerance = Decimal.from_float(repeated.tolerance)
        decimal_threshold = Decimal.from_float(
            selector_configuration.minimum_total_uplift
        )
        for opportunity in range(1, repeated.horizon + 2):
            exact_classification = _classification(
                decimal_scores[opportunity],
                decimal_post,
                decimal_l1,
                decimal_tolerance,
                decimal_threshold,
            )
            float_classification = _classification(
                float_scores[opportunity],
                float_post,
                float_l1,
                repeated.tolerance,
                selector_configuration.minimum_total_uplift,
            )
            if exact_classification != float_classification:
                raise _M32Failure(
                    SCALAR_PROJECTION_FAILURE,
                    "scalar_projection.selector_boundary",
                    "binary64 projection cannot safely preserve selector classification",
                )
    return float_scores


def _selector_output(
    report: _m27.AutomaticCommitmentSelectionReport,
) -> dict[str, Any]:
    raw = report.to_dict()
    return {
        "selector_kernel": SELECTOR_KERNEL,
        "kernel_output_identity": _identity(raw),
        "legacy_two_player_label_is_not_m32_semantics": True,
        "status": report.status,
        "baseline_identity": report.baseline_identity,
        "baseline_value_transport": raw["baseline_value"],
        "selection_configuration": raw["selection_configuration"],
        "search_coverage": raw["search_coverage"],
        "timing_row_evaluation_count": report.timing_row_evaluation_count,
        "rows": [row.to_dict() for row in report.rows],
    }


def _validate_selector_report(
    report: Any,
    candidates: tuple[ThreePlayerCandidateEvaluation, ...],
    repeated: ThreePlayerRepeatedConfig,
    timing_rows: int,
    float_scores: Mapping[int, Mapping[str, float]],
) -> None:
    if (
        type(report) is not _m27.AutomaticCommitmentSelectionReport
        or report.status != _m27.AUTOMATIC_COMMITMENT_SELECTION_STATUS_COMPLETE
        or len(report.rows) != repeated.horizon + 1
        or report.timing_row_evaluation_count != timing_rows
    ):
        raise _M32Failure(
            SELECTOR_FAILURE,
            "selector_output",
            "M27 returned an incomplete selector report",
        )
    candidate_ids = tuple(sorted(item.candidate.candidate_id for item in candidates))
    for opportunity, row in enumerate(report.rows, start=1):
        if row.adaptation_opportunity != opportunity:
            raise _M32Failure(
                SELECTOR_FAILURE,
                "selector_output",
                "noncanonical adaptation opportunity",
            )
        if not set(row.primary_tie_candidate_ids).issubset(candidate_ids):
            raise _M32Failure(
                SELECTOR_FAILURE,
                "selector_output",
                "selector returned an unknown primary-tie candidate",
            )
        evidence_ids = tuple(item.candidate_id for item in row.tie_break_evidence)
        if evidence_ids != row.primary_tie_candidate_ids:
            raise _M32Failure(
                SELECTOR_FAILURE,
                "selector_output",
                "full primary tie evidence is incomplete",
            )
        expected_scores = float_scores[opportunity]
        expected_best = max(expected_scores.values()) if expected_scores else None
        if row.best_total_hero_ev_delta != expected_best:
            raise _M32Failure(
                SELECTOR_FAILURE,
                "selector_output",
                "M27 row disagrees with independent repeated formula",
            )


def _record_count(value: Any, maximum: int) -> int:
    count = 0
    stack = [value]
    while stack:
        current = stack.pop()
        count += 1
        if count > maximum:
            raise _M32Failure(
                CAP_EXCEEDED,
                "output.records",
                "max_outer_output_records exceeded",
            )
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    return count


def _identity_record_count(value: Any, maximum: int) -> int:
    count = 0
    stack = [value]
    while stack:
        current = stack.pop()
        if type(current) is str and _IDENTITY_RE.fullmatch(current):
            count += 1
            if count > maximum:
                raise _M32Failure(
                    CAP_EXCEEDED,
                    "output.identities",
                    "max_identity_records exceeded",
                )
        elif isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    return count


def _evaluate(
    scenario: _m31.ThreePlayerRiverRakeScenario,
    baseline_fixed_hero_policy: _m31.ExactBehaviorPolicy,
    initial_profile: _m31.OpponentInitialProfile,
    *,
    attestation: _m31.PerfectRecallAttestation,
    generation: ThreePlayerCandidateGenerationConfig,
    repeated: ThreePlayerRepeatedConfig,
    selector_configuration: _m27.AutomaticCommitmentSelectionConfig,
    limits: ThreePlayerCandidateRepeatedLimits,
    m31_limits: _m31.RiverRakeLimits,
    m30_limits: _m30.ExactResponseLimits,
    m31_pins: _m31.RiverRakeIdentityPins,
    m30_response_pins: _m30.ResponseIdentityPins,
    pins: ThreePlayerCandidateRepeatedPins,
) -> ThreePlayerCandidateRepeatedResult:
    _validate_limits(limits)
    _validate_pins(pins)
    (
        shifts,
        generation_identity,
        repeated_identity,
        selector_transport_identity,
    ) = _validate_configuration(
        generation, repeated, selector_configuration, limits
    )
    _check_pin(
        pins, "candidate_generation_config_identity", generation_identity
    )
    _check_pin(pins, "repeated_config_identity", repeated_identity)
    _check_pin(pins, "selector_transport_identity", selector_transport_identity)
    _validate_initial_types(
        initial_profile,
        attestation,
        m31_limits,
        m30_limits,
        m31_pins,
        m30_response_pins,
    )
    action_order = _hero_action_order(scenario)
    baseline_policy, baseline_projection = _parse_complete_hero_policy(
        baseline_fixed_hero_policy, action_order, limits
    )

    baseline_result = _m31.evaluate_three_player_river_rake(
        scenario,
        baseline_fixed_hero_policy,
        attestation=attestation,
        initial_profile=initial_profile,
        limits=m31_limits,
        m30_limits=m30_limits,
        pins=m31_pins,
        response_pins=m30_response_pins,
    )
    baseline_scenario, baseline_response, baseline_utility = _validate_m31_result(
        baseline_result, limits, baseline=True
    )
    baseline_ids = baseline_scenario["identities"]
    expected_baseline_identity = _identity(
        {
            "tree_structure_identity": baseline_ids["tree_structure"],
            "complete_policy": baseline_projection,
        }
    )
    if expected_baseline_identity != baseline_ids["fixed_hero"]:
        raise _M32Failure(
            BASELINE_RESPONSE_FAILURE,
            "baseline_response.identity",
            "baseline fixed-Hero identity does not match the complete policy",
        )
    for pin_name, identity_name in (
        ("scenario_identity", "scenario"),
        ("tree_structure_identity", "tree_structure"),
        ("baseline_fixed_hero_identity", "fixed_hero"),
        ("initial_profile_identity", "initial_profile"),
    ):
        _check_pin(pins, pin_name, baseline_ids[identity_name])

    q_counts, c1, c2, theoretical_count = _feasible_counts(
        baseline_policy,
        action_order,
        shifts,
        generation.max_simultaneous_info_sets,
        limits,
    )
    components = _component_universe(baseline_policy, action_order, shifts)
    if {key: len(value) for key, value in components.items()} != q_counts:
        raise _M32Failure(
            INTERNAL_FAILURE,
            "candidate_generation",
            "feasible component count changed during materialisation",
        )
    universe_projection = {
        "contract_version": CONTRACT_VERSION,
        "candidate_generation_config_identity": generation_identity,
        "baseline_fixed_hero_identity": baseline_ids["fixed_hero"],
        "tree_structure_identity": baseline_ids["tree_structure"],
        "semantic_action_order": {
            key: list(value) for key, value in action_order.items()
        },
        "feasible_components": [
            component.projection()
            for information_set in sorted(components)
            for component in components[information_set]
        ],
        "counts_before_dedup": {
            "q_by_information_set": q_counts,
            "one_information_set": c1,
            "two_information_sets": c2,
            "total": theoretical_count,
        },
    }
    universe_identity = _identity(universe_projection)
    _check_pin(pins, "candidate_universe_identity", universe_identity)
    candidates = _materialize_candidates(
        baseline_policy,
        action_order,
        components,
        generation.max_simultaneous_info_sets,
        baseline_ids["tree_structure"],
        universe_identity,
        baseline_ids["fixed_hero"],
        limits,
    )
    candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise _M32Failure(
            INTERNAL_FAILURE,
            "candidate_generation",
            "duplicate candidate identity after canonical dedup",
        )
    _check_pin(pins, "ordered_candidate_ids", candidate_ids)

    phase_b = _phase_b_caps(
        len(candidates),
        baseline_scenario,
        baseline_response,
        repeated,
        m31_limits,
        limits,
    )
    baseline_exact_text, baseline_exact = _exact_initial_values(
        baseline_utility, limits, "baseline_initial"
    )
    raw_evaluations: list[tuple[Any, ...]] = []
    for candidate in candidates:
        result = _m31.evaluate_three_player_river_rake(
            scenario,
            candidate.policy,
            attestation=attestation,
            initial_profile=initial_profile,
            limits=m31_limits,
            m30_limits=m30_limits,
            pins=m31_pins,
            response_pins=m30_response_pins,
        )
        candidate_scenario, response, utility = _validate_m31_result(
            result,
            limits,
            baseline=False,
            candidate_id=candidate.candidate_id,
        )
        candidate_m31_ids = candidate_scenario["identities"]
        if (
            candidate_m31_ids["fixed_hero"]
            != candidate.resulting_policy_identity
            or candidate_m31_ids["scenario"] != baseline_ids["scenario"]
            or candidate_m31_ids["tree_structure"]
            != baseline_ids["tree_structure"]
            or candidate_m31_ids["initial_profile"]
            != baseline_ids["initial_profile"]
            or candidate_m31_ids["perfect_recall_evidence"]
            != baseline_ids["perfect_recall_evidence"]
            or candidate_m31_ids["rake_convention"]
            != baseline_ids["rake_convention"]
        ):
            raise _M32Failure(
                CANDIDATE_RESPONSE_FAILURE,
                "candidate_response.identity",
                "candidate M31 source identities do not bind the fixed universe",
                candidate_id=candidate.candidate_id,
            )
        initial_text, initial_exact = _exact_initial_values(
            utility,
            limits,
            f"candidate.{candidate.candidate_id}.initial",
        )
        worst = _parse_rational(
            response["hero_worst"],
            limits,
            f"candidate.{candidate.candidate_id}.response.hero_worst",
        )
        best = _parse_rational(
            response["hero_best"],
            limits,
            f"candidate.{candidate.candidate_id}.response.hero_best",
        )
        l1 = _parse_rational(
            candidate.l1_distance_exact,
            limits,
            f"candidate.{candidate.candidate_id}.l1",
        )
        run_identity = _identity(
            {
                "contract_version": CONTRACT_VERSION,
                "algorithm_version": ALGORITHM_VERSION,
                "candidate_identity": candidate.candidate_id,
                "fixed_hero_identity": candidate_m31_ids["fixed_hero"],
                "payoff_table_identity": candidate_m31_ids["payoff_table"],
                "response_game_identity": candidate_m31_ids["response_game"],
                "response_run_identity": candidate_m31_ids["response_run"],
                "initial_profile_identity": candidate_m31_ids["initial_profile"],
                "search_mode": generation.search_mode,
                "versions": {
                    "m31": _m31.CONTRACT_VERSION,
                    "m30": _m30.CONTRACT_VERSION,
                    "m27": SELECTOR_KERNEL,
                },
            }
        )
        raw_evaluations.append(
            (
                candidate,
                result,
                candidate_m31_ids,
                initial_text,
                initial_exact,
                worst,
                best,
                l1,
                run_identity,
            )
        )

    fixed_hero_mapping = tuple(
        (item[0].candidate_id, item[2]["fixed_hero"])
        for item in raw_evaluations
    )
    response_game_mapping = tuple(
        (item[0].candidate_id, item[1].response["response_game_identity"])
        for item in raw_evaluations
    )
    response_run_mapping = tuple(
        (item[0].candidate_id, item[1].response["response_run_identity"])
        for item in raw_evaluations
    )
    candidate_run_mapping = tuple(
        (item[0].candidate_id, item[8]) for item in raw_evaluations
    )
    for name, actual in (
        ("candidate_fixed_hero_identities", fixed_hero_mapping),
        ("candidate_response_game_identities", response_game_mapping),
        ("candidate_response_run_identities", response_run_mapping),
        ("candidate_run_identities", candidate_run_mapping),
    ):
        _check_pin(pins, name, actual)

    timing_rows = _phase_c_timing_cap(
        len(raw_evaluations),
        repeated,
        selector_configuration,
        limits,
    )
    baseline_scalar = {
        name: _project_fraction(value, f"scalar_projection.baseline.{name}")
        for name, value in baseline_exact.items()
    }
    baseline_opponents = _check_fraction_bits(
        baseline_exact["O1"] + baseline_exact["O2"],
        limits,
        "scalar_projection.baseline.O1_plus_O2",
    )
    baseline_scalar["O1_plus_O2"] = _project_fraction(
        baseline_opponents, "scalar_projection.baseline.O1_plus_O2"
    )
    candidate_evaluations: list[ThreePlayerCandidateEvaluation] = []
    candidate_exact_for_safety: list[
        tuple[str, Fraction, Fraction, Fraction]
    ] = []
    candidate_projection_for_safety: dict[
        str, tuple[float, float, float]
    ] = {}
    value_candidates: list[_m27.AutomaticCommitmentValueCandidate] = []
    for (
        candidate,
        result,
        _candidate_m31_ids,
        initial_text,
        initial_exact,
        worst,
        best,
        l1,
        run_identity,
    ) in raw_evaluations:
        projections = {
            name: _project_fraction(
                value,
                f"scalar_projection.candidate.{candidate.candidate_id}.{name}",
            )
            for name, value in initial_exact.items()
        }
        opponents = _check_fraction_bits(
            initial_exact["O1"] + initial_exact["O2"],
            limits,
            f"candidate.{candidate.candidate_id}.O1_plus_O2",
        )
        projections["O1_plus_O2"] = _project_fraction(
            opponents,
            f"scalar_projection.candidate.{candidate.candidate_id}.O1_plus_O2",
        )
        projections["hero_worst"] = _project_fraction(
            worst,
            f"scalar_projection.candidate.{candidate.candidate_id}.hero_worst",
        )
        projections["hero_best"] = _project_fraction(
            best,
            f"scalar_projection.candidate.{candidate.candidate_id}.hero_best",
        )
        projections["l1_distance"] = _project_fraction(
            l1,
            f"scalar_projection.candidate.{candidate.candidate_id}.l1",
        )
        exact_worst_diff = _check_fraction_bits(
            worst - baseline_exact["H"],
            limits,
            f"candidate.{candidate.candidate_id}.worst_diff",
        )
        projected_worst_diff = (
            projections["hero_worst"].binary64
            - baseline_scalar["H"].binary64
        )
        if not math.isfinite(projected_worst_diff):
            raise _M32Failure(
                SCALAR_PROJECTION_FAILURE,
                f"candidate.{candidate.candidate_id}.worst_diff",
                "binary64 worst-minus-baseline difference is non-finite",
            )
        exact_values = {
            "initial_profile": initial_text,
            "response": {
                "hero_worst": _rational_text(worst),
                "hero_best": _rational_text(best),
                "safety_scalar_source_path": (
                    "m31_scenario_response.response.hero_worst"
                ),
            },
            "hero_worst_minus_baseline_H": _rational_text(exact_worst_diff),
            "selector_worst_diff_transport": {
                "exact_source": _rational_text(exact_worst_diff),
                "binary64_float_worst_minus_float_baseline": (
                    projected_worst_diff
                ),
            },
            "l1_distance": candidate.l1_distance_exact,
        }
        candidate_evaluations.append(
            ThreePlayerCandidateEvaluation(
                candidate=candidate,
                scenario_response=result,
                exact_values=exact_values,
                scalar_projections=projections,
                candidate_run_identity=run_identity,
            )
        )
        candidate_exact_for_safety.append(
            (candidate.candidate_id, initial_exact["H"], worst, l1)
        )
        candidate_projection_for_safety[candidate.candidate_id] = (
            projections["H"].binary64,
            projections["hero_worst"].binary64,
            projections["l1_distance"].binary64,
        )
        value_candidates.append(
            _m27.AutomaticCommitmentValueCandidate(
                candidate_id=candidate.candidate_id,
                fixed_profile_value=FixedProfileValue(
                    hero_ev=projections["H"].binary64,
                    villain_ev=projections["O1_plus_O2"].binary64,
                    house_rake=projections["R"].binary64,
                ),
                post_response_hero_ev_worst=(
                    projections["hero_worst"].binary64
                ),
                post_response_hero_ev_best=(
                    projections["hero_best"].binary64
                ),
                post_response_hero_ev_worst_diff=projected_worst_diff,
                l1_distance=projections["l1_distance"].binary64,
            )
        )
    float_scores = _validate_projection_safety(
        baseline_exact["H"],
        candidate_exact_for_safety,
        baseline_scalar["H"],
        candidate_projection_for_safety,
        repeated,
        selector_configuration,
        limits,
    )
    coverage = _m27.AutomaticCommitmentSearchCoverage(
        input_candidate_ids=candidate_ids,
        kept_candidate_ids=candidate_ids,
        source="m32_robust_all_complete",
        shift_amounts=tuple(
            _project_fraction(value, "scalar_projection.shift_amount").binary64
            for value in shifts
        ),
        max_simultaneous_info_sets=generation.max_simultaneous_info_sets,
        generation_max_candidates=limits.max_generated_candidates,
        filtering_applied=False,
    )
    try:
        selector_report = _m27.select_automatic_commitment_values(
            FixedProfileValue(
                hero_ev=baseline_scalar["H"].binary64,
                villain_ev=baseline_scalar["O1_plus_O2"].binary64,
                house_rake=baseline_scalar["R"].binary64,
            ),
            tuple(value_candidates),
            horizon=repeated.horizon,
            discount=repeated.discount,
            tolerance=repeated.tolerance,
            max_horizon=limits.max_horizon,
            configuration=selector_configuration,
            search_coverage=coverage,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise _M32Failure(
            SELECTOR_FAILURE, "selector", str(exc)
        ) from None
    _validate_selector_report(
        selector_report,
        tuple(candidate_evaluations),
        repeated,
        timing_rows,
        float_scores,
    )
    selector_output_identity = _identity(selector_report.to_dict())
    m32_run_identity = _identity(
        {
            "contract_version": CONTRACT_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "baseline_response_run_identity": baseline_ids["response_run"],
            "ordered_candidate_ids": list(candidate_ids),
            "candidate_run_identities": list(candidate_run_mapping),
            "candidate_generation_config_identity": generation_identity,
            "candidate_universe_identity": universe_identity,
            "repeated_config_identity": repeated_identity,
            "selector_transport_identity": selector_transport_identity,
            "selector_output_identity": selector_output_identity,
            "effective_caps": {
                "m32": asdict(limits),
                "m31": asdict(m31_limits),
                "m30": asdict(m30_limits),
                "m27": selector_configuration.to_dict(),
            },
        }
    )
    _check_pin(pins, "m32_run_identity", m32_run_identity)

    payload = {
        "contract_version": CONTRACT_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "projection_version": PROJECTION_VERSION,
        "status": EXACT_THREE_PLAYER_CANDIDATE_REPEATED_COMPLETE,
        "coverage": "complete",
        "partial_result": False,
        "search_mode": generation.search_mode,
        "adaptation_mode": generation.adaptation_mode,
        "response_semantics": RESPONSE_SEMANTICS,
        "selector_kernel": SELECTOR_KERNEL,
        "selector_transport_opponent_value": (
            SELECTOR_TRANSPORT_OPPONENT_VALUE
        ),
        "claim_boundary": {
            "stationary_one_shot_sensitivity_analysis": True,
            "repeated_game_equilibrium_claim": False,
            "opponent_learning_or_switch_prediction": False,
            "coalition_or_transferable_utility_claim": False,
            "current_cfr_used": False,
            "hero_safety_source": (
                "each candidate M31 exact response.hero_worst only"
            ),
        },
        "identities": {
            "candidate_generation_config": generation_identity,
            "candidate_universe": universe_identity,
            "repeated_config": repeated_identity,
            "selector_transport": selector_transport_identity,
            "selector_output": selector_output_identity,
            "m32_run": m32_run_identity,
            "ordered_candidate_ids": list(candidate_ids),
            "candidate_run_identities": [
                {"candidate_id": key, "identity": value}
                for key, value in candidate_run_mapping
            ],
        },
        "candidate_generation": {
            "configuration": {
                "shift_amounts": [
                    _rational_text(value) for value in shifts
                ],
                "max_simultaneous_info_sets": (
                    generation.max_simultaneous_info_sets
                ),
            },
            "semantic_action_order": {
                key: list(value) for key, value in action_order.items()
            },
            "counts_before_dedup": universe_projection[
                "counts_before_dedup"
            ],
            "candidate_count_after_exact_policy_dedup": len(candidates),
            "ordering": (
                "information-set ID; source outer/target inner tree action "
                "order; numeric shift; one-edit before two-distinct-info-set"
            ),
        },
        "repeated_value_convention": {
            "baseline": "sum(t=1..N,discount^(t-1)*baseline_initial_H)",
            "candidate": (
                "t<m candidate_initial_H; t>=m candidate_response_hero_worst"
            ),
            "opportunities": f"1..{repeated.horizon + 1}",
            "horizon": repeated.horizon,
            "discount": repeated.discount,
            "tolerance": repeated.tolerance,
            "minimum_total_uplift": (
                selector_configuration.minimum_total_uplift
            ),
        },
        "caps": {
            "m32": asdict(limits),
            "m31": asdict(m31_limits),
            "m30": asdict(m30_limits),
            "m27": selector_configuration.to_dict(),
            "phase_b_preflight": phase_b,
            "timing_row_evaluations": timing_rows,
        },
    }
    analysis = ThreePlayerCandidateRepeatedAnalysis(
        baseline_response=baseline_result,
        baseline_exact_values=baseline_exact_text,
        baseline_scalar_projections=baseline_scalar,
        candidates=tuple(candidate_evaluations),
        selector_report=selector_report,
        payload=payload,
    )
    success = ThreePlayerCandidateRepeatedResult(
        status=EXACT_THREE_PLAYER_CANDIDATE_REPEATED_COMPLETE,
        analysis=analysis,
        error=None,
        partial_result=False,
    )
    output = success.to_dict()
    embedded_bytes = sum(
        len(_canonical_json_bytes(result.to_dict()))
        for result in (
            baseline_result,
            *(item.scenario_response for item in candidate_evaluations),
        )
    )
    if embedded_bytes > limits.max_embedded_m31_output_bytes:
        raise _M32Failure(
            CAP_EXCEEDED,
            "output.embedded_m31_bytes",
            "actual embedded M31 output bytes exceed cap",
        )
    output_records = _record_count(output, limits.max_outer_output_records)
    identity_records = _identity_record_count(
        output, limits.max_identity_records
    )
    encoded = _canonical_json_bytes(output)
    if len(encoded) > limits.max_outer_output_bytes:
        raise _M32Failure(
            CAP_EXCEEDED,
            "output.bytes",
            "max_outer_output_bytes exceeded",
        )
    payload["caps"]["actual"] = {
        "embedded_m31_output_bytes": embedded_bytes,
        "outer_output_records_before_actual_record": output_records,
        "identity_records_before_actual_record": identity_records,
        "outer_output_bytes_before_actual_record": len(encoded),
    }
    final_output = success.to_dict()
    _record_count(final_output, limits.max_outer_output_records)
    _identity_record_count(final_output, limits.max_identity_records)
    if len(_canonical_json_bytes(final_output)) > limits.max_outer_output_bytes:
        raise _M32Failure(
            CAP_EXCEEDED,
            "output.bytes",
            "max_outer_output_bytes exceeded after final cap evidence",
        )
    return success


def evaluate_three_player_candidate_repeated(
    scenario: _m31.ThreePlayerRiverRakeScenario,
    baseline_fixed_hero_policy: _m31.ExactBehaviorPolicy,
    initial_profile: _m31.OpponentInitialProfile,
    *,
    attestation: _m31.PerfectRecallAttestation,
    generation: ThreePlayerCandidateGenerationConfig = (
        ThreePlayerCandidateGenerationConfig()
    ),
    repeated: ThreePlayerRepeatedConfig = ThreePlayerRepeatedConfig(),
    selector_configuration: _m27.AutomaticCommitmentSelectionConfig = (
        _m27.AutomaticCommitmentSelectionConfig()
    ),
    limits: ThreePlayerCandidateRepeatedLimits = (
        ThreePlayerCandidateRepeatedLimits()
    ),
    m31_limits: _m31.RiverRakeLimits = _m31.RiverRakeLimits(),
    m30_limits: _m30.ExactResponseLimits = _m30.ExactResponseLimits(),
    m31_pins: _m31.RiverRakeIdentityPins = _m31.RiverRakeIdentityPins(),
    m30_response_pins: _m30.ResponseIdentityPins = _m30.ResponseIdentityPins(),
    pins: ThreePlayerCandidateRepeatedPins = (
        ThreePlayerCandidateRepeatedPins()
    ),
) -> ThreePlayerCandidateRepeatedResult:
    """Evaluate the complete bounded M32 robust-all universe or no analysis.

    The baseline M31 evaluator runs exactly once.  After candidate-count and
    aggregate workload preflights, every canonical candidate is evaluated by a
    fresh public M31 call.  Only direct ``response.hero_worst`` values enter the
    public M27 domain-neutral selector.  Any failure returns ``analysis=None``
    and ``partial_result=False``.
    """

    try:
        return _evaluate(
            scenario,
            baseline_fixed_hero_policy,
            initial_profile,
            attestation=attestation,
            generation=generation,
            repeated=repeated,
            selector_configuration=selector_configuration,
            limits=limits,
            m31_limits=m31_limits,
            m30_limits=m30_limits,
            m31_pins=m31_pins,
            m30_response_pins=m30_response_pins,
            pins=pins,
        )
    except _M32Failure as exc:
        return _failure(exc)
    except Exception:
        return _failure(
            _M32Failure(
                INTERNAL_FAILURE,
                "internal",
                "unexpected three-player candidate/repeated integration failure",
            )
        )


def exact_three_player_candidate_repeated_json(
    result: ThreePlayerCandidateRepeatedResult,
) -> str:
    """Serialize one M32 outer result as deterministic strict one-line JSON."""

    if type(result) is not ThreePlayerCandidateRepeatedResult:
        raise TypeError("result must be ThreePlayerCandidateRepeatedResult")
    return _canonical_json_bytes(result.to_dict()).decode("utf-8")
