"""Certified global real-card AIoF preflop commitment integration.

This module connects the M36 exact-rational certified global optimizer to the
M28 heads-up, fee-zero, real-card AIoF preflop semantics without changing the
existing finite-shift bridge.

The complete Hero domain is derived after public dead-card and joint blocker
conditioning.  Every surviving exact Hero combo is one information set with
both legal push/fold actions.  No candidate list, shift amount, grid, local
bound, or warm-start neighbourhood is accepted.

For a fixed adaptation opportunity ``m`` in a horizon ``N``, let ``b`` be the
baseline fixed-profile Hero ChipEV, ``a(p)`` the Hero ChipEV of policy ``p``
against the fixed baseline opponent, and ``l(p)`` the Hero-worst value over the
fresh complete opponent best-response correspondence.  With exact discount
weights

``W_pre = sum(t=0..m-2, delta**t)`` and
``W_post = sum(t=m-1..N-1, delta**t)``,

the locked candidate total is

``V(p) = W_pre*a(p) + W_post*l(p)``

and the comparison total is ``V_base = (W_pre + W_post)*b``.  The baseline
policy itself denotes the no-commitment comparison and therefore has exactly
zero uplift, as required by the M36 core.  A non-baseline policy uses the
locked formula above.

Both ``a`` and every pure opponent-action contribution are affine in the Hero
behavior probabilities.  The fresh response has the factorized form

``l(p) = c(p) + sum_j min_r L[j,r](p)``.

For a closed M36 cell ``C`` intersected with every two-action simplex,

``max_C l <= max_C c + sum_j min_r max_C L[j,r]``.

All repeated weights are nonnegative, so the consumer whole-cell uplift bound

``max(0, W_pre*max_C a
          + W_post*(max_C c + sum_j min_r max_C L[j,r])
          - V_base)``

is sound for the actual M37 objective.  Each affine maximum uses the effective
active-action interval implied by both action intervals and the exact
sum-to-one constraint.  It is not a sampled, vertex-only, local, metadata, or
finite-candidate bound.  At a singleton cell it equals the point objective,
which gives the M36 branch-and-bound a sound convergent consumer oracle.

Range weights, baseline probabilities, and game amounts supplied through the
existing binary64 M13/M28 dataclasses are lifted losslessly with
``float.as_integer_ratio()``.  No decimal approximation or tolerance is
promoted into a certificate.  Exact exhaustive integer showdown counts and
the declared exact discount rational are the remaining coefficient inputs.
Non-finite values, Monte Carlo, nonzero response tolerance, or any domain for
which this construction cannot be made are rejected without a partial
payload.

Success claims only the specified-tolerance global maximum of this identified
real-card preflop scalar objective.  It does not claim equilibrium,
solver-grade scalability, ICM, real-world profitability, strategy advice, or
unlimited game or range size.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from numbers import Real
from typing import Any, Callable

from .aiof_cards import (
    AiofContractError,
    AiofLimits,
    AiofStatus,
    PreparedRanges,
    RangeSpec,
    WeightBasis,
    canonicalize_exact_combo,
    canonicalize_hand_class,
    prepare_compatible_ranges,
)
from .aiof_chip_ev import (
    CHIP_ACCOUNTING_ID,
    ComboActionProbability,
    HeadsUpChipEvGame,
    SuppliedProfile,
)
from .aiof_equity import (
    EquityAlgorithm,
    _iter_exact_matchup_outcomes,
)
from .certified_global_optimizer import (
    BOUND_CONTRACT_VERSION,
    CERTIFIED_EPSILON_GLOBAL,
    CERTIFIED_GLOBAL,
    INVALID_INPUT,
    LIMIT_REACHED_NO_CERTIFICATE,
    NUMERIC_FAILURE,
    OBJECTIVE_SEMANTICS,
    ORACLE_FAILURE,
    RESPONSE_SEMANTICS,
    STALE_INPUT,
    UNSUPPORTED_DOMAIN,
    CertifiedGlobalOptimizerLimits,
    CertifiedGlobalOptimizerPins,
    CertifiedGlobalOptimizerResult,
    CertifiedGlobalWorkCounters,
    ExactActionProbability,
    ExactBehaviorCell,
    ExactBehaviorPolicy,
    ExactBehaviorRow,
    HeroBehaviorScenario,
    HeroInformationSet,
    ScalarOracleBound,
    ScalarOracleEvaluation,
    ScalarOracleResourceLimit,
    UnsupportedScalarOracleDomain,
    behavior_cell_identity,
    behavior_policy_identity,
    optimize_certified_global_hero_commitment,
    scalar_oracle_bound_identity,
)
from .repeated import DEFAULT_MAX_HORIZON


__all__ = [
    "AIOF_PREFLOP_CERTIFIED_GLOBAL_CONTRACT_VERSION",
    "AiofPreflopCertifiedGlobalLimits",
    "AiofPreflopCertifiedGlobalPins",
    "AiofPreflopCertifiedGlobalRequest",
    "AiofPreflopCompleteResponseRow",
    "AiofPreflopCompleteResponse",
    "AiofPreflopCertifiedPointRecord",
    "AiofPreflopCertifiedOracleWorkCounters",
    "AiofPreflopCertifiedGlobalPreparation",
    "AiofPreflopCertifiedGlobalPayload",
    "AiofPreflopCertifiedGlobalError",
    "AiofPreflopCertifiedGlobalResult",
    "AiofPreflopCertifiedScalarOracle",
    "prepare_aiof_preflop_certified_global_oracle",
    "analyze_aiof_preflop_certified_global",
    "exact_aiof_preflop_certified_global_json",
]


AIOF_PREFLOP_CERTIFIED_GLOBAL_CONTRACT_VERSION = (
    "m37-real-card-aiof-preflop-certified-global-v1"
)
AIOF_PREFLOP_CERTIFIED_GLOBAL_ALGORITHM = (
    "exact-real-card-factorized-response-affine-cell-bound-v1"
)
AIOF_PREFLOP_CERTIFIED_GLOBAL_DOMAIN = (
    "blocker-conditioned-surviving-exact-combo-full-push-fold-product-simplex-v1"
)
AIOF_PREFLOP_CERTIFIED_GLOBAL_OBJECTIVE = (
    "fixed-adaptation-opportunity-baseline-relative-total-repeated-hero-chipev-v1"
)
AIOF_PREFLOP_CERTIFIED_GLOBAL_RESPONSE = (
    "fresh-factorized-complete-exact-opponent-best-response-hero-worst-v1"
)
AIOF_PREFLOP_CERTIFIED_GLOBAL_BOUND = (
    "nonnegative-weight-affine-min-cell-upper-bound-v1"
)
AIOF_PREFLOP_CERTIFIED_GLOBAL_CLAIM_SCOPE = (
    "specified-tolerance-global-maximum-of-identified-real-card-preflop-scalar-objective-v1"
)

MAX_INFORMATION_SETS_PER_SEAT = 1_326
MAX_COMPATIBLE_PAIRS = 1_624_350
MAX_MATCHUP_ROWS = 1_624_350
MAX_EXACT_BOARD_EVALUATIONS = 10_000_000
MAX_COEFFICIENT_RECORDS = 10_000_000
MAX_TOTAL_RESPONSE_ROWS = 4_000_000
MAX_ORACLE_BOUND_RECORDS = 1_048_575
MAX_OUTPUT_RECORDS = 2_000_000
MAX_OUTPUT_BYTES = 64_000_000

_IDENTITY_RE_LENGTH = 64


def _rational_text(value: Fraction | int) -> str:
    rational = Fraction(value)
    if rational.denominator == 1:
        return str(rational.numerator)
    return f"{rational.numerator}/{rational.denominator}"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _identity(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_identity(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != _IDENTITY_RE_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _exact_binary64(value: object, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, f"{name} must be a finite real number"
        )
    try:
        binary = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            f"{name} is not representable as binary64",
        ) from exc
    if not math.isfinite(binary):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, f"{name} must be finite"
        )
    numerator, denominator = binary.as_integer_ratio()
    return Fraction(numerator, denominator)


def _parse_exact_rational(
    value: object,
    name: str,
    *,
    nonnegative: bool = False,
) -> Fraction:
    if type(value) is int:
        rational = Fraction(value)
    elif type(value) is str:
        try:
            rational = Fraction(value)
        except (ValueError, ZeroDivisionError) as exc:
            raise AiofContractError(
                AiofStatus.INVALID_INPUT,
                f"{name} must be an exact integer or rational string",
            ) from exc
        if _rational_text(rational) != value:
            raise AiofContractError(
                AiofStatus.INVALID_INPUT,
                f"{name} must use canonical exact rational text",
            )
    else:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            f"{name} must be an exact integer or rational string",
        )
    if nonnegative and rational < 0:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, f"{name} must be nonnegative"
        )
    return rational


def _require_plain_int(
    value: object, name: str, *, minimum: int = 1
) -> int:
    if type(value) is not int or value < minimum:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            f"{name} must be an integer at least {minimum}",
        )
    return value


@dataclass(frozen=True)
class AiofPreflopCertifiedGlobalLimits:
    """Caller-lowerable integration caps with immutable hard ceilings."""

    max_information_sets_per_seat: int = MAX_INFORMATION_SETS_PER_SEAT
    max_horizon: int = DEFAULT_MAX_HORIZON
    max_compatible_pairs: int = MAX_COMPATIBLE_PAIRS
    max_matchup_rows: int = MAX_MATCHUP_ROWS
    max_exact_board_evaluations: int = MAX_EXACT_BOARD_EVALUATIONS
    max_coefficient_records: int = MAX_COEFFICIENT_RECORDS
    max_total_response_rows: int = MAX_TOTAL_RESPONSE_ROWS
    max_oracle_bound_records: int = MAX_ORACLE_BOUND_RECORDS
    max_output_records: int = MAX_OUTPUT_RECORDS
    max_output_bytes: int = MAX_OUTPUT_BYTES

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class AiofPreflopCertifiedGlobalPins:
    """Optional M37 identity pins checked before the M36 oracle is called."""

    prepared_ranges_identity: str | None = None
    scenario_identity: str | None = None
    response_oracle_identity: str | None = None
    objective_identity: str | None = None
    analysis_identity: str | None = None


@dataclass(frozen=True)
class AiofPreflopCertifiedGlobalRequest:
    """One exact real-card certified-global preflop request.

    ``discount`` and both gap tolerances are canonical exact rational strings
    (or plain integers).  ``response_tolerance`` is deliberately restricted to
    exact zero: a floating/tolerance tie cannot be elevated into a global
    certificate.
    """

    game: HeadsUpChipEvGame
    sb_range: RangeSpec
    bb_range: RangeSpec
    dead_cards: tuple[str, ...]
    baseline_profile: SuppliedProfile
    hero_seat: str
    horizon: int
    adaptation_opportunity: int
    discount: str = "1"
    response_tolerance: str = "0"
    absolute_gap_tolerance: str = "0"
    relative_gap_tolerance: str = "0"
    algorithm: EquityAlgorithm = EquityAlgorithm.EXACT_EXHAUSTIVE
    aiof_limits: AiofLimits = AiofLimits()
    integration_limits: AiofPreflopCertifiedGlobalLimits = (
        AiofPreflopCertifiedGlobalLimits()
    )
    optimizer_limits: CertifiedGlobalOptimizerLimits = (
        CertifiedGlobalOptimizerLimits()
    )
    pins: AiofPreflopCertifiedGlobalPins = (
        AiofPreflopCertifiedGlobalPins()
    )
    optimizer_pins: CertifiedGlobalOptimizerPins = (
        CertifiedGlobalOptimizerPins()
    )


@dataclass(frozen=True)
class AiofPreflopCompleteResponseRow:
    """One factorized opponent information-set response row."""

    information_set_id: str
    combo: str
    compatible_probability: str
    information_reach_probability: str
    action_hero_contributions: tuple[tuple[str, str], ...]
    best_actions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "information_set_id": self.information_set_id,
            "combo": self.combo,
            "compatible_probability": self.compatible_probability,
            "information_reach_probability": (
                self.information_reach_probability
            ),
            "action_hero_contributions": [
                {"action": action, "hero_contribution": value}
                for action, value in self.action_hero_contributions
            ],
            "best_actions": list(self.best_actions),
        }


@dataclass(frozen=True)
class AiofPreflopCompleteResponse:
    """Fresh complete factorized opponent best-response correspondence."""

    opponent_seat: str
    hero_worst_value: str
    rows: tuple[AiofPreflopCompleteResponseRow, ...]
    correspondence_identity: str
    complete_response_record_count: int
    hero_worst_witness_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantics": AIOF_PREFLOP_CERTIFIED_GLOBAL_RESPONSE,
            "opponent_seat": self.opponent_seat,
            "hero_worst_value": self.hero_worst_value,
            "rows": [row.to_dict() for row in self.rows],
            "correspondence_identity": self.correspondence_identity,
            "complete_response_record_count": (
                self.complete_response_record_count
            ),
            "hero_worst_witness_count": self.hero_worst_witness_count,
        }


@dataclass(frozen=True)
class AiofPreflopCertifiedPointRecord:
    """Exact consumer record behind one M36 point evaluation."""

    policy_identity: str
    fixed_opponent_hero_ev: str
    post_response_hero_ev_worst: str
    baseline_total_repeated_hero_ev: str
    candidate_total_repeated_hero_ev: str
    uplift: str
    response: AiofPreflopCompleteResponse
    native_evaluation: ScalarOracleEvaluation

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_identity": self.policy_identity,
            "fixed_opponent_hero_ev": self.fixed_opponent_hero_ev,
            "post_response_hero_ev_worst": (
                self.post_response_hero_ev_worst
            ),
            "baseline_total_repeated_hero_ev": (
                self.baseline_total_repeated_hero_ev
            ),
            "candidate_total_repeated_hero_ev": (
                self.candidate_total_repeated_hero_ev
            ),
            "uplift": self.uplift,
            "complete_opponent_response": self.response.to_dict(),
            "native_evaluation": self.native_evaluation.to_dict(),
        }


@dataclass(frozen=True)
class AiofPreflopCertifiedOracleWorkCounters:
    """Nested consumer work completed before success or failure."""

    compatible_pairs: int = 0
    matchup_rows: int = 0
    exact_board_evaluations: int = 0
    coefficient_records: int = 0
    point_evaluations: int = 0
    response_rows: int = 0
    response_action_records: int = 0
    bound_records: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class AiofPreflopCertifiedGlobalPreparation:
    """Public prepared scenario, baseline policy, and conforming M36 oracle."""

    scenario: HeroBehaviorScenario
    baseline_policy: ExactBehaviorPolicy
    prepared_ranges: PreparedRanges
    prepared_ranges_projection: dict[str, Any]
    matchup_identity: str
    exact_support_identity: str
    response_oracle_identity: str
    objective_identity: str
    analysis_identity: str
    effective_aiof_limits: AiofLimits
    oracle: "AiofPreflopCertifiedScalarOracle"

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": {
                "scenario_identity": self.scenario.scenario_identity,
                "information_sets": [
                    {
                        "information_set_id": row.information_set_id,
                        "legal_action_ids": list(row.legal_action_ids),
                    }
                    for row in self.scenario.information_sets
                ],
            },
            "baseline_policy": self.baseline_policy.to_dict(),
            "prepared_ranges": self.prepared_ranges_projection,
            "matchup_identity": self.matchup_identity,
            "exact_support_identity": self.exact_support_identity,
            "response_oracle_identity": self.response_oracle_identity,
            "objective_identity": self.objective_identity,
            "analysis_identity": self.analysis_identity,
            "effective_aiof_limits": asdict(self.effective_aiof_limits),
        }


@dataclass(frozen=True)
class AiofPreflopCertifiedGlobalPayload:
    """Complete M37 success payload retaining the native M36 result."""

    preparation: AiofPreflopCertifiedGlobalPreparation
    baseline_point: AiofPreflopCertifiedPointRecord
    selected_point: AiofPreflopCertifiedPointRecord | None
    native_optimizer_result: CertifiedGlobalOptimizerResult
    oracle_work_counters: AiofPreflopCertifiedOracleWorkCounters
    request: AiofPreflopCertifiedGlobalRequest

    def to_dict(self) -> dict[str, Any]:
        native_payload = self.native_optimizer_result.payload
        return {
            "contract_version": (
                AIOF_PREFLOP_CERTIFIED_GLOBAL_CONTRACT_VERSION
            ),
            "algorithm_version": AIOF_PREFLOP_CERTIFIED_GLOBAL_ALGORITHM,
            "domain_contract": AIOF_PREFLOP_CERTIFIED_GLOBAL_DOMAIN,
            "objective_contract": AIOF_PREFLOP_CERTIFIED_GLOBAL_OBJECTIVE,
            "response_contract": AIOF_PREFLOP_CERTIFIED_GLOBAL_RESPONSE,
            "bound_contract": AIOF_PREFLOP_CERTIFIED_GLOBAL_BOUND,
            "claim_scope": AIOF_PREFLOP_CERTIFIED_GLOBAL_CLAIM_SCOPE,
            "accounting": CHIP_ACCOUNTING_ID,
            "hero_seat": self.request.hero_seat,
            "opponent_seat": (
                "bb" if self.request.hero_seat == "sb" else "sb"
            ),
            "preparation": self.preparation.to_dict(),
            "repeated_configuration": {
                "horizon": self.request.horizon,
                "adaptation_opportunity": (
                    self.request.adaptation_opportunity
                ),
                "discount": self.request.discount,
                "response_tolerance": self.request.response_tolerance,
                "absolute_gap_tolerance": (
                    self.request.absolute_gap_tolerance
                ),
                "relative_gap_tolerance": (
                    self.request.relative_gap_tolerance
                ),
            },
            "caps": {
                "aiof": asdict(self.request.aiof_limits),
                "effective_aiof": asdict(
                    self.preparation.effective_aiof_limits
                ),
                "integration": self.request.integration_limits.to_dict(),
                "optimizer": asdict(self.request.optimizer_limits),
            },
            "baseline_point": self.baseline_point.to_dict(),
            "selected_point": (
                None
                if self.selected_point is None
                else self.selected_point.to_dict()
            ),
            "no_beneficial_commitment": (
                native_payload.no_beneficial_commitment
                if native_payload is not None
                else None
            ),
            "native_optimizer_result": (
                self.native_optimizer_result.to_dict()
            ),
            "oracle_work_counters": self.oracle_work_counters.to_dict(),
        }


@dataclass(frozen=True)
class AiofPreflopCertifiedGlobalError:
    """Sanitized fail-closed error carrying no support or certificate."""

    phase: str
    message: str
    cause_status: str | None = None


@dataclass(frozen=True)
class AiofPreflopCertifiedGlobalResult:
    """Exclusive certified success or null-payload M37 failure."""

    status: str
    payload: AiofPreflopCertifiedGlobalPayload | None
    error: AiofPreflopCertifiedGlobalError | None
    optimizer_work_counters: CertifiedGlobalWorkCounters
    oracle_work_counters: AiofPreflopCertifiedOracleWorkCounters
    partial_result: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "payload": None if self.payload is None else self.payload.to_dict(),
            "error": None if self.error is None else asdict(self.error),
            "optimizer_work_counters": self.optimizer_work_counters.to_dict(),
            "oracle_work_counters": self.oracle_work_counters.to_dict(),
            "partial_result": self.partial_result,
        }


@dataclass(frozen=True)
class _Affine:
    constant: Fraction
    coefficients: tuple[tuple[str, Fraction], ...]

    @classmethod
    def make(
        cls, constant: Fraction, coefficients: dict[str, Fraction]
    ) -> "_Affine":
        return cls(
            Fraction(constant),
            tuple(
                (key, Fraction(value))
                for key, value in sorted(coefficients.items())
                if value
            ),
        )

    def value(self, probabilities: dict[str, Fraction]) -> Fraction:
        return self.constant + sum(
            (
                coefficient * probabilities[information_set_id]
                for information_set_id, coefficient in self.coefficients
            ),
            Fraction(0),
        )


@dataclass(frozen=True)
class _Matchup:
    sb_combo: str
    bb_combo: str
    probability: Fraction
    showdown_sb: Fraction
    wins: int
    losses: int
    ties: int
    trials: int

    def identity_projection(self) -> dict[str, Any]:
        return {
            "sb_combo": self.sb_combo,
            "bb_combo": self.bb_combo,
            "probability": _rational_text(self.probability),
            "showdown_sb": _rational_text(self.showdown_sb),
            "counts": {
                "wins": self.wins,
                "losses": self.losses,
                "ties": self.ties,
                "trials": self.trials,
            },
        }


@dataclass(frozen=True)
class _ResponseAffineRow:
    information_set_id: str
    combo: str
    marginal: Fraction
    reach: _Affine
    actions: tuple[tuple[str, _Affine], ...]


def _validate_integration_limits(
    limits: object,
) -> AiofPreflopCertifiedGlobalLimits:
    if type(limits) is not AiofPreflopCertifiedGlobalLimits:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            "integration_limits must be AiofPreflopCertifiedGlobalLimits",
        )
    ceilings = {
        "max_information_sets_per_seat": MAX_INFORMATION_SETS_PER_SEAT,
        "max_horizon": DEFAULT_MAX_HORIZON,
        "max_compatible_pairs": MAX_COMPATIBLE_PAIRS,
        "max_matchup_rows": MAX_MATCHUP_ROWS,
        "max_exact_board_evaluations": MAX_EXACT_BOARD_EVALUATIONS,
        "max_coefficient_records": MAX_COEFFICIENT_RECORDS,
        "max_total_response_rows": MAX_TOTAL_RESPONSE_ROWS,
        "max_oracle_bound_records": MAX_ORACLE_BOUND_RECORDS,
        "max_output_records": MAX_OUTPUT_RECORDS,
        "max_output_bytes": MAX_OUTPUT_BYTES,
    }
    for field in fields(limits):
        value = getattr(limits, field.name)
        _require_plain_int(value, f"integration_limits.{field.name}")
        if value > ceilings[field.name]:
            raise AiofContractError(
                AiofStatus.INVALID_INPUT,
                f"integration_limits.{field.name} exceeds immutable hard ceiling "
                f"{ceilings[field.name]}",
            )
    return limits


def _validate_pins(
    pins: object,
) -> AiofPreflopCertifiedGlobalPins:
    if type(pins) is not AiofPreflopCertifiedGlobalPins:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            "pins must be AiofPreflopCertifiedGlobalPins",
        )
    for field in fields(pins):
        value = getattr(pins, field.name)
        if value is not None:
            try:
                _require_identity(value, f"pins.{field.name}")
            except ValueError as exc:
                raise AiofContractError(
                    AiofStatus.INVALID_INPUT, str(exc)
                ) from exc
    return pins


def _check_pin(
    supplied: str | None, actual: str, name: str
) -> None:
    if supplied is not None and supplied != actual:
        raise _StaleInput(f"{name} mismatch")


class _StaleInput(ValueError):
    pass


def _effective_aiof_limits(
    request: AiofPreflopCertifiedGlobalRequest,
    limits: AiofPreflopCertifiedGlobalLimits,
) -> AiofLimits:
    source = request.aiof_limits
    if type(source) is not AiofLimits:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, "aiof_limits must be AiofLimits"
        )
    # This is the intersection of two explicitly reported caller limits, not
    # permissive clamping of an invalid value above a hard ceiling.
    return AiofLimits(
        max_range_entries_per_side=source.max_range_entries_per_side,
        max_exact_combos_per_side=min(
            source.max_exact_combos_per_side,
            limits.max_information_sets_per_seat,
        ),
        max_compatible_combo_pairs=min(
            source.max_compatible_combo_pairs,
            limits.max_compatible_pairs,
            limits.max_matchup_rows,
        ),
        max_dead_cards=source.max_dead_cards,
        max_exact_board_evaluations=min(
            source.max_exact_board_evaluations,
            limits.max_exact_board_evaluations,
        ),
        max_monte_carlo_samples=source.max_monte_carlo_samples,
        max_sampling_attempts=source.max_sampling_attempts,
        max_cache_entries=source.max_cache_entries,
        max_trace_points=source.max_trace_points,
    )


def _validate_game(game: object) -> tuple[Fraction, Fraction, Fraction]:
    if type(game) is not HeadsUpChipEvGame:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, "game must be HeadsUpChipEvGame"
        )
    sb_stack = _exact_binary64(game.starting_stack_sb, "starting_stack_sb")
    bb_stack = _exact_binary64(game.starting_stack_bb, "starting_stack_bb")
    small_blind = _exact_binary64(game.small_blind, "small_blind")
    big_blind = _exact_binary64(game.big_blind, "big_blind")
    ante = _exact_binary64(game.ante, "ante")
    fee = _exact_binary64(game.fee, "fee")
    dead_money = _exact_binary64(
        game.third_party_dead_money, "third_party_dead_money"
    )
    if sb_stack <= 0 or bb_stack <= 0 or small_blind <= 0:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            "stacks and small blind must be positive",
        )
    if big_blind < small_blind or ante < 0:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, "invalid blind or ante ordering"
        )
    if type(game.side_pot) is not bool:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, "side_pot must be bool"
        )
    if fee != 0 or dead_money != 0 or game.side_pot:
        raise AiofContractError(
            AiofStatus.UNSUPPORTED_MODEL,
            "fee, third-party money, and side pots are unsupported",
        )
    sb_post = small_blind + ante
    bb_post = big_blind + ante
    if sb_stack < sb_post or bb_stack < bb_post:
        raise AiofContractError(
            AiofStatus.UNSUPPORTED_MODEL,
            "a stack cannot fully cover its mandatory post",
        )
    return -sb_post, bb_post, min(sb_stack, bb_stack)


def _canonical_source_label(label: object) -> tuple[str, str]:
    if type(label) is not str:
        raise AiofContractError(
            AiofStatus.INVALID_RANGE, "range label must be a string"
        )
    if len(label) == 4:
        try:
            return "exact", canonicalize_exact_combo(label)
        except AiofContractError:
            pass
    return "class", canonicalize_hand_class(label)


def _class_multiplicity(label: str) -> int:
    if len(label) == 2:
        return 6
    if len(label) == 3 and label[2] == "s":
        return 4
    if len(label) == 3 and label[2] == "o":
        return 12
    raise AiofContractError(
        AiofStatus.INVALID_RANGE,
        f"unsupported canonical class label {label!r}",
    )


def _exact_source_weights(spec: RangeSpec, name: str) -> dict[str, Fraction]:
    if type(spec) is not RangeSpec or type(spec.entries) is not tuple:
        raise AiofContractError(
            AiofStatus.INVALID_RANGE, f"{name} must be RangeSpec"
        )
    result: dict[str, Fraction] = {}
    for index, entry in enumerate(spec.entries):
        if not hasattr(entry, "label") or not hasattr(entry, "weight"):
            raise AiofContractError(
                AiofStatus.INVALID_RANGE,
                f"{name}[{index}] is not a range entry",
            )
        kind, label = _canonical_source_label(entry.label)
        expected_basis = (
            WeightBasis.EXACT_COMBO_MASS
            if kind == "exact"
            else WeightBasis.CLASS_TOTAL_MASS
        )
        if entry.weight_basis is not expected_basis:
            raise AiofContractError(
                AiofStatus.INVALID_RANGE,
                f"{name}[{index}] weight basis does not match label",
            )
        weight = _exact_binary64(entry.weight, f"{name}[{index}].weight")
        if weight <= 0:
            raise AiofContractError(
                AiofStatus.INVALID_RANGE,
                f"{name}[{index}].weight must be positive",
            )
        if label in result:
            raise AiofContractError(
                AiofStatus.DUPLICATE_COMBO,
                f"{name} contains duplicate source label {label}",
            )
        result[label] = weight
    return result


def _exact_combo_masses(
    prepared: PreparedRanges,
    sb_spec: RangeSpec,
    bb_spec: RangeSpec,
) -> tuple[dict[str, Fraction], dict[str, Fraction], str]:
    sb_sources = _exact_source_weights(sb_spec, "sb_range")
    bb_sources = _exact_source_weights(bb_spec, "bb_range")

    def side(
        combos: tuple[Any, ...],
        sources: dict[str, Fraction],
        name: str,
    ) -> dict[str, Fraction]:
        output: dict[str, Fraction] = {}
        for combo in combos:
            source = combo.source_label
            if source not in sources:
                raise AiofContractError(
                    AiofStatus.ORACLE_MISMATCH,
                    f"{name} prepared source is absent from request",
                )
            divisor = (
                1 if len(source) == 4 else _class_multiplicity(source)
            )
            mass = sources[source] / divisor
            if mass <= 0:
                raise AiofContractError(
                    AiofStatus.NUMERIC_FAILURE,
                    f"{name} reconstructed mass is nonpositive",
                )
            output[combo.combo] = mass
        return output

    sb = side(prepared.sb_range.combos, sb_sources, "sb_range")
    bb = side(prepared.bb_range.combos, bb_sources, "bb_range")
    support_projection = {
        "lift": "lossless-binary64-as-integer-ratio-v1",
        "class_expansion": (
            "uniform-within-full-class-before-dead-card-conditioning-v1"
        ),
        "sb": [
            {"combo": combo, "raw_mass": _rational_text(mass)}
            for combo, mass in sorted(sb.items())
        ],
        "bb": [
            {"combo": combo, "raw_mass": _rational_text(mass)}
            for combo, mass in sorted(bb.items())
        ],
    }
    return sb, bb, _identity(support_projection)


def _profile_probability_map(
    entries: tuple[ComboActionProbability, ...],
    expected: tuple[str, ...],
    name: str,
) -> dict[str, Fraction]:
    if type(entries) is not tuple:
        raise AiofContractError(
            AiofStatus.INVALID_STRATEGY, f"{name} must be a tuple"
        )
    result: dict[str, Fraction] = {}
    for entry in entries:
        if type(entry) is not ComboActionProbability:
            raise AiofContractError(
                AiofStatus.INVALID_STRATEGY,
                f"{name} contains an invalid entry",
            )
        combo = canonicalize_exact_combo(entry.combo)
        probability = _exact_binary64(
            entry.probability, f"{name}[{combo}]"
        )
        if not 0 <= probability <= 1:
            raise AiofContractError(
                AiofStatus.INVALID_STRATEGY,
                f"{name}[{combo}] must be in [0,1]",
            )
        if combo in result:
            raise AiofContractError(
                AiofStatus.INVALID_STRATEGY,
                f"{name} contains duplicate combo {combo}",
            )
        result[combo] = probability
    if set(result) != set(expected):
        raise AiofContractError(
            AiofStatus.INVALID_STRATEGY,
            f"{name} must contain every and only surviving exact combo",
        )
    return result


def _canonical_baseline(
    profile: object, prepared: PreparedRanges
) -> tuple[dict[str, Fraction], dict[str, Fraction]]:
    if type(profile) is not SuppliedProfile:
        raise AiofContractError(
            AiofStatus.INVALID_STRATEGY,
            "baseline_profile must be SuppliedProfile",
        )
    sb_expected = tuple(item.combo for item in prepared.sb_marginals)
    bb_expected = tuple(item.combo for item in prepared.bb_marginals)
    sb = _profile_probability_map(
        profile.sb_shove, sb_expected, "baseline_profile.sb_shove"
    )
    bb = _profile_probability_map(
        profile.bb_call, bb_expected, "baseline_profile.bb_call"
    )
    return sb, bb


def _prepared_projection(
    prepared: PreparedRanges,
    sb_spec: RangeSpec,
    bb_spec: RangeSpec,
    sb_masses: dict[str, Fraction],
    bb_masses: dict[str, Fraction],
) -> dict[str, Any]:
    def source_projection(
        spec: RangeSpec,
    ) -> list[dict[str, str]]:
        rows = []
        for entry in spec.entries:
            kind, label = _canonical_source_label(entry.label)
            rows.append(
                {
                    "kind": kind,
                    "label": label,
                    "weight": _rational_text(
                        _exact_binary64(
                            entry.weight, f"range[{label}].weight"
                        )
                    ),
                    "weight_basis": entry.weight_basis.value,
                }
            )
        return sorted(rows, key=lambda row: (row["kind"], row["label"]))

    return {
        "public_prepared_ranges_identity": prepared.content_identity,
        "dead_cards": list(prepared.dead_cards),
        "dead_card_ids": list(prepared.dead_card_ids),
        "compatible_pair_count": prepared.compatible_pair_count,
        "pre_blocker_ranges": {
            "sb": source_projection(sb_spec),
            "bb": source_projection(bb_spec),
        },
        "expansion": {
            "sb": {
                "content_identity": prepared.sb_range.content_identity,
                "projected_combo_count": (
                    prepared.sb_range.projected_combo_count
                ),
                "removed_combo_count": (
                    prepared.sb_range.removed_combo_count
                ),
            },
            "bb": {
                "content_identity": prepared.bb_range.content_identity,
                "projected_combo_count": (
                    prepared.bb_range.projected_combo_count
                ),
                "removed_combo_count": (
                    prepared.bb_range.removed_combo_count
                ),
            },
        },
        "sb": [
            {
                "combo": combo.combo,
                "card_ids": list(combo.card_ids),
                "source_label": combo.source_label,
                "raw_mass": _rational_text(sb_masses[combo.combo]),
            }
            for combo in prepared.sb_range.combos
        ],
        "bb": [
            {
                "combo": combo.combo,
                "card_ids": list(combo.card_ids),
                "source_label": combo.source_label,
                "raw_mass": _rational_text(bb_masses[combo.combo]),
            }
            for combo in prepared.bb_range.combos
        ],
        "conditioning": (
            "exact-product-compatible-condition-with-exact-rational-masses-v1"
        ),
    }


def _build_matchups(
    prepared: PreparedRanges,
    sb_masses: dict[str, Fraction],
    bb_masses: dict[str, Fraction],
    effective_stack: Fraction,
    limits: AiofPreflopCertifiedGlobalLimits,
) -> tuple[tuple[_Matchup, ...], str, int]:
    pair_count = prepared.compatible_pair_count
    if pair_count > limits.max_matchup_rows:
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            "max_matchup_rows exceeded before matchup materialization",
        )
    remaining = 52 - len(prepared.dead_card_ids) - 4
    if remaining < 5:
        raise AiofContractError(
            AiofStatus.INVALID_CARD_INPUT,
            "fewer than five board cards remain",
        )
    evaluations = pair_count * math.comb(remaining, 5)
    if evaluations > limits.max_exact_board_evaluations:
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            "max_exact_board_evaluations exceeded before exact enumeration",
        )
    joint_mass = sum(
        (
            sb_masses[sb.combo] * bb_masses[bb.combo]
            for sb in prepared.sb_range.combos
            for bb in prepared.bb_range.combos
            if not set(sb.card_ids) & set(bb.card_ids)
        ),
        Fraction(0),
    )
    if joint_mass <= 0:
        raise AiofContractError(
            AiofStatus.EMPTY_COMPATIBLE_SUPPORT,
            "compatible joint support has no positive exact mass",
        )
    rows: list[_Matchup] = []
    for sb_combo, bb_combo, _float_probability, counts in (
        _iter_exact_matchup_outcomes(prepared)
    ):
        probability = (
            sb_masses[sb_combo.combo]
            * bb_masses[bb_combo.combo]
            / joint_mass
        )
        showdown = (
            effective_stack
            * Fraction(counts.wins - counts.losses, counts.trials)
        )
        rows.append(
            _Matchup(
                sb_combo=sb_combo.combo,
                bb_combo=bb_combo.combo,
                probability=probability,
                showdown_sb=showdown,
                wins=counts.wins,
                losses=counts.losses,
                ties=counts.ties,
                trials=counts.trials,
            )
        )
    if len(rows) != pair_count:
        raise AiofContractError(
            AiofStatus.ORACLE_MISMATCH,
            "exact matchup enumeration count differs from prepared support",
        )
    projection = [row.identity_projection() for row in rows]
    return tuple(rows), _identity(projection), evaluations


def _add_coefficient(
    coefficients: dict[str, Fraction],
    key: str,
    value: Fraction,
) -> None:
    coefficients[key] = coefficients.get(key, Fraction(0)) + value


def _build_affines(
    *,
    hero_seat: str,
    matchups: tuple[_Matchup, ...],
    sb_baseline: dict[str, Fraction],
    bb_baseline: dict[str, Fraction],
    sb_fold: Fraction,
    sb_shove_bb_fold: Fraction,
) -> tuple[_Affine, _Affine, tuple[_ResponseAffineRow, ...]]:
    if hero_seat == "sb":
        pre_constant = sum(
            (row.probability * sb_fold for row in matchups), Fraction(0)
        )
        pre_coefficients: dict[str, Fraction] = {}
        common_coefficients: dict[str, Fraction] = {}
        by_bb: dict[str, dict[str, dict[str, Fraction]]] = {}
        bb_marginals: dict[str, Fraction] = {}
        for row in matchups:
            information_set = f"hero:sb:{row.sb_combo}"
            opponent_probability = bb_baseline[row.bb_combo]
            shove_value = (
                (1 - opponent_probability) * sb_shove_bb_fold
                + opponent_probability * row.showdown_sb
            )
            _add_coefficient(
                pre_coefficients,
                information_set,
                row.probability * (shove_value - sb_fold),
            )
            _add_coefficient(
                common_coefficients,
                information_set,
                -row.probability * sb_fold,
            )
            bb_marginals[row.bb_combo] = (
                bb_marginals.get(row.bb_combo, Fraction(0))
                + row.probability
            )
            actions = by_bb.setdefault(
                row.bb_combo, {"call": {}, "fold": {}}
            )
            _add_coefficient(
                actions["call"],
                information_set,
                row.probability * row.showdown_sb,
            )
            _add_coefficient(
                actions["fold"],
                information_set,
                row.probability * sb_shove_bb_fold,
            )
        response_rows = tuple(
            _ResponseAffineRow(
                information_set_id=f"opponent:bb:{combo}",
                combo=combo,
                marginal=bb_marginals[combo],
                reach=_Affine.make(Fraction(0), {}),
                actions=tuple(
                    (
                        action,
                        _Affine.make(Fraction(0), coefficients),
                    )
                    for action, coefficients in sorted(actions.items())
                ),
            )
            for combo, actions in sorted(by_bb.items())
        )
        return (
            _Affine.make(pre_constant, pre_coefficients),
            _Affine.make(sb_fold, common_coefficients),
            response_rows,
        )

    pre_constant = Fraction(0)
    pre_coefficients = {}
    by_sb: dict[str, list[_Matchup]] = {}
    for row in matchups:
        shove_probability = sb_baseline[row.sb_combo]
        pre_constant += -row.probability * (
            (1 - shove_probability) * sb_fold
            + shove_probability * sb_shove_bb_fold
        )
        information_set = f"hero:bb:{row.bb_combo}"
        _add_coefficient(
            pre_coefficients,
            information_set,
            -row.probability
            * shove_probability
            * (row.showdown_sb - sb_shove_bb_fold),
        )
        by_sb.setdefault(row.sb_combo, []).append(row)
    response_rows_list: list[_ResponseAffineRow] = []
    for combo, rows in sorted(by_sb.items()):
        marginal = sum((row.probability for row in rows), Fraction(0))
        fold_contribution = -marginal * sb_fold
        shove_constant = -marginal * sb_shove_bb_fold
        shove_coefficients: dict[str, Fraction] = {}
        for row in rows:
            _add_coefficient(
                shove_coefficients,
                f"hero:bb:{row.bb_combo}",
                row.probability
                * (sb_shove_bb_fold - row.showdown_sb),
            )
        response_rows_list.append(
            _ResponseAffineRow(
                information_set_id=f"opponent:sb:{combo}",
                combo=combo,
                marginal=marginal,
                reach=_Affine.make(marginal, {}),
                actions=(
                    ("fold", _Affine.make(fold_contribution, {})),
                    (
                        "shove",
                        _Affine.make(shove_constant, shove_coefficients),
                    ),
                ),
            )
        )
    return (
        _Affine.make(pre_constant, pre_coefficients),
        _Affine.make(Fraction(0), {}),
        tuple(response_rows_list),
    )


def _discount_weights(
    horizon: int,
    adaptation_opportunity: int,
    discount: Fraction,
) -> tuple[Fraction, Fraction, Fraction]:
    terms: list[Fraction] = []
    term = Fraction(1)
    for _ in range(horizon):
        terms.append(term)
        term *= discount
    prefix_count = adaptation_opportunity - 1
    pre = sum(terms[:prefix_count], Fraction(0))
    post = sum(terms[prefix_count:], Fraction(0))
    return pre, post, pre + post


@dataclass(frozen=True)
class _JsonObjectProjection:
    """Lazy JSON object fields used only by output-cap preflight."""

    items: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class _DeferredJsonProjection:
    """Delay one component projection until deterministic traversal reaches it."""

    factory: Callable[[], Any]


class _OutputLimitReached(RuntimeError):
    """The final public success result exceeded an integration output cap."""


def _payload_output_projection(
    payload: AiofPreflopCertifiedGlobalPayload,
) -> _JsonObjectProjection:
    native_payload = payload.native_optimizer_result.payload
    return _JsonObjectProjection(
        (
            (
                "contract_version",
                AIOF_PREFLOP_CERTIFIED_GLOBAL_CONTRACT_VERSION,
            ),
            ("algorithm_version", AIOF_PREFLOP_CERTIFIED_GLOBAL_ALGORITHM),
            ("domain_contract", AIOF_PREFLOP_CERTIFIED_GLOBAL_DOMAIN),
            ("objective_contract", AIOF_PREFLOP_CERTIFIED_GLOBAL_OBJECTIVE),
            ("response_contract", AIOF_PREFLOP_CERTIFIED_GLOBAL_RESPONSE),
            ("bound_contract", AIOF_PREFLOP_CERTIFIED_GLOBAL_BOUND),
            ("claim_scope", AIOF_PREFLOP_CERTIFIED_GLOBAL_CLAIM_SCOPE),
            ("accounting", CHIP_ACCOUNTING_ID),
            ("hero_seat", payload.request.hero_seat),
            (
                "opponent_seat",
                "bb" if payload.request.hero_seat == "sb" else "sb",
            ),
            (
                "preparation",
                _DeferredJsonProjection(payload.preparation.to_dict),
            ),
            (
                "repeated_configuration",
                _JsonObjectProjection(
                    (
                        ("horizon", payload.request.horizon),
                        (
                            "adaptation_opportunity",
                            payload.request.adaptation_opportunity,
                        ),
                        ("discount", payload.request.discount),
                        (
                            "response_tolerance",
                            payload.request.response_tolerance,
                        ),
                        (
                            "absolute_gap_tolerance",
                            payload.request.absolute_gap_tolerance,
                        ),
                        (
                            "relative_gap_tolerance",
                            payload.request.relative_gap_tolerance,
                        ),
                    )
                ),
            ),
            (
                "caps",
                _JsonObjectProjection(
                    (
                        (
                            "aiof",
                            _DeferredJsonProjection(
                                lambda: asdict(payload.request.aiof_limits)
                            ),
                        ),
                        (
                            "effective_aiof",
                            _DeferredJsonProjection(
                                lambda: asdict(
                                    payload.preparation.effective_aiof_limits
                                )
                            ),
                        ),
                        (
                            "integration",
                            _DeferredJsonProjection(
                                payload.request.integration_limits.to_dict
                            ),
                        ),
                        (
                            "optimizer",
                            _DeferredJsonProjection(
                                lambda: asdict(
                                    payload.request.optimizer_limits
                                )
                            ),
                        ),
                    )
                ),
            ),
            (
                "baseline_point",
                _DeferredJsonProjection(payload.baseline_point.to_dict),
            ),
            (
                "selected_point",
                (
                    None
                    if payload.selected_point is None
                    else _DeferredJsonProjection(
                        payload.selected_point.to_dict
                    )
                ),
            ),
            (
                "no_beneficial_commitment",
                (
                    native_payload.no_beneficial_commitment
                    if native_payload is not None
                    else None
                ),
            ),
            (
                "native_optimizer_result",
                _DeferredJsonProjection(
                    payload.native_optimizer_result.to_dict
                ),
            ),
            (
                "oracle_work_counters",
                _DeferredJsonProjection(
                    payload.oracle_work_counters.to_dict
                ),
            ),
        )
    )


def _success_output_projection(
    result: AiofPreflopCertifiedGlobalResult,
) -> _JsonObjectProjection:
    if result.payload is None or result.error is not None:
        raise RuntimeError("success output projection requires a success result")
    return _JsonObjectProjection(
        (
            ("status", result.status),
            ("payload", _payload_output_projection(result.payload)),
            ("error", None),
            (
                "optimizer_work_counters",
                _DeferredJsonProjection(
                    result.optimizer_work_counters.to_dict
                ),
            ),
            (
                "oracle_work_counters",
                _DeferredJsonProjection(
                    result.oracle_work_counters.to_dict
                ),
            ),
            ("partial_result", result.partial_result),
        )
    )


def _preflight_success_output(
    result: AiofPreflopCertifiedGlobalResult,
    *,
    max_records: int,
    max_bytes: int,
) -> tuple[int, int]:
    """Measure the final public JSON shape without its aggregate projection.

    Records use the same node-count convention as the prior cap.  Bytes are
    those produced by sorted-key, compact, strict UTF-8 JSON.  Traversal stops
    as soon as either budget is exceeded, before constructing a complete
    result/payload dictionary or a complete encoded byte string.
    """

    record_count = 0
    byte_count = 0

    def add_bytes(fragment: str) -> None:
        nonlocal byte_count
        byte_count += len(fragment.encode("utf-8"))
        if byte_count > max_bytes:
            raise _OutputLimitReached(
                "max_output_bytes exceeded before success output "
                "materialization"
            )

    def visit(value: Any) -> None:
        nonlocal record_count
        if isinstance(value, _DeferredJsonProjection):
            visit(value.factory())
            return

        record_count += 1
        if record_count > max_records:
            raise _OutputLimitReached(
                "max_output_records exceeded before success output "
                "materialization"
            )

        if isinstance(value, _JsonObjectProjection):
            items = sorted(value.items, key=lambda item: item[0])
            add_bytes("{")
            for index, (key, item) in enumerate(items):
                if index:
                    add_bytes(",")
                add_bytes(
                    json.dumps(
                        key,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                )
                add_bytes(":")
                visit(item)
            add_bytes("}")
            return

        if isinstance(value, dict):
            add_bytes("{")
            for index, key in enumerate(sorted(value)):
                if type(key) is not str:
                    raise TypeError("canonical output keys must be strings")
                if index:
                    add_bytes(",")
                add_bytes(
                    json.dumps(
                        key,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                )
                add_bytes(":")
                visit(value[key])
            add_bytes("}")
            return

        if isinstance(value, (list, tuple)):
            add_bytes("[")
            for index, item in enumerate(value):
                if index:
                    add_bytes(",")
                visit(item)
            add_bytes("]")
            return

        add_bytes(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        )

    visit(_success_output_projection(result))
    return record_count, byte_count


class AiofPreflopCertifiedScalarOracle:
    """Exact M37 point/whole-cell oracle conforming to the M36 protocol."""

    response_semantics = RESPONSE_SEMANTICS
    bound_contract_version = BOUND_CONTRACT_VERSION

    def __init__(
        self,
        *,
        scenario: HeroBehaviorScenario,
        baseline_policy: ExactBehaviorPolicy,
        hero_seat: str,
        hero_active_action: str,
        pre_affine: _Affine,
        response_common: _Affine,
        response_rows: tuple[_ResponseAffineRow, ...],
        baseline_hero_ev: Fraction,
        pre_weight: Fraction,
        post_weight: Fraction,
        total_weight: Fraction,
        response_oracle_identity: str,
        objective_identity: str,
        limits: AiofPreflopCertifiedGlobalLimits,
        initial_counters: AiofPreflopCertifiedOracleWorkCounters,
    ) -> None:
        self.scenario = scenario
        self.baseline_policy = baseline_policy
        self.hero_seat = hero_seat
        self.hero_active_action = hero_active_action
        self.pre_affine = pre_affine
        self.response_common = response_common
        self.response_rows = response_rows
        self.baseline_hero_ev = baseline_hero_ev
        self.pre_weight = pre_weight
        self.post_weight = post_weight
        self.total_weight = total_weight
        self.response_oracle_identity = response_oracle_identity
        self.objective_identity = objective_identity
        self.limits = limits
        self._baseline_policy_identity = behavior_policy_identity(
            scenario, baseline_policy
        )
        self._records: dict[str, AiofPreflopCertifiedPointRecord] = {}
        self._compatible_pairs = initial_counters.compatible_pairs
        self._matchup_rows = initial_counters.matchup_rows
        self._exact_board_evaluations = (
            initial_counters.exact_board_evaluations
        )
        self._coefficient_records = initial_counters.coefficient_records
        self._point_evaluations = 0
        self._response_rows_created = 0
        self._response_action_records = 0
        self._bound_records = 0

    @property
    def work_counters(self) -> AiofPreflopCertifiedOracleWorkCounters:
        return AiofPreflopCertifiedOracleWorkCounters(
            compatible_pairs=self._compatible_pairs,
            matchup_rows=self._matchup_rows,
            exact_board_evaluations=self._exact_board_evaluations,
            coefficient_records=self._coefficient_records,
            point_evaluations=self._point_evaluations,
            response_rows=self._response_rows_created,
            response_action_records=self._response_action_records,
            bound_records=self._bound_records,
        )

    def point_record(
        self, policy_identity: str
    ) -> AiofPreflopCertifiedPointRecord:
        return self._records[policy_identity]

    def _probabilities(
        self, policy: ExactBehaviorPolicy
    ) -> dict[str, Fraction]:
        if type(policy) is not ExactBehaviorPolicy or type(policy.rows) is not tuple:
            raise UnsupportedScalarOracleDomain(
                "policy is not a complete exact behavior policy"
            )
        expected = {
            row.information_set_id: tuple(row.legal_action_ids)
            for row in self.scenario.information_sets
        }
        result: dict[str, Fraction] = {}
        seen: set[str] = set()
        for row in policy.rows:
            if row.information_set_id not in expected:
                raise UnsupportedScalarOracleDomain(
                    "policy contains an unknown information set"
                )
            if row.information_set_id in seen:
                raise UnsupportedScalarOracleDomain(
                    "policy contains a duplicate information set"
                )
            seen.add(row.information_set_id)
            actions: dict[str, Fraction] = {}
            for action in row.actions:
                try:
                    probability = Fraction(action.probability)
                except (ValueError, ZeroDivisionError) as exc:
                    raise UnsupportedScalarOracleDomain(
                        "policy probability is not exact rational text"
                    ) from exc
                actions[action.action_id] = probability
            if (
                set(actions) != set(expected[row.information_set_id])
                or sum(actions.values(), Fraction(0)) != 1
                or any(value < 0 or value > 1 for value in actions.values())
            ):
                raise UnsupportedScalarOracleDomain(
                    "policy does not satisfy the complete legal simplex"
                )
            result[row.information_set_id] = actions[
                self.hero_active_action
            ]
        if seen != set(expected):
            raise UnsupportedScalarOracleDomain(
                "policy omits a Hero information set"
            )
        return result

    def _response(
        self, probabilities: dict[str, Fraction]
    ) -> AiofPreflopCompleteResponse:
        projected_rows = len(self.response_rows)
        action_records = sum(
            len(row.actions) for row in self.response_rows
        )
        if (
            self._response_rows_created + projected_rows
            > self.limits.max_total_response_rows
        ):
            raise ScalarOracleResourceLimit(
                "max_total_response_rows reached before response materialization"
            )
        if (
            self._response_rows_created
            + self._response_action_records
            + projected_rows
            + action_records
            > self.limits.max_coefficient_records
        ):
            raise ScalarOracleResourceLimit(
                "response action-record cap reached before materialization"
            )
        common = self.response_common.value(probabilities)
        hero_worst = common
        rows: list[AiofPreflopCompleteResponseRow] = []
        correspondence_count = 1
        for row in self.response_rows:
            values = tuple(
                (action, affine.value(probabilities))
                for action, affine in row.actions
            )
            minimum = min(value for _, value in values)
            best_actions = tuple(
                action for action, value in values if value == minimum
            )
            correspondence_count *= len(best_actions)
            hero_worst += minimum
            if self.hero_seat == "sb":
                reach = row.reach.value(probabilities)
            else:
                reach = row.marginal
            rows.append(
                AiofPreflopCompleteResponseRow(
                    information_set_id=row.information_set_id,
                    combo=row.combo,
                    compatible_probability=_rational_text(row.marginal),
                    information_reach_probability=_rational_text(reach),
                    action_hero_contributions=tuple(
                        (action, _rational_text(value))
                        for action, value in values
                    ),
                    best_actions=best_actions,
                )
            )
        response_projection = {
            "semantics": AIOF_PREFLOP_CERTIFIED_GLOBAL_RESPONSE,
            "opponent_seat": "bb" if self.hero_seat == "sb" else "sb",
            "common_hero_contribution": _rational_text(common),
            "hero_worst_value": _rational_text(hero_worst),
            "rows": [row.to_dict() for row in rows],
            "factorized_correspondence_cardinality": correspondence_count,
        }
        correspondence_identity = _identity(response_projection)
        self._response_rows_created += projected_rows
        self._response_action_records += action_records
        return AiofPreflopCompleteResponse(
            opponent_seat=response_projection["opponent_seat"],
            hero_worst_value=_rational_text(hero_worst),
            rows=tuple(rows),
            correspondence_identity=correspondence_identity,
            complete_response_record_count=correspondence_count,
            hero_worst_witness_count=correspondence_count,
        )

    def evaluate(
        self, policy: ExactBehaviorPolicy
    ) -> ScalarOracleEvaluation:
        policy_identity = behavior_policy_identity(self.scenario, policy)
        cached = self._records.get(policy_identity)
        if cached is not None:
            return cached.native_evaluation
        probabilities = self._probabilities(policy)
        fixed = self.pre_affine.value(probabilities)
        response = self._response(probabilities)
        post = Fraction(response.hero_worst_value)
        baseline_total = self.total_weight * self.baseline_hero_ev
        if policy_identity == self._baseline_policy_identity:
            candidate_total = baseline_total
        else:
            candidate_total = (
                self.pre_weight * fixed + self.post_weight * post
            )
        uplift = candidate_total - baseline_total
        native = ScalarOracleEvaluation(
            policy_identity=policy_identity,
            response_oracle_identity=self.response_oracle_identity,
            objective_identity=self.objective_identity,
            baseline_total_repeated_hero_ev=_rational_text(baseline_total),
            candidate_total_repeated_hero_ev=_rational_text(candidate_total),
            uplift=_rational_text(uplift),
            complete_response_correspondence_identity=(
                response.correspondence_identity
            ),
            complete_response_record_count=(
                response.complete_response_record_count
            ),
            hero_worst_witness_count=response.hero_worst_witness_count,
        )
        record = AiofPreflopCertifiedPointRecord(
            policy_identity=policy_identity,
            fixed_opponent_hero_ev=_rational_text(fixed),
            post_response_hero_ev_worst=_rational_text(post),
            baseline_total_repeated_hero_ev=_rational_text(baseline_total),
            candidate_total_repeated_hero_ev=_rational_text(candidate_total),
            uplift=_rational_text(uplift),
            response=response,
            native_evaluation=native,
        )
        self._records[policy_identity] = record
        self._point_evaluations += 1
        return native

    def _cell_intervals(
        self, cell: ExactBehaviorCell
    ) -> dict[str, tuple[Fraction, Fraction]]:
        if type(cell) is not ExactBehaviorCell:
            raise UnsupportedScalarOracleDomain(
                "bound request must be ExactBehaviorCell"
            )
        expected = {
            row.information_set_id: tuple(row.legal_action_ids)
            for row in self.scenario.information_sets
        }
        intervals: dict[str, tuple[Fraction, Fraction]] = {}
        seen: set[str] = set()
        for row in cell.rows:
            if row.information_set_id not in expected:
                raise UnsupportedScalarOracleDomain(
                    "cell contains an unknown information set"
                )
            if row.information_set_id in seen:
                raise UnsupportedScalarOracleDomain(
                    "cell contains a duplicate information set"
                )
            seen.add(row.information_set_id)
            action_intervals: dict[str, tuple[Fraction, Fraction]] = {}
            for action in row.actions:
                try:
                    lower = Fraction(action.lower)
                    upper = Fraction(action.upper)
                except (ValueError, ZeroDivisionError) as exc:
                    raise UnsupportedScalarOracleDomain(
                        "cell interval is not exact rational text"
                    ) from exc
                action_intervals[action.action_id] = (lower, upper)
            if set(action_intervals) != set(expected[row.information_set_id]):
                raise UnsupportedScalarOracleDomain(
                    "cell does not contain both legal actions"
                )
            active_lower, active_upper = action_intervals[
                self.hero_active_action
            ]
            passive_action = next(
                action
                for action in expected[row.information_set_id]
                if action != self.hero_active_action
            )
            passive_lower, passive_upper = action_intervals[passive_action]
            effective_lower = max(active_lower, 1 - passive_upper)
            effective_upper = min(active_upper, 1 - passive_lower)
            if (
                effective_lower < 0
                or effective_upper > 1
                or effective_lower > effective_upper
            ):
                raise UnsupportedScalarOracleDomain(
                    "cell does not intersect the exact two-action simplex"
                )
            intervals[row.information_set_id] = (
                effective_lower,
                effective_upper,
            )
        if seen != set(expected):
            raise UnsupportedScalarOracleDomain(
                "cell omits a Hero information set"
            )
        return intervals

    @staticmethod
    def _affine_max(
        affine: _Affine,
        intervals: dict[str, tuple[Fraction, Fraction]],
    ) -> Fraction:
        value = affine.constant
        for information_set_id, coefficient in affine.coefficients:
            lower, upper = intervals[information_set_id]
            value += coefficient * (upper if coefficient >= 0 else lower)
        return value

    def upper_bound(
        self, cell: ExactBehaviorCell
    ) -> ScalarOracleBound:
        if self._bound_records + 1 > self.limits.max_oracle_bound_records:
            raise ScalarOracleResourceLimit(
                "max_oracle_bound_records reached before bound construction"
            )
        intervals = self._cell_intervals(cell)
        pre_upper = self._affine_max(self.pre_affine, intervals)
        post_upper = self._affine_max(
            self.response_common, intervals
        )
        surrogate_coefficients: dict[str, Fraction] = {}
        for information_set_id, coefficient in self.pre_affine.coefficients:
            _add_coefficient(
                surrogate_coefficients,
                information_set_id,
                self.pre_weight * coefficient,
            )
        for information_set_id, coefficient in (
            self.response_common.coefficients
        ):
            _add_coefficient(
                surrogate_coefficients,
                information_set_id,
                self.post_weight * coefficient,
            )
        for row in self.response_rows:
            action_maxima = tuple(
                (
                    action,
                    affine,
                    self._affine_max(affine, intervals),
                )
                for action, affine in row.actions
            )
            _, selected_affine, selected_maximum = min(
                action_maxima, key=lambda item: (item[2], item[0])
            )
            post_upper += selected_maximum
            for information_set_id, coefficient in (
                selected_affine.coefficients
            ):
                _add_coefficient(
                    surrogate_coefficients,
                    information_set_id,
                    self.post_weight * coefficient,
                )
        baseline_total = self.total_weight * self.baseline_hero_ev
        locked_upper = (
            self.pre_weight * pre_upper
            + self.post_weight * post_upper
            - baseline_total
        )
        upper = max(Fraction(0), locked_upper)
        candidate_rows: list[ExactBehaviorRow] = []
        for cell_row in cell.rows:
            lower, upper_probability = intervals[
                cell_row.information_set_id
            ]
            coefficient = surrogate_coefficients.get(
                cell_row.information_set_id, Fraction(0)
            )
            active_probability = (
                upper_probability if coefficient >= 0 else lower
            )
            candidate_rows.append(
                ExactBehaviorRow(
                    information_set_id=cell_row.information_set_id,
                    actions=tuple(
                        ExactActionProbability(
                            action.action_id,
                            _rational_text(
                                active_probability
                                if action.action_id
                                == self.hero_active_action
                                else 1 - active_probability
                            ),
                        )
                        for action in cell_row.actions
                    ),
                )
            )
        candidate = ExactBehaviorPolicy(rows=tuple(candidate_rows))
        candidate_identity = behavior_policy_identity(
            self.scenario, candidate
        )
        cell_identity = behavior_cell_identity(self.scenario, cell)
        upper_text = _rational_text(upper)
        bound_identity = scalar_oracle_bound_identity(
            cell_identity=cell_identity,
            response_oracle_identity=self.response_oracle_identity,
            objective_identity=self.objective_identity,
            upper_bound=upper_text,
            candidate_policy_identity=candidate_identity,
        )
        self._bound_records += 1
        return ScalarOracleBound(
            cell_identity=cell_identity,
            response_oracle_identity=self.response_oracle_identity,
            objective_identity=self.objective_identity,
            upper_bound=upper_text,
            bound_identity=bound_identity,
            candidate_policy=candidate,
            valid_for_entire_cell=True,
            bound_contract_version=BOUND_CONTRACT_VERSION,
        )


def _scenario_and_policy(
    *,
    hero_seat: str,
    prepared: PreparedRanges,
    sb_baseline: dict[str, Fraction],
    bb_baseline: dict[str, Fraction],
    scenario_identity: str,
) -> tuple[HeroBehaviorScenario, ExactBehaviorPolicy, str]:
    if hero_seat == "sb":
        combos = tuple(item.combo for item in prepared.sb_marginals)
        active_action = "shove"
        passive_action = "fold"
        baseline = sb_baseline
    else:
        combos = tuple(item.combo for item in prepared.bb_marginals)
        active_action = "call"
        passive_action = "fold"
        baseline = bb_baseline
    rows = tuple(
        HeroInformationSet(
            information_set_id=f"hero:{hero_seat}:{combo}",
            legal_action_ids=(active_action, passive_action),
        )
        for combo in sorted(combos)
    )
    scenario = HeroBehaviorScenario(
        scenario_identity=scenario_identity,
        information_sets=rows,
    )
    policy = ExactBehaviorPolicy(
        rows=tuple(
            ExactBehaviorRow(
                information_set_id=row.information_set_id,
                actions=(
                    ExactActionProbability(
                        active_action,
                        _rational_text(
                            baseline[row.information_set_id.rsplit(":", 1)[1]]
                        ),
                    ),
                    ExactActionProbability(
                        passive_action,
                        _rational_text(
                            1
                            - baseline[
                                row.information_set_id.rsplit(":", 1)[1]
                            ]
                        ),
                    ),
                ),
            )
            for row in rows
        )
    )
    return scenario, policy, active_action


def _request_projection(
    request: AiofPreflopCertifiedGlobalRequest,
) -> dict[str, Any]:
    game = {
        field.name: (
            getattr(request.game, field.name)
            if type(getattr(request.game, field.name)) is bool
            else _rational_text(
                _exact_binary64(
                    getattr(request.game, field.name),
                    f"game.{field.name}",
                )
            )
        )
        for field in fields(request.game)
    }
    return {
        "contract_version": AIOF_PREFLOP_CERTIFIED_GLOBAL_CONTRACT_VERSION,
        "algorithm": request.algorithm.value,
        "game": game,
        "hero_seat": request.hero_seat,
        "horizon": request.horizon,
        "adaptation_opportunity": request.adaptation_opportunity,
        "discount": request.discount,
        "response_tolerance": request.response_tolerance,
        "absolute_gap_tolerance": request.absolute_gap_tolerance,
        "relative_gap_tolerance": request.relative_gap_tolerance,
        "aiof_limits": asdict(request.aiof_limits),
        "integration_limits": request.integration_limits.to_dict(),
        "optimizer_limits": asdict(request.optimizer_limits),
        "optimizer_pins": asdict(request.optimizer_pins),
    }


def prepare_aiof_preflop_certified_global_oracle(
    request: AiofPreflopCertifiedGlobalRequest,
) -> AiofPreflopCertifiedGlobalPreparation:
    """Prepare the full M37 domain and exact M36 consumer oracle.

    This public preparation raises on invalid/unsupported input.  Use
    :func:`analyze_aiof_preflop_certified_global` for the fail-closed result
    wrapper.  All support, matchup, coefficient, and identity caps are checked
    before their corresponding M37 materialization.
    """

    if type(request) is not AiofPreflopCertifiedGlobalRequest:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            "request must be AiofPreflopCertifiedGlobalRequest",
        )
    limits = _validate_integration_limits(request.integration_limits)
    pins = _validate_pins(request.pins)
    if request.algorithm is not EquityAlgorithm.EXACT_EXHAUSTIVE:
        raise AiofContractError(
            AiofStatus.UNSUPPORTED_MODEL,
            "certified optimization requires exact exhaustive equity",
        )
    if request.hero_seat not in ("sb", "bb"):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, "hero_seat must be 'sb' or 'bb'"
        )
    horizon = _require_plain_int(request.horizon, "horizon")
    if horizon > limits.max_horizon:
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            "max_horizon exceeded before discount-weight materialization",
        )
    adaptation = _require_plain_int(
        request.adaptation_opportunity, "adaptation_opportunity"
    )
    if adaptation > horizon + 1:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            "adaptation_opportunity must be in 1..horizon+1",
        )
    discount = _parse_exact_rational(
        request.discount, "discount", nonnegative=True
    )
    if not 0 < discount <= 1:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            "discount must satisfy 0 < discount <= 1",
        )
    response_tolerance = _parse_exact_rational(
        request.response_tolerance,
        "response_tolerance",
        nonnegative=True,
    )
    if response_tolerance != 0:
        raise AiofContractError(
            AiofStatus.UNSUPPORTED_MODEL,
            "nonzero response tolerance cannot be certified exactly",
        )
    _parse_exact_rational(
        request.absolute_gap_tolerance,
        "absolute_gap_tolerance",
        nonnegative=True,
    )
    _parse_exact_rational(
        request.relative_gap_tolerance,
        "relative_gap_tolerance",
        nonnegative=True,
    )
    if type(request.optimizer_limits) is not CertifiedGlobalOptimizerLimits:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            "optimizer_limits must be CertifiedGlobalOptimizerLimits",
        )
    if type(request.optimizer_pins) is not CertifiedGlobalOptimizerPins:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            "optimizer_pins must be CertifiedGlobalOptimizerPins",
        )

    sb_fold, sb_shove_bb_fold, effective_stack = _validate_game(
        request.game
    )
    effective_aiof = _effective_aiof_limits(request, limits)
    prepared = prepare_compatible_ranges(
        request.sb_range,
        request.bb_range,
        request.dead_cards,
        effective_aiof,
    )
    _check_pin(
        pins.prepared_ranges_identity,
        prepared.content_identity,
        "pins.prepared_ranges_identity",
    )
    if (
        len(prepared.sb_range.combos) > limits.max_information_sets_per_seat
        or len(prepared.bb_range.combos)
        > limits.max_information_sets_per_seat
    ):
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            "information-set cap exceeded before scenario materialization",
        )
    sb_masses, bb_masses, exact_support_identity = _exact_combo_masses(
        prepared, request.sb_range, request.bb_range
    )
    sb_baseline, bb_baseline = _canonical_baseline(
        request.baseline_profile, prepared
    )
    projected_coefficients = (
        prepared.compatible_pair_count
        + len(prepared.sb_range.combos)
        + len(prepared.bb_range.combos) * 3
    )
    if projected_coefficients > limits.max_coefficient_records:
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            "max_coefficient_records exceeded before coefficient materialization",
        )
    matchups, matchup_identity, board_evaluations = _build_matchups(
        prepared,
        sb_masses,
        bb_masses,
        effective_stack,
        limits,
    )
    prepared_projection = _prepared_projection(
        prepared,
        request.sb_range,
        request.bb_range,
        sb_masses,
        bb_masses,
    )
    scenario_projection = {
        "domain_contract": AIOF_PREFLOP_CERTIFIED_GLOBAL_DOMAIN,
        "deck_and_blockers": prepared_projection,
        "exact_support_identity": exact_support_identity,
        "matchup_identity": matchup_identity,
        "game": _request_projection(request)["game"],
        "accounting": CHIP_ACCOUNTING_ID,
        "hero_seat": request.hero_seat,
        "hero_information_sets": (
            sorted(sb_masses) if request.hero_seat == "sb" else sorted(bb_masses)
        ),
        "legal_actions": (
            ["fold", "shove"]
            if request.hero_seat == "sb"
            else ["call", "fold"]
        ),
    }
    scenario_identity = _identity(scenario_projection)
    _check_pin(
        pins.scenario_identity,
        scenario_identity,
        "pins.scenario_identity",
    )
    scenario, baseline_policy, active_action = _scenario_and_policy(
        hero_seat=request.hero_seat,
        prepared=prepared,
        sb_baseline=sb_baseline,
        bb_baseline=bb_baseline,
        scenario_identity=scenario_identity,
    )
    pre_affine, response_common, response_rows = _build_affines(
        hero_seat=request.hero_seat,
        matchups=matchups,
        sb_baseline=sb_baseline,
        bb_baseline=bb_baseline,
        sb_fold=sb_fold,
        sb_shove_bb_fold=sb_shove_bb_fold,
    )
    # Add exact, payoff-independent reach coefficients for Hero-SB / BB
    # responses.  They are private identity/provenance records, not actions.
    if request.hero_seat == "sb":
        rebuilt_rows: list[_ResponseAffineRow] = []
        for response_row in response_rows:
            combo = response_row.combo
            reach_coefficients: dict[str, Fraction] = {}
            for matchup in matchups:
                if matchup.bb_combo == combo:
                    _add_coefficient(
                        reach_coefficients,
                        f"hero:sb:{matchup.sb_combo}",
                        matchup.probability,
                    )
            rebuilt_rows.append(
                _ResponseAffineRow(
                    response_row.information_set_id,
                    response_row.combo,
                    response_row.marginal,
                    _Affine.make(Fraction(0), reach_coefficients),
                    response_row.actions,
                )
            )
        response_rows = tuple(rebuilt_rows)

    baseline_probabilities = {
        row.information_set_id: next(
            Fraction(action.probability)
            for action in row.actions
            if action.action_id == active_action
        )
        for row in baseline_policy.rows
    }
    baseline_hero_ev = pre_affine.value(baseline_probabilities)
    pre_weight, post_weight, total_weight = _discount_weights(
        horizon, adaptation, discount
    )
    response_identity = _identity(
        {
            "algorithm": AIOF_PREFLOP_CERTIFIED_GLOBAL_RESPONSE,
            "scenario_identity": scenario_identity,
            "matchup_identity": matchup_identity,
            "hero_seat": request.hero_seat,
            "response_tolerance": "0",
            "factorized_rows": [
                {
                    "information_set_id": row.information_set_id,
                    "combo": row.combo,
                    "marginal": _rational_text(row.marginal),
                    "reach": {
                        "constant": _rational_text(row.reach.constant),
                        "coefficients": [
                            {
                                "information_set_id": key,
                                "coefficient": _rational_text(value),
                            }
                            for key, value in row.reach.coefficients
                        ],
                    },
                    "actions": [
                        {
                            "action": action,
                            "constant": _rational_text(affine.constant),
                            "coefficients": [
                                {
                                    "information_set_id": key,
                                    "coefficient": _rational_text(value),
                                }
                                for key, value in affine.coefficients
                            ],
                        }
                        for action, affine in row.actions
                    ],
                }
                for row in response_rows
            ],
        }
    )
    _check_pin(
        pins.response_oracle_identity,
        response_identity,
        "pins.response_oracle_identity",
    )
    objective_identity = _identity(
        {
            "algorithm": AIOF_PREFLOP_CERTIFIED_GLOBAL_OBJECTIVE,
            "m36_objective_semantics": OBJECTIVE_SEMANTICS,
            "scenario_identity": scenario_identity,
            "baseline_policy_identity": behavior_policy_identity(
                scenario, baseline_policy
            ),
            "response_oracle_identity": response_identity,
            "baseline_hero_ev": _rational_text(baseline_hero_ev),
            "horizon": horizon,
            "adaptation_opportunity": adaptation,
            "discount": _rational_text(discount),
            "pre_weight": _rational_text(pre_weight),
            "post_weight": _rational_text(post_weight),
            "total_weight": _rational_text(total_weight),
            "baseline_policy_is_no_commitment_comparison": True,
        }
    )
    _check_pin(
        pins.objective_identity,
        objective_identity,
        "pins.objective_identity",
    )
    analysis_identity = _identity(
        {
            "contract_version": (
                AIOF_PREFLOP_CERTIFIED_GLOBAL_CONTRACT_VERSION
            ),
            "scenario_identity": scenario_identity,
            "prepared_ranges_identity": prepared.content_identity,
            "exact_support_identity": exact_support_identity,
            "matchup_identity": matchup_identity,
            "response_oracle_identity": response_identity,
            "objective_identity": objective_identity,
            "request": _request_projection(request),
        }
    )
    _check_pin(
        pins.analysis_identity,
        analysis_identity,
        "pins.analysis_identity",
    )
    initial_counters = AiofPreflopCertifiedOracleWorkCounters(
        compatible_pairs=prepared.compatible_pair_count,
        matchup_rows=len(matchups),
        exact_board_evaluations=board_evaluations,
        coefficient_records=projected_coefficients,
    )
    oracle = AiofPreflopCertifiedScalarOracle(
        scenario=scenario,
        baseline_policy=baseline_policy,
        hero_seat=request.hero_seat,
        hero_active_action=active_action,
        pre_affine=pre_affine,
        response_common=response_common,
        response_rows=response_rows,
        baseline_hero_ev=baseline_hero_ev,
        pre_weight=pre_weight,
        post_weight=post_weight,
        total_weight=total_weight,
        response_oracle_identity=response_identity,
        objective_identity=objective_identity,
        limits=limits,
        initial_counters=initial_counters,
    )
    return AiofPreflopCertifiedGlobalPreparation(
        scenario=scenario,
        baseline_policy=baseline_policy,
        prepared_ranges=prepared,
        prepared_ranges_projection=prepared_projection,
        matchup_identity=matchup_identity,
        exact_support_identity=exact_support_identity,
        response_oracle_identity=response_identity,
        objective_identity=objective_identity,
        analysis_identity=analysis_identity,
        effective_aiof_limits=effective_aiof,
        oracle=oracle,
    )


def _status_from_aiof(status: AiofStatus) -> str:
    if status is AiofStatus.CAP_EXCEEDED:
        return LIMIT_REACHED_NO_CERTIFICATE
    if status is AiofStatus.UNSUPPORTED_MODEL:
        return UNSUPPORTED_DOMAIN
    if status in (
        AiofStatus.NUMERIC_FAILURE,
        AiofStatus.ACCOUNTING_MISMATCH,
    ):
        return NUMERIC_FAILURE
    if status is AiofStatus.ORACLE_MISMATCH:
        return ORACLE_FAILURE
    return INVALID_INPUT


def _clean_message(message: str, fallback: str) -> str:
    cleaned = " ".join(
        (message or fallback).replace("\r", " ").replace("\n", " ").split()
    )
    return cleaned[:512] or fallback


def _empty_optimizer_counters() -> CertifiedGlobalWorkCounters:
    return CertifiedGlobalWorkCounters()


def analyze_aiof_preflop_certified_global(
    request: AiofPreflopCertifiedGlobalRequest,
) -> AiofPreflopCertifiedGlobalResult:
    """Run M37 and return only certified success or a null failure payload."""

    preparation: AiofPreflopCertifiedGlobalPreparation | None = None
    native: CertifiedGlobalOptimizerResult | None = None
    try:
        preparation = prepare_aiof_preflop_certified_global_oracle(request)
        native = optimize_certified_global_hero_commitment(
            preparation.scenario,
            preparation.baseline_policy,
            preparation.oracle,
            absolute_gap_tolerance=request.absolute_gap_tolerance,
            relative_gap_tolerance=request.relative_gap_tolerance,
            limits=request.optimizer_limits,
            pins=request.optimizer_pins,
        )
        if native.status not in (
            CERTIFIED_GLOBAL,
            CERTIFIED_EPSILON_GLOBAL,
        ):
            return AiofPreflopCertifiedGlobalResult(
                status=native.status,
                payload=None,
                error=AiofPreflopCertifiedGlobalError(
                    phase=(
                        native.error.phase
                        if native.error is not None
                        else "optimizer"
                    ),
                    message=(
                        native.error.message
                        if native.error is not None
                        else "optimizer returned no certificate"
                    ),
                    cause_status=(
                        native.error.cause_status
                        if native.error is not None
                        else None
                    ),
                ),
                optimizer_work_counters=native.work_counters,
                oracle_work_counters=preparation.oracle.work_counters,
                partial_result=False,
            )
        if native.payload is None or native.error is not None:
            raise RuntimeError("malformed successful native optimizer result")
        baseline_identity = behavior_policy_identity(
            preparation.scenario, preparation.baseline_policy
        )
        baseline_point = preparation.oracle.point_record(baseline_identity)
        selected_point = (
            None
            if native.payload.selected_policy_identity is None
            else preparation.oracle.point_record(
                native.payload.selected_policy_identity
            )
        )
        payload = AiofPreflopCertifiedGlobalPayload(
            preparation=preparation,
            baseline_point=baseline_point,
            selected_point=selected_point,
            native_optimizer_result=native,
            oracle_work_counters=preparation.oracle.work_counters,
            request=request,
        )
        result = AiofPreflopCertifiedGlobalResult(
            status=native.status,
            payload=payload,
            error=None,
            optimizer_work_counters=native.work_counters,
            oracle_work_counters=preparation.oracle.work_counters,
            partial_result=False,
        )
        _preflight_success_output(
            result,
            max_records=request.integration_limits.max_output_records,
            max_bytes=request.integration_limits.max_output_bytes,
        )
        return result
    except _StaleInput as exc:
        return AiofPreflopCertifiedGlobalResult(
            status=STALE_INPUT,
            payload=None,
            error=AiofPreflopCertifiedGlobalError(
                phase="pins", message=_clean_message(str(exc), "stale input")
            ),
            optimizer_work_counters=_empty_optimizer_counters(),
            oracle_work_counters=(
                AiofPreflopCertifiedOracleWorkCounters()
                if preparation is None
                else preparation.oracle.work_counters
            ),
            partial_result=False,
        )
    except _OutputLimitReached as exc:
        return AiofPreflopCertifiedGlobalResult(
            status=LIMIT_REACHED_NO_CERTIFICATE,
            payload=None,
            error=AiofPreflopCertifiedGlobalError(
                phase="output",
                message=_clean_message(
                    str(exc), "output resource limit"
                ),
            ),
            optimizer_work_counters=(
                _empty_optimizer_counters()
                if native is None
                else native.work_counters
            ),
            oracle_work_counters=(
                AiofPreflopCertifiedOracleWorkCounters()
                if preparation is None
                else preparation.oracle.work_counters
            ),
            partial_result=False,
        )
    except AiofContractError as exc:
        return AiofPreflopCertifiedGlobalResult(
            status=_status_from_aiof(exc.status),
            payload=None,
            error=AiofPreflopCertifiedGlobalError(
                phase="preparation",
                message=_clean_message(str(exc), exc.status.value),
                cause_status=exc.status.value,
            ),
            optimizer_work_counters=_empty_optimizer_counters(),
            oracle_work_counters=(
                AiofPreflopCertifiedOracleWorkCounters()
                if preparation is None
                else preparation.oracle.work_counters
            ),
            partial_result=False,
        )
    except UnsupportedScalarOracleDomain as exc:
        return AiofPreflopCertifiedGlobalResult(
            status=UNSUPPORTED_DOMAIN,
            payload=None,
            error=AiofPreflopCertifiedGlobalError(
                phase="oracle",
                message=_clean_message(str(exc), "unsupported oracle domain"),
            ),
            optimizer_work_counters=_empty_optimizer_counters(),
            oracle_work_counters=(
                AiofPreflopCertifiedOracleWorkCounters()
                if preparation is None
                else preparation.oracle.work_counters
            ),
            partial_result=False,
        )
    except ScalarOracleResourceLimit as exc:
        return AiofPreflopCertifiedGlobalResult(
            status=LIMIT_REACHED_NO_CERTIFICATE,
            payload=None,
            error=AiofPreflopCertifiedGlobalError(
                phase="oracle",
                message=_clean_message(str(exc), "oracle resource limit"),
            ),
            optimizer_work_counters=_empty_optimizer_counters(),
            oracle_work_counters=(
                AiofPreflopCertifiedOracleWorkCounters()
                if preparation is None
                else preparation.oracle.work_counters
            ),
            partial_result=False,
        )
    except Exception:
        return AiofPreflopCertifiedGlobalResult(
            status=ORACLE_FAILURE,
            payload=None,
            error=AiofPreflopCertifiedGlobalError(
                phase="internal",
                message="unexpected M37 integration failure",
            ),
            optimizer_work_counters=_empty_optimizer_counters(),
            oracle_work_counters=(
                AiofPreflopCertifiedOracleWorkCounters()
                if preparation is None
                else preparation.oracle.work_counters
            ),
            partial_result=False,
        )


def exact_aiof_preflop_certified_global_json(
    result: AiofPreflopCertifiedGlobalResult,
) -> str:
    """Serialize one M37 result as deterministic strict one-line JSON."""

    if type(result) is not AiofPreflopCertifiedGlobalResult:
        raise TypeError(
            "result must be AiofPreflopCertifiedGlobalResult"
        )
    return _canonical_json_bytes(result.to_dict()).decode("utf-8")
