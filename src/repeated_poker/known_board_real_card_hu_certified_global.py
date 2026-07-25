"""Certified global Hero-policy optimization for M29 known-board HU river.

This module is the M38 consumer of the M36 exact-rational optimizer.  It
prepares the same fixed five-card board, once-conditioned ordered private-card
support, seven-line IP-vs-OOP river tree, and rake accounting as M29.  The
optimization domain is every legal Hero behavior distribution at every
surviving M29 Hero information set; M29 shift candidates are never consulted.

For a Hero policy ``p``, ``b`` is the baseline fixed-profile Hero value,
``a(p)`` is the Hero value while Villain keeps that same complete baseline
profile, and ``l(p)`` is the Hero-worst value over a fresh complete exact
Villain best-response correspondence.  With adaptation opportunity ``m``,
horizon ``N``, and discount ``d``, the certified scalar is

``W_pre*a(p) + W_post*l(p) - (W_pre + W_post)*b``.

Every public binary64 probability, mass, rake, and game input used by this
consumer is lifted losslessly with ``float.as_integer_ratio``.  Conditioned
masses and terminal payoffs are reconstructed exactly and checked against the
native M29 binary64 tree.  The point oracle and the river-specific whole-cell
bound then use only ``Fraction``.
The bound is an interval version of M29's factorized perfect-recall response
DP: it retains every Villain action that can be optimal anywhere in the cell.
At singleton cells it is exactly the point DP.  No response pure-strategy
space is enumerated.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from typing import Any, Callable

from .aiof_cards import (
    AiofContractError,
    AiofLimits,
    AiofStatus,
    RangeSpec,
    canonicalize_exact_combo,
    canonicalize_hand_class,
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
from .game import (
    ChanceNode,
    HeroNode,
    TerminalNode,
    VillainNode,
    collect_hero_info_sets,
    collect_villain_info_sets,
    iter_nodes,
)
from .known_board_real_card_hu_river import (
    HERO_DECISIONS,
    OOP_FIRST,
    OOP_VS_IP_BET,
    OOP_VS_IP_RAISE,
    VILLAIN_DECISIONS,
    ActionProbability,
    ComboBucketMap,
    KnownBoardJointRow,
    KnownBoardRealCardHuRiverLimits,
    KnownBoardRealCardHuRiverRequest,
    RiverActionProfile,
    RiverProfileRow,
    _board_identity,
    _build_tree,
    _canonical_mapping,
    _canonical_profile,
    _mapping_identity,
    _materialize_joint_rows,
    _prepared_joint_identity,
    _prepared_ranges_checked,
    _project_joint_support,
    _tree_identity,
    _validate_request,
)

__all__ = [
    "KNOWN_BOARD_REAL_CARD_HU_CERTIFIED_GLOBAL_CONTRACT_VERSION",
    "KnownBoardRealCardHuCertifiedGlobalLimits",
    "KnownBoardRealCardHuCertifiedGlobalPins",
    "KnownBoardRealCardHuCertifiedGlobalRequest",
    "KnownBoardRealCardHuCompleteResponseRow",
    "KnownBoardRealCardHuCompleteResponse",
    "KnownBoardRealCardHuCertifiedPointRecord",
    "KnownBoardRealCardHuCertifiedOracleWorkCounters",
    "KnownBoardRealCardHuCertifiedGlobalPreparation",
    "KnownBoardRealCardHuCertifiedGlobalPayload",
    "KnownBoardRealCardHuCertifiedGlobalError",
    "KnownBoardRealCardHuCertifiedGlobalResult",
    "KnownBoardRealCardHuCertifiedScalarOracle",
    "prepare_known_board_real_card_hu_certified_global_oracle",
    "analyze_known_board_real_card_hu_certified_global",
    "exact_known_board_real_card_hu_certified_global_json",
]


KNOWN_BOARD_REAL_CARD_HU_CERTIFIED_GLOBAL_CONTRACT_VERSION = (
    "m38-known-board-real-card-hu-certified-global-integration-v1"
)
ALGORITHM_VERSION = "exact-rational-known-board-river-lex-dp-bnb-v1"
SCHEMA_VERSION = "m38-public-result-schema-v1"
DOMAIN_CONTRACT = "m29-full-surviving-hero-action-simplex-product-v1"
OBJECTIVE_CONTRACT = "m29-b-a-l-fixed-adaptation-total-uplift-v1"
RESPONSE_CONTRACT = "m29-complete-factorized-oop-lex-response-v1"
BOUND_CONTRACT = "m29-river-terminal-affine-lex-interval-dp-bound-v1"
CLAIM_SCOPE = "identified-bounded-scalar-objective-global-maximum-only-v1"
RATIONAL_LIFT = "lossless-public-binary64-as-integer-ratio-v1"
ACCOUNTING = "heads-up-river-rake-net-chips-v1"
UNIT = "net_chips_before_initial_commitments_per_river_opportunity"
POSITION = "hero_ip_villain_oop"

MAX_PREPARATION_RECORDS = 300_000
MAX_AFFINE_COEFFICIENTS = 1_000_000
MAX_RESPONSE_ROWS = 200_000
MAX_RESPONSE_ACTION_RECORDS = 600_000
MAX_ORACLE_POINT_EVALUATIONS = 2_097_152
MAX_ORACLE_BOUND_RECORDS = 1_048_575
MAX_OUTPUT_RECORDS = 2_000_000
MAX_OUTPUT_BYTES = 256_000_000
MAX_HORIZON = 100_000


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


def _rational_text(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _exact_binary64(value: object, name: str) -> Fraction:
    if type(value) is not float or not math.isfinite(value):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, f"{name} must be a finite binary64 float"
        )
    numerator, denominator = value.as_integer_ratio()
    return Fraction(numerator, denominator)


def _parse_rational(value: object, name: str, *, nonnegative: bool = False) -> Fraction:
    if type(value) is int:
        result = Fraction(value)
    elif type(value) is str:
        try:
            result = Fraction(value)
        except (ValueError, ZeroDivisionError) as exc:
            raise AiofContractError(
                AiofStatus.INVALID_INPUT, f"{name} must be exact rational text"
            ) from exc
        if _rational_text(result) != value:
            raise AiofContractError(
                AiofStatus.INVALID_INPUT, f"{name} must be canonical rational text"
            )
    else:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, f"{name} must be exact rational text"
        )
    if nonnegative and result < 0:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, f"{name} must be nonnegative"
        )
    return result


def _validate_positive_int(value: object, name: str, ceiling: int) -> int:
    if type(value) is not int or not 0 < value <= ceiling:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            f"{name} must be a positive int at most {ceiling}",
        )
    return value


@dataclass(frozen=True)
class KnownBoardRealCardHuCertifiedGlobalLimits:
    """Caller-lowerable M38 caps, each bounded by an immutable ceiling."""

    max_preparation_records: int = MAX_PREPARATION_RECORDS
    max_affine_coefficients: int = MAX_AFFINE_COEFFICIENTS
    max_response_rows: int = MAX_RESPONSE_ROWS
    max_response_action_records: int = MAX_RESPONSE_ACTION_RECORDS
    max_oracle_point_evaluations: int = MAX_ORACLE_POINT_EVALUATIONS
    max_oracle_bound_records: int = MAX_ORACLE_BOUND_RECORDS
    max_output_records: int = MAX_OUTPUT_RECORDS
    max_output_bytes: int = MAX_OUTPUT_BYTES
    max_horizon: int = MAX_HORIZON

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class KnownBoardRealCardHuCertifiedGlobalPins:
    """Optional identities checked before M36 is invoked."""

    request_identity: str | None = None
    board_identity: str | None = None
    prepared_joint_identity: str | None = None
    profile_mapping_identity: str | None = None
    tree_identity: str | None = None
    baseline_identity: str | None = None
    scenario_identity: str | None = None
    response_oracle_identity: str | None = None
    objective_identity: str | None = None
    analysis_identity: str | None = None


@dataclass(frozen=True)
class KnownBoardRealCardHuCertifiedGlobalRequest:
    """One bounded M38 in-memory request.

    Card/range/profile/game inputs intentionally reuse M29 types and binary64
    conventions.  Discount and M36 gap tolerances are canonical exact rational
    text.  ``response_tolerance`` must be exact zero.
    """

    board: tuple[str, ...]
    hero_range: RangeSpec
    villain_range: RangeSpec
    baseline_hero_profile: RiverActionProfile
    dead_cards: tuple[str, ...] = ()
    hero_combo_to_bucket: ComboBucketMap | None = None
    villain_combo_to_bucket: ComboBucketMap | None = None
    baseline_villain_profile: RiverActionProfile | None = None
    initial_commitment_hero: float = 1.0
    initial_commitment_villain: float = 1.0
    rake_rate: float = 0.0
    rake_cap: float | None = None
    oop_bet_size: float = 1.0
    ip_bet_after_check_size: float = 1.0
    ip_raise_to_size: float = 3.0
    horizon: int = 1
    adaptation_opportunity: int = 1
    discount: str = "1"
    response_tolerance: str = "0"
    absolute_gap_tolerance: str = "0"
    relative_gap_tolerance: str = "0"
    aiof_limits: AiofLimits = AiofLimits()
    river_limits: KnownBoardRealCardHuRiverLimits = (
        KnownBoardRealCardHuRiverLimits()
    )
    integration_limits: KnownBoardRealCardHuCertifiedGlobalLimits = (
        KnownBoardRealCardHuCertifiedGlobalLimits()
    )
    optimizer_limits: CertifiedGlobalOptimizerLimits = (
        CertifiedGlobalOptimizerLimits()
    )
    pins: KnownBoardRealCardHuCertifiedGlobalPins = (
        KnownBoardRealCardHuCertifiedGlobalPins()
    )
    optimizer_pins: CertifiedGlobalOptimizerPins = CertifiedGlobalOptimizerPins()


@dataclass(frozen=True)
class KnownBoardRealCardHuCompleteResponseRow:
    information_set_id: str
    villain_bucket_id: str
    villain_history: tuple[tuple[str, str], ...]
    conditional_best_actions: tuple[str, ...]
    globally_appearing_actions: tuple[str, ...]
    action_values: tuple[tuple[str, str, str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "information_set_id": self.information_set_id,
            "villain_bucket_id": self.villain_bucket_id,
            "villain_history": [
                {"information_set_id": info, "action": action}
                for info, action in self.villain_history
            ],
            "conditional_best_actions": list(self.conditional_best_actions),
            "globally_appearing_actions": list(self.globally_appearing_actions),
            "action_values": [
                {
                    "action": action,
                    "hero_contribution": hero,
                    "villain_contribution": villain,
                    "house_rake": rake,
                }
                for action, hero, villain, rake in self.action_values
            ],
        }


@dataclass(frozen=True)
class KnownBoardRealCardHuCompleteResponse:
    hero_worst_value: str
    hero_best_value: str
    villain_max_value: str
    house_rake_at_hero_worst: str
    rows: tuple[KnownBoardRealCardHuCompleteResponseRow, ...]
    action_variation_information_sets: tuple[str, ...]
    correspondence_identity: str
    complete_response_record_count: int
    hero_worst_witness_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantics": RESPONSE_CONTRACT,
            "opponent_seat": "oop",
            "hero_worst_value": self.hero_worst_value,
            "hero_best_value": self.hero_best_value,
            "villain_max_value": self.villain_max_value,
            "house_rake_at_hero_worst": self.house_rake_at_hero_worst,
            "rows": [row.to_dict() for row in self.rows],
            "action_variation_information_sets": list(
                self.action_variation_information_sets
            ),
            "correspondence_identity": self.correspondence_identity,
            "complete_response_record_count": self.complete_response_record_count,
            "hero_worst_witness_count": self.hero_worst_witness_count,
        }


@dataclass(frozen=True)
class KnownBoardRealCardHuCertifiedPointRecord:
    policy_identity: str
    fixed_baseline_villain_hero_ev: str
    post_response_hero_ev_worst: str
    baseline_total_repeated_hero_ev: str
    candidate_total_repeated_hero_ev: str
    uplift: str
    response: KnownBoardRealCardHuCompleteResponse
    native_evaluation: ScalarOracleEvaluation

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_identity": self.policy_identity,
            "fixed_baseline_villain_hero_ev": (
                self.fixed_baseline_villain_hero_ev
            ),
            "post_response_hero_ev_worst": self.post_response_hero_ev_worst,
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
class KnownBoardRealCardHuCertifiedOracleWorkCounters:
    compatible_pairs: int = 0
    fixed_board_evaluations: int = 0
    tree_nodes: int = 0
    terminal_records: int = 0
    affine_coefficients: int = 0
    point_evaluations: int = 0
    response_rows: int = 0
    response_action_records: int = 0
    bound_records: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class _Affine:
    constant: Fraction
    coefficients: tuple[tuple[str, str, Fraction], ...]

    @classmethod
    def make(
        cls,
        constant: Fraction = Fraction(0),
        coefficients: dict[tuple[str, str], Fraction] | None = None,
    ) -> "_Affine":
        return cls(
            Fraction(constant),
            tuple(
                (info, action, Fraction(value))
                for (info, action), value in sorted((coefficients or {}).items())
                if value
            ),
        )

    def value(self, probabilities: dict[tuple[str, str], Fraction]) -> Fraction:
        return self.constant + sum(
            (
                coefficient * probabilities[(info, action)]
                for info, action, coefficient in self.coefficients
            ),
            Fraction(0),
        )

    def plus(self, other: "_Affine") -> "_Affine":
        output: dict[tuple[str, str], Fraction] = {}
        for info, action, value in self.coefficients + other.coefficients:
            key = (info, action)
            output[key] = output.get(key, Fraction(0)) + value
        return _Affine.make(self.constant + other.constant, output)

    def scale(self, factor: Fraction) -> "_Affine":
        return _Affine.make(
            self.constant * factor,
            {
                (info, action): coefficient * factor
                for info, action, coefficient in self.coefficients
            },
        )

    def projection(self) -> dict[str, Any]:
        return {
            "constant": _rational_text(self.constant),
            "coefficients": [
                {
                    "information_set_id": info,
                    "action": action,
                    "coefficient": _rational_text(coefficient),
                }
                for info, action, coefficient in self.coefficients
            ],
        }


@dataclass(frozen=True)
class _TripleAffine:
    hero: _Affine
    villain: _Affine
    rake: _Affine

    def plus(self, other: "_TripleAffine") -> "_TripleAffine":
        return _TripleAffine(
            self.hero.plus(other.hero),
            self.villain.plus(other.villain),
            self.rake.plus(other.rake),
        )

    def value(
        self, probabilities: dict[tuple[str, str], Fraction]
    ) -> tuple[Fraction, Fraction, Fraction]:
        return (
            self.hero.value(probabilities),
            self.villain.value(probabilities),
            self.rake.value(probabilities),
        )

    def scale(self, factor: Fraction) -> "_TripleAffine":
        return _TripleAffine(
            self.hero.scale(factor),
            self.villain.scale(factor),
            self.rake.scale(factor),
        )


_ZERO_TRIPLE = _TripleAffine(_Affine.make(), _Affine.make(), _Affine.make())


@dataclass(frozen=True)
class _RiverRowAffines:
    villain_bucket_id: str
    check_direct: _TripleAffine
    check_bet_actions: tuple[tuple[str, _TripleAffine], ...]
    bet_direct: _TripleAffine
    bet_raise_actions: tuple[tuple[str, _TripleAffine], ...]


@dataclass(frozen=True)
class _BucketAffines:
    villain_bucket_id: str
    root_information_set_id: str
    check_bet_information_set_id: str
    bet_raise_information_set_id: str
    check_direct: _TripleAffine
    check_bet_actions: tuple[tuple[str, _TripleAffine], ...]
    bet_direct: _TripleAffine
    bet_raise_actions: tuple[tuple[str, _TripleAffine], ...]

    def projection(self) -> dict[str, Any]:
        def action_rows(rows: tuple[tuple[str, _TripleAffine], ...]) -> list[dict]:
            return [
                {
                    "action": action,
                    "hero": triple.hero.projection(),
                    "villain": triple.villain.projection(),
                    "rake": triple.rake.projection(),
                }
                for action, triple in rows
            ]

        return {
            "villain_bucket_id": self.villain_bucket_id,
            "information_sets": {
                "root": self.root_information_set_id,
                "after_check_bet": self.check_bet_information_set_id,
                "after_bet_raise": self.bet_raise_information_set_id,
            },
            "check_direct": {
                "hero": self.check_direct.hero.projection(),
                "villain": self.check_direct.villain.projection(),
                "rake": self.check_direct.rake.projection(),
            },
            "check_bet_actions": action_rows(self.check_bet_actions),
            "bet_direct": {
                "hero": self.bet_direct.hero.projection(),
                "villain": self.bet_direct.villain.projection(),
                "rake": self.bet_direct.rake.projection(),
            },
            "bet_raise_actions": action_rows(self.bet_raise_actions),
        }


@dataclass(frozen=True)
class _PointSolution:
    villain: Fraction
    hero_worst: Fraction
    hero_best: Fraction
    rake_worst: Fraction
    best_actions: tuple[str, ...]
    hero_worst_actions: tuple[str, ...]
    values: tuple[tuple[str, Fraction, Fraction, Fraction], ...]


@dataclass(frozen=True)
class _BucketPoint:
    bucket: _BucketAffines
    check_bet: _PointSolution
    bet_raise: _PointSolution
    root: _PointSolution
    complete_count: int
    hero_worst_count: int


@dataclass(frozen=True)
class KnownBoardRealCardHuCertifiedGlobalPreparation:
    """Prepared M29 surface and the exact M38/M36 oracle."""

    scenario: HeroBehaviorScenario
    baseline_policy: ExactBehaviorPolicy
    board: tuple[str, ...]
    dead_cards: tuple[str, ...]
    joint_rows: tuple[KnownBoardJointRow, ...]
    exact_joint_probabilities: tuple[str, ...]
    joint_provenance: Any
    hero_mapping: ComboBucketMap
    villain_mapping: ComboBucketMap
    board_identity: str
    prepared_joint_identity: str
    profile_mapping_identity: str
    tree_identity: str
    baseline_identity: str
    request_identity: str
    response_oracle_identity: str
    objective_identity: str
    analysis_identity: str
    oracle: "KnownBoardRealCardHuCertifiedScalarOracle"

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
            "position": POSITION,
            "board": list(self.board),
            "extra_dead_cards": list(self.dead_cards),
            "identities": {
                "request": self.request_identity,
                "board": self.board_identity,
                "prepared_joint": self.prepared_joint_identity,
                "profile_mapping": self.profile_mapping_identity,
                "tree": self.tree_identity,
                "baseline": self.baseline_identity,
                "scenario": self.scenario.scenario_identity,
                "response_oracle": self.response_oracle_identity,
                "objective": self.objective_identity,
                "analysis": self.analysis_identity,
            },
            "joint_provenance": self.joint_provenance.to_dict(),
            "mappings": {
                "hero": self.hero_mapping.to_dict(),
                "villain": self.villain_mapping.to_dict(),
            },
            "joint_rows": [row.to_dict() for row in self.joint_rows],
            "exact_conditioned_joint": [
                {
                    "hero_combo": row.hero_combo,
                    "villain_combo": row.villain_combo,
                    "probability": probability,
                }
                for row, probability in zip(
                    self.joint_rows, self.exact_joint_probabilities
                )
            ],
            "rational_lift": RATIONAL_LIFT,
        }


@dataclass(frozen=True)
class KnownBoardRealCardHuCertifiedGlobalPayload:
    preparation: KnownBoardRealCardHuCertifiedGlobalPreparation
    baseline_point: KnownBoardRealCardHuCertifiedPointRecord
    selected_point: KnownBoardRealCardHuCertifiedPointRecord | None
    native_optimizer_result: CertifiedGlobalOptimizerResult
    oracle_work_counters: KnownBoardRealCardHuCertifiedOracleWorkCounters
    request: KnownBoardRealCardHuCertifiedGlobalRequest

    def to_dict(self) -> dict[str, Any]:
        native_payload = self.native_optimizer_result.payload
        return {
            "contract_version": KNOWN_BOARD_REAL_CARD_HU_CERTIFIED_GLOBAL_CONTRACT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "domain_contract": DOMAIN_CONTRACT,
            "objective_contract": OBJECTIVE_CONTRACT,
            "response_contract": RESPONSE_CONTRACT,
            "bound_contract": BOUND_CONTRACT,
            "certificate_claim": CLAIM_SCOPE,
            "accounting": ACCOUNTING,
            "unit": UNIT,
            "position": POSITION,
            "preparation": self.preparation.to_dict(),
            "repeated_configuration": {
                "horizon": self.request.horizon,
                "adaptation_opportunity": self.request.adaptation_opportunity,
                "discount": self.request.discount,
                "response_tolerance": self.request.response_tolerance,
                "absolute_gap_tolerance": self.request.absolute_gap_tolerance,
                "relative_gap_tolerance": self.request.relative_gap_tolerance,
            },
            "caps": {
                "aiof": asdict(self.request.aiof_limits),
                "river": self.request.river_limits.to_dict(),
                "integration": self.request.integration_limits.to_dict(),
                "optimizer": asdict(self.request.optimizer_limits),
            },
            "baseline_point": self.baseline_point.to_dict(),
            "selected_point": (
                None if self.selected_point is None else self.selected_point.to_dict()
            ),
            "no_beneficial_commitment": (
                None
                if native_payload is None
                else native_payload.no_beneficial_commitment
            ),
            "native_optimizer_result": self.native_optimizer_result.to_dict(),
            "oracle_work_counters": self.oracle_work_counters.to_dict(),
        }


@dataclass(frozen=True)
class KnownBoardRealCardHuCertifiedGlobalError:
    phase: str
    message: str
    cause_status: str | None = None


@dataclass(frozen=True)
class KnownBoardRealCardHuCertifiedGlobalResult:
    """Strict exclusive success/failure wrapper; partial output is forbidden."""

    status: str
    payload: KnownBoardRealCardHuCertifiedGlobalPayload | None
    error: KnownBoardRealCardHuCertifiedGlobalError | None
    optimizer_work_counters: CertifiedGlobalWorkCounters
    oracle_work_counters: KnownBoardRealCardHuCertifiedOracleWorkCounters
    partial_result: bool = False

    def __post_init__(self) -> None:
        success = self.status in (CERTIFIED_GLOBAL, CERTIFIED_EPSILON_GLOBAL)
        if success != (self.payload is not None and self.error is None):
            raise ValueError("success requires payload/non-error; failure requires error/null")
        if not success and (self.payload is not None or self.error is None):
            raise ValueError("failure must have null payload and one error")
        if self.partial_result is not False:
            raise ValueError("M38 never exposes a partial result")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "status": self.status,
            "payload": None if self.payload is None else self.payload.to_dict(),
            "error": None if self.error is None else asdict(self.error),
            "optimizer_work_counters": self.optimizer_work_counters.to_dict(),
            "oracle_work_counters": self.oracle_work_counters.to_dict(),
            "partial_result": False,
        }
        _canonical_json_bytes(result)
        return result


class _StaleInput(ValueError):
    pass


class _OutputLimitReached(RuntimeError):
    pass


def _validate_limits(
    value: object,
) -> KnownBoardRealCardHuCertifiedGlobalLimits:
    if type(value) is not KnownBoardRealCardHuCertifiedGlobalLimits:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            "integration_limits must be KnownBoardRealCardHuCertifiedGlobalLimits",
        )
    ceilings = {
        "max_preparation_records": MAX_PREPARATION_RECORDS,
        "max_affine_coefficients": MAX_AFFINE_COEFFICIENTS,
        "max_response_rows": MAX_RESPONSE_ROWS,
        "max_response_action_records": MAX_RESPONSE_ACTION_RECORDS,
        "max_oracle_point_evaluations": MAX_ORACLE_POINT_EVALUATIONS,
        "max_oracle_bound_records": MAX_ORACLE_BOUND_RECORDS,
        "max_output_records": MAX_OUTPUT_RECORDS,
        "max_output_bytes": MAX_OUTPUT_BYTES,
        "max_horizon": MAX_HORIZON,
    }
    for field in fields(value):
        _validate_positive_int(
            getattr(value, field.name),
            f"integration_limits.{field.name}",
            ceilings[field.name],
        )
    return value


def _validate_pins(value: object) -> KnownBoardRealCardHuCertifiedGlobalPins:
    if type(value) is not KnownBoardRealCardHuCertifiedGlobalPins:
        raise AiofContractError(AiofStatus.INVALID_INPUT, "pins has invalid type")
    normalized: dict[str, str | None] = {}
    for field in fields(value):
        item = getattr(value, field.name)
        token = (
            item[7:]
            if type(item) is str and item.startswith("sha256:")
            else item
        )
        if item is not None and (
            type(token) is not str
            or len(token) != 64
            or any(char not in "0123456789abcdef" for char in token)
        ):
            raise AiofContractError(
                AiofStatus.INVALID_INPUT,
                f"pins.{field.name} must be lowercase SHA-256 or None",
            )
        normalized[field.name] = token
    return KnownBoardRealCardHuCertifiedGlobalPins(**normalized)


def _check_pin(expected: str | None, actual: str, name: str) -> None:
    actual_token = actual[7:] if actual.startswith("sha256:") else actual
    if expected is not None and expected != actual_token:
        raise _StaleInput(f"{name} mismatch")


def _discount_weights(
    horizon: int, adaptation_opportunity: int, discount: Fraction
) -> tuple[Fraction, Fraction, Fraction]:
    terms: list[Fraction] = []
    term = Fraction(1)
    for _ in range(horizon):
        terms.append(term)
        term *= discount
    prefix = adaptation_opportunity - 1
    pre = sum(terms[:prefix], Fraction(0))
    post = sum(terms[prefix:], Fraction(0))
    return pre, post, pre + post


def _request_projection(
    request: KnownBoardRealCardHuCertifiedGlobalRequest,
) -> dict[str, Any]:
    def binary(value: float | None, name: str) -> str | None:
        return None if value is None else _rational_text(_exact_binary64(value, name))

    def mapping(value: ComboBucketMap | None) -> dict[str, Any] | None:
        if value is None:
            return None
        return {
            "bucket_ids": sorted(value.bucket_ids),
            "assignments": sorted(
                (
                    {
                        "combo": canonicalize_exact_combo(row.combo),
                        "bucket_id": row.bucket_id,
                    }
                    for row in value.assignments
                ),
                key=lambda row: (row["combo"], row["bucket_id"]),
            ),
        }

    def profile(value: RiverActionProfile | None, name: str) -> list[dict] | None:
        if value is None:
            return None
        return sorted(
            (
                {
                    "bucket_id": row.bucket_id,
                    "decision": row.decision,
                    "actions": sorted(
                        (
                            {
                                "action": action.action,
                                "probability": binary(
                                    action.probability, f"{name}.probability"
                                ),
                            }
                            for action in row.actions
                        ),
                        key=lambda item: item["action"],
                    ),
                }
                for row in value.rows
            ),
            key=lambda row: (row["bucket_id"], row["decision"]),
        )

    return {
        "contract_version": KNOWN_BOARD_REAL_CARD_HU_CERTIFIED_GLOBAL_CONTRACT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "board": sorted(request.board),
        "dead_cards": sorted(request.dead_cards),
        "game": {
            "initial_commitment_hero": binary(
                request.initial_commitment_hero, "initial_commitment_hero"
            ),
            "initial_commitment_villain": binary(
                request.initial_commitment_villain, "initial_commitment_villain"
            ),
            "rake_rate": binary(request.rake_rate, "rake_rate"),
            "rake_cap": binary(request.rake_cap, "rake_cap"),
            "oop_bet_size": binary(request.oop_bet_size, "oop_bet_size"),
            "ip_bet_after_check_size": binary(
                request.ip_bet_after_check_size, "ip_bet_after_check_size"
            ),
            "ip_raise_to_size": binary(
                request.ip_raise_to_size, "ip_raise_to_size"
            ),
        },
        "range_inputs": {
            "hero": sorted([
                {
                    "label": _canonical_range_label(item.label),
                    "weight": binary(item.weight, "hero_range.weight"),
                    "weight_basis": item.weight_basis.value,
                }
                for item in request.hero_range.entries
            ], key=lambda item: (item["label"], item["weight_basis"])),
            "villain": sorted([
                {
                    "label": _canonical_range_label(item.label),
                    "weight": binary(item.weight, "villain_range.weight"),
                    "weight_basis": item.weight_basis.value,
                }
                for item in request.villain_range.entries
            ], key=lambda item: (item["label"], item["weight_basis"])),
        },
        "hero_mapping": mapping(request.hero_combo_to_bucket),
        "villain_mapping": mapping(request.villain_combo_to_bucket),
        "baseline_hero_profile": profile(
            request.baseline_hero_profile, "baseline_hero_profile"
        ),
        "baseline_villain_profile": profile(
            request.baseline_villain_profile, "baseline_villain_profile"
        ),
        "repeated": {
            "horizon": request.horizon,
            "adaptation_opportunity": request.adaptation_opportunity,
            "discount": request.discount,
            "response_tolerance": request.response_tolerance,
            "absolute_gap_tolerance": request.absolute_gap_tolerance,
            "relative_gap_tolerance": request.relative_gap_tolerance,
        },
        "caps": {
            "aiof": asdict(request.aiof_limits),
            "river": request.river_limits.to_dict(),
            "integration": request.integration_limits.to_dict(),
            "optimizer": asdict(request.optimizer_limits),
        },
    }


def _exact_profile_map(
    profile: RiverActionProfile,
) -> dict[str, dict[str, Fraction]]:
    result: dict[str, dict[str, Fraction]] = {}
    for row in profile.rows:
        info_set = {
            "after_oop_check": "IP_after_OOP_check",
            "vs_oop_bet": "IP_vs_OOP_bet",
            "oop_first": OOP_FIRST,
            "vs_ip_bet": OOP_VS_IP_BET,
            "vs_ip_raise": OOP_VS_IP_RAISE,
        }[row.decision] + f"::{row.bucket_id}"
        action_map = {
            action.action: _exact_binary64(
                action.probability, f"profile[{info_set}].{action.action}"
            )
            for action in row.actions
        }
        if sum(action_map.values(), Fraction(0)) != 1:
            raise AiofContractError(
                AiofStatus.INVALID_STRATEGY,
                f"profile[{info_set}] does not sum exactly to one after lossless lift",
            )
        result[info_set] = action_map
    return result


def _scenario_and_policy(
    hero_info_sets: dict[str, tuple[str, ...]],
    hero_profile: RiverActionProfile,
    scenario_identity: str,
) -> tuple[HeroBehaviorScenario, ExactBehaviorPolicy]:
    exact = _exact_profile_map(hero_profile)
    rows = tuple(
        HeroInformationSet(info_set, tuple(hero_info_sets[info_set]))
        for info_set in sorted(hero_info_sets)
    )
    scenario = HeroBehaviorScenario(scenario_identity, rows)
    policy = ExactBehaviorPolicy(
        rows=tuple(
            ExactBehaviorRow(
                row.information_set_id,
                tuple(
                    ExactActionProbability(
                        action,
                        _rational_text(exact[row.information_set_id][action]),
                    )
                    for action in row.legal_action_ids
                ),
            )
            for row in rows
        )
    )
    return scenario, policy


def _terminal_triple(
    probability: Fraction,
    information_set_id: str,
    hero_action: str,
    terminal: TerminalNode,
    exact_values: tuple[Fraction, Fraction, Fraction],
) -> _TripleAffine:
    key = (information_set_id, hero_action)
    hero, villain, rake = exact_values
    native = (terminal.hero_ev, terminal.villain_ev, terminal.house_rake)
    if any(
        abs(float(exact) - observed) > 1e-12
        for exact, observed in zip(exact_values, native)
    ):
        raise AiofContractError(
            AiofStatus.ORACLE_MISMATCH,
            f"{terminal.node_id} differs from exact M29 payoff reconstruction",
        )
    if hero + villain + rake != 0 or rake < 0:
        raise AiofContractError(
            AiofStatus.ACCOUNTING_MISMATCH,
            f"{terminal.node_id} fails exact terminal conservation",
        )
    return _TripleAffine(
        _Affine.make(coefficients={key: probability * hero}),
        _Affine.make(coefficients={key: probability * villain}),
        _Affine.make(coefficients={key: probability * rake}),
    )


def _action_child(node: HeroNode | VillainNode, action: str) -> Any:
    for candidate, child in node.actions:
        if candidate == action:
            return child
    raise AiofContractError(
        AiofStatus.ORACLE_MISMATCH,
        f"{node.node_id} lacks expected action {action}",
    )


def _as_terminal(value: Any, name: str) -> TerminalNode:
    if not isinstance(value, TerminalNode):
        raise AiofContractError(
            AiofStatus.ORACLE_MISMATCH, f"{name} is not a terminal"
        )
    return value


def _exact_showdown(
    hero_invested: Fraction,
    villain_invested: Fraction,
    result: str,
    rate: Fraction,
    cap: Fraction | None,
) -> tuple[Fraction, Fraction, Fraction]:
    pot = hero_invested + villain_invested
    rake = rate * pot
    if cap is not None:
        rake = min(rake, cap)
    awarded = pot - rake
    if result == "hero":
        received = (awarded, Fraction(0))
    elif result == "villain":
        received = (Fraction(0), awarded)
    else:
        received = (awarded / 2, awarded / 2)
    return (
        received[0] - hero_invested,
        received[1] - villain_invested,
        rake,
    )


def _exact_terminal_lines(
    request: KnownBoardRealCardHuRiverRequest,
    row: KnownBoardJointRow,
) -> dict[str, tuple[Fraction, Fraction, Fraction]]:
    hero_initial = _exact_binary64(
        request.initial_commitment_hero, "initial_commitment_hero"
    )
    villain_initial = _exact_binary64(
        request.initial_commitment_villain, "initial_commitment_villain"
    )
    oop_bet = _exact_binary64(request.oop_bet_size, "oop_bet_size")
    ip_bet = _exact_binary64(
        request.ip_bet_after_check_size, "ip_bet_after_check_size"
    )
    raise_to = _exact_binary64(request.ip_raise_to_size, "ip_raise_to_size")
    rate = _exact_binary64(request.rake_rate, "rake_rate")
    cap = (
        None
        if request.rake_cap is None
        else _exact_binary64(request.rake_cap, "rake_cap")
    )
    result = row.showdown_result
    return {
        "cc": _exact_showdown(
            hero_initial, villain_initial, result, rate, cap
        ),
        "cbc": _exact_showdown(
            hero_initial + ip_bet,
            villain_initial + ip_bet,
            result,
            rate,
            cap,
        ),
        "cbf": (villain_initial, -villain_initial, Fraction(0)),
        "bc": _exact_showdown(
            hero_initial + oop_bet,
            villain_initial + oop_bet,
            result,
            rate,
            cap,
        ),
        "bf": (-hero_initial, hero_initial, Fraction(0)),
        "brc": _exact_showdown(
            hero_initial + raise_to,
            villain_initial + raise_to,
            result,
            rate,
            cap,
        ),
        "brf": (
            villain_initial + oop_bet,
            -(villain_initial + oop_bet),
            Fraction(0),
        ),
    }


def _row_affines(
    probability: Fraction,
    row: KnownBoardJointRow,
    root: VillainNode,
    lines: dict[str, tuple[Fraction, Fraction, Fraction]],
) -> _RiverRowAffines:
    if root.info_set != f"{OOP_FIRST}::{row.villain_bucket_id}":
        raise AiofContractError(
            AiofStatus.ORACLE_MISMATCH, "M29 root information set changed"
        )
    after_check = _action_child(root, "check")
    after_bet = _action_child(root, "bet")
    if not isinstance(after_check, HeroNode) or not isinstance(after_bet, HeroNode):
        raise AiofContractError(
            AiofStatus.ORACLE_MISMATCH, "M29 Hero decision shape changed"
        )
    check_direct = _terminal_triple(
        probability,
        after_check.info_set,
        "check",
        _as_terminal(_action_child(after_check, "check"), "check-check"),
        lines["cc"],
    )
    check_bet = _action_child(after_check, "bet")
    if not isinstance(check_bet, VillainNode):
        raise AiofContractError(
            AiofStatus.ORACLE_MISMATCH, "M29 check-bet response shape changed"
        )
    check_bet_actions = tuple(
        (
            action,
            _terminal_triple(
                probability,
                after_check.info_set,
                "bet",
                _as_terminal(child, "check-bet terminal"),
                lines["cbc" if action == "call" else "cbf"],
            ),
        )
        for action, child in check_bet.actions
    )
    bet_direct = _terminal_triple(
        probability,
        after_bet.info_set,
        "call",
        _as_terminal(_action_child(after_bet, "call"), "bet-call"),
        lines["bc"],
    ).plus(
        _terminal_triple(
            probability,
            after_bet.info_set,
            "fold",
            _as_terminal(_action_child(after_bet, "fold"), "bet-fold"),
            lines["bf"],
        )
    )
    bet_raise = _action_child(after_bet, "raise")
    if not isinstance(bet_raise, VillainNode):
        raise AiofContractError(
            AiofStatus.ORACLE_MISMATCH, "M29 bet-raise response shape changed"
        )
    bet_raise_actions = tuple(
        (
            action,
            _terminal_triple(
                probability,
                after_bet.info_set,
                "raise",
                _as_terminal(child, "bet-raise terminal"),
                lines["brc" if action == "call" else "brf"],
            ),
        )
        for action, child in bet_raise.actions
    )
    return _RiverRowAffines(
        row.villain_bucket_id,
        check_direct,
        check_bet_actions,
        bet_direct,
        bet_raise_actions,
    )


def _sum_triples(values: Any) -> _TripleAffine:
    result = _ZERO_TRIPLE
    for value in values:
        result = result.plus(value)
    return result


def _canonical_range_label(value: str) -> str:
    if len(value) == 4:
        try:
            return canonicalize_exact_combo(value)
        except AiofContractError:
            pass
    return canonicalize_hand_class(value)


def _class_divisor(source_label: str) -> int:
    if len(source_label) == 4:
        return 1
    if len(source_label) == 2:
        return 6
    if source_label.endswith("s"):
        return 4
    if source_label.endswith("o"):
        return 12
    raise AiofContractError(
        AiofStatus.ORACLE_MISMATCH,
        f"unsupported prepared source label {source_label!r}",
    )


def _exact_expanded_masses(
    spec: RangeSpec, expanded: Any, name: str
) -> dict[str, Fraction]:
    sources: dict[str, Fraction] = {}
    for index, entry in enumerate(spec.entries):
        label = _canonical_range_label(entry.label)
        if label in sources:
            raise AiofContractError(
                AiofStatus.INVALID_RANGE, f"{name} has duplicate source label"
            )
        sources[label] = _exact_binary64(
            entry.weight, f"{name}[{index}].weight"
        )
    output: dict[str, Fraction] = {}
    for combo in expanded.combos:
        if combo.source_label not in sources:
            raise AiofContractError(
                AiofStatus.ORACLE_MISMATCH,
                f"{name} prepared source is absent from request",
            )
        output[combo.combo] = (
            sources[combo.source_label] / _class_divisor(combo.source_label)
        )
    return output


def _exact_joint_probabilities(
    request: KnownBoardRealCardHuRiverRequest,
    projection: Any,
    rows: tuple[KnownBoardJointRow, ...],
) -> tuple[Fraction, ...]:
    hero = _exact_expanded_masses(
        request.hero_range, projection.hero_final, "hero_range"
    )
    villain = _exact_expanded_masses(
        request.villain_range, projection.villain_final, "villain_range"
    )
    masses = tuple(
        hero[row.hero_combo] * villain[row.villain_combo] for row in rows
    )
    total = sum(masses, Fraction(0))
    if total <= 0:
        raise AiofContractError(
            AiofStatus.NUMERIC_FAILURE, "exact compatible joint mass is not positive"
        )
    for row, mass in zip(rows, masses):
        if abs(float(mass) - row.raw_joint_mass) > 1e-12 * max(
            1.0, abs(row.raw_joint_mass)
        ):
            raise AiofContractError(
                AiofStatus.ORACLE_MISMATCH,
                "exact source-mass reconstruction differs from M29 joint row",
            )
    probabilities = tuple(mass / total for mass in masses)
    if sum(probabilities, Fraction(0)) != 1:
        raise AiofContractError(
            AiofStatus.ACCOUNTING_MISMATCH,
            "exact conditioned joint probabilities do not sum to one",
        )
    return probabilities


def _aggregate_bucket_affines(
    tree: Any,
    rows: tuple[KnownBoardJointRow, ...],
    request: KnownBoardRealCardHuRiverRequest,
    probabilities: tuple[Fraction, ...],
) -> tuple[_BucketAffines, ...]:
    if not isinstance(tree.root, ChanceNode) or len(tree.root.children) != len(rows):
        raise AiofContractError(
            AiofStatus.ORACLE_MISMATCH, "M29 chance-root shape changed"
        )
    materialized: list[_RiverRowAffines] = []
    for (_, root), row, probability in zip(
        tree.root.children, rows, probabilities
    ):
        if not isinstance(root, VillainNode):
            raise AiofContractError(
                AiofStatus.ORACLE_MISMATCH, "M29 chance child is not Villain root"
            )
        materialized.append(
            _row_affines(
                probability,
                row,
                root,
                _exact_terminal_lines(request, row),
            )
        )

    output: list[_BucketAffines] = []
    for bucket in sorted({row.villain_bucket_id for row in materialized}):
        bucket_rows = [row for row in materialized if row.villain_bucket_id == bucket]

        def sum_actions(
            attribute: str, legal_actions: tuple[str, ...]
        ) -> tuple[tuple[str, _TripleAffine], ...]:
            return tuple(
                (
                    action,
                    _sum_triples(
                        dict(getattr(row, attribute))[action] for row in bucket_rows
                    ),
                )
                for action in legal_actions
            )

        output.append(
            _BucketAffines(
                villain_bucket_id=bucket,
                root_information_set_id=f"{OOP_FIRST}::{bucket}",
                check_bet_information_set_id=f"{OOP_VS_IP_BET}::{bucket}",
                bet_raise_information_set_id=f"{OOP_VS_IP_RAISE}::{bucket}",
                check_direct=_sum_triples(row.check_direct for row in bucket_rows),
                check_bet_actions=sum_actions(
                    "check_bet_actions", VILLAIN_DECISIONS["vs_ip_bet"]
                ),
                bet_direct=_sum_triples(row.bet_direct for row in bucket_rows),
                bet_raise_actions=sum_actions(
                    "bet_raise_actions", VILLAIN_DECISIONS["vs_ip_raise"]
                ),
            )
        )
    return tuple(output)


def _point_solution(
    actions: tuple[tuple[str, _TripleAffine], ...],
    probabilities: dict[tuple[str, str], Fraction],
    *,
    hero_best_by_action: dict[str, Fraction] | None = None,
) -> _PointSolution:
    values = tuple(
        (action, *triple.value(probabilities)) for action, triple in actions
    )
    villain_max = max(row[2] for row in values)
    best = tuple(row for row in values if row[2] == villain_max)
    hero_worst = min(row[1] for row in best)
    hero_best = max(
        (
            hero_best_by_action[row[0]]
            if hero_best_by_action is not None
            else row[1]
        )
        for row in best
    )
    worst_row = next(row for row in best if row[1] == hero_worst)
    return _PointSolution(
        villain=villain_max,
        hero_worst=hero_worst,
        hero_best=hero_best,
        rake_worst=worst_row[3],
        best_actions=tuple(row[0] for row in best),
        hero_worst_actions=tuple(row[0] for row in best if row[1] == hero_worst),
        values=values,
    )


def _constant_triple(
    hero: Fraction, villain: Fraction, rake: Fraction
) -> _TripleAffine:
    return _TripleAffine(
        _Affine.make(hero), _Affine.make(villain), _Affine.make(rake)
    )


def _bucket_point(
    bucket: _BucketAffines,
    probabilities: dict[tuple[str, str], Fraction],
) -> _BucketPoint:
    check_bet = _point_solution(bucket.check_bet_actions, probabilities)
    bet_raise = _point_solution(bucket.bet_raise_actions, probabilities)
    root_actions = (
        (
            "check",
            bucket.check_direct.plus(
                _constant_triple(
                    check_bet.hero_worst,
                    check_bet.villain,
                    check_bet.rake_worst,
                )
            ),
        ),
        (
            "bet",
            bucket.bet_direct.plus(
                _constant_triple(
                    bet_raise.hero_worst,
                    bet_raise.villain,
                    bet_raise.rake_worst,
                )
            ),
        ),
    )
    root = _point_solution(
        root_actions,
        probabilities,
        hero_best_by_action={
            "check": (
                bucket.check_direct.hero.value(probabilities)
                + check_bet.hero_best
            ),
            "bet": (
                bucket.bet_direct.hero.value(probabilities)
                + bet_raise.hero_best
            ),
        },
    )

    complete_count = 0
    worst_count = 0
    for action in root.best_actions:
        if action == "check":
            complete_count += len(check_bet.best_actions) * len(
                bucket.bet_raise_actions
            )
        else:
            complete_count += len(bet_raise.best_actions) * len(
                bucket.check_bet_actions
            )
    for action in root.hero_worst_actions:
        if action == "check":
            worst_count += len(check_bet.hero_worst_actions) * len(
                bucket.bet_raise_actions
            )
        else:
            worst_count += len(bet_raise.hero_worst_actions) * len(
                bucket.check_bet_actions
            )
    return _BucketPoint(
        bucket,
        check_bet,
        bet_raise,
        root,
        complete_count,
        worst_count,
    )


def _baseline_villain_map_from_points(
    points: tuple[_BucketPoint, ...],
) -> dict[str, dict[str, Fraction]]:
    result: dict[str, dict[str, Fraction]] = {}
    for point in points:
        bucket = point.bucket
        chosen_root = point.root.best_actions[0]
        chosen_check = point.check_bet.best_actions[0]
        chosen_raise = point.bet_raise.best_actions[0]
        result[bucket.root_information_set_id] = {
            action: Fraction(action == chosen_root)
            for action in VILLAIN_DECISIONS["oop_first"]
        }
        result[bucket.check_bet_information_set_id] = {
            action: Fraction(action == chosen_check)
            for action in VILLAIN_DECISIONS["vs_ip_bet"]
        }
        result[bucket.bet_raise_information_set_id] = {
            action: Fraction(action == chosen_raise)
            for action in VILLAIN_DECISIONS["vs_ip_raise"]
        }
    return result


def _fixed_affine(
    buckets: tuple[_BucketAffines, ...],
    villain: dict[str, dict[str, Fraction]],
) -> _TripleAffine:
    total = _ZERO_TRIPLE
    for bucket in buckets:
        check_child = _sum_triples(
            triple.scale(
                villain[bucket.check_bet_information_set_id][action]
            )
            for action, triple in bucket.check_bet_actions
        )
        bet_child = _sum_triples(
            triple.scale(
                villain[bucket.bet_raise_information_set_id][action]
            )
            for action, triple in bucket.bet_raise_actions
        )
        check = bucket.check_direct.plus(check_child).scale(
            villain[bucket.root_information_set_id]["check"]
        )
        bet = bucket.bet_direct.plus(bet_child).scale(
            villain[bucket.root_information_set_id]["bet"]
        )
        total = total.plus(check).plus(bet)
    return total


def _profile_from_exact_villain_map(
    mapping: dict[str, dict[str, Fraction]],
    bucket_ids: tuple[str, ...],
) -> RiverActionProfile:
    prefix_to_decision = {
        OOP_FIRST: "oop_first",
        OOP_VS_IP_BET: "vs_ip_bet",
        OOP_VS_IP_RAISE: "vs_ip_raise",
    }
    rows: list[RiverProfileRow] = []
    for bucket in sorted(bucket_ids):
        for prefix in (OOP_FIRST, OOP_VS_IP_BET, OOP_VS_IP_RAISE):
            info = f"{prefix}::{bucket}"
            rows.append(
                RiverProfileRow(
                    bucket,
                    prefix_to_decision[prefix],
                    tuple(
                        ActionProbability(action, float(probability))
                        for action, probability in mapping[info].items()
                    ),
                )
            )
    return RiverActionProfile(tuple(rows))


def _affine_coefficient_count(
    fixed: _TripleAffine, buckets: tuple[_BucketAffines, ...]
) -> int:
    count = sum(
        len(affine.coefficients)
        for affine in (fixed.hero, fixed.villain, fixed.rake)
    )
    for bucket in buckets:
        triples = (
            (bucket.check_direct, bucket.bet_direct)
            + tuple(value for _, value in bucket.check_bet_actions)
            + tuple(value for _, value in bucket.bet_raise_actions)
        )
        count += sum(
            len(affine.coefficients)
            for triple in triples
            for affine in (triple.hero, triple.villain, triple.rake)
        )
    return count


def _policy_probabilities(
    scenario: HeroBehaviorScenario, policy: ExactBehaviorPolicy
) -> dict[tuple[str, str], Fraction]:
    expected = {
        row.information_set_id: tuple(row.legal_action_ids)
        for row in scenario.information_sets
    }
    if type(policy) is not ExactBehaviorPolicy or type(policy.rows) is not tuple:
        raise UnsupportedScalarOracleDomain("policy is not ExactBehaviorPolicy")
    output: dict[tuple[str, str], Fraction] = {}
    seen: set[str] = set()
    for row in policy.rows:
        if row.information_set_id not in expected or row.information_set_id in seen:
            raise UnsupportedScalarOracleDomain(
                "policy has unknown or duplicate information set"
            )
        seen.add(row.information_set_id)
        actions: dict[str, Fraction] = {}
        for action in row.actions:
            if action.action_id in actions:
                raise UnsupportedScalarOracleDomain("policy has duplicate action")
            try:
                actions[action.action_id] = Fraction(action.probability)
            except (ValueError, ZeroDivisionError) as exc:
                raise UnsupportedScalarOracleDomain(
                    "policy probability is not exact rational text"
                ) from exc
        if (
            set(actions) != set(expected[row.information_set_id])
            or sum(actions.values(), Fraction(0)) != 1
            or any(value < 0 or value > 1 for value in actions.values())
        ):
            raise UnsupportedScalarOracleDomain(
                "policy does not satisfy the complete legal simplex"
            )
        for action, probability in actions.items():
            output[(row.information_set_id, action)] = probability
    if seen != set(expected):
        raise UnsupportedScalarOracleDomain("policy omits information set")
    return output


def _solution_response_row(
    info_set: str,
    bucket: str,
    history: tuple[tuple[str, str], ...],
    solution: _PointSolution,
    appearing: tuple[str, ...],
) -> KnownBoardRealCardHuCompleteResponseRow:
    return KnownBoardRealCardHuCompleteResponseRow(
        information_set_id=info_set,
        villain_bucket_id=bucket,
        villain_history=history,
        conditional_best_actions=solution.best_actions,
        globally_appearing_actions=appearing,
        action_values=tuple(
            (
                action,
                _rational_text(hero),
                _rational_text(villain),
                _rational_text(rake),
            )
            for action, hero, villain, rake in solution.values
        ),
    )


def _cell_intervals(
    scenario: HeroBehaviorScenario, cell: ExactBehaviorCell
) -> dict[tuple[str, str], tuple[Fraction, Fraction]]:
    if type(cell) is not ExactBehaviorCell or type(cell.rows) is not tuple:
        raise UnsupportedScalarOracleDomain("bound request is not ExactBehaviorCell")
    expected = {
        row.information_set_id: tuple(row.legal_action_ids)
        for row in scenario.information_sets
    }
    output: dict[tuple[str, str], tuple[Fraction, Fraction]] = {}
    seen: set[str] = set()
    for row in cell.rows:
        info = row.information_set_id
        if info not in expected or info in seen:
            raise UnsupportedScalarOracleDomain(
                "cell has unknown or duplicate information set"
            )
        seen.add(info)
        raw: dict[str, tuple[Fraction, Fraction]] = {}
        for action in row.actions:
            if action.action_id in raw:
                raise UnsupportedScalarOracleDomain("cell has duplicate action")
            try:
                lower, upper = Fraction(action.lower), Fraction(action.upper)
            except (ValueError, ZeroDivisionError) as exc:
                raise UnsupportedScalarOracleDomain(
                    "cell interval is not exact rational text"
                ) from exc
            raw[action.action_id] = (lower, upper)
        if set(raw) != set(expected[info]):
            raise UnsupportedScalarOracleDomain(
                "cell action order/coverage differs from scenario"
            )
        if any(lower < 0 or upper > 1 or lower > upper for lower, upper in raw.values()):
            raise UnsupportedScalarOracleDomain("cell interval is outside [0,1]")
        if sum(lower for lower, _ in raw.values()) > 1 or sum(
            upper for _, upper in raw.values()
        ) < 1:
            raise UnsupportedScalarOracleDomain(
                "cell does not intersect the exact simplex"
            )
        for action, (lower, upper) in raw.items():
            other = [value for key, value in raw.items() if key != action]
            effective_lower = max(lower, 1 - sum(v[1] for v in other))
            effective_upper = min(upper, 1 - sum(v[0] for v in other))
            if effective_lower > effective_upper:
                raise UnsupportedScalarOracleDomain(
                    "cell has empty simplex-intersection action interval"
                )
            output[(info, action)] = (effective_lower, effective_upper)
    if seen != set(expected):
        raise UnsupportedScalarOracleDomain("cell omits information set")
    return output


def _affine_extreme(
    affine: _Affine,
    scenario: HeroBehaviorScenario,
    intervals: dict[tuple[str, str], tuple[Fraction, Fraction]],
    maximize: bool,
) -> Fraction:
    coefficients = {
        (info, action): value for info, action, value in affine.coefficients
    }
    total = affine.constant
    for row in scenario.information_sets:
        info = row.information_set_id
        actions = row.legal_action_ids
        probabilities = {
            action: intervals[(info, action)][0] for action in actions
        }
        remaining = 1 - sum(probabilities.values(), Fraction(0))
        order = sorted(
            actions,
            key=lambda action: (
                coefficients.get((info, action), Fraction(0)),
                action,
            ),
            reverse=maximize,
        )
        for action in order:
            capacity = intervals[(info, action)][1] - probabilities[action]
            addition = min(remaining, capacity)
            probabilities[action] += addition
            remaining -= addition
        if remaining != 0:
            raise UnsupportedScalarOracleDomain(
                "simplex extreme allocation did not close"
            )
        total += sum(
            (
                coefficients.get((info, action), Fraction(0))
                * probabilities[action]
                for action in actions
            ),
            Fraction(0),
        )
    return total


@dataclass(frozen=True)
class _CellResponseInterval:
    villain_lower: Fraction
    villain_upper: Fraction
    hero_upper: Fraction


def _possible_optimal_actions(
    values: tuple[tuple[str, Fraction, Fraction, Fraction], ...]
) -> tuple[str, ...]:
    # row = action, villain_lower, villain_upper, hero_upper
    threshold = max(row[1] for row in values)
    return tuple(row[0] for row in values if row[2] >= threshold)


def _action_cell_rows(
    actions: tuple[tuple[str, _TripleAffine], ...],
    scenario: HeroBehaviorScenario,
    intervals: dict[tuple[str, str], tuple[Fraction, Fraction]],
) -> tuple[tuple[str, Fraction, Fraction, Fraction], ...]:
    return tuple(
        (
            action,
            _affine_extreme(triple.villain, scenario, intervals, False),
            _affine_extreme(triple.villain, scenario, intervals, True),
            _affine_extreme(triple.hero, scenario, intervals, True),
        )
        for action, triple in actions
    )


def _response_cell_upper(
    buckets: tuple[_BucketAffines, ...],
    scenario: HeroBehaviorScenario,
    intervals: dict[tuple[str, str], tuple[Fraction, Fraction]],
) -> Fraction:
    if all(lower == upper for lower, upper in intervals.values()):
        probabilities = {key: lower for key, (lower, _) in intervals.items()}
        return sum(
            (_bucket_point(bucket, probabilities).root.hero_worst for bucket in buckets),
            Fraction(0),
        )

    total = Fraction(0)
    for bucket in buckets:
        check_rows = _action_cell_rows(
            bucket.check_bet_actions, scenario, intervals
        )
        raise_rows = _action_cell_rows(
            bucket.bet_raise_actions, scenario, intervals
        )
        check_possible = _possible_optimal_actions(check_rows)
        raise_possible = _possible_optimal_actions(raise_rows)
        check_child = _CellResponseInterval(
            villain_lower=max(row[1] for row in check_rows),
            villain_upper=max(row[2] for row in check_rows),
            hero_upper=max(row[3] for row in check_rows if row[0] in check_possible),
        )
        raise_child = _CellResponseInterval(
            villain_lower=max(row[1] for row in raise_rows),
            villain_upper=max(row[2] for row in raise_rows),
            hero_upper=max(row[3] for row in raise_rows if row[0] in raise_possible),
        )
        root_rows = (
            (
                "check",
                _affine_extreme(
                    bucket.check_direct.villain, scenario, intervals, False
                )
                + check_child.villain_lower,
                _affine_extreme(
                    bucket.check_direct.villain, scenario, intervals, True
                )
                + check_child.villain_upper,
                _affine_extreme(
                    bucket.check_direct.hero, scenario, intervals, True
                )
                + check_child.hero_upper,
            ),
            (
                "bet",
                _affine_extreme(
                    bucket.bet_direct.villain, scenario, intervals, False
                )
                + raise_child.villain_lower,
                _affine_extreme(
                    bucket.bet_direct.villain, scenario, intervals, True
                )
                + raise_child.villain_upper,
                _affine_extreme(
                    bucket.bet_direct.hero, scenario, intervals, True
                )
                + raise_child.hero_upper,
            ),
        )
        root_possible = _possible_optimal_actions(root_rows)
        lex_upper = max(
            row[3] for row in root_rows if row[0] in root_possible
        )
        # Exact reconstructed M29 terminals obey H+V+R=0 and R>=0.
        # The selected response therefore also obeys
        # H=-max(V)-R <= -max_a min_C(V_a).
        conservation_upper = -max(row[1] for row in root_rows)
        total += min(lex_upper, conservation_upper)
    return total


class KnownBoardRealCardHuCertifiedScalarOracle:
    """Exact point and sound whole-cell oracle supplied to M36."""

    response_semantics = RESPONSE_SEMANTICS
    bound_contract_version = BOUND_CONTRACT_VERSION

    def __init__(
        self,
        *,
        scenario: HeroBehaviorScenario,
        baseline_policy: ExactBehaviorPolicy,
        fixed_affine: _TripleAffine,
        buckets: tuple[_BucketAffines, ...],
        baseline_hero_ev: Fraction,
        pre_weight: Fraction,
        post_weight: Fraction,
        total_weight: Fraction,
        response_oracle_identity: str,
        objective_identity: str,
        limits: KnownBoardRealCardHuCertifiedGlobalLimits,
        initial_counters: KnownBoardRealCardHuCertifiedOracleWorkCounters,
    ) -> None:
        self.scenario = scenario
        self.baseline_policy = baseline_policy
        self.fixed_affine = fixed_affine
        self.buckets = buckets
        self.baseline_hero_ev = baseline_hero_ev
        self.pre_weight = pre_weight
        self.post_weight = post_weight
        self.total_weight = total_weight
        self.response_oracle_identity = response_oracle_identity
        self.objective_identity = objective_identity
        self.limits = limits
        self._baseline_identity = behavior_policy_identity(scenario, baseline_policy)
        self._records: dict[str, KnownBoardRealCardHuCertifiedPointRecord] = {}
        self._compatible_pairs = initial_counters.compatible_pairs
        self._fixed_board_evaluations = initial_counters.fixed_board_evaluations
        self._tree_nodes = initial_counters.tree_nodes
        self._terminal_records = initial_counters.terminal_records
        self._affine_coefficients = initial_counters.affine_coefficients
        self._point_evaluations = 0
        self._response_rows = 0
        self._response_action_records = 0
        self._bound_records = 0

    @property
    def work_counters(self) -> KnownBoardRealCardHuCertifiedOracleWorkCounters:
        return KnownBoardRealCardHuCertifiedOracleWorkCounters(
            compatible_pairs=self._compatible_pairs,
            fixed_board_evaluations=self._fixed_board_evaluations,
            tree_nodes=self._tree_nodes,
            terminal_records=self._terminal_records,
            affine_coefficients=self._affine_coefficients,
            point_evaluations=self._point_evaluations,
            response_rows=self._response_rows,
            response_action_records=self._response_action_records,
            bound_records=self._bound_records,
        )

    def point_record(
        self, policy_identity: str
    ) -> KnownBoardRealCardHuCertifiedPointRecord:
        return self._records[policy_identity]

    def _response(
        self, probabilities: dict[tuple[str, str], Fraction]
    ) -> KnownBoardRealCardHuCompleteResponse:
        projected_rows = 3 * len(self.buckets)
        projected_actions = 6 * len(self.buckets)
        if self._response_rows + projected_rows > self.limits.max_response_rows:
            raise ScalarOracleResourceLimit(
                "max_response_rows reached before response materialization"
            )
        if (
            self._response_action_records + projected_actions
            > self.limits.max_response_action_records
        ):
            raise ScalarOracleResourceLimit(
                "max_response_action_records reached before materialization"
            )
        points = tuple(_bucket_point(bucket, probabilities) for bucket in self.buckets)
        rows: list[KnownBoardRealCardHuCompleteResponseRow] = []
        complete_count = 1
        worst_count = 1
        for point in points:
            bucket = point.bucket
            root_appearing = point.root.best_actions
            check_appearing = (
                point.check_bet.best_actions
                if point.root.best_actions == ("check",)
                else tuple(action for action, _ in bucket.check_bet_actions)
            )
            raise_appearing = (
                point.bet_raise.best_actions
                if point.root.best_actions == ("bet",)
                else tuple(action for action, _ in bucket.bet_raise_actions)
            )
            rows.extend(
                (
                    _solution_response_row(
                        bucket.root_information_set_id,
                        bucket.villain_bucket_id,
                        (),
                        point.root,
                        root_appearing,
                    ),
                    _solution_response_row(
                        bucket.check_bet_information_set_id,
                        bucket.villain_bucket_id,
                        ((bucket.root_information_set_id, "check"),),
                        point.check_bet,
                        check_appearing,
                    ),
                    _solution_response_row(
                        bucket.bet_raise_information_set_id,
                        bucket.villain_bucket_id,
                        ((bucket.root_information_set_id, "bet"),),
                        point.bet_raise,
                        raise_appearing,
                    ),
                )
            )
            complete_count *= point.complete_count
            worst_count *= point.hero_worst_count
        projection = {
            "response_contract": RESPONSE_CONTRACT,
            "response_oracle_identity": self.response_oracle_identity,
            "hero_worst_value": _rational_text(
                sum((p.root.hero_worst for p in points), Fraction(0))
            ),
            "rows": [row.to_dict() for row in rows],
            "complete_response_record_count": complete_count,
            "hero_worst_witness_count": worst_count,
        }
        response = KnownBoardRealCardHuCompleteResponse(
            hero_worst_value=projection["hero_worst_value"],
            hero_best_value=_rational_text(
                sum((p.root.hero_best for p in points), Fraction(0))
            ),
            villain_max_value=_rational_text(
                sum((p.root.villain for p in points), Fraction(0))
            ),
            house_rake_at_hero_worst=_rational_text(
                sum((p.root.rake_worst for p in points), Fraction(0))
            ),
            rows=tuple(rows),
            action_variation_information_sets=tuple(
                row.information_set_id
                for row in rows
                if len(row.globally_appearing_actions) > 1
            ),
            correspondence_identity=_identity(projection),
            complete_response_record_count=complete_count,
            hero_worst_witness_count=worst_count,
        )
        self._response_rows += projected_rows
        self._response_action_records += projected_actions
        return response

    def evaluate(self, policy: ExactBehaviorPolicy) -> ScalarOracleEvaluation:
        policy_identity = behavior_policy_identity(self.scenario, policy)
        cached = self._records.get(policy_identity)
        if cached is not None:
            return cached.native_evaluation
        if self._point_evaluations + 1 > self.limits.max_oracle_point_evaluations:
            raise ScalarOracleResourceLimit(
                "max_oracle_point_evaluations reached before point evaluation"
            )
        probabilities = _policy_probabilities(self.scenario, policy)
        fixed = self.fixed_affine.hero.value(probabilities)
        response = self._response(probabilities)
        post = Fraction(response.hero_worst_value)
        baseline_total = self.total_weight * self.baseline_hero_ev
        candidate_total = (
            baseline_total
            if policy_identity == self._baseline_identity
            else self.pre_weight * fixed + self.post_weight * post
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
            complete_response_record_count=response.complete_response_record_count,
            hero_worst_witness_count=response.hero_worst_witness_count,
        )
        record = KnownBoardRealCardHuCertifiedPointRecord(
            policy_identity,
            _rational_text(fixed),
            _rational_text(post),
            _rational_text(baseline_total),
            _rational_text(candidate_total),
            _rational_text(uplift),
            response,
            native,
        )
        self._records[policy_identity] = record
        self._point_evaluations += 1
        return native

    def upper_bound(self, cell: ExactBehaviorCell) -> ScalarOracleBound:
        if self._bound_records + 1 > self.limits.max_oracle_bound_records:
            raise ScalarOracleResourceLimit(
                "max_oracle_bound_records reached before bound construction"
            )
        intervals = _cell_intervals(self.scenario, cell)
        fixed_upper = _affine_extreme(
            self.fixed_affine.hero, self.scenario, intervals, True
        )
        response_upper = _response_cell_upper(
            self.buckets, self.scenario, intervals
        )
        baseline_total = self.total_weight * self.baseline_hero_ev
        locked_upper = (
            self.pre_weight * fixed_upper
            + self.post_weight * response_upper
            - baseline_total
        )
        upper = max(Fraction(0), locked_upper)
        cell_identity = behavior_cell_identity(self.scenario, cell)
        upper_text = _rational_text(upper)
        bound_identity = scalar_oracle_bound_identity(
            cell_identity=cell_identity,
            response_oracle_identity=self.response_oracle_identity,
            objective_identity=self.objective_identity,
            upper_bound=upper_text,
            candidate_policy_identity=None,
        )
        self._bound_records += 1
        return ScalarOracleBound(
            cell_identity=cell_identity,
            response_oracle_identity=self.response_oracle_identity,
            objective_identity=self.objective_identity,
            upper_bound=upper_text,
            bound_identity=bound_identity,
            candidate_policy=None,
            valid_for_entire_cell=True,
            bound_contract_version=BOUND_CONTRACT_VERSION,
        )


def _shadow_m29_request(
    request: KnownBoardRealCardHuCertifiedGlobalRequest,
    discount: Fraction,
) -> KnownBoardRealCardHuRiverRequest:
    return KnownBoardRealCardHuRiverRequest(
        board=request.board,
        hero_range=request.hero_range,
        villain_range=request.villain_range,
        baseline_hero_profile=request.baseline_hero_profile,
        dead_cards=request.dead_cards,
        hero_combo_to_bucket=request.hero_combo_to_bucket,
        villain_combo_to_bucket=request.villain_combo_to_bucket,
        baseline_villain_profile=request.baseline_villain_profile,
        initial_commitment_hero=request.initial_commitment_hero,
        initial_commitment_villain=request.initial_commitment_villain,
        rake_rate=request.rake_rate,
        rake_cap=request.rake_cap,
        oop_bet_size=request.oop_bet_size,
        ip_bet_after_check_size=request.ip_bet_after_check_size,
        ip_raise_to_size=request.ip_raise_to_size,
        shift_amounts=(),
        max_simultaneous_info_sets=1,
        horizon=request.horizon,
        discount=float(discount),
        tolerance=0.0,
        minimum_total_uplift=0.0,
        aiof_limits=request.aiof_limits,
        limits=request.river_limits,
        max_horizon=request.integration_limits.max_horizon,
        expected_baseline_identity=None,
    )


def _canonical_villain_profile(
    shadow: KnownBoardRealCardHuRiverRequest,
    bucket_ids: tuple[str, ...],
) -> tuple[RiverActionProfile | None, dict[str, dict[str, Fraction]] | None]:
    if shadow.baseline_villain_profile is None:
        return None, None
    canonical, _ = _canonical_profile(
        shadow.baseline_villain_profile, bucket_ids, "villain", 0.0
    )
    return canonical, _exact_profile_map(canonical)


def _validate_request_top(
    request: object,
) -> tuple[
    KnownBoardRealCardHuCertifiedGlobalRequest,
    KnownBoardRealCardHuCertifiedGlobalLimits,
    KnownBoardRealCardHuCertifiedGlobalPins,
    Fraction,
    Fraction,
    Fraction,
]:
    if type(request) is not KnownBoardRealCardHuCertifiedGlobalRequest:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            "request must be KnownBoardRealCardHuCertifiedGlobalRequest",
        )
    limits = _validate_limits(request.integration_limits)
    pins = _validate_pins(request.pins)
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
    horizon = _validate_positive_int(
        request.horizon, "horizon", limits.max_horizon
    )
    if (
        type(request.adaptation_opportunity) is not int
        or not 1 <= request.adaptation_opportunity <= horizon + 1
    ):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            "adaptation_opportunity must be in 1..horizon+1",
        )
    discount = _parse_rational(request.discount, "discount")
    if not 0 < discount <= 1:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, "discount must satisfy 0 < discount <= 1"
        )
    response_tolerance = _parse_rational(
        request.response_tolerance, "response_tolerance", nonnegative=True
    )
    if response_tolerance != 0:
        raise AiofContractError(
            AiofStatus.UNSUPPORTED_MODEL,
            "nonzero response tolerance cannot support an exact certificate",
        )
    _parse_rational(
        request.absolute_gap_tolerance,
        "absolute_gap_tolerance",
        nonnegative=True,
    )
    _parse_rational(
        request.relative_gap_tolerance,
        "relative_gap_tolerance",
        nonnegative=True,
    )
    pre, post, total = _discount_weights(
        horizon, request.adaptation_opportunity, discount
    )
    return request, limits, pins, pre, post, total


def prepare_known_board_real_card_hu_certified_global_oracle(
    request: KnownBoardRealCardHuCertifiedGlobalRequest,
) -> KnownBoardRealCardHuCertifiedGlobalPreparation:
    """Prepare the canonical M29 surface and exact M38 scalar oracle.

    Invalid, stale, unsupported, or over-cap inputs raise.  The fail-closed
    public wrapper is :func:`analyze_known_board_real_card_hu_certified_global`.
    """

    request, limits, pins, pre_weight, post_weight, total_weight = (
        _validate_request_top(request)
    )
    discount = _parse_rational(request.discount, "discount")
    shadow = _shadow_m29_request(request, discount)
    (
        shadow,
        board,
        dead,
        unavailable,
        _,
        _,
    ) = _validate_request(shadow)

    projection = _project_joint_support(shadow, board, unavailable)
    hero_combos = tuple(item.combo for item in projection.hero_final.combos)
    villain_combos = tuple(item.combo for item in projection.villain_final.combos)
    hero_mapping = _canonical_mapping(
        shadow.hero_combo_to_bucket,
        hero_combos,
        "hero",
        shadow.limits.max_buckets_per_seat,
    )
    villain_mapping = _canonical_mapping(
        shadow.villain_combo_to_bucket,
        villain_combos,
        "villain",
        shadow.limits.max_buckets_per_seat,
    )
    hero_profile, _ = _canonical_profile(
        shadow.baseline_hero_profile,
        hero_mapping.bucket_ids,
        "hero",
        0.0,
    )
    _, supplied_villain_map = _canonical_villain_profile(
        shadow, villain_mapping.bucket_ids
    )

    pair_count = projection.provenance.compatible_pair_count
    projected_tree_nodes = 12 * pair_count + 1
    projected_terminals = 7 * pair_count
    projected_preparation_records = (
        pair_count + projected_tree_nodes + projected_terminals
    )
    if projected_preparation_records > limits.max_preparation_records:
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            "max_preparation_records exceeded before joint/tree materialization",
        )
    projected_coefficients = 60 * pair_count + 12 * len(
        villain_mapping.bucket_ids
    )
    if projected_coefficients > limits.max_affine_coefficients:
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            "max_affine_coefficients exceeded before affine materialization",
        )
    if 3 * len(villain_mapping.bucket_ids) > limits.max_response_rows:
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            "max_response_rows exceeded before response structure materialization",
        )
    if 6 * len(villain_mapping.bucket_ids) > limits.max_response_action_records:
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            "max_response_action_records exceeded before response materialization",
        )

    prepared = _prepared_ranges_checked(shadow, unavailable, projection)
    joint_rows = _materialize_joint_rows(
        projection, board, hero_mapping, villain_mapping
    )
    tree = _build_tree(shadow, joint_rows)
    actual_nodes = sum(1 for _ in iter_nodes(tree.root))
    if actual_nodes != projected_tree_nodes:
        raise AiofContractError(
            AiofStatus.ORACLE_MISMATCH, "M29 tree-node projection changed"
        )
    exact_joint_probabilities = _exact_joint_probabilities(
        shadow, projection, joint_rows
    )
    buckets = _aggregate_bucket_affines(
        tree, joint_rows, shadow, exact_joint_probabilities
    )
    hero_info_sets = collect_hero_info_sets(tree)
    villain_info_sets = collect_villain_info_sets(tree)
    if len(hero_info_sets) != 2 * len(hero_mapping.bucket_ids) or len(
        villain_info_sets
    ) != 3 * len(villain_mapping.bucket_ids):
        raise AiofContractError(
            AiofStatus.ORACLE_MISMATCH, "M29 information-set topology changed"
        )

    board_identity = _board_identity(board, dead)
    prepared_joint_identity = _prepared_joint_identity(
        board_identity, projection, prepared, joint_rows
    )
    profile_mapping_identity = _mapping_identity(hero_mapping, villain_mapping)
    tree_identity = _tree_identity(
        shadow, prepared_joint_identity, profile_mapping_identity
    )

    scenario_projection = {
        "domain_contract": DOMAIN_CONTRACT,
        "position": POSITION,
        "prepared_joint_identity": prepared_joint_identity,
        "profile_mapping_identity": profile_mapping_identity,
        "tree_identity": tree_identity,
        "information_sets": [
            {
                "information_set_id": info,
                "legal_action_ids": list(hero_info_sets[info]),
            }
            for info in sorted(hero_info_sets)
        ],
    }
    scenario_identity = _identity(scenario_projection)
    scenario, baseline_policy = _scenario_and_policy(
        hero_info_sets, hero_profile, scenario_identity
    )
    baseline_probabilities = _policy_probabilities(scenario, baseline_policy)

    if supplied_villain_map is None:
        baseline_response_points = tuple(
            _bucket_point(bucket, baseline_probabilities) for bucket in buckets
        )
        baseline_villain_map = _baseline_villain_map_from_points(
            baseline_response_points
        )
        baseline_source = "fresh_exact_response_deterministic_representative"
        canonical_villain_profile = _profile_from_exact_villain_map(
            baseline_villain_map, villain_mapping.bucket_ids
        )
    else:
        baseline_villain_map = supplied_villain_map
        baseline_source = "supplied_complete_profile"
        canonical_villain_profile, _ = _canonical_profile(
            shadow.baseline_villain_profile,
            villain_mapping.bucket_ids,
            "villain",
            0.0,
        )

    fixed_affine = _fixed_affine(buckets, baseline_villain_map)
    actual_coefficients = _affine_coefficient_count(fixed_affine, buckets)
    if actual_coefficients > limits.max_affine_coefficients:
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            "max_affine_coefficients exceeded before retaining affines",
        )
    baseline_hero_ev = fixed_affine.hero.value(baseline_probabilities)
    baseline_identity = _identity(
        {
            "algorithm": "m38-exact-fixed-baseline-sha256-v1",
            "tree_identity": tree_identity,
            "source": baseline_source,
            "hero_profile": hero_profile.to_dict(),
            "villain_profile": canonical_villain_profile.to_dict(),
            "baseline_hero_ev": _rational_text(baseline_hero_ev),
        }
    )
    response_oracle_identity = _identity(
        {
            "response_contract": RESPONSE_CONTRACT,
            "tree_identity": tree_identity,
            "rational_lift": RATIONAL_LIFT,
            "factorized_buckets": [bucket.projection() for bucket in buckets],
            "response_tolerance": "0",
        }
    )
    objective_identity = _identity(
        {
            "objective_contract": OBJECTIVE_CONTRACT,
            "m36_objective_semantics": OBJECTIVE_SEMANTICS,
            "scenario_identity": scenario_identity,
            "baseline_identity": baseline_identity,
            "baseline_policy_identity": behavior_policy_identity(
                scenario, baseline_policy
            ),
            "response_oracle_identity": response_oracle_identity,
            "b": _rational_text(baseline_hero_ev),
            "horizon": request.horizon,
            "adaptation_opportunity": request.adaptation_opportunity,
            "discount": request.discount,
            "pre_weight": _rational_text(pre_weight),
            "post_weight": _rational_text(post_weight),
            "total_weight": _rational_text(total_weight),
            "baseline_policy_is_no_commitment_comparison": True,
        }
    )
    request_identity = _identity(_request_projection(request))
    analysis_identity = _identity(
        {
            "contract_version": KNOWN_BOARD_REAL_CARD_HU_CERTIFIED_GLOBAL_CONTRACT_VERSION,
            "request_identity": request_identity,
            "prepared_joint_identity": prepared_joint_identity,
            "baseline_identity": baseline_identity,
            "scenario_identity": scenario_identity,
            "response_oracle_identity": response_oracle_identity,
            "objective_identity": objective_identity,
        }
    )
    for expected, actual, name in (
        (pins.request_identity, request_identity, "pins.request_identity"),
        (pins.board_identity, board_identity, "pins.board_identity"),
        (
            pins.prepared_joint_identity,
            prepared_joint_identity,
            "pins.prepared_joint_identity",
        ),
        (
            pins.profile_mapping_identity,
            profile_mapping_identity,
            "pins.profile_mapping_identity",
        ),
        (pins.tree_identity, tree_identity, "pins.tree_identity"),
        (pins.baseline_identity, baseline_identity, "pins.baseline_identity"),
        (pins.scenario_identity, scenario_identity, "pins.scenario_identity"),
        (
            pins.response_oracle_identity,
            response_oracle_identity,
            "pins.response_oracle_identity",
        ),
        (pins.objective_identity, objective_identity, "pins.objective_identity"),
        (pins.analysis_identity, analysis_identity, "pins.analysis_identity"),
    ):
        _check_pin(expected, actual, name)

    counters = KnownBoardRealCardHuCertifiedOracleWorkCounters(
        compatible_pairs=pair_count,
        fixed_board_evaluations=pair_count,
        tree_nodes=actual_nodes,
        terminal_records=projected_terminals,
        affine_coefficients=actual_coefficients,
    )
    oracle = KnownBoardRealCardHuCertifiedScalarOracle(
        scenario=scenario,
        baseline_policy=baseline_policy,
        fixed_affine=fixed_affine,
        buckets=buckets,
        baseline_hero_ev=baseline_hero_ev,
        pre_weight=pre_weight,
        post_weight=post_weight,
        total_weight=total_weight,
        response_oracle_identity=response_oracle_identity,
        objective_identity=objective_identity,
        limits=limits,
        initial_counters=counters,
    )
    return KnownBoardRealCardHuCertifiedGlobalPreparation(
        scenario,
        baseline_policy,
        board,
        dead,
        joint_rows,
        tuple(_rational_text(value) for value in exact_joint_probabilities),
        projection.provenance,
        hero_mapping,
        villain_mapping,
        board_identity,
        prepared_joint_identity,
        profile_mapping_identity,
        tree_identity,
        baseline_identity,
        request_identity,
        response_oracle_identity,
        objective_identity,
        analysis_identity,
        oracle,
    )


@dataclass(frozen=True)
class _JsonObjectProjection:
    items: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class _DeferredJsonProjection:
    factory: Callable[[], Any]


def _payload_output_projection(
    payload: KnownBoardRealCardHuCertifiedGlobalPayload,
) -> _JsonObjectProjection:
    native_payload = payload.native_optimizer_result.payload
    return _JsonObjectProjection(
        (
            (
                "contract_version",
                KNOWN_BOARD_REAL_CARD_HU_CERTIFIED_GLOBAL_CONTRACT_VERSION,
            ),
            ("schema_version", SCHEMA_VERSION),
            ("algorithm_version", ALGORITHM_VERSION),
            ("domain_contract", DOMAIN_CONTRACT),
            ("objective_contract", OBJECTIVE_CONTRACT),
            ("response_contract", RESPONSE_CONTRACT),
            ("bound_contract", BOUND_CONTRACT),
            ("certificate_claim", CLAIM_SCOPE),
            ("accounting", ACCOUNTING),
            ("unit", UNIT),
            ("position", POSITION),
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
                            "river",
                            _DeferredJsonProjection(
                                payload.request.river_limits.to_dict
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
                                lambda: asdict(payload.request.optimizer_limits)
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
                    else _DeferredJsonProjection(payload.selected_point.to_dict)
                ),
            ),
            (
                "no_beneficial_commitment",
                (
                    None
                    if native_payload is None
                    else native_payload.no_beneficial_commitment
                ),
            ),
            (
                "native_optimizer_result",
                _DeferredJsonProjection(payload.native_optimizer_result.to_dict),
            ),
            (
                "oracle_work_counters",
                _DeferredJsonProjection(payload.oracle_work_counters.to_dict),
            ),
        )
    )


def _success_output_projection(
    result: KnownBoardRealCardHuCertifiedGlobalResult,
) -> _JsonObjectProjection:
    if result.payload is None or result.error is not None:
        raise RuntimeError("success projection requires successful result")
    return _JsonObjectProjection(
        (
            ("status", result.status),
            ("payload", _payload_output_projection(result.payload)),
            ("error", None),
            (
                "optimizer_work_counters",
                _DeferredJsonProjection(result.optimizer_work_counters.to_dict),
            ),
            (
                "oracle_work_counters",
                _DeferredJsonProjection(result.oracle_work_counters.to_dict),
            ),
            ("partial_result", False),
        )
    )


def _preflight_success_output(
    result: KnownBoardRealCardHuCertifiedGlobalResult,
    *,
    max_records: int,
    max_bytes: int,
) -> tuple[int, int]:
    """Traverse the final wrapper lazily and stop at the first exceeded cap."""

    record_count = 0
    byte_count = 0

    def add(fragment: str) -> None:
        nonlocal byte_count
        byte_count += len(fragment.encode("utf-8"))
        if byte_count > max_bytes:
            raise _OutputLimitReached(
                "max_output_bytes exceeded before success output materialization"
            )

    def visit(value: Any) -> None:
        nonlocal record_count
        if isinstance(value, _DeferredJsonProjection):
            visit(value.factory())
            return
        record_count += 1
        if record_count > max_records:
            raise _OutputLimitReached(
                "max_output_records exceeded before success output materialization"
            )
        if isinstance(value, _JsonObjectProjection):
            items = sorted(value.items, key=lambda item: item[0])
            add("{")
            for index, (key, child) in enumerate(items):
                if index:
                    add(",")
                add(
                    json.dumps(
                        key,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                )
                add(":")
                visit(child)
            add("}")
            return
        if isinstance(value, dict):
            add("{")
            for index, key in enumerate(sorted(value)):
                if type(key) is not str:
                    raise TypeError("public JSON keys must be strings")
                if index:
                    add(",")
                add(
                    json.dumps(
                        key,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                )
                add(":")
                visit(value[key])
            add("}")
            return
        if isinstance(value, (tuple, list)):
            add("[")
            for index, child in enumerate(value):
                if index:
                    add(",")
                visit(child)
            add("]")
            return
        add(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        )

    visit(_success_output_projection(result))
    return record_count, byte_count


def _materialize_output_projection(value: Any) -> Any:
    """Reviewer/test helper: independently expand one lazy projection."""

    if isinstance(value, _DeferredJsonProjection):
        return _materialize_output_projection(value.factory())
    if isinstance(value, _JsonObjectProjection):
        output: dict[str, Any] = {}
        for key, child in value.items:
            if key in output:
                raise ValueError(f"duplicate projection key {key}")
            output[key] = _materialize_output_projection(child)
        return output
    if isinstance(value, dict):
        return {
            key: _materialize_output_projection(child)
            for key, child in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_materialize_output_projection(child) for child in value]
    return value


def _empty_optimizer_counters() -> CertifiedGlobalWorkCounters:
    return CertifiedGlobalWorkCounters()


def _empty_oracle_counters() -> KnownBoardRealCardHuCertifiedOracleWorkCounters:
    return KnownBoardRealCardHuCertifiedOracleWorkCounters()


def _clean_message(value: str, fallback: str) -> str:
    cleaned = " ".join(
        (value or fallback).replace("\r", " ").replace("\n", " ").split()
    )
    return cleaned[:512] or fallback


def _status_from_aiof(status: AiofStatus) -> str:
    if status is AiofStatus.CAP_EXCEEDED:
        return LIMIT_REACHED_NO_CERTIFICATE
    if status is AiofStatus.UNSUPPORTED_MODEL:
        return UNSUPPORTED_DOMAIN
    if status in (AiofStatus.NUMERIC_FAILURE, AiofStatus.ACCOUNTING_MISMATCH):
        return NUMERIC_FAILURE
    if status is AiofStatus.ORACLE_MISMATCH:
        return ORACLE_FAILURE
    return INVALID_INPUT


def analyze_known_board_real_card_hu_certified_global(
    request: KnownBoardRealCardHuCertifiedGlobalRequest,
) -> KnownBoardRealCardHuCertifiedGlobalResult:
    """Run M38; success is certified and every failure has a null payload."""

    preparation: KnownBoardRealCardHuCertifiedGlobalPreparation | None = None
    native: CertifiedGlobalOptimizerResult | None = None
    try:
        preparation = prepare_known_board_real_card_hu_certified_global_oracle(
            request
        )
        native = optimize_certified_global_hero_commitment(
            preparation.scenario,
            preparation.baseline_policy,
            preparation.oracle,
            absolute_gap_tolerance=request.absolute_gap_tolerance,
            relative_gap_tolerance=request.relative_gap_tolerance,
            limits=request.optimizer_limits,
            pins=request.optimizer_pins,
        )
        if native.status not in (CERTIFIED_GLOBAL, CERTIFIED_EPSILON_GLOBAL):
            return KnownBoardRealCardHuCertifiedGlobalResult(
                native.status,
                None,
                KnownBoardRealCardHuCertifiedGlobalError(
                    (
                        native.error.phase
                        if native.error is not None
                        else "optimizer"
                    ),
                    (
                        native.error.message
                        if native.error is not None
                        else "optimizer returned no certificate"
                    ),
                    (
                        native.error.cause_status
                        if native.error is not None
                        else None
                    ),
                ),
                native.work_counters,
                preparation.oracle.work_counters,
                False,
            )
        if native.payload is None or native.error is not None:
            raise RuntimeError("malformed successful M36 result")
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
        payload = KnownBoardRealCardHuCertifiedGlobalPayload(
            preparation,
            baseline_point,
            selected_point,
            native,
            preparation.oracle.work_counters,
            request,
        )
        result = KnownBoardRealCardHuCertifiedGlobalResult(
            native.status,
            payload,
            None,
            native.work_counters,
            preparation.oracle.work_counters,
            False,
        )
        _preflight_success_output(
            result,
            max_records=request.integration_limits.max_output_records,
            max_bytes=request.integration_limits.max_output_bytes,
        )
        return result
    except _StaleInput as exc:
        return KnownBoardRealCardHuCertifiedGlobalResult(
            STALE_INPUT,
            None,
            KnownBoardRealCardHuCertifiedGlobalError(
                "pins", _clean_message(str(exc), "stale input")
            ),
            _empty_optimizer_counters(),
            (
                _empty_oracle_counters()
                if preparation is None
                else preparation.oracle.work_counters
            ),
            False,
        )
    except _OutputLimitReached as exc:
        return KnownBoardRealCardHuCertifiedGlobalResult(
            LIMIT_REACHED_NO_CERTIFICATE,
            None,
            KnownBoardRealCardHuCertifiedGlobalError(
                "output", _clean_message(str(exc), "output resource limit")
            ),
            (
                _empty_optimizer_counters()
                if native is None
                else native.work_counters
            ),
            (
                _empty_oracle_counters()
                if preparation is None
                else preparation.oracle.work_counters
            ),
            False,
        )
    except AiofContractError as exc:
        return KnownBoardRealCardHuCertifiedGlobalResult(
            _status_from_aiof(exc.status),
            None,
            KnownBoardRealCardHuCertifiedGlobalError(
                "preparation",
                _clean_message(str(exc), exc.status.value),
                exc.status.value,
            ),
            _empty_optimizer_counters(),
            (
                _empty_oracle_counters()
                if preparation is None
                else preparation.oracle.work_counters
            ),
            False,
        )
    except UnsupportedScalarOracleDomain as exc:
        return KnownBoardRealCardHuCertifiedGlobalResult(
            UNSUPPORTED_DOMAIN,
            None,
            KnownBoardRealCardHuCertifiedGlobalError(
                "oracle", _clean_message(str(exc), "unsupported oracle domain")
            ),
            _empty_optimizer_counters(),
            (
                _empty_oracle_counters()
                if preparation is None
                else preparation.oracle.work_counters
            ),
            False,
        )
    except ScalarOracleResourceLimit as exc:
        return KnownBoardRealCardHuCertifiedGlobalResult(
            LIMIT_REACHED_NO_CERTIFICATE,
            None,
            KnownBoardRealCardHuCertifiedGlobalError(
                "oracle", _clean_message(str(exc), "oracle resource limit")
            ),
            _empty_optimizer_counters(),
            (
                _empty_oracle_counters()
                if preparation is None
                else preparation.oracle.work_counters
            ),
            False,
        )
    except Exception:
        return KnownBoardRealCardHuCertifiedGlobalResult(
            ORACLE_FAILURE,
            None,
            KnownBoardRealCardHuCertifiedGlobalError(
                "internal", "unexpected M38 integration failure"
            ),
            (
                _empty_optimizer_counters()
                if native is None
                else native.work_counters
            ),
            (
                _empty_oracle_counters()
                if preparation is None
                else preparation.oracle.work_counters
            ),
            False,
        )


def exact_known_board_real_card_hu_certified_global_json(
    result: KnownBoardRealCardHuCertifiedGlobalResult,
) -> str:
    """Return deterministic compact strict one-line public JSON."""

    if type(result) is not KnownBoardRealCardHuCertifiedGlobalResult:
        raise TypeError(
            "result must be KnownBoardRealCardHuCertifiedGlobalResult"
        )
    return _canonical_json_bytes(result.to_dict()).decode("utf-8")
