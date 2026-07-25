"""Certified global Hero-policy optimization for exact three-player rivers.

This module integrates the M36 branch-and-bound core with both the abstract
M31 river/rake surface and the M35 known-board real-card three-player adapter.
The search domain is always the complete product of scenario-derived Hero
action simplexes.  The point oracle uses a fresh complete M30 two-opponent
non-cooperative response correspondence.

The certificate concerns one identified bounded scalar repeated-value
objective only.  It is not an equilibrium, profitability, strategy-advice, or
solver-scale claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from fractions import Fraction
import hashlib
import json
import math
import re
from typing import Any, Callable, Mapping

from . import certified_global_optimizer as _m36
from . import known_board_real_card_three_player_river as _m35
from . import three_player_response as _m30
from . import three_player_river_rake as _m31
from .aiof_cards import AiofContractError, AiofStatus


CONTRACT_VERSION = (
    "m39-abstract-real-card-three-player-certified-global-integration-v1"
)
ALGORITHM_VERSION = "exact-m31-point-terminal-envelope-m36-bnb-v1"
SCHEMA_VERSION = "m39-public-result-schema-v1"
DOMAIN_CONTRACT = "m31-full-surviving-hero-action-simplex-product-v1"
OBJECTIVE_CONTRACT = "m32-b-a-l-fixed-adaptation-total-uplift-v1"
RESPONSE_CONTRACT = "m30-complete-two-opponent-noncooperative-hero-worst-v1"
BOUND_CONTRACT = "m31-exact-terminal-envelope-whole-cell-bound-v1"
CLAIM_SCOPE = "identified-bounded-scalar-objective-global-maximum-only-v1"
RATIONAL_LIFT = "lossless-public-binary64-as-integer-ratio-v1"

ABSTRACT = "abstract_m31"
KNOWN_BOARD_REAL_CARD = "known_board_real_card_m35"

MAX_POINT_EVALUATIONS = 4_096
MAX_BOUND_EVALUATIONS = 1_048_575
MAX_AGGREGATE_RESPONSE_RECORDS = 4_000_000
MAX_AGGREGATE_JOINT_PROFILES = 147_456
MAX_AGGREGATE_TERMINAL_EVALUATIONS = 37_748_736
MAX_TERMINAL_RECORDS = 256
MAX_HORIZON = 100_000
MAX_OUTPUT_RECORDS = 2_000_000
MAX_OUTPUT_BYTES = 256_000_000
MAX_RATIONAL_BITS = 4_096

_IDENTITY_RE = re.compile(r"(?:sha256:)?[0-9a-f]{64}\Z")


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


def _rational_text(value: Fraction | int) -> str:
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _parse_exact(
    value: object,
    name: str,
    *,
    nonnegative: bool = False,
    maximum_bits: int = MAX_RATIONAL_BITS,
) -> Fraction:
    if type(value) is int:
        parsed = Fraction(value)
    elif type(value) is float:
        if not math.isfinite(value):
            raise _M39Failure(_m36.INVALID_INPUT, name, "value must be finite")
        numerator, denominator = value.as_integer_ratio()
        parsed = Fraction(numerator, denominator)
    elif type(value) is str:
        if not value or value != value.strip():
            raise _M39Failure(
                _m36.INVALID_INPUT, name, "rational text must be canonical"
            )
        try:
            parsed = Fraction(value)
        except (ValueError, ZeroDivisionError):
            raise _M39Failure(
                _m36.INVALID_INPUT, name, "invalid exact rational"
            ) from None
        if _rational_text(parsed) != value:
            raise _M39Failure(
                _m36.INVALID_INPUT, name, "rational text must be reduced"
            )
    else:
        raise _M39Failure(
            _m36.INVALID_INPUT,
            name,
            "value must be an exact integer/rational string or finite binary64",
        )
    _check_fraction(parsed, name, maximum_bits)
    if nonnegative and parsed < 0:
        raise _M39Failure(_m36.INVALID_INPUT, name, "value must be nonnegative")
    return parsed


def _check_fraction(value: Fraction, phase: str, maximum_bits: int) -> Fraction:
    if (
        value.numerator.bit_length() > maximum_bits
        or value.denominator.bit_length() > maximum_bits
    ):
        raise _M39Failure(
            _m36.NUMERIC_FAILURE,
            phase,
            "exact rational growth exceeds the configured bit cap",
        )
    return value


def _checked_add(
    left: Fraction, right: Fraction, phase: str, maximum_bits: int
) -> Fraction:
    return _check_fraction(left + right, phase, maximum_bits)


def _checked_mul(
    left: Fraction, right: Fraction, phase: str, maximum_bits: int
) -> Fraction:
    return _check_fraction(left * right, phase, maximum_bits)


def _require_positive_int(value: object, name: str, ceiling: int) -> int:
    if type(value) is not int or value <= 0 or value > ceiling:
        raise _M39Failure(
            _m36.INVALID_INPUT,
            name,
            f"value must be a positive int no greater than {ceiling}",
        )
    return value


def _normalized_identity(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or _IDENTITY_RE.fullmatch(value) is None:
        raise _M39Failure(
            _m36.INVALID_INPUT,
            "pins",
            f"{name} must be raw or sha256-prefixed lowercase SHA-256",
        )
    return value.removeprefix("sha256:")


def _check_pin(expected: str | None, actual: str, name: str) -> None:
    if expected is not None and expected != actual.removeprefix("sha256:"):
        raise _M39Failure(
            _m36.STALE_INPUT, "pins", f"pins.{name} mismatch"
        )


@dataclass(frozen=True)
class ThreePlayerCertifiedRepeatedConfig:
    """Exact fixed-adaptation repeated-value configuration."""

    horizon: int = 1
    adaptation_opportunity: int = 1
    discount: object = 1.0


@dataclass(frozen=True)
class ThreePlayerCertifiedGlobalLimits:
    """Caller-lowerable M39 caps with immutable hard ceilings."""

    max_point_evaluations: int = 64
    max_bound_evaluations: int = 16_383
    max_aggregate_response_records: int = 500_000
    max_aggregate_joint_profiles: int = 2_304
    max_aggregate_terminal_evaluations: int = 589_824
    max_terminal_records: int = 128
    max_horizon: int = 1_000
    max_rational_numerator_bits: int = 512
    max_rational_denominator_bits: int = 512
    max_output_records: int = 500_000
    max_output_bytes: int = 64_000_000

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ThreePlayerCertifiedGlobalPins:
    """Optional M39 identities; raw and ``sha256:`` forms are equivalent."""

    request_identity: str | None = None
    scenario_identity: str | None = None
    tree_structure_identity: str | None = None
    baseline_identity: str | None = None
    domain_identity: str | None = None
    response_oracle_identity: str | None = None
    objective_identity: str | None = None
    analysis_identity: str | None = None


@dataclass(frozen=True)
class AbstractThreePlayerCertifiedGlobalRequest:
    """One abstract M31 three-player certified-global request."""

    scenario: _m31.ThreePlayerRiverRakeScenario
    baseline_fixed_hero_policy: _m31.ExactBehaviorPolicy
    initial_profile: _m31.OpponentInitialProfile
    attestation: _m31.PerfectRecallAttestation
    repeated: ThreePlayerCertifiedRepeatedConfig = (
        ThreePlayerCertifiedRepeatedConfig()
    )
    absolute_gap_tolerance: object = "0"
    relative_gap_tolerance: object = "0"
    integration_limits: ThreePlayerCertifiedGlobalLimits = (
        ThreePlayerCertifiedGlobalLimits()
    )
    optimizer_limits: _m36.CertifiedGlobalOptimizerLimits = (
        _m36.CertifiedGlobalOptimizerLimits()
    )
    m31_limits: _m31.RiverRakeLimits = _m31.RiverRakeLimits()
    m30_limits: _m30.ExactResponseLimits = _m30.ExactResponseLimits()
    m31_pins: _m31.RiverRakeIdentityPins = _m31.RiverRakeIdentityPins()
    m30_pins: _m30.ResponseIdentityPins = _m30.ResponseIdentityPins()
    optimizer_pins: _m36.CertifiedGlobalOptimizerPins = (
        _m36.CertifiedGlobalOptimizerPins()
    )
    pins: ThreePlayerCertifiedGlobalPins = ThreePlayerCertifiedGlobalPins()


@dataclass(frozen=True)
class KnownBoardRealCardThreePlayerCertifiedGlobalRequest:
    """One M35-derived known-board real-card certified-global request.

    ``source`` supplies the M35 card/range/profile/tree contract.  Its finite
    M32 generation and selector fields are deliberately not used by M39 and do
    not constrain the M39 domain.
    """

    source: _m35.KnownBoardRealCardThreePlayerRequest
    adaptation_opportunity: int = 1
    absolute_gap_tolerance: object = "0"
    relative_gap_tolerance: object = "0"
    integration_limits: ThreePlayerCertifiedGlobalLimits = (
        ThreePlayerCertifiedGlobalLimits()
    )
    optimizer_limits: _m36.CertifiedGlobalOptimizerLimits = (
        _m36.CertifiedGlobalOptimizerLimits()
    )
    optimizer_pins: _m36.CertifiedGlobalOptimizerPins = (
        _m36.CertifiedGlobalOptimizerPins()
    )
    pins: ThreePlayerCertifiedGlobalPins = ThreePlayerCertifiedGlobalPins()


@dataclass(frozen=True)
class ThreePlayerCertifiedGlobalOracleWorkCounters:
    """Completed M39 preparation, point, response, and bound work."""

    preparation_m31_runs: int = 0
    point_evaluations: int = 0
    point_m31_runs: int = 0
    bound_evaluations: int = 0
    singleton_bound_evaluations: int = 0
    complete_response_records: int = 0
    joint_profiles_evaluated: int = 0
    terminal_path_evaluations: int = 0
    terminal_records_prepared: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ThreePlayerCertifiedPointRecord:
    """One complete fixed-profile/fresh-response exact point result."""

    policy_identity: str
    is_baseline_identity: bool
    fixed_profile_utility: Mapping[str, str]
    post_response_hero_worst: str
    post_response_hero_best: str
    baseline_total_repeated_hero_ev: str
    candidate_total_repeated_hero_ev: str
    uplift: str
    complete_response_correspondence_identity: str
    complete_response_record_count: int
    hero_worst_witness_count: int
    scenario_response: _m31.ScenarioResponseResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_identity": self.policy_identity,
            "is_baseline_identity": self.is_baseline_identity,
            "fixed_profile_utility": dict(self.fixed_profile_utility),
            "post_response_hero_worst": self.post_response_hero_worst,
            "post_response_hero_best": self.post_response_hero_best,
            "baseline_total_repeated_hero_ev": (
                self.baseline_total_repeated_hero_ev
            ),
            "candidate_total_repeated_hero_ev": (
                self.candidate_total_repeated_hero_ev
            ),
            "uplift": self.uplift,
            "complete_response_correspondence_identity": (
                self.complete_response_correspondence_identity
            ),
            "complete_response_record_count": (
                self.complete_response_record_count
            ),
            "hero_worst_witness_count": self.hero_worst_witness_count,
            "scenario_response": self.scenario_response.to_dict(),
        }


@dataclass(frozen=True)
class KnownBoardRealCardPreparation:
    """M35 exact real-card provenance retained without running M32 candidates."""

    prepared_support: _m35.PreparedThreePlayerSupport
    showdown_rows: tuple[_m35.KnownBoardShowdownRow, ...]
    canonical_profile: _m35.RealCardThreePlayerProfile
    bucket_maps: Mapping[str, Any]
    workload: Mapping[str, int]
    baseline_identity: str
    identities: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prepared_support": self.prepared_support.to_dict(),
            "showdown_rows": [row.to_dict() for row in self.showdown_rows],
            "canonical_profile": self.canonical_profile.to_dict(),
            "bucket_maps": {
                player: _m35._bucket_map_dict(self.bucket_maps[player])
                for player in _m35.PLAYERS
            },
            "workload": dict(self.workload),
            "baseline_identity": self.baseline_identity,
            "identities": dict(self.identities),
        }


@dataclass(frozen=True)
class ThreePlayerCertifiedGlobalPreparation:
    """Canonical public preparation and sound-bound provenance."""

    surface_kind: str
    m36_scenario: _m36.HeroBehaviorScenario
    baseline_policy: _m36.ExactBehaviorPolicy
    baseline_point: ThreePlayerCertifiedPointRecord
    terminal_hero_utility_minimum: str
    terminal_hero_utility_maximum: str
    terminal_record_count: int
    repeated_projection: Mapping[str, Any]
    identities: Mapping[str, str]
    real_card: KnownBoardRealCardPreparation | None
    oracle: "ThreePlayerCertifiedScalarOracle"

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_kind": self.surface_kind,
            "m36_scenario": {
                "scenario_identity": self.m36_scenario.scenario_identity,
                "information_sets": [
                    {
                        "information_set_id": row.information_set_id,
                        "legal_action_ids": list(row.legal_action_ids),
                    }
                    for row in self.m36_scenario.information_sets
                ],
            },
            "baseline_policy": self.baseline_policy.to_dict(),
            "baseline_point": self.baseline_point.to_dict(),
            "terminal_hero_utility_minimum": (
                self.terminal_hero_utility_minimum
            ),
            "terminal_hero_utility_maximum": (
                self.terminal_hero_utility_maximum
            ),
            "terminal_record_count": self.terminal_record_count,
            "repeated_projection": dict(self.repeated_projection),
            "identities": dict(self.identities),
            "real_card": None if self.real_card is None else self.real_card.to_dict(),
        }


@dataclass(frozen=True)
class ThreePlayerCertifiedGlobalPayload:
    """Complete M39 success payload."""

    preparation: ThreePlayerCertifiedGlobalPreparation
    baseline_point: ThreePlayerCertifiedPointRecord
    selected_point: ThreePlayerCertifiedPointRecord | None
    native_optimizer_result: _m36.CertifiedGlobalOptimizerResult
    oracle_work_counters: ThreePlayerCertifiedGlobalOracleWorkCounters

    def to_dict(self) -> dict[str, Any]:
        native_payload = self.native_optimizer_result.payload
        return {
            "contract_version": CONTRACT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "domain_contract": DOMAIN_CONTRACT,
            "objective_contract": OBJECTIVE_CONTRACT,
            "response_contract": RESPONSE_CONTRACT,
            "bound_contract": BOUND_CONTRACT,
            "certificate_claim": CLAIM_SCOPE,
            "rational_lift": RATIONAL_LIFT,
            "preparation": self.preparation.to_dict(),
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
class ThreePlayerCertifiedGlobalError:
    """Bounded error metadata carrying no partial payload."""

    phase: str
    message: str
    cause_status: str | None = None


@dataclass(frozen=True)
class ThreePlayerCertifiedGlobalResult:
    """Exclusive certified success or fail-closed M39 result."""

    status: str
    payload: ThreePlayerCertifiedGlobalPayload | None
    error: ThreePlayerCertifiedGlobalError | None
    optimizer_work_counters: _m36.CertifiedGlobalWorkCounters
    oracle_work_counters: ThreePlayerCertifiedGlobalOracleWorkCounters
    partial_result: bool = False

    def __post_init__(self) -> None:
        success = self.status in (
            _m36.CERTIFIED_GLOBAL,
            _m36.CERTIFIED_EPSILON_GLOBAL,
        )
        if self.partial_result:
            raise ValueError("M39 partial results are forbidden")
        if success != (self.payload is not None and self.error is None):
            raise ValueError("M39 success/failure wrapper is inconsistent")
        if not success and (self.payload is not None or self.error is None):
            raise ValueError("M39 failure must have payload null and error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "payload": None if self.payload is None else self.payload.to_dict(),
            "error": None if self.error is None else asdict(self.error),
            "optimizer_work_counters": self.optimizer_work_counters.to_dict(),
            "oracle_work_counters": self.oracle_work_counters.to_dict(),
            "partial_result": False,
        }


@dataclass
class _MutableOracleCounters:
    preparation_m31_runs: int = 0
    point_evaluations: int = 0
    point_m31_runs: int = 0
    bound_evaluations: int = 0
    singleton_bound_evaluations: int = 0
    complete_response_records: int = 0
    joint_profiles_evaluated: int = 0
    terminal_path_evaluations: int = 0
    terminal_records_prepared: int = 0

    def snapshot(self) -> ThreePlayerCertifiedGlobalOracleWorkCounters:
        return ThreePlayerCertifiedGlobalOracleWorkCounters(**asdict(self))


class _M39Failure(ValueError):
    def __init__(
        self,
        status: str,
        phase: str,
        message: str,
        *,
        cause_status: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.phase = phase
        self.message = message[:512]
        self.cause_status = cause_status


class _OutputLimitReached(RuntimeError):
    pass


def _validate_limits(
    value: object,
) -> ThreePlayerCertifiedGlobalLimits:
    if type(value) is not ThreePlayerCertifiedGlobalLimits:
        raise _M39Failure(
            _m36.INVALID_INPUT,
            "limits",
            "integration_limits has the wrong type",
        )
    ceilings = {
        "max_point_evaluations": MAX_POINT_EVALUATIONS,
        "max_bound_evaluations": MAX_BOUND_EVALUATIONS,
        "max_aggregate_response_records": MAX_AGGREGATE_RESPONSE_RECORDS,
        "max_aggregate_joint_profiles": MAX_AGGREGATE_JOINT_PROFILES,
        "max_aggregate_terminal_evaluations": (
            MAX_AGGREGATE_TERMINAL_EVALUATIONS
        ),
        "max_terminal_records": MAX_TERMINAL_RECORDS,
        "max_horizon": MAX_HORIZON,
        "max_rational_numerator_bits": MAX_RATIONAL_BITS,
        "max_rational_denominator_bits": MAX_RATIONAL_BITS,
        "max_output_records": MAX_OUTPUT_RECORDS,
        "max_output_bytes": MAX_OUTPUT_BYTES,
    }
    for item in fields(ThreePlayerCertifiedGlobalLimits):
        _require_positive_int(
            getattr(value, item.name), f"limits.{item.name}", ceilings[item.name]
        )
    return value


def _validate_pins(value: object) -> ThreePlayerCertifiedGlobalPins:
    if type(value) is not ThreePlayerCertifiedGlobalPins:
        raise _M39Failure(_m36.INVALID_INPUT, "pins", "pins has the wrong type")
    normalized = {
        item.name: _normalized_identity(getattr(value, item.name), item.name)
        for item in fields(ThreePlayerCertifiedGlobalPins)
    }
    return ThreePlayerCertifiedGlobalPins(**normalized)


def _repeated_projection(
    value: object, limits: ThreePlayerCertifiedGlobalLimits
) -> tuple[dict[str, Any], Fraction, Fraction, Fraction]:
    if type(value) is not ThreePlayerCertifiedRepeatedConfig:
        raise _M39Failure(
            _m36.INVALID_INPUT, "repeated", "repeated has the wrong type"
        )
    horizon = _require_positive_int(
        value.horizon, "repeated.horizon", limits.max_horizon
    )
    if (
        type(value.adaptation_opportunity) is not int
        or value.adaptation_opportunity < 1
        or value.adaptation_opportunity > horizon + 1
    ):
        raise _M39Failure(
            _m36.INVALID_INPUT,
            "repeated.adaptation_opportunity",
            "adaptation opportunity must be in 1..horizon+1",
        )
    bit_cap = min(
        limits.max_rational_numerator_bits,
        limits.max_rational_denominator_bits,
    )
    discount = _parse_exact(
        value.discount,
        "repeated.discount",
        nonnegative=True,
        maximum_bits=bit_cap,
    )
    term = Fraction(1)
    pre = Fraction(0)
    post = Fraction(0)
    for opportunity in range(1, horizon + 1):
        if opportunity < value.adaptation_opportunity:
            pre = _checked_add(pre, term, "repeated.pre_weight", bit_cap)
        else:
            post = _checked_add(post, term, "repeated.post_weight", bit_cap)
        term = _checked_mul(term, discount, "repeated.discount_power", bit_cap)
    total = _checked_add(pre, post, "repeated.total_weight", bit_cap)
    projection = {
        "horizon": horizon,
        "adaptation_opportunity": value.adaptation_opportunity,
        "discount_exact": _rational_text(discount),
        "discount_source": (
            "lossless_binary64"
            if type(value.discount) is float
            else "canonical_exact"
        ),
        "pre_adaptation_weight": _rational_text(pre),
        "post_adaptation_weight": _rational_text(post),
        "total_weight": _rational_text(total),
    }
    return projection, pre, post, total


def _hero_information_sets(
    root: _m31.RiverNode,
) -> tuple[_m36.HeroInformationSet, ...]:
    rows: dict[str, tuple[str, ...]] = {}
    seen_nodes: set[int] = set()

    def visit(node: _m31.RiverNode) -> None:
        marker = id(node)
        if marker in seen_nodes:
            return
        seen_nodes.add(marker)
        if isinstance(node, _m31.RiverChanceNode):
            for outcome in node.outcomes:
                visit(outcome.child)
            return
        if isinstance(node, _m31.RiverTerminalNode):
            return
        if type(node) is not _m31.RiverDecisionNode:
            raise _M39Failure(
                _m36.UNSUPPORTED_DOMAIN, "domain", "unsupported M31 node"
            )
        actions = tuple(sorted(action.action_id for action in node.actions))
        if node.owner == "H":
            previous = rows.setdefault(node.information_set_id, actions)
            if previous != actions:
                raise _M39Failure(
                    _m36.UNSUPPORTED_DOMAIN,
                    "domain",
                    "Hero information-set action signatures disagree",
                )
        for action in node.actions:
            visit(action.child)

    visit(root)
    if not rows:
        raise _M39Failure(
            _m36.UNSUPPORTED_DOMAIN,
            "domain",
            "scenario has no surviving Hero information set",
        )
    return tuple(
        _m36.HeroInformationSet(key, rows[key]) for key in sorted(rows)
    )


def _m36_policy(
    policy: _m31.ExactBehaviorPolicy,
    information_sets: tuple[_m36.HeroInformationSet, ...],
) -> _m36.ExactBehaviorPolicy:
    if type(policy) is not _m31.ExactBehaviorPolicy:
        raise _M39Failure(
            _m36.INVALID_INPUT, "baseline_policy", "baseline policy type mismatch"
        )
    rows = []
    for information_set in information_sets:
        mapping = policy.probabilities.get(information_set.information_set_id)
        if not isinstance(mapping, Mapping):
            raise _M39Failure(
                _m36.INVALID_INPUT,
                "baseline_policy",
                "baseline policy is incomplete",
            )
        rows.append(
            _m36.ExactBehaviorRow(
                information_set_id=information_set.information_set_id,
                actions=tuple(
                    _m36.ExactActionProbability(action, mapping[action])
                    for action in information_set.legal_action_ids
                ),
            )
        )
    return _m36.ExactBehaviorPolicy(tuple(rows))


def _m31_policy(policy: _m36.ExactBehaviorPolicy) -> _m31.ExactBehaviorPolicy:
    return _m31.ExactBehaviorPolicy(
        probabilities={
            row.information_set_id: {
                action.action_id: action.probability for action in row.actions
            }
            for row in policy.rows
        }
    )


def _parse_utility(
    value: Mapping[str, Any],
    limits: ThreePlayerCertifiedGlobalLimits,
    phase: str,
) -> dict[str, Fraction]:
    cap = min(
        limits.max_rational_numerator_bits,
        limits.max_rational_denominator_bits,
    )
    return {
        player: _parse_exact(
            value[player], f"{phase}.{player}", maximum_bits=cap
        )
        for player in ("H", "O1", "O2", "R")
    }


def _validated_m31(
    result: _m31.ScenarioResponseResult, phase: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        type(result) is not _m31.ScenarioResponseResult
        or result.status != _m31.EXACT_SCENARIO_RESPONSE_COMPLETE
        or result.scenario_evaluation is None
        or result.payoff_table is None
        or result.response is None
        or result.error is not None
        or result.partial_result
        or result.response.get("status")
        != _m30.EXACT_CORRESPONDENCE_COMPLETE
        or result.response.get("coverage") != "complete"
        or result.response.get("partial_response") is not False
    ):
        cause = getattr(result, "status", None)
        raise _M39Failure(
            _m36.ORACLE_FAILURE,
            phase,
            "M31 did not return a complete exact M30 correspondence",
            cause_status=cause,
        )
    return result.scenario_evaluation, result.response


def _response_counts(
    scenario: Mapping[str, Any],
    response: Mapping[str, Any],
) -> tuple[int, int, int]:
    try:
        records = response["counts"]["output_records_projected"]
        joint = scenario["counts"]["joint_profiles"]
        terminals = scenario["counts"]["independent_terminal_path_evaluations"]
    except (KeyError, TypeError):
        raise _M39Failure(
            _m36.ORACLE_FAILURE,
            "response",
            "complete response counters are missing",
        ) from None
    if any(type(value) is not int or value < 0 for value in (records, joint, terminals)):
        raise _M39Failure(
            _m36.ORACLE_FAILURE,
            "response",
            "complete response counters are malformed",
        )
    return records, joint, terminals


class ThreePlayerCertifiedScalarOracle:
    """M39 point oracle plus sound exact terminal-envelope cell bound."""

    response_semantics = _m36.RESPONSE_SEMANTICS
    bound_contract_version = _m36.BOUND_CONTRACT_VERSION

    def __init__(
        self,
        *,
        scenario: _m31.ThreePlayerRiverRakeScenario,
        attestation: _m31.PerfectRecallAttestation,
        initial_profile: _m31.OpponentInitialProfile,
        m36_scenario: _m36.HeroBehaviorScenario,
        baseline_policy: _m36.ExactBehaviorPolicy,
        baseline_result: _m31.ScenarioResponseResult,
        baseline_value: Fraction,
        terminal_minimum: Fraction,
        terminal_maximum: Fraction,
        pre_weight: Fraction,
        post_weight: Fraction,
        total_weight: Fraction,
        limits: ThreePlayerCertifiedGlobalLimits,
        m31_limits: _m31.RiverRakeLimits,
        m30_limits: _m30.ExactResponseLimits,
        m31_pins: _m31.RiverRakeIdentityPins,
        m30_pins: _m30.ResponseIdentityPins,
        response_oracle_identity: str,
        objective_identity: str,
        counters: _MutableOracleCounters,
    ) -> None:
        self.scenario = scenario
        self.attestation = attestation
        self.initial_profile = initial_profile
        self.m36_scenario = m36_scenario
        self.baseline_policy = baseline_policy
        self.baseline_policy_identity = _m36.behavior_policy_identity(
            m36_scenario, baseline_policy
        )
        self.baseline_result = baseline_result
        self.baseline_value = baseline_value
        self.terminal_minimum = terminal_minimum
        self.terminal_maximum = terminal_maximum
        self.pre_weight = pre_weight
        self.post_weight = post_weight
        self.total_weight = total_weight
        self.limits = limits
        self.m31_limits = m31_limits
        self.m30_limits = m30_limits
        self.m31_pins = m31_pins
        self.m30_pins = m30_pins
        self.response_oracle_identity = response_oracle_identity
        self.objective_identity = objective_identity
        self._counters = counters
        self._records: dict[str, ThreePlayerCertifiedPointRecord] = {}

    def work_counters(self) -> ThreePlayerCertifiedGlobalOracleWorkCounters:
        return self._counters.snapshot()

    def point_record(
        self, policy_identity: str
    ) -> ThreePlayerCertifiedPointRecord | None:
        return self._records.get(policy_identity)

    def _result_for(
        self, policy: _m36.ExactBehaviorPolicy, policy_identity: str
    ) -> _m31.ScenarioResponseResult:
        if policy_identity == self.baseline_policy_identity:
            return self.baseline_result
        if self._counters.point_m31_runs + 1 > self.limits.max_point_evaluations:
            raise _m36.ScalarOracleResourceLimit(
                "M39 point M31-run cap exceeded"
            )
        self._counters.point_m31_runs += 1
        return _m31.evaluate_three_player_river_rake(
            self.scenario,
            _m31_policy(policy),
            attestation=self.attestation,
            initial_profile=self.initial_profile,
            limits=self.m31_limits,
            m30_limits=self.m30_limits,
            pins=self.m31_pins,
            response_pins=self.m30_pins,
        )

    def evaluate(
        self, policy: _m36.ExactBehaviorPolicy
    ) -> _m36.ScalarOracleEvaluation:
        if self._counters.point_evaluations + 1 > self.limits.max_point_evaluations:
            raise _m36.ScalarOracleResourceLimit(
                "M39 point-evaluation cap exceeded"
            )
        self._counters.point_evaluations += 1
        policy_identity = _m36.behavior_policy_identity(
            self.m36_scenario, policy
        )
        existing = self._records.get(policy_identity)
        if existing is not None:
            return _m36.ScalarOracleEvaluation(
                policy_identity=existing.policy_identity,
                response_oracle_identity=self.response_oracle_identity,
                objective_identity=self.objective_identity,
                baseline_total_repeated_hero_ev=(
                    existing.baseline_total_repeated_hero_ev
                ),
                candidate_total_repeated_hero_ev=(
                    existing.candidate_total_repeated_hero_ev
                ),
                uplift=existing.uplift,
                complete_response_correspondence_identity=(
                    existing.complete_response_correspondence_identity
                ),
                complete_response_record_count=(
                    existing.complete_response_record_count
                ),
                hero_worst_witness_count=existing.hero_worst_witness_count,
            )
        result = self._result_for(policy, policy_identity)
        scenario, response = _validated_m31(result, "point.response")
        records, joint, terminals = _response_counts(scenario, response)
        if (
            self._counters.complete_response_records + records
            > self.limits.max_aggregate_response_records
            or self._counters.joint_profiles_evaluated + joint
            > self.limits.max_aggregate_joint_profiles
            or self._counters.terminal_path_evaluations + terminals
            > self.limits.max_aggregate_terminal_evaluations
        ):
            raise _m36.ScalarOracleResourceLimit(
                "M39 aggregate complete-response workload cap exceeded"
            )
        self._counters.complete_response_records += records
        self._counters.joint_profiles_evaluated += joint
        self._counters.terminal_path_evaluations += terminals
        initial = scenario.get("initial_profile_comparison")
        if not isinstance(initial, Mapping) or not isinstance(
            initial.get("utility"), Mapping
        ):
            raise RuntimeError("M31 fixed-profile utility is missing")
        fixed = _parse_utility(
            initial["utility"], self.limits, "point.fixed_profile"
        )
        cap = min(
            self.limits.max_rational_numerator_bits,
            self.limits.max_rational_denominator_bits,
        )
        worst = _parse_exact(
            response["hero_worst"],
            "point.hero_worst",
            maximum_bits=cap,
        )
        best = _parse_exact(
            response["hero_best"],
            "point.hero_best",
            maximum_bits=cap,
        )
        baseline_total = _checked_mul(
            self.total_weight,
            self.baseline_value,
            "point.baseline_total",
            cap,
        )
        is_baseline = policy_identity == self.baseline_policy_identity
        if is_baseline:
            candidate_total = baseline_total
            uplift = Fraction(0)
        else:
            pre_value = _checked_mul(
                self.pre_weight, fixed["H"], "point.pre_value", cap
            )
            post_value = _checked_mul(
                self.post_weight, worst, "point.post_value", cap
            )
            candidate_total = _checked_add(
                pre_value, post_value, "point.candidate_total", cap
            )
            uplift = _checked_add(
                candidate_total,
                -baseline_total,
                "point.uplift",
                cap,
            )
        correspondence_identity = _identity(response)
        witnesses = response.get("hero_worst_witnesses")
        if not isinstance(witnesses, list) or not witnesses:
            raise RuntimeError("M30 Hero-worst witnesses are missing")
        record = ThreePlayerCertifiedPointRecord(
            policy_identity=policy_identity,
            is_baseline_identity=is_baseline,
            fixed_profile_utility={
                player: _rational_text(fixed[player])
                for player in ("H", "O1", "O2", "R")
            },
            post_response_hero_worst=_rational_text(worst),
            post_response_hero_best=_rational_text(best),
            baseline_total_repeated_hero_ev=_rational_text(baseline_total),
            candidate_total_repeated_hero_ev=_rational_text(candidate_total),
            uplift=_rational_text(uplift),
            complete_response_correspondence_identity=(
                correspondence_identity
            ),
            complete_response_record_count=records,
            hero_worst_witness_count=len(witnesses),
            scenario_response=result,
        )
        self._records[policy_identity] = record
        return _m36.ScalarOracleEvaluation(
            policy_identity=policy_identity,
            response_oracle_identity=self.response_oracle_identity,
            objective_identity=self.objective_identity,
            baseline_total_repeated_hero_ev=_rational_text(baseline_total),
            candidate_total_repeated_hero_ev=_rational_text(candidate_total),
            uplift=_rational_text(uplift),
            complete_response_correspondence_identity=(
                correspondence_identity
            ),
            complete_response_record_count=records,
            hero_worst_witness_count=len(witnesses),
        )

    def upper_bound(
        self, cell: _m36.ExactBehaviorCell
    ) -> _m36.ScalarOracleBound:
        if self._counters.bound_evaluations + 1 > self.limits.max_bound_evaluations:
            raise _m36.ScalarOracleResourceLimit("M39 bound cap exceeded")
        self._counters.bound_evaluations += 1
        cell_identity = _m36.behavior_cell_identity(self.m36_scenario, cell)
        singleton = True
        policy_rows = []
        for row in cell.rows:
            parsed_intervals = [
                (
                    action,
                    Fraction(action.lower),
                    Fraction(action.upper),
                )
                for action in row.actions
            ]
            actions = []
            for index, (action, lower, upper) in enumerate(parsed_intervals):
                other_lower = sum(
                    item[1]
                    for other_index, item in enumerate(parsed_intervals)
                    if other_index != index
                )
                other_upper = sum(
                    item[2]
                    for other_index, item in enumerate(parsed_intervals)
                    if other_index != index
                )
                effective_lower = max(lower, Fraction(1) - other_upper)
                effective_upper = min(upper, Fraction(1) - other_lower)
                if effective_lower != effective_upper:
                    singleton = False
                actions.append(
                    _m36.ExactActionProbability(
                        action.action_id, _rational_text(effective_lower)
                    )
                )
            policy_rows.append(
                _m36.ExactBehaviorRow(row.information_set_id, tuple(actions))
            )
        candidate: _m36.ExactBehaviorPolicy | None = None
        if singleton:
            self._counters.singleton_bound_evaluations += 1
            candidate = _m36.ExactBehaviorPolicy(tuple(policy_rows))
            evaluation = self.evaluate(candidate)
            upper = _parse_exact(evaluation.uplift, "bound.singleton")
            proof_kind = "singleton_complete_m30_point_equality"
        else:
            cap = min(
                self.limits.max_rational_numerator_bits,
                self.limits.max_rational_denominator_bits,
            )
            maximum_total = _checked_mul(
                self.total_weight,
                self.terminal_maximum,
                "bound.terminal_envelope",
                cap,
            )
            baseline_total = _checked_mul(
                self.total_weight,
                self.baseline_value,
                "bound.baseline_total",
                cap,
            )
            upper = max(
                Fraction(0),
                _checked_add(
                    maximum_total,
                    -baseline_total,
                    "bound.uplift",
                    cap,
                ),
            )
            proof_kind = (
                "all_exact_m31_terminal_hero_utilities_envelope;"
                "all_complete_m30_responses_included"
            )
        candidate_identity = (
            None
            if candidate is None
            else _m36.behavior_policy_identity(self.m36_scenario, candidate)
        )
        bound_identity = _m36.scalar_oracle_bound_identity(
            cell_identity=cell_identity,
            response_oracle_identity=self.response_oracle_identity,
            objective_identity=self.objective_identity,
            upper_bound=_rational_text(upper),
            candidate_policy_identity=candidate_identity,
        )
        return _m36.ScalarOracleBound(
            cell_identity=cell_identity,
            response_oracle_identity=self.response_oracle_identity,
            objective_identity=self.objective_identity,
            upper_bound=_rational_text(upper),
            bound_identity=bound_identity,
            candidate_policy=candidate,
        )


@dataclass(frozen=True)
class _PreparedInputs:
    surface_kind: str
    scenario: _m31.ThreePlayerRiverRakeScenario
    baseline_policy: _m31.ExactBehaviorPolicy
    initial_profile: _m31.OpponentInitialProfile
    attestation: _m31.PerfectRecallAttestation
    repeated: ThreePlayerCertifiedRepeatedConfig
    absolute_gap_tolerance: object
    relative_gap_tolerance: object
    integration_limits: ThreePlayerCertifiedGlobalLimits
    optimizer_limits: _m36.CertifiedGlobalOptimizerLimits
    m31_limits: _m31.RiverRakeLimits
    m30_limits: _m30.ExactResponseLimits
    m31_pins: _m31.RiverRakeIdentityPins
    m30_pins: _m30.ResponseIdentityPins
    optimizer_pins: _m36.CertifiedGlobalOptimizerPins
    pins: ThreePlayerCertifiedGlobalPins
    real_card: KnownBoardRealCardPreparation | None


def _real_card_inputs(
    request: KnownBoardRealCardThreePlayerCertifiedGlobalRequest,
) -> _PreparedInputs:
    if type(request.source) is not _m35.KnownBoardRealCardThreePlayerRequest:
        raise _M39Failure(
            _m36.INVALID_INPUT, "real_card", "source has the wrong type"
        )
    source = _m35._validate_request_types(request.source)
    limits = source.limits
    initial = _m35._parse_rational(
        source.initial_contribution,
        "initial_contribution",
        minimum=Fraction(0),
        limits=limits,
    )
    bet_to = _m35._parse_rational(
        source.bet_to, "bet_to", minimum=Fraction(0), limits=limits
    )
    rake_rate = _m35._parse_rational(
        source.rake_rate,
        "rake_rate",
        minimum=Fraction(0),
        maximum=Fraction(1),
        limits=limits,
    )
    rake_cap = (
        None
        if source.rake_cap is None
        else _m35._parse_rational(
            source.rake_cap,
            "rake_cap",
            minimum=Fraction(0),
            limits=limits,
        )
    )
    if initial <= 0 or bet_to <= initial:
        raise _M39Failure(
            _m36.UNSUPPORTED_DOMAIN,
            "real_card",
            "M35 initial contribution/bet contract is unsupported",
        )
    prepared = _m35._prepare_support_internal(
        board=source.board,
        dead_cards=source.dead_cards,
        hero_range=source.hero_range,
        o1_range=source.o1_range,
        o2_range=source.o2_range,
        limits=limits,
        aiof_limits=source.aiof_limits,
    )
    combos = {
        seat.player_id: tuple(combo.combo for combo in seat.final.combos)
        for seat in prepared.seats
    }
    bucket_maps = {
        "H": _m35._canonical_bucket_map(
            source.hero_combo_to_bucket,
            combos["H"],
            "H",
            limits.max_buckets_per_seat,
        ),
        "O1": _m35._canonical_bucket_map(
            source.o1_combo_to_bucket,
            combos["O1"],
            "O1",
            limits.max_buckets_per_seat,
        ),
        "O2": _m35._canonical_bucket_map(
            source.o2_combo_to_bucket,
            combos["O2"],
            "O2",
            limits.max_buckets_per_seat,
        ),
    }
    canonical_profile, hero_policy, initial_profile = _m35._canonical_profile(
        source.baseline_profile, bucket_maps, limits
    )
    lookups = {
        player: _m35._combo_bucket_lookup(bucket_maps[player])
        for player in _m35.PLAYERS
    }
    triple_count = prepared.public.compatible_triple_count
    projected_nodes = 1 + 11 * triple_count
    hero_info_sets = len(bucket_maps["H"].bucket_ids)
    o1_info_sets = 2 * len(bucket_maps["O1"].bucket_ids)
    o2_info_sets = 3 * len(bucket_maps["O2"].bucket_ids)
    o1_plans = 2 ** len(bucket_maps["O1"].bucket_ids)
    o2_plans = 4 ** len(bucket_maps["O2"].bucket_ids)
    if (
        projected_nodes > limits.max_tree_nodes
        or projected_nodes > source.m31_limits.max_nodes
        or triple_count > source.m31_limits.max_chance_outcomes
        or hero_info_sets > source.m31_limits.max_fixed_hero_info_sets
        or o1_info_sets > source.m31_limits.max_info_sets_per_opponent
        or o2_info_sets > source.m31_limits.max_info_sets_per_opponent
        or o1_info_sets + o2_info_sets
        > source.m31_limits.max_opponent_info_sets_total
        or o1_plans > source.m31_limits.max_pure_plans_o1
        or o2_plans > source.m31_limits.max_pure_plans_o2
        or o1_plans * o2_plans > source.m31_limits.max_joint_pure_profiles
    ):
        raise _M39Failure(
            _m36.LIMIT_REACHED_NO_CERTIFICATE,
            "real_card.preparation",
            "M35/M31 pre-allocation tree or response cap exceeded",
        )
    identity_records = (
        triple_count * 32
        + sum(len(mapping.assignments) for mapping in bucket_maps.values())
        + len(canonical_profile.rows) * 8
        + 256
    )
    if identity_records > limits.max_identity_records:
        raise _M39Failure(
            _m36.LIMIT_REACHED_NO_CERTIFICATE,
            "real_card.preparation",
            "M35 identity cap exceeded before rank/tree allocation",
        )
    ranks = _m35._rank_cache(prepared, limits)
    showdown_rows = _m35._showdown_rows(prepared, lookups, ranks)
    outcomes = tuple(
        _m31.RiverChanceOutcome(
            outcome_id=f"triple-{index:04d}",
            probability=row.probability_exact,
            observation=_m31.RiverObservation(
                public_observation_id=(
                    f"known-board::{_m35._identity(prepared.public.board)[:24]}"
                ),
                private_observation_id_by_player={
                    "H": f"bucket::{row.hero_bucket_id}",
                    "O1": f"bucket::{row.o1_bucket_id}",
                    "O2": f"bucket::{row.o2_bucket_id}",
                },
            ),
            child=_m35._betting_child(
                f"{index:04d}", row, _m35._rational_text(bet_to)
            ),
        )
        for index, row in enumerate(showdown_rows, 1)
    )
    scenario = _m31.ThreePlayerRiverRakeScenario(
        root=_m31.RiverChanceNode("known-board-triple-root", outcomes),
        button_player_id="O2",
        seat_order=("H", "O1", "O2"),
        river_action_order=("H", "O1", "O2"),
        initial_observation=None,
        initial_pot=_m35._rational_text(3 * initial),
        initial_contribution={
            player: _m35._rational_text(initial) for player in _m35.PLAYERS
        },
        max_total_contribution={
            player: _m35._rational_text(bet_to + initial)
            for player in _m35.PLAYERS
        },
        rake_rate=_m35._rational_text(rake_rate),
        rake_cap=None if rake_cap is None else _m35._rational_text(rake_cap),
    )
    evidence = _m31.create_perfect_recall_attestation(
        scenario,
        verifier=source.attestation.verifier,
        verification_date=source.attestation.verification_date,
        evidence_version=source.attestation.evidence_version,
        o1_confirmed=source.attestation.o1_confirmed,
        o2_confirmed=source.attestation.o2_confirmed,
        limits=source.m31_limits,
    )
    profile_identity = _m35._identity(
        {
            "profile_version": _m35.PROFILE_VERSION,
            "profile": canonical_profile,
            "bucket_maps": bucket_maps,
        }
    )
    scenario_parameters_identity = _m35._identity(
        {
            "betting_tree_version": _m35.BETTING_TREE_VERSION,
            "initial_contribution": initial,
            "bet_to": bet_to,
            "rake_rate": rake_rate,
            "rake_cap": rake_cap,
            "prepared_support": prepared.public.content_identity,
            "bucket_maps": bucket_maps,
        }
    )
    baseline_identity = _m35._identity(
        {
            "identity_version": _m35.IDENTITY_VERSION,
            "prepared_support": prepared.public.content_identity,
            "profile": profile_identity,
            "scenario_parameters": scenario_parameters_identity,
            "perfect_recall": asdict(evidence),
        }
    )
    if (
        source.expected_baseline_identity is not None
        and source.expected_baseline_identity != baseline_identity
    ):
        raise _M39Failure(
            _m36.STALE_INPUT,
            "real_card.pins",
            "M35 expected baseline identity is stale",
        )
    real_card = KnownBoardRealCardPreparation(
        prepared_support=prepared.public,
        showdown_rows=showdown_rows,
        canonical_profile=canonical_profile,
        bucket_maps=bucket_maps,
        workload={
            "cartesian_triples": prepared.public.cartesian_triple_count,
            "compatible_triples": triple_count,
            "fixed_board_evaluations": len(ranks),
            "tree_nodes": projected_nodes,
            "profile_rows": len(canonical_profile.rows),
            "o1_pure_plans": o1_plans,
            "o2_pure_plans": o2_plans,
            "joint_pure_profiles": o1_plans * o2_plans,
        },
        baseline_identity=baseline_identity,
        identities={
            "prepared_support": prepared.public.content_identity,
            "profile": profile_identity,
            "scenario_parameters": scenario_parameters_identity,
            "baseline": baseline_identity,
            "evaluator": _m35._identity({"evaluator": _m35.EVALUATOR_ID}),
        },
    )
    repeated = ThreePlayerCertifiedRepeatedConfig(
        horizon=source.repeated.horizon,
        adaptation_opportunity=request.adaptation_opportunity,
        discount=source.repeated.discount,
    )
    return _PreparedInputs(
        surface_kind=KNOWN_BOARD_REAL_CARD,
        scenario=scenario,
        baseline_policy=hero_policy,
        initial_profile=initial_profile,
        attestation=evidence,
        repeated=repeated,
        absolute_gap_tolerance=request.absolute_gap_tolerance,
        relative_gap_tolerance=request.relative_gap_tolerance,
        integration_limits=request.integration_limits,
        optimizer_limits=request.optimizer_limits,
        m31_limits=source.m31_limits,
        m30_limits=source.m30_limits,
        m31_pins=_m31.RiverRakeIdentityPins(),
        m30_pins=_m30.ResponseIdentityPins(),
        optimizer_pins=request.optimizer_pins,
        pins=request.pins,
        real_card=real_card,
    )


def _abstract_inputs(
    request: AbstractThreePlayerCertifiedGlobalRequest,
) -> _PreparedInputs:
    if type(request) is not AbstractThreePlayerCertifiedGlobalRequest:
        raise _M39Failure(
            _m36.INVALID_INPUT, "request", "abstract request has the wrong type"
        )
    nested = (
        (request.scenario, _m31.ThreePlayerRiverRakeScenario, "scenario"),
        (
            request.baseline_fixed_hero_policy,
            _m31.ExactBehaviorPolicy,
            "baseline_fixed_hero_policy",
        ),
        (
            request.initial_profile,
            _m31.OpponentInitialProfile,
            "initial_profile",
        ),
        (
            request.attestation,
            _m31.PerfectRecallAttestation,
            "attestation",
        ),
        (
            request.optimizer_limits,
            _m36.CertifiedGlobalOptimizerLimits,
            "optimizer_limits",
        ),
        (request.m31_limits, _m31.RiverRakeLimits, "m31_limits"),
        (request.m30_limits, _m30.ExactResponseLimits, "m30_limits"),
        (request.m31_pins, _m31.RiverRakeIdentityPins, "m31_pins"),
        (request.m30_pins, _m30.ResponseIdentityPins, "m30_pins"),
        (
            request.optimizer_pins,
            _m36.CertifiedGlobalOptimizerPins,
            "optimizer_pins",
        ),
    )
    for value, expected, name in nested:
        if type(value) is not expected:
            raise _M39Failure(
                _m36.INVALID_INPUT, "request", f"{name} has the wrong type"
            )
    return _PreparedInputs(
        surface_kind=ABSTRACT,
        scenario=request.scenario,
        baseline_policy=request.baseline_fixed_hero_policy,
        initial_profile=request.initial_profile,
        attestation=request.attestation,
        repeated=request.repeated,
        absolute_gap_tolerance=request.absolute_gap_tolerance,
        relative_gap_tolerance=request.relative_gap_tolerance,
        integration_limits=request.integration_limits,
        optimizer_limits=request.optimizer_limits,
        m31_limits=request.m31_limits,
        m30_limits=request.m30_limits,
        m31_pins=request.m31_pins,
        m30_pins=request.m30_pins,
        optimizer_pins=request.optimizer_pins,
        pins=request.pins,
        real_card=None,
    )


def _request_projection(
    inputs: _PreparedInputs,
    *,
    repeated_projection: Mapping[str, Any],
    scenario_identity: str,
    tree_identity: str,
    baseline_identity: str,
) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "surface_kind": inputs.surface_kind,
        "scenario_identity": scenario_identity,
        "tree_structure_identity": tree_identity,
        "baseline_identity": baseline_identity,
        "repeated": dict(repeated_projection),
        "absolute_gap_tolerance": inputs.absolute_gap_tolerance,
        "relative_gap_tolerance": inputs.relative_gap_tolerance,
        "caps": {
            "integration": inputs.integration_limits.to_dict(),
            "optimizer": asdict(inputs.optimizer_limits),
            "m31": asdict(inputs.m31_limits),
            "m30": asdict(inputs.m30_limits),
        },
        "real_card_baseline_identity": (
            None
            if inputs.real_card is None
            else inputs.real_card.baseline_identity
        ),
        "finite_m32_candidate_universe_used_for_domain": False,
    }


def _prepare(
    inputs: _PreparedInputs,
    counters: _MutableOracleCounters,
) -> tuple[
    ThreePlayerCertifiedGlobalPreparation,
    ThreePlayerCertifiedScalarOracle,
    ThreePlayerCertifiedGlobalPins,
    str,
]:
    limits = _validate_limits(inputs.integration_limits)
    pins = _validate_pins(inputs.pins)
    repeated_projection, pre, post, total = _repeated_projection(
        inputs.repeated, limits
    )
    counters.preparation_m31_runs += 1
    baseline_result = _m31.evaluate_three_player_river_rake(
        inputs.scenario,
        inputs.baseline_policy,
        attestation=inputs.attestation,
        initial_profile=inputs.initial_profile,
        limits=inputs.m31_limits,
        m30_limits=inputs.m30_limits,
        pins=inputs.m31_pins,
        response_pins=inputs.m30_pins,
    )
    scenario, response = _validated_m31(
        baseline_result, "preparation.baseline_response"
    )
    records, joint, terminals = _response_counts(scenario, response)
    if (
        records > limits.max_aggregate_response_records
        or joint > limits.max_aggregate_joint_profiles
        or terminals > limits.max_aggregate_terminal_evaluations
    ):
        raise _M39Failure(
            _m36.LIMIT_REACHED_NO_CERTIFICATE,
            "preparation.response",
            "baseline complete-response workload exceeds M39 caps",
        )
    counters.complete_response_records += records
    counters.joint_profiles_evaluated += joint
    counters.terminal_path_evaluations += terminals
    terminal_records = scenario.get("terminal_records")
    if not isinstance(terminal_records, list) or not terminal_records:
        raise _M39Failure(
            _m36.ORACLE_FAILURE,
            "preparation.terminal",
            "M31 exact terminal records are missing",
        )
    if len(terminal_records) > limits.max_terminal_records:
        raise _M39Failure(
            _m36.LIMIT_REACHED_NO_CERTIFICATE,
            "preparation.terminal",
            "terminal-record cap exceeded",
        )
    counters.terminal_records_prepared = len(terminal_records)
    terminal_values = [
        _parse_exact(
            record["utility"]["H"],
            "preparation.terminal.H",
            maximum_bits=min(
                limits.max_rational_numerator_bits,
                limits.max_rational_denominator_bits,
            ),
        )
        for record in terminal_records
    ]
    initial = scenario.get("initial_profile_comparison")
    if not isinstance(initial, Mapping) or not isinstance(
        initial.get("utility"), Mapping
    ):
        raise _M39Failure(
            _m36.ORACLE_FAILURE,
            "preparation.fixed_profile",
            "complete supplied fixed profile is required",
        )
    baseline_utility = _parse_utility(
        initial["utility"], limits, "preparation.baseline_utility"
    )
    ids = scenario.get("identities")
    if not isinstance(ids, Mapping):
        raise _M39Failure(
            _m36.ORACLE_FAILURE,
            "preparation.identity",
            "M31 identities are missing",
        )
    information_sets = _hero_information_sets(inputs.scenario.root)
    m36_scenario = _m36.HeroBehaviorScenario(
        scenario_identity=ids["scenario"],
        information_sets=information_sets,
    )
    baseline_policy = _m36_policy(inputs.baseline_policy, information_sets)
    domain_identity = _identity(
        {
            "contract": DOMAIN_CONTRACT,
            "scenario_identity": ids["scenario"],
            "information_sets": [
                {
                    "information_set_id": row.information_set_id,
                    "legal_action_ids": list(row.legal_action_ids),
                }
                for row in information_sets
            ],
        }
    )
    baseline_identity = _m36.behavior_policy_identity(
        m36_scenario, baseline_policy
    )
    response_oracle_identity = _identity(
        {
            "contract": RESPONSE_CONTRACT,
            "m30_contract": _m30.CONTRACT_VERSION,
            "m31_contract": _m31.CONTRACT_VERSION,
            "scenario_identity": ids["scenario"],
            "tree_structure_identity": ids["tree_structure"],
            "baseline_fixed_hero_identity": ids["fixed_hero"],
            "initial_profile_identity": ids["initial_profile"],
            "terminal_records_identity": _identity(terminal_records),
            "terminal_envelope": {
                "minimum": _rational_text(min(terminal_values)),
                "maximum": _rational_text(max(terminal_values)),
            },
            "caps": {
                "m31": asdict(inputs.m31_limits),
                "m30": asdict(inputs.m30_limits),
                "integration": limits.to_dict(),
            },
        }
    )
    objective_identity = _identity(
        {
            "contract": OBJECTIVE_CONTRACT,
            "baseline_value": _rational_text(baseline_utility["H"]),
            "baseline_policy_identity": baseline_identity,
            "response_oracle_identity": response_oracle_identity,
            "repeated": repeated_projection,
            "baseline_identity_special_case": "exact_zero_uplift",
        }
    )
    request_projection = _request_projection(
        inputs,
        repeated_projection=repeated_projection,
        scenario_identity=ids["scenario"],
        tree_identity=ids["tree_structure"],
        baseline_identity=baseline_identity,
    )
    request_identity = _identity(request_projection)
    analysis_identity = _identity(
        {
            "request_identity": request_identity,
            "domain_identity": domain_identity,
            "response_oracle_identity": response_oracle_identity,
            "objective_identity": objective_identity,
            "algorithm": ALGORITHM_VERSION,
            "bound": BOUND_CONTRACT,
        }
    )
    actual_pins = {
        "request_identity": request_identity,
        "scenario_identity": ids["scenario"],
        "tree_structure_identity": ids["tree_structure"],
        "baseline_identity": baseline_identity,
        "domain_identity": domain_identity,
        "response_oracle_identity": response_oracle_identity,
        "objective_identity": objective_identity,
        "analysis_identity": analysis_identity,
    }
    for name, actual in actual_pins.items():
        _check_pin(getattr(pins, name), actual, name)
    oracle = ThreePlayerCertifiedScalarOracle(
        scenario=inputs.scenario,
        attestation=inputs.attestation,
        initial_profile=inputs.initial_profile,
        m36_scenario=m36_scenario,
        baseline_policy=baseline_policy,
        baseline_result=baseline_result,
        baseline_value=baseline_utility["H"],
        terminal_minimum=min(terminal_values),
        terminal_maximum=max(terminal_values),
        pre_weight=pre,
        post_weight=post,
        total_weight=total,
        limits=limits,
        m31_limits=inputs.m31_limits,
        m30_limits=inputs.m30_limits,
        m31_pins=inputs.m31_pins,
        m30_pins=inputs.m30_pins,
        response_oracle_identity=response_oracle_identity,
        objective_identity=objective_identity,
        counters=counters,
    )
    baseline_evaluation = oracle.evaluate(baseline_policy)
    baseline_point = oracle.point_record(baseline_evaluation.policy_identity)
    if baseline_point is None or baseline_evaluation.uplift != "0":
        raise _M39Failure(
            _m36.INVALID_ORACLE_CONTRACT,
            "preparation.baseline",
            "baseline zero-uplift identity failed",
        )
    preparation = ThreePlayerCertifiedGlobalPreparation(
        surface_kind=inputs.surface_kind,
        m36_scenario=m36_scenario,
        baseline_policy=baseline_policy,
        baseline_point=baseline_point,
        terminal_hero_utility_minimum=_rational_text(min(terminal_values)),
        terminal_hero_utility_maximum=_rational_text(max(terminal_values)),
        terminal_record_count=len(terminal_records),
        repeated_projection=repeated_projection,
        identities=actual_pins,
        real_card=inputs.real_card,
        oracle=oracle,
    )
    return preparation, oracle, pins, analysis_identity


@dataclass(frozen=True)
class _JsonObjectProjection:
    items: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class _DeferredJsonProjection:
    factory: Callable[[], Any]


def _payload_output_projection(
    payload: ThreePlayerCertifiedGlobalPayload,
) -> _JsonObjectProjection:
    native_payload = payload.native_optimizer_result.payload
    return _JsonObjectProjection(
        (
            ("contract_version", CONTRACT_VERSION),
            ("schema_version", SCHEMA_VERSION),
            ("algorithm_version", ALGORITHM_VERSION),
            ("domain_contract", DOMAIN_CONTRACT),
            ("objective_contract", OBJECTIVE_CONTRACT),
            ("response_contract", RESPONSE_CONTRACT),
            ("bound_contract", BOUND_CONTRACT),
            ("certificate_claim", CLAIM_SCOPE),
            ("rational_lift", RATIONAL_LIFT),
            (
                "preparation",
                _DeferredJsonProjection(payload.preparation.to_dict),
            ),
            (
                "baseline_point",
                _DeferredJsonProjection(payload.baseline_point.to_dict),
            ),
            (
                "selected_point",
                None
                if payload.selected_point is None
                else _DeferredJsonProjection(payload.selected_point.to_dict),
            ),
            (
                "no_beneficial_commitment",
                None
                if native_payload is None
                else native_payload.no_beneficial_commitment,
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
    result: ThreePlayerCertifiedGlobalResult,
) -> _JsonObjectProjection:
    if result.payload is None or result.error is not None:
        raise RuntimeError("success projection requires success")
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
                _DeferredJsonProjection(result.oracle_work_counters.to_dict),
            ),
            ("partial_result", False),
        )
    )


def _preflight_success_output(
    result: ThreePlayerCertifiedGlobalResult,
    *,
    max_records: int,
    max_bytes: int,
) -> tuple[int, int]:
    record_count = 0
    byte_count = 0

    def add(fragment: str) -> None:
        nonlocal byte_count
        byte_count += len(fragment.encode("utf-8"))
        if byte_count > max_bytes:
            raise _OutputLimitReached(
                "max_output_bytes exceeded before success materialization"
            )

    def visit(value: Any) -> None:
        nonlocal record_count
        if isinstance(value, _DeferredJsonProjection):
            visit(value.factory())
            return
        record_count += 1
        if record_count > max_records:
            raise _OutputLimitReached(
                "max_output_records exceeded before success materialization"
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
    """Expand the lazy public projection for independent tests/review."""

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


def _empty_optimizer_counters() -> _m36.CertifiedGlobalWorkCounters:
    return _m36.CertifiedGlobalWorkCounters()


def _clean_message(value: object, fallback: str) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return (text or fallback)[:512]


def _failure_result(
    exc: _M39Failure,
    optimizer: _m36.CertifiedGlobalWorkCounters,
    oracle: ThreePlayerCertifiedGlobalOracleWorkCounters,
) -> ThreePlayerCertifiedGlobalResult:
    return ThreePlayerCertifiedGlobalResult(
        status=exc.status,
        payload=None,
        error=ThreePlayerCertifiedGlobalError(
            exc.phase, exc.message, exc.cause_status
        ),
        optimizer_work_counters=optimizer,
        oracle_work_counters=oracle,
        partial_result=False,
    )


def _analyze(inputs: _PreparedInputs) -> ThreePlayerCertifiedGlobalResult:
    counters = _MutableOracleCounters()
    optimizer_counters = _empty_optimizer_counters()
    try:
        preparation, oracle, _pins, _analysis_identity = _prepare(
            inputs, counters
        )
        native = _m36.optimize_certified_global_hero_commitment(
            preparation.m36_scenario,
            preparation.baseline_policy,
            oracle,
            absolute_gap_tolerance=inputs.absolute_gap_tolerance,
            relative_gap_tolerance=inputs.relative_gap_tolerance,
            limits=inputs.optimizer_limits,
            pins=inputs.optimizer_pins,
        )
        optimizer_counters = native.work_counters
        if (
            native.status
            not in (_m36.CERTIFIED_GLOBAL, _m36.CERTIFIED_EPSILON_GLOBAL)
            or native.payload is None
            or native.error is not None
            or native.partial_result
        ):
            message = (
                native.status
                if native.error is None
                else f"{native.error.phase}: {native.error.message}"
            )
            raise _M39Failure(
                native.status,
                "optimizer",
                _clean_message(message, "M36 optimizer failure"),
                cause_status=native.status,
            )
        baseline_point = oracle.point_record(
            native.payload.baseline_evaluation.policy_identity
        )
        if baseline_point is None:
            raise _M39Failure(
                _m36.INVALID_ORACLE_CONTRACT,
                "optimizer",
                "baseline point record is missing",
            )
        selected_point = (
            None
            if native.payload.selected_policy_identity is None
            else oracle.point_record(native.payload.selected_policy_identity)
        )
        if (
            native.payload.selected_policy_identity is not None
            and selected_point is None
        ):
            raise _M39Failure(
                _m36.INVALID_ORACLE_CONTRACT,
                "optimizer",
                "selected point record is missing",
            )
        oracle_snapshot = oracle.work_counters()
        payload = ThreePlayerCertifiedGlobalPayload(
            preparation=preparation,
            baseline_point=baseline_point,
            selected_point=selected_point,
            native_optimizer_result=native,
            oracle_work_counters=oracle_snapshot,
        )
        provisional = ThreePlayerCertifiedGlobalResult(
            status=native.status,
            payload=payload,
            error=None,
            optimizer_work_counters=optimizer_counters,
            oracle_work_counters=oracle_snapshot,
            partial_result=False,
        )
        try:
            _preflight_success_output(
                provisional,
                max_records=inputs.integration_limits.max_output_records,
                max_bytes=inputs.integration_limits.max_output_bytes,
            )
        except _OutputLimitReached as exc:
            raise _M39Failure(
                _m36.LIMIT_REACHED_NO_CERTIFICATE,
                "output",
                str(exc),
            ) from None
        return provisional
    except _M39Failure as exc:
        return _failure_result(
            exc, optimizer_counters, counters.snapshot()
        )
    except AiofContractError as exc:
        mapping = {
            AiofStatus.CAP_EXCEEDED: _m36.LIMIT_REACHED_NO_CERTIFICATE,
            AiofStatus.UNSUPPORTED_MODEL: _m36.UNSUPPORTED_DOMAIN,
            AiofStatus.NUMERIC_FAILURE: _m36.NUMERIC_FAILURE,
        }
        return _failure_result(
            _M39Failure(
                mapping.get(exc.status, _m36.INVALID_INPUT),
                "real_card.preparation",
                _clean_message(exc, "M35 preparation failure"),
                cause_status=exc.status.value,
            ),
            optimizer_counters,
            counters.snapshot(),
        )
    except Exception:
        return _failure_result(
            _M39Failure(
                _m36.INTERNAL_FAILURE,
                "internal",
                "unexpected M39 three-player certified-global failure",
            ),
            optimizer_counters,
            counters.snapshot(),
        )


def analyze_abstract_three_player_certified_global(
    request: AbstractThreePlayerCertifiedGlobalRequest,
) -> ThreePlayerCertifiedGlobalResult:
    """Certify one abstract M31 repeated Hero-policy scalar objective.

    The complete legal Hero policy domain is derived from the scenario.  Each
    evaluated non-baseline point receives a fresh complete M30 O1/O2
    non-cooperative response; only correspondence-wide Hero-worst enters the
    repeated objective.  Failure returns no payload or partial certificate.
    """

    try:
        return _analyze(_abstract_inputs(request))
    except _M39Failure as exc:
        return _failure_result(
            exc, _empty_optimizer_counters(),
            ThreePlayerCertifiedGlobalOracleWorkCounters(),
        )


def prepare_abstract_three_player_certified_global_oracle(
    request: AbstractThreePlayerCertifiedGlobalRequest,
) -> ThreePlayerCertifiedGlobalPreparation:
    """Prepare the canonical abstract M39 domain and conforming exact oracle.

    Controlled contract failures are raised because this preparation API
    returns no public result wrapper.  The analysis API converts the same
    failures into fail-closed result statuses.
    """

    counters = _MutableOracleCounters()
    inputs = _abstract_inputs(request)
    preparation, _oracle, _pins, _analysis_identity = _prepare(inputs, counters)
    return preparation


def analyze_known_board_real_card_three_player_certified_global(
    request: KnownBoardRealCardThreePlayerCertifiedGlobalRequest,
) -> ThreePlayerCertifiedGlobalResult:
    """Certify the M35 known-board real-card three-player scalar objective.

    M35 ordered triple conditioning, blockers, bucket mapping, showdown,
    rake/cap, uncalled excess, and terminal conservation are reconstructed
    exactly before M36 is invoked.  M32 finite candidates never constrain the
    search domain.
    """

    try:
        if type(request) is not KnownBoardRealCardThreePlayerCertifiedGlobalRequest:
            raise _M39Failure(
                _m36.INVALID_INPUT,
                "request",
                "real-card request has the wrong type",
            )
        return _analyze(_real_card_inputs(request))
    except _M39Failure as exc:
        return _failure_result(
            exc, _empty_optimizer_counters(),
            ThreePlayerCertifiedGlobalOracleWorkCounters(),
        )
    except AiofContractError as exc:
        return _failure_result(
            _M39Failure(
                _m36.INVALID_INPUT,
                "real_card.preparation",
                _clean_message(exc, "M35 preparation failure"),
                cause_status=exc.status.value,
            ),
            _empty_optimizer_counters(),
            ThreePlayerCertifiedGlobalOracleWorkCounters(),
        )


def prepare_known_board_real_card_three_player_certified_global_oracle(
    request: KnownBoardRealCardThreePlayerCertifiedGlobalRequest,
) -> ThreePlayerCertifiedGlobalPreparation:
    """Prepare the exact M35-derived M39 domain and conforming oracle."""

    if type(request) is not KnownBoardRealCardThreePlayerCertifiedGlobalRequest:
        raise TypeError("request has the wrong type")
    counters = _MutableOracleCounters()
    inputs = _real_card_inputs(request)
    preparation, _oracle, _pins, _analysis_identity = _prepare(inputs, counters)
    return preparation


def exact_three_player_certified_global_json(
    result: ThreePlayerCertifiedGlobalResult,
) -> str:
    """Serialize one M39 result as strict deterministic one-line JSON."""

    if type(result) is not ThreePlayerCertifiedGlobalResult:
        raise TypeError("result must be ThreePlayerCertifiedGlobalResult")
    return _canonical_json_bytes(result.to_dict()).decode("utf-8")
