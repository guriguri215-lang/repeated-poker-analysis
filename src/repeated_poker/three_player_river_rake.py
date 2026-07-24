"""Exact bounded three-player river/rake scenario response evaluator.

This isolated in-memory adapter evaluates a caller-declared abstract one-street
river tree with exact rational arithmetic.  Hero is fixed at every Hero
information set, O1 and O2 are the only response players, and ``R`` is a
non-strategic rake account.  The adapter enumerates the complete O1/O2 pure-plan
rectangle, verifies every payoff row with an independently dispatched terminal
path enumerator, and passes the lossless table to
``repeated_poker.three_player_response``.

The module deliberately does not evaluate real cards, generate Hero
candidates, model repeated play, support side pots/all-ins/odd-chip rounding,
or expose a package-level/public workflow.  Every failure is fail-closed:
scenario evaluation, payoff table, and response are all absent, and partial,
truncated, sampled, normalized, clamped, or fallback results are never
returned.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from dataclasses import asdict, dataclass, fields, replace
from fractions import Fraction
from typing import Any, Literal, Mapping, Sequence, TypeAlias

import repeated_poker.three_player_response as _m30
from repeated_poker.three_player_response import (
    ExactPayoffRow,
    ExactReducedGame,
    ExactResponseLimits,
    ResponseIdentityPins,
)


CONTRACT_VERSION = "m31-three-player-river-rake-scenario-response-v1"
ACCOUNTING_VERSION = "three-player-river-rake-net-chips-exact-v1"
STRUCTURAL_EVIDENCE_VERSION = "m31-machine-structural-recall-evidence-v1"
INDEPENDENT_EVALUATOR_VERSION = "m31-independent-terminal-path-enumerator-v1"

EXACT_SCENARIO_RESPONSE_COMPLETE = "EXACT_SCENARIO_RESPONSE_COMPLETE"
INVALID_INPUT = "INVALID_INPUT"
UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"
STALE_INPUT = "STALE_INPUT"
CAP_EXCEEDED = "CAP_EXCEEDED"
NUMERIC_FAILURE = "NUMERIC_FAILURE"
M30_RESPONSE_FAILURE = "M30_RESPONSE_FAILURE"
INTERNAL_FAILURE = "INTERNAL_FAILURE"

PLAYERS = ("H", "O1", "O2")
ALL_ACCOUNTS = ("H", "O1", "O2", "R")
ACTION_KINDS = ("check", "bet", "call", "raise", "fold")

DEFAULT_MAX_NODES = 200
DEFAULT_MAX_TERMINALS = 128
DEFAULT_MAX_FIXED_HERO_INFO_SETS = 12
DEFAULT_MAX_INFO_SETS_PER_OPPONENT = 12
DEFAULT_MAX_OPPONENT_INFO_SETS_TOTAL = 24
DEFAULT_MAX_ACTIONS_PER_INFO_SET = 4
DEFAULT_MAX_CHANCE_OUTCOMES = 16
DEFAULT_MAX_PRIVATE_OBSERVATION_RECORDS = 64
DEFAULT_MAX_TERMINAL_CONTRIBUTION_RECORDS = 384
DEFAULT_MAX_PURE_PLANS_O1 = 4
DEFAULT_MAX_PURE_PLANS_O2 = 4
DEFAULT_MAX_JOINT_PURE_PROFILES = 16
DEFAULT_MAX_TERMINAL_EVALUATIONS = 2_048
DEFAULT_MAX_PAYOFF_ROWS = 16
DEFAULT_MAX_RATIONAL_NUMERATOR_BITS = 256
DEFAULT_MAX_RATIONAL_DENOMINATOR_BITS = 256
DEFAULT_MAX_IDENTITY_RECORDS = 10_000
DEFAULT_MAX_OUTPUT_RECORDS = 50_000
DEFAULT_MAX_OUTPUT_BYTES = 4_000_000

HARD_MAX_NODES = 500
HARD_MAX_TERMINALS = 256
HARD_MAX_FIXED_HERO_INFO_SETS = 16
HARD_MAX_INFO_SETS_PER_OPPONENT = 16
HARD_MAX_OPPONENT_INFO_SETS_TOTAL = 32
HARD_MAX_ACTIONS_PER_INFO_SET = 4
HARD_MAX_CHANCE_OUTCOMES = 32
HARD_MAX_PRIVATE_OBSERVATION_RECORDS = 256
HARD_MAX_TERMINAL_CONTRIBUTION_RECORDS = 768
HARD_MAX_PURE_PLANS_O1 = 6
HARD_MAX_PURE_PLANS_O2 = 6
HARD_MAX_JOINT_PURE_PROFILES = 36
HARD_MAX_TERMINAL_EVALUATIONS = 9_216
HARD_MAX_PAYOFF_ROWS = 36
HARD_MAX_RATIONAL_NUMERATOR_BITS = 1_024
HARD_MAX_RATIONAL_DENOMINATOR_BITS = 1_024
HARD_MAX_IDENTITY_RECORDS = 100_000
HARD_MAX_OUTPUT_RECORDS = 500_000
HARD_MAX_OUTPUT_BYTES = 32_000_000

_INTEGER_RE = re.compile(r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)")
_FRACTION_RE = re.compile(r"(-?[1-9][0-9]*)/([1-9][0-9]*)")
_IDENTITY_RE = re.compile(r"[0-9a-f]{64}")
_DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


@dataclass(frozen=True)
class RiverObservation:
    """Public and player-private abstract river observation identities."""

    public_observation_id: str
    private_observation_id_by_player: Mapping[str, str]


@dataclass(frozen=True)
class AwardShare:
    """One active showdown winner and its exact pot share."""

    player_id: str
    share: str


@dataclass(frozen=True)
class RiverAction:
    """One ordered public action edge."""

    action_id: str
    kind: str
    target_total_contribution: str | None
    child: "RiverNode"


@dataclass(frozen=True)
class RiverDecisionNode:
    """A decision node owned explicitly by H, O1, or O2."""

    node_id: str
    owner: str
    information_set_id: str
    actions: tuple[RiverAction, ...]


@dataclass(frozen=True)
class RiverTerminalNode:
    """A fold or caller-declared abstract showdown terminal."""

    node_id: str
    kind: str
    award_shares: tuple[AwardShare, ...] = ()


@dataclass(frozen=True)
class RiverChanceOutcome:
    """One ordered root chance outcome."""

    outcome_id: str
    probability: str
    observation: RiverObservation
    child: "RiverNode"


@dataclass(frozen=True)
class RiverChanceNode:
    """The optional, root-only exact chance distribution."""

    node_id: str
    outcomes: tuple[RiverChanceOutcome, ...]


RiverNode: TypeAlias = RiverDecisionNode | RiverTerminalNode | RiverChanceNode


@dataclass(frozen=True)
class ThreePlayerRiverRakeScenario:
    """Strict abstract exact one-street river/rake scenario."""

    root: RiverNode
    button_player_id: str
    seat_order: tuple[str, ...]
    river_action_order: tuple[str, ...]
    initial_observation: RiverObservation | None
    initial_pot: str
    initial_contribution: Mapping[str, str]
    max_total_contribution: Mapping[str, str]
    rake_rate: str
    rake_cap: str | None = None
    rounding_unit: str | None = None
    rounding_rule: str = "exact_none"
    accounting_version: str = ACCOUNTING_VERSION
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class ExactBehaviorPolicy:
    """Complete exact behavior policy keyed by information set and action."""

    probabilities: Mapping[str, Mapping[str, str]]


@dataclass(frozen=True)
class OpponentInitialProfile:
    """Optional complete O1 and O2 pre-adaptation behavior profile."""

    o1_probabilities: Mapping[str, Mapping[str, str]] | None
    o2_probabilities: Mapping[str, Mapping[str, str]] | None


@dataclass(frozen=True)
class PerfectRecallAttestation:
    """Human-traceable confirmation bound to tree and machine evidence."""

    tree_content_identity: str
    structural_evidence_identity: str
    o1_confirmed: bool
    o2_confirmed: bool
    verifier: str
    verification_date: str
    evidence_version: str


@dataclass(frozen=True)
class RiverRakeLimits:
    """Caller-lowerable M31 bounds with immutable hard ceilings."""

    max_nodes: int = DEFAULT_MAX_NODES
    max_terminals: int = DEFAULT_MAX_TERMINALS
    max_fixed_hero_info_sets: int = DEFAULT_MAX_FIXED_HERO_INFO_SETS
    max_info_sets_per_opponent: int = DEFAULT_MAX_INFO_SETS_PER_OPPONENT
    max_opponent_info_sets_total: int = DEFAULT_MAX_OPPONENT_INFO_SETS_TOTAL
    max_actions_per_info_set: int = DEFAULT_MAX_ACTIONS_PER_INFO_SET
    max_chance_outcomes: int = DEFAULT_MAX_CHANCE_OUTCOMES
    max_private_observation_records: int = (
        DEFAULT_MAX_PRIVATE_OBSERVATION_RECORDS
    )
    max_terminal_contribution_records: int = (
        DEFAULT_MAX_TERMINAL_CONTRIBUTION_RECORDS
    )
    max_pure_plans_o1: int = DEFAULT_MAX_PURE_PLANS_O1
    max_pure_plans_o2: int = DEFAULT_MAX_PURE_PLANS_O2
    max_joint_pure_profiles: int = DEFAULT_MAX_JOINT_PURE_PROFILES
    max_terminal_evaluations: int = DEFAULT_MAX_TERMINAL_EVALUATIONS
    max_payoff_rows: int = DEFAULT_MAX_PAYOFF_ROWS
    max_rational_numerator_bits: int = DEFAULT_MAX_RATIONAL_NUMERATOR_BITS
    max_rational_denominator_bits: int = DEFAULT_MAX_RATIONAL_DENOMINATOR_BITS
    max_identity_records: int = DEFAULT_MAX_IDENTITY_RECORDS
    max_output_records: int = DEFAULT_MAX_OUTPUT_RECORDS
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES


@dataclass(frozen=True)
class RiverRakeIdentityPins:
    """Optional M31 source identities whose mismatch is stale input."""

    scenario_identity: str | None = None
    tree_structure_identity: str | None = None
    fixed_hero_identity: str | None = None
    initial_profile_identity: str | None = None
    perfect_recall_evidence_identity: str | None = None
    rake_convention_identity: str | None = None
    payoff_table_identity: str | None = None
    evaluator_config_identity: str | None = None


@dataclass(frozen=True)
class ScenarioResponseError:
    """Bounded error metadata without any partial payload."""

    phase: str
    message: str
    cause_status: str | None = None


@dataclass(frozen=True)
class ScenarioResponseResult:
    """Exclusive complete-success or fail-closed outer result."""

    status: str
    scenario_evaluation: dict[str, Any] | None
    payoff_table: dict[str, Any] | None
    response: dict[str, Any] | None
    error: ScenarioResponseError | None
    partial_result: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly result."""

        return {
            "status": self.status,
            "scenario_evaluation": self.scenario_evaluation,
            "payoff_table": self.payoff_table,
            "response": self.response,
            "error": None if self.error is None else asdict(self.error),
            "partial_result": self.partial_result,
        }


@dataclass(frozen=True)
class _Utility:
    H: Fraction
    O1: Fraction
    O2: Fraction
    R: Fraction

    def component(self, name: str) -> Fraction:
        return getattr(self, name)


@dataclass(frozen=True)
class _ParsedScenario:
    initial_pot: Fraction
    initial_contribution: dict[str, Fraction]
    max_total_contribution: dict[str, Fraction]
    rake_rate: Fraction
    rake_cap: Fraction | None


@dataclass(frozen=True)
class _PathState:
    active: tuple[str, ...]
    contributions: dict[str, Fraction]
    amount_to_call: Fraction | None
    pending: tuple[str, ...]
    raise_used: bool
    last_aggressor: str | None


@dataclass(frozen=True)
class _InfoSet:
    owner: str
    actions: tuple[str, ...]
    legal_signature: tuple[tuple[str, str, str | None], ...]


@dataclass
class _Structure:
    parsed: _ParsedScenario
    node_count: int
    terminal_count: int
    chance_outcome_count: int
    private_observation_records: int
    terminal_contribution_records: int
    info_sets: dict[str, dict[str, _InfoSet]]
    tree_projection: dict[str, Any]
    terminal_utilities: dict[str, _Utility]
    terminal_records: list[dict[str, Any]]
    chance_probabilities: dict[str, tuple[Fraction, ...]]
    occurrence_records: list[dict[str, Any]]
    structure_projection: dict[str, Any]
    evidence_projection: dict[str, Any]
    rake_projection: dict[str, Any]
    scenario_projection: dict[str, Any]
    scenario_identity: str
    tree_structure_identity: str
    structural_evidence_identity: str
    rake_convention_identity: str


class _ScenarioFailure(ValueError):
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
        self.cause_status = cause_status


class _IdentityRecordBudget:
    """Consume identity-owned records before projection or hash allocation."""

    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.used = 0

    def reserve(self, count: int = 1) -> None:
        if count < 0 or self.used > self.maximum - count:
            raise _ScenarioFailure(
                CAP_EXCEEDED,
                "identity",
                "max_identity_records exceeded before identity record allocation",
            )
        self.used += count


_HARD_LIMITS = {
    "max_nodes": HARD_MAX_NODES,
    "max_terminals": HARD_MAX_TERMINALS,
    "max_fixed_hero_info_sets": HARD_MAX_FIXED_HERO_INFO_SETS,
    "max_info_sets_per_opponent": HARD_MAX_INFO_SETS_PER_OPPONENT,
    "max_opponent_info_sets_total": HARD_MAX_OPPONENT_INFO_SETS_TOTAL,
    "max_actions_per_info_set": HARD_MAX_ACTIONS_PER_INFO_SET,
    "max_chance_outcomes": HARD_MAX_CHANCE_OUTCOMES,
    "max_private_observation_records": HARD_MAX_PRIVATE_OBSERVATION_RECORDS,
    "max_terminal_contribution_records": (
        HARD_MAX_TERMINAL_CONTRIBUTION_RECORDS
    ),
    "max_pure_plans_o1": HARD_MAX_PURE_PLANS_O1,
    "max_pure_plans_o2": HARD_MAX_PURE_PLANS_O2,
    "max_joint_pure_profiles": HARD_MAX_JOINT_PURE_PROFILES,
    "max_terminal_evaluations": HARD_MAX_TERMINAL_EVALUATIONS,
    "max_payoff_rows": HARD_MAX_PAYOFF_ROWS,
    "max_rational_numerator_bits": HARD_MAX_RATIONAL_NUMERATOR_BITS,
    "max_rational_denominator_bits": HARD_MAX_RATIONAL_DENOMINATOR_BITS,
    "max_identity_records": HARD_MAX_IDENTITY_RECORDS,
    "max_output_records": HARD_MAX_OUTPUT_RECORDS,
    "max_output_bytes": HARD_MAX_OUTPUT_BYTES,
}

_M30_HARD_LIMITS = {
    "max_pure_plans_o1": _m30.HARD_MAX_PURE_PLANS_O1,
    "max_pure_plans_o2": _m30.HARD_MAX_PURE_PLANS_O2,
    "max_joint_pure_profiles": _m30.HARD_MAX_JOINT_PURE_PROFILES,
    "max_support_pairs": _m30.HARD_MAX_SUPPORT_PAIRS,
    "max_support_size_o1": _m30.HARD_MAX_SUPPORT_SIZE_O1,
    "max_support_size_o2": _m30.HARD_MAX_SUPPORT_SIZE_O2,
    "max_exact_linear_systems": _m30.HARD_MAX_EXACT_LINEAR_SYSTEMS,
    "max_equilibrium_support_cells": (
        _m30.HARD_MAX_EQUILIBRIUM_SUPPORT_CELLS
    ),
    "max_vertices_per_cell": _m30.HARD_MAX_VERTICES_PER_CELL,
    "max_total_vertices": _m30.HARD_MAX_TOTAL_VERTICES,
    "max_vertex_pairs_evaluated": _m30.HARD_MAX_VERTEX_PAIRS_EVALUATED,
    "max_rational_numerator_bits": _m30.HARD_MAX_RATIONAL_NUMERATOR_BITS,
    "max_rational_denominator_bits": _m30.HARD_MAX_RATIONAL_DENOMINATOR_BITS,
    "max_verifier_operations": _m30.HARD_MAX_VERIFIER_OPERATIONS,
    "max_output_records": _m30.HARD_MAX_OUTPUT_RECORDS,
    "max_output_bytes": _m30.HARD_MAX_OUTPUT_BYTES,
}


def _clean_text(value: str, maximum: int) -> str:
    return value.replace("\r", " ").replace("\n", " ")[:maximum]


def _failure(exc: _ScenarioFailure) -> ScenarioResponseResult:
    return ScenarioResponseResult(
        status=exc.status,
        scenario_evaluation=None,
        payoff_table=None,
        response=None,
        error=ScenarioResponseError(
            phase=_clean_text(exc.phase, 64),
            message=_clean_text(str(exc), 500),
            cause_status=exc.cause_status,
        ),
        partial_result=False,
    )


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


def _bounded_text(value: Any, phase: str, *, maximum: int = 128) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        raise _ScenarioFailure(
            INVALID_INPUT, phase, f"text must contain 1..{maximum} characters"
        )
    if any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in value):
        raise _ScenarioFailure(INVALID_INPUT, phase, "control characters are forbidden")
    return value


def _validate_identity(value: Any, phase: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if type(value) is not str or _IDENTITY_RE.fullmatch(value) is None:
        raise _ScenarioFailure(
            INVALID_INPUT,
            phase,
            "identity must be a lowercase 64-character SHA-256 string",
        )


def _validate_limits(limits: RiverRakeLimits) -> None:
    if type(limits) is not RiverRakeLimits:
        raise _ScenarioFailure(
            INVALID_INPUT, "limits", "limits must be RiverRakeLimits"
        )
    for field in fields(RiverRakeLimits):
        value = getattr(limits, field.name)
        hard = _HARD_LIMITS[field.name]
        if type(value) is not int or value < 1 or value > hard:
            raise _ScenarioFailure(
                INVALID_INPUT,
                "limits",
                f"{field.name} must be an int in [1,{hard}]",
            )


def _validate_m30_limits(limits: ExactResponseLimits) -> None:
    if type(limits) is not ExactResponseLimits:
        raise _ScenarioFailure(
            INVALID_INPUT,
            "m30_limits",
            "m30_limits must be ExactResponseLimits",
        )
    for field in fields(ExactResponseLimits):
        value = getattr(limits, field.name)
        hard = _M30_HARD_LIMITS[field.name]
        if type(value) is not int or value < 1 or value > hard:
            raise _ScenarioFailure(
                INVALID_INPUT,
                "m30_limits",
                f"{field.name} must be an int in [1,{hard}]",
            )


def _validate_pins(pins: RiverRakeIdentityPins) -> None:
    if type(pins) is not RiverRakeIdentityPins:
        raise _ScenarioFailure(
            INVALID_INPUT, "identity_pins", "pins must be RiverRakeIdentityPins"
        )
    for field in fields(RiverRakeIdentityPins):
        _validate_identity(
            getattr(pins, field.name),
            f"identity_pins.{field.name}",
            optional=True,
        )


def _validate_response_pins(pins: ResponseIdentityPins) -> None:
    if type(pins) is not ResponseIdentityPins:
        raise _ScenarioFailure(
            INVALID_INPUT,
            "response_identity_pins",
            "response pins must be ResponseIdentityPins",
        )
    for field in fields(ResponseIdentityPins):
        _validate_identity(
            getattr(pins, field.name),
            f"response_identity_pins.{field.name}",
            optional=True,
        )


def _rational_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _check_bits(value: Fraction, limits: RiverRakeLimits, phase: str) -> Fraction:
    if abs(value.numerator).bit_length() > limits.max_rational_numerator_bits:
        raise _ScenarioFailure(
            CAP_EXCEEDED, phase, "max_rational_numerator_bits exceeded"
        )
    if value.denominator.bit_length() > limits.max_rational_denominator_bits:
        raise _ScenarioFailure(
            CAP_EXCEEDED, phase, "max_rational_denominator_bits exceeded"
        )
    return value


def _parse_rational(
    value: Any,
    phase: str,
    limits: RiverRakeLimits,
    *,
    minimum: Fraction | None = None,
    maximum: Fraction | None = None,
) -> Fraction:
    if type(value) is not str:
        raise _ScenarioFailure(
            INVALID_INPUT, phase, "value must be a canonical rational string"
        )
    if _INTEGER_RE.fullmatch(value) is not None:
        numerator_text, denominator_text = value, "1"
    else:
        match = _FRACTION_RE.fullmatch(value)
        if match is None:
            raise _ScenarioFailure(
                INVALID_INPUT, phase, "noncanonical rational string"
            )
        numerator_text, denominator_text = match.groups()
    maximum_num_digits = (
        math.ceil(limits.max_rational_numerator_bits * math.log10(2)) + 1
    )
    maximum_den_digits = (
        math.ceil(limits.max_rational_denominator_bits * math.log10(2)) + 1
    )
    if len(numerator_text.lstrip("-")) > maximum_num_digits:
        raise _ScenarioFailure(
            CAP_EXCEEDED, phase, "max_rational_numerator_bits exceeded"
        )
    if len(denominator_text) > maximum_den_digits:
        raise _ScenarioFailure(
            CAP_EXCEEDED, phase, "max_rational_denominator_bits exceeded"
        )
    numerator = int(numerator_text)
    denominator = int(denominator_text)
    if denominator != 1 and (
        numerator == 0 or math.gcd(abs(numerator), denominator) != 1
    ):
        raise _ScenarioFailure(
            INVALID_INPUT, phase, "fraction must be nonzero and reduced"
        )
    parsed = _check_bits(Fraction(numerator, denominator), limits, phase)
    if _rational_text(parsed) != value:
        raise _ScenarioFailure(
            INVALID_INPUT, phase, "rational string is not canonical"
        )
    if minimum is not None and parsed < minimum:
        raise _ScenarioFailure(INVALID_INPUT, phase, "value is below its minimum")
    if maximum is not None and parsed > maximum:
        raise _ScenarioFailure(INVALID_INPUT, phase, "value exceeds its maximum")
    return parsed


def _fadd(
    left: Fraction,
    right: Fraction,
    limits: RiverRakeLimits,
    phase: str,
) -> Fraction:
    return _check_bits(left + right, limits, phase)


def _fsub(
    left: Fraction,
    right: Fraction,
    limits: RiverRakeLimits,
    phase: str,
) -> Fraction:
    return _check_bits(left - right, limits, phase)


def _fmul(
    left: Fraction,
    right: Fraction,
    limits: RiverRakeLimits,
    phase: str,
) -> Fraction:
    return _check_bits(left * right, limits, phase)


def _checked_product(left: int, right: int, maximum: int, phase: str) -> int:
    if left < 0 or right < 0:
        raise _ScenarioFailure(INTERNAL_FAILURE, phase, "negative checked product")
    if right and left > maximum // right:
        raise _ScenarioFailure(
            CAP_EXCEEDED, phase, "checked product cap exceeded"
        )
    return left * right


def _checked_sum(left: int, right: int, maximum: int, phase: str) -> int:
    if left < 0 or right < 0 or left > maximum - right:
        raise _ScenarioFailure(CAP_EXCEEDED, phase, "checked sum cap exceeded")
    return left + right


def _validate_exact_player_mapping(
    value: Any,
    phase: str,
    *,
    value_name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _ScenarioFailure(INVALID_INPUT, phase, f"{value_name} must be a mapping")
    if set(value) != set(PLAYERS):
        raise _ScenarioFailure(
            INVALID_INPUT, phase, f"{value_name} must have exactly H/O1/O2 keys"
        )
    return {player: value[player] for player in PLAYERS}


def _validate_observation(
    observation: Any,
    phase: str,
) -> dict[str, Any]:
    if type(observation) is not RiverObservation:
        raise _ScenarioFailure(
            INVALID_INPUT, phase, "observation must be RiverObservation"
        )
    public = _bounded_text(
        observation.public_observation_id,
        f"{phase}.public_observation_id",
    )
    private = _validate_exact_player_mapping(
        observation.private_observation_id_by_player,
        f"{phase}.private_observation_id_by_player",
        value_name="private observation map",
    )
    return {
        "public_observation_id": public,
        "private_observation_id_by_player": {
            player: _bounded_text(
                private[player], f"{phase}.private_observation_id_by_player.{player}"
            )
            for player in PLAYERS
        },
    }


def _utility_record(utility: _Utility) -> dict[str, str]:
    return {name: _rational_text(utility.component(name)) for name in ALL_ACCOUNTS}


def _zero_utility() -> _Utility:
    return _Utility(Fraction(0), Fraction(0), Fraction(0), Fraction(0))


def _add_utility(
    left: _Utility,
    right: _Utility,
    limits: RiverRakeLimits,
    phase: str,
) -> _Utility:
    return _Utility(
        **{
            name: _fadd(
                left.component(name), right.component(name), limits, f"{phase}.{name}"
            )
            for name in ALL_ACCOUNTS
        }
    )


def _scale_utility(
    probability: Fraction,
    utility: _Utility,
    limits: RiverRakeLimits,
    phase: str,
) -> _Utility:
    return _Utility(
        **{
            name: _fmul(
                probability,
                utility.component(name),
                limits,
                f"{phase}.{name}",
            )
            for name in ALL_ACCOUNTS
        }
    )


def _require_conservation(
    utility: _Utility, limits: RiverRakeLimits, phase: str
) -> None:
    total = Fraction(0)
    for name in ALL_ACCOUNTS:
        total = _fadd(total, utility.component(name), limits, phase)
    if total != 0:
        raise _ScenarioFailure(
            NUMERIC_FAILURE, phase, "exact utility conservation mismatch"
        )


def _cyclic_after(
    actor: str, active: Sequence[str], action_order: tuple[str, ...]
) -> tuple[str, ...]:
    active_set = set(active)
    start = action_order.index(actor)
    return tuple(
        action_order[(start + offset) % len(action_order)]
        for offset in range(1, len(action_order) + 1)
        if action_order[(start + offset) % len(action_order)] in active_set
        and action_order[(start + offset) % len(action_order)] != actor
    )


def _transition(
    state: _PathState,
    actor: str,
    action: RiverAction,
    target: Fraction | None,
    action_order: tuple[str, ...],
    maximums: Mapping[str, Fraction],
) -> tuple[_PathState, str | None]:
    if not state.pending or state.pending[0] != actor:
        raise _ScenarioFailure(
            INVALID_INPUT, "tree", "decision owner is not the expected next actor"
        )
    active = list(state.active)
    contributions = dict(state.contributions)
    pending = list(state.pending)
    amount_to_call = state.amount_to_call
    raise_used = state.raise_used
    last_aggressor = state.last_aggressor
    kind = action.kind

    if kind in ("check", "fold") and target is not None:
        raise _ScenarioFailure(
            INVALID_INPUT,
            "tree.actions",
            "check/fold target_total_contribution must be null",
        )
    if kind in ("bet", "call", "raise") and target is None:
        raise _ScenarioFailure(
            INVALID_INPUT,
            "tree.actions",
            "bet/call/raise target_total_contribution is required",
        )

    if kind == "check":
        if amount_to_call is not None:
            raise _ScenarioFailure(
                INVALID_INPUT, "tree.actions", "check is illegal when facing a wager"
            )
        pending.pop(0)
    elif kind == "bet":
        if amount_to_call is not None:
            raise _ScenarioFailure(
                INVALID_INPUT, "tree.actions", "bet is illegal when facing a wager"
            )
        assert target is not None
        if target <= contributions[actor]:
            raise _ScenarioFailure(
                INVALID_INPUT,
                "tree.actions",
                "bet target must strictly increase actor contribution",
            )
        contributions[actor] = target
        amount_to_call = target
        last_aggressor = actor
        pending = list(_cyclic_after(actor, active, action_order))
    elif kind == "call":
        if amount_to_call is None:
            raise _ScenarioFailure(
                INVALID_INPUT, "tree.actions", "call requires a wager"
            )
        assert target is not None
        if target != amount_to_call or target <= contributions[actor]:
            raise _ScenarioFailure(
                INVALID_INPUT,
                "tree.actions",
                "call target must exactly equal the current amount to call",
            )
        contributions[actor] = target
        pending.pop(0)
    elif kind == "raise":
        if amount_to_call is None:
            raise _ScenarioFailure(
                INVALID_INPUT, "tree.actions", "raise requires a wager"
            )
        if raise_used:
            raise _ScenarioFailure(
                UNSUPPORTED_MODEL,
                "tree.actions",
                "re-raise is unsupported",
            )
        assert target is not None
        if target <= amount_to_call or target <= contributions[actor]:
            raise _ScenarioFailure(
                INVALID_INPUT,
                "tree.actions",
                "raise target must strictly exceed the current wager",
            )
        contributions[actor] = target
        amount_to_call = target
        raise_used = True
        last_aggressor = actor
        pending = list(_cyclic_after(actor, active, action_order))
    elif kind == "fold":
        if amount_to_call is None:
            raise _ScenarioFailure(
                INVALID_INPUT, "tree.actions", "fold requires a wager"
            )
        active.remove(actor)
        pending = [player for player in pending[1:] if player in active]
    else:
        raise _ScenarioFailure(INVALID_INPUT, "tree.actions", "unknown action kind")

    if target is not None and target >= maximums[actor]:
        raise _ScenarioFailure(
            UNSUPPORTED_MODEL,
            "tree.actions",
            "all-in or stack-depleting contribution is unsupported",
        )
    terminal_kind: str | None = None
    if len(active) == 1:
        terminal_kind = "fold"
    elif not pending:
        terminal_kind = "showdown"
    return (
        _PathState(
            active=tuple(active),
            contributions=contributions,
            amount_to_call=amount_to_call,
            pending=tuple(pending),
            raise_used=raise_used,
            last_aggressor=last_aggressor,
        ),
        terminal_kind,
    )


def _terminal_accounting(
    node: RiverTerminalNode,
    state: _PathState,
    parsed: _ParsedScenario,
    limits: RiverRakeLimits,
) -> tuple[_Utility, dict[str, Any]]:
    if node.kind not in ("fold", "showdown"):
        raise _ScenarioFailure(
            INVALID_INPUT, "terminals", "terminal kind must be fold or showdown"
        )
    if type(node.award_shares) is not tuple:
        raise _ScenarioFailure(
            INVALID_INPUT, "terminals", "award_shares must be an ordered tuple"
        )
    gross = dict(state.contributions)
    maximum = max(gross.values())
    maximum_players = [player for player in PLAYERS if gross[player] == maximum]
    returns = {player: Fraction(0) for player in PLAYERS}
    if len(maximum_players) == 1:
        maximum_player = maximum_players[0]
        second = max(gross[player] for player in PLAYERS if player != maximum_player)
        returns[maximum_player] = _fsub(
            maximum, second, limits, f"terminal.{node.node_id}.uncalled_return"
        )
    kept = {
        player: _fsub(
            gross[player],
            returns[player],
            limits,
            f"terminal.{node.node_id}.kept_contribution",
        )
        for player in PLAYERS
    }
    pot_before_rake = Fraction(0)
    for player in PLAYERS:
        pot_before_rake = _fadd(
            pot_before_rake,
            kept[player],
            limits,
            f"terminal.{node.node_id}.pot_before_rake",
        )

    awards = {player: Fraction(0) for player in PLAYERS}
    shares: dict[str, Fraction] = {}
    if node.kind == "fold":
        if len(state.active) != 1:
            raise _ScenarioFailure(
                INVALID_INPUT, "terminals", "fold terminal requires one active player"
            )
        if node.award_shares:
            raise _ScenarioFailure(
                INVALID_INPUT, "terminals", "fold terminal must not declare shares"
            )
        rake = Fraction(0)
        awards[state.active[0]] = pot_before_rake
    else:
        if len(state.active) not in (2, 3):
            raise _ScenarioFailure(
                INVALID_INPUT,
                "terminals",
                "showdown requires two or three active players",
            )
        active_kept = {kept[player] for player in state.active}
        if len(active_kept) != 1:
            raise _ScenarioFailure(
                UNSUPPORTED_MODEL,
                "terminals",
                "showdown requires a side pot after uncalled return",
            )
        if not node.award_shares:
            raise _ScenarioFailure(
                INVALID_INPUT, "terminals", "showdown shares are required"
            )
        share_total = Fraction(0)
        for index, award in enumerate(node.award_shares):
            if type(award) is not AwardShare:
                raise _ScenarioFailure(
                    INVALID_INPUT, "terminals", "award share must be AwardShare"
                )
            player = _bounded_text(
                award.player_id, f"terminals.{node.node_id}.shares[{index}].player"
            )
            if player not in state.active or player in shares:
                raise _ScenarioFailure(
                    INVALID_INPUT,
                    "terminals",
                    "showdown winner must be unique and active",
                )
            share = _parse_rational(
                award.share,
                f"terminals.{node.node_id}.shares[{index}].share",
                limits,
                minimum=Fraction(0),
                maximum=Fraction(1),
            )
            if share <= 0:
                raise _ScenarioFailure(
                    INVALID_INPUT, "terminals", "award share must be positive"
                )
            shares[player] = share
            share_total = _fadd(
                share_total, share, limits, f"terminal.{node.node_id}.share_total"
            )
        if share_total != 1:
            raise _ScenarioFailure(
                INVALID_INPUT, "terminals", "showdown shares must sum exactly to one"
            )
        raw_rake = _fmul(
            parsed.rake_rate,
            pot_before_rake,
            limits,
            f"terminal.{node.node_id}.raw_rake",
        )
        rake = (
            raw_rake
            if parsed.rake_cap is None
            else min(raw_rake, parsed.rake_cap)
        )
        pot_after_rake = _fsub(
            pot_before_rake,
            rake,
            limits,
            f"terminal.{node.node_id}.pot_after_rake",
        )
        for player, share in shares.items():
            awards[player] = _fmul(
                pot_after_rake,
                share,
                limits,
                f"terminal.{node.node_id}.award.{player}",
            )

    pot_after_rake = _fsub(
        pot_before_rake,
        rake,
        limits,
        f"terminal.{node.node_id}.pot_after_rake",
    )
    player_utility = {}
    for player in PLAYERS:
        received = _fadd(
            awards[player],
            returns[player],
            limits,
            f"terminal.{node.node_id}.received.{player}",
        )
        player_utility[player] = _fsub(
            received,
            gross[player],
            limits,
            f"terminal.{node.node_id}.utility.{player}",
        )
    utility = _Utility(
        H=player_utility["H"],
        O1=player_utility["O1"],
        O2=player_utility["O2"],
        R=rake,
    )
    _require_conservation(utility, limits, f"terminal.{node.node_id}.conservation")
    record = {
        "node_id": node.node_id,
        "kind": node.kind,
        "active_players": list(state.active),
        "gross_contribution": {
            player: _rational_text(gross[player]) for player in PLAYERS
        },
        "uncalled_return": {
            player: _rational_text(returns[player]) for player in PLAYERS
        },
        "kept_contribution": {
            player: _rational_text(kept[player]) for player in PLAYERS
        },
        "pot_before_rake": _rational_text(pot_before_rake),
        "rake_amount": _rational_text(rake),
        "pot_after_rake": _rational_text(pot_after_rake),
        "award_share": {
            player: _rational_text(shares[player]) for player in sorted(shares)
        },
        "awarded": {
            player: _rational_text(awards[player]) for player in PLAYERS
        },
        "utility": _utility_record(utility),
        "conservation": "0",
    }
    return utility, record


def _parse_scenario(
    scenario: ThreePlayerRiverRakeScenario,
    limits: RiverRakeLimits,
) -> _ParsedScenario:
    if type(scenario) is not ThreePlayerRiverRakeScenario:
        raise _ScenarioFailure(
            INVALID_INPUT,
            "scenario",
            "scenario must be ThreePlayerRiverRakeScenario",
        )
    if scenario.contract_version != CONTRACT_VERSION:
        raise _ScenarioFailure(
            UNSUPPORTED_MODEL, "contract_version", "unsupported contract version"
        )
    if scenario.accounting_version != ACCOUNTING_VERSION:
        raise _ScenarioFailure(
            UNSUPPORTED_MODEL,
            "accounting_version",
            "unsupported accounting version",
        )
    if scenario.rounding_unit is not None or scenario.rounding_rule != "exact_none":
        raise _ScenarioFailure(
            UNSUPPORTED_MODEL,
            "rounding",
            "odd-chip or discrete rounding is unsupported",
        )
    if scenario.button_player_id not in PLAYERS:
        raise _ScenarioFailure(INVALID_INPUT, "button", "invalid button player")
    for name, order in (
        ("seat_order", scenario.seat_order),
        ("river_action_order", scenario.river_action_order),
    ):
        if type(order) is not tuple or len(order) != 3 or set(order) != set(PLAYERS):
            raise _ScenarioFailure(
                INVALID_INPUT, name, f"{name} must contain H/O1/O2 exactly once"
            )
    initial_raw = _validate_exact_player_mapping(
        scenario.initial_contribution,
        "initial_contribution",
        value_name="initial contribution",
    )
    maximum_raw = _validate_exact_player_mapping(
        scenario.max_total_contribution,
        "max_total_contribution",
        value_name="maximum contribution",
    )
    initial = {
        player: _parse_rational(
            initial_raw[player],
            f"initial_contribution.{player}",
            limits,
            minimum=Fraction(0),
        )
        for player in PLAYERS
    }
    maximums = {
        player: _parse_rational(
            maximum_raw[player],
            f"max_total_contribution.{player}",
            limits,
            minimum=Fraction(0),
        )
        for player in PLAYERS
    }
    for player in PLAYERS:
        if initial[player] >= maximums[player]:
            raise _ScenarioFailure(
                UNSUPPORTED_MODEL,
                f"max_total_contribution.{player}",
                "initial all-in or exhausted stack is unsupported",
            )
    initial_pot = _parse_rational(
        scenario.initial_pot, "initial_pot", limits, minimum=Fraction(0)
    )
    contribution_sum = Fraction(0)
    for player in PLAYERS:
        contribution_sum = _fadd(
            contribution_sum, initial[player], limits, "initial_pot"
        )
    if contribution_sum != initial_pot:
        raise _ScenarioFailure(
            INVALID_INPUT,
            "initial_pot",
            "initial pot must exactly equal H/O1/O2 initial contributions",
        )
    rake_rate = _parse_rational(
        scenario.rake_rate,
        "rake_rate",
        limits,
        minimum=Fraction(0),
        maximum=Fraction(1),
    )
    rake_cap = (
        None
        if scenario.rake_cap is None
        else _parse_rational(
            scenario.rake_cap,
            "rake_cap",
            limits,
            minimum=Fraction(0),
        )
    )
    return _ParsedScenario(
        initial_pot=initial_pot,
        initial_contribution=initial,
        max_total_contribution=maximums,
        rake_rate=rake_rate,
        rake_cap=rake_cap,
    )


def _prepare_structure(
    scenario: ThreePlayerRiverRakeScenario,
    limits: RiverRakeLimits,
    identity_budget: _IdentityRecordBudget,
) -> _Structure:
    # Fixed schema/version/identity slots are part of every identity projection.
    # Reserve them before parsing or appending any identity-owned tree record so
    # a caller-lowered cap cannot permit projection materialization first.
    identity_budget.reserve(64)
    parsed = _parse_scenario(scenario, limits)
    info_sets: dict[str, dict[str, _InfoSet]] = {
        "H": {},
        "O1": {},
        "O2": {},
    }
    owner_by_information_set: dict[str, str] = {}
    occurrence_signatures: dict[str, tuple[Any, ...]] = {}
    occurrence_records: list[dict[str, Any]] = []
    terminal_utilities: dict[str, _Utility] = {}
    terminal_records: list[dict[str, Any]] = []
    chance_probabilities: dict[str, tuple[Fraction, ...]] = {}
    seen_node_ids: set[str] = set()
    seen_objects: set[int] = set()
    node_count = 0
    terminal_count = 0

    if isinstance(scenario.root, RiverChanceNode):
        if scenario.initial_observation is not None:
            raise _ScenarioFailure(
                INVALID_INPUT,
                "initial_observation",
                "initial observation must be null when root is chance",
            )
        if type(scenario.root.outcomes) is not tuple:
            raise _ScenarioFailure(
                INVALID_INPUT, "chance", "chance outcomes must be an ordered tuple"
            )
        chance_outcome_count = len(scenario.root.outcomes)
        if chance_outcome_count < 1:
            raise _ScenarioFailure(INVALID_INPUT, "chance", "chance has no outcomes")
        if chance_outcome_count > limits.max_chance_outcomes:
            raise _ScenarioFailure(
                CAP_EXCEEDED, "chance", "max_chance_outcomes exceeded"
            )
    else:
        if scenario.initial_observation is None:
            raise _ScenarioFailure(
                INVALID_INPUT,
                "initial_observation",
                "initial observation is required without root chance",
            )
        chance_outcome_count = 0
    observation_count = _checked_product(
        chance_outcome_count or 1,
        len(PLAYERS),
        limits.max_private_observation_records,
        "observations",
    )
    identity_budget.reserve(observation_count)

    initial_state = _PathState(
        active=PLAYERS,
        contributions=dict(parsed.initial_contribution),
        amount_to_call=None,
        pending=scenario.river_action_order,
        raise_used=False,
        last_aggressor=None,
    )

    def register_node(node: RiverNode) -> str:
        nonlocal node_count
        if not isinstance(
            node, (RiverDecisionNode, RiverTerminalNode, RiverChanceNode)
        ):
            raise _ScenarioFailure(INVALID_INPUT, "tree", "unsupported node type")
        node_id = _bounded_text(node.node_id, "tree.node_id")
        if node_id in seen_node_ids or id(node) in seen_objects:
            raise _ScenarioFailure(INVALID_INPUT, "tree", "duplicate/cyclic node")
        identity_budget.reserve()
        seen_node_ids.add(node_id)
        seen_objects.add(id(node))
        node_count += 1
        if node_count > limits.max_nodes:
            raise _ScenarioFailure(CAP_EXCEEDED, "tree", "max_nodes exceeded")
        return node_id

    def register_information_set(
        node: RiverDecisionNode,
        legal_signature: tuple[tuple[str, str, str | None], ...],
        observation: dict[str, Any],
        public_history: tuple[tuple[str, str, str, str | None], ...],
        own_information_sets: Mapping[str, tuple[str, ...]],
        own_actions: Mapping[str, tuple[str, ...]],
    ) -> None:
        information_set = _bounded_text(
            node.information_set_id,
            f"tree.{node.node_id}.information_set_id",
        )
        previous_owner = owner_by_information_set.get(information_set)
        if previous_owner is not None and previous_owner != node.owner:
            raise _ScenarioFailure(
                INVALID_INPUT,
                "information_sets",
                "information set ID is shared across owners",
            )
        owner_by_information_set[information_set] = node.owner
        if information_set in own_information_sets[node.owner]:
            raise _ScenarioFailure(
                INVALID_INPUT,
                "perfect_recall",
                "information set repeats on one root-to-terminal path",
            )
        actions = tuple(item[0] for item in legal_signature)
        existing = info_sets[node.owner].get(information_set)
        candidate = _InfoSet(node.owner, actions, legal_signature)
        if existing is None:
            maximum = (
                limits.max_fixed_hero_info_sets
                if node.owner == "H"
                else limits.max_info_sets_per_opponent
            )
            if len(info_sets[node.owner]) + 1 > maximum:
                raise _ScenarioFailure(
                    CAP_EXCEEDED,
                    "information_sets",
                    "per-player information-set cap exceeded",
                )
            if node.owner in ("O1", "O2"):
                current_total = _checked_sum(
                    len(info_sets["O1"]),
                    len(info_sets["O2"]),
                    limits.max_opponent_info_sets_total,
                    "information_sets",
                )
                _checked_sum(
                    current_total,
                    1,
                    limits.max_opponent_info_sets_total,
                    "information_sets",
                )
            identity_budget.reserve()
            info_sets[node.owner][information_set] = candidate
        elif existing != candidate:
            raise _ScenarioFailure(
                INVALID_INPUT,
                "information_sets",
                "owner or ordered legal-action signature mismatch",
            )
        signature = (
            node.owner,
            legal_signature,
            observation["public_observation_id"],
            public_history,
            observation["private_observation_id_by_player"][node.owner],
            own_information_sets[node.owner],
            own_actions[node.owner],
        )
        previous = occurrence_signatures.get(information_set)
        if previous is not None and previous != signature:
            raise _ScenarioFailure(
                INVALID_INPUT,
                "perfect_recall",
                "information-set observation or own-recall signature mismatch",
            )
        occurrence_signatures[information_set] = signature
        identity_budget.reserve()
        occurrence_records.append(
            {
                "node_id": node.node_id,
                "owner": node.owner,
                "information_set_id": information_set,
                "public_observation_id": observation["public_observation_id"],
                "private_observation_id": observation[
                    "private_observation_id_by_player"
                ][node.owner],
                "public_action_history": [
                    {
                        "owner": item[0],
                        "action_id": item[1],
                        "kind": item[2],
                        "target_total_contribution": item[3],
                    }
                    for item in public_history
                ],
                "own_past_information_sets": list(
                    own_information_sets[node.owner]
                ),
                "own_past_actions": list(own_actions[node.owner]),
                "ordered_legal_action_signature": [
                    {
                        "action_id": item[0],
                        "kind": item[1],
                        "target_total_contribution": item[2],
                    }
                    for item in legal_signature
                ],
            }
        )

    def walk(
        node: RiverNode,
        state: _PathState,
        observation: dict[str, Any],
        public_history: tuple[tuple[str, str, str, str | None], ...],
        own_information_sets: dict[str, tuple[str, ...]],
        own_actions: dict[str, tuple[str, ...]],
        expected_terminal_kind: str | None,
        *,
        allow_chance: bool = False,
    ) -> dict[str, Any]:
        nonlocal terminal_count
        node_id = register_node(node)
        if isinstance(node, RiverChanceNode):
            if not allow_chance:
                raise _ScenarioFailure(
                    UNSUPPORTED_MODEL,
                    "chance",
                    "chance is supported only at the root",
                )
            outcome_ids: set[str] = set()
            probabilities: list[Fraction] = []
            encoded_outcomes = []
            for index, outcome in enumerate(node.outcomes):
                if type(outcome) is not RiverChanceOutcome:
                    raise _ScenarioFailure(
                        INVALID_INPUT, "chance", "outcome must be RiverChanceOutcome"
                    )
                outcome_id = _bounded_text(
                    outcome.outcome_id, f"chance.outcomes[{index}].outcome_id"
                )
                if outcome_id in outcome_ids:
                    raise _ScenarioFailure(
                        INVALID_INPUT, "chance", "duplicate chance outcome ID"
                    )
                outcome_ids.add(outcome_id)
                probability = _parse_rational(
                    outcome.probability,
                    f"chance.outcomes[{index}].probability",
                    limits,
                    minimum=Fraction(0),
                    maximum=Fraction(1),
                )
                probabilities.append(probability)
                encoded_observation = _validate_observation(
                    outcome.observation, f"chance.outcomes[{index}].observation"
                )
                child = walk(
                    outcome.child,
                    initial_state,
                    encoded_observation,
                    (),
                    {player: () for player in PLAYERS},
                    {player: () for player in PLAYERS},
                    None,
                )
                encoded_outcomes.append(
                    {
                        "outcome_id": outcome_id,
                        "probability": _rational_text(probability),
                        "observation": encoded_observation,
                        "child": child,
                    }
                )
            total = Fraction(0)
            for probability in probabilities:
                total = _fadd(total, probability, limits, "chance.probability_sum")
            if total != 1:
                raise _ScenarioFailure(
                    INVALID_INPUT, "chance", "chance probabilities must sum to one"
                )
            chance_probabilities[node_id] = tuple(probabilities)
            return {
                "type": "chance",
                "node_id": node_id,
                "owner": "chance",
                "outcomes": encoded_outcomes,
            }

        if isinstance(node, RiverTerminalNode):
            if expected_terminal_kind is None:
                raise _ScenarioFailure(
                    INVALID_INPUT, "terminals", "terminal reached before round completion"
                )
            if node.kind != expected_terminal_kind:
                raise _ScenarioFailure(
                    INVALID_INPUT,
                    "terminals",
                    "terminal kind does not match path state",
                )
            terminal_count += 1
            if terminal_count > limits.max_terminals:
                raise _ScenarioFailure(
                    CAP_EXCEEDED, "tree", "max_terminals exceeded"
                )
            contribution_records = _checked_product(
                terminal_count,
                len(PLAYERS),
                limits.max_terminal_contribution_records,
                "terminals",
            )
            identity_budget.reserve(len(PLAYERS))
            utility, record = _terminal_accounting(node, state, parsed, limits)
            terminal_utilities[node_id] = utility
            terminal_records.append(record)
            return {
                "type": "terminal",
                "node_id": node_id,
                "kind": node.kind,
                "award_shares": record["award_share"],
                "terminal_contribution_records": contribution_records,
            }

        if expected_terminal_kind is not None:
            raise _ScenarioFailure(
                INVALID_INPUT,
                "tree",
                "path must terminate immediately when the betting round completes",
            )
        if type(node) is not RiverDecisionNode:
            raise _ScenarioFailure(INVALID_INPUT, "tree", "unsupported decision node")
        if node.owner not in PLAYERS:
            raise _ScenarioFailure(
                INVALID_INPUT, "tree.owner", "decision owner must be H/O1/O2"
            )
        if not state.pending or node.owner != state.pending[0]:
            raise _ScenarioFailure(
                INVALID_INPUT,
                "tree.owner",
                "decision owner does not match river action state",
            )
        if type(node.actions) is not tuple or not node.actions:
            raise _ScenarioFailure(INVALID_INPUT, "tree.actions", "actions are required")
        if len(node.actions) > limits.max_actions_per_info_set:
            raise _ScenarioFailure(
                CAP_EXCEEDED, "tree", "max_actions_per_info_set exceeded"
            )
        action_ids: set[str] = set()
        action_kinds: set[str] = set()
        parsed_targets: list[Fraction | None] = []
        legal_signature = []
        for index, action in enumerate(node.actions):
            if type(action) is not RiverAction:
                raise _ScenarioFailure(
                    INVALID_INPUT, "tree.actions", "action must be RiverAction"
                )
            action_id = _bounded_text(
                action.action_id, f"tree.{node_id}.actions[{index}].action_id"
            )
            kind = _bounded_text(
                action.kind, f"tree.{node_id}.actions[{index}].kind"
            )
            if kind not in ACTION_KINDS:
                raise _ScenarioFailure(
                    INVALID_INPUT, "tree.actions", "unknown action kind"
                )
            if action_id in action_ids or kind in action_kinds:
                raise _ScenarioFailure(
                    INVALID_INPUT,
                    "tree.actions",
                    "action IDs and action kinds must be unique at a node",
                )
            action_ids.add(action_id)
            action_kinds.add(kind)
            target = (
                None
                if action.target_total_contribution is None
                else _parse_rational(
                    action.target_total_contribution,
                    f"tree.{node_id}.actions[{index}].target",
                    limits,
                    minimum=Fraction(0),
                )
            )
            parsed_targets.append(target)
            legal_signature.append(
                (
                    action_id,
                    kind,
                    None if target is None else _rational_text(target),
                )
            )
        register_information_set(
            node,
            tuple(legal_signature),
            observation,
            public_history,
            own_information_sets,
            own_actions,
        )
        encoded_actions = []
        for action, target, signature_item in zip(
            node.actions, parsed_targets, legal_signature
        ):
            next_state, terminal_kind = _transition(
                state,
                node.owner,
                action,
                target,
                scenario.river_action_order,
                parsed.max_total_contribution,
            )
            next_public_history = public_history + (
                (node.owner, signature_item[0], signature_item[1], signature_item[2]),
            )
            next_own_information_sets = dict(own_information_sets)
            next_own_actions = dict(own_actions)
            next_own_information_sets[node.owner] = (
                own_information_sets[node.owner] + (node.information_set_id,)
            )
            next_own_actions[node.owner] = own_actions[node.owner] + (
                action.action_id,
            )
            encoded_actions.append(
                {
                    "action_id": signature_item[0],
                    "kind": signature_item[1],
                    "target_total_contribution": signature_item[2],
                    "child": walk(
                        action.child,
                        next_state,
                        observation,
                        next_public_history,
                        next_own_information_sets,
                        next_own_actions,
                        terminal_kind,
                    ),
                }
            )
        return {
            "type": "decision",
            "node_id": node_id,
            "owner": node.owner,
            "information_set_id": node.information_set_id,
            "actions": encoded_actions,
        }

    if isinstance(scenario.root, RiverChanceNode):
        placeholder = {
            "public_observation_id": "",
            "private_observation_id_by_player": {player: "" for player in PLAYERS},
        }
        tree_projection = walk(
            scenario.root,
            initial_state,
            placeholder,
            (),
            {player: () for player in PLAYERS},
            {player: () for player in PLAYERS},
            None,
            allow_chance=True,
        )
    else:
        observation = _validate_observation(
            scenario.initial_observation, "initial_observation"
        )
        tree_projection = walk(
            scenario.root,
            initial_state,
            observation,
            (),
            {player: () for player in PLAYERS},
            {player: () for player in PLAYERS},
            None,
        )
    terminal_contribution_records = _checked_product(
        terminal_count,
        len(PLAYERS),
        limits.max_terminal_contribution_records,
        "terminals",
    )
    structure_projection = {
        "version": CONTRACT_VERSION,
        "button_player_id": scenario.button_player_id,
        "seat_order": list(scenario.seat_order),
        "river_action_order": list(scenario.river_action_order),
        "initial_observation": (
            None if isinstance(scenario.root, RiverChanceNode) else observation
        ),
        "tree": tree_projection,
        "plan_ordering": (
            "sorted information-set IDs; tree-declared ordered actions; "
            "O1 outer / O2 inner"
        ),
    }
    evidence_projection = {
        "version": STRUCTURAL_EVIDENCE_VERSION,
        "tree_structure_identity": None,
        "occurrences": occurrence_records,
    }
    rake_projection = {
        "accounting_version": ACCOUNTING_VERSION,
        "rake_rate": _rational_text(parsed.rake_rate),
        "rake_cap": (
            None if parsed.rake_cap is None else _rational_text(parsed.rake_cap)
        ),
        "uncalled_return_order": "unique-max-before-rake",
        "fold_rake": "0",
        "showdown_rake": "min(rate*pot_before_rake,cap)",
        "rounding_unit": None,
        "rounding_rule": "exact_none",
    }
    scenario_projection = {
        "contract_version": CONTRACT_VERSION,
        "structure": structure_projection,
        "initial_pot": _rational_text(parsed.initial_pot),
        "initial_contribution": {
            player: _rational_text(parsed.initial_contribution[player])
            for player in PLAYERS
        },
        "max_total_contribution": {
            player: _rational_text(parsed.max_total_contribution[player])
            for player in PLAYERS
        },
        "rake": rake_projection,
    }
    return _Structure(
        parsed=parsed,
        node_count=node_count,
        terminal_count=terminal_count,
        chance_outcome_count=chance_outcome_count,
        private_observation_records=observation_count,
        terminal_contribution_records=terminal_contribution_records,
        info_sets=info_sets,
        tree_projection=tree_projection,
        terminal_utilities=terminal_utilities,
        terminal_records=terminal_records,
        chance_probabilities=chance_probabilities,
        occurrence_records=occurrence_records,
        structure_projection=structure_projection,
        evidence_projection=evidence_projection,
        rake_projection=rake_projection,
        scenario_projection=scenario_projection,
        scenario_identity="",
        tree_structure_identity="",
        structural_evidence_identity="",
        rake_convention_identity="",
    )


def _finalize_structure_identities(structure: _Structure) -> None:
    """Hash already-budgeted projections without allocating new records."""

    structure.tree_structure_identity = _identity(structure.structure_projection)
    structure.evidence_projection["tree_structure_identity"] = (
        structure.tree_structure_identity
    )
    structure.structural_evidence_identity = _identity(
        structure.evidence_projection
    )
    structure.rake_convention_identity = _identity(structure.rake_projection)
    structure.scenario_identity = _identity(structure.scenario_projection)


def create_perfect_recall_attestation(
    scenario: ThreePlayerRiverRakeScenario,
    *,
    verifier: str,
    verification_date: str,
    evidence_version: str,
    o1_confirmed: bool,
    o2_confirmed: bool,
    limits: RiverRakeLimits = RiverRakeLimits(),
) -> PerfectRecallAttestation:
    """Create human evidence bound to the current validated tree and records."""

    _validate_limits(limits)
    identity_budget = _IdentityRecordBudget(limits.max_identity_records)
    structure = _prepare_structure(scenario, limits, identity_budget)
    _finalize_structure_identities(structure)
    return PerfectRecallAttestation(
        tree_content_identity=structure.tree_structure_identity,
        structural_evidence_identity=structure.structural_evidence_identity,
        o1_confirmed=o1_confirmed,
        o2_confirmed=o2_confirmed,
        verifier=verifier,
        verification_date=verification_date,
        evidence_version=evidence_version,
    )


def _validate_attestation(
    attestation: Any,
    structure: _Structure,
) -> str:
    if attestation is None or type(attestation) is not PerfectRecallAttestation:
        raise _ScenarioFailure(
            UNSUPPORTED_MODEL,
            "perfect_recall",
            "perfect-recall attestation is required",
        )
    if (
        type(attestation.o1_confirmed) is not bool
        or type(attestation.o2_confirmed) is not bool
        or not attestation.o1_confirmed
        or not attestation.o2_confirmed
    ):
        raise _ScenarioFailure(
            UNSUPPORTED_MODEL,
            "perfect_recall",
            "perfect-recall evidence is not confirmed for O1/O2",
        )
    _validate_identity(
        attestation.tree_content_identity,
        "perfect_recall.tree_content_identity",
    )
    _validate_identity(
        attestation.structural_evidence_identity,
        "perfect_recall.structural_evidence_identity",
    )
    for name in ("verifier", "evidence_version"):
        _bounded_text(getattr(attestation, name), f"perfect_recall.{name}")
    if (
        type(attestation.verification_date) is not str
        or _DATE_RE.fullmatch(attestation.verification_date) is None
    ):
        raise _ScenarioFailure(
            UNSUPPORTED_MODEL,
            "perfect_recall.verification_date",
            "verification date must be YYYY-MM-DD",
        )
    if (
        attestation.tree_content_identity != structure.tree_structure_identity
        or attestation.structural_evidence_identity
        != structure.structural_evidence_identity
    ):
        raise _ScenarioFailure(
            STALE_INPUT,
            "perfect_recall",
            "perfect-recall evidence is stale for the current tree",
        )
    return _identity(
        {
            "machine_evidence_version": STRUCTURAL_EVIDENCE_VERSION,
            "tree_content_identity": attestation.tree_content_identity,
            "structural_evidence_identity": (
                attestation.structural_evidence_identity
            ),
            "human_attestation": {
                "o1_confirmed": True,
                "o2_confirmed": True,
                "verifier": attestation.verifier,
                "verification_date": attestation.verification_date,
                "evidence_version": attestation.evidence_version,
            },
        }
    )


def _validate_policy(
    raw_probabilities: Any,
    info_sets: Mapping[str, _InfoSet],
    label: str,
    limits: RiverRakeLimits,
) -> tuple[dict[str, dict[str, Fraction]], list[dict[str, Any]]]:
    if not isinstance(raw_probabilities, Mapping):
        raise _ScenarioFailure(INVALID_INPUT, label, f"{label} must be a mapping")
    if set(raw_probabilities) != set(info_sets):
        missing = sorted(set(info_sets) - set(raw_probabilities))
        unknown = sorted(set(raw_probabilities) - set(info_sets))
        raise _ScenarioFailure(
            INVALID_INPUT,
            label,
            f"{label} must be complete; missing={missing}, unknown={unknown}",
        )
    parsed: dict[str, dict[str, Fraction]] = {}
    projection: list[dict[str, Any]] = []
    for information_set in sorted(info_sets):
        distribution = raw_probabilities[information_set]
        if not isinstance(distribution, Mapping):
            raise _ScenarioFailure(
                INVALID_INPUT, label, "policy distribution must be a mapping"
            )
        legal_actions = info_sets[information_set].actions
        if set(distribution) != set(legal_actions):
            raise _ScenarioFailure(
                INVALID_INPUT,
                label,
                "policy action keys must exactly equal legal actions",
            )
        values: dict[str, Fraction] = {}
        total = Fraction(0)
        for action in legal_actions:
            probability = _parse_rational(
                distribution[action],
                f"{label}.{information_set}.{action}",
                limits,
                minimum=Fraction(0),
                maximum=Fraction(1),
            )
            values[action] = probability
            total = _fadd(total, probability, limits, label)
        if total != 1:
            raise _ScenarioFailure(
                INVALID_INPUT, label, "policy probabilities must sum exactly to one"
            )
        parsed[information_set] = values
        projection.append(
            {
                "information_set_id": information_set,
                "probabilities": {
                    action: _rational_text(values[action])
                    for action in legal_actions
                },
            }
        )
    return parsed, projection


def _effective_limits(
    limits: RiverRakeLimits,
    m30_limits: ExactResponseLimits,
) -> tuple[dict[str, int], ExactResponseLimits]:
    effective = {
        "max_pure_plans_o1": min(
            limits.max_pure_plans_o1, m30_limits.max_pure_plans_o1
        ),
        "max_pure_plans_o2": min(
            limits.max_pure_plans_o2, m30_limits.max_pure_plans_o2
        ),
        "max_joint_pure_profiles": min(
            limits.max_joint_pure_profiles,
            m30_limits.max_joint_pure_profiles,
        ),
        "max_rational_numerator_bits": min(
            limits.max_rational_numerator_bits,
            m30_limits.max_rational_numerator_bits,
        ),
        "max_rational_denominator_bits": min(
            limits.max_rational_denominator_bits,
            m30_limits.max_rational_denominator_bits,
        ),
        "max_output_records": min(
            limits.max_output_records, m30_limits.max_output_records
        ),
        "max_output_bytes": min(
            limits.max_output_bytes, m30_limits.max_output_bytes
        ),
    }
    effective_m30 = replace(
        m30_limits,
        max_pure_plans_o1=effective["max_pure_plans_o1"],
        max_pure_plans_o2=effective["max_pure_plans_o2"],
        max_joint_pure_profiles=effective["max_joint_pure_profiles"],
        max_rational_numerator_bits=effective[
            "max_rational_numerator_bits"
        ],
        max_rational_denominator_bits=effective[
            "max_rational_denominator_bits"
        ],
        max_output_records=effective["max_output_records"],
        max_output_bytes=effective["max_output_bytes"],
    )
    return effective, effective_m30


def _plan_count(
    info_sets: Mapping[str, _InfoSet],
    maximum: int,
    phase: str,
) -> int:
    count = 1
    for information_set in sorted(info_sets):
        count = _checked_product(
            count, len(info_sets[information_set].actions), maximum, phase
        )
    return count


def _materialize_plans(
    player: str,
    info_sets: Mapping[str, _InfoSet],
    expected_count: int,
) -> tuple[tuple[str, ...], tuple[dict[str, str], ...]]:
    ordered_info_sets = tuple(sorted(info_sets))
    action_lists = tuple(info_sets[item].actions for item in ordered_info_sets)
    combinations = itertools.product(*action_lists) if action_lists else [()]
    plans = tuple(
        {
            information_set: action
            for information_set, action in zip(ordered_info_sets, combination)
        }
        for combination in combinations
    )
    if len(plans) != expected_count:
        raise _ScenarioFailure(
            NUMERIC_FAILURE, "plans", "pure-plan materialization count mismatch"
        )
    plan_ids = tuple(f"{player}:{index}" for index in range(expected_count))
    return plan_ids, plans


def _selected_child(node: RiverDecisionNode, action_id: str) -> RiverNode:
    for action in node.actions:
        if action.action_id == action_id:
            return action.child
    raise _ScenarioFailure(
        NUMERIC_FAILURE, "evaluation", "pure plan selected an unknown action"
    )


def _primary_evaluate(
    node: RiverNode,
    structure: _Structure,
    hero_policy: Mapping[str, Mapping[str, Fraction]],
    o1_plan: Mapping[str, str],
    o2_plan: Mapping[str, str],
    limits: RiverRakeLimits,
) -> _Utility:
    if isinstance(node, RiverTerminalNode):
        return structure.terminal_utilities[node.node_id]
    if isinstance(node, RiverChanceNode):
        total = _zero_utility()
        probabilities = structure.chance_probabilities[node.node_id]
        for probability, outcome in zip(probabilities, node.outcomes):
            branch = _primary_evaluate(
                outcome.child,
                structure,
                hero_policy,
                o1_plan,
                o2_plan,
                limits,
            )
            total = _add_utility(
                total,
                _scale_utility(probability, branch, limits, "evaluation.chance"),
                limits,
                "evaluation.chance",
            )
        return total
    if isinstance(node, RiverDecisionNode):
        if node.owner == "H":
            total = _zero_utility()
            distribution = hero_policy[node.information_set_id]
            for action in node.actions:
                branch = _primary_evaluate(
                    action.child,
                    structure,
                    hero_policy,
                    o1_plan,
                    o2_plan,
                    limits,
                )
                total = _add_utility(
                    total,
                    _scale_utility(
                        distribution[action.action_id],
                        branch,
                        limits,
                        "evaluation.hero",
                    ),
                    limits,
                    "evaluation.hero",
                )
            return total
        plan = o1_plan if node.owner == "O1" else o2_plan
        return _primary_evaluate(
            _selected_child(node, plan[node.information_set_id]),
            structure,
            hero_policy,
            o1_plan,
            o2_plan,
            limits,
        )
    raise _ScenarioFailure(NUMERIC_FAILURE, "evaluation", "unknown node type")


def _independent_path_evaluate(
    root: RiverNode,
    structure: _Structure,
    hero_policy: Mapping[str, Mapping[str, Fraction]],
    o1_plan: Mapping[str, str],
    o2_plan: Mapping[str, str],
    limits: RiverRakeLimits,
) -> tuple[_Utility, int]:
    """Enumerate terminal paths without the primary recursion dispatcher."""

    total = _zero_utility()
    terminal_paths = 0
    stack: list[tuple[RiverNode, Fraction]] = [(root, Fraction(1))]
    while stack:
        current, path_probability = stack.pop()
        if type(current) is RiverTerminalNode:
            terminal_paths += 1
            utility = structure.terminal_utilities[current.node_id]
            total = _add_utility(
                total,
                _scale_utility(
                    path_probability, utility, limits, "independent.paths"
                ),
                limits,
                "independent.paths",
            )
            continue
        if type(current) is RiverChanceNode:
            probabilities = structure.chance_probabilities[current.node_id]
            for index in range(len(current.outcomes) - 1, -1, -1):
                probability = _fmul(
                    path_probability,
                    probabilities[index],
                    limits,
                    "independent.chance",
                )
                stack.append((current.outcomes[index].child, probability))
            continue
        if type(current) is RiverDecisionNode:
            if current.owner == "H":
                distribution = hero_policy[current.information_set_id]
                for action in reversed(current.actions):
                    probability = _fmul(
                        path_probability,
                        distribution[action.action_id],
                        limits,
                        "independent.hero",
                    )
                    stack.append((action.child, probability))
                continue
            selected_plan = o1_plan if current.owner == "O1" else o2_plan
            selected = selected_plan[current.information_set_id]
            chosen: RiverNode | None = None
            for action in current.actions:
                if action.action_id == selected:
                    chosen = action.child
                    break
            if chosen is None:
                raise _ScenarioFailure(
                    NUMERIC_FAILURE,
                    "independent",
                    "independent enumerator selected unknown action",
                )
            stack.append((chosen, path_probability))
            continue
        raise _ScenarioFailure(
            NUMERIC_FAILURE, "independent", "unknown node in path enumerator"
        )
    return total, terminal_paths


def _behavior_evaluate(
    node: RiverNode,
    structure: _Structure,
    profiles: Mapping[str, Mapping[str, Mapping[str, Fraction]]],
    limits: RiverRakeLimits,
) -> _Utility:
    if isinstance(node, RiverTerminalNode):
        return structure.terminal_utilities[node.node_id]
    if isinstance(node, RiverChanceNode):
        total = _zero_utility()
        for probability, outcome in zip(
            structure.chance_probabilities[node.node_id], node.outcomes
        ):
            total = _add_utility(
                total,
                _scale_utility(
                    probability,
                    _behavior_evaluate(outcome.child, structure, profiles, limits),
                    limits,
                    "initial_profile.chance",
                ),
                limits,
                "initial_profile.chance",
            )
        return total
    if isinstance(node, RiverDecisionNode):
        total = _zero_utility()
        distribution = profiles[node.owner][node.information_set_id]
        for action in node.actions:
            total = _add_utility(
                total,
                _scale_utility(
                    distribution[action.action_id],
                    _behavior_evaluate(action.child, structure, profiles, limits),
                    limits,
                    "initial_profile.decision",
                ),
                limits,
                "initial_profile.decision",
            )
        return total
    raise _ScenarioFailure(NUMERIC_FAILURE, "initial_profile", "unknown node")


def _check_pin(
    pins: RiverRakeIdentityPins, name: str, actual: str | None
) -> None:
    expected = getattr(pins, name)
    if expected is not None and expected != actual:
        raise _ScenarioFailure(
            STALE_INPUT, f"identity_pins.{name}", f"stale {name}"
        )


def _record_count(value: Any, maximum: int) -> int:
    count = 1
    if isinstance(value, Mapping):
        children = value.values()
    elif isinstance(value, (list, tuple)):
        children = value
    else:
        return count
    for item in children:
        count = _checked_sum(
            count,
            _record_count(item, maximum),
            maximum,
            "output",
        )
    return count


def _m31_owned_output_record_count(
    structure: _Structure,
    o1_count: int,
    o2_count: int,
    joint_count: int,
    *,
    include_initial_comparison: bool,
    maximum: int,
) -> int:
    """Count every nested M31-owned output record before plan/row allocation."""

    terminal_records = _record_count(structure.terminal_records, maximum)
    o1_plan_records = _checked_sum(
        1,
        o1_count * (3 + len(structure.info_sets["O1"])),
        maximum,
        "output",
    )
    o2_plan_records = _checked_sum(
        1,
        o2_count * (3 + len(structure.info_sets["O2"])),
        maximum,
        "output",
    )
    # One list record plus, per row: row mapping, three scalar identifiers,
    # utility mapping + four scalars, and the conservation scalar.
    row_records = _checked_sum(1, joint_count * 10, maximum, "output")
    payoff_table_records = 4
    for increment in (o1_plan_records, o2_plan_records, row_records):
        payoff_table_records = _checked_sum(
            payoff_table_records, increment, maximum, "output"
        )

    identity_records = 1 + 11
    count_records = 1 + 14
    ordering_records = 1 + 2 * (1 + len(PLAYERS)) + 4
    rake_records = 1 + 5
    human_attestation_records = 1 + 7
    perfect_recall_records = 1 + 4 + human_attestation_records
    independent_records = 1 + 5
    initial_records = 9 if include_initial_comparison else 1
    limits_records = (
        1
        + 1
        + len(fields(RiverRakeLimits))
        + 1
        + len(fields(ExactResponseLimits))
        + 1
        + len(fields(ExactResponseLimits))
    )
    m32_records = 1 + 3
    scenario_records = 1 + 3
    for increment in (
        identity_records,
        count_records,
        ordering_records,
        rake_records,
        terminal_records,
        perfect_recall_records,
        independent_records,
        initial_records,
        limits_records,
        m32_records,
    ):
        scenario_records = _checked_sum(
            scenario_records, increment, maximum, "output"
        )

    # Outer mapping plus status/error/partial scalars.  The downstream response
    # subtree is deliberately excluded and receives the remaining shared budget.
    outer_owned = _checked_sum(4, scenario_records, maximum, "output")
    return _checked_sum(outer_owned, payoff_table_records, maximum, "output")


def _minimum_m30_output_records(o1_count: int, o2_count: int) -> int:
    """Return a strict lower bound for every complete M30 response."""

    support_pairs = ((1 << o1_count) - 1) * ((1 << o2_count) - 1)
    # M30 base 256 + complete audit; at least one cell with one source pair and
    # one vertex per marginal; eight account-extremum witnesses; and one
    # coalition-stress witness.  The pure subset may legitimately be empty.
    return 276 + support_pairs


def _minimum_m30_preflight_bytes(
    o1_count: int,
    o2_count: int,
    records: int,
    limits: ExactResponseLimits,
) -> int:
    """Lower bound the conservative byte preflight enforced by M30 itself."""

    maximum_label = max(
        len(f"O1:{o1_count - 1}"),
        len(f"O2:{o2_count - 1}"),
    )
    maximum_rational_chars = (
        math.ceil(limits.max_rational_numerator_bits * math.log10(2))
        + math.ceil(limits.max_rational_denominator_bits * math.log10(2))
        + 4
    )
    minimum_vertex_scalars = o1_count + o2_count
    return (
        8_192
        + records * (128 + 4 * maximum_label)
        + minimum_vertex_scalars
        * (maximum_rational_chars + maximum_label + 8)
    )


def _preflight_outer_output_allocation(
    structure: _Structure,
    o1_count: int,
    o2_count: int,
    joint_count: int,
    *,
    include_initial_comparison: bool,
    limits: RiverRakeLimits,
    effective_m30_limits: ExactResponseLimits,
) -> tuple[int, int]:
    """Exactly split logical output records before plan allocation."""

    m31_records = _m31_owned_output_record_count(
        structure,
        o1_count,
        o2_count,
        joint_count,
        include_initial_comparison=include_initial_comparison,
        maximum=limits.max_output_records,
    )
    minimum_m30_records = _minimum_m30_output_records(o1_count, o2_count)
    _checked_sum(
        m31_records,
        minimum_m30_records,
        limits.max_output_records,
        "output",
    )
    if (
        _minimum_m30_preflight_bytes(
            o1_count,
            o2_count,
            minimum_m30_records,
            effective_m30_limits,
        )
        > limits.max_output_bytes
    ):
        raise _ScenarioFailure(
            CAP_EXCEEDED,
            "output",
            "max_output_bytes cannot satisfy the M30 minimum preflight",
        )
    return m31_records, minimum_m30_records


def _expected_m30_identities(
    reduced_game: ExactReducedGame,
    limits: ExactResponseLimits,
) -> dict[str, str]:
    canonical_rows = [
        {
            "o1_plan_id": row.o1_plan_id,
            "o2_plan_id": row.o2_plan_id,
            "utility": {
                name: getattr(row, name) for name in ALL_ACCOUNTS
            },
        }
        for row in reduced_game.payoff_rows
    ]
    payoff_table_identity = _identity(
        {
            "o1_plan_ids": list(reduced_game.o1_plan_ids),
            "o2_plan_ids": list(reduced_game.o2_plan_ids),
            "rows": canonical_rows,
        }
    )
    response_game_identity = _identity(
        {
            "semantics": "fixed-hero-two-opponent-noncooperative-bimatrix-v1",
            "o1_plan_ids": list(reduced_game.o1_plan_ids),
            "o2_plan_ids": list(reduced_game.o2_plan_ids),
            "payoff_table_identity": payoff_table_identity,
            "response_game_structure_identity": (
                reduced_game.response_game_structure_identity
            ),
            "fixed_hero_identity": reduced_game.fixed_hero_identity,
            "perfect_recall_evidence_identity": (
                reduced_game.perfect_recall_evidence_identity
            ),
            "rake_convention_identity": reduced_game.rake_convention_identity,
        }
    )
    config_identity = _identity(
        {
            "contract_version": _m30.CONTRACT_VERSION,
            "algorithm_version": _m30.ALGORITHM_VERSION,
            "verifier_version": _m30.VERIFIER_VERSION,
            "limits": asdict(limits),
            "ordering": (
                "plan order; support cardinality; lexicographic support indices; "
                "exact rational vertices; support_cell_id"
            ),
        }
    )
    response_run_identity = _identity(
        {
            "response_game_identity": response_game_identity,
            "config_identity": config_identity,
            "supplied_profile_identity": reduced_game.supplied_profile_identity,
            "candidate_identity": reduced_game.candidate_identity,
            "search_mode_context_identity": (
                reduced_game.search_mode_context_identity
            ),
            "run_context_identity": reduced_game.run_context_identity,
        }
    )
    return {
        "payoff_table": payoff_table_identity,
        "config": config_identity,
        "response_game": response_game_identity,
        "response_run": response_run_identity,
    }


def _m30_invalid(message: str) -> None:
    raise _ScenarioFailure(M30_RESPONSE_FAILURE, "m30_response", message)


def _m30_mapping(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        _m30_invalid(f"M30 {label} must be a dict")
    return value


def _m30_list(value: Any, label: str) -> list[Any]:
    if type(value) is not list:
        _m30_invalid(f"M30 {label} must be a list")
    return value


def _m30_int(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum:
        _m30_invalid(f"M30 {label} must be a bounded nonnegative int")
    if maximum is not None and value > maximum:
        _m30_invalid(f"M30 {label} exceeds its bound")
    return value


def _m30_fraction(
    value: Any,
    label: str,
    limits: ExactResponseLimits,
) -> Fraction:
    if type(value) is not str:
        _m30_invalid(f"M30 {label} must be a canonical rational string")
    if _INTEGER_RE.fullmatch(value) is not None:
        numerator_text, denominator_text = value, "1"
    else:
        match = _FRACTION_RE.fullmatch(value)
        if match is None:
            _m30_invalid(f"M30 {label} is not canonical rational text")
        numerator_text, denominator_text = match.groups()
    numerator = int(numerator_text)
    denominator = int(denominator_text)
    if denominator != 1 and (
        numerator == 0 or math.gcd(abs(numerator), denominator) != 1
    ):
        _m30_invalid(f"M30 {label} is not reduced")
    parsed = Fraction(numerator, denominator)
    if (
        abs(parsed.numerator).bit_length()
        > limits.max_rational_numerator_bits
        or parsed.denominator.bit_length()
        > limits.max_rational_denominator_bits
        or _rational_text(parsed) != value
    ):
        _m30_invalid(f"M30 {label} violates rational bounds or canonical form")
    return parsed


def _m30_utility(
    value: Any,
    label: str,
    limits: ExactResponseLimits,
) -> dict[str, Fraction]:
    mapping = _m30_mapping(value, label)
    if set(mapping) != set(ALL_ACCOUNTS):
        _m30_invalid(f"M30 {label} must contain H/O1/O2/R")
    utility = {
        name: _m30_fraction(mapping[name], f"{label}.{name}", limits)
        for name in ALL_ACCOUNTS
    }
    if sum(utility.values(), Fraction(0)) != 0:
        _m30_invalid(f"M30 {label} violates exact conservation")
    return utility


def _m30_mixture(
    value: Any,
    plan_ids: tuple[str, ...],
    label: str,
    limits: ExactResponseLimits,
) -> tuple[Fraction, ...]:
    mapping = _m30_mapping(value, label)
    if set(mapping) != set(plan_ids):
        _m30_invalid(f"M30 {label} must cover every plan exactly")
    mixture = tuple(
        _m30_fraction(mapping[plan_id], f"{label}.{plan_id}", limits)
        for plan_id in plan_ids
    )
    if any(item < 0 for item in mixture) or sum(mixture, Fraction(0)) != 1:
        _m30_invalid(f"M30 {label} is not an exact probability simplex")
    return mixture


def _m30_support_pair(
    value: Any,
    o1_plan_ids: tuple[str, ...],
    o2_plan_ids: tuple[str, ...],
    label: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    mapping = _m30_mapping(value, label)
    if set(mapping) != {"O1", "O2"}:
        _m30_invalid(f"M30 {label} must contain O1/O2 supports")
    supports: list[tuple[str, ...]] = []
    for player, plan_ids in (("O1", o1_plan_ids), ("O2", o2_plan_ids)):
        raw = _m30_list(mapping[player], f"{label}.{player}")
        support = tuple(raw)
        if (
            not support
            or len(set(support)) != len(support)
            or any(type(item) is not str or item not in plan_ids for item in support)
            or tuple(item for item in plan_ids if item in set(support)) != support
        ):
            _m30_invalid(f"M30 {label}.{player} is not a canonical support")
        supports.append(support)
    return supports[0], supports[1]


def _m30_exact_row_reduce(
    rows: Sequence[tuple[tuple[Fraction, ...], Fraction]],
    variable_count: int,
) -> tuple[list[list[Fraction]], tuple[int, ...], bool]:
    """Independently row-reduce an exact system for M30 response validation."""

    matrix = [list(coefficients) + [rhs] for coefficients, rhs in rows]
    pivots: list[int] = []
    pivot_row = 0
    for column in range(variable_count):
        selected = next(
            (
                row
                for row in range(pivot_row, len(matrix))
                if matrix[row][column] != 0
            ),
            None,
        )
        if selected is None:
            continue
        matrix[pivot_row], matrix[selected] = matrix[selected], matrix[pivot_row]
        pivot = matrix[pivot_row][column]
        matrix[pivot_row] = [value / pivot for value in matrix[pivot_row]]
        for row in range(len(matrix)):
            if row == pivot_row or matrix[row][column] == 0:
                continue
            factor = matrix[row][column]
            matrix[row] = [
                matrix[row][index] - factor * matrix[pivot_row][index]
                for index in range(variable_count + 1)
            ]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(matrix):
            break
    inconsistent = any(
        all(row[column] == 0 for column in range(variable_count))
        and row[-1] != 0
        for row in matrix
    )
    return matrix, tuple(pivots), inconsistent


def _m30_exact_rank(
    rows: Sequence[Sequence[Fraction]],
    variable_count: int,
) -> int:
    equations = [(tuple(row), Fraction(0)) for row in rows]
    _, pivots, _ = _m30_exact_row_reduce(equations, variable_count)
    return len(pivots)


def _m30_affine_dimension(vertices: Sequence[tuple[Fraction, ...]]) -> int:
    """Compute affine dimension from vertex differences without M30 helpers."""

    if len(vertices) <= 1:
        return 0
    origin = vertices[0]
    differences = [
        tuple(value - origin[index] for index, value in enumerate(vertex))
        for vertex in vertices[1:]
    ]
    return _m30_exact_rank(differences, len(origin))


def _m30_validation_constraints(
    payoff_by_pair: Mapping[tuple[str, str], Mapping[str, Fraction]],
    o1_plan_ids: tuple[str, ...],
    o2_plan_ids: tuple[str, ...],
    support_pair: tuple[tuple[str, ...], tuple[str, ...]],
    *,
    mixture_player: str,
) -> tuple[
    list[tuple[tuple[Fraction, ...], Fraction]],
    list[tuple[tuple[Fraction, ...], Fraction]],
]:
    """Rebuild one marginal response polytope directly from the payoff game."""

    o1_support, o2_support = support_pair
    if mixture_player == "O1":
        variable_ids = o1_support
        best_ids = o2_support
        all_best_ids = o2_plan_ids
        payoff_name = "O2"

        def payoff(variable: str, best: str) -> Fraction:
            return payoff_by_pair[(variable, best)][payoff_name]

    else:
        variable_ids = o2_support
        best_ids = o1_support
        all_best_ids = o1_plan_ids
        payoff_name = "O1"

        def payoff(variable: str, best: str) -> Fraction:
            return payoff_by_pair[(best, variable)][payoff_name]

    reference = best_ids[0]
    equalities: list[tuple[tuple[Fraction, ...], Fraction]] = [
        (tuple(Fraction(1) for _ in variable_ids), Fraction(1))
    ]
    for best in best_ids[1:]:
        equalities.append(
            (
                tuple(
                    payoff(variable, best) - payoff(variable, reference)
                    for variable in variable_ids
                ),
                Fraction(0),
            )
        )
    inequalities: list[tuple[tuple[Fraction, ...], Fraction]] = []
    for local_index in range(len(variable_ids)):
        inequalities.append(
            (
                tuple(
                    Fraction(-1) if index == local_index else Fraction(0)
                    for index in range(len(variable_ids))
                ),
                Fraction(0),
            )
        )
    best_set = set(best_ids)
    for best in all_best_ids:
        if best in best_set:
            continue
        inequalities.append(
            (
                tuple(
                    payoff(variable, best) - payoff(variable, reference)
                    for variable in variable_ids
                ),
                Fraction(0),
            )
        )
    return equalities, inequalities


def _m30_validation_system_count(
    equalities: Sequence[tuple[tuple[Fraction, ...], Fraction]],
    inequalities: Sequence[tuple[tuple[Fraction, ...], Fraction]],
    variable_count: int,
) -> int:
    _, pivots, inconsistent = _m30_exact_row_reduce(
        equalities, variable_count
    )
    needed = variable_count - len(pivots)
    if inconsistent or needed < 0 or needed > len(inequalities):
        return 0
    return math.comb(len(inequalities), needed)


def _m30_validation_has_vertex(
    equalities: Sequence[tuple[tuple[Fraction, ...], Fraction]],
    inequalities: Sequence[tuple[tuple[Fraction, ...], Fraction]],
    variable_count: int,
) -> bool:
    """Independently decide whether an exact bounded marginal has a vertex."""

    _, equality_pivots, inconsistent = _m30_exact_row_reduce(
        equalities, variable_count
    )
    needed = variable_count - len(equality_pivots)
    if inconsistent or needed < 0 or needed > len(inequalities):
        return False
    for active in itertools.combinations(range(len(inequalities)), needed):
        equations = list(equalities) + [
            inequalities[index] for index in active
        ]
        matrix, pivots, inconsistent = _m30_exact_row_reduce(
            equations, variable_count
        )
        if inconsistent or len(pivots) != variable_count:
            continue
        solution = [Fraction(0) for _ in range(variable_count)]
        for row_index, column in enumerate(pivots):
            solution[column] = matrix[row_index][-1]
        if any(
            sum(
                coefficient * solution[index]
                for index, coefficient in enumerate(coefficients)
            )
            != rhs
            for coefficients, rhs in equalities
        ):
            continue
        if any(
            sum(
                coefficient * solution[index]
                for index, coefficient in enumerate(coefficients)
            )
            > rhs
            for coefficients, rhs in inequalities
        ):
            continue
        return True
    return False


def _m30_reconstructed_system_counts(
    payoff_by_pair: Mapping[tuple[str, str], Mapping[str, Fraction]],
    o1_plan_ids: tuple[str, ...],
    o2_plan_ids: tuple[str, ...],
    support_pairs: Sequence[tuple[tuple[str, ...], tuple[str, ...]]],
    maximum: int,
) -> tuple[int, int]:
    """Reconstruct M30 preflight/solved counts without solver-side evidence."""

    preflight = 0
    solved = 0
    for support_pair in support_pairs:
        systems: dict[
            str,
            tuple[
                list[tuple[tuple[Fraction, ...], Fraction]],
                list[tuple[tuple[Fraction, ...], Fraction]],
                int,
                int,
            ],
        ] = {}
        for mixture_player, variable_count in (
            ("O1", len(support_pair[0])),
            ("O2", len(support_pair[1])),
        ):
            equalities, inequalities = _m30_validation_constraints(
                payoff_by_pair,
                o1_plan_ids,
                o2_plan_ids,
                support_pair,
                mixture_player=mixture_player,
            )
            count = _m30_validation_system_count(
                equalities, inequalities, variable_count
            )
            preflight += count
            if preflight > maximum:
                _m30_invalid(
                    "M30 independently reconstructed preflight systems exceed "
                    "the exact-system cap"
                )
            systems[mixture_player] = (
                equalities,
                inequalities,
                variable_count,
                count,
            )
        o1_equalities, o1_inequalities, o1_variables, o1_count = systems["O1"]
        solved += o1_count
        if solved > maximum:
            _m30_invalid(
                "M30 independently reconstructed solved systems exceed the "
                "exact-system cap"
            )
        if _m30_validation_has_vertex(
            o1_equalities, o1_inequalities, o1_variables
        ):
            solved += systems["O2"][3]
            if solved > maximum:
                _m30_invalid(
                    "M30 independently reconstructed solved systems exceed "
                    "the exact-system cap"
                )
    return preflight, solved


def _validate_m30_response(
    response: Any,
    *,
    reduced_game: ExactReducedGame,
    limits: ExactResponseLimits,
    expected_identities: Mapping[str, str],
) -> int:
    """Boundedly validate all M30 evidence required for M32 usability."""

    item = _m30_mapping(response, "response")
    required_top = {
        "contract_version",
        "algorithm_version",
        "verifier_version",
        "status",
        "coverage",
        "partial_response",
        "response_game_identity",
        "response_run_identity",
        "content_identities",
        "counts",
        "ordering",
        "limits",
        "support_cells",
        "support_pair_audit",
        "pure_profile_unilateral_stability",
        "utility_extrema",
        "hero_worst",
        "hero_best",
        "hero_worst_witnesses",
        "hero_best_witnesses",
        "hero_min_joint_plan_stress",
        "unilateral_residual_semantics",
        "unilateral_residual_certificate",
        "independent_verification",
    }
    if set(item) != required_top:
        _m30_invalid("M30 response top-level schema is incomplete or unknown")
    if (
        item["contract_version"] != _m30.CONTRACT_VERSION
        or item["algorithm_version"] != _m30.ALGORITHM_VERSION
        or item["verifier_version"] != _m30.VERIFIER_VERSION
        or item["status"] != _m30.EXACT_CORRESPONDENCE_COMPLETE
        or item["coverage"] != "complete"
        or item["partial_response"] is not False
    ):
        _m30_invalid("M30 response version/status/coverage is inconsistent")
    if (
        item["response_game_identity"] != expected_identities["response_game"]
        or item["response_run_identity"] != expected_identities["response_run"]
    ):
        _m30_invalid("M30 response game/run identity is inconsistent")

    content = _m30_mapping(item["content_identities"], "content_identities")
    expected_content = {
        "response_game_structure": reduced_game.response_game_structure_identity,
        "fixed_hero": reduced_game.fixed_hero_identity,
        "perfect_recall_evidence": reduced_game.perfect_recall_evidence_identity,
        "rake_convention": reduced_game.rake_convention_identity,
        "payoff_table": expected_identities["payoff_table"],
        "config": expected_identities["config"],
        "supplied_profile": reduced_game.supplied_profile_identity,
        "candidate": reduced_game.candidate_identity,
        "search_mode_context": reduced_game.search_mode_context_identity,
        "run_context": reduced_game.run_context_identity,
    }
    if content != expected_content or item["limits"] != asdict(limits):
        _m30_invalid("M30 content identity or effective limits are inconsistent")
    expected_ordering = {
        "plans": "caller-declared O1 order then caller-declared O2 order",
        "support_pairs": (
            "O1 support cardinality/lexicographic, then O2 support "
            "cardinality/lexicographic"
        ),
        "vertices": "lexicographic exact rational mixtures",
        "support_cells": (
            "minimum source support cardinality/lexicographic, then support_cell_id"
        ),
        "witnesses": "exact mixture then support_cell_id",
    }
    if item["ordering"] != expected_ordering:
        _m30_invalid("M30 deterministic ordering descriptor is inconsistent")

    o1_plan_ids = reduced_game.o1_plan_ids
    o2_plan_ids = reduced_game.o2_plan_ids
    payoff_by_pair = {
        (row.o1_plan_id, row.o2_plan_id): {
            account: Fraction(getattr(row, account)) for account in ALL_ACCOUNTS
        }
        for row in reduced_game.payoff_rows
    }
    expected_payoff_pairs = {
        (o1_plan_id, o2_plan_id)
        for o1_plan_id in o1_plan_ids
        for o2_plan_id in o2_plan_ids
    }
    if set(payoff_by_pair) != expected_payoff_pairs:
        _m30_invalid("M30 validation source payoff rectangle is incomplete")

    def expected_utility(
        o1_mixture: tuple[Fraction, ...],
        o2_mixture: tuple[Fraction, ...],
    ) -> dict[str, Fraction]:
        return {
            account: sum(
                (
                    o1_probability
                    * o2_probability
                    * payoff_by_pair[(o1_plan_id, o2_plan_id)][account]
                    for o1_plan_id, o1_probability in zip(
                        o1_plan_ids, o1_mixture
                    )
                    for o2_plan_id, o2_probability in zip(
                        o2_plan_ids, o2_mixture
                    )
                ),
                Fraction(0),
            )
            for account in ALL_ACCOUNTS
        }

    expected_support_pair_sequence = tuple(
        (o1_support, o2_support)
        for o1_size in range(1, len(o1_plan_ids) + 1)
        for o1_support in itertools.combinations(o1_plan_ids, o1_size)
        for o2_size in range(1, len(o2_plan_ids) + 1)
        for o2_support in itertools.combinations(o2_plan_ids, o2_size)
    )
    expected_support_pairs = set(expected_support_pair_sequence)
    expected_preflight_systems, expected_solved_systems = (
        _m30_reconstructed_system_counts(
            payoff_by_pair,
            o1_plan_ids,
            o2_plan_ids,
            expected_support_pair_sequence,
            limits.max_exact_linear_systems,
        )
    )
    audit = _m30_list(item["support_pair_audit"], "support_pair_audit")
    audit_by_pair: dict[tuple[tuple[str, ...], tuple[str, ...]], dict[str, Any]] = {}
    audit_sequence: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    nonempty_outcomes = {"new_support_cell", "exact_duplicate_support_cell"}
    empty_outcomes = {"empty_o1_mixture_polytope", "empty_o2_mixture_polytope"}
    for index, raw in enumerate(audit):
        record = _m30_mapping(raw, f"support_pair_audit[{index}]")
        if set(record) != {"support_pair", "outcome", "support_cell_id"}:
            _m30_invalid("M30 support-pair audit schema is inconsistent")
        pair = _m30_support_pair(
            record["support_pair"],
            o1_plan_ids,
            o2_plan_ids,
            f"support_pair_audit[{index}].support_pair",
        )
        if pair in audit_by_pair:
            _m30_invalid("M30 support-pair audit contains a duplicate")
        outcome = record["outcome"]
        if outcome not in nonempty_outcomes | empty_outcomes:
            _m30_invalid("M30 support-pair audit has an unknown outcome")
        if outcome in empty_outcomes and record["support_cell_id"] is not None:
            _m30_invalid("M30 empty support-pair outcome names a support cell")
        audit_by_pair[pair] = record
        audit_sequence.append(pair)
    if (
        set(audit_by_pair) != expected_support_pairs
        or tuple(audit_sequence) != expected_support_pair_sequence
    ):
        _m30_invalid("M30 support-pair audit is not complete")

    cells = _m30_list(item["support_cells"], "support_cells")
    cell_ids: set[str] = set()
    cell_id_sequence: list[str] = []
    o1_positions = {
        plan_id: index for index, plan_id in enumerate(o1_plan_ids)
    }
    o2_positions = {
        plan_id: index for index, plan_id in enumerate(o2_plan_ids)
    }

    def source_order_key(
        pair: tuple[tuple[str, ...], tuple[str, ...]],
    ) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
        return (
            len(pair[0]) + len(pair[1]),
            tuple(o1_positions[plan_id] for plan_id in pair[0]),
            tuple(o2_positions[plan_id] for plan_id in pair[1]),
        )

    source_to_cell: dict[tuple[tuple[str, ...], tuple[str, ...]], str] = {}
    cell_vertex_pairs: dict[
        str,
        set[tuple[tuple[Fraction, ...], tuple[Fraction, ...]]],
    ] = {}
    total_vertices = 0
    vertex_pairs = 0
    for cell_index, raw_cell in enumerate(cells):
        cell = _m30_mapping(raw_cell, f"support_cells[{cell_index}]")
        expected_cell_keys = {
            "support_cell_id",
            "representation",
            "kind",
            "dimension",
            "source_support_pairs",
            "o1_mixture_polytope",
            "o2_mixture_polytope",
        }
        if set(cell) != expected_cell_keys:
            _m30_invalid("M30 support cell schema is inconsistent")
        cell_id = cell["support_cell_id"]
        if (
            type(cell_id) is not str
            or _IDENTITY_RE.fullmatch(cell_id) is None
            or cell_id in cell_ids
        ):
            _m30_invalid("M30 support cell identity is invalid or duplicated")
        if cell["representation"] != (
            "Cartesian product of complete exact-rational bounded marginal "
            "polytope V-representations"
        ):
            _m30_invalid("M30 support cell representation is inconsistent")
        cell_ids.add(cell_id)
        cell_id_sequence.append(cell_id)
        sources = _m30_list(
            cell["source_support_pairs"],
            f"support_cells[{cell_index}].source_support_pairs",
        )
        if not sources:
            _m30_invalid("M30 support cell has no source support pair")
        parsed_sources = []
        for source_index, source in enumerate(sources):
            pair = _m30_support_pair(
                source,
                o1_plan_ids,
                o2_plan_ids,
                f"support_cells[{cell_index}].source_support_pairs[{source_index}]",
            )
            parsed_sources.append(pair)
            if pair in source_to_cell:
                _m30_invalid("M30 source support pair occurs in multiple cells")
            source_to_cell[pair] = cell_id
        canonical_sources = sorted(
            parsed_sources,
            key=source_order_key,
        )
        if parsed_sources != canonical_sources:
            _m30_invalid("M30 source support pairs are not canonically ordered")
        dimensions = []
        vertex_counts = []
        marginal_vertices: list[tuple[tuple[Fraction, ...], ...]] = []
        for player, plan_ids, key in (
            ("O1", o1_plan_ids, "o1_mixture_polytope"),
            ("O2", o2_plan_ids, "o2_mixture_polytope"),
        ):
            polytope = _m30_mapping(
                cell[key], f"support_cells[{cell_index}].{key}"
            )
            if set(polytope) != {"dimension", "vertex_count", "vertices"}:
                _m30_invalid("M30 marginal polytope schema is inconsistent")
            dimension = _m30_int(
                polytope["dimension"], f"{key}.dimension", maximum=len(plan_ids) - 1
            )
            vertices = _m30_list(polytope["vertices"], f"{key}.vertices")
            vertex_count = _m30_int(
                polytope["vertex_count"],
                f"{key}.vertex_count",
                minimum=1,
                maximum=limits.max_vertices_per_cell,
            )
            if vertex_count != len(vertices):
                _m30_invalid("M30 marginal vertex count is inconsistent")
            canonical_vertices = tuple(
                _m30_mixture(
                    vertex,
                    plan_ids,
                    f"support_cells[{cell_index}].{player}.vertices",
                    limits,
                )
                for vertex in vertices
            )
            if (
                len(set(canonical_vertices)) != vertex_count
                or canonical_vertices != tuple(sorted(canonical_vertices))
            ):
                _m30_invalid("M30 marginal vertices are duplicated")
            exact_dimension = _m30_affine_dimension(canonical_vertices)
            if dimension != exact_dimension:
                _m30_invalid(
                    "M30 marginal affine dimension does not match its vertices"
                )
            dimensions.append(dimension)
            vertex_counts.append(vertex_count)
            marginal_vertices.append(canonical_vertices)
        cell_dimension = _m30_int(
            cell["dimension"],
            f"support_cells[{cell_index}].dimension",
            maximum=len(o1_plan_ids) + len(o2_plan_ids) - 2,
        )
        if (
            cell_dimension != sum(dimensions)
            or cell["kind"]
            != ("singleton" if sum(dimensions) == 0 else "continuum")
        ):
            _m30_invalid("M30 support-cell dimension/kind is inconsistent")
        expected_cell_id = _identity(
            {
                "representation": "exact-rational-support-cell-v1",
                "o1_vertices": [
                    [_rational_text(value) for value in vertex]
                    for vertex in sorted(marginal_vertices[0])
                ],
                "o2_vertices": [
                    [_rational_text(value) for value in vertex]
                    for vertex in sorted(marginal_vertices[1])
                ],
            }
        )
        if cell_id != expected_cell_id:
            _m30_invalid("M30 support cell identity does not match its vertices")
        cell_vertex_pairs[cell_id] = set(
            itertools.product(marginal_vertices[0], marginal_vertices[1])
        )
        total_vertices += sum(vertex_counts)
        vertex_pairs += vertex_counts[0] * vertex_counts[1]
    if not cells:
        _m30_invalid("M30 complete response contains no support cells")
    for pair, audit_record in audit_by_pair.items():
        outcome = audit_record["outcome"]
        if outcome in nonempty_outcomes:
            if source_to_cell.get(pair) != audit_record["support_cell_id"]:
                _m30_invalid("M30 audit/support-cell source mapping is inconsistent")
        elif pair in source_to_cell:
            _m30_invalid("M30 empty audit pair appears in a support cell")
    if set(source_to_cell) != {
        pair
        for pair, record in audit_by_pair.items()
        if record["outcome"] in nonempty_outcomes
    }:
        _m30_invalid("M30 support-cell source coverage is incomplete")
    for cell_id in cell_ids:
        cell_sources = [
            pair for pair, source_cell_id in source_to_cell.items()
            if source_cell_id == cell_id
        ]
        new_sources = [
            pair
            for pair in cell_sources
            if audit_by_pair[pair]["outcome"] == "new_support_cell"
        ]
        if len(new_sources) != 1:
            _m30_invalid("M30 support cell must have exactly one new source pair")
    discovered_cell_ids = [
        record["support_cell_id"]
        for record in audit
        if record["outcome"] == "new_support_cell"
    ]
    canonical_cell_id_sequence = sorted(
        discovered_cell_ids,
        key=lambda cell_id: (
            min(
                source_order_key(pair)
                for pair, source_cell_id in source_to_cell.items()
                if source_cell_id == cell_id
            ),
            cell_id,
        ),
    )
    if cell_id_sequence != canonical_cell_id_sequence:
        _m30_invalid(
            "M30 support cells do not follow canonical audit-derived order"
        )

    counts = _m30_mapping(item["counts"], "counts")
    expected_count_keys = {
        "pure_plans",
        "joint_pure_profiles",
        "support_pairs_total",
        "support_pairs_visited",
        "raw_nonempty_support_cells",
        "exact_duplicate_support_cells",
        "canonical_support_cells",
        "total_marginal_vertices",
        "vertex_pairs_evaluated",
        "exact_linear_systems_preflight",
        "exact_linear_systems_solved",
        "pure_profile_unilateral_stability_rows",
        "output_records_projected",
    }
    if set(counts) != expected_count_keys:
        _m30_invalid("M30 count schema is incomplete or unknown")
    if counts["pure_plans"] != {"O1": len(o1_plan_ids), "O2": len(o2_plan_ids)}:
        _m30_invalid("M30 pure-plan counts are inconsistent")
    nonempty_count = sum(
        record["outcome"] in nonempty_outcomes for record in audit_by_pair.values()
    )
    duplicate_count = sum(
        record["outcome"] == "exact_duplicate_support_cell"
        for record in audit_by_pair.values()
    )
    expected_counts = {
        "joint_pure_profiles": len(o1_plan_ids) * len(o2_plan_ids),
        "support_pairs_total": len(expected_support_pairs),
        "support_pairs_visited": len(audit),
        "raw_nonempty_support_cells": nonempty_count,
        "exact_duplicate_support_cells": duplicate_count,
        "canonical_support_cells": len(cells),
        "total_marginal_vertices": total_vertices,
        "vertex_pairs_evaluated": vertex_pairs,
    }
    for key, expected in expected_counts.items():
        if counts[key] != expected:
            _m30_invalid(f"M30 count {key} is inconsistent")
    preflight_systems = _m30_int(
        counts["exact_linear_systems_preflight"],
        "counts.exact_linear_systems_preflight",
        maximum=limits.max_exact_linear_systems,
    )
    solved_systems = _m30_int(
        counts["exact_linear_systems_solved"],
        "counts.exact_linear_systems_solved",
        maximum=preflight_systems,
    )
    if (
        preflight_systems != expected_preflight_systems
        or solved_systems != expected_solved_systems
    ):
        _m30_invalid("M30 exact-linear-system counts are inconsistent")
    projected_records = _m30_int(
        counts["output_records_projected"],
        "counts.output_records_projected",
        minimum=1,
        maximum=limits.max_output_records,
    )

    vertex_pair_cells: dict[
        tuple[tuple[Fraction, ...], tuple[Fraction, ...]], set[str]
    ] = {}
    for cell_id, pairs in cell_vertex_pairs.items():
        for pair in pairs:
            vertex_pair_cells.setdefault(pair, set()).add(cell_id)
    if not vertex_pair_cells:
        _m30_invalid("M30 response has no canonical vertex pair")
    vertex_pair_utilities = {
        pair: expected_utility(*pair) for pair in vertex_pair_cells
    }

    extrema = _m30_mapping(item["utility_extrema"], "utility_extrema")
    if set(extrema) != set(ALL_ACCOUNTS):
        _m30_invalid("M30 extrema must contain H/O1/O2/R")
    for account in ALL_ACCOUNTS:
        record = _m30_mapping(extrema[account], f"utility_extrema.{account}")
        if set(record) != {
            "minimum",
            "maximum",
            "minimum_witnesses",
            "maximum_witnesses",
            "witness_scope",
        }:
            _m30_invalid("M30 extremum schema is inconsistent")
        minimum = _m30_fraction(
            record["minimum"], f"utility_extrema.{account}.minimum", limits
        )
        maximum = _m30_fraction(
            record["maximum"], f"utility_extrema.{account}.maximum", limits
        )
        expected_minimum = min(
            utility[account] for utility in vertex_pair_utilities.values()
        )
        expected_maximum = max(
            utility[account] for utility in vertex_pair_utilities.values()
        )
        if (
            minimum != expected_minimum
            or maximum != expected_maximum
            or record["witness_scope"]
            != "all_canonical_vertex_pair_tie_witnesses"
        ):
            _m30_invalid("M30 extremum range or witness scope is inconsistent")
        for direction, expected_value in (
            ("minimum", minimum),
            ("maximum", maximum),
        ):
            witnesses = _m30_list(
                record[f"{direction}_witnesses"],
                f"utility_extrema.{account}.{direction}_witnesses",
            )
            if not witnesses:
                _m30_invalid("M30 required extremum witness list is empty")
            witnessed_pairs: set[
                tuple[tuple[Fraction, ...], tuple[Fraction, ...]]
            ] = set()
            witnessed_pair_sequence: list[
                tuple[tuple[Fraction, ...], tuple[Fraction, ...]]
            ] = []
            for witness_index, raw_witness in enumerate(witnesses):
                witness = _m30_mapping(
                    raw_witness,
                    f"utility_extrema.{account}.{direction}[{witness_index}]",
                )
                required_witness = {
                    "support_cell_ids",
                    "support_cell_id",
                    "o1_support",
                    "o2_support",
                    "o1_mixture",
                    "o2_mixture",
                    "utility",
                    "unilateral_residual",
                    "extremum_value",
                }
                if set(witness) != required_witness:
                    _m30_invalid("M30 extremum witness schema is inconsistent")
                witness_cells = _m30_list(
                    witness["support_cell_ids"], "witness.support_cell_ids"
                )
                if (
                    not witness_cells
                    or witness_cells != sorted(set(witness_cells))
                    or any(cell_id not in cell_ids for cell_id in witness_cells)
                    or witness["support_cell_id"] != witness_cells[0]
                ):
                    _m30_invalid("M30 witness support-cell identity is inconsistent")
                o1_mixture = _m30_mixture(
                    witness["o1_mixture"], o1_plan_ids, "witness.o1_mixture", limits
                )
                o2_mixture = _m30_mixture(
                    witness["o2_mixture"], o2_plan_ids, "witness.o2_mixture", limits
                )
                mixture_pair = (o1_mixture, o2_mixture)
                expected_cells = sorted(vertex_pair_cells.get(mixture_pair, set()))
                if witness["o1_support"] != [
                    plan_id
                    for plan_id, probability in zip(o1_plan_ids, o1_mixture)
                    if probability > 0
                ] or witness["o2_support"] != [
                    plan_id
                    for plan_id, probability in zip(o2_plan_ids, o2_mixture)
                    if probability > 0
                ]:
                    _m30_invalid("M30 witness positive support is inconsistent")
                utility = _m30_utility(witness["utility"], "witness.utility", limits)
                residual = _m30_mapping(
                    witness["unilateral_residual"],
                    "witness.unilateral_residual",
                )
                if residual != {"O1": "0", "O2": "0"}:
                    _m30_invalid("M30 witness unilateral residual is nonzero")
                if (
                    _m30_fraction(
                        witness["extremum_value"], "witness.extremum_value", limits
                    )
                    != expected_value
                    or utility[account] != expected_value
                    or utility != expected_utility(o1_mixture, o2_mixture)
                    or witness_cells != expected_cells
                    or mixture_pair in witnessed_pairs
                ):
                    _m30_invalid("M30 witness does not attain its extremum")
                witnessed_pairs.add(mixture_pair)
                witnessed_pair_sequence.append(mixture_pair)
            expected_witness_pairs = {
                pair
                for pair, utility in vertex_pair_utilities.items()
                if utility[account] == expected_value
            }
            if (
                witnessed_pairs != expected_witness_pairs
                or witnessed_pair_sequence != sorted(expected_witness_pairs)
            ):
                _m30_invalid("M30 extremum tie-witness coverage is incomplete")
    if (
        item["hero_worst"] != extrema["H"]["minimum"]
        or item["hero_best"] != extrema["H"]["maximum"]
        or item["hero_worst_witnesses"] != extrema["H"]["minimum_witnesses"]
        or item["hero_best_witnesses"] != extrema["H"]["maximum_witnesses"]
    ):
        _m30_invalid("M30 Hero extrema aliases are inconsistent")

    pure = _m30_mapping(
        item["pure_profile_unilateral_stability"],
        "pure_profile_unilateral_stability",
    )
    if set(pure) != {"coverage", "profile_count", "rows", "residual_semantics"}:
        _m30_invalid("M30 pure subset schema is inconsistent")
    pure_rows = _m30_list(pure["rows"], "pure_profile_unilateral_stability.rows")
    expected_pure_rows = []
    for o1_plan_id in o1_plan_ids:
        for o2_plan_id in o2_plan_ids:
            utility = payoff_by_pair[(o1_plan_id, o2_plan_id)]
            if (
                utility["O1"]
                >= max(
                    payoff_by_pair[(alternative, o2_plan_id)]["O1"]
                    for alternative in o1_plan_ids
                )
                and utility["O2"]
                >= max(
                    payoff_by_pair[(o1_plan_id, alternative)]["O2"]
                    for alternative in o2_plan_ids
                )
            ):
                expected_pure_rows.append(
                    {
                        "profile_id": f"{o1_plan_id}|{o2_plan_id}",
                        "plans": {"O1": o1_plan_id, "O2": o2_plan_id},
                        "utility": {
                            account: _rational_text(utility[account])
                            for account in ALL_ACCOUNTS
                        },
                        "unilateral_residual": {"O1": "0", "O2": "0"},
                    }
                )
    expected_pure = {
        "coverage": "complete",
        "profile_count": len(expected_pure_rows),
        "rows": expected_pure_rows,
        "residual_semantics": (
            "max(0,best_response_value-current_value), including the current "
            "strategy among best-response candidates"
        ),
    }
    if (
        pure != expected_pure
        or counts["pure_profile_unilateral_stability_rows"]
        != len(expected_pure_rows)
    ):
        _m30_invalid("M30 pure subset coverage/content/count is inconsistent")

    stress = _m30_mapping(
        item["hero_min_joint_plan_stress"], "hero_min_joint_plan_stress"
    )
    if set(stress) != {
        "status",
        "coverage",
        "diagnostic_identity",
        "hero_minimum",
        "witnesses",
        "hero_worst_response_minus_stress",
        "primary_response_status_influence",
        "opponent_individual_rationality_not_required",
        "transferable_utility_assumed",
        "coalition_equilibrium_claim",
    }:
        _m30_invalid("M30 coalition stress schema is inconsistent")
    expected_stress_identity = _identity(
        {
            "diagnostic": "hero-min-joint-plan-stress-v1",
            "response_game_identity": expected_identities["response_game"],
        }
    )
    stress_minimum = min(
        utility["H"] for utility in payoff_by_pair.values()
    )
    expected_stress_witnesses = []
    for o1_plan_id in o1_plan_ids:
        for o2_plan_id in o2_plan_ids:
            utility = payoff_by_pair[(o1_plan_id, o2_plan_id)]
            if utility["H"] == stress_minimum:
                expected_stress_witnesses.append(
                    {
                        "profile_id": f"{o1_plan_id}|{o2_plan_id}",
                        "plans": {"O1": o1_plan_id, "O2": o2_plan_id},
                        "utility": {
                            account: _rational_text(utility[account])
                            for account in ALL_ACCOUNTS
                        },
                    }
                )
    hero_worst = _m30_fraction(item["hero_worst"], "hero_worst", limits)
    expected_stress = {
        "status": "COMPLETE",
        "coverage": "complete",
        "diagnostic_identity": expected_stress_identity,
        "hero_minimum": _rational_text(stress_minimum),
        "witnesses": expected_stress_witnesses,
        "hero_worst_response_minus_stress": _rational_text(
            hero_worst - stress_minimum
        ),
        "primary_response_status_influence": False,
        "opponent_individual_rationality_not_required": True,
        "transferable_utility_assumed": False,
        "coalition_equilibrium_claim": False,
    }
    if stress != expected_stress:
        _m30_invalid("M30 coalition separation/content is inconsistent")

    certificate = _m30_mapping(
        item["unilateral_residual_certificate"],
        "unilateral_residual_certificate",
    )
    if (
        item["unilateral_residual_semantics"]
        != (
            "exact nonnegative max(0,best-response value-current value); "
            "R is excluded from response conditions"
        )
        or certificate
        != {
            "scope": (
                "all support cells by exact best-response constraints; all "
                "canonical marginal vertex pairs independently verified"
            ),
            "maximum": {"O1": "0", "O2": "0"},
        }
    ):
        _m30_invalid("M30 unilateral residual certificate is incomplete")
    expected_projected_records = 256 + len(audit)
    expected_projected_records += sum(
        8
        + len(cell["source_support_pairs"])
        + cell["o1_mixture_polytope"]["vertex_count"]
        + cell["o2_mixture_polytope"]["vertex_count"]
        for cell in cells
    )
    expected_projected_records += len(expected_pure_rows)
    expected_projected_records += len(expected_stress_witnesses)
    expected_projected_records += sum(
        len(extrema[account]["minimum_witnesses"])
        + len(extrema[account]["maximum_witnesses"])
        for account in ALL_ACCOUNTS
    )
    if projected_records != expected_projected_records:
        _m30_invalid("M30 projected output-record count is inconsistent")
    verifier = _m30_mapping(
        item["independent_verification"], "independent_verification"
    )
    if set(verifier) != {
        "version",
        "status",
        "support_pairs_reconstructed",
        "support_cells_reconstructed",
        "vertex_pairs_verified",
        "operations_used",
        "solver_helpers_reused",
    }:
        _m30_invalid("M30 independent-verification schema is inconsistent")
    if (
        verifier["version"] != _m30.VERIFIER_VERSION
        or verifier["status"] != "VERIFIED"
        or verifier["support_pairs_reconstructed"] != len(audit)
        or verifier["support_cells_reconstructed"] != len(cells)
        or verifier["vertex_pairs_verified"] != vertex_pairs
        or verifier["solver_helpers_reused"] is not False
    ):
        _m30_invalid("M30 independent verification is incomplete")
    _m30_int(
        verifier["operations_used"],
        "independent_verification.operations_used",
        maximum=limits.max_verifier_operations,
    )
    return projected_records


def _build_m31_owned_payloads(
    *,
    scenario: ThreePlayerRiverRakeScenario,
    structure: _Structure,
    attestation: PerfectRecallAttestation,
    limits: RiverRakeLimits,
    caller_m30_limits: ExactResponseLimits,
    effective_m30_limits: ExactResponseLimits,
    fixed_hero_identity: str,
    initial_profile_identity: str | None,
    payoff_table_identity: str,
    evaluator_config_identity: str,
    response_game_identity: str,
    response_run_identity: str,
    perfect_recall_identity: str,
    o1_plan_ids: tuple[str, ...],
    o2_plan_ids: tuple[str, ...],
    o1_plans: tuple[dict[str, str], ...],
    o2_plans: tuple[dict[str, str], ...],
    output_rows: list[dict[str, Any]],
    independent_terminal_paths: int,
    initial_comparison: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payoff_table = {
        "payoff_table_identity": payoff_table_identity,
        "ordering": "O1 outer / O2 inner",
        "o1_plans": [
            {"plan_id": plan_id, "actions_by_information_set": plan}
            for plan_id, plan in zip(o1_plan_ids, o1_plans)
        ],
        "o2_plans": [
            {"plan_id": plan_id, "actions_by_information_set": plan}
            for plan_id, plan in zip(o2_plan_ids, o2_plans)
        ],
        "rows": output_rows,
        "rectangular_complete": True,
    }
    scenario_evaluation = {
        "contract_version": CONTRACT_VERSION,
        "accounting_version": ACCOUNTING_VERSION,
        "status": EXACT_SCENARIO_RESPONSE_COMPLETE,
        "identities": {
            "scenario": structure.scenario_identity,
            "tree_structure": structure.tree_structure_identity,
            "fixed_hero": fixed_hero_identity,
            "initial_profile": initial_profile_identity,
            "rake_convention": structure.rake_convention_identity,
            "structural_evidence": structure.structural_evidence_identity,
            "perfect_recall_evidence": perfect_recall_identity,
            "payoff_table": payoff_table_identity,
            "evaluator_config": evaluator_config_identity,
            "response_game": response_game_identity,
            "response_run": response_run_identity,
        },
        "counts": {
            "nodes": structure.node_count,
            "terminals": structure.terminal_count,
            "fixed_hero_information_sets": len(structure.info_sets["H"]),
            "o1_information_sets": len(structure.info_sets["O1"]),
            "o2_information_sets": len(structure.info_sets["O2"]),
            "opponent_information_sets_total": (
                len(structure.info_sets["O1"]) + len(structure.info_sets["O2"])
            ),
            "chance_outcomes": structure.chance_outcome_count,
            "private_observation_records": (
                structure.private_observation_records
            ),
            "terminal_contribution_records": (
                structure.terminal_contribution_records
            ),
            "pure_plans_o1": len(o1_plan_ids),
            "pure_plans_o2": len(o2_plan_ids),
            "joint_profiles": len(output_rows),
            "payoff_rows": len(output_rows),
            "independent_terminal_path_evaluations": (
                independent_terminal_paths
            ),
        },
        "ordering": {
            "seat_order": list(scenario.seat_order),
            "river_action_order": list(scenario.river_action_order),
            "chance_outcomes": "caller order",
            "actions": "caller order",
            "plans": "sorted information-set IDs / caller action order",
            "payoff_rows": "O1 outer / O2 inner",
        },
        "rake_convention": {
            "fold_rake": "0",
            "showdown_rake": "min(rate * pot_before_rake, cap)",
            "uncalled_return": "unique maximum before rake",
            "rounding_unit": None,
            "rounding_rule": "exact_none",
        },
        "terminal_records": structure.terminal_records,
        "perfect_recall": {
            "machine_evidence_version": STRUCTURAL_EVIDENCE_VERSION,
            "structural_evidence_identity": (
                structure.structural_evidence_identity
            ),
            "verified_occurrence_count": len(structure.occurrence_records),
            "all_occurrences_equal_within_information_set": True,
            "human_attestation": {
                "tree_content_identity": attestation.tree_content_identity,
                "structural_evidence_identity": (
                    attestation.structural_evidence_identity
                ),
                "o1_confirmed": True,
                "o2_confirmed": True,
                "verifier": attestation.verifier,
                "verification_date": attestation.verification_date,
                "evidence_version": attestation.evidence_version,
            },
        },
        "independent_evaluation": {
            "version": INDEPENDENT_EVALUATOR_VERSION,
            "verified_row_count": len(output_rows),
            "terminal_path_evaluation_count": independent_terminal_paths,
            "all_rows_exactly_equal": True,
            "evidence_identity": _identity(
                {
                    "version": INDEPENDENT_EVALUATOR_VERSION,
                    "payoff_table_identity": payoff_table_identity,
                    "verified_row_count": len(output_rows),
                    "terminal_path_evaluation_count": independent_terminal_paths,
                    "all_rows_exactly_equal": True,
                }
            ),
        },
        "initial_profile_comparison": initial_comparison,
        "limits": {
            "m31": asdict(limits),
            "caller_m30": asdict(caller_m30_limits),
            "effective_m30": asdict(effective_m30_limits),
        },
        "m32_handoff": {
            "m32_usable": True,
            "requires_status": _m30.EXACT_CORRESPONDENCE_COMPLETE,
            "partial": False,
        },
    }
    return scenario_evaluation, payoff_table


def _solve(
    scenario: ThreePlayerRiverRakeScenario,
    fixed_hero_policy: ExactBehaviorPolicy,
    attestation: PerfectRecallAttestation | None,
    initial_profile: OpponentInitialProfile | None,
    limits: RiverRakeLimits,
    m30_limits: ExactResponseLimits,
    pins: RiverRakeIdentityPins,
    response_pins: ResponseIdentityPins,
) -> ScenarioResponseResult:
    _validate_limits(limits)
    _validate_m30_limits(m30_limits)
    _validate_pins(pins)
    _validate_response_pins(response_pins)
    effective, effective_m30_limits = _effective_limits(limits, m30_limits)
    effective_limits = replace(
        limits,
        max_pure_plans_o1=effective["max_pure_plans_o1"],
        max_pure_plans_o2=effective["max_pure_plans_o2"],
        max_joint_pure_profiles=effective["max_joint_pure_profiles"],
        max_rational_numerator_bits=effective["max_rational_numerator_bits"],
        max_rational_denominator_bits=effective["max_rational_denominator_bits"],
        max_output_records=effective["max_output_records"],
        max_output_bytes=effective["max_output_bytes"],
    )
    identity_budget = _IdentityRecordBudget(effective_limits.max_identity_records)
    structure = _prepare_structure(
        scenario, effective_limits, identity_budget
    )
    o1_count = _plan_count(
        structure.info_sets["O1"],
        effective["max_pure_plans_o1"],
        "plans.O1",
    )
    o2_count = _plan_count(
        structure.info_sets["O2"],
        effective["max_pure_plans_o2"],
        "plans.O2",
    )
    joint_count = _checked_product(
        o1_count,
        o2_count,
        effective["max_joint_pure_profiles"],
        "plans.joint",
    )
    if joint_count > limits.max_payoff_rows:
        raise _ScenarioFailure(
            CAP_EXCEEDED, "payoff_table", "max_payoff_rows exceeded"
        )
    _checked_product(
        joint_count,
        structure.terminal_count,
        limits.max_terminal_evaluations,
        "evaluation",
    )
    for count in (o1_count, o2_count, joint_count):
        identity_budget.reserve(count)
    _finalize_structure_identities(structure)
    _check_pin(pins, "scenario_identity", structure.scenario_identity)
    _check_pin(
        pins, "tree_structure_identity", structure.tree_structure_identity
    )
    _check_pin(
        pins, "rake_convention_identity", structure.rake_convention_identity
    )
    perfect_recall_identity = _validate_attestation(attestation, structure)
    _check_pin(
        pins, "perfect_recall_evidence_identity", perfect_recall_identity
    )

    if type(fixed_hero_policy) is not ExactBehaviorPolicy:
        raise _ScenarioFailure(
            INVALID_INPUT,
            "fixed_hero_policy",
            "fixed Hero policy must be ExactBehaviorPolicy",
        )
    hero_policy, hero_projection = _validate_policy(
        fixed_hero_policy.probabilities,
        structure.info_sets["H"],
        "fixed_hero_policy",
        effective_limits,
    )
    fixed_hero_identity = _identity(
        {
            "tree_structure_identity": structure.tree_structure_identity,
            "complete_policy": hero_projection,
        }
    )
    _check_pin(pins, "fixed_hero_identity", fixed_hero_identity)

    parsed_initial: dict[
        str, dict[str, dict[str, Fraction]]
    ] | None = None
    initial_projection: dict[str, Any] | None = None
    initial_profile_identity: str | None = None
    if initial_profile is not None:
        if type(initial_profile) is not OpponentInitialProfile:
            raise _ScenarioFailure(
                INVALID_INPUT,
                "initial_profile",
                "initial profile must be OpponentInitialProfile",
            )
        if (
            initial_profile.o1_probabilities is None
            or initial_profile.o2_probabilities is None
        ):
            raise _ScenarioFailure(
                INVALID_INPUT,
                "initial_profile",
                "both O1 and O2 complete profiles are required",
            )
        o1_initial, o1_projection = _validate_policy(
            initial_profile.o1_probabilities,
            structure.info_sets["O1"],
            "initial_profile.O1",
            effective_limits,
        )
        o2_initial, o2_projection = _validate_policy(
            initial_profile.o2_probabilities,
            structure.info_sets["O2"],
            "initial_profile.O2",
            effective_limits,
        )
        parsed_initial = {"O1": o1_initial, "O2": o2_initial}
        initial_projection = {"O1": o1_projection, "O2": o2_projection}
        initial_profile_identity = _identity(
            {
                "tree_structure_identity": structure.tree_structure_identity,
                "complete_profiles": initial_projection,
            }
        )
    _check_pin(pins, "initial_profile_identity", initial_profile_identity)

    m31_output_records, minimum_m30_output_records = (
        _preflight_outer_output_allocation(
            structure,
            o1_count,
            o2_count,
            joint_count,
            include_initial_comparison=parsed_initial is not None,
            limits=effective_limits,
            effective_m30_limits=effective_m30_limits,
        )
    )
    remaining_m30_records = (
        effective["max_output_records"] - m31_output_records
    )
    if remaining_m30_records < minimum_m30_output_records:
        raise _ScenarioFailure(
            CAP_EXCEEDED,
            "output",
            "combined M31/M30 output record budget is insufficient",
        )
    effective_m30_limits = replace(
        effective_m30_limits,
        max_output_records=min(
            effective_m30_limits.max_output_records,
            remaining_m30_records,
        ),
    )
    minimum_success_bytes = len(
        _canonical_json_bytes(
            {
                "status": EXACT_SCENARIO_RESPONSE_COMPLETE,
                "scenario_evaluation": {},
                "payoff_table": {},
                "response": {},
                "error": None,
                "partial_result": False,
            }
        )
    )
    if minimum_success_bytes > effective["max_output_bytes"]:
        raise _ScenarioFailure(
            CAP_EXCEEDED,
            "output",
            "max_output_bytes exceeded before output-row allocation",
        )

    o1_plan_ids, o1_plans = _materialize_plans(
        "O1", structure.info_sets["O1"], o1_count
    )
    o2_plan_ids, o2_plans = _materialize_plans(
        "O2", structure.info_sets["O2"], o2_count
    )
    payoff_rows: list[ExactPayoffRow] = []
    output_rows: list[dict[str, Any]] = []
    independent_terminal_paths = 0
    for o1_index, (o1_plan_id, o1_plan) in enumerate(
        zip(o1_plan_ids, o1_plans)
    ):
        for o2_index, (o2_plan_id, o2_plan) in enumerate(
            zip(o2_plan_ids, o2_plans)
        ):
            primary = _primary_evaluate(
                scenario.root,
                structure,
                hero_policy,
                o1_plan,
                o2_plan,
                effective_limits,
            )
            independent, path_count = _independent_path_evaluate(
                scenario.root,
                structure,
                hero_policy,
                o1_plan,
                o2_plan,
                effective_limits,
            )
            independent_terminal_paths = _checked_sum(
                independent_terminal_paths,
                path_count,
                limits.max_terminal_evaluations,
                "independent",
            )
            if primary != independent:
                raise _ScenarioFailure(
                    NUMERIC_FAILURE,
                    "independent",
                    "primary recursion and independent path enumerator disagree",
                )
            _require_conservation(primary, effective_limits, "payoff_row")
            utility_record = _utility_record(primary)
            payoff_rows.append(
                ExactPayoffRow(
                    o1_plan_id=o1_plan_id,
                    o2_plan_id=o2_plan_id,
                    **utility_record,
                )
            )
            output_rows.append(
                {
                    "row_index": o1_index * o2_count + o2_index,
                    "o1_plan_id": o1_plan_id,
                    "o2_plan_id": o2_plan_id,
                    "utility": utility_record,
                    "conservation": "0",
                }
            )
    if len(payoff_rows) != joint_count:
        raise _ScenarioFailure(
            NUMERIC_FAILURE, "payoff_table", "rectangular row count mismatch"
        )
    if (
        len(o1_plan_ids) != o1_count
        or len(o2_plan_ids) != o2_count
        or len(set(o1_plan_ids)) != o1_count
        or len(set(o2_plan_ids)) != o2_count
    ):
        raise _ScenarioFailure(
            INVALID_INPUT,
            "plans",
            "pure-plan IDs are missing, duplicated, or colliding",
        )
    expected_pairs = tuple(
        (o1_plan_id, o2_plan_id)
        for o1_plan_id in o1_plan_ids
        for o2_plan_id in o2_plan_ids
    )
    actual_pairs = tuple(
        (row.o1_plan_id, row.o2_plan_id) for row in payoff_rows
    )
    if (
        actual_pairs != expected_pairs
        or len(set(actual_pairs)) != joint_count
        or any(
            o1_plan_id not in set(o1_plan_ids)
            or o2_plan_id not in set(o2_plan_ids)
            for o1_plan_id, o2_plan_id in actual_pairs
        )
    ):
        raise _ScenarioFailure(
            INVALID_INPUT,
            "payoff_table",
            "payoff rows must be one complete canonical rectangle",
        )
    payoff_projection = {
        "ordering": "O1 plan outer / O2 plan inner",
        "o1_plan_ids": list(o1_plan_ids),
        "o2_plan_ids": list(o2_plan_ids),
        "rows": output_rows,
        "fixed_hero_identity": fixed_hero_identity,
        "scenario_identity": structure.scenario_identity,
        "rake_convention_identity": structure.rake_convention_identity,
    }
    payoff_table_identity = _identity(payoff_projection)
    _check_pin(pins, "payoff_table_identity", payoff_table_identity)

    initial_comparison = None
    if parsed_initial is not None:
        behavior_profiles = {
            "H": hero_policy,
            "O1": parsed_initial["O1"],
            "O2": parsed_initial["O2"],
        }
        initial_utility = _behavior_evaluate(
            scenario.root, structure, behavior_profiles, effective_limits
        )
        _require_conservation(
            initial_utility, effective_limits, "initial_profile_comparison"
        )
        initial_comparison = {
            "initial_profile_identity": initial_profile_identity,
            "utility": _utility_record(initial_utility),
            "conservation": "0",
            "response_claim": False,
        }

    original_m30_output_bytes = effective_m30_limits.max_output_bytes
    scenario_evaluation: dict[str, Any]
    payoff_table: dict[str, Any]
    reduced_game: ExactReducedGame
    expected_m30_identities: dict[str, str]
    evaluator_config_identity = ""
    owned_output_bytes = 0
    for _ in range(16):
        evaluator_config_identity = _identity(
            {
                "contract_version": CONTRACT_VERSION,
                "independent_evaluator_version": INDEPENDENT_EVALUATOR_VERSION,
                "m31_limits": asdict(limits),
                "caller_m30_limits": asdict(m30_limits),
                "effective_m30_limits": asdict(effective_m30_limits),
            }
        )
        reduced_game = ExactReducedGame(
            o1_plan_ids=o1_plan_ids,
            o2_plan_ids=o2_plan_ids,
            payoff_rows=tuple(payoff_rows),
            response_game_structure_identity=structure.tree_structure_identity,
            fixed_hero_identity=fixed_hero_identity,
            perfect_recall_evidence_identity=perfect_recall_identity,
            rake_convention_identity=structure.rake_convention_identity,
            supplied_profile_identity=initial_profile_identity,
            candidate_identity=None,
            search_mode_context_identity=None,
            run_context_identity=evaluator_config_identity,
        )
        expected_m30_identities = _expected_m30_identities(
            reduced_game, effective_m30_limits
        )
        scenario_evaluation, payoff_table = _build_m31_owned_payloads(
            scenario=scenario,
            structure=structure,
            attestation=attestation,
            limits=limits,
            caller_m30_limits=m30_limits,
            effective_m30_limits=effective_m30_limits,
            fixed_hero_identity=fixed_hero_identity,
            initial_profile_identity=initial_profile_identity,
            payoff_table_identity=payoff_table_identity,
            evaluator_config_identity=evaluator_config_identity,
            response_game_identity=expected_m30_identities["response_game"],
            response_run_identity=expected_m30_identities["response_run"],
            perfect_recall_identity=perfect_recall_identity,
            o1_plan_ids=o1_plan_ids,
            o2_plan_ids=o2_plan_ids,
            o1_plans=o1_plans,
            o2_plans=o2_plans,
            output_rows=output_rows,
            independent_terminal_paths=independent_terminal_paths,
            initial_comparison=initial_comparison,
        )
        owned_projection = {
            "status": EXACT_SCENARIO_RESPONSE_COMPLETE,
            "scenario_evaluation": scenario_evaluation,
            "payoff_table": payoff_table,
            "response": None,
            "error": None,
            "partial_result": False,
        }
        actual_owned_records = (
            _record_count(owned_projection, effective["max_output_records"]) - 1
        )
        if actual_owned_records != m31_output_records:
            raise _ScenarioFailure(
                NUMERIC_FAILURE,
                "output",
                "M31 exact output-record preflight disagrees with allocation",
            )
        owned_output_bytes = len(_canonical_json_bytes(owned_projection)) - len(
            b"null"
        )
        remaining_m30_bytes = effective["max_output_bytes"] - owned_output_bytes
        if remaining_m30_bytes < 1:
            raise _ScenarioFailure(
                CAP_EXCEEDED,
                "output",
                "M31-owned exact output envelope exhausts max_output_bytes",
            )
        next_m30_output_bytes = min(
            original_m30_output_bytes, remaining_m30_bytes
        )
        if next_m30_output_bytes == effective_m30_limits.max_output_bytes:
            break
        effective_m30_limits = replace(
            effective_m30_limits,
            max_output_bytes=next_m30_output_bytes,
        )
    else:
        raise _ScenarioFailure(
            NUMERIC_FAILURE,
            "output",
            "effective M30 byte-budget split did not converge",
        )

    _check_pin(pins, "evaluator_config_identity", evaluator_config_identity)
    minimum_m30_preflight_bytes = _minimum_m30_preflight_bytes(
        o1_count,
        o2_count,
        minimum_m30_output_records,
        effective_m30_limits,
    )
    if minimum_m30_preflight_bytes > effective_m30_limits.max_output_bytes:
        raise _ScenarioFailure(
            CAP_EXCEEDED,
            "output",
            "remaining output bytes cannot pass the M30 preflight",
        )

    m30_result = _m30.solve_three_player_response(
        reduced_game,
        limits=effective_m30_limits,
        pins=response_pins,
    )
    if (
        m30_result.status != _m30.EXACT_CORRESPONDENCE_COMPLETE
        or m30_result.response is None
        or m30_result.error is not None
        or m30_result.partial_response
    ):
        mapped = {
            _m30.CAP_EXCEEDED: CAP_EXCEEDED,
            _m30.STALE_INPUT: STALE_INPUT,
            _m30.INVALID_INPUT: INVALID_INPUT,
            _m30.UNSUPPORTED_MODEL: UNSUPPORTED_MODEL,
            _m30.NUMERIC_FAILURE: NUMERIC_FAILURE,
        }.get(m30_result.status, M30_RESPONSE_FAILURE)
        raise _ScenarioFailure(
            mapped,
            "m30_response",
            "M30 exact response did not return a complete correspondence",
            cause_status=m30_result.status,
        )
    response = m30_result.response
    m30_output_records = _validate_m30_response(
        response,
        reduced_game=reduced_game,
        limits=effective_m30_limits,
        expected_identities=expected_m30_identities,
    )
    if (
        m31_output_records + m30_output_records
        > effective["max_output_records"]
    ):
        _m30_invalid("M30 output record evidence exceeds the shared outer budget")
    if len(_canonical_json_bytes(response)) > effective_m30_limits.max_output_bytes:
        _m30_invalid("M30 response exceeds its exact allocated byte budget")

    outer_projection = {
        "status": EXACT_SCENARIO_RESPONSE_COMPLETE,
        "scenario_evaluation": scenario_evaluation,
        "payoff_table": payoff_table,
        "response": response,
        "error": None,
        "partial_result": False,
    }
    encoded = _canonical_json_bytes(outer_projection)
    if len(encoded) > effective["max_output_bytes"]:
        raise _ScenarioFailure(
            NUMERIC_FAILURE,
            "output",
            "exact outer byte-budget split invariant failed",
        )
    return ScenarioResponseResult(
        status=EXACT_SCENARIO_RESPONSE_COMPLETE,
        scenario_evaluation=scenario_evaluation,
        payoff_table=payoff_table,
        response=response,
        error=None,
        partial_result=False,
    )


def evaluate_three_player_river_rake(
    scenario: ThreePlayerRiverRakeScenario,
    fixed_hero_policy: ExactBehaviorPolicy,
    *,
    attestation: PerfectRecallAttestation | None,
    initial_profile: OpponentInitialProfile | None = None,
    limits: RiverRakeLimits = RiverRakeLimits(),
    m30_limits: ExactResponseLimits = ExactResponseLimits(),
    pins: RiverRakeIdentityPins = RiverRakeIdentityPins(),
    response_pins: ResponseIdentityPins = ResponseIdentityPins(),
) -> ScenarioResponseResult:
    """Evaluate the complete bounded exact scenario or return no payload."""

    try:
        return _solve(
            scenario,
            fixed_hero_policy,
            attestation,
            initial_profile,
            limits,
            m30_limits,
            pins,
            response_pins,
        )
    except _ScenarioFailure as exc:
        return _failure(exc)
    except Exception:
        return _failure(
            _ScenarioFailure(
                INTERNAL_FAILURE,
                "internal",
                "unexpected three-player river/rake evaluation failure",
            )
        )


def exact_scenario_response_json(result: ScenarioResponseResult) -> str:
    """Serialize one outer result as strict deterministic one-line JSON."""

    if type(result) is not ScenarioResponseResult:
        raise TypeError("result must be ScenarioResponseResult")
    return _canonical_json_bytes(result.to_dict()).decode("utf-8")
