"""Bounded prepared abstract one- or two-street heads-up betting games.

This module builds a finite :class:`~repeated_poker.game.GameTree` from flat,
already-prepared abstract tables.  The original contract uses factorized Hero
and Villain root weights; the joint-root v2 contract accepts an explicit sparse
joint distribution without normalizing or filling impossible pairs.  It
supports fixed-profile evaluation and an exact response to a fixed Hero through
compatible downstream adapters.  It does not parse scenario JSON, evaluate real
cards, certify an external solver, or compute an equilibrium.

The builder is deliberately fail closed.  It validates identities, accounting,
observation/recall conditions, exact table coverage, and projected allocation
counts before it creates any generic game-tree or terminal object.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .game import (
    ChanceNode,
    GameTree,
    HeroNode,
    TerminalNode,
    VillainNode,
    collect_hero_info_sets,
    collect_villain_info_sets,
    iter_nodes,
    validate_tree,
)
from .payoffs import HERO, VILLAIN, make_equity_showdown_terminal, make_fold_terminal


PREPARED_TWO_STREET_CONTRACT_VERSION = "betting-tree-v2-prepared-two-street-1"
PREPARED_TWO_STREET_BUILDER_ID = "betting-tree-v2-prepared-two-street-builder-v1"
PREPARED_JOINT_ROOT_CONTRACT_VERSION = (
    "betting-tree-v2-prepared-two-street-joint-root-2"
)
PREPARED_JOINT_ROOT_BUILDER_ID = (
    "betting-tree-v2-prepared-two-street-joint-root-builder-v2"
)
PREPARED_JOINT_ROOT_DISTRIBUTION_ID = "explicit-positive-joint-root-matchups-v1"
PREPARED_CHANCE_NORMALIZATION_ID = "positive-fsum-normalize-v1"
PREPARED_INFORMATION_KEY_ID = "canonical-private-public-recall-key-v1"

_ACTION_LABEL_ID = "betting-tree-v2-action-label-v1"
_CONTENT_IDENTITY_ID = "sha256-floathex-canonical-json-v1"
_ORDERED_TREE_ID = "betting-tree-v2-ordered-tree-sha256-v1"
_PROBABILITY_TOLERANCE = 1e-9


__all__ = [
    "PREPARED_TWO_STREET_CONTRACT_VERSION",
    "PREPARED_TWO_STREET_BUILDER_ID",
    "PREPARED_JOINT_ROOT_CONTRACT_VERSION",
    "PREPARED_JOINT_ROOT_BUILDER_ID",
    "PREPARED_JOINT_ROOT_DISTRIBUTION_ID",
    "PREPARED_CHANCE_NORMALIZATION_ID",
    "PREPARED_INFORMATION_KEY_ID",
    "PreparedPlayer",
    "PreparedActionKind",
    "PreparedRoundCloseReason",
    "PreparedTwoStreetStatus",
    "PreparedTwoStreetLimits",
    "PreparedContentIdentity",
    "PreparedDataAttestation",
    "PreparedHeadsUpChips",
    "PreparedRake",
    "PreparedBucket",
    "PreparedActionOption",
    "PreparedActionEvent",
    "PreparedStreetCloseEvent",
    "PreparedChanceEvent",
    "PreparedStreet",
    "PreparedDecisionMenu",
    "PreparedChanceEdge",
    "PreparedTransitionRow",
    "PreparedShowdownValue",
    "PreparedTwoStreetSpec",
    "PreparedRootMatchup",
    "PreparedJointRootTwoStreetSpec",
    "PreparedInformationSetKey",
    "PreparedInformationSetObservation",
    "PreparedChanceNormalizationRecord",
    "PreparedBuildCounts",
    "PreparedTwoStreetIdentity",
    "PreparedTwoStreetBuild",
    "PreparedBuildError",
    "PreparedTwoStreetBuildResult",
    "prepared_public_history_id",
    "prepared_semantic_sha256",
    "build_prepared_two_street_game",
]


class PreparedPlayer(str, Enum):
    HERO = "hero"
    VILLAIN = "villain"


class PreparedActionKind(str, Enum):
    CHECK = "check"
    FOLD = "fold"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"


class PreparedRoundCloseReason(str, Enum):
    CHECK_CHECK = "check-check"
    CALL = "call"
    FOLD = "fold"
    ALL_IN_CALL = "all-in-call"


class PreparedTwoStreetStatus(str, Enum):
    SUCCESS = "SUCCESS"
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_ACTION_GRAMMAR = "INVALID_ACTION_GRAMMAR"
    ACCOUNTING_MISMATCH = "ACCOUNTING_MISMATCH"
    UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"
    EMPTY_CHANCE_SUPPORT = "EMPTY_CHANCE_SUPPORT"
    INVALID_CHANCE_SUPPORT = "INVALID_CHANCE_SUPPORT"
    CONTENT_HASH_MISMATCH = "CONTENT_HASH_MISMATCH"
    INFORMATION_MODEL_MISMATCH = "INFORMATION_MODEL_MISMATCH"
    PERFECT_RECALL_ATTESTATION_MISSING = "PERFECT_RECALL_ATTESTATION_MISSING"
    CAP_EXCEEDED = "CAP_EXCEEDED"
    NUMERIC_FAILURE = "NUMERIC_FAILURE"
    NON_REPRODUCIBLE = "NON_REPRODUCIBLE"
    ORACLE_MISMATCH = "ORACLE_MISMATCH"
    UNSUPPORTED_DOWNSTREAM = "UNSUPPORTED_DOWNSTREAM"


@dataclass(frozen=True)
class PreparedTwoStreetLimits:
    max_streets: int = 2
    max_full_raises_per_street: int = 2
    max_actions_per_decision: int = 8
    max_depth_edges: int = 32
    max_root_matchups: int = 10_000
    max_transition_rows: int = 50_000
    max_chance_outcomes_per_row: int = 32
    max_total_chance_edges: int = 100_000
    max_decision_nodes: int = 100_000
    max_terminal_nodes: int = 100_000
    max_total_nodes: int = 250_000
    max_information_sets_per_player: int = 1_000
    max_information_sets_total: int = 2_000
    max_validation_trace_records: int = 10_000
    max_br_correspondence_materialization: int = 100_000


@dataclass(frozen=True)
class PreparedContentIdentity:
    raw_sha256: str
    semantic_sha256: str


@dataclass(frozen=True)
class PreparedDataAttestation:
    source: str
    bucket_semantics: str
    conditional_probability_semantics: str
    observation_mapping: str
    perfect_recall_attested: bool


@dataclass(frozen=True)
class PreparedHeadsUpChips:
    hero: float
    villain: float


@dataclass(frozen=True)
class PreparedRake:
    rate: float
    cap: float | None


@dataclass(frozen=True)
class PreparedBucket:
    bucket_id: str
    weight: float


@dataclass(frozen=True)
class PreparedActionOption:
    kind: PreparedActionKind
    size_id: str | None
    raise_to: float | None
    is_all_in: bool


@dataclass(frozen=True)
class PreparedActionEvent:
    street_id: str
    player: PreparedPlayer
    kind: PreparedActionKind
    size_id: str | None
    raise_to: float | None
    is_all_in: bool
    reopen: bool


@dataclass(frozen=True)
class PreparedStreetCloseEvent:
    street_id: str
    reason: PreparedRoundCloseReason


@dataclass(frozen=True)
class PreparedChanceEvent:
    transition_id: str
    public_outcome_id: str


_PublicEvent = PreparedActionEvent | PreparedStreetCloseEvent | PreparedChanceEvent


@dataclass(frozen=True)
class PreparedStreet:
    street_id: str
    label: str
    first_actor: PreparedPlayer
    min_open_bet: float


@dataclass(frozen=True)
class PreparedDecisionMenu:
    public_history_id: str
    street_id: str
    player: PreparedPlayer
    actions: tuple[PreparedActionOption, ...]


@dataclass(frozen=True)
class PreparedChanceEdge:
    public_outcome_id: str
    next_hero_bucket_id: str
    next_villain_bucket_id: str
    probability: float


@dataclass(frozen=True)
class PreparedTransitionRow:
    transition_id: str
    source_public_state_id: str
    hero_bucket_id: str
    villain_bucket_id: str
    edges: tuple[PreparedChanceEdge, ...]


@dataclass(frozen=True)
class PreparedShowdownValue:
    public_state_id: str
    hero_bucket_id: str
    villain_bucket_id: str
    hero_pot_share: float


@dataclass(frozen=True)
class PreparedTwoStreetSpec:
    contract_version: str
    attestation: PreparedDataAttestation
    starting_chips: PreparedHeadsUpChips
    initial_committed: PreparedHeadsUpChips
    rake: PreparedRake
    streets: tuple[PreparedStreet, ...]
    hero_buckets: tuple[PreparedBucket, ...]
    villain_buckets: tuple[PreparedBucket, ...]
    decision_menus: tuple[PreparedDecisionMenu, ...]
    transition_id: str | None
    transition_rows: tuple[PreparedTransitionRow, ...]
    showdown_values: tuple[PreparedShowdownValue, ...]


@dataclass(frozen=True)
class PreparedRootMatchup:
    """One positive Hero/Villain root-bucket joint probability."""

    hero_bucket_id: str
    villain_bucket_id: str
    probability: float


@dataclass(frozen=True)
class PreparedJointRootTwoStreetSpec(PreparedTwoStreetSpec):
    """Prepared v2 spec with an explicit, non-factorized root distribution.

    ``root_matchups`` may omit impossible bucket pairs.  Its positive
    probabilities must sum to one, cover every declared bucket, and reproduce
    the declared per-player bucket weights as marginals.  The builder never
    normalizes, fills, truncates, or factorizes this distribution.
    """

    root_matchups: tuple[PreparedRootMatchup, ...]


@dataclass(frozen=True)
class PreparedInformationSetKey:
    contract_version: str
    player: PreparedPlayer
    street_id: str
    own_current_bucket_id: str
    own_bucket_history: tuple[str, ...]
    public_history_id: str
    own_action_history: tuple[str, ...]


@dataclass(frozen=True)
class PreparedInformationSetObservation:
    info_set_id: str
    key: PreparedInformationSetKey
    public_observation_identity: str
    legal_action_labels: tuple[str, ...]


@dataclass(frozen=True)
class PreparedChanceNormalizationRecord:
    row_identity: tuple[str, str, str, str]
    edge_identities: tuple[tuple[str, str, str], ...]
    raw_sum: float
    normalization_factor: float
    effective_probabilities: tuple[float, ...]


@dataclass(frozen=True)
class PreparedBuildCounts:
    root_matchups: int
    transition_rows: int
    chance_edges: int
    decision_nodes: int
    terminal_nodes: int
    total_nodes: int
    max_depth_edges: int
    hero_information_sets: int
    villain_information_sets: int
    hero_pure_plans: int
    villain_pure_strategies: int


@dataclass(frozen=True)
class PreparedTwoStreetIdentity:
    contract_version: str
    builder_id: str
    action_label_id: str
    normalization_id: str
    information_key_id: str
    raw_sha256: str
    semantic_sha256: str
    ordered_tree_sha256: str
    run_identity: str


@dataclass(frozen=True)
class PreparedTwoStreetBuild:
    tree: GameTree
    identity: PreparedTwoStreetIdentity
    counts: PreparedBuildCounts
    chance_normalization: tuple[PreparedChanceNormalizationRecord, ...]
    information_sets: tuple[PreparedInformationSetObservation, ...]


@dataclass(frozen=True)
class PreparedBuildError:
    message: str
    phase: str


@dataclass(frozen=True)
class PreparedTwoStreetBuildResult:
    status: PreparedTwoStreetStatus
    build: PreparedTwoStreetBuild | None
    error: PreparedBuildError | None


_DEFAULT_LIMITS = PreparedTwoStreetLimits()
_LIMIT_CEILINGS = {
    field.name: getattr(_DEFAULT_LIMITS, field.name) for field in fields(_DEFAULT_LIMITS)
}


class _BuildFailure(Exception):
    def __init__(self, status: PreparedTwoStreetStatus, phase: str, message: str):
        super().__init__(message)
        self.status = status
        self.phase = phase
        self.message = message


def _fail(status: PreparedTwoStreetStatus, phase: str, message: str) -> None:
    raise _BuildFailure(status, phase, message)


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical numeric values must be finite")
        return {"__floathex__": value.hex()}
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": type(value).__name__,
            "fields": {field.name: _canonical_value(getattr(value, field.name)) for field in fields(value)},
        }
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("canonical mapping keys must be strings")
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    raise ValueError(f"unsupported canonical value type {type(value).__name__}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(char in "0123456789abcdef" for char in value)
    )


def prepared_public_history_id(events: tuple[_PublicEvent, ...]) -> str:
    """Return the canonical public-history digest for prepared public events."""

    if not isinstance(events, tuple):
        raise ValueError("events must be a tuple")
    allowed = (PreparedActionEvent, PreparedStreetCloseEvent, PreparedChanceEvent)
    if any(not isinstance(event, allowed) for event in events):
        raise ValueError("events contains an unsupported public event")
    return "sha256:" + _sha256(events)


def prepared_semantic_sha256(spec: PreparedTwoStreetSpec) -> str:
    """Return the floathex canonical semantic SHA-256 of a prepared spec."""

    if not isinstance(spec, PreparedTwoStreetSpec):
        raise ValueError("spec must be PreparedTwoStreetSpec")
    payload: dict[str, Any] = {"algorithm": _CONTENT_IDENTITY_ID, "spec": spec}
    if type(spec) is PreparedJointRootTwoStreetSpec:
        payload["root_distribution_id"] = PREPARED_JOINT_ROOT_DISTRIBUTION_ID
    return hashlib.sha256(
        _canonical_bytes(payload)
    ).hexdigest()


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(PreparedTwoStreetStatus.INVALID_INPUT, "input", f"{name} must be a non-empty string")
    return value


def _number(value: Any, name: str, *, minimum: float | None = None, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(PreparedTwoStreetStatus.INVALID_INPUT, "numeric", f"{name} must be a non-bool number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        _fail(PreparedTwoStreetStatus.NUMERIC_FAILURE, "numeric", f"{name} cannot be represented as binary64")
    if not math.isfinite(result):
        _fail(PreparedTwoStreetStatus.INVALID_INPUT, "numeric", f"{name} must be finite")
    if positive and result <= 0.0:
        _fail(PreparedTwoStreetStatus.INVALID_INPUT, "numeric", f"{name} must be strictly positive")
    if minimum is not None and result < minimum:
        _fail(PreparedTwoStreetStatus.INVALID_INPUT, "numeric", f"{name} must be at least {minimum}")
    return result


def _chance_probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(PreparedTwoStreetStatus.INVALID_INPUT, "chance", f"{name} must be a non-bool number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        _fail(PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT, "chance", f"{name} must be finite and strictly positive")
    if not math.isfinite(result) or result <= 0.0:
        _fail(PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT, "chance", f"{name} must be finite and strictly positive")
    return result


def _finite_derived(value: float, name: str, *, positive_reach: bool = False) -> float:
    if not math.isfinite(value):
        _fail(PreparedTwoStreetStatus.NUMERIC_FAILURE, "numeric", f"non-finite derived {name}")
    if positive_reach and value <= 0.0:
        _fail(PreparedTwoStreetStatus.NUMERIC_FAILURE, "numeric", f"positive {name} underflowed to zero")
    return value


def _validate_limits(limits: Any) -> PreparedTwoStreetLimits:
    if not isinstance(limits, PreparedTwoStreetLimits):
        _fail(PreparedTwoStreetStatus.INVALID_INPUT, "limits", "limits must be PreparedTwoStreetLimits")
    for name, ceiling in _LIMIT_CEILINGS.items():
        value = getattr(limits, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            _fail(PreparedTwoStreetStatus.INVALID_INPUT, "limits", f"{name} must be a strictly positive integer")
        if value > ceiling:
            _fail(PreparedTwoStreetStatus.INVALID_INPUT, "limits", f"{name} exceeds hard ceiling {ceiling}")
    return limits


def _enum(value: Any, enum_type: type[Enum], name: str) -> None:
    if not isinstance(value, enum_type):
        _fail(PreparedTwoStreetStatus.INVALID_INPUT, "input", f"{name} must be {enum_type.__name__}")


def _tuple(value: Any, item_type: type, name: str) -> tuple:
    if not isinstance(value, tuple) or any(not isinstance(item, item_type) for item in value):
        _fail(PreparedTwoStreetStatus.INVALID_INPUT, "input", f"{name} must be a tuple of {item_type.__name__}")
    return value


def _validate_outer(spec: Any, raw_input_bytes: Any, identity: Any, limits: Any) -> tuple[PreparedTwoStreetSpec, PreparedTwoStreetLimits]:
    if type(spec) not in (PreparedTwoStreetSpec, PreparedJointRootTwoStreetSpec):
        _fail(
            PreparedTwoStreetStatus.INVALID_INPUT,
            "input",
            "spec must be PreparedTwoStreetSpec or PreparedJointRootTwoStreetSpec",
        )
    if type(raw_input_bytes) is not bytes:
        _fail(PreparedTwoStreetStatus.INVALID_INPUT, "identity", "raw_input_bytes must be bytes")
    if not isinstance(identity, PreparedContentIdentity):
        _fail(PreparedTwoStreetStatus.INVALID_INPUT, "identity", "content_identity must be PreparedContentIdentity")
    limits = _validate_limits(limits)
    expected_contract = (
        PREPARED_JOINT_ROOT_CONTRACT_VERSION
        if type(spec) is PreparedJointRootTwoStreetSpec
        else PREPARED_TWO_STREET_CONTRACT_VERSION
    )
    if spec.contract_version != expected_contract:
        _fail(PreparedTwoStreetStatus.INVALID_INPUT, "contract", "unsupported prepared contract version")
    if not _valid_sha256(identity.raw_sha256) or not _valid_sha256(identity.semantic_sha256):
        _fail(PreparedTwoStreetStatus.INVALID_INPUT, "identity", "content hashes must be lowercase SHA-256 hex")

    if not isinstance(spec.attestation, PreparedDataAttestation):
        _fail(PreparedTwoStreetStatus.INVALID_INPUT, "attestation", "attestation has wrong type")
    for name in ("source", "bucket_semantics", "conditional_probability_semantics", "observation_mapping"):
        _require_string(getattr(spec.attestation, name), f"attestation.{name}")
    if type(spec.attestation.perfect_recall_attested) is not bool or not spec.attestation.perfect_recall_attested:
        _fail(
            PreparedTwoStreetStatus.PERFECT_RECALL_ATTESTATION_MISSING,
            "attestation",
            "perfect_recall_attested must be exactly True",
        )

    for obj, name in (
        (spec.starting_chips, "starting_chips"),
        (spec.initial_committed, "initial_committed"),
    ):
        if not isinstance(obj, PreparedHeadsUpChips):
            _fail(PreparedTwoStreetStatus.INVALID_INPUT, "input", f"{name} has wrong type")
        _number(obj.hero, f"{name}.hero", minimum=0.0)
        _number(obj.villain, f"{name}.villain", minimum=0.0)
    if spec.starting_chips.hero < spec.initial_committed.hero or spec.starting_chips.villain < spec.initial_committed.villain:
        _fail(PreparedTwoStreetStatus.INVALID_INPUT, "accounting", "initial committed exceeds starting chips")
    _finite_derived(spec.starting_chips.hero + spec.starting_chips.villain, "total starting chips")
    _finite_derived(spec.initial_committed.hero + spec.initial_committed.villain, "initial pot")

    if not isinstance(spec.rake, PreparedRake):
        _fail(PreparedTwoStreetStatus.INVALID_INPUT, "input", "rake has wrong type")
    rate = _number(spec.rake.rate, "rake.rate", minimum=0.0)
    if rate > 1.0:
        _fail(PreparedTwoStreetStatus.INVALID_INPUT, "numeric", "rake.rate must be within [0, 1]")
    if spec.rake.cap is not None:
        _number(spec.rake.cap, "rake.cap", minimum=0.0)

    _tuple(spec.streets, PreparedStreet, "streets")
    _tuple(spec.hero_buckets, PreparedBucket, "hero_buckets")
    _tuple(spec.villain_buckets, PreparedBucket, "villain_buckets")
    _tuple(spec.decision_menus, PreparedDecisionMenu, "decision_menus")
    _tuple(spec.transition_rows, PreparedTransitionRow, "transition_rows")
    _tuple(spec.showdown_values, PreparedShowdownValue, "showdown_values")
    if len(spec.streets) not in (1, 2):
        _fail(PreparedTwoStreetStatus.UNSUPPORTED_MODEL, "contract", "street count must be one or two")
    if len(spec.streets) > limits.max_streets:
        _fail(PreparedTwoStreetStatus.CAP_EXCEEDED, "limits", "street count exceeds max_streets")

    street_ids: set[str] = set()
    for index, street in enumerate(spec.streets):
        _require_string(street.street_id, f"streets[{index}].street_id")
        _require_string(street.label, f"streets[{index}].label")
        _enum(street.first_actor, PreparedPlayer, f"streets[{index}].first_actor")
        _number(street.min_open_bet, f"streets[{index}].min_open_bet", positive=True)
        if street.street_id in street_ids:
            _fail(PreparedTwoStreetStatus.INVALID_INPUT, "input", "duplicate street_id")
        street_ids.add(street.street_id)

    if type(spec) is PreparedJointRootTwoStreetSpec:
        _tuple(spec.root_matchups, PreparedRootMatchup, "root_matchups")
        if not spec.root_matchups:
            _fail(
                PreparedTwoStreetStatus.EMPTY_CHANCE_SUPPORT,
                "joint-root",
                "joint root matchup support must not be empty",
            )
        root_matchups = len(spec.root_matchups)
    else:
        root_matchups = len(spec.hero_buckets) * len(spec.villain_buckets)
    if root_matchups > limits.max_root_matchups:
        _fail(PreparedTwoStreetStatus.CAP_EXCEEDED, "flat-count", "root matchup cap exceeded")
    hero_ids = _validate_buckets(spec.hero_buckets, "hero_buckets")
    villain_ids = _validate_buckets(spec.villain_buckets, "villain_buckets")
    if hero_ids & villain_ids:
        _fail(PreparedTwoStreetStatus.INVALID_INPUT, "input", "Hero and Villain bucket ids must be disjoint")
    if type(spec) is PreparedJointRootTwoStreetSpec:
        _validate_joint_root(spec, hero_ids, villain_ids)

    if len(spec.streets) == 1:
        if spec.transition_id is not None or spec.transition_rows:
            _fail(PreparedTwoStreetStatus.UNSUPPORTED_MODEL, "transition", "one-street specs cannot contain a transition")
    else:
        _require_string(spec.transition_id, "transition_id")

    _validate_flat_entries(spec, street_ids, hero_ids, villain_ids, limits)

    raw_actual = hashlib.sha256(raw_input_bytes).hexdigest()
    if raw_actual != identity.raw_sha256:
        _fail(PreparedTwoStreetStatus.CONTENT_HASH_MISMATCH, "identity", "raw SHA-256 mismatch")
    try:
        semantic_actual = prepared_semantic_sha256(spec)
    except (ValueError, TypeError, OverflowError):
        _fail(PreparedTwoStreetStatus.INVALID_INPUT, "identity", "spec cannot be canonicalized")
    if semantic_actual != identity.semantic_sha256:
        _fail(PreparedTwoStreetStatus.CONTENT_HASH_MISMATCH, "identity", "semantic SHA-256 mismatch")
    return spec, limits


def _validate_buckets(buckets: tuple[PreparedBucket, ...], name: str) -> set[str]:
    if not buckets:
        _fail(PreparedTwoStreetStatus.INVALID_INPUT, "chance", f"{name} must not be empty")
    ids: set[str] = set()
    weights: list[float] = []
    for index, bucket in enumerate(buckets):
        bucket_id = _require_string(bucket.bucket_id, f"{name}[{index}].bucket_id")
        if bucket_id in ids:
            _fail(PreparedTwoStreetStatus.INVALID_INPUT, "chance", f"duplicate bucket id {bucket_id!r}")
        ids.add(bucket_id)
        weights.append(_number(bucket.weight, f"{name}[{index}].weight", positive=True))
    if abs(math.fsum(weights) - 1.0) > _PROBABILITY_TOLERANCE:
        _fail(PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT, "chance", f"{name} weights do not sum to one")
    return ids


def _validate_joint_root(
    spec: PreparedJointRootTwoStreetSpec,
    hero_ids: set[str],
    villain_ids: set[str],
) -> None:
    seen_pairs: set[tuple[str, str]] = set()
    seen_hero: set[str] = set()
    seen_villain: set[str] = set()
    hero_terms: dict[str, list[float]] = {bucket_id: [] for bucket_id in hero_ids}
    villain_terms: dict[str, list[float]] = {
        bucket_id: [] for bucket_id in villain_ids
    }
    probabilities: list[float] = []
    for index, matchup in enumerate(spec.root_matchups):
        hero_id = _require_string(
            matchup.hero_bucket_id,
            f"root_matchups[{index}].hero_bucket_id",
        )
        villain_id = _require_string(
            matchup.villain_bucket_id,
            f"root_matchups[{index}].villain_bucket_id",
        )
        if hero_id not in hero_ids or villain_id not in villain_ids:
            _fail(
                PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT,
                "joint-root",
                "joint root matchup references an unknown bucket",
            )
        pair = (hero_id, villain_id)
        if pair in seen_pairs:
            _fail(
                PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT,
                "joint-root",
                "duplicate joint root matchup",
            )
        seen_pairs.add(pair)
        seen_hero.add(hero_id)
        seen_villain.add(villain_id)
        probability = _chance_probability(
            matchup.probability,
            f"root_matchups[{index}].probability",
        )
        probabilities.append(probability)
        hero_terms[hero_id].append(probability)
        villain_terms[villain_id].append(probability)

    if seen_hero != hero_ids or seen_villain != villain_ids:
        _fail(
            PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT,
            "joint-root",
            "joint root support must give every declared bucket positive reach",
        )
    total = _finite_derived(math.fsum(probabilities), "joint root probability sum")
    if abs(total - 1.0) > _PROBABILITY_TOLERANCE:
        _fail(
            PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT,
            "joint-root",
            "joint root probabilities do not sum to one",
        )

    declared_hero = {bucket.bucket_id: bucket.weight for bucket in spec.hero_buckets}
    declared_villain = {
        bucket.bucket_id: bucket.weight for bucket in spec.villain_buckets
    }
    for bucket_id in hero_ids:
        marginal = _finite_derived(
            math.fsum(hero_terms[bucket_id]),
            "joint root Hero marginal",
            positive_reach=True,
        )
        if abs(marginal - declared_hero[bucket_id]) > _PROBABILITY_TOLERANCE:
            _fail(
                PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT,
                "joint-root",
                "joint root Hero marginal disagrees with bucket weight",
            )
    for bucket_id in villain_ids:
        marginal = _finite_derived(
            math.fsum(villain_terms[bucket_id]),
            "joint root Villain marginal",
            positive_reach=True,
        )
        if abs(marginal - declared_villain[bucket_id]) > _PROBABILITY_TOLERANCE:
            _fail(
                PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT,
                "joint-root",
                "joint root Villain marginal disagrees with bucket weight",
            )


def _validate_flat_entries(
    spec: PreparedTwoStreetSpec,
    street_ids: set[str],
    hero_ids: set[str],
    villain_ids: set[str],
    limits: PreparedTwoStreetLimits,
) -> None:
    if len(spec.transition_rows) > limits.max_transition_rows:
        _fail(PreparedTwoStreetStatus.CAP_EXCEEDED, "flat-count", "transition row count exceeds cap")
    total_input_edges = 0
    for index, menu in enumerate(spec.decision_menus):
        _require_string(menu.public_history_id, f"decision_menus[{index}].public_history_id")
        if menu.street_id not in street_ids:
            _fail(PreparedTwoStreetStatus.INVALID_INPUT, "menu", "decision menu has unknown street")
        _enum(menu.player, PreparedPlayer, f"decision_menus[{index}].player")
        _tuple(menu.actions, PreparedActionOption, f"decision_menus[{index}].actions")
        if not 1 <= len(menu.actions) <= limits.max_actions_per_decision:
            status = PreparedTwoStreetStatus.CAP_EXCEEDED if len(menu.actions) > limits.max_actions_per_decision else PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR
            _fail(status, "menu", "decision action count is outside the supported range")
        for action_index, option in enumerate(menu.actions):
            _validate_action_option(option, f"decision_menus[{index}].actions[{action_index}]")

    for row_index, row in enumerate(spec.transition_rows):
        _require_string(row.transition_id, f"transition_rows[{row_index}].transition_id")
        _require_string(row.source_public_state_id, f"transition_rows[{row_index}].source_public_state_id")
        _require_string(row.hero_bucket_id, f"transition_rows[{row_index}].hero_bucket_id")
        _require_string(row.villain_bucket_id, f"transition_rows[{row_index}].villain_bucket_id")
        _tuple(row.edges, PreparedChanceEdge, f"transition_rows[{row_index}].edges")
        if not row.edges:
            _fail(PreparedTwoStreetStatus.EMPTY_CHANCE_SUPPORT, "chance", "transition row has empty support")
        if len(row.edges) > limits.max_chance_outcomes_per_row:
            _fail(PreparedTwoStreetStatus.CAP_EXCEEDED, "flat-count", "per-row chance edge count exceeds cap")
        total_input_edges += len(row.edges)
        if total_input_edges > limits.max_total_chance_edges:
            _fail(PreparedTwoStreetStatus.CAP_EXCEEDED, "flat-count", "input chance edge count exceeds cap")
        for edge_index, edge in enumerate(row.edges):
            _require_string(edge.public_outcome_id, f"transition_rows[{row_index}].edges[{edge_index}].public_outcome_id")
            if edge.next_hero_bucket_id not in hero_ids or edge.next_villain_bucket_id not in villain_ids:
                _fail(PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT, "chance", "transition edge references unknown successor bucket")
            _chance_probability(
                edge.probability,
                f"transition_rows[{row_index}].edges[{edge_index}].probability",
            )

    for index, value in enumerate(spec.showdown_values):
        _require_string(value.public_state_id, f"showdown_values[{index}].public_state_id")
        if value.hero_bucket_id not in hero_ids or value.villain_bucket_id not in villain_ids:
            _fail(PreparedTwoStreetStatus.INVALID_INPUT, "showdown", "showdown value references unknown bucket")
        share = _number(value.hero_pot_share, f"showdown_values[{index}].hero_pot_share", minimum=0.0)
        if share > 1.0:
            _fail(PreparedTwoStreetStatus.INVALID_INPUT, "showdown", "hero_pot_share must be within [0, 1]")


def _validate_action_option(option: PreparedActionOption, name: str) -> None:
    _enum(option.kind, PreparedActionKind, f"{name}.kind")
    if type(option.is_all_in) is not bool:
        _fail(PreparedTwoStreetStatus.INVALID_INPUT, "menu", f"{name}.is_all_in must be bool")
    if option.kind in (PreparedActionKind.CHECK, PreparedActionKind.FOLD, PreparedActionKind.CALL):
        if option.size_id is not None or option.raise_to is not None or option.is_all_in:
            _fail(PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR, "menu", "passive action carries sizing or all-in metadata")
    else:
        _require_string(option.size_id, f"{name}.size_id")
        _number(option.raise_to, f"{name}.raise_to", positive=True)


def _action_label(option: PreparedActionOption) -> str:
    if option.kind in (PreparedActionKind.CHECK, PreparedActionKind.FOLD, PreparedActionKind.CALL):
        return option.kind.value
    return f"{option.kind.value}::{option.size_id}"


def _other(player: PreparedPlayer) -> PreparedPlayer:
    return PreparedPlayer.VILLAIN if player is PreparedPlayer.HERO else PreparedPlayer.HERO


@dataclass(frozen=True)
class _RoundState:
    stack_hero: float
    stack_villain: float
    committed_hero: float
    committed_villain: float
    street_hero: float
    street_villain: float
    pot: float
    pending_uncalled: float
    last_full_raise_increment: float
    full_raise_count: int
    reopen: bool
    consecutive_checks: int

    def stack(self, player: PreparedPlayer) -> float:
        return self.stack_hero if player is PreparedPlayer.HERO else self.stack_villain

    def street(self, player: PreparedPlayer) -> float:
        return self.street_hero if player is PreparedPlayer.HERO else self.street_villain

    def committed(self, player: PreparedPlayer) -> float:
        return self.committed_hero if player is PreparedPlayer.HERO else self.committed_villain


def _state_replace_player(
    state: _RoundState,
    player: PreparedPlayer,
    *,
    stack: float | None = None,
    street: float | None = None,
    committed: float | None = None,
    **kwargs: Any,
) -> _RoundState:
    values: dict[str, Any] = dict(kwargs)
    suffix = "hero" if player is PreparedPlayer.HERO else "villain"
    if stack is not None:
        values[f"stack_{suffix}"] = stack
    if street is not None:
        values[f"street_{suffix}"] = street
    if committed is not None:
        values[f"committed_{suffix}"] = committed
    return replace(state, **values)


def _check_accounting(state: _RoundState, total_starting: float, tolerance: float) -> None:
    values = tuple(getattr(state, field.name) for field in fields(state) if isinstance(getattr(state, field.name), float))
    if any(not math.isfinite(value) for value in values):
        _fail(PreparedTwoStreetStatus.NUMERIC_FAILURE, "accounting", "non-finite betting state")
    if any(value < -tolerance for value in (state.stack_hero, state.stack_villain, state.committed_hero, state.committed_villain, state.street_hero, state.street_villain, state.pot, state.pending_uncalled)):
        _fail(PreparedTwoStreetStatus.ACCOUNTING_MISMATCH, "accounting", "negative stack, commitment, pot, or pending excess")
    if abs(state.pot - (state.committed_hero + state.committed_villain)) > tolerance:
        _fail(PreparedTwoStreetStatus.ACCOUNTING_MISMATCH, "accounting", "pot differs from total committed chips")
    if abs(
        state.stack_hero + state.stack_villain + state.committed_hero + state.committed_villain - total_starting
    ) > tolerance:
        _fail(PreparedTwoStreetStatus.ACCOUNTING_MISMATCH, "accounting", "in-hand chip conservation failed")
    expected_pending = abs(state.street_hero - state.street_villain)
    if abs(state.pending_uncalled - expected_pending) > tolerance:
        _fail(PreparedTwoStreetStatus.ACCOUNTING_MISMATCH, "accounting", "pending uncalled excess is inconsistent")


def _refund_uncalled(state: _RoundState, tolerance: float) -> _RoundState:
    difference = state.street_hero - state.street_villain
    if abs(difference) <= tolerance:
        return replace(state, pending_uncalled=0.0)
    player = PreparedPlayer.HERO if difference > 0 else PreparedPlayer.VILLAIN
    amount = abs(difference)
    result = _state_replace_player(
        state,
        player,
        stack=state.stack(player) + amount,
        street=state.street(player) - amount,
        committed=state.committed(player) - amount,
        pot=state.pot - amount,
        pending_uncalled=0.0,
    )
    return result


def _apply_action(
    state: _RoundState,
    player: PreparedPlayer,
    option: PreparedActionOption,
    street: PreparedStreet,
    limits: PreparedTwoStreetLimits,
    tolerance: float,
) -> tuple[_RoundState, PreparedRoundCloseReason | None]:
    current = state.street(player)
    opponent = _other(player)
    current_target = max(state.street_hero, state.street_villain)
    to_call = current_target - current
    stack = state.stack(player)

    if option.kind is PreparedActionKind.CHECK:
        if to_call != 0.0:
            _fail(PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR, "action", "check while facing a bet")
        result = replace(state, consecutive_checks=state.consecutive_checks + 1)
        return result, PreparedRoundCloseReason.CHECK_CHECK if result.consecutive_checks == 2 else None

    if option.kind is PreparedActionKind.FOLD:
        if to_call <= 0.0:
            _fail(PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR, "action", "fold without facing a bet")
        return _refund_uncalled(state, tolerance), PreparedRoundCloseReason.FOLD

    if option.kind is PreparedActionKind.CALL:
        if to_call <= 0.0:
            _fail(PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR, "action", "call without facing a bet")
        paid = min(to_call, stack)
        result = _state_replace_player(
            state,
            player,
            stack=stack - paid,
            street=current + paid,
            committed=state.committed(player) + paid,
            pot=state.pot + paid,
            consecutive_checks=0,
        )
        result = replace(result, pending_uncalled=abs(result.street_hero - result.street_villain))
        short = paid < to_call or result.stack(player) == 0.0 or result.stack(opponent) == 0.0
        result = _refund_uncalled(result, tolerance)
        return result, PreparedRoundCloseReason.ALL_IN_CALL if short else PreparedRoundCloseReason.CALL

    if option.kind is PreparedActionKind.BET:
        if to_call != 0.0:
            _fail(PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR, "action", "bet while facing a bet")
        if option.raise_to < street.min_open_bet or option.raise_to <= current:
            _fail(PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR, "action", "open bet is below the exact minimum")
        increment = option.raise_to - current
        if increment > stack:
            _fail(PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR, "action", "bet exceeds stack")
        all_in_exact = increment == stack
        if option.is_all_in is not all_in_exact:
            _fail(PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR, "action", "bet all-in flag does not match exact stack use")
        result = _state_replace_player(
            state,
            player,
            stack=stack - increment,
            street=option.raise_to,
            committed=state.committed(player) + increment,
            pot=state.pot + increment,
            pending_uncalled=option.raise_to - state.street(opponent),
            last_full_raise_increment=option.raise_to - state.street(opponent),
            reopen=True,
            consecutive_checks=0,
        )
        return result, None

    if option.kind is PreparedActionKind.RAISE:
        if to_call <= 0.0:
            _fail(PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR, "action", "raise without facing a bet")
        if not state.reopen:
            _fail(PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR, "action", "raise after action was not reopened")
        if option.raise_to <= current_target:
            _fail(PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR, "action", "raise_to must exceed current target")
        delta = option.raise_to - current
        if delta > stack:
            _fail(PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR, "action", "raise exceeds stack")
        all_in_exact = delta == stack
        if option.is_all_in is not all_in_exact:
            _fail(PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR, "action", "raise all-in flag does not match exact stack use")
        raise_increment = option.raise_to - current_target
        full = raise_increment >= state.last_full_raise_increment
        if full:
            if state.full_raise_count >= limits.max_full_raises_per_street:
                status = (
                    PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR
                    if state.full_raise_count >= _DEFAULT_LIMITS.max_full_raises_per_street
                    else PreparedTwoStreetStatus.CAP_EXCEEDED
                )
                _fail(status, "action", "full raise count exceeds cap")
            new_count = state.full_raise_count + 1
            reopen = True
            last_increment = raise_increment
        else:
            if not option.is_all_in:
                _fail(PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR, "action", "below-min raise is not a short all-in")
            new_count = state.full_raise_count
            reopen = False
            last_increment = state.last_full_raise_increment
        result = _state_replace_player(
            state,
            player,
            stack=stack - delta,
            street=option.raise_to,
            committed=state.committed(player) + delta,
            pot=state.pot + delta,
            pending_uncalled=option.raise_to - state.street(opponent),
            last_full_raise_increment=last_increment,
            full_raise_count=new_count,
            reopen=reopen,
            consecutive_checks=0,
        )
        return result, None

    _fail(PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR, "action", "unsupported action kind")


def _validate_menu_for_state(
    menu: PreparedDecisionMenu,
    state: _RoundState,
    actor: PreparedPlayer,
    limits: PreparedTwoStreetLimits,
) -> None:
    labels = tuple(_action_label(option) for option in menu.actions)
    if len(labels) != len(set(labels)):
        _fail(PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR, "menu", "duplicate action label")
    size_ids = [option.size_id for option in menu.actions if option.size_id is not None]
    if len(size_ids) != len(set(size_ids)):
        _fail(PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR, "menu", "duplicate size_id in menu")
    semantic = [(option.kind, option.size_id, option.raise_to, option.is_all_in) for option in menu.actions]
    if len(semantic) != len(set(semantic)):
        _fail(PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR, "menu", "duplicate action semantics")
    to_call = max(state.street_hero, state.street_villain) - state.street(actor)
    kinds = [option.kind for option in menu.actions]
    if to_call == 0.0:
        if kinds.count(PreparedActionKind.CHECK) != 1 or any(
            kind not in (PreparedActionKind.CHECK, PreparedActionKind.BET) for kind in kinds
        ):
            _fail(PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR, "menu", "no-bet menu must contain one check and only bets")
    else:
        if kinds.count(PreparedActionKind.FOLD) != 1 or kinds.count(PreparedActionKind.CALL) != 1 or any(
            kind not in (PreparedActionKind.FOLD, PreparedActionKind.CALL, PreparedActionKind.RAISE) for kind in kinds
        ):
            _fail(PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR, "menu", "facing-bet menu must contain one fold, one call, and only raises")
        if not state.reopen and PreparedActionKind.RAISE in kinds:
            _fail(PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR, "menu", "short all-in response menu reopens raising")
        if state.full_raise_count >= limits.max_full_raises_per_street and PreparedActionKind.RAISE in kinds:
            status = (
                PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR
                if state.full_raise_count >= _DEFAULT_LIMITS.max_full_raises_per_street
                else PreparedTwoStreetStatus.CAP_EXCEEDED
            )
            _fail(status, "menu", "menu offers a full raise beyond the configured cap")


def _menu_key(history_id: str, street_id: str, player: PreparedPlayer) -> tuple[str, str, PreparedPlayer]:
    return (history_id, street_id, player)


def _transition_key(row: PreparedTransitionRow) -> tuple[str, str, str, str]:
    return (row.transition_id, row.source_public_state_id, row.hero_bucket_id, row.villain_bucket_id)


def _showdown_key(value: PreparedShowdownValue) -> tuple[str, str, str]:
    return (value.public_state_id, value.hero_bucket_id, value.villain_bucket_id)


def _bounded_index(items: Iterable[Any], key: Callable[[Any], tuple], *, status: PreparedTwoStreetStatus, name: str) -> dict[tuple, Any]:
    result: dict[tuple, Any] = {}
    for item in items:
        identity = key(item)
        if identity in result:
            _fail(status, "index", f"duplicate {name} identity")
        result[identity] = item
    return result


@dataclass
class _Preflight:
    spec: PreparedTwoStreetSpec
    limits: PreparedTwoStreetLimits
    menus: dict[tuple[str, str, PreparedPlayer], PreparedDecisionMenu]
    rows: dict[tuple[str, str, str, str], PreparedTransitionRow]
    showdowns: dict[tuple[str, str, str], PreparedShowdownValue]
    used_menus: set[tuple[str, str, PreparedPlayer]]
    used_rows: set[tuple[str, str, str, str]]
    used_showdowns: set[tuple[str, str, str]]
    normalized_rows: dict[tuple[str, str, str, str], tuple[tuple[PreparedChanceEdge, float], ...]]
    normalization_records: dict[tuple[str, str, str, str], PreparedChanceNormalizationRecord]
    observations: dict[str, PreparedInformationSetObservation]
    info_key_semantics: dict[PreparedInformationSetKey, tuple[Any, ...]]
    info_digest_keys: dict[str, PreparedInformationSetKey]
    decision_nodes: int = 0
    terminal_nodes: int = 0
    chance_nodes: int = 1
    chance_edges: int = 0
    max_depth: int = 0

    def bump_decision(self, depth: int) -> None:
        self.decision_nodes += 1
        self._caps(depth)

    def bump_terminal(self, depth: int) -> None:
        self.terminal_nodes += 1
        self._caps(depth)

    def bump_chance(self, edges: int, depth: int) -> None:
        self.chance_nodes += 1
        self.chance_edges += edges
        self._caps(depth)

    def _caps(self, depth: int) -> None:
        self.max_depth = max(self.max_depth, depth)
        total = self.decision_nodes + self.terminal_nodes + self.chance_nodes
        if self.decision_nodes > self.limits.max_decision_nodes:
            _fail(PreparedTwoStreetStatus.CAP_EXCEEDED, "count-preflight", "decision node cap exceeded")
        if self.terminal_nodes > self.limits.max_terminal_nodes:
            _fail(PreparedTwoStreetStatus.CAP_EXCEEDED, "count-preflight", "terminal node cap exceeded")
        if total > self.limits.max_total_nodes:
            _fail(PreparedTwoStreetStatus.CAP_EXCEEDED, "count-preflight", "total node cap exceeded")
        if self.chance_edges > self.limits.max_total_chance_edges:
            _fail(PreparedTwoStreetStatus.CAP_EXCEEDED, "count-preflight", "chance edge cap exceeded")
        if self.max_depth > self.limits.max_depth_edges:
            _fail(PreparedTwoStreetStatus.CAP_EXCEEDED, "count-preflight", "depth cap exceeded")


def _information_digest(key: PreparedInformationSetKey) -> str:
    return hashlib.sha256(_canonical_bytes(key)).hexdigest()


def _public_observation_identity(events: tuple[_PublicEvent, ...], state: _RoundState, actor: PreparedPlayer) -> str:
    public = {
        "events": events,
        "actor": actor,
        "pot": state.pot,
        "committed": (state.committed_hero, state.committed_villain),
        "stack_behind": (state.stack_hero, state.stack_villain),
        "committed_this_street": (state.street_hero, state.street_villain),
        "pending_uncalled": state.pending_uncalled,
        "last_full_raise_increment": state.last_full_raise_increment,
        "full_raise_count": state.full_raise_count,
        "reopen": state.reopen,
    }
    return "obs:sha256:" + _sha256(public)


def _make_information_artifact(
    key: PreparedInformationSetKey,
    events: tuple[_PublicEvent, ...],
    state: _RoundState,
    actor: PreparedPlayer,
    menu: PreparedDecisionMenu,
) -> PreparedInformationSetObservation:
    digest = _information_digest(key)
    return PreparedInformationSetObservation(
        info_set_id="info:sha256:" + digest,
        key=key,
        public_observation_identity=_public_observation_identity(events, state, actor),
        legal_action_labels=tuple(_action_label(option) for option in menu.actions),
    )


def _register_information(
    preflight: _Preflight,
    *,
    actor: PreparedPlayer,
    street: PreparedStreet,
    hero_bucket: str,
    villain_bucket: str,
    hero_history: tuple[str, ...],
    villain_history: tuple[str, ...],
    hero_actions: tuple[str, ...],
    villain_actions: tuple[str, ...],
    events: tuple[_PublicEvent, ...],
    state: _RoundState,
    menu: PreparedDecisionMenu,
) -> str:
    own_bucket = hero_bucket if actor is PreparedPlayer.HERO else villain_bucket
    own_bucket_history = hero_history if actor is PreparedPlayer.HERO else villain_history
    own_actions = hero_actions if actor is PreparedPlayer.HERO else villain_actions
    expected_key = PreparedInformationSetKey(
        contract_version=preflight.spec.contract_version,
        player=actor,
        street_id=street.street_id,
        own_current_bucket_id=own_bucket,
        own_bucket_history=own_bucket_history,
        public_history_id=prepared_public_history_id(events),
        own_action_history=own_actions,
    )
    expected_info_id = "info:sha256:" + _information_digest(expected_key)
    is_new_information_set = expected_info_id not in preflight.observations
    if is_new_information_set:
        hero_count = sum(item.key.player is PreparedPlayer.HERO for item in preflight.observations.values())
        villain_count = len(preflight.observations) - hero_count
        next_hero_count = hero_count + int(actor is PreparedPlayer.HERO)
        next_villain_count = villain_count + int(actor is PreparedPlayer.VILLAIN)
        if (
            next_hero_count > preflight.limits.max_information_sets_per_player
            or next_villain_count > preflight.limits.max_information_sets_per_player
        ):
            _fail(PreparedTwoStreetStatus.CAP_EXCEEDED, "count-preflight", "per-player information-set cap exceeded")
        if len(preflight.observations) + 1 > preflight.limits.max_information_sets_total:
            _fail(PreparedTwoStreetStatus.CAP_EXCEEDED, "count-preflight", "total information-set cap exceeded")
        if (
            len(preflight.observations) + len(preflight.normalization_records) + 1
            > preflight.limits.max_validation_trace_records
        ):
            _fail(PreparedTwoStreetStatus.CAP_EXCEEDED, "count-preflight", "validation trace record cap exceeded")
    artifact = _make_information_artifact(expected_key, events, state, actor, menu)
    if artifact.key != expected_key:
        _fail(PreparedTwoStreetStatus.INFORMATION_MODEL_MISMATCH, "information", "forged canonical information key")
    expected_observation = _public_observation_identity(events, state, actor)
    if artifact.public_observation_identity != expected_observation:
        _fail(PreparedTwoStreetStatus.INFORMATION_MODEL_MISMATCH, "information", "forged public observation identity")
    expected_labels = tuple(_action_label(option) for option in menu.actions)
    if artifact.legal_action_labels != expected_labels:
        _fail(PreparedTwoStreetStatus.INFORMATION_MODEL_MISMATCH, "information", "forged legal action observation")
    if artifact.info_set_id != expected_info_id:
        _fail(PreparedTwoStreetStatus.INFORMATION_MODEL_MISMATCH, "information", "forged information-set digest")
    digest = artifact.info_set_id.removeprefix("info:sha256:")
    collision_key = preflight.info_digest_keys.get(digest)
    if collision_key is not None and collision_key != expected_key:
        _fail(PreparedTwoStreetStatus.INFORMATION_MODEL_MISMATCH, "information", "information-set digest collision")
    preflight.info_digest_keys[digest] = expected_key
    semantics = (
        expected_labels,
        tuple((option.kind, option.size_id, option.raise_to, option.is_all_in) for option in menu.actions),
        expected_observation,
    )
    prior = preflight.info_key_semantics.get(expected_key)
    if prior is not None and prior != semantics:
        _fail(PreparedTwoStreetStatus.INFORMATION_MODEL_MISMATCH, "information", "shared information set has inconsistent actions or observation")
    preflight.info_key_semantics[expected_key] = semantics
    prior_artifact = preflight.observations.get(artifact.info_set_id)
    if prior_artifact is not None and prior_artifact != artifact:
        _fail(PreparedTwoStreetStatus.INFORMATION_MODEL_MISMATCH, "information", "information-set member mismatch")
    preflight.observations[artifact.info_set_id] = artifact
    return artifact.info_set_id


def _normalize_row(preflight: _Preflight, key: tuple[str, str, str, str]) -> tuple[tuple[PreparedChanceEdge, float], ...]:
    cached = preflight.normalized_rows.get(key)
    if cached is not None:
        return cached
    row = preflight.rows.get(key)
    if row is None:
        _fail(PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT, "chance", "missing reachable transition row")
    identities: set[tuple[str, str, str]] = set()
    ordered = sorted(
        row.edges,
        key=lambda edge: (edge.public_outcome_id, edge.next_hero_bucket_id, edge.next_villain_bucket_id),
    )
    for edge in ordered:
        identity = (edge.public_outcome_id, edge.next_hero_bucket_id, edge.next_villain_bucket_id)
        if identity in identities:
            _fail(PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT, "chance", "duplicate chance edge identity")
        identities.add(identity)
    raw_sum = _finite_derived(math.fsum(edge.probability for edge in ordered), "chance raw sum")
    if abs(raw_sum - 1.0) > _PROBABILITY_TOLERANCE:
        _fail(PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT, "chance", "chance row sum is outside tolerance")
    factor = _finite_derived(1.0 / raw_sum, "chance normalization factor")
    effective: list[float] = []
    for edge in ordered:
        probability = _finite_derived(edge.probability * factor, "effective chance probability", positive_reach=True)
        effective.append(probability)
    if abs(math.fsum(effective) - 1.0) > _PROBABILITY_TOLERANCE:
        _fail(PreparedTwoStreetStatus.NUMERIC_FAILURE, "chance", "effective chance probabilities do not sum to one")
    if (
        len(preflight.observations) + len(preflight.normalization_records) + 1
        > preflight.limits.max_validation_trace_records
    ):
        _fail(PreparedTwoStreetStatus.CAP_EXCEEDED, "count-preflight", "validation trace record cap exceeded")
    result = tuple(zip(ordered, effective))
    record = PreparedChanceNormalizationRecord(
        row_identity=key,
        edge_identities=tuple((edge.public_outcome_id, edge.next_hero_bucket_id, edge.next_villain_bucket_id) for edge in ordered),
        raw_sum=raw_sum,
        normalization_factor=factor,
        effective_probabilities=tuple(effective),
    )
    preflight.normalized_rows[key] = result
    preflight.normalization_records[key] = record
    return result


def _terminal_conservation(
    terminal: TerminalNode,
    state: _RoundState,
    spec: PreparedTwoStreetSpec,
    tolerance: float,
    *,
    hero_share: float | None,
    winner: PreparedPlayer | None,
) -> None:
    if abs(terminal.hero_ev + terminal.villain_ev + terminal.house_rake) > tolerance:
        _fail(PreparedTwoStreetStatus.ACCOUNTING_MISMATCH, "terminal", "terminal utility conservation failed")
    if hero_share is not None:
        awarded = state.pot - terminal.house_rake
        hero_received = awarded * hero_share
        villain_received = awarded * (1.0 - hero_share)
        final_hero = state.stack_hero + hero_received
        final_villain = state.stack_villain + villain_received
    else:
        if winner is PreparedPlayer.HERO:
            final_hero = state.stack_hero + state.pot
            final_villain = state.stack_villain
        else:
            final_hero = state.stack_hero
            final_villain = state.stack_villain + state.pot
    total_starting = spec.starting_chips.hero + spec.starting_chips.villain
    if abs(final_hero + final_villain + terminal.house_rake - total_starting) > tolerance:
        _fail(PreparedTwoStreetStatus.ACCOUNTING_MISMATCH, "terminal", "final-stack chip conservation failed")
    if abs((final_hero - spec.starting_chips.hero) - terminal.hero_ev) > tolerance or abs(
        (final_villain - spec.starting_chips.villain) - terminal.villain_ev
    ) > tolerance:
        _fail(PreparedTwoStreetStatus.ACCOUNTING_MISMATCH, "terminal", "terminal utility differs from final-stack delta")


def _preflight_showdown_accounting(
    state: _RoundState,
    spec: PreparedTwoStreetSpec,
    hero_share: float,
    tolerance: float,
) -> None:
    if abs(state.pot - (state.committed_hero + state.committed_villain)) > tolerance:
        _fail(PreparedTwoStreetStatus.ACCOUNTING_MISMATCH, "terminal-preflight", "showdown pot differs from total investment")
    rake = _finite_derived(state.pot * spec.rake.rate, "showdown rake")
    if spec.rake.cap is not None:
        rake = min(rake, spec.rake.cap)
    awarded = _finite_derived(state.pot - rake, "showdown awarded pot")
    hero_received = _finite_derived(awarded * hero_share, "showdown Hero receipt")
    villain_received = _finite_derived(awarded * (1.0 - hero_share), "showdown Villain receipt")
    hero_ev = _finite_derived(hero_received - state.committed_hero, "showdown Hero utility")
    villain_ev = _finite_derived(villain_received - state.committed_villain, "showdown Villain utility")
    if abs(hero_ev + villain_ev + rake) > tolerance:
        _fail(PreparedTwoStreetStatus.ACCOUNTING_MISMATCH, "terminal-preflight", "showdown utility conservation failed")
    final_hero = _finite_derived(state.stack_hero + hero_received, "showdown Hero final stack")
    final_villain = _finite_derived(state.stack_villain + villain_received, "showdown Villain final stack")
    total_starting = spec.starting_chips.hero + spec.starting_chips.villain
    if abs(final_hero + final_villain + rake - total_starting) > tolerance:
        _fail(PreparedTwoStreetStatus.ACCOUNTING_MISMATCH, "terminal-preflight", "showdown final-stack conservation failed")
    if abs((final_hero - spec.starting_chips.hero) - hero_ev) > tolerance or abs(
        (final_villain - spec.starting_chips.villain) - villain_ev
    ) > tolerance:
        _fail(PreparedTwoStreetStatus.ACCOUNTING_MISMATCH, "terminal-preflight", "showdown utility differs from final-stack delta")


def _preflight_fold_accounting(
    state: _RoundState,
    spec: PreparedTwoStreetSpec,
    winner: PreparedPlayer,
    tolerance: float,
) -> None:
    loser = _other(winner)
    loser_committed = state.committed(loser)
    hero_ev = loser_committed if winner is PreparedPlayer.HERO else -loser_committed
    villain_ev = -hero_ev
    if abs(hero_ev + villain_ev) > tolerance:
        _fail(PreparedTwoStreetStatus.ACCOUNTING_MISMATCH, "terminal-preflight", "fold utility conservation failed")
    if winner is PreparedPlayer.HERO:
        final_hero = _finite_derived(state.stack_hero + state.pot, "fold Hero final stack")
        final_villain = state.stack_villain
    else:
        final_hero = state.stack_hero
        final_villain = _finite_derived(state.stack_villain + state.pot, "fold Villain final stack")
    total_starting = spec.starting_chips.hero + spec.starting_chips.villain
    if abs(final_hero + final_villain - total_starting) > tolerance:
        _fail(PreparedTwoStreetStatus.ACCOUNTING_MISMATCH, "terminal-preflight", "fold final-stack conservation failed")
    if abs((final_hero - spec.starting_chips.hero) - hero_ev) > tolerance or abs(
        (final_villain - spec.starting_chips.villain) - villain_ev
    ) > tolerance:
        _fail(PreparedTwoStreetStatus.ACCOUNTING_MISMATCH, "terminal-preflight", "fold utility differs from final-stack delta")


def _initial_state(spec: PreparedTwoStreetSpec) -> _RoundState:
    return _RoundState(
        stack_hero=spec.starting_chips.hero - spec.initial_committed.hero,
        stack_villain=spec.starting_chips.villain - spec.initial_committed.villain,
        committed_hero=spec.initial_committed.hero,
        committed_villain=spec.initial_committed.villain,
        street_hero=0.0,
        street_villain=0.0,
        pot=spec.initial_committed.hero + spec.initial_committed.villain,
        pending_uncalled=0.0,
        last_full_raise_increment=0.0,
        full_raise_count=0,
        reopen=True,
        consecutive_checks=0,
    )


def _reset_for_next_street(state: _RoundState) -> _RoundState:
    return replace(
        state,
        street_hero=0.0,
        street_villain=0.0,
        pending_uncalled=0.0,
        last_full_raise_increment=0.0,
        full_raise_count=0,
        reopen=True,
        consecutive_checks=0,
    )


def _root_matchup_distribution(
    spec: PreparedTwoStreetSpec,
) -> tuple[PreparedRootMatchup, ...]:
    if type(spec) is PreparedJointRootTwoStreetSpec:
        return spec.root_matchups
    return tuple(
        PreparedRootMatchup(
            hero_bucket_id=hero.bucket_id,
            villain_bucket_id=villain.bucket_id,
            probability=_finite_derived(
                hero.weight * villain.weight,
                "factorized root matchup probability",
                positive_reach=True,
            ),
        )
        for hero in spec.hero_buckets
        for villain in spec.villain_buckets
    )


def _preflight_game(spec: PreparedTwoStreetSpec, limits: PreparedTwoStreetLimits) -> tuple[_Preflight, PreparedBuildCounts]:
    menus = _bounded_index(spec.decision_menus, lambda menu: _menu_key(menu.public_history_id, menu.street_id, menu.player), status=PreparedTwoStreetStatus.INVALID_INPUT, name="decision menu")
    rows = _bounded_index(spec.transition_rows, _transition_key, status=PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT, name="transition row")
    showdowns = _bounded_index(spec.showdown_values, _showdown_key, status=PreparedTwoStreetStatus.INVALID_INPUT, name="showdown")
    preflight = _Preflight(
        spec=spec,
        limits=limits,
        menus=menus,
        rows=rows,
        showdowns=showdowns,
        used_menus=set(),
        used_rows=set(),
        used_showdowns=set(),
        normalized_rows={},
        normalization_records={},
        observations={},
        info_key_semantics={},
        info_digest_keys={},
    )
    root_distribution = _root_matchup_distribution(spec)
    root_matchups = len(root_distribution)
    if root_matchups > limits.max_root_matchups:
        _fail(PreparedTwoStreetStatus.CAP_EXCEEDED, "flat-count", "root matchup cap exceeded")
    preflight.chance_edges = root_matchups
    preflight._caps(0)
    weights: list[float] = []
    for matchup in root_distribution:
        probability = _finite_derived(
            matchup.probability,
            "root matchup probability",
            positive_reach=True,
        )
        weights.append(probability)
        _walk_round(
            preflight,
            materialize=False,
            street_index=0,
            actor=spec.streets[0].first_actor,
            state=_initial_state(spec),
            hero_bucket=matchup.hero_bucket_id,
            villain_bucket=matchup.villain_bucket_id,
            hero_history=(matchup.hero_bucket_id,),
            villain_history=(matchup.villain_bucket_id,),
            hero_actions=(),
            villain_actions=(),
            events=(),
            depth=1,
            path_info=frozenset(),
        )
    if abs(math.fsum(weights) - 1.0) > _PROBABILITY_TOLERANCE:
        label = "joint" if type(spec) is PreparedJointRootTwoStreetSpec else "factorized"
        _fail(
            PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT,
            "chance",
            f"{label} root probabilities do not sum to one",
        )
    if preflight.used_menus != set(menus):
        _fail(PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR, "coverage", "missing reachable decision menu or extra unreachable menu")
    if preflight.used_rows != set(rows):
        _fail(PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT, "coverage", "missing reachable transition row or extra unreachable row")
    if preflight.used_showdowns != set(showdowns):
        _fail(PreparedTwoStreetStatus.INVALID_INPUT, "coverage", "missing reachable showdown value or extra unreachable value")
    if len(preflight.normalization_records) + len(preflight.observations) > limits.max_validation_trace_records:
        _fail(PreparedTwoStreetStatus.CAP_EXCEEDED, "count-preflight", "validation trace record cap exceeded")
    hero_infos = {key: semantics for key, semantics in preflight.info_key_semantics.items() if key.player is PreparedPlayer.HERO}
    villain_infos = {key: semantics for key, semantics in preflight.info_key_semantics.items() if key.player is PreparedPlayer.VILLAIN}
    hero_pure = math.prod(len(semantics[0]) for semantics in hero_infos.values())
    villain_pure = math.prod(len(semantics[0]) for semantics in villain_infos.values())
    total_nodes = preflight.decision_nodes + preflight.terminal_nodes + preflight.chance_nodes
    counts = PreparedBuildCounts(
        root_matchups=root_matchups,
        transition_rows=len(preflight.used_rows),
        chance_edges=preflight.chance_edges,
        decision_nodes=preflight.decision_nodes,
        terminal_nodes=preflight.terminal_nodes,
        total_nodes=total_nodes,
        max_depth_edges=preflight.max_depth,
        hero_information_sets=len(hero_infos),
        villain_information_sets=len(villain_infos),
        hero_pure_plans=hero_pure,
        villain_pure_strategies=villain_pure,
    )
    return preflight, counts


def _walk_round(
    preflight: _Preflight,
    *,
    materialize: bool,
    street_index: int,
    actor: PreparedPlayer,
    state: _RoundState,
    hero_bucket: str,
    villain_bucket: str,
    hero_history: tuple[str, ...],
    villain_history: tuple[str, ...],
    hero_actions: tuple[str, ...],
    villain_actions: tuple[str, ...],
    events: tuple[_PublicEvent, ...],
    depth: int,
    path_info: frozenset[tuple[PreparedPlayer, str]],
):
    spec = preflight.spec
    tolerance = max(1e-9, 1e-12 * max(1.0, spec.starting_chips.hero + spec.starting_chips.villain))
    _check_accounting(state, spec.starting_chips.hero + spec.starting_chips.villain, tolerance)
    # An all-in aggressor may have zero stack while the opponent still owes a
    # fold/call response.  Skip decisions only after no unmatched response is
    # pending (for example after an all-in call and refund).
    if (state.stack_hero == 0.0 or state.stack_villain == 0.0) and state.pending_uncalled == 0.0:
        if street_index < len(spec.streets) - 1:
            close_events = events + (
                PreparedStreetCloseEvent(
                    spec.streets[street_index].street_id,
                    PreparedRoundCloseReason.ALL_IN_CALL,
                ),
            )
            return _after_round_close(
                preflight,
                materialize=materialize,
                street_index=street_index,
                state=state,
                hero_bucket=hero_bucket,
                villain_bucket=villain_bucket,
                hero_history=hero_history,
                villain_history=villain_history,
                hero_actions=hero_actions,
                villain_actions=villain_actions,
                events=close_events,
                depth=depth,
                path_info=path_info,
            )
        return _finish_showdown(
            preflight,
            materialize=materialize,
            state=state,
            hero_bucket=hero_bucket,
            villain_bucket=villain_bucket,
            hero_history=hero_history,
            villain_history=villain_history,
            events=events,
            depth=depth,
        )
    street = spec.streets[street_index]
    history_id = prepared_public_history_id(events)
    key = _menu_key(history_id, street.street_id, actor)
    menu = preflight.menus.get(key)
    if menu is None:
        _fail(PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR, "coverage", "reachable decision has no exact menu")
    preflight.used_menus.add(key)
    _validate_menu_for_state(menu, state, actor, preflight.limits)
    info_set_id = _register_information(
        preflight,
        actor=actor,
        street=street,
        hero_bucket=hero_bucket,
        villain_bucket=villain_bucket,
        hero_history=hero_history,
        villain_history=villain_history,
        hero_actions=hero_actions,
        villain_actions=villain_actions,
        events=events,
        state=state,
        menu=menu,
    )
    marker = (actor, info_set_id)
    if marker in path_info:
        _fail(PreparedTwoStreetStatus.INFORMATION_MODEL_MISMATCH, "information", "same information set repeats on one path")
    if not materialize:
        preflight.bump_decision(depth)
    actions: list[tuple[str, Any]] = []
    for option in menu.actions:
        label = _action_label(option)
        next_state, close_reason = _apply_action(state, actor, option, street, preflight.limits, tolerance)
        _check_accounting(next_state, spec.starting_chips.hero + spec.starting_chips.villain, tolerance)
        action_event = PreparedActionEvent(
            street_id=street.street_id,
            player=actor,
            kind=option.kind,
            size_id=option.size_id,
            raise_to=option.raise_to,
            is_all_in=option.is_all_in,
            reopen=next_state.reopen,
        )
        next_events = events + (action_event,)
        next_hero_actions = hero_actions + ((label,) if actor is PreparedPlayer.HERO else ())
        next_villain_actions = villain_actions + ((label,) if actor is PreparedPlayer.VILLAIN else ())
        if close_reason is PreparedRoundCloseReason.FOLD:
            child = _finish_fold(
                preflight,
                materialize=materialize,
                state=next_state,
                winner=_other(actor),
                hero_history=hero_history,
                villain_history=villain_history,
                events=next_events,
                depth=depth + 1,
            )
        elif close_reason is not None:
            close_events = next_events + (PreparedStreetCloseEvent(street.street_id, close_reason),)
            child = _after_round_close(
                preflight,
                materialize=materialize,
                street_index=street_index,
                state=next_state,
                hero_bucket=hero_bucket,
                villain_bucket=villain_bucket,
                hero_history=hero_history,
                villain_history=villain_history,
                hero_actions=next_hero_actions,
                villain_actions=next_villain_actions,
                events=close_events,
                depth=depth + 1,
                path_info=path_info | {marker},
            )
        else:
            child = _walk_round(
                preflight,
                materialize=materialize,
                street_index=street_index,
                actor=_other(actor),
                state=next_state,
                hero_bucket=hero_bucket,
                villain_bucket=villain_bucket,
                hero_history=hero_history,
                villain_history=villain_history,
                hero_actions=next_hero_actions,
                villain_actions=next_villain_actions,
                events=next_events,
                depth=depth + 1,
                path_info=path_info | {marker},
            )
        if materialize:
            actions.append((label, child))
    if not materialize:
        return None
    node_id = _node_id("decision", events, hero_history, villain_history, actor.value)
    node_cls = HeroNode if actor is PreparedPlayer.HERO else VillainNode
    return node_cls(node_id=node_id, info_set=info_set_id, actions=tuple(actions))


def _after_round_close(
    preflight: _Preflight,
    *,
    materialize: bool,
    street_index: int,
    state: _RoundState,
    hero_bucket: str,
    villain_bucket: str,
    hero_history: tuple[str, ...],
    villain_history: tuple[str, ...],
    hero_actions: tuple[str, ...],
    villain_actions: tuple[str, ...],
    events: tuple[_PublicEvent, ...],
    depth: int,
    path_info: frozenset[tuple[PreparedPlayer, str]],
):
    if state.pending_uncalled != 0.0:
        _fail(PreparedTwoStreetStatus.ACCOUNTING_MISMATCH, "accounting", "round closed before uncalled excess refund")
    if street_index == len(preflight.spec.streets) - 1:
        return _finish_showdown(
            preflight,
            materialize=materialize,
            state=state,
            hero_bucket=hero_bucket,
            villain_bucket=villain_bucket,
            hero_history=hero_history,
            villain_history=villain_history,
            events=events,
            depth=depth,
        )
    source_id = prepared_public_history_id(events)
    row_key = (preflight.spec.transition_id, source_id, hero_bucket, villain_bucket)
    normalized = _normalize_row(preflight, row_key)
    preflight.used_rows.add(row_key)
    if not materialize:
        preflight.bump_chance(len(normalized), depth)
    children: list[tuple[float, Any]] = []
    for edge, probability in normalized:
        next_events = events + (PreparedChanceEvent(preflight.spec.transition_id, edge.public_outcome_id),)
        next_state = _reset_for_next_street(state)
        if next_state.stack_hero == 0.0 or next_state.stack_villain == 0.0:
            child = _finish_showdown(
                preflight,
                materialize=materialize,
                state=next_state,
                hero_bucket=edge.next_hero_bucket_id,
                villain_bucket=edge.next_villain_bucket_id,
                hero_history=hero_history + (edge.next_hero_bucket_id,),
                villain_history=villain_history + (edge.next_villain_bucket_id,),
                events=next_events,
                depth=depth + 1,
            )
        else:
            child = _walk_round(
                preflight,
                materialize=materialize,
                street_index=street_index + 1,
                actor=preflight.spec.streets[street_index + 1].first_actor,
                state=next_state,
                hero_bucket=edge.next_hero_bucket_id,
                villain_bucket=edge.next_villain_bucket_id,
                hero_history=hero_history + (edge.next_hero_bucket_id,),
                villain_history=villain_history + (edge.next_villain_bucket_id,),
                hero_actions=hero_actions,
                villain_actions=villain_actions,
                events=next_events,
                depth=depth + 1,
                path_info=path_info,
            )
        if materialize:
            children.append((probability, child))
    if not materialize:
        return None
    return ChanceNode(
        node_id=_node_id("transition", events, hero_history, villain_history, preflight.spec.transition_id),
        children=tuple(children),
    )


def _finish_showdown(
    preflight: _Preflight,
    *,
    materialize: bool,
    state: _RoundState,
    hero_bucket: str,
    villain_bucket: str,
    hero_history: tuple[str, ...],
    villain_history: tuple[str, ...],
    events: tuple[_PublicEvent, ...],
    depth: int,
):
    if state.pending_uncalled != 0.0:
        _fail(PreparedTwoStreetStatus.ACCOUNTING_MISMATCH, "accounting", "showdown reached before uncalled excess refund")
    state_id = prepared_public_history_id(events)
    key = (state_id, hero_bucket, villain_bucket)
    value = preflight.showdowns.get(key)
    if value is None:
        _fail(PreparedTwoStreetStatus.INVALID_INPUT, "coverage", "reachable showdown has no exact prepared value")
    preflight.used_showdowns.add(key)
    if not materialize:
        preflight.bump_terminal(depth)
        tolerance = max(
            1e-9,
            1e-12 * max(1.0, preflight.spec.starting_chips.hero + preflight.spec.starting_chips.villain),
        )
        _preflight_showdown_accounting(state, preflight.spec, value.hero_pot_share, tolerance)
        return None
    tolerance = max(1e-9, 1e-12 * max(1.0, preflight.spec.starting_chips.hero + preflight.spec.starting_chips.villain))
    terminal = make_equity_showdown_terminal(
        _node_id("showdown", events, hero_history, villain_history, "terminal"),
        state.pot,
        state.committed_hero,
        state.committed_villain,
        value.hero_pot_share,
        preflight.spec.rake.rate,
        preflight.spec.rake.cap,
        tolerance=tolerance,
    )
    _terminal_conservation(terminal, state, preflight.spec, tolerance, hero_share=value.hero_pot_share, winner=None)
    return terminal


def _finish_fold(
    preflight: _Preflight,
    *,
    materialize: bool,
    state: _RoundState,
    winner: PreparedPlayer,
    hero_history: tuple[str, ...],
    villain_history: tuple[str, ...],
    events: tuple[_PublicEvent, ...],
    depth: int,
):
    if not materialize:
        preflight.bump_terminal(depth)
        tolerance = max(
            1e-9,
            1e-12 * max(1.0, preflight.spec.starting_chips.hero + preflight.spec.starting_chips.villain),
        )
        _preflight_fold_accounting(state, preflight.spec, winner, tolerance)
        return None
    loser = _other(winner)
    terminal = make_fold_terminal(
        _node_id("fold", events, hero_history, villain_history, winner.value),
        HERO if winner is PreparedPlayer.HERO else VILLAIN,
        state.committed(loser),
    )
    tolerance = max(1e-9, 1e-12 * max(1.0, preflight.spec.starting_chips.hero + preflight.spec.starting_chips.villain))
    _terminal_conservation(terminal, state, preflight.spec, tolerance, hero_share=None, winner=winner)
    return terminal


def _node_id(kind: str, events: tuple[_PublicEvent, ...], hero_history: tuple[str, ...], villain_history: tuple[str, ...], discriminator: str) -> str:
    return "node:sha256:" + _sha256(
        {
            "kind": kind,
            "public_history_id": prepared_public_history_id(events),
            "hero_bucket_history": hero_history,
            "villain_bucket_history": villain_history,
            "discriminator": discriminator,
        }
    )


def _materialize(preflight: _Preflight) -> GameTree:
    spec = preflight.spec
    children: list[tuple[float, Any]] = []
    for matchup in _root_matchup_distribution(spec):
        probability = _finite_derived(
            matchup.probability,
            "root matchup probability",
            positive_reach=True,
        )
        child = _walk_round(
            preflight,
            materialize=True,
            street_index=0,
            actor=spec.streets[0].first_actor,
            state=_initial_state(spec),
            hero_bucket=matchup.hero_bucket_id,
            villain_bucket=matchup.villain_bucket_id,
            hero_history=(matchup.hero_bucket_id,),
            villain_history=(matchup.villain_bucket_id,),
            hero_actions=(),
            villain_actions=(),
            events=(),
            depth=1,
            path_info=frozenset(),
        )
        children.append((probability, child))
    return GameTree(root=ChanceNode(node_id="node:prepared-two-street-root", children=tuple(children)))


def _ordered_tree_sha256(tree: GameTree) -> str:
    seen_ids: dict[str, Any] = {}
    seen_objects: set[int] = set()

    def walk(node: Any) -> Any:
        if id(node) in seen_objects:
            _fail(PreparedTwoStreetStatus.INFORMATION_MODEL_MISMATCH, "tree-identity", "shared-child DAG or cycle detected")
        seen_objects.add(id(node))
        prior = seen_ids.get(node.node_id)
        if prior is not None and prior is not node:
            _fail(PreparedTwoStreetStatus.INFORMATION_MODEL_MISMATCH, "tree-identity", "duplicate node ID")
        seen_ids[node.node_id] = node
        if isinstance(node, TerminalNode):
            return ("terminal", node.node_id, node.hero_ev.hex(), node.villain_ev.hex(), node.house_rake.hex())
        if isinstance(node, ChanceNode):
            return ("chance", node.node_id, tuple((prob.hex(), walk(child)) for prob, child in node.children))
        if isinstance(node, HeroNode):
            return ("hero", node.node_id, node.info_set, tuple((label, walk(child)) for label, child in node.actions))
        if isinstance(node, VillainNode):
            return ("villain", node.node_id, node.info_set, tuple((label, walk(child)) for label, child in node.actions))
        _fail(PreparedTwoStreetStatus.INFORMATION_MODEL_MISMATCH, "tree-identity", "unknown generic node type")

    return _sha256({"algorithm": _ORDERED_TREE_ID, "tree": walk(tree.root)})


def _actual_counts(tree: GameTree) -> tuple[int, int, int, int, int]:
    decision = terminal = chance_edges = 0
    max_depth = 0

    def walk(node: Any, depth: int) -> None:
        nonlocal decision, terminal, chance_edges, max_depth
        max_depth = max(max_depth, depth)
        if isinstance(node, TerminalNode):
            terminal += 1
        elif isinstance(node, ChanceNode):
            chance_edges += len(node.children)
            for _, child in node.children:
                walk(child, depth + 1)
        elif isinstance(node, (HeroNode, VillainNode)):
            decision += 1
            for _, child in node.actions:
                walk(child, depth + 1)

    walk(tree.root, 0)
    total = sum(1 for _ in iter_nodes(tree.root))
    return decision, terminal, total, chance_edges, max_depth


def _success_result(
    spec: PreparedTwoStreetSpec,
    content_identity: PreparedContentIdentity,
    preflight: _Preflight,
    counts: PreparedBuildCounts,
) -> PreparedTwoStreetBuildResult:
    tree = _materialize(preflight)
    tolerance = max(1e-9, 1e-12 * max(1.0, spec.starting_chips.hero + spec.starting_chips.villain))
    validate_tree(tree, tolerance=tolerance)
    actual = _actual_counts(tree)
    expected = (counts.decision_nodes, counts.terminal_nodes, counts.total_nodes, counts.chance_edges, counts.max_depth_edges)
    if actual != expected:
        _fail(PreparedTwoStreetStatus.NON_REPRODUCIBLE, "materialization", "materialized tree counts differ from preflight")
    if len(collect_hero_info_sets(tree)) != counts.hero_information_sets or len(collect_villain_info_sets(tree)) != counts.villain_information_sets:
        _fail(PreparedTwoStreetStatus.NON_REPRODUCIBLE, "materialization", "materialized information-set counts differ from preflight")
    ordered_hash = _ordered_tree_sha256(tree)
    joint_root = type(spec) is PreparedJointRootTwoStreetSpec
    identity_base = {
        "contract_version": spec.contract_version,
        "builder_id": (
            PREPARED_JOINT_ROOT_BUILDER_ID
            if joint_root
            else PREPARED_TWO_STREET_BUILDER_ID
        ),
        "action_label_id": _ACTION_LABEL_ID,
        "normalization_id": PREPARED_CHANCE_NORMALIZATION_ID,
        "information_key_id": PREPARED_INFORMATION_KEY_ID,
        "raw_sha256": content_identity.raw_sha256,
        "semantic_sha256": content_identity.semantic_sha256,
        "ordered_tree_sha256": ordered_hash,
    }
    identity = PreparedTwoStreetIdentity(
        **identity_base,
        run_identity=hashlib.sha256(_canonical_bytes(identity_base)).hexdigest(),
    )
    build = PreparedTwoStreetBuild(
        tree=tree,
        identity=identity,
        counts=counts,
        chance_normalization=tuple(preflight.normalization_records[key] for key in sorted(preflight.normalization_records)),
        information_sets=tuple(preflight.observations[key] for key in sorted(preflight.observations)),
    )
    return PreparedTwoStreetBuildResult(status=PreparedTwoStreetStatus.SUCCESS, build=build, error=None)


def build_prepared_two_street_game(
    spec: PreparedTwoStreetSpec,
    raw_input_bytes: bytes,
    content_identity: PreparedContentIdentity,
    limits: PreparedTwoStreetLimits = PreparedTwoStreetLimits(),
) -> PreparedTwoStreetBuildResult:
    """Build a bounded prepared abstract game, or return one fail-closed error.

    A success contains only a validated generic tree plus identities, counts,
    normalization evidence, and canonical information observations.  A
    :class:`PreparedJointRootTwoStreetSpec` preserves caller-supplied correlated
    root reach after validating its exact support and marginals; it is not
    silently factorized.  The result does not contain a strategy, response,
    candidate, equilibrium, or real-card certification.  Every failure has
    ``build=None`` and a non-empty bounded error; no partial tree or payoff is
    returned.
    """

    try:
        valid_spec, valid_limits = _validate_outer(spec, raw_input_bytes, content_identity, limits)
        preflight, counts = _preflight_game(valid_spec, valid_limits)
        return _success_result(valid_spec, content_identity, preflight, counts)
    except _BuildFailure as exc:
        return PreparedTwoStreetBuildResult(
            status=exc.status,
            build=None,
            error=PreparedBuildError(message=exc.message or "prepared build failed", phase=exc.phase or "build"),
        )
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        message = str(exc).strip() or type(exc).__name__
        return PreparedTwoStreetBuildResult(
            status=PreparedTwoStreetStatus.INVALID_INPUT,
            build=None,
            error=PreparedBuildError(message=message[:500], phase="validation"),
        )
    except Exception:
        return PreparedTwoStreetBuildResult(
            status=PreparedTwoStreetStatus.INVALID_INPUT,
            build=None,
            error=PreparedBuildError(message="unexpected prepared build failure", phase="build"),
        )


def _oracle_comparison_result(
    dp_value: tuple[float, ...],
    enumerate_value: tuple[float, ...],
    tolerance: float = 1e-9,
) -> PreparedTwoStreetBuildResult | None:
    """Private negative-test seam for independent DP/enumerate comparison."""

    if len(dp_value) != len(enumerate_value) or any(
        not math.isfinite(left)
        or not math.isfinite(right)
        or abs(left - right) > tolerance
        for left, right in zip(dp_value, enumerate_value)
    ):
        return PreparedTwoStreetBuildResult(
            status=PreparedTwoStreetStatus.ORACLE_MISMATCH,
            build=None,
            error=PreparedBuildError(
                message="independent DP and enumeration oracle values disagree",
                phase="oracle",
            ),
        )
    return None
