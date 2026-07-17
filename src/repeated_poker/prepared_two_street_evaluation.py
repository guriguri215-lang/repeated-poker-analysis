"""Bounded evaluation adapter for a successful prepared two-street build.

The adapter validates a complete fixed Hero profile and an optional complete
fixed Villain profile, returns Villain's exact response to fixed Hero on every
successful call, and adds a fixed-profile value only when Villain is supplied.
It does not compute an equilibrium, Nash or optimal Hero strategy, certify a
solver or real-card model, assert profitability, or make real-money advice.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from typing import Any, Mapping

from .exact_response import (
    BestResponseResult,
    count_villain_pure_strategies,
    solve_exact_response,
)
from .fixed_profile import evaluate_fixed_profile
from .game import (
    ChanceNode,
    GameTree,
    HeroNode,
    HeroStrategy,
    TerminalNode,
    VillainNode,
    VillainStrategy,
    collect_hero_info_sets,
    collect_villain_info_sets,
    validate_tree,
)
from .prepared_two_street import (
    PREPARED_CHANCE_NORMALIZATION_ID,
    PREPARED_INFORMATION_KEY_ID,
    PREPARED_TWO_STREET_BUILDER_ID,
    PREPARED_TWO_STREET_CONTRACT_VERSION,
    PreparedBuildCounts,
    PreparedChanceNormalizationRecord,
    PreparedInformationSetKey,
    PreparedInformationSetObservation,
    PreparedPlayer,
    PreparedTwoStreetBuild,
    PreparedTwoStreetIdentity,
)


PREPARED_TWO_STREET_EVALUATION_ADAPTER_ID = (
    "prepared-two-street-evaluation-adapter-v1"
)
PREPARED_PROFILE_NORMALIZATION_ID = "prepared-profile-positive-fsum-normalize-v1"
PREPARED_PROFILE_RAW_ID = "prepared-profile-raw-sha256-v1"
PREPARED_PROFILE_EFFECTIVE_ID = "prepared-profile-effective-sha256-v1"
PREPARED_EVALUATION_OUTPUT_ID = "prepared-two-street-evaluation-output-sha256-v1"
PREPARED_PROFILE_ABSENT = "ABSENT"
PREPARED_PROFILE_NORMALIZATION_TOLERANCE = 1e-9
PREPARED_EVALUATION_TOLERANCE = 1e-9

_M14_ACTION_LABEL_ID = "betting-tree-v2-action-label-v1"
_M14_ORDERED_TREE_ID = "betting-tree-v2-ordered-tree-sha256-v1"
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_INFO_RE = re.compile(r"info:sha256:[0-9a-f]{64}\Z")
_OBS_RE = re.compile(r"obs:sha256:[0-9a-f]{64}\Z")
_ACTION_RE = re.compile(r"(?:check|fold|call|(?:bet|raise)::.+)\Z")


__all__ = [
    "PREPARED_TWO_STREET_EVALUATION_ADAPTER_ID",
    "PREPARED_PROFILE_NORMALIZATION_ID",
    "PREPARED_PROFILE_RAW_ID",
    "PREPARED_PROFILE_EFFECTIVE_ID",
    "PREPARED_EVALUATION_OUTPUT_ID",
    "PREPARED_PROFILE_ABSENT",
    "PREPARED_PROFILE_NORMALIZATION_TOLERANCE",
    "PREPARED_EVALUATION_TOLERANCE",
    "PreparedEvaluationStatus",
    "PreparedCorrespondenceMode",
    "PreparedCorrespondenceStatus",
    "PreparedEvaluationLimits",
    "PreparedEvaluationRequest",
    "PreparedActionProbability",
    "PreparedProfileEntry",
    "PreparedPlayerProfile",
    "PreparedProfileNormalizationRecord",
    "PreparedProfileNormalization",
    "PreparedFixedProfileValue",
    "PreparedResponseAssignment",
    "PreparedPureResponse",
    "PreparedResponseActionSet",
    "PreparedResponseVariation",
    "PreparedExactResponseValue",
    "PreparedEvaluationTraceRecord",
    "PreparedEvaluation",
    "PreparedEvaluationIdentity",
    "PreparedEvaluationError",
    "PreparedEvaluationResult",
    "evaluate_prepared_two_street",
]


class PreparedEvaluationStatus(str, Enum):
    SUCCESS = "SUCCESS"
    INVALID_INPUT = "INVALID_INPUT"
    INCOMPLETE_PROFILE = "INCOMPLETE_PROFILE"
    ILLEGAL_PROFILE_REFERENCE = "ILLEGAL_PROFILE_REFERENCE"
    INVALID_PROFILE_PROBABILITY = "INVALID_PROFILE_PROBABILITY"
    BUILD_MISMATCH = "BUILD_MISMATCH"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    CAP_EXCEEDED = "CAP_EXCEEDED"
    NUMERIC_FAILURE = "NUMERIC_FAILURE"
    FIXED_PROFILE_FAILURE = "FIXED_PROFILE_FAILURE"
    EXACT_RESPONSE_FAILURE = "EXACT_RESPONSE_FAILURE"
    NON_REPRODUCIBLE = "NON_REPRODUCIBLE"
    ORACLE_MISMATCH = "ORACLE_MISMATCH"
    UNSUPPORTED_DOWNSTREAM = "UNSUPPORTED_DOWNSTREAM"


class PreparedCorrespondenceMode(str, Enum):
    NOT_REQUESTED = "NOT_REQUESTED"
    FULL = "FULL"


class PreparedCorrespondenceStatus(str, Enum):
    NOT_REQUESTED = "NOT_REQUESTED"
    MATERIALIZED = "MATERIALIZED"


@dataclass(frozen=True)
class PreparedEvaluationLimits:
    max_profile_information_sets_per_player: int = 1_000
    max_actions_per_profile_entry: int = 8
    max_total_profile_probabilities: int = 16_000
    max_validation_trace_records: int = 10_000
    max_full_correspondence_strategies: int = 100_000
    max_enumerator_pure_strategies: int = 100_000
    max_result_records: int = 100_000


@dataclass(frozen=True)
class PreparedEvaluationRequest:
    method: str = "dp"
    correspondence_mode: PreparedCorrespondenceMode = PreparedCorrespondenceMode.NOT_REQUESTED
    oracle_check: bool = False
    downstream_request: str | None = None
    expected_output_semantic_sha256: str | None = None


@dataclass(frozen=True)
class PreparedActionProbability:
    action_label: str
    probability: float


@dataclass(frozen=True)
class PreparedProfileEntry:
    info_set_id: str
    action_probabilities: tuple[PreparedActionProbability, ...]


@dataclass(frozen=True)
class PreparedPlayerProfile:
    entries: tuple[PreparedProfileEntry, ...]


@dataclass(frozen=True)
class PreparedProfileNormalizationRecord:
    info_set_id: str
    legal_action_labels: tuple[str, ...]
    raw_probabilities: tuple[float, ...]
    raw_sum: float
    normalization_factor: float
    effective_probabilities: tuple[float, ...]


@dataclass(frozen=True)
class PreparedProfileNormalization:
    raw_profile_sha256: str
    effective_profile_sha256: str
    records: tuple[PreparedProfileNormalizationRecord, ...]


@dataclass(frozen=True)
class PreparedFixedProfileValue:
    hero_ev: float
    villain_ev: float
    house_rake: float
    conservation_residual: float


@dataclass(frozen=True)
class PreparedResponseAssignment:
    info_set_id: str
    action_label: str


@dataclass(frozen=True)
class PreparedPureResponse:
    assignments: tuple[PreparedResponseAssignment, ...]


@dataclass(frozen=True)
class PreparedResponseActionSet:
    info_set_id: str
    action_labels: tuple[str, ...]


@dataclass(frozen=True)
class PreparedResponseVariation:
    info_set_id: str
    action_labels: tuple[str, ...]


@dataclass(frozen=True)
class PreparedExactResponseValue:
    villain_max_ev: float
    hero_ev_worst: float
    hero_ev_best: float
    house_rake_worst: float
    house_rake_best: float
    num_villain_pure_strategies: int
    num_best_response_strategies: int
    representative_pure_response: PreparedPureResponse
    best_response_action_sets: tuple[PreparedResponseActionSet, ...]
    best_response_action_variation: tuple[PreparedResponseVariation, ...]
    off_path_info_sets: tuple[str, ...]
    full_correspondence_status: PreparedCorrespondenceStatus
    full_correspondence: tuple[PreparedPureResponse, ...] | None


@dataclass(frozen=True)
class PreparedEvaluationTraceRecord:
    phase: str
    subject: str
    outcome: str


@dataclass(frozen=True)
class PreparedEvaluation:
    hero_profile_normalization: PreparedProfileNormalization
    villain_profile_normalization: PreparedProfileNormalization | None
    fixed_profile_value: PreparedFixedProfileValue | None
    exact_response: PreparedExactResponseValue
    validation_trace: tuple[PreparedEvaluationTraceRecord, ...]


@dataclass(frozen=True)
class PreparedEvaluationIdentity:
    m14_contract_version: str
    m14_builder_id: str
    m14_action_label_id: str
    m14_normalization_id: str
    m14_information_key_id: str
    m14_raw_sha256: str
    m14_prepared_semantic_sha256: str
    m14_ordered_tree_sha256: str
    m14_run_identity: str
    hero_raw_profile_sha256: str
    hero_effective_profile_sha256: str
    villain_raw_profile_sha256: str
    villain_effective_profile_sha256: str
    evaluation_adapter_id: str
    profile_normalization_id: str
    profile_raw_id: str
    profile_effective_id: str
    output_semantic_id: str
    exact_response_method: str
    profile_tolerance_hex: str
    evaluation_tolerance_hex: str
    effective_limits: PreparedEvaluationLimits
    correspondence_mode: PreparedCorrespondenceMode
    oracle_check: bool
    output_semantic_sha256: str


@dataclass(frozen=True)
class PreparedEvaluationError:
    message: str
    phase: str


@dataclass(frozen=True)
class PreparedEvaluationResult:
    status: PreparedEvaluationStatus
    evaluation: PreparedEvaluation | None
    identity: PreparedEvaluationIdentity | None
    error: PreparedEvaluationError | None


_DEFAULT_LIMITS = PreparedEvaluationLimits()
_M14_COUNT_LIMITS = {
    "root_matchups": 10_000,
    "transition_rows": 50_000,
    "chance_edges": 100_000,
    "decision_nodes": 100_000,
    "terminal_nodes": 100_000,
    "total_nodes": 250_000,
    "max_depth_edges": 32,
    "hero_information_sets": 1_000,
    "villain_information_sets": 1_000,
}


class _Failure(Exception):
    def __init__(self, status: PreparedEvaluationStatus, phase: str, message: str):
        self.status = status
        self.phase = phase
        self.message = message


def _fail(status: PreparedEvaluationStatus, phase: str, message: str) -> None:
    raise _Failure(status, phase, message)


def _failure(status: PreparedEvaluationStatus, phase: str, message: str) -> PreparedEvaluationResult:
    clean_phase = str(phase)[:64] or "evaluation"
    clean_message = str(message).replace("\r", " ").replace("\n", " ")[:500]
    if not clean_message:
        clean_message = "evaluation failed"
    return PreparedEvaluationResult(status, None, None, PreparedEvaluationError(clean_message, clean_phase))


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical numeric values must be finite")
        return {"__floathex__": value.hex()}
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": type(value).__name__,
            "fields": {field.name: _canonical_value(getattr(value, field.name)) for field in fields(value)},
        }
    if isinstance(value, tuple) or isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("canonical mapping keys must be strings")
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    raise ValueError(f"unsupported canonical value type {type(value).__name__}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(_canonical_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA_RE.fullmatch(value) is not None


def _checked_add(a: int, b: int, cap: int, phase: str, label: str) -> int:
    value = a + b
    if value > cap:
        _fail(PreparedEvaluationStatus.CAP_EXCEEDED, phase, f"{label} cap exceeded")
    return value


def _checked_mul(a: int, b: int, cap: int, phase: str, label: str) -> int:
    if a and b > cap // a:
        _fail(PreparedEvaluationStatus.CAP_EXCEEDED, phase, f"{label} cap exceeded")
    return a * b


def _validate_outer(build: Any, hero: Any, villain: Any, limits: Any, request: Any) -> tuple[PreparedEvaluationLimits, PreparedEvaluationRequest]:
    if type(build) is not PreparedTwoStreetBuild:
        _fail(PreparedEvaluationStatus.INVALID_INPUT, "input", "build must be PreparedTwoStreetBuild")
    if type(hero) is not PreparedPlayerProfile or (villain is not None and type(villain) is not PreparedPlayerProfile):
        _fail(PreparedEvaluationStatus.INVALID_INPUT, "input", "profiles must use PreparedPlayerProfile")
    if type(limits) is not PreparedEvaluationLimits or type(request) is not PreparedEvaluationRequest:
        _fail(PreparedEvaluationStatus.INVALID_INPUT, "input", "limits and request must use exact public dataclasses")
    for field in fields(limits):
        value = getattr(limits, field.name)
        ceiling = getattr(_DEFAULT_LIMITS, field.name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > ceiling:
            _fail(PreparedEvaluationStatus.INVALID_INPUT, "input", f"invalid {field.name}")
    if not isinstance(request.method, str):
        _fail(PreparedEvaluationStatus.INVALID_INPUT, "request", "method must be a string")
    if type(request.correspondence_mode) is not PreparedCorrespondenceMode or type(request.oracle_check) is not bool:
        _fail(PreparedEvaluationStatus.INVALID_INPUT, "request", "invalid correspondence or oracle request")
    if request.downstream_request is not None and not isinstance(request.downstream_request, str):
        _fail(PreparedEvaluationStatus.INVALID_INPUT, "request", "downstream_request must be str or None")
    if request.expected_output_semantic_sha256 is not None and not _valid_sha(request.expected_output_semantic_sha256):
        _fail(PreparedEvaluationStatus.INVALID_INPUT, "request", "expected output hash must be lowercase SHA-256")
    if request.downstream_request is not None:
        _fail(PreparedEvaluationStatus.UNSUPPORTED_DOWNSTREAM, "request", "downstream requests are unsupported")
    if request.method not in ("dp", "enumerate") or (request.method == "enumerate" and not request.oracle_check):
        _fail(PreparedEvaluationStatus.UNSUPPORTED_DOWNSTREAM, "request", "unsupported exact-response method request")
    return limits, request


def _outer_caps(build: PreparedTwoStreetBuild, hero: PreparedPlayerProfile, villain: PreparedPlayerProfile | None, limits: PreparedEvaluationLimits, request: PreparedEvaluationRequest) -> None:
    if type(build.counts) is PreparedBuildCounts:
        if build.counts.hero_information_sets > limits.max_profile_information_sets_per_player or build.counts.villain_information_sets > limits.max_profile_information_sets_per_player:
            _fail(PreparedEvaluationStatus.CAP_EXCEEDED, "profile-cap", "build information-set count exceeds effective profile cap")
    profiles = (("Hero", hero),) + (("Villain", villain),) if villain is not None else (("Hero", hero),)
    total = 0
    for player, profile in profiles:
        if not isinstance(profile.entries, tuple):
            _fail(PreparedEvaluationStatus.INVALID_INPUT, "profile", f"{player} entries must be a tuple")
        if len(profile.entries) > limits.max_profile_information_sets_per_player:
            _fail(PreparedEvaluationStatus.CAP_EXCEEDED, "profile-cap", f"{player} information-set cap exceeded")
        for entry in profile.entries:
            if type(entry) is not PreparedProfileEntry or not isinstance(entry.action_probabilities, tuple):
                _fail(PreparedEvaluationStatus.INVALID_INPUT, "profile", "profile entries and actions must use exact tuple records")
            if len(entry.action_probabilities) > limits.max_actions_per_profile_entry:
                _fail(PreparedEvaluationStatus.CAP_EXCEEDED, "profile-cap", "actions-per-entry cap exceeded")
            total = _checked_add(total, len(entry.action_probabilities), limits.max_total_profile_probabilities, "profile-cap", "total profile probability")
    trace_count = 7 + len(hero.entries) + (len(villain.entries) if villain is not None else 0) + 1 + (1 if villain is not None else 0) + (1 if request.oracle_check else 0) + (1 if request.correspondence_mode is PreparedCorrespondenceMode.FULL else 0)
    if trace_count > limits.max_validation_trace_records:
        _fail(PreparedEvaluationStatus.CAP_EXCEEDED, "trace-cap", "validation trace cap exceeded")
    if type(build.counts) is PreparedBuildCounts:
        villain_info_count = build.counts.villain_information_sets
        minimum_records = 3 + 1 + len(hero.entries) + trace_count
        if villain is not None:
            minimum_records += 2 + len(villain.entries)
        minimum_records += 1 + villain_info_count
        minimum_records += 2 * villain_info_count
        if request.correspondence_mode is PreparedCorrespondenceMode.FULL:
            minimum_records += 1 + villain_info_count
        if minimum_records > limits.max_result_records:
            _fail(PreparedEvaluationStatus.CAP_EXCEEDED, "result-cap", "minimum result record cap exceeded")


@dataclass(frozen=True)
class _BuildView:
    hero_actions: dict[str, tuple[str, ...]]
    villain_actions: dict[str, tuple[str, ...]]


def _verify_build(build: PreparedTwoStreetBuild) -> _BuildView:
    if type(build.tree) is not GameTree or type(build.identity) is not PreparedTwoStreetIdentity or type(build.counts) is not PreparedBuildCounts:
        _fail(PreparedEvaluationStatus.INVALID_INPUT, "build", "invalid M14 public dataclass boundary")
    if not isinstance(build.chance_normalization, tuple) or not isinstance(build.information_sets, tuple):
        _fail(PreparedEvaluationStatus.INVALID_INPUT, "build", "M14 artifacts must be tuples")
    counts = build.counts
    for field in fields(counts):
        value = getattr(counts, field.name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "build-counts", "invalid declared count")
    if counts.hero_pure_plans <= 0 or counts.villain_pure_strategies <= 0:
        _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "build-counts", "pure strategy counts must be positive")
    for name, ceiling in _M14_COUNT_LIMITS.items():
        if getattr(counts, name) > ceiling:
            _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "build-counts", f"declared {name} exceeds M14 ceiling")
    if counts.hero_information_sets + counts.villain_information_sets > 2_000:
        _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "build-counts", "total information-set ceiling exceeded")
    if len(build.information_sets) != counts.hero_information_sets + counts.villain_information_sets or len(build.chance_normalization) != counts.transition_rows:
        _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "build-counts", "artifact lengths disagree with counts")
    identity = build.identity
    expected_ids = (
        (identity.contract_version, PREPARED_TWO_STREET_CONTRACT_VERSION),
        (identity.builder_id, PREPARED_TWO_STREET_BUILDER_ID),
        (identity.action_label_id, _M14_ACTION_LABEL_ID),
        (identity.normalization_id, PREPARED_CHANCE_NORMALIZATION_ID),
        (identity.information_key_id, PREPARED_INFORMATION_KEY_ID),
    )
    if any(actual != expected for actual, expected in expected_ids):
        _fail(PreparedEvaluationStatus.IDENTITY_MISMATCH, "build-identity", "M14 algorithm identity mismatch")
    for value in (identity.raw_sha256, identity.semantic_sha256, identity.ordered_tree_sha256, identity.run_identity):
        if not _valid_sha(value):
            _fail(PreparedEvaluationStatus.IDENTITY_MISMATCH, "build-identity", "invalid M14 SHA-256")

    seen_objects: set[int] = set()
    seen_ids: dict[str, Any] = {}
    transition_probabilities: dict[str, tuple[float, ...]] = {}
    decision = terminal = chance_edges = total = max_depth = 0

    def walk(node: Any, depth: int) -> Any:
        nonlocal decision, terminal, chance_edges, total, max_depth
        marker = id(node)
        if marker in seen_objects:
            _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "ordered-tree", "shared-child DAG or cycle detected")
        seen_objects.add(marker)
        node_id = getattr(node, "node_id", None)
        if not isinstance(node_id, str) or not node_id:
            _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "ordered-tree", "invalid node ID")
        if node_id in seen_ids and seen_ids[node_id] is not node:
            _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "ordered-tree", "duplicate node ID")
        seen_ids[node_id] = node
        total += 1
        max_depth = max(max_depth, depth)
        if total > 250_000 or total > counts.total_nodes or depth > 32:
            _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "ordered-tree", "actual tree exceeds declared or hard bounds")
        if type(node) is TerminalNode:
            terminal += 1
            values = (node.hero_ev, node.villain_ev, node.house_rake)
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in values):
                _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "ordered-tree", "non-finite terminal")
            return ("terminal", node_id, float(node.hero_ev).hex(), float(node.villain_ev).hex(), float(node.house_rake).hex())
        if type(node) is ChanceNode:
            if not isinstance(node.children, tuple) or any(not isinstance(edge, tuple) or len(edge) != 2 for edge in node.children):
                _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "ordered-tree", "invalid chance child structure")
            chance_edges += len(node.children)
            if chance_edges > 100_000 or chance_edges > counts.chance_edges:
                _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "ordered-tree", "chance edge count exceeds bound")
            rows = []
            probabilities = []
            for probability, child in node.children:
                if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(float(probability)):
                    _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "ordered-tree", "invalid chance probability")
                number = float(probability)
                probabilities.append(number)
                rows.append((number.hex(), walk(child, depth + 1)))
            if node_id != "node:prepared-two-street-root":
                transition_probabilities[node_id] = tuple(probabilities)
            return ("chance", node_id, tuple(rows))
        if type(node) in (HeroNode, VillainNode):
            if not isinstance(node.actions, tuple) or any(not isinstance(edge, tuple) or len(edge) != 2 or not isinstance(edge[0], str) for edge in node.actions):
                _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "ordered-tree", "invalid decision action structure")
            decision += 1
            if decision > 100_000 or decision > counts.decision_nodes or not isinstance(node.info_set, str):
                _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "ordered-tree", "decision count or information set invalid")
            kind = "hero" if type(node) is HeroNode else "villain"
            return (kind, node_id, node.info_set, tuple((label, walk(child, depth + 1)) for label, child in node.actions))
        _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "ordered-tree", "unknown generic node type")

    if type(build.tree.root) is not ChanceNode or build.tree.root.node_id != "node:prepared-two-street-root":
        _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "build-root", "invalid prepared tree root")
    record = walk(build.tree.root, 0)
    actual = (len(build.tree.root.children), decision, terminal, total, chance_edges, max_depth)
    declared = (counts.root_matchups, counts.decision_nodes, counts.terminal_nodes, counts.total_nodes, counts.chance_edges, counts.max_depth_edges)
    if actual != declared:
        _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "build-counts", "actual tree counts disagree with declaration")
    ordered = _sha({"algorithm": _M14_ORDERED_TREE_ID, "tree": record})
    if ordered != identity.ordered_tree_sha256:
        _fail(PreparedEvaluationStatus.IDENTITY_MISMATCH, "ordered-tree", "ordered-tree identity mismatch")

    row_ids: set[tuple[str, str, str, str]] = set()
    transition_node_ids: set[str] = set()
    previous_row: tuple[str, str, str, str] | None = None
    transition_edge_count = 0
    for item in build.chance_normalization:
        if type(item) is not PreparedChanceNormalizationRecord or not isinstance(item.row_identity, tuple) or not isinstance(item.edge_identities, tuple):
            _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "normalization-artifacts", "invalid normalization record")
        if len(item.row_identity) != 4 or any(not isinstance(value, str) for value in item.row_identity) or any(not isinstance(edge, tuple) or len(edge) != 3 or any(not isinstance(value, str) for value in edge) for edge in item.edge_identities):
            _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "normalization-artifacts", "invalid normalization identity shape")
        if item.row_identity in row_ids or (previous_row is not None and item.row_identity <= previous_row):
            _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "normalization-artifacts", "duplicate or unordered normalization row")
        row_ids.add(item.row_identity)
        previous_row = item.row_identity
        if tuple(sorted(item.edge_identities)) != item.edge_identities or len(set(item.edge_identities)) != len(item.edge_identities):
            _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "normalization-artifacts", "duplicate or unordered normalization edge")
        values = (item.raw_sum, item.normalization_factor, *item.effective_probabilities)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) <= 0 for v in values):
            _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "normalization-artifacts", "invalid normalization numeric")
        raw_sum = float(item.raw_sum)
        factor = float(item.normalization_factor)
        effective = tuple(float(value) for value in item.effective_probabilities)
        if abs(raw_sum - 1.0) > 1e-9 or factor != 1.0 / raw_sum:
            _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "normalization-artifacts", "normalization raw sum or factor mismatch")
        transition_id, source_id, hero_bucket, villain_bucket = item.row_identity
        transition_node_id = "node:sha256:" + _sha({
            "kind": "transition",
            "public_history_id": source_id,
            "hero_bucket_history": (hero_bucket,),
            "villain_bucket_history": (villain_bucket,),
            "discriminator": transition_id,
        })
        if transition_node_id in transition_node_ids or transition_probabilities.get(transition_node_id) != effective:
            _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "normalization-artifacts", "normalization artifact disagrees with materialized transition")
        transition_node_ids.add(transition_node_id)
        if len(item.edge_identities) != len(effective) or abs(math.fsum(effective) - 1.0) > 1e-9:
            _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "normalization-artifacts", "normalization arithmetic mismatch")
        transition_edge_count += len(item.edge_identities)
    if transition_edge_count != counts.chance_edges - counts.root_matchups:
        _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "normalization-artifacts", "normalization edge count disagrees with tree counts")
    if transition_node_ids != set(transition_probabilities):
        _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "normalization-artifacts", "normalization rows do not exactly cover materialized transitions")

    hero_artifacts: dict[str, tuple[str, ...]] = {}
    villain_artifacts: dict[str, tuple[str, ...]] = {}
    previous_info: str | None = None
    for item in build.information_sets:
        if type(item) is not PreparedInformationSetObservation or type(item.key) is not PreparedInformationSetKey:
            _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "information-artifacts", "invalid information artifact")
        if not _INFO_RE.fullmatch(item.info_set_id) or previous_info is not None and item.info_set_id <= previous_info:
            _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "information-artifacts", "invalid, duplicate, or unordered information ID")
        previous_info = item.info_set_id
        if not _OBS_RE.fullmatch(item.public_observation_identity):
            _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "information-artifacts", "invalid public observation identity")
        if "info:sha256:" + _sha(item.key) != item.info_set_id:
            _fail(PreparedEvaluationStatus.IDENTITY_MISMATCH, "information-artifacts", "information key digest mismatch")
        if item.key.player not in (PreparedPlayer.HERO, PreparedPlayer.VILLAIN) or not isinstance(item.legal_action_labels, tuple) or not item.legal_action_labels or any(not isinstance(a, str) or not _ACTION_RE.fullmatch(a) for a in item.legal_action_labels):
            _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "information-artifacts", "invalid player or legal action labels")
        target = hero_artifacts if item.key.player is PreparedPlayer.HERO else villain_artifacts
        target[item.info_set_id] = item.legal_action_labels
    try:
        hero_tree = collect_hero_info_sets(build.tree)
        villain_tree = collect_villain_info_sets(build.tree)
    except Exception as exc:
        _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "information-artifacts", f"generic information collection failed: {type(exc).__name__}")
    if hero_tree != hero_artifacts or villain_tree != villain_artifacts or len(hero_artifacts) != counts.hero_information_sets or len(villain_artifacts) != counts.villain_information_sets:
        _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "information-artifacts", "tree and information artifacts disagree")
    if math.prod(len(actions) for actions in hero_artifacts.values()) != counts.hero_pure_plans or math.prod(len(actions) for actions in villain_artifacts.values()) != counts.villain_pure_strategies:
        _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "build-counts", "declared pure-plan count disagrees with information artifacts")
    base = {
        "contract_version": identity.contract_version,
        "builder_id": identity.builder_id,
        "action_label_id": identity.action_label_id,
        "normalization_id": identity.normalization_id,
        "information_key_id": identity.information_key_id,
        "raw_sha256": identity.raw_sha256,
        "semantic_sha256": identity.semantic_sha256,
        "ordered_tree_sha256": identity.ordered_tree_sha256,
    }
    if _sha(base) != identity.run_identity:
        _fail(PreparedEvaluationStatus.IDENTITY_MISMATCH, "run-identity", "M14 run identity mismatch")
    try:
        validate_tree(build.tree, tolerance=1e-9)
    except Exception as exc:
        _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "validate-tree", f"tree validation failed: {type(exc).__name__}")
    return _BuildView(hero_artifacts, villain_artifacts)


def _normalize_profile(profile: PreparedPlayerProfile, legal: dict[str, tuple[str, ...]], player: str) -> tuple[PreparedProfileNormalization, dict[str, dict[str, float]]]:
    by_info: dict[str, PreparedProfileEntry] = {}
    for entry in profile.entries:
        if type(entry) is not PreparedProfileEntry or not isinstance(entry.info_set_id, str):
            _fail(PreparedEvaluationStatus.INVALID_INPUT, "profile", "invalid profile entry")
        if entry.info_set_id in by_info:
            _fail(PreparedEvaluationStatus.INVALID_INPUT, "profile", "duplicate information-set entry")
        by_info[entry.info_set_id] = entry
    missing = set(legal) - set(by_info)
    extra = set(by_info) - set(legal)
    if missing:
        _fail(PreparedEvaluationStatus.INCOMPLETE_PROFILE, "profile", f"{player} profile is incomplete")
    if extra:
        _fail(PreparedEvaluationStatus.ILLEGAL_PROFILE_REFERENCE, "profile", f"{player} profile references an unknown information set")
    records: list[PreparedProfileNormalizationRecord] = []
    strategy: dict[str, dict[str, float]] = {}
    raw_entries = []
    effective_entries = []
    for info_id in sorted(legal):
        actions = legal[info_id]
        supplied: dict[str, PreparedActionProbability] = {}
        for item in by_info[info_id].action_probabilities:
            if type(item) is not PreparedActionProbability or not isinstance(item.action_label, str):
                _fail(PreparedEvaluationStatus.INVALID_INPUT, "profile", "invalid action probability record")
            if item.action_label in supplied:
                _fail(PreparedEvaluationStatus.INVALID_INPUT, "profile", "duplicate action entry")
            supplied[item.action_label] = item
        missing_actions = set(actions) - set(supplied)
        extra_actions = set(supplied) - set(actions)
        if missing_actions:
            _fail(PreparedEvaluationStatus.INCOMPLETE_PROFILE, "profile", "profile action coverage is incomplete")
        if extra_actions:
            _fail(PreparedEvaluationStatus.ILLEGAL_PROFILE_REFERENCE, "profile", "profile references an unknown action")
        raw_values = []
        for action in actions:
            value = supplied[action].probability
            if isinstance(value, bool):
                _fail(PreparedEvaluationStatus.INVALID_INPUT, "profile-probability", "bool probability is invalid")
            if not isinstance(value, (int, float)):
                _fail(PreparedEvaluationStatus.INVALID_INPUT, "profile-probability", "probability must be numeric")
            try:
                number = float(value)
            except (OverflowError, ValueError):
                _fail(PreparedEvaluationStatus.INVALID_PROFILE_PROBABILITY, "profile-probability", "probability is not binary64")
            if not math.isfinite(number) or number < 0:
                _fail(PreparedEvaluationStatus.INVALID_PROFILE_PROBABILITY, "profile-probability", "probability must be finite and non-negative")
            raw_values.append(number)
        raw = tuple(raw_values)
        try:
            raw_sum = math.fsum(raw)
        except (OverflowError, ValueError):
            _fail(PreparedEvaluationStatus.INVALID_PROFILE_PROBABILITY, "profile-probability", "profile probability sum is not finite binary64")
        if not math.isfinite(raw_sum) or raw_sum <= 0 or abs(raw_sum - 1.0) > PREPARED_PROFILE_NORMALIZATION_TOLERANCE:
            _fail(PreparedEvaluationStatus.INVALID_PROFILE_PROBABILITY, "profile-probability", "probabilities must sum to one within tolerance")
        factor = 1.0 / raw_sum
        effective = tuple(value * factor for value in raw)
        if not math.isfinite(factor) or any(not math.isfinite(value) for value in effective) or any(r > 0 and e == 0 for r, e in zip(raw, effective)) or abs(math.fsum(effective) - 1.0) > PREPARED_PROFILE_NORMALIZATION_TOLERANCE:
            _fail(PreparedEvaluationStatus.NUMERIC_FAILURE, "profile-normalization", "profile normalization failed")
        records.append(PreparedProfileNormalizationRecord(info_id, actions, raw, raw_sum, factor, effective))
        strategy[info_id] = dict(zip(actions, effective))
        raw_entries.append({"info_set_id": info_id, "probabilities": [{"action_label": a, "probability_hex": p.hex()} for a, p in zip(actions, raw)]})
        effective_entries.append({"info_set_id": info_id, "probabilities": [{"action_label": a, "probability_hex": p.hex()} for a, p in zip(actions, effective)]})
    raw_object = {"algorithm": PREPARED_PROFILE_RAW_ID, "player": player, "entries": raw_entries}
    effective_object = {"algorithm": PREPARED_PROFILE_EFFECTIVE_ID, "player": player, "entries": effective_entries}
    normalization = PreparedProfileNormalization(_sha_plain(raw_object), _sha_plain(effective_object), tuple(records))
    return normalization, strategy


def _sha_plain(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _finite_core(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        _fail(PreparedEvaluationStatus.NUMERIC_FAILURE, "core-output", f"non-finite {name}")
    return float(value)


def _canonical_response(strategy: Any, legal: dict[str, tuple[str, ...]]) -> PreparedPureResponse:
    if not isinstance(strategy, dict) or set(strategy) != set(legal):
        _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "exact-response", "response does not exactly cover Villain information sets")
    assignments = []
    for info_id in sorted(legal):
        action = strategy[info_id]
        if action not in legal[info_id]:
            _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "exact-response", "response contains an illegal action")
        assignments.append(PreparedResponseAssignment(info_id, action))
    return PreparedPureResponse(tuple(assignments))


@dataclass(frozen=True)
class _ResponseSummary:
    numeric: tuple[float, float, float, float, float]
    pure_count: int
    best_count: int
    representative: PreparedPureResponse
    action_sets: tuple[PreparedResponseActionSet, ...]
    variation: tuple[PreparedResponseVariation, ...]
    off_path: tuple[str, ...]


@dataclass(frozen=True)
class _ResponseShape:
    numeric: tuple[float, float, float, float, float]
    pure_count: int
    best_count: int
    response_records: int
    action_set_records: int
    materialized_records: int
    representative: tuple[tuple[str, str], ...]
    action_sets: tuple[tuple[str, tuple[str, ...]], ...] | None
    variation: tuple[tuple[str, tuple[str, ...]], ...]
    off_path: tuple[str, ...]
    materialized: tuple[tuple[tuple[str, str], ...], ...]


def _response_shape(
    result: Any,
    legal: dict[str, tuple[str, ...]],
    *,
    require_action_sets: bool,
    record_cap: int,
) -> _ResponseShape:
    if type(result) is not BestResponseResult:
        _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "exact-response", "invalid response result type")
    numeric = tuple(_finite_core(v, n) for v, n in zip((result.villain_max_ev, result.ev_h_worst, result.ev_h_best, result.expected_house_rake_worst, result.expected_house_rake_best), ("villain max", "Hero worst", "Hero best", "rake worst", "rake best")))
    if result.ev_h_worst > result.ev_h_best + 1e-9 or result.expected_house_rake_worst < -1e-9 or result.expected_house_rake_best < -1e-9:
        _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "exact-response", "invalid response interval or rake")
    for value in (result.num_villain_pure_strategies, result.num_best_response_strategies):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "exact-response", "invalid response count")
    if result.num_best_response_strategies > result.num_villain_pure_strategies:
        _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "exact-response", "best-response count exceeds pure-strategy count")
    if not isinstance(result.best_response_strategies, list) or not result.best_response_strategies:
        _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "exact-response", "response representative is absent")
    if len(result.best_response_strategies) not in (1, result.num_best_response_strategies):
        _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "exact-response", "invalid materialized response count")
    strategy_keys = []
    for strategy in result.best_response_strategies:
        if not isinstance(strategy, dict) or set(strategy) != set(legal):
            _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "exact-response", "response does not exactly cover Villain information sets")
        key = []
        for info_id in sorted(legal):
            action = strategy[info_id]
            if action not in legal[info_id]:
                _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "exact-response", "response contains an illegal action")
            key.append((info_id, action))
        strategy_keys.append(tuple(key))
    if len(strategy_keys) != len(set(strategy_keys)):
        _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "exact-response", "duplicate materialized response")
    materialized_records = 0
    for strategy_key in strategy_keys:
        materialized_records = _checked_add(
            materialized_records,
            1,
            record_cap,
            "result-cap",
            "materialized full response outer records",
        )
        for _assignment in strategy_key:
            materialized_records = _checked_add(
                materialized_records,
                1,
                record_cap,
                "result-cap",
                "materialized full response assignments",
            )
    raw_sets = result.best_response_action_sets
    if require_action_sets and (not isinstance(raw_sets, dict) or set(raw_sets) != set(legal)):
        _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "exact-response", "conditional action sets are incomplete")
    if raw_sets is not None and (not isinstance(raw_sets, dict) or set(raw_sets) != set(legal)):
        _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "exact-response", "invalid conditional action-set structure")
    action_set_records = 0
    action_sets = None
    if raw_sets is not None:
        canonical_sets = []
        for info_id in sorted(legal):
            values = raw_sets.get(info_id)
            if not isinstance(values, list) or not values or len(values) != len(set(values)) or any(v not in legal[info_id] for v in values):
                _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "exact-response", "invalid conditional action set")
            if require_action_sets and result.best_response_strategies[0][info_id] not in values:
                _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "exact-response", "representative disagrees with conditional action set")
            ordered = tuple(action for action in legal[info_id] if action in values)
            canonical_sets.append((info_id, ordered))
            action_set_records += 1 + len(values)
        action_sets = tuple(canonical_sets)
    if not isinstance(result.best_response_action_variation, dict):
        _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "exact-response", "invalid response variation")
    if any(not isinstance(info_id, str) or info_id not in legal for info_id in result.best_response_action_variation):
        _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "exact-response", "unknown variation information set")
    variation_records = 0
    canonical_variation = []
    for info_id in sorted(result.best_response_action_variation):
        if info_id not in legal:
            _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "exact-response", "unknown variation information set")
        values = result.best_response_action_variation[info_id]
        if not isinstance(values, list) or len(values) < 2 or len(values) != len(set(values)) or any(v not in legal[info_id] for v in values):
            _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "exact-response", "invalid response variation actions")
        if result.best_response_strategies[0][info_id] not in values:
            _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "exact-response", "representative disagrees with response variation")
        canonical_variation.append((info_id, tuple(action for action in legal[info_id] if action in values)))
        variation_records += 1 + len(values)
    if not isinstance(result.off_path_info_sets, list) or any(not isinstance(i, str) for i in result.off_path_info_sets) or result.off_path_info_sets != sorted(result.off_path_info_sets) or len(result.off_path_info_sets) != len(set(result.off_path_info_sets)) or any(i not in legal for i in result.off_path_info_sets):
        _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "exact-response", "invalid off-path information sets")
    for info_id in result.off_path_info_sets:
        if len(legal[info_id]) > 1 and set(result.best_response_action_variation.get(info_id, ())) != set(legal[info_id]):
            _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "exact-response", "off-path freedom disagrees with response variation")
    if len(result.best_response_strategies) == result.num_best_response_strategies:
        materialized_variation = {
            info_id: {strategy[info_id] for strategy in result.best_response_strategies}
            for info_id in legal
        }
        expected_variation = {
            info_id: set(actions)
            for info_id, actions in result.best_response_action_variation.items()
        }
        if {info_id: actions for info_id, actions in materialized_variation.items() if len(actions) > 1} != expected_variation:
            _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "exact-response", "full responses disagree with response variation")
    records = 1 + len(legal) + action_set_records + variation_records + len(result.off_path_info_sets)
    return _ResponseShape(
        numeric,
        result.num_villain_pure_strategies,
        result.num_best_response_strategies,
        records,
        action_set_records,
        materialized_records,
        strategy_keys[0],
        action_sets,
        tuple(canonical_variation),
        tuple(result.off_path_info_sets),
        tuple(sorted(strategy_keys)),
    )


def _response_summary(result: Any, legal: dict[str, tuple[str, ...]], *, require_action_sets: bool, shape: _ResponseShape | None = None) -> _ResponseSummary:
    if shape is None:
        shape = _response_shape(
            result,
            legal,
            require_action_sets=require_action_sets,
            record_cap=_DEFAULT_LIMITS.max_result_records,
        )
    representative = _canonical_response(result.best_response_strategies[0], legal)
    action_sets = []
    if result.best_response_action_sets is not None:
        for info_id in sorted(legal):
            values = result.best_response_action_sets[info_id]
            action_sets.append(PreparedResponseActionSet(info_id, tuple(action for action in legal[info_id] if action in values)))
    variation = []
    for info_id in sorted(result.best_response_action_variation):
        values = result.best_response_action_variation[info_id]
        variation.append(PreparedResponseVariation(info_id, tuple(action for action in legal[info_id] if action in values)))
    return _ResponseSummary(shape.numeric, shape.pure_count, shape.best_count, representative, tuple(action_sets), tuple(variation), tuple(result.off_path_info_sets))


def _full_tuple(result: BestResponseResult, legal: dict[str, tuple[str, ...]], expected: int) -> tuple[PreparedPureResponse, ...]:
    if len(result.best_response_strategies) != expected:
        _fail(PreparedEvaluationStatus.NON_REPRODUCIBLE, "full-correspondence", "full correspondence length mismatch")
    values = tuple(sorted((_canonical_response(strategy, legal) for strategy in result.best_response_strategies), key=lambda item: tuple((a.info_set_id, a.action_label) for a in item.assignments)))
    if len(set(values)) != len(values):
        _fail(PreparedEvaluationStatus.NON_REPRODUCIBLE, "full-correspondence", "duplicate full response")
    return values


def _raw_summary_equal(left: _ResponseShape, right: _ResponseShape) -> bool:
    return (
        all(abs(a - b) <= 1e-9 for a, b in zip(left.numeric, right.numeric))
        and left.pure_count == right.pure_count
        and left.best_count == right.best_count
        and left.variation == right.variation
        and left.off_path == right.off_path
    )


def _raw_repeated_dp_equal(first: _ResponseShape, second: _ResponseShape) -> bool:
    return (
        _raw_summary_equal(first, second)
        and first.representative == second.representative
        and first.action_sets == second.action_sets
        and len(second.materialized) == second.best_count
        and second.representative in second.materialized
    )


def _raw_oracle_equal(dp: _ResponseShape, enumerated: _ResponseShape) -> bool:
    return (
        _raw_summary_equal(dp, enumerated)
        and enumerated.action_sets is None
        and dp.materialized == enumerated.materialized
    )


def _result_record_count(hero_norm: PreparedProfileNormalization, villain_norm: PreparedProfileNormalization | None, fixed: PreparedFixedProfileValue | None, summary: _ResponseSummary, trace_count: int, full_count: int, villain_info_count: int, cap: int) -> int:
    total = 3 + 1 + len(hero_norm.records) + trace_count
    if villain_norm is not None:
        total += 1 + len(villain_norm.records)
    if fixed is not None:
        total += 1
    total += 1 + len(summary.representative.assignments)
    total += sum(1 + len(item.action_labels) for item in summary.action_sets)
    total += sum(1 + len(item.action_labels) for item in summary.variation)
    total += len(summary.off_path)
    total = _checked_add(total, _checked_mul(full_count, 1 + villain_info_count, cap, "result-cap", "full correspondence records"), cap, "result-cap", "result records")
    return total


def _raw_result_record_count(hero_norm: PreparedProfileNormalization, villain_norm: PreparedProfileNormalization | None, fixed: PreparedFixedProfileValue | None, shape: _ResponseShape, trace_count: int, full_records: int, cap: int) -> int:
    total = 3 + 1 + len(hero_norm.records) + trace_count
    if villain_norm is not None:
        total += 1 + len(villain_norm.records)
    if fixed is not None:
        total += 1
    total = _checked_add(total, shape.response_records, cap, "result-cap", "result records")
    return _checked_add(total, full_records, cap, "result-cap", "full correspondence records")


def _projected_full_response_records(best_count: int, villain_info_count: int, cap: int) -> int:
    records_per_response = _checked_add(
        1,
        villain_info_count,
        cap,
        "result-cap",
        "full response outer plus assignments",
    )
    return _checked_mul(
        best_count,
        records_per_response,
        cap,
        "result-cap",
        "full correspondence records",
    )


def _trace(hero_norm: PreparedProfileNormalization, villain_norm: PreparedProfileNormalization | None, fixed: bool, oracle: bool, full: bool) -> tuple[PreparedEvaluationTraceRecord, ...]:
    rows = [
        PreparedEvaluationTraceRecord("build", "algorithm-ids", "PASS"),
        PreparedEvaluationTraceRecord("build", "ordered-tree", "PASS"),
        PreparedEvaluationTraceRecord("build", "run-identity", "PASS"),
        PreparedEvaluationTraceRecord("build", "counts", "PASS"),
        PreparedEvaluationTraceRecord("build", "information-artifacts", "PASS"),
        PreparedEvaluationTraceRecord("build", "normalization-artifacts", "PASS"),
        PreparedEvaluationTraceRecord("build", "validate-tree", "PASS"),
    ]
    rows.extend(PreparedEvaluationTraceRecord("profile", f"hero-normalization:{item.info_set_id}", "PASS") for item in hero_norm.records)
    if villain_norm is not None:
        rows.extend(PreparedEvaluationTraceRecord("profile", f"villain-normalization:{item.info_set_id}", "PASS") for item in villain_norm.records)
    rows.append(PreparedEvaluationTraceRecord("evaluation", "exact-response", "PASS"))
    if fixed:
        rows.append(PreparedEvaluationTraceRecord("evaluation", "fixed-profile", "PASS"))
    if oracle:
        rows.append(PreparedEvaluationTraceRecord("evaluation", "oracle", "PASS"))
    if full:
        rows.append(PreparedEvaluationTraceRecord("evaluation", "full-correspondence", "PASS"))
    return tuple(rows)


def evaluate_prepared_two_street(
    build: PreparedTwoStreetBuild,
    hero_profile: PreparedPlayerProfile,
    villain_profile: PreparedPlayerProfile | None = None,
    limits: PreparedEvaluationLimits = PreparedEvaluationLimits(),
    request: PreparedEvaluationRequest = PreparedEvaluationRequest(),
) -> PreparedEvaluationResult:
    """Validate and evaluate a successful M14 prepared build in one bounded call.

    The DP conditional action sets remain the source for that field even when
    the explicitly requested tiny enumerator is the primary numeric result.
    """
    try:
        limits, request = _validate_outer(build, hero_profile, villain_profile, limits, request)
        _outer_caps(build, hero_profile, villain_profile, limits, request)
        view = _verify_build(build)
        hero_norm, hero_map = _normalize_profile(hero_profile, view.hero_actions, "hero")
        villain_norm = None
        villain_map = None
        if villain_profile is not None:
            villain_norm, villain_map = _normalize_profile(villain_profile, view.villain_actions, "villain")
        hero_strategy = HeroStrategy(hero_map)
        villain_strategy = VillainStrategy(villain_map) if villain_map is not None else None

        fixed_value = None
        if villain_strategy is not None:
            try:
                raw_fixed = evaluate_fixed_profile(build.tree, hero_strategy, villain_strategy, tolerance=1e-9, allow_negative_residual=False)
            except Exception as exc:
                _fail(PreparedEvaluationStatus.FIXED_PROFILE_FAILURE, "fixed-profile", f"fixed evaluator failed: {type(exc).__name__}")
            try:
                fixed_fields = (raw_fixed.hero_ev, raw_fixed.villain_ev, raw_fixed.house_rake)
            except Exception as exc:
                _fail(PreparedEvaluationStatus.FIXED_PROFILE_FAILURE, "fixed-profile", f"invalid fixed evaluator structure: {type(exc).__name__}")
            hero_ev = _finite_core(fixed_fields[0], "fixed Hero EV")
            villain_ev = _finite_core(fixed_fields[1], "fixed Villain EV")
            rake = _finite_core(fixed_fields[2], "fixed rake")
            residual = math.fsum((hero_ev, villain_ev, rake))
            if not math.isfinite(residual):
                _fail(PreparedEvaluationStatus.NUMERIC_FAILURE, "fixed-profile", "non-finite fixed residual")
            if rake < 0 or abs(residual) > 1e-9:
                _fail(PreparedEvaluationStatus.FIXED_PROFILE_FAILURE, "fixed-profile", "invalid fixed-profile accounting")
            fixed_value = PreparedFixedProfileValue(hero_ev, villain_ev, rake, residual)

        try:
            pure_count = count_villain_pure_strategies(build.tree)
        except Exception as exc:
            _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "exact-response", f"pure count failed: {type(exc).__name__}")
        if pure_count != build.counts.villain_pure_strategies:
            _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "exact-response", "Villain pure count disagrees with M14")
        try:
            dp_first = solve_exact_response(build.tree, hero_strategy, tolerance=1e-9, max_pure_strategies=1, method="dp", allow_negative_residual=False)
        except Exception as exc:
            _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "exact-response", f"DP failed: {type(exc).__name__}")
        dp_shape = _response_shape(
            dp_first,
            view.villain_actions,
            require_action_sets=True,
            record_cap=limits.max_result_records,
        )
        if dp_shape.pure_count != pure_count:
            _fail(PreparedEvaluationStatus.BUILD_MISMATCH, "exact-response", "DP pure count mismatch")

        variation = dp_first.best_response_action_variation
        generation = 1
        generation_cap = None
        if request.correspondence_mode is PreparedCorrespondenceMode.FULL:
            generation_cap = limits.max_full_correspondence_strategies
        elif request.oracle_check:
            generation_cap = limits.max_enumerator_pure_strategies
        if generation_cap is not None:
            for info_id in sorted(view.villain_actions):
                generation = _checked_mul(generation, len(variation.get(info_id, ("representative",))), generation_cap, "correspondence", "generation space")

        need_dp_full = request.correspondence_mode is PreparedCorrespondenceMode.FULL or request.oracle_check
        dp_full_result = None
        trace_count = 7 + len(hero_norm.records) + (len(villain_norm.records) if villain_norm else 0) + 1 + (1 if fixed_value else 0) + (1 if request.oracle_check else 0) + (1 if request.correspondence_mode is PreparedCorrespondenceMode.FULL else 0)
        if request.correspondence_mode is PreparedCorrespondenceMode.FULL:
            if generation > limits.max_full_correspondence_strategies or dp_shape.best_count > limits.max_full_correspondence_strategies:
                _fail(PreparedEvaluationStatus.CAP_EXCEEDED, "full-correspondence", "full correspondence cap exceeded")
            _checked_mul(dp_shape.best_count, len(view.villain_actions), limits.max_result_records, "result-cap", "nested full assignments")
        projected_full_records = (
            _projected_full_response_records(
                dp_shape.best_count,
                len(view.villain_actions),
                limits.max_result_records,
            )
            if request.correspondence_mode is PreparedCorrespondenceMode.FULL
            else 0
        )
        final_records = _raw_result_record_count(
            hero_norm,
            villain_norm,
            fixed_value,
            dp_shape,
            trace_count,
            projected_full_records,
            limits.max_result_records,
        )
        if request.oracle_check:
            if pure_count > limits.max_enumerator_pure_strategies:
                _fail(PreparedEvaluationStatus.CAP_EXCEEDED, "oracle", "enumerator pure-strategy cap exceeded")
        diagnostics = 0
        if request.oracle_check:
            diagnostics = _checked_mul(pure_count, len(view.villain_actions) + 2, limits.max_result_records, "oracle", "enumerator diagnostic records")
            projected_comparison = _projected_full_response_records(
                dp_shape.best_count,
                len(view.villain_actions),
                limits.max_result_records,
            )
            projected_diagnostics = _checked_add(
                diagnostics,
                projected_comparison,
                limits.max_result_records,
                "oracle",
                "oracle diagnostic records",
            )
            _checked_add(projected_diagnostics, final_records, limits.max_result_records, "oracle", "oracle diagnostics plus result records")
        if need_dp_full:
            dp_cap = limits.max_full_correspondence_strategies if request.correspondence_mode is PreparedCorrespondenceMode.FULL else limits.max_enumerator_pure_strategies
            if generation > dp_cap or dp_shape.best_count > dp_cap:
                _fail(PreparedEvaluationStatus.CAP_EXCEEDED, "correspondence", "internal DP correspondence cap exceeded")
            try:
                dp_full_result = solve_exact_response(build.tree, hero_strategy, tolerance=1e-9, max_pure_strategies=dp_cap, method="dp", allow_negative_residual=False)
            except Exception as exc:
                _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "full-correspondence", f"second DP failed: {type(exc).__name__}")
            dp_second_shape = _response_shape(
                dp_full_result,
                view.villain_actions,
                require_action_sets=True,
                record_cap=limits.max_result_records,
            )
            dp_second_full_records = (
                dp_second_shape.materialized_records
                if request.correspondence_mode is PreparedCorrespondenceMode.FULL
                else 0
            )
            dp_second_records = _raw_result_record_count(
                hero_norm,
                villain_norm,
                fixed_value,
                dp_second_shape,
                trace_count,
                dp_second_full_records,
                limits.max_result_records,
            )
            if request.oracle_check:
                dp_second_diagnostics = _checked_add(
                    diagnostics,
                    dp_second_shape.materialized_records,
                    limits.max_result_records,
                    "oracle",
                    "oracle diagnostic records",
                )
                _checked_add(dp_second_diagnostics, dp_second_records, limits.max_result_records, "oracle", "oracle diagnostics plus result records")
            if not _raw_repeated_dp_equal(dp_shape, dp_second_shape):
                _fail(PreparedEvaluationStatus.NON_REPRODUCIBLE, "full-correspondence", "repeated DP result mismatch")

        enum_result = None
        enum_shape = None
        if request.oracle_check:
            try:
                enum_result = solve_exact_response(build.tree, hero_strategy, tolerance=1e-9, max_pure_strategies=limits.max_enumerator_pure_strategies, method="enumerate", allow_negative_residual=False)
            except Exception as exc:
                _fail(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, "oracle", f"enumerator failed: {type(exc).__name__}")
            enum_shape = _response_shape(
                enum_result,
                view.villain_actions,
                require_action_sets=False,
                record_cap=limits.max_result_records,
            )
            enum_full_records = (
                enum_shape.materialized_records
                if request.correspondence_mode is PreparedCorrespondenceMode.FULL
                else 0
            )
            enum_records = _raw_result_record_count(
                hero_norm,
                villain_norm,
                fixed_value,
                enum_shape,
                trace_count,
                enum_full_records,
                limits.max_result_records,
            )
            if request.method == "enumerate" and enum_shape.action_sets is None:
                enum_records = _checked_add(
                    enum_records,
                    dp_shape.action_set_records,
                    limits.max_result_records,
                    "result-cap",
                    "DP conditional action-set records",
                )
            enum_diagnostics = _checked_add(
                diagnostics,
                enum_shape.materialized_records,
                limits.max_result_records,
                "oracle",
                "oracle diagnostic records",
            )
            _checked_add(enum_diagnostics, enum_records, limits.max_result_records, "oracle", "oracle diagnostics plus result records")
            if not _raw_oracle_equal(dp_second_shape, enum_shape):
                _fail(PreparedEvaluationStatus.ORACLE_MISMATCH, "oracle", "DP and enumerator disagree")

        dp_summary = _response_summary(dp_first, view.villain_actions, require_action_sets=True, shape=dp_shape)
        primary_summary = dp_summary
        full_source = dp_full_result
        if request.oracle_check:
            if request.method == "enumerate":
                enum_summary_raw = _response_summary(enum_result, view.villain_actions, require_action_sets=False, shape=enum_shape)
                primary_summary = replace(enum_summary_raw, action_sets=dp_summary.action_sets)
                full_source = enum_result

        returned_full = None
        if request.correspondence_mode is PreparedCorrespondenceMode.FULL:
            returned_full = _full_tuple(full_source, view.villain_actions, primary_summary.best_count)
        _result_record_count(hero_norm, villain_norm, fixed_value, primary_summary, trace_count, len(returned_full) if returned_full else 0, len(view.villain_actions), limits.max_result_records)
        trace = _trace(hero_norm, villain_norm, fixed_value is not None, request.oracle_check, request.correspondence_mode is PreparedCorrespondenceMode.FULL)
        exact = PreparedExactResponseValue(
            *primary_summary.numeric,
            primary_summary.pure_count,
            primary_summary.best_count,
            primary_summary.representative,
            dp_summary.action_sets,
            primary_summary.variation,
            primary_summary.off_path,
            PreparedCorrespondenceStatus.MATERIALIZED if returned_full is not None else PreparedCorrespondenceStatus.NOT_REQUESTED,
            returned_full,
        )
        evaluation = PreparedEvaluation(hero_norm, villain_norm, fixed_value, exact, trace)
        m14 = build.identity
        identity_base = PreparedEvaluationIdentity(
            m14.contract_version, m14.builder_id, m14.action_label_id, m14.normalization_id, m14.information_key_id,
            m14.raw_sha256, m14.semantic_sha256, m14.ordered_tree_sha256, m14.run_identity,
            hero_norm.raw_profile_sha256, hero_norm.effective_profile_sha256,
            villain_norm.raw_profile_sha256 if villain_norm else PREPARED_PROFILE_ABSENT,
            villain_norm.effective_profile_sha256 if villain_norm else PREPARED_PROFILE_ABSENT,
            PREPARED_TWO_STREET_EVALUATION_ADAPTER_ID, PREPARED_PROFILE_NORMALIZATION_ID,
            PREPARED_PROFILE_RAW_ID, PREPARED_PROFILE_EFFECTIVE_ID, PREPARED_EVALUATION_OUTPUT_ID,
            request.method, PREPARED_PROFILE_NORMALIZATION_TOLERANCE.hex(), PREPARED_EVALUATION_TOLERANCE.hex(),
            limits, request.correspondence_mode, request.oracle_check, "",
        )
        identity_without_output = {
            field.name: getattr(identity_base, field.name)
            for field in fields(identity_base)
            if field.name != "output_semantic_sha256"
        }
        output_hash = _sha({"algorithm": PREPARED_EVALUATION_OUTPUT_ID, "identity": identity_without_output, "evaluation": evaluation})
        identity = replace(identity_base, output_semantic_sha256=output_hash)
        if request.expected_output_semantic_sha256 is not None and request.expected_output_semantic_sha256 != output_hash:
            _fail(PreparedEvaluationStatus.NON_REPRODUCIBLE, "output-identity", "expected output semantic SHA-256 mismatch")
        return PreparedEvaluationResult(PreparedEvaluationStatus.SUCCESS, evaluation, identity, None)
    except _Failure as exc:
        return _failure(exc.status, exc.phase, exc.message)
    except Exception as exc:
        return _failure(PreparedEvaluationStatus.INVALID_INPUT, "evaluation", f"unexpected evaluation failure: {type(exc).__name__}")
