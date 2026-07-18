"""Strict versioned JSON workflow for prepared one- or two-street games.

The two-phase workflow first builds a profile template from generated M14
information-set identifiers, then accepts a complete Hero profile (and an
optional complete Villain profile) for M16 orchestration.  It is a bounded
adapter over the existing abstract prepared contracts, not a solver, an
equilibrium calculation, a real-card model, or gambling advice.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable

from .prepared_two_street import (
    PREPARED_TWO_STREET_CONTRACT_VERSION,
    PreparedActionEvent,
    PreparedActionKind,
    PreparedActionOption,
    PreparedBucket,
    PreparedChanceEdge,
    PreparedChanceEvent,
    PreparedContentIdentity,
    PreparedDataAttestation,
    PreparedDecisionMenu,
    PreparedHeadsUpChips,
    PreparedPlayer,
    PreparedRake,
    PreparedRoundCloseReason,
    PreparedShowdownValue,
    PreparedStreet,
    PreparedStreetCloseEvent,
    PreparedTransitionRow,
    PreparedTwoStreetSpec,
    PreparedTwoStreetStatus,
    build_prepared_two_street_game,
    prepared_public_history_id,
    prepared_semantic_sha256,
)
from .prepared_two_street_evaluation import (
    PreparedActionProbability,
    PreparedPlayerProfile,
    PreparedProfileEntry,
)
from .prepared_two_street_orchestration import (
    PreparedOrchestrationStatus,
    PreparedTwoStreetOrchestrationRequest,
    run_prepared_two_street_orchestration,
)


PREPARED_TWO_STREET_FILE_FORMAT = "prepared-two-street-file-v1"
PREPARED_TWO_STREET_TEMPLATE_ID = "prepared-two-street-profile-template-sha256-v1"
PREPARED_TWO_STREET_FILE_OUTPUT_ID = "prepared-two-street-file-output-v1"

__all__ = [
    "PREPARED_TWO_STREET_FILE_FORMAT",
    "PREPARED_TWO_STREET_TEMPLATE_ID",
    "PREPARED_TWO_STREET_FILE_OUTPUT_ID",
    "PreparedFileWorkflowStatus",
    "PreparedFileWorkflowLimits",
    "PreparedFileWorkflowError",
    "PreparedFileWorkflowResult",
    "inspect_prepared_two_street_file",
    "run_prepared_two_street_file",
    "prepared_file_workflow_json",
]


class PreparedFileWorkflowStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARSE_FAILURE = "PARSE_FAILURE"
    INVALID_INPUT = "INVALID_INPUT"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    BUILD_FAILURE = "BUILD_FAILURE"
    PROFILE_FAILURE = "PROFILE_FAILURE"
    ORCHESTRATION_FAILURE = "ORCHESTRATION_FAILURE"
    CAP_EXCEEDED = "CAP_EXCEEDED"
    NON_REPRODUCIBLE = "NON_REPRODUCIBLE"
    INTERNAL_FAILURE = "INTERNAL_FAILURE"


@dataclass(frozen=True)
class PreparedFileWorkflowLimits:
    max_input_bytes: int = 1_000_000
    max_json_depth: int = 32
    max_total_json_values: int = 250_000
    max_history_events: int = 32
    max_output_records: int = 250_000


@dataclass(frozen=True)
class PreparedFileWorkflowError:
    phase: str
    message: str
    nested_status: str | None


@dataclass(frozen=True)
class PreparedFileWorkflowResult:
    status: PreparedFileWorkflowStatus
    output: dict[str, Any] | None
    error: PreparedFileWorkflowError | None


_DEFAULT_LIMITS = PreparedFileWorkflowLimits()
_LIMIT_CEILINGS = asdict(_DEFAULT_LIMITS)


class _WorkflowFailure(ValueError):
    def __init__(
        self,
        status: PreparedFileWorkflowStatus,
        phase: str,
        message: str,
        nested_status: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.phase = phase
        self.nested_status = nested_status


def _failure(exc: _WorkflowFailure) -> PreparedFileWorkflowResult:
    message = str(exc).replace("\r", " ").replace("\n", " ")[:500]
    return PreparedFileWorkflowResult(
        exc.status,
        None,
        PreparedFileWorkflowError(exc.phase[:64], message, exc.nested_status),
    )


def _validate_limits(limits: PreparedFileWorkflowLimits) -> None:
    if type(limits) is not PreparedFileWorkflowLimits:
        raise _WorkflowFailure(
            PreparedFileWorkflowStatus.INVALID_INPUT, "limits", "limits have the wrong type"
        )
    for name, ceiling in _LIMIT_CEILINGS.items():
        value = getattr(limits, name)
        if type(value) is not int or value <= 0 or value > ceiling:
            raise _WorkflowFailure(
                PreparedFileWorkflowStatus.INVALID_INPUT,
                "limits",
                f"{name} must be a positive int no greater than {ceiling}",
            )


def _duplicates_rejected(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _WorkflowFailure(
                PreparedFileWorkflowStatus.PARSE_FAILURE,
                "json",
                f"duplicate JSON key: {key}",
            )
        result[key] = value
    return result


def _parse(raw: bytes, limits: PreparedFileWorkflowLimits) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise _WorkflowFailure(
            PreparedFileWorkflowStatus.INVALID_INPUT, "input", "input must be bytes"
        )
    if len(raw) > limits.max_input_bytes:
        raise _WorkflowFailure(
            PreparedFileWorkflowStatus.CAP_EXCEEDED,
            "input",
            "input byte cap exceeded",
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _WorkflowFailure(
            PreparedFileWorkflowStatus.PARSE_FAILURE, "json", "input is not UTF-8"
        ) from exc

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=_duplicates_rejected,
            parse_constant=reject_constant,
        )
    except _WorkflowFailure:
        raise
    except RecursionError as exc:
        raise _WorkflowFailure(
            PreparedFileWorkflowStatus.CAP_EXCEEDED,
            "json",
            "JSON nesting exceeds the parser depth cap",
        ) from exc
    except (ValueError, TypeError) as exc:
        raise _WorkflowFailure(
            PreparedFileWorkflowStatus.PARSE_FAILURE,
            "json",
            f"invalid JSON: {exc}",
        ) from exc
    if type(value) is not dict:
        raise _WorkflowFailure(
            PreparedFileWorkflowStatus.INVALID_INPUT,
            "document",
            "top-level JSON value must be an object",
        )
    count = 0
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        count += 1
        if count > limits.max_total_json_values:
            raise _WorkflowFailure(
                PreparedFileWorkflowStatus.CAP_EXCEEDED,
                "json",
                "total JSON value cap exceeded",
            )
        if depth > limits.max_json_depth:
            raise _WorkflowFailure(
                PreparedFileWorkflowStatus.CAP_EXCEEDED,
                "json",
                "JSON depth cap exceeded",
            )
        if type(item) is dict:
            stack.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            stack.extend((child, depth + 1) for child in item)
    return value


def _object(
    value: Any,
    required: set[str],
    optional: set[str] = frozenset(),
    *,
    phase: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise _WorkflowFailure(
            PreparedFileWorkflowStatus.INVALID_INPUT, phase, "value must be an object"
        )
    keys = set(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise _WorkflowFailure(
            PreparedFileWorkflowStatus.INVALID_INPUT,
            phase,
            f"missing keys: {', '.join(sorted(missing))}",
        )
    if unknown:
        raise _WorkflowFailure(
            PreparedFileWorkflowStatus.INVALID_INPUT,
            phase,
            f"unknown keys: {', '.join(sorted(unknown))}",
        )
    return value


def _array(value: Any, cap: int, phase: str) -> list[Any]:
    if type(value) is not list:
        raise _WorkflowFailure(
            PreparedFileWorkflowStatus.INVALID_INPUT, phase, "value must be an array"
        )
    if len(value) > cap:
        raise _WorkflowFailure(
            PreparedFileWorkflowStatus.CAP_EXCEEDED, phase, "array cap exceeded"
        )
    return value


def _text(value: Any, phase: str) -> str:
    if type(value) is not str or not value or len(value) > 200:
        raise _WorkflowFailure(
            PreparedFileWorkflowStatus.INVALID_INPUT,
            phase,
            "value must be a non-empty string of at most 200 characters",
        )
    return value


def _number(value: Any, phase: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise _WorkflowFailure(
            PreparedFileWorkflowStatus.INVALID_INPUT, phase, "value must be a finite number"
        )
    return float(value)


def _nullable_number(value: Any, phase: str) -> float | None:
    return None if value is None else _number(value, phase)


def _boolean(value: Any, phase: str) -> bool:
    if type(value) is not bool:
        raise _WorkflowFailure(
            PreparedFileWorkflowStatus.INVALID_INPUT, phase, "value must be a boolean"
        )
    return value


def _enum(enum_type: type[Enum], value: Any, phase: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise _WorkflowFailure(
            PreparedFileWorkflowStatus.INVALID_INPUT, phase, "unsupported enum value"
        ) from exc


def _event(value: Any, phase: str) -> Any:
    base = _object(value, {"type"}, {
        "street_id", "player", "kind", "size_id", "raise_to", "is_all_in",
        "reopen", "reason", "transition_id", "public_outcome_id"
    }, phase=phase)
    event_type = base["type"]
    if event_type == "action":
        item = _object(value, {
            "type", "street_id", "player", "kind", "size_id", "raise_to",
            "is_all_in", "reopen"
        }, phase=phase)
        return PreparedActionEvent(
            _text(item["street_id"], phase),
            _enum(PreparedPlayer, item["player"], phase),
            _enum(PreparedActionKind, item["kind"], phase),
            None if item["size_id"] is None else _text(item["size_id"], phase),
            _nullable_number(item["raise_to"], phase),
            _boolean(item["is_all_in"], phase),
            _boolean(item["reopen"], phase),
        )
    if event_type == "street_close":
        item = _object(value, {"type", "street_id", "reason"}, phase=phase)
        return PreparedStreetCloseEvent(
            _text(item["street_id"], phase),
            _enum(PreparedRoundCloseReason, item["reason"], phase),
        )
    if event_type == "chance":
        item = _object(
            value, {"type", "transition_id", "public_outcome_id"}, phase=phase
        )
        return PreparedChanceEvent(
            _text(item["transition_id"], phase),
            _text(item["public_outcome_id"], phase),
        )
    raise _WorkflowFailure(
        PreparedFileWorkflowStatus.INVALID_INPUT, phase, "unsupported event type"
    )


def _history(value: Any, limits: PreparedFileWorkflowLimits, phase: str) -> str:
    items = _array(value, limits.max_history_events, phase)
    return prepared_public_history_id(
        tuple(_event(item, f"{phase}[{index}]") for index, item in enumerate(items))
    )


def _spec(
    value: Any, limits: PreparedFileWorkflowLimits
) -> tuple[PreparedTwoStreetSpec, bytes, PreparedContentIdentity]:
    item = _object(value, {
        "contract_version", "attestation", "starting_chips", "initial_committed",
        "rake", "streets", "hero_buckets", "villain_buckets", "decision_menus",
        "transition_id", "transition_rows", "showdown_values"
    }, phase="spec")
    if item["contract_version"] != PREPARED_TWO_STREET_CONTRACT_VERSION:
        raise _WorkflowFailure(
            PreparedFileWorkflowStatus.INVALID_INPUT,
            "spec",
            "unsupported prepared contract_version",
        )
    att = _object(item["attestation"], {
        "source", "bucket_semantics", "conditional_probability_semantics",
        "observation_mapping", "perfect_recall_attested"
    }, phase="spec.attestation")
    def chips(source: Any, phase: str) -> PreparedHeadsUpChips:
        obj = _object(source, {"hero", "villain"}, phase=phase)
        return PreparedHeadsUpChips(_number(obj["hero"], phase), _number(obj["villain"], phase))
    rake_obj = _object(item["rake"], {"rate", "cap"}, phase="spec.rake")
    streets = _array(item["streets"], 2, "spec.streets")
    hero_buckets = _array(item["hero_buckets"], 10_000, "spec.hero_buckets")
    villain_buckets = _array(item["villain_buckets"], 10_000, "spec.villain_buckets")
    menus = _array(item["decision_menus"], 100_000, "spec.decision_menus")
    rows = _array(item["transition_rows"], 50_000, "spec.transition_rows")
    showdowns = _array(item["showdown_values"], 100_000, "spec.showdown_values")

    def bucket(source: Any, phase: str) -> PreparedBucket:
        obj = _object(source, {"bucket_id", "weight"}, phase=phase)
        return PreparedBucket(_text(obj["bucket_id"], phase), _number(obj["weight"], phase))

    def action(source: Any, phase: str) -> PreparedActionOption:
        obj = _object(
            source, {"kind", "size_id", "raise_to", "is_all_in"}, phase=phase
        )
        return PreparedActionOption(
            _enum(PreparedActionKind, obj["kind"], phase),
            None if obj["size_id"] is None else _text(obj["size_id"], phase),
            _nullable_number(obj["raise_to"], phase),
            _boolean(obj["is_all_in"], phase),
        )

    street_values = []
    for index, source in enumerate(streets):
        phase = f"spec.streets[{index}]"
        obj = _object(
            source, {"street_id", "label", "first_actor", "min_open_bet"}, phase=phase
        )
        street_values.append(PreparedStreet(
            _text(obj["street_id"], phase), _text(obj["label"], phase),
            _enum(PreparedPlayer, obj["first_actor"], phase),
            _number(obj["min_open_bet"], phase),
        ))
    menu_values = []
    for index, source in enumerate(menus):
        phase = f"spec.decision_menus[{index}]"
        obj = _object(source, {"history", "street_id", "player", "actions"}, phase=phase)
        actions = _array(obj["actions"], 8, f"{phase}.actions")
        menu_values.append(PreparedDecisionMenu(
            _history(obj["history"], limits, f"{phase}.history"),
            _text(obj["street_id"], phase), _enum(PreparedPlayer, obj["player"], phase),
            tuple(action(child, f"{phase}.actions[{i}]") for i, child in enumerate(actions)),
        ))
    row_values = []
    total_edges = 0
    for index, source in enumerate(rows):
        phase = f"spec.transition_rows[{index}]"
        obj = _object(source, {
            "transition_id", "source_history", "hero_bucket_id", "villain_bucket_id", "edges"
        }, phase=phase)
        edges = _array(obj["edges"], 32, f"{phase}.edges")
        total_edges += len(edges)
        if total_edges > 100_000:
            raise _WorkflowFailure(PreparedFileWorkflowStatus.CAP_EXCEEDED, phase, "chance edge cap exceeded")
        edge_values = []
        for edge_index, source_edge in enumerate(edges):
            edge_phase = f"{phase}.edges[{edge_index}]"
            edge = _object(source_edge, {
                "public_outcome_id", "next_hero_bucket_id", "next_villain_bucket_id", "probability"
            }, phase=edge_phase)
            edge_values.append(PreparedChanceEdge(
                _text(edge["public_outcome_id"], edge_phase),
                _text(edge["next_hero_bucket_id"], edge_phase),
                _text(edge["next_villain_bucket_id"], edge_phase),
                _number(edge["probability"], edge_phase),
            ))
        row_values.append(PreparedTransitionRow(
            _text(obj["transition_id"], phase),
            _history(obj["source_history"], limits, f"{phase}.source_history"),
            _text(obj["hero_bucket_id"], phase), _text(obj["villain_bucket_id"], phase),
            tuple(edge_values),
        ))
    showdown_values = []
    for index, source in enumerate(showdowns):
        phase = f"spec.showdown_values[{index}]"
        obj = _object(source, {
            "history", "hero_bucket_id", "villain_bucket_id", "hero_pot_share"
        }, phase=phase)
        showdown_values.append(PreparedShowdownValue(
            _history(obj["history"], limits, f"{phase}.history"),
            _text(obj["hero_bucket_id"], phase), _text(obj["villain_bucket_id"], phase),
            _number(obj["hero_pot_share"], phase),
        ))
    spec = PreparedTwoStreetSpec(
        PREPARED_TWO_STREET_CONTRACT_VERSION,
        PreparedDataAttestation(
            _text(att["source"], "spec.attestation"),
            _text(att["bucket_semantics"], "spec.attestation"),
            _text(att["conditional_probability_semantics"], "spec.attestation"),
            _text(att["observation_mapping"], "spec.attestation"),
            _boolean(att["perfect_recall_attested"], "spec.attestation"),
        ),
        chips(item["starting_chips"], "spec.starting_chips"),
        chips(item["initial_committed"], "spec.initial_committed"),
        PreparedRake(_number(rake_obj["rate"], "spec.rake"), _nullable_number(rake_obj["cap"], "spec.rake")),
        tuple(street_values),
        tuple(bucket(child, f"spec.hero_buckets[{i}]") for i, child in enumerate(hero_buckets)),
        tuple(bucket(child, f"spec.villain_buckets[{i}]") for i, child in enumerate(villain_buckets)),
        tuple(menu_values),
        None if item["transition_id"] is None else _text(item["transition_id"], "spec.transition_id"),
        tuple(row_values), tuple(showdown_values),
    )
    try:
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        semantic = prepared_semantic_sha256(spec)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _WorkflowFailure(
            PreparedFileWorkflowStatus.INVALID_INPUT, "spec.identity", f"cannot canonicalize spec: {exc}"
        ) from exc
    identity = PreparedContentIdentity(hashlib.sha256(canonical).hexdigest(), semantic)
    return spec, canonical, identity


def _template(build: Any) -> tuple[list[dict[str, Any]], str]:
    profiles = []
    for observation in build.information_sets:
        key = observation.key
        profiles.append({
            "info_set_id": observation.info_set_id,
            "player": key.player.value,
            "street_id": key.street_id,
            "own_current_bucket_id": key.own_current_bucket_id,
            "own_bucket_history": list(key.own_bucket_history),
            "public_history_id": key.public_history_id,
            "own_action_history": list(key.own_action_history),
            "actions": [
                {"action_label": label, "probability": None}
                for label in observation.legal_action_labels
            ],
        })
    canonical = json.dumps(profiles, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return profiles, hashlib.sha256(canonical).hexdigest()


def _base_document(
    raw: bytes, operation: str, limits: PreparedFileWorkflowLimits
) -> tuple[dict[str, Any], PreparedTwoStreetSpec, bytes, PreparedContentIdentity, Any, list[dict[str, Any]], str]:
    _validate_limits(limits)
    document = _parse(raw, limits)
    required = {"format_version", "operation", "spec"}
    optional = set() if operation == "inspect" else {"template_identity", "hero_profile", "villain_profile"}
    doc = _object(document, required | optional, phase="document")
    if doc["format_version"] != PREPARED_TWO_STREET_FILE_FORMAT:
        raise _WorkflowFailure(PreparedFileWorkflowStatus.INVALID_INPUT, "document", "unsupported format_version")
    if doc["operation"] != operation:
        raise _WorkflowFailure(PreparedFileWorkflowStatus.INVALID_INPUT, "document", f"operation must be {operation}")
    spec, canonical, identity = _spec(doc["spec"], limits)
    built = build_prepared_two_street_game(spec, canonical, identity)
    if built.status is not PreparedTwoStreetStatus.SUCCESS or built.build is None:
        if built.status is PreparedTwoStreetStatus.CAP_EXCEEDED:
            status = PreparedFileWorkflowStatus.CAP_EXCEEDED
        elif built.status is PreparedTwoStreetStatus.NON_REPRODUCIBLE:
            status = PreparedFileWorkflowStatus.NON_REPRODUCIBLE
        else:
            status = PreparedFileWorkflowStatus.BUILD_FAILURE
        message = built.error.message if built.error is not None else "prepared builder failed"
        raise _WorkflowFailure(status, "builder", message, built.status.value)
    profiles, template_hash = _template(built.build)
    if len(profiles) > limits.max_output_records:
        raise _WorkflowFailure(PreparedFileWorkflowStatus.CAP_EXCEEDED, "template", "output record cap exceeded")
    return doc, spec, canonical, identity, built.build, profiles, template_hash


def _identity_payload(identity: PreparedContentIdentity, template_hash: str) -> dict[str, str]:
    return {
        "raw_sha256": identity.raw_sha256,
        "semantic_sha256": identity.semantic_sha256,
        "template_id": PREPARED_TWO_STREET_TEMPLATE_ID,
        "template_sha256": template_hash,
    }


def inspect_prepared_two_street_file(
    raw: bytes, limits: PreparedFileWorkflowLimits = _DEFAULT_LIMITS
) -> PreparedFileWorkflowResult:
    """Validate and build a generated, unfilled profile template."""
    try:
        _, _, _, identity, build, profiles, template_hash = _base_document(raw, "inspect", limits)
        return PreparedFileWorkflowResult(PreparedFileWorkflowStatus.SUCCESS, {
            "format_version": PREPARED_TWO_STREET_FILE_FORMAT,
            "operation": "inspect",
            "output_id": PREPARED_TWO_STREET_FILE_OUTPUT_ID,
            "identity": _identity_payload(identity, template_hash),
            "builder_status": PreparedTwoStreetStatus.SUCCESS.value,
            "counts": asdict(build.counts),
            "profile_template": profiles,
        }, None)
    except _WorkflowFailure as exc:
        return _failure(exc)
    except Exception:
        return _failure(_WorkflowFailure(
            PreparedFileWorkflowStatus.INTERNAL_FAILURE, "internal", "unexpected inspect failure"
        ))


def _profile(
    value: Any,
    player: PreparedPlayer,
    build: Any,
    phase: str,
) -> PreparedPlayerProfile:
    items = _array(value, 1_000, phase)
    expected = {
        observation.info_set_id: observation.legal_action_labels
        for observation in build.information_sets if observation.key.player is player
    }
    if len(items) != len(expected):
        raise _WorkflowFailure(PreparedFileWorkflowStatus.PROFILE_FAILURE, phase, "profile is incomplete")
    entries = []
    seen = set()
    for index, source in enumerate(items):
        item_phase = f"{phase}[{index}]"
        obj = _object(source, {"info_set_id", "actions"}, phase=item_phase)
        info_id = _text(obj["info_set_id"], item_phase)
        if info_id in seen or info_id not in expected:
            raise _WorkflowFailure(PreparedFileWorkflowStatus.PROFILE_FAILURE, item_phase, "duplicate or foreign info_set_id")
        seen.add(info_id)
        actions = _array(obj["actions"], 8, f"{item_phase}.actions")
        labels = expected[info_id]
        if len(actions) != len(labels):
            raise _WorkflowFailure(PreparedFileWorkflowStatus.PROFILE_FAILURE, item_phase, "action profile is incomplete")
        probabilities = []
        seen_labels = set()
        for action_index, source_action in enumerate(actions):
            action_phase = f"{item_phase}.actions[{action_index}]"
            action_obj = _object(source_action, {"action_label", "probability"}, phase=action_phase)
            label = _text(action_obj["action_label"], action_phase)
            if label in seen_labels or label not in labels:
                raise _WorkflowFailure(PreparedFileWorkflowStatus.PROFILE_FAILURE, action_phase, "duplicate or illegal action_label")
            seen_labels.add(label)
            probabilities.append(PreparedActionProbability(label, _number(action_obj["probability"], action_phase)))
        if seen_labels != set(labels):
            raise _WorkflowFailure(PreparedFileWorkflowStatus.PROFILE_FAILURE, item_phase, "action profile is incomplete")
        entries.append(PreparedProfileEntry(info_id, tuple(probabilities)))
    if seen != set(expected):
        raise _WorkflowFailure(PreparedFileWorkflowStatus.PROFILE_FAILURE, phase, "profile is incomplete")
    return PreparedPlayerProfile(tuple(entries))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {name: _json_safe(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def run_prepared_two_street_file(
    raw: bytes, limits: PreparedFileWorkflowLimits = _DEFAULT_LIMITS
) -> PreparedFileWorkflowResult:
    """Validate complete profiles and return a bounded M16 result summary."""
    try:
        doc, spec, canonical, identity, build, _, template_hash = _base_document(raw, "run", limits)
        supplied_identity = _object(doc["template_identity"], {
            "raw_sha256", "semantic_sha256", "template_id", "template_sha256"
        }, phase="template_identity")
        expected_identity = _identity_payload(identity, template_hash)
        if supplied_identity != expected_identity:
            raise _WorkflowFailure(
                PreparedFileWorkflowStatus.IDENTITY_MISMATCH,
                "template_identity",
                "template identity does not match the supplied spec",
            )
        hero = _profile(doc["hero_profile"], PreparedPlayer.HERO, build, "hero_profile")
        villain = None if doc["villain_profile"] is None else _profile(
            doc["villain_profile"], PreparedPlayer.VILLAIN, build, "villain_profile"
        )
        result = run_prepared_two_street_orchestration(PreparedTwoStreetOrchestrationRequest(
            spec=spec, raw_input_bytes=canonical, content_identity=identity,
            hero_profile=hero, villain_profile=villain,
        ))
        if result.status is not PreparedOrchestrationStatus.SUCCESS or result.run is None or result.identity is None:
            if result.status is PreparedOrchestrationStatus.CAP_EXCEEDED:
                status = PreparedFileWorkflowStatus.CAP_EXCEEDED
            elif result.status is PreparedOrchestrationStatus.NON_REPRODUCIBLE:
                status = PreparedFileWorkflowStatus.NON_REPRODUCIBLE
            else:
                status = PreparedFileWorkflowStatus.ORCHESTRATION_FAILURE
            message = result.error.message if result.error is not None else "prepared orchestration failed"
            raise _WorkflowFailure(status, "orchestration", message, result.status.value)
        evaluation = result.run.evaluation
        output = {
            "format_version": PREPARED_TWO_STREET_FILE_FORMAT,
            "operation": "run",
            "output_id": PREPARED_TWO_STREET_FILE_OUTPUT_ID,
            "identity": expected_identity,
            "orchestration_status": result.status.value,
            "builder_status": result.builder_status.value if result.builder_status else None,
            "evaluation_status": result.evaluation_status.value if result.evaluation_status else None,
            "counts": asdict(result.run.build.counts),
            "orchestration_identity": _json_safe(result.identity),
            "fixed_profile_value": _json_safe(evaluation.fixed_profile_value),
            "exact_response": _json_safe(evaluation.exact_response),
        }
        encoded = json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode("utf-8")) > limits.max_input_bytes * 4:
            raise _WorkflowFailure(PreparedFileWorkflowStatus.CAP_EXCEEDED, "output", "encoded output byte cap exceeded")
        return PreparedFileWorkflowResult(PreparedFileWorkflowStatus.SUCCESS, output, None)
    except _WorkflowFailure as exc:
        return _failure(exc)
    except Exception:
        return _failure(_WorkflowFailure(
            PreparedFileWorkflowStatus.INTERNAL_FAILURE, "internal", "unexpected run failure"
        ))


def prepared_file_workflow_json(result: PreparedFileWorkflowResult) -> str:
    """Serialize one workflow result as deterministic strict JSON."""
    if type(result) is not PreparedFileWorkflowResult:
        raise TypeError("result must be PreparedFileWorkflowResult")
    payload = {
        "status": result.status.value,
        "output": result.output,
        "error": None if result.error is None else asdict(result.error),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
