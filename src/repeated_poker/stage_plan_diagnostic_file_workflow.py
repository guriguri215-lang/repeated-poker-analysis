"""Strict two-phase file adapter for the bounded stage-plan diagnostic.

``inspect`` validates and content-binds one exact-rational stage-game fixture,
then returns unconfirmed human-evidence templates without running the analytic
diagnostic. ``run`` revalidates the inspection identity and complete,
human-authored model-class and perfect-recall attestations before calling the
existing public diagnostic exactly once. Controlled failures never expose
partial values, deviation rows, counts, or identities.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from enum import Enum
from fractions import Fraction
from typing import Any, Mapping

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
from .stage_plan_diagnostic import (
    COOPERATE,
    HERO,
    PLAYERS,
    PUBLIC_STATES,
    PUNISH,
    VILLAIN,
    DiagnosticStatus,
    ManualPerfectRecallAttestation,
    ModelClassAttestation,
    NumericErrorBound,
    PublicAction,
    PublicMonitoring,
    PublicSignal,
    RecallHistory,
    diagnose_stage_plan_deviations,
    tree_content_identity,
)


__all__ = [
    "STAGE_PLAN_DIAGNOSTIC_FILE_FORMAT",
    "STAGE_PLAN_DIAGNOSTIC_INSPECTION_ID",
    "StagePlanDiagnosticFileStatus",
    "StagePlanDiagnosticFileLimits",
    "StagePlanDiagnosticFileError",
    "StagePlanDiagnosticFileResult",
    "inspect_stage_plan_diagnostic_file",
    "run_stage_plan_diagnostic_file",
    "process_stage_plan_diagnostic_file",
    "stage_plan_diagnostic_file_json",
]


STAGE_PLAN_DIAGNOSTIC_FILE_FORMAT = "stage-plan-diagnostic-file-v1"
STAGE_PLAN_DIAGNOSTIC_INSPECTION_ID = (
    "stage-plan-diagnostic-inspection-sha256-v1"
)
_OUTPUT_ID = "stage-plan-diagnostic-file-output-v1"
_QUALIFIED_CLAIM = "bounded exhaustive one-period stage-plan deviation diagnostic"


class StagePlanDiagnosticFileStatus(str, Enum):
    """Stable outer status classes for the strict file adapter."""

    SUCCESS = "SUCCESS"
    PARSE_FAILURE = "PARSE_FAILURE"
    INVALID_INPUT = "INVALID_INPUT"
    CAP_EXCEEDED = "CAP_EXCEEDED"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    ATTESTATION_FAILURE = "ATTESTATION_FAILURE"
    DIAGNOSTIC_FAILURE = "DIAGNOSTIC_FAILURE"
    INTERNAL_FAILURE = "INTERNAL_FAILURE"


@dataclass(frozen=True)
class StagePlanDiagnosticFileLimits:
    """Caller-lowerable ceilings outside the existing M11 core cap."""

    max_input_bytes: int = 1_000_000
    max_json_depth: int = 64
    max_total_json_values: int = 100_000
    max_tree_depth: int = 64
    max_tree_nodes: int = 500
    max_tree_branches: int = 2_000
    max_public_action_nodes: int = 500
    max_terminal_observables: int = 500
    max_signals: int = 2_000
    max_signal_actions: int = 64
    max_transitions: int = 4_000
    max_profile_rows: int = 500
    max_profile_actions: int = 2_000
    max_attestation_records: int = 2_000
    max_attestation_text_length: int = 500
    max_plan_rows: int = 20_000
    max_output_records: int = 100_000
    max_output_bytes: int = 4_000_000


@dataclass(frozen=True)
class StagePlanDiagnosticFileError:
    """Bounded controlled-failure metadata with no partial result."""

    phase: str
    message: str
    nested_status: str | None


@dataclass(frozen=True)
class StagePlanDiagnosticFileResult:
    """Exclusive success-output or controlled-failure wrapper."""

    status: StagePlanDiagnosticFileStatus
    output: dict[str, Any] | None
    error: StagePlanDiagnosticFileError | None


@dataclass(frozen=True)
class _NumericSpec:
    delta: Fraction
    stage_payoff_bound: Fraction
    input_tolerance: Fraction
    epsilon_claim: Fraction
    error_bound: NumericErrorBound


@dataclass(frozen=True)
class _TreeMetadata:
    counts: dict[str, int]
    info_sets: Mapping[str, Mapping[str, tuple[str, ...]]]
    members: Mapping[str, Mapping[str, tuple[str, ...]]]
    histories: Mapping[str, Mapping[str, Mapping[str, RecallHistory]]]
    text_bytes: int


@dataclass(frozen=True)
class _ParsedSpec:
    request_id: str
    fixture_version: str
    tree: GameTree
    monitoring: PublicMonitoring
    profile: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Fraction]]]]
    numeric: _NumericSpec
    max_plans_per_player: int
    workflow_limits: StagePlanDiagnosticFileLimits
    tree_identity: str
    inspection_identity: dict[str, str]
    metadata: _TreeMetadata
    counts: dict[str, Any]


_DEFAULT_LIMITS = StagePlanDiagnosticFileLimits()
_LIMIT_CEILINGS = asdict(_DEFAULT_LIMITS)
_WORKFLOW_LIMIT_KEYS = set(_LIMIT_CEILINGS)
_BASE_KEYS = {
    "format_version",
    "operation",
    "request_id",
    "fixture_version",
    "tree",
    "monitoring",
    "profiles",
    "numeric",
    "core_limits",
    "workflow_limits",
}
_RUN_KEYS = _BASE_KEYS | {
    "inspection_identity",
    "model_attestation",
    "perfect_recall_attestation",
}
_CORE_LIMIT_KEYS = {"max_plans_per_player"}
_NUMERIC_KEYS = {
    "delta",
    "stage_payoff_bound",
    "input_tolerance",
    "epsilon_claim",
    "numeric_error_bound",
}
_ERROR_BOUND_KEYS = {
    "probability_representation",
    "probability_normalization",
    "stage_expectation",
    "continuation_expectation",
    "subtraction",
    "maximum",
    "bellman_residual",
    "residual_evaluation",
    "enclosure_established",
}
_TREE_KEYS = {"root"}
_TERMINAL_KEYS = {
    "type",
    "node_id",
    "hero_payoff",
    "villain_payoff",
    "house_residual",
}
_CHANCE_KEYS = {"type", "node_id", "children"}
_DECISION_KEYS = {"type", "node_id", "info_set", "actions"}
_CHILD_KEYS = {"probability", "child"}
_ACTION_KEYS = {"action", "child"}
_MONITORING_KEYS = {
    "public_action_node_ids",
    "terminal_observables",
    "signal_alphabet",
    "transitions",
}
_OBSERVABLE_KEYS = {"node_id", "observable"}
_SIGNAL_KEYS = {"action_trace", "terminal_observable"}
_PUBLIC_ACTION_KEYS = {"actor", "action"}
_TRANSITION_KEYS = {"state", "signal_index", "next_state"}
_PROFILE_STATE_KEYS = set(PUBLIC_STATES)
_PROFILE_PLAYER_KEYS = set(PLAYERS)
_PROFILE_ROW_KEYS = {"info_set", "actions"}
_PROFILE_ACTION_KEYS = {"action", "probability"}
_INSPECTION_IDENTITY_KEYS = {
    "template_id",
    "semantic_sha256",
    "tree_content_identity",
}
_MODEL_FIELDS = tuple(field.name for field in fields(ModelClassAttestation))
_MODEL_KEYS = set(_MODEL_FIELDS)
_RECALL_FIELDS = tuple(field.name for field in fields(ManualPerfectRecallAttestation))
_RECALL_KEYS = set(_RECALL_FIELDS)
_PLAYER_EVIDENCE_KEYS = set(PLAYERS)
_INFO_SET_MEMBERS_KEYS = {"info_set", "members"}
_HISTORY_GROUP_KEYS = {"info_set", "members"}
_HISTORY_MEMBER_KEYS = {
    "node_id",
    "observations",
    "own_actions",
    "information_sets",
}
_LEGAL_ACTIONS_KEYS = {"info_set", "actions"}


class _WorkflowFailure(ValueError):
    def __init__(
        self,
        status: StagePlanDiagnosticFileStatus,
        phase: str,
        message: str,
        nested_status: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.phase = phase
        self.nested_status = nested_status


def _clean_text(value: str, maximum: int) -> str:
    return value.replace("\r", " ").replace("\n", " ")[:maximum]


def _failure(exc: _WorkflowFailure) -> StagePlanDiagnosticFileResult:
    return StagePlanDiagnosticFileResult(
        exc.status,
        None,
        StagePlanDiagnosticFileError(
            _clean_text(exc.phase, 64),
            _clean_text(str(exc), 500),
            exc.nested_status,
        ),
    )


def _duplicates_rejected(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _WorkflowFailure(
                StagePlanDiagnosticFileStatus.PARSE_FAILURE,
                "json",
                f"duplicate JSON key: {key}",
            )
        result[key] = value
    return result


def _validate_outer_limits(limits: StagePlanDiagnosticFileLimits) -> None:
    if type(limits) is not StagePlanDiagnosticFileLimits:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            "limits",
            "workflow limits have the wrong type",
        )
    for name, ceiling in _LIMIT_CEILINGS.items():
        value = getattr(limits, name)
        if type(value) is not int or value <= 0 or value > ceiling:
            raise _WorkflowFailure(
                StagePlanDiagnosticFileStatus.INVALID_INPUT,
                "limits",
                f"{name} must be a positive int no greater than {ceiling}",
            )


def _measure_json(value: Any, limits: StagePlanDiagnosticFileLimits) -> None:
    count = 0
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        count += 1
        if count > limits.max_total_json_values:
            raise _WorkflowFailure(
                StagePlanDiagnosticFileStatus.CAP_EXCEEDED,
                "json",
                "total JSON value cap exceeded",
            )
        if depth > limits.max_json_depth:
            raise _WorkflowFailure(
                StagePlanDiagnosticFileStatus.CAP_EXCEEDED,
                "json",
                "JSON depth cap exceeded",
            )
        if type(item) is dict:
            stack.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            stack.extend((child, depth + 1) for child in item)


def _parse(raw: bytes, limits: StagePlanDiagnosticFileLimits) -> dict[str, Any]:
    _validate_outer_limits(limits)
    if type(raw) is not bytes:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            "input",
            "input must be bytes",
        )
    if len(raw) > limits.max_input_bytes:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.CAP_EXCEEDED,
            "input",
            "input byte cap exceeded",
        )
    if raw.startswith(b"\xef\xbb\xbf"):
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.PARSE_FAILURE,
            "json",
            "UTF-8 BOM is not allowed",
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.PARSE_FAILURE,
            "json",
            "input is not UTF-8",
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
            StagePlanDiagnosticFileStatus.CAP_EXCEEDED,
            "json",
            "JSON nesting exceeds the parser depth cap",
        ) from exc
    except (TypeError, ValueError) as exc:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.PARSE_FAILURE,
            "json",
            f"invalid JSON: {exc}",
        ) from exc
    if type(value) is not dict:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            "document",
            "top-level JSON value must be an object",
        )
    _measure_json(value, limits)
    return value


def _object(value: Any, keys: set[str], phase: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            phase,
            "value must be an object",
        )
    actual = set(value)
    missing = keys - actual
    unknown = actual - keys
    if missing:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            phase,
            f"missing keys: {', '.join(sorted(missing))}",
        )
    if unknown:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            phase,
            f"unknown keys: {', '.join(sorted(unknown))}",
        )
    return value


def _array(value: Any, cap: int, phase: str) -> list[Any]:
    if type(value) is not list:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            phase,
            "value must be an array",
        )
    if len(value) > cap:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.CAP_EXCEEDED,
            phase,
            "array cap exceeded",
        )
    return value


def _text(value: Any, phase: str, maximum: int = 128) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > maximum
        or any(ord(character) < 0x20 for character in value)
    ):
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            phase,
            f"value must be a non-empty control-free string of at most {maximum} characters",
        )
    return value


def _plain_int(value: Any, phase: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            phase,
            f"value must be an integer in [{minimum}, {maximum}]",
        )
    return value


def _boolean(value: Any, phase: str) -> bool:
    if type(value) is not bool:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            phase,
            "value must be a boolean",
        )
    return value


def _canonical_fraction(
    value: Any,
    phase: str,
    *,
    non_negative: bool = False,
) -> Fraction:
    text = _text(value, phase, 200)
    try:
        fraction = Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            phase,
            "value must be a canonical rational string",
        ) from exc
    if str(fraction) != text or (non_negative and fraction < 0):
        qualifier = " non-negative" if non_negative else ""
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            phase,
            f"value must be a canonical{qualifier} rational string",
        )
    return fraction


def _workflow_limits(
    value: Any,
    envelope: StagePlanDiagnosticFileLimits,
) -> StagePlanDiagnosticFileLimits:
    document = _object(value, _WORKFLOW_LIMIT_KEYS, "workflow_limits")
    values: dict[str, int] = {}
    for name, ceiling in _LIMIT_CEILINGS.items():
        effective_ceiling = min(ceiling, getattr(envelope, name))
        values[name] = _plain_int(
            document[name],
            f"workflow_limits.{name}",
            1,
            effective_ceiling,
        )
    return StagePlanDiagnosticFileLimits(**values)


def _numeric(value: Any) -> _NumericSpec:
    document = _object(value, _NUMERIC_KEYS, "numeric")
    delta = _canonical_fraction(document["delta"], "numeric.delta", non_negative=True)
    payoff_bound = _canonical_fraction(
        document["stage_payoff_bound"],
        "numeric.stage_payoff_bound",
        non_negative=True,
    )
    tolerance = _canonical_fraction(
        document["input_tolerance"],
        "numeric.input_tolerance",
        non_negative=True,
    )
    claim = _canonical_fraction(
        document["epsilon_claim"],
        "numeric.epsilon_claim",
        non_negative=True,
    )
    if not 0 < delta < 1:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            "numeric.delta",
            "delta must satisfy 0 < delta < 1",
        )
    bound_document = _object(
        document["numeric_error_bound"],
        _ERROR_BOUND_KEYS,
        "numeric.numeric_error_bound",
    )
    components = {
        name: _canonical_fraction(
            bound_document[name],
            f"numeric.numeric_error_bound.{name}",
            non_negative=True,
        )
        for name in _ERROR_BOUND_KEYS - {"enclosure_established"}
    }
    error_bound = NumericErrorBound(
        **components,
        enclosure_established=_boolean(
            bound_document["enclosure_established"],
            "numeric.numeric_error_bound.enclosure_established",
        ),
    )
    return _NumericSpec(delta, payoff_bound, tolerance, claim, error_bound)


def _preflight_tree(
    value: Any,
    limits: StagePlanDiagnosticFileLimits,
    payoff_bound: Fraction,
) -> dict[str, Any]:
    document = _object(value, _TREE_KEYS, "tree")
    node_ids: set[str] = set()
    decision_ids: set[str] = set()
    terminal_ids: set[str] = set()
    node_count = branch_count = terminal_count = chance_count = 0
    hero_nodes = villain_nodes = text_bytes = 0
    stack: list[tuple[Any, int, str]] = [(document["root"], 1, "tree.root")]
    while stack:
        source, depth, phase = stack.pop()
        if depth > limits.max_tree_depth:
            raise _WorkflowFailure(
                StagePlanDiagnosticFileStatus.CAP_EXCEEDED,
                phase,
                "tree depth cap exceeded",
            )
        if type(source) is not dict:
            raise _WorkflowFailure(
                StagePlanDiagnosticFileStatus.INVALID_INPUT,
                phase,
                "tree node must be an object",
            )
        node_type = source.get("type")
        keys = (
            _TERMINAL_KEYS
            if node_type == "terminal"
            else _CHANCE_KEYS
            if node_type == "chance"
            else _DECISION_KEYS
            if node_type in ("hero", "villain")
            else set()
        )
        if not keys:
            raise _WorkflowFailure(
                StagePlanDiagnosticFileStatus.INVALID_INPUT,
                f"{phase}.type",
                "unsupported tree node type",
            )
        node = _object(source, keys, phase)
        node_id = _text(node["node_id"], f"{phase}.node_id")
        text_bytes += len(node_id.encode("utf-8"))
        if node_id in node_ids:
            raise _WorkflowFailure(
                StagePlanDiagnosticFileStatus.INVALID_INPUT,
                f"{phase}.node_id",
                "tree node IDs must be unique",
            )
        node_ids.add(node_id)
        node_count += 1
        if node_count > limits.max_tree_nodes:
            raise _WorkflowFailure(
                StagePlanDiagnosticFileStatus.CAP_EXCEEDED,
                phase,
                "tree node cap exceeded",
            )
        if node_type == "terminal":
            hero = _canonical_fraction(node["hero_payoff"], f"{phase}.hero_payoff")
            villain = _canonical_fraction(
                node["villain_payoff"], f"{phase}.villain_payoff"
            )
            residual = _canonical_fraction(
                node["house_residual"], f"{phase}.house_residual"
            )
            if abs(hero) > payoff_bound or abs(villain) > payoff_bound:
                raise _WorkflowFailure(
                    StagePlanDiagnosticFileStatus.INVALID_INPUT,
                    phase,
                    "terminal player payoff exceeds stage_payoff_bound",
                )
            if hero + villain + residual != 0:
                raise _WorkflowFailure(
                    StagePlanDiagnosticFileStatus.INVALID_INPUT,
                    phase,
                    "terminal violates the accounting identity",
                )
            terminal_count += 1
            terminal_ids.add(node_id)
            continue
        if node_type == "chance":
            children = _array(node["children"], limits.max_tree_branches, f"{phase}.children")
            if not children:
                raise _WorkflowFailure(
                    StagePlanDiagnosticFileStatus.INVALID_INPUT,
                    f"{phase}.children",
                    "chance node must have children",
                )
            probabilities: list[Fraction] = []
            chance_count += 1
            for index, raw_child in enumerate(children):
                child_phase = f"{phase}.children[{index}]"
                child = _object(raw_child, _CHILD_KEYS, child_phase)
                probability = _canonical_fraction(
                    child["probability"],
                    f"{child_phase}.probability",
                    non_negative=True,
                )
                probabilities.append(probability)
                stack.append((child["child"], depth + 1, f"{child_phase}.child"))
            if sum(probabilities, Fraction(0)) != 1:
                raise _WorkflowFailure(
                    StagePlanDiagnosticFileStatus.INVALID_INPUT,
                    f"{phase}.children",
                    "chance probabilities must sum exactly to one",
                )
            branch_count += len(children)
        else:
            info_set = _text(node["info_set"], f"{phase}.info_set")
            text_bytes += len(info_set.encode("utf-8"))
            actions = _array(node["actions"], limits.max_tree_branches, f"{phase}.actions")
            if not actions:
                raise _WorkflowFailure(
                    StagePlanDiagnosticFileStatus.INVALID_INPUT,
                    f"{phase}.actions",
                    "decision node must have actions",
                )
            labels: set[str] = set()
            for index, raw_action in enumerate(actions):
                action_phase = f"{phase}.actions[{index}]"
                action = _object(raw_action, _ACTION_KEYS, action_phase)
                label = _text(action["action"], f"{action_phase}.action")
                text_bytes += len(label.encode("utf-8"))
                if label in labels:
                    raise _WorkflowFailure(
                        StagePlanDiagnosticFileStatus.INVALID_INPUT,
                        f"{action_phase}.action",
                        "decision actions must be unique",
                    )
                labels.add(label)
                stack.append((action["child"], depth + 1, f"{action_phase}.child"))
            decision_ids.add(node_id)
            if node_type == "hero":
                hero_nodes += 1
            else:
                villain_nodes += 1
            branch_count += len(actions)
        if branch_count > limits.max_tree_branches:
            raise _WorkflowFailure(
                StagePlanDiagnosticFileStatus.CAP_EXCEEDED,
                phase,
                "tree branch cap exceeded",
            )
    if terminal_count == 0:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            "tree",
            "tree must contain a terminal",
        )
    return {
        "document": document,
        "node_ids": node_ids,
        "decision_ids": decision_ids,
        "terminal_ids": terminal_ids,
        "counts": {
            "nodes": node_count,
            "terminals": terminal_count,
            "chance_nodes": chance_count,
            "hero_decision_nodes": hero_nodes,
            "villain_decision_nodes": villain_nodes,
            "branches": branch_count,
        },
        "text_bytes": text_bytes,
    }


def _build_node(source: dict[str, Any]) -> Any:
    node_type = source["type"]
    if node_type == "terminal":
        return TerminalNode(
            source["node_id"],
            _canonical_fraction(source["hero_payoff"], "tree.hero_payoff"),
            _canonical_fraction(source["villain_payoff"], "tree.villain_payoff"),
            _canonical_fraction(source["house_residual"], "tree.house_residual"),
        )
    if node_type == "chance":
        return ChanceNode(
            source["node_id"],
            tuple(
                (
                    _canonical_fraction(
                        child["probability"],
                        "tree.chance_probability",
                        non_negative=True,
                    ),
                    _build_node(child["child"]),
                )
                for child in source["children"]
            ),
        )
    node_type_class = HeroNode if node_type == "hero" else VillainNode
    return node_type_class(
        source["node_id"],
        source["info_set"],
        tuple((action["action"], _build_node(action["child"])) for action in source["actions"]),
    )


def _tree_metadata(tree: GameTree, preflight: dict[str, Any]) -> _TreeMetadata:
    info_sets = {
        HERO: collect_hero_info_sets(tree),
        VILLAIN: collect_villain_info_sets(tree),
    }
    members: dict[str, dict[str, list[str]]] = {HERO: {}, VILLAIN: {}}
    histories: dict[str, dict[str, dict[str, RecallHistory]]] = {HERO: {}, VILLAIN: {}}

    def walk(
        node: Any,
        hero_info: tuple[str, ...],
        hero_actions: tuple[str, ...],
        villain_info: tuple[str, ...],
        villain_actions: tuple[str, ...],
    ) -> None:
        if isinstance(node, TerminalNode):
            return
        if isinstance(node, ChanceNode):
            for _, child in node.children:
                walk(child, hero_info, hero_actions, villain_info, villain_actions)
            return
        if isinstance(node, HeroNode):
            members[HERO].setdefault(node.info_set, []).append(node.node_id)
            histories[HERO].setdefault(node.info_set, {})[node.node_id] = RecallHistory(
                (), hero_actions, hero_info
            )
            for action, child in node.actions:
                walk(
                    child,
                    hero_info + (node.info_set,),
                    hero_actions + (action,),
                    villain_info,
                    villain_actions,
                )
            return
        members[VILLAIN].setdefault(node.info_set, []).append(node.node_id)
        histories[VILLAIN].setdefault(node.info_set, {})[node.node_id] = RecallHistory(
            (), villain_actions, villain_info
        )
        for action, child in node.actions:
            walk(
                child,
                hero_info,
                hero_actions,
                villain_info + (node.info_set,),
                villain_actions + (action,),
            )

    walk(tree.root, (), (), (), ())
    frozen_members = {
        player: {info: tuple(nodes) for info, nodes in members[player].items()}
        for player in PLAYERS
    }
    counts = dict(preflight["counts"])
    counts.update(
        {
            "hero_information_sets": len(info_sets[HERO]),
            "villain_information_sets": len(info_sets[VILLAIN]),
        }
    )
    return _TreeMetadata(
        counts,
        info_sets,
        frozen_members,
        histories,
        preflight["text_bytes"],
    )


def _monitoring(
    value: Any,
    tree: GameTree,
    preflight: dict[str, Any],
    limits: StagePlanDiagnosticFileLimits,
) -> tuple[PublicMonitoring, dict[str, int], int]:
    document = _object(value, _MONITORING_KEYS, "monitoring")
    action_ids_raw = _array(
        document["public_action_node_ids"],
        limits.max_public_action_nodes,
        "monitoring.public_action_node_ids",
    )
    action_ids = tuple(
        _text(item, f"monitoring.public_action_node_ids[{index}]")
        for index, item in enumerate(action_ids_raw)
    )
    if len(action_ids) != len(set(action_ids)):
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            "monitoring.public_action_node_ids",
            "public action node IDs must be unique",
        )
    if not set(action_ids) <= preflight["decision_ids"]:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            "monitoring.public_action_node_ids",
            "public action node IDs must reference decision nodes",
        )
    observable_rows = _array(
        document["terminal_observables"],
        limits.max_terminal_observables,
        "monitoring.terminal_observables",
    )
    observables: dict[str, str] = {}
    text_bytes = sum(len(item.encode("utf-8")) for item in action_ids)
    for index, item in enumerate(observable_rows):
        phase = f"monitoring.terminal_observables[{index}]"
        row = _object(item, _OBSERVABLE_KEYS, phase)
        node_id = _text(row["node_id"], f"{phase}.node_id")
        observable = _text(row["observable"], f"{phase}.observable")
        if node_id in observables:
            raise _WorkflowFailure(
                StagePlanDiagnosticFileStatus.INVALID_INPUT,
                phase,
                "terminal observable node IDs must be unique",
            )
        observables[node_id] = observable
        text_bytes += len(node_id.encode("utf-8")) + len(observable.encode("utf-8"))
    if set(observables) != preflight["terminal_ids"]:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            "monitoring.terminal_observables",
            "terminal observables must specify every and only terminal node",
        )
    signal_rows = _array(
        document["signal_alphabet"], limits.max_signals, "monitoring.signal_alphabet"
    )
    signals: list[PublicSignal] = []
    signal_action_count = 0
    for index, item in enumerate(signal_rows):
        phase = f"monitoring.signal_alphabet[{index}]"
        row = _object(item, _SIGNAL_KEYS, phase)
        trace_rows = _array(
            row["action_trace"], limits.max_signal_actions, f"{phase}.action_trace"
        )
        signal_action_count += len(trace_rows)
        if signal_action_count > limits.max_signal_actions * limits.max_signals:
            raise _WorkflowFailure(
                StagePlanDiagnosticFileStatus.CAP_EXCEEDED,
                "monitoring.signal_alphabet",
                "public signal action cap exceeded",
            )
        trace: list[PublicAction] = []
        for action_index, raw_action in enumerate(trace_rows):
            action_phase = f"{phase}.action_trace[{action_index}]"
            action = _object(raw_action, _PUBLIC_ACTION_KEYS, action_phase)
            actor = _text(action["actor"], f"{action_phase}.actor")
            label = _text(action["action"], f"{action_phase}.action")
            if actor not in PLAYERS:
                raise _WorkflowFailure(
                    StagePlanDiagnosticFileStatus.INVALID_INPUT,
                    f"{action_phase}.actor",
                    "public action actor must be Hero or Villain",
                )
            trace.append(PublicAction(actor, label))
            text_bytes += len(actor.encode("utf-8")) + len(label.encode("utf-8"))
        terminal_observable = _text(
            row["terminal_observable"], f"{phase}.terminal_observable"
        )
        text_bytes += len(terminal_observable.encode("utf-8"))
        signals.append(PublicSignal(tuple(trace), terminal_observable))
    if not signals or len(signals) != len(set(signals)):
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            "monitoring.signal_alphabet",
            "signal alphabet must be non-empty and unique",
        )
    transition_rows = _array(
        document["transitions"], limits.max_transitions, "monitoring.transitions"
    )
    transitions: dict[tuple[str, PublicSignal], str] = {}
    for index, item in enumerate(transition_rows):
        phase = f"monitoring.transitions[{index}]"
        row = _object(item, _TRANSITION_KEYS, phase)
        state = _text(row["state"], f"{phase}.state")
        next_state = _text(row["next_state"], f"{phase}.next_state")
        signal_index = _plain_int(
            row["signal_index"], f"{phase}.signal_index", 0, len(signals) - 1
        )
        if state not in PUBLIC_STATES or next_state not in PUBLIC_STATES:
            raise _WorkflowFailure(
                StagePlanDiagnosticFileStatus.INVALID_INPUT,
                phase,
                "transition states must be C or P",
            )
        key = (state, signals[signal_index])
        if key in transitions:
            raise _WorkflowFailure(
                StagePlanDiagnosticFileStatus.INVALID_INPUT,
                phase,
                "transition keys must be unique",
            )
        transitions[key] = next_state
    expected = {(state, signal) for state in PUBLIC_STATES for signal in signals}
    if set(transitions) != expected:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            "monitoring.transitions",
            "transitions must be total on C/P times the signal alphabet",
        )
    if any(transitions[(PUNISH, signal)] != PUNISH for signal in signals):
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            "monitoring.transitions",
            "P must transition to P for every signal",
        )
    monitoring = PublicMonitoring(
        frozenset(action_ids), observables, tuple(signals), transitions
    )

    feasible: list[PublicSignal] = []

    def walk(node: Any, trace: tuple[PublicAction, ...]) -> None:
        if isinstance(node, TerminalNode):
            feasible.append(PublicSignal(trace, observables[node.node_id]))
            return
        if isinstance(node, ChanceNode):
            for _, child in node.children:
                walk(child, trace)
            return
        actor = HERO if isinstance(node, HeroNode) else VILLAIN
        for action, child in node.actions:
            next_trace = (
                trace + (PublicAction(actor, action),)
                if node.node_id in monitoring.public_action_node_ids
                else trace
            )
            walk(child, next_trace)

    walk(tree.root, ())
    if set(signals) != set(feasible):
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            "monitoring.signal_alphabet",
            "signal alphabet must equal the complete feasible public signal set",
        )
    return monitoring, {
        "public_action_nodes": len(action_ids),
        "terminal_observables": len(observables),
        "public_signals": len(signals),
        "public_signal_actions": signal_action_count,
        "transitions": len(transitions),
    }, text_bytes


def _profile_rows(
    value: Any,
    expected: Mapping[str, tuple[str, ...]],
    phase: str,
    limits: StagePlanDiagnosticFileLimits,
) -> tuple[dict[str, dict[str, Fraction]], int, int]:
    rows = _array(value, limits.max_profile_rows, phase)
    if len(rows) != len(expected):
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            phase,
            "profile information-set support is incomplete",
        )
    result: dict[str, dict[str, Fraction]] = {}
    action_total = text_bytes = 0
    for index, item in enumerate(rows):
        item_phase = f"{phase}[{index}]"
        row = _object(item, _PROFILE_ROW_KEYS, item_phase)
        info_set = _text(row["info_set"], f"{item_phase}.info_set")
        if info_set in result or info_set not in expected:
            raise _WorkflowFailure(
                StagePlanDiagnosticFileStatus.INVALID_INPUT,
                f"{item_phase}.info_set",
                "profile information set is duplicate or unknown",
            )
        action_rows = _array(
            row["actions"], limits.max_profile_actions, f"{item_phase}.actions"
        )
        action_total += len(action_rows)
        if action_total > limits.max_profile_actions:
            raise _WorkflowFailure(
                StagePlanDiagnosticFileStatus.CAP_EXCEEDED,
                phase,
                "profile action cap exceeded",
            )
        probabilities: dict[str, Fraction] = {}
        for action_index, raw_action in enumerate(action_rows):
            action_phase = f"{item_phase}.actions[{action_index}]"
            action = _object(raw_action, _PROFILE_ACTION_KEYS, action_phase)
            label = _text(action["action"], f"{action_phase}.action")
            if label in probabilities:
                raise _WorkflowFailure(
                    StagePlanDiagnosticFileStatus.INVALID_INPUT,
                    action_phase,
                    "profile actions must be unique",
                )
            probabilities[label] = _canonical_fraction(
                action["probability"],
                f"{action_phase}.probability",
                non_negative=True,
            )
            text_bytes += len(label.encode("utf-8"))
        if set(probabilities) != set(expected[info_set]):
            raise _WorkflowFailure(
                StagePlanDiagnosticFileStatus.INVALID_INPUT,
                item_phase,
                "profile actions must equal the legal action set",
            )
        if sum(probabilities.values(), Fraction(0)) != 1:
            raise _WorkflowFailure(
                StagePlanDiagnosticFileStatus.INVALID_INPUT,
                item_phase,
                "profile probabilities must sum exactly to one",
            )
        result[info_set] = {
            action: probabilities[action] for action in expected[info_set]
        }
        text_bytes += len(info_set.encode("utf-8"))
    if set(result) != set(expected):
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            phase,
            "profile information-set support is incomplete",
        )
    return result, len(rows), action_total + text_bytes


def _profiles(
    value: Any,
    metadata: _TreeMetadata,
    limits: StagePlanDiagnosticFileLimits,
) -> tuple[Mapping[str, Mapping[str, Mapping[str, Mapping[str, Fraction]]]], dict[str, int], int]:
    document = _object(value, _PROFILE_STATE_KEYS, "profiles")
    result: dict[str, dict[str, dict[str, dict[str, Fraction]]]] = {}
    row_count = action_count = text_bytes = 0
    for state in PUBLIC_STATES:
        state_doc = _object(document[state], _PROFILE_PLAYER_KEYS, f"profiles.{state}")
        result[state] = {}
        for player in PLAYERS:
            parsed, rows, aggregate = _profile_rows(
                state_doc[player],
                metadata.info_sets[player],
                f"profiles.{state}.{player}",
                limits,
            )
            result[state][player] = parsed
            row_count += rows
            action_count += sum(len(actions) for actions in parsed.values())
            text_bytes += aggregate
    if row_count > limits.max_profile_rows or action_count > limits.max_profile_actions:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.CAP_EXCEEDED,
            "profiles",
            "complete profile cap exceeded",
        )
    return result, {"rows": row_count, "actions": action_count}, text_bytes


def _plan_counts(metadata: _TreeMetadata) -> dict[str, int]:
    result: dict[str, int] = {}
    for player in PLAYERS:
        count = 1
        for actions in metadata.info_sets[player].values():
            count *= len(actions)
        result[player] = count
    return result


def _fraction_text(value: Any) -> str:
    if type(value) is not Fraction:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INTERNAL_FAILURE,
            "output",
            "diagnostic returned a non-rational value",
        )
    return str(value)


def _identity_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, tuple):
        return [_identity_value(item) for item in value]
    if isinstance(value, frozenset):
        return sorted(_identity_value(item) for item in value)
    if isinstance(value, list):
        return [_identity_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _identity_value(value[key]) for key in sorted(value, key=str)}
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _identity_value(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    return value


def _sha256(value: Any) -> str:
    encoded = json.dumps(
        _identity_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _inspection_identity(
    fixture_version: str,
    tree_identity: str,
    monitoring: PublicMonitoring,
    profile: Mapping[str, Any],
    numeric: _NumericSpec,
    max_plans_per_player: int,
    limits: StagePlanDiagnosticFileLimits,
) -> dict[str, str]:
    transitions = [
        {
            "state": state,
            "signal_index": monitoring.signal_alphabet.index(signal),
            "next_state": monitoring.transitions[(state, signal)],
        }
        for state in PUBLIC_STATES
        for signal in monitoring.signal_alphabet
    ]
    semantic = {
        "format_version": STAGE_PLAN_DIAGNOSTIC_FILE_FORMAT,
        "fixture_version": fixture_version,
        "tree_content_identity": tree_identity,
        "monitoring": {
            "public_action_node_ids": sorted(monitoring.public_action_node_ids),
            "terminal_observables": dict(sorted(monitoring.terminal_observables.items())),
            "signal_alphabet": monitoring.signal_alphabet,
            "transitions": transitions,
        },
        "profiles": profile,
        "numeric": numeric,
        "core_limits": {"max_plans_per_player": max_plans_per_player},
        "workflow_limits": limits,
    }
    return {
        "template_id": STAGE_PLAN_DIAGNOSTIC_INSPECTION_ID,
        "semantic_sha256": _sha256(semantic),
        "tree_content_identity": tree_identity,
    }


def _model_template() -> dict[str, None]:
    return {name: None for name in _MODEL_FIELDS}


def _player_members_template(metadata: _TreeMetadata) -> dict[str, list[dict[str, Any]]]:
    return {
        player: [
            {"info_set": info_set, "members": list(metadata.members[player][info_set])}
            for info_set in metadata.info_sets[player]
        ]
        for player in PLAYERS
    }


def _history_template(metadata: _TreeMetadata) -> dict[str, list[dict[str, Any]]]:
    return {
        player: [
            {
                "info_set": info_set,
                "members": [
                    {
                        "node_id": node_id,
                        "observations": None,
                        "own_actions": list(
                            metadata.histories[player][info_set][node_id].own_actions
                        ),
                        "information_sets": list(
                            metadata.histories[player][info_set][node_id].information_sets
                        ),
                    }
                    for node_id in metadata.members[player][info_set]
                ],
            }
            for info_set in metadata.info_sets[player]
        ]
        for player in PLAYERS
    }


def _legal_actions_template(metadata: _TreeMetadata) -> dict[str, list[dict[str, Any]]]:
    return {
        player: [
            {"info_set": info_set, "actions": list(actions)}
            for info_set, actions in metadata.info_sets[player].items()
        ]
        for player in PLAYERS
    }


def _recall_template(spec: _ParsedSpec) -> dict[str, Any]:
    return {
        "fixture_id": None,
        "tree_content_identity": spec.tree_identity,
        "target_version": spec.fixture_version,
        "information_set_members": _player_members_template(spec.metadata),
        "member_histories": _history_template(spec.metadata),
        "legal_actions": _legal_actions_template(spec.metadata),
        "reviewer": None,
        "review_date": None,
        "review_method": None,
        "evidence": None,
        "result_confirmed": None,
        "known_limitations": None,
        "invalidation_conditions": None,
        "valid_through_version": None,
        "invalidated": None,
    }


def _output_preflight(spec: _ParsedSpec, operation: str) -> None:
    plans = spec.counts["plans"]
    deviation_rows = spec.counts["predicted_deviation_rows"]
    info_count = (
        spec.metadata.counts["hero_information_sets"]
        + spec.metadata.counts["villain_information_sets"]
    )
    template_records = (
        256
        + spec.metadata.counts["nodes"] * 8
        + spec.counts["monitoring"]["public_signal_actions"] * 4
        + spec.counts["profiles"]["actions"] * 4
        + info_count * 16
    )
    projected_records = (
        template_records
        if operation == "inspect"
        else 512 + deviation_rows * (16 + 4 * max(1, info_count))
    )
    if projected_records > spec.workflow_limits.max_output_records:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.CAP_EXCEEDED,
            "output",
            "output record cap exceeded",
        )
    projected_bytes = (
        8_192
        + 24 * spec.metadata.text_bytes
        + 1_024 * info_count
        + 512 * spec.counts["profiles"]["actions"]
    )
    if operation == "run":
        projected_bytes += deviation_rows * (1_536 + 256 * max(1, info_count))
    if projected_bytes > spec.workflow_limits.max_output_bytes:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.CAP_EXCEEDED,
            "output",
            "output byte cap exceeded",
        )
    if max(plans.values()) > spec.max_plans_per_player:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.CAP_EXCEEDED,
            "plans",
            "complete stage-plan count exceeds max_plans_per_player",
            DiagnosticStatus.UNSUPPORTED.value,
        )
    if deviation_rows > spec.workflow_limits.max_plan_rows:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.CAP_EXCEEDED,
            "plans",
            "complete deviation row cap exceeded",
        )


def _base_document(
    raw: bytes,
    expected_operation: str | None,
    envelope: StagePlanDiagnosticFileLimits,
) -> tuple[dict[str, Any], _ParsedSpec]:
    document = _parse(raw, envelope)
    operation = document.get("operation")
    if operation not in ("inspect", "run"):
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            "document.operation",
            "operation must be inspect or run",
        )
    if expected_operation is not None and operation != expected_operation:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            "document.operation",
            f"operation must be {expected_operation}",
        )
    root = _object(document, _BASE_KEYS if operation == "inspect" else _RUN_KEYS, "document")
    if root["format_version"] != STAGE_PLAN_DIAGNOSTIC_FILE_FORMAT:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            "document.format_version",
            "unsupported format_version",
        )
    request_id = _text(root["request_id"], "document.request_id")
    fixture_version = _text(root["fixture_version"], "document.fixture_version")
    limits = _workflow_limits(root["workflow_limits"], envelope)
    _measure_json(document, limits)
    numeric = _numeric(root["numeric"])
    core_limits = _object(root["core_limits"], _CORE_LIMIT_KEYS, "core_limits")
    max_plans = _plain_int(
        core_limits["max_plans_per_player"],
        "core_limits.max_plans_per_player",
        1,
        100_000,
    )
    preflight = _preflight_tree(root["tree"], limits, numeric.stage_payoff_bound)
    tree = GameTree(_build_node(preflight["document"]["root"]))
    try:
        validate_tree(tree, tolerance=0, allow_negative_residual=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.INVALID_INPUT,
            "tree",
            str(exc),
        ) from exc
    metadata = _tree_metadata(tree, preflight)
    monitoring, monitoring_counts, monitoring_text = _monitoring(
        root["monitoring"], tree, preflight, limits
    )
    profile, profile_counts, profile_text = _profiles(root["profiles"], metadata, limits)
    metadata = _TreeMetadata(
        metadata.counts,
        metadata.info_sets,
        metadata.members,
        metadata.histories,
        metadata.text_bytes + monitoring_text + profile_text,
    )
    tree_identity = tree_content_identity(tree)
    identity = _inspection_identity(
        fixture_version,
        tree_identity,
        monitoring,
        profile,
        numeric,
        max_plans,
        limits,
    )
    plans = _plan_counts(metadata)
    counts = {
        "tree": metadata.counts,
        "monitoring": monitoring_counts,
        "profiles": profile_counts,
        "plans": plans,
        "predicted_deviation_rows": len(PUBLIC_STATES) * sum(plans.values()),
        "model_attestation_fields": len(_MODEL_FIELDS),
        "perfect_recall_attestation_fields": len(_RECALL_FIELDS),
    }
    spec = _ParsedSpec(
        request_id,
        fixture_version,
        tree,
        monitoring,
        profile,
        numeric,
        max_plans,
        limits,
        tree_identity,
        identity,
        metadata,
        counts,
    )
    _output_preflight(spec, operation)
    return root, spec


def _numeric_output(spec: _ParsedSpec) -> dict[str, Any]:
    bound = spec.numeric.error_bound
    return {
        "delta": str(spec.numeric.delta),
        "stage_payoff_bound": str(spec.numeric.stage_payoff_bound),
        "input_tolerance": str(spec.numeric.input_tolerance),
        "epsilon_claim": str(spec.numeric.epsilon_claim),
        "numeric_error_bound": {
            name: (
                getattr(bound, name)
                if name == "enclosure_established"
                else str(getattr(bound, name))
            )
            for name in _ERROR_BOUND_KEYS
        },
    }


def _check_output_caps(output: dict[str, Any], spec: _ParsedSpec) -> None:
    count = 0
    stack = [output]
    while stack:
        value = stack.pop()
        count += 1
        if count > spec.workflow_limits.max_output_records:
            raise _WorkflowFailure(
                StagePlanDiagnosticFileStatus.CAP_EXCEEDED,
                "output",
                "output record cap exceeded",
            )
        if type(value) is dict:
            stack.extend(value.values())
        elif type(value) is list:
            stack.extend(value)
    encoded = json.dumps(
        output,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > spec.workflow_limits.max_output_bytes:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.CAP_EXCEEDED,
            "output",
            "output byte cap exceeded",
        )


def _inspect_output(spec: _ParsedSpec) -> dict[str, Any]:
    output = {
        "format_version": STAGE_PLAN_DIAGNOSTIC_FILE_FORMAT,
        "operation": "inspect",
        "output_id": _OUTPUT_ID,
        "request_id": spec.request_id,
        "fixture_version": spec.fixture_version,
        "inspection_identity": spec.inspection_identity,
        "tree_content_identity": spec.tree_identity,
        "counts": spec.counts,
        "effective_numeric": _numeric_output(spec),
        "core_limits": {"max_plans_per_player": spec.max_plans_per_player},
        "workflow_limits": asdict(spec.workflow_limits),
        "model_attestation_template": _model_template(),
        "perfect_recall_attestation_template": _recall_template(spec),
    }
    _check_output_caps(output, spec)
    return output


def _model_attestation(value: Any) -> ModelClassAttestation:
    document = _object(value, _MODEL_KEYS, "model_attestation")
    values: dict[str, bool] = {}
    for name in _MODEL_FIELDS:
        confirmed = _boolean(document[name], f"model_attestation.{name}")
        if confirmed is not True:
            raise _WorkflowFailure(
                StagePlanDiagnosticFileStatus.ATTESTATION_FAILURE,
                f"model_attestation.{name}",
                "every model-class assertion must be explicitly confirmed",
                DiagnosticStatus.UNSUPPORTED.value,
            )
        values[name] = confirmed
    return ModelClassAttestation(**values)


def _text_list(
    value: Any,
    phase: str,
    limits: StagePlanDiagnosticFileLimits,
    *,
    nonempty: bool = False,
) -> tuple[str, ...]:
    items = _array(value, limits.max_attestation_records, phase)
    if nonempty and not items:
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.ATTESTATION_FAILURE,
            phase,
            "human evidence list must be non-empty",
            DiagnosticStatus.UNSUPPORTED.value,
        )
    return tuple(
        _text(item, f"{phase}[{index}]", limits.max_attestation_text_length)
        for index, item in enumerate(items)
    )


def _members_evidence(
    value: Any,
    spec: _ParsedSpec,
) -> Mapping[str, Mapping[str, tuple[str, ...]]]:
    document = _object(value, _PLAYER_EVIDENCE_KEYS, "perfect_recall_attestation.information_set_members")
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for player in PLAYERS:
        rows = _array(
            document[player],
            spec.workflow_limits.max_attestation_records,
            f"perfect_recall_attestation.information_set_members.{player}",
        )
        by_info: dict[str, tuple[str, ...]] = {}
        for index, item in enumerate(rows):
            phase = f"perfect_recall_attestation.information_set_members.{player}[{index}]"
            row = _object(item, _INFO_SET_MEMBERS_KEYS, phase)
            info_set = _text(row["info_set"], f"{phase}.info_set", spec.workflow_limits.max_attestation_text_length)
            if info_set in by_info:
                raise _WorkflowFailure(StagePlanDiagnosticFileStatus.ATTESTATION_FAILURE, phase, "duplicate information set", DiagnosticStatus.UNSUPPORTED.value)
            by_info[info_set] = _text_list(row["members"], f"{phase}.members", spec.workflow_limits)
        result[player] = by_info
        if by_info != dict(spec.metadata.members[player]):
            raise _WorkflowFailure(
                StagePlanDiagnosticFileStatus.ATTESTATION_FAILURE,
                f"perfect_recall_attestation.information_set_members.{player}",
                "information-set members do not match the inspected tree",
                DiagnosticStatus.UNSUPPORTED.value,
            )
    return result


def _legal_evidence(
    value: Any,
    spec: _ParsedSpec,
) -> Mapping[str, Mapping[str, tuple[str, ...]]]:
    document = _object(value, _PLAYER_EVIDENCE_KEYS, "perfect_recall_attestation.legal_actions")
    result: dict[str, dict[str, tuple[str, ...]]] = {}
    for player in PLAYERS:
        rows = _array(document[player], spec.workflow_limits.max_attestation_records, f"perfect_recall_attestation.legal_actions.{player}")
        by_info: dict[str, tuple[str, ...]] = {}
        for index, item in enumerate(rows):
            phase = f"perfect_recall_attestation.legal_actions.{player}[{index}]"
            row = _object(item, _LEGAL_ACTIONS_KEYS, phase)
            info_set = _text(row["info_set"], f"{phase}.info_set", spec.workflow_limits.max_attestation_text_length)
            if info_set in by_info:
                raise _WorkflowFailure(StagePlanDiagnosticFileStatus.ATTESTATION_FAILURE, phase, "duplicate information set", DiagnosticStatus.UNSUPPORTED.value)
            by_info[info_set] = _text_list(row["actions"], f"{phase}.actions", spec.workflow_limits)
        result[player] = by_info
        if by_info != dict(spec.metadata.info_sets[player]):
            raise _WorkflowFailure(
                StagePlanDiagnosticFileStatus.ATTESTATION_FAILURE,
                f"perfect_recall_attestation.legal_actions.{player}",
                "legal actions do not match the inspected tree",
                DiagnosticStatus.UNSUPPORTED.value,
            )
    return result


def _history_evidence(
    value: Any,
    spec: _ParsedSpec,
) -> Mapping[str, Mapping[str, Mapping[str, RecallHistory]]]:
    document = _object(value, _PLAYER_EVIDENCE_KEYS, "perfect_recall_attestation.member_histories")
    result: dict[str, dict[str, dict[str, RecallHistory]]] = {}
    for player in PLAYERS:
        groups = _array(document[player], spec.workflow_limits.max_attestation_records, f"perfect_recall_attestation.member_histories.{player}")
        by_info: dict[str, dict[str, RecallHistory]] = {}
        for group_index, item in enumerate(groups):
            phase = f"perfect_recall_attestation.member_histories.{player}[{group_index}]"
            group = _object(item, _HISTORY_GROUP_KEYS, phase)
            info_set = _text(group["info_set"], f"{phase}.info_set", spec.workflow_limits.max_attestation_text_length)
            if info_set in by_info:
                raise _WorkflowFailure(StagePlanDiagnosticFileStatus.ATTESTATION_FAILURE, phase, "duplicate information set", DiagnosticStatus.UNSUPPORTED.value)
            members = _array(group["members"], spec.workflow_limits.max_attestation_records, f"{phase}.members")
            by_node: dict[str, RecallHistory] = {}
            for member_index, raw_member in enumerate(members):
                member_phase = f"{phase}.members[{member_index}]"
                member = _object(raw_member, _HISTORY_MEMBER_KEYS, member_phase)
                node_id = _text(member["node_id"], f"{member_phase}.node_id", spec.workflow_limits.max_attestation_text_length)
                if node_id in by_node:
                    raise _WorkflowFailure(StagePlanDiagnosticFileStatus.ATTESTATION_FAILURE, member_phase, "duplicate member node", DiagnosticStatus.UNSUPPORTED.value)
                history = RecallHistory(
                    observations=_text_list(member["observations"], f"{member_phase}.observations", spec.workflow_limits),
                    own_actions=_text_list(member["own_actions"], f"{member_phase}.own_actions", spec.workflow_limits),
                    information_sets=_text_list(member["information_sets"], f"{member_phase}.information_sets", spec.workflow_limits),
                )
                expected = spec.metadata.histories[player].get(info_set, {}).get(node_id)
                if expected is None or history.own_actions != expected.own_actions or history.information_sets != expected.information_sets:
                    raise _WorkflowFailure(
                        StagePlanDiagnosticFileStatus.ATTESTATION_FAILURE,
                        member_phase,
                        "member history does not match the inspected tree path",
                        DiagnosticStatus.UNSUPPORTED.value,
                    )
                by_node[node_id] = history
            expected_nodes = set(spec.metadata.members[player].get(info_set, ()))
            if set(by_node) != expected_nodes:
                raise _WorkflowFailure(StagePlanDiagnosticFileStatus.ATTESTATION_FAILURE, phase, "member histories are incomplete", DiagnosticStatus.UNSUPPORTED.value)
            values = list(by_node.values())
            if values and any(history != values[0] for history in values[1:]):
                raise _WorkflowFailure(StagePlanDiagnosticFileStatus.ATTESTATION_FAILURE, phase, "histories differ within one information set", DiagnosticStatus.UNSUPPORTED.value)
            by_info[info_set] = by_node
        if set(by_info) != set(spec.metadata.info_sets[player]):
            raise _WorkflowFailure(StagePlanDiagnosticFileStatus.ATTESTATION_FAILURE, f"perfect_recall_attestation.member_histories.{player}", "member histories are incomplete", DiagnosticStatus.UNSUPPORTED.value)
        result[player] = by_info
    return result


def _perfect_recall_attestation(
    value: Any,
    spec: _ParsedSpec,
) -> ManualPerfectRecallAttestation:
    document = _object(value, _RECALL_KEYS, "perfect_recall_attestation")
    maximum = spec.workflow_limits.max_attestation_text_length
    if document["tree_content_identity"] != spec.tree_identity:
        raise _WorkflowFailure(StagePlanDiagnosticFileStatus.ATTESTATION_FAILURE, "perfect_recall_attestation.tree_content_identity", "perfect-recall tree identity mismatch", DiagnosticStatus.UNSUPPORTED.value)
    if document["target_version"] != spec.fixture_version or document["valid_through_version"] != spec.fixture_version:
        raise _WorkflowFailure(StagePlanDiagnosticFileStatus.ATTESTATION_FAILURE, "perfect_recall_attestation.target_version", "perfect-recall evidence is stale or version-mismatched", DiagnosticStatus.UNSUPPORTED.value)
    if document["result_confirmed"] is not True or document["invalidated"] is not False:
        raise _WorkflowFailure(StagePlanDiagnosticFileStatus.ATTESTATION_FAILURE, "perfect_recall_attestation", "perfect-recall evidence must be confirmed and not invalidated", DiagnosticStatus.UNSUPPORTED.value)
    members = _members_evidence(document["information_set_members"], spec)
    histories = _history_evidence(document["member_histories"], spec)
    legal = _legal_evidence(document["legal_actions"], spec)
    return ManualPerfectRecallAttestation(
        fixture_id=_text(document["fixture_id"], "perfect_recall_attestation.fixture_id", maximum),
        tree_content_identity=spec.tree_identity,
        target_version=spec.fixture_version,
        information_set_members=members,
        member_histories=histories,
        legal_actions=legal,
        reviewer=_text(document["reviewer"], "perfect_recall_attestation.reviewer", maximum),
        review_date=_text(document["review_date"], "perfect_recall_attestation.review_date", maximum),
        review_method=_text(document["review_method"], "perfect_recall_attestation.review_method", maximum),
        evidence=_text(document["evidence"], "perfect_recall_attestation.evidence", maximum),
        result_confirmed=True,
        known_limitations=_text_list(document["known_limitations"], "perfect_recall_attestation.known_limitations", spec.workflow_limits, nonempty=True),
        invalidation_conditions=_text_list(document["invalidation_conditions"], "perfect_recall_attestation.invalidation_conditions", spec.workflow_limits, nonempty=True),
        valid_through_version=spec.fixture_version,
        invalidated=False,
    )


def _run_output(spec: _ParsedSpec, result: Any) -> dict[str, Any]:
    if (
        result.status not in (DiagnosticStatus.PASS, DiagnosticStatus.FAIL)
        or result.plan_counts != spec.counts["plans"]
        or len(result.deviations) != spec.counts["predicted_deviation_rows"]
        or set(result.prescribed_values) != {
            (player, state) for player in PLAYERS for state in PUBLIC_STATES
        }
        or result.maximum_lower is None
        or result.maximum_upper is None
    ):
        nested = getattr(result.status, "value", str(result.status))
        raise _WorkflowFailure(
            StagePlanDiagnosticFileStatus.DIAGNOSTIC_FAILURE,
            "diagnostic",
            "diagnostic did not return a complete PASS or FAIL projection",
            nested,
        )
    output = {
        "format_version": STAGE_PLAN_DIAGNOSTIC_FILE_FORMAT,
        "operation": "run",
        "output_id": _OUTPUT_ID,
        "request_id": spec.request_id,
        "fixture_version": spec.fixture_version,
        "qualified_claim": _QUALIFIED_CLAIM,
        "inspection_identity": spec.inspection_identity,
        "tree_content_identity": spec.tree_identity,
        "status": result.status.value,
        "message": result.message,
        "configuration": _numeric_output(spec),
        "core_limits": {"max_plans_per_player": spec.max_plans_per_player},
        "workflow_limits": asdict(spec.workflow_limits),
        "counts": {
            "plans": dict(result.plan_counts),
            "deviation_rows": len(result.deviations),
        },
        "prescribed_values": [
            {
                "player": player,
                "state": state,
                "value": _fraction_text(result.prescribed_values[(player, state)]),
            }
            for player in PLAYERS
            for state in PUBLIC_STATES
        ],
        "deviations": [
            {
                "player": row.player,
                "state": row.state,
                "plan": [
                    {"info_set": info_set, "action": action}
                    for info_set, action in row.plan.actions
                ],
                "prescribed_value": _fraction_text(row.prescribed_value),
                "deviation_value": _fraction_text(row.deviation_value),
                "gain": _fraction_text(row.gain),
                "lower": _fraction_text(row.lower),
                "upper": _fraction_text(row.upper),
            }
            for row in result.deviations
        ],
        "maximum": {
            "lower": _fraction_text(result.maximum_lower),
            "upper": _fraction_text(result.maximum_upper),
        },
        "numeric_enclosure": {
            "unnormalized_value_scale": _fraction_text(result.unnormalized_value_scale),
            "error_bound": _numeric_output(spec)["numeric_error_bound"],
        },
    }
    _check_output_caps(output, spec)
    return output


def _execute(
    raw: bytes,
    expected_operation: str | None,
    limits: StagePlanDiagnosticFileLimits,
) -> StagePlanDiagnosticFileResult:
    try:
        document, spec = _base_document(raw, expected_operation, limits)
        if document["operation"] == "inspect":
            output = _inspect_output(spec)
        else:
            supplied_identity = _object(document["inspection_identity"], _INSPECTION_IDENTITY_KEYS, "inspection_identity")
            if supplied_identity != spec.inspection_identity:
                raise _WorkflowFailure(StagePlanDiagnosticFileStatus.IDENTITY_MISMATCH, "inspection_identity", "inspection identity does not match the supplied specification")
            model = _model_attestation(document["model_attestation"])
            recall = _perfect_recall_attestation(document["perfect_recall_attestation"], spec)
            try:
                diagnostic = diagnose_stage_plan_deviations(
                    tree=spec.tree,
                    fixture_version=spec.fixture_version,
                    profile=spec.profile,
                    monitoring=spec.monitoring,
                    model_attestation=model,
                    perfect_recall_attestation=recall,
                    delta=spec.numeric.delta,
                    stage_payoff_bound=spec.numeric.stage_payoff_bound,
                    input_tolerance=spec.numeric.input_tolerance,
                    epsilon_claim=spec.numeric.epsilon_claim,
                    numeric_error_bound=spec.numeric.error_bound,
                    max_plans_per_player=spec.max_plans_per_player,
                )
            except (TypeError, ValueError) as exc:
                raise _WorkflowFailure(StagePlanDiagnosticFileStatus.DIAGNOSTIC_FAILURE, "diagnostic", str(exc)) from exc
            if diagnostic.status not in (DiagnosticStatus.PASS, DiagnosticStatus.FAIL):
                raise _WorkflowFailure(
                    StagePlanDiagnosticFileStatus.DIAGNOSTIC_FAILURE,
                    "diagnostic",
                    "diagnostic did not produce a complete PASS or FAIL result",
                    diagnostic.status.value,
                )
            output = _run_output(spec, diagnostic)
        return StagePlanDiagnosticFileResult(StagePlanDiagnosticFileStatus.SUCCESS, output, None)
    except _WorkflowFailure as exc:
        return _failure(exc)
    except Exception:
        return _failure(_WorkflowFailure(StagePlanDiagnosticFileStatus.INTERNAL_FAILURE, "internal", "unexpected workflow failure"))


def inspect_stage_plan_diagnostic_file(
    raw: bytes,
    limits: StagePlanDiagnosticFileLimits = _DEFAULT_LIMITS,
) -> StagePlanDiagnosticFileResult:
    """Inspect one strict exact-rational document without analytic execution."""

    return _execute(raw, "inspect", limits)


def run_stage_plan_diagnostic_file(
    raw: bytes,
    limits: StagePlanDiagnosticFileLimits = _DEFAULT_LIMITS,
) -> StagePlanDiagnosticFileResult:
    """Run one identity-bound, fully human-attested bounded diagnostic."""

    return _execute(raw, "run", limits)


def process_stage_plan_diagnostic_file(
    raw: bytes,
    limits: StagePlanDiagnosticFileLimits = _DEFAULT_LIMITS,
) -> StagePlanDiagnosticFileResult:
    """Dispatch one strict document by its explicit inspect/run operation."""

    return _execute(raw, None, limits)


def stage_plan_diagnostic_file_json(result: StagePlanDiagnosticFileResult) -> str:
    """Serialize one result as deterministic strict one-line JSON."""

    if type(result) is not StagePlanDiagnosticFileResult:
        raise TypeError("result must be StagePlanDiagnosticFileResult")
    payload = {
        "status": result.status.value,
        "output": result.output,
        "error": None if result.error is None else asdict(result.error),
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
