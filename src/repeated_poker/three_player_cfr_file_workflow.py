"""Strict two-phase file adapter for the three-player CFR-style diagnostic.

``inspect`` validates a bounded recursive tree and complete fixed-Hero policy,
then returns content-bound human-attestation material without running CFR or
the oracle. ``run`` revalidates the inspection identity and a complete,
human-authored perfect-recall attestation before delegating exactly once to the
existing public M12 diagnostic. Controlled failures expose no partial strategy,
utility, regret, oracle, identity, or count payload.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Any

from .three_player_cfr import (
    ALGORITHM_VERSION,
    CAP_EXCEEDED as CORE_CAP_EXCEEDED,
    CONTRACT_VERSION,
    DIAGNOSTIC_COMPLETE,
    NON_REPRODUCIBLE as CORE_NON_REPRODUCIBLE,
    NOT_REQUESTED,
    UNSUPPORTED_MODEL,
    BehaviorStrategy,
    CfrConfig,
    CfrSafetyLimits,
    DiagnosticContractError,
    FixedHeroNode,
    OpponentDecisionNode,
    PerfectRecallAttestation,
    ThreePlayerChanceNode,
    ThreePlayerGameTree,
    ThreePlayerTerminalNode,
    UtilityVector,
    iter_three_player_nodes,
    run_three_player_cfr_diagnostic,
    tree_content_identity,
    validate_three_player_tree,
)


__all__ = [
    "THREE_PLAYER_CFR_FILE_FORMAT",
    "THREE_PLAYER_CFR_INSPECTION_ID",
    "ThreePlayerCfrFileStatus",
    "ThreePlayerCfrFileLimits",
    "ThreePlayerCfrFileError",
    "ThreePlayerCfrFileResult",
    "inspect_three_player_cfr_file",
    "run_three_player_cfr_file",
    "process_three_player_cfr_file",
    "three_player_cfr_file_json",
]


THREE_PLAYER_CFR_FILE_FORMAT = "three-player-cfr-file-v1"
THREE_PLAYER_CFR_INSPECTION_ID = "three-player-cfr-inspection-sha256-v1"
_OUTPUT_ID = "three-player-cfr-file-output-v1"


class ThreePlayerCfrFileStatus(str, Enum):
    """Stable outer status classes for the versioned file adapter."""

    SUCCESS = "SUCCESS"
    PARSE_FAILURE = "PARSE_FAILURE"
    INVALID_INPUT = "INVALID_INPUT"
    CAP_EXCEEDED = "CAP_EXCEEDED"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    ATTESTATION_FAILURE = "ATTESTATION_FAILURE"
    DIAGNOSTIC_FAILURE = "DIAGNOSTIC_FAILURE"
    NON_REPRODUCIBLE = "NON_REPRODUCIBLE"
    INTERNAL_FAILURE = "INTERNAL_FAILURE"


@dataclass(frozen=True)
class ThreePlayerCfrFileLimits:
    """Caller-lowerable adapter ceilings outside the M12 core limits."""

    max_input_bytes: int = 1_000_000
    max_json_depth: int = 64
    max_total_json_values: int = 100_000
    max_tree_depth: int = 64
    max_tree_nodes: int = 500
    max_tree_branches: int = 2_000
    max_policy_info_sets: int = 16
    max_policy_actions: int = 64
    max_attestation_text_length: int = 200
    max_output_records: int = 100_000
    max_output_bytes: int = 4_000_000


@dataclass(frozen=True)
class ThreePlayerCfrFileError:
    """Bounded controlled-failure metadata with no partial result."""

    phase: str
    message: str
    nested_status: str | None


@dataclass(frozen=True)
class ThreePlayerCfrFileResult:
    """Exclusive success-output or controlled-failure wrapper."""

    status: ThreePlayerCfrFileStatus
    output: dict[str, Any] | None
    error: ThreePlayerCfrFileError | None


@dataclass(frozen=True)
class _ParsedSpec:
    request_id: str
    tree: ThreePlayerGameTree
    fixed_hero_policy: BehaviorStrategy
    config: CfrConfig
    core_limits: CfrSafetyLimits
    workflow_limits: ThreePlayerCfrFileLimits
    tree_identity: str
    inspection_identity: dict[str, str]
    counts: dict[str, Any]
    opponent_template: dict[str, list[dict[str, Any]]]
    fixed_hero_echo: list[dict[str, Any]]


_DEFAULT_FILE_LIMITS = ThreePlayerCfrFileLimits()
_FILE_LIMIT_CEILINGS = asdict(_DEFAULT_FILE_LIMITS)
_DEFAULT_CORE_LIMITS = CfrSafetyLimits()
_CORE_LIMIT_CEILINGS = {
    field.name: getattr(_DEFAULT_CORE_LIMITS, field.name)
    for field in fields(CfrSafetyLimits)
}
_CORE_LIMIT_KEYS = set(_CORE_LIMIT_CEILINGS)
_WORKFLOW_LIMIT_KEYS = set(_FILE_LIMIT_CEILINGS)
_BASE_KEYS = {
    "format_version",
    "operation",
    "request_id",
    "tree",
    "fixed_hero_policy",
    "config",
    "core_limits",
    "workflow_limits",
}
_RUN_KEYS = _BASE_KEYS | {"inspection_identity", "attestation"}
_TREE_KEYS = {"description", "root"}
_TERMINAL_KEYS = {"type", "node_id", "utility"}
_CHANCE_KEYS = {"type", "node_id", "children"}
_DECISION_KEYS = {"type", "node_id", "info_set", "actions"}
_UTILITY_KEYS = {"H", "O1", "O2", "R"}
_CHILD_KEYS = {"probability", "child"}
_ACTION_KEYS = {"action", "child"}
_POLICY_KEYS = {"info_sets"}
_POLICY_ROW_KEYS = {"info_set", "actions"}
_POLICY_ACTION_KEYS = {"action", "probability"}
_CONFIG_KEYS = {
    "iterations",
    "input_tolerance",
    "epsilon_deviation",
    "oracle_compare_tolerance",
    "reproducibility_tolerance",
    "compute_deviation_gains",
    "request_oracle",
    "include_oracle_rows",
    "trace_checkpoint_interval",
    "seed",
}
_INSPECTION_IDENTITY_KEYS = {
    "template_id",
    "semantic_sha256",
    "tree_content_identity",
}
_ATTESTATION_KEYS = {
    "tree_content_identity",
    "o1_confirmed",
    "o2_confirmed",
    "verifier",
    "verification_date",
    "evidence_version",
}


class _WorkflowFailure(ValueError):
    def __init__(
        self,
        status: ThreePlayerCfrFileStatus,
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


def _failure(exc: _WorkflowFailure) -> ThreePlayerCfrFileResult:
    return ThreePlayerCfrFileResult(
        exc.status,
        None,
        ThreePlayerCfrFileError(
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
                ThreePlayerCfrFileStatus.PARSE_FAILURE,
                "json",
                f"duplicate JSON key: {key}",
            )
        result[key] = value
    return result


def _validate_file_limits(limits: ThreePlayerCfrFileLimits) -> None:
    if type(limits) is not ThreePlayerCfrFileLimits:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.INVALID_INPUT,
            "limits",
            "workflow limits have the wrong type",
        )
    for name, ceiling in _FILE_LIMIT_CEILINGS.items():
        value = getattr(limits, name)
        if type(value) is not int or value <= 0 or value > ceiling:
            raise _WorkflowFailure(
                ThreePlayerCfrFileStatus.INVALID_INPUT,
                "limits",
                f"{name} must be a positive int no greater than {ceiling}",
            )


def _measure_json(value: Any, limits: ThreePlayerCfrFileLimits) -> None:
    count = 0
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        count += 1
        if count > limits.max_total_json_values:
            raise _WorkflowFailure(
                ThreePlayerCfrFileStatus.CAP_EXCEEDED,
                "json",
                "total JSON value cap exceeded",
            )
        if depth > limits.max_json_depth:
            raise _WorkflowFailure(
                ThreePlayerCfrFileStatus.CAP_EXCEEDED,
                "json",
                "JSON depth cap exceeded",
            )
        if type(item) is dict:
            stack.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            stack.extend((child, depth + 1) for child in item)


def _parse(raw: bytes, limits: ThreePlayerCfrFileLimits) -> dict[str, Any]:
    _validate_file_limits(limits)
    if type(raw) is not bytes:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.INVALID_INPUT, "input", "input must be bytes"
        )
    if len(raw) > limits.max_input_bytes:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.CAP_EXCEEDED,
            "input",
            "input byte cap exceeded",
        )
    if raw.startswith(b"\xef\xbb\xbf"):
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.PARSE_FAILURE,
            "json",
            "UTF-8 BOM is not allowed",
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.PARSE_FAILURE,
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
            ThreePlayerCfrFileStatus.CAP_EXCEEDED,
            "json",
            "JSON nesting exceeds the parser depth cap",
        ) from exc
    except (TypeError, ValueError) as exc:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.PARSE_FAILURE,
            "json",
            f"invalid JSON: {exc}",
        ) from exc
    if type(value) is not dict:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.INVALID_INPUT,
            "document",
            "top-level JSON value must be an object",
        )
    _measure_json(value, limits)
    return value


def _object(value: Any, keys: set[str], phase: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.INVALID_INPUT, phase, "value must be an object"
        )
    actual = set(value)
    missing = keys - actual
    unknown = actual - keys
    if missing:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.INVALID_INPUT,
            phase,
            f"missing keys: {', '.join(sorted(missing))}",
        )
    if unknown:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.INVALID_INPUT,
            phase,
            f"unknown keys: {', '.join(sorted(unknown))}",
        )
    return value


def _array(value: Any, cap: int, phase: str) -> list[Any]:
    if type(value) is not list:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.INVALID_INPUT, phase, "value must be an array"
        )
    if len(value) > cap:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.CAP_EXCEEDED, phase, "array cap exceeded"
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
            ThreePlayerCfrFileStatus.INVALID_INPUT,
            phase,
            f"value must be a non-empty control-free string of at most {maximum} characters",
        )
    return value


def _number(value: Any, phase: str) -> float:
    if type(value) not in (int, float):
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.INVALID_INPUT,
            phase,
            "value must be a finite binary64 number",
        )
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.INVALID_INPUT,
            phase,
            "value must be a finite binary64 number",
        ) from exc
    if not math.isfinite(converted):
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.INVALID_INPUT,
            phase,
            "value must be a finite binary64 number",
        )
    return converted


def _plain_int(value: Any, phase: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.INVALID_INPUT,
            phase,
            f"value must be an integer in [{minimum}, {maximum}]",
        )
    return value


def _boolean(value: Any, phase: str) -> bool:
    if type(value) is not bool:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.INVALID_INPUT, phase, "value must be a boolean"
        )
    return value


def _workflow_limits(
    value: Any, envelope: ThreePlayerCfrFileLimits
) -> ThreePlayerCfrFileLimits:
    document = _object(value, _WORKFLOW_LIMIT_KEYS, "workflow_limits")
    parsed = {
        name: _plain_int(document[name], f"workflow_limits.{name}", 1, ceiling)
        for name, ceiling in _FILE_LIMIT_CEILINGS.items()
    }
    requested = ThreePlayerCfrFileLimits(**parsed)
    return ThreePlayerCfrFileLimits(
        **{
            name: min(getattr(requested, name), getattr(envelope, name))
            for name in _FILE_LIMIT_CEILINGS
        }
    )


def _core_limits(value: Any) -> CfrSafetyLimits:
    document = _object(value, _CORE_LIMIT_KEYS, "core_limits")
    parsed = {
        name: _plain_int(document[name], f"core_limits.{name}", 1, ceiling)
        for name, ceiling in _CORE_LIMIT_CEILINGS.items()
    }
    try:
        return CfrSafetyLimits(**parsed)
    except ValueError as exc:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.INVALID_INPUT, "core_limits", str(exc)
        ) from exc


def _preflight_tree(
    value: Any,
    core_limits: CfrSafetyLimits,
    workflow_limits: ThreePlayerCfrFileLimits,
) -> tuple[dict[str, Any], int, int]:
    tree = _object(value, _TREE_KEYS, "tree")
    _text(tree["description"], "tree.description", 512)
    node_count = 0
    terminal_count = 0
    branch_count = 0
    stack: list[tuple[Any, int, str]] = [(tree["root"], 1, "tree.root")]
    while stack:
        raw_node, depth, phase = stack.pop()
        if depth > workflow_limits.max_tree_depth:
            raise _WorkflowFailure(
                ThreePlayerCfrFileStatus.CAP_EXCEEDED,
                phase,
                "tree depth cap exceeded",
            )
        if type(raw_node) is not dict:
            raise _WorkflowFailure(
                ThreePlayerCfrFileStatus.INVALID_INPUT,
                phase,
                "tree node must be an object",
            )
        node_type = raw_node.get("type")
        node_count += 1
        if node_count > min(workflow_limits.max_tree_nodes, core_limits.max_nodes):
            raise _WorkflowFailure(
                ThreePlayerCfrFileStatus.CAP_EXCEEDED,
                phase,
                "tree node cap exceeded",
            )
        if node_type == "terminal":
            terminal_count += 1
            if terminal_count > core_limits.max_terminal_nodes:
                raise _WorkflowFailure(
                    ThreePlayerCfrFileStatus.CAP_EXCEEDED,
                    phase,
                    "terminal node cap exceeded",
                )
            node = _object(raw_node, _TERMINAL_KEYS, phase)
            _text(node["node_id"], f"{phase}.node_id")
            utility = _object(node["utility"], _UTILITY_KEYS, f"{phase}.utility")
            for player in ("H", "O1", "O2", "R"):
                _number(utility[player], f"{phase}.utility.{player}")
            continue
        if node_type == "chance":
            node = _object(raw_node, _CHANCE_KEYS, phase)
            _text(node["node_id"], f"{phase}.node_id")
            children = _array(
                node["children"], core_limits.max_chance_outcomes_per_node, f"{phase}.children"
            )
            if not children:
                raise _WorkflowFailure(
                    ThreePlayerCfrFileStatus.INVALID_INPUT,
                    f"{phase}.children",
                    "chance node must have at least one child",
                )
            branch_count += len(children)
            for index, raw_child in enumerate(children):
                child_phase = f"{phase}.children[{index}]"
                child = _object(raw_child, _CHILD_KEYS, child_phase)
                _number(child["probability"], f"{child_phase}.probability")
                stack.append((child["child"], depth + 1, f"{child_phase}.child"))
        elif node_type in ("fixed_hero", "opponent_1", "opponent_2"):
            node = _object(raw_node, _DECISION_KEYS, phase)
            _text(node["node_id"], f"{phase}.node_id")
            _text(node["info_set"], f"{phase}.info_set")
            actions = _array(
                node["actions"], core_limits.max_actions_per_info_set, f"{phase}.actions"
            )
            if not actions:
                raise _WorkflowFailure(
                    ThreePlayerCfrFileStatus.INVALID_INPUT,
                    f"{phase}.actions",
                    "decision node must have at least one action",
                )
            branch_count += len(actions)
            for index, raw_action in enumerate(actions):
                action_phase = f"{phase}.actions[{index}]"
                action = _object(raw_action, _ACTION_KEYS, action_phase)
                _text(action["action"], f"{action_phase}.action")
                stack.append((action["child"], depth + 1, f"{action_phase}.child"))
        else:
            raise _WorkflowFailure(
                ThreePlayerCfrFileStatus.INVALID_INPUT,
                f"{phase}.type",
                "unsupported tree node type",
            )
        if branch_count > workflow_limits.max_tree_branches:
            raise _WorkflowFailure(
                ThreePlayerCfrFileStatus.CAP_EXCEEDED,
                phase,
                "tree branch cap exceeded",
            )
    return tree, node_count, branch_count


def _build_tree(value: dict[str, Any]) -> ThreePlayerGameTree:
    def build(node: dict[str, Any]) -> Any:
        node_type = node["type"]
        if node_type == "terminal":
            utility = node["utility"]
            return ThreePlayerTerminalNode(
                node["node_id"],
                UtilityVector(
                    _number(utility["H"], "tree.utility.H"),
                    _number(utility["O1"], "tree.utility.O1"),
                    _number(utility["O2"], "tree.utility.O2"),
                    _number(utility["R"], "tree.utility.R"),
                ),
            )
        if node_type == "chance":
            return ThreePlayerChanceNode(
                node["node_id"],
                tuple(
                    (_number(child["probability"], "tree.chance.probability"), build(child["child"]))
                    for child in node["children"]
                ),
            )
        actions = tuple((item["action"], build(item["child"])) for item in node["actions"])
        if node_type == "fixed_hero":
            return FixedHeroNode(node["node_id"], node["info_set"], actions)
        return OpponentDecisionNode(
            node["node_id"],
            "opponent_1" if node_type == "opponent_1" else "opponent_2",
            node["info_set"],
            actions,
        )

    return ThreePlayerGameTree(build(value["root"]), value["description"])


def _fixed_hero_policy(
    value: Any, limits: ThreePlayerCfrFileLimits
) -> BehaviorStrategy:
    document = _object(value, _POLICY_KEYS, "fixed_hero_policy")
    rows = _array(
        document["info_sets"], limits.max_policy_info_sets, "fixed_hero_policy.info_sets"
    )
    probabilities: dict[str, dict[str, float]] = {}
    action_count = 0
    for index, raw_row in enumerate(rows):
        phase = f"fixed_hero_policy.info_sets[{index}]"
        row = _object(raw_row, _POLICY_ROW_KEYS, phase)
        info_set = _text(row["info_set"], f"{phase}.info_set")
        if info_set in probabilities:
            raise _WorkflowFailure(
                ThreePlayerCfrFileStatus.INVALID_INPUT, phase, "duplicate policy info_set"
            )
        actions = _array(row["actions"], limits.max_policy_actions, f"{phase}.actions")
        distribution: dict[str, float] = {}
        for action_index, raw_action in enumerate(actions):
            action_phase = f"{phase}.actions[{action_index}]"
            action = _object(raw_action, _POLICY_ACTION_KEYS, action_phase)
            label = _text(action["action"], f"{action_phase}.action")
            if label in distribution:
                raise _WorkflowFailure(
                    ThreePlayerCfrFileStatus.INVALID_INPUT,
                    action_phase,
                    "duplicate policy action",
                )
            distribution[label] = _number(
                action["probability"], f"{action_phase}.probability"
            )
            action_count += 1
            if action_count > limits.max_policy_actions:
                raise _WorkflowFailure(
                    ThreePlayerCfrFileStatus.CAP_EXCEEDED,
                    "fixed_hero_policy",
                    "policy action cap exceeded",
                )
        probabilities[info_set] = distribution
    return BehaviorStrategy(probabilities)


def _config(value: Any, limits: CfrSafetyLimits) -> CfrConfig:
    document = _object(value, _CONFIG_KEYS, "config")
    if document["trace_checkpoint_interval"] is not None:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.INVALID_INPUT,
            "config.trace_checkpoint_interval",
            "v1 requires trace_checkpoint_interval=null",
        )
    if document["seed"] is not None:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.INVALID_INPUT,
            "config.seed",
            "v1 requires seed=null",
        )
    compute = _boolean(document["compute_deviation_gains"], "config.compute_deviation_gains")
    request_oracle = _boolean(document["request_oracle"], "config.request_oracle")
    include_rows = _boolean(document["include_oracle_rows"], "config.include_oracle_rows")
    if not compute or include_rows:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.INVALID_INPUT,
            "config",
            "v1 requires deviation gains enabled and oracle rows disabled",
        )
    try:
        return CfrConfig(
            iterations=_plain_int(document["iterations"], "config.iterations", 1, limits.max_iterations),
            input_tolerance=_number(document["input_tolerance"], "config.input_tolerance"),
            epsilon_deviation=_number(document["epsilon_deviation"], "config.epsilon_deviation"),
            oracle_compare_tolerance=_number(document["oracle_compare_tolerance"], "config.oracle_compare_tolerance"),
            reproducibility_tolerance=_number(document["reproducibility_tolerance"], "config.reproducibility_tolerance"),
            limits=limits,
            compute_deviation_gains=True,
            request_oracle=request_oracle,
            include_oracle_rows=False,
            trace_checkpoint_interval=None,
            seed=None,
        )
    except DiagnosticContractError as exc:
        status = (
            ThreePlayerCfrFileStatus.CAP_EXCEEDED
            if exc.status == CORE_CAP_EXCEEDED
            else ThreePlayerCfrFileStatus.INVALID_INPUT
        )
        raise _WorkflowFailure(status, "config", str(exc), exc.status) from exc
    except ValueError as exc:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.INVALID_INPUT, "config", str(exc)
        ) from exc


def _identity_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _WorkflowFailure(
                ThreePlayerCfrFileStatus.INTERNAL_FAILURE,
                "identity",
                "identity contains a non-finite number",
            )
        return {"float_hex": value.hex()}
    if isinstance(value, tuple):
        return [_identity_value(item) for item in value]
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
    tree_identity: str,
    policy: BehaviorStrategy,
    config: CfrConfig,
    workflow_limits: ThreePlayerCfrFileLimits,
) -> dict[str, str]:
    config_payload = {
        "iterations": config.iterations,
        "input_tolerance": config.input_tolerance,
        "epsilon_deviation": config.epsilon_deviation,
        "oracle_compare_tolerance": config.oracle_compare_tolerance,
        "reproducibility_tolerance": config.reproducibility_tolerance,
        "compute_deviation_gains": config.compute_deviation_gains,
        "request_oracle": config.request_oracle,
        "include_oracle_rows": config.include_oracle_rows,
        "trace_checkpoint_interval": None,
        "seed": None,
    }
    semantic = {
        "format_version": THREE_PLAYER_CFR_FILE_FORMAT,
        "tree_content_identity": tree_identity,
        "fixed_hero_policy": policy.probabilities,
        "config": config_payload,
        "core_limits": config.limits,
        "workflow_limits": workflow_limits,
    }
    return {
        "template_id": THREE_PLAYER_CFR_INSPECTION_ID,
        "semantic_sha256": _sha256(semantic),
        "tree_content_identity": tree_identity,
    }


def _metadata(tree: ThreePlayerGameTree, policy: BehaviorStrategy) -> tuple[
    dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]
]:
    opponent: dict[str, dict[str, tuple[str, ...]]] = {"O1": {}, "O2": {}}
    fixed: dict[str, tuple[str, ...]] = {}
    node_count = terminal_count = chance_count = branch_count = action_total = 0
    action_max = 0
    for node in iter_three_player_nodes(tree.root):
        node_count += 1
        if isinstance(node, ThreePlayerTerminalNode):
            terminal_count += 1
        elif isinstance(node, ThreePlayerChanceNode):
            chance_count += 1
            branch_count += len(node.children)
        elif isinstance(node, FixedHeroNode):
            actions = tuple(action for action, _ in node.actions)
            fixed.setdefault(node.info_set, actions)
            branch_count += len(actions)
            action_total += len(actions)
            action_max = max(action_max, len(actions))
        elif isinstance(node, OpponentDecisionNode):
            player = "O1" if node.owner == "opponent_1" else "O2"
            actions = tuple(action for action, _ in node.actions)
            opponent[player].setdefault(node.info_set, actions)
            branch_count += len(actions)
            action_total += len(actions)
            action_max = max(action_max, len(actions))
    template = {
        player: [
            {"info_set": info_set, "actions": list(opponent[player][info_set])}
            for info_set in sorted(opponent[player])
        ]
        for player in ("O1", "O2")
    }
    hero_echo = [
        {
            "info_set": info_set,
            "actions": [
                {"action": action, "probability": policy.probabilities[info_set][action]}
                for action in fixed[info_set]
            ],
        }
        for info_set in sorted(fixed)
    ]
    counts = {
        "nodes": node_count,
        "terminals": terminal_count,
        "chance_nodes": chance_count,
        "branches": branch_count,
        "fixed_hero_info_sets": len(fixed),
        "opponent_info_sets": {"O1": len(opponent["O1"]), "O2": len(opponent["O2"])},
        "decision_actions_total": action_total,
        "actions_per_info_set_max": action_max,
    }
    return counts, template, hero_echo


def _base_document(
    raw: bytes,
    expected_operation: str | None,
    envelope: ThreePlayerCfrFileLimits,
) -> tuple[dict[str, Any], _ParsedSpec]:
    document = _parse(raw, envelope)
    operation = document.get("operation")
    if operation not in ("inspect", "run"):
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.INVALID_INPUT,
            "document.operation",
            "operation must be inspect or run",
        )
    if expected_operation is not None and operation != expected_operation:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.INVALID_INPUT,
            "document.operation",
            f"operation must be {expected_operation}",
        )
    root = _object(document, _BASE_KEYS if operation == "inspect" else _RUN_KEYS, "document")
    if root["format_version"] != THREE_PLAYER_CFR_FILE_FORMAT:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.INVALID_INPUT,
            "document.format_version",
            "unsupported format_version",
        )
    request_id = _text(root["request_id"], "request_id")
    effective_workflow_limits = _workflow_limits(root["workflow_limits"], envelope)
    _measure_json(document, effective_workflow_limits)
    core_limits = _core_limits(root["core_limits"])
    raw_tree, _, _ = _preflight_tree(root["tree"], core_limits, effective_workflow_limits)
    tree = _build_tree(raw_tree)
    policy = _fixed_hero_policy(root["fixed_hero_policy"], effective_workflow_limits)
    config = _config(root["config"], core_limits)
    try:
        identity = tree_content_identity(tree)
        validate_three_player_tree(
            tree,
            policy,
            tolerance=config.input_tolerance,
            limits=core_limits,
        )
    except (TypeError, ValueError) as exc:
        message = str(exc)
        status = (
            ThreePlayerCfrFileStatus.CAP_EXCEEDED
            if "safety limit exceeded" in message
            else ThreePlayerCfrFileStatus.INVALID_INPUT
        )
        raise _WorkflowFailure(status, "tree", message) from exc
    counts, opponent_template, fixed_echo = _metadata(tree, policy)
    inspection = _inspection_identity(identity, policy, config, effective_workflow_limits)
    return root, _ParsedSpec(
        request_id,
        tree,
        policy,
        config,
        core_limits,
        effective_workflow_limits,
        identity,
        inspection,
        counts,
        opponent_template,
        fixed_echo,
    )


def _output_caps(
    output: dict[str, Any], projected_records: int, limits: ThreePlayerCfrFileLimits
) -> None:
    if projected_records > limits.max_output_records:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.CAP_EXCEEDED,
            "output",
            "output record cap exceeded",
        )
    encoded = json.dumps(
        output,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > limits.max_output_bytes:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.CAP_EXCEEDED,
            "output",
            "output byte cap exceeded",
        )


def _inspect_output(spec: _ParsedSpec) -> dict[str, Any]:
    projected = 128 + 12 * spec.counts["branches"] + 16 * spec.counts["decision_actions_total"]
    if projected > spec.workflow_limits.max_output_records:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.CAP_EXCEEDED,
            "output",
            "output record cap exceeded",
        )
    projected_bytes = (
        8_192
        + 512 * spec.counts["nodes"]
        + 256 * spec.counts["decision_actions_total"]
    )
    if projected_bytes > spec.workflow_limits.max_output_bytes:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.CAP_EXCEEDED,
            "output",
            "output byte cap exceeded",
        )
    output = {
        "format_version": THREE_PLAYER_CFR_FILE_FORMAT,
        "operation": "inspect",
        "output_id": _OUTPUT_ID,
        "request_id": spec.request_id,
        "inspection_identity": spec.inspection_identity,
        "tree_content_identity": spec.tree_identity,
        "counts": spec.counts,
        "opponent_action_template": spec.opponent_template,
        "fixed_hero_policy": spec.fixed_hero_echo,
        "effective_config": {
            key: getattr(spec.config, key)
            for key in _CONFIG_KEYS
        },
        "core_limits": asdict(spec.core_limits),
        "workflow_limits": asdict(spec.workflow_limits),
        "attestation_template": {
            "tree_content_identity": spec.tree_identity,
            "o1_confirmed": None,
            "o2_confirmed": None,
            "verifier": None,
            "verification_date": None,
            "evidence_version": None,
        },
    }
    _output_caps(output, projected, spec.workflow_limits)
    return output


def _attestation(value: Any, spec: _ParsedSpec) -> PerfectRecallAttestation:
    document = _object(value, _ATTESTATION_KEYS, "attestation")
    if document["tree_content_identity"] != spec.tree_identity:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.ATTESTATION_FAILURE,
            "attestation.tree_content_identity",
            "attestation tree identity mismatch",
            UNSUPPORTED_MODEL,
        )
    if document["o1_confirmed"] is not True or document["o2_confirmed"] is not True:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.ATTESTATION_FAILURE,
            "attestation",
            "human attestation must explicitly confirm O1 and O2",
            UNSUPPORTED_MODEL,
        )
    maximum = spec.workflow_limits.max_attestation_text_length
    try:
        return PerfectRecallAttestation(
            spec.tree_identity,
            True,
            True,
            _text(document["verifier"], "attestation.verifier", maximum),
            _text(document["verification_date"], "attestation.verification_date", maximum),
            _text(document["evidence_version"], "attestation.evidence_version", maximum),
        )
    except ValueError as exc:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.ATTESTATION_FAILURE,
            "attestation",
            str(exc),
            UNSUPPORTED_MODEL,
        ) from exc


def _preflight_analysis(spec: _ParsedSpec) -> None:
    plan_counts: dict[str, int] = {}
    for player in ("O1", "O2"):
        count = 1
        for row in spec.opponent_template[player]:
            count *= len(row["actions"])
        plan_counts[player] = count
    joint = plan_counts["O1"] * plan_counts["O2"]
    if joint > spec.core_limits.max_oracle_pure_profiles:
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.CAP_EXCEEDED,
            "deviation",
            "unilateral-deviation pure-profile cap exceeded",
        )
    if spec.config.request_oracle:
        evaluations = 2 * joint + plan_counts["O1"] + plan_counts["O2"] + 1
        if max(plan_counts.values()) > spec.core_limits.max_oracle_pure_plans_per_player:
            raise _WorkflowFailure(ThreePlayerCfrFileStatus.CAP_EXCEEDED, "oracle", "oracle pure-plan cap exceeded")
        if joint > spec.core_limits.max_oracle_joint_profiles:
            raise _WorkflowFailure(ThreePlayerCfrFileStatus.CAP_EXCEEDED, "oracle", "oracle joint-profile cap exceeded")
        if evaluations > spec.core_limits.max_oracle_profile_evaluations:
            raise _WorkflowFailure(ThreePlayerCfrFileStatus.CAP_EXCEEDED, "oracle", "oracle evaluation cap exceeded")
    projected_records = 256 + 20 * spec.counts["decision_actions_total"] + 12 * spec.counts["branches"]
    if projected_records > spec.workflow_limits.max_output_records:
        raise _WorkflowFailure(ThreePlayerCfrFileStatus.CAP_EXCEEDED, "output", "output record cap exceeded")
    label_bytes = sum(
        len(row["info_set"]) + sum(len(action) for action in row["actions"])
        for player in ("O1", "O2")
        for row in spec.opponent_template[player]
    )
    projected_bytes = 16_384 + 512 * spec.counts["nodes"] + 256 * spec.counts["decision_actions_total"] + 8 * label_bytes
    if projected_bytes > spec.workflow_limits.max_output_bytes:
        raise _WorkflowFailure(ThreePlayerCfrFileStatus.CAP_EXCEEDED, "output", "output byte cap exceeded")


def _finite(value: Any) -> None:
    stack = [value]
    while stack:
        item = stack.pop()
        if type(item) is float and not math.isfinite(item):
            raise _WorkflowFailure(
                ThreePlayerCfrFileStatus.DIAGNOSTIC_FAILURE,
                "diagnostic",
                "diagnostic output contains a non-finite number",
            )
        if type(item) is dict:
            stack.extend(item.values())
        elif type(item) in (list, tuple):
            stack.extend(item)


def _run_output(spec: _ParsedSpec, result: Any) -> dict[str, Any]:
    oracle = result.oracle_attachment
    expected_oracle_status = "MATCH" if spec.config.request_oracle else NOT_REQUESTED
    expected_coverage = "complete" if spec.config.request_oracle else "none"
    if (
        result.component_status != DIAGNOSTIC_COMPLETE
        or result.overall_status != DIAGNOSTIC_COMPLETE
        or result.requested_iterations != spec.config.iterations
        or result.completed_iterations != spec.config.iterations
        or result.deterministic_full_traversal is not True
        or result.stopped_by_safety_cap is not False
        or oracle.get("status") != expected_oracle_status
        or oracle.get("coverage") != expected_coverage
        or oracle.get("rows") != []
    ):
        nested = str(result.overall_status)
        raise _WorkflowFailure(
            ThreePlayerCfrFileStatus.DIAGNOSTIC_FAILURE,
            "diagnostic",
            "diagnostic did not produce the required complete no-row result",
            nested,
        )
    counts = oracle.get("counts") if spec.config.request_oracle else None
    output = {
        "format_version": THREE_PLAYER_CFR_FILE_FORMAT,
        "operation": "run",
        "output_id": _OUTPUT_ID,
        "request_id": spec.request_id,
        "contract_version": result.contract_version,
        "algorithm_version": result.algorithm_version,
        "inspection_identity": spec.inspection_identity,
        "tree_content_identity": spec.tree_identity,
        "status": {
            "component": result.component_status,
            "overall": result.overall_status,
        },
        "iterations": {
            "requested": result.requested_iterations,
            "completed": result.completed_iterations,
        },
        "counts": {
            "nodes": result.node_count,
            "terminals": result.terminal_count,
            "opponent_info_sets": result.info_set_count_by_player,
            "actions_per_info_set_max": result.action_count_max,
        },
        "strategies": {
            "current": result.current_strategy_by_player,
            "average": result.average_strategy_by_player,
        },
        "expected_utility": result.expected_utility_vector,
        "payoff_conservation_residual_max": result.payoff_conservation_residual_max,
        "positive_regret": {
            "semantics": result.positive_regret_semantics,
            "average_by_player": result.average_positive_regret_by_player,
            "max_by_player": result.max_positive_regret_by_player,
        },
        "unilateral_deviation_gain": result.unilateral_deviation_gain_by_player,
        "tolerances": result.tolerances,
        "normalization_records": list(result.normalization_records),
        "warnings": list(result.warnings),
        "oracle": {
            "requested": spec.config.request_oracle,
            "status": oracle["status"],
            "coverage": oracle["coverage"],
            "counts": counts,
            "stable_profile_count": oracle.get("stable_profile_count"),
            "warnings": list(oracle.get("warnings", [])),
        },
    }
    _finite(output)
    projected = 256 + 20 * spec.counts["decision_actions_total"] + 12 * spec.counts["branches"]
    _output_caps(output, projected, spec.workflow_limits)
    return output


def _diagnostic_failure(exc: DiagnosticContractError) -> ThreePlayerCfrFileResult:
    if exc.status == CORE_CAP_EXCEEDED:
        status = ThreePlayerCfrFileStatus.CAP_EXCEEDED
    elif exc.status == CORE_NON_REPRODUCIBLE:
        status = ThreePlayerCfrFileStatus.NON_REPRODUCIBLE
    elif exc.status == UNSUPPORTED_MODEL:
        status = ThreePlayerCfrFileStatus.ATTESTATION_FAILURE
    else:
        status = ThreePlayerCfrFileStatus.DIAGNOSTIC_FAILURE
    return _failure(_WorkflowFailure(status, "diagnostic", str(exc), exc.status))


def _execute(
    raw: bytes,
    expected_operation: str | None,
    limits: ThreePlayerCfrFileLimits,
) -> ThreePlayerCfrFileResult:
    try:
        document, spec = _base_document(raw, expected_operation, limits)
        if document["operation"] == "inspect":
            output = _inspect_output(spec)
        else:
            supplied_identity = _object(
                document["inspection_identity"],
                _INSPECTION_IDENTITY_KEYS,
                "inspection_identity",
            )
            if supplied_identity != spec.inspection_identity:
                raise _WorkflowFailure(
                    ThreePlayerCfrFileStatus.IDENTITY_MISMATCH,
                    "inspection_identity",
                    "inspection identity does not match the supplied specification",
                )
            attestation = _attestation(document["attestation"], spec)
            _preflight_analysis(spec)
            try:
                result = run_three_player_cfr_diagnostic(
                    spec.tree,
                    spec.fixed_hero_policy,
                    config=spec.config,
                    attestation=attestation,
                )
            except DiagnosticContractError as exc:
                return _diagnostic_failure(exc)
            output = _run_output(spec, result)
        return ThreePlayerCfrFileResult(ThreePlayerCfrFileStatus.SUCCESS, output, None)
    except _WorkflowFailure as exc:
        return _failure(exc)
    except Exception:
        return _failure(
            _WorkflowFailure(
                ThreePlayerCfrFileStatus.INTERNAL_FAILURE,
                "internal",
                "unexpected workflow failure",
            )
        )


def inspect_three_player_cfr_file(
    raw: bytes,
    limits: ThreePlayerCfrFileLimits = _DEFAULT_FILE_LIMITS,
) -> ThreePlayerCfrFileResult:
    """Validate one inspect document without running CFR or the oracle."""

    return _execute(raw, "inspect", limits)


def run_three_player_cfr_file(
    raw: bytes,
    limits: ThreePlayerCfrFileLimits = _DEFAULT_FILE_LIMITS,
) -> ThreePlayerCfrFileResult:
    """Run one identity-bound, human-attested, complete diagnostic."""

    return _execute(raw, "run", limits)


def process_three_player_cfr_file(
    raw: bytes,
    limits: ThreePlayerCfrFileLimits = _DEFAULT_FILE_LIMITS,
) -> ThreePlayerCfrFileResult:
    """Dispatch one strict document by its explicit inspect/run operation."""

    return _execute(raw, None, limits)


def three_player_cfr_file_json(result: ThreePlayerCfrFileResult) -> str:
    """Serialize one result as deterministic strict one-line JSON."""

    if type(result) is not ThreePlayerCfrFileResult:
        raise TypeError("result must be ThreePlayerCfrFileResult")
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
