"""Strict two-phase file adapter for supplied-profile AIoF analysis.

``inspect`` prepares canonical exact-combo support without running equity or
strategy analysis.  ``run`` verifies the identity-bound complete profile and
delegates exact ChipEV and both fixed-opponent best responses to the existing
M13 public API.  Controlled failures are fail-closed and contain no partial
template, profile, value, identity, count, or best-response payload.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Any

from .aiof_cards import (
    AiofContractError,
    AiofLimits,
    AiofStatus,
    RangeEntry,
    RangeSpec,
    WeightBasis,
    canonicalize_exact_combo,
    prepare_compatible_ranges,
)
from .aiof_chip_ev import (
    ComboActionProbability,
    HeadsUpChipEvGame,
    PushFoldRequest,
    PushFoldRunResult,
    SuppliedProfile,
    analyze_pushfold,
)
from .aiof_equity import EquityAlgorithm


__all__ = [
    "AIOF_SUPPLIED_PROFILE_FILE_FORMAT",
    "AIOF_SUPPLIED_PROFILE_TEMPLATE_ID",
    "AiofSuppliedProfileFileStatus",
    "AiofSuppliedProfileFileLimits",
    "AiofSuppliedProfileFileError",
    "AiofSuppliedProfileFileResult",
    "inspect_aiof_supplied_profile_file",
    "run_aiof_supplied_profile_file",
    "process_aiof_supplied_profile_file",
    "aiof_supplied_profile_file_json",
]


AIOF_SUPPLIED_PROFILE_FILE_FORMAT = "aiof-supplied-profile-file-v1"
AIOF_SUPPLIED_PROFILE_TEMPLATE_ID = "aiof-supplied-profile-template-sha256-v1"
_OUTPUT_ID = "aiof-supplied-profile-file-output-v1"


class AiofSuppliedProfileFileStatus(str, Enum):
    """Stable outer result classes for the versioned-file adapter."""

    SUCCESS = "SUCCESS"
    PARSE_FAILURE = "PARSE_FAILURE"
    INVALID_INPUT = "INVALID_INPUT"
    CAP_EXCEEDED = "CAP_EXCEEDED"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    PROFILE_FAILURE = "PROFILE_FAILURE"
    ANALYSIS_FAILURE = "ANALYSIS_FAILURE"
    NON_REPRODUCIBLE = "NON_REPRODUCIBLE"
    INTERNAL_FAILURE = "INTERNAL_FAILURE"


@dataclass(frozen=True)
class AiofSuppliedProfileFileLimits:
    """Caller-lowerable adapter ceilings, separate from phase-1 ceilings."""

    max_input_bytes: int = 2_000_000
    max_json_depth: int = 16
    max_total_json_values: int = 100_000
    max_range_entries_per_side: int = 1_326
    max_dead_card_items: int = 43
    max_profile_rows_per_side: int = 1_326
    max_output_records: int = 100_000
    max_output_bytes: int = 4_000_000


@dataclass(frozen=True)
class AiofSuppliedProfileFileError:
    """Bounded controlled-failure metadata with no partial computation."""

    phase: str
    message: str
    nested_status: str | None


@dataclass(frozen=True)
class AiofSuppliedProfileFileResult:
    """Exclusive success-output or failure-error wrapper."""

    status: AiofSuppliedProfileFileStatus
    output: dict[str, Any] | None
    error: AiofSuppliedProfileFileError | None


@dataclass(frozen=True)
class _ParsedSpec:
    request_id: str
    sb_range: RangeSpec
    bb_range: RangeSpec
    dead_cards: tuple[str, ...]
    game: HeadsUpChipEvGame
    analysis: dict[str, Any]
    limits: AiofLimits
    prepared: Any
    identity: dict[str, str]


_DEFAULT_FILE_LIMITS = AiofSuppliedProfileFileLimits()
_FILE_LIMIT_CEILINGS = asdict(_DEFAULT_FILE_LIMITS)
_DEFAULT_AIOF_LIMITS = AiofLimits()
_AIOF_LIMIT_CEILINGS = {
    field.name: getattr(_DEFAULT_AIOF_LIMITS, field.name) for field in fields(AiofLimits)
}
_AIOF_LIMIT_KEYS = set(_AIOF_LIMIT_CEILINGS)
_BASE_KEYS = {
    "format_version",
    "operation",
    "request_id",
    "sb_range",
    "bb_range",
    "dead_cards",
    "game",
    "analysis",
    "limits",
}
_RUN_KEYS = _BASE_KEYS | {"template_identity", "profile"}
_RANGE_ENTRY_KEYS = {"label", "weight", "weight_basis"}
_GAME_KEYS = {
    "starting_stack_sb",
    "starting_stack_bb",
    "small_blind",
    "big_blind",
    "ante",
    "fee",
    "third_party_dead_money",
    "side_pot",
}
_ANALYSIS_KEYS = {
    "equity_algorithm",
    "requested_trace_points",
    "best_response_seats",
    "deviation_tolerance",
    "seed",
    "samples",
}
_TEMPLATE_IDENTITY_KEYS = {
    "template_id",
    "semantic_sha256",
    "prepared_ranges_identity",
    "support_sha256",
}
_PROFILE_KEYS = {"sb_shove", "bb_call"}
_PROFILE_ROW_KEYS = {"combo", "probability"}


class _WorkflowFailure(ValueError):
    def __init__(
        self,
        status: AiofSuppliedProfileFileStatus,
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


def _failure(exc: _WorkflowFailure) -> AiofSuppliedProfileFileResult:
    return AiofSuppliedProfileFileResult(
        exc.status,
        None,
        AiofSuppliedProfileFileError(
            _clean_text(exc.phase, 64),
            _clean_text(str(exc), 500),
            exc.nested_status,
        ),
    )


def _validate_file_limits(limits: AiofSuppliedProfileFileLimits) -> None:
    if type(limits) is not AiofSuppliedProfileFileLimits:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            "limits",
            "workflow limits have the wrong type",
        )
    for name, ceiling in _FILE_LIMIT_CEILINGS.items():
        value = getattr(limits, name)
        if type(value) is not int or value <= 0 or value > ceiling:
            raise _WorkflowFailure(
                AiofSuppliedProfileFileStatus.INVALID_INPUT,
                "limits",
                f"{name} must be a positive int no greater than {ceiling}",
            )


def _duplicates_rejected(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _WorkflowFailure(
                AiofSuppliedProfileFileStatus.PARSE_FAILURE,
                "json",
                f"duplicate JSON key: {key}",
            )
        result[key] = value
    return result


def _parse(raw: bytes, limits: AiofSuppliedProfileFileLimits) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            "input",
            "input must be bytes",
        )
    if len(raw) > limits.max_input_bytes:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.CAP_EXCEEDED,
            "input",
            "input byte cap exceeded",
        )
    if raw.startswith(b"\xef\xbb\xbf"):
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.PARSE_FAILURE,
            "json",
            "UTF-8 BOM is not allowed",
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.PARSE_FAILURE,
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
            AiofSuppliedProfileFileStatus.CAP_EXCEEDED,
            "json",
            "JSON nesting exceeds the parser depth cap",
        ) from exc
    except (TypeError, ValueError) as exc:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.PARSE_FAILURE,
            "json",
            f"invalid JSON: {exc}",
        ) from exc
    if type(value) is not dict:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
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
                AiofSuppliedProfileFileStatus.CAP_EXCEEDED,
                "json",
                "total JSON value cap exceeded",
            )
        if depth > limits.max_json_depth:
            raise _WorkflowFailure(
                AiofSuppliedProfileFileStatus.CAP_EXCEEDED,
                "json",
                "JSON depth cap exceeded",
            )
        if type(item) is dict:
            stack.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            stack.extend((child, depth + 1) for child in item)
    return value


def _object(value: Any, keys: set[str], phase: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            phase,
            "value must be an object",
        )
    actual = set(value)
    missing = keys - actual
    unknown = actual - keys
    if missing:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            phase,
            f"missing keys: {', '.join(sorted(missing))}",
        )
    if unknown:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            phase,
            f"unknown keys: {', '.join(sorted(unknown))}",
        )
    return value


def _array(value: Any, cap: int, phase: str) -> list[Any]:
    if type(value) is not list:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            phase,
            "value must be an array",
        )
    if len(value) > cap:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.CAP_EXCEEDED,
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
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            phase,
            f"value must be a non-empty control-free string of at most {maximum} characters",
        )
    return value


def _number(value: Any, phase: str) -> float:
    if type(value) not in (int, float):
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            phase,
            "value must be a finite binary64 number",
        )
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            phase,
            "value must be a finite binary64 number",
        ) from exc
    if not math.isfinite(number):
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            phase,
            "value must be a finite binary64 number",
        )
    return number


def _plain_int(value: Any, phase: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            phase,
            f"value must be an integer in [{minimum}, {maximum}]",
        )
    return value


def _boolean(value: Any, phase: str) -> bool:
    if type(value) is not bool:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            phase,
            "value must be a boolean",
        )
    return value


def _limits(value: Any) -> AiofLimits:
    document = _object(value, _AIOF_LIMIT_KEYS, "limits")
    parsed: dict[str, int | None] = {}
    for name, ceiling in _AIOF_LIMIT_CEILINGS.items():
        item = document[name]
        if name == "max_sampling_attempts":
            if item is not None:
                raise _WorkflowFailure(
                    AiofSuppliedProfileFileStatus.INVALID_INPUT,
                    f"limits.{name}",
                    "exact v1 requires max_sampling_attempts=null",
                )
            parsed[name] = None
        else:
            assert isinstance(ceiling, int)
            parsed[name] = _plain_int(item, f"limits.{name}", 0, ceiling)
    try:
        return AiofLimits(**parsed)
    except AiofContractError as exc:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            "limits",
            str(exc),
            exc.status.value,
        ) from exc


def _range(
    value: Any,
    phase: str,
    core_limits: AiofLimits,
    workflow_limits: AiofSuppliedProfileFileLimits,
) -> RangeSpec:
    cap = min(
        core_limits.max_range_entries_per_side,
        workflow_limits.max_range_entries_per_side,
    )
    items = _array(value, cap, phase)
    if not items:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            phase,
            "range must not be empty",
        )
    entries: list[RangeEntry] = []
    for index, item in enumerate(items):
        item_phase = f"{phase}[{index}]"
        entry = _object(item, _RANGE_ENTRY_KEYS, item_phase)
        label = _text(entry["label"], f"{item_phase}.label", 8)
        weight = _number(entry["weight"], f"{item_phase}.weight")
        if weight <= 0:
            raise _WorkflowFailure(
                AiofSuppliedProfileFileStatus.INVALID_INPUT,
                f"{item_phase}.weight",
                "weight must be positive",
            )
        try:
            basis = WeightBasis(entry["weight_basis"])
        except (TypeError, ValueError) as exc:
            raise _WorkflowFailure(
                AiofSuppliedProfileFileStatus.INVALID_INPUT,
                f"{item_phase}.weight_basis",
                "unsupported weight basis",
            ) from exc
        entries.append(RangeEntry(label, weight, basis))
    return RangeSpec(tuple(entries))


def _dead_cards(
    value: Any, limits: AiofSuppliedProfileFileLimits
) -> tuple[str, ...]:
    items = _array(value, limits.max_dead_card_items, "dead_cards")
    result: list[str] = []
    for index, item in enumerate(items):
        card = _text(item, f"dead_cards[{index}]", 2)
        if len(card) != 2:
            raise _WorkflowFailure(
                AiofSuppliedProfileFileStatus.INVALID_INPUT,
                f"dead_cards[{index}]",
                "card must use the canonical two-character spelling",
            )
        result.append(card)
    return tuple(result)


def _game(value: Any) -> HeadsUpChipEvGame:
    document = _object(value, _GAME_KEYS, "game")
    game = HeadsUpChipEvGame(
        _number(document["starting_stack_sb"], "game.starting_stack_sb"),
        _number(document["starting_stack_bb"], "game.starting_stack_bb"),
        _number(document["small_blind"], "game.small_blind"),
        _number(document["big_blind"], "game.big_blind"),
        _number(document["ante"], "game.ante"),
        _number(document["fee"], "game.fee"),
        _number(document["third_party_dead_money"], "game.third_party_dead_money"),
        _boolean(document["side_pot"], "game.side_pot"),
    )
    if game.fee != 0 or game.third_party_dead_money != 0 or game.side_pot:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            "game",
            "v1 requires fee=0, third_party_dead_money=0, and side_pot=false",
        )
    if (
        game.starting_stack_sb <= 0
        or game.starting_stack_bb <= 0
        or game.small_blind <= 0
        or game.big_blind < game.small_blind
        or game.ante < 0
        or game.starting_stack_sb < game.small_blind + game.ante
        or game.starting_stack_bb < game.big_blind + game.ante
    ):
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            "game",
            "invalid stacks, blinds, or ante for the fee-zero heads-up game",
        )
    return game


def _analysis(value: Any, core_limits: AiofLimits) -> dict[str, Any]:
    document = _object(value, _ANALYSIS_KEYS, "analysis")
    try:
        algorithm = EquityAlgorithm(document["equity_algorithm"])
    except (TypeError, ValueError) as exc:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            "analysis.equity_algorithm",
            "unsupported equity algorithm",
        ) from exc
    if algorithm is not EquityAlgorithm.EXACT_EXHAUSTIVE:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            "analysis.equity_algorithm",
            "v1 requires exact_exhaustive-v1",
        )
    trace = _plain_int(
        document["requested_trace_points"],
        "analysis.requested_trace_points",
        0,
        core_limits.max_trace_points,
    )
    if trace != 0:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            "analysis.requested_trace_points",
            "v1 requires requested_trace_points=0",
        )
    seats = document["best_response_seats"]
    if type(seats) is not list or seats != ["sb", "bb"]:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            "analysis.best_response_seats",
            "v1 requires best_response_seats=[\"sb\",\"bb\"]",
        )
    tolerance = _number(document["deviation_tolerance"], "analysis.deviation_tolerance")
    if tolerance < 0:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            "analysis.deviation_tolerance",
            "deviation_tolerance must be non-negative",
        )
    if document["seed"] is not None or document["samples"] is not None:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            "analysis",
            "exact v1 requires null seed and samples",
        )
    return {
        "equity_algorithm": algorithm.value,
        "requested_trace_points": 0,
        "best_response_seats": ["sb", "bb"],
        "deviation_tolerance": tolerance,
        "seed": None,
        "samples": None,
    }


def _identity_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _WorkflowFailure(
                AiofSuppliedProfileFileStatus.INTERNAL_FAILURE,
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


def _template_support(prepared: Any) -> dict[str, list[dict[str, Any]]]:
    return {
        "sb_shove": [
            {"combo": item.combo, "probability": None} for item in prepared.sb_marginals
        ],
        "bb_call": [
            {"combo": item.combo, "probability": None} for item in prepared.bb_marginals
        ],
    }


def _template_identity(
    game: HeadsUpChipEvGame,
    analysis: dict[str, Any],
    limits: AiofLimits,
    prepared: Any,
) -> dict[str, str]:
    support = {
        "sb_shove": [item.combo for item in prepared.sb_marginals],
        "bb_call": [item.combo for item in prepared.bb_marginals],
    }
    semantic = {
        "format_version": AIOF_SUPPLIED_PROFILE_FILE_FORMAT,
        "dead_cards": prepared.dead_cards,
        "game": game,
        "analysis": analysis,
        "limits": limits,
        "prepared_ranges_identity": prepared.content_identity,
        "support": support,
    }
    return {
        "template_id": AIOF_SUPPLIED_PROFILE_TEMPLATE_ID,
        "semantic_sha256": _sha256(semantic),
        "prepared_ranges_identity": prepared.content_identity,
        "support_sha256": _sha256(support),
    }


def _base_document(
    raw: bytes,
    expected_operation: str | None,
    workflow_limits: AiofSuppliedProfileFileLimits,
) -> tuple[dict[str, Any], _ParsedSpec]:
    _validate_file_limits(workflow_limits)
    document = _parse(raw, workflow_limits)
    operation = document.get("operation")
    if operation not in ("inspect", "run"):
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            "document.operation",
            "operation must be inspect or run",
        )
    if expected_operation is not None and operation != expected_operation:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            "document.operation",
            f"operation must be {expected_operation}",
        )
    root = _object(document, _BASE_KEYS if operation == "inspect" else _RUN_KEYS, "document")
    if root["format_version"] != AIOF_SUPPLIED_PROFILE_FILE_FORMAT:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INVALID_INPUT,
            "document.format_version",
            "unsupported format_version",
        )
    request_id = _text(root["request_id"], "request_id")
    core_limits = _limits(root["limits"])
    sb_range = _range(root["sb_range"], "sb_range", core_limits, workflow_limits)
    bb_range = _range(root["bb_range"], "bb_range", core_limits, workflow_limits)
    dead_cards = _dead_cards(root["dead_cards"], workflow_limits)
    game = _game(root["game"])
    analysis = _analysis(root["analysis"], core_limits)
    try:
        prepared = prepare_compatible_ranges(sb_range, bb_range, dead_cards, core_limits)
    except AiofContractError as exc:
        status = (
            AiofSuppliedProfileFileStatus.CAP_EXCEEDED
            if exc.status is AiofStatus.CAP_EXCEEDED
            else AiofSuppliedProfileFileStatus.INVALID_INPUT
        )
        raise _WorkflowFailure(status, "prepare", str(exc), exc.status.value) from exc
    identity = _template_identity(game, analysis, core_limits, prepared)
    return root, _ParsedSpec(
        request_id,
        sb_range,
        bb_range,
        prepared.dead_cards,
        game,
        analysis,
        core_limits,
        prepared,
        identity,
    )


def _check_output_caps(
    output: dict[str, Any],
    projected_records: int,
    limits: AiofSuppliedProfileFileLimits,
) -> None:
    if projected_records > limits.max_output_records:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.CAP_EXCEEDED,
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
            AiofSuppliedProfileFileStatus.CAP_EXCEEDED,
            "output",
            "output byte cap exceeded",
        )


def _inspect_output(
    spec: _ParsedSpec, limits: AiofSuppliedProfileFileLimits
) -> dict[str, Any]:
    sb_count = len(spec.prepared.sb_marginals)
    bb_count = len(spec.prepared.bb_marginals)
    projected = 64 + 3 * (sb_count + bb_count)
    if projected > limits.max_output_records:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.CAP_EXCEEDED,
            "output",
            "output record cap exceeded",
        )
    output = {
        "format_version": AIOF_SUPPLIED_PROFILE_FILE_FORMAT,
        "operation": "inspect",
        "output_id": _OUTPUT_ID,
        "request_id": spec.request_id,
        "algorithm": EquityAlgorithm.EXACT_EXHAUSTIVE.value,
        "identity": spec.identity,
        "counts": {
            "compatible_pairs": spec.prepared.compatible_pair_count,
            "sb": {
                "projected_combos": spec.prepared.sb_range.projected_combo_count,
                "removed_combos": spec.prepared.sb_range.removed_combo_count,
                "surviving_combos": sb_count,
            },
            "bb": {
                "projected_combos": spec.prepared.bb_range.projected_combo_count,
                "removed_combos": spec.prepared.bb_range.removed_combo_count,
                "surviving_combos": bb_count,
            },
        },
        "profile_template": _template_support(spec.prepared),
    }
    _check_output_caps(output, projected, limits)
    return output


def _profile_rows(
    value: Any,
    expected: tuple[str, ...],
    phase: str,
    limits: AiofSuppliedProfileFileLimits,
) -> tuple[ComboActionProbability, ...]:
    try:
        items = _array(value, limits.max_profile_rows_per_side, phase)
    except _WorkflowFailure as exc:
        if exc.status is AiofSuppliedProfileFileStatus.CAP_EXCEEDED:
            raise
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.PROFILE_FAILURE, phase, str(exc)
        ) from exc
    if len(items) != len(expected):
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.PROFILE_FAILURE,
            phase,
            "profile support is incomplete",
        )
    probabilities: dict[str, float] = {}
    for index, item in enumerate(items):
        item_phase = f"{phase}[{index}]"
        try:
            row = _object(item, _PROFILE_ROW_KEYS, item_phase)
        except _WorkflowFailure as exc:
            raise _WorkflowFailure(
                AiofSuppliedProfileFileStatus.PROFILE_FAILURE,
                item_phase,
                str(exc),
            ) from exc
        try:
            combo = canonicalize_exact_combo(_text(row["combo"], f"{item_phase}.combo", 4))
        except (AiofContractError, _WorkflowFailure) as exc:
            raise _WorkflowFailure(
                AiofSuppliedProfileFileStatus.PROFILE_FAILURE,
                f"{item_phase}.combo",
                "profile combo must be a canonical exact combo",
            ) from exc
        if combo != row["combo"] or combo in probabilities:
            raise _WorkflowFailure(
                AiofSuppliedProfileFileStatus.PROFILE_FAILURE,
                item_phase,
                "profile combo is noncanonical or duplicated",
            )
        try:
            probability = _number(
                row["probability"], f"{item_phase}.probability"
            )
        except _WorkflowFailure as exc:
            raise _WorkflowFailure(
                AiofSuppliedProfileFileStatus.PROFILE_FAILURE,
                f"{item_phase}.probability",
                str(exc),
            ) from exc
        if not 0 <= probability <= 1:
            raise _WorkflowFailure(
                AiofSuppliedProfileFileStatus.PROFILE_FAILURE,
                f"{item_phase}.probability",
                "profile probability must be in [0, 1]",
            )
        probabilities[combo] = probability
    if set(probabilities) != set(expected):
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.PROFILE_FAILURE,
            phase,
            "profile support has missing or extra combos",
        )
    return tuple(ComboActionProbability(combo, probabilities[combo]) for combo in expected)


def _profile(
    value: Any,
    spec: _ParsedSpec,
    limits: AiofSuppliedProfileFileLimits,
) -> SuppliedProfile:
    document = _object(value, _PROFILE_KEYS, "profile")
    sb_expected = tuple(item.combo for item in spec.prepared.sb_marginals)
    bb_expected = tuple(item.combo for item in spec.prepared.bb_marginals)
    return SuppliedProfile(
        _profile_rows(document["sb_shove"], sb_expected, "profile.sb_shove", limits),
        _profile_rows(document["bb_call"], bb_expected, "profile.bb_call", limits),
    )


def _analysis_failure(result: PushFoldRunResult) -> AiofSuppliedProfileFileResult:
    if result.status is AiofStatus.CAP_EXCEEDED:
        status = AiofSuppliedProfileFileStatus.CAP_EXCEEDED
    elif result.status is AiofStatus.NON_REPRODUCIBLE:
        status = AiofSuppliedProfileFileStatus.NON_REPRODUCIBLE
    elif result.status is AiofStatus.INVALID_STRATEGY:
        status = AiofSuppliedProfileFileStatus.PROFILE_FAILURE
    else:
        status = AiofSuppliedProfileFileStatus.ANALYSIS_FAILURE
    return _failure(
        _WorkflowFailure(
            status,
            "analysis",
            result.error_message or "analysis failed without error metadata",
            result.status.value,
        )
    )


def _best_response_output(response: Any) -> dict[str, Any]:
    return {
        "seat": response.seat,
        "supplied_profile_value": response.supplied_profile_value,
        "best_response_value": response.best_response_value,
        "raw_gain": response.raw_gain,
        "rows": [
            {
                "combo": row.combo,
                "compatible_probability": row.compatible_probability,
                "information_reach_probability": row.information_reach_probability,
                "action_values": [
                    {"action": action, "value": value}
                    for action, value in row.action_values
                ],
                "best_actions": list(row.best_actions),
                "supplied_action_probability": row.supplied_action_probability,
                "raw_gain": row.raw_gain,
            }
            for row in response.rows
        ],
    }


def _run_output(
    spec: _ParsedSpec,
    profile: SuppliedProfile,
    result: PushFoldRunResult,
    limits: AiofSuppliedProfileFileLimits,
) -> dict[str, Any]:
    analysis = result.analysis
    if (
        result.status is not AiofStatus.SUCCESS
        or analysis is None
        or result.error_message is not None
        or analysis.algorithm is not EquityAlgorithm.EXACT_EXHAUSTIVE
        or tuple(response.seat for response in analysis.best_responses) != ("sb", "bb")
        or analysis.accepted_samples is not None
        or analysis.profile_sample_variance is not None
        or analysis.profile_standard_error is not None
    ):
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.INTERNAL_FAILURE,
            "analysis",
            "analysis success invariant failed",
        )
    sb_count = len(profile.sb_shove)
    bb_count = len(profile.bb_call)
    response_rows = sum(len(response.rows) for response in analysis.best_responses)
    projected = 160 + 3 * (sb_count + bb_count) + 18 * response_rows
    if projected > limits.max_output_records:
        raise _WorkflowFailure(
            AiofSuppliedProfileFileStatus.CAP_EXCEEDED,
            "output",
            "output record cap exceeded",
        )
    counts = analysis.outcome_counts
    probabilities = analysis.outcome_probabilities
    output = {
        "format_version": AIOF_SUPPLIED_PROFILE_FILE_FORMAT,
        "operation": "run",
        "output_id": _OUTPUT_ID,
        "request_id": spec.request_id,
        "identity": spec.identity,
        "algorithm": analysis.algorithm.value,
        "chip_accounting_id": analysis.chip_accounting_id,
        "input_identity": analysis.input_identity,
        "prepared_ranges_identity": analysis.prepared_ranges_identity,
        "compatible_pair_count": analysis.compatible_pair_count,
        "board_evaluations": analysis.board_evaluations,
        "outcome_counts": {
            "wins": counts.wins,
            "losses": counts.losses,
            "ties": counts.ties,
            "trials": counts.trials,
        },
        "outcome_probabilities": {
            "win": probabilities.win,
            "loss": probabilities.loss,
            "tie": probabilities.tie,
        },
        "profile": {
            "sb_shove": [
                {"combo": row.combo, "probability": row.probability}
                for row in profile.sb_shove
            ],
            "bb_call": [
                {"combo": row.combo, "probability": row.probability}
                for row in profile.bb_call
            ],
        },
        "profile_values": {
            "sb": analysis.profile_value_sb,
            "bb": analysis.profile_value_bb,
            "conservation_sum": analysis.profile_value_sb + analysis.profile_value_bb,
        },
        "sampling": {
            "accepted_samples": analysis.accepted_samples,
            "rejected_hole_draws": analysis.rejected_hole_draws,
            "profile_sample_variance": analysis.profile_sample_variance,
            "profile_standard_error": analysis.profile_standard_error,
            "seed": None,
            "requested_samples": None,
        },
        "best_responses": [
            _best_response_output(response) for response in analysis.best_responses
        ],
    }
    _check_output_caps(output, projected, limits)
    return output


def _execute(
    raw: bytes,
    expected_operation: str | None,
    limits: AiofSuppliedProfileFileLimits,
) -> AiofSuppliedProfileFileResult:
    try:
        document, spec = _base_document(raw, expected_operation, limits)
        if document["operation"] == "inspect":
            output = _inspect_output(spec, limits)
        else:
            supplied_identity = _object(
                document["template_identity"],
                _TEMPLATE_IDENTITY_KEYS,
                "template_identity",
            )
            if supplied_identity != spec.identity:
                raise _WorkflowFailure(
                    AiofSuppliedProfileFileStatus.IDENTITY_MISMATCH,
                    "template_identity",
                    "template identity does not match the supplied specification",
                )
            profile = _profile(document["profile"], spec, limits)
            run = analyze_pushfold(
                PushFoldRequest(
                    spec.sb_range,
                    spec.bb_range,
                    spec.dead_cards,
                    EquityAlgorithm.EXACT_EXHAUSTIVE,
                    spec.limits,
                    0,
                    spec.game,
                    profile,
                    ("sb", "bb"),
                    spec.analysis["deviation_tolerance"],
                    None,
                    None,
                )
            )
            if run.status is not AiofStatus.SUCCESS:
                return _analysis_failure(run)
            output = _run_output(spec, profile, run, limits)
        return AiofSuppliedProfileFileResult(
            AiofSuppliedProfileFileStatus.SUCCESS,
            output,
            None,
        )
    except _WorkflowFailure as exc:
        return _failure(exc)
    except Exception:
        return _failure(
            _WorkflowFailure(
                AiofSuppliedProfileFileStatus.INTERNAL_FAILURE,
                "internal",
                "unexpected workflow failure",
            )
        )


def inspect_aiof_supplied_profile_file(
    raw: bytes,
    limits: AiofSuppliedProfileFileLimits = _DEFAULT_FILE_LIMITS,
) -> AiofSuppliedProfileFileResult:
    """Prepare an identity-bound complete profile template without analysis."""

    return _execute(raw, "inspect", limits)


def run_aiof_supplied_profile_file(
    raw: bytes,
    limits: AiofSuppliedProfileFileLimits = _DEFAULT_FILE_LIMITS,
) -> AiofSuppliedProfileFileResult:
    """Validate a complete profile and run exact ChipEV plus both fixed BRs."""

    return _execute(raw, "run", limits)


def process_aiof_supplied_profile_file(
    raw: bytes,
    limits: AiofSuppliedProfileFileLimits = _DEFAULT_FILE_LIMITS,
) -> AiofSuppliedProfileFileResult:
    """Dispatch one strict document by its explicit inspect/run operation."""

    return _execute(raw, None, limits)


def aiof_supplied_profile_file_json(result: AiofSuppliedProfileFileResult) -> str:
    """Serialize one result as deterministic strict one-line JSON."""

    if type(result) is not AiofSuppliedProfileFileResult:
        raise TypeError("result must be AiofSuppliedProfileFileResult")
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
