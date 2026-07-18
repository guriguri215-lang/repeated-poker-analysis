"""Strict versioned-file adapter for the exact AIoF rational-lift workflow.

The module is deliberately an adapter over the existing M13 public API.  It
does not implement a solver, change the declared game, or expose runtime-bound
identities.  Controlled failures are fail-closed and never contain partial
strategy, payoff, pivot, or witness payloads.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from enum import Enum
from fractions import Fraction
from typing import Any

from .aiof_cards import RangeEntry, RangeSpec, WeightBasis
from .aiof_chip_ev import HeadsUpChipEvGame
from .aiof_equity import EquityAlgorithm
from .aiof_strategy import (
    AiofStrategyAlgorithm,
    AiofStrategyLimits,
    AiofStrategyStatus,
    RationalStrategyRequest,
    RationalStrategyRunResult,
    generate_rational_lift_strategy,
)


__all__ = [
    "AIOF_RATIONAL_LIFT_FILE_FORMAT",
    "AiofRationalLiftFileStatus",
    "AiofRationalLiftFileLimits",
    "AiofRationalLiftFileError",
    "AiofRationalLiftFileResult",
    "run_aiof_rational_lift_file",
    "aiof_rational_lift_file_json",
]


AIOF_RATIONAL_LIFT_FILE_FORMAT = "aiof-rational-lift-file-v1"


class AiofRationalLiftFileStatus(str, Enum):
    """Stable outer classifications for the versioned-file adapter."""

    SUCCESS = "SUCCESS"
    PARSE_FAILURE = "PARSE_FAILURE"
    INVALID_INPUT = "INVALID_INPUT"
    CAP_EXCEEDED = "CAP_EXCEEDED"
    STRATEGY_FAILURE = "STRATEGY_FAILURE"
    NON_REPRODUCIBLE = "NON_REPRODUCIBLE"
    INTERNAL_FAILURE = "INTERNAL_FAILURE"


@dataclass(frozen=True)
class AiofRationalLiftFileLimits:
    """Caller-lowerable adapter ceilings, separate from solver ceilings."""

    max_input_bytes: int = 1_000_000
    max_json_depth: int = 16
    max_total_json_values: int = 10_000
    max_range_entries_per_side: int = 64
    max_dead_card_items: int = 43
    max_output_records: int = 10_000
    max_output_bytes: int = 1_000_000


@dataclass(frozen=True)
class AiofRationalLiftFileError:
    """Bounded controlled-failure metadata with no partial computation."""

    phase: str
    message: str
    nested_status: str | None


@dataclass(frozen=True)
class AiofRationalLiftFileResult:
    """Exclusive success-output or failure-error wrapper."""

    status: AiofRationalLiftFileStatus
    output: dict[str, Any] | None
    error: AiofRationalLiftFileError | None


_DEFAULT_FILE_LIMITS = AiofRationalLiftFileLimits()
_FILE_LIMIT_CEILINGS = asdict(_DEFAULT_FILE_LIMITS)
_DEFAULT_STRATEGY_LIMITS = AiofStrategyLimits()
_STRATEGY_LIMIT_CEILINGS = {
    field.name: getattr(_DEFAULT_STRATEGY_LIMITS, field.name)
    for field in fields(AiofStrategyLimits)
}
_STRATEGY_LIMIT_KEYS = set(_STRATEGY_LIMIT_CEILINGS)
_TOP_LEVEL_KEYS = {
    "format_version",
    "request_id",
    "sb_range",
    "bb_range",
    "dead_cards",
    "game",
    "strategy",
    "limits",
}
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
_STRATEGY_KEYS = {
    "algorithm",
    "equity_algorithm",
    "claim_epsilon",
    "display_tie_tolerance",
    "requested_trace_points",
    "run_reference_oracle",
    "run_phase1_float_diagnostic",
    "seed",
    "samples",
}
_RANGE_ENTRY_KEYS = {"label", "weight", "weight_basis"}


class _WorkflowFailure(ValueError):
    def __init__(
        self,
        status: AiofRationalLiftFileStatus,
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


def _failure(exc: _WorkflowFailure) -> AiofRationalLiftFileResult:
    return AiofRationalLiftFileResult(
        exc.status,
        None,
        AiofRationalLiftFileError(
            _clean_text(exc.phase, 64),
            _clean_text(str(exc), 500),
            exc.nested_status,
        ),
    )


def _validate_file_limits(limits: AiofRationalLiftFileLimits) -> None:
    if type(limits) is not AiofRationalLiftFileLimits:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.INVALID_INPUT,
            "limits",
            "workflow limits have the wrong type",
        )
    for name, ceiling in _FILE_LIMIT_CEILINGS.items():
        value = getattr(limits, name)
        if type(value) is not int or value <= 0 or value > ceiling:
            raise _WorkflowFailure(
                AiofRationalLiftFileStatus.INVALID_INPUT,
                "limits",
                f"{name} must be a positive int no greater than {ceiling}",
            )


def _duplicates_rejected(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _WorkflowFailure(
                AiofRationalLiftFileStatus.PARSE_FAILURE,
                "json",
                f"duplicate JSON key: {key}",
            )
        result[key] = value
    return result


def _parse(raw: bytes, limits: AiofRationalLiftFileLimits) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.INVALID_INPUT,
            "input",
            "input must be bytes",
        )
    if len(raw) > limits.max_input_bytes:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.CAP_EXCEEDED,
            "input",
            "input byte cap exceeded",
        )
    if raw.startswith(b"\xef\xbb\xbf"):
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.PARSE_FAILURE,
            "json",
            "UTF-8 BOM is not allowed",
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.PARSE_FAILURE,
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
            AiofRationalLiftFileStatus.CAP_EXCEEDED,
            "json",
            "JSON nesting exceeds the parser depth cap",
        ) from exc
    except (TypeError, ValueError) as exc:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.PARSE_FAILURE,
            "json",
            f"invalid JSON: {exc}",
        ) from exc
    if type(value) is not dict:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.INVALID_INPUT,
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
                AiofRationalLiftFileStatus.CAP_EXCEEDED,
                "json",
                "total JSON value cap exceeded",
            )
        if depth > limits.max_json_depth:
            raise _WorkflowFailure(
                AiofRationalLiftFileStatus.CAP_EXCEEDED,
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
            AiofRationalLiftFileStatus.INVALID_INPUT,
            phase,
            "value must be an object",
        )
    actual = set(value)
    missing = keys - actual
    unknown = actual - keys
    if missing:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.INVALID_INPUT,
            phase,
            f"missing keys: {', '.join(sorted(missing))}",
        )
    if unknown:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.INVALID_INPUT,
            phase,
            f"unknown keys: {', '.join(sorted(unknown))}",
        )
    return value


def _array(value: Any, cap: int, phase: str) -> list[Any]:
    if type(value) is not list:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.INVALID_INPUT,
            phase,
            "value must be an array",
        )
    if len(value) > cap:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.CAP_EXCEEDED,
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
            AiofRationalLiftFileStatus.INVALID_INPUT,
            phase,
            f"value must be a non-empty control-free string of at most {maximum} characters",
        )
    return value


def _number(value: Any, phase: str) -> float:
    if type(value) not in (int, float):
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.INVALID_INPUT,
            phase,
            "value must be a finite binary64 number",
        )
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.INVALID_INPUT,
            phase,
            "value must be a finite binary64 number",
        ) from exc
    if not math.isfinite(converted):
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.INVALID_INPUT,
            phase,
            "value must be a finite binary64 number",
        )
    return converted


def _plain_int(value: Any, phase: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.INVALID_INPUT,
            phase,
            f"value must be an integer in [{minimum}, {maximum}]",
        )
    return value


def _boolean(value: Any, phase: str) -> bool:
    if type(value) is not bool:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.INVALID_INPUT,
            phase,
            "value must be a boolean",
        )
    return value


def _canonical_fraction(value: Any, phase: str) -> Fraction:
    text = _text(value, phase, 200)
    try:
        fraction = Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.INVALID_INPUT,
            phase,
            "value must be a canonical rational string",
        ) from exc
    if fraction < 0 or str(fraction) != text:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.INVALID_INPUT,
            phase,
            "value must be a canonical non-negative rational string",
        )
    return fraction


def _strategy_limits(value: Any) -> AiofStrategyLimits:
    document = _object(value, _STRATEGY_LIMIT_KEYS, "limits")
    parsed: dict[str, int] = {}
    for name, ceiling in _STRATEGY_LIMIT_CEILINGS.items():
        minimum = 0 if name == "max_trace_points" else 1
        parsed[name] = _plain_int(document[name], f"limits.{name}", minimum, ceiling)
    return AiofStrategyLimits(**parsed)


def _range(
    value: Any,
    phase: str,
    strategy_limits: AiofStrategyLimits,
    workflow_limits: AiofRationalLiftFileLimits,
) -> RangeSpec:
    cap = min(
        strategy_limits.max_solver_combos_per_side,
        workflow_limits.max_range_entries_per_side,
    )
    items = _array(value, cap, phase)
    if not items:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.INVALID_INPUT,
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
                AiofRationalLiftFileStatus.INVALID_INPUT,
                f"{item_phase}.weight",
                "weight must be positive",
            )
        try:
            basis = WeightBasis(entry["weight_basis"])
        except (TypeError, ValueError) as exc:
            raise _WorkflowFailure(
                AiofRationalLiftFileStatus.INVALID_INPUT,
                f"{item_phase}.weight_basis",
                "unsupported weight basis",
            ) from exc
        entries.append(RangeEntry(label, weight, basis))
    return RangeSpec(tuple(entries))


def _dead_cards(value: Any, limits: AiofRationalLiftFileLimits) -> tuple[str, ...]:
    items = _array(value, limits.max_dead_card_items, "dead_cards")
    cards: list[str] = []
    for index, item in enumerate(items):
        card = _text(item, f"dead_cards[{index}]", 2)
        if len(card) != 2:
            raise _WorkflowFailure(
                AiofRationalLiftFileStatus.INVALID_INPUT,
                f"dead_cards[{index}]",
                "card must use the canonical two-character spelling",
            )
        cards.append(card)
    return tuple(cards)


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
            AiofRationalLiftFileStatus.INVALID_INPUT,
            "game",
            "v1 requires fee=0, third_party_dead_money=0, and side_pot=false",
        )
    return game


def _request(
    document: dict[str, Any],
    workflow_limits: AiofRationalLiftFileLimits,
) -> tuple[str, RationalStrategyRequest]:
    root = _object(document, _TOP_LEVEL_KEYS, "document")
    if root["format_version"] != AIOF_RATIONAL_LIFT_FILE_FORMAT:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.INVALID_INPUT,
            "document",
            "unsupported format_version",
        )
    request_id = _text(root["request_id"], "request_id")
    limits = _strategy_limits(root["limits"])
    sb_range = _range(root["sb_range"], "sb_range", limits, workflow_limits)
    bb_range = _range(root["bb_range"], "bb_range", limits, workflow_limits)
    dead_cards = _dead_cards(root["dead_cards"], workflow_limits)
    game = _game(root["game"])

    strategy = _object(root["strategy"], _STRATEGY_KEYS, "strategy")
    try:
        algorithm = AiofStrategyAlgorithm(strategy["algorithm"])
    except (TypeError, ValueError) as exc:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.INVALID_INPUT,
            "strategy.algorithm",
            "unsupported strategy algorithm",
        ) from exc
    if algorithm is not AiofStrategyAlgorithm.COMPACT_RATIONAL_LP:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.INVALID_INPUT,
            "strategy.algorithm",
            "v1 requires the compact rational LP algorithm",
        )
    try:
        equity_algorithm = EquityAlgorithm(strategy["equity_algorithm"])
    except (TypeError, ValueError) as exc:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.INVALID_INPUT,
            "strategy.equity_algorithm",
            "unsupported equity algorithm",
        ) from exc
    if equity_algorithm is not EquityAlgorithm.EXACT_EXHAUSTIVE:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.INVALID_INPUT,
            "strategy.equity_algorithm",
            "v1 requires exact_exhaustive-v1",
        )
    claim_epsilon = _canonical_fraction(
        strategy["claim_epsilon"], "strategy.claim_epsilon"
    )
    display_tie_tolerance = _canonical_fraction(
        strategy["display_tie_tolerance"], "strategy.display_tie_tolerance"
    )
    requested_trace_points = _plain_int(
        strategy["requested_trace_points"],
        "strategy.requested_trace_points",
        0,
        limits.max_trace_points,
    )
    run_reference_oracle = _boolean(
        strategy["run_reference_oracle"], "strategy.run_reference_oracle"
    )
    run_phase1_float_diagnostic = _boolean(
        strategy["run_phase1_float_diagnostic"],
        "strategy.run_phase1_float_diagnostic",
    )
    if run_reference_oracle or run_phase1_float_diagnostic:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.INVALID_INPUT,
            "strategy",
            "v1 does not expose reference-oracle or phase-1 diagnostics",
        )
    if strategy["seed"] is not None or strategy["samples"] is not None:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.INVALID_INPUT,
            "strategy",
            "exact v1 requires null seed and samples",
        )
    return request_id, RationalStrategyRequest(
        sb_range,
        bb_range,
        dead_cards,
        equity_algorithm,
        game,
        limits,
        algorithm,
        claim_epsilon,
        display_tie_tolerance,
        requested_trace_points,
        False,
        False,
        None,
        None,
    )


def _fraction_text(value: Fraction) -> str:
    if type(value) is not Fraction:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.INTERNAL_FAILURE,
            "output",
            "unexpected non-rational result",
        )
    return str(value)


def _json_value_count(value: Any, cap: int) -> int:
    count = 0
    stack = [value]
    while stack:
        item = stack.pop()
        count += 1
        if count > cap:
            raise _WorkflowFailure(
                AiofRationalLiftFileStatus.CAP_EXCEEDED,
                "output",
                "output record cap exceeded",
            )
        if type(item) is dict:
            stack.extend(item.values())
        elif type(item) is list:
            stack.extend(item)
    return count


def _success_output(
    request_id: str,
    run: RationalStrategyRunResult,
    limits: AiofRationalLiftFileLimits,
) -> dict[str, Any]:
    strategy = run.strategy_result
    if (
        run.status is not AiofStrategyStatus.SUCCESS
        or strategy is None
        or run.error is not None
        or strategy.oracle_comparison is not None
        or strategy.phase1_float_diagnostic is not None
    ):
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.INTERNAL_FAILURE,
            "strategy",
            "strategy success invariant failed",
        )
    witness = strategy.witness
    gains = witness.gains
    projected_records = 64 + 3 * (
        len(strategy.profile.sb_shove) + len(strategy.profile.bb_call)
    )
    if projected_records > limits.max_output_records:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.CAP_EXCEEDED,
            "output",
            "output record cap exceeded",
        )
    output: dict[str, Any] = {
        "format_version": AIOF_RATIONAL_LIFT_FILE_FORMAT,
        "request_id": request_id,
        "algorithm": strategy.algorithm.value,
        "game_id": strategy.game_id,
        "verifier_id": strategy.verifier_id,
        "prepared_ranges_identity": strategy.prepared_ranges_identity,
        "payoff_identity": strategy.payoff_identity,
        "semantic_identity": strategy.semantic_identity,
        "input_identity": strategy.input_identity,
        "payoff_cell_count": strategy.payoff_cell_count,
        "exact_board_evaluations": strategy.exact_board_evaluations,
        "profile": {
            "content_identity": strategy.profile.content_identity,
            "sb_shove": [
                {"combo": item.combo, "probability": _fraction_text(item.probability)}
                for item in strategy.profile.sb_shove
            ],
            "bb_call": [
                {"combo": item.combo, "probability": _fraction_text(item.probability)}
                for item in strategy.profile.bb_call
            ],
        },
        "witness": {
            "verification_identity": witness.verification_identity,
            "claim_kind": f"{strategy.game_id}:{witness.claim_kind.value}",
            "claim_epsilon": _fraction_text(witness.claim_epsilon),
            "numeric_error_bound": _fraction_text(witness.numeric_error_bound),
            "lower_objective": _fraction_text(witness.lower_objective),
            "upper_objective": _fraction_text(witness.upper_objective),
            "primal_feasible": witness.primal_feasible,
            "dual_feasible": witness.dual_feasible,
            "zero_objective_gap": witness.zero_objective_gap,
            "gains": {
                "profile_value": _fraction_text(gains.profile_value),
                "sb_best_response_value": _fraction_text(gains.sb_best_response_value),
                "bb_best_response_sb_value": _fraction_text(
                    gains.bb_best_response_sb_value
                ),
                "g_sb": _fraction_text(gains.g_sb),
                "g_bb": _fraction_text(gains.g_bb),
                "nash_conv": _fraction_text(gains.nash_conv),
                "max_unilateral_gain": _fraction_text(gains.max_unilateral_gain),
                "value_lower": _fraction_text(gains.value_lower),
                "value_upper": _fraction_text(gains.value_upper),
            },
        },
        "oracle_comparison": None,
        "phase1_float_diagnostic": None,
    }
    _json_value_count(output, limits.max_output_records)
    encoded = json.dumps(
        output,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > limits.max_output_bytes:
        raise _WorkflowFailure(
            AiofRationalLiftFileStatus.CAP_EXCEEDED,
            "output",
            "output byte cap exceeded",
        )
    return output


def _strategy_failure(run: RationalStrategyRunResult) -> AiofRationalLiftFileResult:
    nested_status = run.status.value
    if run.status in {
        AiofStrategyStatus.CAP_EXCEEDED,
        AiofStrategyStatus.EXACT_ARITHMETIC_CAP_EXCEEDED,
        AiofStrategyStatus.SOLVER_LIMIT_REACHED,
    }:
        status = AiofRationalLiftFileStatus.CAP_EXCEEDED
    elif run.status is AiofStrategyStatus.NON_REPRODUCIBLE:
        status = AiofRationalLiftFileStatus.NON_REPRODUCIBLE
    else:
        status = AiofRationalLiftFileStatus.STRATEGY_FAILURE
    error = run.error
    phase = error.phase if error is not None else "strategy"
    message = error.message if error is not None else "strategy failed without error metadata"
    return _failure(_WorkflowFailure(status, phase, message, nested_status))


def run_aiof_rational_lift_file(
    raw: bytes,
    limits: AiofRationalLiftFileLimits = _DEFAULT_FILE_LIMITS,
) -> AiofRationalLiftFileResult:
    """Run one strict versioned document through the public exact strategy API."""

    try:
        _validate_file_limits(limits)
        document = _parse(raw, limits)
        request_id, request = _request(document, limits)
        run = generate_rational_lift_strategy(request)
        if run.status is not AiofStrategyStatus.SUCCESS:
            return _strategy_failure(run)
        output = _success_output(request_id, run, limits)
        return AiofRationalLiftFileResult(
            AiofRationalLiftFileStatus.SUCCESS,
            output,
            None,
        )
    except _WorkflowFailure as exc:
        return _failure(exc)
    except Exception:
        return _failure(
            _WorkflowFailure(
                AiofRationalLiftFileStatus.INTERNAL_FAILURE,
                "internal",
                "unexpected workflow failure",
            )
        )


def aiof_rational_lift_file_json(result: AiofRationalLiftFileResult) -> str:
    """Serialize one result as deterministic strict one-line JSON."""

    if type(result) is not AiofRationalLiftFileResult:
        raise TypeError("result must be AiofRationalLiftFileResult")
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
