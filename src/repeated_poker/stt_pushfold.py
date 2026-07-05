"""STT SB-vs-BB push/fold scenario loading and game construction.

This module builds a small abstract single-table-tournament preflop spot:
everyone folds to the small blind, SB may ``shove`` or ``fold``, and BB may
``call`` or ``fold`` against a shove. Terminal utilities are modelled tournament
prize-EV deltas from Malmuth-Harville ICM, not chip EV and not real tournament
predictions.

The showdown matchup inputs are abstract bucket probabilities supplied by the
scenario. No real-card evaluator, range parser, card-removal engine, side-pot
logic, non-all-in sizing, Future-ICM, FGS, or tournament simulation is
implemented here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .exact_response import solve_exact_response
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
    require_finite,
    validate_hero_strategy,
    validate_tree,
    validate_villain_strategy,
)
from .icm import DEFAULT_MAX_ICM_ORDERINGS, calculate_icm_equities

STT_PUSHFOLD_FORMAT_VERSION = "stt_pushfold-1"
SUPPORTED_STT_PUSHFOLD_FORMAT_VERSIONS = (STT_PUSHFOLD_FORMAT_VERSION,)
DEFAULT_MAX_STT_MATCHUPS = 100_000

_TOLERANCE = 1e-9
_SB_ACTIONS = ("shove", "fold")
_BB_ACTIONS = ("call", "fold")
_HERO_SEATS = ("sb", "bb")
_BASELINE_VILLAIN_AUTO = "auto_best_response"
_BASELINE_VILLAIN_EXPLICIT = "explicit"


@dataclass(frozen=True)
class SttPushFoldRangeBucket:
    """One abstract weighted hand bucket for either SB or BB."""

    bucket_id: str
    weight: float


@dataclass(frozen=True)
class SttPushFoldOutcome:
    """Showdown probabilities for one SB bucket versus one BB bucket."""

    sb_win: float
    bb_win: float
    chop: float


@dataclass(frozen=True)
class SttPushFoldRepeatedConfig:
    """Optional repeated-game configuration for a fixed STT spot."""

    horizon: Optional[int]
    discount: float


@dataclass(frozen=True)
class SttPushFoldScenario:
    """A validated ``stt_pushfold-1`` JSON scenario.

    Ranges are independent weighted abstract buckets. ``outcome_matrix`` (or the
    scalar ``sb_win_probability_matrix`` shorthand) supplies showdown
    probabilities directly; the builder never evaluates cards. ``hero_seat``
    selects which side is locked by the candidate-analysis pipeline.
    """

    format_version: str
    scenario_id: str
    description: str
    stacks: List[float]
    sb_index: int
    bb_index: int
    prizes: List[float]
    small_blind: float
    big_blind: float
    ante: float
    hero_seat: str
    sb_range: List[SttPushFoldRangeBucket]
    bb_range: List[SttPushFoldRangeBucket]
    outcome_matrix: Dict[str, Dict[str, SttPushFoldOutcome]]
    outcome_input_type: str
    baseline_sb_strategy: Optional[Dict[str, Dict[str, float]]]
    baseline_bb_strategy: Optional[Dict[str, Dict[str, float]]]
    shift_amounts: Optional[List[float]]
    repeated: Optional[SttPushFoldRepeatedConfig]
    max_simultaneous_info_sets: int = 1
    max_icm_orderings: int = DEFAULT_MAX_ICM_ORDERINGS
    max_matchups: int = DEFAULT_MAX_STT_MATCHUPS


@dataclass(frozen=True)
class SttPushFoldBuildResult:
    """The game objects and pipeline inputs built from an STT scenario."""

    tree: GameTree
    baseline_hero_strategy: HeroStrategy
    baseline_villain_strategy: VillainStrategy
    baseline_villain_source: str
    terminal_reveals: Dict[str, Optional[Tuple[str, ...]]]
    metadata: dict
    shift_amounts: Optional[List[float]]
    repeated: Optional[SttPushFoldRepeatedConfig]
    max_simultaneous_info_sets: int = 1


def _as_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    number = float(value)
    require_finite(number, name)
    return number


def _require_positive(value: float, name: str) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


def _require_non_negative(value: float, name: str) -> float:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return value


def _parse_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return value


def _parse_positive_int(value, name: str) -> int:
    parsed = _parse_int(value, name)
    if parsed < 1:
        raise ValueError(f"{name} must be at least 1, got {parsed}")
    return parsed


def _validate_action_distribution(distribution, legal_actions, name: str) -> Dict[str, float]:
    if not isinstance(distribution, dict):
        raise ValueError(f"{name} must be a mapping of action to probability")
    unknown = set(distribution) - set(legal_actions)
    if unknown:
        raise ValueError(f"{name} has unknown actions {sorted(unknown)}")
    probabilities: Dict[str, float] = {}
    total = 0.0
    for action in legal_actions:
        probability = _as_number(distribution.get(action, 0.0), f"{name}[{action!r}]")
        _require_non_negative(probability, f"{name}[{action!r}]")
        probabilities[action] = probability
        total += probability
    if abs(total - 1.0) > _TOLERANCE:
        raise ValueError(f"{name} probabilities sum to {total}, expected 1")
    return probabilities


def _parse_format_version(data: dict) -> str:
    value = data.get("format_version")
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(
            "format_version must be the string "
            f"{STT_PUSHFOLD_FORMAT_VERSION!r}, got {value!r}"
        )
    if value not in SUPPORTED_STT_PUSHFOLD_FORMAT_VERSIONS:
        raise ValueError(
            f"unsupported format_version {value!r}; supported versions are "
            f"{list(SUPPORTED_STT_PUSHFOLD_FORMAT_VERSIONS)}"
        )
    return value


def _parse_stacks(data) -> List[float]:
    if isinstance(data, (str, bytes)):
        raise ValueError("stacks must be a list of positive numbers")
    try:
        raw_values = list(data)
    except TypeError as exc:
        raise ValueError("stacks must be a list of positive numbers") from exc
    if len(raw_values) < 2:
        raise ValueError("stacks must contain at least two players")
    stacks: List[float] = []
    for index, value in enumerate(raw_values):
        stack = _require_positive(_as_number(value, f"stacks[{index}]"), f"stacks[{index}]")
        stacks.append(stack)
    return stacks


def _parse_prizes(data, player_count: int) -> List[float]:
    if isinstance(data, (str, bytes)):
        raise ValueError("prizes must be a list of numbers")
    try:
        raw_values = list(data)
    except TypeError as exc:
        raise ValueError("prizes must be a list of numbers") from exc
    if not raw_values:
        raise ValueError("prizes must be non-empty")
    if len(raw_values) > player_count:
        raise ValueError("prizes length must be at most the number of players")
    prizes: List[float] = []
    previous = math.inf
    for index, value in enumerate(raw_values):
        prize = _require_non_negative(
            _as_number(value, f"prizes[{index}]"), f"prizes[{index}]"
        )
        if prize > previous:
            raise ValueError("prizes must be non-increasing")
        prizes.append(prize)
        previous = prize
    return prizes


def _parse_range(data, name: str) -> List[SttPushFoldRangeBucket]:
    if not isinstance(data, list):
        raise ValueError(f"{name} must be a list of bucket objects")
    if not data:
        raise ValueError(f"{name} must contain at least one bucket")
    buckets: List[SttPushFoldRangeBucket] = []
    seen: set[str] = set()
    total = 0.0
    for index, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise ValueError(f"{name}[{index}] must be an object")
        bucket_id = raw.get("id")
        if not isinstance(bucket_id, str) or not bucket_id:
            raise ValueError(f"{name}[{index}].id must be a non-empty string")
        if bucket_id in seen:
            raise ValueError(f"{name} has duplicate id {bucket_id!r}")
        seen.add(bucket_id)
        weight = _require_positive(
            _as_number(raw.get("weight"), f"{name}[{index}].weight"),
            f"{name}[{index}].weight",
        )
        buckets.append(SttPushFoldRangeBucket(bucket_id=bucket_id, weight=weight))
        total += weight
    if abs(total - 1.0) > _TOLERANCE:
        raise ValueError(f"{name} weights sum to {total}, expected 1")
    return buckets


def _validate_matrix_keys(data, sb_range, bb_range, name: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be an object keyed by SB bucket id")
    expected_sb = {bucket.bucket_id for bucket in sb_range}
    expected_bb = {bucket.bucket_id for bucket in bb_range}
    actual_sb = set(data)
    missing_sb = expected_sb - actual_sb
    if missing_sb:
        raise ValueError(f"{name} is missing SB ids {sorted(missing_sb)}")
    extra_sb = actual_sb - expected_sb
    if extra_sb:
        raise ValueError(f"{name} has unknown SB ids {sorted(extra_sb)}")
    for sb_bucket in sb_range:
        row = data[sb_bucket.bucket_id]
        if not isinstance(row, dict):
            raise ValueError(
                f"{name}[{sb_bucket.bucket_id!r}] must be an object keyed by BB id"
            )
        actual_bb = set(row)
        missing_bb = expected_bb - actual_bb
        if missing_bb:
            raise ValueError(
                f"{name}[{sb_bucket.bucket_id!r}] is missing BB ids "
                f"{sorted(missing_bb)}"
            )
        extra_bb = actual_bb - expected_bb
        if extra_bb:
            raise ValueError(
                f"{name}[{sb_bucket.bucket_id!r}] has unknown BB ids "
                f"{sorted(extra_bb)}"
            )


def _parse_outcome_matrix(data, sb_range, bb_range) -> Dict[str, Dict[str, SttPushFoldOutcome]]:
    _validate_matrix_keys(data, sb_range, bb_range, "outcome_matrix")
    matrix: Dict[str, Dict[str, SttPushFoldOutcome]] = {}
    for sb_bucket in sb_range:
        parsed_row: Dict[str, SttPushFoldOutcome] = {}
        row = data[sb_bucket.bucket_id]
        for bb_bucket in bb_range:
            cell = row[bb_bucket.bucket_id]
            if not isinstance(cell, dict):
                raise ValueError(
                    "outcome_matrix"
                    f"[{sb_bucket.bucket_id!r}][{bb_bucket.bucket_id!r}] "
                    "must be an object with sb_win, bb_win, and chop"
                )
            sb_win = _require_non_negative(
                _as_number(
                    cell.get("sb_win"),
                    "outcome_matrix"
                    f"[{sb_bucket.bucket_id!r}][{bb_bucket.bucket_id!r}].sb_win",
                ),
                "outcome_matrix"
                f"[{sb_bucket.bucket_id!r}][{bb_bucket.bucket_id!r}].sb_win",
            )
            bb_win = _require_non_negative(
                _as_number(
                    cell.get("bb_win"),
                    "outcome_matrix"
                    f"[{sb_bucket.bucket_id!r}][{bb_bucket.bucket_id!r}].bb_win",
                ),
                "outcome_matrix"
                f"[{sb_bucket.bucket_id!r}][{bb_bucket.bucket_id!r}].bb_win",
            )
            chop = _require_non_negative(
                _as_number(
                    cell.get("chop"),
                    "outcome_matrix"
                    f"[{sb_bucket.bucket_id!r}][{bb_bucket.bucket_id!r}].chop",
                ),
                "outcome_matrix"
                f"[{sb_bucket.bucket_id!r}][{bb_bucket.bucket_id!r}].chop",
            )
            total = sb_win + bb_win + chop
            if abs(total - 1.0) > _TOLERANCE:
                raise ValueError(
                    "outcome_matrix"
                    f"[{sb_bucket.bucket_id!r}][{bb_bucket.bucket_id!r}] "
                    f"probabilities sum to {total}, expected 1"
                )
            parsed_row[bb_bucket.bucket_id] = SttPushFoldOutcome(
                sb_win=sb_win, bb_win=bb_win, chop=chop
            )
        matrix[sb_bucket.bucket_id] = parsed_row
    return matrix


def _parse_sb_win_probability_matrix(data, sb_range, bb_range):
    _validate_matrix_keys(data, sb_range, bb_range, "sb_win_probability_matrix")
    matrix: Dict[str, Dict[str, SttPushFoldOutcome]] = {}
    for sb_bucket in sb_range:
        row = data[sb_bucket.bucket_id]
        parsed_row: Dict[str, SttPushFoldOutcome] = {}
        for bb_bucket in bb_range:
            probability = _as_number(
                row[bb_bucket.bucket_id],
                "sb_win_probability_matrix"
                f"[{sb_bucket.bucket_id!r}][{bb_bucket.bucket_id!r}]",
            )
            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    "sb_win_probability_matrix"
                    f"[{sb_bucket.bucket_id!r}][{bb_bucket.bucket_id!r}] "
                    f"must be within [0, 1], got {probability!r}"
                )
            parsed_row[bb_bucket.bucket_id] = SttPushFoldOutcome(
                sb_win=probability,
                bb_win=1.0 - probability,
                chop=0.0,
            )
        matrix[sb_bucket.bucket_id] = parsed_row
    return matrix


def _parse_baseline_strategy(data, buckets, legal_actions, name: str):
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be an object keyed by bucket id")
    expected = {bucket.bucket_id for bucket in buckets}
    actual = set(data)
    missing = expected - actual
    if missing:
        raise ValueError(f"{name} is missing bucket ids {sorted(missing)}")
    extra = actual - expected
    if extra:
        raise ValueError(f"{name} has unknown bucket ids {sorted(extra)}")
    return {
        bucket.bucket_id: _validate_action_distribution(
            data[bucket.bucket_id],
            legal_actions,
            f"{name}[{bucket.bucket_id!r}]",
        )
        for bucket in buckets
    }


def _parse_repeated(data) -> Optional[SttPushFoldRepeatedConfig]:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("repeated must be an object")
    if "horizon" in data:
        horizon = _parse_positive_int(data["horizon"], "repeated.horizon")
    else:
        horizon = None
    discount = _as_number(data.get("discount", 1.0), "repeated.discount")
    if not 0.0 < discount <= 1.0:
        raise ValueError(
            f"repeated.discount must satisfy 0 < discount <= 1, got {discount!r}"
        )
    return SttPushFoldRepeatedConfig(horizon=horizon, discount=discount)


def _parse_candidates(data) -> Tuple[Optional[List[float]], int]:
    if data is None:
        return None, 1
    if not isinstance(data, dict):
        raise ValueError("candidates must be an object")
    raw_shift_amounts = data.get("shift_amounts")
    shift_amounts: Optional[List[float]]
    if raw_shift_amounts is None:
        shift_amounts = None
    else:
        if not isinstance(raw_shift_amounts, list):
            raise ValueError("candidates.shift_amounts must be a list")
        shift_amounts = []
        for index, raw in enumerate(raw_shift_amounts):
            amount = _require_positive(
                _as_number(raw, f"candidates.shift_amounts[{index}]"),
                f"candidates.shift_amounts[{index}]",
            )
            shift_amounts.append(amount)

    if "max_simultaneous_info_sets" in data:
        max_simultaneous = _parse_int(
            data["max_simultaneous_info_sets"],
            "candidates.max_simultaneous_info_sets",
        )
        if max_simultaneous < 1 or max_simultaneous > 2:
            raise ValueError(
                "candidates.max_simultaneous_info_sets must be 1 or 2, got "
                f"{max_simultaneous}"
            )
    else:
        max_simultaneous = 1
    return shift_amounts, max_simultaneous


def _validate_posting_capacity(
    stacks: List[float],
    sb_index: int,
    bb_index: int,
    small_blind: float,
    big_blind: float,
    ante: float,
) -> None:
    for index, stack in enumerate(stacks):
        if index in (sb_index, bb_index):
            continue
        if stack <= ante:
            raise ValueError(
                f"stacks[{index}] must exceed ante so the bystander can post it"
            )
    if stacks[sb_index] < ante + small_blind:
        raise ValueError(
            "SB stack must cover ante + small_blind; side pots and partial "
            "blind posting are out of scope"
        )
    if stacks[bb_index] < ante + big_blind:
        raise ValueError(
            "BB stack must cover ante + big_blind; side pots and partial "
            "blind posting are out of scope"
        )


def stt_pushfold_scenario_from_dict(data) -> SttPushFoldScenario:
    """Parse and validate a ``stt_pushfold-1`` scenario from a plain dict."""

    if not isinstance(data, dict):
        raise ValueError("scenario must be a JSON object")

    format_version = _parse_format_version(data)
    scenario_id = data.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("scenario_id must be a non-empty string")
    description = data.get("description", "")
    if not isinstance(description, str):
        raise ValueError("description must be a string")

    stacks = _parse_stacks(data.get("stacks"))
    player_count = len(stacks)
    sb_index = _parse_int(data.get("sb_index"), "sb_index")
    bb_index = _parse_int(data.get("bb_index"), "bb_index")
    if not 0 <= sb_index < player_count:
        raise ValueError(f"sb_index must be within [0, {player_count}), got {sb_index}")
    if not 0 <= bb_index < player_count:
        raise ValueError(f"bb_index must be within [0, {player_count}), got {bb_index}")
    if sb_index == bb_index:
        raise ValueError("sb_index and bb_index must be different")

    prizes = _parse_prizes(data.get("prizes"), player_count)
    small_blind = _require_positive(
        _as_number(data.get("small_blind"), "small_blind"), "small_blind"
    )
    big_blind = _require_positive(
        _as_number(data.get("big_blind"), "big_blind"), "big_blind"
    )
    if small_blind > big_blind:
        raise ValueError("small_blind must be less than or equal to big_blind")
    ante = _require_non_negative(_as_number(data.get("ante", 0.0), "ante"), "ante")
    _validate_posting_capacity(stacks, sb_index, bb_index, small_blind, big_blind, ante)

    hero_seat = data.get("hero_seat")
    if hero_seat not in _HERO_SEATS:
        raise ValueError(f"hero_seat must be one of {_HERO_SEATS}, got {hero_seat!r}")

    sb_range = _parse_range(data.get("sb_range"), "sb_range")
    bb_range = _parse_range(data.get("bb_range"), "bb_range")
    max_matchups = _parse_positive_int(
        data.get("max_matchups", DEFAULT_MAX_STT_MATCHUPS), "max_matchups"
    )
    matchup_count = len(sb_range) * len(bb_range)
    if matchup_count > max_matchups:
        raise ValueError(
            f"STT matchup count {matchup_count} exceeds max_matchups={max_matchups}"
        )

    if "equity_matrix" in data:
        raise ValueError(
            "equity_matrix is not supported for STT push/fold; use "
            "outcome_matrix or sb_win_probability_matrix"
        )
    has_outcome = "outcome_matrix" in data
    has_scalar = "sb_win_probability_matrix" in data
    if has_outcome == has_scalar:
        raise ValueError(
            "provide exactly one of outcome_matrix or sb_win_probability_matrix"
        )
    if has_outcome:
        outcome_matrix = _parse_outcome_matrix(
            data["outcome_matrix"], sb_range, bb_range
        )
        outcome_input_type = "outcome_matrix"
    else:
        outcome_matrix = _parse_sb_win_probability_matrix(
            data["sb_win_probability_matrix"], sb_range, bb_range
        )
        outcome_input_type = "sb_win_probability_matrix"

    baseline_sb_strategy = _parse_baseline_strategy(
        data.get("baseline_sb_strategy"), sb_range, _SB_ACTIONS, "baseline_sb_strategy"
    )
    baseline_bb_strategy = _parse_baseline_strategy(
        data.get("baseline_bb_strategy"), bb_range, _BB_ACTIONS, "baseline_bb_strategy"
    )
    if hero_seat == "sb" and baseline_sb_strategy is None:
        raise ValueError("baseline_sb_strategy is required when hero_seat is 'sb'")
    if hero_seat == "bb" and baseline_bb_strategy is None:
        raise ValueError("baseline_bb_strategy is required when hero_seat is 'bb'")

    shift_amounts, max_simultaneous = _parse_candidates(data.get("candidates"))
    repeated = _parse_repeated(data.get("repeated"))
    max_icm_orderings = _parse_positive_int(
        data.get("max_icm_orderings", DEFAULT_MAX_ICM_ORDERINGS),
        "max_icm_orderings",
    )

    return SttPushFoldScenario(
        format_version=format_version,
        scenario_id=scenario_id,
        description=description,
        stacks=stacks,
        sb_index=sb_index,
        bb_index=bb_index,
        prizes=prizes,
        small_blind=small_blind,
        big_blind=big_blind,
        ante=ante,
        hero_seat=hero_seat,
        sb_range=sb_range,
        bb_range=bb_range,
        outcome_matrix=outcome_matrix,
        outcome_input_type=outcome_input_type,
        baseline_sb_strategy=baseline_sb_strategy,
        baseline_bb_strategy=baseline_bb_strategy,
        shift_amounts=shift_amounts,
        repeated=repeated,
        max_simultaneous_info_sets=max_simultaneous,
        max_icm_orderings=max_icm_orderings,
        max_matchups=max_matchups,
    )


def load_stt_pushfold_scenario_json(path: Union[str, Path]) -> SttPushFoldScenario:
    """Load and validate an STT push/fold scenario from a JSON file."""

    text = Path(path).read_text(encoding="utf-8")
    return stt_pushfold_scenario_from_dict(json.loads(text))


def _sb_info_set(bucket_id: str) -> str:
    return f"SB:{bucket_id}"


def _bb_info_set(bucket_id: str) -> str:
    return f"BB_vs_shove:{bucket_id}"


def _matchup_suffix(sb_id: str, bb_id: str) -> str:
    return f"{sb_id}__{bb_id}"


def _terminal_stack_vectors(scenario: SttPushFoldScenario) -> Dict[str, Tuple[float, ...]]:
    stacks = scenario.stacks
    sb_index = scenario.sb_index
    bb_index = scenario.bb_index
    small_blind = scenario.small_blind
    big_blind = scenario.big_blind
    ante = scenario.ante
    player_count = len(stacks)
    all_in_amount = min(stacks[sb_index] - ante, stacks[bb_index] - ante)

    def bystanders_post_ante() -> List[float]:
        return [
            stack - ante if index not in (sb_index, bb_index) else stack
            for index, stack in enumerate(stacks)
        ]

    result: Dict[str, Tuple[float, ...]] = {}

    sb_fold = bystanders_post_ante()
    sb_fold[sb_index] = stacks[sb_index] - ante - small_blind
    sb_fold[bb_index] = stacks[bb_index] + small_blind + (player_count - 1) * ante
    result["sb_fold"] = tuple(sb_fold)

    shove_bb_fold = bystanders_post_ante()
    shove_bb_fold[sb_index] = stacks[sb_index] + big_blind + (player_count - 1) * ante
    shove_bb_fold[bb_index] = stacks[bb_index] - ante - big_blind
    result["shove_bb_fold"] = tuple(shove_bb_fold)

    call_sb_win = bystanders_post_ante()
    call_sb_win[sb_index] = stacks[sb_index] + all_in_amount + (player_count - 1) * ante
    call_sb_win[bb_index] = stacks[bb_index] - ante - all_in_amount
    result["call_sb_win"] = tuple(call_sb_win)

    call_bb_win = bystanders_post_ante()
    call_bb_win[sb_index] = stacks[sb_index] - ante - all_in_amount
    call_bb_win[bb_index] = stacks[bb_index] + all_in_amount + (player_count - 1) * ante
    result["call_bb_win"] = tuple(call_bb_win)

    call_chop = bystanders_post_ante()
    chop_ante_share = (player_count - 2) * ante / 2.0
    call_chop[sb_index] = stacks[sb_index] + chop_ante_share
    call_chop[bb_index] = stacks[bb_index] + chop_ante_share
    result["call_chop"] = tuple(call_chop)

    starting_total = math.fsum(stacks)
    for name, terminal_stacks in result.items():
        total = math.fsum(terminal_stacks)
        if abs(total - starting_total) > _TOLERANCE:
            raise ValueError(
                f"terminal stack vector {name!r} sums to {total}, "
                f"expected {starting_total}"
            )
        for index, stack in enumerate(terminal_stacks):
            if stack < -_TOLERANCE:
                raise ValueError(
                    f"terminal stack vector {name!r} has negative stack "
                    f"at index {index}: {stack}"
                )
    return result


def _payoff_from_deltas(
    scenario: SttPushFoldScenario, deltas: List[float]
) -> Tuple[float, float, float]:
    sb_delta = deltas[scenario.sb_index]
    bb_delta = deltas[scenario.bb_index]
    residual = math.fsum(
        delta
        for index, delta in enumerate(deltas)
        if index not in (scenario.sb_index, scenario.bb_index)
    )
    if scenario.hero_seat == "sb":
        return sb_delta, bb_delta, residual
    return bb_delta, sb_delta, residual


def _terminal_payoffs(scenario: SttPushFoldScenario):
    base_equities = calculate_icm_equities(
        scenario.stacks,
        scenario.prizes,
        max_orderings=scenario.max_icm_orderings,
    )
    stack_vectors = _terminal_stack_vectors(scenario)
    payoffs: Dict[str, Tuple[float, float, float]] = {}
    equity_cache: Dict[str, List[float]] = {}
    for name, terminal_stacks in stack_vectors.items():
        equities = calculate_icm_equities(
            terminal_stacks,
            scenario.prizes,
            max_orderings=scenario.max_icm_orderings,
        )
        equity_cache[name] = equities
        deltas = [equity - base for equity, base in zip(equities, base_equities)]
        payoffs[name] = _payoff_from_deltas(scenario, deltas)
    return base_equities, stack_vectors, equity_cache, payoffs


def _terminal(node_id: str, payoff: Tuple[float, float, float]) -> TerminalNode:
    return TerminalNode(
        node_id=node_id,
        hero_ev=payoff[0],
        villain_ev=payoff[1],
        house_rake=payoff[2],
    )


def _weighted_payoff(
    outcome: SttPushFoldOutcome,
    payoffs: Dict[str, Tuple[float, float, float]],
) -> Tuple[float, float, float]:
    weights = (
        (outcome.sb_win, payoffs["call_sb_win"]),
        (outcome.bb_win, payoffs["call_bb_win"]),
        (outcome.chop, payoffs["call_chop"]),
    )
    return tuple(
        math.fsum(probability * payoff[index] for probability, payoff in weights)
        for index in range(3)
    )


def _hero_strategy_from_side(scenario: SttPushFoldScenario) -> HeroStrategy:
    if scenario.hero_seat == "sb":
        probabilities = {
            _sb_info_set(bucket.bucket_id): dict(
                scenario.baseline_sb_strategy[bucket.bucket_id]
            )
            for bucket in scenario.sb_range
        }
    else:
        probabilities = {
            _bb_info_set(bucket.bucket_id): dict(
                scenario.baseline_bb_strategy[bucket.bucket_id]
            )
            for bucket in scenario.bb_range
        }
    return HeroStrategy(probabilities=probabilities)


def _explicit_villain_strategy_from_side(
    scenario: SttPushFoldScenario,
) -> Optional[VillainStrategy]:
    if scenario.hero_seat == "sb":
        if scenario.baseline_bb_strategy is None:
            return None
        probabilities = {
            _bb_info_set(bucket.bucket_id): dict(
                scenario.baseline_bb_strategy[bucket.bucket_id]
            )
            for bucket in scenario.bb_range
        }
    else:
        if scenario.baseline_sb_strategy is None:
            return None
        probabilities = {
            _sb_info_set(bucket.bucket_id): dict(
                scenario.baseline_sb_strategy[bucket.bucket_id]
            )
            for bucket in scenario.sb_range
        }
    return VillainStrategy(probabilities=probabilities)


def _villain_baseline_best_response(
    tree: GameTree, baseline_hero_strategy: HeroStrategy
) -> VillainStrategy:
    response = solve_exact_response(
        tree,
        baseline_hero_strategy,
        allow_negative_residual=True,
    )
    chosen = response.best_response_strategies[0]
    villain_info_sets = collect_villain_info_sets(tree)
    return VillainStrategy(
        probabilities={
            info_set: {
                action: (1.0 if action == chosen[info_set] else 0.0)
                for action in actions
            }
            for info_set, actions in villain_info_sets.items()
        }
    )


def _base_metadata(
    scenario: SttPushFoldScenario,
    *,
    base_equities: List[float],
    stack_vectors: Dict[str, Tuple[float, ...]],
    terminal_equities: Dict[str, List[float]],
) -> dict:
    return {
        "format_version": scenario.format_version,
        "scenario_id": scenario.scenario_id,
        "description": scenario.description,
        "model_kind": "stt_pushfold_icm",
        "value_unit": "modelled_tournament_prize_ev_delta",
        "hero_seat": scenario.hero_seat,
        "sb_index": scenario.sb_index,
        "bb_index": scenario.bb_index,
        "stacks": list(scenario.stacks),
        "prizes": list(scenario.prizes),
        "small_blind": scenario.small_blind,
        "big_blind": scenario.big_blind,
        "ante": scenario.ante,
        "sb_buckets": [
            {"id": bucket.bucket_id, "weight": bucket.weight}
            for bucket in scenario.sb_range
        ],
        "bb_buckets": [
            {"id": bucket.bucket_id, "weight": bucket.weight}
            for bucket in scenario.bb_range
        ],
        "outcome_input_type": scenario.outcome_input_type,
        "initial_icm_equities": list(base_equities),
        "terminal_stack_vectors": {
            name: list(stacks) for name, stacks in stack_vectors.items()
        },
        "terminal_icm_equities": {
            name: list(equities) for name, equities in terminal_equities.items()
        },
        "max_icm_orderings": scenario.max_icm_orderings,
        "max_matchups": scenario.max_matchups,
    }


def build_stt_pushfold_game(
    scenario: SttPushFoldScenario,
    *,
    tolerance: float = _TOLERANCE,
) -> SttPushFoldBuildResult:
    """Build ``GameTree`` and baseline strategies for an STT push/fold scenario.

    The terminal payoff triple is ``(hero_ev, villain_ev, residual)`` in
    modelled prize-EV delta units. For STT this third slot is the bystander
    prize-EV delta, not house rake, and it may be negative. The tree is therefore
    validated with ``allow_negative_residual=True``.
    """

    if not isinstance(scenario, SttPushFoldScenario):
        raise TypeError(
            "scenario must be a SttPushFoldScenario, got "
            f"{type(scenario).__name__}"
        )
    require_finite(tolerance, "tolerance")
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance!r}")

    base_equities, stack_vectors, terminal_equities, payoffs = _terminal_payoffs(
        scenario
    )
    children = []
    terminal_reveals: Dict[str, Optional[Tuple[str, ...]]] = {}
    for sb_bucket in scenario.sb_range:
        sb_info_set = _sb_info_set(sb_bucket.bucket_id)
        for bb_bucket in scenario.bb_range:
            bb_info_set = _bb_info_set(bb_bucket.bucket_id)
            suffix = _matchup_suffix(sb_bucket.bucket_id, bb_bucket.bucket_id)

            sb_fold_id = f"T_sb_fold::{suffix}"
            shove_fold_id = f"T_shove_bb_fold::{suffix}"
            call_id = f"T_call::{suffix}"
            sb_fold = _terminal(sb_fold_id, payoffs["sb_fold"])
            shove_fold = _terminal(shove_fold_id, payoffs["shove_bb_fold"])
            call = _terminal(
                call_id,
                _weighted_payoff(
                    scenario.outcome_matrix[sb_bucket.bucket_id][bb_bucket.bucket_id],
                    payoffs,
                ),
            )

            bb_actions = (("call", call), ("fold", shove_fold))
            if scenario.hero_seat == "bb":
                bb_node = HeroNode(
                    node_id=f"bb::{suffix}",
                    info_set=bb_info_set,
                    actions=bb_actions,
                )
                sb_node_cls = VillainNode
            else:
                bb_node = VillainNode(
                    node_id=f"bb::{suffix}",
                    info_set=bb_info_set,
                    actions=bb_actions,
                )
                sb_node_cls = HeroNode

            sb_node = sb_node_cls(
                node_id=f"sb::{suffix}",
                info_set=sb_info_set,
                actions=(("shove", bb_node), ("fold", sb_fold)),
            )
            children.append((sb_bucket.weight * bb_bucket.weight, sb_node))
            terminal_reveals[sb_fold_id] = None
            terminal_reveals[shove_fold_id] = None
            terminal_reveals[call_id] = (sb_bucket.bucket_id, bb_bucket.bucket_id)

    tree = GameTree(root=ChanceNode(node_id="stt_matchup", children=tuple(children)))
    validate_tree(
        tree,
        tolerance=tolerance,
        allow_negative_residual=True,
    )

    baseline_hero_strategy = _hero_strategy_from_side(scenario)
    validate_hero_strategy(tree, baseline_hero_strategy, tolerance=tolerance)

    explicit_villain = _explicit_villain_strategy_from_side(scenario)
    if explicit_villain is None:
        baseline_villain_strategy = _villain_baseline_best_response(
            tree, baseline_hero_strategy
        )
        baseline_villain_source = _BASELINE_VILLAIN_AUTO
    else:
        validate_villain_strategy(tree, explicit_villain, tolerance=tolerance)
        baseline_villain_strategy = explicit_villain
        baseline_villain_source = _BASELINE_VILLAIN_EXPLICIT

    metadata = _base_metadata(
        scenario,
        base_equities=base_equities,
        stack_vectors=stack_vectors,
        terminal_equities=terminal_equities,
    )
    metadata["baseline_villain_source"] = baseline_villain_source
    metadata["hero_info_sets"] = sorted(collect_hero_info_sets(tree))
    metadata["villain_info_sets"] = sorted(collect_villain_info_sets(tree))

    return SttPushFoldBuildResult(
        tree=tree,
        baseline_hero_strategy=baseline_hero_strategy,
        baseline_villain_strategy=baseline_villain_strategy,
        baseline_villain_source=baseline_villain_source,
        terminal_reveals=terminal_reveals,
        metadata=metadata,
        shift_amounts=scenario.shift_amounts,
        repeated=scenario.repeated,
        max_simultaneous_info_sets=scenario.max_simultaneous_info_sets,
    )
