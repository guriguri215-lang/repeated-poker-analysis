"""Build an abstract river steal game from a JSON scenario input.

This is the v1 scenario-input layer. It reads a small, abstract river spot from
JSON (or a plain dict) and builds the existing ``GameTree`` / ``HeroStrategy``
objects, so a user can drive the analysis pipeline from an external file instead
of hand-written Python.

Scope of v1 (intentionally narrow, but named to extend later toward hand-range
inputs):

* a single river spot with two players;
* OOP (the free-responding Villain) acts first and may ``check`` or ``bet``;
* IP (the locked Hero) faces a bet and may ``call`` or ``fold``;
* the showdown result is ``"chop"``, ``"hero"`` (IP wins), or ``"villain"``
  (OOP wins);
* one fixed bet size, a rake rate and cap, and the initial committed chips;
* a baseline Hero strategy at the ``IP_vs_bet`` information set;
* optional candidate shift amounts and optional repeated-game horizons/discount.

Three input modes are supported and are mutually exclusive:

* *single-hand mode* (the original v1 form): a single ``showdown`` result and a
  single ``baseline_hero_strategy`` at the ``IP_vs_bet`` information set;
* *Hero-range-only mode*: a ``hero_range`` list of abstract weighted hands, each
  with its own ``showdown`` result and ``baseline_strategy``. A chance node
  draws the Hero bucket; Villain shares one ``OOP_river`` information set across
  all buckets (it does not observe Hero's hand), while Hero gets a per-hand
  information set ``IP_vs_bet::<hand_id>``;
* *matrix mode* (the hand range model): a ``hero_range`` (without per-hand
  ``showdown``), a ``villain_range``, and exactly one matchup matrix keyed by
  ``[hero_id][villain_id]`` -- either a ``showdown_matrix`` of discrete
  ``chop`` / ``hero`` / ``villain`` outcomes, or an ``equity_matrix`` of Hero
  pot shares before rake in ``[0, 1]`` (``1.0`` = Hero wins, ``0.5`` = chop,
  ``0.0`` = Villain wins). A chance node draws the ``(hero, villain)`` pair with
  probability ``hero_weight * villain_weight``; Villain gets a per-bucket
  information set ``OOP_river::<villain_id>`` (shared across Hero buckets,
  because Villain knows its own bucket but not Hero's), and Hero keeps a
  per-bucket information set ``IP_vs_bet::<hero_id>`` (shared across Villain
  buckets, because Hero knows its own bucket but not Villain's).

Mixing modes raises ``ValueError`` (for example ``hero_range`` together with a
top-level ``showdown`` / ``baseline_hero_strategy``, ``villain_range`` without
any matrix, or both ``showdown_matrix`` and ``equity_matrix`` at once).

These ranges are not a real card range parser. The matchup outcomes (discrete
results or equities) are given directly as abstract inputs -- for example an
``equity_matrix`` precomputed by an external tool -- with no real card or hand
evaluation in this module. Hero and Villain hand ids must be disjoint in matrix
mode: although the ``IP_vs_bet::`` / ``OOP_river::`` prefixes would namespace
them, v1 rejects an id shared between the two ranges to keep the matrix keys and
any future cross-references unambiguous.

For the river action tree, the simple-tree modes (single-hand, Hero-range-only,
and matrix mode without ``betting_tree``) treat an OOP ``check`` as an immediate
check-check showdown, with no IP action after the check. Matrix mode may instead
add an optional ``betting_tree`` (see :mod:`repeated_poker.river_betting_tree`),
under which an OOP ``check`` leads to an IP ``check`` / ``bet`` and an OOP ``bet``
allows an IP ``call`` / ``fold`` / ``raise``. Re-raises, arbitrary nested betting
trees, multiple bet sizes per node, and street transitions remain future
extensions, even though the core ``GameTree`` itself allows arbitrary action
labels. The core ``game`` / ``payoffs`` / ``exact_response`` / ``repeated``
modules are reused unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from .exact_response import solve_exact_response
from .game import (
    ChanceNode,
    GameTree,
    HeroNode,
    HeroStrategy,
    VillainNode,
    VillainStrategy,
    collect_villain_info_sets,
    require_finite,
    validate_hero_strategy,
    validate_tree,
)
from .river_betting_tree import build_betting_tree_from_scenario
from .payoffs import (
    CHOP,
    HERO,
    VILLAIN,
    make_equity_showdown_terminal,
    make_fold_terminal,
    make_showdown_terminal,
)

_TOLERANCE = 1e-9
_SHOWDOWN_RESULTS = (CHOP, HERO, VILLAIN)
_OOP_INFO_SET = "OOP_river"
_IP_INFO_SET = "IP_vs_bet"
_OOP_ACTIONS = ("check", "bet")
_IP_ACTIONS = ("call", "fold")
# Betting-tree mode (river one-street tree) Hero decision points.
_IP_AFTER_CHECK_ACTIONS = ("check", "bet")
_IP_VS_BET_ACTIONS = ("call", "fold", "raise")


# ---------------------------------------------------------------------------
# Scenario data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiverScenarioRake:
    """Rake rule for a river scenario."""

    rate: float
    cap: Optional[float]


@dataclass(frozen=True)
class RiverScenarioInitialCommitment:
    """Chips each player has already committed before the river decision."""

    hero: float
    villain: float


@dataclass(frozen=True)
class RiverScenarioRepeatedConfig:
    """Optional repeated-game configuration."""

    horizons: Optional[List[int]]
    discount: float


@dataclass(frozen=True)
class RiverScenarioHeroRangeHand:
    """One abstract weighted hand in a Hero range.

    ``baseline_strategy`` is the Hero ``call`` / ``fold`` distribution at this
    hand's own ``IP_vs_bet::<hand_id>`` information set (simple action tree).
    ``showdown`` is the per-hand fixed outcome in Hero-range-only mode and is
    ``None`` in matrix mode (where the outcome comes from the matrix).

    In betting-tree mode ``baseline_strategy`` is ``None`` and
    ``baseline_strategies`` instead carries the Hero distributions at the two
    betting-tree decision points: ``"after_oop_check"`` (``check`` / ``bet``) and
    ``"vs_oop_bet"`` (``call`` / ``fold`` / ``raise``).
    """

    hand_id: str
    weight: float
    showdown: Optional[str]
    baseline_strategy: Optional[Dict[str, float]]
    baseline_strategies: Optional[Dict[str, Dict[str, float]]] = None


@dataclass(frozen=True)
class RiverScenarioHeroRange:
    """An abstract weighted Hero range whose weights sum to one."""

    hands: List[RiverScenarioHeroRangeHand]


@dataclass(frozen=True)
class RiverScenarioVillainRangeHand:
    """One abstract weighted hand in a Villain range (matrix mode only)."""

    hand_id: str
    weight: float


@dataclass(frozen=True)
class RiverScenarioVillainRange:
    """An abstract weighted Villain range whose weights sum to one."""

    hands: List[RiverScenarioVillainRangeHand]


@dataclass(frozen=True)
class RiverScenarioBettingTree:
    """River one-street betting-tree sizes (betting-tree mode only).

    ``ip_raise_size`` is the *total* chips each player commits once IP's raise is
    called (not the raise increment): if OOP bets ``oop_bet_size`` and IP raises
    to ``ip_raise_size`` total, a called raise leaves both players invested at
    ``initial_commitment + ip_raise_size``. It must exceed ``oop_bet_size``.
    """

    oop_bet_size: float
    ip_bet_after_check_size: float
    ip_raise_size: float


@dataclass(frozen=True)
class RiverScenario:
    """A validated abstract river steal scenario.

    Three input modes are supported and are mutually exclusive:

    * *single-hand mode*: ``showdown`` and ``baseline_hero_strategy`` are set and
      every range field is ``None``;
    * *Hero-range-only mode*: ``hero_range`` is set with a per-hand ``showdown``,
      while ``villain_range`` / ``showdown_matrix`` are ``None``;
    * *matrix mode*: ``hero_range`` (without per-hand ``showdown``),
      ``villain_range``, and exactly one of ``showdown_matrix`` (discrete
      ``chop`` / ``hero`` / ``villain`` outcomes) or ``equity_matrix`` (Hero pot
      share before rake, in ``[0, 1]``) are set.

    :meth:`is_range_mode` is true in both range modes; :meth:`is_matrix_mode`
    distinguishes matrix mode.
    """

    scenario_id: str
    description: str
    rake: RiverScenarioRake
    initial_commitment: RiverScenarioInitialCommitment
    bet_size: float
    showdown: Optional[str]
    baseline_hero_strategy: Optional[Dict[str, Dict[str, float]]]
    shift_amounts: Optional[List[float]]
    repeated: Optional[RiverScenarioRepeatedConfig]
    hero_range: Optional[RiverScenarioHeroRange] = None
    villain_range: Optional[RiverScenarioVillainRange] = None
    showdown_matrix: Optional[Dict[str, Dict[str, str]]] = None
    equity_matrix: Optional[Dict[str, Dict[str, float]]] = None
    betting_tree: Optional[RiverScenarioBettingTree] = None

    @property
    def is_range_mode(self) -> bool:
        return self.hero_range is not None

    @property
    def is_matrix_mode(self) -> bool:
        return self.villain_range is not None

    @property
    def is_betting_tree_mode(self) -> bool:
        return self.betting_tree is not None


@dataclass(frozen=True)
class RiverScenarioBuildResult:
    """The game objects and pipeline inputs built from a scenario.

    ``baseline_villain_strategy`` is the pure Villain best response to the
    baseline Hero strategy, i.e. the single-hand-equilibrium aggressor policy, so
    the result is directly usable as a baseline profile for the pipeline.
    """

    scenario_id: str
    description: str
    tree: GameTree
    baseline_hero_strategy: HeroStrategy
    baseline_villain_strategy: VillainStrategy
    shift_amounts: Optional[List[float]]
    repeated: Optional[RiverScenarioRepeatedConfig]
    metadata: dict


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _as_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    number = float(value)
    require_finite(number, name)
    return number


def _require_non_negative(value: float, name: str) -> float:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return value


def _require_positive(value: float, name: str) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


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


def _parse_rake(data) -> RiverScenarioRake:
    if not isinstance(data, dict):
        raise ValueError("rake must be an object with 'rate' and optional 'cap'")
    rate = _as_number(data.get("rate"), "rake.rate")
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rake.rate must be within [0, 1], got {rate!r}")
    cap_raw = data.get("cap", None)
    if cap_raw is None:
        cap: Optional[float] = None
    else:
        cap = _require_non_negative(_as_number(cap_raw, "rake.cap"), "rake.cap")
    return RiverScenarioRake(rate=rate, cap=cap)


def _parse_initial_commitment(data) -> RiverScenarioInitialCommitment:
    if not isinstance(data, dict):
        raise ValueError("initial_commitment must be an object with 'hero' and 'villain'")
    hero = _require_non_negative(
        _as_number(data.get("hero"), "initial_commitment.hero"),
        "initial_commitment.hero",
    )
    villain = _require_non_negative(
        _as_number(data.get("villain"), "initial_commitment.villain"),
        "initial_commitment.villain",
    )
    return RiverScenarioInitialCommitment(hero=hero, villain=villain)


def _parse_shift_amounts(data) -> Optional[List[float]]:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("candidate_generation must be an object")
    raw = data.get("shift_amounts")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("candidate_generation.shift_amounts must be a list")
    shift_amounts: List[float] = []
    for index, value in enumerate(raw):
        amount = _as_number(value, f"shift_amounts[{index}]")
        _require_positive(amount, f"shift_amounts[{index}]")
        shift_amounts.append(amount)
    return shift_amounts


def _parse_hero_range(
    data, require_showdown: bool, betting_tree: bool = False
) -> RiverScenarioHeroRange:
    # Abstract Hero-range parsing only: this validates weighted Hero hand
    # buckets. It is not a real card range parser (see the module docstring for
    # the v1 scope). ``require_showdown`` is true in Hero-range-only mode (each
    # hand carries its own fixed ``showdown``) and false in matrix mode (the
    # outcome comes from the matrix, so a per-hand ``showdown`` is rejected here).
    # In ``betting_tree`` mode each hand carries ``baseline_strategies`` (two
    # decision points) instead of the simple ``baseline_strategy``.
    if not isinstance(data, list):
        raise ValueError("hero_range must be a list of hand objects")
    if not data:
        raise ValueError("hero_range must contain at least one hand")
    hands: List[RiverScenarioHeroRangeHand] = []
    seen_ids: set = set()
    weight_total = 0.0
    for index, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise ValueError(f"hero_range[{index}] must be an object")
        hand_id = raw.get("hand_id")
        if not isinstance(hand_id, str) or not hand_id:
            raise ValueError(f"hero_range[{index}].hand_id must be a non-empty string")
        if hand_id in seen_ids:
            raise ValueError(f"hero_range has duplicate hand_id {hand_id!r}")
        seen_ids.add(hand_id)
        weight = _require_positive(
            _as_number(raw.get("weight"), f"hero_range[{index}].weight"),
            f"hero_range[{index}].weight",
        )
        if require_showdown:
            showdown = raw.get("showdown")
            if showdown not in _SHOWDOWN_RESULTS:
                raise ValueError(
                    f"hero_range[{index}].showdown must be one of "
                    f"{list(_SHOWDOWN_RESULTS)}, got {showdown!r}"
                )
        else:
            if "showdown" in raw:
                raise ValueError(
                    f"hero_range[{index}].showdown must not be set in matrix mode; "
                    "the outcome comes from the matrix"
                )
            showdown = None

        baseline_strategy: Optional[Dict[str, float]] = None
        baseline_strategies: Optional[Dict[str, Dict[str, float]]] = None
        if betting_tree:
            if "baseline_strategy" in raw:
                raise ValueError(
                    f"hero_range[{index}] uses betting_tree mode and must provide "
                    "baseline_strategies, not the old baseline_strategy"
                )
            baseline_strategies = _parse_betting_tree_baseline(raw, index)
        else:
            baseline_strategy = _validate_action_distribution(
                raw.get("baseline_strategy"),
                _IP_ACTIONS,
                f"hero_range[{index}].baseline_strategy",
            )
        hands.append(
            RiverScenarioHeroRangeHand(
                hand_id=hand_id,
                weight=weight,
                showdown=showdown,
                baseline_strategy=baseline_strategy,
                baseline_strategies=baseline_strategies,
            )
        )
        weight_total += weight
    if abs(weight_total - 1.0) > _TOLERANCE:
        raise ValueError(
            f"hero_range weights sum to {weight_total}, expected 1"
        )
    return RiverScenarioHeroRange(hands=hands)


def _parse_betting_tree_baseline(raw, index: int) -> Dict[str, Dict[str, float]]:
    strategies = raw.get("baseline_strategies")
    if not isinstance(strategies, dict):
        raise ValueError(
            f"hero_range[{index}].baseline_strategies must be an object with "
            "'after_oop_check' and 'vs_oop_bet'"
        )
    if "after_oop_check" not in strategies:
        raise ValueError(
            f"hero_range[{index}].baseline_strategies must contain 'after_oop_check'"
        )
    if "vs_oop_bet" not in strategies:
        raise ValueError(
            f"hero_range[{index}].baseline_strategies must contain 'vs_oop_bet'"
        )
    extra = set(strategies) - {"after_oop_check", "vs_oop_bet"}
    if extra:
        raise ValueError(
            f"hero_range[{index}].baseline_strategies has unknown keys {sorted(extra)}"
        )
    after_check = _validate_action_distribution(
        strategies["after_oop_check"],
        _IP_AFTER_CHECK_ACTIONS,
        f"hero_range[{index}].baseline_strategies['after_oop_check']",
    )
    vs_bet = _validate_action_distribution(
        strategies["vs_oop_bet"],
        _IP_VS_BET_ACTIONS,
        f"hero_range[{index}].baseline_strategies['vs_oop_bet']",
    )
    return {"after_oop_check": after_check, "vs_oop_bet": vs_bet}


def _parse_betting_tree(data) -> RiverScenarioBettingTree:
    if not isinstance(data, dict):
        raise ValueError("betting_tree must be an object")
    oop_bet_size = _require_positive(
        _as_number(data.get("oop_bet_size"), "betting_tree.oop_bet_size"),
        "betting_tree.oop_bet_size",
    )
    ip_bet_after_check_size = _require_positive(
        _as_number(
            data.get("ip_bet_after_check_size"), "betting_tree.ip_bet_after_check_size"
        ),
        "betting_tree.ip_bet_after_check_size",
    )
    ip_raise_size = _require_positive(
        _as_number(data.get("ip_raise_size"), "betting_tree.ip_raise_size"),
        "betting_tree.ip_raise_size",
    )
    if ip_raise_size <= oop_bet_size:
        raise ValueError(
            "betting_tree.ip_raise_size must be greater than oop_bet_size "
            f"({ip_raise_size} <= {oop_bet_size}); it is the total committed after "
            "the raise is called, not the raise increment"
        )
    return RiverScenarioBettingTree(
        oop_bet_size=oop_bet_size,
        ip_bet_after_check_size=ip_bet_after_check_size,
        ip_raise_size=ip_raise_size,
    )


def _parse_villain_range(data) -> RiverScenarioVillainRange:
    if not isinstance(data, list):
        raise ValueError("villain_range must be a list of hand objects")
    if not data:
        raise ValueError("villain_range must contain at least one hand")
    hands: List[RiverScenarioVillainRangeHand] = []
    seen_ids: set = set()
    weight_total = 0.0
    for index, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise ValueError(f"villain_range[{index}] must be an object")
        hand_id = raw.get("hand_id")
        if not isinstance(hand_id, str) or not hand_id:
            raise ValueError(f"villain_range[{index}].hand_id must be a non-empty string")
        if hand_id in seen_ids:
            raise ValueError(f"villain_range has duplicate hand_id {hand_id!r}")
        seen_ids.add(hand_id)
        weight = _require_positive(
            _as_number(raw.get("weight"), f"villain_range[{index}].weight"),
            f"villain_range[{index}].weight",
        )
        hands.append(RiverScenarioVillainRangeHand(hand_id=hand_id, weight=weight))
        weight_total += weight
    if abs(weight_total - 1.0) > _TOLERANCE:
        raise ValueError(
            f"villain_range weights sum to {weight_total}, expected 1"
        )
    return RiverScenarioVillainRange(hands=hands)


def _parse_showdown_matrix(
    data,
    hero_range: RiverScenarioHeroRange,
    villain_range: RiverScenarioVillainRange,
) -> Dict[str, Dict[str, str]]:
    _validate_matrix_keys(data, hero_range, villain_range, "showdown_matrix")
    matrix: Dict[str, Dict[str, str]] = {}
    for hero in hero_range.hands:
        row = data[hero.hand_id]
        parsed_row: Dict[str, str] = {}
        for villain in villain_range.hands:
            result = row[villain.hand_id]
            if result not in _SHOWDOWN_RESULTS:
                raise ValueError(
                    f"showdown_matrix[{hero.hand_id!r}][{villain.hand_id!r}] must be "
                    f"one of {list(_SHOWDOWN_RESULTS)}, got {result!r}"
                )
            parsed_row[villain.hand_id] = result
        matrix[hero.hand_id] = parsed_row
    return matrix


def _validate_matrix_keys(
    data,
    hero_range: RiverScenarioHeroRange,
    villain_range: RiverScenarioVillainRange,
    name: str,
) -> None:
    """Check that a matchup matrix covers exactly the Hero/Villain id grid."""

    if not isinstance(data, dict):
        raise ValueError(f"{name} must be an object keyed by hero hand id")
    expected_hero = {hand.hand_id for hand in hero_range.hands}
    expected_villain = {hand.hand_id for hand in villain_range.hands}

    matrix_hero_ids = set(data)
    missing_hero = expected_hero - matrix_hero_ids
    if missing_hero:
        raise ValueError(f"{name} is missing hero ids {sorted(missing_hero)}")
    extra_hero = matrix_hero_ids - expected_hero
    if extra_hero:
        raise ValueError(f"{name} has unknown hero ids {sorted(extra_hero)}")

    for hand in hero_range.hands:
        row = data[hand.hand_id]
        if not isinstance(row, dict):
            raise ValueError(
                f"{name}[{hand.hand_id!r}] must be an object keyed by villain hand id"
            )
        row_villain_ids = set(row)
        missing_villain = expected_villain - row_villain_ids
        if missing_villain:
            raise ValueError(
                f"{name}[{hand.hand_id!r}] is missing villain ids "
                f"{sorted(missing_villain)}"
            )
        extra_villain = row_villain_ids - expected_villain
        if extra_villain:
            raise ValueError(
                f"{name}[{hand.hand_id!r}] has unknown villain ids "
                f"{sorted(extra_villain)}"
            )


def _parse_equity_matrix(
    data,
    hero_range: RiverScenarioHeroRange,
    villain_range: RiverScenarioVillainRange,
) -> Dict[str, Dict[str, float]]:
    _validate_matrix_keys(data, hero_range, villain_range, "equity_matrix")
    matrix: Dict[str, Dict[str, float]] = {}
    for hero in hero_range.hands:
        row = data[hero.hand_id]
        parsed_row: Dict[str, float] = {}
        for villain in villain_range.hands:
            equity = _as_number(
                row[villain.hand_id],
                f"equity_matrix[{hero.hand_id!r}][{villain.hand_id!r}]",
            )
            if not 0.0 <= equity <= 1.0:
                raise ValueError(
                    f"equity_matrix[{hero.hand_id!r}][{villain.hand_id!r}] must be "
                    f"within [0, 1], got {equity!r}"
                )
            parsed_row[villain.hand_id] = equity
        matrix[hero.hand_id] = parsed_row
    return matrix


def _parse_repeated(data) -> Optional[RiverScenarioRepeatedConfig]:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("repeated must be an object")
    horizons_raw = data.get("horizons")
    horizons: Optional[List[int]]
    if horizons_raw is None:
        horizons = None
    else:
        if not isinstance(horizons_raw, list):
            raise ValueError("repeated.horizons must be a list")
        horizons = []
        for index, value in enumerate(horizons_raw):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"repeated.horizons[{index}] must be a positive integer")
            if value < 1:
                raise ValueError(f"repeated.horizons[{index}] must be at least 1, got {value}")
            horizons.append(value)
    discount = _as_number(data.get("discount", 1.0), "repeated.discount")
    if not 0.0 < discount <= 1.0:
        raise ValueError(f"repeated.discount must satisfy 0 < discount <= 1, got {discount!r}")
    return RiverScenarioRepeatedConfig(horizons=horizons, discount=discount)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def river_scenario_from_dict(data) -> RiverScenario:
    """Parse and validate a river scenario from a plain dict."""

    if not isinstance(data, dict):
        raise ValueError("scenario must be a JSON object")

    scenario_id = data.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("scenario_id must be a non-empty string")

    description = data.get("description", "")
    if not isinstance(description, str):
        raise ValueError("description must be a string")

    rake = _parse_rake(data.get("rake"))
    initial_commitment = _parse_initial_commitment(data.get("initial_commitment"))

    has_hero_range = "hero_range" in data
    has_villain_range = "villain_range" in data
    has_showdown_matrix = "showdown_matrix" in data
    has_equity_matrix = "equity_matrix" in data
    has_matrix = has_showdown_matrix or has_equity_matrix
    has_single = "showdown" in data or "baseline_hero_strategy" in data
    has_betting_tree = "betting_tree" in data

    betting_tree = _parse_betting_tree(data.get("betting_tree")) if has_betting_tree else None

    # ``bet_size`` is the simple-tree bet. Betting-tree mode carries its sizes in
    # ``betting_tree`` instead, so the top-level ``bet_size`` is optional there
    # and defaults to ``betting_tree.oop_bet_size``. If both are given they must
    # agree, so the recorded ``bet_size`` cannot contradict the built tree.
    if "bet_size" in data:
        bet_size = _require_positive(_as_number(data.get("bet_size"), "bet_size"), "bet_size")
        if betting_tree is not None and abs(bet_size - betting_tree.oop_bet_size) > _TOLERANCE:
            raise ValueError(
                f"bet_size ({bet_size}) must match betting_tree.oop_bet_size "
                f"({betting_tree.oop_bet_size}) in betting-tree mode, or be omitted"
            )
    elif betting_tree is not None:
        bet_size = betting_tree.oop_bet_size
    else:
        bet_size = _require_positive(_as_number(data.get("bet_size"), "bet_size"), "bet_size")

    showdown: Optional[str] = None
    baseline_hero_strategy: Optional[Dict[str, Dict[str, float]]] = None
    hero_range: Optional[RiverScenarioHeroRange] = None
    villain_range: Optional[RiverScenarioVillainRange] = None
    showdown_matrix: Optional[Dict[str, Dict[str, str]]] = None
    equity_matrix: Optional[Dict[str, Dict[str, float]]] = None

    if has_villain_range or has_matrix:
        # Matrix mode: requires hero_range, villain_range, and exactly one of
        # showdown_matrix / equity_matrix, and forbids the single-hand fields.
        if not has_villain_range:
            raise ValueError("showdown_matrix / equity_matrix requires villain_range")
        if not has_matrix:
            raise ValueError(
                "villain_range requires a showdown_matrix or an equity_matrix"
            )
        if has_showdown_matrix and has_equity_matrix:
            raise ValueError(
                "provide exactly one of showdown_matrix or equity_matrix, not both"
            )
        if not has_hero_range:
            raise ValueError(
                "matrix mode requires hero_range together with villain_range "
                "and a showdown_matrix or equity_matrix"
            )
        if has_single:
            raise ValueError(
                "villain_range / showdown_matrix / equity_matrix cannot be "
                "combined with a top-level showdown or baseline_hero_strategy"
            )
        hero_range = _parse_hero_range(
            data.get("hero_range"), require_showdown=False, betting_tree=has_betting_tree
        )
        villain_range = _parse_villain_range(data.get("villain_range"))
        if has_showdown_matrix:
            showdown_matrix = _parse_showdown_matrix(
                data.get("showdown_matrix"), hero_range, villain_range
            )
        else:
            equity_matrix = _parse_equity_matrix(
                data.get("equity_matrix"), hero_range, villain_range
            )
        shared_ids = {hand.hand_id for hand in hero_range.hands} & {
            hand.hand_id for hand in villain_range.hands
        }
        if shared_ids:
            raise ValueError(
                "hero_range and villain_range must use disjoint hand ids; shared "
                f"ids {sorted(shared_ids)}"
            )
    elif has_hero_range:
        # Hero-range-only mode: per-hand showdown, no Villain range.
        if has_single:
            raise ValueError(
                "hero_range cannot be combined with a top-level showdown or "
                "baseline_hero_strategy; use either single-hand mode or range mode"
            )
        if has_betting_tree:
            raise ValueError(
                "betting_tree requires matrix mode (hero_range + villain_range + "
                "showdown_matrix or equity_matrix); it is not supported in "
                "Hero-range-only mode in v1"
            )
        hero_range = _parse_hero_range(data.get("hero_range"), require_showdown=True)
    else:
        if has_betting_tree:
            raise ValueError(
                "betting_tree requires matrix mode (hero_range + villain_range + "
                "showdown_matrix or equity_matrix); it is not supported in "
                "single-hand mode in v1"
            )
        # Single-hand mode.
        showdown = data.get("showdown")
        if showdown not in _SHOWDOWN_RESULTS:
            raise ValueError(
                f"showdown must be one of {list(_SHOWDOWN_RESULTS)}, got {showdown!r}"
            )

        baseline = data.get("baseline_hero_strategy")
        if not isinstance(baseline, dict):
            raise ValueError("baseline_hero_strategy must be an object")
        if _IP_INFO_SET not in baseline:
            raise ValueError(f"baseline_hero_strategy must contain {_IP_INFO_SET!r}")
        extra_info_sets = set(baseline) - {_IP_INFO_SET}
        if extra_info_sets:
            raise ValueError(
                f"baseline_hero_strategy has unknown information sets {sorted(extra_info_sets)}"
            )
        ip_distribution = _validate_action_distribution(
            baseline[_IP_INFO_SET], _IP_ACTIONS, f"baseline_hero_strategy[{_IP_INFO_SET!r}]"
        )
        baseline_hero_strategy = {_IP_INFO_SET: ip_distribution}

    shift_amounts = _parse_shift_amounts(data.get("candidate_generation"))
    repeated = _parse_repeated(data.get("repeated"))

    return RiverScenario(
        scenario_id=scenario_id,
        description=description,
        rake=rake,
        initial_commitment=initial_commitment,
        bet_size=bet_size,
        showdown=showdown,
        baseline_hero_strategy=baseline_hero_strategy,
        shift_amounts=shift_amounts,
        repeated=repeated,
        hero_range=hero_range,
        villain_range=villain_range,
        showdown_matrix=showdown_matrix,
        equity_matrix=equity_matrix,
        betting_tree=betting_tree,
    )


def load_river_scenario_json(path: Union[str, Path]) -> RiverScenario:
    """Load and validate a river scenario from a JSON file."""

    text = Path(path).read_text(encoding="utf-8")
    return river_scenario_from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# Game construction
# ---------------------------------------------------------------------------


def build_river_steal_game_from_scenario(
    scenario: RiverScenario,
) -> RiverScenarioBuildResult:
    """Build the game tree and pipeline inputs from a validated scenario.

    Dispatches on the scenario mode: a single ``VillainNode`` root in single-hand
    mode, a ``ChanceNode`` over weighted Hero buckets in Hero-range-only mode, a
    ``ChanceNode`` over ``(hero, villain)`` matchup pairs in matrix mode, or the
    fuller one-street tree in betting-tree mode.
    """

    if scenario.is_betting_tree_mode:
        tree, baseline_hero_strategy, metadata = _build_betting_tree(scenario)
    elif scenario.is_matrix_mode:
        tree, baseline_hero_strategy, metadata = _build_matrix_tree(scenario)
    elif scenario.is_range_mode:
        tree, baseline_hero_strategy, metadata = _build_range_tree(scenario)
    else:
        tree, baseline_hero_strategy, metadata = _build_single_hand_tree(scenario)

    validate_hero_strategy(tree, baseline_hero_strategy)
    baseline_villain_strategy = _villain_baseline_best_response(tree, baseline_hero_strategy)

    return RiverScenarioBuildResult(
        scenario_id=scenario.scenario_id,
        description=scenario.description,
        tree=tree,
        baseline_hero_strategy=baseline_hero_strategy,
        baseline_villain_strategy=baseline_villain_strategy,
        shift_amounts=scenario.shift_amounts,
        repeated=scenario.repeated,
        metadata=metadata,
    )


def _hand_subtree(
    scenario: RiverScenario,
    showdown: Optional[str],
    suffix: str,
    ip_info_set: str,
    oop_info_set: str = _OOP_INFO_SET,
    hero_equity: Optional[float] = None,
):
    """Build the OOP/IP subtree for one matchup, with unique node ids.

    The showdown payoff is a discrete ``showdown`` result when ``hero_equity`` is
    ``None``, or a fractional Hero pot share when ``hero_equity`` is given.

    OOP ``check`` leads straight to a check-check showdown terminal: JSON
    scenario v1 has no IP action after an OOP check, so no ``IP_vs_check``
    information set is created here.
    """

    hero_initial = scenario.initial_commitment.hero
    villain_initial = scenario.initial_commitment.villain
    bet_size = scenario.bet_size
    rate = scenario.rake.rate
    cap = scenario.rake.cap

    def _showdown_terminal(node_id, pot, hero_invested, villain_invested):
        if hero_equity is None:
            return make_showdown_terminal(
                node_id, pot, hero_invested, villain_invested, showdown, rate, cap
            )
        return make_equity_showdown_terminal(
            node_id, pot, hero_invested, villain_invested, hero_equity, rate, cap
        )

    check_terminal = _showdown_terminal(
        f"T_check_check{suffix}",
        hero_initial + villain_initial,
        hero_initial,
        villain_initial,
    )
    bet_call_terminal = _showdown_terminal(
        f"T_bet_call{suffix}",
        hero_initial + villain_initial + 2.0 * bet_size,
        hero_initial + bet_size,
        villain_initial + bet_size,
    )
    # IP folds to the bet: the uncalled bet is returned and OOP (Villain) wins
    # IP's committed chips.
    bet_fold_terminal = make_fold_terminal(f"T_bet_fold{suffix}", VILLAIN, hero_initial)

    ip_node = HeroNode(
        node_id=f"ip{suffix}",
        info_set=ip_info_set,
        actions=(("call", bet_call_terminal), ("fold", bet_fold_terminal)),
    )
    oop_node = VillainNode(
        node_id=f"oop{suffix}",
        info_set=oop_info_set,
        actions=(("check", check_terminal), ("bet", ip_node)),
    )
    return oop_node


def _base_metadata(scenario: RiverScenario) -> dict:
    return {
        "scenario_id": scenario.scenario_id,
        "description": scenario.description,
        "bet_size": scenario.bet_size,
        "rake": {"rate": scenario.rake.rate, "cap": scenario.rake.cap},
        "initial_commitment": {
            "hero": scenario.initial_commitment.hero,
            "villain": scenario.initial_commitment.villain,
        },
    }


def _build_single_hand_tree(scenario: RiverScenario):
    oop_node = _hand_subtree(scenario, scenario.showdown, "", _IP_INFO_SET)
    tree = GameTree(root=oop_node)
    baseline_hero_strategy = HeroStrategy(
        probabilities={
            _IP_INFO_SET: dict(scenario.baseline_hero_strategy[_IP_INFO_SET])
        }
    )
    metadata = _base_metadata(scenario)
    metadata["mode"] = "single_hand"
    metadata["showdown"] = scenario.showdown
    return tree, baseline_hero_strategy, metadata


def _build_range_tree(scenario: RiverScenario):
    """Build a chance-node tree over the weighted Hero hand buckets.

    Villain shares the single ``OOP_river`` information set across every bucket,
    so it cannot condition on Hero's hidden hand. Hero gets a per-hand
    information set ``IP_vs_bet::<hand_id>`` because Hero knows its own hand.
    """

    hands = scenario.hero_range.hands
    children = []
    hero_probabilities: Dict[str, Dict[str, float]] = {}
    for hand in hands:
        ip_info_set = f"{_IP_INFO_SET}::{hand.hand_id}"
        suffix = f"::{hand.hand_id}"
        oop_node = _hand_subtree(scenario, hand.showdown, suffix, ip_info_set)
        children.append((hand.weight, oop_node))
        hero_probabilities[ip_info_set] = dict(hand.baseline_strategy)

    root = ChanceNode(node_id="hand_bucket", children=tuple(children))
    tree = GameTree(root=root)
    # Guard the chance probabilities, terminal invariants, and per-player
    # information-set consistency before the tree leaves this module.
    validate_tree(tree)

    baseline_hero_strategy = HeroStrategy(probabilities=hero_probabilities)
    metadata = _base_metadata(scenario)
    metadata["mode"] = "range"
    metadata["hand_buckets"] = [
        {"hand_id": hand.hand_id, "weight": hand.weight, "showdown": hand.showdown}
        for hand in hands
    ]
    return tree, baseline_hero_strategy, metadata


def _build_matrix_tree(scenario: RiverScenario):
    """Build a chance-node tree over weighted ``(hero, villain)`` matchup pairs.

    Each Villain bucket gets its own ``OOP_river::<villain_id>`` information set,
    shared across Hero buckets because Villain knows its own bucket but not
    Hero's. Each Hero bucket keeps its own ``IP_vs_bet::<hero_id>`` information
    set, shared across Villain buckets because Hero knows its own bucket but not
    Villain's. The outcome of each pair comes from the showdown matrix (discrete
    result) or the equity matrix (fractional Hero pot share).
    """

    hero_hands = scenario.hero_range.hands
    villain_hands = scenario.villain_range.hands
    use_equity = scenario.equity_matrix is not None
    matrix = scenario.equity_matrix if use_equity else scenario.showdown_matrix

    children = []
    hero_probabilities: Dict[str, Dict[str, float]] = {}
    for hero in hero_hands:
        ip_info_set = f"{_IP_INFO_SET}::{hero.hand_id}"
        hero_probabilities[ip_info_set] = dict(hero.baseline_strategy)

    for hero in hero_hands:
        ip_info_set = f"{_IP_INFO_SET}::{hero.hand_id}"
        for villain in villain_hands:
            oop_info_set = f"{_OOP_INFO_SET}::{villain.hand_id}"
            suffix = f"::{hero.hand_id}__{villain.hand_id}"
            cell = matrix[hero.hand_id][villain.hand_id]
            if use_equity:
                oop_node = _hand_subtree(
                    scenario, None, suffix, ip_info_set, oop_info_set,
                    hero_equity=cell,
                )
            else:
                oop_node = _hand_subtree(
                    scenario, cell, suffix, ip_info_set, oop_info_set
                )
            children.append((hero.weight * villain.weight, oop_node))

    root = ChanceNode(node_id="hand_matchup", children=tuple(children))
    tree = GameTree(root=root)
    # Guard the chance probabilities, terminal invariants, and per-player
    # information-set consistency before the tree leaves this module.
    validate_tree(tree)

    baseline_hero_strategy = HeroStrategy(probabilities=hero_probabilities)
    metadata = _base_metadata(scenario)
    metadata["mode"] = "range_matrix"
    metadata["matrix_type"] = "equity" if use_equity else "showdown"
    metadata["hero_buckets"] = [
        {"hand_id": hand.hand_id, "weight": hand.weight} for hand in hero_hands
    ]
    metadata["villain_buckets"] = [
        {"hand_id": hand.hand_id, "weight": hand.weight} for hand in villain_hands
    ]
    if use_equity:
        metadata["equity_matrix"] = {
            hero_id: dict(row) for hero_id, row in matrix.items()
        }
    else:
        metadata["showdown_matrix"] = {
            hero_id: dict(row) for hero_id, row in matrix.items()
        }
    return tree, baseline_hero_strategy, metadata


def _villain_baseline_best_response(
    tree: GameTree, baseline_hero_strategy: HeroStrategy
) -> VillainStrategy:
    """Return the pure Villain best response to the baseline Hero strategy.

    Works for any number of Villain information sets: the first best response
    returned by :func:`solve_exact_response` assigns one action to every Villain
    information set, and this turns that assignment into a pure
    :class:`VillainStrategy`.

    When several pure best responses tie, this deterministically picks the first
    one returned by :func:`solve_exact_response` (its enumeration order). This is
    only a stable, reproducible tie-break for building a concrete baseline
    profile; it makes no equilibrium-selection claim and does not assert the
    chosen response is preferred among the tied best responses.
    """

    response = solve_exact_response(tree, baseline_hero_strategy)
    chosen = response.best_response_strategies[0]
    # Use each information set's own legal actions so this also covers
    # betting-tree mode, where Villain information sets have different action
    # sets (check/bet vs call/fold).
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


def _build_betting_tree(scenario: RiverScenario):
    """Build the river one-street betting tree (betting-tree mode).

    Delegates the tree and Hero-strategy construction to
    :mod:`repeated_poker.river_betting_tree` and assembles the metadata here so
    the heavier accounting stays out of this module.
    """

    tree, baseline_hero_strategy = build_betting_tree_from_scenario(scenario)
    use_equity = scenario.equity_matrix is not None
    betting = scenario.betting_tree
    metadata = _base_metadata(scenario)
    metadata["mode"] = "range_matrix"
    metadata["matrix_type"] = "equity" if use_equity else "showdown"
    metadata["betting_tree"] = {
        "oop_bet_size": betting.oop_bet_size,
        "ip_bet_after_check_size": betting.ip_bet_after_check_size,
        "ip_raise_size": betting.ip_raise_size,
    }
    metadata["hero_buckets"] = [
        {"hand_id": hand.hand_id, "weight": hand.weight}
        for hand in scenario.hero_range.hands
    ]
    metadata["villain_buckets"] = [
        {"hand_id": hand.hand_id, "weight": hand.weight}
        for hand in scenario.villain_range.hands
    ]
    return tree, baseline_hero_strategy, metadata
