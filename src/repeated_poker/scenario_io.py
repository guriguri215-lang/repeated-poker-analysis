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

It does not parse real cards, hand ranges, or external solver exports; those are
future extensions. The core ``game`` / ``payoffs`` / ``exact_response`` /
``repeated`` modules are reused unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from .exact_response import solve_exact_response
from .game import (
    GameTree,
    HeroNode,
    HeroStrategy,
    VillainNode,
    VillainStrategy,
    require_finite,
    validate_hero_strategy,
)
from .payoffs import CHOP, HERO, VILLAIN, make_fold_terminal, make_showdown_terminal

_TOLERANCE = 1e-9
_SHOWDOWN_RESULTS = (CHOP, HERO, VILLAIN)
_OOP_INFO_SET = "OOP_river"
_IP_INFO_SET = "IP_vs_bet"
_OOP_ACTIONS = ("check", "bet")
_IP_ACTIONS = ("call", "fold")


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
class RiverScenario:
    """A validated abstract river steal scenario."""

    scenario_id: str
    description: str
    rake: RiverScenarioRake
    initial_commitment: RiverScenarioInitialCommitment
    bet_size: float
    showdown: str
    baseline_hero_strategy: Dict[str, Dict[str, float]]
    shift_amounts: Optional[List[float]]
    repeated: Optional[RiverScenarioRepeatedConfig]


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
    bet_size = _require_positive(_as_number(data.get("bet_size"), "bet_size"), "bet_size")

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
    """Build the game tree and pipeline inputs from a validated scenario."""

    hero_initial = scenario.initial_commitment.hero
    villain_initial = scenario.initial_commitment.villain
    bet_size = scenario.bet_size
    rate = scenario.rake.rate
    cap = scenario.rake.cap
    result = scenario.showdown

    check_terminal = make_showdown_terminal(
        "T_check_check",
        hero_initial + villain_initial,
        hero_initial,
        villain_initial,
        result,
        rate,
        cap,
    )
    bet_call_terminal = make_showdown_terminal(
        "T_bet_call",
        hero_initial + villain_initial + 2.0 * bet_size,
        hero_initial + bet_size,
        villain_initial + bet_size,
        result,
        rate,
        cap,
    )
    # IP folds to the bet: the uncalled bet is returned and OOP (Villain) wins
    # IP's committed chips.
    bet_fold_terminal = make_fold_terminal("T_bet_fold", VILLAIN, hero_initial)

    ip_node = HeroNode(
        node_id="ip",
        info_set=_IP_INFO_SET,
        actions=(("call", bet_call_terminal), ("fold", bet_fold_terminal)),
    )
    oop_node = VillainNode(
        node_id="oop",
        info_set=_OOP_INFO_SET,
        actions=(("check", check_terminal), ("bet", ip_node)),
    )
    tree = GameTree(root=oop_node)

    baseline_hero_strategy = HeroStrategy(
        probabilities={
            _IP_INFO_SET: dict(scenario.baseline_hero_strategy[_IP_INFO_SET])
        }
    )
    validate_hero_strategy(tree, baseline_hero_strategy)

    baseline_villain_strategy = _single_hand_villain_baseline(tree, baseline_hero_strategy)

    metadata = {
        "scenario_id": scenario.scenario_id,
        "description": scenario.description,
        "bet_size": scenario.bet_size,
        "showdown": scenario.showdown,
        "rake": {"rate": scenario.rake.rate, "cap": scenario.rake.cap},
        "initial_commitment": {
            "hero": scenario.initial_commitment.hero,
            "villain": scenario.initial_commitment.villain,
        },
    }

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


def _single_hand_villain_baseline(
    tree: GameTree, baseline_hero_strategy: HeroStrategy
) -> VillainStrategy:
    """Return the pure Villain best response to the baseline Hero strategy.

    When several pure best responses tie, this deterministically picks the first
    one returned by :func:`solve_exact_response` (its enumeration order). This is
    only a stable, reproducible tie-break for building a concrete baseline
    profile; it makes no equilibrium-selection claim and does not assert the
    chosen response is preferred among the tied best responses.
    """

    response = solve_exact_response(tree, baseline_hero_strategy)
    chosen = response.best_response_strategies[0][_OOP_INFO_SET]
    return VillainStrategy(
        probabilities={
            _OOP_INFO_SET: {
                action: (1.0 if action == chosen else 0.0) for action in _OOP_ACTIONS
            }
        }
    )
