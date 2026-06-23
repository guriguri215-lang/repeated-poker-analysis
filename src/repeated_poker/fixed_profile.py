"""Exact expected value of a fully fixed Hero-and-Villain profile.

Unlike :mod:`repeated_poker.exact_response`, which optimises Villain freely,
this module evaluates a profile where *both* players are locked to fixed mixed
strategies.  It recurses over chance nodes, Hero-mixed decision nodes,
Villain-mixed decision nodes, and terminal payoffs to produce the exact
expected ``(hero_ev, villain_ev, house_rake)`` triple.

This is used to score a candidate Hero policy while Villain remains at a fixed
baseline strategy.  It does not compute a best response.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .game import (
    ChanceNode,
    GameTree,
    HeroNode,
    HeroStrategy,
    Node,
    TerminalNode,
    VillainNode,
    VillainStrategy,
    require_valid_tolerance,
    validate_hero_strategy,
    validate_tree,
    validate_villain_strategy,
)

# An expected payoff triple (hero_ev, villain_ev, house_rake).
Payoff = Tuple[float, float, float]


@dataclass(frozen=True)
class FixedProfileValue:
    """The exact expected payoff of a fixed Hero-and-Villain profile."""

    hero_ev: float
    villain_ev: float
    house_rake: float

    def to_dict(self) -> dict:
        """Return a JSON-serialisable summary with English keys."""

        return {
            "hero_ev": self.hero_ev,
            "villain_ev": self.villain_ev,
            "house_rake": self.house_rake,
        }


def evaluate_fixed_profile(
    tree: GameTree,
    hero_strategy: HeroStrategy,
    villain_strategy: VillainStrategy,
    tolerance: float = 1e-9,
) -> FixedProfileValue:
    """Return the exact expected value of a fixed Hero-and-Villain profile.

    Validates the tree and both strategies, then recurses over the tree mixing
    chance, Hero, and Villain probabilities to accumulate the expected payoff
    triple.
    """

    require_valid_tolerance(tolerance)
    validate_tree(tree, tolerance=tolerance)
    validate_hero_strategy(tree, hero_strategy, tolerance=tolerance)
    validate_villain_strategy(tree, villain_strategy, tolerance=tolerance)

    hero_ev, villain_ev, house_rake = _expected_payoff_profile(
        tree.root, hero_strategy, villain_strategy
    )
    return FixedProfileValue(
        hero_ev=hero_ev, villain_ev=villain_ev, house_rake=house_rake
    )


def _expected_payoff_profile(
    node: Node, hero_strategy: HeroStrategy, villain_strategy: VillainStrategy
) -> Payoff:
    """Exact expected payoff triple for ``node`` under both fixed strategies."""

    if isinstance(node, TerminalNode):
        return (node.hero_ev, node.villain_ev, node.house_rake)

    if isinstance(node, ChanceNode):
        weighted = node.children
    elif isinstance(node, HeroNode):
        weighted = tuple(
            (hero_strategy.action_probability(node.info_set, action), child)
            for action, child in node.actions
        )
    elif isinstance(node, VillainNode):
        weighted = tuple(
            (villain_strategy.action_probability(node.info_set, action), child)
            for action, child in node.actions
        )
    else:
        raise TypeError(f"unknown node type: {type(node)!r}")

    hero = villain = rake = 0.0
    for weight, child in weighted:
        if weight == 0.0:
            continue
        ch, cv, cr = _expected_payoff_profile(child, hero_strategy, villain_strategy)
        hero += weight * ch
        villain += weight * cv
        rake += weight * cr
    return (hero, villain, rake)
