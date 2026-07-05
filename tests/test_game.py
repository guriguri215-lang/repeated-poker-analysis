"""Tests for the game data model and its validation."""

import math

import pytest

from repeated_poker.game import (
    ChanceNode,
    GameTree,
    HeroNode,
    HeroStrategy,
    TerminalNode,
    VillainNode,
    validate_hero_strategy,
    validate_tree,
)


def _terminal(node_id="t"):
    return TerminalNode(node_id=node_id, hero_ev=0.0, villain_ev=0.0, house_rake=0.0)


def test_hero_strategy_must_sum_to_one():
    tree = GameTree(
        root=HeroNode(
            node_id="h",
            info_set="H",
            actions=(("a", _terminal("ta")), ("b", _terminal("tb"))),
        )
    )
    bad = HeroStrategy(probabilities={"H": {"a": 0.6, "b": 0.6}})
    with pytest.raises(ValueError, match="sums to"):
        validate_hero_strategy(tree, bad)

    good = HeroStrategy(probabilities={"H": {"a": 0.6, "b": 0.4}})
    validate_hero_strategy(tree, good)  # does not raise


def test_hero_strategy_rejects_illegal_action():
    tree = GameTree(
        root=HeroNode(
            node_id="h",
            info_set="H",
            actions=(("a", _terminal("ta")), ("b", _terminal("tb"))),
        )
    )
    bad = HeroStrategy(probabilities={"H": {"a": 0.5, "b": 0.5, "c": 0.0}})
    with pytest.raises(ValueError, match="illegal actions"):
        validate_hero_strategy(tree, bad)


def test_hero_strategy_rejects_non_finite_probability():
    tree = GameTree(
        root=HeroNode(
            node_id="h",
            info_set="H",
            actions=(("a", _terminal("ta")), ("b", _terminal("tb"))),
        )
    )
    bad = HeroStrategy(probabilities={"H": {"a": math.nan, "b": 0.5}})
    with pytest.raises(ValueError, match="non-finite"):
        validate_hero_strategy(tree, bad)


def test_validate_tree_rejects_non_finite_terminal_payoff():
    tree = GameTree(
        root=TerminalNode("t", hero_ev=math.inf, villain_ev=0.0, house_rake=0.0)
    )
    with pytest.raises(ValueError, match="non-finite"):
        validate_tree(tree)


def test_validate_tree_rejects_negative_house_rake():
    tree = GameTree(
        root=TerminalNode("t", hero_ev=1.0, villain_ev=0.0, house_rake=-1.0)
    )
    with pytest.raises(ValueError, match="negative house_rake"):
        validate_tree(tree)


def test_validate_tree_can_allow_negative_residual_accounting():
    tree = GameTree(
        root=TerminalNode("t", hero_ev=1.0, villain_ev=0.0, house_rake=-1.0)
    )

    validate_tree(tree, allow_negative_residual=True)


def test_validate_tree_rejects_non_bool_allow_negative_residual():
    tree = _single_hero_tree()

    with pytest.raises(ValueError, match="allow_negative_residual"):
        validate_tree(tree, allow_negative_residual=1)


def test_validate_tree_rejects_non_zero_sum_terminal():
    tree = GameTree(
        root=TerminalNode("t", hero_ev=1.0, villain_ev=0.0, house_rake=0.0)
    )
    with pytest.raises(ValueError, match="hero_ev \\+ villain_ev \\+ house_rake"):
        validate_tree(tree)


def test_validate_tree_rejects_repeated_info_set_on_path():
    # The Hero info set "H" appears twice on a single root-to-terminal path
    # (with consistent legal actions, so only the recall guard can fire).
    inner = HeroNode("h2", info_set="H", actions=(("a", _terminal("t")),))
    outer = HeroNode("h1", info_set="H", actions=(("a", inner),))
    tree = GameTree(root=outer)
    with pytest.raises(ValueError, match="repeats on a single"):
        validate_tree(tree)


def test_inconsistent_villain_info_set_legal_actions_raise():
    # The same Villain info set "V" appears with different legal-action sets.
    tree = GameTree(
        root=ChanceNode(
            node_id="c",
            children=(
                (
                    0.5,
                    VillainNode(
                        node_id="v1",
                        info_set="V",
                        actions=(("call", _terminal("t1")), ("fold", _terminal("t2"))),
                    ),
                ),
                (
                    0.5,
                    VillainNode(
                        node_id="v2",
                        info_set="V",
                        actions=(("call", _terminal("t3")),),
                    ),
                ),
            ),
        )
    )
    with pytest.raises(ValueError, match="inconsistent legal"):
        validate_tree(tree)


def _single_hero_tree():
    return GameTree(
        root=HeroNode(
            node_id="h",
            info_set="H",
            actions=(("a", _terminal("ta")), ("b", _terminal("tb"))),
        )
    )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -1.0])
def test_validate_tree_rejects_invalid_tolerance(bad):
    with pytest.raises(ValueError, match="tolerance"):
        validate_tree(_single_hero_tree(), tolerance=bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -1.0])
def test_validate_hero_strategy_rejects_invalid_tolerance(bad):
    tree = _single_hero_tree()
    good = HeroStrategy(probabilities={"H": {"a": 0.5, "b": 0.5}})
    with pytest.raises(ValueError, match="tolerance"):
        validate_hero_strategy(tree, good, tolerance=bad)


def test_chance_probabilities_must_sum_to_one():
    tree = GameTree(
        root=ChanceNode(
            node_id="c",
            children=((0.5, _terminal("t1")), (0.4, _terminal("t2"))),
        )
    )
    with pytest.raises(ValueError, match="sum to"):
        validate_tree(tree)
