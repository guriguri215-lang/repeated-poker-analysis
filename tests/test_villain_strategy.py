"""Tests for VillainStrategy validation and the Hero unknown-info-set check."""

import math

import pytest

from repeated_poker import (
    GameTree,
    HeroNode,
    HeroStrategy,
    TerminalNode,
    VillainNode,
    VillainStrategy,
    validate_hero_strategy,
    validate_villain_strategy,
)


def _zero_terminal(node_id):
    return TerminalNode(node_id, hero_ev=0.0, villain_ev=0.0, house_rake=0.0)


def _villain_tree():
    return GameTree(
        root=VillainNode(
            node_id="v",
            info_set="V",
            actions=(("call", _zero_terminal("t1")), ("fold", _zero_terminal("t2"))),
        )
    )


def test_valid_villain_strategy_passes():
    validate_villain_strategy(
        _villain_tree(), VillainStrategy({"V": {"call": 0.5, "fold": 0.5}})
    )


def test_villain_strategy_must_sum_to_one():
    with pytest.raises(ValueError, match="sums to"):
        validate_villain_strategy(
            _villain_tree(), VillainStrategy({"V": {"call": 0.5, "fold": 0.4}})
        )


def test_villain_strategy_rejects_illegal_action():
    with pytest.raises(ValueError, match="illegal actions"):
        validate_villain_strategy(
            _villain_tree(),
            VillainStrategy({"V": {"call": 0.5, "fold": 0.5, "raise": 0.0}}),
        )


def test_villain_strategy_rejects_unknown_info_set():
    with pytest.raises(ValueError, match="unknown information sets"):
        validate_villain_strategy(
            _villain_tree(),
            VillainStrategy(
                {"V": {"call": 0.5, "fold": 0.5}, "Z": {"call": 1.0}}
            ),
        )


def test_villain_strategy_rejects_missing_info_set():
    with pytest.raises(ValueError, match="missing information set"):
        validate_villain_strategy(_villain_tree(), VillainStrategy({}))


def test_villain_strategy_rejects_non_finite_probability():
    with pytest.raises(ValueError, match="non-finite"):
        validate_villain_strategy(
            _villain_tree(), VillainStrategy({"V": {"call": math.nan, "fold": 1.0}})
        )


def test_villain_strategy_rejects_negative_probability():
    with pytest.raises(ValueError, match="negative"):
        validate_villain_strategy(
            _villain_tree(), VillainStrategy({"V": {"call": -0.5, "fold": 1.5}})
        )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -1.0])
def test_villain_strategy_rejects_invalid_tolerance(bad):
    with pytest.raises(ValueError, match="tolerance"):
        validate_villain_strategy(
            _villain_tree(),
            VillainStrategy({"V": {"call": 0.5, "fold": 0.5}}),
            tolerance=bad,
        )


def test_hero_strategy_rejects_unknown_info_set():
    tree = GameTree(
        root=HeroNode(
            node_id="h",
            info_set="H",
            actions=(("a", _zero_terminal("t1")), ("b", _zero_terminal("t2"))),
        )
    )
    bad = HeroStrategy({"H": {"a": 0.5, "b": 0.5}, "Z": {"a": 1.0}})
    with pytest.raises(ValueError, match="unknown information sets"):
        validate_hero_strategy(tree, bad)
