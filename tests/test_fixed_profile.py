"""Tests for the fixed Hero-and-Villain profile evaluator."""

import pytest

from repeated_poker import (
    GameTree,
    HeroNode,
    HeroStrategy,
    TerminalNode,
    VillainNode,
    VillainStrategy,
    evaluate_fixed_profile,
)


def _profile_tree():
    """Hero bets/checks; on a bet Villain calls/folds. Non-zero rake on a call."""

    villain = VillainNode(
        node_id="v",
        info_set="V",
        actions=(
            ("call", TerminalNode("t_call", hero_ev=2.0, villain_ev=-3.0, house_rake=1.0)),
            ("fold", TerminalNode("t_fold", hero_ev=1.0, villain_ev=-1.0, house_rake=0.0)),
        ),
    )
    root = HeroNode(
        node_id="h",
        info_set="H",
        actions=(
            ("bet", villain),
            ("check", TerminalNode("t_check", hero_ev=0.0, villain_ev=0.0, house_rake=0.0)),
        ),
    )
    return GameTree(root=root)


def test_fixed_profile_hand_computed_value():
    tree = _profile_tree()
    hero = HeroStrategy({"H": {"bet": 0.5, "check": 0.5}})
    villain = VillainStrategy({"V": {"call": 0.25, "fold": 0.75}})

    value = evaluate_fixed_profile(tree, hero, villain)

    # bet branch: 0.25*(2,-3,1) + 0.75*(1,-1,0) = (1.25, -1.5, 0.25)
    # total:      0.5*(1.25,-1.5,0.25) + 0.5*(0,0,0) = (0.625, -0.75, 0.125)
    assert value.hero_ev == pytest.approx(0.625)
    assert value.villain_ev == pytest.approx(-0.75)
    assert value.house_rake == pytest.approx(0.125)
    # The triple still nets to zero because rake is the only non-zero-sum part.
    assert value.hero_ev + value.villain_ev + value.house_rake == pytest.approx(0.0)


def test_fixed_profile_validates_villain_strategy():
    tree = _profile_tree()
    hero = HeroStrategy({"H": {"bet": 0.5, "check": 0.5}})
    bad_villain = VillainStrategy({"V": {"call": 0.5, "fold": 0.4}})
    with pytest.raises(ValueError, match="sums to"):
        evaluate_fixed_profile(tree, hero, bad_villain)
