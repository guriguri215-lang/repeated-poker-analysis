"""Tests for the exact Villain best-response engine."""

import pytest

from nuts_chop_river import build_nuts_chop_river
from nuts_chop_river import default_hero_strategy as nuts_chop_hero_strategy
from value_bluff_river import (
    build_value_bluff_river,
    default_hero_strategy,
    indifference_hero_strategy,
)
from repeated_poker import (
    ChanceNode,
    GameTree,
    HeroNode,
    HeroStrategy,
    TerminalNode,
    VillainNode,
    count_villain_pure_strategies,
    enumerate_villain_pure_strategies,
    iter_terminals,
    solve_exact_response,
)

TOL = 1e-9


def _terminals_by_id(tree):
    return {t.node_id: t for t in iter_terminals(tree.root)}


def _indifference_tree():
    """Villain is exactly indifferent between call and fold (both yield -1)."""

    root = HeroNode(
        node_id="h",
        info_set="H",
        actions=(
            (
                "bet",
                VillainNode(
                    node_id="v",
                    info_set="V",
                    actions=(
                        # Showdown after a call: Villain loses 1, rake 2.
                        (
                            "call",
                            TerminalNode("t_call", hero_ev=-1.0, villain_ev=-1.0, house_rake=2.0),
                        ),
                        # Fold: Villain loses 1, no rake, Hero wins 1.
                        (
                            "fold",
                            TerminalNode("t_fold", hero_ev=1.0, villain_ev=-1.0, house_rake=0.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    return GameTree(root=root), HeroStrategy(probabilities={"H": {"bet": 1.0}})


def test_indifferent_villain_returns_multiple_best_responses():
    tree, hero_strategy = _indifference_tree()
    result = solve_exact_response(tree, hero_strategy)

    assert result.villain_max_ev == pytest.approx(-1.0)
    assert len(result.best_response_strategies) == 2
    assert result.best_response_action_variation == {"V": ["call", "fold"]}
    # The tie is genuine on-path indifference, not off-path freedom.
    assert result.off_path_info_sets == []


def test_ev_h_worst_and_best_are_computed():
    tree, hero_strategy = _indifference_tree()
    result = solve_exact_response(tree, hero_strategy)

    # Hero EV is -1 when Villain calls (worst) and +1 when Villain folds (best).
    assert result.ev_h_worst == pytest.approx(-1.0)
    assert result.ev_h_best == pytest.approx(1.0)
    # Expected rake differs across the Hero-EV extremes.
    assert result.expected_house_rake_worst == pytest.approx(2.0)
    assert result.expected_house_rake_best == pytest.approx(0.0)


def _shared_info_set_tree():
    """One Villain info set "V" reached on two chance branches.

    Picking ``call`` per node would be worth +2 to Villain, but a single shared
    action forces ``call`` to average to -3, so the best response is ``fold``.
    """

    def branch(hand, call_villain_ev):
        return HeroNode(
            node_id=f"h_{hand}",
            info_set="H",
            actions=(
                (
                    "go",
                    VillainNode(
                        node_id=f"v_{hand}",
                        info_set="V",
                        actions=(
                            (
                                "call",
                                TerminalNode(
                                    f"t_call_{hand}",
                                    hero_ev=-call_villain_ev,
                                    villain_ev=call_villain_ev,
                                    house_rake=0.0,
                                ),
                            ),
                            (
                                "fold",
                                TerminalNode(
                                    f"t_fold_{hand}",
                                    hero_ev=0.0,
                                    villain_ev=0.0,
                                    house_rake=0.0,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

    root = ChanceNode(
        node_id="c",
        children=((0.5, branch("a", 4.0)), (0.5, branch("b", -10.0))),
    )
    return GameTree(root=root), HeroStrategy(probabilities={"H": {"go": 1.0}})


def test_villain_action_is_unified_across_an_info_set():
    tree, hero_strategy = _shared_info_set_tree()

    # Only one Villain info set with two actions -> two pure strategies, not four.
    assert count_villain_pure_strategies(tree) == 2
    assert len(enumerate_villain_pure_strategies(tree)) == 2

    result = solve_exact_response(tree, hero_strategy)
    # Best response is fold (EV 0), not the per-node optimum of +2.
    assert result.villain_max_ev == pytest.approx(0.0)
    assert result.best_response_strategies == [{"V": "fold"}]


def _off_path_tree():
    """Hero never bets, so the Villain bet node is never reached (off-path)."""

    root = HeroNode(
        node_id="h",
        info_set="H1",
        actions=(
            (
                "check",
                VillainNode(
                    node_id="vck",
                    info_set="V_check",
                    actions=(
                        ("check", TerminalNode("t_cc", 0.0, 0.0, 0.0)),
                    ),
                ),
            ),
            (
                "bet",
                VillainNode(
                    node_id="vbet",
                    info_set="V_bet",
                    actions=(
                        ("call", TerminalNode("t_call", -1.0, 1.0, 0.0)),
                        ("fold", TerminalNode("t_fold", 1.0, -1.0, 0.0)),
                    ),
                ),
            ),
        ),
    )
    hero_strategy = HeroStrategy(probabilities={"H1": {"check": 1.0, "bet": 0.0}})
    return GameTree(root=root), hero_strategy


def test_off_path_action_variation_is_reported_separately():
    tree, hero_strategy = _off_path_tree()
    result = solve_exact_response(tree, hero_strategy)

    # V_bet is never reached, so both of its actions are optimal (free).
    assert "V_bet" in result.best_response_action_variation
    assert "V_bet" in result.off_path_info_sets
    # The reached information set is not off-path.
    assert "V_check" not in result.off_path_info_sets


def test_enumeration_safety_limit_is_enforced():
    tree, hero_strategy = _indifference_tree()  # two Villain pure strategies

    with pytest.raises(ValueError, match="safety limit"):
        enumerate_villain_pure_strategies(tree, max_pure_strategies=1)
    with pytest.raises(ValueError, match="safety limit"):
        solve_exact_response(tree, hero_strategy, max_pure_strategies=1)

    # At or above the true size the enumeration proceeds.
    result = solve_exact_response(tree, hero_strategy, max_pure_strategies=2)
    assert result.num_villain_pure_strategies == 2


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -1.0])
def test_solve_exact_response_rejects_invalid_tolerance(bad):
    tree, hero_strategy = _indifference_tree()
    with pytest.raises(ValueError, match="tolerance"):
        solve_exact_response(tree, hero_strategy, tolerance=bad)


# ---------------------------------------------------------------------------
# Value/bluff example
# ---------------------------------------------------------------------------


def test_value_bluff_terminals_are_zero_sum_without_rake():
    tree = build_value_bluff_river(rate=0.0, cap=None)
    for terminal in iter_terminals(tree.root):
        total = terminal.hero_ev + terminal.villain_ev + terminal.house_rake
        assert total == pytest.approx(0.0, abs=TOL)


def test_value_bluff_expected_payoffs_are_zero_sum_without_rake():
    tree = build_value_bluff_river(rate=0.0, cap=None)
    result = solve_exact_response(tree, default_hero_strategy())
    # At the worst-for-Hero optimum the expected triple must still sum to zero.
    total = result.ev_h_worst + result.villain_max_ev + result.expected_house_rake_worst
    assert total == pytest.approx(0.0, abs=TOL)


def test_value_bluff_strategy_can_make_a_villain_info_set_vary():
    tree = build_value_bluff_river(rate=0.0, cap=None)
    result = solve_exact_response(tree, indifference_hero_strategy())
    # Villain trash is indifferent between checking and bluff-betting.
    assert "V_check_trash" in result.best_response_action_variation
    assert result.best_response_action_variation["V_check_trash"] == ["bet", "check"]
    # The variation is genuine on-path indifference, not off-path freedom.
    assert "V_check_trash" not in result.off_path_info_sets
    assert len(result.best_response_strategies) > 1
    # In this rake-free (zero-sum) tree, indifference for Villain implies the
    # same Hero EV across the tie; a Hero-EV spread requires action-dependent
    # rake (covered by the dedicated indifference-tree test above).
    assert result.ev_h_worst == pytest.approx(result.ev_h_best)


def test_value_bluff_with_rake_produces_positive_house_take():
    tree = build_value_bluff_river(rate=0.05, cap=3.0)
    result = solve_exact_response(tree, default_hero_strategy())
    assert result.best_response_strategies  # non-empty correspondence
    assert result.ev_h_worst <= result.ev_h_best + TOL
    assert result.expected_house_rake_worst >= 0.0
    assert result.expected_house_rake_best >= 0.0


# ---------------------------------------------------------------------------
# True nuts-chop benchmark
# ---------------------------------------------------------------------------


def test_nuts_chop_every_showdown_is_a_chop():
    tree = build_nuts_chop_river(rate=0.05, cap=3.0)
    for terminal in iter_terminals(tree.root):
        if "fold" in terminal.node_id:
            continue  # fold terminals award the whole pot, not a chop
        # On a chop both players net the same amount.
        assert terminal.hero_ev == pytest.approx(terminal.villain_ev)


def test_nuts_chop_terminals_are_zero_sum_without_rake():
    tree = build_nuts_chop_river(rate=0.0, cap=None)
    for terminal in iter_terminals(tree.root):
        total = terminal.hero_ev + terminal.villain_ev + terminal.house_rake
        assert total == pytest.approx(0.0, abs=TOL)


def test_nuts_chop_raked_line_payoffs():
    terminals = _terminals_by_id(build_nuts_chop_river(rate=0.05, cap=3.0))

    # check-check: pot 20, rake 1, each nets (20 - 1) / 2 - 10 = -0.5.
    cc = terminals["T_checkcheck_a"]
    assert cc.house_rake == pytest.approx(1.0)
    assert cc.hero_ev == pytest.approx(-0.5)
    assert cc.villain_ev == pytest.approx(-0.5)

    # bet-call: pot 40, rake 2, each nets (40 - 2) / 2 - 20 = -1.0.
    bet_call = terminals["T_betcall_a"]
    assert bet_call.house_rake == pytest.approx(2.0)
    assert bet_call.hero_ev == pytest.approx(-1.0)
    assert bet_call.villain_ev == pytest.approx(-1.0)

    # bet-fold: no rake, Hero wins Villain's committed 10.
    bet_fold = terminals["T_betfold_a"]
    assert bet_fold.house_rake == 0.0
    assert bet_fold.hero_ev == pytest.approx(10.0)
    assert bet_fold.villain_ev == pytest.approx(-10.0)


def test_nuts_chop_default_best_response():
    tree = build_nuts_chop_river(rate=0.05, cap=3.0)
    result = solve_exact_response(tree, nuts_chop_hero_strategy())

    # Villain minimises rake loss: check back, and call (not raise) a bet.
    # Per holding: 0.4 * (-0.5) + 0.6 * (-1.0) = -0.8.
    assert result.villain_max_ev == pytest.approx(-0.8)
    assert result.ev_h_worst == pytest.approx(-0.8)
    assert result.ev_h_best == pytest.approx(-0.8)
    # Expected rake: 0.4 * 1.0 + 0.6 * 2.0 = 1.6.
    assert result.expected_house_rake_worst == pytest.approx(1.6)
    # The best response is unique, with no off-path freedom.
    assert result.best_response_action_variation == {}
    assert result.off_path_info_sets == []
