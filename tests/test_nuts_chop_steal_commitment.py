"""Nuts-board chop-only river with rake: a steal versus a calling commitment.

This is the original motivating nuts-chop steal scenario for analysing poker as
a repeated game:

* The board is the nuts, so every showdown is a chop (no value betting exists).
* The aggressor (OOP, modelled as the free-responding Villain) acts first and
  may check or shove.
* The caller (IP, modelled as the locked Hero) faces a shove and may call or
  fold.
* Rake applies and is below its cap, so a contested pot loses chips to the
  house.

The only profit source of a shove is fold equity: if the caller folds, the
aggressor steals the pot; if the caller calls, the larger pot just pays more
rake, which is worse for the aggressor than checking.

Whether the caller folds in a single hand depends on the ratio of the initial
commitment to the rake on the (much larger) called pot. Folding loses only the
caller's initial commitment ``c``; calling a chop returns the caller's equity
minus half the rake on the called pot, i.e. it loses ``rake_called / 2``. So the
caller folds in a single hand exactly when ``rake_called / 2 > c``.

* Large initial pot (``c`` large relative to the rake): calling beats folding,
  so a rational caller already calls in a single hand.
* Small initial pot with a large overbet (``rake_called / 2 > c``): folding
  beats calling, so a rational caller folds in a single hand even though the
  board is a pure chop. This is the configuration where committing to call in a
  repeated game changes the outcome: it removes the aggressor's only profit
  source (fold equity), so the aggressor's exact best response becomes a check.

The worked repeated-game case below uses initial commitment = 1, bet = 98,
rake = 5%, cap = 4. There the single-hand baseline is OOP bet / IP fold; locking
IP to call makes OOP's exact best response a check; and ``T_deadline`` is
checked for N = 10, 20, 50, 100.
"""

import pytest

from repeated_poker import (
    GameTree,
    HeroNode,
    HeroStrategy,
    VillainNode,
    calculate_adaptation_deadline,
    iter_terminals,
    solve_exact_response,
)
from repeated_poker.payoffs import CHOP, VILLAIN, make_fold_terminal, make_showdown_terminal


def build_nuts_chop_steal_tree(
    initial_commitment: float,
    bet_size: float,
    rake_rate: float = 0.05,
    rake_cap: float = 3.0,
) -> GameTree:
    """Build the OOP-acts-first nuts-chop steal tree.

    Both players have ``initial_commitment`` in the pot before the river
    (``POT0 = 2 * initial_commitment``). OOP (Villain) may check or shove
    ``bet_size``; IP (Hero) may then call or fold. Every showdown is a chop.
    """

    pot0 = 2.0 * initial_commitment
    check_check = make_showdown_terminal(
        "T_check_check",
        pot0,
        initial_commitment,
        initial_commitment,
        CHOP,
        rake_rate,
        rake_cap,
    )
    invested = initial_commitment + bet_size
    shove_call = make_showdown_terminal(
        "T_shove_call",
        pot0 + 2.0 * bet_size,
        invested,
        invested,
        CHOP,
        rake_rate,
        rake_cap,
    )
    # IP folds to the shove: OOP (Villain) wins IP's committed chips.
    shove_fold = make_fold_terminal("T_shove_fold", VILLAIN, initial_commitment)

    ip_vs_shove = HeroNode(
        node_id="ip",
        info_set="IP_vs_shove",
        actions=(("call", shove_call), ("fold", shove_fold)),
    )
    oop = VillainNode(
        node_id="oop",
        info_set="OOP_river",
        actions=(("check", check_check), ("shove", ip_vs_shove)),
    )
    return GameTree(root=oop)


def _terminals(tree: GameTree):
    return {t.node_id: t for t in iter_terminals(tree.root)}


def _always_call() -> HeroStrategy:
    return HeroStrategy({"IP_vs_shove": {"call": 1.0, "fold": 0.0}})


def _always_fold() -> HeroStrategy:
    return HeroStrategy({"IP_vs_shove": {"call": 0.0, "fold": 1.0}})


def _single_hand_caller_choice(tree: GameTree) -> str:
    """Return the caller's single-hand best action facing the shove."""

    terminals = _terminals(tree)
    call_ev = terminals["T_shove_call"].hero_ev
    fold_ev = terminals["T_shove_fold"].hero_ev
    return "call" if call_ev > fold_ev else "fold"


# ---------------------------------------------------------------------------
# 1. Large initial pot: the single-hand caller prefers to call.
# ---------------------------------------------------------------------------


def test_large_initial_pot_single_hand_caller_prefers_call():
    # c = 10, B = 20: rake_called = min(0.05 * 60, 3) = 3, rake/2 = 1.5 < c = 10.
    tree = build_nuts_chop_steal_tree(initial_commitment=10.0, bet_size=20.0)
    terminals = _terminals(tree)
    assert terminals["T_shove_call"].hero_ev == pytest.approx(-1.5)
    assert terminals["T_shove_fold"].hero_ev == pytest.approx(-10.0)
    assert _single_hand_caller_choice(tree) == "call"


# ---------------------------------------------------------------------------
# 2. Small initial pot / large bet: the single-hand caller prefers to fold.
# ---------------------------------------------------------------------------


def test_small_pot_large_bet_single_hand_caller_prefers_fold():
    # c = 1, B = 20: rake_called = min(0.05 * 42, 3) = 2.1, rake/2 = 1.05 > c = 1.
    tree = build_nuts_chop_steal_tree(initial_commitment=1.0, bet_size=20.0)
    terminals = _terminals(tree)
    assert terminals["T_check_check"].villain_ev == pytest.approx(-0.05)
    assert terminals["T_check_check"].hero_ev == pytest.approx(-0.05)
    assert terminals["T_shove_fold"].hero_ev == pytest.approx(-1.0)
    assert terminals["T_shove_fold"].villain_ev == pytest.approx(1.0)
    assert terminals["T_shove_call"].hero_ev == pytest.approx(-1.05)
    assert terminals["T_shove_call"].villain_ev == pytest.approx(-1.05)
    # Folding (-1.0) beats calling (-1.05): the caller folds in a single hand.
    assert _single_hand_caller_choice(tree) == "fold"


@pytest.mark.parametrize("bet_size", [20.0, 25.0])
def test_small_pot_large_bet_single_hand_fold_for_multiple_bets(bet_size):
    tree = build_nuts_chop_steal_tree(initial_commitment=1.0, bet_size=bet_size)
    assert _single_hand_caller_choice(tree) == "fold"


# ---------------------------------------------------------------------------
# 3. Small initial pot / large bet, caller locked to call: OOP checks.
# ---------------------------------------------------------------------------


def test_small_pot_large_bet_locked_call_makes_oop_check():
    tree = build_nuts_chop_steal_tree(initial_commitment=1.0, bet_size=20.0)
    result = solve_exact_response(tree, _always_call())
    # The calling commitment removes fold equity, so the aggressor checks.
    assert result.best_response_strategies == [{"OOP_river": "check"}]
    assert result.villain_max_ev == pytest.approx(-0.05)


def test_small_pot_large_bet_locked_fold_lets_oop_steal():
    tree = build_nuts_chop_steal_tree(initial_commitment=1.0, bet_size=20.0)
    result = solve_exact_response(tree, _always_fold())
    # A folding caller leaves the steal profitable, so the aggressor shoves.
    assert result.best_response_strategies == [{"OOP_river": "shove"}]
    assert result.villain_max_ev == pytest.approx(1.0)


def test_shove_profit_source_is_only_fold_equity_small_pot():
    terminals = _terminals(build_nuts_chop_steal_tree(1.0, 20.0))
    check_check = terminals["T_check_check"]
    shove_call = terminals["T_shove_call"]
    shove_fold = terminals["T_shove_fold"]
    # The steal beats checking, but a called shove is worse than checking.
    assert shove_fold.villain_ev > check_check.villain_ev
    assert shove_call.villain_ev < check_check.villain_ev


# ---------------------------------------------------------------------------
# Repeated-game evaluation for the c=1, bet=98, rake cap=4 case.
#
# Here the shove ("bet") is so large that a called pot reaches the rake cap.
# Terminal Hero/Villain EVs: check-check -0.05/-0.05, bet-fold -1.0/+1.0,
# bet-call -2.0/-2.0. The single-hand caller folds (-1.0 > -2.0), so the
# single-hand baseline is OOP bet / IP fold. Locking the caller to call makes
# the aggressor check. T_deadline reports how late the aggressor may switch to
# checking while the calling commitment still beats the baseline.
# ---------------------------------------------------------------------------

_BET98 = dict(initial_commitment=1.0, bet_size=98.0, rake_rate=0.05, rake_cap=4.0)


def _bet98_tree() -> GameTree:
    return build_nuts_chop_steal_tree(**_BET98)


def test_bet98_terminal_evs_match_hand_calculation():
    terminals = _terminals(_bet98_tree())
    check_check = terminals["T_check_check"]
    bet_fold = terminals["T_shove_fold"]
    bet_call = terminals["T_shove_call"]

    # check-check: pot 2, rake 0.1, each receives 0.95, invested 1.0.
    assert check_check.hero_ev == pytest.approx(-0.05)
    assert check_check.villain_ev == pytest.approx(-0.05)
    # bet-fold: uncalled bet returned, no rake.
    assert bet_fold.hero_ev == pytest.approx(-1.0)
    assert bet_fold.villain_ev == pytest.approx(1.0)
    # bet-call: pot 198, rake min(9.9, 4) = 4, each receives 97, invested 99.
    assert bet_call.hero_ev == pytest.approx(-2.0)
    assert bet_call.villain_ev == pytest.approx(-2.0)


def test_bet98_single_hand_equilibrium_is_oop_bet_ip_fold():
    tree = _bet98_tree()
    # The single-hand caller folds: fold -1.0 beats call -2.0.
    assert _single_hand_caller_choice(tree) == "fold"
    # Against a folding caller, the aggressor's exact best response is to bet.
    result = solve_exact_response(tree, _always_fold())
    assert result.best_response_strategies == [{"OOP_river": "shove"}]
    # Baseline (single-hand equilibrium) EVs: Hero -1.0, Villain +1.0.
    assert result.ev_h_worst == pytest.approx(-1.0)
    assert result.villain_max_ev == pytest.approx(1.0)


def test_bet98_locked_call_makes_oop_check():
    tree = _bet98_tree()
    result = solve_exact_response(tree, _always_call())
    # The calling commitment removes fold equity, so the aggressor checks.
    assert result.best_response_strategies == [{"OOP_river": "check"}]
    # Post-response EVs: Hero -0.05, Villain -0.05.
    assert result.ev_h_worst == pytest.approx(-0.05)
    assert result.villain_max_ev == pytest.approx(-0.05)


def _bet98_regime_hero_evs():
    """Return (baseline, pre_adaptation, post_adaptation) Hero EVs from the tree."""

    terminals = _terminals(_bet98_tree())
    baseline = terminals["T_shove_fold"].hero_ev  # OOP bet / IP fold
    pre_adaptation = terminals["T_shove_call"].hero_ev  # IP locked call, OOP still bets
    post_adaptation = terminals["T_check_check"].hero_ev  # OOP adapted to check
    return baseline, pre_adaptation, post_adaptation


def test_bet98_regime_hero_evs():
    baseline, pre_adaptation, post_adaptation = _bet98_regime_hero_evs()
    assert baseline == pytest.approx(-1.0)
    assert pre_adaptation == pytest.approx(-2.0)
    assert post_adaptation == pytest.approx(-0.05)


@pytest.mark.parametrize(
    "horizon, expected_t_deadline",
    [(10, 5), (20, 10), (50, 25), (100, 49)],
)
def test_bet98_t_deadline_for_horizons(horizon, expected_t_deadline):
    baseline, pre_adaptation, post_adaptation = _bet98_regime_hero_evs()
    result = calculate_adaptation_deadline(
        baseline_hero_ev=baseline,
        pre_adaptation_hero_ev=pre_adaptation,
        post_adaptation_hero_ev=post_adaptation,
        horizon=horizon,
        discount=1.0,
    )
    assert result.t_deadline == expected_t_deadline
