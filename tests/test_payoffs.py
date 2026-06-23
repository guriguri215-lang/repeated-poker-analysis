"""Tests for rake computation and terminal-payoff zero-sum accounting."""

import math

import pytest

from repeated_poker.payoffs import (
    CHOP,
    HERO,
    VILLAIN,
    compute_rake,
    make_fold_terminal,
    make_showdown_terminal,
)

TOL = 1e-9


def test_compute_rake_below_cap():
    assert compute_rake(20.0, 0.05, cap=3.0) == pytest.approx(1.0)


def test_compute_rake_reaches_cap():
    # 0.05 * 80 = 4.0, capped to 3.0.
    assert compute_rake(80.0, 0.05, cap=3.0) == pytest.approx(3.0)


def test_compute_rake_without_cap():
    assert compute_rake(80.0, 0.05) == pytest.approx(4.0)


def test_compute_rake_rejects_negative():
    with pytest.raises(ValueError):
        compute_rake(-1.0, 0.05)
    with pytest.raises(ValueError):
        compute_rake(20.0, -0.05)
    with pytest.raises(ValueError):
        compute_rake(20.0, 0.05, cap=-1.0)


def test_compute_rake_rejects_rate_above_one():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        compute_rake(20.0, 1.5)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_compute_rake_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        compute_rake(bad, 0.05)
    with pytest.raises(ValueError, match="finite"):
        compute_rake(20.0, bad)
    with pytest.raises(ValueError, match="finite"):
        compute_rake(20.0, 0.05, cap=bad)


def test_showdown_terminal_rejects_inconsistent_pot():
    # pot 50 does not match the total investment of 40.
    with pytest.raises(ValueError, match="does not equal total investment"):
        make_showdown_terminal("t", 50.0, 20.0, 20.0, CHOP, rate=0.05, cap=3.0)


def test_showdown_terminal_rejects_non_finite_investment():
    with pytest.raises(ValueError, match="finite"):
        make_showdown_terminal("t", 40.0, math.inf, 20.0, CHOP, rate=0.05)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_showdown_terminal_rejects_invalid_tolerance(bad):
    with pytest.raises(ValueError, match="tolerance"):
        make_showdown_terminal("t", 40.0, 20.0, 20.0, CHOP, rate=0.05, tolerance=bad)


def test_showdown_terminal_rejects_negative_tolerance():
    with pytest.raises(ValueError, match="tolerance must be non-negative"):
        make_showdown_terminal("t", 40.0, 20.0, 20.0, CHOP, rate=0.05, tolerance=-1.0)


@pytest.mark.parametrize("winner", [HERO, VILLAIN])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_fold_terminal_rejects_non_finite_loser_committed(winner, bad):
    with pytest.raises(ValueError, match="finite"):
        make_fold_terminal("t", winner, loser_committed=bad)


@pytest.mark.parametrize("result", [CHOP, HERO, VILLAIN])
def test_showdown_terminal_is_zero_sum_with_rake_zero(result):
    terminal = make_showdown_terminal("t", 40.0, 20.0, 20.0, result, rate=0.0)
    assert terminal.house_rake == pytest.approx(0.0)
    total = terminal.hero_ev + terminal.villain_ev + terminal.house_rake
    assert total == pytest.approx(0.0, abs=TOL)


@pytest.mark.parametrize("result", [CHOP, HERO, VILLAIN])
def test_showdown_terminal_is_zero_sum_with_rake(result):
    terminal = make_showdown_terminal("t", 40.0, 20.0, 20.0, result, rate=0.05, cap=3.0)
    assert terminal.house_rake == pytest.approx(2.0)
    total = terminal.hero_ev + terminal.villain_ev + terminal.house_rake
    assert total == pytest.approx(0.0, abs=TOL)


def test_showdown_terminal_uses_cap():
    terminal = make_showdown_terminal("t", 80.0, 40.0, 40.0, CHOP, rate=0.05, cap=3.0)
    assert terminal.house_rake == pytest.approx(3.0)
    # Net for each player on a chop: (80 - 3) / 2 - 40 = -1.5.
    assert terminal.hero_ev == pytest.approx(-1.5)
    assert terminal.villain_ev == pytest.approx(-1.5)


@pytest.mark.parametrize("winner", [HERO, VILLAIN])
def test_fold_terminal_is_zero_sum(winner):
    terminal = make_fold_terminal("t", winner, loser_committed=10.0)
    assert terminal.house_rake == 0.0
    total = terminal.hero_ev + terminal.villain_ev + terminal.house_rake
    assert total == pytest.approx(0.0, abs=TOL)
