"""Rake computation and terminal-payoff builders.

Utilities are net chip results over the whole hand (amount received minus
amount invested).  The house takes ``house_rake`` from the awarded pot, so each
terminal satisfies ``hero_ev + villain_ev + house_rake == 0``.  When the rake
rate is zero the game is zero-sum.
"""

from __future__ import annotations

from typing import Optional

from .game import TerminalNode, require_finite, require_valid_tolerance

HERO = "hero"
VILLAIN = "villain"
CHOP = "chop"


def compute_rake(pot: float, rate: float, cap: Optional[float] = None) -> float:
    """Return the rake taken from ``pot``.

    The rake is ``rate * pot`` clipped to ``cap`` when a cap is supplied.

    ``pot`` must be finite and non-negative, ``rate`` must be finite and within
    ``[0, 1]``, and ``cap`` (when given) must be finite and non-negative.
    Raises :class:`ValueError` otherwise.
    """

    require_finite(pot, "pot")
    require_finite(rate, "rate")
    if pot < 0:
        raise ValueError("pot must be non-negative")
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be within [0, 1], got {rate!r}")
    raked = rate * pot
    if cap is not None:
        require_finite(cap, "cap")
        if cap < 0:
            raise ValueError("cap must be non-negative")
        raked = min(raked, cap)
    return raked


def make_showdown_terminal(
    node_id: str,
    pot: float,
    hero_invested: float,
    villain_invested: float,
    result: str,
    rate: float,
    cap: Optional[float] = None,
    tolerance: float = 1e-9,
) -> TerminalNode:
    """Build a showdown terminal where the pot is awarded after rake.

    ``result`` is one of :data:`HERO`, :data:`VILLAIN`, or :data:`CHOP`.
    ``hero_invested`` / ``villain_invested`` are the totals each player has put
    into the pot over the whole hand.

    At a showdown every chip is called, so the pot must equal the sum of both
    players' investments.  The function rejects an inconsistent
    ``pot`` versus ``hero_invested + villain_invested`` when the absolute
    difference exceeds ``tolerance``.
    """

    require_valid_tolerance(tolerance)
    require_finite(hero_invested, "hero_invested")
    require_finite(villain_invested, "villain_invested")
    invested_total = hero_invested + villain_invested
    if abs(pot - invested_total) > tolerance:
        raise ValueError(
            f"showdown pot {pot} does not equal total investment {invested_total} "
            f"within tolerance {tolerance}"
        )

    rake = compute_rake(pot, rate, cap)
    awarded = pot - rake
    if result == CHOP:
        hero_received = villain_received = awarded / 2.0
    elif result == HERO:
        hero_received, villain_received = awarded, 0.0
    elif result == VILLAIN:
        hero_received, villain_received = 0.0, awarded
    else:
        raise ValueError(f"unknown showdown result {result!r}")

    return TerminalNode(
        node_id=node_id,
        hero_ev=hero_received - hero_invested,
        villain_ev=villain_received - villain_invested,
        house_rake=rake,
    )


def make_equity_showdown_terminal(
    node_id: str,
    pot: float,
    hero_invested: float,
    villain_invested: float,
    hero_equity: float,
    rate: float,
    cap: Optional[float] = None,
    tolerance: float = 1e-9,
) -> TerminalNode:
    """Build a showdown terminal where the pot is split by ``hero_equity``.

    ``hero_equity`` is Hero's pot share *before* rake, in ``[0, 1]``: ``1.0``
    awards Hero the whole raked pot, ``0.0`` awards it all to Villain, and
    ``0.5`` is a chop. This generalises :func:`make_showdown_terminal`, whose
    discrete ``HERO`` / ``VILLAIN`` / ``CHOP`` results correspond to equities
    ``1.0`` / ``0.0`` / ``0.5``.

    As with :func:`make_showdown_terminal`, every chip is called at a showdown,
    so ``pot`` must equal ``hero_invested + villain_invested`` within
    ``tolerance``. The rake is taken from the awarded pot first, so the terminal
    still satisfies ``hero_ev + villain_ev + house_rake == 0``.
    """

    require_valid_tolerance(tolerance)
    require_finite(hero_invested, "hero_invested")
    require_finite(villain_invested, "villain_invested")
    require_finite(hero_equity, "hero_equity")
    if not 0.0 <= hero_equity <= 1.0:
        raise ValueError(f"hero_equity must be within [0, 1], got {hero_equity!r}")
    invested_total = hero_invested + villain_invested
    if abs(pot - invested_total) > tolerance:
        raise ValueError(
            f"showdown pot {pot} does not equal total investment {invested_total} "
            f"within tolerance {tolerance}"
        )

    rake = compute_rake(pot, rate, cap)
    awarded = pot - rake
    hero_received = awarded * hero_equity
    villain_received = awarded * (1.0 - hero_equity)

    return TerminalNode(
        node_id=node_id,
        hero_ev=hero_received - hero_invested,
        villain_ev=villain_received - villain_invested,
        house_rake=rake,
    )


def make_fold_terminal(node_id: str, winner: str, loser_committed: float) -> TerminalNode:
    """Build a terminal reached when one player folds.

    No rake is taken on a fold.  Uncalled chips are returned, so the winner
    gains exactly the chips the loser had committed to the pot and the loser
    loses that amount.
    """

    require_finite(loser_committed, "loser_committed")
    if loser_committed < 0:
        raise ValueError("loser_committed must be non-negative")
    if winner == HERO:
        hero_ev, villain_ev = loser_committed, -loser_committed
    elif winner == VILLAIN:
        hero_ev, villain_ev = -loser_committed, loser_committed
    else:
        raise ValueError(f"unknown winner {winner!r}")

    return TerminalNode(
        node_id=node_id,
        hero_ev=hero_ev,
        villain_ev=villain_ev,
        house_rake=0.0,
    )
