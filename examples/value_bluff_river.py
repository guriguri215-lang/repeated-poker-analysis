"""A tiny value/bluff river spot with rake.

Hero always holds the nuts.  Nature deals Villain either the nuts (a chop on a
showdown) or a losing hand, each with probability 1/2.  Hero does not observe
Villain's hand, so every Hero decision lives in a single information set that is
shared across both chance branches.

This is a value/bluff spot, *not* a nuts-chop benchmark.  Hero always holds the
nuts, so Villain is at best tied (with the nuts holding) or behind (with the
trash holding); Villain is never ahead, and showdowns are not always chops.
See ``nuts_chop_river.py`` for the true nuts-chop benchmark where every showdown
chops.

The tree exercises the full action vocabulary required by the project:

* ``check-check``    -- Hero checks, Villain checks back, showdown;
* ``bet-call``       -- Hero bets, Villain calls, showdown;
* ``bet-fold``       -- Hero bets, Villain folds;
* a check-then-bet line and a bet-then-raise line that add Villain ``bet`` and
  ``raise`` actions and the matching Hero ``call`` / ``fold`` responses.

Money is tracked as net chips over the whole hand, so with ``rate == 0`` every
terminal satisfies ``hero_ev + villain_ev + house_rake == 0``.
"""

from __future__ import annotations

import json

from repeated_poker import (
    ChanceNode,
    GameTree,
    HeroNode,
    HeroStrategy,
    VillainNode,
    solve_exact_response,
)
from repeated_poker.payoffs import (
    CHOP,
    HERO,
    VILLAIN,
    make_fold_terminal,
    make_showdown_terminal,
)

POT0 = 20.0  # chips already in the pot at the start of the river (10 + 10).
BET = 10.0  # river bet size.
RAISE_TO = 30.0  # total river chips a raiser commits.


def _hero_subtree(hand: str, rate: float, cap: float | None) -> HeroNode:
    """Build Hero's river decision for one Villain hand (``"nuts"``/``"trash"``)."""

    showdown = CHOP if hand == "nuts" else HERO

    # Hero checks -> Villain may check (showdown) or bet (Hero then calls/folds).
    check_branch = VillainNode(
        node_id=f"V_check_{hand}",
        info_set=f"V_check_{hand}",
        actions=(
            (
                "check",
                make_showdown_terminal(
                    f"T_checkcheck_{hand}", POT0, 10.0, 10.0, showdown, rate, cap
                ),
            ),
            (
                "bet",
                HeroNode(
                    node_id=f"H_vs_bet_{hand}",
                    info_set="H_vs_bet",
                    actions=(
                        (
                            "call",
                            make_showdown_terminal(
                                f"T_checkbetcall_{hand}",
                                POT0 + 2 * BET,
                                20.0,
                                20.0,
                                showdown,
                                rate,
                                cap,
                            ),
                        ),
                        (
                            "fold",
                            make_fold_terminal(
                                f"T_checkbetfold_{hand}", VILLAIN, 10.0
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    # Hero bets -> Villain may fold, call (showdown), or raise (Hero calls/folds).
    bet_branch = VillainNode(
        node_id=f"V_bet_{hand}",
        info_set=f"V_bet_{hand}",
        actions=(
            ("fold", make_fold_terminal(f"T_betfold_{hand}", HERO, 10.0)),
            (
                "call",
                make_showdown_terminal(
                    f"T_betcall_{hand}",
                    POT0 + 2 * BET,
                    20.0,
                    20.0,
                    showdown,
                    rate,
                    cap,
                ),
            ),
            (
                "raise",
                HeroNode(
                    node_id=f"H_vs_raise_{hand}",
                    info_set="H_vs_raise",
                    actions=(
                        (
                            "call",
                            make_showdown_terminal(
                                f"T_betraisecall_{hand}",
                                POT0 + 2 * RAISE_TO,
                                40.0,
                                40.0,
                                showdown,
                                rate,
                                cap,
                            ),
                        ),
                        (
                            "fold",
                            make_fold_terminal(
                                f"T_betraisefold_{hand}", VILLAIN, 20.0
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    return HeroNode(
        node_id=f"H1_{hand}",
        info_set="H1",
        actions=(("check", check_branch), ("bet", bet_branch)),
    )


def build_value_bluff_river(rate: float = 0.05, cap: float | None = 3.0) -> GameTree:
    """Build the value/bluff river game tree with the given rake rule."""

    root = ChanceNode(
        node_id="deal",
        children=(
            (0.5, _hero_subtree("nuts", rate, cap)),
            (0.5, _hero_subtree("trash", rate, cap)),
        ),
    )
    return GameTree(root=root)


def default_hero_strategy() -> HeroStrategy:
    """A fixed Hero strategy: Hero bets 60% with the nuts and always calls."""

    return HeroStrategy(
        probabilities={
            "H1": {"check": 0.4, "bet": 0.6},
            "H_vs_bet": {"call": 1.0, "fold": 0.0},
            "H_vs_raise": {"call": 1.0, "fold": 0.0},
        }
    )


def indifference_hero_strategy() -> HeroStrategy:
    """A fixed Hero strategy that makes a Villain information set indifferent.

    With a rake-free tree and Hero calling a check-then-bet exactly 2/3 of the
    time, a Villain trash hand is indifferent between checking (EV -10) and
    bluff-betting (EV 10 - 30 * 2/3 = -10) at ``V_check_trash``.
    """

    return HeroStrategy(
        probabilities={
            "H1": {"check": 0.4, "bet": 0.6},
            "H_vs_bet": {"call": 2.0 / 3.0, "fold": 1.0 / 3.0},
            "H_vs_raise": {"call": 1.0, "fold": 0.0},
        }
    )


def main() -> None:
    print("# Unique best response (rake 5%, cap 3):")
    tree = build_value_bluff_river()
    result = solve_exact_response(tree, default_hero_strategy())
    print(json.dumps(result.to_dict(), indent=2))

    print("\n# Varying best response from a tie (rake 0):")
    tree = build_value_bluff_river(rate=0.0, cap=None)
    result = solve_exact_response(tree, indifference_hero_strategy())
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
