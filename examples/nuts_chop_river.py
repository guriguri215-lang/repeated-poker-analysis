"""A true nuts-chop river benchmark with rake.

The board itself is the nuts: whatever legal hole cards either player holds,
every showdown is a chop.  Nature deals Villain one of two equally likely
holdings; both chop at showdown, so Villain's private cards never change the
result and exist only to demonstrate that *any* dealt combination chops.  Hero
is locked and does not observe Villain's cards, so the Hero decisions share one
information set across both deals.

Lines exercised, all with rake on the called / showdown pots:

* ``check-check``  -- both check, chop;
* ``bet-call``     -- Hero bets, Villain calls, chop;
* ``bet-fold``     -- Hero bets, Villain folds;
* a check-then-bet line and a bet-then-raise line that add Villain ``bet`` and
  ``raise`` actions and the matching Hero ``call`` / ``fold`` responses.

Because every showdown chops, committing more money only burns more rake, so a
rake-aware Villain minimises the pot.  Money is tracked as net chips over the
whole hand, so with ``rate == 0`` every terminal satisfies
``hero_ev + villain_ev + house_rake == 0``.
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


def _hero_subtree(holding: str, rate: float, cap: float | None) -> HeroNode:
    """Build Hero's river decision for one (always-chopping) Villain holding."""

    # Hero checks -> Villain may check (showdown chop) or bet (Hero calls/folds).
    check_branch = VillainNode(
        node_id=f"V_check_{holding}",
        info_set=f"V_check_{holding}",
        actions=(
            (
                "check",
                make_showdown_terminal(
                    f"T_checkcheck_{holding}", POT0, 10.0, 10.0, CHOP, rate, cap
                ),
            ),
            (
                "bet",
                HeroNode(
                    node_id=f"H_vs_bet_{holding}",
                    info_set="H_vs_bet",
                    actions=(
                        (
                            "call",
                            make_showdown_terminal(
                                f"T_checkbetcall_{holding}",
                                POT0 + 2 * BET,
                                20.0,
                                20.0,
                                CHOP,
                                rate,
                                cap,
                            ),
                        ),
                        (
                            "fold",
                            make_fold_terminal(
                                f"T_checkbetfold_{holding}", VILLAIN, 10.0
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    # Hero bets -> Villain may fold, call (showdown chop), or raise.
    bet_branch = VillainNode(
        node_id=f"V_bet_{holding}",
        info_set=f"V_bet_{holding}",
        actions=(
            ("fold", make_fold_terminal(f"T_betfold_{holding}", HERO, 10.0)),
            (
                "call",
                make_showdown_terminal(
                    f"T_betcall_{holding}",
                    POT0 + 2 * BET,
                    20.0,
                    20.0,
                    CHOP,
                    rate,
                    cap,
                ),
            ),
            (
                "raise",
                HeroNode(
                    node_id=f"H_vs_raise_{holding}",
                    info_set="H_vs_raise",
                    actions=(
                        (
                            "call",
                            make_showdown_terminal(
                                f"T_betraisecall_{holding}",
                                POT0 + 2 * RAISE_TO,
                                40.0,
                                40.0,
                                CHOP,
                                rate,
                                cap,
                            ),
                        ),
                        (
                            "fold",
                            make_fold_terminal(
                                f"T_betraisefold_{holding}", VILLAIN, 20.0
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    return HeroNode(
        node_id=f"H1_{holding}",
        info_set="H1",
        actions=(("check", check_branch), ("bet", bet_branch)),
    )


def build_nuts_chop_river(rate: float = 0.05, cap: float | None = 3.0) -> GameTree:
    """Build the nuts-chop river game tree with the given rake rule.

    Two equally likely Villain holdings both chop at showdown, illustrating
    that any legal hole-card combination on this nut board produces a chop.
    """

    root = ChanceNode(
        node_id="deal",
        children=(
            (0.5, _hero_subtree("a", rate, cap)),
            (0.5, _hero_subtree("b", rate, cap)),
        ),
    )
    return GameTree(root=root)


def default_hero_strategy() -> HeroStrategy:
    """A fixed Hero strategy: Hero bets 60% on the chop board and always calls."""

    return HeroStrategy(
        probabilities={
            "H1": {"check": 0.4, "bet": 0.6},
            "H_vs_bet": {"call": 1.0, "fold": 0.0},
            "H_vs_raise": {"call": 1.0, "fold": 0.0},
        }
    )


def main() -> None:
    print("# Nuts-chop best response (rake 5%, cap 3):")
    tree = build_nuts_chop_river()
    result = solve_exact_response(tree, default_hero_strategy())
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
