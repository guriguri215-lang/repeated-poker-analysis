"""Build the river one-street betting tree for a scenario (betting-tree v1).

This is the construction/accounting helper for *betting-tree mode* of the JSON
river scenario input. It is split out of :mod:`repeated_poker.scenario_io`
because the terminal accounting, the seven betting lines, and the betting-tree
information-set naming would otherwise make that module hard to read. Parsing and
validation stay in ``scenario_io``; this module only builds the
:class:`~repeated_poker.game.GameTree` and the baseline
:class:`~repeated_poker.game.HeroStrategy` from an already-validated scenario.

Scope (v1): a single river street with one OOP bet size, one IP stab size after
an OOP check, and one IP raise size versus an OOP bet. There is no re-raise, no
multiple sizes per node, no nested betting trees, and no street transitions.

The action tree per ``(hero, villain)`` matchup pair is::

    OOP_first
      check -> IP_after_OOP_check
                 check -> check-check showdown
                 bet   -> OOP_vs_IP_bet
                            call -> showdown (both invest the IP stab)
                            fold -> Hero wins Villain's initial commitment
      bet   -> IP_vs_OOP_bet
                 call  -> showdown (both invest the OOP bet)
                 fold  -> Villain wins Hero's initial commitment
                 raise -> OOP_vs_IP_raise
                            call -> showdown (both invest the raise total)
                            fold -> Hero wins Villain's initial commitment + bet

Information sets follow the same observation model as matrix mode: Villain knows
its own bucket but not Hero's, so each Villain information set is keyed by the
Villain id and shared across Hero buckets; Hero knows its own bucket but not
Villain's, so each Hero information set is keyed by the Hero id and shared across
Villain buckets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

from .game import ChanceNode, GameTree, HeroNode, HeroStrategy, VillainNode, validate_tree
from .payoffs import (
    HERO,
    VILLAIN,
    make_equity_showdown_terminal,
    make_fold_terminal,
    make_showdown_terminal,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .scenario_io import RiverScenario

# Information-set name prefixes for the betting tree (kept distinct from the
# simple-tree names so existing modes stay backward compatible).
HERO_AFTER_OOP_CHECK = "IP_after_OOP_check"
HERO_VS_OOP_BET = "IP_vs_OOP_bet"
OOP_FIRST = "OOP_first"
OOP_VS_IP_BET = "OOP_vs_IP_bet"
OOP_VS_IP_RAISE = "OOP_vs_IP_raise"


def build_betting_tree_from_scenario(
    scenario: "RiverScenario",
) -> Tuple[GameTree, HeroStrategy]:
    """Return the betting-tree ``GameTree`` and baseline ``HeroStrategy``.

    ``scenario`` must be a validated betting-tree-mode scenario (matrix mode with
    a ``betting_tree`` and per-hand ``baseline_strategies``).
    """

    hero_hands = scenario.hero_range.hands
    villain_hands = scenario.villain_range.hands
    use_equity = scenario.equity_matrix is not None
    matrix = scenario.equity_matrix if use_equity else scenario.showdown_matrix

    betting = scenario.betting_tree
    oop_bet = betting.oop_bet_size
    ip_bet = betting.ip_bet_after_check_size
    ip_raise = betting.ip_raise_size

    hero_initial = scenario.initial_commitment.hero
    villain_initial = scenario.initial_commitment.villain
    rate = scenario.rake.rate
    cap = scenario.rake.cap

    def showdown_terminal(node_id, hero_invested, villain_invested, cell):
        pot = hero_invested + villain_invested
        if use_equity:
            return make_equity_showdown_terminal(
                node_id, pot, hero_invested, villain_invested, cell, rate, cap
            )
        return make_showdown_terminal(
            node_id, pot, hero_invested, villain_invested, cell, rate, cap
        )

    # Baseline Hero strategy: one distribution per Hero decision point per bucket.
    hero_probabilities: Dict[str, Dict[str, float]] = {}
    for hero in hero_hands:
        strategies = hero.baseline_strategies
        hero_probabilities[f"{HERO_AFTER_OOP_CHECK}::{hero.hand_id}"] = dict(
            strategies["after_oop_check"]
        )
        hero_probabilities[f"{HERO_VS_OOP_BET}::{hero.hand_id}"] = dict(
            strategies["vs_oop_bet"]
        )

    children: List[Tuple[float, VillainNode]] = []
    for hero in hero_hands:
        hero_id = hero.hand_id
        for villain in villain_hands:
            villain_id = villain.hand_id
            cell = matrix[hero_id][villain_id]
            suffix = f"::{hero_id}__{villain_id}"

            # OOP check branch -------------------------------------------------
            # A. check -> IP check -> check-check showdown.
            line_a = showdown_terminal(
                f"T_check_check{suffix}", hero_initial, villain_initial, cell
            )
            # C. check -> IP bet -> OOP call -> showdown (both invest the stab).
            line_c = showdown_terminal(
                f"T_check_bet_call{suffix}",
                hero_initial + ip_bet,
                villain_initial + ip_bet,
                cell,
            )
            # B. check -> IP bet -> OOP fold: Hero wins Villain's initial chips;
            # the uncalled IP stab is returned.
            line_b = make_fold_terminal(
                f"T_check_bet_fold{suffix}", HERO, villain_initial
            )
            oop_vs_ip_bet = VillainNode(
                node_id=f"oop_vs_ip_bet{suffix}",
                info_set=f"{OOP_VS_IP_BET}::{villain_id}",
                actions=(("call", line_c), ("fold", line_b)),
            )
            ip_after_check = HeroNode(
                node_id=f"ip_after_check{suffix}",
                info_set=f"{HERO_AFTER_OOP_CHECK}::{hero_id}",
                actions=(("check", line_a), ("bet", oop_vs_ip_bet)),
            )

            # OOP bet branch ---------------------------------------------------
            # E. bet -> IP call -> showdown (both invest the OOP bet).
            line_e = showdown_terminal(
                f"T_bet_call{suffix}",
                hero_initial + oop_bet,
                villain_initial + oop_bet,
                cell,
            )
            # D. bet -> IP fold: Villain wins Hero's initial chips; uncalled OOP
            # bet is returned.
            line_d = make_fold_terminal(f"T_bet_fold{suffix}", VILLAIN, hero_initial)
            # G. bet -> IP raise -> OOP call -> showdown (both invest the raise
            # total).
            line_g = showdown_terminal(
                f"T_bet_raise_call{suffix}",
                hero_initial + ip_raise,
                villain_initial + ip_raise,
                cell,
            )
            # F. bet -> IP raise -> OOP fold: Hero wins Villain's committed chips
            # (initial + the OOP bet); the uncalled raise increment is returned.
            line_f = make_fold_terminal(
                f"T_bet_raise_fold{suffix}", HERO, villain_initial + oop_bet
            )
            oop_vs_ip_raise = VillainNode(
                node_id=f"oop_vs_ip_raise{suffix}",
                info_set=f"{OOP_VS_IP_RAISE}::{villain_id}",
                actions=(("call", line_g), ("fold", line_f)),
            )
            ip_vs_oop_bet = HeroNode(
                node_id=f"ip_vs_oop_bet{suffix}",
                info_set=f"{HERO_VS_OOP_BET}::{hero_id}",
                actions=(
                    ("call", line_e),
                    ("fold", line_d),
                    ("raise", oop_vs_ip_raise),
                ),
            )

            oop_first = VillainNode(
                node_id=f"oop_first{suffix}",
                info_set=f"{OOP_FIRST}::{villain_id}",
                actions=(("check", ip_after_check), ("bet", ip_vs_oop_bet)),
            )
            children.append((hero.weight * villain.weight, oop_first))

    root = ChanceNode(node_id="hand_matchup", children=tuple(children))
    tree = GameTree(root=root)
    # Guard the chance probabilities, terminal invariants, information-set
    # consistency, and the perfect-recall structure before the tree leaves here.
    validate_tree(tree)

    baseline_hero_strategy = HeroStrategy(probabilities=hero_probabilities)
    return tree, baseline_hero_strategy
