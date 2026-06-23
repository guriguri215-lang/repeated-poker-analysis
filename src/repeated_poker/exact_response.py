"""Exact Villain best-response analysis against a fully fixed Hero strategy.

Hero is locked at every Hero information set.  Villain keeps every legal
action and must play one action per information set (perfect recall).  Because
the game is small and finite, this module enumerates *every* Villain pure
strategy exactly rather than approximating with CFR.

For a finite extensive-form game with perfect recall and expected-utility
maximisation, at least one pure best response always exists, so enumeration is
sufficient and exact.  When Villain is indifferent, several pure responses (and
their mixtures) tie; the report exposes the whole correspondence and the
resulting Hero-EV interval.

This v0 enumerator is intended for *small abstract trees*.  It materialises the
entire Villain pure-strategy space, whose size is the product of the number of
legal actions over the Villain information sets, and is therefore guarded by an
explicit ``max_pure_strategies`` limit.  It is not yet the scalable,
range-based response engine described in the implementation plan.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .game import (
    Action,
    ChanceNode,
    GameTree,
    HeroNode,
    HeroStrategy,
    InfoSetId,
    Node,
    TerminalNode,
    VillainNode,
    collect_villain_info_sets,
    require_valid_tolerance,
    validate_hero_strategy,
    validate_tree,
)

# A Villain pure strategy maps each Villain information set to one action.
VillainPureStrategy = Dict[InfoSetId, Action]

# An expected payoff triple (hero_ev, villain_ev, house_rake).
Payoff = Tuple[float, float, float]

# Conservative default ceiling on the Villain pure-strategy space this v0 exact
# enumerator is willing to materialise.  Raise it deliberately for larger trees.
DEFAULT_MAX_PURE_STRATEGIES = 100_000


@dataclass(frozen=True)
class StrategyEvaluation:
    """The exact expected payoff of one Villain pure strategy."""

    villain_strategy: VillainPureStrategy
    hero_ev: float
    villain_ev: float
    house_rake: float


@dataclass
class BestResponseResult:
    """The full Villain best-response correspondence against a fixed Hero.

    ``best_response_action_variation`` lists the Villain information sets whose
    chosen action differs across the globally optimal pure strategies.  This is
    *not* the same as a locally indifferent information set: an action can vary
    because Villain is genuinely indifferent there, or because the information
    set is never reached and its action is free.  ``off_path_info_sets`` reports
    the Villain information sets that carry zero reach probability under chance
    and the fixed Hero strategy, so action variation restricted to those sets is
    off-path freedom rather than on-path indifference.

    The reach computation treats Villain's own actions as free, so an
    information set that is reachable in principle but unreached under a
    particular best response (because Villain folds earlier) is *not* classified
    as off-path here; that finer conditional distinction is left to a later
    version.
    """

    villain_max_ev: float
    best_response_strategies: List[VillainPureStrategy]
    ev_h_worst: float
    ev_h_best: float
    expected_house_rake_worst: float
    expected_house_rake_best: float
    best_response_action_variation: Dict[InfoSetId, List[Action]]
    off_path_info_sets: List[InfoSetId]
    num_villain_pure_strategies: int
    all_evaluations: List[StrategyEvaluation] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable summary with English keys."""

        return {
            "villain_max_ev": self.villain_max_ev,
            "best_response_strategies": self.best_response_strategies,
            "ev_h_worst": self.ev_h_worst,
            "ev_h_best": self.ev_h_best,
            "expected_house_rake_worst": self.expected_house_rake_worst,
            "expected_house_rake_best": self.expected_house_rake_best,
            "best_response_action_variation": self.best_response_action_variation,
            "off_path_info_sets": self.off_path_info_sets,
            "num_villain_pure_strategies": self.num_villain_pure_strategies,
        }


def count_villain_pure_strategies(tree: GameTree) -> int:
    """Return the size of the Villain pure-strategy space without building it.

    The size is the product of the number of legal actions over the Villain
    information sets, never over individual nodes: a single information set
    contributes a single shared action choice.
    """

    villain_info_sets = collect_villain_info_sets(tree)
    return math.prod(len(actions) for actions in villain_info_sets.values())


def enumerate_villain_pure_strategies(
    tree: GameTree, max_pure_strategies: int = DEFAULT_MAX_PURE_STRATEGIES
) -> List[VillainPureStrategy]:
    """Return every Villain pure strategy as an info-set -> action mapping.

    The count is the product of the number of legal actions over the Villain
    information sets, never over individual nodes: a single information set
    yields a single shared action.

    The strategy-space size is computed first and checked against
    ``max_pure_strategies``; an infeasible request raises :class:`ValueError`
    *before* any list is allocated.
    """

    size = count_villain_pure_strategies(tree)
    if size > max_pure_strategies:
        raise ValueError(
            f"Villain pure-strategy space has {size} strategies, exceeding the "
            f"safety limit max_pure_strategies={max_pure_strategies}. This v0 "
            "exact enumerator targets small abstract trees; raise the limit "
            "deliberately only if you truly intend to enumerate this many."
        )

    villain_info_sets = collect_villain_info_sets(tree)
    info_set_ids = sorted(villain_info_sets)
    action_lists = [villain_info_sets[i] for i in info_set_ids]
    strategies: List[VillainPureStrategy] = []
    for combo in itertools.product(*action_lists):
        strategies.append(dict(zip(info_set_ids, combo)))
    return strategies


def _expected_payoff(
    node: Node, hero_strategy: HeroStrategy, villain_strategy: VillainPureStrategy
) -> Payoff:
    """Exact expected payoff triple for the subtree rooted at ``node``."""

    if isinstance(node, TerminalNode):
        return (node.hero_ev, node.villain_ev, node.house_rake)

    if isinstance(node, ChanceNode):
        hero = villain = rake = 0.0
        for prob, child in node.children:
            ch, cv, cr = _expected_payoff(child, hero_strategy, villain_strategy)
            hero += prob * ch
            villain += prob * cv
            rake += prob * cr
        return (hero, villain, rake)

    if isinstance(node, HeroNode):
        hero = villain = rake = 0.0
        for action, child in node.actions:
            prob = hero_strategy.action_probability(node.info_set, action)
            if prob == 0.0:
                continue
            ch, cv, cr = _expected_payoff(child, hero_strategy, villain_strategy)
            hero += prob * ch
            villain += prob * cv
            rake += prob * cr
        return (hero, villain, rake)

    if isinstance(node, VillainNode):
        chosen = villain_strategy[node.info_set]
        for action, child in node.actions:
            if action == chosen:
                return _expected_payoff(child, hero_strategy, villain_strategy)
        raise ValueError(
            f"Villain strategy chose {chosen!r} at {node.info_set!r}, "
            "which is not a legal action there"
        )

    raise TypeError(f"unknown node type: {type(node)!r}")


def solve_exact_response(
    tree: GameTree,
    hero_strategy: HeroStrategy,
    tolerance: float = 1e-9,
    max_pure_strategies: int = DEFAULT_MAX_PURE_STRATEGIES,
) -> BestResponseResult:
    """Compute Villain's exact best-response correspondence against fixed Hero.

    Validates the tree and the Hero strategy, checks the Villain pure-strategy
    space against ``max_pure_strategies``, enumerates every Villain pure
    strategy, and reports Villain's maximum EV, the full set of optimal pure
    strategies, the Hero-EV interval over those optima, the information sets
    whose action varies across optima, the off-path Villain information sets,
    and the expected rake at the Hero-EV extremes.
    """

    require_valid_tolerance(tolerance)
    validate_tree(tree, tolerance=tolerance)
    validate_hero_strategy(tree, hero_strategy, tolerance=tolerance)

    evaluations: List[StrategyEvaluation] = []
    for villain_strategy in enumerate_villain_pure_strategies(
        tree, max_pure_strategies=max_pure_strategies
    ):
        hero_ev, villain_ev, house_rake = _expected_payoff(
            tree.root, hero_strategy, villain_strategy
        )
        evaluations.append(
            StrategyEvaluation(
                villain_strategy=villain_strategy,
                hero_ev=hero_ev,
                villain_ev=villain_ev,
                house_rake=house_rake,
            )
        )

    villain_max_ev = max(e.villain_ev for e in evaluations)
    best = [e for e in evaluations if e.villain_ev >= villain_max_ev - tolerance]

    worst = min(best, key=lambda e: e.hero_ev)
    best_for_hero = max(best, key=lambda e: e.hero_ev)

    action_variation = _best_response_action_variation(best)
    reach = _villain_info_set_reach(tree.root, hero_strategy)
    off_path = sorted(
        info_set for info_set, weight in reach.items() if weight <= tolerance
    )

    return BestResponseResult(
        villain_max_ev=villain_max_ev,
        best_response_strategies=[e.villain_strategy for e in best],
        ev_h_worst=worst.hero_ev,
        ev_h_best=best_for_hero.hero_ev,
        expected_house_rake_worst=worst.house_rake,
        expected_house_rake_best=best_for_hero.house_rake,
        best_response_action_variation=action_variation,
        off_path_info_sets=off_path,
        num_villain_pure_strategies=len(evaluations),
        all_evaluations=evaluations,
    )


def _best_response_action_variation(
    best: List[StrategyEvaluation],
) -> Dict[InfoSetId, List[Action]]:
    """Return Villain info sets whose action varies across best pure responses."""

    actions_by_info_set: Dict[InfoSetId, set] = {}
    for evaluation in best:
        for info_set, action in evaluation.villain_strategy.items():
            actions_by_info_set.setdefault(info_set, set()).add(action)
    return {
        info_set: sorted(actions)
        for info_set, actions in actions_by_info_set.items()
        if len(actions) > 1
    }


def _villain_info_set_reach(
    node: Node, hero_strategy: HeroStrategy, weight: float = 1.0, reach=None
) -> Dict[InfoSetId, float]:
    """Accumulate reach weight per Villain info set under chance and Hero.

    Chance and Hero edges scale the weight; Villain edges are traversed without
    scaling (Villain's own choices are treated as free).  An information set
    whose total weight is zero is unreachable given chance and the fixed Hero
    strategy, hence off-path.  Every Villain information set in the tree is
    registered, even those reached only through zero-weight branches.
    """

    if reach is None:
        reach = {}
    if isinstance(node, TerminalNode):
        return reach
    if isinstance(node, ChanceNode):
        for prob, child in node.children:
            _villain_info_set_reach(child, hero_strategy, weight * prob, reach)
        return reach
    if isinstance(node, HeroNode):
        for action, child in node.actions:
            prob = hero_strategy.action_probability(node.info_set, action)
            _villain_info_set_reach(child, hero_strategy, weight * prob, reach)
        return reach
    if isinstance(node, VillainNode):
        reach[node.info_set] = reach.get(node.info_set, 0.0) + weight
        for _, child in node.actions:
            _villain_info_set_reach(child, hero_strategy, weight, reach)
        return reach
    raise TypeError(f"unknown node type: {type(node)!r}")
