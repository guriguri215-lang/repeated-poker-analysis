"""Exact Villain best-response analysis against a fully fixed Hero strategy.

Hero is locked at every Hero information set.  Villain keeps every legal
action and must play one action per information set (perfect recall).  For a
finite extensive-form game with perfect recall and expected-utility
maximisation, at least one pure best response always exists, so both methods
below are exact.  When Villain is indifferent, several pure responses (and
their mixtures) tie; the report exposes the correspondence and the resulting
Hero-EV interval.

Two exact methods are available through :func:`solve_exact_response`:

* ``method="dp"`` (default): lexicographic backward induction over Villain
  information sets.  The primary criterion maximises Villain EV; among the
  Villain-EV-optimal actions the secondary criteria take the worst and best
  Hero EV.  Its cost is linear in the tree size, so it scales to Villain
  strategy spaces the enumerator cannot materialise.  It relies on the
  perfect-recall input contract and additionally verifies that every node of
  a Villain information set shares the same Villain action history, raising
  :class:`ValueError` otherwise.
* ``method="enumerate"``: the v0 enumerator, kept as a small-tree oracle.  It
  materialises the entire Villain pure-strategy space, whose size is the
  product of the number of legal actions over the Villain information sets,
  and is therefore guarded by an explicit ``max_pure_strategies`` limit.

Both methods compute Villain's best response to a *fixed* Hero strategy.
Neither is an equilibrium computation: no claim is made about Hero playing
optimally, and the commitment analysis built on top of this module keeps that
scope.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

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

# Conservative default ceiling on the Villain pure-strategy space the exact
# enumerator is willing to materialise.  The DP method uses the same value as
# the ceiling on materialising the best-response correspondence.  Raise it
# deliberately for larger trees.
DEFAULT_MAX_PURE_STRATEGIES = 100_000

# Method names accepted by solve_exact_response.
METHOD_DP = "dp"
METHOD_ENUMERATE = "enumerate"
_VALID_METHODS = (METHOD_DP, METHOD_ENUMERATE)


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

    ``num_best_response_strategies`` is the exact size of the best-response
    correspondence.  With ``method="enumerate"`` it always equals
    ``len(best_response_strategies)``.  With ``method="dp"`` the correspondence
    is only materialised while its generation space stays within
    ``max_pure_strategies``; beyond that, ``best_response_strategies`` holds a
    single deterministic representative and ``num_best_response_strategies``
    still reports the true count.  It is ``None`` only on manually constructed
    results.

    ``best_response_action_sets`` (``method="dp"`` only, ``None`` otherwise)
    maps each Villain information set to the actions that maximise Villain's
    continuation EV there, conditional on the information set being reached and
    assuming Villain-EV-maximising play at every later information set.  This
    describes Villain's best response to the fixed Hero strategy only; it makes
    no equilibrium claim.  Note it can differ from
    ``best_response_action_variation``: an information set avoided by some
    globally optimal strategy may vary over *all* its actions even when its
    conditional Villain-EV-optimal action is unique.

    ``all_evaluations`` is an enumeration-only diagnostic (the exact payoff of
    every Villain pure strategy); the DP method leaves it empty.
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
    num_best_response_strategies: Optional[int] = None
    best_response_action_sets: Optional[Dict[InfoSetId, List[Action]]] = None

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
            "num_best_response_strategies": self.num_best_response_strategies,
            "best_response_action_sets": self.best_response_action_sets,
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
    method: str = METHOD_DP,
) -> BestResponseResult:
    """Compute Villain's exact best-response correspondence against fixed Hero.

    Validates the tree and the Hero strategy, then solves the fixed-Hero
    Villain decision problem with the requested ``method``:

    * ``"dp"`` (default): lexicographic backward induction over Villain
      information sets.  ``max_pure_strategies`` is *not* a solve limit here;
      it only caps how large a best-response correspondence is materialised
      into ``best_response_strategies`` (a single deterministic representative
      is returned beyond the cap, with the true count in
      ``num_best_response_strategies``).
    * ``"enumerate"``: the v0 oracle.  Enumerates every Villain pure strategy
      and raises :class:`ValueError` when the space exceeds
      ``max_pure_strategies``, *before* any list is allocated.

    Both report Villain's maximum EV, optimal pure strategies, the Hero-EV
    interval over the optima, the information sets whose action varies across
    optima, the off-path Villain information sets, and the expected rake at
    the Hero-EV extremes.  Ties are resolved with the shared
    ``>= max - tolerance`` convention.

    Tolerance semantics differ between the methods.  The enumerator applies
    ``tolerance`` to the *global* Villain EV of each pure strategy; the DP
    applies it *per Villain information set*.  The two agree on trees whose
    Villain-EV ties are exact (the normal case: identical payoffs tie
    exactly).  When distinct Villain EVs fall within ``tolerance`` of each
    other, DP near-tie acceptance can chain across successive information
    sets, so a strategy in the DP correspondence (and hence the Hero-EV
    interval) may sit up to roughly ``depth * tolerance`` below the global
    optimum, which the enumerator would exclude.  With the default
    ``tolerance`` this needs distinct Villain EVs within ``1e-9`` of each
    other.  When raising ``tolerance`` as a research judgment, prefer
    ``method="enumerate"`` on small trees, or read the DP correspondence as
    per-information-set near-optimality rather than a global EV band.

    This is Villain's best response to a fixed Hero strategy, not an
    equilibrium computation.
    """

    if method not in _VALID_METHODS:
        raise ValueError(
            f"method must be one of {list(_VALID_METHODS)}, got {method!r}"
        )
    require_valid_tolerance(tolerance)
    validate_tree(tree, tolerance=tolerance)
    validate_hero_strategy(tree, hero_strategy, tolerance=tolerance)

    if method == METHOD_ENUMERATE:
        return _solve_by_enumeration(
            tree, hero_strategy, tolerance, max_pure_strategies
        )
    return _solve_by_backward_induction(
        tree, hero_strategy, tolerance, max_pure_strategies
    )


def _solve_by_enumeration(
    tree: GameTree,
    hero_strategy: HeroStrategy,
    tolerance: float,
    max_pure_strategies: int,
) -> BestResponseResult:
    """The v0 exact enumerator, kept unchanged as the DP oracle.

    Enumerates every Villain pure strategy (guarded by
    ``max_pure_strategies``), evaluates each exactly, and keeps the whole
    evaluation list as a diagnostic.
    """

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
        num_best_response_strategies=len(best),
        best_response_action_sets=None,
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


# ---------------------------------------------------------------------------
# Backward-induction (DP) method
# ---------------------------------------------------------------------------

# A Villain action history: the (information set, action) pairs of Villain's
# own decisions on the path from the root to a node, in order.
_VillainHistory = Tuple[Tuple[InfoSetId, Action], ...]


@dataclass
class _VillainInfoSetData:
    """One Villain information set as seen by the DP.

    ``history`` is the Villain action history shared by every node of the
    information set (verified; perfect recall makes it unique), ``actions``
    the legal actions in tree order, and ``nodes`` the member nodes with
    their chance/Hero reach weight (Villain edges traversed freely).
    """

    history: _VillainHistory
    actions: Tuple[Action, ...]
    nodes: List[Tuple[VillainNode, float]]


@dataclass(frozen=True)
class _InfoSetSolution:
    """Lexicographic DP value of one Villain information-set subforest.

    All values are reach-weighted contributions to the game total (weighted by
    chance and the fixed Hero strategy from the root), not per-reach
    conditional values.  ``villain_ev`` maximises Villain EV;
    ``hero_ev_worst`` / ``hero_ev_best`` are the Hero-EV extremes over the
    Villain-EV-optimal actions, with the matching expected rake.
    ``optimal_actions`` is the Villain-EV-optimal action set in tree order.
    """

    villain_ev: float
    hero_ev_worst: float
    hero_ev_best: float
    house_rake_worst: float
    house_rake_best: float
    optimal_actions: Tuple[Action, ...]


def _collect_villain_decision_data(
    tree: GameTree, hero_strategy: HeroStrategy
) -> Dict[InfoSetId, _VillainInfoSetData]:
    """Gather Villain nodes, reach weights, and action histories in one walk.

    Raises :class:`ValueError` when two nodes of the same Villain information
    set carry different Villain action histories.  Such a tree violates the
    perfect-recall input contract (Villain would have to forget its own past
    actions), and backward induction over information sets is not valid there;
    the enumeration method still is.

    Recursive; assumes the shallow trees this package targets (recursion depth
    is the tree depth).
    """

    data: Dict[InfoSetId, _VillainInfoSetData] = {}

    def walk(node: Node, weight: float, history: _VillainHistory) -> None:
        if isinstance(node, TerminalNode):
            return
        if isinstance(node, ChanceNode):
            for prob, child in node.children:
                walk(child, weight * prob, history)
            return
        if isinstance(node, HeroNode):
            for action, child in node.actions:
                prob = hero_strategy.action_probability(node.info_set, action)
                walk(child, weight * prob, history)
            return
        if isinstance(node, VillainNode):
            entry = data.get(node.info_set)
            if entry is None:
                entry = _VillainInfoSetData(
                    history=history,
                    actions=tuple(action for action, _ in node.actions),
                    nodes=[],
                )
                data[node.info_set] = entry
            elif entry.history != history:
                raise ValueError(
                    f"Villain information set {node.info_set!r} is reached with "
                    f"two different Villain action histories {entry.history} vs "
                    f"{history}, violating the perfect-recall input contract; "
                    "the dp method requires perfect recall (use "
                    "method='enumerate' for such trees)"
                )
            entry.nodes.append((node, weight))
            for action, child in node.actions:
                walk(child, weight, history + ((node.info_set, action),))
            return
        raise TypeError(f"unknown node type: {type(node)!r}")

    walk(tree.root, 1.0, ())
    return data


def _direct_payoff(node: Node, hero_strategy: HeroStrategy) -> Payoff:
    """Expected payoff collected below ``node`` before any Villain decision.

    Sums terminal payoffs over the paths from ``node`` that do not pass
    through a Villain node, weighted by chance and Hero probabilities relative
    to ``node``.  Villain subtrees contribute zero here; their value is
    accounted per information set by the DP.

    Recursive; assumes the shallow trees this package targets.
    """

    if isinstance(node, TerminalNode):
        return (node.hero_ev, node.villain_ev, node.house_rake)
    if isinstance(node, ChanceNode):
        hero = villain = rake = 0.0
        for prob, child in node.children:
            ch, cv, cr = _direct_payoff(child, hero_strategy)
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
            ch, cv, cr = _direct_payoff(child, hero_strategy)
            hero += prob * ch
            villain += prob * cv
            rake += prob * cr
        return (hero, villain, rake)
    if isinstance(node, VillainNode):
        return (0.0, 0.0, 0.0)
    raise TypeError(f"unknown node type: {type(node)!r}")


def _assignment_attains_villain_optimum(
    assignment: VillainPureStrategy,
    reach_conditions: List[Tuple[InfoSetId, _VillainHistory, FrozenSet[Action]]],
) -> bool:
    """Check whether a pure strategy belongs to the optimal correspondence.

    A Villain pure strategy attains the Villain-EV optimum exactly when it
    plays a Villain-EV-optimal action at every information set it actually
    reaches; its actions at unreached information sets are free.  Under
    perfect recall an information set is reached by the strategy exactly when
    its chance/Hero reach weight is positive *and* the strategy plays every
    action on the information set's own Villain action history, so no tree
    walk is needed here: ``reach_conditions`` carries, for each Villain
    information set with positive chance/Hero reach, its history and its
    Villain-EV-optimal action set.
    """

    for info_set, history, optimal_actions in reach_conditions:
        if assignment[info_set] not in optimal_actions and all(
            assignment[h_info_set] == h_action for h_info_set, h_action in history
        ):
            return False
    return True


def _solve_by_backward_induction(
    tree: GameTree,
    hero_strategy: HeroStrategy,
    tolerance: float,
    max_pure_strategies: int,
) -> BestResponseResult:
    """Solve the fixed-Hero Villain decision problem by lexicographic DP.

    With Hero fixed, the game is a one-player finite decision problem for
    Villain; under perfect recall the Villain information sets form a forest
    ordered by Villain's own action history, and backward induction over that
    forest is exact.  Ties use the shared ``>= max - tolerance`` convention at
    each information set.  This computes Villain's best response to the fixed
    Hero strategy; it is not an equilibrium computation.
    """

    data = _collect_villain_decision_data(tree, hero_strategy)

    # Forest edges: an information set hangs under the last (info set, action)
    # pair of its Villain action history.  Perfect recall (verified above)
    # makes that pair unique.
    children: Dict[Tuple[InfoSetId, Action], List[InfoSetId]] = {}
    for info_set, entry in data.items():
        if entry.history:
            children.setdefault(entry.history[-1], []).append(info_set)

    # Backward induction, deepest Villain histories first.
    order = sorted(data, key=lambda i: (-len(data[i].history), i))
    solutions: Dict[InfoSetId, _InfoSetSolution] = {}
    for info_set in order:
        entry = data[info_set]
        per_action: Dict[Action, List[float]] = {
            # [villain_ev, hero_worst, hero_best, rake_worst, rake_best]
            action: [0.0, 0.0, 0.0, 0.0, 0.0]
            for action in entry.actions
        }
        for node, weight in entry.nodes:
            for action, child in node.actions:
                d_hero, d_villain, d_rake = _direct_payoff(child, hero_strategy)
                acc = per_action[action]
                acc[0] += weight * d_villain
                acc[1] += weight * d_hero
                acc[2] += weight * d_hero
                acc[3] += weight * d_rake
                acc[4] += weight * d_rake
        for action in entry.actions:
            acc = per_action[action]
            for child_info_set in children.get((info_set, action), ()):
                child_solution = solutions[child_info_set]
                acc[0] += child_solution.villain_ev
                acc[1] += child_solution.hero_ev_worst
                acc[2] += child_solution.hero_ev_best
                acc[3] += child_solution.house_rake_worst
                acc[4] += child_solution.house_rake_best

        max_villain_ev = max(acc[0] for acc in per_action.values())
        optimal_actions = tuple(
            action
            for action in entry.actions
            if per_action[action][0] >= max_villain_ev - tolerance
        )
        hero_worst = hero_best = None
        rake_worst = rake_best = 0.0
        for action in optimal_actions:
            acc = per_action[action]
            if hero_worst is None or acc[1] < hero_worst:
                hero_worst = acc[1]
                rake_worst = acc[3]
            if hero_best is None or acc[2] > hero_best:
                hero_best = acc[2]
                rake_best = acc[4]
        solutions[info_set] = _InfoSetSolution(
            villain_ev=max_villain_ev,
            hero_ev_worst=hero_worst,
            hero_ev_best=hero_best,
            house_rake_worst=rake_worst,
            house_rake_best=rake_best,
            optimal_actions=optimal_actions,
        )

    # Game totals: payoff collected before any Villain decision plus the
    # value of each root information set (empty Villain history).
    root_hero, root_villain, root_rake = _direct_payoff(tree.root, hero_strategy)
    roots = [info_set for info_set, entry in data.items() if not entry.history]
    villain_max_ev = root_villain + sum(solutions[i].villain_ev for i in roots)
    ev_h_worst = root_hero + sum(solutions[i].hero_ev_worst for i in roots)
    ev_h_best = root_hero + sum(solutions[i].hero_ev_best for i in roots)
    rake_worst = root_rake + sum(solutions[i].house_rake_worst for i in roots)
    rake_best = root_rake + sum(solutions[i].house_rake_best for i in roots)

    # Actions appearing across the globally optimal pure strategies.  An
    # information set reached under *every* optimal strategy contributes its
    # Villain-EV-optimal actions; one that some optimal strategy avoids (or
    # that chance/Hero already make unreachable) is free, so every action
    # appears.
    appear: Dict[InfoSetId, List[Action]] = {}
    for info_set, entry in data.items():
        chance_hero_reach = sum(weight for _, weight in entry.nodes)
        optimal_actions = solutions[info_set].optimal_actions
        always_reached = chance_hero_reach > tolerance and all(
            solutions[h_info_set].optimal_actions == (h_action,)
            for h_info_set, h_action in entry.history
        )
        appear[info_set] = (
            list(optimal_actions) if always_reached else list(entry.actions)
        )
    action_variation = {
        info_set: sorted(actions)
        for info_set, actions in appear.items()
        if len(actions) > 1
    }

    # Exact size of the optimal correspondence: strategies must play optimal
    # actions along reached branches, while every information set under an
    # unchosen action (and everything below it) is a free assignment.
    free_count: Dict[InfoSetId, int] = {}
    optimal_count: Dict[InfoSetId, int] = {}
    for info_set in order:
        entry = data[info_set]
        free = len(entry.actions)
        for action in entry.actions:
            for child_info_set in children.get((info_set, action), ()):
                free *= free_count[child_info_set]
        free_count[info_set] = free
        total = 0
        for action in solutions[info_set].optimal_actions:
            term = 1
            for other_action in entry.actions:
                for child_info_set in children.get((info_set, other_action), ()):
                    if other_action == action:
                        term *= optimal_count[child_info_set]
                    else:
                        term *= free_count[child_info_set]
            total += term
        optimal_count[info_set] = total
    num_best_responses = math.prod(optimal_count[i] for i in roots)

    # The reach condition of each on-path information set: reached by a pure
    # strategy exactly when that strategy plays the set's own action history.
    reach_conditions = [
        (
            info_set,
            entry.history,
            frozenset(solutions[info_set].optimal_actions),
        )
        for info_set, entry in data.items()
        if sum(weight for _, weight in entry.nodes) > tolerance
    ]

    # Materialise the correspondence only while the generation space (the
    # product of the appearing-action counts, checked *before* any list is
    # allocated) stays within the safety limit; otherwise return one
    # deterministic representative and report the true count separately.
    appear_space = math.prod(len(actions) for actions in appear.values())
    strategies: List[VillainPureStrategy]
    if appear_space <= max_pure_strategies:
        info_set_ids = sorted(data)
        strategies = []
        for combo in itertools.product(*(appear[i] for i in info_set_ids)):
            assignment = dict(zip(info_set_ids, combo))
            if _assignment_attains_villain_optimum(assignment, reach_conditions):
                strategies.append(assignment)
    else:
        strategies = [
            {
                info_set: solutions[info_set].optimal_actions[0]
                for info_set in data
            }
        ]

    reach = _villain_info_set_reach(tree.root, hero_strategy)
    off_path = sorted(
        info_set for info_set, weight in reach.items() if weight <= tolerance
    )

    return BestResponseResult(
        villain_max_ev=villain_max_ev,
        best_response_strategies=strategies,
        ev_h_worst=ev_h_worst,
        ev_h_best=ev_h_best,
        expected_house_rake_worst=rake_worst,
        expected_house_rake_best=rake_best,
        best_response_action_variation=action_variation,
        off_path_info_sets=off_path,
        num_villain_pure_strategies=count_villain_pure_strategies(tree),
        all_evaluations=[],
        num_best_response_strategies=num_best_responses,
        best_response_action_sets={
            info_set: list(solutions[info_set].optimal_actions)
            for info_set in data
        },
    )
