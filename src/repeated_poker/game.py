"""Minimal data model for a finite two-player extensive-form poker game.

The model represents a single analysed spot as a game tree with four node
kinds:

* :class:`ChanceNode`   -- nature draws an outcome with fixed probabilities;
* :class:`HeroNode`     -- a Hero decision node belonging to a Hero info set;
* :class:`VillainNode`  -- a Villain decision node belonging to a Villain info
  set;
* :class:`TerminalNode` -- a leaf carrying the net payoff triple
  ``(hero_ev, villain_ev, house_rake)``.

Hero is locked: a fixed mixed strategy is supplied at every Hero info set.
Villain keeps every legal action.  A player must choose the *same* action at
every node that shares an information set.  Utilities are net chip results over
the whole hand, so a rake-free game satisfies
``hero_ev + villain_ev + house_rake == 0`` at every terminal.

Perfect recall remains an *input contract*: the caller is responsible for
supplying a tree whose information sets are consistent with a player never
forgetting its own past actions or observations.  :func:`validate_tree` adds a
structural guard that rejects the same player's information-set ID appearing
twice on a single root-to-terminal path (a repeated-information-set /
absent-mindedness error).  That guard is a necessary condition, not a complete
proof of perfect recall.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Mapping, Tuple, Union

Action = str
InfoSetId = str
NodeId = str


# ---------------------------------------------------------------------------
# Shared numeric-validation helpers
# ---------------------------------------------------------------------------


def require_finite(value: float, name: str) -> None:
    """Raise :class:`ValueError` if ``value`` is NaN or infinite."""

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def require_valid_tolerance(value: float, name: str = "tolerance") -> None:
    """Raise :class:`ValueError` if a tolerance is non-finite or negative."""

    require_finite(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


@dataclass(frozen=True)
class TerminalNode:
    """A leaf of the tree carrying the net payoff triple."""

    node_id: NodeId
    hero_ev: float
    villain_ev: float
    house_rake: float


@dataclass(frozen=True)
class ChanceNode:
    """A nature node. ``children`` is a tuple of ``(probability, child)``."""

    node_id: NodeId
    children: Tuple[Tuple[float, "Node"], ...]


@dataclass(frozen=True)
class HeroNode:
    """A Hero decision node. ``actions`` is a tuple of ``(action, child)``."""

    node_id: NodeId
    info_set: InfoSetId
    actions: Tuple[Tuple[Action, "Node"], ...]


@dataclass(frozen=True)
class VillainNode:
    """A Villain decision node. ``actions`` is a tuple of ``(action, child)``."""

    node_id: NodeId
    info_set: InfoSetId
    actions: Tuple[Tuple[Action, "Node"], ...]


Node = Union[TerminalNode, ChanceNode, HeroNode, VillainNode]


@dataclass(frozen=True)
class GameTree:
    """A finite extensive-form game rooted at ``root``."""

    root: Node


@dataclass
class HeroStrategy:
    """A fixed mixed strategy for Hero, given per information set.

    ``probabilities[info_set][action]`` is the probability that Hero plays
    ``action`` at ``info_set``.
    """

    probabilities: Dict[InfoSetId, Dict[Action, float]]

    def action_probability(self, info_set: InfoSetId, action: Action) -> float:
        return self.probabilities.get(info_set, {}).get(action, 0.0)


@dataclass
class VillainStrategy:
    """A fixed mixed strategy for Villain, symmetric to :class:`HeroStrategy`.

    ``probabilities[info_set][action]`` is the probability that Villain plays
    ``action`` at ``info_set``.  This represents a fixed Villain baseline; the
    exact best-response engine in :mod:`repeated_poker.exact_response` does not
    use it and instead optimises Villain freely.
    """

    probabilities: Dict[InfoSetId, Dict[Action, float]]

    def action_probability(self, info_set: InfoSetId, action: Action) -> float:
        return self.probabilities.get(info_set, {}).get(action, 0.0)


# ---------------------------------------------------------------------------
# Tree traversal helpers
# ---------------------------------------------------------------------------


def iter_nodes(node: Node) -> Iterator[Node]:
    """Yield every node in the subtree rooted at ``node`` (pre-order)."""

    yield node
    if isinstance(node, ChanceNode):
        for _, child in node.children:
            yield from iter_nodes(child)
    elif isinstance(node, (HeroNode, VillainNode)):
        for _, child in node.actions:
            yield from iter_nodes(child)


def iter_terminals(node: Node) -> Iterator[TerminalNode]:
    """Yield every terminal node in the subtree rooted at ``node``."""

    for current in iter_nodes(node):
        if isinstance(current, TerminalNode):
            yield current


def collect_hero_info_sets(tree: GameTree) -> Dict[InfoSetId, Tuple[Action, ...]]:
    """Return ``{info_set: legal_actions}`` for every Hero info set.

    Raises :class:`ValueError` if a Hero info set is reached by nodes whose
    legal-action sets differ, which would violate perfect recall.
    """

    return _collect_info_sets(tree, HeroNode)


def collect_villain_info_sets(tree: GameTree) -> Dict[InfoSetId, Tuple[Action, ...]]:
    """Return ``{info_set: legal_actions}`` for every Villain info set.

    Raises :class:`ValueError` on inconsistent legal-action sets.
    """

    return _collect_info_sets(tree, VillainNode)


def _collect_info_sets(tree: GameTree, node_type) -> Dict[InfoSetId, Tuple[Action, ...]]:
    info_sets: Dict[InfoSetId, Tuple[Action, ...]] = {}
    for node in iter_nodes(tree.root):
        if not isinstance(node, node_type):
            continue
        actions = tuple(action for action, _ in node.actions)
        if node.info_set in info_sets:
            if info_sets[node.info_set] != actions:
                raise ValueError(
                    f"information set {node.info_set!r} has inconsistent legal "
                    f"actions: {info_sets[node.info_set]} vs {actions}"
                )
        else:
            info_sets[node.info_set] = actions
    return info_sets


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_tree(tree: GameTree, tolerance: float = 1e-9) -> None:
    """Validate structural and numeric invariants of ``tree``.

    Checks, for every node:

    * chance probabilities are finite, non-negative, and sum to one;
    * decision nodes have at least one action and no duplicate action labels;
    * terminal payoffs are finite, ``house_rake`` is non-negative, and the
      triple satisfies ``hero_ev + villain_ev + house_rake == 0`` within
      ``tolerance``;
    * information sets have consistent legal actions for each player; and
    * no player's information-set ID repeats on a single root-to-terminal path
      (the perfect-recall structural guard).
    """

    require_valid_tolerance(tolerance)

    for node in iter_nodes(tree.root):
        if isinstance(node, TerminalNode):
            for name, value in (
                ("hero_ev", node.hero_ev),
                ("villain_ev", node.villain_ev),
                ("house_rake", node.house_rake),
            ):
                if not math.isfinite(value):
                    raise ValueError(
                        f"terminal {node.node_id!r} has non-finite {name}: {value!r}"
                    )
            if node.house_rake < 0:
                raise ValueError(
                    f"terminal {node.node_id!r} has negative house_rake "
                    f"{node.house_rake}"
                )
            total = node.hero_ev + node.villain_ev + node.house_rake
            if abs(total) > tolerance:
                raise ValueError(
                    f"terminal {node.node_id!r} violates "
                    f"hero_ev + villain_ev + house_rake == 0 (sum {total})"
                )
        elif isinstance(node, ChanceNode):
            if not node.children:
                raise ValueError(f"chance node {node.node_id!r} has no children")
            probs = [p for p, _ in node.children]
            if any(not math.isfinite(p) for p in probs):
                raise ValueError(
                    f"chance node {node.node_id!r} has a non-finite probability"
                )
            if any(p < 0 for p in probs):
                raise ValueError(
                    f"chance node {node.node_id!r} has a negative probability"
                )
            total = sum(probs)
            if abs(total - 1.0) > tolerance:
                raise ValueError(
                    f"chance node {node.node_id!r} probabilities sum to {total}, "
                    "expected 1"
                )
        elif isinstance(node, (HeroNode, VillainNode)):
            if not node.actions:
                raise ValueError(f"decision node {node.node_id!r} has no actions")
            labels = [action for action, _ in node.actions]
            if len(labels) != len(set(labels)):
                raise ValueError(
                    f"decision node {node.node_id!r} has duplicate action labels"
                )

    # Information-set consistency (raises on conflict).
    collect_hero_info_sets(tree)
    collect_villain_info_sets(tree)

    # Perfect-recall structural guard (raises on a repeated information set).
    _check_no_repeated_info_set_on_path(tree.root, frozenset(), frozenset())


def _check_no_repeated_info_set_on_path(
    node: Node, hero_seen: frozenset, villain_seen: frozenset
) -> None:
    """Reject a tree where one player's info set repeats on a single path.

    A player that revisits the same information set on one root-to-terminal
    path would have to forget that it had already acted there, which is exactly
    the repeated-information-set / absent-mindedness violation of perfect
    recall.  This is a necessary structural condition only; it does not by
    itself prove that the whole tree has perfect recall.
    """

    if isinstance(node, TerminalNode):
        return
    if isinstance(node, ChanceNode):
        for _, child in node.children:
            _check_no_repeated_info_set_on_path(child, hero_seen, villain_seen)
        return
    if isinstance(node, HeroNode):
        if node.info_set in hero_seen:
            raise ValueError(
                f"Hero information set {node.info_set!r} repeats on a single "
                "root-to-terminal path, violating the perfect-recall contract"
            )
        hero_seen = hero_seen | {node.info_set}
        for _, child in node.actions:
            _check_no_repeated_info_set_on_path(child, hero_seen, villain_seen)
        return
    if isinstance(node, VillainNode):
        if node.info_set in villain_seen:
            raise ValueError(
                f"Villain information set {node.info_set!r} repeats on a single "
                "root-to-terminal path, violating the perfect-recall contract"
            )
        villain_seen = villain_seen | {node.info_set}
        for _, child in node.actions:
            _check_no_repeated_info_set_on_path(child, hero_seen, villain_seen)
        return
    raise TypeError(f"unknown node type: {type(node)!r}")


def _validate_player_strategy(
    player: str,
    probabilities: Dict[InfoSetId, Dict[Action, float]],
    info_sets: Dict[InfoSetId, Tuple[Action, ...]],
    tolerance: float,
) -> None:
    """Validate a player's mixed strategy against the tree's information sets.

    Shared by Hero and Villain validation so the two stay at the same level.
    Every information set must carry a distribution over exactly its legal
    actions, with finite, non-negative probabilities summing to one.  A
    strategy that references an information set absent from the tree (an
    ``unknown`` information set) is rejected.
    """

    require_valid_tolerance(tolerance)

    unknown_info_sets = set(probabilities) - set(info_sets)
    if unknown_info_sets:
        raise ValueError(
            f"{player} strategy has unknown information sets "
            f"{sorted(unknown_info_sets)}"
        )

    for info_set, legal_actions in info_sets.items():
        if info_set not in probabilities:
            raise ValueError(
                f"{player} strategy is missing information set {info_set!r}"
            )
        dist = probabilities[info_set]
        unknown = set(dist) - set(legal_actions)
        if unknown:
            raise ValueError(
                f"{player} strategy for {info_set!r} has illegal actions "
                f"{sorted(unknown)}"
            )
        if any(not math.isfinite(p) for p in dist.values()):
            raise ValueError(
                f"{player} strategy for {info_set!r} has a non-finite probability"
            )
        if any(p < 0 for p in dist.values()):
            raise ValueError(
                f"{player} strategy for {info_set!r} has a negative probability"
            )
        total = sum(dist.get(action, 0.0) for action in legal_actions)
        if abs(total - 1.0) > tolerance:
            raise ValueError(
                f"{player} strategy for {info_set!r} sums to {total}, expected 1"
            )


def validate_hero_strategy(
    tree: GameTree, hero_strategy: HeroStrategy, tolerance: float = 1e-9
) -> None:
    """Validate that ``hero_strategy`` is a proper mixed strategy on ``tree``.

    Every Hero info set must be assigned a probability distribution over
    exactly its legal actions, with finite, non-negative probabilities summing
    to one.  Strategies that reference an unknown Hero information set are
    rejected.
    """

    _validate_player_strategy(
        "Hero", hero_strategy.probabilities, collect_hero_info_sets(tree), tolerance
    )


def validate_villain_strategy(
    tree: GameTree, villain_strategy: VillainStrategy, tolerance: float = 1e-9
) -> None:
    """Validate that ``villain_strategy`` is a proper mixed strategy on ``tree``.

    Every Villain info set must be assigned a probability distribution over
    exactly its legal actions, with finite, non-negative probabilities summing
    to one.  Strategies that reference an unknown Villain information set are
    rejected.
    """

    _validate_player_strategy(
        "Villain",
        villain_strategy.probabilities,
        collect_villain_info_sets(tree),
        tolerance,
    )
