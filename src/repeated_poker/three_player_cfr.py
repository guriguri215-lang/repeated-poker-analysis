"""Tiny 3-player river commitment diagnostic prototype.

This module is intentionally isolated from the two-player exact-response core.
It models a small abstract tree with a fixed Hero policy and two strategic
opponents, then runs deterministic full-tree CFR-style regret diagnostics for
the two opponents only.

The output is a diagnostic for small abstract trees. It is not an exact
best-response solver for the two-opponent subgame, not an equilibrium
certificate, not real-money advice, and not a full poker solver.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, Literal, Mapping, Optional, Tuple, Union

Action = str
InfoSetId = str
NodeId = str
PlayerId = Literal["H", "O1", "O2"]
OpponentId = Literal["O1", "O2"]
Owner = Literal["chance", "fixed_hero", "opponent_1", "opponent_2", "terminal"]

STRATEGIC_PLAYERS: Tuple[PlayerId, ...] = ("H", "O1", "O2")
OPPONENT_PLAYERS: Tuple[OpponentId, ...] = ("O1", "O2")

DEFAULT_MAX_NODES = 200
DEFAULT_MAX_TERMINAL_NODES = 128
DEFAULT_MAX_OPPONENT_INFO_SETS_TOTAL = 24
DEFAULT_MAX_INFO_SETS_PER_OPPONENT = 12
DEFAULT_MAX_FIXED_HERO_INFO_SETS = 12
DEFAULT_MAX_ACTIONS_PER_INFO_SET = 4
DEFAULT_MAX_CHANCE_OUTCOMES_PER_NODE = 16
DEFAULT_MAX_ITERATIONS = 5_000
DEFAULT_MAX_ORACLE_PURE_PROFILES = 4_096

HARD_MAX_NODES = 500
HARD_MAX_TERMINAL_NODES = 256
HARD_MAX_OPPONENT_INFO_SETS_TOTAL = 32
HARD_MAX_INFO_SETS_PER_OPPONENT = 16
HARD_MAX_FIXED_HERO_INFO_SETS = 16
HARD_MAX_ACTIONS_PER_INFO_SET = 4
HARD_MAX_CHANCE_OUTCOMES_PER_NODE = 32
HARD_MAX_ITERATIONS = 50_000
HARD_MAX_ORACLE_PURE_PROFILES = 16_384


@dataclass(frozen=True)
class UtilityVector:
    """Terminal or expected utility vector for ``H``, ``O1``, ``O2``, and ``R``.

    ``R`` is a non-strategic accounting residual. It chooses no actions and is
    never included in regret updates.
    """

    H: float
    O1: float
    O2: float
    R: float = 0.0

    def for_player(self, player: PlayerId) -> float:
        """Return the utility component for one strategic player."""

        if player == "H":
            return self.H
        if player == "O1":
            return self.O1
        if player == "O2":
            return self.O2
        raise ValueError(f"unknown player {player!r}")

    def conservation_residual(self) -> float:
        """Return ``H + O1 + O2 + R`` for payoff-conservation diagnostics."""

        return _checked_float_sum(
            (self.H, self.O1, self.O2, self.R),
            "utility conservation residual",
        )

    def to_dict(self) -> Dict[str, float]:
        """Return a deterministic JSON-friendly dictionary."""

        return {"H": self.H, "O1": self.O1, "O2": self.O2, "R": self.R}


@dataclass(frozen=True)
class ThreePlayerTerminalNode:
    """A leaf carrying a 3-player utility vector plus optional residual."""

    node_id: NodeId
    utility: UtilityVector


@dataclass(frozen=True)
class ThreePlayerChanceNode:
    """A fixed-probability transition node."""

    node_id: NodeId
    children: Tuple[Tuple[float, "ThreePlayerNode"], ...]


@dataclass(frozen=True)
class FixedHeroNode:
    """A fixed Hero policy / node-lock transition.

    The probabilities live in the supplied fixed Hero policy. Hero is fixed
    input and is not optimized by the CFR-style diagnostic.
    """

    node_id: NodeId
    info_set: InfoSetId
    actions: Tuple[Tuple[Action, "ThreePlayerNode"], ...]


@dataclass(frozen=True)
class OpponentDecisionNode:
    """A decision node owned by one of the two strategic opponents."""

    node_id: NodeId
    owner: Literal["opponent_1", "opponent_2"]
    info_set: InfoSetId
    actions: Tuple[Tuple[Action, "ThreePlayerNode"], ...]


ThreePlayerNode = Union[
    ThreePlayerTerminalNode,
    ThreePlayerChanceNode,
    FixedHeroNode,
    OpponentDecisionNode,
]


@dataclass(frozen=True)
class ThreePlayerGameTree:
    """A small abstract 3-player river commitment tree."""

    root: ThreePlayerNode
    description: str = ""


@dataclass(frozen=True)
class BehaviorStrategy:
    """A mixed behavior strategy as ``{info_set: {action: probability}}``."""

    probabilities: Mapping[InfoSetId, Mapping[Action, float]]

    def action_probability(self, info_set: InfoSetId, action: Action) -> float:
        """Return the probability for ``action`` at ``info_set``."""

        return self.probabilities.get(info_set, {}).get(action, 0.0)


@dataclass(frozen=True)
class CfrSafetyLimits:
    """Safety limits for the tiny prototype.

    Limits are validated before regret tables or pure-profile enumerations are
    materialised. The hard caps keep this module in diagnostic-prototype scope.
    """

    max_nodes: int = DEFAULT_MAX_NODES
    max_terminal_nodes: int = DEFAULT_MAX_TERMINAL_NODES
    max_opponent_info_sets_total: int = DEFAULT_MAX_OPPONENT_INFO_SETS_TOTAL
    max_info_sets_per_opponent: int = DEFAULT_MAX_INFO_SETS_PER_OPPONENT
    max_fixed_hero_info_sets: int = DEFAULT_MAX_FIXED_HERO_INFO_SETS
    max_actions_per_info_set: int = DEFAULT_MAX_ACTIONS_PER_INFO_SET
    max_chance_outcomes_per_node: int = DEFAULT_MAX_CHANCE_OUTCOMES_PER_NODE
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_oracle_pure_profiles: int = DEFAULT_MAX_ORACLE_PURE_PROFILES

    def __post_init__(self) -> None:
        _require_int_limit(self.max_nodes, "max_nodes", HARD_MAX_NODES)
        _require_int_limit(
            self.max_terminal_nodes,
            "max_terminal_nodes",
            HARD_MAX_TERMINAL_NODES,
        )
        _require_int_limit(
            self.max_opponent_info_sets_total,
            "max_opponent_info_sets_total",
            HARD_MAX_OPPONENT_INFO_SETS_TOTAL,
        )
        _require_int_limit(
            self.max_info_sets_per_opponent,
            "max_info_sets_per_opponent",
            HARD_MAX_INFO_SETS_PER_OPPONENT,
        )
        _require_int_limit(
            self.max_fixed_hero_info_sets,
            "max_fixed_hero_info_sets",
            HARD_MAX_FIXED_HERO_INFO_SETS,
        )
        _require_int_limit(
            self.max_actions_per_info_set,
            "max_actions_per_info_set",
            HARD_MAX_ACTIONS_PER_INFO_SET,
        )
        _require_int_limit(
            self.max_chance_outcomes_per_node,
            "max_chance_outcomes_per_node",
            HARD_MAX_CHANCE_OUTCOMES_PER_NODE,
        )
        _require_int_limit(
            self.max_iterations,
            "max_iterations",
            HARD_MAX_ITERATIONS,
        )
        _require_int_limit(
            self.max_oracle_pure_profiles,
            "max_oracle_pure_profiles",
            HARD_MAX_ORACLE_PURE_PROFILES,
        )


@dataclass(frozen=True)
class CfrConfig:
    """Configuration for the CFR-style regret diagnostic."""

    iterations: int = 1_000
    tolerance: float = 1e-9
    limits: CfrSafetyLimits = field(default_factory=CfrSafetyLimits)
    compute_deviation_gains: bool = True

    def __post_init__(self) -> None:
        _require_int_limit(self.iterations, "iterations", self.limits.max_iterations)
        _require_valid_tolerance(self.tolerance)
        if not isinstance(self.compute_deviation_gains, bool):
            raise ValueError(
                "compute_deviation_gains must be a bool, got "
                f"{self.compute_deviation_gains!r}"
            )


@dataclass(frozen=True)
class CfrDiagnosticResult:
    """Result of the small-tree CFR-style regret diagnostic.

    The fields are diagnostics only. They do not assert that an equilibrium was
    found, that a two-opponent exact response was solved, or that a strategy is
    profitable in any real-money setting.
    """

    iterations: int
    node_count: int
    terminal_count: int
    info_set_count_by_player: Dict[str, int]
    action_count_max: int
    average_strategy_by_player: Dict[str, Dict[str, Dict[str, float]]]
    current_strategy_by_player: Dict[str, Dict[str, Dict[str, float]]]
    expected_utility_vector: Dict[str, float]
    payoff_conservation_residual_max: float
    average_positive_regret_by_player: Dict[str, float]
    max_positive_regret_by_player: Dict[str, float]
    unilateral_deviation_gain_by_player: Dict[str, Optional[float]]
    oracle_profile_count: Optional[int]
    deviation_gain_unavailable_reason: Optional[str]
    deterministic_full_traversal: bool
    stopped_by_iteration_cap: bool
    stopped_by_safety_cap: bool

    def to_dict(self) -> dict:
        """Return a deterministic JSON-friendly diagnostic dictionary."""

        return {
            "iterations": self.iterations,
            "node_count": self.node_count,
            "terminal_count": self.terminal_count,
            "info_set_count_by_player": self.info_set_count_by_player,
            "action_count_max": self.action_count_max,
            "average_strategy_by_player": self.average_strategy_by_player,
            "current_strategy_by_player": self.current_strategy_by_player,
            "expected_utility_vector": self.expected_utility_vector,
            "payoff_conservation_residual_max": (
                self.payoff_conservation_residual_max
            ),
            "average_positive_regret_by_player": (
                self.average_positive_regret_by_player
            ),
            "max_positive_regret_by_player": self.max_positive_regret_by_player,
            "unilateral_deviation_gain_by_player": (
                self.unilateral_deviation_gain_by_player
            ),
            "oracle_profile_count": self.oracle_profile_count,
            "deviation_gain_unavailable_reason": (
                self.deviation_gain_unavailable_reason
            ),
            "deterministic_full_traversal": self.deterministic_full_traversal,
            "stopped_by_iteration_cap": self.stopped_by_iteration_cap,
            "stopped_by_safety_cap": self.stopped_by_safety_cap,
        }


@dataclass(frozen=True)
class _TreeMetadata:
    node_count: int
    terminal_count: int
    opponent_info_sets: Dict[OpponentId, Dict[InfoSetId, Tuple[Action, ...]]]
    fixed_hero_info_sets: Dict[InfoSetId, Tuple[Action, ...]]
    action_count_max: int
    payoff_conservation_residual_max: float


def iter_three_player_nodes(node: ThreePlayerNode) -> Iterator[ThreePlayerNode]:
    """Yield every node in pre-order."""

    yield node
    if isinstance(node, ThreePlayerChanceNode):
        for _, child in node.children:
            yield from iter_three_player_nodes(child)
    elif isinstance(node, FixedHeroNode):
        for _, child in node.actions:
            yield from iter_three_player_nodes(child)
    elif isinstance(node, OpponentDecisionNode):
        for _, child in node.actions:
            yield from iter_three_player_nodes(child)


def iter_three_player_terminals(
    node: ThreePlayerNode,
) -> Iterator[ThreePlayerTerminalNode]:
    """Yield every terminal node in the subtree."""

    for current in iter_three_player_nodes(node):
        if isinstance(current, ThreePlayerTerminalNode):
            yield current


def validate_three_player_tree(
    tree: ThreePlayerGameTree,
    fixed_hero_policy: BehaviorStrategy,
    *,
    tolerance: float = 1e-9,
    limits: Optional[CfrSafetyLimits] = None,
) -> None:
    """Validate a tiny 3-player tree and the fixed Hero policy.

    This checks structural consistency, payoff conservation, fixed-probability
    transitions, information-set action consistency, the repeated-info-set path
    guard, and all safety caps. Perfect recall remains an input contract; the
    path guard is necessary but not a full proof.
    """

    limits = _limits_or_default(limits)
    metadata = _collect_tree_metadata(tree, tolerance=tolerance, limits=limits)
    _validate_behavior_strategy(
        "fixed Hero",
        fixed_hero_policy,
        metadata.fixed_hero_info_sets,
        tolerance,
    )


def evaluate_three_player_profile(
    tree: ThreePlayerGameTree,
    fixed_hero_policy: BehaviorStrategy,
    opponent_strategies: Mapping[OpponentId, BehaviorStrategy],
    *,
    tolerance: float = 1e-9,
    limits: Optional[CfrSafetyLimits] = None,
) -> UtilityVector:
    """Evaluate a fixed Hero policy and fixed opponent behavior strategies.

    This is exact expected-value recursion over a small abstract tree. It does
    not optimize Hero or either opponent.
    """

    limits = _limits_or_default(limits)
    metadata = _collect_tree_metadata(tree, tolerance=tolerance, limits=limits)
    _validate_behavior_strategy(
        "fixed Hero",
        fixed_hero_policy,
        metadata.fixed_hero_info_sets,
        tolerance,
    )
    _validate_opponent_strategy_mapping(
        opponent_strategies,
        metadata.opponent_info_sets,
        tolerance,
    )
    return _expected_utility(tree.root, fixed_hero_policy, opponent_strategies)


def run_three_player_cfr_diagnostic(
    tree: ThreePlayerGameTree,
    fixed_hero_policy: BehaviorStrategy,
    *,
    config: Optional[CfrConfig] = None,
) -> CfrDiagnosticResult:
    """Run deterministic full-tree CFR-style opponent regret diagnostics.

    Regret is updated only for ``O1`` and ``O2``. Chance and fixed Hero nodes
    contribute reach probability. The returned average profile and utility
    vector are diagnostics for a tiny general-sum opponent subgame
    approximation, not an equilibrium certificate.
    """

    config = config or CfrConfig()
    metadata = _collect_tree_metadata(
        tree,
        tolerance=config.tolerance,
        limits=config.limits,
    )
    _validate_behavior_strategy(
        "fixed Hero",
        fixed_hero_policy,
        metadata.fixed_hero_info_sets,
        config.tolerance,
    )

    regrets = _empty_action_table(metadata.opponent_info_sets)
    strategy_sums = _empty_action_table(metadata.opponent_info_sets)

    current_strategy = _strategy_from_regrets(regrets, metadata.opponent_info_sets)
    for _ in range(config.iterations):
        current_strategy = _strategy_from_regrets(
            regrets,
            metadata.opponent_info_sets,
        )
        _cfr_traverse(
            tree.root,
            fixed_hero_policy,
            current_strategy,
            regrets,
            strategy_sums,
            chance_hero_reach=1.0,
            o1_reach=1.0,
            o2_reach=1.0,
        )

    average_strategy = _average_strategy(strategy_sums, metadata.opponent_info_sets)
    current_strategy = _strategy_from_regrets(regrets, metadata.opponent_info_sets)
    expected = _expected_utility(
        tree.root,
        fixed_hero_policy,
        {
            "O1": BehaviorStrategy(average_strategy["O1"]),
            "O2": BehaviorStrategy(average_strategy["O2"]),
        },
    )
    regret_stats = _positive_regret_stats(regrets)

    gain_by_player: Dict[str, Optional[float]] = {"O1": None, "O2": None}
    oracle_profile_count: Optional[int] = None
    unavailable_reason: Optional[str] = None
    stopped_by_safety_cap = False
    if config.compute_deviation_gains:
        deviation = compute_unilateral_deviation_gains(
            tree,
            fixed_hero_policy,
            {
                "O1": BehaviorStrategy(average_strategy["O1"]),
                "O2": BehaviorStrategy(average_strategy["O2"]),
            },
            limits=config.limits,
            tolerance=config.tolerance,
        )
        gain_by_player = deviation["gain_by_player"]
        oracle_profile_count = deviation["oracle_profile_count"]
        unavailable_reason = deviation["unavailable_reason"]
        stopped_by_safety_cap = unavailable_reason is not None

    return CfrDiagnosticResult(
        iterations=config.iterations,
        node_count=metadata.node_count,
        terminal_count=metadata.terminal_count,
        info_set_count_by_player={
            "O1": len(metadata.opponent_info_sets["O1"]),
            "O2": len(metadata.opponent_info_sets["O2"]),
        },
        action_count_max=metadata.action_count_max,
        average_strategy_by_player=average_strategy,
        current_strategy_by_player=current_strategy,
        expected_utility_vector=expected.to_dict(),
        payoff_conservation_residual_max=max(
            metadata.payoff_conservation_residual_max,
            abs(expected.conservation_residual()),
        ),
        average_positive_regret_by_player=regret_stats["average"],
        max_positive_regret_by_player=regret_stats["max"],
        unilateral_deviation_gain_by_player=gain_by_player,
        oracle_profile_count=oracle_profile_count,
        deviation_gain_unavailable_reason=unavailable_reason,
        deterministic_full_traversal=True,
        stopped_by_iteration_cap=config.iterations == config.limits.max_iterations,
        stopped_by_safety_cap=stopped_by_safety_cap,
    )


def count_opponent_pure_profiles(
    tree: ThreePlayerGameTree,
    *,
    limits: Optional[CfrSafetyLimits] = None,
) -> int:
    """Count joint opponent pure profiles without materialising them."""

    limits = _limits_or_default(limits)
    metadata = _collect_tree_metadata(tree, tolerance=1e-9, limits=limits)
    count = 1
    for player in OPPONENT_PLAYERS:
        for actions in metadata.opponent_info_sets[player].values():
            count = _checked_multiply(count, len(actions), "opponent pure profiles")
    return count


def compute_unilateral_deviation_gains(
    tree: ThreePlayerGameTree,
    fixed_hero_policy: BehaviorStrategy,
    average_opponent_strategies: Mapping[OpponentId, BehaviorStrategy],
    *,
    limits: Optional[CfrSafetyLimits] = None,
    tolerance: float = 1e-9,
) -> dict:
    """Compute unilateral deviation-gain diagnostics by tiny enumeration.

    Each opponent is optimized one at a time against the other opponent's fixed
    mixed strategy and the fixed Hero policy. This deliberately does not test
    joint or coalition deviations.
    """

    limits = _limits_or_default(limits)
    metadata = _collect_tree_metadata(tree, tolerance=tolerance, limits=limits)
    _validate_behavior_strategy(
        "fixed Hero",
        fixed_hero_policy,
        metadata.fixed_hero_info_sets,
        tolerance,
    )
    _validate_opponent_strategy_mapping(
        average_opponent_strategies,
        metadata.opponent_info_sets,
        tolerance,
    )
    oracle_profile_count = count_opponent_pure_profiles(tree, limits=limits)
    if oracle_profile_count > limits.max_oracle_pure_profiles:
        return {
            "gain_by_player": {"O1": None, "O2": None},
            "oracle_profile_count": oracle_profile_count,
            "unavailable_reason": (
                "opponent pure-profile space has "
                f"{oracle_profile_count} profiles, exceeding "
                f"max_oracle_pure_profiles={limits.max_oracle_pure_profiles}"
            ),
        }

    current_utility = _expected_utility(
        tree.root,
        fixed_hero_policy,
        average_opponent_strategies,
    )
    gains: Dict[str, Optional[float]] = {}
    for player in OPPONENT_PLAYERS:
        other: OpponentId = "O2" if player == "O1" else "O1"
        candidate_utility = current_utility.for_player(player)
        for pure in _iter_player_pure_strategies(
            metadata.opponent_info_sets[player]
        ):
            candidate_strategies = {
                player: _pure_strategy_to_behavior(
                    metadata.opponent_info_sets[player],
                    pure,
                ),
                other: average_opponent_strategies[other],
            }
            utility = _expected_utility(
                tree.root,
                fixed_hero_policy,
                candidate_strategies,
            ).for_player(player)
            if utility > candidate_utility:
                candidate_utility = utility
        gain = _checked_float_subtract(
            candidate_utility,
            current_utility.for_player(player),
            f"unilateral deviation gain for {player}",
        )
        gains[player] = 0.0 if abs(gain) <= tolerance else gain

    return {
        "gain_by_player": gains,
        "oracle_profile_count": oracle_profile_count,
        "unavailable_reason": None,
    }


def _limits_or_default(limits: Optional[CfrSafetyLimits]) -> CfrSafetyLimits:
    return limits if limits is not None else CfrSafetyLimits()


def _require_number(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _require_finite_derived(value: float, label: str) -> float:
    try:
        finite = math.isfinite(value)
    except (OverflowError, TypeError) as exc:
        raise ValueError(
            f"{label} could not be represented as a finite float"
        ) from exc
    if not finite:
        raise ValueError(f"{label} became non-finite")
    return value


def _checked_float_add(left: float, right: float, label: str) -> float:
    try:
        result = left + right
    except OverflowError as exc:
        raise ValueError(f"{label} overflowed") from exc
    return _require_finite_derived(result, label)


def _checked_float_subtract(left: float, right: float, label: str) -> float:
    try:
        result = left - right
    except OverflowError as exc:
        raise ValueError(f"{label} overflowed") from exc
    return _require_finite_derived(result, label)


def _checked_float_multiply(left: float, right: float, label: str) -> float:
    try:
        result = left * right
    except OverflowError as exc:
        raise ValueError(f"{label} overflowed") from exc
    return _require_finite_derived(result, label)


def _checked_float_divide(numerator: float, denominator: float, label: str) -> float:
    try:
        result = numerator / denominator
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError(f"{label} could not be computed") from exc
    return _require_finite_derived(result, label)


def _checked_float_sum(values: Iterable[float], label: str) -> float:
    total = 0.0
    for value in values:
        _require_finite_derived(value, label)
        total = _checked_float_add(total, value, label)
    return total


def _require_valid_tolerance(value: float, name: str = "tolerance") -> None:
    _require_number(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _require_probability(value: float, name: str) -> None:
    _require_number(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _require_int_limit(value: int, name: str, hard_cap: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value!r}")
    if value > hard_cap:
        raise ValueError(f"{name}={value} exceeds hard cap {hard_cap}")


def _checked_multiply(current: int, factor: int, label: str) -> int:
    result = current * factor
    if result < current:
        raise ValueError(f"{label} count overflowed")
    return result


def _owner_to_player(owner: Literal["opponent_1", "opponent_2"]) -> OpponentId:
    if owner == "opponent_1":
        return "O1"
    if owner == "opponent_2":
        return "O2"
    raise ValueError(f"unknown opponent owner {owner!r}")


def _zero() -> UtilityVector:
    return UtilityVector(0.0, 0.0, 0.0, 0.0)


def _add_scaled(left: UtilityVector, scale: float, right: UtilityVector) -> UtilityVector:
    def component(left_value: float, right_value: float, name: str) -> float:
        scaled = _checked_float_multiply(
            scale,
            right_value,
            f"utility accumulation for {name}",
        )
        return _checked_float_add(
            left_value,
            scaled,
            f"utility accumulation for {name}",
        )

    return UtilityVector(
        component(left.H, right.H, "H"),
        component(left.O1, right.O1, "O1"),
        component(left.O2, right.O2, "O2"),
        component(left.R, right.R, "R"),
    )


def _collect_tree_metadata(
    tree: ThreePlayerGameTree,
    *,
    tolerance: float,
    limits: CfrSafetyLimits,
) -> _TreeMetadata:
    _require_valid_tolerance(tolerance)
    if not isinstance(tree, ThreePlayerGameTree):
        raise ValueError(f"tree must be a ThreePlayerGameTree, got {tree!r}")

    node_count = 0
    terminal_count = 0
    action_count_max = 0
    residual_max = 0.0
    opponent_info_sets: Dict[OpponentId, Dict[InfoSetId, Tuple[Action, ...]]] = {
        "O1": {},
        "O2": {},
    }
    fixed_hero_info_sets: Dict[InfoSetId, Tuple[Action, ...]] = {}
    owner_by_info_set: Dict[InfoSetId, str] = {}

    def register_actions(
        owner_label: str,
        info_sets: Dict[InfoSetId, Tuple[Action, ...]],
        info_set: InfoSetId,
        actions: Tuple[Action, ...],
    ) -> None:
        if not info_set:
            raise ValueError("information set ids must be non-empty")
        previous_owner = owner_by_info_set.get(info_set)
        if previous_owner is not None and previous_owner != owner_label:
            raise ValueError(
                f"information set {info_set!r} appears under both "
                f"{previous_owner} and {owner_label}"
            )
        owner_by_info_set[info_set] = owner_label
        existing = info_sets.get(info_set)
        if existing is not None:
            if existing != actions:
                raise ValueError(
                    f"information set {info_set!r} has inconsistent legal "
                    f"actions: {existing} vs {actions}"
                )
            return
        if owner_label == "fixed_hero":
            if len(info_sets) + 1 > limits.max_fixed_hero_info_sets:
                raise ValueError("max_fixed_hero_info_sets safety limit exceeded")
        else:
            player = "O1" if owner_label == "opponent_1" else "O2"
            if len(info_sets) + 1 > limits.max_info_sets_per_opponent:
                raise ValueError(
                    f"max_info_sets_per_opponent safety limit exceeded for {player}"
                )
            opponent_total = sum(len(v) for v in opponent_info_sets.values()) + 1
            if opponent_total > limits.max_opponent_info_sets_total:
                raise ValueError(
                    "max_opponent_info_sets_total safety limit exceeded"
                )
        info_sets[info_set] = actions

    def validate_action_labels(
        node_id: NodeId,
        raw_actions: Tuple[Tuple[Action, ThreePlayerNode], ...],
    ) -> Tuple[Action, ...]:
        nonlocal action_count_max
        if not raw_actions:
            raise ValueError(f"decision node {node_id!r} has no actions")
        if len(raw_actions) > limits.max_actions_per_info_set:
            raise ValueError("max_actions_per_info_set safety limit exceeded")
        actions = tuple(action for action, _ in raw_actions)
        if any(not isinstance(action, str) or not action for action in actions):
            raise ValueError(f"decision node {node_id!r} has invalid action label")
        if len(set(actions)) != len(actions):
            raise ValueError(f"decision node {node_id!r} has duplicate actions")
        action_count_max = max(action_count_max, len(actions))
        return actions

    def walk(
        node: ThreePlayerNode,
        seen_o1: frozenset[InfoSetId],
        seen_o2: frozenset[InfoSetId],
    ) -> None:
        nonlocal node_count, terminal_count, residual_max
        node_count += 1
        if node_count > limits.max_nodes:
            raise ValueError("max_nodes safety limit exceeded")

        if isinstance(node, ThreePlayerTerminalNode):
            terminal_count += 1
            if terminal_count > limits.max_terminal_nodes:
                raise ValueError("max_terminal_nodes safety limit exceeded")
            for name, value in (
                ("H", node.utility.H),
                ("O1", node.utility.O1),
                ("O2", node.utility.O2),
                ("R", node.utility.R),
            ):
                _require_number(value, f"terminal {node.node_id!r} utility {name}")
            residual = abs(node.utility.conservation_residual())
            residual_max = max(residual_max, residual)
            if residual > tolerance:
                raise ValueError(
                    f"terminal {node.node_id!r} violates payoff conservation "
                    f"(residual {node.utility.conservation_residual()})"
                )
            return

        if isinstance(node, ThreePlayerChanceNode):
            if not node.children:
                raise ValueError(f"chance node {node.node_id!r} has no children")
            if len(node.children) > limits.max_chance_outcomes_per_node:
                raise ValueError("max_chance_outcomes_per_node safety limit exceeded")
            total = 0.0
            for prob, child in node.children:
                _require_probability(prob, f"chance node {node.node_id!r} probability")
                total = _checked_float_add(
                    total,
                    prob,
                    f"chance node {node.node_id!r} probability sum",
                )
                walk(child, seen_o1, seen_o2)
            if abs(total - 1.0) > tolerance:
                raise ValueError(
                    f"chance node {node.node_id!r} probabilities sum to {total}, "
                    "expected 1"
                )
            return

        if isinstance(node, FixedHeroNode):
            actions = validate_action_labels(node.node_id, node.actions)
            register_actions(
                "fixed_hero",
                fixed_hero_info_sets,
                node.info_set,
                actions,
            )
            for _, child in node.actions:
                walk(child, seen_o1, seen_o2)
            return

        if isinstance(node, OpponentDecisionNode):
            actions = validate_action_labels(node.node_id, node.actions)
            player = _owner_to_player(node.owner)
            if player == "O1":
                if node.info_set in seen_o1:
                    raise ValueError(
                        f"O1 information set {node.info_set!r} repeats on a "
                        "single root-to-terminal path"
                    )
                register_actions(
                    "opponent_1",
                    opponent_info_sets["O1"],
                    node.info_set,
                    actions,
                )
                next_o1 = seen_o1 | {node.info_set}
                for _, child in node.actions:
                    walk(child, next_o1, seen_o2)
                return
            if node.info_set in seen_o2:
                raise ValueError(
                    f"O2 information set {node.info_set!r} repeats on a "
                    "single root-to-terminal path"
                )
            register_actions(
                "opponent_2",
                opponent_info_sets["O2"],
                node.info_set,
                actions,
            )
            next_o2 = seen_o2 | {node.info_set}
            for _, child in node.actions:
                walk(child, seen_o1, next_o2)
            return

        raise TypeError(f"unknown node type: {type(node)!r}")

    walk(tree.root, frozenset(), frozenset())
    return _TreeMetadata(
        node_count=node_count,
        terminal_count=terminal_count,
        opponent_info_sets=opponent_info_sets,
        fixed_hero_info_sets=fixed_hero_info_sets,
        action_count_max=action_count_max,
        payoff_conservation_residual_max=residual_max,
    )


def _validate_behavior_strategy(
    label: str,
    strategy: BehaviorStrategy,
    info_sets: Mapping[InfoSetId, Tuple[Action, ...]],
    tolerance: float,
) -> None:
    _require_valid_tolerance(tolerance)
    if not isinstance(strategy, BehaviorStrategy):
        raise ValueError(f"{label} policy must be a BehaviorStrategy")
    unknown = set(strategy.probabilities) - set(info_sets)
    if unknown:
        raise ValueError(f"{label} policy has unknown information sets {sorted(unknown)}")
    for info_set, legal_actions in info_sets.items():
        if info_set not in strategy.probabilities:
            raise ValueError(f"{label} policy is missing information set {info_set!r}")
        distribution = strategy.probabilities[info_set]
        keys = set(distribution)
        if keys != set(legal_actions):
            raise ValueError(
                f"{label} policy for {info_set!r} must assign exactly legal "
                f"actions {list(legal_actions)}, got {sorted(keys)}"
            )
        total = 0.0
        for action in legal_actions:
            probability = distribution[action]
            _require_probability(probability, f"{label} probability {info_set}.{action}")
            total = _checked_float_add(
                total,
                probability,
                f"{label} policy for {info_set!r} probability sum",
            )
        if abs(total - 1.0) > tolerance:
            raise ValueError(
                f"{label} policy for {info_set!r} sums to {total}, expected 1"
            )


def _validate_opponent_strategy_mapping(
    strategies: Mapping[OpponentId, BehaviorStrategy],
    info_sets: Mapping[OpponentId, Mapping[InfoSetId, Tuple[Action, ...]]],
    tolerance: float,
) -> None:
    if set(strategies) != set(OPPONENT_PLAYERS):
        raise ValueError("opponent strategies must contain exactly O1 and O2")
    for player in OPPONENT_PLAYERS:
        _validate_behavior_strategy(
            player,
            strategies[player],
            info_sets[player],
            tolerance,
        )


def _expected_utility(
    node: ThreePlayerNode,
    fixed_hero_policy: BehaviorStrategy,
    opponent_strategies: Mapping[OpponentId, BehaviorStrategy],
) -> UtilityVector:
    if isinstance(node, ThreePlayerTerminalNode):
        return node.utility
    if isinstance(node, ThreePlayerChanceNode):
        value = _zero()
        for prob, child in node.children:
            if prob == 0.0:
                continue
            value = _add_scaled(
                value,
                prob,
                _expected_utility(child, fixed_hero_policy, opponent_strategies),
            )
        return value
    if isinstance(node, FixedHeroNode):
        value = _zero()
        for action, child in node.actions:
            prob = fixed_hero_policy.action_probability(node.info_set, action)
            if prob == 0.0:
                continue
            value = _add_scaled(
                value,
                prob,
                _expected_utility(child, fixed_hero_policy, opponent_strategies),
            )
        return value
    if isinstance(node, OpponentDecisionNode):
        player = _owner_to_player(node.owner)
        value = _zero()
        for action, child in node.actions:
            prob = opponent_strategies[player].action_probability(node.info_set, action)
            if prob == 0.0:
                continue
            value = _add_scaled(
                value,
                prob,
                _expected_utility(child, fixed_hero_policy, opponent_strategies),
            )
        return value
    raise TypeError(f"unknown node type: {type(node)!r}")


def _empty_action_table(
    info_sets: Mapping[OpponentId, Mapping[InfoSetId, Tuple[Action, ...]]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    return {
        player: {
            info_set: {action: 0.0 for action in info_sets[player][info_set]}
            for info_set in sorted(info_sets[player])
        }
        for player in OPPONENT_PLAYERS
    }


def _strategy_from_regrets(
    regrets: Mapping[str, Mapping[str, Mapping[str, float]]],
    info_sets: Mapping[OpponentId, Mapping[InfoSetId, Tuple[Action, ...]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    strategy: Dict[str, Dict[str, Dict[str, float]]] = {"O1": {}, "O2": {}}
    for player in OPPONENT_PLAYERS:
        for info_set in sorted(info_sets[player]):
            actions = info_sets[player][info_set]
            positives = []
            for action in actions:
                regret = regrets[player][info_set][action]
                _require_finite_derived(
                    regret,
                    f"regret matching value for {player}.{info_set}.{action}",
                )
                positives.append(max(regret, 0.0))
            total = _checked_float_sum(
                positives,
                f"regret matching total for {player}.{info_set}",
            )
            if total > 0.0:
                strategy[player][info_set] = {
                    action: _checked_float_divide(
                        positive,
                        total,
                        f"regret matching probability for {player}.{info_set}.{action}",
                    )
                    for action, positive in zip(actions, positives)
                }
            else:
                uniform = 1.0 / len(actions)
                strategy[player][info_set] = {action: uniform for action in actions}
    return strategy


def _average_strategy(
    strategy_sums: Mapping[str, Mapping[str, Mapping[str, float]]],
    info_sets: Mapping[OpponentId, Mapping[InfoSetId, Tuple[Action, ...]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    average: Dict[str, Dict[str, Dict[str, float]]] = {"O1": {}, "O2": {}}
    for player in OPPONENT_PLAYERS:
        for info_set in sorted(info_sets[player]):
            actions = info_sets[player][info_set]
            action_sums = []
            for action in actions:
                action_sum = strategy_sums[player][info_set][action]
                _require_finite_derived(
                    action_sum,
                    f"average strategy sum for {player}.{info_set}.{action}",
                )
                action_sums.append(action_sum)
            total = _checked_float_sum(
                action_sums,
                f"average strategy total for {player}.{info_set}",
            )
            if total > 0.0:
                average[player][info_set] = {
                    action: _checked_float_divide(
                        strategy_sums[player][info_set][action],
                        total,
                        f"average strategy probability for {player}.{info_set}.{action}",
                    )
                    for action in actions
                }
            else:
                uniform = 1.0 / len(actions)
                average[player][info_set] = {action: uniform for action in actions}
    return average


def _positive_regret_stats(
    regrets: Mapping[str, Mapping[str, Mapping[str, float]]]
) -> Dict[str, Dict[str, float]]:
    average: Dict[str, float] = {}
    maximum: Dict[str, float] = {}
    for player in OPPONENT_PLAYERS:
        values = []
        for info_set_id, info_set in regrets[player].items():
            for action, value in info_set.items():
                _require_finite_derived(
                    value,
                    f"positive regret statistic for {player}.{info_set_id}.{action}",
                )
                values.append(max(value, 0.0))
        if values:
            total = _checked_float_sum(
                values,
                f"positive regret statistic total for {player}",
            )
            average[player] = _checked_float_divide(
                total,
                len(values),
                f"average positive regret for {player}",
            )
        else:
            average[player] = 0.0
        maximum[player] = max(values) if values else 0.0
    return {"average": average, "max": maximum}


def _cfr_traverse(
    node: ThreePlayerNode,
    fixed_hero_policy: BehaviorStrategy,
    current_strategy: Mapping[str, Mapping[str, Mapping[str, float]]],
    regrets: Dict[str, Dict[str, Dict[str, float]]],
    strategy_sums: Dict[str, Dict[str, Dict[str, float]]],
    *,
    chance_hero_reach: float,
    o1_reach: float,
    o2_reach: float,
) -> UtilityVector:
    if isinstance(node, ThreePlayerTerminalNode):
        return node.utility

    if isinstance(node, ThreePlayerChanceNode):
        value = _zero()
        for prob, child in node.children:
            if prob == 0.0:
                continue
            child_value = _cfr_traverse(
                child,
                fixed_hero_policy,
                current_strategy,
                regrets,
                strategy_sums,
                chance_hero_reach=_checked_float_multiply(
                    chance_hero_reach,
                    prob,
                    "chance reach",
                ),
                o1_reach=o1_reach,
                o2_reach=o2_reach,
            )
            value = _add_scaled(value, prob, child_value)
        return value

    if isinstance(node, FixedHeroNode):
        value = _zero()
        for action, child in node.actions:
            prob = fixed_hero_policy.action_probability(node.info_set, action)
            if prob == 0.0:
                continue
            child_value = _cfr_traverse(
                child,
                fixed_hero_policy,
                current_strategy,
                regrets,
                strategy_sums,
                chance_hero_reach=_checked_float_multiply(
                    chance_hero_reach,
                    prob,
                    "fixed Hero reach",
                ),
                o1_reach=o1_reach,
                o2_reach=o2_reach,
            )
            value = _add_scaled(value, prob, child_value)
        return value

    if isinstance(node, OpponentDecisionNode):
        player = _owner_to_player(node.owner)
        strategy = current_strategy[player][node.info_set]
        action_values: Dict[Action, UtilityVector] = {}
        node_value = _zero()
        for action, child in node.actions:
            prob = strategy[action]
            next_o1 = (
                _checked_float_multiply(o1_reach, prob, "O1 reach")
                if player == "O1"
                else o1_reach
            )
            next_o2 = (
                _checked_float_multiply(o2_reach, prob, "O2 reach")
                if player == "O2"
                else o2_reach
            )
            child_value = _cfr_traverse(
                child,
                fixed_hero_policy,
                current_strategy,
                regrets,
                strategy_sums,
                chance_hero_reach=chance_hero_reach,
                o1_reach=next_o1,
                o2_reach=next_o2,
            )
            action_values[action] = child_value
            node_value = _add_scaled(node_value, prob, child_value)

        counterfactual_reach = _checked_float_multiply(
            chance_hero_reach,
            o2_reach if player == "O1" else o1_reach,
            f"counterfactual reach for {player}",
        )
        owner_reach = o1_reach if player == "O1" else o2_reach
        average_weight = _checked_float_multiply(
            chance_hero_reach,
            owner_reach,
            f"average strategy weight for {player}",
        )
        for action, _ in node.actions:
            action_advantage = _checked_float_subtract(
                action_values[action].for_player(player),
                node_value.for_player(player),
                f"regret advantage for {player}.{node.info_set}.{action}",
            )
            regret_delta = _checked_float_multiply(
                counterfactual_reach,
                action_advantage,
                f"regret update for {player}.{node.info_set}.{action}",
            )
            regrets[player][node.info_set][action] = _checked_float_add(
                regrets[player][node.info_set][action],
                regret_delta,
                f"cumulative regret for {player}.{node.info_set}.{action}",
            )
            strategy_delta = _checked_float_multiply(
                average_weight,
                strategy[action],
                f"strategy averaging update for {player}.{node.info_set}.{action}",
            )
            strategy_sums[player][node.info_set][action] = _checked_float_add(
                strategy_sums[player][node.info_set][action],
                strategy_delta,
                f"cumulative strategy sum for {player}.{node.info_set}.{action}",
            )
        return node_value

    raise TypeError(f"unknown node type: {type(node)!r}")


def _iter_player_pure_strategies(
    info_sets: Mapping[InfoSetId, Tuple[Action, ...]]
) -> Iterator[Dict[InfoSetId, Action]]:
    info_set_ids = sorted(info_sets)
    if not info_set_ids:
        yield {}
        return
    action_lists = [info_sets[info_set] for info_set in info_set_ids]
    for combo in itertools.product(*action_lists):
        yield dict(zip(info_set_ids, combo))


def _pure_strategy_to_behavior(
    info_sets: Mapping[InfoSetId, Tuple[Action, ...]],
    pure: Mapping[InfoSetId, Action],
) -> BehaviorStrategy:
    return BehaviorStrategy(
        {
            info_set: {
                action: 1.0 if pure[info_set] == action else 0.0
                for action in actions
            }
            for info_set, actions in info_sets.items()
        }
    )
