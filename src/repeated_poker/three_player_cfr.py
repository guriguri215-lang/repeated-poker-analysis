"""Tiny three-player fixed-Hero diagnostic prototype.

This module is intentionally isolated from the two-player exact-response core.
It models a small abstract tree with a fixed Hero policy and two strategic
opponents, then runs deterministic full-tree CFR-style regret diagnostics for
the two opponents only.

The output is a deterministic finite-iteration CFR-style diagnostic for small
abstract trees.  Its claims are limited to the quantities explicitly reported.
"""

from __future__ import annotations

import itertools
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Literal, Mapping, Optional, Tuple, Union

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
DEFAULT_MAX_ORACLE_PURE_PLANS_PER_PLAYER = 4_096
DEFAULT_MAX_ORACLE_JOINT_PROFILES = 4_096
DEFAULT_MAX_ORACLE_PROFILE_EVALUATIONS = 100_000
DEFAULT_MAX_ORACLE_OUTPUT_ROWS = 4_096
DEFAULT_MAX_TRACE_POINTS = 100

HARD_MAX_NODES = 500
HARD_MAX_TERMINAL_NODES = 256
HARD_MAX_OPPONENT_INFO_SETS_TOTAL = 32
HARD_MAX_INFO_SETS_PER_OPPONENT = 16
HARD_MAX_FIXED_HERO_INFO_SETS = 16
HARD_MAX_ACTIONS_PER_INFO_SET = 4
HARD_MAX_CHANCE_OUTCOMES_PER_NODE = 32
HARD_MAX_ITERATIONS = 50_000
HARD_MAX_ORACLE_PURE_PROFILES = 16_384
HARD_MAX_ORACLE_PURE_PLANS_PER_PLAYER = 16_384
HARD_MAX_ORACLE_JOINT_PROFILES = 16_384
HARD_MAX_ORACLE_PROFILE_EVALUATIONS = 500_000
HARD_MAX_ORACLE_OUTPUT_ROWS = 16_384
HARD_MAX_TRACE_POINTS = 1_000

CONTRACT_VERSION = "m12-cfr-primary-oracle-attachment-v1"
ALGORITHM_VERSION = "simultaneous-full-tree-regret-matching-v1"

DIAGNOSTIC_COMPLETE = "DIAGNOSTIC_COMPLETE"
NOT_REQUESTED = "NOT_REQUESTED"
INVALID_INPUT = "INVALID_INPUT"
UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"
CAP_EXCEEDED = "CAP_EXCEEDED"
ORACLE_UNAVAILABLE_CAP = "ORACLE_UNAVAILABLE_CAP"
NUMERIC_FAILURE = "NUMERIC_FAILURE"
ORACLE_MISMATCH = "ORACLE_MISMATCH"
INDETERMINATE_TOLERANCE = "INDETERMINATE_TOLERANCE"
NON_REPRODUCIBLE = "NON_REPRODUCIBLE"


class DiagnosticContractError(ValueError):
    """Fail-closed contract error carrying a stable diagnostic status."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


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
class PerfectRecallAttestation:
    """Human-traceable confirmation bound to canonical tree content."""

    tree_content_identity: str
    o1_confirmed: bool
    o2_confirmed: bool
    verifier: str
    verification_date: str
    evidence_version: str

    def __post_init__(self) -> None:
        for name in (
            "tree_content_identity",
            "verifier",
            "verification_date",
            "evidence_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.o1_confirmed, bool) or not isinstance(
            self.o2_confirmed, bool
        ):
            raise ValueError("attestation confirmations must be bool values")


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
    max_oracle_pure_plans_per_player: int = (
        DEFAULT_MAX_ORACLE_PURE_PLANS_PER_PLAYER
    )
    max_oracle_joint_profiles: int = DEFAULT_MAX_ORACLE_JOINT_PROFILES
    max_oracle_profile_evaluations: int = (
        DEFAULT_MAX_ORACLE_PROFILE_EVALUATIONS
    )
    max_oracle_output_rows: int = DEFAULT_MAX_ORACLE_OUTPUT_ROWS
    max_trace_points: int = DEFAULT_MAX_TRACE_POINTS

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
        _require_int_limit(
            self.max_oracle_pure_plans_per_player,
            "max_oracle_pure_plans_per_player",
            HARD_MAX_ORACLE_PURE_PLANS_PER_PLAYER,
        )
        _require_int_limit(
            self.max_oracle_joint_profiles,
            "max_oracle_joint_profiles",
            HARD_MAX_ORACLE_JOINT_PROFILES,
        )
        _require_int_limit(
            self.max_oracle_profile_evaluations,
            "max_oracle_profile_evaluations",
            HARD_MAX_ORACLE_PROFILE_EVALUATIONS,
        )
        _require_int_limit(
            self.max_oracle_output_rows,
            "max_oracle_output_rows",
            HARD_MAX_ORACLE_OUTPUT_ROWS,
        )
        _require_int_limit(
            self.max_trace_points,
            "max_trace_points",
            HARD_MAX_TRACE_POINTS,
        )


@dataclass(frozen=True)
class CfrConfig:
    """Configuration for the CFR-style regret diagnostic."""

    iterations: int = 1_000
    input_tolerance: float = 1e-9
    epsilon_deviation: float = 1e-9
    oracle_compare_tolerance: float = 1e-9
    reproducibility_tolerance: float = 1e-12
    limits: CfrSafetyLimits = field(default_factory=CfrSafetyLimits)
    compute_deviation_gains: bool = True
    request_oracle: bool = False
    include_oracle_rows: bool = False
    trace_checkpoint_interval: Optional[int] = None
    seed: None = None

    @property
    def tolerance(self) -> float:
        """Compatibility view of the input-validation tolerance."""

        return self.input_tolerance

    def __post_init__(self) -> None:
        try:
            _require_int_limit(
                self.iterations, "iterations", self.limits.max_iterations
            )
        except ValueError as exc:
            if "exceeds hard cap" in str(exc):
                raise DiagnosticContractError(CAP_EXCEEDED, str(exc)) from exc
            raise
        for name in (
            "input_tolerance",
            "epsilon_deviation",
            "oracle_compare_tolerance",
            "reproducibility_tolerance",
        ):
            _require_valid_tolerance(getattr(self, name), name)
        for name in (
            "compute_deviation_gains",
            "request_oracle",
            "include_oracle_rows",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool, got {getattr(self, name)!r}")
        if self.include_oracle_rows and not self.request_oracle:
            raise ValueError("include_oracle_rows requires request_oracle")
        if self.trace_checkpoint_interval is not None:
            if (
                isinstance(self.trace_checkpoint_interval, bool)
                or not isinstance(self.trace_checkpoint_interval, int)
                or self.trace_checkpoint_interval < 1
            ):
                raise ValueError("trace_checkpoint_interval must be a positive int")
        if self.seed is not None:
            raise DiagnosticContractError(
                UNSUPPORTED_MODEL,
                "seed must be None for deterministic full traversal",
            )


@dataclass(frozen=True)
class CfrDiagnosticResult:
    """Result of the deterministic finite-iteration CFR-style diagnostic."""

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
    contract_version: str = CONTRACT_VERSION
    algorithm_version: str = ALGORITHM_VERSION
    component_status: str = DIAGNOSTIC_COMPLETE
    overall_status: str = DIAGNOSTIC_COMPLETE
    requested_iterations: int = 0
    completed_iterations: int = 0
    content_identity: Dict[str, str] = field(default_factory=dict)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    perfect_recall_attestation: Dict[str, Any] = field(default_factory=dict)
    normalization_records: Tuple[dict, ...] = ()
    tolerances: Dict[str, float] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()
    positive_regret_semantics: str = (
        "cumulative positive action-regret table summary; not iteration-normalized"
    )
    trace: Dict[str, Any] = field(default_factory=dict)
    oracle_attachment: Dict[str, Any] = field(default_factory=dict)

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
            "contract_version": self.contract_version,
            "algorithm_version": self.algorithm_version,
            "component_status": self.component_status,
            "overall_status": self.overall_status,
            "requested_iterations": self.requested_iterations,
            "completed_iterations": self.completed_iterations,
            "content_identity": self.content_identity,
            "execution_metadata": self.execution_metadata,
            "perfect_recall_attestation": self.perfect_recall_attestation,
            "normalization_records": list(self.normalization_records),
            "tolerances": self.tolerances,
            "warnings": list(self.warnings),
            "positive_regret_semantics": self.positive_regret_semantics,
            "trace": self.trace,
            "oracle_attachment": self.oracle_attachment,
        }


@dataclass(frozen=True)
class _TreeMetadata:
    node_count: int
    terminal_count: int
    opponent_info_sets: Dict[OpponentId, Dict[InfoSetId, Tuple[Action, ...]]]
    fixed_hero_info_sets: Dict[InfoSetId, Tuple[Action, ...]]
    action_count_max: int
    payoff_conservation_residual_max: float


def tree_content_identity(tree: ThreePlayerGameTree) -> str:
    """Return the SHA-256 identity of ordered canonical tree content."""

    seen: set[str] = set()

    def encode(node: ThreePlayerNode) -> dict:
        if not isinstance(node.node_id, str) or not node.node_id:
            raise ValueError("node ids must be non-empty strings")
        if node.node_id in seen:
            raise ValueError(f"duplicate node id {node.node_id!r}")
        seen.add(node.node_id)
        if isinstance(node, ThreePlayerTerminalNode):
            return {
                "type": "terminal",
                "node_id": node.node_id,
                "owner": "terminal",
                "utility": [
                    node.utility.H,
                    node.utility.O1,
                    node.utility.O2,
                    node.utility.R,
                ],
            }
        if isinstance(node, ThreePlayerChanceNode):
            return {
                "type": "chance",
                "node_id": node.node_id,
                "owner": "chance",
                "children": [
                    {"raw_probability": probability, "child": encode(child)}
                    for probability, child in node.children
                ],
            }
        if isinstance(node, FixedHeroNode):
            return {
                "type": "decision",
                "node_id": node.node_id,
                "owner": "fixed_hero",
                "information_set": node.info_set,
                "actions": [
                    {"action": action, "child": encode(child)}
                    for action, child in node.actions
                ],
            }
        if isinstance(node, OpponentDecisionNode):
            return {
                "type": "decision",
                "node_id": node.node_id,
                "owner": node.owner,
                "information_set": node.info_set,
                "actions": [
                    {"action": action, "child": encode(child)}
                    for action, child in node.actions
                ],
            }
        raise TypeError(f"unknown node type: {type(node)!r}")

    if not isinstance(tree, ThreePlayerGameTree):
        raise ValueError("tree must be a ThreePlayerGameTree")
    payload = {"description": tree.description, "root": encode(tree.root)}
    return _content_hash(payload)


def create_perfect_recall_attestation(
    tree: ThreePlayerGameTree,
    *,
    verifier: str,
    verification_date: str,
    evidence_version: str,
    o1_confirmed: bool = True,
    o2_confirmed: bool = True,
) -> PerfectRecallAttestation:
    """Create a traceable attestation bound to the current tree content."""

    return PerfectRecallAttestation(
        tree_content_identity=tree_content_identity(tree),
        o1_confirmed=o1_confirmed,
        o2_confirmed=o2_confirmed,
        verifier=verifier,
        verification_date=verification_date,
        evidence_version=evidence_version,
    )


def _content_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_attestation(
    tree: ThreePlayerGameTree,
    attestation: Optional[PerfectRecallAttestation],
) -> str:
    identity = tree_content_identity(tree)
    if attestation is None:
        raise DiagnosticContractError(
            UNSUPPORTED_MODEL, "perfect-recall attestation is required"
        )
    if not isinstance(attestation, PerfectRecallAttestation):
        raise DiagnosticContractError(
            UNSUPPORTED_MODEL, "perfect-recall attestation has an unsupported type"
        )
    if not attestation.o1_confirmed or not attestation.o2_confirmed:
        raise DiagnosticContractError(
            UNSUPPORTED_MODEL, "perfect-recall attestation is not confirmed for O1/O2"
        )
    if attestation.tree_content_identity != identity:
        raise DiagnosticContractError(
            UNSUPPORTED_MODEL, "perfect-recall attestation tree identity mismatch"
        )
    return identity


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
    guard, and all safety caps. Perfect recall remains a separately attested
    input contract; the path guard is only a structural check.
    """

    limits = _limits_or_default(limits)
    _effective_inputs(
        tree,
        fixed_hero_policy,
        None,
        input_tolerance=tolerance,
        limits=limits,
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
    effective_tree, effective_hero, effective_opponents, _, _ = _effective_inputs(
        tree,
        fixed_hero_policy,
        opponent_strategies,
        input_tolerance=tolerance,
        limits=limits,
    )
    assert effective_opponents is not None
    return _expected_utility(
        effective_tree.root, effective_hero, effective_opponents
    )


def run_three_player_cfr_diagnostic(
    tree: ThreePlayerGameTree,
    fixed_hero_policy: BehaviorStrategy,
    *,
    config: Optional[CfrConfig] = None,
    attestation: Optional[PerfectRecallAttestation] = None,
) -> CfrDiagnosticResult:
    """Run the fail-closed content-bound diagnostic contract."""

    try:
        return _run_three_player_cfr_diagnostic(
            tree,
            fixed_hero_policy,
            config=config,
            attestation=attestation,
        )
    except DiagnosticContractError:
        raise
    except ValueError as exc:
        message = str(exc)
        numeric_markers = (
            "non-finite",
            "overflow",
            "finite float",
            "normalization failed",
        )
        if "safety limit exceeded" in message:
            status = CAP_EXCEEDED
        elif any(marker in message for marker in numeric_markers):
            status = NUMERIC_FAILURE
        else:
            status = INVALID_INPUT
        raise DiagnosticContractError(status, message) from exc


def _run_three_player_cfr_diagnostic(
    tree: ThreePlayerGameTree,
    fixed_hero_policy: BehaviorStrategy,
    *,
    config: Optional[CfrConfig] = None,
    attestation: Optional[PerfectRecallAttestation] = None,
) -> CfrDiagnosticResult:
    """Run the content-bound deterministic finite-iteration diagnostic."""

    config = config or CfrConfig()
    raw_tree_identity = _require_attestation(tree, attestation)
    (
        effective_tree,
        effective_hero,
        _,
        normalization_records,
        metadata,
    ) = _effective_inputs(
        tree,
        fixed_hero_policy,
        None,
        input_tolerance=config.input_tolerance,
        limits=config.limits,
    )

    trace_iterations = _trace_schedule(config)

    regrets = _empty_action_table(metadata.opponent_info_sets)
    strategy_sums = _empty_action_table(metadata.opponent_info_sets)
    trace_points: list[dict] = []

    current_strategy = _strategy_from_regrets(regrets, metadata.opponent_info_sets)
    for iteration in range(1, config.iterations + 1):
        current_strategy = _strategy_from_regrets(
            regrets,
            metadata.opponent_info_sets,
        )
        _cfr_traverse(
            effective_tree.root,
            effective_hero,
            current_strategy,
            regrets,
            strategy_sums,
            chance_hero_reach=1.0,
            o1_reach=1.0,
            o2_reach=1.0,
        )
        if iteration in trace_iterations:
            checkpoint_average = _average_strategy(
                strategy_sums, metadata.opponent_info_sets
            )
            checkpoint_utility = _expected_utility(
                effective_tree.root,
                effective_hero,
                {
                    "O1": BehaviorStrategy(checkpoint_average["O1"]),
                    "O2": BehaviorStrategy(checkpoint_average["O2"]),
                },
            )
            trace_points.append(
                {
                    "iteration": iteration,
                    "expected_utility": checkpoint_utility.to_dict(),
                    "cumulative_positive_regret_summary": _positive_regret_stats(
                        regrets
                    ),
                }
            )

    average_strategy = _average_strategy(strategy_sums, metadata.opponent_info_sets)
    current_strategy = _strategy_from_regrets(regrets, metadata.opponent_info_sets)
    expected = _expected_utility(
        effective_tree.root,
        effective_hero,
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
            input_tolerance=config.input_tolerance,
            attestation=attestation,
        )
        gain_by_player = deviation["gain_by_player"]
        oracle_profile_count = deviation["oracle_profile_count"]
        unavailable_reason = deviation["unavailable_reason"]
        stopped_by_safety_cap = unavailable_reason is not None

    oracle_attachment = _build_oracle_attachment(
        raw_tree=tree,
        effective_tree=effective_tree,
        effective_hero=effective_hero,
        average_strategy=average_strategy,
        metadata=metadata,
        config=config,
        requested=config.request_oracle,
    )
    oracle_status = oracle_attachment["status"]
    overall_status = (
        DIAGNOSTIC_COMPLETE
        if oracle_status in (NOT_REQUESTED, "MATCH")
        else oracle_status
    )
    warnings = list(oracle_attachment.get("warnings", []))
    if config.iterations == config.limits.max_iterations:
        warnings.append("iteration_cap_reached")

    tolerances = {
        "input_tolerance": config.input_tolerance,
        "epsilon_deviation": config.epsilon_deviation,
        "oracle_compare_tolerance": config.oracle_compare_tolerance,
        "reproducibility_tolerance": config.reproducibility_tolerance,
    }
    limits_dict = {
        name: getattr(config.limits, name)
        for name in config.limits.__dataclass_fields__
    }
    hero_identity = _content_hash(
        {
            "probabilities": effective_hero.probabilities,
            "normalization": list(normalization_records),
        }
    )
    config_identity = _content_hash(
        {
            "contract_version": CONTRACT_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "iterations": config.iterations,
            "tolerances": tolerances,
            "limits": limits_dict,
            "compute_deviation_gains": config.compute_deviation_gains,
            "request_oracle": config.request_oracle,
            "include_oracle_rows": config.include_oracle_rows,
            "trace_schedule": sorted(trace_iterations),
            "seed": None,
        }
    )

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
        requested_iterations=config.iterations,
        completed_iterations=config.iterations,
        overall_status=overall_status,
        content_identity={
            "tree": raw_tree_identity,
            "fixed_hero": hero_identity,
            "config": config_identity,
        },
        execution_metadata=_execution_metadata(),
        perfect_recall_attestation={
            "tree_content_identity": attestation.tree_content_identity,
            "o1_confirmed": attestation.o1_confirmed,
            "o2_confirmed": attestation.o2_confirmed,
            "verifier": attestation.verifier,
            "verification_date": attestation.verification_date,
            "evidence_version": attestation.evidence_version,
        },
        normalization_records=normalization_records,
        tolerances=tolerances,
        warnings=tuple(warnings),
        trace={
            "enabled": config.trace_checkpoint_interval is not None,
            "coverage": "checkpoints" if trace_points else "none",
            "schedule": sorted(trace_iterations),
            "points": trace_points,
        },
        oracle_attachment=oracle_attachment,
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
    input_tolerance: float = 1e-9,
    attestation: Optional[PerfectRecallAttestation] = None,
) -> dict:
    """Compute content-bound unilateral pure-enumeration diagnostics."""

    try:
        return _compute_unilateral_deviation_gains(
            tree,
            fixed_hero_policy,
            average_opponent_strategies,
            limits=limits,
            input_tolerance=input_tolerance,
            attestation=attestation,
        )
    except DiagnosticContractError:
        raise
    except ValueError as exc:
        message = str(exc)
        status = (
            NUMERIC_FAILURE
            if any(word in message for word in ("non-finite", "overflow", "finite float"))
            else INVALID_INPUT
        )
        raise DiagnosticContractError(status, message) from exc


def _compute_unilateral_deviation_gains(
    tree: ThreePlayerGameTree,
    fixed_hero_policy: BehaviorStrategy,
    average_opponent_strategies: Mapping[OpponentId, BehaviorStrategy],
    *,
    limits: Optional[CfrSafetyLimits] = None,
    input_tolerance: float = 1e-9,
    attestation: Optional[PerfectRecallAttestation] = None,
) -> dict:
    """Compute unilateral deviation-gain diagnostics by tiny enumeration.

    Each opponent is optimized one at a time against the other opponent's fixed
    mixed strategy and the fixed Hero policy. This deliberately does not test
    joint or coalition deviations.
    """

    _require_attestation(tree, attestation)
    limits = _limits_or_default(limits)
    (
        effective_tree,
        effective_hero,
        effective_opponents,
        normalization_records,
        metadata,
    ) = _effective_inputs(
        tree,
        fixed_hero_policy,
        average_opponent_strategies,
        input_tolerance=input_tolerance,
        limits=limits,
    )
    assert effective_opponents is not None
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
        effective_tree.root,
        effective_hero,
        effective_opponents,
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
                other: effective_opponents[other],
            }
            utility = _expected_utility(
                effective_tree.root,
                effective_hero,
                candidate_strategies,
            ).for_player(player)
            if utility > candidate_utility:
                candidate_utility = utility
        gain = _checked_float_subtract(
            candidate_utility,
            current_utility.for_player(player),
            f"unilateral deviation gain for {player}",
        )
        gains[player] = gain

    return {
        "gain_by_player": gains,
        "oracle_profile_count": oracle_profile_count,
        "unavailable_reason": None,
        "normalization_records": list(normalization_records),
    }


def _trace_schedule(config: CfrConfig) -> frozenset[int]:
    interval = config.trace_checkpoint_interval
    if interval is None:
        return frozenset()
    point_count = config.iterations // interval
    if config.iterations % interval:
        point_count += 1
    if point_count > config.limits.max_trace_points:
        raise DiagnosticContractError(
            CAP_EXCEEDED,
            f"trace point count {point_count} exceeds max_trace_points="
            f"{config.limits.max_trace_points}",
        )
    schedule = list(range(interval, config.iterations + 1, interval))
    if not schedule or schedule[-1] != config.iterations:
        schedule.append(config.iterations)
    return frozenset(schedule)


def _execution_metadata() -> Dict[str, Any]:
    try:
        package_version = importlib.metadata.version("repeated-poker-analysis")
    except importlib.metadata.PackageNotFoundError:
        package_version = "unknown"
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": sys.version,
        "package_version": package_version,
        "git_commit_best_effort": commit,
        "platform": platform.platform(),
        "float_metadata": {
            "radix": sys.float_info.radix,
            "mantissa_digits": sys.float_info.mant_dig,
            "max": sys.float_info.max,
            "epsilon": sys.float_info.epsilon,
        },
        "initialization": "zero cumulative regret; uniform initial strategy",
        "traversal": "deterministic full tree",
        "ordering": "players O1/O2; information-set id sort; tree action order",
        "averaging": "cumulative reach-weighted behavioural strategy sums",
        "seed": None,
    }


def _plan_count(
    info_sets: Mapping[InfoSetId, Tuple[Action, ...]], player: str
) -> int:
    count = 1
    for info_set in sorted(info_sets):
        count = _checked_multiply(
            count, len(info_sets[info_set]), f"{player} pure plans"
        )
    return count


def _pure_plan_probability(
    plan: Mapping[InfoSetId, Action], strategy: Mapping[str, Mapping[str, float]]
) -> float:
    probability = 1.0
    for info_set in sorted(plan):
        probability = _checked_float_multiply(
            probability,
            strategy[info_set][plan[info_set]],
            f"pure plan mixture weight for {info_set}",
        )
    return probability


def _oracle_direct_evaluate(
    tree: ThreePlayerGameTree,
    hero: BehaviorStrategy,
    profiles: Mapping[OpponentId, BehaviorStrategy],
) -> UtilityVector:
    return _expected_utility(tree.root, hero, profiles)


def _build_oracle_attachment(
    *,
    raw_tree: ThreePlayerGameTree,
    effective_tree: ThreePlayerGameTree,
    effective_hero: BehaviorStrategy,
    average_strategy: Dict[str, Dict[str, Dict[str, float]]],
    metadata: _TreeMetadata,
    config: CfrConfig,
    requested: bool,
) -> Dict[str, Any]:
    if not requested:
        return {
            "status": NOT_REQUESTED,
            "coverage": "none",
            "rows": [],
        }

    n1 = _plan_count(metadata.opponent_info_sets["O1"], "O1")
    n2 = _plan_count(metadata.opponent_info_sets["O2"], "O2")
    n_joint = _checked_multiply(n1, n2, "joint pure profiles")
    predicted_evaluations = 2 * n_joint + n1 + n2 + 1
    predicted_rows = n_joint if config.include_oracle_rows else 0
    counts = {
        "pure_plans_by_player": {"O1": n1, "O2": n2},
        "joint_profiles": n_joint,
        "predicted_profile_evaluations": predicted_evaluations,
        "predicted_output_rows": predicted_rows,
    }
    failures = []
    if max(n1, n2) > config.limits.max_oracle_pure_plans_per_player:
        failures.append("max_oracle_pure_plans_per_player")
    if n_joint > config.limits.max_oracle_joint_profiles:
        failures.append("max_oracle_joint_profiles")
    if predicted_evaluations > config.limits.max_oracle_profile_evaluations:
        failures.append("max_oracle_profile_evaluations")
    if predicted_rows > config.limits.max_oracle_output_rows:
        failures.append("max_oracle_output_rows")
    if failures:
        return {
            "status": ORACLE_UNAVAILABLE_CAP,
            "coverage": "none",
            "counts": counts,
            "cap_failures": failures,
            "rows": [],
        }

    plans1 = list(_iter_player_pure_strategies(metadata.opponent_info_sets["O1"]))
    plans2 = list(_iter_player_pure_strategies(metadata.opponent_info_sets["O2"]))
    table: list[list[UtilityVector]] = []
    max_pure_delta = 0.0
    evaluation_count = 0
    for plan1 in plans1:
        table_row = []
        for plan2 in plans2:
            profiles = {
                "O1": _pure_strategy_to_behavior(
                    metadata.opponent_info_sets["O1"], plan1
                ),
                "O2": _pure_strategy_to_behavior(
                    metadata.opponent_info_sets["O2"], plan2
                ),
            }
            utility = _expected_utility(
                effective_tree.root, effective_hero, profiles
            )
            direct = _oracle_direct_evaluate(
                effective_tree, effective_hero, profiles
            )
            evaluation_count += 2
            for player in ("H", "O1", "O2", "R"):
                delta = abs(getattr(utility, player) - getattr(direct, player))
                max_pure_delta = max(max_pure_delta, delta)
            table_row.append(utility)
        table.append(table_row)

    weights1 = [_pure_plan_probability(plan, average_strategy["O1"]) for plan in plans1]
    weights2 = [_pure_plan_probability(plan, average_strategy["O2"]) for plan in plans2]
    mixture_utility = _zero()
    for i, row in enumerate(table):
        for j, utility in enumerate(row):
            weight = _checked_float_multiply(
                weights1[i], weights2[j], "joint pure plan mixture weight"
            )
            mixture_utility = _add_scaled(mixture_utility, weight, utility)
    average_profiles = {
        "O1": BehaviorStrategy(average_strategy["O1"]),
        "O2": BehaviorStrategy(average_strategy["O2"]),
    }
    direct_average = _oracle_direct_evaluate(
        effective_tree, effective_hero, average_profiles
    )
    evaluation_count += 1
    mixed_utility_delta = max(
        abs(getattr(mixture_utility, p) - getattr(direct_average, p))
        for p in ("H", "O1", "O2", "R")
    )

    oracle_mixed_gains: Dict[str, float] = {}
    direct_mixed_gains: Dict[str, float] = {}
    for player in OPPONENT_PLAYERS:
        base = direct_average.for_player(player)
        candidate_values = []
        direct_values = []
        if player == "O1":
            for i, plan in enumerate(plans1):
                candidate_values.append(
                    _ordered_fsum(
                        (weights2[j] * table[i][j].O1 for j in range(n2)),
                        "oracle O1 mixed alternative",
                    )
                )
                direct_values.append(
                    _oracle_direct_evaluate(
                        effective_tree,
                        effective_hero,
                        {
                            "O1": _pure_strategy_to_behavior(
                                metadata.opponent_info_sets["O1"], plan
                            ),
                            "O2": average_profiles["O2"],
                        },
                    ).O1
                )
        else:
            for j, plan in enumerate(plans2):
                candidate_values.append(
                    _ordered_fsum(
                        (weights1[i] * table[i][j].O2 for i in range(n1)),
                        "oracle O2 mixed alternative",
                    )
                )
                direct_values.append(
                    _oracle_direct_evaluate(
                        effective_tree,
                        effective_hero,
                        {
                            "O1": average_profiles["O1"],
                            "O2": _pure_strategy_to_behavior(
                                metadata.opponent_info_sets["O2"], plan
                            ),
                        },
                    ).O2
                )
        evaluation_count += len(direct_values)
        oracle_mixed_gains[player] = _checked_float_subtract(
            max(candidate_values), base, f"oracle mixed gain for {player}"
        )
        direct_mixed_gains[player] = _checked_float_subtract(
            max(direct_values), base, f"direct mixed gain for {player}"
        )

    rows = []
    stable_ids = []
    hash_rows = []
    for i, plan1 in enumerate(plans1):
        for j, plan2 in enumerate(plans2):
            utility = table[i][j]
            if n1 == 1:
                gain1 = 0.0
            else:
                gain1 = max(table[k][j].O1 for k in range(n1) if k != i) - utility.O1
            if n2 == 1:
                gain2 = 0.0
            else:
                gain2 = max(table[i][k].O2 for k in range(n2) if k != j) - utility.O2
            profile_id = f"O1:{i}|O2:{j}"
            stable = (
                gain1 <= config.epsilon_deviation
                and gain2 <= config.epsilon_deviation
            )
            row = {
                "profile_id": profile_id,
                "plans": {"O1": plan1, "O2": plan2},
                "utility": utility.to_dict(),
                "unilateral_gain": {"O1": gain1, "O2": gain2},
                "pure_profile_unilateral_stability": stable,
            }
            hash_rows.append(row)
            if stable:
                stable_ids.append(profile_id)
            if config.include_oracle_rows:
                rows.append(row)

    gain_delta = max(
        abs(oracle_mixed_gains[p] - direct_mixed_gains[p])
        for p in OPPONENT_PLAYERS
    )
    maximum_delta = max(max_pure_delta, mixed_utility_delta, gain_delta)
    if maximum_delta > config.oracle_compare_tolerance:
        if (
            maximum_delta - config.oracle_compare_tolerance
            <= config.reproducibility_tolerance
        ):
            status = INDETERMINATE_TOLERANCE
        else:
            status = ORACLE_MISMATCH
    else:
        status = "MATCH"
    oracle_warnings = []
    if not stable_ids:
        oracle_warnings.append("zero_pure_profile_unilateral_stability_rows")
    if len(stable_ids) > 1:
        oracle_warnings.append("multiple_pure_profile_unilateral_stability_rows")
    utility_signatures = [
        tuple(row["utility"][player] for player in ("H", "O1", "O2", "R"))
        for row in hash_rows
    ]
    if len(set(utility_signatures)) < len(utility_signatures):
        oracle_warnings.append("pure_profile_utility_ties_present")
    return {
        "status": status,
        "coverage": "complete",
        "counts": {
            **counts,
            "actual_profile_evaluations": evaluation_count,
            "actual_output_rows": len(rows),
            "complete_table_rows": n_joint,
        },
        "ordering": "O1 plan index then O2 plan index; sorted information sets; tree action order",
        "stable_profile_ids": stable_ids,
        "stable_profile_count": len(stable_ids),
        "warnings": oracle_warnings,
        "table_content_identity": _content_hash(hash_rows),
        "comparisons": {
            "pure_utility_max_delta": max_pure_delta,
            "average_profile_utility_max_delta": mixed_utility_delta,
            "unilateral_gain_max_delta": gain_delta,
            "tolerance": config.oracle_compare_tolerance,
        },
        "average_profile_unilateral_gain": {
            "oracle_table": oracle_mixed_gains,
            "direct": direct_mixed_gains,
        },
        "rows": rows,
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


def _ordered_fsum(values: Iterable[float], label: str) -> float:
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} could not be represented as a finite float") from exc
    return _require_finite_derived(result, label)


def _normalize_ordered_distribution(
    *,
    label: str,
    keys: Tuple[str, ...],
    values: Tuple[float, ...],
    input_tolerance: float,
) -> Tuple[Tuple[float, ...], dict]:
    for key, value in zip(keys, values):
        _require_probability(value, f"{label} probability {key}")
    raw_sum = _ordered_fsum(values, f"{label} raw probability sum")
    if raw_sum <= 0.0:
        raise ValueError(f"{label} raw probability sum must be positive")
    if abs(raw_sum - 1.0) > input_tolerance:
        raise ValueError(
            f"{label} raw probability sum {raw_sum} is outside input_tolerance"
        )
    factor = _checked_float_divide(1.0, raw_sum, f"{label} normalization factor")
    effective = tuple(
        _checked_float_divide(value, raw_sum, f"{label} effective probability {key}")
        for key, value in zip(keys, values)
    )
    effective_sum = _ordered_fsum(effective, f"{label} effective probability sum")
    if not math.isfinite(effective_sum):
        raise DiagnosticContractError(NUMERIC_FAILURE, f"{label} normalization failed")
    return effective, {
        "label": label,
        "ordered_keys": list(keys),
        "raw_probabilities": {
            key: value for key, value in zip(keys, values)
        },
        "raw_sum": raw_sum,
        "normalization_factor": factor,
        "effective_probabilities": {
            key: probability for key, probability in zip(keys, effective)
        },
        "effective_sum": effective_sum,
    }


def _effective_inputs(
    tree: ThreePlayerGameTree,
    fixed_hero_policy: BehaviorStrategy,
    opponent_strategies: Optional[Mapping[OpponentId, BehaviorStrategy]],
    *,
    input_tolerance: float,
    limits: CfrSafetyLimits,
) -> Tuple[
    ThreePlayerGameTree,
    BehaviorStrategy,
    Optional[Dict[OpponentId, BehaviorStrategy]],
    Tuple[dict, ...],
    _TreeMetadata,
]:
    metadata = _collect_tree_metadata(
        tree, tolerance=input_tolerance, limits=limits
    )
    _validate_behavior_strategy(
        "fixed Hero", fixed_hero_policy, metadata.fixed_hero_info_sets, input_tolerance
    )
    if opponent_strategies is not None:
        _validate_opponent_strategy_mapping(
            opponent_strategies, metadata.opponent_info_sets, input_tolerance
        )
    records: list[dict] = []

    def normalized_strategy(
        label: str,
        strategy: BehaviorStrategy,
        info_sets: Mapping[InfoSetId, Tuple[Action, ...]],
    ) -> BehaviorStrategy:
        result: Dict[str, Dict[str, float]] = {}
        for info_set in sorted(info_sets):
            actions = info_sets[info_set]
            values = tuple(float(strategy.probabilities[info_set][a]) for a in actions)
            effective, record = _normalize_ordered_distribution(
                label=f"{label}.{info_set}",
                keys=actions,
                values=values,
                input_tolerance=input_tolerance,
            )
            records.append(record)
            result[info_set] = dict(zip(actions, effective))
        return BehaviorStrategy(result)

    def normalized_node(node: ThreePlayerNode) -> ThreePlayerNode:
        if isinstance(node, ThreePlayerTerminalNode):
            return node
        if isinstance(node, ThreePlayerChanceNode):
            keys = tuple(str(index) for index in range(len(node.children)))
            values = tuple(float(probability) for probability, _ in node.children)
            effective, record = _normalize_ordered_distribution(
                label=f"chance.{node.node_id}",
                keys=keys,
                values=values,
                input_tolerance=input_tolerance,
            )
            records.append(record)
            return ThreePlayerChanceNode(
                node.node_id,
                tuple(
                    (probability, normalized_node(child))
                    for probability, (_, child) in zip(effective, node.children)
                ),
            )
        if isinstance(node, FixedHeroNode):
            return FixedHeroNode(
                node.node_id,
                node.info_set,
                tuple((action, normalized_node(child)) for action, child in node.actions),
            )
        if isinstance(node, OpponentDecisionNode):
            return OpponentDecisionNode(
                node.node_id,
                node.owner,
                node.info_set,
                tuple((action, normalized_node(child)) for action, child in node.actions),
            )
        raise TypeError(f"unknown node type: {type(node)!r}")

    effective_tree = ThreePlayerGameTree(
        normalized_node(tree.root), description=tree.description
    )
    effective_hero = normalized_strategy(
        "fixed_hero", fixed_hero_policy, metadata.fixed_hero_info_sets
    )
    effective_opponents = None
    if opponent_strategies is not None:
        effective_opponents = {
            player: normalized_strategy(
                player, opponent_strategies[player], metadata.opponent_info_sets[player]
            )
            for player in OPPONENT_PLAYERS
        }
    return effective_tree, effective_hero, effective_opponents, tuple(records), metadata


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
    seen_node_ids: set[str] = set()

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
        if not isinstance(node.node_id, str) or not node.node_id:
            raise ValueError("node ids must be non-empty strings")
        if node.node_id in seen_node_ids:
            raise ValueError(f"duplicate node id {node.node_id!r}")
        seen_node_ids.add(node.node_id)
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
            probabilities = []
            for prob, child in node.children:
                _require_probability(prob, f"chance node {node.node_id!r} probability")
                probabilities.append(float(prob))
                walk(child, seen_o1, seen_o2)
            total = _ordered_fsum(
                probabilities, f"chance node {node.node_id!r} probability sum"
            )
            if total <= 0.0:
                raise ValueError(
                    f"chance node {node.node_id!r} probability sum must be positive"
                )
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
        probabilities = []
        for action in legal_actions:
            probability = distribution[action]
            _require_probability(probability, f"{label} probability {info_set}.{action}")
            probabilities.append(float(probability))
        total = _ordered_fsum(
            probabilities, f"{label} policy for {info_set!r} probability sum"
        )
        if total <= 0.0:
            raise ValueError(
                f"{label} policy for {info_set!r} probability sum must be positive"
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
