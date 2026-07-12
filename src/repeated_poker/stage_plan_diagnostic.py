"""Bounded exhaustive one-period stage-plan deviation diagnostic.

The reference path in this module accepts exact rational probabilities,
payoffs, discounting, thresholds, and error budgets.  It reports the maximum
checked one-period deviation gain at period boundaries.  A PASS means only
that no checked one-period pure stage-plan gain exceeds ``epsilon_claim``
under the stated numerical bound.

The supported model has two public states, ``C`` and absorbing ``P``.  A stage
finishes before its public signal updates the state used in the next period.
The result does not make claims about decisions after an information set is
reached, zero-reach action quality, beliefs, arbitrary full-history profiles,
finite punishments, or known finite horizons.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Mapping, Optional, Tuple

from .game import (
    ChanceNode,
    GameTree,
    HeroNode,
    Node,
    TerminalNode,
    VillainNode,
    collect_hero_info_sets,
    collect_villain_info_sets,
    iter_nodes,
)

HERO = "Hero"
VILLAIN = "Villain"
PLAYERS = (HERO, VILLAIN)
COOPERATE = "C"
PUNISH = "P"
PUBLIC_STATES = (COOPERATE, PUNISH)


class DiagnosticStatus(str, Enum):
    """Fail-closed outcome classes for the bounded diagnostic."""

    PASS = "PASS"
    FAIL = "FAIL"
    INDETERMINATE = "INDETERMINATE"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class PublicAction:
    """One publicly observed action, retaining actor and order."""

    actor: str
    action: str


@dataclass(frozen=True)
class PublicSignal:
    """The ordered public action trace and public terminal observable."""

    action_trace: Tuple[PublicAction, ...]
    terminal_observable: str


@dataclass(frozen=True)
class PublicMonitoring:
    """Finite public observation map and total deterministic state update."""

    public_action_node_ids: frozenset[str]
    terminal_observables: Mapping[str, str]
    signal_alphabet: Tuple[PublicSignal, ...]
    transitions: Mapping[Tuple[str, PublicSignal], str]


@dataclass(frozen=True)
class ModelClassAttestation:
    """Explicit fixture assertions that cannot be inferred from a stage tree."""

    iid_stage_kernel: bool
    no_persistent_private_state: bool
    no_cross_period_correlation: bool
    no_private_payoff_state: bool
    public_state_does_not_change_stage_kernel: bool
    public_state_is_sufficient: bool
    signal_partition_is_public: bool
    signal_excludes_private_information: bool
    signal_excludes_deviator_identity: bool
    no_known_finite_horizon: bool
    absorbing_grim_only: bool


@dataclass(frozen=True)
class RecallHistory:
    """A player's prior observations, own actions, and information sets.

    ``information_sets`` and ``own_actions`` are checked against the stage-tree
    path.  ``observations`` retains fixture-specific manual evidence that the
    tree cannot derive.
    """

    observations: Tuple[str, ...]
    own_actions: Tuple[str, ...]
    information_sets: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ManualPerfectRecallAttestation:
    """Fixture-specific record required in addition to structural validation."""

    fixture_id: str
    tree_content_identity: str
    target_version: str
    information_set_members: Mapping[str, Mapping[str, Tuple[str, ...]]]
    member_histories: Mapping[
        str, Mapping[str, Mapping[str, RecallHistory]]
    ]
    legal_actions: Mapping[str, Mapping[str, Tuple[str, ...]]]
    reviewer: str
    review_date: str
    review_method: str
    evidence: str
    result_confirmed: bool
    known_limitations: Tuple[str, ...]
    invalidation_conditions: Tuple[str, ...]
    valid_through_version: str
    invalidated: bool


@dataclass(frozen=True)
class NumericErrorBound:
    """Explicit upper bounds for every numerical error component.

    All fields are on the unnormalized discounted-value scale.  The exact
    reference path uses zeros.  Larger non-negative bounds may conservatively
    widen every per-deviation interval.
    """

    probability_representation: Fraction
    probability_normalization: Fraction
    stage_expectation: Fraction
    continuation_expectation: Fraction
    subtraction: Fraction
    maximum: Fraction
    bellman_residual: Fraction
    residual_evaluation: Fraction
    enclosure_established: bool


@dataclass(frozen=True)
class PureStagePlan:
    """One legal action at every stage information set of one player."""

    player: str
    actions: Tuple[Tuple[str, str], ...]

    def as_mapping(self) -> Mapping[str, str]:
        return dict(self.actions)


@dataclass(frozen=True)
class DeviationEnclosure:
    """Exact estimate and enclosing interval for one checked deviation."""

    player: str
    state: str
    plan: PureStagePlan
    prescribed_value: Fraction
    deviation_value: Fraction
    gain: Fraction
    lower: Fraction
    upper: Fraction


@dataclass(frozen=True)
class DiagnosticResult:
    """Result of the bounded exhaustive one-period stage-plan diagnostic."""

    status: DiagnosticStatus
    message: str
    input_tolerance: Optional[Fraction]
    epsilon_claim: Optional[Fraction]
    delta: Optional[Fraction]
    stage_payoff_bound: Optional[Fraction]
    unnormalized_value_scale: Optional[Fraction]
    numeric_error_bound: NumericErrorBound
    plan_counts: Mapping[str, int]
    prescribed_values: Mapping[Tuple[str, str], Fraction]
    deviations: Tuple[DeviationEnclosure, ...]
    maximum_lower: Optional[Fraction]
    maximum_upper: Optional[Fraction]


class _NonExactNumeric(Exception):
    pass


def exact_zero_error_bound() -> NumericErrorBound:
    """Return an explicit all-zero error budget for the exact rational path."""

    zero = Fraction(0)
    return NumericErrorBound(
        probability_representation=zero,
        probability_normalization=zero,
        stage_expectation=zero,
        continuation_expectation=zero,
        subtraction=zero,
        maximum=zero,
        bellman_residual=zero,
        residual_evaluation=zero,
        enclosure_established=True,
    )


def _fraction(value: object, name: str) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"{name} must not be bool")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        raise _NonExactNumeric(name)
    raise ValueError(f"{name} must be an exact rational number")


def _canonical_number(value: object) -> str:
    number = _fraction(value, "tree numeric value")
    return f"{number.numerator}/{number.denominator}"


def tree_content_identity(tree: GameTree) -> str:
    """Return a stable identity for the complete stage-tree content."""

    def encode(node: Node) -> tuple:
        if isinstance(node, TerminalNode):
            return (
                "T",
                node.node_id,
                _canonical_number(node.hero_ev),
                _canonical_number(node.villain_ev),
                _canonical_number(node.house_rake),
            )
        if isinstance(node, ChanceNode):
            return (
                "C",
                node.node_id,
                tuple((_canonical_number(p), encode(child)) for p, child in node.children),
            )
        if isinstance(node, HeroNode):
            return ("H", node.node_id, node.info_set, tuple((a, encode(c)) for a, c in node.actions))
        if isinstance(node, VillainNode):
            return ("V", node.node_id, node.info_set, tuple((a, encode(c)) for a, c in node.actions))
        raise TypeError(f"unknown node type: {type(node)!r}")

    return hashlib.sha256(repr(encode(tree.root)).encode("utf-8")).hexdigest()


def _info_sets(tree: GameTree) -> Mapping[str, Mapping[str, Tuple[str, ...]]]:
    return {
        HERO: collect_hero_info_sets(tree),
        VILLAIN: collect_villain_info_sets(tree),
    }


def _member_nodes(tree: GameTree) -> Mapping[str, Mapping[str, Tuple[str, ...]]]:
    members = {HERO: {}, VILLAIN: {}}
    for node in iter_nodes(tree.root):
        if isinstance(node, HeroNode):
            members[HERO].setdefault(node.info_set, []).append(node.node_id)
        elif isinstance(node, VillainNode):
            members[VILLAIN].setdefault(node.info_set, []).append(node.node_id)
    return {
        player: {key: tuple(value) for key, value in per_player.items()}
        for player, per_player in members.items()
    }


def _member_path_histories(
    tree: GameTree,
) -> Mapping[str, Mapping[str, Mapping[str, RecallHistory]]]:
    """Derive each member's prior own information sets and actions."""

    histories = {HERO: {}, VILLAIN: {}}

    def walk(
        node: Node,
        hero_information_sets: Tuple[str, ...],
        hero_actions: Tuple[str, ...],
        villain_information_sets: Tuple[str, ...],
        villain_actions: Tuple[str, ...],
    ) -> None:
        if isinstance(node, TerminalNode):
            return
        if isinstance(node, ChanceNode):
            for _, child in node.children:
                walk(
                    child,
                    hero_information_sets,
                    hero_actions,
                    villain_information_sets,
                    villain_actions,
                )
            return
        if isinstance(node, HeroNode):
            histories[HERO].setdefault(node.info_set, {})[node.node_id] = RecallHistory(
                observations=(),
                own_actions=hero_actions,
                information_sets=hero_information_sets,
            )
            for action, child in node.actions:
                walk(
                    child,
                    hero_information_sets + (node.info_set,),
                    hero_actions + (action,),
                    villain_information_sets,
                    villain_actions,
                )
            return
        if isinstance(node, VillainNode):
            histories[VILLAIN].setdefault(node.info_set, {})[node.node_id] = RecallHistory(
                observations=(),
                own_actions=villain_actions,
                information_sets=villain_information_sets,
            )
            for action, child in node.actions:
                walk(
                    child,
                    hero_information_sets,
                    hero_actions,
                    villain_information_sets + (node.info_set,),
                    villain_actions + (action,),
                )
            return
        raise TypeError(f"unknown node type: {type(node)!r}")

    walk(tree.root, (), (), (), ())
    return histories


def _validate_unique_node_ids(tree: GameTree) -> None:
    ids = [node.node_id for node in iter_nodes(tree.root)]
    if len(ids) != len(set(ids)):
        raise ValueError("stage tree node IDs must be unique")


def _validate_exact_tree(tree: GameTree, payoff_bound: Fraction) -> None:
    _validate_unique_node_ids(tree)
    seen_terminal = False
    for node in iter_nodes(tree.root):
        if isinstance(node, TerminalNode):
            seen_terminal = True
            hero = _fraction(node.hero_ev, f"terminal {node.node_id} Hero payoff")
            villain = _fraction(node.villain_ev, f"terminal {node.node_id} Villain payoff")
            residual = _fraction(node.house_rake, f"terminal {node.node_id} residual")
            if abs(hero) > payoff_bound or abs(villain) > payoff_bound:
                raise ValueError(f"terminal {node.node_id!r} exceeds stage_payoff_bound")
            if hero + villain + residual != 0:
                raise ValueError(f"terminal {node.node_id!r} violates the accounting identity")
        elif isinstance(node, ChanceNode):
            if not node.children:
                raise ValueError(f"chance node {node.node_id!r} has no children")
            probabilities = [_fraction(p, f"chance node {node.node_id} probability") for p, _ in node.children]
            if any(p < 0 for p in probabilities):
                raise ValueError(f"chance node {node.node_id!r} has a negative probability")
            if sum(probabilities, Fraction(0)) != 1:
                raise ValueError(f"chance node {node.node_id!r} probabilities must sum exactly to one")
        elif isinstance(node, (HeroNode, VillainNode)):
            actions = [action for action, _ in node.actions]
            if not actions or len(actions) != len(set(actions)):
                raise ValueError(f"decision node {node.node_id!r} has invalid legal actions")
        else:
            raise TypeError(f"unknown node type: {type(node)!r}")
    if not seen_terminal:
        raise ValueError("stage tree has no terminal")
    _info_sets(tree)

    def check_paths(
        node: Node, hero_seen: frozenset[str], villain_seen: frozenset[str]
    ) -> None:
        if isinstance(node, TerminalNode):
            return
        if isinstance(node, ChanceNode):
            for _, child in node.children:
                check_paths(child, hero_seen, villain_seen)
            return
        if isinstance(node, HeroNode):
            if node.info_set in hero_seen:
                raise ValueError("a Hero information set repeats on one stage path")
            for _, child in node.actions:
                check_paths(child, hero_seen | {node.info_set}, villain_seen)
            return
        if isinstance(node, VillainNode):
            if node.info_set in villain_seen:
                raise ValueError("a Villain information set repeats on one stage path")
            for _, child in node.actions:
                check_paths(child, hero_seen, villain_seen | {node.info_set})
            return
        raise TypeError(f"unknown node type: {type(node)!r}")

    check_paths(tree.root, frozenset(), frozenset())


def _unsupported_attestation_reason(
    tree: GameTree,
    version: str,
    attestation: Optional[ManualPerfectRecallAttestation],
) -> Optional[str]:
    if attestation is None:
        return "manual perfect-recall attestation is missing"
    required_text = (
        attestation.fixture_id,
        attestation.target_version,
        attestation.reviewer,
        attestation.review_date,
        attestation.review_method,
        attestation.evidence,
    )
    if any(not value for value in required_text):
        return "manual perfect-recall attestation is incomplete"
    if not attestation.invalidation_conditions:
        return "manual perfect-recall attestation lacks invalidation conditions"
    if not attestation.result_confirmed or attestation.invalidated:
        return "manual perfect-recall attestation is not current and confirmed"
    identity = tree_content_identity(tree)
    if attestation.tree_content_identity != identity:
        return "manual perfect-recall attestation tree identity does not match"
    if attestation.target_version != version or attestation.valid_through_version != version:
        return "manual perfect-recall attestation version is expired or mismatched"

    expected_actions = _info_sets(tree)
    expected_members = _member_nodes(tree)
    expected_histories = _member_path_histories(tree)
    if set(attestation.information_set_members) != set(PLAYERS):
        return "manual perfect-recall attestation player set does not match"
    if set(attestation.member_histories) != set(PLAYERS):
        return "manual perfect-recall attestation history player set does not match"
    if set(attestation.legal_actions) != set(PLAYERS):
        return "manual perfect-recall attestation legal-action player set does not match"
    for player in PLAYERS:
        if dict(attestation.information_set_members[player]) != dict(expected_members[player]):
            return f"manual perfect-recall attestation members do not match for {player}"
        if dict(attestation.legal_actions[player]) != dict(expected_actions[player]):
            return f"manual perfect-recall attestation legal actions do not match for {player}"
        histories = attestation.member_histories[player]
        if set(histories) != set(expected_members[player]):
            return f"manual perfect-recall attestation histories do not match for {player}"
        for info_set, nodes in expected_members[player].items():
            by_node = histories[info_set]
            if set(by_node) != set(nodes):
                return f"manual perfect-recall attestation node histories do not match for {player}"
            values = [by_node[node_id] for node_id in nodes]
            if not all(isinstance(value, RecallHistory) for value in values):
                return "manual perfect-recall attestation has an invalid history record"
            for node_id, value in zip(nodes, values):
                actual = expected_histories[player][info_set][node_id]
                if (
                    value.information_sets != actual.information_sets
                    or value.own_actions != actual.own_actions
                ):
                    return f"manual perfect-recall history does not match tree path at {node_id!r}"
            if values and any(value != values[0] for value in values[1:]):
                return f"manual perfect-recall histories differ within {info_set!r}"
    return None


def _unsupported_model_reason(attestation: ModelClassAttestation) -> Optional[str]:
    for field_name in attestation.__dataclass_fields__:
        if getattr(attestation, field_name) is not True:
            return f"unsupported model-class assertion: {field_name}"
    return None


def _terminal_signals(
    tree: GameTree, monitoring: PublicMonitoring
) -> Mapping[str, PublicSignal]:
    all_ids = {node.node_id for node in iter_nodes(tree.root)}
    decision_ids = {
        node.node_id
        for node in iter_nodes(tree.root)
        if isinstance(node, (HeroNode, VillainNode))
    }
    if not monitoring.public_action_node_ids <= decision_ids:
        raise ValueError("public_action_node_ids contains an unknown or non-decision node")
    terminal_ids = {
        node.node_id for node in iter_nodes(tree.root) if isinstance(node, TerminalNode)
    }
    if set(monitoring.terminal_observables) != terminal_ids:
        raise ValueError("terminal_observables must specify every and only terminal node")
    if any(not isinstance(value, str) or not value for value in monitoring.terminal_observables.values()):
        raise ValueError("every terminal observable must be a non-empty string")
    if not monitoring.public_action_node_ids <= all_ids:
        raise ValueError("public action declaration contains an unknown node")

    result = {}

    def walk(node: Node, trace: Tuple[PublicAction, ...]) -> None:
        if isinstance(node, TerminalNode):
            result[node.node_id] = PublicSignal(trace, monitoring.terminal_observables[node.node_id])
            return
        if isinstance(node, ChanceNode):
            for _, child in node.children:
                walk(child, trace)
            return
        if isinstance(node, (HeroNode, VillainNode)):
            actor = HERO if isinstance(node, HeroNode) else VILLAIN
            for action, child in node.actions:
                next_trace = trace
                if node.node_id in monitoring.public_action_node_ids:
                    next_trace = trace + (PublicAction(actor, action),)
                walk(child, next_trace)
            return
        raise TypeError(f"unknown node type: {type(node)!r}")

    walk(tree.root, ())
    feasible = tuple(dict.fromkeys(result.values()))
    if len(monitoring.signal_alphabet) != len(set(monitoring.signal_alphabet)):
        raise ValueError("signal_alphabet contains duplicates")
    if set(monitoring.signal_alphabet) != set(feasible):
        raise ValueError("signal_alphabet must equal the complete feasible public signal set")
    expected_keys = {
        (state, signal) for state in PUBLIC_STATES for signal in monitoring.signal_alphabet
    }
    if set(monitoring.transitions) != expected_keys:
        raise ValueError("transitions must be total on Q x Y with no unknown entries")
    if any(next_state not in PUBLIC_STATES for next_state in monitoring.transitions.values()):
        raise ValueError("transition has an unknown next state")
    for signal in monitoring.signal_alphabet:
        if monitoring.transitions[(PUNISH, signal)] != PUNISH:
            raise ValueError("P must transition to P for every feasible signal")
    return result


def _validate_profile(
    profile: Mapping[str, Mapping[str, Mapping[str, Mapping[str, object]]]],
    info_sets: Mapping[str, Mapping[str, Tuple[str, ...]]],
) -> Mapping[str, Mapping[str, Mapping[str, Mapping[str, Fraction]]]]:
    if set(profile) != set(PUBLIC_STATES):
        raise ValueError("profile must specify exactly states C and P")
    exact = {}
    for state in PUBLIC_STATES:
        if set(profile[state]) != set(PLAYERS):
            raise ValueError(f"profile at {state} must specify exactly both players")
        exact[state] = {}
        for player in PLAYERS:
            supplied = profile[state][player]
            legal = info_sets[player]
            if set(supplied) != set(legal):
                raise ValueError(f"profile at {state} for {player} has missing or unknown information sets")
            exact[state][player] = {}
            for info_set, actions in legal.items():
                distribution = supplied[info_set]
                if set(distribution) != set(actions):
                    raise ValueError(f"profile at {state}/{player}/{info_set} has missing or illegal actions")
                probabilities = {
                    action: _fraction(distribution[action], f"profile probability {state}/{player}/{info_set}/{action}")
                    for action in actions
                }
                if any(probability < 0 for probability in probabilities.values()):
                    raise ValueError(f"profile at {state}/{player}/{info_set} has a negative probability")
                if sum(probabilities.values(), Fraction(0)) != 1:
                    raise ValueError(f"profile at {state}/{player}/{info_set} must sum exactly to one")
                exact[state][player][info_set] = probabilities
    return exact


def _validate_error_bound(bound: NumericErrorBound) -> None:
    for field_name in (
        "probability_representation",
        "probability_normalization",
        "stage_expectation",
        "continuation_expectation",
        "subtraction",
        "maximum",
        "bellman_residual",
        "residual_evaluation",
    ):
        component = getattr(bound, field_name)
        if isinstance(component, float) and math.isfinite(component) and component < 0:
            raise ValueError(f"numeric_error_bound.{field_name} must be non-negative")
        value = _fraction(component, f"numeric_error_bound.{field_name}")
        if value < 0:
            raise ValueError(f"numeric_error_bound.{field_name} must be non-negative")
    if not isinstance(bound.enclosure_established, bool):
        raise ValueError("numeric_error_bound.enclosure_established must be bool")


def _plan_count(actions_by_info_set: Mapping[str, Tuple[str, ...]]) -> int:
    count = 1
    for actions in actions_by_info_set.values():
        count *= len(actions)
    return count


def _plans(player: str, actions_by_info_set: Mapping[str, Tuple[str, ...]]) -> Tuple[PureStagePlan, ...]:
    info_sets = tuple(actions_by_info_set)
    choices = tuple(actions_by_info_set[info_set] for info_set in info_sets)
    return tuple(
        PureStagePlan(player, tuple(zip(info_sets, selected)))
        for selected in itertools.product(*choices)
    )


Vector = Tuple[Fraction, Fraction, Fraction, Fraction, Fraction]


def _add_weighted(total: Vector, weight: Fraction, value: Vector) -> Vector:
    return tuple(a + weight * b for a, b in zip(total, value))  # type: ignore[return-value]


def _evaluate_stage(
    node: Node,
    state: str,
    profile: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Fraction]]]],
    signals: Mapping[str, PublicSignal],
    monitoring: PublicMonitoring,
    deviating_player: Optional[str] = None,
    plan: Optional[Mapping[str, str]] = None,
) -> Vector:
    if isinstance(node, TerminalNode):
        next_state = monitoring.transitions[(state, signals[node.node_id])]
        return (
            _fraction(node.hero_ev, "Hero payoff"),
            _fraction(node.villain_ev, "Villain payoff"),
            _fraction(node.house_rake, "house residual"),
            Fraction(next_state == COOPERATE),
            Fraction(next_state == PUNISH),
        )
    if isinstance(node, ChanceNode):
        total: Vector = (Fraction(0),) * 5  # type: ignore[assignment]
        for probability, child in node.children:
            total = _add_weighted(
                total,
                _fraction(probability, "chance probability"),
                _evaluate_stage(child, state, profile, signals, monitoring, deviating_player, plan),
            )
        return total
    if isinstance(node, (HeroNode, VillainNode)):
        player = HERO if isinstance(node, HeroNode) else VILLAIN
        if player == deviating_player:
            assert plan is not None
            selected = plan[node.info_set]
            for action, child in node.actions:
                if action == selected:
                    return _evaluate_stage(child, state, profile, signals, monitoring, deviating_player, plan)
            raise AssertionError("validated plan action was not found")
        total = (Fraction(0),) * 5  # type: ignore[assignment]
        distribution = profile[state][player][node.info_set]
        for action, child in node.actions:
            total = _add_weighted(
                total,
                distribution[action],
                _evaluate_stage(child, state, profile, signals, monitoring, deviating_player, plan),
            )
        return total
    raise TypeError(f"unknown node type: {type(node)!r}")


def _prescribed_values(
    tree: GameTree,
    profile: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Fraction]]]],
    signals: Mapping[str, PublicSignal],
    monitoring: PublicMonitoring,
    delta: Fraction,
) -> Tuple[Mapping[Tuple[str, str], Fraction], Mapping[str, Vector]]:
    rows = {
        state: _evaluate_stage(tree.root, state, profile, signals, monitoring)
        for state in PUBLIC_STATES
    }
    values = {}
    for player, payoff_index in ((HERO, 0), (VILLAIN, 1)):
        p_row = rows[PUNISH]
        value_p = p_row[payoff_index] / (1 - delta)
        c_row = rows[COOPERATE]
        denominator = 1 - delta * c_row[3]
        value_c = (c_row[payoff_index] + delta * c_row[4] * value_p) / denominator
        values[(player, COOPERATE)] = value_c
        values[(player, PUNISH)] = value_p
    return values, rows


def nuts_chop_negative_oracle(delta: Fraction) -> Mapping[str, Fraction]:
    """Return the exact fixed nuts-chop payoffs and rewarding-state gain."""

    delta = _fraction(delta, "delta")
    if not 0 < delta < 1:
        raise ValueError("delta must satisfy 0 < delta < 1")
    return {
        "check_check_hero": Fraction(-1, 20),
        "check_check_villain": Fraction(-1, 20),
        "shove_fold_hero": Fraction(-1),
        "shove_fold_villain": Fraction(1),
        "shove_call_hero": Fraction(-2),
        "shove_call_villain": Fraction(-2),
        "fold_minus_call_same_next_state": Fraction(1),
        "villain_gain": (Fraction(-39, 20) + 3 * delta) / (1 - delta),
    }


def _negative_oracle_consistent() -> bool:
    tie = nuts_chop_negative_oracle(Fraction(13, 20))
    above = nuts_chop_negative_oracle(Fraction(2, 3))
    return (
        tie["fold_minus_call_same_next_state"] == 1
        and tie["villain_gain"] == 0
        and above["villain_gain"] > 0
    )


def _empty_result(
    status: DiagnosticStatus,
    message: str,
    numeric_error_bound: NumericErrorBound,
    *,
    input_tolerance: Optional[Fraction] = None,
    epsilon_claim: Optional[Fraction] = None,
    delta: Optional[Fraction] = None,
    payoff_bound: Optional[Fraction] = None,
    plan_counts: Optional[Mapping[str, int]] = None,
) -> DiagnosticResult:
    scale = None if delta is None or payoff_bound is None else payoff_bound / (1 - delta)
    return DiagnosticResult(
        status=status,
        message=message,
        input_tolerance=input_tolerance,
        epsilon_claim=epsilon_claim,
        delta=delta,
        stage_payoff_bound=payoff_bound,
        unnormalized_value_scale=scale,
        numeric_error_bound=numeric_error_bound,
        plan_counts={} if plan_counts is None else plan_counts,
        prescribed_values={},
        deviations=(),
        maximum_lower=None,
        maximum_upper=None,
    )


def diagnose_stage_plan_deviations(
    *,
    tree: GameTree,
    fixture_version: str,
    profile: Mapping[str, Mapping[str, Mapping[str, Mapping[str, object]]]],
    monitoring: PublicMonitoring,
    model_attestation: ModelClassAttestation,
    perfect_recall_attestation: Optional[ManualPerfectRecallAttestation],
    delta: object,
    stage_payoff_bound: object,
    input_tolerance: object,
    epsilon_claim: object,
    numeric_error_bound: NumericErrorBound,
    max_plans_per_player: int,
) -> DiagnosticResult:
    """Run the bounded exhaustive one-period stage-plan deviation diagnostic.

    Every numerical input is explicit.  Exact rational inputs produce exact
    Bellman values and exact gains; the supplied error budget encloses each
    reported gain.  A finite non-rational numeric representation is handled as
    ``INDETERMINATE`` because this reference path has no representation-error
    construction for it.
    """

    if not isinstance(max_plans_per_player, int) or isinstance(max_plans_per_player, bool):
        raise ValueError("max_plans_per_player must be an integer")
    if max_plans_per_player < 1:
        raise ValueError("max_plans_per_player must be positive")
    if not isinstance(fixture_version, str) or not fixture_version:
        raise ValueError("fixture_version must be a non-empty string")
    try:
        _validate_error_bound(numeric_error_bound)
        exact_delta = _fraction(delta, "delta")
        payoff_bound = _fraction(stage_payoff_bound, "stage_payoff_bound")
        tolerance = _fraction(input_tolerance, "input_tolerance")
        claim = _fraction(epsilon_claim, "epsilon_claim")
        if not 0 < exact_delta < 1:
            raise ValueError("delta must satisfy 0 < delta < 1")
        if payoff_bound < 0:
            raise ValueError("stage_payoff_bound must be non-negative")
        if tolerance < 0:
            raise ValueError("input_tolerance must be non-negative")
        if claim < 0:
            raise ValueError("epsilon_claim must be non-negative")
        _validate_exact_tree(tree, payoff_bound)
        info_sets = _info_sets(tree)
        exact_profile = _validate_profile(profile, info_sets)
    except _NonExactNumeric:
        return _empty_result(
            DiagnosticStatus.INDETERMINATE,
            "rigorous numeric enclosure is unavailable for a non-rational input representation",
            numeric_error_bound,
        )

    signals = _terminal_signals(tree, monitoring)

    model_reason = _unsupported_model_reason(model_attestation)
    if model_reason is not None:
        return _empty_result(
            DiagnosticStatus.UNSUPPORTED,
            model_reason,
            numeric_error_bound,
            input_tolerance=tolerance,
            epsilon_claim=claim,
            delta=exact_delta,
            payoff_bound=payoff_bound,
        )
    recall_reason = _unsupported_attestation_reason(tree, fixture_version, perfect_recall_attestation)
    if recall_reason is not None:
        return _empty_result(
            DiagnosticStatus.UNSUPPORTED,
            recall_reason,
            numeric_error_bound,
            input_tolerance=tolerance,
            epsilon_claim=claim,
            delta=exact_delta,
            payoff_bound=payoff_bound,
        )

    counts = {player: _plan_count(info_sets[player]) for player in PLAYERS}
    if any(count > max_plans_per_player for count in counts.values()):
        return _empty_result(
            DiagnosticStatus.UNSUPPORTED,
            "complete stage-plan enumeration exceeds max_plans_per_player",
            numeric_error_bound,
            input_tolerance=tolerance,
            epsilon_claim=claim,
            delta=exact_delta,
            payoff_bound=payoff_bound,
            plan_counts=counts,
        )
    if not numeric_error_bound.enclosure_established:
        return _empty_result(
            DiagnosticStatus.INDETERMINATE,
            "the numerical error enclosure is not established",
            numeric_error_bound,
            input_tolerance=tolerance,
            epsilon_claim=claim,
            delta=exact_delta,
            payoff_bound=payoff_bound,
            plan_counts=counts,
        )
    if not _negative_oracle_consistent():
        return _empty_result(
            DiagnosticStatus.INDETERMINATE,
            "the required negative oracle is inconsistent",
            numeric_error_bound,
            input_tolerance=tolerance,
            epsilon_claim=claim,
            delta=exact_delta,
            payoff_bound=payoff_bound,
            plan_counts=counts,
        )

    values, rows = _prescribed_values(tree, exact_profile, signals, monitoring, exact_delta)
    for player, payoff_index in ((HERO, 0), (VILLAIN, 1)):
        for state in PUBLIC_STATES:
            row = rows[state]
            residual = abs(
                values[(player, state)]
                - (row[payoff_index] + exact_delta * (row[3] * values[(player, COOPERATE)] + row[4] * values[(player, PUNISH)]))
            )
            if residual > numeric_error_bound.bellman_residual + numeric_error_bound.residual_evaluation:
                return _empty_result(
                    DiagnosticStatus.INDETERMINATE,
                    "the Bellman residual is outside the stated enclosure",
                    numeric_error_bound,
                    input_tolerance=tolerance,
                    epsilon_claim=claim,
                    delta=exact_delta,
                    payoff_bound=payoff_bound,
                    plan_counts=counts,
                )

    value_error = (
        numeric_error_bound.bellman_residual + numeric_error_bound.residual_evaluation
    ) / (1 - exact_delta)
    gain_error = (
        numeric_error_bound.probability_representation
        + numeric_error_bound.probability_normalization
        + numeric_error_bound.stage_expectation
        + numeric_error_bound.continuation_expectation
        + numeric_error_bound.subtraction
        + numeric_error_bound.maximum
        + (1 + exact_delta) * value_error
    )

    deviations = []
    for player, payoff_index in ((HERO, 0), (VILLAIN, 1)):
        player_plans = _plans(player, info_sets[player])
        if len(player_plans) != counts[player]:
            return _empty_result(
                DiagnosticStatus.UNSUPPORTED,
                "complete stage-plan enumeration did not match its precomputed count",
                numeric_error_bound,
                input_tolerance=tolerance,
                epsilon_claim=claim,
                delta=exact_delta,
                payoff_bound=payoff_bound,
                plan_counts=counts,
            )
        for state in PUBLIC_STATES:
            for pure_plan in player_plans:
                row = _evaluate_stage(
                    tree.root,
                    state,
                    exact_profile,
                    signals,
                    monitoring,
                    player,
                    pure_plan.as_mapping(),
                )
                deviation_value = row[payoff_index] + exact_delta * (
                    row[3] * values[(player, COOPERATE)]
                    + row[4] * values[(player, PUNISH)]
                )
                gain = deviation_value - values[(player, state)]
                deviations.append(
                    DeviationEnclosure(
                        player=player,
                        state=state,
                        plan=pure_plan,
                        prescribed_value=values[(player, state)],
                        deviation_value=deviation_value,
                        gain=gain,
                        lower=gain - gain_error,
                        upper=gain + gain_error,
                    )
                )

    maximum_lower = max(item.lower for item in deviations)
    maximum_upper = max(item.upper for item in deviations)
    if maximum_upper <= claim:
        status = DiagnosticStatus.PASS
        message = "no checked one-period pure stage-plan gain exceeds epsilon_claim under the stated numerical bound"
    elif maximum_lower > claim:
        status = DiagnosticStatus.FAIL
        message = "maximum checked one-period deviation gain at period boundaries exceeds epsilon_claim"
    else:
        status = DiagnosticStatus.INDETERMINATE
        message = "the per-deviation interval crosses epsilon_claim"
    return DiagnosticResult(
        status=status,
        message=message,
        input_tolerance=tolerance,
        epsilon_claim=claim,
        delta=exact_delta,
        stage_payoff_bound=payoff_bound,
        unnormalized_value_scale=payoff_bound / (1 - exact_delta),
        numeric_error_bound=numeric_error_bound,
        plan_counts=counts,
        prescribed_values=values,
        deviations=tuple(deviations),
        maximum_lower=maximum_lower,
        maximum_upper=maximum_upper,
    )
