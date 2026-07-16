"""Bounded endogenous strategy generation for the declared AIoF game.

The primary path solves the finite ``aiof-rational-lift-game-v1`` with two
compact exact-rational linear programs and independently verifies the selected
complete behavioural profile.  The optional normal-form oracle is tiny-input
only, and the alternating best-response entry point is a diagnostic with no
general convergence claim.  Nothing in this module is a poker chart, a
profitability claim, or a solver for games outside the stated heads-up,
fee-zero preflop model.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import sys
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any, Iterable, Sequence

from .aiof_cards import (
    AiofContractError,
    AiofLimits,
    AiofStatus,
    PreparedRanges,
    RangeEntry,
    RangeSpec,
    WeightBasis,
    prepare_compatible_ranges,
)
from .aiof_chip_ev import (
    ComboActionProbability,
    HeadsUpChipEvGame,
    PushFoldRequest,
    SuppliedProfile,
    analyze_pushfold,
)
from .aiof_equity import (
    EquityAlgorithm,
    EquityRequest,
    OutcomeCounts,
    calculate_preflop_equity,
)


__all__ = [
    "AiofStrategyStatus",
    "AiofStrategyAlgorithm",
    "StrategyClaimKind",
    "HeuristicUpdateMode",
    "HeuristicDiagnosticStatus",
    "AiofStrategyLimits",
    "StrategyError",
    "ExactComboActionProbability",
    "ExactBehaviourProfile",
    "ExactActionValue",
    "ExactStrategyRow",
    "ExactGainSnapshot",
    "SimplexTracePoint",
    "SimplexRunSummary",
    "RationalVerificationWitness",
    "ReferenceOracleComparison",
    "Phase1FloatDiagnostic",
    "RationalStrategyRequest",
    "RationalStrategyResult",
    "RationalStrategyRunResult",
    "AlternatingBrDiagnosticRequest",
    "HeuristicTracePoint",
    "AlternatingBrDiagnostic",
    "AlternatingBrDiagnosticRunResult",
    "generate_rational_lift_strategy",
    "run_alternating_br_diagnostic",
]


GAME_ID = "aiof-rational-lift-game-v1"
SIMPLEX_ID = "aiof-bounded-two-phase-bland-simplex-v1"
VERIFIER_ID = "aiof-full-unilateral-rational-verifier-v1"
ORACLE_ID = "aiof-normal-form-support-oracle-v1"
HEURISTIC_ID = "aiof-alternating-exact-br-diagnostic-v1"
INITIALIZATION_ID = "half-half-complete-profile-v1"
EXACT_ID = "exact_exhaustive-v1"
EVALUATOR_ID = "repo-reference-best5of7-v1"
CHIP_ACCOUNTING_ID = "heads-up-fee0-net-chipev-v1"
SB_ACTIONS = ("shove", "fold")
BB_ACTIONS = ("call", "fold")


class AiofStrategyStatus(str, Enum):
    """Fail-closed outer statuses for strategy and diagnostic requests."""

    SUCCESS = "SUCCESS"
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_STRATEGY = "INVALID_STRATEGY"
    INPUT_PREPARATION_FAILED = "INPUT_PREPARATION_FAILED"
    UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"
    EXACT_PAYOFF_REQUIRED = "EXACT_PAYOFF_REQUIRED"
    PAYOFF_CONSTRUCTION_FAILED = "PAYOFF_CONSTRUCTION_FAILED"
    CAP_EXCEEDED = "CAP_EXCEEDED"
    EXACT_ARITHMETIC_CAP_EXCEEDED = "EXACT_ARITHMETIC_CAP_EXCEEDED"
    SOLVER_LIMIT_REACHED = "SOLVER_LIMIT_REACHED"
    SOLVER_CONTRACT_FAILURE = "SOLVER_CONTRACT_FAILURE"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    ORACLE_MISMATCH = "ORACLE_MISMATCH"
    NON_REPRODUCIBLE = "NON_REPRODUCIBLE"
    NUMERIC_FAILURE = "NUMERIC_FAILURE"


class AiofStrategyAlgorithm(str, Enum):
    """Supported primary strategy-generation algorithms."""

    COMPACT_RATIONAL_LP = "aiof-compact-zero-sum-rational-lp-v1"


class StrategyClaimKind(str, Enum):
    """Fully qualified claim class for the declared rational-lift game."""

    EXACT_NASH = "EXACT_NASH"
    EPSILON_NASH = "EPSILON_NASH"


class HeuristicUpdateMode(str, Enum):
    """Fixed update ordering for the separate BR diagnostic."""

    SIMULTANEOUS = "SIMULTANEOUS"
    SEQUENTIAL_SB_THEN_BB = "SEQUENTIAL_SB_THEN_BB"


class HeuristicDiagnosticStatus(str, Enum):
    """Bounded stopping classifications for the BR diagnostic."""

    DIAGNOSTIC_COMPLETE = "DIAGNOSTIC_COMPLETE"
    CYCLE_DETECTED = "CYCLE_DETECTED"
    ITERATION_CAP_REACHED = "ITERATION_CAP_REACHED"


@dataclass(frozen=True)
class AiofStrategyLimits:
    """Caller-lowerable hard ceilings for all phase-2 work."""

    max_solver_combos_per_side: int = 64
    max_payoff_cells: int = 4_096
    max_exact_board_evaluations: int = 10_000_000
    max_lp_variables_per_problem: int = 128
    max_lp_constraints_per_problem: int = 256
    max_tableau_cells: int = 100_000
    max_simplex_pivots: int = 100_000
    max_exact_rational_bits: int = 8_192
    max_trace_points: int = 200
    max_oracle_combos_per_side: int = 6
    max_oracle_pure_plans_per_side: int = 64
    max_oracle_matrix_cells: int = 4_096
    max_oracle_support_systems: int = 100_000
    max_heuristic_iterations: int = 10_000


@dataclass(frozen=True)
class StrategyError:
    """Bounded failure metadata without partial computational payloads."""

    message: str
    phase: str
    upstream_status: AiofStatus | None
    completed_payoff_cells: int
    completed_pivots: int


@dataclass(frozen=True)
class ExactComboActionProbability:
    """One canonical combo and an exact shove or call probability."""

    combo: str
    probability: Fraction


@dataclass(frozen=True)
class ExactBehaviourProfile:
    """Complete canonical behavioural profile for both seats."""

    sb_shove: tuple[ExactComboActionProbability, ...]
    bb_call: tuple[ExactComboActionProbability, ...]
    content_identity: str


@dataclass(frozen=True)
class ExactActionValue:
    """One canonical action and its exact conditional value, when reached."""

    action: str
    value: Fraction | None


@dataclass(frozen=True)
class ExactStrategyRow:
    """One combo's reach, action values, correspondences, and root gain."""

    seat: str
    combo: str
    compatible_probability: Fraction
    information_reach_probability: Fraction
    active_action: str
    active_probability: Fraction
    fold_probability: Fraction
    action_values: tuple[ExactActionValue, ...]
    exact_best_actions: tuple[str, ...]
    display_best_actions: tuple[str, ...]
    unilateral_gain: Fraction


@dataclass(frozen=True)
class ExactGainSnapshot:
    """Exact profile value, unilateral gains, and value enclosure."""

    profile_value: Fraction
    sb_best_response_value: Fraction
    bb_best_response_sb_value: Fraction
    g_sb: Fraction
    g_bb: Fraction
    nash_conv: Fraction
    max_unilateral_gain: Fraction
    value_lower: Fraction
    value_upper: Fraction


@dataclass(frozen=True)
class SimplexTracePoint:
    """One bounded deterministic exact-simplex pivot trace point."""

    problem: str
    phase: int
    pivot_index: int
    entering_variable: str
    leaving_variable: str
    objective: Fraction


@dataclass(frozen=True)
class SimplexRunSummary:
    """Bounded exact-simplex run summary for one compact LP."""

    problem: str
    algorithm_id: str
    objective: Fraction
    phase_one_pivots: int
    phase_two_pivots: int
    selected_basis: tuple[str, ...]
    primal_feasible: bool
    dual_feasible: bool
    trace: tuple[SimplexTracePoint, ...]
    trace_identity: str
    trace_truncated: bool


@dataclass(frozen=True)
class RationalVerificationWitness:
    """Content-bound exact re-verification record for the declared game."""

    verifier_id: str
    game_id: str
    payoff_identity: str
    profile_identity: str
    verification_identity: str
    lower_objective: Fraction
    upper_objective: Fraction
    claim_kind: StrategyClaimKind
    claim_epsilon: Fraction
    numeric_error_bound: Fraction
    primal_feasible: bool
    dual_feasible: bool
    zero_objective_gap: bool
    gains: ExactGainSnapshot
    sb_rows: tuple[ExactStrategyRow, ...]
    bb_rows: tuple[ExactStrategyRow, ...]


@dataclass(frozen=True)
class ReferenceOracleComparison:
    """Tiny normal-form reference comparison and its bounded counts."""

    oracle_id: str
    sb_pure_plan_count: int
    bb_pure_plan_count: int
    normal_form_cell_count: int
    support_system_count: int
    oracle_lower_value: Fraction
    oracle_upper_value: Fraction
    value_matches: bool
    selected_profile_value_matches: bool
    selected_profile_gains_match: bool
    tie_classification_matches: bool
    off_path_classification_matches: bool
    comparison_identity: str


@dataclass(frozen=True)
class Phase1FloatDiagnostic:
    """Non-authoritative comparison with the phase-1 binary64 public path."""

    phase1_status: AiofStatus
    comparison_tolerance: float
    phase1_profile_value: float | None
    phase1_g_sb: float | None
    phase1_g_bb: float | None
    profile_value_difference: float | None
    g_sb_difference: float | None
    g_bb_difference: float | None
    tie_classification_matches: bool | None
    off_path_classification_matches: bool | None
    within_display_bound: bool | None
    error_message: str | None


@dataclass(frozen=True)
class RationalStrategyRequest:
    """Exact-only bounded primary request."""

    sb_range: RangeSpec
    bb_range: RangeSpec
    dead_cards: tuple[str, ...]
    equity_algorithm: EquityAlgorithm
    game: HeadsUpChipEvGame
    limits: AiofStrategyLimits
    algorithm: AiofStrategyAlgorithm = AiofStrategyAlgorithm.COMPACT_RATIONAL_LP
    claim_epsilon: Fraction = Fraction(0)
    display_tie_tolerance: Fraction = Fraction(0)
    requested_trace_points: int = 0
    run_reference_oracle: bool = False
    run_phase1_float_diagnostic: bool = False
    seed: int | None = None
    samples: int | None = None


@dataclass(frozen=True)
class RationalStrategyResult:
    """Successful deterministic primary payload after independent checks."""

    algorithm: AiofStrategyAlgorithm
    game_id: str
    verifier_id: str
    prepared_ranges_identity: str
    payoff_identity: str
    semantic_identity: str
    input_identity: str
    runtime_identity: str
    run_identity: str
    payoff_cell_count: int
    exact_board_evaluations: int
    profile: ExactBehaviourProfile
    sb_lp: SimplexRunSummary
    bb_lp: SimplexRunSummary
    witness: RationalVerificationWitness
    oracle_comparison: ReferenceOracleComparison | None
    phase1_float_diagnostic: Phase1FloatDiagnostic | None


@dataclass(frozen=True)
class RationalStrategyRunResult:
    """Fail-closed primary wrapper with success/failure payload invariants."""

    status: AiofStrategyStatus
    strategy_result: RationalStrategyResult | None
    error: StrategyError | None


@dataclass(frozen=True)
class AlternatingBrDiagnosticRequest:
    """Exact-only request for the separate bounded BR diagnostic."""

    sb_range: RangeSpec
    bb_range: RangeSpec
    dead_cards: tuple[str, ...]
    equity_algorithm: EquityAlgorithm
    game: HeadsUpChipEvGame
    limits: AiofStrategyLimits
    update_mode: HeuristicUpdateMode
    damping: Fraction
    max_iterations: int
    display_tie_tolerance: Fraction = Fraction(0)
    requested_trace_points: int = 0
    seed: int | None = None
    samples: int | None = None


@dataclass(frozen=True)
class HeuristicTracePoint:
    """Post-update current and arithmetic-average gain snapshot."""

    iteration: int
    state_identity: str
    current_profile_value: Fraction
    current_g_sb: Fraction
    current_g_bb: Fraction
    current_nash_conv: Fraction
    current_max_unilateral_gain: Fraction
    average_profile_value: Fraction
    average_g_sb: Fraction
    average_g_bb: Fraction
    average_nash_conv: Fraction
    average_max_unilateral_gain: Fraction


@dataclass(frozen=True)
class AlternatingBrDiagnostic:
    """Bounded BR-dynamics payload with no primary-solver claim."""

    status: HeuristicDiagnosticStatus
    algorithm_id: str
    game_id: str
    initialization_id: str
    update_mode: HeuristicUpdateMode
    damping: Fraction
    semantic_identity: str
    input_identity: str
    runtime_identity: str
    run_identity: str
    iterations_completed: int
    repeated_state_first_seen_iteration: int | None
    cycle_length: int | None
    current_state_identity: str
    current_profile: ExactBehaviourProfile
    arithmetic_average_profile: ExactBehaviourProfile
    current_gains: ExactGainSnapshot
    average_gains: ExactGainSnapshot
    trace: tuple[HeuristicTracePoint, ...]
    trace_identity: str
    trace_truncated: bool


@dataclass(frozen=True)
class AlternatingBrDiagnosticRunResult:
    """Fail-closed wrapper for the separate diagnostic entry point."""

    status: AiofStrategyStatus
    diagnostic: AlternatingBrDiagnostic | None
    error: StrategyError | None


class _StrategyFailure(Exception):
    def __init__(
        self,
        status: AiofStrategyStatus,
        message: str,
        phase: str,
        *,
        upstream_status: AiofStatus | None = None,
        completed_payoff_cells: int = 0,
        completed_pivots: int = 0,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.phase = phase
        self.upstream_status = upstream_status
        self.completed_payoff_cells = completed_payoff_cells
        self.completed_pivots = completed_pivots


@dataclass(frozen=True)
class _RationalGame:
    sb_combos: tuple[str, ...]
    bb_combos: tuple[str, ...]
    sb_cards: tuple[tuple[int, int], ...]
    bb_cards: tuple[tuple[int, int], ...]
    probabilities: tuple[tuple[Fraction | None, ...], ...]
    showdown: tuple[tuple[Fraction | None, ...], ...]
    counts: tuple[tuple[OutcomeCounts | None, ...], ...]
    p_h: tuple[Fraction, ...]
    p_v: tuple[Fraction, ...]
    f: Fraction
    g: Fraction
    effective: Fraction
    prepared_identity: str
    payoff_identity: str
    payoff_cells: int
    board_evaluations: int


def _canonical(value: Any) -> Any:
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if hasattr(value, "__dataclass_fields__"):
        return {name: _canonical(getattr(value, name)) for name in value.__dataclass_fields__}
    return value


def _identity(value: Any) -> str:
    data = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _runtime_identity() -> str:
    return _identity(
        {
            "implementation": "repeated-poker-analysis-aiof-strategy-v1",
            "python_implementation": sys.implementation.name,
            "python_version": platform.python_version(),
        }
    )


def _plain_int(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise _StrategyFailure(
            AiofStrategyStatus.INVALID_INPUT,
            f"{name} must be an integer in [{minimum}, {maximum}]",
            "validation",
        )
    return value


def _exact_nonnegative(value: object, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, Fraction) or value < 0:
        raise _StrategyFailure(
            AiofStrategyStatus.INVALID_INPUT,
            f"{name} must be a non-negative Fraction",
            "validation",
        )
    return value


def _check_fraction(value: Fraction, limits: AiofStrategyLimits, phase: str) -> Fraction:
    if (
        abs(value.numerator).bit_length() > limits.max_exact_rational_bits
        or value.denominator.bit_length() > limits.max_exact_rational_bits
    ):
        raise _StrategyFailure(
            AiofStrategyStatus.EXACT_ARITHMETIC_CAP_EXCEEDED,
            "exact rational bit cap exceeded",
            phase,
        )
    return value


def _f(value: float, limits: AiofStrategyLimits, phase: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _StrategyFailure(AiofStrategyStatus.INVALID_INPUT, "chip input must be numeric", phase)
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise _StrategyFailure(AiofStrategyStatus.INVALID_INPUT, "chip input is not binary64", phase) from exc
    if not math.isfinite(number):
        raise _StrategyFailure(AiofStrategyStatus.INVALID_INPUT, "chip input must be finite", phase)
    numerator, denominator = number.as_integer_ratio()
    return _check_fraction(Fraction(numerator, denominator), limits, phase)


_LIMIT_CEILINGS = {
    name: field.default
    for name, field in AiofStrategyLimits.__dataclass_fields__.items()
}


def _validate_limits(limits: object) -> AiofStrategyLimits:
    if not isinstance(limits, AiofStrategyLimits):
        raise _StrategyFailure(AiofStrategyStatus.INVALID_INPUT, "limits must be AiofStrategyLimits", "validation")
    for name, ceiling in _LIMIT_CEILINGS.items():
        value = getattr(limits, name)
        minimum = 0 if name == "max_trace_points" else 1
        _plain_int(value, name, minimum, ceiling)
    return limits


def _validate_common(request: object, *, heuristic: bool) -> tuple[AiofStrategyLimits, int]:
    expected = AlternatingBrDiagnosticRequest if heuristic else RationalStrategyRequest
    if not isinstance(request, expected):
        raise _StrategyFailure(AiofStrategyStatus.INVALID_INPUT, f"request must be {expected.__name__}", "validation")
    limits = _validate_limits(request.limits)
    if not isinstance(request.sb_range, RangeSpec) or not isinstance(request.bb_range, RangeSpec):
        raise _StrategyFailure(AiofStrategyStatus.INVALID_INPUT, "ranges must be RangeSpec", "validation")
    if not isinstance(request.dead_cards, tuple):
        raise _StrategyFailure(AiofStrategyStatus.INVALID_INPUT, "dead_cards must be a tuple", "validation")
    if request.equity_algorithm is not EquityAlgorithm.EXACT_EXHAUSTIVE or request.seed is not None or request.samples is not None:
        raise _StrategyFailure(AiofStrategyStatus.EXACT_PAYOFF_REQUIRED, "exact payoff with no seed or samples is required", "validation")
    if isinstance(request.requested_trace_points, bool) or not isinstance(request.requested_trace_points, int) or request.requested_trace_points < 0:
        raise _StrategyFailure(AiofStrategyStatus.INVALID_INPUT, "requested_trace_points must be a non-negative integer", "validation")
    if request.requested_trace_points > limits.max_trace_points:
        raise _StrategyFailure(AiofStrategyStatus.CAP_EXCEEDED, "trace-point cap exceeded", "validation")
    trace = request.requested_trace_points
    display_tolerance = _exact_nonnegative(request.display_tie_tolerance, "display_tie_tolerance")
    _check_fraction(display_tolerance, limits, "validation")
    if heuristic:
        if not isinstance(request.update_mode, HeuristicUpdateMode):
            raise _StrategyFailure(AiofStrategyStatus.INVALID_INPUT, "invalid update_mode", "validation")
        if isinstance(request.damping, bool) or not isinstance(request.damping, Fraction) or not 0 < request.damping <= 1:
            raise _StrategyFailure(AiofStrategyStatus.INVALID_INPUT, "damping must be a Fraction in (0, 1]", "validation")
        _check_fraction(request.damping, limits, "validation")
        _plain_int(request.max_iterations, "max_iterations", 1, limits.max_heuristic_iterations)
    else:
        if not isinstance(request.algorithm, AiofStrategyAlgorithm):
            raise _StrategyFailure(AiofStrategyStatus.INVALID_INPUT, "invalid strategy algorithm", "validation")
        claim_epsilon = _exact_nonnegative(request.claim_epsilon, "claim_epsilon")
        _check_fraction(claim_epsilon, limits, "validation")
        if not isinstance(request.run_reference_oracle, bool) or not isinstance(request.run_phase1_float_diagnostic, bool):
            raise _StrategyFailure(AiofStrategyStatus.INVALID_INPUT, "diagnostic flags must be bool", "validation")
    return limits, trace


def _validate_game(game: object, limits: AiofStrategyLimits) -> tuple[Fraction, Fraction, Fraction]:
    if not isinstance(game, HeadsUpChipEvGame):
        raise _StrategyFailure(AiofStrategyStatus.INVALID_INPUT, "game must be HeadsUpChipEvGame", "game")
    sb = _f(game.starting_stack_sb, limits, "game")
    bb = _f(game.starting_stack_bb, limits, "game")
    small = _f(game.small_blind, limits, "game")
    big = _f(game.big_blind, limits, "game")
    ante = _f(game.ante, limits, "game")
    fee = _f(game.fee, limits, "game")
    dead = _f(game.third_party_dead_money, limits, "game")
    if not isinstance(game.side_pot, bool):
        raise _StrategyFailure(AiofStrategyStatus.INVALID_INPUT, "side_pot must be bool", "game")
    if fee or dead or game.side_pot:
        raise _StrategyFailure(AiofStrategyStatus.UNSUPPORTED_MODEL, "fee, third-party money, and side pots are unsupported", "game")
    if sb <= 0 or bb <= 0 or small <= 0 or big < small or ante < 0:
        raise _StrategyFailure(AiofStrategyStatus.INVALID_INPUT, "invalid stack, blind, or ante", "game")
    sb_post = _check_fraction(small + ante, limits, "game")
    bb_post = _check_fraction(big + ante, limits, "game")
    if sb < sb_post or bb < bb_post:
        raise _StrategyFailure(AiofStrategyStatus.UNSUPPORTED_MODEL, "mandatory post is not covered", "game")
    return -sb_post, bb_post, min(sb, bb)


def _phase1_limits(limits: AiofStrategyLimits) -> AiofLimits:
    return AiofLimits(
        max_range_entries_per_side=min(1_326, limits.max_solver_combos_per_side),
        max_exact_combos_per_side=min(64, limits.max_solver_combos_per_side),
        max_compatible_combo_pairs=min(4_096, limits.max_payoff_cells),
        max_dead_cards=43,
        max_exact_board_evaluations=min(10_000_000, limits.max_exact_board_evaluations),
        max_monte_carlo_samples=10_000_000,
        max_sampling_attempts=None,
        max_cache_entries=min(100_000, limits.max_payoff_cells),
        max_trace_points=0,
    )


def _disjoint(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] not in right and left[1] not in right


def _build_rational_game(
    request: RationalStrategyRequest | AlternatingBrDiagnosticRequest,
    limits: AiofStrategyLimits,
    *,
    extra_float_pass: bool,
) -> _RationalGame:
    f_value, g_value, effective = _validate_game(request.game, limits)
    phase1_limits = _phase1_limits(limits)
    try:
        prepared = prepare_compatible_ranges(
            request.sb_range, request.bb_range, request.dead_cards, phase1_limits
        )
    except AiofContractError as exc:
        raise _StrategyFailure(
            AiofStrategyStatus.CAP_EXCEEDED
            if exc.status is AiofStatus.CAP_EXCEEDED
            else AiofStrategyStatus.INPUT_PREPARATION_FAILED,
            str(exc) or exc.status.value,
            "input_preparation",
            upstream_status=exc.status,
        ) from exc
    h_count = len(prepared.sb_range.combos)
    v_count = len(prepared.bb_range.combos)
    if h_count > limits.max_solver_combos_per_side or v_count > limits.max_solver_combos_per_side:
        raise _StrategyFailure(AiofStrategyStatus.CAP_EXCEEDED, "solver combo cap exceeded", "payoff_preflight")
    pair_count = sum(
        1
        for h in prepared.sb_range.combos
        for v in prepared.bb_range.combos
        if _disjoint(h.card_ids, v.card_ids)
    )
    if pair_count != prepared.compatible_pair_count:
        raise _StrategyFailure(AiofStrategyStatus.NUMERIC_FAILURE, "compatible pair count mismatch", "payoff_preflight")
    if pair_count > limits.max_payoff_cells:
        raise _StrategyFailure(AiofStrategyStatus.CAP_EXCEEDED, "payoff cell cap exceeded", "payoff_preflight")
    remaining = 52 - len(prepared.dead_card_ids) - 4
    if remaining < 5:
        raise _StrategyFailure(AiofStrategyStatus.INVALID_INPUT, "fewer than five board cards remain", "payoff_preflight")
    boards_per_cell = math.comb(remaining, 5)
    primary_boards = pair_count * boards_per_cell
    total_boards = primary_boards * (2 if extra_float_pass else 1)
    if total_boards > limits.max_exact_board_evaluations:
        raise _StrategyFailure(AiofStrategyStatus.CAP_EXCEEDED, "global exact board cap exceeded", "payoff_preflight")
    # Both LPs have |H|+|V| original variables and constraints.
    original_variables = h_count + v_count
    original_constraints = h_count + v_count
    if original_variables > limits.max_lp_variables_per_problem or original_constraints > limits.max_lp_constraints_per_problem:
        raise _StrategyFailure(AiofStrategyStatus.CAP_EXCEEDED, "LP dimension cap exceeded", "payoff_preflight")
    # Each problem has one <= row per own action and one >= row per opponent action.
    augmented_columns = original_variables + original_constraints + max(h_count, v_count)
    if (original_constraints + 1) * (augmented_columns + 1) > limits.max_tableau_cells:
        raise _StrategyFailure(AiofStrategyStatus.CAP_EXCEEDED, "tableau cap exceeded", "payoff_preflight")

    raw_h = tuple(_f(combo.raw_mass, limits, "rational_lift") for combo in prepared.sb_range.combos)
    raw_v = tuple(_f(combo.raw_mass, limits, "rational_lift") for combo in prepared.bb_range.combos)
    raw_products: list[Fraction] = []
    for i, h in enumerate(prepared.sb_range.combos):
        for j, v in enumerate(prepared.bb_range.combos):
            if _disjoint(h.card_ids, v.card_ids):
                raw_products.append(_check_fraction(raw_h[i] * raw_v[j], limits, "rational_lift"))
    joint = _check_fraction(sum(raw_products, Fraction(0)), limits, "rational_lift")
    if joint <= 0:
        raise _StrategyFailure(AiofStrategyStatus.NUMERIC_FAILURE, "exact lifted joint mass is not positive", "rational_lift")

    probabilities: list[list[Fraction | None]] = [[None] * v_count for _ in range(h_count)]
    showdown: list[list[Fraction | None]] = [[None] * v_count for _ in range(h_count)]
    counts_grid: list[list[OutcomeCounts | None]] = [[None] * v_count for _ in range(h_count)]
    cache: dict[tuple[Any, ...], OutcomeCounts] = {}
    completed = 0
    for i, h in enumerate(prepared.sb_range.combos):
        for j, v in enumerate(prepared.bb_range.combos):
            if not _disjoint(h.card_ids, v.card_ids):
                continue
            key = (
                prepared.content_identity,
                h.combo,
                v.combo,
                prepared.dead_cards,
                EXACT_ID,
                EVALUATOR_ID,
            )
            if key in cache:
                counts = cache[key]
            else:
                singleton_sb = RangeSpec((RangeEntry(h.combo, 1.0, WeightBasis.EXACT_COMBO_MASS),))
                singleton_bb = RangeSpec((RangeEntry(v.combo, 1.0, WeightBasis.EXACT_COMBO_MASS),))
                result = calculate_preflop_equity(
                    EquityRequest(
                        singleton_sb,
                        singleton_bb,
                        prepared.dead_cards,
                        EquityAlgorithm.EXACT_EXHAUSTIVE,
                        AiofLimits(
                            max_range_entries_per_side=1,
                            max_exact_combos_per_side=1,
                            max_compatible_combo_pairs=1,
                            max_dead_cards=43,
                            max_exact_board_evaluations=boards_per_cell,
                            max_monte_carlo_samples=10_000_000,
                            max_sampling_attempts=None,
                            max_cache_entries=1,
                            max_trace_points=0,
                        ),
                        0,
                        None,
                        None,
                    )
                )
                if result.status is not AiofStatus.SUCCESS or result.estimate is None:
                    raise _StrategyFailure(
                        AiofStrategyStatus.PAYOFF_CONSTRUCTION_FAILED,
                        result.error_message or "singleton equity failed",
                        "payoff_construction",
                        upstream_status=result.status,
                        completed_payoff_cells=completed,
                    )
                estimate = result.estimate
                counts = estimate.unweighted_counts
                if (
                    counts.trials <= 0
                    or counts.wins + counts.losses + counts.ties != counts.trials
                    or estimate.board_evaluations != boards_per_cell
                ):
                    raise _StrategyFailure(
                        AiofStrategyStatus.PAYOFF_CONSTRUCTION_FAILED,
                        "singleton equity counts mismatch",
                        "payoff_construction",
                        upstream_status=result.status,
                        completed_payoff_cells=completed,
                    )
                cache[key] = counts
            completed += 1
            probability = _check_fraction(raw_h[i] * raw_v[j] / joint, limits, "rational_lift")
            c_value = _check_fraction(
                effective * Fraction(counts.wins - counts.losses, counts.trials),
                limits,
                "rational_lift",
            )
            probabilities[i][j] = probability
            showdown[i][j] = c_value
            counts_grid[i][j] = counts
    if len(cache) != pair_count or completed != pair_count:
        raise _StrategyFailure(AiofStrategyStatus.NUMERIC_FAILURE, "run-local payoff cache mismatch", "payoff_construction", completed_payoff_cells=completed)
    p_h = tuple(_check_fraction(sum((p for p in row if p is not None), Fraction(0)), limits, "rational_lift") for row in probabilities)
    p_v = tuple(
        _check_fraction(sum((probabilities[i][j] or Fraction(0) for i in range(h_count)), Fraction(0)), limits, "rational_lift")
        for j in range(v_count)
    )
    if any(value <= 0 for value in p_h + p_v) or sum(p_h, Fraction(0)) != 1 or sum(p_v, Fraction(0)) != 1:
        raise _StrategyFailure(AiofStrategyStatus.NUMERIC_FAILURE, "exact marginals are invalid", "rational_lift")
    payoff_identity = _identity(
        {
            "game_id": GAME_ID,
            "prepared": prepared.content_identity,
            "raw_h": raw_h,
            "raw_v": raw_v,
            "probabilities": tuple(tuple(row) for row in probabilities),
            "counts": tuple(tuple(row) for row in counts_grid),
            "f": f_value,
            "g": g_value,
            "effective": effective,
        }
    )
    return _RationalGame(
        tuple(combo.combo for combo in prepared.sb_range.combos),
        tuple(combo.combo for combo in prepared.bb_range.combos),
        tuple(combo.card_ids for combo in prepared.sb_range.combos),
        tuple(combo.card_ids for combo in prepared.bb_range.combos),
        tuple(tuple(row) for row in probabilities),
        tuple(tuple(row) for row in showdown),
        tuple(tuple(row) for row in counts_grid),
        p_h,
        p_v,
        f_value,
        g_value,
        effective,
        prepared.content_identity,
        payoff_identity,
        pair_count,
        primary_boards,
    )


@dataclass(frozen=True)
class _Constraint:
    coefficients: tuple[Fraction, ...]
    sense: str
    rhs: Fraction


@dataclass(frozen=True)
class _LinearProgram:
    problem: str
    variable_names: tuple[str, ...]
    objective: tuple[Fraction, ...]
    objective_constant: Fraction
    constraints: tuple[_Constraint, ...]


@dataclass
class _Tableau:
    names: list[str]
    rows: list[list[Fraction]]
    rhs: list[Fraction]
    basis: list[int]
    artificial: set[int]
    row_indices: list[int]


@dataclass(frozen=True)
class _LpSolution:
    summary: SimplexRunSummary
    values: dict[str, Fraction]
    basis: tuple[str, ...]
    row_indices: tuple[int, ...]


def _game_coefficients(game: _RationalGame, limits: AiofStrategyLimits) -> tuple[tuple[Fraction, ...], tuple[tuple[Fraction | None, ...], ...]]:
    a = tuple(
        _check_fraction(game.p_h[i] * (game.g - game.f), limits, "lp_construction")
        for i in range(len(game.sb_combos))
    )
    b: list[list[Fraction | None]] = []
    for i in range(len(game.sb_combos)):
        row: list[Fraction | None] = []
        for j in range(len(game.bb_combos)):
            probability = game.probabilities[i][j]
            showdown = game.showdown[i][j]
            row.append(
                None
                if probability is None or showdown is None
                else _check_fraction(probability * (showdown - game.g), limits, "lp_construction")
            )
        b.append(row)
    return a, tuple(tuple(row) for row in b)


def _make_lp(game: _RationalGame, limits: AiofStrategyLimits, problem: str) -> _LinearProgram:
    h_count = len(game.sb_combos)
    v_count = len(game.bb_combos)
    a, b = _game_coefficients(game, limits)
    if problem == "SB":
        names = tuple(f"x[{combo}]" for combo in game.sb_combos) + tuple(
            f"q[{combo}]" for combo in game.bb_combos
        )
        objective = a + (Fraction(-1),) * v_count
        constraints: list[_Constraint] = []
        for i in range(h_count):
            coefficients = [Fraction(0)] * len(names)
            coefficients[i] = Fraction(1)
            constraints.append(_Constraint(tuple(coefficients), "<=", Fraction(1)))
        for j in range(v_count):
            coefficients = [Fraction(0)] * len(names)
            for i in range(h_count):
                coefficients[i] = b[i][j] or Fraction(0)
            coefficients[h_count + j] = Fraction(1)
            constraints.append(_Constraint(tuple(coefficients), ">=", Fraction(0)))
        return _LinearProgram(problem, names, objective, game.f, tuple(constraints))
    names = tuple(f"y[{combo}]" for combo in game.bb_combos) + tuple(
        f"r[{combo}]" for combo in game.sb_combos
    )
    objective = (Fraction(0),) * v_count + (Fraction(-1),) * h_count
    constraints = []
    for j in range(v_count):
        coefficients = [Fraction(0)] * len(names)
        coefficients[j] = Fraction(1)
        constraints.append(_Constraint(tuple(coefficients), "<=", Fraction(1)))
    for i in range(h_count):
        coefficients = [Fraction(0)] * len(names)
        for j in range(v_count):
            coefficients[j] = -(b[i][j] or Fraction(0))
        coefficients[v_count + i] = Fraction(1)
        constraints.append(_Constraint(tuple(coefficients), ">=", a[i]))
    return _LinearProgram(problem, names, objective, -game.f, tuple(constraints))


def _normalized_constraint(constraint: _Constraint) -> _Constraint:
    if constraint.rhs >= 0:
        return constraint
    flipped = {"<=": ">=", ">=": "<=", "=": "="}[constraint.sense]
    return _Constraint(tuple(-value for value in constraint.coefficients), flipped, -constraint.rhs)


def _tableau_for(lp: _LinearProgram, limits: AiofStrategyLimits) -> _Tableau:
    normalized = tuple(_normalized_constraint(item) for item in lp.constraints)
    names = list(lp.variable_names)
    additions: list[tuple[int, str, str]] = []
    for row_index, row in enumerate(normalized):
        if row.sense == "<=":
            additions.append((row_index, "slack", f"s[{row_index}]"))
        elif row.sense == ">=":
            additions.append((row_index, "surplus", f"u[{row_index}]"))
            additions.append((row_index, "artificial", f"a[{row_index}]"))
        else:
            additions.append((row_index, "artificial", f"a[{row_index}]"))
    names.extend(item[2] for item in additions)
    if (len(normalized) + 1) * (len(names) + 1) > limits.max_tableau_cells:
        raise _StrategyFailure(AiofStrategyStatus.CAP_EXCEEDED, "tableau cap exceeded", "simplex_preflight")
    rows: list[list[Fraction]] = []
    basis: list[int] = []
    artificial: set[int] = set()
    for row_index, constraint in enumerate(normalized):
        values = list(constraint.coefficients) + [Fraction(0)] * len(additions)
        for offset, (target, kind, _) in enumerate(additions):
            if target != row_index:
                continue
            column = len(lp.variable_names) + offset
            if kind == "slack":
                values[column] = Fraction(1)
                basis.append(column)
            elif kind == "surplus":
                values[column] = Fraction(-1)
            else:
                values[column] = Fraction(1)
                artificial.add(column)
                basis.append(column)
        rows.append(values)
    return _Tableau(names, rows, [item.rhs for item in normalized], basis, artificial, list(range(len(normalized))))


def _objective_value(tableau: _Tableau, costs: Sequence[Fraction], constant: Fraction, limits: AiofStrategyLimits) -> Fraction:
    return _check_fraction(
        constant + sum((costs[basic] * rhs for basic, rhs in zip(tableau.basis, tableau.rhs)), Fraction(0)),
        limits,
        "simplex",
    )


def _reduced_costs(tableau: _Tableau, costs: Sequence[Fraction], limits: AiofStrategyLimits) -> list[Fraction]:
    basic_set = set(tableau.basis)
    result: list[Fraction] = []
    for column in range(len(tableau.names)):
        if column in basic_set:
            result.append(Fraction(0))
            continue
        value = costs[column] - sum(
            (costs[basic] * tableau.rows[row][column] for row, basic in enumerate(tableau.basis)),
            Fraction(0),
        )
        result.append(_check_fraction(value, limits, "simplex"))
    return result


def _pivot(tableau: _Tableau, row: int, column: int, limits: AiofStrategyLimits) -> None:
    pivot = tableau.rows[row][column]
    if pivot == 0:
        raise _StrategyFailure(AiofStrategyStatus.SOLVER_CONTRACT_FAILURE, "zero simplex pivot", "simplex")
    new_pivot_row = [_check_fraction(value / pivot, limits, "simplex") for value in tableau.rows[row]]
    new_rhs = _check_fraction(tableau.rhs[row] / pivot, limits, "simplex")
    new_rows: list[list[Fraction]] = []
    new_right: list[Fraction] = []
    for index, old in enumerate(tableau.rows):
        if index == row:
            new_rows.append(new_pivot_row)
            new_right.append(new_rhs)
            continue
        factor = old[column]
        replacement = [
            _check_fraction(value - factor * pivot_value, limits, "simplex")
            for value, pivot_value in zip(old, new_pivot_row)
        ]
        new_rows.append(replacement)
        new_right.append(_check_fraction(tableau.rhs[index] - factor * new_rhs, limits, "simplex"))
    tableau.rows = new_rows
    tableau.rhs = new_right
    tableau.basis[row] = column


def _simplex_phase(
    tableau: _Tableau,
    costs: Sequence[Fraction],
    constant: Fraction,
    phase: int,
    limits: AiofStrategyLimits,
    problem: str,
    trace: list[SimplexTracePoint],
    start_pivots: int,
    trace_capacity: int,
) -> int:
    pivots = 0
    while True:
        reduced = _reduced_costs(tableau, costs, limits)
        entering_candidates = [
            column
            for column, value in enumerate(reduced)
            if value > 0 and column not in tableau.basis
        ]
        if not entering_candidates:
            return pivots
        entering = min(entering_candidates)
        candidates: list[tuple[Fraction, int, int]] = []
        for row, coefficient in enumerate(item[entering] for item in tableau.rows):
            if coefficient > 0:
                candidates.append((tableau.rhs[row] / coefficient, tableau.basis[row], row))
        if not candidates:
            raise _StrategyFailure(AiofStrategyStatus.SOLVER_CONTRACT_FAILURE, "declared LP is unbounded", "simplex", completed_pivots=start_pivots + pivots)
        _, _, leaving_row = min(candidates)
        if start_pivots + pivots >= limits.max_simplex_pivots:
            raise _StrategyFailure(AiofStrategyStatus.SOLVER_LIMIT_REACHED, "simplex pivot cap reached", "simplex", completed_pivots=start_pivots + pivots)
        leaving_name = tableau.names[tableau.basis[leaving_row]]
        _pivot(tableau, leaving_row, entering, limits)
        pivots += 1
        if len(trace) < trace_capacity:
            trace.append(
                SimplexTracePoint(
                    problem,
                    phase,
                    start_pivots + pivots,
                    tableau.names[entering],
                    leaving_name,
                    _objective_value(tableau, costs, constant, limits),
                )
            )


def _solve_lp(lp: _LinearProgram, limits: AiofStrategyLimits, requested_trace: int) -> _LpSolution:
    tableau = _tableau_for(lp, limits)
    trace_all: list[SimplexTracePoint] = []
    phase_one_costs = [Fraction(-1) if index in tableau.artificial else Fraction(0) for index in range(len(tableau.names))]
    phase_one = _simplex_phase(
        tableau, phase_one_costs, Fraction(0), 1, limits, lp.problem, trace_all, 0, requested_trace
    )
    phase_one_value = _objective_value(tableau, phase_one_costs, Fraction(0), limits)
    if phase_one_value != 0:
        raise _StrategyFailure(AiofStrategyStatus.SOLVER_CONTRACT_FAILURE, "Phase I optimum is not zero", "simplex", completed_pivots=phase_one)
    # Pivot artificial basics out, or delete exactly redundant zero rows.
    row = 0
    removal_pivots = 0
    while row < len(tableau.rows):
        if tableau.basis[row] not in tableau.artificial:
            row += 1
            continue
        candidates = [
            column
            for column in range(len(tableau.names))
            if column not in tableau.artificial
            and column not in tableau.basis
            and tableau.rows[row][column] != 0
        ]
        if candidates:
            if phase_one + removal_pivots >= limits.max_simplex_pivots:
                raise _StrategyFailure(AiofStrategyStatus.SOLVER_LIMIT_REACHED, "simplex pivot cap reached", "simplex", completed_pivots=phase_one + removal_pivots)
            entering = min(candidates)
            leaving_name = tableau.names[tableau.basis[row]]
            _pivot(tableau, row, entering, limits)
            removal_pivots += 1
            if len(trace_all) < requested_trace:
                trace_all.append(
                    SimplexTracePoint(lp.problem, 1, phase_one + removal_pivots, tableau.names[entering], leaving_name, Fraction(0))
                )
            row += 1
        else:
            if tableau.rhs[row] != 0:
                raise _StrategyFailure(AiofStrategyStatus.SOLVER_CONTRACT_FAILURE, "nonzero redundant artificial row", "simplex", completed_pivots=phase_one + removal_pivots)
            del tableau.rows[row]
            del tableau.rhs[row]
            del tableau.basis[row]
            del tableau.row_indices[row]
    keep = [column for column in range(len(tableau.names)) if column not in tableau.artificial]
    remap = {old: new for new, old in enumerate(keep)}
    tableau.names = [tableau.names[column] for column in keep]
    tableau.rows = [[row[column] for column in keep] for row in tableau.rows]
    tableau.basis = [remap[column] for column in tableau.basis]
    tableau.artificial = set()
    costs = [lp.objective[lp.variable_names.index(name)] if name in lp.variable_names else Fraction(0) for name in tableau.names]
    phase_two = _simplex_phase(
        tableau,
        costs,
        lp.objective_constant,
        2,
        limits,
        lp.problem,
        trace_all,
        phase_one + removal_pivots,
        requested_trace,
    )
    objective = _objective_value(tableau, costs, lp.objective_constant, limits)
    values = {name: Fraction(0) for name in tableau.names}
    for row_index, column in enumerate(tableau.basis):
        values[tableau.names[column]] = tableau.rhs[row_index]
    stored_trace = tuple(trace_all[:requested_trace])
    summary = SimplexRunSummary(
        lp.problem,
        SIMPLEX_ID,
        objective,
        phase_one + removal_pivots,
        phase_two,
        tuple(tableau.names[column] for column in tableau.basis),
        False,
        False,
        stored_trace,
        _identity(stored_trace),
        phase_one + removal_pivots + phase_two > requested_trace,
    )
    return _LpSolution(summary, values, summary.selected_basis, tuple(tableau.row_indices))


def _standard_nonart(lp: _LinearProgram) -> tuple[tuple[str, ...], tuple[tuple[Fraction, ...], ...], tuple[Fraction, ...]]:
    normalized = tuple(_normalized_constraint(item) for item in lp.constraints)
    additions: list[tuple[int, str, str]] = []
    for row_index, row in enumerate(normalized):
        if row.sense == "<=":
            additions.append((row_index, "slack", f"s[{row_index}]"))
        elif row.sense == ">=":
            additions.append((row_index, "surplus", f"u[{row_index}]"))
    names = lp.variable_names + tuple(item[2] for item in additions)
    matrix: list[tuple[Fraction, ...]] = []
    for row_index, constraint in enumerate(normalized):
        values = list(constraint.coefficients) + [Fraction(0)] * len(additions)
        for offset, (target, kind, _) in enumerate(additions):
            if target == row_index:
                values[len(lp.variable_names) + offset] = Fraction(1 if kind == "slack" else -1)
        matrix.append(tuple(values))
    return names, tuple(matrix), tuple(item.rhs for item in normalized)


def _gaussian_solve(
    matrix: Sequence[Sequence[Fraction]],
    rhs: Sequence[Fraction],
    limits: AiofStrategyLimits,
    phase: str,
) -> tuple[Fraction, ...]:
    size = len(rhs)
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise _StrategyFailure(AiofStrategyStatus.VERIFICATION_FAILED, "Gaussian system is not square", phase)
    augmented = [list(row) + [rhs[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot_rows = [row for row in range(column, size) if augmented[row][column] != 0]
        if not pivot_rows:
            raise _StrategyFailure(AiofStrategyStatus.VERIFICATION_FAILED, "singular Gaussian system", phase)
        pivot_row = min(pivot_rows)
        augmented[column], augmented[pivot_row] = augmented[pivot_row], augmented[column]
        pivot = augmented[column][column]
        augmented[column] = [_check_fraction(value / pivot, limits, phase) for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    _check_fraction(value - factor * pivot_value, limits, phase)
                    for value, pivot_value in zip(augmented[row], augmented[column])
                ]
    return tuple(augmented[index][-1] for index in range(size))


def _verify_lp(
    lp: _LinearProgram,
    solution: _LpSolution,
    limits: AiofStrategyLimits,
) -> tuple[Fraction, bool, bool]:
    # Original inequalities and variable bounds are checked directly.
    for name in lp.variable_names:
        if solution.values.get(name, Fraction(0)) < 0:
            raise _StrategyFailure(AiofStrategyStatus.VERIFICATION_FAILED, "negative LP variable", "verification")
    for constraint in lp.constraints:
        lhs = sum(
            (coefficient * solution.values.get(name, Fraction(0)) for coefficient, name in zip(constraint.coefficients, lp.variable_names)),
            Fraction(0),
        )
        valid = lhs <= constraint.rhs if constraint.sense == "<=" else lhs >= constraint.rhs if constraint.sense == ">=" else lhs == constraint.rhs
        if not valid:
            raise _StrategyFailure(AiofStrategyStatus.VERIFICATION_FAILED, "original LP inequality failed", "verification")
    names, matrix, rhs = _standard_nonart(lp)
    active_rows = solution.row_indices
    active_matrix = tuple(matrix[index] for index in active_rows)
    active_rhs = tuple(rhs[index] for index in active_rows)
    z = tuple(solution.values.get(name, Fraction(0)) for name in names)
    if any(value < 0 for value in z):
        raise _StrategyFailure(AiofStrategyStatus.VERIFICATION_FAILED, "standard primal is negative", "verification")
    for row, target in zip(active_matrix, active_rhs):
        if sum((coefficient * value for coefficient, value in zip(row, z)), Fraction(0)) != target:
            raise _StrategyFailure(AiofStrategyStatus.VERIFICATION_FAILED, "standard primal equality failed", "verification")
    if len(solution.basis) != len(active_rows):
        raise _StrategyFailure(AiofStrategyStatus.VERIFICATION_FAILED, "selected basis dimension mismatch", "verification")
    try:
        basis_columns = tuple(names.index(name) for name in solution.basis)
    except ValueError as exc:
        raise _StrategyFailure(AiofStrategyStatus.VERIFICATION_FAILED, "selected basis contains unknown variable", "verification") from exc
    # B^T lambda = c_B.
    transpose = tuple(
        tuple(active_matrix[row][basis_columns[column]] for row in range(len(active_rows)))
        for column in range(len(basis_columns))
    )
    full_costs = tuple(
        lp.objective[lp.variable_names.index(name)] if name in lp.variable_names else Fraction(0)
        for name in names
    )
    c_basis = tuple(full_costs[column] for column in basis_columns)
    dual = _gaussian_solve(transpose, c_basis, limits, "verification")
    for column, cost in enumerate(full_costs):
        lhs = sum((active_matrix[row][column] * dual[row] for row in range(len(active_rows))), Fraction(0))
        if lhs < cost:
            raise _StrategyFailure(AiofStrategyStatus.VERIFICATION_FAILED, "standard dual inequality failed", "verification")
    primal_objective = _check_fraction(
        lp.objective_constant
        + sum((coefficient * solution.values.get(name, Fraction(0)) for coefficient, name in zip(lp.objective, lp.variable_names)), Fraction(0)),
        limits,
        "verification",
    )
    dual_objective = _check_fraction(
        lp.objective_constant + sum((target * value for target, value in zip(active_rhs, dual)), Fraction(0)),
        limits,
        "verification",
    )
    if primal_objective != dual_objective or primal_objective != solution.summary.objective:
        raise _StrategyFailure(AiofStrategyStatus.VERIFICATION_FAILED, "strong dual or objective check failed", "verification")
    return primal_objective, True, True


def _profile(
    game: _RationalGame,
    x: Sequence[Fraction],
    y: Sequence[Fraction],
) -> ExactBehaviourProfile:
    sb = tuple(ExactComboActionProbability(combo, value) for combo, value in zip(game.sb_combos, x))
    bb = tuple(ExactComboActionProbability(combo, value) for combo, value in zip(game.bb_combos, y))
    return ExactBehaviourProfile(sb, bb, _identity({"sb_shove": sb, "bb_call": bb}))


def _validate_exact_profile(profile: object, game: _RationalGame) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    if not isinstance(profile, ExactBehaviourProfile):
        raise _StrategyFailure(AiofStrategyStatus.INVALID_STRATEGY, "profile must be ExactBehaviourProfile", "strategy_validation")

    def side(
        entries: object, expected: tuple[str, ...], name: str
    ) -> tuple[Fraction, ...]:
        if not isinstance(entries, tuple) or any(not isinstance(item, ExactComboActionProbability) for item in entries):
            raise _StrategyFailure(AiofStrategyStatus.INVALID_STRATEGY, f"{name} must be an exact tuple", "strategy_validation")
        combos = tuple(item.combo for item in entries)
        if combos != expected or len(set(combos)) != len(combos):
            raise _StrategyFailure(AiofStrategyStatus.INVALID_STRATEGY, f"{name} support is incomplete or noncanonical", "strategy_validation")
        probabilities = tuple(item.probability for item in entries)
        if any(
            isinstance(value, bool)
            or not isinstance(value, Fraction)
            or value < 0
            or value > 1
            for value in probabilities
        ):
            raise _StrategyFailure(AiofStrategyStatus.INVALID_STRATEGY, f"{name} probability is invalid", "strategy_validation")
        return probabilities

    return side(profile.sb_shove, game.sb_combos, "sb_shove"), side(
        profile.bb_call, game.bb_combos, "bb_call"
    )


def _best_sets(
    actions: tuple[str, ...],
    values: tuple[Fraction, ...],
    tolerance: Fraction,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    maximum = max(values)
    exact = tuple(action for action, value in zip(actions, values) if value == maximum)
    display = tuple(action for action, value in zip(actions, values) if value >= maximum - tolerance)
    return exact, display


def _audit_profile(
    game: _RationalGame,
    x: Sequence[Fraction],
    y: Sequence[Fraction],
    display_tolerance: Fraction,
    limits: AiofStrategyLimits,
) -> tuple[ExactGainSnapshot, tuple[ExactStrategyRow, ...], tuple[ExactStrategyRow, ...]]:
    if (
        len(x) != len(game.sb_combos)
        or len(y) != len(game.bb_combos)
        or any(
            isinstance(value, bool)
            or not isinstance(value, Fraction)
            or value < 0
            or value > 1
            for value in tuple(x) + tuple(y)
        )
    ):
        raise _StrategyFailure(AiofStrategyStatus.INVALID_STRATEGY, "invalid exact profile", "verification")
    profile_terms: list[Fraction] = []
    for i in range(len(x)):
        for j in range(len(y)):
            probability = game.probabilities[i][j]
            showdown = game.showdown[i][j]
            if probability is None or showdown is None:
                continue
            pair = (1 - x[i]) * game.f + x[i] * ((1 - y[j]) * game.g + y[j] * showdown)
            profile_terms.append(_check_fraction(probability * pair, limits, "verification"))
    profile_value = _check_fraction(sum(profile_terms, Fraction(0)), limits, "verification")
    sb_rows: list[ExactStrategyRow] = []
    sb_gains: list[Fraction] = []
    for i, combo in enumerate(game.sb_combos):
        shove = _check_fraction(
            sum(
                (
                    (game.probabilities[i][j] or Fraction(0))
                    * ((1 - y[j]) * game.g + y[j] * (game.showdown[i][j] or Fraction(0)))
                    for j in range(len(y))
                ),
                Fraction(0),
            )
            / game.p_h[i],
            limits,
            "verification",
        )
        fold = game.f
        exact, display = _best_sets(SB_ACTIONS, (shove, fold), display_tolerance)
        supplied = _check_fraction(x[i] * shove + (1 - x[i]) * fold, limits, "verification")
        gain = _check_fraction(game.p_h[i] * (max(shove, fold) - supplied), limits, "verification")
        if gain < 0:
            raise _StrategyFailure(AiofStrategyStatus.VERIFICATION_FAILED, "negative SB unilateral gain", "verification")
        sb_gains.append(gain)
        sb_rows.append(
            ExactStrategyRow(
                "sb",
                combo,
                game.p_h[i],
                game.p_h[i],
                "shove",
                x[i],
                1 - x[i],
                (ExactActionValue("shove", shove), ExactActionValue("fold", fold)),
                exact,
                display,
                gain,
            )
        )
    bb_rows: list[ExactStrategyRow] = []
    bb_gains: list[Fraction] = []
    for j, combo in enumerate(game.bb_combos):
        reach = _check_fraction(
            sum(((game.probabilities[i][j] or Fraction(0)) * x[i] for i in range(len(x))), Fraction(0)),
            limits,
            "verification",
        )
        if reach == 0:
            bb_rows.append(
                ExactStrategyRow(
                    "bb",
                    combo,
                    game.p_v[j],
                    Fraction(0),
                    "call",
                    y[j],
                    1 - y[j],
                    (ExactActionValue("call", None), ExactActionValue("fold", None)),
                    BB_ACTIONS,
                    BB_ACTIONS,
                    Fraction(0),
                )
            )
            continue
        call = _check_fraction(
            sum(
                (
                    (game.probabilities[i][j] or Fraction(0))
                    * x[i]
                    * -(game.showdown[i][j] or Fraction(0))
                    for i in range(len(x))
                ),
                Fraction(0),
            )
            / reach,
            limits,
            "verification",
        )
        fold = -game.g
        exact, display = _best_sets(BB_ACTIONS, (call, fold), display_tolerance)
        supplied = _check_fraction(y[j] * call + (1 - y[j]) * fold, limits, "verification")
        gain = _check_fraction(reach * (max(call, fold) - supplied), limits, "verification")
        if gain < 0:
            raise _StrategyFailure(AiofStrategyStatus.VERIFICATION_FAILED, "negative BB unilateral gain", "verification")
        bb_gains.append(gain)
        bb_rows.append(
            ExactStrategyRow(
                "bb",
                combo,
                game.p_v[j],
                reach,
                "call",
                y[j],
                1 - y[j],
                (ExactActionValue("call", call), ExactActionValue("fold", fold)),
                exact,
                display,
                gain,
            )
        )
    g_sb = _check_fraction(sum(sb_gains, Fraction(0)), limits, "verification")
    g_bb = _check_fraction(sum(bb_gains, Fraction(0)), limits, "verification")
    sb_br = _check_fraction(profile_value + g_sb, limits, "verification")
    bb_br_sb = _check_fraction(profile_value - g_bb, limits, "verification")
    if bb_br_sb > profile_value or profile_value > sb_br:
        raise _StrategyFailure(AiofStrategyStatus.VERIFICATION_FAILED, "inverted value enclosure", "verification")
    snapshot = ExactGainSnapshot(
        profile_value,
        sb_br,
        bb_br_sb,
        g_sb,
        g_bb,
        _check_fraction(g_sb + g_bb, limits, "verification"),
        max(g_sb, g_bb),
        bb_br_sb,
        sb_br,
    )
    return snapshot, tuple(sb_rows), tuple(bb_rows)


def _verified_witness(
    game: _RationalGame,
    profile: ExactBehaviourProfile,
    x: Sequence[Fraction],
    y: Sequence[Fraction],
    sb_lp: _LinearProgram,
    bb_lp: _LinearProgram,
    sb_solution: _LpSolution,
    bb_solution: _LpSolution,
    limits: AiofStrategyLimits,
    claim_epsilon: Fraction,
    display_tolerance: Fraction,
) -> tuple[RationalVerificationWitness, SimplexRunSummary, SimplexRunSummary]:
    profile_x, profile_y = _validate_exact_profile(profile, game)
    if tuple(x) != profile_x or tuple(y) != profile_y:
        raise _StrategyFailure(AiofStrategyStatus.INVALID_STRATEGY, "profile ratios do not match selected variables", "strategy_validation")
    lower, sb_primal, sb_dual = _verify_lp(sb_lp, sb_solution, limits)
    bb_max, bb_primal, bb_dual = _verify_lp(bb_lp, bb_solution, limits)
    upper = -bb_max
    if lower != upper:
        raise _StrategyFailure(AiofStrategyStatus.VERIFICATION_FAILED, "lower and upper LP objectives differ", "verification")
    gains, sb_rows, bb_rows = _audit_profile(game, x, y, display_tolerance, limits)
    if gains.profile_value != lower or gains.g_sb != 0 or gains.g_bb != 0:
        raise _StrategyFailure(AiofStrategyStatus.VERIFICATION_FAILED, "selected profile does not match exact zero-gap objectives", "verification")
    if gains.max_unilateral_gain > claim_epsilon:
        raise _StrategyFailure(AiofStrategyStatus.VERIFICATION_FAILED, "claim epsilon is below unilateral gain", "verification")
    claim = StrategyClaimKind.EXACT_NASH if gains.g_sb == gains.g_bb == 0 else StrategyClaimKind.EPSILON_NASH
    verification_identity = _identity(
        {
            "verifier": VERIFIER_ID,
            "game": GAME_ID,
            "payoff": game.payoff_identity,
            "profile": profile.content_identity,
            "lower": lower,
            "upper": upper,
            "gains": gains,
            "claim": claim,
            "epsilon": claim_epsilon,
        }
    )
    witness = RationalVerificationWitness(
        VERIFIER_ID,
        GAME_ID,
        game.payoff_identity,
        profile.content_identity,
        verification_identity,
        lower,
        upper,
        claim,
        claim_epsilon,
        Fraction(0),
        sb_primal and bb_primal,
        sb_dual and bb_dual,
        lower == upper,
        gains,
        sb_rows,
        bb_rows,
    )
    sb_summary = SimplexRunSummary(
        **{**sb_solution.summary.__dict__, "primal_feasible": sb_primal, "dual_feasible": sb_dual}
    )
    bb_summary = SimplexRunSummary(
        **{**bb_solution.summary.__dict__, "primal_feasible": bb_primal, "dual_feasible": bb_dual}
    )
    return witness, sb_summary, bb_summary


def _pure_plan_value(game: _RationalGame, sb_plan: Sequence[str], bb_plan: Sequence[str]) -> Fraction:
    total = Fraction(0)
    for i, sb_action in enumerate(sb_plan):
        x = Fraction(sb_action == "shove")
        for j, bb_action in enumerate(bb_plan):
            probability = game.probabilities[i][j]
            showdown = game.showdown[i][j]
            if probability is None or showdown is None:
                continue
            y = Fraction(bb_action == "call")
            total += probability * ((1 - x) * game.f + x * ((1 - y) * game.g + y * showdown))
    return total


def _oracle_linear_solve(
    matrix: Sequence[Sequence[Fraction]], rhs: Sequence[Fraction], limits: AiofStrategyLimits
) -> tuple[Fraction, ...] | None:
    try:
        return _gaussian_solve(matrix, rhs, limits, "oracle")
    except _StrategyFailure as exc:
        if exc.status is AiofStrategyStatus.VERIFICATION_FAILED and "singular" in str(exc):
            return None
        raise


def _reference_oracle(
    game: _RationalGame,
    x: Sequence[Fraction],
    y: Sequence[Fraction],
    witness: RationalVerificationWitness,
    limits: AiofStrategyLimits,
    display_tolerance: Fraction,
) -> ReferenceOracleComparison:
    h_count = len(game.sb_combos)
    v_count = len(game.bb_combos)
    if h_count > limits.max_oracle_combos_per_side or v_count > limits.max_oracle_combos_per_side:
        raise _StrategyFailure(AiofStrategyStatus.CAP_EXCEEDED, "oracle combo cap exceeded", "oracle_preflight")
    m = 2**h_count
    n = 2**v_count
    if m > limits.max_oracle_pure_plans_per_side or n > limits.max_oracle_pure_plans_per_side:
        raise _StrategyFailure(AiofStrategyStatus.CAP_EXCEEDED, "oracle pure-plan cap exceeded", "oracle_preflight")
    cells = m * n
    if cells > limits.max_oracle_matrix_cells:
        raise _StrategyFailure(AiofStrategyStatus.CAP_EXCEEDED, "oracle matrix cap exceeded", "oracle_preflight")
    max_support = min(m, n)
    support_systems = 2 * sum(math.comb(m, k) * math.comb(n, k) for k in range(1, max_support + 1))
    if support_systems > limits.max_oracle_support_systems:
        raise _StrategyFailure(AiofStrategyStatus.CAP_EXCEEDED, "oracle support-system cap exceeded", "oracle_preflight")
    sb_plans = tuple(itertools.product(SB_ACTIONS, repeat=h_count))
    bb_plans = tuple(itertools.product(BB_ACTIONS, repeat=v_count))
    matrix = tuple(
        tuple(_check_fraction(_pure_plan_value(game, sb_plan, bb_plan), limits, "oracle") for bb_plan in bb_plans)
        for sb_plan in sb_plans
    )
    sb_candidates: list[tuple[Fraction, tuple[Fraction, ...]]] = []
    bb_candidates: list[tuple[Fraction, tuple[Fraction, ...]]] = []
    for k in range(1, max_support + 1):
        for support_i in itertools.combinations(range(m), k):
            for support_j in itertools.combinations(range(n), k):
                # SB probabilities on I, with every J payoff equal to value.
                system = []
                target = []
                for column in support_j:
                    system.append([matrix[row][column] for row in support_i] + [Fraction(-1)])
                    target.append(Fraction(0))
                system.append([Fraction(1)] * k + [Fraction(0)])
                target.append(Fraction(1))
                solved = _oracle_linear_solve(system, target, limits)
                if solved is not None:
                    probabilities = solved[:k]
                    value = solved[-1]
                    full = tuple(
                        probabilities[support_i.index(row)] if row in support_i else Fraction(0)
                        for row in range(m)
                    )
                    if all(probability >= 0 for probability in probabilities) and all(
                        sum((full[row] * matrix[row][column] for row in range(m)), Fraction(0)) >= value
                        for column in range(n)
                    ):
                        sb_candidates.append((value, full))
                # BB probabilities on J, with every I payoff equal to value.
                system = []
                target = []
                for row in support_i:
                    system.append([matrix[row][column] for column in support_j] + [Fraction(-1)])
                    target.append(Fraction(0))
                system.append([Fraction(1)] * k + [Fraction(0)])
                target.append(Fraction(1))
                solved = _oracle_linear_solve(system, target, limits)
                if solved is not None:
                    probabilities = solved[:k]
                    value = solved[-1]
                    full = tuple(
                        probabilities[support_j.index(column)] if column in support_j else Fraction(0)
                        for column in range(n)
                    )
                    if all(probability >= 0 for probability in probabilities) and all(
                        sum((matrix[row][column] * full[column] for column in range(n)), Fraction(0)) <= value
                        for row in range(m)
                    ):
                        bb_candidates.append((value, full))
    if not sb_candidates or not bb_candidates:
        raise _StrategyFailure(AiofStrategyStatus.ORACLE_MISMATCH, "oracle found no valid support candidate", "oracle")
    lower_value = max(item[0] for item in sb_candidates)
    upper_value = min(item[0] for item in bb_candidates)
    sb_selected = min(item[1] for item in sb_candidates if item[0] == lower_value)
    bb_selected = min(item[1] for item in bb_candidates if item[0] == upper_value)
    if lower_value != upper_value:
        raise _StrategyFailure(AiofStrategyStatus.ORACLE_MISMATCH, "oracle lower and upper values differ", "oracle")

    def behavioural_plan_probability(plan: Sequence[str], values: Sequence[Fraction], active: str) -> Fraction:
        result = Fraction(1)
        for action, probability in zip(plan, values):
            result *= probability if action == active else 1 - probability
        return result

    selected_sb = tuple(behavioural_plan_probability(plan, x, "shove") for plan in sb_plans)
    selected_bb = tuple(behavioural_plan_probability(plan, y, "call") for plan in bb_plans)
    selected_value = sum(
        (
            selected_sb[row] * selected_bb[column] * matrix[row][column]
            for row in range(m)
            for column in range(n)
        ),
        Fraction(0),
    )
    sb_br = max(
        sum((selected_bb[column] * matrix[row][column] for column in range(n)), Fraction(0))
        for row in range(m)
    )
    bb_br = min(
        sum((selected_sb[row] * matrix[row][column] for row in range(m)), Fraction(0))
        for column in range(n)
    )
    gains_match = (
        sb_br - selected_value == witness.gains.g_sb
        and selected_value - bb_br == witness.gains.g_bb
    )
    # Recompute classifications from game coefficients, not simplex state.
    audit, sb_rows, bb_rows = _audit_profile(game, x, y, display_tolerance, limits)
    tie_matches = (
        tuple(row.exact_best_actions for row in sb_rows) == tuple(row.exact_best_actions for row in witness.sb_rows)
        and tuple(row.exact_best_actions for row in bb_rows) == tuple(row.exact_best_actions for row in witness.bb_rows)
    )
    off_path_matches = tuple(row.information_reach_probability == 0 for row in bb_rows) == tuple(
        row.information_reach_probability == 0 for row in witness.bb_rows
    )
    value_matches = lower_value == witness.lower_objective == witness.upper_objective
    selected_value_matches = selected_value == witness.gains.profile_value == audit.profile_value
    if not (value_matches and selected_value_matches and gains_match and tie_matches and off_path_matches):
        raise _StrategyFailure(AiofStrategyStatus.ORACLE_MISMATCH, "oracle comparison mismatch", "oracle")
    comparison_identity = _identity(
        {
            "oracle": ORACLE_ID,
            "matrix": matrix,
            "sb_selected": sb_selected,
            "bb_selected": bb_selected,
            "lower": lower_value,
            "upper": upper_value,
            "profile": witness.profile_identity,
        }
    )
    return ReferenceOracleComparison(
        ORACLE_ID,
        m,
        n,
        cells,
        support_systems,
        lower_value,
        upper_value,
        value_matches,
        selected_value_matches,
        gains_match,
        tie_matches,
        off_path_matches,
        comparison_identity,
    )


def _phase1_float_diagnostic(
    request: RationalStrategyRequest,
    profile: ExactBehaviourProfile,
    witness: RationalVerificationWitness,
) -> Phase1FloatDiagnostic:
    sb = tuple(ComboActionProbability(item.combo, float(item.probability)) for item in profile.sb_shove)
    bb = tuple(ComboActionProbability(item.combo, float(item.probability)) for item in profile.bb_call)
    tolerance = float(request.display_tie_tolerance)
    result = analyze_pushfold(
        PushFoldRequest(
            request.sb_range,
            request.bb_range,
            request.dead_cards,
            EquityAlgorithm.EXACT_EXHAUSTIVE,
            _phase1_limits(request.limits),
            0,
            request.game,
            SuppliedProfile(sb, bb),
            ("sb", "bb"),
            tolerance,
            None,
            None,
        )
    )
    if result.status is not AiofStatus.SUCCESS or result.analysis is None:
        return Phase1FloatDiagnostic(
            result.status,
            tolerance,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            result.error_message or result.status.value,
        )
    analysis = result.analysis
    responses = {item.seat: item for item in analysis.best_responses}
    phase_g_sb = responses["sb"].raw_gain
    phase_g_bb = responses["bb"].raw_gain
    exact_value = float(witness.gains.profile_value)
    exact_g_sb = float(witness.gains.g_sb)
    exact_g_bb = float(witness.gains.g_bb)
    tie_matches = (
        tuple(row.best_actions for row in responses["sb"].rows)
        == tuple(row.display_best_actions for row in witness.sb_rows)
        and tuple(row.best_actions for row in responses["bb"].rows)
        == tuple(row.display_best_actions for row in witness.bb_rows)
    )
    off_path_matches = tuple(row.information_reach_probability == 0.0 for row in responses["bb"].rows) == tuple(
        row.information_reach_probability == 0 for row in witness.bb_rows
    )
    differences = (
        analysis.profile_value_sb - exact_value,
        phase_g_sb - exact_g_sb,
        phase_g_bb - exact_g_bb,
    )
    bound = max(tolerance, 1e-12)
    return Phase1FloatDiagnostic(
        result.status,
        tolerance,
        analysis.profile_value_sb,
        phase_g_sb,
        phase_g_bb,
        differences[0],
        differences[1],
        differences[2],
        tie_matches,
        off_path_matches,
        all(abs(value) <= bound for value in differences),
        None,
    )


def _error_result(exc: _StrategyFailure) -> RationalStrategyRunResult:
    return RationalStrategyRunResult(
        exc.status,
        None,
        StrategyError(
            str(exc) or exc.status.value,
            exc.phase,
            exc.upstream_status,
            exc.completed_payoff_cells,
            exc.completed_pivots,
        ),
    )


def generate_rational_lift_strategy(
    request: RationalStrategyRequest,
) -> RationalStrategyRunResult:
    """Generate and independently verify one deterministic basic profile.

    A successful result is an exact claim only for
    ``aiof-rational-lift-game-v1``.  Failure returns no partial payoff, LP,
    strategy, value, or witness payload.
    """

    try:
        limits, trace_points = _validate_common(request, heuristic=False)
        game = _build_rational_game(
            request,
            limits,
            extra_float_pass=request.run_phase1_float_diagnostic,
        )
        sb_lp = _make_lp(game, limits, "SB")
        bb_lp = _make_lp(game, limits, "BB")
        sb_solution = _solve_lp(sb_lp, limits, trace_points)
        bb_solution = _solve_lp(bb_lp, limits, trace_points)
        x = tuple(sb_solution.values[f"x[{combo}]"] for combo in game.sb_combos)
        y = tuple(bb_solution.values[f"y[{combo}]"] for combo in game.bb_combos)
        profile = _profile(game, x, y)
        witness, sb_summary, bb_summary = _verified_witness(
            game,
            profile,
            x,
            y,
            sb_lp,
            bb_lp,
            sb_solution,
            bb_solution,
            limits,
            request.claim_epsilon,
            request.display_tie_tolerance,
        )
        oracle = (
            _reference_oracle(game, x, y, witness, limits, request.display_tie_tolerance)
            if request.run_reference_oracle
            else None
        )
        float_diagnostic = (
            _phase1_float_diagnostic(request, profile, witness)
            if request.run_phase1_float_diagnostic
            else None
        )
        semantic_identity = _identity(
            {
                "game": GAME_ID,
                "prepared": game.prepared_identity,
                "payoff": game.payoff_identity,
                "algorithms": (
                    request.algorithm,
                    SIMPLEX_ID,
                    VERIFIER_ID,
                    ORACLE_ID,
                    EXACT_ID,
                    CHIP_ACCOUNTING_ID,
                ),
                "profile": profile,
                "witness": witness.verification_identity,
                "limits": limits,
            }
        )
        input_identity = _identity(
            {
                "semantic": semantic_identity,
                "claim_epsilon": request.claim_epsilon,
                "display_tie_tolerance": request.display_tie_tolerance,
                "trace": request.requested_trace_points,
                "oracle": request.run_reference_oracle,
                "phase1_float": request.run_phase1_float_diagnostic,
            }
        )
        runtime_identity = _runtime_identity()
        run_identity = _identity(
            {
                "input_identity": input_identity,
                "semantic_identity": semantic_identity,
                "runtime_identity": runtime_identity,
            }
        )
        strategy_result = RationalStrategyResult(
            request.algorithm,
            GAME_ID,
            VERIFIER_ID,
            game.prepared_identity,
            game.payoff_identity,
            semantic_identity,
            input_identity,
            runtime_identity,
            run_identity,
            game.payoff_cells,
            game.board_evaluations * (2 if request.run_phase1_float_diagnostic else 1),
            profile,
            sb_summary,
            bb_summary,
            witness,
            oracle,
            float_diagnostic,
        )
        return RationalStrategyRunResult(AiofStrategyStatus.SUCCESS, strategy_result, None)
    except _StrategyFailure as exc:
        return _error_result(exc)
    except AiofContractError as exc:
        return _error_result(
            _StrategyFailure(AiofStrategyStatus.NUMERIC_FAILURE, str(exc) or exc.status.value, "internal", upstream_status=exc.status)
        )
    except (ArithmeticError, OverflowError, ValueError, TypeError) as exc:
        return _error_result(
            _StrategyFailure(AiofStrategyStatus.NUMERIC_FAILURE, str(exc) or "numeric failure", "internal")
        )


def _state_identity(
    game: _RationalGame,
    update_mode: HeuristicUpdateMode,
    damping: Fraction,
    x: Sequence[Fraction],
    y: Sequence[Fraction],
) -> str:
    return _identity(
        {
            "game": game.payoff_identity,
            "update_mode": update_mode,
            "damping": damping,
            "sb": tuple(zip(game.sb_combos, x)),
            "bb": tuple(zip(game.bb_combos, y)),
        }
    )


def _representative_sb(game: _RationalGame, y: Sequence[Fraction]) -> tuple[Fraction, ...]:
    result = []
    for i in range(len(game.sb_combos)):
        shove = sum(
            (
                (game.probabilities[i][j] or Fraction(0))
                * ((1 - y[j]) * game.g + y[j] * (game.showdown[i][j] or Fraction(0)))
                for j in range(len(y))
            ),
            Fraction(0),
        ) / game.p_h[i]
        result.append(Fraction(shove >= game.f))
    return tuple(result)


def _representative_bb(game: _RationalGame, x: Sequence[Fraction]) -> tuple[Fraction, ...]:
    result = []
    for j in range(len(game.bb_combos)):
        reach = sum(
            ((game.probabilities[i][j] or Fraction(0)) * x[i] for i in range(len(x))),
            Fraction(0),
        )
        if reach == 0:
            result.append(Fraction(1))
            continue
        call = sum(
            (
                (game.probabilities[i][j] or Fraction(0))
                * x[i]
                * -(game.showdown[i][j] or Fraction(0))
                for i in range(len(x))
            ),
            Fraction(0),
        ) / reach
        result.append(Fraction(call >= -game.g))
    return tuple(result)


def _heuristic_error(exc: _StrategyFailure) -> AlternatingBrDiagnosticRunResult:
    return AlternatingBrDiagnosticRunResult(
        exc.status,
        None,
        StrategyError(
            str(exc) or exc.status.value,
            exc.phase,
            exc.upstream_status,
            exc.completed_payoff_cells,
            exc.completed_pivots,
        ),
    )


def run_alternating_br_diagnostic(
    request: AlternatingBrDiagnosticRequest,
) -> AlternatingBrDiagnosticRunResult:
    """Run the bounded alternating exact-BR diagnostic.

    Fixed points, cycles, small gains, and arithmetic-average improvement are
    diagnostic observations only.  This entry point makes no general Nash
    convergence or primary-solver claim.
    """

    try:
        limits, trace_points = _validate_common(request, heuristic=True)
        game = _build_rational_game(request, limits, extra_float_pass=False)
        half = Fraction(1, 2)
        x: tuple[Fraction, ...] = (half,) * len(game.sb_combos)
        y: tuple[Fraction, ...] = (half,) * len(game.bb_combos)
        sum_x = [Fraction(0)] * len(x)
        sum_y = [Fraction(0)] * len(y)
        initial_identity = _state_identity(game, request.update_mode, request.damping, x, y)
        seen = {initial_identity: 0}
        trace: list[HeuristicTracePoint] = []
        diagnostic_status = HeuristicDiagnosticStatus.ITERATION_CAP_REACHED
        repeated_first: int | None = None
        cycle_length: int | None = None
        current_identity = initial_identity
        average_x = x
        average_y = y
        current_gains, _, _ = _audit_profile(game, x, y, request.display_tie_tolerance, limits)
        average_gains = current_gains
        iterations_completed = 0
        for iteration in range(1, request.max_iterations + 1):
            previous_identity = current_identity
            br_x = _representative_sb(game, y)
            new_x = tuple(
                _check_fraction((1 - request.damping) * old + request.damping * best, limits, "heuristic")
                for old, best in zip(x, br_x)
            )
            if request.update_mode is HeuristicUpdateMode.SIMULTANEOUS:
                br_y = _representative_bb(game, x)
            else:
                br_y = _representative_bb(game, new_x)
            new_y = tuple(
                _check_fraction((1 - request.damping) * old + request.damping * best, limits, "heuristic")
                for old, best in zip(y, br_y)
            )
            x, y = new_x, new_y
            iterations_completed = iteration
            for index, value in enumerate(x):
                sum_x[index] = _check_fraction(sum_x[index] + value, limits, "heuristic")
            for index, value in enumerate(y):
                sum_y[index] = _check_fraction(sum_y[index] + value, limits, "heuristic")
            average_x = tuple(_check_fraction(value / iteration, limits, "heuristic") for value in sum_x)
            average_y = tuple(_check_fraction(value / iteration, limits, "heuristic") for value in sum_y)
            current_gains, _, _ = _audit_profile(game, x, y, request.display_tie_tolerance, limits)
            average_gains, _, _ = _audit_profile(
                game, average_x, average_y, request.display_tie_tolerance, limits
            )
            current_identity = _state_identity(game, request.update_mode, request.damping, x, y)
            if len(trace) < trace_points:
                trace.append(
                    HeuristicTracePoint(
                        iteration,
                        current_identity,
                        current_gains.profile_value,
                        current_gains.g_sb,
                        current_gains.g_bb,
                        current_gains.nash_conv,
                        current_gains.max_unilateral_gain,
                        average_gains.profile_value,
                        average_gains.g_sb,
                        average_gains.g_bb,
                        average_gains.nash_conv,
                        average_gains.max_unilateral_gain,
                    )
                )
            if current_identity in seen:
                repeated_first = seen[current_identity]
                cycle_length = iteration - repeated_first
                diagnostic_status = (
                    HeuristicDiagnosticStatus.DIAGNOSTIC_COMPLETE
                    if current_identity == previous_identity
                    else HeuristicDiagnosticStatus.CYCLE_DETECTED
                )
                break
            seen[current_identity] = iteration
        current_profile = _profile(game, x, y)
        average_profile = _profile(game, average_x, average_y)
        semantic_identity = _identity(
            {
                "algorithm": HEURISTIC_ID,
                "game": game.payoff_identity,
                "initialization": INITIALIZATION_ID,
                "update_mode": request.update_mode,
                "damping": request.damping,
                "current": current_profile,
                "average": average_profile,
                "status": diagnostic_status,
                "iterations": iterations_completed,
                "limits": limits,
            }
        )
        input_identity = _identity(
            {
                "semantic": semantic_identity,
                "max_iterations": request.max_iterations,
                "display_tie_tolerance": request.display_tie_tolerance,
                "trace": request.requested_trace_points,
            }
        )
        runtime_identity = _runtime_identity()
        run_identity = _identity(
            {
                "input_identity": input_identity,
                "semantic_identity": semantic_identity,
                "runtime_identity": runtime_identity,
            }
        )
        trace_tuple = tuple(trace)
        diagnostic = AlternatingBrDiagnostic(
            diagnostic_status,
            HEURISTIC_ID,
            GAME_ID,
            INITIALIZATION_ID,
            request.update_mode,
            request.damping,
            semantic_identity,
            input_identity,
            runtime_identity,
            run_identity,
            iterations_completed,
            repeated_first,
            cycle_length,
            current_identity,
            current_profile,
            average_profile,
            current_gains,
            average_gains,
            trace_tuple,
            _identity(trace_tuple),
            iterations_completed > trace_points,
        )
        return AlternatingBrDiagnosticRunResult(AiofStrategyStatus.SUCCESS, diagnostic, None)
    except _StrategyFailure as exc:
        return _heuristic_error(exc)
    except AiofContractError as exc:
        return _heuristic_error(
            _StrategyFailure(AiofStrategyStatus.NUMERIC_FAILURE, str(exc) or exc.status.value, "internal", upstream_status=exc.status)
        )
    except (ArithmeticError, OverflowError, ValueError, TypeError) as exc:
        return _heuristic_error(
            _StrategyFailure(AiofStrategyStatus.NUMERIC_FAILURE, str(exc) or "numeric failure", "internal")
        )
