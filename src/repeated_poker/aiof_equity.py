"""Bounded exact and deterministic Monte Carlo preflop equity calculation.

Exact exhaustive evaluation and Monte Carlo estimation are distinct algorithms.
The API never silently substitutes one for the other, never uses external
equity data, and returns no partial estimate on failure.
"""

from __future__ import annotations

import math
import platform
import sys
from dataclasses import dataclass
from enum import Enum
from itertools import combinations

from .aiof_cards import (
    DECK_ID,
    JOINT_CONDITIONING_ID,
    RANGE_EXPANSION_ID,
    RANGE_GRAMMAR_ID,
    AiofContractError,
    AiofLimits,
    AiofStatus,
    ExpandedCombo,
    PreparedRanges,
    RangeSpec,
    _content_identity,
    prepare_compatible_ranges,
)
from .aiof_evaluator import EVALUATOR_ID, _evaluate_seven_ids


__all__ = [
    "EquityAlgorithm",
    "EquityRequest",
    "OutcomeCounts",
    "OutcomeProbabilities",
    "EquityTracePoint",
    "EquityEstimate",
    "EquityRunResult",
    "calculate_preflop_equity",
]


EXACT_ALGORITHM_ID = "exact_exhaustive-v1"
MONTE_CARLO_ALGORITHM_ID = "deterministic_monte_carlo-v1"
PRNG_ID = "pcg32-v1"
DEFAULT_MONTE_CARLO_SAMPLES = 100_000


class EquityAlgorithm(str, Enum):
    """Supported, non-interchangeable preflop equity algorithms."""

    EXACT_EXHAUSTIVE = EXACT_ALGORITHM_ID
    DETERMINISTIC_MONTE_CARLO = MONTE_CARLO_ALGORITHM_ID


@dataclass(frozen=True)
class EquityRequest:
    """One bounded equity request with an explicit algorithm selection."""

    sb_range: RangeSpec
    bb_range: RangeSpec
    dead_cards: tuple[str, ...]
    algorithm: EquityAlgorithm
    limits: AiofLimits
    requested_trace_points: int
    seed: int | None = None
    samples: int | None = None


@dataclass(frozen=True)
class OutcomeCounts:
    """SB-oriented integer showdown counts."""

    wins: int
    losses: int
    ties: int
    trials: int


@dataclass(frozen=True)
class OutcomeProbabilities:
    """SB-oriented weighted or empirical showdown probabilities."""

    win: float
    loss: float
    tie: float


@dataclass(frozen=True)
class EquityTracePoint:
    """One bounded exact-matchup summary or accepted Monte Carlo sample."""

    index: int
    sb_combo: str
    bb_combo: str
    outcome: str | None
    counts: OutcomeCounts | None
    joint_probability: float | None
    attempt: int | None


@dataclass(frozen=True)
class EquityEstimate:
    """Successful bounded result; it is not an error bound or strategy claim."""

    algorithm: EquityAlgorithm
    deck_id: str
    range_grammar_id: str
    range_expansion_id: str
    joint_conditioning_id: str
    evaluator_id: str
    prng_id: str | None
    input_identity: str
    runtime_identity: str
    run_identity: str
    prepared_ranges_identity: str
    compatible_pair_count: int
    compatible_raw_joint_mass: float
    unweighted_counts: OutcomeCounts
    probabilities: OutcomeProbabilities
    sb_equity: float
    board_evaluations: int
    accepted_samples: int | None
    rejected_hole_draws: int
    seed: int | None
    requested_samples: int | None
    sample_variance: float | None
    standard_error: float | None
    trace: tuple[EquityTracePoint, ...]
    trace_truncated: bool


@dataclass(frozen=True)
class EquityRunResult:
    """Fail-closed result wrapper with payload/status invariants."""

    status: AiofStatus
    estimate: EquityEstimate | None
    error_message: str | None


class _Pcg32:
    _MULTIPLIER = 6_364_136_223_846_793_005
    _INCREMENT = 109
    _MASK64 = (1 << 64) - 1

    def __init__(self, seed: int) -> None:
        self.state = 0
        self.next_u32()
        self.state = (self.state + seed) & self._MASK64
        self.next_u32()

    def next_u32(self) -> int:
        oldstate = self.state
        self.state = (oldstate * self._MULTIPLIER + self._INCREMENT) & self._MASK64
        xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) & 0xFFFFFFFF
        rotation = (oldstate >> 59) & 31
        return ((xorshifted >> rotation) | (xorshifted << ((-rotation) & 31))) & 0xFFFFFFFF

    def next_u64(self) -> int:
        return (self.next_u32() << 32) | self.next_u32()

    def bounded(self, bound: int) -> int:
        if isinstance(bound, bool) or not isinstance(bound, int) or not 0 < bound <= 2**32:
            raise AiofContractError(AiofStatus.INVALID_INPUT, "invalid bounded draw size")
        threshold = ((1 << 32) - bound) % bound
        while True:
            value = self.next_u32()
            if value >= threshold:
                return value % bound


@dataclass(frozen=True)
class _MonteCarloTrial:
    accepted_index: int
    attempt: int
    sb_combo: ExpandedCombo
    bb_combo: ExpandedCombo
    outcome: str


def _require_plain_int(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, f"{name} must be an integer at least {minimum}"
        )
    return value


def _runtime_identity() -> str:
    return _content_identity(
        {
            "implementation": "repeated-poker-analysis-aiof-phase1-v1",
            "python_implementation": sys.implementation.name,
            "python_version": platform.python_version(),
            "evaluator": EVALUATOR_ID,
        }
    )


def _algorithm_identity(algorithm: EquityAlgorithm) -> dict[str, str | None]:
    return {
        "deck": DECK_ID,
        "range_grammar": RANGE_GRAMMAR_ID,
        "range_expansion": RANGE_EXPANSION_ID,
        "joint_conditioning": JOINT_CONDITIONING_ID,
        "evaluator": EVALUATOR_ID,
        "equity": algorithm.value,
        "prng": PRNG_ID if algorithm is EquityAlgorithm.DETERMINISTIC_MONTE_CARLO else None,
    }


def _validate_request(request: EquityRequest) -> tuple[int | None, int | None, int]:
    if not isinstance(request, EquityRequest):
        raise AiofContractError(AiofStatus.INVALID_INPUT, "request must be EquityRequest")
    if not isinstance(request.algorithm, EquityAlgorithm):
        raise AiofContractError(AiofStatus.INVALID_INPUT, "invalid equity algorithm")
    if not isinstance(request.limits, AiofLimits):
        raise AiofContractError(AiofStatus.INVALID_INPUT, "limits must be AiofLimits")
    trace_points = _require_plain_int(request.requested_trace_points, "requested_trace_points")
    if trace_points > request.limits.max_trace_points:
        raise AiofContractError(AiofStatus.CAP_EXCEEDED, "trace-point cap exceeded")
    if request.algorithm is EquityAlgorithm.EXACT_EXHAUSTIVE:
        if request.seed is not None or request.samples is not None:
            raise AiofContractError(AiofStatus.INVALID_INPUT, "exact requests require seed=None and samples=None")
        return None, None, trace_points
    if isinstance(request.seed, bool) or not isinstance(request.seed, int) or not 0 <= request.seed < 2**64:
        raise AiofContractError(AiofStatus.INVALID_INPUT, "Monte Carlo seed must be a 64-bit integer")
    samples = DEFAULT_MONTE_CARLO_SAMPLES if request.samples is None else request.samples
    samples = _require_plain_int(samples, "samples", 2)
    if samples > request.limits.max_monte_carlo_samples:
        raise AiofContractError(AiofStatus.CAP_EXCEEDED, "Monte Carlo sample cap exceeded")
    return request.seed, samples, trace_points


def _joint_probability(prepared: PreparedRanges, sb: ExpandedCombo, bb: ExpandedCombo) -> float:
    return sb.raw_mass * bb.raw_mass * prepared.normalization_factor


def _outcome(sb_cards: tuple[int, int], bb_cards: tuple[int, int], board: tuple[int, ...]) -> str:
    sb_rank = _evaluate_seven_ids(sb_cards + board)
    bb_rank = _evaluate_seven_ids(bb_cards + board)
    if sb_rank > bb_rank:
        return "win"
    if sb_rank < bb_rank:
        return "loss"
    return "tie"


def _iter_exact_matchup_outcomes(prepared: PreparedRanges):
    """Yield canonical compatible matchups and their exhaustive counts."""

    dead = set(prepared.dead_card_ids)
    for sb_combo in prepared.sb_range.combos:
        for bb_combo in prepared.bb_range.combos:
            if set(sb_combo.card_ids) & set(bb_combo.card_ids):
                continue
            excluded = dead | set(sb_combo.card_ids) | set(bb_combo.card_ids)
            remaining = tuple(value for value in range(52) if value not in excluded)
            wins = losses = ties = 0
            for board in combinations(remaining, 5):
                result = _outcome(sb_combo.card_ids, bb_combo.card_ids, board)
                if result == "win":
                    wins += 1
                elif result == "loss":
                    losses += 1
                else:
                    ties += 1
            trials = wins + losses + ties
            yield sb_combo, bb_combo, _joint_probability(prepared, sb_combo, bb_combo), OutcomeCounts(
                wins, losses, ties, trials
            )


def _weighted_cdf(combos: tuple[ExpandedCombo, ...]) -> tuple[float, ...]:
    masses = tuple(combo.raw_mass for combo in combos)
    total = math.fsum(masses)
    if not math.isfinite(total) or total <= 0.0:
        raise AiofContractError(AiofStatus.NUMERIC_FAILURE, "invalid weighted draw support")
    boundaries = [math.fsum(masses[: index + 1]) / total for index in range(len(masses))]
    boundaries[-1] = 1.0
    if any(not math.isfinite(value) for value in boundaries):
        raise AiofContractError(AiofStatus.NUMERIC_FAILURE, "invalid weighted CDF")
    return tuple(boundaries)


def _weighted_draw(
    rng: _Pcg32, combos: tuple[ExpandedCombo, ...], boundaries: tuple[float, ...]
) -> ExpandedCombo:
    value = rng.next_u64() / 2**64
    for combo, boundary in zip(combos, boundaries):
        if value < boundary:
            return combo
    raise AiofContractError(AiofStatus.NUMERIC_FAILURE, "weighted draw missed final boundary")


def _iter_monte_carlo_trials(
    prepared: PreparedRanges, seed: int, samples: int, attempt_limit: int
):
    """Yield accepted trials in the contract-fixed sampling order."""

    rng = _Pcg32(seed)
    sb_cdf = _weighted_cdf(prepared.sb_range.combos)
    bb_cdf = _weighted_cdf(prepared.bb_range.combos)
    dead = set(prepared.dead_card_ids)
    accepted = 0
    for attempt in range(1, attempt_limit + 1):
        sb_combo = _weighted_draw(rng, prepared.sb_range.combos, sb_cdf)
        bb_combo = _weighted_draw(rng, prepared.bb_range.combos, bb_cdf)
        if set(sb_combo.card_ids) & set(bb_combo.card_ids):
            continue
        excluded = dead | set(sb_combo.card_ids) | set(bb_combo.card_ids)
        remaining = [value for value in range(52) if value not in excluded]
        for index in range(5):
            chosen = index + rng.bounded(len(remaining) - index)
            remaining[index], remaining[chosen] = remaining[chosen], remaining[index]
        board = tuple(remaining[:5])
        accepted += 1
        yield _MonteCarloTrial(
            accepted, attempt, sb_combo, bb_combo, _outcome(sb_combo.card_ids, bb_combo.card_ids, board)
        )
        if accepted == samples:
            return
    raise AiofContractError(
        AiofStatus.SAMPLING_ATTEMPT_CAP_EXCEEDED,
        f"accepted {accepted} of {samples} samples before attempt cap",
    )


def _projected_exact_evaluations(prepared: PreparedRanges) -> int:
    remaining = 52 - len(prepared.dead_card_ids) - 4
    if remaining < 5:
        raise AiofContractError(AiofStatus.INVALID_CARD_INPUT, "fewer than five board cards remain")
    return prepared.compatible_pair_count * math.comb(remaining, 5)


def _make_identities(
    request: EquityRequest,
    prepared: PreparedRanges,
    seed: int | None,
    samples: int | None,
) -> tuple[str, str, str]:
    input_identity = _content_identity(
        {
            "algorithms": _algorithm_identity(request.algorithm),
            "prepared_ranges": prepared.content_identity,
            "seed": seed,
            "samples": samples,
            "limits": request.limits,
            "requested_trace_points": request.requested_trace_points,
        }
    )
    runtime_identity = _runtime_identity()
    return input_identity, runtime_identity, _content_identity(
        {"input_identity": input_identity, "runtime_identity": runtime_identity}
    )


def _exact_estimate(
    request: EquityRequest, prepared: PreparedRanges, trace_points: int
) -> EquityEstimate:
    projected = _projected_exact_evaluations(prepared)
    if projected > request.limits.max_exact_board_evaluations:
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            f"projected {projected} board evaluations exceed cap",
        )
    wins = losses = ties = trials = 0
    win_terms: list[float] = []
    loss_terms: list[float] = []
    tie_terms: list[float] = []
    trace: list[EquityTracePoint] = []
    matchup_index = 0
    for sb_combo, bb_combo, probability, counts in _iter_exact_matchup_outcomes(prepared):
        matchup_index += 1
        if counts.trials <= 0:
            raise AiofContractError(AiofStatus.NUMERIC_FAILURE, "exact matchup has zero boards")
        wins += counts.wins
        losses += counts.losses
        ties += counts.ties
        trials += counts.trials
        win_terms.append(probability * counts.wins / counts.trials)
        loss_terms.append(probability * counts.losses / counts.trials)
        tie_terms.append(probability * counts.ties / counts.trials)
        if matchup_index <= trace_points:
            trace.append(
                EquityTracePoint(
                    matchup_index, sb_combo.combo, bb_combo.combo, None, counts, probability, None
                )
            )
    probabilities = OutcomeProbabilities(math.fsum(win_terms), math.fsum(loss_terms), math.fsum(tie_terms))
    total_probability = math.fsum((probabilities.win, probabilities.loss, probabilities.tie))
    if not all(math.isfinite(value) for value in (probabilities.win, probabilities.loss, probabilities.tie)) or abs(total_probability - 1.0) > 1e-10:
        raise AiofContractError(AiofStatus.NUMERIC_FAILURE, "weighted exact probabilities are invalid")
    equity = probabilities.win + 0.5 * probabilities.tie
    input_id, runtime_id, run_id = _make_identities(request, prepared, None, None)
    return EquityEstimate(
        request.algorithm,
        DECK_ID,
        RANGE_GRAMMAR_ID,
        RANGE_EXPANSION_ID,
        JOINT_CONDITIONING_ID,
        EVALUATOR_ID,
        None,
        input_id,
        runtime_id,
        run_id,
        prepared.content_identity,
        prepared.compatible_pair_count,
        prepared.compatible_raw_joint_mass,
        OutcomeCounts(wins, losses, ties, trials),
        probabilities,
        equity,
        projected,
        None,
        0,
        None,
        None,
        None,
        None,
        tuple(trace),
        prepared.compatible_pair_count > trace_points,
    )


def _monte_carlo_estimate(
    request: EquityRequest,
    prepared: PreparedRanges,
    seed: int,
    samples: int,
    trace_points: int,
) -> EquityEstimate:
    contract_attempt_limit = 100 * samples
    attempt_limit = (
        contract_attempt_limit
        if request.limits.max_sampling_attempts is None
        else min(contract_attempt_limit, request.limits.max_sampling_attempts)
    )
    wins = losses = ties = 0
    final_attempt = 0
    trace: list[EquityTracePoint] = []
    for trial in _iter_monte_carlo_trials(prepared, seed, samples, attempt_limit):
        final_attempt = trial.attempt
        if trial.outcome == "win":
            wins += 1
        elif trial.outcome == "loss":
            losses += 1
        else:
            ties += 1
        if trial.accepted_index <= trace_points:
            trace.append(
                EquityTracePoint(
                    trial.accepted_index,
                    trial.sb_combo.combo,
                    trial.bb_combo.combo,
                    trial.outcome,
                    None,
                    None,
                    trial.attempt,
                )
            )
    accepted = wins + losses + ties
    if accepted != samples:
        raise AiofContractError(AiofStatus.NON_REPRODUCIBLE, "Monte Carlo completed with wrong sample count")
    mean = (wins + 0.5 * ties) / samples
    squared = math.fsum(
        (
            wins * (1.0 - mean) ** 2,
            losses * mean**2,
            ties * (0.5 - mean) ** 2,
        )
    )
    variance = squared / (samples - 1)
    standard_error = math.sqrt(variance / samples)
    if not all(math.isfinite(value) for value in (mean, variance, standard_error)) or variance < 0.0:
        raise AiofContractError(AiofStatus.NUMERIC_FAILURE, "invalid Monte Carlo statistics")
    probabilities = OutcomeProbabilities(wins / samples, losses / samples, ties / samples)
    input_id, runtime_id, run_id = _make_identities(request, prepared, seed, samples)
    return EquityEstimate(
        request.algorithm,
        DECK_ID,
        RANGE_GRAMMAR_ID,
        RANGE_EXPANSION_ID,
        JOINT_CONDITIONING_ID,
        EVALUATOR_ID,
        PRNG_ID,
        input_id,
        runtime_id,
        run_id,
        prepared.content_identity,
        prepared.compatible_pair_count,
        prepared.compatible_raw_joint_mass,
        OutcomeCounts(wins, losses, ties, samples),
        probabilities,
        mean,
        samples,
        samples,
        final_attempt - samples,
        seed,
        samples,
        variance,
        standard_error,
        tuple(trace),
        samples > trace_points,
    )


def calculate_preflop_equity(request: EquityRequest) -> EquityRunResult:
    """Calculate one bounded equity result without fallback or partial payload."""

    try:
        seed, samples, trace_points = _validate_request(request)
        prepared = prepare_compatible_ranges(
            request.sb_range, request.bb_range, request.dead_cards, request.limits
        )
        if request.algorithm is EquityAlgorithm.EXACT_EXHAUSTIVE:
            estimate = _exact_estimate(request, prepared, trace_points)
        else:
            assert seed is not None and samples is not None
            estimate = _monte_carlo_estimate(request, prepared, seed, samples, trace_points)
        return EquityRunResult(AiofStatus.SUCCESS, estimate, None)
    except AiofContractError as exc:
        return EquityRunResult(exc.status, None, str(exc) or exc.status.value)
    except (ArithmeticError, OverflowError) as exc:
        return EquityRunResult(AiofStatus.NUMERIC_FAILURE, None, str(exc) or "numeric failure")
