"""Bounded heads-up fee-zero AIoF ChipEV and fixed-opponent diagnostics.

Values are net chip deltas from stacks immediately before mandatory posts.
They are not ICM, Future-ICM, tournament dollar EV, equilibrium certificates,
optimal charts, profitability claims, or real-money recommendations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .aiof_cards import (
    AiofContractError,
    AiofLimits,
    AiofStatus,
    PreparedRanges,
    RangeSpec,
    _content_identity,
    canonicalize_exact_combo,
    prepare_compatible_ranges,
)
from .aiof_equity import (
    DEFAULT_MONTE_CARLO_SAMPLES,
    EquityAlgorithm,
    EquityRequest,
    EquityTracePoint,
    OutcomeCounts,
    OutcomeProbabilities,
    _iter_exact_matchup_outcomes,
    _iter_monte_carlo_trials,
    _make_identities,
    _projected_exact_evaluations,
    _validate_request,
)


__all__ = [
    "HeadsUpChipEvGame",
    "ComboActionProbability",
    "SuppliedProfile",
    "PushFoldRequest",
    "BestResponseRow",
    "UnilateralBestResponse",
    "PushFoldAnalysis",
    "PushFoldRunResult",
    "analyze_pushfold",
]


CHIP_ACCOUNTING_ID = "heads-up-fee0-net-chipev-v1"
SB_ACTIONS = ("shove", "fold")
BB_ACTIONS = ("call", "fold")


@dataclass(frozen=True)
class HeadsUpChipEvGame:
    """Supported two-player preflop accounting inputs."""

    starting_stack_sb: float
    starting_stack_bb: float
    small_blind: float
    big_blind: float
    ante: float
    fee: float = 0.0
    third_party_dead_money: float = 0.0
    side_pot: bool = False


@dataclass(frozen=True)
class ComboActionProbability:
    """One exact own combo and its shove or call probability."""

    combo: str
    probability: float


@dataclass(frozen=True)
class SuppliedProfile:
    """Complete exact-combo-conditioned SB shove and BB call profile."""

    sb_shove: tuple[ComboActionProbability, ...]
    bb_call: tuple[ComboActionProbability, ...]


@dataclass(frozen=True)
class PushFoldRequest:
    """One bounded supplied-profile ChipEV and optional exact-BR request."""

    sb_range: RangeSpec
    bb_range: RangeSpec
    dead_cards: tuple[str, ...]
    algorithm: EquityAlgorithm
    limits: AiofLimits
    requested_trace_points: int
    game: HeadsUpChipEvGame
    profile: SuppliedProfile
    best_response_seats: tuple[str, ...] = ()
    deviation_tolerance: float = 0.0
    seed: int | None = None
    samples: int | None = None


@dataclass(frozen=True)
class BestResponseRow:
    """Per-own-combo finite-action comparison under a fixed opponent."""

    combo: str
    compatible_probability: float
    information_reach_probability: float
    action_values: tuple[tuple[str, float | None], ...]
    best_actions: tuple[str, ...]
    supplied_action_probability: float
    raw_gain: float


@dataclass(frozen=True)
class UnilateralBestResponse:
    """Exact fixed-opponent unilateral best-response correspondence."""

    seat: str
    supplied_profile_value: float
    best_response_value: float
    raw_gain: float
    rows: tuple[BestResponseRow, ...]


@dataclass(frozen=True)
class PushFoldAnalysis:
    """Successful profile result and any requested exact correspondences."""

    algorithm: EquityAlgorithm
    chip_accounting_id: str
    input_identity: str
    runtime_identity: str
    run_identity: str
    prepared_ranges_identity: str
    compatible_pair_count: int
    outcome_counts: OutcomeCounts
    outcome_probabilities: OutcomeProbabilities
    profile_value_sb: float
    profile_value_bb: float
    profile_sample_variance: float | None
    profile_standard_error: float | None
    board_evaluations: int
    accepted_samples: int | None
    rejected_hole_draws: int
    trace: tuple[EquityTracePoint, ...]
    trace_truncated: bool
    best_responses: tuple[UnilateralBestResponse, ...]


@dataclass(frozen=True)
class PushFoldRunResult:
    """Fail-closed analysis wrapper with no failure payload."""

    status: AiofStatus
    analysis: PushFoldAnalysis | None
    error_message: str | None


@dataclass(frozen=True)
class _GameValues:
    sb_fold: float
    sb_shove_bb_fold: float
    showdown_effective: float
    tolerance: float


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AiofContractError(AiofStatus.INVALID_INPUT, f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise AiofContractError(AiofStatus.INVALID_INPUT, f"{name} must be finite")
    return number


def _validate_game(game: HeadsUpChipEvGame) -> _GameValues:
    if not isinstance(game, HeadsUpChipEvGame):
        raise AiofContractError(AiofStatus.INVALID_INPUT, "game must be HeadsUpChipEvGame")
    sb_stack = _finite_number(game.starting_stack_sb, "starting_stack_sb")
    bb_stack = _finite_number(game.starting_stack_bb, "starting_stack_bb")
    small_blind = _finite_number(game.small_blind, "small_blind")
    big_blind = _finite_number(game.big_blind, "big_blind")
    ante = _finite_number(game.ante, "ante")
    fee = _finite_number(game.fee, "fee")
    dead_money = _finite_number(game.third_party_dead_money, "third_party_dead_money")
    if sb_stack <= 0.0 or bb_stack <= 0.0 or small_blind <= 0.0:
        raise AiofContractError(AiofStatus.INVALID_INPUT, "stacks and small blind must be positive")
    if big_blind < small_blind or ante < 0.0:
        raise AiofContractError(AiofStatus.INVALID_INPUT, "invalid blind or ante ordering")
    if not isinstance(game.side_pot, bool):
        raise AiofContractError(AiofStatus.INVALID_INPUT, "side_pot must be bool")
    if fee != 0.0 or dead_money != 0.0 or game.side_pot:
        raise AiofContractError(AiofStatus.UNSUPPORTED_MODEL, "fee, third-party money, and side pots are unsupported")
    if sb_stack < small_blind + ante or bb_stack < big_blind + ante:
        raise AiofContractError(AiofStatus.UNSUPPORTED_MODEL, "a stack cannot fully cover its mandatory post")
    tolerance = max(1e-9, 1e-12 * max(1.0, sb_stack + bb_stack))
    return _GameValues(-(small_blind + ante), big_blind + ante, min(sb_stack, bb_stack), tolerance)


def _validate_deviation_request(request: PushFoldRequest) -> float:
    tolerance = _finite_number(request.deviation_tolerance, "deviation_tolerance")
    if tolerance < 0.0:
        raise AiofContractError(AiofStatus.INVALID_INPUT, "deviation_tolerance must be non-negative")
    if not isinstance(request.best_response_seats, tuple):
        raise AiofContractError(AiofStatus.INVALID_INPUT, "best_response_seats must be a tuple")
    if len(set(request.best_response_seats)) != len(request.best_response_seats):
        raise AiofContractError(AiofStatus.INVALID_INPUT, "best_response_seats contains duplicates")
    if any(seat not in ("sb", "bb") for seat in request.best_response_seats):
        raise AiofContractError(AiofStatus.INVALID_INPUT, "unknown best-response seat")
    if tuple(seat for seat in ("sb", "bb") if seat in request.best_response_seats) != request.best_response_seats:
        raise AiofContractError(AiofStatus.INVALID_INPUT, "best_response_seats is not canonical")
    if request.best_response_seats and request.algorithm is not EquityAlgorithm.EXACT_EXHAUSTIVE:
        raise AiofContractError(AiofStatus.UNSUPPORTED_MODEL, "Monte Carlo cannot produce exact best responses")
    return tolerance


def _strategy_map(
    entries: tuple[ComboActionProbability, ...], expected: tuple[str, ...], name: str
) -> dict[str, float]:
    if not isinstance(entries, tuple):
        raise AiofContractError(AiofStatus.INVALID_STRATEGY, f"{name} must be a tuple")
    result: dict[str, float] = {}
    for entry in entries:
        if not isinstance(entry, ComboActionProbability):
            raise AiofContractError(AiofStatus.INVALID_STRATEGY, f"{name} contains an invalid entry")
        try:
            combo = canonicalize_exact_combo(entry.combo)
        except AiofContractError as exc:
            raise AiofContractError(AiofStatus.INVALID_STRATEGY, str(exc)) from exc
        try:
            probability = _finite_number(entry.probability, f"{name}[{combo}]")
        except AiofContractError as exc:
            raise AiofContractError(AiofStatus.INVALID_STRATEGY, str(exc)) from exc
        if not 0.0 <= probability <= 1.0:
            raise AiofContractError(AiofStatus.INVALID_STRATEGY, "strategy probability must be in [0, 1]")
        if combo in result:
            raise AiofContractError(AiofStatus.INVALID_STRATEGY, f"duplicate strategy combo {combo}")
        result[combo] = probability
    if set(result) != set(expected):
        missing = sorted(set(expected) - set(result))
        extra = sorted(set(result) - set(expected))
        raise AiofContractError(
            AiofStatus.INVALID_STRATEGY, f"strategy support mismatch; missing={missing}, extra={extra}"
        )
    return result


def _validate_profile(profile: SuppliedProfile, prepared: PreparedRanges) -> tuple[dict[str, float], dict[str, float]]:
    if not isinstance(profile, SuppliedProfile):
        raise AiofContractError(AiofStatus.INVALID_STRATEGY, "profile must be SuppliedProfile")
    sb_expected = tuple(item.combo for item in prepared.sb_marginals)
    bb_expected = tuple(item.combo for item in prepared.bb_marginals)
    return (
        _strategy_map(profile.sb_shove, sb_expected, "sb_shove"),
        _strategy_map(profile.bb_call, bb_expected, "bb_call"),
    )


def _showdown_sb_value(outcome: str, effective: float) -> float:
    return effective if outcome == "win" else (-effective if outcome == "loss" else 0.0)


def _showdown_from_counts(counts: OutcomeCounts, effective: float) -> float:
    return effective * (counts.wins - counts.losses) / counts.trials


def _check_conservation(sb_value: float, bb_value: float, tolerance: float) -> None:
    if not math.isfinite(sb_value) or not math.isfinite(bb_value):
        raise AiofContractError(AiofStatus.NUMERIC_FAILURE, "ChipEV is non-finite")
    if abs(sb_value + bb_value) > tolerance:
        raise AiofContractError(AiofStatus.ACCOUNTING_MISMATCH, "ChipEV conservation failed")


def _check_terminal_accounting(values: _GameValues) -> None:
    for sb_value in (
        values.sb_fold,
        values.sb_shove_bb_fold,
        values.showdown_effective,
        -values.showdown_effective,
        0.0,
    ):
        _check_conservation(sb_value, -sb_value, values.tolerance)


def _profile_sb_value(
    shove_probability: float,
    call_probability: float,
    showdown_value: float,
    values: _GameValues,
) -> float:
    shove_value = (1.0 - call_probability) * values.sb_shove_bb_fold + call_probability * showdown_value
    return (1.0 - shove_probability) * values.sb_fold + shove_probability * shove_value


def _best_actions(
    actions: tuple[str, ...], values: tuple[float, ...], tolerance: float
) -> tuple[str, ...]:
    maximum = max(values)
    return tuple(action for action, value in zip(actions, values) if value >= maximum - tolerance)


def _analysis_identities(
    request: PushFoldRequest,
    prepared: PreparedRanges,
    seed: int | None,
    samples: int | None,
    sb_strategy: dict[str, float],
    bb_strategy: dict[str, float],
) -> tuple[str, str, str]:
    equity_request = EquityRequest(
        request.sb_range,
        request.bb_range,
        request.dead_cards,
        request.algorithm,
        request.limits,
        request.requested_trace_points,
        seed,
        samples,
    )
    equity_input, runtime, _ = _make_identities(equity_request, prepared, seed, samples)
    input_identity = _content_identity(
        {
            "equity_input": equity_input,
            "chip_accounting": CHIP_ACCOUNTING_ID,
            "game": request.game,
            "sb_strategy": sb_strategy,
            "bb_strategy": bb_strategy,
            "best_response_seats": request.best_response_seats,
            "deviation_tolerance": request.deviation_tolerance,
        }
    )
    return input_identity, runtime, _content_identity(
        {"input_identity": input_identity, "runtime_identity": runtime}
    )


def _exact_analysis(
    request: PushFoldRequest,
    prepared: PreparedRanges,
    values: _GameValues,
    sb_strategy: dict[str, float],
    bb_strategy: dict[str, float],
    tolerance: float,
    trace_points: int,
) -> PushFoldAnalysis:
    projected = _projected_exact_evaluations(prepared)
    if projected > request.limits.max_exact_board_evaluations:
        raise AiofContractError(AiofStatus.CAP_EXCEEDED, "projected exact board evaluations exceed cap")
    wins = losses = ties = trials = 0
    weighted_win: list[float] = []
    weighted_loss: list[float] = []
    weighted_tie: list[float] = []
    profile_terms: list[float] = []
    trace: list[EquityTracePoint] = []
    sb_shove_numerators = {item.combo: 0.0 for item in prepared.sb_marginals}
    bb_call_numerators = {item.combo: 0.0 for item in prepared.bb_marginals}
    bb_reach = {item.combo: 0.0 for item in prepared.bb_marginals}
    matchup_index = 0
    for sb_combo, bb_combo, pair_probability, counts in _iter_exact_matchup_outcomes(prepared):
        matchup_index += 1
        wins += counts.wins
        losses += counts.losses
        ties += counts.ties
        trials += counts.trials
        win_probability = counts.wins / counts.trials
        loss_probability = counts.losses / counts.trials
        tie_probability = counts.ties / counts.trials
        weighted_win.append(pair_probability * win_probability)
        weighted_loss.append(pair_probability * loss_probability)
        weighted_tie.append(pair_probability * tie_probability)
        showdown_sb = _showdown_from_counts(counts, values.showdown_effective)
        shove = sb_strategy[sb_combo.combo]
        call = bb_strategy[bb_combo.combo]
        pair_profile = _profile_sb_value(shove, call, showdown_sb, values)
        _check_conservation(pair_profile, -pair_profile, values.tolerance)
        profile_terms.append(pair_probability * pair_profile)
        sb_shove = (1.0 - call) * values.sb_shove_bb_fold + call * showdown_sb
        sb_shove_numerators[sb_combo.combo] += pair_probability * sb_shove
        bb_reach[bb_combo.combo] += pair_probability * shove
        bb_call_numerators[bb_combo.combo] += pair_probability * shove * (-showdown_sb)
        if matchup_index <= trace_points:
            trace.append(
                EquityTracePoint(matchup_index, sb_combo.combo, bb_combo.combo, None, counts, pair_probability, None)
            )
    profile_sb = math.fsum(profile_terms)
    profile_bb = -profile_sb
    _check_conservation(profile_sb, profile_bb, values.tolerance)
    probabilities = OutcomeProbabilities(math.fsum(weighted_win), math.fsum(weighted_loss), math.fsum(weighted_tie))
    if abs(math.fsum((probabilities.win, probabilities.loss, probabilities.tie)) - 1.0) > 1e-10:
        raise AiofContractError(AiofStatus.NUMERIC_FAILURE, "showdown probabilities do not sum to one")

    responses: list[UnilateralBestResponse] = []
    if "sb" in request.best_response_seats:
        rows: list[BestResponseRow] = []
        gains: list[float] = []
        marginal_by_combo = {item.combo: item.probability for item in prepared.sb_marginals}
        for combo in (item.combo for item in prepared.sb_marginals):
            marginal = marginal_by_combo[combo]
            shove_value = sb_shove_numerators[combo] / marginal
            fold_value = values.sb_fold
            best = _best_actions(SB_ACTIONS, (shove_value, fold_value), tolerance)
            supplied = sb_strategy[combo] * shove_value + (1.0 - sb_strategy[combo]) * fold_value
            gain = marginal * (max(shove_value, fold_value) - supplied)
            gains.append(gain)
            rows.append(
                BestResponseRow(
                    combo,
                    marginal,
                    marginal,
                    (("shove", shove_value), ("fold", fold_value)),
                    best,
                    sb_strategy[combo],
                    gain,
                )
            )
        gain = math.fsum(gains)
        responses.append(UnilateralBestResponse("sb", profile_sb, profile_sb + gain, gain, tuple(rows)))
    if "bb" in request.best_response_seats:
        rows = []
        gains = []
        marginal_by_combo = {item.combo: item.probability for item in prepared.bb_marginals}
        for combo in (item.combo for item in prepared.bb_marginals):
            marginal = marginal_by_combo[combo]
            reach = bb_reach[combo]
            if reach == 0.0:
                rows.append(
                    BestResponseRow(
                        combo,
                        marginal,
                        0.0,
                        (("call", None), ("fold", None)),
                        BB_ACTIONS,
                        bb_strategy[combo],
                        0.0,
                    )
                )
                continue
            call_value = bb_call_numerators[combo] / reach
            fold_value = -values.sb_shove_bb_fold
            best = _best_actions(BB_ACTIONS, (call_value, fold_value), tolerance)
            supplied = bb_strategy[combo] * call_value + (1.0 - bb_strategy[combo]) * fold_value
            gain = reach * (max(call_value, fold_value) - supplied)
            gains.append(gain)
            rows.append(
                BestResponseRow(
                    combo,
                    marginal,
                    reach,
                    (("call", call_value), ("fold", fold_value)),
                    best,
                    bb_strategy[combo],
                    gain,
                )
            )
        gain = math.fsum(gains)
        responses.append(UnilateralBestResponse("bb", profile_bb, profile_bb + gain, gain, tuple(rows)))
    input_id, runtime_id, run_id = _analysis_identities(
        request, prepared, None, None, sb_strategy, bb_strategy
    )
    return PushFoldAnalysis(
        request.algorithm,
        CHIP_ACCOUNTING_ID,
        input_id,
        runtime_id,
        run_id,
        prepared.content_identity,
        prepared.compatible_pair_count,
        OutcomeCounts(wins, losses, ties, trials),
        probabilities,
        profile_sb,
        profile_bb,
        None,
        None,
        projected,
        None,
        0,
        tuple(trace),
        prepared.compatible_pair_count > trace_points,
        tuple(responses),
    )


def _monte_carlo_analysis(
    request: PushFoldRequest,
    prepared: PreparedRanges,
    values: _GameValues,
    sb_strategy: dict[str, float],
    bb_strategy: dict[str, float],
    seed: int,
    samples: int,
    trace_points: int,
) -> PushFoldAnalysis:
    attempt_limit = 100 * samples
    if request.limits.max_sampling_attempts is not None:
        attempt_limit = min(attempt_limit, request.limits.max_sampling_attempts)
    wins = losses = ties = 0
    final_attempt = 0
    score_mean = 0.0
    score_m2 = 0.0
    trace: list[EquityTracePoint] = []
    for trial in _iter_monte_carlo_trials(prepared, seed, samples, attempt_limit):
        final_attempt = trial.attempt
        if trial.outcome == "win":
            wins += 1
        elif trial.outcome == "loss":
            losses += 1
        else:
            ties += 1
        showdown = _showdown_sb_value(trial.outcome, values.showdown_effective)
        score = _profile_sb_value(
            sb_strategy[trial.sb_combo.combo], bb_strategy[trial.bb_combo.combo], showdown, values
        )
        _check_conservation(score, -score, values.tolerance)
        delta = score - score_mean
        score_mean += delta / trial.accepted_index
        score_m2 += delta * (score - score_mean)
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
        raise AiofContractError(AiofStatus.NON_REPRODUCIBLE, "Monte Carlo profile sample count mismatch")
    mean = score_mean
    if score_m2 < 0.0:
        raise AiofContractError(AiofStatus.NUMERIC_FAILURE, "negative profile sample variance")
    variance = score_m2 / (samples - 1)
    standard_error = math.sqrt(variance / samples)
    if not all(math.isfinite(value) for value in (mean, variance, standard_error)):
        raise AiofContractError(AiofStatus.NUMERIC_FAILURE, "non-finite profile Monte Carlo statistics")
    _check_conservation(mean, -mean, values.tolerance)
    probabilities = OutcomeProbabilities(wins / samples, losses / samples, ties / samples)
    input_id, runtime_id, run_id = _analysis_identities(
        request, prepared, seed, samples, sb_strategy, bb_strategy
    )
    return PushFoldAnalysis(
        request.algorithm,
        CHIP_ACCOUNTING_ID,
        input_id,
        runtime_id,
        run_id,
        prepared.content_identity,
        prepared.compatible_pair_count,
        OutcomeCounts(wins, losses, ties, samples),
        probabilities,
        mean,
        -mean,
        variance,
        standard_error,
        samples,
        samples,
        final_attempt - samples,
        tuple(trace),
        samples > trace_points,
        (),
    )


def analyze_pushfold(request: PushFoldRequest) -> PushFoldRunResult:
    """Evaluate a supplied profile and optional exact fixed-opponent BRs."""

    try:
        if not isinstance(request, PushFoldRequest):
            raise AiofContractError(AiofStatus.INVALID_INPUT, "request must be PushFoldRequest")
        values = _validate_game(request.game)
        _check_terminal_accounting(values)
        tolerance = _validate_deviation_request(request)
        equity_request = EquityRequest(
            request.sb_range,
            request.bb_range,
            request.dead_cards,
            request.algorithm,
            request.limits,
            request.requested_trace_points,
            request.seed,
            request.samples,
        )
        seed, samples, trace_points = _validate_request(equity_request)
        prepared = prepare_compatible_ranges(
            request.sb_range, request.bb_range, request.dead_cards, request.limits
        )
        sb_strategy, bb_strategy = _validate_profile(request.profile, prepared)
        if request.algorithm is EquityAlgorithm.EXACT_EXHAUSTIVE:
            analysis = _exact_analysis(
                request, prepared, values, sb_strategy, bb_strategy, tolerance, trace_points
            )
        else:
            assert seed is not None and samples is not None
            analysis = _monte_carlo_analysis(
                request, prepared, values, sb_strategy, bb_strategy, seed, samples, trace_points
            )
        return PushFoldRunResult(AiofStatus.SUCCESS, analysis, None)
    except AiofContractError as exc:
        return PushFoldRunResult(exc.status, None, str(exc) or exc.status.value)
    except (ArithmeticError, OverflowError) as exc:
        return PushFoldRunResult(AiofStatus.NUMERIC_FAILURE, None, str(exc) or "numeric failure")
