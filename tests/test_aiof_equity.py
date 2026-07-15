import math

import pytest

from repeated_poker.aiof_cards import (
    AiofLimits,
    AiofStatus,
    RangeEntry,
    RangeSpec,
    WeightBasis,
    card_from_id,
    card_id,
)
from repeated_poker.aiof_equity import (
    EquityAlgorithm,
    EquityRequest,
    _Pcg32,
    calculate_preflop_equity,
)


def exact_range(*labels):
    return RangeSpec(tuple(RangeEntry(label, 1.0, WeightBasis.EXACT_COMBO_MASS) for label in labels))


def dead_except(live_cards):
    live = {card_id(card) for card in live_cards}
    return tuple(card_from_id(value) for value in range(52) if value not in live)


def request(sb, bb, dead, *, limits=None, trace=0, seed=None, samples=None, algorithm=EquityAlgorithm.EXACT_EXHAUSTIVE):
    return EquityRequest(
        sb,
        bb,
        tuple(dead),
        algorithm,
        limits or AiofLimits(),
        trace,
        seed,
        samples,
    )


@pytest.mark.parametrize(
    ("sb", "bb", "board", "expected"),
    [
        ("AsAh", "KsKh", ("2c", "3d", "4h", "5s", "7c"), (1, 0, 0)),
        ("KsKh", "AsAh", ("2c", "3d", "4h", "5s", "7c"), (0, 1, 0)),
        ("2c3d", "4c5d", ("Ts", "Js", "Qs", "Ks", "As"), (0, 0, 1)),
    ],
)
def test_dead_43_exact_known_win_loss_and_tie(sb, bb, board, expected):
    dead = dead_except((sb[:2], sb[2:], bb[:2], bb[2:]) + board)
    result = calculate_preflop_equity(request(exact_range(sb), exact_range(bb), dead, trace=1))
    assert result.status is AiofStatus.SUCCESS
    assert result.error_message is None
    estimate = result.estimate
    assert estimate is not None
    assert (estimate.unweighted_counts.wins, estimate.unweighted_counts.losses, estimate.unweighted_counts.ties) == expected
    assert estimate.unweighted_counts.trials == 1
    assert estimate.board_evaluations == 1
    assert len(estimate.trace) == 1


def test_dead_42_enumerates_six_boards_and_equity_formula_matches_counts():
    live = ("As", "Ah", "Ks", "Kh", "2c", "3d", "4h", "5s", "7c", "8d")
    result = calculate_preflop_equity(request(exact_range("AsAh"), exact_range("KsKh"), dead_except(live)))
    estimate = result.estimate
    assert result.status is AiofStatus.SUCCESS and estimate is not None
    counts = estimate.unweighted_counts
    assert counts.trials == counts.wins + counts.losses + counts.ties == math.comb(6, 5)
    assert estimate.sb_equity == pytest.approx((2 * counts.wins + counts.ties) / (2 * counts.trials))


def test_weighted_two_combo_range_uses_joint_mass_not_unweighted_counts():
    sb = RangeSpec(
        (
            RangeEntry("AsAh", 3.0, WeightBasis.EXACT_COMBO_MASS),
            RangeEntry("KsKh", 1.0, WeightBasis.EXACT_COMBO_MASS),
        )
    )
    bb = exact_range("QsQh")
    live = ("As", "Ah", "Ks", "Kh", "Qs", "Qh", "2c", "3d", "4h", "5s", "7c")
    result = calculate_preflop_equity(request(sb, bb, dead_except(live)))
    estimate = result.estimate
    assert result.status is AiofStatus.SUCCESS and estimate is not None
    assert estimate.compatible_pair_count == 2
    assert estimate.board_evaluations == 2 * math.comb(7, 5) == 42
    assert estimate.probabilities.win + estimate.probabilities.loss + estimate.probabilities.tie == pytest.approx(1.0)


def test_player_swap_complements_equity_and_tie_is_symmetric():
    board = ("2c", "3d", "4h", "5s", "7c")
    live = ("As", "Ah", "Ks", "Kh") + board
    dead = dead_except(live)
    forward = calculate_preflop_equity(request(exact_range("AsAh"), exact_range("KsKh"), dead)).estimate
    reverse = calculate_preflop_equity(request(exact_range("KsKh"), exact_range("AsAh"), dead)).estimate
    assert forward is not None and reverse is not None
    assert forward.sb_equity + reverse.sb_equity == pytest.approx(1.0)
    assert forward.probabilities.tie == reverse.probabilities.tie


def test_suit_permutation_invariance_for_tiny_exact_fixture():
    live = ("As", "Ah", "Ks", "Kh", "2c", "3d", "4h", "5s", "7c")
    first = calculate_preflop_equity(
        request(exact_range("AsAh"), exact_range("KsKh"), dead_except(live))
    ).estimate
    translate = str.maketrans("cdhs", "dhsc")
    mapped = tuple(card[0] + card[1].translate(translate) for card in live)
    second = calculate_preflop_equity(
        request(exact_range("AcAs"), exact_range("KcKs"), dead_except(mapped))
    ).estimate
    assert first is not None and second is not None
    assert first.sb_equity == second.sb_equity


def test_projected_no_dead_single_pair_cap_is_checked_before_evaluation_and_no_fallback():
    limits = AiofLimits(max_exact_board_evaluations=1_712_303)
    result = calculate_preflop_equity(request(exact_range("AsAh"), exact_range("KsKh"), (), limits=limits))
    assert result.status is AiofStatus.CAP_EXCEEDED
    assert result.estimate is None
    assert "1712304" in result.error_message


def test_pcg32_seed_42_golden_vector():
    rng = _Pcg32(42)
    assert [f"{rng.next_u32():08x}" for _ in range(5)] == [
        "a15c02b7",
        "7b47f409",
        "ba1d3330",
        "83d2f293",
        "bfa4784b",
    ]


def test_monte_carlo_same_seed_reproduces_counts_trace_and_statistics():
    kwargs = dict(
        algorithm=EquityAlgorithm.DETERMINISTIC_MONTE_CARLO,
        seed=42,
        samples=200,
        trace=5,
    )
    # Exact-combo supports keep this fixture small while still sampling boards.
    first = calculate_preflop_equity(request(exact_range("AsAh"), exact_range("KsKh"), (), **kwargs))
    second = calculate_preflop_equity(request(exact_range("AsAh"), exact_range("KsKh"), (), **kwargs))
    assert first.status is second.status is AiofStatus.SUCCESS
    assert first.estimate == second.estimate
    estimate = first.estimate
    assert estimate is not None
    counts = estimate.unweighted_counts
    assert counts.trials == counts.wins + counts.losses + counts.ties == 200
    mean = (counts.wins + 0.5 * counts.ties) / 200
    independent_variance = (
        counts.wins * (1 - mean) ** 2
        + counts.losses * mean**2
        + counts.ties * (0.5 - mean) ** 2
    ) / 199
    assert estimate.sb_equity == mean
    assert estimate.sample_variance == pytest.approx(independent_variance)
    assert estimate.standard_error == pytest.approx(math.sqrt(independent_variance / 200))
    assert math.isfinite(estimate.standard_error)


def test_different_monte_carlo_seed_changes_run_identity():
    common = dict(algorithm=EquityAlgorithm.DETERMINISTIC_MONTE_CARLO, samples=20)
    first = calculate_preflop_equity(request(exact_range("AsAh"), exact_range("KsKh"), (), seed=1, **common)).estimate
    second = calculate_preflop_equity(request(exact_range("AsAh"), exact_range("KsKh"), (), seed=2, **common)).estimate
    assert first is not None and second is not None
    assert first.run_identity != second.run_identity


@pytest.mark.parametrize(("seed", "samples"), [(None, 10), (True, 10), (1, 1), (1, 11)])
def test_invalid_monte_carlo_seed_and_sample_limits_fail_closed(seed, samples):
    limits = AiofLimits(max_monte_carlo_samples=10)
    result = calculate_preflop_equity(
        request(
            exact_range("AsAh"),
            exact_range("KsKh"),
            (),
            algorithm=EquityAlgorithm.DETERMINISTIC_MONTE_CARLO,
            seed=seed,
            samples=samples,
            limits=limits,
        )
    )
    assert result.status is not AiofStatus.SUCCESS
    assert result.estimate is None and result.error_message


def test_trace_cap_and_exact_seed_mix_are_rejected():
    assert calculate_preflop_equity(
        request(exact_range("AsAh"), exact_range("KsKh"), (), limits=AiofLimits(max_trace_points=1), trace=2)
    ).status is AiofStatus.CAP_EXCEEDED
    assert calculate_preflop_equity(
        request(exact_range("AsAh"), exact_range("KsKh"), (), seed=42)
    ).status is AiofStatus.INVALID_INPUT


def test_overlap_heavy_sampling_attempt_cap_returns_no_partial_estimate():
    range_spec = RangeSpec(
        (
            RangeEntry("AsAh", 1e100, WeightBasis.EXACT_COMBO_MASS),
            RangeEntry("KsKh", 1.0, WeightBasis.EXACT_COMBO_MASS),
        )
    )
    result = calculate_preflop_equity(
        request(
            range_spec,
            range_spec,
            (),
            algorithm=EquityAlgorithm.DETERMINISTIC_MONTE_CARLO,
            seed=42,
            samples=2,
            limits=AiofLimits(max_sampling_attempts=2),
        )
    )
    assert result.status is AiofStatus.SAMPLING_ATTEMPT_CAP_EXCEEDED
    assert result.estimate is None and result.error_message
