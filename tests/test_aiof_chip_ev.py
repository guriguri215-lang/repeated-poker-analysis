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
from repeated_poker.aiof_chip_ev import (
    ComboActionProbability,
    HeadsUpChipEvGame,
    PushFoldRequest,
    SuppliedProfile,
    analyze_pushfold,
)
from repeated_poker.aiof_equity import EquityAlgorithm


def exact_range(label):
    return RangeSpec((RangeEntry(label, 1.0, WeightBasis.EXACT_COMBO_MASS),))


def dead_except(live_cards):
    live = {card_id(card) for card in live_cards}
    return tuple(card_from_id(value) for value in range(52) if value not in live)


WIN_BOARD = ("2c", "3d", "4h", "5s", "7c")
TIE_BOARD = ("Ts", "Js", "Qs", "Ks", "As")


def profile(sb_combo, bb_combo, shove, call):
    return SuppliedProfile(
        (ComboActionProbability(sb_combo, shove),),
        (ComboActionProbability(bb_combo, call),),
    )


def game(sb=10.0, bb=10.0, ante=0.0, **kwargs):
    return HeadsUpChipEvGame(sb, bb, 0.5, 1.0, ante, **kwargs)


def make_request(
    sb_combo="AsAh",
    bb_combo="KsKh",
    board=WIN_BOARD,
    shove=0.0,
    call=0.0,
    game_value=None,
    best=(),
    algorithm=EquityAlgorithm.EXACT_EXHAUSTIVE,
    seed=None,
    samples=None,
    limits=None,
    supplied_profile=None,
):
    live = (sb_combo[:2], sb_combo[2:], bb_combo[:2], bb_combo[2:]) + tuple(board)
    return PushFoldRequest(
        exact_range(sb_combo),
        exact_range(bb_combo),
        dead_except(live) if algorithm is EquityAlgorithm.EXACT_EXHAUSTIVE else (),
        algorithm,
        limits or AiofLimits(),
        2,
        game_value or game(),
        supplied_profile or profile(sb_combo, bb_combo, shove, call),
        best,
        0.0,
        seed,
        samples,
    )


@pytest.mark.parametrize(
    ("shove", "call", "sb_combo", "bb_combo", "board", "game_value", "expected"),
    [
        (0.0, 0.0, "AsAh", "KsKh", WIN_BOARD, game(), -0.5),
        (1.0, 0.0, "AsAh", "KsKh", WIN_BOARD, game(), 1.0),
        (1.0, 1.0, "AsAh", "KsKh", WIN_BOARD, game(10.0, 5.0), 5.0),
        (1.0, 1.0, "KsKh", "AsAh", WIN_BOARD, game(10.0, 5.0), -5.0),
        (1.0, 1.0, "2c3d", "4c5d", TIE_BOARD, game(10.0, 5.0), 0.0),
    ],
)
def test_terminal_accounting_equal_unequal_uncalled_excess_and_tie(
    shove, call, sb_combo, bb_combo, board, game_value, expected
):
    result = analyze_pushfold(
        make_request(sb_combo, bb_combo, board, shove, call, game_value)
    )
    assert result.status is AiofStatus.SUCCESS
    analysis = result.analysis
    assert analysis is not None
    assert analysis.profile_value_sb == pytest.approx(expected)
    assert analysis.profile_value_bb == pytest.approx(-expected)
    assert analysis.profile_value_sb + analysis.profile_value_bb == 0.0


def test_ante_is_included_in_mandatory_post_fold_value():
    result = analyze_pushfold(make_request(game_value=game(ante=0.25)))
    assert result.analysis is not None
    assert result.analysis.profile_value_sb == pytest.approx(-0.75)


@pytest.mark.parametrize(
    "unsupported",
    [
        game(sb=0.5, ante=0.1),
        game(fee=0.01),
        game(third_party_dead_money=1.0),
        game(side_pot=True),
    ],
)
def test_unsupported_accounting_models_fail_closed(unsupported):
    result = analyze_pushfold(make_request(game_value=unsupported))
    assert result.status is AiofStatus.UNSUPPORTED_MODEL
    assert result.analysis is None and result.error_message


def test_mixed_profile_matches_hand_calculation_on_known_sb_win_board():
    result = analyze_pushfold(make_request(shove=0.25, call=0.4, game_value=game(10, 5)))
    analysis = result.analysis
    assert result.status is AiofStatus.SUCCESS and analysis is not None
    shove_value = 0.6 * 1.0 + 0.4 * 5.0
    expected = 0.75 * -0.5 + 0.25 * shove_value
    assert analysis.profile_value_sb == pytest.approx(expected)


def test_monte_carlo_profile_uses_conditional_expectation_and_conserves_chips():
    result = analyze_pushfold(
        make_request(
            shove=0.3,
            call=0.7,
            algorithm=EquityAlgorithm.DETERMINISTIC_MONTE_CARLO,
            seed=42,
            samples=100,
        )
    )
    analysis = result.analysis
    assert result.status is AiofStatus.SUCCESS and analysis is not None
    assert analysis.accepted_samples == 100
    assert analysis.profile_value_sb == -analysis.profile_value_bb
    assert math.isfinite(analysis.profile_sample_variance)
    assert math.isfinite(analysis.profile_standard_error)


@pytest.mark.parametrize(
    "bad_profile",
    [
        SuppliedProfile((), (ComboActionProbability("KsKh", 0.0),)),
        SuppliedProfile((ComboActionProbability("AsAh", 0.0),), (ComboActionProbability("QsQh", 0.0),)),
        SuppliedProfile(
            (ComboActionProbability("AsAh", 0.0), ComboActionProbability("AhAs", 1.0)),
            (ComboActionProbability("KsKh", 0.0),),
        ),
        SuppliedProfile((ComboActionProbability("AsAh", True),), (ComboActionProbability("KsKh", 0.0),)),
        SuppliedProfile((ComboActionProbability("AsAh", math.nan),), (ComboActionProbability("KsKh", 0.0),)),
        SuppliedProfile((ComboActionProbability("AsAh", 1.1),), (ComboActionProbability("KsKh", 0.0),)),
    ],
)
def test_strategy_support_and_probability_failures_return_no_payload(bad_profile):
    result = analyze_pushfold(make_request(supplied_profile=bad_profile))
    assert result.status is AiofStatus.INVALID_STRATEGY
    assert result.analysis is None


def test_fixed_bb_folds_makes_sb_shove_exact_best_response():
    result = analyze_pushfold(make_request(shove=0.0, call=0.0, best=("sb",)))
    response = result.analysis.best_responses[0]
    assert response.seat == "sb"
    assert response.rows[0].best_actions == ("shove",)
    assert response.raw_gain == pytest.approx(1.5)


def test_fixed_bb_calls_and_sb_loses_makes_sb_fold_best_response():
    result = analyze_pushfold(
        make_request("KsKh", "AsAh", WIN_BOARD, shove=1.0, call=1.0, best=("sb",))
    )
    assert result.analysis.best_responses[0].rows[0].best_actions == ("fold",)


def test_fixed_sb_shoves_and_bb_wins_makes_bb_call_best_response():
    result = analyze_pushfold(
        make_request("KsKh", "AsAh", WIN_BOARD, shove=1.0, call=0.0, best=("bb",))
    )
    assert result.analysis.best_responses[0].rows[0].best_actions == ("call",)


def test_exact_bb_tie_retains_full_response_set():
    result = analyze_pushfold(
        make_request(
            "AsAh",
            "KsKh",
            WIN_BOARD,
            shove=1.0,
            call=0.0,
            game_value=game(10.0, 1.0),
            best=("bb",),
        )
    )
    row = result.analysis.best_responses[0].rows[0]
    assert row.best_actions == ("call", "fold")


def test_unreachable_bb_information_set_has_none_values_full_set_and_zero_gain():
    result = analyze_pushfold(make_request(shove=0.0, call=0.2, best=("bb",)))
    response = result.analysis.best_responses[0]
    row = response.rows[0]
    assert row.action_values == (("call", None), ("fold", None))
    assert row.best_actions == ("call", "fold")
    assert row.raw_gain == response.raw_gain == 0.0


def test_monte_carlo_best_response_request_is_unsupported_without_partial_analysis():
    result = analyze_pushfold(
        make_request(
            algorithm=EquityAlgorithm.DETERMINISTIC_MONTE_CARLO,
            seed=42,
            samples=10,
            best=("sb",),
        )
    )
    assert result.status is AiofStatus.UNSUPPORTED_MODEL
    assert result.analysis is None


def test_exact_cap_failure_returns_no_profile_or_best_response_payload():
    result = analyze_pushfold(
        make_request(
            board=WIN_BOARD,
            best=("sb",),
            limits=AiofLimits(max_exact_board_evaluations=0),
        )
    )
    assert result.status is AiofStatus.CAP_EXCEEDED
    assert result.analysis is None and result.error_message
