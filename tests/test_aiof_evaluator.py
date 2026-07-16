import pytest

from repeated_poker.aiof_cards import AiofContractError, AiofStatus
from repeated_poker.aiof_evaluator import (
    HandCategory,
    evaluate_five_card_hand,
    evaluate_seven_card_hand,
)


def hand(text):
    return tuple(text.split())


def test_all_nine_categories_have_contract_order():
    examples = (
        "As Jd 9h 6c 3s",
        "As Ad 9h 6c 3s",
        "As Ad 9h 9c 3s",
        "As Ad Ah 6c 3s",
        "2s 3d 4h 5c 6s",
        "As Js 9s 6s 3s",
        "As Ad Ah 6c 6s",
        "As Ad Ah Ac 3s",
        "2s 3s 4s 5s 6s",
    )
    ranks = tuple(evaluate_five_card_hand(hand(value)) for value in examples)
    assert tuple(rank.category for rank in ranks) == tuple(HandCategory)
    assert all(left < right for left, right in zip(ranks, ranks[1:]))


@pytest.mark.parametrize(
    ("stronger", "weaker"),
    [
        ("As Qd 9h 6c 3s", "As Jd 9h 6c 3s"),
        ("As Ad Qh 6c 3s", "As Ad Jh 6c 3s"),
        ("As Ad Kh Kc Qs", "As Ad Qh Qc Ks"),
        ("As Ad Ah Qc 3s", "As Ad Ah Jc 9s"),
        ("Ks Kd Kh 2c 2s", "Qs Qd Qh Ac As"),
        ("As Ad Ah Ac Ks", "As Ad Ah Ac Qs"),
        ("As Qs 9s 6s 3s", "As Js 9s 6s 3s"),
    ],
)
def test_category_tiebreak_vectors_are_complete(stronger, weaker):
    assert evaluate_five_card_hand(hand(stronger)) > evaluate_five_card_hand(hand(weaker))


def test_wheel_and_ace_high_straights():
    wheel = evaluate_five_card_hand(hand("As 2d 3h 4c 5s"))
    six_high = evaluate_five_card_hand(hand("2s 3d 4h 5c 6s"))
    ace_high = evaluate_five_card_hand(hand("Ts Jd Qh Kc As"))
    assert wheel.tiebreak == (5,)
    assert wheel < six_high < ace_high
    assert evaluate_five_card_hand(hand("As 2s 3s 4s 5s")).category is HandCategory.STRAIGHT_FLUSH


def test_suit_permutation_does_not_change_rank():
    assert evaluate_five_card_hand(hand("As Ad Qh 8c 3s")) == evaluate_five_card_hand(
        hand("Ah Ac Qs 8d 3h")
    )


@pytest.mark.parametrize(
    ("cards", "category", "tiebreak"),
    [
        ("2c 3d 4h 8s 5c 6d Kh", HandCategory.STRAIGHT, (6,)),
        ("As Ad 2c 3d 4h 5s Kc", HandCategory.STRAIGHT, (5,)),
        ("As Ad Ah Kc Kd Kh 2s", HandCategory.FULL_HOUSE, (14, 13)),
        ("As Qs 9s 6s 3s 2d Kh", HandCategory.FLUSH, (14, 12, 9, 6, 3)),
        ("As Ad Ah Ac Ks Qd 2c", HandCategory.FOUR_OF_A_KIND, (14, 13)),
    ],
)
def test_seven_card_best_of_21_known_oracles(cards, category, tiebreak):
    result = evaluate_seven_card_hand(hand(cards))
    assert result.category is category
    assert result.tiebreak == tiebreak


def test_public_evaluators_reject_length_duplicate_and_invalid_card():
    with pytest.raises(AiofContractError) as error:
        evaluate_five_card_hand(hand("As Ks Qs Js"))
    assert error.value.status is AiofStatus.INVALID_CARD_INPUT
    with pytest.raises(AiofContractError):
        evaluate_seven_card_hand(hand("As As Qs Js Ts 9s 8s"))
    with pytest.raises(AiofContractError):
        evaluate_five_card_hand(("As", "Ks", "Qs", "Js", "10s"))
