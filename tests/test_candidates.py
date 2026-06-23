"""Tests for Hero candidate-library generation by probability shifts."""

import copy

import pytest

from nuts_chop_river import build_nuts_chop_river, default_hero_strategy

from repeated_poker import generate_shift_candidates, validate_hero_strategy


def test_generated_candidates_are_valid_and_sum_to_one():
    tree = build_nuts_chop_river()
    candidates = generate_shift_candidates(tree, default_hero_strategy(), [0.2])

    assert candidates  # non-empty
    for candidate in candidates:
        # A proper Hero strategy: validation enforces probabilities summing to 1.
        validate_hero_strategy(tree, candidate.hero_strategy)
        for dist in candidate.hero_strategy.probabilities.values():
            assert sum(dist.values()) == pytest.approx(1.0)


def test_generation_does_not_mutate_baseline():
    tree = build_nuts_chop_river()
    baseline = default_hero_strategy()
    snapshot = copy.deepcopy(baseline.probabilities)

    generate_shift_candidates(tree, baseline, [0.1, 0.2])

    assert baseline.probabilities == snapshot


def test_expected_candidate_count_and_no_impossible_shift():
    tree = build_nuts_chop_river()
    candidates = generate_shift_candidates(tree, default_hero_strategy(), [0.2])

    # H1 (check<->bet): 2 candidates. H_vs_bet and H_vs_raise: only call->fold is
    # possible (fold has baseline probability 0), so 1 each. Total 4.
    assert len(candidates) == 4

    # No candidate shifts from an action whose baseline probability is below the
    # shift amount (here, "fold" at H_vs_bet / H_vs_raise has probability 0).
    for candidate in candidates:
        if candidate.info_set in ("H_vs_bet", "H_vs_raise"):
            assert candidate.source_action == "call"


def test_source_and_target_always_differ():
    tree = build_nuts_chop_river()
    candidates = generate_shift_candidates(tree, default_hero_strategy(), [0.1, 0.2])
    for candidate in candidates:
        assert candidate.source_action != candidate.target_action


def test_duplicate_shift_amounts_do_not_duplicate_candidates():
    tree = build_nuts_chop_river()
    once = generate_shift_candidates(tree, default_hero_strategy(), [0.2])
    twice = generate_shift_candidates(tree, default_hero_strategy(), [0.2, 0.2])
    assert len(once) == len(twice)


def test_l1_distance_is_twice_the_shift():
    tree = build_nuts_chop_river()
    candidates = generate_shift_candidates(tree, default_hero_strategy(), [0.2])
    for candidate in candidates:
        assert candidate.l1_distance == pytest.approx(0.4)


@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan"), float("inf")])
def test_non_positive_or_non_finite_shift_is_rejected(bad):
    tree = build_nuts_chop_river()
    with pytest.raises(ValueError):
        generate_shift_candidates(tree, default_hero_strategy(), [bad])
