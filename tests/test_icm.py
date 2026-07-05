import math

import pytest

from repeated_poker import DEFAULT_MAX_ICM_ORDERINGS, calculate_icm_equities


def assert_equities_close(actual, expected, *, tolerance=1e-9):
    assert actual == pytest.approx(expected, abs=tolerance)


def test_icm_known_three_player_oracle():
    equities = calculate_icm_equities((50, 30, 20), (50, 30, 20))

    assert_equities_close(equities, (1075 / 28, 131 / 4, 202 / 7))


def test_icm_equal_stacks_split_prize_pool_symmetrically():
    equities = calculate_icm_equities((25, 25, 25, 25), (50, 30, 20))

    assert_equities_close(equities, (25, 25, 25, 25))


def test_icm_winner_take_all_matches_chip_share():
    equities = calculate_icm_equities((50, 30, 20), (100,))

    assert_equities_close(equities, (50, 30, 20))


@pytest.mark.parametrize(
    "stacks, prizes, expected",
    [
        ((60, 0, 40), (50, 30, 20), (42, 20, 38)),
        ((100, 0, 0), (50, 30, 20), (50, 25, 25)),
    ],
)
def test_icm_zero_stack_players_take_bottom_prizes(stacks, prizes, expected):
    equities = calculate_icm_equities(stacks, prizes)

    assert_equities_close(equities, expected)


@pytest.mark.parametrize(
    "stacks, prizes",
    [
        ((50, 30, 20), (50, 30, 20)),
        ((100, 50, 25, 25), (60, 25, 15)),
        ((60, 0, 40), (50, 30, 20)),
        ((100, 0, 0), (50, 30, 20)),
        ((20, 20, 20, 20), (40, 30, 20, 10)),
        ((20, 10, 5, 0), (70, 20)),
    ],
)
def test_icm_equities_conserve_prize_pool(stacks, prizes):
    equities = calculate_icm_equities(stacks, prizes)

    assert math.fsum(equities) == pytest.approx(math.fsum(prizes), abs=1e-9)


@pytest.mark.parametrize(
    "stacks",
    [
        (100, 50, 25),
        (100, 75, 25, 25),
        (1, 0, 0),
    ],
)
def test_icm_is_monotone_in_stacks_for_fixed_prizes(stacks):
    prizes = tuple(reversed(range(10, 10 * (len(stacks) + 1), 10)))
    equities = calculate_icm_equities(stacks, prizes)

    for i, stack_i in enumerate(stacks):
        for j, stack_j in enumerate(stacks):
            if stack_i > stack_j:
                assert equities[i] >= equities[j]


@pytest.mark.parametrize(
    "stacks, prizes",
    [
        ((), (50,)),
        ((100,), (50,)),
        ((0, 0), (50,)),
        ((100, -1), (50,)),
        ((100, math.nan), (50,)),
        ((100, math.inf), (50,)),
        ((100, True), (50,)),
        ((100, "50"), (50,)),
        ((100, 50), ()),
        ((100, 50), (50, 30, 20)),
        ((100, 50), (50, -1)),
        ((100, 50), (50, math.nan)),
        ((100, 50), (50, math.inf)),
        ((100, 50), (50, True)),
        ((100, 50), (50, "30")),
        ((100, 50), (30, 50)),
    ],
)
def test_icm_rejects_invalid_stacks_and_prizes(stacks, prizes):
    with pytest.raises(ValueError):
        calculate_icm_equities(stacks, prizes)


@pytest.mark.parametrize("max_orderings", [0, -1, 1.5, True, "10"])
def test_icm_rejects_invalid_max_orderings(max_orderings):
    with pytest.raises(ValueError):
        calculate_icm_equities((50, 30, 20), (50, 30), max_orderings=max_orderings)


def test_icm_rejects_ordering_count_above_guard():
    with pytest.raises(ValueError, match="exceeds max_orderings"):
        calculate_icm_equities((50, 30, 20), (50, 30), max_orderings=5)


def test_icm_default_max_orderings_is_exported():
    assert DEFAULT_MAX_ICM_ORDERINGS == 1_000_000
