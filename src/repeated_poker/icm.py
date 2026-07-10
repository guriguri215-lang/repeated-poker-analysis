"""Independent Chip Model prize-equity calculations.

This module implements Malmuth-Harville ICM as a stack-to-prize-equity model.
It assigns tournament prize equity from chip stacks only: it ignores position,
blind increases, skill edges, future hands, and all dynamics outside the given
stack vector. The result is modelled tournament prize EV, not a real tournament
prediction or a strategy recommendation.
"""

from __future__ import annotations

import math
from typing import List, Sequence

from .game import require_finite

DEFAULT_MAX_ICM_ORDERINGS = 1_000_000


def calculate_icm_equities(
    stacks: Sequence[float],
    prizes: Sequence[float],
    *,
    max_orderings: int = DEFAULT_MAX_ICM_ORDERINGS,
) -> List[float]:
    """Return Malmuth-Harville ICM prize equities for ``stacks``.

    ``stacks`` must contain at least two finite non-negative chip stacks and at
    least one positive stack. ``prizes`` must be a non-empty, finite,
    non-negative, non-increasing payout vector no longer than ``stacks``. Missing
    trailing prizes are treated as zero.

    Players with a zero stack are treated as occupying the bottom remaining
    places; when several zero-stack players tie, they split those lower prizes
    equally. Positive-stack players receive the remaining top prizes according to
    the standard Malmuth-Harville recursive finish-order probabilities.

    The calculation enumerates positive-stack finish prefixes with a safety
    guard. ``max_orderings`` is checked before recursion and raises
    :class:`ValueError` if the required number of orderings would exceed it.
    """

    stack_values = _validate_stacks(stacks)
    prize_values = _validate_prizes(prizes, len(stack_values))
    _validate_max_orderings(max_orderings)

    n = len(stack_values)
    positive_indices = [idx for idx, stack in enumerate(stack_values) if stack > 0]
    zero_indices = [idx for idx, stack in enumerate(stack_values) if stack == 0]
    positive_count = len(positive_indices)
    paid_positive_places = min(len(prize_values), positive_count)

    _validate_ordering_count(
        player_count=positive_count,
        place_count=paid_positive_places,
        max_orderings=max_orderings,
    )

    full_prizes = list(prize_values) + [0.0] * (n - len(prize_values))
    equities = [0.0 for _ in stack_values]

    if zero_indices:
        bottom_prizes = full_prizes[positive_count:n]
        zero_equity = math.fsum(bottom_prizes) / len(zero_indices)
        for idx in zero_indices:
            equities[idx] = zero_equity

    if paid_positive_places == 0:
        return equities

    remaining_stacks = {idx: stack_values[idx] for idx in positive_indices}
    _add_positive_stack_equities(
        equities=equities,
        prizes=full_prizes,
        remaining_indices=tuple(positive_indices),
        remaining_stacks=remaining_stacks,
        place=0,
        max_places=paid_positive_places,
        path_probability=1.0,
    )
    return equities


def _validate_stacks(stacks: Sequence[float]) -> List[float]:
    if isinstance(stacks, (str, bytes)):
        raise ValueError("stacks must be a sequence of numbers")
    try:
        values = list(stacks)
    except TypeError as exc:
        raise ValueError("stacks must be a sequence of numbers") from exc
    if len(values) < 2:
        raise ValueError("stacks must contain at least two players")

    result: List[float] = []
    for idx, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"stacks[{idx}] must be a number, got {value!r}")
        number = float(value)
        require_finite(number, f"stacks[{idx}]")
        if number < 0:
            raise ValueError(f"stacks[{idx}] must be non-negative, got {value!r}")
        result.append(number)
    if not any(value > 0 for value in result):
        raise ValueError("stacks must contain at least one positive stack")
    return result


def _validate_prizes(prizes: Sequence[float], player_count: int) -> List[float]:
    if isinstance(prizes, (str, bytes)):
        raise ValueError("prizes must be a sequence of numbers")
    try:
        values = list(prizes)
    except TypeError as exc:
        raise ValueError("prizes must be a sequence of numbers") from exc
    if not values:
        raise ValueError("prizes must be non-empty")
    if len(values) > player_count:
        raise ValueError(
            f"prizes length must be at most number of players ({player_count})"
        )

    result: List[float] = []
    previous = math.inf
    for idx, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"prizes[{idx}] must be a number, got {value!r}")
        number = float(value)
        require_finite(number, f"prizes[{idx}]")
        if number < 0:
            raise ValueError(f"prizes[{idx}] must be non-negative, got {value!r}")
        if number > previous:
            raise ValueError("prizes must be non-increasing")
        result.append(number)
        previous = number
    return result


def _validate_max_orderings(max_orderings: int) -> None:
    if isinstance(max_orderings, bool) or not isinstance(max_orderings, int):
        raise ValueError(f"max_orderings must be an int, got {max_orderings!r}")
    if max_orderings < 1:
        raise ValueError(
            f"max_orderings must be at least 1, got {max_orderings!r}"
        )


def _validate_ordering_count(
    *, player_count: int, place_count: int, max_orderings: int
) -> None:
    ordering_count = 1
    for offset in range(place_count):
        ordering_count *= player_count - offset
        if ordering_count > max_orderings:
            raise ValueError(
                "ICM ordering count "
                f"{ordering_count} exceeds max_orderings={max_orderings}"
            )


def _add_positive_stack_equities(
    *,
    equities: List[float],
    prizes: Sequence[float],
    remaining_indices: Sequence[int],
    remaining_stacks: dict[int, float],
    place: int,
    max_places: int,
    path_probability: float,
) -> None:
    prize = prizes[place]
    maximum_stack = max(remaining_stacks[idx] for idx in remaining_indices)
    scaled_total = math.fsum(
        remaining_stacks[idx] / maximum_stack for idx in remaining_indices
    )
    for idx in remaining_indices:
        finish_probability = (remaining_stacks[idx] / maximum_stack) / scaled_total
        branch_probability = path_probability * finish_probability
        equities[idx] += branch_probability * prize

        next_place = place + 1
        if next_place >= max_places:
            continue

        next_remaining = tuple(
            remaining_idx for remaining_idx in remaining_indices if remaining_idx != idx
        )
        _add_positive_stack_equities(
            equities=equities,
            prizes=prizes,
            remaining_indices=next_remaining,
            remaining_stacks=remaining_stacks,
            place=next_place,
            max_places=max_places,
            path_probability=branch_probability,
        )
