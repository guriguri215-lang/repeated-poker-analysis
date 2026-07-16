"""Deterministic repository-authored Texas Hold'em reference evaluator.

The seven-card path deliberately streams all 21 five-card subsets.  It is a
clarity-first correctness reference, not a lookup-table accelerator and not a
claim about strategy quality.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations

from .aiof_cards import AiofContractError, AiofStatus, card_from_id, card_id


__all__ = [
    "HandCategory",
    "HandRank",
    "evaluate_five_card_hand",
    "evaluate_seven_card_hand",
]


EVALUATOR_ID = "repo-reference-best5of7-v1"


class HandCategory(IntEnum):
    """Texas Hold'em hand categories in increasing strength order."""

    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8


@dataclass(frozen=True, order=True)
class HandRank:
    """Completely ordered category and category-local tie-break vector."""

    category: HandCategory
    tiebreak: tuple[int, ...]


def _straight_high(ranks: set[int]) -> int | None:
    if 14 in ranks:
        ranks = ranks | {1}
    ordered = sorted(ranks)
    run = 1
    best: int | None = None
    for previous, current in zip(ordered, ordered[1:]):
        if current == previous + 1:
            run += 1
            if run >= 5:
                best = current
        elif current != previous:
            run = 1
    return 5 if best == 5 else best


def _evaluate_five_ids(cards: tuple[int, ...]) -> HandRank:
    ranks = tuple(value // 4 + 2 for value in cards)
    suits = tuple(value % 4 for value in cards)
    counts = Counter(ranks)
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    flush = len(set(suits)) == 1
    straight_high = _straight_high(set(ranks))
    if flush and straight_high is not None:
        return HandRank(HandCategory.STRAIGHT_FLUSH, (straight_high,))
    if groups[0][0] == 4:
        quad = groups[0][1]
        kicker = max(rank for rank in ranks if rank != quad)
        return HandRank(HandCategory.FOUR_OF_A_KIND, (quad, kicker))
    if groups[0][0] == 3 and groups[1][0] == 2:
        return HandRank(HandCategory.FULL_HOUSE, (groups[0][1], groups[1][1]))
    descending = tuple(sorted(ranks, reverse=True))
    if flush:
        return HandRank(HandCategory.FLUSH, descending)
    if straight_high is not None:
        return HandRank(HandCategory.STRAIGHT, (straight_high,))
    if groups[0][0] == 3:
        trip = groups[0][1]
        kickers = tuple(sorted((rank for rank in ranks if rank != trip), reverse=True))
        return HandRank(HandCategory.THREE_OF_A_KIND, (trip,) + kickers)
    pair_ranks = sorted((rank for rank, count in counts.items() if count == 2), reverse=True)
    if len(pair_ranks) == 2:
        kicker = max(rank for rank, count in counts.items() if count == 1)
        return HandRank(HandCategory.TWO_PAIR, (pair_ranks[0], pair_ranks[1], kicker))
    if len(pair_ranks) == 1:
        pair = pair_ranks[0]
        kickers = tuple(sorted((rank for rank in ranks if rank != pair), reverse=True))
        return HandRank(HandCategory.ONE_PAIR, (pair,) + kickers)
    return HandRank(HandCategory.HIGH_CARD, descending)


def _evaluate_seven_ids(cards: tuple[int, ...]) -> HandRank:
    return max(_evaluate_five_ids(subset) for subset in combinations(tuple(sorted(cards)), 5))


def _validated_ids(cards: tuple[str, ...], expected: int) -> tuple[int, ...]:
    if not isinstance(cards, tuple) or len(cards) != expected:
        raise AiofContractError(
            AiofStatus.INVALID_CARD_INPUT, f"expected exactly {expected} cards"
        )
    ids = tuple(card_id(card) for card in cards)
    if len(set(ids)) != expected:
        raise AiofContractError(AiofStatus.INVALID_CARD_INPUT, "cards must be distinct")
    return ids


def evaluate_five_card_hand(cards: tuple[str, ...]) -> HandRank:
    """Evaluate exactly five distinct strict card IDs using the reference order."""

    return _evaluate_five_ids(_validated_ids(cards, 5))


def evaluate_seven_card_hand(cards: tuple[str, ...]) -> HandRank:
    """Evaluate exactly seven cards by streaming all 21 five-card subsets."""

    return _evaluate_seven_ids(_validated_ids(cards, 7))
