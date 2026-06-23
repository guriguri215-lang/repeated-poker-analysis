"""Finite Hero candidate-library generation by single-action probability shifts.

A candidate is built from the baseline Hero strategy by moving ``shift_amount``
of probability from one ``source_action`` to one ``target_action`` at a single
Hero information set.  Candidates are generated systematically over every Hero
information set, every ordered legal ``(source, target)`` pair, and every
requested shift amount.

The ``l1_distance`` carried by each candidate is the L1 distance between the
candidate and the baseline *strategy vectors*, i.e. a distance in strategy
space.  It is **not** an observable behavioural distance: it does not account
for reach probabilities, showdown frequency, hidden cards, or any opponent
observation model.  Observable-distance measures are a separate, later concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .game import (
    Action,
    GameTree,
    HeroStrategy,
    InfoSetId,
    collect_hero_info_sets,
    require_finite,
    validate_hero_strategy,
)

# Rounding used only to canonicalise resulting strategies for de-duplication.
_DEDUP_DECIMALS = 12


@dataclass(frozen=True)
class HeroStrategyCandidate:
    """One Hero candidate produced by a single-action probability shift.

    ``l1_distance`` is the strategy-space L1 distance from the baseline (the sum
    of absolute per-action probability changes), not an observable behavioural
    distance.
    """

    candidate_id: str
    info_set: InfoSetId
    source_action: Action
    target_action: Action
    shift_amount: float
    hero_strategy: HeroStrategy
    l1_distance: float


def _copy_probabilities(
    probabilities: Dict[InfoSetId, Dict[Action, float]]
) -> Dict[InfoSetId, Dict[Action, float]]:
    return {info_set: dict(dist) for info_set, dist in probabilities.items()}


def _canonical_key(probabilities: Dict[InfoSetId, Dict[Action, float]]):
    return tuple(
        (
            info_set,
            tuple(
                (action, round(prob, _DEDUP_DECIMALS))
                for action, prob in sorted(probabilities[info_set].items())
            ),
        )
        for info_set in sorted(probabilities)
    )


def generate_shift_candidates(
    tree: GameTree,
    baseline_hero_strategy: HeroStrategy,
    shift_amounts: Sequence[float],
    tolerance: float = 1e-9,
) -> List[HeroStrategyCandidate]:
    """Generate single-shift Hero candidates from a baseline Hero strategy.

    For every Hero information set, every ordered pair of distinct legal actions
    ``(source, target)``, and every value in ``shift_amounts``, move that amount
    of probability from ``source`` to ``target``.

    Rules:

    * each shift amount must be finite and strictly positive;
    * a candidate is skipped when the baseline ``source`` probability is less
      than the shift amount (an impossible shift);
    * ``source == target`` is never generated;
    * duplicate candidates (identical resulting strategy) are returned once; and
    * the baseline strategy is never mutated.

    The baseline strategy is validated against ``tree`` before generation.
    """

    validate_hero_strategy(tree, baseline_hero_strategy, tolerance=tolerance)
    for shift in shift_amounts:
        require_finite(shift, "shift_amount")
        if shift <= 0.0:
            raise ValueError(f"shift_amount must be strictly positive, got {shift!r}")

    hero_info_sets = collect_hero_info_sets(tree)
    candidates: List[HeroStrategyCandidate] = []
    seen_keys = set()

    for info_set in sorted(hero_info_sets):
        legal_actions = hero_info_sets[info_set]
        baseline_dist = baseline_hero_strategy.probabilities[info_set]
        for source in legal_actions:
            for target in legal_actions:
                if source == target:
                    continue
                for shift in shift_amounts:
                    source_probability = baseline_dist.get(source, 0.0)
                    if source_probability + tolerance < shift:
                        continue  # impossible shift: source has less than shift

                    new_probabilities = _copy_probabilities(
                        baseline_hero_strategy.probabilities
                    )
                    new_dist = new_probabilities[info_set]
                    new_dist[source] = source_probability - shift
                    if -tolerance < new_dist[source] < 0.0:
                        new_dist[source] = 0.0
                    new_dist[target] = new_dist.get(target, 0.0) + shift

                    key = _canonical_key(new_probabilities)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    l1_distance = sum(
                        abs(new_dist[action] - baseline_dist.get(action, 0.0))
                        for action in legal_actions
                    )
                    candidate_id = f"{info_set}|{source}->{target}|shift={shift}"
                    candidates.append(
                        HeroStrategyCandidate(
                            candidate_id=candidate_id,
                            info_set=info_set,
                            source_action=source,
                            target_action=target,
                            shift_amount=shift,
                            hero_strategy=HeroStrategy(probabilities=new_probabilities),
                            l1_distance=l1_distance,
                        )
                    )

    return candidates
