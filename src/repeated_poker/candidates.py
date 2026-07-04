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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

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

# Safety upper bound on the number of multi-shift candidates a single call may
# enumerate, checked before the generation loop runs (see
# ``generate_multi_shift_candidates``). It guards against the combinatorial blow
# up of pairing information sets on a large tree; it is deliberately generous.
DEFAULT_MAX_CANDIDATES = 100_000


@dataclass(frozen=True)
class ShiftComponent:
    """One single-information-set probability shift.

    A candidate is the composition of one (single-shift) or several (multi-shift)
    of these, each moving ``shift_amount`` of probability from ``source_action``
    to ``target_action`` at ``info_set``. Every component of a candidate acts on a
    distinct information set.
    """

    info_set: InfoSetId
    source_action: Action
    target_action: Action
    shift_amount: float

    @property
    def component_id(self) -> str:
        """Stable id for this component (the historic single-shift id format)."""

        return (
            f"{self.info_set}|{self.source_action}->{self.target_action}"
            f"|shift={self.shift_amount}"
        )

    def to_dict(self) -> dict:
        return {
            "info_set": self.info_set,
            "source_action": self.source_action,
            "target_action": self.target_action,
            "shift_amount": self.shift_amount,
        }


# Separator joining component ids into a multi-shift ``candidate_id``. Component
# ids never contain it, so the composite id is unambiguous and reversible.
_MULTI_SHIFT_ID_SEPARATOR = " + "


@dataclass(frozen=True)
class HeroStrategyCandidate:
    """One Hero candidate produced by one or more single-action probability shifts.

    A *single-shift* candidate moves probability at a single Hero information set;
    its ``info_set`` / ``source_action`` / ``target_action`` / ``shift_amount``
    scalar fields describe that shift. A *multi-shift* candidate (M2-T2) composes
    shifts at several distinct information sets; there is no single scalar shift,
    so those scalar fields are ``None`` and the full breakdown lives in
    ``shifts``.

    ``shifts`` is always populated: for a single-shift candidate it is derived
    from the scalar fields (so existing constructions are unchanged), and it holds
    one :class:`ShiftComponent` per changed information set.

    ``l1_distance`` is the strategy-space L1 distance from the baseline (the sum
    of absolute per-action probability changes), not an observable behavioural
    distance.
    """

    candidate_id: str
    info_set: Optional[InfoSetId]
    source_action: Optional[Action]
    target_action: Optional[Action]
    shift_amount: Optional[float]
    hero_strategy: HeroStrategy
    l1_distance: float
    shifts: Tuple[ShiftComponent, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.shifts:
            # Backward-compatible single-shift construction: derive the single
            # component from the scalar fields.
            if self.info_set is None:
                raise ValueError(
                    "HeroStrategyCandidate needs either the single-shift scalar "
                    "fields or a non-empty shifts tuple"
                )
            object.__setattr__(
                self,
                "shifts",
                (
                    ShiftComponent(
                        info_set=self.info_set,
                        source_action=self.source_action,
                        target_action=self.target_action,
                        shift_amount=self.shift_amount,
                    ),
                ),
            )

    @property
    def is_multi_shift(self) -> bool:
        return len(self.shifts) > 1

    @property
    def info_sets(self) -> Tuple[InfoSetId, ...]:
        """The distinct information sets this candidate changes, in shift order."""

        return tuple(component.info_set for component in self.shifts)


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


def generate_multi_shift_candidates(
    tree: GameTree,
    baseline_hero_strategy: HeroStrategy,
    shift_amounts: Sequence[float],
    num_info_sets: int = 2,
    tolerance: float = 1e-9,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> List[HeroStrategyCandidate]:
    """Generate simultaneous multi-information-set shift candidates (M2-T2).

    A multi-shift candidate applies ``num_info_sets`` single shifts at that many
    *distinct* Hero information sets at once. It is built by composing the
    single-shift candidates from :func:`generate_shift_candidates` across
    different information sets, so each per-information-set change is exactly one
    of the already-validated single shifts and the two layers stay consistent.

    Only ``num_info_sets == 2`` is supported for now (the roadmap starts k=2);
    other values raise :class:`ValueError`. Because the changed information sets
    are disjoint, the composed ``l1_distance`` is the sum of the component
    distances, and the composite ``candidate_id`` is the component ids joined by
    ``" + "`` in information-set order.

    ``max_candidates`` bounds the enumeration: the projected number of pairings
    is checked *before* the generation loop and exceeding it raises rather than
    building a huge library.
    """

    if isinstance(num_info_sets, bool) or not isinstance(num_info_sets, int):
        raise ValueError(f"num_info_sets must be an int, got {num_info_sets!r}")
    if num_info_sets != 2:
        raise ValueError(
            f"only num_info_sets=2 is supported in this version, got {num_info_sets}"
        )
    _validate_max_candidates(max_candidates)

    singles = generate_shift_candidates(
        tree, baseline_hero_strategy, shift_amounts, tolerance=tolerance
    )
    singles_by_info_set: Dict[InfoSetId, List[HeroStrategyCandidate]] = {}
    for candidate in singles:
        singles_by_info_set.setdefault(candidate.info_set, []).append(candidate)

    info_sets = sorted(singles_by_info_set)

    # Safety bound checked before the loop: the total number of ordered pairings
    # cannot exceed this, so a large tree cannot silently enumerate a huge library.
    projected = sum(
        len(singles_by_info_set[info_sets[i]]) * len(singles_by_info_set[info_sets[j]])
        for i in range(len(info_sets))
        for j in range(i + 1, len(info_sets))
    )
    if projected > max_candidates:
        raise ValueError(
            f"multi-shift generation would enumerate up to {projected} candidate "
            f"pairings, exceeding max_candidates={max_candidates}; reduce the shift "
            "amounts or raise max_candidates"
        )

    baseline_probabilities = baseline_hero_strategy.probabilities
    candidates: List[HeroStrategyCandidate] = []
    seen_keys = set()

    for i in range(len(info_sets)):
        first_set = info_sets[i]
        for j in range(i + 1, len(info_sets)):
            second_set = info_sets[j]
            for first in singles_by_info_set[first_set]:
                for second in singles_by_info_set[second_set]:
                    new_probabilities = _copy_probabilities(baseline_probabilities)
                    new_probabilities[first_set] = dict(
                        first.hero_strategy.probabilities[first_set]
                    )
                    new_probabilities[second_set] = dict(
                        second.hero_strategy.probabilities[second_set]
                    )

                    key = _canonical_key(new_probabilities)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    # first_set precedes second_set (info sets iterated in sorted
                    # order), so the composite id and shift order are deterministic.
                    shifts = first.shifts + second.shifts
                    candidate_id = _MULTI_SHIFT_ID_SEPARATOR.join(
                        component.component_id for component in shifts
                    )
                    candidates.append(
                        HeroStrategyCandidate(
                            candidate_id=candidate_id,
                            info_set=None,
                            source_action=None,
                            target_action=None,
                            shift_amount=None,
                            hero_strategy=HeroStrategy(probabilities=new_probabilities),
                            l1_distance=first.l1_distance + second.l1_distance,
                            shifts=shifts,
                        )
                    )

    return candidates


def generate_candidate_library(
    tree: GameTree,
    baseline_hero_strategy: HeroStrategy,
    shift_amounts: Sequence[float],
    max_simultaneous_info_sets: int = 1,
    tolerance: float = 1e-9,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> List[HeroStrategyCandidate]:
    """Generate the candidate library, optionally including multi-shift candidates.

    With ``max_simultaneous_info_sets == 1`` (the default) this returns exactly
    the single-shift library from :func:`generate_shift_candidates`, so existing
    behaviour is unchanged. With ``max_simultaneous_info_sets == 2`` the
    simultaneous two-information-set candidates are appended after the single
    shifts (single-shift candidates keep their original order and ids).

    Only ``1`` and ``2`` are supported for now; other values raise
    :class:`ValueError`.
    """

    if isinstance(max_simultaneous_info_sets, bool) or not isinstance(
        max_simultaneous_info_sets, int
    ):
        raise ValueError(
            "max_simultaneous_info_sets must be an int, got "
            f"{max_simultaneous_info_sets!r}"
        )
    if max_simultaneous_info_sets < 1 or max_simultaneous_info_sets > 2:
        raise ValueError(
            "max_simultaneous_info_sets must be 1 or 2 in this version, got "
            f"{max_simultaneous_info_sets}"
        )

    candidates = generate_shift_candidates(
        tree, baseline_hero_strategy, shift_amounts, tolerance=tolerance
    )
    if max_simultaneous_info_sets >= 2:
        candidates = candidates + generate_multi_shift_candidates(
            tree,
            baseline_hero_strategy,
            shift_amounts,
            num_info_sets=2,
            tolerance=tolerance,
            max_candidates=max_candidates,
        )
    return candidates


def _validate_max_candidates(max_candidates: int) -> None:
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int):
        raise ValueError(f"max_candidates must be an int, got {max_candidates!r}")
    if max_candidates < 1:
        raise ValueError(f"max_candidates must be at least 1, got {max_candidates}")
