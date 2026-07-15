"""Canonical cards, strict ranges, and compatible range conditioning for AIoF.

This module implements the bounded, self-contained real-card input layer.  It
parses only explicit 169 classes or exact two-card combinations and makes no
claim about chart syntax, solver ranges, strategy quality, or equilibrium.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Any


__all__ = [
    "AiofStatus",
    "WeightBasis",
    "AiofContractError",
    "AiofLimits",
    "RangeEntry",
    "RangeSpec",
    "ExpandedCombo",
    "ExpandedRange",
    "ComboMarginal",
    "PreparedRanges",
    "card_id",
    "card_from_id",
    "canonicalize_exact_combo",
    "canonicalize_hand_class",
    "expand_range",
    "prepare_compatible_ranges",
]


RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK_ID = "standard-52-rank2toA-suitcdhs-v1"
RANGE_GRAMMAR_ID = "explicit-class-or-combo-v1"
RANGE_EXPANSION_ID = "uniform_within_class_before_conditioning-v1"
JOINT_CONDITIONING_ID = "exact-product-compatible-condition-v1"

MAX_RANGE_ENTRIES_PER_SIDE = 1_326
MAX_EXACT_COMBOS_PER_SIDE = 1_326
MAX_COMPATIBLE_COMBO_PAIRS = 1_624_350
MAX_DEAD_CARDS = 43
MAX_EXACT_BOARD_EVALUATIONS = 10_000_000
MAX_MONTE_CARLO_SAMPLES = 10_000_000
MAX_CACHE_ENTRIES = 100_000
MAX_TRACE_POINTS = 200


class AiofStatus(str, Enum):
    """Fail-closed status classes shared by all phase-1 AIoF modules."""

    SUCCESS = "SUCCESS"
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_CARD_INPUT = "INVALID_CARD_INPUT"
    INVALID_RANGE = "INVALID_RANGE"
    DUPLICATE_COMBO = "DUPLICATE_COMBO"
    EMPTY_COMPATIBLE_SUPPORT = "EMPTY_COMPATIBLE_SUPPORT"
    ZERO_COMPATIBLE_MARGINAL = "ZERO_COMPATIBLE_MARGINAL"
    INVALID_STRATEGY = "INVALID_STRATEGY"
    CAP_EXCEEDED = "CAP_EXCEEDED"
    SAMPLING_ATTEMPT_CAP_EXCEEDED = "SAMPLING_ATTEMPT_CAP_EXCEEDED"
    UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"
    ACCOUNTING_MISMATCH = "ACCOUNTING_MISMATCH"
    ORACLE_MISMATCH = "ORACLE_MISMATCH"
    NON_REPRODUCIBLE = "NON_REPRODUCIBLE"
    NUMERIC_FAILURE = "NUMERIC_FAILURE"


class WeightBasis(str, Enum):
    """Meaning of one range entry's weight."""

    CLASS_TOTAL_MASS = "class_total_mass"
    EXACT_COMBO_MASS = "exact_combo_mass"


class AiofContractError(ValueError):
    """Validated contract failure carrying a stable :class:`AiofStatus`."""

    def __init__(self, status: AiofStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


def _require_plain_int(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AiofContractError(AiofStatus.INVALID_INPUT, f"{name} must be an integer")
    if value < minimum:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, f"{name} must be at least {minimum}"
        )
    return value


@dataclass(frozen=True)
class AiofLimits:
    """Caller-lowerable limits; values may never exceed contract ceilings."""

    max_range_entries_per_side: int = MAX_RANGE_ENTRIES_PER_SIDE
    max_exact_combos_per_side: int = MAX_EXACT_COMBOS_PER_SIDE
    max_compatible_combo_pairs: int = MAX_COMPATIBLE_COMBO_PAIRS
    max_dead_cards: int = MAX_DEAD_CARDS
    max_exact_board_evaluations: int = MAX_EXACT_BOARD_EVALUATIONS
    max_monte_carlo_samples: int = MAX_MONTE_CARLO_SAMPLES
    max_sampling_attempts: int | None = None
    max_cache_entries: int = MAX_CACHE_ENTRIES
    max_trace_points: int = MAX_TRACE_POINTS

    def __post_init__(self) -> None:
        ceilings = (
            ("max_range_entries_per_side", self.max_range_entries_per_side, MAX_RANGE_ENTRIES_PER_SIDE),
            ("max_exact_combos_per_side", self.max_exact_combos_per_side, MAX_EXACT_COMBOS_PER_SIDE),
            ("max_compatible_combo_pairs", self.max_compatible_combo_pairs, MAX_COMPATIBLE_COMBO_PAIRS),
            ("max_dead_cards", self.max_dead_cards, MAX_DEAD_CARDS),
            ("max_exact_board_evaluations", self.max_exact_board_evaluations, MAX_EXACT_BOARD_EVALUATIONS),
            ("max_monte_carlo_samples", self.max_monte_carlo_samples, MAX_MONTE_CARLO_SAMPLES),
            ("max_cache_entries", self.max_cache_entries, MAX_CACHE_ENTRIES),
            ("max_trace_points", self.max_trace_points, MAX_TRACE_POINTS),
        )
        for name, value, ceiling in ceilings:
            _require_plain_int(value, name)
            if value > ceiling:
                raise AiofContractError(
                    AiofStatus.INVALID_INPUT,
                    f"{name}={value} exceeds hard ceiling {ceiling}",
                )
        if self.max_sampling_attempts is not None:
            _require_plain_int(self.max_sampling_attempts, "max_sampling_attempts", 1)


@dataclass(frozen=True)
class RangeEntry:
    """One strict class or exact-combo range entry."""

    label: str
    weight: float
    weight_basis: WeightBasis


@dataclass(frozen=True)
class RangeSpec:
    """A finite tuple of explicit range entries."""

    entries: tuple[RangeEntry, ...]


@dataclass(frozen=True)
class ExpandedCombo:
    """One surviving exact combo with unconditioned raw mass."""

    combo: str
    card_ids: tuple[int, int]
    raw_mass: float
    source_label: str


@dataclass(frozen=True)
class ExpandedRange:
    """Canonical exact support after public-dead-card removal."""

    combos: tuple[ExpandedCombo, ...]
    raw_mass_before_dead: float
    raw_mass_after_dead: float
    removed_combo_count: int
    projected_combo_count: int
    expansion_id: str
    content_identity: str


@dataclass(frozen=True)
class ComboMarginal:
    """One compatible own-combo marginal under joint conditioning."""

    combo: str
    card_ids: tuple[int, int]
    raw_mass: float
    compatible_raw_mass: float
    probability: float


@dataclass(frozen=True)
class PreparedRanges:
    """Two expanded ranges conditioned once on card-disjoint pair support."""

    sb_range: ExpandedRange
    bb_range: ExpandedRange
    dead_cards: tuple[str, ...]
    dead_card_ids: tuple[int, ...]
    compatible_pair_count: int
    compatible_raw_joint_mass: float
    normalization_factor: float
    sb_marginals: tuple[ComboMarginal, ...]
    bb_marginals: tuple[ComboMarginal, ...]
    content_identity: str


def _identity_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AiofContractError(AiofStatus.NUMERIC_FAILURE, "identity contains non-finite float")
        return {"float_hex": value.hex()}
    if isinstance(value, tuple):
        return [_identity_value(item) for item in value]
    if isinstance(value, list):
        return [_identity_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _identity_value(value[key]) for key in sorted(value, key=str)}
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _identity_value(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    return value


def _content_identity(value: Any) -> str:
    encoded = json.dumps(
        _identity_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def card_id(card: str) -> int:
    """Return the canonical integer ID for one strict two-character card."""

    if not isinstance(card, str) or len(card) != 2:
        raise AiofContractError(AiofStatus.INVALID_CARD_INPUT, f"invalid card {card!r}")
    rank, suit = card
    if rank not in RANKS or suit not in SUITS:
        raise AiofContractError(AiofStatus.INVALID_CARD_INPUT, f"invalid card {card!r}")
    return RANKS.index(rank) * 4 + SUITS.index(suit)


def card_from_id(value: int) -> str:
    """Return the strict card spelling for a canonical integer card ID."""

    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 52:
        raise AiofContractError(AiofStatus.INVALID_CARD_INPUT, f"invalid card ID {value!r}")
    rank_index, suit_index = divmod(value, 4)
    return RANKS[rank_index] + SUITS[suit_index]


def canonicalize_exact_combo(label: str) -> str:
    """Validate an exact combo and place the higher card ID first."""

    if not isinstance(label, str) or len(label) != 4:
        raise AiofContractError(AiofStatus.INVALID_RANGE, f"invalid exact combo {label!r}")
    try:
        first = card_id(label[:2])
        second = card_id(label[2:])
    except AiofContractError as exc:
        raise AiofContractError(AiofStatus.INVALID_RANGE, str(exc)) from exc
    if first == second:
        raise AiofContractError(AiofStatus.INVALID_RANGE, "an exact combo must contain distinct cards")
    high, low = sorted((first, second), reverse=True)
    return card_from_id(high) + card_from_id(low)


def canonicalize_hand_class(label: str) -> str:
    """Validate and return one canonical pair, suited, or offsuit class."""

    if not isinstance(label, str):
        raise AiofContractError(AiofStatus.INVALID_RANGE, f"invalid hand class {label!r}")
    if len(label) == 2 and label[0] in RANKS and label[0] == label[1]:
        return label
    if (
        len(label) == 3
        and label[0] in RANKS
        and label[1] in RANKS
        and label[2] in "so"
        and RANKS.index(label[0]) > RANKS.index(label[1])
    ):
        return label
    raise AiofContractError(AiofStatus.INVALID_RANGE, f"invalid hand class {label!r}")


def _canonical_dead_cards(dead_cards: tuple[str, ...], limits: AiofLimits) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if not isinstance(dead_cards, tuple):
        raise AiofContractError(AiofStatus.INVALID_CARD_INPUT, "dead_cards must be a tuple")
    if len(dead_cards) > limits.max_dead_cards:
        raise AiofContractError(AiofStatus.CAP_EXCEEDED, "dead-card cap exceeded")
    ids = tuple(card_id(card) for card in dead_cards)
    if len(set(ids)) != len(ids):
        raise AiofContractError(AiofStatus.INVALID_CARD_INPUT, "dead cards must be distinct")
    ordered_ids = tuple(sorted(ids))
    return tuple(card_from_id(value) for value in ordered_ids), ordered_ids


def _class_multiplicity(label: str) -> int:
    return 6 if len(label) == 2 else (4 if label[2] == "s" else 12)


def _class_card_pairs(label: str):
    first_rank = RANKS.index(label[0])
    second_rank = RANKS.index(label[1])
    if len(label) == 2:
        cards = tuple(first_rank * 4 + suit for suit in range(4))
        yield from combinations(cards, 2)
    elif label[2] == "s":
        for suit in range(4):
            yield (second_rank * 4 + suit, first_rank * 4 + suit)
    else:
        for first_suit in range(4):
            for second_suit in range(4):
                if first_suit != second_suit:
                    yield (second_rank * 4 + second_suit, first_rank * 4 + first_suit)


def _entry_kind(label: object) -> tuple[str, str]:
    if isinstance(label, str) and len(label) == 4:
        try:
            return "exact", canonicalize_exact_combo(label)
        except AiofContractError:
            pass
    try:
        return "class", canonicalize_hand_class(label)  # type: ignore[arg-type]
    except AiofContractError as exc:
        raise AiofContractError(AiofStatus.INVALID_RANGE, f"invalid range label {label!r}") from exc


def expand_range(
    spec: RangeSpec, dead_cards: tuple[str, ...], limits: AiofLimits
) -> ExpandedRange:
    """Expand a strict range without renormalizing within dead-card-hit classes."""

    if not isinstance(spec, RangeSpec) or not isinstance(spec.entries, tuple):
        raise AiofContractError(AiofStatus.INVALID_RANGE, "range entries must be a tuple")
    if not isinstance(limits, AiofLimits):
        raise AiofContractError(AiofStatus.INVALID_INPUT, "limits must be AiofLimits")
    if len(spec.entries) > limits.max_range_entries_per_side:
        raise AiofContractError(AiofStatus.CAP_EXCEEDED, "range-entry cap exceeded")
    canonical_dead, dead_ids = _canonical_dead_cards(dead_cards, limits)
    dead_set = set(dead_ids)

    validated: list[tuple[str, str, float]] = []
    projected = 0
    for entry in spec.entries:
        if not isinstance(entry, RangeEntry):
            raise AiofContractError(AiofStatus.INVALID_RANGE, "range contains a non-RangeEntry")
        if isinstance(entry.weight, bool) or not isinstance(entry.weight, (int, float)):
            raise AiofContractError(AiofStatus.INVALID_RANGE, "range weight must be a number")
        weight = float(entry.weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise AiofContractError(AiofStatus.INVALID_RANGE, "range weight must be finite and positive")
        if not isinstance(entry.weight_basis, WeightBasis):
            raise AiofContractError(AiofStatus.INVALID_RANGE, "invalid weight basis")
        kind, canonical = _entry_kind(entry.label)
        expected = WeightBasis.CLASS_TOTAL_MASS if kind == "class" else WeightBasis.EXACT_COMBO_MASS
        if entry.weight_basis is not expected:
            raise AiofContractError(AiofStatus.INVALID_RANGE, "weight basis does not match label kind")
        count = _class_multiplicity(canonical) if kind == "class" else 1
        projected += count
        if projected > limits.max_exact_combos_per_side:
            raise AiofContractError(AiofStatus.CAP_EXCEEDED, "exact-combo cap exceeded")
        validated.append((kind, canonical, weight))

    generated: dict[tuple[int, int], ExpandedCombo] = {}
    before_masses: list[float] = []
    for kind, canonical, weight in validated:
        if kind == "class":
            multiplicity = _class_multiplicity(canonical)
            mass = weight / multiplicity
            pairs = _class_card_pairs(canonical)
        else:
            canonical_exact = canonicalize_exact_combo(canonical)
            ids = tuple(sorted((card_id(canonical_exact[:2]), card_id(canonical_exact[2:]))))
            mass = weight
            pairs = (ids,)
        if not math.isfinite(mass) or mass <= 0.0:
            raise AiofContractError(AiofStatus.NUMERIC_FAILURE, "expanded combo mass is invalid")
        for pair in pairs:
            ids = tuple(sorted(pair))
            if ids in generated:
                raise AiofContractError(
                    AiofStatus.DUPLICATE_COMBO,
                    f"duplicate generated combo {card_from_id(ids[1]) + card_from_id(ids[0])}",
                )
            combo = card_from_id(ids[1]) + card_from_id(ids[0])
            generated[ids] = ExpandedCombo(combo, ids, mass, canonical)
            before_masses.append(mass)

    before = math.fsum(before_masses)
    if not generated or not math.isfinite(before) or before <= 0.0:
        raise AiofContractError(AiofStatus.EMPTY_COMPATIBLE_SUPPORT, "range has no positive support")
    surviving = tuple(
        generated[ids]
        for ids in sorted(generated)
        if ids[0] not in dead_set and ids[1] not in dead_set
    )
    after = math.fsum(combo.raw_mass for combo in surviving)
    if not surviving or not math.isfinite(after) or after <= 0.0:
        raise AiofContractError(AiofStatus.EMPTY_COMPATIBLE_SUPPORT, "dead cards remove all range support")
    identity = _content_identity(
        {
            "deck": DECK_ID,
            "grammar": RANGE_GRAMMAR_ID,
            "expansion": RANGE_EXPANSION_ID,
            "dead_cards": canonical_dead,
            "combos": tuple((c.combo, c.raw_mass, c.source_label) for c in surviving),
            "raw_mass_before_dead": before,
        }
    )
    return ExpandedRange(
        combos=surviving,
        raw_mass_before_dead=before,
        raw_mass_after_dead=after,
        removed_combo_count=len(generated) - len(surviving),
        projected_combo_count=projected,
        expansion_id=RANGE_EXPANSION_ID,
        content_identity=identity,
    )


def _disjoint(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return first[0] not in second and first[1] not in second


def prepare_compatible_ranges(
    sb: RangeSpec,
    bb: RangeSpec,
    dead_cards: tuple[str, ...],
    limits: AiofLimits,
) -> PreparedRanges:
    """Condition factorized range masses once on card-disjoint ordered pairs."""

    canonical_dead, dead_ids = _canonical_dead_cards(dead_cards, limits)
    sb_range = expand_range(sb, canonical_dead, limits)
    bb_range = expand_range(bb, canonical_dead, limits)
    sb_compatible: list[float] = []
    row_joint_masses: list[float] = []
    compatible_count = 0
    for sb_combo in sb_range.combos:
        row_terms: list[float] = []
        for bb_index, bb_combo in enumerate(bb_range.combos):
            if not _disjoint(sb_combo.card_ids, bb_combo.card_ids):
                continue
            compatible_count += 1
            if compatible_count > limits.max_compatible_combo_pairs:
                raise AiofContractError(AiofStatus.CAP_EXCEEDED, "compatible-pair cap exceeded")
            joint = sb_combo.raw_mass * bb_combo.raw_mass
            if not math.isfinite(joint) or joint <= 0.0:
                raise AiofContractError(AiofStatus.NUMERIC_FAILURE, "invalid joint mass")
            row_terms.append(joint)
        row_mass = math.fsum(row_terms)
        sb_compatible.append(row_mass)
        row_joint_masses.append(row_mass)
    bb_compatible = [
        math.fsum(
            sb_combo.raw_mass * bb_combo.raw_mass
            for sb_combo in sb_range.combos
            if _disjoint(sb_combo.card_ids, bb_combo.card_ids)
        )
        for bb_combo in bb_range.combos
    ]
    joint_mass = math.fsum(row_joint_masses)
    if compatible_count == 0 or not math.isfinite(joint_mass) or joint_mass <= 0.0:
        status = AiofStatus.NUMERIC_FAILURE if compatible_count else AiofStatus.EMPTY_COMPATIBLE_SUPPORT
        raise AiofContractError(status, "compatible joint support has no finite positive mass")
    if any(value <= 0.0 for value in sb_compatible) or any(value <= 0.0 for value in bb_compatible):
        raise AiofContractError(
            AiofStatus.ZERO_COMPATIBLE_MARGINAL,
            "a surviving own combo has zero compatible marginal",
        )
    if not all(math.isfinite(value) for value in bb_compatible):
        raise AiofContractError(AiofStatus.NUMERIC_FAILURE, "compatible marginal is non-finite")
    factor = 1.0 / joint_mass
    if not math.isfinite(factor):
        raise AiofContractError(AiofStatus.NUMERIC_FAILURE, "normalization factor is non-finite")
    sb_marginals = tuple(
        ComboMarginal(combo.combo, combo.card_ids, combo.raw_mass, mass, mass * factor)
        for combo, mass in zip(sb_range.combos, sb_compatible)
    )
    bb_marginals = tuple(
        ComboMarginal(combo.combo, combo.card_ids, combo.raw_mass, mass, mass * factor)
        for combo, mass in zip(bb_range.combos, bb_compatible)
    )
    sb_probability_sum = math.fsum(item.probability for item in sb_marginals)
    bb_probability_sum = math.fsum(item.probability for item in bb_marginals)
    if (
        not math.isfinite(sb_probability_sum)
        or not math.isfinite(bb_probability_sum)
        or abs(sb_probability_sum - 1.0) > 1e-12
        or abs(bb_probability_sum - 1.0) > 1e-12
    ):
        raise AiofContractError(AiofStatus.NUMERIC_FAILURE, "normalized marginals do not sum to one")
    identity = _content_identity(
        {
            "deck": DECK_ID,
            "joint": JOINT_CONDITIONING_ID,
            "sb": sb_range.content_identity,
            "bb": bb_range.content_identity,
            "dead_cards": canonical_dead,
            "compatible_pair_count": compatible_count,
            "joint_mass": joint_mass,
        }
    )
    return PreparedRanges(
        sb_range=sb_range,
        bb_range=bb_range,
        dead_cards=canonical_dead,
        dead_card_ids=dead_ids,
        compatible_pair_count=compatible_count,
        compatible_raw_joint_mass=joint_mass,
        normalization_factor=factor,
        sb_marginals=sb_marginals,
        bb_marginals=bb_marginals,
        content_identity=identity,
    )
