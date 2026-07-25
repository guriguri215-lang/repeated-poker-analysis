"""Known-board real-card three-player river/rake commitment adapter.

The module has two deliberately separate layers.  The chance layer expands
three M13 ranges, removes board/dead/private-card collisions, and conditions
the compatible combo triples exactly once.  The betting layer converts those
triples into one bounded M31 chance tree, then delegates exact O1/O2 response
and repeated candidate evaluation to the public M31/M30/M32 APIs.

The supported claim is conditional and bounded.  This module is not a global
or continuous optimizer, a full poker solver, an equilibrium certificate, a
raw solver-export parser, a population model, or strategy advice.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from fractions import Fraction
from typing import Any, Mapping

from . import three_player_candidate_repeated as _m32
from . import three_player_response as _m30
from . import three_player_river_rake as _m31
from .aiof_cards import (
    AiofContractError,
    AiofLimits,
    AiofStatus,
    ExpandedCombo,
    ExpandedRange,
    RangeSpec,
    card_from_id,
    card_id,
    canonicalize_exact_combo,
    expand_range,
)
from .aiof_evaluator import EVALUATOR_ID, HandRank, evaluate_seven_card_hand
from .automatic_commitment_selection import AutomaticCommitmentSelectionConfig
from .known_board_real_card_hu_river import (
    ComboBucketAssignment,
    ComboBucketMap,
)


__all__ = [
    "KNOWN_BOARD_REAL_CARD_THREE_PLAYER_RIVER_CONTRACT_VERSION",
    "ExactActionProbability",
    "RealCardProfileRow",
    "RealCardThreePlayerProfile",
    "GeneratedTreeAttestation",
    "KnownBoardRealCardThreePlayerLimits",
    "KnownBoardRealCardThreePlayerRequest",
    "ThreePlayerRangeProvenance",
    "ThreePlayerComboMarginal",
    "CompatibleComboTriple",
    "PreparedThreePlayerSupport",
    "KnownBoardShowdownRow",
    "KnownBoardRealCardThreePlayerPayload",
    "KnownBoardRealCardThreePlayerResult",
    "prepare_known_board_three_player_support",
    "analyze_known_board_real_card_three_player_river",
    "exact_known_board_real_card_three_player_json",
]


KNOWN_BOARD_REAL_CARD_THREE_PLAYER_RIVER_CONTRACT_VERSION = (
    "known-board-real-card-three-player-river-rake-v1"
)
JOINT_CONDITIONING_VERSION = "exact-product-compatible-triple-condition-v1"
BETTING_TREE_VERSION = "h-open-o1-call-fold-o2-call-fold-v1"
PROFILE_VERSION = "complete-exact-combo-bucket-profile-v1"
IDENTITY_VERSION = "known-board-three-player-canonical-json-sha256-v1"
CLAIM_SCOPE = (
    "bounded_known_board_real_card_three_player_exact_response_repeated_analysis"
)

PLAYERS = ("H", "O1", "O2")
DECISIONS = {
    "H": {"open": ("check", "bet")},
    "O1": {
        "after_hero_check": ("check",),
        "vs_hero_bet": ("call", "fold"),
    },
    "O2": {
        "after_hero_o1_check": ("check",),
        "vs_hero_bet_o1_call": ("call", "fold"),
        "vs_hero_bet_o1_fold": ("call", "fold"),
    },
}
INFO_PREFIX = {
    ("H", "open"): "H_open",
    ("O1", "after_hero_check"): "O1_after_H_check",
    ("O1", "vs_hero_bet"): "O1_vs_H_bet",
    ("O2", "after_hero_o1_check"): "O2_after_H_O1_check",
    ("O2", "vs_hero_bet_o1_call"): "O2_vs_H_bet_O1_call",
    ("O2", "vs_hero_bet_o1_fold"): "O2_vs_H_bet_O1_fold",
}

MAX_CARTESIAN_TRIPLES = 1_000_000
MAX_COMPATIBLE_TRIPLES = 32
MAX_FIXED_BOARD_EVALUATIONS = 96
MAX_PROFILE_ROWS = 512
MAX_BUCKETS_PER_SEAT = 32
MAX_TREE_NODES = 500
MAX_IDENTITY_RECORDS = 200_000
MAX_OUTPUT_BYTES = 256_000_000
MAX_RATIONAL_BITS = 1_024

_RATIONAL_RE = re.compile(r"(?:0|[1-9][0-9]*|-[1-9][0-9]*)(?:/[1-9][0-9]*)?")
_BUCKET_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,47}")
_IDENTITY_RE = re.compile(r"[0-9a-f]{64}")
_DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


@dataclass(frozen=True)
class ExactActionProbability:
    """One exact canonical rational action probability."""

    action: str
    probability: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class RealCardProfileRow:
    """One complete bucket/decision action distribution."""

    player_id: str
    bucket_id: str
    decision: str
    actions: tuple[ExactActionProbability, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "bucket_id": self.bucket_id,
            "decision": self.decision,
            "actions": [action.to_dict() for action in self.actions],
        }


@dataclass(frozen=True)
class RealCardThreePlayerProfile:
    """Complete H/O1/O2 baseline profile over all surviving buckets."""

    rows: tuple[RealCardProfileRow, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"rows": [row.to_dict() for row in self.rows]}


@dataclass(frozen=True)
class GeneratedTreeAttestation:
    """Human evidence fields used to bind M31 perfect-recall attestation."""

    verifier: str
    verification_date: str
    evidence_version: str
    o1_confirmed: bool = True
    o2_confirmed: bool = True


@dataclass(frozen=True)
class KnownBoardRealCardThreePlayerLimits:
    """Caller-lowerable adapter caps with immutable hard ceilings."""

    max_cartesian_triples: int = 100_000
    max_compatible_triples: int = 16
    max_fixed_board_evaluations: int = 48
    max_profile_rows: int = 256
    max_buckets_per_seat: int = 16
    max_tree_nodes: int = 200
    max_identity_records: int = 100_000
    max_output_bytes: int = 32_000_000
    max_rational_numerator_bits: int = 256
    max_rational_denominator_bits: int = 256


@dataclass(frozen=True)
class KnownBoardRealCardThreePlayerRequest:
    """One bounded known-board real-card three-player river request."""

    board: tuple[str, ...]
    dead_cards: tuple[str, ...]
    hero_range: RangeSpec
    o1_range: RangeSpec
    o2_range: RangeSpec
    baseline_profile: RealCardThreePlayerProfile
    attestation: GeneratedTreeAttestation
    initial_contribution: str = "10"
    bet_to: str = "20"
    rake_rate: str = "0"
    rake_cap: str | None = None
    hero_combo_to_bucket: ComboBucketMap | None = None
    o1_combo_to_bucket: ComboBucketMap | None = None
    o2_combo_to_bucket: ComboBucketMap | None = None
    generation: _m32.ThreePlayerCandidateGenerationConfig = (
        _m32.ThreePlayerCandidateGenerationConfig()
    )
    repeated: _m32.ThreePlayerRepeatedConfig = _m32.ThreePlayerRepeatedConfig()
    selector_configuration: AutomaticCommitmentSelectionConfig = (
        AutomaticCommitmentSelectionConfig()
    )
    limits: KnownBoardRealCardThreePlayerLimits = (
        KnownBoardRealCardThreePlayerLimits()
    )
    aiof_limits: AiofLimits = AiofLimits()
    m32_limits: _m32.ThreePlayerCandidateRepeatedLimits = (
        _m32.ThreePlayerCandidateRepeatedLimits()
    )
    m31_limits: _m31.RiverRakeLimits = _m31.RiverRakeLimits()
    m30_limits: _m30.ExactResponseLimits = _m30.ExactResponseLimits()
    expected_baseline_identity: str | None = None


@dataclass(frozen=True)
class ThreePlayerRangeProvenance:
    """Pre-removal and surviving identities/counts for one seat."""

    player_id: str
    initial_range_identity: str
    after_board_range_identity: str
    surviving_range_identity: str
    initial_combo_count: int
    after_board_combo_count: int
    surviving_combo_count: int
    board_removed_combo_count: int
    dead_removed_combo_count: int
    raw_mass_before_removal: float
    raw_mass_after_removal: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ThreePlayerComboMarginal:
    """One exact combo marginal after joint triple conditioning."""

    player_id: str
    combo: str
    card_ids: tuple[int, int]
    raw_mass: float
    compatible_raw_mass_exact: str
    probability_exact: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "combo": self.combo,
            "card_ids": list(self.card_ids),
            "raw_mass": self.raw_mass,
            "compatible_raw_mass_exact": self.compatible_raw_mass_exact,
            "probability_exact": self.probability_exact,
        }


@dataclass(frozen=True)
class CompatibleComboTriple:
    """One compatible H/O1/O2 exact-combo chance row."""

    hero_combo: str
    o1_combo: str
    o2_combo: str
    raw_joint_mass_exact: str
    probability_exact: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedThreePlayerSupport:
    """Complete compatible triple support conditioned exactly once."""

    board: tuple[str, ...]
    dead_cards: tuple[str, ...]
    unavailable_cards: tuple[str, ...]
    range_provenance: tuple[ThreePlayerRangeProvenance, ...]
    cartesian_triple_count: int
    private_collision_excluded_count: int
    compatible_triple_count: int
    compatible_raw_joint_mass_exact: str
    triples: tuple[CompatibleComboTriple, ...]
    marginals: tuple[ThreePlayerComboMarginal, ...]
    content_identity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "board": list(self.board),
            "dead_cards": list(self.dead_cards),
            "unavailable_cards": list(self.unavailable_cards),
            "range_provenance": [
                provenance.to_dict() for provenance in self.range_provenance
            ],
            "cartesian_triple_count": self.cartesian_triple_count,
            "private_collision_excluded_count": (
                self.private_collision_excluded_count
            ),
            "compatible_triple_count": self.compatible_triple_count,
            "compatible_raw_joint_mass_exact": (
                self.compatible_raw_joint_mass_exact
            ),
            "triples": [triple.to_dict() for triple in self.triples],
            "marginals": [marginal.to_dict() for marginal in self.marginals],
            "content_identity": self.content_identity,
        }


@dataclass(frozen=True)
class KnownBoardShowdownRow:
    """One ranked compatible triple and all active-player winner sets."""

    hero_combo: str
    o1_combo: str
    o2_combo: str
    hero_bucket_id: str
    o1_bucket_id: str
    o2_bucket_id: str
    probability_exact: str
    hero_rank: HandRank
    o1_rank: HandRank
    o2_rank: HandRank
    three_way_winners: tuple[str, ...]
    heads_up_winners: tuple[tuple[str, tuple[str, ...]], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hero_combo": self.hero_combo,
            "o1_combo": self.o1_combo,
            "o2_combo": self.o2_combo,
            "hero_bucket_id": self.hero_bucket_id,
            "o1_bucket_id": self.o1_bucket_id,
            "o2_bucket_id": self.o2_bucket_id,
            "probability_exact": self.probability_exact,
            "hero_rank": _hand_rank_dict(self.hero_rank),
            "o1_rank": _hand_rank_dict(self.o1_rank),
            "o2_rank": _hand_rank_dict(self.o2_rank),
            "three_way_winners": list(self.three_way_winners),
            "heads_up_winners": {
                key: list(winners) for key, winners in self.heads_up_winners
            },
        }


@dataclass(frozen=True)
class KnownBoardRealCardThreePlayerPayload:
    """Complete real-card chance and native M32 analysis."""

    contract_version: str
    claim_scope: str
    prepared_support: PreparedThreePlayerSupport
    showdown_rows: tuple[KnownBoardShowdownRow, ...]
    canonical_profile: RealCardThreePlayerProfile
    bucket_maps: Mapping[str, ComboBucketMap]
    workload: Mapping[str, int]
    baseline_identity: str
    native_m32_result: _m32.ThreePlayerCandidateRepeatedResult
    identities: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "claim_scope": self.claim_scope,
            "prepared_support": self.prepared_support.to_dict(),
            "showdown_rows": [row.to_dict() for row in self.showdown_rows],
            "canonical_profile": self.canonical_profile.to_dict(),
            "bucket_maps": {
                player: _bucket_map_dict(self.bucket_maps[player])
                for player in PLAYERS
            },
            "workload": dict(self.workload),
            "baseline_identity": self.baseline_identity,
            "native_m32_result": self.native_m32_result.to_dict(),
            "identities": dict(self.identities),
        }


@dataclass(frozen=True)
class KnownBoardRealCardThreePlayerResult:
    """Exclusive complete-success or fail-closed outer result."""

    status: AiofStatus
    payload: KnownBoardRealCardThreePlayerPayload | None
    error_message: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "payload": None if self.payload is None else self.payload.to_dict(),
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class _ExpandedSeat:
    player_id: str
    initial: ExpandedRange
    after_board: ExpandedRange
    final: ExpandedRange


@dataclass(frozen=True)
class _PreparedInternal:
    public: PreparedThreePlayerSupport
    seats: tuple[_ExpandedSeat, ...]


_HARD_LIMITS = {
    "max_cartesian_triples": MAX_CARTESIAN_TRIPLES,
    "max_compatible_triples": MAX_COMPATIBLE_TRIPLES,
    "max_fixed_board_evaluations": MAX_FIXED_BOARD_EVALUATIONS,
    "max_profile_rows": MAX_PROFILE_ROWS,
    "max_buckets_per_seat": MAX_BUCKETS_PER_SEAT,
    "max_tree_nodes": MAX_TREE_NODES,
    "max_identity_records": MAX_IDENTITY_RECORDS,
    "max_output_bytes": MAX_OUTPUT_BYTES,
    "max_rational_numerator_bits": MAX_RATIONAL_BITS,
    "max_rational_denominator_bits": MAX_RATIONAL_BITS,
}


def _validate_plain_positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, f"{name} must be a positive integer"
        )
    return value


def _validate_limits(value: object) -> KnownBoardRealCardThreePlayerLimits:
    if type(value) is not KnownBoardRealCardThreePlayerLimits:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            "limits must be KnownBoardRealCardThreePlayerLimits",
        )
    for field in fields(value):
        current = _validate_plain_positive_int(
            getattr(value, field.name), field.name
        )
        if current > _HARD_LIMITS[field.name]:
            raise AiofContractError(
                AiofStatus.INVALID_INPUT,
                f"{field.name}={current} exceeds hard ceiling "
                f"{_HARD_LIMITS[field.name]}",
            )
    return value


def _rational_text(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _check_fraction_bits(
    value: Fraction,
    name: str,
    limits: KnownBoardRealCardThreePlayerLimits,
) -> Fraction:
    if (
        value.numerator.bit_length() > limits.max_rational_numerator_bits
        or value.denominator.bit_length()
        > limits.max_rational_denominator_bits
    ):
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED, f"{name} rational bit cap exceeded"
        )
    return value


def _parse_rational(
    value: object,
    name: str,
    *,
    minimum: Fraction | None = None,
    maximum: Fraction | None = None,
    limits: KnownBoardRealCardThreePlayerLimits,
) -> Fraction:
    if type(value) is not str or _RATIONAL_RE.fullmatch(value) is None:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            f"{name} must be a canonical rational string",
        )
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, f"{name} is invalid"
        ) from exc
    if _rational_text(parsed) != value:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, f"{name} must be reduced and canonical"
        )
    if (
        parsed.numerator.bit_length() > limits.max_rational_numerator_bits
        or parsed.denominator.bit_length()
        > limits.max_rational_denominator_bits
    ):
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED, f"{name} rational bit cap exceeded"
        )
    if minimum is not None and parsed < minimum:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, f"{name} is below its minimum"
        )
    if maximum is not None and parsed > maximum:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, f"{name} is above its maximum"
        )
    return parsed


def _identity_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Fraction):
        return {"rational": _rational_text(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AiofContractError(
                AiofStatus.NUMERIC_FAILURE,
                "identity contains a non-finite float",
            )
        return {"float_hex": value.hex()}
    if isinstance(value, HandRank):
        return _hand_rank_dict(value)
    if isinstance(value, tuple):
        return [_identity_value(item) for item in value]
    if isinstance(value, list):
        return [_identity_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _identity_value(value[key])
            for key in sorted(value, key=str)
        }
    if is_dataclass(value):
        return {
            field.name: _identity_value(getattr(value, field.name))
            for field in fields(value)
        }
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _identity_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _identity(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _hand_rank_dict(rank: HandRank) -> dict[str, Any]:
    return {
        "category": rank.category.name,
        "category_value": int(rank.category),
        "tiebreak": list(rank.tiebreak),
    }


def _canonical_board_dead(
    board: object,
    dead_cards: object,
    aiof_limits: AiofLimits,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if type(board) is not tuple or len(board) != 5:
        raise AiofContractError(
            AiofStatus.INVALID_CARD_INPUT,
            "board must be a tuple of exactly five cards",
        )
    if type(dead_cards) is not tuple:
        raise AiofContractError(
            AiofStatus.INVALID_CARD_INPUT, "dead_cards must be a tuple"
        )
    board_ids = tuple(card_id(card) for card in board)
    dead_ids = tuple(card_id(card) for card in dead_cards)
    if len(set(board_ids)) != 5:
        raise AiofContractError(
            AiofStatus.INVALID_CARD_INPUT, "board cards must be distinct"
        )
    if len(set(dead_ids)) != len(dead_ids):
        raise AiofContractError(
            AiofStatus.INVALID_CARD_INPUT, "dead cards must be distinct"
        )
    if set(board_ids) & set(dead_ids):
        raise AiofContractError(
            AiofStatus.INVALID_CARD_INPUT,
            "board and dead cards must be disjoint",
        )
    if len(board_ids) + len(dead_ids) > aiof_limits.max_dead_cards:
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            "board plus dead cards exceed the configured dead-card cap",
        )
    canonical_board = tuple(card_from_id(value) for value in sorted(board_ids))
    canonical_dead = tuple(card_from_id(value) for value in sorted(dead_ids))
    unavailable = tuple(
        card_from_id(value) for value in sorted(board_ids + dead_ids)
    )
    return canonical_board, canonical_dead, unavailable


def _expanded_seat(
    player_id: str,
    range_spec: RangeSpec,
    board: tuple[str, ...],
    unavailable: tuple[str, ...],
    aiof_limits: AiofLimits,
) -> _ExpandedSeat:
    return _ExpandedSeat(
        player_id=player_id,
        initial=expand_range(range_spec, (), aiof_limits),
        after_board=expand_range(range_spec, board, aiof_limits),
        final=expand_range(range_spec, unavailable, aiof_limits),
    )


def _range_provenance(seat: _ExpandedSeat) -> ThreePlayerRangeProvenance:
    return ThreePlayerRangeProvenance(
        player_id=seat.player_id,
        initial_range_identity=seat.initial.content_identity,
        after_board_range_identity=seat.after_board.content_identity,
        surviving_range_identity=seat.final.content_identity,
        initial_combo_count=len(seat.initial.combos),
        after_board_combo_count=len(seat.after_board.combos),
        surviving_combo_count=len(seat.final.combos),
        board_removed_combo_count=(
            len(seat.initial.combos) - len(seat.after_board.combos)
        ),
        dead_removed_combo_count=(
            len(seat.after_board.combos) - len(seat.final.combos)
        ),
        raw_mass_before_removal=seat.initial.raw_mass_before_dead,
        raw_mass_after_removal=seat.final.raw_mass_after_dead,
    )


def _raw_mass_fraction(
    combo: ExpandedCombo,
    limits: KnownBoardRealCardThreePlayerLimits,
) -> Fraction:
    value = _check_fraction_bits(
        Fraction.from_float(combo.raw_mass),
        f"range combo {combo.combo} mass",
        limits,
    )
    if value <= 0:
        raise AiofContractError(
            AiofStatus.NUMERIC_FAILURE, "expanded combo mass is not positive"
        )
    return value


def _cards_disjoint(*combos: ExpandedCombo) -> bool:
    cards = [card for combo in combos for card in combo.card_ids]
    return len(cards) == len(set(cards))


def _prepare_support_internal(
    *,
    board: tuple[str, ...],
    dead_cards: tuple[str, ...],
    hero_range: RangeSpec,
    o1_range: RangeSpec,
    o2_range: RangeSpec,
    limits: KnownBoardRealCardThreePlayerLimits,
    aiof_limits: AiofLimits,
) -> _PreparedInternal:
    limits = _validate_limits(limits)
    if type(aiof_limits) is not AiofLimits:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, "aiof_limits must be AiofLimits"
        )
    canonical_board, canonical_dead, unavailable = _canonical_board_dead(
        board, dead_cards, aiof_limits
    )
    seats = (
        _expanded_seat(
            "H", hero_range, canonical_board, unavailable, aiof_limits
        ),
        _expanded_seat(
            "O1", o1_range, canonical_board, unavailable, aiof_limits
        ),
        _expanded_seat(
            "O2", o2_range, canonical_board, unavailable, aiof_limits
        ),
    )
    combo_sets = tuple(seat.final.combos for seat in seats)
    cartesian = (
        len(combo_sets[0]) * len(combo_sets[1]) * len(combo_sets[2])
    )
    if cartesian > limits.max_cartesian_triples:
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            "Cartesian triple cap exceeded before triple enumeration",
        )

    compatible_count = 0
    collision_count = 0
    compatible_mass = Fraction(0)
    marginal_mass = {
        seat.player_id: {combo.combo: Fraction(0) for combo in seat.final.combos}
        for seat in seats
    }
    for hero in combo_sets[0]:
        hero_mass = _raw_mass_fraction(hero, limits)
        for o1 in combo_sets[1]:
            o1_mass = _raw_mass_fraction(o1, limits)
            for o2 in combo_sets[2]:
                if not _cards_disjoint(hero, o1, o2):
                    collision_count += 1
                    continue
                compatible_count += 1
                if compatible_count > limits.max_compatible_triples:
                    raise AiofContractError(
                        AiofStatus.CAP_EXCEEDED,
                        "compatible triple cap exceeded before support allocation",
                    )
                joint = _check_fraction_bits(
                    hero_mass
                    * o1_mass
                    * _raw_mass_fraction(o2, limits),
                    "compatible triple raw mass",
                    limits,
                )
                compatible_mass += joint
                marginal_mass["H"][hero.combo] += joint
                marginal_mass["O1"][o1.combo] += joint
                marginal_mass["O2"][o2.combo] += joint
    if compatible_count == 0:
        raise AiofContractError(
            AiofStatus.EMPTY_COMPATIBLE_SUPPORT,
            "compatible three-player support is empty",
        )
    if compatible_mass <= 0:
        raise AiofContractError(
            AiofStatus.NUMERIC_FAILURE,
            "compatible three-player mass is not positive",
        )
    if any(
        mass <= 0
        for seat_marginals in marginal_mass.values()
        for mass in seat_marginals.values()
    ):
        raise AiofContractError(
            AiofStatus.ZERO_COMPATIBLE_MARGINAL,
            "a surviving combo has zero compatible triple marginal",
        )

    triples: list[CompatibleComboTriple] = []
    compatible_mass = _check_fraction_bits(
        compatible_mass, "compatible support total mass", limits
    )
    for hero in combo_sets[0]:
        hero_mass = _raw_mass_fraction(hero, limits)
        for o1 in combo_sets[1]:
            o1_mass = _raw_mass_fraction(o1, limits)
            for o2 in combo_sets[2]:
                if not _cards_disjoint(hero, o1, o2):
                    continue
                joint = _check_fraction_bits(
                    hero_mass
                    * o1_mass
                    * _raw_mass_fraction(o2, limits),
                    "compatible triple raw mass",
                    limits,
                )
                probability = _check_fraction_bits(
                    joint / compatible_mass,
                    "compatible triple probability",
                    limits,
                )
                triples.append(
                    CompatibleComboTriple(
                        hero_combo=hero.combo,
                        o1_combo=o1.combo,
                        o2_combo=o2.combo,
                        raw_joint_mass_exact=_rational_text(joint),
                        probability_exact=_rational_text(probability),
                    )
                )
    if len(triples) != compatible_count:
        raise AiofContractError(
            AiofStatus.ACCOUNTING_MISMATCH,
            "compatible triple count changed during materialization",
        )
    probability_sum = sum(
        (Fraction(row.probability_exact) for row in triples), Fraction(0)
    )
    if probability_sum != 1:
        raise AiofContractError(
            AiofStatus.ACCOUNTING_MISMATCH,
            "conditioned triple probabilities do not sum exactly to one",
        )

    marginals: list[ThreePlayerComboMarginal] = []
    for seat in seats:
        for combo in seat.final.combos:
            mass = _check_fraction_bits(
                marginal_mass[seat.player_id][combo.combo],
                f"{seat.player_id} compatible marginal mass",
                limits,
            )
            probability = _check_fraction_bits(
                mass / compatible_mass,
                f"{seat.player_id} conditional marginal probability",
                limits,
            )
            marginals.append(
                ThreePlayerComboMarginal(
                    player_id=seat.player_id,
                    combo=combo.combo,
                    card_ids=combo.card_ids,
                    raw_mass=combo.raw_mass,
                    compatible_raw_mass_exact=_rational_text(mass),
                    probability_exact=_rational_text(probability),
                )
            )
    for player in PLAYERS:
        if (
            sum(
                (
                    Fraction(row.probability_exact)
                    for row in marginals
                    if row.player_id == player
                ),
                Fraction(0),
            )
            != 1
        ):
            raise AiofContractError(
                AiofStatus.ACCOUNTING_MISMATCH,
                f"{player} conditional marginals do not sum exactly to one",
            )

    provenance = tuple(_range_provenance(seat) for seat in seats)
    identity_payload = {
        "identity_version": IDENTITY_VERSION,
        "joint_conditioning_version": JOINT_CONDITIONING_VERSION,
        "board": canonical_board,
        "dead_cards": canonical_dead,
        "range_provenance": provenance,
        "cartesian_triple_count": cartesian,
        "private_collision_excluded_count": collision_count,
        "compatible_raw_joint_mass": compatible_mass,
        "triples": tuple(triples),
        "marginals": tuple(marginals),
    }
    public = PreparedThreePlayerSupport(
        board=canonical_board,
        dead_cards=canonical_dead,
        unavailable_cards=unavailable,
        range_provenance=provenance,
        cartesian_triple_count=cartesian,
        private_collision_excluded_count=collision_count,
        compatible_triple_count=compatible_count,
        compatible_raw_joint_mass_exact=_rational_text(compatible_mass),
        triples=tuple(triples),
        marginals=tuple(marginals),
        content_identity=_identity(identity_payload),
    )
    return _PreparedInternal(public=public, seats=seats)


def prepare_known_board_three_player_support(
    *,
    board: tuple[str, ...],
    dead_cards: tuple[str, ...],
    hero_range: RangeSpec,
    o1_range: RangeSpec,
    o2_range: RangeSpec,
    limits: KnownBoardRealCardThreePlayerLimits = (
        KnownBoardRealCardThreePlayerLimits()
    ),
    aiof_limits: AiofLimits = AiofLimits(),
) -> PreparedThreePlayerSupport:
    """Prepare complete compatible H/O1/O2 support without betting analysis."""

    return _prepare_support_internal(
        board=board,
        dead_cards=dead_cards,
        hero_range=hero_range,
        o1_range=o1_range,
        o2_range=o2_range,
        limits=limits,
        aiof_limits=aiof_limits,
    ).public


def _bucket_id(value: object) -> str:
    if type(value) is not str or _BUCKET_RE.fullmatch(value) is None:
        raise AiofContractError(
            AiofStatus.INVALID_STRATEGY, f"invalid bucket id {value!r}"
        )
    return value


def _canonical_bucket_map(
    value: object,
    combos: tuple[str, ...],
    player: str,
    maximum: int,
) -> ComboBucketMap:
    if value is None:
        if len(combos) > maximum:
            raise AiofContractError(
                AiofStatus.CAP_EXCEEDED, f"{player} bucket cap exceeded"
            )
        return ComboBucketMap(
            bucket_ids=combos,
            assignments=tuple(
                ComboBucketAssignment(combo, combo) for combo in combos
            ),
        )
    if type(value) is not ComboBucketMap:
        raise AiofContractError(
            AiofStatus.INVALID_STRATEGY,
            f"{player} combo map must be ComboBucketMap or None",
        )
    if type(value.bucket_ids) is not tuple or type(value.assignments) is not tuple:
        raise AiofContractError(
            AiofStatus.INVALID_STRATEGY,
            f"{player} combo map fields must be tuples",
        )
    buckets = tuple(_bucket_id(bucket) for bucket in value.bucket_ids)
    if len(buckets) != len(set(buckets)):
        raise AiofContractError(
            AiofStatus.INVALID_STRATEGY, f"duplicate {player} bucket id"
        )
    if len(buckets) > maximum:
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED, f"{player} bucket cap exceeded"
        )
    assignments: dict[str, str] = {}
    for row in value.assignments:
        if type(row) is not ComboBucketAssignment:
            raise AiofContractError(
                AiofStatus.INVALID_STRATEGY,
                f"invalid {player} combo map row",
            )
        combo = canonicalize_exact_combo(row.combo)
        bucket = _bucket_id(row.bucket_id)
        if combo in assignments:
            raise AiofContractError(
                AiofStatus.INVALID_STRATEGY,
                f"duplicate {player} combo assignment",
            )
        assignments[combo] = bucket
    if set(assignments) != set(combos):
        raise AiofContractError(
            AiofStatus.INVALID_STRATEGY,
            f"{player} combo map support mismatch",
        )
    if set(assignments.values()) - set(buckets):
        raise AiofContractError(
            AiofStatus.INVALID_STRATEGY,
            f"{player} combo map references an unknown bucket",
        )
    if set(buckets) - set(assignments.values()):
        raise AiofContractError(
            AiofStatus.INVALID_STRATEGY,
            f"{player} combo map declares an empty bucket",
        )
    return ComboBucketMap(
        bucket_ids=tuple(sorted(buckets)),
        assignments=tuple(
            ComboBucketAssignment(combo, assignments[combo])
            for combo in sorted(assignments)
        ),
    )


def _bucket_map_dict(value: ComboBucketMap) -> dict[str, Any]:
    return {
        "bucket_ids": list(value.bucket_ids),
        "assignments": [
            {"combo": row.combo, "bucket_id": row.bucket_id}
            for row in value.assignments
        ],
    }


def _combo_bucket_lookup(value: ComboBucketMap) -> dict[str, str]:
    return {row.combo: row.bucket_id for row in value.assignments}


def _info_id(player: str, decision: str, bucket: str) -> str:
    return f"{INFO_PREFIX[(player, decision)]}::{bucket}"


def _canonical_profile(
    profile: object,
    bucket_maps: Mapping[str, ComboBucketMap],
    limits: KnownBoardRealCardThreePlayerLimits,
) -> tuple[
    RealCardThreePlayerProfile,
    _m31.ExactBehaviorPolicy,
    _m31.OpponentInitialProfile,
]:
    if type(profile) is not RealCardThreePlayerProfile or type(profile.rows) is not tuple:
        raise AiofContractError(
            AiofStatus.INVALID_STRATEGY,
            "baseline_profile must be RealCardThreePlayerProfile",
        )
    if len(profile.rows) > limits.max_profile_rows:
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            "profile row cap exceeded before profile allocation",
        )
    expected = {
        (player, bucket, decision)
        for player in PLAYERS
        for bucket in bucket_maps[player].bucket_ids
        for decision in DECISIONS[player]
    }
    canonical: dict[
        tuple[str, str, str], tuple[ExactActionProbability, ...]
    ] = {}
    for row in profile.rows:
        if type(row) is not RealCardProfileRow or type(row.actions) is not tuple:
            raise AiofContractError(
                AiofStatus.INVALID_STRATEGY, "invalid baseline profile row"
            )
        if row.player_id not in PLAYERS:
            raise AiofContractError(
                AiofStatus.INVALID_STRATEGY, "invalid profile player"
            )
        bucket = _bucket_id(row.bucket_id)
        if row.decision not in DECISIONS[row.player_id]:
            raise AiofContractError(
                AiofStatus.INVALID_STRATEGY, "invalid profile decision"
            )
        key = (row.player_id, bucket, row.decision)
        if key in canonical:
            raise AiofContractError(
                AiofStatus.INVALID_STRATEGY, "duplicate profile row"
            )
        actions: dict[str, Fraction] = {}
        for item in row.actions:
            if type(item) is not ExactActionProbability:
                raise AiofContractError(
                    AiofStatus.INVALID_STRATEGY,
                    "invalid action probability row",
                )
            if type(item.action) is not str or item.action in actions:
                raise AiofContractError(
                    AiofStatus.INVALID_STRATEGY,
                    "duplicate or invalid profile action",
                )
            actions[item.action] = _parse_rational(
                item.probability,
                "profile probability",
                minimum=Fraction(0),
                maximum=Fraction(1),
                limits=limits,
            )
        legal = DECISIONS[row.player_id][row.decision]
        if set(actions) != set(legal):
            raise AiofContractError(
                AiofStatus.INVALID_STRATEGY,
                "profile action keys do not match the decision contract",
            )
        if sum((actions[action] for action in legal), Fraction(0)) != 1:
            raise AiofContractError(
                AiofStatus.INVALID_STRATEGY,
                "profile probabilities must sum exactly to one",
            )
        canonical[key] = tuple(
            ExactActionProbability(action, _rational_text(actions[action]))
            for action in legal
        )
    if set(canonical) != expected:
        raise AiofContractError(
            AiofStatus.INVALID_STRATEGY,
            "baseline profile coverage is incomplete or stale",
        )

    rows = tuple(
        RealCardProfileRow(player, bucket, decision, canonical[(player, bucket, decision)])
        for player in PLAYERS
        for bucket in sorted(bucket_maps[player].bucket_ids)
        for decision in DECISIONS[player]
    )
    hero = {
        _info_id("H", decision, bucket): {
            action.action: action.probability
            for action in canonical[("H", bucket, decision)]
        }
        for bucket in sorted(bucket_maps["H"].bucket_ids)
        for decision in DECISIONS["H"]
    }
    o1 = {
        _info_id("O1", decision, bucket): {
            action.action: action.probability
            for action in canonical[("O1", bucket, decision)]
        }
        for bucket in sorted(bucket_maps["O1"].bucket_ids)
        for decision in DECISIONS["O1"]
    }
    o2 = {
        _info_id("O2", decision, bucket): {
            action.action: action.probability
            for action in canonical[("O2", bucket, decision)]
        }
        for bucket in sorted(bucket_maps["O2"].bucket_ids)
        for decision in DECISIONS["O2"]
    }
    return (
        RealCardThreePlayerProfile(rows),
        _m31.ExactBehaviorPolicy(hero),
        _m31.OpponentInitialProfile(o1, o2),
    )


def _rank_cache(
    prepared: _PreparedInternal,
    limits: KnownBoardRealCardThreePlayerLimits,
) -> dict[str, HandRank]:
    combos = sorted(
        {
            marginal.combo
            for marginal in prepared.public.marginals
        }
    )
    if len(combos) > limits.max_fixed_board_evaluations:
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            "fixed-board evaluation cap exceeded before hand ranking",
        )
    board = prepared.public.board
    return {
        combo: evaluate_seven_card_hand(
            (combo[:2], combo[2:]) + board
        )
        for combo in combos
    }


def _winners(
    ranks: Mapping[str, HandRank], active_players: tuple[str, ...]
) -> tuple[str, ...]:
    maximum = max(ranks[player] for player in active_players)
    return tuple(
        player for player in active_players if ranks[player] == maximum
    )


def _award_shares(
    ranks: Mapping[str, HandRank], active_players: tuple[str, ...]
) -> tuple[_m31.AwardShare, ...]:
    winners = _winners(ranks, active_players)
    share = _rational_text(Fraction(1, len(winners)))
    return tuple(_m31.AwardShare(player, share) for player in winners)


def _showdown_rows(
    prepared: _PreparedInternal,
    bucket_lookups: Mapping[str, Mapping[str, str]],
    ranks: Mapping[str, HandRank],
) -> tuple[KnownBoardShowdownRow, ...]:
    rows: list[KnownBoardShowdownRow] = []
    for triple in prepared.public.triples:
        per_player = {
            "H": ranks[triple.hero_combo],
            "O1": ranks[triple.o1_combo],
            "O2": ranks[triple.o2_combo],
        }
        heads_up = tuple(
            (
                "-".join(active),
                _winners(per_player, active),
            )
            for active in (("H", "O1"), ("H", "O2"), ("O1", "O2"))
        )
        rows.append(
            KnownBoardShowdownRow(
                hero_combo=triple.hero_combo,
                o1_combo=triple.o1_combo,
                o2_combo=triple.o2_combo,
                hero_bucket_id=bucket_lookups["H"][triple.hero_combo],
                o1_bucket_id=bucket_lookups["O1"][triple.o1_combo],
                o2_bucket_id=bucket_lookups["O2"][triple.o2_combo],
                probability_exact=triple.probability_exact,
                hero_rank=per_player["H"],
                o1_rank=per_player["O1"],
                o2_rank=per_player["O2"],
                three_way_winners=_winners(per_player, PLAYERS),
                heads_up_winners=heads_up,
            )
        )
    return tuple(rows)


def _terminal(
    suffix: str,
    label: str,
    ranks: Mapping[str, HandRank],
    active: tuple[str, ...],
) -> _m31.RiverTerminalNode:
    if len(active) == 1:
        return _m31.RiverTerminalNode(
            node_id=f"terminal-{label}-{suffix}",
            kind="fold",
        )
    return _m31.RiverTerminalNode(
        node_id=f"terminal-{label}-{suffix}",
        kind="showdown",
        award_shares=_award_shares(ranks, active),
    )


def _betting_child(
    suffix: str,
    row: KnownBoardShowdownRow,
    bet_to: str,
) -> _m31.RiverDecisionNode:
    ranks = {"H": row.hero_rank, "O1": row.o1_rank, "O2": row.o2_rank}
    check_terminal = _terminal(suffix, "check", ranks, PLAYERS)
    o2_after_check = _m31.RiverDecisionNode(
        node_id=f"node-O2-after-check-{suffix}",
        owner="O2",
        information_set_id=_info_id(
            "O2", "after_hero_o1_check", row.o2_bucket_id
        ),
        actions=(
            _m31.RiverAction(
                "check", "check", None, check_terminal
            ),
        ),
    )
    o1_after_check = _m31.RiverDecisionNode(
        node_id=f"node-O1-after-check-{suffix}",
        owner="O1",
        information_set_id=_info_id(
            "O1", "after_hero_check", row.o1_bucket_id
        ),
        actions=(
            _m31.RiverAction(
                "check", "check", None, o2_after_check
            ),
        ),
    )

    o2_after_o1_call = _m31.RiverDecisionNode(
        node_id=f"node-O2-after-O1-call-{suffix}",
        owner="O2",
        information_set_id=_info_id(
            "O2", "vs_hero_bet_o1_call", row.o2_bucket_id
        ),
        actions=(
            _m31.RiverAction(
                "call",
                "call",
                bet_to,
                _terminal(suffix, "both-call", ranks, PLAYERS),
            ),
            _m31.RiverAction(
                "fold",
                "fold",
                None,
                _terminal(suffix, "O1-call-O2-fold", ranks, ("H", "O1")),
            ),
        ),
    )
    o2_after_o1_fold = _m31.RiverDecisionNode(
        node_id=f"node-O2-after-O1-fold-{suffix}",
        owner="O2",
        information_set_id=_info_id(
            "O2", "vs_hero_bet_o1_fold", row.o2_bucket_id
        ),
        actions=(
            _m31.RiverAction(
                "call",
                "call",
                bet_to,
                _terminal(suffix, "O1-fold-O2-call", ranks, ("H", "O2")),
            ),
            _m31.RiverAction(
                "fold",
                "fold",
                None,
                _terminal(suffix, "both-fold", ranks, ("H",)),
            ),
        ),
    )
    o1_vs_bet = _m31.RiverDecisionNode(
        node_id=f"node-O1-vs-bet-{suffix}",
        owner="O1",
        information_set_id=_info_id(
            "O1", "vs_hero_bet", row.o1_bucket_id
        ),
        actions=(
            _m31.RiverAction(
                "call", "call", bet_to, o2_after_o1_call
            ),
            _m31.RiverAction(
                "fold", "fold", None, o2_after_o1_fold
            ),
        ),
    )
    return _m31.RiverDecisionNode(
        node_id=f"node-H-open-{suffix}",
        owner="H",
        information_set_id=_info_id("H", "open", row.hero_bucket_id),
        actions=(
            _m31.RiverAction(
                "check", "check", None, o1_after_check
            ),
            _m31.RiverAction(
                "bet", "bet", bet_to, o1_vs_bet
            ),
        ),
    )


def _validate_attestation_input(value: object) -> GeneratedTreeAttestation:
    if type(value) is not GeneratedTreeAttestation:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            "attestation must be GeneratedTreeAttestation",
        )
    if (
        type(value.verifier) is not str
        or not value.verifier
        or len(value.verifier) > 128
        or type(value.evidence_version) is not str
        or not value.evidence_version
        or len(value.evidence_version) > 128
        or type(value.verification_date) is not str
        or _DATE_RE.fullmatch(value.verification_date) is None
        or type(value.o1_confirmed) is not bool
        or type(value.o2_confirmed) is not bool
        or not value.o1_confirmed
        or not value.o2_confirmed
    ):
        raise AiofContractError(
            AiofStatus.UNSUPPORTED_MODEL,
            "complete human-traceable O1/O2 perfect-recall evidence is required",
        )
    return value


def _m32_status(status: str) -> AiofStatus:
    if status == _m32.CAP_EXCEEDED:
        return AiofStatus.CAP_EXCEEDED
    if status == _m32.UNSUPPORTED_MODE:
        return AiofStatus.UNSUPPORTED_MODEL
    if status in (_m32.INVALID_INPUT, _m32.STALE_INPUT):
        return AiofStatus.INVALID_INPUT
    if status in (
        _m32.NUMERIC_FAILURE,
        _m32.SCALAR_PROJECTION_FAILURE,
    ):
        return AiofStatus.NUMERIC_FAILURE
    if status in (
        _m32.BASELINE_RESPONSE_FAILURE,
        _m32.CANDIDATE_RESPONSE_FAILURE,
        _m32.SELECTOR_FAILURE,
    ):
        return AiofStatus.ORACLE_MISMATCH
    return AiofStatus.NUMERIC_FAILURE


def _validate_request_types(
    request: object,
) -> KnownBoardRealCardThreePlayerRequest:
    if type(request) is not KnownBoardRealCardThreePlayerRequest:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            "request must be KnownBoardRealCardThreePlayerRequest",
        )
    _validate_limits(request.limits)
    nested = (
        (request.aiof_limits, AiofLimits, "aiof_limits"),
        (
            request.generation,
            _m32.ThreePlayerCandidateGenerationConfig,
            "generation",
        ),
        (request.repeated, _m32.ThreePlayerRepeatedConfig, "repeated"),
        (
            request.selector_configuration,
            AutomaticCommitmentSelectionConfig,
            "selector_configuration",
        ),
        (
            request.m32_limits,
            _m32.ThreePlayerCandidateRepeatedLimits,
            "m32_limits",
        ),
        (request.m31_limits, _m31.RiverRakeLimits, "m31_limits"),
        (request.m30_limits, _m30.ExactResponseLimits, "m30_limits"),
    )
    for value, expected, name in nested:
        if type(value) is not expected:
            raise AiofContractError(
                AiofStatus.INVALID_INPUT, f"{name} has the wrong type"
            )
    if request.expected_baseline_identity is not None and (
        type(request.expected_baseline_identity) is not str
        or _IDENTITY_RE.fullmatch(request.expected_baseline_identity) is None
    ):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            "expected_baseline_identity must be a lowercase SHA-256 identity",
        )
    _validate_attestation_input(request.attestation)
    return request


def _execute(
    request: KnownBoardRealCardThreePlayerRequest,
) -> KnownBoardRealCardThreePlayerPayload:
    request = _validate_request_types(request)
    limits = request.limits
    initial = _parse_rational(
        request.initial_contribution,
        "initial_contribution",
        minimum=Fraction(0),
        limits=limits,
    )
    if initial <= 0:
        raise AiofContractError(
            AiofStatus.UNSUPPORTED_MODEL,
            "initial contribution must be positive",
        )
    bet_to = _parse_rational(
        request.bet_to,
        "bet_to",
        minimum=Fraction(0),
        limits=limits,
    )
    if bet_to <= initial:
        raise AiofContractError(
            AiofStatus.UNSUPPORTED_MODEL,
            "bet_to must exceed the equal initial contribution",
        )
    rake_rate = _parse_rational(
        request.rake_rate,
        "rake_rate",
        minimum=Fraction(0),
        maximum=Fraction(1),
        limits=limits,
    )
    rake_cap = (
        None
        if request.rake_cap is None
        else _parse_rational(
            request.rake_cap,
            "rake_cap",
            minimum=Fraction(0),
            limits=limits,
        )
    )

    prepared = _prepare_support_internal(
        board=request.board,
        dead_cards=request.dead_cards,
        hero_range=request.hero_range,
        o1_range=request.o1_range,
        o2_range=request.o2_range,
        limits=limits,
        aiof_limits=request.aiof_limits,
    )
    combo_by_player = {
        seat.player_id: tuple(combo.combo for combo in seat.final.combos)
        for seat in prepared.seats
    }
    bucket_maps = {
        "H": _canonical_bucket_map(
            request.hero_combo_to_bucket,
            combo_by_player["H"],
            "H",
            limits.max_buckets_per_seat,
        ),
        "O1": _canonical_bucket_map(
            request.o1_combo_to_bucket,
            combo_by_player["O1"],
            "O1",
            limits.max_buckets_per_seat,
        ),
        "O2": _canonical_bucket_map(
            request.o2_combo_to_bucket,
            combo_by_player["O2"],
            "O2",
            limits.max_buckets_per_seat,
        ),
    }
    canonical_profile, hero_policy, initial_profile = _canonical_profile(
        request.baseline_profile, bucket_maps, limits
    )
    bucket_lookups = {
        player: _combo_bucket_lookup(bucket_maps[player])
        for player in PLAYERS
    }

    triple_count = prepared.public.compatible_triple_count
    projected_nodes = 1 + 11 * triple_count
    if (
        projected_nodes > limits.max_tree_nodes
        or projected_nodes > request.m31_limits.max_nodes
    ):
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            "tree-node cap exceeded before ranking or tree allocation",
        )
    if triple_count > request.m31_limits.max_chance_outcomes:
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            "M31 chance-outcome cap exceeded before tree allocation",
        )
    hero_info_sets = len(bucket_maps["H"].bucket_ids)
    o1_info_sets = 2 * len(bucket_maps["O1"].bucket_ids)
    o2_info_sets = 3 * len(bucket_maps["O2"].bucket_ids)
    if (
        hero_info_sets > request.m31_limits.max_fixed_hero_info_sets
        or o1_info_sets > request.m31_limits.max_info_sets_per_opponent
        or o2_info_sets > request.m31_limits.max_info_sets_per_opponent
        or o1_info_sets + o2_info_sets
        > request.m31_limits.max_opponent_info_sets_total
    ):
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            "M31 information-set cap exceeded before tree allocation",
        )
    o1_plan_count = 2 ** len(bucket_maps["O1"].bucket_ids)
    o2_plan_count = 4 ** len(bucket_maps["O2"].bucket_ids)
    if (
        o1_plan_count > request.m31_limits.max_pure_plans_o1
        or o2_plan_count > request.m31_limits.max_pure_plans_o2
        or o1_plan_count * o2_plan_count
        > request.m31_limits.max_joint_pure_profiles
    ):
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            "M31 pure-plan cap exceeded before tree allocation",
        )
    identity_records = (
        triple_count * 32
        + sum(len(mapping.assignments) for mapping in bucket_maps.values())
        + len(canonical_profile.rows) * 8
        + 256
    )
    if identity_records > limits.max_identity_records:
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            "identity-record cap exceeded before rank/tree allocation",
        )

    ranks = _rank_cache(prepared, limits)
    showdown_rows = _showdown_rows(prepared, bucket_lookups, ranks)
    outcomes = tuple(
        _m31.RiverChanceOutcome(
            outcome_id=f"triple-{index:04d}",
            probability=row.probability_exact,
            observation=_m31.RiverObservation(
                public_observation_id=(
                    f"known-board::{_identity(prepared.public.board)[:24]}"
                ),
                private_observation_id_by_player={
                    "H": f"bucket::{row.hero_bucket_id}",
                    "O1": f"bucket::{row.o1_bucket_id}",
                    "O2": f"bucket::{row.o2_bucket_id}",
                },
            ),
            child=_betting_child(
                f"{index:04d}", row, _rational_text(bet_to)
            ),
        )
        for index, row in enumerate(showdown_rows, 1)
    )
    scenario = _m31.ThreePlayerRiverRakeScenario(
        root=_m31.RiverChanceNode("known-board-triple-root", outcomes),
        button_player_id="O2",
        seat_order=("H", "O1", "O2"),
        river_action_order=("H", "O1", "O2"),
        initial_observation=None,
        initial_pot=_rational_text(3 * initial),
        initial_contribution={
            player: _rational_text(initial) for player in PLAYERS
        },
        max_total_contribution={
            player: _rational_text(bet_to + initial) for player in PLAYERS
        },
        rake_rate=_rational_text(rake_rate),
        rake_cap=None if rake_cap is None else _rational_text(rake_cap),
    )
    evidence = _m31.create_perfect_recall_attestation(
        scenario,
        verifier=request.attestation.verifier,
        verification_date=request.attestation.verification_date,
        evidence_version=request.attestation.evidence_version,
        o1_confirmed=request.attestation.o1_confirmed,
        o2_confirmed=request.attestation.o2_confirmed,
        limits=request.m31_limits,
    )
    profile_identity = _identity(
        {
            "profile_version": PROFILE_VERSION,
            "profile": canonical_profile,
            "bucket_maps": bucket_maps,
        }
    )
    scenario_parameters_identity = _identity(
        {
            "betting_tree_version": BETTING_TREE_VERSION,
            "initial_contribution": initial,
            "bet_to": bet_to,
            "rake_rate": rake_rate,
            "rake_cap": rake_cap,
            "prepared_support": prepared.public.content_identity,
            "bucket_maps": bucket_maps,
        }
    )
    baseline_identity = _identity(
        {
            "identity_version": IDENTITY_VERSION,
            "prepared_support": prepared.public.content_identity,
            "profile": profile_identity,
            "scenario_parameters": scenario_parameters_identity,
            "perfect_recall": asdict(evidence),
        }
    )
    if (
        request.expected_baseline_identity is not None
        and request.expected_baseline_identity != baseline_identity
    ):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, "expected baseline identity is stale"
        )

    native = _m32.evaluate_three_player_candidate_repeated(
        scenario,
        hero_policy,
        initial_profile,
        attestation=evidence,
        generation=request.generation,
        repeated=request.repeated,
        selector_configuration=request.selector_configuration,
        limits=request.m32_limits,
        m31_limits=request.m31_limits,
        m30_limits=request.m30_limits,
    )
    if (
        native.status
        != _m32.EXACT_THREE_PLAYER_CANDIDATE_REPEATED_COMPLETE
        or native.analysis is None
        or native.error is not None
        or native.partial_result
    ):
        message = (
            native.status
            if native.error is None
            else f"{native.status}: {native.error.phase}: {native.error.message}"
        )
        raise AiofContractError(_m32_status(native.status), message)

    native_identity = _identity(native.to_dict())
    analysis_identity = _identity(
        {
            "identity_version": IDENTITY_VERSION,
            "baseline_identity": baseline_identity,
            "generation": request.generation,
            "repeated": request.repeated,
            "selector_configuration": request.selector_configuration,
            "native_m32_result": native_identity,
        }
    )
    payload = KnownBoardRealCardThreePlayerPayload(
        contract_version=KNOWN_BOARD_REAL_CARD_THREE_PLAYER_RIVER_CONTRACT_VERSION,
        claim_scope=CLAIM_SCOPE,
        prepared_support=prepared.public,
        showdown_rows=showdown_rows,
        canonical_profile=canonical_profile,
        bucket_maps=bucket_maps,
        workload={
            "cartesian_triples": prepared.public.cartesian_triple_count,
            "compatible_triples": triple_count,
            "fixed_board_evaluations": len(ranks),
            "tree_nodes": projected_nodes,
            "profile_rows": len(canonical_profile.rows),
            "o1_pure_plans": o1_plan_count,
            "o2_pure_plans": o2_plan_count,
            "joint_pure_profiles": o1_plan_count * o2_plan_count,
        },
        baseline_identity=baseline_identity,
        native_m32_result=native,
        identities={
            "prepared_support": prepared.public.content_identity,
            "profile": profile_identity,
            "scenario_parameters": scenario_parameters_identity,
            "baseline": baseline_identity,
            "native_m32_result": native_identity,
            "analysis": analysis_identity,
            "evaluator": _identity({"evaluator": EVALUATOR_ID}),
        },
    )
    encoded = json.dumps(
        KnownBoardRealCardThreePlayerResult(
            AiofStatus.SUCCESS, payload, None
        ).to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > limits.max_output_bytes:
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED,
            "outer output byte cap exceeded with no partial payload",
        )
    return payload


def _clean_error(message: str, fallback: str) -> str:
    value = " ".join(str(message).split())
    return (value or fallback)[:512]


def analyze_known_board_real_card_three_player_river(
    request: KnownBoardRealCardThreePlayerRequest,
) -> KnownBoardRealCardThreePlayerResult:
    """Run complete real-card M31/M30/M32 analysis or return no payload."""

    try:
        return KnownBoardRealCardThreePlayerResult(
            AiofStatus.SUCCESS, _execute(request), None
        )
    except AiofContractError as exc:
        return KnownBoardRealCardThreePlayerResult(
            exc.status,
            None,
            _clean_error(str(exc), exc.status.value),
        )
    except (ArithmeticError, OverflowError, ValueError) as exc:
        return KnownBoardRealCardThreePlayerResult(
            AiofStatus.NUMERIC_FAILURE,
            None,
            _clean_error(str(exc), "numeric failure"),
        )
    except Exception:
        return KnownBoardRealCardThreePlayerResult(
            AiofStatus.NUMERIC_FAILURE,
            None,
            "unexpected known-board three-player analysis failure",
        )


def exact_known_board_real_card_three_player_json(
    result: KnownBoardRealCardThreePlayerResult,
) -> str:
    """Serialize an outer result as deterministic strict one-line JSON."""

    if type(result) is not KnownBoardRealCardThreePlayerResult:
        raise TypeError("result must be KnownBoardRealCardThreePlayerResult")
    return json.dumps(
        result.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
