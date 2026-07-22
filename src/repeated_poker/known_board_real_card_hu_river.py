"""Bounded known-board real-card heads-up river/rake commitment adapter.

The adapter preserves the once-conditioned ordered Hero/Villain combo joint
distribution, evaluates every compatible pair on one fixed five-card board,
and builds a native seven-line river :class:`~repeated_poker.game.GameTree`.
It compares a complete Hero baseline and its declared one-/two-information-set
shift library against one fixed Villain baseline and exact DP responses, then
reuses the M27 automatic commitment selector unchanged.

The result is a conditional comparison over a finite caller-declared lattice.
It is not a board/runout enumerator, an equilibrium solver, a continuous or
global optimizer, an external-solver certificate, or strategy advice.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from numbers import Real
from typing import Any

from .aiof_cards import (
    AiofContractError,
    AiofLimits,
    AiofStatus,
    ExpandedRange,
    PreparedRanges,
    RangeSpec,
    card_from_id,
    card_id,
    canonicalize_exact_combo,
    expand_range,
    prepare_compatible_ranges,
)
from .aiof_evaluator import EVALUATOR_ID, HandRank, evaluate_seven_card_hand
from .automatic_commitment_selection import (
    AutomaticCommitmentSearchCoverage,
    AutomaticCommitmentSelectionConfig,
    AutomaticCommitmentSelectionReport,
    select_automatic_commitments,
    validate_automatic_commitment_selection_parameters,
)
from .candidates import HeroStrategyCandidate, ShiftComponent
from .comparison import (
    CandidateComparison,
    CandidateComparisonReport,
    compare_candidates,
)
from .exact_response import BestResponseResult, solve_exact_response
from .fixed_profile import FixedProfileValue, evaluate_fixed_profile
from .game import (
    ChanceNode,
    GameTree,
    HeroNode,
    HeroStrategy,
    VillainNode,
    VillainStrategy,
    collect_hero_info_sets,
    collect_villain_info_sets,
    iter_nodes,
    validate_hero_strategy,
    validate_tree,
    validate_villain_strategy,
)
from .payoffs import (
    CHOP,
    HERO,
    VILLAIN,
    make_fold_terminal,
    make_showdown_terminal,
)
from .repeated import DEFAULT_MAX_HORIZON


__all__ = [
    "KNOWN_BOARD_REAL_CARD_HU_RIVER_CONTRACT_VERSION",
    "ActionProbability",
    "RiverProfileRow",
    "RiverActionProfile",
    "ComboBucketAssignment",
    "ComboBucketMap",
    "KnownBoardRealCardHuRiverLimits",
    "KnownBoardRealCardHuRiverRequest",
    "RangeRemovalProvenance",
    "JointSupportProvenance",
    "KnownBoardJointRow",
    "KnownBoardRiverWorkloadProjection",
    "KnownBoardRiverCandidateCoverage",
    "KnownBoardRiverBaselineRecord",
    "KnownBoardRiverCandidateRecord",
    "KnownBoardRealCardHuRiverPayload",
    "KnownBoardRealCardHuRiverResult",
    "analyze_known_board_real_card_hu_river",
]


KNOWN_BOARD_REAL_CARD_HU_RIVER_CONTRACT_VERSION = (
    "known-board-real-card-hu-river-rake-adapter-v1"
)
KNOWN_BOARD_RIVER_CLAIM_SCOPE = (
    "bounded_known_board_joint_combo_candidate_library_conditional_decision"
)
KNOWN_BOARD_RIVER_ACCOUNTING = "heads-up-river-rake-net-chips-v1"
KNOWN_BOARD_RIVER_UNIT = "net_chips_before_initial_commitments_per_river_opportunity"
KNOWN_BOARD_RIVER_POSITION = "hero_ip_villain_oop"
KNOWN_BOARD_RIVER_RESPONSE_METHOD = "dp"

BOARD_IDENTITY_ALGORITHM = "known-board-sha256-canonical-json-v1"
JOINT_IDENTITY_ALGORITHM = "known-board-joint-sha256-canonical-json-v1"
MAPPING_IDENTITY_ALGORITHM = "combo-action-bucket-sha256-canonical-json-v1"
TREE_IDENTITY_ALGORITHM = "known-board-river-tree-sha256-canonical-json-v1"
BASELINE_IDENTITY_ALGORITHM = "known-board-river-baseline-sha256-canonical-json-v1"
CANDIDATE_IDENTITY_ALGORITHM = "known-board-river-candidate-sha256-canonical-json-v1"
RESPONSE_IDENTITY_ALGORITHM = "known-board-river-response-sha256-canonical-json-v1"
ANALYSIS_IDENTITY_ALGORITHM = "known-board-river-analysis-sha256-canonical-json-v1"

MAX_JOINT_MATCHUPS = 2_000
MAX_FIXED_BOARD_EVALUATIONS = 2_000
MAX_TREE_NODES = 24_001
MAX_BUCKETS_PER_SEAT = 64
MAX_HERO_INFO_SETS = 128
MAX_VILLAIN_INFO_SETS = 192
MAX_CANDIDATES = 2_000
MAX_CANDIDATE_PROBABILITY_CELLS = 1_000_000
MAX_FIXED_PROFILE_NODE_VISITS = 5_000_000
MAX_RESPONSE_NODE_VISITS = 5_000_000
MAX_TIMING_ROWS = 1_000_000
MAX_BR_LIST_MATERIALIZATION = 100_000

HERO_AFTER_OOP_CHECK = "IP_after_OOP_check"
HERO_VS_OOP_BET = "IP_vs_OOP_bet"
OOP_FIRST = "OOP_first"
OOP_VS_IP_BET = "OOP_vs_IP_bet"
OOP_VS_IP_RAISE = "OOP_vs_IP_raise"

HERO_DECISIONS = {
    "after_oop_check": ("check", "bet"),
    "vs_oop_bet": ("call", "fold", "raise"),
}
VILLAIN_DECISIONS = {
    "oop_first": ("check", "bet"),
    "vs_ip_bet": ("call", "fold"),
    "vs_ip_raise": ("call", "fold"),
}
DECISION_PREFIX = {
    "after_oop_check": HERO_AFTER_OOP_CHECK,
    "vs_oop_bet": HERO_VS_OOP_BET,
    "oop_first": OOP_FIRST,
    "vs_ip_bet": OOP_VS_IP_BET,
    "vs_ip_raise": OOP_VS_IP_RAISE,
}
_BUCKET_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class ActionProbability:
    """One explicit action probability in a complete bucket profile row."""

    action: str
    probability: float

    def to_dict(self) -> dict:
        return {"action": self.action, "probability": self.probability}


@dataclass(frozen=True)
class RiverProfileRow:
    """One bucket/decision distribution; every legal action is mandatory."""

    bucket_id: str
    decision: str
    actions: tuple[ActionProbability, ...]

    def to_dict(self) -> dict:
        return {
            "bucket_id": self.bucket_id,
            "decision": self.decision,
            "actions": [item.to_dict() for item in self.actions],
        }


@dataclass(frozen=True)
class RiverActionProfile:
    """A complete, off-path-inclusive profile for one seat."""

    rows: tuple[RiverProfileRow, ...]

    def to_dict(self) -> dict:
        return {"rows": [row.to_dict() for row in self.rows]}


@dataclass(frozen=True)
class ComboBucketAssignment:
    """Map one surviving exact combo to one declared action-profile bucket."""

    combo: str
    bucket_id: str

    def to_dict(self) -> dict:
        return {"combo": self.combo, "bucket_id": self.bucket_id}


@dataclass(frozen=True)
class ComboBucketMap:
    """Explicit bucket declarations plus complete combo assignments.

    ``bucket_ids`` makes an accidentally empty declared bucket observable and
    rejectable.  Passing ``None`` in the request selects the default one-combo
    per bucket mapping instead.
    """

    bucket_ids: tuple[str, ...]
    assignments: tuple[ComboBucketAssignment, ...]

    def to_dict(self) -> dict:
        return {
            "bucket_ids": list(self.bucket_ids),
            "assignments": [row.to_dict() for row in self.assignments],
        }


@dataclass(frozen=True)
class KnownBoardRealCardHuRiverLimits:
    """Caller-lowerable limits; every field has a frozen hard ceiling."""

    max_joint_matchups: int = MAX_JOINT_MATCHUPS
    max_fixed_board_evaluations: int = MAX_FIXED_BOARD_EVALUATIONS
    max_tree_nodes: int = MAX_TREE_NODES
    max_buckets_per_seat: int = MAX_BUCKETS_PER_SEAT
    max_hero_info_sets: int = MAX_HERO_INFO_SETS
    max_villain_info_sets: int = MAX_VILLAIN_INFO_SETS
    max_candidates: int = MAX_CANDIDATES
    max_candidate_probability_cells: int = MAX_CANDIDATE_PROBABILITY_CELLS
    max_fixed_profile_node_visits: int = MAX_FIXED_PROFILE_NODE_VISITS
    max_response_node_visits: int = MAX_RESPONSE_NODE_VISITS
    max_timing_rows: int = MAX_TIMING_ROWS
    max_br_list_materialization: int = MAX_BR_LIST_MATERIALIZATION

    def to_dict(self) -> dict:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True)
class KnownBoardRealCardHuRiverRequest:
    """One in-memory v1 analysis request for Hero=IP and Villain=OOP."""

    board: tuple[str, ...]
    hero_range: RangeSpec
    villain_range: RangeSpec
    baseline_hero_profile: RiverActionProfile
    dead_cards: tuple[str, ...] = ()
    hero_combo_to_bucket: ComboBucketMap | None = None
    villain_combo_to_bucket: ComboBucketMap | None = None
    baseline_villain_profile: RiverActionProfile | None = None
    initial_commitment_hero: float = 1.0
    initial_commitment_villain: float = 1.0
    rake_rate: float = 0.0
    rake_cap: float | None = None
    oop_bet_size: float = 1.0
    ip_bet_after_check_size: float = 1.0
    ip_raise_to_size: float = 3.0
    shift_amounts: tuple[float, ...] = ()
    max_simultaneous_info_sets: int = 1
    horizon: int = 1
    discount: float = 1.0
    tolerance: float = 1e-9
    minimum_total_uplift: float = 0.0
    aiof_limits: AiofLimits = AiofLimits()
    limits: KnownBoardRealCardHuRiverLimits = KnownBoardRealCardHuRiverLimits()
    max_horizon: int = DEFAULT_MAX_HORIZON
    expected_baseline_identity: str | None = None


@dataclass(frozen=True)
class RangeRemovalProvenance:
    """Per-seat raw mass and combo removal split by board and extra dead."""

    pre_removal_combo_count: int
    pre_removal_raw_mass: float
    board_collision_combo_count: int
    board_collision_raw_mass: float
    extra_dead_collision_combo_count: int
    extra_dead_collision_raw_mass: float
    surviving_combo_count: int
    surviving_raw_mass: float

    def to_dict(self) -> dict:
        return {
            "pre_removal_combo_count": self.pre_removal_combo_count,
            "pre_removal_raw_mass": self.pre_removal_raw_mass,
            "board_collision_combo_count": self.board_collision_combo_count,
            "board_collision_raw_mass": self.board_collision_raw_mass,
            "extra_dead_collision_combo_count": self.extra_dead_collision_combo_count,
            "extra_dead_collision_raw_mass": self.extra_dead_collision_raw_mass,
            "surviving_combo_count": self.surviving_combo_count,
            "surviving_raw_mass": self.surviving_raw_mass,
        }


@dataclass(frozen=True)
class JointSupportProvenance:
    """Complete ordered-pair projection and its single conditioning factor."""

    hero: RangeRemovalProvenance
    villain: RangeRemovalProvenance
    cross_product_pair_count: int
    cross_product_raw_mass: float
    private_overlap_excluded_pair_count: int
    private_overlap_excluded_raw_mass: float
    compatible_pair_count: int
    compatible_raw_joint_mass: float
    normalization_factor: float

    def to_dict(self) -> dict:
        return {
            "hero": self.hero.to_dict(),
            "villain": self.villain.to_dict(),
            "cross_product_pair_count": self.cross_product_pair_count,
            "cross_product_raw_mass": self.cross_product_raw_mass,
            "private_overlap_excluded_pair_count": (
                self.private_overlap_excluded_pair_count
            ),
            "private_overlap_excluded_raw_mass": (
                self.private_overlap_excluded_raw_mass
            ),
            "compatible_pair_count": self.compatible_pair_count,
            "compatible_raw_joint_mass": self.compatible_raw_joint_mass,
            "normalization_factor": self.normalization_factor,
        }


@dataclass(frozen=True)
class KnownBoardJointRow:
    """One exact combo pair, conditioned chance mass, and fixed showdown."""

    hero_combo: str
    villain_combo: str
    hero_bucket_id: str
    villain_bucket_id: str
    raw_joint_mass: float
    probability: float
    hero_rank: HandRank
    villain_rank: HandRank
    showdown_result: str

    def to_dict(self) -> dict:
        return {
            "hero_combo": self.hero_combo,
            "villain_combo": self.villain_combo,
            "hero_bucket_id": self.hero_bucket_id,
            "villain_bucket_id": self.villain_bucket_id,
            "raw_joint_mass": self.raw_joint_mass,
            "probability": self.probability,
            "hero_rank": _hand_rank_to_dict(self.hero_rank),
            "villain_rank": _hand_rank_to_dict(self.villain_rank),
            "showdown_result": self.showdown_result,
        }


@dataclass(frozen=True)
class KnownBoardRiverWorkloadProjection:
    """All integer workloads checked before joint/tree/candidate allocation."""

    joint_matchups: int
    fixed_board_evaluations: int
    tree_nodes: int
    hero_buckets: int
    villain_buckets: int
    hero_information_sets: int
    villain_information_sets: int
    feasible_single_shift_count: int
    feasible_shift_counts_by_info_set: tuple[tuple[str, int], ...]
    candidate_count: int
    candidate_probability_cells: int
    fixed_profile_node_visits: int
    response_node_visits: int
    timing_rows: int

    def to_dict(self) -> dict:
        return {
            "joint_matchups": self.joint_matchups,
            "fixed_board_evaluations": self.fixed_board_evaluations,
            "tree_nodes": self.tree_nodes,
            "hero_buckets": self.hero_buckets,
            "villain_buckets": self.villain_buckets,
            "hero_information_sets": self.hero_information_sets,
            "villain_information_sets": self.villain_information_sets,
            "feasible_single_shift_count": self.feasible_single_shift_count,
            "feasible_shift_counts_by_info_set": [
                {"info_set": info_set, "count": count}
                for info_set, count in self.feasible_shift_counts_by_info_set
            ],
            "candidate_count": self.candidate_count,
            "candidate_probability_cells": self.candidate_probability_cells,
            "fixed_profile_node_visits": self.fixed_profile_node_visits,
            "response_node_visits": self.response_node_visits,
            "timing_rows": self.timing_rows,
        }


@dataclass(frozen=True)
class KnownBoardRiverCandidateCoverage:
    """Exact unfiltered generated/kept candidate coverage."""

    generated_candidate_ids: tuple[str, ...]
    kept_candidate_ids: tuple[str, ...]
    shift_amounts: tuple[float, ...]
    max_simultaneous_info_sets: int
    filtering_applied: bool = False

    def to_dict(self) -> dict:
        return {
            "generated_candidate_count": len(self.generated_candidate_ids),
            "generated_candidate_ids": list(self.generated_candidate_ids),
            "kept_candidate_count": len(self.kept_candidate_ids),
            "kept_candidate_ids": list(self.kept_candidate_ids),
            "shift_amounts": list(self.shift_amounts),
            "max_simultaneous_info_sets": self.max_simultaneous_info_sets,
            "filtering_applied": self.filtering_applied,
        }


@dataclass(frozen=True)
class KnownBoardRiverBaselineRecord:
    """The selected complete fixed Villain baseline and native provenance."""

    source: str
    hero_profile: RiverActionProfile
    villain_profile: RiverActionProfile
    villain_strategy: VillainStrategy
    fixed_profile_value: FixedProfileValue
    auto_best_response: BestResponseResult | None
    response_identity: str | None

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "hero_profile": self.hero_profile.to_dict(),
            "villain_profile": self.villain_profile.to_dict(),
            "fixed_profile_value": self.fixed_profile_value.to_dict(),
            "auto_best_response": (
                None
                if self.auto_best_response is None
                else _best_response_to_dict(self.auto_best_response)
            ),
            "response_identity": self.response_identity,
        }


@dataclass(frozen=True)
class KnownBoardRiverCandidateRecord:
    """One native comparison with identity-bound shift/response provenance."""

    candidate_identity: str
    response_identity: str
    shifts: tuple[ShiftComponent, ...]
    comparison: CandidateComparison

    def to_dict(self) -> dict:
        return {
            "candidate_identity": self.candidate_identity,
            "response_identity": self.response_identity,
            "shifts": [shift.to_dict() for shift in self.shifts],
            "l1_distance": self.comparison.candidate.l1_distance,
            "fixed_profile_value": self.comparison.fixed_profile_value.to_dict(),
            "values": {
                "a": self.comparison.fixed_profile_value.hero_ev,
                "l_worst": self.comparison.best_response.ev_h_worst,
                "l_best": self.comparison.best_response.ev_h_best,
                "post_response_hero_ev_worst_diff": (
                    self.comparison.post_response_hero_ev_worst_diff
                ),
            },
            "best_response": _best_response_to_dict(
                self.comparison.best_response
            ),
        }


@dataclass(frozen=True)
class KnownBoardRealCardHuRiverPayload:
    """Complete successful payload retaining native tree/comparison objects."""

    board: tuple[str, ...]
    dead_cards: tuple[str, ...]
    board_identity: str
    prepared_joint_identity: str
    profile_mapping_identity: str
    tree_identity: str
    baseline_identity: str
    analysis_identity: str
    prepared_ranges: PreparedRanges
    provenance: JointSupportProvenance
    hero_mapping: ComboBucketMap
    villain_mapping: ComboBucketMap
    joint_rows: tuple[KnownBoardJointRow, ...]
    tree: GameTree
    baseline: KnownBoardRiverBaselineRecord
    workload: KnownBoardRiverWorkloadProjection
    coverage: KnownBoardRiverCandidateCoverage
    candidate_records: tuple[KnownBoardRiverCandidateRecord, ...]
    comparison_report: CandidateComparisonReport
    automatic_selection: AutomaticCommitmentSelectionReport
    request: KnownBoardRealCardHuRiverRequest

    def to_dict(self) -> dict:
        payload = {
            "contract_version": KNOWN_BOARD_REAL_CARD_HU_RIVER_CONTRACT_VERSION,
            "claim_scope": KNOWN_BOARD_RIVER_CLAIM_SCOPE,
            "position": KNOWN_BOARD_RIVER_POSITION,
            "accounting": KNOWN_BOARD_RIVER_ACCOUNTING,
            "unit": KNOWN_BOARD_RIVER_UNIT,
            "response_method": KNOWN_BOARD_RIVER_RESPONSE_METHOD,
            "evaluator": EVALUATOR_ID,
            "board": list(self.board),
            "extra_dead_cards": list(self.dead_cards),
            "identities": {
                "board": self.board_identity,
                "prepared_joint": self.prepared_joint_identity,
                "profile_mapping": self.profile_mapping_identity,
                "tree": self.tree_identity,
                "baseline": self.baseline_identity,
                "analysis": self.analysis_identity,
            },
            "joint_provenance": self.provenance.to_dict(),
            "mappings": {
                "hero": self.hero_mapping.to_dict(),
                "villain": self.villain_mapping.to_dict(),
            },
            "joint_rows": [row.to_dict() for row in self.joint_rows],
            "tree_summary": {
                "node_count": sum(1 for _ in iter_nodes(self.tree.root)),
                "hero_information_sets": sorted(collect_hero_info_sets(self.tree)),
                "villain_information_sets": sorted(
                    collect_villain_info_sets(self.tree)
                ),
                "terminal_lines_per_joint_row": 7,
            },
            "baseline": self.baseline.to_dict(),
            "b": self.comparison_report.baseline_value.hero_ev,
            "candidate_count_projection": self.workload.candidate_count,
            "candidate_count": len(self.candidate_records),
            "workload_projection": self.workload.to_dict(),
            "coverage": self.coverage.to_dict(),
            "caps": {
                "aiof": _dataclass_plain_dict(self.request.aiof_limits),
                "adapter": self.request.limits.to_dict(),
                "max_horizon": self.request.max_horizon,
            },
            "repeated_configuration": {
                "horizon": self.request.horizon,
                "discount": self.request.discount,
                "tolerance": self.request.tolerance,
                "minimum_total_uplift": self.request.minimum_total_uplift,
            },
            "candidates": [row.to_dict() for row in self.candidate_records],
            "automatic_selection": self.automatic_selection.to_dict(),
        }
        _assert_json_safe(payload)
        return payload


@dataclass(frozen=True)
class KnownBoardRealCardHuRiverResult:
    """AiofStatus no-partial wrapper with mutually exclusive payload/error."""

    status: AiofStatus
    payload: KnownBoardRealCardHuRiverPayload | None
    error_message: str | None

    def __post_init__(self) -> None:
        success = self.status is AiofStatus.SUCCESS
        if success != (self.payload is not None):
            raise ValueError("SUCCESS must have one payload and failures must not")
        if success != (self.error_message is None):
            raise ValueError("SUCCESS must not have an error and failures must")

    def to_dict(self) -> dict:
        value = {
            "status": self.status.value,
            "payload": None if self.payload is None else self.payload.to_dict(),
            "error": self.error_message,
        }
        _assert_json_safe(value)
        return value


def _require_finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, f"{name} must be a finite number"
        )
    number = float(value)
    if not math.isfinite(number):
        raise AiofContractError(AiofStatus.INVALID_INPUT, f"{name} must be finite")
    return number


def _finite_fsum(values, name: str) -> float:
    try:
        total = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise AiofContractError(
            AiofStatus.NUMERIC_FAILURE, f"{name} aggregation failed"
        ) from exc
    if not math.isfinite(total):
        raise AiofContractError(AiofStatus.NUMERIC_FAILURE, f"{name} is non-finite")
    return total


def _canonical_identity_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AiofContractError(
                AiofStatus.NUMERIC_FAILURE, "identity contains non-finite float"
            )
        return {"float_hex": value.hex()}
    if isinstance(value, HandRank):
        return _hand_rank_to_dict(value)
    if is_dataclass(value):
        return {
            field.name: _canonical_identity_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {
            str(key): _canonical_identity_value(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_identity_value(item) for item in value]
    return value


def _sha256_identity(payload: object) -> str:
    encoded = json.dumps(
        _canonical_identity_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _assert_json_safe(value: object) -> None:
    try:
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise AiofContractError(
            AiofStatus.NUMERIC_FAILURE, "public projection is not strict JSON"
        ) from exc


def _dataclass_plain_dict(value: object) -> dict:
    if not is_dataclass(value):
        raise TypeError("value must be a dataclass")
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _hand_rank_to_dict(rank: HandRank) -> dict:
    return {
        "category": rank.category.name,
        "category_value": int(rank.category),
        "tiebreak": list(rank.tiebreak),
    }


def _strategy_to_profile(
    strategy: VillainStrategy,
    buckets: tuple[str, ...],
) -> RiverActionProfile:
    rows = []
    for bucket in buckets:
        for decision in VILLAIN_DECISIONS:
            info_set = f"{DECISION_PREFIX[decision]}::{bucket}"
            dist = strategy.probabilities[info_set]
            rows.append(
                RiverProfileRow(
                    bucket,
                    decision,
                    tuple(
                        ActionProbability(action, dist[action])
                        for action in VILLAIN_DECISIONS[decision]
                    ),
                )
            )
    return RiverActionProfile(tuple(rows))


def _best_response_to_dict(response: BestResponseResult) -> dict:
    exact_count = response.num_best_response_strategies
    materialized_count = len(response.best_response_strategies)
    complete = exact_count is not None and exact_count == materialized_count
    return {
        "method": KNOWN_BOARD_RIVER_RESPONSE_METHOD,
        "villain_max_ev": response.villain_max_ev,
        "ev_h_worst": response.ev_h_worst,
        "ev_h_best": response.ev_h_best,
        "expected_house_rake_worst": response.expected_house_rake_worst,
        "expected_house_rake_best": response.expected_house_rake_best,
        "num_villain_pure_strategies": response.num_villain_pure_strategies,
        "exact_best_response_count": exact_count,
        "materialized_best_response_count": materialized_count,
        "correspondence_materialization_complete": complete,
        "best_response_strategies": [
            {key: strategy[key] for key in sorted(strategy)}
            for strategy in response.best_response_strategies
        ],
        "best_response_action_variation": {
            key: list(response.best_response_action_variation[key])
            for key in sorted(response.best_response_action_variation)
        },
        "best_response_action_sets": (
            None
            if response.best_response_action_sets is None
            else {
                key: list(response.best_response_action_sets[key])
                for key in sorted(response.best_response_action_sets)
            }
        ),
        "off_path_info_sets": list(response.off_path_info_sets),
    }


def _validate_positive_cap(value: object, name: str, ceiling: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, f"{name} must be a positive integer"
        )
    if value > ceiling:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            f"{name}={value} exceeds hard ceiling {ceiling}",
        )
    return value


def _validate_limits(value: object) -> KnownBoardRealCardHuRiverLimits:
    if not isinstance(value, KnownBoardRealCardHuRiverLimits):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            "limits must be KnownBoardRealCardHuRiverLimits",
        )
    ceilings = {
        "max_joint_matchups": MAX_JOINT_MATCHUPS,
        "max_fixed_board_evaluations": MAX_FIXED_BOARD_EVALUATIONS,
        "max_tree_nodes": MAX_TREE_NODES,
        "max_buckets_per_seat": MAX_BUCKETS_PER_SEAT,
        "max_hero_info_sets": MAX_HERO_INFO_SETS,
        "max_villain_info_sets": MAX_VILLAIN_INFO_SETS,
        "max_candidates": MAX_CANDIDATES,
        "max_candidate_probability_cells": MAX_CANDIDATE_PROBABILITY_CELLS,
        "max_fixed_profile_node_visits": MAX_FIXED_PROFILE_NODE_VISITS,
        "max_response_node_visits": MAX_RESPONSE_NODE_VISITS,
        "max_timing_rows": MAX_TIMING_ROWS,
        "max_br_list_materialization": MAX_BR_LIST_MATERIALIZATION,
    }
    for name, ceiling in ceilings.items():
        _validate_positive_cap(getattr(value, name), name, ceiling)
    return value


def _canonical_board_and_dead(
    board: object, dead_cards: object, aiof_limits: AiofLimits
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if not isinstance(board, tuple) or len(board) != 5:
        raise AiofContractError(
            AiofStatus.INVALID_CARD_INPUT, "board must contain exactly five cards"
        )
    if not isinstance(dead_cards, tuple):
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
            AiofStatus.INVALID_CARD_INPUT, "extra dead cards must be distinct"
        )
    if set(board_ids) & set(dead_ids):
        raise AiofContractError(
            AiofStatus.INVALID_CARD_INPUT, "board and extra dead cards collide"
        )
    if len(board_ids) + len(dead_ids) > aiof_limits.max_dead_cards:
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED, "combined board/dead-card cap exceeded"
        )
    ordered_board = tuple(card_from_id(value) for value in sorted(board_ids))
    ordered_dead = tuple(card_from_id(value) for value in sorted(dead_ids))
    return ordered_board, ordered_dead, tuple(
        card_from_id(value) for value in sorted(board_ids + dead_ids)
    )


def _canonical_shift_amounts(value: object) -> tuple[float, ...]:
    if not isinstance(value, tuple):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, "shift_amounts must be a tuple"
        )
    shifts = tuple(_require_finite_number(item, "shift_amount") for item in value)
    if any(item <= 0.0 or item > 1.0 for item in shifts):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            "shift amounts must be strictly positive and at most one",
        )
    if len(set(shifts)) != len(shifts):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, "shift amounts must be unique"
        )
    return tuple(sorted(shifts))


def _validate_request(
    value: object,
) -> tuple[
    KnownBoardRealCardHuRiverRequest,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[float, ...],
    AutomaticCommitmentSelectionConfig,
]:
    if not isinstance(value, KnownBoardRealCardHuRiverRequest):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            "request must be KnownBoardRealCardHuRiverRequest",
        )
    request = value
    if not isinstance(request.hero_range, RangeSpec) or not isinstance(
        request.villain_range, RangeSpec
    ):
        raise AiofContractError(
            AiofStatus.INVALID_RANGE,
            "hero_range and villain_range must be RangeSpec",
        )
    if not isinstance(request.aiof_limits, AiofLimits):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, "aiof_limits must be AiofLimits"
        )
    limits = _validate_limits(request.limits)
    board, dead, unavailable = _canonical_board_and_dead(
        request.board, request.dead_cards, request.aiof_limits
    )
    if isinstance(request.max_simultaneous_info_sets, bool) or (
        request.max_simultaneous_info_sets not in (1, 2)
    ):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            "max_simultaneous_info_sets must be 1 or 2",
        )
    if request.expected_baseline_identity is not None and (
        not isinstance(request.expected_baseline_identity, str)
        or not request.expected_baseline_identity
    ):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            "expected_baseline_identity must be a non-empty string or None",
        )
    for name in (
        "initial_commitment_hero",
        "initial_commitment_villain",
        "rake_rate",
        "oop_bet_size",
        "ip_bet_after_check_size",
        "ip_raise_to_size",
    ):
        _require_finite_number(getattr(request, name), name)
    if request.rake_cap is not None:
        _require_finite_number(request.rake_cap, "rake_cap")
    if request.initial_commitment_hero < 0 or request.initial_commitment_villain < 0:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, "initial commitments must be non-negative"
        )
    if not 0.0 <= request.rake_rate <= 1.0:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, "rake_rate must be within [0, 1]"
        )
    if request.rake_cap is not None and request.rake_cap < 0:
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, "rake_cap must be non-negative"
        )
    if (
        request.oop_bet_size <= 0
        or request.ip_bet_after_check_size <= 0
        or request.ip_raise_to_size <= request.oop_bet_size
    ):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT,
            "bet sizes must be positive and ip_raise_to_size must exceed oop_bet_size",
        )
    shifts = _canonical_shift_amounts(request.shift_amounts)
    selector_configuration = AutomaticCommitmentSelectionConfig(
        minimum_total_uplift=request.minimum_total_uplift,
        max_candidates=limits.max_candidates,
        max_timing_rows=limits.max_timing_rows,
    )
    try:
        validate_automatic_commitment_selection_parameters(
            horizon=request.horizon,
            discount=request.discount,
            tolerance=request.tolerance,
            max_horizon=request.max_horizon,
            configuration=selector_configuration,
        )
    except ValueError as exc:
        raise AiofContractError(AiofStatus.INVALID_INPUT, str(exc)) from exc
    return request, board, dead, unavailable, shifts, selector_configuration


def _range_removal_provenance(
    initial: ExpandedRange,
    after_board: ExpandedRange,
    final: ExpandedRange,
) -> RangeRemovalProvenance:
    board_count = len(initial.combos) - len(after_board.combos)
    dead_count = len(after_board.combos) - len(final.combos)
    board_mass = initial.raw_mass_after_dead - after_board.raw_mass_after_dead
    dead_mass = after_board.raw_mass_after_dead - final.raw_mass_after_dead
    values = (
        initial.raw_mass_after_dead,
        board_mass,
        dead_mass,
        final.raw_mass_after_dead,
    )
    if not all(math.isfinite(item) for item in values):
        raise AiofContractError(
            AiofStatus.NUMERIC_FAILURE, "range removal mass is non-finite"
        )
    return RangeRemovalProvenance(
        pre_removal_combo_count=len(initial.combos),
        pre_removal_raw_mass=initial.raw_mass_after_dead,
        board_collision_combo_count=board_count,
        board_collision_raw_mass=board_mass,
        extra_dead_collision_combo_count=dead_count,
        extra_dead_collision_raw_mass=dead_mass,
        surviving_combo_count=len(final.combos),
        surviving_raw_mass=final.raw_mass_after_dead,
    )


@dataclass(frozen=True)
class _PreparedProjection:
    hero_initial: ExpandedRange
    hero_after_board: ExpandedRange
    hero_final: ExpandedRange
    villain_initial: ExpandedRange
    villain_after_board: ExpandedRange
    villain_final: ExpandedRange
    provenance: JointSupportProvenance


def _project_joint_support(
    request: KnownBoardRealCardHuRiverRequest,
    board: tuple[str, ...],
    unavailable: tuple[str, ...],
) -> _PreparedProjection:
    hero_initial = expand_range(request.hero_range, (), request.aiof_limits)
    villain_initial = expand_range(request.villain_range, (), request.aiof_limits)
    hero_after_board = expand_range(request.hero_range, board, request.aiof_limits)
    villain_after_board = expand_range(
        request.villain_range, board, request.aiof_limits
    )
    hero_final = expand_range(
        request.hero_range, unavailable, request.aiof_limits
    )
    villain_final = expand_range(
        request.villain_range, unavailable, request.aiof_limits
    )
    compatible_count = 0
    overlap_count = 0
    compatible_mass = 0.0
    overlap_mass = 0.0
    hero_compatible = {item.combo: 0.0 for item in hero_final.combos}
    villain_compatible = {item.combo: 0.0 for item in villain_final.combos}
    for hero in hero_final.combos:
        for villain in villain_final.combos:
            joint = hero.raw_mass * villain.raw_mass
            if not math.isfinite(joint) or joint <= 0.0:
                raise AiofContractError(
                    AiofStatus.NUMERIC_FAILURE, "invalid projected joint mass"
            )
            if set(hero.card_ids) & set(villain.card_ids):
                overlap_count += 1
                overlap_mass = _finite_fsum(
                    (overlap_mass, joint), "private-overlap joint mass"
                )
                continue
            compatible_count += 1
            if compatible_count > request.limits.max_joint_matchups:
                raise AiofContractError(
                    AiofStatus.CAP_EXCEEDED, "joint matchup cap exceeded"
                )
            compatible_mass = _finite_fsum(
                (compatible_mass, joint), "compatible joint mass"
            )
            hero_compatible[hero.combo] = _finite_fsum(
                (hero_compatible[hero.combo], joint),
                "Hero compatible marginal",
            )
            villain_compatible[villain.combo] = _finite_fsum(
                (villain_compatible[villain.combo], joint),
                "Villain compatible marginal",
            )
    if compatible_count == 0:
        raise AiofContractError(
            AiofStatus.EMPTY_COMPATIBLE_SUPPORT,
            "compatible joint support is empty",
        )
    if compatible_mass <= 0.0:
        raise AiofContractError(
            AiofStatus.NUMERIC_FAILURE, "compatible joint mass is not positive"
        )
    if any(value <= 0.0 for value in hero_compatible.values()) or any(
        value <= 0.0 for value in villain_compatible.values()
    ):
        raise AiofContractError(
            AiofStatus.ZERO_COMPATIBLE_MARGINAL,
            "a surviving combo has zero compatible marginal",
        )
    factor = 1.0 / compatible_mass
    if not math.isfinite(factor):
        raise AiofContractError(
            AiofStatus.NUMERIC_FAILURE, "normalization factor is non-finite"
        )
    cross_mass = hero_final.raw_mass_after_dead * villain_final.raw_mass_after_dead
    numeric_scale = max(1.0, abs(cross_mass), abs(compatible_mass), abs(overlap_mass))
    if abs((compatible_mass + overlap_mass) - cross_mass) > 1e-12 * numeric_scale:
        raise AiofContractError(
            AiofStatus.ACCOUNTING_MISMATCH,
            "compatible and private-overlap masses do not reconcile",
        )
    provenance = JointSupportProvenance(
        hero=_range_removal_provenance(
            hero_initial, hero_after_board, hero_final
        ),
        villain=_range_removal_provenance(
            villain_initial, villain_after_board, villain_final
        ),
        cross_product_pair_count=len(hero_final.combos) * len(villain_final.combos),
        cross_product_raw_mass=cross_mass,
        private_overlap_excluded_pair_count=overlap_count,
        private_overlap_excluded_raw_mass=overlap_mass,
        compatible_pair_count=compatible_count,
        compatible_raw_joint_mass=compatible_mass,
        normalization_factor=factor,
    )
    return _PreparedProjection(
        hero_initial,
        hero_after_board,
        hero_final,
        villain_initial,
        villain_after_board,
        villain_final,
        provenance,
    )


def _validate_bucket_id(value: object) -> str:
    if not isinstance(value, str) or _BUCKET_PATTERN.fullmatch(value) is None:
        raise AiofContractError(
            AiofStatus.INVALID_STRATEGY, f"invalid bucket id {value!r}"
        )
    return value


def _canonical_mapping(
    value: object,
    combos: tuple[str, ...],
    seat: str,
    max_buckets: int,
) -> ComboBucketMap:
    if value is None:
        if len(combos) > max_buckets:
            raise AiofContractError(
                AiofStatus.CAP_EXCEEDED, f"{seat} bucket cap exceeded"
            )
        return ComboBucketMap(
            bucket_ids=combos,
            assignments=tuple(ComboBucketAssignment(combo, combo) for combo in combos),
        )
    if not isinstance(value, ComboBucketMap):
        raise AiofContractError(
            AiofStatus.INVALID_STRATEGY,
            f"{seat}_combo_to_bucket must be ComboBucketMap or None",
        )
    if not isinstance(value.bucket_ids, tuple) or not isinstance(
        value.assignments, tuple
    ):
        raise AiofContractError(
            AiofStatus.INVALID_STRATEGY,
            f"{seat} mapping fields must be tuples",
        )
    buckets = tuple(_validate_bucket_id(item) for item in value.bucket_ids)
    if len(buckets) != len(set(buckets)):
        raise AiofContractError(
            AiofStatus.INVALID_STRATEGY, f"duplicate {seat} bucket id"
        )
    if len(buckets) > max_buckets:
        raise AiofContractError(
            AiofStatus.CAP_EXCEEDED, f"{seat} bucket cap exceeded"
        )
    assignments: dict[str, str] = {}
    for row in value.assignments:
        if not isinstance(row, ComboBucketAssignment):
            raise AiofContractError(
                AiofStatus.INVALID_STRATEGY, f"invalid {seat} mapping row"
            )
        try:
            combo = canonicalize_exact_combo(row.combo)
        except AiofContractError as exc:
            raise AiofContractError(AiofStatus.INVALID_STRATEGY, str(exc)) from exc
        bucket = _validate_bucket_id(row.bucket_id)
        if combo in assignments:
            raise AiofContractError(
                AiofStatus.INVALID_STRATEGY,
                f"duplicate {seat} mapping combo {combo}",
            )
        assignments[combo] = bucket
    if set(assignments) != set(combos):
        missing = sorted(set(combos) - set(assignments))
        extra = sorted(set(assignments) - set(combos))
        raise AiofContractError(
            AiofStatus.INVALID_STRATEGY,
            f"{seat} mapping support mismatch; missing={missing}, extra={extra}",
        )
    if set(assignments.values()) - set(buckets):
        raise AiofContractError(
            AiofStatus.INVALID_STRATEGY,
            f"{seat} mapping references an unknown bucket",
        )
    empty = sorted(set(buckets) - set(assignments.values()))
    if empty:
        raise AiofContractError(
            AiofStatus.INVALID_STRATEGY,
            f"{seat} mapping declares empty buckets {empty}",
        )
    return ComboBucketMap(
        bucket_ids=tuple(sorted(buckets)),
        assignments=tuple(
            ComboBucketAssignment(combo, assignments[combo])
            for combo in sorted(assignments)
        ),
    )


def _mapping_dict(value: ComboBucketMap) -> dict[str, str]:
    return {row.combo: row.bucket_id for row in value.assignments}


def _canonical_profile(
    value: object,
    buckets: tuple[str, ...],
    seat: str,
    tolerance: float,
) -> tuple[RiverActionProfile, dict[str, dict[str, float]]]:
    if not isinstance(value, RiverActionProfile) or not isinstance(value.rows, tuple):
        raise AiofContractError(
            AiofStatus.INVALID_STRATEGY,
            f"baseline_{seat}_profile must be RiverActionProfile",
        )
    decisions = HERO_DECISIONS if seat == "hero" else VILLAIN_DECISIONS
    canonical: dict[tuple[str, str], tuple[ActionProbability, ...]] = {}
    for row in value.rows:
        if not isinstance(row, RiverProfileRow) or not isinstance(row.actions, tuple):
            raise AiofContractError(
                AiofStatus.INVALID_STRATEGY, f"invalid {seat} profile row"
            )
        bucket = _validate_bucket_id(row.bucket_id)
        if row.decision not in decisions:
            raise AiofContractError(
                AiofStatus.INVALID_STRATEGY,
                f"unknown {seat} decision {row.decision!r}",
            )
        key = (bucket, row.decision)
        if key in canonical:
            raise AiofContractError(
                AiofStatus.INVALID_STRATEGY,
                f"duplicate {seat} profile row {key}",
            )
        action_values: dict[str, float] = {}
        for item in row.actions:
            if not isinstance(item, ActionProbability):
                raise AiofContractError(
                    AiofStatus.INVALID_STRATEGY,
                    f"invalid {seat} action probability row",
                )
            if not isinstance(item.action, str) or item.action in action_values:
                raise AiofContractError(
                    AiofStatus.INVALID_STRATEGY,
                    f"duplicate or invalid {seat} action",
                )
            try:
                probability = _require_finite_number(
                    item.probability, f"{seat} profile probability"
                )
            except AiofContractError as exc:
                raise AiofContractError(AiofStatus.INVALID_STRATEGY, str(exc)) from exc
            if probability < 0.0:
                raise AiofContractError(
                    AiofStatus.INVALID_STRATEGY,
                    f"{seat} profile probability must be non-negative",
                )
            action_values[item.action] = probability
        legal = decisions[row.decision]
        if set(action_values) != set(legal):
            raise AiofContractError(
                AiofStatus.INVALID_STRATEGY,
                f"{seat} profile action keys must be exactly {list(legal)}",
            )
        total = _finite_fsum(
            (action_values[action] for action in legal),
            f"{seat} profile probability",
        )
        if abs(total - 1.0) > tolerance:
            raise AiofContractError(
                AiofStatus.INVALID_STRATEGY,
                f"{seat} profile probabilities must sum to one",
            )
        canonical[key] = tuple(
            ActionProbability(action, action_values[action]) for action in legal
        )
    expected = {(bucket, decision) for bucket in buckets for decision in decisions}
    if set(canonical) != expected:
        missing = sorted(expected - set(canonical))
        extra = sorted(set(canonical) - expected)
        raise AiofContractError(
            AiofStatus.INVALID_STRATEGY,
            f"{seat} profile coverage mismatch; missing={missing}, extra={extra}",
        )
    rows = tuple(
        RiverProfileRow(bucket, decision, canonical[(bucket, decision)])
        for bucket in sorted(buckets)
        for decision in decisions
    )
    probabilities = {
        f"{DECISION_PREFIX[decision]}::{bucket}": {
            item.action: item.probability
            for item in canonical[(bucket, decision)]
        }
        for bucket in sorted(buckets)
        for decision in decisions
    }
    return RiverActionProfile(rows), probabilities


def _feasible_shift_options(
    hero_probabilities: dict[str, dict[str, float]],
    shift_amounts: tuple[float, ...],
) -> dict[str, tuple[ShiftComponent, ...]]:
    options: dict[str, tuple[ShiftComponent, ...]] = {}
    for info_set in sorted(hero_probabilities):
        dist = hero_probabilities[info_set]
        rows = []
        for source in dist:
            for target in dist:
                if source == target:
                    continue
                for shift in shift_amounts:
                    if dist[source] >= shift:
                        rows.append(
                            ShiftComponent(info_set, source, target, shift)
                        )
        options[info_set] = tuple(
            sorted(
                rows,
                key=lambda item: (
                    item.info_set,
                    item.source_action,
                    item.target_action,
                    item.shift_amount.hex(),
                ),
            )
        )
    return options


def _project_feasible_shift_counts(
    hero_probabilities: dict[str, dict[str, float]],
    shift_amounts: tuple[float, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for info_set in sorted(hero_probabilities):
        dist = hero_probabilities[info_set]
        count = 0
        for source in dist:
            for target in dist:
                if source == target:
                    continue
                count += sum(dist[source] >= shift for shift in shift_amounts)
        counts[info_set] = count
    return counts


def _project_candidate_count(
    option_counts: dict[str, int],
    max_simultaneous_info_sets: int,
) -> tuple[int, int]:
    counts = [option_counts[key] for key in sorted(option_counts)]
    singles = sum(counts)
    if max_simultaneous_info_sets == 1:
        return singles, singles
    pairs = sum(
        counts[first] * counts[second]
        for first in range(len(counts))
        for second in range(first + 1, len(counts))
    )
    return singles, singles + pairs


def _project_workload(
    request: KnownBoardRealCardHuRiverRequest,
    provenance: JointSupportProvenance,
    hero_mapping: ComboBucketMap,
    villain_mapping: ComboBucketMap,
    option_counts: dict[str, int],
    auto_baseline: bool,
) -> KnownBoardRiverWorkloadProjection:
    limits = request.limits
    joint = provenance.compatible_pair_count
    evaluations = joint
    nodes = 12 * joint + 1
    hero_buckets = len(hero_mapping.bucket_ids)
    villain_buckets = len(villain_mapping.bucket_ids)
    hero_info_sets = 2 * hero_buckets
    villain_info_sets = 3 * villain_buckets
    singles, candidate_count = _project_candidate_count(
        option_counts, request.max_simultaneous_info_sets
    )
    candidate_cells = candidate_count * 5 * hero_buckets
    fixed_visits = (candidate_count + 1) * nodes
    response_visits = (candidate_count + int(auto_baseline)) * nodes
    timing_rows = candidate_count * (request.horizon + 1)
    checks = (
        (joint, limits.max_joint_matchups, "joint matchup"),
        (evaluations, limits.max_fixed_board_evaluations, "fixed-board evaluation"),
        (nodes, limits.max_tree_nodes, "tree node"),
        (hero_buckets, limits.max_buckets_per_seat, "Hero bucket"),
        (villain_buckets, limits.max_buckets_per_seat, "Villain bucket"),
        (hero_info_sets, limits.max_hero_info_sets, "Hero information-set"),
        (
            villain_info_sets,
            limits.max_villain_info_sets,
            "Villain information-set",
        ),
        (candidate_count, limits.max_candidates, "candidate"),
        (
            candidate_cells,
            limits.max_candidate_probability_cells,
            "candidate probability-cell",
        ),
        (
            fixed_visits,
            limits.max_fixed_profile_node_visits,
            "fixed-profile node-visit",
        ),
        (
            response_visits,
            limits.max_response_node_visits,
            "response node-visit",
        ),
        (timing_rows, limits.max_timing_rows, "timing-row"),
    )
    for projected, cap, label in checks:
        if projected > cap:
            raise AiofContractError(
                AiofStatus.CAP_EXCEEDED,
                f"projected {label} workload {projected} exceeds cap {cap}",
            )
    return KnownBoardRiverWorkloadProjection(
        joint_matchups=joint,
        fixed_board_evaluations=evaluations,
        tree_nodes=nodes,
        hero_buckets=hero_buckets,
        villain_buckets=villain_buckets,
        hero_information_sets=hero_info_sets,
        villain_information_sets=villain_info_sets,
        feasible_single_shift_count=singles,
        feasible_shift_counts_by_info_set=tuple(
            (key, option_counts[key]) for key in sorted(option_counts)
        ),
        candidate_count=candidate_count,
        candidate_probability_cells=candidate_cells,
        fixed_profile_node_visits=fixed_visits,
        response_node_visits=response_visits,
        timing_rows=timing_rows,
    )


def _materialize_joint_rows(
    projection: _PreparedProjection,
    board: tuple[str, ...],
    hero_mapping: ComboBucketMap,
    villain_mapping: ComboBucketMap,
) -> tuple[KnownBoardJointRow, ...]:
    hero_buckets = _mapping_dict(hero_mapping)
    villain_buckets = _mapping_dict(villain_mapping)
    rows = []
    for hero in projection.hero_final.combos:
        hero_cards = (
            card_from_id(hero.card_ids[0]),
            card_from_id(hero.card_ids[1]),
        )
        for villain in projection.villain_final.combos:
            if set(hero.card_ids) & set(villain.card_ids):
                continue
            villain_cards = (
                card_from_id(villain.card_ids[0]),
                card_from_id(villain.card_ids[1]),
            )
            raw_mass = hero.raw_mass * villain.raw_mass
            probability = raw_mass * projection.provenance.normalization_factor
            if not math.isfinite(probability) or probability <= 0.0:
                raise AiofContractError(
                    AiofStatus.NUMERIC_FAILURE,
                    "conditioned joint probability is invalid",
                )
            hero_rank = evaluate_seven_card_hand(hero_cards + board)
            villain_rank = evaluate_seven_card_hand(villain_cards + board)
            if hero_rank > villain_rank:
                result = HERO
            elif hero_rank < villain_rank:
                result = VILLAIN
            else:
                result = CHOP
            rows.append(
                KnownBoardJointRow(
                    hero.combo,
                    villain.combo,
                    hero_buckets[hero.combo],
                    villain_buckets[villain.combo],
                    raw_mass,
                    probability,
                    hero_rank,
                    villain_rank,
                    result,
                )
            )
    if len(rows) != projection.provenance.compatible_pair_count:
        raise AiofContractError(
            AiofStatus.ACCOUNTING_MISMATCH,
            "joint row count differs from the pre-allocation projection",
        )
    probability_sum = _finite_fsum(
        (row.probability for row in rows), "conditioned chance probability"
    )
    if abs(probability_sum - 1.0) > 1e-12:
        raise AiofContractError(
            AiofStatus.ACCOUNTING_MISMATCH,
            "conditioned chance probabilities do not sum to one",
        )
    return tuple(rows)


def _build_tree(
    request: KnownBoardRealCardHuRiverRequest,
    rows: tuple[KnownBoardJointRow, ...],
) -> GameTree:
    children = []
    for index, row in enumerate(rows):
        suffix = f"::{index:04d}__{row.hero_combo}__{row.villain_combo}"
        hero_initial = request.initial_commitment_hero
        villain_initial = request.initial_commitment_villain

        def showdown(node_id: str, hero_invested: float, villain_invested: float):
            return make_showdown_terminal(
                node_id,
                hero_invested + villain_invested,
                hero_invested,
                villain_invested,
                row.showdown_result,
                request.rake_rate,
                request.rake_cap,
                tolerance=max(request.tolerance, 1e-12),
            )

        line_check_check = showdown(
            f"T_check_check{suffix}", hero_initial, villain_initial
        )
        line_check_bet_call = showdown(
            f"T_check_bet_call{suffix}",
            hero_initial + request.ip_bet_after_check_size,
            villain_initial + request.ip_bet_after_check_size,
        )
        line_check_bet_fold = make_fold_terminal(
            f"T_check_bet_fold{suffix}", HERO, villain_initial
        )
        oop_vs_ip_bet = VillainNode(
            node_id=f"oop_vs_ip_bet{suffix}",
            info_set=f"{OOP_VS_IP_BET}::{row.villain_bucket_id}",
            actions=(("call", line_check_bet_call), ("fold", line_check_bet_fold)),
        )
        ip_after_check = HeroNode(
            node_id=f"ip_after_check{suffix}",
            info_set=f"{HERO_AFTER_OOP_CHECK}::{row.hero_bucket_id}",
            actions=(("check", line_check_check), ("bet", oop_vs_ip_bet)),
        )
        line_bet_call = showdown(
            f"T_bet_call{suffix}",
            hero_initial + request.oop_bet_size,
            villain_initial + request.oop_bet_size,
        )
        line_bet_fold = make_fold_terminal(
            f"T_bet_fold{suffix}", VILLAIN, hero_initial
        )
        line_bet_raise_call = showdown(
            f"T_bet_raise_call{suffix}",
            hero_initial + request.ip_raise_to_size,
            villain_initial + request.ip_raise_to_size,
        )
        line_bet_raise_fold = make_fold_terminal(
            f"T_bet_raise_fold{suffix}",
            HERO,
            villain_initial + request.oop_bet_size,
        )
        oop_vs_ip_raise = VillainNode(
            node_id=f"oop_vs_ip_raise{suffix}",
            info_set=f"{OOP_VS_IP_RAISE}::{row.villain_bucket_id}",
            actions=(("call", line_bet_raise_call), ("fold", line_bet_raise_fold)),
        )
        ip_vs_oop_bet = HeroNode(
            node_id=f"ip_vs_oop_bet{suffix}",
            info_set=f"{HERO_VS_OOP_BET}::{row.hero_bucket_id}",
            actions=(
                ("call", line_bet_call),
                ("fold", line_bet_fold),
                ("raise", oop_vs_ip_raise),
            ),
        )
        oop_first = VillainNode(
            node_id=f"oop_first{suffix}",
            info_set=f"{OOP_FIRST}::{row.villain_bucket_id}",
            actions=(("check", ip_after_check), ("bet", ip_vs_oop_bet)),
        )
        children.append((row.probability, oop_first))
    tree = GameTree(ChanceNode("known_board_joint_matchup", tuple(children)))
    try:
        validate_tree(tree, tolerance=max(request.tolerance, 1e-12))
    except ValueError as exc:
        raise AiofContractError(AiofStatus.ACCOUNTING_MISMATCH, str(exc)) from exc
    return tree


def _candidate_identity(
    baseline_identity: str, shifts: tuple[ShiftComponent, ...]
) -> str:
    return _sha256_identity(
        {
            "algorithm": CANDIDATE_IDENTITY_ALGORITHM,
            "baseline_identity": baseline_identity,
            "shifts": shifts,
        }
    )


def _copy_probabilities(
    probabilities: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {key: dict(value) for key, value in probabilities.items()}


def _materialize_candidates(
    baseline: HeroStrategy,
    baseline_identity: str,
    options: dict[str, tuple[ShiftComponent, ...]],
    max_simultaneous_info_sets: int,
) -> tuple[HeroStrategyCandidate, ...]:
    groups: list[tuple[ShiftComponent, ...]] = [
        (component,)
        for info_set in sorted(options)
        for component in options[info_set]
    ]
    if max_simultaneous_info_sets == 2:
        info_sets = sorted(options)
        for first_index, first_info_set in enumerate(info_sets):
            for second_info_set in info_sets[first_index + 1 :]:
                for first in options[first_info_set]:
                    for second in options[second_info_set]:
                        groups.append((first, second))
    groups.sort(
        key=lambda group: tuple(
            (
                item.info_set,
                item.source_action,
                item.target_action,
                item.shift_amount.hex(),
            )
            for item in group
        )
    )
    candidates = []
    for shifts in groups:
        probabilities = _copy_probabilities(baseline.probabilities)
        for shift in shifts:
            dist = probabilities[shift.info_set]
            source = dist[shift.source_action]
            target = dist[shift.target_action]
            new_source = source - shift.shift_amount
            new_target = target + shift.shift_amount
            if (
                not math.isfinite(new_source)
                or not math.isfinite(new_target)
                or new_source < 0.0
                or new_target > 1.0
            ):
                raise AiofContractError(
                    AiofStatus.NUMERIC_FAILURE,
                    "candidate probability derivation is invalid",
                )
            dist[shift.source_action] = new_source
            dist[shift.target_action] = new_target
        strategy = HeroStrategy(probabilities)
        l1_distance = 2.0 * _finite_fsum(
            (item.shift_amount for item in shifts), "candidate L1 distance"
        )
        candidate_id = _candidate_identity(baseline_identity, shifts)
        if len(shifts) == 1:
            component = shifts[0]
            candidates.append(
                HeroStrategyCandidate(
                    candidate_id,
                    component.info_set,
                    component.source_action,
                    component.target_action,
                    component.shift_amount,
                    strategy,
                    l1_distance,
                    shifts,
                )
            )
        else:
            candidates.append(
                HeroStrategyCandidate(
                    candidate_id,
                    None,
                    None,
                    None,
                    None,
                    strategy,
                    l1_distance,
                    shifts,
                )
            )
    identifiers = [item.candidate_id for item in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, "candidate identity collision detected"
        )
    return tuple(candidates)


def _bounded_aiof_limits(
    request: KnownBoardRealCardHuRiverRequest,
) -> AiofLimits:
    values = _dataclass_plain_dict(request.aiof_limits)
    values["max_compatible_combo_pairs"] = min(
        request.aiof_limits.max_compatible_combo_pairs,
        request.limits.max_joint_matchups,
    )
    return AiofLimits(**values)


def _board_identity(
    board: tuple[str, ...], dead: tuple[str, ...]
) -> str:
    return _sha256_identity(
        {
            "algorithm": BOARD_IDENTITY_ALGORITHM,
            "contract_version": KNOWN_BOARD_REAL_CARD_HU_RIVER_CONTRACT_VERSION,
            "board": board,
            "extra_dead_cards": dead,
        }
    )


def _prepared_joint_identity(
    board_identity: str,
    projection: _PreparedProjection,
    prepared: PreparedRanges,
    rows: tuple[KnownBoardJointRow, ...],
) -> str:
    return _sha256_identity(
        {
            "algorithm": JOINT_IDENTITY_ALGORITHM,
            "board_identity": board_identity,
            "hero_range_identity": projection.hero_final.content_identity,
            "villain_range_identity": projection.villain_final.content_identity,
            "prepared_ranges_identity": prepared.content_identity,
            "provenance": projection.provenance,
            "ordered_joint_rows": tuple(
                (
                    row.hero_combo,
                    row.villain_combo,
                    row.raw_joint_mass,
                    row.probability,
                )
                for row in rows
            ),
        }
    )


def _mapping_identity(
    hero: ComboBucketMap, villain: ComboBucketMap
) -> str:
    return _sha256_identity(
        {
            "algorithm": MAPPING_IDENTITY_ALGORITHM,
            "hero": hero,
            "villain": villain,
        }
    )


def _tree_identity(
    request: KnownBoardRealCardHuRiverRequest,
    joint_identity: str,
    mapping_identity: str,
) -> str:
    return _sha256_identity(
        {
            "algorithm": TREE_IDENTITY_ALGORITHM,
            "joint_identity": joint_identity,
            "mapping_identity": mapping_identity,
            "position": KNOWN_BOARD_RIVER_POSITION,
            "initial_commitment_hero": request.initial_commitment_hero,
            "initial_commitment_villain": request.initial_commitment_villain,
            "rake_rate": request.rake_rate,
            "rake_cap": request.rake_cap,
            "oop_bet_size": request.oop_bet_size,
            "ip_bet_after_check_size": request.ip_bet_after_check_size,
            "ip_raise_to_size": request.ip_raise_to_size,
            "terminal_lines": (
                "check-check",
                "check-bet-call",
                "check-bet-fold",
                "bet-call",
                "bet-fold",
                "bet-raise-call",
                "bet-raise-fold",
            ),
        }
    )


def _baseline_identity(
    tree_identity: str,
    hero_profile: RiverActionProfile,
    villain_profile: RiverActionProfile,
    source: str,
) -> str:
    return _sha256_identity(
        {
            "algorithm": BASELINE_IDENTITY_ALGORITHM,
            "tree_identity": tree_identity,
            "hero_profile": hero_profile,
            "villain_profile": villain_profile,
            "villain_profile_source": source,
        }
    )


def _response_identity(
    tree_identity: str,
    strategy_identity: str,
    request: KnownBoardRealCardHuRiverRequest,
) -> str:
    return _sha256_identity(
        {
            "algorithm": RESPONSE_IDENTITY_ALGORITHM,
            "tree_identity": tree_identity,
            "strategy_identity": strategy_identity,
            "method": KNOWN_BOARD_RIVER_RESPONSE_METHOD,
            "tolerance": request.tolerance,
            "max_br_list_materialization": (
                request.limits.max_br_list_materialization
            ),
        }
    )


def _analysis_identity(
    request: KnownBoardRealCardHuRiverRequest,
    baseline_identity: str,
    candidate_ids: tuple[str, ...],
    response_ids: tuple[str, ...],
    shifts: tuple[float, ...],
    workload: KnownBoardRiverWorkloadProjection,
) -> str:
    return _sha256_identity(
        {
            "algorithm": ANALYSIS_IDENTITY_ALGORITHM,
            "baseline_identity": baseline_identity,
            "candidate_ids": candidate_ids,
            "response_ids": response_ids,
            "generation": {
                "shift_amounts": shifts,
                "max_simultaneous_info_sets": (
                    request.max_simultaneous_info_sets
                ),
            },
            "repeated": {
                "horizon": request.horizon,
                "discount": request.discount,
                "tolerance": request.tolerance,
                "minimum_total_uplift": request.minimum_total_uplift,
                "max_horizon": request.max_horizon,
            },
            "aiof_limits": request.aiof_limits,
            "adapter_limits": request.limits,
            "coverage": {
                "generated_candidate_ids": candidate_ids,
                "kept_candidate_ids": candidate_ids,
                "filtering_applied": False,
            },
            "workload": workload,
        }
    )


def _deterministic_villain_strategy(
    tree: GameTree, response: BestResponseResult
) -> VillainStrategy:
    if not response.best_response_strategies:
        raise AiofContractError(
            AiofStatus.ORACLE_MISMATCH,
            "DP response returned no deterministic representative",
        )
    representative = response.best_response_strategies[0]
    info_sets = collect_villain_info_sets(tree)
    if set(representative) != set(info_sets):
        raise AiofContractError(
            AiofStatus.ORACLE_MISMATCH,
            "DP representative is not complete",
        )
    return VillainStrategy(
        {
            info_set: {
                action: 1.0 if action == representative[info_set] else 0.0
                for action in info_sets[info_set]
            }
            for info_set in sorted(info_sets)
        }
    )


def _validate_response_numbers(response: BestResponseResult, name: str) -> None:
    values = (
        response.villain_max_ev,
        response.ev_h_worst,
        response.ev_h_best,
        response.expected_house_rake_worst,
        response.expected_house_rake_best,
    )
    if not all(math.isfinite(value) for value in values):
        raise AiofContractError(
            AiofStatus.NUMERIC_FAILURE, f"{name} contains non-finite values"
        )
    if response.ev_h_worst > response.ev_h_best:
        raise AiofContractError(
            AiofStatus.ORACLE_MISMATCH,
            f"{name} Hero worst value exceeds best value",
        )
    if response.num_best_response_strategies is None or (
        response.num_best_response_strategies
        < len(response.best_response_strategies)
    ):
        raise AiofContractError(
            AiofStatus.ORACLE_MISMATCH,
            f"{name} correspondence counts are inconsistent",
        )


def _validate_comparison_numbers(report: CandidateComparisonReport) -> None:
    baseline = report.baseline_value
    if not all(
        math.isfinite(value)
        for value in (baseline.hero_ev, baseline.villain_ev, baseline.house_rake)
    ):
        raise AiofContractError(
            AiofStatus.NUMERIC_FAILURE, "baseline fixed profile is non-finite"
        )
    for row in report.comparisons:
        values = (
            row.fixed_profile_value.hero_ev,
            row.fixed_profile_value.villain_ev,
            row.fixed_profile_value.house_rake,
            row.hero_ev_diff_from_baseline,
            row.villain_ev_diff_from_baseline,
            row.post_response_hero_ev_worst_diff,
            row.post_response_hero_ev_best_diff,
            row.candidate.l1_distance,
        )
        if not all(math.isfinite(value) for value in values):
            raise AiofContractError(
                AiofStatus.NUMERIC_FAILURE,
                "candidate comparison contains non-finite values",
            )
        _validate_response_numbers(row.best_response, "candidate response")


def _prepared_ranges_checked(
    request: KnownBoardRealCardHuRiverRequest,
    unavailable: tuple[str, ...],
    projection: _PreparedProjection,
) -> PreparedRanges:
    prepared = prepare_compatible_ranges(
        request.hero_range,
        request.villain_range,
        unavailable,
        _bounded_aiof_limits(request),
    )
    provenance = projection.provenance
    if (
        prepared.compatible_pair_count != provenance.compatible_pair_count
        or not math.isclose(
            prepared.compatible_raw_joint_mass,
            provenance.compatible_raw_joint_mass,
            rel_tol=0.0,
            abs_tol=1e-12 * max(1.0, abs(provenance.compatible_raw_joint_mass)),
        )
        or not math.isclose(
            prepared.normalization_factor,
            provenance.normalization_factor,
            rel_tol=0.0,
            abs_tol=1e-12 * max(1.0, abs(provenance.normalization_factor)),
        )
    ):
        raise AiofContractError(
            AiofStatus.ACCOUNTING_MISMATCH,
            "M13 prepared range projection disagrees with adapter projection",
        )
    return prepared


def _execute(
    request: KnownBoardRealCardHuRiverRequest,
    board: tuple[str, ...],
    dead: tuple[str, ...],
    unavailable: tuple[str, ...],
    shift_amounts: tuple[float, ...],
    selector_configuration: AutomaticCommitmentSelectionConfig,
) -> KnownBoardRealCardHuRiverPayload:
    # Phase 1-3: bounded range expansion and streaming pair projection.
    projection = _project_joint_support(request, board, unavailable)
    hero_combos = tuple(item.combo for item in projection.hero_final.combos)
    villain_combos = tuple(item.combo for item in projection.villain_final.combos)

    # Phase 4: strict mapping/profile coverage, before tree/profile allocation.
    hero_mapping = _canonical_mapping(
        request.hero_combo_to_bucket,
        hero_combos,
        "hero",
        request.limits.max_buckets_per_seat,
    )
    villain_mapping = _canonical_mapping(
        request.villain_combo_to_bucket,
        villain_combos,
        "villain",
        request.limits.max_buckets_per_seat,
    )
    hero_profile, hero_probability_map = _canonical_profile(
        request.baseline_hero_profile,
        hero_mapping.bucket_ids,
        "hero",
        request.tolerance,
    )
    supplied_villain_profile = None
    supplied_villain_probabilities = None
    if request.baseline_villain_profile is not None:
        supplied_villain_profile, supplied_villain_probabilities = _canonical_profile(
            request.baseline_villain_profile,
            villain_mapping.bucket_ids,
            "villain",
            request.tolerance,
        )
    option_counts = _project_feasible_shift_counts(
        hero_probability_map, shift_amounts
    )

    # Phase 5: every remaining allocation/analysis projection.
    workload = _project_workload(
        request,
        projection.provenance,
        hero_mapping,
        villain_mapping,
        option_counts,
        auto_baseline=supplied_villain_profile is None,
    )

    # Phase 6: only now materialize shift components, pair ranks, and the tree.
    options = _feasible_shift_options(hero_probability_map, shift_amounts)
    prepared = _prepared_ranges_checked(request, unavailable, projection)
    joint_rows = _materialize_joint_rows(
        projection, board, hero_mapping, villain_mapping
    )
    tree = _build_tree(request, joint_rows)
    actual_nodes = sum(1 for _ in iter_nodes(tree.root))
    if actual_nodes != workload.tree_nodes:
        raise AiofContractError(
            AiofStatus.ACCOUNTING_MISMATCH,
            "native tree node count differs from projection",
        )
    baseline_hero_strategy = HeroStrategy(hero_probability_map)
    try:
        validate_hero_strategy(
            tree, baseline_hero_strategy, tolerance=request.tolerance
        )
    except ValueError as exc:
        raise AiofContractError(AiofStatus.INVALID_STRATEGY, str(exc)) from exc

    board_identity = _board_identity(board, dead)
    prepared_joint_identity = _prepared_joint_identity(
        board_identity, projection, prepared, joint_rows
    )
    profile_mapping_identity = _mapping_identity(hero_mapping, villain_mapping)
    tree_identity = _tree_identity(
        request, prepared_joint_identity, profile_mapping_identity
    )

    # Auto baseline is a deterministic complete representative, not the scalar
    # Hero-worst end of the correspondence.
    auto_response = None
    baseline_response_identity = None
    if supplied_villain_probabilities is None:
        try:
            auto_response = solve_exact_response(
                tree,
                baseline_hero_strategy,
                tolerance=request.tolerance,
                max_pure_strategies=request.limits.max_br_list_materialization,
                method=KNOWN_BOARD_RIVER_RESPONSE_METHOD,
            )
        except ValueError as exc:
            raise AiofContractError(AiofStatus.INVALID_STRATEGY, str(exc)) from exc
        _validate_response_numbers(auto_response, "auto baseline response")
        baseline_villain_strategy = _deterministic_villain_strategy(
            tree, auto_response
        )
        baseline_villain_profile = _strategy_to_profile(
            baseline_villain_strategy, villain_mapping.bucket_ids
        )
        baseline_source = "auto_best_response"
        baseline_response_identity = _response_identity(
            tree_identity,
            _sha256_identity(
                {
                    "hero_profile": hero_profile,
                    "purpose": "auto_baseline",
                }
            ),
            request,
        )
    else:
        baseline_villain_strategy = VillainStrategy(
            supplied_villain_probabilities
        )
        baseline_villain_profile = supplied_villain_profile
        baseline_source = "supplied_profile"
    try:
        validate_villain_strategy(
            tree, baseline_villain_strategy, tolerance=request.tolerance
        )
    except ValueError as exc:
        raise AiofContractError(AiofStatus.INVALID_STRATEGY, str(exc)) from exc

    baseline_identity = _baseline_identity(
        tree_identity,
        hero_profile,
        baseline_villain_profile,
        baseline_source,
    )
    if (
        request.expected_baseline_identity is not None
        and request.expected_baseline_identity != baseline_identity
    ):
        raise AiofContractError(
            AiofStatus.INVALID_INPUT, "expected baseline identity mismatch"
        )

    # Phase 7: candidate allocation and identity-collision checks precede all
    # fixed-profile/candidate analyses.
    candidates = _materialize_candidates(
        baseline_hero_strategy,
        baseline_identity,
        options,
        request.max_simultaneous_info_sets,
    )
    if len(candidates) != workload.candidate_count:
        raise AiofContractError(
            AiofStatus.ORACLE_MISMATCH,
            "candidate materialization count differs from projection",
        )
    for candidate in candidates:
        try:
            validate_hero_strategy(
                tree, candidate.hero_strategy, tolerance=request.tolerance
            )
        except ValueError as exc:
            raise AiofContractError(AiofStatus.INVALID_STRATEGY, str(exc)) from exc

    candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
    response_ids = tuple(
        _response_identity(tree_identity, candidate_id, request)
        for candidate_id in candidate_ids
    )
    analysis_identity = _analysis_identity(
        request,
        baseline_identity,
        candidate_ids,
        response_ids,
        shift_amounts,
        workload,
    )

    # Phase 8: native fixed-profile comparison and native DP responses.
    try:
        comparison_report = compare_candidates(
            tree,
            baseline_hero_strategy,
            baseline_villain_strategy,
            candidates,
            tolerance=request.tolerance,
            max_pure_strategies=request.limits.max_br_list_materialization,
        )
    except ValueError as exc:
        raise AiofContractError(AiofStatus.INVALID_STRATEGY, str(exc)) from exc
    _validate_comparison_numbers(comparison_report)
    if len(comparison_report.comparisons) != len(candidates):
        raise AiofContractError(
            AiofStatus.ORACLE_MISMATCH,
            "comparison report is incomplete",
        )
    comparison_by_id = {
        row.candidate.candidate_id: row for row in comparison_report.comparisons
    }
    if tuple(sorted(comparison_by_id)) != tuple(sorted(candidate_ids)):
        raise AiofContractError(
            AiofStatus.ORACLE_MISMATCH,
            "comparison candidate identities do not match",
        )

    coverage = KnownBoardRiverCandidateCoverage(
        generated_candidate_ids=tuple(sorted(candidate_ids)),
        kept_candidate_ids=tuple(sorted(candidate_ids)),
        shift_amounts=shift_amounts,
        max_simultaneous_info_sets=request.max_simultaneous_info_sets,
        filtering_applied=False,
    )
    selector_coverage = AutomaticCommitmentSearchCoverage(
        input_candidate_ids=coverage.generated_candidate_ids,
        kept_candidate_ids=coverage.kept_candidate_ids,
        source="known_board_real_card_hu_river_generated",
        shift_amounts=shift_amounts,
        max_simultaneous_info_sets=request.max_simultaneous_info_sets,
        generation_max_candidates=request.limits.max_candidates,
        filtering_applied=False,
    )
    try:
        automatic_selection = select_automatic_commitments(
            comparison_report,
            horizon=request.horizon,
            discount=request.discount,
            tolerance=request.tolerance,
            max_horizon=request.max_horizon,
            configuration=selector_configuration,
            search_coverage=selector_coverage,
        )
    except ValueError as exc:
        raise AiofContractError(AiofStatus.INVALID_INPUT, str(exc)) from exc

    baseline_value = comparison_report.baseline_value
    if auto_response is not None:
        # The fixed representative value, not the response Hero-worst scalar,
        # is the baseline b used by the native comparison.
        direct_baseline = evaluate_fixed_profile(
            tree,
            baseline_hero_strategy,
            baseline_villain_strategy,
            tolerance=request.tolerance,
        )
        if direct_baseline != baseline_value:
            raise AiofContractError(
                AiofStatus.ORACLE_MISMATCH,
                "auto baseline representative value changed unexpectedly",
            )
    baseline_record = KnownBoardRiverBaselineRecord(
        baseline_source,
        hero_profile,
        baseline_villain_profile,
        baseline_villain_strategy,
        baseline_value,
        auto_response,
        baseline_response_identity,
    )
    candidate_records = tuple(
        KnownBoardRiverCandidateRecord(
            candidate_id,
            response_id,
            comparison_by_id[candidate_id].candidate.shifts,
            comparison_by_id[candidate_id],
        )
        for candidate_id, response_id in sorted(zip(candidate_ids, response_ids))
    )
    payload = KnownBoardRealCardHuRiverPayload(
        board,
        dead,
        board_identity,
        prepared_joint_identity,
        profile_mapping_identity,
        tree_identity,
        baseline_identity,
        analysis_identity,
        prepared,
        projection.provenance,
        hero_mapping,
        villain_mapping,
        joint_rows,
        tree,
        baseline_record,
        workload,
        coverage,
        candidate_records,
        comparison_report,
        automatic_selection,
        request,
    )
    _assert_json_safe(payload.to_dict())
    return payload


def _clean_error(message: str, fallback: str) -> str:
    cleaned = " ".join(
        (message or fallback).replace("\r", " ").replace("\n", " ").split()
    )
    return cleaned[:500] or fallback


def analyze_known_board_real_card_hu_river(
    request: KnownBoardRealCardHuRiverRequest,
) -> KnownBoardRealCardHuRiverResult:
    """Run the bounded adapter with DP-only response and no partial failures.

    Success contains every conditioned combo row, the dedicated native tree,
    the complete fixed baseline, every declared candidate, native comparison
    and exact-response objects, and M27 selections.  Any controlled or
    unexpected failure returns ``payload=None``; no prefix result is exposed.
    """

    try:
        (
            validated,
            board,
            dead,
            unavailable,
            shifts,
            selector_configuration,
        ) = _validate_request(request)
        payload = _execute(
            validated,
            board,
            dead,
            unavailable,
            shifts,
            selector_configuration,
        )
        return KnownBoardRealCardHuRiverResult(
            AiofStatus.SUCCESS, payload, None
        )
    except AiofContractError as exc:
        return KnownBoardRealCardHuRiverResult(
            exc.status,
            None,
            _clean_error(str(exc), exc.status.value),
        )
    except (ArithmeticError, OverflowError) as exc:
        return KnownBoardRealCardHuRiverResult(
            AiofStatus.NUMERIC_FAILURE,
            None,
            _clean_error(str(exc), "numeric failure"),
        )
    except Exception:
        return KnownBoardRealCardHuRiverResult(
            AiofStatus.NUMERIC_FAILURE,
            None,
            "unexpected adapter failure",
        )
