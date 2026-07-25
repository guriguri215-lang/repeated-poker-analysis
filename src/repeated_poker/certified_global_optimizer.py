"""Certified continuous/global Hero commitment optimizer core.

The optimizer derives the complete product of legal Hero behavior-policy
simplexes from :class:`HeroBehaviorScenario`.  It never accepts candidate
lists, probability shifts, a grid resolution, local bounds, or a warm-start
neighbourhood.  The supplied baseline policy is an objective comparison point
and initial feasible incumbent only; it does not restrict the domain.

The scalar response-oracle boundary is deliberately strict.  A conforming
oracle evaluates the baseline-relative total repeated Hero-EV uplift using
only the complete response correspondence's ``hero_worst`` value and returns a
valid upper bound over every requested behavior-policy cell.  The core cannot
manufacture a safe poker-specific bound.  A consumer that cannot provide one
must raise :class:`UnsupportedScalarOracleDomain`; it is never silently
replaced by a local optimizer, random restart, finite grid, sample, or partial
search.

All probabilities, values, bounds, splits, and gaps are exact rationals.  A
cell is the intersection of per-action closed intervals with each information
set's nonnegative, sum-to-one simplex.  Splitting one effective action interval
at its exact midpoint produces two closed children whose union is the parent.
Thus the active leaves cover the entire unpruned domain.  Because every active
leaf carries an oracle-attested whole-cell upper bound, the maximum of those
bounds and the incumbent value is a valid global upper bound.  A leaf is
pruned only when its bound cannot improve the incumbent.  These invariants are
the source-level soundness argument for the returned certificate, conditional
on the explicitly identified scalar oracle satisfying its bound contract.

Success means only ``CERTIFIED_GLOBAL`` or ``CERTIFIED_EPSILON_GLOBAL`` for the
specified tolerance.  It does not claim a poker equilibrium, solver-grade
scalability, real-world profitability, strategy advice, or unlimited game
size.  Every limit or contract failure returns a null payload with no selected
commitment.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import re
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from typing import Any, Protocol


CONTRACT_VERSION = "m36-certified-continuous-global-optimizer-core-v1"
ALGORITHM_VERSION = "exact-rational-simplex-cell-branch-and-bound-v1"
BOUND_CONTRACT_VERSION = "whole-cell-scalar-global-upper-bound-v1"
DOMAIN_CONTRACT = "full-product-of-legal-hero-behavior-policy-simplexes-v1"
OBJECTIVE_SEMANTICS = "baseline-relative-total-repeated-hero-ev-uplift-v1"
RESPONSE_SEMANTICS = (
    "complete-response-correspondence-hero-worst-only-v1"
)
CLAIM_SCOPE = "specified-tolerance-certified-global-commitment-optimum-v1"

CERTIFIED_GLOBAL = "CERTIFIED_GLOBAL"
CERTIFIED_EPSILON_GLOBAL = "CERTIFIED_EPSILON_GLOBAL"
LIMIT_REACHED_NO_CERTIFICATE = "LIMIT_REACHED_NO_CERTIFICATE"
UNSUPPORTED_DOMAIN = "UNSUPPORTED_DOMAIN"
INVALID_INPUT = "INVALID_INPUT"
STALE_INPUT = "STALE_INPUT"
INVALID_ORACLE_CONTRACT = "INVALID_ORACLE_CONTRACT"
ORACLE_FAILURE = "ORACLE_FAILURE"
NUMERIC_FAILURE = "NUMERIC_FAILURE"
INTERNAL_FAILURE = "INTERNAL_FAILURE"

DEFAULT_MAX_INFORMATION_SETS = 16
DEFAULT_MAX_ACTIONS_PER_INFORMATION_SET = 8
DEFAULT_MAX_DECISION_VARIABLES = 64
DEFAULT_MAX_CELLS = 16_383
DEFAULT_MAX_NODES = 8_192
DEFAULT_MAX_ORACLE_CALLS = 32_768
DEFAULT_MAX_BOUND_RECORDS = 16_383
DEFAULT_MAX_IDENTITY_RECORDS = 200_000
DEFAULT_MAX_RATIONAL_NUMERATOR_BITS = 512
DEFAULT_MAX_RATIONAL_DENOMINATOR_BITS = 512
DEFAULT_MAX_OUTPUT_RECORDS = 200_000
DEFAULT_MAX_OUTPUT_BYTES = 8_000_000

HARD_MAX_INFORMATION_SETS = 64
HARD_MAX_ACTIONS_PER_INFORMATION_SET = 32
HARD_MAX_DECISION_VARIABLES = 512
HARD_MAX_CELLS = 1_048_575
HARD_MAX_NODES = 524_288
HARD_MAX_ORACLE_CALLS = 2_097_152
HARD_MAX_BOUND_RECORDS = 1_048_575
HARD_MAX_IDENTITY_RECORDS = 4_000_000
HARD_MAX_RATIONAL_NUMERATOR_BITS = 4_096
HARD_MAX_RATIONAL_DENOMINATOR_BITS = 4_096
HARD_MAX_OUTPUT_RECORDS = 2_000_000
HARD_MAX_OUTPUT_BYTES = 256_000_000

_IDENTITY_RE = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9_.:@/+~-]{1,128}")
_INTEGER_RE = re.compile(r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)")
_FRACTION_RE = re.compile(r"(-?[1-9][0-9]*)/([1-9][0-9]*)")


@dataclass(frozen=True)
class HeroInformationSet:
    """One Hero information set and all of its legal actions."""

    information_set_id: str
    legal_action_ids: tuple[str, ...]


@dataclass(frozen=True)
class HeroBehaviorScenario:
    """Scenario identity and the complete legal Hero behavior-policy surface."""

    scenario_identity: str
    information_sets: tuple[HeroInformationSet, ...]


@dataclass(frozen=True)
class ExactActionProbability:
    """One canonical exact action probability."""

    action_id: str
    probability: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ExactBehaviorRow:
    """One complete exact behavior distribution."""

    information_set_id: str
    actions: tuple[ExactActionProbability, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "information_set_id": self.information_set_id,
            "actions": [action.to_dict() for action in self.actions],
        }


@dataclass(frozen=True)
class ExactBehaviorPolicy:
    """A complete Hero behavior policy with exact simplex probabilities."""

    rows: tuple[ExactBehaviorRow, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"rows": [row.to_dict() for row in self.rows]}


@dataclass(frozen=True)
class ExactActionInterval:
    """One action's closed exact-rational interval inside a search cell."""

    action_id: str
    lower: str
    upper: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ExactBehaviorCellRow:
    """All action intervals for one simplex-constrained information set."""

    information_set_id: str
    actions: tuple[ExactActionInterval, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "information_set_id": self.information_set_id,
            "actions": [action.to_dict() for action in self.actions],
            "constraint": "nonnegative-and-sum-to-one-exact",
        }


@dataclass(frozen=True)
class ExactBehaviorCell:
    """A closed product cell intersected with every exact simplex."""

    rows: tuple[ExactBehaviorCellRow, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"rows": [row.to_dict() for row in self.rows]}


@dataclass(frozen=True)
class ScalarOracleEvaluation:
    """One exact point evaluation under the mandatory scalar semantics.

    ``candidate_total_repeated_hero_ev`` must use the complete response
    correspondence's Hero-worst value wherever response safety enters the
    repeated objective.  ``complete_response_correspondence_identity`` binds
    the full correspondence rather than a first witness or pure subset.
    """

    policy_identity: str
    response_oracle_identity: str
    objective_identity: str
    baseline_total_repeated_hero_ev: str
    candidate_total_repeated_hero_ev: str
    uplift: str
    complete_response_correspondence_identity: str
    complete_response_record_count: int
    hero_worst_witness_count: int
    response_semantics: str = RESPONSE_SEMANTICS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScalarOracleBound:
    """A valid exact upper bound over one entire behavior-policy cell.

    ``candidate_policy`` is optional feasible lower-bound guidance.  It never
    changes the cell or search domain and is evaluated by the point oracle
    before it may become an incumbent.
    """

    cell_identity: str
    response_oracle_identity: str
    objective_identity: str
    upper_bound: str
    bound_identity: str
    candidate_policy: ExactBehaviorPolicy | None = None
    valid_for_entire_cell: bool = True
    bound_contract_version: str = BOUND_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_identity": self.cell_identity,
            "response_oracle_identity": self.response_oracle_identity,
            "objective_identity": self.objective_identity,
            "upper_bound": self.upper_bound,
            "bound_identity": self.bound_identity,
            "candidate_policy": (
                None
                if self.candidate_policy is None
                else self.candidate_policy.to_dict()
            ),
            "valid_for_entire_cell": self.valid_for_entire_cell,
            "bound_contract_version": self.bound_contract_version,
        }


class CertifiedScalarResponseOracle(Protocol):
    """Structural protocol for a deterministic certified scalar oracle."""

    response_oracle_identity: str
    objective_identity: str
    response_semantics: str
    bound_contract_version: str

    def evaluate(self, policy: ExactBehaviorPolicy) -> ScalarOracleEvaluation:
        """Return the exact baseline-relative point objective."""

    def upper_bound(self, cell: ExactBehaviorCell) -> ScalarOracleBound:
        """Return a valid upper bound for every policy in ``cell``."""


class UnsupportedScalarOracleDomain(ValueError):
    """The oracle cannot construct a safe bound for the requested cell."""


class ScalarOracleResourceLimit(ValueError):
    """The oracle reached a nested hard limit before producing a certificate."""


@dataclass(frozen=True)
class CertifiedGlobalOptimizerLimits:
    """Caller-controlled limits bounded by immutable hard ceilings."""

    max_information_sets: int = DEFAULT_MAX_INFORMATION_SETS
    max_actions_per_information_set: int = (
        DEFAULT_MAX_ACTIONS_PER_INFORMATION_SET
    )
    max_decision_variables: int = DEFAULT_MAX_DECISION_VARIABLES
    max_cells: int = DEFAULT_MAX_CELLS
    max_nodes: int = DEFAULT_MAX_NODES
    max_oracle_calls: int = DEFAULT_MAX_ORACLE_CALLS
    max_bound_records: int = DEFAULT_MAX_BOUND_RECORDS
    max_identity_records: int = DEFAULT_MAX_IDENTITY_RECORDS
    max_rational_numerator_bits: int = (
        DEFAULT_MAX_RATIONAL_NUMERATOR_BITS
    )
    max_rational_denominator_bits: int = (
        DEFAULT_MAX_RATIONAL_DENOMINATOR_BITS
    )
    max_output_records: int = DEFAULT_MAX_OUTPUT_RECORDS
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES


@dataclass(frozen=True)
class CertifiedGlobalOptimizerPins:
    """Optional exact identity pins; mismatches are stale input."""

    domain_identity: str | None = None
    baseline_policy_identity: str | None = None
    response_oracle_identity: str | None = None
    objective_identity: str | None = None
    run_identity: str | None = None


@dataclass(frozen=True)
class CertifiedGlobalWorkCounters:
    """Deterministic completed-work counters for success or failure."""

    domain_information_sets: int = 0
    domain_actions: int = 0
    decision_variables: int = 0
    cells_created: int = 0
    nodes_processed: int = 0
    active_cells_peak: int = 0
    cells_pruned: int = 0
    splits: int = 0
    oracle_point_calls: int = 0
    oracle_bound_calls: int = 0
    oracle_calls_total: int = 0
    point_cache_hits: int = 0
    bound_records: int = 0
    identity_records: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class CertifiedGlobalCertificate:
    """Sound global lower/upper bounds and their exact termination gap."""

    incumbent_lower_bound: str
    valid_global_upper_bound: str
    absolute_gap: str
    relative_gap: str
    requested_absolute_tolerance: str
    requested_relative_tolerance: str
    domain_identity: str
    response_oracle_identity: str
    objective_identity: str
    baseline_policy_identity: str
    run_identity: str
    bound_contract_version: str
    domain_contract: str
    objective_semantics: str
    response_semantics: str
    work_counters: CertifiedGlobalWorkCounters

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["work_counters"] = self.work_counters.to_dict()
        return output


@dataclass(frozen=True)
class CertifiedGlobalOptimizerPayload:
    """Complete successful certified optimization payload."""

    domain: dict[str, Any]
    baseline_policy: ExactBehaviorPolicy
    baseline_evaluation: ScalarOracleEvaluation
    selected_commitment: ExactBehaviorPolicy | None
    selected_evaluation: ScalarOracleEvaluation | None
    selected_policy_identity: str | None
    incumbent_tied_policy_identities: tuple[str, ...]
    no_beneficial_commitment: bool
    certificate: CertifiedGlobalCertificate

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "claim_scope": CLAIM_SCOPE,
            "domain": self.domain,
            "baseline_policy": self.baseline_policy.to_dict(),
            "baseline_evaluation": self.baseline_evaluation.to_dict(),
            "selected_commitment": (
                None
                if self.selected_commitment is None
                else self.selected_commitment.to_dict()
            ),
            "selected_evaluation": (
                None
                if self.selected_evaluation is None
                else self.selected_evaluation.to_dict()
            ),
            "selected_policy_identity": self.selected_policy_identity,
            "incumbent_tied_policy_identities": list(
                self.incumbent_tied_policy_identities
            ),
            "no_beneficial_commitment": self.no_beneficial_commitment,
            "certificate": self.certificate.to_dict(),
        }


@dataclass(frozen=True)
class CertifiedGlobalOptimizerError:
    """Bounded failure metadata carrying no partial optimizer payload."""

    phase: str
    message: str
    cause_status: str | None = None


@dataclass(frozen=True)
class CertifiedGlobalOptimizerResult:
    """Exclusive certified-success or fail-closed result."""

    status: str
    payload: CertifiedGlobalOptimizerPayload | None
    error: CertifiedGlobalOptimizerError | None
    work_counters: CertifiedGlobalWorkCounters
    partial_result: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "payload": None if self.payload is None else self.payload.to_dict(),
            "error": None if self.error is None else asdict(self.error),
            "work_counters": self.work_counters.to_dict(),
            "partial_result": self.partial_result,
        }


@dataclass
class _MutableCounters:
    domain_information_sets: int = 0
    domain_actions: int = 0
    decision_variables: int = 0
    cells_created: int = 0
    nodes_processed: int = 0
    active_cells_peak: int = 0
    cells_pruned: int = 0
    splits: int = 0
    oracle_point_calls: int = 0
    oracle_bound_calls: int = 0
    oracle_calls_total: int = 0
    point_cache_hits: int = 0
    bound_records: int = 0
    identity_records: int = 0

    def snapshot(self) -> CertifiedGlobalWorkCounters:
        return CertifiedGlobalWorkCounters(
            **{
                field.name: getattr(self, field.name)
                for field in fields(CertifiedGlobalWorkCounters)
            }
        )


@dataclass(frozen=True)
class _CanonicalScenario:
    scenario_identity: str
    rows: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class _CellRecord:
    cell: ExactBehaviorCell
    cell_identity: str
    upper_bound: Fraction
    depth: int


@dataclass
class _Incumbent:
    value: Fraction
    policies: dict[
        str, tuple[ExactBehaviorPolicy, ScalarOracleEvaluation]
    ]


class _OptimizerFailure(ValueError):
    def __init__(
        self,
        status: str,
        phase: str,
        message: str,
        *,
        cause_status: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.phase = phase
        self.cause_status = cause_status


_HARD_LIMITS = {
    "max_information_sets": HARD_MAX_INFORMATION_SETS,
    "max_actions_per_information_set": HARD_MAX_ACTIONS_PER_INFORMATION_SET,
    "max_decision_variables": HARD_MAX_DECISION_VARIABLES,
    "max_cells": HARD_MAX_CELLS,
    "max_nodes": HARD_MAX_NODES,
    "max_oracle_calls": HARD_MAX_ORACLE_CALLS,
    "max_bound_records": HARD_MAX_BOUND_RECORDS,
    "max_identity_records": HARD_MAX_IDENTITY_RECORDS,
    "max_rational_numerator_bits": HARD_MAX_RATIONAL_NUMERATOR_BITS,
    "max_rational_denominator_bits": HARD_MAX_RATIONAL_DENOMINATOR_BITS,
    "max_output_records": HARD_MAX_OUTPUT_RECORDS,
    "max_output_bytes": HARD_MAX_OUTPUT_BYTES,
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _identity(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _rational_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _parse_rational(
    value: object,
    phase: str,
    limits: CertifiedGlobalOptimizerLimits,
    *,
    nonnegative: bool = False,
) -> Fraction:
    if isinstance(value, bool):
        raise _OptimizerFailure(
            INVALID_INPUT, phase, f"{phase} must be an exact rational"
        )
    result: Fraction
    if isinstance(value, Fraction):
        result = value
    elif type(value) is int:
        result = Fraction(value)
    elif type(value) is str:
        if _INTEGER_RE.fullmatch(value):
            result = Fraction(int(value))
        else:
            match = _FRACTION_RE.fullmatch(value)
            if match is None:
                raise _OptimizerFailure(
                    INVALID_INPUT,
                    phase,
                    f"{phase} must be a canonical integer or fraction",
                )
            numerator = int(match.group(1))
            denominator = int(match.group(2))
            if denominator == 1:
                raise _OptimizerFailure(
                    INVALID_INPUT,
                    phase,
                    f"{phase} fraction with denominator 1 is noncanonical",
                )
            result = Fraction(numerator, denominator)
            if _rational_text(result) != value:
                raise _OptimizerFailure(
                    INVALID_INPUT,
                    phase,
                    f"{phase} must be reduced canonical rational text",
                )
    else:
        raise _OptimizerFailure(
            INVALID_INPUT, phase, f"{phase} must be an exact rational"
        )
    if result.numerator.bit_length() > limits.max_rational_numerator_bits:
        raise _OptimizerFailure(
            NUMERIC_FAILURE, phase, f"{phase} numerator bit cap exceeded"
        )
    if result.denominator.bit_length() > limits.max_rational_denominator_bits:
        raise _OptimizerFailure(
            NUMERIC_FAILURE, phase, f"{phase} denominator bit cap exceeded"
        )
    if nonnegative and result < 0:
        raise _OptimizerFailure(
            INVALID_INPUT, phase, f"{phase} must be nonnegative"
        )
    return result


def _require_identity(value: object, phase: str) -> str:
    if type(value) is not str or _IDENTITY_RE.fullmatch(value) is None:
        raise _OptimizerFailure(
            INVALID_INPUT, phase, f"{phase} must be lowercase SHA-256"
        )
    return value


def _require_identifier(value: object, phase: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise _OptimizerFailure(
            INVALID_INPUT, phase, f"{phase} has invalid identifier grammar"
        )
    return value


def _validate_limits(limits: object) -> CertifiedGlobalOptimizerLimits:
    if type(limits) is not CertifiedGlobalOptimizerLimits:
        raise _OptimizerFailure(
            INVALID_INPUT,
            "limits",
            "limits must be CertifiedGlobalOptimizerLimits",
        )
    for field in fields(CertifiedGlobalOptimizerLimits):
        value = getattr(limits, field.name)
        if type(value) is not int or value <= 0:
            raise _OptimizerFailure(
                INVALID_INPUT,
                f"limits.{field.name}",
                f"{field.name} must be a positive int",
            )
        maximum = _HARD_LIMITS[field.name]
        if value > maximum:
            raise _OptimizerFailure(
                INVALID_INPUT,
                f"limits.{field.name}",
                f"{field.name} exceeds immutable hard ceiling {maximum}",
            )
    return limits


def _canonical_scenario(
    scenario: object,
    limits: CertifiedGlobalOptimizerLimits,
    counters: _MutableCounters,
) -> _CanonicalScenario:
    if type(scenario) is not HeroBehaviorScenario:
        raise _OptimizerFailure(
            INVALID_INPUT,
            "domain",
            "scenario must be HeroBehaviorScenario",
        )
    scenario_identity = _require_identity(
        scenario.scenario_identity, "domain.scenario_identity"
    )
    if type(scenario.information_sets) is not tuple:
        raise _OptimizerFailure(
            INVALID_INPUT,
            "domain.information_sets",
            "information_sets must be a tuple",
        )
    information_set_count = len(scenario.information_sets)
    if information_set_count == 0:
        raise _OptimizerFailure(
            UNSUPPORTED_DOMAIN,
            "domain.information_sets",
            "at least one Hero information set is required",
        )
    if information_set_count > limits.max_information_sets:
        raise _OptimizerFailure(
            UNSUPPORTED_DOMAIN,
            "domain.projection",
            "max_information_sets exceeded before domain materialization",
        )

    total_actions = 0
    decision_variables = 0
    for index, row in enumerate(scenario.information_sets):
        if type(row) is not HeroInformationSet:
            raise _OptimizerFailure(
                INVALID_INPUT,
                f"domain.information_sets[{index}]",
                "every information set must be HeroInformationSet",
            )
        if type(row.legal_action_ids) is not tuple:
            raise _OptimizerFailure(
                INVALID_INPUT,
                f"domain.information_sets[{index}].legal_action_ids",
                "legal_action_ids must be a tuple",
            )
        action_count = len(row.legal_action_ids)
        if action_count == 0:
            raise _OptimizerFailure(
                UNSUPPORTED_DOMAIN,
                f"domain.information_sets[{index}]",
                "every information set requires at least one legal action",
            )
        if action_count > limits.max_actions_per_information_set:
            raise _OptimizerFailure(
                UNSUPPORTED_DOMAIN,
                "domain.projection",
                "max_actions_per_information_set exceeded before domain materialization",
            )
        total_actions += action_count
        decision_variables += action_count - 1
        if decision_variables > limits.max_decision_variables:
            raise _OptimizerFailure(
                UNSUPPORTED_DOMAIN,
                "domain.projection",
                "max_decision_variables exceeded before domain materialization",
            )

    counters.domain_information_sets = information_set_count
    counters.domain_actions = total_actions
    counters.decision_variables = decision_variables

    canonical_rows: list[tuple[str, tuple[str, ...]]] = []
    seen_information_sets: set[str] = set()
    for index, row in enumerate(scenario.information_sets):
        information_set_id = _require_identifier(
            row.information_set_id,
            f"domain.information_sets[{index}].information_set_id",
        )
        if information_set_id in seen_information_sets:
            raise _OptimizerFailure(
                INVALID_INPUT,
                "domain.information_sets",
                "duplicate information_set_id",
            )
        seen_information_sets.add(information_set_id)
        actions = tuple(
            _require_identifier(
                action,
                f"domain.{information_set_id}.legal_action_ids[{action_index}]",
            )
            for action_index, action in enumerate(row.legal_action_ids)
        )
        if len(set(actions)) != len(actions):
            raise _OptimizerFailure(
                INVALID_INPUT,
                f"domain.{information_set_id}.legal_action_ids",
                "duplicate legal action",
            )
        canonical_rows.append((information_set_id, tuple(sorted(actions))))
    canonical_rows.sort(key=lambda item: item[0])
    return _CanonicalScenario(
        scenario_identity=scenario_identity,
        rows=tuple(canonical_rows),
    )


def _canonical_policy(
    scenario: _CanonicalScenario,
    policy: object,
    limits: CertifiedGlobalOptimizerLimits,
    phase: str,
) -> ExactBehaviorPolicy:
    if type(policy) is not ExactBehaviorPolicy:
        raise _OptimizerFailure(
            INVALID_INPUT, phase, f"{phase} must be ExactBehaviorPolicy"
        )
    if type(policy.rows) is not tuple:
        raise _OptimizerFailure(
            INVALID_INPUT, phase, f"{phase}.rows must be a tuple"
        )
    row_map: dict[str, ExactBehaviorRow] = {}
    for index, row in enumerate(policy.rows):
        if type(row) is not ExactBehaviorRow:
            raise _OptimizerFailure(
                INVALID_INPUT,
                phase,
                f"{phase}.rows[{index}] must be ExactBehaviorRow",
            )
        information_set_id = _require_identifier(
            row.information_set_id,
            f"{phase}.rows[{index}].information_set_id",
        )
        if information_set_id in row_map:
            raise _OptimizerFailure(
                INVALID_INPUT, phase, f"{phase} has duplicate information set"
            )
        row_map[information_set_id] = row
    expected_information_sets = {row[0] for row in scenario.rows}
    if set(row_map) != expected_information_sets:
        raise _OptimizerFailure(
            INVALID_INPUT,
            phase,
            f"{phase} must contain every and only scenario information set",
        )
    canonical_rows: list[ExactBehaviorRow] = []
    for information_set_id, legal_actions in scenario.rows:
        source = row_map[information_set_id]
        if type(source.actions) is not tuple:
            raise _OptimizerFailure(
                INVALID_INPUT,
                phase,
                f"{phase}.{information_set_id}.actions must be a tuple",
            )
        action_map: dict[str, Fraction] = {}
        for action_index, action in enumerate(source.actions):
            if type(action) is not ExactActionProbability:
                raise _OptimizerFailure(
                    INVALID_INPUT,
                    phase,
                    f"{phase}.{information_set_id}.actions[{action_index}] invalid",
                )
            action_id = _require_identifier(
                action.action_id,
                f"{phase}.{information_set_id}.action_id",
            )
            if action_id in action_map:
                raise _OptimizerFailure(
                    INVALID_INPUT,
                    phase,
                    f"{phase}.{information_set_id} has duplicate action",
                )
            probability = _parse_rational(
                action.probability,
                f"{phase}.{information_set_id}.{action_id}",
                limits,
                nonnegative=True,
            )
            if probability > 1:
                raise _OptimizerFailure(
                    INVALID_INPUT,
                    phase,
                    f"{phase}.{information_set_id}.{action_id} exceeds one",
                )
            action_map[action_id] = probability
        if set(action_map) != set(legal_actions):
            raise _OptimizerFailure(
                INVALID_INPUT,
                phase,
                f"{phase}.{information_set_id} must contain every legal action",
            )
        if sum(action_map.values(), Fraction(0)) != 1:
            raise _OptimizerFailure(
                INVALID_INPUT,
                phase,
                f"{phase}.{information_set_id} must sum exactly to one",
            )
        canonical_rows.append(
            ExactBehaviorRow(
                information_set_id=information_set_id,
                actions=tuple(
                    ExactActionProbability(
                        action_id=action_id,
                        probability=_rational_text(action_map[action_id]),
                    )
                    for action_id in legal_actions
                ),
            )
        )
    return ExactBehaviorPolicy(rows=tuple(canonical_rows))


def _domain_projection(scenario: _CanonicalScenario) -> dict[str, Any]:
    return {
        "contract": DOMAIN_CONTRACT,
        "scenario_identity": scenario.scenario_identity,
        "information_sets": [
            {
                "information_set_id": information_set_id,
                "legal_action_ids": list(actions),
                "constraint": "probabilities>=0-and-sum=1-exact",
                "continuous": True,
            }
            for information_set_id, actions in scenario.rows
        ],
    }


def behavior_policy_identity(
    scenario: HeroBehaviorScenario, policy: ExactBehaviorPolicy
) -> str:
    """Return the canonical identity of a complete exact policy."""

    limits = CertifiedGlobalOptimizerLimits(
        max_information_sets=HARD_MAX_INFORMATION_SETS,
        max_actions_per_information_set=HARD_MAX_ACTIONS_PER_INFORMATION_SET,
        max_decision_variables=HARD_MAX_DECISION_VARIABLES,
        max_cells=HARD_MAX_CELLS,
        max_nodes=HARD_MAX_NODES,
        max_oracle_calls=HARD_MAX_ORACLE_CALLS,
        max_bound_records=HARD_MAX_BOUND_RECORDS,
        max_identity_records=HARD_MAX_IDENTITY_RECORDS,
        max_rational_numerator_bits=HARD_MAX_RATIONAL_NUMERATOR_BITS,
        max_rational_denominator_bits=HARD_MAX_RATIONAL_DENOMINATOR_BITS,
        max_output_records=HARD_MAX_OUTPUT_RECORDS,
        max_output_bytes=HARD_MAX_OUTPUT_BYTES,
    )
    counters = _MutableCounters()
    canonical_scenario = _canonical_scenario(scenario, limits, counters)
    canonical = _canonical_policy(
        canonical_scenario, policy, limits, "policy"
    )
    return _identity(
        {
            "algorithm": "exact-behavior-policy-sha256-canonical-json-v1",
            "domain": _domain_projection(canonical_scenario),
            "policy": canonical.to_dict(),
        }
    )


def _root_cell(scenario: _CanonicalScenario) -> ExactBehaviorCell:
    return ExactBehaviorCell(
        rows=tuple(
            ExactBehaviorCellRow(
                information_set_id=information_set_id,
                actions=tuple(
                    ExactActionInterval(
                        action_id=action_id, lower="0", upper="1"
                    )
                    for action_id in actions
                ),
            )
            for information_set_id, actions in scenario.rows
        )
    )


def _cell_fraction_rows(
    cell: ExactBehaviorCell,
    limits: CertifiedGlobalOptimizerLimits,
    phase: str,
) -> tuple[tuple[str, tuple[tuple[str, Fraction, Fraction], ...]], ...]:
    rows = []
    for row in cell.rows:
        actions = []
        for action in row.actions:
            lower = _parse_rational(
                action.lower, f"{phase}.{action.action_id}.lower", limits
            )
            upper = _parse_rational(
                action.upper, f"{phase}.{action.action_id}.upper", limits
            )
            if lower < 0 or upper > 1 or lower > upper:
                raise _OptimizerFailure(
                    INVALID_ORACLE_CONTRACT,
                    phase,
                    "cell interval is outside [0,1] or reversed",
                )
            actions.append((action.action_id, lower, upper))
        if (
            sum(action[1] for action in actions) > 1
            or sum(action[2] for action in actions) < 1
        ):
            raise _OptimizerFailure(
                INVALID_ORACLE_CONTRACT,
                phase,
                "cell does not intersect its exact simplex",
            )
        rows.append((row.information_set_id, tuple(actions)))
    return tuple(rows)


def behavior_cell_identity(
    scenario: HeroBehaviorScenario, cell: ExactBehaviorCell
) -> str:
    """Return the canonical identity of one full-cell bound request."""

    limits = CertifiedGlobalOptimizerLimits(
        max_information_sets=HARD_MAX_INFORMATION_SETS,
        max_actions_per_information_set=HARD_MAX_ACTIONS_PER_INFORMATION_SET,
        max_decision_variables=HARD_MAX_DECISION_VARIABLES,
        max_cells=HARD_MAX_CELLS,
        max_nodes=HARD_MAX_NODES,
        max_oracle_calls=HARD_MAX_ORACLE_CALLS,
        max_bound_records=HARD_MAX_BOUND_RECORDS,
        max_identity_records=HARD_MAX_IDENTITY_RECORDS,
        max_rational_numerator_bits=HARD_MAX_RATIONAL_NUMERATOR_BITS,
        max_rational_denominator_bits=HARD_MAX_RATIONAL_DENOMINATOR_BITS,
        max_output_records=HARD_MAX_OUTPUT_RECORDS,
        max_output_bytes=HARD_MAX_OUTPUT_BYTES,
    )
    counters = _MutableCounters()
    canonical_scenario = _canonical_scenario(scenario, limits, counters)
    if type(cell) is not ExactBehaviorCell:
        raise ValueError("cell must be ExactBehaviorCell")
    _cell_fraction_rows(cell, limits, "cell")
    expected_rows = tuple(row[0] for row in canonical_scenario.rows)
    if tuple(row.information_set_id for row in cell.rows) != expected_rows:
        raise ValueError("cell rows are not canonical for scenario")
    return _identity(
        {
            "algorithm": "exact-behavior-cell-sha256-canonical-json-v1",
            "domain": _domain_projection(canonical_scenario),
            "cell": cell.to_dict(),
        }
    )


def scalar_oracle_bound_identity(
    *,
    cell_identity: str,
    response_oracle_identity: str,
    objective_identity: str,
    upper_bound: str,
    candidate_policy_identity: str | None,
) -> str:
    """Return the required deterministic identity for an oracle bound."""

    for name, value in (
        ("cell_identity", cell_identity),
        ("response_oracle_identity", response_oracle_identity),
        ("objective_identity", objective_identity),
    ):
        if type(value) is not str or _IDENTITY_RE.fullmatch(value) is None:
            raise ValueError(f"{name} must be lowercase SHA-256")
    hard = CertifiedGlobalOptimizerLimits(
        max_information_sets=HARD_MAX_INFORMATION_SETS,
        max_actions_per_information_set=HARD_MAX_ACTIONS_PER_INFORMATION_SET,
        max_decision_variables=HARD_MAX_DECISION_VARIABLES,
        max_cells=HARD_MAX_CELLS,
        max_nodes=HARD_MAX_NODES,
        max_oracle_calls=HARD_MAX_ORACLE_CALLS,
        max_bound_records=HARD_MAX_BOUND_RECORDS,
        max_identity_records=HARD_MAX_IDENTITY_RECORDS,
        max_rational_numerator_bits=HARD_MAX_RATIONAL_NUMERATOR_BITS,
        max_rational_denominator_bits=HARD_MAX_RATIONAL_DENOMINATOR_BITS,
        max_output_records=HARD_MAX_OUTPUT_RECORDS,
        max_output_bytes=HARD_MAX_OUTPUT_BYTES,
    )
    try:
        parsed = _parse_rational(upper_bound, "upper_bound", hard)
    except _OptimizerFailure as exc:
        raise ValueError(str(exc)) from exc
    if candidate_policy_identity is not None:
        if (
            type(candidate_policy_identity) is not str
            or _IDENTITY_RE.fullmatch(candidate_policy_identity) is None
        ):
            raise ValueError(
                "candidate_policy_identity must be lowercase SHA-256 or None"
            )
    return _identity(
        {
            "algorithm": "scalar-oracle-bound-sha256-canonical-json-v1",
            "bound_contract_version": BOUND_CONTRACT_VERSION,
            "cell_identity": cell_identity,
            "response_oracle_identity": response_oracle_identity,
            "objective_identity": objective_identity,
            "upper_bound": _rational_text(parsed),
            "candidate_policy_identity": candidate_policy_identity,
        }
    )


def _reserve_identity(
    counters: _MutableCounters,
    limits: CertifiedGlobalOptimizerLimits,
    phase: str,
    count: int = 1,
) -> None:
    if counters.identity_records + count > limits.max_identity_records:
        raise _OptimizerFailure(
            LIMIT_REACHED_NO_CERTIFICATE,
            phase,
            "max_identity_records reached before identity materialization",
        )
    counters.identity_records += count


def _policy_identity(
    domain_projection: dict[str, Any],
    policy: ExactBehaviorPolicy,
    counters: _MutableCounters,
    limits: CertifiedGlobalOptimizerLimits,
) -> str:
    _reserve_identity(counters, limits, "identity.policy")
    return _identity(
        {
            "algorithm": "exact-behavior-policy-sha256-canonical-json-v1",
            "domain": domain_projection,
            "policy": policy.to_dict(),
        }
    )


def _cell_identity(
    domain_projection: dict[str, Any],
    cell: ExactBehaviorCell,
    counters: _MutableCounters,
    limits: CertifiedGlobalOptimizerLimits,
) -> str:
    _reserve_identity(counters, limits, "identity.cell")
    return _identity(
        {
            "algorithm": "exact-behavior-cell-sha256-canonical-json-v1",
            "domain": domain_projection,
            "cell": cell.to_dict(),
        }
    )


def _policy_in_cell(
    policy: ExactBehaviorPolicy,
    cell: ExactBehaviorCell,
) -> bool:
    for policy_row, cell_row in zip(policy.rows, cell.rows):
        for probability, interval in zip(
            policy_row.actions, cell_row.actions
        ):
            value = Fraction(probability.probability)
            if value < Fraction(interval.lower) or value > Fraction(
                interval.upper
            ):
                return False
    return True


def _cell_center(
    cell: ExactBehaviorCell,
    limits: CertifiedGlobalOptimizerLimits,
) -> ExactBehaviorPolicy:
    rows: list[ExactBehaviorRow] = []
    parsed_rows = _cell_fraction_rows(cell, limits, "cell.center")
    for information_set_id, actions in parsed_rows:
        lowers = [action[1] for action in actions]
        capacities = [action[2] - action[1] for action in actions]
        residual = Fraction(1) - sum(lowers, Fraction(0))
        total_capacity = sum(capacities, Fraction(0))
        if total_capacity == 0:
            if residual != 0:
                raise _OptimizerFailure(
                    NUMERIC_FAILURE,
                    "cell.center",
                    "singleton cell is infeasible",
                )
            values = lowers
        else:
            values = [
                lower + residual * capacity / total_capacity
                for lower, capacity in zip(lowers, capacities)
            ]
        if sum(values, Fraction(0)) != 1:
            raise _OptimizerFailure(
                NUMERIC_FAILURE,
                "cell.center",
                "exact center does not sum to one",
            )
        rows.append(
            ExactBehaviorRow(
                information_set_id=information_set_id,
                actions=tuple(
                    ExactActionProbability(
                        action_id=action[0],
                        probability=_rational_text(value),
                    )
                    for action, value in zip(actions, values)
                ),
            )
        )
    return ExactBehaviorPolicy(rows=tuple(rows))


def _oracle_metadata(
    oracle: object,
) -> tuple[str, str]:
    try:
        response_identity = getattr(oracle, "response_oracle_identity")
        objective_identity = getattr(oracle, "objective_identity")
        response_semantics = getattr(oracle, "response_semantics")
        bound_contract = getattr(oracle, "bound_contract_version")
        evaluate = getattr(oracle, "evaluate")
        upper_bound = getattr(oracle, "upper_bound")
    except Exception as exc:
        raise _OptimizerFailure(
            INVALID_ORACLE_CONTRACT,
            "oracle.metadata",
            "oracle is missing required deterministic contract metadata",
        ) from exc
    _require_identity(response_identity, "oracle.response_oracle_identity")
    _require_identity(objective_identity, "oracle.objective_identity")
    if response_semantics != RESPONSE_SEMANTICS:
        raise _OptimizerFailure(
            INVALID_ORACLE_CONTRACT,
            "oracle.response_semantics",
            "oracle must use complete response correspondence hero_worst only",
        )
    if bound_contract != BOUND_CONTRACT_VERSION:
        raise _OptimizerFailure(
            INVALID_ORACLE_CONTRACT,
            "oracle.bound_contract_version",
            "oracle does not implement the required whole-cell bound contract",
        )
    if not callable(evaluate) or not callable(upper_bound):
        raise _OptimizerFailure(
            INVALID_ORACLE_CONTRACT,
            "oracle.methods",
            "oracle evaluate and upper_bound must be callable",
        )
    return response_identity, objective_identity


def _preflight_oracle_calls(
    counters: _MutableCounters,
    limits: CertifiedGlobalOptimizerLimits,
    count: int,
    phase: str,
) -> None:
    if counters.oracle_calls_total + count > limits.max_oracle_calls:
        raise _OptimizerFailure(
            LIMIT_REACHED_NO_CERTIFICATE,
            phase,
            "max_oracle_calls reached before oracle invocation",
        )


def _validate_evaluation(
    evaluation: object,
    *,
    policy_identity: str,
    response_identity: str,
    objective_identity: str,
    limits: CertifiedGlobalOptimizerLimits,
    phase: str,
) -> tuple[ScalarOracleEvaluation, Fraction]:
    if type(evaluation) is not ScalarOracleEvaluation:
        raise _OptimizerFailure(
            INVALID_ORACLE_CONTRACT,
            phase,
            "oracle evaluate must return ScalarOracleEvaluation",
        )
    if evaluation.policy_identity != policy_identity:
        raise _OptimizerFailure(
            INVALID_ORACLE_CONTRACT, phase, "evaluation policy identity mismatch"
        )
    if evaluation.response_oracle_identity != response_identity:
        raise _OptimizerFailure(
            INVALID_ORACLE_CONTRACT,
            phase,
            "evaluation response oracle identity mismatch",
        )
    if evaluation.objective_identity != objective_identity:
        raise _OptimizerFailure(
            INVALID_ORACLE_CONTRACT,
            phase,
            "evaluation objective identity mismatch",
        )
    if evaluation.response_semantics != RESPONSE_SEMANTICS:
        raise _OptimizerFailure(
            INVALID_ORACLE_CONTRACT,
            phase,
            "evaluation response semantics mismatch",
        )
    _require_identity(
        evaluation.complete_response_correspondence_identity,
        f"{phase}.complete_response_correspondence_identity",
    )
    if (
        type(evaluation.complete_response_record_count) is not int
        or evaluation.complete_response_record_count <= 0
    ):
        raise _OptimizerFailure(
            INVALID_ORACLE_CONTRACT,
            phase,
            "complete response record count must be positive",
        )
    if (
        type(evaluation.hero_worst_witness_count) is not int
        or evaluation.hero_worst_witness_count <= 0
        or evaluation.hero_worst_witness_count
        > evaluation.complete_response_record_count
    ):
        raise _OptimizerFailure(
            INVALID_ORACLE_CONTRACT,
            phase,
            "hero_worst witness count is invalid",
        )
    baseline = _parse_rational(
        evaluation.baseline_total_repeated_hero_ev,
        f"{phase}.baseline_total_repeated_hero_ev",
        limits,
    )
    candidate = _parse_rational(
        evaluation.candidate_total_repeated_hero_ev,
        f"{phase}.candidate_total_repeated_hero_ev",
        limits,
    )
    uplift = _parse_rational(
        evaluation.uplift, f"{phase}.uplift", limits
    )
    if candidate - baseline != uplift:
        raise _OptimizerFailure(
            INVALID_ORACLE_CONTRACT,
            phase,
            "uplift must equal candidate minus baseline total repeated Hero EV",
        )
    return evaluation, uplift


def _evaluate_policy(
    oracle: CertifiedScalarResponseOracle,
    policy: ExactBehaviorPolicy,
    policy_identity: str,
    *,
    response_identity: str,
    objective_identity: str,
    limits: CertifiedGlobalOptimizerLimits,
    counters: _MutableCounters,
    cache: dict[
        str, tuple[ExactBehaviorPolicy, ScalarOracleEvaluation, Fraction]
    ],
    phase: str,
) -> tuple[ScalarOracleEvaluation, Fraction]:
    cached = cache.get(policy_identity)
    if cached is not None:
        counters.point_cache_hits += 1
        return cached[1], cached[2]
    _preflight_oracle_calls(counters, limits, 1, phase)
    counters.oracle_calls_total += 1
    counters.oracle_point_calls += 1
    try:
        raw = oracle.evaluate(policy)
    except UnsupportedScalarOracleDomain as exc:
        raise _OptimizerFailure(
            UNSUPPORTED_DOMAIN, phase, str(exc) or "oracle domain unsupported"
        ) from exc
    except ScalarOracleResourceLimit as exc:
        raise _OptimizerFailure(
            LIMIT_REACHED_NO_CERTIFICATE,
            phase,
            str(exc) or "nested oracle resource limit reached",
        ) from exc
    except Exception as exc:
        raise _OptimizerFailure(
            ORACLE_FAILURE, phase, "scalar point oracle failed"
        ) from exc
    evaluation, uplift = _validate_evaluation(
        raw,
        policy_identity=policy_identity,
        response_identity=response_identity,
        objective_identity=objective_identity,
        limits=limits,
        phase=phase,
    )
    cache[policy_identity] = (policy, evaluation, uplift)
    return evaluation, uplift


def _validate_bound(
    raw: object,
    *,
    cell: ExactBehaviorCell,
    cell_identity: str,
    scenario: _CanonicalScenario,
    domain_projection: dict[str, Any],
    response_identity: str,
    objective_identity: str,
    limits: CertifiedGlobalOptimizerLimits,
    counters: _MutableCounters,
    phase: str,
) -> tuple[ScalarOracleBound, Fraction, ExactBehaviorPolicy | None, str | None]:
    if type(raw) is not ScalarOracleBound:
        raise _OptimizerFailure(
            INVALID_ORACLE_CONTRACT,
            phase,
            "oracle upper_bound must return ScalarOracleBound",
        )
    if raw.cell_identity != cell_identity:
        raise _OptimizerFailure(
            INVALID_ORACLE_CONTRACT, phase, "bound cell identity mismatch"
        )
    if raw.response_oracle_identity != response_identity:
        raise _OptimizerFailure(
            INVALID_ORACLE_CONTRACT,
            phase,
            "bound response oracle identity mismatch",
        )
    if raw.objective_identity != objective_identity:
        raise _OptimizerFailure(
            INVALID_ORACLE_CONTRACT, phase, "bound objective identity mismatch"
        )
    if raw.bound_contract_version != BOUND_CONTRACT_VERSION:
        raise _OptimizerFailure(
            INVALID_ORACLE_CONTRACT, phase, "bound contract version mismatch"
        )
    if raw.valid_for_entire_cell is not True:
        raise _OptimizerFailure(
            INVALID_ORACLE_CONTRACT,
            phase,
            "bound must attest validity for the entire cell",
        )
    upper = _parse_rational(
        raw.upper_bound, f"{phase}.upper_bound", limits
    )
    candidate: ExactBehaviorPolicy | None = None
    candidate_identity: str | None = None
    if raw.candidate_policy is not None:
        candidate = _canonical_policy(
            scenario, raw.candidate_policy, limits, f"{phase}.candidate_policy"
        )
        if not _policy_in_cell(candidate, cell):
            raise _OptimizerFailure(
                INVALID_ORACLE_CONTRACT,
                phase,
                "bound candidate policy is outside the bounded cell",
            )
        candidate_identity = _policy_identity(
            domain_projection, candidate, counters, limits
        )
    _reserve_identity(counters, limits, f"{phase}.bound_identity")
    expected_bound_identity = scalar_oracle_bound_identity(
        cell_identity=cell_identity,
        response_oracle_identity=response_identity,
        objective_identity=objective_identity,
        upper_bound=_rational_text(upper),
        candidate_policy_identity=candidate_identity,
    )
    if raw.bound_identity != expected_bound_identity:
        raise _OptimizerFailure(
            INVALID_ORACLE_CONTRACT, phase, "bound identity mismatch"
        )
    return raw, upper, candidate, candidate_identity


def _bound_cell(
    oracle: CertifiedScalarResponseOracle,
    cell: ExactBehaviorCell,
    cell_identity: str,
    *,
    scenario: _CanonicalScenario,
    domain_projection: dict[str, Any],
    response_identity: str,
    objective_identity: str,
    limits: CertifiedGlobalOptimizerLimits,
    counters: _MutableCounters,
    phase: str,
) -> tuple[Fraction, ExactBehaviorPolicy | None, str | None]:
    if counters.bound_records + 1 > limits.max_bound_records:
        raise _OptimizerFailure(
            LIMIT_REACHED_NO_CERTIFICATE,
            phase,
            "max_bound_records reached before bound request",
        )
    _preflight_oracle_calls(counters, limits, 1, phase)
    counters.oracle_calls_total += 1
    counters.oracle_bound_calls += 1
    counters.bound_records += 1
    try:
        raw = oracle.upper_bound(cell)
    except UnsupportedScalarOracleDomain as exc:
        raise _OptimizerFailure(
            UNSUPPORTED_DOMAIN,
            phase,
            str(exc) or "oracle cannot construct a safe whole-cell bound",
        ) from exc
    except ScalarOracleResourceLimit as exc:
        raise _OptimizerFailure(
            LIMIT_REACHED_NO_CERTIFICATE,
            phase,
            str(exc) or "nested oracle resource limit reached",
        ) from exc
    except Exception as exc:
        raise _OptimizerFailure(
            ORACLE_FAILURE, phase, "scalar bound oracle failed"
        ) from exc
    _, upper, candidate, candidate_identity = _validate_bound(
        raw,
        cell=cell,
        cell_identity=cell_identity,
        scenario=scenario,
        domain_projection=domain_projection,
        response_identity=response_identity,
        objective_identity=objective_identity,
        limits=limits,
        counters=counters,
        phase=phase,
    )
    return upper, candidate, candidate_identity


def _update_incumbent(
    incumbent: _Incumbent,
    policy: ExactBehaviorPolicy,
    evaluation: ScalarOracleEvaluation,
    uplift: Fraction,
    policy_identity: str,
    counters: _MutableCounters,
    limits: CertifiedGlobalOptimizerLimits,
) -> None:
    if uplift > incumbent.value:
        incumbent.value = uplift
        incumbent.policies.clear()
        incumbent.policies[policy_identity] = (policy, evaluation)
    elif uplift == incumbent.value and policy_identity not in incumbent.policies:
        _reserve_identity(counters, limits, "identity.incumbent_tie")
        incumbent.policies[policy_identity] = (policy, evaluation)


def _effective_split(
    cell: ExactBehaviorCell,
    limits: CertifiedGlobalOptimizerLimits,
) -> tuple[int, int, Fraction] | None:
    rows = _cell_fraction_rows(cell, limits, "cell.split")
    best: tuple[int, int, Fraction] | None = None
    for row_index, (_, actions) in enumerate(rows):
        total_lower = sum(action[1] for action in actions)
        total_upper = sum(action[2] for action in actions)
        for action_index, (_, lower, upper) in enumerate(actions):
            effective_lower = max(
                lower, Fraction(1) - (total_upper - upper)
            )
            effective_upper = min(
                upper, Fraction(1) - (total_lower - lower)
            )
            width = effective_upper - effective_lower
            if width <= 0:
                continue
            if best is None or width > best[2]:
                best = (row_index, action_index, width)
    return best


def _split_cell(
    cell: ExactBehaviorCell,
    split: tuple[int, int, Fraction],
    limits: CertifiedGlobalOptimizerLimits,
) -> tuple[ExactBehaviorCell, ExactBehaviorCell]:
    row_index, action_index, _ = split
    parsed = _cell_fraction_rows(cell, limits, "cell.split")
    _, actions = parsed[row_index]
    target = actions[action_index]
    total_lower = sum(action[1] for action in actions)
    total_upper = sum(action[2] for action in actions)
    effective_lower = max(
        target[1], Fraction(1) - (total_upper - target[2])
    )
    effective_upper = min(
        target[2], Fraction(1) - (total_lower - target[1])
    )
    midpoint = (effective_lower + effective_upper) / 2

    children: list[ExactBehaviorCell] = []
    for side in ("left", "right"):
        rows: list[ExactBehaviorCellRow] = []
        for current_row_index, row in enumerate(cell.rows):
            intervals: list[ExactActionInterval] = []
            for current_action_index, interval in enumerate(row.actions):
                lower = Fraction(interval.lower)
                upper = Fraction(interval.upper)
                if (
                    current_row_index == row_index
                    and current_action_index == action_index
                ):
                    if side == "left":
                        upper = min(upper, midpoint)
                    else:
                        lower = max(lower, midpoint)
                intervals.append(
                    ExactActionInterval(
                        action_id=interval.action_id,
                        lower=_rational_text(lower),
                        upper=_rational_text(upper),
                    )
                )
            rows.append(
                ExactBehaviorCellRow(
                    information_set_id=row.information_set_id,
                    actions=tuple(intervals),
                )
            )
        child = ExactBehaviorCell(rows=tuple(rows))
        _cell_fraction_rows(child, limits, "cell.child")
        children.append(child)
    return children[0], children[1]


def _relative_gap(lower: Fraction, gap: Fraction) -> Fraction:
    denominator = max(Fraction(1), abs(lower))
    return gap / denominator


def _certified_status(
    lower: Fraction,
    upper: Fraction,
    absolute_tolerance: Fraction,
    relative_tolerance: Fraction,
) -> str | None:
    if upper < lower:
        raise _OptimizerFailure(
            NUMERIC_FAILURE,
            "certificate",
            "global upper bound is below incumbent lower bound",
        )
    gap = upper - lower
    if gap == 0:
        return CERTIFIED_GLOBAL
    if gap <= absolute_tolerance:
        return CERTIFIED_EPSILON_GLOBAL
    if relative_tolerance > 0 and _relative_gap(lower, gap) <= relative_tolerance:
        return CERTIFIED_EPSILON_GLOBAL
    return None


def _count_output_records(value: Any, maximum: int) -> int:
    count = 0
    stack = [value]
    while stack:
        current = stack.pop()
        count += 1
        if count > maximum:
            raise _OptimizerFailure(
                LIMIT_REACHED_NO_CERTIFICATE,
                "output.records",
                "max_output_records exceeded before output serialization",
            )
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    return count


def _check_output_bytes(value: Any, maximum: int) -> int:
    encoder = json.JSONEncoder(
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    total = 0
    for chunk in encoder.iterencode(value):
        total += len(chunk.encode("utf-8"))
        if total > maximum:
            raise _OptimizerFailure(
                LIMIT_REACHED_NO_CERTIFICATE,
                "output.bytes",
                "max_output_bytes exceeded during bounded streaming preflight",
            )
    return total


def _check_pin(
    supplied: str | None, actual: str, phase: str
) -> None:
    if supplied is None:
        return
    _require_identity(supplied, phase)
    if supplied != actual:
        raise _OptimizerFailure(STALE_INPUT, phase, f"{phase} mismatch")


def _validate_pins(pins: object) -> CertifiedGlobalOptimizerPins:
    if type(pins) is not CertifiedGlobalOptimizerPins:
        raise _OptimizerFailure(
            INVALID_INPUT,
            "pins",
            "pins must be CertifiedGlobalOptimizerPins",
        )
    for field in fields(CertifiedGlobalOptimizerPins):
        value = getattr(pins, field.name)
        if value is not None:
            _require_identity(value, f"pins.{field.name}")
    return pins


def _success_result(
    *,
    status: str,
    scenario: _CanonicalScenario,
    domain_projection: dict[str, Any],
    domain_identity: str,
    baseline_policy: ExactBehaviorPolicy,
    baseline_policy_identity: str,
    baseline_evaluation: ScalarOracleEvaluation,
    response_identity: str,
    objective_identity: str,
    run_identity: str,
    incumbent: _Incumbent,
    global_upper: Fraction,
    absolute_tolerance: Fraction,
    relative_tolerance: Fraction,
    counters: _MutableCounters,
    limits: CertifiedGlobalOptimizerLimits,
) -> CertifiedGlobalOptimizerResult:
    del scenario
    tied_ids = tuple(sorted(incumbent.policies))
    selected_id = tied_ids[0]
    selected_policy, selected_evaluation = incumbent.policies[selected_id]
    no_benefit = incumbent.value <= 0
    certificate = CertifiedGlobalCertificate(
        incumbent_lower_bound=_rational_text(incumbent.value),
        valid_global_upper_bound=_rational_text(global_upper),
        absolute_gap=_rational_text(global_upper - incumbent.value),
        relative_gap=_rational_text(
            _relative_gap(incumbent.value, global_upper - incumbent.value)
        ),
        requested_absolute_tolerance=_rational_text(absolute_tolerance),
        requested_relative_tolerance=_rational_text(relative_tolerance),
        domain_identity=domain_identity,
        response_oracle_identity=response_identity,
        objective_identity=objective_identity,
        baseline_policy_identity=baseline_policy_identity,
        run_identity=run_identity,
        bound_contract_version=BOUND_CONTRACT_VERSION,
        domain_contract=DOMAIN_CONTRACT,
        objective_semantics=OBJECTIVE_SEMANTICS,
        response_semantics=RESPONSE_SEMANTICS,
        work_counters=counters.snapshot(),
    )
    payload = CertifiedGlobalOptimizerPayload(
        domain={
            **domain_projection,
            "domain_identity": domain_identity,
            "information_set_count": counters.domain_information_sets,
            "legal_action_count": counters.domain_actions,
            "decision_variable_count": counters.decision_variables,
            "domain_source": "scenario-only",
            "candidate_list_used": False,
            "shift_amounts_used": False,
            "grid_resolution_used": False,
            "local_bounds_used": False,
        },
        baseline_policy=baseline_policy,
        baseline_evaluation=baseline_evaluation,
        selected_commitment=None if no_benefit else selected_policy,
        selected_evaluation=None if no_benefit else selected_evaluation,
        selected_policy_identity=None if no_benefit else selected_id,
        incumbent_tied_policy_identities=tied_ids,
        no_beneficial_commitment=no_benefit,
        certificate=certificate,
    )
    result = CertifiedGlobalOptimizerResult(
        status=status,
        payload=payload,
        error=None,
        work_counters=counters.snapshot(),
    )
    output = result.to_dict()
    _count_output_records(output, limits.max_output_records)
    _check_output_bytes(output, limits.max_output_bytes)
    return result


def _optimize(
    scenario_input: HeroBehaviorScenario,
    baseline_policy_input: ExactBehaviorPolicy,
    oracle: CertifiedScalarResponseOracle,
    *,
    absolute_gap_tolerance: object,
    relative_gap_tolerance: object,
    limits_input: CertifiedGlobalOptimizerLimits,
    pins_input: CertifiedGlobalOptimizerPins,
    counters: _MutableCounters,
) -> CertifiedGlobalOptimizerResult:
    limits = _validate_limits(limits_input)
    pins = _validate_pins(pins_input)
    absolute_tolerance = _parse_rational(
        absolute_gap_tolerance,
        "tolerance.absolute",
        limits,
        nonnegative=True,
    )
    relative_tolerance = _parse_rational(
        relative_gap_tolerance,
        "tolerance.relative",
        limits,
        nonnegative=True,
    )
    scenario = _canonical_scenario(scenario_input, limits, counters)
    domain_projection = _domain_projection(scenario)
    _reserve_identity(counters, limits, "identity.domain")
    domain_identity = _identity(
        {
            "algorithm": "full-hero-behavior-domain-sha256-canonical-json-v1",
            "domain": domain_projection,
        }
    )
    _check_pin(pins.domain_identity, domain_identity, "pins.domain_identity")

    baseline_policy = _canonical_policy(
        scenario, baseline_policy_input, limits, "baseline_policy"
    )
    baseline_policy_identity = _policy_identity(
        domain_projection, baseline_policy, counters, limits
    )
    _check_pin(
        pins.baseline_policy_identity,
        baseline_policy_identity,
        "pins.baseline_policy_identity",
    )
    response_identity, objective_identity = _oracle_metadata(oracle)
    _check_pin(
        pins.response_oracle_identity,
        response_identity,
        "pins.response_oracle_identity",
    )
    _check_pin(
        pins.objective_identity,
        objective_identity,
        "pins.objective_identity",
    )
    _reserve_identity(counters, limits, "identity.run")
    run_identity = _identity(
        {
            "contract_version": CONTRACT_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "domain_identity": domain_identity,
            "baseline_policy_identity": baseline_policy_identity,
            "response_oracle_identity": response_identity,
            "objective_identity": objective_identity,
            "objective_semantics": OBJECTIVE_SEMANTICS,
            "response_semantics": RESPONSE_SEMANTICS,
            "absolute_gap_tolerance": _rational_text(absolute_tolerance),
            "relative_gap_tolerance": _rational_text(relative_tolerance),
            "limits": asdict(limits),
        }
    )
    _check_pin(pins.run_identity, run_identity, "pins.run_identity")

    # Baseline point + root bound + possible new root point are preflighted
    # together so an undersized cap fails before the search begins.
    _preflight_oracle_calls(counters, limits, 3, "oracle.root_preflight")
    cache: dict[
        str, tuple[ExactBehaviorPolicy, ScalarOracleEvaluation, Fraction]
    ] = {}
    baseline_evaluation, baseline_uplift = _evaluate_policy(
        oracle,
        baseline_policy,
        baseline_policy_identity,
        response_identity=response_identity,
        objective_identity=objective_identity,
        limits=limits,
        counters=counters,
        cache=cache,
        phase="oracle.baseline",
    )
    if baseline_uplift != 0:
        raise _OptimizerFailure(
            INVALID_ORACLE_CONTRACT,
            "oracle.baseline",
            "baseline policy uplift must be exactly zero",
        )
    if (
        baseline_evaluation.baseline_total_repeated_hero_ev
        != baseline_evaluation.candidate_total_repeated_hero_ev
    ):
        raise _OptimizerFailure(
            INVALID_ORACLE_CONTRACT,
            "oracle.baseline",
            "baseline point must reproduce the baseline total repeated Hero EV",
        )
    incumbent = _Incumbent(
        value=Fraction(0),
        policies={
            baseline_policy_identity: (
                baseline_policy,
                baseline_evaluation,
            )
        },
    )

    if limits.max_cells < 1:
        raise _OptimizerFailure(
            LIMIT_REACHED_NO_CERTIFICATE,
            "search.root",
            "max_cells reached before root cell allocation",
        )
    root = _root_cell(scenario)
    counters.cells_created = 1
    root_identity = _cell_identity(
        domain_projection, root, counters, limits
    )
    root_upper, root_candidate, root_candidate_identity = _bound_cell(
        oracle,
        root,
        root_identity,
        scenario=scenario,
        domain_projection=domain_projection,
        response_identity=response_identity,
        objective_identity=objective_identity,
        limits=limits,
        counters=counters,
        phase="oracle.bound.root",
    )
    point = root_candidate or _cell_center(root, limits)
    point_identity = root_candidate_identity or _policy_identity(
        domain_projection, point, counters, limits
    )
    point_evaluation, point_uplift = _evaluate_policy(
        oracle,
        point,
        point_identity,
        response_identity=response_identity,
        objective_identity=objective_identity,
        limits=limits,
        counters=counters,
        cache=cache,
        phase="oracle.point.root",
    )
    if root_upper < max(Fraction(0), point_uplift):
        raise _OptimizerFailure(
            INVALID_ORACLE_CONTRACT,
            "oracle.bound.root",
            "whole-cell upper bound is below an evaluated point in the cell",
        )
    _update_incumbent(
        incumbent,
        point,
        point_evaluation,
        point_uplift,
        point_identity,
        counters,
        limits,
    )

    queue: list[tuple[Fraction, int, str, _CellRecord]] = []
    if root_upper > incumbent.value:
        root_record = _CellRecord(
            cell=root,
            cell_identity=root_identity,
            upper_bound=root_upper,
            depth=0,
        )
        heapq.heappush(
            queue,
            (-root_upper, 0, root_identity, root_record),
        )
        counters.active_cells_peak = 1
    else:
        counters.cells_pruned = 1

    while True:
        global_upper = (
            incumbent.value
            if not queue
            else max(incumbent.value, -queue[0][0])
        )
        status = _certified_status(
            incumbent.value,
            global_upper,
            absolute_tolerance,
            relative_tolerance,
        )
        if status is not None:
            return _success_result(
                status=status,
                scenario=scenario,
                domain_projection=domain_projection,
                domain_identity=domain_identity,
                baseline_policy=baseline_policy,
                baseline_policy_identity=baseline_policy_identity,
                baseline_evaluation=baseline_evaluation,
                response_identity=response_identity,
                objective_identity=objective_identity,
                run_identity=run_identity,
                incumbent=incumbent,
                global_upper=global_upper,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                counters=counters,
                limits=limits,
            )
        if not queue:
            raise _OptimizerFailure(
                NUMERIC_FAILURE,
                "search.coverage",
                "empty active cover did not certify the incumbent",
            )
        if counters.nodes_processed + 1 > limits.max_nodes:
            raise _OptimizerFailure(
                LIMIT_REACHED_NO_CERTIFICATE,
                "search.nodes",
                "max_nodes reached before processing next active cell",
            )
        _, _, _, record = heapq.heappop(queue)
        counters.nodes_processed += 1
        if record.upper_bound <= incumbent.value:
            counters.cells_pruned += 1 + len(queue)
            queue.clear()
            continue

        split = _effective_split(record.cell, limits)
        if split is None:
            raise _OptimizerFailure(
                UNSUPPORTED_DOMAIN,
                "search.bound_convergence",
                "oracle bound stays loose on a singleton cell",
            )
        # Both children, both whole-cell bounds, and at most two fresh point
        # evaluations are preflighted before either child is materialized.
        if counters.cells_created + 2 > limits.max_cells:
            raise _OptimizerFailure(
                LIMIT_REACHED_NO_CERTIFICATE,
                "search.cells",
                "max_cells reached before child cell allocation",
            )
        if counters.bound_records + 2 > limits.max_bound_records:
            raise _OptimizerFailure(
                LIMIT_REACHED_NO_CERTIFICATE,
                "search.bounds",
                "max_bound_records reached before child bound requests",
            )
        _preflight_oracle_calls(
            counters, limits, 4, "oracle.children_preflight"
        )
        left, right = _split_cell(record.cell, split, limits)
        counters.cells_created += 2
        counters.splits += 1
        for child_index, child in enumerate((left, right)):
            child_identity = _cell_identity(
                domain_projection, child, counters, limits
            )
            child_upper, candidate, candidate_identity = _bound_cell(
                oracle,
                child,
                child_identity,
                scenario=scenario,
                domain_projection=domain_projection,
                response_identity=response_identity,
                objective_identity=objective_identity,
                limits=limits,
                counters=counters,
                phase=f"oracle.bound.child[{child_index}]",
            )
            child_point = candidate or _cell_center(child, limits)
            child_point_identity = candidate_identity or _policy_identity(
                domain_projection, child_point, counters, limits
            )
            child_evaluation, child_uplift = _evaluate_policy(
                oracle,
                child_point,
                child_point_identity,
                response_identity=response_identity,
                objective_identity=objective_identity,
                limits=limits,
                counters=counters,
                cache=cache,
                phase=f"oracle.point.child[{child_index}]",
            )
            if child_upper < child_uplift:
                raise _OptimizerFailure(
                    INVALID_ORACLE_CONTRACT,
                    f"oracle.bound.child[{child_index}]",
                    "whole-cell upper bound is below its evaluated point",
                )
            _update_incumbent(
                incumbent,
                child_point,
                child_evaluation,
                child_uplift,
                child_point_identity,
                counters,
                limits,
            )
            if child_upper > incumbent.value:
                child_record = _CellRecord(
                    cell=child,
                    cell_identity=child_identity,
                    upper_bound=child_upper,
                    depth=record.depth + 1,
                )
                heapq.heappush(
                    queue,
                    (
                        -child_upper,
                        child_record.depth,
                        child_identity,
                        child_record,
                    ),
                )
            else:
                counters.cells_pruned += 1
        counters.active_cells_peak = max(
            counters.active_cells_peak, len(queue)
        )


def _failure_result(
    exc: _OptimizerFailure,
    counters: _MutableCounters,
) -> CertifiedGlobalOptimizerResult:
    return CertifiedGlobalOptimizerResult(
        status=exc.status,
        payload=None,
        error=CertifiedGlobalOptimizerError(
            phase=exc.phase,
            message=str(exc)[:512] or "certified optimizer failure",
            cause_status=exc.cause_status,
        ),
        work_counters=counters.snapshot(),
        partial_result=False,
    )


def optimize_certified_global_hero_commitment(
    scenario: HeroBehaviorScenario,
    baseline_policy: ExactBehaviorPolicy,
    oracle: CertifiedScalarResponseOracle,
    *,
    absolute_gap_tolerance: object = "0",
    relative_gap_tolerance: object = "0",
    limits: CertifiedGlobalOptimizerLimits = CertifiedGlobalOptimizerLimits(),
    pins: CertifiedGlobalOptimizerPins = CertifiedGlobalOptimizerPins(),
) -> CertifiedGlobalOptimizerResult:
    """Certify the global optimum over the full scenario-derived Hero domain.

    The objective is the oracle's exact baseline-relative total repeated Hero
    EV uplift.  The baseline is evaluated as a feasible point but does not
    constrain the search.  A success payload is returned only after the active
    whole-domain cover proves the requested absolute or relative gap.  A
    positive commitment is selected only when a positive evaluated uplift is
    certified; otherwise ``selected_commitment`` is null.
    """

    counters = _MutableCounters()
    try:
        return _optimize(
            scenario,
            baseline_policy,
            oracle,
            absolute_gap_tolerance=absolute_gap_tolerance,
            relative_gap_tolerance=relative_gap_tolerance,
            limits_input=limits,
            pins_input=pins,
            counters=counters,
        )
    except _OptimizerFailure as exc:
        return _failure_result(exc, counters)
    except Exception:
        return _failure_result(
            _OptimizerFailure(
                INTERNAL_FAILURE,
                "internal",
                "unexpected certified global optimizer failure",
            ),
            counters,
        )


def exact_certified_global_optimizer_json(
    result: CertifiedGlobalOptimizerResult,
) -> str:
    """Serialize one result as deterministic strict one-line JSON."""

    if type(result) is not CertifiedGlobalOptimizerResult:
        raise TypeError("result must be CertifiedGlobalOptimizerResult")
    return _canonical_json_bytes(result.to_dict()).decode("utf-8")
