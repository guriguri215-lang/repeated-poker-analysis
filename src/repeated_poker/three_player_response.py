"""Bounded exact response correspondence for a fixed-Hero reduced game.

The input is a finite exact-rational bimatrix whose strategic players are O1
and O2.  Hero is already fixed and ``R`` is a non-strategic rake account.  The
primary object is the complete mixed Nash response correspondence of O1 and
O2, represented as a deterministic union of ``support_cell`` records.

For every nonempty support pair ``(I, J)``, the O1 mixture constraints depend
only on O2's best-response conditions and the O2 mixture constraints depend
only on O1's best-response conditions.  Their Cartesian product is therefore
one support cell.  Each bounded marginal rational polytope is emitted by its
complete exact vertex set.  Exact duplicate cells are merged; partial overlaps
remain separate.  This module does not claim a topological connectivity
classification.

Vertices are found by complete active-constraint enumeration.  A separately
implemented verifier reconstructs constraints from the payoff table, repeats
active-set enumeration with independent elimination code, and checks every
support-pair outcome, vertex, residual, utility, and conservation identity.

Each payoff is bilinear on a product of two polytopes.  Holding one mixture
fixed makes it linear in the other, so a minimum or maximum can successively be
moved to a vertex of each marginal polytope.  Consequently all global H/O1/O2/R
extrema are obtained by evaluating every marginal vertex pair in every support
cell.  This statement does not make ``R`` strategic.

Default/hard ceilings are: pure plans O1 ``4/6`` and O2 ``4/6``; joint
profiles ``16/36``; support pairs ``225/3969``; support sizes O1 ``4/6`` and
O2 ``4/6``; exact linear systems ``10000/250000``; equilibrium support cells
``225/3969``; total marginal vertices per cell ``128/1024``; total marginal
vertices ``4096/100000``; evaluated vertex pairs ``65536/1000000``; rational
numerator and denominator bits ``256/1024`` each; verifier operations
``1000000/10000000``; output records ``50000/500000``; and output bytes
``4000000/32000000``.  Callers may lower defaults or raise them only to these
hard ceilings.  Before materializing supports, the solver checks
``(2**n1-1)*(2**n2-1)``.  Before active-set enumeration it sums
``C(q, variables-rank(equalities))`` for both marginal polytopes of every
support pair.

The scope is deliberately tiny and exact.  It contains no approximate solver,
CFR integration, tree builder, file workflow, pipeline, Hero optimization, or
top-level package export.  Every failure returns a null primary response;
partial cells, truncation, cap clamping, and fallback profiles are forbidden.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from typing import Any, Mapping, Sequence


CONTRACT_VERSION = "m30-three-player-exact-response-v1"
ALGORITHM_VERSION = "all-support-rational-polytope-enumeration-v1"
VERIFIER_VERSION = "independent-active-set-verifier-v1"

EXACT_CORRESPONDENCE_COMPLETE = "EXACT_CORRESPONDENCE_COMPLETE"
CAP_EXCEEDED = "CAP_EXCEEDED"
STALE_INPUT = "STALE_INPUT"
INVALID_INPUT = "INVALID_INPUT"
UNSUPPORTED_MODEL = "UNSUPPORTED_MODEL"
NUMERIC_FAILURE = "NUMERIC_FAILURE"
INTERNAL_FAILURE = "INTERNAL_FAILURE"

DEFAULT_MAX_PURE_PLANS_O1 = 4
DEFAULT_MAX_PURE_PLANS_O2 = 4
DEFAULT_MAX_JOINT_PURE_PROFILES = 16
DEFAULT_MAX_SUPPORT_PAIRS = 225
DEFAULT_MAX_SUPPORT_SIZE_O1 = 4
DEFAULT_MAX_SUPPORT_SIZE_O2 = 4
DEFAULT_MAX_EXACT_LINEAR_SYSTEMS = 10_000
DEFAULT_MAX_EQUILIBRIUM_SUPPORT_CELLS = 225
DEFAULT_MAX_VERTICES_PER_CELL = 128
DEFAULT_MAX_TOTAL_VERTICES = 4_096
DEFAULT_MAX_VERTEX_PAIRS_EVALUATED = 65_536
DEFAULT_MAX_RATIONAL_NUMERATOR_BITS = 256
DEFAULT_MAX_RATIONAL_DENOMINATOR_BITS = 256
DEFAULT_MAX_VERIFIER_OPERATIONS = 1_000_000
DEFAULT_MAX_OUTPUT_RECORDS = 50_000
DEFAULT_MAX_OUTPUT_BYTES = 4_000_000

HARD_MAX_PURE_PLANS_O1 = 6
HARD_MAX_PURE_PLANS_O2 = 6
HARD_MAX_JOINT_PURE_PROFILES = 36
HARD_MAX_SUPPORT_PAIRS = 3_969
HARD_MAX_SUPPORT_SIZE_O1 = 6
HARD_MAX_SUPPORT_SIZE_O2 = 6
HARD_MAX_EXACT_LINEAR_SYSTEMS = 250_000
HARD_MAX_EQUILIBRIUM_SUPPORT_CELLS = 3_969
HARD_MAX_VERTICES_PER_CELL = 1_024
HARD_MAX_TOTAL_VERTICES = 100_000
HARD_MAX_VERTEX_PAIRS_EVALUATED = 1_000_000
HARD_MAX_RATIONAL_NUMERATOR_BITS = 1_024
HARD_MAX_RATIONAL_DENOMINATOR_BITS = 1_024
HARD_MAX_VERIFIER_OPERATIONS = 10_000_000
HARD_MAX_OUTPUT_RECORDS = 500_000
HARD_MAX_OUTPUT_BYTES = 32_000_000

_IDENTITY_RE = re.compile(r"[0-9a-f]{64}")
_INTEGER_RE = re.compile(r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)")
_FRACTION_RE = re.compile(r"(-?[1-9][0-9]*)/([1-9][0-9]*)")
_PLAN_ID_RE = re.compile(r"[A-Za-z0-9_.:@/+~-]{1,128}")


@dataclass(frozen=True)
class ExactPayoffRow:
    """One canonical joint pure-profile payoff row."""

    o1_plan_id: str
    o2_plan_id: str
    H: str
    O1: str
    O2: str
    R: str


@dataclass(frozen=True)
class ExactReducedGame:
    """Strict exact reduced game and semantic run context."""

    o1_plan_ids: tuple[str, ...]
    o2_plan_ids: tuple[str, ...]
    payoff_rows: tuple[ExactPayoffRow, ...]
    response_game_structure_identity: str
    fixed_hero_identity: str
    perfect_recall_evidence_identity: str
    rake_convention_identity: str
    supplied_profile_identity: str | None = None
    candidate_identity: str | None = None
    search_mode_context_identity: str | None = None
    run_context_identity: str | None = None
    contract_version: str = CONTRACT_VERSION
    algorithm_version: str = ALGORITHM_VERSION


@dataclass(frozen=True)
class ResponseIdentityPins:
    """Optional caller pins whose mismatch is classified as stale input."""

    response_game_structure_identity: str | None = None
    fixed_hero_identity: str | None = None
    perfect_recall_evidence_identity: str | None = None
    rake_convention_identity: str | None = None
    payoff_table_identity: str | None = None
    supplied_profile_identity: str | None = None
    candidate_identity: str | None = None
    search_mode_context_identity: str | None = None
    run_context_identity: str | None = None
    config_identity: str | None = None
    response_game_identity: str | None = None
    response_run_identity: str | None = None


@dataclass(frozen=True)
class ExactResponseLimits:
    """Resource bounds for complete exact correspondence enumeration.

    Callers may lower the defaults or explicitly raise them only up to the
    immutable hard ceilings below.  A cap never changes algorithm semantics.
    """

    max_pure_plans_o1: int = DEFAULT_MAX_PURE_PLANS_O1
    max_pure_plans_o2: int = DEFAULT_MAX_PURE_PLANS_O2
    max_joint_pure_profiles: int = DEFAULT_MAX_JOINT_PURE_PROFILES
    max_support_pairs: int = DEFAULT_MAX_SUPPORT_PAIRS
    max_support_size_o1: int = DEFAULT_MAX_SUPPORT_SIZE_O1
    max_support_size_o2: int = DEFAULT_MAX_SUPPORT_SIZE_O2
    max_exact_linear_systems: int = DEFAULT_MAX_EXACT_LINEAR_SYSTEMS
    max_equilibrium_support_cells: int = (
        DEFAULT_MAX_EQUILIBRIUM_SUPPORT_CELLS
    )
    max_vertices_per_cell: int = DEFAULT_MAX_VERTICES_PER_CELL
    max_total_vertices: int = DEFAULT_MAX_TOTAL_VERTICES
    max_vertex_pairs_evaluated: int = DEFAULT_MAX_VERTEX_PAIRS_EVALUATED
    max_rational_numerator_bits: int = DEFAULT_MAX_RATIONAL_NUMERATOR_BITS
    max_rational_denominator_bits: int = DEFAULT_MAX_RATIONAL_DENOMINATOR_BITS
    max_verifier_operations: int = DEFAULT_MAX_VERIFIER_OPERATIONS
    max_output_records: int = DEFAULT_MAX_OUTPUT_RECORDS
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES


@dataclass(frozen=True)
class ExactResponseError:
    """Bounded failure metadata; it never carries partial response data."""

    phase: str
    message: str


@dataclass(frozen=True)
class ExactResponseResult:
    """Exclusive complete-success or fail-closed result wrapper."""

    status: str
    response: dict[str, Any] | None
    error: ExactResponseError | None
    partial_response: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly outer result."""

        return {
            "status": self.status,
            "response": self.response,
            "error": None if self.error is None else asdict(self.error),
            "partial_response": self.partial_response,
        }


@dataclass(frozen=True)
class _Utility:
    H: Fraction
    O1: Fraction
    O2: Fraction
    R: Fraction

    def for_name(self, name: str) -> Fraction:
        return getattr(self, name)


@dataclass
class _SupportCell:
    support_cell_id: str
    o1_vertices: tuple[tuple[Fraction, ...], ...]
    o2_vertices: tuple[tuple[Fraction, ...], ...]
    source_support_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]]
    o1_dimension: int
    o2_dimension: int


@dataclass
class _SolveCounters:
    linear_systems_preflight: int = 0
    linear_systems_solved: int = 0
    raw_nonempty_support_cells: int = 0
    exact_duplicate_support_cells: int = 0
    total_vertices: int = 0


class _ResponseFailure(ValueError):
    def __init__(self, status: str, phase: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.phase = phase


class _VerifierBudget:
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.used = 0

    def spend(self, count: int = 1) -> None:
        self.used += count
        if self.used > self.maximum:
            raise _ResponseFailure(
                CAP_EXCEEDED,
                "verifier",
                "max_verifier_operations exceeded",
            )


class _OutputRecordBudget:
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.used = 0

    def reserve(self, count: int = 1) -> None:
        if count < 0 or self.used > self.maximum - count:
            raise _ResponseFailure(
                CAP_EXCEEDED,
                "output",
                "max_output_records exceeded before record allocation",
            )
        self.used += count

    def release(self, count: int) -> None:
        if count < 0 or count > self.used:
            raise _ResponseFailure(
                INTERNAL_FAILURE,
                "output",
                "invalid output record budget release",
            )
        self.used -= count


_HARD_LIMITS = {
    "max_pure_plans_o1": HARD_MAX_PURE_PLANS_O1,
    "max_pure_plans_o2": HARD_MAX_PURE_PLANS_O2,
    "max_joint_pure_profiles": HARD_MAX_JOINT_PURE_PROFILES,
    "max_support_pairs": HARD_MAX_SUPPORT_PAIRS,
    "max_support_size_o1": HARD_MAX_SUPPORT_SIZE_O1,
    "max_support_size_o2": HARD_MAX_SUPPORT_SIZE_O2,
    "max_exact_linear_systems": HARD_MAX_EXACT_LINEAR_SYSTEMS,
    "max_equilibrium_support_cells": HARD_MAX_EQUILIBRIUM_SUPPORT_CELLS,
    "max_vertices_per_cell": HARD_MAX_VERTICES_PER_CELL,
    "max_total_vertices": HARD_MAX_TOTAL_VERTICES,
    "max_vertex_pairs_evaluated": HARD_MAX_VERTEX_PAIRS_EVALUATED,
    "max_rational_numerator_bits": HARD_MAX_RATIONAL_NUMERATOR_BITS,
    "max_rational_denominator_bits": HARD_MAX_RATIONAL_DENOMINATOR_BITS,
    "max_verifier_operations": HARD_MAX_VERIFIER_OPERATIONS,
    "max_output_records": HARD_MAX_OUTPUT_RECORDS,
    "max_output_bytes": HARD_MAX_OUTPUT_BYTES,
}


def _clean_text(value: str, maximum: int) -> str:
    return value.replace("\r", " ").replace("\n", " ")[:maximum]


def _failure(exc: _ResponseFailure) -> ExactResponseResult:
    return ExactResponseResult(
        status=exc.status,
        response=None,
        error=ExactResponseError(
            phase=_clean_text(exc.phase, 64),
            message=_clean_text(str(exc), 500),
        ),
        partial_response=False,
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _identity(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _validate_limits(limits: ExactResponseLimits) -> None:
    if type(limits) is not ExactResponseLimits:
        raise _ResponseFailure(
            INVALID_INPUT, "limits", "limits must be ExactResponseLimits"
        )
    for field in fields(ExactResponseLimits):
        name = field.name
        value = getattr(limits, name)
        hard = _HARD_LIMITS[name]
        if type(value) is not int or value < 1 or value > hard:
            raise _ResponseFailure(
                INVALID_INPUT,
                "limits",
                f"{name} must be an int in [1,{hard}]",
            )


def _validate_identity(value: Any, phase: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if type(value) is not str or _IDENTITY_RE.fullmatch(value) is None:
        raise _ResponseFailure(
            INVALID_INPUT,
            phase,
            "identity must be a lowercase 64-character SHA-256 string",
        )


def _validate_plan_ids(
    value: Any, phase: str, maximum: int
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise _ResponseFailure(INVALID_INPUT, phase, "plan IDs must be a tuple")
    if not value:
        raise _ResponseFailure(INVALID_INPUT, phase, "plan set must be nonempty")
    if len(value) > maximum:
        raise _ResponseFailure(CAP_EXCEEDED, phase, "pure-plan cap exceeded")
    seen: set[str] = set()
    for plan_id in value:
        if type(plan_id) is not str or _PLAN_ID_RE.fullmatch(plan_id) is None:
            raise _ResponseFailure(
                INVALID_INPUT,
                phase,
                "plan ID must use the bounded canonical plan-ID grammar",
            )
        if plan_id in seen:
            raise _ResponseFailure(INVALID_INPUT, phase, "duplicate plan ID")
        seen.add(plan_id)
    return value


def _check_fraction_bits(
    value: Fraction, limits: ExactResponseLimits, phase: str
) -> Fraction:
    if abs(value.numerator).bit_length() > limits.max_rational_numerator_bits:
        raise _ResponseFailure(
            CAP_EXCEEDED, phase, "max_rational_numerator_bits exceeded"
        )
    if value.denominator.bit_length() > limits.max_rational_denominator_bits:
        raise _ResponseFailure(
            CAP_EXCEEDED, phase, "max_rational_denominator_bits exceeded"
        )
    return value


def _parse_rational(
    value: Any, phase: str, limits: ExactResponseLimits
) -> Fraction:
    if type(value) is not str:
        raise _ResponseFailure(
            INVALID_INPUT, phase, "utility must be a canonical rational string"
        )
    if _INTEGER_RE.fullmatch(value) is not None:
        numerator_text = value
        denominator_text = "1"
    else:
        match = _FRACTION_RE.fullmatch(value)
        if match is None:
            raise _ResponseFailure(
                INVALID_INPUT, phase, "noncanonical rational string"
            )
        numerator_text, denominator_text = match.groups()
    max_num_digits = (
        math.ceil(limits.max_rational_numerator_bits * math.log10(2)) + 1
    )
    max_den_digits = (
        math.ceil(limits.max_rational_denominator_bits * math.log10(2)) + 1
    )
    if len(numerator_text.lstrip("-")) > max_num_digits:
        raise _ResponseFailure(
            CAP_EXCEEDED, phase, "max_rational_numerator_bits exceeded"
        )
    if len(denominator_text) > max_den_digits:
        raise _ResponseFailure(
            CAP_EXCEEDED, phase, "max_rational_denominator_bits exceeded"
        )
    numerator = int(numerator_text)
    denominator = int(denominator_text)
    if denominator == 0:
        raise _ResponseFailure(INVALID_INPUT, phase, "zero denominator")
    if denominator != 1:
        if numerator == 0 or math.gcd(abs(numerator), denominator) != 1:
            raise _ResponseFailure(
                INVALID_INPUT, phase, "fraction must be nonzero and reduced"
            )
    parsed = Fraction(numerator, denominator)
    if _rational_text(parsed) != value:
        raise _ResponseFailure(
            INVALID_INPUT, phase, "rational string is not canonical"
        )
    return _check_fraction_bits(parsed, limits, phase)


def _rational_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _fadd(
    left: Fraction,
    right: Fraction,
    limits: ExactResponseLimits,
    phase: str,
) -> Fraction:
    return _check_fraction_bits(left + right, limits, phase)


def _fsub(
    left: Fraction,
    right: Fraction,
    limits: ExactResponseLimits,
    phase: str,
) -> Fraction:
    return _check_fraction_bits(left - right, limits, phase)


def _fmul(
    left: Fraction,
    right: Fraction,
    limits: ExactResponseLimits,
    phase: str,
) -> Fraction:
    return _check_fraction_bits(left * right, limits, phase)


def _fdiv(
    numerator: Fraction,
    denominator: Fraction,
    limits: ExactResponseLimits,
    phase: str,
) -> Fraction:
    if denominator == 0:
        raise _ResponseFailure(NUMERIC_FAILURE, phase, "zero pivot")
    return _check_fraction_bits(numerator / denominator, limits, phase)


def _checked_product(
    left: int, right: int, maximum: int, phase: str
) -> int:
    result = left * right
    if result > maximum:
        raise _ResponseFailure(CAP_EXCEEDED, phase, "checked product cap exceeded")
    return result


def _payoff_table(
    game: ExactReducedGame,
    o1_plan_ids: tuple[str, ...],
    o2_plan_ids: tuple[str, ...],
    limits: ExactResponseLimits,
) -> tuple[tuple[tuple[_Utility, ...], ...], str]:
    if type(game.payoff_rows) is not tuple:
        raise _ResponseFailure(
            INVALID_INPUT, "payoff_rows", "payoff rows must be a tuple"
        )
    expected = len(o1_plan_ids) * len(o2_plan_ids)
    if len(game.payoff_rows) != expected:
        raise _ResponseFailure(
            INVALID_INPUT, "payoff_rows", "payoff table is not rectangular and complete"
        )
    o1_index = {plan_id: index for index, plan_id in enumerate(o1_plan_ids)}
    o2_index = {plan_id: index for index, plan_id in enumerate(o2_plan_ids)}
    parsed: dict[tuple[int, int], _Utility] = {}
    canonical_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(game.payoff_rows):
        phase = f"payoff_rows[{row_index}]"
        if type(row) is not ExactPayoffRow:
            raise _ResponseFailure(
                INVALID_INPUT, phase, "row must be ExactPayoffRow"
            )
        if row.o1_plan_id not in o1_index or row.o2_plan_id not in o2_index:
            raise _ResponseFailure(
                INVALID_INPUT, phase, "payoff row references an unknown plan ID"
            )
        key = (o1_index[row.o1_plan_id], o2_index[row.o2_plan_id])
        if key in parsed:
            raise _ResponseFailure(INVALID_INPUT, phase, "duplicate payoff row")
        utility = _Utility(
            H=_parse_rational(row.H, f"{phase}.H", limits),
            O1=_parse_rational(row.O1, f"{phase}.O1", limits),
            O2=_parse_rational(row.O2, f"{phase}.O2", limits),
            R=_parse_rational(row.R, f"{phase}.R", limits),
        )
        conservation = Fraction(0)
        for name in ("H", "O1", "O2", "R"):
            conservation = _fadd(
                conservation,
                utility.for_name(name),
                limits,
                f"{phase}.conservation",
            )
        if conservation != 0:
            raise _ResponseFailure(
                INVALID_INPUT, phase, "exact payoff conservation mismatch"
            )
        parsed[key] = utility
    missing = [
        (i, j)
        for i in range(len(o1_plan_ids))
        for j in range(len(o2_plan_ids))
        if (i, j) not in parsed
    ]
    if missing:
        raise _ResponseFailure(INVALID_INPUT, "payoff_rows", "missing payoff row")
    matrix: list[tuple[_Utility, ...]] = []
    for i, o1_plan_id in enumerate(o1_plan_ids):
        matrix_row: list[_Utility] = []
        for j, o2_plan_id in enumerate(o2_plan_ids):
            utility = parsed[(i, j)]
            matrix_row.append(utility)
            canonical_rows.append(
                {
                    "o1_plan_id": o1_plan_id,
                    "o2_plan_id": o2_plan_id,
                    "utility": {
                        name: _rational_text(utility.for_name(name))
                        for name in ("H", "O1", "O2", "R")
                    },
                }
            )
        matrix.append(tuple(matrix_row))
    payoff_identity = _identity(
        {
            "o1_plan_ids": list(o1_plan_ids),
            "o2_plan_ids": list(o2_plan_ids),
            "rows": canonical_rows,
        }
    )
    return tuple(matrix), payoff_identity


def _support_subsets(size: int) -> list[tuple[int, ...]]:
    return [
        support
        for cardinality in range(1, size + 1)
        for support in itertools.combinations(range(size), cardinality)
    ]


def _solver_rref(
    rows: Sequence[tuple[tuple[Fraction, ...], Fraction]],
    variable_count: int,
    limits: ExactResponseLimits,
) -> tuple[list[list[Fraction]], list[int], bool]:
    matrix = [list(coefficients) + [rhs] for coefficients, rhs in rows]
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(variable_count):
        selected = next(
            (
                row
                for row in range(pivot_row, len(matrix))
                if matrix[row][column] != 0
            ),
            None,
        )
        if selected is None:
            continue
        matrix[pivot_row], matrix[selected] = matrix[selected], matrix[pivot_row]
        pivot = matrix[pivot_row][column]
        matrix[pivot_row] = [
            _fdiv(
                value,
                pivot,
                limits,
                "solver.linear_system",
            )
            for value in matrix[pivot_row]
        ]
        for row in range(len(matrix)):
            if row == pivot_row or matrix[row][column] == 0:
                continue
            factor = matrix[row][column]
            matrix[row] = [
                _fsub(
                    matrix[row][index],
                    _fmul(
                        factor,
                        matrix[pivot_row][index],
                        limits,
                        "solver.linear_system",
                    ),
                    limits,
                    "solver.linear_system",
                )
                for index in range(variable_count + 1)
            ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(matrix):
            break
    inconsistent = any(
        all(row[column] == 0 for column in range(variable_count))
        and row[-1] != 0
        for row in matrix
    )
    return matrix, pivot_columns, inconsistent


def _solver_equality_rank(
    equalities: Sequence[tuple[tuple[Fraction, ...], Fraction]],
    variable_count: int,
    limits: ExactResponseLimits,
) -> tuple[int, bool]:
    _, pivots, inconsistent = _solver_rref(
        equalities, variable_count, limits
    )
    return len(pivots), inconsistent


def _solver_unique_solution(
    equations: Sequence[tuple[tuple[Fraction, ...], Fraction]],
    variable_count: int,
    limits: ExactResponseLimits,
) -> tuple[Fraction, ...] | None:
    matrix, pivots, inconsistent = _solver_rref(
        equations, variable_count, limits
    )
    if inconsistent or len(pivots) != variable_count:
        return None
    result = [Fraction(0) for _ in range(variable_count)]
    for row_index, column in enumerate(pivots):
        result[column] = _check_fraction_bits(
            matrix[row_index][-1], limits, "solver.solution"
        )
    return tuple(result)


def _solver_constraints(
    table: tuple[tuple[_Utility, ...], ...],
    o1_support: tuple[int, ...],
    o2_support: tuple[int, ...],
    *,
    mixture_player: str,
    limits: ExactResponseLimits,
) -> tuple[
    list[tuple[tuple[Fraction, ...], Fraction]],
    list[tuple[tuple[Fraction, ...], Fraction]],
]:
    if mixture_player == "O1":
        variable_indices = o1_support
        best_indices = o2_support
        all_best_count = len(table[0])
        payoff_name = "O2"
        reference = best_indices[0]

        def payoff(variable: int, best: int) -> Fraction:
            return table[variable][best].for_name(payoff_name)

    else:
        variable_indices = o2_support
        best_indices = o1_support
        all_best_count = len(table)
        payoff_name = "O1"
        reference = best_indices[0]

        def payoff(variable: int, best: int) -> Fraction:
            return table[best][variable].for_name(payoff_name)

    variable_count = len(variable_indices)
    equalities: list[tuple[tuple[Fraction, ...], Fraction]] = [
        (tuple(Fraction(1) for _ in variable_indices), Fraction(1))
    ]
    for best in best_indices[1:]:
        equalities.append(
            (
                tuple(
                    _fsub(
                        payoff(variable, best),
                        payoff(variable, reference),
                        limits,
                        "solver.constraints",
                    )
                    for variable in variable_indices
                ),
                Fraction(0),
            )
        )
    inequalities: list[tuple[tuple[Fraction, ...], Fraction]] = []
    for local_index in range(variable_count):
        coefficients = [
            Fraction(-1) if index == local_index else Fraction(0)
            for index in range(variable_count)
        ]
        inequalities.append((tuple(coefficients), Fraction(0)))
    best_set = set(best_indices)
    for best in range(all_best_count):
        if best in best_set:
            continue
        inequalities.append(
            (
                tuple(
                    _fsub(
                        payoff(variable, best),
                        payoff(variable, reference),
                        limits,
                        "solver.constraints",
                    )
                    for variable in variable_indices
                ),
                Fraction(0),
            )
        )
    return equalities, inequalities


def _dot(
    coefficients: Sequence[Fraction],
    point: Sequence[Fraction],
    limits: ExactResponseLimits,
    phase: str,
) -> Fraction:
    total = Fraction(0)
    for coefficient, value in zip(coefficients, point):
        total = _fadd(
            total,
            _fmul(coefficient, value, limits, phase),
            limits,
            phase,
        )
    return total


def _solver_vertices(
    equalities: Sequence[tuple[tuple[Fraction, ...], Fraction]],
    inequalities: Sequence[tuple[tuple[Fraction, ...], Fraction]],
    variable_count: int,
    limits: ExactResponseLimits,
    counters: _SolveCounters,
) -> tuple[tuple[Fraction, ...], ...]:
    rank, inconsistent = _solver_equality_rank(
        equalities, variable_count, limits
    )
    if inconsistent:
        return ()
    needed = variable_count - rank
    if needed < 0 or needed > len(inequalities):
        return ()
    vertices: set[tuple[Fraction, ...]] = set()
    for active_indices in itertools.combinations(range(len(inequalities)), needed):
        counters.linear_systems_solved += 1
        if counters.linear_systems_solved > limits.max_exact_linear_systems:
            raise _ResponseFailure(
                CAP_EXCEEDED,
                "solver",
                "max_exact_linear_systems exceeded",
            )
        equations = list(equalities) + [
            inequalities[index] for index in active_indices
        ]
        solution = _solver_unique_solution(equations, variable_count, limits)
        if solution is None:
            continue
        if any(
            _dot(coefficients, solution, limits, "solver.verify_vertex") != rhs
            for coefficients, rhs in equalities
        ):
            continue
        if any(
            _dot(coefficients, solution, limits, "solver.verify_vertex") > rhs
            for coefficients, rhs in inequalities
        ):
            continue
        if solution not in vertices:
            if len(vertices) + 1 > limits.max_vertices_per_cell:
                raise _ResponseFailure(
                    CAP_EXCEEDED,
                    "solver",
                    "max_vertices_per_cell exceeded before vertex allocation",
                )
            vertices.add(solution)
    return tuple(sorted(vertices))


def _matrix_rank(
    rows: Sequence[Sequence[Fraction]],
    variable_count: int,
    limits: ExactResponseLimits,
) -> int:
    equations = [
        (tuple(row), Fraction(0))
        for row in rows
    ]
    _, pivots, _ = _solver_rref(equations, variable_count, limits)
    return len(pivots)


def _polytope_dimension(
    vertices: tuple[tuple[Fraction, ...], ...],
    limits: ExactResponseLimits,
) -> int:
    if len(vertices) <= 1:
        return 0
    origin = vertices[0]
    differences = [
        [
            _fsub(value, origin[index], limits, "solver.dimension")
            for index, value in enumerate(vertex)
        ]
        for vertex in vertices[1:]
    ]
    return _matrix_rank(differences, len(origin), limits)


def _expand_vertex(
    vertex: tuple[Fraction, ...],
    support: tuple[int, ...],
    full_size: int,
) -> tuple[Fraction, ...]:
    result = [Fraction(0) for _ in range(full_size)]
    for local_index, global_index in enumerate(support):
        result[global_index] = vertex[local_index]
    return tuple(result)


def _preflight_linear_systems(
    table: tuple[tuple[_Utility, ...], ...],
    o1_supports: Sequence[tuple[int, ...]],
    o2_supports: Sequence[tuple[int, ...]],
    limits: ExactResponseLimits,
) -> int:
    total = 0
    for o1_support in o1_supports:
        for o2_support in o2_supports:
            for mixture_player in ("O1", "O2"):
                equalities, inequalities = _solver_constraints(
                    table,
                    o1_support,
                    o2_support,
                    mixture_player=mixture_player,
                    limits=limits,
                )
                variable_count = (
                    len(o1_support) if mixture_player == "O1" else len(o2_support)
                )
                rank, inconsistent = _solver_equality_rank(
                    equalities, variable_count, limits
                )
                count = (
                    0
                    if inconsistent or variable_count - rank > len(inequalities)
                    else math.comb(len(inequalities), variable_count - rank)
                )
                total += count
                if total > limits.max_exact_linear_systems:
                    raise _ResponseFailure(
                        CAP_EXCEEDED,
                        "solver_preflight",
                        "max_exact_linear_systems exceeded before enumeration",
                    )
    return total


def _cell_identity(
    o1_vertices: tuple[tuple[Fraction, ...], ...],
    o2_vertices: tuple[tuple[Fraction, ...], ...],
) -> str:
    return _identity(
        {
            "representation": "exact-rational-support-cell-v1",
            "o1_vertices": [
                [_rational_text(value) for value in vertex]
                for vertex in o1_vertices
            ],
            "o2_vertices": [
                [_rational_text(value) for value in vertex]
                for vertex in o2_vertices
            ],
        }
    )


def _enumerate_support_cells(
    table: tuple[tuple[_Utility, ...], ...],
    o1_supports: Sequence[tuple[int, ...]],
    o2_supports: Sequence[tuple[int, ...]],
    limits: ExactResponseLimits,
) -> tuple[list[_SupportCell], list[dict[str, Any]], _SolveCounters]:
    counters = _SolveCounters()
    counters.linear_systems_preflight = _preflight_linear_systems(
        table, o1_supports, o2_supports, limits
    )
    cells_by_key: dict[
        tuple[
            tuple[tuple[Fraction, ...], ...],
            tuple[tuple[Fraction, ...], ...],
        ],
        _SupportCell,
    ] = {}
    audit: list[dict[str, Any]] = []
    for o1_support in o1_supports:
        for o2_support in o2_supports:
            eq1, ineq1 = _solver_constraints(
                table,
                o1_support,
                o2_support,
                mixture_player="O1",
                limits=limits,
            )
            local_o1 = _solver_vertices(
                eq1, ineq1, len(o1_support), limits, counters
            )
            pair_key = (o1_support, o2_support)
            if not local_o1:
                audit.append(
                    {
                        "support_pair": pair_key,
                        "outcome": "empty_o1_mixture_polytope",
                        "support_cell_id": None,
                    }
                )
                continue
            eq2, ineq2 = _solver_constraints(
                table,
                o1_support,
                o2_support,
                mixture_player="O2",
                limits=limits,
            )
            local_o2 = _solver_vertices(
                eq2, ineq2, len(o2_support), limits, counters
            )
            if not local_o2:
                audit.append(
                    {
                        "support_pair": pair_key,
                        "outcome": "empty_o2_mixture_polytope",
                        "support_cell_id": None,
                    }
                )
                continue
            o1_vertices = tuple(
                sorted(
                    _expand_vertex(vertex, o1_support, len(table))
                    for vertex in local_o1
                )
            )
            o2_vertices = tuple(
                sorted(
                    _expand_vertex(vertex, o2_support, len(table[0]))
                    for vertex in local_o2
                )
            )
            if len(o1_vertices) + len(o2_vertices) > limits.max_vertices_per_cell:
                raise _ResponseFailure(
                    CAP_EXCEEDED,
                    "solver",
                    "max_vertices_per_cell exceeded",
                )
            counters.raw_nonempty_support_cells += 1
            key = (o1_vertices, o2_vertices)
            existing = cells_by_key.get(key)
            if existing is not None:
                existing.source_support_pairs.append(pair_key)
                counters.exact_duplicate_support_cells += 1
                audit.append(
                    {
                        "support_pair": pair_key,
                        "outcome": "exact_duplicate_support_cell",
                        "support_cell_id": existing.support_cell_id,
                    }
                )
                continue
            if len(cells_by_key) + 1 > limits.max_equilibrium_support_cells:
                raise _ResponseFailure(
                    CAP_EXCEEDED,
                    "solver",
                    "max_equilibrium_support_cells exceeded",
                )
            added_vertices = len(o1_vertices) + len(o2_vertices)
            if counters.total_vertices + added_vertices > limits.max_total_vertices:
                raise _ResponseFailure(
                    CAP_EXCEEDED, "solver", "max_total_vertices exceeded"
                )
            counters.total_vertices += added_vertices
            support_cell_id = _cell_identity(o1_vertices, o2_vertices)
            cell = _SupportCell(
                support_cell_id=support_cell_id,
                o1_vertices=o1_vertices,
                o2_vertices=o2_vertices,
                source_support_pairs=[pair_key],
                o1_dimension=_polytope_dimension(o1_vertices, limits),
                o2_dimension=_polytope_dimension(o2_vertices, limits),
            )
            cells_by_key[key] = cell
            audit.append(
                {
                    "support_pair": pair_key,
                    "outcome": "new_support_cell",
                    "support_cell_id": support_cell_id,
                }
            )
    cells = sorted(
        cells_by_key.values(),
        key=lambda cell: (
            min(
                (
                    len(o1_support) + len(o2_support),
                    o1_support,
                    o2_support,
                )
                for o1_support, o2_support in cell.source_support_pairs
            ),
            cell.support_cell_id,
        ),
    )
    for cell in cells:
        cell.source_support_pairs.sort(
            key=lambda pair: (len(pair[0]) + len(pair[1]), pair[0], pair[1])
        )
    return cells, audit, counters


def _v_checked(
    value: Fraction,
    limits: ExactResponseLimits,
    budget: _VerifierBudget,
) -> Fraction:
    budget.spend()
    return _check_fraction_bits(value, limits, "verifier")


def _v_sub(
    left: Fraction,
    right: Fraction,
    limits: ExactResponseLimits,
    budget: _VerifierBudget,
) -> Fraction:
    return _v_checked(left - right, limits, budget)


def _v_mul(
    left: Fraction,
    right: Fraction,
    limits: ExactResponseLimits,
    budget: _VerifierBudget,
) -> Fraction:
    return _v_checked(left * right, limits, budget)


def _v_div(
    numerator: Fraction,
    denominator: Fraction,
    limits: ExactResponseLimits,
    budget: _VerifierBudget,
) -> Fraction:
    budget.spend()
    if denominator == 0:
        raise _ResponseFailure(NUMERIC_FAILURE, "verifier", "zero verifier pivot")
    return _check_fraction_bits(numerator / denominator, limits, "verifier")


def _verifier_constraints(
    table: tuple[tuple[_Utility, ...], ...],
    o1_support: tuple[int, ...],
    o2_support: tuple[int, ...],
    mixture_player: str,
    limits: ExactResponseLimits,
    budget: _VerifierBudget,
) -> tuple[
    list[tuple[tuple[Fraction, ...], Fraction]],
    list[tuple[tuple[Fraction, ...], Fraction]],
]:
    """Reconstruct response constraints without calling solver helpers."""

    if mixture_player == "O1":
        variables = o1_support
        selected_responses = o2_support
        response_count = len(table[0])
        reference = selected_responses[0]

        def raw_payoff(variable: int, response: int) -> Fraction:
            budget.spend()
            return table[variable][response].O2

    else:
        variables = o2_support
        selected_responses = o1_support
        response_count = len(table)
        reference = selected_responses[0]

        def raw_payoff(variable: int, response: int) -> Fraction:
            budget.spend()
            return table[response][variable].O1

    equalities: list[tuple[tuple[Fraction, ...], Fraction]] = [
        (tuple(Fraction(1) for _ in variables), Fraction(1))
    ]
    for response in selected_responses[1:]:
        coefficients = []
        for variable in variables:
            coefficients.append(
                _v_sub(
                    raw_payoff(variable, response),
                    raw_payoff(variable, reference),
                    limits,
                    budget,
                )
            )
        equalities.append((tuple(coefficients), Fraction(0)))

    inequalities: list[tuple[tuple[Fraction, ...], Fraction]] = []
    for local_index in range(len(variables)):
        inequalities.append(
            (
                tuple(
                    Fraction(-1) if index == local_index else Fraction(0)
                    for index in range(len(variables))
                ),
                Fraction(0),
            )
        )
    selected_set = frozenset(selected_responses)
    for response in range(response_count):
        if response in selected_set:
            continue
        coefficients = []
        for variable in variables:
            coefficients.append(
                _v_sub(
                    raw_payoff(variable, response),
                    raw_payoff(variable, reference),
                    limits,
                    budget,
                )
            )
        inequalities.append((tuple(coefficients), Fraction(0)))
    return equalities, inequalities


def _verifier_reduce(
    equations: Sequence[tuple[tuple[Fraction, ...], Fraction]],
    variable_count: int,
    limits: ExactResponseLimits,
    budget: _VerifierBudget,
) -> tuple[list[list[Fraction]], list[tuple[int, int]], bool]:
    """Independent Gauss-Jordan path used only by the verifier."""

    work = [list(coefficients) + [rhs] for coefficients, rhs in equations]
    pivots: list[tuple[int, int]] = []
    row_cursor = 0
    column_cursor = 0
    while row_cursor < len(work) and column_cursor < variable_count:
        budget.spend()
        selected = None
        for candidate in range(row_cursor, len(work)):
            budget.spend()
            if work[candidate][column_cursor] != 0:
                selected = candidate
                break
        if selected is None:
            column_cursor += 1
            continue
        work[row_cursor], work[selected] = work[selected], work[row_cursor]
        divisor = work[row_cursor][column_cursor]
        for column in range(column_cursor, variable_count + 1):
            work[row_cursor][column] = _v_div(
                work[row_cursor][column], divisor, limits, budget
            )
        for other in range(len(work)):
            if other == row_cursor:
                continue
            multiplier = work[other][column_cursor]
            budget.spend()
            if multiplier == 0:
                continue
            for column in range(column_cursor, variable_count + 1):
                work[other][column] = _v_sub(
                    work[other][column],
                    _v_mul(
                        multiplier,
                        work[row_cursor][column],
                        limits,
                        budget,
                    ),
                    limits,
                    budget,
                )
        pivots.append((row_cursor, column_cursor))
        row_cursor += 1
        column_cursor += 1
    inconsistent = False
    for row in work:
        budget.spend(variable_count + 1)
        if all(value == 0 for value in row[:variable_count]) and row[-1] != 0:
            inconsistent = True
            break
    return work, pivots, inconsistent


def _verifier_rank(
    equalities: Sequence[tuple[tuple[Fraction, ...], Fraction]],
    variable_count: int,
    limits: ExactResponseLimits,
    budget: _VerifierBudget,
) -> tuple[int, bool]:
    _, pivots, inconsistent = _verifier_reduce(
        equalities, variable_count, limits, budget
    )
    return len(pivots), inconsistent


def _verifier_unique(
    equations: Sequence[tuple[tuple[Fraction, ...], Fraction]],
    variable_count: int,
    limits: ExactResponseLimits,
    budget: _VerifierBudget,
) -> tuple[Fraction, ...] | None:
    reduced, pivots, inconsistent = _verifier_reduce(
        equations, variable_count, limits, budget
    )
    if inconsistent or len(pivots) != variable_count:
        return None
    result = [Fraction(0) for _ in range(variable_count)]
    for row, column in pivots:
        result[column] = _v_checked(reduced[row][-1], limits, budget)
    return tuple(result)


def _verifier_dot(
    coefficients: Sequence[Fraction],
    point: Sequence[Fraction],
    limits: ExactResponseLimits,
    budget: _VerifierBudget,
) -> Fraction:
    total = Fraction(0)
    for coefficient, value in zip(coefficients, point):
        total = _v_checked(
            total + _v_mul(coefficient, value, limits, budget),
            limits,
            budget,
        )
    return total


def _verifier_vertices(
    equalities: Sequence[tuple[tuple[Fraction, ...], Fraction]],
    inequalities: Sequence[tuple[tuple[Fraction, ...], Fraction]],
    variable_count: int,
    limits: ExactResponseLimits,
    budget: _VerifierBudget,
) -> tuple[tuple[Fraction, ...], ...]:
    rank, inconsistent = _verifier_rank(
        equalities, variable_count, limits, budget
    )
    if inconsistent:
        return ()
    active_count = variable_count - rank
    if active_count < 0 or active_count > len(inequalities):
        return ()
    vertices: set[tuple[Fraction, ...]] = set()
    for active in itertools.combinations(range(len(inequalities)), active_count):
        budget.spend()
        equations = list(equalities)
        equations.extend(inequalities[index] for index in active)
        point = _verifier_unique(
            equations, variable_count, limits, budget
        )
        if point is None:
            continue
        valid = True
        for coefficients, rhs in equalities:
            if _verifier_dot(
                coefficients, point, limits, budget
            ) != rhs:
                valid = False
                break
        if not valid:
            continue
        for coefficients, rhs in inequalities:
            if _verifier_dot(
                coefficients, point, limits, budget
            ) > rhs:
                valid = False
                break
        if valid and point not in vertices:
            if len(vertices) + 1 > limits.max_vertices_per_cell:
                raise _ResponseFailure(
                    CAP_EXCEEDED,
                    "verifier",
                    "max_vertices_per_cell exceeded before vertex allocation",
                )
            vertices.add(point)
    return tuple(sorted(vertices))


def _verifier_utility_at(
    table: tuple[tuple[_Utility, ...], ...],
    o1_mixture: tuple[Fraction, ...],
    o2_mixture: tuple[Fraction, ...],
    limits: ExactResponseLimits,
    budget: _VerifierBudget,
) -> _Utility:
    """Recompute all utilities without the solver/extrema evaluator."""

    totals = {name: Fraction(0) for name in ("H", "O1", "O2", "R")}
    for i, x_probability in enumerate(o1_mixture):
        for j, y_probability in enumerate(o2_mixture):
            weight = _v_mul(
                x_probability, y_probability, limits, budget
            )
            for name in ("H", "O1", "O2", "R"):
                weighted_payoff = _v_mul(
                    weight,
                    table[i][j].for_name(name),
                    limits,
                    budget,
                )
                totals[name] = _v_checked(
                    totals[name] + weighted_payoff, limits, budget
                )
    return _Utility(**totals)


def _verifier_residuals(
    table: tuple[tuple[_Utility, ...], ...],
    o1_mixture: tuple[Fraction, ...],
    o2_mixture: tuple[Fraction, ...],
    utility: _Utility,
    limits: ExactResponseLimits,
    budget: _VerifierBudget,
) -> dict[str, Fraction]:
    """Recompute unilateral residuals directly from the input payoff table."""

    o1_values: list[Fraction] = []
    for i in range(len(table)):
        value = Fraction(0)
        for j, probability in enumerate(o2_mixture):
            product = _v_mul(
                probability, table[i][j].O1, limits, budget
            )
            value = _v_checked(value + product, limits, budget)
        o1_values.append(value)
    o2_values: list[Fraction] = []
    for j in range(len(table[0])):
        value = Fraction(0)
        for i, probability in enumerate(o1_mixture):
            product = _v_mul(
                probability, table[i][j].O2, limits, budget
            )
            value = _v_checked(value + product, limits, budget)
        o2_values.append(value)
    budget.spend(len(o1_values) + len(o2_values))
    o1_gain = _v_sub(max(o1_values), utility.O1, limits, budget)
    o2_gain = _v_sub(max(o2_values), utility.O2, limits, budget)
    return {
        "O1": max(Fraction(0), o1_gain),
        "O2": max(Fraction(0), o2_gain),
    }


def _utility_at(
    table: tuple[tuple[_Utility, ...], ...],
    o1_mixture: tuple[Fraction, ...],
    o2_mixture: tuple[Fraction, ...],
    limits: ExactResponseLimits,
) -> _Utility:
    totals = {name: Fraction(0) for name in ("H", "O1", "O2", "R")}
    for i, x_probability in enumerate(o1_mixture):
        if x_probability == 0:
            continue
        for j, y_probability in enumerate(o2_mixture):
            if y_probability == 0:
                continue
            weight = _fmul(
                x_probability,
                y_probability,
                limits,
                "utility.weight",
            )
            for name in ("H", "O1", "O2", "R"):
                totals[name] = _fadd(
                    totals[name],
                    _fmul(
                        weight,
                        table[i][j].for_name(name),
                        limits,
                        f"utility.{name}",
                    ),
                    limits,
                    f"utility.{name}",
                )
    return _Utility(**totals)


def _unilateral_residuals(
    table: tuple[tuple[_Utility, ...], ...],
    o1_mixture: tuple[Fraction, ...],
    o2_mixture: tuple[Fraction, ...],
    utility: _Utility,
    limits: ExactResponseLimits,
) -> dict[str, Fraction]:
    o1_values: list[Fraction] = []
    for i in range(len(table)):
        value = Fraction(0)
        for j, probability in enumerate(o2_mixture):
            value = _fadd(
                value,
                _fmul(
                    probability,
                    table[i][j].O1,
                    limits,
                    "residual.O1",
                ),
                limits,
                "residual.O1",
            )
        o1_values.append(value)
    o2_values: list[Fraction] = []
    for j in range(len(table[0])):
        value = Fraction(0)
        for i, probability in enumerate(o1_mixture):
            value = _fadd(
                value,
                _fmul(
                    probability,
                    table[i][j].O2,
                    limits,
                    "residual.O2",
                ),
                limits,
                "residual.O2",
            )
        o2_values.append(value)
    return {
        "O1": max(
            Fraction(0),
            _fsub(max(o1_values), utility.O1, limits, "residual.O1"),
        ),
        "O2": max(
            Fraction(0),
            _fsub(max(o2_values), utility.O2, limits, "residual.O2"),
        ),
    }


def _verify_correspondence(
    table: tuple[tuple[_Utility, ...], ...],
    o1_supports: Sequence[tuple[int, ...]],
    o2_supports: Sequence[tuple[int, ...]],
    cells: Sequence[_SupportCell],
    audit: Sequence[dict[str, Any]],
    limits: ExactResponseLimits,
) -> dict[str, Any]:
    """Independently reconstruct and verify the complete support-cell union."""

    budget = _VerifierBudget(limits.max_verifier_operations)
    expected_sources: dict[
        tuple[
            tuple[tuple[Fraction, ...], ...],
            tuple[tuple[Fraction, ...], ...],
        ],
        list[tuple[tuple[int, ...], tuple[int, ...]]],
    ] = {}
    expected_outcome_by_pair: dict[
        tuple[tuple[int, ...], tuple[int, ...]], str
    ] = {}
    for o1_support in o1_supports:
        for o2_support in o2_supports:
            equalities1, inequalities1 = _verifier_constraints(
                table,
                o1_support,
                o2_support,
                "O1",
                limits,
                budget,
            )
            local_o1 = _verifier_vertices(
                equalities1,
                inequalities1,
                len(o1_support),
                limits,
                budget,
            )
            pair = (o1_support, o2_support)
            if not local_o1:
                expected_outcome_by_pair[pair] = "empty_o1_mixture_polytope"
                continue
            equalities2, inequalities2 = _verifier_constraints(
                table,
                o1_support,
                o2_support,
                "O2",
                limits,
                budget,
            )
            local_o2 = _verifier_vertices(
                equalities2,
                inequalities2,
                len(o2_support),
                limits,
                budget,
            )
            if not local_o2:
                expected_outcome_by_pair[pair] = "empty_o2_mixture_polytope"
                continue
            o1_vertices = tuple(
                sorted(
                    _expand_vertex(vertex, o1_support, len(table))
                    for vertex in local_o1
                )
            )
            o2_vertices = tuple(
                sorted(
                    _expand_vertex(vertex, o2_support, len(table[0]))
                    for vertex in local_o2
                )
            )
            key = (o1_vertices, o2_vertices)
            sources = expected_sources.setdefault(key, [])
            expected_outcome_by_pair[pair] = (
                "new_support_cell"
                if not sources
                else "exact_duplicate_support_cell"
            )
            sources.append(pair)

    solver_sources = {
        (cell.o1_vertices, cell.o2_vertices): list(cell.source_support_pairs)
        for cell in cells
    }
    if set(solver_sources) != set(expected_sources):
        raise _ResponseFailure(
            NUMERIC_FAILURE,
            "verifier",
            "solver/verifier support-cell union mismatch",
        )
    for key in expected_sources:
        expected = sorted(
            expected_sources[key],
            key=lambda pair: (len(pair[0]) + len(pair[1]), pair[0], pair[1]),
        )
        if solver_sources[key] != expected:
            raise _ResponseFailure(
                NUMERIC_FAILURE,
                "verifier",
                "solver/verifier support-source mismatch",
            )
    actual_audit = {
        item["support_pair"]: item["outcome"] for item in audit
    }
    if actual_audit != expected_outcome_by_pair:
        raise _ResponseFailure(
            NUMERIC_FAILURE,
            "verifier",
            "solver/verifier support-pair outcome mismatch",
        )

    verified_vertex_pairs = 0
    for cell in cells:
        for o1_vertex in cell.o1_vertices:
            budget.spend(len(o1_vertex))
            if any(value < 0 for value in o1_vertex) or sum(o1_vertex) != 1:
                raise _ResponseFailure(
                    NUMERIC_FAILURE, "verifier", "invalid O1 simplex vertex"
                )
        for o2_vertex in cell.o2_vertices:
            budget.spend(len(o2_vertex))
            if any(value < 0 for value in o2_vertex) or sum(o2_vertex) != 1:
                raise _ResponseFailure(
                    NUMERIC_FAILURE, "verifier", "invalid O2 simplex vertex"
                )
        for o1_vertex in cell.o1_vertices:
            for o2_vertex in cell.o2_vertices:
                budget.spend()
                verified_vertex_pairs += 1
                utility = _verifier_utility_at(
                    table,
                    o1_vertex,
                    o2_vertex,
                    limits,
                    budget,
                )
                conservation = Fraction(0)
                for name in ("H", "O1", "O2", "R"):
                    conservation = _v_checked(
                        conservation + utility.for_name(name),
                        limits,
                        budget,
                    )
                if conservation != 0:
                    raise _ResponseFailure(
                        NUMERIC_FAILURE,
                        "verifier",
                        "mixture conservation mismatch",
                    )
                residuals = _verifier_residuals(
                    table,
                    o1_vertex,
                    o2_vertex,
                    utility,
                    limits,
                    budget,
                )
                if residuals != {"O1": Fraction(0), "O2": Fraction(0)}:
                    raise _ResponseFailure(
                        NUMERIC_FAILURE,
                        "verifier",
                        "returned vertex pair has a positive unilateral residual",
                    )
    if not cells:
        raise _ResponseFailure(
            NUMERIC_FAILURE,
            "verifier",
            "valid finite game produced zero equilibrium support cells",
        )
    return {
        "version": VERIFIER_VERSION,
        "status": "VERIFIED",
        "support_pairs_reconstructed": len(expected_outcome_by_pair),
        "support_cells_reconstructed": len(expected_sources),
        "vertex_pairs_verified": verified_vertex_pairs,
        "operations_used": budget.used,
        "solver_helpers_reused": False,
    }


def _mixture_record(
    plan_ids: tuple[str, ...], mixture: tuple[Fraction, ...]
) -> dict[str, str]:
    return {
        plan_id: _rational_text(mixture[index])
        for index, plan_id in enumerate(plan_ids)
    }


def _positive_support(
    plan_ids: tuple[str, ...], mixture: tuple[Fraction, ...]
) -> list[str]:
    return [
        plan_id
        for index, plan_id in enumerate(plan_ids)
        if mixture[index] > 0
    ]


def _utility_record(utility: _Utility) -> dict[str, str]:
    return {
        name: _rational_text(utility.for_name(name))
        for name in ("H", "O1", "O2", "R")
    }


def _witness_key(
    o1_mixture: tuple[Fraction, ...],
    o2_mixture: tuple[Fraction, ...],
    utility: _Utility,
) -> tuple[Any, ...]:
    return (
        o1_mixture,
        o2_mixture,
        tuple(utility.for_name(name) for name in ("H", "O1", "O2", "R")),
    )


def _extrema_and_vertex_pairs(
    table: tuple[tuple[_Utility, ...], ...],
    cells: Sequence[_SupportCell],
    o1_plan_ids: tuple[str, ...],
    o2_plan_ids: tuple[str, ...],
    limits: ExactResponseLimits,
    output_budget: _OutputRecordBudget,
) -> tuple[dict[str, Any], int]:
    vertex_pair_count = 0
    for cell in cells:
        vertex_pair_count += len(cell.o1_vertices) * len(cell.o2_vertices)
        if vertex_pair_count > limits.max_vertex_pairs_evaluated:
            raise _ResponseFailure(
                CAP_EXCEEDED,
                "extrema",
                "max_vertex_pairs_evaluated exceeded",
            )
    extrema: dict[str, dict[str, Any]] = {
        name: {
            "minimum": None,
            "maximum": None,
            "minimum_witnesses": {},
            "maximum_witnesses": {},
        }
        for name in ("H", "O1", "O2", "R")
    }

    def update(
        name: str,
        value: Fraction,
        key: tuple[Any, ...],
        cell_id: str,
        o1_mixture: tuple[Fraction, ...],
        o2_mixture: tuple[Fraction, ...],
        utility: _Utility,
        residuals: Mapping[str, Fraction],
    ) -> None:
        record = extrema[name]
        for direction, comparison in (
            ("minimum", lambda candidate, current: candidate < current),
            ("maximum", lambda candidate, current: candidate > current),
        ):
            current = record[direction]
            witnesses: dict[tuple[Any, ...], dict[str, Any]] = record[
                f"{direction}_witnesses"
            ]
            if current is None or comparison(value, current):
                record[direction] = value
                output_budget.release(len(witnesses))
                witnesses.clear()
            if value == record[direction]:
                witness = witnesses.get(key)
                if witness is None:
                    output_budget.reserve()
                    witness = {
                        "support_cell_ids": [],
                        "o1_support": _positive_support(o1_plan_ids, o1_mixture),
                        "o2_support": _positive_support(o2_plan_ids, o2_mixture),
                        "o1_mixture": _mixture_record(o1_plan_ids, o1_mixture),
                        "o2_mixture": _mixture_record(o2_plan_ids, o2_mixture),
                        "utility": _utility_record(utility),
                        "unilateral_residual": {
                            "O1": _rational_text(residuals["O1"]),
                            "O2": _rational_text(residuals["O2"]),
                        },
                        "extremum_value": _rational_text(value),
                    }
                    witnesses[key] = witness
                if cell_id not in witness["support_cell_ids"]:
                    witness["support_cell_ids"].append(cell_id)

    evaluated = 0
    for cell in cells:
        for o1_mixture in cell.o1_vertices:
            for o2_mixture in cell.o2_vertices:
                evaluated += 1
                utility = _utility_at(
                    table, o1_mixture, o2_mixture, limits
                )
                residuals = _unilateral_residuals(
                    table, o1_mixture, o2_mixture, utility, limits
                )
                if residuals["O1"] != 0 or residuals["O2"] != 0:
                    raise _ResponseFailure(
                        NUMERIC_FAILURE,
                        "extrema",
                        "support-cell vertex pair failed equilibrium residual check",
                    )
                key = _witness_key(o1_mixture, o2_mixture, utility)
                for name in ("H", "O1", "O2", "R"):
                    update(
                        name,
                        utility.for_name(name),
                        key,
                        cell.support_cell_id,
                        o1_mixture,
                        o2_mixture,
                        utility,
                        residuals,
                    )
    if evaluated != vertex_pair_count or evaluated == 0:
        raise _ResponseFailure(
            NUMERIC_FAILURE, "extrema", "vertex-pair coverage mismatch"
        )

    output: dict[str, Any] = {}
    for name in ("H", "O1", "O2", "R"):
        record = extrema[name]
        witness_lists: list[list[dict[str, Any]]] = []
        for direction in ("minimum", "maximum"):
            entries = list(record[f"{direction}_witnesses"].items())
            for _, witness in entries:
                witness["support_cell_ids"].sort()
                witness["support_cell_id"] = witness["support_cell_ids"][0]
            entries.sort(
                key=lambda item: (
                    item[0],
                    tuple(item[1]["support_cell_ids"]),
                )
            )
            witness_lists.append([witness for _, witness in entries])
        minimum_witnesses, maximum_witnesses = witness_lists
        output[name] = {
            "minimum": _rational_text(record["minimum"]),
            "maximum": _rational_text(record["maximum"]),
            "minimum_witnesses": minimum_witnesses,
            "maximum_witnesses": maximum_witnesses,
            "witness_scope": "all_canonical_vertex_pair_tie_witnesses",
        }
    return output, evaluated


def _pure_stability_subset(
    table: tuple[tuple[_Utility, ...], ...],
    o1_plan_ids: tuple[str, ...],
    o2_plan_ids: tuple[str, ...],
    limits: ExactResponseLimits,
    output_budget: _OutputRecordBudget,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for i, o1_plan_id in enumerate(o1_plan_ids):
        for j, o2_plan_id in enumerate(o2_plan_ids):
            o1_mixture = tuple(
                Fraction(1) if index == i else Fraction(0)
                for index in range(len(o1_plan_ids))
            )
            o2_mixture = tuple(
                Fraction(1) if index == j else Fraction(0)
                for index in range(len(o2_plan_ids))
            )
            utility = table[i][j]
            residuals = _unilateral_residuals(
                table, o1_mixture, o2_mixture, utility, limits
            )
            if residuals["O1"] == 0 and residuals["O2"] == 0:
                output_budget.reserve()
                rows.append(
                    {
                        "profile_id": f"O1:{i}|O2:{j}",
                        "plans": {"O1": o1_plan_id, "O2": o2_plan_id},
                        "utility": _utility_record(utility),
                        "unilateral_residual": {
                            "O1": _rational_text(residuals["O1"]),
                            "O2": _rational_text(residuals["O2"]),
                        },
                    }
                )
    return {
        "coverage": "complete",
        "profile_count": len(rows),
        "rows": rows,
        "residual_semantics": (
            "max(0,best_response_value-current_value), including the current "
            "strategy among best-response candidates"
        ),
    }


def _joint_plan_stress(
    table: tuple[tuple[_Utility, ...], ...],
    o1_plan_ids: tuple[str, ...],
    o2_plan_ids: tuple[str, ...],
    response_game_identity: str,
    hero_worst_response: Fraction,
    limits: ExactResponseLimits,
    output_budget: _OutputRecordBudget,
) -> dict[str, Any]:
    minimum = min(
        table[i][j].H
        for i in range(len(o1_plan_ids))
        for j in range(len(o2_plan_ids))
    )
    witnesses = []
    for i, o1_plan_id in enumerate(o1_plan_ids):
        for j, o2_plan_id in enumerate(o2_plan_ids):
            utility = table[i][j]
            if utility.H == minimum:
                output_budget.reserve()
                witnesses.append(
                    {
                        "profile_id": f"O1:{i}|O2:{j}",
                        "plans": {"O1": o1_plan_id, "O2": o2_plan_id},
                        "utility": _utility_record(utility),
                    }
                )
    difference = _fsub(
        hero_worst_response,
        minimum,
        limits,
        "hero_min_joint_plan_stress",
    )
    return {
        "status": "COMPLETE",
        "coverage": "complete",
        "diagnostic_identity": _identity(
            {
                "diagnostic": "hero-min-joint-plan-stress-v1",
                "response_game_identity": response_game_identity,
            }
        ),
        "hero_minimum": _rational_text(minimum),
        "witnesses": witnesses,
        "hero_worst_response_minus_stress": _rational_text(difference),
        "primary_response_status_influence": False,
        "opponent_individual_rationality_not_required": True,
        "transferable_utility_assumed": False,
        "coalition_equilibrium_claim": False,
    }


def _support_pair_record(
    pair: tuple[tuple[int, ...], tuple[int, ...]],
    o1_plan_ids: tuple[str, ...],
    o2_plan_ids: tuple[str, ...],
) -> dict[str, list[str]]:
    return {
        "O1": [o1_plan_ids[index] for index in pair[0]],
        "O2": [o2_plan_ids[index] for index in pair[1]],
    }


def _cell_output(
    cell: _SupportCell,
    o1_plan_ids: tuple[str, ...],
    o2_plan_ids: tuple[str, ...],
) -> dict[str, Any]:
    total_dimension = cell.o1_dimension + cell.o2_dimension
    return {
        "support_cell_id": cell.support_cell_id,
        "representation": (
            "Cartesian product of complete exact-rational bounded marginal "
            "polytope V-representations"
        ),
        "kind": "singleton" if total_dimension == 0 else "continuum",
        "dimension": total_dimension,
        "source_support_pairs": [
            _support_pair_record(pair, o1_plan_ids, o2_plan_ids)
            for pair in cell.source_support_pairs
        ],
        "o1_mixture_polytope": {
            "dimension": cell.o1_dimension,
            "vertex_count": len(cell.o1_vertices),
            "vertices": [
                _mixture_record(o1_plan_ids, vertex)
                for vertex in cell.o1_vertices
            ],
        },
        "o2_mixture_polytope": {
            "dimension": cell.o2_dimension,
            "vertex_count": len(cell.o2_vertices),
            "vertices": [
                _mixture_record(o2_plan_ids, vertex)
                for vertex in cell.o2_vertices
            ],
        },
    }


def _projected_output_records(
    cells: Sequence[_SupportCell],
    audit_count: int,
    pure_count: int,
    extrema: Mapping[str, Any],
    stress_count: int,
) -> int:
    count = _base_output_records(cells, audit_count)
    count += pure_count + stress_count
    for name in ("H", "O1", "O2", "R"):
        count += len(extrema[name]["minimum_witnesses"])
        count += len(extrema[name]["maximum_witnesses"])
    return count


def _base_output_records(
    cells: Sequence[_SupportCell],
    audit_count: int,
) -> int:
    count = 256 + audit_count
    for cell in cells:
        count += (
            8
            + len(cell.source_support_pairs)
            + len(cell.o1_vertices)
            + len(cell.o2_vertices)
        )
    return count


def _preflight_output(
    records: int,
    cells: Sequence[_SupportCell],
    o1_plan_ids: tuple[str, ...],
    o2_plan_ids: tuple[str, ...],
    limits: ExactResponseLimits,
) -> None:
    if records > limits.max_output_records:
        raise _ResponseFailure(
            CAP_EXCEEDED, "output", "max_output_records exceeded"
        )
    maximum_label = max(len(value) for value in o1_plan_ids + o2_plan_ids)
    maximum_rational_chars = (
        math.ceil(limits.max_rational_numerator_bits * math.log10(2))
        + math.ceil(limits.max_rational_denominator_bits * math.log10(2))
        + 4
    )
    vertex_scalars = sum(
        len(cell.o1_vertices) * len(o1_plan_ids)
        + len(cell.o2_vertices) * len(o2_plan_ids)
        for cell in cells
    )
    conservative_bytes = (
        8_192
        + records * (128 + 4 * maximum_label)
        + vertex_scalars * (maximum_rational_chars + maximum_label + 8)
    )
    if conservative_bytes > limits.max_output_bytes:
        raise _ResponseFailure(
            CAP_EXCEEDED, "output", "max_output_bytes preflight exceeded"
        )


def _validate_context_and_pins(
    game: ExactReducedGame,
    pins: ResponseIdentityPins,
) -> None:
    if game.contract_version != CONTRACT_VERSION:
        raise _ResponseFailure(
            UNSUPPORTED_MODEL, "contract_version", "unsupported contract version"
        )
    if game.algorithm_version != ALGORITHM_VERSION:
        raise _ResponseFailure(
            UNSUPPORTED_MODEL, "algorithm_version", "unsupported algorithm version"
        )
    _validate_identity(
        game.response_game_structure_identity,
        "response_game_structure_identity",
    )
    _validate_identity(game.fixed_hero_identity, "fixed_hero_identity")
    if game.perfect_recall_evidence_identity is None:
        raise _ResponseFailure(
            UNSUPPORTED_MODEL,
            "perfect_recall_evidence_identity",
            "perfect-recall evidence identity is required",
        )
    _validate_identity(
        game.perfect_recall_evidence_identity,
        "perfect_recall_evidence_identity",
    )
    _validate_identity(
        game.rake_convention_identity, "rake_convention_identity"
    )
    for name in (
        "supplied_profile_identity",
        "candidate_identity",
        "search_mode_context_identity",
        "run_context_identity",
    ):
        _validate_identity(getattr(game, name), name, optional=True)
    if type(pins) is not ResponseIdentityPins:
        raise _ResponseFailure(
            INVALID_INPUT, "identity_pins", "pins must be ResponseIdentityPins"
        )
    for field in fields(ResponseIdentityPins):
        _validate_identity(
            getattr(pins, field.name),
            f"identity_pins.{field.name}",
            optional=True,
        )
    actual_components = {
        "response_game_structure_identity": game.response_game_structure_identity,
        "fixed_hero_identity": game.fixed_hero_identity,
        "perfect_recall_evidence_identity": game.perfect_recall_evidence_identity,
        "rake_convention_identity": game.rake_convention_identity,
        "supplied_profile_identity": game.supplied_profile_identity,
        "candidate_identity": game.candidate_identity,
        "search_mode_context_identity": game.search_mode_context_identity,
        "run_context_identity": game.run_context_identity,
    }
    for name, actual in actual_components.items():
        expected = getattr(pins, name)
        if expected is not None and expected != actual:
            raise _ResponseFailure(
                STALE_INPUT, f"identity_pins.{name}", f"stale {name}"
            )


def _solve_exact_response(
    game: ExactReducedGame,
    limits: ExactResponseLimits,
    pins: ResponseIdentityPins,
) -> ExactResponseResult:
    if type(game) is not ExactReducedGame:
        raise _ResponseFailure(
            INVALID_INPUT, "game", "game must be ExactReducedGame"
        )
    _validate_limits(limits)
    _validate_context_and_pins(game, pins)
    o1_plan_ids = _validate_plan_ids(
        game.o1_plan_ids, "o1_plan_ids", limits.max_pure_plans_o1
    )
    o2_plan_ids = _validate_plan_ids(
        game.o2_plan_ids, "o2_plan_ids", limits.max_pure_plans_o2
    )
    joint_profiles = _checked_product(
        len(o1_plan_ids),
        len(o2_plan_ids),
        limits.max_joint_pure_profiles,
        "joint_pure_profiles",
    )
    if len(o1_plan_ids) > limits.max_support_size_o1:
        raise _ResponseFailure(
            CAP_EXCEEDED,
            "support_preflight",
            "max_support_size_o1 cannot cover the full plan set",
        )
    if len(o2_plan_ids) > limits.max_support_size_o2:
        raise _ResponseFailure(
            CAP_EXCEEDED,
            "support_preflight",
            "max_support_size_o2 cannot cover the full plan set",
        )
    support_pair_count = _checked_product(
        (1 << len(o1_plan_ids)) - 1,
        (1 << len(o2_plan_ids)) - 1,
        limits.max_support_pairs,
        "support_pairs",
    )
    table, payoff_table_identity = _payoff_table(
        game, o1_plan_ids, o2_plan_ids, limits
    )
    if (
        pins.payoff_table_identity is not None
        and pins.payoff_table_identity != payoff_table_identity
    ):
        raise _ResponseFailure(
            STALE_INPUT,
            "identity_pins.payoff_table_identity",
            "stale payoff_table_identity",
        )

    game_projection = {
        "semantics": "fixed-hero-two-opponent-noncooperative-bimatrix-v1",
        "o1_plan_ids": list(o1_plan_ids),
        "o2_plan_ids": list(o2_plan_ids),
        "payoff_table_identity": payoff_table_identity,
        "response_game_structure_identity": (
            game.response_game_structure_identity
        ),
        "fixed_hero_identity": game.fixed_hero_identity,
        "perfect_recall_evidence_identity": (
            game.perfect_recall_evidence_identity
        ),
        "rake_convention_identity": game.rake_convention_identity,
    }
    response_game_identity = _identity(game_projection)
    config_projection = {
        "contract_version": CONTRACT_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "verifier_version": VERIFIER_VERSION,
        "limits": asdict(limits),
        "ordering": (
            "plan order; support cardinality; lexicographic support indices; "
            "exact rational vertices; support_cell_id"
        ),
    }
    config_identity = _identity(config_projection)
    run_projection = {
        "response_game_identity": response_game_identity,
        "config_identity": config_identity,
        "supplied_profile_identity": game.supplied_profile_identity,
        "candidate_identity": game.candidate_identity,
        "search_mode_context_identity": game.search_mode_context_identity,
        "run_context_identity": game.run_context_identity,
    }
    response_run_identity = _identity(run_projection)
    for name, actual in (
        ("config_identity", config_identity),
        ("response_game_identity", response_game_identity),
        ("response_run_identity", response_run_identity),
    ):
        expected = getattr(pins, name)
        if expected is not None and expected != actual:
            raise _ResponseFailure(
                STALE_INPUT, f"identity_pins.{name}", f"stale {name}"
            )

    o1_supports = _support_subsets(len(o1_plan_ids))
    o2_supports = _support_subsets(len(o2_plan_ids))
    if len(o1_supports) * len(o2_supports) != support_pair_count:
        raise _ResponseFailure(
            NUMERIC_FAILURE,
            "support_preflight",
            "support-pair count mismatch",
        )
    cells, audit, counters = _enumerate_support_cells(
        table, o1_supports, o2_supports, limits
    )
    if len(audit) != support_pair_count:
        raise _ResponseFailure(
            NUMERIC_FAILURE,
            "solver",
            "not every nonempty support pair was visited",
        )
    verifier = _verify_correspondence(
        table, o1_supports, o2_supports, cells, audit, limits
    )
    output_budget = _OutputRecordBudget(limits.max_output_records)
    output_budget.reserve(_base_output_records(cells, len(audit)))
    extrema, vertex_pairs_evaluated = _extrema_and_vertex_pairs(
        table,
        cells,
        o1_plan_ids,
        o2_plan_ids,
        limits,
        output_budget,
    )
    pure_subset = _pure_stability_subset(
        table, o1_plan_ids, o2_plan_ids, limits, output_budget
    )
    stress = _joint_plan_stress(
        table,
        o1_plan_ids,
        o2_plan_ids,
        response_game_identity,
        Fraction(extrema["H"]["minimum"]),
        limits,
        output_budget,
    )
    records = _projected_output_records(
        cells,
        len(audit),
        pure_subset["profile_count"],
        extrema,
        len(stress["witnesses"]),
    )
    if records != output_budget.used:
        raise _ResponseFailure(
            NUMERIC_FAILURE,
            "output",
            "output record budget accounting mismatch",
        )
    _preflight_output(
        records, cells, o1_plan_ids, o2_plan_ids, limits
    )

    audit_output = []
    for item in audit:
        audit_output.append(
            {
                "support_pair": _support_pair_record(
                    item["support_pair"], o1_plan_ids, o2_plan_ids
                ),
                "outcome": item["outcome"],
                "support_cell_id": item["support_cell_id"],
            }
        )
    response = {
        "contract_version": CONTRACT_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "verifier_version": VERIFIER_VERSION,
        "status": EXACT_CORRESPONDENCE_COMPLETE,
        "coverage": "complete",
        "partial_response": False,
        "response_game_identity": response_game_identity,
        "response_run_identity": response_run_identity,
        "content_identities": {
            "response_game_structure": game.response_game_structure_identity,
            "fixed_hero": game.fixed_hero_identity,
            "perfect_recall_evidence": game.perfect_recall_evidence_identity,
            "rake_convention": game.rake_convention_identity,
            "payoff_table": payoff_table_identity,
            "config": config_identity,
            "supplied_profile": game.supplied_profile_identity,
            "candidate": game.candidate_identity,
            "search_mode_context": game.search_mode_context_identity,
            "run_context": game.run_context_identity,
        },
        "counts": {
            "pure_plans": {
                "O1": len(o1_plan_ids),
                "O2": len(o2_plan_ids),
            },
            "joint_pure_profiles": joint_profiles,
            "support_pairs_total": support_pair_count,
            "support_pairs_visited": len(audit),
            "raw_nonempty_support_cells": counters.raw_nonempty_support_cells,
            "exact_duplicate_support_cells": (
                counters.exact_duplicate_support_cells
            ),
            "canonical_support_cells": len(cells),
            "total_marginal_vertices": counters.total_vertices,
            "vertex_pairs_evaluated": vertex_pairs_evaluated,
            "exact_linear_systems_preflight": (
                counters.linear_systems_preflight
            ),
            "exact_linear_systems_solved": counters.linear_systems_solved,
            "pure_profile_unilateral_stability_rows": (
                pure_subset["profile_count"]
            ),
            "output_records_projected": records,
        },
        "ordering": {
            "plans": "caller-declared O1 order then caller-declared O2 order",
            "support_pairs": (
                "O1 support cardinality/lexicographic, then O2 support "
                "cardinality/lexicographic"
            ),
            "vertices": "lexicographic exact rational mixtures",
            "support_cells": (
                "minimum source support cardinality/lexicographic, then "
                "support_cell_id"
            ),
            "witnesses": "exact mixture then support_cell_id",
        },
        "limits": asdict(limits),
        "support_cells": [
            _cell_output(cell, o1_plan_ids, o2_plan_ids)
            for cell in cells
        ],
        "support_pair_audit": audit_output,
        "pure_profile_unilateral_stability": pure_subset,
        "utility_extrema": extrema,
        "hero_worst": extrema["H"]["minimum"],
        "hero_best": extrema["H"]["maximum"],
        "hero_worst_witnesses": extrema["H"]["minimum_witnesses"],
        "hero_best_witnesses": extrema["H"]["maximum_witnesses"],
        "hero_min_joint_plan_stress": stress,
        "unilateral_residual_semantics": (
            "exact nonnegative max(0,best-response value-current value); "
            "R is excluded from response conditions"
        ),
        "unilateral_residual_certificate": {
            "scope": (
                "all support cells by exact best-response constraints; all "
                "canonical marginal vertex pairs independently verified"
            ),
            "maximum": {"O1": "0", "O2": "0"},
        },
        "independent_verification": verifier,
    }
    encoded = _canonical_json_bytes(response)
    if len(encoded) > limits.max_output_bytes:
        raise _ResponseFailure(
            CAP_EXCEEDED, "output", "max_output_bytes exceeded"
        )
    return ExactResponseResult(
        status=EXACT_CORRESPONDENCE_COMPLETE,
        response=response,
        error=None,
        partial_response=False,
    )


def solve_three_player_response(
    game: ExactReducedGame,
    *,
    limits: ExactResponseLimits = ExactResponseLimits(),
    pins: ResponseIdentityPins = ResponseIdentityPins(),
) -> ExactResponseResult:
    """Return the complete bounded exact response correspondence or no payload.

    ``EXACT_CORRESPONDENCE_COMPLETE`` is returned only after every nonempty
    support pair and every support-cell vertex has passed independent exact
    verification.  All controlled and unexpected failures have
    ``response=None`` and ``partial_response=False``.
    """

    try:
        return _solve_exact_response(game, limits, pins)
    except _ResponseFailure as exc:
        return _failure(exc)
    except Exception:
        return _failure(
            _ResponseFailure(
                INTERNAL_FAILURE,
                "internal",
                "unexpected exact response failure",
            )
        )


def exact_response_json(result: ExactResponseResult) -> str:
    """Serialize one outer result as strict deterministic one-line JSON."""

    if type(result) is not ExactResponseResult:
        raise TypeError("result must be ExactResponseResult")
    return _canonical_json_bytes(result.to_dict()).decode("utf-8")
