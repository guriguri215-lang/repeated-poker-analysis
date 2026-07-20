"""Bounded in-memory orchestration for prepared one- or two-street games.

This module forwards one flat prepared abstract heads-up request to the M14
builder and then to the M15 evaluator.  Hero is completely fixed, so the
response is Villain's exact response to that fixed Hero; an optional Villain
profile is only a chosen fixed comparison profile.  This is not an equilibrium,
Nash calculation, optimal-Hero or candidate strategy generator, profitability
claim, solver-grade certificate, or real-money, gambling, bankroll, financial,
or legal advice.

The scope excludes real cards, hand evaluation, ranges, card removal, equity
generation, raw solver exports, arbitrary nested trees, three or more streets,
multiway or side-pot play, full charts, large-scale solvers, and real-opponent
models.  It performs no filesystem or scenario JSON I/O and has no CLI,
pipeline, manifest, report, export, or GUI integration.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Mapping

from .prepared_two_street import (
    PREPARED_CHANCE_NORMALIZATION_ID,
    PREPARED_INFORMATION_KEY_ID,
    PREPARED_JOINT_ROOT_BUILDER_ID,
    PREPARED_JOINT_ROOT_CONTRACT_VERSION,
    PREPARED_TWO_STREET_BUILDER_ID,
    PREPARED_TWO_STREET_CONTRACT_VERSION,
    PreparedBuildCounts,
    PreparedBuildError,
    PreparedContentIdentity,
    PreparedJointRootTwoStreetSpec,
    PreparedTwoStreetBuild,
    PreparedTwoStreetBuildResult,
    PreparedTwoStreetIdentity,
    PreparedTwoStreetLimits,
    PreparedTwoStreetSpec,
    PreparedTwoStreetStatus,
    build_prepared_two_street_game,
    prepared_semantic_sha256,
)
from .prepared_two_street_evaluation import (
    PREPARED_EVALUATION_OUTPUT_ID,
    PREPARED_EVALUATION_TOLERANCE,
    PREPARED_PROFILE_ABSENT,
    PREPARED_PROFILE_EFFECTIVE_ID,
    PREPARED_PROFILE_NORMALIZATION_ID,
    PREPARED_PROFILE_NORMALIZATION_TOLERANCE,
    PREPARED_PROFILE_RAW_ID,
    PREPARED_TWO_STREET_EVALUATION_ADAPTER_ID,
    PreparedCorrespondenceMode,
    PreparedEvaluation,
    PreparedEvaluationError,
    PreparedEvaluationIdentity,
    PreparedEvaluationLimits,
    PreparedEvaluationRequest,
    PreparedEvaluationResult,
    PreparedEvaluationStatus,
    PreparedPlayerProfile,
    PreparedProfileNormalization,
    evaluate_prepared_two_street,
)


PREPARED_TWO_STREET_ORCHESTRATION_ID = (
    "prepared-two-street-orchestration-v1"
)
PREPARED_TWO_STREET_ORCHESTRATION_OUTPUT_ID = (
    "prepared-two-street-orchestration-output-sha256-v1"
)


__all__ = [
    "PREPARED_TWO_STREET_ORCHESTRATION_ID",
    "PREPARED_TWO_STREET_ORCHESTRATION_OUTPUT_ID",
    "PreparedOrchestrationStatus",
    "PreparedOrchestrationLimits",
    "PreparedTwoStreetOrchestrationRequest",
    "PreparedOrchestrationTraceRecord",
    "PreparedTwoStreetOrchestrationRun",
    "PreparedTwoStreetOrchestrationIdentity",
    "PreparedOrchestrationError",
    "PreparedTwoStreetOrchestrationResult",
    "run_prepared_two_street_orchestration",
]


class PreparedOrchestrationStatus(str, Enum):
    SUCCESS = "SUCCESS"
    INVALID_INPUT = "INVALID_INPUT"
    BUILD_FAILURE = "BUILD_FAILURE"
    BUILD_IDENTITY_MISMATCH = "BUILD_IDENTITY_MISMATCH"
    EVALUATION_INPUT_FAILURE = "EVALUATION_INPUT_FAILURE"
    EVALUATION_CORE_FAILURE = "EVALUATION_CORE_FAILURE"
    CAP_EXCEEDED = "CAP_EXCEEDED"
    NUMERIC_FAILURE = "NUMERIC_FAILURE"
    NON_REPRODUCIBLE = "NON_REPRODUCIBLE"
    UNSUPPORTED_DOWNSTREAM = "UNSUPPORTED_DOWNSTREAM"
    INTERNAL_FAILURE = "INTERNAL_FAILURE"


@dataclass(frozen=True)
class PreparedOrchestrationLimits:
    max_trace_records: int = 16
    max_result_records: int = 64


@dataclass(frozen=True)
class PreparedTwoStreetOrchestrationRequest:
    spec: PreparedTwoStreetSpec | PreparedJointRootTwoStreetSpec
    raw_input_bytes: bytes
    content_identity: PreparedContentIdentity
    hero_profile: PreparedPlayerProfile
    villain_profile: PreparedPlayerProfile | None = None
    builder_limits: PreparedTwoStreetLimits = PreparedTwoStreetLimits()
    evaluation_limits: PreparedEvaluationLimits = PreparedEvaluationLimits()
    evaluation_request: PreparedEvaluationRequest = PreparedEvaluationRequest()
    orchestration_limits: PreparedOrchestrationLimits = PreparedOrchestrationLimits()
    expected_output_semantic_sha256: str | None = None


@dataclass(frozen=True)
class PreparedOrchestrationTraceRecord:
    phase: str
    subject: str
    outcome: str


@dataclass(frozen=True)
class PreparedTwoStreetOrchestrationRun:
    build: PreparedTwoStreetBuild
    evaluation: PreparedEvaluation
    trace: tuple[PreparedOrchestrationTraceRecord, ...]


@dataclass(frozen=True)
class PreparedTwoStreetOrchestrationIdentity:
    orchestration_id: str
    output_semantic_id: str
    m14_identity: PreparedTwoStreetIdentity
    m15_identity: PreparedEvaluationIdentity
    effective_builder_limits: PreparedTwoStreetLimits
    effective_evaluation_limits: PreparedEvaluationLimits
    effective_orchestration_limits: PreparedOrchestrationLimits
    correspondence_mode: PreparedCorrespondenceMode
    oracle_check: bool
    run_identity: str
    output_semantic_sha256: str


@dataclass(frozen=True)
class PreparedOrchestrationError:
    phase: str
    message: str
    build_identity: PreparedTwoStreetIdentity | None
    build_counts: PreparedBuildCounts | None


@dataclass(frozen=True)
class PreparedTwoStreetOrchestrationResult:
    status: PreparedOrchestrationStatus
    builder_status: PreparedTwoStreetStatus | None
    evaluation_status: PreparedEvaluationStatus | None
    run: PreparedTwoStreetOrchestrationRun | None
    identity: PreparedTwoStreetOrchestrationIdentity | None
    error: PreparedOrchestrationError | None


_DEFAULT_LIMITS = PreparedOrchestrationLimits()
_M14_ACTION_LABEL_ID = "betting-tree-v2-action-label-v1"
_M14_CONTRACT_BUILDER_PAIRS = frozenset(
    {
        (
            PREPARED_TWO_STREET_CONTRACT_VERSION,
            PREPARED_TWO_STREET_BUILDER_ID,
        ),
        (
            PREPARED_JOINT_ROOT_CONTRACT_VERSION,
            PREPARED_JOINT_ROOT_BUILDER_ID,
        ),
    }
)
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_TRACE_VALUES = (
    ("orchestration", "input", "PASS"),
    ("builder", "completed", "PASS"),
    ("evaluation", "completed", "PASS"),
    ("orchestration", "identity", "PASS"),
)

_BUILD_STATUS_MAP = {
    PreparedTwoStreetStatus.INVALID_INPUT: PreparedOrchestrationStatus.BUILD_FAILURE,
    PreparedTwoStreetStatus.INVALID_ACTION_GRAMMAR: PreparedOrchestrationStatus.BUILD_FAILURE,
    PreparedTwoStreetStatus.ACCOUNTING_MISMATCH: PreparedOrchestrationStatus.BUILD_FAILURE,
    PreparedTwoStreetStatus.UNSUPPORTED_MODEL: PreparedOrchestrationStatus.BUILD_FAILURE,
    PreparedTwoStreetStatus.EMPTY_CHANCE_SUPPORT: PreparedOrchestrationStatus.BUILD_FAILURE,
    PreparedTwoStreetStatus.INVALID_CHANCE_SUPPORT: PreparedOrchestrationStatus.BUILD_FAILURE,
    PreparedTwoStreetStatus.CONTENT_HASH_MISMATCH: PreparedOrchestrationStatus.BUILD_FAILURE,
    PreparedTwoStreetStatus.INFORMATION_MODEL_MISMATCH: PreparedOrchestrationStatus.BUILD_FAILURE,
    PreparedTwoStreetStatus.PERFECT_RECALL_ATTESTATION_MISSING: PreparedOrchestrationStatus.BUILD_FAILURE,
    PreparedTwoStreetStatus.CAP_EXCEEDED: PreparedOrchestrationStatus.CAP_EXCEEDED,
    PreparedTwoStreetStatus.NUMERIC_FAILURE: PreparedOrchestrationStatus.NUMERIC_FAILURE,
    PreparedTwoStreetStatus.NON_REPRODUCIBLE: PreparedOrchestrationStatus.NON_REPRODUCIBLE,
    PreparedTwoStreetStatus.ORACLE_MISMATCH: PreparedOrchestrationStatus.BUILD_FAILURE,
    PreparedTwoStreetStatus.UNSUPPORTED_DOWNSTREAM: PreparedOrchestrationStatus.UNSUPPORTED_DOWNSTREAM,
}

_EVALUATION_STATUS_MAP = {
    PreparedEvaluationStatus.INVALID_INPUT: PreparedOrchestrationStatus.EVALUATION_INPUT_FAILURE,
    PreparedEvaluationStatus.INCOMPLETE_PROFILE: PreparedOrchestrationStatus.EVALUATION_INPUT_FAILURE,
    PreparedEvaluationStatus.ILLEGAL_PROFILE_REFERENCE: PreparedOrchestrationStatus.EVALUATION_INPUT_FAILURE,
    PreparedEvaluationStatus.INVALID_PROFILE_PROBABILITY: PreparedOrchestrationStatus.EVALUATION_INPUT_FAILURE,
    PreparedEvaluationStatus.BUILD_MISMATCH: PreparedOrchestrationStatus.BUILD_IDENTITY_MISMATCH,
    PreparedEvaluationStatus.IDENTITY_MISMATCH: PreparedOrchestrationStatus.BUILD_IDENTITY_MISMATCH,
    PreparedEvaluationStatus.CAP_EXCEEDED: PreparedOrchestrationStatus.CAP_EXCEEDED,
    PreparedEvaluationStatus.NUMERIC_FAILURE: PreparedOrchestrationStatus.NUMERIC_FAILURE,
    PreparedEvaluationStatus.FIXED_PROFILE_FAILURE: PreparedOrchestrationStatus.EVALUATION_CORE_FAILURE,
    PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE: PreparedOrchestrationStatus.EVALUATION_CORE_FAILURE,
    PreparedEvaluationStatus.NON_REPRODUCIBLE: PreparedOrchestrationStatus.NON_REPRODUCIBLE,
    PreparedEvaluationStatus.ORACLE_MISMATCH: PreparedOrchestrationStatus.EVALUATION_CORE_FAILURE,
    PreparedEvaluationStatus.UNSUPPORTED_DOWNSTREAM: PreparedOrchestrationStatus.UNSUPPORTED_DOWNSTREAM,
}


class _CanonicalNumericError(ValueError):
    pass


def _is_sha256(value: object) -> bool:
    return type(value) is str and _SHA_RE.fullmatch(value) is not None


def _bounded_text(value: str, limit: int) -> str:
    return value.replace("\r", " ").replace("\n", " ")[:limit]


def _provenance(
    build: PreparedTwoStreetBuild | None,
) -> tuple[PreparedTwoStreetIdentity | None, PreparedBuildCounts | None]:
    if build is None:
        return None, None
    identity = build.identity
    counts = build.counts
    if type(identity) is PreparedTwoStreetIdentity and type(counts) is PreparedBuildCounts:
        return identity, counts
    return None, None


def _failure(
    status: PreparedOrchestrationStatus,
    builder_status: PreparedTwoStreetStatus | None,
    evaluation_status: PreparedEvaluationStatus | None,
    phase: str,
    message: str,
    build: PreparedTwoStreetBuild | None = None,
) -> PreparedTwoStreetOrchestrationResult:
    build_identity, build_counts = _provenance(build)
    error = PreparedOrchestrationError(
        _bounded_text(phase or "internal", 64),
        _bounded_text(message or "unexpected orchestration failure", 500),
        build_identity,
        build_counts,
    )
    return PreparedTwoStreetOrchestrationResult(
        status, builder_status, evaluation_status, None, None, error
    )


def _checked_add(total: int, increment: int, cap: int) -> int:
    value = total + increment
    if value > cap:
        raise OverflowError
    return value


def _validate_request(request: object) -> bool:
    if type(request) is not PreparedTwoStreetOrchestrationRequest:
        return False
    if type(request.spec) not in (
        PreparedTwoStreetSpec,
        PreparedJointRootTwoStreetSpec,
    ):
        return False
    limits = request.orchestration_limits
    if type(limits) is not PreparedOrchestrationLimits:
        return False
    for item in fields(_DEFAULT_LIMITS):
        value = getattr(limits, item.name)
        ceiling = getattr(_DEFAULT_LIMITS, item.name)
        if type(value) is not int or value <= 0 or value > ceiling:
            return False
    witness = request.expected_output_semantic_sha256
    return witness is None or _is_sha256(witness)


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if type(value) is float:
        if not math.isfinite(value):
            raise _CanonicalNumericError
        return {"__floathex__": value.hex()}
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is bytes:
        return {"__bytes_hex__": value.hex()}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": type(value).__name__,
            "fields": {
                item.name: _canonical_value(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if type(value) in (tuple, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("canonical mapping keys must be strings")
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    raise TypeError(f"unsupported canonical type: {type(value).__name__}")


def _sha256(value: Any) -> str:
    encoded = json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_builder_result(result: object) -> tuple[bool, PreparedTwoStreetStatus | None]:
    if type(result) is not PreparedTwoStreetBuildResult:
        return False, None
    status = result.status if type(result.status) is PreparedTwoStreetStatus else None
    if status is None:
        return False, None
    if status is PreparedTwoStreetStatus.SUCCESS:
        return type(result.build) is PreparedTwoStreetBuild and result.error is None, status
    valid_error = (
        type(result.error) is PreparedBuildError
        and type(result.error.phase) is str
        and bool(result.error.phase)
        and type(result.error.message) is str
        and bool(result.error.message)
    )
    return result.build is None and valid_error, status


def _valid_evaluation_result(
    result: object,
) -> tuple[bool, PreparedEvaluationStatus | None]:
    if type(result) is not PreparedEvaluationResult:
        return False, None
    status = result.status if type(result.status) is PreparedEvaluationStatus else None
    if status is None:
        return False, None
    if status is PreparedEvaluationStatus.SUCCESS:
        valid = (
            type(result.evaluation) is PreparedEvaluation
            and type(result.identity) is PreparedEvaluationIdentity
            and result.error is None
        )
        return valid, status
    valid_error = (
        type(result.error) is PreparedEvaluationError
        and type(result.error.phase) is str
        and bool(result.error.phase)
        and type(result.error.message) is str
        and bool(result.error.message)
    )
    return result.evaluation is None and result.identity is None and valid_error, status


def _identity_chain_matches(
    request: PreparedTwoStreetOrchestrationRequest,
    build: PreparedTwoStreetBuild,
    evaluation: PreparedEvaluation,
    identity: PreparedEvaluationIdentity,
) -> bool:
    m14 = build.identity
    if type(m14) is not PreparedTwoStreetIdentity or type(build.counts) is not PreparedBuildCounts:
        return False
    expected_pair = (
        (
            PREPARED_JOINT_ROOT_CONTRACT_VERSION,
            PREPARED_JOINT_ROOT_BUILDER_ID,
        )
        if type(request.spec) is PreparedJointRootTwoStreetSpec
        else (
            PREPARED_TWO_STREET_CONTRACT_VERSION,
            PREPARED_TWO_STREET_BUILDER_ID,
        )
    )
    if (
        type(m14.contract_version) is not str
        or type(m14.builder_id) is not str
        or (m14.contract_version, m14.builder_id)
        not in _M14_CONTRACT_BUILDER_PAIRS
        or (m14.contract_version, m14.builder_id) != expected_pair
        or m14.action_label_id != _M14_ACTION_LABEL_ID
        or m14.normalization_id != PREPARED_CHANCE_NORMALIZATION_ID
        or m14.information_key_id != PREPARED_INFORMATION_KEY_ID
        or not all(
            _is_sha256(value)
            for value in (
                m14.raw_sha256,
                m14.semantic_sha256,
                m14.ordered_tree_sha256,
                m14.run_identity,
            )
        )
    ):
        return False
    request_content = (
        type(request.raw_input_bytes) is bytes
        and type(request.content_identity) is PreparedContentIdentity
        and _is_sha256(request.content_identity.raw_sha256)
        and _is_sha256(request.content_identity.semantic_sha256)
        and request.content_identity.raw_sha256
        == hashlib.sha256(request.raw_input_bytes).hexdigest()
        and request.content_identity.semantic_sha256
        == prepared_semantic_sha256(request.spec)
        and m14.raw_sha256 == request.content_identity.raw_sha256
        and m14.semantic_sha256 == request.content_identity.semantic_sha256
    )
    m14_run_matches = m14.run_identity == _sha256(
        {
            "contract_version": m14.contract_version,
            "builder_id": m14.builder_id,
            "action_label_id": m14.action_label_id,
            "normalization_id": m14.normalization_id,
            "information_key_id": m14.information_key_id,
            "raw_sha256": m14.raw_sha256,
            "semantic_sha256": m14.semantic_sha256,
            "ordered_tree_sha256": m14.ordered_tree_sha256,
        }
    )
    continuity = (
        identity.m14_contract_version == m14.contract_version
        and identity.m14_builder_id == m14.builder_id
        and identity.m14_action_label_id == m14.action_label_id
        and identity.m14_normalization_id == m14.normalization_id
        and identity.m14_information_key_id == m14.information_key_id
        and identity.m14_raw_sha256 == m14.raw_sha256
        and identity.m14_prepared_semantic_sha256 == m14.semantic_sha256
        and identity.m14_ordered_tree_sha256 == m14.ordered_tree_sha256
        and identity.m14_run_identity == m14.run_identity
    )
    public_contract = (
        identity.evaluation_adapter_id == PREPARED_TWO_STREET_EVALUATION_ADAPTER_ID
        and identity.profile_normalization_id == PREPARED_PROFILE_NORMALIZATION_ID
        and identity.profile_raw_id == PREPARED_PROFILE_RAW_ID
        and identity.profile_effective_id == PREPARED_PROFILE_EFFECTIVE_ID
        and identity.output_semantic_id == PREPARED_EVALUATION_OUTPUT_ID
        and identity.profile_tolerance_hex == PREPARED_PROFILE_NORMALIZATION_TOLERANCE.hex()
        and identity.evaluation_tolerance_hex == PREPARED_EVALUATION_TOLERANCE.hex()
        and identity.effective_limits == request.evaluation_limits
        and identity.exact_response_method == request.evaluation_request.method
        and identity.correspondence_mode is request.evaluation_request.correspondence_mode
        and identity.oracle_check is request.evaluation_request.oracle_check
    )
    hero = evaluation.hero_profile_normalization
    if type(hero) is not PreparedProfileNormalization:
        return False
    hero_ok = (
        _is_sha256(identity.hero_raw_profile_sha256)
        and _is_sha256(identity.hero_effective_profile_sha256)
        and identity.hero_raw_profile_sha256 == hero.raw_profile_sha256
        and identity.hero_effective_profile_sha256 == hero.effective_profile_sha256
    )
    if request.villain_profile is None:
        villain_ok = (
            evaluation.villain_profile_normalization is None
            and identity.villain_raw_profile_sha256 == PREPARED_PROFILE_ABSENT
            and identity.villain_effective_profile_sha256 == PREPARED_PROFILE_ABSENT
        )
    else:
        villain = evaluation.villain_profile_normalization
        villain_ok = (
            type(villain) is PreparedProfileNormalization
            and _is_sha256(identity.villain_raw_profile_sha256)
            and _is_sha256(identity.villain_effective_profile_sha256)
            and identity.villain_raw_profile_sha256 == villain.raw_profile_sha256
            and identity.villain_effective_profile_sha256 == villain.effective_profile_sha256
        )
    return (
        request_content
        and m14_run_matches
        and continuity
        and public_contract
        and hero_ok
        and villain_ok
        and _is_sha256(identity.output_semantic_sha256)
    )


def run_prepared_two_street_orchestration(
    request: PreparedTwoStreetOrchestrationRequest,
) -> PreparedTwoStreetOrchestrationResult:
    """Run bounded M14 build then M15 fixed-Hero response evaluation.

    The function only composes a flat prepared abstract one-/two-street
    heads-up build and evaluation in memory.  Hero is completely fixed and the
    returned response is Villain's exact response to that Hero; an optional
    Villain profile is merely a chosen fixed comparison profile.

    It does not compute or prove an equilibrium or Nash result, compute an
    optimal Hero strategy, generate a Hero strategy or candidates, certify a
    solver or an external solver's correctness, claim profitability or
    solver-grade output, or provide real-money, gambling, bankroll, financial,
    or legal advice.  It
    does not process real cards, hands, ranges, card removal, equity generation,
    or raw solver exports.  It excludes arbitrary nested trees, three or more
    streets, multiway and side-pot play, full charts, large-scale solving, and
    real-opponent models.  It performs no filesystem or scenario JSON I/O and
    has no CLI, pipeline, manifest, report, export, or GUI integration.
    """

    if not _validate_request(request):
        return _failure(
            PreparedOrchestrationStatus.INVALID_INPUT,
            None,
            None,
            "request",
            "invalid orchestration request",
        )
    try:
        trace_count = 0
        for _ in _TRACE_VALUES:
            trace_count = _checked_add(
                trace_count, 1, request.orchestration_limits.max_trace_records
            )
        result_count = 0
        for increment in (1, 1, 1, 1, 1, trace_count):
            result_count = _checked_add(
                result_count,
                increment,
                request.orchestration_limits.max_result_records,
            )
    except OverflowError:
        return _failure(
            PreparedOrchestrationStatus.CAP_EXCEEDED,
            None,
            None,
            "orchestration-cap",
            "orchestration record cap exceeded",
        )

    try:
        builder_result = build_prepared_two_street_game(
            request.spec,
            request.raw_input_bytes,
            request.content_identity,
            request.builder_limits,
        )
    except Exception as exc:
        return _failure(
            PreparedOrchestrationStatus.INTERNAL_FAILURE,
            None,
            None,
            "builder",
            f"unexpected builder failure: {type(exc).__name__}",
        )
    valid, builder_status = _valid_builder_result(builder_result)
    if not valid:
        return _failure(
            PreparedOrchestrationStatus.INTERNAL_FAILURE,
            builder_status,
            None,
            "builder-result",
            "invalid builder result",
        )
    if builder_status is not PreparedTwoStreetStatus.SUCCESS:
        return _failure(
            _BUILD_STATUS_MAP[builder_status],
            builder_status,
            None,
            "builder",
            f"prepared builder returned {builder_status.value}",
        )
    build = builder_result.build

    try:
        evaluation_result = evaluate_prepared_two_street(
            build,
            request.hero_profile,
            request.villain_profile,
            request.evaluation_limits,
            request.evaluation_request,
        )
    except Exception as exc:
        return _failure(
            PreparedOrchestrationStatus.INTERNAL_FAILURE,
            PreparedTwoStreetStatus.SUCCESS,
            None,
            "evaluation",
            f"unexpected evaluation failure: {type(exc).__name__}",
            build,
        )
    valid, evaluation_status = _valid_evaluation_result(evaluation_result)
    if not valid:
        return _failure(
            PreparedOrchestrationStatus.INTERNAL_FAILURE,
            PreparedTwoStreetStatus.SUCCESS,
            evaluation_status,
            "evaluation-result",
            "invalid evaluation result",
            build,
        )
    if evaluation_status is not PreparedEvaluationStatus.SUCCESS:
        return _failure(
            _EVALUATION_STATUS_MAP[evaluation_status],
            PreparedTwoStreetStatus.SUCCESS,
            evaluation_status,
            "evaluation",
            f"prepared evaluator returned {evaluation_status.value}",
            build,
        )

    evaluation = evaluation_result.evaluation
    m15_identity = evaluation_result.identity
    try:
        if not _identity_chain_matches(request, build, evaluation, m15_identity):
            return _failure(
                PreparedOrchestrationStatus.BUILD_IDENTITY_MISMATCH,
                PreparedTwoStreetStatus.SUCCESS,
                PreparedEvaluationStatus.SUCCESS,
                "identity",
                "prepared identity chain mismatch",
                build,
            )
        run_payload = {
            "algorithm": PREPARED_TWO_STREET_ORCHESTRATION_ID,
            "output_semantic_id": PREPARED_TWO_STREET_ORCHESTRATION_OUTPUT_ID,
            "m14_identity": build.identity,
            "m15_identity": m15_identity,
            "effective_limits": {
                "builder": request.builder_limits,
                "evaluation": request.evaluation_limits,
                "orchestration": request.orchestration_limits,
            },
            "correspondence_mode": request.evaluation_request.correspondence_mode,
            "oracle_check": request.evaluation_request.oracle_check,
        }
        run_identity = _sha256(run_payload)
        canonical_trace = tuple(
            {
                "__type__": "PreparedOrchestrationTraceRecord",
                "fields": {"phase": phase, "subject": subject, "outcome": outcome},
            }
            for phase, subject, outcome in _TRACE_VALUES
        )
        output_hash = _sha256(
            {
                "algorithm": PREPARED_TWO_STREET_ORCHESTRATION_OUTPUT_ID,
                "run_identity": run_identity,
                "build_counts": build.counts,
                "builder_status": PreparedTwoStreetStatus.SUCCESS,
                "evaluation_status": PreparedEvaluationStatus.SUCCESS,
                "trace": canonical_trace,
            }
        )
    except _CanonicalNumericError:
        return _failure(
            PreparedOrchestrationStatus.NUMERIC_FAILURE,
            PreparedTwoStreetStatus.SUCCESS,
            PreparedEvaluationStatus.SUCCESS,
            "identity",
            "orchestration identity construction failed",
            build,
        )
    except Exception:
        return _failure(
            PreparedOrchestrationStatus.INTERNAL_FAILURE,
            PreparedTwoStreetStatus.SUCCESS,
            PreparedEvaluationStatus.SUCCESS,
            "identity",
            "orchestration identity construction failed",
            build,
        )
    if (
        request.expected_output_semantic_sha256 is not None
        and request.expected_output_semantic_sha256 != output_hash
    ):
        return _failure(
            PreparedOrchestrationStatus.NON_REPRODUCIBLE,
            PreparedTwoStreetStatus.SUCCESS,
            PreparedEvaluationStatus.SUCCESS,
            "output-identity",
            "orchestration output identity mismatch",
            build,
        )

    try:
        trace = tuple(
            PreparedOrchestrationTraceRecord(phase, subject, outcome)
            for phase, subject, outcome in _TRACE_VALUES
        )
        identity = PreparedTwoStreetOrchestrationIdentity(
            PREPARED_TWO_STREET_ORCHESTRATION_ID,
            PREPARED_TWO_STREET_ORCHESTRATION_OUTPUT_ID,
            build.identity,
            m15_identity,
            request.builder_limits,
            request.evaluation_limits,
            request.orchestration_limits,
            request.evaluation_request.correspondence_mode,
            request.evaluation_request.oracle_check,
            run_identity,
            output_hash,
        )
        run = PreparedTwoStreetOrchestrationRun(build, evaluation, trace)
        return PreparedTwoStreetOrchestrationResult(
            PreparedOrchestrationStatus.SUCCESS,
            PreparedTwoStreetStatus.SUCCESS,
            PreparedEvaluationStatus.SUCCESS,
            run,
            identity,
            None,
        )
    except Exception as exc:
        return _failure(
            PreparedOrchestrationStatus.INTERNAL_FAILURE,
            PreparedTwoStreetStatus.SUCCESS,
            PreparedEvaluationStatus.SUCCESS,
            "internal",
            f"unexpected orchestration failure: {type(exc).__name__}",
            build,
        )
