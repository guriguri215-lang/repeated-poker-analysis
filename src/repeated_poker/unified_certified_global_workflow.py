"""Unified public workflow for the M37, M38, and M39 certified consumers.

The module is deliberately an orchestration boundary.  It validates one
discriminated request, calls exactly one existing public analyzer, and retains
that analyzer's native result without reconstructing any poker, response,
objective, bound, or certificate semantics.

Finite candidate, grid, vertex, local-box, warm-start, sampling, truncation,
and fallback inputs are not part of this contract.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
from enum import Enum
from fractions import Fraction
import hashlib
import json
import math
import re
from typing import Any, Callable, Mapping

from . import aiof_preflop_certified_global as _m37
from . import certified_global_optimizer as _m36
from . import known_board_real_card_hu_certified_global as _m38
from . import three_player_certified_global as _m39


CONTRACT_VERSION = "m40-unified-certified-global-public-workflow-v1"
SCHEMA_VERSION = "m40-unified-certified-global-result-schema-v1"
WORKFLOW_VERSION = "exactly-one-native-certified-consumer-dispatch-v1"
CLAIM_SCOPE = (
    "selected-variant-identified-bounded-scalar-objective-"
    "specified-tolerance-global-maximum-only-v1"
)

REAL_CARD_AIOF_PREFLOP = "real_card_aiof_preflop_m37"
KNOWN_BOARD_REAL_CARD_HU_RIVER = "known_board_real_card_hu_river_m38"
ABSTRACT_THREE_PLAYER_RIVER = "abstract_three_player_river_m39"
KNOWN_BOARD_REAL_CARD_THREE_PLAYER_RIVER = (
    "known_board_real_card_three_player_river_m39"
)
SUPPORTED_VARIANTS = (
    REAL_CARD_AIOF_PREFLOP,
    KNOWN_BOARD_REAL_CARD_HU_RIVER,
    ABSTRACT_THREE_PLAYER_RIVER,
    KNOWN_BOARD_REAL_CARD_THREE_PLAYER_RIVER,
)

MAX_OUTPUT_RECORDS = 2_500_000
MAX_OUTPUT_BYTES = 320_000_000

_IDENTITY_RE = re.compile(r"(?:sha256:)?[0-9a-f]{64}\Z")
_SUCCESS_STATUSES = (
    _m36.CERTIFIED_GLOBAL,
    _m36.CERTIFIED_EPSILON_GLOBAL,
)

_REQUEST_TYPES = {
    REAL_CARD_AIOF_PREFLOP: _m37.AiofPreflopCertifiedGlobalRequest,
    KNOWN_BOARD_REAL_CARD_HU_RIVER: (
        _m38.KnownBoardRealCardHuCertifiedGlobalRequest
    ),
    ABSTRACT_THREE_PLAYER_RIVER: (
        _m39.AbstractThreePlayerCertifiedGlobalRequest
    ),
    KNOWN_BOARD_REAL_CARD_THREE_PLAYER_RIVER: (
        _m39.KnownBoardRealCardThreePlayerCertifiedGlobalRequest
    ),
}

_RESULT_TYPES = {
    REAL_CARD_AIOF_PREFLOP: _m37.AiofPreflopCertifiedGlobalResult,
    KNOWN_BOARD_REAL_CARD_HU_RIVER: (
        _m38.KnownBoardRealCardHuCertifiedGlobalResult
    ),
    ABSTRACT_THREE_PLAYER_RIVER: _m39.ThreePlayerCertifiedGlobalResult,
    KNOWN_BOARD_REAL_CARD_THREE_PLAYER_RIVER: (
        _m39.ThreePlayerCertifiedGlobalResult
    ),
}

CertifiedGlobalNestedRequest = (
    _m37.AiofPreflopCertifiedGlobalRequest
    | _m38.KnownBoardRealCardHuCertifiedGlobalRequest
    | _m39.AbstractThreePlayerCertifiedGlobalRequest
    | _m39.KnownBoardRealCardThreePlayerCertifiedGlobalRequest
)
CertifiedGlobalNestedResult = (
    _m37.AiofPreflopCertifiedGlobalResult
    | _m38.KnownBoardRealCardHuCertifiedGlobalResult
    | _m39.ThreePlayerCertifiedGlobalResult
)


@dataclass(frozen=True)
class UnifiedCertifiedGlobalLimits:
    """Caller-lowerable final-wrapper caps with immutable hard ceilings."""

    max_output_records: int = MAX_OUTPUT_RECORDS
    max_output_bytes: int = MAX_OUTPUT_BYTES

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class UnifiedCertifiedGlobalPins:
    """Optional raw or ``sha256:`` identities for the M40 boundary."""

    contract_identity: str | None = None
    schema_identity: str | None = None
    variant_identity: str | None = None
    nested_request_identity: str | None = None
    workflow_identity: str | None = None
    nested_analysis_identity: str | None = None
    analysis_identity: str | None = None


@dataclass(frozen=True)
class UnifiedCertifiedGlobalRequest:
    """One explicitly discriminated M40 request."""

    variant: str
    nested_request: CertifiedGlobalNestedRequest
    limits: UnifiedCertifiedGlobalLimits = UnifiedCertifiedGlobalLimits()
    pins: UnifiedCertifiedGlobalPins = UnifiedCertifiedGlobalPins()


@dataclass(frozen=True)
class UnifiedCertifiedGlobalError:
    """Sanitized exact failure phase and nested cause, if any."""

    phase: str
    message: str
    cause_status: str | None = None


@dataclass(frozen=True)
class UnifiedCertifiedGlobalFailureEvidence:
    """Completed nested work retained without exposing a partial success."""

    variant: str
    nested_status: str
    nested_error: Mapping[str, Any] | None
    optimizer_work_counters: Mapping[str, int]
    oracle_work_counters: Mapping[str, int]
    nested_result_identity: str | None
    native_failure_result: CertifiedGlobalNestedResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "nested_status": self.nested_status,
            "nested_error": (
                None
                if self.nested_error is None
                else dict(self.nested_error)
            ),
            "optimizer_work_counters": dict(
                self.optimizer_work_counters
            ),
            "oracle_work_counters": dict(self.oracle_work_counters),
            "nested_result_identity": self.nested_result_identity,
            "native_failure_result": (
                None
                if self.native_failure_result is None
                else self.native_failure_result.to_dict()
            ),
        }


@dataclass(frozen=True)
class UnifiedCertifiedGlobalPayload:
    """Lossless native certified result plus M40 provenance."""

    variant: str
    nested_request: CertifiedGlobalNestedRequest
    nested_result: CertifiedGlobalNestedResult
    identities: Mapping[str, str]
    limits: UnifiedCertifiedGlobalLimits

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "workflow_version": WORKFLOW_VERSION,
            "claim_scope": CLAIM_SCOPE,
            "variant": self.variant,
            "nested_request_type": _type_name(self.nested_request),
            "nested_result_type": _type_name(self.nested_result),
            "nested_result": self.nested_result.to_dict(),
            "identities": dict(self.identities),
            "limits": self.limits.to_dict(),
        }


@dataclass(frozen=True)
class UnifiedCertifiedGlobalResult:
    """Exclusive certified success or fail-closed M40 result."""

    status: str
    payload: UnifiedCertifiedGlobalPayload | None
    error: UnifiedCertifiedGlobalError | None
    failure_evidence: UnifiedCertifiedGlobalFailureEvidence | None
    optimizer_work_counters: Mapping[str, int]
    oracle_work_counters: Mapping[str, int]
    partial_result: bool = False

    def __post_init__(self) -> None:
        success = self.status in _SUCCESS_STATUSES
        if self.partial_result is not False:
            raise ValueError("M40 partial results are forbidden")
        if success != (
            self.payload is not None
            and self.error is None
            and self.failure_evidence is None
        ):
            raise ValueError("M40 success/failure wrapper is inconsistent")
        if not success and (self.payload is not None or self.error is None):
            raise ValueError("M40 failure requires payload null and one error")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "status": self.status,
            "payload": None if self.payload is None else self.payload.to_dict(),
            "error": None if self.error is None else asdict(self.error),
            "failure_evidence": (
                None
                if self.failure_evidence is None
                else self.failure_evidence.to_dict()
            ),
            "optimizer_work_counters": dict(
                self.optimizer_work_counters
            ),
            "oracle_work_counters": dict(self.oracle_work_counters),
            "partial_result": False,
        }
        _canonical_json_bytes(result)
        return result


class _InvalidRequest(ValueError):
    pass


class _StaleInput(ValueError):
    pass


class _OutputLimitReached(RuntimeError):
    pass


@dataclass(frozen=True)
class _DeferredJsonProjection:
    factory: Callable[[], Any]


@dataclass(frozen=True)
class _JsonObjectProjection:
    items: tuple[tuple[str, Any], ...]


def _type_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


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


def _public_value(value: Any) -> Any:
    """Return a deterministic public projection without private reconstruction."""

    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _InvalidRequest("nested request contains a non-finite float")
        return value
    if isinstance(value, Fraction):
        return (
            str(value.numerator)
            if value.denominator == 1
            else f"{value.numerator}/{value.denominator}"
        )
    if isinstance(value, Enum):
        return _public_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _public_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key in sorted(value):
            if type(key) is not str:
                raise _InvalidRequest(
                    "nested request mapping keys must be strings"
                )
            output[key] = _public_value(value[key])
        return output
    if isinstance(value, (tuple, list)):
        return [_public_value(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _public_value(to_dict())
    raise _InvalidRequest(
        f"nested request contains unsupported public value {_type_name(value)}"
    )


def _default_for_field(field: Any) -> Any:
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:
        return field.default_factory()
    return MISSING


def _validate_nested_limit_objects(value: Any, path: str) -> None:
    """Validate all public ``*limits`` objects before calling a consumer."""

    if not is_dataclass(value) or isinstance(value, type):
        return
    for field in fields(value):
        child = getattr(value, field.name)
        child_path = f"{path}.{field.name}"
        if field.name == "limits" or field.name.endswith("_limits"):
            default = _default_for_field(field)
            if default is MISSING or not is_dataclass(default):
                raise _InvalidRequest(
                    f"{child_path} lacks a declared dataclass contract"
                )
            if type(child) is not type(default):
                raise _InvalidRequest(
                    f"{child_path} must be {_type_name(default)}"
                )
            for limit_field in fields(child):
                limit_value = getattr(child, limit_field.name)
                default_limit_value = getattr(default, limit_field.name)
                if (
                    limit_value is None
                    and default_limit_value is None
                ):
                    continue
                if type(limit_value) is not int or limit_value < 0:
                    raise _InvalidRequest(
                        f"{child_path}.{limit_field.name} must be a "
                        "nonnegative plain int"
                    )
        if is_dataclass(child) and not isinstance(child, type):
            _validate_nested_limit_objects(child, child_path)


def _validate_limits(value: object) -> UnifiedCertifiedGlobalLimits:
    if type(value) is not UnifiedCertifiedGlobalLimits:
        raise _InvalidRequest(
            "limits must be UnifiedCertifiedGlobalLimits"
        )
    for name, ceiling in (
        ("max_output_records", MAX_OUTPUT_RECORDS),
        ("max_output_bytes", MAX_OUTPUT_BYTES),
    ):
        actual = getattr(value, name)
        if type(actual) is not int or not 1 <= actual <= ceiling:
            raise _InvalidRequest(
                f"limits.{name} must be a positive int at most {ceiling}"
            )
    return value


def _normalize_pin(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or _IDENTITY_RE.fullmatch(value) is None:
        raise _InvalidRequest(
            f"pins.{name} must be a raw or sha256-prefixed identity"
        )
    return value.removeprefix("sha256:")


def _check_pin(
    supplied: str | None,
    actual: str,
    name: str,
) -> None:
    normalized = _normalize_pin(supplied, name)
    if normalized is not None and normalized != actual:
        raise _StaleInput(f"pins.{name} does not match")


def _base_identities(
    *,
    variant: str,
    nested_request: Any,
) -> dict[str, str]:
    contract_identity = _identity(
        {
            "contract_version": CONTRACT_VERSION,
            "claim_scope": CLAIM_SCOPE,
        }
    )
    schema_identity = _identity(
        {
            "schema_version": SCHEMA_VERSION,
            "result_fields": [
                "status",
                "payload",
                "error",
                "failure_evidence",
                "optimizer_work_counters",
                "oracle_work_counters",
                "partial_result",
            ],
        }
    )
    variant_identity = _identity(
        {
            "variant": variant,
            "nested_request_type": _type_name(nested_request),
        }
    )
    return {
        "contract_identity": contract_identity,
        "schema_identity": schema_identity,
        "variant_identity": variant_identity,
    }


def _check_base_pins(
    pins: UnifiedCertifiedGlobalPins,
    identities: Mapping[str, str],
) -> None:
    for name in (
        "contract_identity",
        "schema_identity",
        "variant_identity",
    ):
        _check_pin(getattr(pins, name), identities[name], name)


def _extract_nested_identities(
    variant: str, native: Any
) -> tuple[str, str]:
    payload = native.payload
    if payload is None:
        raise RuntimeError("nested success has no payload")
    if variant == REAL_CARD_AIOF_PREFLOP:
        analysis = payload.preparation.analysis_identity
        request = _identity(
            {
                "variant": variant,
                "m37_analysis_identity": analysis,
                "note": "M37 has no distinct public request identity",
            }
        )
        return request, analysis
    if variant == KNOWN_BOARD_REAL_CARD_HU_RIVER:
        return (
            payload.preparation.request_identity,
            payload.preparation.analysis_identity,
        )
    request = payload.preparation.identities.get("request_identity")
    analysis = payload.preparation.identities.get("analysis_identity")
    if (
        type(request) is not str
        or _IDENTITY_RE.fullmatch(request) is None
        or type(analysis) is not str
        or _IDENTITY_RE.fullmatch(analysis) is None
    ):
        raise RuntimeError("M39 nested analysis identity is missing")
    return request, analysis


def _native_serializer(variant: str) -> Callable[[Any], str]:
    if variant == REAL_CARD_AIOF_PREFLOP:
        return _m37.exact_aiof_preflop_certified_global_json
    if variant == KNOWN_BOARD_REAL_CARD_HU_RIVER:
        return _m38.exact_known_board_real_card_hu_certified_global_json
    return _m39.exact_three_player_certified_global_json


def _dispatch(variant: str, nested_request: Any) -> Any:
    if variant == REAL_CARD_AIOF_PREFLOP:
        return _m37.analyze_aiof_preflop_certified_global(nested_request)
    if variant == KNOWN_BOARD_REAL_CARD_HU_RIVER:
        return _m38.analyze_known_board_real_card_hu_certified_global(
            nested_request
        )
    if variant == ABSTRACT_THREE_PLAYER_RIVER:
        return _m39.analyze_abstract_three_player_certified_global(
            nested_request
        )
    return (
        _m39.analyze_known_board_real_card_three_player_certified_global(
            nested_request
        )
    )


def _counter_dict(value: Any) -> dict[str, int]:
    output = value.to_dict()
    if type(output) is not dict:
        raise RuntimeError("nested work counters are not a dictionary")
    if any(type(key) is not str or type(item) is not int for key, item in output.items()):
        raise RuntimeError("nested work counters have an invalid schema")
    return dict(output)


def _nested_counters(native: Any) -> tuple[dict[str, int], dict[str, int]]:
    return (
        _counter_dict(native.optimizer_work_counters),
        _counter_dict(native.oracle_work_counters),
    )


def _native_error_dict(native: Any) -> dict[str, Any] | None:
    if native.error is None:
        return None
    return asdict(native.error)


def _failure(
    status: str,
    phase: str,
    message: str,
    *,
    cause_status: str | None = None,
    evidence: UnifiedCertifiedGlobalFailureEvidence | None = None,
    optimizer_counters: Mapping[str, int] | None = None,
    oracle_counters: Mapping[str, int] | None = None,
) -> UnifiedCertifiedGlobalResult:
    return UnifiedCertifiedGlobalResult(
        status=status,
        payload=None,
        error=UnifiedCertifiedGlobalError(
            phase=phase,
            message=message,
            cause_status=cause_status,
        ),
        failure_evidence=evidence,
        optimizer_work_counters=(
            {} if optimizer_counters is None else dict(optimizer_counters)
        ),
        oracle_work_counters=(
            {} if oracle_counters is None else dict(oracle_counters)
        ),
        partial_result=False,
    )


def _success_projection(
    *,
    status: str,
    variant: str,
    nested_request: Any,
    nested_result: Any,
    identities: Mapping[str, str],
    limits: UnifiedCertifiedGlobalLimits,
    optimizer_counters: Mapping[str, int],
    oracle_counters: Mapping[str, int],
) -> _JsonObjectProjection:
    payload = _JsonObjectProjection(
        (
            ("contract_version", CONTRACT_VERSION),
            ("schema_version", SCHEMA_VERSION),
            ("workflow_version", WORKFLOW_VERSION),
            ("claim_scope", CLAIM_SCOPE),
            ("variant", variant),
            ("nested_request_type", _type_name(nested_request)),
            ("nested_result_type", _type_name(nested_result)),
            (
                "nested_result",
                _DeferredJsonProjection(nested_result.to_dict),
            ),
            ("identities", dict(identities)),
            ("limits", limits.to_dict()),
        )
    )
    return _JsonObjectProjection(
        (
            ("status", status),
            ("payload", payload),
            ("error", None),
            ("failure_evidence", None),
            ("optimizer_work_counters", dict(optimizer_counters)),
            ("oracle_work_counters", dict(oracle_counters)),
            ("partial_result", False),
        )
    )


def _lower_bound_success_projection(
    *,
    variant: str,
    nested_request: Any,
    identities: Mapping[str, str],
    limits: UnifiedCertifiedGlobalLimits,
) -> _JsonObjectProjection:
    lower_identities = dict(identities)
    lower_identities.update(
        {
            "nested_request_identity": "0" * 64,
            "workflow_identity": "0" * 64,
            "nested_analysis_identity": "0" * 64,
            "nested_result_identity": "0" * 64,
            "analysis_identity": "0" * 64,
        }
    )
    return _JsonObjectProjection(
        (
            ("status", _m36.CERTIFIED_GLOBAL),
            (
                "payload",
                _JsonObjectProjection(
                    (
                        ("contract_version", CONTRACT_VERSION),
                        ("schema_version", SCHEMA_VERSION),
                        ("workflow_version", WORKFLOW_VERSION),
                        ("claim_scope", CLAIM_SCOPE),
                        ("variant", variant),
                        (
                            "nested_request_type",
                            _type_name(nested_request),
                        ),
                        ("nested_result_type", ""),
                        ("nested_result", None),
                        ("identities", lower_identities),
                        ("limits", limits.to_dict()),
                    )
                ),
            ),
            ("error", None),
            ("failure_evidence", None),
            ("optimizer_work_counters", {}),
            ("oracle_work_counters", {}),
            ("partial_result", False),
        )
    )


def _measure_projection(
    value: Any,
    *,
    max_records: int,
    max_bytes: int,
) -> tuple[int, int]:
    """Measure compact sorted JSON lazily and stop at the first exceeded cap."""

    record_count = 0
    byte_count = 0

    def add(fragment: str) -> None:
        nonlocal byte_count
        byte_count += len(fragment.encode("utf-8"))
        if byte_count > max_bytes:
            raise _OutputLimitReached(
                "max_output_bytes exceeded before success materialization"
            )

    def visit(child: Any) -> None:
        nonlocal record_count
        if isinstance(child, _DeferredJsonProjection):
            visit(child.factory())
            return
        record_count += 1
        if record_count > max_records:
            raise _OutputLimitReached(
                "max_output_records exceeded before success materialization"
            )
        if isinstance(child, _JsonObjectProjection):
            items = sorted(child.items, key=lambda item: item[0])
            add("{")
            for index, (key, item) in enumerate(items):
                if index:
                    add(",")
                add(
                    json.dumps(
                        key,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                )
                add(":")
                visit(item)
            add("}")
            return
        if isinstance(child, Mapping):
            add("{")
            for index, key in enumerate(sorted(child)):
                if type(key) is not str:
                    raise TypeError("public JSON keys must be strings")
                if index:
                    add(",")
                add(
                    json.dumps(
                        key,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                )
                add(":")
                visit(child[key])
            add("}")
            return
        if isinstance(child, (tuple, list)):
            add("[")
            for index, item in enumerate(child):
                if index:
                    add(",")
                visit(item)
            add("]")
            return
        add(
            json.dumps(
                child,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        )

    visit(value)
    return record_count, byte_count


def measure_unified_certified_global_result(
    result: UnifiedCertifiedGlobalResult,
) -> tuple[int, int]:
    """Return exact compact-JSON record and UTF-8 byte counts."""

    if type(result) is not UnifiedCertifiedGlobalResult:
        raise TypeError("result must be UnifiedCertifiedGlobalResult")
    value = result.to_dict()

    def count(child: Any) -> int:
        if isinstance(child, Mapping):
            return 1 + sum(count(item) for item in child.values())
        if isinstance(child, (tuple, list)):
            return 1 + sum(count(item) for item in child)
        return 1

    return count(value), len(_canonical_json_bytes(value))


def analyze_unified_certified_global(
    request: UnifiedCertifiedGlobalRequest,
) -> UnifiedCertifiedGlobalResult:
    """Validate, dispatch exactly once, and preserve one native result."""

    try:
        if type(request) is not UnifiedCertifiedGlobalRequest:
            raise _InvalidRequest(
                "request must be UnifiedCertifiedGlobalRequest"
            )
        limits = _validate_limits(request.limits)
        if type(request.pins) is not UnifiedCertifiedGlobalPins:
            raise _InvalidRequest(
                "pins must be UnifiedCertifiedGlobalPins"
            )
        for field in fields(request.pins):
            _normalize_pin(getattr(request.pins, field.name), field.name)
        if type(request.variant) is not str or request.variant not in _REQUEST_TYPES:
            raise _InvalidRequest(
                "variant must be one of the four supported M40 variants"
            )
        expected_request_type = _REQUEST_TYPES[request.variant]
        if type(request.nested_request) is not expected_request_type:
            raise _InvalidRequest(
                "variant and nested_request type do not match"
            )
        _validate_nested_limit_objects(
            request.nested_request, "nested_request"
        )
        identities = _base_identities(
            variant=request.variant,
            nested_request=request.nested_request,
        )
        _check_base_pins(request.pins, identities)
        _measure_projection(
            _lower_bound_success_projection(
                variant=request.variant,
                nested_request=request.nested_request,
                identities=identities,
                limits=limits,
            ),
            max_records=limits.max_output_records,
            max_bytes=limits.max_output_bytes,
        )
    except _StaleInput as exc:
        return _failure(
            _m36.STALE_INPUT,
            "request.pins",
            str(exc),
        )
    except (_InvalidRequest, TypeError, ValueError) as exc:
        return _failure(
            _m36.INVALID_INPUT,
            "request",
            str(exc),
        )
    except _OutputLimitReached as exc:
        return _failure(
            _m36.LIMIT_REACHED_NO_CERTIFICATE,
            "output.preflight",
            str(exc),
        )

    try:
        native = _dispatch(request.variant, request.nested_request)
        expected_result_type = _RESULT_TYPES[request.variant]
        if type(native) is not expected_result_type:
            raise RuntimeError("nested analyzer returned the wrong result type")
        optimizer_counters, oracle_counters = _nested_counters(native)
        native_json = _native_serializer(request.variant)(native)
        nested_result_identity = hashlib.sha256(
            native_json.encode("utf-8")
        ).hexdigest()
    except Exception as exc:
        return _failure(
            _m36.ORACLE_FAILURE,
            "dispatch",
            f"nested analyzer contract failure: {type(exc).__name__}",
        )

    if native.status not in _SUCCESS_STATUSES:
        nested_error = _native_error_dict(native)
        evidence = UnifiedCertifiedGlobalFailureEvidence(
            variant=request.variant,
            nested_status=native.status,
            nested_error=nested_error,
            optimizer_work_counters=optimizer_counters,
            oracle_work_counters=oracle_counters,
            nested_result_identity=nested_result_identity,
            native_failure_result=native,
        )
        phase = (
            "nested"
            if nested_error is None
            else f"nested.{nested_error['phase']}"
        )
        message = (
            "nested certified consumer failed without error metadata"
            if nested_error is None
            else str(nested_error["message"])
        )
        cause_status = (
            native.status
            if nested_error is None
            else nested_error.get("cause_status") or native.status
        )
        return _failure(
            native.status,
            phase,
            message,
            cause_status=cause_status,
            evidence=evidence,
            optimizer_counters=optimizer_counters,
            oracle_counters=oracle_counters,
        )

    try:
        if native.payload is None or native.error is not None or native.partial_result:
            raise RuntimeError("nested certified success is inconsistent")
        (
            nested_request_identity,
            nested_analysis_identity,
        ) = _extract_nested_identities(
            request.variant, native
        )
        if (
            _IDENTITY_RE.fullmatch(nested_request_identity) is None
            or _IDENTITY_RE.fullmatch(nested_analysis_identity) is None
        ):
            raise RuntimeError("nested request or analysis identity is malformed")
        workflow_identity = _identity(
            {
                "contract_identity": identities["contract_identity"],
                "schema_identity": identities["schema_identity"],
                "workflow_version": WORKFLOW_VERSION,
                "variant_identity": identities["variant_identity"],
                "nested_request_identity": nested_request_identity,
                "limits": limits.to_dict(),
            }
        )
        identities.update(
            {
                "nested_request_identity": nested_request_identity,
                "nested_analysis_identity": nested_analysis_identity,
                "nested_result_identity": nested_result_identity,
                "workflow_identity": workflow_identity,
            }
        )
        analysis_identity = _identity(
            {
                "workflow_identity": identities["workflow_identity"],
                "nested_analysis_identity": nested_analysis_identity,
                "nested_result_identity": nested_result_identity,
                "status": native.status,
            }
        )
        identities["analysis_identity"] = analysis_identity
        _check_pin(
            request.pins.nested_request_identity,
            nested_request_identity,
            "nested_request_identity",
        )
        _check_pin(
            request.pins.workflow_identity,
            workflow_identity,
            "workflow_identity",
        )
        _check_pin(
            request.pins.nested_analysis_identity,
            nested_analysis_identity,
            "nested_analysis_identity",
        )
        _check_pin(
            request.pins.analysis_identity,
            analysis_identity,
            "analysis_identity",
        )
        projection = _success_projection(
            status=native.status,
            variant=request.variant,
            nested_request=request.nested_request,
            nested_result=native,
            identities=identities,
            limits=limits,
            optimizer_counters=optimizer_counters,
            oracle_counters=oracle_counters,
        )
        _measure_projection(
            projection,
            max_records=limits.max_output_records,
            max_bytes=limits.max_output_bytes,
        )
    except _StaleInput as exc:
        evidence = UnifiedCertifiedGlobalFailureEvidence(
            variant=request.variant,
            nested_status=native.status,
            nested_error=None,
            optimizer_work_counters=optimizer_counters,
            oracle_work_counters=oracle_counters,
            nested_result_identity=nested_result_identity,
        )
        return _failure(
            _m36.STALE_INPUT,
            "request.pins",
            str(exc),
            cause_status=native.status,
            evidence=evidence,
            optimizer_counters=optimizer_counters,
            oracle_counters=oracle_counters,
        )
    except _OutputLimitReached as exc:
        evidence = UnifiedCertifiedGlobalFailureEvidence(
            variant=request.variant,
            nested_status=native.status,
            nested_error=None,
            optimizer_work_counters=optimizer_counters,
            oracle_work_counters=oracle_counters,
            nested_result_identity=nested_result_identity,
        )
        return _failure(
            _m36.LIMIT_REACHED_NO_CERTIFICATE,
            "output",
            str(exc),
            cause_status=native.status,
            evidence=evidence,
            optimizer_counters=optimizer_counters,
            oracle_counters=oracle_counters,
        )
    except Exception as exc:
        evidence = UnifiedCertifiedGlobalFailureEvidence(
            variant=request.variant,
            nested_status=native.status,
            nested_error=None,
            optimizer_work_counters=optimizer_counters,
            oracle_work_counters=oracle_counters,
            nested_result_identity=nested_result_identity,
        )
        return _failure(
            _m36.ORACLE_FAILURE,
            "output",
            f"unified success projection failed: {type(exc).__name__}",
            cause_status=native.status,
            evidence=evidence,
            optimizer_counters=optimizer_counters,
            oracle_counters=oracle_counters,
        )

    return UnifiedCertifiedGlobalResult(
        status=native.status,
        payload=UnifiedCertifiedGlobalPayload(
            variant=request.variant,
            nested_request=request.nested_request,
            nested_result=native,
            identities=dict(identities),
            limits=limits,
        ),
        error=None,
        failure_evidence=None,
        optimizer_work_counters=optimizer_counters,
        oracle_work_counters=oracle_counters,
        partial_result=False,
    )


def exact_unified_certified_global_json(
    result: UnifiedCertifiedGlobalResult,
) -> str:
    """Serialize one M40 result as deterministic strict one-line JSON."""

    if type(result) is not UnifiedCertifiedGlobalResult:
        raise TypeError("result must be UnifiedCertifiedGlobalResult")
    return _canonical_json_bytes(result.to_dict()).decode("utf-8")
