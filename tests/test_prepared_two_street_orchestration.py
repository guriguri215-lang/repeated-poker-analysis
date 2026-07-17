"""Focused acceptance tests for prepared two-street end-to-end orchestration."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

import repeated_poker
import repeated_poker.prepared_two_street_orchestration as orchestration_module
from repeated_poker.game import GameTree
from repeated_poker.prepared_two_street import (
    PREPARED_TWO_STREET_CONTRACT_VERSION,
    PreparedActionEvent,
    PreparedActionKind,
    PreparedActionOption,
    PreparedBucket,
    PreparedBuildError,
    PreparedChanceEdge,
    PreparedChanceEvent,
    PreparedContentIdentity,
    PreparedDataAttestation,
    PreparedDecisionMenu,
    PreparedHeadsUpChips,
    PreparedPlayer,
    PreparedRake,
    PreparedRoundCloseReason,
    PreparedShowdownValue,
    PreparedStreet,
    PreparedStreetCloseEvent,
    PreparedTransitionRow,
    PreparedTwoStreetBuildResult,
    PreparedTwoStreetLimits,
    PreparedTwoStreetSpec,
    PreparedTwoStreetStatus,
    build_prepared_two_street_game,
    prepared_public_history_id,
    prepared_semantic_sha256,
)
from repeated_poker.prepared_two_street_evaluation import (
    PREPARED_PROFILE_ABSENT,
    PreparedActionProbability,
    PreparedCorrespondenceMode,
    PreparedCorrespondenceStatus,
    PreparedEvaluationError,
    PreparedEvaluationLimits,
    PreparedEvaluationRequest,
    PreparedEvaluationResult,
    PreparedEvaluationStatus,
    PreparedPlayerProfile,
    PreparedProfileEntry,
    evaluate_prepared_two_street,
)
from repeated_poker.prepared_two_street_orchestration import *


def _attestation() -> PreparedDataAttestation:
    return PreparedDataAttestation(
        "M16 public fixture",
        "abstract buckets",
        "prepared conditional mass",
        "public actions/outcomes and own bucket only",
        True,
    )


def _passive(kind: PreparedActionKind) -> PreparedActionOption:
    return PreparedActionOption(kind, None, None, False)


def _aggressive(
    kind: PreparedActionKind, size: str, amount: float
) -> PreparedActionOption:
    return PreparedActionOption(kind, size, amount, False)


def _event(
    street: str,
    player: PreparedPlayer,
    kind: PreparedActionKind,
    size: str | None = None,
    amount: float | None = None,
) -> PreparedActionEvent:
    return PreparedActionEvent(street, player, kind, size, amount, False, True)


def _one_street_spec(
    *, check_share: float = 0.5, rake: float = 0.05
) -> PreparedTwoStreetSpec:
    street = "river"
    v_check = _event(street, PreparedPlayer.VILLAIN, PreparedActionKind.CHECK)
    h_check = _event(street, PreparedPlayer.HERO, PreparedActionKind.CHECK)
    v_bet = _event(
        street, PreparedPlayer.VILLAIN, PreparedActionKind.BET, "open-2", 2.0
    )
    h_call = _event(street, PreparedPlayer.HERO, PreparedActionKind.CALL)
    check_state = prepared_public_history_id(
        (
            v_check,
            h_check,
            PreparedStreetCloseEvent(
                street, PreparedRoundCloseReason.CHECK_CHECK
            ),
        )
    )
    call_state = prepared_public_history_id(
        (
            v_bet,
            h_call,
            PreparedStreetCloseEvent(street, PreparedRoundCloseReason.CALL),
        )
    )
    return PreparedTwoStreetSpec(
        PREPARED_TWO_STREET_CONTRACT_VERSION,
        _attestation(),
        PreparedHeadsUpChips(10.0, 10.0),
        PreparedHeadsUpChips(1.0, 1.0),
        PreparedRake(rake, 3.0),
        (PreparedStreet(street, "River", PreparedPlayer.VILLAIN, 2.0),),
        (PreparedBucket("H0", 1.0),),
        (PreparedBucket("V0", 1.0),),
        (
            PreparedDecisionMenu(
                prepared_public_history_id(()),
                street,
                PreparedPlayer.VILLAIN,
                (
                    _passive(PreparedActionKind.CHECK),
                    _aggressive(PreparedActionKind.BET, "open-2", 2.0),
                ),
            ),
            PreparedDecisionMenu(
                prepared_public_history_id((v_check,)),
                street,
                PreparedPlayer.HERO,
                (_passive(PreparedActionKind.CHECK),),
            ),
            PreparedDecisionMenu(
                prepared_public_history_id((v_bet,)),
                street,
                PreparedPlayer.HERO,
                (
                    _passive(PreparedActionKind.FOLD),
                    _passive(PreparedActionKind.CALL),
                ),
            ),
        ),
        None,
        (),
        (
            PreparedShowdownValue(check_state, "H0", "V0", check_share),
            PreparedShowdownValue(call_state, "H0", "V0", 0.7),
        ),
    )


def _two_street_spec() -> PreparedTwoStreetSpec:
    first, second, transition = "flop", "turn", "deal-turn"
    v1 = _event(first, PreparedPlayer.VILLAIN, PreparedActionKind.CHECK)
    h1 = _event(first, PreparedPlayer.HERO, PreparedActionKind.CHECK)
    close1 = PreparedStreetCloseEvent(
        first, PreparedRoundCloseReason.CHECK_CHECK
    )
    before = (v1, h1, close1)
    menus = [
        PreparedDecisionMenu(
            prepared_public_history_id(()),
            first,
            PreparedPlayer.VILLAIN,
            (_passive(PreparedActionKind.CHECK),),
        ),
        PreparedDecisionMenu(
            prepared_public_history_id((v1,)),
            first,
            PreparedPlayer.HERO,
            (_passive(PreparedActionKind.CHECK),),
        ),
    ]
    showdowns = []
    for outcome, share in (("red", 0.8), ("black", 0.2)):
        chance = PreparedChanceEvent(transition, outcome)
        h2 = _event(second, PreparedPlayer.HERO, PreparedActionKind.CHECK)
        v2 = _event(second, PreparedPlayer.VILLAIN, PreparedActionKind.CHECK)
        events = before + (chance,)
        menus.extend(
            (
                PreparedDecisionMenu(
                    prepared_public_history_id(events),
                    second,
                    PreparedPlayer.HERO,
                    (_passive(PreparedActionKind.CHECK),),
                ),
                PreparedDecisionMenu(
                    prepared_public_history_id(events + (h2,)),
                    second,
                    PreparedPlayer.VILLAIN,
                    (_passive(PreparedActionKind.CHECK),),
                ),
            )
        )
        showdowns.append(
            PreparedShowdownValue(
                prepared_public_history_id(
                    events
                    + (
                        h2,
                        v2,
                        PreparedStreetCloseEvent(
                            second, PreparedRoundCloseReason.CHECK_CHECK
                        ),
                    )
                ),
                "H0",
                "V0",
                share,
            )
        )
    row = PreparedTransitionRow(
        transition,
        prepared_public_history_id(before),
        "H0",
        "V0",
        (
            PreparedChanceEdge("red", "H0", "V0", 0.6),
            PreparedChanceEdge("black", "H0", "V0", 0.4),
        ),
    )
    return PreparedTwoStreetSpec(
        PREPARED_TWO_STREET_CONTRACT_VERSION,
        _attestation(),
        PreparedHeadsUpChips(10.0, 10.0),
        PreparedHeadsUpChips(1.0, 1.0),
        PreparedRake(0.1, None),
        (
            PreparedStreet(first, "First", PreparedPlayer.VILLAIN, 1.0),
            PreparedStreet(second, "Second", PreparedPlayer.HERO, 1.0),
        ),
        (PreparedBucket("H0", 1.0),),
        (PreparedBucket("V0", 1.0),),
        tuple(menus),
        transition,
        (row,),
        tuple(showdowns),
    )


def _initial_all_in_spec() -> PreparedTwoStreetSpec:
    first, second, transition = "flop", "turn", "deal-turn"
    before = (PreparedStreetCloseEvent(first, PreparedRoundCloseReason.ALL_IN_CALL),)
    return PreparedTwoStreetSpec(
        PREPARED_TWO_STREET_CONTRACT_VERSION,
        _attestation(),
        PreparedHeadsUpChips(1.0, 1.0),
        PreparedHeadsUpChips(1.0, 1.0),
        PreparedRake(0.0, None),
        (
            PreparedStreet(first, "First", PreparedPlayer.VILLAIN, 1.0),
            PreparedStreet(second, "Second", PreparedPlayer.HERO, 1.0),
        ),
        (PreparedBucket("H0", 1.0),),
        (PreparedBucket("V0", 1.0),),
        (),
        transition,
        (
            PreparedTransitionRow(
                transition,
                prepared_public_history_id(before),
                "H0",
                "V0",
                (
                    PreparedChanceEdge("low", "H0", "V0", 0.25),
                    PreparedChanceEdge("high", "H0", "V0", 0.75),
                ),
            ),
        ),
        (
            PreparedShowdownValue(
                prepared_public_history_id(
                    before + (PreparedChanceEvent(transition, "low"),)
                ),
                "H0",
                "V0",
                0.0,
            ),
            PreparedShowdownValue(
                prepared_public_history_id(
                    before + (PreparedChanceEvent(transition, "high"),)
                ),
                "H0",
                "V0",
                1.0,
            ),
        ),
    )


def _build(spec: PreparedTwoStreetSpec):
    raw = b"m16-public-fixture"
    identity = PreparedContentIdentity(
        hashlib.sha256(raw).hexdigest(), prepared_semantic_sha256(spec)
    )
    result = build_prepared_two_street_game(spec, raw, identity)
    assert result.status is PreparedTwoStreetStatus.SUCCESS, result.error
    return raw, identity, result.build


def _profile(build, player: PreparedPlayer, overrides=None) -> PreparedPlayerProfile:
    overrides = overrides or {}
    entries = []
    for artifact in build.information_sets:
        if artifact.key.player is not player:
            continue
        values = overrides.get(artifact.info_set_id, {})
        probabilities = tuple(
            PreparedActionProbability(
                label, values.get(label, 1.0 if index == 0 else 0.0)
            )
            for index, label in enumerate(artifact.legal_action_labels)
        )
        entries.append(PreparedProfileEntry(artifact.info_set_id, probabilities))
    return PreparedPlayerProfile(tuple(entries))


def _artifact(build, player: PreparedPlayer, actions: tuple[str, ...]):
    return next(
        item
        for item in build.information_sets
        if item.key.player is player and item.legal_action_labels == actions
    )


def _request(
    spec: PreparedTwoStreetSpec | None = None,
    *,
    villain: bool = False,
    hero_overrides=None,
    villain_overrides=None,
    evaluation_request: PreparedEvaluationRequest | None = None,
    builder_limits: PreparedTwoStreetLimits | None = None,
    evaluation_limits: PreparedEvaluationLimits | None = None,
    orchestration_limits: PreparedOrchestrationLimits | None = None,
    expected: str | None = None,
) -> PreparedTwoStreetOrchestrationRequest:
    spec = spec or _one_street_spec()
    raw, content_identity, build = _build(spec)
    return PreparedTwoStreetOrchestrationRequest(
        spec,
        raw,
        content_identity,
        _profile(build, PreparedPlayer.HERO, hero_overrides),
        _profile(build, PreparedPlayer.VILLAIN, villain_overrides)
        if villain
        else None,
        builder_limits or PreparedTwoStreetLimits(),
        evaluation_limits or PreparedEvaluationLimits(),
        evaluation_request or PreparedEvaluationRequest(),
        orchestration_limits or PreparedOrchestrationLimits(),
        expected,
    )


def _assert_success(result):
    assert result.status is PreparedOrchestrationStatus.SUCCESS
    assert result.run is not None and result.identity is not None
    assert result.error is None


def _assert_failure(result, status):
    assert result.status is status
    assert result.run is None and result.identity is None
    assert result.error is not None
    assert result.error.phase and len(result.error.phase) <= 64
    assert result.error.message and len(result.error.message) <= 500


def _independent_value(value):
    if isinstance(value, Enum):
        return value.value
    if type(value) is float:
        assert math.isfinite(value)
        return {"__floathex__": value.hex()}
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is bytes:
        return {"__bytes_hex__": value.hex()}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": type(value).__name__,
            "fields": {
                item.name: _independent_value(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if type(value) in (tuple, list):
        return [_independent_value(item) for item in value]
    if isinstance(value, dict):
        assert all(type(key) is str for key in value)
        return {key: _independent_value(value[key]) for key in sorted(value)}
    raise AssertionError(type(value))


def _independent_sha(value) -> str:
    raw = json.dumps(
        _independent_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def test_exact_public_api_constants_all_enum_fields_defaults_and_signature():
    assert PREPARED_TWO_STREET_ORCHESTRATION_ID == "prepared-two-street-orchestration-v1"
    assert PREPARED_TWO_STREET_ORCHESTRATION_OUTPUT_ID == "prepared-two-street-orchestration-output-sha256-v1"
    assert orchestration_module.__all__ == [
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
    assert [(item.name, item.value) for item in PreparedOrchestrationStatus] == [
        (name, name)
        for name in (
            "SUCCESS INVALID_INPUT BUILD_FAILURE BUILD_IDENTITY_MISMATCH "
            "EVALUATION_INPUT_FAILURE EVALUATION_CORE_FAILURE CAP_EXCEEDED "
            "NUMERIC_FAILURE NON_REPRODUCIBLE UNSUPPORTED_DOWNSTREAM INTERNAL_FAILURE"
        ).split()
    ]
    expected_fields = {
        PreparedOrchestrationLimits: ["max_trace_records", "max_result_records"],
        PreparedTwoStreetOrchestrationRequest: [
            "spec", "raw_input_bytes", "content_identity", "hero_profile",
            "villain_profile", "builder_limits", "evaluation_limits",
            "evaluation_request", "orchestration_limits",
            "expected_output_semantic_sha256",
        ],
        PreparedOrchestrationTraceRecord: ["phase", "subject", "outcome"],
        PreparedTwoStreetOrchestrationRun: ["build", "evaluation", "trace"],
        PreparedTwoStreetOrchestrationIdentity: [
            "orchestration_id", "output_semantic_id", "m14_identity",
            "m15_identity", "effective_builder_limits",
            "effective_evaluation_limits", "effective_orchestration_limits",
            "correspondence_mode", "oracle_check", "run_identity",
            "output_semantic_sha256",
        ],
        PreparedOrchestrationError: [
            "phase", "message", "build_identity", "build_counts"
        ],
        PreparedTwoStreetOrchestrationResult: [
            "status", "builder_status", "evaluation_status", "run", "identity", "error"
        ],
    }
    for cls, names in expected_fields.items():
        assert [item.name for item in fields(cls)] == names
        assert cls.__dataclass_params__.frozen
    assert PreparedOrchestrationLimits() == PreparedOrchestrationLimits(16, 64)
    with pytest.raises(FrozenInstanceError):
        PreparedOrchestrationLimits().max_trace_records = 1
    assert str(inspect.signature(run_prepared_two_street_orchestration)) == (
        "(request: 'PreparedTwoStreetOrchestrationRequest') -> "
        "'PreparedTwoStreetOrchestrationResult'"
    )


def test_top_level_package_does_not_export_orchestration_api():
    for name in orchestration_module.__all__:
        assert name not in repeated_poker.__all__
        assert not hasattr(repeated_poker, name)


def test_source_uses_only_public_m14_m15_imports_and_standard_library():
    source = Path(orchestration_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_modules = {
        "game", "fixed_profile", "exact_response", "payoffs", "pipeline",
        "scenario_pipeline", "stt_pushfold_pipeline", "run_manifest",
        "analysis_report", "report_export", "scenario_io",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {
            "prepared_two_street", "prepared_two_street_evaluation"
        }:
            assert all(not alias.name.startswith("_") for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            assert node.module not in forbidden_modules


def test_module_and_public_function_claim_boundaries():
    text = (orchestration_module.__doc__ or "") + (
        run_prepared_two_street_orchestration.__doc__ or ""
    )
    for phrase in (
        "fixed", "not an equilibrium", "optimal Hero", "real cards",
        "solver", "profitability", "advice", "scenario JSON", "CLI",
    ):
        assert phrase.lower() in text.lower()


def test_success_calls_builder_then_evaluator_once_in_exact_order(monkeypatch):
    events = []
    real_builder = orchestration_module.build_prepared_two_street_game
    real_evaluator = orchestration_module.evaluate_prepared_two_street
    monkeypatch.setattr(orchestration_module, "build_prepared_two_street_game", lambda *a: (events.append("builder"), real_builder(*a))[1])
    monkeypatch.setattr(orchestration_module, "evaluate_prepared_two_street", lambda *a: (events.append("evaluation"), real_evaluator(*a))[1])
    _assert_success(run_prepared_two_street_orchestration(_request()))
    assert events == ["builder", "evaluation"]


def test_request_fields_are_forwarded_by_identity_without_reassembly_or_clamp(monkeypatch):
    request = _request(villain=True)
    real_builder = orchestration_module.build_prepared_two_street_game
    real_evaluator = orchestration_module.evaluate_prepared_two_street
    seen = {}
    def builder(*args):
        seen["builder"] = args
        return real_builder(*args)
    def evaluator(*args):
        seen["evaluation"] = args
        return real_evaluator(*args)
    monkeypatch.setattr(orchestration_module, "build_prepared_two_street_game", builder)
    monkeypatch.setattr(orchestration_module, "evaluate_prepared_two_street", evaluator)
    _assert_success(run_prepared_two_street_orchestration(request))
    assert all(left is right for left, right in zip(seen["builder"], (request.spec, request.raw_input_bytes, request.content_identity, request.builder_limits)))
    assert all(left is right for left, right in zip(seen["evaluation"][1:], (request.hero_profile, request.villain_profile, request.evaluation_limits, request.evaluation_request)))


def test_success_returns_exact_trace_identity_references_and_no_error():
    result = run_prepared_two_street_orchestration(_request())
    _assert_success(result)
    assert [(row.phase, row.subject, row.outcome) for row in result.run.trace] == [
        ("orchestration", "input", "PASS"),
        ("builder", "completed", "PASS"),
        ("evaluation", "completed", "PASS"),
        ("orchestration", "identity", "PASS"),
    ]
    assert result.identity.m14_identity is result.run.build.identity
    assert result.identity.m15_identity.output_semantic_sha256 == result.run.evaluation.hero_profile_normalization.raw_profile_sha256 or result.identity.m15_identity is not None


def test_success_reuses_nested_objects_without_counting_copying_or_materializing(monkeypatch):
    request = _request()
    captured = {}
    real_eval = orchestration_module.evaluate_prepared_two_street
    def evaluator(build, *args):
        result = real_eval(build, *args)
        captured["build"] = build
        captured["evaluation"] = result.evaluation
        return result
    monkeypatch.setattr(orchestration_module, "evaluate_prepared_two_street", evaluator)
    result = run_prepared_two_street_orchestration(request)
    _assert_success(result)
    assert result.run.build is captured["build"]
    assert result.run.evaluation is captured["evaluation"]
    assert len(result.run.trace) == 4


@pytest.mark.parametrize("status", [item for item in PreparedTwoStreetStatus if item is not PreparedTwoStreetStatus.SUCCESS])
def test_every_m14_non_success_status_maps_exactly_and_evaluator_is_not_called(monkeypatch, status):
    calls = []
    monkeypatch.setattr(orchestration_module, "build_prepared_two_street_game", lambda *a: PreparedTwoStreetBuildResult(status, None, PreparedBuildError("failure", "build")))
    monkeypatch.setattr(orchestration_module, "evaluate_prepared_two_street", lambda *a: calls.append(a))
    result = run_prepared_two_street_orchestration(_request())
    expected = {
        PreparedTwoStreetStatus.CAP_EXCEEDED: PreparedOrchestrationStatus.CAP_EXCEEDED,
        PreparedTwoStreetStatus.NUMERIC_FAILURE: PreparedOrchestrationStatus.NUMERIC_FAILURE,
        PreparedTwoStreetStatus.NON_REPRODUCIBLE: PreparedOrchestrationStatus.NON_REPRODUCIBLE,
        PreparedTwoStreetStatus.UNSUPPORTED_DOWNSTREAM: PreparedOrchestrationStatus.UNSUPPORTED_DOWNSTREAM,
    }.get(status, PreparedOrchestrationStatus.BUILD_FAILURE)
    _assert_failure(result, expected)
    assert result.builder_status is status and result.evaluation_status is None and calls == []


@pytest.mark.parametrize("status", [item for item in PreparedEvaluationStatus if item is not PreparedEvaluationStatus.SUCCESS])
def test_every_m15_non_success_status_maps_exactly_and_keeps_only_build_provenance(monkeypatch, status):
    monkeypatch.setattr(orchestration_module, "evaluate_prepared_two_street", lambda *a: PreparedEvaluationResult(status, None, None, PreparedEvaluationError("failure", "evaluation")))
    result = run_prepared_two_street_orchestration(_request())
    expected = {
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
    }[status]
    _assert_failure(result, expected)
    assert result.builder_status is PreparedTwoStreetStatus.SUCCESS
    assert result.evaluation_status is status
    assert result.error.build_identity is not None and result.error.build_counts is not None


def test_every_outer_non_success_obeys_unified_no_partial_invariant(monkeypatch):
    cases = [
        PreparedTwoStreetOrchestrationRequest,
        replace(_request(), orchestration_limits=PreparedOrchestrationLimits(3, 64)),
        replace(_request(), expected_output_semantic_sha256="0" * 64),
    ]
    results = [run_prepared_two_street_orchestration(item) for item in cases]
    assert {item.status for item in results} == {
        PreparedOrchestrationStatus.INVALID_INPUT,
        PreparedOrchestrationStatus.CAP_EXCEEDED,
        PreparedOrchestrationStatus.NON_REPRODUCIBLE,
    }
    for result in results:
        _assert_failure(result, result.status)


def test_build_success_evaluation_failure_keeps_only_identity_and_counts(monkeypatch):
    monkeypatch.setattr(orchestration_module, "evaluate_prepared_two_street", lambda *a: PreparedEvaluationResult(PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE, None, None, PreparedEvaluationError("failure", "evaluation")))
    result = run_prepared_two_street_orchestration(_request())
    _assert_failure(result, PreparedOrchestrationStatus.EVALUATION_CORE_FAILURE)
    assert set(vars(result.error)) == {"phase", "message", "build_identity", "build_counts"}
    assert not any(isinstance(value, GameTree) for value in vars(result.error).values())


def test_malformed_builder_and_evaluator_results_fail_closed_as_internal_failure(monkeypatch):
    monkeypatch.setattr(orchestration_module, "build_prepared_two_street_game", lambda *a: object())
    result = run_prepared_two_street_orchestration(_request())
    _assert_failure(result, PreparedOrchestrationStatus.INTERNAL_FAILURE)


@pytest.mark.parametrize("builder_result", [
    PreparedTwoStreetBuildResult("SUCCESS", None, None),
    PreparedTwoStreetBuildResult(PreparedTwoStreetStatus.SUCCESS, None, None),
    PreparedTwoStreetBuildResult(
        PreparedTwoStreetStatus.INVALID_INPUT,
        SimpleNamespace(),
        PreparedBuildError("failure", "builder"),
    ),
    PreparedTwoStreetBuildResult(
        PreparedTwoStreetStatus.INVALID_INPUT, None, None
    ),
])
def test_malformed_builder_shapes_preserve_only_recognizable_status(
    monkeypatch, builder_result
):
    monkeypatch.setattr(
        orchestration_module,
        "build_prepared_two_street_game",
        lambda *args: builder_result,
    )
    result = run_prepared_two_street_orchestration(_request())
    _assert_failure(result, PreparedOrchestrationStatus.INTERNAL_FAILURE)
    expected = (
        builder_result.status
        if type(builder_result.status) is PreparedTwoStreetStatus
        else None
    )
    assert result.builder_status is expected


@pytest.mark.parametrize("evaluation_result", [
    PreparedEvaluationResult("SUCCESS", None, None, None),
    PreparedEvaluationResult(
        PreparedEvaluationStatus.SUCCESS, None, None, None
    ),
    PreparedEvaluationResult(
        PreparedEvaluationStatus.INVALID_INPUT,
        SimpleNamespace(),
        None,
        PreparedEvaluationError("failure", "evaluation"),
    ),
    PreparedEvaluationResult(
        PreparedEvaluationStatus.INVALID_INPUT, None, None, None
    ),
])
def test_malformed_evaluator_shapes_preserve_only_recognizable_status(
    monkeypatch, evaluation_result
):
    monkeypatch.setattr(
        orchestration_module,
        "evaluate_prepared_two_street",
        lambda *args: evaluation_result,
    )
    result = run_prepared_two_street_orchestration(_request())
    _assert_failure(result, PreparedOrchestrationStatus.INTERNAL_FAILURE)
    expected = (
        evaluation_result.status
        if type(evaluation_result.status) is PreparedEvaluationStatus
        else None
    )
    assert result.builder_status is PreparedTwoStreetStatus.SUCCESS
    assert result.evaluation_status is expected
    monkeypatch.setattr(orchestration_module, "build_prepared_two_street_game", build_prepared_two_street_game)
    monkeypatch.setattr(orchestration_module, "evaluate_prepared_two_street", lambda *a: SimpleNamespace(status="SUCCESS"))
    result = run_prepared_two_street_orchestration(_request())
    _assert_failure(result, PreparedOrchestrationStatus.INTERNAL_FAILURE)


def test_optional_villain_absent_and_present_paths():
    absent = run_prepared_two_street_orchestration(_request())
    present = run_prepared_two_street_orchestration(_request(villain=True))
    _assert_success(absent); _assert_success(present)
    assert absent.identity.m15_identity.villain_raw_profile_sha256 == PREPARED_PROFILE_ABSENT
    assert absent.run.evaluation.fixed_profile_value is None
    assert present.identity.m15_identity.villain_raw_profile_sha256 != PREPARED_PROFILE_ABSENT
    assert present.run.evaluation.fixed_profile_value is not None


def test_default_dp_and_explicit_enumerate_oracle_paths():
    default = run_prepared_two_street_orchestration(_request())
    oracle = run_prepared_two_street_orchestration(_request(evaluation_request=PreparedEvaluationRequest(method="enumerate", oracle_check=True)))
    _assert_success(default); _assert_success(oracle)
    assert default.identity.m15_identity.exact_response_method == "dp"
    assert oracle.identity.m15_identity.exact_response_method == "enumerate"


def test_representative_counts_action_sets_variation_off_path_and_full_modes():
    spec = _one_street_spec(check_share=0.0, rake=0.0)
    _, _, build = _build(spec)
    facing = _artifact(build, PreparedPlayer.HERO, ("fold", "call"))
    overrides = {facing.info_set_id: {"fold": 1.0, "call": 0.0}}
    compact = run_prepared_two_street_orchestration(_request(spec, hero_overrides=overrides))
    full = run_prepared_two_street_orchestration(_request(spec, hero_overrides=overrides, evaluation_request=PreparedEvaluationRequest(correspondence_mode=PreparedCorrespondenceMode.FULL)))
    _assert_success(compact); _assert_success(full)
    exact = compact.run.evaluation.exact_response
    assert exact.num_best_response_strategies == 2
    assert len(exact.representative_pure_response.assignments) == 1
    assert exact.full_correspondence is None
    assert full.run.evaluation.exact_response.full_correspondence_status is PreparedCorrespondenceStatus.MATERIALIZED
    assert len(full.run.evaluation.exact_response.full_correspondence) == 2


@pytest.mark.parametrize("profile,eval_request,status", [
    (PreparedPlayerProfile(()), PreparedEvaluationRequest(), PreparedOrchestrationStatus.EVALUATION_INPUT_FAILURE),
    (None, PreparedEvaluationRequest(method="unknown"), PreparedOrchestrationStatus.UNSUPPORTED_DOWNSTREAM),
    (None, PreparedEvaluationRequest(downstream_request="pipeline"), PreparedOrchestrationStatus.UNSUPPORTED_DOWNSTREAM),
])
def test_profile_and_request_fault_matrix_maps_through_m15(profile, eval_request, status):
    base = _request(evaluation_request=eval_request)
    if profile is not None:
        base = replace(base, hero_profile=profile)
    result = run_prepared_two_street_orchestration(base)
    _assert_failure(result, status)


@pytest.mark.parametrize("field_name", [
    "m14_contract_version", "m14_builder_id", "m14_action_label_id",
    "m14_normalization_id", "m14_information_key_id", "m14_raw_sha256",
    "m14_prepared_semantic_sha256", "m14_ordered_tree_sha256", "m14_run_identity",
    "hero_raw_profile_sha256", "hero_effective_profile_sha256",
    "evaluation_adapter_id", "profile_normalization_id", "profile_raw_id",
    "profile_effective_id", "output_semantic_id", "exact_response_method",
    "profile_tolerance_hex", "evaluation_tolerance_hex", "effective_limits",
    "correspondence_mode", "oracle_check", "output_semantic_sha256",
])
def test_success_cross_checks_all_public_m14_and_m15_identity_fields(monkeypatch, field_name):
    real = orchestration_module.evaluate_prepared_two_street
    def evaluator(*args):
        result = real(*args)
        value = getattr(result.identity, field_name)
        if field_name == "effective_limits":
            bad = replace(value, max_result_records=value.max_result_records - 1)
        elif field_name == "correspondence_mode":
            bad = PreparedCorrespondenceMode.FULL
        elif field_name == "oracle_check":
            bad = not value
        elif field_name == "output_semantic_sha256":
            bad = "fault"
        else:
            bad = ("f" * 64) if isinstance(value, str) and len(value) == 64 else "fault"
        return replace(result, identity=replace(result.identity, **{field_name: bad}))
    monkeypatch.setattr(orchestration_module, "evaluate_prepared_two_street", evaluator)
    result = run_prepared_two_street_orchestration(_request())
    _assert_failure(result, PreparedOrchestrationStatus.BUILD_IDENTITY_MISMATCH)


@pytest.mark.parametrize("field_name,bad", [
    ("contract_version", "fault"),
    ("builder_id", "fault"),
    ("action_label_id", "fault"),
    ("normalization_id", "fault"),
    ("information_key_id", "fault"),
    ("raw_sha256", "fault"),
    ("semantic_sha256", "fault"),
    ("ordered_tree_sha256", "fault"),
    ("run_identity", "fault"),
])
def test_success_cross_checks_each_public_m14_identity_field(
    monkeypatch, field_name, bad
):
    request = _request()
    baseline_build = build_prepared_two_street_game(
        request.spec,
        request.raw_input_bytes,
        request.content_identity,
        request.builder_limits,
    )
    baseline_evaluation = evaluate_prepared_two_street(
        baseline_build.build,
        request.hero_profile,
        request.villain_profile,
        request.evaluation_limits,
        request.evaluation_request,
    )
    faulty_identity = replace(
        baseline_build.build.identity, **{field_name: bad}
    )
    faulty_build = replace(baseline_build.build, identity=faulty_identity)
    monkeypatch.setattr(
        orchestration_module,
        "build_prepared_two_street_game",
        lambda *args: replace(baseline_build, build=faulty_build),
    )
    monkeypatch.setattr(
        orchestration_module,
        "evaluate_prepared_two_street",
        lambda *args: baseline_evaluation,
    )
    result = run_prepared_two_street_orchestration(request)
    _assert_failure(result, PreparedOrchestrationStatus.BUILD_IDENTITY_MISMATCH)


@pytest.mark.parametrize(
    "field_name",
    ["villain_raw_profile_sha256", "villain_effective_profile_sha256"],
)
def test_present_villain_identity_fields_are_cross_checked(monkeypatch, field_name):
    real = orchestration_module.evaluate_prepared_two_street
    def evaluator(*args):
        result = real(*args)
        return replace(
            result,
            identity=replace(result.identity, **{field_name: "fault"}),
        )
    monkeypatch.setattr(orchestration_module, "evaluate_prepared_two_street", evaluator)
    result = run_prepared_two_street_orchestration(_request(villain=True))
    _assert_failure(result, PreparedOrchestrationStatus.BUILD_IDENTITY_MISMATCH)


def test_identity_fault_returns_build_identity_mismatch_without_partial_payload(monkeypatch):
    real = orchestration_module.evaluate_prepared_two_street
    monkeypatch.setattr(orchestration_module, "evaluate_prepared_two_street", lambda *a: replace(real(*a), identity=replace(real(*a).identity, m14_builder_id="fault")))
    result = run_prepared_two_street_orchestration(_request())
    _assert_failure(result, PreparedOrchestrationStatus.BUILD_IDENTITY_MISMATCH)
    assert result.builder_status is PreparedTwoStreetStatus.SUCCESS
    assert result.evaluation_status is PreparedEvaluationStatus.SUCCESS
    assert result.error.build_identity is not None


@pytest.mark.parametrize(
    "fault,expected",
    [
        (float("inf"), PreparedOrchestrationStatus.NUMERIC_FAILURE),
        (object(), PreparedOrchestrationStatus.INTERNAL_FAILURE),
    ],
)
def test_canonical_numeric_and_type_failures_are_bounded_no_partial(
    monkeypatch, fault, expected
):
    request = _request()
    baseline_build = build_prepared_two_street_game(
        request.spec,
        request.raw_input_bytes,
        request.content_identity,
        request.builder_limits,
    )
    baseline_evaluation = evaluate_prepared_two_street(
        baseline_build.build,
        request.hero_profile,
        request.villain_profile,
        request.evaluation_limits,
        request.evaluation_request,
    )
    counts = replace(baseline_build.build.counts, root_matchups=fault)
    faulty_build = replace(baseline_build.build, counts=counts)
    monkeypatch.setattr(
        orchestration_module,
        "build_prepared_two_street_game",
        lambda *args: replace(baseline_build, build=faulty_build),
    )
    monkeypatch.setattr(
        orchestration_module,
        "evaluate_prepared_two_street",
        lambda *args: baseline_evaluation,
    )
    result = run_prepared_two_street_orchestration(request)
    _assert_failure(result, expected)
    assert result.error.phase == "identity"
    assert result.error.message == "orchestration identity construction failed"
    assert result.builder_status is PreparedTwoStreetStatus.SUCCESS
    assert result.evaluation_status is PreparedEvaluationStatus.SUCCESS


def test_three_effective_limit_namespaces_are_distinct_and_exact():
    builder = replace(PreparedTwoStreetLimits(), max_root_matchups=9_999)
    evaluation = replace(PreparedEvaluationLimits(), max_result_records=99_999)
    wrapper = PreparedOrchestrationLimits(15, 63)
    request = _request(builder_limits=builder, evaluation_limits=evaluation, orchestration_limits=wrapper)
    result = run_prepared_two_street_orchestration(request)
    _assert_success(result)
    assert result.identity.effective_builder_limits is builder
    assert result.identity.effective_evaluation_limits is evaluation
    assert result.identity.effective_orchestration_limits is wrapper


def test_m15_and_m16_expected_witnesses_are_independent():
    m15 = run_prepared_two_street_orchestration(_request(evaluation_request=PreparedEvaluationRequest(expected_output_semantic_sha256="0" * 64)))
    m16 = run_prepared_two_street_orchestration(_request(expected="0" * 64))
    _assert_failure(m15, PreparedOrchestrationStatus.NON_REPRODUCIBLE)
    _assert_failure(m16, PreparedOrchestrationStatus.NON_REPRODUCIBLE)
    assert m15.evaluation_status is PreparedEvaluationStatus.NON_REPRODUCIBLE
    assert m16.evaluation_status is PreparedEvaluationStatus.SUCCESS


def test_m16_expected_output_hash_accepts_exact_and_rejects_one_hex_fault():
    initial = run_prepared_two_street_orchestration(_request())
    _assert_success(initial)
    digest = initial.identity.output_semantic_sha256
    accepted = run_prepared_two_street_orchestration(_request(expected=digest))
    rejected = run_prepared_two_street_orchestration(_request(expected=("0" if digest[0] != "0" else "1") + digest[1:]))
    _assert_success(accepted)
    _assert_failure(rejected, PreparedOrchestrationStatus.NON_REPRODUCIBLE)


def test_same_request_same_runtime_repeats_all_statuses_identities_trace_and_hashes_exactly():
    request = _request()
    left = run_prepared_two_street_orchestration(request)
    right = run_prepared_two_street_orchestration(request)
    assert left == right


def test_independent_serializer_matches_exact_run_and_output_payload_schema():
    request = _request()
    result = run_prepared_two_street_orchestration(request)
    _assert_success(result)
    run_payload = {
        "algorithm": PREPARED_TWO_STREET_ORCHESTRATION_ID,
        "output_semantic_id": PREPARED_TWO_STREET_ORCHESTRATION_OUTPUT_ID,
        "m14_identity": result.run.build.identity,
        "m15_identity": result.identity.m15_identity,
        "effective_limits": {"builder": request.builder_limits, "evaluation": request.evaluation_limits, "orchestration": request.orchestration_limits},
        "correspondence_mode": request.evaluation_request.correspondence_mode,
        "oracle_check": request.evaluation_request.oracle_check,
    }
    run_hash = _independent_sha(run_payload)
    output_hash = _independent_sha({
        "algorithm": PREPARED_TWO_STREET_ORCHESTRATION_OUTPUT_ID,
        "run_identity": run_hash,
        "build_counts": result.run.build.counts,
        "builder_status": PreparedTwoStreetStatus.SUCCESS,
        "evaluation_status": PreparedEvaluationStatus.SUCCESS,
        "trace": result.run.trace,
    })
    assert run_hash == result.identity.run_identity
    assert output_hash == result.identity.output_semantic_sha256


def test_pinned_python_310_313_identity_and_numeric_fixture():
    result = run_prepared_two_street_orchestration(_request(villain=True))
    _assert_success(result)
    assert result.run.build.identity.run_identity == "9341738c0c60597afda6ffa04ccb6135d3038a5029a9c461688a7729254247e0"
    assert result.identity.m15_identity.hero_raw_profile_sha256 == "71dcfb4951ccacedafa6868730f3adf0b90ce3b5534252d1410f26739c24e944"
    assert result.identity.m15_identity.hero_effective_profile_sha256 == "3d4f4879a5fd2102cf50eef9fce9386d3679d983b988521445fd5532ef70b718"
    assert result.identity.m15_identity.output_semantic_sha256 == "8bb20188dab89e6a3cb6dc2b436516346009d2c842bc2f56ae5c66c4b49f91f1"
    assert result.identity.run_identity == "0f44ffb0f8b4ad016a050d2553501f591f1d08cf012c4236939783b61c1bda19"
    assert result.identity.output_semantic_sha256 == "f2fa83821ffcc67f35f616a67b03615ee809034a42e79e5c4468ef62d6ecdf15"
    value = result.run.evaluation.fixed_profile_value
    assert abs(value.hero_ev + value.villain_ev + value.house_rake) <= 1e-9


@pytest.mark.parametrize("limits,expected", [
    (object(), PreparedOrchestrationStatus.INVALID_INPUT),
    (PreparedOrchestrationLimits(True, 64), PreparedOrchestrationStatus.INVALID_INPUT),
    (PreparedOrchestrationLimits(0, 64), PreparedOrchestrationStatus.INVALID_INPUT),
    (PreparedOrchestrationLimits(-1, 64), PreparedOrchestrationStatus.INVALID_INPUT),
    (PreparedOrchestrationLimits(16.0, 64), PreparedOrchestrationStatus.INVALID_INPUT),
    (PreparedOrchestrationLimits(17, 64), PreparedOrchestrationStatus.INVALID_INPUT),
])
def test_orchestration_limit_validation_matrix_is_fail_closed_before_builder(monkeypatch, limits, expected):
    calls = []
    monkeypatch.setattr(orchestration_module, "build_prepared_two_street_game", lambda *a: calls.append(a))
    result = run_prepared_two_street_orchestration(replace(_request(), orchestration_limits=limits))
    _assert_failure(result, expected)
    assert calls == []


def test_invalid_expected_witness_is_rejected_before_builder(monkeypatch):
    calls = []
    monkeypatch.setattr(
        orchestration_module,
        "build_prepared_two_street_game",
        lambda *args: calls.append(args),
    )
    result = run_prepared_two_street_orchestration(
        replace(_request(), expected_output_semantic_sha256="A" * 64)
    )
    _assert_failure(result, PreparedOrchestrationStatus.INVALID_INPUT)
    assert calls == []


def test_orchestration_trace_cap_four_accepts_three_rejects_before_builder(monkeypatch):
    _assert_success(run_prepared_two_street_orchestration(_request(orchestration_limits=PreparedOrchestrationLimits(4, 64))))
    calls = []
    monkeypatch.setattr(orchestration_module, "build_prepared_two_street_game", lambda *a: calls.append(a))
    rejected = run_prepared_two_street_orchestration(_request(orchestration_limits=PreparedOrchestrationLimits(3, 64)))
    _assert_failure(rejected, PreparedOrchestrationStatus.CAP_EXCEEDED)
    assert calls == []


def test_orchestration_result_cap_nine_accepts_eight_rejects_before_builder(monkeypatch):
    _assert_success(run_prepared_two_street_orchestration(_request(orchestration_limits=PreparedOrchestrationLimits(16, 9))))
    calls = []
    monkeypatch.setattr(orchestration_module, "build_prepared_two_street_game", lambda *a: calls.append(a))
    rejected = run_prepared_two_street_orchestration(_request(orchestration_limits=PreparedOrchestrationLimits(16, 8)))
    _assert_failure(rejected, PreparedOrchestrationStatus.CAP_EXCEEDED)
    assert calls == []


def test_builder_cap_boundary_forwards_exactly_and_blocks_evaluator_on_failure(monkeypatch):
    baseline = run_prepared_two_street_orchestration(_request())
    exact = baseline.run.build.counts.total_nodes
    good = replace(PreparedTwoStreetLimits(), max_total_nodes=exact)
    bad = replace(good, max_total_nodes=exact - 1)
    _assert_success(run_prepared_two_street_orchestration(_request(builder_limits=good)))
    calls = []
    real = orchestration_module.evaluate_prepared_two_street
    monkeypatch.setattr(orchestration_module, "evaluate_prepared_two_street", lambda *a: (calls.append(a), real(*a))[1])
    result = run_prepared_two_street_orchestration(_request(builder_limits=bad))
    _assert_failure(result, PreparedOrchestrationStatus.CAP_EXCEEDED)
    assert calls == []


def test_evaluation_cap_boundary_forwards_exactly_and_blocks_success_constructors_on_failure():
    success_cap = None
    for cap in range(1, 100):
        result = run_prepared_two_street_orchestration(_request(evaluation_limits=replace(PreparedEvaluationLimits(), max_result_records=cap)))
        if result.status is PreparedOrchestrationStatus.SUCCESS:
            success_cap = cap
            break
    assert success_cap is not None
    _assert_success(run_prepared_two_street_orchestration(_request(evaluation_limits=replace(PreparedEvaluationLimits(), max_result_records=success_cap))))
    failed = run_prepared_two_street_orchestration(_request(evaluation_limits=replace(PreparedEvaluationLimits(), max_result_records=success_cap - 1)))
    _assert_failure(failed, PreparedOrchestrationStatus.CAP_EXCEEDED)


def test_constructor_call_count_matrix(monkeypatch):
    counts = {
        "trace": 0, "identity": 0, "run": 0, "success": 0,
        "error": 0, "failure": 0,
    }
    originals = {name: getattr(orchestration_module, name) for name in (
        "PreparedOrchestrationTraceRecord", "PreparedTwoStreetOrchestrationIdentity",
        "PreparedTwoStreetOrchestrationRun", "PreparedOrchestrationError",
        "PreparedTwoStreetOrchestrationResult",
    )}
    monkeypatch.setattr(orchestration_module, "PreparedOrchestrationTraceRecord", lambda *a: (counts.__setitem__("trace", counts["trace"] + 1), originals["PreparedOrchestrationTraceRecord"](*a))[1])
    monkeypatch.setattr(orchestration_module, "PreparedTwoStreetOrchestrationIdentity", lambda *a: (counts.__setitem__("identity", counts["identity"] + 1), originals["PreparedTwoStreetOrchestrationIdentity"](*a))[1])
    monkeypatch.setattr(orchestration_module, "PreparedTwoStreetOrchestrationRun", lambda *a: (counts.__setitem__("run", counts["run"] + 1), originals["PreparedTwoStreetOrchestrationRun"](*a))[1])
    monkeypatch.setattr(orchestration_module, "PreparedOrchestrationError", lambda *a: (counts.__setitem__("error", counts["error"] + 1), originals["PreparedOrchestrationError"](*a))[1])
    def result_ctor(*args):
        key = "success" if args[0] is PreparedOrchestrationStatus.SUCCESS else "failure"
        counts[key] += 1
        return originals["PreparedTwoStreetOrchestrationResult"](*args)
    monkeypatch.setattr(orchestration_module, "PreparedTwoStreetOrchestrationResult", result_ctor)
    actual_builder = orchestration_module.build_prepared_two_street_game
    actual_evaluator = orchestration_module.evaluate_prepared_two_street
    expected_success = {
        "trace": 4, "identity": 1, "run": 1, "success": 1,
        "error": 0, "failure": 0,
    }
    expected_failure = {
        "trace": 0, "identity": 0, "run": 0, "success": 0,
        "error": 1, "failure": 1,
    }
    _assert_success(run_prepared_two_street_orchestration(_request()))
    assert counts == expected_success

    def reset():
        counts.update({key: 0 for key in counts})
        monkeypatch.setattr(
            orchestration_module,
            "build_prepared_two_street_game",
            actual_builder,
        )
        monkeypatch.setattr(
            orchestration_module,
            "evaluate_prepared_two_street",
            actual_evaluator,
        )

    reset()
    _assert_failure(
        run_prepared_two_street_orchestration(PreparedTwoStreetOrchestrationRequest),
        PreparedOrchestrationStatus.INVALID_INPUT,
    )
    assert counts == expected_failure
    reset()
    _assert_failure(
        run_prepared_two_street_orchestration(
            _request(orchestration_limits=PreparedOrchestrationLimits(3, 64))
        ),
        PreparedOrchestrationStatus.CAP_EXCEEDED,
    )
    assert counts == expected_failure
    reset()
    monkeypatch.setattr(
        orchestration_module,
        "build_prepared_two_street_game",
        lambda *args: PreparedTwoStreetBuildResult(
            PreparedTwoStreetStatus.INVALID_INPUT,
            None,
            PreparedBuildError("failure", "builder"),
        ),
    )
    _assert_failure(
        run_prepared_two_street_orchestration(_request()),
        PreparedOrchestrationStatus.BUILD_FAILURE,
    )
    assert counts == expected_failure
    reset()
    monkeypatch.setattr(
        orchestration_module,
        "build_prepared_two_street_game",
        lambda *args: (_ for _ in ()).throw(RuntimeError()),
    )
    _assert_failure(
        run_prepared_two_street_orchestration(_request()),
        PreparedOrchestrationStatus.INTERNAL_FAILURE,
    )
    assert counts == expected_failure
    reset()
    monkeypatch.setattr(
        orchestration_module,
        "evaluate_prepared_two_street",
        lambda *args: PreparedEvaluationResult(
            PreparedEvaluationStatus.INVALID_INPUT,
            None,
            None,
            PreparedEvaluationError("failure", "evaluation"),
        ),
    )
    _assert_failure(
        run_prepared_two_street_orchestration(_request()),
        PreparedOrchestrationStatus.EVALUATION_INPUT_FAILURE,
    )
    assert counts == expected_failure
    reset()
    monkeypatch.setattr(
        orchestration_module,
        "evaluate_prepared_two_street",
        lambda *args: (_ for _ in ()).throw(RuntimeError()),
    )
    _assert_failure(
        run_prepared_two_street_orchestration(_request()),
        PreparedOrchestrationStatus.INTERNAL_FAILURE,
    )
    assert counts == expected_failure
    reset()
    _assert_failure(
        run_prepared_two_street_orchestration(_request(expected="0" * 64)),
        PreparedOrchestrationStatus.NON_REPRODUCIBLE,
    )
    assert counts == expected_failure


def test_public_mixed_profile_hand_oracle_is_independent():
    spec = _one_street_spec()
    _, _, build = _build(spec)
    root = _artifact(build, PreparedPlayer.VILLAIN, ("check", "bet::open-2"))
    facing = _artifact(build, PreparedPlayer.HERO, ("fold", "call"))
    hero_overrides = {facing.info_set_id: {"fold": 0.4, "call": 0.6}}
    villain_overrides = {root.info_set_id: {"check": 0.25, "bet::open-2": 0.75}}
    result = run_prepared_two_street_orchestration(_request(spec, villain=True, hero_overrides=hero_overrides, villain_overrides=villain_overrides))
    _assert_success(result)
    # Independent terminal arithmetic: check -0.05; bet/fold -1; bet/call 0.99.
    hero = 0.25 * -0.05 + 0.75 * (0.4 * -1.0 + 0.6 * 0.99)
    rake = 0.25 * 0.1 + 0.75 * 0.6 * 0.3
    villain = -hero - rake
    value = result.run.evaluation.fixed_profile_value
    assert (hero, villain, rake) == pytest.approx((0.133, -0.293, 0.16))
    assert (value.hero_ev, value.villain_ev, value.house_rake, value.conservation_residual) == pytest.approx((hero, villain, rake, 0.0))


@pytest.mark.parametrize("hero_choice,expected", [
    ({"fold": 1.0, "call": 0.0}, (-1.0, 1.0, 0.0)),
    ({"fold": 0.0, "call": 1.0}, (0.99, -1.29, 0.3)),
])
def test_public_one_street_fold_and_called_showdown_oracles_end_to_end(hero_choice, expected):
    spec = _one_street_spec()
    _, _, build = _build(spec)
    root = _artifact(build, PreparedPlayer.VILLAIN, ("check", "bet::open-2"))
    facing = _artifact(build, PreparedPlayer.HERO, ("fold", "call"))
    result = run_prepared_two_street_orchestration(_request(spec, villain=True, hero_overrides={facing.info_set_id: hero_choice}, villain_overrides={root.info_set_id: {"check": 0.0, "bet::open-2": 1.0}}))
    _assert_success(result)
    value = result.run.evaluation.fixed_profile_value
    assert (value.hero_ev, value.villain_ev, value.house_rake) == pytest.approx(expected)


def test_public_two_street_fixed_profile_oracle_end_to_end():
    result = run_prepared_two_street_orchestration(_request(_two_street_spec(), villain=True))
    _assert_success(result)
    value = result.run.evaluation.fixed_profile_value
    assert (value.hero_ev, value.villain_ev, value.house_rake) == pytest.approx((0.008, -0.208, 0.2))


def test_public_initial_all_in_oracle_end_to_end():
    result = run_prepared_two_street_orchestration(_request(_initial_all_in_spec(), villain=True))
    _assert_success(result)
    value = result.run.evaluation.fixed_profile_value
    assert (value.hero_ev, value.villain_ev, value.house_rake) == pytest.approx((0.5, -0.5, 0.0))


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_builder_base_exceptions_propagate(monkeypatch, exc):
    monkeypatch.setattr(orchestration_module, "build_prepared_two_street_game", lambda *a: (_ for _ in ()).throw(exc()))
    with pytest.raises(exc):
        run_prepared_two_street_orchestration(_request())


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_evaluator_base_exceptions_propagate(monkeypatch, exc):
    monkeypatch.setattr(orchestration_module, "evaluate_prepared_two_street", lambda *a: (_ for _ in ()).throw(exc()))
    with pytest.raises(exc):
        run_prepared_two_street_orchestration(_request())


def test_builder_unexpected_ordinary_exception_is_bounded_internal_failure(monkeypatch):
    monkeypatch.setattr(orchestration_module, "build_prepared_two_street_game", lambda *a: (_ for _ in ()).throw(RuntimeError("sensitive C:\\absolute\npath")))
    result = run_prepared_two_street_orchestration(_request())
    _assert_failure(result, PreparedOrchestrationStatus.INTERNAL_FAILURE)
    assert result.error.message == "unexpected builder failure: RuntimeError"
    assert result.error.build_identity is None


def test_evaluator_unexpected_ordinary_exception_is_bounded_internal_failure_with_build_provenance(monkeypatch):
    monkeypatch.setattr(orchestration_module, "evaluate_prepared_two_street", lambda *a: (_ for _ in ()).throw(RuntimeError("sensitive C:\\absolute\npath")))
    result = run_prepared_two_street_orchestration(_request())
    _assert_failure(result, PreparedOrchestrationStatus.INTERNAL_FAILURE)
    assert result.error.message == "unexpected evaluation failure: RuntimeError"
    assert result.error.build_identity is not None and result.error.build_counts is not None


def test_error_never_includes_exception_message_absolute_path_or_newline(monkeypatch):
    monkeypatch.setattr(orchestration_module, "evaluate_prepared_two_street", lambda *a: (_ for _ in ()).throw(ValueError("secret C:\\Users\\name\r\nraw payload")))
    result = run_prepared_two_street_orchestration(_request())
    text = result.error.message
    assert text == "unexpected evaluation failure: ValueError"
    assert "secret" not in text and "C:\\" not in text and "\r" not in text and "\n" not in text
