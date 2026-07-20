"""Focused acceptance tests for prepared two-street end-to-end orchestration."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
from dataclasses import MISSING, FrozenInstanceError, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

import repeated_poker
import repeated_poker.prepared_two_street_orchestration as orchestration_module
from repeated_poker.game import GameTree
from repeated_poker.prepared_two_street import (
    PREPARED_JOINT_ROOT_BUILDER_ID,
    PREPARED_JOINT_ROOT_CONTRACT_VERSION,
    PREPARED_TWO_STREET_BUILDER_ID,
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
    PreparedJointRootTwoStreetSpec,
    PreparedPlayer,
    PreparedRake,
    PreparedRootMatchup,
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
    PreparedEvaluation,
    PreparedEvaluationError,
    PreparedEvaluationIdentity,
    PreparedEvaluationLimits,
    PreparedEvaluationRequest,
    PreparedEvaluationResult,
    PreparedEvaluationStatus,
    PreparedPlayerProfile,
    PreparedProfileEntry,
    PreparedProfileNormalization,
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


def _joint_root_spec() -> PreparedJointRootTwoStreetSpec:
    street = "river"
    villain_check = _event(
        street, PreparedPlayer.VILLAIN, PreparedActionKind.CHECK
    )
    hero_check = _event(
        street, PreparedPlayer.HERO, PreparedActionKind.CHECK
    )
    terminal_state = prepared_public_history_id(
        (
            villain_check,
            hero_check,
            PreparedStreetCloseEvent(
                street, PreparedRoundCloseReason.CHECK_CHECK
            ),
        )
    )
    return PreparedJointRootTwoStreetSpec(
        contract_version=PREPARED_JOINT_ROOT_CONTRACT_VERSION,
        attestation=PreparedDataAttestation(
            "joint-root hand oracle",
            "exact synthetic private buckets",
            "explicit positive joint root mass; no factorization",
            "public actions and own bucket only",
            True,
        ),
        starting_chips=PreparedHeadsUpChips(10.0, 10.0),
        initial_committed=PreparedHeadsUpChips(1.0, 1.0),
        rake=PreparedRake(0.0, None),
        streets=(
            PreparedStreet(
                street, "River", PreparedPlayer.VILLAIN, 1.0
            ),
        ),
        hero_buckets=(
            PreparedBucket("H0", 0.6),
            PreparedBucket("H1", 0.4),
        ),
        villain_buckets=(
            PreparedBucket("V0", 0.7),
            PreparedBucket("V1", 0.3),
        ),
        decision_menus=(
            PreparedDecisionMenu(
                prepared_public_history_id(()),
                street,
                PreparedPlayer.VILLAIN,
                (_passive(PreparedActionKind.CHECK),),
            ),
            PreparedDecisionMenu(
                prepared_public_history_id((villain_check,)),
                street,
                PreparedPlayer.HERO,
                (_passive(PreparedActionKind.CHECK),),
            ),
        ),
        transition_id=None,
        transition_rows=(),
        showdown_values=(
            PreparedShowdownValue(terminal_state, "H0", "V0", 1.0),
            PreparedShowdownValue(terminal_state, "H0", "V1", 0.5),
            PreparedShowdownValue(terminal_state, "H1", "V0", 0.0),
            PreparedShowdownValue(terminal_state, "H1", "V1", 0.25),
        ),
        root_matchups=(
            PreparedRootMatchup("H0", "V0", 0.5),
            PreparedRootMatchup("H0", "V1", 0.1),
            PreparedRootMatchup("H1", "V0", 0.2),
            PreparedRootMatchup("H1", "V1", 0.2),
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
    spec: PreparedTwoStreetSpec | PreparedJointRootTwoStreetSpec | None = None,
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
    return hashlib.sha256(_independent_bytes(value)).hexdigest()


def _independent_bytes(value) -> bytes:
    return json.dumps(
        _independent_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


class _IterationTrap:
    def __iter__(self):
        raise AssertionError("nested iterable was traversed")

    def __len__(self):
        raise AssertionError("nested iterable was counted")

    def __getitem__(self, key):
        raise AssertionError("nested iterable was indexed")

    def __copy__(self):
        raise AssertionError("nested iterable was copied")

    def __deepcopy__(self, memo):
        raise AssertionError("nested iterable was materialized")


def _reachable_objects(root):
    """Walk only public dataclass/container edges in a returned result graph."""
    pending = [root]
    seen = set()
    while pending:
        value = pending.pop()
        marker = id(value)
        if marker in seen:
            continue
        seen.add(marker)
        yield value
        if is_dataclass(value) and not isinstance(value, type):
            pending.extend(getattr(value, item.name) for item in fields(value))
        elif isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, (tuple, list, set, frozenset)):
            pending.extend(value)


def _install_m16_constructor_counters(monkeypatch):
    counts = {
        "trace": 0,
        "identity": 0,
        "run": 0,
        "success": 0,
        "error": 0,
        "failure": 0,
    }
    originals = {
        name: getattr(orchestration_module, name)
        for name in (
            "PreparedOrchestrationTraceRecord",
            "PreparedTwoStreetOrchestrationIdentity",
            "PreparedTwoStreetOrchestrationRun",
            "PreparedOrchestrationError",
            "PreparedTwoStreetOrchestrationResult",
        )
    }

    def wrap(name, key):
        def constructor(*args, **kwargs):
            counts[key] += 1
            return originals[name](*args, **kwargs)
        monkeypatch.setattr(orchestration_module, name, constructor)

    wrap("PreparedOrchestrationTraceRecord", "trace")
    wrap("PreparedTwoStreetOrchestrationIdentity", "identity")
    wrap("PreparedTwoStreetOrchestrationRun", "run")
    wrap("PreparedOrchestrationError", "error")

    def result_constructor(*args, **kwargs):
        status = args[0] if args else kwargs["status"]
        counts[
            "success"
            if status is PreparedOrchestrationStatus.SUCCESS
            else "failure"
        ] += 1
        return originals["PreparedTwoStreetOrchestrationResult"](*args, **kwargs)

    monkeypatch.setattr(
        orchestration_module,
        "PreparedTwoStreetOrchestrationResult",
        result_constructor,
    )
    return counts


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
        PreparedOrchestrationLimits: [
            ("max_trace_records", "int", 16),
            ("max_result_records", "int", 64),
        ],
        PreparedTwoStreetOrchestrationRequest: [
            (
                "spec",
                "PreparedTwoStreetSpec | PreparedJointRootTwoStreetSpec",
                MISSING,
            ),
            ("raw_input_bytes", "bytes", MISSING),
            ("content_identity", "PreparedContentIdentity", MISSING),
            ("hero_profile", "PreparedPlayerProfile", MISSING),
            ("villain_profile", "PreparedPlayerProfile | None", None),
            ("builder_limits", "PreparedTwoStreetLimits", PreparedTwoStreetLimits()),
            ("evaluation_limits", "PreparedEvaluationLimits", PreparedEvaluationLimits()),
            ("evaluation_request", "PreparedEvaluationRequest", PreparedEvaluationRequest()),
            ("orchestration_limits", "PreparedOrchestrationLimits", PreparedOrchestrationLimits()),
            ("expected_output_semantic_sha256", "str | None", None),
        ],
        PreparedOrchestrationTraceRecord: [
            ("phase", "str", MISSING),
            ("subject", "str", MISSING),
            ("outcome", "str", MISSING),
        ],
        PreparedTwoStreetOrchestrationRun: [
            ("build", "PreparedTwoStreetBuild", MISSING),
            ("evaluation", "PreparedEvaluation", MISSING),
            (
                "trace",
                "tuple[PreparedOrchestrationTraceRecord, ...]",
                MISSING,
            ),
        ],
        PreparedTwoStreetOrchestrationIdentity: [
            ("orchestration_id", "str", MISSING),
            ("output_semantic_id", "str", MISSING),
            ("m14_identity", "PreparedTwoStreetIdentity", MISSING),
            ("m15_identity", "PreparedEvaluationIdentity", MISSING),
            ("effective_builder_limits", "PreparedTwoStreetLimits", MISSING),
            ("effective_evaluation_limits", "PreparedEvaluationLimits", MISSING),
            ("effective_orchestration_limits", "PreparedOrchestrationLimits", MISSING),
            ("correspondence_mode", "PreparedCorrespondenceMode", MISSING),
            ("oracle_check", "bool", MISSING),
            ("run_identity", "str", MISSING),
            ("output_semantic_sha256", "str", MISSING),
        ],
        PreparedOrchestrationError: [
            ("phase", "str", MISSING),
            ("message", "str", MISSING),
            ("build_identity", "PreparedTwoStreetIdentity | None", MISSING),
            ("build_counts", "PreparedBuildCounts | None", MISSING),
        ],
        PreparedTwoStreetOrchestrationResult: [
            ("status", "PreparedOrchestrationStatus", MISSING),
            ("builder_status", "PreparedTwoStreetStatus | None", MISSING),
            ("evaluation_status", "PreparedEvaluationStatus | None", MISSING),
            ("run", "PreparedTwoStreetOrchestrationRun | None", MISSING),
            ("identity", "PreparedTwoStreetOrchestrationIdentity | None", MISSING),
            ("error", "PreparedOrchestrationError | None", MISSING),
        ],
    }
    for cls, expected in expected_fields.items():
        actual = fields(cls)
        assert [(item.name, item.type) for item in actual] == [
            (name, annotation) for name, annotation, _ in expected
        ]
        for item, (_, _, default) in zip(actual, expected):
            if default is MISSING:
                assert item.default is MISSING
            else:
                assert item.default == default
            assert item.default_factory is MISSING
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


def test_joint_root_v2_orchestration_preserves_correlated_mass_and_identity_chain():
    request = _request(_joint_root_spec(), villain=True)
    result = run_prepared_two_street_orchestration(request)
    _assert_success(result)
    fixed = result.run.evaluation.fixed_profile_value
    assert fixed is not None
    factorized = (
        0.6 * 0.7 * 1.0
        + 0.6 * 0.3 * 0.0
        + 0.4 * 0.7 * -1.0
        + 0.4 * 0.3 * -0.5
    )
    assert fixed.hero_ev == pytest.approx(0.2)
    assert factorized == pytest.approx(0.08)
    assert fixed.hero_ev != pytest.approx(factorized)
    m14 = result.run.build.identity
    m15 = result.identity.m15_identity
    assert m14.contract_version == PREPARED_JOINT_ROOT_CONTRACT_VERSION
    assert m14.builder_id == PREPARED_JOINT_ROOT_BUILDER_ID
    assert m15.m14_contract_version == m14.contract_version
    assert m15.m14_builder_id == m14.builder_id
    assert m15.m14_raw_sha256 == m14.raw_sha256
    assert m15.m14_prepared_semantic_sha256 == m14.semantic_sha256
    assert m15.m14_ordered_tree_sha256 == m14.ordered_tree_sha256
    assert m15.m14_run_identity == m14.run_identity
    assert result.identity.m14_identity == m14
    assert result.identity.m15_identity == m15


def test_joint_root_v2_m15_m16_identities_match_independent_oracles_and_pins():
    request = _request(_joint_root_spec(), villain=True)
    result = run_prepared_two_street_orchestration(request)
    _assert_success(result)
    m14 = result.run.build.identity
    m15 = result.identity.m15_identity
    m15_without_output = {
        item.name: getattr(m15, item.name)
        for item in fields(m15)
        if item.name != "output_semantic_sha256"
    }
    assert m15.output_semantic_sha256 == _independent_sha(
        {
            "algorithm": orchestration_module.PREPARED_EVALUATION_OUTPUT_ID,
            "identity": m15_without_output,
            "evaluation": result.run.evaluation,
        }
    )
    run_payload = {
        "algorithm": PREPARED_TWO_STREET_ORCHESTRATION_ID,
        "output_semantic_id": PREPARED_TWO_STREET_ORCHESTRATION_OUTPUT_ID,
        "m14_identity": m14,
        "m15_identity": m15,
        "effective_limits": {
            "builder": request.builder_limits,
            "evaluation": request.evaluation_limits,
            "orchestration": request.orchestration_limits,
        },
        "correspondence_mode": request.evaluation_request.correspondence_mode,
        "oracle_check": request.evaluation_request.oracle_check,
    }
    run_identity = _independent_sha(run_payload)
    canonical_trace = tuple(
        {
            "__type__": "PreparedOrchestrationTraceRecord",
            "fields": {
                "phase": phase,
                "subject": subject,
                "outcome": outcome,
            },
        }
        for phase, subject, outcome in (
            ("orchestration", "input", "PASS"),
            ("builder", "completed", "PASS"),
            ("evaluation", "completed", "PASS"),
            ("orchestration", "identity", "PASS"),
        )
    )
    output_identity = _independent_sha(
        {
            "algorithm": PREPARED_TWO_STREET_ORCHESTRATION_OUTPUT_ID,
            "run_identity": run_identity,
            "build_counts": result.run.build.counts,
            "builder_status": PreparedTwoStreetStatus.SUCCESS,
            "evaluation_status": PreparedEvaluationStatus.SUCCESS,
            "trace": canonical_trace,
        }
    )
    assert run_identity == result.identity.run_identity
    assert output_identity == result.identity.output_semantic_sha256
    assert m14.semantic_sha256 == (
        "70eaf7051dc8db647d65c8bb682f44aa47da45c0c439c9eca383db79147f49dc"
    )
    assert m14.ordered_tree_sha256 == (
        "316e385f355bb04326e682b3ace14956b3b5cd290430106a0069d296a70ff3f5"
    )
    assert m14.run_identity == (
        "b2ce10bad8a6e569bc032aa6a1dbf3384950f13b9eb9922d30e8c0df148bd3cd"
    )
    assert m15.output_semantic_sha256 == (
        "30abedc20607c05b7173c9ab01d926f6d9755de277bd190de3c34f04c6d47047"
    )
    assert result.identity.run_identity == (
        "74e3f88c3ebfc44f8a9986730905d157b7e944c2644944350c3375c8bcaa826d"
    )
    assert result.identity.output_semantic_sha256 == (
        "152f52a229519a6ca6609ff9bbdbc991a4845b44adac99c58295f932e40bde16"
    )


def test_joint_root_v2_row_reverse_preserves_ev_but_changes_m14_m15_m16_identity():
    forward_spec = _joint_root_spec()
    reverse_spec = replace(
        forward_spec,
        root_matchups=tuple(reversed(forward_spec.root_matchups)),
    )
    forward = run_prepared_two_street_orchestration(
        _request(forward_spec, villain=True)
    )
    reverse = run_prepared_two_street_orchestration(
        _request(reverse_spec, villain=True)
    )
    _assert_success(forward)
    _assert_success(reverse)
    assert forward.run.evaluation.fixed_profile_value.hero_ev == pytest.approx(0.2)
    assert reverse.run.evaluation.fixed_profile_value.hero_ev == pytest.approx(0.2)
    assert (
        forward.run.build.identity.semantic_sha256
        != reverse.run.build.identity.semantic_sha256
    )
    assert (
        forward.run.build.identity.ordered_tree_sha256
        != reverse.run.build.identity.ordered_tree_sha256
    )
    assert (
        forward.run.build.identity.run_identity
        != reverse.run.build.identity.run_identity
    )
    assert (
        forward.identity.m15_identity.output_semantic_sha256
        != reverse.identity.m15_identity.output_semantic_sha256
    )
    assert forward.identity.run_identity != reverse.identity.run_identity
    assert (
        forward.identity.output_semantic_sha256
        != reverse.identity.output_semantic_sha256
    )


@pytest.mark.parametrize(
    "contract_version,builder_id",
    [
        (PREPARED_TWO_STREET_CONTRACT_VERSION, PREPARED_JOINT_ROOT_BUILDER_ID),
        (PREPARED_JOINT_ROOT_CONTRACT_VERSION, PREPARED_TWO_STREET_BUILDER_ID),
        ("unknown-prepared-contract", PREPARED_JOINT_ROOT_BUILDER_ID),
        ("", PREPARED_JOINT_ROOT_BUILDER_ID),
        (object(), PREPARED_JOINT_ROOT_BUILDER_ID),
        (PREPARED_JOINT_ROOT_CONTRACT_VERSION, object()),
        ([], PREPARED_JOINT_ROOT_BUILDER_ID),
        (PREPARED_JOINT_ROOT_CONTRACT_VERSION, []),
    ],
)
def test_joint_root_v2_identity_pair_faults_fail_closed_without_partial_payload(
    monkeypatch, contract_version, builder_id
):
    request = _request(_joint_root_spec(), villain=True)
    baseline = build_prepared_two_street_game(
        request.spec,
        request.raw_input_bytes,
        request.content_identity,
        request.builder_limits,
    )
    evaluation = evaluate_prepared_two_street(
        baseline.build,
        request.hero_profile,
        request.villain_profile,
        request.evaluation_limits,
        request.evaluation_request,
    )
    faulty_identity = replace(
        baseline.build.identity,
        contract_version=contract_version,
        builder_id=builder_id,
    )
    faulty_build = replace(baseline.build, identity=faulty_identity)
    faulty_m15 = replace(
        evaluation.identity,
        m14_contract_version=contract_version,
        m14_builder_id=builder_id,
    )
    monkeypatch.setattr(
        orchestration_module,
        "build_prepared_two_street_game",
        lambda *args: replace(baseline, build=faulty_build),
    )
    monkeypatch.setattr(
        orchestration_module,
        "evaluate_prepared_two_street",
        lambda *args: replace(evaluation, identity=faulty_m15),
    )
    result = run_prepared_two_street_orchestration(request)
    _assert_failure(
        result, PreparedOrchestrationStatus.BUILD_IDENTITY_MISMATCH
    )
    assert result.builder_status is PreparedTwoStreetStatus.SUCCESS
    assert result.evaluation_status is PreparedEvaluationStatus.SUCCESS


def test_joint_root_v2_recomputes_m14_run_identity_after_continuity_forgery(
    monkeypatch,
):
    request = _request(_joint_root_spec(), villain=True)
    baseline = build_prepared_two_street_game(
        request.spec,
        request.raw_input_bytes,
        request.content_identity,
        request.builder_limits,
    )
    evaluation = evaluate_prepared_two_street(
        baseline.build,
        request.hero_profile,
        request.villain_profile,
        request.evaluation_limits,
        request.evaluation_request,
    )
    forged_run_identity = "f" * 64
    faulty_build = replace(
        baseline.build,
        identity=replace(
            baseline.build.identity,
            run_identity=forged_run_identity,
        ),
    )
    faulty_evaluation = replace(
        evaluation,
        identity=replace(
            evaluation.identity,
            m14_run_identity=forged_run_identity,
        ),
    )
    monkeypatch.setattr(
        orchestration_module,
        "build_prepared_two_street_game",
        lambda *args: replace(baseline, build=faulty_build),
    )
    monkeypatch.setattr(
        orchestration_module,
        "evaluate_prepared_two_street",
        lambda *args: faulty_evaluation,
    )
    result = run_prepared_two_street_orchestration(request)
    _assert_failure(
        result, PreparedOrchestrationStatus.BUILD_IDENTITY_MISMATCH
    )
    assert result.builder_status is PreparedTwoStreetStatus.SUCCESS
    assert result.evaluation_status is PreparedEvaluationStatus.SUCCESS


def test_joint_root_v2_binds_request_content_before_accepting_identity_chain(
    monkeypatch,
):
    request = _request(_joint_root_spec(), villain=True)
    baseline = build_prepared_two_street_game(
        request.spec,
        request.raw_input_bytes,
        request.content_identity,
        request.builder_limits,
    )
    evaluation = evaluate_prepared_two_street(
        baseline.build,
        request.hero_profile,
        request.villain_profile,
        request.evaluation_limits,
        request.evaluation_request,
    )
    faulty_request = replace(
        request,
        content_identity=replace(
            request.content_identity,
            raw_sha256="f" * 64,
        ),
    )
    monkeypatch.setattr(
        orchestration_module,
        "build_prepared_two_street_game",
        lambda *args: baseline,
    )
    monkeypatch.setattr(
        orchestration_module,
        "evaluate_prepared_two_street",
        lambda *args: evaluation,
    )
    result = run_prepared_two_street_orchestration(faulty_request)
    _assert_failure(
        result, PreparedOrchestrationStatus.BUILD_IDENTITY_MISMATCH
    )
    assert result.builder_status is PreparedTwoStreetStatus.SUCCESS
    assert result.evaluation_status is PreparedEvaluationStatus.SUCCESS


def test_joint_root_v2_hash_counts_and_tree_faults_map_through_m15_no_partial(
    monkeypatch,
):
    request = _request(_joint_root_spec(), villain=True)
    baseline = build_prepared_two_street_game(
        request.spec,
        request.raw_input_bytes,
        request.content_identity,
        request.builder_limits,
    )
    reversed_root = replace(
        baseline.build.tree.root,
        children=tuple(reversed(baseline.build.tree.root.children)),
    )
    faults = (
        replace(
            baseline.build,
            identity=replace(
                baseline.build.identity, raw_sha256="f" * 64
            ),
        ),
        replace(
            baseline.build,
            counts=replace(
                baseline.build.counts,
                root_matchups=baseline.build.counts.root_matchups + 1,
            ),
        ),
        replace(baseline.build, tree=GameTree(reversed_root)),
    )
    for faulty_build in faults:
        monkeypatch.setattr(
            orchestration_module,
            "build_prepared_two_street_game",
            lambda *args, _build=faulty_build: replace(
                baseline, build=_build
            ),
        )
        result = run_prepared_two_street_orchestration(request)
        _assert_failure(
            result, PreparedOrchestrationStatus.BUILD_IDENTITY_MISMATCH
        )
        assert result.builder_status is PreparedTwoStreetStatus.SUCCESS
        assert result.evaluation_status in {
            PreparedEvaluationStatus.IDENTITY_MISMATCH,
            PreparedEvaluationStatus.BUILD_MISMATCH,
        }


def test_orchestration_rejects_non_v1_v2_spec_type_before_builder(monkeypatch):
    calls = []
    monkeypatch.setattr(
        orchestration_module,
        "build_prepared_two_street_game",
        lambda *args: calls.append(args),
    )
    result = run_prepared_two_street_orchestration(
        replace(_request(), spec=object())
    )
    _assert_failure(result, PreparedOrchestrationStatus.INVALID_INPUT)
    assert calls == []


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
    accepted = {}
    real_builder = orchestration_module.build_prepared_two_street_game
    real_evaluator = orchestration_module.evaluate_prepared_two_street

    def builder(*args):
        events.append("builder")
        result = real_builder(*args)
        accepted["build"] = result.build
        return result

    def evaluator(*args):
        events.append("evaluation")
        assert args[0] is accepted["build"]
        result = real_evaluator(*args)
        accepted["evaluation"] = result.evaluation
        accepted["m15_identity"] = result.identity
        return result

    monkeypatch.setattr(orchestration_module, "build_prepared_two_street_game", builder)
    monkeypatch.setattr(orchestration_module, "evaluate_prepared_two_street", evaluator)
    result = run_prepared_two_street_orchestration(_request())
    _assert_success(result)
    assert events == ["builder", "evaluation"]
    assert result.run.build is accepted["build"]
    assert result.run.evaluation is accepted["evaluation"]
    assert result.identity.m15_identity is accepted["m15_identity"]


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
    result = run_prepared_two_street_orchestration(request)
    _assert_success(result)
    assert len(seen["builder"]) == 4
    assert len(seen["evaluation"]) == 5
    assert all(left is right for left, right in zip(seen["builder"], (request.spec, request.raw_input_bytes, request.content_identity, request.builder_limits)))
    assert seen["evaluation"][0] is result.run.build
    assert all(left is right for left, right in zip(seen["evaluation"][1:], (request.hero_profile, request.villain_profile, request.evaluation_limits, request.evaluation_request)))


def test_success_returns_exact_trace_identity_references_and_no_error(monkeypatch):
    accepted = {}
    real_evaluator = orchestration_module.evaluate_prepared_two_street

    def evaluator(*args):
        result = real_evaluator(*args)
        accepted["build"] = args[0]
        accepted["evaluation"] = result.evaluation
        accepted["identity"] = result.identity
        return result

    monkeypatch.setattr(orchestration_module, "evaluate_prepared_two_street", evaluator)
    result = run_prepared_two_street_orchestration(_request())
    _assert_success(result)
    assert [(row.phase, row.subject, row.outcome) for row in result.run.trace] == [
        ("orchestration", "input", "PASS"),
        ("builder", "completed", "PASS"),
        ("evaluation", "completed", "PASS"),
        ("orchestration", "identity", "PASS"),
    ]
    assert result.run.build is accepted["build"]
    assert result.run.evaluation is accepted["evaluation"]
    assert result.identity.m14_identity is accepted["build"].identity
    assert result.identity.m15_identity is accepted["identity"]


@pytest.mark.parametrize("nested_count", [0, 37], ids=["empty", "many"])
def test_success_reuses_nested_objects_without_counting_copying_or_materializing(
    monkeypatch, nested_count
):
    request = _request()
    baseline_build_result = build_prepared_two_street_game(
        request.spec,
        request.raw_input_bytes,
        request.content_identity,
        request.builder_limits,
    )
    baseline_evaluation_result = evaluate_prepared_two_street(
        baseline_build_result.build,
        request.hero_profile,
        request.villain_profile,
        request.evaluation_limits,
        request.evaluation_request,
    )
    tree_trap = _IterationTrap()
    evaluation_trap = _IterationTrap()
    record_trap = _IterationTrap()
    trapped_build = replace(
        baseline_build_result.build,
        tree=GameTree(tree_trap),
        chance_normalization=tuple(object() for _ in range(nested_count)),
        information_sets=(
            baseline_build_result.build.information_sets * nested_count
        ),
    )
    trapped_hero_normalization = replace(
        baseline_evaluation_result.evaluation.hero_profile_normalization,
        records=record_trap,
    )
    trapped_evaluation = replace(
        baseline_evaluation_result.evaluation,
        hero_profile_normalization=trapped_hero_normalization,
        validation_trace=evaluation_trap,
    )
    trapped_evaluation_result = replace(
        baseline_evaluation_result,
        evaluation=trapped_evaluation,
    )

    def builder(*args):
        return replace(baseline_build_result, build=trapped_build)

    def evaluator(build, *args):
        assert build is trapped_build
        return trapped_evaluation_result

    monkeypatch.setattr(orchestration_module, "build_prepared_two_street_game", builder)
    monkeypatch.setattr(orchestration_module, "evaluate_prepared_two_street", evaluator)
    request = replace(
        request,
        orchestration_limits=PreparedOrchestrationLimits(4, 9),
    )
    result = run_prepared_two_street_orchestration(request)
    _assert_success(result)
    assert result.run.build is trapped_build
    assert result.run.build.tree.root is tree_trap
    assert len(result.run.build.chance_normalization) == nested_count
    assert result.run.evaluation is trapped_evaluation
    assert result.run.evaluation.validation_trace is evaluation_trap
    assert result.run.evaluation.hero_profile_normalization.records is record_trap
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


@pytest.mark.parametrize(
    "case_name,expected",
    [
        ("invalid-input", PreparedOrchestrationStatus.INVALID_INPUT),
        ("build-failure", PreparedOrchestrationStatus.BUILD_FAILURE),
        ("build-identity-mismatch", PreparedOrchestrationStatus.BUILD_IDENTITY_MISMATCH),
        ("evaluation-input", PreparedOrchestrationStatus.EVALUATION_INPUT_FAILURE),
        ("evaluation-core", PreparedOrchestrationStatus.EVALUATION_CORE_FAILURE),
        ("cap", PreparedOrchestrationStatus.CAP_EXCEEDED),
        ("numeric", PreparedOrchestrationStatus.NUMERIC_FAILURE),
        ("non-reproducible", PreparedOrchestrationStatus.NON_REPRODUCIBLE),
        ("unsupported", PreparedOrchestrationStatus.UNSUPPORTED_DOWNSTREAM),
        ("internal", PreparedOrchestrationStatus.INTERNAL_FAILURE),
    ],
)
def test_every_outer_non_success_obeys_unified_no_partial_invariant(
    monkeypatch, case_name, expected
):
    request = _request()
    if case_name == "invalid-input":
        request = PreparedTwoStreetOrchestrationRequest
    elif case_name == "cap":
        request = replace(
            request,
            orchestration_limits=PreparedOrchestrationLimits(3, 64),
        )
    elif case_name == "non-reproducible":
        request = replace(request, expected_output_semantic_sha256="0" * 64)
    elif case_name == "internal":
        monkeypatch.setattr(
            orchestration_module,
            "build_prepared_two_street_game",
            lambda *args: object(),
        )
    elif case_name in {"build-failure", "numeric", "unsupported"}:
        nested = {
            "build-failure": PreparedTwoStreetStatus.INVALID_INPUT,
            "numeric": PreparedTwoStreetStatus.NUMERIC_FAILURE,
            "unsupported": PreparedTwoStreetStatus.UNSUPPORTED_DOWNSTREAM,
        }[case_name]
        monkeypatch.setattr(
            orchestration_module,
            "build_prepared_two_street_game",
            lambda *args: PreparedTwoStreetBuildResult(
                nested,
                None,
                PreparedBuildError("failure", "builder"),
            ),
        )
    else:
        nested = {
            "build-identity-mismatch": PreparedEvaluationStatus.BUILD_MISMATCH,
            "evaluation-input": PreparedEvaluationStatus.INCOMPLETE_PROFILE,
            "evaluation-core": PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE,
        }[case_name]
        monkeypatch.setattr(
            orchestration_module,
            "evaluate_prepared_two_street",
            lambda *args: PreparedEvaluationResult(
                nested,
                None,
                None,
                PreparedEvaluationError("failure", "evaluation"),
            ),
        )
    result = run_prepared_two_street_orchestration(request)
    _assert_failure(result, expected)
    assert result.status is expected
    assert not any(
        type(value) in {
            PreparedTwoStreetOrchestrationRun,
            PreparedTwoStreetOrchestrationIdentity,
            PreparedOrchestrationTraceRecord,
            PreparedEvaluation,
        }
        for value in _reachable_objects(result)
    )


def test_build_success_evaluation_failure_keeps_only_identity_and_counts(monkeypatch):
    request = _request(villain=True)
    build_result = build_prepared_two_street_game(
        request.spec,
        request.raw_input_bytes,
        request.content_identity,
        request.builder_limits,
    )
    evaluation_result = evaluate_prepared_two_street(
        build_result.build,
        request.hero_profile,
        request.villain_profile,
        request.evaluation_limits,
        request.evaluation_request,
    )
    forbidden_objects = (
        id(build_result.build.tree),
        id(build_result.build.chance_normalization),
        id(build_result.build.information_sets),
        id(evaluation_result.evaluation),
        id(evaluation_result.evaluation.hero_profile_normalization),
        id(evaluation_result.evaluation.hero_profile_normalization.records),
        id(evaluation_result.evaluation.villain_profile_normalization),
        id(evaluation_result.evaluation.villain_profile_normalization.records),
        id(evaluation_result.evaluation.fixed_profile_value),
        id(evaluation_result.evaluation.exact_response),
        id(evaluation_result.evaluation.exact_response.representative_pure_response),
        id(evaluation_result.evaluation.exact_response.best_response_action_sets),
        id(evaluation_result.evaluation.exact_response.best_response_action_variation),
        id(evaluation_result.evaluation.exact_response.full_correspondence),
        id(evaluation_result.evaluation.validation_trace),
    )
    forbidden = {
        marker
        for marker, value in zip(
            forbidden_objects,
            (
                build_result.build.tree,
                build_result.build.chance_normalization,
                build_result.build.information_sets,
                evaluation_result.evaluation,
                evaluation_result.evaluation.hero_profile_normalization,
                evaluation_result.evaluation.hero_profile_normalization.records,
                evaluation_result.evaluation.villain_profile_normalization,
                evaluation_result.evaluation.villain_profile_normalization.records,
                evaluation_result.evaluation.fixed_profile_value,
                evaluation_result.evaluation.exact_response,
                evaluation_result.evaluation.exact_response.representative_pure_response,
                evaluation_result.evaluation.exact_response.best_response_action_sets,
                evaluation_result.evaluation.exact_response.best_response_action_variation,
                evaluation_result.evaluation.exact_response.full_correspondence,
                evaluation_result.evaluation.validation_trace,
            ),
        )
        if value is not None
    }
    monkeypatch.setattr(
        orchestration_module,
        "build_prepared_two_street_game",
        lambda *args: build_result,
    )
    monkeypatch.setattr(
        orchestration_module,
        "evaluate_prepared_two_street",
        lambda *args: PreparedEvaluationResult(
            PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE,
            None,
            None,
            PreparedEvaluationError("failure", "evaluation"),
        ),
    )
    result = run_prepared_two_street_orchestration(request)
    _assert_failure(result, PreparedOrchestrationStatus.EVALUATION_CORE_FAILURE)
    assert set(vars(result.error)) == {"phase", "message", "build_identity", "build_counts"}
    assert result.error.build_identity is build_result.build.identity
    assert result.error.build_counts is build_result.build.counts
    graph = tuple(_reachable_objects(result))
    assert not forbidden.intersection(map(id, graph))
    assert not any(
        type(value) in {
            GameTree,
            PreparedEvaluation,
            PreparedOrchestrationTraceRecord,
        }
        for value in graph
    )


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


@pytest.mark.parametrize(
    "fault,outer,nested",
    [
        (
            "missing-info-set",
            PreparedOrchestrationStatus.EVALUATION_INPUT_FAILURE,
            PreparedEvaluationStatus.INCOMPLETE_PROFILE,
        ),
        (
            "extra-info-set",
            PreparedOrchestrationStatus.EVALUATION_INPUT_FAILURE,
            PreparedEvaluationStatus.ILLEGAL_PROFILE_REFERENCE,
        ),
        (
            "duplicate-info-set",
            PreparedOrchestrationStatus.EVALUATION_INPUT_FAILURE,
            PreparedEvaluationStatus.INVALID_INPUT,
        ),
        (
            "missing-action",
            PreparedOrchestrationStatus.EVALUATION_INPUT_FAILURE,
            PreparedEvaluationStatus.INCOMPLETE_PROFILE,
        ),
        (
            "unknown-action",
            PreparedOrchestrationStatus.EVALUATION_INPUT_FAILURE,
            PreparedEvaluationStatus.ILLEGAL_PROFILE_REFERENCE,
        ),
        (
            "duplicate-action",
            PreparedOrchestrationStatus.EVALUATION_INPUT_FAILURE,
            PreparedEvaluationStatus.INVALID_INPUT,
        ),
        (
            "invalid-probability",
            PreparedOrchestrationStatus.EVALUATION_INPUT_FAILURE,
            PreparedEvaluationStatus.INVALID_PROFILE_PROBABILITY,
        ),
        (
            "unsupported-method",
            PreparedOrchestrationStatus.UNSUPPORTED_DOWNSTREAM,
            PreparedEvaluationStatus.UNSUPPORTED_DOWNSTREAM,
        ),
        (
            "unsupported-downstream",
            PreparedOrchestrationStatus.UNSUPPORTED_DOWNSTREAM,
            PreparedEvaluationStatus.UNSUPPORTED_DOWNSTREAM,
        ),
    ],
)
def test_profile_and_request_fault_matrix_maps_through_m15(
    fault, outer, nested
):
    request = _request()
    profile = request.hero_profile
    entries = profile.entries
    target_index = next(
        index
        for index, entry in enumerate(entries)
        if len(entry.action_probabilities) > 1
    )
    target = entries[target_index]
    actions = target.action_probabilities

    if fault == "missing-info-set":
        profile = PreparedPlayerProfile(entries[:-1])
    elif fault == "extra-info-set":
        profile = PreparedPlayerProfile(
            entries + (PreparedProfileEntry("unknown-info-set", ()),)
        )
    elif fault == "duplicate-info-set":
        profile = PreparedPlayerProfile(entries + (entries[0],))
    elif fault == "missing-action":
        profile = PreparedPlayerProfile(
            entries[:target_index]
            + (replace(target, action_probabilities=actions[:-1]),)
            + entries[target_index + 1 :]
        )
    elif fault == "unknown-action":
        profile = PreparedPlayerProfile(
            entries[:target_index]
            + (
                replace(
                    target,
                    action_probabilities=actions
                    + (PreparedActionProbability("unknown-action", 0.0),),
                ),
            )
            + entries[target_index + 1 :]
        )
    elif fault == "duplicate-action":
        profile = PreparedPlayerProfile(
            entries[:target_index]
            + (replace(target, action_probabilities=actions + (actions[0],)),)
            + entries[target_index + 1 :]
        )
    elif fault == "invalid-probability":
        profile = PreparedPlayerProfile(
            entries[:target_index]
            + (
                replace(
                    target,
                    action_probabilities=(
                        replace(actions[0], probability=-1.0),
                    )
                    + actions[1:],
                ),
            )
            + entries[target_index + 1 :]
        )
    elif fault == "unsupported-method":
        request = replace(
            request,
            evaluation_request=PreparedEvaluationRequest(method="unknown"),
        )
    elif fault == "unsupported-downstream":
        request = replace(
            request,
            evaluation_request=PreparedEvaluationRequest(
                downstream_request="pipeline"
            ),
        )
    if fault not in {"unsupported-method", "unsupported-downstream"}:
        request = replace(request, hero_profile=profile)
    result = run_prepared_two_street_orchestration(request)
    _assert_failure(result, outer)
    assert result.builder_status is PreparedTwoStreetStatus.SUCCESS
    assert result.evaluation_status is nested


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
    digest = "f2fa83821ffcc67f35f616a67b03615ee809034a42e79e5c4468ef62d6ecdf15"
    accepted = run_prepared_two_street_orchestration(_request(villain=True, expected=digest))
    rejected = run_prepared_two_street_orchestration(
        _request(villain=True, expected="0" + digest[1:])
    )
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
    pre_constructor_trace = tuple(
        {
            "__type__": "PreparedOrchestrationTraceRecord",
            "fields": {
                "phase": phase,
                "subject": subject,
                "outcome": outcome,
            },
        }
        for phase, subject, outcome in (
            ("orchestration", "input", "PASS"),
            ("builder", "completed", "PASS"),
            ("evaluation", "completed", "PASS"),
            ("orchestration", "identity", "PASS"),
        )
    )
    assert _independent_bytes(pre_constructor_trace) == _independent_bytes(
        result.run.trace
    )
    output_payload = {
        "algorithm": PREPARED_TWO_STREET_ORCHESTRATION_OUTPUT_ID,
        "run_identity": run_hash,
        "build_counts": result.run.build.counts,
        "builder_status": PreparedTwoStreetStatus.SUCCESS,
        "evaluation_status": PreparedEvaluationStatus.SUCCESS,
        "trace": pre_constructor_trace,
    }
    output_hash = _independent_sha(output_payload)
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


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("limits-object", object()),
        ("max_trace_records", object()),
        ("max_trace_records", True),
        ("max_trace_records", 0),
        ("max_trace_records", -1),
        ("max_trace_records", 16.0),
        ("max_trace_records", 17),
        ("max_result_records", object()),
        ("max_result_records", True),
        ("max_result_records", 0),
        ("max_result_records", -1),
        ("max_result_records", 64.0),
        ("max_result_records", 65),
    ],
)
def test_orchestration_limit_validation_matrix_is_fail_closed_before_builder(
    monkeypatch, field_name, value
):
    calls = []
    monkeypatch.setattr(orchestration_module, "build_prepared_two_street_game", lambda *a: calls.append(a))
    limits = (
        value
        if field_name == "limits-object"
        else replace(PreparedOrchestrationLimits(), **{field_name: value})
    )
    result = run_prepared_two_street_orchestration(
        replace(_request(), orchestration_limits=limits)
    )
    _assert_failure(result, PreparedOrchestrationStatus.INVALID_INPUT)
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
    # Independent public-fixture count: matchup root + three decisions +
    # three terminals.
    exact_total_nodes = 7
    good = replace(
        PreparedTwoStreetLimits(), max_total_nodes=exact_total_nodes
    )
    bad = replace(good, max_total_nodes=exact_total_nodes - 1)
    _assert_success(run_prepared_two_street_orchestration(_request(builder_limits=good)))
    calls = []
    constructors = _install_m16_constructor_counters(monkeypatch)
    real = orchestration_module.evaluate_prepared_two_street
    monkeypatch.setattr(orchestration_module, "evaluate_prepared_two_street", lambda *a: (calls.append(a), real(*a))[1])
    result = run_prepared_two_street_orchestration(_request(builder_limits=bad))
    _assert_failure(result, PreparedOrchestrationStatus.CAP_EXCEEDED)
    assert calls == []
    assert constructors == {
        "trace": 0,
        "identity": 0,
        "run": 0,
        "success": 0,
        "error": 1,
        "failure": 1,
    }


def test_evaluation_cap_boundary_forwards_exactly_and_blocks_success_constructors_on_failure(
    monkeypatch,
):
    # Independent count for the absent-Villain public fixture:
    # 3 result objects + profile outer 1 + Hero records 2 + trace 10 +
    # exact-response outer 1 + representative assignment 1 +
    # action-set outer 1 + selected action 1 = 20.
    exact_result_records = 20
    exact = replace(
        PreparedEvaluationLimits(), max_result_records=exact_result_records
    )
    lower = replace(exact, max_result_records=exact_result_records - 1)
    _assert_success(
        run_prepared_two_street_orchestration(
            _request(evaluation_limits=exact)
        )
    )
    constructors = _install_m16_constructor_counters(monkeypatch)
    failed = run_prepared_two_street_orchestration(
        _request(evaluation_limits=lower)
    )
    _assert_failure(failed, PreparedOrchestrationStatus.CAP_EXCEEDED)
    assert failed.evaluation_status is PreparedEvaluationStatus.CAP_EXCEEDED
    assert constructors == {
        "trace": 0,
        "identity": 0,
        "run": 0,
        "success": 0,
        "error": 1,
        "failure": 1,
    }


@pytest.mark.parametrize(
    "case_name,expected_status,expected_builder,expected_evaluator",
    [
        ("invalid-request", PreparedOrchestrationStatus.INVALID_INPUT, 0, 0),
        ("m16-cap", PreparedOrchestrationStatus.CAP_EXCEEDED, 0, 0),
        ("m14-non-success", PreparedOrchestrationStatus.BUILD_FAILURE, 1, 0),
        ("builder-exception", PreparedOrchestrationStatus.INTERNAL_FAILURE, 1, 0),
        ("builder-cap-boundary", PreparedOrchestrationStatus.CAP_EXCEEDED, 1, 0),
        ("m15-non-success", PreparedOrchestrationStatus.EVALUATION_INPUT_FAILURE, 1, 1),
        ("evaluator-exception", PreparedOrchestrationStatus.INTERNAL_FAILURE, 1, 1),
        ("evaluation-cap-boundary", PreparedOrchestrationStatus.CAP_EXCEEDED, 1, 1),
        ("identity-mismatch", PreparedOrchestrationStatus.BUILD_IDENTITY_MISMATCH, 1, 1),
        ("m16-witness", PreparedOrchestrationStatus.NON_REPRODUCIBLE, 1, 1),
        ("success", PreparedOrchestrationStatus.SUCCESS, 1, 1),
    ],
)
def test_constructor_call_count_matrix(
    monkeypatch,
    case_name,
    expected_status,
    expected_builder,
    expected_evaluator,
):
    constructor_counts = _install_m16_constructor_counters(monkeypatch)
    stage_counts = {"builder": 0, "evaluator": 0}
    real_builder = orchestration_module.build_prepared_two_street_game
    real_evaluator = orchestration_module.evaluate_prepared_two_street

    def builder(*args):
        stage_counts["builder"] += 1
        if case_name == "m14-non-success":
            return PreparedTwoStreetBuildResult(
                PreparedTwoStreetStatus.INVALID_INPUT,
                None,
                PreparedBuildError("failure", "builder"),
            )
        if case_name == "builder-exception":
            raise RuntimeError
        return real_builder(*args)

    def evaluator(*args):
        stage_counts["evaluator"] += 1
        if case_name == "m15-non-success":
            return PreparedEvaluationResult(
                PreparedEvaluationStatus.INVALID_INPUT,
                None,
                None,
                PreparedEvaluationError("failure", "evaluation"),
            )
        if case_name == "evaluator-exception":
            raise RuntimeError
        result = real_evaluator(*args)
        if case_name == "identity-mismatch":
            result = replace(
                result,
                identity=replace(result.identity, m14_builder_id="fault"),
            )
        return result

    monkeypatch.setattr(
        orchestration_module, "build_prepared_two_street_game", builder
    )
    monkeypatch.setattr(
        orchestration_module, "evaluate_prepared_two_street", evaluator
    )

    request = _request()
    if case_name == "invalid-request":
        request = PreparedTwoStreetOrchestrationRequest
    elif case_name == "m16-cap":
        request = replace(
            request,
            orchestration_limits=PreparedOrchestrationLimits(3, 64),
        )
    elif case_name == "builder-cap-boundary":
        request = replace(
            request,
            builder_limits=replace(
                PreparedTwoStreetLimits(), max_total_nodes=6
            ),
        )
    elif case_name == "evaluation-cap-boundary":
        request = replace(
            request,
            evaluation_limits=replace(
                PreparedEvaluationLimits(), max_result_records=19
            ),
        )
    elif case_name == "m16-witness":
        request = replace(request, expected_output_semantic_sha256="0" * 64)

    result = run_prepared_two_street_orchestration(request)
    assert result.status is expected_status
    assert stage_counts == {
        "builder": expected_builder,
        "evaluator": expected_evaluator,
    }
    expected_constructors = (
        {
            "trace": 4,
            "identity": 1,
            "run": 1,
            "success": 1,
            "error": 0,
            "failure": 0,
        }
        if case_name == "success"
        else {
            "trace": 0,
            "identity": 0,
            "run": 0,
            "success": 0,
            "error": 1,
            "failure": 1,
        }
    )
    assert constructor_counts == expected_constructors


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
