"""Focused acceptance tests for the prepared two-street file workflow."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, is_dataclass
from enum import Enum
from pathlib import Path

import pytest

import repeated_poker.prepared_two_street_file_workflow as workflow_module
from scripts.run_prepared_two_street_file import main as workflow_cli_main
from repeated_poker.prepared_two_street import (
    PREPARED_JOINT_ROOT_CONTRACT_VERSION,
    PREPARED_TWO_STREET_CONTRACT_VERSION,
    PreparedActionEvent,
    PreparedActionKind,
    PreparedActionOption,
    PreparedBucket,
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
    PreparedTwoStreetSpec,
    PreparedTwoStreetStatus,
    build_prepared_two_street_game,
    prepared_public_history_id,
    prepared_semantic_sha256,
)
from repeated_poker.prepared_two_street_evaluation import (
    PreparedActionProbability,
    PreparedPlayerProfile,
    PreparedProfileEntry,
)
from repeated_poker.prepared_two_street_file_workflow import *
from repeated_poker.prepared_two_street_orchestration import (
    PreparedOrchestrationStatus,
    PreparedTwoStreetOrchestrationRequest,
    run_prepared_two_street_orchestration,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "prepared_two_street_file_v1.json"
EXAMPLE_V2 = ROOT / "examples" / "prepared_two_street_file_v2.json"


def _document() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def _document_v2() -> dict:
    return json.loads(EXAMPLE_V2.read_text(encoding="utf-8"))


def _raw(document: dict) -> bytes:
    return json.dumps(
        document, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _independent_json_value(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _independent_json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_independent_json_value(item) for item in value]
    return value


def _inspection(document: dict | None = None):
    return inspect_prepared_two_street_file(_raw(document or _document()))


def _inspection_v2(document: dict | None = None):
    return inspect_prepared_two_street_file(_raw(document or _document_v2()))


def _complete_run_document(document: dict, *, include_villain: bool = True) -> dict:
    inspected = inspect_prepared_two_street_file(_raw(document))
    assert inspected.status is PreparedFileWorkflowStatus.SUCCESS

    def profile(player: str) -> list[dict]:
        return [
            {
                "info_set_id": row["info_set_id"],
                "actions": [
                    {"action_label": action["action_label"], "probability": 1.0}
                    for action in row["actions"]
                ],
            }
            for row in inspected.output["profile_template"]
            if row["player"] == player
        ]

    completed = deepcopy(document)
    completed["operation"] = "run"
    completed["template_identity"] = inspected.output["identity"]
    completed["hero_profile"] = profile("hero")
    completed["villain_profile"] = profile("villain") if include_villain else None
    return completed


def _run_document(*, include_villain: bool = False) -> dict:
    return _complete_run_document(_document(), include_villain=include_villain)


def _run_document_v2(*, include_villain: bool = True) -> dict:
    return _complete_run_document(_document_v2(), include_villain=include_villain)


def _oracle_spec_payload() -> dict:
    def action(player: str, kind: str, size_id=None, raise_to=None) -> dict:
        return {
            "type": "action", "street_id": "river", "player": player,
            "kind": kind, "size_id": size_id, "raise_to": raise_to,
            "is_all_in": False, "reopen": True,
        }

    v_check = action("villain", "check")
    h_check = action("hero", "check")
    v_bet = action("villain", "bet", "open-2", 2.0)
    h_call = action("hero", "call")
    return {
        "contract_version": PREPARED_TWO_STREET_CONTRACT_VERSION,
        "attestation": {
            "source": "M17 test-owned nontrivial oracle",
            "bucket_semantics": "one abstract bucket per player",
            "conditional_probability_semantics": "no chance transition",
            "observation_mapping": "public actions and own bucket only",
            "perfect_recall_attested": True,
        },
        "starting_chips": {"hero": 10.0, "villain": 10.0},
        "initial_committed": {"hero": 1.0, "villain": 1.0},
        "rake": {"rate": 0.05, "cap": 3.0},
        "streets": [{
            "street_id": "river", "label": "River",
            "first_actor": "villain", "min_open_bet": 2.0,
        }],
        "hero_buckets": [{"bucket_id": "H0", "weight": 1.0}],
        "villain_buckets": [{"bucket_id": "V0", "weight": 1.0}],
        "decision_menus": [
            {
                "history": [], "street_id": "river", "player": "villain",
                "actions": [
                    {"kind": "check", "size_id": None, "raise_to": None, "is_all_in": False},
                    {"kind": "bet", "size_id": "open-2", "raise_to": 2.0, "is_all_in": False},
                ],
            },
            {
                "history": [v_check], "street_id": "river", "player": "hero",
                "actions": [
                    {"kind": "check", "size_id": None, "raise_to": None, "is_all_in": False},
                ],
            },
            {
                "history": [v_bet], "street_id": "river", "player": "hero",
                "actions": [
                    {"kind": "fold", "size_id": None, "raise_to": None, "is_all_in": False},
                    {"kind": "call", "size_id": None, "raise_to": None, "is_all_in": False},
                ],
            },
        ],
        "transition_id": None,
        "transition_rows": [],
        "showdown_values": [
            {
                "history": [v_check, h_check, {
                    "type": "street_close", "street_id": "river", "reason": "check-check",
                }],
                "hero_bucket_id": "H0", "villain_bucket_id": "V0",
                "hero_pot_share": 0.5,
            },
            {
                "history": [v_bet, h_call, {
                    "type": "street_close", "street_id": "river", "reason": "call",
                }],
                "hero_bucket_id": "H0", "villain_bucket_id": "V0",
                "hero_pot_share": 0.7,
            },
        ],
    }


def _oracle_manual_spec() -> PreparedTwoStreetSpec:
    street = "river"
    v_check = PreparedActionEvent(street, PreparedPlayer.VILLAIN, PreparedActionKind.CHECK, None, None, False, True)
    h_check = PreparedActionEvent(street, PreparedPlayer.HERO, PreparedActionKind.CHECK, None, None, False, True)
    v_bet = PreparedActionEvent(street, PreparedPlayer.VILLAIN, PreparedActionKind.BET, "open-2", 2.0, False, True)
    h_call = PreparedActionEvent(street, PreparedPlayer.HERO, PreparedActionKind.CALL, None, None, False, True)
    passive = lambda kind: PreparedActionOption(kind, None, None, False)
    return PreparedTwoStreetSpec(
        PREPARED_TWO_STREET_CONTRACT_VERSION,
        PreparedDataAttestation(
            "M17 test-owned nontrivial oracle", "one abstract bucket per player",
            "no chance transition", "public actions and own bucket only", True,
        ),
        PreparedHeadsUpChips(10.0, 10.0), PreparedHeadsUpChips(1.0, 1.0),
        PreparedRake(0.05, 3.0),
        (PreparedStreet(street, "River", PreparedPlayer.VILLAIN, 2.0),),
        (PreparedBucket("H0", 1.0),), (PreparedBucket("V0", 1.0),),
        (
            PreparedDecisionMenu(
                prepared_public_history_id(()), street, PreparedPlayer.VILLAIN,
                (passive(PreparedActionKind.CHECK), PreparedActionOption(PreparedActionKind.BET, "open-2", 2.0, False)),
            ),
            PreparedDecisionMenu(
                prepared_public_history_id((v_check,)), street, PreparedPlayer.HERO,
                (passive(PreparedActionKind.CHECK),),
            ),
            PreparedDecisionMenu(
                prepared_public_history_id((v_bet,)), street, PreparedPlayer.HERO,
                (passive(PreparedActionKind.FOLD), passive(PreparedActionKind.CALL)),
            ),
        ),
        None,
        (),
        (
            PreparedShowdownValue(
                prepared_public_history_id((v_check, h_check, PreparedStreetCloseEvent(street, PreparedRoundCloseReason.CHECK_CHECK))),
                "H0", "V0", 0.5,
            ),
            PreparedShowdownValue(
                prepared_public_history_id((v_bet, h_call, PreparedStreetCloseEvent(street, PreparedRoundCloseReason.CALL))),
                "H0", "V0", 0.7,
            ),
        ),
    )


def _joint_manual_spec() -> PreparedJointRootTwoStreetSpec:
    street = "river"
    villain_check = PreparedActionEvent(
        street, PreparedPlayer.VILLAIN, PreparedActionKind.CHECK,
        None, None, False, True,
    )
    hero_check = PreparedActionEvent(
        street, PreparedPlayer.HERO, PreparedActionKind.CHECK,
        None, None, False, True,
    )
    showdown_history = prepared_public_history_id((
        villain_check,
        hero_check,
        PreparedStreetCloseEvent(street, PreparedRoundCloseReason.CHECK_CHECK),
    ))
    check = PreparedActionOption(PreparedActionKind.CHECK, None, None, False)
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
        streets=(PreparedStreet(
            street, "River", PreparedPlayer.VILLAIN, 1.0,
        ),),
        hero_buckets=(PreparedBucket("H0", 0.6), PreparedBucket("H1", 0.4)),
        villain_buckets=(PreparedBucket("V0", 0.7), PreparedBucket("V1", 0.3)),
        decision_menus=(
            PreparedDecisionMenu(
                prepared_public_history_id(()), street,
                PreparedPlayer.VILLAIN, (check,),
            ),
            PreparedDecisionMenu(
                prepared_public_history_id((villain_check,)), street,
                PreparedPlayer.HERO, (check,),
            ),
        ),
        transition_id=None,
        transition_rows=(),
        showdown_values=tuple(
            PreparedShowdownValue(showdown_history, hero, villain, share)
            for hero, villain, share in (
                ("H0", "V0", 1.0),
                ("H0", "V1", 0.5),
                ("H1", "V0", 0.0),
                ("H1", "V1", 0.25),
            )
        ),
        root_matchups=tuple(
            PreparedRootMatchup(hero, villain, probability)
            for hero, villain, probability in (
                ("H0", "V0", 0.5),
                ("H0", "V1", 0.1),
                ("H1", "V0", 0.2),
                ("H1", "V1", 0.2),
            )
        ),
    )


def test_public_contract_is_frozen_and_immutable():
    assert PREPARED_TWO_STREET_FILE_FORMAT == "prepared-two-street-file-v1"
    assert PREPARED_TWO_STREET_TEMPLATE_ID == "prepared-two-street-profile-template-sha256-v1"
    assert PREPARED_TWO_STREET_FILE_OUTPUT_ID == "prepared-two-street-file-output-v1"
    assert PREPARED_TWO_STREET_FILE_FORMAT_V2 == "prepared-two-street-file-v2"
    assert PREPARED_TWO_STREET_TEMPLATE_ID_V2 == "prepared-two-street-profile-template-sha256-v2"
    assert PREPARED_TWO_STREET_FILE_OUTPUT_ID_V2 == "prepared-two-street-file-output-v2"
    assert workflow_module.__all__ == [
        "PREPARED_TWO_STREET_FILE_FORMAT",
        "PREPARED_TWO_STREET_TEMPLATE_ID",
        "PREPARED_TWO_STREET_FILE_OUTPUT_ID",
        "PREPARED_TWO_STREET_FILE_FORMAT_V2",
        "PREPARED_TWO_STREET_TEMPLATE_ID_V2",
        "PREPARED_TWO_STREET_FILE_OUTPUT_ID_V2",
        "PreparedFileWorkflowStatus",
        "PreparedFileWorkflowLimits",
        "PreparedFileWorkflowError",
        "PreparedFileWorkflowResult",
        "inspect_prepared_two_street_file",
        "run_prepared_two_street_file",
        "prepared_file_workflow_json",
    ]
    result = _inspection()
    with pytest.raises(FrozenInstanceError):
        result.status = PreparedFileWorkflowStatus.INVALID_INPUT


def test_inspect_example_generates_complete_ordered_template():
    result = _inspection()
    assert result.status is PreparedFileWorkflowStatus.SUCCESS
    assert result.error is None
    assert result.output["builder_status"] == "SUCCESS"
    assert result.output["counts"]["hero_information_sets"] == 3
    assert result.output["counts"]["villain_information_sets"] == 3
    rows = result.output["profile_template"]
    assert len(rows) == 6
    assert {row["player"] for row in rows} == {"hero", "villain"}
    assert all(action["probability"] is None for row in rows for action in row["actions"])
    assert len({row["info_set_id"] for row in rows}) == 6


def test_inspect_identity_ignores_whitespace_and_object_key_order():
    document = _document()
    compact = _inspection(document)
    reordered = inspect_prepared_two_street_file(
        json.dumps(document, sort_keys=True, indent=3).encode("utf-8")
    )
    assert reordered.status is PreparedFileWorkflowStatus.SUCCESS
    assert reordered.output["identity"] == compact.output["identity"]
    assert reordered.output["profile_template"] == compact.output["profile_template"]


def test_inspect_v2_preserves_explicit_joint_root_contract():
    result = _inspection_v2()
    assert result.status is PreparedFileWorkflowStatus.SUCCESS
    assert result.error is None
    assert result.output["format_version"] == PREPARED_TWO_STREET_FILE_FORMAT_V2
    assert result.output["output_id"] == PREPARED_TWO_STREET_FILE_OUTPUT_ID_V2
    assert result.output["identity"]["template_id"] == PREPARED_TWO_STREET_TEMPLATE_ID_V2
    assert result.output["builder_status"] == "SUCCESS"
    assert result.output["counts"]["root_matchups"] == 4
    assert result.output["counts"]["hero_information_sets"] == 2
    assert result.output["counts"]["villain_information_sets"] == 2
    assert len(result.output["profile_template"]) == 4


def test_v1_and_v2_examples_have_frozen_identity_and_output_pins():
    v1 = _inspection()
    assert v1.status is PreparedFileWorkflowStatus.SUCCESS
    assert hashlib.sha256(EXAMPLE.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == (
        "301b742b3e8e1a804aa35cd4814cd378b5c204b90dc615aa04feef53d9e6db44"
    )
    assert v1.output["identity"] == {
        "raw_sha256": "47b7ff02c8340cc8cfc03c0fe6d214f7b6c45af8bd6732c9f1e10b4deff5a5bd",
        "semantic_sha256": "68e08e948b7bf5d3bd8c3a5dd25c44e2383de791100693fca3eadc9c2a8957fc",
        "template_id": "prepared-two-street-profile-template-sha256-v1",
        "template_sha256": "c076ffd57ed80c9e739cf7202929517bca4f810c5ad8904c509975805e17eca5",
    }
    assert hashlib.sha256(
        prepared_file_workflow_json(v1).encode("utf-8")
    ).hexdigest() == "f60c7c40fe7efc9745387061a065260d861deb27e88ded6efbee77cecd9f2fd2"

    v2 = _inspection_v2()
    assert v2.status is PreparedFileWorkflowStatus.SUCCESS
    assert hashlib.sha256(EXAMPLE_V2.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == (
        "72b0e7dad3e64a3abe9864a6e357105107dacd64fde95102574228ec8571d13a"
    )
    assert v2.output["identity"] == {
        "raw_sha256": "26ad813083c8b71fb153858965bce76f9fe44502f8a5c7ff0675807c4e6605bc",
        "semantic_sha256": "70eaf7051dc8db647d65c8bb682f44aa47da45c0c439c9eca383db79147f49dc",
        "template_id": "prepared-two-street-profile-template-sha256-v2",
        "template_sha256": "bc8390b95a6e5cdb7f1dd9ff8e932ba01e994f3d43259585d964daf2e97a6f94",
    }
    assert hashlib.sha256(
        prepared_file_workflow_json(v2).encode("utf-8")
    ).hexdigest() == "188756ca37f9f94717587649b58d7b382497e935aff59ce34be11259109de5bd"

    run = run_prepared_two_street_file(_raw(_run_document_v2()))
    assert run.status is PreparedFileWorkflowStatus.SUCCESS
    assert run.output["orchestration_identity"]["m14_identity"]["run_identity"] == (
        "5acfca6afc367c911d2ed327a5192fbae28ee4b4e68264e0f9d60aa6dd6e368e"
    )
    assert run.output["orchestration_identity"]["m15_identity"]["output_semantic_sha256"] == (
        "c3c1902ad4606012ac92a438ef357f2bd91878dd65c88ed9d98bd08b435cd158"
    )
    assert run.output["orchestration_identity"]["output_semantic_sha256"] == (
        "6dab21624aa649f66dc81304a16f1e26a757ad7021c45da7a7ad2338a854ab37"
    )
    assert hashlib.sha256(
        prepared_file_workflow_json(run).encode("utf-8")
    ).hexdigest() == "bcd2efece241a29d386a24e9eb50fbefe298ce1f0521137294fff0fa15b60176"


@pytest.mark.parametrize("include_villain", [False, True])
def test_run_succeeds_with_complete_profiles(include_villain):
    result = run_prepared_two_street_file(_raw(_run_document(include_villain=include_villain)))
    assert result.status is PreparedFileWorkflowStatus.SUCCESS
    assert result.error is None
    assert result.output["orchestration_status"] == "SUCCESS"
    assert result.output["builder_status"] == "SUCCESS"
    assert result.output["evaluation_status"] == "SUCCESS"
    assert result.output["exact_response"]["num_villain_pure_strategies"] == 1
    if include_villain:
        assert result.output["fixed_profile_value"] is not None
    else:
        assert result.output["fixed_profile_value"] is None


def test_run_matches_independently_assembled_direct_m16_request():
    spec_payload = _oracle_spec_payload()
    canonical = json.dumps(
        spec_payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    spec = _oracle_manual_spec()
    identity = PreparedContentIdentity(
        hashlib.sha256(canonical).hexdigest(), prepared_semantic_sha256(spec)
    )
    built = build_prepared_two_street_game(spec, canonical, identity)
    assert built.status is PreparedTwoStreetStatus.SUCCESS

    template = []
    for observation in built.build.information_sets:
        key = observation.key
        template.append({
            "info_set_id": observation.info_set_id,
            "player": key.player.value,
            "street_id": key.street_id,
            "own_current_bucket_id": key.own_current_bucket_id,
            "own_bucket_history": list(key.own_bucket_history),
            "public_history_id": key.public_history_id,
            "own_action_history": list(key.own_action_history),
            "actions": [
                {"action_label": label, "probability": None}
                for label in observation.legal_action_labels
            ],
        })
    template_hash = hashlib.sha256(json.dumps(
        template, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    expected_identity = {
        "raw_sha256": identity.raw_sha256,
        "semantic_sha256": identity.semantic_sha256,
        "template_id": PREPARED_TWO_STREET_TEMPLATE_ID,
        "template_sha256": template_hash,
    }
    assert expected_identity == {
        "raw_sha256": "d8ad5c63c386cb26c28e011ada13e327e1bc4283193e2ca61c317016c88bc2c5",
        "semantic_sha256": "a78e8aac352a75ce4e5f6d6c0d293a4a2e850cb1fd39bd9474b01e60de0d0dd2",
        "template_id": "prepared-two-street-profile-template-sha256-v1",
        "template_sha256": "3b3a2b3ba5ad7f277f24950f66e3fc4b2757c1e1d00ee3ffd5879a2fcb055ab4",
    }
    inspection_document = {
        "format_version": PREPARED_TWO_STREET_FILE_FORMAT,
        "operation": "inspect",
        "spec": spec_payload,
    }
    inspection = inspect_prepared_two_street_file(_raw(inspection_document))
    assert inspection.status is PreparedFileWorkflowStatus.SUCCESS
    assert inspection.output["identity"] == expected_identity
    assert inspection.output["profile_template"] == template

    profiles = {}
    for player in (PreparedPlayer.HERO, PreparedPlayer.VILLAIN):
        entries = []
        for observation in built.build.information_sets:
            if observation.key.player is player:
                probabilities = {label: 0.0 for label in observation.legal_action_labels}
                if player is PreparedPlayer.VILLAIN:
                    probabilities["check"] = 1.0
                elif "fold" in probabilities:
                    probabilities["fold"] = 1.0
                else:
                    probabilities[observation.legal_action_labels[0]] = 1.0
                entries.append(PreparedProfileEntry(
                    observation.info_set_id,
                    tuple(
                        PreparedActionProbability(label, probabilities[label])
                        for label in observation.legal_action_labels
                    ),
                ))
        profiles[player] = PreparedPlayerProfile(tuple(entries))

    def file_profile(profile: PreparedPlayerProfile) -> list[dict]:
        return [{
            "info_set_id": entry.info_set_id,
            "actions": [{
                "action_label": probability.action_label,
                "probability": probability.probability,
            } for probability in entry.action_probabilities],
        } for entry in profile.entries]

    document = {
        "format_version": PREPARED_TWO_STREET_FILE_FORMAT,
        "operation": "run",
        "spec": spec_payload,
        "template_identity": expected_identity,
        "hero_profile": file_profile(profiles[PreparedPlayer.HERO]),
        "villain_profile": file_profile(profiles[PreparedPlayer.VILLAIN]),
    }
    workflow = run_prepared_two_street_file(_raw(document))
    assert workflow.status is PreparedFileWorkflowStatus.SUCCESS
    direct = run_prepared_two_street_orchestration(PreparedTwoStreetOrchestrationRequest(
        spec=spec, raw_input_bytes=canonical, content_identity=identity,
        hero_profile=profiles[PreparedPlayer.HERO], villain_profile=profiles[PreparedPlayer.VILLAIN],
    ))
    assert direct.status is PreparedOrchestrationStatus.SUCCESS
    exact = workflow.output["exact_response"]
    assert exact["num_villain_pure_strategies"] == 2
    assert exact["num_best_response_strategies"] == 1
    assert exact["villain_max_ev"] == pytest.approx(1.0)
    assert exact["hero_ev_worst"] == exact["hero_ev_best"] == pytest.approx(-1.0)
    assert exact["representative_pure_response"]["assignments"][0]["action_label"] == "bet::open-2"
    fixed = workflow.output["fixed_profile_value"]
    assert (
        fixed["hero_ev"], fixed["villain_ev"], fixed["house_rake"],
        fixed["conservation_residual"],
    ) == pytest.approx((-0.05, -0.05, 0.1, 0.0))
    assert workflow.output["exact_response"] == _independent_json_value(direct.run.evaluation.exact_response)
    assert workflow.output["fixed_profile_value"] == _independent_json_value(direct.run.evaluation.fixed_profile_value)
    assert workflow.output["orchestration_identity"] == _independent_json_value(direct.identity)
    assert workflow.output["identity"] == expected_identity


def test_run_v2_matches_independent_public_m16_request_and_joint_oracle():
    spec_payload = _document_v2()["spec"]
    canonical = json.dumps(
        spec_payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    spec = _joint_manual_spec()
    identity = PreparedContentIdentity(
        hashlib.sha256(canonical).hexdigest(), prepared_semantic_sha256(spec)
    )
    built = build_prepared_two_street_game(spec, canonical, identity)
    assert built.status is PreparedTwoStreetStatus.SUCCESS
    assert tuple(
        (row.hero_bucket_id, row.villain_bucket_id, row.probability)
        for row in spec.root_matchups
    ) == (
        ("H0", "V0", 0.5),
        ("H0", "V1", 0.1),
        ("H1", "V0", 0.2),
        ("H1", "V1", 0.2),
    )

    profiles = {}
    for player in (PreparedPlayer.HERO, PreparedPlayer.VILLAIN):
        profiles[player] = PreparedPlayerProfile(tuple(
            PreparedProfileEntry(
                observation.info_set_id,
                tuple(
                    PreparedActionProbability(label, 1.0)
                    for label in observation.legal_action_labels
                ),
            )
            for observation in built.build.information_sets
            if observation.key.player is player
        ))
    direct = run_prepared_two_street_orchestration(
        PreparedTwoStreetOrchestrationRequest(
            spec=spec,
            raw_input_bytes=canonical,
            content_identity=identity,
            hero_profile=profiles[PreparedPlayer.HERO],
            villain_profile=profiles[PreparedPlayer.VILLAIN],
        )
    )
    assert direct.status is PreparedOrchestrationStatus.SUCCESS

    workflow = run_prepared_two_street_file(_raw(_run_document_v2()))
    assert workflow.status is PreparedFileWorkflowStatus.SUCCESS
    assert workflow.error is None
    assert workflow.output["format_version"] == PREPARED_TWO_STREET_FILE_FORMAT_V2
    assert workflow.output["output_id"] == PREPARED_TWO_STREET_FILE_OUTPUT_ID_V2
    assert workflow.output["fixed_profile_value"]["hero_ev"] == pytest.approx(0.2)
    assert workflow.output["fixed_profile_value"]["hero_ev"] != pytest.approx(0.08)
    assert workflow.output["exact_response"]["hero_ev_worst"] == pytest.approx(0.2)
    assert workflow.output["exact_response"]["hero_ev_best"] == pytest.approx(0.2)
    assert workflow.output["fixed_profile_value"] == _independent_json_value(
        direct.run.evaluation.fixed_profile_value
    )
    assert workflow.output["exact_response"] == _independent_json_value(
        direct.run.evaluation.exact_response
    )
    assert workflow.output["orchestration_identity"] == _independent_json_value(
        direct.identity
    )


def test_file_format_and_prepared_contract_pairing_fails_closed():
    cases = []

    v1_with_joint = _document()
    v1_with_joint["spec"]["contract_version"] = PREPARED_JOINT_ROOT_CONTRACT_VERSION
    v1_with_joint["spec"]["root_matchups"] = deepcopy(
        _document_v2()["spec"]["root_matchups"]
    )
    cases.append(v1_with_joint)

    v2_without_joint = _document_v2()
    v2_without_joint["spec"]["contract_version"] = PREPARED_TWO_STREET_CONTRACT_VERSION
    del v2_without_joint["spec"]["root_matchups"]
    cases.append(v2_without_joint)

    v2_hybrid = _document_v2()
    v2_hybrid["spec"]["contract_version"] = PREPARED_TWO_STREET_CONTRACT_VERSION
    cases.append(v2_hybrid)

    unknown = _document_v2()
    unknown["format_version"] = "prepared-two-street-file-v999"
    cases.append(unknown)

    for document in cases:
        result = inspect_prepared_two_street_file(_raw(document))
        assert result.status is PreparedFileWorkflowStatus.INVALID_INPUT
        assert result.output is None
        assert result.error is not None


@pytest.mark.parametrize(
    "mutation",
    [
        "empty",
        "duplicate",
        "unknown_bucket",
        "zero",
        "negative",
        "boolean",
        "marginal_mismatch",
    ],
)
def test_v2_invalid_root_rows_are_controlled_without_partial_output(mutation):
    document = _document_v2()
    rows = document["spec"]["root_matchups"]
    if mutation == "empty":
        rows.clear()
    elif mutation == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif mutation == "unknown_bucket":
        rows[0]["hero_bucket_id"] = "UNKNOWN"
    elif mutation == "zero":
        rows[0]["probability"] = 0.0
    elif mutation == "negative":
        rows[0]["probability"] = -0.1
    elif mutation == "boolean":
        rows[0]["probability"] = True
    elif mutation == "marginal_mismatch":
        rows[0]["probability"] = 0.49
        rows[1]["probability"] = 0.11

    result = _inspection_v2(document)
    assert result.status in {
        PreparedFileWorkflowStatus.INVALID_INPUT,
        PreparedFileWorkflowStatus.BUILD_FAILURE,
    }
    assert result.output is None
    assert result.error is not None


@pytest.mark.parametrize("key_action", ["missing", "unknown"])
def test_v2_root_row_shape_is_exact(key_action):
    document = _document_v2()
    if key_action == "missing":
        del document["spec"]["root_matchups"][0]["probability"]
    else:
        document["spec"]["root_matchups"][0]["extra"] = 1
    result = _inspection_v2(document)
    assert result.status is PreparedFileWorkflowStatus.INVALID_INPUT
    assert result.output is None
    assert result.error.phase == "spec.root_matchups[0]"


def test_v2_root_row_cap_precedes_row_materialization(monkeypatch):
    document = _document_v2()
    document["spec"]["root_matchups"] = [
        {"hero_bucket_id": "H0", "villain_bucket_id": "V0", "probability": 0.5}
    ] * 10_001
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(workflow_module, "PreparedRootMatchup", forbidden)
    result = _inspection_v2(document)
    assert result.status is PreparedFileWorkflowStatus.CAP_EXCEEDED
    assert result.output is None
    assert result.error.phase == "spec.root_matchups"
    assert not called


def test_v2_root_row_order_changes_identity_but_not_evaluation():
    first = _document_v2()
    reordered = _document_v2()
    reordered["spec"]["root_matchups"][0], reordered["spec"]["root_matchups"][1] = (
        reordered["spec"]["root_matchups"][1],
        reordered["spec"]["root_matchups"][0],
    )
    first_inspection = _inspection_v2(first)
    reordered_inspection = _inspection_v2(reordered)
    assert first_inspection.status is PreparedFileWorkflowStatus.SUCCESS
    assert reordered_inspection.status is PreparedFileWorkflowStatus.SUCCESS
    assert first_inspection.output["identity"]["raw_sha256"] != reordered_inspection.output["identity"]["raw_sha256"]
    assert first_inspection.output["identity"]["semantic_sha256"] != reordered_inspection.output["identity"]["semantic_sha256"]

    first_run = run_prepared_two_street_file(
        _raw(_complete_run_document(first))
    )
    reordered_run = run_prepared_two_street_file(
        _raw(_complete_run_document(reordered))
    )
    assert first_run.status is PreparedFileWorkflowStatus.SUCCESS
    assert reordered_run.status is PreparedFileWorkflowStatus.SUCCESS
    assert first_run.output["fixed_profile_value"] == reordered_run.output["fixed_profile_value"]
    assert first_run.output["exact_response"] == reordered_run.output["exact_response"]


def test_run_rejects_template_identity_mismatch_without_partial_output():
    document = _run_document()
    document["template_identity"]["template_sha256"] = "0" * 64
    result = run_prepared_two_street_file(_raw(document))
    assert result.status is PreparedFileWorkflowStatus.IDENTITY_MISMATCH
    assert result.output is None
    assert result.error.phase == "template_identity"


def test_run_rejects_incomplete_profile_without_orchestration(monkeypatch):
    document = _run_document()
    document["hero_profile"].pop()
    called = False
    def forbidden(_request):
        nonlocal called
        called = True
        raise AssertionError
    monkeypatch.setattr(workflow_module, "run_prepared_two_street_orchestration", forbidden)
    result = run_prepared_two_street_file(_raw(document))
    assert result.status is PreparedFileWorkflowStatus.PROFILE_FAILURE
    assert result.output is None
    assert result.error.nested_status is None
    assert result.error.builder_status is None
    assert result.error.evaluation_status is None
    assert not called


def test_run_rejects_duplicate_and_foreign_action_labels():
    document = _run_document()
    document["hero_profile"][0]["actions"][0]["action_label"] = "foreign"
    result = run_prepared_two_street_file(_raw(document))
    assert result.status is PreparedFileWorkflowStatus.PROFILE_FAILURE
    assert result.output is None


@pytest.mark.parametrize("probability", [-0.25, 0.5, 2.0])
def test_m16_profile_probability_failure_preserves_exact_nested_statuses(probability):
    document = _run_document()
    document["hero_profile"][0]["actions"][0]["probability"] = probability
    result = run_prepared_two_street_file(_raw(document))
    assert result.status is PreparedFileWorkflowStatus.PROFILE_FAILURE
    assert result.output is None
    assert result.error.phase == "orchestration"
    assert result.error.nested_status == "EVALUATION_INPUT_FAILURE"
    assert result.error.builder_status == "SUCCESS"
    assert result.error.evaluation_status == "INVALID_PROFILE_PROBABILITY"


def test_local_profile_numeric_type_failures_are_profile_failures():
    boolean = _run_document()
    boolean["hero_profile"][0]["actions"][0]["probability"] = True
    boolean_result = run_prepared_two_street_file(_raw(boolean))
    assert boolean_result.status is PreparedFileWorkflowStatus.PROFILE_FAILURE
    assert boolean_result.output is None
    assert boolean_result.error.nested_status is None

    overflow = _raw(_run_document()).replace(
        b'"probability":1.0', b'"probability":1e400', 1
    )
    overflow_result = run_prepared_two_street_file(overflow)
    assert overflow_result.status is PreparedFileWorkflowStatus.PROFILE_FAILURE
    assert overflow_result.output is None
    assert overflow_result.error.nested_status is None


@pytest.mark.parametrize(
    ("raw", "status"),
    [
        (b"{", PreparedFileWorkflowStatus.PARSE_FAILURE),
        (b'\xff', PreparedFileWorkflowStatus.PARSE_FAILURE),
        (b'[]', PreparedFileWorkflowStatus.INVALID_INPUT),
        (b'{"format_version":"x","format_version":"y"}', PreparedFileWorkflowStatus.PARSE_FAILURE),
    ],
)
def test_malformed_json_is_controlled_and_no_partial(raw, status):
    result = inspect_prepared_two_street_file(raw)
    assert result.status is status
    assert result.output is None
    assert result.error is not None


def test_unknown_and_missing_keys_are_rejected():
    unknown = _document()
    unknown["extra"] = 1
    assert _inspection(unknown).status is PreparedFileWorkflowStatus.INVALID_INPUT
    missing = _document()
    del missing["spec"]["rake"]
    assert _inspection(missing).status is PreparedFileWorkflowStatus.INVALID_INPUT


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_json_number_token_is_rejected(value):
    document = _document()
    document["spec"]["rake"]["rate"] = value
    raw = json.dumps(document, allow_nan=True).encode("utf-8")
    result = inspect_prepared_two_street_file(raw)
    assert result.status is PreparedFileWorkflowStatus.PARSE_FAILURE
    assert result.output is None


@pytest.mark.parametrize("value", [10**4000, -(10**4000), True])
def test_non_binary64_number_is_controlled_invalid_input(value):
    document = _document()
    document["spec"]["rake"]["rate"] = value
    result = _inspection(document)
    assert result.status is PreparedFileWorkflowStatus.INVALID_INPUT
    assert result.output is None
    assert result.error.phase == "spec.rake"


def test_binary64_subnormal_remains_a_valid_finite_number():
    document = _document()
    document["spec"]["rake"]["rate"] = 5e-324
    result = _inspection(document)
    assert result.status is PreparedFileWorkflowStatus.SUCCESS
    assert result.error is None


def test_input_byte_cap_precedes_json_parse(monkeypatch):
    called = False
    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError
    monkeypatch.setattr(workflow_module.json, "loads", forbidden)
    result = inspect_prepared_two_street_file(
        b"{}", PreparedFileWorkflowLimits(max_input_bytes=1)
    )
    assert result.status is PreparedFileWorkflowStatus.CAP_EXCEEDED
    assert result.output is None
    assert not called


def test_template_record_cap_precedes_template_materialization(monkeypatch):
    called = False

    def forbidden(_build):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(workflow_module, "_template", forbidden)
    result = inspect_prepared_two_street_file(
        _raw(_document()), PreparedFileWorkflowLimits(max_output_records=1)
    )
    assert result.status is PreparedFileWorkflowStatus.CAP_EXCEEDED
    assert result.output is None
    assert result.error.phase == "template"
    assert result.error.builder_status == "SUCCESS"
    assert not called


def test_json_depth_cap_is_controlled():
    result = inspect_prepared_two_street_file(
        b'{"a":{"b":1}}', PreparedFileWorkflowLimits(max_json_depth=2)
    )
    assert result.status is PreparedFileWorkflowStatus.CAP_EXCEEDED
    assert result.output is None


def test_parser_recursion_limit_is_controlled_as_a_cap():
    raw = ("{\"a\":" + "[" * 2_000 + "0" + "]" * 2_000 + "}").encode("utf-8")
    result = inspect_prepared_two_street_file(raw)
    assert result.status is PreparedFileWorkflowStatus.CAP_EXCEEDED
    assert result.output is None


def test_limit_ceiling_is_not_silently_clamped():
    result = inspect_prepared_two_street_file(
        b"{}", PreparedFileWorkflowLimits(max_input_bytes=1_000_001)
    )
    assert result.status is PreparedFileWorkflowStatus.INVALID_INPUT
    assert result.output is None


def test_result_json_is_deterministic_strict_and_bounded_on_error():
    result = inspect_prepared_two_street_file(b"{")
    first = prepared_file_workflow_json(result)
    assert first == prepared_file_workflow_json(result)
    payload = json.loads(first)
    assert payload["output"] is None
    assert payload["status"] == "PARSE_FAILURE"
    assert "\n" not in payload["error"]["message"]
    assert len(payload["error"]["message"]) <= 500


def test_cli_inspect_success_and_controlled_failure(tmp_path):
    script = ROOT / "scripts" / "run_prepared_two_street_file.py"
    success = subprocess.run(
        [sys.executable, str(script), "inspect", str(EXAMPLE)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert success.returncode == 0
    assert json.loads(success.stdout)["status"] == "SUCCESS"
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    failure = subprocess.run(
        [sys.executable, str(script), "inspect", str(bad)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert failure.returncode == 2
    assert json.loads(failure.stdout)["status"] == "PARSE_FAILURE"
    assert failure.stderr == ""


def test_cli_inspect_and_run_v2_use_the_existing_surface(tmp_path):
    script = ROOT / "scripts" / "run_prepared_two_street_file.py"
    inspection = subprocess.run(
        [sys.executable, str(script), "inspect", str(EXAMPLE_V2)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert inspection.returncode == 0
    inspected_payload = json.loads(inspection.stdout)
    assert inspected_payload["status"] == "SUCCESS"
    assert inspected_payload["output"]["format_version"] == PREPARED_TWO_STREET_FILE_FORMAT_V2

    completed = tmp_path / "prepared-v2-run.json"
    completed.write_text(
        json.dumps(_run_document_v2(), ensure_ascii=False), encoding="utf-8"
    )
    run = subprocess.run(
        [sys.executable, str(script), "run", str(completed)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert run.returncode == 0
    run_payload = json.loads(run.stdout)
    assert run_payload["status"] == "SUCCESS"
    assert run_payload["output"]["fixed_profile_value"]["hero_ev"] == pytest.approx(0.2)
    assert run.stderr == ""


@pytest.mark.parametrize("path_kind", ["missing", "directory"])
def test_cli_read_failure_is_bounded_json_without_silent_fallback(tmp_path, path_kind):
    script = ROOT / "scripts" / "run_prepared_two_street_file.py"
    path = tmp_path / "input"
    if path_kind == "directory":
        path.mkdir()
    failure = subprocess.run(
        [sys.executable, str(script), "inspect", str(path)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert failure.returncode == 2
    payload = json.loads(failure.stdout)
    assert payload["status"] == "INVALID_INPUT"
    assert payload["output"] is None
    assert payload["error"]["phase"] == "input"
    assert payload["error"]["message"] == "cannot read input file"
    assert str(path) not in failure.stdout
    assert failure.stderr == ""


def test_cli_permission_failure_is_bounded_json(monkeypatch, capsys, tmp_path):
    def denied(_path):
        raise PermissionError("secret path must not escape")

    monkeypatch.setattr(Path, "read_bytes", denied)
    exit_code = workflow_cli_main(["inspect", str(tmp_path / "secret.json")])
    captured = capsys.readouterr()
    assert exit_code == 2
    payload = json.loads(captured.out)
    assert payload["status"] == "INVALID_INPUT"
    assert payload["output"] is None
    assert payload["error"]["message"] == "cannot read input file"
    assert "secret" not in payload["error"]["message"]
    assert captured.err == ""


def test_m16_core_modules_and_top_level_exports_are_unchanged():
    init_source = (ROOT / "src" / "repeated_poker" / "__init__.py").read_text(encoding="utf-8")
    assert "prepared_two_street_file_workflow" not in init_source
    assert "PREPARED_TWO_STREET_FILE_FORMAT_V2" not in init_source
    assert "filesystem" in (ROOT / "src" / "repeated_poker" / "prepared_two_street_orchestration.py").read_text(encoding="utf-8")


def test_readme_and_workflow_doc_publish_v2_example_and_guardrails():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "prepared_two_street_file_workflow.md").read_text(encoding="utf-8")
    assert "examples/prepared_two_street_file_v2.json" in readme
    assert "prepared-two-street-file-v2" in readme
    for phrase in (
        "prepared-two-street-file-v2",
        "root_matchups",
        "never sorts",
        "10,000-row root cap",
        "fails closed",
    ):
        assert phrase in guide
