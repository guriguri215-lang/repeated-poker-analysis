"""Focused acceptance tests for the prepared two-street file workflow."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields, is_dataclass
from enum import Enum
from pathlib import Path

import pytest

import repeated_poker.prepared_two_street_file_workflow as workflow_module
from repeated_poker.prepared_two_street import (
    PREPARED_TWO_STREET_CONTRACT_VERSION,
    PreparedActionEvent,
    PreparedActionKind,
    PreparedActionOption,
    PreparedBucket,
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


def _document() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


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


def _run_document(*, include_villain: bool = False) -> dict:
    document = _document()
    inspected = _inspection(document)
    assert inspected.status is PreparedFileWorkflowStatus.SUCCESS
    template = inspected.output["profile_template"]

    def profile(player: str) -> list[dict]:
        return [
            {
                "info_set_id": row["info_set_id"],
                "actions": [
                    {"action_label": action["action_label"], "probability": 1.0}
                    for action in row["actions"]
                ],
            }
            for row in template
            if row["player"] == player
        ]

    document["operation"] = "run"
    document["template_identity"] = inspected.output["identity"]
    document["hero_profile"] = profile("hero")
    document["villain_profile"] = profile("villain") if include_villain else None
    return document


def _manual_spec() -> PreparedTwoStreetSpec:
    first, second, transition = "flop", "turn", "deal-turn"
    v1 = PreparedActionEvent(first, PreparedPlayer.VILLAIN, PreparedActionKind.CHECK, None, None, False, True)
    h1 = PreparedActionEvent(first, PreparedPlayer.HERO, PreparedActionKind.CHECK, None, None, False, True)
    close1 = PreparedStreetCloseEvent(first, PreparedRoundCloseReason.CHECK_CHECK)
    before = (v1, h1, close1)
    passive = PreparedActionOption(PreparedActionKind.CHECK, None, None, False)
    menus = [
        PreparedDecisionMenu(prepared_public_history_id(()), first, PreparedPlayer.VILLAIN, (passive,)),
        PreparedDecisionMenu(prepared_public_history_id((v1,)), first, PreparedPlayer.HERO, (passive,)),
    ]
    showdowns = []
    for outcome, share in (("red", 0.8), ("black", 0.2)):
        chance = PreparedChanceEvent(transition, outcome)
        h2 = PreparedActionEvent(second, PreparedPlayer.HERO, PreparedActionKind.CHECK, None, None, False, True)
        v2 = PreparedActionEvent(second, PreparedPlayer.VILLAIN, PreparedActionKind.CHECK, None, None, False, True)
        history = before + (chance,)
        menus.extend((
            PreparedDecisionMenu(prepared_public_history_id(history), second, PreparedPlayer.HERO, (passive,)),
            PreparedDecisionMenu(prepared_public_history_id(history + (h2,)), second, PreparedPlayer.VILLAIN, (passive,)),
        ))
        showdowns.append(PreparedShowdownValue(
            prepared_public_history_id(history + (h2, v2, PreparedStreetCloseEvent(second, PreparedRoundCloseReason.CHECK_CHECK))),
            "H0", "V0", share,
        ))
    row = PreparedTransitionRow(
        transition, prepared_public_history_id(before), "H0", "V0",
        (PreparedChanceEdge("red", "H0", "V0", 0.6), PreparedChanceEdge("black", "H0", "V0", 0.4)),
    )
    return PreparedTwoStreetSpec(
        PREPARED_TWO_STREET_CONTRACT_VERSION,
        PreparedDataAttestation(
            "worked abstract example", "one abstract bucket per player",
            "prepared conditional chance mass", "public events and own bucket only", True,
        ),
        PreparedHeadsUpChips(10.0, 10.0), PreparedHeadsUpChips(1.0, 1.0),
        PreparedRake(0.1, None),
        (PreparedStreet(first, "First", PreparedPlayer.VILLAIN, 1.0), PreparedStreet(second, "Second", PreparedPlayer.HERO, 1.0)),
        (PreparedBucket("H0", 1.0),), (PreparedBucket("V0", 1.0),),
        tuple(menus), transition, (row,), tuple(showdowns),
    )


def test_public_contract_is_frozen_and_immutable():
    assert PREPARED_TWO_STREET_FILE_FORMAT == "prepared-two-street-file-v1"
    assert PREPARED_TWO_STREET_TEMPLATE_ID.endswith("sha256-v1")
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
    document = _run_document(include_villain=True)
    workflow = run_prepared_two_street_file(_raw(document))
    assert workflow.status is PreparedFileWorkflowStatus.SUCCESS
    spec = _manual_spec()
    canonical = json.dumps(
        _document()["spec"], sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    identity = PreparedContentIdentity(
        hashlib.sha256(canonical).hexdigest(), prepared_semantic_sha256(spec)
    )
    built = build_prepared_two_street_game(spec, canonical, identity)
    assert built.status is PreparedTwoStreetStatus.SUCCESS
    profiles = {}
    for player in (PreparedPlayer.HERO, PreparedPlayer.VILLAIN):
        entries = []
        for observation in built.build.information_sets:
            if observation.key.player is player:
                entries.append(PreparedProfileEntry(
                    observation.info_set_id,
                    tuple(PreparedActionProbability(label, 1.0) for label in observation.legal_action_labels),
                ))
        profiles[player] = PreparedPlayerProfile(tuple(entries))
    direct = run_prepared_two_street_orchestration(PreparedTwoStreetOrchestrationRequest(
        spec=spec, raw_input_bytes=canonical, content_identity=identity,
        hero_profile=profiles[PreparedPlayer.HERO], villain_profile=profiles[PreparedPlayer.VILLAIN],
    ))
    assert direct.status is PreparedOrchestrationStatus.SUCCESS
    assert workflow.output["exact_response"] == _independent_json_value(direct.run.evaluation.exact_response)
    assert workflow.output["fixed_profile_value"] == _independent_json_value(direct.run.evaluation.fixed_profile_value)
    assert workflow.output["orchestration_identity"] == _independent_json_value(direct.identity)


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
    assert not called


def test_run_rejects_duplicate_and_foreign_action_labels():
    document = _run_document()
    document["hero_profile"][0]["actions"][0]["action_label"] = "foreign"
    result = run_prepared_two_street_file(_raw(document))
    assert result.status is PreparedFileWorkflowStatus.PROFILE_FAILURE
    assert result.output is None


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


def test_non_finite_number_is_rejected():
    document = _document()
    document["spec"]["rake"]["rate"] = float("nan")
    raw = json.dumps(document, allow_nan=True).encode("utf-8")
    result = inspect_prepared_two_street_file(raw)
    assert result.status is PreparedFileWorkflowStatus.PARSE_FAILURE
    assert result.output is None


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


def test_m16_core_modules_and_top_level_exports_are_unchanged():
    assert "prepared_two_street_file_workflow" not in (ROOT / "src" / "repeated_poker" / "__init__.py").read_text(encoding="utf-8")
    assert "filesystem" in (ROOT / "src" / "repeated_poker" / "prepared_two_street_orchestration.py").read_text(encoding="utf-8")
