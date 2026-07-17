"""Focused acceptance tests for the prepared two-street evaluation adapter."""

from __future__ import annotations

import ast
import hashlib
import inspect
import math
from dataclasses import fields, replace
from pathlib import Path

import pytest

import repeated_poker
import repeated_poker.prepared_two_street_evaluation as evaluation_module
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
    build_prepared_two_street_game,
    prepared_public_history_id,
    prepared_semantic_sha256,
)
from repeated_poker.prepared_two_street_evaluation import *
from repeated_poker.scenario_io import RiverScenarioBettingTree, SUPPORTED_FORMAT_VERSIONS
from repeated_poker.game import ChanceNode, HeroNode, VillainNode
from repeated_poker.game import GameTree, HeroStrategy, TerminalNode, VillainStrategy
from repeated_poker.fixed_profile import evaluate_fixed_profile
from repeated_poker.exact_response import solve_exact_response


def _attestation() -> PreparedDataAttestation:
    return PreparedDataAttestation(
        "M15 independent fixture", "abstract buckets", "prepared conditional mass",
        "public actions/outcomes and own bucket only", True,
    )


def _passive(kind: PreparedActionKind) -> PreparedActionOption:
    return PreparedActionOption(kind, None, None, False)


def _aggressive(kind: PreparedActionKind, size: str, amount: float) -> PreparedActionOption:
    return PreparedActionOption(kind, size, amount, False)


def _event(street: str, player: PreparedPlayer, kind: PreparedActionKind, size=None, amount=None) -> PreparedActionEvent:
    return PreparedActionEvent(street, player, kind, size, amount, False, True)


def _one_street_spec(*, check_share: float = 0.5, rake: float = 0.05) -> PreparedTwoStreetSpec:
    street = "river"
    v_check = _event(street, PreparedPlayer.VILLAIN, PreparedActionKind.CHECK)
    h_check = _event(street, PreparedPlayer.HERO, PreparedActionKind.CHECK)
    v_bet = _event(street, PreparedPlayer.VILLAIN, PreparedActionKind.BET, "open-2", 2.0)
    h_call = _event(street, PreparedPlayer.HERO, PreparedActionKind.CALL)
    check_state = prepared_public_history_id((v_check, h_check, PreparedStreetCloseEvent(street, PreparedRoundCloseReason.CHECK_CHECK)))
    call_state = prepared_public_history_id((v_bet, h_call, PreparedStreetCloseEvent(street, PreparedRoundCloseReason.CALL)))
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
            PreparedDecisionMenu(prepared_public_history_id(()), street, PreparedPlayer.VILLAIN, (_passive(PreparedActionKind.CHECK), _aggressive(PreparedActionKind.BET, "open-2", 2.0))),
            PreparedDecisionMenu(prepared_public_history_id((v_check,)), street, PreparedPlayer.HERO, (_passive(PreparedActionKind.CHECK),)),
            PreparedDecisionMenu(prepared_public_history_id((v_bet,)), street, PreparedPlayer.HERO, (_passive(PreparedActionKind.FOLD), _passive(PreparedActionKind.CALL))),
        ),
        None,
        (),
        (PreparedShowdownValue(check_state, "H0", "V0", check_share), PreparedShowdownValue(call_state, "H0", "V0", 0.7)),
    )


def _two_street_spec() -> PreparedTwoStreetSpec:
    first, second, transition = "flop", "turn", "deal-turn"
    v1 = _event(first, PreparedPlayer.VILLAIN, PreparedActionKind.CHECK)
    h1 = _event(first, PreparedPlayer.HERO, PreparedActionKind.CHECK)
    close1 = PreparedStreetCloseEvent(first, PreparedRoundCloseReason.CHECK_CHECK)
    before = (v1, h1, close1)
    menus = [
        PreparedDecisionMenu(prepared_public_history_id(()), first, PreparedPlayer.VILLAIN, (_passive(PreparedActionKind.CHECK),)),
        PreparedDecisionMenu(prepared_public_history_id((v1,)), first, PreparedPlayer.HERO, (_passive(PreparedActionKind.CHECK),)),
    ]
    showdowns = []
    for outcome, share in (("red", 0.8), ("black", 0.2)):
        chance = PreparedChanceEvent(transition, outcome)
        h2 = _event(second, PreparedPlayer.HERO, PreparedActionKind.CHECK)
        v2 = _event(second, PreparedPlayer.VILLAIN, PreparedActionKind.CHECK)
        events = before + (chance,)
        menus.extend((
            PreparedDecisionMenu(prepared_public_history_id(events), second, PreparedPlayer.HERO, (_passive(PreparedActionKind.CHECK),)),
            PreparedDecisionMenu(prepared_public_history_id(events + (h2,)), second, PreparedPlayer.VILLAIN, (_passive(PreparedActionKind.CHECK),)),
        ))
        showdowns.append(PreparedShowdownValue(prepared_public_history_id(events + (h2, v2, PreparedStreetCloseEvent(second, PreparedRoundCloseReason.CHECK_CHECK))), "H0", "V0", share))
    row = PreparedTransitionRow(transition, prepared_public_history_id(before), "H0", "V0", (
        PreparedChanceEdge("red", "H0", "V0", 0.6), PreparedChanceEdge("black", "H0", "V0", 0.4),
    ))
    return PreparedTwoStreetSpec(
        PREPARED_TWO_STREET_CONTRACT_VERSION, _attestation(), PreparedHeadsUpChips(10.0, 10.0),
        PreparedHeadsUpChips(1.0, 1.0), PreparedRake(0.1, None),
        (PreparedStreet(first, "First", PreparedPlayer.VILLAIN, 1.0), PreparedStreet(second, "Second", PreparedPlayer.HERO, 1.0)),
        (PreparedBucket("H0", 1.0),), (PreparedBucket("V0", 1.0),), tuple(menus), transition, (row,), tuple(showdowns),
    )


def _initial_all_in_spec() -> PreparedTwoStreetSpec:
    first, second, transition = "flop", "turn", "deal-turn"
    before = (PreparedStreetCloseEvent(first, PreparedRoundCloseReason.ALL_IN_CALL),)
    return PreparedTwoStreetSpec(
        PREPARED_TWO_STREET_CONTRACT_VERSION, _attestation(), PreparedHeadsUpChips(1.0, 1.0),
        PreparedHeadsUpChips(1.0, 1.0), PreparedRake(0.0, None),
        (PreparedStreet(first, "First", PreparedPlayer.VILLAIN, 1.0), PreparedStreet(second, "Second", PreparedPlayer.HERO, 1.0)),
        (PreparedBucket("H0", 1.0),), (PreparedBucket("V0", 1.0),), (), transition,
        (PreparedTransitionRow(transition, prepared_public_history_id(before), "H0", "V0", (
            PreparedChanceEdge("low", "H0", "V0", 0.25), PreparedChanceEdge("high", "H0", "V0", 0.75),
        )),),
        (
            PreparedShowdownValue(prepared_public_history_id(before + (PreparedChanceEvent(transition, "low"),)), "H0", "V0", 0.0),
            PreparedShowdownValue(prepared_public_history_id(before + (PreparedChanceEvent(transition, "high"),)), "H0", "V0", 1.0),
        ),
    )


def _avoided_villain_subtree_spec() -> PreparedTwoStreetSpec:
    street = "river"
    v_bet = _event(street, PreparedPlayer.VILLAIN, PreparedActionKind.BET, "b2", 2.0)
    v_check = _event(street, PreparedPlayer.VILLAIN, PreparedActionKind.CHECK)
    h_raise = _event(street, PreparedPlayer.HERO, PreparedActionKind.RAISE, "r4", 4.0)
    h_check = _event(street, PreparedPlayer.HERO, PreparedActionKind.CHECK)
    menus = (
        PreparedDecisionMenu(prepared_public_history_id(()), street, PreparedPlayer.VILLAIN, (_aggressive(PreparedActionKind.BET, "b2", 2.0), _passive(PreparedActionKind.CHECK))),
        PreparedDecisionMenu(prepared_public_history_id((v_check,)), street, PreparedPlayer.HERO, (_passive(PreparedActionKind.CHECK),)),
        PreparedDecisionMenu(prepared_public_history_id((v_bet,)), street, PreparedPlayer.HERO, (_aggressive(PreparedActionKind.RAISE, "r4", 4.0), _passive(PreparedActionKind.FOLD), _passive(PreparedActionKind.CALL))),
        PreparedDecisionMenu(prepared_public_history_id((v_bet, h_raise)), street, PreparedPlayer.VILLAIN, (_passive(PreparedActionKind.FOLD), _passive(PreparedActionKind.CALL))),
    )
    check_state = prepared_public_history_id((v_check, h_check, PreparedStreetCloseEvent(street, PreparedRoundCloseReason.CHECK_CHECK)))
    bet_call = _event(street, PreparedPlayer.HERO, PreparedActionKind.CALL)
    raise_call = _event(street, PreparedPlayer.VILLAIN, PreparedActionKind.CALL)
    showdowns = (
        PreparedShowdownValue(check_state, "H0", "V0", 0.0),
        PreparedShowdownValue(prepared_public_history_id((v_bet, bet_call, PreparedStreetCloseEvent(street, PreparedRoundCloseReason.CALL))), "H0", "V0", 1.0),
        PreparedShowdownValue(prepared_public_history_id((v_bet, h_raise, raise_call, PreparedStreetCloseEvent(street, PreparedRoundCloseReason.CALL))), "H0", "V0", 1.0),
    )
    return PreparedTwoStreetSpec(
        PREPARED_TWO_STREET_CONTRACT_VERSION, _attestation(), PreparedHeadsUpChips(10.0, 10.0),
        PreparedHeadsUpChips(1.0, 1.0), PreparedRake(0.0, None),
        (PreparedStreet(street, "River", PreparedPlayer.VILLAIN, 2.0),),
        (PreparedBucket("H0", 1.0),), (PreparedBucket("V0", 1.0),), menus, None, (), showdowns,
    )


def _build(spec: PreparedTwoStreetSpec):
    raw = b"m15-independent-fixture"
    result = build_prepared_two_street_game(spec, raw, PreparedContentIdentity(hashlib.sha256(raw).hexdigest(), prepared_semantic_sha256(spec)))
    assert result.build is not None, result.error
    return result.build


def _profile(build, player: PreparedPlayer, overrides=None, *, reverse=False) -> PreparedPlayerProfile:
    overrides = overrides or {}
    entries = []
    for artifact in build.information_sets:
        if artifact.key.player is not player:
            continue
        values = overrides.get(artifact.info_set_id, {})
        actions = tuple(PreparedActionProbability(label, values.get(label, 1.0 if index == 0 else 0.0)) for index, label in enumerate(artifact.legal_action_labels))
        entries.append(PreparedProfileEntry(artifact.info_set_id, tuple(reversed(actions)) if reverse else actions))
    return PreparedPlayerProfile(tuple(reversed(entries)) if reverse else tuple(entries))


def _artifact(build, player, actions):
    return next(item for item in build.information_sets if item.key.player is player and item.legal_action_labels == actions)


def _assert_success(result):
    assert result.status is PreparedEvaluationStatus.SUCCESS
    assert result.evaluation is not None and result.identity is not None and result.error is None


def _assert_failure(result, status):
    assert result.status is status
    assert result.evaluation is None and result.identity is None
    assert result.error is not None and result.error.message and len(result.error.message) <= 500 and len(result.error.phase) <= 64


def _returned_record_count(result):
    evaluation = result.evaluation
    exact = evaluation.exact_response
    total = 3 + 1 + len(evaluation.hero_profile_normalization.records) + len(evaluation.validation_trace)
    if evaluation.villain_profile_normalization is not None:
        total += 1 + len(evaluation.villain_profile_normalization.records)
    if evaluation.fixed_profile_value is not None:
        total += 1
    total += 1 + len(exact.representative_pure_response.assignments)
    total += sum(1 + len(item.action_labels) for item in exact.best_response_action_sets)
    total += sum(1 + len(item.action_labels) for item in exact.best_response_action_variation)
    total += len(exact.off_path_info_sets)
    if exact.full_correspondence is not None:
        total += sum(1 + len(strategy.assignments) for strategy in exact.full_correspondence)
    return total


def _ordered_tree_record(node):
    if isinstance(node, TerminalNode):
        return ("terminal", node.node_id, node.hero_ev.hex(), node.villain_ev.hex(), node.house_rake.hex())
    if isinstance(node, ChanceNode):
        return ("chance", node.node_id, tuple((probability.hex(), _ordered_tree_record(child)) for probability, child in node.children))
    kind = "hero" if isinstance(node, HeroNode) else "villain"
    return (kind, node.node_id, node.info_set, tuple((action, _ordered_tree_record(child)) for action, child in node.actions))


def _transition_node_id(record):
    transition, source, hero_bucket, villain_bucket = record.row_identity
    return "node:sha256:" + evaluation_module._sha({
        "kind": "transition",
        "public_history_id": source,
        "hero_bucket_history": (hero_bucket,),
        "villain_bucket_history": (villain_bucket,),
        "discriminator": transition,
    })


def _replace_transition_probabilities_and_rehash(build, probabilities):
    target = _transition_node_id(build.chance_normalization[0])

    def rewrite(node):
        if node.node_id == target:
            return replace(node, children=tuple((probability, child) for probability, (_, child) in zip(probabilities, node.children)))
        if isinstance(node, ChanceNode):
            return replace(node, children=tuple((probability, rewrite(child)) for probability, child in node.children))
        if isinstance(node, (HeroNode, VillainNode)):
            return replace(node, actions=tuple((action, rewrite(child)) for action, child in node.actions))
        return node

    tree = GameTree(rewrite(build.tree.root))
    ordered = evaluation_module._sha({
        "algorithm": "betting-tree-v2-ordered-tree-sha256-v1",
        "tree": _ordered_tree_record(tree.root),
    })
    identity = replace(build.identity, ordered_tree_sha256=ordered)
    identity = replace(identity, run_identity=evaluation_module._sha({
        "contract_version": identity.contract_version,
        "builder_id": identity.builder_id,
        "action_label_id": identity.action_label_id,
        "normalization_id": identity.normalization_id,
        "information_key_id": identity.information_key_id,
        "raw_sha256": identity.raw_sha256,
        "semantic_sha256": identity.semantic_sha256,
        "ordered_tree_sha256": identity.ordered_tree_sha256,
    }))
    return replace(build, tree=tree, identity=identity)


def test_exact_public_api_constants_enums_fields_and_signature():
    expected = [
        "PREPARED_TWO_STREET_EVALUATION_ADAPTER_ID", "PREPARED_PROFILE_NORMALIZATION_ID", "PREPARED_PROFILE_RAW_ID",
        "PREPARED_PROFILE_EFFECTIVE_ID", "PREPARED_EVALUATION_OUTPUT_ID", "PREPARED_PROFILE_ABSENT",
        "PREPARED_PROFILE_NORMALIZATION_TOLERANCE", "PREPARED_EVALUATION_TOLERANCE", "PreparedEvaluationStatus",
        "PreparedCorrespondenceMode", "PreparedCorrespondenceStatus", "PreparedEvaluationLimits", "PreparedEvaluationRequest",
        "PreparedActionProbability", "PreparedProfileEntry", "PreparedPlayerProfile", "PreparedProfileNormalizationRecord",
        "PreparedProfileNormalization", "PreparedFixedProfileValue", "PreparedResponseAssignment", "PreparedPureResponse",
        "PreparedResponseActionSet", "PreparedResponseVariation", "PreparedExactResponseValue", "PreparedEvaluationTraceRecord",
        "PreparedEvaluation", "PreparedEvaluationIdentity", "PreparedEvaluationError", "PreparedEvaluationResult",
        "evaluate_prepared_two_street",
    ]
    assert evaluation_module.__all__ == expected
    assert PREPARED_TWO_STREET_EVALUATION_ADAPTER_ID == "prepared-two-street-evaluation-adapter-v1"
    assert [item.value for item in PreparedCorrespondenceMode] == ["NOT_REQUESTED", "FULL"]
    assert [item.value for item in PreparedEvaluationStatus] == ["SUCCESS", "INVALID_INPUT", "INCOMPLETE_PROFILE", "ILLEGAL_PROFILE_REFERENCE", "INVALID_PROFILE_PROBABILITY", "BUILD_MISMATCH", "IDENTITY_MISMATCH", "CAP_EXCEEDED", "NUMERIC_FAILURE", "FIXED_PROFILE_FAILURE", "EXACT_RESPONSE_FAILURE", "NON_REPRODUCIBLE", "ORACLE_MISMATCH", "UNSUPPORTED_DOWNSTREAM"]
    expected_fields = {
        PreparedEvaluationLimits: ["max_profile_information_sets_per_player", "max_actions_per_profile_entry", "max_total_profile_probabilities", "max_validation_trace_records", "max_full_correspondence_strategies", "max_enumerator_pure_strategies", "max_result_records"],
        PreparedEvaluationRequest: ["method", "correspondence_mode", "oracle_check", "downstream_request", "expected_output_semantic_sha256"],
        PreparedActionProbability: ["action_label", "probability"], PreparedProfileEntry: ["info_set_id", "action_probabilities"], PreparedPlayerProfile: ["entries"],
        PreparedProfileNormalizationRecord: ["info_set_id", "legal_action_labels", "raw_probabilities", "raw_sum", "normalization_factor", "effective_probabilities"],
        PreparedProfileNormalization: ["raw_profile_sha256", "effective_profile_sha256", "records"],
        PreparedFixedProfileValue: ["hero_ev", "villain_ev", "house_rake", "conservation_residual"],
        PreparedResponseAssignment: ["info_set_id", "action_label"], PreparedPureResponse: ["assignments"],
        PreparedResponseActionSet: ["info_set_id", "action_labels"], PreparedResponseVariation: ["info_set_id", "action_labels"],
        PreparedExactResponseValue: ["villain_max_ev", "hero_ev_worst", "hero_ev_best", "house_rake_worst", "house_rake_best", "num_villain_pure_strategies", "num_best_response_strategies", "representative_pure_response", "best_response_action_sets", "best_response_action_variation", "off_path_info_sets", "full_correspondence_status", "full_correspondence"],
        PreparedEvaluationTraceRecord: ["phase", "subject", "outcome"],
        PreparedEvaluation: ["hero_profile_normalization", "villain_profile_normalization", "fixed_profile_value", "exact_response", "validation_trace"],
        PreparedEvaluationIdentity: ["m14_contract_version", "m14_builder_id", "m14_action_label_id", "m14_normalization_id", "m14_information_key_id", "m14_raw_sha256", "m14_prepared_semantic_sha256", "m14_ordered_tree_sha256", "m14_run_identity", "hero_raw_profile_sha256", "hero_effective_profile_sha256", "villain_raw_profile_sha256", "villain_effective_profile_sha256", "evaluation_adapter_id", "profile_normalization_id", "profile_raw_id", "profile_effective_id", "output_semantic_id", "exact_response_method", "profile_tolerance_hex", "evaluation_tolerance_hex", "effective_limits", "correspondence_mode", "oracle_check", "output_semantic_sha256"],
        PreparedEvaluationError: ["message", "phase"], PreparedEvaluationResult: ["status", "evaluation", "identity", "error"],
    }
    for data_type, names in expected_fields.items():
        assert [field.name for field in fields(data_type)] == names
    assert list(inspect.signature(evaluate_prepared_two_street).parameters) == ["build", "hero_profile", "villain_profile", "limits", "request"]
    assert not hasattr(repeated_poker, "evaluate_prepared_two_street")


def test_adapter_imports_no_m14_private_helper():
    tree = ast.parse(Path(evaluation_module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("prepared_two_street"):
            assert all(not alias.name.startswith("_") for alias in node.names)


def test_dp_is_default_primary_and_absent_villain_is_explicit():
    build = _build(_one_street_spec())
    result = evaluate_prepared_two_street(build, _profile(build, PreparedPlayer.HERO))
    _assert_success(result)
    assert result.evaluation.fixed_profile_value is None
    assert result.evaluation.villain_profile_normalization is None
    assert result.identity.villain_raw_profile_sha256 == result.identity.villain_effective_profile_sha256 == PREPARED_PROFILE_ABSENT
    assert result.evaluation.exact_response.full_correspondence is None
    assert result.evaluation.exact_response.full_correspondence_status is PreparedCorrespondenceStatus.NOT_REQUESTED


@pytest.mark.parametrize("evaluation_request", [PreparedEvaluationRequest(method="enumerate"), PreparedEvaluationRequest(method="other"), PreparedEvaluationRequest(downstream_request="pipeline")])
def test_enumerate_requires_explicit_oracle_and_unknown_methods_are_unsupported(evaluation_request):
    build = _build(_one_street_spec())
    _assert_failure(evaluate_prepared_two_street(build, _profile(build, PreparedPlayer.HERO), request=evaluation_request), PreparedEvaluationStatus.UNSUPPORTED_DOWNSTREAM)


def test_downstream_requests_are_unsupported_without_fallback():
    build = _build(_one_street_spec())
    _assert_failure(evaluate_prepared_two_street(build, _profile(build, PreparedPlayer.HERO), request=PreparedEvaluationRequest(downstream_request="equilibrium")), PreparedEvaluationStatus.UNSUPPORTED_DOWNSTREAM)


def _fixed_result(spec, hero_overrides, villain_overrides):
    build = _build(spec)
    hero = _profile(build, PreparedPlayer.HERO, hero_overrides(build))
    villain = _profile(build, PreparedPlayer.VILLAIN, villain_overrides(build))
    result = evaluate_prepared_two_street(build, hero, villain)
    _assert_success(result)
    return result.evaluation.fixed_profile_value


def test_m14_one_street_fold_oracle():
    value = _fixed_result(_one_street_spec(), lambda b: {_artifact(b, PreparedPlayer.HERO, ("fold", "call")).info_set_id: {"fold": 1.0, "call": 0.0}}, lambda b: {_artifact(b, PreparedPlayer.VILLAIN, ("check", "bet::open-2")).info_set_id: {"check": 0.0, "bet::open-2": 1.0}})
    assert value.hero_ev == pytest.approx(-1.0) and value.villain_ev == pytest.approx(1.0) and value.house_rake == 0.0


def test_m14_called_showdown_oracle():
    value = _fixed_result(_one_street_spec(), lambda b: {_artifact(b, PreparedPlayer.HERO, ("fold", "call")).info_set_id: {"fold": 0.0, "call": 1.0}}, lambda b: {_artifact(b, PreparedPlayer.VILLAIN, ("check", "bet::open-2")).info_set_id: {"check": 0.0, "bet::open-2": 1.0}})
    assert (value.hero_ev, value.villain_ev, value.house_rake) == pytest.approx((0.99, -1.29, 0.3))


def test_m14_two_street_fixed_oracle():
    build = _build(_two_street_spec())
    result = evaluate_prepared_two_street(build, _profile(build, PreparedPlayer.HERO), _profile(build, PreparedPlayer.VILLAIN))
    _assert_success(result)
    assert (result.evaluation.fixed_profile_value.hero_ev, result.evaluation.fixed_profile_value.villain_ev, result.evaluation.fixed_profile_value.house_rake) == pytest.approx((0.008, -0.208, 0.2))


def test_m14_initial_all_in_oracle():
    build = _build(_initial_all_in_spec())
    result = evaluate_prepared_two_street(build, PreparedPlayerProfile(()), PreparedPlayerProfile(()))
    _assert_success(result)
    assert (result.evaluation.fixed_profile_value.hero_ev, result.evaluation.fixed_profile_value.villain_ev, result.evaluation.fixed_profile_value.house_rake) == pytest.approx((0.5, -0.5, 0.0))


def test_independent_mixed_fixed_profile_oracle():
    build = _build(_one_street_spec())
    root = _artifact(build, PreparedPlayer.VILLAIN, ("check", "bet::open-2"))
    facing = _artifact(build, PreparedPlayer.HERO, ("fold", "call"))
    hero = _profile(build, PreparedPlayer.HERO, {facing.info_set_id: {"fold": 0.4, "call": 0.6}})
    villain = _profile(build, PreparedPlayer.VILLAIN, {root.info_set_id: {"check": 0.25, "bet::open-2": 0.75}})
    result = evaluate_prepared_two_street(build, hero, villain)
    _assert_success(result)
    value = result.evaluation.fixed_profile_value
    assert (value.hero_ev, value.villain_ev, value.house_rake, value.conservation_residual) == pytest.approx((0.133, -0.293, 0.16, 0.0))


def test_tiny_dp_enumerate_oracle_agreement():
    build = _build(_one_street_spec())
    result = evaluate_prepared_two_street(build, _profile(build, PreparedPlayer.HERO), request=PreparedEvaluationRequest(method="enumerate", oracle_check=True, correspondence_mode=PreparedCorrespondenceMode.FULL))
    _assert_success(result)
    exact = result.evaluation.exact_response
    assert exact.full_correspondence_status is PreparedCorrespondenceStatus.MATERIALIZED
    assert len(exact.full_correspondence) == exact.num_best_response_strategies
    assert exact.villain_max_ev == pytest.approx(1.0)


def test_multiple_best_responses_keep_representative_count_summary_and_full_distinct():
    build = _build(_one_street_spec(check_share=0.0, rake=0.0))
    facing = _artifact(build, PreparedPlayer.HERO, ("fold", "call"))
    hero = _profile(build, PreparedPlayer.HERO, {facing.info_set_id: {"fold": 1.0, "call": 0.0}})
    default = evaluate_prepared_two_street(build, hero)
    full = evaluate_prepared_two_street(build, hero, request=PreparedEvaluationRequest(correspondence_mode=PreparedCorrespondenceMode.FULL))
    _assert_success(default); _assert_success(full)
    assert default.evaluation.exact_response.num_best_response_strategies == 2
    assert default.evaluation.exact_response.full_correspondence is None
    assert len(default.evaluation.exact_response.representative_pure_response.assignments) == 1
    assert len(full.evaluation.exact_response.full_correspondence) == 2


def test_zero_reach_hero_and_villain_info_sets_still_require_complete_profiles():
    build = _build(_one_street_spec())
    hero = _profile(build, PreparedPlayer.HERO)
    _assert_failure(evaluate_prepared_two_street(build, PreparedPlayerProfile(hero.entries[:-1])), PreparedEvaluationStatus.INCOMPLETE_PROFILE)
    villain = _profile(build, PreparedPlayer.VILLAIN)
    _assert_failure(evaluate_prepared_two_street(build, hero, PreparedPlayerProfile(())), PreparedEvaluationStatus.INCOMPLETE_PROFILE)
    _assert_success(evaluate_prepared_two_street(build, hero, villain))


def test_zero_reach_villain_global_freedom_differs_from_conditional_action_set():
    build = _build(_avoided_villain_subtree_spec())
    facing = _artifact(build, PreparedPlayer.HERO, ("raise::r4", "fold", "call"))
    hero = _profile(build, PreparedPlayer.HERO, {facing.info_set_id: {"raise::r4": 1.0, "fold": 0.0, "call": 0.0}})
    result = evaluate_prepared_two_street(build, hero)
    _assert_success(result)
    exact = result.evaluation.exact_response
    downstream = next(item for item in exact.best_response_action_sets if item.action_labels == ("fold",))
    variation = next(item for item in exact.best_response_action_variation if item.info_set_id == downstream.info_set_id)
    assert variation.action_labels == ("fold", "call")
    assert downstream.info_set_id not in exact.off_path_info_sets


def test_dp_near_tie_fields_are_not_asserted_as_one_exact_triple():
    build = _build(_one_street_spec(check_share=0.0, rake=0.0))
    result = evaluate_prepared_two_street(build, _profile(build, PreparedPlayer.HERO))
    _assert_success(result)
    assert result.evaluation.exact_response.hero_ev_worst <= result.evaluation.exact_response.hero_ev_best
    assert result.evaluation.exact_response.house_rake_worst >= 0


def _mutate_profile(build, kind):
    profile = _profile(build, PreparedPlayer.HERO)
    first = profile.entries[0]
    if kind == "missing_info": return PreparedPlayerProfile(profile.entries[1:])
    if kind == "extra_info": return PreparedPlayerProfile(profile.entries + (replace(first, info_set_id="info:sha256:" + "f" * 64),))
    if kind == "duplicate_info": return PreparedPlayerProfile(profile.entries + (first,))
    if kind == "missing_action": return replace(profile, entries=(replace(first, action_probabilities=first.action_probabilities[:-1]),) + profile.entries[1:])
    if kind == "unknown_action": return replace(profile, entries=(replace(first, action_probabilities=first.action_probabilities + (PreparedActionProbability("raise::unknown", 0.0),)),) + profile.entries[1:])
    if kind == "duplicate_action": return replace(profile, entries=(replace(first, action_probabilities=first.action_probabilities + (first.action_probabilities[0],)),) + profile.entries[1:])


@pytest.mark.parametrize(("kind", "status"), [("missing_info", PreparedEvaluationStatus.INCOMPLETE_PROFILE), ("extra_info", PreparedEvaluationStatus.ILLEGAL_PROFILE_REFERENCE), ("duplicate_info", PreparedEvaluationStatus.INVALID_INPUT), ("missing_action", PreparedEvaluationStatus.INCOMPLETE_PROFILE), ("unknown_action", PreparedEvaluationStatus.ILLEGAL_PROFILE_REFERENCE), ("duplicate_action", PreparedEvaluationStatus.INVALID_INPUT)])
def test_profile_structure_failure_matrix(kind, status):
    build = _build(_one_street_spec())
    _assert_failure(evaluate_prepared_two_street(build, _mutate_profile(build, kind)), status)


@pytest.mark.parametrize(("value", "status"), [(True, PreparedEvaluationStatus.INVALID_INPUT), ("1", PreparedEvaluationStatus.INVALID_INPUT), (math.nan, PreparedEvaluationStatus.INVALID_PROFILE_PROBABILITY), (math.inf, PreparedEvaluationStatus.INVALID_PROFILE_PROBABILITY), (-math.inf, PreparedEvaluationStatus.INVALID_PROFILE_PROBABILITY), (-0.1, PreparedEvaluationStatus.INVALID_PROFILE_PROBABILITY), (0.0, PreparedEvaluationStatus.INVALID_PROFILE_PROBABILITY), (1.000000002, PreparedEvaluationStatus.INVALID_PROFILE_PROBABILITY)])
def test_probability_failure_matrix(value, status):
    build = _build(_one_street_spec())
    profile = _profile(build, PreparedPlayer.HERO)
    entry = profile.entries[0]
    probabilities = tuple(replace(item, probability=value if index == 0 else 0.0) for index, item in enumerate(entry.action_probabilities))
    broken = replace(profile, entries=(replace(entry, action_probabilities=probabilities),) + profile.entries[1:])
    _assert_failure(evaluate_prepared_two_street(build, broken), status)


def test_zero_probability_and_within_tolerance_normalization_are_valid():
    build = _build(_one_street_spec())
    facing = _artifact(build, PreparedPlayer.HERO, ("fold", "call"))
    profile = _profile(build, PreparedPlayer.HERO, {facing.info_set_id: {"fold": 0.5000000004, "call": 0.5}})
    result = evaluate_prepared_two_street(build, profile)
    _assert_success(result)
    record = next(item for item in result.evaluation.hero_profile_normalization.records if item.info_set_id == facing.info_set_id)
    assert record.raw_sum == math.fsum((0.5000000004, 0.5))
    assert record.normalization_factor == 1.0 / record.raw_sum
    assert record.effective_probabilities == tuple(value * record.normalization_factor for value in record.raw_probabilities)


def test_profile_input_order_is_semantically_irrelevant():
    build = _build(_one_street_spec())
    forward = evaluate_prepared_two_street(build, _profile(build, PreparedPlayer.HERO))
    reverse = evaluate_prepared_two_street(build, _profile(build, PreparedPlayer.HERO, reverse=True))
    _assert_success(forward); _assert_success(reverse)
    assert forward.identity == reverse.identity and forward.evaluation == reverse.evaluation


def test_raw_and_effective_profile_hashes_are_distinct_and_floathex_bound():
    build = _build(_one_street_spec())
    facing = _artifact(build, PreparedPlayer.HERO, ("fold", "call"))
    exact = evaluate_prepared_two_street(build, _profile(build, PreparedPlayer.HERO, {facing.info_set_id: {"fold": 0.5, "call": 0.5}}))
    perturbed = evaluate_prepared_two_street(build, _profile(build, PreparedPlayer.HERO, {facing.info_set_id: {"fold": 0.5000000004, "call": 0.5}}))
    _assert_success(exact); _assert_success(perturbed)
    assert exact.identity.hero_raw_profile_sha256 != perturbed.identity.hero_raw_profile_sha256
    assert perturbed.identity.hero_raw_profile_sha256 != perturbed.identity.hero_effective_profile_sha256


def test_build_tree_count_info_normalization_run_identity_fault_injection():
    build = _build(_two_street_spec())
    hero = _profile(build, PreparedPlayer.HERO)
    faults = [
        (replace(build, identity=replace(build.identity, builder_id="wrong")), PreparedEvaluationStatus.IDENTITY_MISMATCH),
        (replace(build, identity=replace(build.identity, raw_sha256="x")), PreparedEvaluationStatus.IDENTITY_MISMATCH),
        (replace(build, counts=replace(build.counts, total_nodes=build.counts.total_nodes + 1)), PreparedEvaluationStatus.BUILD_MISMATCH),
        (replace(build, counts=replace(build.counts, hero_pure_plans=build.counts.hero_pure_plans + 1)), PreparedEvaluationStatus.BUILD_MISMATCH),
        (replace(build, counts=replace(build.counts, villain_pure_strategies=build.counts.villain_pure_strategies + 1)), PreparedEvaluationStatus.BUILD_MISMATCH),
        (replace(build, information_sets=build.information_sets[:-1]), PreparedEvaluationStatus.BUILD_MISMATCH),
        (replace(build, chance_normalization=build.chance_normalization[:-1]), PreparedEvaluationStatus.BUILD_MISMATCH),
    ]
    for faulty, status in faults:
        _assert_failure(evaluate_prepared_two_street(faulty, hero), status)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record: replace(record, raw_sum=2.0, normalization_factor=0.5),
        lambda record: replace(record, raw_sum=1.0 + 5e-10),
        lambda record: replace(record, normalization_factor=math.nextafter(record.normalization_factor, math.inf)),
        lambda record: replace(record, effective_probabilities=tuple(reversed(record.effective_probabilities))),
    ],
)
def test_normalization_raw_factor_and_effective_faults_fail_before_core(monkeypatch, mutate):
    build = _build(_two_street_spec())
    hero = _profile(build, PreparedPlayer.HERO)
    record = build.chance_normalization[0]
    faulty = replace(build, chance_normalization=(mutate(record),))
    calls = {"fixed": 0, "dp": 0}
    monkeypatch.setattr(evaluation_module, "evaluate_fixed_profile", lambda *a, **k: calls.__setitem__("fixed", calls["fixed"] + 1))
    monkeypatch.setattr(evaluation_module, "solve_exact_response", lambda *a, **k: calls.__setitem__("dp", calls["dp"] + 1))
    _assert_failure(evaluate_prepared_two_street(faulty, hero), PreparedEvaluationStatus.BUILD_MISMATCH)
    assert calls == {"fixed": 0, "dp": 0}


def test_normalization_artifact_is_bound_to_materialized_transition_probabilities(monkeypatch):
    build = _build(_two_street_spec())
    hero = _profile(build, PreparedPlayer.HERO)
    record = build.chance_normalization[0]
    faulty = _replace_transition_probabilities_and_rehash(build, tuple(reversed(record.effective_probabilities)))
    calls = {"dp": 0}
    monkeypatch.setattr(evaluation_module, "solve_exact_response", lambda *a, **k: calls.__setitem__("dp", calls["dp"] + 1))
    _assert_failure(evaluate_prepared_two_street(faulty, hero), PreparedEvaluationStatus.BUILD_MISMATCH)
    assert calls == {"dp": 0}


def test_duplicate_node_id_dag_cycle_faults_are_rejected():
    build = _build(_two_street_spec())
    hero = _profile(build, PreparedPlayer.HERO)
    root = build.tree.root
    def find_transition(node):
        if isinstance(node, ChanceNode) and len(node.children) == 2:
            return node
        edges = node.children if isinstance(node, ChanceNode) else node.actions if isinstance(node, (HeroNode, VillainNode)) else ()
        for _, child in edges:
            found = find_transition(child)
            if found is not None:
                return found
        return None
    transition = find_transition(root)
    original_children = transition.children
    duplicate = replace(original_children[1][1], node_id=original_children[0][1].node_id)
    object.__setattr__(transition, "children", (original_children[0], (original_children[1][0], duplicate)))
    _assert_failure(evaluate_prepared_two_street(build, hero), PreparedEvaluationStatus.BUILD_MISMATCH)
    object.__setattr__(transition, "children", original_children)
    object.__setattr__(transition, "children", (original_children[0], (original_children[1][0], original_children[0][1])))
    _assert_failure(evaluate_prepared_two_street(build, hero), PreparedEvaluationStatus.BUILD_MISMATCH)
    object.__setattr__(transition, "children", original_children)


def test_cap_boundaries_and_cap_before_core_allocation(monkeypatch):
    build = _build(_one_street_spec())
    hero = _profile(build, PreparedPlayer.HERO)
    exact_trace = 7 + len(hero.entries) + 1
    _assert_success(evaluate_prepared_two_street(build, hero, limits=replace(PreparedEvaluationLimits(), max_validation_trace_records=exact_trace)))
    calls = {"fixed": 0, "dp": 0}
    monkeypatch.setattr(evaluation_module, "evaluate_fixed_profile", lambda *a, **k: calls.__setitem__("fixed", calls["fixed"] + 1))
    monkeypatch.setattr(evaluation_module, "solve_exact_response", lambda *a, **k: calls.__setitem__("dp", calls["dp"] + 1))
    _assert_failure(evaluate_prepared_two_street(build, hero, limits=replace(PreparedEvaluationLimits(), max_validation_trace_records=exact_trace - 1)), PreparedEvaluationStatus.CAP_EXCEEDED)
    assert calls == {"fixed": 0, "dp": 0}


@pytest.mark.parametrize("field,value", [("max_actions_per_profile_entry", 1), ("max_total_profile_probabilities", 2), ("max_profile_information_sets_per_player", 1)])
def test_profile_caps_reject_before_mapping_or_core(monkeypatch, field, value):
    build = _build(_one_street_spec())
    hero = _profile(build, PreparedPlayer.HERO)
    calls = {"normalize": 0, "dp": 0}
    monkeypatch.setattr(evaluation_module, "_normalize_profile", lambda *a, **k: calls.__setitem__("normalize", calls["normalize"] + 1))
    monkeypatch.setattr(evaluation_module, "solve_exact_response", lambda *a, **k: calls.__setitem__("dp", calls["dp"] + 1))
    _assert_failure(evaluate_prepared_two_street(build, hero, limits=replace(PreparedEvaluationLimits(), **{field: value})), PreparedEvaluationStatus.CAP_EXCEEDED)
    assert calls == {"normalize": 0, "dp": 0}


def test_profile_cap_exact_boundaries_are_accepted():
    build = _build(_one_street_spec())
    limits = replace(PreparedEvaluationLimits(), max_profile_information_sets_per_player=2, max_actions_per_profile_entry=2, max_total_profile_probabilities=3)
    _assert_success(evaluate_prepared_two_street(build, _profile(build, PreparedPlayer.HERO), limits=limits))


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, 1001])
def test_invalid_or_above_ceiling_limits_are_rejected(value):
    build = _build(_one_street_spec())
    limits = replace(PreparedEvaluationLimits(), max_profile_information_sets_per_player=value)
    _assert_failure(evaluate_prepared_two_street(build, _profile(build, PreparedPlayer.HERO), limits=limits), PreparedEvaluationStatus.INVALID_INPUT)


def test_full_and_enumerator_caps_fail_without_payload(monkeypatch):
    build = _build(_one_street_spec(check_share=0.0, rake=0.0))
    hero = _profile(build, PreparedPlayer.HERO)
    original = evaluation_module.solve_exact_response
    calls = {"dp": 0, "enumerate": 0}
    def counted(*args, **kwargs):
        calls[kwargs["method"]] += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(evaluation_module, "solve_exact_response", counted)
    _assert_failure(evaluate_prepared_two_street(build, hero, limits=replace(PreparedEvaluationLimits(), max_full_correspondence_strategies=1), request=PreparedEvaluationRequest(correspondence_mode=PreparedCorrespondenceMode.FULL)), PreparedEvaluationStatus.CAP_EXCEEDED)
    assert calls == {"dp": 1, "enumerate": 0}
    calls = {"dp": 0, "enumerate": 0}
    _assert_failure(evaluate_prepared_two_street(build, hero, limits=replace(PreparedEvaluationLimits(), max_enumerator_pure_strategies=1), request=PreparedEvaluationRequest(oracle_check=True)), PreparedEvaluationStatus.CAP_EXCEEDED)
    assert calls == {"dp": 1, "enumerate": 0}


def test_result_and_oracle_record_caps_accept_exact_and_reject_cap_minus_one():
    build = _build(_one_street_spec())
    hero = _profile(build, PreparedPlayer.HERO)
    baseline = evaluate_prepared_two_street(build, hero)
    _assert_success(baseline)
    exact = _returned_record_count(baseline)
    _assert_success(evaluate_prepared_two_street(build, hero, limits=replace(PreparedEvaluationLimits(), max_result_records=exact)))
    _assert_failure(evaluate_prepared_two_street(build, hero, limits=replace(PreparedEvaluationLimits(), max_result_records=exact - 1)), PreparedEvaluationStatus.CAP_EXCEEDED)
    oracle = evaluate_prepared_two_street(build, hero, request=PreparedEvaluationRequest(oracle_check=True))
    _assert_success(oracle)
    villain_info_count = len(oracle.evaluation.exact_response.representative_pure_response.assignments)
    diagnostics = oracle.evaluation.exact_response.num_villain_pure_strategies * (villain_info_count + 2)
    comparison = oracle.evaluation.exact_response.num_best_response_strategies * (villain_info_count + 1)
    oracle_exact = _returned_record_count(oracle) + diagnostics + comparison
    _assert_success(evaluate_prepared_two_street(build, hero, limits=replace(PreparedEvaluationLimits(), max_result_records=oracle_exact), request=PreparedEvaluationRequest(oracle_check=True)))
    _assert_failure(evaluate_prepared_two_street(build, hero, limits=replace(PreparedEvaluationLimits(), max_result_records=oracle_exact - 1), request=PreparedEvaluationRequest(oracle_check=True)), PreparedEvaluationStatus.CAP_EXCEEDED)


def test_minimum_result_cap_precedes_mapping_strategy_core_and_nested_allocations(monkeypatch):
    build = _build(_one_street_spec())
    hero = _profile(build, PreparedPlayer.HERO)
    calls = {"normalize": 0, "strategy": 0, "dp": 0, "canonical": 0, "full": 0, "trace": 0}
    originals = {
        "normalize": evaluation_module._normalize_profile,
        "strategy": evaluation_module.HeroStrategy,
        "dp": evaluation_module.solve_exact_response,
        "canonical": evaluation_module._canonical_response,
        "full": evaluation_module._full_tuple,
        "trace": evaluation_module._trace,
    }

    def counted(name):
        def call(*args, **kwargs):
            calls[name] += 1
            return originals[name](*args, **kwargs)
        return call

    monkeypatch.setattr(evaluation_module, "_normalize_profile", counted("normalize"))
    monkeypatch.setattr(evaluation_module, "HeroStrategy", counted("strategy"))
    monkeypatch.setattr(evaluation_module, "solve_exact_response", counted("dp"))
    monkeypatch.setattr(evaluation_module, "_canonical_response", counted("canonical"))
    monkeypatch.setattr(evaluation_module, "_full_tuple", counted("full"))
    monkeypatch.setattr(evaluation_module, "_trace", counted("trace"))
    result = evaluate_prepared_two_street(
        build,
        hero,
        limits=replace(PreparedEvaluationLimits(), max_result_records=1),
    )
    _assert_failure(result, PreparedEvaluationStatus.CAP_EXCEEDED)
    assert calls == {"normalize": 0, "strategy": 0, "dp": 0, "canonical": 0, "full": 0, "trace": 0}


def test_exact_response_record_cap_precedes_prepared_nested_conversion(monkeypatch):
    build = _build(_one_street_spec(check_share=0.0, rake=0.0))
    hero = _profile(build, PreparedPlayer.HERO)
    baseline = evaluate_prepared_two_street(build, hero)
    _assert_success(baseline)
    exact = _returned_record_count(baseline)
    calls = {"dp": 0, "canonical": 0, "trace": 0}
    original_dp = evaluation_module.solve_exact_response
    original_canonical = evaluation_module._canonical_response
    original_trace = evaluation_module._trace

    def counted_dp(*args, **kwargs):
        calls["dp"] += 1
        return original_dp(*args, **kwargs)

    def counted_canonical(*args, **kwargs):
        calls["canonical"] += 1
        return original_canonical(*args, **kwargs)

    def counted_trace(*args, **kwargs):
        calls["trace"] += 1
        return original_trace(*args, **kwargs)

    monkeypatch.setattr(evaluation_module, "solve_exact_response", counted_dp)
    monkeypatch.setattr(evaluation_module, "_canonical_response", counted_canonical)
    monkeypatch.setattr(evaluation_module, "_trace", counted_trace)
    _assert_failure(
        evaluate_prepared_two_street(build, hero, limits=replace(PreparedEvaluationLimits(), max_result_records=exact - 1)),
        PreparedEvaluationStatus.CAP_EXCEEDED,
    )
    assert calls == {"dp": 1, "canonical": 0, "trace": 0}


def test_same_runtime_identity_output_hash_and_expected_hash_contract():
    build = _build(_one_street_spec())
    hero = _profile(build, PreparedPlayer.HERO)
    first = evaluate_prepared_two_street(build, hero)
    second = evaluate_prepared_two_street(build, hero)
    _assert_success(first); _assert_success(second)
    assert first == second
    accepted = evaluate_prepared_two_street(build, hero, request=PreparedEvaluationRequest(expected_output_semantic_sha256=first.identity.output_semantic_sha256))
    _assert_success(accepted)
    wrong = ("0" if first.identity.output_semantic_sha256[0] != "0" else "1") + first.identity.output_semantic_sha256[1:]
    _assert_failure(evaluate_prepared_two_street(build, hero, request=PreparedEvaluationRequest(expected_output_semantic_sha256=wrong)), PreparedEvaluationStatus.NON_REPRODUCIBLE)


def test_pinned_deterministic_identity_profile_and_output_hash():
    build = _build(_one_street_spec())
    result = evaluate_prepared_two_street(build, _profile(build, PreparedPlayer.HERO))
    _assert_success(result)
    assert build.identity.run_identity == "5744f433637099f44538c12474c0dbf8a497c78f03d969d055c605db7c0df430"
    assert result.identity.hero_raw_profile_sha256 == "71dcfb4951ccacedafa6868730f3adf0b90ce3b5534252d1410f26739c24e944"
    assert result.identity.hero_effective_profile_sha256 == "3d4f4879a5fd2102cf50eef9fce9386d3679d983b988521445fd5532ef70b718"
    assert result.identity.output_semantic_sha256 == "f84a0b8d4fa40b62d8d887b2f7bf256813ebfaefee113cfd92eb570469f6d996"


@pytest.mark.parametrize("fault", ["best_gt_pure", "duplicate_full", "variation", "off_path"])
def test_invalid_exact_response_count_and_discrete_structure_fail_without_partial(monkeypatch, fault):
    build = _build(_one_street_spec(check_share=0.0, rake=0.0))
    hero = _profile(build, PreparedPlayer.HERO)
    original = evaluation_module.solve_exact_response

    def broken(*args, **kwargs):
        result = original(*args, **kwargs)
        if fault == "best_gt_pure":
            result.num_best_response_strategies = result.num_villain_pure_strategies + 1
        elif fault == "duplicate_full":
            result.best_response_strategies = [result.best_response_strategies[0], result.best_response_strategies[0]]
            result.num_best_response_strategies = 2
        elif fault == "variation":
            info_id = next(iter(result.best_response_action_variation))
            result.best_response_action_variation[info_id] = ["unknown", result.best_response_action_variation[info_id][0]]
        else:
            result.off_path_info_sets = ["info:sha256:" + "f" * 64]
        return result

    monkeypatch.setattr(evaluation_module, "solve_exact_response", broken)
    _assert_failure(evaluate_prepared_two_street(build, hero), PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE)


def test_positive_subnormal_stays_positive_and_nonfinite_derived_fails_numeric(monkeypatch):
    build = _build(_one_street_spec())
    facing = _artifact(build, PreparedPlayer.HERO, ("fold", "call"))
    hero = _profile(build, PreparedPlayer.HERO, {facing.info_set_id: {"fold": math.ulp(0.0), "call": 1.0}})
    accepted = evaluate_prepared_two_street(build, hero)
    _assert_success(accepted)
    record = next(item for item in accepted.evaluation.hero_profile_normalization.records if item.info_set_id == facing.info_set_id)
    assert record.raw_probabilities[0] > 0.0 and record.effective_probabilities[0] > 0.0

    original_fsum = evaluation_module.math.fsum
    original_verify = evaluation_module._verify_build
    calls = {"count": 0, "armed": False}

    def nonfinite_effective(values):
        if calls["armed"]:
            calls["count"] += 1
        if calls["armed"] and calls["count"] == 2:
            return math.inf
        return original_fsum(values)

    def verify_then_arm(candidate):
        view = original_verify(candidate)
        calls["armed"] = True
        return view

    monkeypatch.setattr(evaluation_module.math, "fsum", nonfinite_effective)
    monkeypatch.setattr(evaluation_module, "_verify_build", verify_then_arm)
    _assert_failure(evaluate_prepared_two_street(build, hero), PreparedEvaluationStatus.NUMERIC_FAILURE)


def test_fixed_dp_oracle_injected_failures_have_no_partial_payload(monkeypatch):
    build = _build(_one_street_spec())
    hero = _profile(build, PreparedPlayer.HERO)
    villain = _profile(build, PreparedPlayer.VILLAIN)
    monkeypatch.setattr(evaluation_module, "evaluate_fixed_profile", lambda *a, **k: (_ for _ in ()).throw(ValueError("fixed")))
    _assert_failure(evaluate_prepared_two_street(build, hero, villain), PreparedEvaluationStatus.FIXED_PROFILE_FAILURE)
    monkeypatch.undo()
    monkeypatch.setattr(evaluation_module, "solve_exact_response", lambda *a, **k: (_ for _ in ()).throw(ValueError("dp")))
    _assert_failure(evaluate_prepared_two_street(build, hero), PreparedEvaluationStatus.EXACT_RESPONSE_FAILURE)


def test_nonfinite_core_and_oracle_mismatch_have_no_partial_payload(monkeypatch):
    build = _build(_one_street_spec())
    hero = _profile(build, PreparedPlayer.HERO)
    original = evaluation_module.solve_exact_response
    def nonfinite(*args, **kwargs):
        result = original(*args, **kwargs)
        result.villain_max_ev = math.nan
        return result
    monkeypatch.setattr(evaluation_module, "solve_exact_response", nonfinite)
    _assert_failure(evaluate_prepared_two_street(build, hero), PreparedEvaluationStatus.NUMERIC_FAILURE)
    monkeypatch.undo()
    def mismatch(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("method") == "enumerate":
            result.villain_max_ev += 1.0
        return result
    monkeypatch.setattr(evaluation_module, "solve_exact_response", mismatch)
    _assert_failure(evaluate_prepared_two_street(build, hero, request=PreparedEvaluationRequest(oracle_check=True)), PreparedEvaluationStatus.ORACLE_MISMATCH)


def test_keyboard_interrupt_and_system_exit_are_not_normal_failures(monkeypatch):
    build = _build(_one_street_spec())
    hero = _profile(build, PreparedPlayer.HERO)
    for exc in (KeyboardInterrupt, SystemExit):
        monkeypatch.setattr(evaluation_module, "count_villain_pure_strategies", lambda *a, _exc=exc, **k: (_ for _ in ()).throw(_exc()))
        with pytest.raises(exc):
            evaluate_prepared_two_street(build, hero)


def test_scenario_v1_and_existing_public_types_are_unchanged():
    assert SUPPORTED_FORMAT_VERSIONS == ("1",)
    assert fields(RiverScenarioBettingTree)
    assert not hasattr(repeated_poker, "PreparedEvaluationResult")


def test_existing_generic_fixed_profile_and_exact_response_semantics_are_reasserted():
    villain_node = VillainNode("v", "V", (
        ("call", TerminalNode("call", 2.0, -3.0, 1.0)),
        ("fold", TerminalNode("fold", 1.0, -1.0, 0.0)),
    ))
    tree = GameTree(HeroNode("h", "H", (("bet", villain_node), ("check", TerminalNode("check", 0.0, 0.0, 0.0)))))
    fixed = evaluate_fixed_profile(tree, HeroStrategy({"H": {"bet": 0.5, "check": 0.5}}), VillainStrategy({"V": {"call": 0.25, "fold": 0.75}}))
    assert (fixed.hero_ev, fixed.villain_ev, fixed.house_rake) == pytest.approx((0.625, -0.75, 0.125))
    tie_tree = GameTree(VillainNode("root", "V", (("call", TerminalNode("a", -1.0, -1.0, 2.0)), ("fold", TerminalNode("b", 1.0, -1.0, 0.0)))))
    dp = solve_exact_response(tie_tree, HeroStrategy({}), max_pure_strategies=1, method="dp")
    assert dp.num_best_response_strategies == 2
    assert len(dp.best_response_strategies) == 1
    assert dp.best_response_action_variation == {"V": ["call", "fold"]}
