"""M31 contract tests for the exact three-player river/rake adapter."""

from __future__ import annotations

import copy
import inspect
import subprocess
from collections import OrderedDict
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path

import pytest

import repeated_poker.three_player_river_rake as module
from repeated_poker.three_player_response import ExactResponseLimits
from repeated_poker.three_player_river_rake import (
    CAP_EXCEEDED,
    EXACT_SCENARIO_RESPONSE_COMPLETE,
    INVALID_INPUT,
    M30_RESPONSE_FAILURE,
    NUMERIC_FAILURE,
    STALE_INPUT,
    UNSUPPORTED_MODEL,
    AwardShare,
    ExactBehaviorPolicy,
    OpponentInitialProfile,
    PerfectRecallAttestation,
    RiverAction,
    RiverChanceNode,
    RiverChanceOutcome,
    RiverDecisionNode,
    RiverObservation,
    RiverRakeIdentityPins,
    RiverRakeLimits,
    RiverTerminalNode,
    ThreePlayerRiverRakeScenario,
    create_perfect_recall_attestation,
    evaluate_three_player_river_rake,
    exact_scenario_response_json,
)


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DATE = "2026-07-24"
EVIDENCE_VERSION = "m31-t1-test-human-review-v1"
AUTO = object()


def observation(
    suffix: str = "base",
    *,
    private: dict[str, str] | None = None,
) -> RiverObservation:
    return RiverObservation(
        public_observation_id=f"public-{suffix}",
        private_observation_id_by_player=private
        or {
            "H": f"hero-{suffix}",
            "O1": f"o1-{suffix}",
            "O2": f"o2-{suffix}",
        },
    )


def check_tree(
    order: tuple[str, str, str] = ("O1", "O2", "H"),
    *,
    suffix: str = "base",
    shares: tuple[AwardShare, ...] = (AwardShare("H", "1"),),
) -> RiverDecisionNode:
    child: RiverDecisionNode | RiverTerminalNode = RiverTerminalNode(
        node_id=f"terminal-{suffix}",
        kind="showdown",
        award_shares=shares,
    )
    for player in reversed(order):
        child = RiverDecisionNode(
            node_id=f"node-{player}-{suffix}",
            owner=player,
            information_set_id=f"{player}_root_{suffix}",
            actions=(
                RiverAction(
                    action_id="check",
                    kind="check",
                    target_total_contribution=None,
                    child=child,
                ),
            ),
        )
    return child


def check_scenario(
    *,
    order: tuple[str, str, str] = ("O1", "O2", "H"),
    suffix: str = "base",
    shares: tuple[AwardShare, ...] = (AwardShare("H", "1"),),
    initial_observation: RiverObservation | None = None,
    **overrides: object,
) -> ThreePlayerRiverRakeScenario:
    values: dict[str, object] = {
        "root": check_tree(order, suffix=suffix, shares=shares),
        "button_player_id": "H",
        "seat_order": ("H", "O1", "O2"),
        "river_action_order": order,
        "initial_observation": initial_observation or observation(suffix),
        "initial_pot": "30",
        "initial_contribution": {"H": "10", "O1": "10", "O2": "10"},
        "max_total_contribution": {"H": "100", "O1": "100", "O2": "100"},
        "rake_rate": "0",
        "rake_cap": None,
    }
    values.update(overrides)
    return ThreePlayerRiverRakeScenario(**values)


def hero_policy_for_check(
    suffix: str = "base",
) -> ExactBehaviorPolicy:
    return ExactBehaviorPolicy({f"H_root_{suffix}": {"check": "1"}})


def attest(
    scenario: ThreePlayerRiverRakeScenario,
    **overrides: object,
) -> PerfectRecallAttestation:
    values = asdict(
        create_perfect_recall_attestation(
            scenario,
            verifier="M31-T1 test reviewer",
            verification_date=VALIDATION_DATE,
            evidence_version=EVIDENCE_VERSION,
            o1_confirmed=True,
            o2_confirmed=True,
        )
    )
    values.update(overrides)
    return PerfectRecallAttestation(**values)


def solve(
    scenario: ThreePlayerRiverRakeScenario,
    policy: ExactBehaviorPolicy,
    *,
    attestation: PerfectRecallAttestation | None | object = AUTO,
    **kwargs: object,
):
    evidence = attest(scenario) if attestation is AUTO else attestation
    return evaluate_three_player_river_rake(
        scenario,
        policy,
        attestation=evidence,
        **kwargs,
    )


def success(
    scenario: ThreePlayerRiverRakeScenario,
    policy: ExactBehaviorPolicy,
    **kwargs: object,
):
    result = solve(scenario, policy, **kwargs)
    assert result.status == EXACT_SCENARIO_RESPONSE_COMPLETE
    assert result.error is None
    assert result.partial_result is False
    assert result.scenario_evaluation is not None
    assert result.payoff_table is not None
    assert result.response is not None
    return result


def assert_no_payload(result, status: str) -> None:
    assert result.status == status
    assert result.scenario_evaluation is None
    assert result.payoff_table is None
    assert result.response is None
    assert result.error is not None
    assert result.partial_result is False
    encoded = exact_scenario_response_json(result)
    assert '"scenario_evaluation":null' in encoded
    assert '"payoff_table":null' in encoded
    assert '"response":null' in encoded
    for forbidden in (
        '"support_cells"',
        '"terminal_records"',
        '"rows"',
        '"hero_worst"',
    ):
        assert forbidden not in encoded


def multiple_response_scenario() -> ThreePlayerRiverRakeScenario:
    """Two pure equilibria have exact Hero values -20 and 20."""

    after_fold = RiverDecisionNode(
        node_id="o2-after-o1-fold",
        owner="O2",
        information_set_id="O2_after_o1_fold",
        actions=(
            RiverAction(
                "call",
                "call",
                "20",
                RiverTerminalNode(
                    "terminal-o1-fold-o2-call",
                    "showdown",
                    (AwardShare("O2", "1"),),
                ),
            ),
            RiverAction(
                "fold",
                "fold",
                None,
                RiverTerminalNode("terminal-both-fold", "fold"),
            ),
        ),
    )
    after_call = RiverDecisionNode(
        node_id="o2-after-o1-call",
        owner="O2",
        information_set_id="O2_after_o1_call",
        actions=(
            RiverAction(
                "call",
                "call",
                "20",
                RiverTerminalNode(
                    "terminal-both-call",
                    "showdown",
                    (AwardShare("H", "1"),),
                ),
            ),
            RiverAction(
                "fold",
                "fold",
                None,
                RiverTerminalNode(
                    "terminal-o1-call-o2-fold",
                    "showdown",
                    (AwardShare("H", "4/5"), AwardShare("O1", "1/5")),
                ),
            ),
        ),
    )
    o1 = RiverDecisionNode(
        node_id="o1-facing-hero-bet",
        owner="O1",
        information_set_id="O1_facing_hero_bet",
        actions=(
            RiverAction("fold", "fold", None, after_fold),
            RiverAction("call", "call", "20", after_call),
        ),
    )
    root = RiverDecisionNode(
        node_id="hero-bet-root",
        owner="H",
        information_set_id="H_bet_root",
        actions=(RiverAction("bet", "bet", "20", o1),),
    )
    return ThreePlayerRiverRakeScenario(
        root=root,
        button_player_id="H",
        seat_order=("H", "O1", "O2"),
        river_action_order=("H", "O1", "O2"),
        initial_observation=observation("multiple"),
        initial_pot="30",
        initial_contribution={"H": "10", "O1": "10", "O2": "10"},
        max_total_contribution={"H": "100", "O1": "100", "O2": "100"},
        rake_rate="0",
    )


def multiple_hero_policy() -> ExactBehaviorPolicy:
    return ExactBehaviorPolicy({"H_bet_root": {"bet": "1"}})


def multiple_initial_profile(o1_action: str) -> OpponentInitialProfile:
    return OpponentInitialProfile(
        o1_probabilities={
            "O1_facing_hero_bet": {
                "fold": "1" if o1_action == "fold" else "0",
                "call": "1" if o1_action == "call" else "0",
            }
        },
        o2_probabilities={
            "O2_after_o1_call": {"call": "0", "fold": "1"},
            "O2_after_o1_fold": {"call": "1", "fold": "0"},
        },
    )


def valid_raise_scenario(*, reraise: bool = False) -> ThreePlayerRiverRakeScenario:
    terminal = RiverTerminalNode(
        "raise-terminal",
        "showdown",
        (AwardShare("H", "1"),),
    )
    hero_call = RiverDecisionNode(
        "hero-call-raise",
        "H",
        "H_call_raise",
        (RiverAction("call", "call", "30", terminal),),
    )
    if reraise:
        o2_action = RiverAction("raise", "raise", "40", hero_call)
    else:
        o2_action = RiverAction("call", "call", "30", hero_call)
    o2 = RiverDecisionNode(
        "o2-facing-raise",
        "O2",
        "O2_facing_raise",
        (o2_action,),
    )
    o1 = RiverDecisionNode(
        "o1-facing-bet",
        "O1",
        "O1_facing_bet",
        (RiverAction("raise", "raise", "30", o2),),
    )
    root = RiverDecisionNode(
        "hero-bet",
        "H",
        "H_open",
        (RiverAction("bet", "bet", "20", o1),),
    )
    return ThreePlayerRiverRakeScenario(
        root,
        "H",
        ("H", "O1", "O2"),
        ("H", "O1", "O2"),
        observation("raise"),
        "30",
        {"H": "10", "O1": "10", "O2": "10"},
        {"H": "100", "O1": "100", "O2": "100"},
        "0",
    )


def test_a1_owner_and_non_strategic_rake_account_are_separated():
    candidate = check_scenario()
    result = success(candidate, hero_policy_for_check())
    assert result.payoff_table["rows"][0]["utility"] == {
        "H": "20",
        "O1": "-10",
        "O2": "-10",
        "R": "0",
    }
    assert result.response["counts"]["pure_plans"] == {"O1": 1, "O2": 1}
    assert "R" not in result.response["counts"]["pure_plans"]
    bad_root = replace(candidate.root, owner="R")
    bad = replace(candidate, root=bad_root)
    assert_no_payload(
        solve(bad, hero_policy_for_check(), attestation=None),
        INVALID_INPUT,
    )


def test_a2_seat_button_action_order_and_private_observation_split_identity():
    base = success(check_scenario(), hero_policy_for_check())
    seat = success(
        check_scenario(seat_order=("O1", "H", "O2")),
        hero_policy_for_check(),
    )
    button = success(
        check_scenario(button_player_id="O1"),
        hero_policy_for_check(),
    )
    ordered = success(
        check_scenario(
            order=("H", "O1", "O2"),
            suffix="base",
        ),
        hero_policy_for_check(),
    )
    private = success(
        check_scenario(
            initial_observation=observation(
                "base",
                private={"H": "hero-base", "O1": "o1-edited", "O2": "o2-base"},
            )
        ),
        hero_policy_for_check(),
    )
    identities = [
        item.scenario_evaluation["identities"]["tree_structure"]
        for item in (base, seat, button, ordered, private)
    ]
    assert len(set(identities)) == len(identities)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("initial_pot", 30.0),
        ("initial_pot", True),
        ("initial_pot", "030"),
        ("rake_rate", "2/4"),
        ("rake_rate", "0/3"),
        ("rake_rate", "+1"),
        ("rake_rate", "1.0"),
        ("rake_rate", " 0"),
    ],
)
def test_a3_only_canonical_rational_strings_are_accepted(field, value):
    candidate = replace(check_scenario(), **{field: value})
    assert_no_payload(
        solve(candidate, hero_policy_for_check(), attestation=None),
        INVALID_INPUT,
    )


def test_a4_initial_pot_must_equal_exact_contribution_sum():
    bad = replace(check_scenario(), initial_pot="31")
    assert_no_payload(
        solve(bad, hero_policy_for_check(), attestation=None),
        INVALID_INPUT,
    )


@pytest.mark.parametrize(
    ("kind", "target", "status"),
    [
        ("check", "20", INVALID_INPUT),
        ("bet", None, INVALID_INPUT),
        ("call", "20", INVALID_INPUT),
        ("fold", None, INVALID_INPUT),
        ("bet", "10", INVALID_INPUT),
        ("raise", "20", INVALID_INPUT),
        ("bet", "100", UNSUPPORTED_MODEL),
    ],
)
def test_a5_action_state_validation_rejects_illegal_edges(kind, target, status):
    candidate = check_scenario()
    root = candidate.root
    action = replace(
        root.actions[0],
        kind=kind,
        target_total_contribution=target,
    )
    bad = replace(candidate, root=replace(root, actions=(action,)))
    assert_no_payload(
        solve(bad, hero_policy_for_check(), attestation=None),
        status,
    )


def test_a5_one_raise_is_legal_and_reraise_is_unsupported():
    legal = valid_raise_scenario()
    policy = ExactBehaviorPolicy(
        {"H_open": {"bet": "1"}, "H_call_raise": {"call": "1"}}
    )
    success(legal, policy)
    assert_no_payload(
        solve(valid_raise_scenario(reraise=True), policy, attestation=None),
        UNSUPPORTED_MODEL,
    )


def test_a5_multiple_bet_sizes_at_one_node_are_rejected():
    candidate = check_scenario()
    root = candidate.root
    actions = (
        RiverAction("bet-small", "bet", "20", root.actions[0].child),
        RiverAction("bet-large", "bet", "30", root.actions[0].child),
    )
    bad = replace(candidate, root=replace(root, actions=actions))
    assert_no_payload(
        solve(bad, hero_policy_for_check(), attestation=None),
        INVALID_INPUT,
    )


def test_a6_fold_uncalled_return_and_zero_rake_hand_calculation():
    result = success(multiple_response_scenario(), multiple_hero_policy())
    record = next(
        item
        for item in result.scenario_evaluation["terminal_records"]
        if item["node_id"] == "terminal-both-fold"
    )
    assert record["gross_contribution"] == {"H": "20", "O1": "10", "O2": "10"}
    assert record["uncalled_return"] == {"H": "10", "O1": "0", "O2": "0"}
    assert record["pot_before_rake"] == "30"
    assert record["rake_amount"] == "0"
    assert record["utility"] == {"H": "20", "O1": "-10", "O2": "-10", "R": "0"}


def test_a6_contract_example_gross_15_10_4_returns_five_to_hero():
    terminal = RiverTerminalNode("fold-15-10-4", "fold")
    o2 = RiverDecisionNode(
        "o2-fold-15",
        "O2",
        "O2_fold_15",
        (RiverAction("fold", "fold", None, terminal),),
    )
    o1 = RiverDecisionNode(
        "o1-fold-15",
        "O1",
        "O1_fold_15",
        (RiverAction("fold", "fold", None, o2),),
    )
    root = RiverDecisionNode(
        "hero-bet-15",
        "H",
        "H_bet_15",
        (RiverAction("bet", "bet", "15", o1),),
    )
    candidate = ThreePlayerRiverRakeScenario(
        root,
        "H",
        ("H", "O1", "O2"),
        ("H", "O1", "O2"),
        observation("fold-15"),
        "24",
        {"H": "10", "O1": "10", "O2": "4"},
        {"H": "100", "O1": "100", "O2": "100"},
        "1/10",
        "2",
    )
    result = success(
        candidate,
        ExactBehaviorPolicy({"H_bet_15": {"bet": "1"}}),
    )
    record = result.scenario_evaluation["terminal_records"][0]
    assert record["gross_contribution"] == {"H": "15", "O1": "10", "O2": "4"}
    assert record["uncalled_return"] == {"H": "5", "O1": "0", "O2": "0"}
    assert record["pot_before_rake"] == record["pot_after_rake"] == "24"
    assert record["rake_amount"] == "0"
    assert record["utility"] == {"H": "14", "O1": "-10", "O2": "-4", "R": "0"}


def test_a7_zero_rake_showdown_hand_calculation():
    result = success(check_scenario(), hero_policy_for_check())
    record = result.scenario_evaluation["terminal_records"][0]
    assert record["pot_before_rake"] == "30"
    assert record["pot_after_rake"] == "30"
    assert record["utility"] == {"H": "20", "O1": "-10", "O2": "-10", "R": "0"}


def test_a7_root_chance_is_evaluated_with_exact_probabilities():
    chance = RiverChanceNode(
        "exact-root-chance",
        (
            RiverChanceOutcome(
                "hero-wins",
                "1/3",
                observation("chance-hero"),
                check_tree(suffix="chance-hero"),
            ),
            RiverChanceOutcome(
                "o1-wins",
                "2/3",
                observation("chance-o1"),
                check_tree(
                    suffix="chance-o1",
                    shares=(AwardShare("O1", "1"),),
                ),
            ),
        ),
    )
    candidate = ThreePlayerRiverRakeScenario(
        chance,
        "H",
        ("H", "O1", "O2"),
        ("O1", "O2", "H"),
        None,
        "30",
        {"H": "10", "O1": "10", "O2": "10"},
        {"H": "100", "O1": "100", "O2": "100"},
        "0",
    )
    policy = ExactBehaviorPolicy(
        {
            "H_root_chance-hero": {"check": "1"},
            "H_root_chance-o1": {"check": "1"},
        }
    )
    result = success(candidate, policy)
    assert result.payoff_table["rows"][0]["utility"] == {
        "H": "0",
        "O1": "10",
        "O2": "-10",
        "R": "0",
    }
    assert result.scenario_evaluation["counts"]["chance_outcomes"] == 2


@pytest.mark.parametrize(
    ("cap", "rake", "hero"),
    [(None, "3", "17"), ("2", "2", "18"), ("3", "3", "17")],
)
def test_a8_positive_rake_below_and_at_cap(cap, rake, hero):
    candidate = check_scenario(rake_rate="1/10", rake_cap=cap)
    result = success(candidate, hero_policy_for_check())
    record = result.scenario_evaluation["terminal_records"][0]
    assert record["rake_amount"] == rake
    assert record["utility"] == {
        "H": hero,
        "O1": "-10",
        "O2": "-10",
        "R": rake,
    }


@pytest.mark.parametrize(
    ("shares", "expected"),
    [
        (
            (AwardShare("H", "1/2"), AwardShare("O1", "1/2")),
            {"H": "4", "O1": "4", "O2": "-10", "R": "2"},
        ),
        (
            (
                AwardShare("H", "1/3"),
                AwardShare("O1", "1/3"),
                AwardShare("O2", "1/3"),
            ),
            {"H": "-2/3", "O1": "-2/3", "O2": "-2/3", "R": "2"},
        ),
    ],
)
def test_a9_exact_two_and_three_way_splits_conserve(shares, expected):
    candidate = check_scenario(shares=shares, rake_rate="1/10", rake_cap="2")
    result = success(candidate, hero_policy_for_check())
    record = result.scenario_evaluation["terminal_records"][0]
    assert record["utility"] == expected
    assert sum(Fraction(value) for value in expected.values()) == 0


def test_a10_side_pot_all_in_and_odd_chip_models_fail_closed():
    side_pot = check_scenario(
        initial_pot="30",
        initial_contribution={"H": "15", "O1": "10", "O2": "5"},
    )
    all_in = replace(
        check_scenario(),
        max_total_contribution={"H": "10", "O1": "100", "O2": "100"},
    )
    odd_chip = replace(
        check_scenario(),
        rounding_unit="1",
        rounding_rule="floor",
    )
    for candidate in (side_pot, all_in, odd_chip):
        assert_no_payload(
            solve(candidate, hero_policy_for_check(), attestation=None),
            UNSUPPORTED_MODEL,
        )


def test_a11_owner_and_information_set_action_signatures_must_match():
    candidate = check_scenario()
    o1 = candidate.root
    o2 = o1.actions[0].child
    shared = "shared-information-set"
    bad_owner = replace(
        candidate,
        root=replace(
            o1,
            information_set_id=shared,
            actions=(
                replace(
                    o1.actions[0],
                    child=replace(o2, information_set_id=shared),
                ),
            ),
        ),
    )
    assert_no_payload(
        solve(bad_owner, hero_policy_for_check(), attestation=None),
        INVALID_INPUT,
    )


def two_outcome_shared_infoset(
    *,
    changed_private: bool = False,
    changed_action: bool = False,
) -> ThreePlayerRiverRakeScenario:
    branches = []
    for index in range(2):
        suffix = f"chance-{index}"
        branch = check_tree(suffix=suffix)
        branch = replace(
            branch,
            information_set_id="O1_shared_chance",
            actions=(
                replace(
                    branch.actions[0],
                    action_id="check-edited"
                    if changed_action and index == 1
                    else "check",
                ),
            ),
        )
        private = {
            "H": "hero-shared",
            "O1": "o1-edited" if changed_private and index == 1 else "o1-shared",
            "O2": "o2-shared",
        }
        branches.append(
            RiverChanceOutcome(
                f"outcome-{index}",
                "1/2",
                RiverObservation("public-shared", private),
                branch,
            )
        )
    return ThreePlayerRiverRakeScenario(
        RiverChanceNode("root-chance", tuple(branches)),
        "H",
        ("H", "O1", "O2"),
        ("O1", "O2", "H"),
        None,
        "30",
        {"H": "10", "O1": "10", "O2": "10"},
        {"H": "100", "O1": "100", "O2": "100"},
        "0",
    )


def own_action_recall_mismatch_scenario() -> ThreePlayerRiverRakeScenario:
    def shared_fold(prefix: str, o1_gross: str) -> RiverDecisionNode:
        terminal = RiverTerminalNode(
            f"{prefix}-terminal",
            "showdown",
            (AwardShare("H", "1"),),
        )
        shared = RiverDecisionNode(
            f"{prefix}-o1-shared",
            "O1",
            "O1_shared_response",
            (RiverAction("fold", "fold", None, terminal),),
        )
        hero = RiverDecisionNode(
            f"{prefix}-hero-call",
            "H",
            f"H_{prefix}_call",
            (RiverAction("call", "call", "30", shared),),
        )
        kind = "bet" if o1_gross == "10" else "raise"
        o2 = RiverDecisionNode(
            f"{prefix}-o2-aggress",
            "O2",
            f"O2_{prefix}_aggress",
            (RiverAction(kind, kind, "30", hero),),
        )
        return o2

    check_branch = shared_fold("checked", "10")
    bet_branch = shared_fold("bet", "20")
    root = RiverDecisionNode(
        "o1-recall-root",
        "O1",
        "O1_recall_root",
        (
            RiverAction("check", "check", None, check_branch),
            RiverAction("bet", "bet", "20", bet_branch),
        ),
    )
    return ThreePlayerRiverRakeScenario(
        root,
        "H",
        ("H", "O1", "O2"),
        ("O1", "O2", "H"),
        observation("recall"),
        "30",
        {"H": "10", "O1": "10", "O2": "10"},
        {"H": "100", "O1": "100", "O2": "100"},
        "0",
    )


def test_a11_same_infoset_requires_same_ordered_legal_action_signature():
    candidate = two_outcome_shared_infoset(changed_action=True)
    assert_no_payload(
        solve(candidate, ExactBehaviorPolicy({}), attestation=None),
        INVALID_INPUT,
    )


def test_a12_private_observation_and_own_action_recall_mismatch_are_rejected():
    private = two_outcome_shared_infoset(changed_private=True)
    recall = own_action_recall_mismatch_scenario()
    for candidate in (private, recall):
        assert_no_payload(
            solve(candidate, ExactBehaviorPolicy({}), attestation=None),
            INVALID_INPUT,
        )


@pytest.mark.parametrize("confirmed_field", ["o1_confirmed", "o2_confirmed"])
def test_a13_missing_or_false_human_recall_evidence_is_unsupported(confirmed_field):
    candidate = check_scenario()
    policy = hero_policy_for_check()
    assert_no_payload(
        solve(candidate, policy, attestation=None),
        UNSUPPORTED_MODEL,
    )
    evidence = attest(candidate, **{confirmed_field: False})
    assert_no_payload(
        solve(candidate, policy, attestation=evidence),
        UNSUPPORTED_MODEL,
    )


def test_a14_tree_bound_recall_evidence_detects_staleness():
    candidate = check_scenario()
    evidence = attest(candidate)
    stale = replace(evidence, tree_content_identity="0" * 64)
    assert_no_payload(
        solve(candidate, hero_policy_for_check(), attestation=stale),
        STALE_INPUT,
    )


@pytest.mark.parametrize(
    "policy",
    [
        ExactBehaviorPolicy({}),
        ExactBehaviorPolicy({"H_root_base": {"check": "1"}, "unknown": {}}),
        ExactBehaviorPolicy({"H_root_base": {}}),
        ExactBehaviorPolicy({"H_root_base": {"check": "1/2"}}),
    ],
)
def test_a15_fixed_hero_policy_must_be_complete_exact_and_unmodified(policy):
    assert_no_payload(solve(check_scenario(), policy), INVALID_INPUT)


def test_a16_optional_initial_profiles_require_both_complete_off_path_maps():
    candidate = multiple_response_scenario()
    policy = multiple_hero_policy()
    one_sided = OpponentInitialProfile(
        o1_probabilities=multiple_initial_profile("fold").o1_probabilities,
        o2_probabilities=None,
    )
    missing = replace(
        multiple_initial_profile("fold"),
        o2_probabilities={
            "O2_after_o1_call": {"call": "0", "fold": "1"},
        },
    )
    unknown = replace(
        multiple_initial_profile("fold"),
        o1_probabilities={
            "O1_facing_hero_bet": {"fold": "1", "call": "0"},
            "unknown": {"fold": "1"},
        },
    )
    for profile in (one_sided, missing, unknown):
        assert_no_payload(
            solve(candidate, policy, initial_profile=profile),
            INVALID_INPUT,
        )


def test_a17_profile_only_edit_preserves_game_identity_and_changes_run_identity():
    candidate = multiple_response_scenario()
    first = success(
        candidate,
        multiple_hero_policy(),
        initial_profile=multiple_initial_profile("fold"),
    )
    second = success(
        candidate,
        multiple_hero_policy(),
        initial_profile=multiple_initial_profile("call"),
    )
    assert (
        first.response["response_game_identity"]
        == second.response["response_game_identity"]
    )
    assert (
        first.response["response_run_identity"]
        != second.response["response_run_identity"]
    )
    assert first.response["support_cells"] == second.response["support_cells"]


def two_action_hero_scenario() -> ThreePlayerRiverRakeScenario:
    showdown = RiverTerminalNode(
        "hero-check-terminal",
        "showdown",
        (AwardShare("H", "1"),),
    )
    o2_check = RiverDecisionNode(
        "o2-check-after-hero",
        "O2",
        "O2_after_hero_check",
        (RiverAction("check", "check", None, showdown),),
    )
    o1_check = RiverDecisionNode(
        "o1-check-after-hero",
        "O1",
        "O1_after_hero_check",
        (RiverAction("check", "check", None, o2_check),),
    )
    fold_terminal = RiverTerminalNode("hero-bet-both-fold", "fold")
    o2_fold = RiverDecisionNode(
        "o2-fold-after-hero",
        "O2",
        "O2_after_hero_bet",
        (RiverAction("fold", "fold", None, fold_terminal),),
    )
    o1_fold = RiverDecisionNode(
        "o1-fold-after-hero",
        "O1",
        "O1_after_hero_bet",
        (RiverAction("fold", "fold", None, o2_fold),),
    )
    root = RiverDecisionNode(
        "hero-two-actions",
        "H",
        "H_two_actions",
        (
            RiverAction("check", "check", None, o1_check),
            RiverAction("bet", "bet", "20", o1_fold),
        ),
    )
    return ThreePlayerRiverRakeScenario(
        root,
        "H",
        ("H", "O1", "O2"),
        ("H", "O1", "O2"),
        observation("hero-two"),
        "30",
        {"H": "10", "O1": "10", "O2": "10"},
        {"H": "100", "O1": "100", "O2": "100"},
        "0",
    )


def test_a18_hero_rake_and_tree_edits_change_payoff_or_response_game_identity():
    hero_scenario = two_action_hero_scenario()
    check_policy = ExactBehaviorPolicy(
        {"H_two_actions": {"check": "1", "bet": "0"}}
    )
    bet_policy = ExactBehaviorPolicy(
        {"H_two_actions": {"check": "0", "bet": "1"}}
    )
    hero_first = success(hero_scenario, check_policy)
    hero_second = success(hero_scenario, bet_policy)
    assert (
        hero_first.response["response_game_identity"]
        != hero_second.response["response_game_identity"]
    )
    zero = success(check_scenario(), hero_policy_for_check())
    raked = success(
        check_scenario(rake_rate="1/10", rake_cap="2"),
        hero_policy_for_check(),
    )
    private = success(
        check_scenario(
            initial_observation=observation(
                "base",
                private={"H": "H2", "O1": "o1-base", "O2": "o2-base"},
            )
        ),
        hero_policy_for_check(),
    )
    assert (
        zero.response["response_game_identity"]
        != raked.response["response_game_identity"]
    )
    assert (
        zero.response["response_game_identity"]
        != private.response["response_game_identity"]
    )


def test_a19_pure_plan_cap_is_checked_before_plan_allocation(monkeypatch):
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("plan allocation reached")

    monkeypatch.setattr(module, "_materialize_plans", forbidden)
    result = solve(
        multiple_response_scenario(),
        multiple_hero_policy(),
        limits=replace(RiverRakeLimits(), max_pure_plans_o1=1),
    )
    assert_no_payload(result, CAP_EXCEEDED)
    assert called is False


@pytest.mark.parametrize(
    "limits",
    [
        replace(RiverRakeLimits(), max_joint_pure_profiles=7),
        replace(RiverRakeLimits(), max_terminal_evaluations=31),
        replace(RiverRakeLimits(), max_payoff_rows=7),
        replace(RiverRakeLimits(), max_output_records=20),
        replace(RiverRakeLimits(), max_output_bytes=1),
    ],
)
def test_a20_table_shape_caps_precede_plan_and_table_allocation(
    monkeypatch, limits
):
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("plan allocation reached")

    monkeypatch.setattr(module, "_materialize_plans", forbidden)
    result = solve(
        multiple_response_scenario(),
        multiple_hero_policy(),
        limits=limits,
    )
    assert_no_payload(result, CAP_EXCEEDED)
    assert called is False


def test_a21_full_rectangular_o1_outer_o2_inner_table_is_complete():
    result = success(multiple_response_scenario(), multiple_hero_policy())
    table = result.payoff_table
    assert table["rectangular_complete"] is True
    o1_ids = [item["plan_id"] for item in table["o1_plans"]]
    o2_ids = [item["plan_id"] for item in table["o2_plans"]]
    expected = [(o1, o2) for o1 in o1_ids for o2 in o2_ids]
    actual = [
        (row["o1_plan_id"], row["o2_plan_id"]) for row in table["rows"]
    ]
    assert actual == expected
    assert len(actual) == len(set(actual)) == len(o1_ids) * len(o2_ids)


def test_a21_semantic_chance_and_award_sequences_must_be_ordered_tuples():
    terminal = RiverTerminalNode(
        "list-awards",
        "showdown",
        [AwardShare("H", "1")],
    )
    root = check_tree()
    bad_awards = replace(
        check_scenario(),
        root=replace(
            root,
            actions=(
                replace(
                    root.actions[0],
                    child=replace(
                        root.actions[0].child,
                        actions=(
                            replace(
                                root.actions[0].child.actions[0],
                                child=replace(
                                    root.actions[0].child.actions[0].child,
                                    actions=(
                                        replace(
                                            root.actions[0]
                                            .child.actions[0]
                                            .child.actions[0],
                                            child=terminal,
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    assert_no_payload(
        solve(bad_awards, hero_policy_for_check(), attestation=None),
        INVALID_INPUT,
    )

    chance = RiverChanceNode(
        "list-chance",
        [
            RiverChanceOutcome(
                "only",
                "1",
                observation("list-chance"),
                check_tree(suffix="list-chance"),
            )
        ],
    )
    bad_chance = ThreePlayerRiverRakeScenario(
        chance,
        "H",
        ("H", "O1", "O2"),
        ("O1", "O2", "H"),
        None,
        "30",
        {"H": "10", "O1": "10", "O2": "10"},
        {"H": "100", "O1": "100", "O2": "100"},
        "0",
    )
    assert_no_payload(
        solve(bad_chance, ExactBehaviorPolicy({}), attestation=None),
        INVALID_INPUT,
    )


def test_a21_duplicate_generated_plan_ids_fail_closed(monkeypatch):
    original = module._materialize_plans

    def duplicate(player, info_sets, expected_count):
        ids, plans = original(player, info_sets, expected_count)
        if player == "O1" and len(ids) > 1:
            ids = (ids[0],) * len(ids)
        return ids, plans

    monkeypatch.setattr(module, "_materialize_plans", duplicate)
    result = solve(multiple_response_scenario(), multiple_hero_policy())
    assert_no_payload(result, INVALID_INPUT)


def test_a22_independent_enumerator_agrees_all_rows_and_mismatch_fails(monkeypatch):
    baseline = success(multiple_response_scenario(), multiple_hero_policy())
    evidence = baseline.scenario_evaluation["independent_evaluation"]
    assert evidence["all_rows_exactly_equal"] is True
    assert evidence["verified_row_count"] == 8
    original = module._independent_path_evaluate

    def mismatch(*args, **kwargs):
        utility, count = original(*args, **kwargs)
        return replace(utility, H=utility.H + 1), count

    monkeypatch.setattr(module, "_independent_path_evaluate", mismatch)
    assert_no_payload(
        solve(multiple_response_scenario(), multiple_hero_policy()),
        NUMERIC_FAILURE,
    )


def test_a23_terminal_rows_and_m30_mixture_evidence_conserve_exactly():
    result = success(multiple_response_scenario(), multiple_hero_policy())
    for terminal in result.scenario_evaluation["terminal_records"]:
        assert terminal["conservation"] == "0"
        assert sum(Fraction(x) for x in terminal["utility"].values()) == 0
    for row in result.payoff_table["rows"]:
        assert row["conservation"] == "0"
        assert sum(Fraction(x) for x in row["utility"].values()) == 0
    response = result.response
    assert response["independent_verification"]["status"] == "VERIFIED"
    for account in ("H", "O1", "O2", "R"):
        extrema = response["utility_extrema"][account]
        assert "minimum_witnesses" in extrema
        assert "maximum_witnesses" in extrema


def test_a24_multiple_response_fixture_has_nontrivial_hero_interval():
    response = success(
        multiple_response_scenario(), multiple_hero_policy()
    ).response
    assert response["hero_worst"] == "-20"
    assert response["hero_best"] == "20"
    assert response["hero_worst"] != response["hero_best"]
    pure = response["pure_profile_unilateral_stability"]["rows"]
    assert any(row["plans"] == {"O1": "O1:0", "O2": "O2:2"} for row in pure)
    assert any(row["plans"] == {"O1": "O1:1", "O2": "O2:2"} for row in pure)


def test_a25_m30_support_cells_extrema_and_all_witnesses_are_lossless(
    monkeypatch,
):
    captured = {}
    original = module._m30.solve_three_player_response

    def recording(game, **kwargs):
        output = original(game, **kwargs)
        captured["response"] = output.response
        return output

    monkeypatch.setattr(module._m30, "solve_three_player_response", recording)
    result = success(multiple_response_scenario(), multiple_hero_policy())
    assert result.response == captured["response"]
    for key in (
        "support_cells",
        "hero_worst",
        "hero_best",
        "hero_worst_witnesses",
        "utility_extrema",
        "independent_verification",
    ):
        assert key in result.response


def test_a26_pure_subset_and_coalition_stress_remain_separate_diagnostics():
    response = success(
        multiple_response_scenario(), multiple_hero_policy()
    ).response
    pure = response["pure_profile_unilateral_stability"]
    stress = response["hero_min_joint_plan_stress"]
    assert pure["coverage"] == "complete"
    assert "best_response_value-current_value" in pure["residual_semantics"]
    assert stress["coalition_equilibrium_claim"] is False
    assert stress["primary_response_status_influence"] is False
    assert stress["opponent_individual_rationality_not_required"] is True
    assert response["coverage"] == "complete"


def test_a27_every_failure_status_has_three_null_payloads(monkeypatch):
    invalid = solve(
        replace(check_scenario(), initial_pot="31"),
        hero_policy_for_check(),
        attestation=None,
    )
    unsupported = solve(
        replace(check_scenario(), rounding_rule="floor", rounding_unit="1"),
        hero_policy_for_check(),
        attestation=None,
    )
    stale = solve(
        check_scenario(),
        hero_policy_for_check(),
        pins=RiverRakeIdentityPins(scenario_identity="0" * 64),
    )
    capped = solve(
        check_scenario(),
        hero_policy_for_check(),
        limits=replace(RiverRakeLimits(), max_nodes=3),
    )
    monkeypatch.setattr(
        module._m30,
        "solve_three_player_response",
        lambda *_args, **_kwargs: module._m30.ExactResponseResult(
            status=module._m30.INTERNAL_FAILURE,
            response=None,
            error=module._m30.ExactResponseError("solver", "forced"),
            partial_response=False,
        ),
    )
    downstream = solve(check_scenario(), hero_policy_for_check())
    for result, status in (
        (invalid, INVALID_INPUT),
        (unsupported, UNSUPPORTED_MODEL),
        (stale, STALE_INPUT),
        (capped, CAP_EXCEEDED),
        (downstream, M30_RESPONSE_FAILURE),
    ):
        assert_no_payload(result, status)


def test_a28_incomplete_probability_is_not_normalized_and_caps_do_not_truncate():
    scenario = two_action_hero_scenario()
    incomplete = ExactBehaviorPolicy(
        {"H_two_actions": {"check": "1/3", "bet": "1/3"}}
    )
    assert_no_payload(solve(scenario, incomplete), INVALID_INPUT)
    capped = solve(
        multiple_response_scenario(),
        multiple_hero_policy(),
        limits=replace(RiverRakeLimits(), max_payoff_rows=7),
    )
    assert_no_payload(capped, CAP_EXCEEDED)


def test_a29_same_process_canonical_bytes_are_deterministic():
    scenario = multiple_response_scenario()
    evidence = attest(scenario)
    first = solve(
        scenario,
        multiple_hero_policy(),
        attestation=evidence,
    )
    second = solve(
        scenario,
        multiple_hero_policy(),
        attestation=evidence,
    )
    assert exact_scenario_response_json(first) == exact_scenario_response_json(
        second
    )


def test_a30_unordered_mapping_permutation_is_stable_ordered_action_edit_is_not():
    scenario = check_scenario()
    permuted = replace(
        scenario,
        initial_contribution=OrderedDict(
            (item, scenario.initial_contribution[item])
            for item in ("O2", "H", "O1")
        ),
        max_total_contribution=OrderedDict(
            (item, scenario.max_total_contribution[item])
            for item in ("O1", "O2", "H")
        ),
        initial_observation=RiverObservation(
            scenario.initial_observation.public_observation_id,
            OrderedDict(
                (item, scenario.initial_observation.private_observation_id_by_player[item])
                for item in ("O2", "O1", "H")
            ),
        ),
    )
    first = success(scenario, hero_policy_for_check())
    second = success(permuted, hero_policy_for_check())
    assert exact_scenario_response_json(first) == exact_scenario_response_json(
        second
    )

    multiple = multiple_response_scenario()
    hero_root = multiple.root
    o1 = hero_root.actions[0].child
    reordered_o1 = replace(o1, actions=tuple(reversed(o1.actions)))
    reordered = replace(
        multiple,
        root=replace(
            hero_root,
            actions=(replace(hero_root.actions[0], child=reordered_o1),),
        ),
    )
    original = success(multiple, multiple_hero_policy())
    edited = success(reordered, multiple_hero_policy())
    assert (
        original.scenario_evaluation["identities"]["tree_structure"]
        != edited.scenario_evaluation["identities"]["tree_structure"]
    )


def test_a31_m31_caps_match_contract_and_lower_m30_caps_are_effective(
    monkeypatch,
):
    defaults = RiverRakeLimits()
    assert (
        defaults.max_nodes,
        defaults.max_terminals,
        defaults.max_fixed_hero_info_sets,
        defaults.max_info_sets_per_opponent,
        defaults.max_opponent_info_sets_total,
        defaults.max_actions_per_info_set,
        defaults.max_chance_outcomes,
        defaults.max_private_observation_records,
        defaults.max_terminal_contribution_records,
        defaults.max_pure_plans_o1,
        defaults.max_pure_plans_o2,
        defaults.max_joint_pure_profiles,
        defaults.max_terminal_evaluations,
        defaults.max_payoff_rows,
        defaults.max_rational_numerator_bits,
        defaults.max_rational_denominator_bits,
        defaults.max_identity_records,
        defaults.max_output_records,
        defaults.max_output_bytes,
    ) == (
        200,
        128,
        12,
        12,
        24,
        4,
        16,
        64,
        384,
        4,
        4,
        16,
        2048,
        16,
        256,
        256,
        10000,
        50000,
        4000000,
    )
    hard = module._HARD_LIMITS
    assert (
        hard["max_nodes"],
        hard["max_terminals"],
        hard["max_fixed_hero_info_sets"],
        hard["max_info_sets_per_opponent"],
        hard["max_opponent_info_sets_total"],
        hard["max_actions_per_info_set"],
        hard["max_chance_outcomes"],
        hard["max_private_observation_records"],
        hard["max_terminal_contribution_records"],
        hard["max_pure_plans_o1"],
        hard["max_pure_plans_o2"],
        hard["max_joint_pure_profiles"],
        hard["max_terminal_evaluations"],
        hard["max_payoff_rows"],
        hard["max_rational_numerator_bits"],
        hard["max_rational_denominator_bits"],
        hard["max_identity_records"],
        hard["max_output_records"],
        hard["max_output_bytes"],
    ) == (
        500,
        256,
        16,
        16,
        32,
        4,
        32,
        256,
        768,
        6,
        6,
        36,
        9216,
        36,
        1024,
        1024,
        100000,
        500000,
        32000000,
    )

    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("plan allocation reached")

    monkeypatch.setattr(module, "_materialize_plans", forbidden)
    result = solve(
        multiple_response_scenario(),
        multiple_hero_policy(),
        m30_limits=replace(ExactResponseLimits(), max_pure_plans_o2=3),
    )
    assert_no_payload(result, CAP_EXCEEDED)
    assert called is False


def test_a31_structural_numeric_identity_and_byte_caps_are_enforced():
    base = check_scenario()
    multiple = multiple_response_scenario()
    raised = valid_raise_scenario()
    chance = ThreePlayerRiverRakeScenario(
        RiverChanceNode(
            "cap-chance",
            (
                RiverChanceOutcome(
                    "cap-a",
                    "1/2",
                    observation("cap-a"),
                    check_tree(suffix="cap-a"),
                ),
                RiverChanceOutcome(
                    "cap-b",
                    "1/2",
                    observation("cap-b"),
                    check_tree(suffix="cap-b"),
                ),
            ),
        ),
        "H",
        ("H", "O1", "O2"),
        ("O1", "O2", "H"),
        None,
        "30",
        {"H": "10", "O1": "10", "O2": "10"},
        {"H": "100", "O1": "100", "O2": "100"},
        "0",
    )
    cases = (
        (
            base,
            hero_policy_for_check(),
            replace(RiverRakeLimits(), max_nodes=3),
        ),
        (
            multiple,
            multiple_hero_policy(),
            replace(RiverRakeLimits(), max_terminals=3),
        ),
        (
            raised,
            ExactBehaviorPolicy(
                {"H_open": {"bet": "1"}, "H_call_raise": {"call": "1"}}
            ),
            replace(RiverRakeLimits(), max_fixed_hero_info_sets=1),
        ),
        (
            multiple,
            multiple_hero_policy(),
            replace(RiverRakeLimits(), max_info_sets_per_opponent=1),
        ),
        (
            multiple,
            multiple_hero_policy(),
            replace(RiverRakeLimits(), max_opponent_info_sets_total=2),
        ),
        (
            multiple,
            multiple_hero_policy(),
            replace(RiverRakeLimits(), max_actions_per_info_set=1),
        ),
        (
            chance,
            ExactBehaviorPolicy(
                {
                    "H_root_cap-a": {"check": "1"},
                    "H_root_cap-b": {"check": "1"},
                }
            ),
            replace(RiverRakeLimits(), max_chance_outcomes=1),
        ),
        (
            base,
            hero_policy_for_check(),
            replace(RiverRakeLimits(), max_private_observation_records=2),
        ),
        (
            base,
            hero_policy_for_check(),
            replace(RiverRakeLimits(), max_terminal_contribution_records=2),
        ),
        (
            base,
            hero_policy_for_check(),
            replace(RiverRakeLimits(), max_rational_numerator_bits=3),
        ),
        (
            replace(base, rake_rate="1/8"),
            hero_policy_for_check(),
            replace(RiverRakeLimits(), max_rational_denominator_bits=3),
        ),
        (
            base,
            hero_policy_for_check(),
            replace(RiverRakeLimits(), max_identity_records=1),
        ),
        (
            base,
            hero_policy_for_check(),
            replace(RiverRakeLimits(), max_output_bytes=1),
        ),
    )
    for scenario, policy, limits in cases:
        assert_no_payload(
            solve(scenario, policy, limits=limits),
            CAP_EXCEEDED,
        )


def test_a31_every_m31_hard_ceiling_is_immutable():
    for field, hard in module._HARD_LIMITS.items():
        invalid = replace(RiverRakeLimits(), **{field: hard + 1})
        assert_no_payload(
            solve(check_scenario(), hero_policy_for_check(), limits=invalid),
            INVALID_INPUT,
        )


def test_m31_t3_caps_reject_before_hash_record_allocation_and_m30_entry(
    monkeypatch,
):
    scenario = check_scenario()
    policy = hero_policy_for_check()
    evidence = attest(scenario)
    baseline = success(scenario, policy)
    assert len(module._canonical_json_bytes(baseline.to_dict())) == 14_391
    original_identity = module._identity
    original_materialize = module._materialize_plans
    original_m30 = module._m30.solve_three_player_response
    calls = {"identity": 0, "materialize": 0, "m30": 0}

    def counted_identity(value):
        calls["identity"] += 1
        return original_identity(value)

    monkeypatch.setattr(module, "_identity", counted_identity)
    identity_capped = evaluate_three_player_river_rake(
        scenario,
        policy,
        attestation=evidence,
        limits=replace(RiverRakeLimits(), max_identity_records=1),
    )
    assert_no_payload(identity_capped, CAP_EXCEEDED)
    assert calls == {"identity": 0, "materialize": 0, "m30": 0}

    monkeypatch.setattr(module, "_identity", original_identity)

    def counted_materialize(*args, **kwargs):
        calls["materialize"] += 1
        return original_materialize(*args, **kwargs)

    def counted_m30(*args, **kwargs):
        calls["m30"] += 1
        return original_m30(*args, **kwargs)

    monkeypatch.setattr(module, "_materialize_plans", counted_materialize)
    monkeypatch.setattr(module._m30, "solve_three_player_response", counted_m30)
    record_capped = evaluate_three_player_river_rake(
        scenario,
        policy,
        attestation=evidence,
        limits=replace(RiverRakeLimits(), max_output_records=288),
    )
    assert_no_payload(record_capped, CAP_EXCEEDED)
    assert calls == {"identity": 0, "materialize": 0, "m30": 0}

    byte_capped = evaluate_three_player_river_rake(
        scenario,
        policy,
        attestation=evidence,
        limits=replace(RiverRakeLimits(), max_output_bytes=14_390),
    )
    assert_no_payload(byte_capped, CAP_EXCEEDED)
    assert calls == {"identity": 0, "materialize": 0, "m30": 0}


def _corrupt_m30_response(response: dict, fault: str) -> None:
    if fault == "status":
        response["status"] = "BROKEN"
    elif fault == "contract_version":
        response["contract_version"] = "broken"
    elif fault == "algorithm_version":
        response["algorithm_version"] = "broken"
    elif fault == "verifier_version":
        response["verifier_version"] = "broken"
    elif fault == "response_game_identity":
        response["response_game_identity"] = "0" * 64
    elif fault == "response_run_identity":
        response["response_run_identity"] = "0" * 64
    elif fault == "content_identity":
        response["content_identities"]["config"] = "0" * 64
    elif fault == "effective_limit":
        response["limits"]["max_output_records"] -= 1
    elif fault == "ordering_descriptor":
        response["ordering"]["support_cells"] = "broken"
    elif fault == "support_pair_count":
        response["counts"]["support_pairs_visited"] -= 1
    elif fault == "support_pair_audit":
        response["support_pair_audit"].pop()
    elif fault == "support_cell_kind":
        response["support_cells"][0]["kind"] = "broken"
    elif fault == "o2_extremum":
        response["utility_extrema"]["O2"]["maximum"] = "999"
    elif fault == "r_witness":
        response["utility_extrema"]["R"]["minimum_witnesses"] = []
    elif fault == "hero_alias":
        response["hero_worst"] = "999"
    elif fault == "independent_status":
        response["independent_verification"]["status"] = "BROKEN"
    elif fault == "pure_subset_coverage":
        response["pure_profile_unilateral_stability"]["coverage"] = "partial"
    elif fault == "coalition_flag":
        response["hero_min_joint_plan_stress"][
            "coalition_equilibrium_claim"
        ] = True
    elif fault == "actual_support_cell_order":
        assert len(response["support_cells"]) == 7
        response["support_cells"].reverse()
    elif fault == "actual_source_pair_order":
        cell = next(
            item
            for item in response["support_cells"]
            if len(item["source_support_pairs"]) > 1
        )
        cell["source_support_pairs"].reverse()
    elif fault == "actual_vertex_order":
        cell = next(
            item
            for item in response["support_cells"]
            if item["o2_mixture_polytope"]["vertex_count"] > 1
        )
        cell["o2_mixture_polytope"]["vertices"].reverse()
    elif fault == "actual_witness_order":
        witnesses = response["utility_extrema"]["H"]["minimum_witnesses"]
        assert len(witnesses) == 2
        witnesses.reverse()
    elif fault == "false_dimension_kind":
        cell = next(
            item
            for item in response["support_cells"]
            if item["o1_mixture_polytope"]["dimension"] == 0
        )
        cell["o1_mixture_polytope"]["dimension"] = 1
        cell["dimension"] += 1
        cell["kind"] = "continuum"
    elif fault == "false_system_count":
        assert response["counts"]["exact_linear_systems_preflight"] == 176
        assert response["counts"]["exact_linear_systems_solved"] == 53
        response["counts"]["exact_linear_systems_solved"] = 0
    else:
        raise AssertionError(f"unknown test fault: {fault}")


@pytest.mark.parametrize(
    "fault",
    (
        "status",
        "contract_version",
        "algorithm_version",
        "verifier_version",
        "response_game_identity",
        "response_run_identity",
        "content_identity",
        "effective_limit",
        "ordering_descriptor",
        "support_pair_count",
        "support_pair_audit",
        "support_cell_kind",
        "o2_extremum",
        "r_witness",
        "hero_alias",
        "independent_status",
        "pure_subset_coverage",
        "coalition_flag",
        "actual_support_cell_order",
        "actual_source_pair_order",
        "actual_vertex_order",
        "actual_witness_order",
        "false_dimension_kind",
        "false_system_count",
    ),
)
def test_m31_t5_rejects_inconsistent_typed_m30_success(
    monkeypatch,
    fault,
):
    strict_faults = {
        "actual_support_cell_order",
        "actual_source_pair_order",
        "actual_vertex_order",
        "actual_witness_order",
        "false_dimension_kind",
        "false_system_count",
    }
    scenario = (
        multiple_response_scenario()
        if fault in strict_faults
        else check_scenario()
    )
    policy = (
        multiple_hero_policy()
        if fault in strict_faults
        else hero_policy_for_check()
    )
    evidence = attest(scenario)
    original = module._m30.solve_three_player_response
    calls = 0

    def corrupted(game, **kwargs):
        nonlocal calls
        calls += 1
        result = original(game, **kwargs)
        response = copy.deepcopy(result.response)
        _corrupt_m30_response(response, fault)
        return replace(result, response=response)

    monkeypatch.setattr(module._m30, "solve_three_player_response", corrupted)
    result = evaluate_three_player_river_rake(
        scenario,
        policy,
        attestation=evidence,
    )
    assert calls == 1
    assert_no_payload(result, M30_RESPONSE_FAILURE)


def test_a32_protected_scope_and_future_work_remain_untouched():
    completed = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    changed = {
        line[3:].replace("\\", "/")
        for line in completed.stdout.splitlines()
        if line
    }
    assert changed <= {
        "src/repeated_poker/three_player_river_rake.py",
        "tests/test_three_player_river_rake.py",
    }
    source = inspect.getsource(module)
    for prohibited_import in (
        "known_board_real_card",
        "three_player_cfr",
        "candidate_selector",
        "repeated_game",
    ):
        assert f"import {prohibited_import}" not in source
