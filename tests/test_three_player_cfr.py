"""Tests for the isolated three-player CFR-style diagnostic contract."""

import copy
import inspect
import math
import re

import pytest
import repeated_poker.three_player_cfr as three_player_module

from repeated_poker.three_player_cfr import (
    BehaviorStrategy,
    CfrConfig,
    CfrDiagnosticResult,
    CfrSafetyLimits,
    DiagnosticContractError,
    CAP_EXCEEDED,
    INVALID_INPUT,
    FixedHeroNode,
    OpponentDecisionNode,
    PerfectRecallAttestation,
    ThreePlayerChanceNode,
    ThreePlayerGameTree,
    ThreePlayerTerminalNode,
    UtilityVector,
    INDETERMINATE_TOLERANCE,
    NOT_REQUESTED,
    ORACLE_MISMATCH,
    ORACLE_UNAVAILABLE_CAP,
    NUMERIC_FAILURE,
    UNSUPPORTED_MODEL,
    compute_unilateral_deviation_gains,
    count_opponent_pure_profiles,
    create_perfect_recall_attestation,
    evaluate_three_player_profile,
    run_three_player_cfr_diagnostic,
    tree_content_identity,
    validate_three_player_tree,
)

TOL = 1e-9


def _u(hero, o1, o2, residual=0.0):
    return UtilityVector(H=hero, O1=o1, O2=o2, R=residual)


def _t(node_id, hero, o1, o2, residual=0.0):
    return ThreePlayerTerminalNode(node_id, _u(hero, o1, o2, residual))


def _empty_hero():
    return BehaviorStrategy({})


def _attest(tree, **overrides):
    values = {
        "verifier": "test-reviewer",
        "verification_date": "2026-07-12",
        "evidence_version": "test-v1",
    }
    values.update(overrides)
    return create_perfect_recall_attestation(tree, **values)


def _normal_form_tree(payoffs):
    """Build a 2x2 simultaneous-move game as an imperfect-information tree."""

    def o2_node(o1_action):
        return OpponentDecisionNode(
            node_id=f"o2_after_{o1_action}",
            owner="opponent_2",
            info_set="O2_root",
            actions=(
                ("L", _t(f"t_{o1_action}_L", *payoffs[(o1_action, "L")])),
                ("R", _t(f"t_{o1_action}_R", *payoffs[(o1_action, "R")])),
            ),
        )

    root = OpponentDecisionNode(
        node_id="o1",
        owner="opponent_1",
        info_set="O1_root",
        actions=(("A", o2_node("A")), ("B", o2_node("B"))),
    )
    return ThreePlayerGameTree(root)


def test_tiny_normal_form_profile_evaluation_matches_hand_calculation():
    tree = _normal_form_tree(
        {
            ("A", "L"): (-3.0, 1.0, 2.0),
            ("A", "R"): (0.0, 0.0, 0.0),
            ("B", "L"): (0.0, 0.0, 0.0),
            ("B", "R"): (-4.0, 2.0, 2.0),
        }
    )
    strategies = {
        "O1": BehaviorStrategy({"O1_root": {"A": 0.25, "B": 0.75}}),
        "O2": BehaviorStrategy({"O2_root": {"L": 0.4, "R": 0.6}}),
    }

    value = evaluate_three_player_profile(tree, _empty_hero(), strategies)

    assert value.H == pytest.approx(-2.1)
    assert value.O1 == pytest.approx(1.0)
    assert value.O2 == pytest.approx(1.1)
    assert value.conservation_residual() == pytest.approx(0.0, abs=TOL)
    assert count_opponent_pure_profiles(tree) == 4


def test_zero_sum_degenerate_normal_form_converges_near_known_mixed_strategy():
    tree = _normal_form_tree(
        {
            ("A", "L"): (0.0, 1.0, -1.0),
            ("A", "R"): (0.0, -1.0, 1.0),
            ("B", "L"): (0.0, -1.0, 1.0),
            ("B", "R"): (0.0, 1.0, -1.0),
        }
    )

    result = run_three_player_cfr_diagnostic(
        tree,
        _empty_hero(),
        config=CfrConfig(iterations=2_000),
        attestation=_attest(tree),
    )

    o1 = result.average_strategy_by_player["O1"]["O1_root"]
    o2 = result.average_strategy_by_player["O2"]["O2_root"]
    assert o1["A"] == pytest.approx(0.5, abs=0.03)
    assert o2["L"] == pytest.approx(0.5, abs=0.03)
    assert result.expected_utility_vector["O1"] == pytest.approx(0.0, abs=0.05)
    assert result.expected_utility_vector["O2"] == pytest.approx(0.0, abs=0.05)
    assert result.deterministic_full_traversal is True

    forbidden = (
        "equilibrium",
        "exploitability",
        "best_response",
        "profitable",
        "solver_grade",
        "real_money",
        "advice",
        "guarantee",
        "optimal",
        "nash",
    )
    assert all(key not in result.to_dict() for key in forbidden)


def test_tiny_extensive_form_with_public_o1_action_matches_manual_value():
    root = OpponentDecisionNode(
        "o1",
        "opponent_1",
        "O1_first",
        (
            (
                "a",
                OpponentDecisionNode(
                    "o2_after_a",
                    "opponent_2",
                    "O2_after_a",
                    (
                        ("x", _t("t_ax", -3.0, 1.0, 2.0)),
                        ("y", _t("t_ay", 0.0, 0.0, 0.0)),
                    ),
                ),
            ),
            (
                "b",
                OpponentDecisionNode(
                    "o2_after_b",
                    "opponent_2",
                    "O2_after_b",
                    (
                        ("x", _t("t_bx", -1.0, 1.0, 0.0)),
                        ("y", _t("t_by", -2.0, 0.0, 2.0)),
                    ),
                ),
            ),
        ),
    )
    tree = ThreePlayerGameTree(root)
    strategies = {
        "O1": BehaviorStrategy({"O1_first": {"a": 0.25, "b": 0.75}}),
        "O2": BehaviorStrategy(
            {
                "O2_after_a": {"x": 0.2, "y": 0.8},
                "O2_after_b": {"x": 0.6, "y": 0.4},
            }
        ),
    }

    value = evaluate_three_player_profile(tree, _empty_hero(), strategies)

    assert value.H == pytest.approx(-1.2)
    assert value.O1 == pytest.approx(0.5)
    assert value.O2 == pytest.approx(0.7)


def test_tiny_extensive_form_with_imperfect_information_reuses_o2_info_set():
    tree = _normal_form_tree(
        {
            ("A", "L"): (-2.0, 1.0, 1.0),
            ("A", "R"): (-1.0, 0.0, 1.0),
            ("B", "L"): (-1.0, 1.0, 0.0),
            ("B", "R"): (0.0, 0.0, 0.0),
        }
    )
    strategies = {
        "O1": BehaviorStrategy({"O1_root": {"A": 0.5, "B": 0.5}}),
        "O2": BehaviorStrategy({"O2_root": {"L": 0.25, "R": 0.75}}),
    }

    value = evaluate_three_player_profile(tree, _empty_hero(), strategies)

    assert value.H == pytest.approx(-0.75)
    assert value.O1 == pytest.approx(0.25)
    assert value.O2 == pytest.approx(0.5)


def test_fixed_hero_mixed_transition_is_input_not_regret_updated():
    tree = ThreePlayerGameTree(
        FixedHeroNode(
            "hero_lock",
            "H_lock",
            (
                (
                    "left",
                    OpponentDecisionNode(
                        "o1_left",
                        "opponent_1",
                        "O1_left",
                        (
                            ("take", _t("t_left_take", -3.0, 2.0, 1.0)),
                            ("pass", _t("t_left_pass", 0.0, 0.0, 0.0)),
                        ),
                    ),
                ),
                ("right", _t("t_right", -1.0, 0.0, 1.0)),
            ),
        )
    )
    hero_policy = BehaviorStrategy({"H_lock": {"left": 0.25, "right": 0.75}})
    strategies = {
        "O1": BehaviorStrategy({"O1_left": {"take": 0.5, "pass": 0.5}}),
        "O2": BehaviorStrategy({}),
    }

    value = evaluate_three_player_profile(tree, hero_policy, strategies)
    result = run_three_player_cfr_diagnostic(
        tree,
        hero_policy,
        config=CfrConfig(iterations=10),
        attestation=_attest(tree),
    )

    assert value.H == pytest.approx(-1.125)
    assert "H" not in result.info_set_count_by_player
    assert result.info_set_count_by_player == {"O1": 1, "O2": 0}
    assert result.average_strategy_by_player["O2"] == {}


def test_repeated_opponent_information_set_on_single_path_is_rejected():
    tree = ThreePlayerGameTree(
        OpponentDecisionNode(
            "o1_first",
            "opponent_1",
            "O1_loop",
            (
                (
                    "again",
                    OpponentDecisionNode(
                        "o1_repeat",
                        "opponent_1",
                        "O1_loop",
                        (("stop", _t("t_repeat_stop", 0.0, 0.0, 0.0)),),
                    ),
                ),
                ("stop", _t("t_first_stop", 0.0, 0.0, 0.0)),
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="repeats on a single root-to-terminal path",
    ):
        validate_three_player_tree(tree, _empty_hero())


def test_reused_information_set_requires_consistent_legal_actions():
    tree = ThreePlayerGameTree(
        OpponentDecisionNode(
            "o1_root",
            "opponent_1",
            "O1_root",
            (
                (
                    "a",
                    OpponentDecisionNode(
                        "o2_after_a",
                        "opponent_2",
                        "O2_shared",
                        (
                            ("call", _t("t_a_call", -2.0, 1.0, 1.0)),
                            ("fold", _t("t_a_fold", 0.0, 0.0, 0.0)),
                        ),
                    ),
                ),
                (
                    "b",
                    OpponentDecisionNode(
                        "o2_after_b",
                        "opponent_2",
                        "O2_shared",
                        (
                            ("call", _t("t_b_call", -1.0, 1.0, 0.0)),
                            ("raise", _t("t_b_raise", -3.0, 1.0, 2.0)),
                        ),
                    ),
                ),
            ),
        )
    )

    with pytest.raises(ValueError, match="inconsistent legal actions"):
        validate_three_player_tree(tree, _empty_hero())


def test_fixed_hero_policy_rejects_unknown_and_missing_info_sets():
    tree = ThreePlayerGameTree(
        FixedHeroNode(
            "hero_lock",
            "H_lock",
            (
                ("left", _t("t_left", 0.0, 0.0, 0.0)),
                ("right", _t("t_right", 0.0, 0.0, 0.0)),
            ),
        )
    )

    with pytest.raises(ValueError, match="unknown information sets"):
        validate_three_player_tree(
            tree,
            BehaviorStrategy(
                {
                    "H_lock": {"left": 0.5, "right": 0.5},
                    "H_extra": {"unused": 1.0},
                }
            ),
        )
    with pytest.raises(ValueError, match="missing information set 'H_lock'"):
        validate_three_player_tree(tree, BehaviorStrategy({}))


def test_opponent_strategy_mapping_rejects_missing_extra_and_info_set_keys():
    tree = _normal_form_tree(
        {
            ("A", "L"): (-2.0, 1.0, 1.0),
            ("A", "R"): (0.0, 0.0, 0.0),
            ("B", "L"): (0.0, 0.0, 0.0),
            ("B", "R"): (-2.0, 1.0, 1.0),
        }
    )
    valid_o1 = BehaviorStrategy({"O1_root": {"A": 0.5, "B": 0.5}})
    valid_o2 = BehaviorStrategy({"O2_root": {"L": 0.5, "R": 0.5}})

    with pytest.raises(ValueError, match="exactly O1 and O2"):
        evaluate_three_player_profile(tree, _empty_hero(), {"O1": valid_o1})
    with pytest.raises(ValueError, match="exactly O1 and O2"):
        evaluate_three_player_profile(
            tree,
            _empty_hero(),
            {"O1": valid_o1, "O2": valid_o2, "H": BehaviorStrategy({})},
        )
    with pytest.raises(ValueError, match="unknown information sets"):
        evaluate_three_player_profile(
            tree,
            _empty_hero(),
            {
                "O1": BehaviorStrategy(
                    {
                        "O1_root": {"A": 0.5, "B": 0.5},
                        "O1_shadow": {"A": 1.0},
                    }
                ),
                "O2": valid_o2,
            },
        )
    with pytest.raises(ValueError, match="missing information set 'O2_root'"):
        evaluate_three_player_profile(
            tree,
            _empty_hero(),
            {"O1": valid_o1, "O2": BehaviorStrategy({})},
        )
    with pytest.raises(ValueError, match="must assign exactly legal actions"):
        evaluate_three_player_profile(
            tree,
            _empty_hero(),
            {"O1": BehaviorStrategy({"O1_root": {"A": 1.0}}), "O2": valid_o2},
        )


def test_unilateral_deviation_gain_does_not_evaluate_coalition_joint_deviation():
    """Only one opponent is varied at a time; no coalition deviation is tested."""

    tree = _normal_form_tree(
        {
            ("A", "L"): (-2.0, 1.0, 1.0),
            ("A", "R"): (0.0, 0.0, 0.0),
            ("B", "L"): (0.0, 0.0, 0.0),
            ("B", "R"): (-2.0, 1.0, 1.0),
        }
    )
    strategies = {
        "O1": BehaviorStrategy({"O1_root": {"A": 1.0, "B": 0.0}}),
        "O2": BehaviorStrategy({"O2_root": {"L": 0.5, "R": 0.5}}),
    }

    gains = compute_unilateral_deviation_gains(
        tree, _empty_hero(), strategies, attestation=_attest(tree)
    )

    assert gains["gain_by_player"]["O1"] == pytest.approx(0.0)
    assert gains["gain_by_player"]["O2"] == pytest.approx(0.5)
    assert gains["oracle_profile_count"] == 4
    assert gains["unavailable_reason"] is None


def test_payoff_conservation_and_non_zero_sum_opponent_subgame_diagnostic():
    tree = _normal_form_tree(
        {
            ("A", "L"): (-2.0, 1.0, 1.0),
            ("A", "R"): (-1.0, 1.0, 0.0),
            ("B", "L"): (0.0, 0.0, 0.0),
            ("B", "R"): (-3.0, 1.0, 2.0),
        }
    )
    profile_a = {
        "O1": BehaviorStrategy({"O1_root": {"A": 1.0, "B": 0.0}}),
        "O2": BehaviorStrategy({"O2_root": {"L": 1.0, "R": 0.0}}),
    }
    profile_b = {
        "O1": BehaviorStrategy({"O1_root": {"A": 0.0, "B": 1.0}}),
        "O2": BehaviorStrategy({"O2_root": {"L": 1.0, "R": 0.0}}),
    }

    value_a = evaluate_three_player_profile(tree, _empty_hero(), profile_a)
    value_b = evaluate_three_player_profile(tree, _empty_hero(), profile_b)

    assert value_a.conservation_residual() == pytest.approx(0.0, abs=TOL)
    assert value_b.conservation_residual() == pytest.approx(0.0, abs=TOL)
    assert value_a.O1 + value_a.O2 == pytest.approx(2.0)
    assert value_b.O1 + value_b.O2 == pytest.approx(0.0)


def test_deterministic_reproducibility_is_dict_level_stable():
    tree = _normal_form_tree(
        {
            ("A", "L"): (-2.0, 1.0, 1.0),
            ("A", "R"): (0.0, 0.0, 0.0),
            ("B", "L"): (0.0, 0.0, 0.0),
            ("B", "R"): (-2.0, 1.0, 1.0),
        }
    )
    config = CfrConfig(iterations=75)

    first = run_three_player_cfr_diagnostic(
        tree, _empty_hero(), config=config, attestation=_attest(tree)
    )
    second = run_three_player_cfr_diagnostic(
        tree, _empty_hero(), config=config, attestation=_attest(tree)
    )

    assert first.to_dict() == second.to_dict()


def test_safety_caps_reject_before_large_structures_are_needed():
    tree = _normal_form_tree(
        {
            ("A", "L"): (-2.0, 1.0, 1.0),
            ("A", "R"): (0.0, 0.0, 0.0),
            ("B", "L"): (0.0, 0.0, 0.0),
            ("B", "R"): (-2.0, 1.0, 1.0),
        }
    )

    with pytest.raises(ValueError, match="max_nodes"):
        validate_three_player_tree(
            tree,
            _empty_hero(),
            limits=CfrSafetyLimits(max_nodes=1),
        )
    with pytest.raises(ValueError, match="max_terminal_nodes"):
        validate_three_player_tree(
            tree,
            _empty_hero(),
            limits=CfrSafetyLimits(max_terminal_nodes=1),
        )
    with pytest.raises(ValueError, match="max_actions_per_info_set"):
        validate_three_player_tree(
            tree,
            _empty_hero(),
            limits=CfrSafetyLimits(max_actions_per_info_set=1),
        )
    with pytest.raises(ValueError, match="max_opponent_info_sets_total"):
        validate_three_player_tree(
            tree,
            _empty_hero(),
            limits=CfrSafetyLimits(max_opponent_info_sets_total=1),
        )
    o1_two_info_tree = ThreePlayerGameTree(
        OpponentDecisionNode(
            "o1_a",
            "opponent_1",
            "O1_a",
            (
                (
                    "next",
                    OpponentDecisionNode(
                        "o1_b",
                        "opponent_1",
                        "O1_b",
                        (("end", _t("t_o1_b_end", 0.0, 0.0, 0.0)),),
                    ),
                ),
            ),
        )
    )
    with pytest.raises(ValueError, match="max_info_sets_per_opponent"):
        validate_three_player_tree(
            o1_two_info_tree,
            _empty_hero(),
            limits=CfrSafetyLimits(max_info_sets_per_opponent=1),
        )
    hero_two_info_tree = ThreePlayerGameTree(
        FixedHeroNode(
            "h_a",
            "H_a",
            (
                (
                    "next",
                    FixedHeroNode(
                        "h_b",
                        "H_b",
                        (("end", _t("t_h_b_end", 0.0, 0.0, 0.0)),),
                    ),
                ),
            ),
        )
    )
    with pytest.raises(ValueError, match="max_fixed_hero_info_sets"):
        validate_three_player_tree(
            hero_two_info_tree,
            _empty_hero(),
            limits=CfrSafetyLimits(max_fixed_hero_info_sets=1),
        )
    chance_two_outcome_tree = ThreePlayerGameTree(
        ThreePlayerChanceNode(
            "chance",
            (
                (0.5, _t("t_chance_a", 0.0, 0.0, 0.0)),
                (0.5, _t("t_chance_b", 0.0, 0.0, 0.0)),
            ),
        )
    )
    with pytest.raises(ValueError, match="max_chance_outcomes_per_node"):
        validate_three_player_tree(
            chance_two_outcome_tree,
            _empty_hero(),
            limits=CfrSafetyLimits(max_chance_outcomes_per_node=1),
        )
    with pytest.raises(ValueError, match="iterations"):
        CfrConfig(iterations=2, limits=CfrSafetyLimits(max_iterations=1))
    hero_bool_tree = ThreePlayerGameTree(
        FixedHeroNode(
            "h",
            "H_bool",
            (("a", _t("t_a", 0.0, 0.0, 0.0)), ("b", _t("t_b", 0.0, 0.0, 0.0))),
        )
    )
    with pytest.raises(ValueError, match="finite number"):
        validate_three_player_tree(
            hero_bool_tree,
            BehaviorStrategy({"H_bool": {"a": True, "b": 0.0}}),
        )


def test_oracle_cap_makes_deviation_gain_unavailable_without_enumeration():
    tree = ThreePlayerGameTree(
        OpponentDecisionNode(
            "o1_a",
            "opponent_1",
            "O1_a",
            (
                (
                    "a",
                    OpponentDecisionNode(
                        "o1_b",
                        "opponent_1",
                        "O1_b",
                        (
                            (
                                "c",
                                OpponentDecisionNode(
                                    "o2",
                                    "opponent_2",
                                    "O2_a",
                                    (
                                        ("x", _t("t_acx", -3.0, 1.0, 2.0)),
                                        ("y", _t("t_acy", -2.0, 1.0, 1.0)),
                                    ),
                                ),
                            ),
                            ("d", _t("t_ad", 0.0, 0.0, 0.0)),
                        ),
                    ),
                ),
                ("b", _t("t_b", 0.0, 0.0, 0.0)),
            ),
        )
    )

    result = run_three_player_cfr_diagnostic(
        tree,
        _empty_hero(),
        config=CfrConfig(
            iterations=3,
            limits=CfrSafetyLimits(max_oracle_pure_profiles=3),
        ),
        attestation=_attest(tree),
    )

    assert result.oracle_profile_count == 8
    assert result.unilateral_deviation_gain_by_player == {"O1": None, "O2": None}
    assert result.deviation_gain_unavailable_reason is not None
    assert result.stopped_by_safety_cap is True


def test_chance_and_probability_validation_rejects_bool_probability():
    tree = ThreePlayerGameTree(
        ThreePlayerChanceNode(
            "chance",
            (
                (True, _t("t_true", 0.0, 0.0, 0.0)),
                (0.0, _t("t_false", 0.0, 0.0, 0.0)),
            ),
        )
    )

    with pytest.raises(ValueError, match="finite number"):
        validate_three_player_tree(tree, _empty_hero())


def test_high_dynamic_range_is_normalized_and_regret_overflow_fails_closed():
    high = 1.797693134e308
    tree = ThreePlayerGameTree(
        FixedHeroNode(
            "hero_high",
            "H_high",
            (("only", _t("t_high", high, -high, 0.0)),),
        )
    )
    hero_policy = BehaviorStrategy({"H_high": {"only": 1.0000000005}})
    opponent_strategies = {"O1": BehaviorStrategy({}), "O2": BehaviorStrategy({})}

    value = evaluate_three_player_profile(tree, hero_policy, opponent_strategies)
    result = run_three_player_cfr_diagnostic(
        tree,
        hero_policy,
        config=CfrConfig(iterations=1),
        attestation=_attest(tree),
    )
    gains = compute_unilateral_deviation_gains(
        tree,
        hero_policy,
        opponent_strategies,
        attestation=_attest(tree),
    )

    assert value.H == high
    assert result.expected_utility_vector["H"] == high
    hero_record = next(
        record
        for record in result.normalization_records
        if record["label"] == "fixed_hero.H_high"
    )
    assert hero_record["raw_sum"] == 1.0000000005
    assert hero_record["effective_probabilities"] == {"only": 1.0}
    assert gains["gain_by_player"] == {"O1": 0.0, "O2": 0.0}

    regret_tree = ThreePlayerGameTree(
        OpponentDecisionNode(
            "o1_high",
            "opponent_1",
            "O1_high",
            (
                ("high", _t("t_o1_high", -high, high, 0.0)),
                ("low", _t("t_o1_low", high, -high, 0.0)),
            ),
        )
    )
    with pytest.raises(ValueError, match="regret advantage.*non-finite"):
        run_three_player_cfr_diagnostic(
            regret_tree,
            _empty_hero(),
            config=CfrConfig(iterations=2),
            attestation=_attest(regret_tree),
        )


def test_unrepresentable_numeric_inputs_fail_closed_at_all_public_boundaries():
    huge = 10**1000
    terminal_tree = ThreePlayerGameTree(_t("huge", huge, -huge, 0.0))
    chance_tree = ThreePlayerGameTree(
        ThreePlayerChanceNode(
            "chance_huge",
            ((huge, _t("chance_a", 0.0, 0.0, 0.0)), (0.0, _t("chance_b", 0.0, 0.0, 0.0))),
        )
    )
    hero_tree = ThreePlayerGameTree(
        FixedHeroNode(
            "hero_huge",
            "H_huge",
            (("a", _t("hero_a", 0.0, 0.0, 0.0)), ("b", _t("hero_b", 0.0, 0.0, 0.0))),
        )
    )
    opponent_tree = ThreePlayerGameTree(
        OpponentDecisionNode(
            "opponent_huge",
            "opponent_1",
            "O1_huge",
            (("a", _t("opponent_a", 0.0, 0.0, 0.0)), ("b", _t("opponent_b", 0.0, 0.0, 0.0))),
        )
    )

    cases = (
        lambda: run_three_player_cfr_diagnostic(
            terminal_tree,
            _empty_hero(),
            config=CfrConfig(iterations=1),
            attestation=_attest(terminal_tree),
        ),
        lambda: run_three_player_cfr_diagnostic(
            chance_tree,
            _empty_hero(),
            config=CfrConfig(iterations=1),
            attestation=_attest(chance_tree),
        ),
        lambda: run_three_player_cfr_diagnostic(
            hero_tree,
            BehaviorStrategy({"H_huge": {"a": huge, "b": 0.0}}),
            config=CfrConfig(iterations=1),
            attestation=_attest(hero_tree),
        ),
        lambda: compute_unilateral_deviation_gains(
            opponent_tree,
            _empty_hero(),
            {
                "O1": BehaviorStrategy({"O1_huge": {"a": huge, "b": 0.0}}),
                "O2": BehaviorStrategy({}),
            },
            attestation=_attest(opponent_tree),
        ),
    )
    for invoke in cases:
        with pytest.raises(DiagnosticContractError) as failure:
            invoke()
        assert failure.value.status == NUMERIC_FAILURE
        assert "finite float" in str(failure.value)


def test_oracle_pure_gain_overflow_is_numeric_failure_before_hashing():
    high = 1.797693134e308
    tree = ThreePlayerGameTree(
        OpponentDecisionNode(
            "o1_gain_overflow",
            "opponent_1",
            "O1_gain_overflow",
            (
                ("high", _t("gain_high", -high, high, 0.0)),
                ("low", _t("gain_low", high, -high, 0.0)),
            ),
        )
    )
    with pytest.raises(DiagnosticContractError) as failure:
        run_three_player_cfr_diagnostic(
            tree,
            _empty_hero(),
            config=CfrConfig(
                iterations=1, request_oracle=True, include_oracle_rows=True
            ),
            attestation=_attest(tree),
        )
    assert failure.value.status == NUMERIC_FAILURE
    assert "oracle pure gain for O1" in str(failure.value)


def test_attestation_is_required_confirmed_and_bound_to_tree_content():
    tree = _normal_form_tree(
        {key: (0.0, 0.0, 0.0) for key in (("A", "L"), ("A", "R"), ("B", "L"), ("B", "R"))}
    )
    valid = _attest(tree)
    assert run_three_player_cfr_diagnostic(
        tree, _empty_hero(), config=CfrConfig(iterations=1), attestation=valid
    ).component_status == "DIAGNOSTIC_COMPLETE"

    with pytest.raises(DiagnosticContractError) as missing:
        run_three_player_cfr_diagnostic(
            tree, _empty_hero(), config=CfrConfig(iterations=1)
        )
    assert missing.value.status == UNSUPPORTED_MODEL
    with pytest.raises(DiagnosticContractError) as unconfirmed:
        run_three_player_cfr_diagnostic(
            tree,
            _empty_hero(),
            config=CfrConfig(iterations=1),
            attestation=_attest(tree, o2_confirmed=False),
        )
    assert unconfirmed.value.status == UNSUPPORTED_MODEL
    changed = ThreePlayerGameTree(tree.root, description="changed")
    with pytest.raises(DiagnosticContractError, match="identity mismatch"):
        run_three_player_cfr_diagnostic(
            changed,
            _empty_hero(),
            config=CfrConfig(iterations=1),
            attestation=valid,
        )


def test_duplicate_node_id_is_rejected():
    tree = ThreePlayerGameTree(
        ThreePlayerChanceNode(
            "chance",
            ((0.5, _t("duplicate", 0.0, 0.0, 0.0)), (0.5, _t("duplicate", 0.0, 0.0, 0.0))),
        )
    )
    with pytest.raises(ValueError, match="duplicate node id"):
        validate_three_player_tree(tree, _empty_hero())


def test_chance_hero_and_opponent_normalization_records_without_mutation():
    tree = ThreePlayerGameTree(
        ThreePlayerChanceNode(
            "chance",
            (
                (
                    0.50000000025,
                    FixedHeroNode(
                        "hero",
                        "H",
                        (("x", _t("tx", 0.0, 0.0, 0.0)), ("y", _t("ty", 0.0, 0.0, 0.0))),
                    ),
                ),
                (0.50000000025, _t("other", 0.0, 0.0, 0.0)),
            ),
        )
    )
    hero = BehaviorStrategy({"H": {"x": 0.50000000025, "y": 0.50000000025}})
    original_tree = copy.deepcopy(tree)
    original_hero = copy.deepcopy(hero)
    result = run_three_player_cfr_diagnostic(
        tree,
        hero,
        config=CfrConfig(iterations=1),
        attestation=_attest(tree),
    )
    records = {record["label"]: record for record in result.normalization_records}
    assert records["chance.chance"]["raw_sum"] == 1.0000000005
    assert records["chance.chance"]["effective_sum"] == 1.0
    assert records["fixed_hero.H"]["effective_probabilities"] == {"x": 0.5, "y": 0.5}
    assert tree == original_tree
    assert hero == original_hero

    opponent_tree = ThreePlayerGameTree(
        OpponentDecisionNode(
            "o1", "opponent_1", "I", (("a", _t("ta", 0.0, 0.0, 0.0)), ("b", _t("tb", 0.0, 0.0, 0.0)))
        )
    )
    profiles = {
        "O1": BehaviorStrategy({"I": {"a": 0.50000000025, "b": 0.50000000025}}),
        "O2": BehaviorStrategy({}),
    }
    gains = compute_unilateral_deviation_gains(
        opponent_tree,
        _empty_hero(),
        profiles,
        attestation=_attest(opponent_tree),
    )
    opponent_record = next(r for r in gains["normalization_records"] if r["label"] == "O1.I")
    assert opponent_record["raw_sum"] == 1.0000000005
    assert opponent_record["effective_probabilities"] == {"a": 0.5, "b": 0.5}
    assert profiles["O1"].probabilities["I"]["a"] == 0.50000000025


def test_probability_outside_tolerance_rejected_and_non_null_seed_rejected():
    tree = ThreePlayerGameTree(
        ThreePlayerChanceNode(
            "chance",
            ((0.6, _t("a", 0.0, 0.0, 0.0)), (0.5, _t("b", 0.0, 0.0, 0.0))),
        )
    )
    with pytest.raises(ValueError, match="expected 1"):
        validate_three_player_tree(tree, _empty_hero())
    with pytest.raises(DiagnosticContractError) as seed_error:
        CfrConfig(seed=7)
    assert seed_error.value.status == UNSUPPORTED_MODEL


def test_trace_default_checkpoints_and_preallocation_cap():
    tree = _normal_form_tree(
        {key: (0.0, 0.0, 0.0) for key in (("A", "L"), ("A", "R"), ("B", "L"), ("B", "R"))}
    )
    default = run_three_player_cfr_diagnostic(
        tree, _empty_hero(), config=CfrConfig(iterations=3), attestation=_attest(tree)
    )
    assert default.trace == {"enabled": False, "coverage": "none", "schedule": [], "points": []}
    traced = run_three_player_cfr_diagnostic(
        tree,
        _empty_hero(),
        config=CfrConfig(iterations=5, trace_checkpoint_interval=2),
        attestation=_attest(tree),
    )
    assert traced.trace["coverage"] == "checkpoints"
    assert [point["iteration"] for point in traced.trace["points"]] == [2, 4, 5]
    with pytest.raises(DiagnosticContractError, match="trace point count"):
        run_three_player_cfr_diagnostic(
            tree,
            _empty_hero(),
            config=CfrConfig(
                iterations=5,
                trace_checkpoint_interval=1,
                limits=CfrSafetyLimits(max_trace_points=4),
            ),
            attestation=_attest(tree),
        )


def _oracle_result(tree, **config_values):
    config = CfrConfig(iterations=2, request_oracle=True, **config_values)
    return run_three_player_cfr_diagnostic(
        tree, _empty_hero(), config=config, attestation=_attest(tree)
    )


def test_complete_oracle_coordination_matching_pennies_and_coalition_counterexample():
    coordination = _normal_form_tree(
        {
            ("A", "L"): (-2.0, 1.0, 1.0),
            ("A", "R"): (0.0, 0.0, 0.0),
            ("B", "L"): (0.0, 0.0, 0.0),
            ("B", "R"): (-4.0, 2.0, 2.0),
        }
    )
    oracle = _oracle_result(coordination, include_oracle_rows=True).oracle_attachment
    assert oracle["status"] == "MATCH"
    assert oracle["coverage"] == "complete"
    assert oracle["stable_profile_ids"] == ["O1:0|O2:0", "O1:1|O2:1"]
    stable_rows = [row for row in oracle["rows"] if row["pure_profile_unilateral_stability"]]
    assert {row["utility"]["H"] for row in stable_rows} == {-2.0, -4.0}
    first = stable_rows[0]
    assert first["unilateral_gain"]["O1"] < 0.0
    assert first["unilateral_gain"]["O2"] < 0.0
    # The (B,R) row improves both opponent components jointly; no joint field is emitted.
    assert all("coalition" not in key and "joint_gain" not in key for key in first)

    matching = _normal_form_tree(
        {
            ("A", "L"): (0.0, 1.0, -1.0),
            ("A", "R"): (0.0, -1.0, 1.0),
            ("B", "L"): (0.0, -1.0, 1.0),
            ("B", "R"): (0.0, 1.0, -1.0),
        }
    )
    matching_oracle = _oracle_result(matching).oracle_attachment
    assert matching_oracle["stable_profile_count"] == 0
    assert matching_oracle["status"] == "MATCH"


def test_oracle_threshold_raw_gain_and_unreachable_shared_information_set():
    delta = 0.25
    tree = ThreePlayerGameTree(
        OpponentDecisionNode(
            "o1", "opponent_1", "O1", (("a", _t("a", 0.0, 0.0, 0.0)), ("b", _t("b", -delta, delta, 0.0)))
        )
    )
    exact = _oracle_result(tree, epsilon_deviation=delta, include_oracle_rows=True)
    row_a = exact.oracle_attachment["rows"][0]
    assert row_a["unilateral_gain"]["O1"] == delta
    assert row_a["pure_profile_unilateral_stability"] is True
    below = _oracle_result(tree, epsilon_deviation=delta - 1e-12, include_oracle_rows=True)
    assert below.oracle_attachment["rows"][0]["pure_profile_unilateral_stability"] is False
    above = _oracle_result(tree, epsilon_deviation=delta + 1e-12, include_oracle_rows=True)
    assert above.oracle_attachment["rows"][0]["pure_profile_unilateral_stability"] is True

    unreachable = ThreePlayerGameTree(
        ThreePlayerChanceNode(
            "c",
            (
                (1.0, _t("live", 0.0, 0.0, 0.0)),
                (
                    0.0,
                    OpponentDecisionNode(
                        "o2a", "opponent_2", "shared", (("x", _t("ux", 0.0, 0.0, 0.0)), ("y", _t("uy", 0.0, 0.0, 0.0)))
                    ),
                ),
            ),
        )
    )
    unreachable_result = _oracle_result(unreachable)
    assert unreachable_result.oracle_attachment["coverage"] == "complete"
    assert "conditional" not in str(unreachable_result.to_dict()).lower()


@pytest.mark.parametrize(
    ("limits", "include_rows", "failure"),
    [
        (CfrSafetyLimits(max_oracle_pure_plans_per_player=1), False, "max_oracle_pure_plans_per_player"),
        (CfrSafetyLimits(max_oracle_joint_profiles=3), False, "max_oracle_joint_profiles"),
        (CfrSafetyLimits(max_oracle_profile_evaluations=12), False, "max_oracle_profile_evaluations"),
        (CfrSafetyLimits(max_oracle_output_rows=3), True, "max_oracle_output_rows"),
    ],
)
def test_four_oracle_caps_fail_before_rows(limits, include_rows, failure):
    tree = _normal_form_tree(
        {key: (0.0, 0.0, 0.0) for key in (("A", "L"), ("A", "R"), ("B", "L"), ("B", "R"))}
    )
    result = _oracle_result(tree, limits=limits, include_oracle_rows=include_rows)
    assert result.overall_status == ORACLE_UNAVAILABLE_CAP
    assert result.oracle_attachment["rows"] == []
    assert failure in result.oracle_attachment["cap_failures"]


def test_oracle_matches_utilities_mixture_and_gains_and_not_requested_status():
    tree = _normal_form_tree(
        {
            ("A", "L"): (-3.0, 1.0, 2.0),
            ("A", "R"): (0.0, 0.0, 0.0),
            ("B", "L"): (0.0, 0.0, 0.0),
            ("B", "R"): (-4.0, 2.0, 2.0),
        }
    )
    result = _oracle_result(tree)
    assert result.oracle_attachment["status"] == "MATCH"
    assert result.overall_status == "DIAGNOSTIC_COMPLETE"
    assert result.oracle_attachment["comparisons"]["pure_utility_max_delta"] == 0.0
    assert all(value == 0.0 for key, value in result.oracle_attachment["comparisons"].items() if key.endswith("delta"))
    not_requested = run_three_player_cfr_diagnostic(
        tree, _empty_hero(), config=CfrConfig(iterations=1), attestation=_attest(tree)
    )
    assert not_requested.oracle_attachment["status"] == NOT_REQUESTED


def test_oracle_mismatch_and_tolerance_crossing_fail_closed(monkeypatch):
    tree = _normal_form_tree(
        {key: (0.0, 0.0, 0.0) for key in (("A", "L"), ("A", "R"), ("B", "L"), ("B", "R"))}
    )
    original = three_player_module._oracle_direct_evaluate

    def shifted(*args, **kwargs):
        value = original(*args, **kwargs)
        return UtilityVector(value.H + 1e-6, value.O1, value.O2, value.R - 1e-6)

    monkeypatch.setattr(three_player_module, "_oracle_direct_evaluate", shifted)
    mismatch = _oracle_result(tree, oracle_compare_tolerance=1e-9)
    assert mismatch.overall_status == ORACLE_MISMATCH

    def boundary_shift(*args, **kwargs):
        value = original(*args, **kwargs)
        shift = 1.0005e-9
        return UtilityVector(value.H + shift, value.O1, value.O2, value.R - shift)

    monkeypatch.setattr(three_player_module, "_oracle_direct_evaluate", boundary_shift)
    crossing = _oracle_result(
        tree, oracle_compare_tolerance=1e-9, reproducibility_tolerance=1e-12
    )
    assert crossing.overall_status == INDETERMINATE_TOLERANCE


def test_pure_utility_reference_path_detects_one_sided_perturbation(monkeypatch):
    tree = _normal_form_tree(
        {key: (0.0, 0.0, 0.0) for key in (("A", "L"), ("A", "R"), ("B", "L"), ("B", "R"))}
    )
    original = three_player_module._oracle_direct_evaluate

    def perturb_reference_only(*args, **kwargs):
        value = original(*args, **kwargs)
        return UtilityVector(value.H + 1e-6, value.O1, value.O2, value.R)

    monkeypatch.setattr(
        three_player_module, "_oracle_direct_evaluate", perturb_reference_only
    )
    mismatch = _oracle_result(tree, oracle_compare_tolerance=1e-9)
    assert mismatch.overall_status == ORACLE_MISMATCH
    assert mismatch.oracle_attachment["comparisons"]["pure_utility_max_delta"] == pytest.approx(1e-6)


def test_semantic_result_is_exact_order_stable_finite_and_uses_bounded_wording():
    tree = _normal_form_tree(
        {
            ("A", "L"): (-2.0, 1.0, 1.0),
            ("A", "R"): (0.0, 0.0, 0.0),
            ("B", "L"): (0.0, 0.0, 0.0),
            ("B", "R"): (-2.0, 1.0, 1.0),
        }
    )
    first = _oracle_result(tree, trace_checkpoint_interval=1).to_dict()
    second = _oracle_result(tree, trace_checkpoint_interval=1).to_dict()
    assert first == second
    assert list(first["average_strategy_by_player"]) == ["O1", "O2"]

    def assert_finite(value):
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return
        if isinstance(value, (int, float)):
            assert math.isfinite(value)
        elif isinstance(value, dict):
            for nested in value.values():
                assert_finite(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                assert_finite(nested)

    assert_finite(first)
    forbidden_keys = {
        "solution",
        "equilibrium",
        "nash",
        "certificate",
        "proof",
        "converged",
        "optimal",
        "exploitability",
        "profitable",
    }
    assert not (forbidden_keys & set(first))


def test_invalid_cap_and_numeric_failures_carry_distinct_statuses():
    invalid = ThreePlayerGameTree(
        ThreePlayerChanceNode(
            "c", ((0.6, _t("x", 0.0, 0.0, 0.0)), (0.5, _t("y", 0.0, 0.0, 0.0)))
        )
    )
    with pytest.raises(DiagnosticContractError) as invalid_error:
        run_three_player_cfr_diagnostic(
            invalid,
            _empty_hero(),
            config=CfrConfig(iterations=1),
            attestation=_attest(invalid),
        )
    assert invalid_error.value.status == INVALID_INPUT

    capped = _normal_form_tree(
        {key: (0.0, 0.0, 0.0) for key in (("A", "L"), ("A", "R"), ("B", "L"), ("B", "R"))}
    )
    with pytest.raises(DiagnosticContractError) as cap_error:
        run_three_player_cfr_diagnostic(
            capped,
            _empty_hero(),
            config=CfrConfig(iterations=1, limits=CfrSafetyLimits(max_nodes=1)),
            attestation=_attest(capped),
        )
    assert cap_error.value.status == CAP_EXCEEDED

    high = 1.797693134e308
    numeric = ThreePlayerGameTree(
        OpponentDecisionNode(
            "o1",
            "opponent_1",
            "I",
            (("high", _t("high", -high, high, 0.0)), ("low", _t("low", high, -high, 0.0))),
        )
    )
    with pytest.raises(DiagnosticContractError) as numeric_error:
        run_three_player_cfr_diagnostic(
            numeric,
            _empty_hero(),
            config=CfrConfig(iterations=2),
            attestation=_attest(numeric),
        )
    assert numeric_error.value.status == NUMERIC_FAILURE


def test_public_contract_docstrings_and_fields_avoid_forbidden_claims():
    public_objects = (
        CfrConfig,
        CfrDiagnosticResult,
        PerfectRecallAttestation,
        run_three_player_cfr_diagnostic,
        compute_unilateral_deviation_gains,
        tree_content_identity,
    )
    public_text = "\n".join(inspect.getdoc(obj) or "" for obj in public_objects).lower()
    public_text += "\n" + "\n".join(CfrDiagnosticResult.__dataclass_fields__).lower()
    forbidden = (
        "solution",
        "equilibrium",
        "nash",
        "ppe",
        "certificate",
        "proof",
        "converged",
        "convergence guarantee",
        "optimal",
        "solver-grade",
        "exploitability",
        "profitable",
        "real-money recommendation",
    )
    assert all(
        re.search(rf"\b{re.escape(term)}\b", public_text) is None
        for term in forbidden
    )
