"""Tests for the isolated 3-player CFR-style diagnostic prototype."""

import pytest

from repeated_poker.three_player_cfr import (
    BehaviorStrategy,
    CfrConfig,
    CfrSafetyLimits,
    FixedHeroNode,
    OpponentDecisionNode,
    ThreePlayerChanceNode,
    ThreePlayerGameTree,
    ThreePlayerTerminalNode,
    UtilityVector,
    compute_unilateral_deviation_gains,
    count_opponent_pure_profiles,
    evaluate_three_player_profile,
    run_three_player_cfr_diagnostic,
    validate_three_player_tree,
)

TOL = 1e-9


def _u(hero, o1, o2, residual=0.0):
    return UtilityVector(H=hero, O1=o1, O2=o2, R=residual)


def _t(node_id, hero, o1, o2, residual=0.0):
    return ThreePlayerTerminalNode(node_id, _u(hero, o1, o2, residual))


def _empty_hero():
    return BehaviorStrategy({})


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
    )

    assert value.H == pytest.approx(-1.125)
    assert "H" not in result.info_set_count_by_player
    assert result.info_set_count_by_player == {"O1": 1, "O2": 0}
    assert result.average_strategy_by_player["O2"] == {}


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

    gains = compute_unilateral_deviation_gains(tree, _empty_hero(), strategies)

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

    first = run_three_player_cfr_diagnostic(tree, _empty_hero(), config=config)
    second = run_three_player_cfr_diagnostic(tree, _empty_hero(), config=config)

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
