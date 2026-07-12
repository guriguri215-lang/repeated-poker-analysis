from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path

import pytest

from repeated_poker.game import (
    ChanceNode,
    GameTree,
    HeroNode,
    TerminalNode,
    VillainNode,
    collect_hero_info_sets,
    collect_villain_info_sets,
    iter_nodes,
)
from repeated_poker.stage_plan_diagnostic import (
    COOPERATE,
    HERO,
    PLAYERS,
    PUBLIC_STATES,
    PUNISH,
    VILLAIN,
    DiagnosticStatus,
    ManualPerfectRecallAttestation,
    ModelClassAttestation,
    NumericErrorBound,
    PublicAction,
    PublicMonitoring,
    PublicSignal,
    RecallHistory,
    diagnose_stage_plan_deviations,
    exact_zero_error_bound,
    nuts_chop_negative_oracle,
    tree_content_identity,
)


def _model_attestation(**changes):
    values = {
        "iid_stage_kernel": True,
        "no_persistent_private_state": True,
        "no_cross_period_correlation": True,
        "no_private_payoff_state": True,
        "public_state_does_not_change_stage_kernel": True,
        "public_state_is_sufficient": True,
        "signal_partition_is_public": True,
        "signal_excludes_private_information": True,
        "signal_excludes_deviator_identity": True,
        "no_known_finite_horizon": True,
        "absorbing_grim_only": True,
    }
    values.update(changes)
    return ModelClassAttestation(**values)


def _info_sets(tree):
    return {
        HERO: collect_hero_info_sets(tree),
        VILLAIN: collect_villain_info_sets(tree),
    }


def _attestation(tree, *, version="v1", invalidated=False, valid_through=None):
    members = {HERO: {}, VILLAIN: {}}
    histories = {HERO: {}, VILLAIN: {}}

    def walk(
        node,
        hero_information_sets=(),
        hero_actions=(),
        villain_information_sets=(),
        villain_actions=(),
    ):
        if isinstance(node, TerminalNode):
            return
        if isinstance(node, ChanceNode):
            for _, child in node.children:
                walk(
                    child,
                    hero_information_sets,
                    hero_actions,
                    villain_information_sets,
                    villain_actions,
                )
            return
        if isinstance(node, HeroNode):
            members[HERO].setdefault(node.info_set, []).append(node.node_id)
            histories[HERO].setdefault(node.info_set, {})[node.node_id] = RecallHistory(
                (), hero_actions, hero_information_sets
            )
            for action, child in node.actions:
                walk(
                    child,
                    hero_information_sets + (node.info_set,),
                    hero_actions + (action,),
                    villain_information_sets,
                    villain_actions,
                )
            return
        members[VILLAIN].setdefault(node.info_set, []).append(node.node_id)
        histories[VILLAIN].setdefault(node.info_set, {})[node.node_id] = RecallHistory(
            (), villain_actions, villain_information_sets
        )
        for action, child in node.actions:
            walk(
                child,
                hero_information_sets,
                hero_actions,
                villain_information_sets + (node.info_set,),
                villain_actions + (action,),
            )

    walk(tree.root)
    frozen_members = {
        player: {info: tuple(nodes) for info, nodes in members[player].items()}
        for player in PLAYERS
    }
    return ManualPerfectRecallAttestation(
        fixture_id="test-fixture",
        tree_content_identity=tree_content_identity(tree),
        target_version=version,
        information_set_members=frozen_members,
        member_histories=histories,
        legal_actions=_info_sets(tree),
        reviewer="test reviewer",
        review_date="2026-07-12",
        review_method="manual path comparison",
        evidence="fixture construction and path table",
        result_confirmed=True,
        known_limitations=("fixture-specific",),
        invalidation_conditions=("tree or information partition changes",),
        valid_through_version=version if valid_through is None else valid_through,
        invalidated=invalidated,
    )


def _signals(tree, public_node_ids, terminal_observables):
    result = []

    def walk(node, trace):
        if isinstance(node, TerminalNode):
            result.append(PublicSignal(trace, terminal_observables[node.node_id]))
        elif isinstance(node, ChanceNode):
            for _, child in node.children:
                walk(child, trace)
        else:
            actor = HERO if isinstance(node, HeroNode) else VILLAIN
            for action, child in node.actions:
                extra = (PublicAction(actor, action),) if node.node_id in public_node_ids else ()
                walk(child, trace + extra)

    walk(tree.root, ())
    return tuple(dict.fromkeys(result))


def _monitoring(tree, c_to_p=()):
    public_nodes = frozenset(
        node.node_id
        for node in iter_nodes(tree.root)
        if isinstance(node, (HeroNode, VillainNode))
    )
    observables = {
        node.node_id: "terminal"
        for node in iter_nodes(tree.root)
        if isinstance(node, TerminalNode)
    }
    alphabet = _signals(tree, public_nodes, observables)
    trigger = set(c_to_p)
    transitions = {}
    for signal in alphabet:
        transitions[(COOPERATE, signal)] = PUNISH if signal in trigger else COOPERATE
        transitions[(PUNISH, signal)] = PUNISH
    return PublicMonitoring(public_nodes, observables, alphabet, transitions)


def _pure_profile(tree, *, c=None, p=None):
    info_sets = _info_sets(tree)
    c = {} if c is None else c
    p = c if p is None else p

    def state_profile(selected):
        result = {}
        for player in PLAYERS:
            result[player] = {}
            for info_set, actions in info_sets[player].items():
                chosen = selected.get((player, info_set), actions[0])
                result[player][info_set] = {
                    action: F(action == chosen) for action in actions
                }
        return result

    return {COOPERATE: state_profile(c), PUNISH: state_profile(p)}


def _run(tree, *, profile=None, monitoring=None, attestation="default", **changes):
    arguments = {
        "tree": tree,
        "fixture_version": "v1",
        "profile": _pure_profile(tree) if profile is None else profile,
        "monitoring": _monitoring(tree) if monitoring is None else monitoring,
        "model_attestation": _model_attestation(),
        "perfect_recall_attestation": (
            _attestation(tree) if attestation == "default" else attestation
        ),
        "delta": F(1, 2),
        "stage_payoff_bound": F(10),
        "input_tolerance": F(0),
        "epsilon_claim": F(0),
        "numeric_error_bound": exact_zero_error_bound(),
        "max_plans_per_player": 100,
    }
    arguments.update(changes)
    return diagnose_stage_plan_deviations(**arguments)


def _leaf(node_id, hero, villain=None, residual=0):
    hero = F(hero)
    residual = F(residual)
    villain = -(hero + residual) if villain is None else F(villain)
    return TerminalNode(node_id, hero, villain, residual)


def test_exact_two_state_bellman_values_and_gain():
    root = HeroNode(
        "h",
        "H1",
        (("stay", _leaf("t_stay", 1)), ("trigger", _leaf("t_trigger", 3))),
    )
    tree = GameTree(root)
    monitoring = _monitoring(tree)
    trigger_signal = next(
        signal for signal in monitoring.signal_alphabet if signal.action_trace[-1].action == "trigger"
    )
    monitoring = _monitoring(tree, (trigger_signal,))
    profile = _pure_profile(
        tree,
        c={(HERO, "H1"): "stay"},
        p={(HERO, "H1"): "trigger"},
    )
    result = _run(tree, profile=profile, monitoring=monitoring)

    assert result.prescribed_values[(HERO, COOPERATE)] == 2
    assert result.prescribed_values[(HERO, PUNISH)] == 6
    trigger = next(
        item
        for item in result.deviations
        if item.player == HERO
        and item.state == COOPERATE
        and dict(item.plan.actions)["H1"] == "trigger"
    )
    assert trigger.deviation_value == 6
    assert trigger.gain == 4
    assert (trigger.lower, trigger.upper) == (F(4), F(4))
    assert result.status == DiagnosticStatus.FAIL


def test_full_plan_enumeration_covers_both_players_states_and_information_sets():
    tree = GameTree(
        HeroNode(
            "h1",
            "H1",
            (
                ("a", VillainNode("v1", "V1", (("x", _leaf("t1", 0)), ("y", _leaf("t2", 1))))),
                ("b", HeroNode("h2", "H2", (("c", _leaf("t3", 2)), ("d", _leaf("t4", 3))))),
            ),
        )
    )
    result = _run(tree)

    assert result.plan_counts == {HERO: 4, VILLAIN: 2}
    assert len(result.deviations) == 2 * (4 + 2)
    assert {item.state for item in result.deviations} == set(PUBLIC_STATES)
    assert {item.player for item in result.deviations} == set(PLAYERS)
    hero_plans = {item.plan.actions for item in result.deviations if item.player == HERO}
    assert all({info for info, _ in plan} == {"H1", "H2"} for plan in hero_plans)


def test_full_plan_finds_joint_change_that_single_information_set_checks_miss():
    second = HeroNode("h2", "H2", (("bad", _leaf("bad", -1)), ("good", _leaf("good", 2))))
    tree = GameTree(HeroNode("h1", "H1", (("stop", _leaf("stop", 0)), ("go", second))))
    result = _run(tree)
    gains = {
        tuple(item.plan.actions): item.gain
        for item in result.deviations
        if item.player == HERO and item.state == COOPERATE
    }

    assert gains[(("H1", "stop"), ("H2", "good"))] == 0
    assert gains[(("H1", "go"), ("H2", "bad"))] == -1
    assert gains[(("H1", "go"), ("H2", "good"))] == 2


def test_zero_reach_plan_gain_does_not_assess_reached_action_choice():
    second = HeroNode("h2", "H2", (("bad", _leaf("bad", -5)), ("good", _leaf("good", 5))))
    tree = GameTree(HeroNode("h1", "H1", (("stop", _leaf("stop", 0)), ("go", second))))
    result = _run(tree)
    zero_reach = next(
        item
        for item in result.deviations
        if item.player == HERO
        and item.state == COOPERATE
        and item.plan.actions == (("H1", "stop"), ("H2", "good"))
    )
    assert zero_reach.gain == 0
    assert 5 - (-5) == 10


def test_p_is_absorbing_repeated_state_not_a_terminal_state():
    tree = GameTree(HeroNode("h", "H1", (("one", _leaf("t1", 1)), ("zero", _leaf("t0", 0)))))
    profile = _pure_profile(tree, p={(HERO, "H1"): "one"})
    result = _run(tree, profile=profile, delta=F(3, 4))
    assert result.prescribed_values[(HERO, PUNISH)] == 4


def test_zero_probability_legal_signal_still_requires_transition():
    tree = GameTree(HeroNode("h", "H1", (("a", _leaf("ta", 0)), ("b", _leaf("tb", 1)))))
    monitoring = _monitoring(tree)
    omitted_signal = next(signal for signal in monitoring.signal_alphabet if signal.action_trace[-1].action == "b")
    transitions = dict(monitoring.transitions)
    transitions.pop((COOPERATE, omitted_signal))
    partial = PublicMonitoring(
        monitoring.public_action_node_ids,
        monitoring.terminal_observables,
        monitoring.signal_alphabet,
        transitions,
    )
    with pytest.raises(ValueError, match="total on Q x Y"):
        _run(tree, monitoring=partial)


def test_partial_transition_fails_closed_even_when_omission_is_in_p():
    tree = GameTree(_leaf("t", 0))
    monitoring = _monitoring(tree)
    transitions = dict(monitoring.transitions)
    transitions.pop((PUNISH, monitoring.signal_alphabet[0]))
    partial = PublicMonitoring(
        monitoring.public_action_node_ids,
        monitoring.terminal_observables,
        monitoring.signal_alphabet,
        transitions,
    )
    with pytest.raises(ValueError, match="total on Q x Y"):
        _run(tree, monitoring=partial)


def test_mixed_support_action_is_not_augmented_with_deviator_identity():
    tree = GameTree(HeroNode("h", "H1", (("a", _leaf("ta", 0)), ("b", _leaf("tb", 0)))))
    profile = _pure_profile(tree)
    for state in PUBLIC_STATES:
        profile[state][HERO]["H1"] = {"a": F(1, 2), "b": F(1, 2)}
    result = _run(tree, profile=profile)
    signals = {item.plan.actions: item for item in result.deviations if item.player == HERO}
    assert result.status == DiagnosticStatus.PASS
    assert all(item.gain == 0 for item in signals.values())
    assert all(not hasattr(signal, "deviator") for signal in _monitoring(tree).signal_alphabet)


def test_enumeration_cap_returns_no_partial_result():
    tree = GameTree(HeroNode("h1", "H1", (("a", _leaf("ta", 0)), ("b", HeroNode("h2", "H2", (("x", _leaf("tx", 0)), ("y", _leaf("ty", 0))))))))
    result = _run(tree, max_plans_per_player=3)
    assert result.status == DiagnosticStatus.UNSUPPORTED
    assert result.plan_counts[HERO] == 4
    assert result.deviations == ()


@pytest.mark.parametrize(
    "attestation",
    [None, "expired"],
)
def test_missing_or_expired_manual_attestation_is_unsupported(attestation):
    tree = GameTree(_leaf("t", 0))
    supplied = None if attestation is None else _attestation(tree, valid_through="v0")
    result = _run(tree, attestation=supplied)
    assert result.status == DiagnosticStatus.UNSUPPORTED
    assert result.deviations == ()


def test_invalid_numeric_probability_and_missing_profile_are_input_errors():
    tree = GameTree(HeroNode("h", "H1", (("a", _leaf("ta", 0)), ("b", _leaf("tb", 0)))))
    profile = _pure_profile(tree)
    profile[COOPERATE][HERO]["H1"]["a"] = F(-1)
    profile[COOPERATE][HERO]["H1"]["b"] = F(2)
    with pytest.raises(ValueError, match="negative probability"):
        _run(tree, profile=profile)

    profile = _pure_profile(tree)
    del profile[PUNISH]
    with pytest.raises(ValueError, match="states C and P"):
        _run(tree, profile=profile)

    with pytest.raises(ValueError, match="finite"):
        _run(tree, delta=float("nan"))


def test_finite_float_input_is_indeterminate_without_representation_enclosure():
    tree = GameTree(_leaf("t", 0))
    result = _run(tree, delta=0.5)
    assert result.status == DiagnosticStatus.INDETERMINATE
    assert result.deviations == ()


def test_finite_float_error_bound_is_indeterminate_without_representation_enclosure():
    tree = GameTree(_leaf("t", 0))
    bound = NumericErrorBound(0.0, F(0), F(0), F(0), F(0), F(0), F(0), F(0), True)
    result = _run(tree, numeric_error_bound=bound)
    assert result.status == DiagnosticStatus.INDETERMINATE
    assert result.deviations == ()


@pytest.mark.parametrize("component", [float("nan"), float("inf"), F(-1)])
def test_invalid_numeric_error_bound_remains_an_input_error(component):
    tree = GameTree(_leaf("t", 0))
    bound = NumericErrorBound(component, F(0), F(0), F(0), F(0), F(0), F(0), F(0), True)
    with pytest.raises(ValueError, match="finite|non-negative"):
        _run(tree, numeric_error_bound=bound)


def test_attested_empty_history_cannot_hide_forgotten_own_action():
    tree = GameTree(
        HeroNode(
            "h0",
            "H0",
            (
                ("a", HeroNode("ha", "I", (("x", _leaf("tax", 0)),))),
                ("b", HeroNode("hb", "I", (("x", _leaf("tbx", 0)),))),
            ),
        )
    )
    attestation = _attestation(tree)
    histories = {
        player: {
            info_set: dict(by_node)
            for info_set, by_node in attestation.member_histories[player].items()
        }
        for player in PLAYERS
    }
    histories[HERO]["I"] = {
        "ha": RecallHistory((), ()),
        "hb": RecallHistory((), ()),
    }
    result = _run(tree, attestation=replace(attestation, member_histories=histories))
    assert result.status == DiagnosticStatus.UNSUPPORTED
    assert result.deviations == ()


def test_correct_nonempty_history_is_accepted_for_multistage_tree():
    tree = GameTree(
        HeroNode(
            "h0",
            "H0",
            (("a", HeroNode("h1", "H1", (("x", _leaf("t", 0)),))),),
        )
    )
    attestation = _attestation(tree)
    assert attestation.member_histories[HERO]["H1"]["h1"] == RecallHistory(
        (), ("a",), ("H0",)
    )
    result = _run(tree, attestation=attestation)
    assert result.status == DiagnosticStatus.PASS


def test_interval_crossing_epsilon_claim_is_indeterminate():
    tree = GameTree(_leaf("t", 0))
    bound = NumericErrorBound(F(0), F(0), F(1, 10), F(0), F(0), F(0), F(0), F(0), True)
    result = _run(tree, numeric_error_bound=bound, epsilon_claim=F(1, 20))
    assert result.maximum_lower == F(-1, 10)
    assert result.maximum_upper == F(1, 10)
    assert result.status == DiagnosticStatus.INDETERMINATE


def test_unestablished_residual_and_error_enclosure_is_indeterminate():
    tree = GameTree(_leaf("t", 0))
    bound = NumericErrorBound(F(0), F(0), F(0), F(0), F(0), F(0), F(0), F(0), False)
    result = _run(tree, numeric_error_bound=bound)
    assert result.status == DiagnosticStatus.INDETERMINATE
    assert result.deviations == ()


def _nuts_chop_game():
    call_fold = HeroNode(
        "hero_vs_shove",
        "H_vs_shove",
        (
            ("fold", TerminalNode("shove_fold", F(-1), F(1), F(0))),
            ("call", TerminalNode("shove_call", F(-2), F(-2), F(4))),
        ),
    )
    return GameTree(
        VillainNode(
            "villain",
            "V_action",
            (
                ("check", TerminalNode("check_check", F(-1, 20), F(-1, 20), F(1, 10))),
                ("shove", call_fold),
            ),
        )
    )


def _nuts_chop_inputs(delta):
    tree = _nuts_chop_game()
    monitoring = _monitoring(tree)
    trigger = next(
        signal
        for signal in monitoring.signal_alphabet
        if tuple(action.action for action in signal.action_trace) == ("shove", "call")
    )
    monitoring = _monitoring(tree, (trigger,))
    profile = _pure_profile(
        tree,
        c={(VILLAIN, "V_action"): "check", (HERO, "H_vs_shove"): "call"},
        p={(VILLAIN, "V_action"): "shove", (HERO, "H_vs_shove"): "fold"},
    )
    return tree, profile, monitoring, delta


def test_nuts_chop_negative_oracle_and_exact_boundary():
    oracle = nuts_chop_negative_oracle(F(13, 20))
    assert oracle["check_check_hero"] == oracle["check_check_villain"] == F(-1, 20)
    assert oracle["shove_fold_hero"] == -1
    assert oracle["shove_fold_villain"] == 1
    assert oracle["shove_call_hero"] == oracle["shove_call_villain"] == -2
    assert oracle["fold_minus_call_same_next_state"] == 1
    assert oracle["villain_gain"] == 0

    tree, profile, monitoring, delta = _nuts_chop_inputs(F(13, 20))
    tie = _run(tree, profile=profile, monitoring=monitoring, delta=delta, stage_payoff_bound=F(2))
    villain_shove = next(
        item
        for item in tie.deviations
        if item.player == VILLAIN
        and item.state == COOPERATE
        and dict(item.plan.actions)["V_action"] == "shove"
    )
    assert villain_shove.gain == 0

    tree, profile, monitoring, delta = _nuts_chop_inputs(F(2, 3))
    above = _run(tree, profile=profile, monitoring=monitoring, delta=delta, stage_payoff_bound=F(2))
    assert nuts_chop_negative_oracle(delta)["villain_gain"] > 0
    assert above.status == DiagnosticStatus.FAIL


def test_call_fold_with_different_next_states_includes_continuation_difference():
    tree = GameTree(
        HeroNode(
            "hero",
            "H1",
            (("call", _leaf("call", -2, -2, 4)), ("fold", _leaf("fold", -1, 1, 0))),
        )
    )
    monitoring = _monitoring(tree)
    fold_signal = next(signal for signal in monitoring.signal_alphabet if signal.action_trace[-1].action == "fold")
    monitoring = _monitoring(tree, (fold_signal,))
    profile = _pure_profile(tree, c={(HERO, "H1"): "call"}, p={(HERO, "H1"): "fold"})
    result = _run(tree, profile=profile, monitoring=monitoring, stage_payoff_bound=F(2))
    fold = next(
        item
        for item in result.deviations
        if item.player == HERO and item.state == COOPERATE and dict(item.plan.actions)["H1"] == "fold"
    )
    assert result.prescribed_values[(HERO, COOPERATE)] == -4
    assert result.prescribed_values[(HERO, PUNISH)] == -2
    assert fold.deviation_value == -2
    assert fold.gain == 2
    assert fold.gain != 1


def test_house_residual_is_not_used_as_player_gain():
    tree = GameTree(
        HeroNode(
            "h",
            "H1",
            (
                ("low_residual", TerminalNode("low", F(0), F(0), F(0))),
                ("high_residual", TerminalNode("high", F(0), F(-5), F(5))),
            ),
        )
    )
    result = _run(tree)
    hero = [item for item in result.deviations if item.player == HERO]
    assert all(item.gain == 0 for item in hero)


def test_unsupported_model_class_does_not_pass():
    tree = GameTree(_leaf("t", 0))
    result = _run(tree, model_attestation=_model_attestation(no_cross_period_correlation=False))
    assert result.status == DiagnosticStatus.UNSUPPORTED


def test_required_oracle_mismatch_is_indeterminate(monkeypatch):
    import repeated_poker.stage_plan_diagnostic as module

    tree = GameTree(_leaf("t", 0))
    monkeypatch.setattr(module, "_negative_oracle_consistent", lambda: False)
    result = _run(tree)
    assert result.status == DiagnosticStatus.INDETERMINATE


def test_public_claim_wording_avoids_disallowed_result_terms():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "repeated_poker"
        / "stage_plan_diagnostic.py"
    ).read_text(encoding="utf-8").lower()
    disallowed = (
        "certifi" + "cate",
        "nash " + "equilibrium",
        "subgame-perfect " + "equilibrium",
        "sequential " + "equilibrium",
        "solver-" + "grade",
        "proved " + "optimal",
    )
    assert all(term not in source for term in disallowed)
