"""Tests for STT SB-vs-BB push/fold scenario loading and building."""

import copy
import math

import pytest

from repeated_poker import (
    build_stt_pushfold_game,
    calculate_icm_equities,
    solve_exact_response,
    stt_pushfold_scenario_from_dict,
)
from repeated_poker.game import (
    HeroNode,
    TerminalNode,
    VillainNode,
    collect_hero_info_sets,
    collect_villain_info_sets,
    iter_nodes,
    iter_terminals,
    validate_tree,
)


def _single_pair_dict():
    return {
        "format_version": "stt_pushfold-1",
        "scenario_id": "stt_single_pair",
        "description": "single abstract STT push/fold matchup",
        "stacks": [50.0, 30.0, 20.0],
        "sb_index": 1,
        "bb_index": 2,
        "prizes": [50.0, 30.0, 20.0],
        "small_blind": 5.0,
        "big_blind": 10.0,
        "ante": 0.0,
        "hero_seat": "sb",
        "sb_range": [{"id": "sb", "weight": 1.0}],
        "bb_range": [{"id": "bb", "weight": 1.0}],
        "outcome_matrix": {
            "sb": {
                "bb": {
                    "sb_win": 1.0,
                    "bb_win": 0.0,
                    "chop": 0.0,
                }
            }
        },
        "baseline_sb_strategy": {
            "sb": {
                "shove": 1.0,
                "fold": 0.0,
            }
        },
        "candidates": {
            "shift_amounts": [0.25],
        },
        "repeated": {
            "horizon": 10,
            "discount": 1.0,
        },
    }


def _build(data=None):
    return build_stt_pushfold_game(stt_pushfold_scenario_from_dict(data or _single_pair_dict()))


def _terminal_by_id(build, node_id):
    for terminal in iter_terminals(build.tree.root):
        if terminal.node_id == node_id:
            return terminal
    raise AssertionError(f"terminal {node_id!r} not found")


def _payoff_for_stack_vector(data, stack_vector, hero_seat="sb"):
    base = calculate_icm_equities(data["stacks"], data["prizes"])
    after = calculate_icm_equities(stack_vector, data["prizes"])
    deltas = [post - pre for post, pre in zip(after, base)]
    sb_index = data["sb_index"]
    bb_index = data["bb_index"]
    residual = math.fsum(
        delta for index, delta in enumerate(deltas) if index not in (sb_index, bb_index)
    )
    if hero_seat == "sb":
        return (deltas[sb_index], deltas[bb_index], residual)
    return (deltas[bb_index], deltas[sb_index], residual)


def test_terminal_stack_vectors_conserve_chips_with_ante():
    data = _single_pair_dict()
    data["ante"] = 2.0
    data["stacks"] = [50.0, 35.0, 40.0]
    data["sb_index"] = 1
    data["bb_index"] = 2
    build = _build(data)

    starting_total = math.fsum(data["stacks"])
    for stack_vector in build.metadata["terminal_stack_vectors"].values():
        assert math.fsum(stack_vector) == pytest.approx(starting_total)


def test_all_terminals_conserve_prize_ev_and_relaxed_tree_validation_passes():
    build = _build()

    for terminal in iter_terminals(build.tree.root):
        assert (
            terminal.hero_ev + terminal.villain_ev + terminal.house_rake
        ) == pytest.approx(0.0, abs=1e-9)
    validate_tree(build.tree, allow_negative_residual=True)


def test_default_tree_validation_still_rejects_negative_residual():
    build = _build()

    with pytest.raises(ValueError, match="negative house_rake"):
        validate_tree(build.tree)


def test_fold_terminal_matches_hand_calculated_icm_oracle():
    build = _build()
    terminal = _terminal_by_id(build, "T_sb_fold::sb__bb")

    assert terminal.hero_ev == pytest.approx(-23 / 12)
    assert terminal.villain_ev == pytest.approx(83 / 42)
    assert terminal.house_rake == pytest.approx(-5 / 84)


def test_n2_winner_take_all_terminals_match_chip_delta_scale():
    data = _single_pair_dict()
    data["stacks"] = [100.0, 50.0]
    data["sb_index"] = 0
    data["bb_index"] = 1
    data["prizes"] = [1.0]
    data["small_blind"] = 5.0
    data["big_blind"] = 10.0
    build = _build(data)
    scale = data["prizes"][0] / math.fsum(data["stacks"])
    vectors = build.metadata["terminal_stack_vectors"]

    expected_by_terminal = {
        "T_sb_fold::sb__bb": vectors["sb_fold"],
        "T_shove_bb_fold::sb__bb": vectors["shove_bb_fold"],
        "T_call::sb__bb": vectors["call_sb_win"],
    }
    for terminal_id, stack_vector in expected_by_terminal.items():
        terminal = _terminal_by_id(build, terminal_id)
        assert terminal.hero_ev == pytest.approx(
            (stack_vector[data["sb_index"]] - data["stacks"][data["sb_index"]]) * scale
        )
        assert terminal.villain_ev == pytest.approx(
            (stack_vector[data["bb_index"]] - data["stacks"][data["bb_index"]]) * scale
        )
        assert terminal.house_rake == pytest.approx(0.0)


def test_showdown_terminal_is_linear_mix_of_outcome_stack_deltas():
    data = _single_pair_dict()
    data["outcome_matrix"]["sb"]["bb"] = {
        "sb_win": 0.2,
        "bb_win": 0.3,
        "chop": 0.5,
    }
    build = _build(data)
    terminal = _terminal_by_id(build, "T_call::sb__bb")
    vectors = build.metadata["terminal_stack_vectors"]
    expected_components = [
        (0.2, _payoff_for_stack_vector(data, vectors["call_sb_win"])),
        (0.3, _payoff_for_stack_vector(data, vectors["call_bb_win"])),
        (0.5, _payoff_for_stack_vector(data, vectors["call_chop"])),
    ]
    expected = tuple(
        math.fsum(weight * payoff[index] for weight, payoff in expected_components)
        for index in range(3)
    )

    assert (terminal.hero_ev, terminal.villain_ev, terminal.house_rake) == pytest.approx(
        expected
    )


def test_bb_best_response_calls_when_call_wins_every_showdown():
    data = _single_pair_dict()
    data["stacks"] = [100.0, 100.0, 100.0]
    data["sb_index"] = 0
    data["bb_index"] = 1
    data["small_blind"] = 5.0
    data["big_blind"] = 10.0
    data["outcome_matrix"]["sb"]["bb"] = {
        "sb_win": 0.0,
        "bb_win": 1.0,
        "chop": 0.0,
    }
    build = _build(data)
    response = solve_exact_response(
        build.tree,
        build.baseline_hero_strategy,
        allow_negative_residual=True,
    )

    assert response.best_response_strategies[0]["BB_vs_shove:bb"] == "call"
    assert build.baseline_villain_strategy.probabilities["BB_vs_shove:bb"] == {
        "call": 1.0,
        "fold": 0.0,
    }


def test_hero_seat_bb_builds_bb_as_hero_and_sb_as_villain():
    data = _single_pair_dict()
    data["hero_seat"] = "bb"
    data["baseline_bb_strategy"] = {"bb": {"call": 1.0, "fold": 0.0}}
    build = _build(data)

    assert set(collect_hero_info_sets(build.tree)) == {"BB_vs_shove:bb"}
    assert set(collect_villain_info_sets(build.tree)) == {"SB:sb"}
    assert any(isinstance(node, HeroNode) and node.info_set == "BB_vs_shove:bb" for node in iter_nodes(build.tree.root))
    assert any(isinstance(node, VillainNode) and node.info_set == "SB:sb" for node in iter_nodes(build.tree.root))


def test_explicit_villain_baseline_source_is_recorded():
    data = _single_pair_dict()
    data["baseline_bb_strategy"] = {"bb": {"call": 0.25, "fold": 0.75}}
    build = _build(data)

    assert build.baseline_villain_source == "explicit"
    assert build.metadata["baseline_villain_source"] == "explicit"
    assert build.baseline_villain_strategy.probabilities["BB_vs_shove:bb"] == {
        "call": 0.25,
        "fold": 0.75,
    }


def test_terminal_reveals_show_buckets_only_on_call():
    build = _build()

    assert build.terminal_reveals["T_call::sb__bb"] == ("sb", "bb")
    assert build.terminal_reveals["T_sb_fold::sb__bb"] is None
    assert build.terminal_reveals["T_shove_bb_fold::sb__bb"] is None


def test_scalar_sb_win_probability_matrix_is_accepted():
    data = _single_pair_dict()
    del data["outcome_matrix"]
    data["sb_win_probability_matrix"] = {"sb": {"bb": 0.25}}
    scenario = stt_pushfold_scenario_from_dict(data)
    build = build_stt_pushfold_game(scenario)

    outcome = scenario.outcome_matrix["sb"]["bb"]
    assert outcome.sb_win == 0.25
    assert outcome.bb_win == 0.75
    assert outcome.chop == 0.0
    assert build.metadata["outcome_input_type"] == "sb_win_probability_matrix"


def test_parse_does_not_mutate_input_dict():
    data = _single_pair_dict()
    before = copy.deepcopy(data)

    stt_pushfold_scenario_from_dict(data)

    assert data == before


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda data: data.__setitem__("format_version", "1"), "unsupported"),
        (lambda data: data.__setitem__("sb_index", True), "sb_index"),
        (lambda data: data.__setitem__("hero_seat", "button"), "hero_seat"),
        (lambda data: data["stacks"].__setitem__(1, 4.0), "SB stack"),
        (
            lambda data: data["outcome_matrix"]["sb"]["bb"].__setitem__("chop", 0.5),
            "probabilities sum",
        ),
        (
            lambda data: data["outcome_matrix"]["sb"].pop("bb"),
            "missing BB ids",
        ),
        (
            lambda data: data.__setitem__("equity_matrix", {"sb": {"bb": 1.0}}),
            "equity_matrix is not supported",
        ),
        (
            lambda data: data.pop("baseline_sb_strategy"),
            "baseline_sb_strategy is required",
        ),
    ],
)
def test_invalid_inputs_are_rejected(mutate, match):
    data = _single_pair_dict()
    mutate(data)

    with pytest.raises(ValueError, match=match):
        stt_pushfold_scenario_from_dict(data)
