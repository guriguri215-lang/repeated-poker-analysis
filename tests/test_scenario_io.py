"""Tests for the JSON river-scenario input layer."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from repeated_poker import (
    HeroStrategy,
    build_river_steal_game_from_scenario,
    calculate_adaptation_deadline,
    iter_terminals,
    load_river_scenario_json,
    river_scenario_from_dict,
    solve_exact_response,
)

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / "examples" / "scenarios" / "nuts_chop_steal_bet98.json"
_SCRIPT = _ROOT / "scripts" / "run_river_scenario.py"


def _sample_dict() -> dict:
    return json.loads(_SAMPLE.read_text(encoding="utf-8"))


def _build_sample():
    return build_river_steal_game_from_scenario(load_river_scenario_json(_SAMPLE))


def _terminals(tree):
    return {t.node_id: t for t in iter_terminals(tree.root)}


def test_sample_scenario_loads():
    scenario = load_river_scenario_json(_SAMPLE)
    assert scenario.scenario_id == "nuts_chop_steal_bet98"
    assert scenario.bet_size == 98.0
    assert scenario.rake.rate == 0.05
    assert scenario.rake.cap == 4.0
    assert scenario.showdown == "chop"
    assert scenario.shift_amounts == [1.0]
    assert scenario.repeated.horizons == [10, 20, 50, 100]
    assert scenario.repeated.discount == 1.0


def test_sample_tree_terminal_evs_match_hand_calculation():
    terminals = _terminals(_build_sample().tree)
    check_check = terminals["T_check_check"]
    bet_fold = terminals["T_bet_fold"]
    bet_call = terminals["T_bet_call"]

    assert check_check.hero_ev == pytest.approx(-0.05)
    assert check_check.villain_ev == pytest.approx(-0.05)
    assert bet_fold.hero_ev == pytest.approx(-1.0)
    assert bet_fold.villain_ev == pytest.approx(1.0)
    assert bet_call.hero_ev == pytest.approx(-2.0)
    assert bet_call.villain_ev == pytest.approx(-2.0)


def test_baseline_hero_strategy_is_fold():
    build = _build_sample()
    dist = build.baseline_hero_strategy.probabilities["IP_vs_bet"]
    assert dist["fold"] == pytest.approx(1.0)
    assert dist["call"] == pytest.approx(0.0)


def test_baseline_villain_best_response_is_bet():
    build = _build_sample()
    result = solve_exact_response(build.tree, build.baseline_hero_strategy)
    assert result.best_response_strategies == [{"OOP_river": "bet"}]
    # The build result records the same single-hand baseline as a strategy.
    villain_dist = build.baseline_villain_strategy.probabilities["OOP_river"]
    assert villain_dist["bet"] == pytest.approx(1.0)
    assert villain_dist["check"] == pytest.approx(0.0)


def test_locked_call_makes_oop_check():
    build = _build_sample()
    locked_call = HeroStrategy({"IP_vs_bet": {"call": 1.0, "fold": 0.0}})
    result = solve_exact_response(build.tree, locked_call)
    assert result.best_response_strategies == [{"OOP_river": "check"}]


@pytest.mark.parametrize(
    "horizon, expected_t_deadline",
    [(10, 5), (20, 10), (50, 25), (100, 49)],
)
def test_t_deadline_matches_for_horizons(horizon, expected_t_deadline):
    result = calculate_adaptation_deadline(
        baseline_hero_ev=-1.0,
        pre_adaptation_hero_ev=-2.0,
        post_adaptation_hero_ev=-0.05,
        horizon=horizon,
        discount=1.0,
    )
    assert result.t_deadline == expected_t_deadline


def test_build_does_not_mutate_scenario_strategy():
    data = _sample_dict()
    before = copy.deepcopy(data["baseline_hero_strategy"])
    scenario = river_scenario_from_dict(data)
    build_river_steal_game_from_scenario(scenario)
    assert data["baseline_hero_strategy"] == before


# ---------------------------------------------------------------------------
# Validation rejections
# ---------------------------------------------------------------------------


def _valid_dict() -> dict:
    return _sample_dict()


def test_missing_scenario_id_is_rejected():
    data = _valid_dict()
    del data["scenario_id"]
    with pytest.raises(ValueError, match="scenario_id"):
        river_scenario_from_dict(data)


def test_invalid_showdown_is_rejected():
    data = _valid_dict()
    data["showdown"] = "split"
    with pytest.raises(ValueError, match="showdown"):
        river_scenario_from_dict(data)


def test_negative_bet_size_is_rejected():
    data = _valid_dict()
    data["bet_size"] = -1.0
    with pytest.raises(ValueError, match="bet_size"):
        river_scenario_from_dict(data)


def test_probabilities_not_summing_to_one_are_rejected():
    data = _valid_dict()
    data["baseline_hero_strategy"]["IP_vs_bet"] = {"call": 0.4, "fold": 0.4}
    with pytest.raises(ValueError, match="sum"):
        river_scenario_from_dict(data)


def test_missing_ip_vs_bet_is_rejected():
    data = _valid_dict()
    data["baseline_hero_strategy"] = {"WRONG": {"call": 1.0, "fold": 0.0}}
    with pytest.raises(ValueError, match="IP_vs_bet"):
        river_scenario_from_dict(data)


def test_invalid_horizon_is_rejected():
    data = _valid_dict()
    data["repeated"]["horizons"] = [10, 0]
    with pytest.raises(ValueError, match="horizons"):
        river_scenario_from_dict(data)


def test_invalid_discount_is_rejected():
    data = _valid_dict()
    data["repeated"]["discount"] = 1.5
    with pytest.raises(ValueError, match="discount"):
        river_scenario_from_dict(data)


def test_invalid_rake_rate_is_rejected():
    data = _valid_dict()
    data["rake"]["rate"] = 1.5
    with pytest.raises(ValueError, match="rake.rate"):
        river_scenario_from_dict(data)


def test_negative_initial_commitment_is_rejected():
    data = _valid_dict()
    data["initial_commitment"]["hero"] = -1.0
    with pytest.raises(ValueError, match="initial_commitment"):
        river_scenario_from_dict(data)


def test_non_positive_shift_amount_is_rejected():
    data = _valid_dict()
    data["candidate_generation"]["shift_amounts"] = [0.0]
    with pytest.raises(ValueError, match="shift_amounts"):
        river_scenario_from_dict(data)


# ---------------------------------------------------------------------------
# CLI script
# ---------------------------------------------------------------------------


def test_cli_script_reports_expected_lines():
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), str(_SAMPLE)],
        capture_output=True,
        text=True,
        check=True,
    )
    stdout = completed.stdout
    for fragment in (
        "nuts_chop_steal_bet98",
        "OOP bet / IP fold",
        "Locked-call",
        "OOP check",
        "T_deadline",
        "N=10",
        "N=100",
        "49",
        "-0.0500",
        "-1.0000",
        "-2.0000",
    ):
        assert fragment in stdout, f"missing {fragment!r} in script output:\n{stdout}"
