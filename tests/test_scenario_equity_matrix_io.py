"""Tests for the abstract Hero/Villain equity matrix scenario input (v1)."""

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from repeated_poker import (
    build_river_steal_game_from_scenario,
    load_river_scenario_json,
    make_equity_showdown_terminal,
    make_showdown_terminal,
    river_scenario_from_dict,
    run_river_scenario_analysis,
)
from repeated_poker.game import ChanceNode, iter_terminals

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / "examples" / "scenarios" / "range_equity_steal_bet98.json"
_SCRIPT = _ROOT / "scripts" / "run_river_scenario_analysis.py"


def _sample_dict() -> dict:
    return json.loads(_SAMPLE.read_text(encoding="utf-8"))


def _build_sample():
    return build_river_steal_game_from_scenario(load_river_scenario_json(_SAMPLE))


# ---------------------------------------------------------------------------
# make_equity_showdown_terminal
# ---------------------------------------------------------------------------


def test_equity_terminal_hand_calculation():
    # pot=100, 50/50 invested, equity 0.7, rake 5% capped at 4.
    terminal = make_equity_showdown_terminal("t", 100.0, 50.0, 50.0, 0.7, 0.05, 4.0)
    assert terminal.house_rake == pytest.approx(4.0)  # min(5, 4)
    assert terminal.hero_ev == pytest.approx(17.2)  # 96 * 0.7 - 50
    assert terminal.villain_ev == pytest.approx(-21.2)  # 96 * 0.3 - 50
    assert terminal.hero_ev + terminal.villain_ev + terminal.house_rake == pytest.approx(0.0)


@pytest.mark.parametrize("equity, result", [(1.0, "hero"), (0.5, "chop"), (0.0, "villain")])
def test_equity_extremes_match_discrete_results(equity, result):
    pot, hero_inv, villain_inv = 100.0, 40.0, 60.0
    equity_terminal = make_equity_showdown_terminal(
        "t_eq", pot, hero_inv, villain_inv, equity, 0.05, 4.0
    )
    discrete_terminal = make_showdown_terminal(
        "t_d", pot, hero_inv, villain_inv, result, 0.05, 4.0
    )
    assert equity_terminal.hero_ev == pytest.approx(discrete_terminal.hero_ev)
    assert equity_terminal.villain_ev == pytest.approx(discrete_terminal.villain_ev)
    assert equity_terminal.house_rake == pytest.approx(discrete_terminal.house_rake)


@pytest.mark.parametrize("bad_equity", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_equity_is_rejected(bad_equity):
    with pytest.raises(ValueError):
        make_equity_showdown_terminal("t", 100.0, 50.0, 50.0, bad_equity, 0.05, 4.0)


def test_equity_terminal_rejects_inconsistent_pot():
    with pytest.raises(ValueError, match="pot"):
        make_equity_showdown_terminal("t", 100.0, 50.0, 49.0, 0.7, 0.05, 4.0)


# ---------------------------------------------------------------------------
# Parsing / building
# ---------------------------------------------------------------------------


def test_sample_equity_scenario_loads():
    scenario = load_river_scenario_json(_SAMPLE)
    assert scenario.is_matrix_mode
    assert scenario.equity_matrix is not None
    assert scenario.showdown_matrix is None
    assert scenario.equity_matrix["hero_medium"]["villain_weak"] == 0.65
    assert scenario.equity_matrix["hero_strong"]["villain_strong"] == 0.55


def test_equity_metadata_matrix_type():
    build = _build_sample()
    assert build.metadata["mode"] == "range_matrix"
    assert build.metadata["matrix_type"] == "equity"


def test_equity_root_is_chance_node():
    build = _build_sample()
    assert isinstance(build.tree.root, ChanceNode)
    total = sum(p for p, _ in build.tree.root.children)
    assert total == pytest.approx(1.0)


def test_equity_terminal_evs_are_fractional_non_chop():
    build = _build_sample()
    terminals = {t.node_id: t for t in iter_terminals(build.tree.root)}
    # hero_medium vs villain_weak, equity 0.65 at bet-call: pot 198 (= 1 + 1 +
    # 2*98), hero invested 99, rake min(9.9, 4) = 4, awarded 194, so the Hero EV
    # is 194 * 0.65 - 99 = 27.1.
    bet_call = terminals["T_bet_call::hero_medium__villain_weak"]
    assert bet_call.hero_ev == pytest.approx(194.0 * 0.65 - 99.0)
    # The value is neither a chop nor a clean win/loss, so it is fractional.
    assert not math.isclose(bet_call.hero_ev, round(bet_call.hero_ev))
    assert bet_call.hero_ev + bet_call.villain_ev + bet_call.house_rake == pytest.approx(0.0)


def test_run_river_scenario_analysis_succeeds_for_equity():
    result = run_river_scenario_analysis(_SAMPLE)
    assert result.scenario_id == "range_equity_steal_bet98"
    assert result.pipeline_result.generated_candidates
    assert result.pipeline_result.analysis_report.rows
    assert "Candidate Analysis Summary" in result.markdown_summary


def test_run_does_not_mutate_equity_scenario_dict():
    data = _sample_dict()
    before = copy.deepcopy(data)
    scenario = river_scenario_from_dict(data)
    run_river_scenario_analysis(scenario)
    assert data == before


# ---------------------------------------------------------------------------
# Validation rejections
# ---------------------------------------------------------------------------


def test_both_matrices_present_is_rejected():
    data = _sample_dict()
    data["showdown_matrix"] = {
        "hero_medium": {"villain_weak": "chop", "villain_strong": "villain"},
        "hero_strong": {"villain_weak": "hero", "villain_strong": "chop"},
    }
    with pytest.raises(ValueError, match="exactly one of showdown_matrix or equity_matrix"):
        river_scenario_from_dict(data)


def test_villain_range_without_any_matrix_is_rejected():
    data = _sample_dict()
    del data["equity_matrix"]
    with pytest.raises(ValueError, match="villain_range requires a showdown_matrix or an equity_matrix"):
        river_scenario_from_dict(data)


def test_incomplete_equity_matrix_is_rejected():
    data = _sample_dict()
    del data["equity_matrix"]["hero_medium"]["villain_strong"]
    with pytest.raises(ValueError, match="missing villain ids"):
        river_scenario_from_dict(data)


def test_extra_equity_hero_key_is_rejected():
    data = _sample_dict()
    data["equity_matrix"]["hero_ghost"] = {"villain_weak": 0.5, "villain_strong": 0.5}
    with pytest.raises(ValueError, match="unknown hero ids"):
        river_scenario_from_dict(data)


def test_extra_equity_villain_key_is_rejected():
    data = _sample_dict()
    data["equity_matrix"]["hero_medium"]["villain_ghost"] = 0.5
    with pytest.raises(ValueError, match="unknown villain ids"):
        river_scenario_from_dict(data)


@pytest.mark.parametrize("bad_value", [-0.1, 1.5, "chop"])
def test_invalid_equity_value_is_rejected(bad_value):
    data = _sample_dict()
    data["equity_matrix"]["hero_medium"]["villain_weak"] = bad_value
    with pytest.raises(ValueError, match="equity_matrix"):
        river_scenario_from_dict(data)


def test_equity_mode_with_top_level_showdown_is_rejected():
    data = _sample_dict()
    data["showdown"] = "chop"
    with pytest.raises(ValueError, match="cannot be combined"):
        river_scenario_from_dict(data)


# ---------------------------------------------------------------------------
# CLI script
# ---------------------------------------------------------------------------


def test_cli_analysis_script_runs_for_equity():
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), str(_SAMPLE)],
        capture_output=True,
        text=True,
        check=True,
    )
    stdout = completed.stdout
    assert "range_equity_steal_bet98" in stdout
    assert "Candidate Analysis Summary" in stdout
