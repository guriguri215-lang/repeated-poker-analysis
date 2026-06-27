"""Tests for the river betting-tree scenario input (betting-tree v1)."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from repeated_poker import (
    build_river_steal_game_from_scenario,
    generate_shift_candidates,
    load_river_scenario_json,
    river_scenario_from_dict,
    run_river_scenario_analysis,
    solve_exact_response,
)
from repeated_poker.game import (
    ChanceNode,
    collect_hero_info_sets,
    collect_villain_info_sets,
    iter_terminals,
)

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / "examples" / "scenarios" / "range_equity_betting_tree_bet98.json"
_SCRIPT = _ROOT / "scripts" / "run_river_scenario_analysis.py"


def _sample_dict() -> dict:
    return json.loads(_SAMPLE.read_text(encoding="utf-8"))


def _build_sample():
    return build_river_steal_game_from_scenario(load_river_scenario_json(_SAMPLE))


def _terminals(tree):
    return {t.node_id: t for t in iter_terminals(tree.root)}


# ---------------------------------------------------------------------------
# Loading / metadata
# ---------------------------------------------------------------------------


def test_sample_betting_tree_loads():
    scenario = load_river_scenario_json(_SAMPLE)
    assert scenario.is_betting_tree_mode
    assert scenario.is_matrix_mode
    assert scenario.betting_tree.oop_bet_size == 98.0
    assert scenario.betting_tree.ip_bet_after_check_size == 98.0
    assert scenario.betting_tree.ip_raise_size == 196.0


def test_betting_tree_metadata():
    build = _build_sample()
    assert build.metadata["mode"] == "range_matrix"
    assert build.metadata["matrix_type"] == "equity"
    assert build.metadata["betting_tree"] == {
        "oop_bet_size": 98.0,
        "ip_bet_after_check_size": 98.0,
        "ip_raise_size": 196.0,
    }


# ---------------------------------------------------------------------------
# Information sets
# ---------------------------------------------------------------------------


def test_root_is_chance_node():
    build = _build_sample()
    assert isinstance(build.tree.root, ChanceNode)
    assert sum(p for p, _ in build.tree.root.children) == pytest.approx(1.0)


def test_hero_info_sets():
    build = _build_sample()
    assert set(collect_hero_info_sets(build.tree)) == {
        "IP_after_OOP_check::hero_medium",
        "IP_after_OOP_check::hero_strong",
        "IP_vs_OOP_bet::hero_medium",
        "IP_vs_OOP_bet::hero_strong",
    }


def test_villain_info_sets():
    build = _build_sample()
    assert set(collect_villain_info_sets(build.tree)) == {
        "OOP_first::villain_weak",
        "OOP_first::villain_strong",
        "OOP_vs_IP_bet::villain_weak",
        "OOP_vs_IP_bet::villain_strong",
        "OOP_vs_IP_raise::villain_weak",
        "OOP_vs_IP_raise::villain_strong",
    }


def test_baseline_hero_strategy_covers_all_hero_info_sets():
    build = _build_sample()
    assert set(build.baseline_hero_strategy.probabilities) == set(
        collect_hero_info_sets(build.tree)
    )


# ---------------------------------------------------------------------------
# Terminal accounting
# ---------------------------------------------------------------------------


def test_check_check_showdown_equity_terminal():
    # hero_medium vs villain_weak, equity 0.65: pot = 1 + 1 = 2, rake
    # min(0.1, 4) = 0.1, awarded 1.9, Hero EV = 1.9 * 0.65 - 1 = 0.235.
    terminals = _terminals(_build_sample().tree)
    term = terminals["T_check_check::hero_medium__villain_weak"]
    assert term.hero_ev == pytest.approx(1.9 * 0.65 - 1.0)
    assert term.hero_ev + term.villain_ev + term.house_rake == pytest.approx(0.0)


def test_bet_raise_call_uses_raise_total_investment():
    # bet-raise-call: both invest initial + ip_raise_size = 1 + 196 = 197, pot
    # 394, rake min(19.7, 4) = 4, awarded 390, Hero EV = 390 * 0.65 - 197.
    terminals = _terminals(_build_sample().tree)
    term = terminals["T_bet_raise_call::hero_medium__villain_weak"]
    assert term.hero_ev == pytest.approx(390.0 * 0.65 - 197.0)


def test_ip_raise_oop_fold_terminal_uses_villain_initial_plus_oop_bet():
    # bet -> IP raise -> OOP fold: Hero wins villain_initial + oop_bet = 1 + 98.
    terminals = _terminals(_build_sample().tree)
    term = terminals["T_bet_raise_fold::hero_medium__villain_weak"]
    assert term.hero_ev == pytest.approx(99.0)
    assert term.villain_ev == pytest.approx(-99.0)
    assert term.house_rake == pytest.approx(0.0)


def test_check_bet_fold_terminal_uses_villain_initial():
    # check -> IP bet -> OOP fold: Hero wins villain_initial = 1.
    terminals = _terminals(_build_sample().tree)
    term = terminals["T_check_bet_fold::hero_medium__villain_weak"]
    assert term.hero_ev == pytest.approx(1.0)
    assert term.villain_ev == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Pipeline / candidate generation
# ---------------------------------------------------------------------------


def test_solve_exact_response_works():
    build = _build_sample()
    response = solve_exact_response(build.tree, build.baseline_hero_strategy)
    assert set(response.best_response_strategies[0]) == set(
        collect_villain_info_sets(build.tree)
    )


def test_generate_shift_candidates_covers_both_hero_decision_points():
    build = _build_sample()
    candidates = generate_shift_candidates(
        build.tree, build.baseline_hero_strategy, build.shift_amounts
    )
    shifted = {c.info_set for c in candidates}
    assert any(s.startswith("IP_after_OOP_check::") for s in shifted)
    assert any(s.startswith("IP_vs_OOP_bet::") for s in shifted)


def test_run_river_scenario_analysis_succeeds():
    result = run_river_scenario_analysis(_SAMPLE)
    assert result.scenario_id == "range_equity_betting_tree_bet98"
    assert result.pipeline_result.generated_candidates
    assert result.pipeline_result.analysis_report.rows
    assert "Candidate Analysis Summary" in result.markdown_summary


def test_run_does_not_mutate_scenario_dict():
    data = _sample_dict()
    before = copy.deepcopy(data)
    scenario = river_scenario_from_dict(data)
    run_river_scenario_analysis(scenario)
    assert data == before


# ---------------------------------------------------------------------------
# Validation rejections
# ---------------------------------------------------------------------------


def test_missing_oop_bet_size_is_rejected():
    data = _sample_dict()
    del data["betting_tree"]["oop_bet_size"]
    with pytest.raises(ValueError, match="oop_bet_size"):
        river_scenario_from_dict(data)


def test_negative_bet_size_is_rejected():
    data = _sample_dict()
    data["betting_tree"]["ip_bet_after_check_size"] = -1.0
    with pytest.raises(ValueError, match="ip_bet_after_check_size"):
        river_scenario_from_dict(data)


def test_raise_size_not_greater_than_oop_bet_is_rejected():
    data = _sample_dict()
    data["betting_tree"]["ip_raise_size"] = 98.0  # == oop_bet_size
    with pytest.raises(ValueError, match="ip_raise_size must be greater"):
        river_scenario_from_dict(data)


def test_betting_tree_without_baseline_strategies_is_rejected():
    data = _sample_dict()
    del data["hero_range"][0]["baseline_strategies"]
    with pytest.raises(ValueError, match="baseline_strategies"):
        river_scenario_from_dict(data)


def test_baseline_strategies_missing_after_oop_check_is_rejected():
    data = _sample_dict()
    del data["hero_range"][0]["baseline_strategies"]["after_oop_check"]
    with pytest.raises(ValueError, match="after_oop_check"):
        river_scenario_from_dict(data)


def test_baseline_strategies_missing_vs_oop_bet_is_rejected():
    data = _sample_dict()
    del data["hero_range"][0]["baseline_strategies"]["vs_oop_bet"]
    with pytest.raises(ValueError, match="vs_oop_bet"):
        river_scenario_from_dict(data)


def test_invalid_baseline_strategy_probabilities_are_rejected():
    data = _sample_dict()
    data["hero_range"][0]["baseline_strategies"]["vs_oop_bet"] = {
        "call": 0.4,
        "fold": 0.4,
        "raise": 0.0,
    }
    with pytest.raises(ValueError, match="sum"):
        river_scenario_from_dict(data)


def test_old_baseline_strategy_with_betting_tree_is_rejected():
    data = _sample_dict()
    data["hero_range"][0]["baseline_strategy"] = {"call": 0.0, "fold": 1.0}
    with pytest.raises(ValueError, match="baseline_strategies, not the old baseline_strategy"):
        river_scenario_from_dict(data)


def test_betting_tree_requires_matrix_mode():
    data = _sample_dict()
    del data["villain_range"]
    del data["equity_matrix"]
    with pytest.raises(ValueError, match="betting_tree requires matrix mode"):
        river_scenario_from_dict(data)


# ---------------------------------------------------------------------------
# CLI script
# ---------------------------------------------------------------------------


def test_cli_analysis_script_runs_for_betting_tree():
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), str(_SAMPLE)],
        capture_output=True,
        text=True,
        check=True,
    )
    stdout = completed.stdout
    assert "range_equity_betting_tree_bet98" in stdout
    assert "Candidate Analysis Summary" in stdout
