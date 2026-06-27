"""Tests for the abstract Hero/Villain range matrix scenario input (v1)."""

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
)

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / "examples" / "scenarios" / "range_matrix_steal_bet98.json"
_SCRIPT = _ROOT / "scripts" / "run_river_scenario_analysis.py"


def _sample_dict() -> dict:
    return json.loads(_SAMPLE.read_text(encoding="utf-8"))


def _build_sample():
    return build_river_steal_game_from_scenario(load_river_scenario_json(_SAMPLE))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_sample_matrix_scenario_loads():
    scenario = load_river_scenario_json(_SAMPLE)
    assert scenario.is_matrix_mode
    assert scenario.is_range_mode  # range fields are present too
    assert scenario.scenario_id == "range_matrix_steal_bet98"
    assert [h.hand_id for h in scenario.hero_range.hands] == ["hero_chop", "hero_strong"]
    assert [h.hand_id for h in scenario.villain_range.hands] == [
        "villain_chop",
        "villain_strong",
    ]
    assert all(h.showdown is None for h in scenario.hero_range.hands)
    assert scenario.showdown_matrix["hero_chop"]["villain_strong"] == "villain"
    assert scenario.showdown_matrix["hero_strong"]["villain_chop"] == "hero"


# ---------------------------------------------------------------------------
# Game construction
# ---------------------------------------------------------------------------


def test_matrix_metadata_mode():
    build = _build_sample()
    assert build.metadata["mode"] == "range_matrix"
    assert [b["hand_id"] for b in build.metadata["hero_buckets"]] == [
        "hero_chop",
        "hero_strong",
    ]
    assert [b["hand_id"] for b in build.metadata["villain_buckets"]] == [
        "villain_chop",
        "villain_strong",
    ]


def test_matrix_root_is_chance_node():
    build = _build_sample()
    assert isinstance(build.tree.root, ChanceNode)


def test_matrix_chance_probabilities_sum_to_one():
    build = _build_sample()
    total = sum(p for p, _ in build.tree.root.children)
    assert total == pytest.approx(1.0)


def test_matrix_pair_probabilities_are_weight_products():
    build = _build_sample()
    probs = sorted(p for p, _ in build.tree.root.children)
    # 0.8*0.7, 0.8*0.3, 0.2*0.7, 0.2*0.3 = 0.56, 0.24, 0.14, 0.06
    assert probs == pytest.approx([0.06, 0.14, 0.24, 0.56])


def test_villain_info_sets_are_per_villain_bucket():
    build = _build_sample()
    assert set(collect_villain_info_sets(build.tree)) == {
        "OOP_river::villain_chop",
        "OOP_river::villain_strong",
    }


def test_same_villain_id_shares_info_set_across_hero_branches():
    # Each villain info set must appear under both hero branches: 2 hero buckets
    # share one villain info set, so the tree has exactly 2 distinct villain
    # info sets even though there are 4 villain nodes.
    build = _build_sample()
    from repeated_poker.game import VillainNode, iter_nodes

    villain_nodes = [n for n in iter_nodes(build.tree.root) if isinstance(n, VillainNode)]
    assert len(villain_nodes) == 4
    assert len({n.info_set for n in villain_nodes}) == 2


def test_hero_info_sets_are_per_hero_bucket():
    build = _build_sample()
    assert set(collect_hero_info_sets(build.tree)) == {
        "IP_vs_bet::hero_chop",
        "IP_vs_bet::hero_strong",
    }


def test_same_hero_id_shares_info_set_across_villain_branches():
    build = _build_sample()
    from repeated_poker.game import HeroNode, iter_nodes

    hero_nodes = [n for n in iter_nodes(build.tree.root) if isinstance(n, HeroNode)]
    assert len(hero_nodes) == 4
    assert len({n.info_set for n in hero_nodes}) == 2


def test_baseline_hero_strategy_has_one_info_set_per_hero_bucket():
    build = _build_sample()
    probs = build.baseline_hero_strategy.probabilities
    assert set(probs) == {"IP_vs_bet::hero_chop", "IP_vs_bet::hero_strong"}
    assert probs["IP_vs_bet::hero_chop"] == {"call": 0.0, "fold": 1.0}
    assert probs["IP_vs_bet::hero_strong"] == {"call": 1.0, "fold": 0.0}


def test_solve_exact_response_handles_multiple_villain_info_sets():
    build = _build_sample()
    response = solve_exact_response(build.tree, build.baseline_hero_strategy)
    chosen = response.best_response_strategies[0]
    assert set(chosen) == {"OOP_river::villain_chop", "OOP_river::villain_strong"}
    # The build records a pure Villain strategy over both info sets.
    assert set(build.baseline_villain_strategy.probabilities) == set(chosen)


def test_generate_shift_candidates_produces_multiple_candidates():
    build = _build_sample()
    candidates = generate_shift_candidates(
        build.tree, build.baseline_hero_strategy, build.shift_amounts
    )
    # 2 hero info sets x 3 shift amounts = 6 candidates.
    assert len(candidates) == 6


# ---------------------------------------------------------------------------
# Pipeline / runner integration
# ---------------------------------------------------------------------------


def test_run_river_scenario_analysis_succeeds_for_matrix():
    result = run_river_scenario_analysis(_SAMPLE)
    assert result.scenario_id == "range_matrix_steal_bet98"
    assert result.pipeline_result.generated_candidates
    assert result.pipeline_result.analysis_report.rows
    assert "Candidate Analysis Summary" in result.markdown_summary


def test_run_does_not_mutate_matrix_scenario_dict():
    data = _sample_dict()
    before = copy.deepcopy(data)
    scenario = river_scenario_from_dict(data)
    run_river_scenario_analysis(scenario)
    assert data == before


# ---------------------------------------------------------------------------
# Validation rejections
# ---------------------------------------------------------------------------


def test_villain_range_without_matrix_is_rejected():
    data = _sample_dict()
    del data["showdown_matrix"]
    with pytest.raises(ValueError, match="villain_range requires a showdown_matrix"):
        river_scenario_from_dict(data)


def test_matrix_without_villain_range_is_rejected():
    data = _sample_dict()
    del data["villain_range"]
    with pytest.raises(ValueError, match="requires villain_range"):
        river_scenario_from_dict(data)


def test_matrix_mode_without_hero_range_is_rejected():
    data = _sample_dict()
    del data["hero_range"]
    with pytest.raises(ValueError, match="matrix mode requires hero_range"):
        river_scenario_from_dict(data)


def test_incomplete_matrix_is_rejected():
    data = _sample_dict()
    del data["showdown_matrix"]["hero_chop"]["villain_strong"]
    with pytest.raises(ValueError, match="missing villain ids"):
        river_scenario_from_dict(data)


def test_extra_matrix_hero_key_is_rejected():
    data = _sample_dict()
    data["showdown_matrix"]["hero_ghost"] = {"villain_chop": "chop", "villain_strong": "chop"}
    with pytest.raises(ValueError, match="unknown hero ids"):
        river_scenario_from_dict(data)


def test_extra_matrix_villain_key_is_rejected():
    data = _sample_dict()
    data["showdown_matrix"]["hero_chop"]["villain_ghost"] = "chop"
    with pytest.raises(ValueError, match="unknown villain ids"):
        river_scenario_from_dict(data)


def test_invalid_matrix_result_is_rejected():
    data = _sample_dict()
    data["showdown_matrix"]["hero_chop"]["villain_chop"] = "split"
    with pytest.raises(ValueError, match="showdown_matrix"):
        river_scenario_from_dict(data)


def test_duplicate_villain_hand_id_is_rejected():
    data = _sample_dict()
    data["villain_range"][1]["hand_id"] = "villain_chop"
    with pytest.raises(ValueError, match="duplicate hand_id"):
        river_scenario_from_dict(data)


def test_villain_weights_not_summing_to_one_are_rejected():
    data = _sample_dict()
    data["villain_range"][0]["weight"] = 0.5  # 0.5 + 0.3 != 1
    with pytest.raises(ValueError, match="villain_range weights sum"):
        river_scenario_from_dict(data)


def test_matrix_mode_hand_with_per_hand_showdown_is_rejected():
    data = _sample_dict()
    data["hero_range"][0]["showdown"] = "chop"
    with pytest.raises(ValueError, match="must not be set in matrix mode"):
        river_scenario_from_dict(data)


def test_shared_hero_villain_hand_id_is_rejected():
    data = _sample_dict()
    data["villain_range"][0]["hand_id"] = "hero_chop"
    data["showdown_matrix"]["hero_chop"]["hero_chop"] = data["showdown_matrix"][
        "hero_chop"
    ].pop("villain_chop")
    data["showdown_matrix"]["hero_strong"]["hero_chop"] = data["showdown_matrix"][
        "hero_strong"
    ].pop("villain_chop")
    with pytest.raises(ValueError, match="disjoint hand ids"):
        river_scenario_from_dict(data)


def test_matrix_mode_with_top_level_showdown_is_rejected():
    data = _sample_dict()
    data["showdown"] = "chop"
    with pytest.raises(ValueError, match="cannot be combined"):
        river_scenario_from_dict(data)


# ---------------------------------------------------------------------------
# CLI script
# ---------------------------------------------------------------------------


def test_cli_analysis_script_runs_for_matrix():
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), str(_SAMPLE)],
        capture_output=True,
        text=True,
        check=True,
    )
    stdout = completed.stdout
    assert "range_matrix_steal_bet98" in stdout
    assert "Candidate Analysis Summary" in stdout
