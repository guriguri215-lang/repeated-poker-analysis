"""Tests for the abstract weighted Hero range scenario input (v1)."""

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
)
from repeated_poker.game import (
    ChanceNode,
    collect_hero_info_sets,
    collect_villain_info_sets,
)

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / "examples" / "scenarios" / "abstract_range_steal_bet98.json"
_SCRIPT = _ROOT / "scripts" / "run_river_scenario_analysis.py"


def _sample_dict() -> dict:
    return json.loads(_SAMPLE.read_text(encoding="utf-8"))


def _build_sample():
    return build_river_steal_game_from_scenario(load_river_scenario_json(_SAMPLE))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_sample_range_scenario_loads():
    scenario = load_river_scenario_json(_SAMPLE)
    assert scenario.is_range_mode
    assert scenario.scenario_id == "abstract_range_steal_bet98"
    assert scenario.showdown is None
    assert scenario.baseline_hero_strategy is None
    hands = scenario.hero_range.hands
    assert [h.hand_id for h in hands] == ["chop_fold_candidate", "hero_winner"]
    assert [h.weight for h in hands] == [0.8, 0.2]
    assert [h.showdown for h in hands] == ["chop", "hero"]
    assert hands[0].baseline_strategy == {"call": 0.0, "fold": 1.0}
    assert hands[1].baseline_strategy == {"call": 1.0, "fold": 0.0}


def test_weight_not_summing_to_one_is_rejected():
    data = _sample_dict()
    data["hero_range"][0]["weight"] = 0.5  # 0.5 + 0.2 != 1
    with pytest.raises(ValueError, match="weights sum"):
        river_scenario_from_dict(data)


def test_duplicate_hand_id_is_rejected():
    data = _sample_dict()
    data["hero_range"][1]["hand_id"] = "chop_fold_candidate"
    with pytest.raises(ValueError, match="duplicate hand_id"):
        river_scenario_from_dict(data)


def test_invalid_hand_showdown_is_rejected():
    data = _sample_dict()
    data["hero_range"][0]["showdown"] = "split"
    with pytest.raises(ValueError, match="showdown"):
        river_scenario_from_dict(data)


def test_invalid_hand_baseline_probability_sum_is_rejected():
    data = _sample_dict()
    data["hero_range"][0]["baseline_strategy"] = {"call": 0.4, "fold": 0.4}
    with pytest.raises(ValueError, match="sum"):
        river_scenario_from_dict(data)


def test_non_positive_weight_is_rejected():
    data = _sample_dict()
    data["hero_range"][0]["weight"] = 0.0
    with pytest.raises(ValueError, match="weight"):
        river_scenario_from_dict(data)


def test_empty_hand_id_is_rejected():
    data = _sample_dict()
    data["hero_range"][0]["hand_id"] = ""
    with pytest.raises(ValueError, match="hand_id"):
        river_scenario_from_dict(data)


def test_range_and_single_hand_fields_are_mutually_exclusive():
    data = _sample_dict()
    data["showdown"] = "chop"
    with pytest.raises(ValueError, match="hero_range cannot be combined"):
        river_scenario_from_dict(data)


def test_range_and_baseline_hero_strategy_are_mutually_exclusive():
    data = _sample_dict()
    data["baseline_hero_strategy"] = {"IP_vs_bet": {"call": 1.0, "fold": 0.0}}
    with pytest.raises(ValueError, match="hero_range cannot be combined"):
        river_scenario_from_dict(data)


# ---------------------------------------------------------------------------
# Game construction
# ---------------------------------------------------------------------------


def test_range_tree_root_is_chance_node():
    build = _build_sample()
    assert isinstance(build.tree.root, ChanceNode)
    weights = [p for p, _ in build.tree.root.children]
    assert weights == [0.8, 0.2]


def test_villain_info_set_is_shared_across_hands():
    build = _build_sample()
    villain_info_sets = collect_villain_info_sets(build.tree)
    assert set(villain_info_sets) == {"OOP_river"}


def test_hero_info_sets_are_per_hand():
    build = _build_sample()
    hero_info_sets = set(collect_hero_info_sets(build.tree))
    assert hero_info_sets == {
        "IP_vs_bet::chop_fold_candidate",
        "IP_vs_bet::hero_winner",
    }


def test_baseline_hero_strategy_is_per_hand():
    build = _build_sample()
    probs = build.baseline_hero_strategy.probabilities
    assert probs["IP_vs_bet::chop_fold_candidate"] == {"call": 0.0, "fold": 1.0}
    assert probs["IP_vs_bet::hero_winner"] == {"call": 1.0, "fold": 0.0}


def test_range_metadata_records_buckets():
    build = _build_sample()
    assert build.metadata["mode"] == "range"
    assert [b["hand_id"] for b in build.metadata["hand_buckets"]] == [
        "chop_fold_candidate",
        "hero_winner",
    ]


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


def test_run_river_scenario_analysis_succeeds_for_range():
    result = run_river_scenario_analysis(_SAMPLE)
    assert result.scenario_id == "abstract_range_steal_bet98"
    assert result.pipeline_result.generated_candidates
    assert result.pipeline_result.analysis_report.rows
    assert "Candidate Analysis Summary" in result.markdown_summary


def test_run_does_not_mutate_range_scenario_dict():
    data = _sample_dict()
    before = copy.deepcopy(data)
    scenario = river_scenario_from_dict(data)
    run_river_scenario_analysis(scenario)
    assert data == before


# ---------------------------------------------------------------------------
# CLI script
# ---------------------------------------------------------------------------


def test_cli_analysis_script_runs_for_range():
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), str(_SAMPLE)],
        capture_output=True,
        text=True,
        check=True,
    )
    stdout = completed.stdout
    assert "abstract_range_steal_bet98" in stdout
    assert "Candidate Analysis Summary" in stdout
