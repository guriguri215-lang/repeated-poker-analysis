"""Tests for the JSON scenario analysis runner."""

import copy
import subprocess
import sys
from pathlib import Path

import pytest

from repeated_poker import (
    RiverScenarioAnalysisConfig,
    RiverScenarioAnalysisResult,
    river_scenario_from_dict,
    run_river_scenario_analysis,
)
from repeated_poker.scenario_io import load_river_scenario_json

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / "examples" / "scenarios" / "nuts_chop_steal_bet98.json"
_SCRIPT = _ROOT / "scripts" / "run_river_scenario_analysis.py"


def _scenario_dict() -> dict:
    import json

    return json.loads(_SAMPLE.read_text(encoding="utf-8"))


def test_run_from_path_succeeds():
    result = run_river_scenario_analysis(_SAMPLE)
    assert isinstance(result, RiverScenarioAnalysisResult)
    assert result.scenario_id == "nuts_chop_steal_bet98"
    assert result.pipeline_result.generated_candidates  # not empty
    assert result.pipeline_result.analysis_report.rows  # not empty
    assert "Candidate Analysis Summary" in result.markdown_summary


def test_run_from_parsed_scenario_succeeds():
    scenario = load_river_scenario_json(_SAMPLE)
    result = run_river_scenario_analysis(scenario)
    assert result.scenario_id == "nuts_chop_steal_bet98"
    assert result.scenario is scenario


def test_default_horizon_is_max_of_scenario_horizons():
    result = run_river_scenario_analysis(_SAMPLE)
    assert result.horizon == 100  # max([10, 20, 50, 100])
    assert result.discount == 1.0


def test_explicit_horizon_overrides_scenario():
    result = run_river_scenario_analysis(
        _SAMPLE, RiverScenarioAnalysisConfig(horizon=20)
    )
    assert result.horizon == 20


def test_explicit_discount_overrides_scenario():
    result = run_river_scenario_analysis(
        _SAMPLE, RiverScenarioAnalysisConfig(discount=0.9)
    )
    assert result.discount == 0.9


def test_missing_shift_amounts_raises_clear_error():
    data = _scenario_dict()
    data.pop("candidate_generation", None)
    scenario = river_scenario_from_dict(data)
    with pytest.raises(ValueError, match="shift_amounts"):
        run_river_scenario_analysis(scenario)


def test_missing_horizon_without_override_raises_clear_error():
    data = _scenario_dict()
    data.pop("repeated", None)
    scenario = river_scenario_from_dict(data)
    with pytest.raises(ValueError, match="horizon"):
        run_river_scenario_analysis(scenario)


def test_missing_repeated_horizon_with_override_succeeds():
    data = _scenario_dict()
    data.pop("repeated", None)
    scenario = river_scenario_from_dict(data)
    result = run_river_scenario_analysis(
        scenario, RiverScenarioAnalysisConfig(horizon=10)
    )
    assert result.horizon == 10
    assert result.discount == 1.0  # default when no repeated config


def test_ranking_result_returned_when_criterion_given():
    result = run_river_scenario_analysis(
        _SAMPLE, RiverScenarioAnalysisConfig(ranking_criterion="t_deadline")
    )
    assert result.ranking_result is not None
    assert result.ranking_result.criterion == "t_deadline"
    assert result.ranking_result.ranked_rows


def test_ranking_result_none_by_default():
    result = run_river_scenario_analysis(_SAMPLE)
    assert result.ranking_result is None


def test_no_markdown_disables_summary():
    result = run_river_scenario_analysis(
        _SAMPLE, RiverScenarioAnalysisConfig(markdown=False)
    )
    assert result.markdown_summary is None


def test_to_dict_contains_core_fields():
    result = run_river_scenario_analysis(_SAMPLE)
    payload = result.to_dict()
    assert payload["scenario_id"] == "nuts_chop_steal_bet98"
    assert payload["horizon"] == 100
    assert payload["generated"] >= 1
    assert "report" in payload


def test_run_does_not_mutate_scenario_dict():
    data = _scenario_dict()
    before = copy.deepcopy(data)
    scenario = river_scenario_from_dict(data)
    run_river_scenario_analysis(scenario)
    assert data == before


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
        "Candidate Analysis Summary",
        "generated",
        "kept",
        "T_deadline",
    ):
        assert fragment in stdout, f"missing {fragment!r} in script output:\n{stdout}"


def test_cli_script_ranking_section():
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), str(_SAMPLE), "--rank-by", "t_deadline"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Ranking by t_deadline" in completed.stdout
