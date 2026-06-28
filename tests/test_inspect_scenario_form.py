"""Tests for the form-model inspection CLI (inspect_scenario_form)."""

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "inspect_scenario_form.py"
_SCENARIOS = _ROOT / "examples" / "scenarios"

sys.path.insert(0, str(_ROOT / "src"))
from repeated_poker import detect_scenario_form_mode  # noqa: E402

# (filename, scenario_id, expected mode, expected form class name).
_CASES = [
    ("nuts_chop_steal_bet98.json", "nuts_chop_steal_bet98", "single-hand", "SingleHandScenarioForm"),
    (
        "abstract_range_steal_bet98.json",
        "abstract_range_steal_bet98",
        "hero-range",
        "HeroRangeScenarioForm",
    ),
    (
        "range_matrix_steal_bet98.json",
        "range_matrix_steal_bet98",
        "showdown-matrix",
        "ShowdownMatrixScenarioForm",
    ),
    (
        "range_equity_steal_bet98.json",
        "range_equity_steal_bet98",
        "equity-matrix",
        "EquityMatrixScenarioForm",
    ),
    (
        "range_equity_betting_tree_bet98.json",
        "range_equity_betting_tree_bet98",
        "betting-tree",
        "BettingTreeScenarioForm",
    ),
]


def _run(*args):
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# CLI: the five bundled scenarios
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename,scenario_id,mode,form_class", _CASES)
def test_cli_inspects_each_bundled_mode(filename, scenario_id, mode, form_class):
    result = _run(str(_SCENARIOS / filename))
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "Scenario form inspection" in out
    assert f"scenario_id: {scenario_id}" in out
    assert f"mode: {mode}" in out
    assert f"form: {form_class}" in out
    assert "validation: ok" in out
    assert "round_trip_parse: ok" in out
    assert "round_trip_build: ok" in out


# ---------------------------------------------------------------------------
# CLI: error handling (clean message, non-zero, no traceback)
# ---------------------------------------------------------------------------


def test_cli_missing_file_errors_cleanly():
    result = _run(str(_SCENARIOS / "does_not_exist.json"))
    assert result.returncode != 0
    assert result.stderr.startswith("error:")
    assert "Traceback" not in result.stderr


def test_cli_invalid_json_errors_cleanly(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    result = _run(str(bad))
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_invalid_scenario_errors_cleanly(tmp_path):
    # Parses as JSON but is not a valid scenario (no scenario_id, etc.).
    bad = tmp_path / "empty.json"
    bad.write_text("{}", encoding="utf-8")
    result = _run(str(bad))
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_non_object_json_errors_cleanly(tmp_path):
    bad = tmp_path / "list.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    result = _run(str(bad))
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_requires_a_scenario_argument():
    result = _run()
    assert result.returncode != 0  # argparse usage error


# ---------------------------------------------------------------------------
# detect_scenario_form_mode unit checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename,scenario_id,mode,form_class", _CASES)
def test_detect_mode_matches_each_sample(filename, scenario_id, mode, form_class):
    import json

    data = json.loads((_SCENARIOS / filename).read_text(encoding="utf-8"))
    assert detect_scenario_form_mode(data) == mode


def test_detect_mode_prefers_betting_tree():
    # betting_tree wins even with a matrix present (matrix mode + betting_tree).
    data = {"hero_range": [], "villain_range": [], "equity_matrix": {}, "betting_tree": {}}
    assert detect_scenario_form_mode(data) == "betting-tree"


def test_detect_mode_equity_over_showdown_key_precedence():
    data = {"hero_range": [], "villain_range": [], "equity_matrix": {}}
    assert detect_scenario_form_mode(data) == "equity-matrix"


def test_detect_mode_hero_range_only():
    assert detect_scenario_form_mode({"hero_range": []}) == "hero-range"


def test_detect_mode_defaults_to_single_hand():
    assert detect_scenario_form_mode({"showdown": "chop"}) == "single-hand"


def test_detect_mode_rejects_non_dict():
    with pytest.raises(ValueError):
        detect_scenario_form_mode([1, 2, 3])
