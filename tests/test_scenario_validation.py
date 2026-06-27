"""Tests for the scenario validation report, its exporter, and the CLI."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from repeated_poker import (
    ScenarioValidationConfig,
    ScenarioValidationResult,
    validate_river_scenario_inputs,
    write_validation_json,
)

_ROOT = Path(__file__).resolve().parents[1]
_SCENARIOS = _ROOT / "examples" / "scenarios"
_SCRIPT = _ROOT / "scripts" / "validate_river_scenario.py"
_NUTS = _SCENARIOS / "nuts_chop_steal_bet98.json"
_RANGE = _SCENARIOS / "abstract_range_steal_bet98.json"


# ---------------------------------------------------------------------------
# Input expansion
# ---------------------------------------------------------------------------


def test_directory_input_reads_json_in_name_order():
    validation = validate_river_scenario_inputs(_SCENARIOS)
    expected = sorted(p.name for p in _SCENARIOS.glob("*.json"))
    got = [Path(row.source_path).name for row in validation.rows]
    assert got == expected
    assert isinstance(validation, ScenarioValidationResult)


def test_explicit_file_list_preserves_order():
    validation = validate_river_scenario_inputs([_RANGE, _NUTS])
    ids = [row.scenario_id for row in validation.rows]
    assert ids == ["abstract_range_steal_bet98", "nuts_chop_steal_bet98"]


# ---------------------------------------------------------------------------
# Success rows
# ---------------------------------------------------------------------------


def test_valid_sample_scenarios_are_ok():
    validation = validate_river_scenario_inputs(_SCENARIOS)
    assert validation.error_count == 0
    assert validation.ok_count == len(validation.rows)
    assert all(row.ok for row in validation.rows)


def test_success_row_includes_descriptive_fields():
    validation = validate_river_scenario_inputs([_NUTS])
    row = validation.rows[0]
    assert row.ok is True
    assert row.scenario_id == "nuts_chop_steal_bet98"
    assert row.model_kind == "single_hand"
    assert row.hero_info_set_count == 1
    assert row.villain_info_set_count == 1
    assert row.terminal_count == 3
    assert row.has_chance_node is False
    assert row.chance_outcome_count is None
    assert row.error_type is None
    assert row.error_message is None


def test_success_row_reports_chance_outcomes_and_horizons():
    validation = validate_river_scenario_inputs([_RANGE])
    row = validation.rows[0]
    assert row.model_kind == "range"
    assert row.has_chance_node is True
    assert row.chance_outcome_count is not None and row.chance_outcome_count >= 1
    # The sample range scenario carries a repeated config with horizons.
    assert row.horizons is not None
    assert row.discount is not None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def _write_bad_scenario(tmp_path) -> Path:
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    return bad


def test_fail_fast_raises_on_bad_scenario(tmp_path):
    bad = _write_bad_scenario(tmp_path)
    with pytest.raises(ValueError, match="failed to validate scenario"):
        validate_river_scenario_inputs([bad])


def test_continue_on_error_records_error_row(tmp_path):
    bad = _write_bad_scenario(tmp_path)
    config = ScenarioValidationConfig(continue_on_error=True)
    validation = validate_river_scenario_inputs([bad, _NUTS], config)
    assert validation.error_count == 1
    assert validation.ok_count == 1
    bad_row = validation.rows[0]
    assert bad_row.ok is False
    assert bad_row.scenario_id is None
    assert bad_row.error_type is not None
    assert bad_row.error_message is not None
    ok_row = validation.rows[1]
    assert ok_row.ok is True
    assert ok_row.scenario_id == "nuts_chop_steal_bet98"


def test_source_path_does_not_leak_absolute_user_path(tmp_path):
    bad = _write_bad_scenario(tmp_path)
    config = ScenarioValidationConfig(continue_on_error=True)
    # An absolute, cwd-contained path stays cwd-relative; a bad file outside the
    # cwd is shown by file name only, and neither row leaks an absolute path.
    validation = validate_river_scenario_inputs([str(_NUTS.resolve()), bad], config)
    ok_row, bad_row = validation.rows
    assert not Path(ok_row.source_path).is_absolute()
    assert ok_row.source_path.startswith("examples/scenarios/")
    assert bad_row.source_path == "bad.json"
    assert str(tmp_path) not in bad_row.source_path
    assert str(tmp_path) not in (bad_row.error_message or "")


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def test_validation_json_writes_expected_keys(tmp_path):
    validation = validate_river_scenario_inputs([_NUTS, _RANGE])
    path = tmp_path / "validation.json"
    write_validation_json(validation, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["ok_count"] == 2
    assert payload["error_count"] == 0
    assert len(payload["rows"]) == 2
    first = payload["rows"][0]
    for key in (
        "source_path",
        "ok",
        "scenario_id",
        "model_kind",
        "hero_info_set_count",
        "villain_info_set_count",
        "terminal_count",
    ):
        assert key in first


def test_validation_strict_json_parses(tmp_path):
    validation = validate_river_scenario_inputs(_SCENARIOS)
    path = tmp_path / "validation_strict.json"
    write_validation_json(validation, path, strict=True)
    text = path.read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert "NaN" not in text
    json.loads(text)  # parses as strict JSON


def test_validation_json_parent_directory_auto_created(tmp_path):
    validation = validate_river_scenario_inputs([_NUTS])
    path = tmp_path / "nested" / "deep" / "validation.json"
    write_validation_json(validation, path)
    assert path.exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_directory_input():
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), str(_SCENARIOS)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "scenarios:" in completed.stdout
    assert "nuts_chop_steal_bet98" in completed.stdout
    assert "Traceback" not in completed.stderr


def test_cli_explicit_file_list():
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), str(_RANGE), str(_NUTS)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "abstract_range_steal_bet98" in completed.stdout
    assert "nuts_chop_steal_bet98" in completed.stdout


def test_cli_continue_on_error(tmp_path):
    bad = _write_bad_scenario(tmp_path)
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), str(bad), str(_NUTS), "--continue-on-error"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "errors: 1" in completed.stdout


def test_cli_fail_fast_reports_error_without_traceback(tmp_path):
    bad = _write_bad_scenario(tmp_path)
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), str(bad)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "error:" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert "Traceback" not in completed.stdout


def test_cli_writes_output_json(tmp_path):
    json_path = tmp_path / "validation.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            str(_SCENARIOS),
            "--output-json",
            str(json_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "saved JSON to" in completed.stdout
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["ok_count"] == len(payload["rows"])


def test_cli_strict_json_writes_rfc_json(tmp_path):
    json_path = tmp_path / "strict.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            str(_SCENARIOS),
            "--output-json",
            str(json_path),
            "--strict-json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "saved JSON to" in completed.stdout
    text = json_path.read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert "NaN" not in text
    json.loads(text)  # parses as strict JSON
