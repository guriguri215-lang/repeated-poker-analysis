"""Tests for the scenario template generator API and CLI."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from repeated_poker import (
    available_scenario_template_kinds,
    build_river_steal_game_from_scenario,
    create_scenario_template,
    river_scenario_from_dict,
)

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "create_scenario_template.py"
_VALIDATE_SCRIPT = _ROOT / "scripts" / "validate_river_scenario.py"

_KINDS = [
    "single-hand",
    "hero-range",
    "range-matrix-showdown",
    "range-matrix-equity",
    "range-matrix-equity-betting-tree",
]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_available_kinds_are_the_five_expected():
    assert available_scenario_template_kinds() == _KINDS


@pytest.mark.parametrize("kind", _KINDS)
def test_template_declares_format_version_one(kind):
    template = create_scenario_template(kind)
    assert template["format_version"] == "1"


@pytest.mark.parametrize("kind", _KINDS)
def test_template_parses_and_builds(kind):
    scenario = river_scenario_from_dict(create_scenario_template(kind))
    build = build_river_steal_game_from_scenario(scenario)
    assert build.metadata["format_version"] == "1"


@pytest.mark.parametrize("kind", _KINDS)
def test_template_default_scenario_id_is_non_empty(kind):
    assert create_scenario_template(kind)["scenario_id"]


def test_scenario_id_override_is_used():
    template = create_scenario_template("single-hand", scenario_id="my_custom_id")
    assert template["scenario_id"] == "my_custom_id"


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown template kind"):
        create_scenario_template("not-a-kind")


def test_betting_tree_kind_uses_baseline_strategies_not_baseline_strategy():
    template = create_scenario_template("range-matrix-equity-betting-tree")
    for hand in template["hero_range"]:
        assert "baseline_strategies" in hand
        assert "baseline_strategy" not in hand
    assert "betting_tree" in template
    assert "equity_matrix" in template
    assert "showdown_matrix" not in template


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_list_kinds():
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), "--list-kinds"],
        capture_output=True,
        text=True,
        check=True,
    )
    for kind in _KINDS:
        assert kind in completed.stdout


@pytest.mark.parametrize("kind", _KINDS)
def test_cli_stdout_generation_is_valid_json(kind):
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), "--kind", kind],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["format_version"] == "1"


def test_cli_requires_kind_or_list():
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "--kind is required" in completed.stderr


def test_cli_invalid_kind_rejected():
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), "--kind", "bogus"],
        capture_output=True,
        text=True,
    )
    # argparse rejects an out-of-choices value with exit code 2.
    assert completed.returncode != 0


def test_cli_writes_output_file(tmp_path):
    out = tmp_path / "nested" / "template.json"
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), "--kind", "single-hand", "--output", str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "saved template to" in completed.stdout
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["format_version"] == "1"


def test_cli_existing_file_without_force_is_rejected(tmp_path):
    out = tmp_path / "template.json"
    out.write_text("existing", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), "--kind", "single-hand", "--output", str(out)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "already exists" in completed.stderr
    # The original file is left untouched.
    assert out.read_text(encoding="utf-8") == "existing"


def test_cli_existing_file_with_force_overwrites(tmp_path):
    out = tmp_path / "template.json"
    out.write_text("existing", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--kind",
            "single-hand",
            "--output",
            str(out),
            "--force",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "saved template to" in completed.stdout
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == "template_single_hand"


def test_cli_default_validates(tmp_path):
    # The default run validates; a valid kind succeeds without --no-validate.
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), "--kind", "range-matrix-equity"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout)["format_version"] == "1"


def test_cli_no_validate_runs(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), "--kind", "single-hand", "--no-validate"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout)["format_version"] == "1"


@pytest.mark.parametrize("kind", _KINDS)
def test_generated_file_passes_validation_cli(tmp_path, kind):
    out = tmp_path / f"{kind}.json"
    subprocess.run(
        [sys.executable, str(_SCRIPT), "--kind", kind, "--output", str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    completed = subprocess.run(
        [sys.executable, str(_VALIDATE_SCRIPT), str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "errors: 0" in completed.stdout
