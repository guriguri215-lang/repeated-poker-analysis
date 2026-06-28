"""Tests for the single-hand scenario form edit CLI (edit_scenario_form)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "edit_scenario_form.py"
_SCENARIOS = _ROOT / "examples" / "scenarios"
_SINGLE_HAND = _SCENARIOS / "nuts_chop_steal_bet98.json"
_MATRIX = _SCENARIOS / "range_matrix_steal_bet98.json"

sys.path.insert(0, str(_ROOT / "src"))
from repeated_poker import (  # noqa: E402
    build_river_steal_game_from_scenario,
    river_scenario_from_dict,
)


def _run(*args):
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
    )


def _assert_round_trips(data: dict) -> None:
    build_river_steal_game_from_scenario(river_scenario_from_dict(data))


# ---------------------------------------------------------------------------
# Editing and writing
# ---------------------------------------------------------------------------


def test_edit_writes_round_trippable_file(tmp_path):
    out = tmp_path / "edited.json"
    result = _run(
        str(_SINGLE_HAND),
        "--set",
        "scenario_id=edited_single_hand",
        "--set",
        "bet_size=50",
        "--set",
        "horizons=10,20",
        "--output",
        str(out),
    )
    assert result.returncode == 0, result.stderr
    assert f"wrote {out}" in result.stdout
    data = json.loads(out.read_text(encoding="utf-8"))
    _assert_round_trips(data)
    assert data["scenario_id"] == "edited_single_hand"
    assert data["bet_size"] == 50.0
    assert data["repeated"]["horizons"] == [10, 20]


def test_multiple_sets_all_apply(tmp_path):
    out = tmp_path / "edited.json"
    result = _run(
        str(_SINGLE_HAND),
        "--set",
        "rake_rate=0.0",
        "--set",
        "rake_cap=none",
        "--set",
        "shift_amounts=0.25,0.5,1.0",
        "--set",
        "discount=0.9",
        "--set",
        "description=edited via cli",
        "--output",
        str(out),
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    _assert_round_trips(data)
    assert data["rake"]["rate"] == 0.0
    assert data["rake"]["cap"] is None
    assert data["candidate_generation"]["shift_amounts"] == [0.25, 0.5, 1.0]
    assert data["repeated"]["discount"] == 0.9
    assert data["description"] == "edited via cli"


def test_baseline_probabilities_edit(tmp_path):
    out = tmp_path / "edited.json"
    result = _run(
        str(_SINGLE_HAND),
        "--set",
        "baseline_call_probability=1.0",
        "--set",
        "baseline_fold_probability=0.0",
        "--output",
        str(out),
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    dist = data["baseline_hero_strategy"]["IP_vs_bet"]
    assert dist["call"] == 1.0
    assert dist["fold"] == 0.0


def test_dotted_aliases_apply(tmp_path):
    out = tmp_path / "edited.json"
    result = _run(
        str(_SINGLE_HAND),
        "--set",
        "rake.rate=0.0",
        "--set",
        "rake.cap=null",
        "--set",
        "initial_commitment.hero=2",
        "--set",
        "baseline.call=1.0",
        "--set",
        "baseline.fold=0.0",
        "--output",
        str(out),
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["rake"]["rate"] == 0.0
    assert data["rake"]["cap"] is None
    assert data["initial_commitment"]["hero"] == 2.0
    assert data["baseline_hero_strategy"]["IP_vs_bet"]["call"] == 1.0


def test_no_set_round_trips_unchanged_id(tmp_path):
    out = tmp_path / "edited.json"
    result = _run(str(_SINGLE_HAND), "--output", str(out))
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["scenario_id"] == "nuts_chop_steal_bet98"


# ---------------------------------------------------------------------------
# Output handling
# ---------------------------------------------------------------------------


def test_stdout_is_json_only():
    result = _run(str(_SINGLE_HAND), "--set", "bet_size=42")
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["bet_size"] == 42.0
    assert "wrote" not in result.stdout


def test_output_dash_is_stdout():
    result = _run(str(_SINGLE_HAND), "--set", "bet_size=42", "--output", "-")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["bet_size"] == 42.0


def test_refuses_overwrite_without_force(tmp_path):
    out = tmp_path / "out.json"
    out.write_text("ORIGINAL", encoding="utf-8")
    result = _run(str(_SINGLE_HAND), "--set", "bet_size=42", "--output", str(out))
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
    assert out.read_text(encoding="utf-8") == "ORIGINAL"


def test_force_overwrites(tmp_path):
    out = tmp_path / "out.json"
    out.write_text("ORIGINAL", encoding="utf-8")
    result = _run(str(_SINGLE_HAND), "--set", "bet_size=42", "--output", str(out), "--force")
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["bet_size"] == 42.0


def test_creates_parent_directories(tmp_path):
    out = tmp_path / "nested" / "deeper" / "out.json"
    result = _run(str(_SINGLE_HAND), "--output", str(out))
    assert result.returncode == 0, result.stderr
    assert out.is_file()


def test_strict_json_is_valid(tmp_path):
    out = tmp_path / "out.json"
    result = _run(str(_SINGLE_HAND), "--set", "bet_size=42", "--output", str(out), "--strict-json")
    assert result.returncode == 0, result.stderr
    _assert_round_trips(json.loads(out.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Rejections / error paths
# ---------------------------------------------------------------------------


def test_rejects_non_single_hand_scenario():
    result = _run(str(_MATRIX), "--set", "bet_size=50", "--output", "-")
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "single-hand" in result.stderr
    assert "Traceback" not in result.stderr


def test_rejects_unknown_field():
    result = _run(str(_SINGLE_HAND), "--set", "nope=1", "--output", "-")
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_rejects_malformed_set():
    result = _run(str(_SINGLE_HAND), "--set", "bet_size", "--output", "-")
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_rejects_bad_numeric_value():
    result = _run(str(_SINGLE_HAND), "--set", "bet_size=abc", "--output", "-")
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_invalid_value_blocks_write(tmp_path):
    # Editing call probability so call+fold != 1 yields a validation message; the
    # file must not be written.
    out = tmp_path / "out.json"
    result = _run(
        str(_SINGLE_HAND),
        "--set",
        "baseline_call_probability=0.3",
        "--output",
        str(out),
    )
    assert result.returncode != 0
    assert "validation message" in result.stderr
    assert not out.exists()


def test_missing_file_errors_cleanly():
    result = _run(str(_SCENARIOS / "does_not_exist.json"), "--output", "-")
    assert result.returncode != 0
    assert result.stderr.startswith("error:")
    assert "Traceback" not in result.stderr


def test_invalid_json_errors_cleanly(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    result = _run(str(bad), "--output", "-")
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_output_directory_errors_cleanly(tmp_path):
    result = _run(str(_SINGLE_HAND), "--output", str(tmp_path))
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
