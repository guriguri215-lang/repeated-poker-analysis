"""Tests for the form-model round-trip writer CLI (roundtrip_scenario_form)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "roundtrip_scenario_form.py"
_SCENARIOS = _ROOT / "examples" / "scenarios"
_SINGLE_HAND = _SCENARIOS / "nuts_chop_steal_bet98.json"

sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))
from repeated_poker import (  # noqa: E402
    FormValidationMessage,
    build_river_steal_game_from_scenario,
    river_scenario_from_dict,
    single_hand_form_from_dict,
    single_hand_form_to_dict,
)
import roundtrip_scenario_form as writer  # noqa: E402

_SAMPLES = [
    "nuts_chop_steal_bet98.json",
    "abstract_range_steal_bet98.json",
    "range_matrix_steal_bet98.json",
    "range_equity_steal_bet98.json",
    "range_equity_betting_tree_bet98.json",
]


def _run(*args):
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
    )


def _assert_round_trips(data: dict) -> None:
    build_river_steal_game_from_scenario(river_scenario_from_dict(data))


# ---------------------------------------------------------------------------
# stdout output across the five modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", _SAMPLES)
def test_stdout_emits_round_trippable_json(filename):
    result = _run(str(_SCENARIOS / filename))
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)  # stdout is JSON only
    _assert_round_trips(data)
    assert data["format_version"] == "1"


@pytest.mark.parametrize("filename", _SAMPLES)
def test_output_dash_is_stdout(filename):
    result = _run(str(_SCENARIOS / filename), "--output", "-")
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    _assert_round_trips(data)


def test_stdout_has_no_extra_lines():
    # stdout must be pure JSON so it can be piped/redirected.
    result = _run(str(_SINGLE_HAND))
    assert result.returncode == 0
    assert result.stdout.lstrip().startswith("{")
    assert "wrote" not in result.stdout


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------


def test_writes_new_file(tmp_path):
    out = tmp_path / "out.json"
    result = _run(str(_SINGLE_HAND), "--output", str(out))
    assert result.returncode == 0, result.stderr
    assert f"wrote {out}" in result.stdout
    _assert_round_trips(json.loads(out.read_text(encoding="utf-8")))


def test_refuses_to_overwrite_without_force(tmp_path):
    out = tmp_path / "out.json"
    out.write_text("ORIGINAL", encoding="utf-8")
    result = _run(str(_SINGLE_HAND), "--output", str(out))
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
    # The existing file is left untouched.
    assert out.read_text(encoding="utf-8") == "ORIGINAL"


def test_force_overwrites_existing_file(tmp_path):
    out = tmp_path / "out.json"
    out.write_text("ORIGINAL", encoding="utf-8")
    result = _run(str(_SINGLE_HAND), "--output", str(out), "--force")
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    _assert_round_trips(data)


def test_creates_parent_directories(tmp_path):
    out = tmp_path / "nested" / "deeper" / "out.json"
    result = _run(str(_SINGLE_HAND), "--output", str(out))
    assert result.returncode == 0, result.stderr
    assert out.is_file()


# ---------------------------------------------------------------------------
# strict-json
# ---------------------------------------------------------------------------


def test_strict_json_is_valid_and_matches_lenient(tmp_path):
    # A valid scenario has no non-finite numbers (the parser rejects them), so
    # strict and lenient output decode to the same data; the flag is wired through
    # to report_export's serialiser without changing valid output.
    lenient = _run(str(_SCENARIOS / "range_equity_steal_bet98.json"))
    strict = _run(str(_SCENARIOS / "range_equity_steal_bet98.json"), "--strict-json")
    assert lenient.returncode == 0 and strict.returncode == 0
    assert json.loads(strict.stdout) == json.loads(lenient.stdout)
    _assert_round_trips(json.loads(strict.stdout))


def test_strict_json_file_round_trips(tmp_path):
    out = tmp_path / "strict.json"
    result = _run(
        str(_SCENARIOS / "range_equity_betting_tree_bet98.json"),
        "--output",
        str(out),
        "--strict-json",
    )
    assert result.returncode == 0, result.stderr
    _assert_round_trips(json.loads(out.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_missing_file_errors_cleanly():
    result = _run(str(_SCENARIOS / "does_not_exist.json"))
    assert result.returncode != 0
    assert result.stderr.startswith("error:")
    assert "Traceback" not in result.stderr


def test_invalid_json_errors_cleanly(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    result = _run(str(bad))
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_invalid_scenario_errors_cleanly(tmp_path):
    bad = tmp_path / "empty.json"
    bad.write_text("{}", encoding="utf-8")
    result = _run(str(bad))
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_output_directory_errors_cleanly(tmp_path):
    result = _run(str(_SINGLE_HAND), "--output", str(tmp_path))
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_validation_messages_block_write(tmp_path, monkeypatch):
    # A form with validation messages must not be written; the function raises and
    # leaves no file. Patch the single-hand validator to return a message.
    def fake_validate(form):
        return [FormValidationMessage("scenario_id", "forced message")]

    monkeypatch.setitem(
        writer._MODE_HELPERS,
        "single-hand",
        (single_hand_form_from_dict, fake_validate, single_hand_form_to_dict),
    )
    out = tmp_path / "out.json"
    with pytest.raises(ValueError):
        writer.roundtrip_scenario_form(str(_SINGLE_HAND), output=str(out))
    assert not out.exists()


def test_round_trip_failure_blocks_write(tmp_path, monkeypatch):
    # If to_dict produces something the parser rejects, nothing is written.
    def broken_to_dict(form):
        return {"format_version": "1", "scenario_id": "x"}  # not a valid scenario

    monkeypatch.setitem(
        writer._MODE_HELPERS,
        "single-hand",
        (single_hand_form_from_dict, lambda form: [], broken_to_dict),
    )
    out = tmp_path / "out.json"
    with pytest.raises(ValueError):
        writer.roundtrip_scenario_form(str(_SINGLE_HAND), output=str(out))
    assert not out.exists()
