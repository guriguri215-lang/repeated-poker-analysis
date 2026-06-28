"""Tests for the interactive scenario wizard CLI."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "wizard_create_scenario.py"
_VALIDATE_SCRIPT = _ROOT / "scripts" / "validate_river_scenario.py"

sys.path.insert(0, str(_ROOT / "scripts"))
import wizard_create_scenario as wizard  # noqa: E402


def _run(argv, answers):
    """Run the wizard with scripted answers, returning (rc, printed_lines)."""

    feed = iter(answers)
    printed = []
    rc = wizard.main(
        argv=argv,
        input_func=lambda _prompt: next(feed),
        print_func=printed.append,
    )
    return rc, printed


# ---------------------------------------------------------------------------
# Interactive flow
# ---------------------------------------------------------------------------


def test_wizard_writes_single_hand_json(tmp_path):
    out = tmp_path / "scenario.json"
    answers = [
        "single-hand",  # kind
        "wiz_single",  # scenario_id
        "a description",  # description
        "0.05",  # rake rate
        "4.0",  # rake cap
        "1",  # hero commit
        "1",  # villain commit
        "98",  # bet size
        "10,20,50,100",  # horizons
        "1.0",  # discount
        str(out),  # output path
    ]
    rc, printed = _run([], answers)
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["format_version"] == "1"
    assert payload["scenario_id"] == "wiz_single"
    assert payload["description"] == "a description"


def test_wizard_empty_input_keeps_template_defaults(tmp_path):
    out = tmp_path / "scenario.json"
    # Every answer is empty except the output path, so the template defaults win.
    answers = [
        "single-hand",  # kind
        "",  # scenario_id -> default
        "",  # description -> default
        "",  # rake rate
        "",  # rake cap
        "",  # hero commit
        "",  # villain commit
        "",  # bet size
        "",  # horizons
        "",  # discount
        str(out),  # output path
    ]
    rc, _ = _run([], answers)
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == "template_single_hand"
    assert payload["rake"] == {"rate": 0.05, "cap": 4.0}
    assert payload["repeated"]["horizons"] == [10, 20, 50, 100]


def test_wizard_skips_questions_for_provided_flags(tmp_path):
    out = tmp_path / "scenario.json"
    # --kind and --output are given, so those questions are not asked; only the
    # remaining fields are prompted.
    answers = [
        "",  # scenario_id -> default
        "",  # description -> default
        "",  # rake rate
        "",  # rake cap
        "",  # hero commit
        "",  # villain commit
        "",  # bet size
        "",  # horizons
        "",  # discount
    ]
    rc, _ = _run(["--kind", "single-hand", "--output", str(out)], answers)
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == "template_single_hand"


def test_wizard_parses_comma_separated_horizons(tmp_path):
    out = tmp_path / "scenario.json"
    answers = [
        "",  # scenario_id
        "",  # description
        "",  # rake rate
        "",  # rake cap
        "",  # hero commit
        "",  # villain commit
        "",  # bet size
        "5,15,25",  # horizons
        "",  # discount
    ]
    rc, _ = _run(["--kind", "single-hand", "--output", str(out)], answers)
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["repeated"]["horizons"] == [5, 15, 25]


def test_wizard_cap_none_sets_null(tmp_path):
    out = tmp_path / "scenario.json"
    answers = [
        "",  # scenario_id
        "",  # description
        "",  # rake rate
        "none",  # rake cap -> null
        "",  # hero commit
        "",  # villain commit
        "",  # bet size
        "",  # horizons
        "",  # discount
    ]
    rc, _ = _run(["--kind", "single-hand", "--output", str(out)], answers)
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["rake"]["cap"] is None


def test_wizard_prints_toy_values_note(tmp_path):
    out = tmp_path / "scenario.json"
    answers = ["", "", "", "", "", "", "", "", ""]
    rc, printed = _run(["--kind", "single-hand", "--output", str(out)], answers)
    assert rc == 0
    assert any("toy values" in line for line in printed)
    assert any("edit the JSON manually" in line for line in printed)


def test_wizard_generates_validatable_json_for_betting_tree(tmp_path):
    # The betting-tree kind has no bet_size question; the wizard must skip it.
    out = tmp_path / "scenario.json"
    answers = [
        "",  # scenario_id
        "",  # description
        "",  # rake rate
        "",  # rake cap
        "",  # hero commit
        "",  # villain commit
        # no bet size question for betting-tree
        "",  # horizons
        "",  # discount
    ]
    rc, _ = _run(
        ["--kind", "range-matrix-equity-betting-tree", "--output", str(out)], answers
    )
    assert rc == 0
    completed = subprocess.run(
        [sys.executable, str(_VALIDATE_SCRIPT), str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "errors: 0" in completed.stdout


# ---------------------------------------------------------------------------
# Overwrite handling
# ---------------------------------------------------------------------------


def test_wizard_refuses_existing_file_without_force(tmp_path):
    out = tmp_path / "scenario.json"
    out.write_text("existing", encoding="utf-8")
    # Non-interactive so no overwrite prompt is consulted.
    rc = wizard.main(
        argv=["--kind", "single-hand", "--non-interactive", "--output", str(out)],
        input_func=lambda _p: "",
        print_func=lambda _m: None,
    )
    assert rc == 1
    assert out.read_text(encoding="utf-8") == "existing"


def test_wizard_force_overwrites_existing_file(tmp_path):
    out = tmp_path / "scenario.json"
    out.write_text("existing", encoding="utf-8")
    rc = wizard.main(
        argv=[
            "--kind",
            "single-hand",
            "--non-interactive",
            "--output",
            str(out),
            "--force",
        ],
        input_func=lambda _p: "",
        print_func=lambda _m: None,
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == "template_single_hand"


# ---------------------------------------------------------------------------
# Non-interactive mode
# ---------------------------------------------------------------------------


def test_non_interactive_requires_kind():
    rc = wizard.main(
        argv=["--non-interactive"],
        input_func=lambda _p: "",
        print_func=lambda _m: None,
    )
    assert rc == 1


def test_non_interactive_writes_defaults(tmp_path):
    out = tmp_path / "scenario.json"
    rc = wizard.main(
        argv=["--kind", "range-matrix-equity", "--non-interactive", "--output", str(out)],
        input_func=lambda _p: "",
        print_func=lambda _m: None,
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["format_version"] == "1"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def test_cli_help_runs():
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "wizard" in completed.stdout.lower()


def test_cli_non_interactive_end_to_end(tmp_path):
    out = tmp_path / "scenario.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--kind",
            "single-hand",
            "--non-interactive",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "saved scenario to" in completed.stdout
    assert out.exists()
    validate = subprocess.run(
        [sys.executable, str(_VALIDATE_SCRIPT), str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "errors: 0" in validate.stdout
