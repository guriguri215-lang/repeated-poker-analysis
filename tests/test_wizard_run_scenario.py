"""Tests for the guided scenario workflow CLI (wizard_run_scenario)."""

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "wizard_run_scenario.py"
_SAMPLE = _ROOT / "examples" / "scenarios" / "nuts_chop_steal_bet98.json"

sys.path.insert(0, str(_ROOT / "scripts"))
import wizard_run_scenario as workflow  # noqa: E402


def _run(argv, answers=()):
    """Run the workflow with scripted answers, returning (rc, printed_lines)."""

    feed = iter(answers)
    printed = []
    rc = workflow.main(
        argv=argv,
        input_func=lambda _prompt: next(feed),
        print_func=printed.append,
    )
    return rc, printed


# ---------------------------------------------------------------------------
# Existing-file mode
# ---------------------------------------------------------------------------


def test_existing_file_mode_analyses_sample():
    rc, printed = _run(["--scenario", str(_SAMPLE)])
    assert rc == 0
    text = "\n".join(printed)
    assert "scenario_id: nuts_chop_steal_bet98" in text
    assert "validation: ok" in text
    assert "candidates: generated=" in text


def test_existing_file_is_not_modified(tmp_path):
    scenario = tmp_path / "scenario.json"
    original = _SAMPLE.read_text(encoding="utf-8")
    scenario.write_text(original, encoding="utf-8")
    rc, _ = _run(["--scenario", str(scenario)])
    assert rc == 0
    assert scenario.read_text(encoding="utf-8") == original


def test_existing_file_missing_is_clean_error(tmp_path):
    missing = tmp_path / "nope.json"
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), "--scenario", str(missing)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "error:" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert "Traceback" not in completed.stdout


# ---------------------------------------------------------------------------
# Create-and-run mode
# ---------------------------------------------------------------------------


def test_create_and_run_non_interactive(tmp_path):
    out = tmp_path / "scenario.json"
    rc, printed = _run(
        ["--kind", "single-hand", "--scenario-output", str(out), "--non-interactive"]
    )
    assert rc == 0
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["format_version"] == "1"
    text = "\n".join(printed)
    assert "scenario_id: template_single_hand" in text
    assert "validation: ok" in text
    assert any("toy values" in line for line in printed)


def test_create_and_run_existing_output_without_force_rejected(tmp_path):
    out = tmp_path / "scenario.json"
    out.write_text("existing", encoding="utf-8")
    rc, _ = _run(
        ["--kind", "single-hand", "--scenario-output", str(out), "--non-interactive"]
    )
    assert rc == 1
    assert out.read_text(encoding="utf-8") == "existing"


def test_create_and_run_existing_output_with_force(tmp_path):
    out = tmp_path / "scenario.json"
    out.write_text("existing", encoding="utf-8")
    rc, _ = _run(
        [
            "--kind",
            "single-hand",
            "--scenario-output",
            str(out),
            "--non-interactive",
            "--force",
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == "template_single_hand"


# ---------------------------------------------------------------------------
# Mode / argument errors
# ---------------------------------------------------------------------------


def test_scenario_and_kind_together_rejected(tmp_path):
    rc, _ = _run(["--scenario", str(_SAMPLE), "--kind", "single-hand"])
    assert rc == 1


def test_non_interactive_without_scenario_or_kind_rejected():
    rc, _ = _run(["--non-interactive"])
    assert rc == 1


def test_non_interactive_create_without_output_rejected():
    rc, _ = _run(["--kind", "single-hand", "--non-interactive"])
    assert rc == 1


# ---------------------------------------------------------------------------
# Create-only flags rejected with an existing scenario
# ---------------------------------------------------------------------------


def _run_cli(extra_argv):
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--scenario", str(_SAMPLE), *extra_argv],
        capture_output=True,
        text=True,
    )


def test_existing_scenario_with_scenario_id_rejected():
    completed = _run_cli(["--scenario-id", "foo"])
    assert completed.returncode != 0
    assert "error:" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert "Traceback" not in completed.stdout


def test_existing_scenario_with_description_rejected():
    completed = _run_cli(["--description", "something"])
    assert completed.returncode != 0
    assert "error:" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert "Traceback" not in completed.stdout


def test_existing_scenario_with_force_rejected():
    completed = _run_cli(["--force"])
    assert completed.returncode != 0
    assert "error:" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert "Traceback" not in completed.stdout


# ---------------------------------------------------------------------------
# Validation happens before the scenario file is written (create mode)
# ---------------------------------------------------------------------------


def test_create_validation_failure_does_not_write_file(tmp_path, monkeypatch, capsys):
    out = tmp_path / "scenario.json"

    def _boom(_scenario):
        raise ValueError("forced validation failure")

    monkeypatch.setattr(workflow, "build_river_steal_game_from_scenario", _boom)
    rc = workflow.main(
        argv=[
            "--kind",
            "single-hand",
            "--scenario-output",
            str(out),
            "--non-interactive",
        ],
        input_func=lambda _p: "",
        print_func=lambda _m: None,
    )
    assert rc == 1
    # The invalid scenario must not have been written.
    assert not out.exists()
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "Traceback" not in captured.err


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def test_output_json_strict(tmp_path):
    out_json = tmp_path / "result.json"
    rc, printed = _run(
        ["--scenario", str(_SAMPLE), "--output-json", str(out_json), "--strict-json"]
    )
    assert rc == 0
    text = out_json.read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert "NaN" not in text
    payload = json.loads(text)  # parses as strict JSON
    assert payload["format_version"] == "1"
    assert any("saved JSON to" in line for line in printed)


def test_output_markdown(tmp_path):
    out_md = tmp_path / "result.md"
    rc, printed = _run(["--scenario", str(_SAMPLE), "--output-markdown", str(out_md)])
    assert rc == 0
    assert out_md.exists()
    assert out_md.read_text(encoding="utf-8").strip()
    assert any("saved Markdown to" in line for line in printed)


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
    assert completed.returncode == 0
    assert "--scenario" in completed.stdout


def test_cli_existing_file_end_to_end():
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), "--scenario", str(_SAMPLE)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "scenario_id: nuts_chop_steal_bet98" in completed.stdout
    assert "validation: ok" in completed.stdout
    assert "Traceback" not in completed.stderr
