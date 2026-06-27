"""Tests for the scenario analysis output exporters (JSON / Markdown / CSV)."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from repeated_poker import (
    RiverScenarioAnalysisConfig,
    run_river_scenario_analysis,
    write_analysis_csv,
    write_analysis_json,
    write_analysis_markdown,
)

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / "examples" / "scenarios" / "range_equity_betting_tree_bet98.json"
_SCRIPT = _ROOT / "scripts" / "run_river_scenario_analysis.py"


@pytest.fixture(scope="module")
def result():
    return run_river_scenario_analysis(
        _SAMPLE, RiverScenarioAnalysisConfig(ranking_criterion="t_deadline")
    )


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def test_write_json_contains_core_fields(tmp_path, result):
    path = tmp_path / "out.json"
    write_analysis_json(result, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == "range_equity_betting_tree_bet98"
    assert payload["selected_horizon"] == result.horizon
    assert payload["counts"] == {
        "generated": len(result.pipeline_result.generated_candidates),
        "kept": result.pipeline_result.filter_result.summary_counts.kept,
        "excluded": result.pipeline_result.filter_result.summary_counts.excluded,
    }


def test_write_json_contains_report_and_ranking(tmp_path, result):
    path = tmp_path / "out.json"
    write_analysis_json(result, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "candidate_rows" in payload["analysis_report"]
    assert payload["analysis_report"]["candidate_rows"]  # non-empty
    assert payload["ranking"]["criterion"] == "t_deadline"
    assert payload["ranking"]["ranked_rows"]


def test_write_json_contains_filter_result(tmp_path, result):
    path = tmp_path / "out.json"
    write_analysis_json(result, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    filter_result = payload["filter_result"]
    assert "summary_counts" in filter_result
    assert "kept_candidate_ids" in filter_result
    assert "excluded" in filter_result


def test_write_json_creates_parent_directory(tmp_path, result):
    path = tmp_path / "nested" / "deep" / "out.json"
    write_analysis_json(result, path)
    assert path.exists()


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def test_write_markdown_contains_summary(tmp_path, result):
    path = tmp_path / "out.md"
    write_analysis_markdown(result, path)
    text = path.read_text(encoding="utf-8")
    assert "Candidate Analysis Summary" in text


def test_write_markdown_without_summary_raises(tmp_path):
    no_md = run_river_scenario_analysis(
        _SAMPLE, RiverScenarioAnalysisConfig(markdown=False)
    )
    with pytest.raises(ValueError, match="markdown_summary"):
        write_analysis_markdown(no_md, tmp_path / "out.md")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def test_write_csv_header_and_rows(tmp_path, result):
    path = tmp_path / "out.csv"
    write_analysis_csv(result, path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    assert "candidate_id" in header
    assert "t_deadline" in header
    assert len(rows) - 1 == len(result.pipeline_result.analysis_report.rows)
    assert len(rows) >= 2  # header + at least one candidate


def test_write_csv_formats_none_and_bool(tmp_path, result):
    path = tmp_path / "out.csv"
    write_analysis_csv(result, path)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        first = next(reader)
    # is_eligible is a bool rendered as true/false.
    assert first["is_eligible"] in {"true", "false"}
    # Detection is disabled in the sample, so t_detect is empty (None -> "").
    assert first["t_detect_estimated_opportunities"] == ""


# ---------------------------------------------------------------------------
# CLI subprocess
# ---------------------------------------------------------------------------


def test_cli_writes_all_outputs(tmp_path):
    json_path = tmp_path / "r.json"
    md_path = tmp_path / "r.md"
    csv_path = tmp_path / "r.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            str(_SAMPLE),
            "--output-json",
            str(json_path),
            "--output-markdown",
            str(md_path),
            "--output-csv",
            str(csv_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stdout = completed.stdout
    assert "saved JSON to" in stdout
    assert "saved Markdown to" in stdout
    assert "saved CSV to" in stdout
    assert json_path.exists()
    assert md_path.exists()
    assert csv_path.exists()
    # Existing stdout behaviour is preserved.
    assert "scenario_id: range_equity_betting_tree_bet98" in stdout
    assert "Candidate Analysis Summary" in stdout


def test_cli_output_markdown_overrides_no_markdown(tmp_path):
    md_path = tmp_path / "r.md"
    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            str(_SAMPLE),
            "--no-markdown",
            "--output-markdown",
            str(md_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    # File is written even though stdout omits the summary.
    assert md_path.exists()
    assert "Candidate Analysis Summary" in md_path.read_text(encoding="utf-8")
    assert "Candidate Analysis Summary" not in completed.stdout


def test_cli_without_outputs_writes_no_file(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), str(_SAMPLE)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "saved " not in completed.stdout
    assert "scenario_id: range_equity_betting_tree_bet98" in completed.stdout
