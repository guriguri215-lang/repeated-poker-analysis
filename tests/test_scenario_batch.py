"""Tests for the batch scenario runner and its exporters."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from repeated_poker import (
    BatchScenarioAnalysisConfig,
    BatchScenarioAnalysisResult,
    BatchScenarioRow,
    RiverScenarioAnalysisConfig,
    batch_result_to_dict,
    run_batch_scenario_analysis,
    write_batch_csv,
    write_batch_json,
    write_batch_markdown,
)

_ROOT = Path(__file__).resolve().parents[1]
_SCENARIOS = _ROOT / "examples" / "scenarios"
_SCRIPT = _ROOT / "scripts" / "run_scenario_batch.py"
_NUTS = _SCENARIOS / "nuts_chop_steal_bet98.json"
_RANGE = _SCENARIOS / "abstract_range_steal_bet98.json"


# ---------------------------------------------------------------------------
# Input expansion
# ---------------------------------------------------------------------------


def test_directory_input_reads_json_in_name_order():
    batch = run_batch_scenario_analysis(_SCENARIOS)
    expected = sorted(p.name for p in _SCENARIOS.glob("*.json"))
    got = [Path(row.source_path).name for row in batch.rows]
    assert got == expected
    assert isinstance(batch, BatchScenarioAnalysisResult)


def test_explicit_file_list_preserves_order():
    batch = run_batch_scenario_analysis([_RANGE, _NUTS])
    ids = [row.scenario_id for row in batch.rows]
    assert ids == ["abstract_range_steal_bet98", "nuts_chop_steal_bet98"]


def test_source_path_is_relative_form_not_absolute():
    # Pass a relative path; the recorded source_path should stay relative.
    rel = "examples/scenarios/nuts_chop_steal_bet98.json"
    batch = run_batch_scenario_analysis([rel])
    assert batch.rows[0].source_path == rel


def test_absolute_directory_input_yields_relative_source_paths():
    # An absolute directory inside the cwd should still produce cwd-relative
    # source paths (no absolute local path leaks into the summary).
    batch = run_batch_scenario_analysis(str(_SCENARIOS.resolve()))
    for row in batch.rows:
        assert not Path(row.source_path).is_absolute()
        assert row.source_path.startswith("examples/scenarios/")


# ---------------------------------------------------------------------------
# Summary rows
# ---------------------------------------------------------------------------


def test_summary_row_contains_required_fields():
    batch = run_batch_scenario_analysis([_NUTS])
    row = batch.rows[0]
    assert row.scenario_id == "nuts_chop_steal_bet98"
    assert row.model_kind == "single_hand"
    assert row.horizon == 100
    assert row.discount == 1.0
    assert row.generated_candidates == 1
    assert row.kept_candidates == 1
    assert row.excluded_candidates == 0
    assert row.eligible_candidates == 1
    assert row.pareto_frontier_candidates >= 0
    assert row.minimum_villain_ev_candidates >= 0
    assert row.error is None


def test_summary_row_includes_format_version():
    batch = run_batch_scenario_analysis([_NUTS])
    assert batch.rows[0].format_version == "1"


def test_batch_row_positional_construction_is_backward_compatible():
    # format_version is appended last (default None), so the original positional
    # field order is preserved: model_kind stays the 3rd positional field.
    row = BatchScenarioRow(
        "sid",  # scenario_id
        "src.json",  # source_path
        "single_hand",  # model_kind
        10,  # horizon
        1.0,  # discount
        1,  # generated_candidates
        1,  # kept_candidates
        0,  # excluded_candidates
        1,  # eligible_candidates
        1,  # pareto_frontier_candidates
        1,  # minimum_villain_ev_candidates
        "cand",  # top_candidate_id
        None,  # top_candidate_sort_key
        49,  # top_candidate_t_deadline
        0.5,  # top_candidate_post_response_hero_ev_worst_diff
        True,  # top_candidate_detected_adaptation_is_at_least_baseline
        None,  # error
    )
    assert row.model_kind == "single_hand"
    assert row.error is None
    assert row.format_version is None


def test_model_kind_reflects_betting_tree_and_matrix_type():
    batch = run_batch_scenario_analysis(
        [_SCENARIOS / "range_equity_betting_tree_bet98.json"]
    )
    assert batch.rows[0].model_kind == "range_matrix:equity+betting_tree"


def test_ranking_populates_top_candidate_fields():
    config = BatchScenarioAnalysisConfig(
        analysis=RiverScenarioAnalysisConfig(ranking_criterion="t_deadline", ranking_top_k=1)
    )
    batch = run_batch_scenario_analysis([_NUTS], config)
    row = batch.rows[0]
    assert row.top_candidate_id is not None
    assert row.top_candidate_sort_key is not None
    assert row.top_candidate_t_deadline is not None


def test_no_ranking_leaves_top_candidate_none():
    batch = run_batch_scenario_analysis([_NUTS])
    assert batch.rows[0].top_candidate_id is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def _write_bad_scenario(tmp_path) -> Path:
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    return bad


def test_fail_fast_raises_on_bad_scenario(tmp_path):
    bad = _write_bad_scenario(tmp_path)
    with pytest.raises(ValueError, match="failed to analyse scenario"):
        run_batch_scenario_analysis([bad])


def test_continue_on_error_records_error_row(tmp_path):
    bad = _write_bad_scenario(tmp_path)
    config = BatchScenarioAnalysisConfig(continue_on_error=True)
    batch = run_batch_scenario_analysis([bad, _NUTS], config)
    assert batch.error_count == 1
    assert batch.ok_count == 1
    bad_row = batch.rows[0]
    assert bad_row.error is not None
    assert bad_row.scenario_id is None
    # The successful scenario keeps its detailed result.
    ok_row = batch.rows[1]
    assert ok_row.scenario_id == "nuts_chop_steal_bet98"
    assert ok_row.source_path in batch.results
    # A bad file in a temp dir outside the cwd must not leak its absolute path.
    assert bad_row.source_path == "bad.json"
    assert str(tmp_path) not in (bad_row.error or "")
    assert str(tmp_path) not in bad_row.source_path


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def batch():
    config = BatchScenarioAnalysisConfig(
        analysis=RiverScenarioAnalysisConfig(ranking_criterion="t_deadline", ranking_top_k=1)
    )
    return run_batch_scenario_analysis([_NUTS, _RANGE], config)


def test_batch_json_contains_rows_and_results(tmp_path, batch):
    path = tmp_path / "batch.json"
    write_batch_json(batch, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["summary_rows"]) == 2
    assert payload["summary_rows"][0]["scenario_id"] == "nuts_chop_steal_bet98"
    assert payload["summary_rows"][0]["format_version"] == "1"
    assert set(payload["scenario_results"]) == set(batch.results)


def test_batch_to_dict_matches_writer(batch):
    payload = batch_result_to_dict(batch)
    assert "summary_rows" in payload
    assert "scenario_results" in payload


def test_batch_csv_header_and_rows(tmp_path, batch):
    path = tmp_path / "batch.csv"
    write_batch_csv(batch, path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    # The file starts with the run-manifest comment line, then the header.
    assert rows[0][0].startswith("# run_manifest: ")
    assert "scenario_id" in rows[1]
    assert "top_candidate_t_deadline" in rows[1]
    assert len(rows) - 2 == len(batch.rows)


def test_batch_csv_includes_important_columns(tmp_path, batch):
    path = tmp_path / "batch.csv"
    write_batch_csv(batch, path)
    with path.open(encoding="utf-8", newline="") as handle:
        manifest_line = handle.readline()
        assert manifest_line.startswith("# run_manifest: ")
        header = next(csv.reader(handle))
    for column in (
        "format_version",
        "top_candidate_post_response_hero_ev_worst_diff",
        "top_candidate_detected_adaptation_is_at_least_baseline",
        "error",
    ):
        assert column in header


def test_batch_markdown_contains_table(tmp_path, batch):
    path = tmp_path / "batch.md"
    write_batch_markdown(batch, path)
    text = path.read_text(encoding="utf-8")
    assert "Batch Scenario Analysis Summary" in text
    assert "| scenario_id |" in text


def test_batch_markdown_has_overview_and_notes(tmp_path, batch):
    path = tmp_path / "batch.md"
    write_batch_markdown(batch, path)
    text = path.read_text(encoding="utf-8")
    assert "### Overview" in text
    assert f"- total scenarios: {len(batch.rows)}" in text
    assert f"- ok: {batch.ok_count}" in text
    assert f"- errors: {batch.error_count}" in text
    assert "### Notes" in text
    assert "not a new solver model" in text


def test_batch_markdown_includes_important_columns(tmp_path, batch):
    path = tmp_path / "batch.md"
    write_batch_markdown(batch, path)
    header = next(
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("| scenario_id")
    )
    for column in (
        "format_version",
        "minimum_villain_ev_candidates",
        "top_candidate_post_response_hero_ev_worst_diff",
        "top_candidate_detected_adaptation_is_at_least_baseline",
    ):
        assert column in header


def test_batch_markdown_formats_none_bool_and_float(tmp_path):
    # None -> "-", float -> 6 decimals, bool -> yes/no.
    none_row = _row_with(scenario_id="none_row", top_candidate_id=None)
    yes_row = _row_with(
        scenario_id="yes_row",
        top_candidate_post_response_hero_ev_worst_diff=0.123456789,
        top_candidate_detected_adaptation_is_at_least_baseline=True,
    )
    no_row = _row_with(
        scenario_id="no_row",
        top_candidate_detected_adaptation_is_at_least_baseline=False,
    )
    result = BatchScenarioAnalysisResult(rows=[none_row, yes_row, no_row], results={})
    path = tmp_path / "fmt.md"
    write_batch_markdown(result, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    none_line = next(line for line in lines if line.startswith("| none_row"))
    yes_line = next(line for line in lines if line.startswith("| yes_row"))
    no_line = next(line for line in lines if line.startswith("| no_row"))
    # top_candidate_id is None on the none_row, rendered as "-".
    assert " - " in none_line
    # float rounded to 6 decimals.
    assert "0.123457" in yes_line
    # discount is 1.0 on every _row_with, so a float column always has 6 decimals.
    assert "1.000000" in none_line
    assert "yes" in yes_line
    assert "no" in no_line


def _detection_batch():
    config = BatchScenarioAnalysisConfig(
        analysis=RiverScenarioAnalysisConfig(
            detection_log_likelihood_threshold=3.0,
            detection_occurrence_probability_per_opportunity=0.5,
        )
    )
    # The betting-tree scenario yields non-finite KL divergences under detection.
    return run_batch_scenario_analysis(
        [_SCENARIOS / "range_equity_betting_tree_bet98.json"], config
    )


def test_batch_lenient_json_keeps_infinity(tmp_path):
    path = tmp_path / "batch_lenient.json"
    write_batch_json(_detection_batch(), path)  # default strict=False
    assert "Infinity" in path.read_text(encoding="utf-8")


def test_batch_strict_json_has_no_non_finite(tmp_path):
    path = tmp_path / "batch_strict.json"
    write_batch_json(_detection_batch(), path, strict=True)
    text = path.read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert "NaN" not in text
    json.loads(text)  # parses as strict JSON


def test_parent_directory_auto_created(tmp_path, batch):
    path = tmp_path / "nested" / "deep" / "batch.json"
    write_batch_json(batch, path)
    assert path.exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_directory_input(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), str(_SCENARIOS)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "scenarios:" in completed.stdout
    assert "nuts_chop_steal_bet98" in completed.stdout


def test_cli_writes_outputs(tmp_path):
    json_path = tmp_path / "b.json"
    csv_path = tmp_path / "b.csv"
    md_path = tmp_path / "b.md"
    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            str(_NUTS),
            str(_RANGE),
            "--rank-by",
            "t_deadline",
            "--output-json",
            str(json_path),
            "--output-csv",
            str(csv_path),
            "--output-markdown",
            str(md_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "saved JSON to" in completed.stdout
    assert "saved CSV to" in completed.stdout
    assert "saved Markdown to" in completed.stdout
    assert json_path.exists()
    assert csv_path.exists()
    assert md_path.exists()


def test_cli_physical_hand_conversion_records_batch_and_scenario_manifest(tmp_path):
    json_path = tmp_path / "batch_physical.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            str(_NUTS),
            "--no-markdown",
            "--detection-log-likelihood-threshold",
            "3.0",
            "--detection-occurrence-probability-per-opportunity",
            "0.5",
            "--detection-comparable-spot-occurrence-probability-per-physical-hand",
            "0.25",
            "--output-json",
            str(json_path),
            "--strict-json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "saved JSON to" in completed.stdout
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    parameter_name = "comparable_spot_occurrence_probability_per_physical_hand"
    assert payload["manifest"]["parameters"][parameter_name] == 0.25
    scenario_payload = next(iter(payload["scenario_results"].values()))
    assert scenario_payload["manifest"]["parameters"][parameter_name] == 0.25
    detection_config = scenario_payload["analysis_report"][
        "detection_configuration"
    ]
    assert detection_config[parameter_name] == 0.25
    rows = scenario_payload["analysis_report"]["candidate_rows"]
    assert any(row["t_detect_estimated_physical_hands"] for row in rows)


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


def test_cli_strict_json_without_output_json_succeeds(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), str(_NUTS), "--strict-json"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "scenarios:" in completed.stdout
    assert "saved " not in completed.stdout


def test_cli_no_markdown_runs(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), str(_NUTS), "--no-markdown"],
        capture_output=True,
        text=True,
        check=True,
    )
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


# ---------------------------------------------------------------------------
# Markdown escaping
# ---------------------------------------------------------------------------


def _row_with(**overrides) -> BatchScenarioRow:
    base = dict(
        scenario_id=None,
        source_path="x.json",
        format_version="1",
        model_kind="single_hand",
        horizon=10,
        discount=1.0,
        generated_candidates=1,
        kept_candidates=1,
        excluded_candidates=0,
        eligible_candidates=1,
        pareto_frontier_candidates=1,
        minimum_villain_ev_candidates=1,
        top_candidate_id=None,
        top_candidate_sort_key=None,
        top_candidate_t_deadline=None,
        top_candidate_post_response_hero_ev_worst_diff=None,
        top_candidate_detected_adaptation_is_at_least_baseline=None,
        error=None,
    )
    base.update(overrides)
    return BatchScenarioRow(**base)


def test_batch_markdown_escapes_pipe_and_newline(tmp_path):
    row = _row_with(scenario_id="bad|id", error="oops | pipe\nand newline")
    batch = BatchScenarioAnalysisResult(rows=[row], results={})
    path = tmp_path / "esc.md"
    write_batch_markdown(batch, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    data_lines = [line for line in lines if line.startswith("| bad")]
    assert len(data_lines) == 1
    data = data_lines[0]
    # Pipes from the values are escaped, and the newline is flattened to a space.
    assert "bad\\|id" in data
    assert "oops \\| pipe and newline" in data
    # The header row and this data row have the same number of cell separators,
    # so the literal pipes did not add extra columns.
    header = next(line for line in lines if line.startswith("| scenario_id"))
    assert data.count(" | ") == header.count(" | ")


def test_cli_help_documents_options():
    # The batch CLI --help exits 0 and documents its options (each used to have no
    # help text); the output names the horizon/discount overrides and the outputs.
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    out = completed.stdout
    assert "--horizon" in out
    assert "--discount" in out
    assert "--rank-by" in out
    assert "--output-json" in out
    assert "--output-csv" in out
    assert "--output-markdown" in out
    assert (
        "--detection-comparable-spot-occurrence-probability-per-physical-hand"
        in out
    )
    # A couple of the help strings themselves, not just the option names.
    assert "every scenario" in out
    assert "one CSV summary row per scenario" in out
