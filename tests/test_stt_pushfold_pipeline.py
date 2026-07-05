"""Tests for the STT push/fold analysis runner."""

import json
import subprocess
import sys
from pathlib import Path

from repeated_poker import (
    DETECTION_METHOD_REACH_WEIGHTED_V1,
    OBSERVATION_MODEL_SHOWDOWN_REVEAL,
    SttPushFoldAnalysisConfig,
    SttPushFoldAnalysisResult,
    load_stt_pushfold_scenario_json,
    run_stt_pushfold_analysis,
    write_analysis_json,
)

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / "examples" / "stt_pushfold_2x2.json"
_SCRIPT = _ROOT / "scripts" / "run_stt_pushfold_analysis.py"


def test_run_from_path_succeeds():
    result = run_stt_pushfold_analysis(_SAMPLE)

    assert isinstance(result, SttPushFoldAnalysisResult)
    assert result.scenario_id == "stt_pushfold_2x2"
    assert result.horizon == 50
    assert result.discount == 1.0
    assert result.pipeline_result.generated_candidates
    assert result.pipeline_result.analysis_report.rows
    assert "Candidate Analysis Summary" in result.markdown_summary


def test_run_from_parsed_scenario_succeeds():
    scenario = load_stt_pushfold_scenario_json(_SAMPLE)
    result = run_stt_pushfold_analysis(scenario)

    assert result.scenario is scenario
    assert result.scenario.format_version == "stt_pushfold-1"


def test_existing_positional_config_slots_are_preserved():
    config = SttPushFoldAnalysisConfig(
        None,
        None,
        "best",
        0.25,
        0.5,
        3.0,
        0.4,
        "local_v0",
        None,
        123,
        ["SB"],
        0.7,
        10,
        False,
        12,
        "t_deadline",
        True,
        True,
        2,
        1e-8,
        321,
    )

    assert config.detection_method == "local_v0"
    assert config.max_detection_terminals == 123
    assert config.filter_allowed_info_sets == ["SB"]
    assert config.markdown is False
    assert config.max_pure_strategies == 321
    assert (
        config.detection_comparable_spot_occurrence_probability_per_physical_hand
        is None
    )


def test_manifest_contains_stt_format_version_and_prize_unit(tmp_path):
    result = run_stt_pushfold_analysis(_SAMPLE)
    path = tmp_path / "stt.json"
    write_analysis_json(result, path, strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["format_version"] == "stt_pushfold-1"
    assert payload["manifest"]["scenario_format_version"] == "stt_pushfold-1"
    assert (
        payload["manifest"]["parameters"]["value_unit"]
        == "modelled_tournament_prize_ev_delta"
    )
    assert payload["build_metadata"]["baseline_villain_source"] == "explicit"


def test_local_detection_runs_without_error():
    result = run_stt_pushfold_analysis(
        _SAMPLE,
        SttPushFoldAnalysisConfig(
            detection_log_likelihood_threshold=3.0,
            detection_occurrence_probability_per_opportunity=0.5,
        ),
    )

    row = result.pipeline_result.analysis_report.rows[0].to_dict()
    assert "t_detect_estimated_opportunities" in row
    assert "detection_kl_divergence_nats" in row


def test_programmatic_physical_hand_conversion_runs_through_stt_runner():
    result = run_stt_pushfold_analysis(
        _SAMPLE,
        SttPushFoldAnalysisConfig(
            detection_log_likelihood_threshold=3.0,
            detection_occurrence_probability_per_opportunity=0.5,
            detection_comparable_spot_occurrence_probability_per_physical_hand=0.25,
            markdown=False,
        ),
    )

    report = result.pipeline_result.analysis_report
    assert (
        report.detection_configuration.comparable_spot_occurrence_probability_per_physical_hand
        == 0.25
    )
    assert result.manifest.parameters[
        "comparable_spot_occurrence_probability_per_physical_hand"
    ] == 0.25
    assert any(row.t_detect_estimated_physical_hands for row in report.rows)


def test_reach_weighted_showdown_filter_runs_through_stt_runner():
    result = run_stt_pushfold_analysis(
        _SAMPLE,
        SttPushFoldAnalysisConfig(
            detection_log_likelihood_threshold=3.0,
            detection_method=DETECTION_METHOD_REACH_WEIGHTED_V1,
            detection_observation_model=OBSERVATION_MODEL_SHOWDOWN_REVEAL,
            filter_min_required_observations=1_000_000,
            markdown=False,
        ),
    )

    counts = result.pipeline_result.filter_result.summary_counts
    assert counts.total == len(result.pipeline_result.generated_candidates)
    assert counts.excluded > 0
    assert len(result.pipeline_result.analysis_report.rows) == counts.kept
    assert result.pipeline_result.analysis_report.detection_configuration.method == (
        DETECTION_METHOD_REACH_WEIGHTED_V1
    )
    assert result.pipeline_result.analysis_report.detection_configuration.observation_model == (
        OBSERVATION_MODEL_SHOWDOWN_REVEAL
    )


def test_missing_shift_amounts_raise_clear_error():
    scenario = load_stt_pushfold_scenario_json(_SAMPLE)
    scenario = type(scenario)(
        **{
            **scenario.__dict__,
            "shift_amounts": None,
        }
    )

    try:
        run_stt_pushfold_analysis(scenario)
    except ValueError as exc:
        assert "shift_amounts" in str(exc)
    else:
        raise AssertionError("expected missing shift_amounts to raise")


def test_cli_reports_expected_lines():
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT), str(_SAMPLE)],
        capture_output=True,
        text=True,
        check=True,
    )
    stdout = completed.stdout

    for fragment in (
        "stt_pushfold_2x2",
        "stt_pushfold-1",
        "modelled tournament prize EV delta",
        "Candidate Analysis Summary",
        "T_deadline",
    ):
        assert fragment in stdout


def test_cli_writes_json_markdown_and_csv(tmp_path):
    json_path = tmp_path / "stt.json"
    md_path = tmp_path / "stt.md"
    csv_path = tmp_path / "stt.csv"

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
            "--strict-json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "saved JSON to" in completed.stdout
    assert "saved Markdown to" in completed.stdout
    assert "saved CSV to" in completed.stdout
    assert json_path.exists()
    assert md_path.exists()
    assert csv_path.exists()


def test_cli_physical_hand_conversion_writes_json_manifest(tmp_path):
    json_path = tmp_path / "stt_physical.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            str(_SAMPLE),
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
    parameters = payload["manifest"]["parameters"]
    parameter_name = "comparable_spot_occurrence_probability_per_physical_hand"
    assert parameters[parameter_name] == 0.25
    assert parameters["value_unit"] == "modelled_tournament_prize_ev_delta"
    detection_config = payload["analysis_report"]["detection_configuration"]
    assert detection_config[parameter_name] == 0.25
    rows = payload["analysis_report"]["candidate_rows"]
    assert any(row["t_detect_estimated_physical_hands"] for row in rows)
