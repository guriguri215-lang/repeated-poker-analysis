"""Tests for the reproducibility run manifest."""

import csv
import hashlib
import json
import re
from pathlib import Path

import pytest

import repeated_poker
from repeated_poker import (
    PACKAGE_VERSION,
    RESPONSE_MODE_WORST,
    RiverScenarioAnalysisConfig,
    load_river_scenario_json,
    run_batch_scenario_analysis,
    run_river_scenario_analysis,
    sha256_of_file,
    write_analysis_csv,
    write_analysis_json,
    write_analysis_markdown,
    write_batch_csv,
    write_batch_json,
    write_batch_markdown,
)

_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE = _ROOT / "examples" / "scenarios" / "nuts_chop_steal_bet98.json"

_TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@pytest.fixture(scope="module")
def result():
    return run_river_scenario_analysis(_SAMPLE)


@pytest.fixture(scope="module")
def batch():
    return run_batch_scenario_analysis(_SAMPLE)


# ---------------------------------------------------------------------------
# Version and hashing helpers
# ---------------------------------------------------------------------------


def test_package_version_matches_pyproject():
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
    assert match is not None
    assert PACKAGE_VERSION == match.group(1)
    assert repeated_poker.__version__ == PACKAGE_VERSION


def test_sha256_of_file_matches_hashlib(tmp_path):
    path = tmp_path / "payload.json"
    path.write_bytes(b'{"format_version": "1"}')
    assert sha256_of_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Manifest on a single-scenario run
# ---------------------------------------------------------------------------


def test_run_from_file_carries_full_manifest(result):
    manifest = result.manifest
    assert manifest is not None
    assert manifest.scenario_sha256 == sha256_of_file(_SAMPLE)
    assert manifest.scenario_format_version == "1"
    assert manifest.package_version == PACKAGE_VERSION
    assert _TIMESTAMP_PATTERN.match(manifest.timestamp_utc)
    # Best effort: a hex commit in a git checkout, None otherwise.
    assert manifest.git_commit is None or _COMMIT_PATTERN.match(manifest.git_commit)


def test_manifest_records_effective_parameters(result):
    parameters = result.manifest.parameters
    assert parameters["horizon"] == result.horizon
    assert parameters["discount"] == result.discount
    assert parameters["response_mode"] == RESPONSE_MODE_WORST
    assert parameters["tolerance"] == 1e-9
    assert parameters["ranking_criterion"] is None


def test_manifest_parameters_follow_overrides():
    overridden = run_river_scenario_analysis(
        _SAMPLE, RiverScenarioAnalysisConfig(horizon=7, markdown=False)
    )
    assert overridden.manifest.parameters["horizon"] == 7
    assert overridden.horizon == 7


def test_run_from_memory_has_null_sha256():
    scenario = load_river_scenario_json(_SAMPLE)
    result = run_river_scenario_analysis(
        scenario, RiverScenarioAnalysisConfig(markdown=False)
    )
    assert result.manifest is not None
    assert result.manifest.scenario_sha256 is None
    assert result.manifest.scenario_format_version == "1"


def test_result_to_dict_contains_manifest(result):
    payload = result.to_dict()
    assert payload["manifest"]["package_version"] == PACKAGE_VERSION
    assert payload["manifest"]["scenario_sha256"] == sha256_of_file(_SAMPLE)


# ---------------------------------------------------------------------------
# Single-scenario exports
# ---------------------------------------------------------------------------


def test_json_export_contains_manifest(tmp_path, result):
    path = tmp_path / "out.json"
    write_analysis_json(result, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = payload["manifest"]
    assert manifest["scenario_sha256"] == sha256_of_file(_SAMPLE)
    assert manifest["scenario_format_version"] == "1"
    assert manifest["package_version"] == PACKAGE_VERSION
    assert manifest["parameters"]["horizon"] == result.horizon


def test_strict_json_export_keeps_manifest(tmp_path, result):
    path = tmp_path / "strict.json"
    write_analysis_json(result, path, strict=True)
    text = path.read_text(encoding="utf-8")
    assert "Infinity" not in text
    payload = json.loads(text)
    assert payload["manifest"]["scenario_sha256"] == sha256_of_file(_SAMPLE)


def test_csv_export_has_manifest_comment_line(tmp_path, result):
    path = tmp_path / "out.csv"
    write_analysis_csv(result, path)
    with path.open(encoding="utf-8", newline="") as handle:
        first = handle.readline().rstrip("\r\n")
        header = handle.readline()
    assert first.startswith("# run_manifest: ")
    manifest = json.loads(first.removeprefix("# run_manifest: "))
    assert manifest["scenario_sha256"] == sha256_of_file(_SAMPLE)
    assert header.startswith("candidate_id,")


def test_csv_export_rows_parse_after_manifest_line(tmp_path, result):
    path = tmp_path / "out.csv"
    write_analysis_csv(result, path)
    with path.open(encoding="utf-8", newline="") as handle:
        handle.readline()  # skip the manifest comment line
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(result.pipeline_result.analysis_report.rows)


def test_markdown_export_has_manifest_section(tmp_path, result):
    path = tmp_path / "out.md"
    write_analysis_markdown(result, path)
    text = path.read_text(encoding="utf-8")
    assert "### Run manifest" in text
    assert sha256_of_file(_SAMPLE) in text
    assert f"- package_version: {PACKAGE_VERSION}" in text


# ---------------------------------------------------------------------------
# Batch exports
# ---------------------------------------------------------------------------


def test_batch_run_carries_batch_level_manifest(batch):
    manifest = batch.manifest
    assert manifest is not None
    # Scenario-specific fields live on the per-scenario manifests instead.
    assert manifest.scenario_sha256 is None
    assert manifest.scenario_format_version is None
    assert manifest.package_version == PACKAGE_VERSION
    assert manifest.parameters["continue_on_error"] is False
    # The per-scenario result still carries its own full manifest.
    (scenario_result,) = batch.results.values()
    assert scenario_result.manifest.scenario_sha256 == sha256_of_file(_SAMPLE)


def test_batch_json_export_contains_manifests(tmp_path, batch):
    path = tmp_path / "batch.json"
    write_batch_json(batch, path, strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["manifest"]["package_version"] == PACKAGE_VERSION
    assert payload["manifest"]["scenario_sha256"] is None
    (scenario_payload,) = payload["scenario_results"].values()
    assert scenario_payload["manifest"]["scenario_sha256"] == sha256_of_file(_SAMPLE)


def test_batch_csv_export_has_manifest_comment_line(tmp_path, batch):
    path = tmp_path / "batch.csv"
    write_batch_csv(batch, path)
    with path.open(encoding="utf-8", newline="") as handle:
        first = handle.readline().rstrip("\r\n")
        header = handle.readline()
    assert first.startswith("# run_manifest: ")
    manifest = json.loads(first.removeprefix("# run_manifest: "))
    assert manifest["package_version"] == PACKAGE_VERSION
    assert header.startswith("scenario_id,")


def test_batch_markdown_export_has_manifest_section(tmp_path, batch):
    path = tmp_path / "batch.md"
    write_batch_markdown(batch, path)
    text = path.read_text(encoding="utf-8")
    assert "### Run manifest" in text
    assert f"- package_version: {PACKAGE_VERSION}" in text
