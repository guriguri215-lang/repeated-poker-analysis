"""Write a scenario analysis result to JSON, Markdown, or CSV files.

These are thin persistence helpers for
:class:`~repeated_poker.scenario_pipeline.RiverScenarioAnalysisResult` and
:class:`~repeated_poker.stt_pushfold_pipeline.SttPushFoldAnalysisResult`. They
reuse the existing ``to_dict`` / ``summary_rows`` / ``markdown_summary`` outputs
rather than recomputing anything, so the on-disk content matches what the
pipeline already produces.

Each writer creates missing parent directories and overwrites an existing file
at ``path``. Output is UTF-8. By default the JSON writers use the standard
library ``json`` module, which may serialise a non-finite
``detection_kl_divergence_nats`` as ``Infinity``. This is not strict RFC 8259
JSON, and JavaScript ``JSON.parse`` will reject it. Passing ``strict=True`` to a
JSON writer instead maps every non-finite float (``inf`` / ``-inf`` / ``nan``)
to ``null`` and emits RFC 8259-compatible JSON (``json.dumps(..., allow_nan=False)``).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

from .scenario_batch import BATCH_ROW_COLUMNS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .run_manifest import RunManifest
    from .scenario_batch import BatchScenarioAnalysisResult
    from .scenario_pipeline import RiverScenarioAnalysisResult
    from .scenario_validation import ScenarioValidationResult
    from .stt_pushfold_pipeline import SttPushFoldAnalysisResult

PathLike = Union[str, Path]

# CSV columns, in order. Each name is a key of ``CandidateAnalysisRow.to_dict``.
_CSV_COLUMNS: List[str] = [
    "candidate_id",
    "info_set",
    "source_action",
    "target_action",
    "shift_amount",
    "num_shifts",
    "info_sets",
    "l1_distance",
    "observation_distance",
    "is_eligible",
    "exclusion_reasons",
    "fixed_hero_ev",
    "fixed_villain_ev",
    "post_response_hero_ev_worst",
    "post_response_hero_ev_worst_diff",
    "is_ev_observation_deadline_pareto_candidate",
    "t_deadline",
    "t_detect_estimated_opportunities",
    "t_detect_estimated_physical_hands",
    "detected_adaptation_delta_from_baseline",
    "detected_adaptation_is_at_least_baseline",
    "detection_kl_per_hand_nats",
    "detection_tv_per_hand",
    "detection_baseline_impossible_mass_per_hand",
    "t_detect_hands",
    "detection_time_basis",
]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _strict_json_safe(value):
    """Recursively replace non-finite floats with ``None`` for strict JSON.

    ``inf`` / ``-inf`` / ``nan`` floats become ``None`` (JSON ``null``); finite
    floats, ints, bools, strings, and ``None`` pass through unchanged. Mappings
    and sequences are processed recursively. ``bool`` is left intact (it is an
    ``int`` subclass, not a float).
    """

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _strict_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_safe(item) for item in value]
    return value


def _dump_json(payload, strict: bool) -> str:
    """Serialise ``payload`` as indented JSON.

    With ``strict=True`` the payload is first passed through
    :func:`_strict_json_safe` and dumped with ``allow_nan=False`` so the result
    is RFC 8259-compatible. With ``strict=False`` the default (lenient) behaviour
    is kept, which may emit ``Infinity`` / ``NaN``.
    """

    if strict:
        return json.dumps(_strict_json_safe(payload), indent=2, allow_nan=False)
    return json.dumps(payload, indent=2)


def _manifest_dict(manifest: Optional["RunManifest"]) -> Optional[dict]:
    """Return the manifest payload, or ``None`` when a run carries none."""

    return manifest.to_dict() if manifest is not None else None


def _manifest_csv_comment(manifest: "RunManifest") -> str:
    """One ``# run_manifest: {...}`` comment line for a CSV export.

    The JSON after the prefix is strict (non-finite floats become ``null``), so
    the line parses with ``json.loads`` after stripping the prefix. CSV readers
    should skip lines starting with ``#`` (for example pandas ``comment="#"``).
    """

    payload = _strict_json_safe(manifest.to_dict())
    return "# run_manifest: " + json.dumps(payload, allow_nan=False)


def _manifest_markdown_lines(manifest: "RunManifest") -> List[str]:
    """Render the manifest as a Markdown section (one bullet per field)."""

    manifest_dict = manifest.to_dict()
    parameters = manifest_dict.pop("parameters")
    lines = ["### Run manifest", ""]
    for key, value in manifest_dict.items():
        lines.append(f"- {key}: {_markdown_cell(value)}")
    if parameters is not None:
        rendered = ", ".join(
            f"{key}={_markdown_cell(value)}" for key, value in parameters.items()
        )
        lines.append(f"- parameters: {rendered}")
    return lines


def analysis_result_to_dict(result: "RiverScenarioAnalysisResult") -> dict:
    """Return the full JSON-serialisable payload written by the JSON exporter.

    Includes the run's reproducibility ``manifest`` (``null`` on a manually
    constructed result without one).
    """

    counts = result.pipeline_result.filter_result.summary_counts
    payload = {
        "format_version": result.scenario.format_version,
        "scenario_id": result.scenario_id,
        "manifest": _manifest_dict(result.manifest),
        "selected_horizon": result.horizon,
        "selected_discount": result.discount,
        "build_metadata": result.build.metadata,
        "counts": {
            "generated": len(result.pipeline_result.generated_candidates),
            "kept": counts.kept,
            "excluded": counts.excluded,
        },
        "filter_result": result.pipeline_result.filter_result.to_dict(),
        "analysis_report": result.pipeline_result.analysis_report.to_dict(),
        "markdown_summary": result.markdown_summary,
        "ranking": None,
    }
    if result.ranking_result is not None:
        ranking = result.ranking_result
        payload["ranking"] = {
            "criterion": ranking.criterion,
            "descending": ranking.descending,
            "ranked_rows": [
                {
                    "rank": ranked.rank,
                    "candidate_id": ranked.row.candidate_id,
                    "sort_key": ranked.sort_key,
                }
                for ranked in ranking.ranked_rows
            ],
        }
    return payload


def write_analysis_json(
    result: "RiverScenarioAnalysisResult", path: PathLike, strict: bool = False
) -> None:
    """Write the analysis result as indented JSON to ``path``.

    With ``strict=True`` non-finite floats are mapped to ``null`` and the output
    is RFC 8259-compatible; the default ``strict=False`` keeps the lenient
    behaviour (which may emit ``Infinity`` / ``NaN``).
    """

    target = Path(path)
    _ensure_parent(target)
    payload = analysis_result_to_dict(result)
    target.write_text(_dump_json(payload, strict), encoding="utf-8")


def write_analysis_markdown(result: "RiverScenarioAnalysisResult", path: PathLike) -> None:
    """Write the Markdown summary to ``path``.

    Raises :class:`ValueError` if the result has no Markdown summary (Markdown
    rendering was disabled); the caller should enable Markdown when requesting a
    Markdown file.  When the result carries a run manifest, a ``Run manifest``
    section is appended after the summary.
    """

    if result.markdown_summary is None:
        raise ValueError(
            "result has no markdown_summary to write; run the analysis with "
            "Markdown rendering enabled before writing a Markdown file"
        )
    target = Path(path)
    _ensure_parent(target)
    text = result.markdown_summary
    if result.manifest is not None:
        lines = [text.rstrip("\n"), ""]
        lines.extend(_manifest_markdown_lines(result.manifest))
        lines.append("")
        text = "\n".join(lines)
    target.write_text(text, encoding="utf-8")


def _csv_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "; ".join(str(item) for item in value)
    return str(value)


def _markdown_cell(value) -> str:
    """Render a value as a human-readable Markdown table cell.

    Formatting, aimed at side-by-side scenario comparison rather than at machine
    parsing (the CSV/JSON exports stay machine-friendly):

    * ``None`` becomes ``-``;
    * ``bool`` becomes ``yes`` / ``no``;
    * ``float`` is shown with 6 decimals;
    * a list / tuple is comma-separated;
    * everything else uses ``str``.

    The result then escapes ``|`` (which would otherwise split the cell) and
    flattens carriage returns / newlines to spaces, so a multi-line or
    pipe-containing value cannot break the table.
    """

    if value is None:
        text = "-"
    elif isinstance(value, bool):
        text = "yes" if value else "no"
    elif isinstance(value, float):
        text = f"{value:.6f}"
    elif isinstance(value, (list, tuple)):
        text = ", ".join(str(item) for item in value)
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def write_analysis_csv(result: "RiverScenarioAnalysisResult", path: PathLike) -> None:
    """Write one CSV row per candidate (in report order) to ``path``.

    When the result carries a run manifest, the file starts with a single
    ``# run_manifest: {...}`` comment line before the header row; skip lines
    starting with ``#`` when parsing (for example pandas ``comment="#"``).
    """

    target = Path(path)
    _ensure_parent(target)
    rows = result.pipeline_result.analysis_report.summary_rows()
    with target.open("w", encoding="utf-8", newline="") as handle:
        if result.manifest is not None:
            handle.write(_manifest_csv_comment(result.manifest) + "\r\n")
        writer = csv.writer(handle)
        writer.writerow(_CSV_COLUMNS)
        for row in rows:
            flat = _flatten_row_for_csv(row)
            writer.writerow([_csv_cell(flat.get(column)) for column in _CSV_COLUMNS])


def _flatten_row_for_csv(row: dict) -> dict:
    """Add flat, single-cell columns derived from a row's ``shifts`` list.

    ``num_shifts`` is the number of changed information sets and ``info_sets`` is
    the list of them (rendered ``;``-separated by :func:`_csv_cell`), so a CSV
    consumer can see a multi-shift candidate's information sets without parsing
    ``candidate_id`` (which also encodes the full combination).
    """

    shifts = row.get("shifts") or []
    flat = dict(row)
    flat["num_shifts"] = len(shifts)
    flat["info_sets"] = [component["info_set"] for component in shifts]
    return flat


# ---------------------------------------------------------------------------
# Batch exporters
# ---------------------------------------------------------------------------

# Batch row columns rendered in the Markdown comparison table, in display order.
# A wider set than the compact stdout table so a Markdown report is self-contained
# for side-by-side scenario comparison.
_BATCH_MARKDOWN_COLUMNS: List[str] = [
    "scenario_id",
    "format_version",
    "model_kind",
    "horizon",
    "discount",
    "generated_candidates",
    "kept_candidates",
    "eligible_candidates",
    "pareto_frontier_candidates",
    "minimum_villain_ev_candidates",
    "top_candidate_id",
    "top_candidate_t_deadline",
    "top_candidate_post_response_hero_ev_worst_diff",
    "top_candidate_detected_adaptation_is_at_least_baseline",
    "error",
]


def batch_result_to_dict(batch: "BatchScenarioAnalysisResult") -> dict:
    """Return the JSON payload for a batch run.

    The payload keeps the comparison ``summary_rows`` plus the full per-scenario
    ``scenario_results`` (each via :func:`analysis_result_to_dict`, keyed by
    display path) so a batch file is self-contained for later comparison. Failed
    scenarios appear only in ``summary_rows`` (with an ``error``). The
    batch-level ``manifest`` carries the shared reproducibility metadata; each
    per-scenario result additionally embeds its own manifest (with the scenario
    file's SHA-256 and resolved parameters).
    """

    return {
        "manifest": _manifest_dict(batch.manifest),
        "summary_rows": [row.to_dict() for row in batch.rows],
        "scenario_results": {
            path: analysis_result_to_dict(result)
            for path, result in batch.results.items()
        },
    }


def write_batch_json(
    batch: "BatchScenarioAnalysisResult", path: PathLike, strict: bool = False
) -> None:
    """Write the batch summary rows and per-scenario results as JSON to ``path``.

    With ``strict=True`` non-finite floats are mapped to ``null`` and the output
    is RFC 8259-compatible; the default ``strict=False`` keeps the lenient
    behaviour (which may emit ``Infinity`` / ``NaN``).
    """

    target = Path(path)
    _ensure_parent(target)
    target.write_text(_dump_json(batch_result_to_dict(batch), strict), encoding="utf-8")


def write_batch_csv(batch: "BatchScenarioAnalysisResult", path: PathLike) -> None:
    """Write one CSV row per scenario (summary rows only) to ``path``.

    When the batch carries a run manifest, the file starts with a single
    ``# run_manifest: {...}`` comment line before the header row; skip lines
    starting with ``#`` when parsing (for example pandas ``comment="#"``).
    """

    target = Path(path)
    _ensure_parent(target)
    with target.open("w", encoding="utf-8", newline="") as handle:
        if batch.manifest is not None:
            handle.write(_manifest_csv_comment(batch.manifest) + "\r\n")
        writer = csv.writer(handle)
        writer.writerow(BATCH_ROW_COLUMNS)
        for row in batch.rows:
            row_dict = row.to_dict()
            writer.writerow([_csv_cell(row_dict.get(column)) for column in BATCH_ROW_COLUMNS])


def write_validation_json(
    validation: "ScenarioValidationResult", path: PathLike, strict: bool = False
) -> None:
    """Write the scenario validation rows as JSON to ``path``.

    Reuses :func:`ScenarioValidationResult.to_dict` and the shared
    :func:`_dump_json` writer, so ``strict=True`` maps non-finite floats to
    ``null`` and emits RFC 8259-compatible JSON, while the default ``strict=False``
    keeps the lenient behaviour.
    """

    target = Path(path)
    _ensure_parent(target)
    target.write_text(_dump_json(validation.to_dict(), strict), encoding="utf-8")


_BATCH_MARKDOWN_NOTES: List[str] = [
    "This batch summary is a reporting helper, not a new solver model; it only "
    "renders the existing per-scenario summary rows side by side.",
    "The `top_candidate_*` columns are only populated when ranking is enabled "
    "(for example `--rank-by t_deadline`); otherwise they show `-`.",
    "`top_candidate_detected_adaptation_is_at_least_baseline` is only available "
    "when detection output exists; otherwise it shows `-`.",
    "Strict JSON output is separate from this Markdown (and from the CSV); it "
    "only affects the JSON export.",
]


def write_batch_markdown(batch: "BatchScenarioAnalysisResult", path: PathLike) -> None:
    """Write a GitHub-flavoured Markdown comparison report to ``path``.

    The report has an overview (total / ok / error counts), a comparison table of
    the per-scenario summary rows, and a short notes section. This is an
    analysis/reporting helper, not a new solver model: it only renders the
    existing rows side by side.
    """

    target = Path(path)
    _ensure_parent(target)
    lines = [
        "## Batch Scenario Analysis Summary",
        "",
        "### Overview",
        "",
        f"- total scenarios: {len(batch.rows)}",
        f"- ok: {batch.ok_count}",
        f"- errors: {batch.error_count}",
        "",
        "### Comparison",
        "",
        "| " + " | ".join(_BATCH_MARKDOWN_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in _BATCH_MARKDOWN_COLUMNS) + " |",
    ]
    for row in batch.rows:
        row_dict = row.to_dict()
        cells = [_markdown_cell(row_dict.get(column)) for column in _BATCH_MARKDOWN_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    if batch.manifest is not None:
        lines.append("")
        lines.extend(_manifest_markdown_lines(batch.manifest))
    lines.extend(["", "### Notes", ""])
    lines.extend(f"- {note}" for note in _BATCH_MARKDOWN_NOTES)
    lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")
