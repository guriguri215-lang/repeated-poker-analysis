"""Write a scenario analysis result to JSON, Markdown, or CSV files.

These are thin persistence helpers for
:class:`~repeated_poker.scenario_pipeline.RiverScenarioAnalysisResult`. They
reuse the existing ``to_dict`` / ``summary_rows`` / ``markdown_summary`` outputs
rather than recomputing anything, so the on-disk content matches what the
pipeline already produces.

Each writer creates missing parent directories and overwrites an existing file
at ``path``. Output is UTF-8. The JSON writer uses the standard library ``json``
module, which by default may serialise a non-finite
``detection_kl_divergence_nats`` as ``Infinity``. This is not strict RFC 8259
JSON, and JavaScript ``JSON.parse`` will reject it. Strict JSON output is out of
scope for v1.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

from .scenario_batch import BATCH_ROW_COLUMNS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .scenario_batch import BatchScenarioAnalysisResult
    from .scenario_pipeline import RiverScenarioAnalysisResult

PathLike = Union[str, Path]

# CSV columns, in order. Each name is a key of ``CandidateAnalysisRow.to_dict``.
_CSV_COLUMNS: List[str] = [
    "candidate_id",
    "info_set",
    "source_action",
    "target_action",
    "shift_amount",
    "l1_distance",
    "is_eligible",
    "exclusion_reasons",
    "fixed_hero_ev",
    "fixed_villain_ev",
    "post_response_hero_ev_worst",
    "post_response_hero_ev_worst_diff",
    "t_deadline",
    "t_detect_estimated_opportunities",
    "detected_adaptation_delta_from_baseline",
    "detected_adaptation_is_at_least_baseline",
]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def analysis_result_to_dict(result: "RiverScenarioAnalysisResult") -> dict:
    """Return the full JSON-serialisable payload written by the JSON exporter."""

    counts = result.pipeline_result.filter_result.summary_counts
    payload = {
        "scenario_id": result.scenario_id,
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


def write_analysis_json(result: "RiverScenarioAnalysisResult", path: PathLike) -> None:
    """Write the analysis result as indented JSON to ``path``."""

    target = Path(path)
    _ensure_parent(target)
    payload = analysis_result_to_dict(result)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_analysis_markdown(result: "RiverScenarioAnalysisResult", path: PathLike) -> None:
    """Write the Markdown summary to ``path``.

    Raises :class:`ValueError` if the result has no Markdown summary (Markdown
    rendering was disabled); the caller should enable Markdown when requesting a
    Markdown file.
    """

    if result.markdown_summary is None:
        raise ValueError(
            "result has no markdown_summary to write; run the analysis with "
            "Markdown rendering enabled before writing a Markdown file"
        )
    target = Path(path)
    _ensure_parent(target)
    target.write_text(result.markdown_summary, encoding="utf-8")


def _csv_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "; ".join(str(item) for item in value)
    return str(value)


def _markdown_cell(value) -> str:
    """Render a value as a Markdown table cell.

    Escapes ``|`` (which would otherwise split the cell) and flattens newlines to
    spaces so a multi-line or pipe-containing value cannot break the table.
    """

    text = _csv_cell(value)
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def write_analysis_csv(result: "RiverScenarioAnalysisResult", path: PathLike) -> None:
    """Write one CSV row per candidate (in report order) to ``path``."""

    target = Path(path)
    _ensure_parent(target)
    rows = result.pipeline_result.analysis_report.summary_rows()
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_CSV_COLUMNS)
        for row in rows:
            writer.writerow([_csv_cell(row.get(column)) for column in _CSV_COLUMNS])


# ---------------------------------------------------------------------------
# Batch exporters
# ---------------------------------------------------------------------------

# Subset of the batch row columns rendered in the compact Markdown table.
_BATCH_MARKDOWN_COLUMNS: List[str] = [
    "scenario_id",
    "model_kind",
    "horizon",
    "generated_candidates",
    "kept_candidates",
    "eligible_candidates",
    "pareto_frontier_candidates",
    "top_candidate_id",
    "top_candidate_t_deadline",
    "error",
]


def batch_result_to_dict(batch: "BatchScenarioAnalysisResult") -> dict:
    """Return the JSON payload for a batch run.

    The payload keeps the comparison ``summary_rows`` plus the full per-scenario
    ``scenario_results`` (each via :func:`analysis_result_to_dict`, keyed by
    display path) so a batch file is self-contained for later comparison. Failed
    scenarios appear only in ``summary_rows`` (with an ``error``).
    """

    return {
        "summary_rows": [row.to_dict() for row in batch.rows],
        "scenario_results": {
            path: analysis_result_to_dict(result)
            for path, result in batch.results.items()
        },
    }


def write_batch_json(batch: "BatchScenarioAnalysisResult", path: PathLike) -> None:
    """Write the batch summary rows and per-scenario results as JSON to ``path``."""

    target = Path(path)
    _ensure_parent(target)
    target.write_text(json.dumps(batch_result_to_dict(batch), indent=2), encoding="utf-8")


def write_batch_csv(batch: "BatchScenarioAnalysisResult", path: PathLike) -> None:
    """Write one CSV row per scenario (summary rows only) to ``path``."""

    target = Path(path)
    _ensure_parent(target)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(BATCH_ROW_COLUMNS)
        for row in batch.rows:
            row_dict = row.to_dict()
            writer.writerow([_csv_cell(row_dict.get(column)) for column in BATCH_ROW_COLUMNS])


def write_batch_markdown(batch: "BatchScenarioAnalysisResult", path: PathLike) -> None:
    """Write a compact GitHub-flavoured Markdown comparison table to ``path``.

    This is an analysis/reporting helper, not a new solver model: it only renders
    the existing per-scenario summary rows side by side.
    """

    target = Path(path)
    _ensure_parent(target)
    lines = [
        "## Batch Scenario Analysis Summary",
        "",
        f"- scenarios: {len(batch.rows)} (ok: {batch.ok_count}, errors: {batch.error_count})",
        "",
        "| " + " | ".join(_BATCH_MARKDOWN_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in _BATCH_MARKDOWN_COLUMNS) + " |",
    ]
    for row in batch.rows:
        row_dict = row.to_dict()
        cells = [_markdown_cell(row_dict.get(column)) for column in _BATCH_MARKDOWN_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")
