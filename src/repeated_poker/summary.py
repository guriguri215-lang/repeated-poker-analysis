"""Human-readable Markdown rendering of a CandidateAnalysisReport.

This is a *presentation-only* layer: it turns an existing
:class:`~repeated_poker.analysis_report.CandidateAnalysisReport` into a Markdown
string for quick human inspection.  It does not recompute or change any
analysis result, it adds no CLI, and it never writes files.
"""

from __future__ import annotations

import math
from typing import List, Optional

from .analysis_report import CandidateAnalysisReport, CandidateAnalysisRow

_FLOAT_DECIMALS = 6

# Table columns: (header, accessor returning a preformatted cell string).
_TABLE_HEADERS = [
    "candidate_id",
    "info_set",
    "shift",
    "l1_distance",
    "observation_distance",
    "fixed_hero_ev",
    "post_response_hero_ev_worst",
    "post_response_hero_ev_worst_diff",
    "robustly_profitable",
    "is_eligible",
    "is_ev_obs_deadline_pareto",
    "t_deadline",
    "t_detect_estimated_opportunities",
    "detected_adaptation_delta_from_baseline",
    "detected_adaptation_is_at_least_baseline",
    "exclusion_reasons",
]


def _escape(text: str) -> str:
    """Escape characters that would break a Markdown table cell."""

    return text.replace("|", "\\|")


def _format_value(value) -> str:
    """Format a scalar/list cell value per the display rules."""

    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return "nan"
        return f"{value:.{_FLOAT_DECIMALS}f}"
    if isinstance(value, list):
        if not value:
            return "-"
        return ", ".join(_escape(str(item)) for item in value)
    if isinstance(value, str):
        return _escape(value)
    return _escape(str(value))


def _format_component(component: dict) -> str:
    return (
        f"{_escape(str(component['info_set']))}: "
        f"{_escape(str(component['source_action']))} -> "
        f"{_escape(str(component['target_action']))} "
        f"({component['shift_amount']:+g})"
    )


def _format_info_set(row: CandidateAnalysisRow) -> str:
    """Information set cell: the single info set, or the joined multi-shift sets."""

    if row.info_set is not None:
        return _format_value(row.info_set)
    return _escape(" + ".join(str(component["info_set"]) for component in row.shifts))


def _format_shift(row: CandidateAnalysisRow) -> str:
    """Shift cell: the single shift, or the joined multi-shift components."""

    if row.shift_amount is not None:
        return (
            f"{_escape(str(row.source_action))} -> {_escape(str(row.target_action))} "
            f"({row.shift_amount:+g})"
        )
    return " + ".join(_format_component(component) for component in row.shifts)


def _row_cells(row: CandidateAnalysisRow) -> List[str]:
    return [
        _format_value(row.candidate_id),
        _format_info_set(row),
        _format_shift(row),
        _format_value(row.l1_distance),
        _format_value(row.observation_distance),
        _format_value(row.fixed_hero_ev),
        _format_value(row.post_response_hero_ev_worst),
        _format_value(row.post_response_hero_ev_worst_diff),
        _format_value(row.robustly_profitable),
        _format_value(row.is_eligible),
        _format_value(row.is_ev_observation_deadline_pareto_candidate),
        _format_value(row.t_deadline),
        _format_value(row.t_detect_estimated_opportunities),
        _format_value(row.detected_adaptation_delta_from_baseline),
        _format_value(row.detected_adaptation_is_at_least_baseline),
        _format_value(row.exclusion_reasons),
    ]


def _markdown_table_row(cells: List[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def validate_markdown_max_rows(max_rows: Optional[int]) -> None:
    """Validate a Markdown ``max_rows`` argument (``None`` or non-negative int).

    Public helper so that callers such as the pipeline can apply the same rule
    without depending on a private function.
    """

    if max_rows is None:
        return
    if isinstance(max_rows, bool) or not isinstance(max_rows, int):
        raise ValueError(f"max_rows must be a non-negative integer or None, got {max_rows!r}")
    if max_rows < 0:
        raise ValueError(f"max_rows must be non-negative, got {max_rows}")


def format_candidate_analysis_markdown(
    report: CandidateAnalysisReport, max_rows: Optional[int] = None
) -> str:
    """Render ``report`` as a human-readable Markdown string.

    ``max_rows`` (``None`` or a non-negative integer) caps how many candidate
    rows are printed; the remaining count is reported as a trailing note.  This
    function only formats existing results; it changes nothing in the analysis.
    """

    validate_markdown_max_rows(max_rows)

    selection = report.selection_configuration
    deadline = report.deadline_configuration
    detection = report.detection_configuration
    counts = report.summary_counts

    lines: List[str] = []
    lines.append("## Candidate Analysis Summary")
    lines.append("")

    lines.append("### Configurations")
    lines.append("")
    lines.append("**Selection**")
    lines.append(f"- profit_tolerance: {_format_value(selection.profit_tolerance)}")
    lines.append(f"- max_l1_distance: {_format_value(selection.max_l1_distance)}")
    lines.append(f"- tolerance: {_format_value(selection.tolerance)}")
    lines.append("")
    lines.append("**Deadline**")
    lines.append(f"- horizon: {_format_value(deadline.horizon)}")
    lines.append(f"- discount: {_format_value(deadline.discount)}")
    lines.append(f"- response_mode: {_format_value(deadline.response_mode)}")
    lines.append(f"- max_horizon: {_format_value(deadline.max_horizon)}")
    lines.append("")
    lines.append("**Detection**")
    lines.append(f"- enabled: {_format_value(detection.enabled)}")
    lines.append(
        f"- log_likelihood_threshold: "
        f"{_format_value(detection.log_likelihood_threshold)}"
    )
    lines.append(
        f"- occurrence_probability_per_opportunity: "
        f"{_format_value(detection.occurrence_probability_per_opportunity)}"
    )
    lines.append(f"- tolerance: {_format_value(detection.tolerance)}")
    lines.append("")

    lines.append("### Summary Counts")
    lines.append(f"- total: {_format_value(counts.total)}")
    lines.append(f"- eligible: {_format_value(counts.eligible)}")
    lines.append(f"- excluded: {_format_value(counts.excluded)}")
    lines.append(f"- minimum_villain_ev: {_format_value(counts.minimum_villain_ev)}")
    lines.append(f"- pareto_frontier: {_format_value(counts.pareto_frontier)}")
    lines.append(
        "- ev_observation_deadline_pareto_frontier: "
        f"{_format_value(counts.ev_observation_deadline_pareto_frontier)}"
    )
    lines.append("")

    lines.append("### Candidate Rows")
    lines.append("")
    lines.append(_markdown_table_row(_TABLE_HEADERS))
    lines.append(_markdown_table_row(["---"] * len(_TABLE_HEADERS)))

    total_rows = len(report.rows)
    shown_rows = report.rows if max_rows is None else report.rows[:max_rows]
    for row in shown_rows:
        lines.append(_markdown_table_row(_row_cells(row)))

    remaining = total_rows - len(shown_rows)
    if remaining > 0:
        lines.append("")
        lines.append(f"... {remaining} more rows not shown.")
    lines.append("")

    lines.append("### Notes")
    lines.append("- `T_deadline` is an economic adaptation deadline.")
    lines.append(
        "- `T_detect` is a local observable-distribution sensitivity estimate."
    )
    lines.append(
        "- `t_detect_is_no_later_than_t_deadline` is only a timing comparison and "
        "not an economic-safety statement."
    )
    lines.append(
        "- Use `detected_adaptation_is_at_least_baseline` and "
        "`detected_adaptation_delta_from_baseline` for the economic read at the "
        "estimated detection timing."
    )
    lines.append(
        "- The summary does not model full tree reach, real learning speed, or "
        "actual opponent adaptation."
    )
    lines.append(
        "- `observation_distance` is the observable-distribution (total-variation) "
        "distance at the changed information set(s); it is not a strategy-space "
        "distance."
    )
    lines.append(
        "- `is_ev_obs_deadline_pareto` marks the trade-off frontier over "
        "post-response worst-case Hero EV (higher better), observation distance "
        "(lower better), and `T_deadline` (higher better); it is a trade-off "
        "surface, not an equilibrium or optimality claim."
    )
    lines.append(
        "- For a multi-shift candidate the `info_set` / `shift` cells list every "
        "changed information set; `candidate_id` also encodes the full combination."
    )

    return "\n".join(lines)
