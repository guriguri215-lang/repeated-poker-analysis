"""Diagnostic ranking / sorting of candidate analysis rows.

``rank_candidate_rows`` reorders the rows of a
:class:`~repeated_poker.analysis_report.CandidateAnalysisReport` by a chosen
diagnostic criterion so an analyst can scan them more easily. It is a
*presentation / diagnostic* helper: it does not auto-select a candidate and
makes no optimality claim.

Rows whose criterion value is ``None`` (for example, ``t_deadline`` when no
opportunity is safe, or detection fields when detection is disabled) are always
placed last, regardless of sort direction. Ties keep the original report order
(stable sort). The input report is never mutated.

A future version could let a ranking feed directly into
``format_candidate_analysis_markdown``; that integration is intentionally out of
scope here, and the Markdown renderer is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from .analysis_report import CandidateAnalysisReport, CandidateAnalysisRow

# Ranking criteria. Each value is also the ``CandidateAnalysisRow`` attribute
# name used as the sort key.
RANK_BY_POST_RESPONSE_HERO_EV_WORST_DIFF = "post_response_hero_ev_worst_diff"
RANK_BY_FIXED_VILLAIN_EV = "fixed_villain_ev"
RANK_BY_L1_DISTANCE = "l1_distance"
RANK_BY_T_DEADLINE = "t_deadline"
RANK_BY_T_DETECT = "t_detect_estimated_opportunities"
RANK_BY_DETECTED_ADAPTATION_DELTA = "detected_adaptation_delta_from_baseline"

# Criterion -> default ``descending`` (True means higher value ranks first).
_DEFAULT_DESCENDING = {
    RANK_BY_POST_RESPONSE_HERO_EV_WORST_DIFF: True,
    RANK_BY_FIXED_VILLAIN_EV: False,
    RANK_BY_L1_DISTANCE: False,
    RANK_BY_T_DEADLINE: True,
    RANK_BY_T_DETECT: True,
    RANK_BY_DETECTED_ADAPTATION_DELTA: True,
}


@dataclass(frozen=True)
class RankedCandidateRow:
    """One ranked row: its 1-based ``rank``, the row, and its sort key."""

    rank: int
    row: CandidateAnalysisRow
    sort_key: Any


@dataclass(frozen=True)
class CandidateRankingResult:
    """The ordered rows under a chosen criterion and direction."""

    criterion: str
    descending: bool
    ranked_rows: List[RankedCandidateRow]


def rank_candidate_rows(
    report: CandidateAnalysisReport,
    criterion: str,
    *,
    descending: Optional[bool] = None,
    eligible_only: bool = False,
    top_k: Optional[int] = None,
) -> CandidateRankingResult:
    """Rank the report rows by ``criterion`` for easier inspection.

    ``descending`` defaults to the criterion's natural "better first" direction
    (for example, higher ``post_response_hero_ev_worst_diff`` or lower
    ``fixed_villain_ev``). Rows whose criterion value is ``None`` are always
    placed last. Ties keep the original report order. With ``eligible_only`` only
    eligible rows are ranked, and ``top_k`` limits how many ranked rows are
    returned (``top_k=0`` yields an empty result). The report is not modified.
    """

    if criterion not in _DEFAULT_DESCENDING:
        raise ValueError(
            f"unknown criterion {criterion!r}; expected one of "
            f"{sorted(_DEFAULT_DESCENDING)}"
        )
    if descending is not None and not isinstance(descending, bool):
        raise ValueError(f"descending must be None or a bool, got {descending!r}")
    if not isinstance(eligible_only, bool):
        raise ValueError(f"eligible_only must be a bool, got {eligible_only!r}")
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError(f"top_k must be None or a non-negative int, got {top_k!r}")
        if top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {top_k}")

    resolved_descending = (
        _DEFAULT_DESCENDING[criterion] if descending is None else descending
    )

    rows = [
        row
        for row in report.rows
        if not eligible_only or row.is_eligible
    ]

    present = [(row, getattr(row, criterion)) for row in rows]
    with_value = [(row, value) for row, value in present if value is not None]
    without_value = [(row, value) for row, value in present if value is None]

    # ``sorted`` is stable, so ties keep the original order. Negating the key for
    # the descending case keeps ties stable too (the criteria are all numeric).
    if resolved_descending:
        with_value.sort(key=lambda pair: -pair[1])
    else:
        with_value.sort(key=lambda pair: pair[1])

    ordered = with_value + without_value
    if top_k is not None:
        ordered = ordered[:top_k]

    ranked_rows = [
        RankedCandidateRow(rank=index + 1, row=row, sort_key=value)
        for index, (row, value) in enumerate(ordered)
    ]
    return CandidateRankingResult(
        criterion=criterion,
        descending=resolved_descending,
        ranked_rows=ranked_rows,
    )
