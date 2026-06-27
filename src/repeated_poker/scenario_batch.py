"""Run several river scenarios through the single-scenario analysis runner.

This is a thin batch wrapper over
:func:`repeated_poker.scenario_pipeline.run_river_scenario_analysis`. It adds no
new analysis maths: it loads several scenario files, runs the existing
per-scenario pipeline on each, keeps the detailed results, and builds one
comparison summary row per scenario so a set of scenarios can be scanned side by
side.

Inputs may be a single path (a directory, whose ``*.json`` files are read in
filename order, or a single scenario file) or a sequence of paths (each a
directory or a file, expanded in the given order). A scenario's ``source_path``
is shown as a cwd-relative POSIX path when it is inside the current working
directory, and as just its file name otherwise, so summary output and error
messages never leak an absolute local path (see ``display_scenario_path``).

Errors are handled per the ``continue_on_error`` flag: fail-fast (the default)
re-raises the first failure with the offending display path, while
continue-on-error records the error on that scenario's summary row and proceeds,
keeping the successful scenarios' detailed results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .scenario_pipeline import (
    RiverScenarioAnalysisConfig,
    RiverScenarioAnalysisResult,
    run_river_scenario_analysis,
)

PathLike = Union[str, Path]
BatchInput = Union[PathLike, Sequence[PathLike]]


@dataclass(frozen=True)
class BatchScenarioAnalysisConfig:
    """Configuration for :func:`run_batch_scenario_analysis`.

    ``analysis`` is forwarded unchanged to each per-scenario run.
    ``continue_on_error`` switches between fail-fast (raise on the first failing
    scenario) and continue-on-error (record the error on the row and proceed).
    """

    analysis: RiverScenarioAnalysisConfig = field(
        default_factory=RiverScenarioAnalysisConfig
    )
    continue_on_error: bool = False


@dataclass(frozen=True)
class BatchScenarioRow:
    """One comparison summary row per scenario.

    All analysis fields are ``None`` when the scenario failed; ``error`` is then a
    short ``"<ExceptionType>: <message>"`` string and is ``None`` on success.
    """

    scenario_id: Optional[str]
    source_path: str
    format_version: Optional[str]
    model_kind: Optional[str]
    horizon: Optional[int]
    discount: Optional[float]
    generated_candidates: Optional[int]
    kept_candidates: Optional[int]
    excluded_candidates: Optional[int]
    eligible_candidates: Optional[int]
    pareto_frontier_candidates: Optional[int]
    minimum_villain_ev_candidates: Optional[int]
    top_candidate_id: Optional[str]
    top_candidate_sort_key: Optional[Any]
    top_candidate_t_deadline: Optional[int]
    top_candidate_post_response_hero_ev_worst_diff: Optional[float]
    top_candidate_detected_adaptation_is_at_least_baseline: Optional[bool]
    error: Optional[str]

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "source_path": self.source_path,
            "format_version": self.format_version,
            "model_kind": self.model_kind,
            "horizon": self.horizon,
            "discount": self.discount,
            "generated_candidates": self.generated_candidates,
            "kept_candidates": self.kept_candidates,
            "excluded_candidates": self.excluded_candidates,
            "eligible_candidates": self.eligible_candidates,
            "pareto_frontier_candidates": self.pareto_frontier_candidates,
            "minimum_villain_ev_candidates": self.minimum_villain_ev_candidates,
            "top_candidate_id": self.top_candidate_id,
            "top_candidate_sort_key": self.top_candidate_sort_key,
            "top_candidate_t_deadline": self.top_candidate_t_deadline,
            "top_candidate_post_response_hero_ev_worst_diff": (
                self.top_candidate_post_response_hero_ev_worst_diff
            ),
            "top_candidate_detected_adaptation_is_at_least_baseline": (
                self.top_candidate_detected_adaptation_is_at_least_baseline
            ),
            "error": self.error,
        }


# CSV / table column order, matching ``BatchScenarioRow.to_dict`` keys.
BATCH_ROW_COLUMNS: List[str] = list(
    BatchScenarioRow(
        scenario_id=None,
        source_path="",
        format_version=None,
        model_kind=None,
        horizon=None,
        discount=None,
        generated_candidates=None,
        kept_candidates=None,
        excluded_candidates=None,
        eligible_candidates=None,
        pareto_frontier_candidates=None,
        minimum_villain_ev_candidates=None,
        top_candidate_id=None,
        top_candidate_sort_key=None,
        top_candidate_t_deadline=None,
        top_candidate_post_response_hero_ev_worst_diff=None,
        top_candidate_detected_adaptation_is_at_least_baseline=None,
        error=None,
    ).to_dict()
)


@dataclass(frozen=True)
class BatchScenarioAnalysisResult:
    """The summary rows and the detailed results of a batch run.

    ``rows`` has one entry per input scenario, in input order. ``results`` maps
    each successful scenario's display path to its
    :class:`RiverScenarioAnalysisResult`; failed scenarios are absent from
    ``results`` but still present in ``rows`` with an ``error``.
    """

    rows: List[BatchScenarioRow]
    results: Dict[str, RiverScenarioAnalysisResult]

    @property
    def ok_count(self) -> int:
        return sum(1 for row in self.rows if row.error is None)

    @property
    def error_count(self) -> int:
        return sum(1 for row in self.rows if row.error is not None)


def display_scenario_path(path: Path) -> str:
    """Return a stable, non-leaking display string for ``path``.

    A path inside the current working directory is shown as a cwd-relative POSIX
    path (so a directory input given as an absolute path still yields a
    repo-relative ``source_path``). A path outside the cwd is shown as just its
    file name, so summary output and error messages do not leak an absolute local
    path such as ``C:/Users/<name>/...``.
    """

    try:
        resolved = path.resolve()
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except (ValueError, OSError):
        return path.name


def expand_scenario_inputs(inputs: BatchInput) -> List[Path]:
    if isinstance(inputs, (str, Path)):
        items: Sequence[PathLike] = [inputs]
    else:
        items = list(inputs)
    if not items:
        raise ValueError("inputs must contain at least one path")

    expanded: List[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.json"), key=lambda p: p.name))
        else:
            expanded.append(path)
    if not expanded:
        raise ValueError("no scenario files found in the given inputs")
    return expanded


def model_kind_from_metadata(metadata: dict) -> Optional[str]:
    """Derive a compact model-kind label from a scenario build's ``metadata``.

    Returns the ``mode`` directly for single-hand and Hero-range-only scenarios
    (``"single_hand"`` / ``"range"``), and a ``range_matrix:<matrix_type>`` label
    for matrix scenarios, with a ``+betting_tree`` suffix when a betting tree is
    present (for example ``"range_matrix:equity+betting_tree"``).
    """

    mode = metadata.get("mode")
    if mode != "range_matrix":
        return mode
    matrix_type = metadata.get("matrix_type")
    suffix = "+betting_tree" if metadata.get("betting_tree") else ""
    return f"range_matrix:{matrix_type}{suffix}"


def _success_row(display_path: str, result: RiverScenarioAnalysisResult) -> BatchScenarioRow:
    filter_counts = result.pipeline_result.filter_result.summary_counts
    selection_counts = result.pipeline_result.analysis_report.summary_counts
    ranking = result.ranking_result
    top = ranking.ranked_rows[0] if (ranking and ranking.ranked_rows) else None
    return BatchScenarioRow(
        scenario_id=result.scenario_id,
        source_path=display_path,
        format_version=result.scenario.format_version,
        model_kind=model_kind_from_metadata(result.build.metadata),
        horizon=result.horizon,
        discount=result.discount,
        generated_candidates=len(result.pipeline_result.generated_candidates),
        kept_candidates=filter_counts.kept,
        excluded_candidates=filter_counts.excluded,
        eligible_candidates=selection_counts.eligible,
        pareto_frontier_candidates=selection_counts.pareto_frontier,
        minimum_villain_ev_candidates=selection_counts.minimum_villain_ev,
        top_candidate_id=top.row.candidate_id if top else None,
        top_candidate_sort_key=top.sort_key if top else None,
        top_candidate_t_deadline=top.row.t_deadline if top else None,
        top_candidate_post_response_hero_ev_worst_diff=(
            top.row.post_response_hero_ev_worst_diff if top else None
        ),
        top_candidate_detected_adaptation_is_at_least_baseline=(
            top.row.detected_adaptation_is_at_least_baseline if top else None
        ),
        error=None,
    )


def _error_row(display_path: str, exc: BaseException) -> BatchScenarioRow:
    return BatchScenarioRow(
        scenario_id=None,
        source_path=display_path,
        format_version=None,
        model_kind=None,
        horizon=None,
        discount=None,
        generated_candidates=None,
        kept_candidates=None,
        excluded_candidates=None,
        eligible_candidates=None,
        pareto_frontier_candidates=None,
        minimum_villain_ev_candidates=None,
        top_candidate_id=None,
        top_candidate_sort_key=None,
        top_candidate_t_deadline=None,
        top_candidate_post_response_hero_ev_worst_diff=None,
        top_candidate_detected_adaptation_is_at_least_baseline=None,
        error=f"{type(exc).__name__}: {exc}",
    )


def run_batch_scenario_analysis(
    inputs: BatchInput,
    config: Optional[BatchScenarioAnalysisConfig] = None,
) -> BatchScenarioAnalysisResult:
    """Run each scenario through ``run_river_scenario_analysis`` and summarise.

    See the module docstring for input expansion and error handling. Returns a
    :class:`BatchScenarioAnalysisResult` whose ``rows`` are in input order.
    """

    config = config or BatchScenarioAnalysisConfig()
    paths = expand_scenario_inputs(inputs)

    rows: List[BatchScenarioRow] = []
    results: Dict[str, RiverScenarioAnalysisResult] = {}
    for path in paths:
        display = display_scenario_path(path)
        try:
            result = run_river_scenario_analysis(path, config.analysis)
        except Exception as exc:  # noqa: BLE001 - surfaced via row or re-raised
            if not config.continue_on_error:
                raise ValueError(
                    f"failed to analyse scenario {display!r}: {exc}"
                ) from exc
            rows.append(_error_row(display, exc))
            continue
        results[display] = result
        rows.append(_success_row(display, result))

    return BatchScenarioAnalysisResult(rows=rows, results=results)
