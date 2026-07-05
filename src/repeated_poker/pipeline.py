"""High-level orchestration of the candidate-analysis stages.

``run_candidate_analysis_pipeline`` wires the existing building blocks together
for a small abstract game: candidate generation, optional pre-filtering, fixed
profile comparison, consolidated analysis reporting, and optional Markdown
rendering.  It is an orchestration helper, not a new solver: it adds no new
analysis maths, no CLI, and no file output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

from .analysis_report import CandidateAnalysisReport, build_candidate_analysis_report
from .candidate_filters import CandidateFilterResult, filter_candidates
from .candidates import (
    DEFAULT_MAX_CANDIDATES,
    HeroStrategyCandidate,
    generate_candidate_library,
)
from .comparison import CandidateComparisonReport, compare_candidates
from .detection import (
    DEFAULT_MAX_DETECTION_TERMINALS,
    DETECTION_METHOD_LOCAL_V0,
    TerminalReveals,
)
from .exact_response import DEFAULT_MAX_PURE_STRATEGIES
from .game import GameTree, HeroStrategy, VillainStrategy
from .repeated import DEFAULT_MAX_HORIZON, RESPONSE_MODE_WORST
from .summary import format_candidate_analysis_markdown, validate_markdown_max_rows


@dataclass(frozen=True)
class CandidateGenerationConfig:
    """Configuration for the candidate-generation stage.

    ``max_simultaneous_info_sets`` selects the candidate families: ``1`` (the
    default) generates only single-information-set shifts, exactly as before;
    ``2`` additionally generates the simultaneous two-information-set shift
    candidates (M2-T2). Only ``1`` and ``2`` are supported for now.
    """

    shift_amounts: Sequence[float]
    max_simultaneous_info_sets: int = 1
    max_candidates: int = DEFAULT_MAX_CANDIDATES


@dataclass(frozen=True)
class CandidateFilterConfig:
    """Configuration for the optional pre-comparison filter stage."""

    allowed_info_sets: Optional[Set[str]] = None
    max_l1_distance: Optional[float] = None
    min_required_observations: Optional[int] = None


@dataclass(frozen=True)
class CandidateAnalysisPipelineResult:
    """The artefacts produced by the candidate-analysis pipeline.

    ``generated_candidates`` is the full pre-filter list; ``filter_result.kept``
    is the subset carried into comparison; ``comparison_report`` and
    ``analysis_report`` cover the kept candidates only, in input order.
    ``markdown_summary`` is ``None`` when Markdown rendering is disabled.
    """

    generated_candidates: List[HeroStrategyCandidate]
    filter_result: CandidateFilterResult
    comparison_report: CandidateComparisonReport
    analysis_report: CandidateAnalysisReport
    markdown_summary: Optional[str]


def run_candidate_analysis_pipeline(
    tree: GameTree,
    baseline_hero_strategy: HeroStrategy,
    baseline_villain_strategy: VillainStrategy,
    *,
    generation: CandidateGenerationConfig,
    horizon: int,
    discount: float = 1.0,
    response_mode: str = RESPONSE_MODE_WORST,
    profit_tolerance: float = 0.0,
    max_selection_l1_distance: Optional[float] = None,
    detection_log_likelihood_threshold: Optional[float] = None,
    detection_occurrence_probability_per_opportunity: Optional[float] = None,
    detection_comparable_spot_occurrence_probability_per_physical_hand: Optional[
        float
    ] = None,
    detection_method: str = DETECTION_METHOD_LOCAL_V0,
    detection_observation_model: Optional[str] = None,
    terminal_reveals: Optional[TerminalReveals] = None,
    filtering: Optional[CandidateFilterConfig] = None,
    render_markdown: bool = True,
    markdown_max_rows: Optional[int] = None,
    tolerance: float = 1e-9,
    max_horizon: int = DEFAULT_MAX_HORIZON,
    max_detection_terminals: int = DEFAULT_MAX_DETECTION_TERMINALS,
    max_pure_strategies: int = DEFAULT_MAX_PURE_STRATEGIES,
    allow_negative_residual: bool = False,
) -> CandidateAnalysisPipelineResult:
    """Run candidate generation through to an optional Markdown summary.

    Stages, in order: ``generate_candidate_library`` -> ``filter_candidates``
    (no-op when ``filtering`` is ``None``) -> ``compare_candidates`` over the
    kept candidates -> ``build_candidate_analysis_report`` (detection integrated
    when ``detection_log_likelihood_threshold`` is given) ->
    ``format_candidate_analysis_markdown`` (only when ``render_markdown``).

    Most validation is delegated to the underlying stage APIs.  The pipeline
    additionally rejects an empty ``generation.shift_amounts``, a non-boolean
    ``render_markdown``, an invalid ``markdown_max_rows``, and a filter that
    requests ``min_required_observations`` without a
    ``detection_log_likelihood_threshold``. ``allow_negative_residual`` is
    forwarded to comparison-time tree validation for model families whose third
    terminal slot is a signed accounting residual.
    """

    if not generation.shift_amounts:
        raise ValueError("generation.shift_amounts must not be empty")
    if not isinstance(render_markdown, bool):
        raise ValueError(f"render_markdown must be a bool, got {render_markdown!r}")
    validate_markdown_max_rows(markdown_max_rows)
    if (
        filtering is not None
        and filtering.min_required_observations is not None
        and detection_log_likelihood_threshold is None
    ):
        raise ValueError(
            "detection_log_likelihood_threshold is required when "
            "filtering.min_required_observations is given"
        )

    generated_candidates = generate_candidate_library(
        tree,
        baseline_hero_strategy,
        generation.shift_amounts,
        max_simultaneous_info_sets=generation.max_simultaneous_info_sets,
        tolerance=tolerance,
        max_candidates=generation.max_candidates,
    )

    if filtering is None:
        filter_result = filter_candidates(generated_candidates, tolerance=tolerance)
    else:
        filter_result = filter_candidates(
            generated_candidates,
            allowed_info_sets=filtering.allowed_info_sets,
            max_l1_distance=filtering.max_l1_distance,
            min_required_observations=filtering.min_required_observations,
            baseline_hero_strategy=baseline_hero_strategy,
            tree=tree,
            baseline_villain_strategy=baseline_villain_strategy,
            detection_log_likelihood_threshold=detection_log_likelihood_threshold,
            detection_method=detection_method,
            detection_observation_model=detection_observation_model,
            terminal_reveals=terminal_reveals,
            max_detection_terminals=max_detection_terminals,
            tolerance=tolerance,
        )

    comparison_report = compare_candidates(
        tree,
        baseline_hero_strategy,
        baseline_villain_strategy,
        filter_result.kept,
        tolerance=tolerance,
        max_pure_strategies=max_pure_strategies,
        allow_negative_residual=allow_negative_residual,
    )

    analysis_report = build_candidate_analysis_report(
        comparison_report,
        horizon=horizon,
        discount=discount,
        response_mode=response_mode,
        profit_tolerance=profit_tolerance,
        max_l1_distance=max_selection_l1_distance,
        tolerance=tolerance,
        max_horizon=max_horizon,
        baseline_hero_strategy=baseline_hero_strategy,
        tree=tree,
        baseline_villain_strategy=baseline_villain_strategy,
        detection_log_likelihood_threshold=detection_log_likelihood_threshold,
        detection_occurrence_probability_per_opportunity=(
            detection_occurrence_probability_per_opportunity
        ),
        detection_comparable_spot_occurrence_probability_per_physical_hand=(
            detection_comparable_spot_occurrence_probability_per_physical_hand
        ),
        detection_method=detection_method,
        detection_observation_model=detection_observation_model,
        terminal_reveals=terminal_reveals,
        max_detection_terminals=max_detection_terminals,
    )

    markdown_summary: Optional[str]
    if render_markdown:
        markdown_summary = format_candidate_analysis_markdown(
            analysis_report, max_rows=markdown_max_rows
        )
    else:
        markdown_summary = None

    return CandidateAnalysisPipelineResult(
        generated_candidates=generated_candidates,
        filter_result=filter_result,
        comparison_report=comparison_report,
        analysis_report=analysis_report,
        markdown_summary=markdown_summary,
    )
