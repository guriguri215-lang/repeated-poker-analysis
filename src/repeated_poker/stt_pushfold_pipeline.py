"""Run an STT push/fold JSON scenario through the analysis pipeline.

This module adapts :mod:`repeated_poker.stt_pushfold` to the existing
candidate-analysis pipeline. It adds no new solver math: after the STT builder
has produced a ``GameTree`` and baseline strategies, candidate generation,
exact response, repeated-game timing, detection, summaries, and export all use
the shared pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from .detection import (
    DEFAULT_MAX_DETECTION_TERMINALS,
    DETECTION_METHOD_LOCAL_V0,
    resolve_detection_observation_model,
)
from .exact_response import DEFAULT_MAX_PURE_STRATEGIES
from .pipeline import (
    CandidateAnalysisPipelineResult,
    CandidateFilterConfig,
    CandidateGenerationConfig,
    run_candidate_analysis_pipeline,
)
from .ranking import CandidateRankingResult, rank_candidate_rows
from .repeated import RESPONSE_MODE_WORST
from .run_manifest import RunManifest, build_run_manifest
from .stt_pushfold import (
    SttPushFoldBuildResult,
    SttPushFoldScenario,
    build_stt_pushfold_game,
    load_stt_pushfold_scenario_json,
)

ScenarioInput = Union[SttPushFoldScenario, str, Path]


@dataclass(frozen=True)
class SttPushFoldAnalysisConfig:
    """Optional overrides for :func:`run_stt_pushfold_analysis`."""

    horizon: Optional[int] = None
    discount: Optional[float] = None
    response_mode: str = RESPONSE_MODE_WORST
    profit_tolerance: float = 0.0
    max_selection_l1_distance: Optional[float] = None
    detection_log_likelihood_threshold: Optional[float] = None
    detection_occurrence_probability_per_opportunity: Optional[float] = None
    detection_comparable_spot_occurrence_probability_per_physical_hand: Optional[
        float
    ] = None
    detection_method: str = DETECTION_METHOD_LOCAL_V0
    detection_observation_model: Optional[str] = None
    max_detection_terminals: int = DEFAULT_MAX_DETECTION_TERMINALS
    filter_allowed_info_sets: Optional[List[str]] = None
    filter_max_l1_distance: Optional[float] = None
    filter_min_required_observations: Optional[int] = None
    markdown: bool = True
    markdown_max_rows: Optional[int] = None
    ranking_criterion: Optional[str] = None
    ranking_descending: Optional[bool] = None
    ranking_eligible_only: bool = False
    ranking_top_k: Optional[int] = None
    tolerance: float = 1e-9
    max_pure_strategies: int = DEFAULT_MAX_PURE_STRATEGIES


@dataclass(frozen=True)
class SttPushFoldAnalysisResult:
    """Everything produced by :func:`run_stt_pushfold_analysis`."""

    scenario: SttPushFoldScenario
    build: SttPushFoldBuildResult
    horizon: int
    discount: float
    pipeline_result: CandidateAnalysisPipelineResult
    ranking_result: Optional[CandidateRankingResult] = None
    markdown_summary: Optional[str] = field(default=None)
    manifest: Optional[RunManifest] = field(default=None)

    @property
    def scenario_id(self) -> str:
        return self.scenario.scenario_id

    def to_dict(self) -> dict:
        """Return a small JSON-serialisable summary of the run."""

        counts = self.pipeline_result.filter_result.summary_counts
        return {
            "format_version": self.scenario.format_version,
            "scenario_id": self.scenario_id,
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "horizon": self.horizon,
            "discount": self.discount,
            "generated": len(self.pipeline_result.generated_candidates),
            "kept": counts.kept,
            "excluded": counts.excluded,
            "report": self.pipeline_result.analysis_report.to_dict(),
            "ranking_criterion": (
                self.ranking_result.criterion if self.ranking_result else None
            ),
        }


def _resolve_scenario(scenario: ScenarioInput) -> SttPushFoldScenario:
    if isinstance(scenario, SttPushFoldScenario):
        return scenario
    if isinstance(scenario, (str, Path)):
        return load_stt_pushfold_scenario_json(scenario)
    raise TypeError(
        "scenario must be a SttPushFoldScenario, a path string, or a Path, "
        f"got {type(scenario).__name__}"
    )


def _resolve_horizon(
    scenario: SttPushFoldScenario, config: SttPushFoldAnalysisConfig
) -> int:
    if config.horizon is not None:
        if isinstance(config.horizon, bool) or not isinstance(config.horizon, int):
            raise ValueError(f"horizon must be an int, got {config.horizon!r}")
        if config.horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {config.horizon}")
        return config.horizon
    if scenario.repeated is not None and scenario.repeated.horizon is not None:
        return scenario.repeated.horizon
    raise ValueError(
        "no horizon available: the scenario has no repeated.horizon and no "
        "explicit config.horizon was given"
    )


def _resolve_discount(
    scenario: SttPushFoldScenario, config: SttPushFoldAnalysisConfig
) -> float:
    if config.discount is not None:
        return config.discount
    if scenario.repeated is not None:
        return scenario.repeated.discount
    return 1.0


def _build_filter_config(
    config: SttPushFoldAnalysisConfig,
) -> Optional[CandidateFilterConfig]:
    if (
        config.filter_allowed_info_sets is None
        and config.filter_max_l1_distance is None
        and config.filter_min_required_observations is None
    ):
        return None
    allowed: Optional[set]
    if config.filter_allowed_info_sets is None:
        allowed = None
    elif isinstance(config.filter_allowed_info_sets, (str, bytes)):
        raise ValueError(
            "filter_allowed_info_sets must be a collection of strings, not a "
            f"bare string; got {config.filter_allowed_info_sets!r}"
        )
    else:
        allowed = set(config.filter_allowed_info_sets)
    return CandidateFilterConfig(
        allowed_info_sets=allowed,
        max_l1_distance=config.filter_max_l1_distance,
        min_required_observations=config.filter_min_required_observations,
    )


def run_stt_pushfold_analysis(
    scenario: ScenarioInput,
    config: Optional[SttPushFoldAnalysisConfig] = None,
) -> SttPushFoldAnalysisResult:
    """Run a JSON STT push/fold scenario through the full analysis pipeline."""

    config = config or SttPushFoldAnalysisConfig()
    resolved_scenario = _resolve_scenario(scenario)
    scenario_path = Path(scenario) if isinstance(scenario, (str, Path)) else None
    build = build_stt_pushfold_game(resolved_scenario, tolerance=config.tolerance)

    if not build.shift_amounts:
        raise ValueError(
            "scenario has no candidates.shift_amounts; the analysis runner needs "
            "at least one shift amount to generate candidates"
        )

    horizon = _resolve_horizon(resolved_scenario, config)
    discount = _resolve_discount(resolved_scenario, config)
    filtering = _build_filter_config(config)
    resolved_detection_observation_model = resolve_detection_observation_model(
        config.detection_method, config.detection_observation_model
    )

    manifest = build_run_manifest(
        scenario_path=scenario_path,
        scenario_format_version=resolved_scenario.format_version,
        parameters={
            "horizon": horizon,
            "discount": discount,
            "response_mode": config.response_mode,
            "profit_tolerance": config.profit_tolerance,
            "max_selection_l1_distance": config.max_selection_l1_distance,
            "detection_log_likelihood_threshold": (
                config.detection_log_likelihood_threshold
            ),
            "detection_occurrence_probability_per_opportunity": (
                config.detection_occurrence_probability_per_opportunity
            ),
            "detection_comparable_spot_occurrence_probability_per_physical_hand": (
                config.detection_comparable_spot_occurrence_probability_per_physical_hand
            ),
            "detection_method": config.detection_method,
            "detection_observation_model": resolved_detection_observation_model,
            "max_detection_terminals": config.max_detection_terminals,
            "tolerance": config.tolerance,
            "max_pure_strategies": config.max_pure_strategies,
            "ranking_criterion": config.ranking_criterion,
            "value_unit": "modelled_tournament_prize_ev_delta",
        },
    )

    pipeline_result = run_candidate_analysis_pipeline(
        build.tree,
        build.baseline_hero_strategy,
        build.baseline_villain_strategy,
        generation=CandidateGenerationConfig(
            shift_amounts=build.shift_amounts,
            max_simultaneous_info_sets=build.max_simultaneous_info_sets,
        ),
        horizon=horizon,
        discount=discount,
        response_mode=config.response_mode,
        profit_tolerance=config.profit_tolerance,
        max_selection_l1_distance=config.max_selection_l1_distance,
        detection_log_likelihood_threshold=config.detection_log_likelihood_threshold,
        detection_occurrence_probability_per_opportunity=(
            config.detection_occurrence_probability_per_opportunity
        ),
        detection_comparable_spot_occurrence_probability_per_physical_hand=(
            config.detection_comparable_spot_occurrence_probability_per_physical_hand
        ),
        detection_method=config.detection_method,
        detection_observation_model=config.detection_observation_model,
        terminal_reveals=build.terminal_reveals,
        filtering=filtering,
        render_markdown=config.markdown,
        markdown_max_rows=config.markdown_max_rows,
        tolerance=config.tolerance,
        max_detection_terminals=config.max_detection_terminals,
        max_pure_strategies=config.max_pure_strategies,
        allow_negative_residual=True,
    )

    ranking_result: Optional[CandidateRankingResult] = None
    if config.ranking_criterion is not None:
        ranking_result = rank_candidate_rows(
            pipeline_result.analysis_report,
            config.ranking_criterion,
            descending=config.ranking_descending,
            eligible_only=config.ranking_eligible_only,
            top_k=config.ranking_top_k,
        )

    return SttPushFoldAnalysisResult(
        scenario=resolved_scenario,
        build=build,
        horizon=horizon,
        discount=discount,
        pipeline_result=pipeline_result,
        ranking_result=ranking_result,
        markdown_summary=pipeline_result.markdown_summary,
        manifest=manifest,
    )
