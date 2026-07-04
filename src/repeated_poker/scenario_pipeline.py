"""Run a JSON river scenario through the candidate-analysis pipeline.

This connects the v1 scenario-input layer (:mod:`repeated_poker.scenario_io`) to
the existing orchestration in :mod:`repeated_poker.pipeline`, so a user can drive
the whole analysis from a single JSON scenario file: load and validate, build the
game, generate and filter candidates, compare, report, render Markdown, and
optionally rank the rows.

It adds no new analysis maths. ``run_river_scenario_analysis`` only resolves the
repeated-game horizon/discount and filter settings from the scenario plus an
optional :class:`RiverScenarioAnalysisConfig`, then delegates to
``run_candidate_analysis_pipeline`` and the existing ``rank_candidate_rows``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

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
from .scenario_io import (
    RiverScenario,
    RiverScenarioBuildResult,
    build_river_steal_game_from_scenario,
    load_river_scenario_json,
)

ScenarioInput = Union[RiverScenario, str, Path]


@dataclass(frozen=True)
class RiverScenarioAnalysisConfig:
    """Optional overrides for :func:`run_river_scenario_analysis`.

    Horizon and discount default to the scenario's repeated-game configuration
    (horizon to the maximum of ``scenario.repeated.horizons``); an explicit value
    here takes precedence. Filter fields are only applied when at least one of
    them is set. Detection fields, ``profit_tolerance``, and
    ``max_selection_l1_distance`` are forwarded to the pipeline unchanged.
    """

    horizon: Optional[int] = None
    discount: Optional[float] = None
    response_mode: str = RESPONSE_MODE_WORST
    profit_tolerance: float = 0.0
    max_selection_l1_distance: Optional[float] = None
    detection_log_likelihood_threshold: Optional[float] = None
    detection_occurrence_probability_per_opportunity: Optional[float] = None
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
class RiverScenarioAnalysisResult:
    """Everything produced by :func:`run_river_scenario_analysis`.

    ``horizon`` and ``discount`` are the resolved values actually used.
    ``ranking_result`` is ``None`` unless a ranking criterion was requested.
    ``markdown_summary`` mirrors ``pipeline_result.markdown_summary`` for
    convenience and is ``None`` when Markdown rendering is disabled.
    ``manifest`` is the reproducibility manifest of the run (scenario file
    SHA-256 when run from a file, format version, package version, best-effort
    git commit, UTC timestamp, and the effective parameters); it is
    descriptive metadata only and changes no analysis result.
    """

    scenario: RiverScenario
    build: RiverScenarioBuildResult
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


def _resolve_scenario(scenario: ScenarioInput) -> RiverScenario:
    if isinstance(scenario, RiverScenario):
        return scenario
    if isinstance(scenario, (str, Path)):
        return load_river_scenario_json(scenario)
    raise TypeError(
        "scenario must be a RiverScenario, a path string, or a Path, "
        f"got {type(scenario).__name__}"
    )


def _resolve_horizon(scenario: RiverScenario, config: RiverScenarioAnalysisConfig) -> int:
    if config.horizon is not None:
        if isinstance(config.horizon, bool) or not isinstance(config.horizon, int):
            raise ValueError(f"horizon must be an int, got {config.horizon!r}")
        if config.horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {config.horizon}")
        return config.horizon
    if scenario.repeated is not None and scenario.repeated.horizons:
        return max(scenario.repeated.horizons)
    raise ValueError(
        "no horizon available: the scenario has no repeated.horizons and no "
        "explicit config.horizon was given"
    )


def _resolve_discount(scenario: RiverScenario, config: RiverScenarioAnalysisConfig) -> float:
    if config.discount is not None:
        return config.discount
    if scenario.repeated is not None:
        return scenario.repeated.discount
    return 1.0


def _build_filter_config(
    config: RiverScenarioAnalysisConfig,
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
        # A bare string is iterable but would be split into characters; reject it
        # here rather than silently building a character set that bypasses the
        # filter's own bare-string guard.
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


def run_river_scenario_analysis(
    scenario: ScenarioInput,
    config: Optional[RiverScenarioAnalysisConfig] = None,
) -> RiverScenarioAnalysisResult:
    """Run a JSON river scenario through the full candidate-analysis pipeline.

    ``scenario`` may be a parsed :class:`RiverScenario` or a path to a JSON
    scenario file. The scenario's ``shift_amounts`` drive candidate generation
    (a missing or empty list is rejected), and its repeated-game configuration
    supplies the default horizon (its maximum) and discount, both overridable via
    ``config``. The remaining work is delegated to
    ``run_candidate_analysis_pipeline``; when ``config.ranking_criterion`` is
    given, the report rows are additionally ranked via ``rank_candidate_rows``.

    The returned result carries a :class:`RunManifest` with the scenario file's
    SHA-256 (``None`` when ``scenario`` is an in-memory :class:`RiverScenario`),
    the scenario format version, the package version, a best-effort git commit
    of the package source (``None`` when unavailable), a UTC timestamp, and the
    resolved analysis parameters.  The manifest is reproducibility metadata
    only; it changes no analysis result.
    """

    config = config or RiverScenarioAnalysisConfig()
    resolved_scenario = _resolve_scenario(scenario)
    scenario_path = Path(scenario) if isinstance(scenario, (str, Path)) else None
    build = build_river_steal_game_from_scenario(resolved_scenario)

    if not build.shift_amounts:
        raise ValueError(
            "scenario has no candidate_generation.shift_amounts; the analysis "
            "runner needs at least one shift amount to generate candidates"
        )

    horizon = _resolve_horizon(resolved_scenario, config)
    discount = _resolve_discount(resolved_scenario, config)
    filtering = _build_filter_config(config)

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
            "tolerance": config.tolerance,
            "max_pure_strategies": config.max_pure_strategies,
            "ranking_criterion": config.ranking_criterion,
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
        filtering=filtering,
        render_markdown=config.markdown,
        markdown_max_rows=config.markdown_max_rows,
        tolerance=config.tolerance,
        max_pure_strategies=config.max_pure_strategies,
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

    return RiverScenarioAnalysisResult(
        scenario=resolved_scenario,
        build=build,
        horizon=horizon,
        discount=discount,
        pipeline_result=pipeline_result,
        ranking_result=ranking_result,
        markdown_summary=pipeline_result.markdown_summary,
        manifest=manifest,
    )
