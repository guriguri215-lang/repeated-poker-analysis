"""Tests for the high-level candidate-analysis pipeline."""

import pytest

from candidate_library import baseline_villain_strategy
from nuts_chop_river import build_nuts_chop_river, default_hero_strategy

from repeated_poker import (
    CandidateFilterConfig,
    CandidateGenerationConfig,
    DETECTION_METHOD_REACH_WEIGHTED_V1,
    GameTree,
    HeroNode,
    HeroStrategy,
    TerminalNode,
    VillainNode,
    VillainStrategy,
    run_candidate_analysis_pipeline,
)


def _run(**overrides):
    tree = build_nuts_chop_river()
    baseline_hero = default_hero_strategy()
    baseline_villain = baseline_villain_strategy()
    kwargs = dict(
        generation=CandidateGenerationConfig(shift_amounts=[0.1, 0.2]),
        horizon=5,
    )
    kwargs.update(overrides)
    return run_candidate_analysis_pipeline(
        tree, baseline_hero, baseline_villain, **kwargs
    )


def _zero_terminal(node_id: str) -> TerminalNode:
    return TerminalNode(node_id=node_id, hero_ev=0.0, villain_ev=0.0, house_rake=0.0)


def _zero_reach_detection_inputs():
    hero_node = HeroNode(
        node_id="hero",
        info_set="H",
        actions=(
            ("call", _zero_terminal("T_call")),
            ("fold", _zero_terminal("T_fold")),
        ),
    )
    tree = GameTree(
        root=VillainNode(
            node_id="villain",
            info_set="V",
            actions=(("check", _zero_terminal("T_check")), ("bet", hero_node)),
        )
    )
    baseline_hero = HeroStrategy({"H": {"call": 0.5, "fold": 0.5}})
    baseline_villain = VillainStrategy({"V": {"check": 1.0, "bet": 0.0}})
    return tree, baseline_hero, baseline_villain


def test_pipeline_runs_end_to_end():
    result = _run()
    assert result.generated_candidates
    assert result.filter_result is not None
    assert result.comparison_report is not None
    assert result.analysis_report is not None
    assert isinstance(result.markdown_summary, str)


def test_generated_count_matches_filter_summary():
    result = _run()
    counts = result.filter_result.summary_counts
    assert counts.total == len(result.generated_candidates)
    assert counts.kept + counts.excluded == counts.total
    assert counts.kept == len(result.filter_result.kept)


def test_reports_cover_kept_candidates_only_in_order():
    result = _run(
        filtering=CandidateFilterConfig(max_l1_distance=0.3),
    )
    kept_ids = [c.candidate_id for c in result.filter_result.kept]
    assert [c.candidate.candidate_id for c in result.comparison_report.comparisons] == (
        kept_ids
    )
    assert [row.candidate_id for row in result.analysis_report.rows] == kept_ids


def test_render_markdown_true_returns_string():
    result = _run(render_markdown=True)
    assert isinstance(result.markdown_summary, str)
    assert "## Candidate Analysis Summary" in result.markdown_summary


def test_render_markdown_false_returns_none():
    result = _run(render_markdown=False)
    assert result.markdown_summary is None


def test_filtering_none_keeps_all():
    result = _run(filtering=None)
    counts = result.filter_result.summary_counts
    assert counts.excluded == 0
    assert counts.kept == counts.total
    assert len(result.comparison_report.comparisons) == len(result.generated_candidates)


def test_filtering_excludes_some_candidates():
    result = _run(filtering=CandidateFilterConfig(max_l1_distance=0.3))
    counts = result.filter_result.summary_counts
    assert counts.excluded > 0
    assert counts.kept < counts.total


def test_pipeline_reach_weighted_filter_uses_v1_none_semantics():
    tree, baseline_hero, baseline_villain = _zero_reach_detection_inputs()

    result = run_candidate_analysis_pipeline(
        tree,
        baseline_hero,
        baseline_villain,
        generation=CandidateGenerationConfig(shift_amounts=[0.2]),
        horizon=5,
        detection_log_likelihood_threshold=3.0,
        detection_method=DETECTION_METHOD_REACH_WEIGHTED_V1,
        filtering=CandidateFilterConfig(min_required_observations=10_000),
        render_markdown=False,
    )

    counts = result.filter_result.summary_counts
    assert counts.total == len(result.generated_candidates)
    assert counts.total > 0
    assert counts.excluded == 0
    assert counts.kept == counts.total
    assert result.analysis_report.detection_configuration.method == (
        DETECTION_METHOD_REACH_WEIGHTED_V1
    )
    assert all(row.t_detect_hands is None for row in result.analysis_report.rows)


def test_detection_enabled_flag_is_set():
    result = _run(detection_log_likelihood_threshold=3.0)
    assert result.analysis_report.detection_configuration.enabled is True


def test_detection_disabled_by_default():
    result = _run()
    assert result.analysis_report.detection_configuration.enabled is False


def test_filter_min_required_without_threshold_is_rejected():
    with pytest.raises(ValueError, match="detection_log_likelihood_threshold is required"):
        _run(filtering=CandidateFilterConfig(min_required_observations=5))


def test_empty_shift_amounts_is_rejected():
    with pytest.raises(ValueError, match="shift_amounts must not be empty"):
        _run(generation=CandidateGenerationConfig(shift_amounts=[]))


def test_invalid_render_markdown_is_rejected():
    with pytest.raises(ValueError, match="render_markdown"):
        _run(render_markdown="yes")


@pytest.mark.parametrize("bad", [-1, True])
def test_invalid_markdown_max_rows_is_rejected(bad):
    with pytest.raises(ValueError, match="max_rows"):
        _run(markdown_max_rows=bad)


def test_zero_kept_does_not_break_pipeline():
    # An empty allowed-info-set excludes every candidate.
    result = _run(filtering=CandidateFilterConfig(allowed_info_sets=set()))
    assert result.filter_result.summary_counts.kept == 0
    assert result.comparison_report.comparisons == []
    assert result.analysis_report.rows == []
    assert result.analysis_report.summary_counts.total == 0
    assert isinstance(result.markdown_summary, str)
    assert "## Candidate Analysis Summary" in result.markdown_summary
