"""Tests for the Markdown candidate-analysis summary renderer."""

import math

import pytest

from candidate_library import baseline_villain_strategy
from nuts_chop_river import build_nuts_chop_river, default_hero_strategy

from repeated_poker import (
    CandidateAnalysisReport,
    CandidateAnalysisRow,
    DeadlineConfiguration,
    DetectionConfiguration,
    FixedProfileValue,
    SelectionConfiguration,
    SelectionSummaryCounts,
    build_candidate_analysis_report,
    compare_candidates,
    format_candidate_analysis_markdown,
    generate_shift_candidates,
)


def _row(**overrides) -> CandidateAnalysisRow:
    defaults = dict(
        candidate_id="c1",
        info_set="H1",
        source_action="check",
        target_action="bet",
        shift_amount=0.1,
        l1_distance=0.2,
        fixed_hero_ev=0.5,
        fixed_villain_ev=-1.0,
        fixed_house_rake=0.1,
        hero_ev_diff_from_baseline=0.1,
        villain_ev_diff_from_baseline=-0.1,
        post_response_hero_ev_worst=-0.8,
        post_response_hero_ev_best=-0.5,
        post_response_hero_ev_worst_diff=-1.2,
        post_response_hero_ev_best_diff=-0.9,
        robustly_profitable=False,
        is_eligible=True,
        exclusion_reasons=[],
        is_minimum_villain_ev_candidate=False,
        is_pareto_frontier_candidate=False,
        response_mode="worst",
        t_deadline=3,
        baseline_total_hero_ev=2.25,
        never_adapts_total_hero_ev=3.1,
        never_adapts_delta_from_baseline=0.85,
        detection_total_variation_distance=0.3,
        detection_kl_divergence_nats=0.22,
        detection_required_observations=14,
        detection_estimated_opportunities=28,
        t_detect_estimated_opportunities=28,
        t_detect_is_no_later_than_t_deadline=False,
        detected_adaptation_opportunity=4,
        detected_adaptation_delta_from_baseline=0.85,
        detected_adaptation_is_at_least_baseline=True,
    )
    defaults.update(overrides)
    return CandidateAnalysisRow(**defaults)


def _report(rows, counts=None) -> CandidateAnalysisReport:
    if counts is None:
        counts = SelectionSummaryCounts(
            total=len(rows),
            eligible=len(rows),
            excluded=0,
            minimum_villain_ev=0,
            pareto_frontier=0,
        )
    return CandidateAnalysisReport(
        baseline_value=FixedProfileValue(hero_ev=0.45, villain_ev=-1.95, house_rake=1.5),
        selection_configuration=SelectionConfiguration(
            profit_tolerance=-2.0, max_l1_distance=0.3, tolerance=1e-9
        ),
        deadline_configuration=DeadlineConfiguration(
            horizon=5, discount=1.0, response_mode="worst", tolerance=1e-9,
            max_horizon=100000,
        ),
        detection_configuration=DetectionConfiguration(
            enabled=True,
            log_likelihood_threshold=3.0,
            occurrence_probability_per_opportunity=0.5,
            tolerance=1e-9,
        ),
        rows=list(rows),
        summary_counts=counts,
    )


def _rows(n):
    return [_row(candidate_id=f"c{i}") for i in range(1, n + 1)]


def test_contains_main_sections():
    md = format_candidate_analysis_markdown(_report(_rows(1)))
    assert "## Candidate Analysis Summary" in md
    assert "### Configurations" in md
    assert "### Summary Counts" in md
    assert "### Candidate Rows" in md
    assert "### Notes" in md


def test_contains_configuration_values():
    md = format_candidate_analysis_markdown(_report(_rows(1)))
    assert "profit_tolerance: -2.000000" in md
    assert "max_l1_distance: 0.300000" in md
    assert "horizon: 5" in md
    assert "discount: 1.000000" in md
    assert "response_mode: worst" in md
    assert "max_horizon: 100000" in md
    assert "enabled: yes" in md
    assert "log_likelihood_threshold: 3.000000" in md
    assert "occurrence_probability_per_opportunity: 0.500000" in md


def test_contains_summary_counts():
    counts = SelectionSummaryCounts(
        total=3, eligible=2, excluded=1, minimum_villain_ev=1, pareto_frontier=1
    )
    md = format_candidate_analysis_markdown(_report(_rows(3), counts=counts))
    assert "total: 3" in md
    assert "eligible: 2" in md
    assert "excluded: 1" in md
    assert "minimum_villain_ev: 1" in md
    assert "pareto_frontier: 1" in md


def test_contains_candidate_columns_and_shift_format():
    md = format_candidate_analysis_markdown(_report(_rows(1)))
    for header in [
        "candidate_id",
        "info_set",
        "shift",
        "l1_distance",
        "fixed_hero_ev",
        "post_response_hero_ev_worst",
        "post_response_hero_ev_worst_diff",
        "robustly_profitable",
        "is_eligible",
        "t_deadline",
        "t_detect_estimated_opportunities",
        "detected_adaptation_delta_from_baseline",
        "detected_adaptation_is_at_least_baseline",
        "exclusion_reasons",
    ]:
        assert header in md
    assert "check -> bet (+0.1)" in md
    assert "c1" in md


def test_none_is_rendered_as_dash():
    md = format_candidate_analysis_markdown(_report([_row(t_deadline=None)]))
    assert "| - |" in md


def test_bool_is_rendered_as_yes_no():
    md = format_candidate_analysis_markdown(
        _report([_row(robustly_profitable=True, is_eligible=False)])
    )
    assert "yes" in md
    assert "no" in md


def test_list_join_and_empty_list():
    md_with = format_candidate_analysis_markdown(
        _report([_row(exclusion_reasons=["not_robustly_profitable", "x"])])
    )
    assert "not_robustly_profitable, x" in md_with

    md_empty = format_candidate_analysis_markdown(
        _report([_row(exclusion_reasons=[])])
    )
    # The empty list renders as a dash cell.
    assert "| - |" in md_empty


def test_inf_is_rendered_as_inf():
    md = format_candidate_analysis_markdown(
        _report([_row(detected_adaptation_delta_from_baseline=math.inf)])
    )
    assert "| inf |" in md


def test_pipe_is_escaped_in_cells():
    md = format_candidate_analysis_markdown(_report([_row(candidate_id="a|b")]))
    assert "a\\|b" in md
    # The raw unescaped form must not appear as a bare cell value.
    assert "| a|b |" not in md


def test_max_rows_limits_and_reports_remaining():
    md = format_candidate_analysis_markdown(_report(_rows(3)), max_rows=1)
    assert "c1" in md
    assert "c2" not in md
    assert "c3" not in md
    assert "... 2 more rows not shown." in md


def test_max_rows_zero_shows_no_rows():
    md = format_candidate_analysis_markdown(_report(_rows(3)), max_rows=0)
    assert "c1" not in md
    assert "c2" not in md
    assert "c3" not in md
    assert "... 3 more rows not shown." in md
    # The table header is still present.
    assert "### Candidate Rows" in md


def test_no_remaining_message_when_all_rows_shown():
    md = format_candidate_analysis_markdown(_report(_rows(2)))
    assert "more rows not shown" not in md


@pytest.mark.parametrize("bad", [-1, True, 2.5])
def test_invalid_max_rows_is_rejected(bad):
    with pytest.raises(ValueError, match="max_rows"):
        format_candidate_analysis_markdown(_report(_rows(1)), max_rows=bad)


def test_integration_from_nuts_chop_report():
    tree = build_nuts_chop_river()
    baseline_hero = default_hero_strategy()
    baseline_villain = baseline_villain_strategy()
    candidates = generate_shift_candidates(tree, baseline_hero, [0.1, 0.2])
    comparison_report = compare_candidates(
        tree, baseline_hero, baseline_villain, candidates
    )
    report = build_candidate_analysis_report(
        comparison_report,
        horizon=5,
        baseline_hero_strategy=baseline_hero,
        detection_log_likelihood_threshold=3.0,
        detection_occurrence_probability_per_opportunity=0.5,
    )

    md = format_candidate_analysis_markdown(report)
    assert "## Candidate Analysis Summary" in md
    for row in report.rows:
        # Real candidate ids contain "|", which the renderer escapes.
        assert row.candidate_id.replace("|", "\\|") in md
