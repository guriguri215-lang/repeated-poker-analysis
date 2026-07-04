"""Tests for the diagnostic candidate-row ranking helper."""

import copy

import pytest

from candidate_library import baseline_villain_strategy
from nuts_chop_river import build_nuts_chop_river, default_hero_strategy

from repeated_poker import (
    RANK_BY_DETECTED_ADAPTATION_DELTA,
    RANK_BY_FIXED_VILLAIN_EV,
    RANK_BY_L1_DISTANCE,
    RANK_BY_POST_RESPONSE_HERO_EV_WORST_DIFF,
    RANK_BY_T_DEADLINE,
    RANK_BY_T_DETECT,
    CandidateAnalysisReport,
    CandidateAnalysisRow,
    CandidateGenerationConfig,
    DeadlineConfiguration,
    DetectionConfiguration,
    FixedProfileValue,
    SelectionConfiguration,
    SelectionSummaryCounts,
    rank_candidate_rows,
    run_candidate_analysis_pipeline,
)


def _row(candidate_id, **overrides) -> CandidateAnalysisRow:
    defaults = dict(
        candidate_id=candidate_id,
        info_set="H1",
        source_action="check",
        target_action="bet",
        shift_amount=0.1,
        shifts=[
            {
                "info_set": "H1",
                "source_action": "check",
                "target_action": "bet",
                "shift_amount": 0.1,
            }
        ],
        l1_distance=0.2,
        observation_distance=0.15,
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
        is_ev_observation_deadline_pareto_candidate=False,
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


def _report(rows) -> CandidateAnalysisReport:
    return CandidateAnalysisReport(
        baseline_value=FixedProfileValue(0.0, 0.0, 0.0),
        selection_configuration=SelectionConfiguration(0.0, None, 1e-9),
        deadline_configuration=DeadlineConfiguration(5, 1.0, "worst", 1e-9, 100000),
        detection_configuration=DetectionConfiguration(False, None, None, 1e-9),
        rows=list(rows),
        summary_counts=SelectionSummaryCounts(len(rows), 0, 0, 0, 0),
    )


def _order(result):
    return [ranked.row.candidate_id for ranked in result.ranked_rows]


def test_default_descending_higher_is_first():
    report = _report(
        [
            _row("low", post_response_hero_ev_worst_diff=-2.0),
            _row("high", post_response_hero_ev_worst_diff=1.0),
            _row("mid", post_response_hero_ev_worst_diff=-0.5),
        ]
    )
    result = rank_candidate_rows(report, RANK_BY_POST_RESPONSE_HERO_EV_WORST_DIFF)
    assert result.descending is True
    assert _order(result) == ["high", "mid", "low"]
    assert [r.rank for r in result.ranked_rows] == [1, 2, 3]


def test_default_ascending_lower_is_first():
    report = _report(
        [
            _row("a", fixed_villain_ev=-1.0),
            _row("b", fixed_villain_ev=-3.0),
            _row("c", fixed_villain_ev=-2.0),
        ]
    )
    result = rank_candidate_rows(report, RANK_BY_FIXED_VILLAIN_EV)
    assert result.descending is False
    assert _order(result) == ["b", "c", "a"]


def test_l1_distance_default_is_ascending():
    report = _report(
        [_row("far", l1_distance=0.4), _row("near", l1_distance=0.2)]
    )
    result = rank_candidate_rows(report, RANK_BY_L1_DISTANCE)
    assert result.descending is False
    assert _order(result) == ["near", "far"]


def test_explicit_descending_overrides_default():
    report = _report(
        [_row("near", l1_distance=0.2), _row("far", l1_distance=0.4)]
    )
    result = rank_candidate_rows(report, RANK_BY_L1_DISTANCE, descending=True)
    assert result.descending is True
    assert _order(result) == ["far", "near"]


@pytest.mark.parametrize("descending", [True, False])
def test_none_values_are_always_last(descending):
    report = _report(
        [
            _row("none1", t_deadline=None),
            _row("two", t_deadline=2),
            _row("none2", t_deadline=None),
            _row("five", t_deadline=5),
        ]
    )
    result = rank_candidate_rows(report, RANK_BY_T_DEADLINE, descending=descending)
    order = _order(result)
    # The two None rows are last, preserving their original order.
    assert order[-2:] == ["none1", "none2"]
    assert set(order[:2]) == {"two", "five"}


def test_stable_order_for_equal_keys():
    report = _report(
        [
            _row("first", fixed_villain_ev=-1.0),
            _row("second", fixed_villain_ev=-1.0),
            _row("third", fixed_villain_ev=-1.0),
        ]
    )
    result = rank_candidate_rows(report, RANK_BY_FIXED_VILLAIN_EV)
    assert _order(result) == ["first", "second", "third"]
    # Descending must also keep the original order on ties.
    result_desc = rank_candidate_rows(
        report, RANK_BY_FIXED_VILLAIN_EV, descending=True
    )
    assert _order(result_desc) == ["first", "second", "third"]


def test_eligible_only_filters_rows():
    report = _report(
        [
            _row("keep", is_eligible=True, fixed_villain_ev=-1.0),
            _row("drop", is_eligible=False, fixed_villain_ev=-5.0),
        ]
    )
    result = rank_candidate_rows(
        report, RANK_BY_FIXED_VILLAIN_EV, eligible_only=True
    )
    assert _order(result) == ["keep"]


def test_top_k_limits_rows():
    report = _report(
        [
            _row("a", l1_distance=0.1),
            _row("b", l1_distance=0.2),
            _row("c", l1_distance=0.3),
        ]
    )
    result = rank_candidate_rows(report, RANK_BY_L1_DISTANCE, top_k=2)
    assert _order(result) == ["a", "b"]


def test_top_k_zero_is_empty():
    report = _report([_row("a"), _row("b")])
    result = rank_candidate_rows(report, RANK_BY_L1_DISTANCE, top_k=0)
    assert result.ranked_rows == []


def test_unknown_criterion_is_rejected():
    report = _report([_row("a")])
    with pytest.raises(ValueError, match="unknown criterion"):
        rank_candidate_rows(report, "not_a_field")


@pytest.mark.parametrize("bad", [1, "yes", 0.0])
def test_invalid_descending_is_rejected(bad):
    report = _report([_row("a")])
    with pytest.raises(ValueError, match="descending"):
        rank_candidate_rows(report, RANK_BY_L1_DISTANCE, descending=bad)


@pytest.mark.parametrize("bad", [1, "yes", None])
def test_invalid_eligible_only_is_rejected(bad):
    report = _report([_row("a")])
    with pytest.raises(ValueError, match="eligible_only"):
        rank_candidate_rows(report, RANK_BY_L1_DISTANCE, eligible_only=bad)


@pytest.mark.parametrize("bad", [-1, 2.5, True])
def test_invalid_top_k_is_rejected(bad):
    report = _report([_row("a")])
    with pytest.raises(ValueError, match="top_k"):
        rank_candidate_rows(report, RANK_BY_L1_DISTANCE, top_k=bad)


def test_report_rows_are_not_mutated():
    rows = [_row("a", l1_distance=0.3), _row("b", l1_distance=0.1)]
    report = _report(rows)
    before = copy.deepcopy([r.candidate_id for r in report.rows])
    rank_candidate_rows(report, RANK_BY_L1_DISTANCE)
    assert [r.candidate_id for r in report.rows] == before


def test_detected_adaptation_delta_default_descending():
    report = _report(
        [
            _row("worse", detected_adaptation_delta_from_baseline=-1.0),
            _row("better", detected_adaptation_delta_from_baseline=2.0),
        ]
    )
    result = rank_candidate_rows(report, RANK_BY_DETECTED_ADAPTATION_DELTA)
    assert result.descending is True
    assert _order(result) == ["better", "worse"]


def test_t_detect_default_descending():
    report = _report(
        [
            _row("easy", t_detect_estimated_opportunities=2),
            _row("hard", t_detect_estimated_opportunities=50),
        ]
    )
    result = rank_candidate_rows(report, RANK_BY_T_DETECT)
    assert result.descending is True
    assert _order(result) == ["hard", "easy"]


def test_integration_from_nuts_chop_pipeline():
    tree = build_nuts_chop_river()
    baseline_hero = default_hero_strategy()
    baseline_villain = baseline_villain_strategy()
    result = run_candidate_analysis_pipeline(
        tree,
        baseline_hero,
        baseline_villain,
        generation=CandidateGenerationConfig(shift_amounts=[0.1, 0.2]),
        horizon=5,
        detection_log_likelihood_threshold=3.0,
        detection_occurrence_probability_per_opportunity=0.5,
        render_markdown=False,
    )
    report = result.analysis_report

    ranking = rank_candidate_rows(
        report, RANK_BY_POST_RESPONSE_HERO_EV_WORST_DIFF
    )
    assert len(ranking.ranked_rows) == len(report.rows)
    assert [r.rank for r in ranking.ranked_rows] == list(
        range(1, len(report.rows) + 1)
    )
    # The criterion is descending by default: each value is >= the next.
    values = [r.sort_key for r in ranking.ranked_rows]
    assert values == sorted(values, reverse=True)
