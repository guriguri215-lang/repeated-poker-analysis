"""Tests for the consolidated candidate analysis report."""

import json

import pytest

from candidate_library import baseline_villain_strategy
from nuts_chop_river import build_nuts_chop_river, default_hero_strategy

from repeated_poker import (
    BestResponseResult,
    CandidateComparison,
    CandidateComparisonReport,
    FixedProfileValue,
    HeroStrategy,
    HeroStrategyCandidate,
    build_candidate_analysis_report,
    calculate_candidate_adaptation_deadline,
    compare_candidates,
    generate_shift_candidates,
    select_candidates,
)

HORIZON = 5
PROFIT_TOLERANCE = -2.0
MAX_L1_DISTANCE = 0.3


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------


def _best_response(ev_h_worst: float, ev_h_best: float) -> BestResponseResult:
    return BestResponseResult(
        villain_max_ev=0.0,
        best_response_strategies=[],
        ev_h_worst=ev_h_worst,
        ev_h_best=ev_h_best,
        expected_house_rake_worst=0.0,
        expected_house_rake_best=0.0,
        best_response_action_variation={},
        off_path_info_sets=[],
        num_villain_pure_strategies=0,
    )


def _comparison(
    candidate_id: str,
    fixed_hero_ev: float = 2.0,
    fixed_villain_ev: float = -1.0,
    worst_diff: float = 0.5,
    best_diff: float = 0.5,
    l1_distance: float = 0.2,
    ev_h_worst: float = -1.0,
    ev_h_best: float = 0.5,
) -> CandidateComparison:
    candidate = HeroStrategyCandidate(
        candidate_id=candidate_id,
        info_set="H1",
        source_action="check",
        target_action="bet",
        shift_amount=0.1,
        hero_strategy=HeroStrategy({}),
        l1_distance=l1_distance,
    )
    return CandidateComparison(
        candidate=candidate,
        fixed_profile_value=FixedProfileValue(
            hero_ev=fixed_hero_ev, villain_ev=fixed_villain_ev, house_rake=0.0
        ),
        villain_ev_diff_from_baseline=0.0,
        hero_ev_diff_from_baseline=0.0,
        best_response=_best_response(ev_h_worst, ev_h_best),
        post_response_hero_ev_worst_diff=worst_diff,
        post_response_hero_ev_best_diff=best_diff,
        robustly_profitable=worst_diff > 0.0,
    )


def _report(comparisons, baseline_hero_ev: float = 1.0) -> CandidateComparisonReport:
    return CandidateComparisonReport(
        baseline_value=FixedProfileValue(
            hero_ev=baseline_hero_ev, villain_ev=0.0, house_rake=0.0
        ),
        comparisons=list(comparisons),
    )


# ---------------------------------------------------------------------------
# nuts-chop fixtures
# ---------------------------------------------------------------------------


def _nuts_chop_comparison_report() -> CandidateComparisonReport:
    tree = build_nuts_chop_river()
    baseline_hero = default_hero_strategy()
    baseline_villain = baseline_villain_strategy()
    candidates = generate_shift_candidates(tree, baseline_hero, [0.1, 0.2])
    return compare_candidates(tree, baseline_hero, baseline_villain, candidates)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_every_candidate_has_one_row_in_order():
    comparison_report = _nuts_chop_comparison_report()
    report = build_candidate_analysis_report(comparison_report, horizon=HORIZON)

    assert len(report.rows) == len(comparison_report.comparisons)
    assert [row.candidate_id for row in report.rows] == [
        c.candidate.candidate_id for c in comparison_report.comparisons
    ]


def test_selection_labels_match_selection_api():
    comparison_report = _nuts_chop_comparison_report()
    report = build_candidate_analysis_report(
        comparison_report,
        horizon=HORIZON,
        profit_tolerance=PROFIT_TOLERANCE,
        max_l1_distance=MAX_L1_DISTANCE,
    )
    selection = select_candidates(
        comparison_report,
        profit_tolerance=PROFIT_TOLERANCE,
        max_l1_distance=MAX_L1_DISTANCE,
    )
    eligible_ids = {c.candidate.candidate_id for c in selection.eligible}
    excluded_reasons = {
        e.comparison.candidate.candidate_id: e.reasons for e in selection.excluded
    }
    minimum_ids = {
        c.candidate.candidate_id for c in selection.minimum_villain_ev_candidates
    }
    pareto_ids = {c.candidate.candidate_id for c in selection.pareto_frontier}

    for row in report.rows:
        assert row.is_eligible == (row.candidate_id in eligible_ids)
        assert row.exclusion_reasons == excluded_reasons.get(row.candidate_id, [])
        assert row.is_minimum_villain_ev_candidate == (row.candidate_id in minimum_ids)
        assert row.is_pareto_frontier_candidate == (row.candidate_id in pareto_ids)


def test_summary_counts_are_correct():
    comparison_report = _nuts_chop_comparison_report()
    report = build_candidate_analysis_report(
        comparison_report,
        horizon=HORIZON,
        profit_tolerance=PROFIT_TOLERANCE,
        max_l1_distance=MAX_L1_DISTANCE,
    )
    selection = select_candidates(
        comparison_report,
        profit_tolerance=PROFIT_TOLERANCE,
        max_l1_distance=MAX_L1_DISTANCE,
    )
    counts = report.summary_counts
    assert counts.total == len(comparison_report.comparisons)
    assert counts.eligible == len(selection.eligible)
    assert counts.excluded == len(selection.excluded)
    assert counts.minimum_villain_ev == len(selection.minimum_villain_ev_candidates)
    assert counts.pareto_frontier == len(selection.pareto_frontier)
    assert counts.eligible + counts.excluded == counts.total


def test_response_mode_switches_deadline_input():
    # Pre-adaptation a=2 (b=1, N=3). worst l=-5 -> never reaches baseline (None);
    # best l=5 -> always at least baseline (t_deadline=3).
    comparison = _comparison("c", fixed_hero_ev=2.0, ev_h_worst=-5.0, ev_h_best=5.0)
    comparison_report = _report([comparison], baseline_hero_ev=1.0)

    worst = build_candidate_analysis_report(
        comparison_report, horizon=3, response_mode="worst", profit_tolerance=-100.0
    )
    best = build_candidate_analysis_report(
        comparison_report, horizon=3, response_mode="best", profit_tolerance=-100.0
    )

    assert worst.rows[0].response_mode == "worst"
    assert worst.rows[0].t_deadline is None
    assert best.rows[0].response_mode == "best"
    assert best.rows[0].t_deadline == 3


def test_t_deadline_none_is_preserved_and_json_serialisable():
    comparison = _comparison("c", fixed_hero_ev=2.0, ev_h_worst=-5.0, ev_h_best=-5.0)
    comparison_report = _report([comparison], baseline_hero_ev=1.0)
    report = build_candidate_analysis_report(
        comparison_report, horizon=3, profit_tolerance=-100.0
    )
    assert report.rows[0].t_deadline is None
    dumped = json.loads(json.dumps(report.summary_rows()))
    assert dumped[0]["t_deadline"] is None


def test_never_adapts_delta_matches_m_n_plus_one_row():
    comparison = _comparison("c", fixed_hero_ev=2.0, ev_h_worst=-1.0, ev_h_best=0.5)
    comparison_report = _report([comparison], baseline_hero_ev=1.0)
    report = build_candidate_analysis_report(
        comparison_report, horizon=4, profit_tolerance=-100.0
    )
    reference = calculate_candidate_adaptation_deadline(
        comparison_report, comparison, horizon=4
    )
    last_row = reference.result.timing[-1]  # m = N+1
    assert last_row.adaptation_opportunity == 5
    assert report.rows[0].never_adapts_delta_from_baseline == pytest.approx(
        last_row.delta_from_baseline
    )
    assert report.rows[0].never_adapts_total_hero_ev == pytest.approx(
        last_row.locked_total_hero_ev
    )


def test_to_dict_and_summary_rows_are_json_serialisable():
    comparison_report = _nuts_chop_comparison_report()
    report = build_candidate_analysis_report(
        comparison_report,
        horizon=HORIZON,
        profit_tolerance=PROFIT_TOLERANCE,
        max_l1_distance=MAX_L1_DISTANCE,
    )
    # Neither call should raise; round-trip to confirm serialisability.
    as_dict = json.loads(json.dumps(report.to_dict()))
    rows = json.loads(json.dumps(report.summary_rows()))
    assert as_dict["summary_counts"]["total"] == len(report.rows)
    assert len(rows) == len(report.rows)
    assert as_dict["candidate_rows"] == rows


def test_duplicate_candidate_id_is_rejected():
    comparison_report = _report(
        [_comparison("dup"), _comparison("dup")], baseline_hero_ev=1.0
    )
    with pytest.raises(ValueError, match="duplicate candidate_id"):
        build_candidate_analysis_report(comparison_report, horizon=3)


def _empty_report() -> CandidateComparisonReport:
    return CandidateComparisonReport(
        baseline_value=FixedProfileValue(hero_ev=1.0, villain_ev=0.0, house_rake=0.0),
        comparisons=[],
    )


def test_empty_report_with_valid_input_returns_empty_analysis():
    report = build_candidate_analysis_report(_empty_report(), horizon=3)

    assert report.rows == []
    counts = report.summary_counts
    assert (
        counts.total
        == counts.eligible
        == counts.excluded
        == counts.minimum_villain_ev
        == counts.pareto_frontier
        == 0
    )
    # Still JSON-serialisable.
    as_dict = json.loads(json.dumps(report.to_dict()))
    assert as_dict["candidate_rows"] == []
    assert json.loads(json.dumps(report.summary_rows())) == []


@pytest.mark.parametrize("bad_horizon", [0, -1, 2.5, True])
def test_empty_report_rejects_invalid_horizon(bad_horizon):
    with pytest.raises(ValueError, match="horizon"):
        build_candidate_analysis_report(_empty_report(), horizon=bad_horizon)


@pytest.mark.parametrize("bad_discount", [0.0, -0.1, 1.5, float("nan"), float("inf")])
def test_empty_report_rejects_invalid_discount(bad_discount):
    with pytest.raises(ValueError, match="discount"):
        build_candidate_analysis_report(
            _empty_report(), horizon=3, discount=bad_discount
        )


def test_empty_report_rejects_unknown_response_mode():
    with pytest.raises(ValueError, match="unknown response_mode"):
        build_candidate_analysis_report(
            _empty_report(), horizon=3, response_mode="mixed"
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_empty_report_rejects_invalid_tolerance(bad):
    with pytest.raises(ValueError, match="tolerance"):
        build_candidate_analysis_report(_empty_report(), horizon=3, tolerance=bad)


@pytest.mark.parametrize("bad_max_horizon", [0, -1, 2.5, True])
def test_empty_report_rejects_invalid_max_horizon(bad_max_horizon):
    with pytest.raises(ValueError, match="max_horizon"):
        build_candidate_analysis_report(
            _empty_report(), horizon=1, max_horizon=bad_max_horizon
        )


def test_empty_report_rejects_horizon_above_max_horizon():
    with pytest.raises(ValueError, match="safety limit"):
        build_candidate_analysis_report(_empty_report(), horizon=10, max_horizon=5)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_empty_report_rejects_invalid_profit_tolerance(bad):
    with pytest.raises(ValueError, match="profit_tolerance"):
        build_candidate_analysis_report(
            _empty_report(), horizon=3, profit_tolerance=bad
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_empty_report_rejects_invalid_max_l1_distance(bad):
    with pytest.raises(ValueError, match="max_l1_distance"):
        build_candidate_analysis_report(
            _empty_report(), horizon=3, max_l1_distance=bad
        )


def test_integration_full_pipeline_is_consistent():
    comparison_report = _nuts_chop_comparison_report()
    report = build_candidate_analysis_report(
        comparison_report,
        horizon=HORIZON,
        discount=1.0,
        response_mode="worst",
        profit_tolerance=PROFIT_TOLERANCE,
        max_l1_distance=MAX_L1_DISTANCE,
    )

    # Every row carries the shared configuration.
    assert report.deadline_configuration.horizon == HORIZON
    assert report.deadline_configuration.response_mode == "worst"
    assert report.selection_configuration.max_l1_distance == MAX_L1_DISTANCE
    for row, comparison in zip(report.rows, comparison_report.comparisons):
        assert row.response_mode == "worst"
        # Row values mirror the underlying comparison.
        assert row.post_response_hero_ev_worst == pytest.approx(
            comparison.best_response.ev_h_worst
        )
        assert row.post_response_hero_ev_best == pytest.approx(
            comparison.best_response.ev_h_best
        )
        assert row.fixed_villain_ev == pytest.approx(
            comparison.fixed_profile_value.villain_ev
        )
    assert report.summary_counts.total == len(report.rows)
