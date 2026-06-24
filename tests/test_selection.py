"""Tests for the candidate-selection / screening API."""

import math

import pytest

from candidate_library import baseline_villain_strategy
from nuts_chop_river import build_nuts_chop_river, default_hero_strategy

from repeated_poker import (
    L1_DISTANCE_EXCEEDS_LIMIT,
    NOT_ROBUSTLY_PROFITABLE,
    BestResponseResult,
    CandidateComparison,
    CandidateComparisonReport,
    FixedProfileValue,
    HeroStrategy,
    HeroStrategyCandidate,
    compare_candidates,
    generate_shift_candidates,
    pareto_frontier,
    select_candidates,
    select_minimum_villain_ev,
)


def _best_response() -> BestResponseResult:
    """A placeholder best-response result; selection ignores its contents."""

    return BestResponseResult(
        villain_max_ev=0.0,
        best_response_strategies=[],
        ev_h_worst=0.0,
        ev_h_best=0.0,
        expected_house_rake_worst=0.0,
        expected_house_rake_best=0.0,
        best_response_action_variation={},
        off_path_info_sets=[],
        num_villain_pure_strategies=0,
    )


def _comparison(
    candidate_id: str, villain_ev: float, worst_diff: float, l1_distance: float
) -> CandidateComparison:
    candidate = HeroStrategyCandidate(
        candidate_id=candidate_id,
        info_set="H",
        source_action="a",
        target_action="b",
        shift_amount=0.1,
        hero_strategy=HeroStrategy({}),
        l1_distance=l1_distance,
    )
    return CandidateComparison(
        candidate=candidate,
        fixed_profile_value=FixedProfileValue(
            hero_ev=0.0, villain_ev=villain_ev, house_rake=0.0
        ),
        villain_ev_diff_from_baseline=0.0,
        hero_ev_diff_from_baseline=0.0,
        best_response=_best_response(),
        post_response_hero_ev_worst_diff=worst_diff,
        post_response_hero_ev_best_diff=worst_diff,
        robustly_profitable=worst_diff > 0.0,
    )


def _report(comparisons) -> CandidateComparisonReport:
    return CandidateComparisonReport(
        baseline_value=FixedProfileValue(0.0, 0.0, 0.0), comparisons=list(comparisons)
    )


def test_unprofitable_candidate_is_excluded():
    report = _report(
        [
            _comparison("good", villain_ev=-1.0, worst_diff=0.5, l1_distance=0.2),
            _comparison("bad", villain_ev=-2.0, worst_diff=-0.5, l1_distance=0.2),
        ]
    )
    result = select_candidates(report, profit_tolerance=0.0)

    eligible_ids = {c.candidate.candidate_id for c in result.eligible}
    assert eligible_ids == {"good"}
    excluded = {e.comparison.candidate.candidate_id: e.reasons for e in result.excluded}
    assert excluded == {"bad": [NOT_ROBUSTLY_PROFITABLE]}


def test_candidate_over_l1_limit_is_excluded():
    report = _report(
        [
            _comparison("near", villain_ev=-1.0, worst_diff=0.5, l1_distance=0.2),
            _comparison("far", villain_ev=-1.0, worst_diff=0.5, l1_distance=0.8),
        ]
    )
    result = select_candidates(report, profit_tolerance=0.0, max_l1_distance=0.5)

    eligible_ids = {c.candidate.candidate_id for c in result.eligible}
    assert eligible_ids == {"near"}
    excluded = {e.comparison.candidate.candidate_id: e.reasons for e in result.excluded}
    assert excluded == {"far": [L1_DISTANCE_EXCEEDS_LIMIT]}


def test_multiple_reasons_are_reported():
    report = _report(
        [_comparison("both", villain_ev=-1.0, worst_diff=-0.5, l1_distance=0.8)]
    )
    result = select_candidates(report, profit_tolerance=0.0, max_l1_distance=0.5)
    assert result.eligible == []
    assert result.excluded[0].reasons == [
        NOT_ROBUSTLY_PROFITABLE,
        L1_DISTANCE_EXCEEDS_LIMIT,
    ]


def test_multiple_minimum_villain_ev_candidates_are_kept():
    report = _report(
        [
            _comparison("a", villain_ev=-2.0, worst_diff=0.5, l1_distance=0.2),
            _comparison("b", villain_ev=-2.0, worst_diff=0.5, l1_distance=0.4),
            _comparison("c", villain_ev=-1.0, worst_diff=0.5, l1_distance=0.2),
        ]
    )
    result = select_candidates(report, profit_tolerance=0.0)
    minimum_ids = {c.candidate.candidate_id for c in result.minimum_villain_ev_candidates}
    assert minimum_ids == {"a", "b"}


def test_no_eligible_candidates_returns_empty_selection():
    report = _report(
        [_comparison("bad", villain_ev=-2.0, worst_diff=-0.5, l1_distance=0.2)]
    )
    result = select_candidates(report, profit_tolerance=0.0)
    assert result.eligible == []
    assert result.minimum_villain_ev_candidates == []
    assert result.pareto_frontier == []
    assert result.has_eligible_candidates is False


def test_dominated_candidate_is_off_the_frontier():
    # "winner" is better on all three objectives than "loser".
    winner = _comparison("winner", villain_ev=-2.0, worst_diff=1.0, l1_distance=0.2)
    loser = _comparison("loser", villain_ev=-1.0, worst_diff=0.5, l1_distance=0.4)
    frontier_ids = {c.candidate.candidate_id for c in pareto_frontier([winner, loser])}
    assert frontier_ids == {"winner"}


def test_trade_off_candidates_are_all_kept_on_frontier():
    # A has lower Villain EV but larger L1; B has smaller L1 but higher Villain
    # EV; equal robust Hero EV. Neither dominates the other.
    a = _comparison("a", villain_ev=-2.0, worst_diff=0.5, l1_distance=0.4)
    b = _comparison("b", villain_ev=-1.0, worst_diff=0.5, l1_distance=0.2)
    frontier_ids = {c.candidate.candidate_id for c in pareto_frontier([a, b])}
    assert frontier_ids == {"a", "b"}


def test_equal_objective_candidates_are_all_kept():
    a = _comparison("a", villain_ev=-1.0, worst_diff=0.5, l1_distance=0.3)
    b = _comparison("b", villain_ev=-1.0, worst_diff=0.5, l1_distance=0.3)
    frontier_ids = {c.candidate.candidate_id for c in pareto_frontier([a, b])}
    assert frontier_ids == {"a", "b"}


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_invalid_profit_tolerance_is_rejected(bad):
    report = _report([_comparison("x", -1.0, 0.5, 0.2)])
    with pytest.raises(ValueError, match="profit_tolerance"):
        select_candidates(report, profit_tolerance=bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0])
def test_invalid_max_l1_distance_is_rejected(bad):
    report = _report([_comparison("x", -1.0, 0.5, 0.2)])
    with pytest.raises(ValueError, match="max_l1_distance"):
        select_candidates(report, profit_tolerance=0.0, max_l1_distance=bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0])
def test_invalid_tolerance_is_rejected(bad):
    report = _report([_comparison("x", -1.0, 0.5, 0.2)])
    with pytest.raises(ValueError, match="tolerance"):
        select_candidates(report, profit_tolerance=0.0, tolerance=bad)


def test_minimum_villain_ev_empty_input():
    assert select_minimum_villain_ev([]) == []


def test_integration_from_nuts_chop_candidate_library():
    tree = build_nuts_chop_river()
    baseline_hero = default_hero_strategy()
    baseline_villain = baseline_villain_strategy()
    candidates = generate_shift_candidates(tree, baseline_hero, [0.1, 0.2])
    report = compare_candidates(tree, baseline_hero, baseline_villain, candidates)

    # A permissive profit threshold keeps some candidates; the L1 cap of 0.3
    # excludes the shift=0.2 candidates (L1 distance 0.4).
    result = select_candidates(
        report, profit_tolerance=-2.0, max_l1_distance=0.3
    )

    assert len(result.eligible) + len(result.excluded) == len(report.comparisons)
    assert result.eligible  # some candidates pass the permissive threshold
    assert result.has_eligible_candidates is True

    known_reasons = {NOT_ROBUSTLY_PROFITABLE, L1_DISTANCE_EXCEEDS_LIMIT}
    for excluded in result.excluded:
        assert set(excluded.reasons) <= known_reasons

    eligible_set = {id(c) for c in result.eligible}
    assert all(id(c) in eligible_set for c in result.minimum_villain_ev_candidates)
    assert all(id(c) in eligible_set for c in result.pareto_frontier)

    # Every L1-excluded candidate indeed exceeds the cap.
    for excluded in result.excluded:
        if L1_DISTANCE_EXCEEDS_LIMIT in excluded.reasons:
            assert excluded.comparison.candidate.l1_distance > 0.3
