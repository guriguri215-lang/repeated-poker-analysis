"""Tests for the adaptation-deadline (T_deadline) repeated-game measure."""

import math

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
    calculate_adaptation_deadline,
    calculate_candidate_adaptation_deadline,
    calculate_candidate_adaptation_deadlines,
    compare_candidates,
    generate_shift_candidates,
)


def test_undiscounted_hand_computed_values_and_deadline():
    # b=1, a=3, l=0, N=3, delta=1. V_base = 3. V_lock(m) = 3*(m-1).
    result = calculate_adaptation_deadline(
        baseline_hero_ev=1.0,
        pre_adaptation_hero_ev=3.0,
        post_adaptation_hero_ev=0.0,
        horizon=3,
        discount=1.0,
    )

    assert result.baseline_total_hero_ev == pytest.approx(3.0)
    locked = [row.locked_total_hero_ev for row in result.timing]
    assert locked == pytest.approx([0.0, 3.0, 6.0, 9.0])  # m = 1..4
    flags = [row.is_at_least_baseline for row in result.timing]
    assert flags == [False, True, True, True]
    deltas = [row.delta_from_baseline for row in result.timing]
    assert deltas == pytest.approx([-3.0, 0.0, 3.0, 6.0])

    assert result.t_deadline == 3  # latest m in 1..N with V_lock >= V_base
    assert result.never_adapts_total_hero_ev == pytest.approx(9.0)  # m = N+1 row
    assert result.timing[-1].adaptation_opportunity == 4  # N + 1


def test_discounted_hand_computed_values_and_deadline():
    # b=1, a=3, l=0, N=3, delta=0.5.
    # G(1)=1, G(2)=1.5, G(3)=1.75. V_base = 1.75. V_lock(m) = 3*G(m-1).
    result = calculate_adaptation_deadline(
        baseline_hero_ev=1.0,
        pre_adaptation_hero_ev=3.0,
        post_adaptation_hero_ev=0.0,
        horizon=3,
        discount=0.5,
    )

    assert result.baseline_total_hero_ev == pytest.approx(1.75)
    locked = [row.locked_total_hero_ev for row in result.timing]
    assert locked == pytest.approx([0.0, 3.0, 4.5, 5.25])  # m = 1..4
    flags = [row.is_at_least_baseline for row in result.timing]
    assert flags == [False, True, True, True]

    assert result.t_deadline == 3
    assert result.never_adapts_total_hero_ev == pytest.approx(5.25)


def test_no_deadline_returns_none():
    # b=10, a=1, l=0, N=3, delta=1. V_base = 30; V_lock(m) = m-1 < 30 always.
    result = calculate_adaptation_deadline(
        baseline_hero_ev=10.0,
        pre_adaptation_hero_ev=1.0,
        post_adaptation_hero_ev=0.0,
        horizon=3,
    )
    assert result.t_deadline is None
    assert all(not row.is_at_least_baseline for row in result.timing[:3])


def test_interior_deadline_scans_all_opportunities():
    # b=0, a=-1, l=2, N=5, delta=1. V_lock(m) = 10 - 3*(m-1), V_base = 0.
    # m=1..5 -> 10,7,4,1,-2; passing m: 1..4; latest is 4 (interior, < N).
    result = calculate_adaptation_deadline(
        baseline_hero_ev=0.0,
        pre_adaptation_hero_ev=-1.0,
        post_adaptation_hero_ev=2.0,
        horizon=5,
        discount=1.0,
    )
    assert result.t_deadline == 4
    assert result.never_adapts_total_hero_ev == pytest.approx(-5.0)


def test_m_equals_one_is_all_post_adaptation():
    # m=1 means Villain adapts from the very first opportunity: all 'l'.
    result = calculate_adaptation_deadline(
        baseline_hero_ev=0.0,
        pre_adaptation_hero_ev=5.0,
        post_adaptation_hero_ev=2.0,
        horizon=4,
        discount=1.0,
    )
    first_row = result.timing[0]
    assert first_row.adaptation_opportunity == 1
    assert first_row.locked_total_hero_ev == pytest.approx(2.0 * 4)  # l * N


def test_timing_has_horizon_plus_one_rows():
    result = calculate_adaptation_deadline(0.0, 1.0, 0.0, horizon=6)
    assert len(result.timing) == 7
    assert [row.adaptation_opportunity for row in result.timing] == list(range(1, 8))


@pytest.mark.parametrize("bad_horizon", [0, -1, 2.5, True])
def test_invalid_horizon_is_rejected(bad_horizon):
    with pytest.raises(ValueError, match="horizon"):
        calculate_adaptation_deadline(0.0, 1.0, 0.0, horizon=bad_horizon)


def test_horizon_safety_limit_is_enforced():
    with pytest.raises(ValueError, match="safety limit"):
        calculate_adaptation_deadline(0.0, 1.0, 0.0, horizon=10, max_horizon=5)


@pytest.mark.parametrize("bad_discount", [0.0, -0.1, 1.5, math.nan, math.inf])
def test_invalid_discount_is_rejected(bad_discount):
    with pytest.raises(ValueError, match="discount"):
        calculate_adaptation_deadline(0.0, 1.0, 0.0, horizon=3, discount=bad_discount)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_ev_inputs_are_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        calculate_adaptation_deadline(bad, 1.0, 0.0, horizon=3)
    with pytest.raises(ValueError, match="finite"):
        calculate_adaptation_deadline(0.0, bad, 0.0, horizon=3)
    with pytest.raises(ValueError, match="finite"):
        calculate_adaptation_deadline(0.0, 1.0, bad, horizon=3)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0])
def test_invalid_tolerance_is_rejected(bad):
    with pytest.raises(ValueError, match="tolerance"):
        calculate_adaptation_deadline(0.0, 1.0, 0.0, horizon=3, tolerance=bad)


# ---------------------------------------------------------------------------
# CandidateComparison integration
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


def _comparison(pre_adaptation_hero_ev: float) -> CandidateComparison:
    candidate = HeroStrategyCandidate(
        candidate_id="cand-1",
        info_set="H",
        source_action="a",
        target_action="b",
        shift_amount=0.1,
        hero_strategy=HeroStrategy({}),
        l1_distance=0.2,
    )
    return CandidateComparison(
        candidate=candidate,
        fixed_profile_value=FixedProfileValue(
            hero_ev=pre_adaptation_hero_ev, villain_ev=0.0, house_rake=0.0
        ),
        villain_ev_diff_from_baseline=0.0,
        hero_ev_diff_from_baseline=0.0,
        best_response=_best_response(ev_h_worst=-1.0, ev_h_best=0.5),
        post_response_hero_ev_worst_diff=0.0,
        post_response_hero_ev_best_diff=0.0,
        robustly_profitable=False,
    )


def _report(comparison: CandidateComparison) -> CandidateComparisonReport:
    return CandidateComparisonReport(
        baseline_value=FixedProfileValue(hero_ev=1.0, villain_ev=0.0, house_rake=0.0),
        comparisons=[comparison],
    )


def test_candidate_integration_worst_mode_uses_worst_post_adaptation_ev():
    comparison = _comparison(pre_adaptation_hero_ev=2.0)
    report = _report(comparison)

    deadline = calculate_candidate_adaptation_deadline(
        report, comparison, horizon=3, response_mode="worst"
    )

    assert deadline.candidate_id == "cand-1"
    assert deadline.response_mode == "worst"
    assert deadline.result.post_adaptation_hero_ev == pytest.approx(-1.0)
    assert deadline.result.pre_adaptation_hero_ev == pytest.approx(2.0)
    assert deadline.result.baseline_total_hero_ev == pytest.approx(3.0)  # b=1, N=3


def test_candidate_integration_best_mode_uses_best_post_adaptation_ev():
    comparison = _comparison(pre_adaptation_hero_ev=2.0)
    report = _report(comparison)

    deadline = calculate_candidate_adaptation_deadline(
        report, comparison, horizon=3, response_mode="best"
    )
    assert deadline.response_mode == "best"
    assert deadline.result.post_adaptation_hero_ev == pytest.approx(0.5)


def test_candidate_integration_default_mode_is_worst():
    comparison = _comparison(pre_adaptation_hero_ev=2.0)
    report = _report(comparison)
    deadline = calculate_candidate_adaptation_deadline(report, comparison, horizon=3)
    assert deadline.response_mode == "worst"
    assert deadline.result.post_adaptation_hero_ev == pytest.approx(-1.0)


def test_unknown_response_mode_is_rejected():
    comparison = _comparison(pre_adaptation_hero_ev=2.0)
    report = _report(comparison)
    with pytest.raises(ValueError, match="unknown response_mode"):
        calculate_candidate_adaptation_deadline(
            report, comparison, horizon=3, response_mode="mixed"
        )


def test_integration_from_nuts_chop_candidate_library():
    tree = build_nuts_chop_river()
    baseline_hero = default_hero_strategy()
    baseline_villain = baseline_villain_strategy()
    candidates = generate_shift_candidates(tree, baseline_hero, [0.1, 0.2])
    report = compare_candidates(tree, baseline_hero, baseline_villain, candidates)

    deadlines = calculate_candidate_adaptation_deadlines(report, horizon=5)

    assert len(deadlines) == len(report.comparisons)
    for deadline, comparison in zip(deadlines, report.comparisons):
        assert deadline.candidate_id == comparison.candidate.candidate_id
        assert deadline.response_mode == "worst"
        # Worst mode uses the post-response worst-case Hero EV as 'l'.
        assert deadline.result.post_adaptation_hero_ev == pytest.approx(
            comparison.best_response.ev_h_worst
        )
        assert deadline.result.pre_adaptation_hero_ev == pytest.approx(
            comparison.fixed_profile_value.hero_ev
        )
        # delta=1 so the baseline total is b * N.
        assert deadline.result.baseline_total_hero_ev == pytest.approx(
            report.baseline_value.hero_ev * 5
        )
        assert len(deadline.result.timing) == 6  # m = 1..N+1
