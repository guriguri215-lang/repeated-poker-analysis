"""Tests for the candidate-vs-baseline comparison API."""

import pytest

from candidate_library import baseline_villain_strategy
from nuts_chop_river import build_nuts_chop_river, default_hero_strategy

from repeated_poker import (
    compare_candidates,
    evaluate_fixed_profile,
    generate_shift_candidates,
    solve_exact_response,
)


def _report():
    tree = build_nuts_chop_river()
    baseline_hero = default_hero_strategy()
    baseline_villain = baseline_villain_strategy()
    candidates = generate_shift_candidates(tree, baseline_hero, [0.2])
    report = compare_candidates(tree, baseline_hero, baseline_villain, candidates)
    return tree, baseline_hero, baseline_villain, report


def test_baseline_value_matches_direct_evaluation():
    tree, baseline_hero, baseline_villain, report = _report()
    direct = evaluate_fixed_profile(tree, baseline_hero, baseline_villain)
    assert report.baseline_value == direct


def test_candidate_ev_differences_from_baseline():
    tree, _, baseline_villain, report = _report()
    baseline_value = report.baseline_value

    assert report.comparisons  # non-empty
    for comparison in report.comparisons:
        direct = evaluate_fixed_profile(
            tree, comparison.candidate.hero_strategy, baseline_villain
        )
        assert comparison.fixed_profile_value == direct
        assert comparison.hero_ev_diff_from_baseline == pytest.approx(
            direct.hero_ev - baseline_value.hero_ev
        )
        assert comparison.villain_ev_diff_from_baseline == pytest.approx(
            direct.villain_ev - baseline_value.villain_ev
        )


def test_post_response_diffs_match_exact_response():
    tree, _, _, report = _report()
    baseline_hero_ev = report.baseline_value.hero_ev

    for comparison in report.comparisons:
        exact = solve_exact_response(tree, comparison.candidate.hero_strategy)
        assert comparison.best_response.ev_h_worst == pytest.approx(exact.ev_h_worst)
        assert comparison.best_response.ev_h_best == pytest.approx(exact.ev_h_best)
        assert comparison.post_response_hero_ev_worst_diff == pytest.approx(
            exact.ev_h_worst - baseline_hero_ev
        )
        assert comparison.post_response_hero_ev_best_diff == pytest.approx(
            exact.ev_h_best - baseline_hero_ev
        )


def test_robustly_profitable_flag_matches_definition():
    _, _, _, report = _report()
    baseline_hero_ev = report.baseline_value.hero_ev
    for comparison in report.comparisons:
        expected = comparison.best_response.ev_h_worst > baseline_hero_ev
        assert comparison.robustly_profitable == expected
