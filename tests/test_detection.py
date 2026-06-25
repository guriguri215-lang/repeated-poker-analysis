"""Tests for the observable-distribution detection-time (T_detect) model."""

import math

import pytest

from repeated_poker import (
    HeroStrategy,
    HeroStrategyCandidate,
    calculate_candidate_local_detection,
    calculate_detection_time,
)


def test_binary_distribution_hand_computed():
    baseline = {"check": 0.8, "bet": 0.2}
    candidate = {"check": 0.5, "bet": 0.5}

    result = calculate_detection_time(
        baseline, candidate, log_likelihood_threshold=3.0
    )

    assert result.event_count == 2
    assert result.total_variation_distance == pytest.approx(0.3)
    # D(candidate||baseline) = 0.5*ln(0.5/0.8) + 0.5*ln(0.5/0.2).
    expected_kl = 0.5 * math.log(0.5 / 0.8) + 0.5 * math.log(0.5 / 0.2)
    assert result.kl_divergence_nats == pytest.approx(expected_kl)
    # ceil(3.0 / 0.22314355) = 14.
    assert result.required_observations == 14
    assert result.occurrence_probability_per_opportunity is None
    assert result.estimated_opportunities is None


def test_identical_distribution_has_zero_kl_and_no_required_observations():
    distribution = {"check": 0.7, "bet": 0.3}
    result = calculate_detection_time(
        dict(distribution),
        dict(distribution),
        log_likelihood_threshold=2.0,
        occurrence_probability_per_opportunity=0.5,
    )
    assert result.kl_divergence_nats == 0.0
    assert result.total_variation_distance == pytest.approx(0.0)
    assert result.required_observations is None
    # estimated_opportunities is None when required_observations is None.
    assert result.estimated_opportunities is None


def test_zero_baseline_positive_candidate_gives_infinite_kl():
    baseline = {"check": 1.0, "bet": 0.0}
    candidate = {"check": 0.5, "bet": 0.5}
    result = calculate_detection_time(
        baseline, candidate, log_likelihood_threshold=3.0
    )
    assert result.kl_divergence_nats == math.inf
    assert result.required_observations == 1
    assert result.total_variation_distance == pytest.approx(0.5)


def test_zero_candidate_term_is_skipped():
    baseline = {"check": 0.5, "bet": 0.5}
    candidate = {"check": 1.0, "bet": 0.0}
    result = calculate_detection_time(
        baseline, candidate, log_likelihood_threshold=1.0
    )
    # Only the "check" term contributes: 1.0 * ln(1.0 / 0.5) = ln 2.
    assert result.kl_divergence_nats == pytest.approx(math.log(2.0))
    assert result.required_observations == math.ceil(1.0 / math.log(2.0))


def test_estimated_opportunities_uses_occurrence_probability():
    baseline = {"check": 0.8, "bet": 0.2}
    candidate = {"check": 0.5, "bet": 0.5}
    result = calculate_detection_time(
        baseline,
        candidate,
        log_likelihood_threshold=3.0,
        occurrence_probability_per_opportunity=0.5,
    )
    assert result.required_observations == 14
    # ceil(14 / 0.5) = 28.
    assert result.estimated_opportunities == 28


def test_infinite_kl_estimated_opportunities():
    baseline = {"check": 1.0, "bet": 0.0}
    candidate = {"check": 0.5, "bet": 0.5}
    result = calculate_detection_time(
        baseline,
        candidate,
        log_likelihood_threshold=3.0,
        occurrence_probability_per_opportunity=0.25,
    )
    assert result.required_observations == 1
    assert result.estimated_opportunities == 4  # ceil(1 / 0.25)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

_VALID_BASELINE = {"check": 0.8, "bet": 0.2}
_VALID_CANDIDATE = {"check": 0.5, "bet": 0.5}


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_invalid_log_likelihood_threshold_is_rejected(bad):
    with pytest.raises(ValueError, match="log_likelihood_threshold"):
        calculate_detection_time(_VALID_BASELINE, _VALID_CANDIDATE, bad)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, math.nan, math.inf])
def test_invalid_occurrence_probability_is_rejected(bad):
    with pytest.raises(ValueError, match="occurrence_probability_per_opportunity"):
        calculate_detection_time(
            _VALID_BASELINE,
            _VALID_CANDIDATE,
            log_likelihood_threshold=3.0,
            occurrence_probability_per_opportunity=bad,
        )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0])
def test_invalid_tolerance_is_rejected(bad):
    with pytest.raises(ValueError, match="tolerance"):
        calculate_detection_time(
            _VALID_BASELINE, _VALID_CANDIDATE, log_likelihood_threshold=3.0, tolerance=bad
        )


def test_distribution_sum_mismatch_is_rejected():
    with pytest.raises(ValueError, match="sums to"):
        calculate_detection_time(
            {"check": 0.8, "bet": 0.3}, _VALID_CANDIDATE, log_likelihood_threshold=3.0
        )


def test_negative_probability_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        calculate_detection_time(
            {"check": 1.2, "bet": -0.2}, _VALID_CANDIDATE, log_likelihood_threshold=3.0
        )


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_probability_is_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        calculate_detection_time(
            {"check": bad, "bet": 0.2}, _VALID_CANDIDATE, log_likelihood_threshold=3.0
        )


def test_event_set_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same event set"):
        calculate_detection_time(
            {"check": 0.8, "bet": 0.2},
            {"check": 0.5, "raise": 0.5},
            log_likelihood_threshold=3.0,
        )


def test_missing_key_is_not_treated_as_zero():
    # The candidate omits "bet"; it must be rejected, not zero-filled.
    with pytest.raises(ValueError, match="same event set"):
        calculate_detection_time(
            {"check": 0.8, "bet": 0.2},
            {"check": 1.0},
            log_likelihood_threshold=3.0,
        )


# ---------------------------------------------------------------------------
# Candidate local detection helper
# ---------------------------------------------------------------------------


def _candidate(info_set, hero_probabilities):
    return HeroStrategyCandidate(
        candidate_id="cand-1",
        info_set=info_set,
        source_action="check",
        target_action="bet",
        shift_amount=0.3,
        hero_strategy=HeroStrategy(hero_probabilities),
        l1_distance=0.6,
    )


def test_candidate_local_detection_uses_info_set_distributions():
    baseline_hero = HeroStrategy(
        {"H1": {"check": 0.8, "bet": 0.2}, "Other": {"x": 1.0}}
    )
    candidate = _candidate(
        "H1", {"H1": {"check": 0.5, "bet": 0.5}, "Other": {"x": 1.0}}
    )

    result = calculate_candidate_local_detection(
        baseline_hero, candidate, log_likelihood_threshold=3.0
    )
    expected = calculate_detection_time(
        {"check": 0.8, "bet": 0.2},
        {"check": 0.5, "bet": 0.5},
        log_likelihood_threshold=3.0,
    )
    assert result == expected


def test_candidate_local_detection_missing_info_set_is_rejected():
    baseline_hero = HeroStrategy({"Other": {"x": 1.0}})
    candidate = _candidate("H1", {"H1": {"check": 0.5, "bet": 0.5}})
    with pytest.raises(ValueError, match="missing information set"):
        calculate_candidate_local_detection(
            baseline_hero, candidate, log_likelihood_threshold=3.0
        )
