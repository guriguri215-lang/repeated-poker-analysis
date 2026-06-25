"""Tests for the pre-comparison candidate filter."""

import copy
import math

import pytest

from nuts_chop_river import build_nuts_chop_river, default_hero_strategy

from repeated_poker import (
    INFO_SET_NOT_ALLOWED,
    L1_DISTANCE_EXCEEDS_LIMIT,
    REQUIRED_OBSERVATIONS_BELOW_LIMIT,
    HeroStrategy,
    HeroStrategyCandidate,
    filter_candidates,
    generate_shift_candidates,
)

# A baseline that places zero on "bet", so any candidate with positive "bet"
# yields an infinite KL (required_observations == 1).
_DETECTION_BASELINE = HeroStrategy(
    {"H": {"check": 1.0, "bet": 0.0}, "Z": {"check": 1.0, "bet": 0.0}}
)


def _candidate(candidate_id, info_set="H", l1_distance=0.2, dist=None):
    if dist is None:
        dist = {"check": 0.5, "bet": 0.5}
    return HeroStrategyCandidate(
        candidate_id=candidate_id,
        info_set=info_set,
        source_action="check",
        target_action="bet",
        shift_amount=0.1,
        hero_strategy=HeroStrategy({info_set: dist}),
        l1_distance=l1_distance,
    )


def test_no_filters_keeps_all():
    candidates = [_candidate("c1"), _candidate("c2")]
    result = filter_candidates(candidates)
    assert [c.candidate_id for c in result.kept] == ["c1", "c2"]
    assert result.excluded == []
    assert result.summary_counts.total == 2
    assert result.summary_counts.kept == 2
    assert result.summary_counts.excluded == 0


def test_allowed_info_sets_excludes_other_info_sets():
    candidates = [_candidate("h", info_set="H"), _candidate("z", info_set="Z")]
    result = filter_candidates(candidates, allowed_info_sets={"H"})
    assert [c.candidate_id for c in result.kept] == ["h"]
    assert len(result.excluded) == 1
    assert result.excluded[0].candidate.candidate_id == "z"
    assert result.excluded[0].reasons == [INFO_SET_NOT_ALLOWED]


def test_allowed_info_sets_rejects_bare_string():
    with pytest.raises(ValueError, match="bare string"):
        filter_candidates([_candidate("c")], allowed_info_sets="H")


def test_allowed_info_sets_rejects_non_iterable():
    with pytest.raises(ValueError, match="allowed_info_sets"):
        filter_candidates([_candidate("c")], allowed_info_sets=123)


def test_allowed_info_sets_accepts_list():
    candidates = [_candidate("h", info_set="H"), _candidate("z", info_set="Z")]
    result = filter_candidates(candidates, allowed_info_sets=["H"])
    assert [c.candidate_id for c in result.kept] == ["h"]
    assert result.excluded[0].candidate.candidate_id == "z"
    assert result.excluded[0].reasons == [INFO_SET_NOT_ALLOWED]


def test_empty_allowed_info_sets_excludes_all():
    candidates = [_candidate("c1"), _candidate("c2")]
    result = filter_candidates(candidates, allowed_info_sets=set())
    assert result.kept == []
    assert result.summary_counts.excluded == 2


def test_max_l1_distance_excludes():
    candidates = [_candidate("near", l1_distance=0.2), _candidate("far", l1_distance=0.8)]
    result = filter_candidates(candidates, max_l1_distance=0.5)
    assert [c.candidate_id for c in result.kept] == ["near"]
    assert result.excluded[0].candidate.candidate_id == "far"
    assert result.excluded[0].reasons == [L1_DISTANCE_EXCEEDS_LIMIT]


def test_min_required_observations_excludes_short_detection():
    # KL == inf -> required_observations == 1, which is below the limit of 5.
    candidate = _candidate("c", dist={"check": 0.5, "bet": 0.5})
    result = filter_candidates(
        [candidate],
        min_required_observations=5,
        baseline_hero_strategy=_DETECTION_BASELINE,
        detection_log_likelihood_threshold=3.0,
    )
    assert result.kept == []
    assert result.excluded[0].reasons == [REQUIRED_OBSERVATIONS_BELOW_LIMIT]


def test_none_required_observations_is_not_excluded():
    # Identical distribution -> KL == 0 -> required_observations is None.
    candidate = _candidate("c", dist={"check": 1.0, "bet": 0.0})
    result = filter_candidates(
        [candidate],
        min_required_observations=1000,
        baseline_hero_strategy=_DETECTION_BASELINE,
        detection_log_likelihood_threshold=3.0,
    )
    assert [c.candidate_id for c in result.kept] == ["c"]
    assert result.excluded == []


def test_multiple_reasons_are_all_recorded():
    candidate = _candidate("z", info_set="Z", l1_distance=0.8, dist={"check": 0.5, "bet": 0.5})
    result = filter_candidates(
        [candidate],
        allowed_info_sets={"H"},
        max_l1_distance=0.5,
        min_required_observations=5,
        baseline_hero_strategy=_DETECTION_BASELINE,
        detection_log_likelihood_threshold=3.0,
    )
    assert result.excluded[0].reasons == [
        INFO_SET_NOT_ALLOWED,
        L1_DISTANCE_EXCEEDS_LIMIT,
        REQUIRED_OBSERVATIONS_BELOW_LIMIT,
    ]


def test_input_order_is_preserved():
    candidates = [
        _candidate("c1", l1_distance=0.2),
        _candidate("c2", l1_distance=0.8),
        _candidate("c3", l1_distance=0.2),
    ]
    result = filter_candidates(candidates, max_l1_distance=0.5)
    assert [c.candidate_id for c in result.kept] == ["c1", "c3"]
    assert [e.candidate.candidate_id for e in result.excluded] == ["c2"]


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_invalid_min_required_observations_is_rejected(bad):
    with pytest.raises(ValueError, match="min_required_observations"):
        filter_candidates(
            [_candidate("c")],
            min_required_observations=bad,
            baseline_hero_strategy=_DETECTION_BASELINE,
            detection_log_likelihood_threshold=3.0,
        )


def test_detection_filter_requires_baseline_hero_strategy():
    with pytest.raises(ValueError, match="baseline_hero_strategy is required"):
        filter_candidates(
            [_candidate("c")],
            min_required_observations=5,
            detection_log_likelihood_threshold=3.0,
        )


def test_detection_filter_requires_threshold():
    with pytest.raises(ValueError, match="detection_log_likelihood_threshold is required"):
        filter_candidates(
            [_candidate("c")],
            min_required_observations=5,
            baseline_hero_strategy=_DETECTION_BASELINE,
        )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0])
def test_invalid_max_l1_distance_is_rejected(bad):
    with pytest.raises(ValueError, match="max_l1_distance"):
        filter_candidates([_candidate("c")], max_l1_distance=bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0])
def test_invalid_tolerance_is_rejected(bad):
    with pytest.raises(ValueError, match="tolerance"):
        filter_candidates([_candidate("c")], tolerance=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_invalid_detection_threshold_is_rejected(bad):
    with pytest.raises(ValueError, match="log_likelihood_threshold"):
        filter_candidates(
            [_candidate("c")],
            min_required_observations=5,
            baseline_hero_strategy=_DETECTION_BASELINE,
            detection_log_likelihood_threshold=bad,
        )


def test_candidates_are_not_mutated():
    candidate = _candidate("c", l1_distance=0.8)
    before = copy.deepcopy(candidate.hero_strategy.probabilities)
    result = filter_candidates([candidate], max_l1_distance=0.5)
    # Same object kept in the excluded list, unmodified.
    assert result.excluded[0].candidate is candidate
    assert candidate.hero_strategy.probabilities == before


def test_integration_with_nuts_chop_candidates():
    tree = build_nuts_chop_river()
    baseline_hero = default_hero_strategy()
    candidates = generate_shift_candidates(tree, baseline_hero, [0.1, 0.2])

    result = filter_candidates(
        candidates,
        max_l1_distance=0.3,
        min_required_observations=5,
        baseline_hero_strategy=baseline_hero,
        detection_log_likelihood_threshold=3.0,
    )

    assert result.summary_counts.total == len(candidates)
    assert result.summary_counts.kept + result.summary_counts.excluded == len(candidates)
    # Order is preserved: kept and excluded are subsequences of the input.
    kept_ids = [c.candidate_id for c in result.kept]
    excluded_ids = [e.candidate.candidate_id for e in result.excluded]
    input_ids = [c.candidate_id for c in candidates]
    assert kept_ids == [cid for cid in input_ids if cid in set(kept_ids)]
    assert excluded_ids == [cid for cid in input_ids if cid in set(excluded_ids)]

    # The call->fold candidates have an infinite KL (required_observations == 1),
    # so they are excluded for being below the detection minimum.
    for excluded in result.excluded:
        if "call->fold" in excluded.candidate.candidate_id:
            assert REQUIRED_OBSERVATIONS_BELOW_LIMIT in excluded.reasons
