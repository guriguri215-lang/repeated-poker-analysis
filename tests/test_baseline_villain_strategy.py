"""Tests for the optional explicit ``baseline_villain_strategy`` input (M2-T1).

These cover three things:

* backward compatibility -- omitting the field keeps the automatic
  best-response baseline and the ``auto_best_response`` provenance;
* explicit reflection -- an explicit profile that differs from the automatic
  best response moves the fixed-profile baseline value; and
* validation -- the same numeric / information-set guards the Hero baseline
  gets, with no silent fallback, plus the scenario-form rejection.
"""

import math

import pytest

from repeated_poker import (
    HeroStrategyCandidate,
    HeroStrategy,
    build_river_steal_game_from_scenario,
    compare_candidates,
    evaluate_fixed_profile,
    river_scenario_from_dict,
)
from repeated_poker import scenario_form as sf
from repeated_poker.scenario_pipeline import (
    RiverScenarioAnalysisConfig,
    run_river_scenario_analysis,
)
from repeated_poker.report_export import analysis_result_to_dict


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------


def _single_hand(**overrides) -> dict:
    """A single-hand scenario where baseline Hero folds to a bet.

    With a pure-fold Hero the automatic best-response Villain bets (stealing
    Hero's committed chip), so an explicit "always check" Villain is a genuinely
    different baseline. This makes the baseline value observably profile-
    dependent by hand: auto baseline Hero EV = -1.0 (bet, Hero folds), explicit
    check-check baseline Hero EV = 0.0 (rake-free chop returns each chip).
    """

    data = {
        "scenario_id": "explicit_villain_single_hand",
        "rake": {"rate": 0.0, "cap": None},
        "initial_commitment": {"hero": 1.0, "villain": 1.0},
        "bet_size": 2.0,
        "showdown": "chop",
        "baseline_hero_strategy": {"IP_vs_bet": {"call": 0.0, "fold": 1.0}},
        "candidate_generation": {"shift_amounts": [1.0]},
        "repeated": {"horizons": [10], "discount": 1.0},
    }
    data.update(overrides)
    return data


def _matrix(**overrides) -> dict:
    """A minimal showdown-matrix scenario with two Villain buckets."""

    data = {
        "scenario_id": "explicit_villain_matrix",
        "rake": {"rate": 0.0, "cap": None},
        "initial_commitment": {"hero": 1.0, "villain": 1.0},
        "bet_size": 2.0,
        "hero_range": [
            {"hand_id": "h1", "weight": 1.0, "baseline_strategy": {"call": 1.0, "fold": 0.0}}
        ],
        "villain_range": [
            {"hand_id": "v1", "weight": 0.5},
            {"hand_id": "v2", "weight": 0.5},
        ],
        "showdown_matrix": {"h1": {"v1": "hero", "v2": "villain"}},
        "candidate_generation": {"shift_amounts": [1.0]},
    }
    data.update(overrides)
    return data


def _build(data: dict):
    return build_river_steal_game_from_scenario(river_scenario_from_dict(data))


# ---------------------------------------------------------------------------
# 1. Backward compatibility (field omitted)
# ---------------------------------------------------------------------------


def test_omitted_field_keeps_auto_best_response_provenance():
    scenario = river_scenario_from_dict(_single_hand())
    assert scenario.baseline_villain_strategy is None

    build = _build(_single_hand())
    assert build.baseline_villain_source == "auto_best_response"
    assert build.metadata["baseline_villain_source"] == "auto_best_response"
    # The automatic best response to a pure-fold Hero is to bet.
    assert build.baseline_villain_strategy.probabilities["OOP_river"] == {
        "check": 0.0,
        "bet": 1.0,
    }


def test_auto_baseline_value_unchanged_by_the_new_code_path():
    build = _build(_single_hand())
    value = evaluate_fixed_profile(
        build.tree, build.baseline_hero_strategy, build.baseline_villain_strategy
    )
    # Villain bets, Hero folds, Villain wins Hero's committed chip.
    assert value.hero_ev == pytest.approx(-1.0)


def test_analysis_export_reports_auto_provenance():
    result = run_river_scenario_analysis(
        river_scenario_from_dict(_single_hand()), RiverScenarioAnalysisConfig()
    )
    payload = analysis_result_to_dict(result)
    assert payload["build_metadata"]["baseline_villain_source"] == "auto_best_response"


# ---------------------------------------------------------------------------
# 2. Explicit baseline is reflected in the baseline value
# ---------------------------------------------------------------------------


def test_explicit_profile_sets_provenance_and_distribution():
    build = _build(
        _single_hand(baseline_villain_strategy={"OOP_river": {"check": 0.7, "bet": 0.3}})
    )
    assert build.baseline_villain_source == "explicit"
    assert build.metadata["baseline_villain_source"] == "explicit"
    assert build.baseline_villain_strategy.probabilities["OOP_river"] == {
        "check": 0.7,
        "bet": 0.3,
    }


def test_explicit_profile_moves_fixed_profile_baseline_value():
    auto = _build(_single_hand())
    explicit = _build(
        _single_hand(baseline_villain_strategy={"OOP_river": {"check": 1.0, "bet": 0.0}})
    )

    auto_value = evaluate_fixed_profile(
        auto.tree, auto.baseline_hero_strategy, auto.baseline_villain_strategy
    )
    explicit_value = evaluate_fixed_profile(
        explicit.tree, explicit.baseline_hero_strategy, explicit.baseline_villain_strategy
    )

    # Auto BR bets and Hero folds (-1.0); the explicit check-check baseline is a
    # rake-free chop (0.0). The baseline value tracks the chosen profile.
    assert auto_value.hero_ev == pytest.approx(-1.0)
    assert explicit_value.hero_ev == pytest.approx(0.0)
    assert auto_value.hero_ev != pytest.approx(explicit_value.hero_ev)


def test_explicit_profile_moves_report_baseline_value():
    explicit = _build(
        _single_hand(baseline_villain_strategy={"OOP_river": {"check": 1.0, "bet": 0.0}})
    )
    # One trivial candidate so compare_candidates has something to report; only
    # the baseline_value is asserted here.
    candidate = HeroStrategyCandidate(
        candidate_id="baseline_copy",
        info_set="IP_vs_bet",
        source_action="fold",
        target_action="call",
        shift_amount=0.0,
        hero_strategy=HeroStrategy(
            probabilities={"IP_vs_bet": {"call": 0.0, "fold": 1.0}}
        ),
        l1_distance=0.0,
    )
    report = compare_candidates(
        explicit.tree,
        explicit.baseline_hero_strategy,
        explicit.baseline_villain_strategy,
        [candidate],
    )
    assert report.baseline_value.hero_ev == pytest.approx(0.0)


def test_explicit_profile_need_not_be_a_best_response():
    # A dominated-for-Villain baseline (always check when betting is strictly
    # better) is accepted: the field asserts no best-response / equilibrium
    # property, only that it is the chosen comparison profile.
    build = _build(
        _single_hand(baseline_villain_strategy={"OOP_river": {"check": 1.0, "bet": 0.0}})
    )
    assert build.baseline_villain_source == "explicit"


# ---------------------------------------------------------------------------
# 3. Missing-action semantics (a legal action omitted is taken as 0)
# ---------------------------------------------------------------------------


def test_missing_action_defaults_to_zero():
    build = _build(
        _single_hand(baseline_villain_strategy={"OOP_river": {"bet": 1.0}})
    )
    assert build.baseline_villain_strategy.probabilities["OOP_river"] == {
        "check": 0.0,
        "bet": 1.0,
    }


# ---------------------------------------------------------------------------
# 3. Validation (no silent fallback)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "profile, needle",
    [
        # Unknown Villain information set.
        ({"OOP_river": {"check": 1.0}, "OOP_ghost": {"check": 1.0}}, "unknown Villain"),
        # Hero information set gets a dedicated message.
        ({"IP_vs_bet": {"call": 1.0, "fold": 0.0}}, "Hero information sets"),
        # Unknown action.
        ({"OOP_river": {"check": 0.5, "raise": 0.5}}, "unknown actions"),
        # Negative probability.
        ({"OOP_river": {"check": -0.1, "bet": 1.1}}, "non-negative"),
        # Not normalised.
        ({"OOP_river": {"check": 0.5, "bet": 0.2}}, "sum to"),
    ],
)
def test_invalid_explicit_profiles_are_rejected(profile, needle):
    with pytest.raises(ValueError) as excinfo:
        _build(_single_hand(baseline_villain_strategy=profile))
    assert needle in str(excinfo.value)


@pytest.mark.parametrize("bad", [True, False])
def test_boolean_probability_is_rejected(bad):
    with pytest.raises(ValueError, match="must be a number"):
        _build(_single_hand(baseline_villain_strategy={"OOP_river": {"check": bad, "bet": 0.0}}))


@pytest.mark.parametrize("bad", ["0.5", None, [0.5]])
def test_non_number_probability_is_rejected(bad):
    with pytest.raises(ValueError, match="must be a number"):
        _build(_single_hand(baseline_villain_strategy={"OOP_river": {"check": bad, "bet": 0.5}}))


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), -float("inf")])
def test_non_finite_probability_is_rejected(bad):
    assert not math.isfinite(bad)
    with pytest.raises(ValueError, match="finite"):
        _build(_single_hand(baseline_villain_strategy={"OOP_river": {"check": bad, "bet": 0.0}}))


def test_missing_villain_info_set_is_rejected_no_silent_fallback():
    # Matrix mode has OOP_river::v1 and OOP_river::v2; omitting v2 must be an
    # error rather than a silent completion from the automatic baseline.
    with pytest.raises(ValueError, match="missing Villain information set"):
        _build(_matrix(baseline_villain_strategy={"OOP_river::v1": {"check": 1.0, "bet": 0.0}}))


def test_empty_profile_is_rejected_at_parse():
    with pytest.raises(ValueError, match="must not be empty"):
        river_scenario_from_dict(_single_hand(baseline_villain_strategy={}))


@pytest.mark.parametrize("bad", [[], "OOP_river", 3])
def test_non_object_profile_is_rejected_at_parse(bad):
    with pytest.raises(ValueError, match="must be an object"):
        river_scenario_from_dict(_single_hand(baseline_villain_strategy=bad))


def test_non_object_distribution_is_rejected_at_parse():
    with pytest.raises(ValueError, match="mapping of action to probability"):
        river_scenario_from_dict(
            _single_hand(baseline_villain_strategy={"OOP_river": [("check", 1.0)]})
        )


def test_explicit_profile_supported_in_matrix_mode():
    build = _build(
        _matrix(
            baseline_villain_strategy={
                "OOP_river::v1": {"check": 1.0, "bet": 0.0},
                "OOP_river::v2": {"check": 0.25, "bet": 0.75},
            }
        )
    )
    assert build.baseline_villain_source == "explicit"
    probs = build.baseline_villain_strategy.probabilities
    assert probs["OOP_river::v1"] == {"check": 1.0, "bet": 0.0}
    assert probs["OOP_river::v2"] == {"check": 0.25, "bet": 0.75}


# ---------------------------------------------------------------------------
# 4. Scenario-form round-trip rejection (avoid silent data loss)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "from_dict",
    [
        sf.single_hand_form_from_dict,
        sf.hero_range_form_from_dict,
        sf.showdown_matrix_form_from_dict,
        sf.equity_matrix_form_from_dict,
        sf.betting_tree_form_from_dict,
    ],
)
def test_scenario_forms_reject_explicit_baseline_villain(from_dict):
    data = _single_hand(baseline_villain_strategy={"OOP_river": {"check": 0.7, "bet": 0.3}})
    with pytest.raises(ValueError, match="does not support 'baseline_villain_strategy'"):
        from_dict(data)


def test_single_hand_form_round_trip_still_works_without_the_field():
    form = sf.single_hand_form_from_dict(_single_hand())
    restored = sf.single_hand_form_to_dict(form)
    assert "baseline_villain_strategy" not in restored
    assert restored["scenario_id"] == "explicit_villain_single_hand"
