"""Focused M32 robust-all integration tests and implementer-owned oracles."""

from __future__ import annotations

import copy
import inspect
import json
import math
import subprocess
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path

import pytest

import repeated_poker.three_player_candidate_repeated as module
from repeated_poker.automatic_commitment_selection import (
    AUTOMATIC_COMMITMENT_SELECTION_ROW_STATUS_SELECTED,
    NO_BENEFICIAL_COMMITMENT,
    AutomaticCommitmentSelectionConfig,
)
from repeated_poker.three_player_candidate_repeated import (
    BASELINE_RESPONSE_FAILURE,
    CANDIDATE_RESPONSE_FAILURE,
    CAP_EXCEEDED,
    EXACT_THREE_PLAYER_CANDIDATE_REPEATED_COMPLETE,
    INVALID_INPUT,
    SCALAR_PROJECTION_FAILURE,
    STALE_INPUT,
    UNSUPPORTED_MODE,
    ThreePlayerCandidateGenerationConfig,
    ThreePlayerCandidateRepeatedLimits,
    ThreePlayerCandidateRepeatedPins,
    ThreePlayerRepeatedConfig,
    evaluate_three_player_candidate_repeated,
    exact_three_player_candidate_repeated_json,
)
from repeated_poker.three_player_response import (
    EXACT_CORRESPONDENCE_COMPLETE,
)
from repeated_poker.three_player_river_rake import (
    AwardShare,
    ExactBehaviorPolicy,
    OpponentInitialProfile,
    RiverAction,
    RiverChanceNode,
    RiverChanceOutcome,
    RiverDecisionNode,
    RiverObservation,
    RiverTerminalNode,
    ScenarioResponseError,
    ScenarioResponseResult,
    ThreePlayerRiverRakeScenario,
    create_perfect_recall_attestation,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "repeated_poker" / "three_player_candidate_repeated.py"
TEST = ROOT / "tests" / "test_three_player_candidate_repeated.py"
PROTECTED = (
    ROOT / "src" / "repeated_poker" / "three_player_river_rake.py",
    ROOT / "tests" / "test_three_player_river_rake.py",
    ROOT / "src" / "repeated_poker" / "three_player_response.py",
    ROOT / "tests" / "test_three_player_response.py",
    ROOT / "src" / "repeated_poker" / "automatic_commitment_selection.py",
    ROOT / "tests" / "test_automatic_commitment_selection.py",
    ROOT / "src" / "repeated_poker" / "repeated.py",
    ROOT / "tests" / "test_repeated.py",
    ROOT / "src" / "repeated_poker" / "__init__.py",
)


def observation(suffix: str) -> RiverObservation:
    return RiverObservation(
        f"public-{suffix}",
        {
            "H": f"hero-{suffix}",
            "O1": f"o1-{suffix}",
            "O2": f"o2-{suffix}",
        },
    )


def hero_branch(suffix: str) -> RiverDecisionNode:
    showdown = RiverTerminalNode(
        f"showdown-{suffix}", "showdown", (AwardShare("H", "1"),)
    )
    o2_check = RiverDecisionNode(
        f"o2-check-{suffix}",
        "O2",
        f"O2_after_check_{suffix}",
        (RiverAction("check", "check", None, showdown),),
    )
    o1_check = RiverDecisionNode(
        f"o1-check-{suffix}",
        "O1",
        f"O1_after_check_{suffix}",
        (RiverAction("check", "check", None, o2_check),),
    )
    fold = RiverTerminalNode(f"fold-{suffix}", "fold")
    o2_fold = RiverDecisionNode(
        f"o2-fold-{suffix}",
        "O2",
        f"O2_after_bet_{suffix}",
        (RiverAction("fold", "fold", None, fold),),
    )
    o1_fold = RiverDecisionNode(
        f"o1-fold-{suffix}",
        "O1",
        f"O1_after_bet_{suffix}",
        (RiverAction("fold", "fold", None, o2_fold),),
    )
    return RiverDecisionNode(
        f"hero-{suffix}",
        "H",
        f"H_{suffix}",
        (
            RiverAction("check", "check", None, o1_check),
            RiverAction("bet", "bet", "20", o1_fold),
        ),
    )


def scenario_one(*, reverse_actions: bool = False) -> ThreePlayerRiverRakeScenario:
    root = hero_branch("one")
    if reverse_actions:
        root = replace(root, actions=tuple(reversed(root.actions)))
    return ThreePlayerRiverRakeScenario(
        root=root,
        button_player_id="H",
        seat_order=("H", "O1", "O2"),
        river_action_order=("H", "O1", "O2"),
        initial_observation=observation("one"),
        initial_pot="30",
        initial_contribution={"H": "10", "O1": "10", "O2": "10"},
        max_total_contribution={"H": "100", "O1": "100", "O2": "100"},
        rake_rate="0",
    )


def scenario_two() -> ThreePlayerRiverRakeScenario:
    root = RiverChanceNode(
        "root-chance",
        (
            RiverChanceOutcome("left", "1/2", observation("left"), hero_branch("left")),
            RiverChanceOutcome(
                "right", "1/2", observation("right"), hero_branch("right")
            ),
        ),
    )
    return ThreePlayerRiverRakeScenario(
        root=root,
        button_player_id="H",
        seat_order=("H", "O1", "O2"),
        river_action_order=("H", "O1", "O2"),
        initial_observation=None,
        initial_pot="30",
        initial_contribution={"H": "10", "O1": "10", "O2": "10"},
        max_total_contribution={"H": "100", "O1": "100", "O2": "100"},
        rake_rate="0",
    )


def hero_policy_one(check: str = "1", bet: str = "0") -> ExactBehaviorPolicy:
    return ExactBehaviorPolicy({"H_one": {"check": check, "bet": bet}})


def hero_policy_two() -> ExactBehaviorPolicy:
    return ExactBehaviorPolicy(
        {
            "H_left": {"check": "1", "bet": "0"},
            "H_right": {"check": "1", "bet": "0"},
        }
    )


def initial_profile(suffixes: tuple[str, ...] = ("one",)) -> OpponentInitialProfile:
    return OpponentInitialProfile(
        o1_probabilities={
            **{
                f"O1_after_check_{suffix}": {"check": "1"}
                for suffix in suffixes
            },
            **{
                f"O1_after_bet_{suffix}": {"fold": "1"}
                for suffix in suffixes
            },
        },
        o2_probabilities={
            **{
                f"O2_after_check_{suffix}": {"check": "1"}
                for suffix in suffixes
            },
            **{
                f"O2_after_bet_{suffix}": {"fold": "1"}
                for suffix in suffixes
            },
        },
    )


def attestation(scenario: ThreePlayerRiverRakeScenario):
    return create_perfect_recall_attestation(
        scenario,
        verifier="M32-T1 implementer fixture",
        verification_date="2026-07-24",
        evidence_version="m32-t1-implementer-fixture-v1",
        o1_confirmed=True,
        o2_confirmed=True,
    )


def run_one(
    *,
    scenario: ThreePlayerRiverRakeScenario | None = None,
    policy: ExactBehaviorPolicy | None = None,
    profile: OpponentInitialProfile | None = None,
    shifts: tuple[str, ...] = ("1/2", "1"),
    horizon: int = 3,
    generation: ThreePlayerCandidateGenerationConfig | None = None,
    repeated: ThreePlayerRepeatedConfig | None = None,
    **kwargs,
):
    candidate = scenario or scenario_one()
    return evaluate_three_player_candidate_repeated(
        candidate,
        policy or hero_policy_one(),
        profile or initial_profile(),
        attestation=attestation(candidate),
        generation=generation
        or ThreePlayerCandidateGenerationConfig(shift_amounts=shifts),
        repeated=repeated or ThreePlayerRepeatedConfig(horizon=horizon),
        **kwargs,
    )


def assert_success(result):
    assert result.status == EXACT_THREE_PLAYER_CANDIDATE_REPEATED_COMPLETE
    assert result.analysis is not None
    assert result.error is None
    assert result.partial_result is False
    return result.analysis


def assert_no_analysis(result, status: str):
    assert result.status == status
    assert result.analysis is None
    assert result.error is not None
    assert result.partial_result is False
    payload = result.to_dict()
    assert payload["analysis"] is None
    encoded = exact_three_player_candidate_repeated_json(result)
    assert '"analysis":null' in encoded
    assert '"support_cells"' not in encoded


@pytest.fixture(scope="module")
def completed():
    return run_one()


def _rational(value: Fraction | int | str) -> str:
    value = Fraction(value)
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _with_scalars(
    result: ScenarioResponseResult,
    *,
    initial_h: Fraction | int,
    worst: Fraction | int,
    best: Fraction | int,
) -> ScenarioResponseResult:
    scenario_evaluation = copy.deepcopy(result.scenario_evaluation)
    response = copy.deepcopy(result.response)
    initial_h = Fraction(initial_h)
    worst = Fraction(worst)
    best = Fraction(best)
    scenario_evaluation["initial_profile_comparison"]["utility"] = {
        "H": _rational(initial_h),
        "O1": _rational(-initial_h),
        "O2": "0",
        "R": "0",
    }
    response["hero_worst"] = _rational(worst)
    response["hero_best"] = _rational(best)
    response["utility_extrema"]["H"]["minimum"] = _rational(worst)
    response["utility_extrema"]["H"]["maximum"] = _rational(best)
    for witness in response["hero_worst_witnesses"]:
        witness["extremum_value"] = _rational(worst)
        witness["utility"]["H"] = _rational(worst)
    for witness in response["hero_best_witnesses"]:
        witness["extremum_value"] = _rational(best)
        witness["utility"]["H"] = _rational(best)
    response["utility_extrema"]["H"]["minimum_witnesses"] = response[
        "hero_worst_witnesses"
    ]
    response["utility_extrema"]["H"]["maximum_witnesses"] = response[
        "hero_best_witnesses"
    ]
    return ScenarioResponseResult(
        status=result.status,
        scenario_evaluation=scenario_evaluation,
        payoff_table=copy.deepcopy(result.payoff_table),
        response=response,
        error=None,
        partial_result=False,
    )


def _scalar_wrapper(monkeypatch, scalar_by_check):
    original = module._m31.evaluate_three_player_river_rake
    calls = []

    def wrapped(scenario, policy, **kwargs):
        result = original(scenario, policy, **kwargs)
        calls.append(result)
        check = policy.probabilities["H_one"]["check"]
        initial_h, worst, best = scalar_by_check[check]
        return _with_scalars(
            result, initial_h=initial_h, worst=worst, best=best
        )

    monkeypatch.setattr(module._m31, "evaluate_three_player_river_rake", wrapped)
    return calls


def test_a1_canonical_shift_grammar_duplicate_float_bool_and_noncanonical_rejected():
    for shifts in (
        ("2/4",),
        ("0",),
        ("-1/2",),
        ("2",),
        ("1/2", "1/2"),
        (0.5,),
        (True,),
    ):
        assert_no_analysis(run_one(shifts=shifts), INVALID_INPUT)


def test_a2_complete_baseline_hero_and_complete_o1_o2_profile_are_mandatory():
    incomplete_hero = ExactBehaviorPolicy({"H_one": {"check": "1"}})
    assert_no_analysis(run_one(policy=incomplete_hero), INVALID_INPUT)
    missing_o2 = OpponentInitialProfile(
        o1_probabilities=initial_profile().o1_probabilities,
        o2_probabilities=None,
    )
    assert_no_analysis(run_one(profile=missing_o2), INVALID_INPUT)


def test_a3_one_info_set_exact_source_target_action_order_and_l1(completed):
    analysis = assert_success(completed)
    assert len(analysis.candidates) == 2
    half, full = analysis.candidates
    assert half.candidate.edits[0].to_dict() == {
        "information_set_id": "H_one",
        "source_action_id": "check",
        "target_action_id": "bet",
        "shift_amount": "1/2",
    }
    assert half.candidate.policy.probabilities == {
        "H_one": {"check": "1/2", "bet": "1/2"}
    }
    assert half.candidate.l1_distance_exact == "1"
    assert full.candidate.policy.probabilities == {
        "H_one": {"check": "0", "bet": "1"}
    }
    assert full.candidate.l1_distance_exact == "2"


def test_a4_two_distinct_info_set_edits_are_complete_and_above_two_is_rejected():
    candidate = scenario_two()
    result = evaluate_three_player_candidate_repeated(
        candidate,
        hero_policy_two(),
        initial_profile(("left", "right")),
        attestation=attestation(candidate),
        generation=ThreePlayerCandidateGenerationConfig(
            shift_amounts=("1",), max_simultaneous_info_sets=2
        ),
        repeated=ThreePlayerRepeatedConfig(horizon=1),
    )
    analysis = assert_success(result)
    assert len(analysis.candidates) == 3
    two_edit = [item for item in analysis.candidates if len(item.candidate.edits) == 2]
    assert len(two_edit) == 1
    assert {edit.information_set_id for edit in two_edit[0].candidate.edits} == {
        "H_left",
        "H_right",
    }
    assert_no_analysis(
        run_one(
            shifts=("1",),
            generation=ThreePlayerCandidateGenerationConfig(
                shift_amounts=("1",), max_simultaneous_info_sets=3
            ),
        ),
        INVALID_INPUT,
    )


def test_a5_a19_candidate_count_cap_precedes_policy_materialisation(monkeypatch):
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("candidate policy materialised")

    monkeypatch.setattr(module, "_materialize_candidate_policy", forbidden)
    result = run_one(
        limits=replace(
            ThreePlayerCandidateRepeatedLimits(), max_generated_candidates=1
        )
    )
    assert_no_analysis(result, CAP_EXCEEDED)
    assert called is False


def test_a6_candidate_and_universe_identity_ignore_shift_input_permutation():
    first = assert_success(run_one(shifts=("1", "1/2")))
    second = assert_success(run_one(shifts=("1/2", "1")))
    assert first.payload["identities"]["candidate_universe"] == second.payload[
        "identities"
    ]["candidate_universe"]
    assert [
        item.candidate.candidate_id for item in first.candidates
    ] == [item.candidate.candidate_id for item in second.candidates]


def test_a7_semantic_action_order_and_baseline_policy_change_identity():
    normal = assert_success(run_one(shifts=("1/2",)))
    reversed_scenario = scenario_one(reverse_actions=True)
    reversed_result = assert_success(
        run_one(scenario=reversed_scenario, shifts=("1/2",))
    )
    assert normal.payload["identities"]["candidate_universe"] != (
        reversed_result.payload["identities"]["candidate_universe"]
    )
    changed = assert_success(
        run_one(policy=hero_policy_one("1/2", "1/2"), shifts=("1/2",))
    )
    assert normal.payload["identities"]["candidate_universe"] != changed.payload[
        "identities"
    ]["candidate_universe"]


def test_a8_baseline_exact_success_and_m32_usable_are_required(monkeypatch):
    original = module._m31.evaluate_three_player_river_rake

    def unusable(*args, **kwargs):
        result = original(*args, **kwargs)
        scenario_evaluation = copy.deepcopy(result.scenario_evaluation)
        scenario_evaluation["m32_handoff"]["m32_usable"] = False
        return replace(result, scenario_evaluation=scenario_evaluation)

    monkeypatch.setattr(module._m31, "evaluate_three_player_river_rake", unusable)
    assert_no_analysis(run_one(shifts=()), BASELINE_RESPONSE_FAILURE)


def test_a9_every_candidate_gets_a_fresh_public_m31_call(monkeypatch):
    original = module._m31.evaluate_three_player_river_rake
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(module._m31, "evaluate_three_player_river_rake", counted)
    analysis = assert_success(run_one())
    assert len(calls) == 1 + len(analysis.candidates) == 3
    assert len({json.dumps(call.probabilities, sort_keys=True) for call in calls}) == 3


def test_a10_a13_safety_uses_only_hero_worst_and_crossing_winner(monkeypatch):
    _scalar_wrapper(
        monkeypatch,
        {
            "1": (5, 5, 5),
            "1/2": (Fraction(25, 2), Fraction(-15, 2), 10_000),
            "0": (20, -20, 20_000),
        },
    )
    analysis = assert_success(run_one())
    rows = analysis.selector_report.rows
    half_id, full_id = [item.candidate.candidate_id for item in analysis.candidates]
    assert [row.best_total_hero_ev_delta for row in rows] == [
        -37.5,
        -17.5,
        5.0,
        45.0,
    ]
    assert [row.selected_candidate_id for row in rows] == [
        None,
        None,
        full_id,
        full_id,
    ]
    assert rows[0].primary_tie_candidate_ids == (half_id,)
    assert all(
        item.exact_values["response"]["safety_scalar_source_path"]
        == "m31_scenario_response.response.hero_worst"
        for item in analysis.candidates
    )


def test_a11_exact_sources_and_binary64_projections_are_separate(completed):
    analysis = assert_success(completed)
    candidate = analysis.candidates[0]
    assert candidate.exact_values["response"]["hero_worst"] == (
        candidate.scenario_response.response["hero_worst"]
    )
    projection = candidate.scalar_projections["hero_worst"]
    assert projection.source_exact == candidate.exact_values["response"]["hero_worst"]
    assert isinstance(projection.binary64, float)
    assert "hero_worst_minus_baseline_H" in candidate.exact_values


def test_a12_m1_is_all_post_and_m_n_plus_one_is_all_pre(monkeypatch):
    _scalar_wrapper(
        monkeypatch,
        {
            "1": (1, 1, 1),
            "0": (3, 0, 0),
        },
    )
    analysis = assert_success(run_one(shifts=("1",), horizon=3))
    rows = analysis.selector_report.rows
    assert rows[0].best_total_hero_ev_delta == -3.0
    assert rows[-1].best_total_hero_ev_delta == 6.0


def test_a14_nonpositive_and_strict_threshold_boundary_are_no_benefit(monkeypatch):
    _scalar_wrapper(
        monkeypatch,
        {
            "1": (0, 0, 0),
            "0": (Fraction(5, 4), Fraction(5, 4), Fraction(5, 4)),
        },
    )
    analysis = assert_success(
        run_one(
            shifts=("1",),
            horizon=1,
            repeated=ThreePlayerRepeatedConfig(horizon=1, tolerance=0.25),
            selector_configuration=AutomaticCommitmentSelectionConfig(
                minimum_total_uplift=1.0
            ),
        )
    )
    assert analysis.selector_report.rows[0].status == NO_BENEFICIAL_COMMITMENT
    assert analysis.selector_report.rows[0].selected_candidate_id is None


def test_a15_full_ties_secondary_display_and_no_benefit_evidence(monkeypatch):
    _scalar_wrapper(
        monkeypatch,
        {
            "1": (0, 0, 0),
            "1/2": (0, 0, 1000),
            "0": (0, 0, 2000),
        },
    )
    analysis = assert_success(run_one(horizon=1))
    row = analysis.selector_report.rows[0]
    ids = tuple(sorted(item.candidate.candidate_id for item in analysis.candidates))
    assert row.status == NO_BENEFICIAL_COMMITMENT
    assert row.primary_tie_candidate_ids == ids
    assert tuple(item.candidate_id for item in row.tie_break_evidence) == ids
    assert row.primary_tie_display_candidate_id in ids


def test_a16_a17_full_m31_response_multiplicity_and_diagnostics_are_lossless(completed):
    analysis = assert_success(completed)
    payload = completed.to_dict()["analysis"]
    for runtime, serialized in zip(analysis.candidates, payload["candidates"]):
        response = runtime.scenario_response.response
        nested = serialized["m31_scenario_response"]["response"]
        assert nested == response
        assert nested["support_cells"] == response["support_cells"]
        assert nested["support_pair_audit"] == response["support_pair_audit"]
        assert nested["hero_worst_witnesses"] == response["hero_worst_witnesses"]
        assert nested["utility_extrema"]["O1"] == response["utility_extrema"]["O1"]
        assert nested["utility_extrema"]["O2"] == response["utility_extrema"]["O2"]
        assert nested["utility_extrema"]["R"] == response["utility_extrema"]["R"]
        assert nested["pure_profile_unilateral_stability"] == response[
            "pure_profile_unilateral_stability"
        ]
        assert nested["hero_min_joint_plan_stress"] == response[
            "hero_min_joint_plan_stress"
        ]


def test_a18_o1_plus_o2_transport_is_explicitly_not_coalition(completed):
    payload = assert_success(completed).to_dict()
    assert payload["response_semantics"] == module.RESPONSE_SEMANTICS
    assert payload["selector_transport_opponent_value"] == (
        "o1_plus_o2_accounting_only_not_a_coalition"
    )
    assert payload["claim_boundary"]["coalition_or_transferable_utility_claim"] is False
    assert (
        payload["selector"]["legacy_two_player_label_is_not_m32_semantics"]
        is True
    )
    assert "villain_exact_best_response_hero_worst" not in json.dumps(
        payload, sort_keys=True
    )


def test_a20_aggregate_cap_precedes_first_candidate_m31_call(monkeypatch):
    original = module._m31.evaluate_three_player_river_rake
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module._m31, "evaluate_three_player_river_rake", counted)
    result = run_one(
        limits=replace(
            ThreePlayerCandidateRepeatedLimits(), max_total_m31_runs=1
        )
    )
    assert_no_analysis(result, CAP_EXCEEDED)
    assert calls == 1


def test_a21_timing_cap_precedes_m27_row_materialisation(monkeypatch):
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("M27 timing rows reached")

    monkeypatch.setattr(
        module._m27, "select_automatic_commitment_values", forbidden
    )
    result = run_one(
        limits=replace(
            ThreePlayerCandidateRepeatedLimits(),
            max_timing_row_evaluations=1,
        )
    )
    assert_no_analysis(result, CAP_EXCEEDED)
    assert called is False


def test_a22_output_record_and_byte_caps_return_null_analysis():
    for limits in (
        replace(
            ThreePlayerCandidateRepeatedLimits(),
            max_outer_output_records=1,
        ),
        replace(
            ThreePlayerCandidateRepeatedLimits(),
            max_outer_output_bytes=1,
        ),
    ):
        assert_no_analysis(run_one(limits=limits), CAP_EXCEEDED)


def test_a23_middle_candidate_failure_discards_successful_prefix(monkeypatch):
    original = module._m31.evaluate_three_player_river_rake
    calls = 0

    def fail_third(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            return ScenarioResponseResult(
                status="M30_RESPONSE_FAILURE",
                scenario_evaluation=None,
                payoff_table=None,
                response=None,
                error=ScenarioResponseError(
                    phase="fault", message="middle candidate fault"
                ),
                partial_result=False,
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(
        module._m31, "evaluate_three_player_river_rake", fail_third
    )
    result = run_one()
    assert_no_analysis(result, CANDIDATE_RESPONSE_FAILURE)
    assert result.error.candidate_id is not None
    assert calls == 3


def test_a24_no_truncation_sampling_fallback_skip_clamp_or_partial_contract():
    doc = inspect.getdoc(module) or ""
    assert "no truncation, sampling, fallback" in doc.lower()
    source = SOURCE.read_text(encoding="utf-8")
    assert "current_cfr_used" in source
    result = run_one(
        limits=replace(
            ThreePlayerCandidateRepeatedLimits(), max_generated_candidates=1
        )
    )
    assert result.analysis is None and result.partial_result is False


def test_a25_unsafe_binary64_fails_only_after_all_exact_responses(monkeypatch):
    with pytest.raises(module._M32Failure) as overflow:
        module._project_fraction(Fraction(10**400), "test.overflow")
    assert overflow.value.status == SCALAR_PROJECTION_FAILURE
    with pytest.raises(module._M32Failure) as underflow:
        module._project_fraction(Fraction(1, 10**400), "test.underflow")
    assert underflow.value.status == SCALAR_PROJECTION_FAILURE
    original_evaluator = module._m31.evaluate_three_player_river_rake
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_evaluator(*args, **kwargs)

    def unsafe(*args, **kwargs):
        raise module._M32Failure(
            SCALAR_PROJECTION_FAILURE, "fault", "unsafe projection"
        )

    monkeypatch.setattr(module._m31, "evaluate_three_player_river_rake", counted)
    monkeypatch.setattr(module, "_project_fraction", unsafe)
    assert_no_analysis(run_one(), SCALAR_PROJECTION_FAILURE)
    assert calls == 3


def test_a26_stale_source_universe_response_and_run_pins_are_rejected(completed):
    analysis = assert_success(completed)
    payload = analysis.to_dict()
    stale = "0" * 64
    for pins in (
        ThreePlayerCandidateRepeatedPins(scenario_identity=stale),
        ThreePlayerCandidateRepeatedPins(candidate_universe_identity=stale),
        ThreePlayerCandidateRepeatedPins(
            candidate_response_run_identities=((analysis.candidates[0].candidate.candidate_id, stale),)
        ),
        ThreePlayerCandidateRepeatedPins(m32_run_identity=stale),
    ):
        assert_no_analysis(run_one(pins=pins), STALE_INPUT)
    assert payload["identities"]["m32_run"] != stale


def test_a27_empty_candidate_universe_has_n_plus_one_no_benefit_rows():
    analysis = assert_success(run_one(shifts=(), horizon=3))
    assert analysis.candidates == ()
    assert len(analysis.selector_report.rows) == 4
    assert analysis.selector_report.timing_row_evaluation_count == 0
    assert all(
        row.status == NO_BENEFICIAL_COMMITMENT
        and row.selected_candidate_id is None
        and row.primary_tie_candidate_ids == ()
        for row in analysis.selector_report.rows
    )


@pytest.mark.parametrize(
    "generation",
    [
        ThreePlayerCandidateGenerationConfig(search_mode="hybrid"),
        ThreePlayerCandidateGenerationConfig(search_mode="baseline_targeted"),
        ThreePlayerCandidateGenerationConfig(adaptation_mode="individual_o1_o2"),
    ],
)
def test_a28_other_search_and_individual_timing_modes_are_unsupported(generation):
    assert_no_analysis(
        evaluate_three_player_candidate_repeated(
            scenario_one(),
            hero_policy_one(),
            initial_profile(),
            attestation=attestation(scenario_one()),
            generation=generation,
        ),
        UNSUPPORTED_MODE,
    )


def test_a29_same_input_twice_has_identical_canonical_json_bytes():
    first = exact_three_player_candidate_repeated_json(run_one())
    second = exact_three_player_candidate_repeated_json(run_one())
    assert first == second
    assert "\n" not in first and "\r" not in first
    assert "NaN" not in first and "Infinity" not in first


def test_a30_all_m27_rows_match_implementer_owned_formula_oracle(monkeypatch):
    _scalar_wrapper(
        monkeypatch,
        {
            "1": (5, 5, 5),
            "1/2": (Fraction(25, 2), Fraction(-15, 2), Fraction(25, 2)),
            "0": (20, -20, 20),
        },
    )
    analysis = assert_success(run_one())
    b = Fraction(5)
    expected = []
    candidates = [
        (Fraction(25, 2), Fraction(-15, 2)),
        (Fraction(20), Fraction(-20)),
    ]
    for opportunity in range(1, 5):
        deltas = [
            (opportunity - 1) * pre
            + (4 - opportunity) * post
            - 3 * b
            for pre, post in candidates
        ]
        expected.append(float(max(deltas)))
    assert [
        row.best_total_hero_ev_delta for row in analysis.selector_report.rows
    ] == expected
    assert expected == [-37.5, -17.5, 5.0, 45.0]


def test_a31_focused_regression_public_modules_remain_external_consumers():
    source = SOURCE.read_text(encoding="utf-8")
    assert "select_automatic_commitment_values" in source
    assert "evaluate_three_player_river_rake" in source
    assert "three_player_cfr" not in source
    assert all(path.exists() for path in PROTECTED)
    package_init = (ROOT / "src" / "repeated_poker" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert "three_player_candidate_repeated" not in package_init


def test_a32_a33_exact_two_file_scope_protected_diff_and_no_repo_cache():
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={ROOT}",
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    changed = {
        line[3:].replace("\\", "/")
        for line in completed.stdout.splitlines()
        if line.strip()
    }
    assert changed.issubset(
        {
            "src/repeated_poker/three_player_candidate_repeated.py",
            "tests/test_three_player_candidate_repeated.py",
        }
    )
    assert all(path.exists() for path in PROTECTED)
    tracked = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={ROOT}",
            "ls-files",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    tracked_cache_paths = [
        value
        for value in tracked.stdout.splitlines()
        if "/__pycache__/" in f"/{value}"
        or "/.pytest_cache/" in f"/{value}"
        or value.endswith((".pyc", ".pyo"))
    ]
    assert tracked_cache_paths == []
