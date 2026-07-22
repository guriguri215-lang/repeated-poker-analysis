"""Focused contract tests for the exact real-card M28 repeated bridge."""

import json
from dataclasses import replace

import pytest

import repeated_poker.aiof_preflop_candidate_repeated as bridge_module
from repeated_poker.aiof_cards import (
    AiofStatus,
    RangeEntry,
    RangeSpec,
    WeightBasis,
    card_from_id,
    card_id,
)
from repeated_poker.aiof_chip_ev import (
    ComboActionProbability,
    HeadsUpChipEvGame,
    PushFoldRunResult,
    SuppliedProfile,
)
from repeated_poker.aiof_equity import EquityAlgorithm
from repeated_poker.aiof_preflop_candidate_repeated import (
    AiofPreflopCandidateRepeatedLimits,
    AiofPreflopCandidateRepeatedRequest,
    analyze_aiof_preflop_candidate_repeated,
)
from repeated_poker.automatic_commitment_selection import NO_BENEFICIAL_COMMITMENT


WIN_BOARD = ("2c", "3d", "4h", "5s", "7c")


def exact_range(combo):
    return RangeSpec((RangeEntry(combo, 1.0, WeightBasis.EXACT_COMBO_MASS),))


def dead_except(live_cards):
    live = {card_id(card) for card in live_cards}
    return tuple(card_from_id(value) for value in range(52) if value not in live)


def profile(sb_rows, bb_rows):
    return SuppliedProfile(
        tuple(ComboActionProbability(*row) for row in sb_rows),
        tuple(ComboActionProbability(*row) for row in bb_rows),
    )


def game(**overrides):
    values = {
        "starting_stack_sb": 10.0,
        "starting_stack_bb": 10.0,
        "small_blind": 0.5,
        "big_blind": 1.0,
        "ante": 0.0,
    }
    values.update(overrides)
    return HeadsUpChipEvGame(**values)


def one_pair_request(
    *,
    hero_seat="sb",
    shove=0.0,
    call=0.0,
    shift_amounts=(1.0,),
    max_shifted_combos=1,
    horizon=2,
    game_value=None,
    bridge_limits=None,
):
    sb_combo = "AsAh"
    bb_combo = "KsKh"
    live = ("As", "Ah", "Ks", "Kh") + WIN_BOARD
    return AiofPreflopCandidateRepeatedRequest(
        game=game_value or game(),
        sb_range=exact_range(sb_combo),
        bb_range=exact_range(bb_combo),
        dead_cards=dead_except(live),
        baseline_profile=profile(((sb_combo, shove),), ((bb_combo, call),)),
        hero_seat=hero_seat,
        shift_amounts=shift_amounts,
        max_shifted_combos=max_shifted_combos,
        horizon=horizon,
        bridge_limits=bridge_limits or AiofPreflopCandidateRepeatedLimits(),
    )


def mixed_request(*, reverse=False, max_shifted_combos=2):
    sb_entries = (
        RangeEntry("AA", 0.7, WeightBasis.CLASS_TOTAL_MASS),
        RangeEntry("KcKd", 0.3, WeightBasis.EXACT_COMBO_MASS),
    )
    live = ("As", "Ah", "Kc", "Kd", "Qs", "Qh") + WIN_BOARD
    sb_rows = (("AsAh", 0.5), ("KcKd", 0.5))
    shift_amounts = (0.25, 0.5)
    dead = dead_except(live)
    if reverse:
        sb_entries = tuple(reversed(sb_entries))
        sb_rows = tuple(reversed(sb_rows))
        shift_amounts = tuple(reversed(shift_amounts))
        dead = tuple(reversed(dead))
    return AiofPreflopCandidateRepeatedRequest(
        game=game(),
        sb_range=RangeSpec(sb_entries),
        bb_range=exact_range("QsQh"),
        dead_cards=dead,
        baseline_profile=profile(sb_rows, (("QsQh", 0.25),)),
        hero_seat="sb",
        shift_amounts=shift_amounts,
        max_shifted_combos=max_shifted_combos,
        horizon=2,
    )


def successful(request):
    result = analyze_aiof_preflop_candidate_repeated(request)
    assert result.status is AiofStatus.SUCCESS, result.error_message
    assert result.payload is not None and result.error_message is None
    return result.payload


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def test_sb_known_board_oracle_composes_b_a_l_and_repeated_selection():
    payload = successful(one_pair_request())
    assert payload.hero_active_action == "shove"
    assert payload.opponent_active_action == "call"
    assert payload.workload.candidate_count == 1
    assert payload.workload.exact_board_evaluations_per_analysis == 1
    assert payload.baseline.baseline_hero_ev == pytest.approx(-0.5)
    candidate = payload.candidates[0]
    assert candidate.fixed_opponent_hero_ev == pytest.approx(1.0)
    assert candidate.post_response_hero_ev_worst == pytest.approx(1.0)
    assert candidate.post_response_hero_ev_best == pytest.approx(1.0)
    assert candidate.post_response_hero_ev_worst_diff == pytest.approx(1.5)
    assert candidate.l1_distance == pytest.approx(2.0)
    assert candidate.shifts[0].source_action == "fold"
    assert candidate.shifts[0].target_action == "shove"
    assert all(row.selected_candidate_id == candidate.candidate_id for row in payload.automatic_selection.rows)


def test_bb_known_board_oracle_uses_call_as_hero_active_action():
    payload = successful(one_pair_request(hero_seat="bb", shove=1.0, call=0.0))
    assert payload.hero_active_action == "call"
    assert payload.opponent_active_action == "shove"
    assert payload.baseline.baseline_hero_ev == pytest.approx(-1.0)
    candidate = payload.candidates[0]
    assert candidate.fixed_opponent_hero_ev == pytest.approx(-10.0)
    assert candidate.post_response_hero_ev_worst == pytest.approx(-10.0)
    assert candidate.shifts[0].target_action == "call"
    assert all(row.status == NO_BENEFICIAL_COMMITMENT for row in payload.automatic_selection.rows)


def test_native_response_tie_is_retained_factorized_without_pure_product():
    request = one_pair_request(
        game_value=game(starting_stack_bb=1.0), shove=0.0, call=0.0
    )
    payload = successful(request)
    response = payload.candidates[0].opponent_response
    assert response.rows[0].best_actions == ("call", "fold")
    assert payload.candidates[0].post_response_hero_ev_worst == pytest.approx(1.0)
    serialized = payload.to_dict()
    assert "best_response_strategies" not in canonical_bytes(serialized)
    assert len(serialized["candidates"][0]["factorized_opponent_response"]["rows"]) == 1


def test_mixed_class_exact_ranges_nonuniform_profile_candidate_formula_and_l1():
    payload = successful(mixed_request())
    assert tuple(item.combo for item in payload.prepared_ranges.sb_marginals) == (
        "KdKc",
        "AsAh",
    )
    # Each of two Hero combos has four feasible directed shifts: +/- .25, +/- .5.
    # S=8 and sum(n_i^2)=32, so C=8+(64-32)/2=24.
    assert payload.workload.feasible_single_shift_count == 8
    assert payload.workload.candidate_count == 24
    assert len(payload.candidates) == 24
    assert len({candidate.candidate_id for candidate in payload.candidates}) == 24
    assert payload.coverage.generated_candidate_ids == payload.coverage.kept_candidate_ids
    assert payload.coverage.filtering_applied is False
    for candidate in payload.candidates:
        assert len(candidate.shifts) in (1, 2)
        assert len({shift.combo for shift in candidate.shifts}) == len(candidate.shifts)
        assert candidate.l1_distance == pytest.approx(
            2.0 * sum(shift.shift_amount for shift in candidate.shifts)
        )
        assert len(candidate.profile.sb_shove) == 2
        assert len(candidate.profile.bb_call) == 1


def test_order_permutations_are_byte_identical():
    forward = successful(mixed_request()).to_dict()
    reverse = successful(mixed_request(reverse=True)).to_dict()
    assert canonical_bytes(forward) == canonical_bytes(reverse)


def test_single_shift_projection_excludes_two_combo_candidates():
    payload = successful(mixed_request(max_shifted_combos=1))
    assert payload.workload.feasible_single_shift_count == 8
    assert payload.workload.candidate_count == 8
    assert all(len(candidate.shifts) == 1 for candidate in payload.candidates)


def test_empty_candidate_universe_is_success_with_all_n_plus_one_no_beneficial_rows():
    payload = successful(
        one_pair_request(shove=0.5, shift_amounts=(1.0,), horizon=3)
    )
    assert payload.workload.candidate_count == 0
    assert payload.candidates == ()
    assert len(payload.automatic_selection.rows) == 4
    assert payload.automatic_selection.timing_row_evaluation_count == 0
    assert all(row.status == NO_BENEFICIAL_COMMITMENT for row in payload.automatic_selection.rows)


@pytest.mark.parametrize(
    "limits",
    [
        AiofPreflopCandidateRepeatedLimits(max_candidates=1),
        AiofPreflopCandidateRepeatedLimits(max_total_board_evaluations=2),
        AiofPreflopCandidateRepeatedLimits(max_response_rows=2),
        AiofPreflopCandidateRepeatedLimits(max_timing_rows=3),
    ],
)
def test_every_bridge_cap_fails_before_candidate_materialization_or_analysis(
    limits, monkeypatch
):
    calls = {"materialize": 0, "analyze": 0}

    def forbidden_materialize(*args, **kwargs):
        calls["materialize"] += 1
        raise AssertionError("candidate materialization must not run")

    def forbidden_analyze(*args, **kwargs):
        calls["analyze"] += 1
        raise AssertionError("analysis must not run")

    monkeypatch.setattr(bridge_module, "_materialize_candidates", forbidden_materialize)
    monkeypatch.setattr(bridge_module, "analyze_pushfold", forbidden_analyze)
    result = analyze_aiof_preflop_candidate_repeated(
        one_pair_request(
            shove=0.5,
            shift_amounts=(0.25,),
            horizon=1,
            bridge_limits=limits,
        )
    )
    assert result.status is AiofStatus.CAP_EXCEEDED
    assert result.payload is None and result.error_message
    assert calls == {"materialize": 0, "analyze": 0}


def test_candidate_analysis_mid_failure_discards_all_partial_payload(monkeypatch):
    original = bridge_module.analyze_pushfold
    calls = {"count": 0}

    def fail_second(request):
        calls["count"] += 1
        if calls["count"] == 2:
            return PushFoldRunResult(AiofStatus.NUMERIC_FAILURE, None, "controlled")
        return original(request)

    monkeypatch.setattr(bridge_module, "analyze_pushfold", fail_second)
    result = analyze_aiof_preflop_candidate_repeated(
        one_pair_request(shove=0.5, shift_amounts=(0.25,))
    )
    assert calls["count"] == 2
    assert result.status is AiofStatus.NUMERIC_FAILURE
    assert result.payload is None and result.error_message == "controlled"


@pytest.mark.parametrize(
    "changes",
    [
        {"algorithm": EquityAlgorithm.DETERMINISTIC_MONTE_CARLO},
        {"seed": 7},
        {"samples": 10},
    ],
)
def test_monte_carlo_and_sampling_controls_are_unsupported_without_fallback(changes):
    result = analyze_aiof_preflop_candidate_repeated(
        replace(one_pair_request(), **changes)
    )
    assert result.status is AiofStatus.UNSUPPORTED_MODEL
    assert result.payload is None and result.error_message


@pytest.mark.parametrize(
    ("bridge_request", "status"),
    [
        (replace(one_pair_request(), hero_seat="button"), AiofStatus.INVALID_INPUT),
        (replace(one_pair_request(), shift_amounts=(0.0,)), AiofStatus.INVALID_INPUT),
        (replace(one_pair_request(), max_shifted_combos=3), AiofStatus.INVALID_INPUT),
        (
            replace(
                one_pair_request(),
                baseline_profile=SuppliedProfile(
                    (), (ComboActionProbability("KsKh", 0.0),)
                ),
            ),
            AiofStatus.INVALID_STRATEGY,
        ),
        (
            replace(one_pair_request(), game=game(fee=0.01)),
            AiofStatus.UNSUPPORTED_MODEL,
        ),
    ],
)
def test_invalid_inputs_and_unsupported_accounting_fail_closed(
    bridge_request, status
):
    result = analyze_aiof_preflop_candidate_repeated(bridge_request)
    assert result.status is status
    assert result.payload is None and result.error_message


def test_expected_baseline_identity_and_identity_field_bindings():
    request = one_pair_request(shove=0.5, shift_amounts=(0.25,))
    first = successful(request)
    pinned = successful(
        replace(request, expected_baseline_identity=first.baseline_identity)
    )
    assert pinned.baseline_identity == first.baseline_identity
    mismatch = analyze_aiof_preflop_candidate_repeated(
        replace(request, expected_baseline_identity="sha256:" + "0" * 64)
    )
    assert mismatch.status is AiofStatus.INVALID_INPUT
    assert mismatch.payload is None

    changed_profile = successful(
        replace(
            request,
            baseline_profile=profile((("AsAh", 0.25),), (("KsKh", 0.0),)),
        )
    )
    changed_hero = successful(replace(request, hero_seat="bb"))
    assert changed_profile.baseline_identity != first.baseline_identity
    assert changed_hero.baseline_identity != first.baseline_identity
    changed_horizon = successful(replace(request, horizon=3))
    assert changed_horizon.baseline_identity == first.baseline_identity
    assert changed_horizon.analysis_identity != first.analysis_identity
    assert tuple(c.candidate_id for c in changed_horizon.candidates) == tuple(
        c.candidate_id for c in first.candidates
    )


def test_candidate_identity_collision_fails_before_any_analysis(monkeypatch):
    calls = {"count": 0}

    def collision(*args, **kwargs):
        return "sha256:" + "f" * 64

    def forbidden(*args, **kwargs):
        calls["count"] += 1
        raise AssertionError("analysis must not run after identity collision")

    monkeypatch.setattr(bridge_module, "_candidate_identity", collision)
    monkeypatch.setattr(bridge_module, "analyze_pushfold", forbidden)
    result = analyze_aiof_preflop_candidate_repeated(
        one_pair_request(shove=0.5, shift_amounts=(0.25,))
    )
    assert result.status is AiofStatus.INVALID_INPUT
    assert result.payload is None and calls["count"] == 0


def test_success_output_is_deterministic_and_has_no_partial_or_nonfinite_values():
    request = one_pair_request(shove=0.5, call=0.4, shift_amounts=(0.25,))
    first = analyze_aiof_preflop_candidate_repeated(request).to_dict()
    second = analyze_aiof_preflop_candidate_repeated(request).to_dict()
    assert canonical_bytes(first) == canonical_bytes(second)
    assert "NaN" not in canonical_bytes(first)
    assert first["status"] == AiofStatus.SUCCESS.value
    assert first["error"] is None and first["payload"] is not None
